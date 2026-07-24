from __future__ import annotations

"""SAM v1 ViT-B wrapper for point-prompted pseudo-mask generation."""

from pathlib import Path
from typing import Any

import numpy as np

# Structural type: TumorComponent fields are consumed by predict_from_components.
Component = Any


class SAMPredictor:
    """Thin fail-closed wrapper around the official SAM v1 ViT-B model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM ViT-B checkpoint not found at {checkpoint_path}. "
                "Provide the official sam_vit_b_01ec64.pth file explicitly."
            )

        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        self._predictor = SamPredictor(sam)
        self.last_prompt_stats: dict[str, int] = {}

    def predict_from_points(
        self,
        image_rgb: np.ndarray,
        point_prompts: list[tuple[int, int]],
        debug_dir: str | Path | None = None,
        image_pil=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not point_prompts:
            h, w = image_rgb.shape[:2]
            self.last_prompt_stats = self._empty_prompt_stats()
            return np.zeros((0, h, w), dtype=bool), np.zeros(0, dtype=np.float32)

        self._predictor.set_image(image_rgb)

        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for r, c in point_prompts:
            # SAM expects (x, y) == (col, row)
            point_coords = np.array([[c, r]], dtype=np.float32)
            point_labels = np.ones(1, dtype=np.int32)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            all_masks.append(masks)    # [3, H, W]
            all_scores.append(scores)  # [3]

        combined_masks = np.concatenate(all_masks, axis=0)
        combined_scores = np.concatenate(all_scores, axis=0)
        unique_points = len(set(point_prompts))
        self.last_prompt_stats = {
            "sam_prompt_calls": len(point_prompts),
            "unique_positive_prompt_points": unique_points,
            "unique_negative_prompt_points": 0,
            "unique_prompt_points": unique_points,
            "box_prompt_calls": 0,
        }

        if debug_dir is not None:
            self._save_debug(debug_dir, image_rgb, image_pil, combined_masks, combined_scores)

        return combined_masks, combined_scores

    def predict_from_components(
        self,
        image_rgb: np.ndarray,
        components: list[Component],
        prompt_mode: str = "box_point",
        multimask_output: bool = True,
        negative_points_per_component: int = 0,
        prompt_border_margin: int = 2,
        max_box_area_ratio: float | None = None,
        debug_dir: str | Path | None = None,
        image_pil=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_modes = {"point", "joint_points", "box", "box_point"}
        if prompt_mode not in valid_modes:
            raise ValueError(f"Unknown prompt_mode '{prompt_mode}'. Choose from {sorted(valid_modes)}.")
        if not components:
            h, w = image_rgb.shape[:2]
            self.last_prompt_stats = self._empty_prompt_stats()
            return (
                np.zeros((0, h, w), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        self._predictor.set_image(image_rgb)
        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        component_ids: list[int] = []
        used_positive_points: set[tuple[int, int]] = set()
        used_negative_points: set[tuple[int, int]] = set()
        sam_prompt_calls = 0
        box_prompt_calls = 0

        h, w = image_rgb.shape[:2]

        for component in components:
            original_points = list(component.positive_points)
            points = self._filter_border_points(
                original_points,
                image_shape=(h, w),
                margin=prompt_border_margin,
            )
            if prompt_mode == "point":
                points = points[:1]

            point_coords = None
            point_labels = None
            # Pure box mode must pass both point arrays as None. Configured
            # negative points are intentionally ignored in this mode.
            if prompt_mode != "box" and points:
                point_coords = np.asarray([(col, row) for row, col in points], dtype=np.float32)
                point_labels = np.ones(len(points), dtype=np.int32)
                used_positive_points.update(points)
                if negative_points_per_component > 0:
                    # Prefer negative points precomputed from low-CAM pixels
                    # inside the component (see TumorComponent.negative_points)
                    # over the bbox-corner heuristic below, which can still
                    # land close to the lesion when the support region is wide.
                    precomputed_negatives = getattr(component, "negative_points", ())
                    if precomputed_negatives:
                        negative_points = list(precomputed_negatives[:negative_points_per_component])
                    else:
                        negative_points = self._sample_negative_points(
                            component,
                            count=negative_points_per_component,
                        )
                    if negative_points:
                        used_negative_points.update(negative_points)
                        negative_coords = np.asarray(
                            [(col, row) for row, col in negative_points],
                            dtype=np.float32,
                        )
                        point_coords = np.concatenate([point_coords, negative_coords], axis=0)
                        point_labels = np.concatenate(
                            [point_labels, np.zeros(len(negative_points), dtype=np.int32)],
                            axis=0,
                        )

            box = None
            if prompt_mode in {"box", "box_point"}:
                use_box = True
                if max_box_area_ratio is not None:
                    x0, y0, x1, y1 = component.bbox
                    box_area = max(1, x1 - x0 + 1) * max(1, y1 - y0 + 1)
                    use_box = (box_area / float(h * w)) <= max_box_area_ratio
                if use_box:
                    box = np.asarray(component.bbox, dtype=np.float32)

            # Reject invalid prompt combinations before calling SAM.  In
            # particular, a box-only candidate whose oversized box was
            # dropped must not silently become a promptless predictor call.
            if prompt_mode == "box" and box is None:
                continue
            if prompt_mode in {"point", "joint_points", "box_point"} and point_coords is None and box is None:
                continue
            if (point_coords is None) != (point_labels is None):
                raise RuntimeError("SAM point_coords and point_labels must both be set or both be None")

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
            all_masks.append(masks)
            all_scores.append(scores)
            component_ids.extend([component.component_id] * masks.shape[0])
            sam_prompt_calls += 1
            box_prompt_calls += int(box is not None)

        if not all_masks:
            self.last_prompt_stats = self._empty_prompt_stats()
            return (
                np.zeros((0, h, w), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )
        combined_masks = np.concatenate(all_masks, axis=0)
        combined_scores = np.concatenate(all_scores, axis=0)
        component_id_array = np.asarray(component_ids, dtype=np.int32)
        self.last_prompt_stats = {
            "sam_prompt_calls": sam_prompt_calls,
            "unique_positive_prompt_points": len(used_positive_points),
            "unique_negative_prompt_points": len(used_negative_points),
            "unique_prompt_points": len(used_positive_points | used_negative_points),
            "box_prompt_calls": box_prompt_calls,
        }
        if debug_dir is not None:
            self._save_debug(
                debug_dir,
                image_rgb,
                image_pil,
                combined_masks,
                combined_scores,
                component_ids=component_id_array,
            )
        return combined_masks, combined_scores, component_id_array

    @staticmethod
    def _empty_prompt_stats() -> dict[str, int]:
        return {
            "sam_prompt_calls": 0,
            "unique_positive_prompt_points": 0,
            "unique_negative_prompt_points": 0,
            "unique_prompt_points": 0,
            "box_prompt_calls": 0,
        }

    @staticmethod
    def _filter_border_points(
        points: list[tuple[int, int]],
        image_shape: tuple[int, int],
        margin: int,
    ) -> list[tuple[int, int]]:
        if margin <= 0:
            return points
        h, w = image_shape
        return [
            (row, col)
            for row, col in points
            if margin <= row < h - margin and margin <= col < w - margin
        ]

    @staticmethod
    def _sample_negative_points(
        component: Component,
        count: int,
    ) -> list[tuple[int, int]]:
        """Choose deterministic background points inside the expanded box."""
        x0, y0, x1, y1 = component.bbox
        mask = component.mask.astype(bool)
        candidates = [
            (y0, x0),
            (y0, x1),
            (y1, x0),
            (y1, x1),
            ((y0 + y1) // 2, x0),
            ((y0 + y1) // 2, x1),
            (y0, (x0 + x1) // 2),
            (y1, (x0 + x1) // 2),
        ]
        positives = component.positive_points
        selected: list[tuple[int, int]] = []
        for row, col in candidates:
            if mask[row, col]:
                continue
            if any((row - pr) ** 2 + (col - pc) ** 2 < 8 ** 2 for pr, pc in positives):
                continue
            selected.append((row, col))
            if len(selected) >= count:
                break
        return selected

    def _save_debug(
        self,
        debug_dir: str | Path,
        image_rgb: np.ndarray,
        image_pil,
        masks: np.ndarray,
        scores: np.ndarray,
        component_ids: np.ndarray | None = None,
    ) -> None:
        """Save candidate masks, overlays, and scores JSON for debugging."""
        import json
        from PIL import Image as _Image

        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        base_img = np.array(image_pil.convert("RGB")) if image_pil is not None else image_rgb

        score_info: dict[str, dict] = {}
        for idx in range(masks.shape[0]):
            mask = masks[idx]  # bool [H, W]
            area = int(mask.sum())

            # mask PNG (white on black)
            mask_path = debug_dir / f"mask_{idx}.png"
            _Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)

            # overlay PNG
            overlay = base_img.copy().astype(np.float32)
            green = np.zeros_like(overlay)
            green[..., 1] = 255.0
            overlay[mask] = overlay[mask] * 0.4 + green[mask] * 0.6
            overlay_path = debug_dir / f"overlay_mask_{idx}.png"
            _Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(overlay_path)

            score_info[f"mask_{idx}"] = {
                "score": round(float(scores[idx]), 4),
                "area": area,
            }
            if component_ids is not None:
                score_info[f"mask_{idx}"]["component_id"] = int(component_ids[idx])

        scores_path = debug_dir / "scores.json"
        with scores_path.open("w") as f:
            json.dump(score_info, f, indent=2)
