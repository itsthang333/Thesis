from __future__ import annotations

"""SAM v1 wrapper for point-prompted pseudo-mask generation.

The official Segment Anything registry exposes three matched SAM-v1 model
types (ViT-B/L/H).  Keeping the model type explicit is important for the G4
backbone ablation: checkpoint identity alone must never silently choose an
architecture.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# Structural type: TumorComponent fields are consumed by predict_from_components.
Component = Any


class SAMPredictor:
    """Thin fail-closed wrapper around an official promptable SAM model.

    ``sam_v1`` remains the default and is byte-for-byte compatible with the
    thesis baseline.  The other backends are loaded only from an explicitly
    supplied, hash-locked official source checkout.  Keeping one wrapper is
    deliberate: prompt construction, gallery filtering and downstream
    selection must not change during the G4 backbone/foundation-model panel.
    """

    MODEL_TYPES = ("vit_b", "vit_l", "vit_h")
    BACKENDS = ("sam_v1", "sam2", "sam_med2d", "medsam")

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_type: str = "vit_b",
        backend: str = "sam_v1",
        source_root: str | Path | None = None,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unsupported SAM backend {backend!r}; choose from {self.BACKENDS}")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM {backend}/{model_type} checkpoint not found at {checkpoint_path}. "
                "Provide the matching official SAM-v1 checkpoint explicitly."
            )

        if backend == "sam_v1":
            if model_type not in self.MODEL_TYPES:
                raise ValueError(
                    f"Unsupported SAM-v1 model type {model_type!r}; choose from {self.MODEL_TYPES}"
                )
            try:
                from segment_anything import (
                    SamAutomaticMaskGenerator,
                    SamPredictor,
                    sam_model_registry,
                )
            except ImportError as exc:
                raise ImportError(
                    "segment_anything is not installed. Run: pip install "
                    "git+https://github.com/facebookresearch/segment-anything.git"
                ) from exc
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            predictor = SamPredictor(sam)
            automatic_mask_generator_cls = SamAutomaticMaskGenerator
        elif backend == "sam2":
            root = self._require_source_root(source_root, "sam2")
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            allowed = {"sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"}
            if model_type not in allowed:
                raise ValueError(
                    f"Unsupported SAM2 model type {model_type!r}; choose from {tuple(allowed)}"
                )
            sam = build_sam2(
                allowed[model_type],
                str(checkpoint_path),
                device=device,
                apply_postprocessing=True,
            )
            predictor = SAM2ImagePredictor(sam)
            automatic_mask_generator_cls = SAM2AutomaticMaskGenerator
        elif backend == "sam_med2d":
            root = self._require_source_root(source_root, "SAM-Med2D")
            package = self._load_segment_anything_alias(
                "btxrd_sam_med2d_segment_anything", root / "segment_anything"
            )
            predictor_mod = importlib.import_module(
                f"{package.__name__}.predictor_sammed"
            )
            if model_type != "vit_b_256_adapter":
                raise ValueError("SAM-Med2D official screening requires vit_b_256_adapter")
            args = SimpleNamespace(
                image_size=256,
                sam_checkpoint=str(checkpoint_path),
                encoder_adapter=True,
            )
            sam = package.sam_model_registry["vit_b"](args)
            sam.to(device=device)
            sam.eval()
            predictor = predictor_mod.SammedPredictor(sam)
            automatic_mask_generator_cls = package.SamAutomaticMaskGenerator
        else:  # medsam
            root = self._require_source_root(source_root, "MedSAM")
            package = self._load_segment_anything_alias(
                "btxrd_medsam_segment_anything", root / "segment_anything"
            )
            if model_type != "vit_b":
                raise ValueError("Official MedSAM checkpoint is a ViT-B model")
            sam = package.sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            sam.eval()
            predictor = package.SamPredictor(sam)
            automatic_mask_generator_cls = package.SamAutomaticMaskGenerator

        self._predictor = predictor
        self._automatic_mask_generator_cls = automatic_mask_generator_cls
        self.backend = backend
        self.model_type = model_type
        self.last_prompt_stats: dict[str, int] = {}

    @staticmethod
    def _require_source_root(source_root: str | Path | None, name: str) -> Path:
        if source_root is None:
            raise ValueError(f"{name} requires an explicit --sam-source-root")
        root = Path(source_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Official {name} source root not found: {root}")
        return root

    @staticmethod
    def _load_segment_anything_alias(alias: str, package_dir: Path):
        """Load a SAM-family fork without shadowing installed SAM-v1.

        MedSAM and SAM-Med2D both retain the top-level package name
        ``segment_anything``.  Loading them under a private alias preserves
        relative imports and prevents accidental import-order contamination.
        """
        if alias in sys.modules:
            return sys.modules[alias]
        init_path = package_dir / "__init__.py"
        if not init_path.is_file():
            raise FileNotFoundError(f"Missing official package entrypoint: {init_path}")
        spec = importlib.util.spec_from_file_location(
            alias,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official SAM package at {package_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module

    def predict_grid_gallery(
        self,
        image_rgb: np.ndarray,
        *,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a Pro2SAM-style dense grid gallery with official SAM.

        This deliberately delegates point batching, duplicate removal, and
        stability filtering to ``SamAutomaticMaskGenerator``.  Each retained
        mask remains an independent candidate; no class label, annotation, or
        lesion-size information is used.
        """
        if points_per_side <= 0:
            raise ValueError("points_per_side must be positive")
        if points_per_batch <= 0:
            raise ValueError("points_per_batch must be positive")
        if not 0.0 <= pred_iou_thresh <= 1.0:
            raise ValueError("pred_iou_thresh must be in [0, 1]")
        if not 0.0 <= stability_score_thresh <= 1.0:
            raise ValueError("stability_score_thresh must be in [0, 1]")
        if not 0.0 <= box_nms_thresh <= 1.0:
            raise ValueError("box_nms_thresh must be in [0, 1]")

        generator = self._automatic_mask_generator_cls(
            self._predictor.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=0,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )
        records = generator.generate(image_rgb)
        h, w = image_rgb.shape[:2]
        if records:
            masks = np.stack(
                [np.asarray(record["segmentation"], dtype=bool) for record in records],
                axis=0,
            )
            scores = np.asarray(
                [float(record["predicted_iou"]) for record in records],
                dtype=np.float32,
            )
        else:
            masks = np.zeros((0, h, w), dtype=bool)
            scores = np.zeros(0, dtype=np.float32)

        grid_points = points_per_side * points_per_side
        self.last_prompt_stats = {
            "sam_prompt_calls": grid_points,
            "unique_positive_prompt_points": grid_points,
            "unique_negative_prompt_points": 0,
            "unique_prompt_points": grid_points,
            "box_prompt_calls": 0,
        }
        return masks, scores

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
