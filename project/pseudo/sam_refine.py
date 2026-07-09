from __future__ import annotations

"""SAM ViT-B wrapper for point-prompted mask generation.

Designed for Google Colab + Google Drive workflow:
  - Checkpoint path passed explicitly (e.g. from Drive mount)
  - Falls back to automatic download if checkpoint not found and
    auto_download=True (useful for first-run on Colab)
"""

from pathlib import Path
from typing import Any

import numpy as np

# Accepts either pseudo.bone_morphology.BoneComponent or
# pseudo.tumor_morphology.TumorComponent — both share the same fields
# (mask, bbox, positive_points, component_id), selected via --dataset.
Component = Any


_SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
_DEFAULT_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"

_SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
_SAM2_DEFAULT_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"
_SAM2_DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# MedSAM2 (bowang-lab/MedSAM2) fine-tunes SAM2's Hiera-tiny backbone on medical
# imagery and ships its own vendored `sam2/` package + config
# (configs/sam2.1_hiera_t512.yaml, note: no "sam2.1/" subfolder, unlike the
# original SAM2 repo's configs/sam2.1/sam2.1_hiera_t.yaml). Its build_sam2()/
# SAM2ImagePredictor API is identical to original SAM2's (confirmed via
# MedSAM2's own app.py), so _init_v2 below is reused as-is — only the
# checkpoint source and config path differ. The two `sam2` packages
# (facebookresearch/sam2 vs bowang-lab/MedSAM2's vendored copy) occupy the
# same Python import name and cannot both be installed in one environment;
# switching sam_version between "v2" and "medsam2" in the same venv requires
# reinstalling the matching package first.
_MEDSAM2_CHECKPOINT_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
_MEDSAM2_DEFAULT_CHECKPOINT_NAME = "MedSAM2_latest.pt"
_MEDSAM2_DEFAULT_MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"


def _download_checkpoint(url: str, dest: Path, label: str) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] Downloading checkpoint to {dest} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"[{label}] Download complete.")


def _download_sam_checkpoint(dest: Path) -> None:
    _download_checkpoint(_SAM_CHECKPOINT_URL, dest, "SAM")


class SAMPredictor:
    """Thin wrapper around segment_anything.SamPredictor / SAM2's SAM2ImagePredictor.

    Both expose the same predict(point_coords, point_labels, box,
    multimask_output) -> (masks, scores, logits) signature, so every method
    below other than __init__ is identical regardless of --sam-version.

    Usage:
        predictor = SAMPredictor(checkpoint_path="/drive/MyDrive/sam_vit_b_01ec64.pth")
        masks = predictor.predict_from_points(image_np, point_prompts)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        auto_download: bool = True,
        device: str = "cuda",
        sam_version: str = "v1",
        sam2_model_cfg: str | None = None,
    ) -> None:
        if sam_version not in {"v1", "v2", "medsam2"}:
            raise ValueError(f"Unknown sam_version '{sam_version}'. Choose from: v1, v2, medsam2.")
        self._sam_version = sam_version
        self._device = device

        if sam_version == "v1":
            self._init_v1(checkpoint_path, auto_download, device)
        elif sam_version == "v2":
            self._init_v2(
                checkpoint_path, auto_download, device,
                sam2_model_cfg or _SAM2_DEFAULT_MODEL_CFG,
                default_checkpoint_name=_SAM2_DEFAULT_CHECKPOINT_NAME,
                checkpoint_url=_SAM2_CHECKPOINT_URL,
                label="SAM2",
            )
        else:
            self._init_v2(
                checkpoint_path, auto_download, device,
                sam2_model_cfg or _MEDSAM2_DEFAULT_MODEL_CFG,
                default_checkpoint_name=_MEDSAM2_DEFAULT_CHECKPOINT_NAME,
                checkpoint_url=_MEDSAM2_CHECKPOINT_URL,
                label="MedSAM2",
            )

    def _init_v1(self, checkpoint_path, auto_download, device) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if checkpoint_path is None:
            checkpoint_path = Path(_DEFAULT_CHECKPOINT_NAME)
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            if auto_download:
                _download_sam_checkpoint(checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"SAM checkpoint not found at {checkpoint_path}. "
                    "Pass auto_download=True or provide the correct path."
                )

        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        self._predictor = SamPredictor(sam)

    def _init_v2(
        self,
        checkpoint_path,
        auto_download,
        device,
        model_cfg,
        default_checkpoint_name,
        checkpoint_url,
        label,
    ) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            install_hint = (
                "pip install git+https://github.com/bowang-lab/MedSAM2.git"
                if label == "MedSAM2"
                else "pip install git+https://github.com/facebookresearch/sam2.git"
            )
            raise ImportError(
                f"sam2 is not installed (needed for --sam-version, {label}). Run: {install_hint}"
            ) from exc

        if checkpoint_path is None:
            checkpoint_path = Path(default_checkpoint_name)
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            if auto_download:
                _download_checkpoint(checkpoint_url, checkpoint_path, label)
            else:
                raise FileNotFoundError(
                    f"{label} checkpoint not found at {checkpoint_path}. "
                    "Pass auto_download=True or provide the correct path."
                )

        sam2_model = self._build_sam2_with_config_fallback(build_sam2, model_cfg, checkpoint_path, device)
        self._predictor = SAM2ImagePredictor(sam2_model)

    @staticmethod
    def _build_sam2_with_config_fallback(build_sam2, model_cfg, checkpoint_path, device):
        """Call build_sam2(), recovering from Hydra not finding the config.

        build_sam2() resolves model_cfg (e.g. "configs/sam2.1_hiera_t512.yaml")
        through Hydra's search path *inside the installed sam2 package*
        (pkg://sam2). Some pip-installable forks (observed with
        bowang-lab/MedSAM2's git+pip install) don't ship their configs/*.yaml
        files as package data, so the .yaml exists in the git checkout but not
        in the installed package — build_sam2() then raises
        hydra.errors.MissingConfigException even though the file is real.
        When that happens, this locates the actual sam2/configs directory on
        disk (next to the installed sam2 package, which pip *does* leave
        alongside the package even when it's not registered as package data)
        and points Hydra at it directly via initialize_config_dir(), then
        retries with just the config's basename.
        """
        try:
            return build_sam2(model_cfg, str(checkpoint_path), device=device)
        except Exception as exc:
            if "MissingConfigException" not in type(exc).__name__:
                raise

        import sam2
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        sam2_package_dir = Path(sam2.__file__).resolve().parent
        config_basename = Path(model_cfg).name
        candidate_dirs = [
            sam2_package_dir / "configs",
            sam2_package_dir.parent / "configs",
            sam2_package_dir.parent / "sam2" / "configs",
        ]
        config_dir = next((d for d in candidate_dirs if (d / config_basename).exists()), None)
        if config_dir is None:
            searched = ", ".join(str(d) for d in candidate_dirs)
            raise FileNotFoundError(
                f"Could not locate '{config_basename}' on disk near the installed sam2 "
                f"package to work around Hydra's MissingConfigException. Searched: {searched}. "
                "The package may need to be reinstalled from source (pip install -e .) "
                "instead of a plain pip install."
            )

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            return build_sam2(config_basename, str(checkpoint_path), device=device)

    def predict_from_points(
        self,
        image_rgb: np.ndarray,
        point_prompts: list[tuple[int, int]],
        debug_dir: str | Path | None = None,
        image_pil=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM with foreground point prompts.

        SAM's multimask_output=True always returns exactly 3 masks regardless
        of how many points are provided. We therefore run predict() once per
        point so that each bone peak generates 3 independent candidate masks.

        Args:
            image_rgb:     [H, W, 3] uint8 RGB numpy array.
            point_prompts: list of (row, col) tuples from extract_prompts.
            debug_dir:     If set, saves mask PNGs, overlay PNGs, and scores.json.
            image_pil:     PIL Image used for overlays (falls back to image_rgb).

        Returns:
            masks:  [P*3, H, W] bool array — 3 candidates per prompt point.
            scores: [P*3] float array — SAM confidence scores per mask.
        """
        if not point_prompts:
            h, w = image_rgb.shape[:2]
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
        """Prompt SAM once per selected bone component.

        prompt_mode:
          point        - strongest structured point only
          joint_points - all structured points in one SAM call
          box          - component bounding box only
          box_point    - component box plus all structured points

        prompt_border_margin removes positive points that lie directly on the
        image border. Those points often make SAM lock onto the hand/wrist
        silhouette instead of the internal bone support.

        max_box_area_ratio drops the box prompt when a component bbox is too
        large relative to the image. The positive points are still used, so SAM
        can refine locally without being encouraged to segment the full hand.
        """
        valid_modes = {"point", "joint_points", "box", "box_point"}
        if prompt_mode not in valid_modes:
            raise ValueError(f"Unknown prompt_mode '{prompt_mode}'. Choose from {sorted(valid_modes)}.")
        if not components:
            h, w = image_rgb.shape[:2]
            return (
                np.zeros((0, h, w), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )

        self._predictor.set_image(image_rgb)
        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        component_ids: list[int] = []

        h, w = image_rgb.shape[:2]

        for component in components:
            original_points = list(component.positive_points)
            points = self._filter_border_points(
                original_points,
                image_shape=(h, w),
                margin=prompt_border_margin,
            )
            if not points:
                points = original_points[:1]
            if prompt_mode == "point":
                points = points[:1]

            point_coords = None
            point_labels = None
            if prompt_mode in {"point", "joint_points", "box_point"} and points:
                point_coords = np.asarray([(col, row) for row, col in points], dtype=np.float32)
                point_labels = np.ones(len(points), dtype=np.int32)
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

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
            all_masks.append(masks)
            all_scores.append(scores)
            component_ids.extend([component.component_id] * masks.shape[0])

        combined_masks = np.concatenate(all_masks, axis=0)
        combined_scores = np.concatenate(all_scores, axis=0)
        component_id_array = np.asarray(component_ids, dtype=np.int32)
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
