from __future__ import annotations

"""SAM ViT-B wrapper for point-prompted mask generation.

Designed for Google Colab + Google Drive workflow:
  - Checkpoint path passed explicitly (e.g. from Drive mount)
  - Falls back to automatic download if checkpoint not found and
    auto_download=True (useful for first-run on Colab)
"""

from pathlib import Path

import numpy as np


_SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
_DEFAULT_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"


def _download_sam_checkpoint(dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAM] Downloading SAM ViT-B checkpoint to {dest} ...")
    urllib.request.urlretrieve(_SAM_CHECKPOINT_URL, str(dest))
    print("[SAM] Download complete.")


class SAMPredictor:
    """Thin wrapper around segment_anything.SamPredictor.

    Usage:
        predictor = SAMPredictor(checkpoint_path="/drive/MyDrive/sam_vit_b_01ec64.pth")
        masks = predictor.predict_from_points(image_np, point_prompts)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        auto_download: bool = True,
        device: str = "cuda",
    ) -> None:
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
        self._device = device

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

    def _save_debug(
        self,
        debug_dir: str | Path,
        image_rgb: np.ndarray,
        image_pil,
        masks: np.ndarray,
        scores: np.ndarray,
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

        scores_path = debug_dir / "scores.json"
        with scores_path.open("w") as f:
            json.dump(score_info, f, indent=2)
