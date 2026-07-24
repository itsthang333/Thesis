from __future__ import annotations

"""Final deployment inference for the trained BTXRD U-Net.

CAM, morphology, and SAM are training-time pseudo-label components and are
intentionally absent from this deployment entrypoint.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import apply_clahe, make_segmentation_image_transform
from models.unet import (
    architecture_metadata,
    architecture_name_from_metadata,
    build_segmentation_model,
)
from pseudo.visualization import overlay_heatmap, save_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final BTXRD U-Net deployment inference")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "segmentation" / "best_unet.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inference")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Explicit override; otherwise restored from the checkpoint.",
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=None,
        help="Explicit override; otherwise restored from the checkpoint.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_segmentation_architecture(checkpoint: dict[str, object]) -> str:
    """Resolve and strictly validate checkpoint architecture metadata.

    Checkpoints created before architecture provenance was added are accepted
    only as the legacy plain U-Net. New checkpoints must match one of the
    canonical metadata dictionaries exactly.
    """
    metadata = checkpoint.get("architecture")
    architecture_name = architecture_name_from_metadata(metadata)
    expected_metadata = architecture_metadata(architecture_name)
    if metadata is not None and metadata != expected_metadata:
        raise ValueError(
            f"Unsupported checkpoint architecture: {metadata!r}; "
            f"expected {expected_metadata!r}"
        )
    return architecture_name


def load_segmentation_model(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture_name = resolve_segmentation_architecture(checkpoint)
    dataset = checkpoint.get("dataset")
    if dataset not in (None, "btxrd"):
        raise ValueError(f"Expected a BTXRD checkpoint, got dataset={dataset!r}")
    model = build_segmentation_model(architecture_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def run_inference(
    image_path: Path,
    output_dir: Path,
    model: nn.Module,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
    image_size_override: int | None,
    threshold_override: float | None,
) -> None:
    image_size = image_size_override or int(checkpoint.get("image_size", 320))
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(checkpoint.get("decision_threshold", 0.5))
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"segmentation threshold must be in [0,1], got {threshold}")

    use_clahe = bool(checkpoint.get("use_clahe", False))
    source_image = Image.open(image_path).convert("RGB")
    original_size = source_image.size
    model_image = apply_clahe(source_image) if use_clahe else source_image
    tensor = make_segmentation_image_transform(image_size)(model_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_original = np.asarray(
        Image.fromarray(probability.astype(np.float32), mode="F").resize(
            original_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    mask = probability_original >= threshold
    save_mask(mask, output_dir / f"{image_path.stem}_segmentation_mask.png")
    Image.fromarray(overlay_heatmap(source_image, probability_original, alpha=0.35)).save(
        output_dir / f"{image_path.stem}_final_overlay.png"
    )
    metadata = {
        "mode": "unet",
        "dataset": "btxrd",
        "source_image": str(image_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "image_size": image_size,
        "decision_threshold": threshold,
        "architecture": checkpoint.get(
            "architecture",
            architecture_metadata("unet"),
        ),
        "use_clahe": use_clahe,
        "original_size": list(original_size),
        "device": str(device),
    }
    (output_dir / f"{image_path.stem}_inference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_segmentation_model(args.segmentation_checkpoint, device)
    run_inference(
        args.image_path,
        args.output_dir,
        model,
        checkpoint,
        args.segmentation_checkpoint,
        device,
        args.image_size,
        args.segmentation_threshold,
    )
    print(f"U-Net deployment outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
