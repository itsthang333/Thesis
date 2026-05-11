from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.fracatlas import _make_image_transform
from models.classifier import DenseNet121AnatomyClassifier
from models.gradcam import GradCAM
from models.unet import UNet
from pseudo.cam_to_mask import aggregate_cam_heatmaps, cam_to_pseudo_mask, normalize_min_max
from pseudo.visualization import overlay_heatmap, save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FracAtlas WSSS inference pipeline")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, default=ROOT / "outputs" / "classifier" / "best.pt")
    parser.add_argument("--segmentation-checkpoint", type=Path, default=ROOT / "outputs" / "segmentation" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inference")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--fusion", type=str, default="weighted_mean", choices=("weighted_mean", "max"))
    return parser.parse_args()


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    transform = _make_image_transform(image_size, augment=False)
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def load_classifier(path: Path, device: torch.device) -> tuple[DenseNet121AnatomyClassifier, list[str]]:
    checkpoint = torch.load(path, map_location="cpu")
    target_columns = checkpoint.get("target_columns", ["hand", "leg", "hip", "shoulder"])
    model = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), list(target_columns)


def load_segmentation_model(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location="cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, target_columns = load_classifier(args.classifier_checkpoint, device)
    segmentation_model = load_segmentation_model(args.segmentation_checkpoint, device)
    gradcam = GradCAM(classifier, classifier.features.denseblock4, device=device)

    image_tensor = load_image(args.image_path, args.image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(image_tensor)
        class_weights = torch.sigmoid(logits)[0].detach().cpu().numpy()

    per_class_cams: list[np.ndarray] = []
    class_names = target_columns
    for class_index in range(len(class_names)):
        cam_output = gradcam.cam_for_class(image_tensor, class_index=class_index)
        per_class_cams.append(cam_output.cam[0].detach().cpu().numpy())

    cam = aggregate_cam_heatmaps(per_class_cams, weights=class_weights, mode=args.fusion)
    pseudo_mask = cam_to_pseudo_mask(cam, percentile=args.percentile, min_area=args.min_area, kernel_size=args.kernel_size)
    with torch.no_grad():
        segmentation_logits = segmentation_model(image_tensor)
        segmentation_prob = torch.sigmoid(segmentation_logits)[0, 0].detach().cpu().numpy()
        segmentation_mask = (segmentation_prob >= 0.5).astype(np.uint8)

    image_pil = tensor_to_pil(image_tensor[0].detach().cpu())
    stem = args.image_path.stem
    save_overlay(image_pil, cam, args.output_dir / f"{stem}_cam_overlay.png")
    save_mask(pseudo_mask, args.output_dir / f"{stem}_pseudo_mask.png")
    save_mask(segmentation_mask, args.output_dir / f"{stem}_segmentation_mask.png")

    final_overlay = overlay_heatmap(image_pil, segmentation_prob, alpha=0.35)
    Image.fromarray(final_overlay).save(args.output_dir / f"{stem}_final_overlay.png")
    gradcam.close()


if __name__ == "__main__":
    main()
