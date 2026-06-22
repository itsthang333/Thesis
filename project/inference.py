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

from datasets.fracatlas import _make_classification_transform
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from models.unet import UNet
from pseudo.generate_layercam import generate_fused_cam
from pseudo.extract_prompts import extract_point_prompts
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import overlay_heatmap, save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full WSSS inference pipeline")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--segmentation-checkpoint", type=Path,
                        default=ROOT / "outputs" / "segmentation" / "best_unet.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inference")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--cam-percentile", type=float, default=85.0)
    parser.add_argument("--max-points", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--mask-score-threshold", type=float, default=0.4)
    parser.add_argument("--closing-kernel", type=int, default=5)
    parser.add_argument("--opening-kernel", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--selection-method", type=str, default="mean",
                        choices=["mean", "sum", "mean_area"],
                        help="CAM-guided mask scoring method")
    parser.add_argument("--debug", action="store_true",
                        help="Save SAM candidate masks and prompt overlays for debugging")
    return parser.parse_args()


def load_classifier(path: Path, device: torch.device) -> tuple[DenseNet121AnatomyClassifier, list[str]]:
    checkpoint = torch.load(path, map_location="cpu")
    target_columns = checkpoint.get("target_columns", ["hand", "leg", "hip", "shoulder"])
    model = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), list(target_columns)


def load_segmentation_model(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location="cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image_path.stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, target_columns = load_classifier(args.classifier_checkpoint, device)
    segmentation_model = load_segmentation_model(args.segmentation_checkpoint, device)
    layercam = LayerCAM(classifier, device=device)
    sam_predictor = SAMPredictor(
        checkpoint_path=args.sam_checkpoint,
        auto_download=(args.sam_checkpoint is None),
        device=str(device),
    )

    try:
        # ── Load image ──────────────────────────────────────────────────────
        transform = _make_classification_transform(args.image_size, augment=False)
        image_pil = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(device)  # [1,3,H,W]

        # ── Stage 1: classifier → class weights ─────────────────────────────
        with torch.no_grad():
            logits = classifier(image_tensor)
            class_weights = torch.sigmoid(logits)[0].detach().cpu().numpy()

        # ── Stage 2: LayerCAM → fused CAM ───────────────────────────────────
        fused_cam, per_class_cams, active_indices = generate_fused_cam(
            layercam,
            image_tensor,
            class_weights=class_weights,
            confidence_threshold=args.confidence_threshold,
        )

        # save per-class overlays
        image_pil_denorm = tensor_to_pil(image_tensor[0].detach().cpu())
        for local_i, cls_i in enumerate(active_indices):
            save_overlay(
                image_pil_denorm,
                per_class_cams[local_i],
                args.output_dir / f"{stem}_{target_columns[cls_i]}_cam.png",
            )
        save_overlay(image_pil_denorm, fused_cam,
                     args.output_dir / f"{stem}_fused_layercam.png")

        # ── Stage 3: prompt extraction ───────────────────────────────────────
        debug_dir = args.output_dir / "debug" / stem if args.debug else None
        point_prompts = extract_point_prompts(
            fused_cam,
            cam_percentile=args.cam_percentile,
            max_points=args.max_points,
            min_component_area=args.min_component_area,
            debug_dir=debug_dir,
            image_pil=image_pil_denorm,
        )

        # ── Stage 4: SAM ────────────────────────────────────────────────────
        image_rgb = np.array(image_pil_denorm, dtype=np.uint8)
        sam_masks, _ = sam_predictor.predict_from_points(
            image_rgb, point_prompts,
            debug_dir=debug_dir,
            image_pil=image_pil_denorm,
        )

        # ── Stage 5: CAM-guided mask selection ──────────────────────────────
        refined = select_and_fuse_masks(
            sam_masks, fused_cam,
            mask_score_threshold=args.mask_score_threshold,
            selection_method=args.selection_method,
        )

        # ── Stage 6: morphological refinement ───────────────────────────────
        pseudo_mask = morphological_refinement(
            refined,
            closing_kernel=args.closing_kernel,
            opening_kernel=args.opening_kernel,
            min_size=args.min_size,
        )
        save_mask(pseudo_mask, args.output_dir / f"{stem}_pseudo_mask.png")

        # ── Stage 7: U-Net segmentation ─────────────────────────────────────
        with torch.no_grad():
            seg_logits = segmentation_model(image_tensor)
            seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
            seg_mask = (seg_prob >= 0.5).astype(np.uint8)

        save_mask(seg_mask, args.output_dir / f"{stem}_segmentation_mask.png")

        final_overlay = overlay_heatmap(image_pil_denorm, seg_prob, alpha=0.35)
        Image.fromarray(final_overlay).save(args.output_dir / f"{stem}_final_overlay.png")
    finally:
        layercam.close()

    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
