from __future__ import annotations

"""Stage 2 — Generate pseudo bone masks using LayerCAM + SAM.

Pipeline per pipeline.md:
  Image
  → DenseNet121 → logits → sigmoid weights
  → LayerCAM (denseblock2/3/4, confidence-filtered, weighted-average fused)
  → Adaptive threshold → connected components → peak extraction (SAM prompts)
  → SAM ViT-B → candidate masks
  → CAM-guided mask selection (score = mean CAM inside mask)
  → Morphological refinement (closing → opening → fill_holes → remove_small)
  → Final pseudo mask saved as PNG
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.fracatlas import FracAtlasClassificationDataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.extract_prompts import extract_point_prompts
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo masks via LayerCAM + SAM")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="Path to sam_vit_b_01ec64.pth (auto-downloaded if absent)")
    parser.add_argument("--target-columns", type=str, default="hand,leg,hip,shoulder")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pseudo_masks")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Min sigmoid score for a class CAM to participate in fusion")
    parser.add_argument("--cam-percentile", type=float, default=85.0,
                        help="Percentile threshold for foreground extraction (85/90/95)")
    parser.add_argument("--max-points", type=int, default=5,
                        help="Max SAM prompt points per image")
    parser.add_argument("--min-component-area", type=int, default=100,
                        help="Min pixels for a CAM component to generate a prompt")
    parser.add_argument("--mask-score-threshold", type=float, default=0.4,
                        help="Min mean-CAM score for a SAM mask to be kept")
    parser.add_argument("--closing-kernel", type=int, default=5)
    parser.add_argument("--opening-kernel", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--selection-method", type=str, default="mean",
                        choices=["mean", "sum", "mean_area"],
                        help="CAM-guided mask scoring method")
    parser.add_argument("--debug", action="store_true",
                        help="Save per-image debug outputs (SAM masks, prompt overlays, scores)")
    return parser.parse_args()


def load_classifier(
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> tuple[DenseNet121AnatomyClassifier, str]:
    model = DenseNet121AnatomyClassifier(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, state.get("task", "multi-label")


def classifier_class_weights(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "single-label":
        return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def tensor_to_rgb_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [3,H,W] normalised tensor to [H,W,3] uint8 RGB numpy for SAM."""
    pil = tensor_to_pil(image_tensor.detach().cpu())
    return np.array(pil, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path or (args.data_root / "dataset.csv")
    image_root = args.image_root or (args.data_root / "images")
    target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]

    dataset = FracAtlasClassificationDataset(
        csv_path=csv_path,
        image_roots=image_root,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, classifier_task = load_classifier(args.classifier_checkpoint, len(target_columns), device)
    print(f"Loaded classifier checkpoint task={classifier_task}")
    layercam = LayerCAM(classifier, device=device)

    sam_predictor = SAMPredictor(
        checkpoint_path=args.sam_checkpoint,
        auto_download=(args.sam_checkpoint is None),
        device=str(device),
    )

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    try:
        for images, _, image_names in tqdm(loader, desc="pseudo-masks"):
            images = images.to(device)

            for idx, image_name in enumerate(image_names):
                image_tensor = images[idx : idx + 1]  # [1,3,H,W]
                mask_path = mask_dir / f"{Path(image_name).stem}.png"

                # ── 1. Classifier forward ─────────────────────────────────────
                with torch.no_grad():
                    logits = classifier(image_tensor)
                    class_weights = classifier_class_weights(logits, classifier_task)

                # For multi-label checkpoints, low confidence can mean no reliable anatomy class.
                # For single-label checkpoints, LayerCAM will fall back to the top softmax class.
                if classifier_task != "single-label" and float(class_weights.max()) < args.confidence_threshold:
                    save_mask(np.zeros((args.image_size, args.image_size), dtype=np.uint8), mask_path)
                    skipped += 1
                    continue

                # ── 2. LayerCAM fusion ────────────────────────────────────────
                fused_cam, per_class_cams, active_indices = generate_fused_cam(
                    layercam,
                    image_tensor,
                    class_weights=class_weights,
                    confidence_threshold=args.confidence_threshold,
                )

                # save per-class and fused CAM overlays
                image_pil = tensor_to_pil(image_tensor[0].detach().cpu())
                for local_i, cls_i in enumerate(active_indices):
                    cls_name = target_columns[cls_i]
                    save_overlay(
                        image_pil,
                        per_class_cams[local_i],
                        overlay_dir / f"{Path(image_name).stem}_{cls_name}.png",
                    )
                save_overlay(
                    image_pil,
                    fused_cam,
                    overlay_dir / f"{Path(image_name).stem}_fused_layercam.png",
                )

                # ── 3. Prompt extraction ──────────────────────────────────────
                debug_dir = (
                    args.output_dir / "debug" / Path(image_name).stem
                    if args.debug else None
                )
                point_prompts = extract_point_prompts(
                    fused_cam,
                    cam_percentile=args.cam_percentile,
                    max_points=args.max_points,
                    min_component_area=args.min_component_area,
                    debug_dir=debug_dir,
                    image_pil=image_pil,
                )

                # ── 4. SAM candidate masks ────────────────────────────────────
                image_rgb = tensor_to_rgb_numpy(image_tensor[0])
                sam_masks, _sam_scores = sam_predictor.predict_from_points(
                    image_rgb, point_prompts,
                    debug_dir=debug_dir,
                    image_pil=image_pil,
                )

                # ── 5. CAM-guided mask selection ──────────────────────────────
                refined = select_and_fuse_masks(
                    sam_masks,
                    fused_cam,
                    mask_score_threshold=args.mask_score_threshold,
                    selection_method=args.selection_method,
                )

                # ── 6. Morphological refinement ───────────────────────────────
                final_mask = morphological_refinement(
                    refined,
                    closing_kernel=args.closing_kernel,
                    opening_kernel=args.opening_kernel,
                    min_size=args.min_size,
                )

                # ── 7. Save ───────────────────────────────────────────────────
                save_mask(final_mask, mask_path)
    finally:
        layercam.close()

    print(f"\nDone. Masks saved to {mask_dir} (skipped {skipped} low-confidence images)")


if __name__ == "__main__":
    main()
