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

from datasets.ramh1200 import RAMH1200ClassificationDataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.extract_prompts import extract_point_prompts
from pseudo.bone_morphology import (
    build_bone_guidance,
    build_class_conditioned_components,
    fuse_cam_with_bone_guidance,
)
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAM-H1200 pseudo masks via LayerCAM + SAM")
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="Path to sam_vit_b_01ec64.pth (auto-downloaded if absent)")
    parser.add_argument("--target-columns", type=str, default="hand")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pseudo_masks")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Preview mode limit; 0 processes all images (the notebook preview passes 10)")
    parser.add_argument("--process-all", action="store_true",
                        help="Process the full dataset when generating pseudo masks for segmentation training")
    parser.add_argument("--save-visuals-limit", type=int, default=10,
                        help="Maximum number of images for which CAM overlays/debug visuals are saved")
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
    parser.add_argument("--opening-kernel", type=int, default=0,
                        help="0 disables opening; recommended for thin hand/wrist bones")
    parser.add_argument("--min-size", type=int, default=40,
                        help="Minimum final component size; kept small for phalanges/carpal bones")
    parser.add_argument("--max-hole-area", type=int, default=500,
                        help="Only fill enclosed holes up to this area; preserves gaps between bones")
    parser.add_argument("--bone-seed-percentile", type=float, default=88.0)
    parser.add_argument("--bone-support-percentile", type=float, default=68.0)
    parser.add_argument("--morphology-fusion-mode", type=str, default="components",
                        choices=["components", "weighted"])
    parser.add_argument("--sam-prompt-mode", type=str, default="box_point",
                        choices=["point", "joint_points", "box", "box_point"])
    parser.add_argument("--max-bone-components", type=int, default=12)
    parser.add_argument("--points-per-component", type=int, default=3)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.02)
    parser.add_argument("--negative-points-per-component", type=int, default=0)
    parser.add_argument("--sam-single-mask", action="store_true")
    parser.add_argument("--disable-bone-morphology", action="store_true",
                        help="Run the original CAM-only baseline without pre-SAM bone morphology")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--selection-method", type=str, default="bone_hybrid",
                        choices=["mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid"],
                        help="CAM-guided mask scoring method")
    parser.add_argument("--fusion-topk", type=int, default=3,
                        help="0=OR all above-thresh, 1=top-1 only, k>1=union top-k, k<0=intersect top-|k|")
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
    target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]

    dataset = RAMH1200ClassificationDataset(
        root=args.ram_root,
        split=args.split,
        target_columns=target_columns,
        image_size=args.image_size,
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
    processed = 0
    process_limit = None if args.process_all or args.max_images <= 0 else args.max_images
    try:
        for images, _, image_names in tqdm(loader, desc="pseudo-masks"):
            images = images.to(device)

            for idx, image_name in enumerate(image_names):
                if process_limit is not None and processed >= process_limit:
                    break
                image_tensor = images[idx : idx + 1]  # [1,3,H,W]
                mask_path = mask_dir / f"{Path(image_name).stem}.png"
                save_visuals = processed < max(0, args.save_visuals_limit)

                # ── 1. Classifier forward ─────────────────────────────────────
                with torch.no_grad():
                    logits = classifier(image_tensor)
                    class_weights = classifier_class_weights(logits, classifier_task)

                # For multi-label checkpoints, low confidence can mean no reliable anatomy class.
                # For single-label checkpoints, LayerCAM will fall back to the top softmax class.
                if classifier_task != "single-label" and float(class_weights.max()) < args.confidence_threshold:
                    save_mask(np.zeros((args.image_size, args.image_size), dtype=np.uint8), mask_path)
                    skipped += 1
                    processed += 1
                    continue

                # ── 2. LayerCAM fusion ────────────────────────────────────────
                fused_cam, per_class_cams, active_indices = generate_fused_cam(
                    layercam,
                    image_tensor,
                    class_weights=class_weights,
                    confidence_threshold=args.confidence_threshold,
                )

                image_pil = tensor_to_pil(image_tensor[0].detach().cpu())
                image_rgb = tensor_to_rgb_numpy(image_tensor[0])
                if save_visuals:
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
                    if args.debug and save_visuals else None
                )
                bone_likelihood = None
                bone_support = None
                bone_components = []
                prompt_map = fused_cam
                if not args.disable_bone_morphology:
                    if args.morphology_fusion_mode == "components":
                        active_weights = [float(class_weights[i]) for i in active_indices]
                        bone_likelihood, bone_support, bone_components = build_class_conditioned_components(
                            image_rgb,
                            per_class_cams,
                            active_weights,
                            seed_percentile=args.bone_seed_percentile,
                            support_percentile=args.bone_support_percentile,
                            min_component_area=max(20, args.min_component_area // 2),
                            max_components=args.max_bone_components,
                            points_per_component=args.points_per_component,
                            bbox_padding_ratio=args.bbox_padding_ratio,
                            debug_dir=debug_dir,
                        )
                    else:
                        bone_likelihood, bone_support = build_bone_guidance(
                            image_rgb,
                            fused_cam,
                            seed_percentile=args.bone_seed_percentile,
                            support_percentile=args.bone_support_percentile,
                            min_component_area=max(20, args.min_component_area // 2),
                            debug_dir=debug_dir,
                        )
                    prompt_map = fuse_cam_with_bone_guidance(
                        fused_cam,
                        bone_likelihood,
                        bone_support,
                    )

                # ── 4. SAM candidate masks ────────────────────────────────────
                component_ids = None
                if bone_components:
                    sam_masks, sam_scores, component_ids = sam_predictor.predict_from_components(
                        image_rgb,
                        bone_components,
                        prompt_mode=args.sam_prompt_mode,
                        multimask_output=not args.sam_single_mask,
                        negative_points_per_component=args.negative_points_per_component,
                        debug_dir=debug_dir,
                        image_pil=image_pil,
                    )
                else:
                    point_prompts = extract_point_prompts(
                        prompt_map,
                        cam_percentile=args.cam_percentile,
                        max_points=args.max_points,
                        min_component_area=args.min_component_area,
                        support_mask=bone_support,
                        debug_dir=debug_dir,
                        image_pil=image_pil,
                    )
                    sam_masks, sam_scores = sam_predictor.predict_from_points(
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
                    fusion_topk=args.fusion_topk,
                    bone_likelihood=bone_likelihood,
                    bone_support=bone_support,
                    sam_scores=sam_scores,
                    component_ids=component_ids,
                    component_masks=(
                        np.stack([component.mask for component in bone_components])
                        if bone_components else None
                    ),
                    best_per_component=component_ids is not None,
                )

                # ── 6. Morphological refinement ───────────────────────────────
                final_mask = morphological_refinement(
                    refined,
                    closing_kernel=args.closing_kernel,
                    opening_kernel=args.opening_kernel,
                    min_size=args.min_size,
                    guidance_map=bone_likelihood,
                    max_hole_area=args.max_hole_area,
                )

                # ── 7. Save ───────────────────────────────────────────────────
                save_mask(final_mask, mask_path)
                processed += 1
            if process_limit is not None and processed >= process_limit:
                break
    finally:
        layercam.close()

    mode = "full dataset" if args.process_all else f"preview ({processed} images)"
    print(f"\nDone: {mode}. Masks saved to {mask_dir} (skipped {skipped} low-confidence images)")


if __name__ == "__main__":
    main()
