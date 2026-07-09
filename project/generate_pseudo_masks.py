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

from config import DATASET_TARGET_COLUMNS, DEFAULT_DATASET, SUPPORTED_DATASETS
from datasets.factory import build_classification_dataset, build_segmentation_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.extract_prompts import extract_point_prompts
from pseudo.morphology_factory import get_morphology_module
from pseudo.prompt_metrics import (
    binary_mask_localization_metrics,
    cam_localization_metrics,
    point_prompt_hit_rate,
)
from pseudo.oracle_diagnostics import oracle_vs_selected_metrics
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAM-H1200/BTXRD pseudo masks via LayerCAM + SAM")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS)
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1",
                        help="Dataset root (RAM-H1200 root or BTXRD root, depending on --dataset)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="Path to sam_vit_b_01ec64.pth (v1) or sam2.1_hiera_tiny.pt (v2); "
                        "auto-downloaded if absent")
    parser.add_argument("--sam-version", type=str, default="v1", choices=["v1", "v2", "medsam2"],
                        help="v1=original SAM (ViT-B, SamPredictor API); "
                        "v2=SAM2 (Hiera-tiny by default, SAM2ImagePredictor — same "
                        "point/box prompt API, different checkpoint/package); "
                        "medsam2=SAM2 fine-tuned on medical imagery (bowang-lab/MedSAM2), "
                        "same API, its own vendored sam2 package/checkpoint/config")
    parser.add_argument("--sam2-model-cfg", type=str, default=None,
                        help="Overrides the default config for --sam-version v2/medsam2 "
                        "(v2 default: configs/sam2.1/sam2.1_hiera_t.yaml; "
                        "medsam2 default: configs/sam2.1_hiera_t512.yaml)")
    parser.add_argument("--target-columns", type=str, default=None,
                        help="Defaults to 'hand' for ramh1200 or 'tumor' for btxrd")
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
    parser.add_argument("--closing-kernel", type=int, default=0)
    parser.add_argument("--opening-kernel", type=int, default=0,
                        help="0 disables opening; recommended for thin hand/wrist bones")
    parser.add_argument("--min-size", type=int, default=40,
                        help="Minimum final component size; kept small for phalanges/carpal bones")
    parser.add_argument("--max-hole-area", type=int, default=0,
                        help="Only fill enclosed holes up to this area; preserves gaps between bones")
    parser.add_argument("--guidance-threshold", type=float, default=0.40,
                        help="Minimum mean bone-likelihood for keeping a final mask component")
    parser.add_argument("--bone-seed-percentile", type=float, default=None,
                        help="Defaults to 88.0 for ramh1200 or 82.0 for btxrd")
    parser.add_argument("--bone-support-percentile", type=float, default=None,
                        help="Defaults to 68.0 for ramh1200 or 55.0 for btxrd")
    parser.add_argument("--morphology-fusion-mode", type=str, default="components",
                        choices=["components", "weighted"])
    parser.add_argument("--sam-prompt-mode", type=str, default="box_point",
                        choices=["point", "joint_points", "box", "box_point"])
    parser.add_argument("--max-bone-components", type=int, default=12)
    parser.add_argument("--points-per-component", type=int, default=3)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.02)
    parser.add_argument("--negative-points-per-component", type=int, default=4)
    parser.add_argument("--prompt-border-margin", type=int, default=2,
                        help="Drop positive SAM points this many pixels from image borders")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.35,
                        help="Drop SAM box prompts larger than this fraction of the image; <=0 disables")
    parser.add_argument("--sam-single-mask", action="store_true")
    parser.add_argument("--disable-bone-morphology", action="store_true",
                        help="Run the original CAM-only baseline without pre-SAM bone morphology")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--preprocessing-mode", type=str, default="none",
                        choices=["none", "clahe", "contrast", "gamma", "foreground_crop"],
                        help="Optional X-ray preprocessing before classifier/CAM")
    parser.add_argument("--selection-method", type=str, default="bone_hybrid",
                        choices=["mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid"],
                        help="CAM-guided mask scoring method")
    parser.add_argument("--fusion-topk", type=int, default=3,
                        help="0=OR all above-thresh, 1=top-1 only, k>1=union top-k, k<0=intersect top-|k|. "
                        "Only used when --disable-best-per-component is set or components are unavailable "
                        "-- best_per_component (on by default whenever component_ids exist) takes priority "
                        "over fusion_topk in select_and_fuse_masks and returns before fusion_topk's branch runs.")
    parser.add_argument("--disable-best-per-component", action="store_true",
                        help="Disable per-component best-candidate selection (which unions the single best "
                        "SAM candidate from every kept morphology component, up to --max-bone-components) "
                        "and fall back to fusion_topk's global top-k selection across all candidates instead. "
                        "Per-component selection can union in a non-lesion component's mask alongside the "
                        "real lesion's, which is invisible to fusion_topk and can dilute Dice more than "
                        "fusion_topk ever could -- this flag isolates that effect for A/B testing.")
    parser.add_argument("--support-clip-kernel", type=int, default=5,
                        help="Clip fused SAM masks to dilated bone support; 0/1 means no dilation, -1 disables")
    parser.add_argument("--debug", action="store_true",
                        help="Save per-image debug outputs (SAM masks, prompt overlays, scores)")
    parser.add_argument("--evaluate-prompt-quality", action="store_true",
                        help="Log CAM localization and point-prompt hit-rate against ground-truth "
                        "masks to prompt_quality.csv. Isolates CAM/prompt failure from SAM/mask-"
                        "selection failure, unlike the final pseudo-mask Dice/IoU. Only meaningful "
                        "on images that actually have a lesion/bone GT mask.")
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
    if args.target_columns is None:
        target_columns = list(DATASET_TARGET_COLUMNS[args.dataset])
    else:
        target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]

    morphology = get_morphology_module(args.dataset)
    default_seed_percentile, default_support_percentile = (
        (88.0, 68.0) if args.dataset == "ramh1200" else (82.0, 55.0)
    )
    bone_seed_percentile = (
        args.bone_seed_percentile if args.bone_seed_percentile is not None else default_seed_percentile
    )
    bone_support_percentile = (
        args.bone_support_percentile if args.bone_support_percentile is not None else default_support_percentile
    )

    dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        preprocessing_mode=args.preprocessing_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    gt_masks_by_name: dict[str, np.ndarray] = {}
    if args.evaluate_prompt_quality:
        seg_dataset = build_segmentation_dataset(
            args.dataset,
            root=args.ram_root,
            split=args.split,
            image_size=args.image_size,
            augment=False,
        )
        for index in range(len(seg_dataset)):
            _, mask_tensor, image_name = seg_dataset[index]
            gt_masks_by_name[str(image_name)] = (mask_tensor[0].numpy() > 0.5)
        print(f"Loaded {len(gt_masks_by_name)} ground-truth masks for prompt-quality evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, classifier_task = load_classifier(args.classifier_checkpoint, len(target_columns), device)
    print(f"Loaded classifier checkpoint task={classifier_task}")
    layercam = LayerCAM(classifier, device=device)

    sam_predictor = SAMPredictor(
        checkpoint_path=args.sam_checkpoint,
        auto_download=(args.sam_checkpoint is None),
        device=str(device),
        sam_version=args.sam_version,
        sam2_model_cfg=args.sam2_model_cfg,
    )

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    prompt_quality_rows: list[list[object]] = []
    skipped_image_names: list[str] = []

    skipped = 0
    processed = 0
    visualized = 0
    process_limit = None if args.process_all or args.max_images <= 0 else args.max_images
    try:
        for images, _, image_names in tqdm(loader, desc="pseudo-masks"):
            images = images.to(device)

            for idx, image_name in enumerate(image_names):
                if process_limit is not None and processed >= process_limit:
                    break
                image_tensor = images[idx : idx + 1]  # [1,3,H,W]
                mask_path = mask_dir / f"{Path(image_name).stem}.png"
                save_visuals = visualized < max(0, args.save_visuals_limit)

                # ── 1. Classifier forward ─────────────────────────────────────
                with torch.no_grad():
                    logits = classifier(image_tensor)
                    class_weights = classifier_class_weights(logits, classifier_task)

                # For multi-label checkpoints, low confidence can mean no reliable anatomy class.
                # For single-label checkpoints, LayerCAM will fall back to the top softmax class.
                if classifier_task != "single-label" and float(class_weights.max()) < args.confidence_threshold:
                    save_mask(np.zeros((args.image_size, args.image_size), dtype=np.uint8), mask_path)
                    skipped += 1
                    skipped_image_names.append(str(image_name))
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
                    visualized += 1

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
                        # Precomputed low-CAM negative points are BTXRD-only for now
                        # (see tumor_morphology.TumorComponent.negative_points);
                        # bone_morphology.build_class_conditioned_components doesn't
                        # accept this kwarg, so only pass it for that dataset.
                        extra_component_kwargs = (
                            {"negative_points_per_component": args.negative_points_per_component}
                            if args.dataset == "btxrd"
                            else {}
                        )
                        bone_likelihood, bone_support, bone_components = morphology.build_class_conditioned_components(
                            image_rgb,
                            per_class_cams,
                            active_weights,
                            seed_percentile=bone_seed_percentile,
                            support_percentile=bone_support_percentile,
                            min_component_area=max(20, args.min_component_area // 2),
                            max_components=args.max_bone_components,
                            points_per_component=args.points_per_component,
                            bbox_padding_ratio=args.bbox_padding_ratio,
                            debug_dir=debug_dir,
                            **extra_component_kwargs,
                        )
                    else:
                        bone_likelihood, bone_support = morphology.build_bone_guidance(
                            image_rgb,
                            fused_cam,
                            seed_percentile=bone_seed_percentile,
                            support_percentile=bone_support_percentile,
                            min_component_area=max(20, args.min_component_area // 2),
                            debug_dir=debug_dir,
                        )
                    prompt_map = morphology.fuse_cam_with_bone_guidance(
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
                        prompt_border_margin=args.prompt_border_margin,
                        max_box_area_ratio=(
                            args.max_box_area_ratio
                            if args.max_box_area_ratio and args.max_box_area_ratio > 0
                            else None
                        ),
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

                # ── 4b. Prompt-quality metrics (optional, pre-SAM diagnostics) ──
                gt_mask = gt_masks_by_name.get(str(image_name)) if args.evaluate_prompt_quality else None
                prompt_quality_entry: list[object] | None = None
                if gt_mask is not None:
                    all_points = (
                        [point for component in bone_components for point in component.positive_points]
                        if bone_components
                        else point_prompts
                    )
                    # bone_components/bone_support come from morphological
                    # reconstruction (seed+support thresholds, not a single
                    # percentile cut), so compare that concrete mask
                    # directly rather than recomputing a percentile cut on
                    # prompt_map, which would not reflect what SAM actually
                    # receives in this mode.
                    if bone_support is not None:
                        fg_metrics = binary_mask_localization_metrics(bone_support, gt_mask)
                    else:
                        fg_metrics = cam_localization_metrics(prompt_map, gt_mask, percentile=args.cam_percentile)
                        fg_metrics = {
                            "iou": fg_metrics["cam_iou"],
                            "recall": fg_metrics["cam_recall"],
                            "precision": fg_metrics["cam_precision"],
                        }
                    hit_metrics = point_prompt_hit_rate(all_points, gt_mask)
                    prompt_quality_entry = [
                        image_name,
                        fg_metrics["iou"],
                        fg_metrics["recall"],
                        fg_metrics["precision"],
                        hit_metrics["point_hit_rate"],
                        hit_metrics["num_points"],
                        hit_metrics["num_hits"],
                    ]

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
                    best_per_component=component_ids is not None and not args.disable_best_per_component,
                    support_clip_kernel=args.support_clip_kernel,
                )

                # ── 5b. SAM-vs-selection oracle diagnostic (optional) ───────────
                if prompt_quality_entry is not None:
                    oracle_metrics = oracle_vs_selected_metrics(
                        sam_masks,
                        refined,
                        gt_mask,
                        bone_support=bone_support,
                        selection_method=args.selection_method,
                        support_clip_kernel=args.support_clip_kernel,
                    )
                    prompt_quality_entry.extend([
                        oracle_metrics["best_single_dice"],
                        oracle_metrics["best_single_dice_clipped"],
                        oracle_metrics["selected_dice"],
                        oracle_metrics["gap_dice"],
                        oracle_metrics["support_loss_dice"],
                        oracle_metrics["selection_loss_dice"],
                    ])
                    prompt_quality_rows.append(prompt_quality_entry)

                # ── 6. Morphological refinement ───────────────────────────────
                final_mask = morphological_refinement(
                    refined,
                    closing_kernel=args.closing_kernel,
                    opening_kernel=args.opening_kernel,
                    min_size=args.min_size,
                    guidance_map=bone_likelihood,
                    guidance_threshold=args.guidance_threshold,
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

    if skipped_image_names:
        skipped_path = args.output_dir / "skipped_low_confidence.txt"
        skipped_path.write_text("\n".join(skipped_image_names) + "\n", encoding="utf-8")
        print(f"Saved list of {len(skipped_image_names)} skipped image names to {skipped_path}")

    if args.evaluate_prompt_quality:
        import csv

        quality_csv = args.output_dir / "prompt_quality.csv"
        with quality_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image_name", "foreground_iou", "foreground_recall", "foreground_precision",
                "point_hit_rate", "num_points", "num_hits",
                "oracle_best_single_dice", "oracle_best_single_dice_clipped", "selected_dice",
                "oracle_gap_dice", "support_loss_dice", "selection_loss_dice",
            ])
            writer.writerows(prompt_quality_rows)

        def _mean(column_index: int) -> float:
            values = [row[column_index] for row in prompt_quality_rows if row[column_index] == row[column_index]]
            return sum(values) / len(values) if values else float("nan")

        print(
            f"Prompt quality ({len(prompt_quality_rows)} images with GT): "
            f"mean foreground_iou={_mean(1):.4f} mean foreground_recall={_mean(2):.4f} "
            f"mean foreground_precision={_mean(3):.4f} mean point_hit_rate={_mean(4):.4f}"
        )
        print(
            f"SAM-vs-selection oracle diagnostic (total gap decomposed into support-clip loss "
            f"vs. mask-selection loss): "
            f"mean oracle_best_single_dice={_mean(7):.4f} "
            f"mean oracle_best_single_dice_clipped={_mean(8):.4f} "
            f"mean selected_dice={_mean(9):.4f} mean total_gap={_mean(10):.4f} "
            f"mean support_loss={_mean(11):.4f} mean selection_loss={_mean(12):.4f} "
            "(large support_loss => bone_support under-covers the lesion, fix morphology "
            "seed/support percentiles, not mask_selection.py; large selection_loss => "
            "bone_hybrid scoring is discarding a good clipped candidate, fix mask_selection.py)"
        )
        print(f"Saved per-image prompt-quality metrics to {quality_csv}")


if __name__ == "__main__":
    main()
