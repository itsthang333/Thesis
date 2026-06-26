from __future__ import annotations

"""Visualize the full WSSS pipeline for a single image.

Produces a 6-panel figure (like paper figures):
  Original | LayerCAM | Foreground | Prompt Points | SAM Mask | Pseudo Mask

Optionally adds a 7th panel for the final U-Net segmentation overlay.

Usage:
    python visualize_pipeline.py \
        --image-path /path/to/image.jpg \
        --classifier-checkpoint outputs/classifier/best_classifier.pt \
        --sam-checkpoint /drive/MyDrive/sam_vit_b_01ec64.pth \
        [--segmentation-checkpoint outputs/segmentation/best_unet.pt] \
        [--output-path outputs/pipeline_viz.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_ANATOMY_COLUMNS
from datasets.common import make_classification_transform
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from models.unet import UNet
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
from pseudo.visualization import overlay_heatmap, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full WSSS pipeline for one image")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--segmentation-checkpoint", type=Path, default=None,
                        help="Optional: add U-Net overlay as 7th panel")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Where to save the figure (default: outputs/viz/<stem>_pipeline.png)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--cam-percentile", type=float, default=85.0)
    parser.add_argument("--max-points", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--mask-score-threshold", type=float, default=0.4)
    parser.add_argument("--selection-method", type=str, default="bone_hybrid",
                        choices=["mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid"])
    parser.add_argument("--fusion-topk", type=int, default=3,
                        help="0=OR all above-thresh, k>1=union top-k, k<0=intersect top-|k|")
    parser.add_argument("--closing-kernel", type=int, default=5)
    parser.add_argument("--opening-kernel", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--max-hole-area", type=int, default=500)
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
    parser.add_argument("--disable-bone-morphology", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Save SAM candidate masks, prompt overlays, scores.json alongside the strip")
    return parser.parse_args()


def _jet_rgb(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _blend(image_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    ovl = overlay_rgb.astype(np.float32)
    return np.clip((1 - alpha) * img + alpha * ovl, 0, 255).astype(np.uint8)


def _make_foreground_panel(image_rgb: np.ndarray, bone_cam: np.ndarray, cam_percentile: float) -> np.ndarray:
    threshold = float(np.percentile(bone_cam, cam_percentile))
    fg = (bone_cam > threshold).astype(np.uint8)
    # tint foreground cyan on original image
    base = image_rgb.astype(np.float32)
    cyan = np.zeros_like(base)
    cyan[..., 1] = 255.0
    cyan[..., 2] = 255.0
    result = base.copy()
    mask_bool = fg.astype(bool)
    result[mask_bool] = base[mask_bool] * 0.4 + cyan[mask_bool] * 0.6
    return result.clip(0, 255).astype(np.uint8)


def _make_prompts_panel(image_rgb: np.ndarray, bone_cam: np.ndarray, point_prompts: list[tuple[int, int]]) -> np.ndarray:
    cam_jet = _jet_rgb(bone_cam)
    blended = _blend(image_rgb, cam_jet, alpha=0.45)
    pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil)
    h, w = bone_cam.shape
    radius = max(6, min(h, w) // 60)
    for i, (r, c) in enumerate(point_prompts):
        draw.ellipse([c - radius, r - radius, c + radius, r + radius],
                     fill=(255, 0, 0), outline=(255, 255, 255))
        draw.text((c + radius + 2, r - radius), str(i + 1), fill=(255, 255, 0))
    return np.array(pil)


def _make_component_prompts_panel(
    image_rgb: np.ndarray,
    bone_cam: np.ndarray,
    components,
) -> np.ndarray:
    points = [point for component in components for point in component.positive_points]
    panel = _make_prompts_panel(image_rgb, bone_cam, points)
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    for index, component in enumerate(components):
        x0, y0, x1, y1 = component.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        draw.text((x0 + 2, y0 + 2), f"C{index}", fill=(0, 255, 0))
    return np.array(pil)


def _make_sam_panel(image_rgb: np.ndarray, sam_masks: np.ndarray) -> np.ndarray:
    """Show all candidate SAM masks as distinct color overlays."""
    result = image_rgb.astype(np.float32)
    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
    ]
    for i in range(min(sam_masks.shape[0], len(colors))):
        m = sam_masks[i].astype(bool)
        c = np.array(colors[i % len(colors)], dtype=np.float32)
        result[m] = result[m] * 0.35 + c * 0.65
    return result.clip(0, 255).astype(np.uint8)


def _add_label(panel: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(20, 20, 20))
    draw.text((4, 4), label, fill=(255, 255, 255))
    return np.array(pil)


def build_figure(panels: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    labeled = [_add_label(img, title) for title, img in panels]
    widths = [p.shape[1] for p in labeled]
    heights = [p.shape[0] for p in labeled]
    total_w = sum(widths) + 4 * (len(labeled) - 1)
    max_h = max(heights)
    canvas = np.full((max_h, total_w, 3), 30, dtype=np.uint8)
    x = 0
    for p in labeled:
        h, w = p.shape[:2]
        canvas[:h, x:x + w] = p
        x += w + 4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    print(f"Saved pipeline visualization to {output_path}")


def classifier_class_weights(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "single-label":
        return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    stem = args.image_path.stem
    output_path = args.output_path or (ROOT / "outputs" / "viz" / f"{stem}_pipeline.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ──────────────────────────────────────────────────────────
    clf_ckpt = torch.load(args.classifier_checkpoint, map_location="cpu")
    target_columns = clf_ckpt.get("target_columns", list(DEFAULT_ANATOMY_COLUMNS))
    classifier_task = clf_ckpt.get("task", "multi-label")
    classifier = DenseNet121AnatomyClassifier(num_classes=len(target_columns), pretrained=False)
    classifier.load_state_dict(clf_ckpt["model_state_dict"], strict=True)
    classifier = classifier.to(device).eval()

    layercam = LayerCAM(classifier, device=device)
    sam_predictor = SAMPredictor(
        checkpoint_path=args.sam_checkpoint,
        auto_download=(args.sam_checkpoint is None),
        device=str(device),
    )

    seg_model = None
    if args.segmentation_checkpoint and args.segmentation_checkpoint.exists():
        seg_ckpt = torch.load(args.segmentation_checkpoint, map_location="cpu")
        seg_model = UNet(in_channels=3, out_channels=1, base_channels=64)
        seg_model.load_state_dict(seg_ckpt["model_state_dict"], strict=True)
        seg_model = seg_model.to(device).eval()

    try:
        # ── Load & preprocess image ──────────────────────────────────────────
        transform = make_classification_transform(args.image_size, augment=False)
        image_pil = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        image_pil_denorm = tensor_to_pil(image_tensor[0].detach().cpu())
        image_rgb = np.array(image_pil_denorm, dtype=np.uint8)

        # ── Stage 1: classifier ──────────────────────────────────────────────
        with torch.no_grad():
            logits = classifier(image_tensor)
            class_weights = classifier_class_weights(logits, classifier_task)

        active_labels = [target_columns[i] for i, w in enumerate(class_weights) if w >= args.confidence_threshold]
        print(f"Classifier task: {classifier_task}")
        print(f"Active classes: {active_labels} (scores: {class_weights.round(3)})")

        # ── Stage 2: LayerCAM ────────────────────────────────────────────────
        fused_cam, per_class_cams, active_indices = generate_fused_cam(
            layercam, image_tensor,
            class_weights=class_weights,
            confidence_threshold=args.confidence_threshold,
        )

        # ── Stage 3: Prompt extraction ───────────────────────────────────────
        debug_dir = output_path.parent / "debug" / stem if args.debug else None
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
            prompt_map = fuse_cam_with_bone_guidance(fused_cam, bone_likelihood, bone_support)
        point_prompts = [
            point
            for component in bone_components
            for point in component.positive_points
        ]
        if not point_prompts:
            point_prompts = extract_point_prompts(
                prompt_map,
                cam_percentile=args.cam_percentile,
                max_points=args.max_points,
                min_component_area=args.min_component_area,
                support_mask=bone_support,
                debug_dir=debug_dir,
                image_pil=image_pil_denorm,
            )
        print(f"Prompt points: {point_prompts}")

        # ── Stage 4: SAM ─────────────────────────────────────────────────────
        component_ids = None
        if bone_components:
            sam_masks, sam_scores, component_ids = sam_predictor.predict_from_components(
                image_rgb,
                bone_components,
                prompt_mode=args.sam_prompt_mode,
                multimask_output=not args.sam_single_mask,
                negative_points_per_component=args.negative_points_per_component,
                debug_dir=debug_dir,
                image_pil=image_pil_denorm,
            )
        else:
            sam_masks, sam_scores = sam_predictor.predict_from_points(
                image_rgb, point_prompts,
                debug_dir=debug_dir,
                image_pil=image_pil_denorm,
            )
        print(f"SAM masks: {sam_masks.shape[0]}, scores: {sam_scores.round(3)}")

        # ── Stage 5: Mask selection ──────────────────────────────────────────
        refined = select_and_fuse_masks(
            sam_masks, fused_cam,
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

        # ── Stage 6: Morphological refinement ───────────────────────────────
        pseudo_mask = morphological_refinement(
            refined,
            closing_kernel=args.closing_kernel,
            opening_kernel=args.opening_kernel,
            min_size=args.min_size,
            guidance_map=bone_likelihood,
            max_hole_area=args.max_hole_area,
        )

        # ── Build panels ─────────────────────────────────────────────────────
        panels: list[tuple[str, np.ndarray]] = [
            ("Original", image_rgb),
            ("LayerCAM (fused)", overlay_heatmap(image_pil_denorm, fused_cam)),
            ("Bone guidance", overlay_heatmap(image_pil_denorm, prompt_map)),
            (f"Bone foreground (P{int(args.cam_percentile)})", _make_foreground_panel(image_rgb, prompt_map, args.cam_percentile)),
            (
                f"{args.sam_prompt_mode} ({len(bone_components)} comps)",
                _make_component_prompts_panel(image_rgb, prompt_map, bone_components)
                if bone_components
                else _make_prompts_panel(image_rgb, prompt_map, point_prompts),
            ),
            (f"SAM ({sam_masks.shape[0]} masks)", _make_sam_panel(image_rgb, sam_masks)),
            ("Pseudo Mask", np.stack([pseudo_mask * 255] * 3, axis=-1).astype(np.uint8)),
        ]

        if seg_model is not None:
            with torch.no_grad():
                seg_logits = seg_model(image_tensor)
                seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
            panels.append(("U-Net Seg", overlay_heatmap(image_pil_denorm, seg_prob, alpha=0.35)))

        build_figure(panels, output_path)

    finally:
        layercam.close()


if __name__ == "__main__":
    main()
