from __future__ import annotations

"""Visualize the full WSSS pipeline for a single image.

Saves 12 separate stage images plus one composite strip:
  01_input.png
  02_layercam.png
  03_bone_likelihood.png
  04_bone_support.png
  05_prompt_points.png
  06_all_sam_masks.png
  07_ranked_masks.png
  08_selected_masks.png
  09_pseudo_mask.png
  10_gt_mask.png          (only when --gt-mask-path provided)
  11_overlay_prediction.png
  12_overlay_gt.png       (only when --gt-mask-path provided)
  pipeline_strip.png      (composite of all available panels)

Usage:
    python visualize_pipeline.py \
        --image-path /path/to/image.bmp \
        --classifier-checkpoint outputs/classifier/best_classifier.pt \
        --sam-checkpoint /drive/MyDrive/sam_vit_b_01ec64.pth \
        [--segmentation-checkpoint outputs/segmentation/best_unet.pt] \
        [--gt-mask-path /path/to/gt.png] \
        [--output-dir outputs/viz/<stem>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
from pseudo.mask_selection import select_and_fuse_masks, score_masks, score_masks_detailed, save_ranking_csv
from pseudo.morphology import morphological_refinement
from pseudo.visualization import overlay_heatmap, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full WSSS pipeline for one image")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--segmentation-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--gt-mask-path", type=Path, default=None,
                        help="Optional ground-truth mask PNG for comparison panels")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for output images (default: outputs/viz/<stem>)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--cam-percentile", type=float, default=90.0)
    parser.add_argument("--max-points", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--mask-score-threshold", type=float, default=0.4)
    parser.add_argument("--selection-method", type=str, default="bone_hybrid",
                        choices=["mean", "sum", "mean_area", "coverage", "hybrid", "bone_hybrid"])
    parser.add_argument("--fusion-topk", type=int, default=3)
    parser.add_argument("--closing-kernel", type=int, default=3)
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
    parser.add_argument("--save-ranking-csv", action="store_true",
                        help="Save per-candidate ranking breakdown CSV")
    parser.add_argument("--debug", action="store_true",
                        help="Alias for --save-ranking-csv; saves per-candidate ranking breakdown")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def _jet_rgb(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _blend(image_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(
        (1 - alpha) * image_rgb.astype(np.float32) + alpha * overlay_rgb.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def _overlay_heatmap_rgb(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return _blend(image_rgb, _jet_rgb(heatmap), alpha)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    v = (mask.astype(bool).astype(np.uint8) * 255)
    return np.stack([v, v, v], axis=-1)


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(0, 200, 80), alpha: float = 0.55) -> np.ndarray:
    out = image_rgb.astype(np.float32)
    m = mask.astype(bool)
    tint = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1 - alpha) + tint * alpha
    return out.clip(0, 255).astype(np.uint8)


def _add_label(img: np.ndarray, label: str, font_size: int = 16) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    bar_h = font_size + 6
    draw.rectangle([0, 0, pil.width, bar_h], fill=(15, 15, 15))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 3), label, fill=(240, 240, 240), font=font)
    return np.array(pil)


def _save(img_rgb: np.ndarray, path: Path, label: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _add_label(img_rgb, label or path.stem) if label is not False else img_rgb
    Image.fromarray(out).save(path)


def _make_prompt_panel(
    image_rgb: np.ndarray,
    bone_cam: np.ndarray,
    point_prompts: list[tuple[int, int]],
    components=None,
) -> np.ndarray:
    jet = _jet_rgb(bone_cam)
    blended = _blend(image_rgb, jet, alpha=0.45)
    pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil)
    h, w = bone_cam.shape
    radius = max(6, min(h, w) // 60)
    for i, (r, c) in enumerate(point_prompts):
        draw.ellipse([c - radius, r - radius, c + radius, r + radius],
                     fill=(255, 50, 50), outline=(255, 255, 255), width=2)
        draw.text((c + radius + 2, r - radius), str(i + 1), fill=(255, 230, 0))
    if components:
        for idx, comp in enumerate(components):
            x0, y0, x1, y1 = comp.bbox
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 100), width=2)
            draw.text((x0 + 2, y0 + 2), f"C{idx}", fill=(0, 255, 100))
    return np.array(pil)


def _make_all_sam_panel(image_rgb: np.ndarray, sam_masks: np.ndarray) -> np.ndarray:
    result = image_rgb.astype(np.float32)
    palette = [
        (255, 80, 80), (80, 200, 80), (80, 120, 255),
        (255, 220, 60), (200, 80, 255), (60, 230, 230),
        (255, 150, 50), (150, 255, 150), (200, 150, 255),
    ]
    for i in range(sam_masks.shape[0]):
        m = sam_masks[i].astype(bool)
        c = np.array(palette[i % len(palette)], dtype=np.float32)
        result[m] = result[m] * 0.35 + c * 0.65
    return result.clip(0, 255).astype(np.uint8)


def _make_ranked_panel(
    image_rgb: np.ndarray,
    sam_masks: np.ndarray,
    scores: np.ndarray,
    top_n: int = 6,
) -> np.ndarray:
    order = np.argsort(scores)[::-1][:top_n]
    result = image_rgb.astype(np.float32)
    palette = [(255, 80, 80), (255, 160, 40), (255, 230, 40),
               (100, 220, 80), (60, 180, 255), (180, 80, 255)]
    pil_base = None
    for rank, idx in enumerate(order):
        m = sam_masks[idx].astype(bool)
        c = np.array(palette[rank % len(palette)], dtype=np.float32)
        result[m] = result[m] * 0.35 + c * 0.65
    pil = Image.fromarray(result.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for rank, idx in enumerate(order):
        rows, cols = np.where(sam_masks[idx].astype(bool))
        if rows.size == 0:
            continue
        cy, cx = int(rows.mean()), int(cols.mean())
        draw.text((cx - 10, cy - 6), f"#{rank+1} {scores[idx]:.2f}", fill=(255, 255, 255), font=font)
    return np.array(pil)


def _make_selected_panel(image_rgb: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    return _overlay_mask(image_rgb, selected_mask, color=(50, 200, 100), alpha=0.55)


def _build_strip(panels: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    labeled = [_add_label(img, title) for title, img in panels]
    total_w = sum(p.shape[1] for p in labeled) + 4 * (len(labeled) - 1)
    max_h = max(p.shape[0] for p in labeled)
    canvas = np.full((max_h, total_w, 3), 20, dtype=np.uint8)
    x = 0
    for p in labeled:
        h, w = p.shape[:2]
        canvas[:h, x:x + w] = p
        x += w + 4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    print(f"Saved strip: {output_path}")


def classifier_class_weights(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "single-label":
        return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    stem = args.image_path.stem
    out_dir = args.output_dir or (ROOT / "outputs" / "viz" / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # ── Load image ───────────────────────────────────────────────────────
        transform = make_classification_transform(args.image_size, augment=False)
        image_pil = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        image_pil_denorm = tensor_to_pil(image_tensor[0].detach().cpu())
        image_rgb = np.array(image_pil_denorm, dtype=np.uint8)

        # 01 — Input
        _save(image_rgb, out_dir / "01_input.png", "01 Input")

        # ── Stage 1: classifier ──────────────────────────────────────────────
        with torch.no_grad():
            logits = classifier(image_tensor)
            class_weights = classifier_class_weights(logits, classifier_task)

        active_labels = [target_columns[i] for i, w in enumerate(class_weights) if w >= args.confidence_threshold]
        print(f"Active classes: {active_labels} | scores: {class_weights.round(3)}")

        # ── Stage 2: LayerCAM ────────────────────────────────────────────────
        fused_cam, per_class_cams, active_indices = generate_fused_cam(
            layercam, image_tensor,
            class_weights=class_weights,
            confidence_threshold=args.confidence_threshold,
        )

        # 02 — LayerCAM
        _save(_overlay_heatmap_rgb(image_rgb, fused_cam, alpha=0.55),
              out_dir / "02_layercam.png", "02 LayerCAM (fused)")

        # ── Stage 3: Bone morphology ─────────────────────────────────────────
        bone_likelihood = None
        bone_support = None
        bone_components = []
        prompt_map = fused_cam

        if not args.disable_bone_morphology:
            if args.morphology_fusion_mode == "components":
                active_weights = [float(class_weights[i]) for i in active_indices]
                bone_likelihood, bone_support, bone_components = build_class_conditioned_components(
                    image_rgb, per_class_cams, active_weights,
                    seed_percentile=args.bone_seed_percentile,
                    support_percentile=args.bone_support_percentile,
                    min_component_area=max(20, args.min_component_area // 2),
                    max_components=args.max_bone_components,
                    points_per_component=args.points_per_component,
                    bbox_padding_ratio=args.bbox_padding_ratio,
                )
            else:
                bone_likelihood, bone_support = build_bone_guidance(
                    image_rgb, fused_cam,
                    seed_percentile=args.bone_seed_percentile,
                    support_percentile=args.bone_support_percentile,
                    min_component_area=max(20, args.min_component_area // 2),
                )
            prompt_map = fuse_cam_with_bone_guidance(fused_cam, bone_likelihood, bone_support)

        # 03 — Bone likelihood
        if bone_likelihood is not None:
            _save(_overlay_heatmap_rgb(image_rgb, bone_likelihood, alpha=0.6),
                  out_dir / "03_bone_likelihood.png", "03 Bone Likelihood")
        else:
            _save(_overlay_heatmap_rgb(image_rgb, fused_cam, alpha=0.55),
                  out_dir / "03_bone_likelihood.png", "03 CAM (no bone guidance)")

        # 04 — Bone support
        if bone_support is not None:
            _save(_overlay_mask(image_rgb, bone_support, color=(0, 180, 255), alpha=0.5),
                  out_dir / "04_bone_support.png", "04 Bone Support")
        else:
            _save(image_rgb.copy(), out_dir / "04_bone_support.png", "04 Bone Support (none)")

        # ── Stage 3: Prompt extraction ───────────────────────────────────────
        point_prompts = [p for comp in bone_components for p in comp.positive_points]
        if not point_prompts:
            point_prompts = extract_point_prompts(
                prompt_map,
                cam_percentile=args.cam_percentile,
                max_points=args.max_points,
                min_component_area=args.min_component_area,
                support_mask=bone_support,
            )
        print(f"Prompt points: {len(point_prompts)}")

        # 05 — Prompt points
        _save(_make_prompt_panel(image_rgb, prompt_map, point_prompts, bone_components),
              out_dir / "05_prompt_points.png",
              f"05 Prompt Points ({len(point_prompts)}, {len(bone_components)} comps)")

        # ── Stage 4: SAM ─────────────────────────────────────────────────────
        component_ids = None
        if bone_components:
            sam_masks, sam_scores, component_ids = sam_predictor.predict_from_components(
                image_rgb, bone_components,
                prompt_mode=args.sam_prompt_mode,
                multimask_output=not args.sam_single_mask,
                negative_points_per_component=args.negative_points_per_component,
            )
        else:
            sam_masks, sam_scores = sam_predictor.predict_from_points(image_rgb, point_prompts)

        print(f"SAM masks: {sam_masks.shape[0]} | scores: {sam_scores.round(3)}")
        if sam_masks.shape[0] == 0:
            print("[WARNING] SAM returned 0 masks. Falling back to CAM foreground.")
            thresh = float(np.percentile(fused_cam, args.cam_percentile))
            sam_masks = (fused_cam > thresh).astype(np.uint8)[np.newaxis]  # [1, H, W]
            sam_scores = np.array([1.0], dtype=np.float32)
            component_ids = None

        # 06 — All SAM masks
        _save(_make_all_sam_panel(image_rgb, sam_masks),
              out_dir / "06_all_sam_masks.png",
              f"06 All SAM Masks ({sam_masks.shape[0]})")

        # ── Stage 5: Mask scoring ────────────────────────────────────────────
        candidate_scores = score_masks(
            sam_masks, fused_cam,
            method=args.selection_method,
            bone_likelihood=bone_likelihood,
            bone_support=bone_support,
            sam_scores=sam_scores,
        )

        # 07 — Ranked masks (top-6 by score, colour-coded)
        _save(_make_ranked_panel(image_rgb, sam_masks, candidate_scores, top_n=6),
              out_dir / "07_ranked_masks.png",
              f"07 Ranked Masks ({args.selection_method})")

        # Save ranking CSV
        if args.save_ranking_csv or args.debug:
            ranking_rows = score_masks_detailed(
                sam_masks, fused_cam,
                method=args.selection_method,
                bone_likelihood=bone_likelihood,
                bone_support=bone_support,
                sam_scores=sam_scores,
                component_ids=component_ids,
            )
            save_ranking_csv(ranking_rows, out_dir / "ranking.csv")
            print(f"Saved ranking CSV: {out_dir / 'ranking.csv'}")

        # ── Stage 5: Mask selection + fusion ────────────────────────────────
        component_stack = (
            np.stack([c.mask for c in bone_components]) if bone_components else None
        )
        refined = select_and_fuse_masks(
            sam_masks, fused_cam,
            mask_score_threshold=args.mask_score_threshold,
            selection_method=args.selection_method,
            fusion_topk=args.fusion_topk,
            bone_likelihood=bone_likelihood,
            bone_support=bone_support,
            sam_scores=sam_scores,
            component_ids=component_ids,
            component_masks=component_stack,
            best_per_component=component_ids is not None,
        )

        # 08 — Selected (fused) masks before morphology
        _save(_make_selected_panel(image_rgb, refined),
              out_dir / "08_selected_masks.png", "08 Selected Masks (pre-morph)")

        # ── Stage 6: Morphological refinement ───────────────────────────────
        pseudo_mask = morphological_refinement(
            refined,
            closing_kernel=args.closing_kernel,
            opening_kernel=args.opening_kernel,
            min_size=args.min_size,
            guidance_map=bone_likelihood,
            max_hole_area=args.max_hole_area,
        )

        # 09 — Pseudo mask
        _save(_mask_to_rgb(pseudo_mask), out_dir / "09_pseudo_mask.png", "09 Pseudo Mask")

        # 10 — GT mask (optional)
        gt_mask = None
        if args.gt_mask_path and args.gt_mask_path.exists():
            gt_pil = Image.open(args.gt_mask_path).convert("L").resize(
                (args.image_size, args.image_size), Image.NEAREST
            )
            gt_mask = (np.array(gt_pil) > 127).astype(np.uint8)
            _save(_mask_to_rgb(gt_mask), out_dir / "10_gt_mask.png", "10 GT Mask")

        # 11 — Overlay prediction on image
        _save(_overlay_mask(image_rgb, pseudo_mask, color=(50, 200, 100), alpha=0.5),
              out_dir / "11_overlay_prediction.png", "11 Prediction Overlay")

        # 12 — Overlay GT on image (optional)
        if gt_mask is not None:
            _save(_overlay_mask(image_rgb, gt_mask, color=(255, 100, 50), alpha=0.5),
                  out_dir / "12_overlay_gt.png", "12 GT Overlay")

        # ── U-Net overlay (bonus) ────────────────────────────────────────────
        seg_mask = None
        if seg_model is not None:
            with torch.no_grad():
                seg_logits = seg_model(image_tensor)
                seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
                seg_mask = (seg_prob >= 0.5).astype(np.uint8)
            _save(_overlay_mask(image_rgb, seg_mask, color=(80, 160, 255), alpha=0.5),
                  out_dir / "13_unet_prediction.png", "13 U-Net Prediction")

        # ── Build composite strip ────────────────────────────────────────────
        panels: list[tuple[str, np.ndarray]] = [
            ("01 Input", image_rgb),
            ("02 LayerCAM", _overlay_heatmap_rgb(image_rgb, fused_cam, alpha=0.55)),
        ]
        if bone_likelihood is not None:
            panels.append(("03 Bone Likelihood", _overlay_heatmap_rgb(image_rgb, bone_likelihood, alpha=0.6)))
        if bone_support is not None:
            panels.append(("04 Bone Support", _overlay_mask(image_rgb, bone_support, color=(0, 180, 255), alpha=0.5)))
        panels += [
            ("05 Prompts", _make_prompt_panel(image_rgb, prompt_map, point_prompts, bone_components)),
            (f"06 SAM ({sam_masks.shape[0]})", _make_all_sam_panel(image_rgb, sam_masks)),
            ("07 Ranked", _make_ranked_panel(image_rgb, sam_masks, candidate_scores, top_n=6)),
            ("08 Selected", _make_selected_panel(image_rgb, refined)),
            ("09 Pseudo", _overlay_mask(image_rgb, pseudo_mask, color=(50, 200, 100), alpha=0.5)),
        ]
        if gt_mask is not None:
            panels.append(("10 GT", _overlay_mask(image_rgb, gt_mask, color=(255, 100, 50), alpha=0.5)))
        if seg_mask is not None:
            panels.append(("11 U-Net", _overlay_mask(image_rgb, seg_mask, color=(80, 160, 255), alpha=0.5)))

        _build_strip(panels, out_dir / "pipeline_strip.png")
        print(f"\nAll outputs saved to {out_dir}")

    finally:
        layercam.close()


if __name__ == "__main__":
    main()
