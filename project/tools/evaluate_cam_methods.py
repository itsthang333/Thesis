from __future__ import annotations

"""Validation-only CAM method diagnostic for BTXRD.

This script reads polygon masks solely to score localization maps.  It never
passes a polygon, bbox, or derived mask to pseudo-mask generation.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
from datasets.factory import build_classification_dataset, build_segmentation_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM


def _norm(cam: torch.Tensor, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    cam = F.relu(cam).float()
    if cam.ndim == 4:
        cam = cam[:, 0]
    if cam.ndim == 2:
        cam = cam[None, None]
    elif cam.ndim == 3:
        cam = cam[:, None]
    else:
        raise ValueError(f"expected [H,W] or [B,H,W], got {tuple(cam.shape)}")
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
    flat = cam.flatten()
    cam = (cam - flat.min()) / (flat.max() - flat.min() + 1e-8)
    return cam.detach().cpu().numpy().astype(np.float32)


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / max(denom, 1e-8)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / max(union, 1e-8)


def _largest(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _threshold_metrics(cam: np.ndarray, gt: np.ndarray, percentiles: tuple[float, ...]) -> dict[str, float]:
    best = (-1.0, 0.0, 0.0)
    for p in percentiles:
        threshold = float(np.percentile(cam, p))
        raw = cam >= threshold
        for suffix, pred in (("", raw), ("_largest", _largest(raw))):
            dice = _dice(pred, gt)
            iou = _iou(pred, gt)
            if dice > best[0]:
                best = (dice, iou, p)
    return {"best_dice": best[0], "best_iou": best[1], "best_percentile": best[2]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--binary-checkpoint", type=Path, default=None,
                        help="Optional one-logit tumor checkpoint for binary CAM comparison.")
    parser.add_argument("--ensemble-checkpoint", type=Path, default=None,
                        help="Optional second 10-class checkpoint for mean contrastive CAM diagnostics.")
    parser.add_argument("--multiscale-sizes", type=str, default="",
                        help="Optional comma-separated input sizes for a contrastive CAM ensemble, "
                             "e.g. '224,256,288'. Each map is normalized before averaging.")
    parser.add_argument("--tile-grid", type=int, default=0,
                        help="Optional square tile grid for contrastive CAM (e.g. 2 -> 2x2 crops); "
                             "tile maps are resized back and max-pooled as a small-lesion diagnostic.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    multiscale_sizes = tuple(
        int(value.strip()) for value in args.multiscale_sizes.split(",") if value.strip()
    )
    if any(size < 32 for size in multiscale_sizes):
        raise ValueError("--multiscale-sizes values must be >= 32")
    if args.tile_grid < 0 or args.tile_grid > 4:
        raise ValueError("--tile-grid must be between 0 and 4")
    state = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(state.get("num_classes", 10))
    model = DenseNet121AnatomyClassifier(
        num_classes=num_classes, pretrained=False,
        anatomy_num_classes=int(state.get("anatomy_num_classes", 0)),
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    binary_model = None
    binary_layercam = None
    if args.binary_checkpoint is not None:
        binary_state = torch.load(args.binary_checkpoint, map_location="cpu")
        binary_model = DenseNet121AnatomyClassifier(
            num_classes=1, pretrained=False,
            anatomy_num_classes=int(binary_state.get("anatomy_num_classes", 0)),
        )
        binary_model.load_state_dict(binary_state["model_state_dict"], strict=True)
        binary_model.to(device).eval()
    ensemble_model = None
    ensemble_layercam = None
    if args.ensemble_checkpoint is not None:
        ensemble_state = torch.load(args.ensemble_checkpoint, map_location="cpu")
        ensemble_num_classes = int(ensemble_state.get("num_classes", num_classes))
        if ensemble_num_classes != num_classes:
            raise ValueError("--ensemble-checkpoint must have the same class count as --checkpoint")
        ensemble_model = DenseNet121AnatomyClassifier(
            num_classes=num_classes, pretrained=False,
            anatomy_num_classes=int(ensemble_state.get("anatomy_num_classes", 0)),
        )
        ensemble_model.load_state_dict(ensemble_state["model_state_dict"], strict=True)
        ensemble_model.to(device).eval()

    cls = build_classification_dataset(
        "btxrd", args.dataset_root, "val", ["tumor_type"], args.image_size,
        normalization=state.get("normalization", "imagenet"),
    )
    seg = build_segmentation_dataset("btxrd", args.dataset_root, "val", args.image_size)
    if args.image_list is None:
        names = {str(sample["image_id"]) for sample in cls.samples}
    else:
        names = {
            line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    cls_by_name = {str(s["image_id"]): s for s in cls.samples if str(s["image_id"]) in names}
    seg_by_name = {str(seg.samples[i]["image_id"]): seg[i][1][0].numpy() > 0.5
                   for i in range(len(seg)) if str(seg.samples[i]["image_id"]) in names}

    layercam = LayerCAM(model, device=device)
    if binary_model is not None:
        binary_layercam = LayerCAM(binary_model, device=device)
    if ensemble_model is not None:
        ensemble_layercam = LayerCAM(ensemble_model, device=device)
    rows: list[dict[str, object]] = []
    percentiles = (70.0, 75.0, 80.0, 85.0, 90.0, 92.0, 95.0, 97.0, 98.0, 99.0)
    try:
        for image_name, sample in sorted(cls_by_name.items()):
            image_tensor, target, _ = cls[cls.samples.index(sample)]
            image_tensor = image_tensor.unsqueeze(0).to(device)
            target_class = int(target.item())
            gt = seg_by_name[image_name]
            cam_size = tuple(int(value) for value in image_tensor.shape[-2:])

            # Standard class activation map: classifier weight dot final feature map.
            with torch.no_grad():
                features = model.forward_features(image_tensor)
                logits = model(image_tensor)
                pred_class = int(logits.argmax(dim=1).item())
                cam_weight = model.classifier.weight[target_class]
                direct = _norm((features * cam_weight[None, :, None, None]).sum(dim=1), cam_size)

            # Current LayerCAM and its individual dense-block maps.
            out = layercam.cam_for_class(image_tensor, target_class)
            cams = {"layercam_fused": out.cam[0].detach().cpu().numpy()}
            for block_index, block_name in enumerate(("denseblock2", "denseblock3", "denseblock4")):
                cams["layercam_" + block_name] = layercam._compute_layer_cam(
                    layercam._states[block_index], image_tensor.shape[-2:]
                )[0, 0].detach().cpu().numpy()
            # Diagnostic variant: retain both positive and negative gradient
            # evidence.  It is not used by the production pipeline unless an
            # explicit source change is later justified by this comparison.
            abs_layer_cams = []
            for state in layercam._states:
                assert state.activations is not None and state.gradients is not None
                raw = (state.activations * state.gradients.abs()).sum(dim=1, keepdim=True)
                abs_layer_cams.append(_norm(raw, cam_size)[...])
            cams["absgrad_b2_b3_b4_mean"] = sum(abs_layer_cams) / len(abs_layer_cams)
            cams["absgrad_fixed_fusion"] = (
                0.2 * abs_layer_cams[0] + 0.3 * abs_layer_cams[1] + 0.5 * abs_layer_cams[2]
            )
            # Layer-wise combinations are diagnostic only.  Each component is
            # independently normalized by LayerCAM before combination so the
            # comparison is not dominated by activation scale.
            cams["layercam_b2_b3_mean"] = (
                0.5 * cams["layercam_denseblock2"] + 0.5 * cams["layercam_denseblock3"]
            )
            cams["layercam_b2_b3_max"] = np.maximum(
                cams["layercam_denseblock2"], cams["layercam_denseblock3"]
            )
            cams["layercam_b2_b3_b4_mean"] = (
                cams["layercam_denseblock2"] + cams["layercam_denseblock3"] + cams["layercam_denseblock4"]
            ) / 3.0

            # Grad-CAM on denseblock4 (global-average-pooled positive gradient).
            state4 = layercam._states[2]
            assert state4.activations is not None and state4.gradients is not None
            weights = state4.gradients.mean(dim=(2, 3), keepdim=True)
            gradcam = _norm((state4.activations * weights).sum(dim=1), cam_size)
            cams["gradcam_denseblock4"] = gradcam
            cams["direct_cam"] = direct
            # Contrastive LayerCAM: evidence for the target class relative to
            # the normal (class 0) logit.  This is diagnostic only; it can
            # suppress generic bone/background evidence that is shared by all
            # classes while retaining lesion-specific evidence.
            if target_class != 0:
                contrast_out = layercam.cam_for_class_contrast(
                    image_tensor, target_class, reference_index=0
                )
                contrast_cam = contrast_out.cam[0].detach().cpu().numpy()
                cams["layercam_contrast"] = contrast_cam
                cams["layercam_contrast_mean"] = 0.5 * (
                    cams["layercam_fused"] + contrast_cam
                )
                contrast_layer_cams = []
                for contrast_state in layercam._states:
                    contrast_layer_cams.append(
                        _norm(layercam._compute_layer_cam(contrast_state, image_tensor.shape[-2:]), cam_size)
                    )
                cams["layercam_contrast_denseblock2"] = contrast_layer_cams[0]
                cams["layercam_contrast_denseblock3"] = contrast_layer_cams[1]
                cams["layercam_contrast_denseblock4"] = contrast_layer_cams[2]
                cams["layercam_contrast_b2_b3_mean"] = 0.5 * (
                    contrast_layer_cams[0] + contrast_layer_cams[1]
                )
                # Class-agnostic tumor evidence.  This is a validation-only
                # diagnostic to test whether per-class confusion is limiting
                # localization; it remains entirely image-level (no polygon
                # or segmentation target is used to construct the map).
                union_contrast_out = layercam.cam_for_tumor_union_contrast(image_tensor)
                cams["layercam_union_contrast"] = (
                    union_contrast_out.cam[0].detach().cpu().numpy()
                )
                if multiscale_sizes:
                    multiscale_maps = [contrast_cam]
                    for scale in multiscale_sizes:
                        if scale == image_tensor.shape[-1]:
                            continue
                        scaled_tensor = F.interpolate(
                            image_tensor, size=(scale, scale), mode="bilinear", align_corners=False
                        )
                        scaled_out = layercam.cam_for_class_contrast(
                            scaled_tensor, target_class, reference_index=0
                        )
                        multiscale_maps.append(_norm(scaled_out.cam, cam_size))
                    cams["layercam_contrast_multiscale"] = np.mean(
                        np.stack(multiscale_maps, axis=0), axis=0
                    ).astype(np.float32)
                if args.tile_grid > 1:
                    grid = args.tile_grid
                    height, width = image_tensor.shape[-2:]
                    tiled_cam = np.zeros((height, width), dtype=np.float32)
                    for row_index in range(grid):
                        y0 = int(round(row_index * height / grid))
                        y1 = int(round((row_index + 1) * height / grid))
                        for col_index in range(grid):
                            x0 = int(round(col_index * width / grid))
                            x1 = int(round((col_index + 1) * width / grid))
                            crop = image_tensor[:, :, y0:y1, x0:x1]
                            crop_out = layercam.cam_for_class_contrast(
                                F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False),
                                target_class, reference_index=0,
                            )
                            crop_cam = crop_out.cam[0]
                            crop_cam = F.interpolate(
                                crop_cam[None, None], size=(y1 - y0, x1 - x0),
                                mode="bilinear", align_corners=False,
                            )[0, 0]
                            crop_cam = crop_cam.detach().cpu().numpy().astype(np.float32)
                            crop_cam = (crop_cam - float(crop_cam.min())) / (
                                float(crop_cam.max()) - float(crop_cam.min()) + 1e-8
                            )
                            tiled_cam[y0:y1, x0:x1] = np.maximum(
                                tiled_cam[y0:y1, x0:x1], crop_cam
                            )
                    cams["layercam_contrast_tiles_max"] = tiled_cam
            if pred_class != target_class:
                # Run the alternative class only after all target-class maps
                # above have been captured; hooks otherwise contain the
                # predicted-class gradients and would mislabel Grad-CAM.
                pred_out = layercam.cam_for_class(image_tensor, pred_class)
                pred_cam = pred_out.cam[0].detach().cpu().numpy()
                cams["layercam_predicted"] = pred_cam
                cams["layercam_target_pred_mean"] = 0.5 * (cams["layercam_fused"] + pred_cam)

            if ensemble_layercam is not None and target_class != 0:
                ensemble_out = ensemble_layercam.cam_for_class_contrast(
                    image_tensor, target_class, reference_index=0
                )
                ensemble_cam = ensemble_out.cam[0].detach().cpu().numpy()
                if "layercam_contrast" in cams:
                    cams["layercam_contrast_ensemble_mean"] = 0.5 * (
                        cams["layercam_contrast"] + ensemble_cam
                    )
                else:
                    cams["layercam_contrast_ensemble_mean"] = ensemble_cam

            row: dict[str, object] = {
                "image_name": image_name,
                "tumor_type": TUMOR_TYPE_CLASS_NAMES[target_class],
                "pred_class": pred_class,
            }
            if binary_layercam is not None and binary_model is not None:
                binary_out = binary_layercam.cam_for_class(image_tensor, 0)
                binary_cam = binary_out.cam[0].detach().cpu().numpy()
                row["binary_layercam_best_dice"] = _threshold_metrics(binary_cam, gt, percentiles)["best_dice"]
                cams["binary_layercam"] = binary_cam
                cams["binary_target_mean"] = 0.5 * (binary_cam + cams["layercam_fused"])
                cams["binary_target_max"] = np.maximum(binary_cam, cams["layercam_fused"])
                if "layercam_contrast" in cams:
                    cams["binary_contrast_mean"] = 0.5 * (
                        binary_cam + cams["layercam_contrast"]
                    )
                    cams["binary_contrast_max"] = np.maximum(
                        binary_cam, cams["layercam_contrast"]
                    )
            for method, cam in cams.items():
                metrics = _threshold_metrics(cam, gt, percentiles)
                row[method + "_best_dice"] = metrics["best_dice"]
                row[method + "_best_iou"] = metrics["best_iou"]
                row[method + "_best_percentile"] = metrics["best_percentile"]
            rows.append(row)
            print(image_name, row["tumor_type"], "pred", pred_class,
                  "direct", f"{row['direct_cam_best_dice']:.3f}",
                  "layer", f"{row['layercam_fused_best_dice']:.3f}",
                  "grad", f"{row['gradcam_denseblock4_best_dice']:.3f}")
    finally:
        layercam.close()
        if binary_layercam is not None:
            binary_layercam.close()
        if ensemble_layercam is not None:
            ensemble_layercam.close()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for key in fieldnames:
        if key.endswith("_best_dice"):
            vals = [float(row[key]) for row in rows if key in row]
            print("MEAN", key, f"{sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    main()
