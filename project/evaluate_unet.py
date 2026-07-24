from __future__ import annotations

"""Evaluate a trained U-Net against polygon ground truth on a locked split."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.factory import build_segmentation_dataset
from progress import should_disable_tqdm
from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    segmentation_metrics,
    subgroup_summaries,
    summarize_segmentation_rows,
)
from evaluation.frozen_test_guard import verify_frozen_test_config
from models.unet import architecture_name_from_metadata, build_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U-Net checkpoint against segmentation ground truth")
    parser.set_defaults(dataset="btxrd")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--frozen-config", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Immutable derived split manifest. Its assignments are authoritative for BTXRD.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Defaults to the checkpoint image_size, then SegmentationConfig.image_size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Segmentation threshold; defaults to the checkpoint decision_threshold (then 0.5).",
    )
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional validation-only threshold sweep. The primary output still uses --threshold; "
            "the sweep is written beside it and reports the threshold maximizing tumor-only mean Dice. "
            "Sweeping the test split is rejected to prevent test-set tuning."
        ),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Optional directory for one thresholded PNG prediction per evaluated image.",
    )
    parser.add_argument(
        "--qualitative-dir",
        type=Path,
        default=None,
        help="Optional directory for best/median/worst/failure overlay figures.",
    )
    parser.add_argument(
        "--qualitative-count",
        type=int,
        default=12,
        help="Maximum number of deterministic qualitative cases (minimum 3).",
    )
    return parser.parse_args()


def sample_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def rows_from_probabilities(
    probability_records: list[tuple[np.ndarray, np.ndarray, str]],
    metadata_by_name: dict[str, dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for probability, target, image_name in probability_records:
        metrics = segmentation_metrics(probability >= threshold, target > 0.5)
        metadata = metadata_by_name.get(image_name, {})
        tumor_type = int(metadata.get("tumor_type", 0) or 0)
        rows.append({
            "image_name": image_name,
            "group": "tumor" if metrics["gt_positive"] else "normal",
            "group_id": str(metadata.get("group_id", "")),
            "group_source": str(metadata.get("group_source", "")),
            "center": str(metadata.get("center", "")),
            "anatomy": str(metadata.get("anatomy", "")),
            "view": str(metadata.get("view", "")),
            "tumor_type": tumor_type,
            "tumor_type_name": (
                TUMOR_TYPE_CLASS_NAMES[tumor_type]
                if 0 <= tumor_type < len(TUMOR_TYPE_CLASS_NAMES) else "unknown"
            ),
            **metrics,
        })
    return rows


def save_prediction_masks(
    probability_records: list[tuple[np.ndarray, np.ndarray, str]],
    threshold: float,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for probability, _target, image_name in probability_records:
        path = output_dir / f"{Path(image_name).stem}.png"
        mask = (probability >= threshold).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(path)
        saved[image_name] = path
    return saved


def select_qualitative_rows(
    rows: list[dict[str, object]],
    limit: int,
) -> list[tuple[str, dict[str, object]]]:
    tumors = sorted(
        (row for row in rows if bool(row.get("gt_positive"))),
        key=lambda row: (float(row.get("dice", 0.0)), str(row.get("image_name", ""))),
    )
    if not tumors:
        return []
    anchors = [
        ("worst", tumors[0]),
        ("median", tumors[len(tumors) // 2]),
        ("best", tumors[-1]),
    ]
    complete_misses = [
        ("complete_miss", row)
        for row in tumors
        if not bool(row.get("predicted_positive"))
    ]
    normal_false_positives = [
        ("normal_false_positive", row)
        for row in sorted(
            (
                row
                for row in rows
                if not bool(row.get("gt_positive")) and bool(row.get("predicted_positive"))
            ),
            key=lambda row: (
                -float(row.get("pred_area_ratio", 0.0)),
                str(row.get("image_name", "")),
            ),
        )
    ]
    selected: list[tuple[str, dict[str, object]]] = []
    seen: set[str] = set()
    for role, row in anchors + complete_misses + normal_false_positives:
        image_name = str(row["image_name"])
        if image_name in seen:
            continue
        selected.append((role, row))
        seen.add(image_name)
        if len(selected) >= max(3, limit):
            break
    return selected


def save_qualitative_overlays(
    *,
    rows: list[dict[str, object]],
    probability_records: list[tuple[np.ndarray, np.ndarray, str]],
    images_dir: Path,
    image_size: int,
    threshold: float,
    output_dir: Path,
    limit: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_name = {
        image_name: (probability, target)
        for probability, target, image_name in probability_records
    }
    case_rows: list[dict[str, object]] = []
    for role, row in select_qualitative_rows(rows, limit):
        image_name = str(row["image_name"])
        probability, target = records_by_name[image_name]
        prediction = probability >= threshold
        original = Image.open(images_dir / image_name).convert("RGB").resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        overlay = np.asarray(original, dtype=np.float32).copy()
        gt = target > 0.5
        pred_only = prediction & ~gt
        gt_only = gt & ~prediction
        overlap = prediction & gt
        for region, color in (
            (pred_only, np.array([32, 220, 80], dtype=np.float32)),
            (gt_only, np.array([240, 45, 45], dtype=np.float32)),
            (overlap, np.array([255, 210, 35], dtype=np.float32)),
        ):
            overlay[region] = 0.45 * overlay[region] + 0.55 * color
        overlay_path = output_dir / f"{role}__{Path(image_name).stem}.png"
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8), mode="RGB").save(
            overlay_path
        )
        case_rows.append({
            "role": role,
            "image_name": image_name,
            "group": row.get("group"),
            "dice": row.get("dice"),
            "iou": row.get("iou"),
            "gt_area_ratio": row.get("gt_area_ratio"),
            "pred_area_ratio": row.get("pred_area_ratio"),
            "predicted_positive": row.get("predicted_positive"),
            "overlay": overlay_path.name,
            "legend": "green=prediction_only; red=GT_only; yellow=overlap",
        })
    manifest_path = output_dir / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(case_rows[0]) if case_rows else ["role", "image_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(case_rows))
    return manifest_path


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dataset = checkpoint.get("dataset")
    if checkpoint_dataset and checkpoint_dataset != args.dataset:
        raise ValueError(f"Checkpoint dataset={checkpoint_dataset!r}, requested dataset={args.dataset!r}")
    if args.split_manifest is not None:
        manifest_path = args.split_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
        checkpoint_manifest_hash = checkpoint.get("split_manifest_sha256")
        if not checkpoint_manifest_hash:
            raise ValueError(
                "Checkpoint has no split_manifest_sha256; refusing to evaluate it against "
                "a locked split manifest. Retrain or use a checkpoint with provenance."
            )
        actual_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if checkpoint_manifest_hash != actual_manifest_hash:
            raise ValueError(
                "Checkpoint split manifest hash does not match the requested evaluation manifest"
            )
    image_size = args.image_size or int(checkpoint.get("image_size", SegmentationConfig.image_size))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("decision_threshold", 0.5))
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Segmentation threshold must be in [0,1], got {threshold}")
    threshold_grid = sorted(set(args.threshold_grid or []))
    if any(not 0.0 <= value <= 1.0 for value in threshold_grid):
        raise ValueError("Every --threshold-grid value must be in [0,1]")
    if threshold_grid and args.split.lower() == "test":
        raise ValueError(
            "Threshold sweeping is validation-only; select and freeze a threshold before test evaluation."
        )
    if args.qualitative_count < 3:
        raise ValueError("--qualitative-count must be at least 3")
    if args.split.lower() == "test" and (
        args.prediction_dir is None or args.qualitative_dir is None
    ):
        raise ValueError(
            "Final test evaluation requires --prediction-dir and --qualitative-dir"
        )
    for output_dir in (args.prediction_dir, args.qualitative_dir):
        if output_dir is not None and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to reuse non-empty evaluation output directory: {output_dir}"
            )
    frozen_document = verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_checkpoint=args.checkpoint,
        checkpoint_any_of=("unet_checkpoint",),
        requested_threshold=threshold,
        requested_image_size=image_size,
        requested_stage="official_wsss_segmenter",
    )

    dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.split,
        image_size=image_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture_name = architecture_name_from_metadata(checkpoint.get("architecture"))
    model = build_segmentation_model(architecture_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    metadata_by_name = {str(sample["image_id"]): sample for sample in dataset.samples}
    probability_records: list[tuple[np.ndarray, np.ndarray, str]] = []

    with torch.no_grad():
        for images, targets, image_names in tqdm(
            loader,
            desc=f"evaluate-unet-{args.split}",
            disable=should_disable_tqdm(),
        ):
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            target_arrays = targets.numpy()
            for probability, target, image_name in zip(probabilities, target_arrays, image_names):
                probability_records.append((
                    probability[0].astype(np.float16, copy=False),
                    target[0].astype(np.uint8, copy=False),
                    str(image_name),
                ))

    rows = rows_from_probabilities(probability_records, metadata_by_name, threshold)
    prediction_paths = (
        save_prediction_masks(probability_records, threshold, args.prediction_dir)
        if args.prediction_dir is not None
        else {}
    )
    qualitative_manifest = None
    if args.qualitative_dir is not None:
        qualitative_manifest = save_qualitative_overlays(
            rows=rows,
            probability_records=probability_records,
            images_dir=dataset.images_dir,
            image_size=image_size,
            threshold=threshold,
            output_dir=args.qualitative_dir,
            limit=args.qualitative_count,
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "image_size": image_size,
        "threshold": threshold,
        "test_evaluated": args.split.lower() == "test",
        "frozen_config_sha256": (
            frozen_document.get("freeze_sha256") if frozen_document else None
        ),
        "boundary_distance_unit": "pixels on the resized evaluation grid",
        **summarize_segmentation_rows(rows),
    }
    bootstrap = bootstrap_group_confidence_intervals(
        rows,
        group_key="group_id",
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    subgroup_rows = subgroup_summaries(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(json_safe(rows))
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    threshold_sweep_path = None
    threshold_selection_path = None
    if threshold_grid:
        sweep_rows: list[dict[str, object]] = []
        for candidate_threshold in threshold_grid:
            candidate_summary = summarize_segmentation_rows(
                rows_from_probabilities(probability_records, metadata_by_name, candidate_threshold)
            )
            sweep_rows.append({"threshold": candidate_threshold, **candidate_summary})
        selection_key = lambda row: (
            float(row["mean_tumor_dice"]),
            float(row["normal_empty_prediction_rate"]),
            -abs(float(row["threshold"]) - 0.5),
        )
        selected = max(sweep_rows, key=selection_key)
        threshold_sweep_path = args.output_csv.with_name(args.output_csv.stem + "_threshold_sweep.csv")
        with threshold_sweep_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
            writer.writeheader()
            writer.writerows(json_safe(sweep_rows))
        threshold_selection_path = args.output_csv.with_name(
            args.output_csv.stem + "_threshold_selection.json"
        )
        threshold_selection_path.write_text(json.dumps(json_safe({
            "selection_split": args.split,
            "selection_rule": (
                "maximize mean_tumor_dice; then normal_empty_prediction_rate; "
                "then proximity to threshold 0.5"
            ),
            "selected": selected,
            "candidate_count": len(sweep_rows),
        }), indent=2) + "\n", encoding="utf-8")
    subgroup_path = args.output_csv.with_name(args.output_csv.stem + "_subgroups.csv")
    with subgroup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]) if subgroup_rows else ["subgroup_field"])
        writer.writeheader()
        writer.writerows(json_safe(subgroup_rows))
    bootstrap_path = args.output_csv.with_name(args.output_csv.stem + "_bootstrap.json")
    bootstrap_path.write_text(json.dumps(json_safe(bootstrap), indent=2) + "\n", encoding="utf-8")
    confusion_path = args.output_csv.with_name(args.output_csv.stem + "_pixel_confusion.json")
    confusion = {key: summary[key] for key in ("tp_pixels", "fp_pixels", "fn_pixels", "tn_pixels")}
    confusion_path.write_text(json.dumps(confusion, indent=2) + "\n", encoding="utf-8")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT.parent, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        git_commit, git_dirty = "unknown", None
    run_manifest = {
        "entrypoint": str(Path(__file__).resolve()),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "dataset": args.dataset,
        "split": args.split,
        "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
        "split_manifest_sha256": (
            hashlib.sha256(args.split_manifest.read_bytes()).hexdigest() if args.split_manifest else None
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "artifacts": {
            "per_image": str(args.output_csv.resolve()),
            "summary": str(output_json.resolve()),
            "subgroups": str(subgroup_path.resolve()),
            "bootstrap": str(bootstrap_path.resolve()),
            "pixel_confusion": str(confusion_path.resolve()),
            "threshold_sweep": str(threshold_sweep_path.resolve()) if threshold_sweep_path else None,
            "threshold_selection": (
                str(threshold_selection_path.resolve()) if threshold_selection_path else None
            ),
            "prediction_dir": (
                str(args.prediction_dir.resolve()) if prediction_paths else None
            ),
            "prediction_masks": len(prediction_paths),
            "qualitative_dir": (
                str(args.qualitative_dir.resolve()) if qualitative_manifest else None
            ),
            "qualitative_case_manifest": (
                str(qualitative_manifest.resolve()) if qualitative_manifest else None
            ),
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "test_evaluated": args.split.lower() == "test",
        "frozen_config": (
            str(args.frozen_config.resolve()) if args.frozen_config else None
        ),
        "frozen_config_sha256": (
            frozen_document.get("freeze_sha256") if frozen_document else None
        ),
    }
    run_manifest_path = args.output_csv.with_name(args.output_csv.stem + "_run_manifest.json")
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {args.output_csv}")
    print(f"Saved summary to {output_json}")
    if threshold_selection_path is not None:
        print(f"Saved validation threshold sweep and selection to {threshold_selection_path}")
    print(f"Saved subgroup/bootstrap/confusion/run-manifest artifacts next to {args.output_csv}")


if __name__ == "__main__":
    main()
