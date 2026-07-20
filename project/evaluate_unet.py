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
from models.unet import UNet


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
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_checkpoint=args.checkpoint,
        checkpoint_any_of=("unet_checkpoint", "supervised_unet_checkpoint"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
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
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    metadata_by_name = {str(sample["image_id"]): sample for sample in dataset.samples}
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for images, targets, image_names in tqdm(
            loader,
            desc=f"evaluate-unet-{args.split}",
            disable=should_disable_tqdm(),
        ):
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits).cpu() >= threshold).float()
            for pred, target, image_name in zip(predictions, targets, image_names):
                metrics = segmentation_metrics(
                    pred[0].numpy() > 0.5,
                    target[0].numpy() > 0.5,
                )
                metadata = metadata_by_name.get(str(image_name), {})
                tumor_type = int(metadata.get("tumor_type", 0) or 0)
                rows.append({
                    "image_name": str(image_name),
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

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "image_size": image_size,
        "threshold": threshold,
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
        },
    }
    run_manifest_path = args.output_csv.with_name(args.output_csv.stem + "_run_manifest.json")
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {args.output_csv}")
    print(f"Saved summary to {output_json}")
    print(f"Saved subgroup/bootstrap/confusion/run-manifest artifacts next to {args.output_csv}")


if __name__ == "__main__":
    main()
