from __future__ import annotations

"""Evaluate the image classifier and the tumor-vs-normal gate on a locked split."""

import argparse
import csv
import hashlib
import json
import math
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

from config import DEFAULT_DATASET, SUPPORTED_DATASETS
from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
from datasets.factory import build_classification_dataset
from evaluation.classification_metrics import (
    binary_average_precision,
    binary_auroc,
    binary_metrics,
    confusion_from_predictions,
    multiclass_summary,
)
from models.classifier import DenseNet121AnatomyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(dataset="btxrd")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gate-rule",
        choices=("argmax", "probability"),
        default="argmax",
        help="argmax matches generate_pseudo_masks.py for the 10-class classifier.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.gate_threshold <= 1.0:
        raise ValueError("--gate-threshold must be in [0,1]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("dataset") not in (None, args.dataset):
        raise ValueError("Classifier checkpoint is not for BTXRD")
    if args.split_manifest is not None:
        expected_hash = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
        if checkpoint.get("split_manifest_sha256") != expected_hash:
            raise ValueError("Classifier checkpoint split manifest hash mismatch")
    target_columns = list(checkpoint.get("target_columns", ["tumor_type"]))
    task = str(checkpoint.get("task", "single-label"))
    num_classes = int(checkpoint.get("num_classes", 10 if target_columns == ["tumor_type"] else len(target_columns)))
    image_size = int(args.image_size or checkpoint.get("image_size", 320))
    normalization = str(checkpoint.get("normalization", "imagenet"))
    dataset = build_classification_dataset(
        root=args.data_root,
        split=args.split,
        target_columns=target_columns,
        image_size=image_size,
        augment=False,
        normalization=normalization,
        split_manifest=args.split_manifest,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    image_names: list[str] = []
    with torch.no_grad():
        for images, targets, names in tqdm(loader, desc=f"classifier-{args.split}"):
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1) if task == "single-label" else torch.sigmoid(logits)
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            image_names.extend(str(name) for name in names)
    probs = np.concatenate(all_probabilities)
    raw_targets = np.concatenate(all_targets)
    true_class = raw_targets.astype(int).reshape(-1) if task == "single-label" else raw_targets[:, 0].astype(int)
    pred_class = probs.argmax(axis=1) if task == "single-label" else (probs[:, 0] >= 0.5).astype(int)
    gate_target = (true_class != 0).astype(int) if task == "single-label" else true_class
    gate_score = 1.0 - probs[:, 0] if task == "single-label" else probs[:, 0]
    if task == "single-label" and args.gate_rule == "argmax":
        gate_prediction = (pred_class != 0).astype(int)
    else:
        gate_prediction = (gate_score >= args.gate_threshold).astype(int)
    gate_counts = confusion_from_predictions(gate_target, gate_prediction)
    gate_summary = {
        "positive_class": "tumor",
        "decision_rule": args.gate_rule,
        "threshold": args.gate_threshold if args.gate_rule == "probability" else None,
        **gate_counts,
        **binary_metrics(gate_counts),
        "auroc": binary_auroc(gate_target, gate_score),
        "auprc": binary_average_precision(gate_target, gate_score),
    }

    confusion_size = num_classes if task == "single-label" else 2
    confusion = np.zeros((confusion_size, confusion_size), dtype=np.int64)
    for target, pred in zip(true_class, pred_class):
        confusion[int(target), int(pred)] += 1
    class_summary = multiclass_summary(confusion, TUMOR_TYPE_CLASS_NAMES) if task == "single-label" else None
    if task == "single-label":
        one_vs_rest_auroc = []
        one_vs_rest_auprc = []
        supports = []
        for class_index in range(num_classes):
            binary_target = (true_class == class_index).astype(int)
            one_vs_rest_auroc.append(binary_auroc(binary_target, probs[:, class_index]))
            one_vs_rest_auprc.append(binary_average_precision(binary_target, probs[:, class_index]))
            supports.append(int(binary_target.sum()))
        finite_auc = [value for value in one_vs_rest_auroc if math.isfinite(value)]
        finite_ap = [value for value in one_vs_rest_auprc if math.isfinite(value)]
        class_summary.update({
            "macro_ovr_auroc": float(np.mean(finite_auc)) if finite_auc else float("nan"),
            "macro_ovr_auprc": float(np.mean(finite_ap)) if finite_ap else float("nan"),
            "ovr_auroc": one_vs_rest_auroc,
            "ovr_auprc": one_vs_rest_auprc,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, image_name in enumerate(image_names):
        row = {
            "image_name": image_name,
            "true_class": int(true_class[index]),
            "predicted_class": int(pred_class[index]),
            "true_tumor": int(gate_target[index]),
            "tumor_probability": float(gate_score[index]),
            "predicted_tumor": int(gate_prediction[index]),
        }
        for class_index in range(probs.shape[1]):
            row[f"prob_class_{class_index}"] = float(probs[index, class_index])
        rows.append(row)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "multiclass_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *range(confusion_size)])
        for index, row in enumerate(confusion.tolist()):
            writer.writerow([index, *row])
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "task": task,
        "target_columns": target_columns,
        "images": len(rows),
        "image_size": image_size,
        "gate_tumor_vs_normal": gate_summary,
        "multiclass": class_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
    except Exception:
        git_commit = "unknown"
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({
            "git_commit": git_commit,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
            "split_manifest_sha256": (
                hashlib.sha256(args.split_manifest.read_bytes()).hexdigest() if args.split_manifest else None
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
