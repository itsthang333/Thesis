from __future__ import annotations

"""Label-safe classifier evaluation for G4 E1 (binary vs ten-class targets)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    prediction = (probability >= 0.5).astype(np.int64)
    counts = _binary_confusion(y_true, prediction)
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_div(tp * tn - fp * fn, mcc_denominator)

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_values = probability[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    auroc = _safe_div(float(ranks[y_true == 1].sum()) - positives * (positives + 1) / 2, positives * negatives)

    descending = np.argsort(-probability, kind="mergesort")
    sorted_true = y_true[descending]
    cumulative_tp = np.cumsum(sorted_true == 1)
    positive_positions = np.flatnonzero(sorted_true == 1)
    average_precision = (
        float(np.mean(cumulative_tp[positive_positions] / (positive_positions + 1)))
        if positives else 0.0
    )

    ece = 0.0
    bins: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        include = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 14 else probability < edges[index + 1]
        )
        n = int(include.sum())
        if not n:
            continue
        confidence = float(probability[include].mean())
        prevalence = float(y_true[include].mean())
        ece += n / len(y_true) * abs(confidence - prevalence)
        bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "n": n, "confidence": confidence, "prevalence": prevalence})
    return {
        "n": int(len(y_true)),
        "positive": positives,
        "negative": negatives,
        "threshold": 0.5,
        "confusion": counts,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "sensitivity_recall": recall,
        "specificity": specificity,
        "f1": f1,
        "matthews_correlation_coefficient": mcc,
        "auroc": auroc,
        "average_precision_auprc": average_precision,
        "brier_score": float(np.mean((probability - y_true) ** 2)),
        "negative_log_likelihood": float(
            -np.mean(
                y_true * np.log(np.clip(probability, 1e-7, 1.0))
                + (1 - y_true) * np.log(np.clip(1.0 - probability, 1e-7, 1.0))
            )
        ),
        "ece_15_equal_width": float(ece),
        "calibration_bins": bins,
    }


def _multiclass_metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: int,
) -> dict[str, object]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, pred in zip(y_true, predicted):
        confusion[int(target), int(pred)] += 1
    rows = []
    for class_index in range(classes):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        one_vs_rest = _binary_metrics(
            (y_true == class_index).astype(np.int64),
            probabilities[:, class_index],
        )
        rows.append({
            "class_index": class_index,
            "class_name": TUMOR_TYPE_CLASS_NAMES[class_index],
            "support": int(confusion[class_index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "one_vs_rest_auroc": one_vs_rest["auroc"],
            "one_vs_rest_average_precision": one_vs_rest["average_precision_auprc"],
        })
    one_hot = np.eye(classes, dtype=np.float64)[y_true]
    supports = np.asarray([row["support"] for row in rows], dtype=np.float64)
    return {
        "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "balanced_accuracy": float(np.mean([row["recall"] for row in rows])),
        "macro_one_vs_rest_auroc": float(np.mean([row["one_vs_rest_auroc"] for row in rows])),
        "weighted_one_vs_rest_auroc": float(np.average([row["one_vs_rest_auroc"] for row in rows], weights=supports)),
        "macro_one_vs_rest_average_precision": float(np.mean([row["one_vs_rest_average_precision"] for row in rows])),
        "weighted_one_vs_rest_average_precision": float(np.average([row["one_vs_rest_average_precision"] for row in rows], weights=supports)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-7, 1.0)))),
        "confusion": confusion.tolist(),
        "per_class": rows,
    }


def main() -> None:
    args = parse_args()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("split manifest hash mismatch")
    checkpoint_sha = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target_columns = list(checkpoint.get("target_columns", []))
    if target_columns not in (["tumor"], ["tumor_type"]):
        raise ValueError(f"unsupported target contract: {target_columns}")
    if checkpoint.get("split_manifest_sha256") != args.expected_split_sha256:
        raise ValueError("checkpoint was not trained on the exact G4 split manifest")
    image_size = int(checkpoint.get("image_size") or 320)
    classes = int(checkpoint.get("num_classes") or (10 if target_columns == ["tumor_type"] else 1))
    dataset = build_classification_dataset(
        root=args.data_root,
        split="val",
        target_columns=target_columns,
        image_size=image_size,
        use_clahe=False,
        augment=False,
        preprocessing_mode="none",
        normalization=str(checkpoint.get("normalization") or "imagenet"),
        split_manifest=args.split_manifest,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121AnatomyClassifier(num_classes=classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, targets, image_ids in loader:
            logits = model(images.to(device)).detach().cpu()
            if target_columns == ["tumor_type"]:
                probabilities = torch.softmax(logits, dim=1)
                tumor_probability = 1.0 - probabilities[:, 0]
                predicted_class = probabilities.argmax(dim=1)
                target_class = targets.long().view(-1)
                tumor_target = (target_class != 0).long()
            else:
                tumor_probability = torch.sigmoid(logits.view(-1))
                predicted_class = (tumor_probability >= 0.5).long()
                target_class = targets.long().view(-1)
                tumor_target = target_class
            for index, image_id in enumerate(image_ids):
                rows.append({
                    "image_id": str(image_id),
                    "tumor": int(tumor_target[index]),
                    "target_class": int(target_class[index]),
                    "predicted_class": int(predicted_class[index]),
                    "tumor_probability": float(tumor_probability[index]),
                })
                if target_columns == ["tumor_type"]:
                    for class_index in range(classes):
                        rows[-1][f"class_probability_{class_index}"] = float(probabilities[index, class_index])
    y_binary = np.asarray([row["tumor"] for row in rows], dtype=np.int64)
    probability = np.asarray([row["tumor_probability"] for row in rows], dtype=np.float64)
    summary: dict[str, object] = {
        "stage": "g4_e1_label_granularity_classifier_evaluation_v1",
        "split": "val",
        "images": len(rows),
        "target_columns": target_columns,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_metric": checkpoint.get("checkpoint_selection_metric", "historical_task_f1"),
        "split_manifest_sha256": args.expected_split_sha256,
        "binary_endpoint": _binary_metrics(y_binary, probability),
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    if target_columns == ["tumor_type"]:
        summary["ten_class_endpoint"] = _multiclass_metrics(
            np.asarray([row["target_class"] for row in rows]),
            np.asarray([row["predicted_class"] for row in rows]),
            np.asarray(
                [[row[f"class_probability_{index}"] for index in range(classes)] for row in rows],
                dtype=np.float64,
            ),
            classes,
        )
        collapsed = (np.asarray([row["predicted_class"] for row in rows]) != 0).astype(np.int64)
        summary["argmax_collapsed_binary_confusion"] = _binary_confusion(y_binary, collapsed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["predictions_sha256"] = sha256_file(predictions)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
