from __future__ import annotations

"""Independent audit and paired comparison for completed G4 E1 runs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluate_g4_classifier_labels import _binary_metrics


SEEDS = (42, 43, 44)
SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
FIXED_PROTOCOL = {
    "image_size": 320,
    "batch_size": 4,
    "epochs": 30,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "augment": False,
    "random_erasing": False,
    "checkpoint_selection_metric": "binary_f1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--ten-class-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser.parse_args()


def _finite_summary(values: list[float]) -> dict[str, object]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _audit_arm(
    root: Path,
    *,
    arm: str,
    target_columns: list[str],
) -> tuple[dict[str, object], dict[int, list[dict[str, str]]]]:
    summary_path = root / "arm_summary.json"
    arm_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        arm_summary.get("arm") != arm
        or arm_summary.get("test_images_read") != 0
        or arm_summary.get("test_evaluated") is not False
        or len(arm_summary.get("runs", [])) != 3
    ):
        raise ValueError(f"invalid E1 {arm} arm summary")
    by_seed = {int(run["seed"]): run for run in arm_summary["runs"]}
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError(f"E1 {arm} seeds differ: {sorted(by_seed)}")

    audited_runs: list[dict[str, object]] = []
    predictions: dict[int, list[dict[str, str]]] = {}
    for seed in SEEDS:
        run_root = root / f"seed_{seed}"
        training = run_root / "training"
        evaluation = run_root / "label_safe_evaluation"
        metadata = json.loads((training / "training_metadata.json").read_text(encoding="utf-8"))
        budget = json.loads((training / "classifier_epoch_budget_audit.json").read_text(encoding="utf-8"))
        eval_summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
        prediction_path = evaluation / "predictions.csv"
        checkpoint_path = training / "best_classifier.pt"
        if any(metadata.get(key) != value for key, value in FIXED_PROTOCOL.items()):
            raise ValueError(f"E1 {arm} seed {seed} fixed protocol mismatch")
        if (
            metadata.get("split_manifest_sha256") != SPLIT_SHA
            or metadata.get("target_columns") != target_columns
            or int(metadata.get("seed", -1)) != seed
            or budget.get("valid_early_stop") is not True
            or budget.get("metric") != "audited-split validation binary_f1"
            or budget.get("split_manifest_sha256") != SPLIT_SHA
            or eval_summary.get("split_manifest_sha256") != SPLIT_SHA
            or eval_summary.get("spatial_ground_truth_read") is not False
            or eval_summary.get("test_images_read") != 0
            or eval_summary.get("test_evaluated") is not False
            or int(eval_summary.get("images", -1)) != 371
            or sha256(checkpoint_path) != by_seed[seed]["checkpoint_sha256"]
            or sha256(prediction_path) != eval_summary["predictions_sha256"]
        ):
            raise ValueError(f"E1 {arm} seed {seed} provenance/audit mismatch")
        rows = read_csv(prediction_path)
        if len(rows) != 371 or len({row["image_id"] for row in rows}) != 371:
            raise ValueError(f"E1 {arm} seed {seed} prediction cohort mismatch")
        y = np.asarray([int(row["tumor"]) for row in rows], dtype=np.int64)
        probability = np.asarray([float(row["tumor_probability"]) for row in rows])
        enhanced = _binary_metrics(y, probability)
        predictions[seed] = rows
        audited_runs.append({
            "seed": seed,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": int(eval_summary["checkpoint_epoch"]),
            "completed_epochs": int(budget["completed_epochs"]),
            "stopped_early": bool(budget["stopped_early"]),
            "binary_endpoint": enhanced,
            "ten_class_endpoint": eval_summary.get("ten_class_endpoint"),
            "predictions_sha256": sha256(prediction_path),
        })
    metrics = (
        "f1",
        "matthews_correlation_coefficient",
        "auroc",
        "average_precision_auprc",
        "brier_score",
        "negative_log_likelihood",
        "ece_15_equal_width",
    )
    aggregate = {
        metric: _finite_summary([
            float(run["binary_endpoint"][metric]) for run in audited_runs
        ])
        for metric in metrics
    }
    return {
        "arm": arm,
        "runs": audited_runs,
        "aggregate": aggregate,
        "arm_summary_sha256": sha256(summary_path),
    }, predictions


def _paired_bootstrap(
    binary: list[dict[str, str]],
    ten_class: list[dict[str, str]],
    group_by_id: dict[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    binary_by_id = {row["image_id"]: row for row in binary}
    ten_by_id = {row["image_id"]: row for row in ten_class}
    if set(binary_by_id) != set(ten_by_id):
        raise ValueError("E1 paired prediction IDs differ")
    groups: dict[str, list[str]] = {}
    for image_id in sorted(binary_by_id):
        group = group_by_id[image_id]
        groups.setdefault(group, []).append(image_id)
        if binary_by_id[image_id]["tumor"] != ten_by_id[image_id]["tumor"]:
            raise ValueError(f"E1 paired label differs for {image_id}")
    group_ids = sorted(groups)
    metrics = (
        "f1",
        "matthews_correlation_coefficient",
        "auroc",
        "average_precision_auprc",
        "brier_score",
        "negative_log_likelihood",
    )

    def score(ids: list[str], rows: dict[str, dict[str, str]]) -> dict[str, object]:
        y = np.asarray([int(rows[image_id]["tumor"]) for image_id in ids], dtype=np.int64)
        p = np.asarray([float(rows[image_id]["tumor_probability"]) for image_id in ids])
        return _binary_metrics(y, p)

    all_ids = [image_id for group in group_ids for image_id in groups[group]]
    binary_point = score(all_ids, binary_by_id)
    ten_point = score(all_ids, ten_by_id)
    samples = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        ids = [image_id for group in chosen for image_id in groups[str(group)]]
        binary_score = score(ids, binary_by_id)
        ten_score = score(ids, ten_by_id)
        for metric in metrics:
            samples[metric].append(float(ten_score[metric]) - float(binary_score[metric]))
    return {
        "comparison": "ten_class_minus_binary",
        "groups": len(group_ids),
        "group_provenance": "canonical heuristic group_id, not verified patient ID",
        "iterations": iterations,
        "metrics": {
            metric: {
                "binary": float(binary_point[metric]),
                "ten_class": float(ten_point[metric]),
                "delta": float(ten_point[metric]) - float(binary_point[metric]),
                "ci95_low": float(np.percentile(samples[metric], 2.5)),
                "ci95_high": float(np.percentile(samples[metric], 97.5)),
                "probability_delta_gt_zero": float(np.mean(np.asarray(samples[metric]) > 0)),
            }
            for metric in metrics
        },
    }


def main() -> None:
    args = parse_args()
    if sha256(args.split_manifest) != SPLIT_SHA:
        raise ValueError("canonical split SHA mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("E1 audit output must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = [
        row for row in read_csv(args.split_manifest)
        if row.get("split") == "val" and row.get("eligible") == "1"
    ]
    if len(split_rows) != 371:
        raise ValueError("canonical validation count differs")
    group_by_id = {row["image_id"]: row["group_id"] for row in split_rows}

    binary_report, binary_predictions = _audit_arm(
        args.binary_root, arm="binary", target_columns=["tumor"]
    )
    ten_report, ten_predictions = _audit_arm(
        args.ten_class_root, arm="ten_class", target_columns=["tumor_type"]
    )
    paired = {
        str(seed): _paired_bootstrap(
            binary_predictions[seed],
            ten_predictions[seed],
            group_by_id,
            iterations=args.bootstrap_iterations,
            seed=20260806 + seed,
        )
        for seed in SEEDS
    }
    report = {
        "schema_version": 1,
        "study": "G4 E1 completed-run independent audit",
        "split_sha256": SPLIT_SHA,
        "binary": binary_report,
        "ten_class": ten_report,
        "paired_by_seed": paired,
        "interpretation_boundary": (
            "Image-level evidence only. Binary-versus-ten-class WSSS superiority "
            "requires the downstream frozen-mask Dice experiment."
        ),
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "pass": True,
    }
    report_path = args.output_dir / "e1_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "binary_aggregate": binary_report["aggregate"],
        "ten_class_aggregate": ten_report["aggregate"],
        "e1_audit_sha256": sha256(report_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
