from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
EXPECTED_CHECKPOINT_SHA = "05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31"
EXPECTED_CHECKPOINT_BYTES = 230_924_939
EXPECTED_SELECTED_DICE = 0.4951316962732512
EXPECTED_FIXED_DICE = 0.489941358174933
EXPECTED_SELECTED_THRESHOLD = 0.2
MANIFEST_EXCLUSIONS = {"FILE_MANIFEST.csv", "AUDIT_VERIFICATION.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(relative: str) -> dict[str, object]:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def verify_file_manifest() -> int:
    rows = read_csv("FILE_MANIFEST.csv")
    recorded = {row["path"]: row for row in rows}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in MANIFEST_EXCLUSIONS
    }
    require(set(recorded) == actual_paths, "FILE_MANIFEST.csv does not exactly cover the snapshot")
    for relative, row in recorded.items():
        path = ROOT / relative
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {relative}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {relative}")
    return len(rows)


def verify_split() -> dict[str, int]:
    split_path = ROOT / "data/split_manifest.csv"
    require(sha256(split_path) == EXPECTED_SPLIT_SHA, "Split manifest SHA-256 mismatch")
    rows = read_csv("data/split_manifest.csv")
    eligible = [row for row in rows if row["eligible"] == "1"]
    counts = {name: sum(row["split"] == name for row in eligible) for name in ("train", "val", "test")}
    require(counts == {"train": 2981, "val": 371, "test": 373}, f"Unexpected split counts: {counts}")
    for key in ("group_id", "image_sha256"):
        ownership: dict[str, set[str]] = {}
        for row in eligible:
            ownership.setdefault(row[key], set()).add(row["split"])
        overlaps = [value for value, splits in ownership.items() if len(splits) > 1]
        require(not overlaps, f"Cross-split {key} overlap detected")
    return counts


def verify_source() -> int:
    source_files = sorted((ROOT / "project").rglob("*.py"))
    test_files = sorted((ROOT / "tests").rglob("*.py"))
    require(source_files, "No project source files")
    forbidden = ("train_tumor_only", "train-tumor-only", "rejected_legacy_train_tumor_only")
    for path in source_files + test_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"Rejected ablation token {token!r} remains in {path}")
    evaluator = (ROOT / "project/evaluate_unet.py").read_text(encoding="utf-8")
    metrics = (ROOT / "project/evaluation/segmentation_metrics.py").read_text(encoding="utf-8")
    require('args.split.lower() == "test"' in evaluator, "Validation-only threshold-sweep test guard missing")
    require("Threshold sweeping is validation-only" in evaluator, "Threshold-sweep rejection message missing")
    require('tumor = [row for row in rows if bool(row.get("gt_positive"))]' in metrics,
            "Tumor-only population definition missing")
    require('_finite_mean(tumor, "dice")' in metrics, "Mean tumor Dice aggregation missing")
    return len(source_files) + len(test_files)


def verify_training() -> dict[str, object]:
    rows = read_csv("training/training_log.csv")
    epochs = [int(row["epoch"]) for row in rows]
    require(epochs == list(range(1, 31)), "Training log must contain exactly epochs 1..30")
    best = max(rows, key=lambda row: float(row["val_positive_dice"]))
    require(int(best["epoch"]) == 20, "Best validation positive-Dice epoch is not 20")
    require(close(float(best["val_positive_dice"]), 0.49017143767812976), "Best training criterion mismatch")
    require(all(float(row["val_positive_dice"]) <= float(best["val_positive_dice"]) for row in rows[20:]),
            "An epoch after 20 exceeds the frozen best criterion")
    return {"epochs": len(rows), "best_epoch": 20, "best_val_positive_dice": float(best["val_positive_dice"])}


def verify_evaluation() -> dict[str, object]:
    convergence = read_json("convergence_summary.json")
    require(convergence["test_evaluated"] is False, "Convergence summary says test was evaluated")
    require(convergence["environment"]["split_sha256"] == EXPECTED_SPLIT_SHA, "Convergence split SHA mismatch")
    require(convergence["training"]["best_epoch"] == 20, "Convergence best epoch mismatch")
    require(convergence["training"]["last_completed_epoch"] == 30, "Convergence last epoch mismatch")
    require(convergence["training"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA,
            "Convergence checkpoint SHA mismatch")
    require(close(float(convergence["selected_threshold"]), EXPECTED_SELECTED_THRESHOLD),
            "Selected threshold mismatch")
    require(close(float(convergence["selected"]["mean_tumor_dice"]), EXPECTED_SELECTED_DICE),
            "Convergence selected Dice mismatch")
    require(close(float(convergence["fixed_0_5"]["mean_tumor_dice"]), EXPECTED_FIXED_DICE),
            "Convergence fixed Dice mismatch")

    selected = read_json("evaluation/selected_summary.json")
    fixed = read_json("evaluation/fixed_summary.json")
    require(selected["split"] == "val" and fixed["split"] == "val", "Evaluator split is not validation")
    require(selected["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA, "Selected checkpoint SHA mismatch")
    require(fixed["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA, "Fixed checkpoint SHA mismatch")
    require(close(float(selected["threshold"]), EXPECTED_SELECTED_THRESHOLD), "Selected summary threshold mismatch")
    require(close(float(selected["mean_tumor_dice"]), EXPECTED_SELECTED_DICE), "Selected summary Dice mismatch")
    require(close(float(fixed["threshold"]), 0.5), "Fixed summary threshold mismatch")
    require(close(float(fixed["mean_tumor_dice"]), EXPECTED_FIXED_DICE), "Fixed summary Dice mismatch")

    for name in ("selected", "fixed"):
        manifest = read_json(f"evaluation/{name}_per_image_run_manifest.json")
        require(manifest["split"] == "val", f"{name} run manifest is not validation")
        require(manifest["split_manifest_sha256"] == EXPECTED_SPLIT_SHA, f"{name} split SHA mismatch")
        require(manifest["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA, f"{name} checkpoint SHA mismatch")

    rows = read_csv("evaluation/selected_per_image.csv")
    require(len(rows) == 371, "Selected per-image row count is not 371")
    require(len({row["image_name"] for row in rows}) == 371, "Selected per-image IDs are not unique")
    tumor = [row for row in rows if row["gt_positive"].lower() == "true"]
    normal = [row for row in rows if row["gt_positive"].lower() == "false"]
    require((len(tumor), len(normal)) == (184, 187), "Tumor/normal evaluation counts mismatch")
    require(all(row["group_id"] for row in rows), "Missing group_id in per-image evaluation")
    measured = sum(float(row["dice"]) for row in tumor) / len(tumor)
    require(close(measured, EXPECTED_SELECTED_DICE), "Per-image tumor Dice mean mismatch")

    selection = read_json("evaluation/fixed_per_image_threshold_selection.json")
    require(selection["selection_split"] == "val", "Threshold selected outside validation")
    require(selection["candidate_count"] == 14, "Threshold candidate count mismatch")
    require(close(float(selection["selected"]["threshold"]), EXPECTED_SELECTED_THRESHOLD),
            "Threshold-selection result mismatch")
    return {
        "images": len(rows),
        "tumor_images": len(tumor),
        "normal_images": len(normal),
        "selected_threshold": EXPECTED_SELECTED_THRESHOLD,
        "selected_mean_tumor_dice": measured,
        "fixed_mean_tumor_dice": float(fixed["mean_tumor_dice"]),
        "test_evaluated": False,
    }


def main() -> None:
    checkpoint = ROOT / "model/best_unet.pt"
    require(checkpoint.stat().st_size == EXPECTED_CHECKPOINT_BYTES, "Checkpoint byte size mismatch")
    require(sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA, "Checkpoint SHA-256 mismatch")
    report = {
        "status": "PASS",
        "pipeline_id": "fs_resnet18_pw10_full_448_e20",
        "file_manifest_entries": verify_file_manifest(),
        "source_and_test_files": verify_source(),
        "split": verify_split(),
        "training": verify_training(),
        "evaluation": verify_evaluation(),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2), file=sys.stderr)
        raise
