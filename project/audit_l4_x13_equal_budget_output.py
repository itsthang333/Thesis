from __future__ import annotations

"""Fail-closed audit for the L4 X13 equal-budget result bundle."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SOURCES = ("layercam320", "classifier448", "external_saliency")
SUBSETS = {
    "layercam320",
    "classifier448",
    "external_saliency",
    "layercam320+classifier448",
    "layercam320+external_saliency",
    "classifier448+external_saliency",
    "layercam320+classifier448+external_saliency",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, *, expected_source_commit: str, expected_split_sha256: str) -> dict[str, object]:
    freeze_path = root / "choice_freeze.json"
    per_image_path = root / "per_image.csv"
    report_path = root / "report.json"
    if not all(path.is_file() for path in (freeze_path, per_image_path, report_path)):
        raise FileNotFoundError("X13 result bundle is incomplete")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with per_image_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if (
        freeze.get("study") != "L4 X13 equal-budget source complementarity choice freeze"
        or freeze.get("source_commit") != expected_source_commit
        or freeze.get("split_sha256") != expected_split_sha256
        or int(freeze.get("validation_images", -1)) != 371
        or freeze.get("choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or int(freeze.get("test_images_read", -1)) != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("X13 choice-freeze contract differs")
    choices = freeze.get("choices")
    if not isinstance(choices, dict) or len(choices) != 371:
        raise ValueError("X13 choice cohort differs")
    tumor = 0
    normal = 0
    for image_id, choice in choices.items():
        if choice.get("abstained_by_known_normal_label") is True:
            normal += 1
            if int(choice.get("k_i", -1)) != 0 or choice.get("arms") != {}:
                raise ValueError(f"X13 normal abstention differs: {image_id}")
            if int(choice["source_counts"]["external_saliency"]) != 0:
                raise ValueError(f"X13 normal has external proposals: {image_id}")
        else:
            tumor += 1
            budget = int(choice["k_i"])
            if not 1 <= budget <= 27 or set(choice.get("arms", {})) != SUBSETS:
                raise ValueError(f"X13 tumor budget/arms differ: {image_id}")
            for arm in choice["arms"].values():
                if int(arm["budget"]) != budget or len(arm["budgeted_candidate_indices"]) != budget:
                    raise ValueError(f"X13 unequal candidate budget: {image_id}")
    if (tumor, normal) != (184, 187):
        raise ValueError("X13 tumor/normal freeze population differs")

    if len(rows) != 184 * 7 or {row["subset"] for row in rows} != SUBSETS:
        raise ValueError("X13 per-image matrix differs")
    by_subset = {name: [row for row in rows if row["subset"] == name] for name in SUBSETS}
    if any(len(items) != 184 for items in by_subset.values()):
        raise ValueError("X13 subset cohort differs")
    for row in rows:
        for key in ("dice", "iou", "oracle_dice", "selector_regret"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"X13 non-finite metric: {key}")

    if (
        report.get("study") != "L4 X13 equal-budget source complementarity"
        or report.get("split_sha256") != expected_split_sha256
        or report.get("choice_freeze_sha256") != sha256(freeze_path)
        or report.get("per_image_sha256") != sha256(per_image_path)
        or set(report.get("summary", {})) != SUBSETS
        or int(report.get("cohort", {}).get("validation", -1)) != 371
        or int(report.get("cohort", {}).get("tumor", -1)) != 184
        or int(report.get("test_images_read", -1)) != 0
        or report.get("test_evaluated") is not False
    ):
        raise ValueError("X13 report contract differs")
    for name, payload in report["summary"].items():
        if payload.get("candidate_count_equal_for_every_subset") is not True:
            raise ValueError(f"X13 budget flag differs: {name}")
        for group, expected_n in (("overall", 184), ("small", 94), ("medium", 72), ("large", 18)):
            metrics = payload["metrics"][group]
            if int(metrics["n"]) != expected_n:
                raise ValueError(f"X13 subgroup count differs: {name}/{group}")
            for key in ("dice", "iou", "oracle_dice", "selector_regret"):
                if not math.isfinite(float(metrics[key])):
                    raise ValueError(f"X13 summary non-finite: {name}/{group}/{key}")
    return {
        "pass": True,
        "validation_images": 371,
        "tumor_images": 184,
        "normal_abstentions": 187,
        "subsets": 7,
        "per_image_rows": len(rows),
        "choice_freeze_sha256": sha256(freeze_path),
        "per_image_sha256": sha256(per_image_path),
        "report_sha256": sha256(report_path),
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.output_root,
        expected_source_commit=args.expected_source_commit,
        expected_split_sha256=args.expected_split_sha256,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
