from __future__ import annotations

"""Independent fail-closed audit for the exact G4 E4 source ablation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_SUBSETS = {
    "layercam320",
    "classifier448",
    "external_saliency",
    "layercam320+classifier448",
    "layercam320+external_saliency",
    "classifier448+external_saliency",
    "layercam320+classifier448+external_saliency",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audit(root: Path, expected_split_sha256: str) -> dict[str, object]:
    report_path = root / "report.json"
    choice_path = root / "choice_freeze.json"
    per_image_path = root / "per_image.csv"
    if not all(path.is_file() for path in (report_path, choice_path, per_image_path)):
        raise FileNotFoundError("E4 output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    choices = json.loads(choice_path.read_text(encoding="utf-8"))
    rows = read_csv(per_image_path)
    if (
        report.get("study") != "G4 E4 exact final-gallery source-subset ablation"
        or report.get("split_sha256") != expected_split_sha256
        or report.get("choice_freeze_sha256") != sha256(choice_path)
        or report.get("per_image_sha256") != sha256(per_image_path)
        or report.get("validation_gt_opened_after_choice_freeze") is not True
        or report.get("test_images_read") != 0
        or report.get("test_evaluated") is not False
        or choices.get("choices_frozen_before_validation_gt") is not True
        or choices.get("validation_gt_read") is not False
        or choices.get("test_images_read") != 0
        or choices.get("test_evaluated") is not False
    ):
        raise ValueError("E4 provenance/GT boundary differs")
    if set(report.get("summary", {})) != EXPECTED_SUBSETS:
        raise ValueError("E4 subset set differs")
    if len(rows) != 7 * 184:
        raise ValueError("E4 per-image row count differs")
    image_ids = {row["image_id"] for row in rows}
    subsets = {row["subset"] for row in rows}
    subgroup_counts = {
        group: len({row["image_id"] for row in rows if row["size_group"] == group})
        for group in ("small", "medium", "large")
    }
    if (
        len(image_ids) != 184
        or subsets != EXPECTED_SUBSETS
        or subgroup_counts != {"small": 94, "medium": 72, "large": 18}
        or int(choices.get("validation_images", -1)) != 371
        or len(choices.get("choices", {})) != 371
    ):
        raise ValueError("E4 canonical cohort differs")
    full = report["summary"]["layercam320+classifier448+external_saliency"]["metrics"]
    if (
        abs(float(full["overall"]["dice"]) - 0.28872948670665205) > 1e-12
        or abs(float(full["overall"]["oracle_dice"]) - 0.5282983321797708) > 1e-12
        or float(report["resource_metrics"]["total_all_seven_subsets_seconds"]) <= 0
    ):
        raise ValueError("E4 exact baseline/resource endpoint differs")
    return {
        "schema_version": 1,
        "study": "independent G4 E4 source-subset output audit",
        "pass": True,
        "report_sha256": sha256(report_path),
        "choice_freeze_sha256": sha256(choice_path),
        "per_image_sha256": sha256(per_image_path),
        "split_sha256": expected_split_sha256,
        "subsets": 7,
        "validation_images": 371,
        "tumor_images": 184,
        "subgroups": subgroup_counts,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.root.resolve(), args.expected_split_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "audit_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
