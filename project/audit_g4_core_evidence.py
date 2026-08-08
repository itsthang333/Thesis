from __future__ import annotations

"""Bind G4 E0, selector/fusion, grouping, and oracle-reconciliation evidence."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
E0_ARMS = {
    "wsss__native",
    "wsss__320",
    "wsss__448",
    "fully__native",
    "fully__320",
    "fully__448",
}
OFFLINE_ARMS = {
    "E4__All",
    "E4__L320",
    "E4__C448",
    "E4__External",
    "E4__L320+C448",
    "E4__L320+External",
    "E4__C448+External",
    "E5_cap__27",
    "E5_cap__81",
    "E5_cap__162",
    "E5_cap__243",
    "E5_prompt_mode__point",
    "E5_prompt_mode__box",
    "E5_prompt_mode__box_point",
    "E6__random",
    "E6__sam_only",
    "E6__upstream_only",
    "E6__g1_only",
    *{f"E8__R{index}" for index in range(9)},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_matrix(
    rows: list[dict[str, str]], arms: set[str], *, tumor_field: str
) -> tuple[int, int]:
    if len(rows) != 371 * len(arms):
        raise ValueError("per-image arm-matrix row count differs")
    by_image: dict[str, set[str]] = defaultdict(set)
    tumor: dict[str, int] = {}
    for row in rows:
        image_id, arm = row["image_id"], row["arm"]
        if arm not in arms or arm in by_image[image_id]:
            raise ValueError("per-image arm matrix has duplicate/unknown arm")
        by_image[image_id].add(arm)
        value = int(row[tumor_field])
        if image_id in tumor and tumor[image_id] != value:
            raise ValueError("tumor label differs across arms")
        tumor[image_id] = value
    if len(by_image) != 371 or any(value != arms for value in by_image.values()):
        raise ValueError("per-image arm matrix is incomplete")
    if sum(tumor.values()) != 184:
        raise ValueError("per-image tumor count differs")
    return len(by_image), sum(tumor.values())


def _metric(summary: dict[str, object], arm: str) -> dict[str, float]:
    value = summary["summaries"][arm]
    return {
        "native_dice": float(value["mean_tumor_dice"]),
        "native_iou": float(value["mean_tumor_iou"]),
        "micro_dice": float(value["micro_dice"]),
        "small_native_dice": float(
            value["native_subgroups"]["small_lt_1pct"]["mean_tumor_dice"]
        ),
        "medium_native_dice": float(
            value["native_subgroups"]["medium_1_to_5pct"]["mean_tumor_dice"]
        ),
        "large_native_dice": float(
            value["native_subgroups"]["large_ge_5pct"]["mean_tumor_dice"]
        ),
    }


def audit(
    *,
    e0_root: Path,
    offline_root: Path,
    split_manifest: Path,
    e4_results_path: Path,
) -> dict[str, object]:
    if sha256(split_manifest) != SPLIT_SHA:
        raise ValueError("canonical split hash differs")

    e0_summary_path = e0_root / "summary.json"
    e0_per_image_path = e0_root / "per_image.csv"
    e0_audit_path = e0_root / "evaluation_audit.json"
    e0_summary = read_json(e0_summary_path)
    e0_audit = read_json(e0_audit_path)
    if (
        e0_audit.get("pass") is not True
        or e0_audit.get("prediction_bytes_verified_before_annotations") is not True
        or e0_audit.get("summary_sha256") != sha256(e0_summary_path)
        or e0_audit.get("per_image_sha256") != sha256(e0_per_image_path)
        or int(e0_audit.get("validation_annotations_opened", -1)) != 184
        or int(e0_audit.get("test_images_read", -1)) != 0
        or e0_audit.get("test_evaluated") is not False
        or e0_summary.get("study") != "G4 E0 coordinate and evaluator reconciliation"
        or int(e0_summary.get("images", -1)) != 371
        or int(e0_summary.get("tumor_images", -1)) != 184
        or set(e0_summary.get("summaries", {})) != E0_ARMS
        or e0_summary.get("test_evaluated") is not False
    ):
        raise ValueError("E0 evidence contract differs")
    e0_rows = read_csv(e0_per_image_path)
    e0_matrix = [
        {
            "image_id": row["image_id"],
            "arm": f"{row['method']}__{row['grid']}",
            "tumor": "1" if row["gt_positive"].strip().lower() == "true" else "0",
        }
        for row in e0_rows
    ]
    _require_matrix(e0_matrix, E0_ARMS, tumor_field="tumor")

    offline_summary_path = offline_root / "summary.json"
    offline_per_image_path = offline_root / "per_image_all_arms.csv"
    offline_audit_path = offline_root / "evaluation_audit.json"
    offline_summary = read_json(offline_summary_path)
    offline_audit = read_json(offline_audit_path)
    if (
        offline_audit.get("pass") is not True
        or offline_audit.get("choices_frozen_before_annotations") is not True
        or offline_audit.get("summary_sha256") != sha256(offline_summary_path)
        or offline_audit.get("per_image_sha256") != sha256(offline_per_image_path)
        or int(offline_audit.get("images", -1)) != 371
        or int(offline_audit.get("tumor_images", -1)) != 184
        or int(offline_audit.get("arms", -1)) != 27
        or int(offline_audit.get("per_image_rows", -1)) != 10017
        or int(offline_audit.get("validation_annotations_opened", -1)) != 184
        or int(offline_audit.get("test_images_read", -1)) != 0
        or offline_audit.get("test_evaluated") is not False
        or offline_summary.get("baseline_arm") != "E8__R7"
        or set(offline_summary.get("summaries", {})) != OFFLINE_ARMS
        or offline_summary.get("test_evaluated") is not False
    ):
        raise ValueError("offline selector/fusion evidence contract differs")
    _require_matrix(
        read_csv(offline_per_image_path), OFFLINE_ARMS, tumor_field="tumor"
    )

    split_rows = read_csv(split_manifest)
    if len(split_rows) != 3746:
        raise ValueError("canonical split row count differs")
    eligible = [row for row in split_rows if int(row["eligible"]) == 1]
    split_counts = {
        name: sum(row["split"] == name for row in eligible)
        for name in ("train", "val", "test")
    }
    if split_counts != {"train": 2981, "val": 371, "test": 373}:
        raise ValueError("canonical eligible split counts differ")
    expected_source = "consecutive_image_id_plus_stable_metadata_excluding_view"
    expected_limitation = (
        "heuristic only; BTXRD publishes no patient/lesion/case identifier"
    )
    if (
        {row["group_source"] for row in eligible} != {expected_source}
        or {row["grouping_limitation"] for row in eligible} != {expected_limitation}
        or not {row["group_confidence"] for row in eligible}.issubset(
            {"heuristic", "singleton"}
        )
    ):
        raise ValueError("group provenance differs")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        groups[row["group_id"]].append(row)
    cross_split = [
        group_id
        for group_id, rows in groups.items()
        if len({row["split"] for row in rows}) != 1
    ]
    if cross_split:
        raise ValueError("heuristic group crosses canonical splits")
    original_identity_fields = {
        name.lower()
        for name in (split_rows[0].keys() if split_rows else ())
        if name.lower() in {"patient_id", "case_id", "lesion_id", "subject_id"}
    }
    if original_identity_fields:
        raise ValueError("unexpected verified identity field needs a new grouping audit")

    e4 = read_json(e4_results_path)
    complete = e4["summary"]["layercam320+classifier448+external_saliency"][
        "metrics"
    ]["overall"]
    official_oracle = float(complete["oracle_dice"])
    scored_oracle = float(
        offline_summary["summaries"]["E8__R7"]["candidate_oracle_dice_common320"]
    )
    if (
        abs(official_oracle - 0.5282983321797708) > 1e-12
        or abs(scored_oracle - 0.5279020259081278) > 1e-12
    ):
        raise ValueError("oracle reconciliation values differ")

    selector_arms = {
        arm: {
            **_metric(offline_summary, arm),
            "common320_dice": float(
                offline_summary["summaries"][arm]["selected_dice_common320"]
            ),
        }
        for arm in (
            "E6__random",
            "E6__sam_only",
            "E6__upstream_only",
            "E6__g1_only",
            "E8__R0",
            "E8__R1",
            "E8__R2",
            "E8__R5",
            "E8__R7",
            "E8__R8",
        )
    }
    return {
        "schema_version": 1,
        "study": "G4 core evidence completion audit",
        "pass": True,
        "split_sha256": SPLIT_SHA,
        "e0": {
            "pass": True,
            "summary_sha256": sha256(e0_summary_path),
            "per_image_sha256": sha256(e0_per_image_path),
            "evaluation_audit_sha256": sha256(e0_audit_path),
            "images": 371,
            "tumor_images": 184,
            "arms": {arm: _metric(e0_summary, arm) for arm in sorted(E0_ARMS)},
            "test_images_read": 0,
            "test_evaluated": False,
        },
        "selector_and_fusion": {
            "pass": True,
            "summary_sha256": sha256(offline_summary_path),
            "per_image_sha256": sha256(offline_per_image_path),
            "evaluation_audit_sha256": sha256(offline_audit_path),
            "choice_freeze_sha256": offline_audit["choice_freeze_sha256"],
            "images": 371,
            "tumor_images": 184,
            "arms": selector_arms,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        "patient_case_grouping": {
            "verified_patient_case_identifier_available": False,
            "statistical_unit": "image",
            "uncertainty_blocks": "heuristic group_id",
            "group_source": expected_source,
            "grouping_limitation": expected_limitation,
            "eligible_rows": len(eligible),
            "eligible_split_counts": split_counts,
            "heuristic_groups": len(groups),
            "multi_image_heuristic_groups": sum(len(rows) > 1 for rows in groups.values()),
            "cross_split_heuristic_groups": 0,
            "patient_level_claim_permitted": False,
        },
        "oracle_reconciliation": {
            "official_complete_gallery_oracle_common320": official_oracle,
            "g1_eligible_scored_oracle_common320": scored_oracle,
            "absolute_gap": official_oracle - scored_oracle,
            "reason": "different declared candidate populations: complete retained gallery versus G1-eligible/scored candidates",
            "metric_or_cohort_inconsistency": False,
        },
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--e4-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("G4 core evidence output already exists")
    result = audit(
        e0_root=args.e0_root.resolve(),
        offline_root=args.offline_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
        e4_results_path=args.e4_results.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "audit_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
