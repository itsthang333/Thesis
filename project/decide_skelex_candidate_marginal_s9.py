"""Apply the predeclared S9 gates to two hash-frozen evaluation tables."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
PROTOCOL_SHA256 = "0a303c9c86c3c43c750c85a50087e792bf0942a0b43fc9a1cf9e143c4832ee3d"
SOURCE_COMMIT = "7dcd6c6f055c69f3f048a005ed2fea6177dc7ed8"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_PER_IMAGE_SHA256 = "a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_s9_likelihood_equal_rank"
ARMS = (CONTROL_ARM, PRIMARY_ARM)
PRE_GT_STATUS = "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_REPRODUCTION_PASS"
READINESS_STATUS = "FROZEN_AFTER_INDEPENDENT_GT_BLIND_AUDIT_BEFORE_VALIDATION_GT"
SUBGROUPS = ("overall", "small", "medium", "large")
EXPECTED_COUNTS = {"small": 94, "medium": 72, "large": 18}
COHORT = {"validation": 371, "tumor": 184, "normal": 187, **EXPECTED_COUNTS}
GOALS = {
    "overall": 0.34024039,
    "small": 0.17895493,
    "medium": 0.51244178,
    "large": 0.49370336,
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261205
REQUIRED_EVALUATION_FILES = {
    "gate_decision.json",
    "paired_comparison.json",
    "per_image.csv",
    "summary.json",
}
REQUIRED_PER_IMAGE_FIELDS = {
    "image_id",
    "group_id",
    "gt_area_ratio",
    "size_group",
    "dice",
    "oracle_best_single_dice",
    "complete_miss",
    "baseline_dice",
    "baseline_oracle_best_single_dice",
    "baseline_complete_miss",
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_safety(payload: Mapping[str, Any], *, name: str) -> None:
    if (
        payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S9 {name} safety lock mismatch")


def _read_per_image(path: Path, expected_sha256: str) -> dict[str, dict[str, str]]:
    _require_sha(expected_sha256, name=f"{path.name} SHA-256")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Per-image SHA-256 mismatch: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 184 or not rows:
        raise ValueError("Each S9 evaluation must contain exactly 184 tumor rows")
    missing = sorted(REQUIRED_PER_IMAGE_FIELDS - set(rows[0]))
    if missing:
        raise ValueError(f"S9 per-image evaluation lacks fields: {missing}")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["image_id"]
        if not image_id or image_id in indexed:
            raise ValueError("S9 evaluation has duplicate/empty image IDs")
        if row["size_group"] not in EXPECTED_COUNTS:
            raise ValueError("S9 evaluation has an invalid subgroup")
        if row["complete_miss"] not in {"0", "1"}:
            raise ValueError("S9 evaluation has an invalid miss flag")
        if row["baseline_complete_miss"] not in {"0", "1"}:
            raise ValueError("S9 evaluation has an invalid baseline miss flag")
        for field in (
            "gt_area_ratio",
            "dice",
            "oracle_best_single_dice",
            "baseline_dice",
            "baseline_oracle_best_single_dice",
        ):
            value = float(row[field])
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"Invalid {field} for {image_id}")
        indexed[image_id] = row
    counts = {
        subgroup: sum(row["size_group"] == subgroup for row in rows)
        for subgroup in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Frozen S9 subgroup counts differ: {counts}")
    return indexed


def _paired_group_bootstrap(
    deltas: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if deltas.shape != (len(groups),):
        raise ValueError("S9 bootstrap values/groups do not align")
    grouped: dict[str, list[float]] = {}
    for index, value in enumerate(deltas):
        grouped.setdefault(str(groups[index]), []).append(float(value))
    unique = sorted(grouped)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = generator.integers(0, len(unique), size=len(unique))
        values = [value for position in sampled for value in grouped[unique[position]]]
        bootstrap[replicate] = float(np.mean(values))
    return {
        "delta_primary_minus_control": float(deltas.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_images": int(deltas.size),
        "n_groups": len(unique),
    }


def _verify_evaluation(
    root: Path,
    audit_path: Path,
    *,
    expected_audit_sha256: str,
    expected_arm_freeze_sha256: str,
    expected_score_manifest_sha256: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if sha256_file(audit_path) != expected_audit_sha256:
        raise ValueError("S9 evaluation-audit SHA-256 mismatch")
    audit = _json(audit_path)
    _require_safety(audit, name="evaluation audit")
    if (
        audit.get("split_sha256") != SPLIT_SHA256
        or audit.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or audit.get("arm_prediction_freeze_sha256")
        != expected_arm_freeze_sha256
        or audit.get("candidate_score_manifest_sha256")
        != expected_score_manifest_sha256
        or audit.get("baseline_prediction_freeze_sha256")
        != BASELINE_FREEZE_SHA256
        or audit.get("baseline_per_image_sha256") != BASELINE_PER_IMAGE_SHA256
        or audit.get("cohort") != COHORT
        or audit.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or audit.get(
            "validation_gt_read_only_after_all_predictions_frozen_and_verified"
        )
        is not True
    ):
        raise ValueError("S9 evaluation boundary/provenance mismatch")
    output_hashes = audit.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != REQUIRED_EVALUATION_FILES:
        raise ValueError("S9 evaluation output inventory mismatch")
    for name, expected in output_hashes.items():
        if not (root / name).is_file() or sha256_file(root / name) != expected:
            raise ValueError(f"S9 evaluation output hash mismatch: {name}")
    summary = _json(root / "summary.json")
    paired = _json(root / "paired_comparison.json")
    gate = _json(root / "gate_decision.json")
    for name, payload in (("summary", summary), ("paired", paired), ("gate", gate)):
        _require_safety(payload, name=name)
    if (
        summary.get("arm_source_commit") != SOURCE_COMMIT
        or summary.get("arm_protocol_sha256") != PROTOCOL_SHA256
        or summary.get("cohort") != COHORT
        or summary.get(
            "validation_gt_read_only_after_all_predictions_frozen_and_verified"
        )
        is not True
        or paired.get("replicates") != BOOTSTRAP_REPLICATES
        or paired.get("seed_family") != BOOTSTRAP_SEED
    ):
        raise ValueError("S9 evaluator source/protocol/bootstrap mismatch")
    return _read_per_image(root / "per_image.csv", output_hashes["per_image.csv"]), audit


def decide(
    pre_gt_audit_path: Path,
    protocol_path: Path,
    evaluation_addendum_path: Path,
    readiness_path: Path,
    control_evaluation_root: Path,
    control_evaluation_audit_path: Path,
    primary_evaluation_root: Path,
    primary_evaluation_audit_path: Path,
    output_dir: Path,
    *,
    expected_pre_gt_audit_sha256: str,
    expected_evaluation_addendum_sha256: str,
    expected_readiness_sha256: str,
    expected_control_evaluation_audit_sha256: str,
    expected_primary_evaluation_audit_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("S9 decision output already exists")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S9 protocol SHA-256 mismatch")
    if sha256_file(evaluation_addendum_path) != expected_evaluation_addendum_sha256:
        raise ValueError("S9 evaluation addendum SHA-256 mismatch")
    addendum = _json(evaluation_addendum_path)
    if (
        addendum.get("experiment_id") != EXPERIMENT_ID
        or addendum.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or addendum.get("bootstrap_seed") != BOOTSTRAP_SEED
        or addendum.get("arm_order") != list(ARMS)
        or addendum.get("validation_gt_read") is not False
    ):
        raise ValueError("S9 evaluation addendum contract mismatch")
    if sha256_file(pre_gt_audit_path) != expected_pre_gt_audit_sha256:
        raise ValueError("S9 pre-GT audit SHA-256 mismatch")
    pre_gt = _json(pre_gt_audit_path)
    if (
        pre_gt.get("status") != PRE_GT_STATUS
        or pre_gt.get("validation_gt_read") is not False
        or pre_gt.get("consumer_trained") is not False
        or pre_gt.get("test_evaluated") is not False
    ):
        raise ValueError("S9 pre-GT audit contract mismatch")
    if sha256_file(readiness_path) != expected_readiness_sha256:
        raise ValueError("S9 readiness SHA-256 mismatch")
    readiness = _json(readiness_path)
    if (
        readiness.get("status") != READINESS_STATUS
        or readiness.get("experiment_id") != EXPERIMENT_ID
        or readiness.get("protocol_sha256") != PROTOCOL_SHA256
        or readiness.get("evaluation_addendum_sha256")
        != expected_evaluation_addendum_sha256
        or readiness.get("terminal_pre_gt_audit_sha256")
        != expected_pre_gt_audit_sha256
        or readiness.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or readiness.get("bootstrap_seed") != BOOTSTRAP_SEED
        or set(readiness.get("arms", {})) != set(ARMS)
        or readiness.get("validation_gt_read") is not False
    ):
        raise ValueError("S9 readiness contract mismatch")

    control_contract = readiness["arms"][CONTROL_ARM]
    primary_contract = readiness["arms"][PRIMARY_ARM]
    control, control_audit = _verify_evaluation(
        control_evaluation_root,
        control_evaluation_audit_path,
        expected_audit_sha256=expected_control_evaluation_audit_sha256,
        expected_arm_freeze_sha256=control_contract["prediction_freeze_sha256"],
        expected_score_manifest_sha256=control_contract[
            "candidate_score_manifest_sha256"
        ],
    )
    primary, primary_audit = _verify_evaluation(
        primary_evaluation_root,
        primary_evaluation_audit_path,
        expected_audit_sha256=expected_primary_evaluation_audit_sha256,
        expected_arm_freeze_sha256=primary_contract["prediction_freeze_sha256"],
        expected_score_manifest_sha256=primary_contract[
            "candidate_score_manifest_sha256"
        ],
    )
    if set(control) != set(primary):
        raise ValueError("S9 evaluated arm image identities differ")

    paired_rows: list[dict[str, Any]] = []
    control_exactly_baseline = True
    for image_id in sorted(control):
        old = control[image_id]
        new = primary[image_id]
        for field in (
            "group_id",
            "gt_area_ratio",
            "size_group",
            "oracle_best_single_dice",
            "baseline_dice",
            "baseline_complete_miss",
        ):
            if new[field] != old[field]:
                raise ValueError(f"S9 frozen paired field {field} differs: {image_id}")
        old_dice = float(old["dice"])
        new_dice = float(new["dice"])
        old_miss = int(old["complete_miss"])
        new_miss = int(new["complete_miss"])
        control_exactly_baseline &= old_dice == float(old["baseline_dice"])
        control_exactly_baseline &= old_miss == int(old["baseline_complete_miss"])
        paired_rows.append(
            {
                "image_id": image_id,
                "group_id": old["group_id"],
                "size_group": old["size_group"],
                "primary_dice": new_dice,
                "control_dice": old_dice,
                "delta_dice": new_dice - old_dice,
                "primary_complete_miss": new_miss,
                "control_complete_miss": old_miss,
                "miss_recovered": int(old_miss == 1 and new_miss == 0),
                "overlap_lost": int(old_miss == 0 and new_miss == 1),
            }
        )

    metrics: dict[str, dict[str, Any]] = {}
    for subgroup_index, subgroup in enumerate(SUBGROUPS):
        rows = [
            row
            for row in paired_rows
            if subgroup == "overall" or row["size_group"] == subgroup
        ]
        deltas = np.asarray([float(row["delta_dice"]) for row in rows], dtype=np.float64)
        metrics[subgroup] = {
            **_paired_group_bootstrap(
                deltas,
                [str(row["group_id"]) for row in rows],
                seed=BOOTSTRAP_SEED + subgroup_index,
            ),
            "primary_dice": float(
                np.mean([float(row["primary_dice"]) for row in rows])
            ),
            "control_dice": float(
                np.mean([float(row["control_dice"]) for row in rows])
            ),
            "primary_complete_misses": int(
                sum(int(row["primary_complete_miss"]) for row in rows)
            ),
            "control_complete_misses": int(
                sum(int(row["control_complete_miss"]) for row in rows)
            ),
            "misses_recovered": int(sum(int(row["miss_recovered"]) for row in rows)),
            "overlaps_lost": int(sum(int(row["overlap_lost"]) for row in rows)),
        }
    mechanism_checks = {
        "overall_mean_strictly_improves": metrics["overall"][
            "delta_primary_minus_control"
        ]
        > 0.0,
        "small_mean_strictly_improves": metrics["small"][
            "delta_primary_minus_control"
        ]
        > 0.0,
        "medium_mean_no_regression": metrics["medium"][
            "delta_primary_minus_control"
        ]
        >= 0.0,
        "large_mean_no_regression": metrics["large"][
            "delta_primary_minus_control"
        ]
        >= 0.0,
        "complete_misses_no_increase": metrics["overall"][
            "primary_complete_misses"
        ]
        <= metrics["overall"]["control_complete_misses"],
    }
    goal_checks = {
        subgroup: {
            "value": metrics[subgroup]["primary_dice"],
            "goal": GOALS[subgroup],
            "pass": metrics[subgroup]["primary_dice"] >= GOALS[subgroup],
        }
        for subgroup in SUBGROUPS
    }
    safety_checks = {
        "pre_gt_independent_audit_pass": True,
        "control_exactly_accepted_geometry_v3_baseline": control_exactly_baseline,
        "overall_paired_ci95_lower_above_zero": metrics["overall"]["ci95"][0]
        > 0.0,
        "no_subgroup_mean_regression": all(
            metrics[subgroup]["delta_primary_minus_control"] >= 0.0
            for subgroup in ("small", "medium", "large")
        ),
        "no_complete_miss_increase": mechanism_checks[
            "complete_misses_no_increase"
        ],
        "evaluation_boundaries_and_hashes_pass": True,
    }
    mechanism_pass = all(mechanism_checks.values())
    operational_pass = (
        mechanism_pass
        and all(check["pass"] for check in goal_checks.values())
        and all(safety_checks.values())
    )
    status = (
        "OPERATIONAL_PASS"
        if operational_pass
        else "MECHANISM_PASS"
        if mechanism_pass
        else "FAIL"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    paired_path = output_dir / "paired_per_image.csv"
    with paired_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(paired_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(paired_rows)
    comparison = {
        "comparison": f"{PRIMARY_ARM} minus {CONTROL_ARM}",
        "method": "paired complete-group bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed_family": BOOTSTRAP_SEED,
        "cohort": {"tumor": 184, **EXPECTED_COUNTS},
        "metrics": metrics,
        "ground_truth_reopened": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    comparison_path = output_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = {
        "decision_id": "skelex_candidate_marginal_s9_decision_v1",
        "status": status,
        "mechanism_checks": mechanism_checks,
        "operational_goal_checks": goal_checks,
        "final_safety_checks": safety_checks,
        "mechanism_pass": mechanism_pass,
        "operational_pass": operational_pass,
        "consumer_authorized": operational_pass,
        "post_hoc_rescue_or_sweep_authorized": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    gate_path = output_dir / "gate_decision.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "evaluation_addendum_sha256": expected_evaluation_addendum_sha256,
        "postfreeze_readiness_sha256": expected_readiness_sha256,
        "pre_gt_audit_sha256": expected_pre_gt_audit_sha256,
        "control_evaluation_audit_sha256": expected_control_evaluation_audit_sha256,
        "primary_evaluation_audit_sha256": expected_primary_evaluation_audit_sha256,
        "control_evaluation_output_hashes": control_audit["output_hashes"],
        "primary_evaluation_output_hashes": primary_audit["output_hashes"],
        "output_hashes": {
            "paired_per_image.csv": sha256_file(paired_path),
            "paired_comparison.json": sha256_file(comparison_path),
            "gate_decision.json": sha256_file(gate_path),
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prediction_pair_freeze_sha256": readiness[
            "prediction_pair_freeze_sha256"
        ],
        "validation_gt_read_only_after_pair_freeze_and_independent_audit": True,
        "ground_truth_reopened_for_matched_comparison": False,
        "consumer_authorized": operational_pass,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    audit_path = output_dir / "decision_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"comparison": comparison, "gate": gate, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-gt-audit", type=Path, required=True)
    parser.add_argument("--expected-pre-gt-audit-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation-addendum", type=Path, required=True)
    parser.add_argument("--expected-evaluation-addendum-sha256", required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--control-evaluation-root", type=Path, required=True)
    parser.add_argument("--control-evaluation-audit", type=Path, required=True)
    parser.add_argument("--expected-control-evaluation-audit-sha256", required=True)
    parser.add_argument("--primary-evaluation-root", type=Path, required=True)
    parser.add_argument("--primary-evaluation-audit", type=Path, required=True)
    parser.add_argument("--expected-primary-evaluation-audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        args.pre_gt_audit.resolve(),
        args.protocol.resolve(),
        args.evaluation_addendum.resolve(),
        args.readiness.resolve(),
        args.control_evaluation_root.resolve(),
        args.control_evaluation_audit.resolve(),
        args.primary_evaluation_root.resolve(),
        args.primary_evaluation_audit.resolve(),
        args.output_dir.resolve(),
        expected_pre_gt_audit_sha256=args.expected_pre_gt_audit_sha256,
        expected_evaluation_addendum_sha256=args.expected_evaluation_addendum_sha256,
        expected_readiness_sha256=args.expected_readiness_sha256,
        expected_control_evaluation_audit_sha256=(
            args.expected_control_evaluation_audit_sha256
        ),
        expected_primary_evaluation_audit_sha256=(
            args.expected_primary_evaluation_audit_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
