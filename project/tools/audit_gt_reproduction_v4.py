from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from audit_gt_reproduction import audit_gt_reproduction
from audit_gt_reproduction_stability import (
    compare_training_logs,
    environment_difference,
    per_image_stability,
)
from audit_wsl_gt_pair import SIZE_ORDER, build_audit, sha256_file


EXPECTED_PROTOCOL_SHA256 = (
    "f06aad247d505c81d595ea474c960e7ef32816383e4f2656fd50b3ddbdcdd247"
)
MALFORMED_MONITOR_PROTOCOL_DIGEST = (
    "f06aad247d505c81d595ea474c960e7ef32816383e4f2656fd50b3ddbddcdd247"
)
EXPECTED_WRAPPER_SHA256 = (
    "4080d04f6042ea03ba872b5bbee7ac6c0060f64ce08c3719e8a102e5d25f0193"
)
EXPECTED_PRETRAINED_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


def _summary(root: Path) -> dict[str, Any]:
    return json.loads((root / "convergence_summary.json").read_text(encoding="utf-8"))


def _selected_means(root: Path) -> dict[str, float]:
    summary = _summary(root)
    if "selected_lesion_size_subgroups" in summary:
        subgroup_means = {
            subgroup: float(
                summary["selected_lesion_size_subgroups"][subgroup][
                    "mean_tumor_dice"
                ]
            )
            for subgroup in SIZE_ORDER
        }
    else:
        subgroup_means = {
            subgroup: float(summary[f"{subgroup}_selected"]["mean_tumor_dice"])
            for subgroup in SIZE_ORDER
        }
    return {
        "overall": float(summary["selected"]["mean_tumor_dice"]),
        **subgroup_means,
    }


def _training_log(root: Path) -> Path:
    candidates = (
        root / "fs_resnet18_pw10_full_448_seed42" / "training_log.csv",
        root / "training" / "training_log.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No retained training log under {root}")


def _per_image(root: Path) -> Path:
    return root / "evaluation" / "selected_per_image.csv"


def _paired(
    split_manifest: Path,
    left_root: Path,
    right_root: Path,
    *,
    iterations: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    result = build_audit(
        split_manifest,
        _per_image(left_root),
        _per_image(right_root),
        iterations=iterations,
        seed=seed,
    )
    result["protocol"] = label
    return result


def _line_content_equal(left: Path, right: Path) -> bool:
    return left.read_text(encoding="utf-8-sig").splitlines() == right.read_text(
        encoding="utf-8-sig"
    ).splitlines()


def build_v4_audit(
    *,
    reference_lock: Path,
    reference_root: Path,
    v2_root: Path,
    v3_root: Path,
    v4_root: Path,
    v4_protocol: Path,
    locked_wrapper: Path,
    pulled_wrapper: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    protocol_hash = sha256_file(v4_protocol)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V4 protocol SHA-256 mismatch")
    protocol = json.loads(v4_protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_launch":
        raise ValueError("V4 protocol was not locked before launch")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("V4 protocol does not lock test")
    if protocol["execution"].get("expected_kernel_version") != 2:
        raise ValueError("V4 expected kernel version changed")
    if protocol["execution"].get("wrapper_sha256") != EXPECTED_WRAPPER_SHA256:
        raise ValueError("V4 protocol wrapper SHA-256 mismatch")
    if sha256_file(locked_wrapper) != EXPECTED_WRAPPER_SHA256:
        raise ValueError("Locked wrapper physical SHA-256 mismatch")
    if not _line_content_equal(locked_wrapper, pulled_wrapper):
        raise ValueError("Pulled Kaggle wrapper content differs from locked wrapper")

    v4_independent = audit_gt_reproduction(
        reference_lock,
        v4_root,
        wrapper_path=locked_wrapper,
        expected_wrapper_sha256=EXPECTED_WRAPPER_SHA256,
        iterations=iterations,
        seed=seed,
    )
    summary_v4 = _summary(v4_root)
    if summary_v4.get("test_evaluated") is not False:
        raise ValueError("V4 summary does not lock test")
    pretrained = v4_root / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if sha256_file(pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("V4 pretrained ResNet-18 SHA-256 mismatch")

    lock = json.loads(reference_lock.read_text(encoding="utf-8"))
    split_manifest = (reference_lock.parent / lock["data"]["split_manifest"]).resolve()
    paired_v4_v2 = _paired(
        split_manifest,
        v2_root,
        v4_root,
        iterations=iterations,
        seed=seed,
        label="paired v4 minus independent v2; complete validation groups",
    )
    paired_v4_v3 = _paired(
        split_manifest,
        v3_root,
        v4_root,
        iterations=iterations,
        seed=seed,
        label="paired v4 minus independent v3; complete validation groups",
    )

    reference_result = lock["reference_result"]
    run_means = {
        "frozen_reference": {
            "overall": float(reference_result["overall_mean_tumor_dice"]),
            **{
                subgroup: float(
                    reference_result[f"{subgroup}_mean_tumor_dice"]
                )
                for subgroup in SIZE_ORDER
            },
        },
        "independent_v2": _selected_means(v2_root),
        "independent_v3": _selected_means(v3_root),
        "independent_v4": _selected_means(v4_root),
    }
    four_run_range = {
        subgroup: max(means[subgroup] for means in run_means.values())
        - min(means[subgroup] for means in run_means.values())
        for subgroup in ("overall", *SIZE_ORDER)
    }
    fresh_roots = {
        "independent_v2": v2_root,
        "independent_v3": v3_root,
        "independent_v4": v4_root,
    }
    fresh_run_range = {
        subgroup: max(_selected_means(root)[subgroup] for root in fresh_roots.values())
        - min(_selected_means(root)[subgroup] for root in fresh_roots.values())
        for subgroup in ("overall", *SIZE_ORDER)
    }
    checkpoints = {
        name: _summary(root)["training"]["checkpoint_sha256"]
        for name, root in fresh_roots.items()
    }
    thresholds = {
        name: float(_summary(root)["selected_threshold"])
        for name, root in fresh_roots.items()
    }
    best_epochs = {
        name: int(_summary(root)["training"]["best_epoch"])
        for name, root in fresh_roots.items()
    }
    trajectory_v3_v4 = compare_training_logs(
        _training_log(v3_root),
        _training_log(v4_root),
    )

    subgroup_counts = v4_independent["paired_reproduction"]["population"][
        "size_counts"
    ]
    expected_counts = {
        "small_lt_1pct": 94,
        "medium_1_to_5pct": 72,
        "large_ge_5pct": 18,
    }
    if subgroup_counts != expected_counts:
        raise ValueError("V4 lesion-size cohorts changed")
    selected = summary_v4["selected"]
    if (
        int(selected["images"]),
        int(selected["tumor_images"]),
        int(selected["normal_images"]),
    ) != (371, 184, 187):
        raise ValueError("V4 validation cohort changed")
    complete_misses = int(selected["boundary_metric_complete_misses"])
    subgroup_complete_misses = sum(
        int(summary_v4["selected_lesion_size_subgroups"][subgroup][
            "boundary_metric_complete_misses"
        ])
        for subgroup in SIZE_ORDER
    )
    if complete_misses != subgroup_complete_misses:
        raise ValueError("Complete misses are not conserved across subgroups")

    contract_drift = not (
        v4_independent["candidate_contract"]["source_hashes"] == "PASS"
        and v4_independent["candidate_contract"]["split_hash"] == "PASS"
        and v4_independent["candidate_contract"]["wrapper"]["status"] == "PASS"
        and summary_v4.get("test_evaluated") is False
    )
    stochastic_instability = (
        not contract_drift
        and len(set(checkpoints.values())) > 1
        and trajectory_v3_v4["first_numeric_divergence_epoch"] == 1
    )
    large_values = [
        run_means[name]["large_ge_5pct"] for name in fresh_roots
    ]

    return {
        "schema_version": 1,
        "status": "PASS",
        "audit_role": (
            "four-run fully supervised GT reference stability sensitivity; "
            "not a WSL result"
        ),
        "test_evaluated": False,
        "v4_predeclared_protocol": {
            "path": str(v4_protocol.resolve()),
            "sha256": protocol_hash,
            "status": "PASS",
            "monitor_instruction_typo": {
                "value": MALFORMED_MONITOR_PROTOCOL_DIGEST,
                "characters": len(MALFORMED_MONITOR_PROTOCOL_DIGEST),
                "is_valid_sha256_length": False,
                "resolution": (
                    "Use the 64-character SHA-256 of the protocol file committed "
                    "before launch; the monitor text contained one extra 'd'."
                ),
            },
        },
        "wrapper_provenance": {
            "locked_upload_source": {
                "path": str(locked_wrapper.resolve()),
                "sha256": sha256_file(locked_wrapper),
            },
            "pulled_kernel_source": {
                "path": str(pulled_wrapper.resolve()),
                "raw_sha256": sha256_file(pulled_wrapper),
                "line_content_equal_to_locked_upload_source": True,
                "raw_hash_limitation": (
                    "Kaggle CLI pull changed text byte representation; exact "
                    "line content, not pulled raw bytes, is compared."
                ),
            },
            "status": "PASS_WITH_TRANSPORT_BYTE_LIMITATION",
        },
        "pretrained_resnet18": {
            "path": str(pretrained.resolve()),
            "sha256": sha256_file(pretrained),
            "expected_sha256": EXPECTED_PRETRAINED_SHA256,
            "status": "PASS",
        },
        "v4_against_frozen_reference": v4_independent,
        "v4_against_v2": paired_v4_v2,
        "v4_against_v3": paired_v4_v3,
        "v4_vs_v2_per_image_stability": per_image_stability(
            _per_image(v2_root), _per_image(v4_root)
        ),
        "v4_vs_v3_per_image_stability": per_image_stability(
            _per_image(v3_root), _per_image(v4_root)
        ),
        "environment_differences_v3_to_v4": environment_difference(
            v3_root / "convergence_summary.json",
            v4_root / "convergence_summary.json",
        ),
        "training_trajectory_v3_to_v4": trajectory_v3_v4,
        "cohort_and_complete_miss_audit": {
            "validation_images": 371,
            "tumor_images": 184,
            "normal_images": 187,
            "subgroup_counts": subgroup_counts,
            "complete_misses_in_primary_tumor_mean": True,
            "selected_threshold_complete_misses": complete_misses,
            "subgroup_complete_misses_sum": subgroup_complete_misses,
            "status": "PASS",
        },
        "audited_results": {
            "runs": run_means,
            "four_run_range": four_run_range,
            "fresh_epoch1_run_range": fresh_run_range,
            "fresh_epoch1_large_mean": statistics.fmean(large_values),
        },
        "fresh_epoch1_run_contract": {
            "checkpoint_sha256": checkpoints,
            "selected_threshold": thresholds,
            "best_epoch": best_epochs,
        },
        "causal_classification": {
            "contract_drift_detected": contract_drift,
            "stochastic_or_runtime_instability_detected": stochastic_instability,
            "evidence": (
                "V3 and v4 use the same locked wrapper/source/split/seed contract "
                "but diverge numerically at epoch 1 and select different "
                "checkpoint bytes."
            ),
            "single_numeric_kernel_identified": False,
        },
        "decision": {
            "replace_frozen_reference": False,
            "reason": (
                "The frozen reference was hash-locked before these sensitivity "
                "runs; replacing it after observing validation would invalidate "
                "paired WSL comparability."
            ),
            "report_large_subgroup_uncertainty": True,
            "large_subgroup_images": 18,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-lock", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--v4-protocol", type=Path, required=True)
    parser.add_argument("--locked-wrapper", type=Path, required=True)
    parser.add_argument("--pulled-wrapper", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_v4_audit(
        reference_lock=args.reference_lock.resolve(),
        reference_root=args.reference_root.resolve(),
        v2_root=args.v2_root.resolve(),
        v3_root=args.v3_root.resolve(),
        v4_root=args.v4_root.resolve(),
        v4_protocol=args.v4_protocol.resolve(),
        locked_wrapper=args.locked_wrapper.resolve(),
        pulled_wrapper=args.pulled_wrapper.resolve(),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
