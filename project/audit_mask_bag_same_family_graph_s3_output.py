from __future__ import annotations

"""Independent GT-blind physical auditor for frozen S3 graph output."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from audit_mask_bag_normal_prototype_r1_output import (
    _json,
    _require_safety,
    _spearman,
    _verify_validation_evidence,
    sha256_file,
)


PROTOCOL_SHA256 = "7d7636176fc05d407b51a913170ad780e2d43d328d9437b2d9d2656e191471ca"
SOURCE_COMMIT = "293b013cd036d8346fea3852ec3025772172f32d"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_WRAPPER_AUDIT_SHA256 = "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
BASELINE_CHECKPOINT_SHA256 = "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_MANIFEST_SHA256 = "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee"
PHYSICAL_HELPER_SHA256 = "3cc5feeed7fd8fddc2b630448e6bdbd7e18d9020770de850b1e580a40c173a17"
COUNT_SPEARMAN_CEILING = 0.5013777759365411
EXPECTED_GRAPH_CONFIG = {
    "minimum_iou": 0.25,
    "minimum_containment": 0.5,
    "alpha": 0.5,
    "iterations": 10,
}
REQUIRED_RUNTIME_SOURCES = {
    "project/run_mask_bag_same_family_graph_s3_arm.py",
    "project/models/mask_bag_same_family_graph.py",
    "project/models/mask_bag_relational_selector.py",
    "project/models/mask_bag_orbit_relation_training.py",
    "project/models/mask_bag_critical_relation_training.py",
    "project/run_mask_bag_normal_prototype_arm.py",
}


def _require_hex(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return text


def _verify_helper_source() -> None:
    helper = Path(__file__).with_name("audit_mask_bag_normal_prototype_r1_output.py")
    if sha256_file(helper) != PHYSICAL_HELPER_SHA256:
        raise ValueError("Pinned physical-output helper source hash mismatch")


def _verify_launch_binding(
    binding: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    if (
        binding.get("schema_version") != 1
        or binding.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("kernel_version", 0) < 1
    ):
        raise ValueError("S3 launch binding contract mismatch")
    for field in ("kernel", "checkout_commit"):
        if not str(binding.get(field, "")).strip():
            raise ValueError(f"S3 launch binding omits {field}")
    _require_hex(binding.get("bound_wrapper_sha256"), name="bound wrapper SHA-256")
    source_hashes = binding.get("runtime_source_hashes")
    if not isinstance(source_hashes, dict) or not REQUIRED_RUNTIME_SOURCES <= set(
        source_hashes
    ):
        raise ValueError("S3 launch binding omits required runtime sources")
    protocol_hashes = protocol.get("canonical_lf_source_hashes", {})
    for relative, expected in source_hashes.items():
        _require_hex(expected, name=f"runtime source {relative}")
        if protocol_hashes.get(relative) != expected:
            raise ValueError(f"S3 runtime source is not frozen by protocol: {relative}")
    return {str(key): str(value) for key, value in source_hashes.items()}


def _verify_identity_rows(
    root: Path,
    freeze: Mapping[str, Any],
    *,
    expected_validation: int,
) -> dict[str, Any]:
    path = root / "pregraph_identity_audit.csv"
    if sha256_file(path) != freeze.get("pregraph_identity_audit_sha256"):
        raise ValueError("S3 pregraph identity rows differ from prediction freeze")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "candidate_count",
        "base_candidate_logits_sha256",
        "alpha_zero_identity_exact",
        "accepted_selected_index_exact",
        "accepted_selected_logit_exact",
        "accepted_bag_logit_exact",
        "accepted_bag_probability_exact",
        "accepted_map_sha256_exact",
        "accepted_row_exact",
    }
    if (
        len(rows) != expected_validation
        or len({row["image_id"] for row in rows}) != expected_validation
        or not rows
        or set(rows[0]) != required
    ):
        raise ValueError("S3 identity cohort/schema mismatch")
    exact_fields = required - {
        "image_id",
        "candidate_count",
        "base_candidate_logits_sha256",
    }
    for row in rows:
        if int(row["candidate_count"]) <= 0:
            raise ValueError("S3 identity candidate count must be positive")
        _require_hex(
            row["base_candidate_logits_sha256"],
            name="base candidate-vector SHA-256",
        )
        if any(int(row[field]) != 1 for field in exact_fields):
            raise ValueError("S3 alpha-zero or accepted baseline identity failed")
    return {
        "sha256": sha256_file(path),
        "records": expected_validation,
        "alpha_zero_identity_exact_records": expected_validation,
        "accepted_row_exact_records": expected_validation,
        "physical_bytes": path.stat().st_size,
    }


def _verify_gt_blind_diagnostics(
    root: Path,
    freeze: Mapping[str, Any],
    *,
    expected_validation: int,
) -> dict[str, Any]:
    path = root / "gt_blind_diagnostics.csv"
    if sha256_file(path) != freeze.get("gt_blind_diagnostics_sha256"):
        raise ValueError("S3 GT-blind diagnostics differ from prediction freeze")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "candidate_count",
        "bag_probability",
        "view_swap_exact",
        "alpha_zero_identity_exact",
        "graph_symmetric",
        "cross_family_edge_count",
        "non_self_edge_count",
        "isolated_candidate_count",
        "isolated_logits_exact",
    }
    if (
        len(rows) != expected_validation
        or len({row["image_id"] for row in rows}) != expected_validation
        or not rows
        or set(rows[0]) != required
    ):
        raise ValueError("S3 GT-blind diagnostic cohort/schema mismatch")
    counts: list[int] = []
    probabilities: list[float] = []
    binary_sums = {
        "view_swap_exact": 0,
        "alpha_zero_identity_exact": 0,
        "graph_symmetric": 0,
        "isolated_logits_exact": 0,
    }
    cross_family_edges = 0
    non_self_edges = 0
    isolated_candidates = 0
    for row in rows:
        count = int(row["candidate_count"])
        probability = float(row["bag_probability"])
        if count <= 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("S3 GT-blind diagnostic value mismatch")
        counts.append(count)
        probabilities.append(probability)
        for field in binary_sums:
            value = int(row[field])
            if value not in (0, 1):
                raise ValueError(f"S3 diagnostic {field} must be binary")
            binary_sums[field] += value
        row_cross = int(row["cross_family_edge_count"])
        row_edges = int(row["non_self_edge_count"])
        row_isolated = int(row["isolated_candidate_count"])
        if row_cross < 0 or row_edges < 0 or not 0 <= row_isolated <= count:
            raise ValueError("S3 graph diagnostic counts are invalid")
        cross_family_edges += row_cross
        non_self_edges += row_edges
        isolated_candidates += row_isolated
    count_spearman = abs(_spearman(counts, probabilities))
    gate = freeze.get("gt_blind_gate", {})
    expected_pass = bool(
        count_spearman <= COUNT_SPEARMAN_CEILING
        and binary_sums["view_swap_exact"] == expected_validation
        and binary_sums["alpha_zero_identity_exact"] == expected_validation
        and binary_sums["graph_symmetric"] == expected_validation
        and cross_family_edges == 0
        and non_self_edges > 0
        and binary_sums["isolated_logits_exact"] == expected_validation
        and gate.get("accepted_baseline_identity_exact_records")
        == expected_validation
    )
    if (
        abs(
            float(gate.get("absolute_candidate_count_probability_spearman", -1))
            - count_spearman
        )
        > 1.0e-12
        or gate.get("count_probability_spearman_ceiling")
        != COUNT_SPEARMAN_CEILING
        or gate.get("count_probability_gate_pass")
        is not (count_spearman <= COUNT_SPEARMAN_CEILING)
        or gate.get("view_swap_exact_records") != binary_sums["view_swap_exact"]
        or gate.get("alpha_zero_identity_exact_records")
        != binary_sums["alpha_zero_identity_exact"]
        or gate.get("graph_symmetric_records") != binary_sums["graph_symmetric"]
        or gate.get("cross_family_edge_count") != cross_family_edges
        or gate.get("non_self_edge_count") != non_self_edges
        or gate.get("isolated_candidate_count") != isolated_candidates
        or gate.get("isolated_logits_exact_records")
        != binary_sums["isolated_logits_exact"]
        or gate.get("accepted_baseline_identity_exact_records")
        != expected_validation
        or gate.get("gt_blind_gate_pass") is not expected_pass
    ):
        raise ValueError("S3 GT-blind gate does not reproduce")
    return {
        "sha256": sha256_file(path),
        "records": expected_validation,
        "absolute_candidate_count_probability_spearman": count_spearman,
        **{f"{field}_records": value for field, value in binary_sums.items()},
        "cross_family_edge_count": cross_family_edges,
        "non_self_edge_count": non_self_edges,
        "isolated_candidate_count": isolated_candidates,
        "gt_blind_gate_pass": expected_pass,
        "physical_bytes": path.stat().st_size,
    }


def audit_s3_output(
    root: Path,
    protocol_path: Path,
    launch_binding_path: Path,
    *,
    expected_train: int = 2981,
    expected_validation: int = 371,
    expected_map_shape: tuple[int, int] = (320, 320),
) -> dict[str, object]:
    _verify_helper_source()
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S3 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    binding = _json(launch_binding_path)
    runtime_source_hashes = _verify_launch_binding(binding, protocol)
    paths = {
        "freeze": root / "prediction_freeze.json",
        "run": root / "run_manifest.json",
        "wrapper": root / "wrapper_output_audit.json",
        "identity": root / "pregraph_identity_audit.csv",
        "diagnostics": root / "gt_blind_diagnostics.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"S3 output is missing: {path}")
    freeze = _json(paths["freeze"])
    run = _json(paths["run"])
    wrapper = _json(paths["wrapper"])
    for name, payload in (
        ("prediction freeze", freeze),
        ("run manifest", run),
        ("wrapper output audit", wrapper),
    ):
        _require_safety(payload, name=name)
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("baseline_prediction_freeze_sha256")
        != BASELINE_FREEZE_SHA256
        or freeze.get("baseline_prediction_manifest_sha256")
        != BASELINE_MANIFEST_SHA256
        or freeze.get("graph_config") != EXPECTED_GRAPH_CONFIG
        or freeze.get("validation_predictions") != expected_validation
        or freeze.get("arm_fit") != "none_fixed_operator"
        or freeze.get("training_labels") != "image_level_only"
    ):
        raise ValueError("S3 prediction-freeze provenance mismatch")
    if (
        run.get("run_id") != "btxrd_mask_bag_same_family_graph_s3_v1"
        or run.get("source_commit") != SOURCE_COMMIT
        or run.get("protocol_sha256") != PROTOCOL_SHA256
        or run.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or run.get("graph_config") != EXPECTED_GRAPH_CONFIG
        or run.get("validated_cache_records")
        != {"train": expected_train, "validation": expected_validation}
        or run.get("output_hashes") != freeze
    ):
        raise ValueError("S3 run-manifest contract mismatch")
    runtime = run.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("validation_scoring_workers") != 2
        or runtime.get("validation_shards") != [186, 185]
    ):
        raise ValueError("S3 T4x2 runtime contract mismatch")
    if (
        wrapper.get("kernel") != binding["kernel"]
        or wrapper.get("kernel_version") != binding["kernel_version"]
        or wrapper.get("bound_wrapper_sha256")
        != binding["bound_wrapper_sha256"]
        or wrapper.get("checkout_commit") != binding["checkout_commit"]
        or wrapper.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper.get("source_hashes") != runtime_source_hashes
        or wrapper.get("physical_prediction_maps_verified") != expected_validation
        or wrapper.get("physical_candidate_score_payloads_verified")
        != expected_validation
        or wrapper.get("physical_pregraph_identity_rows_verified")
        != expected_validation
        or wrapper.get("physical_gt_blind_diagnostic_rows_verified")
        != expected_validation
    ):
        raise ValueError("S3 wrapper-output audit mismatch")
    t4 = wrapper.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("S3 wrapper T4x2 evidence mismatch")
    cache = wrapper.get("cache", {})
    if (
        cache.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or cache.get("selector_cache_wrapper_audit_sha256")
        != CACHE_WRAPPER_AUDIT_SHA256
        or cache.get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("S3 wrapper cache evidence mismatch")
    if sha256_file(paths["identity"]) != freeze.get(
        "pregraph_identity_audit_sha256"
    ):
        raise ValueError("S3 identity rows differ from prediction freeze")
    if sha256_file(paths["diagnostics"]) != freeze.get(
        "gt_blind_diagnostics_sha256"
    ):
        raise ValueError("S3 diagnostics differ from prediction freeze")
    identity = _verify_identity_rows(
        root,
        freeze,
        expected_validation=expected_validation,
    )
    validation = _verify_validation_evidence(
        root,
        freeze,
        expected_validation=expected_validation,
        expected_map_shape=expected_map_shape,
    )
    diagnostics = _verify_gt_blind_diagnostics(
        root,
        freeze,
        expected_validation=expected_validation,
    )
    if (
        abs(
            float(validation["validation_absolute_count_probability_spearman"])
            - float(diagnostics["absolute_candidate_count_probability_spearman"])
        )
        > 1.0e-12
    ):
        raise ValueError("S3 prediction and diagnostic count associations differ")
    physical_bytes = sum(
        path.stat().st_size for path in (*paths.values(), launch_binding_path)
    )
    return {
        "audit_id": "independent_mask_bag_same_family_graph_s3_output_v1",
        "status": (
            "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
            if diagnostics["gt_blind_gate_pass"]
            else "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_FAIL"
        ),
        "kernel": binding["kernel"],
        "kernel_version": binding["kernel_version"],
        "bound_wrapper_sha256": binding["bound_wrapper_sha256"],
        "prediction_freeze_sha256": sha256_file(paths["freeze"]),
        "run_manifest_sha256": sha256_file(paths["run"]),
        "wrapper_output_audit_sha256": sha256_file(paths["wrapper"]),
        "launch_binding_sha256": sha256_file(launch_binding_path),
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "cache_freeze_sha256": CACHE_FREEZE_SHA256,
        "baseline_prediction_freeze_sha256": BASELINE_FREEZE_SHA256,
        "graph_config": EXPECTED_GRAPH_CONFIG,
        "pregraph_identity": identity,
        "validation": validation,
        "gt_blind_diagnostics": diagnostics,
        "physical_output_bytes_verified": physical_bytes
        + int(validation["physical_validation_evidence_bytes"]),
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch-binding", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_s3_output(
        args.output_root.resolve(),
        args.protocol.resolve(),
        args.launch_binding.resolve(),
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
