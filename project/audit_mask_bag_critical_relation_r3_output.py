from __future__ import annotations

"""Independent GT-blind physical auditor for frozen R3 output."""

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


PROTOCOL_SHA256 = "f7253df27444c1b56706ac19646441a7b2d2a7374a4d6888b8e97d26c2c3fd03"
SOURCE_COMMIT = "84867698dbc652957bd1a1430f4b9d32fa399119"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_WRAPPER_AUDIT_SHA256 = (
    "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
)
BASELINE_CHECKPOINT_SHA256 = (
    "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
)
PHYSICAL_HELPER_SHA256 = (
    "3cc5feeed7fd8fddc2b630448e6bdbd7e18d9020770de850b1e580a40c173a17"
)
COUNT_SPEARMAN_CEILING = 0.5013777759365411
EXPECTED_TRAINING_CONFIG = {
    "epochs": 16,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "hidden_dim": 128,
    "instance_loss_weight": 0.25,
    "consistency_weight": 0.10,
    "instance_warmup_epochs": 2,
    "seed": 42,
}
REQUIRED_RUNTIME_SOURCES = {
    "project/run_mask_bag_critical_relation_arm.py",
    "project/models/mask_bag_critical_relation_training.py",
    "project/models/mask_bag_relational_selector.py",
    "project/run_mask_bag_normal_prototype_arm.py",
}


def _require_hex(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return text


def _verify_helper_source() -> None:
    helper = Path(__file__).with_name("audit_mask_bag_normal_prototype_r1_output.py")
    if sha256_file(helper) != PHYSICAL_HELPER_SHA256:
        raise ValueError("Pinned physical-output helper source hash mismatch")


def _verify_launch_binding(
    binding: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, str]:
    if (
        binding.get("schema_version") != 1
        or binding.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("kernel_version", 0) < 1
    ):
        raise ValueError("R3 launch binding contract mismatch")
    for field in ("kernel", "checkout_commit"):
        if not str(binding.get(field, "")).strip():
            raise ValueError(f"R3 launch binding omits {field}")
    _require_hex(binding.get("bound_wrapper_sha256"), name="bound wrapper SHA-256")
    source_hashes = binding.get("runtime_source_hashes")
    if not isinstance(source_hashes, dict) or not REQUIRED_RUNTIME_SOURCES <= set(source_hashes):
        raise ValueError("R3 launch binding omits required runtime sources")
    protocol_hashes = protocol.get("canonical_lf_source_hashes", {})
    for relative, expected in source_hashes.items():
        _require_hex(expected, name=f"runtime source {relative}")
        if protocol_hashes.get(relative) != expected:
            raise ValueError(f"R3 runtime source is not frozen by the protocol: {relative}")
    return {str(key): str(value) for key, value in source_hashes.items()}


def _verify_history(path: Path) -> None:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 16:
        raise ValueError("R3 history must contain exactly 16 fixed epochs")
    required = {"epoch", "total", "image", "instance", "consistency"}
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict) or set(row) != required or int(row["epoch"]) != index:
            raise ValueError("R3 history schema or epoch order mismatch")
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("R3 history contains a non-finite value")


def _verify_pretraining_audit(
    path: Path, *, expected_train: int, expected_validation: int
) -> dict[str, Any]:
    payload = _json(path)
    _require_safety(payload, name="R3 pretraining identity audit")
    if (
        payload.get("source_commit") != SOURCE_COMMIT
        or payload.get("protocol_sha256") != PROTOCOL_SHA256
        or set(payload) != {
            "train",
            "validation",
            "source_commit",
            "protocol_sha256",
            "validation_gt_read",
            "consumer_trained",
            "test_evaluated",
        }
    ):
        raise ValueError("R3 pretraining identity provenance mismatch")
    for split, expected in (("train", expected_train), ("validation", expected_validation)):
        row = payload[split]
        if (
            row.get("records") != expected
            or int(row.get("candidates", 0)) <= expected
            or row.get("zero_residual_exact") is not True
            or row.get("combined_equals_frozen_base_exact") is not True
            or not 0 <= int(row.get("base_flip_critical_agreement_count", -1)) <= expected
        ):
            raise ValueError(f"R3 pretraining identity mismatch: {split}")
        expected_rate = row["base_flip_critical_agreement_count"] / expected
        if float(row.get("base_flip_critical_agreement", -1)) != expected_rate:
            raise ValueError(f"R3 base critical agreement mismatch: {split}")
    return payload


def _verify_gt_blind_diagnostics(
    root: Path,
    freeze: Mapping[str, Any],
    initial_audit: Mapping[str, Any],
    *,
    expected_validation: int,
) -> dict[str, Any]:
    path = root / "gt_blind_diagnostics.csv"
    if sha256_file(path) != freeze.get("gt_blind_diagnostics_sha256"):
        raise ValueError("R3 GT-blind diagnostics differ from prediction freeze")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "candidate_count",
        "bag_probability",
        "base_critical_agreement",
        "final_selected_agreement",
    }
    if (
        len(rows) != expected_validation
        or len({row["image_id"] for row in rows}) != expected_validation
        or not rows
        or set(rows[0]) != required
    ):
        raise ValueError("R3 GT-blind diagnostic cohort/schema mismatch")
    counts: list[int] = []
    probabilities: list[float] = []
    base: list[int] = []
    final: list[int] = []
    for row in rows:
        count = int(row["candidate_count"])
        probability = float(row["bag_probability"])
        base_value = int(row["base_critical_agreement"])
        final_value = int(row["final_selected_agreement"])
        if count <= 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("R3 GT-blind diagnostic value mismatch")
        if base_value not in (0, 1) or final_value not in (0, 1):
            raise ValueError("R3 agreement diagnostics must be binary")
        counts.append(count)
        probabilities.append(probability)
        base.append(base_value)
        final.append(final_value)
    count_spearman = abs(_spearman(counts, probabilities))
    base_agreement = sum(base) / expected_validation
    final_agreement = sum(final) / expected_validation
    if base_agreement != float(initial_audit["validation"]["base_flip_critical_agreement"]):
        raise ValueError("R3 frozen/recomputed base agreement differs")
    gate = freeze.get("gt_blind_gate", {})
    minimum = base_agreement - 0.01
    expected_count_pass = count_spearman <= COUNT_SPEARMAN_CEILING
    expected_agreement_pass = final_agreement >= minimum
    if (
        abs(float(gate.get("absolute_candidate_count_probability_spearman", -1)) - count_spearman) > 1.0e-12
        or gate.get("count_probability_spearman_ceiling") != COUNT_SPEARMAN_CEILING
        or gate.get("count_probability_gate_pass") is not expected_count_pass
        or float(gate.get("base_flip_critical_agreement", -1)) != base_agreement
        or float(gate.get("final_flip_selected_agreement", -1)) != final_agreement
        or float(gate.get("minimum_allowed_final_agreement", -1)) != minimum
        or gate.get("critical_agreement_gate_pass") is not expected_agreement_pass
        or gate.get("gt_blind_gate_pass") is not (expected_count_pass and expected_agreement_pass)
    ):
        raise ValueError("R3 GT-blind gate does not reproduce")
    return {
        "sha256": sha256_file(path),
        "records": expected_validation,
        "absolute_candidate_count_probability_spearman": count_spearman,
        "base_flip_critical_agreement": base_agreement,
        "final_flip_selected_agreement": final_agreement,
        "gt_blind_gate_pass": expected_count_pass and expected_agreement_pass,
        "physical_bytes": path.stat().st_size,
    }


def audit_r3_output(
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
        raise ValueError("R3 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    binding = _json(launch_binding_path)
    runtime_source_hashes = _verify_launch_binding(binding, protocol)

    paths = {
        "freeze": root / "prediction_freeze.json",
        "run": root / "run_manifest.json",
        "wrapper": root / "wrapper_output_audit.json",
        "checkpoint": root / "critical_relation_residual.pt",
        "history": root / "training_history.json",
        "initial": root / "pretraining_identity_audit.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"R3 output is missing: {path}")
    freeze = _json(paths["freeze"])
    run = _json(paths["run"])
    wrapper = _json(paths["wrapper"])
    for name, payload in (("prediction freeze", freeze), ("run manifest", run), ("wrapper output audit", wrapper)):
        _require_safety(payload, name=name)
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("validation_predictions") != expected_validation
        or freeze.get("training_labels") != "image_level_only"
        or freeze.get("epoch_selection") != "fixed_final_epoch_only"
    ):
        raise ValueError("R3 prediction-freeze provenance mismatch")
    if (
        run.get("run_id") != "btxrd_mask_bag_critical_relation_r3_v1"
        or run.get("source_commit") != SOURCE_COMMIT
        or run.get("protocol_sha256") != PROTOCOL_SHA256
        or run.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or run.get("training_config") != EXPECTED_TRAINING_CONFIG
        or run.get("validated_cache_records") != {"train": expected_train, "validation": expected_validation}
        or run.get("output_hashes") != freeze
    ):
        raise ValueError("R3 run-manifest contract mismatch")
    runtime = run.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("validation_scoring_workers") != 2
        or runtime.get("validation_shards") != [186, 185]
    ):
        raise ValueError("R3 T4x2 runtime contract mismatch")
    if (
        wrapper.get("kernel") != binding["kernel"]
        or wrapper.get("kernel_version") != binding["kernel_version"]
        or wrapper.get("bound_wrapper_sha256") != binding["bound_wrapper_sha256"]
        or wrapper.get("checkout_commit") != binding["checkout_commit"]
        or wrapper.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper.get("source_hashes") != runtime_source_hashes
        or wrapper.get("physical_prediction_maps_verified") != expected_validation
        or wrapper.get("physical_candidate_score_payloads_verified") != expected_validation
    ):
        raise ValueError("R3 wrapper-output audit mismatch")
    t4 = wrapper.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("R3 wrapper T4x2 evidence mismatch")
    cache = wrapper.get("cache", {})
    if (
        cache.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or cache.get("selector_cache_wrapper_audit_sha256") != CACHE_WRAPPER_AUDIT_SHA256
        or cache.get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("R3 wrapper cache evidence mismatch")

    if sha256_file(paths["checkpoint"]) != freeze.get("checkpoint_sha256"):
        raise ValueError("R3 checkpoint differs from prediction freeze")
    if sha256_file(paths["history"]) != freeze.get("training_history_sha256"):
        raise ValueError("R3 training history differs from prediction freeze")
    if sha256_file(paths["initial"]) != freeze.get("pretraining_identity_audit_sha256"):
        raise ValueError("R3 pretraining audit differs from prediction freeze")
    if paths["checkpoint"].stat().st_size <= 0:
        raise ValueError("R3 checkpoint is empty")
    _verify_history(paths["history"])
    initial = _verify_pretraining_audit(
        paths["initial"], expected_train=expected_train, expected_validation=expected_validation
    )
    validation = _verify_validation_evidence(
        root,
        freeze,
        expected_validation=expected_validation,
        expected_map_shape=expected_map_shape,
    )
    diagnostics = _verify_gt_blind_diagnostics(
        root, freeze, initial, expected_validation=expected_validation
    )
    if abs(
        float(validation["validation_absolute_count_probability_spearman"])
        - float(diagnostics["absolute_candidate_count_probability_spearman"])
    ) > 1.0e-12:
        raise ValueError("R3 prediction and diagnostic count associations differ")
    physical_bytes = sum(path.stat().st_size for path in (*paths.values(), launch_binding_path))
    return {
        "audit_id": "independent_mask_bag_critical_relation_r3_output_v1",
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
        "pretraining_identity": initial,
        "validation": validation,
        "gt_blind_diagnostics": diagnostics,
        "physical_output_bytes_verified": (
            physical_bytes
            + int(validation["physical_validation_evidence_bytes"])
            + int(diagnostics["physical_bytes"])
        ),
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
    result = audit_r3_output(
        args.output_root.resolve(),
        args.protocol.resolve(),
        args.launch_binding.resolve(),
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
