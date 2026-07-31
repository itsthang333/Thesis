from __future__ import annotations

"""GT-blind physical auditor for a frozen R2 affinity-residual output."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from audit_mask_bag_normal_prototype_r1_output import (
    _json,
    _require_safety,
    _verify_validation_evidence,
    sha256_file,
)


PROTOCOL_SHA256 = "3f28cc7187ad64f3755ae4c7a10bb380a0085d1733807dcf667c44d92d9f593d"
SOURCE_COMMIT = "c0e38628069ff3bedd4493c4ff004b75bd32e008"
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
AFFINITY_DIM = 24
EXPECTED_OBJECTIVE_CONFIG = {
    "bag_temperature": 0.20,
    "consistency_weight": 0.10,
    "residual_drift_weight": 0.001,
}
EXPECTED_TRAINING_CONFIG = {
    "epochs": 16,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "adapter_hidden_dim": 128,
    "seed": 42,
}
REQUIRED_RUNTIME_SOURCES = {
    "project/run_mask_bag_affinity_residual_arm.py",
    "project/models/mask_bag_affinity_residual_training.py",
    "project/models/mask_bag_affinity_features.py",
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
        raise ValueError("R2 launch binding contract mismatch")
    for field in ("kernel", "checkout_commit"):
        if not str(binding.get(field, "")).strip():
            raise ValueError(f"R2 launch binding omits {field}")
    _require_hex(binding.get("bound_wrapper_sha256"), name="bound wrapper SHA-256")
    source_hashes = binding.get("runtime_source_hashes")
    if not isinstance(source_hashes, dict) or not REQUIRED_RUNTIME_SOURCES <= set(source_hashes):
        raise ValueError("R2 launch binding omits required runtime sources")
    protocol_hashes = protocol.get("canonical_lf_source_hashes", {})
    for relative, expected in source_hashes.items():
        _require_hex(expected, name=f"runtime source {relative}")
        if protocol_hashes.get(relative) != expected:
            raise ValueError(f"R2 runtime source is not frozen by the protocol: {relative}")
    return {str(key): str(value) for key, value in source_hashes.items()}


def _verify_history(path: Path) -> None:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 16:
        raise ValueError("R2 history must contain exactly 16 fixed epochs")
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict) or int(row.get("epoch", -1)) != index:
            raise ValueError("R2 history epoch order mismatch")
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError("R2 history contains a non-finite value")


def audit_r2_output(
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
        raise ValueError("R2 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    binding = _json(launch_binding_path)
    runtime_source_hashes = _verify_launch_binding(binding, protocol)

    freeze_path = root / "prediction_freeze.json"
    run_manifest_path = root / "run_manifest.json"
    wrapper_audit_path = root / "wrapper_output_audit.json"
    checkpoint_path = root / "affinity_residual.pt"
    history_path = root / "training_history.json"
    for path in (
        freeze_path,
        run_manifest_path,
        wrapper_audit_path,
        checkpoint_path,
        history_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"R2 output is missing: {path}")

    freeze = _json(freeze_path)
    run_manifest = _json(run_manifest_path)
    wrapper_audit = _json(wrapper_audit_path)
    for name, payload in (
        ("prediction freeze", freeze),
        ("run manifest", run_manifest),
        ("wrapper output audit", wrapper_audit),
    ):
        _require_safety(payload, name=name)

    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("affinity_dim") != AFFINITY_DIM
        or freeze.get("validation_predictions") != expected_validation
        or freeze.get("training_labels") != "image_level_only"
        or freeze.get("epoch_selection") != "fixed_final_epoch_only"
    ):
        raise ValueError("R2 prediction-freeze provenance mismatch")
    if (
        run_manifest.get("run_id") != "btxrd_mask_bag_affinity_residual_r2_v1"
        or run_manifest.get("source_commit") != SOURCE_COMMIT
        or run_manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or run_manifest.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or run_manifest.get("objective_config") != EXPECTED_OBJECTIVE_CONFIG
        or run_manifest.get("training_config") != EXPECTED_TRAINING_CONFIG
        or run_manifest.get("validated_cache_records")
        != {"train": expected_train, "validation": expected_validation}
        or run_manifest.get("output_hashes") != freeze
    ):
        raise ValueError("R2 run-manifest contract mismatch")
    runtime = run_manifest.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("validation_scoring_workers") != 2
        or runtime.get("validation_shards") != [186, 185]
    ):
        raise ValueError("R2 T4x2 runtime contract mismatch")

    if (
        wrapper_audit.get("kernel") != binding["kernel"]
        or wrapper_audit.get("kernel_version") != binding["kernel_version"]
        or wrapper_audit.get("bound_wrapper_sha256")
        != binding["bound_wrapper_sha256"]
        or wrapper_audit.get("checkout_commit") != binding["checkout_commit"]
        or wrapper_audit.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper_audit.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper_audit.get("source_hashes") != runtime_source_hashes
        or wrapper_audit.get("physical_prediction_maps_verified")
        != expected_validation
        or wrapper_audit.get("physical_candidate_score_payloads_verified")
        != expected_validation
    ):
        raise ValueError("R2 wrapper-output audit mismatch")
    t4 = wrapper_audit.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("R2 wrapper T4x2 evidence mismatch")
    cache = wrapper_audit.get("cache", {})
    if (
        cache.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or cache.get("selector_cache_wrapper_audit_sha256")
        != CACHE_WRAPPER_AUDIT_SHA256
        or cache.get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("R2 wrapper cache evidence mismatch")

    if sha256_file(checkpoint_path) != freeze.get("checkpoint_sha256"):
        raise ValueError("R2 checkpoint differs from prediction freeze")
    if sha256_file(history_path) != freeze.get("training_history_sha256"):
        raise ValueError("R2 training history differs from prediction freeze")
    if checkpoint_path.stat().st_size <= 0:
        raise ValueError("R2 checkpoint is empty")
    _verify_history(history_path)
    validation = _verify_validation_evidence(
        root,
        freeze,
        expected_validation=expected_validation,
        expected_map_shape=expected_map_shape,
    )
    physical_bytes = sum(
        path.stat().st_size
        for path in (
            freeze_path,
            run_manifest_path,
            wrapper_audit_path,
            checkpoint_path,
            history_path,
            launch_binding_path,
        )
    )
    return {
        "audit_id": "independent_mask_bag_affinity_residual_r2_output_v1",
        "status": "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND",
        "kernel": binding["kernel"],
        "kernel_version": binding["kernel_version"],
        "bound_wrapper_sha256": binding["bound_wrapper_sha256"],
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "wrapper_output_audit_sha256": sha256_file(wrapper_audit_path),
        "launch_binding_sha256": sha256_file(launch_binding_path),
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "cache_freeze_sha256": CACHE_FREEZE_SHA256,
        "affinity_dim": AFFINITY_DIM,
        "validation": validation,
        "physical_output_bytes_verified": (
            physical_bytes + validation["physical_validation_evidence_bytes"]
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
    result = audit_r2_output(
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
