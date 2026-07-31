from __future__ import annotations

"""GT-blind physical auditor for a frozen S1 matched pooling pair."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from audit_mask_bag_normal_prototype_r1_output import (
    _close,
    _csv,
    _json,
    _require_safety,
    _safe_child,
    _sigmoid,
    _smooth_pool,
    _spearman,
    sha256_file,
)


PROTOCOL_SHA256 = "62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413"
SOURCE_COMMIT = "f3da1817ee3491f04e8c86335556762ebc675d8d"
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
POOL_MODES = ("standard", "family_balanced")
EXPECTED_MATCHED_VARIABLES = [
    "descriptor_cache",
    "frozen_baseline",
    "adapter_architecture",
    "adapter_initial_state",
    "batch_order",
    "optimizer",
    "epochs",
    "loss_weights",
    "validation_cohort",
]
REQUIRED_RUNTIME_SOURCES = {
    "project/run_mask_bag_family_balanced_pair.py",
    "project/models/mask_bag_pooling_residual_training.py",
    "project/models/mask_bag_relational_selector.py",
    "project/models/mask_bag_residual_objective.py",
    "project/models/rad_dino_mask_bag_mil.py",
    "project/run_mask_bag_normal_prototype_arm.py",
    "project/mae_reconstruction_io.py",
    "project/models/mask_bag_selector_cache_io.py",
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


def _family_balanced_pool(
    logits: np.ndarray,
    family_ids: np.ndarray,
    *,
    temperature: float = 0.20,
) -> float:
    values = np.asarray(logits, dtype=np.float64)
    families = np.asarray(family_ids, dtype=np.int64)
    if (
        values.ndim != 1
        or families.shape != values.shape
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(families < 0)
    ):
        raise ValueError("Family-balanced pool inputs are invalid")
    family_logits = np.asarray(
        [_smooth_pool(values[families == family], temperature) for family in np.unique(families)],
        dtype=np.float64,
    )
    return _smooth_pool(family_logits, temperature)


def _verify_launch_binding(
    binding: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, str]:
    if (
        binding.get("schema_version") != 1
        or binding.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or int(binding.get("kernel_version", 0)) < 1
    ):
        raise ValueError("S1 launch binding contract mismatch")
    for field in ("kernel", "checkout_commit"):
        if not str(binding.get(field, "")).strip():
            raise ValueError(f"S1 launch binding omits {field}")
    _require_hex(binding.get("bound_wrapper_sha256"), name="bound wrapper SHA-256")
    source_hashes = binding.get("runtime_source_hashes")
    if not isinstance(source_hashes, dict) or not REQUIRED_RUNTIME_SOURCES <= set(source_hashes):
        raise ValueError("S1 launch binding omits required runtime sources")
    protocol_hashes = protocol.get("scientific_source", {}).get(
        "canonical_lf_source_hashes", {}
    )
    for relative, expected in source_hashes.items():
        _require_hex(expected, name=f"runtime source {relative}")
        if protocol_hashes.get(relative) != expected:
            raise ValueError(f"S1 runtime source is not frozen by protocol: {relative}")
    return {str(key): str(value) for key, value in source_hashes.items()}


def _verify_history(path: Path) -> None:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 16:
        raise ValueError("S1 history must contain exactly 16 fixed epochs")
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict) or int(row.get("epoch", -1)) != index:
            raise ValueError("S1 history epoch order mismatch")
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError("S1 history contains a non-finite value")


def _verify_family_evidence(
    root: Path,
    expected_manifest_sha256: str,
    *,
    expected_validation: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    family_root = root / "candidate_families"
    manifest = family_root / "candidate_family_manifest.csv"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise ValueError("Candidate-family manifest differs from pair freeze")
    rows = _csv(manifest)
    by_id: dict[str, dict[str, Any]] = {}
    physical_bytes = manifest.stat().st_size
    for row in rows:
        image_id = row["image_id"]
        if image_id in by_id:
            raise ValueError("Candidate-family manifest contains duplicate image IDs")
        path = _safe_child(family_root, row["family_path"])
        if not path.is_file() or sha256_file(path) != row["family_sha256"]:
            raise ValueError(f"Candidate-family payload hash mismatch: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"schema_version", "candidate_indices", "family_ids"}:
                raise ValueError(f"Candidate-family payload fields mismatch: {image_id}")
            schema = int(payload["schema_version"])
            indices = payload["candidate_indices"]
            families = payload["family_ids"]
        if (
            schema != 1
            or indices.dtype != np.int64
            or families.dtype != np.int64
            or indices.ndim != 1
            or families.shape != indices.shape
            or len(indices) != int(row["candidate_count"])
            or len(indices) == 0
            or np.any(indices < 0)
            or np.any(np.diff(indices) <= 0)
            or np.any(families < 0)
            or len(np.unique(families)) != int(row["family_count"])
        ):
            raise ValueError(f"Candidate-family payload content mismatch: {image_id}")
        by_id[image_id] = {"row": row, "indices": indices, "family_ids": families}
        physical_bytes += path.stat().st_size
    if len(rows) != expected_validation or len(by_id) != expected_validation:
        raise ValueError("Candidate-family cohort mismatch")
    return by_id, physical_bytes


def _verify_arm_validation(
    root: Path,
    freeze: Mapping[str, Any],
    families_by_id: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    expected_validation: int,
    expected_map_shape: tuple[int, int],
) -> dict[str, object]:
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    if sha256_file(prediction_manifest) != freeze.get("prediction_manifest_sha256"):
        raise ValueError(f"{mode} prediction manifest differs from freeze")
    if sha256_file(score_manifest) != freeze.get("candidate_score_manifest_sha256"):
        raise ValueError(f"{mode} score manifest differs from freeze")
    predictions = _csv(prediction_manifest)
    scores = _csv(score_manifest)
    prediction_by_id = {row["image_id"]: row for row in predictions}
    score_by_id = {row["image_id"]: row for row in scores}
    if (
        len(predictions) != expected_validation
        or len(prediction_by_id) != expected_validation
        or len(scores) != expected_validation
        or len(score_by_id) != expected_validation
        or set(prediction_by_id) != set(score_by_id)
        or set(prediction_by_id) != set(families_by_id)
    ):
        raise ValueError(f"{mode} validation cohort mismatch")
    counts: list[int] = []
    probabilities: list[float] = []
    physical_bytes = prediction_manifest.stat().st_size + score_manifest.stat().st_size
    for image_id, prediction in prediction_by_id.items():
        score = score_by_id[image_id]
        family = families_by_id[image_id]
        family_row = family["row"]
        for field in (
            "group_id",
            "tumor",
            "candidate_payload_sha256",
            "candidate_count",
            "selected_candidate_index",
            "selected_candidate_logit",
        ):
            if prediction[field] != score[field]:
                raise ValueError(f"{mode} prediction/score mismatch: {image_id}/{field}")
        for field in ("group_id", "tumor", "candidate_payload_sha256", "candidate_count"):
            if prediction[field] != family_row[field]:
                raise ValueError(f"{mode} prediction/family mismatch: {image_id}/{field}")
        if prediction["candidate_logit_tta"] != "mean_original_aligned_horizontal_flip":
            raise ValueError(f"{mode} candidate TTA contract mismatch: {image_id}")
        score_path = _safe_child(root / "candidate_scores", score["score_path"])
        if not score_path.is_file() or sha256_file(score_path) != score["score_sha256"]:
            raise ValueError(f"{mode} score payload hash mismatch: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            if set(payload.files) != {"schema_version", "candidate_indices", "candidate_logits"}:
                raise ValueError(f"{mode} score payload fields mismatch: {image_id}")
            schema = int(payload["schema_version"])
            indices = payload["candidate_indices"]
            logits = payload["candidate_logits"]
        family_indices = np.asarray(family["indices"])
        family_ids = np.asarray(family["family_ids"])
        if (
            schema != 1
            or indices.dtype != np.int64
            or logits.dtype != np.float32
            or indices.ndim != 1
            or logits.shape != indices.shape
            or not np.array_equal(indices, family_indices)
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"{mode} score/family payload content mismatch: {image_id}")
        winner = int(np.argmax(logits))
        if int(indices[winner]) != int(prediction["selected_candidate_index"]):
            raise ValueError(f"{mode} selected candidate mismatch: {image_id}")
        _close(
            logits[winner],
            prediction["selected_candidate_logit"],
            name=f"{mode} selected candidate logit",
        )
        expected_logit = (
            _smooth_pool(logits)
            if mode == "standard"
            else _family_balanced_pool(logits, family_ids)
        )
        _close(prediction["bag_logit"], expected_logit, name=f"{mode} bag logit", atol=2.0e-6)
        probability = float(prediction["bag_probability"])
        _close(probability, _sigmoid(expected_logit), name=f"{mode} bag probability", atol=1.0e-7)
        map_path = _safe_child(root / "predictions", prediction["map_path"])
        if not map_path.is_file() or sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"{mode} prediction-map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != expected_map_shape
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"{mode} prediction-map content mismatch: {image_id}")
        nonzero = values > 0
        _close(
            nonzero.mean(),
            prediction["selected_area_ratio"],
            name=f"{mode} selected area ratio",
            atol=1.0 / np.prod(expected_map_shape) + 1.0e-7,
        )
        if nonzero.any() and not np.allclose(
            values[nonzero].astype(np.float32), np.float16(probability), atol=0.0, rtol=0.0
        ):
            raise ValueError(f"{mode} map is not selected mask times bag probability: {image_id}")
        counts.append(len(indices))
        probabilities.append(probability)
        physical_bytes += score_path.stat().st_size + map_path.stat().st_size
    return {
        "physical_validation_maps_verified": expected_validation,
        "physical_candidate_score_payloads_verified": expected_validation,
        "absolute_count_probability_spearman": abs(_spearman(counts, probabilities)),
        "physical_validation_evidence_bytes": physical_bytes,
    }


def audit_s1_pair_output(
    root: Path,
    protocol_path: Path,
    launch_binding_path: Path,
    *,
    expected_validation: int = 371,
    expected_map_shape: tuple[int, int] = (320, 320),
) -> dict[str, object]:
    _verify_helper_source()
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S1 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    binding = _json(launch_binding_path)
    runtime_source_hashes = _verify_launch_binding(binding, protocol)
    pair_path = root / "pair_prediction_freeze.json"
    run_path = root / "run_manifest.json"
    wrapper_path = root / "wrapper_output_audit.json"
    initial_state_path = root / "matched_initial_state.pt"
    for path in (pair_path, run_path, wrapper_path, initial_state_path):
        if not path.is_file():
            raise FileNotFoundError(f"S1 output is missing: {path}")
    pair = _json(pair_path)
    run = _json(run_path)
    wrapper = _json(wrapper_path)
    for name, payload in (("pair freeze", pair), ("run manifest", run), ("wrapper audit", wrapper)):
        _require_safety(payload, name=name)
    family_manifest_sha = _require_hex(
        pair.get("candidate_family_manifest_sha256"),
        name="candidate-family manifest SHA-256",
    )
    if (
        pair.get("run_id") != "btxrd_mask_bag_family_balanced_s1_pair_v1"
        or pair.get("source_commit") != SOURCE_COMMIT
        or pair.get("protocol_sha256") != PROTOCOL_SHA256
        or pair.get("split_sha256") != SPLIT_SHA256
        or pair.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or pair.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or pair.get("matched_variables") != EXPECTED_MATCHED_VARIABLES
        or pair.get("sole_changed_variable") != "standard_vs_family_balanced_bag_pool"
        or float(pair.get("cross_device_initial_candidate_logit_tolerance", -1.0)) != 5.0e-6
        or float(pair.get("cross_device_initial_candidate_logit_max_delta", math.inf)) > 5.0e-6
        or sha256_file(initial_state_path) != pair.get("initial_state_sha256")
        or set(pair.get("arms", {})) != set(POOL_MODES)
    ):
        raise ValueError("S1 pair-freeze contract mismatch")
    if (
        run.get("pair_prediction_freeze_sha256") != sha256_file(pair_path)
        or any(run.get(key) != value for key, value in pair.items())
        or run.get("validated_cache_records") != {"train": 2981, "validation": expected_validation}
    ):
        raise ValueError("S1 run-manifest contract mismatch")
    runtime = run.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("parallel_training_workers") != 2
        or set(runtime.get("device_assignment", {})) != set(POOL_MODES)
    ):
        raise ValueError("S1 T4x2 runtime contract mismatch")
    if (
        wrapper.get("kernel") != binding["kernel"]
        or wrapper.get("kernel_version") != binding["kernel_version"]
        or wrapper.get("bound_wrapper_sha256") != binding["bound_wrapper_sha256"]
        or wrapper.get("checkout_commit") != binding["checkout_commit"]
        or wrapper.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper.get("source_hashes") != runtime_source_hashes
        or wrapper.get("physical_prediction_maps_verified") != 2 * expected_validation
        or wrapper.get("physical_candidate_score_payloads_verified") != 2 * expected_validation
        or wrapper.get("physical_candidate_family_payloads_verified") != expected_validation
    ):
        raise ValueError("S1 wrapper-output audit mismatch")
    t4 = wrapper.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("S1 wrapper T4x2 evidence mismatch")
    cache = wrapper.get("cache", {})
    if (
        cache.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or cache.get("selector_cache_wrapper_audit_sha256") != CACHE_WRAPPER_AUDIT_SHA256
        or cache.get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("S1 wrapper cache evidence mismatch")
    families_by_id, family_bytes = _verify_family_evidence(
        root,
        family_manifest_sha,
        expected_validation=expected_validation,
    )
    arm_audits: dict[str, object] = {}
    fixed_bytes = sum(
        path.stat().st_size
        for path in (pair_path, run_path, wrapper_path, initial_state_path, launch_binding_path)
    )
    for mode in POOL_MODES:
        arm_root = root / mode
        freeze_path = arm_root / "prediction_freeze.json"
        checkpoint_path = arm_root / "descriptor_residual.pt"
        history_path = arm_root / "training_history.json"
        for path in (freeze_path, checkpoint_path, history_path):
            if not path.is_file():
                raise FileNotFoundError(f"S1 {mode} output is missing: {path}")
        freeze = _json(freeze_path)
        _require_safety(freeze, name=f"{mode} freeze")
        expected_pair_arm = {**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}
        if pair["arms"][mode] != expected_pair_arm:
            raise ValueError(f"S1 {mode} freeze differs from pair freeze")
        if (
            freeze.get("arm") != mode
            or freeze.get("source_commit") != SOURCE_COMMIT
            or freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("split_sha256") != SPLIT_SHA256
            or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
            or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
            or freeze.get("initial_state_sha256") != pair.get("initial_state_sha256")
            or freeze.get("candidate_family_manifest_sha256") != family_manifest_sha
            or freeze.get("validation_predictions") != expected_validation
            or freeze.get("training_labels") != "image_level_only"
            or freeze.get("epoch_selection") != "fixed_final_epoch_only"
            or sha256_file(checkpoint_path) != freeze.get("checkpoint_sha256")
            or sha256_file(history_path) != freeze.get("training_history_sha256")
            or checkpoint_path.stat().st_size <= 0
        ):
            raise ValueError(f"S1 {mode} prediction-freeze provenance mismatch")
        _verify_history(history_path)
        validation = _verify_arm_validation(
            arm_root,
            freeze,
            families_by_id,
            mode=mode,
            expected_validation=expected_validation,
            expected_map_shape=expected_map_shape,
        )
        arm_audits[mode] = validation
        fixed_bytes += freeze_path.stat().st_size + checkpoint_path.stat().st_size + history_path.stat().st_size
    return {
        "audit_id": "independent_mask_bag_family_balanced_s1_pair_output_v1",
        "status": "MATCHED_PAIR_PREDICTIONS_PHYSICALLY_VERIFIED_GT_BLIND",
        "kernel": binding["kernel"],
        "kernel_version": binding["kernel_version"],
        "bound_wrapper_sha256": binding["bound_wrapper_sha256"],
        "pair_prediction_freeze_sha256": sha256_file(pair_path),
        "run_manifest_sha256": sha256_file(run_path),
        "wrapper_output_audit_sha256": sha256_file(wrapper_path),
        "launch_binding_sha256": sha256_file(launch_binding_path),
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_family_manifest_sha256": family_manifest_sha,
        "physical_candidate_family_payloads_verified": expected_validation,
        "arms": arm_audits,
        "physical_output_bytes_verified": fixed_bytes + family_bytes + sum(
            int(arm["physical_validation_evidence_bytes"]) for arm in arm_audits.values()
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
    result = audit_s1_pair_output(
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
