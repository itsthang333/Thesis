from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen S7 prediction pair."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_global_local_instance import (
    GlobalLocalInstanceConfig,
    GlobalLocalInstanceResidual,
)
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import load_selector_cache_record
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


EXPERIMENT_ID = "EXP-20260802-codex-s7-global-local-instance-v1"
RUN_ID = "btxrd_mask_bag_global_local_instance_s7_pair_v1"
ARMS = ("geometry_v3_identity", "global_local_instance")
REPRODUCTION_ATOL = 5.0e-5


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"S7 output path is absolute: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"S7 output path escapes root: {relative}")
    return resolved


def _safety(payload: Mapping[str, object], name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S7 {name} safety boundary failed")


def _smooth_pool(values: np.ndarray, temperature: float = 0.20) -> float:
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or len(logits) == 0 or not np.isfinite(logits).all():
        raise ValueError("S7 candidate logits must be finite and nonempty")
    scaled = logits / temperature
    maximum = float(scaled.max())
    return float(
        temperature
        * (
            maximum
            + math.log(float(np.exp(scaled - maximum).sum()))
            - math.log(len(logits))
        )
    )


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _close(actual: object, expected: object, name: str, atol: float) -> None:
    left = float(actual)
    right = float(expected)
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > atol:
        raise ValueError(f"S7 {name} differs: {left} versus {right}")


def _serialize_prediction_map(mask: np.ndarray, probability: float) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if not np.all(np.logical_or(values == 0.0, values == 1.0)):
        raise ValueError("S7 selected mask is not binary")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("S7 bag probability is invalid")
    return (values * probability).astype(np.float16)


def _target_digest(
    image_ids: Sequence[str],
    targets: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    if not (len(image_ids) == len(targets) == len(weights)):
        raise ValueError("S7 target digest arrays do not align")
    for image_id, target, weight in zip(image_ids, targets, weights):
        encoded = str(image_id).encode("utf-8")
        target32 = np.asarray(target, dtype="<f4")
        weight64 = np.asarray(weight, dtype="<f8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(target32.shape[0].to_bytes(4, "little"))
        digest.update(target32.tobytes(order="C"))
        digest.update(weight64.tobytes(order="C"))
    return digest.hexdigest()


def _family_weights(family_ids: np.ndarray) -> np.ndarray:
    families = np.asarray(family_ids).reshape(-1)
    if families.size == 0:
        raise ValueError("S7 family bag is empty")
    unique = sorted(set(str(value) for value in families.tolist()))
    result = np.zeros(families.size, dtype=np.float64)
    normalized = np.asarray([str(value) for value in families], dtype=object)
    for family in unique:
        members = np.flatnonzero(normalized == family)
        result[members] = 1.0 / (len(unique) * len(members))
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("S7 independent family weights do not sum to one")
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _independent_targets(
    logits: Sequence[np.ndarray],
    labels: Sequence[int],
    family_ids: Sequence[np.ndarray],
    target_mass: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float | int]]:
    weights = [_family_weights(values) for values in family_ids]
    positive = [index for index, label in enumerate(labels) if int(label) == 1]
    if not positive:
        raise ValueError("S7 target audit has no positive bags")
    flat_logits = np.concatenate(
        [np.asarray(logits[index], dtype=np.float64) for index in positive]
    )
    flat_weights = np.concatenate(
        [weights[index] / len(positive) for index in positive]
    )
    lower = -128.0 - float(np.max(np.abs(flat_logits)))
    upper = 128.0 + float(np.max(np.abs(flat_logits)))
    for _ in range(96):
        midpoint = 0.5 * (lower + upper)
        realized = float(
            np.dot(flat_weights, _stable_sigmoid(flat_logits + midpoint))
        )
        if realized < target_mass:
            lower = midpoint
        else:
            upper = midpoint
    bias = 0.5 * (lower + upper)
    projected = _stable_sigmoid(flat_logits + bias).astype(np.float32)
    projected_mass = float(
        np.dot(flat_weights, projected.astype(np.float64))
    )
    targets = [np.zeros(len(values), dtype=np.float32) for values in logits]
    cursor = 0
    for bag_index in positive:
        count = len(logits[bag_index])
        targets[bag_index] = projected[cursor : cursor + count].copy()
        cursor += count
        targets[bag_index][int(np.argmax(logits[bag_index]))] = np.float32(1.0)
    realized_after = float(
        sum(
            np.dot(weights[index], targets[index].astype(np.float64))
            for index in positive
        )
        / len(positive)
    )
    return targets, weights, {
        "projection_bias": bias,
        "projected_mass_before_local_float32": projected_mass,
        "realized_mass_after_local": realized_after,
        "locally_forced_candidates": len(positive),
    }


def _load_baseline(
    path: Path,
    *,
    expected_sha256: str,
    source_commit: str,
    protocol_sha256: str,
    split_sha256: str,
) -> RadDinoMaskBagMIL:
    if sha256_file(path) != expected_sha256:
        raise ValueError("S7 baseline checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "baseline checkpoint")
    if (
        checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("split_sha256") != split_sha256
    ):
        raise ValueError("S7 baseline checkpoint provenance mismatch")
    model = RadDinoMaskBagMIL(MaskBagMILConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval()


def _load_primary(
    path: Path,
    *,
    source_commit: str,
    protocol_sha256: str,
    split_sha256: str,
    cache_freeze_sha256: str,
    baseline_checkpoint_sha256: str,
) -> tuple[GlobalLocalInstanceResidual, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "primary checkpoint")
    if (
        checkpoint.get("experiment_id") != EXPERIMENT_ID
        or checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("split_sha256") != split_sha256
        or checkpoint.get("selector_cache_freeze_sha256") != cache_freeze_sha256
        or checkpoint.get("baseline_checkpoint_sha256")
        != baseline_checkpoint_sha256
        or checkpoint.get("training_labels") != "binary_image_level_only"
    ):
        raise ValueError("S7 primary checkpoint provenance mismatch")
    config = GlobalLocalInstanceConfig(**checkpoint["model_config"])
    model = GlobalLocalInstanceResidual(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval(), checkpoint


@torch.inference_mode()
def _reproduce_scores(
    payload: Mapping[str, Any],
    baseline: RadDinoMaskBagMIL,
    primary: GlobalLocalInstanceResidual,
) -> tuple[np.ndarray, np.ndarray, float]:
    descriptors = torch.from_numpy(
        np.asarray(payload["descriptors"], dtype=np.float32)
    )[None]
    flipped = torch.from_numpy(
        np.asarray(payload["flipped_descriptors"], dtype=np.float32)
    )[None]
    valid = torch.ones(descriptors.shape[:2], dtype=torch.bool)
    base_original, _ = baseline.score_descriptors(descriptors, valid)
    base_flipped, _ = baseline.score_descriptors(flipped, valid)
    residual_original = primary(descriptors, valid)
    residual_flipped = primary(flipped, valid)
    base = (0.5 * (base_original + base_flipped))[0].numpy().astype(np.float32)
    selected = (
        0.5
        * (
            (base_original + residual_original)
            + (base_flipped + residual_flipped)
        )
    )[0].numpy().astype(np.float32)
    return base, selected, _smooth_pool(base)


def _audit_target_snapshots(
    output_root: Path,
    split_path: Path,
    split_sha256: str,
    cache_root: Path,
    cache_manifest_rows: Sequence[Mapping[str, str]],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_rows = load_split_rows_without_annotations(
        split_path, expected_sha256=split_sha256, split="train"
    )
    if len(train_rows) != 2981:
        raise ValueError("S7 target audit train cohort mismatch")
    cache_by_id = {
        row["image_id"]: row for row in cache_manifest_rows if row["split"] == "train"
    }
    if len(cache_by_id) != 2981:
        raise ValueError("S7 target audit cache cohort mismatch")
    image_ids: list[str] = []
    labels: list[int] = []
    families: list[np.ndarray] = []
    candidate_counts: list[int] = []
    for split_row in train_rows:
        image_id = split_row["image_id"]
        cache_row = cache_by_id[image_id]
        payload = load_selector_cache_record(
            cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=False,
        )
        image_ids.append(image_id)
        labels.append(int(split_row["tumor"]))
        family = np.asarray(payload["family_ids"]).reshape(-1)
        families.append(family)
        candidate_counts.append(len(family))

    target_root = output_root / "target_snapshots"
    manifest_path = target_root / "target_snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, list)
        or len(manifest) != 40
        or [int(row["epoch_index"]) for row in manifest] != list(range(40))
    ):
        raise ValueError("S7 target snapshot manifest differs from 40 epochs")
    maximum_target_delta = 0.0
    maximum_weight_delta = 0.0
    bytes_verified = 0
    for epoch_index, (manifest_row, history_row) in enumerate(zip(manifest, history)):
        snapshot_path = _safe_child(target_root, str(manifest_row["snapshot_path"]))
        if sha256_file(snapshot_path) != manifest_row["snapshot_sha256"]:
            raise ValueError(f"S7 target snapshot hash mismatch: epoch {epoch_index}")
        with np.load(snapshot_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "schema_version",
                "epoch_index",
                "image_ids",
                "labels",
                "offsets",
                "current_logits",
                "soft_targets",
                "candidate_weights",
            }:
                raise ValueError(f"S7 target snapshot schema mismatch: {epoch_index}")
            if int(payload["schema_version"][0]) != 1 or int(
                payload["epoch_index"][0]
            ) != epoch_index:
                raise ValueError(f"S7 target snapshot identity mismatch: {epoch_index}")
            saved_ids = payload["image_ids"].tolist()
            saved_labels = payload["labels"].astype(np.int64).tolist()
            offsets = payload["offsets"].astype(np.int64)
            flat_logits = payload["current_logits"].astype(np.float32)
            flat_targets = payload["soft_targets"].astype(np.float32)
            flat_weights = payload["candidate_weights"].astype(np.float64)
        if saved_ids != image_ids or saved_labels != labels:
            raise ValueError(f"S7 target snapshot cohort differs: epoch {epoch_index}")
        expected_offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(candidate_counts))
        )
        if not np.array_equal(offsets, expected_offsets) or not (
            len(flat_logits) == len(flat_targets) == len(flat_weights) == offsets[-1]
        ):
            raise ValueError(f"S7 target snapshot offsets differ: epoch {epoch_index}")
        logits = [
            flat_logits[offsets[index] : offsets[index + 1]]
            for index in range(2981)
        ]
        saved_targets = [
            flat_targets[offsets[index] : offsets[index + 1]]
            for index in range(2981)
        ]
        saved_weights = [
            flat_weights[offsets[index] : offsets[index + 1]]
            for index in range(2981)
        ]
        fraction = min(epoch_index / 20.0, 1.0)
        target_mass = 0.50 + fraction * (0.15 - 0.50)
        targets, weights, diagnostics = _independent_targets(
            logits, labels, families, target_mass
        )
        target_delta = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(saved_targets, targets)
        )
        weight_delta = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(saved_weights, weights)
        )
        maximum_target_delta = max(maximum_target_delta, target_delta)
        maximum_weight_delta = max(maximum_weight_delta, weight_delta)
        if target_delta != 0.0 or weight_delta != 0.0:
            raise ValueError(f"S7 target projection differs: epoch {epoch_index}")
        digest = _target_digest(image_ids, targets, weights)
        target_history = history_row.get("target", {})
        if (
            manifest_row.get("target_sha256") != digest
            or target_history.get("target_sha256") != digest
            or int(history_row.get("epoch", -1)) != epoch_index + 1
            or int(target_history.get("epoch_index", -1)) != epoch_index
            or int(manifest_row.get("locally_forced_candidates", -1))
            != diagnostics["locally_forced_candidates"]
        ):
            raise ValueError(f"S7 target history/hash mismatch: epoch {epoch_index}")
        _close(
            manifest_row["target_positive_mass"],
            target_mass,
            f"target mass epoch {epoch_index}",
            1.0e-15,
        )
        _close(
            manifest_row["realized_mass_after_local"],
            diagnostics["realized_mass_after_local"],
            f"local mass epoch {epoch_index}",
            1.0e-10,
        )
        bytes_verified += snapshot_path.stat().st_size
    return {
        "snapshots": 40,
        "train_records": 2981,
        "maximum_target_reproduction_delta": maximum_target_delta,
        "maximum_weight_reproduction_delta": maximum_weight_delta,
        "snapshot_bytes_verified": bytes_verified,
        "target_snapshot_manifest_sha256": sha256_file(manifest_path),
    }


def audit_output(
    output_root: Path,
    protocol_path: Path,
    binding_path: Path,
    split_path: Path,
    cache_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    protocol_sha256 = sha256_file(protocol_path)
    protocol = _json(protocol_path)
    binding = _json(binding_path)
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != protocol_sha256
        or binding.get("scientific_source_commit")
        != protocol.get("scientific_source", {}).get("commit")
    ):
        raise ValueError("S7 protocol/binding mismatch")
    source_commit = str(binding["scientific_source_commit"])
    inputs = protocol.get("frozen_inputs", {})
    split_sha256 = str(inputs.get("split_sha256"))
    cache_freeze_sha256 = str(inputs.get("selector_cache_freeze_sha256"))
    cache_manifest_sha256 = str(inputs.get("selector_cache_manifest_sha256"))
    baseline_checkpoint_sha256 = str(inputs.get("baseline_checkpoint_sha256"))
    baseline_source_commit = str(inputs.get("baseline_source_commit"))
    baseline_protocol_sha256 = str(inputs.get("baseline_protocol_sha256"))
    if sha256_file(split_path) != split_sha256:
        raise ValueError("S7 split SHA-256 mismatch")
    if sha256_file(cache_root / "selector_cache_freeze.json") != cache_freeze_sha256:
        raise ValueError("S7 selector-cache freeze mismatch")
    if sha256_file(cache_root / "selector_cache_manifest.csv") != cache_manifest_sha256:
        raise ValueError("S7 selector-cache manifest mismatch")
    cache_freeze = _json(cache_root / "selector_cache_freeze.json")
    _safety(cache_freeze, "selector cache")
    cache_manifest_rows = _rows(cache_root / "selector_cache_manifest.csv")

    history_path = output_root / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 40
        or [int(row["epoch"]) for row in history] != list(range(1, 41))
    ):
        raise ValueError("S7 training history differs from 40 fixed epochs")
    for row in history:
        for key in ("total", "instance", "consistency", "drift"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"S7 non-finite training history: {key}")
    identity_path = output_root / "pretraining_identity_audit.json"
    identity = _json(identity_path)
    _safety(identity, "pretraining identity")
    for cohort, expected in (("train", 2981), ("validation", 371)):
        section = identity.get(cohort, {})
        if (
            section.get("records") != expected
            or section.get("exact_candidate_score_records") != expected
            or section.get("exact_selected_index_records") != expected
            or float(section.get("maximum_candidate_score_delta", 1.0)) != 0.0
        ):
            raise ValueError(f"S7 {cohort} zero-initialization identity mismatch")

    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    _safety(pair, "prediction pair")
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("run_id") != RUN_ID
        or pair.get("source_commit") != source_commit
        or pair.get("protocol_sha256") != protocol_sha256
        or pair.get("sole_changed_variable")
        != "global_local_all_instance_selector_residual"
        or pair.get("accepted_bag_probability_preserved") is not True
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("diagnostics_block_prediction_freeze") is not False
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S7 prediction-pair contract mismatch")
    run_manifest = _json(output_root / "run_manifest.json")
    _safety(run_manifest, "run manifest")
    runtime = run_manifest.get("runtime", {})
    if (
        run_manifest.get("run_id") != RUN_ID
        or run_manifest.get("prediction_pair_freeze_sha256") != sha256_file(pair_path)
        or runtime.get("cuda_device_count") != 2
        or len(runtime.get("scoring_device_evidence", [])) != 2
        or {int(row["device_index"]) for row in runtime["scoring_device_evidence"]}
        != {0, 1}
        or not all("T4" in str(name) for name in runtime.get("cuda_device_names", []))
    ):
        raise ValueError("S7 runtime/T4x2 evidence mismatch")

    checkpoint_path = output_root / "global_local_instance_residual.pt"
    primary, checkpoint = _load_primary(
        checkpoint_path,
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        split_sha256=split_sha256,
        cache_freeze_sha256=cache_freeze_sha256,
        baseline_checkpoint_sha256=baseline_checkpoint_sha256,
    )
    expected_training = protocol.get("training", {})
    if checkpoint.get("training_config") != {
        "epochs": expected_training.get("epochs"),
        "batch_size": expected_training.get("batch_size"),
        "learning_rate": expected_training.get("learning_rate"),
        "weight_decay": expected_training.get("weight_decay"),
        "seed": expected_training.get("seed"),
    }:
        raise ValueError("S7 checkpoint training recipe differs from protocol")
    baseline_freeze = baseline_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze) != inputs.get("baseline_prediction_freeze_sha256"):
        raise ValueError("S7 accepted baseline freeze differs")
    baseline = _load_baseline(
        baseline_root / "rad_dino_mask_bag_mil.pt",
        expected_sha256=baseline_checkpoint_sha256,
        source_commit=baseline_source_commit,
        protocol_sha256=baseline_protocol_sha256,
        split_sha256=split_sha256,
    )

    target_audit = _audit_target_snapshots(
        output_root,
        split_path,
        split_sha256,
        cache_root,
        cache_manifest_rows,
        history,
    )
    target_manifest_path = (
        output_root / "target_snapshots" / "target_snapshot_manifest.json"
    )
    if (
        run_manifest.get("target_snapshot_manifest_sha256")
        != sha256_file(target_manifest_path)
    ):
        raise ValueError("S7 run/target manifest mismatch")

    diagnostic_path = output_root / "gt_blind_diagnostics.csv"
    diagnostic_summary_path = output_root / "gt_blind_diagnostic_summary.json"
    diagnostic_rows = {row["image_id"]: row for row in _rows(diagnostic_path)}
    diagnostic_summary = _json(diagnostic_summary_path)
    _safety(diagnostic_summary, "diagnostic summary")
    if len(diagnostic_rows) != 371:
        raise ValueError("S7 diagnostic cohort mismatch")
    arm_prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    arm_score_rows: dict[str, dict[str, dict[str, str]]] = {}
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arms"][arm]:
            raise ValueError(f"S7 {arm} freeze SHA-256 mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"{arm} freeze")
        if (
            freeze.get("arm") != arm
            or freeze.get("validation_predictions") != 371
            or freeze.get("primary_checkpoint_sha256") != sha256_file(checkpoint_path)
            or freeze.get("training_history_sha256") != sha256_file(history_path)
            or freeze.get("target_snapshot_manifest_sha256")
            != sha256_file(target_manifest_path)
            or freeze.get("pretraining_identity_audit_sha256")
            != sha256_file(identity_path)
            or freeze.get("gt_blind_diagnostics_sha256") != sha256_file(diagnostic_path)
            or freeze.get("gt_blind_diagnostic_summary_sha256")
            != sha256_file(diagnostic_summary_path)
            or freeze.get("accepted_bag_probability_preserved") is not True
        ):
            raise ValueError(f"S7 {arm} freeze provenance mismatch")
        arm_prediction_rows[arm] = {
            row["image_id"]: row
            for row in _rows(
                output_root / arm / "predictions" / "prediction_manifest.csv"
            )
        }
        arm_score_rows[arm] = {
            row["image_id"]: row
            for row in _rows(
                output_root / arm / "candidate_scores" / "candidate_score_manifest.csv"
            )
        }
        if len(arm_prediction_rows[arm]) != 371 or len(arm_score_rows[arm]) != 371:
            raise ValueError(f"S7 {arm} physical output cohort mismatch")

    val_rows = load_split_rows_without_annotations(
        split_path, expected_sha256=split_sha256, split="val"
    )
    cache_by_id = {
        row["image_id"]: row for row in cache_manifest_rows if row["split"] == "val"
    }
    if len(val_rows) != 371 or len(cache_by_id) != 371:
        raise ValueError("S7 validation/cache cohort mismatch")
    maximum_score_delta = {arm: 0.0 for arm in ARMS}
    maximum_map_delta = {arm: 0.0 for arm in ARMS}
    physical_bytes = 0
    changed = 0
    for split_row in val_rows:
        image_id = split_row["image_id"]
        cache_row = cache_by_id[image_id]
        payload = load_selector_cache_record(
            cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        base_scores, primary_scores, bag_logit = _reproduce_scores(
            payload, baseline, primary
        )
        expected_by_arm = {
            "geometry_v3_identity": base_scores,
            "global_local_instance": primary_scores,
        }
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        masks = unpack_candidate_masks(payload["packed_masks"]).astype(np.float32)
        diagnostic = diagnostic_rows[image_id]
        if (
            int(diagnostic["tumor"]) != int(split_row["tumor"])
            or int(diagnostic["candidate_count"]) != len(indices)
            or int(diagnostic["base_selected_local_index"]) != int(np.argmax(base_scores))
            or int(diagnostic["primary_selected_local_index"])
            != int(np.argmax(primary_scores))
        ):
            raise ValueError(f"S7 diagnostics differ: {image_id}")
        selected: dict[str, int] = {}
        arm_probabilities: dict[str, float] = {}
        for arm in ARMS:
            score_root = output_root / arm / "candidate_scores"
            score_row = arm_score_rows[arm][image_id]
            score_path = _safe_child(score_root, score_row["score_path"])
            if sha256_file(score_path) != score_row["score_sha256"]:
                raise ValueError(f"S7 {arm} score hash mismatch: {image_id}")
            with np.load(score_path, allow_pickle=False) as score_payload:
                if set(score_payload.files) != {
                    "schema_version",
                    "candidate_indices",
                    "candidate_logits",
                } or int(score_payload["schema_version"]) != 1:
                    raise ValueError(f"S7 {arm} score schema mismatch: {image_id}")
                saved_indices = score_payload["candidate_indices"]
                saved_scores = score_payload["candidate_logits"]
            if not np.array_equal(saved_indices, indices):
                raise ValueError(f"S7 {arm} candidate order mismatch: {image_id}")
            delta = float(np.max(np.abs(saved_scores - expected_by_arm[arm])))
            maximum_score_delta[arm] = max(maximum_score_delta[arm], delta)
            if delta > REPRODUCTION_ATOL:
                raise ValueError(f"S7 {arm} score reproduction mismatch: {image_id}")
            winner = int(np.argmax(saved_scores))
            if winner != int(np.argmax(expected_by_arm[arm])):
                raise ValueError(f"S7 {arm} winner reproduction mismatch: {image_id}")
            selected[arm] = winner
            prediction = arm_prediction_rows[arm][image_id]
            if (
                int(prediction["selected_candidate_index"]) != int(indices[winner])
                or prediction["candidate_payload_sha256"]
                != cache_row["candidate_payload_sha256"]
            ):
                raise ValueError(f"S7 {arm} prediction identity mismatch: {image_id}")
            _close(prediction["bag_logit"], bag_logit, f"bag logit {arm}/{image_id}", REPRODUCTION_ATOL)
            probability = _sigmoid(bag_logit)
            _close(
                prediction["bag_probability"],
                probability,
                f"bag probability {arm}/{image_id}",
                REPRODUCTION_ATOL,
            )
            arm_probabilities[arm] = float(prediction["bag_probability"])
            map_path = _safe_child(
                output_root / arm / "predictions", prediction["map_path"]
            )
            if sha256_file(map_path) != prediction["map_sha256"]:
                raise ValueError(f"S7 {arm} map hash mismatch: {image_id}")
            saved_map = np.load(map_path, allow_pickle=False)
            expected_map = _serialize_prediction_map(masks[winner], probability)
            if saved_map.dtype != np.float16 or saved_map.shape != expected_map.shape:
                raise ValueError(f"S7 {arm} map schema mismatch: {image_id}")
            map_delta = float(
                np.max(
                    np.abs(saved_map.astype(np.float32) - expected_map.astype(np.float32))
                )
            )
            maximum_map_delta[arm] = max(maximum_map_delta[arm], map_delta)
            if map_delta != 0.0:
                raise ValueError(f"S7 {arm} map reproduction mismatch: {image_id}")
            physical_bytes += score_path.stat().st_size + map_path.stat().st_size
        if arm_probabilities[ARMS[0]] != arm_probabilities[ARMS[1]]:
            raise ValueError(f"S7 bag probability changed across arms: {image_id}")
        changed += int(selected[ARMS[0]] != selected[ARMS[1]])
    if (
        diagnostic_summary.get("changed_selection_count") != changed
        or abs(float(diagnostic_summary["changed_selection_fraction"]) - changed / 371.0)
        > 1.0e-12
        or diagnostic_summary.get("bag_probability_changed_from_accepted_baseline")
        is not False
    ):
        raise ValueError("S7 diagnostic summary differs")
    return {
        "audit_id": "independent_mask_bag_global_local_instance_s7_output_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_PASS",
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha256,
        "prediction_pair_freeze_sha256": sha256_file(pair_path),
        "validation_records": 371,
        "physical_candidate_score_payloads": 742,
        "physical_prediction_maps": 742,
        "verified_score_map_bytes": physical_bytes,
        "maximum_candidate_logit_reproduction_delta": maximum_score_delta,
        "maximum_prediction_map_reproduction_delta": maximum_map_delta,
        "target_projection_audit": target_audit,
        "changed_selection_count": changed,
        "changed_selection_fraction": changed / 371.0,
        "accepted_bag_probability_preserved": True,
        "diagnostics_used_for_model_selection": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_output(
        args.output_root,
        args.protocol,
        args.binding,
        args.split,
        args.cache_root,
        args.baseline_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
