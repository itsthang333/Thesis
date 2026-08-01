from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen N1 output.

This source deliberately does not import the N1 runner, its scientific normal-
anomaly primitive, the segmentation evaluator, or any BTXRD dataset class.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260801-codex-n1-normal-only-direct-anomaly-v1"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-normal-anomaly-n1-v1"
SOURCE_COMMIT = "c7ba620ce4492485ba0faa6dd42998e267be872d"
PROTOCOL_SHA256 = "1112515d00ed9db80a05670404ad16127109844788a44018834ff82f452d9b7d"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_MANIFEST_SHA256 = "8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e"
CACHE_WRAPPER_AUDIT_SHA256 = (
    "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
)
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_MANIFEST_SHA256 = "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee"
BASELINE_CHECKPOINT_SHA256 = (
    "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
)
BASELINE_SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
BASELINE_PROTOCOL_SHA256 = "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
EXPECTED_COUNT_PROBABILITY_SPEARMAN = 0.48137777593654113
EXPECTED_TRAIN = 2981
EXPECTED_NORMAL = 1493
EXPECTED_VALIDATION = 371
EXPECTED_DESCRIPTOR_DIM = 1156
EXPECTED_PROTOTYPES = 32
EXPECTED_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    if child.is_absolute():
        raise ValueError(f"N1 output contains an absolute path: {relative}")
    resolved_root = root.resolve()
    resolved = (root / child).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"N1 output path escapes its root: {relative}")
    return resolved


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _close(actual: object, expected: object, *, name: str, atol: float = 1.0e-7) -> None:
    if abs(_finite(actual, name=name) - _finite(expected, name=name)) > atol:
        raise ValueError(f"{name} differs: {actual} versus {expected}")


def _require_safety(payload: Mapping[str, object], *, name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"{name} safety boundary mismatch")


def _load_split(path: Path) -> dict[str, list[dict[str, str]]]:
    if sha256_file(path) != SPLIT_SHA256:
        raise ValueError("N1 split SHA-256 mismatch")
    rows = _csv(path)
    result: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for row in rows:
        split = row.get("split")
        if row.get("eligible") == "1" and split in result:
            if row.get("tumor") not in {"0", "1"}:
                raise ValueError("N1 split contains a non-binary image label")
            result[split].append(row)
    for split in result:
        result[split].sort(key=lambda row: row["image_id"])
        identities = [row["image_id"] for row in result[split]]
        if len(identities) != len(set(identities)):
            raise ValueError(f"N1 split duplicates {split} identities")
    if {key: len(value) for key, value in result.items()} != {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VALIDATION,
    }:
        raise ValueError("N1 split cohort mismatch")
    return result


def _verify_binding(binding_path: Path, protocol: Mapping[str, object]) -> dict[str, Any]:
    binding = _json(binding_path)
    source_hashes = protocol.get("canonical_lf_source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("N1 protocol source hashes are missing")
    if (
        binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("kernel") != KERNEL
        or not isinstance(binding.get("kernel_version"), int)
        or int(binding["kernel_version"]) < 1
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("source_hashes") != source_hashes
        or len(str(binding.get("checkout_commit", ""))) != 40
        or len(str(binding.get("bound_wrapper_sha256", ""))) != 64
        or binding.get("independent_auditor_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("N1 launch binding contract mismatch")
    return binding


def _verify_cache_inventory(
    cache_root: Path,
    split_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    freeze_path = cache_root / "selector_cache_freeze.json"
    manifest_path = cache_root / "selector_cache_manifest.csv"
    wrapper_audit_path = cache_root / "wrapper_output_audit.json"
    if sha256_file(freeze_path) != CACHE_FREEZE_SHA256:
        raise ValueError("N1 selector-cache freeze SHA-256 mismatch")
    freeze = _json(freeze_path)
    if (
        freeze.get("selector_cache_manifest_sha256") != CACHE_MANIFEST_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("baseline_source_commit") != BASELINE_SOURCE_COMMIT
        or freeze.get("baseline_protocol_sha256") != BASELINE_PROTOCOL_SHA256
        or freeze.get("cohort") != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
        or freeze.get("train_masks_discarded") is not True
        or freeze.get("validation_masks_bitpacked") is not True
        or freeze.get("validation_selected_indices_reproduced") != EXPECTED_VALIDATION
        or freeze.get("validation_map_hashes_reproduced") != EXPECTED_VALIDATION
    ):
        raise ValueError("N1 selector-cache provenance mismatch")
    _require_safety(freeze, name="selector cache")
    if sha256_file(manifest_path) != CACHE_MANIFEST_SHA256:
        raise ValueError("N1 selector-cache manifest SHA-256 mismatch")
    if sha256_file(wrapper_audit_path) != CACHE_WRAPPER_AUDIT_SHA256:
        raise ValueError("N1 selector-cache wrapper audit SHA-256 mismatch")
    manifest_rows = _csv(manifest_path)
    indexed = {(row["split"], row["image_id"]): row for row in manifest_rows}
    expected_keys = {
        (split, row["image_id"])
        for split, rows in split_rows.items()
        for row in rows
    }
    if len(manifest_rows) != len(indexed) or set(indexed) != expected_keys:
        raise ValueError("N1 selector-cache physical cohort mismatch")
    nested: dict[str, dict[str, dict[str, str]]] = {"train": {}, "val": {}}
    for split, rows in split_rows.items():
        for row in rows:
            cached = indexed[(split, row["image_id"])]
            if (
                cached["group_id"] != row["group_id"]
                or cached["tumor"] != row["tumor"]
                or int(cached["descriptor_dim"]) != EXPECTED_DESCRIPTOR_DIM
                or bool(int(cached["packed_masks_included"])) != (split == "val")
            ):
                raise ValueError(f"N1 selector-cache identity mismatch: {split}/{row['image_id']}")
            payload_path = _safe_child(cache_root, cached["cache_path"])
            if sha256_file(payload_path) != cached["cache_sha256"]:
                raise ValueError(f"N1 selector-cache record hash mismatch: {split}/{row['image_id']}")
            nested[split][row["image_id"]] = cached
    return nested


def _load_cache_core(path: Path, *, validation: bool) -> dict[str, np.ndarray | int]:
    with np.load(path, allow_pickle=False) as payload:
        schema = payload["schema_version"].copy()
        original = payload["descriptors"].copy()
        flipped = payload["flipped_descriptors"].copy()
        indices = payload["candidate_indices"].copy()
        families = payload["family_ids"].copy()
        fallback_flags = payload["fallback_flags"].copy()
        masks_included = payload["packed_masks_included"].copy()
        result: dict[str, np.ndarray | int] = {
            "descriptors": original,
            "flipped_descriptors": flipped,
            "candidate_indices": indices,
            "family_ids": families,
            "fallback_flags": fallback_flags,
        }
        if validation:
            result.update(
                {
                    "packed_masks": payload["packed_masks"].copy(),
                    "mask_height": int(payload["mask_height"]),
                    "mask_width": int(payload["mask_width"]),
                }
            )
    count = int(original.shape[0])
    if (
        schema.dtype != np.int32
        or schema.shape != ()
        or int(schema) != 2
        or original.dtype != np.float16
        or original.ndim != 2
        or original.shape != flipped.shape
        or original.shape[1] != EXPECTED_DESCRIPTOR_DIM
        or flipped.dtype != np.float16
        or not np.isfinite(original).all()
        or not np.isfinite(flipped).all()
        or indices.dtype != np.int32
        or indices.shape != (count,)
        or count == 0
        or np.any(indices < 0)
        or np.any(np.diff(indices) <= 0)
        or families.dtype != np.int32
        or families.shape != (count,)
        or np.any(families < 0)
        or fallback_flags.dtype != np.uint8
        or fallback_flags.shape != (count,)
        or np.any(fallback_flags > 1)
        or masks_included.dtype != np.uint8
        or masks_included.shape != ()
        or bool(int(masks_included)) != validation
    ):
        raise ValueError(f"N1 selector-cache core schema mismatch: {path}")
    if validation:
        packed = np.asarray(result["packed_masks"])
        height = int(result["mask_height"])
        width = int(result["mask_width"])
        if (
            packed.dtype != np.uint8
            or height <= 0
            or width <= 0
            or packed.shape != (count, (height * width + 7) // 8)
        ):
            raise ValueError(f"N1 packed-mask schema mismatch: {path}")
    return result


def _normal_training_arrays(
    cache_root: Path,
    normal_rows: Sequence[Mapping[str, str]],
    inventory: Mapping[str, Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    descriptor_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    total_candidates = 0
    image_weight = 1.0 / float(len(normal_rows))
    for row in normal_rows:
        cached = inventory[row["image_id"]]
        payload = _load_cache_core(_safe_child(cache_root, cached["cache_path"]), validation=False)
        original = np.asarray(payload["descriptors"], dtype=np.float32)
        flipped = np.asarray(payload["flipped_descriptors"], dtype=np.float32)
        families = np.asarray(payload["family_ids"])
        family_weights = np.zeros(len(families), dtype=np.float64)
        unique_families = np.unique(families)
        for family in unique_families:
            members = families == family
            family_weights[members] = (
                image_weight / float(len(unique_families)) / float(np.count_nonzero(members)) / 2.0
            )
        descriptor_blocks.extend((original, flipped))
        weight_blocks.extend((family_weights, family_weights.copy()))
        total_candidates += len(families)
    values = np.concatenate(descriptor_blocks, axis=0)
    # Match the producer's float32 hierarchical-weight boundary before fitting.
    raw_weights = np.concatenate(weight_blocks).astype(np.float32)
    audit_weight_sum = float(raw_weights.astype(np.float64).sum())
    weights = raw_weights.astype(np.float64)
    weights /= weights.sum()
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("N1 independent hierarchical weights do not sum to one")
    return values, weights, {
        "normal_images": len(normal_rows),
        "normal_candidates": total_candidates,
        "normal_candidate_views": int(values.shape[0]),
        "descriptor_dimension": int(values.shape[1]),
        "weight_sum": audit_weight_sum,
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if result.ndim != 2 or not np.isfinite(result).all() or np.any(norms <= 1.0e-12):
        raise ValueError("N1 independent spherical normalization failed")
    return result / norms


def _fit_spherical_bank(
    descriptors: np.ndarray,
    weights: np.ndarray,
    *,
    prototype_count: int,
    seed: int,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    values = _normalize_rows(descriptors)
    sample_weights = np.asarray(weights, dtype=np.float64).copy()
    if sample_weights.shape != (len(values),) or np.any(sample_weights <= 0):
        raise ValueError("N1 independent weights mismatch")
    sample_weights /= sample_weights.sum()
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(len(values), p=sample_weights))]
    nearest = np.clip(1.0 - values @ values[selected[0]], 0.0, 2.0)
    while len(selected) < prototype_count:
        probabilities = sample_weights * nearest**2
        probabilities[np.asarray(selected, dtype=np.int64)] = 0.0
        total = float(probabilities.sum())
        if total <= 0:
            remaining = np.setdiff1d(np.arange(len(values)), np.asarray(selected))
            selected.append(int(remaining[0]))
        else:
            selected.append(int(rng.choice(len(values), p=probabilities / total)))
        nearest = np.minimum(
            nearest,
            np.clip(1.0 - values @ values[selected[-1]], 0.0, 2.0),
        )
    prototypes = values[np.asarray(selected)].copy()
    for _ in range(maximum_iterations):
        assignments = np.argmax(values @ prototypes.T, axis=1)
        updated = prototypes.copy()
        for cluster in range(prototype_count):
            members = assignments == cluster
            if not np.any(members):
                residual = 1.0 - np.max(values @ updated.T, axis=1)
                updated[cluster] = values[int(np.argmax(sample_weights * residual))]
            else:
                centroid = np.sum(values[members] * sample_weights[members, None], axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm <= 1.0e-12:
                    raise RuntimeError("N1 independent prototype became zero")
                updated[cluster] = centroid / norm
        shift = float(np.max(np.linalg.norm(updated - prototypes, axis=1)))
        prototypes = updated
        if shift <= tolerance:
            break
    assignments = np.argmax(values @ prototypes.T, axis=1).astype(np.int64)
    return prototypes.astype(np.float32), assignments


def _score_views(
    descriptors: np.ndarray,
    flipped_descriptors: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = _normalize_rows(descriptors)
    flipped = _normalize_rows(flipped_descriptors)
    centers = _normalize_rows(prototypes)
    original_distance = (1.0 - np.max(original @ centers.T, axis=1)).astype(np.float32)
    flipped_distance = (1.0 - np.max(flipped @ centers.T, axis=1)).astype(np.float32)
    scores = (0.5 * (original_distance + flipped_distance)).astype(np.float32)
    return original_distance, flipped_distance, scores


def _unpack_masks(packed: np.ndarray, *, count: int, height: int, width: int) -> np.ndarray:
    required_bits = height * width
    if np.asarray(packed).shape != (count, (required_bits + 7) // 8):
        raise ValueError("N1 packed masks do not align")
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, count=required_bits).reshape(
        count, height, width
    ).astype(np.uint8)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("N1 ranks need a finite nontrivial vector")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _absolute_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left, right = _rankdata(first), _rankdata(second)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("N1 Spearman inputs must be nonconstant")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _verify_baseline(
    baseline_root: Path,
    validation_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    freeze_path = baseline_root / "prediction_freeze.json"
    manifest_path = baseline_root / "predictions" / "prediction_manifest.csv"
    if sha256_file(freeze_path) != BASELINE_FREEZE_SHA256:
        raise ValueError("N1 baseline freeze SHA-256 mismatch")
    freeze = _json(freeze_path)
    if (
        freeze.get("checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("source_commit") != BASELINE_SOURCE_COMMIT
        or freeze.get("protocol_sha256") != BASELINE_PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("prediction_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or freeze.get("validation_predictions") != EXPECTED_VALIDATION
    ):
        raise ValueError("N1 baseline provenance mismatch")
    _require_safety(freeze, name="baseline")
    if sha256_file(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise ValueError("N1 baseline manifest SHA-256 mismatch")
    rows = _csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in validation_rows}
    if len(rows) != len(indexed) or set(indexed) != set(expected):
        raise ValueError("N1 baseline cohort mismatch")
    for image_id, row in indexed.items():
        source = expected[image_id]
        map_path = _safe_child(baseline_root / "predictions", row["map_path"])
        probability = _finite(row["bag_probability"], name=f"baseline probability {image_id}")
        if (
            row["group_id"] != source["group_id"]
            or row["tumor"] != source["tumor"]
            or not 0.0 <= probability <= 1.0
            or sha256_file(map_path) != row["map_sha256"]
        ):
            raise ValueError(f"N1 baseline physical mismatch: {image_id}")
        _finite(row["bag_logit"], name=f"baseline logit {image_id}")
    return indexed


def _load_score_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"schema_version", "candidate_indices", "candidate_logits"}:
            raise ValueError("N1 candidate-score payload schema mismatch")
        schema = payload["schema_version"].copy()
        indices = payload["candidate_indices"].copy()
        scores = payload["candidate_logits"].copy()
    if (
        schema.dtype != np.int32
        or schema.shape != ()
        or int(schema) != 1
        or indices.dtype != np.int64
        or scores.dtype != np.float32
        or indices.ndim != 1
        or scores.shape != indices.shape
        or np.any(indices < 0)
        or np.any(np.diff(indices) <= 0)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("N1 candidate-score content mismatch")
    return indices, scores


def audit(
    *,
    protocol_path: Path,
    binding_path: Path,
    split_manifest: Path,
    cache_root: Path,
    baseline_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("N1 protocol SHA-256 mismatch")
    protocol = _json(protocol_path)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("N1 protocol experiment mismatch")
    binding = _verify_binding(binding_path, protocol)
    split_rows = _load_split(split_manifest)
    inventory = _verify_cache_inventory(cache_root, split_rows)
    baseline = _verify_baseline(baseline_root, split_rows["val"])

    freeze_path = output_root / "prediction_freeze.json"
    run_manifest_path = output_root / "run_manifest.json"
    bank_path = output_root / "normal_anomaly_bank.npz"
    bank_audit_path = output_root / "normal_anomaly_bank_audit.json"
    prediction_manifest_path = output_root / "predictions" / "prediction_manifest.csv"
    score_manifest_path = output_root / "candidate_scores" / "candidate_score_manifest.csv"
    evidence_manifest_path = output_root / "normal_anomaly_evidence_manifest.csv"
    freeze = _json(freeze_path)
    run_manifest = _json(run_manifest_path)
    bank_audit = _json(bank_audit_path)
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("selector_cache_manifest_sha256") != CACHE_MANIFEST_SHA256
        or freeze.get("baseline_prediction_freeze_sha256") != BASELINE_FREEZE_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("validation_predictions") != EXPECTED_VALIDATION
        or freeze.get("training_labels") != "image_level_normal_only"
        or freeze.get("ranking_semantics") != "direct_normal_anomaly_distance_not_classification_logit"
        or freeze.get("classification_probabilities") != "exact_accepted_geometry_v3"
    ):
        raise ValueError("N1 prediction freeze provenance mismatch")
    _require_safety(freeze, name="N1 prediction freeze")
    _require_safety(run_manifest, name="N1 run manifest")
    _require_safety(bank_audit, name="N1 bank audit")
    device_names = run_manifest.get("device_names")
    if (
        run_manifest.get("run_id") != "btxrd_mask_bag_normal_only_direct_anomaly_n1_v1"
        or not isinstance(device_names, list)
        or len(device_names) != 2
        or not all("T4" in str(name) for name in device_names)
        or run_manifest.get("normal_images") != EXPECTED_NORMAL
        or run_manifest.get("validation_predictions") != EXPECTED_VALIDATION
        or run_manifest.get("prediction_freeze_sha256") != sha256_file(freeze_path)
        or run_manifest.get("config")
        != {
            "convergence_tolerance": 1.0e-06,
            "maximum_iterations": 100,
            "prototype_count": EXPECTED_PROTOTYPES,
            "seed": EXPECTED_SEED,
        }
    ):
        raise ValueError("N1 runtime manifest mismatch")
    for path, key in (
        (bank_path, "normal_anomaly_bank_sha256"),
        (bank_audit_path, "normal_anomaly_bank_audit_sha256"),
        (prediction_manifest_path, "prediction_manifest_sha256"),
        (score_manifest_path, "candidate_score_manifest_sha256"),
        (evidence_manifest_path, "normal_anomaly_evidence_manifest_sha256"),
    ):
        if sha256_file(path) != freeze.get(key):
            raise ValueError(f"N1 frozen artifact hash mismatch: {key}")

    with np.load(bank_path, allow_pickle=False) as payload:
        if set(payload.files) != {"schema_version", "prototypes"}:
            raise ValueError("N1 bank schema mismatch")
        schema = payload["schema_version"].copy()
        frozen_prototypes = payload["prototypes"].copy()
    if (
        schema.dtype != np.int32
        or schema.shape != ()
        or int(schema) != 1
        or frozen_prototypes.dtype != np.float32
        or frozen_prototypes.shape != (EXPECTED_PROTOTYPES, EXPECTED_DESCRIPTOR_DIM)
        or not np.isfinite(frozen_prototypes).all()
    ):
        raise ValueError("N1 bank physical content mismatch")

    normal_rows = [row for row in split_rows["train"] if row["tumor"] == "0"]
    if len(normal_rows) != EXPECTED_NORMAL:
        raise ValueError("N1 normal-only training cohort mismatch")
    training_values, training_weights, independent_bank_audit = _normal_training_arrays(
        cache_root, normal_rows, inventory["train"]
    )
    reproduced_prototypes, assignments = _fit_spherical_bank(
        training_values,
        training_weights,
        prototype_count=EXPECTED_PROTOTYPES,
        seed=EXPECTED_SEED,
    )
    if not np.array_equal(reproduced_prototypes, frozen_prototypes):
        maximum = float(np.max(np.abs(reproduced_prototypes - frozen_prototypes)))
        raise ValueError(f"N1 independent bank reproduction mismatch: max delta {maximum}")
    cluster_counts = np.bincount(assignments, minlength=EXPECTED_PROTOTYPES).astype(int).tolist()
    expected_bank_audit = {
        **independent_bank_audit,
        "all_training_image_labels_normal": True,
        "view_multiplicity": 2,
        "prototype_count": EXPECTED_PROTOTYPES,
        "seed": EXPECTED_SEED,
        "cluster_counts": cluster_counts,
        "positive_bags_used": 0,
        "learned_residual": False,
    }
    for key, expected in expected_bank_audit.items():
        if isinstance(expected, float):
            _close(bank_audit.get(key), expected, name=f"bank audit {key}", atol=1.0e-12)
        elif bank_audit.get(key) != expected:
            raise ValueError(f"N1 bank audit differs: {key}")
    if (
        bank_audit.get("config")
        != {
            "convergence_tolerance": 1.0e-06,
            "maximum_iterations": 100,
            "prototype_count": EXPECTED_PROTOTYPES,
            "seed": EXPECTED_SEED,
        }
        or bank_audit.get("bank_sha256") != sha256_file(bank_path)
        or bank_audit.get("training_labels") != "image_level_normal_only"
    ):
        raise ValueError("N1 bank audit configuration mismatch")

    prediction_rows = _csv(prediction_manifest_path)
    score_rows = _csv(score_manifest_path)
    evidence_rows = _csv(evidence_manifest_path)
    predictions = {row["image_id"]: row for row in prediction_rows}
    scores = {row["image_id"]: row for row in score_rows}
    evidence = {row["image_id"]: row for row in evidence_rows}
    expected_ids = [row["image_id"] for row in split_rows["val"]]
    if any(
        len(rows) != EXPECTED_VALIDATION or len(indexed) != EXPECTED_VALIDATION or set(indexed) != set(expected_ids)
        for rows, indexed in (
            (prediction_rows, predictions),
            (score_rows, scores),
            (evidence_rows, evidence),
        )
    ):
        raise ValueError("N1 output manifest cohort mismatch")

    candidate_counts: list[int] = []
    probabilities: list[float] = []
    view_agreements = 0
    map_bytes = 0
    for row in split_rows["val"]:
        image_id = row["image_id"]
        cached = inventory["val"][image_id]
        cache_payload = _load_cache_core(
            _safe_child(cache_root, cached["cache_path"]), validation=True
        )
        indices = np.asarray(cache_payload["candidate_indices"], dtype=np.int64)
        original_distance, flipped_distance, reproduced_scores = _score_views(
            np.asarray(cache_payload["descriptors"]),
            np.asarray(cache_payload["flipped_descriptors"]),
            frozen_prototypes,
        )
        selected_position = int(np.argmax(reproduced_scores))
        selected_index = int(indices[selected_position])
        prediction = predictions[image_id]
        score_row = scores[image_id]
        evidence_row = evidence[image_id]
        base = baseline[image_id]
        if (
            prediction["group_id"] != row["group_id"]
            or prediction["tumor"] != row["tumor"]
            or prediction["candidate_payload_sha256"] != cached["candidate_payload_sha256"]
            or score_row["group_id"] != row["group_id"]
            or score_row["tumor"] != row["tumor"]
            or score_row["candidate_payload_sha256"] != cached["candidate_payload_sha256"]
            or int(prediction["candidate_count"]) != len(indices)
            or int(score_row["candidate_count"]) != len(indices)
            or int(evidence_row["candidate_count"]) != len(indices)
            or int(prediction["selected_candidate_index"]) != selected_index
            or int(score_row["selected_candidate_index"]) != selected_index
            or int(evidence_row["selected_candidate_index"]) != selected_index
            or prediction["bag_logit"] != base["bag_logit"]
            or prediction["bag_probability"] != base["bag_probability"]
            or prediction["candidate_logit_tta"] != "mean_original_aligned_horizontal_flip"
            or int(prediction["fallback_count"])
            != int(np.asarray(cache_payload["fallback_flags"]).sum())
        ):
            raise ValueError(f"N1 manifest identity/arithmetic mismatch: {image_id}")
        _close(
            prediction["selected_candidate_logit"],
            reproduced_scores[selected_position],
            name=f"prediction selected score {image_id}",
            atol=0.0,
        )
        _close(
            score_row["selected_candidate_logit"],
            reproduced_scores[selected_position],
            name=f"score selected score {image_id}",
            atol=0.0,
        )

        score_path = _safe_child(output_root / "candidate_scores", score_row["score_path"])
        if sha256_file(score_path) != score_row["score_sha256"]:
            raise ValueError(f"N1 score payload hash mismatch: {image_id}")
        frozen_indices, frozen_scores = _load_score_payload(score_path)
        if not np.array_equal(frozen_indices, indices) or not np.array_equal(
            frozen_scores, reproduced_scores
        ):
            raise ValueError(f"N1 independently reproduced candidate scores differ: {image_id}")

        evidence_path = _safe_child(output_root / "normal_anomaly_evidence", evidence_row["evidence_path"])
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"N1 evidence payload hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "schema_version",
                "candidate_indices",
                "original_normal_distance",
                "flipped_normal_distance",
                "candidate_scores",
            }:
                raise ValueError(f"N1 evidence schema mismatch: {image_id}")
            evidence_schema = payload["schema_version"].copy()
            evidence_indices = payload["candidate_indices"].copy()
            frozen_original = payload["original_normal_distance"].copy()
            frozen_flipped = payload["flipped_normal_distance"].copy()
            frozen_combined = payload["candidate_scores"].copy()
        agreement = int(np.argmax(original_distance) == np.argmax(flipped_distance))
        if (
            evidence_schema.dtype != np.int32
            or evidence_schema.shape != ()
            or int(evidence_schema) != 1
            or not np.array_equal(evidence_indices, indices)
            or not np.array_equal(frozen_original, original_distance)
            or not np.array_equal(frozen_flipped, flipped_distance)
            or not np.array_equal(frozen_combined, reproduced_scores)
            or int(evidence_row["view_selected_agreement"]) != agreement
        ):
            raise ValueError(f"N1 independent view evidence differs: {image_id}")
        view_agreements += agreement

        packed = np.asarray(cache_payload["packed_masks"])
        height = int(cache_payload["mask_height"])
        width = int(cache_payload["mask_width"])
        masks = _unpack_masks(packed, count=len(indices), height=height, width=width)
        probability = _finite(base["bag_probability"], name=f"probability {image_id}")
        expected_map = (masks[selected_position].astype(np.float32) * probability).astype(np.float16)
        map_path = _safe_child(output_root / "predictions", prediction["map_path"])
        if sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"N1 prediction-map hash mismatch: {image_id}")
        frozen_map = np.load(map_path, allow_pickle=False)
        if frozen_map.dtype != np.float16 or not np.array_equal(frozen_map, expected_map):
            raise ValueError(f"N1 independently reproduced map differs: {image_id}")
        _close(
            prediction["selected_area_ratio"],
            float(masks[selected_position].mean()),
            name=f"selected area {image_id}",
        )
        candidate_counts.append(len(indices))
        probabilities.append(probability)
        map_bytes += map_path.stat().st_size

    association = _absolute_spearman(candidate_counts, probabilities)
    _close(
        association,
        EXPECTED_COUNT_PROBABILITY_SPEARMAN,
        name="independent count/probability Spearman",
        atol=1.0e-12,
    )
    _close(
        freeze.get("absolute_candidate_count_probability_spearman"),
        association,
        name="frozen count/probability Spearman",
        atol=1.0e-12,
    )
    return {
        "schema_version": 1,
        "status": "PASS_GT_BLIND_PHYSICAL_RECOMPUTATION",
        "experiment_id": EXPERIMENT_ID,
        "kernel": KERNEL,
        "kernel_version": binding["kernel_version"],
        "scientific_source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "binding_sha256": sha256_file(binding_path),
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "normal_anomaly_bank_sha256": sha256_file(bank_path),
        "normal_images_recomputed": EXPECTED_NORMAL,
        "normal_candidate_views_recomputed": independent_bank_audit["normal_candidate_views"],
        "prototype_count": EXPECTED_PROTOTYPES,
        "prototype_bank_exact": True,
        "validation_predictions_recomputed": EXPECTED_VALIDATION,
        "candidate_score_payloads_recomputed": EXPECTED_VALIDATION,
        "view_evidence_payloads_recomputed": EXPECTED_VALIDATION,
        "physical_maps_recomputed": EXPECTED_VALIDATION,
        "physical_map_bytes": map_bytes,
        "view_selected_agreements": view_agreements,
        "absolute_candidate_count_probability_spearman": association,
        "baseline_probabilities_preserved_exactly": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_output.exists():
        raise FileExistsError(f"N1 audit output already exists: {args.audit_output}")
    result = audit(
        protocol_path=args.protocol,
        binding_path=args.binding,
        split_manifest=args.split_manifest,
        cache_root=args.selector_cache_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
