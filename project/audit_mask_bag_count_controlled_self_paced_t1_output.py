from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen T1 output."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


EXPERIMENT_ID = "EXP-20260801-codex-t1-count-controlled-self-paced-v1"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-count-controlled-t1-v1"
SOURCE_COMMIT = "c7f0937d515ded9bbd8928a2236cbe44b7a25f79"
PROTOCOL_SHA256 = "6a4379e896f3ea3862dce1edcdea20af09a90ec8f9cbbd6eb25bf8eca1306a7c"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_MANIFEST_SHA256 = "8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e"
CACHE_WRAPPER_AUDIT_SHA256 = (
    "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
)
BASELINE_CHECKPOINT_SHA256 = (
    "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
)
COUNT_SPEARMAN_CEILING = 0.5013777759365411
MINIMUM_OOF_AUROC = 0.75
MINIMUM_VIEW_AGREEMENT = 0.60
FOLDS = tuple(range(5))
EXPECTED_FOLD_SUMMARY = [
    {"fold": 0, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 1, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 2, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 3, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 4, "images": 597, "groups": 198, "normal_images": 299, "tumor_images": 298},
]
EXPECTED_TRAINING_CONFIG = {
    "fold_count": 5,
    "producer_epochs": 16,
    "producer_batch_size": 16,
    "producer_learning_rate": 0.0003,
    "producer_weight_decay": 0.0001,
    "view_consistency_weight": 0.1,
    "count_independence_weight": 1.0,
    "maximum_count_spearman": COUNT_SPEARMAN_CEILING,
    "minimum_oof_auroc": MINIMUM_OOF_AUROC,
    "minimum_view_agreement": MINIMUM_VIEW_AGREEMENT,
    "pace_fractions": [0.2, 0.4, 0.6],
    "consumer_epochs": 12,
    "consumer_learning_rate": 0.0001,
    "supervised_contrastive_weight": 0.25,
    "contrastive_temperature": 0.1,
    "residual_hidden_dim": 128,
    "seed": 42,
}
EXPECTED_MODEL_CONFIG = {
    "token_dim": 128,
    "token_layers": 3,
    "hidden_dim": 256,
    "metadata_dim": 4,
    "bag_temperature": 0.2,
    "context_radius": 2,
    "minimum_grid_mass": 0.25,
}
POST_FREEZE_ONLY_SOURCE_PATHS = {
    "project/evaluate_mask_bag_selector_arm.py",
    "project/models/mask_bag_ranking_diagnostics.py",
}


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
        raise ValueError(f"Absolute output path is forbidden: {relative}")
    resolved_root = root.resolve()
    resolved = (root / child).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"Output path escapes its root: {relative}")
    return resolved


def _require_safety(payload: Mapping[str, object], *, name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"{name} safety boundary mismatch")


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _close(actual: object, expected: object, *, name: str, atol: float = 1.0e-7) -> None:
    if abs(_finite(actual, name=name) - _finite(expected, name=name)) > atol:
        raise ValueError(f"{name} differs: {actual} versus {expected}")


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Ranks require a finite nontrivial vector")
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
    if len(first) != len(second):
        raise ValueError("Spearman inputs must align")
    left = _rankdata(first)
    right = _rankdata(second)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("Spearman ranks must be nonconstant")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_array = np.asarray(labels, dtype=np.int32)
    ranks = _rankdata(scores)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("AUROC requires both image labels")
    positive = label_array == 1
    positive_count = int(positive.sum())
    negative_count = len(label_array) - positive_count
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _normalized_smoothmax(values: np.ndarray, temperature: float = 0.2) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("SmoothMax needs a finite nonempty vector")
    scaled = array / temperature
    maximum = float(scaled.max())
    return float(
        temperature
        * (maximum + math.log(float(np.exp(scaled - maximum).sum())) - math.log(len(array)))
    )


def _verify_binding(binding_path: Path, protocol: Mapping[str, object]) -> dict[str, Any]:
    binding = _json(binding_path)
    protocol_hashes = protocol.get("canonical_lf_source_hashes")
    if not isinstance(protocol_hashes, dict):
        raise ValueError("T1 protocol source hashes are missing")
    runtime_hashes = {
        path: digest
        for path, digest in protocol_hashes.items()
        if path not in POST_FREEZE_ONLY_SOURCE_PATHS
    }
    if (
        binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("kernel") != KERNEL
        or not isinstance(binding.get("kernel_version"), int)
        or int(binding["kernel_version"]) < 1
        or binding.get("scientific_source_commit") != SOURCE_COMMIT
        or binding.get("protocol_sha256") != PROTOCOL_SHA256
        or binding.get("source_hashes") != runtime_hashes
        or len(str(binding.get("checkout_commit", ""))) != 40
        or len(str(binding.get("bound_wrapper_sha256", ""))) != 64
    ):
        raise ValueError("T1 launch binding contract mismatch")
    return binding


def _verify_cache(cache_root: Path) -> tuple[dict[str, dict[str, str]], int]:
    freeze_path = cache_root / "selector_cache_freeze.json"
    manifest_path = cache_root / "selector_cache_manifest.csv"
    wrapper_path = cache_root / "wrapper_output_audit.json"
    if sha256_file(freeze_path) != CACHE_FREEZE_SHA256:
        raise ValueError("Selector-cache freeze SHA-256 mismatch")
    freeze = _json(freeze_path)
    if (
        freeze.get("selector_cache_manifest_sha256") != CACHE_MANIFEST_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("cohort") != {"train": 2981, "validation": 371}
    ):
        raise ValueError("Selector-cache freeze provenance mismatch")
    _require_safety(freeze, name="selector cache")
    if sha256_file(manifest_path) != CACHE_MANIFEST_SHA256:
        raise ValueError("Selector-cache manifest SHA-256 mismatch")
    if sha256_file(wrapper_path) != CACHE_WRAPPER_AUDIT_SHA256:
        raise ValueError("Selector-cache wrapper audit SHA-256 mismatch")
    rows = _csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != 3352 or len(indexed) != 3352:
        raise ValueError("Selector-cache cohort mismatch")
    if {
        split: sum(row["split"] == split for row in rows) for split in ("train", "val")
    } != {"train": 2981, "val": 371}:
        raise ValueError("Selector-cache split counts mismatch")
    return indexed, manifest_path.stat().st_size


def _load_cache_payload(cache_root: Path, row: Mapping[str, str]) -> dict[str, np.ndarray]:
    path = _safe_child(cache_root, row["cache_path"])
    if sha256_file(path) != row["cache_sha256"]:
        raise ValueError(f"Selector-cache record hash mismatch: {row['image_id']}")
    required = {
        "descriptors",
        "flipped_descriptors",
        "candidate_indices",
        "family_ids",
        "fallback_flags",
    }
    with np.load(path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise ValueError(f"Selector-cache fields missing: {row['image_id']}")
        result = {name: payload[name].copy() for name in required}
        for name in ("packed_masks", "mask_height", "mask_width"):
            if name in payload.files:
                result[name] = payload[name].copy()
    count = int(row["candidate_count"])
    if (
        result["descriptors"].shape != (count, 1156)
        or result["flipped_descriptors"].shape != (count, 1156)
        or result["candidate_indices"].dtype != np.int32
        or result["candidate_indices"].shape != (count,)
        or result["family_ids"].shape != (count,)
        or result["fallback_flags"].shape != (count,)
    ):
        raise ValueError(f"Selector-cache payload alignment mismatch: {row['image_id']}")
    return result


def _recompute_crossfit_assignment(
    cache_rows: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    train = [row for row in cache_rows.values() if row["split"] == "train"]
    groups = np.asarray([row["group_id"] for row in train], dtype="U128")
    labels = np.asarray([int(row["tumor"]) for row in train], dtype=np.int8)
    group_records: list[tuple[str, int, int]] = []
    for group in np.unique(groups):
        members = groups == group
        group_labels = np.unique(labels[members])
        if len(group_labels) != 1:
            raise ValueError("T1 cross-fit source group contains mixed labels")
        group_records.append((str(group), int(group_labels[0]), int(np.count_nonzero(members))))
    rng = np.random.default_rng(42)
    fold_tie_order = rng.permutation(5)
    fold_tie_rank = np.empty(5, dtype=np.int64)
    fold_tie_rank[fold_tie_order] = np.arange(5)
    fold_rows = np.zeros((5, 2), dtype=np.int64)
    fold_groups = np.zeros((5, 2), dtype=np.int64)
    group_to_fold: dict[str, int] = {}
    for label in (0, 1):
        records = [record for record in group_records if record[1] == label]
        random_ties = {
            group: float(value)
            for (group, _label, _size), value in zip(records, rng.random(len(records)), strict=True)
        }
        records.sort(key=lambda record: (-record[2], random_ties[record[0]]))
        for group, _label, size in records:
            target = min(
                range(5),
                key=lambda fold: (
                    int(fold_rows[fold, label]),
                    int(fold_groups[fold, label]),
                    int(fold_rows[fold].sum()),
                    int(fold_tie_rank[fold]),
                ),
            )
            group_to_fold[group] = target
            fold_rows[target, label] += size
            fold_groups[target, label] += 1
    rows = sorted(
        (
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "image_label": int(row["tumor"]),
                "heldout_fold": int(group_to_fold[row["group_id"]]),
            }
            for row in train
        ),
        key=lambda row: str(row["image_id"]),
    )
    summary = []
    for fold in FOLDS:
        heldout = [row for row in rows if int(row["heldout_fold"]) == fold]
        summary.append(
            {
                "fold": fold,
                "images": len(heldout),
                "groups": len({str(row["group_id"]) for row in heldout}),
                "normal_images": sum(int(row["image_label"]) == 0 for row in heldout),
                "tumor_images": sum(int(row["image_label"]) == 1 for row in heldout),
            }
        )
    return rows, summary


def _torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint root must be a dictionary: {path}")
    return payload


def _training_config_matches(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    normalized = dict(value)
    if isinstance(normalized.get("pace_fractions"), tuple):
        normalized["pace_fractions"] = list(normalized["pace_fractions"])
    return normalized == EXPECTED_TRAINING_CONFIG


def _verify_oof(
    root: Path,
    cache_root: Path,
    cache_rows: Mapping[str, Mapping[str, str]],
    freeze: Mapping[str, object],
    assignment_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, Any]], dict[str, object]]:
    assignment = {str(row["image_id"]): row for row in assignment_rows}
    all_scores: dict[str, dict[str, Any]] = {}
    physical_bytes = 0
    producer_hashes = freeze.get("producer_hashes")
    if not isinstance(producer_hashes, dict) or set(producer_hashes) != {f"fold_{i}" for i in FOLDS}:
        raise ValueError("T1 producer hash table mismatch")
    all_train_groups = {
        row["group_id"] for row in cache_rows.values() if row["split"] == "train"
    }
    for fold in FOLDS:
        fold_root = root / "oof_producers" / f"fold_{fold}"
        checkpoint_path = fold_root / "producer.pt"
        history_path = fold_root / "training_history.json"
        audit_path = fold_root / "fold_audit.json"
        manifest_path = fold_root / "score_manifest.csv"
        expected_hashes = producer_hashes[f"fold_{fold}"]
        observed_hashes = {
            "producer_checkpoint_sha256": sha256_file(checkpoint_path),
            "training_history_sha256": sha256_file(history_path),
            "score_manifest_sha256": sha256_file(manifest_path),
            "fold_audit_sha256": sha256_file(audit_path),
        }
        if observed_hashes != expected_hashes:
            raise ValueError(f"T1 fold artifact hash mismatch: {fold}")
        fold_audit = _json(audit_path)
        checkpoint = _torch_checkpoint(checkpoint_path)
        _require_safety(checkpoint, name=f"producer checkpoint {fold}")
        heldout_groups = {
            str(row["group_id"])
            for row in assignment_rows
            if int(row["heldout_fold"]) == fold
        }
        training_groups = all_train_groups - heldout_groups
        if (
            checkpoint.get("heldout_fold") != fold
            or checkpoint.get("source_commit") != SOURCE_COMMIT
            or checkpoint.get("protocol_sha256") != PROTOCOL_SHA256
            or checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG
            or not _training_config_matches(checkpoint.get("training_config"))
            or set(checkpoint.get("training_groups", [])) != training_groups
            or set(checkpoint.get("heldout_groups", [])) != heldout_groups
            or fold_audit.get("heldout_fold") != fold
            or fold_audit.get("derived_seed") != 1042 + fold
            or fold_audit.get("group_overlap") != 0
            or set(fold_audit.get("training_groups", [])) != training_groups
            or set(fold_audit.get("heldout_groups", [])) != heldout_groups
            or fold_audit.get("producer_checkpoint_sha256") != observed_hashes["producer_checkpoint_sha256"]
            or fold_audit.get("training_history_sha256") != observed_hashes["training_history_sha256"]
            or fold_audit.get("score_manifest_sha256") != observed_hashes["score_manifest_sha256"]
            or fold_audit.get("validation_segmentation_quality_used") is not False
        ):
            raise ValueError(f"T1 producer fold exclusion/provenance mismatch: {fold}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if (
            not isinstance(history, list)
            or len(history) != 16
            or [int(row.get("epoch", -1)) for row in history] != list(range(1, 17))
            or any(
                not all(math.isfinite(float(row[key])) for key in ("total", "image", "consistency", "count"))
                for row in history
            )
        ):
            raise ValueError(f"T1 producer history mismatch: {fold}")
        manifest = _csv(manifest_path)
        expected_ids = {
            image_id for image_id, row in assignment.items() if int(row["heldout_fold"]) == fold
        }
        if len(manifest) != len(expected_ids) or {row["image_id"] for row in manifest} != expected_ids:
            raise ValueError(f"T1 OOF score cohort mismatch: {fold}")
        if fold_audit.get("heldout_records") != len(manifest):
            raise ValueError(f"T1 held-out record count mismatch: {fold}")
        for row in manifest:
            image_id = row["image_id"]
            source = cache_rows[image_id]
            cache = _load_cache_payload(cache_root, source)
            score_path = _safe_child(fold_root / "scores", row["payload_path"])
            if sha256_file(score_path) != row["payload_sha256"]:
                raise ValueError(f"T1 OOF score hash mismatch: {image_id}")
            with np.load(score_path, allow_pickle=False) as payload:
                if set(payload.files) != {"candidate_indices", "original_logits", "flipped_logits"}:
                    raise ValueError(f"T1 OOF score schema mismatch: {image_id}")
                indices = payload["candidate_indices"].copy()
                original = payload["original_logits"].copy()
                flipped = payload["flipped_logits"].copy()
            count = int(row["candidate_count"])
            if (
                row["group_id"] != source["group_id"]
                or int(row["image_label"]) != int(source["tumor"])
                or int(row["heldout_fold"]) != fold
                or count != int(source["candidate_count"])
                or indices.dtype != np.int32
                or original.dtype != np.float32
                or flipped.dtype != np.float32
                or indices.shape != (count,)
                or original.shape != (count,)
                or flipped.shape != (count,)
                or not np.array_equal(indices, cache["candidate_indices"])
                or not np.isfinite(original).all()
                or not np.isfinite(flipped).all()
            ):
                raise ValueError(f"T1 OOF score content mismatch: {image_id}")
            averaged = 0.5 * (original + flipped)
            probability = _sigmoid(_normalized_smoothmax(averaged))
            _close(row["bag_probability"], probability, name="OOF bag probability", atol=2.0e-6)
            if image_id in all_scores:
                raise ValueError(f"T1 duplicate OOF score: {image_id}")
            all_scores[image_id] = {
                "image_id": image_id,
                "group_id": source["group_id"],
                "image_label": int(source["tumor"]),
                "heldout_fold": fold,
                "candidate_count": count,
                "bag_probability": float(row["bag_probability"]),
                "candidate_indices": indices,
                "original_logits": original,
                "flipped_logits": flipped,
                "family_ids": cache["family_ids"],
            }
            physical_bytes += score_path.stat().st_size
        physical_bytes += sum(path.stat().st_size for path in (checkpoint_path, history_path, audit_path, manifest_path))
    if len(all_scores) != 2981 or set(all_scores) != set(assignment):
        raise ValueError("T1 complete OOF coverage mismatch")
    ordered = [all_scores[str(row["image_id"])] for row in assignment_rows]
    association = _absolute_spearman(
        [int(row["candidate_count"]) for row in ordered],
        [float(row["bag_probability"]) for row in ordered],
    )
    auroc = _binary_auroc(
        [int(row["image_label"]) for row in ordered],
        [float(row["bag_probability"]) for row in ordered],
    )
    agreement = float(
        np.mean(
            [
                int(np.argmax(row["original_logits"]) == np.argmax(row["flipped_logits"]))
                for row in ordered
            ]
        )
    )
    gate_path = root / "producer_gate_audit.json"
    gate = _json(gate_path)
    if sha256_file(gate_path) != freeze.get("producer_gate_audit_sha256"):
        raise ValueError("T1 producer-gate hash mismatch")
    _close(gate.get("absolute_candidate_count_probability_spearman"), association, name="OOF count Spearman", atol=1.0e-12)
    _close(gate.get("image_auroc"), auroc, name="OOF image AUROC", atol=1.0e-12)
    _close(gate.get("original_flip_top1_agreement"), agreement, name="OOF view agreement", atol=1.0e-12)
    checks = {
        "count_spearman": association <= COUNT_SPEARMAN_CEILING,
        "image_auroc": auroc >= MINIMUM_OOF_AUROC,
        "view_agreement": agreement >= MINIMUM_VIEW_AGREEMENT,
        "group_exclusion": True,
    }
    if (
        gate.get("records") != 2981
        or gate.get("folds") != 5
        or gate.get("group_overlap") != 0
        or gate.get("checks") != checks
        or gate.get("producer_gate_pass") is not True
        or not all(checks.values())
    ):
        raise ValueError("T1 producer operational gate did not pass independently")
    return all_scores, {
        "records": 2981,
        "folds": 5,
        "absolute_candidate_count_probability_spearman": association,
        "image_auroc": auroc,
        "original_flip_top1_agreement": agreement,
        "checks": checks,
        "physical_oof_bytes_verified": physical_bytes + gate_path.stat().st_size,
    }


def _verify_targets(
    root: Path,
    freeze: Mapping[str, object],
    cache_rows: Mapping[str, Mapping[str, str]],
    oof_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    target_root = root / "self_paced_targets"
    target_freeze_path = target_root / "target_freeze.json"
    if sha256_file(target_freeze_path) != freeze.get("target_freeze_sha256"):
        raise ValueError("T1 target-freeze hash mismatch")
    target_freeze = _json(target_freeze_path)
    _require_safety(target_freeze, name="target freeze")
    if (
        target_freeze.get("producer_gate_sha256") != freeze.get("producer_gate_audit_sha256")
        or target_freeze.get("pace_fractions") != [0.2, 0.4, 0.6]
    ):
        raise ValueError("T1 target-freeze provenance mismatch")
    target_hashes = target_freeze.get("target_hashes")
    names = ["negative_targets.csv", *[f"positive_targets_stage_{stage}.csv" for stage in (1, 2, 3)]]
    if not isinstance(target_hashes, dict) or set(target_hashes) != set(names):
        raise ValueError("T1 target hash table mismatch")
    rows = {}
    physical_bytes = target_freeze_path.stat().st_size
    for name in names:
        path = target_root / name
        if sha256_file(path) != target_hashes[name]:
            raise ValueError(f"T1 target file hash mismatch: {name}")
        rows[name] = _csv(path)
        if not rows[name]:
            raise ValueError(f"T1 target file is empty: {name}")
        physical_bytes += path.stat().st_size
    negative_records = [row for row in cache_rows.values() if row["split"] == "train" and int(row["tumor"]) == 0]
    expected_negative: dict[tuple[str, int], tuple[float, str, int]] = {}
    image_weight = 1.0 / len(negative_records)
    for record in negative_records:
        score = oof_scores[record["image_id"]]
        families = np.asarray(score["family_ids"])
        unique = sorted(set(int(value) for value in families.tolist()))
        for family in unique:
            members = np.flatnonzero(families == family)
            weight = image_weight / len(unique) / len(members)
            for candidate in members:
                expected_negative[(record["image_id"], int(candidate))] = (
                    weight,
                    str(family),
                    int(score["heldout_fold"]),
                )
    observed_negative = rows["negative_targets.csv"]
    if len(observed_negative) != len(expected_negative):
        raise ValueError("T1 negative target count mismatch")
    for row in observed_negative:
        key = (row["image_id"], int(row["candidate_index"]))
        if key not in expected_negative or int(row["target"]) != 0:
            raise ValueError("T1 negative target identity mismatch")
        weight, family, fold = expected_negative[key]
        _close(row["weight"], weight, name="negative target weight", atol=1.0e-15)
        if row["family_id"] != family or int(row["producer_fold"]) != fold:
            raise ValueError("T1 negative target family/fold mismatch")
    eligible: list[dict[str, object]] = []
    for record in cache_rows.values():
        if record["split"] != "train" or int(record["tumor"]) != 1:
            continue
        score = oof_scores[record["image_id"]]
        original = np.asarray(score["original_logits"], dtype=np.float64)
        flipped = np.asarray(score["flipped_logits"], dtype=np.float64)
        candidate = int(np.argmax(original))
        if candidate != int(np.argmax(flipped)):
            continue
        if len(original) == 1:
            margin = float(np.finfo(np.float32).max)
        else:
            others = np.arange(len(original)) != candidate
            margin = min(
                float(original[candidate] - np.max(original[others])),
                float(flipped[candidate] - np.max(flipped[others])),
            )
        if margin > 0:
            eligible.append(
                {
                    "image_id": record["image_id"],
                    "candidate_index": candidate,
                    "margin": margin,
                    "producer_fold": int(score["heldout_fold"]),
                }
            )
    eligible.sort(key=lambda row: (-float(row["margin"]), str(row["image_id"])))
    if not eligible or target_freeze.get("eligible_positive_bags") != len(eligible):
        raise ValueError("T1 eligible positive-bag count mismatch")
    previous: set[tuple[str, int]] = set()
    stage_counts: list[int] = []
    for stage, fraction in enumerate((0.2, 0.4, 0.6), start=1):
        expected_count = min(len(eligible), int(math.ceil(fraction * len(eligible))))
        expected = eligible[:expected_count]
        observed = rows[f"positive_targets_stage_{stage}.csv"]
        if len(observed) != expected_count:
            raise ValueError(f"T1 positive target count mismatch: stage {stage}")
        current: set[tuple[str, int]] = set()
        for stored, wanted in zip(observed, expected, strict=True):
            key = (stored["image_id"], int(stored["candidate_index"]))
            current.add(key)
            if (
                key != (wanted["image_id"], int(wanted["candidate_index"]))
                or int(stored["target"]) != 1
                or int(stored["producer_fold"]) != int(wanted["producer_fold"])
            ):
                raise ValueError(f"T1 positive target identity/order mismatch: stage {stage}")
            _close(stored["margin"], wanted["margin"], name="positive target margin", atol=1.0e-12)
            _close(stored["weight"], 1.0 / expected_count, name="positive target weight", atol=1.0e-15)
            assignment = oof_scores[stored["image_id"]]
            if int(stored["producer_fold"]) != int(assignment["heldout_fold"]):
                raise ValueError("T1 positive target OOF fold mismatch")
        if not previous.issubset(current):
            raise ValueError("T1 positive target stages are not nested")
        previous = current
        stage_counts.append(expected_count)
    return {
        "eligible_positive_bags": len(eligible),
        "negative_candidates": len(expected_negative),
        "positive_stage_counts": stage_counts,
        "physical_target_bytes_verified": physical_bytes,
    }


def _verify_validation(
    root: Path,
    cache_root: Path,
    cache_rows: Mapping[str, Mapping[str, str]],
    freeze: Mapping[str, object],
) -> dict[str, object]:
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    diagnostics_path = root / "gt_blind_diagnostics.csv"
    if sha256_file(prediction_manifest) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("T1 prediction manifest hash mismatch")
    if sha256_file(score_manifest) != freeze.get("candidate_score_manifest_sha256"):
        raise ValueError("T1 candidate-score manifest hash mismatch")
    if sha256_file(diagnostics_path) != freeze.get("gt_blind_diagnostics_sha256"):
        raise ValueError("T1 GT-blind diagnostics hash mismatch")
    predictions = _csv(prediction_manifest)
    scores = _csv(score_manifest)
    diagnostics = _csv(diagnostics_path)
    prediction_by_id = {row["image_id"]: row for row in predictions}
    score_by_id = {row["image_id"]: row for row in scores}
    diagnostic_by_id = {row["image_id"]: row for row in diagnostics}
    expected_ids = {row["image_id"] for row in cache_rows.values() if row["split"] == "val"}
    if (
        len(prediction_by_id) != 371
        or len(score_by_id) != 371
        or len(diagnostic_by_id) != 371
        or set(prediction_by_id) != expected_ids
        or set(score_by_id) != expected_ids
        or set(diagnostic_by_id) != expected_ids
    ):
        raise ValueError("T1 frozen validation cohort mismatch")
    counts: list[int] = []
    probabilities: list[float] = []
    physical_bytes = prediction_manifest.stat().st_size + score_manifest.stat().st_size + diagnostics_path.stat().st_size
    for image_id, prediction in prediction_by_id.items():
        score = score_by_id[image_id]
        diagnostic = diagnostic_by_id[image_id]
        source = cache_rows[image_id]
        cache = _load_cache_payload(cache_root, source)
        for field in (
            "group_id",
            "tumor",
            "candidate_payload_sha256",
            "candidate_count",
            "selected_candidate_index",
            "selected_candidate_logit",
        ):
            if prediction[field] != score[field]:
                raise ValueError(f"T1 prediction/score mismatch: {image_id}/{field}")
        score_path = _safe_child(root / "candidate_scores", score["score_path"])
        if sha256_file(score_path) != score["score_sha256"]:
            raise ValueError(f"T1 candidate-score hash mismatch: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            if set(payload.files) != {"candidate_indices", "candidate_logits"}:
                raise ValueError(f"T1 candidate-score schema mismatch: {image_id}")
            indices = payload["candidate_indices"].copy()
            logits = payload["candidate_logits"].copy()
        count = int(prediction["candidate_count"])
        if (
            indices.dtype != np.int64
            or logits.dtype != np.float32
            or indices.shape != (count,)
            or logits.shape != (count,)
            or not np.array_equal(indices, cache["candidate_indices"].astype(np.int64))
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"T1 candidate-score content mismatch: {image_id}")
        winner = int(np.argmax(logits))
        if int(indices[winner]) != int(prediction["selected_candidate_index"]):
            raise ValueError(f"T1 selected candidate mismatch: {image_id}")
        _close(logits[winner], prediction["selected_candidate_logit"], name="selected logit")
        bag_logit = _normalized_smoothmax(logits)
        probability = _sigmoid(bag_logit)
        stored_probability = _finite(prediction["bag_probability"], name="stored bag probability")
        _close(prediction["bag_logit"], bag_logit, name="bag logit", atol=2.0e-6)
        _close(stored_probability, probability, name="bag probability", atol=2.0e-6)
        if diagnostic["candidate_count"] != prediction["candidate_count"]:
            raise ValueError(f"T1 diagnostic candidate count mismatch: {image_id}")
        residual_path = _safe_child(root / "validation_residual_evidence", diagnostic["residual_evidence_path"])
        if sha256_file(residual_path) != diagnostic["residual_evidence_sha256"]:
            raise ValueError(f"T1 residual-evidence hash mismatch: {image_id}")
        with np.load(residual_path, allow_pickle=False) as evidence:
            required = {
                "candidate_indices",
                "original_base_logits",
                "flipped_base_logits",
                "original_residual_logits",
                "flipped_residual_logits",
                "original_candidate_logits",
                "flipped_candidate_logits",
            }
            if set(evidence.files) != required:
                raise ValueError(f"T1 residual-evidence schema mismatch: {image_id}")
            values = {name: evidence[name].copy() for name in evidence.files}
        arrays = [values[name] for name in required if name != "candidate_indices"]
        if (
            not np.array_equal(values["candidate_indices"], cache["candidate_indices"])
            or any(array.dtype != np.float32 or array.shape != (count,) or not np.isfinite(array).all() for array in arrays)
            or not np.array_equal(
                values["original_candidate_logits"],
                values["original_base_logits"] + values["original_residual_logits"],
            )
            or not np.array_equal(
                values["flipped_candidate_logits"],
                values["flipped_base_logits"] + values["flipped_residual_logits"],
            )
            or not np.array_equal(
                logits,
                0.5 * (values["original_candidate_logits"] + values["flipped_candidate_logits"]),
            )
            or int(diagnostic["selected_view_agreement"])
            != int(np.argmax(values["original_candidate_logits"]) == np.argmax(values["flipped_candidate_logits"]))
        ):
            raise ValueError(f"T1 residual-evidence identity mismatch: {image_id}")
        map_path = _safe_child(root / "predictions", prediction["map_path"])
        if sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"T1 prediction-map hash mismatch: {image_id}")
        prediction_map = np.load(map_path, allow_pickle=False)
        if (
            prediction_map.shape != (320, 320)
            or prediction_map.dtype != np.float16
            or not np.isfinite(prediction_map).all()
            or "packed_masks" not in cache
        ):
            raise ValueError(f"T1 prediction-map content mismatch: {image_id}")
        height = int(cache["mask_height"])
        width = int(cache["mask_width"])
        unpacked = np.unpackbits(cache["packed_masks"], axis=1, count=height * width).reshape(count, height, width)
        expected_map = (unpacked[winner].astype(np.float32) * stored_probability).astype(np.float16)
        if not np.array_equal(prediction_map, expected_map):
            raise ValueError(f"T1 prediction map is not selected mask times probability: {image_id}")
        counts.append(count)
        probabilities.append(stored_probability)
        physical_bytes += score_path.stat().st_size + residual_path.stat().st_size + map_path.stat().st_size
    association = _absolute_spearman(counts, probabilities)
    _close(
        freeze["absolute_candidate_count_probability_spearman"],
        association,
        name="validation count/probability Spearman",
        atol=1.0e-12,
    )
    return {
        "physical_validation_maps_verified": 371,
        "physical_candidate_score_payloads_verified": 371,
        "physical_residual_evidence_payloads_verified": 371,
        "absolute_candidate_count_probability_spearman": association,
        "count_gate_pass": association <= COUNT_SPEARMAN_CEILING,
        "physical_validation_bytes_verified": physical_bytes,
    }


def audit_t1_output(
    root: Path,
    cache_root: Path,
    protocol_path: Path,
    launch_binding_path: Path,
) -> dict[str, object]:
    protocol = _json(protocol_path)
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("T1 protocol SHA-256 mismatch")
    binding = _verify_binding(launch_binding_path, protocol)
    paths = {
        "freeze": root / "prediction_freeze.json",
        "run": root / "run_manifest.json",
        "wrapper": root / "wrapper_output_audit.json",
        "assignment": root / "crossfit_assignment.json",
        "identity": root / "pretraining_identity_audit.json",
        "residual": root / "count_controlled_self_paced_residual.pt",
        "history": root / "consumer_training_history.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"T1 output is missing: {path}")
    freeze = _json(paths["freeze"])
    run = _json(paths["run"])
    wrapper = _json(paths["wrapper"])
    assignment = _json(paths["assignment"])
    for name, payload in (("prediction freeze", freeze), ("run manifest", run), ("wrapper audit", wrapper)):
        _require_safety(payload, name=name)
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("selector_cache_manifest_sha256") != CACHE_MANIFEST_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("validation_predictions") != 371
        or freeze.get("training_labels") != "image_level_only"
        or freeze.get("confirmation_residual_trained_after_producer_gate") is not True
    ):
        raise ValueError("T1 prediction-freeze provenance mismatch")
    if (
        run.get("run_id") != "btxrd_mask_bag_count_controlled_self_paced_t1_v1"
        or run.get("source_commit") != SOURCE_COMMIT
        or run.get("protocol_sha256") != PROTOCOL_SHA256
        or run.get("producer_model_config") != EXPECTED_MODEL_CONFIG
        or not _training_config_matches(run.get("training_config"))
        or run.get("validated_cache_records") != {"train": 2981, "validation": 371}
        or run.get("crossfit") != assignment
        or run.get("output_hashes") != freeze
    ):
        raise ValueError("T1 run-manifest contract mismatch")
    runtime = run.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("oof_parallel_workers") != 2
        or runtime.get("oof_folds_by_device") != [[0, 2, 4], [1, 3]]
        or runtime.get("validation_shards") != [186, 185]
    ):
        raise ValueError("T1 T4x2 runtime contract mismatch")
    cache_rows, cache_bytes = _verify_cache(cache_root)
    assignment_rows, assignment_summary = _recompute_crossfit_assignment(cache_rows)
    row_payload = json.dumps(assignment_rows, sort_keys=True, separators=(",", ":")).encode()
    if (
        assignment.get("schema_version") != 1
        or assignment.get("rows") != 2981
        or assignment.get("folds") != 5
        or assignment.get("fold_summary") != assignment_summary
        or assignment_summary != EXPECTED_FOLD_SUMMARY
        or assignment.get("row_payload_sha256") != hashlib.sha256(row_payload).hexdigest()
        or assignment.get("row_payload_sha256") != "407be430a6aa4408e1baf961ce0cd8eb55e6fe06b34640ceecb3bdfe0cb67ec5"
        or sha256_file(paths["assignment"]) != freeze.get("crossfit_assignment_sha256")
    ):
        raise ValueError("T1 cross-fit assignment contract mismatch")
    protocol_hashes = protocol["canonical_lf_source_hashes"]
    runtime_hashes = {
        path: digest
        for path, digest in protocol_hashes.items()
        if path not in POST_FREEZE_ONLY_SOURCE_PATHS
    }
    if (
        wrapper.get("kernel") != KERNEL
        or wrapper.get("kernel_version") != binding["kernel_version"]
        or wrapper.get("checkout_commit") != binding["checkout_commit"]
        or wrapper.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper.get("bound_wrapper_sha256") != binding["bound_wrapper_sha256"]
        or wrapper.get("source_hashes") != runtime_hashes
    ):
        raise ValueError("T1 wrapper-output provenance mismatch")
    t4 = wrapper.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("T1 wrapper T4x2 evidence mismatch")
    if (
        wrapper.get("cache", {}).get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or wrapper.get("cache", {}).get("selector_cache_wrapper_audit_sha256") != CACHE_WRAPPER_AUDIT_SHA256
        or wrapper.get("cache", {}).get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("T1 wrapper cache provenance mismatch")
    identity = _json(paths["identity"])
    _require_safety(identity, name="pretraining identity")
    if (
        identity.get("train", {}).get("records") != 2981
        or identity.get("validation", {}).get("records") != 371
        or identity.get("train", {}).get("zero_residual_exact") is not True
        or identity.get("validation", {}).get("zero_residual_exact") is not True
        or identity.get("train", {}).get("combined_equals_frozen_base_exact") is not True
        or identity.get("validation", {}).get("combined_equals_frozen_base_exact") is not True
        or sha256_file(paths["identity"]) != freeze.get("pretraining_identity_audit_sha256")
    ):
        raise ValueError("T1 pretraining identity audit mismatch")
    if sha256_file(paths["residual"]) != freeze.get("residual_checkpoint_sha256"):
        raise ValueError("T1 residual checkpoint hash mismatch")
    if sha256_file(paths["history"]) != freeze.get("consumer_training_history_sha256"):
        raise ValueError("T1 confirmation history hash mismatch")
    residual = _torch_checkpoint(paths["residual"])
    _require_safety(residual, name="confirmation residual checkpoint")
    if (
        residual.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or residual.get("model_config") != EXPECTED_MODEL_CONFIG
        or not _training_config_matches(residual.get("training_config"))
        or residual.get("target_freeze_sha256") != freeze.get("target_freeze_sha256")
        or residual.get("source_commit") != SOURCE_COMMIT
        or residual.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("T1 confirmation residual provenance mismatch")
    history = json.loads(paths["history"].read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 12
        or [int(row.get("epoch", -1)) for row in history] != list(range(1, 13))
        or [int(row.get("stage", -1)) for row in history] != [1] * 4 + [2] * 4 + [3] * 4
        or [float(row.get("pace_fraction", -1)) for row in history] != [0.2] * 4 + [0.4] * 4 + [0.6] * 4
        or any(
            not all(math.isfinite(float(row[key])) for key in ("total", "image", "contrastive", "count"))
            for row in history
        )
    ):
        raise ValueError("T1 confirmation history contract mismatch")
    oof_scores, oof = _verify_oof(root, cache_root, cache_rows, freeze, assignment_rows)
    targets = _verify_targets(root, freeze, cache_rows, oof_scores)
    validation = _verify_validation(root, cache_root, cache_rows, freeze)
    gate_pass = bool(all(oof["checks"].values()) and validation["count_gate_pass"])
    return {
        "audit_id": "independent_mask_bag_count_controlled_self_paced_t1_output_v1",
        "status": (
            "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
            if gate_pass
            else "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_FAIL"
        ),
        "kernel": KERNEL,
        "kernel_version": binding["kernel_version"],
        "checkout_commit": binding["checkout_commit"],
        "bound_wrapper_sha256": binding["bound_wrapper_sha256"],
        "launch_binding_sha256": sha256_file(launch_binding_path),
        "prediction_freeze_sha256": sha256_file(paths["freeze"]),
        "run_manifest_sha256": sha256_file(paths["run"]),
        "wrapper_output_audit_sha256": sha256_file(paths["wrapper"]),
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "cache_freeze_sha256": CACHE_FREEZE_SHA256,
        "oof": oof,
        "targets": targets,
        "validation": validation,
        "physical_output_and_cache_bytes_verified": (
            cache_bytes
            + int(oof["physical_oof_bytes_verified"])
            + int(targets["physical_target_bytes_verified"])
            + int(validation["physical_validation_bytes_verified"])
            + sum(path.stat().st_size for path in paths.values())
        ),
        "training_labels": "image_level_only",
        "confirmation_residual_trained_after_producer_gate": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch-binding", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_t1_output(
        args.output_root.resolve(),
        args.selector_cache_root.resolve(),
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
