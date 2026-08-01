from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen S4 output."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "EXP-20260801-codex-s4-oof-proposal-cluster-v1"
KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-proposal-cluster-s4-v1"
SOURCE_COMMIT = "95c4a3378eaf8463c57d57a0dd4e4cac6c69021f"
PROTOCOL_SHA256 = "040227de1347c45bc1823bd5aef5d9614b8005619ecc35d9dceb45bb7eba71e8"
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
    "epochs": 16,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "teacher_instance_loss_weight": 0.25,
    "consistency_weight": 0.1,
    "instance_warmup_epochs": 2,
    "maximum_clusters": 4,
    "minimum_iou": 0.5,
    "minimum_containment": 0.75,
    "start_temperature": 1.0,
    "end_temperature": 0.2,
    "residual_hidden_dim": 128,
    "seed": 42,
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


def _close(
    actual: object,
    expected: object,
    *,
    name: str,
    atol: float = 1.0e-7,
) -> None:
    if abs(_finite(actual, name=name) - _finite(expected, name=name)) > atol:
        raise ValueError(f"{name} differs: {actual} versus {expected}")


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
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
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must be aligned and nontrivial")
    left = _rankdata(first)
    right = _rankdata(second)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("Spearman ranks must be nonconstant")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _normalized_smoothmax(values: np.ndarray, temperature: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("SmoothMax needs a finite nonempty vector")
    scaled = array / temperature
    maximum = float(scaled.max())
    return float(
        temperature
        * (
            maximum
            + math.log(float(np.exp(scaled - maximum).sum()))
            - math.log(len(array))
        )
    )


def _cluster_bag_logit(logits: np.ndarray, clusters: np.ndarray) -> float:
    cluster_logits = [
        _normalized_smoothmax(logits[members], 0.2)
        for members in clusters.astype(bool)
        if members.any()
    ]
    return _normalized_smoothmax(np.asarray(cluster_logits), 0.2)


def _verify_binding(
    binding_path: Path,
    protocol: Mapping[str, object],
) -> dict[str, Any]:
    binding = _json(binding_path)
    protocol_hashes = protocol.get("canonical_lf_source_hashes")
    if not isinstance(protocol_hashes, dict):
        raise ValueError("S4 protocol source hashes are missing")
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
        raise ValueError("S4 launch binding contract mismatch")
    return binding


def _verify_cache(
    cache_root: Path,
) -> tuple[dict[str, dict[str, str]], int]:
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
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val")
    }
    if split_counts != {"train": 2981, "val": 371}:
        raise ValueError("Selector-cache split counts mismatch")
    return indexed, manifest_path.stat().st_size


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
            raise ValueError("S4 cross-fit source group contains mixed labels")
        group_records.append(
            (str(group), int(group_labels[0]), int(np.count_nonzero(members)))
        )
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
            for (group, _label, _size), value in zip(
                records, rng.random(len(records)), strict=True
            )
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


def _load_cache_payload(
    cache_root: Path,
    cache_row: Mapping[str, str],
) -> dict[str, np.ndarray]:
    path = _safe_child(cache_root, cache_row["cache_path"])
    if sha256_file(path) != cache_row["cache_sha256"]:
        raise ValueError(f"Selector-cache record hash mismatch: {cache_row['image_id']}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "candidate_indices",
            "pairwise_iou",
            "pairwise_containment",
            "fallback_flags",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"Selector-cache fields missing: {cache_row['image_id']}")
        result = {name: payload[name].copy() for name in required}
        for name in ("packed_masks", "mask_height", "mask_width"):
            if name in payload.files:
                result[name] = payload[name].copy()
    return result


def _recompute_clusters(
    seed_logits: np.ndarray,
    iou: np.ndarray,
    containment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(seed_logits)
    adjacency = (iou >= 0.5) | (containment >= 0.75)
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    clusters = np.zeros((4, count), dtype=np.uint8)
    valid = np.zeros(4, dtype=np.uint8)
    seeds = np.full(4, -1, dtype=np.int32)
    assigned = np.zeros(count, dtype=bool)
    order = np.argsort(-seed_logits, kind="stable")
    cluster = 0
    for seed in order:
        if assigned[seed]:
            continue
        members = (~assigned) & adjacency[seed]
        members[seed] = True
        clusters[cluster] = members
        valid[cluster] = 1
        seeds[cluster] = int(seed)
        assigned |= members
        cluster += 1
        if cluster == 4:
            break
    return clusters, valid, seeds


def _verify_score_manifest(
    manifest_path: Path,
    score_root: Path,
    *,
    expected_rows: int,
    expected_fold: int | None,
) -> tuple[dict[str, dict[str, str]], int]:
    rows = _csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != expected_rows or len(indexed) != expected_rows:
        raise ValueError(f"Teacher score cohort mismatch: {manifest_path}")
    physical_bytes = manifest_path.stat().st_size
    for row in rows:
        if expected_fold is not None and int(row["heldout_fold"]) != expected_fold:
            raise ValueError("OOF teacher score fold mismatch")
        path = _safe_child(score_root, row["payload_path"])
        if sha256_file(path) != row["payload_sha256"]:
            raise ValueError(f"Teacher score payload hash mismatch: {row['image_id']}")
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                "candidate_indices",
                "original_logits",
                "flipped_logits",
                "conservative_seed_logits",
            }
            if set(payload.files) != expected:
                raise ValueError(f"Teacher score payload schema mismatch: {row['image_id']}")
            indices = payload["candidate_indices"]
            original = payload["original_logits"]
            flipped = payload["flipped_logits"]
            conservative = payload["conservative_seed_logits"]
        count = int(row["candidate_count"])
        if (
            indices.dtype != np.int32
            or original.dtype != np.float32
            or flipped.dtype != np.float32
            or conservative.dtype != np.float32
            or indices.shape != (count,)
            or original.shape != (count,)
            or flipped.shape != (count,)
            or conservative.shape != (count,)
            or not np.array_equal(conservative, np.minimum(original, flipped))
            or not all(np.isfinite(array).all() for array in (original, flipped, conservative))
        ):
            raise ValueError(f"Teacher score payload content mismatch: {row['image_id']}")
        probability = _finite(row["bag_probability"], name="teacher bag probability")
        expected_agreement = int(np.argmax(original)) == int(np.argmax(flipped))
        if (
            not 0.0 <= probability <= 1.0
            or int(row["selected_view_agreement"]) != int(expected_agreement)
        ):
            raise ValueError(f"Teacher score scalar mismatch: {row['image_id']}")
        physical_bytes += path.stat().st_size
    return indexed, physical_bytes


def _verify_oof(
    root: Path,
    freeze: Mapping[str, object],
    assignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    inventory = freeze.get("oof_teacher_hashes")
    if not isinstance(inventory, dict) or set(inventory) != {
        f"fold_{fold}" for fold in FOLDS
    }:
        raise ValueError("S4 OOF teacher inventory mismatch")
    if len(assignment_rows) != 2981:
        raise ValueError("S4 cross-fit assignment rows missing")
    assignment_by_id = {str(row["image_id"]): row for row in assignment_rows}
    score_rows: dict[str, dict[str, str]] = {}
    physical_bytes = 0
    training_group_sets: dict[int, set[str]] = {}
    for fold in FOLDS:
        fold_root = root / "oof_teachers" / f"fold_{fold}"
        expected = inventory[f"fold_{fold}"]
        paths = {
            "teacher_checkpoint_sha256": fold_root / "teacher.pt",
            "training_history_sha256": fold_root / "training_history.json",
            "score_manifest_sha256": fold_root / "score_manifest.csv",
            "fold_audit_sha256": fold_root / "fold_audit.json",
        }
        for key, path in paths.items():
            if not path.is_file() or sha256_file(path) != expected.get(key):
                raise ValueError(f"S4 OOF artifact hash mismatch: fold={fold}, {key}")
            physical_bytes += path.stat().st_size
        history = json.loads(paths["training_history_sha256"].read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != 16:
            raise ValueError(f"S4 OOF history length mismatch: fold={fold}")
        audit = _json(paths["fold_audit_sha256"])
        training_groups = {str(value) for value in audit.get("training_groups", [])}
        heldout_groups = {str(value) for value in audit.get("heldout_groups", [])}
        if (
            audit.get("heldout_fold") != fold
            or audit.get("derived_seed") != 1042 + fold
            or audit.get("group_overlap") != 0
            or audit.get("validation_segmentation_quality_used") is not False
            or not training_groups
            or not heldout_groups
            or training_groups & heldout_groups
        ):
            raise ValueError(f"S4 OOF group exclusion mismatch: fold={fold}")
        training_group_sets[fold] = training_groups
        expected_rows = sum(
            int(row["heldout_fold"]) == fold for row in assignment_rows
        )
        indexed, score_bytes = _verify_score_manifest(
            paths["score_manifest_sha256"],
            fold_root / "scores",
            expected_rows=expected_rows,
            expected_fold=fold,
        )
        physical_bytes += score_bytes - paths["score_manifest_sha256"].stat().st_size
        for image_id, row in indexed.items():
            assigned = assignment_by_id.get(image_id)
            if (
                assigned is None
                or int(assigned["heldout_fold"]) != fold
                or assigned["group_id"] != row["group_id"]
                or int(assigned["image_label"]) != int(row["image_label"])
                or row["group_id"] not in heldout_groups
                or image_id in score_rows
            ):
                raise ValueError(f"S4 OOF score identity mismatch: {image_id}")
            score_rows[image_id] = row
    if len(score_rows) != 2981:
        raise ValueError("S4 OOF scores do not cover clean train exactly once")
    for row in assignment_rows:
        fold = int(row["heldout_fold"])
        if row["group_id"] in training_group_sets[fold]:
            raise ValueError("S4 OOF teacher trained on the held-out group")
    association = _absolute_spearman(
        [int(score_rows[str(row["image_id"])]["candidate_count"]) for row in assignment_rows],
        [float(score_rows[str(row["image_id"])]["bag_probability"]) for row in assignment_rows],
    )
    coverage_path = root / "oof_coverage_audit.json"
    if sha256_file(coverage_path) != freeze.get("oof_coverage_audit_sha256"):
        raise ValueError("S4 OOF coverage audit hash mismatch")
    coverage = _json(coverage_path)
    if (
        coverage.get("complete") is not True
        or coverage.get("records") != 2981
        or coverage.get("folds") != 5
        or coverage.get("group_overlap") != 0
    ):
        raise ValueError("S4 OOF coverage audit contract mismatch")
    _close(
        coverage["absolute_candidate_count_probability_spearman"],
        association,
        name="OOF count/probability Spearman",
        atol=1.0e-12,
    )
    return {
        "physical_oof_bytes_verified": physical_bytes + coverage_path.stat().st_size,
        "physical_oof_score_payloads_verified": 2981,
        "absolute_candidate_count_probability_spearman": association,
        "count_gate_pass": association <= COUNT_SPEARMAN_CEILING,
    }


def _verify_cluster_split(
    root: Path,
    cache_root: Path,
    cache_rows: Mapping[str, Mapping[str, str]],
    freeze: Mapping[str, object],
    *,
    split: str,
    expected_rows: int,
    teacher_scores: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], int]:
    manifest_path = root / "clusters" / f"{split}_cluster_manifest.csv"
    freeze_key = (
        "validation_cluster_manifest_sha256"
        if split == "val"
        else "train_cluster_manifest_sha256"
    )
    if sha256_file(manifest_path) != freeze.get(freeze_key):
        raise ValueError(f"S4 {split} cluster manifest hash mismatch")
    rows = _csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != expected_rows or len(indexed) != expected_rows:
        raise ValueError(f"S4 {split} cluster cohort mismatch")
    output: dict[str, dict[str, Any]] = {}
    physical_bytes = manifest_path.stat().st_size
    for image_id, row in indexed.items():
        cache_row = cache_rows.get(image_id)
        teacher_row = teacher_scores.get(image_id)
        if (
            cache_row is None
            or teacher_row is None
            or cache_row["split"] != split
            or cache_row["group_id"] != row["group_id"]
            or int(cache_row["tumor"]) != int(row["image_label"])
        ):
            raise ValueError(f"S4 {split} cluster identity mismatch: {image_id}")
        cache = _load_cache_payload(cache_root, cache_row)
        path = _safe_child(root / "clusters" / split, row["payload_path"])
        if sha256_file(path) != row["payload_sha256"]:
            raise ValueError(f"S4 {split} cluster payload hash mismatch: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            expected_fields = {
                "candidate_indices",
                "teacher_original_logits",
                "teacher_flipped_logits",
                "teacher_conservative_seed_logits",
                "clusters",
                "cluster_valid",
                "seed_indices",
            }
            if set(payload.files) != expected_fields:
                raise ValueError(f"S4 {split} cluster schema mismatch: {image_id}")
            values = {name: payload[name].copy() for name in payload.files}
        score_path = _safe_child(
            (
                root / "oof_teachers" / f"fold_{int(row['heldout_fold'])}" / "scores"
                if split == "train"
                else root / "full_teacher_validation_scores" / "scores"
            ),
            teacher_row["payload_path"],
        )
        with np.load(score_path, allow_pickle=False) as score:
            original = score["original_logits"]
            flipped = score["flipped_logits"]
            conservative = score["conservative_seed_logits"]
        if (
            not np.array_equal(values["candidate_indices"], cache["candidate_indices"])
            or not np.array_equal(values["teacher_original_logits"], original)
            or not np.array_equal(values["teacher_flipped_logits"], flipped)
            or not np.array_equal(values["teacher_conservative_seed_logits"], conservative)
        ):
            raise ValueError(f"S4 {split} teacher/cluster evidence mismatch: {image_id}")
        recomputed = _recompute_clusters(
            conservative,
            np.asarray(cache["pairwise_iou"], dtype=np.float32),
            np.asarray(cache["pairwise_containment"], dtype=np.float32),
        )
        if not all(
            np.array_equal(values[name], expected)
            for name, expected in zip(
                ("clusters", "cluster_valid", "seed_indices"),
                recomputed,
            )
        ):
            raise ValueError(f"S4 {split} cluster recomputation mismatch: {image_id}")
        clusters = values["clusters"].astype(bool)
        if (
            np.any(clusters.sum(axis=0) > 1)
            or int(values["cluster_valid"].sum()) != int(row["cluster_count"])
            or int(clusters.any(axis=0).sum()) != int(row["cluster_member_count"])
        ):
            raise ValueError(f"S4 {split} cluster disjointness/count mismatch: {image_id}")
        output[image_id] = {
            "row": row,
            "clusters": clusters,
            "cache": cache,
        }
        physical_bytes += path.stat().st_size
    return output, physical_bytes


def _verify_validation(
    root: Path,
    freeze: Mapping[str, object],
    clusters: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    diagnostics_path = root / "gt_blind_diagnostics.csv"
    if sha256_file(prediction_manifest) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("S4 prediction manifest hash mismatch")
    if sha256_file(score_manifest) != freeze.get("candidate_score_manifest_sha256"):
        raise ValueError("S4 candidate-score manifest hash mismatch")
    if sha256_file(diagnostics_path) != freeze.get("gt_blind_diagnostics_sha256"):
        raise ValueError("S4 GT-blind diagnostics hash mismatch")
    predictions = _csv(prediction_manifest)
    scores = _csv(score_manifest)
    diagnostics = _csv(diagnostics_path)
    prediction_by_id = {row["image_id"]: row for row in predictions}
    score_by_id = {row["image_id"]: row for row in scores}
    diagnostic_by_id = {row["image_id"]: row for row in diagnostics}
    if (
        len(prediction_by_id) != 371
        or len(score_by_id) != 371
        or len(diagnostic_by_id) != 371
        or set(prediction_by_id) != set(score_by_id)
        or set(prediction_by_id) != set(diagnostic_by_id)
        or set(prediction_by_id) != set(clusters)
    ):
        raise ValueError("S4 frozen validation cohort mismatch")
    counts: list[int] = []
    probabilities: list[float] = []
    physical_bytes = (
        prediction_manifest.stat().st_size
        + score_manifest.stat().st_size
        + diagnostics_path.stat().st_size
    )
    outside_candidates = 0
    for image_id, prediction in prediction_by_id.items():
        score = score_by_id[image_id]
        diagnostic = diagnostic_by_id[image_id]
        cluster = clusters[image_id]
        cache = cluster["cache"]
        for field in (
            "group_id",
            "tumor",
            "candidate_payload_sha256",
            "candidate_count",
            "selected_candidate_index",
            "selected_candidate_logit",
        ):
            if prediction[field] != score[field]:
                raise ValueError(f"S4 prediction/score mismatch: {image_id}/{field}")
        score_path = _safe_child(root / "candidate_scores", score["score_path"])
        if sha256_file(score_path) != score["score_sha256"]:
            raise ValueError(f"S4 candidate-score hash mismatch: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            indices = payload["candidate_indices"]
            logits = payload["candidate_logits"]
        count = int(prediction["candidate_count"])
        if (
            indices.dtype != np.int64
            or logits.dtype != np.float32
            or indices.shape != (count,)
            or logits.shape != (count,)
            or not np.array_equal(indices, cache["candidate_indices"].astype(np.int64))
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"S4 candidate-score content mismatch: {image_id}")
        winner = int(np.argmax(logits))
        if int(indices[winner]) != int(prediction["selected_candidate_index"]):
            raise ValueError(f"S4 selected candidate mismatch: {image_id}")
        _close(logits[winner], prediction["selected_candidate_logit"], name="selected logit")
        bag_logit = _cluster_bag_logit(logits, cluster["clusters"])
        probability = _sigmoid(bag_logit)
        stored_probability = _finite(
            prediction["bag_probability"], name="stored cluster bag probability"
        )
        _close(prediction["bag_logit"], bag_logit, name="cluster bag logit", atol=2.0e-6)
        _close(stored_probability, probability, name="cluster bag probability", atol=2.0e-6)
        if (
            diagnostic["candidate_count"] != prediction["candidate_count"]
            or int(diagnostic["outside_cluster_original_residual_exact_zero"]) != 1
            or int(diagnostic["outside_cluster_flipped_residual_exact_zero"]) != 1
        ):
            raise ValueError(f"S4 outside-cluster gate mismatch: {image_id}")
        expected_outside = count - int(cluster["clusters"].any(axis=0).sum())
        if int(diagnostic["outside_cluster_count"]) != expected_outside:
            raise ValueError(f"S4 outside-cluster count mismatch: {image_id}")
        residual_path = _safe_child(
            root / "validation_residual_evidence",
            diagnostic["residual_evidence_path"],
        )
        if sha256_file(residual_path) != diagnostic["residual_evidence_sha256"]:
            raise ValueError(f"S4 residual-evidence hash mismatch: {image_id}")
        with np.load(residual_path, allow_pickle=False) as evidence:
            required = {
                "candidate_indices",
                "cluster_member_flags",
                "original_base_logits",
                "flipped_base_logits",
                "original_residual_logits",
                "flipped_residual_logits",
                "original_candidate_logits",
                "flipped_candidate_logits",
            }
            if set(evidence.files) != required:
                raise ValueError(f"S4 residual-evidence schema mismatch: {image_id}")
            residual_values = {name: evidence[name].copy() for name in evidence.files}
        members = cluster["clusters"].any(axis=0)
        original_base = residual_values["original_base_logits"]
        flipped_base = residual_values["flipped_base_logits"]
        original_residual = residual_values["original_residual_logits"]
        flipped_residual = residual_values["flipped_residual_logits"]
        original_combined = residual_values["original_candidate_logits"]
        flipped_combined = residual_values["flipped_candidate_logits"]
        if (
            not np.array_equal(
                residual_values["candidate_indices"],
                cache["candidate_indices"],
            )
            or not np.array_equal(
                residual_values["cluster_member_flags"].astype(bool),
                members,
            )
            or any(
                np.asarray(array).dtype != np.float32
                or np.asarray(array).shape != (count,)
                or not np.isfinite(array).all()
                for array in (
                    original_base,
                    flipped_base,
                    original_residual,
                    flipped_residual,
                    original_combined,
                    flipped_combined,
                )
            )
            or not np.array_equal(original_combined, original_base + original_residual)
            or not np.array_equal(flipped_combined, flipped_base + flipped_residual)
            or not np.array_equal(
                logits,
                0.5 * (original_combined + flipped_combined),
            )
            or np.count_nonzero(original_residual[~members]) != 0
            or np.count_nonzero(flipped_residual[~members]) != 0
            or int(diagnostic["final_selected_view_agreement"])
            != int(np.argmax(original_combined) == np.argmax(flipped_combined))
        ):
            raise ValueError(f"S4 residual-evidence identity mismatch: {image_id}")
        map_path = _safe_child(root / "predictions", prediction["map_path"])
        if sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"S4 prediction-map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or "packed_masks" not in cache
        ):
            raise ValueError(f"S4 prediction-map content mismatch: {image_id}")
        height = int(cache["mask_height"])
        width = int(cache["mask_width"])
        unpacked = np.unpackbits(
            cache["packed_masks"], axis=1, count=height * width
        ).reshape(count, height, width)
        expected_map = (
            unpacked[winner].astype(np.float32) * stored_probability
        ).astype(np.float16)
        if not np.array_equal(values, expected_map):
            raise ValueError(f"S4 prediction map is not selected mask times probability: {image_id}")
        counts.append(count)
        probabilities.append(stored_probability)
        outside_candidates += expected_outside
        physical_bytes += (
            score_path.stat().st_size
            + map_path.stat().st_size
            + residual_path.stat().st_size
        )
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
        "outside_cluster_candidates_verified": outside_candidates,
        "absolute_candidate_count_probability_spearman": association,
        "count_gate_pass": association <= COUNT_SPEARMAN_CEILING,
        "physical_validation_bytes_verified": physical_bytes,
    }


def audit_s4_output(
    root: Path,
    cache_root: Path,
    protocol_path: Path,
    launch_binding_path: Path,
) -> dict[str, object]:
    protocol = _json(protocol_path)
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("S4 protocol SHA-256 mismatch")
    binding = _verify_binding(launch_binding_path, protocol)
    paths = {
        "freeze": root / "prediction_freeze.json",
        "run": root / "run_manifest.json",
        "wrapper": root / "wrapper_output_audit.json",
        "assignment": root / "crossfit_assignment.json",
        "full_group": root / "full_teacher_group_exclusion_audit.json",
        "identity": root / "pretraining_identity_audit.json",
        "full_teacher": root / "full_train_teacher.pt",
        "full_history": root / "full_train_teacher_history.json",
        "residual": root / "proposal_cluster_residual.pt",
        "residual_history": root / "residual_training_history.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"S4 output is missing: {path}")
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
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("validation_predictions") != 371
        or freeze.get("training_labels") != "image_level_only"
    ):
        raise ValueError("S4 prediction-freeze provenance mismatch")
    if (
        run.get("run_id") != "btxrd_mask_bag_proposal_cluster_s4_v1"
        or run.get("source_commit") != SOURCE_COMMIT
        or run.get("protocol_sha256") != PROTOCOL_SHA256
        or run.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or run.get("training_config") != EXPECTED_TRAINING_CONFIG
        or run.get("validated_cache_records") != {"train": 2981, "validation": 371}
        or run.get("crossfit") != assignment
        or run.get("output_hashes") != freeze
    ):
        raise ValueError("S4 run-manifest contract mismatch")
    runtime = run.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("oof_parallel_workers") != 2
        or runtime.get("oof_folds_by_device") != [[0, 2, 4], [1, 3]]
        or runtime.get("validation_shards") != [186, 185]
    ):
        raise ValueError("S4 T4x2 runtime contract mismatch")
    cache_rows, cache_bytes = _verify_cache(cache_root)
    assignment_rows, assignment_summary = _recompute_crossfit_assignment(cache_rows)
    row_payload = json.dumps(
        assignment_rows, sort_keys=True, separators=(",", ":")
    ).encode()
    if (
        assignment.get("schema_version") != 1
        or assignment.get("rows") != 2981
        or assignment.get("folds") != 5
        or assignment.get("fold_summary") != assignment_summary
        or assignment_summary != EXPECTED_FOLD_SUMMARY
        or assignment.get("row_payload_sha256")
        != hashlib.sha256(row_payload).hexdigest()
        or sha256_file(paths["assignment"]) != freeze.get("crossfit_assignment_sha256")
    ):
        raise ValueError("S4 cross-fit assignment contract mismatch")
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
        raise ValueError("S4 wrapper-output provenance mismatch")
    t4 = wrapper.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
    ):
        raise ValueError("S4 wrapper T4x2 evidence mismatch")
    if (
        wrapper.get("cache", {}).get("selector_cache_freeze_sha256")
        != CACHE_FREEZE_SHA256
        or wrapper.get("cache", {}).get("selector_cache_wrapper_audit_sha256")
        != CACHE_WRAPPER_AUDIT_SHA256
        or wrapper.get("cache", {}).get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("S4 wrapper cache provenance mismatch")
    full_group = _json(paths["full_group"])
    training_groups = set(full_group.get("training_groups", []))
    validation_groups = set(full_group.get("validation_groups", []))
    if (
        full_group.get("group_overlap") != 0
        or not training_groups
        or not validation_groups
        or training_groups & validation_groups
        or training_groups
        != {
            row["group_id"]
            for row in cache_rows.values()
            if row["split"] == "train"
        }
        or validation_groups
        != {
            row["group_id"]
            for row in cache_rows.values()
            if row["split"] == "val"
        }
        or sha256_file(paths["full_group"])
        != freeze.get("full_teacher_group_exclusion_audit_sha256")
    ):
        raise ValueError("S4 full-teacher group exclusion mismatch")
    identity = _json(paths["identity"])
    if (
        identity.get("source_commit") != SOURCE_COMMIT
        or identity.get("protocol_sha256") != PROTOCOL_SHA256
        or identity.get("train", {}).get("records") != 2981
        or identity.get("validation", {}).get("records") != 371
        or identity.get("train", {}).get("zero_residual_exact") is not True
        or identity.get("validation", {}).get("zero_residual_exact") is not True
        or sha256_file(paths["identity"]) != freeze.get("pretraining_identity_audit_sha256")
    ):
        raise ValueError("S4 pretraining identity audit mismatch")
    hash_bindings = {
        "full_teacher_checkpoint_sha256": paths["full_teacher"],
        "full_teacher_history_sha256": paths["full_history"],
        "residual_checkpoint_sha256": paths["residual"],
        "residual_training_history_sha256": paths["residual_history"],
    }
    for key, path in hash_bindings.items():
        if sha256_file(path) != freeze.get(key):
            raise ValueError(f"S4 final artifact hash mismatch: {key}")
    for history_path in (paths["full_history"], paths["residual_history"]):
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != 16:
            raise ValueError("S4 final history must contain 16 epochs")
    oof = _verify_oof(root, freeze, assignment_rows)
    full_score_manifest = root / "full_teacher_validation_scores" / "score_manifest.csv"
    if sha256_file(full_score_manifest) != freeze.get(
        "validation_teacher_score_manifest_sha256"
    ):
        raise ValueError("S4 full-teacher validation-score manifest mismatch")
    full_scores, full_score_bytes = _verify_score_manifest(
        full_score_manifest,
        root / "full_teacher_validation_scores" / "scores",
        expected_rows=371,
        expected_fold=None,
    )
    train_scores: dict[str, dict[str, str]] = {}
    for fold in FOLDS:
        manifest = root / "oof_teachers" / f"fold_{fold}" / "score_manifest.csv"
        rows = _csv(manifest)
        for row in rows:
            train_scores[row["image_id"]] = row
    train_clusters, train_cluster_bytes = _verify_cluster_split(
        root,
        cache_root,
        cache_rows,
        freeze,
        split="train",
        expected_rows=2981,
        teacher_scores=train_scores,
    )
    val_clusters, val_cluster_bytes = _verify_cluster_split(
        root,
        cache_root,
        cache_rows,
        freeze,
        split="val",
        expected_rows=371,
        teacher_scores=full_scores,
    )
    validation = _verify_validation(root, freeze, val_clusters)
    gate_pass = bool(
        oof["count_gate_pass"]
        and validation["count_gate_pass"]
        and len(train_clusters) == 2981
        and len(val_clusters) == 371
    )
    return {
        "audit_id": "independent_mask_bag_proposal_cluster_s4_output_v1",
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
        "train_cluster_payloads_verified": len(train_clusters),
        "validation_cluster_payloads_verified": len(val_clusters),
        "validation": validation,
        "physical_output_and_cache_bytes_verified": (
            cache_bytes
            + int(oof["physical_oof_bytes_verified"])
            + full_score_bytes
            + train_cluster_bytes
            + val_cluster_bytes
            + int(validation["physical_validation_bytes_verified"])
            + sum(path.stat().st_size for path in paths.values())
        ),
        "training_labels": "image_level_only",
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
    result = audit_s4_output(
        args.output_root.resolve(),
        args.selector_cache_root.resolve(),
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
