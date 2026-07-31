from __future__ import annotations

"""GT-blind physical auditor for the frozen R1 normal-prototype output."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


KERNEL = "itsthang333/btxrd-rad-dino-mask-bag-normal-prototype-r1-v1"
KERNEL_VERSION = 3
CHECKOUT_COMMIT = "3647997d0c18ad31057709462fd8c922d939fb4f"
SOURCE_COMMIT = "d66c89958baa3344dbbaae6030a9ccd8ecec7b3a"
PROTOCOL_SHA256 = "dded9c638e142576fedf0ae4c8102fdf64198744a4949707865e50b7081f312b"
BOUND_WRAPPER_SHA256 = "a896d45dad1efb46d3f0b7544f01ca152e18a5e0f3eabd95f5f00414efe33114"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_WRAPPER_AUDIT_SHA256 = (
    "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
)
BASELINE_CHECKPOINT_SHA256 = (
    "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
)
BASELINE_ABSOLUTE_COUNT_SPEARMAN = 0.48137777593654113
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
    "prototype_temperature": 0.10,
    "adapter_hidden_dim": 128,
    "seed": 42,
}
PROTOTYPE_COUNTS = (8, 16, 32)
FOLDS = tuple(range(5))
DESCRIPTOR_DIM = 1156
POST_FREEZE_ONLY_SOURCE_PATHS = {
    "project/evaluate_mask_bag_selector_arm.py",
    "project/models/mask_bag_ranking_diagnostics.py",
    "tests/test_evaluate_mask_bag_selector_arm.py",
}
EXPECTED_FOLD_SUMMARY = [
    {"fold": 0, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 1, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 2, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 3, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 4, "images": 597, "groups": 198, "normal_images": 299, "tumor_images": 298},
]


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
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Output manifest path is absolute: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"Output manifest path escapes its root: {relative}")
    return resolved


def _require_safety(payload: Mapping[str, object], *, name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"{name} safety boundary mismatch")


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _close(actual: object, expected: object, *, name: str, atol: float = 1.0e-7) -> None:
    if abs(_finite_float(actual, name=name) - _finite_float(expected, name=name)) > atol:
        raise ValueError(f"{name} differs: {actual} versus {expected}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 2:
        raise ValueError("Spearman inputs must be aligned nontrivial vectors")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _binary_bce(logit: float, label: int) -> float:
    return max(logit, 0.0) - logit * label + math.log1p(math.exp(-abs(logit)))


def _smooth_pool(logits: np.ndarray, temperature: float = 0.20) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Candidate logits must be one finite nonempty vector")
    scaled = values / temperature
    maximum = float(scaled.max())
    return float(temperature * (maximum + math.log(float(np.exp(scaled - maximum).sum())) - math.log(len(values))))


def _recompute_selection(candidates: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    if {int(row["prototype_count"]) for row in candidates} != set(PROTOTYPE_COUNTS):
        raise ValueError("R1 selection must contain the frozen K set")
    maximum_allowed = BASELINE_ABSOLUTE_COUNT_SPEARMAN + 0.02
    normalized: list[dict[str, Any]] = []
    for row in candidates:
        losses = np.asarray(row["fold_image_bce"], dtype=np.float64)
        if losses.shape != (5,) or not np.isfinite(losses).all() or np.any(losses < 0.0):
            raise ValueError("R1 fold BCE values have an invalid contract")
        association = abs(_finite_float(row["count_probability_spearman"], name="count Spearman"))
        normalized.append(
            {
                "prototype_count": int(row["prototype_count"]),
                "mean_oof_image_bce": float(losses.mean()),
                "standard_error_oof_image_bce": float(losses.std(ddof=1) / math.sqrt(5.0)),
                "absolute_count_probability_spearman": association,
                "count_guard_pass": association <= maximum_allowed,
            }
        )
    eligible = [row for row in normalized if row["count_guard_pass"]]
    if not eligible:
        raise ValueError("Every frozen K fails the count-shortcut guard")
    best = min(eligible, key=lambda row: (row["mean_oof_image_bce"], row["prototype_count"]))
    limit = best["mean_oof_image_bce"] + best["standard_error_oof_image_bce"]
    selected = min(
        (row for row in eligible if row["mean_oof_image_bce"] <= limit),
        key=lambda row: row["prototype_count"],
    )
    return {
        "rule": "minimum_K_within_best_five_fold_OOF_BCE_one_standard_error",
        "selected_prototype_count": selected["prototype_count"],
        "best_mean_prototype_count": best["prototype_count"],
        "one_standard_error_limit": limit,
        "maximum_allowed_absolute_count_probability_spearman": maximum_allowed,
        "candidates": sorted(normalized, key=lambda row: row["prototype_count"]),
        "validation_segmentation_quality_used": False,
    }


def _verify_prototype_payload(path: Path, *, prototype_count: int) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"prototypes"}:
            raise ValueError(f"Unexpected prototype payload fields: {path}")
        prototypes = payload["prototypes"]
    if (
        prototypes.shape != (prototype_count, DESCRIPTOR_DIM)
        or prototypes.dtype != np.float32
        or not np.isfinite(prototypes).all()
    ):
        raise ValueError(f"Prototype payload content mismatch: {path}")
    norms = np.linalg.norm(prototypes.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2.0e-5, rtol=0.0):
        raise ValueError(f"Prototype rows are not unit-normalized: {path}")


def _verify_oof(
    root: Path,
    freeze: Mapping[str, object],
    selection: Mapping[str, object],
    *,
    expected_train: int,
) -> dict[str, object]:
    inventory = freeze.get("oof_artifact_hashes")
    if not isinstance(inventory, dict):
        raise ValueError("Prediction freeze lacks the OOF inventory")
    expected_keys = {
        *(f"k_{k}_fold_{fold}" for k in PROTOTYPE_COUNTS for fold in FOLDS),
        *(f"k_{k}_aggregate" for k in PROTOTYPE_COUNTS),
    }
    if set(inventory) != expected_keys:
        raise ValueError("Prediction freeze OOF inventory differs from 15 folds plus 3 aggregates")

    reference_identity: Optional[dict[str, tuple[str, int, int]]] = None
    selection_inputs: list[dict[str, object]] = []
    physical_files = 0
    physical_bytes = 0
    for k in PROTOTYPE_COUNTS:
        aggregate_rows: list[dict[str, str]] = []
        fold_bce: list[float] = []
        expected_exclusion: list[dict[str, int]] = []
        for fold in FOLDS:
            fold_root = root / "oof" / f"k_{k}" / f"fold_{fold}"
            expected = inventory[f"k_{k}_fold_{fold}"]
            if not isinstance(expected, dict):
                raise ValueError("OOF fold inventory entry must be an object")
            paths = {
                "prototype_sha256": fold_root / "normal_prototypes.npz",
                "adapter_sha256": fold_root / "adapter.pt",
                "audit_sha256": fold_root / "fold_audit.json",
                "predictions_sha256": fold_root / "heldout_predictions.csv",
            }
            for key, path in paths.items():
                if not path.is_file() or sha256_file(path) != expected.get(key):
                    raise ValueError(f"OOF artifact hash mismatch: k={k}, fold={fold}, {key}")
                physical_files += 1
                physical_bytes += path.stat().st_size
            _verify_prototype_payload(paths["prototype_sha256"], prototype_count=k)
            audit = _json(paths["audit_sha256"])
            training_groups = {str(value) for value in audit.get("training_groups", [])}
            heldout_groups = {str(value) for value in audit.get("heldout_groups", [])}
            if (
                audit.get("prototype_count") != k
                or audit.get("heldout_fold") != fold
                or audit.get("derived_seed") != 42 + 1000 * k + fold
                or audit.get("group_overlap") != 0
                or audit.get("validation_segmentation_quality_used") is not False
                or not training_groups
                or not heldout_groups
                or training_groups & heldout_groups
            ):
                raise ValueError(f"OOF fold exclusion/provenance mismatch: k={k}, fold={fold}")
            rows = _csv(paths["predictions_sha256"])
            if not rows or len({row["image_id"] for row in rows}) != len(rows):
                raise ValueError(f"OOF fold predictions are empty or duplicated: k={k}, fold={fold}")
            losses: list[float] = []
            for row in rows:
                label = int(row["image_label"])
                logit = _finite_float(row["bag_logit"], name="OOF bag logit")
                probability = _finite_float(row["bag_probability"], name="OOF bag probability")
                loss = _finite_float(row["image_bce"], name="OOF BCE")
                if (
                    label not in (0, 1)
                    or int(row["heldout_fold"]) != fold
                    or row["group_id"] not in heldout_groups
                    or int(row["candidate_count"]) <= 0
                ):
                    raise ValueError(f"OOF identity mismatch: k={k}, fold={fold}")
                _close(probability, _sigmoid(logit), name="OOF probability", atol=1.0e-7)
                _close(loss, _binary_bce(logit, label), name="OOF BCE", atol=1.0e-7)
                losses.append(loss)
            _close(audit["heldout_mean_image_bce"], np.mean(losses), name="fold mean BCE")
            fold_bce.append(float(np.mean(losses)))
            aggregate_rows.extend(rows)
            expected_exclusion.append(
                {
                    "fold": fold,
                    "heldout_groups": len(heldout_groups),
                    "training_groups": len(training_groups),
                    "overlap": 0,
                }
            )

        aggregate_root = root / "oof" / f"k_{k}"
        expected = inventory[f"k_{k}_aggregate"]
        aggregate_paths = {
            "oof_predictions_sha256": aggregate_root / "oof_predictions.csv",
            "oof_summary_sha256": aggregate_root / "oof_summary.json",
        }
        for key, path in aggregate_paths.items():
            if not path.is_file() or sha256_file(path) != expected.get(key):
                raise ValueError(f"OOF aggregate hash mismatch: k={k}, {key}")
            physical_files += 1
            physical_bytes += path.stat().st_size
        rows = _csv(aggregate_paths["oof_predictions_sha256"])
        if len(rows) != expected_train or len({row["image_id"] for row in rows}) != expected_train:
            raise ValueError(f"OOF aggregate cohort mismatch: k={k}")
        folded = {row["image_id"]: row for row in aggregate_rows}
        if len(aggregate_rows) != expected_train or set(folded) != {row["image_id"] for row in rows}:
            raise ValueError(f"OOF fold union differs from aggregate: k={k}")
        for row in rows:
            source = folded[row["image_id"]]
            for field in row:
                if row[field] != source[field]:
                    raise ValueError(f"OOF aggregate row differs from its fold: k={k}, {row['image_id']}")
        identity = {
            row["image_id"]: (row["group_id"], int(row["image_label"]), int(row["heldout_fold"]))
            for row in rows
        }
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError("OOF identity/fold assignment differs across K")
        probabilities = [_finite_float(row["bag_probability"], name="OOF probability") for row in rows]
        counts = [int(row["candidate_count"]) for row in rows]
        losses = [_finite_float(row["image_bce"], name="OOF BCE") for row in rows]
        association = _spearman(counts, probabilities)
        summary = _json(aggregate_paths["oof_summary_sha256"])
        exclusion = summary.get("crossfit_exclusion", {})
        if (
            summary.get("prototype_count") != k
            or summary.get("validation_segmentation_quality_used") is not False
            or exclusion.get("complete") is not True
            or exclusion.get("group_overlap") != 0
            or exclusion.get("folds") != expected_exclusion
        ):
            raise ValueError(f"OOF aggregate safety mismatch: k={k}")
        for actual, expected_loss in zip(summary["fold_image_bce"], fold_bce):
            _close(actual, expected_loss, name="OOF fold BCE")
        _close(summary["mean_oof_image_bce"], np.mean(losses), name="OOF mean BCE")
        _close(summary["count_probability_spearman"], association, name="OOF count Spearman")
        selection_inputs.append(
            {
                "prototype_count": k,
                "fold_image_bce": fold_bce,
                "count_probability_spearman": association,
            }
        )

    recomputed = _recompute_selection(selection_inputs)
    if set(selection) != set(recomputed):
        raise ValueError("Prototype-count selection schema mismatch")
    for key, expected in recomputed.items():
        actual = selection[key]
        if isinstance(expected, float):
            _close(actual, expected, name=f"selection/{key}")
        elif key == "candidates":
            if len(actual) != len(expected):
                raise ValueError("Prototype-count selection candidate count mismatch")
            for actual_row, expected_row in zip(actual, expected):
                for row_key, row_expected in expected_row.items():
                    if isinstance(row_expected, float):
                        _close(actual_row[row_key], row_expected, name=f"selection/{row_key}")
                    elif actual_row[row_key] != row_expected:
                        raise ValueError(f"Prototype-count selection differs: {row_key}")
        elif actual != expected:
            raise ValueError(f"Prototype-count selection differs: {key}")
    return {
        "physical_oof_files_verified": physical_files,
        "physical_oof_bytes_verified": physical_bytes,
        "selection": recomputed,
    }


def _verify_validation_evidence(
    root: Path,
    freeze: Mapping[str, object],
    *,
    expected_validation: int,
    expected_map_shape: tuple[int, int],
) -> dict[str, object]:
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    if sha256_file(prediction_manifest) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("Prediction manifest differs from freeze")
    if sha256_file(score_manifest) != freeze.get("candidate_score_manifest_sha256"):
        raise ValueError("Candidate-score manifest differs from freeze")
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
    ):
        raise ValueError("Validation prediction/score cohort mismatch")
    label_counts = {0: 0, 1: 0}
    count_values: list[int] = []
    probability_values: list[float] = []
    physical_bytes = prediction_manifest.stat().st_size + score_manifest.stat().st_size
    for image_id, prediction in prediction_by_id.items():
        score = score_by_id[image_id]
        for field in ("group_id", "tumor", "candidate_payload_sha256", "candidate_count", "selected_candidate_index", "selected_candidate_logit"):
            if prediction[field] != score[field]:
                raise ValueError(f"Prediction/score provenance differs: {image_id}/{field}")
        label = int(prediction["tumor"])
        if label not in label_counts:
            raise ValueError(f"Invalid image label: {image_id}")
        label_counts[label] += 1
        count = int(prediction["candidate_count"])
        if count <= 0 or prediction["candidate_logit_tta"] != "mean_original_aligned_horizontal_flip":
            raise ValueError(f"Validation candidate contract mismatch: {image_id}")
        score_path = _safe_child(root / "candidate_scores", score["score_path"])
        if not score_path.is_file() or sha256_file(score_path) != score["score_sha256"]:
            raise ValueError(f"Candidate-score payload hash mismatch: {image_id}")
        with np.load(score_path, allow_pickle=False) as payload:
            if set(payload.files) != {"schema_version", "candidate_indices", "candidate_logits"}:
                raise ValueError(f"Candidate-score fields mismatch: {image_id}")
            schema = int(payload["schema_version"])
            indices = payload["candidate_indices"]
            logits = payload["candidate_logits"]
        if (
            schema != 1
            or indices.dtype != np.int64
            or logits.dtype != np.float32
            or indices.ndim != 1
            or logits.shape != indices.shape
            or len(indices) != count
            or np.any(indices < 0)
            or np.any(np.diff(indices) <= 0)
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"Candidate-score content mismatch: {image_id}")
        winner = int(np.argmax(logits))
        if int(indices[winner]) != int(score["selected_candidate_index"]):
            raise ValueError(f"Candidate-score winner index mismatch: {image_id}")
        _close(logits[winner], score["selected_candidate_logit"], name="selected candidate logit")
        bag_logit = _finite_float(prediction["bag_logit"], name="validation bag logit")
        bag_probability = _finite_float(prediction["bag_probability"], name="validation bag probability")
        _close(bag_logit, _smooth_pool(logits), name="validation bag logit", atol=2.0e-6)
        _close(bag_probability, _sigmoid(bag_logit), name="validation bag probability", atol=1.0e-7)
        map_path = _safe_child(root / "predictions", prediction["map_path"])
        if not map_path.is_file() or sha256_file(map_path) != prediction["map_sha256"]:
            raise ValueError(f"Prediction map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != expected_map_shape
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Prediction map content mismatch: {image_id}")
        nonzero = values > 0
        _close(nonzero.mean(), prediction["selected_area_ratio"], name="selected area ratio", atol=1.0 / np.prod(expected_map_shape) + 1.0e-7)
        if nonzero.any():
            positive = values[nonzero].astype(np.float32)
            if not np.allclose(positive, np.float16(bag_probability), atol=0.0, rtol=0.0):
                raise ValueError(f"Prediction map is not selected mask times bag probability: {image_id}")
        count_values.append(count)
        probability_values.append(bag_probability)
        physical_bytes += score_path.stat().st_size + map_path.stat().st_size
    return {
        "physical_validation_maps_verified": expected_validation,
        "physical_candidate_score_payloads_verified": expected_validation,
        "validation_image_label_counts": {"normal": label_counts[0], "tumor": label_counts[1]},
        "validation_absolute_count_probability_spearman": abs(_spearman(count_values, probability_values)),
        "physical_validation_evidence_bytes": physical_bytes,
    }


def audit_r1_output(
    root: Path,
    protocol_path: Path,
    *,
    expected_train: int = 2981,
    expected_validation: int = 371,
    expected_map_shape: tuple[int, int] = (320, 320),
) -> dict[str, object]:
    protocol = _json(protocol_path)
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("R1 protocol SHA-256 mismatch")
    freeze_path = root / "prediction_freeze.json"
    run_manifest_path = root / "run_manifest.json"
    wrapper_audit_path = root / "wrapper_output_audit.json"
    assignment_path = root / "crossfit_assignment.json"
    selection_path = root / "prototype_count_selection.json"
    for path in (freeze_path, run_manifest_path, wrapper_audit_path, assignment_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"R1 output is missing: {path}")
    freeze = _json(freeze_path)
    run_manifest = _json(run_manifest_path)
    wrapper_audit = _json(wrapper_audit_path)
    assignment = _json(assignment_path)
    selection = _json(selection_path)
    for name, payload in (("prediction freeze", freeze), ("run manifest", run_manifest), ("wrapper output audit", wrapper_audit)):
        _require_safety(payload, name=name)
    protocol_hashes = protocol.get("canonical_lf_source_hashes", {})
    expected_runtime_hashes = {
        path: value
        for path, value in protocol_hashes.items()
        if path not in POST_FREEZE_ONLY_SOURCE_PATHS
    }
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze.get("split_sha256") != SPLIT_SHA256
        or freeze.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or freeze.get("baseline_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or freeze.get("validation_predictions") != expected_validation
        or freeze.get("training_labels") != "image_level_only"
    ):
        raise ValueError("R1 prediction-freeze provenance mismatch")
    if (
        run_manifest.get("run_id") != "btxrd_mask_bag_normal_prototype_r1_v1"
        or run_manifest.get("source_commit") != SOURCE_COMMIT
        or run_manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or run_manifest.get("cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or run_manifest.get("prototype_counts") != list(PROTOTYPE_COUNTS)
        or run_manifest.get("objective_config") != EXPECTED_OBJECTIVE_CONFIG
        or run_manifest.get("training_config") != EXPECTED_TRAINING_CONFIG
        or run_manifest.get("validated_cache_records") != {"train": expected_train, "validation": expected_validation}
        or run_manifest.get("crossfit") != assignment
        or run_manifest.get("selection") != selection
        or run_manifest.get("output_hashes") != freeze
    ):
        raise ValueError("R1 run-manifest contract mismatch")
    runtime = run_manifest.get("runtime", {})
    if (
        runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("oof_parallel_workers") != 2
        or runtime.get("oof_jobs_by_device") != [8, 7]
    ):
        raise ValueError("R1 T4x2 runtime contract mismatch")
    if (
        assignment.get("schema_version") != 1
        or assignment.get("rows") != expected_train
        or assignment.get("folds") != 5
        or assignment.get("fold_summary") != EXPECTED_FOLD_SUMMARY
        or len(str(assignment.get("row_payload_sha256", ""))) != 64
    ):
        raise ValueError("R1 cross-fit assignment contract mismatch")
    if (
        wrapper_audit.get("kernel") != KERNEL
        or wrapper_audit.get("checkout_commit") != CHECKOUT_COMMIT
        or wrapper_audit.get("scientific_source_commit") != SOURCE_COMMIT
        or wrapper_audit.get("protocol_sha256") != PROTOCOL_SHA256
        or wrapper_audit.get("source_hashes") != expected_runtime_hashes
        or wrapper_audit.get("physical_prediction_maps_verified") != expected_validation
        or wrapper_audit.get("physical_candidate_score_payloads_verified") != expected_validation
        or wrapper_audit.get("physical_oof_files_verified") != 66
        or wrapper_audit.get("selected_prototype_count") != selection.get("selected_prototype_count")
    ):
        raise ValueError("R1 wrapper-output audit mismatch")
    t4 = wrapper_audit.get("t4x2", {})
    if (
        t4.get("cuda_device_count") != 2
        or len(t4.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4["cuda_device_names"])
        or len(t4.get("real_convolution_checksums", [])) != 2
        or not np.isfinite(np.asarray(t4["real_convolution_checksums"], dtype=np.float64)).all()
    ):
        raise ValueError("R1 wrapper T4x2 evidence mismatch")
    cache = wrapper_audit.get("cache", {})
    if (
        cache.get("selector_cache_freeze_sha256") != CACHE_FREEZE_SHA256
        or cache.get("selector_cache_wrapper_audit_sha256") != CACHE_WRAPPER_AUDIT_SHA256
        or cache.get("physical_cache_records_verified") != 3352
    ):
        raise ValueError("R1 wrapper cache evidence mismatch")
    if sha256_file(assignment_path) != freeze.get("crossfit_assignment_sha256"):
        raise ValueError("Cross-fit assignment differs from freeze")
    if sha256_file(selection_path) != freeze.get("prototype_count_selection_sha256"):
        raise ValueError("Prototype-count selection differs from freeze")
    selected_k = int(selection.get("selected_prototype_count", -1))
    if selected_k not in PROTOTYPE_COUNTS or freeze.get("selected_prototype_count") != selected_k:
        raise ValueError("Selected prototype count differs from freeze")

    final_paths = {
        "final_prototype_sha256": root / "normal_prototypes.npz",
        "final_checkpoint_sha256": root / "normal_prototype_residual.pt",
        "final_training_history_sha256": root / "final_training_history.json",
    }
    physical_bytes = sum(path.stat().st_size for path in (freeze_path, run_manifest_path, wrapper_audit_path, assignment_path, selection_path))
    for key, path in final_paths.items():
        if not path.is_file() or sha256_file(path) != freeze.get(key):
            raise ValueError(f"Final R1 artifact hash mismatch: {key}")
        physical_bytes += path.stat().st_size
    _verify_prototype_payload(final_paths["final_prototype_sha256"], prototype_count=selected_k)
    history = json.loads(final_paths["final_training_history_sha256"].read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 16
        or any(not isinstance(row, dict) for row in history)
        or any(
            not math.isfinite(float(value))
            for row in history
            for value in row.values()
            if isinstance(value, (int, float))
        )
    ):
        raise ValueError("Final R1 training history must contain the frozen 16 epochs")

    oof = _verify_oof(root, freeze, selection, expected_train=expected_train)
    validation = _verify_validation_evidence(
        root,
        freeze,
        expected_validation=expected_validation,
        expected_map_shape=expected_map_shape,
    )
    return {
        "audit_id": "independent_mask_bag_normal_prototype_r1_output_v1",
        "status": "PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "bound_wrapper_sha256": BOUND_WRAPPER_SHA256,
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "wrapper_output_audit_sha256": sha256_file(wrapper_audit_path),
        "crossfit_assignment_sha256": sha256_file(assignment_path),
        "prototype_count_selection_sha256": sha256_file(selection_path),
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "cache_freeze_sha256": CACHE_FREEZE_SHA256,
        "selected_prototype_count": selected_k,
        "oof": oof,
        "validation": validation,
        "physical_output_bytes_verified": physical_bytes + oof["physical_oof_bytes_verified"] + validation["physical_validation_evidence_bytes"],
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_output.exists():
        raise FileExistsError(f"Independent audit output already exists: {args.audit_output}")
    audit = audit_r1_output(args.output_root, args.protocol)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
