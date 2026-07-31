from __future__ import annotations

import csv
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_normal_prototype_r1_output.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r1_output_auditor", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_validation(root: Path):
    module = _load_module()
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index, label in enumerate((0, 1)):
        image_id = f"image_{index}.jpeg"
        logits = np.asarray(
            [-0.2 + index, 0.7 + index] + ([0.1] if index else []),
            dtype=np.float32,
        )
        indices = np.asarray([2, 5] + ([9] if index else []), dtype=np.int64)
        winner = int(np.argmax(logits))
        score_relative = Path("scores") / f"{index}.npz"
        score_path = root / "candidate_scores" / score_relative
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            schema_version=np.asarray(1, dtype=np.int32),
            candidate_indices=indices,
            candidate_logits=logits,
        )
        bag_logit = module._smooth_pool(logits)
        probability = module._sigmoid(bag_logit)
        values = np.zeros((4, 4), dtype=np.float16)
        values[: index + 1, :2] = np.float16(probability)
        map_relative = Path("maps") / f"{index}.npy"
        map_path = root / "predictions" / map_relative
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_path, values, allow_pickle=False)
        common = {
            "image_id": image_id,
            "group_id": f"group_{index}",
            "tumor": label,
            "candidate_payload_sha256": f"{index + 1:064x}",
            "candidate_count": len(indices),
            "selected_candidate_index": int(indices[winner]),
            "selected_candidate_logit": float(logits[winner]),
        }
        score_rows.append(
            {
                **common,
                "score_path": str(score_relative),
                "score_sha256": _sha256(score_path),
            }
        )
        prediction_rows.append(
            {
                **common,
                "candidate_logit_tta": "mean_original_aligned_horizontal_flip",
                "bag_logit": bag_logit,
                "bag_probability": probability,
                "selected_area_ratio": float((values > 0).mean()),
                "fallback_count": 0,
                "map_path": str(map_relative),
                "map_sha256": _sha256(map_path),
            }
        )
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    _write_csv(prediction_manifest, prediction_rows)
    _write_csv(score_manifest, score_rows)
    freeze = {
        "prediction_manifest_sha256": _sha256(prediction_manifest),
        "candidate_score_manifest_sha256": _sha256(score_manifest),
    }
    return module, freeze, prediction_manifest, score_manifest


def _synthetic_oof(root: Path):
    module = _load_module()
    identities = [
        {
            "image_id": f"train_{index}.jpeg",
            "group_id": f"group_{index}",
            "image_label": index % 2,
            "heldout_fold": index // 2,
            "candidate_count": index + 2,
        }
        for index in range(10)
    ]
    logit_ranks = [6, 1, 9, 4, 7, 2, 10, 5, 8, 3]
    inventory: dict[str, dict[str, str]] = {}
    selection_inputs: list[dict[str, object]] = []
    all_groups = {row["group_id"] for row in identities}
    for k in module.PROTOTYPE_COUNTS:
        aggregate_rows: list[dict[str, object]] = []
        fold_bce: list[float] = []
        prototypes = np.zeros((k, module.DESCRIPTOR_DIM), dtype=np.float32)
        prototypes[:, :k] = np.eye(k, dtype=np.float32)
        for fold in module.FOLDS:
            fold_root = root / "oof" / f"k_{k}" / f"fold_{fold}"
            fold_root.mkdir(parents=True)
            prototype_path = fold_root / "normal_prototypes.npz"
            np.savez_compressed(prototype_path, prototypes=prototypes)
            adapter_path = fold_root / "adapter.pt"
            adapter_path.write_bytes(f"adapter-{k}-{fold}".encode())
            heldout = [row for row in identities if row["heldout_fold"] == fold]
            prediction_rows: list[dict[str, object]] = []
            losses: list[float] = []
            for row in heldout:
                index = int(row["image_id"].split("_")[1].split(".")[0])
                logit = (logit_ranks[index] - 5.5) / 10.0 + (k - 16) / 1000.0
                probability = module._sigmoid(logit)
                loss = module._binary_bce(logit, row["image_label"])
                prediction_rows.append(
                    {
                        **row,
                        "bag_logit": logit,
                        "bag_probability": probability,
                        "image_bce": loss,
                    }
                )
                losses.append(loss)
            prediction_path = fold_root / "heldout_predictions.csv"
            _write_csv(prediction_path, prediction_rows)
            heldout_groups = {row["group_id"] for row in heldout}
            audit = {
                "prototype_count": k,
                "heldout_fold": fold,
                "derived_seed": 42 + 1000 * k + fold,
                "training_groups": sorted(all_groups - heldout_groups),
                "heldout_groups": sorted(heldout_groups),
                "group_overlap": 0,
                "heldout_mean_image_bce": float(np.mean(losses)),
                "validation_segmentation_quality_used": False,
            }
            audit_path = fold_root / "fold_audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            inventory[f"k_{k}_fold_{fold}"] = {
                "prototype_sha256": _sha256(prototype_path),
                "adapter_sha256": _sha256(adapter_path),
                "audit_sha256": _sha256(audit_path),
                "predictions_sha256": _sha256(prediction_path),
            }
            aggregate_rows.extend(prediction_rows)
            fold_bce.append(float(np.mean(losses)))
        aggregate_root = root / "oof" / f"k_{k}"
        prediction_path = aggregate_root / "oof_predictions.csv"
        _write_csv(prediction_path, aggregate_rows)
        association = module._spearman(
            [row["candidate_count"] for row in aggregate_rows],
            [row["bag_probability"] for row in aggregate_rows],
        )
        summary = {
            "prototype_count": k,
            "fold_image_bce": fold_bce,
            "mean_oof_image_bce": float(np.mean([row["image_bce"] for row in aggregate_rows])),
            "count_probability_spearman": association,
            "crossfit_exclusion": {
                "complete": True,
                "group_overlap": 0,
                "folds": [
                    {
                        "fold": fold,
                        "heldout_groups": 2,
                        "training_groups": 8,
                        "overlap": 0,
                    }
                    for fold in module.FOLDS
                ],
            },
            "validation_segmentation_quality_used": False,
        }
        summary_path = aggregate_root / "oof_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        inventory[f"k_{k}_aggregate"] = {
            "oof_predictions_sha256": _sha256(prediction_path),
            "oof_summary_sha256": _sha256(summary_path),
        }
        selection_inputs.append(
            {
                "prototype_count": k,
                "fold_image_bce": fold_bce,
                "count_probability_spearman": association,
            }
        )
    return module, {"oof_artifact_hashes": inventory}, module._recompute_selection(selection_inputs)


def test_auditor_surface_is_gt_dataset_evaluator_and_test_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    compile(source, str(SOURCE), "exec")
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "candidate_quality",
        "oracle_candidate",
    ):
        assert forbidden not in lowered
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("evaluate_mask_bag_selector_arm" in name for name in imported)
    assert "validation_gt_read" in source
    assert "training_labels" in source


def test_selection_is_independently_recomputed() -> None:
    module = _load_module()
    selection = module._recompute_selection(
        [
            {"prototype_count": 8, "fold_image_bce": [0.42] * 5, "count_probability_spearman": 0.2},
            {"prototype_count": 16, "fold_image_bce": [0.40] * 5, "count_probability_spearman": 0.21},
            {"prototype_count": 32, "fold_image_bce": [0.35, 0.42, 0.36, 0.43, 0.36], "count_probability_spearman": 0.22},
        ]
    )
    assert selection["best_mean_prototype_count"] == 32
    assert selection["selected_prototype_count"] == 16
    assert selection["validation_segmentation_quality_used"] is False


def test_selection_rejects_count_shortcut_only_candidates() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="count-shortcut"):
        module._recompute_selection(
            [
                {"prototype_count": k, "fold_image_bce": [0.4] * 5, "count_probability_spearman": 0.9}
                for k in module.PROTOTYPE_COUNTS
            ]
        )


def test_constant_spearman_input_is_zero_like_frozen_source() -> None:
    module = _load_module()
    assert module._spearman([2, 2, 2], [0.1, 0.2, 0.3]) == 0.0


def test_validation_evidence_is_physically_recomputed(tmp_path: Path) -> None:
    module, freeze, _prediction_manifest, _score_manifest = _synthetic_validation(tmp_path)
    result = module._verify_validation_evidence(
        tmp_path,
        freeze,
        expected_validation=2,
        expected_map_shape=(4, 4),
    )
    assert result["physical_validation_maps_verified"] == 2
    assert result["physical_candidate_score_payloads_verified"] == 2
    assert result["validation_image_label_counts"] == {"normal": 1, "tumor": 1}
    assert 0.0 <= result["validation_absolute_count_probability_spearman"] <= 1.0


def test_validation_evidence_rejects_manifest_path_escape(tmp_path: Path) -> None:
    module, freeze, _prediction_manifest, score_manifest = _synthetic_validation(tmp_path)
    rows = module._csv(score_manifest)
    rows[0]["score_path"] = "../escape.npz"
    _write_csv(score_manifest, rows)
    freeze["candidate_score_manifest_sha256"] = _sha256(score_manifest)
    with pytest.raises(ValueError, match="escapes"):
        module._verify_validation_evidence(
            tmp_path,
            freeze,
            expected_validation=2,
            expected_map_shape=(4, 4),
        )


def test_prototype_payload_requires_unit_float32_rows(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "prototypes.npz"
    values = np.zeros((8, module.DESCRIPTOR_DIM), dtype=np.float32)
    values[:, 0] = 1.0
    np.savez_compressed(path, prototypes=values)
    module._verify_prototype_payload(path, prototype_count=8)
    values[0, 0] = 2.0
    np.savez_compressed(path, prototypes=values)
    with pytest.raises(ValueError, match="unit-normalized"):
        module._verify_prototype_payload(path, prototype_count=8)


def test_oof_inventory_hashes_exclusion_and_selection_are_recomputed(tmp_path: Path) -> None:
    module, freeze, selection = _synthetic_oof(tmp_path)
    result = module._verify_oof(tmp_path, freeze, selection, expected_train=10)
    assert result["physical_oof_files_verified"] == 66
    assert result["selection"] == selection


def test_oof_rejects_heldout_group_in_training_set(tmp_path: Path) -> None:
    module, freeze, selection = _synthetic_oof(tmp_path)
    audit_path = tmp_path / "oof" / "k_8" / "fold_0" / "fold_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["training_groups"].append(audit["heldout_groups"][0])
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    freeze["oof_artifact_hashes"]["k_8_fold_0"]["audit_sha256"] = _sha256(audit_path)
    with pytest.raises(ValueError, match="exclusion"):
        module._verify_oof(tmp_path, freeze, selection, expected_train=10)
