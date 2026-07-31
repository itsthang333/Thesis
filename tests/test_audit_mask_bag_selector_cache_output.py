from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_selector_cache_output.py"


def _load_module():
    project = str(ROOT / "project")
    if project not in __import__("sys").path:
        __import__("sys").path.insert(0, project)
    spec = importlib.util.spec_from_file_location("selector_cache_output_audit", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_auditor_surface_is_gt_test_and_training_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "test_loader",
        "optimizer",
        ".backward(",
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": false' in lowered
    assert '"consumer_trained": false' in lowered
    assert '"test_evaluated": false' in lowered


def test_family_audit_is_ordered_and_collapses_fallback() -> None:
    module = _load_module()
    payload = {
        "component_ids": np.asarray([2, 1, 9, 3]),
        "prompt_modes": np.asarray(["box", "point", "other", "box"]),
        "proposal_source_ids": np.asarray(["cam", "cam", "teacher", "cam"]),
        "fallback_flags": np.asarray([0, 0, 1, 1], dtype=np.uint8),
    }
    families = module._family_ids(payload)
    assert families.dtype == np.int32
    assert families.tolist() == [0, 1, 2, 2]


def test_independent_shape_and_geometry_matches_manual_masks() -> None:
    module = _load_module()
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, 1:3, 1:4] = True
    shape, iou, containment, distance = module._shape_and_geometry(masks)
    assert shape.shape == (2, 4)
    assert shape[:, 0].tolist() == pytest.approx([0.25, 0.375])
    assert iou[0, 1] == pytest.approx(1.0 / 9.0)
    assert containment[0, 1] == pytest.approx(0.25)
    assert np.array_equal(iou, iou.T)
    assert np.array_equal(containment, containment.T)
    assert np.array_equal(distance, distance.T)
    assert np.allclose(np.diag(iou), 1.0)
    assert np.allclose(np.diag(containment), 1.0)
    assert np.allclose(np.diag(distance), 0.0)


def test_split_reader_accepts_canonical_lf_only_via_exact_reconstruction(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source = ROOT / "artifacts" / "kaggle" / "wsl_source_consensus_val_v1" / "frozen_split_manifest.csv"
    rows = module._read_split(
        source,
        expected_frozen_sha256=module.FROZEN_SPLIT_SHA256,
        expected_counts={"train": 2981, "val": 371},
    )
    assert len(rows["train"]) == 2981
    assert len(rows["val"]) == 371

    corrupted = tmp_path / "split.csv"
    corrupted.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._read_split(
            corrupted,
            expected_frozen_sha256=module.FROZEN_SPLIT_SHA256,
            expected_counts={"train": 2981, "val": 371},
        )


def test_complete_synthetic_cache_audit_contract(tmp_path: Path) -> None:
    module = _load_module()
    from models.mask_bag_selector_cache import pack_candidate_masks
    from models.mask_bag_selector_cache_io import (
        save_selector_cache_record,
        write_selector_cache_manifest,
    )

    split_path = tmp_path / "split.csv"
    buffer = io.StringIO(newline="")
    split_writer = csv.DictWriter(
        buffer,
        fieldnames=["image_id", "group_id", "split", "eligible", "tumor"],
        lineterminator="\r\n",
    )
    split_writer.writeheader()
    split_writer.writerows(
        [
            {
                "image_id": "train.jpeg",
                "group_id": "group-train",
                "split": "train",
                "eligible": "1",
                "tumor": "1",
            },
            {
                "image_id": "val.jpeg",
                "group_id": "group-val",
                "split": "val",
                "eligible": "1",
                "tumor": "0",
            },
        ]
    )
    split_path.write_bytes(buffer.getvalue().encode("utf-8"))
    split_sha = _sha256(split_path)

    candidate_hashes = {"train": "1" * 64, "val": "2" * 64}
    candidate_manifests: dict[str, Path] = {}
    for split in ("train", "val"):
        path = tmp_path / f"{split}_candidates.csv"
        _write_csv(
            path,
            [
                {
                    "image_name": f"{split}.jpeg",
                    "candidate_count": 2,
                    "diagnostic_sha256": candidate_hashes[split],
                }
            ],
        )
        candidate_manifests[split] = path

    baseline_root = tmp_path / "baseline"
    baseline_predictions = baseline_root / "predictions"
    (baseline_predictions / "maps").mkdir(parents=True)
    baseline_map = baseline_predictions / "maps" / "val.npy"
    np.save(baseline_map, np.ones((4, 4), dtype=np.float16), allow_pickle=False)
    map_sha = _sha256(baseline_map)
    prediction_row = {
        "image_id": "val.jpeg",
        "group_id": "group-val",
        "tumor": "0",
        "candidate_payload_sha256": candidate_hashes["val"],
        "candidate_count": "2",
        "selected_candidate_index": "1",
        "candidate_logit_tta": "0.25",
        "fallback_count": "0",
        "map_path": "maps/val.npy",
        "map_sha256": map_sha,
        "selected_candidate_logit": "0.25",
        "bag_logit": "0.5",
        "bag_probability": "0.6224593312",
    }
    baseline_manifest = baseline_predictions / "prediction_manifest.csv"
    _write_csv(baseline_manifest, [prediction_row])
    checkpoint_path = baseline_root / "rad_dino_mask_bag_mil.pt"
    checkpoint_path.write_bytes(b"synthetic frozen checkpoint")
    pseudo_hashes = {"train": "3" * 64, "val": "4" * 64}
    baseline_freeze = {
        "source_commit": module.BASELINE_SOURCE_COMMIT,
        "protocol_sha256": module.BASELINE_PROTOCOL_SHA256,
        "split_sha256": split_sha,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_manifest_sha256": _sha256(baseline_manifest),
        "train_candidate_manifest_sha256": _sha256(candidate_manifests["train"]),
        "train_pseudo_manifest_sha256": pseudo_hashes["train"],
        "val_candidate_manifest_sha256": _sha256(candidate_manifests["val"]),
        "val_pseudo_manifest_sha256": pseudo_hashes["val"],
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    baseline_freeze_path = baseline_root / "prediction_freeze.json"
    baseline_freeze_path.write_text(
        json.dumps(baseline_freeze, sort_keys=True) + "\n", encoding="utf-8"
    )

    cache_root = tmp_path / "cache"
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, :2, :2] = 1
    masks[1, 1:4, 1:4] = 1
    shape, iou, containment, distance = module._shape_and_geometry(masks.astype(bool))
    cache_rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        relative = Path("records") / split / f"{split}.npz"
        saved = save_selector_cache_record(
            cache_root / relative,
            descriptors=np.ones((2, 128), dtype=np.float16),
            flipped_descriptors=np.ones((2, 128), dtype=np.float16),
            affinity_features=np.ones((2, 24), dtype=np.float16),
            flipped_affinity_features=np.ones((2, 24), dtype=np.float16),
            candidate_indices=np.asarray([0, 1], dtype=np.int32),
            family_ids=np.asarray([0, 1], dtype=np.int32),
            component_ids=np.asarray([0, 1], dtype=np.int32),
            prompt_modes=np.asarray(["box", "point"]),
            proposal_source_ids=np.asarray(["cam", "cam"]),
            fallback_flags=np.asarray([0, 0], dtype=np.uint8),
            shape_features=shape,
            pairwise_iou=iou,
            pairwise_containment=containment,
            pairwise_distance=distance,
            packed_masks=pack_candidate_masks(masks) if split == "val" else None,
        )
        cache_rows.append(
            {
                "image_id": f"{split}.jpeg",
                "group_id": f"group-{split}",
                "tumor": int(split == "train"),
                "split": split,
                "candidate_payload_sha256": candidate_hashes[split],
                **saved,
                "cache_path": str(relative),
            }
        )
    cache_summary = write_selector_cache_manifest(cache_root, cache_rows)

    reproduced = cache_root / "baseline_reproduction" / "predictions"
    (reproduced / "maps").mkdir(parents=True)
    shutil.copyfile(baseline_map, reproduced / "maps" / "val.npy")
    shutil.copyfile(baseline_manifest, reproduced / "prediction_manifest.csv")
    reproduction = module._verify_reproduction(
        cache_root, baseline_root, expected_images=1, tolerance=5.0e-6
    )
    reproduction_path = cache_root / "baseline_reproduction_audit.json"
    reproduction_path.write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_commit = "5" * 40
    protocol_sha = "6" * 64
    projection_sha = "7" * 64
    model_hashes = {name: str(index) * 64 for index, name in enumerate(module.MODEL_HASHES, 7)}
    freeze = {
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": split_sha,
        "model_snapshot": {
            name: {"sha256": value} for name, value in model_hashes.items()
        },
        "projection_sha256": projection_sha,
        "baseline_source_commit": module.BASELINE_SOURCE_COMMIT,
        "baseline_protocol_sha256": module.BASELINE_PROTOCOL_SHA256,
        "baseline_prediction_freeze_sha256": _sha256(baseline_freeze_path),
        "baseline_checkpoint_sha256": _sha256(checkpoint_path),
        "baseline_prediction_manifest_sha256": _sha256(baseline_manifest),
        "train_candidate_manifest_sha256": _sha256(candidate_manifests["train"]),
        "train_pseudo_manifest_sha256": pseudo_hashes["train"],
        "val_candidate_manifest_sha256": _sha256(candidate_manifests["val"]),
        "val_pseudo_manifest_sha256": pseudo_hashes["val"],
        "selector_cache_manifest_sha256": cache_summary["manifest_sha256"],
        "baseline_reproduction_audit_sha256": _sha256(reproduction_path),
        "cohort": {"train": 1, "validation": 1},
        "validation_selected_indices_reproduced": 1,
        "validation_map_hashes_reproduced": 1,
        "train_masks_discarded": True,
        "validation_masks_bitpacked": True,
        "affinity_features_cached": True,
        "affinity_feature_dim": 24,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = cache_root / "selector_cache_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_sha = _sha256(freeze_path)
    run_manifest = {
        "cache_freeze_sha256": freeze_sha,
        "cache": cache_summary,
        "baseline_reproduction": reproduction,
        "runtime": {
            "cuda_device_count": 2,
            "cuda_device_names": ["Tesla T4", "Tesla T4"],
            "encoder_data_parallel": True,
        },
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    run_path = cache_root / "run_manifest.json"
    run_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    wrapper = {
        "scientific_source_commit": source_commit,
        "protocol_sha256": protocol_sha,
        "selector_cache_freeze_sha256": freeze_sha,
        "selector_cache_manifest_sha256": cache_summary["manifest_sha256"],
        "baseline_reproduction_audit_sha256": _sha256(reproduction_path),
        "run_manifest_sha256": _sha256(run_path),
        "physical_cache_records_verified": 2,
        "cohort": {"train": 1, "val": 1},
        "t4x2": {
            "cuda_device_count": 2,
            "cuda_device_names": ["Tesla T4", "Tesla T4"],
            "real_convolution_checksums": [1.0, -1.0],
        },
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    wrapper_path = cache_root / "wrapper_output_audit.json"
    wrapper_path.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    audit = module.audit_cache_output(
        cache_root=cache_root,
        expected_cache_freeze_sha256=freeze_sha,
        expected_wrapper_audit_sha256=_sha256(wrapper_path),
        expected_source_commit=source_commit,
        expected_protocol_sha256=protocol_sha,
        split_manifest=split_path,
        expected_split_sha256=split_sha,
        train_candidate_manifest=candidate_manifests["train"],
        expected_train_candidate_manifest_sha256=_sha256(candidate_manifests["train"]),
        val_candidate_manifest=candidate_manifests["val"],
        expected_val_candidate_manifest_sha256=_sha256(candidate_manifests["val"]),
        expected_train_pseudo_manifest_sha256=pseudo_hashes["train"],
        expected_val_pseudo_manifest_sha256=pseudo_hashes["val"],
        baseline_root=baseline_root,
        expected_baseline_freeze_sha256=_sha256(baseline_freeze_path),
        expected_baseline_checkpoint_sha256=_sha256(checkpoint_path),
        expected_baseline_manifest_sha256=_sha256(baseline_manifest),
        expected_counts={"train": 1, "val": 1},
        expected_model_hashes=model_hashes,
        expected_projection_sha256=projection_sha,
    )
    assert audit["physical_cache_records_verified"] == 2
    assert audit["validation_packed_mask_geometry_records_verified"] == 1
    assert audit["baseline_reproduction"]["map_hashes_exact"] == 1
