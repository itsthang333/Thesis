from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

import project.audit_rad_dino_mask_bag_mil_v6_compact_output as audit_module
from project.audit_rad_dino_mask_bag_mil_v6_compact_output import (
    _require,
    audit_compact_output,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_rad_dino_mask_bag_mil_v6_compact_output.py"


def test_compact_auditor_is_dataset_and_annotation_free() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "torch" not in imported
    assert "numpy" not in imported
    assert "build_segmentation_dataset" not in text
    assert "/Annotations/" not in text
    assert "206 passed, 1 skipped" in text
    assert "candidate_payloads_verified_before_training" in text
    assert "physical_prediction_map_hashes_verified" in text
    assert "selected_operational_goal_checks" in text
    assert "oracle_operational_goal_checks" in text
    assert '"consumer_trained"' in text
    assert '"test_evaluated"' in text


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"mask-bag-v6\n")
    assert (
        sha256_file(path)
        == "e4ab015c4a41136f15d1a673de0a450be60c9b7d0d6f9a0a912387638d640314"
    )


def test_require_fails_closed() -> None:
    _require(True, "unused")
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _require(False, "contract mismatch")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_complete_synthetic_compact_contract_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "compact"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    wrapper = provenance / "wrapper.py"
    wrapper.write_text("# frozen wrapper\n", encoding="utf-8")
    protocol = provenance / "protocol.json"
    _write_json(protocol, {"protocol": "v1"})
    correction_hashes: dict[str, str] = {}
    for index in range(1, 6):
        name = f"correction_{index}.json"
        path = provenance / name
        _write_json(path, {"correction": index})
        correction_hashes[name] = sha256_file(path)
    monkeypatch.setattr(audit_module, "WRAPPER_SHA256", sha256_file(wrapper))
    monkeypatch.setattr(audit_module, "PROTOCOL_SHA256", sha256_file(protocol))
    monkeypatch.setattr(audit_module, "CORRECTION_HASHES", correction_hashes)
    monkeypatch.setattr(audit_module, "CHECKOUT_COMMIT", "checkout")
    monkeypatch.setattr(audit_module, "SCIENTIFIC_SOURCE_COMMIT", "scientific")
    monkeypatch.setattr(audit_module, "SPLIT_SHA256", "split")
    monkeypatch.setattr(audit_module, "CLASSIFIER_SHA256", "classifier")
    monkeypatch.setattr(audit_module, "SAM_SHA256", "sam")
    monkeypatch.setattr(audit_module, "BASELINE_SHA256", "baseline")
    monkeypatch.setattr(
        audit_module,
        "RAD_DINO_HASHES",
        {"config.json": "config", "model.safetensors": "weights"},
    )

    candidate_audit: dict[str, object] = {
        "candidate_payloads_verified_before_training": 3352,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    for split, prefix, count in (("train", "train", 2981), ("validation", "val", 371)):
        candidate_manifest = (
            root / "candidate_evidence" / f"{prefix}_candidate_diagnostics_manifest.csv"
        )
        pseudo_manifest = (
            root / "candidate_evidence" / f"{prefix}_pseudo_mask_manifest.csv"
        )
        run_metadata = root / "candidate_evidence" / f"{prefix}_run_metadata.json"
        _write_csv(
            candidate_manifest,
            ["image_name"],
            [{"image_name": f"{split}_{index:04d}.png"} for index in range(count)],
        )
        _write_csv(
            pseudo_manifest,
            ["image_name"],
            [{"image_name": f"{split}_{index:04d}.png"} for index in range(count)],
        )
        _write_json(run_metadata, {"split": split})
        candidate_audit[split] = {
            "split": split,
            "images": count,
            "candidate_manifest_sha256": sha256_file(candidate_manifest),
            "pseudo_manifest_sha256": sha256_file(pseudo_manifest),
            "run_metadata_sha256": sha256_file(run_metadata),
            "physical_payload_hashes_verified": count,
            "maximum_candidates": 81,
            "empty_candidate_bags": 0,
            "normal_pseudo_masks_nonempty": 0,
            "ground_truth_loaded_during_generation": False,
        }
    candidate_audit_path = root / "candidate_input_audit.json"
    _write_json(candidate_audit_path, candidate_audit)

    probe = root / "probe"
    predictions = probe / "predictions"
    predictions.mkdir(parents=True)
    prediction_rows = []
    for index in range(371):
        map_path = predictions / f"map_{index:04d}.bin"
        map_path.write_bytes(f"map-{index}\n".encode())
        prediction_rows.append(
            {
                "image_id": f"val_{index:04d}.png",
                "map_path": map_path.name,
                "map_sha256": sha256_file(map_path),
            }
        )
    prediction_manifest = predictions / "prediction_manifest.csv"
    _write_csv(
        prediction_manifest,
        ["image_id", "map_path", "map_sha256"],
        prediction_rows,
    )
    checkpoint = probe / "rad_dino_mask_bag_mil.pt"
    checkpoint.write_bytes(b"checkpoint")
    history = probe / "training_history.json"
    _write_json(history, {"epochs": 16})
    freeze = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_history_sha256": sha256_file(history),
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
        "validation_predictions": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = probe / "prediction_freeze.json"
    _write_json(freeze_path, freeze)
    _write_json(
        probe / "run_manifest.json",
        {
            "source_commit": "scientific",
            "protocol_sha256": sha256_file(protocol),
            "split_sha256": "split",
            "runtime": {
                "cuda_device_count": 2,
                "cuda_device_names": ["Tesla T4", "Tesla T4"],
                "encoder_data_parallel": True,
            },
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )

    evaluation = root / "evaluation"
    per_image_rows = [
        {
            "image_id": f"tumor_{index:03d}.png",
            "size_group": (
                "small" if index < 94 else "medium" if index < 166 else "large"
            ),
        }
        for index in range(184)
    ]
    per_image_path = evaluation / "per_image.csv"
    _write_csv(per_image_path, ["image_id", "size_group"], per_image_rows)
    summary = {
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
        "tumor_localization": {
            subgroup: {"dice": value, "complete_misses": 0}
            for subgroup, value in {
                "overall": 0.35,
                "small": 0.18,
                "medium": 0.52,
                "large": 0.50,
            }.items()
        },
        "candidate_oracle": {
            "overall": 0.40,
            "small": 0.22,
            "medium": 0.59,
            "large": 0.64,
        },
    }
    summary_path = evaluation / "summary.json"
    _write_json(summary_path, summary)
    paired_path = evaluation / "paired_comparison.json"
    _write_json(paired_path, {"replicates": 10000})
    gate_path = evaluation / "gate_decision.json"
    _write_json(
        gate_path,
        {
            "status": "PASS",
            "all_checks_required": True,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    evaluation_audit = {
        "cohort": audit_module.EXPECTED_COHORT,
        "bootstrap_replicates": 10000,
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "paired_comparison_sha256": sha256_file(paired_path),
        "consumer_trained": False,
        "test_evaluated": False,
    }
    evaluation_audit_path = evaluation / "evaluation_audit.json"
    _write_json(evaluation_audit_path, evaluation_audit)

    gpu = {
        "torch": "2.5.1+cu121",
        "cuda": "12.1",
        "device_count": 2,
        "device_names": ["Tesla T4", "Tesla T4"],
        "real_convolution_sums": [324.0, 324.0],
    }
    independent = {
        "protocol_sha256": sha256_file(protocol),
        **{
            f"wrapper_correction_v{index}_sha256": value
            for index, value in enumerate(correction_hashes.values(), start=1)
        },
        "checkout_commit": "checkout",
        "scientific_source_commit": "scientific",
        "split_sha256": "split",
        "split_counts": audit_module.EXPECTED_SPLIT_COUNTS,
        "classifier_sha256": "classifier",
        "sam_sha256": "sam",
        "baseline_per_image_sha256": "baseline",
        "rad_dino_hashes": audit_module.RAD_DINO_HASHES,
        "gpu": gpu,
        "candidate_input_audit_sha256": sha256_file(candidate_audit_path),
        "candidate_inputs": candidate_audit,
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
        "physical_prediction_map_hashes_verified": 371,
        "evaluation_audit_sha256": sha256_file(evaluation_audit_path),
        "gate_decision_sha256": sha256_file(gate_path),
        "gate_status": "PASS",
        "validation_gt_read_only_after_prediction_freeze": True,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    independent_path = root / "independent_audit.json"
    _write_json(independent_path, independent)
    _write_json(
        root / "completed.json",
        {
            "status": "COMPLETE",
            "independent_audit_sha256": sha256_file(independent_path),
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    (root / "execution.log").write_text(
        "206 passed, 1 skipped\n",
        encoding="utf-8",
    )

    result = audit_compact_output(root)
    assert result["status"] == "PASS"
    assert result["prediction_maps_verified"] == 371
    assert all(result["selected_operational_goal_checks"].values())
    assert all(result["oracle_operational_goal_checks"].values())

    (predictions / "map_0000.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Prediction map hash mismatch"):
        audit_compact_output(root)
