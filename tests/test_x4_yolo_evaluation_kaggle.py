from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from project.run_x4_yolo_evaluation_kaggle import safe_extract, validate_training_receipt
from project.run_x4_yolo_evaluation_kernel import validate_contract
import project.freeze_x4_yolo_predictions as freeze_x4_yolo_predictions


def valid_receipt() -> dict[str, object]:
    return {
        "stage": "x4_yolov8s_seg_kaggle_wrapper_v1",
        "seed": 42,
        "split_sha256": "a" * 64,
        "training_archive_sha256": "b" * 64,
        "best_checkpoint_sha256": "c" * 64,
        "training_report_sha256": "d" * 64,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def test_training_receipt_contract_accepts_exact_values() -> None:
    validate_training_receipt(
        valid_receipt(), seed=42, split_sha256="a" * 64, archive_sha256="b" * 64
    )


@pytest.mark.parametrize("field,value", [("seed", 43), ("test_images_read", 1), ("test_evaluated", True)])
def test_training_receipt_contract_rejects_tamper(field: str, value: object) -> None:
    receipt = valid_receipt()
    receipt[field] = value
    with pytest.raises(RuntimeError):
        validate_training_receipt(
            receipt, seed=42, split_sha256="a" * 64, archive_sha256="b" * 64
        )


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(RuntimeError):
        safe_extract(archive, tmp_path / "out")


def test_kernel_contract_rejects_test_access() -> None:
    contract: dict[str, object] = {
        "schema_version": 1,
        "stage": "x4_yolo_kaggle_evaluation_contract_v1",
        "seed": 42,
        "runtime_manifest_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "training_bundle_name": "training.zip",
        "training_bundle_sha256": "c" * 64,
        "training_receipt_name": "receipt.json",
        "training_receipt_sha256": "d" * 64,
        "runner_sha256": "e" * 64,
        "freeze_runner_sha256": "f" * 64,
        "evaluator_sha256": "0" * 64,
        "output_prefix": "x4_yolo_seed42",
        "test_images_read": 0,
        "test_evaluated": False,
    }
    validate_contract(contract)
    contract["test_images_read"] = 1
    with pytest.raises(RuntimeError):
        validate_contract(contract)


def test_prediction_runner_binds_ultralytics_output_to_writable_stage_a_dir() -> None:
    source = Path(freeze_x4_yolo_predictions.__file__).read_text(encoding="utf-8")
    assert 'project=str(args.output_dir / "_ultralytics")' in source
    assert 'name="predict"' in source
    assert "exist_ok=True" in source
