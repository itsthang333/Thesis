from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from project.run_x4_yolo_evaluation_kaggle import safe_extract, validate_training_receipt


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
