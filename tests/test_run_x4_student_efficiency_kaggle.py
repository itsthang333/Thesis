from __future__ import annotations

import pytest

from run_x4_student_efficiency_kaggle import SPLIT_SHA256, validate_efficiency_freeze


def valid_freeze() -> dict[str, object]:
    return {
        "stage": "x4_student_prediction_freeze_v1",
        "arm": "cam",
        "seed": 42,
        "split": "val",
        "split_sha256": SPLIT_SHA256,
        "checkpoint_sha256": "a" * 64,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "predictions_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_used": False,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "x12_efficiency": {
            "stage": "matched_student_online_inference_and_freeze",
            "timed_images": 371,
            "warmup_iterations": 3,
            "offline_pseudo_label_generation_included": False,
            "device_memory": [{"device_name": "Tesla T4", "peak_allocated_bytes": 10}],
        },
    }


def test_validate_efficiency_freeze_accepts_exact_t4_contract():
    result = validate_efficiency_freeze(
        valid_freeze(), arm="cam", seed=42, checkpoint_sha256="a" * 64
    )
    assert result["timed_images"] == 371


@pytest.mark.parametrize("mutation", ["test", "missing_efficiency", "wrong_gpu"])
def test_validate_efficiency_freeze_rejects_invalid_evidence(mutation: str):
    freeze = valid_freeze()
    if mutation == "test":
        freeze["test_images_read"] = 1
    elif mutation == "missing_efficiency":
        freeze.pop("x12_efficiency")
    else:
        freeze["x12_efficiency"]["device_memory"][0]["device_name"] = "CPU"
    with pytest.raises(RuntimeError):
        validate_efficiency_freeze(
            freeze, arm="cam", seed=42, checkpoint_sha256="a" * 64
        )
