from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"


def test_fully_protocol_is_locked_and_comparison_only() -> None:
    config = json.loads(
        (ROOT / "artifacts" / "final_pipeline" / "final_run_config.json").read_text(
            encoding="utf-8"
        )
    )
    fully = config["fully_supervised_comparison"]
    assert fully["architecture"] == "ResNet18UNet"
    assert fully["image_size"] == 448
    assert fully["final_threshold"] == 0.2
    assert fully["role"].startswith("comparison-only")
    assert fully["pretrained_encoder_weight_sha256"] == (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )


def test_test_lock_requires_supervised_checkpoint() -> None:
    source = (PROJECT / "freeze_final_test_protocol.py").read_text(encoding="utf-8")
    assert '"--supervised-unet-checkpoint"' in source
    assert '"supervised_unet_checkpoint": artifact(args.supervised_unet_checkpoint)' in source
    assert '"joint_test_evaluation"' in source


def test_fully_freeze_is_annotation_free_and_fail_closed() -> None:
    source = (PROJECT / "freeze_fully_supervised_predictions.py").read_text(encoding="utf-8")
    assert "load_split_rows_without_annotations" in source
    assert "build_segmentation_dataset" not in source
    assert 'checkpoint.get("supervision_mode") != "fully_supervised_comparison"' in source
    assert 'checkpoint.get("wsss_eligible") is not False' in source
    assert '"predictions_frozen_before_spatial_gt": True' in source
    assert '"spatial_ground_truth_used": False' in source
    assert '"test_evaluated": False' in source


def test_joint_evaluator_opens_annotation_after_both_freezes() -> None:
    source = (PROJECT / "evaluate_final_rich_gallery.py").read_text(encoding="utf-8")
    boundary = source.index("# Annotation boundary")
    fully_manifest = source.index("fully-supervised prediction manifest changed")
    annotation_decode = source.index("_decode_labelme_polygon_mask(", boundary)
    assert fully_manifest < boundary < annotation_decode
    assert '"joint_annotation_pass": fully_summary is not None' in source
    assert 'comparison_path = args.output_dir / "comparison.csv"' in source


def test_joint_metric_math() -> None:
    import sys

    sys.path.insert(0, str(PROJECT))
    from evaluate_final_rich_gallery import _dice, _iou

    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    assert _dice(prediction, target) == 0.5
    assert _iou(prediction, target) == 1.0 / 3.0


def test_fully_trainer_cannot_become_wsss() -> None:
    source = (PROJECT / "train_segmentation.py").read_text(encoding="utf-8")
    assert 'choices=("fully_supervised_comparison",)' in source
    assert '"comparison_only": True' in source
    assert '"wsss_eligible": False' in source
    assert '"validation_ground_truth_checkpoint_selection": True' in source
    assert 'if args.train_split != "train" or args.val_split != "val"' in source
