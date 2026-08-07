from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from project.evaluate_g4_classifier_labels import _binary_metrics
from project.audit_g4_e1_completed import _paired_bootstrap
from project.models.layercam import collapsed_tumor_log_odds
from project.train_classifier import (
    binary_metrics_from_multiclass_confusion,
    classifier_epoch_budget_audit,
)


def test_binary_collapse_uses_normal_as_class_zero() -> None:
    matrix = torch.tensor(
        [
            [8, 1, 1],
            [2, 5, 1],
            [1, 2, 7],
        ],
        dtype=torch.long,
    )
    metrics = binary_metrics_from_multiclass_confusion(matrix)
    # tn=8, fp=2, fn=3, tp=15
    assert metrics["precision"] == 15 / 17
    assert metrics["recall"] == 15 / 18
    assert metrics["specificity"] == 8 / 10


def test_epoch_budget_audit_can_follow_matched_selection_endpoint() -> None:
    records = [
        {"epoch": 1, "val_f1": 0.9, "selection_value": 0.60},
        {"epoch": 2, "val_f1": 0.8, "selection_value": 0.70},
        {"epoch": 3, "val_f1": 0.7, "selection_value": 0.65},
    ]
    audit = classifier_epoch_budget_audit(
        records,
        requested_epochs=3,
        metric_key="selection_value",
        metric_name="binary F1",
    )
    assert audit["metric"] == "binary F1"
    assert audit["best_epoch"] == 2
    assert audit["best_metric_value"] == 0.70


def test_binary_probability_metrics_have_known_perfect_value() -> None:
    metrics = _binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
    )
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision_auprc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["matthews_correlation_coefficient"] == 1.0
    assert metrics["negative_log_likelihood"] > 0.0


def test_multiclass_attribution_target_is_exact_collapsed_binary_log_odds() -> None:
    logits = torch.tensor([[0.3, -0.2, 1.1, 0.7]], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=1)
    expected = torch.log(probabilities[:, 1:].sum(dim=1) / probabilities[:, 0])
    actual = collapsed_tumor_log_odds(logits)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_e1_runner_is_validation_only_and_matched() -> None:
    source = (Path(__file__).parents[1] / "project" / "run_g4_e1_label_granularity.py").read_text(encoding="utf-8")
    assert '"--image-size", "320"' in source
    assert '"--batch-size", "4"' in source
    assert '"--checkpoint-selection-metric", "binary_f1"' in source
    assert '"--seeds", default="42,43,44"' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source
    assert '"--split", "test"' not in source


def test_e1_paired_bootstrap_preserves_matched_group_sampling() -> None:
    binary = [
        {"image_id": "a0", "tumor": "0", "tumor_probability": "0.4"},
        {"image_id": "a1", "tumor": "1", "tumor_probability": "0.6"},
        {"image_id": "b0", "tumor": "0", "tumor_probability": "0.4"},
        {"image_id": "b1", "tumor": "1", "tumor_probability": "0.6"},
    ]
    ten = [
        {"image_id": "a0", "tumor": "0", "tumor_probability": "0.1"},
        {"image_id": "a1", "tumor": "1", "tumor_probability": "0.9"},
        {"image_id": "b0", "tumor": "0", "tumor_probability": "0.1"},
        {"image_id": "b1", "tumor": "1", "tumor_probability": "0.9"},
    ]
    report = _paired_bootstrap(
        binary,
        ten,
        {"a0": "a", "a1": "a", "b0": "b", "b1": "b"},
        iterations=20,
        seed=7,
    )
    assert report["groups"] == 2
    assert report["metrics"]["negative_log_likelihood"]["delta"] < 0
    assert report["metrics"]["brier_score"]["delta"] < 0
