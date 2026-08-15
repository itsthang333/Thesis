from __future__ import annotations

import torch

from project.train_classifier import binary_counts, summarize


def test_binary_counts_use_fixed_probability_threshold() -> None:
    logits = torch.tensor([[3.0], [-3.0], [3.0], [-3.0]])
    targets = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    assert binary_counts(logits, targets) == (1, 1, 1, 1)


def test_binary_summary_is_finite() -> None:
    result = summarize(2.0, 4, (1, 1, 1, 1))
    assert result["loss"] == 0.5
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
