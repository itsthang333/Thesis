from __future__ import annotations

from collections import Counter

import numpy as np

from project.smile_training import (
    MixedSubtypeBalancedBatchSampler,
    SMILE_STEPS_PER_PASS,
    binary_auroc,
    label_safe_summary,
)


def _canonical_labels() -> tuple[list[int], list[int]]:
    tumor = [0] * 1493
    subtype = [0] * 1493
    counts = [700, 250, 180, 120, 80, 60, 45, 30, 23]
    for value, count in enumerate(counts, start=1):
        tumor.extend([1] * count)
        subtype.extend([value] * count)
    assert len(tumor) == 2981
    return tumor, subtype


def test_sampler_is_deterministic_balanced_and_subtype_aware() -> None:
    tumor, subtype = _canonical_labels()
    first = list(MixedSubtypeBalancedBatchSampler(tumor, subtype, epoch=0))
    second = list(MixedSubtypeBalancedBatchSampler(tumor, subtype, epoch=0))
    changed = list(MixedSubtypeBalancedBatchSampler(tumor, subtype, epoch=1))
    assert first == second and first != changed
    assert len(first) == SMILE_STEPS_PER_PASS
    selected_subtypes = []
    for batch in first:
        assert len(batch) == 2
        assert sorted(tumor[index] for index in batch) == [0, 1]
        selected_subtypes.append(next(subtype[index] for index in batch if tumor[index]))
    counts = Counter(selected_subtypes)
    assert set(counts) == set(range(1, 10))
    assert min(counts.values()) >= 70  # rare types receive material exposure


def test_label_safe_metrics() -> None:
    labels = [0, 0, 1, 1]
    logits = [-2.0, -1.0, 1.0, 2.0]
    subtype = [0, 0, 2, 3]
    subtype_logits = np.eye(10)[subtype] * 4.0
    summary = label_safe_summary(labels, logits, subtype, subtype_logits)
    assert summary["binary_auroc"] == 1.0
    assert summary["binary_f1_at_0_5"] == 1.0
    assert summary["subtype_accuracy"] == 1.0
    assert binary_auroc(labels, logits) == 1.0

