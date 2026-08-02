from __future__ import annotations

"""Deterministic balanced training helpers for the bounded SMILE diagnostic."""

import random
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


SMILE_SEED = 20260802
SMILE_BATCH_SIZE = 2
SMILE_PASSES = 2
SMILE_STEPS_PER_PASS = 1493
SMILE_TERMINAL_STEP = SMILE_PASSES * SMILE_STEPS_PER_PASS
SMILE_LR = 1.0e-4
SMILE_WEIGHT_DECAY = 1.0e-4
SMILE_CONSISTENCY_EVERY = 8
SMILE_REFERENCE_SWAP_WEIGHT = 0.05
SMILE_FLIP_STYLE_WEIGHT = 0.05


class MixedSubtypeBalancedBatchSampler(Sampler[list[int]]):
    """One normal plus one tumor per step; alternate natural/uniform subtype.

    Natural tumor draws preserve cohort prevalence on half the steps.  Uniform
    subtype draws on the other half prevent common osteochondroma from
    monopolizing the local subtype objective.  The rule is deterministic and
    fixed before validation.
    """

    def __init__(
        self,
        tumor: Sequence[int | float],
        subtype: Sequence[int],
        *,
        epoch: int,
        seed: int = SMILE_SEED,
        steps: int = SMILE_STEPS_PER_PASS,
    ) -> None:
        if len(tumor) != len(subtype) or steps <= 0:
            raise ValueError("labels/steps are invalid")
        self.normal = [index for index, label in enumerate(tumor) if int(label) == 0]
        self.tumor = [index for index, label in enumerate(tumor) if int(label) == 1]
        self.by_subtype = {
            value: [index for index in self.tumor if int(subtype[index]) == value]
            for value in range(1, 10)
        }
        if len(self.normal) != 1493 or len(self.tumor) != 1488 or any(
            not values for values in self.by_subtype.values()
        ):
            raise ValueError("canonical train labels/subtypes changed")
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.steps = int(steps)

    def __len__(self) -> int:
        return self.steps

    @staticmethod
    def _cycle(values: list[int], count: int) -> list[int]:
        return [values[index % len(values)] for index in range(count)]

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + 1009 * self.epoch)
        normal = list(self.normal)
        natural = list(self.tumor)
        generator.shuffle(normal)
        generator.shuffle(natural)
        subtype_values = sorted(self.by_subtype)
        subtype_pools = {key: list(values) for key, values in self.by_subtype.items()}
        for values in subtype_pools.values():
            generator.shuffle(values)
        subtype_offsets = {key: 0 for key in subtype_values}
        normal_draws = self._cycle(normal, self.steps)
        natural_draws = self._cycle(natural, self.steps)
        for step in range(self.steps):
            if step % 2 == 0:
                subtype_value = subtype_values[(step // 2) % len(subtype_values)]
                pool = subtype_pools[subtype_value]
                offset = subtype_offsets[subtype_value]
                tumor_index = pool[offset % len(pool)]
                subtype_offsets[subtype_value] += 1
            else:
                tumor_index = natural_draws[step]
            pair = [normal_draws[step], tumor_index]
            if step % 2:
                pair.reverse()
            yield pair


def seed_smile(seed: int = SMILE_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    target = np.asarray(labels, dtype=np.int64)
    value = np.asarray(scores, dtype=np.float64)
    positive = target == 1
    negative = target == 0
    if not positive.any() or not negative.any() or not np.isfinite(value).all():
        raise ValueError("AUROC requires finite two-class inputs")
    order = np.argsort(value, kind="stable")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float(
        (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
        / (positive.sum() * negative.sum())
    )


def label_safe_summary(
    tumor: Sequence[int],
    binary_logits: Sequence[float],
    subtype: Sequence[int],
    subtype_logits: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(tumor, dtype=np.int64)
    binary = np.asarray(binary_logits, dtype=np.float64)
    subtype_target = np.asarray(subtype, dtype=np.int64)
    subtype_score = np.asarray(subtype_logits, dtype=np.float64)
    if labels.shape != binary.shape or subtype_score.shape != (len(labels), 10):
        raise ValueError("label-safe validation arrays differ")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(binary, -40.0, 40.0)))
    prediction = probabilities >= 0.5
    true_positive = int(np.logical_and(prediction, labels == 1).sum())
    false_positive = int(np.logical_and(prediction, labels == 0).sum())
    false_negative = int(np.logical_and(~prediction, labels == 1).sum())
    f1 = 2.0 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
    return {
        "images": int(len(labels)),
        "binary_auroc": binary_auroc(labels, binary),
        "binary_f1_at_0_5": float(f1),
        "subtype_accuracy": float((subtype_score.argmax(axis=1) == subtype_target).mean()),
    }

