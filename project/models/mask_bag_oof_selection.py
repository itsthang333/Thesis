from __future__ import annotations

"""Train-label-only selection rules for finite selector hyperparameters."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def select_prototype_count_one_standard_error(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_absolute_count_association: float,
    maximum_absolute_count_association_increase: float = 0.02,
) -> dict[str, Any]:
    """Select the smallest safe K inside the best OOF BCE standard-error band.

    Each candidate must contain exactly five group-held-out fold BCE values and
    one all-OOF candidate-count versus bag-probability Spearman correlation.
    No segmentation quality is accepted by this API.
    """

    if not candidates:
        raise ValueError("at least one prototype-count candidate is required")
    if (
        not np.isfinite(baseline_absolute_count_association)
        or baseline_absolute_count_association < 0
        or not np.isfinite(maximum_absolute_count_association_increase)
        or maximum_absolute_count_association_increase < 0
    ):
        raise ValueError("count-association controls must be finite and nonnegative")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    maximum_allowed = (
        float(baseline_absolute_count_association)
        + float(maximum_absolute_count_association_increase)
    )
    for candidate in candidates:
        prototype_count = int(candidate["prototype_count"])
        if prototype_count <= 0 or prototype_count in seen:
            raise ValueError("prototype counts must be unique and positive")
        seen.add(prototype_count)
        losses = np.asarray(candidate["fold_image_bce"], dtype=np.float64)
        association = abs(float(candidate["count_probability_spearman"]))
        if (
            losses.shape != (5,)
            or not np.isfinite(losses).all()
            or np.any(losses < 0)
            or not np.isfinite(association)
            or association > 1.0
        ):
            raise ValueError("OOF losses/association have an invalid contract")
        normalized.append(
            {
                "prototype_count": prototype_count,
                "mean_oof_image_bce": float(losses.mean()),
                "standard_error_oof_image_bce": float(
                    losses.std(ddof=1) / np.sqrt(len(losses))
                ),
                "absolute_count_probability_spearman": association,
                "count_guard_pass": association <= maximum_allowed,
            }
        )

    eligible = [row for row in normalized if row["count_guard_pass"]]
    if not eligible:
        raise RuntimeError("all prototype counts increase the frozen count shortcut")
    best = min(
        eligible,
        key=lambda row: (row["mean_oof_image_bce"], row["prototype_count"]),
    )
    one_se_limit = (
        float(best["mean_oof_image_bce"])
        + float(best["standard_error_oof_image_bce"])
    )
    selected = min(
        (
            row
            for row in eligible
            if float(row["mean_oof_image_bce"]) <= one_se_limit
        ),
        key=lambda row: row["prototype_count"],
    )
    return {
        "rule": "minimum_K_within_best_five_fold_OOF_BCE_one_standard_error",
        "selected_prototype_count": int(selected["prototype_count"]),
        "best_mean_prototype_count": int(best["prototype_count"]),
        "one_standard_error_limit": one_se_limit,
        "maximum_allowed_absolute_count_probability_spearman": maximum_allowed,
        "candidates": sorted(normalized, key=lambda row: row["prototype_count"]),
        "validation_segmentation_quality_used": False,
    }


__all__ = ["select_prototype_count_one_standard_error"]
