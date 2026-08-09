from __future__ import annotations

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from summarize_x4_student_study import summarize_study  # noqa: E402
from x4_contract import STUDENT_ARMS, STUDENT_SEEDS  # noqa: E402


def _rows(dice: float) -> list[dict[str, object]]:
    rows = []
    for index in range(371):
        tumor = index < 184
        if index < 94:
            group = "small_lt_1pct"
        elif index < 166:
            group = "medium_1_to_5pct"
        elif index < 184:
            group = "large_ge_5pct"
        else:
            group = "normal"
        rows.append(
            {
                "image_id": f"image-{index:03d}",
                "group_id": f"group-{index // 2:03d}",
                "gt_positive": tumor,
                "predicted_positive": tumor,
                "native_size_group": group,
                "dice": dice if tumor else 1.0,
                "iou": dice / (2.0 - dice) if tumor else 1.0,
                "precision": dice if tumor else 1.0,
                "recall": dice if tumor else 1.0,
                "hd95_px": 1.0 if tumor else 0.0,
                "assd_px": 0.5 if tumor else 0.0,
                "tp_pixels": 1 if tumor else 0,
                "fp_pixels": 0,
                "fn_pixels": 0,
                "tn_pixels": 1,
                "gt_lesions": 1 if tumor else 0,
                "detected_lesions_any_overlap": 1 if tumor else 0,
                "predicted_lesions": 1 if tumor else 0,
                "matched_predicted_lesions_any_overlap": 1 if tumor else 0,
                "lesion_tp_one_to_one_iou10": 1 if tumor else 0,
                "lesion_tp_one_to_one_iou25": 1 if tumor else 0,
                "lesion_tp_one_to_one_iou50": 1 if tumor else 0,
                "pred_area_ratio": 0.0 if not tumor else 0.01,
                "predicted_gt_area_ratio": 1.0 if tumor else float("nan"),
                "relative_area_difference": 0.0 if tumor else float("nan"),
                "empty_prediction": not tumor,
                "zero_overlap": False,
            }
        )
    return rows


def test_x4_study_aggregates_three_seeds_and_five_contrasts() -> None:
    runs = {}
    for arm_index, arm in enumerate(STUDENT_ARMS):
        for seed_index, seed in enumerate(STUDENT_SEEDS):
            runs[(arm, seed)] = _rows(0.2 + arm_index * 0.02 + seed_index * 0.01)
    report = summarize_study(runs, _rows(0.25), iterations=10, seed=7)
    assert set(report["across_seeds"]) == set(STUDENT_ARMS)
    assert len(report["paired_contrasts"]) == 5
    for contrast in report["paired_contrasts"].values():
        assert [row["seed"] for row in contrast["per_seed"]] == list(STUDENT_SEEDS)
        assert [row["bootstrap_seed"] for row in contrast["per_seed"]] == [
            7 + seed for seed in STUDENT_SEEDS
        ]
    rich = report["across_seeds"]["rich_gallery"]["overall"]["mean_tumor_dice"]
    assert rich["n"] == 3
    assert rich["sample_sd"] > 0


def test_x4_study_accepts_frozen_direct_tumor_only_cohort() -> None:
    runs = {
        (arm, seed): _rows(0.2)
        for arm in STUDENT_ARMS
        for seed in STUDENT_SEEDS
    }
    direct_tumor_rows = [row for row in _rows(0.25) if row["gt_positive"]]
    report = summarize_study(runs, direct_tumor_rows, iterations=10, seed=7)
    contrast = report["paired_contrasts"][
        "rich_gallery_student_vs_direct_rich_gallery"
    ]
    assert len(contrast["per_seed"]) == 3
