from __future__ import annotations

from analyze_x4_error_taxonomy import classify_failure, primary_failure


def test_candidate_mechanism_and_extent_flags_can_overlap():
    row = {
        "gt_positive": True,
        "predicted_positive": True,
        "dice": 0.0,
        "zero_overlap": True,
        "predicted_gt_area_ratio": 3.0,
        "gt_lesions": 1,
        "detected_lesions_any_overlap": 0,
        "predicted_lesions": 2,
        "native_size_group": "small_lt_1pct",
    }
    context = {
        "full_gallery_oracle_dice_common320": 0.8,
        "selector_regret_common320": 0.7,
    }
    flags = classify_failure(row, candidate_context=context)
    assert flags["selector_choice_failure"]
    assert flags["complete_miss"]
    assert flags["wrong_anatomical_structure"]
    assert flags["over_segmentation"]
    assert flags["fragmented_mask"]
    assert flags["small_lesion_specific_failure"]
    assert primary_failure(flags) == "selector_choice_failure"


def test_candidate_supply_failure_precedes_phenotype():
    row = {
        "gt_positive": True,
        "predicted_positive": False,
        "dice": 0.0,
        "zero_overlap": True,
        "predicted_gt_area_ratio": 0.0,
        "gt_lesions": 1,
        "detected_lesions_any_overlap": 0,
        "predicted_lesions": 0,
        "native_size_group": "medium_1_to_5pct",
    }
    flags = classify_failure(
        row,
        candidate_context={
            "full_gallery_oracle_dice_common320": 0.05,
            "selector_regret_common320": 0.05,
        },
    )
    assert flags["candidate_supply_failure"]
    assert flags["under_segmentation"]
    assert primary_failure(flags) == "candidate_supply_failure"


def test_normal_false_positive_is_separate_from_tumor_errors():
    flags = classify_failure(
        {
            "gt_positive": False,
            "predicted_positive": True,
            "dice": 0.0,
            "predicted_gt_area_ratio": "nan",
            "gt_lesions": 0,
            "detected_lesions_any_overlap": 0,
            "predicted_lesions": 1,
            "native_size_group": "normal",
        },
        candidate_context=None,
    )
    assert flags["normal_false_positive"]
    assert sum(flags.values()) == 1
    assert primary_failure(flags) == "normal_false_positive"


def test_missing_multifocal_component():
    flags = classify_failure(
        {
            "gt_positive": True,
            "predicted_positive": True,
            "dice": 0.4,
            "zero_overlap": False,
            "predicted_gt_area_ratio": 1.0,
            "gt_lesions": 3,
            "detected_lesions_any_overlap": 2,
            "predicted_lesions": 2,
            "native_size_group": "medium_1_to_5pct",
        },
        candidate_context=None,
    )
    assert flags["missing_component"]
    assert not flags["fragmented_mask"]
