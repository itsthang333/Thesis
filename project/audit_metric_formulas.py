from __future__ import annotations

"""Executable, annotation-free verification of thesis metric formulae."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_g4_classifier_labels import _average_precision, _binary_metrics
from evaluation.segmentation_metrics import segmentation_metrics, summarize_segmentation_rows
from final_selector import fixed_rank_fusion
from frozen_io import sha256_file
from g4_ablation import UpstreamComponents, upstream_score
from models.layercam import collapsed_tumor_log_odds
from models.rad_dino_mask_bag_mil import smooth_mil_pool


def _close(actual: float, expected: float, *, atol: float = 1e-9) -> bool:
    return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []

    def record(name: str, actual: float, expected: float, *, atol: float = 1e-9) -> None:
        checks.append(
            {
                "name": name,
                "actual": float(actual),
                "expected": float(expected),
                "absolute_error": abs(float(actual) - float(expected)),
                "atol": atol,
                "pass": _close(actual, expected, atol=atol),
            }
        )

    target = np.zeros((4, 4), dtype=np.uint8)
    target[:2, :2] = 1
    prediction = np.zeros_like(target)
    prediction[:2, :3] = 1
    segmentation = segmentation_metrics(prediction, target)
    # TP=4, FP=2, FN=0: Dice=8/10 and IoU=4/6.
    record("dice_confusion_fixture", segmentation["dice"], 0.8)
    record("iou_confusion_fixture", segmentation["iou"], 4.0 / 6.0)
    record(
        "dice_iou_identity",
        segmentation["dice"],
        2.0 * float(segmentation["iou"]) / (1.0 + float(segmentation["iou"])),
    )
    record("precision_confusion_fixture", segmentation["precision"], 4.0 / 6.0)
    record("recall_confusion_fixture", segmentation["recall"], 1.0)
    record("relative_volume_difference_fixture", segmentation["relative_area_difference"], 0.5)
    both_empty = segmentation_metrics(np.zeros((3, 3)), np.zeros((3, 3)))
    record("both_empty_dice_convention", both_empty["dice"], 1.0)

    perfect = segmentation_metrics(target, target)
    missed_target = np.ones((4, 4), dtype=np.uint8)
    missed = segmentation_metrics(np.zeros_like(missed_target), missed_target)
    rows = [
        {"image_id": "a", "group_id": "a", **perfect},
        {"image_id": "b", "group_id": "b", **missed},
    ]
    summary = summarize_segmentation_rows(rows)
    record("macro_dice_fixture", summary["mean_tumor_dice"], 0.5)
    # Aggregated TP=4; target pixels=4+16, prediction pixels=4.
    record("micro_dice_fixture", summary["micro_dice"], 8.0 / 24.0)

    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    tied_scores = np.asarray([0.9, 0.9, 0.2, 0.1], dtype=np.float64)
    record(
        "average_precision_tied_threshold_fixture",
        _average_precision(labels, tied_scores),
        0.5 * 0.5 + 0.5 * (2.0 / 3.0),
    )
    binary = _binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.4, 0.4, 0.9], dtype=np.float64),
    )
    # Four positive-negative pairs: wins=3 and ties=1 -> (3+0.5)/4.
    record("auroc_mann_whitney_tie_fixture", binary["auroc"], 0.875)
    expected_brier = np.mean((np.asarray([0.1, 0.4, 0.4, 0.9]) - np.asarray([0, 0, 1, 1])) ** 2)
    record("brier_fixture", binary["brier_score"], expected_brier)
    expected_nll = -np.mean(
        np.asarray([0, 0, 1, 1]) * np.log(np.asarray([0.1, 0.4, 0.4, 0.9]))
        + np.asarray([1, 1, 0, 0]) * np.log(1.0 - np.asarray([0.1, 0.4, 0.4, 0.9]))
    )
    record("binary_nll_fixture", binary["negative_log_likelihood"], expected_nll)

    mcc = _binary_metrics(
        np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64),
        np.asarray([0.9, 0.8, 0.2, 0.7, 0.1, 0.2], dtype=np.float64),
    )
    # TP=2, FN=1, FP=1, TN=2 -> numerator=3, denominator=9.
    record("mcc_confusion_fixture", mcc["matthews_correlation_coefficient"], 1.0 / 3.0)
    calibrated = _binary_metrics(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0.1, 0.9], dtype=np.float64),
    )
    record("ece_15_equal_width_fixture", calibrated["ece_15_equal_width"], 0.1)

    lesion_target = np.zeros((12, 12), dtype=np.uint8)
    lesion_target[1:3, 1:3] = 1
    lesion_target[8:10, 8:10] = 1
    lesion_prediction = lesion_target.copy()
    lesion_prediction[5, 5] = 1
    lesion = segmentation_metrics(lesion_prediction, lesion_target)
    record("lesion_ground_truth_component_count", lesion["gt_lesions"], 2.0)
    record("lesion_predicted_component_count", lesion["predicted_lesions"], 3.0)
    record("lesion_one_to_one_iou50_fixture", lesion["lesion_tp_one_to_one_iou50"], 2.0)

    surface_target = np.zeros((12, 12), dtype=np.uint8)
    surface_target[1:5, 1:9] = 1
    surface_prediction = np.zeros_like(surface_target)
    surface_prediction[2:10, 2:4] = 1
    surface = segmentation_metrics(surface_prediction, surface_target)
    # Independently recorded from MONAI 1.5.1 symmetric surface metrics.
    record("assd_monai_1_5_1_fixture", surface["assd_px"], 2.26246953, atol=1e-6)
    record("hd95_monai_1_5_1_fixture", surface["hd95_px"], 5.00495100, atol=1e-6)

    logits = torch.tensor([[0.3, -0.2, 1.1, 0.7]], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=1)
    expected_log_odds = torch.log(probabilities[:, 1:].sum(dim=1) / probabilities[:, 0])
    actual_log_odds = collapsed_tumor_log_odds(logits)
    record("collapsed_ten_class_tumor_log_odds", actual_log_odds.item(), expected_log_odds.item(), atol=1e-12)

    candidate_logits = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64)
    valid = torch.ones_like(candidate_logits, dtype=torch.bool)
    pooled = smooth_mil_pool(candidate_logits, valid, temperature=0.2)
    record("normalized_logsumexp_equal_logits", pooled.item(), 2.0, atol=1e-12)

    fused = fixed_rank_fusion(
        np.asarray([1.0, 3.0, 2.0]), np.asarray([3.0, 1.0, 2.0])
    )
    for index, expected in enumerate((0.5, 0.5, 0.5)):
        record(f"equal_percentile_rank_fixture_{index}", fused[index], expected)

    upstream = UpstreamComponents(
        sam_score=np.asarray([0.2]),
        cam_density=np.asarray([0.4]),
        cam_mass_coverage=np.asarray([0.8]),
        sam_component_rank=np.asarray([0.6]),
        sam_global_rank=np.asarray([0.7]),
    )
    record("project_specific_U5_algebra", upstream_score(upstream, "U5")[0], 0.53)

    passed = all(bool(check["pass"]) for check in checks)
    report = {
        "schema_version": 1,
        "stage": "thesis_metric_formula_executable_audit_v1",
        "pass": passed,
        "checks_passed": sum(bool(check["pass"]) for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "implementation_sha256": {
            "classifier_metrics": sha256_file(Path(__file__).resolve().parent / "evaluate_g4_classifier_labels.py"),
            "segmentation_metrics": sha256_file(Path(__file__).resolve().parent / "evaluation" / "segmentation_metrics.py"),
            "collapsed_log_odds": sha256_file(Path(__file__).resolve().parent / "models" / "layercam.py"),
            "g1_pooling": sha256_file(Path(__file__).resolve().parent / "models" / "rad_dino_mask_bag_mil.py"),
            "rank_fusion": sha256_file(Path(__file__).resolve().parent / "final_selector.py"),
            "upstream_ablation": sha256_file(Path(__file__).resolve().parent / "g4_ablation.py"),
        },
        "external_references": {
            "metrics_reloaded": "https://www.nature.com/articles/s41592-023-02151-z",
            "taha_hanbury": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/",
            "monai_surface_fixture_version": "1.5.1",
        },
        "boundary_units": "pixels on the declared grid, not millimetres",
        "spatial_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(args.output)}, indent=2))
    if not passed:
        raise SystemExit("metric formula audit failed")


if __name__ == "__main__":
    main()
