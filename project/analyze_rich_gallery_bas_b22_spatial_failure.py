from __future__ import annotations

"""Post-freeze spatial and identifiability analysis for BAS-B2.2."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import sha256_file


BASELINE = "g1_upstream_baseline"
PRIMARY = "g1_upstream_bas_three_way"
EXPECTED_BASELINE_DICE = 0.28872948670665205
SOURCES = ("classifier448", "external_saliency", "layercam320")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--b22-output-root", type=Path, required=True)
    parser.add_argument("--expected-evaluation-audit-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dossier-output", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = _ranks(left)
    y = _ranks(right)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    ranks = _ranks(np.asarray(scores, dtype=np.float64)) + 1.0
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _stable_fold(group_id: str, folds: int = 5) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def _feature_vector(row: dict[str, object]) -> list[float]:
    base_area = float(row["baseline_area_ratio"])
    primary_area = float(row["primary_area_ratio"])
    vector = [
        float(np.log(base_area + 1.0e-6)),
        float(np.log(primary_area + 1.0e-6)),
        float(np.log(primary_area / max(base_area, 1.0e-6))),
        float(row["baseline_border_fraction"]),
        float(row["primary_border_fraction"]),
        float(row["tumor_probability"]),
        float(row["activation_mean"]),
        float(row["activation_std"]),
        float(row["bas_area_spearman"]),
        float(np.log(float(row["candidate_count"]))),
    ]
    vector.extend(float(row["baseline_source"] == source) for source in SOURCES)
    vector.extend(float(row["primary_source"] == source) for source in SOURCES)
    return vector


def _oof_ridge_delta(rows: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
    features = np.asarray([_feature_vector(row) for row in rows], dtype=np.float64)
    target = np.asarray([float(row["delta_dice"]) for row in rows], dtype=np.float64)
    folds = np.asarray([_stable_fold(str(row["group_id"])) for row in rows], dtype=np.int64)
    predictions = np.zeros(len(rows), dtype=np.float64)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1.0e-8] = 1.0
        x_train = (features[train] - mean) / scale
        x_valid = (features[valid] - mean) / scale
        design = np.column_stack((np.ones(int(train.sum())), x_train))
        valid_design = np.column_stack((np.ones(int(valid.sum())), x_valid))
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target[train],
        )
        predictions[valid] = valid_design @ weights
    return predictions, {
        "fold_counts": [int(np.sum(folds == fold)) for fold in range(5)],
        "ridge_lambda": 1.0,
        "decision_threshold": 0.0,
        "features": [
            "log_baseline_area",
            "log_primary_area",
            "log_area_expansion",
            "baseline_border_fraction",
            "primary_border_fraction",
            "tumor_probability",
            "activation_mean",
            "activation_std",
            "bas_area_spearman",
            "log_candidate_count",
            "baseline_source_one_hot",
            "primary_source_one_hot",
        ],
        "validation_gt_trained_diagnostic_only": True,
        "promotable": False,
    }


def _render_dossier(summary: dict[str, object]) -> str:
    actual = summary["actual_metrics"]
    baseline = actual["baseline"]
    primary = actual["primary"]
    switch = summary["baseline_or_b22_oracle_switch"]
    gate = summary["observable_gate_diagnostic"]
    return f"""# Rich-gallery BAS-B2.2 spatial failure dossier

## Frozen endpoint

The B2.2 maps, five candidate-score variants and all 1,855 choices were frozen
before validation polygons. Stage B reproduced the G1+upstream baseline on
371/371 validation images and evaluated 184 tumors split 94/72/18. Test was
not opened.

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1+upstream baseline | {baseline['dice']:.6f} | {baseline['iou']:.6f} | {baseline['small']:.6f} | {baseline['medium']:.6f} | {baseline['large']:.6f} | {baseline['misses']} |
| B2.2 only | {actual['bas_only']['dice']:.6f} | {actual['bas_only']['iou']:.6f} | {actual['bas_only']['small']:.6f} | {actual['bas_only']['medium']:.6f} | {actual['bas_only']['large']:.6f} | {actual['bas_only']['misses']} |
| G1+B2.2 | {actual['g1_bas']['dice']:.6f} | {actual['g1_bas']['iou']:.6f} | {actual['g1_bas']['small']:.6f} | {actual['g1_bas']['medium']:.6f} | {actual['g1_bas']['large']:.6f} | {actual['g1_bas']['misses']} |
| upstream+B2.2 | {actual['upstream_bas']['dice']:.6f} | {actual['upstream_bas']['iou']:.6f} | {actual['upstream_bas']['small']:.6f} | {actual['upstream_bas']['medium']:.6f} | {actual['upstream_bas']['large']:.6f} | {actual['upstream_bas']['misses']} |
| G1+upstream+B2.2 | {primary['dice']:.6f} | {primary['iou']:.6f} | {primary['small']:.6f} | {primary['medium']:.6f} | {primary['large']:.6f} | {primary['misses']} |

The primary delta is `{primary['delta']:+.6f}` with complete-group bootstrap
CI95 `[{primary['ci95_low']:.6f}, {primary['ci95_high']:.6f}]`. B2.2 is
therefore decisively worse overall, not a noisy tie.

## What B2.2 changes

B2.2 changes 80.86% of tumor selections. It wins on 53 images, ties on 47 and
loses on 84. It recovers only 3 baseline misses while turning 27 previous hits
into misses. Positive Dice mass is `+6.1840`; negative mass is `-24.0327`.

The effect is scale-opposite:

- small: `{primary['small_delta']:+.6f}`;
- medium: `{primary['medium_delta']:+.6f}`;
- large: `{primary['large_delta']:+.6f}`.

The large gain is real, but the candidate extent reveals why it cannot be
applied globally. Median selected/GT area changes from
`14.60/1.10/0.38` in the baseline to `158.24/4.93/1.43` for
small/medium/large. B2.2 is a strong expansion prior: it roughly repairs the
baseline's large-lesion under-extent while catastrophically amplifying the
existing small-lesion over-extent.

## No usable label-safe router was found

Choosing the better of baseline and B2.2 per image using GT would reach Dice
`{switch['dice']:.6f}` (`{switch['delta']:+.6f}`), so B2.2 contains limited
complementary choices. That is an oracle diagnostic, not an algorithm.

A deterministic five-fold group-separated ridge diagnostic used only
observable frozen quantities: baseline/B2.2 area and border, image tumor
probability, activation mean/std, candidate-area dependence, candidate count
and source identities. Its out-of-fold routed Dice is
`{gate['dice']:.6f}` (`{gate['delta']:+.6f}`), with rank correlation
`{gate['prediction_delta_spearman']:.6f}` and non-tie sign accuracy
`{gate['non_tie_sign_accuracy']:.3f}`. It improves large lesions but harms both
small and medium, reproducing the overall baseline rather than exceeding it.
This diagnostic is itself validation-GT-trained and non-promotable; its failure
shows that the observed B2.2 benefit cannot be isolated reliably from current
label-safe metadata.

## Exact bottleneck update

The gallery remains adequate: oracle Dice is 0.528298 and truncation regret is
0.000396. B2.2 increases selector regret from 0.239569 to 0.336572. Its
within-selected-source regret rises from 0.168376 to 0.230754 and cross-source
regret from 0.070796 to 0.105422. Thus it worsens both major selector terms.

The missing observable is not another global area, source, confidence, CAM or
anatomy score. It is **candidate-conditioned tumor identity plus signed extent
calibration**:

1. Does the candidate contain tumor-specific content rather than mostly normal
   bone (critical for small lesions)?
2. Does removing the candidate erase tumor evidence, and does keeping only the
   candidate preserve it (critical for medium/large extent)?
3. Is the evidence stronger inside the candidate than in a matched local ring,
   and absent on candidates from train-normal images?

## Research decision

1. Retire BAS-B2.2 and all BAS epoch/weight/threshold/seed routing sweeps.
2. Preserve G1+upstream fixed fusion at Dice 0.2887294867.
3. The next bounded mechanism must operate per candidate, not per image:
   inside-versus-local-ring evidence with exact candidate masking, train-normal
   hard negatives, and a zero-initialized residual on the immutable baseline.
4. Before any full selector training, require a cheap matched diagnostic to
   reduce within-source oracle rank without increasing candidate-area
   dependence; large-only gains or GT-size routing are insufficient.
5. Keep test locked.
"""


def _metric(summary: dict[str, object], variant: str) -> dict[str, float | int]:
    values = summary["actual_binary_mask_metrics"][variant]
    overall = values["overall"]
    return {
        "dice": float(overall["dice"]),
        "iou": float(overall["iou"]),
        "small": float(values["small"]["dice"]),
        "medium": float(values["medium"]["dice"]),
        "large": float(values["large"]["dice"]),
        "misses": int(overall["complete_misses"]),
    }


def main() -> None:
    args = parse_args()
    audit_path = args.stage_b_root / "evaluation_audit.json"
    if sha256_file(audit_path) != args.expected_evaluation_audit_sha256:
        raise ValueError("Stage-B evaluation-audit SHA-256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("audit_pass") is not True
        or audit.get("test_images_read") != 0
        or audit.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-B evaluator did not pass its locked audit")
    summary_path = args.stage_b_root / "summary.json"
    per_image_path = args.stage_b_root / "per_image.csv"
    if (
        sha256_file(summary_path) != audit["summary_sha256"]
        or sha256_file(per_image_path) != audit["per_image_sha256"]
    ):
        raise ValueError("Stage-B summary/per-image payload changed")
    stage_b = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        stage_b.get("candidate_choices_frozen_before_validation_gt") is not True
        or stage_b.get("cohort")
        != {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            "small": 94,
            "medium": 72,
            "large": 18,
        }
        or stage_b.get("test_evaluated") is not False
    ):
        raise ValueError("Stage-B cohort/access contract mismatch")
    baseline_dice = float(
        stage_b["actual_binary_mask_metrics"][BASELINE]["overall"]["dice"]
    )
    if not np.isclose(baseline_dice, EXPECTED_BASELINE_DICE, atol=1.0e-12):
        raise ValueError("Stage-B baseline did not reproduce")

    probe_path = args.b22_output_root / "probe_summary.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    activation_path = args.b22_output_root / "activation_manifest.csv"
    predictions_path = args.b22_output_root / "validation_predictions.csv"
    if (
        sha256_file(activation_path) != probe["activation_manifest_sha256"]
        or sha256_file(predictions_path) != probe["validation_predictions_sha256"]
        or probe.get("validation_gt_read") is not False
        or probe.get("test_evaluated") is not False
    ):
        raise ValueError("B2.2 label-safe payload changed")
    activation = {row["image_id"]: row for row in _read_csv(activation_path)}
    predictions = {row["image_id"]: row for row in _read_csv(predictions_path)}
    evaluated = _read_csv(per_image_path)
    variants: dict[str, dict[str, dict[str, str]]] = {}
    for row in evaluated:
        variants.setdefault(row["variant"], {})[row["image_id"]] = row
    if set(variants[BASELINE]) != set(variants[PRIMARY]) or len(variants[BASELINE]) != 184:
        raise ValueError("baseline/primary per-image cohorts differ")

    compact: list[dict[str, object]] = []
    for image_id, baseline in variants[BASELINE].items():
        primary = variants[PRIMARY][image_id]
        delta = float(primary["dice"]) - float(baseline["dice"])
        compact.append(
            {
                "image_id": image_id,
                "group_id": baseline["group_id"],
                "size_group": baseline["size_group"],
                "gt_area_ratio": float(baseline["gt_area_ratio"]),
                "baseline_dice": float(baseline["dice"]),
                "primary_dice": float(primary["dice"]),
                "delta_dice": delta,
                "baseline_area_ratio": float(baseline["selected_area_ratio"]),
                "primary_area_ratio": float(primary["selected_area_ratio"]),
                "area_expansion": float(primary["selected_area_ratio"])
                / max(float(baseline["selected_area_ratio"]), 1.0e-12),
                "baseline_gt_area_ratio": float(baseline["selected_gt_area_ratio"]),
                "primary_gt_area_ratio": float(primary["selected_gt_area_ratio"]),
                "baseline_source": baseline["selected_source"],
                "primary_source": primary["selected_source"],
                "baseline_border_fraction": float(baseline["selected_border_fraction"]),
                "primary_border_fraction": float(primary["selected_border_fraction"]),
                "tumor_probability": float(predictions[image_id]["tumor_probability"]),
                "activation_mean": float(activation[image_id]["activation_mean"]),
                "activation_std": float(activation[image_id]["activation_std"]),
                "bas_area_spearman": float(activation[image_id]["bas_area_spearman"]),
                "candidate_count": int(activation[image_id]["candidate_count"]),
            }
        )
    compact.sort(key=lambda row: str(row["image_id"]))
    compact_sha = _write_csv(args.output_csv, compact)

    delta = np.asarray([float(row["delta_dice"]) for row in compact])
    baseline_values = np.asarray([float(row["baseline_dice"]) for row in compact])
    primary_values = np.asarray([float(row["primary_dice"]) for row in compact])
    predictions_oof, gate_protocol = _oof_ridge_delta(compact)
    choose_primary = predictions_oof > 0.0
    routed = np.where(choose_primary, primary_values, baseline_values)
    oracle_switch = np.maximum(primary_values, baseline_values)
    non_ties = delta != 0.0
    gate_summary = {
        **gate_protocol,
        "chosen_primary_images": int(choose_primary.sum()),
        "dice": float(routed.mean()),
        "delta": float(routed.mean() - baseline_values.mean()),
        "prediction_delta_spearman": _spearman(predictions_oof, delta),
        "non_tie_sign_accuracy": float(
            np.mean((predictions_oof[non_ties] > 0.0) == (delta[non_ties] > 0.0))
        ),
        "subgroups": {},
    }
    groups = np.asarray([str(row["size_group"]) for row in compact])
    for group in ("small", "medium", "large"):
        selected = groups == group
        gate_summary["subgroups"][group] = {
            "n": int(selected.sum()),
            "chosen_primary": int(choose_primary[selected].sum()),
            "dice": float(routed[selected].mean()),
            "baseline_dice": float(baseline_values[selected].mean()),
            "delta": float(routed[selected].mean() - baseline_values[selected].mean()),
        }

    observable_names = (
        "baseline_area_ratio",
        "primary_area_ratio",
        "area_expansion",
        "tumor_probability",
        "activation_mean",
        "activation_std",
        "bas_area_spearman",
        "candidate_count",
        "baseline_border_fraction",
        "primary_border_fraction",
    )
    win = (delta > 0.0).astype(np.int64)
    univariate = {}
    for name in observable_names:
        values = np.asarray([float(row[name]) for row in compact], dtype=np.float64)
        auc = _auc(win, values)
        univariate[name] = {
            "win_auc": auc,
            "direction_free_auc": max(auc, 1.0 - auc),
            "spearman_with_delta": _spearman(values, delta),
        }

    metrics = {
        "baseline": _metric(stage_b, BASELINE),
        "bas_only": _metric(stage_b, "bas_only"),
        "g1_bas": _metric(stage_b, "g1_bas_two_way"),
        "upstream_bas": _metric(stage_b, "upstream_bas_two_way"),
        "primary": _metric(stage_b, PRIMARY),
    }
    primary_summary = metrics["primary"]
    primary_summary.update(
        {
            "delta": float(primary_summary["dice"] - metrics["baseline"]["dice"]),
            "small_delta": float(primary_summary["small"] - metrics["baseline"]["small"]),
            "medium_delta": float(primary_summary["medium"] - metrics["baseline"]["medium"]),
            "large_delta": float(primary_summary["large"] - metrics["baseline"]["large"]),
            "ci95_low": float(stage_b["paired_bootstrap_primary_vs_baseline"]["overall"]["ci95_low"]),
            "ci95_high": float(stage_b["paired_bootstrap_primary_vs_baseline"]["overall"]["ci95_high"]),
        }
    )
    output = {
        "stage": "rich_gallery_bas_b22_spatial_failure_analysis_v1",
        "actual_metrics": metrics,
        "transitions": stage_b["failure_decomposition"]["hit_miss_transitions"],
        "wins": int(np.sum(delta > 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "positive_dice_mass": float(delta[delta > 0.0].sum()),
        "negative_dice_mass": float(delta[delta < 0.0].sum()),
        "baseline_or_b22_oracle_switch": {
            "dice": float(oracle_switch.mean()),
            "delta": float(oracle_switch.mean() - baseline_values.mean()),
            "validation_gt_oracle_only": True,
            "promotable": False,
        },
        "observable_gate_diagnostic": gate_summary,
        "univariate_observable_diagnostics": univariate,
        "selector_regret": {
            "baseline": float(
                stage_b["actual_binary_mask_metrics"][BASELINE]["overall"]["selector_regret"]
            ),
            "primary": float(
                stage_b["actual_binary_mask_metrics"][PRIMARY]["overall"]["selector_regret"]
            ),
            "baseline_within_source": float(
                stage_b["actual_binary_mask_metrics"][BASELINE]["overall"]["within_selected_source_regret"]
            ),
            "primary_within_source": float(
                stage_b["actual_binary_mask_metrics"][PRIMARY]["overall"]["within_selected_source_regret"]
            ),
            "baseline_cross_source": float(
                stage_b["actual_binary_mask_metrics"][BASELINE]["overall"]["cross_source_regret"]
            ),
            "primary_cross_source": float(
                stage_b["actual_binary_mask_metrics"][PRIMARY]["overall"]["cross_source_regret"]
            ),
            "candidate_truncation": float(
                stage_b["actual_binary_mask_metrics"][PRIMARY]["overall"]["candidate_truncation_regret"]
            ),
        },
        "decision": {
            "bas_b22_retired": True,
            "baseline_preserved": True,
            "observable_router_supported": False,
            "next_bottleneck": "candidate_conditioned_tumor_identity_and_signed_extent",
            "no_sweep": True,
        },
        "input_hashes": {
            "evaluation_audit_sha256": args.expected_evaluation_audit_sha256,
            "stage_b_summary_sha256": audit["summary_sha256"],
            "stage_b_per_image_sha256": audit["per_image_sha256"],
            "b22_probe_summary_sha256": sha256_file(probe_path),
            "b22_activation_manifest_sha256": sha256_file(activation_path),
            "b22_validation_predictions_sha256": sha256_file(predictions_path),
        },
        "compact_per_image_sha256": compact_sha,
        "validation_gt_read_only_after_prediction_freeze": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.dossier_output.parent.mkdir(parents=True, exist_ok=True)
    args.dossier_output.write_text(_render_dossier(output), encoding="utf-8")
    output["dossier_sha256"] = sha256_file(args.dossier_output)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
