from __future__ import annotations

"""Decompose the frozen SMILE rich-gallery validation failure.

This is a validation-only diagnostic.  It never changes a frozen choice and
never reads the test split.  Candidate/GT overlap is used only after Stage A
has frozen every score and selection.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from project.datasets.factory import build_segmentation_dataset
from project.datasets.smile_reference import sha256_file


ARMS = ("control", "full")
VARIANTS = ("baseline", "identity_only", "identity_extent")
PRIMARY = ("full", "identity_extent")
BASELINE = ("control", "baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--control-stage-a", type=Path, required=True)
    parser.add_argument("--full-stage-a", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) < 2 or len(first) != len(second):
        return float("nan")
    a = average_ranks(first)
    b = average_ranks(second)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / max(1, denominator))


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    if "fallback" in lowered:
        return "fallback"
    raise ValueError(f"unknown source: {value!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "positive_fraction": float(np.mean(array > 0)),
    }


def transition_summary(
    rows: dict[tuple[str, str, str], dict[str, str]],
    first: tuple[str, str],
    second: tuple[str, str],
) -> dict[str, object]:
    ids = sorted(image_id for arm, variant, image_id in rows if (arm, variant) == first)
    by_group: dict[str, dict[str, object]] = {}
    for subgroup in ("overall", "small", "medium", "large"):
        current = [
            image_id
            for image_id in ids
            if subgroup == "overall" or rows[(first[0], first[1], image_id)]["size_group"] == subgroup
        ]
        deltas = np.asarray(
            [
                float(rows[(second[0], second[1], image_id)]["dice"])
                - float(rows[(first[0], first[1], image_id)]["dice"])
                for image_id in current
            ],
            dtype=np.float64,
        )
        area_multipliers = np.asarray(
            [
                float(rows[(second[0], second[1], image_id)]["selected_gt_area_ratio"])
                / max(1e-12, float(rows[(first[0], first[1], image_id)]["selected_gt_area_ratio"]))
                for image_id in current
            ],
            dtype=np.float64,
        )
        first_miss = np.asarray(
            [int(rows[(first[0], first[1], image_id)]["complete_miss"]) for image_id in current]
        )
        second_miss = np.asarray(
            [int(rows[(second[0], second[1], image_id)]["complete_miss"]) for image_id in current]
        )
        by_group[subgroup] = {
            "n": len(current),
            "mean_dice_delta": float(deltas.mean()),
            "improved": int(np.sum(deltas > 1e-12)),
            "harmed": int(np.sum(deltas < -1e-12)),
            "unchanged": int(np.sum(np.abs(deltas) <= 1e-12)),
            "same_candidate": int(
                sum(
                    rows[(first[0], first[1], image_id)]["selected_candidate_index"]
                    == rows[(second[0], second[1], image_id)]["selected_candidate_index"]
                    for image_id in current
                )
            ),
            "same_source": int(
                sum(
                    rows[(first[0], first[1], image_id)]["selected_source"]
                    == rows[(second[0], second[1], image_id)]["selected_source"]
                    for image_id in current
                )
            ),
            "miss_recovered": int(np.sum((first_miss == 1) & (second_miss == 0))),
            "new_miss": int(np.sum((first_miss == 0) & (second_miss == 1))),
            "median_selected_area_multiplier": float(np.median(area_multipliers)),
        }
    source_transitions = Counter(
        (
            rows[(first[0], first[1], image_id)]["selected_source"],
            rows[(second[0], second[1], image_id)]["selected_source"],
        )
        for image_id in ids
    )
    return {
        "first": f"{first[0]}::{first[1]}",
        "second": f"{second[0]}::{second[1]}",
        "subgroups": by_group,
        "source_transitions": {
            f"{source}->{target}": count
            for (source, target), count in sorted(source_transitions.items())
        },
    }


def markdown(report: dict[str, object]) -> str:
    decision = report["stage_b_decision"]["summary"]
    baseline = decision["control"]["baseline"]
    primary = decision["full"]["identity_extent"]
    candidate = report["candidate_level"]
    transition = report["transitions"]["baseline_to_primary"]["subgroups"]
    identity = candidate["full_identity_vs_log_area"]
    extent = candidate["full_extent_vs_log_area"]
    quality = candidate["full_score_vs_candidate_dice"]
    source = report["selected_source_counts"]
    return f"""# SMILE + immutable rich gallery failure dossier

## Decision

SMILE is retired in its current readout form.  Frozen validation Dice/IoU is
`{primary['overall']['dice']:.9f}/{primary['overall']['iou']:.9f}` versus the
exact G1 + fixed rank baseline `{baseline['overall']['dice']:.9f}/{baseline['overall']['iou']:.9f}`.
Subgroup Dice changes from
`{baseline['small']['dice']:.6f}/{baseline['medium']['dice']:.6f}/{baseline['large']['dice']:.6f}`
to `{primary['small']['dice']:.6f}/{primary['medium']['dice']:.6f}/{primary['large']['dice']:.6f}`.
The overall paired group-bootstrap interval is
`[{report['stage_b_decision']['bootstrap']['primary_vs_baseline']['overall']['ci_low']:.6f},
{report['stage_b_decision']['bootstrap']['primary_vs_baseline']['overall']['ci_high']:.6f}]`.

## What changed

- Complete misses fall from `{baseline['overall']['complete_misses']}` to
  `{primary['overall']['complete_misses']}`: the local evidence is not empty.
- Precision/recall shifts from `{baseline['overall']['precision']:.4f}/{baseline['overall']['recall']:.4f}`
  to `{primary['overall']['precision']:.4f}/{primary['overall']['recall']:.4f}`.
  SMILE primarily buys recall at a much larger precision cost.
- Median selected/GT area becomes
  `{primary['small']['median_selected_gt_area_ratio']:.2f}x/{primary['medium']['median_selected_gt_area_ratio']:.2f}x/{primary['large']['median_selected_gt_area_ratio']:.2f}x`
  for small/medium/large, versus
  `{baseline['small']['median_selected_gt_area_ratio']:.2f}x/{baseline['medium']['median_selected_gt_area_ratio']:.2f}x/{baseline['large']['median_selected_gt_area_ratio']:.2f}x`.
- Baseline-to-primary choices improve/harm
  `{transition['small']['improved']}/{transition['small']['harmed']}` small,
  `{transition['medium']['improved']}/{transition['medium']['harmed']}` medium and
  `{transition['large']['improved']}/{transition['large']['harmed']}` large images.

## Root cause at candidate level

- The fixed-top-17 identity statistic is positively associated with candidate
  area within an image: median Spearman `{identity['median']:.4f}` (positive in
  `{identity['positive_fraction']:.1%}` of evaluable images).  A larger mask has
  more chances to contain 17 extreme evidence cells, while the surrounding-ring
  median is not an equal-size null.  This is a multiple-comparisons/scale bias,
  not valid lesion-identity evidence.
- The soft extent score is even more area coupled: median within-image Spearman
  `{extent['median']:.4f}` (positive in `{extent['positive_fraction']:.1%}`).
  Image-label MIL makes sparse discriminative evidence, not a calibrated lesion
  occupancy map, so soft Dice between `sigmoid(evidence)` and a proposal cannot
  identify lesion extent.
- Median within-image rank correlation with true candidate Dice is baseline
  `{quality['baseline']['median']:.4f}`, identity `{quality['identity']['median']:.4f}`,
  extent `{quality['extent']['median']:.4f}` and the final residual
  `{quality['identity_extent']['median']:.4f}`.  The small correlation gain of
  the combined score does not transfer to top-1 selection: near the top of the
  baseline ranking, the scale-biased residual promotes larger wrong masks.
- Selected sources move from `{source['baseline']}` to `{source['primary']}`.
  The increased external-source use is a secondary source shortcut; selector
  regret rises from `{baseline['overall']['selector_regret']:.6f}` to
  `{primary['overall']['selector_regret']:.6f}`.

## Mechanistic interpretation

The evidence map contains useful *presence* signal: recall, recovered misses and
large-lesion Dice all improve.  The failure is the conversion from presence to
candidate identity/extent.  Both frozen residual statistics reward scale, so
the same correction expands masks in every subgroup.  That is directionally
correct for large lesions (baseline under-segments) and catastrophic for tiny
lesions (baseline already over-segments).  Matched normal references provide a
small control-relative signal, but do not make lesion burden identifiable.

## Constrained successor

Do not sweep the two residual weights.  Preserve the exact baseline at zero and
replace both biased statistics in one bounded diagnostic:

1. candidate identity uses an equal-area, source-matched null (permutation or
   fixed-count sampling) so its score is corrected for candidate size;
2. extent is activated by a label-safe latent-burden gate learned from the
   *distribution* of local evidence (multi-quantiles and support concentration),
   not selected area alone; and
3. the gate emits signed shrink/neutral/expand behavior, with source-invariance
   control and actual Dice reported for all three subgroups.

The earlier area-only router failed, so candidate area cannot be the gate.  A
new run is justified only after a cheap frozen-map diagnostic shows that these
area-corrected features separate the required signed action.  Test remains
locked.
"""


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != args.split_sha256:
        raise ValueError("canonical split SHA mismatch")
    decision_path = args.stage_b / "decision.json"
    per_image_path = args.stage_b / "per_image.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("split_sha256") != args.split_sha256
        or decision.get("test_evaluated") is not False
        or decision.get("validation_images") != 371
        or decision.get("tumor_validation_images") != 184
    ):
        raise ValueError("Stage-B decision contract mismatch")
    if sha256_file(per_image_path) != decision.get("per_image_sha256"):
        raise ValueError("Stage-B per-image table changed")
    rows_list = read_csv(per_image_path)
    rows = {(row["arm"], row["variant"], row["image_id"]): row for row in rows_list}
    if len(rows) != 184 * len(ARMS) * len(VARIANTS):
        raise ValueError("Stage-B row count mismatch")

    candidate_manifest = {
        row["image_name"]: row for row in read_csv(args.candidate_root / "candidate_diagnostics_manifest.csv")
    }
    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    targets: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, target, image_id = dataset[index]
        key = str(image_id)
        if (BASELINE[0], BASELINE[1], key) in rows:
            targets[key] = target[0].numpy() > 0.5
    if len(targets) != 184:
        raise ValueError("tumor validation target count mismatch")

    correlations: dict[str, list[float]] = defaultdict(list)
    selected_source_counts = {
        "baseline": Counter(),
        "primary": Counter(),
        "control_identity_extent": Counter(),
    }
    candidate_rows = 0
    for image_id, target in sorted(targets.items()):
        candidate_row = candidate_manifest[image_id]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            all_masks = payload["sam_masks"].astype(bool)
        arm_payloads: dict[str, dict[str, np.ndarray]] = {}
        for arm, root in (("control", args.control_stage_a), ("full", args.full_stage_a)):
            score_path = root / "scores" / f"{Path(image_id).stem}.npz"
            with np.load(score_path, allow_pickle=False) as payload:
                arm_payloads[arm] = {key: payload[key].copy() for key in payload.files}
        full = arm_payloads["full"]
        indices = full["candidate_indices"].astype(np.int64)
        masks = all_masks[indices]
        areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float64)
        qualities = np.asarray([binary_dice(mask, target) for mask in masks], dtype=np.float64)
        candidate_rows += len(masks)
        log_area = np.log1p(areas)
        for arm in ARMS:
            payload = arm_payloads[arm]
            correlations[f"{arm}_identity_vs_log_area"].append(
                rank_correlation(payload["identity"], log_area)
            )
            correlations[f"{arm}_extent_vs_log_area"].append(
                rank_correlation(payload["extent"], log_area)
            )
            for name in ("baseline", "identity", "extent", "identity_only", "identity_extent"):
                correlations[f"{arm}_{name}_vs_quality"].append(
                    rank_correlation(payload[name], qualities)
                )
        for label, key in (
            ("baseline", BASELINE),
            ("primary", PRIMARY),
            ("control_identity_extent", ("control", "identity_extent")),
        ):
            selected_source_counts[label][rows[(key[0], key[1], image_id)]["selected_source"]] += 1

    score_quality = {
        name: finite_summary(correlations[f"full_{name}_vs_quality"])
        for name in ("baseline", "identity", "extent", "identity_only", "identity_extent")
    }
    report: dict[str, object] = {
        "method": "smile_rich_gallery_failure_decomposition_v1",
        "stage_b_decision": decision,
        "stage_b_decision_sha256": sha256_file(decision_path),
        "stage_b_per_image_sha256": sha256_file(per_image_path),
        "candidate_level": {
            "images": len(targets),
            "eligible_candidate_rows": candidate_rows,
            "full_identity_vs_log_area": finite_summary(correlations["full_identity_vs_log_area"]),
            "full_extent_vs_log_area": finite_summary(correlations["full_extent_vs_log_area"]),
            "control_identity_vs_log_area": finite_summary(correlations["control_identity_vs_log_area"]),
            "control_extent_vs_log_area": finite_summary(correlations["control_extent_vs_log_area"]),
            "full_score_vs_candidate_dice": score_quality,
            "control_score_vs_candidate_dice": {
                name: finite_summary(correlations[f"control_{name}_vs_quality"])
                for name in ("baseline", "identity", "extent", "identity_only", "identity_extent")
            },
        },
        "transitions": {
            "baseline_to_identity": transition_summary(rows, BASELINE, ("full", "identity_only")),
            "identity_to_identity_extent": transition_summary(
                rows, ("full", "identity_only"), PRIMARY
            ),
            "baseline_to_primary": transition_summary(rows, BASELINE, PRIMARY),
            "control_to_full": transition_summary(rows, ("control", "identity_extent"), PRIMARY),
        },
        "selected_source_counts": {
            name: dict(sorted(counts.items())) for name, counts in selected_source_counts.items()
        },
        "root_cause": {
            "primary": "scale-biased candidate readout, not gallery supply",
            "identity": "fixed top-17 inside evidence has unequal multiple-comparison opportunity across candidate areas",
            "extent": "soft Dice assumes a sparse MIL evidence map is calibrated lesion occupancy",
            "source": "area-coupled residual increases external-source selection",
            "subgroup": "one expansion direction helps under-segmented large lesions and harms already over-segmented small lesions",
            "matched_normal_signal": "present but weak: full improves slightly over query-only control and cannot overcome the readout bias",
        },
        "next_experiment_authorized": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "SMILE_FAILURE_ANALYSIS.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dossier_path = args.output_dir / "SMILE_FAILURE_DOSSIER.md"
    dossier_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "dossier": str(dossier_path),
                "dossier_sha256": sha256_file(dossier_path),
                "test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
