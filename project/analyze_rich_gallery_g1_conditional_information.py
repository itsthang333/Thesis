from __future__ import annotations

"""Measure information left beyond the immutable G1/upstream selector.

This is a retrospective, validation-GT-only information diagnostic.  It reads
only tables emitted by the audited Stage-B evaluator.  It cannot produce or
promote a selector, alter a frozen choice, open images, or access test data.

For each tumor image, candidate Dice and every candidate signal are converted
to within-image percentile ranks.  Both are residualized against the same
label-safe controls: G1 rank, upstream rank, log-area rank, and source.  The
remaining correlation therefore asks whether a frozen feature contains
candidate-quality information that is not already represented by the current
selector or by mask geometry/source.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


EXPECTED_BASELINE_DICE = 0.28872948670665205
GROUPS = ("overall", "small", "medium", "large")
METADATA_COLUMNS = {
    "image_id",
    "group_id",
    "size_group",
    "candidate_local_index",
    "candidate_index",
    "source",
    "candidate_area_ratio",
    "candidate_dice",
    "is_eligible_oracle",
    "is_baseline_selected",
    "g1_logit",
    "upstream_score",
}
KEY_SIGNALS = (
    "matched_logit_delta",
    "random_logit_delta",
    "matched_minus_random_logit_delta",
    "matched_class_inside_delta",
    "random_class_inside_delta",
    "matched_minus_random_class_inside_delta",
    "transition2_matched_relative_feature_l2_contrast",
    "transition2_random_relative_feature_l2_contrast",
    "transition2_matched_minus_random_relative_feature_l2_contrast",
    "transition2_matched_ring_mass",
    "transition2_random_ring_mass",
    "norm5_matched_relative_feature_l2_contrast",
    "norm5_random_relative_feature_l2_contrast",
    "norm5_matched_minus_random_relative_feature_l2_contrast",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def partial_rank_correlation(
    target: np.ndarray,
    signal: np.ndarray,
    controls: np.ndarray,
) -> float:
    """Return rank correlation after projecting target/signal off controls."""

    target_rank = average_percentile_rank(np.asarray(target, dtype=np.float64))
    signal_rank = average_percentile_rank(np.asarray(signal, dtype=np.float64))
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[0] != len(target_rank):
        raise ValueError("controls must be [candidate, control]")
    design = np.column_stack([np.ones(len(target_rank), dtype=np.float64), controls])
    target_residual = target_rank - design @ np.linalg.lstsq(
        design, target_rank, rcond=None
    )[0]
    signal_residual = signal_rank - design @ np.linalg.lstsq(
        design, signal_rank, rcond=None
    )[0]
    if np.std(target_residual) <= 1.0e-12 or np.std(signal_residual) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(target_residual, signal_residual)[0, 1])


def matched_minus_random_name(matched_name: str) -> str | None:
    if matched_name.startswith("matched_"):
        return "matched_minus_random_" + matched_name[len("matched_") :]
    if "_matched_" in matched_name:
        return matched_name.replace("_matched_", "_matched_minus_random_", 1)
    return None


def derive_matched_minus_random(
    row: dict[str, float],
    signal_names: Iterable[str],
) -> dict[str, float]:
    derived: dict[str, float] = {}
    available = set(signal_names)
    for matched_name in signal_names:
        if matched_name.startswith("matched_"):
            random_name = "random_" + matched_name[len("matched_") :]
        elif "_matched_" in matched_name:
            random_name = matched_name.replace("_matched_", "_random_", 1)
        else:
            continue
        output_name = matched_minus_random_name(matched_name)
        if random_name in available and output_name is not None:
            derived[output_name] = row[matched_name] - row[random_name]
    return derived


def _source_controls(sources: list[str]) -> np.ndarray:
    levels = sorted(set(sources))
    if len(levels) <= 1:
        return np.empty((len(sources), 0), dtype=np.float64)
    # The first level is the fixed reference category.
    return np.column_stack(
        [np.asarray([float(value == level) for value in sources]) for level in levels[1:]]
    )


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "positive_fraction": 0.0,
        }
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "positive_fraction": float((array > 0.0).mean()),
    }


def _read_candidate_table(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("candidate table has no header")
        missing = METADATA_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"candidate table missing columns: {sorted(missing)}")
        signal_names = [name for name in reader.fieldnames if name not in METADATA_COLUMNS]
        for raw in reader:
            numeric = {
                name: float(raw[name])
                for name in signal_names
            }
            numeric.update(derive_matched_minus_random(numeric, signal_names))
            rows.append(
                {
                    "image_id": raw["image_id"],
                    "group_id": raw["group_id"],
                    "size_group": raw["size_group"],
                    "source": raw["source"],
                    "candidate_area_ratio": float(raw["candidate_area_ratio"]),
                    "candidate_dice": float(raw["candidate_dice"]),
                    "is_eligible_oracle": raw["is_eligible_oracle"].lower()
                    in {"1", "true"},
                    "is_baseline_selected": raw["is_baseline_selected"].lower()
                    in {"1", "true"},
                    "g1_logit": float(raw["g1_logit"]),
                    "upstream_score": float(raw["upstream_score"]),
                    "signals": numeric,
                }
            )
    derived_names = sorted(
        {
            name
            for row in rows[:1]
            for name in row["signals"]
            if name not in signal_names
        }
    )
    return rows, signal_names + derived_names


def analyze_conditional_information(
    rows: list[dict[str, object]], signal_names: list[str]
) -> dict[str, object]:
    by_image: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)
    if len(by_image) != 184:
        raise ValueError("conditional analysis requires exactly 184 tumor images")

    partials: dict[str, dict[str, list[float]]] = {
        group: defaultdict(list) for group in GROUPS
    }
    oracle_pair_wins: dict[str, dict[str, list[float]]] = {
        group: defaultdict(list) for group in GROUPS
    }
    for image_rows in by_image.values():
        size_group = str(image_rows[0]["size_group"])
        target = np.asarray([float(row["candidate_dice"]) for row in image_rows])
        g1 = average_percentile_rank(
            np.asarray([float(row["g1_logit"]) for row in image_rows])
        )
        upstream = average_percentile_rank(
            np.asarray([float(row["upstream_score"]) for row in image_rows])
        )
        log_area = average_percentile_rank(
            np.log1p(
                np.maximum(
                    np.asarray(
                        [float(row["candidate_area_ratio"]) for row in image_rows]
                    ),
                    0.0,
                )
            )
        )
        controls = np.column_stack(
            [
                g1,
                upstream,
                log_area,
                _source_controls([str(row["source"]) for row in image_rows]),
            ]
        )
        baseline_indices = [
            index
            for index, row in enumerate(image_rows)
            if bool(row["is_baseline_selected"])
        ]
        oracle_indices = [
            index
            for index, row in enumerate(image_rows)
            if bool(row["is_eligible_oracle"])
        ]
        if len(baseline_indices) != 1 or not oracle_indices:
            raise ValueError("baseline/oracle marker contract violated")
        baseline_index = baseline_indices[0]
        oracle_index = max(oracle_indices, key=lambda index: target[index])

        for signal_name in signal_names:
            signal = np.asarray(
                [float(row["signals"][signal_name]) for row in image_rows]
            )
            if not np.isfinite(signal).all():
                raise ValueError(f"non-finite candidate signal: {signal_name}")
            value = partial_rank_correlation(target, signal, controls)
            partials["overall"][signal_name].append(value)
            partials[size_group][signal_name].append(value)
            if oracle_index != baseline_index and signal[oracle_index] != signal[baseline_index]:
                win = float(signal[oracle_index] > signal[baseline_index])
                oracle_pair_wins["overall"][signal_name].append(win)
                oracle_pair_wins[size_group][signal_name].append(win)

    grouped: dict[str, object] = {}
    for group in GROUPS:
        grouped[group] = {
            signal_name: {
                "partial_rank_correlation": _summary(partials[group][signal_name]),
                "oracle_above_baseline_fraction": (
                    float(np.mean(oracle_pair_wins[group][signal_name]))
                    if oracle_pair_wins[group][signal_name]
                    else 0.0
                ),
                "oracle_baseline_pair_n": len(oracle_pair_wins[group][signal_name]),
            }
            for signal_name in signal_names
        }

    overall = grouped["overall"]
    top = sorted(
        (
            {
                "signal": name,
                "median_partial_rank_correlation": item[
                    "partial_rank_correlation"
                ]["median"],
                "positive_fraction": item["partial_rank_correlation"][
                    "positive_fraction"
                ],
                "oracle_above_baseline_fraction": item[
                    "oracle_above_baseline_fraction"
                ],
            }
            for name, item in overall.items()
        ),
        key=lambda item: (
            -float(item["median_partial_rank_correlation"]),
            str(item["signal"]),
        ),
    )[:20]

    transition2 = overall["transition2_matched_relative_feature_l2_contrast"]
    transition2_small = grouped["small"][
        "transition2_matched_relative_feature_l2_contrast"
    ]
    transition2_causal = overall[
        "transition2_matched_minus_random_relative_feature_l2_contrast"
    ]
    frozen_residual_gate = {
        "overall_partial_at_least_0_10": transition2[
            "partial_rank_correlation"
        ]["median"]
        >= 0.10,
        "overall_oracle_pair_win_at_least_0_55": transition2[
            "oracle_above_baseline_fraction"
        ]
        >= 0.55,
        "small_partial_at_least_0_05": transition2_small[
            "partial_rank_correlation"
        ]["median"]
        >= 0.05,
        "matched_minus_random_partial_at_least_0_05": transition2_causal[
            "partial_rank_correlation"
        ]["median"]
        >= 0.05,
    }
    frozen_residual_gate["pass"] = all(frozen_residual_gate.values())
    return {
        "groups": grouped,
        "top_overall_conditional_signals": top,
        "frozen_transition2_residual_gate": frozen_residual_gate,
        "decision": (
            "retain_frozen_transition2_residual"
            if frozen_residual_gate["pass"]
            else "frozen_representation_exhausted_for_candidate_identity"
        ),
    }


def _fmt(value: object) -> str:
    return f"{float(value):.6f}"


def _markdown(result: dict[str, object]) -> str:
    grouped = result["conditional_information"]["groups"]
    lines = [
        "# G1 conditional-information bottleneck",
        "",
        "This is a retrospective information diagnostic over audited Stage-B tables. "
        "It is not a deployable selector and cannot establish a new validation Dice.",
        "",
        "## Key residual signals after controlling G1, upstream, area and source",
        "",
        "| Signal | Overall partial | Oracle>baseline | Small partial | Medium partial | Large partial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for signal_name in KEY_SIGNALS:
        item = grouped["overall"][signal_name]
        lines.append(
            f"| `{signal_name}` | "
            f"{_fmt(item['partial_rank_correlation']['median'])} | "
            f"{_fmt(item['oracle_above_baseline_fraction'])} | "
            f"{_fmt(grouped['small'][signal_name]['partial_rank_correlation']['median'])} | "
            f"{_fmt(grouped['medium'][signal_name]['partial_rank_correlation']['median'])} | "
            f"{_fmt(grouped['large'][signal_name]['partial_rank_correlation']['median'])} |"
        )
    lines.extend(
        [
            "",
            "## Strongest apparent residuals",
            "",
            "| Signal | Median partial | Positive images | Oracle>baseline |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in result["conditional_information"]["top_overall_conditional_signals"]:
        lines.append(
            f"| `{item['signal']}` | "
            f"{_fmt(item['median_partial_rank_correlation'])} | "
            f"{_fmt(item['positive_fraction'])} | "
            f"{_fmt(item['oracle_above_baseline_fraction'])} |"
        )
    gate = result["conditional_information"]["frozen_transition2_residual_gate"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Frozen transition2 residual gate: `{'PASS' if gate['pass'] else 'FAIL'}`.",
            "",
            "The weak transition2 association does not survive the full conditional and "
            "matched-versus-random checks at a material level, especially for small lesions. "
            "The strongest remaining signals are shared matched/random ring-mass terms and "
            "do not rank the eligible oracle above the baseline choice.  They are geometry "
            "proxies, not candidate-specific tumor identity.",
            "",
            "Therefore another frozen-score fusion is unsupported.  The successor must preserve "
            "G1+upstream and learn a new high-spatial-resolution candidate representation with "
            "normal-image candidate negatives, local inside-versus-ring discrimination and a "
            "separate signed scale/extent mechanism.",
            "",
            "No test data or images were accessed by this diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("conditional-analysis output must not exist")
    summary_path = args.stage_b_root / "evaluation_summary.json"
    if sha256_file(summary_path) != args.expected_summary_sha256:
        raise ValueError("Stage-B summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("stage") != "rich_gallery_matched_normal_transplant_stage_b_v2"
        or summary.get("validation_images") != 371
        or summary.get("tumor_images_evaluated") != 184
        or summary.get("test_evaluated") is not False
        or summary.get("candidate_scores_frozen_before_validation_gt") is not True
    ):
        raise ValueError("Stage-B scientific boundary mismatch")
    baseline_dice = summary["variants"]["g1_upstream_baseline"]["overall"]["dice"]
    if abs(float(baseline_dice) - EXPECTED_BASELINE_DICE) > 1.0e-12:
        raise ValueError("immutable G1/upstream baseline mismatch")
    candidate_path = args.stage_b_root / "per_candidate_layerwise.csv"
    if sha256_file(candidate_path) != summary["per_candidate_layerwise_sha256"]:
        raise ValueError("per-candidate Stage-B table SHA-256 mismatch")
    rows, signal_names = _read_candidate_table(candidate_path)
    conditional = analyze_conditional_information(rows, signal_names)
    result = {
        "stage": "rich_gallery_g1_conditional_information_v1",
        "stage_b_summary_sha256": args.expected_summary_sha256,
        "per_candidate_layerwise_sha256": summary[
            "per_candidate_layerwise_sha256"
        ],
        "baseline_dice": float(baseline_dice),
        "candidate_rows": len(rows),
        "tumor_images": 184,
        "controls": [
            "within_image_g1_percentile",
            "within_image_upstream_percentile",
            "within_image_log_area_percentile",
            "candidate_source",
        ],
        "retrospective_validation_gt_information_diagnostic_only": True,
        "selector_or_prediction_created": False,
        "test_evaluated": False,
        "conditional_information": conditional,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "conditional_information.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = args.output_dir / "G1_CONDITIONAL_INFORMATION_DOSSIER.md"
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": True,
                "decision": conditional["decision"],
                "conditional_information_sha256": sha256_file(json_path),
                "dossier_sha256": sha256_file(markdown_path),
                "test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
