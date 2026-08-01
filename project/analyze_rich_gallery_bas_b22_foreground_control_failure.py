from __future__ import annotations

"""Explain the audited BAS-B2.2 foreground-control failure.

The analysis is label-safe.  It consumes only image labels, the immutable
activation maps, candidate-geometry correlations and training logs.  It never
opens validation polygons or any test artifact.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_rich_gallery_bas_b21_softplus_failure import (
    activation_collapse_features,
)
from mae_reconstruction_io import sha256_file


FOREGROUND_CONTROL_WEIGHT = 1.5
AREA_WEIGHT = 1.2
REFERENCE_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _rank(values: np.ndarray) -> np.ndarray:
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


def _spearman(left: list[float], right: list[float]) -> float:
    x = _rank(np.asarray(left, dtype=np.float64))
    y = _rank(np.asarray(right, dtype=np.float64))
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spatial_shape_features(activation: np.ndarray) -> dict[str, float]:
    values = np.asarray(activation, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("activation must be a finite 2-D map")
    height, width = values.shape
    border_h = max(1, int(np.ceil(0.10 * height)))
    border_w = max(1, int(np.ceil(0.10 * width)))
    border = np.zeros_like(values, dtype=bool)
    border[:border_h, :] = True
    border[-border_h:, :] = True
    border[:, :border_w] = True
    border[:, -border_w:] = True
    border_mean = float(values[border].mean())
    interior_mean = float(values[~border].mean())
    return {
        "fraction_ge_0_90": float(np.mean(values >= 0.90)),
        "fraction_ge_0_99": float(np.mean(values >= 0.99)),
        "fraction_le_0_10": float(np.mean(values <= 0.10)),
        "bimodal_extreme_fraction": float(
            np.mean(np.logical_or(values <= 0.10, values >= 0.90))
        ),
        "border_mean": border_mean,
        "interior_mean": interior_mean,
        "border_minus_interior": border_mean - interior_mean,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    keys = (
        "activation_max",
        "activation_mean",
        "activation_std",
        "sigmoid_gradient_max",
        "effective_support_fraction",
        "top_1_percent_mass_fraction",
        "argmax_border",
        "fraction_ge_0_90",
        "fraction_ge_0_99",
        "fraction_le_0_10",
        "bimodal_extreme_fraction",
        "border_minus_interior",
        "bas_area_spearman",
    )
    result: dict[str, float | int] = {"n": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_q10"] = float(np.quantile(values, 0.10))
        result[f"{key}_q90"] = float(np.quantile(values, 0.90))
    return result


def _render_dossier(summary: dict[str, object]) -> str:
    training = summary["training"]
    tumor = summary["activation_shape"]["tumor"]
    normal = summary["activation_shape"]["normal"]
    dynamics = summary["training_dynamics"]
    proof = summary["objective_proof"]
    association = summary["associations"]
    return f"""# BAS-B2.2 foreground-control failure dossier

## Audited outcome

The matched five-epoch probe changed only the B2.1 localization objective.
Independent audit verified all 371 maps and predictions, checkpoint and source
hashes, and no validation polygons or test access.

- final train accuracy: `{training['final_accuracy']:.6f}`;
- validation image AUROC: `{training['validation_auroc']:.6f}`;
- final foreground CE: `{training['final_foreground_ce']:.6f}`;
- final full-image CE: `{training['final_full_ce']:.6f}`;
- mechanics gate: **FAIL**, solely because full-image CE exceeds `0.69`.

B2.2 therefore repairs the exact zero-map optimum of B2.1, but it does not
produce sufficiently identified tumor evidence.

## Exact objective geometry

For a fixed nonnegative target-class map `C`, full score
`S = mean(C)`, localization map `M`, and foreground score
`S_fg = mean(C*M)`, the B2.2 localization term is

`L(M) = 1.5 * (0.5 - S_fg / stopgrad(S)) + 1.2 * mean(M)`.

The reference ratio `0.5` is an additive constant and has **no effect on the
gradient or optimum**.  For each cell `i`,

`dL/dM_i = (1.2 - 1.5 * C_i/S) / N`.

Hence, ignoring the sigmoid parameterization, the box-constrained optimum is

`M_i = 1  iff  C_i / mean(C) > {proof['relative_class_evidence_threshold']:.3f}`.

This condition says only that a location has above-average target-class
evidence. It never says that the location is tumor. Broad anatomy, acquisition
style, border structure, or any image-level shortcut can satisfy it. The
foreground CE reinforces retention of enough class evidence and can expand the
map further; it does not provide pixel-negative supervision.

The localization head also gathers only the ground-truth image-label channel:
normal images train the normal map and tumor images train the tumor map. Thus
the tumor channel receives no direct dense-negative constraint on 1,493 normal
images. A high image AUROC is consequently compatible with a non-specific
tumor localization map.

## Tensor evidence: zero collapse became diffuse saturation

Across 184 tumor validation images:

- median activation mean: `{tumor['activation_mean_median']:.6f}`;
- median fraction of cells >=0.90: `{tumor['fraction_ge_0_90_median']:.6f}`;
- median fraction of cells >=0.99: `{tumor['fraction_ge_0_99_median']:.6f}`;
- median effective support: `{tumor['effective_support_fraction_median']:.6f}`;
- median top-1%-mass: `{tumor['top_1_percent_mass_fraction_median']:.6f}`;
- argmax on the outer 10% border: `{tumor['argmax_border_mean']:.6f}`;
- mean border-minus-interior activation: `{tumor['border_minus_interior_mean']:.6f}`.

The map is no longer empty or a single numerical spike. It is a broad,
strongly bimodal anatomy map: `{tumor['bimodal_extreme_fraction_median']:.3f}`
of cells are <=0.1 or >=0.9. Normal-image tumor maps are also broad (median
mean `{normal['activation_mean_median']:.6f}`), confirming that the tumor
channel is not a dense tumor/background discriminator.

## Why candidate scoring remains an area proxy

The proposed BAS candidate score min-max normalizes each map and computes
coverage/purity harmonic overlap with each candidate. With a broad centered
activation field, coverage grows with mask area while purity mostly measures
whether the mask lies on common anatomy. Observed tumor-image BAS-score versus
candidate-area Spearman is:

- mean `{tumor['bas_area_spearman_mean']:.6f}`;
- median `{tumor['bas_area_spearman_median']:.6f}`;
- fraction above 0.90 `{association['tumor_area_spearman_fraction_above_0_9']:.6f}`.

The per-image map mean itself correlates with the within-bag area dependence
at Spearman `{association['tumor_activation_mean_vs_area_spearman']:.6f}`.
This is the same failure family exposed by G2/consensus/metadata selectors:
they rank anatomy, mask size, or source regularity rather than candidate-
conditioned tumor evidence.

## Optimization evidence

Full CE reached its minimum `{dynamics['minimum_full_ce']:.6f}` at epoch
`{dynamics['minimum_full_ce_epoch']}`, then worsened to
`{training['final_full_ce']:.6f}` while accuracy increased to
`{training['final_accuracy']:.6f}`. This divergence indicates increasingly
confident residual errors/miscalibration and competition between the full and
foreground objectives. More epochs or choosing an epoch by validation labels
would not repair the identifiability defect above.

## Consequence for the 0.288729 baseline

The rich gallery remains strong (oracle Dice 0.528298), but 70.29% of current
regret is within-selected-source ranking. B2.2 does not add the missing
candidate-conditioned tumor signal; it primarily re-expresses candidate area
and common anatomy. It must not replace G1 or be promoted on mechanics metrics.

One post-freeze exploratory fusion/evaluation is still academically valid as
a failure diagnostic because its formula was frozen before polygons and no
weight/threshold/variant is selected using GT. Its Dice must be reported as
exploratory/non-promotable. No B2.2 epoch, weight, threshold, or seed sweep is
authorized.

## Research decision

1. Retire BAS foreground-control as a selector family after the single frozen
   exploratory Dice decomposition.
2. Preserve G1+upstream fixed rank fusion at Dice `0.2887294867`.
3. The successor must add candidate-conditioned evidence that is trained with
   tumor-channel dense negatives on normal images or an equivalent causal
   intervention; it must not be another monotone area/source/anatomy score.
4. Before a full run, require a matched cheap diagnostic to demonstrate lower
   candidate-area dependence and material within-source oracle-rank gain while
   preserving the fixed gallery and G1/upstream baseline.
5. Keep test locked until a frozen validation pipeline satisfies the subgroup
   contract.
"""


def main() -> None:
    args = parse_args()
    args.analysis_output.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.independent_audit) != args.expected_audit_sha256:
        raise ValueError("independent audit hash mismatch")
    audit = json.loads(args.independent_audit.read_text(encoding="utf-8"))
    if (
        audit.get("pass") is not True
        or audit.get("activation_maps_verified") != 371
        or audit.get("mechanics_gate_pass") is not False
        or audit.get("test_images_read") != 0
        or audit.get("validation_gt_read") is not False
    ):
        raise ValueError("B2.2 independent audit is not an audited mechanics failure")

    probe_path = args.output_root / "probe_summary.json"
    manifest_path = args.output_root / "activation_manifest.csv"
    history_path = args.output_root / "training_history.csv"
    predictions_path = args.output_root / "validation_predictions.csv"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    manifest = _read_csv(manifest_path)
    history = _read_csv(history_path)
    predictions = _read_csv(predictions_path)
    if len(manifest) != 371 or len(predictions) != 371 or len(history) != 5:
        raise ValueError("B2.2 output cohort/history mismatch")
    if probe.get("stage") != "rich_gallery_bas_b22_foreground_control_mechanics_probe_v1":
        raise ValueError("unexpected B2.2 probe stage")

    rows: list[dict[str, object]] = []
    for row in manifest:
        map_path = args.output_root / row["activation_path"]
        if sha256_file(map_path) != row["activation_sha256"]:
            raise ValueError(f"activation hash mismatch: {row['image_id']}")
        activation = np.load(map_path, allow_pickle=False)
        rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": int(row["tumor"]),
                "candidate_count": int(row["candidate_count"]),
                "bas_area_spearman": float(row["bas_area_spearman"]),
                **activation_collapse_features(activation),
                **spatial_shape_features(activation),
            }
        )

    per_image_path = args.analysis_output / "per_image_activation_shape.csv"
    per_image_sha = _write_csv(per_image_path, rows)
    tumor_rows = [row for row in rows if int(row["tumor"]) == 1]
    normal_rows = [row for row in rows if int(row["tumor"]) == 0]
    if len(tumor_rows) != 184 or len(normal_rows) != 187:
        raise ValueError("B2.2 validation label cohort mismatch")

    full_ce = [float(row["full_ce"]) for row in history]
    minimum_epoch = int(np.argmin(full_ce)) + 1
    final = history[-1]
    objective_threshold = AREA_WEIGHT / FOREGROUND_CONTROL_WEIGHT
    summary: dict[str, object] = {
        "stage": "rich_gallery_bas_b22_foreground_control_failure_analysis_v1",
        "training": {
            "final_accuracy": float(final["accuracy"]),
            "final_full_ce": float(final["full_ce"]),
            "final_foreground_ce": float(final["foreground_ce"]),
            "final_foreground_control_area": float(final["fgc_area"]),
            "validation_auroc": float(probe["validation_diagnostics"]["auroc"]),
        },
        "training_dynamics": {
            "minimum_full_ce": float(min(full_ce)),
            "minimum_full_ce_epoch": minimum_epoch,
            "final_minus_minimum_full_ce": float(full_ce[-1] - min(full_ce)),
            "accuracy_epoch_4_to_5": float(
                float(history[-1]["accuracy"]) - float(history[-2]["accuracy"])
            ),
        },
        "activation_shape": {
            "all": summarize(rows),
            "normal": summarize(normal_rows),
            "tumor": summarize(tumor_rows),
        },
        "associations": {
            "tumor_activation_mean_vs_candidate_count_spearman": _spearman(
                [float(row["activation_mean"]) for row in tumor_rows],
                [float(row["candidate_count"]) for row in tumor_rows],
            ),
            "tumor_activation_mean_vs_area_spearman": _spearman(
                [float(row["activation_mean"]) for row in tumor_rows],
                [float(row["bas_area_spearman"]) for row in tumor_rows],
            ),
            "tumor_area_spearman_fraction_above_0_9": float(
                np.mean([float(row["bas_area_spearman"]) > 0.90 for row in tumor_rows])
            ),
        },
        "objective_proof": {
            "loss": "lambda_f*(R-S_fg/stopgrad(S))+lambda_a*mean(M)",
            "cell_gradient": "(lambda_a-lambda_f*C_i/S)/N",
            "foreground_control_weight": FOREGROUND_CONTROL_WEIGHT,
            "area_weight": AREA_WEIGHT,
            "reference_ratio": REFERENCE_RATIO,
            "reference_ratio_changes_gradient": False,
            "relative_class_evidence_threshold": objective_threshold,
            "fixed_class_map_box_optimum": "M_i=1 iff C_i/mean(C)>lambda_a/lambda_f",
            "tumor_channel_dense_negative_supervision_on_normal_images": False,
        },
        "decision": {
            "full_training_authorized": False,
            "bas_foreground_control_promotable": False,
            "single_postfreeze_exploratory_dice_diagnostic_authorized": True,
            "weight_epoch_threshold_seed_sweep_authorized": False,
            "baseline_dice": 0.28872948670665205,
        },
        "input_hashes": {
            "independent_audit_sha256": args.expected_audit_sha256,
            "probe_summary_sha256": sha256_file(probe_path),
            "activation_manifest_sha256": sha256_file(manifest_path),
            "training_history_sha256": sha256_file(history_path),
            "validation_predictions_sha256": sha256_file(predictions_path),
        },
        "per_image_sha256": per_image_sha,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    dossier_path = args.analysis_output / "BAS_B22_FOREGROUND_CONTROL_FAILURE_DOSSIER.md"
    dossier_path.write_text(_render_dossier(summary), encoding="utf-8")
    summary["dossier_sha256"] = sha256_file(dossier_path)
    summary_path = args.analysis_output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
