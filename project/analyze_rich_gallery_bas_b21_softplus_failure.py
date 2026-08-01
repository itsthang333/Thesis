from __future__ import annotations

"""Explain the audited BAS-B2.1 Softplus localization collapse.

The analysis is label-safe: it consumes only image labels, frozen activation
maps and training logs.  It never opens validation polygons or test data.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mae_reconstruction_io import sha256_file


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


def activation_collapse_features(activation: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(activation, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("activation must be a finite nonnegative 2-D map")
    flat = values.reshape(-1)
    count = len(flat)
    total = float(flat.sum())
    top_count = max(1, int(np.ceil(0.01 * count)))
    ordered = np.sort(flat)[::-1]
    maximum = float(flat.max())
    clipped = float(np.clip(maximum, 1.0e-300, 1.0 - 1.0e-12))
    sigmoid_logit_max = float(np.log(clipped / (1.0 - clipped)))
    effective_pixels = float(total**2 / max(1.0e-300, float(np.square(flat).sum())))
    argmax = np.unravel_index(int(np.argmax(values)), values.shape)
    border_width = max(1, int(round(0.10 * min(values.shape))))
    argmax_border = int(
        argmax[0] < border_width
        or argmax[0] >= values.shape[0] - border_width
        or argmax[1] < border_width
        or argmax[1] >= values.shape[1] - border_width
    )
    return {
        "activation_min": float(flat.min()),
        "activation_max": maximum,
        "activation_mean": float(flat.mean()),
        "activation_std": float(flat.std()),
        "sigmoid_logit_max": sigmoid_logit_max,
        "sigmoid_gradient_max": float(maximum * (1.0 - maximum)),
        "effective_support_fraction": float(effective_pixels / count),
        "top_1_percent_mass_fraction": float(
            ordered[:top_count].sum() / max(1.0e-300, total)
        ),
        "argmax_border": argmax_border,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n": len(rows)}
    for key in (
        "activation_max",
        "activation_mean",
        "activation_std",
        "sigmoid_logit_max",
        "sigmoid_gradient_max",
        "effective_support_fraction",
        "top_1_percent_mass_fraction",
        "argmax_border",
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_q05"] = float(np.quantile(values, 0.05))
        result[f"{key}_q95"] = float(np.quantile(values, 0.95))
    return result


def _render_dossier(summary: dict[str, object]) -> str:
    history = summary["training"]
    tumor = summary["activation_collapse"]["tumor"]
    return f"""# BAS-B2.1 Softplus failure dossier

## Audited outcome

The five-epoch matched probe changed only the terminal classifier-map activation
from ReLU to Softplus. Independent output audit passed for all 371 validation
images; validation polygons and test were never opened.

- final train accuracy: `{history['final_accuracy']:.6f}`;
- validation image AUROC: `{history['validation_auroc']:.6f}`;
- final full-image CE: `{history['final_full_ce']:.6f}`;
- final foreground-guidance CE: `{history['final_foreground_ce']:.9f}`;
- mechanics gate: **FAIL** (`activation_range` and `nondegenerate_fraction`).

Softplus repairs the dead classifier but does not repair spatial localization.

## Tensor-level localization collapse

Across 184 tumor images:

- mean maximum localization probability: `{tumor['activation_max_mean']:.3e}`;
- mean implied maximum sigmoid preactivation: `{tumor['sigmoid_logit_max_mean']:.3f}`;
- mean maximum sigmoid derivative: `{tumor['sigmoid_gradient_max_mean']:.3e}`;
- median effective support: `{tumor['effective_support_fraction_median']:.6f}` of the 56x56 grid;
- median mass inside the largest 1% of cells: `{tumor['top_1_percent_mass_fraction_median']:.6f}`;
- argmax at the outer 10% border: `{tumor['argmax_border_mean']:.3f}`.

The map is not a weak tumor map. It is a sigmoid-saturated near-zero map plus a
padding-border numerical spike. At preactivation about -15.5, the sigmoid
gradient is attenuated by roughly four to seven million times; continuing epochs
cannot plausibly recover spatial structure.

## Exact objective loophole

Let `M` be the localization map, `S` the target-class full-image activation and
`S_bg(M)` the activation after erasing `M`. The transferred loss is

`L_BAS = gate(S_bg/S) + 1.2*mean(M)`,

where the official implementation sets the ratio to zero whenever `S_bg >= S`.
At `M=0`, the erased and full feature maps are identical, so `S_bg(0)=S`. The
hard gate therefore returns zero and the area term is also zero:

`L_BAS(M=0)=0`.

Thus the empty map is an exact global minimizer of the localization objective.
The area term initially pushes sigmoid logits downward; once saturated, every
remaining localization gradient is multiplied by `M(1-M)`. Meanwhile the
full-image classifier can independently reduce CE and reach AUROC 0.743. The
fixed foreground CE at `log(2)=0.69314718` confirms that no class information
survives through the near-zero map.

## Cross-experiment meaning

B2 failed at the classifier head (`all-zero class logits`, tumor map near one).
B2.1 fixes that exact defect and exposes the next independent defect: the
hard-gated background-ratio plus area loss admits a zero-map optimum. Neither
failure tests useful BAS candidate evidence, and neither score may be fused into
the 0.288729 baseline.

## Research decision

1. Retire the hard-gated background-ratio objective; do not extend epochs,
   sweep area weight, threshold the numerical spikes, or run spatial GT.
2. A successor is allowed only if its foreground objective has nonzero,
   spatially selective gradient at `M=0` and includes a border/support gate.
3. The strongest supported correction is a continuous foreground-control ratio
   `R - S_fg/S` plus the area constraint. Its derivative with respect to a map
   cell is proportional to `lambda_area - lambda_fgc*C_i/S`; high relative
   class-evidence cells are pushed up while background cells are pushed down.
4. This correction must first pass another bounded label-safe mechanics probe;
   only then may it produce candidate scores and actual Dice against 0.288729.
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
        raise ValueError("B2.1 independent audit is not an audited mechanics failure")

    probe_summary_path = args.output_root / "probe_summary.json"
    manifest_path = args.output_root / "activation_manifest.csv"
    history_path = args.output_root / "training_history.csv"
    prediction_path = args.output_root / "validation_predictions.csv"
    probe = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    manifest = _read_csv(manifest_path)
    history = _read_csv(history_path)
    predictions = _read_csv(prediction_path)
    if len(manifest) != 371 or len(predictions) != 371 or len(history) != 5:
        raise ValueError("B2.1 output cohort/history mismatch")

    rows: list[dict[str, object]] = []
    for row in manifest:
        map_path = args.output_root / row["activation_path"]
        if sha256_file(map_path) != row["activation_sha256"]:
            raise ValueError(f"activation hash mismatch: {row['image_id']}")
        features = activation_collapse_features(np.load(map_path, allow_pickle=False))
        rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": int(row["tumor"]),
                "bas_area_spearman": float(row["bas_area_spearman"]),
                **features,
            }
        )

    per_image_path = args.analysis_output / "per_image_activation_collapse.csv"
    per_image_sha = _write_csv(per_image_path, rows)
    final = history[-1]
    summary: dict[str, object] = {
        "stage": "rich_gallery_bas_b21_softplus_failure_analysis_v1",
        "training": {
            "final_accuracy": float(final["accuracy"]),
            "final_full_ce": float(final["full_ce"]),
            "final_foreground_ce": float(final["foreground_ce"]),
            "final_bas": float(final["bas"]),
            "validation_auroc": float(probe["validation_diagnostics"]["auroc"]),
        },
        "activation_collapse": {
            "all": summarize(rows),
            "normal": summarize([row for row in rows if int(row["tumor"]) == 0]),
            "tumor": summarize([row for row in rows if int(row["tumor"]) == 1]),
        },
        "objective_proof": {
            "empty_map_background_equals_full": True,
            "hard_gate_value_when_background_ge_full": 0.0,
            "empty_map_area": 0.0,
            "empty_map_bas_loss": 0.0,
            "sigmoid_gradient_factor": "M*(1-M)",
            "foreground_ce_chance": float(np.log(2.0)),
        },
        "decision": {
            "full_training_authorized": False,
            "hard_gated_background_ratio_retired": True,
            "next_probe": "continuous_foreground_control_ratio",
        },
        "input_hashes": {
            "independent_audit_sha256": args.expected_audit_sha256,
            "probe_summary_sha256": sha256_file(probe_summary_path),
            "activation_manifest_sha256": sha256_file(manifest_path),
            "training_history_sha256": sha256_file(history_path),
            "validation_predictions_sha256": sha256_file(prediction_path),
        },
        "per_image_sha256": per_image_sha,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    dossier_path = args.analysis_output / "BAS_B21_SOFTPLUS_FAILURE_DOSSIER.md"
    dossier_path.write_text(_render_dossier(summary), encoding="utf-8")
    summary["dossier_sha256"] = sha256_file(dossier_path)
    summary_path = args.analysis_output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
