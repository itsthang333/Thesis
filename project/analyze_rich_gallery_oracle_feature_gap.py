from __future__ import annotations

"""Diagnose why the frozen G1/upstream selector misses its gallery oracle.

This is a validation-only, post-freeze diagnostic.  It consumes oracle indices
that were produced by the spatial evaluator, but it never trains, selects a new
pipeline, or opens test data.  Its purpose is to determine which observable
candidate properties differ between the frozen choice and the best available
candidate, and whether those differences have a stable direction across lesion
size groups.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from mae_reconstruction_io import sha256_file
from models.rich_gallery_g2_objective import average_percentile_rank


BASELINE_VARIANT = "g1_upstream_baseline"
SIZE_GROUPS = ("small", "medium", "large")
SCALAR_FEATURES = (
    "area",
    "border_fraction",
    "bone_inside_fraction",
    "bbox_fill",
    "center_distance",
    "compactness",
    "components",
    "prompt_inside_mean",
    "prompt_ring_contrast",
    "g1_rank",
    "upstream_rank",
    "sam_rank",
    "area_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--g1-score-root", type=Path, required=True)
    parser.add_argument("--g1-per-image", type=Path, required=True)
    parser.add_argument("--expected-g1-per-image-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--stage-b-per-image", type=Path, required=True)
    parser.add_argument("--expected-stage-b-per-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return average_percentile_rank(np.asarray(values, dtype=np.float64))


def _candidate_features(
    mask: np.ndarray,
    prompt_map: np.ndarray,
    bone_support: np.ndarray,
) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    count = int(binary.sum())
    if count == 0:
        return {
            "area": 0.0,
            "border_fraction": 0.0,
            "bone_inside_fraction": 0.0,
            "bbox_fill": 0.0,
            "center_distance": 0.0,
            "compactness": 0.0,
            "components": 0.0,
            "prompt_inside_mean": 0.0,
            "prompt_ring_contrast": 0.0,
        }

    border_width = max(1, int(round(0.10 * min(height, width))))
    border = np.zeros_like(binary)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    ys, xs = np.where(binary)
    bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    center_y = float(ys.mean() / max(1, height - 1))
    center_x = float(xs.mean() / max(1, width - 1))
    eroded = ndimage.binary_erosion(binary)
    perimeter = int(np.logical_and(binary, np.logical_not(eroded)).sum())
    ring = np.logical_and(
        ndimage.binary_dilation(binary, iterations=8), np.logical_not(binary)
    )
    prompt_inside = float(np.asarray(prompt_map, dtype=np.float64)[binary].mean())
    prompt_ring = (
        float(np.asarray(prompt_map, dtype=np.float64)[ring].mean())
        if ring.any()
        else float(np.asarray(prompt_map, dtype=np.float64).mean())
    )
    return {
        "area": float(count / binary.size),
        "border_fraction": float(np.logical_and(binary, border).sum() / count),
        "bone_inside_fraction": (
            float(np.logical_and(binary, bone_support).sum() / count)
            if np.asarray(bone_support, dtype=bool).any()
            else 0.0
        ),
        "bbox_fill": float(count / bbox_area),
        "center_distance": float(
            np.sqrt((center_y - 0.5) ** 2 + (center_x - 0.5) ** 2)
        ),
        "compactness": float(4.0 * np.pi * count / max(1.0, perimeter**2)),
        "components": float(ndimage.label(binary)[1]),
        "prompt_inside_mean": prompt_inside,
        "prompt_ring_contrast": float(prompt_inside - prompt_ring),
    }


def _delta_summary(rows: list[dict[str, object]], feature: str) -> dict[str, float]:
    values = np.asarray([float(row[f"delta_{feature}"]) for row in rows])
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "oracle_lower_fraction": float(np.mean(values < 0.0)),
        "oracle_higher_fraction": float(np.mean(values > 0.0)),
        "equal_fraction": float(np.mean(values == 0.0)),
        "direction_consistency": float(max(np.mean(values < 0.0), np.mean(values > 0.0))),
    }


def _distribution(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return {key: float(value / len(rows)) for key, value in sorted(counts.items())}


def _render_dossier(summary: dict[str, object]) -> str:
    overall = summary["feature_deltas"]["overall"]
    subgroups = summary["feature_deltas"]["subgroups"]
    source = summary["source_distributions"]
    lines = [
        "# Rich-gallery oracle feature-gap diagnostic",
        "",
        "## Scope",
        "",
        "This is a post-freeze validation diagnostic. It compares the immutable",
        "`0.5*rank(G1)+0.5*rank(upstream)` choice with the already-evaluated gallery",
        "oracle. It does not train a selector, tune a threshold, or open test data.",
        "",
        f"Tumor images: `{summary['tumor_images']}`; immutable baseline Dice: "
        f"`{summary['baseline_dice']:.9f}`; gallery oracle Dice: "
        f"`{summary['gallery_oracle_dice']:.9f}`.",
        "",
        "## Rank evidence: existing scores suppress the oracle",
        "",
        "| Signal | Oracle lower-ranked fraction | Median oracle-selected rank delta |",
        "|---|---:|---:|",
    ]
    for feature in ("g1_rank", "upstream_rank", "sam_rank"):
        item = overall[feature]
        lines.append(
            f"| `{feature}` | {item['oracle_lower_fraction']:.3f} | {item['median']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "A global weight sweep cannot recover a candidate that is simultaneously",
            "ranked below the frozen choice by the available signals. The failure is",
            "missing candidate-level tumor identity/extent evidence, not coefficient choice.",
            "",
            "## Extent direction changes with lesion size",
            "",
            "| Group | n | Median oracle-selected area | Oracle smaller | Oracle larger |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in SIZE_GROUPS:
        item = subgroups[group]["area"]
        lines.append(
            f"| `{group}` | {summary['subgroup_counts'][group]} | {item['median']:+.6f} | "
            f"{item['oracle_lower_fraction']:.3f} | {item['oracle_higher_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The required correction reverses between tiny and large lesions. Therefore",
            "raw area, dilation, consensus expansion, or one global morphology rule is",
            "structurally incapable of improving all subgroups together.",
            "",
            "## Source mismatch",
            "",
            "| Group | Selected external | Oracle external | Selected classifier | Oracle classifier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in SIZE_GROUPS:
        selected = source["subgroups"][group]["selected"]
        oracle = source["subgroups"][group]["oracle"]
        lines.append(
            f"| `{group}` | {selected.get('external_saliency', 0.0):.3f} | "
            f"{oracle.get('external_saliency', 0.0):.3f} | "
            f"{selected.get('classifier448:layercam', 0.0):.3f} | "
            f"{oracle.get('classifier448:layercam', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "The oracle increasingly shifts to external-saliency candidates as lesion size",
            "grows, while the frozen selector keeps over-selecting classifier-derived masks.",
            "A source router without a reliable lesion-scale/tumor-evidence variable cannot",
            "know when to make that switch; this explains why G2 routing did not beat the",
            "fixed fusion.",
            "",
            "## Research decision",
            "",
            "1. Keep the rich gallery and the 0.288729 baseline immutable.",
            "2. Do not run another G1/upstream/SAM/area/source-weight sweep.",
            "3. Require a new candidate-conditioned positive-evidence score whose direction",
            "   is not inherited from those three ranks and whose extent response is",
            "   conditioned by image evidence rather than raw mask area.",
            "4. The BAS-Softplus probe tests exactly whether inside-versus-background",
            "   activation can supply that missing variable. It must pass mechanics gates",
            "   before any full run and must ultimately beat actual Dice 0.288729.",
            "",
            "Validation GT-derived oracle information in this dossier is diagnostic only.",
            "It cannot authorize a learned selector or count as a deployment result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "candidate_manifest_sha256": _require_hash(
            args.candidate_manifest, args.expected_candidate_manifest_sha256
        ),
        "g1_per_image_sha256": _require_hash(
            args.g1_per_image, args.expected_g1_per_image_sha256
        ),
        "selection_manifest_sha256": _require_hash(
            args.selection_manifest, args.expected_selection_manifest_sha256
        ),
        "stage_b_per_image_sha256": _require_hash(
            args.stage_b_per_image, args.expected_stage_b_per_image_sha256
        ),
    }

    candidate_manifest = {
        Path(row["image_name"]).stem: row for row in _read_csv(args.candidate_manifest)
    }
    oracle_rows = {str(row["image_id"]): row for row in _read_jsonl(args.g1_per_image)}
    choices = {
        str(row["image_id"]): row
        for row in _read_csv(args.selection_manifest)
        if row["variant"] == BASELINE_VARIANT
    }
    metrics = {
        str(row["image_id"]): row
        for row in _read_csv(args.stage_b_per_image)
        if row["variant"] == BASELINE_VARIANT
    }
    score_paths = {
        path.stem.split("_", 1)[1]: path for path in args.g1_score_root.glob("*.npz")
    }
    ids = sorted(set(oracle_rows) & set(choices) & set(metrics))
    if len(ids) != 184:
        raise ValueError(f"expected 184 tumor images, found {len(ids)}")

    paired_rows: list[dict[str, object]] = []
    for image_id in ids:
        stem = Path(image_id).stem
        manifest_row = candidate_manifest[stem]
        candidate_path = args.candidate_root / str(manifest_row["diagnostic_path"])
        if sha256_file(candidate_path) != manifest_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload hash mismatch: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            masks = np.asarray(payload["sam_masks"], dtype=bool)
            prompt_map = np.asarray(payload["prompt_map"], dtype=np.float64)
            bone = np.asarray(payload["bone_support"], dtype=bool)
            sources = np.asarray(payload["proposal_source_ids"])
            prompt_modes = np.asarray(payload["prompt_modes"])
            upstream = np.asarray(payload["selection_scores"], dtype=np.float64)
            sam = np.asarray(payload["sam_scores"], dtype=np.float64)
            with np.load(score_paths[stem], allow_pickle=False) as score_payload:
                indices = np.asarray(score_payload["candidate_indices"], dtype=np.int64)
                g1 = np.empty(len(masks), dtype=np.float64)
                g1[indices] = np.asarray(
                    score_payload["candidate_logits"], dtype=np.float64
                )
            ranks = {
                "g1_rank": _percentile_rank(g1),
                "upstream_rank": _percentile_rank(upstream),
                "sam_rank": _percentile_rank(sam),
                "area_rank": _percentile_rank(masks.reshape(len(masks), -1).mean(axis=1)),
            }
            selected_index = int(choices[image_id]["selected_candidate_index"])
            oracle_index = int(oracle_rows[image_id]["oracle_candidate_index"])
            row: dict[str, object] = {
                "image_id": image_id,
                "group_id": metrics[image_id]["group_id"],
                "size_group": metrics[image_id]["size_group"],
                "baseline_dice": float(metrics[image_id]["dice"]),
                "oracle_dice": float(metrics[image_id]["oracle_dice"]),
                "selector_regret": float(metrics[image_id]["selector_regret"]),
                "selected_index": selected_index,
                "oracle_index": oracle_index,
                "selected_source": str(sources[selected_index]),
                "oracle_source": str(sources[oracle_index]),
                "selected_prompt_mode": str(prompt_modes[selected_index]),
                "oracle_prompt_mode": str(prompt_modes[oracle_index]),
            }
            selected_features = _candidate_features(
                masks[selected_index], prompt_map, bone
            )
            oracle_features = _candidate_features(masks[oracle_index], prompt_map, bone)
            for feature, values in ranks.items():
                selected_features[feature] = float(values[selected_index])
                oracle_features[feature] = float(values[oracle_index])
            for feature in SCALAR_FEATURES:
                selected_value = float(selected_features[feature])
                oracle_value = float(oracle_features[feature])
                row[f"selected_{feature}"] = selected_value
                row[f"oracle_{feature}"] = oracle_value
                row[f"delta_{feature}"] = oracle_value - selected_value
            paired_rows.append(row)

    fieldnames = list(paired_rows[0])
    pair_path = args.output_dir / "per_image_oracle_feature_gap.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_rows)

    feature_deltas = {
        "overall": {
            feature: _delta_summary(paired_rows, feature) for feature in SCALAR_FEATURES
        },
        "subgroups": {
            group: {
                feature: _delta_summary(
                    [row for row in paired_rows if row["size_group"] == group], feature
                )
                for feature in SCALAR_FEATURES
            }
            for group in SIZE_GROUPS
        },
    }
    source_distributions = {
        "overall": {
            "selected": _distribution(paired_rows, "selected_source"),
            "oracle": _distribution(paired_rows, "oracle_source"),
        },
        "subgroups": {
            group: {
                "selected": _distribution(
                    [row for row in paired_rows if row["size_group"] == group],
                    "selected_source",
                ),
                "oracle": _distribution(
                    [row for row in paired_rows if row["size_group"] == group],
                    "oracle_source",
                ),
            }
            for group in SIZE_GROUPS
        },
    }
    summary: dict[str, object] = {
        "stage": "rich_gallery_oracle_feature_gap_diagnostic_v1",
        "tumor_images": len(paired_rows),
        "subgroup_counts": {
            group: sum(row["size_group"] == group for row in paired_rows)
            for group in SIZE_GROUPS
        },
        "baseline_dice": float(
            np.mean([float(row["baseline_dice"]) for row in paired_rows])
        ),
        "gallery_oracle_dice": float(
            np.mean([float(row["oracle_dice"]) for row in paired_rows])
        ),
        "feature_deltas": feature_deltas,
        "source_distributions": source_distributions,
        "input_hashes": hashes,
        "per_image_sha256": sha256_file(pair_path),
        "validation_gt_derived_oracle_used_for_diagnosis": True,
        "training_authorized": False,
        "test_opened": False,
    }
    dossier_path = args.output_dir / "ORACLE_FEATURE_GAP_DOSSIER.md"
    dossier_path.write_text(_render_dossier(summary), encoding="utf-8")
    summary["dossier_sha256"] = sha256_file(dossier_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
