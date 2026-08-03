from __future__ import annotations

"""Post-freeze source-subset ablation for G1 fixed percentile-rank fusion.

Selections for every source subset are frozen before validation polygons are
opened.  This is a validation-only diagnostic, not a promoted selector.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from project.datasets.smile_reference import sha256_file
from project.models.smile_local_evidence import average_percentile_rank


SOURCES = ("layercam320", "classifier448", "external_saliency")
BASELINE_DICE = 0.28872948670665205
BASELINE_SUBGROUP = {
    "small": 0.15772329637374455,
    "medium": 0.4352293348209193,
    "large": 0.38687353265476676,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--g1-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown proposal source: {value!r}")


def subset_name(subset: tuple[str, ...]) -> str:
    return "+".join(subset)


def stable_select(
    score: np.ndarray,
    g1: np.ndarray,
    eligible: np.ndarray,
) -> int:
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return -1
    return int(max(indices.tolist(), key=lambda i: (float(score[i]), float(g1[i]), -i)))


def rank_fusion_subset(
    g1: np.ndarray,
    upstream: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    result = np.full(len(g1), -np.inf, dtype=np.float64)
    if len(indices):
        result[indices] = 0.5 * (
            average_percentile_rank(g1[indices])
            + average_percentile_rank(upstream[indices])
        )
    return result


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / max(1, denominator))


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.logical_and(prediction, target).sum()
        / max(1, np.logical_or(prediction, target).sum())
    )


def size_group(area: float) -> str:
    return "small" if area < 0.01 else "medium" if area < 0.05 else "large"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def paired_group_bootstrap(
    rows: list[dict[str, object]],
    variant: str,
    subgroup: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    indexed = {(str(row["subset"]), str(row["image_id"])): row for row in rows}
    selected = [
        row
        for row in rows
        if row["subset"] == variant
        and (subgroup == "overall" or row["size_group"] == subgroup)
    ]
    by_group: dict[str, list[float]] = defaultdict(list)
    full = subset_name(SOURCES)
    for row in selected:
        baseline = indexed[(full, str(row["image_id"]))]
        by_group[str(row["group_id"])].append(
            float(row["dice"]) - float(baseline["dice"])
        )
    values = np.asarray([np.mean(by_group[group]) for group in sorted(by_group)])
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(replicates, len(values)))
    ].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != args.split_sha256:
        raise ValueError("split SHA-256 mismatch")
    freeze_path = args.g1_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.g1_freeze_sha256:
        raise ValueError("G1 freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 freeze contract mismatch")

    selection_rows = [
        row
        for row in read_csv(args.g1_root / "stage_a_selection_manifest.csv")
        if row["variant"] == "g1_frozen__rank_fusion"
    ]
    if len(selection_rows) != 371:
        raise ValueError("G1 validation selection cohort mismatch")
    candidate_manifest_path = args.candidate_root / "candidate_diagnostics_manifest.csv"
    if sha256_file(candidate_manifest_path) != args.candidate_manifest_sha256:
        raise ValueError("candidate manifest SHA-256 mismatch")
    candidate_rows = {
        Path(row.get("image_id") or row["image_name"]).stem: row
        for row in read_csv(candidate_manifest_path)
    }

    subsets = [
        subset
        for length in range(1, len(SOURCES) + 1)
        for subset in itertools.combinations(SOURCES, length)
    ]
    choices: dict[str, dict[str, dict[str, object]]] = {}
    eligible_by_image: dict[str, dict[str, np.ndarray]] = {}
    candidate_indices_by_image: dict[str, np.ndarray] = {}
    sources_by_image: dict[str, np.ndarray] = {}
    for row in selection_rows:
        image_id = row["image_id"]
        score_path = args.g1_root / row["score_path"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"G1 score payload changed: {image_id}")
        with np.load(score_path, allow_pickle=False) as score:
            candidate_indices = score["candidate_indices"].astype(np.int64)
            g1 = score["g1_frozen_candidate_logits"].astype(np.float64)
            upstream = score["upstream_scores"].astype(np.float64)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            all_sources = payload["proposal_source_ids"].astype(str)
        sources = np.asarray(
            [canonical_source(value) for value in all_sources[candidate_indices]]
        )
        choices[image_id] = {}
        eligible_by_image[image_id] = {}
        candidate_indices_by_image[image_id] = candidate_indices
        sources_by_image[image_id] = sources
        for subset in subsets:
            name = subset_name(subset)
            eligible = np.isin(sources, subset)
            fused = rank_fusion_subset(g1, upstream, eligible)
            local = stable_select(fused, g1, eligible)
            eligible_by_image[image_id][name] = eligible
            choices[image_id][name] = {
                "selected_local_index": local,
                "selected_candidate_index": (
                    int(candidate_indices[local]) if local >= 0 else -1
                ),
                "selected_source": str(sources[local]) if local >= 0 else "none",
                "candidate_count": int(eligible.sum()),
            }
        full_name = subset_name(SOURCES)
        if int(choices[image_id][full_name]["selected_candidate_index"]) != int(
            row["selected_candidate_index"]
        ):
            raise ValueError(f"full-gallery baseline does not reproduce: {image_id}")

    args.output_dir.mkdir(parents=True)
    choice_freeze = {
        "stage": "rich_gallery_source_subset_choices_v1",
        "analysis_role": "post_freeze_validation_diagnostic_not_promotion",
        "subsets": [list(subset) for subset in subsets],
        "validation_images": 371,
        "g1_freeze_sha256": args.g1_freeze_sha256,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "split_sha256": args.split_sha256,
        "choices": choices,
        "choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    choice_path = args.output_dir / "choice_freeze.json"
    choice_path.write_text(
        json.dumps(choice_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Sole spatial-GT boundary: all choices above are immutable first.
    from project.datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    row_by_image = {row["image_id"]: row for row in selection_rows}
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, target_tensor, image_id_value = dataset[index]
        image_id = str(image_id_value)
        row = row_by_image[image_id]
        if row["tumor"] != "1":
            continue
        target = target_tensor[0].numpy() > 0.5
        group = size_group(float(target.mean()))
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)[
                candidate_indices_by_image[image_id]
            ]
        quality = np.asarray([dice(mask, target) for mask in proposals])
        full_count = len(proposals)
        for subset in subsets:
            name = subset_name(subset)
            choice = choices[image_id][name]
            local = int(choice["selected_local_index"])
            eligible = eligible_by_image[image_id][name]
            if local < 0:
                prediction = np.zeros_like(target)
                selected_dice = 0.0
                selected_iou = 0.0
                miss = 1
                area_ratio = 0.0
                selected_source = "none"
            else:
                prediction = proposals[local]
                selected_dice = float(quality[local])
                selected_iou = iou(prediction, target)
                intersection = int(np.logical_and(prediction, target).sum())
                miss = int(intersection == 0)
                area_ratio = float(prediction.sum() / max(1, target.sum()))
                selected_source = str(sources_by_image[image_id][local])
            per_image.append(
                {
                    "subset": name,
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "size_group": group,
                    "dice": selected_dice,
                    "iou": selected_iou,
                    "complete_miss": miss,
                    "selected_gt_area_ratio": area_ratio,
                    "selected_source": selected_source,
                    "subset_oracle_dice": (
                        float(quality[eligible].max()) if eligible.any() else 0.0
                    ),
                    "candidate_count": int(eligible.sum()),
                    "full_candidate_count": full_count,
                }
            )
    counts = Counter(row["size_group"] for row in per_image if row["subset"] == subset_name(SOURCES))
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"subgroup counts changed: {counts}")
    per_image_sha = write_csv(args.output_dir / "per_image.csv", per_image)

    summary: dict[str, object] = {}
    full_name = subset_name(SOURCES)
    for offset, subset in enumerate(subsets):
        name = subset_name(subset)
        selected = [row for row in per_image if row["subset"] == name]
        metrics: dict[str, object] = {}
        for group in ("overall", "small", "medium", "large"):
            current = selected if group == "overall" else [
                row for row in selected if row["size_group"] == group
            ]
            metrics[group] = {
                "n": len(current),
                "dice": float(np.mean([row["dice"] for row in current])),
                "iou": float(np.mean([row["iou"] for row in current])),
                "oracle_dice": float(
                    np.mean([row["subset_oracle_dice"] for row in current])
                ),
                "complete_misses": int(sum(row["complete_miss"] for row in current)),
                "median_selected_gt_area_ratio": float(
                    np.median([row["selected_gt_area_ratio"] for row in current])
                ),
            }
        candidate_counts = [int(row["candidate_count"]) for row in selected]
        full_counts = [int(row["full_candidate_count"]) for row in selected]
        source_counts = Counter(str(row["selected_source"]) for row in selected)
        summary[name] = {
            "sources": list(subset),
            "valid_for_all_371_images": all(
                int(choices[image_id][name]["candidate_count"]) > 0
                for image_id in choices
            ),
            "metrics": metrics,
            "mean_candidates": float(np.mean(candidate_counts)),
            "median_candidates": float(np.median(candidate_counts)),
            "candidate_reduction_fraction": float(
                1.0 - np.sum(candidate_counts) / np.sum(full_counts)
            ),
            "selected_source_counts": dict(source_counts),
            "paired_bootstrap_vs_full": {
                group: paired_group_bootstrap(
                    per_image,
                    name,
                    group,
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed + offset * 10 + group_index,
                )
                for group_index, group in enumerate(("overall", "small", "medium", "large"))
            },
        }
    baseline = summary[full_name]["metrics"]
    if abs(float(baseline["overall"]["dice"]) - BASELINE_DICE) > 1.0e-10:
        raise RuntimeError(f"baseline does not reproduce: {baseline}")
    decisions: dict[str, object] = {}
    for name, result in summary.items():
        metrics = result["metrics"]
        decisions[name] = {
            "overall_not_lower": metrics["overall"]["dice"] >= BASELINE_DICE,
            "small_not_lower": metrics["small"]["dice"] >= BASELINE_SUBGROUP["small"],
            "medium_not_lower": metrics["medium"]["dice"] >= BASELINE_SUBGROUP["medium"],
            "large_not_lower": metrics["large"]["dice"] >= BASELINE_SUBGROUP["large"],
            "oracle_drop_le_0_005": (
                baseline["overall"]["oracle_dice"]
                - metrics["overall"]["oracle_dice"]
                <= 0.005
            ),
            "candidate_count_reduced": result["candidate_reduction_fraction"] > 0.0,
            "all_primary_no_drop": all(
                metrics[group]["dice"] >= (
                    BASELINE_DICE if group == "overall" else BASELINE_SUBGROUP[group]
                )
                for group in ("overall", "small", "medium", "large")
            ),
        }
    report = {
        "stage": "rich_gallery_source_subset_ablation_v1",
        "analysis_role": "post_freeze_validation_diagnostic_not_promotion",
        "baseline_formula": "0.5*percentile_rank(G1)+0.5*percentile_rank(upstream)",
        "baseline_subset": full_name,
        "cohort": {"validation": 371, "tumor": 184, **dict(counts)},
        "summary": summary,
        "decisions": decisions,
        "choice_freeze_sha256": sha256_file(choice_path),
        "per_image_sha256": per_image_sha,
        "validation_gt_opened_after_choice_freeze": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
