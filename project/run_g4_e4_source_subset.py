from __future__ import annotations

"""G4 E4 source-subset ablation on the exact final ViT-B gallery.

All seven non-empty subsets are selected from frozen G1/upstream scores before
the validation polygons are opened.  The script reports actual mask Dice,
oracle/recall, candidate counts and measured offline replay cost.  It does not
retrain G1 or regenerate SAM candidates.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from project.datasets.factory import build_segmentation_dataset
from project.final_selector import average_percentile_rank


SOURCES = ("layercam320", "classifier448", "external_saliency")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256(path)


def canonical_source(value: object) -> str:
    lowered = str(value).casefold()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown proposal source: {value!r}")


def subset_name(subset: tuple[str, ...]) -> str:
    return "+".join(subset)


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


def stable_select(score: np.ndarray, g1: np.ndarray, eligible: np.ndarray) -> int:
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return -1
    return int(max(indices.tolist(), key=lambda i: (float(score[i]), float(g1[i]), -i)))


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
    full = subset_name(SOURCES)
    by_group: dict[str, list[float]] = defaultdict(list)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--expected-arm-summary-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--expected-score-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("split SHA-256 mismatch")
    summary_path = args.arm_root / "summary.json"
    if sha256(summary_path) != args.expected_arm_summary_sha256:
        raise ValueError("arm summary SHA-256 mismatch")
    arm_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        arm_summary.get("sam_model_type") != "vit_b"
        or arm_summary.get("split_sha256") != args.expected_split_sha256
        or arm_summary.get("choices_frozen_before_spatial_gt") is not True
        or arm_summary.get("test_images_read") != 0
        or arm_summary.get("test_evaluated") is not False
    ):
        raise ValueError("E3 ViT-B arm contract differs")

    gallery = args.arm_root / "gallery"
    scores = args.arm_root / "scores"
    choices_root = args.arm_root / "choices"
    candidate_manifest_path = gallery / "candidate_diagnostics_manifest.csv"
    selection_manifest_path = choices_root / "selection_manifest.csv"
    score_freeze_path = scores / "diagnostic_freeze.json"
    if (
        sha256(candidate_manifest_path) != args.expected_candidate_manifest_sha256
        or sha256(selection_manifest_path) != args.expected_selection_manifest_sha256
        or sha256(score_freeze_path) != args.expected_score_freeze_sha256
    ):
        raise ValueError("frozen gallery/selector input differs")

    candidate_rows = {
        str(row["image_name"]): row for row in read_csv(candidate_manifest_path)
    }
    evidence_rows = {
        str(row["image_id"]): row
        for row in read_csv(scores / "descriptor_evidence_manifest.csv")
    }
    selection_rows = read_csv(selection_manifest_path)
    if (
        len(candidate_rows) != 371
        or len(evidence_rows) != 371
        or len(selection_rows) != 371
        or {row["image_id"] for row in selection_rows} != set(candidate_rows)
        or set(evidence_rows) != set(candidate_rows)
    ):
        raise ValueError("E4 canonical cohort differs")

    subsets = [
        subset
        for length in range(1, len(SOURCES) + 1)
        for subset in itertools.combinations(SOURCES, length)
    ]
    stage_a_start = time.perf_counter()
    frozen_choices: dict[str, dict[str, dict[str, object]]] = {}
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in selection_rows:
        image_id = row["image_id"]
        evidence_row = evidence_rows[image_id]
        evidence_path = scores / "descriptor_evidence" / evidence_row["evidence_path"]
        if sha256(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"score evidence changed: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as payload:
            candidate_indices = payload["candidate_indices"].astype(np.int64)
            g1 = payload["candidate_logits"].astype(np.float64)
            upstream = payload["selection_scores"].astype(np.float64)
            sources = np.asarray(
                [canonical_source(value) for value in payload["proposal_source_ids"]]
            )
        if not (
            len(np.unique(candidate_indices)) == len(candidate_indices)
            and np.all(candidate_indices[:-1] < candidate_indices[1:])
            and g1.shape == upstream.shape == sources.shape
        ):
            raise ValueError(f"candidate alignment differs: {image_id}")
        frozen_choices[image_id] = {}
        eligible_map: dict[str, np.ndarray] = {}
        for subset in subsets:
            name = subset_name(subset)
            eligible = np.isin(sources, subset)
            fused = rank_fusion_subset(g1, upstream, eligible)
            local = stable_select(fused, g1, eligible)
            eligible_map[name] = eligible
            frozen_choices[image_id][name] = {
                "selected_local_index": local,
                "selected_candidate_index": (
                    int(candidate_indices[local]) if local >= 0 else -1
                ),
                "selected_source": str(sources[local]) if local >= 0 else "none",
                "candidate_count": int(eligible.sum()),
            }
        full = subset_name(SOURCES)
        if (
            int(frozen_choices[image_id][full]["selected_local_index"])
            != int(row["selected_local_index"])
            or int(frozen_choices[image_id][full]["selected_candidate_index"])
            != int(row["selected_candidate_index"])
        ):
            raise ValueError(f"full baseline does not reproduce: {image_id}")
        cache[image_id] = {
            "candidate_indices": candidate_indices,
            "eligible": eligible_map,
            "sources": sources,
        }
    stage_a_seconds = time.perf_counter() - stage_a_start

    args.output_dir.mkdir(parents=True)
    choice_freeze = {
        "schema_version": 1,
        "study": "G4 E4 exact final-gallery source-subset choice freeze",
        "source_commit": args.source_commit,
        "subsets": [list(subset) for subset in subsets],
        "validation_images": 371,
        "arm_summary_sha256": args.expected_arm_summary_sha256,
        "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
        "selection_manifest_sha256": args.expected_selection_manifest_sha256,
        "score_freeze_sha256": args.expected_score_freeze_sha256,
        "split_sha256": args.expected_split_sha256,
        "choices": frozen_choices,
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

    # Sole spatial-GT boundary: every subset choice is immutable above.
    stage_b_start = time.perf_counter()
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
        candidate_row = candidate_rows[image_id]
        candidate_path = gallery / candidate_row["diagnostic_path"]
        if sha256(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            all_proposals = payload["sam_masks"].astype(bool)
        candidate_indices = cache[image_id]["candidate_indices"]
        if int(candidate_indices.max(initial=-1)) >= len(all_proposals):
            raise ValueError(f"candidate index out of range: {image_id}")
        proposals = all_proposals[candidate_indices]
        quality = np.asarray([dice(mask, target) for mask in proposals])
        sources = cache[image_id]["sources"]
        for subset in subsets:
            name = subset_name(subset)
            eligible = cache[image_id]["eligible"][name]
            local = int(frozen_choices[image_id][name]["selected_local_index"])
            if local < 0:
                prediction = np.zeros_like(target)
                selected_dice = selected_iou = precision = recall = area_ratio = 0.0
                miss = 1
                selected_source = "none"
            else:
                prediction = proposals[local]
                intersection = int(np.logical_and(prediction, target).sum())
                selected_dice = float(quality[local])
                selected_iou = iou(prediction, target)
                precision = float(intersection / max(1, prediction.sum()))
                recall = float(intersection / max(1, target.sum()))
                area_ratio = float(prediction.sum() / max(1, target.sum()))
                miss = int(intersection == 0)
                selected_source = str(sources[local])
            oracle = float(quality[eligible].max()) if eligible.any() else 0.0
            per_image.append({
                "subset": name,
                "image_id": image_id,
                "group_id": row["group_id"],
                "size_group": group,
                "dice": selected_dice,
                "iou": selected_iou,
                "precision": precision,
                "recall": recall,
                "complete_miss": miss,
                "selected_gt_area_ratio": area_ratio,
                "selected_source": selected_source,
                "subset_oracle_dice": oracle,
                "recall_at_0_10": int(oracle >= 0.10),
                "recall_at_0_30": int(oracle >= 0.30),
                "recall_at_0_50": int(oracle >= 0.50),
                "candidate_count": int(eligible.sum()),
                "full_candidate_count": int(len(proposals)),
            })
    stage_b_seconds = time.perf_counter() - stage_b_start
    counts = Counter(
        row["size_group"]
        for row in per_image
        if row["subset"] == subset_name(SOURCES)
    )
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
                "precision": float(np.mean([row["precision"] for row in current])),
                "recall": float(np.mean([row["recall"] for row in current])),
                "oracle_dice": float(np.mean([row["subset_oracle_dice"] for row in current])),
                "candidate_recall_at_0_10": float(np.mean([row["recall_at_0_10"] for row in current])),
                "candidate_recall_at_0_30": float(np.mean([row["recall_at_0_30"] for row in current])),
                "candidate_recall_at_0_50": float(np.mean([row["recall_at_0_50"] for row in current])),
                "complete_misses": int(sum(row["complete_miss"] for row in current)),
                "median_selected_gt_area_ratio": float(np.median([row["selected_gt_area_ratio"] for row in current])),
            }
        candidate_counts = np.asarray([int(row["candidate_count"]) for row in selected])
        source_counts = Counter(str(row["selected_source"]) for row in selected)
        summary[name] = {
            "sources": list(subset),
            "valid_for_all_371_images": all(
                int(frozen_choices[image_id][name]["candidate_count"]) > 0
                for image_id in frozen_choices
            ),
            "metrics": metrics,
            "candidate_count_mean": float(candidate_counts.mean()),
            "candidate_count_median": float(np.median(candidate_counts)),
            "candidate_count_iqr": [
                float(np.percentile(candidate_counts, 25)),
                float(np.percentile(candidate_counts, 75)),
            ],
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
    baseline = arm_summary["summary"]
    full_metrics = summary[full_name]["metrics"]
    for group in ("overall", "small", "medium", "large"):
        if (
            abs(float(full_metrics[group]["dice"]) - float(baseline[group]["dice"])) > 1e-12
            or abs(float(full_metrics[group]["oracle_dice"]) - float(baseline[group]["candidate_oracle_dice"])) > 1e-12
        ):
            raise RuntimeError(f"full baseline does not reproduce for {group}")

    report = {
        "schema_version": 1,
        "study": "G4 E4 exact final-gallery source-subset ablation",
        "source_commit": args.source_commit,
        "baseline_formula": "0.5*percentile_rank(G1)+0.5*percentile_rank(upstream)",
        "baseline_subset": full_name,
        "cohort": {"validation": 371, "tumor": 184, **dict(counts)},
        "summary": summary,
        "resource_metrics": {
            "scope": "offline fixed-selector replay; frozen candidate generation cost is reported by E3",
            "stage_a_all_seven_subsets_seconds": stage_a_seconds,
            "stage_b_all_seven_subsets_seconds": stage_b_seconds,
            "total_all_seven_subsets_seconds": stage_a_seconds + stage_b_seconds,
            "amortized_seconds_per_subset_per_validation_image": (
                (stage_a_seconds + stage_b_seconds) / (7 * 371)
            ),
            "frozen_e3_candidate_generation_elapsed_seconds": arm_summary["resource_metrics"]["candidate_generation_elapsed_seconds"],
            "frozen_e3_gallery_bytes": arm_summary["resource_metrics"]["merged_gallery_bytes"],
        },
        "choice_freeze_sha256": sha256(choice_path),
        "per_image_sha256": per_image_sha,
        "split_sha256": args.expected_split_sha256,
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
