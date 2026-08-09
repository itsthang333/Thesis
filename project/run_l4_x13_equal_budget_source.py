"""L4 X13: equal-budget source-complementarity replay.

The candidate gallery, G1 logits, upstream scores and validation cohort are
frozen inputs.  Before any validation polygon is opened, each image receives a
budget K_i=min(K_max, N_L320, N_C448, N_external).  Every one of the seven
non-empty source subsets is restricted to exactly K_i proposals, chosen by a
predeclared upstream-score ranking, and the unchanged R7 rank fusion is then
applied.  Spatial GT is used only after all choices have been frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

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


def source_budget_for_label(
    counts: dict[str, int], *, tumor: bool, k_max: int
) -> int | None:
    """Return the matched tumor budget; ``None`` means known-normal abstention."""
    if not tumor:
        if counts["external_saliency"] != 0:
            raise ValueError(f"known-normal image has external proposals: {counts}")
        return None
    budget = min(k_max, *(counts[source] for source in SOURCES))
    if budget <= 0:
        raise ValueError(f"tumor image has an empty source: {counts}")
    return budget


def budget_indices(
    subset: tuple[str, ...],
    sources: np.ndarray,
    upstream: np.ndarray,
    g1: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Return exactly ``budget`` eligible indices under a fixed GT-free rule."""

    eligible = np.flatnonzero(np.isin(sources, subset))
    if len(eligible) < budget or budget <= 0:
        raise ValueError("equal-budget subset is infeasible")
    ordered = sorted(
        eligible.tolist(),
        key=lambda index: (-float(upstream[index]), -float(g1[index]), index),
    )
    return np.asarray(ordered[:budget], dtype=np.int64)


def r7_select(g1: np.ndarray, upstream: np.ndarray, indices: np.ndarray) -> int:
    fused = 0.5 * (
        average_percentile_rank(g1[indices])
        + average_percentile_rank(upstream[indices])
    )
    local = max(
        range(len(indices)),
        key=lambda offset: (float(fused[offset]), float(g1[indices[offset]]), -int(indices[offset])),
    )
    return int(indices[local])


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


def grouped_bootstrap_full_minus_subset(
    rows: list[dict[str, object]],
    subset: str,
    subgroup: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    full = subset_name(SOURCES)
    indexed = {(str(row["subset"]), str(row["image_id"])): row for row in rows}
    selected = [
        row for row in rows
        if row["subset"] == subset
        and (subgroup == "overall" or row["size_group"] == subgroup)
    ]
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        full_row = indexed[(full, str(row["image_id"]))]
        by_group[str(row["group_id"])].append(
            float(full_row["dice"]) - float(row["dice"])
        )
    values = np.asarray([np.mean(by_group[key]) for key in sorted(by_group)])
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(replicates, len(values)))
    ].mean(axis=1)
    return {
        "full_minus_subset": float(values.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "heuristic_groups": int(len(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--gallery-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-score-manifest-sha256", required=True)
    parser.add_argument("--expected-score-freeze-sha256", required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--k-max", type=int, default=27)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.k_max != 27:
        raise ValueError("X13 predeclares K_max=27")
    if sha256(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("split SHA-256 mismatch")

    candidate_manifest = args.gallery_root / "candidate_diagnostics_manifest.csv"
    score_manifest = args.score_root / "descriptor_evidence_manifest.csv"
    score_freeze = args.score_root / "diagnostic_freeze.json"
    selection_manifest = args.choice_root / "selection_manifest.csv"
    for path, expected in (
        (candidate_manifest, args.expected_candidate_manifest_sha256),
        (score_manifest, args.expected_score_manifest_sha256),
        (score_freeze, args.expected_score_freeze_sha256),
        (selection_manifest, args.expected_selection_manifest_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen input mismatch: {path.name}")
    freeze_payload = json.loads(score_freeze.read_text(encoding="utf-8"))
    if (
        freeze_payload.get("validation_gt_read") is not False
        or freeze_payload.get("test_images_read") != 0
        or freeze_payload.get("test_evaluated") is not False
    ):
        raise ValueError("score freeze violates validation-only protocol")

    candidate_rows = {row["image_name"]: row for row in read_csv(candidate_manifest)}
    score_rows = {row["image_id"]: row for row in read_csv(score_manifest)}
    choice_rows = read_csv(selection_manifest)
    split_by_id = {
        row["image_id"]: row
        for row in read_csv(args.split_manifest)
        if row.get("split") == "val" and row.get("eligible") == "1"
    }
    image_ids = [row["image_id"] for row in choice_rows]
    if (
        len(image_ids) != 371
        or len(set(image_ids)) != 371
        or set(image_ids) != set(candidate_rows)
        or set(image_ids) != set(score_rows)
        or set(image_ids) != set(split_by_id)
    ):
        raise ValueError("X13 canonical cohort differs")

    subsets = [
        subset
        for length in range(1, len(SOURCES) + 1)
        for subset in itertools.combinations(SOURCES, length)
    ]
    frozen: dict[str, dict[str, object]] = {}
    score_cache: dict[str, dict[str, np.ndarray]] = {}
    for image_id in image_ids:
        row = score_rows[image_id]
        path = args.score_root / "descriptor_evidence" / row["evidence_path"]
        if sha256(path) != row["evidence_sha256"]:
            raise ValueError(f"score payload changed: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            candidate_indices = payload["candidate_indices"].astype(np.int64)
            g1 = payload["candidate_logits"].astype(np.float64)
            upstream = payload["selection_scores"].astype(np.float64)
            sources = np.asarray(
                [canonical_source(value) for value in payload["proposal_source_ids"]]
            )
        if not (
            candidate_indices.shape == g1.shape == upstream.shape == sources.shape
            and len(np.unique(candidate_indices)) == len(candidate_indices)
        ):
            raise ValueError(f"score alignment differs: {image_id}")
        counts = {source: int(np.sum(sources == source)) for source in SOURCES}
        # The frozen Direct-Rich protocol uses the known binary image label as
        # a gate.  External saliency proposals therefore exist only for tumor
        # images; known-normal images abstain and emit an empty mask.  X13's
        # source-complementarity endpoint is macro Dice over the 184 tumor
        # images, so equal-budget selection is defined only where all three
        # sources are intentionally available.  Retaining all 371 rows in the
        # freeze makes this gate explicit rather than silently dropping cases.
        budget = source_budget_for_label(
            counts,
            tumor=split_by_id[image_id]["tumor"] == "1",
            k_max=args.k_max,
        )
        if budget is None:
            frozen[image_id] = {
                "source_counts": counts,
                "k_i": 0,
                "arms": {},
                "abstained_by_known_normal_label": True,
            }
            score_cache[image_id] = {
                "candidate_indices": candidate_indices,
                "sources": sources,
            }
            continue
        arms: dict[str, object] = {}
        for subset in subsets:
            name = subset_name(subset)
            kept = budget_indices(subset, sources, upstream, g1, budget)
            selected = r7_select(g1, upstream, kept)
            arms[name] = {
                "budget": budget,
                "budgeted_local_indices": kept.tolist(),
                "budgeted_candidate_indices": candidate_indices[kept].tolist(),
                "selected_local_index": selected,
                "selected_candidate_index": int(candidate_indices[selected]),
                "selected_source": str(sources[selected]),
            }
        frozen[image_id] = {"source_counts": counts, "k_i": budget, "arms": arms}
        score_cache[image_id] = {
            "candidate_indices": candidate_indices,
            "sources": sources,
        }

    args.output_dir.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "study": "L4 X13 equal-budget source complementarity choice freeze",
        "source_commit": args.source_commit,
        "budget_rule": "K_i=min(27,N_layercam320,N_classifier448,N_external_saliency)",
        "budget_cohort": "184 known-tumor validation images",
        "known_normal_policy": "abstain with empty mask before spatial evaluation",
        "budget_selection_rule": "descending upstream; tie descending G1; tie candidate order",
        "selector": "unchanged R7 equal percentile-rank fusion",
        "validation_images": 371,
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
        "score_manifest_sha256": args.expected_score_manifest_sha256,
        "score_freeze_sha256": args.expected_score_freeze_sha256,
        "selection_manifest_sha256": args.expected_selection_manifest_sha256,
        "choices": frozen,
        "choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "choice_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Sole spatial-GT boundary. All budgets and selections are immutable above.
    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    split_rows = {row["image_id"]: row for row in choice_rows}
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, target_tensor, image_id_value = dataset[index]
        image_id = str(image_id_value)
        split_row = split_rows[image_id]
        if split_row["tumor"] != "1":
            continue
        target = target_tensor[0].numpy() > 0.5
        group = size_group(float(target.mean()))
        candidate_row = candidate_rows[image_id]
        candidate_path = args.gallery_root / candidate_row["diagnostic_path"]
        if sha256(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            masks = payload["sam_masks"].astype(bool)
        candidate_indices = score_cache[image_id]["candidate_indices"]
        proposals = masks[candidate_indices]
        for subset in subsets:
            name = subset_name(subset)
            arm = frozen[image_id]["arms"][name]
            kept = np.asarray(arm["budgeted_local_indices"], dtype=np.int64)
            selected = int(arm["selected_local_index"])
            prediction = proposals[selected]
            selected_dice = dice(prediction, target)
            selected_iou = iou(prediction, target)
            oracle = float(max(dice(proposals[i], target) for i in kept))
            per_image.append({
                "subset": name,
                "image_id": image_id,
                "group_id": split_row["group_id"],
                "size_group": group,
                "k_i": int(frozen[image_id]["k_i"]),
                "selected_source": arm["selected_source"],
                "dice": selected_dice,
                "iou": selected_iou,
                "oracle_dice": oracle,
                "selector_regret": oracle - selected_dice,
                "recall_at_0_10": int(oracle >= 0.10),
                "recall_at_0_30": int(oracle >= 0.30),
                "recall_at_0_50": int(oracle >= 0.50),
            })
    counts = Counter(row["size_group"] for row in per_image if row["subset"] == subset_name(SOURCES))
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"subgroup counts changed: {counts}")
    per_image_sha = write_csv(args.output_dir / "per_image.csv", per_image)

    report_summary: dict[str, object] = {}
    for offset, subset in enumerate(subsets):
        name = subset_name(subset)
        subset_rows = [row for row in per_image if row["subset"] == name]
        metrics: dict[str, object] = {}
        for group_index, group in enumerate(("overall", "small", "medium", "large")):
            current = subset_rows if group == "overall" else [
                row for row in subset_rows if row["size_group"] == group
            ]
            metrics[group] = {
                "n": len(current),
                "dice": float(np.mean([row["dice"] for row in current])),
                "iou": float(np.mean([row["iou"] for row in current])),
                "oracle_dice": float(np.mean([row["oracle_dice"] for row in current])),
                "selector_regret": float(np.mean([row["selector_regret"] for row in current])),
                "recall_at_0_10": float(np.mean([row["recall_at_0_10"] for row in current])),
                "recall_at_0_30": float(np.mean([row["recall_at_0_30"] for row in current])),
                "recall_at_0_50": float(np.mean([row["recall_at_0_50"] for row in current])),
            }
            metrics[group]["paired_bootstrap_full_minus_subset"] = grouped_bootstrap_full_minus_subset(
                per_image,
                name,
                group,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 10 * offset + group_index,
            )
        budgets = np.asarray([row["k_i"] for row in subset_rows], dtype=np.int64)
        report_summary[name] = {
            "sources": list(subset),
            "candidate_count_equal_for_every_subset": True,
            "candidate_count_mean": float(budgets.mean()),
            "candidate_count_median": float(np.median(budgets)),
            "candidate_count_min": int(budgets.min()),
            "candidate_count_max": int(budgets.max()),
            "metrics": metrics,
        }
    report = {
        "schema_version": 1,
        "study": "L4 X13 equal-budget source complementarity",
        "primary_endpoint": "macro Dice over 184 validation tumor images",
        "summary": report_summary,
        "cohort": {"validation": 371, "tumor": 184, **dict(counts)},
        "choice_freeze_sha256": sha256(freeze_path),
        "per_image_sha256": per_image_sha,
        "split_sha256": args.expected_split_sha256,
        "validation_gt_opened_after_choice_freeze": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
