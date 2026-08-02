from __future__ import annotations

"""Actual binary-mask validation for frozen SMILE control/full choices."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from project.datasets.smile_reference import sha256_file
from project.models.smile_local_evidence import average_percentile_rank
from project.pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


ARMS = ("control", "full")
VARIANTS = ("baseline", "identity_only", "identity_extent")
PRIMARY = ("full", "identity_extent")
CONTROL = ("control", "identity_extent")
BASELINE_DICE = 0.2887294867
BASELINE_SUBGROUP = {
    "small": 0.1577232964,
    "medium": 0.4352293348,
    "large": 0.3868735327,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--control-stage-a", type=Path, required=True)
    parser.add_argument("--control-freeze-sha256", required=True)
    parser.add_argument("--full-stage-a", type=Path, required=True)
    parser.add_argument("--full-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / max(1, denominator))


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.logical_and(prediction, target).sum() / max(1, np.logical_or(prediction, target).sum()))


def size_group(area: float) -> str:
    return "small" if area < 0.01 else "medium" if area < 0.05 else "large"


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


def stable_select(score: np.ndarray, g1: np.ndarray) -> int:
    return max(range(len(score)), key=lambda index: (float(score[index]), float(g1[index]), -index))


def _read_stage(
    root: Path,
    *,
    arm: str,
    freeze_sha256: str,
    split_sha256: str,
    candidate_sha256: str,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, str]]]:
    freeze_path = root / "prediction_freeze.json"
    if sha256_file(freeze_path) != freeze_sha256:
        raise ValueError(f"{arm} Stage-A freeze SHA mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "smile_rich_gallery_stage_a_v1"
        or freeze.get("arm") != arm
        or freeze.get("split_sha256") != split_sha256
        or freeze.get("val_candidate_manifest_sha256") != candidate_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError(f"{arm} Stage-A contract mismatch")
    manifest_path = root / "selection_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError(f"{arm} selection manifest changed")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["variant"], row["image_id"]): row for row in rows}
    if len(rows) != 1113 or len(indexed) != 1113:
        raise ValueError(f"{arm} Stage-A selection count mismatch")
    return freeze, indexed


def _summarize(rows: list[dict[str, object]], arm: str, variant: str) -> dict[str, object]:
    selected = [row for row in rows if row["arm"] == arm and row["variant"] == variant]
    result: dict[str, object] = {}
    for group in ("overall", "small", "medium", "large"):
        current = selected if group == "overall" else [row for row in selected if row["size_group"] == group]
        result[group] = {
            "n": len(current),
            "dice": float(np.mean([row["dice"] for row in current])),
            "iou": float(np.mean([row["iou"] for row in current])),
            "precision": float(np.mean([row["precision"] for row in current])),
            "recall": float(np.mean([row["recall"] for row in current])),
            "complete_misses": int(sum(int(row["complete_miss"]) for row in current)),
            "median_selected_gt_area_ratio": float(np.median([row["selected_gt_area_ratio"] for row in current])),
            "selector_regret": float(np.mean([row["selector_regret"] for row in current])),
            "within_source_regret": float(np.mean([row["within_source_regret"] for row in current])),
            "cross_source_regret": float(np.mean([row["cross_source_regret"] for row in current])),
        }
    return result


def _group_bootstrap(
    rows: list[dict[str, object]],
    first: tuple[str, str],
    second: tuple[str, str],
    *,
    subgroup: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    indexed = {(row["arm"], row["variant"], row["image_id"]): row for row in rows}
    first_rows = [
        row
        for row in rows
        if (row["arm"], row["variant"]) == first
        and (subgroup == "overall" or row["size_group"] == subgroup)
    ]
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in first_rows:
        other = indexed[(second[0], second[1], row["image_id"])]
        by_group[str(row["group_id"])].append(float(row["dice"]) - float(other["dice"]))
    groups = sorted(by_group)
    group_values = np.asarray([np.mean(by_group[group]) for group in groups])
    rng = np.random.default_rng(seed)
    samples = group_values[rng.integers(0, len(groups), size=(replicates, len(groups)))].mean(axis=1)
    return {
        "mean_delta": float(group_values.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != args.split_sha256:
        raise ValueError("canonical split SHA mismatch")
    stages = {
        "control": _read_stage(
            args.control_stage_a,
            arm="control",
            freeze_sha256=args.control_freeze_sha256,
            split_sha256=args.split_sha256,
            candidate_sha256=args.val_candidate_manifest_sha256,
        ),
        "full": _read_stage(
            args.full_stage_a,
            arm="full",
            freeze_sha256=args.full_freeze_sha256,
            split_sha256=args.split_sha256,
            candidate_sha256=args.val_candidate_manifest_sha256,
        ),
    }
    if stages["control"][0]["protocol_sha256"] != stages["full"][0]["protocol_sha256"]:
        raise ValueError("control/full protocol differs")

    # This import is the sole spatial-annotation boundary, after both arms froze choices.
    from project.datasets.factory import build_segmentation_dataset

    dataset = build_segmentation_dataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    candidate_rows, audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[str(dataset[index][2]) for index in range(len(dataset))],
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    if audit.get("cohort") != "all":
        raise ValueError("candidate validation cohort incomplete")
    per_image: list[dict[str, object]] = []
    for index in range(len(dataset)):
        _image, target_tensor, image_id_value = dataset[index]
        image_id = str(image_id_value)
        baseline_row = stages["control"][1][("baseline", image_id)]
        if baseline_row["tumor"] != "1":
            continue
        target = target_tensor[0].numpy() > 0.5
        group = size_group(float(target.mean()))
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            proposals = payload["sam_masks"].astype(bool)
            sources = payload["proposal_source_ids"].astype(str)
        quality = np.asarray([dice(mask, target) for mask in proposals])
        for arm in ARMS:
            root = args.control_stage_a if arm == "control" else args.full_stage_a
            for variant in VARIANTS:
                frozen = stages[arm][1][(variant, image_id)]
                score_path = root / frozen["score_path"]
                if sha256_file(score_path) != frozen["score_sha256"]:
                    raise ValueError(f"score payload changed: {(arm, variant, image_id)}")
                with np.load(score_path, allow_pickle=False) as payload:
                    candidate_indices = payload["candidate_indices"].astype(np.int64)
                    g1 = payload["g1_logits"].astype(np.float64)
                    selected_local = stable_select(payload[variant], g1)
                    if selected_local != int(frozen["selected_local_index"]):
                        raise ValueError("Stage-A choice does not reproduce")
                selected_index = int(candidate_indices[selected_local])
                prediction = proposals[selected_index]
                intersection = int(np.logical_and(prediction, target).sum())
                prediction_area = int(prediction.sum())
                target_area = int(target.sum())
                eligible_quality = quality[candidate_indices]
                eligible_sources = np.asarray([canonical_source(sources[value]) for value in candidate_indices])
                selected_source = canonical_source(sources[selected_index])
                source_oracle = float(eligible_quality[eligible_sources == selected_source].max())
                per_image.append(
                    {
                        "arm": arm,
                        "variant": variant,
                        "image_id": image_id,
                        "group_id": frozen["group_id"],
                        "size_group": group,
                        "gt_area_ratio": float(target.mean()),
                        "selected_candidate_index": selected_index,
                        "selected_source": selected_source,
                        "dice": dice(prediction, target),
                        "iou": iou(prediction, target),
                        "precision": float(intersection / max(1, prediction_area)),
                        "recall": float(intersection / max(1, target_area)),
                        "complete_miss": int(intersection == 0),
                        "selected_gt_area_ratio": float(prediction_area / max(1, target_area)),
                        "oracle_dice": float(quality.max()),
                        "eligible_oracle_dice": float(eligible_quality.max()),
                        "selector_regret": float(quality.max() - quality[selected_index]),
                        "within_source_regret": float(source_oracle - quality[selected_index]),
                        "cross_source_regret": float(eligible_quality.max() - source_oracle),
                    }
                )
    if len(per_image) != 184 * len(ARMS) * len(VARIANTS):
        raise RuntimeError("Stage-B cohort mismatch")
    counts = Counter(
        row["size_group"]
        for row in per_image
        if row["arm"] == "full" and row["variant"] == "baseline"
    )
    if counts != Counter({"small": 94, "medium": 72, "large": 18}):
        raise RuntimeError(f"subgroup counts changed: {counts}")
    summary = {
        arm: {variant: _summarize(per_image, arm, variant) for variant in VARIANTS}
        for arm in ARMS
    }
    if abs(summary["control"]["baseline"]["overall"]["dice"] - BASELINE_DICE) > 1e-10:
        raise RuntimeError("immutable baseline Dice does not reproduce")
    if summary["control"]["baseline"] != summary["full"]["baseline"]:
        raise RuntimeError("control/full baseline differs")
    bootstrap = {
        comparison: {
            subgroup: _group_bootstrap(
                per_image,
                PRIMARY,
                BASELINE if comparison == "primary_vs_baseline" else CONTROL,
                subgroup=subgroup,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + offset + (0 if comparison == "primary_vs_baseline" else 20),
            )
            for offset, subgroup in enumerate(("overall", "small", "medium", "large"))
        }
        for comparison in ("primary_vs_baseline", "primary_vs_control")
    }
    primary = summary[PRIMARY[0]][PRIMARY[1]]
    control = summary[CONTROL[0]][CONTROL[1]]
    baseline = summary["control"]["baseline"]
    promotion = {
        "primary": f"{PRIMARY[0]}::{PRIMARY[1]}",
        "overall_beats_baseline": primary["overall"]["dice"] > baseline["overall"]["dice"],
        "overall_beats_control": primary["overall"]["dice"] > control["overall"]["dice"],
        "small_not_below_baseline": primary["small"]["dice"] >= BASELINE_SUBGROUP["small"],
        "medium_not_below_baseline": primary["medium"]["dice"] >= BASELINE_SUBGROUP["medium"],
        "large_not_below_baseline": primary["large"]["dice"] >= BASELINE_SUBGROUP["large"],
        "main_goal_dice_0_30": primary["overall"]["dice"] >= 0.30,
        "stretch_dice_0_311904": primary["overall"]["dice"] >= 0.311904,
    }
    promotion["minimum_pass"] = all(
        promotion[key]
        for key in (
            "overall_beats_baseline",
            "overall_beats_control",
            "small_not_below_baseline",
            "medium_not_below_baseline",
            "large_not_below_baseline",
        )
    )
    args.output_dir.mkdir(parents=True)
    per_image_sha = _write_csv(args.output_dir / "per_image.csv", per_image)
    decision = {
        "method": "smile_plus_immutable_rich_gallery",
        "summary": summary,
        "bootstrap": bootstrap,
        "promotion": promotion,
        "per_image_sha256": per_image_sha,
        "control_freeze_sha256": args.control_freeze_sha256,
        "full_freeze_sha256": args.full_freeze_sha256,
        "split_sha256": args.split_sha256,
        "validation_images": 371,
        "tumor_validation_images": 184,
        "subgroup_counts": dict(counts),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

