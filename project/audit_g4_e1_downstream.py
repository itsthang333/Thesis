from __future__ import annotations

"""Independent audit of one completed three-seed G4 E1 downstream arm."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_g4_e1_downstream import E1_SHA, PROTOCOL_SHA
from run_g4_e3_sam_backbone import G1_SHA, SAM_SHA, SPLIT_SHA


SOURCE_COMMIT = "b119a1dbd470f3802c60669e364db4912d5e755a"
RUNNER_SHA256 = "c2d0b60b13b73f0379168e83b1130aeb92a92bdafa81d6c52f69999a1bdfb4e5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def no_test(
    payload: dict[str, object], *, name: str, require_read_count: bool = True
) -> None:
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} evaluated test")
    if require_read_count and int(payload.get("test_images_read", -1)) != 0:
        raise ValueError(f"{name} did not prove zero test reads")


def audit(root: Path, arm: str) -> dict[str, object]:
    if arm not in E1_SHA:
        raise ValueError(f"unsupported E1 downstream arm: {arm}")
    root = root.resolve()
    arm_path = root / "arm_summary.json"
    if not arm_path.is_file():
        raise FileNotFoundError(arm_path)
    arm_summary = read_json(arm_path)
    no_test(arm_summary, name="arm summary")
    if (
        arm_summary.get("study") != "G4 E1 label granularity downstream WSSS"
        or arm_summary.get("arm") != arm
        or arm_summary.get("protocol_sha256") != PROTOCOL_SHA
        or arm_summary.get("source_commit") != SOURCE_COMMIT
        or arm_summary.get("split_sha256") != SPLIT_SHA
        or arm_summary.get("choices_frozen_before_spatial_gt") is not True
        or int(arm_summary.get("spatial_annotations_opened_per_seed", -1)) != 184
    ):
        raise ValueError("E1 arm-level contract differs")

    reported = arm_summary.get("seed_results")
    if not isinstance(reported, list) or len(reported) != 3:
        raise ValueError("E1 arm must contain exactly three seed results")
    reported_by_seed = {int(item["seed"]): item for item in reported}
    if set(reported_by_seed) != set(E1_SHA[arm]):
        raise ValueError("E1 seed set differs")

    audited_seeds: list[dict[str, object]] = []
    dice_values: list[float] = []
    for seed, checkpoint_sha in sorted(E1_SHA[arm].items()):
        seed_root = root / f"seed_{seed}"
        paths = {
            "seed": seed_root / "summary.json",
            "anchor": seed_root / "anchor" / "candidate_supply_manifest.json",
            "gallery": seed_root / "gallery" / "gallery_merge_contract.json",
            "score": seed_root / "scores" / "diagnostic_freeze.json",
            "choice": seed_root / "choices" / "prediction_freeze.json",
            "selection": seed_root / "choices" / "selection_manifest.csv",
            "evaluation": seed_root / "evaluation" / "summary.json",
            "evaluation_audit": seed_root / "evaluation" / "evaluation_audit.json",
            "per_image": seed_root / "evaluation" / "per_image.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"E1 seed {seed} output is incomplete: {missing}")
        seed_summary = read_json(paths["seed"])
        anchor = read_json(paths["anchor"])
        gallery = read_json(paths["gallery"])
        score = read_json(paths["score"])
        choice = read_json(paths["choice"])
        evaluation = read_json(paths["evaluation"])
        evaluation_audit = read_json(paths["evaluation_audit"])
        for name, payload in (
            ("anchor", anchor),
            ("gallery", gallery),
            ("score", score),
            ("choice", choice),
            ("evaluation", evaluation),
        ):
            no_test(payload, name=f"seed {seed} {name}")
        no_test(
            evaluation_audit,
            name=f"seed {seed} evaluation audit",
            require_read_count=False,
        )

        rows = read_csv(paths["per_image"])
        selections = read_csv(paths["selection"])
        if (
            len(rows) != 184
            or len({row["image_id"] for row in rows}) != 184
            or len(selections) != 371
            or len({row["image_id"] for row in selections}) != 371
        ):
            raise ValueError(f"E1 seed {seed} evaluation/selection cohort differs")
        subgroup_counts = {
            name: sum(row["size_group"] == name for row in rows)
            for name in ("small", "medium", "large")
        }
        if subgroup_counts != {"small": 94, "medium": 72, "large": 18}:
            raise ValueError(f"E1 seed {seed} subgroup cohort differs")

        summary = evaluation.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"E1 seed {seed} evaluation summary is absent")
        reported_seed = reported_by_seed[seed]
        if (
            seed_summary.get("seed") != seed
            or seed_summary.get("classifier_checkpoint_sha256") != checkpoint_sha
            or seed_summary.get("summary") != summary
            or reported_seed != seed_summary
            or reported_seed.get("evaluation_summary_sha256") != sha256(paths["evaluation"])
            or anchor.get("classifier_checkpoint_sha256") != checkpoint_sha
            or anchor.get("protocol_sha256") != PROTOCOL_SHA
            or anchor.get("source_commit") != SOURCE_COMMIT
            or anchor.get("split_sha256") != SPLIT_SHA
            or anchor.get("sam_checkpoint_sha256") != SAM_SHA["vit_b"]
            or int(anchor.get("splits", {}).get("val", {}).get("counts", {}).get("images", -1)) != 371
            or gallery.get("protocol_sha256") != PROTOCOL_SHA
            or gallery.get("split_sha256") != SPLIT_SHA
            or int(gallery.get("cohort", -1)) != 371
            or score.get("protocol_sha256") != PROTOCOL_SHA
            or score.get("source_commit") != SOURCE_COMMIT
            or score.get("split_sha256") != SPLIT_SHA
            or score.get("baseline_checkpoint_sha256") != G1_SHA
            or int(score.get("images", -1)) != 371
            or choice.get("candidate_choices_frozen_before_spatial_gt") is not True
            or choice.get("candidate_choices_frozen_before_validation_gt") is not True
            or choice.get("spatial_ground_truth_used") is not False
            or choice.get("split_sha256") != SPLIT_SHA
            or int(choice.get("images", -1)) != 371
            or evaluation.get("split") != "val"
            or evaluation.get("split_sha256") != SPLIT_SHA
            or evaluation.get("validation_ablation") is not True
            or evaluation.get("candidate_choices_frozen_before_spatial_gt") is not True
            or int(evaluation.get("spatial_annotations_opened", -1)) != 184
            or int(summary.get("overall", {}).get("n", -1)) != 184
            or int(summary.get("small", {}).get("n", -1)) != 94
            or int(summary.get("medium", {}).get("n", -1)) != 72
            or int(summary.get("large", {}).get("n", -1)) != 18
            or evaluation_audit.get("pass") is not True
            or evaluation_audit.get("overall_dice_reproduced") is not False
            or evaluation_audit.get("summary_sha256") != sha256(paths["evaluation"])
            or evaluation_audit.get("per_image_sha256") != sha256(paths["per_image"])
        ):
            raise ValueError(f"E1 seed {seed} provenance/result contract differs")
        dice = float(summary["overall"]["dice"])
        if not math.isfinite(dice) or not 0.0 <= dice <= 1.0:
            raise ValueError(f"E1 seed {seed} Dice is invalid")
        dice_values.append(dice)
        audited_seeds.append({
            "seed": seed,
            "classifier_checkpoint_sha256": checkpoint_sha,
            "evaluation_summary_sha256": sha256(paths["evaluation"]),
            "per_image_sha256": sha256(paths["per_image"]),
            "selection_manifest_sha256": sha256(paths["selection"]),
            "summary": summary,
        })

    recomputed = {
        "mean_tumor_dice": statistics.mean(dice_values),
        "sample_sd_tumor_dice": statistics.stdev(dice_values),
        "seeds": 3,
    }
    aggregate = arm_summary.get("aggregate")
    if not isinstance(aggregate, dict) or any(
        abs(float(aggregate[key]) - float(recomputed[key])) > 1.0e-15
        for key in ("mean_tumor_dice", "sample_sd_tumor_dice", "seeds")
    ):
        raise ValueError("E1 aggregate was not reproduced")
    return {
        "schema_version": 1,
        "study": "independent G4 E1 downstream output audit",
        "pass": True,
        "arm": arm,
        "runner_sha256": RUNNER_SHA256,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA,
        "split_sha256": SPLIT_SHA,
        "seed_results": audited_seeds,
        "aggregate": recomputed,
        "arm_summary_sha256": sha256(arm_path),
        "images_per_seed": 371,
        "tumor_images_per_seed": 184,
        "normal_images_per_seed": 187,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(E1_SHA), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E1 independent audit output already exists")
    report = audit(args.root, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pass": True,
        "arm": args.arm,
        "aggregate": report["aggregate"],
        "audit_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
