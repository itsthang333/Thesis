from __future__ import annotations

"""Summarize the frozen G4 E2 attribution/prompt candidate decomposition.

This reporter never opens images or annotations.  It consumes only audited
per-image tables produced by ``evaluate_g4_pseudo_mask_variant.py`` and keeps
selected-mask quality, CAM-only quality, proposal supply, and selector regret
as distinct estimands.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("cam", "gradcam", "gradcam_plus_plus", "layercam")
PROMPTS = ("point", "box", "box_point")
EXPECTED_ARMS = tuple(f"{method}__{prompt}" for method in METHODS for prompt in PROMPTS)
SIZE_GROUPS = ("small_lt_1pct", "medium_1_to_5pct", "large_ge_5pct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid bool literal: {value!r}")


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if image_id in rows:
                raise ValueError(f"duplicate image {image_id} in {path}")
            rows[image_id] = row
    return rows


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _arm_result(arm_dir: Path) -> dict[str, object]:
    summary_path = arm_dir / "summary.json"
    audit_path = arm_dir / "audit.json"
    per_image_path = arm_dir / "per_image.csv"
    cam_path = arm_dir / "cam_only_per_image.csv"
    if not all(path.is_file() for path in (summary_path, audit_path, per_image_path, cam_path)):
        raise FileNotFoundError(f"incomplete candidate-decomposition arm: {arm_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if audit.get("pass") is not True or report.get("candidate_analysis_enabled") is not True:
        raise ValueError(f"candidate analysis is not audited for {arm_dir.name}")
    if (
        report.get("images") != 371
        or report.get("tumor_images") != 184
        or report.get("validation_annotations_opened") != 184
        or report.get("test_images_read") != 0
        or report.get("test_evaluated") is not False
    ):
        raise ValueError(f"cohort/protocol mismatch for {arm_dir.name}")
    if audit.get("per_image_sha256") != sha256(per_image_path):
        raise ValueError(f"per-image hash mismatch for {arm_dir.name}")
    if audit.get("cam_only_per_image_sha256") != sha256(cam_path):
        raise ValueError(f"CAM-only hash mismatch for {arm_dir.name}")

    selected = _read_rows(per_image_path)
    cam = _read_rows(cam_path)
    if set(selected) != set(cam) or len(selected) != 371:
        raise ValueError(f"selected/CAM populations differ for {arm_dir.name}")
    tumor_ids = sorted(image_id for image_id, row in selected.items() if _bool(row["gt_positive"]))
    if len(tumor_ids) != 184:
        raise ValueError(f"tumor population differs for {arm_dir.name}")
    selected_tumor = [selected[image_id] for image_id in tumor_ids]
    cam_tumor = [cam[image_id] for image_id in tumor_ids]
    oracle = np.asarray([float(row["candidate_oracle_dice"]) for row in selected_tumor])
    selected_dice = np.asarray([float(row["dice"]) for row in selected_tumor])
    cam_dice = np.asarray([float(row["dice"]) for row in cam_tumor])

    subgroup: dict[str, object] = {}
    for group in SIZE_GROUPS:
        keep = np.asarray([row["native_size_group"] == group for row in selected_tumor])
        subgroup[group] = {
            "n": int(keep.sum()),
            "selected_dice": float(selected_dice[keep].mean()),
            "cam_only_dice": float(cam_dice[keep].mean()),
            "proposal_oracle_dice": float(oracle[keep].mean()),
            "selector_regret": float((oracle[keep] - selected_dice[keep]).mean()),
        }
    return {
        "selected_dice": float(selected_dice.mean()),
        "cam_only_dice": float(cam_dice.mean()),
        "proposal_oracle_dice": float(oracle.mean()),
        "selector_regret": float((oracle - selected_dice).mean()),
        "sam_gain_over_cam": float(selected_dice.mean() - cam_dice.mean()),
        "recall_at_dice_0_10": float(np.mean(oracle >= 0.10)),
        "recall_at_dice_0_30": float(np.mean(oracle >= 0.30)),
        "recall_at_dice_0_50": float(np.mean(oracle >= 0.50)),
        "subgroups": subgroup,
        "summary_sha256": sha256(summary_path),
        "per_image_sha256": sha256(per_image_path),
        "cam_only_per_image_sha256": sha256(cam_path),
    }


def summarize(roots: list[Path]) -> dict[str, object]:
    arms: dict[str, dict[str, object]] = {}
    for root in roots:
        for arm_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if arm_dir.name not in EXPECTED_ARMS:
                continue
            if arm_dir.name in arms:
                raise ValueError(f"duplicate arm {arm_dir.name}")
            arms[arm_dir.name] = _arm_result(arm_dir)
    missing = sorted(set(EXPECTED_ARMS) - set(arms))
    if missing:
        raise ValueError(f"candidate decomposition is incomplete: {missing}")

    endpoint_keys = (
        "selected_dice",
        "cam_only_dice",
        "proposal_oracle_dice",
        "selector_regret",
        "sam_gain_over_cam",
    )
    method_marginals = {
        method: {
            key: float(np.mean([arms[f"{method}__{prompt}"][key] for prompt in PROMPTS]))
            for key in endpoint_keys
        }
        for method in METHODS
    }
    prompt_marginals = {
        prompt: {
            key: float(np.mean([arms[f"{method}__{prompt}"][key] for method in METHODS]))
            for key in endpoint_keys
        }
        for prompt in PROMPTS
    }
    best_selected = max(arms, key=lambda arm: float(arms[arm]["selected_dice"]))
    best_oracle = max(arms, key=lambda arm: float(arms[arm]["proposal_oracle_dice"]))
    return {
        "schema_version": 1,
        "study": "G4 E2 CAM-to-SAM candidate decomposition",
        "population": "184 canonical validation tumor images",
        "arms": arms,
        "method_marginals": method_marginals,
        "prompt_marginals": prompt_marginals,
        "best_selected_arm": best_selected,
        "best_oracle_arm": best_oracle,
        "interpretation_rule": (
            "CAM-only measures attribution localization; proposal oracle measures SAM/gallery supply; "
            "selected Dice measures the deployed upstream choice; oracle-selected difference is selector regret."
        ),
        "spatial_ground_truth_opened_by_this_script": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args.evaluation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "best_selected_arm": result["best_selected_arm"],
        "best_oracle_arm": result["best_oracle_arm"],
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
