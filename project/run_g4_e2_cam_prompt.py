from __future__ import annotations

"""Run the bounded G4 E2 attribution x prompt validation factorial.

Phase A freezes every arm's 371 masks without spatial annotations. Phase B is
started only after the complete Phase-A lock has been written.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluation.segmentation_metrics import paired_group_bootstrap_deltas
from run_rich_gallery_candidate_supply import common_generation_args


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_SPLIT_SHA = "7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c"
CLASSIFIER_SHA = "ca630ddf816c1b6a55fab9b99fe824877bba9a83905ce71fd20cf9c2b1640621"
SAM_SHA = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
ATTRIBUTION_METHODS = ("cam", "gradcam", "gradcam_plus_plus", "layercam")
PROMPT_MODES = ("point", "box", "box_point")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_file(root: Path, name: str, expected_sha: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and sha256(path) == expected_sha]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}/{expected_sha}, found {len(matches)}")
    return matches[0]


def unique_project(root: Path) -> Path:
    matches = [
        path.parent for path in root.rglob("run_g4_e2_cam_prompt.py")
        if path.parent.name == "project"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one G4 E2 project, found {len(matches)}")
    return matches[0]


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default=",".join(ATTRIBUTION_METHODS),
        help="Comma-separated bounded subset of the four frozen attribution methods.",
    )
    parser.add_argument(
        "--prompt-modes",
        default=",".join(PROMPT_MODES),
        help="Comma-separated bounded subset of point,box,box_point.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def _validated_subset(value: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)) or any(item not in allowed for item in items):
        raise ValueError(f"{name} must be a unique non-empty subset of {allowed}")
    return items


def generation_command(
    *,
    project: Path,
    data_root: Path,
    split: Path,
    classifier_split: Path,
    classifier: Path,
    sam: Path,
    output_dir: Path,
    method: str,
    prompt_mode: str,
) -> list[str]:
    command = common_generation_args(
        source_root=project.parent,
        data_root=data_root,
        split_manifest=split,
        split="val",
        classifier=classifier,
        sam=sam,
        output_dir=output_dir,
        attribution_method=method,
        prompt_mode=prompt_mode,
        prompt_ensemble=False,
        classifier_split_manifest=classifier_split,
    )
    command.extend([
        "--image-size", "320",
        "--min-component-area", "100",
        "--prompt-border-margin", "2",
        "--support-clip-kernel", "5",
        "--min-size", "40",
    ])
    return command


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    methods = _validated_subset(args.methods, ATTRIBUTION_METHODS, "--methods")
    prompts = _validated_subset(args.prompt_modes, PROMPT_MODES, "--prompt-modes")
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    output_root = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working")) / "g4_e2"
    output_root.mkdir(parents=True, exist_ok=False)
    project = unique_project(input_root)
    split = unique_file(input_root, "canonical_split_manifest_85511.csv", SPLIT_SHA)
    classifier_split = unique_file(
        input_root, "classifier_split_manifest_7b167.csv", CLASSIFIER_SPLIT_SHA
    )
    classifier = unique_file(input_root, "best_classifier.pt", CLASSIFIER_SHA)
    sam = unique_file(input_root, "sam_vit_b_01ec64.pth", SAM_SHA)
    btxrd = [
        path for path in input_root.rglob("BTXRD")
        if path.is_dir() and (path / "images").is_dir() and (path / "Annotations").is_dir()
    ]
    if len(btxrd) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {len(btxrd)}")

    env = os.environ.copy()
    python_paths = [str(project)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(python_paths),
        "PYTHONUNBUFFERED": "1",
        "BTXRD_DISABLE_TQDM": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })

    arms = [f"{method}__{prompt}" for method in methods for prompt in prompts]
    phase_a: dict[str, dict[str, object]] = {}
    for method in methods:
        for prompt in prompts:
            arm = f"{method}__{prompt}"
            prediction_root = output_root / "predictions" / arm
            run(
                generation_command(
                    project=project,
                    data_root=btxrd[0],
                    split=split,
                    classifier_split=classifier_split,
                    classifier=classifier,
                    sam=sam,
                    output_dir=prediction_root,
                    method=method,
                    prompt_mode=prompt,
                ),
                cwd=project.parent,
                env=env,
            )
            metadata = json.loads((prediction_root / "run_metadata.json").read_text(encoding="utf-8"))
            summary_path = prediction_root / "pseudo_mask_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                metadata.get("split") != "val"
                or metadata.get("attribution_method") != method
                or metadata.get("sam_prompt_mode") != prompt
                or metadata.get("sam_prompt_ensemble") is not False
                or int(summary.get("manifest_rows", -1)) != 371
            ):
                raise ValueError(f"E2 Phase-A metadata mismatch for {arm}")
            phase_a[arm] = {
                "attribution_method": method,
                "prompt_mode": prompt,
                "prediction_root": str(prediction_root),
                "pseudo_summary_sha256": sha256(summary_path),
                "pseudo_manifest_sha256": summary["manifest_sha256"],
                "run_metadata_sha256": sha256(prediction_root / "run_metadata.json"),
            }

    freeze = {
        "schema_version": 1,
        "study": "G4 E2 attribution x single-prompt factorial",
        "split_sha256": SPLIT_SHA,
        "classifier_split_sha256": CLASSIFIER_SPLIT_SHA,
        "classifier_sha256": CLASSIFIER_SHA,
        "sam_sha256": SAM_SHA,
        "arms": phase_a,
        "all_choices_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = output_root / "phase_a_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries: dict[str, object] = {}
    per_image: dict[str, list[dict[str, str]]] = {}
    for arm in arms:
        evaluation = output_root / "evaluation" / arm
        run([
            sys.executable,
            str(project / "evaluate_g4_pseudo_mask_variant.py"),
            "--dataset-root", str(btxrd[0]),
            "--split-manifest", str(split),
            "--expected-split-sha256", SPLIT_SHA,
            "--prediction-root", str(Path(str(phase_a[arm]["prediction_root"]))),
            "--expected-pseudo-summary-sha256", str(phase_a[arm]["pseudo_summary_sha256"]),
            "--variant-name", arm,
            "--output-dir", str(evaluation),
            "--bootstrap-iterations", str(args.bootstrap_iterations),
        ], cwd=project.parent, env=env)
        report = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
        summaries[arm] = report["summary"]
        per_image[arm] = _read_csv(evaluation / "per_image.csv")

    reference = "layercam__box_point"
    paired: dict[str, object] = {}
    if reference in per_image:
        for arm in arms:
            if arm == reference:
                continue
            paired[arm] = paired_group_bootstrap_deltas(
                per_image[reference],
                per_image[arm],
                iterations=max(2000, args.bootstrap_iterations),
                seed=20260806,
            )
    report = {
        "schema_version": 1,
        "study": "G4 E2 attribution x single-prompt factorial",
        "primary_endpoint": "actual mean per-tumor binary-mask Dice on common 320 grid",
        "reference_arm": reference if reference in arms else None,
        "phase_a_freeze_sha256": sha256(freeze_path),
        "summaries": summaries,
        "paired_deltas_vs_reference": paired,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = output_root / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase_a_freeze_sha256": sha256(freeze_path),
        "summary_sha256": sha256(report_path),
        "arms": arms,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
