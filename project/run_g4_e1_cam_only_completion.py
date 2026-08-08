from __future__ import annotations

"""Complete the predeclared G4 E1 CAM-only localization endpoint.

The six E1 downstream runs already froze their LayerCAM prompt maps and all
SAM/final choices before validation annotations were opened.  This bounded
CPU reporter reuses those immutable payloads, applies the already declared p90
CAM mask through the independently tested G4 evaluator, and joins that result
to the existing final selected/oracle endpoints.  It never trains or selects a
threshold/candidate from spatial ground truth.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
E1_PROTOCOL_SHA = "48b5431b6f306105b8a1f869fa97f58decd86d3c03b7f7f9b913da58f7286394"
ARMS = ("binary", "ten_class")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_project(root: Path) -> Path:
    matches = sorted(
        {
            path.parent
            for path in root.rglob("run_g4_e1_cam_only_completion.py")
            if path.parent.name == "project"
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one G4 E1 completion project, found {matches}")
    return matches[0]


def unique_by_hash(root: Path, expected: str, names: tuple[str, ...]) -> Path:
    matches = sorted(
        {
            path
            for name in names
            for path in root.rglob(name)
            if path.is_file() and sha256(path) == expected
        },
        key=lambda path: path.as_posix(),
    )
    if not matches:
        raise RuntimeError(f"no exact input for SHA-256 {expected}")
    return matches[0]


def btxrd_root(root: Path) -> Path:
    matches = sorted(
        {
            path
            for path in root.rglob("BTXRD")
            if path.is_dir()
            and (path / "images").is_dir()
            and (path / "Annotations").is_dir()
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {matches}")
    return matches[0]


def find_arm_root(root: Path, arm: str) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in root.rglob("arm_summary.json"):
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("study") == "G4 E1 label granularity downstream WSSS"
            and payload.get("arm") == arm
            and payload.get("protocol_sha256") == E1_PROTOCOL_SHA
            and payload.get("split_sha256") == SPLIT_SHA
            and payload.get("choices_frozen_before_spatial_gt") is True
            and int(payload.get("test_images_read", -1)) == 0
            and payload.get("test_evaluated") is False
        ):
            matches.append((path.parent, payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact E1 {arm} output, found {matches}")
    return matches[0]


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def aggregate_seed_results(seed_results: list[dict[str, object]]) -> dict[str, object]:
    cam_dice = [float(item["cam_only"]["mean_tumor_dice"]) for item in seed_results]
    final_dice = [float(item["final_selected"]["dice"]) for item in seed_results]
    oracle = [float(item["final_selected"]["candidate_oracle_dice"]) for item in seed_results]
    return {
        "seeds": len(seed_results),
        "cam_only_mean_dice": statistics.mean(cam_dice),
        "cam_only_sample_sd_dice": statistics.stdev(cam_dice),
        "final_selected_mean_dice": statistics.mean(final_dice),
        "final_selected_sample_sd_dice": statistics.stdev(final_dice),
        "final_candidate_oracle_mean_dice": statistics.mean(oracle),
        "final_candidate_oracle_sample_sd_dice": statistics.stdev(oracle),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    project = unique_project(input_root)
    source = project.parent
    split = unique_by_hash(
        input_root,
        SPLIT_SHA,
        ("canonical_split_manifest_85511.csv", "split_manifest.csv"),
    )
    data = btxrd_root(input_root)
    output = working / "g4_e1_cam_only_completion"
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join(
            [str(source / "project"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
    })

    arm_results: dict[str, object] = {}
    input_hashes: dict[str, object] = {}
    for arm in ARMS:
        arm_root, arm_summary = find_arm_root(input_root, arm)
        input_hashes[arm] = {
            "arm_summary_sha256": sha256(arm_root / "arm_summary.json"),
            "source_commit": arm_summary["source_commit"],
        }
        indexed = {int(item["seed"]): item for item in arm_summary["seed_results"]}
        if set(indexed) != set(SEEDS):
            raise ValueError(f"E1 {arm} seed set differs: {sorted(indexed)}")
        seed_results: list[dict[str, object]] = []
        for seed in SEEDS:
            seed_root = arm_root / f"seed_{seed}"
            prediction = seed_root / "anchor" / "val"
            final_evaluation = seed_root / "evaluation" / "summary.json"
            pseudo_summary = prediction / "pseudo_mask_summary.json"
            candidate_summary = prediction / "candidate_diagnostics_summary.json"
            expected = indexed[seed]
            if (
                sha256(final_evaluation) != expected["evaluation_summary_sha256"]
                or not pseudo_summary.is_file()
                or not candidate_summary.is_file()
            ):
                raise ValueError(f"E1 {arm}/seed {seed} frozen input differs")
            evaluation = output / arm / f"seed_{seed}"
            run(
                [
                    sys.executable,
                    str(project / "evaluate_g4_pseudo_mask_variant.py"),
                    "--dataset-root", str(data),
                    "--split-manifest", str(split),
                    "--expected-split-sha256", SPLIT_SHA,
                    "--prediction-root", str(prediction),
                    "--expected-pseudo-summary-sha256", sha256(pseudo_summary),
                    "--expected-candidate-summary-sha256", sha256(candidate_summary),
                    "--variant-name", f"E1__{arm}__seed_{seed}__anchor",
                    "--output-dir", str(evaluation),
                    "--bootstrap-iterations", str(args.bootstrap_iterations),
                ],
                cwd=source,
                env=env,
            )
            report = read_json(evaluation / "summary.json")
            audit = read_json(evaluation / "audit.json")
            final_report = read_json(final_evaluation)
            if (
                audit.get("pass") is not True
                or report.get("candidate_analysis_enabled") is not True
                or int(report.get("images", -1)) != 371
                or int(report.get("tumor_images", -1)) != 184
                or int(report.get("validation_annotations_opened", -1)) != 184
                or report.get("test_evaluated") is not False
            ):
                raise ValueError(f"E1 {arm}/seed {seed} CAM-only evaluator failed")
            cam = report["cam_only_summary"]
            seed_results.append({
                "seed": seed,
                "checkpoint_sha256": expected["classifier_checkpoint_sha256"],
                "input_candidate_summary_sha256": sha256(candidate_summary),
                "input_pseudo_summary_sha256": sha256(pseudo_summary),
                "cam_only": {
                    "percentile": 90.0,
                    "mean_tumor_dice": cam["mean_tumor_dice"],
                    "mean_tumor_iou": cam["mean_tumor_iou"],
                    "tumor_zero_overlap_rate": cam["tumor_zero_overlap_rate"],
                    "native_subgroups": cam["native_subgroups"],
                },
                "final_selected": final_report["summary"]["overall"],
                "completion_summary_sha256": sha256(evaluation / "summary.json"),
                "cam_only_per_image_sha256": audit["cam_only_per_image_sha256"],
            })
        arm_results[arm] = {
            "seed_results": seed_results,
            "aggregate": aggregate_seed_results(seed_results),
        }

    result = {
        "schema_version": 1,
        "study": "G4 E1 CAM-only localization completion",
        "source_commit": args.source_commit,
        "e1_protocol_sha256": E1_PROTOCOL_SHA,
        "split_sha256": SPLIT_SHA,
        "cam_mask_rule": "prompt_map >= within-image 90th percentile; constant maps become empty",
        "threshold_selected_from_spatial_gt": False,
        "arms": arm_results,
        "input_hashes": input_hashes,
        "images_per_seed": 371,
        "tumor_images_per_seed": 184,
        "spatial_annotations_opened_per_seed": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_sha256": sha256(path),
        "aggregates": {arm: arm_results[arm]["aggregate"] for arm in ARMS},
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
