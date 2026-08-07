from __future__ import annotations

"""Run one three-seed G4 E1 label-granularity arm end to end.

Only the matched 320px classifier checkpoint and its mathematically equivalent
binary attribution target change.  All later proposal, gallery, G1 and fusion
components are frozen and shared.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from run_g4_e3_sam_backbone import (
    G1_SHA,
    HUB_WHEEL_SHA,
    RAD_CONFIG_SHA,
    RAD_PREPROCESSOR_SHA,
    RAD_WEIGHT_SHA,
    SAM_SHA,
    SPLIT_SHA,
    TOKENIZERS_WHEEL_SHA,
    TRANSFORMERS_WHEEL_SHA,
    btxrd_root,
    canonical_split,
    install_g1_runtime,
    run,
    sha256,
    unique_hash,
    unique_named,
    unique_project,
)


PROTOCOL_SHA = "48b5431b6f306105b8a1f869fa97f58decd86d3c03b7f7f9b913da58f7286394"
ADDITION_CLASSIFIER_SHA = (
    "b40dc5ec0f601ea7392fd0e8ed0be5f1e7cd66ad07d654392db516a0766d451e"
)
E1_SHA = {
    "binary": {
        42: "960a6ea4ec9a452a5a8903a98ec9b8ef6026258b9d1b15d043e3954123f2247b",
        43: "8edb702230eb8e19317194c0e28720aeb9c68c0b0480ff861472dfecd5e8321a",
        44: "3956e771040a991d279afc1c551e9f0136ce635a633805c3bd8304b3da49f6ba",
    },
    "ten_class": {
        42: "e043f43655798f3389005be89cb85605ef09a87977a74733ea07bf9abc0ed7ec",
        43: "5017e7abde408273e5c03b312bb2aa714bfbb8261104336a3e4ef8ec24c29882",
        44: "1bdba52b665edd2d254e23b6b8bcd5fe4889f9297fe649a2403a8ced1e674a22",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(E1_SHA), required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def addition_supply(root: Path) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in root.rglob("candidate_supply_manifest.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("mode") == "addition"
            and payload.get("classifier_checkpoint_sha256") == ADDITION_CLASSIFIER_SHA
            and payload.get("sam_checkpoint_sha256") == SAM_SHA["vit_b"]
            and payload.get("split_sha256") == SPLIT_SHA
            and payload.get("test_evaluated") is False
            and int(payload.get("test_images_read", -1)) == 0
            and int(payload.get("splits", {}).get("val", {}).get("counts", {}).get("images", -1)) == 371
        ):
            matches.append((path.parent, payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen classifier448 addition supply, found {matches}")
    return matches[0]


def anchor_command(
    *,
    project: Path,
    data: Path,
    split: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    sam: Path,
    external_root: Path,
    source_commit: str,
    arm: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(project / "run_rich_gallery_candidate_supply.py"),
        "--mode", "anchor",
        "--source-root", str(project.parent),
        "--data-root", str(data),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--classifier-split-manifest", str(split),
        "--expected-classifier-split-sha256", SPLIT_SHA,
        "--classifier-checkpoint", str(checkpoint),
        "--expected-classifier-sha256", checkpoint_sha,
        "--sam-checkpoint", str(sam),
        "--expected-sam-sha256", SAM_SHA["vit_b"],
        "--sam-model-type", "vit_b",
        "--external-saliency-supply-root", str(external_root),
        "--expected-external-supply-manifest-sha256",
        sha256(external_root / "saliency_supply_manifest.json"),
        "--target-columns", "tumor" if arm == "binary" else "tumor_type",
        "--cam-aggregation", "class" if arm == "binary" else "tumor_log_odds",
        "--source-commit", source_commit,
        "--protocol-sha256", PROTOCOL_SHA,
        "--output-dir", str(output),
        "--splits", "val",
    ]


def main() -> None:
    args = parse_args()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    project = unique_project(input_root)
    source = project.parent
    split = canonical_split(input_root)
    data = btxrd_root(input_root)
    sam = unique_hash(
        input_root, SAM_SHA["vit_b"], names=("sam_vit_b_01ec64.pth",)
    )
    rad_weight = unique_hash(input_root, RAD_WEIGHT_SHA, names=("model.safetensors",))
    rad_dir = rad_weight.parent
    if (
        sha256(rad_dir / "config.json") != RAD_CONFIG_SHA
        or sha256(rad_dir / "preprocessor_config.json") != RAD_PREPROCESSOR_SHA
    ):
        raise ValueError("RAD-DINO model directory is not the frozen snapshot")
    g1 = unique_hash(input_root, G1_SHA, names=("rad_dino_mask_bag_mil.pt",))
    external_manifest = unique_named(input_root, "saliency_supply_manifest.json")
    external_root = external_manifest.parent
    external = json.loads(external_manifest.read_text(encoding="utf-8"))
    if (
        int(external.get("splits", {}).get("val", {}).get("images", -1)) != 371
        or external.get("spatial_ground_truth_read") is not False
        or int(external.get("test_images_read", -1)) != 0
        or external.get("test_evaluated") is not False
    ):
        raise ValueError("external saliency supply violates the validation contract")
    addition_root, addition_manifest = addition_supply(input_root)
    addition_val = addition_manifest["splits"]["val"]

    sam_package = unique_named(input_root, "automatic_mask_generator.py").parent.parent
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join([
            str(source / "project"), str(sam_package), env.get("PYTHONPATH", "")
        ]).rstrip(os.pathsep),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "BTXRD_DISABLE_TQDM": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    install_g1_runtime(input_root, cwd=working, env=env)

    arm_root = working / f"g4_e1_downstream_{args.arm}"
    arm_root.mkdir(parents=True, exist_ok=False)
    seed_results: list[dict[str, object]] = []
    for seed, checkpoint_sha in E1_SHA[args.arm].items():
        checkpoint = unique_hash(
            input_root,
            checkpoint_sha,
            names=(f"e1_{args.arm}_seed_{seed}.pt", "best_classifier.pt"),
        )
        seed_root = arm_root / f"seed_{seed}"
        anchor = seed_root / "anchor"
        run(
            anchor_command(
                project=project,
                data=data,
                split=split,
                checkpoint=checkpoint,
                checkpoint_sha=checkpoint_sha,
                sam=sam,
                external_root=external_root,
                source_commit=args.source_commit,
                arm=args.arm,
                output=anchor,
            ),
            cwd=source,
            env=env,
        )
        anchor_manifest = json.loads(
            (anchor / "candidate_supply_manifest.json").read_text(encoding="utf-8")
        )
        anchor_val = anchor_manifest["splits"]["val"]

        gallery = seed_root / "gallery"
        run([
            sys.executable, str(project / "merge_frozen_candidate_galleries.py"),
            "--split-manifest", str(split),
            "--expected-split-sha256", SPLIT_SHA,
            "--split", "val",
            "--anchor-root", str(anchor / "val"),
            "--anchor-candidate-manifest-sha256", str(anchor_val["candidate_manifest_sha256"]),
            "--anchor-pseudo-manifest-sha256", str(anchor_val["pseudo_manifest_sha256"]),
            "--addition-root", str(addition_root / "val"),
            "--addition-candidate-manifest-sha256", str(addition_val["candidate_manifest_sha256"]),
            "--addition-pseudo-manifest-sha256", str(addition_val["pseudo_manifest_sha256"]),
            "--addition-namespace", "classifier448",
            "--protocol-sha256", PROTOCOL_SHA,
            "--output-dir", str(gallery),
        ], cwd=source, env=env)
        contract = json.loads(
            (gallery / "gallery_merge_contract.json").read_text(encoding="utf-8")
        )

        scores = seed_root / "scores"
        run([
            sys.executable, str(project / "score_final_rich_gallery.py"),
            "--split", "val",
            "--dataset-root", str(data),
            "--split-manifest", str(split),
            "--expected-split-sha256", SPLIT_SHA,
            "--model-dir", str(rad_dir),
            "--expected-config-sha256", RAD_CONFIG_SHA,
            "--expected-preprocessor-sha256", RAD_PREPROCESSOR_SHA,
            "--expected-weight-sha256", RAD_WEIGHT_SHA,
            "--candidate-root", str(gallery),
            "--candidate-manifest-sha256", str(contract["output_manifest_sha256"]),
            "--pseudo-manifest-sha256", str(contract["anchor_pseudo_manifest_sha256"]),
            "--g1-checkpoint", str(g1),
            "--expected-g1-checkpoint-sha256", G1_SHA,
            "--source-commit", args.source_commit,
            "--protocol-sha256", PROTOCOL_SHA,
            "--output-dir", str(scores),
        ], cwd=source, env=env)

        choices = seed_root / "choices"
        run([
            sys.executable, str(project / "freeze_final_rich_gallery.py"),
            "--split", "val",
            "--split-manifest", str(split),
            "--expected-split-sha256", SPLIT_SHA,
            "--g1-diagnostic-root", str(scores),
            "--expected-g1-freeze-sha256", sha256(scores / "diagnostic_freeze.json"),
            "--candidate-root", str(gallery),
            "--expected-candidate-manifest-sha256", str(contract["output_manifest_sha256"]),
            "--expected-pseudo-manifest-sha256", str(contract["anchor_pseudo_manifest_sha256"]),
            "--output-dir", str(choices),
        ], cwd=source, env=env)

        evaluation = seed_root / "evaluation"
        run([
            sys.executable, str(project / "evaluate_final_rich_gallery.py"),
            "--split", "val",
            "--allow-validation-ablation",
            "--dataset-root", str(data),
            "--split-manifest", str(split),
            "--expected-split-sha256", SPLIT_SHA,
            "--selection-root", str(choices),
            "--expected-selection-freeze-sha256", sha256(choices / "prediction_freeze.json"),
            "--candidate-root", str(gallery),
            "--output-dir", str(evaluation),
        ], cwd=source, env=env)
        evaluation_summary = json.loads(
            (evaluation / "summary.json").read_text(encoding="utf-8")
        )
        seed_result = {
            "seed": seed,
            "classifier_checkpoint_sha256": checkpoint_sha,
            "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
            "summary": evaluation_summary["summary"],
        }
        (seed_root / "summary.json").write_text(
            json.dumps(seed_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seed_results.append(seed_result)

    dice = [float(item["summary"]["overall"]["dice"]) for item in seed_results]
    result = {
        "schema_version": 1,
        "study": "G4 E1 label granularity downstream WSSS",
        "arm": args.arm,
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": args.source_commit,
        "split_sha256": SPLIT_SHA,
        "attribution_target": (
            "one-logit tumor score" if args.arm == "binary"
            else "logsumexp(tumor logits)-normal logit"
        ),
        "seed_results": seed_results,
        "aggregate": {
            "mean_tumor_dice": statistics.mean(dice),
            "sample_sd_tumor_dice": statistics.stdev(dice),
            "seeds": len(dice),
        },
        "choices_frozen_before_spatial_gt": True,
        "spatial_annotations_opened_per_seed": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    result_path = arm_root / "arm_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "arm": args.arm,
        "mean_tumor_dice": result["aggregate"]["mean_tumor_dice"],
        "arm_summary_sha256": sha256(result_path),
        "test_evaluated": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
