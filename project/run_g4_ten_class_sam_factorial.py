from __future__ import annotations

"""Run one seed of the matched ten-class x SAM factorial on validation.

The seed-specific ten-class classifier changes the anchor attribution target.
The tested SAM checkpoint is used by both SAM-generated candidate supplies.
External saliency, the gallery contract, frozen G1 and R7 fusion are unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from run_g4_e1_downstream import E1_SHA
from run_g4_e3_sam_backbone import (
    CLASSIFIER_448_SHA,
    G1_SHA,
    HUB_WHEEL_SHA,
    RAD_CONFIG_SHA,
    RAD_PREPROCESSOR_SHA,
    RAD_WEIGHT_SHA,
    SAM_SHA,
    SPLIT_SHA,
    TOKENIZERS_WHEEL_SHA,
    TRANSFORMERS_WHEEL_SHA,
    _supply_command,
    btxrd_root,
    canonical_split,
    install_g1_runtime,
    run,
    sha256,
    unique_hash,
    unique_named,
    unique_project,
)


PROTOCOL_SHA = "949a6f9441fa2f1964a9f2e133e95a871b8701291e5c2fd5507b0bcac9a96df6"
SUPPORTED_SAM = ("vit_l",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=tuple(E1_SHA["ten_class"]), required=True)
    parser.add_argument("--sam-model-type", choices=SUPPORTED_SAM, default="vit_l")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    project = unique_project(input_root)
    source = project.parent
    split = canonical_split(input_root)
    data = btxrd_root(input_root)
    checkpoint_sha = E1_SHA["ten_class"][args.seed]
    classifier = unique_hash(
        input_root,
        checkpoint_sha,
        names=(f"e1_ten_class_seed_{args.seed}.pt", "best_classifier.pt"),
    )
    classifier_448 = unique_hash(
        input_root,
        CLASSIFIER_448_SHA,
        names=("best_classifier448.pt", "best_classifier.pt"),
    )
    sam = unique_hash(
        input_root,
        SAM_SHA[args.sam_model_type],
        names=("sam_vit_l_0b3195.pth",),
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

    arm_root = working / f"g4_ten_class_{args.sam_model_type}_seed_{args.seed}"
    arm_root.mkdir(parents=True, exist_ok=False)
    anchor = arm_root / "anchor"
    anchor_command = _supply_command(
        project=project,
        data=data,
        split=split,
        classifier=classifier,
        sam=sam,
        sam_model_type=args.sam_model_type,
        source_commit=args.source_commit,
        output=anchor,
        mode="anchor",
        external_root=external_root,
    )
    anchor_command.extend([
        "--classifier-split-manifest", str(split),
        "--expected-classifier-split-sha256", SPLIT_SHA,
        "--target-columns", "tumor_type",
        "--cam-aggregation", "tumor_log_odds",
    ])
    # _supply_command carries E3's protocol; replace it with this factorial's
    # immutable protocol while leaving every scientific argument explicit.
    protocol_index = anchor_command.index("--protocol-sha256") + 1
    anchor_command[protocol_index] = PROTOCOL_SHA
    run(anchor_command, cwd=source, env=env)

    addition = arm_root / "addition"
    addition_command = _supply_command(
        project=project,
        data=data,
        split=split,
        classifier=classifier_448,
        sam=sam,
        sam_model_type=args.sam_model_type,
        source_commit=args.source_commit,
        output=addition,
        mode="addition",
    )
    addition_command[addition_command.index("--protocol-sha256") + 1] = PROTOCOL_SHA
    run(addition_command, cwd=source, env=env)

    anchor_manifest = json.loads((anchor / "candidate_supply_manifest.json").read_text(encoding="utf-8"))
    addition_manifest = json.loads((addition / "candidate_supply_manifest.json").read_text(encoding="utf-8"))
    anchor_val = anchor_manifest["splits"]["val"]
    addition_val = addition_manifest["splits"]["val"]

    gallery = arm_root / "gallery"
    run([
        sys.executable, str(project / "merge_frozen_candidate_galleries.py"),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--split", "val",
        "--anchor-root", str(anchor / "val"),
        "--anchor-candidate-manifest-sha256", str(anchor_val["candidate_manifest_sha256"]),
        "--anchor-pseudo-manifest-sha256", str(anchor_val["pseudo_manifest_sha256"]),
        "--addition-root", str(addition / "val"),
        "--addition-candidate-manifest-sha256", str(addition_val["candidate_manifest_sha256"]),
        "--addition-pseudo-manifest-sha256", str(addition_val["pseudo_manifest_sha256"]),
        "--addition-namespace", "classifier448",
        "--protocol-sha256", PROTOCOL_SHA,
        "--output-dir", str(gallery),
    ], cwd=source, env=env)
    contract = json.loads((gallery / "gallery_merge_contract.json").read_text(encoding="utf-8"))

    install_g1_runtime(input_root, cwd=working, env=env)
    scores = arm_root / "scores"
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

    choices = arm_root / "choices"
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

    evaluation = arm_root / "evaluation"
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
    evaluation_summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    resources = [
        json.loads((anchor / "val" / "resource_metrics.json").read_text(encoding="utf-8")),
        json.loads((addition / "val" / "resource_metrics.json").read_text(encoding="utf-8")),
    ]
    result = {
        "schema_version": 1,
        "study": "G4 ten-class x SAM factorial",
        "seed": args.seed,
        "classifier_checkpoint_sha256": checkpoint_sha,
        "sam_model_type": args.sam_model_type,
        "sam_checkpoint_sha256": SAM_SHA[args.sam_model_type],
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": args.source_commit,
        "split_sha256": SPLIT_SHA,
        "attribution_target": "logsumexp(tumor logits)-normal logit",
        "summary": evaluation_summary["summary"],
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "candidate_generation_elapsed_seconds": float(sum(item["elapsed_seconds"] for item in resources)),
        "total_elapsed_seconds": float(time.perf_counter() - started),
        "choices_frozen_before_spatial_gt": True,
        "spatial_annotations_opened": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    result_path = arm_root / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "seed": args.seed,
        "sam_model_type": args.sam_model_type,
        "overall": result["summary"]["overall"],
        "summary_sha256": sha256(result_path),
        "test_evaluated": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
