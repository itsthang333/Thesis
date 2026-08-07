from __future__ import annotations

"""Run one matched end-to-end G4 E3 SAM-v1 backbone arm.

Only the official SAM model type/checkpoint changes.  Both proposal supplies,
gallery merge/dedup/cap, frozen G1 checkpoint, equal percentile-rank fusion,
and spatial evaluator remain identical.  Candidate generation and final
choices are fully frozen before validation polygons are opened.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_320_SPLIT_SHA = (
    "7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c"
)
CLASSIFIER_320_SHA = (
    "ca630ddf816c1b6a55fab9b99fe824877bba9a83905ce71fd20cf9c2b1640621"
)
CLASSIFIER_448_SHA = (
    "b40dc5ec0f601ea7392fd0e8ed0be5f1e7cd66ad07d654392db516a0766d451e"
)
G1_SHA = "634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c"
RAD_CONFIG_SHA = (
    "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede"
)
RAD_PREPROCESSOR_SHA = (
    "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56"
)
RAD_WEIGHT_SHA = (
    "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
)
TRANSFORMERS_WHEEL_SHA = (
    "99bbcddd6570f080aee81f67844f4b46c8025bbdbdb86eafb82cc7d6aaafb190"
)
TOKENIZERS_WHEEL_SHA = (
    "51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff"
)
HUB_WHEEL_SHA = (
    "7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d"
)
PROTOCOL_SHA = "c65e6771cc6e68fe51de39c19374cffab35180259e8eed40eead7eed4ff6fb74"
SAM_SHA = {
    "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
    "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_hash(
    root: Path,
    expected: str,
    *,
    names: tuple[str, ...] = (),
    suffix: str | None = None,
) -> Path:
    if names:
        candidates = [path for name in names for path in root.rglob(name)]
    elif suffix is not None:
        candidates = list(root.rglob(f"*{suffix}"))
    else:
        raise ValueError("unique_hash requires candidate names or a suffix")
    matches = [
        path for path in candidates
        if path.is_file() and sha256(path) == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one input with SHA-256 {expected}, found {len(matches)}"
        )
    return matches[0]


def unique_named(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def unique_project(root: Path) -> Path:
    matches = [
        path.parent for path in root.rglob("run_g4_e3_sam_backbone.py")
        if path.parent.name == "project"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one current G4 project, found {len(matches)}")
    return matches[0]


def btxrd_root(root: Path) -> Path:
    matches = sorted({
        path.parent for path in root.rglob("images")
        if path.is_dir() and path.parent.name == "BTXRD"
    })
    if len(matches) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {matches}")
    return matches[0]


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def install_g1_runtime(root: Path, *, cwd: Path, env: dict[str, str]) -> None:
    wheels = [
        unique_hash(root, TRANSFORMERS_WHEEL_SHA, names=("transformers-4.50.2-py3-none-any.whl",)),
        unique_hash(root, TOKENIZERS_WHEEL_SHA, names=("tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",)),
        unique_hash(root, HUB_WHEEL_SHA, names=("huggingface_hub-0.36.0-py3-none-any.whl",)),
    ]
    run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--force-reinstall", *map(str, wheels),
        ],
        cwd=cwd,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-model-type", choices=tuple(SAM_SHA), required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def _supply_command(
    *,
    project: Path,
    data: Path,
    split: Path,
    classifier: Path,
    sam: Path,
    sam_model_type: str,
    source_commit: str,
    output: Path,
    mode: str,
    external_root: Path | None = None,
    classifier_split: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(project / "run_rich_gallery_candidate_supply.py"),
        "--mode", mode,
        "--source-root", str(project.parent),
        "--data-root", str(data),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--classifier-checkpoint", str(classifier),
        "--expected-classifier-sha256", sha256(classifier),
        "--sam-checkpoint", str(sam),
        "--expected-sam-sha256", SAM_SHA[sam_model_type],
        "--sam-model-type", sam_model_type,
        "--source-commit", source_commit,
        "--protocol-sha256", PROTOCOL_SHA,
        "--output-dir", str(output),
        "--splits", "val",
    ]
    if classifier_split is not None:
        command.extend([
            "--classifier-split-manifest", str(classifier_split),
            "--expected-classifier-split-sha256", CLASSIFIER_320_SPLIT_SHA,
        ])
    if external_root is not None:
        external_manifest = external_root / "saliency_supply_manifest.json"
        command.extend([
            "--external-saliency-supply-root", str(external_root),
            "--expected-external-supply-manifest-sha256", sha256(external_manifest),
        ])
    return command


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    project = unique_project(input_root)
    source = project.parent
    split = source / "artifacts" / "data_audit" / "split_manifest.csv"
    if not split.is_file() or sha256(split) != SPLIT_SHA:
        raise ValueError("current source canonical split is missing or changed")
    classifier_320_split = unique_hash(
        input_root,
        CLASSIFIER_320_SPLIT_SHA,
        names=("split_manifest.csv", "classifier_split_manifest_7b167.csv"),
    )
    classifier_320 = unique_hash(
        input_root,
        CLASSIFIER_320_SHA,
        names=("best_classifier_ca630d.pt", "best_classifier.pt"),
    )
    classifier_448 = unique_hash(
        input_root,
        CLASSIFIER_448_SHA,
        names=("best_classifier448.pt", "best_classifier.pt"),
    )
    sam_names = {
        "vit_b": ("sam_vit_b_01ec64.pth",),
        "vit_l": ("sam_vit_l_0b3195.pth",),
        "vit_h": ("sam_vit_h_4b8939.pth",),
    }
    sam = unique_hash(
        input_root, SAM_SHA[args.sam_model_type], names=sam_names[args.sam_model_type]
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
        or external.get("test_images_read") != 0
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

    arm_root = working / f"g4_e3_{args.sam_model_type}"
    arm_root.mkdir(parents=True, exist_ok=False)
    anchor = arm_root / "anchor"
    addition = arm_root / "addition"
    run(
        _supply_command(
            project=project, data=btxrd_root(input_root), split=split,
            classifier=classifier_320, classifier_split=classifier_320_split,
            sam=sam, sam_model_type=args.sam_model_type,
            source_commit=args.source_commit, output=anchor, mode="anchor",
            external_root=external_root,
        ),
        cwd=source,
        env=env,
    )
    run(
        _supply_command(
            project=project, data=btxrd_root(input_root), split=split,
            classifier=classifier_448, sam=sam,
            sam_model_type=args.sam_model_type, source_commit=args.source_commit,
            output=addition, mode="addition",
        ),
        cwd=source,
        env=env,
    )
    anchor_manifest = json.loads(
        (anchor / "candidate_supply_manifest.json").read_text(encoding="utf-8")
    )
    addition_manifest = json.loads(
        (addition / "candidate_supply_manifest.json").read_text(encoding="utf-8")
    )
    anchor_val = anchor_manifest["splits"]["val"]
    addition_val = addition_manifest["splits"]["val"]

    merge = arm_root / "gallery"
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
        "--output-dir", str(merge),
    ], cwd=source, env=env)
    contract = json.loads((merge / "gallery_merge_contract.json").read_text(encoding="utf-8"))

    install_g1_runtime(input_root, cwd=working, env=env)
    score = arm_root / "scores"
    run([
        sys.executable, str(project / "score_final_rich_gallery.py"),
        "--split", "val",
        "--dataset-root", str(btxrd_root(input_root)),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--model-dir", str(rad_dir),
        "--expected-config-sha256", RAD_CONFIG_SHA,
        "--expected-preprocessor-sha256", RAD_PREPROCESSOR_SHA,
        "--expected-weight-sha256", RAD_WEIGHT_SHA,
        "--candidate-root", str(merge),
        "--candidate-manifest-sha256", str(contract["output_manifest_sha256"]),
        "--pseudo-manifest-sha256", str(contract["anchor_pseudo_manifest_sha256"]),
        "--g1-checkpoint", str(g1),
        "--expected-g1-checkpoint-sha256", G1_SHA,
        "--source-commit", args.source_commit,
        "--protocol-sha256", PROTOCOL_SHA,
        "--output-dir", str(score),
    ], cwd=source, env=env)
    score_freeze = sha256(score / "diagnostic_freeze.json")

    choices = arm_root / "choices"
    run([
        sys.executable, str(project / "freeze_final_rich_gallery.py"),
        "--split", "val",
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--g1-diagnostic-root", str(score),
        "--expected-g1-freeze-sha256", score_freeze,
        "--candidate-root", str(merge),
        "--expected-candidate-manifest-sha256", str(contract["output_manifest_sha256"]),
        "--expected-pseudo-manifest-sha256", str(contract["anchor_pseudo_manifest_sha256"]),
        "--output-dir", str(choices),
    ], cwd=source, env=env)
    choice_freeze = sha256(choices / "prediction_freeze.json")

    evaluation = arm_root / "evaluation"
    run([
        sys.executable, str(project / "evaluate_final_rich_gallery.py"),
        "--split", "val",
        "--allow-validation-ablation",
        "--dataset-root", str(btxrd_root(input_root)),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--selection-root", str(choices),
        "--expected-selection-freeze-sha256", choice_freeze,
        "--candidate-root", str(merge),
        "--output-dir", str(evaluation),
    ], cwd=source, env=env)
    evaluation_summary = json.loads(
        (evaluation / "summary.json").read_text(encoding="utf-8")
    )
    resource_files = [
        anchor / "val" / "resource_metrics.json",
        addition / "val" / "resource_metrics.json",
    ]
    resources = [json.loads(path.read_text(encoding="utf-8")) for path in resource_files]
    result = {
        "schema_version": 1,
        "study": "G4 E3 matched end-to-end SAM-v1 backbone ablation",
        "sam_model_type": args.sam_model_type,
        "sam_checkpoint_sha256": SAM_SHA[args.sam_model_type],
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": args.source_commit,
        "split_sha256": SPLIT_SHA,
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "summary": evaluation_summary["summary"],
        "resource_metrics": {
            "candidate_generation_elapsed_seconds": float(sum(
                item["elapsed_seconds"] for item in resources
            )),
            "candidate_generation_seconds_per_image_per_supply_sum": float(sum(
                item["seconds_per_processed_image"] for item in resources
            )),
            "peak_memory_allocated_bytes": int(max(
                values["peak_memory_allocated_bytes"]
                for item in resources for values in item["cuda"].values()
            )),
            "peak_memory_reserved_bytes": int(max(
                values["peak_memory_reserved_bytes"]
                for item in resources for values in item["cuda"].values()
            )),
            "merged_gallery_bytes": int(sum(
                path.stat().st_size for path in merge.rglob("*") if path.is_file()
            )),
            "total_arm_elapsed_seconds": float(time.perf_counter() - started),
        },
        "gallery_contract": contract,
        "choices_frozen_before_spatial_gt": True,
        "spatial_annotations_opened": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    result_path = arm_root / "summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "sam_model_type": args.sam_model_type,
        "summary_sha256": sha256(result_path),
        "overall": result["summary"]["overall"],
        "test_evaluated": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
