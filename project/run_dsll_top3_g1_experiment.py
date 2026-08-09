from __future__ import annotations

"""Run the frozen DSLL seven-source gallery, G1 selector, and val Dice endpoint.

The wrapper is deliberately a single bounded Kaggle job.  It consumes the
already-frozen DSLL and classifier-448 supplies, merges them, trains G1 on the
canonical train split, freezes validation choices, and only then opens the
validation polygons for the final endpoint.  Test images are never enumerated
or read.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
PROTOCOL_SHA256 = "5c2481dad568ad3851c73f48396be1b514cea36783fa7bc605191c5ab51eec26"
CLASSIFIER448_SHA256 = "b40dc5ec0f601ea7392fd0e8ed0be5f1e7cd66ad07d654392db516a0766d451e"
RAD_CONFIG_SHA256 = "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede"
RAD_PREPROCESSOR_SHA256 = "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56"
RAD_WEIGHT_SHA256 = "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
TRANSFORMERS_WHEEL_SHA256 = "99bbcddd6570f080aee81f67844f4b46c8025bbdbdb86eafb82cc7d6aaafb190"
TOKENIZERS_WHEEL_SHA256 = "51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff"
HUB_WHEEL_SHA256 = "7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(root: Path, name: str, expected: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_file() and sha256(p) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} with SHA {expected}, found {matches}")
    return matches[0]


def run(command: list[str], *, cwd: Path) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def dataset_root(input_root: Path) -> Path:
    roots = sorted({
        p.parent
        for p in input_root.rglob("images")
        if p.is_dir() and (p.parent / "Annotations").is_dir()
    })
    if len(roots) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {roots}")
    return roots[0]


def classifier_supply(input_root: Path) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in input_root.rglob("candidate_supply_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("mode") == "addition"
            and payload.get("classifier_checkpoint_sha256") == CLASSIFIER448_SHA256
        ):
            matches.append((path.parent, payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one locked classifier448 supply, found {matches}")
    root, payload = matches[0]
    if (
        payload.get("spatial_ground_truth_read") is not False
        or payload.get("test_images_read") != 0
        or payload.get("test_evaluated") is not False
    ):
        raise RuntimeError("classifier448 supply violates no-GT/no-test contract")
    return root, payload


def install_runtime(input_root: Path, working: Path) -> None:
    wheels = [
        exact_file(
            input_root,
            "transformers-4.50.2-py3-none-any.whl",
            TRANSFORMERS_WHEEL_SHA256,
        ),
        exact_file(
            input_root,
            "tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            TOKENIZERS_WHEEL_SHA256,
        ),
        exact_file(
            input_root,
            "huggingface_hub-0.36.0-py3-none-any.whl",
            HUB_WHEEL_SHA256,
        ),
    ]
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *[str(path) for path in wheels],
        ],
        cwd=working,
    )


def merge_split(
    *,
    project: Path,
    split_file: Path,
    split: str,
    dsll_root: Path,
    dsll_manifest: dict[str, object],
    classifier_root: Path,
    classifier_manifest: dict[str, object],
    output: Path,
) -> tuple[str, str]:
    addition = classifier_manifest["splits"][split]
    dsll_candidate_sha = sha256(dsll_root / "candidate_diagnostics_manifest.csv")
    dsll_pseudo_sha = sha256(dsll_root / "pseudo_mask_manifest.csv")
    if dsll_candidate_sha != dsll_manifest[f"{split}_candidate_manifest_sha256"]:
        raise RuntimeError(f"DSLL {split} candidate manifest differs")
    if dsll_pseudo_sha != dsll_manifest[f"{split}_pseudo_manifest_sha256"]:
        raise RuntimeError(f"DSLL {split} pseudo manifest differs")
    run(
        [
            sys.executable,
            str(project / "merge_frozen_candidate_galleries.py"),
            "--split-manifest", str(split_file),
            "--expected-split-sha256", SPLIT_SHA256,
            "--split", split,
            "--anchor-root", str(dsll_root),
            "--anchor-candidate-manifest-sha256", dsll_candidate_sha,
            "--anchor-pseudo-manifest-sha256", dsll_pseudo_sha,
            "--addition-root", str(classifier_root / split),
            "--addition-candidate-manifest-sha256", str(addition["candidate_manifest_sha256"]),
            "--addition-pseudo-manifest-sha256", str(addition["pseudo_manifest_sha256"]),
            "--addition-namespace", "classifier448",
            "--protocol-sha256", PROTOCOL_SHA256,
            "--output-dir", str(output),
        ],
        cwd=project.parent,
    )
    contract = json.loads((output / "gallery_merge_contract.json").read_text(encoding="utf-8"))
    if int(contract.get("maximum_candidates", -1)) > 567:
        raise RuntimeError(f"merged {split} gallery exceeds cap 567")
    return str(contract["output_manifest_sha256"]), dsll_pseudo_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--merged-supply-archive-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    os.environ.update({
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    input_root = args.input_root.resolve()
    working = args.working_root.resolve()
    working.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent.parent
    project = source / "project"
    split_file = exact_file(
        input_root, "canonical_split_manifest_85511.csv", SPLIT_SHA256
    )
    archive = exact_file(
        input_root,
        "dsll_merged_supply.zip",
        args.merged_supply_archive_sha256,
    )
    supply_root = working / "dsll_supply"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(supply_root)
    dsll_manifest = {
        "train_candidate_manifest_sha256": "a513e9230eef8401dd73e64623618aee26ca6e75d496816e9b020db8ec008e3e",
        "train_pseudo_manifest_sha256": "16def408e874cc1549715302124ce8d5ea5abf313b1e1bf68977dc2aca4f6f7f",
        "val_candidate_manifest_sha256": "c9757f7ffd76bfa776571ca2f00661b2fbf91a387271ee755d197ca5317a3e4c",
        "val_pseudo_manifest_sha256": "45ea543a2594c5fb8c3f8091e602b7dc923a657874e6475b2a6d89a186339daa",
    }
    classifier_root, classifier_manifest = classifier_supply(input_root)
    install_runtime(input_root, working)
    model_weight = exact_file(input_root, "model.safetensors", RAD_WEIGHT_SHA256)
    model_dir = model_weight.parent
    if sha256(model_dir / "config.json") != RAD_CONFIG_SHA256:
        raise RuntimeError("RAD-DINO config differs")
    if sha256(model_dir / "preprocessor_config.json") != RAD_PREPROCESSOR_SHA256:
        raise RuntimeError("RAD-DINO preprocessor differs")
    data = dataset_root(input_root)
    train_gallery = working / "gallery7_train"
    val_gallery = working / "gallery7_val"
    train_candidate_sha, train_pseudo_sha = merge_split(
        project=project,
        split_file=split_file,
        split="train",
        dsll_root=supply_root / "merged_train",
        dsll_manifest=dsll_manifest,
        classifier_root=classifier_root,
        classifier_manifest=classifier_manifest,
        output=train_gallery,
    )
    val_candidate_sha, val_pseudo_sha = merge_split(
        project=project,
        split_file=split_file,
        split="val",
        dsll_root=supply_root / "merged_val",
        dsll_manifest=dsll_manifest,
        classifier_root=classifier_root,
        classifier_manifest=classifier_manifest,
        output=val_gallery,
    )

    g1 = working / "dsll_g1_seed42"
    run(
        [
            sys.executable,
            str(project / "run_rad_dino_mask_bag_mil_probe.py"),
            "--dataset-root", str(data),
            "--split-manifest", str(split_file),
            "--expected-split-sha256", SPLIT_SHA256,
            "--model-dir", str(model_dir),
            "--expected-config-sha256", RAD_CONFIG_SHA256,
            "--expected-preprocessor-sha256", RAD_PREPROCESSOR_SHA256,
            "--expected-weight-sha256", RAD_WEIGHT_SHA256,
            "--train-candidate-root", str(train_gallery),
            "--train-candidate-manifest-sha256", train_candidate_sha,
            "--train-pseudo-manifest-sha256", train_pseudo_sha,
            "--val-candidate-root", str(val_gallery),
            "--val-candidate-manifest-sha256", val_candidate_sha,
            "--val-pseudo-manifest-sha256", val_pseudo_sha,
            "--source-commit", args.source_commit,
            "--protocol-sha256", PROTOCOL_SHA256,
            "--output-dir", str(g1),
            "--maximum-candidates", "567",
            "--rich-gallery-union",
            "--seed", "42",
        ],
        cwd=source,
    )
    checkpoint = g1 / "rad_dino_mask_bag_mil.pt"
    checkpoint_sha = sha256(checkpoint)

    scores = working / "dsll_g1_val_scores"
    run(
        [
            sys.executable,
            str(project / "score_final_rich_gallery.py"),
            "--split", "val",
            "--dataset-root", str(data),
            "--split-manifest", str(split_file),
            "--expected-split-sha256", SPLIT_SHA256,
            "--model-dir", str(model_dir),
            "--expected-config-sha256", RAD_CONFIG_SHA256,
            "--expected-preprocessor-sha256", RAD_PREPROCESSOR_SHA256,
            "--expected-weight-sha256", RAD_WEIGHT_SHA256,
            "--candidate-root", str(val_gallery),
            "--candidate-manifest-sha256", val_candidate_sha,
            "--pseudo-manifest-sha256", val_pseudo_sha,
            "--g1-checkpoint", str(checkpoint),
            "--expected-g1-checkpoint-sha256", checkpoint_sha,
            "--source-commit", args.source_commit,
            "--protocol-sha256", PROTOCOL_SHA256,
            "--output-dir", str(scores),
            "--maximum-candidates", "567",
            "--seed", "42",
        ],
        cwd=source,
    )
    choices = working / "dsll_g1_val_choices"
    score_freeze_sha = sha256(scores / "diagnostic_freeze.json")
    run(
        [
            sys.executable,
            str(project / "freeze_final_rich_gallery.py"),
            "--split", "val",
            "--split-manifest", str(split_file),
            "--expected-split-sha256", SPLIT_SHA256,
            "--g1-diagnostic-root", str(scores),
            "--expected-g1-freeze-sha256", score_freeze_sha,
            "--candidate-root", str(val_gallery),
            "--expected-candidate-manifest-sha256", val_candidate_sha,
            "--expected-pseudo-manifest-sha256", val_pseudo_sha,
            "--output-dir", str(choices),
        ],
        cwd=source,
    )
    prediction_freeze_sha = sha256(choices / "prediction_freeze.json")

    evaluation = working / "dsll_top3_evaluation"
    run(
        [
            sys.executable,
            str(project / "evaluate_dsll_top3_experiment.py"),
            "--dataset-root", str(data),
            "--split-manifest", str(split_file),
            "--expected-split-sha256", SPLIT_SHA256,
            "--candidate-root", str(val_gallery),
            "--expected-candidate-manifest-sha256", val_candidate_sha,
            "--expected-pseudo-manifest-sha256", val_pseudo_sha,
            "--selection-root", str(choices),
            "--expected-selection-freeze-sha256", prediction_freeze_sha,
            "--output-dir", str(evaluation),
        ],
        cwd=source,
    )

    compact = working / "dsll_top3_final"
    compact.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, compact / checkpoint.name)
    for source_dir in (g1, scores, choices, evaluation):
        target = compact / source_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for path in source_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".jsonl"}:
                shutil.copy2(path, target / path.name)
    archive_out = Path(shutil.make_archive(
        str(working / "dsll_top3_final"), "zip", root_dir=working, base_dir=compact.name
    ))
    receipt = {
        "schema_version": 1,
        "stage": "dsll_top3_g1_validation_experiment_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "merged_supply_archive_sha256": args.merged_supply_archive_sha256,
        "train_candidate_manifest_sha256": train_candidate_sha,
        "val_candidate_manifest_sha256": val_candidate_sha,
        "g1_checkpoint_sha256": checkpoint_sha,
        "score_freeze_sha256": score_freeze_sha,
        "prediction_freeze_sha256": prediction_freeze_sha,
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "archive_sha256": sha256(archive_out),
        "train_images": 2981,
        "validation_images": 371,
        "validation_tumor_images": 184,
        "validation_polygons_opened_after_prediction_freeze": True,
        "spatial_ground_truth_training": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (working / "dsll_top3_final_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
