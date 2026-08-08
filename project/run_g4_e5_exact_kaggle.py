from __future__ import annotations

"""Run the complete exact G4 E5 study on one private/offline Kaggle GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_320_SPLIT_SHA = "7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c"
CLASSIFIER_320_SHA = "ca630ddf816c1b6a55fab9b99fe824877bba9a83905ce71fd20cf9c2b1640621"
CLASSIFIER_448_SHA = "b40dc5ec0f601ea7392fd0e8ed0be5f1e7cd66ad07d654392db516a0766d451e"
SAM_B_SHA = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
G1_SHA = "634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c"
RAD_CONFIG_SHA = "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede"
RAD_PREPROCESSOR_SHA = "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56"
RAD_WEIGHT_SHA = "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
TRANSFORMERS_WHEEL_SHA = "99bbcddd6570f080aee81f67844f4b46c8025bbdbdb86eafb82cc7d6aaafb190"
TOKENIZERS_WHEEL_SHA = "51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff"
HUB_WHEEL_SHA = "7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d"
PROTOCOL_SHA = "1849f253b109cc43eddecb77718ab11fca64c5ac2c4d1508dc9f3d5414c5bf9f"
EXTERNAL_MANIFEST_SHA = "6f18304592d3f88328c8a7e84f2ac2f5ee0d5c2a0ed7e07597bf827cd7305111"
MULTI_ANCHOR_MANIFEST_SHA = "96d4e0930962ad53257726122616ae02a70d41beae1f7449e81b10b1b5ab06b3"
MULTI_ANCHOR_PSEUDO_SHA = "450c0f9cf7f7114da98df9002a54f3620ae952026bcf4113bbd47797f66e1a7c"
MULTI_ADDITION_MANIFEST_SHA = "7c887830dab29624f73569f43888104701768527a4edbcb720300295c30bd436"
MULTI_ADDITION_PSEUDO_SHA = "7f973948f1a027a2030bffc8ee2d6e305fb483d3ea92c5cb6c8c5db92fb257d4"
POST_MANIFEST_SHA = "30e734e223839ee2ed7b445c71b3731a39f0549ee88e2ba77f7bf722d8d9943d"
POST_PSEUDO_SHA = "450c0f9cf7f7114da98df9002a54f3620ae952026bcf4113bbd47797f66e1a7c"
POST_G1_FREEZE_SHA = "5942a6df949f51fc659313416a9c1156db56f6dad8b26b6957a58fc0ad6138ff"
BASELINE_CHOICE_FREEZE_SHA = "a75c0388346b6a1a3ab94f3ddd700a2e495c36be257b63712e38fc451784a620"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_hash(root: Path, expected: str, *, names: tuple[str, ...]) -> Path:
    matches = sorted(
        {
        path
        for name in names
        for path in root.rglob(name)
        if path.is_file() and sha256(path) == expected
        },
        key=lambda path: path.as_posix(),
    )
    # The Kaggle payload may attach the same immutable, content-addressed input
    # through more than one dataset.  Multiple byte-identical copies are
    # scientifically equivalent because every candidate has already been
    # verified against ``expected``.  Fail only when no exact copy exists and
    # select deterministically otherwise.
    if not matches:
        raise RuntimeError(f"expected input SHA-256 {expected}, found no exact copy")
    return matches[0]


def unique_project(root: Path) -> Path:
    matches = [
        path.parent
        for path in root.rglob("run_g4_e5_exact_kaggle.py")
        if path.parent.name == "project"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one G4 E5 project, found {matches}")
    return matches[0]


def canonical_split(root: Path) -> Path:
    candidates = [
        path
        for name in ("canonical_split_manifest_85511.csv", "split_manifest.csv")
        for path in root.rglob(name)
        if path.is_file() and sha256(path) == SPLIT_SHA
    ]
    preferred = [path for path in candidates if "btxrd-g4-e2-inputs-20260806" in path.parts]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"canonical split is ambiguous: {candidates}")


def btxrd_root(root: Path) -> Path:
    matches = sorted(
        {
            path.parent
            for path in root.rglob("images")
            if path.is_dir() and path.parent.name == "BTXRD"
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {matches}")
    return matches[0]


def frozen_supply(
    root: Path,
    *,
    mode: str,
    manifest_sha: str,
    pseudo_sha: str,
) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in root.rglob("candidate_supply_manifest.json"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        val = payload.get("splits", {}).get("val", {})
        if (
            payload.get("mode") == mode
            and val.get("candidate_manifest_sha256") == manifest_sha
            and val.get("pseudo_manifest_sha256") == pseudo_sha
            and payload.get("spatial_ground_truth_read") is False
            and payload.get("test_images_read") == 0
            and payload.get("test_evaluated") is False
        ):
            matches.append((path.parent, payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen {mode} supply, found {matches}")
    return matches[0]


def candidate_root(
    root: Path,
    *,
    manifest_sha: str,
    pseudo_sha: str,
) -> Path:
    matches = []
    for path in root.rglob("candidate_diagnostics_summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("manifest_sha256") == manifest_sha
            and payload.get("pseudo_manifest_sha256") == pseudo_sha
            and payload.get("ground_truth_loaded_during_generation") is False
        ):
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one candidate root {manifest_sha}, found {matches}")
    return matches[0]


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def install_runtime(root: Path, *, cwd: Path, env: dict[str, str]) -> None:
    wheels = [
        unique_hash(root, TRANSFORMERS_WHEEL_SHA, names=("transformers-4.50.2-py3-none-any.whl",)),
        unique_hash(root, TOKENIZERS_WHEEL_SHA, names=("tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",)),
        unique_hash(root, HUB_WHEEL_SHA, names=("huggingface_hub-0.36.0-py3-none-any.whl",)),
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
            *map(str, wheels),
        ],
        cwd=cwd,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def supply_command(
    *,
    project: Path,
    data: Path,
    split: Path,
    classifier: Path,
    sam: Path,
    source_commit: str,
    output: Path,
    mode: str,
    classifier_split: Path | None = None,
    external_root: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(project / "run_rich_gallery_candidate_supply.py"),
        "--mode",
        mode,
        "--source-root",
        str(project.parent),
        "--data-root",
        str(data),
        "--split-manifest",
        str(split),
        "--expected-split-sha256",
        SPLIT_SHA,
        "--classifier-checkpoint",
        str(classifier),
        "--expected-classifier-sha256",
        sha256(classifier),
        "--sam-checkpoint",
        str(sam),
        "--expected-sam-sha256",
        SAM_B_SHA,
        "--sam-model-type",
        "vit_b",
        "--sam-single-mask",
        "--source-commit",
        source_commit,
        "--protocol-sha256",
        PROTOCOL_SHA,
        "--output-dir",
        str(output),
        "--splits",
        "val",
    ]
    if classifier_split is not None:
        command.extend(
            [
                "--classifier-split-manifest",
                str(classifier_split),
                "--expected-classifier-split-sha256",
                CLASSIFIER_320_SPLIT_SHA,
            ]
        )
    if external_root is not None:
        command.extend(
            [
                "--external-saliency-supply-root",
                str(external_root),
                "--expected-external-supply-manifest-sha256",
                EXTERNAL_MANIFEST_SHA,
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    project = unique_project(input_root)
    source = project.parent
    protocol = source / "artifacts" / "final_pipeline" / "g4" / "e5_exact_protocol.json"
    if sha256(protocol) != PROTOCOL_SHA:
        raise ValueError("G4 E5 protocol file changed")
    split = canonical_split(input_root)
    data = btxrd_root(input_root)
    classifier_320_split = unique_hash(
        input_root,
        CLASSIFIER_320_SPLIT_SHA,
        names=("split_manifest.csv", "classifier_split_manifest_7b167.csv"),
    )
    classifier_320 = unique_hash(
        input_root, CLASSIFIER_320_SHA, names=("best_classifier_ca630d.pt", "best_classifier.pt")
    )
    classifier_448 = unique_hash(
        input_root, CLASSIFIER_448_SHA, names=("best_classifier448.pt", "best_classifier.pt")
    )
    sam = unique_hash(input_root, SAM_B_SHA, names=("sam_vit_b_01ec64.pth",))
    rad_weight = unique_hash(input_root, RAD_WEIGHT_SHA, names=("model.safetensors",))
    rad_dir = rad_weight.parent
    if sha256(rad_dir / "config.json") != RAD_CONFIG_SHA or sha256(
        rad_dir / "preprocessor_config.json"
    ) != RAD_PREPROCESSOR_SHA:
        raise ValueError("RAD-DINO snapshot changed")
    g1 = unique_hash(input_root, G1_SHA, names=("rad_dino_mask_bag_mil.pt",))
    external_manifest = unique_hash(
        input_root, EXTERNAL_MANIFEST_SHA, names=("saliency_supply_manifest.json",)
    )
    external_root = external_manifest.parent
    external = json.loads(external_manifest.read_text(encoding="utf-8"))
    if (
        int(external.get("splits", {}).get("val", {}).get("images", -1)) != 371
        or external.get("spatial_ground_truth_read") is not False
        or external.get("test_images_read") != 0
        or external.get("test_evaluated") is not False
    ):
        raise ValueError("external saliency supply violates the E5 boundary")
    multi_anchor_root, multi_anchor = frozen_supply(
        input_root,
        mode="anchor",
        manifest_sha=MULTI_ANCHOR_MANIFEST_SHA,
        pseudo_sha=MULTI_ANCHOR_PSEUDO_SHA,
    )
    multi_addition_root, multi_addition = frozen_supply(
        input_root,
        mode="addition",
        manifest_sha=MULTI_ADDITION_MANIFEST_SHA,
        pseudo_sha=MULTI_ADDITION_PSEUDO_SHA,
    )
    post_root = candidate_root(
        input_root, manifest_sha=POST_MANIFEST_SHA, pseudo_sha=POST_PSEUDO_SHA
    )
    post_g1_freeze = unique_hash(
        input_root, POST_G1_FREEZE_SHA, names=("diagnostic_freeze.json",)
    )
    post_g1_root = post_g1_freeze.parent
    baseline_choice_freeze = unique_hash(
        input_root, BASELINE_CHOICE_FREEZE_SHA, names=("prediction_freeze.json",)
    )
    baseline_choice_root = baseline_choice_freeze.parent
    sam_package = unique_hash(
        input_root,
        "2d8c5a963705d9408cbaef05f57c6dedc708ddabd45dbdbc91c060bab930a0c0",
        names=("automatic_mask_generator.py",),
    ).parent.parent

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(project), str(sam_package), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "BTXRD_DISABLE_TQDM": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    output = working / "g4_e5_exact"
    output.mkdir(parents=True, exist_ok=False)
    single_anchor = output / "single_anchor"
    single_addition = output / "single_addition"
    run(
        supply_command(
            project=project,
            data=data,
            split=split,
            classifier=classifier_320,
            classifier_split=classifier_320_split,
            sam=sam,
            source_commit=args.source_commit,
            output=single_anchor,
            mode="anchor",
            external_root=external_root,
        ),
        cwd=source,
        env=env,
    )
    run(
        supply_command(
            project=project,
            data=data,
            split=split,
            classifier=classifier_448,
            sam=sam,
            source_commit=args.source_commit,
            output=single_addition,
            mode="addition",
        ),
        cwd=source,
        env=env,
    )
    single_anchor_manifest = json.loads(
        (single_anchor / "candidate_supply_manifest.json").read_text(encoding="utf-8")
    )["splits"]["val"]
    single_addition_manifest = json.loads(
        (single_addition / "candidate_supply_manifest.json").read_text(encoding="utf-8")
    )["splits"]["val"]

    pre = output / "pre_dedup"
    run(
        [
            sys.executable,
            str(project / "build_g4_e5_pre_dedup_gallery.py"),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA,
            "--multimask-anchor-root",
            str(multi_anchor_root / "val"),
            "--expected-multimask-anchor-manifest-sha256",
            MULTI_ANCHOR_MANIFEST_SHA,
            "--expected-multimask-anchor-pseudo-sha256",
            MULTI_ANCHOR_PSEUDO_SHA,
            "--multimask-addition-root",
            str(multi_addition_root / "val"),
            "--expected-multimask-addition-manifest-sha256",
            MULTI_ADDITION_MANIFEST_SHA,
            "--expected-multimask-addition-pseudo-sha256",
            MULTI_ADDITION_PSEUDO_SHA,
            "--single-anchor-root",
            str(single_anchor / "val"),
            "--expected-single-anchor-manifest-sha256",
            str(single_anchor_manifest["candidate_manifest_sha256"]),
            "--expected-single-anchor-pseudo-sha256",
            str(single_anchor_manifest["pseudo_manifest_sha256"]),
            "--single-addition-root",
            str(single_addition / "val"),
            "--expected-single-addition-manifest-sha256",
            str(single_addition_manifest["candidate_manifest_sha256"]),
            "--expected-single-addition-pseudo-sha256",
            str(single_addition_manifest["pseudo_manifest_sha256"]),
            "--post-dedup-root",
            str(post_root),
            "--expected-post-dedup-manifest-sha256",
            POST_MANIFEST_SHA,
            "--expected-post-dedup-pseudo-sha256",
            POST_PSEUDO_SHA,
            "--output-dir",
            str(pre),
        ],
        cwd=source,
        env=env,
    )
    pre_contract = json.loads((pre / "g4_e5_pre_dedup_contract.json").read_text(encoding="utf-8"))
    pre_contract_sha = sha256(pre / "g4_e5_pre_dedup_contract.json")

    install_runtime(input_root, cwd=working, env=env)
    pre_g1 = output / "pre_g1"
    run(
        [
            sys.executable,
            str(project / "score_final_rich_gallery.py"),
            "--split",
            "val",
            "--dataset-root",
            str(data),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA,
            "--model-dir",
            str(rad_dir),
            "--expected-config-sha256",
            RAD_CONFIG_SHA,
            "--expected-preprocessor-sha256",
            RAD_PREPROCESSOR_SHA,
            "--expected-weight-sha256",
            RAD_WEIGHT_SHA,
            "--candidate-root",
            str(pre),
            "--candidate-manifest-sha256",
            str(pre_contract["output_manifest_sha256"]),
            "--pseudo-manifest-sha256",
            str(pre_contract["output_pseudo_manifest_sha256"]),
            "--g1-checkpoint",
            str(g1),
            "--expected-g1-checkpoint-sha256",
            G1_SHA,
            "--source-commit",
            args.source_commit,
            "--protocol-sha256",
            PROTOCOL_SHA,
            "--output-dir",
            str(pre_g1),
        ],
        cwd=source,
        env=env,
    )
    pre_g1_freeze_sha = sha256(pre_g1 / "diagnostic_freeze.json")

    stage = output / "stage"
    run(
        [
            sys.executable,
            str(project / "freeze_g4_e5_exact_choices.py"),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA,
            "--pre-dedup-root",
            str(pre),
            "--expected-pre-dedup-manifest-sha256",
            str(pre_contract["output_manifest_sha256"]),
            "--expected-pre-dedup-pseudo-sha256",
            str(pre_contract["output_pseudo_manifest_sha256"]),
            "--expected-pre-dedup-contract-sha256",
            pre_contract_sha,
            "--post-dedup-root",
            str(post_root),
            "--expected-post-dedup-manifest-sha256",
            POST_MANIFEST_SHA,
            "--expected-post-dedup-pseudo-sha256",
            POST_PSEUDO_SHA,
            "--single-anchor-root",
            str(single_anchor / "val"),
            "--expected-single-anchor-manifest-sha256",
            str(single_anchor_manifest["candidate_manifest_sha256"]),
            "--expected-single-anchor-pseudo-sha256",
            str(single_anchor_manifest["pseudo_manifest_sha256"]),
            "--single-addition-root",
            str(single_addition / "val"),
            "--expected-single-addition-manifest-sha256",
            str(single_addition_manifest["candidate_manifest_sha256"]),
            "--expected-single-addition-pseudo-sha256",
            str(single_addition_manifest["pseudo_manifest_sha256"]),
            "--pre-g1-root",
            str(pre_g1),
            "--expected-pre-g1-freeze-sha256",
            pre_g1_freeze_sha,
            "--post-g1-root",
            str(post_g1_root),
            "--expected-post-g1-freeze-sha256",
            POST_G1_FREEZE_SHA,
            "--baseline-choice-root",
            str(baseline_choice_root),
            "--expected-baseline-choice-freeze-sha256",
            BASELINE_CHOICE_FREEZE_SHA,
            "--output-dir",
            str(stage),
        ],
        cwd=source,
        env=env,
    )
    choice_freeze_sha = sha256(stage / "choices" / "g4_choice_freeze.json")
    evaluation = output / "evaluation"
    run(
        [
            sys.executable,
            str(project / "evaluate_g4_offline_ablations.py"),
            "--dataset-root",
            str(data),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA,
            "--candidate-root",
            str(stage / "candidate_gallery"),
            "--choice-root",
            str(stage / "choices"),
            "--expected-choice-freeze-sha256",
            choice_freeze_sha,
            "--output-dir",
            str(evaluation),
            "--primary-grid",
            "native",
            "--bootstrap-iterations",
            "10000",
            "--bootstrap-seed",
            "20260808",
        ],
        cwd=source,
        env=env,
    )
    audit_path = output / "independent_audit.json"
    run(
        [
            sys.executable,
            str(project / "audit_g4_e5_exact_output.py"),
            "--split-manifest",
            str(split),
            "--expected-split-sha256",
            SPLIT_SHA,
            "--candidate-root",
            str(stage / "candidate_gallery"),
            "--choice-root",
            str(stage / "choices"),
            "--expected-choice-freeze-sha256",
            choice_freeze_sha,
            "--evaluation-root",
            str(evaluation),
            "--output",
            str(audit_path),
        ],
        cwd=source,
        env=env,
    )
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    resource_paths = [
        single_anchor / "val" / "resource_metrics.json",
        single_addition / "val" / "resource_metrics.json",
    ]
    resource_rows = [json.loads(path.read_text(encoding="utf-8")) for path in resource_paths]
    result = {
        "schema_version": 1,
        "study": "G4 E5 exact gallery richness, deduplication, and cap necessity",
        "source_commit": args.source_commit,
        "protocol_sha256": PROTOCOL_SHA,
        "split_sha256": SPLIT_SHA,
        "choice_freeze_sha256": choice_freeze_sha,
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "independent_audit_sha256": sha256(audit_path),
        "arms": summary["summaries"],
        "resources": {
            "single_mask_generation_elapsed_seconds": float(
                sum(row["elapsed_seconds"] for row in resource_rows)
            ),
            "single_mask_peak_allocated_bytes": int(
                max(
                    item["peak_memory_allocated_bytes"]
                    for row in resource_rows
                    for item in row["cuda"].values()
                )
            ),
            "single_mask_peak_reserved_bytes": int(
                max(
                    item["peak_memory_reserved_bytes"]
                    for row in resource_rows
                    for item in row["cuda"].values()
                )
            ),
            "single_mask_output_bytes": int(
                sum(path.stat().st_size for root in (single_anchor, single_addition) for path in root.rglob("*") if path.is_file())
            ),
            "pre_dedup_output_bytes": int(
                sum(path.stat().st_size for path in pre.rglob("*") if path.is_file())
            ),
            "unified_stage_output_bytes": int(
                sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
            ),
            "total_elapsed_seconds": float(time.perf_counter() - started),
        },
        "choices_frozen_before_spatial_gt": True,
        "spatial_annotations_opened": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    result_path = output / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "summary_sha256": sha256(result_path),
                "independent_audit_sha256": result["independent_audit_sha256"],
                "arm_dice": {
                    arm: values["mean_tumor_dice"] for arm, values in result["arms"].items()
                },
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
