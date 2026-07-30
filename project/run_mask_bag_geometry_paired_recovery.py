from __future__ import annotations

"""Recover the geometry-v3 audit with a same-gallery legacy control.

This orchestration layer never imports validation segmentation until both
complete prediction arms have been physically frozen.  Training consumes only
the frozen image-level labels and class-agnostic candidate payloads.
"""

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "btxrd_rad_dino_mask_bag_geometry_paired_v1"
TEMP = WORK / "mask_bag_geometry_paired_runtime"

SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
RECOVERY_PROTOCOL_RELATIVE = Path(
    "artifacts/research_protocols/"
    "rad_dino_mask_bag_mil_descriptor_geometry_v3_execution_correction_v3.json"
)
RECOVERY_PROTOCOL_SHA256 = (
    "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
)
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_SHA256 = (
    "f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5"
)
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
BASELINE_SHA256 = (
    "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540"
)
V6_EVALUATION_PER_IMAGE_SHA256 = (
    "94f19a3b7944706573fea03617196ff9a75b592c2b7d00f253ecfccd2253cc51"
)
V6_PAIRED_COMPARISON_SHA256 = (
    "5fde4512169a906d64f622ee6bb0426b5ea3294ebe9706a96a4ff6103a3d8716"
)
V6_MANIFEST_HASHES = {
    "train_candidate": (
        "7dfe43cc90585c58c48bff95dfe6884b84ae3e304c52b8f3bab3f18ea09d6a28"
    ),
    "train_pseudo": (
        "5d50bd39619e3e8859e86fdcbb80b8d9bcfc3de8b6d0a88d0011fc35ccb1dc77"
    ),
    "val_candidate": (
        "59b7828389aee2454460ad2c3733c18c3b1c486266635ae480969c8249693251"
    ),
    "val_pseudo": (
        "11779008871c53afe863173c526b5e2687394cdd37a04d3914287c77eeab458f"
    ),
}
RECOVERED_TRAIN_HASHES = {
    "candidate": (
        "ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1"
    ),
    "pseudo": (
        "5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21"
    ),
}
MODEL_ID = "microsoft/rad-dino"
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
MODEL_HASHES = {
    "config.json": "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede",
    "preprocessor_config.json": (
        "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56"
    ),
    "model.safetensors": (
        "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
    ),
}
EXPECTED_COHORT = {
    "validation": 371,
    "tumor": 184,
    "normal": 187,
    "small": 94,
    "medium": 72,
    "large": 18,
}
OPERATIONAL_GOALS = {
    "overall": 0.34024039,
    "small": 0.17895493,
    "medium": 0.51244178,
    "large": 0.49370336,
}
NUMERIC_TOLERANCE = {
    "cam_max": 2.0e-7,
    "cam_mean": 2.0e-7,
    "cam_std": 1.0e-7,
    "selection_score_min": 0.003,
    "selection_score_mean": 0.003,
    "selection_score_max": 2.0e-7,
    "support_area_ratio": 2.0e-5,
    "selected_area_ratio": 0.006,
}
FRACTIONAL_CHANGE_LIMIT = {
    "unique_prompt_points": 0.01,
    "unique_positive_prompt_points": 0.01,
    "above_threshold_candidates": 0.01,
    "selected_area_ratio": 0.01,
}
CANDIDATE_BYTE_FIELDS = {"diagnostic_sha256", "diagnostic_bytes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> str:
    rendered = subprocess.list2cmdline(command)
    print(f"$ {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(lines)


def find_unique_input(name: str, expected_hash: str) -> Path:
    matches: list[Path] = []
    for candidate in INPUT.rglob(name):
        if not candidate.is_file():
            continue
        normalized = candidate.as_posix()
        if "/thesis_source/" in normalized or "/.git/" in normalized:
            continue
        if sha256(candidate) == expected_hash:
            matches.append(candidate)
    resolved = {path.resolve() for path in matches}
    if len(resolved) != 1:
        raise RuntimeError(
            f"Expected one direct input {name} with hash {expected_hash}, "
            f"found {sorted(str(path) for path in resolved)}"
        )
    return next(iter(resolved))


def locate_dataset_root() -> Path:
    candidates = [
        INPUT / "datasets" / "itsthang333" / "btxrd-raw" / "BTXRD",
        INPUT / "btxrd-raw" / "BTXRD",
        *sorted(INPUT.glob("**/BTXRD")),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if "/thesis_source/" in candidate.as_posix():
            continue
        if (candidate / "images").is_dir():
            return candidate
    raise RuntimeError("BTXRD data root was not found on a direct Kaggle input mount")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_split(split_path: Path) -> dict[str, int]:
    rows = read_csv(split_path)
    counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "val" for row in rows),
        "test": sum(row["split"] == "test" for row in rows),
    }
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    counts.update(
        {
            "train_normal": sum(int(row["tumor"]) == 0 for row in train_rows),
            "train_tumor": sum(int(row["tumor"]) == 1 for row in train_rows),
            "validation_normal": sum(int(row["tumor"]) == 0 for row in val_rows),
            "validation_tumor": sum(int(row["tumor"]) == 1 for row in val_rows),
        }
    )
    expected = {
        "train": 2981,
        "validation": 371,
        "test": 373,
        "train_normal": 1493,
        "train_tumor": 1488,
        "validation_normal": 187,
        "validation_tumor": 184,
    }
    if counts != expected:
        raise RuntimeError(f"Frozen split counts differ: {counts}")
    return counts


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Paired geometry recovery requires exactly two GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"Paired geometry recovery requires T4 x2, got {names}")
    sums = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        layer = torch.nn.Conv2d(1, 1, 3, bias=False).to(device)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            value = layer(torch.ones((1, 1, 8, 8), device=device)).sum()
        sums.append(float(value.item()))
    if sums != [324.0, 324.0]:
        raise RuntimeError(f"Two-device convolution guard failed: {sums}")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": 2,
        "device_names": names,
        "real_convolution_sums": sums,
    }


def find_recovered_train_root() -> Path:
    manifest = find_unique_input(
        "candidate_diagnostics_manifest.csv",
        RECOVERED_TRAIN_HASHES["candidate"],
    )
    root = manifest.parent
    pseudo = root / "pseudo_mask_manifest.csv"
    if not pseudo.is_file() or sha256(pseudo) != RECOVERED_TRAIN_HASHES["pseudo"]:
        raise RuntimeError("Recovered version-3 train pseudo manifest is unavailable")
    return root


def audit_candidate_root(
    root: Path,
    *,
    split: str,
    expected_images: int,
    expected_candidate_hash: str | None,
    expected_pseudo_hash: str | None,
) -> dict[str, object]:
    import numpy as np

    candidate_manifest = root / "candidate_diagnostics_manifest.csv"
    candidate_summary_path = root / "candidate_diagnostics_summary.json"
    pseudo_manifest = root / "pseudo_mask_manifest.csv"
    pseudo_summary_path = root / "pseudo_mask_summary.json"
    run_metadata_path = root / "run_metadata.json"
    for path in (
        candidate_manifest,
        candidate_summary_path,
        pseudo_manifest,
        pseudo_summary_path,
        run_metadata_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"Candidate evidence is incomplete: {path}")
    candidate_hash = sha256(candidate_manifest)
    pseudo_hash = sha256(pseudo_manifest)
    if expected_candidate_hash and candidate_hash != expected_candidate_hash:
        raise RuntimeError(f"Candidate manifest hash mismatch for {split}")
    if expected_pseudo_hash and pseudo_hash != expected_pseudo_hash:
        raise RuntimeError(f"Pseudo manifest hash mismatch for {split}")

    candidate_summary = json.loads(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    pseudo_summary = json.loads(pseudo_summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if (
        candidate_summary.get("split") != split
        or candidate_summary.get("cohort") != "all"
        or candidate_summary.get("expected_images") != expected_images
        or candidate_summary.get("manifest_rows") != expected_images
        or candidate_summary.get("ground_truth_loaded_during_generation") is not False
        or candidate_summary.get("manifest_sha256") != candidate_hash
        or pseudo_summary.get("manifest_sha256") != pseudo_hash
        or metadata.get("candidate_diagnostics_cohort") != "all"
        or metadata.get("force_normal_candidate_gallery") is not True
    ):
        raise RuntimeError(f"Candidate summary/provenance mismatch for {split}")

    candidate_rows = read_csv(candidate_manifest)
    if (
        len(candidate_rows) != expected_images
        or len({row["image_name"] for row in candidate_rows}) != expected_images
    ):
        raise RuntimeError(f"Candidate cohort mismatch for {split}")
    counts: list[int] = []
    semantic_mask_hash = hashlib.sha256()
    for row in candidate_rows:
        payload_path = root / row["diagnostic_path"]
        if not payload_path.is_file() or sha256(payload_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"Candidate payload mismatch: {row['image_name']}")
        with np.load(payload_path, allow_pickle=False) as payload:
            masks = np.asarray(payload["sam_masks"], dtype=np.uint8)
            count = int(masks.shape[0])
            semantic_mask_hash.update(row["image_name"].encode("utf-8"))
            semantic_mask_hash.update(masks.shape.__repr__().encode("ascii"))
            semantic_mask_hash.update(masks.tobytes(order="C"))
        if count != int(row["candidate_count"]) or not 1 <= count <= 81:
            raise RuntimeError(f"Candidate count mismatch: {row['image_name']}")
        counts.append(count)

    pseudo_rows = read_csv(pseudo_manifest)
    if len(pseudo_rows) != expected_images:
        raise RuntimeError(f"Pseudo-mask cohort mismatch for {split}")
    normal_nonempty = [
        row["image_name"]
        for row in pseudo_rows
        if row["true_tumor"] == "0" and int(row["mask_foreground_pixels"]) != 0
    ]
    if normal_nonempty:
        raise RuntimeError(f"Normal pseudo masks are nonempty: {normal_nonempty[:5]}")
    return {
        "split": split,
        "images": expected_images,
        "candidate_manifest_sha256": candidate_hash,
        "pseudo_manifest_sha256": pseudo_hash,
        "run_metadata_sha256": sha256(run_metadata_path),
        "physical_payload_hashes_verified": expected_images,
        "candidate_mask_semantic_sha256": semantic_mask_hash.hexdigest(),
        "maximum_candidates": max(counts),
        "minimum_candidates": min(counts),
        "empty_candidate_bags": 0,
        "normal_pseudo_masks_nonempty": 0,
        "ground_truth_loaded_during_generation": False,
    }


def semantic_reference_audit(
    *,
    split: str,
    current_root: Path,
    reference_candidate: Path,
    reference_pseudo: Path,
) -> dict[str, object]:
    current_candidate_rows = read_csv(
        current_root / "candidate_diagnostics_manifest.csv"
    )
    reference_candidate_rows = read_csv(reference_candidate)
    current_pseudo_rows = read_csv(current_root / "pseudo_mask_manifest.csv")
    reference_pseudo_rows = read_csv(reference_pseudo)
    current_candidates = {row["image_name"]: row for row in current_candidate_rows}
    reference_candidates = {
        row["image_name"]: row for row in reference_candidate_rows
    }
    current_pseudo = {row["image_name"]: row for row in current_pseudo_rows}
    reference_pseudo_index = {row["image_name"]: row for row in reference_pseudo_rows}
    if (
        set(current_candidates) != set(reference_candidates)
        or set(current_pseudo) != set(reference_pseudo_index)
        or set(current_candidates) != set(current_pseudo)
    ):
        raise RuntimeError(f"Semantic reference image set differs for {split}")

    candidate_differences: dict[str, int] = {}
    for image_name in sorted(current_candidates):
        current = current_candidates[image_name]
        reference = reference_candidates[image_name]
        if set(current) != set(reference):
            raise RuntimeError("Candidate manifest schemas differ")
        for field in current:
            if current[field] != reference[field]:
                candidate_differences[field] = (
                    candidate_differences.get(field, 0) + 1
                )
    if set(candidate_differences) - CANDIDATE_BYTE_FIELDS:
        raise RuntimeError(
            f"Candidate semantic fields differ for {split}: "
            f"{candidate_differences}"
        )

    numeric_max = {field: 0.0 for field in NUMERIC_TOLERANCE}
    changed_rows = {field: 0 for field in FRACTIONAL_CHANGE_LIMIT}
    exact_mismatches: dict[str, int] = {}
    for image_name in sorted(current_pseudo):
        current = current_pseudo[image_name]
        reference = reference_pseudo_index[image_name]
        if set(current) != set(reference):
            raise RuntimeError("Pseudo manifest schemas differ")
        for field in current:
            if field in NUMERIC_TOLERANCE:
                if current[field] == reference[field]:
                    continue
                if current[field] == "" or reference[field] == "":
                    raise RuntimeError(
                        f"Numeric blank-state changed: {image_name} {field}"
                    )
                delta = abs(float(current[field]) - float(reference[field]))
                numeric_max[field] = max(numeric_max[field], delta)
                if field in changed_rows:
                    changed_rows[field] += 1
                continue
            if field in FRACTIONAL_CHANGE_LIMIT:
                if current[field] != reference[field]:
                    changed_rows[field] += 1
                continue
            if current[field] != reference[field]:
                exact_mismatches[field] = exact_mismatches.get(field, 0) + 1
    if exact_mismatches:
        raise RuntimeError(
            f"Unlisted pseudo fields differ for {split}: {exact_mismatches}"
        )
    for field, maximum in NUMERIC_TOLERANCE.items():
        if numeric_max[field] > maximum:
            raise RuntimeError(
                f"Pseudo numeric tolerance exceeded for {split}/{field}: "
                f"{numeric_max[field]} > {maximum}"
            )
    images = len(current_pseudo)
    for field, fraction in FRACTIONAL_CHANGE_LIMIT.items():
        allowed = math.floor(images * fraction)
        if changed_rows[field] > allowed:
            raise RuntimeError(
                f"Pseudo changed-row limit exceeded for {split}/{field}: "
                f"{changed_rows[field]} > {allowed}"
            )
    return {
        "split": split,
        "images": images,
        "candidate_differing_columns": candidate_differences,
        "candidate_semantic_field_mismatches": 0,
        "pseudo_unlisted_field_mismatches": 0,
        "maximum_absolute_numeric_delta": numeric_max,
        "changed_rows": changed_rows,
        "final_pseudo_mask_sha256_mismatches": 0,
        "status_mismatches": 0,
        "sam_candidate_count_mismatches": 0,
        "purpose": "comparability only; paired same-gallery arms supply causality",
    }


def generation_command(
    *,
    project: Path,
    data: Path,
    split_manifest: Path,
    classifier: Path,
    sam: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(project / "generate_pseudo_masks.py"),
        "--pipeline-profile",
        "default",
        "--data-root",
        str(data),
        "--split",
        "val",
        "--split-manifest",
        str(split_manifest),
        "--classifier-checkpoint",
        str(classifier),
        "--sam-checkpoint",
        str(sam),
        "--classifier-device",
        "cuda:0",
        "--sam-device",
        "cuda:1",
        "--target-columns",
        "tumor",
        "--image-size",
        "320",
        "--sam-image-size",
        "512",
        "--batch-size",
        "1",
        "--num-workers",
        "2",
        "--output-dir",
        str(output_dir),
        "--process-all",
        "--save-visuals-limit",
        "0",
        "--confidence-threshold",
        "0.5",
        "--cam-percentile",
        "90",
        "--cam-percentile-ensemble",
        "--cam-percentile-values",
        "85,90,95",
        "--cam-tta-flip",
        "--max-points",
        "5",
        "--min-component-area",
        "100",
        "--mask-score-threshold",
        "0.4",
        "--seed-percentile",
        "82",
        "--support-percentile",
        "55",
        "--morphology-fusion-mode",
        "components",
        "--sam-prompt-mode",
        "box_point",
        "--sam-prompt-ensemble",
        "--max-components",
        "3",
        "--all-cam-components",
        "--points-per-component",
        "5",
        "--bbox-padding-ratio",
        "0.02",
        "--negative-points-per-component",
        "4",
        "--prompt-border-margin",
        "2",
        "--max-box-area-ratio",
        "0.35",
        "--layercam-weights",
        "0.20,0.30,0.50",
        "--layercam-gradient-mode",
        "positive",
        "--selection-method",
        "coverage_mass_sam",
        "--fusion-topk",
        "1",
        "--component-topk",
        "3",
        "--support-clip-kernel",
        "5",
        "--closing-kernel",
        "0",
        "--opening-kernel",
        "0",
        "--min-size",
        "40",
        "--max-hole-area",
        "0",
        "--guidance-threshold",
        "0.4",
        "--preprocessing-mode",
        "none",
        "--cam-aggregation",
        "class",
        "--low-score-policy",
        "empty",
        "--cam-target-class",
        "ground_truth",
        "--save-candidate-diagnostics",
        "--candidate-diagnostics-cohort",
        "all",
        "--force-normal-candidate-gallery",
    ]


def download_rad_dino() -> Path:
    from huggingface_hub import snapshot_download

    destination = TEMP / "rad-dino"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=list(MODEL_HASHES),
    )
    for name, expected in MODEL_HASHES.items():
        path = destination / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"RAD-DINO snapshot hash mismatch: {name}")
    return destination


def copy_candidate_evidence(
    *,
    root: Path,
    destination: Path,
    prefix: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_metadata.json",
        "pseudo_mask_manifest.csv",
        "pseudo_mask_summary.json",
        "candidate_diagnostics_manifest.csv",
        "candidate_diagnostics_summary.json",
    ):
        source = root / name
        if not source.is_file():
            raise RuntimeError(f"Candidate evidence file is missing: {source}")
        shutil.copy2(source, destination / f"{prefix}_{name}")


def run_fractional_audit(
    *,
    project: Path,
    data: Path,
    split_manifest: Path,
    split: str,
    candidate_root: Path,
    candidate_audit: dict[str, object],
    output_dir: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, object]:
    run(
        [
            sys.executable,
            str(project / "audit_mask_bag_fractional_grid_mass.py"),
            "--dataset-root",
            str(data),
            "--split-manifest",
            str(split_manifest),
            "--expected-split-sha256",
            SPLIT_SHA256,
            "--split",
            split,
            "--candidate-root",
            str(candidate_root),
            "--expected-candidate-manifest-sha256",
            str(candidate_audit["candidate_manifest_sha256"]),
            "--expected-pseudo-manifest-sha256",
            str(candidate_audit["pseudo_manifest_sha256"]),
            "--output-dir",
            str(output_dir),
            "--maximum-candidates",
            "81",
        ],
        cwd=project,
        env=env,
        log_path=log_path,
    )
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("images") != candidate_audit["images"]
        or summary.get("candidate_manifest_sha256")
        != candidate_audit["candidate_manifest_sha256"]
        or summary.get("consumer_trained") is not False
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError(f"Fractional grid-mass audit mismatch for {split}")
    return summary


def runner_command(
    *,
    project: Path,
    data: Path,
    split_manifest: Path,
    model_dir: Path,
    train_root: Path,
    train_audit: dict[str, object],
    val_root: Path,
    val_audit: dict[str, object],
    geometry: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(project / "run_rad_dino_mask_bag_mil_probe.py"),
        "--dataset-root",
        str(data),
        "--split-manifest",
        str(split_manifest),
        "--expected-split-sha256",
        SPLIT_SHA256,
        "--model-dir",
        str(model_dir),
        "--expected-config-sha256",
        MODEL_HASHES["config.json"],
        "--expected-preprocessor-sha256",
        MODEL_HASHES["preprocessor_config.json"],
        "--expected-weight-sha256",
        MODEL_HASHES["model.safetensors"],
        "--train-candidate-root",
        str(train_root),
        "--train-candidate-manifest-sha256",
        str(train_audit["candidate_manifest_sha256"]),
        "--train-pseudo-manifest-sha256",
        str(train_audit["pseudo_manifest_sha256"]),
        "--val-candidate-root",
        str(val_root),
        "--val-candidate-manifest-sha256",
        str(val_audit["candidate_manifest_sha256"]),
        "--val-pseudo-manifest-sha256",
        str(val_audit["pseudo_manifest_sha256"]),
        "--source-commit",
        SOURCE_COMMIT,
        "--protocol-sha256",
        RECOVERY_PROTOCOL_SHA256,
        "--output-dir",
        str(output_dir),
        "--input-size",
        "448",
        "--projection-dim",
        "128",
        "--projection-seed",
        "42",
        "--encoder-batch-size",
        "4",
        "--train-batch-size",
        "16",
        "--epochs",
        "16",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--instance-loss-weight",
        "0.25",
        "--consistency-loss-weight",
        "0.10",
        "--instance-warmup-epochs",
        "2",
        "--maximum-candidates",
        "81",
        "--seed",
        "42",
        "--descriptor-geometry",
        geometry,
    ]


def verify_prediction_arm(root: Path, expected_mode: str) -> dict[str, object]:
    freeze_path = root / "prediction_freeze.json"
    manifest_path = root / "predictions" / "prediction_manifest.csv"
    run_manifest_path = root / "run_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(manifest_path)
    if (
        freeze.get("source_commit") != SOURCE_COMMIT
        or freeze.get("protocol_sha256") != RECOVERY_PROTOCOL_SHA256
        or freeze.get("candidate_descriptor_geometry", {}).get("mode")
        != expected_mode
        or freeze.get("validation_predictions") != 371
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
        or run_manifest.get("runtime", {}).get("encoder_data_parallel") is not True
        or len(rows) != 371
        or len({row["image_id"] for row in rows}) != 371
        or sha256(manifest_path) != freeze.get("prediction_manifest_sha256")
    ):
        raise RuntimeError(f"Prediction freeze mismatch for {expected_mode}")
    for row in rows:
        map_path = root / "predictions" / row["map_path"]
        if not map_path.is_file() or sha256(map_path) != row["map_sha256"]:
            raise RuntimeError(
                f"Prediction map mismatch for {expected_mode}: {row['image_id']}"
            )
    return {
        "mode": expected_mode,
        "prediction_freeze_sha256": sha256(freeze_path),
        "prediction_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "training_history_sha256": freeze["training_history_sha256"],
        "physical_prediction_maps_verified": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def evaluate_arm(
    *,
    project: Path,
    data: Path,
    split_manifest: Path,
    val_root: Path,
    val_audit: dict[str, object],
    baseline: Path,
    prediction_root: Path,
    output_dir: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, object]:
    freeze_hash = sha256(prediction_root / "prediction_freeze.json")
    run(
        [
            sys.executable,
            str(project / "evaluate_rad_dino_mask_bag_mil_probe.py"),
            "--dataset-root",
            str(data),
            "--split-manifest",
            str(split_manifest),
            "--expected-split-sha256",
            SPLIT_SHA256,
            "--prediction-root",
            str(prediction_root),
            "--expected-prediction-freeze-sha256",
            freeze_hash,
            "--expected-source-commit",
            SOURCE_COMMIT,
            "--expected-protocol-sha256",
            RECOVERY_PROTOCOL_SHA256,
            "--val-candidate-root",
            str(val_root),
            "--expected-val-candidate-manifest-sha256",
            str(val_audit["candidate_manifest_sha256"]),
            "--expected-val-pseudo-manifest-sha256",
            str(val_audit["pseudo_manifest_sha256"]),
            "--baseline-per-image",
            str(baseline),
            "--expected-baseline-per-image-sha256",
            BASELINE_SHA256,
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "10000",
            "--bootstrap-seed",
            "20261001",
        ],
        cwd=project,
        env=env,
        log_path=log_path,
    )
    audit = json.loads(
        (output_dir / "evaluation_audit.json").read_text(encoding="utf-8")
    )
    if (
        audit.get("cohort") != EXPECTED_COHORT
        or audit.get("bootstrap_replicates") != 10000
        or audit.get("validation_gt_read_only_after_prediction_freeze") is not True
        or audit.get("consumer_trained") is not False
        or audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("Evaluation audit contract mismatch")
    return audit


def compare_arms(
    *,
    project: Path,
    candidate: Path,
    candidate_name: str,
    reference: Path,
    reference_name: str,
    output_dir: Path,
    seed: int,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, object]:
    run(
        [
            sys.executable,
            str(project / "compare_mask_bag_evaluated_arms.py"),
            "--candidate-per-image",
            str(candidate),
            "--expected-candidate-sha256",
            sha256(candidate),
            "--candidate-name",
            candidate_name,
            "--reference-per-image",
            str(reference),
            "--expected-reference-sha256",
            sha256(reference),
            "--reference-name",
            reference_name,
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "10000",
            "--bootstrap-seed",
            str(seed),
        ],
        cwd=project,
        env=env,
        log_path=log_path,
    )
    payload = json.loads(
        (output_dir / "paired_comparison.json").read_text(encoding="utf-8")
    )
    expected = {"tumor": 184, "small": 94, "medium": 72, "large": 18}
    if (
        payload.get("cohort") != expected
        or payload.get("replicates") != 10000
        or payload.get("seed_family") != seed
        or payload.get("complete_misses_included") is not True
        or payload.get("ground_truth_reopened") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise RuntimeError("Paired comparison contract mismatch")
    return payload


def selected_dice(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        subgroup: float(summary["tumor_localization"][subgroup]["dice"])
        for subgroup in ("overall", "small", "medium", "large")
    }


def main() -> None:
    clock = time.monotonic()
    started = datetime.now(timezone.utc)
    repository = Path(__file__).resolve().parents[1]
    project = repository / "project"
    protocol = repository / RECOVERY_PROTOCOL_RELATIVE
    if sha256(protocol) != RECOVERY_PROTOCOL_SHA256:
        raise RuntimeError("Paired recovery protocol hash mismatch")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    TEMP.mkdir(parents=True, exist_ok=False)
    log_path = OUTPUT / "execution.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "42",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )

    data = locate_dataset_root()
    split_input = find_unique_input("frozen_split_manifest.csv", SPLIT_SHA256)
    split_manifest = TEMP / "frozen_split_manifest.csv"
    shutil.copy2(split_input, split_manifest)
    split_counts = verify_split(split_manifest)
    classifier = find_unique_input("best_classifier.pt", CLASSIFIER_SHA256)
    sam = find_unique_input("sam_vit_b_01ec64.pth", SAM_SHA256)
    baseline = find_unique_input("per_image.csv", BASELINE_SHA256)
    v6_per_image = find_unique_input(
        "per_image.csv", V6_EVALUATION_PER_IMAGE_SHA256
    )
    v6_paired = find_unique_input(
        "paired_comparison.json", V6_PAIRED_COMPARISON_SHA256
    )
    v6_references = {
        key: find_unique_input(
            (
                "train_candidate_diagnostics_manifest.csv"
                if key == "train_candidate"
                else "train_pseudo_mask_manifest.csv"
                if key == "train_pseudo"
                else "val_candidate_diagnostics_manifest.csv"
                if key == "val_candidate"
                else "val_pseudo_mask_manifest.csv"
            ),
            value,
        )
        for key, value in V6_MANIFEST_HASHES.items()
    }
    gpu = verify_t4x2()

    focused = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_rad_dino_mask_bag_mil.py",
            "tests/test_rad_dino_mask_bag_mil_probe.py",
            "tests/test_candidate_diagnostics.py",
            "tests/test_generate_pseudo_masks_devices.py",
            "tests/test_mask_bag_fractional_grid_mass_audit.py",
            "tests/test_compare_mask_bag_evaluated_arms.py",
        ],
        cwd=repository,
        env=env,
        log_path=log_path,
    )
    whole = run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repository,
        env=env,
        log_path=log_path,
    )
    if "34 passed" not in focused or "223 passed, 1 skipped" not in whole:
        raise RuntimeError("Paired recovery preflight test count differs")

    train_root = find_recovered_train_root()
    train_audit = audit_candidate_root(
        train_root,
        split="train",
        expected_images=2981,
        expected_candidate_hash=RECOVERED_TRAIN_HASHES["candidate"],
        expected_pseudo_hash=RECOVERED_TRAIN_HASHES["pseudo"],
    )
    train_semantic = semantic_reference_audit(
        split="train",
        current_root=train_root,
        reference_candidate=v6_references["train_candidate"],
        reference_pseudo=v6_references["train_pseudo"],
    )

    val_root = TEMP / "val_candidates"
    run(
        generation_command(
            project=project,
            data=data,
            split_manifest=split_manifest,
            classifier=classifier,
            sam=sam,
            output_dir=val_root,
        ),
        cwd=project,
        env=env,
        log_path=log_path,
    )
    val_audit = audit_candidate_root(
        val_root,
        split="val",
        expected_images=371,
        expected_candidate_hash=None,
        expected_pseudo_hash=None,
    )
    val_semantic = semantic_reference_audit(
        split="val",
        current_root=val_root,
        reference_candidate=v6_references["val_candidate"],
        reference_pseudo=v6_references["val_pseudo"],
    )

    model_dir = download_rad_dino()
    fractional_root = OUTPUT / "fractional_grid_mass"
    fractional = {
        "train": run_fractional_audit(
            project=project,
            data=data,
            split_manifest=split_manifest,
            split="train",
            candidate_root=train_root,
            candidate_audit=train_audit,
            output_dir=fractional_root / "train",
            env=env,
            log_path=log_path,
        ),
        "validation": run_fractional_audit(
            project=project,
            data=data,
            split_manifest=split_manifest,
            split="val",
            candidate_root=val_root,
            candidate_audit=val_audit,
            output_dir=fractional_root / "validation",
            env=env,
            log_path=log_path,
        ),
    }
    candidate_audit_payload = {
        "train": train_audit,
        "validation": val_audit,
        "terminal_v6_semantic_reference": {
            "train": train_semantic,
            "validation": val_semantic,
        },
        "same_physical_payloads_for_both_arms": True,
        "physical_payloads_verified_before_optimizer": 3352,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    write_json(OUTPUT / "candidate_input_audit.json", candidate_audit_payload)
    candidate_evidence = OUTPUT / "candidate_evidence"
    copy_candidate_evidence(
        root=train_root,
        destination=candidate_evidence,
        prefix="train",
    )
    copy_candidate_evidence(
        root=val_root,
        destination=candidate_evidence,
        prefix="val",
    )
    shutil.copy2(
        v6_references["train_candidate"],
        candidate_evidence / "terminal_v6_train_candidate_manifest.csv",
    )
    shutil.copy2(
        v6_references["train_pseudo"],
        candidate_evidence / "terminal_v6_train_pseudo_manifest.csv",
    )
    shutil.copy2(
        v6_references["val_candidate"],
        candidate_evidence / "terminal_v6_val_candidate_manifest.csv",
    )
    shutil.copy2(
        v6_references["val_pseudo"],
        candidate_evidence / "terminal_v6_val_pseudo_manifest.csv",
    )
    provenance = OUTPUT / "provenance"
    provenance.mkdir()
    shutil.copy2(protocol, provenance / protocol.name)
    shutil.copy2(
        Path(__file__),
        provenance / "run_mask_bag_geometry_paired_recovery.py",
    )
    for relative in (
        Path("project/run_rad_dino_mask_bag_mil_probe.py"),
        Path("project/models/rad_dino_mask_bag_mil.py"),
        Path("tests/test_rad_dino_mask_bag_mil.py"),
        Path("tests/test_rad_dino_mask_bag_mil_probe.py"),
        Path("tests/test_run_mask_bag_geometry_paired_recovery.py"),
    ):
        source = repository / relative
        destination = provenance / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    legacy_root = OUTPUT / "legacy_control"
    corrected_root = OUTPUT / "square_corrected"
    run(
        runner_command(
            project=project,
            data=data,
            split_manifest=split_manifest,
            model_dir=model_dir,
            train_root=train_root,
            train_audit=train_audit,
            val_root=val_root,
            val_audit=val_audit,
            geometry="legacy_direct_resize",
            output_dir=legacy_root,
        ),
        cwd=project,
        env=env,
        log_path=log_path,
    )
    legacy_freeze = verify_prediction_arm(legacy_root, "legacy_direct_resize")
    run(
        runner_command(
            project=project,
            data=data,
            split_manifest=split_manifest,
            model_dir=model_dir,
            train_root=train_root,
            train_audit=train_audit,
            val_root=val_root,
            val_audit=val_audit,
            geometry="square_corrected_v3",
            output_dir=corrected_root,
        ),
        cwd=project,
        env=env,
        log_path=log_path,
    )
    corrected_freeze = verify_prediction_arm(
        corrected_root, "square_corrected_v3"
    )
    pair_freeze = {
        "protocol_sha256": RECOVERY_PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "train_candidate_manifest_sha256": train_audit[
            "candidate_manifest_sha256"
        ],
        "train_candidate_mask_semantic_sha256": train_audit[
            "candidate_mask_semantic_sha256"
        ],
        "validation_candidate_manifest_sha256": val_audit[
            "candidate_manifest_sha256"
        ],
        "validation_candidate_mask_semantic_sha256": val_audit[
            "candidate_mask_semantic_sha256"
        ],
        "legacy_control": legacy_freeze,
        "square_corrected": corrected_freeze,
        "sole_changed_scientific_variable": (
            "legacy_direct_resize versus square_corrected_v3 descriptor geometry"
        ),
        "both_prediction_cohorts_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_freeze_path = OUTPUT / "paired_prediction_freeze.json"
    write_json(pair_freeze_path, pair_freeze)

    legacy_evaluation = OUTPUT / "legacy_evaluation"
    corrected_evaluation = OUTPUT / "corrected_evaluation"
    legacy_evaluation_audit = evaluate_arm(
        project=project,
        data=data,
        split_manifest=split_manifest,
        val_root=val_root,
        val_audit=val_audit,
        baseline=baseline,
        prediction_root=legacy_root,
        output_dir=legacy_evaluation,
        env=env,
        log_path=log_path,
    )
    corrected_evaluation_audit = evaluate_arm(
        project=project,
        data=data,
        split_manifest=split_manifest,
        val_root=val_root,
        val_audit=val_audit,
        baseline=baseline,
        prediction_root=corrected_root,
        output_dir=corrected_evaluation,
        env=env,
        log_path=log_path,
    )

    comparisons = {
        "corrected_minus_same_gallery_legacy": compare_arms(
            project=project,
            candidate=corrected_evaluation / "per_image.csv",
            candidate_name="square_corrected_v3",
            reference=legacy_evaluation / "per_image.csv",
            reference_name="same_gallery_legacy_direct_resize",
            output_dir=OUTPUT / "comparison_corrected_minus_legacy",
            seed=20261201,
            env=env,
            log_path=log_path,
        ),
        "legacy_minus_terminal_v6_descriptive": compare_arms(
            project=project,
            candidate=legacy_evaluation / "per_image.csv",
            candidate_name="same_gallery_legacy_direct_resize",
            reference=v6_per_image,
            reference_name="terminal_v6_descriptive",
            output_dir=OUTPUT / "comparison_legacy_minus_terminal_v6",
            seed=20261211,
            env=env,
            log_path=log_path,
        ),
        "corrected_minus_terminal_v6_descriptive": compare_arms(
            project=project,
            candidate=corrected_evaluation / "per_image.csv",
            candidate_name="square_corrected_v3",
            reference=v6_per_image,
            reference_name="terminal_v6_descriptive",
            output_dir=OUTPUT / "comparison_corrected_minus_terminal_v6",
            seed=20261221,
            env=env,
            log_path=log_path,
        ),
    }
    legacy_dice = selected_dice(legacy_evaluation / "summary.json")
    corrected_dice = selected_dice(corrected_evaluation / "summary.json")
    goal_checks = {
        subgroup: {
            "observed": corrected_dice[subgroup],
            "minimum": OPERATIONAL_GOALS[subgroup],
            "pass": corrected_dice[subgroup] >= OPERATIONAL_GOALS[subgroup],
        }
        for subgroup in OPERATIONAL_GOALS
    }
    decision = {
        "geometry_coordinate_contract": "SQUARE_CORRECTED_V3_REMAINS_CANONICAL",
        "selected_by_validation_gt": False,
        "legacy_selected_dice": legacy_dice,
        "square_corrected_selected_dice": corrected_dice,
        "square_corrected_operational_goal_checks": goal_checks,
        "all_operational_goals_pass": all(
            item["pass"] for item in goal_checks.values()
        ),
        "terminal_v6_comparisons_are_descriptive_only": True,
        "next_bottleneck_if_goals_fail": "descriptor_selector_campaign",
        "consumer_authorized": False,
        "test_remains_locked": True,
    }
    write_json(OUTPUT / "decision.json", decision)

    independent = {
        "status": "PASS",
        "protocol_sha256": RECOVERY_PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "split_counts": split_counts,
        "classifier_sha256": sha256(classifier),
        "sam_sha256": sha256(sam),
        "baseline_per_image_sha256": sha256(baseline),
        "terminal_v6_per_image_sha256": sha256(v6_per_image),
        "terminal_v6_paired_comparison_sha256": sha256(v6_paired),
        "gpu": gpu,
        "rad_dino_hashes": {
            name: sha256(model_dir / name) for name in MODEL_HASHES
        },
        "candidate_input_audit_sha256": sha256(
            OUTPUT / "candidate_input_audit.json"
        ),
        "fractional_grid_mass": {
            split: {
                "summary_sha256": sha256(
                    fractional_root
                    / ("validation" if split == "validation" else "train")
                    / "summary.json"
                ),
                "images": payload["images"],
            }
            for split, payload in fractional.items()
        },
        "paired_prediction_freeze_sha256": sha256(pair_freeze_path),
        "prediction_maps_verified": {
            "legacy": 371,
            "square_corrected": 371,
        },
        "legacy_evaluation_audit_sha256": sha256(
            legacy_evaluation / "evaluation_audit.json"
        ),
        "corrected_evaluation_audit_sha256": sha256(
            corrected_evaluation / "evaluation_audit.json"
        ),
        "comparison_hashes": {
            "corrected_minus_legacy": sha256(
                OUTPUT
                / "comparison_corrected_minus_legacy"
                / "paired_comparison.json"
            ),
            "legacy_minus_terminal_v6": sha256(
                OUTPUT
                / "comparison_legacy_minus_terminal_v6"
                / "paired_comparison.json"
            ),
            "corrected_minus_terminal_v6": sha256(
                OUTPUT
                / "comparison_corrected_minus_terminal_v6"
                / "paired_comparison.json"
            ),
        },
        "same_gallery_causal_comparison": True,
        "validation_gt_read_only_after_pair_prediction_freeze": True,
        "complete_misses_included": True,
        "bootstrap_replicates": 10000,
        "consumer_trained": False,
        "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - clock,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    independent_path = OUTPUT / "independent_audit.json"
    write_json(independent_path, independent)
    write_json(
        OUTPUT / "completed.json",
        {
            "status": "COMPLETE",
            "all_operational_goals_pass": decision["all_operational_goals_pass"],
            "independent_audit_sha256": sha256(independent_path),
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )

    shutil.rmtree(val_root)
    shutil.rmtree(model_dir)
    print(json.dumps({"decision": decision, "audit": independent}, indent=2))


if __name__ == "__main__":
    main()
