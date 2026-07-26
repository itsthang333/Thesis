from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "btxrd_mae_normality_reconstruction_probe_val_v1"
SOURCE_REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "7292c5d2f7722d273c27eb147b19cbe7b25c9709"
PROTOCOL_SHA256 = "06055700e48980ddab0ab87c9e36dace7c7aab103d5b6fc004fa5dd742f8da06"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
FROZEN_SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
MODEL_ID = "facebook/vit-mae-base"
MODEL_REVISION = "25b184bea5538bf5c4c852c79d221195fdd2778d"
BASE_HASHES = {
    "config.json": "1f7a507e1f7f63a262c5500b86470c3b22ff1a9e42de81bbba27105396572dae",
    "preprocessor_config.json": "a250969c94afba52d785a0e08dd36e13aeda97c4dd2b7fd0d24b457288536cea",
    "model.safetensors": "479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8",
}
SOURCE_HASHES = {
    "train_mae_reconstruction.py": "a4b1b6db0b3d5db290a6a5099512e94bd8954e58f11dfb60f166440af51f8f05",
    "generate_mae_reconstruction_saliency.py": "3efbd24827905db6a566a0540ed64447ce2afe7c6910448a3df56df30647e61e",
    "evaluate_mae_reconstruction_saliency.py": "2830593328fb9282fb6f35728b1ccf4cb5034eaf83fd51a7b314520e16401824",
    "compare_mae_reconstruction_arms.py": "4fe7e818e53376c2b7b9b1a4408cc815fc4de2df31b27dc3225fae546bb0a25c",
    "mae_reconstruction_io.py": "1b14cae95354db53fc2117254701920870510d4b8bb1005f1400340ad4c8859c",
    "models/mae_reconstruction.py": "7baa1a664d0523d454ab19bc1420959ef5115e31da2a242345c29f1dab7417f9",
    "../tests/test_mae_reconstruction.py": "add56c2ac0de5e4ffb4f72e856f274772fd8c286f938b0990677ffe70e7a5693",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command, cwd=cwd, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT
        )


def checkout_source() -> tuple[Path, Path, Path]:
    repository = WORK / "thesis_source"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPOSITORY, str(repository)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "checkout", "--detach", SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")
    project = repository / "project"
    for relative, expected in SOURCE_HASHES.items():
        if sha256(project / relative) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    protocol = repository / "artifacts/research_protocols/mae_normality_reconstruction_probe_val_v1.json"
    if sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("Protocol hash mismatch")
    parsed = json.loads(protocol.read_text(encoding="utf-8"))
    if parsed["status"] != "predeclared_before_execution" or parsed["data_policy"]["test_evaluated"]:
        raise RuntimeError("Protocol contract mismatch")
    git_split = repository / "artifacts/data_audit/split_manifest.csv"
    if sha256(git_split) != GIT_SPLIT_SHA256:
        raise RuntimeError("Git split hash mismatch")
    split = WORK / "frozen_split_manifest.csv"
    split.write_bytes(git_split.read_bytes().replace(b"\n", b"\r\n"))
    if sha256(split) != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen CRLF split hash mismatch")
    return project.resolve(), split.resolve(), protocol.resolve()


def find_btxrd_root() -> Path:
    for candidate in [INPUT / "btxrd-raw/BTXRD", *sorted(INPUT.glob("**/BTXRD"))]:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("BTXRD root not found")


def audit_split(split: Path) -> None:
    with split.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["eligible"] == "1"]
    counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "train_normal": sum(row["split"] == "train" and row["tumor"] == "0" for row in rows),
        "val": sum(row["split"] == "val" for row in rows),
        "val_tumor": sum(row["split"] == "val" and row["tumor"] == "1" for row in rows),
    }
    if counts != {"train": 2981, "train_normal": 1493, "val": 371, "val_tumor": 184}:
        raise RuntimeError(f"Split cohort mismatch: {counts}")


def download_base_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    model_dir = WORK / "vit-mae-base"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=list(BASE_HASHES),
    )
    for name, expected in BASE_HASHES.items():
        if sha256(model_dir / name) != expected:
            raise RuntimeError(f"Base model hash mismatch: {name}")
    return model_dir


def manifest_hash(path: Path, expected_rows: int = 371) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Prediction manifest cohort mismatch: {len(rows)}")
    return sha256(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    log = OUTPUT / "execution.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["TOKENIZERS_PARALLELISM"] = "false"
    project, split, protocol = checkout_source()
    data = find_btxrd_root()
    audit_split(split)
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "transformers==4.50.2"],
        cwd=WORK, env=env, log=log,
    )
    run(
        [sys.executable, "-m", "pytest", "tests/test_mae_reconstruction.py", "-q"],
        cwd=project.parent, env=env, log=log,
    )
    base_model = download_base_snapshot()
    adapted = OUTPUT / "adapted_training"
    run(
        [
            sys.executable, "train_mae_reconstruction.py",
            "--dataset-root", str(data),
            "--split-manifest", str(split),
            "--expected-split-sha256", FROZEN_SPLIT_SHA256,
            "--model-dir", str(base_model),
            "--expected-config-sha256", BASE_HASHES["config.json"],
            "--expected-preprocessor-sha256", BASE_HASHES["preprocessor_config.json"],
            "--expected-weight-sha256", BASE_HASHES["model.safetensors"],
            "--output-dir", str(adapted),
            "--source-commit", SOURCE_COMMIT,
        ],
        cwd=project, env=env, log=log,
    )
    adapted_meta = json.loads((adapted / "run_metadata.json").read_text(encoding="utf-8"))
    adapted_hashes = {
        "config.json": adapted_meta["final_checkpoint"]["config_sha256"],
        "preprocessor_config.json": adapted_meta["final_checkpoint"]["preprocessor_sha256"],
        "model.safetensors": adapted_meta["final_checkpoint"]["weight_sha256"],
    }

    arms = {
        "base": (base_model, BASE_HASHES),
        "normal_adapted": (adapted / "model", adapted_hashes),
    }
    freeze: dict[str, object] = {
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": sha256(protocol),
        "split_sha256": sha256(split),
        "arms": {},
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    for role, (model_dir, hashes) in arms.items():
        prediction = OUTPUT / f"{role}_prediction"
        run(
            [
                sys.executable, "generate_mae_reconstruction_saliency.py",
                "--dataset-root", str(data),
                "--split-manifest", str(split),
                "--expected-split-sha256", FROZEN_SPLIT_SHA256,
                "--model-dir", str(model_dir),
                "--expected-config-sha256", hashes["config.json"],
                "--expected-preprocessor-sha256", hashes["preprocessor_config.json"],
                "--expected-weight-sha256", hashes["model.safetensors"],
                "--model-role", role,
                "--source-commit", SOURCE_COMMIT,
                "--output-dir", str(prediction),
            ],
            cwd=project, env=env, log=log,
        )
        manifest = prediction / "prediction_manifest.csv"
        freeze["arms"][role] = {
            "model_hashes": hashes,
            "prediction_manifest_sha256": manifest_hash(manifest),
            "generation_metadata_sha256": sha256(prediction / "generation_metadata.json"),
        }
    freeze_path = OUTPUT / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    freeze_sha = sha256(freeze_path)

    for role in arms:
        prediction = OUTPUT / f"{role}_prediction"
        evaluation = OUTPUT / f"{role}_evaluation"
        run(
            [
                sys.executable, "evaluate_mae_reconstruction_saliency.py",
                "--dataset-root", str(data),
                "--split-manifest", str(split),
                "--prediction-dir", str(prediction),
                "--expected-manifest-sha256", freeze["arms"][role]["prediction_manifest_sha256"],
                "--output-dir", str(evaluation),
            ],
            cwd=project, env=env, log=log,
        )
    comparison = OUTPUT / "paired_comparison.json"
    run(
        [
            sys.executable, "compare_mae_reconstruction_arms.py",
            "--base-per-image", str(OUTPUT / "base_evaluation/per_image.csv"),
            "--adapted-per-image", str(OUTPUT / "normal_adapted_evaluation/per_image.csv"),
            "--output", str(comparison),
        ],
        cwd=project, env=env, log=log,
    )
    run_manifest = {
        "run_id": "btxrd_mae_normality_reconstruction_probe_val_v1",
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": sha256(protocol),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "split_sha256": sha256(split),
        "prediction_freeze_sha256": freeze_sha,
        "adapted_checkpoint_sha256": adapted_hashes["model.safetensors"],
        "paired_comparison_sha256": sha256(comparison),
        "cohort": {"train": 2981, "train_normal": 1493, "val": 371, "val_tumor": 184},
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the adapted checkpoint in Kaggle output for reproducibility/future fusion.
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
