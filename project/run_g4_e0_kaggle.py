from __future__ import annotations

"""Run G4 E0 fully-mask freeze and common-grid validation on Kaggle."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CANDIDATE_MANIFEST_SHA = "30e734e223839ee2ed7b445c71b3731a39f0549ee88e2ba77f7bf722d8d9943d"
WSSS_FREEZE_SHA = "a75c0388346b6a1a3ab94f3ddd700a2e495c36be257b63712e38fc451784a620"
FULLY_CHECKPOINT_SHA = "becd752d0df2b0adfe0ea0c099117435f5b82da8fe2726eebecfb7af6322f33f"


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
        path.parent for path in root.rglob("run_g4_e0_kaggle.py")
        if path.parent.name == "project"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one E0 project, found {len(matches)}")
    return matches[0]


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    input_root = Path("/kaggle/input")
    output_root = Path("/kaggle/working/g4_e0")
    output_root.mkdir(parents=True, exist_ok=False)
    project = unique_project(input_root)
    split = unique_file(input_root, "canonical_split_manifest_85511.csv", SPLIT_SHA)
    checkpoint = unique_file(input_root, "best_unet.pt", FULLY_CHECKPOINT_SHA)
    candidate_manifest = unique_file(
        input_root, "candidate_diagnostics_manifest.csv", CANDIDATE_MANIFEST_SHA
    )
    wsss_freeze = unique_file(input_root, "prediction_freeze.json", WSSS_FREEZE_SHA)
    btxrd = [
        path for path in input_root.rglob("BTXRD")
        if path.is_dir() and (path / "images").is_dir() and (path / "Annotations").is_dir()
    ]
    if len(btxrd) != 1:
        raise RuntimeError(f"expected one BTXRD root, found {len(btxrd)}")

    fully_masks = output_root / "fully_masks"
    run([
        sys.executable,
        str(project / "freeze_fully_supervised_masks.py"),
        "--dataset-root", str(btxrd[0]),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", FULLY_CHECKPOINT_SHA,
        "--threshold", "0.20",
        "--image-size", "448",
        "--batch-size", "8",
        "--num-workers", "2",
        "--output-dir", str(fully_masks),
    ])
    fully_freeze = fully_masks / "mask_freeze.json"
    evaluation = output_root / "evaluation"
    run([
        sys.executable,
        str(project / "evaluate_g4_e0_common_grids.py"),
        "--dataset-root", str(btxrd[0]),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--candidate-root", str(candidate_manifest.parent),
        "--expected-candidate-manifest-sha256", CANDIDATE_MANIFEST_SHA,
        "--wsss-choice-root", str(wsss_freeze.parent),
        "--expected-wsss-freeze-sha256", WSSS_FREEZE_SHA,
        "--fully-mask-root", str(fully_masks),
        "--expected-fully-freeze-sha256", sha256(fully_freeze),
        "--output-dir", str(evaluation),
        "--bootstrap-iterations", "2000",
    ])
    manifest = {
        "schema_version": 1,
        "study": "G4 E0 common-grid validation",
        "fully_mask_freeze_sha256": sha256(fully_freeze),
        "evaluation_audit_sha256": sha256(evaluation / "evaluation_audit.json"),
        "summary_sha256": sha256(evaluation / "summary.json"),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "run_manifest_sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
