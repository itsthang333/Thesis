from __future__ import annotations

"""Run replayable G4 Stage A and validation-only Stage B on Kaggle."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CANDIDATE_MANIFEST_SHA = "30e734e223839ee2ed7b445c71b3731a39f0549ee88e2ba77f7bf722d8d9943d"
G1_FREEZE_SHA = "5942a6df949f51fc659313416a9c1156db56f6dad8b26b6957a58fc0ad6138ff"
BASELINE_FREEZE_SHA = "a75c0388346b6a1a3ab94f3ddd700a2e495c36be257b63712e38fc451784a620"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_file(root: Path, name: str, expected_sha: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and sha256(path) == expected_sha]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {name} with SHA {expected_sha}, found {len(matches)}"
        )
    return matches[0]


def unique_project(root: Path) -> Path:
    matches = [
        path.parent
        for path in root.rglob("freeze_g4_offline_ablations.py")
        if path.parent.name == "project"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one G4 project directory, found {len(matches)}")
    return matches[0]


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    output_root = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working")) / "g4_offline"
    output_root.mkdir(parents=True, exist_ok=False)

    project = unique_project(input_root)
    split = unique_file(input_root, "canonical_split_manifest_85511.csv", SPLIT_SHA)
    candidate_manifest = unique_file(
        input_root, "candidate_diagnostics_manifest.csv", CANDIDATE_MANIFEST_SHA
    )
    g1_freeze = unique_file(input_root, "diagnostic_freeze.json", G1_FREEZE_SHA)
    baseline_freeze = unique_file(input_root, "prediction_freeze.json", BASELINE_FREEZE_SHA)
    candidate_root = candidate_manifest.parent
    g1_root = g1_freeze.parent
    baseline_root = baseline_freeze.parent

    btxrd_candidates = [
        path for path in input_root.rglob("BTXRD")
        if path.is_dir() and (path / "images").is_dir() and (path / "Annotations").is_dir()
    ]
    if len(btxrd_candidates) != 1:
        raise RuntimeError(f"expected exactly one BTXRD dataset root, found {len(btxrd_candidates)}")
    data_root = btxrd_candidates[0]

    choices = output_root / "choices"
    run([
        sys.executable,
        str(project / "freeze_g4_offline_ablations.py"),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--candidate-root", str(candidate_root),
        "--expected-candidate-manifest-sha256", CANDIDATE_MANIFEST_SHA,
        "--g1-root", str(g1_root),
        "--expected-g1-freeze-sha256", G1_FREEZE_SHA,
        "--baseline-selection-root", str(baseline_root),
        "--expected-baseline-freeze-sha256", BASELINE_FREEZE_SHA,
        "--output-dir", str(choices),
    ])
    choice_freeze = choices / "g4_choice_freeze.json"

    evaluation = output_root / "evaluation_native"
    run([
        sys.executable,
        str(project / "evaluate_g4_offline_ablations.py"),
        "--dataset-root", str(data_root),
        "--split-manifest", str(split),
        "--expected-split-sha256", SPLIT_SHA,
        "--candidate-root", str(candidate_root),
        "--choice-root", str(choices),
        "--expected-choice-freeze-sha256", sha256(choice_freeze),
        "--output-dir", str(evaluation),
        "--primary-grid", "native",
        "--bootstrap-iterations", "2000",
        "--bootstrap-seed", "20260806",
    ])
    manifest = {
        "schema_version": 1,
        "study": "G4 replayable validation-only ablations",
        "split_sha256": SPLIT_SHA,
        "choice_freeze_sha256": sha256(choice_freeze),
        "evaluation_audit_sha256": sha256(evaluation / "evaluation_audit.json"),
        "summary_sha256": sha256(evaluation / "summary.json"),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "run_manifest_sha256": sha256(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
