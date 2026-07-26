from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "btxrd_rad_dino_square_geometry_correction_val_v1"
SOURCE_REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "07fc153a2924e998f4e9cbbd2fca7cf22f8fcf12"
PROTOCOL_RELATIVE = (
    "artifacts/research_protocols/"
    "rad_dino_square_geometry_correction_val_v1.json"
)
PROTOCOL_SHA256 = (
    "391c02cead32cb5708dfcac484478e37fbc499912ff993c0b41ab9885686e109"
)
GIT_SPLIT_SHA256 = (
    "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
)
FROZEN_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
SOURCE_HASHES = {
    "tools/reproject_frozen_square_probe_maps.py": (
        "aad0a03404721e32513b8675226e8586003aca493fe548ffe9ce9bd8837bdc1a"
    ),
    "run_rad_dino_dense_mil_probe.py": (
        "88084d9bfb8ec9bae14dfa558d06d113c926a94d2ef7851c0d254e05050f08fe"
    ),
    "models/mae_reconstruction.py": (
        "7baa1a664d0523d454ab19bc1420959ef5115e31da2a242345c29f1dab7417f9"
    ),
    "mae_reconstruction_io.py": (
        "1b14cae95354db53fc2117254701920870510d4b8bb1005f1400340ad4c8859c"
    ),
    "compare_nominal_patch_memory_arms.py": (
        "4ec0b65c33d47dda20317b94d577489038bf87c8c235bb03d6d9a1d0497a035e"
    ),
    "datasets/btxrd.py": (
        "d8f0804be4e81cdb4d58e4673708c1067eb7d9b49b42bb78cb6051188c156001"
    ),
    "datasets/common.py": (
        "1927eb358a9db1a0e9c2571be5e222c3edd9d69814dfb4bc3375bd3f8593b98a"
    ),
    "datasets/__init__.py": (
        "6b478bbb7af1b77f31e1aa96f197aee1138623b0c87e9385b4b4323542432784"
    ),
}
RUNS = {
    "dense_mil": {
        "probe_id": "rad_dino_dense_mil_probe_val_v1_geometry_corrected",
        "checkpoint_sha256": (
            "945cff3221190014437a0a34dda88935477d6ad6ea07fb03cec54e39c5801d3e"
        ),
        "run_manifest_sha256": (
            "b8954dc237f73cf4aa0e5586f945961e1995af14cb97810afaae75b8355b6d19"
        ),
        "prediction_freeze_sha256": (
            "8b5361e9c610907d6ab6cab089ecb9307440b05507427cf10dea41d333640fb3"
        ),
    },
    "insight": {
        "probe_id": "rad_dino_insight_probe_val_v1_geometry_corrected",
        "checkpoint_sha256": (
            "35bc926bb5768e1f2879dd7ae8ce37ac1e8538ed1223bcd65e81424f2f950286"
        ),
        "run_manifest_sha256": (
            "7d8cec5f1fd84dc93b6e09ad667dfb5a455bccb771ae74fe1e73537d5aec5b39"
        ),
        "prediction_freeze_sha256": (
            "8660872582305a0f783711db1ddf06af9cf90215bcd138f373fdc8e3ff130f8c"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkout_source() -> tuple[Path, Path]:
    repository = WORK / "thesis_source"
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SOURCE_REPOSITORY,
            str(repository),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "--detach", SOURCE_COMMIT],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source commit mismatch: {actual}")
    project = repository / "project"
    for relative, expected in SOURCE_HASHES.items():
        if sha256(project / relative) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    protocol = repository / PROTOCOL_RELATIVE
    if sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("Protocol hash mismatch")
    parsed = json.loads(protocol.read_text(encoding="utf-8"))
    if (
        parsed["status"] != "predeclared_before_corrected_evaluation"
        or parsed["data_contract"]["test_evaluated"]
    ):
        raise RuntimeError("Protocol status/data contract mismatch")
    git_split = repository / "artifacts/data_audit/split_manifest.csv"
    if sha256(git_split) != GIT_SPLIT_SHA256 or b"\r" in git_split.read_bytes():
        raise RuntimeError("Canonical Git split mismatch")
    split = WORK / "frozen_split_manifest.csv"
    split.write_bytes(git_split.read_bytes().replace(b"\n", b"\r\n"))
    if sha256(split) != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen split hash mismatch")
    return project.resolve(), split.resolve()


def find_btxrd_root() -> Path:
    candidates = [INPUT / "btxrd-raw/BTXRD", *sorted(INPUT.glob("**/BTXRD"))]
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("BTXRD root not found")


def find_frozen_run(config: dict[str, str]) -> Path:
    matches: list[Path] = []
    for manifest_path in INPUT.rglob("run_manifest.json"):
        # A later kernel source may embed compact evidence from an earlier
        # run inside its checked-out repository. Only direct Kaggle kernel
        # output is admissible as the frozen source payload.
        if "thesis_source" in manifest_path.parts:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if manifest.get("checkpoint_sha256") != config["checkpoint_sha256"]:
            continue
        root = manifest_path.parent
        freeze = root / "prediction_freeze.json"
        if (
            sha256(manifest_path) == config["run_manifest_sha256"]
            and freeze.is_file()
            and sha256(freeze) == config["prediction_freeze_sha256"]
        ):
            matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exact frozen source run, found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def run_correction(
    *,
    project: Path,
    split: Path,
    data_root: Path,
    name: str,
    config: dict[str, str],
    log: Path,
) -> Path:
    source_run = find_frozen_run(config)
    destination = OUTPUT / name
    command = [
        sys.executable,
        str(project / "tools/reproject_frozen_square_probe_maps.py"),
        "--run-root",
        str(source_run),
        "--dataset-root",
        str(data_root),
        "--split-manifest",
        str(split),
        "--expected-split-sha256",
        FROZEN_SPLIT_SHA256,
        "--protocol-sha256",
        PROTOCOL_SHA256,
        "--probe-id",
        config["probe_id"],
        "--expected-checkpoint-sha256",
        config["checkpoint_sha256"],
        "--output-dir",
        str(destination),
    ]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=project,
            env=os.environ.copy(),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return destination


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    OUTPUT.mkdir()
    project, split = checkout_source()
    data_root = find_btxrd_root()
    log = OUTPUT / "execution.log"
    with log.open("a", encoding="utf-8") as handle:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_square_probe_reprojection.py",
            "-q",
        ]
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=project.parent,
            env=os.environ.copy(),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    results = {
        name: run_correction(
            project=project,
            split=split,
            data_root=data_root,
            name=name,
            config=config,
            log=log,
        )
        for name, config in RUNS.items()
    }
    index = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT,
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "results": {
            name: {
                "path": path.name,
                "run_manifest_sha256": sha256(path / "run_manifest.json"),
                "prediction_freeze_sha256": sha256(path / "prediction_freeze.json"),
            }
            for name, path in results.items()
        },
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (OUTPUT / "run_index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
