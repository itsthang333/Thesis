from __future__ import annotations

"""Fail-closed Kaggle bootstrap for the S5 SKELEX selector ablation."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from urllib.request import urlopen


KERNEL = "itsthang333/btxrd-skelex-mask-bag-selector-s5-v1"
KERNEL_VERSION = 0
LAUNCH_BINDING_READY = False
CHECKOUT_COMMIT = "UNBOUND"
REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "61927cc84ef2340768ea37f9686bf8036c81db30"
CORRECTION_SOURCE_COMMIT = "664578758225501dc163a6fc35d9ecdb9a1947d7"
PROTOCOL_RELATIVE = Path("artifacts/research_protocols/skelex_mask_bag_selector_s5_v1.json")
PROTOCOL_SHA256 = "036e9d1d52a4ba1ee8e2a51cd19ca4fef597c6c7ad0256e7c729c7888ea24280"
NUMERIC_ADDENDUM_RELATIVE = Path(
    "artifacts/research_protocols/"
    "skelex_mask_bag_selector_s5_v1_numeric_correction_addendum.json"
)
NUMERIC_ADDENDUM_SHA256 = "ded254883a13da9ec0b961970ebacbd2b61badd04c644b7b9c64747a6abd2f72"
AUDITOR_RELATIVE = Path("project/audit_skelex_mask_bag_selector_s5_output.py")
AUDITOR_SHA256 = "dbf84451ae32b5fd819af53c48f3357da0c236defdaa5eda2d1b787640e01049"
IMPLEMENTATION_SOURCE_OVERRIDES = {
    "project/models/skelex_mask_bag_descriptor.py":
        "c01197750f289aab31d4cb34c914fd211c70c49214789aa2379f4bcdbb1899b3",
    "project/run_skelex_mask_bag_selector_s5.py":
        "b23b61db262bb67b5ef3faefcd0ae3565c35bfb124925a279060a10340f070bc",
    "project/audit_skelex_mask_bag_selector_s5_output.py":
        "dbf84451ae32b5fd819af53c48f3357da0c236defdaa5eda2d1b787640e01049",
    "tests/test_skelex_mask_bag_descriptor.py":
        "385a4c17eeeaccd55f9a19864b28dd95b6bd41cfaf24eb57c6432755373ee779",
}
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
TRAIN_CANDIDATE_MANIFEST_SHA256 = "ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1"
TRAIN_PSEUDO_MANIFEST_SHA256 = "5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21"
VAL_CANDIDATE_MANIFEST_SHA256 = "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
VAL_PSEUDO_MANIFEST_SHA256 = "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
CACHE_MANIFEST_SHA256 = "8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e"
CACHE_WRAPPER_AUDIT_SHA256 = "cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd"
BASELINE = {
    "freeze": "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3",
    "checkpoint": "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069",
    "manifest": "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee",
    "source_commit": "fda732941664e67d4b87a8c3cba071b6979b2214",
    "protocol": "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555",
}
SKELEX_REVISION = "368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e"
SKELEX_FILES = {
    "config.json": "b48411f4313c2ee6357586b57d185befac8c7c77cc475bc2188ec4487b1bc6f7",
    "preprocessor_config.json": "a250969c94afba52d785a0e08dd36e13aeda97c4dd2b7fd0d24b457288536cea",
    "model.safetensors": "81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8",
}
SKELEX_WEIGHT_BYTES = 1_318_230_232
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
ARMS = (
    "geometry_v3_plus_upstream_equal_rank",
    "geometry_v3_plus_upstream_plus_skelex_equal_rank",
)
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORK / "s5_source"
RUNTIME = WORK / "s5_runtime"
OUTPUT = WORK / "btxrd_skelex_mask_bag_selector_s5_v1"


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print(f"$ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_and_verify() -> tuple[dict[str, str], dict[str, object]]:
    if not LAUNCH_BINDING_READY or KERNEL_VERSION < 1:
        raise RuntimeError("S5 launch binding is not frozen")
    if len(CHECKOUT_COMMIT) != 40 or any(c not in "0123456789abcdef" for c in CHECKOUT_COMMIT):
        raise RuntimeError("S5 checkout commit is invalid")
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(SOURCE)], cwd=WORK)
    run(["git", "checkout", "--detach", CHECKOUT_COMMIT], cwd=SOURCE)
    run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, CHECKOUT_COMMIT], cwd=SOURCE)
    run(
        ["git", "merge-base", "--is-ancestor", CORRECTION_SOURCE_COMMIT, CHECKOUT_COMMIT],
        cwd=SOURCE,
    )
    protocol_path = SOURCE / PROTOCOL_RELATIVE
    if hash_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("S5 protocol hash mismatch")
    addendum_path = SOURCE / NUMERIC_ADDENDUM_RELATIVE
    if hash_file(addendum_path) != NUMERIC_ADDENDUM_SHA256:
        raise RuntimeError("S5 numeric correction addendum hash mismatch")
    if canonical_hash(SOURCE / AUDITOR_RELATIVE) != AUDITOR_SHA256:
        raise RuntimeError("S5 auditor hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_PRELAUNCH"
        or protocol.get("scientific_source", {}).get("commit") != SOURCE_COMMIT
        or protocol.get("inherits_terminal_evidence", {}).get("collaborator_output_accessed") is not False
        or protocol.get("image_label_only_boundary", {}).get("annotation_bytes_opened_or_hashed") is not False
    ):
        raise RuntimeError("S5 protocol safety/provenance mismatch")
    hashes = protocol.get("canonical_lf_source_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("S5 protocol source inventory is missing")
    addendum_overrides = addendum.get("canonical_lf_source_overrides", {})
    if (
        addendum.get("status") != "FROZEN_IMPLEMENTATION_ONLY_CORRECTION"
        or addendum.get("experiment_id") != "EXP-20260802-codex-s5-skelex-selector-v1"
        or addendum.get("correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or addendum.get("scientific_source_commit") != SOURCE_COMMIT
        or addendum.get("scientific_protocol_sha256") != PROTOCOL_SHA256
        or not isinstance(addendum_overrides, dict)
        or set(addendum_overrides) != set(IMPLEMENTATION_SOURCE_OVERRIDES)
    ):
        raise RuntimeError("S5 numeric correction addendum provenance mismatch")
    verified_hashes = dict(hashes)
    for relative, corrected in IMPLEMENTATION_SOURCE_OVERRIDES.items():
        entry = addendum_overrides.get(relative)
        if (
            not isinstance(entry, dict)
            or entry.get("previous_sha256") != hashes.get(relative)
            or entry.get("corrected_sha256") != corrected
        ):
            raise RuntimeError(f"S5 numeric correction override mismatch: {relative}")
        verified_hashes[relative] = corrected
    for relative, expected in verified_hashes.items():
        path = SOURCE / relative
        if not path.is_file() or canonical_hash(path) != expected:
            raise RuntimeError(f"S5 source hash mismatch: {relative}")
    return verified_hashes, protocol


def verify_t4x2() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S5 requires exactly two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"S5 requires T4 x2, got {names}")
    checksums: list[float] = []
    for index in range(2):
        torch.manual_seed(520 + index)
        layer = torch.nn.Conv2d(3, 5, 3, padding=1).to(f"cuda:{index}").eval()
        values = torch.arange(3072, dtype=torch.float32, device=f"cuda:{index}").reshape(1, 3, 32, 32)
        with torch.inference_mode():
            result = layer(values)
        if not torch.isfinite(result).all():
            raise RuntimeError(f"S5 non-finite T4 guard on cuda:{index}")
        checksums.append(float(result.sum().cpu()))
    return {"cuda_device_count": 2, "cuda_device_names": names, "real_convolution_checksums": checksums}


def install_runtime() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            f"transformers=={EXPECTED_TRANSFORMERS_VERSION}",
        ],
        cwd=SOURCE,
    )
    run(
        [
            sys.executable,
            "-c",
            (
                "import transformers; "
                f"assert transformers.__version__ == '{EXPECTED_TRANSFORMERS_VERSION}', "
                "transformers.__version__"
            ),
        ],
        cwd=SOURCE,
    )


def download_skelex() -> tuple[Path, dict[str, object]]:
    root = RUNTIME / "skelex"
    root.mkdir(parents=True, exist_ok=False)
    for name, expected in SKELEX_FILES.items():
        url = f"https://huggingface.co/skhoha/SKELEX/resolve/{SKELEX_REVISION}/{name}?download=true"
        target = root / name
        with urlopen(url, timeout=600) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
        if hash_file(target) != expected:
            raise RuntimeError(f"S5 public model hash mismatch: {name}")
    if (root / "model.safetensors").stat().st_size != SKELEX_WEIGHT_BYTES:
        raise RuntimeError("S5 public model size mismatch")
    return root, {
        "repository": "skhoha/SKELEX",
        "revision": SKELEX_REVISION,
        "license": "CC-BY-NC-ND-4.0",
        "files": SKELEX_FILES,
        "checkpoint_redistributed": False,
    }


def run_static_tests() -> None:
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "project/models/skelex_mask_bag_descriptor.py",
            "project/run_skelex_mask_bag_selector_s5.py",
            "project/audit_skelex_mask_bag_selector_s5_output.py",
        ],
        cwd=SOURCE,
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_skelex_mask_bag_descriptor.py",
            "tests/test_rad_dino_mask_bag_mil.py",
            "tests/test_evaluate_mask_bag_selector_arm.py",
            "tests/test_compare_mask_bag_evaluated_arms.py",
            "tests/test_kaggle_wrapper_skelex_s5_v1.py",
        ],
        cwd=SOURCE,
    )


def prepare_split() -> Path:
    source = SOURCE / "artifacts/kaggle/wsl_source_consensus_val_v1/frozen_split_manifest.csv"
    if hash_file(source) != GIT_SPLIT_SHA256 or b"\r" in source.read_bytes():
        raise RuntimeError("S5 canonical split mismatch")
    target = RUNTIME / "frozen_split_manifest.csv"
    target.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    if hash_file(target) != SPLIT_SHA256:
        raise RuntimeError("S5 frozen split reconstruction mismatch")
    return target


def find_dataset_root() -> Path:
    candidates = [
        INPUT / "btxrd-raw" / "BTXRD",
        INPUT / "datasets" / "itsthang333" / "btxrd-raw" / "BTXRD",
        *sorted(INPUT.glob("**/BTXRD")),
    ]
    valid = {
        path.resolve()
        for path in candidates
        if (path / "images").is_dir()
        and ((path / "dataset.csv").is_file() or (path / "dataset.xlsx").is_file())
    }
    if len(valid) != 1:
        raise RuntimeError(f"Expected one S5 BTXRD root, found {sorted(map(str, valid))}")
    return next(iter(valid))


def find_candidate_root(
    *, manifest_sha: str, pseudo_sha: str, expected_split: str
) -> tuple[Path, dict[str, object]]:
    roots = []
    for manifest in INPUT.rglob("candidate_diagnostics_manifest.csv"):
        root = manifest.parent
        pseudo = root / "pseudo_mask_manifest.csv"
        if hash_file(manifest) == manifest_sha and pseudo.is_file() and hash_file(pseudo) == pseudo_sha:
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S5 {expected_split} candidate root, found {roots}")
    return roots[0], {
        "split": expected_split,
        "candidate_manifest_sha256": manifest_sha,
        "pseudo_manifest_sha256": pseudo_sha,
    }


def find_baseline() -> tuple[Path, dict[str, object]]:
    roots = []
    for freeze in INPUT.rglob("prediction_freeze.json"):
        root = freeze.parent
        checkpoint = root / "rad_dino_mask_bag_mil.pt"
        manifest = root / "predictions" / "prediction_manifest.csv"
        if (
            hash_file(freeze) == BASELINE["freeze"]
            and checkpoint.is_file()
            and hash_file(checkpoint) == BASELINE["checkpoint"]
            and manifest.is_file()
            and hash_file(manifest) == BASELINE["manifest"]
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S5 baseline root, found {roots}")
    return roots[0], dict(BASELINE)


def find_cache() -> tuple[Path, dict[str, object]]:
    roots = []
    for freeze in INPUT.rglob("selector_cache_freeze.json"):
        root = freeze.parent
        manifest = root / "selector_cache_manifest.csv"
        audit = root / "wrapper_output_audit.json"
        if (
            hash_file(freeze) == CACHE_FREEZE_SHA256
            and manifest.is_file()
            and hash_file(manifest) == CACHE_MANIFEST_SHA256
            and audit.is_file()
            and hash_file(audit) == CACHE_WRAPPER_AUDIT_SHA256
        ):
            roots.append(root.resolve())
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one S5 selector-cache root, found {roots}")
    root = roots[0]
    freeze = json.loads((root / "selector_cache_freeze.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "wrapper_output_audit.json").read_text(encoding="utf-8"))
    for payload in (freeze, audit):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S5 selector-cache safety mismatch")
    if freeze.get("cohort") != {"train": 2981, "validation": 371}:
        raise RuntimeError("S5 selector-cache cohort mismatch")
    return root, {
        "freeze_sha256": CACHE_FREEZE_SHA256,
        "manifest_sha256": CACHE_MANIFEST_SHA256,
        "wrapper_audit_sha256": CACHE_WRAPPER_AUDIT_SHA256,
    }


def write_binding(source_hashes: dict[str, str]) -> Path:
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "experiment_id": "EXP-20260802-codex-s5-skelex-selector-v1",
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "numeric_correction_source_commit": CORRECTION_SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "numeric_correction_addendum_sha256": NUMERIC_ADDENDUM_SHA256,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "collaborator_output_accessed": False,
        "annotation_paths_resolved": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    path = RUNTIME / "launch_binding.json"
    path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
        raise RuntimeError(f"S5 output cohort mismatch: {path}")
    return rows


def audit_wrapper_output(
    *,
    source_hashes: dict[str, str],
    t4: dict[str, object],
    model: dict[str, object],
    train_candidates: dict[str, object],
    val_candidates: dict[str, object],
    baseline: dict[str, object],
    cache: dict[str, object],
) -> None:
    pair_path = OUTPUT / "prediction_pair_freeze.json"
    independent_path = OUTPUT / "independent_gt_blind_output_audit.json"
    binding_path = OUTPUT / "launch_binding.json"
    run_path = OUTPUT / "run_manifest.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        pair.get("source_commit") != SOURCE_COMMIT
        or pair.get("protocol_sha256") != PROTOCOL_SHA256
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or set(pair.get("arms", {})) != set(ARMS)
        or independent.get("status")
        != "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED"
        or independent.get("validation_predictions_per_arm") != 371
        or independent.get("pair_freeze_sha256") != hash_file(pair_path)
        or binding.get("source_hashes") != source_hashes
        or binding.get("numeric_correction_source_commit") != CORRECTION_SOURCE_COMMIT
        or binding.get("numeric_correction_addendum_sha256") != NUMERIC_ADDENDUM_SHA256
        or run_manifest.get("cohort") != {"train": 2981, "validation": 371}
    ):
        raise RuntimeError("S5 frozen output contract mismatch")
    for payload in (pair, independent, binding, run_manifest):
        if (
            payload.get("validation_gt_read") is not False
            or payload.get("consumer_trained") is not False
            or payload.get("test_evaluated") is not False
        ):
            raise RuntimeError("S5 output safety mismatch")
    physical_maps = 0
    physical_scores = 0
    for arm in ARMS:
        freeze_path = OUTPUT / arm / "prediction_freeze.json"
        if hash_file(freeze_path) != pair["arms"][arm]:
            raise RuntimeError(f"S5 arm freeze mismatch: {arm}")
        for row in _rows(OUTPUT / arm / "predictions" / "prediction_manifest.csv", 371):
            path = OUTPUT / arm / "predictions" / row["map_path"]
            if not path.is_file() or hash_file(path) != row["map_sha256"]:
                raise RuntimeError(f"S5 map hash mismatch: {arm}/{row['image_id']}")
            physical_maps += 1
        for row in _rows(OUTPUT / arm / "candidate_scores" / "candidate_score_manifest.csv", 371):
            path = OUTPUT / arm / "candidate_scores" / row["score_path"]
            if not path.is_file() or hash_file(path) != row["score_sha256"]:
                raise RuntimeError(f"S5 score hash mismatch: {arm}/{row['image_id']}")
            physical_scores += 1
    evidence_rows = _rows(OUTPUT / "skelex_score_evidence" / "evidence_manifest.csv", 371)
    for row in evidence_rows:
        path = OUTPUT / "skelex_score_evidence" / row["evidence_path"]
        if not path.is_file() or hash_file(path) != row["evidence_sha256"]:
            raise RuntimeError(f"S5 evidence hash mismatch: {row['image_id']}")
    wrapper_audit = {
        "kernel": KERNEL,
        "kernel_version": KERNEL_VERSION,
        "bound_wrapper_sha256": canonical_hash(Path(__file__)),
        "checkout_commit": CHECKOUT_COMMIT,
        "scientific_source_commit": SOURCE_COMMIT,
        "numeric_correction_source_commit": CORRECTION_SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "numeric_correction_addendum_sha256": NUMERIC_ADDENDUM_SHA256,
        "independent_auditor_sha256": AUDITOR_SHA256,
        "source_hashes": source_hashes,
        "t4x2": t4,
        "public_skelex_model": model,
        "train_candidates": train_candidates,
        "validation_candidates": val_candidates,
        "baseline": baseline,
        "selector_cache": cache,
        "launch_binding_sha256": hash_file(binding_path),
        "prediction_pair_freeze_sha256": hash_file(pair_path),
        "independent_gt_blind_output_audit_sha256": hash_file(independent_path),
        "run_manifest_sha256": hash_file(run_path),
        "physical_prediction_maps_verified": physical_maps,
        "physical_candidate_scores_verified": physical_scores,
        "physical_descriptor_evidence_verified": len(evidence_rows),
        "collaborator_output_accessed": False,
        "annotation_paths_resolved": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "python": platform.python_version(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "wrapper_output_audit.json").write_text(
        json.dumps(wrapper_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.update(
        {
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    RUNTIME.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes, _protocol = clone_and_verify()
        t4 = verify_t4x2()
        install_runtime()
        # Static and synthetic tests run before the large public-model download
        # and before resolving any BTXRD input path.
        run_static_tests()
        skelex_root, model_audit = download_skelex()
        split = prepare_split()
        dataset_root = find_dataset_root()
        train_root, train_audit = find_candidate_root(
            manifest_sha=TRAIN_CANDIDATE_MANIFEST_SHA256,
            pseudo_sha=TRAIN_PSEUDO_MANIFEST_SHA256,
            expected_split="train",
        )
        val_root, val_audit = find_candidate_root(
            manifest_sha=VAL_CANDIDATE_MANIFEST_SHA256,
            pseudo_sha=VAL_PSEUDO_MANIFEST_SHA256,
            expected_split="val",
        )
        baseline_root, baseline_audit = find_baseline()
        cache_root, cache_audit = find_cache()
        binding_path = write_binding(source_hashes)
        run(
            [
                sys.executable,
                str(SOURCE / "project/run_skelex_mask_bag_selector_s5.py"),
                "--dataset-root", str(dataset_root),
                "--split-manifest", str(split),
                "--expected-split-sha256", SPLIT_SHA256,
                "--skelex-model-dir", str(skelex_root),
                "--expected-skelex-config-sha256", SKELEX_FILES["config.json"],
                "--expected-skelex-preprocessor-sha256", SKELEX_FILES["preprocessor_config.json"],
                "--expected-skelex-weight-sha256", SKELEX_FILES["model.safetensors"],
                "--selector-cache-root", str(cache_root),
                "--expected-selector-cache-freeze-sha256", CACHE_FREEZE_SHA256,
                "--baseline-root", str(baseline_root),
                "--expected-baseline-checkpoint-sha256", BASELINE["checkpoint"],
                "--expected-baseline-freeze-sha256", BASELINE["freeze"],
                "--expected-baseline-source-commit", BASELINE["source_commit"],
                "--expected-baseline-protocol-sha256", BASELINE["protocol"],
                "--train-candidate-root", str(train_root),
                "--train-candidate-manifest-sha256", TRAIN_CANDIDATE_MANIFEST_SHA256,
                "--train-pseudo-manifest-sha256", TRAIN_PSEUDO_MANIFEST_SHA256,
                "--val-candidate-root", str(val_root),
                "--val-candidate-manifest-sha256", VAL_CANDIDATE_MANIFEST_SHA256,
                "--val-pseudo-manifest-sha256", VAL_PSEUDO_MANIFEST_SHA256,
                "--source-commit", SOURCE_COMMIT,
                "--protocol-sha256", PROTOCOL_SHA256,
                "--output-dir", str(OUTPUT),
                "--encoder-batch-size", "4",
                "--train-batch-size", "16",
                "--epochs", "16",
                "--learning-rate", "0.0003",
                "--weight-decay", "0.0001",
                "--instance-loss-weight", "0.25",
                "--consistency-loss-weight", "0.10",
                "--instance-warmup-epochs", "2",
                "--maximum-candidates", "81",
                "--seed", "42",
            ],
            cwd=SOURCE,
        )
        shutil.copy2(binding_path, OUTPUT / "launch_binding.json")
        run(
            [
                sys.executable,
                str(SOURCE / AUDITOR_RELATIVE),
                "--output-root", str(OUTPUT),
                "--protocol", str(SOURCE / PROTOCOL_RELATIVE),
                "--launch-binding", str(OUTPUT / "launch_binding.json"),
                "--split-manifest", str(split),
                "--selector-cache-root", str(cache_root),
                "--baseline-root", str(baseline_root),
                "--val-candidate-root", str(val_root),
                "--audit-output", str(OUTPUT / "independent_gt_blind_output_audit.json"),
            ],
            cwd=SOURCE,
        )
        audit_wrapper_output(
            source_hashes=source_hashes,
            t4=t4,
            model=model_audit,
            train_candidates=train_audit,
            val_candidates=val_audit,
            baseline=baseline_audit,
            cache=cache_audit,
        )
    finally:
        # Public SKELEX weights are deliberately removed and never enter output.
        for path in (SOURCE, RUNTIME):
            if path.exists() and path.resolve().parent == WORK.resolve():
                shutil.rmtree(path)


if __name__ == "__main__":
    main()
