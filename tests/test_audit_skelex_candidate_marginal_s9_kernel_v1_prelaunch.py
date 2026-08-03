from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from audit_skelex_candidate_marginal_s9_kernel_v1_prelaunch import audit
from bind_skelex_candidate_marginal_s9_wrapper import bind


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "project/kaggle_wrappers/run_skelex_candidate_marginal_s9_v1.py"


def _package(tmp_path: Path) -> tuple[Path, str]:
    checkout = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    package = tmp_path / "package"
    package.mkdir()
    wrapper = package / "btxrd-skelex-candidate-marginal-s9-v1.py"
    binding = package / "launch_binding.json"
    bind(
        TEMPLATE,
        wrapper,
        binding,
        repository_root=ROOT,
        checkout_commit=checkout,
        kernel_version=2,
    )
    metadata = {
        "id": "itsthang333/btxrd-skelex-candidate-marginal-s9-v1",
        "title": "BTXRD SKELEX Candidate Marginal S9 V1",
        "code_file": wrapper.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "itsthang333/btxrd-raw",
            "itsthang333/btxrd-mask-bag-geometry-v3-train-gallery-v1",
            "itsthang333/btxrd-mask-bag-selector-baseline-v1",
        ],
        "kernel_sources": [
            "itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1"
        ],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (package / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return package, checkout


def test_s9_prelaunch_audit_accepts_exact_bound_package(tmp_path: Path) -> None:
    package, checkout = _package(tmp_path)
    result = audit(package, ROOT)
    assert result["authorized_launch"] is True
    assert result["checkout_commit"] == checkout
    assert result["kernel_version"] == 2
    assert result["validation_prediction_created"] is False
    assert result["validation_gt_read"] is False
    assert result["consumer_trained"] is False
    assert result["test_evaluated"] is False


def test_s9_prelaunch_audit_rejects_metadata_transport_drift(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    metadata_path = package / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["kernel_sources"] = []
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        audit(package, ROOT)
