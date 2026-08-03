from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from bind_skelex_candidate_marginal_s9_wrapper import (
    CORRECTION_SHA256,
    CORRECTION_SOURCE_COMMIT,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    TEMPLATE_SHA256,
    bind,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "project/kaggle_wrappers/run_skelex_candidate_marginal_s9_v1.py"
)


def _canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s9_template_and_protocol_constants_match_repository() -> None:
    assert _canonical_sha(TEMPLATE) == TEMPLATE_SHA256
    protocol = ROOT / (
        "artifacts/research_protocols/skelex_candidate_marginal_s9_v1.json"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    correction = ROOT / (
        "artifacts/research_protocols/"
        "skelex_candidate_marginal_s9_v1_rank_exactness_correction.json"
    )
    assert hashlib.sha256(correction.read_bytes()).hexdigest() == CORRECTION_SHA256
    assert len(SOURCE_COMMIT) == 40
    assert len(CORRECTION_SOURCE_COMMIT) == 40


def test_s9_binding_is_invertible_fail_closed_and_one_time(tmp_path: Path) -> None:
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    checkout = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = bind(
        TEMPLATE,
        output,
        binding_path,
        repository_root=ROOT,
        checkout_commit=checkout,
        kernel_version=2,
    )
    source = output.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 2" in source
    assert "LAUNCH_BINDING_READY = True" in source
    assert f'CHECKOUT_COMMIT = "{checkout}"' in source
    assert result["inverse_reconstruction_matches_template"] is True
    assert result["collaborator_output_accessed"] is False
    assert result["validation_gt_read"] is False
    assert result["consumer_trained"] is False
    assert result["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        bind(
            TEMPLATE,
            output,
            binding_path,
            repository_root=ROOT,
            checkout_commit=checkout,
            kernel_version=2,
        )


@pytest.mark.parametrize(
    ("checkout", "version"),
    [("not-a-commit", 1), ("0" * 40, 0)],
)
def test_s9_binding_rejects_invalid_launch_identity(
    tmp_path: Path, checkout: str, version: int
) -> None:
    with pytest.raises(ValueError):
        bind(
            TEMPLATE,
            tmp_path / "bound.py",
            tmp_path / "binding.json",
            repository_root=ROOT,
            checkout_commit=checkout,
            kernel_version=version,
        )
