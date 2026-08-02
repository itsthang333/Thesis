from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from bind_skelex_reconstruction_selector_s8_wrapper import (
    ADDENDUM_SHA256,
    CORRECTION_SOURCE_COMMIT,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    TEMPLATE_SHA256,
    bind,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_v1.py"
)


def _canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s8_template_protocol_and_addendum_constants_match_repository() -> None:
    assert _canonical_sha(TEMPLATE) == TEMPLATE_SHA256
    protocol = ROOT / (
        "artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json"
    )
    addendum = ROOT / (
        "artifacts/research_protocols/"
        "skelex_reconstruction_selector_s8_v1_auditor_completeness_addendum.json"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256(addendum.read_bytes()).hexdigest() == ADDENDUM_SHA256
    assert len(SOURCE_COMMIT) == len(CORRECTION_SOURCE_COMMIT) == 40


def test_s8_binding_is_invertible_and_one_time(tmp_path: Path) -> None:
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
        kernel_version=1,
    )
    source = output.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 1" in source
    assert "LAUNCH_BINDING_READY = True" in source
    assert f'CHECKOUT_COMMIT = "{checkout}"' in source
    assert result["inverse_reconstruction_matches_template"] is True
    assert result["validation_gt_read"] is False
    with pytest.raises(FileExistsError):
        bind(
            TEMPLATE,
            output,
            binding_path,
            repository_root=ROOT,
            checkout_commit=checkout,
            kernel_version=1,
        )
