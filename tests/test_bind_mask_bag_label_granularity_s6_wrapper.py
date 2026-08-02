from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from project.bind_mask_bag_label_granularity_s6_wrapper import (
    TEMPLATE_SHA256,
    bind,
    canonical_bytes,
    digest,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "project"
    / "kaggle_wrappers"
    / "run_mask_bag_label_granularity_s6_v1.py"
)


def test_s6_template_hash_and_one_time_binding(tmp_path: Path) -> None:
    assert digest(canonical_bytes(TEMPLATE)) == TEMPLATE_SHA256
    checkout = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    binding = bind(
        TEMPLATE,
        output,
        binding_path,
        repository_root=ROOT,
        checkout_commit=checkout,
        kernel_version=1,
    )
    bound = canonical_bytes(output)
    assert b"KERNEL_VERSION = 1" in bound
    assert b"LAUNCH_BINDING_READY = True" in bound
    assert checkout.encode() in bound
    assert binding["replacement_count"] == 3
    assert binding["inverse_reconstruction_matches_template"] is True
    assert binding["auditor_numeric_correction_sha256"] == (
        "b0dca40bf4f8bd933a902facb7bfdf5ec393c429672b0beb0b0594f2d15dfc63"
    )
    assert binding["bound_wrapper_sha256"] == digest(bound)
    assert json.loads(binding_path.read_text(encoding="utf-8")) == binding
    with pytest.raises(FileExistsError):
        bind(
            TEMPLATE,
            output,
            binding_path,
            repository_root=ROOT,
            checkout_commit=checkout,
            kernel_version=1,
        )


def test_s6_binder_rejects_invalid_checkout_and_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkout_commit"):
        bind(
            TEMPLATE,
            tmp_path / "a.py",
            tmp_path / "a.json",
            repository_root=ROOT,
            checkout_commit="bad",
            kernel_version=1,
        )
    with pytest.raises(ValueError, match="kernel_version"):
        bind(
            TEMPLATE,
            tmp_path / "b.py",
            tmp_path / "b.json",
            repository_root=ROOT,
            checkout_commit="0" * 40,
            kernel_version=0,
        )
