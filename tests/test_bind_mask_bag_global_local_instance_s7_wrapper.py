from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bind_mask_bag_global_local_instance_s7_wrapper import (
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    TEMPLATE_SHA256,
    bind,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "project/kaggle_wrappers/run_mask_bag_global_local_instance_s7_v1.py"
)


def _canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s7_template_and_protocol_constants_match_repository() -> None:
    assert _canonical_sha(TEMPLATE) == TEMPLATE_SHA256
    protocol = (
        ROOT
        / "artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1.json"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert len(SOURCE_COMMIT) == 40


def test_s7_binding_is_invertible_and_one_time(tmp_path) -> None:
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    checkout = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip()
    )
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
    assert result["accepted_bag_probability_preserved"] is True
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
