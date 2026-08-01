from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from bind_mask_bag_proposal_cluster_s4_wrapper import (
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
    / "run_mask_bag_proposal_cluster_s4_v1.py"
)


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def test_s4_binder_matches_template_and_changes_exactly_three_fields(tmp_path: Path) -> None:
    assert digest(canonical_bytes(TEMPLATE)) == TEMPLATE_SHA256
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    result = bind(
        TEMPLATE,
        output,
        binding_path,
        repository_root=ROOT,
        checkout_commit=_head(),
        kernel_version=1,
    )
    bound = canonical_bytes(output)
    reconstructed = bound
    replacements = {
        b"KERNEL_VERSION = 0": b"KERNEL_VERSION = 1",
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{_head()}"'.encode(),
    }
    for old, new in reversed(list(replacements.items())):
        reconstructed = reconstructed.replace(new, old)
    assert reconstructed == canonical_bytes(TEMPLATE)
    assert result["schema_version"] == 1
    assert result["replacement_count"] == 3
    assert result["inverse_reconstruction_matches_template"] is True
    assert result["bound_wrapper_sha256"] == digest(bound)
    assert json.loads(binding_path.read_text(encoding="utf-8")) == result


def test_s4_binder_refuses_invalid_version_or_existing_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        bind(
            TEMPLATE,
            tmp_path / "bound.py",
            tmp_path / "binding.json",
            repository_root=ROOT,
            checkout_commit=_head(),
            kernel_version=0,
        )
    output = tmp_path / "existing.py"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        bind(
            TEMPLATE,
            output,
            tmp_path / "new-binding.json",
            repository_root=ROOT,
            checkout_commit=_head(),
            kernel_version=1,
        )
