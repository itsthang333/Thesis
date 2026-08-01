from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_mask_bag_normal_only_direct_anomaly_n1_output import _verify_binding
from bind_mask_bag_normal_only_direct_anomaly_n1_wrapper import (
    TEMPLATE_SHA256,
    bind,
    canonical_bytes,
    digest,
)


TEMPLATE = (
    PROJECT
    / "kaggle_wrappers"
    / "run_mask_bag_normal_only_direct_anomaly_n1_v1.py"
)
PROTOCOL = (
    ROOT
    / "artifacts"
    / "research_protocols"
    / "rad_dino_mask_bag_normal_only_direct_anomaly_n1_v1.json"
)


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def test_n1_wrapper_is_fail_closed_gt_blind_and_audits_before_completion() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "KERNEL_VERSION = 0" in source
    assert "LAUNCH_BINDING_READY = False" in source
    assert 'CHECKOUT_COMMIT = "UNBOUND"' in source
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("from ", "import "))
    )
    assert "evaluate_mask_bag_selector_arm" not in import_lines
    assert "validation_gt" not in import_lines
    assert source.index("verify_t4x2()") < source.index("prepare_split()")
    main_body = source[source.index("def main() -> None:") :]
    assert main_body.index("run_mask_bag_normal_only_direct_anomaly_n1.py") < main_body.index(
        "audit_mask_bag_normal_only_direct_anomaly_n1_output.py"
    )
    assert main_body.index("independent_gt_blind_output_audit.json") < main_body.index(
        "audit_output(source_hashes, cache, baseline, t4)"
    )
    assert digest(canonical_bytes(TEMPLATE)) == TEMPLATE_SHA256


def test_n1_binder_changes_exactly_three_fields_and_matches_auditor(tmp_path: Path) -> None:
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
    replacements = {
        b"KERNEL_VERSION = 0": b"KERNEL_VERSION = 1",
        b"LAUNCH_BINDING_READY = False": b"LAUNCH_BINDING_READY = True",
        b'CHECKOUT_COMMIT = "UNBOUND"': f'CHECKOUT_COMMIT = "{_head()}"'.encode(),
    }
    reconstructed = bound
    for old, new in reversed(list(replacements.items())):
        reconstructed = reconstructed.replace(new, old)
    assert reconstructed == canonical_bytes(TEMPLATE)
    assert result["replacement_count"] == 3
    assert result["inverse_reconstruction_matches_template"] is True
    assert result["bound_wrapper_sha256"] == digest(bound)
    assert json.loads(binding_path.read_text(encoding="utf-8")) == result
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert _verify_binding(binding_path, protocol) == result


def test_n1_binder_refuses_invalid_version_or_existing_output(tmp_path: Path) -> None:
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
