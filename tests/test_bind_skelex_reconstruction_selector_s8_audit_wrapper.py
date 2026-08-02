from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from bind_skelex_reconstruction_selector_s8_audit_wrapper import bind


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_audit_v1.py"


def test_s8_audit_only_binding_is_invertible_and_one_time(tmp_path: Path) -> None:
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
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
    assert result["prediction_changed"] is False
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
