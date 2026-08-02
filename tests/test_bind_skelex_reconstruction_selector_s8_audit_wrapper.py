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
    assert result["transport_dataset"] == "itsthang333/btxrd-skelex-s8-v1-frozen-output"
    assert result["transport_archive_sha256"] == "c516437824ff7d7e32594bfe02e3f654d98d9976d2ddb40595641bf5f8ca1737"
    assert result["transport_correction_sha256"] == "ee42bbe43d4f81ffba570a8aa46454cb55acbf9bb6338ed4d746aaf38ce32d1d"
    assert result["null_device_correction_sha256"] == "be1bb0bf1c253ded4999e78fea164abbfb1c4e1ae412e94e55b8ba5fe8e03725"
    with pytest.raises(FileExistsError):
        bind(
            TEMPLATE,
            output,
            binding_path,
            repository_root=ROOT,
            checkout_commit=checkout,
            kernel_version=1,
        )
