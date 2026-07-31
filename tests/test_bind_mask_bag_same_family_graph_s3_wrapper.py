from __future__ import annotations

from pathlib import Path

import pytest

from bind_mask_bag_same_family_graph_s3_wrapper import bind, canonical_bytes, digest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "project"
    / "kaggle_wrappers"
    / "run_mask_bag_same_family_graph_s3_v1.py"
)


def test_s3_binder_exactly_replaces_three_external_fields(tmp_path: Path) -> None:
    checkout = "ceb7003a29303b9840d1f30fd6378ffd964313e1"
    output = tmp_path / "bound.py"
    audit = tmp_path / "binding.json"
    result = bind(
        TEMPLATE,
        output,
        audit,
        repository_root=ROOT,
        checkout_commit=checkout,
        kernel_version=1,
    )
    payload = canonical_bytes(output)
    assert result["replacement_count"] == 3
    assert result["schema_version"] == 2
    assert result["numeric_identity_addendum_sha256"] == (
        "41e88ae7011c3f994f7d47a6a9216730ba9448ccb6f9fc8599d277a0679f0d51"
    )
    assert result["inverse_reconstruction_matches_template"] is True
    assert result["bound_wrapper_sha256"] == digest(payload)
    assert b"KERNEL_VERSION = 1" in payload
    assert b"LAUNCH_BINDING_READY = True" in payload
    assert checkout.encode() in payload


def test_s3_binder_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "bound.py"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        bind(
            TEMPLATE,
            output,
            tmp_path / "binding.json",
            repository_root=ROOT,
            checkout_commit="0" * 40,
            kernel_version=1,
        )
