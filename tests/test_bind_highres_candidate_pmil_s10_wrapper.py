from __future__ import annotations

import json
from pathlib import Path

import pytest

import project.bind_highres_candidate_pmil_s10_wrapper as binder


def test_bind_changes_only_three_launch_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    template = root / binder.TEMPLATE_PATH
    output = tmp_path / "bound.py"
    binding_path = tmp_path / "binding.json"
    commit = "a" * 40

    def git_bytes(_root: Path, _commit: str, relative: str) -> bytes:
        return (root / relative).read_bytes().replace(b"\r\n", b"\n")

    monkeypatch.setattr(binder, "_git_bytes", git_bytes)
    monkeypatch.setattr(binder.subprocess, "run", lambda *args, **kwargs: None)
    result = binder.bind(
        template,
        output,
        binding_path,
        repository_root=root,
        checkout_commit=commit,
        kernel_version=1,
    )
    assert result["replacement_count"] == 3
    assert result["inverse_reconstruction_matches_template"] is True
    payload = output.read_bytes().replace(b"\r\n", b"\n")
    assert b"KERNEL_VERSION = 1" in payload
    assert b"LAUNCH_BINDING_READY = True" in payload
    assert commit.encode() in payload
    assert json.loads(binding_path.read_text(encoding="utf-8")) == result


def test_bind_refuses_overwrite_and_invalid_version(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    template = root / binder.TEMPLATE_PATH
    output = tmp_path / "bound.py"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        binder.bind(
            template,
            output,
            tmp_path / "binding.json",
            repository_root=root,
            checkout_commit="0" * 40,
            kernel_version=1,
        )
