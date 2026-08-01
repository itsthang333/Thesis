from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "download_kaggle_output_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("kaggle_output_inventory", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_target_rejects_absolute_and_escape_paths(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Absolute"):
        module.safe_target(tmp_path, str((tmp_path / "outside").resolve()))
    with pytest.raises(ValueError, match="escapes"):
        module.safe_target(tmp_path, "../outside.bin")
    assert module.safe_target(tmp_path, "compact/maps/one.npy") == (
        tmp_path / "compact/maps/one.npy"
    ).resolve()


def test_nested_targets_can_be_precreated_before_parallel_download(tmp_path: Path) -> None:
    module = _load_module()
    items = [
        module.OutputItem(relative=f"fold_{fold}/scores/{row}.npz", url="unused")
        for fold in range(5)
        for row in range(4)
    ]
    targets = [module.safe_target(tmp_path, item.relative) for item in items]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    assert all(target.parent.is_dir() for target in targets)
    assert len({target.parent for target in targets}) == 5


def test_download_is_atomic_and_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 1024 * 1024
            return iter((b"first", b"second"))

    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(module.requests, "get", fake_get)
    item = module.OutputItem("pair/maps/one.npy", "https://example.invalid/one")
    assert module.download_item(item, tmp_path, max_attempts=1) == "downloaded"
    target = tmp_path / item.relative
    assert target.read_bytes() == b"firstsecond"
    assert not target.with_name(target.name + ".part").exists()
    assert module.download_item(item, tmp_path, max_attempts=1) == "existing"
    assert len(calls) == 1


def test_downloader_does_not_query_kernel_status_or_create_monitor() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    assert "kernels_status" not in source
    assert "kernel_status" not in source
    assert "monitor" not in source
    assert '"status_queried": false' in source
