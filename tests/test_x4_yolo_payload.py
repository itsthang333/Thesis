from __future__ import annotations

import csv
from pathlib import Path

import pytest

from project.build_x4_yolo_kaggle_payload import SOURCE_FILES, verify_payload, write_manifest


def test_payload_contains_legacy_script_import_closure() -> None:
    """Guard the exact import path used by the offline Kaggle wrapper."""

    assert "project/config.py" in SOURCE_FILES
    assert "project/datasets/common.py" in SOURCE_FILES
    assert "project/datasets/btxrd.py" in SOURCE_FILES


def test_payload_manifest_exact_match_and_tamper(tmp_path: Path) -> None:
    source = tmp_path / "x4_yolo_source" / "project" / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('ok')\n", encoding="utf-8")
    rows = [{"path": "x4_yolo_source/project/runner.py", "bytes": source.stat().st_size, "sha256": ""}]
    import hashlib

    rows[0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_manifest(tmp_path, rows)
    assert verify_payload(tmp_path)["exact_match"] is True
    source.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        verify_payload(tmp_path)


def test_payload_rejects_unlisted_file(tmp_path: Path) -> None:
    manifest = tmp_path / "payload_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set differs"):
        verify_payload(tmp_path)
