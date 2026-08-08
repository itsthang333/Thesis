from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pytest

from run_g4_e5_exact_kaggle import (  # noqa: E402
    PROTOCOL_SHA,
    sha256,
    supply_command,
    unique_hash,
)


def test_protocol_file_is_bound_to_runner() -> None:
    protocol = PROJECT.parent / "artifacts" / "final_pipeline" / "g4" / "e5_exact_protocol.json"
    assert sha256(protocol) == PROTOCOL_SHA


def test_single_mask_supply_is_explicitly_validation_only(tmp_path: Path) -> None:
    classifier = tmp_path / "classifier.pt"
    classifier.write_bytes(b"frozen-classifier")
    command = supply_command(
        project=Path("/source/project"),
        data=Path("/data/BTXRD"),
        split=Path("/split.csv"),
        classifier=classifier,
        sam=Path("/sam.pth"),
        source_commit="commit",
        output=Path("/output"),
        mode="addition",
    )
    assert "--sam-single-mask" in command
    assert command[command.index("--splits") + 1] == "val"
    assert "test" not in command


def test_unique_hash_accepts_duplicate_content_addressed_copies(tmp_path: Path) -> None:
    left = tmp_path / "a" / "split_manifest.csv"
    right = tmp_path / "b" / "split_manifest.csv"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"same immutable split")
    right.write_bytes(left.read_bytes())
    expected = sha256(left)

    selected = unique_hash(tmp_path, expected, names=("split_manifest.csv",))

    assert selected == left
    assert sha256(selected) == expected


def test_unique_hash_still_fails_when_exact_copy_is_absent(tmp_path: Path) -> None:
    candidate = tmp_path / "split_manifest.csv"
    candidate.write_bytes(b"wrong split")

    with pytest.raises(RuntimeError, match="found no exact copy"):
        unique_hash(tmp_path, "0" * 64, names=("split_manifest.csv",))
