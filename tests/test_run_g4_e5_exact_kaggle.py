from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run_g4_e5_exact_kaggle import PROTOCOL_SHA, sha256, supply_command  # noqa: E402


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
