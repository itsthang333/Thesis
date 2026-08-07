from __future__ import annotations

import json
from pathlib import Path

from project.run_g4_e1_downstream import (
    ADDITION_CLASSIFIER_SHA,
    PROTOCOL_SHA,
    addition_supply,
    anchor_command,
    sha256,
)
from project.run_g4_e3_sam_backbone import SAM_SHA, SPLIT_SHA


def test_e1_downstream_protocol_file_is_exactly_bound() -> None:
    protocol = (
        Path(__file__).parents[1]
        / "artifacts"
        / "final_pipeline"
        / "g4"
        / "e1_downstream_protocol.json"
    )
    assert sha256(protocol) == PROTOCOL_SHA
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    assert payload["seeds"] == [42, 43, 44]
    assert payload["test_evaluated"] is False


def test_e1_ten_class_command_uses_exact_collapsed_log_odds(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "saliency_supply_manifest.json").write_text("{}\n", encoding="utf-8")
    common = dict(
        project=Path("/source/project"),
        data=Path("/data/BTXRD"),
        split=Path("/split.csv"),
        checkpoint_sha="a" * 64,
        sam=Path("/sam.pth"),
        external_root=external,
        source_commit="b" * 40,
        output=Path("/output"),
    )
    binary = anchor_command(
        checkpoint=Path("/binary.pt"), arm="binary", **common
    )
    ten_class = anchor_command(
        checkpoint=Path("/ten.pt"), arm="ten_class", **common
    )
    assert binary[binary.index("--target-columns") + 1] == "tumor"
    assert binary[binary.index("--cam-aggregation") + 1] == "class"
    assert ten_class[ten_class.index("--target-columns") + 1] == "tumor_type"
    assert ten_class[ten_class.index("--cam-aggregation") + 1] == "tumor_log_odds"


def test_addition_supply_resolver_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    payload = {
        "mode": "addition",
        "classifier_checkpoint_sha256": ADDITION_CLASSIFIER_SHA,
        "sam_checkpoint_sha256": SAM_SHA["vit_b"],
        "split_sha256": SPLIT_SHA,
        "test_evaluated": False,
        "test_images_read": 0,
        "splits": {"val": {"counts": {"images": 371}}},
    }
    (root / "candidate_supply_manifest.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    resolved, loaded = addition_supply(tmp_path)
    assert resolved == root
    assert loaded == payload


def test_e1_downstream_runner_never_requests_test() -> None:
    source = (
        Path(__file__).parents[1] / "project" / "run_g4_e1_downstream.py"
    ).read_text(encoding="utf-8")
    assert '"--splits", "val"' in source
    assert '"--split", "test"' not in source
    assert '"test_images_read": 0' in source
