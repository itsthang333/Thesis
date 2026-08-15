from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project.run_rich_gallery_candidate_supply import build_generation_command  # noqa: E402


def command(mode: str) -> list[str]:
    kwargs = dict(
        mode=mode,
        source_root=Path("/source"),
        data_root=Path("/data"),
        split_manifest=Path("/split.csv"),
        split="train",
        classifier=Path("/classifier.pt"),
        sam=Path("/sam.pth"),
        output_dir=Path("/out"),
    )
    if mode == "anchor":
        kwargs.update(
            external_root=Path("/saliency/train"),
            external_manifest_sha256="a" * 64,
            external_metadata_sha256="b" * 64,
            external_source_commit="c" * 40,
            external_weight_sha256="d" * 64,
        )
    return build_generation_command(**kwargs)


def test_anchor_command_is_full_cohort_320_and_external_locked() -> None:
    values = command("anchor")
    joined = " ".join(values)
    assert "--image-size 320" in joined
    assert "--external-saliency-role proposal_gallery" in joined
    assert "--candidate-diagnostics-cohort all" in joined
    assert "--force-normal-candidate-gallery" in values
    assert "--cam-target-class ground_truth" in joined
    assert "--support-clip-kernel 5" in joined


def test_addition_command_reproduces_448_geometry_without_external_saliency() -> None:
    values = command("addition")
    joined = " ".join(values)
    assert "--image-size 448" in joined
    assert "--min-component-area 196" in joined
    assert "--min-size 78" in joined
    assert "--prompt-border-margin 3" in joined
    assert "--support-clip-kernel 7" in joined
    assert "external-saliency" not in joined


def test_anchor_rejects_partial_external_lock() -> None:
    with pytest.raises(ValueError, match="complete external-saliency"):
        build_generation_command(
            mode="anchor",
            source_root=Path("/source"),
            data_root=Path("/data"),
            split_manifest=Path("/split.csv"),
            split="val",
            classifier=Path("/classifier.pt"),
            sam=Path("/sam.pth"),
            output_dir=Path("/out"),
            external_root=Path("/saliency/val"),
        )


def test_test_generation_requires_and_propagates_frozen_config() -> None:
    with pytest.raises(ValueError, match="requires --frozen-config"):
        build_generation_command(
            mode="addition",
            source_root=Path("/source"),
            data_root=Path("/data"),
            split_manifest=Path("/split.csv"),
            split="test",
            classifier=Path("/classifier.pt"),
            sam=Path("/sam.pth"),
            output_dir=Path("/out"),
        )
    values = build_generation_command(
        mode="addition",
        source_root=Path("/source"),
        data_root=Path("/data"),
        split_manifest=Path("/split.csv"),
        split="test",
        classifier=Path("/classifier.pt"),
        sam=Path("/sam.pth"),
        output_dir=Path("/out"),
        frozen_config=Path("/lock.json"),
    )
    assert values[values.index("--frozen-config") + 1] == str(Path("/lock.json"))
