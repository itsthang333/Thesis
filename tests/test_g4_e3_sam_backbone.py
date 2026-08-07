from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from run_g4_e3_sam_backbone import (  # noqa: E402
    CLASSIFIER_320_SHA,
    CLASSIFIER_448_SHA,
    SAM_SHA,
    _supply_command,
    canonical_split,
)


def test_e3_official_checkpoint_hashes_are_distinct_and_complete() -> None:
    assert set(SAM_SHA) == {"vit_b", "vit_l", "vit_h"}
    assert len(set(SAM_SHA.values())) == 3
    assert all(len(value) == 64 for value in SAM_SHA.values())


def test_e3_anchor_changes_only_explicit_sam_architecture_inputs() -> None:
    common = dict(
        project=Path("/source/project"),
        data=Path("/data/BTXRD"),
        split=Path("/split.csv"),
        classifier=Path("/classifier320.pt"),
        classifier_split=Path("/classifier-split.csv"),
        source_commit="a" * 40,
        output=Path("/out"),
        mode="anchor",
        external_root=Path("/external"),
    )
    # The external manifest hash is resolved at execution; give the builder a
    # real tiny file so this test also exercises its fail-closed binding.
    # Platform-neutral temp replacement for the absolute pseudo paths.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "saliency_supply_manifest.json").write_text("{}\n")
        classifier = root / "classifier320.pt"
        classifier.write_bytes(b"matched classifier fixture")
        common["external_root"] = root
        common["classifier"] = classifier
        b = _supply_command(sam=Path("/sam_b.pth"), sam_model_type="vit_b", **common)
        l = _supply_command(sam=Path("/sam_l.pth"), sam_model_type="vit_l", **common)
        b_without_sam = [
            value for index, value in enumerate(b)
            if index not in {
                b.index("--sam-checkpoint") + 1,
                b.index("--expected-sam-sha256") + 1,
                b.index("--sam-model-type") + 1,
            }
        ]
        l_without_sam = [
            value for index, value in enumerate(l)
            if index not in {
                l.index("--sam-checkpoint") + 1,
                l.index("--expected-sam-sha256") + 1,
                l.index("--sam-model-type") + 1,
            }
        ]
    assert b_without_sam == l_without_sam
    assert b[b.index("--sam-model-type") + 1] == "vit_b"
    assert l[l.index("--sam-model-type") + 1] == "vit_l"


def test_e3_classifier_hash_contract_matches_final_pipeline() -> None:
    assert CLASSIFIER_320_SHA.startswith("ca630d")
    assert CLASSIFIER_448_SHA.startswith("b40dc5")


def test_e3_canonical_split_prefers_dedicated_byte_exact_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    historical = tmp_path / "historical" / "split_manifest.csv"
    preferred = (
        tmp_path
        / "btxrd-matched-normal-transplant-inputs-20260802"
        / "source_snapshot"
        / "artifacts"
        / "data_audit"
        / "split_manifest.csv"
    )
    historical.parent.mkdir(parents=True)
    preferred.parent.mkdir(parents=True)
    historical.write_bytes(b"same canonical bytes")
    preferred.write_bytes(b"same canonical bytes")
    module = sys.modules[canonical_split.__module__]
    monkeypatch.setattr(module, "SPLIT_SHA", module.sha256(preferred))
    assert canonical_split(tmp_path) == preferred
