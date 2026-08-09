from pathlib import Path

import pytest

from project.pseudo.sam_refine import SAMPredictor
from project.run_rich_gallery_candidate_supply import common_generation_args


def test_backend_contract_is_explicit() -> None:
    assert SAMPredictor.BACKENDS == ("sam_v1", "sam2", "sam_med2d", "medsam")


def test_non_v1_backend_requires_official_source_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="sam2 requires an explicit"):
        SAMPredictor(
            checkpoint,
            device="cpu",
            model_type="sam2.1_hiera_large",
            backend="sam2",
        )


def test_alias_loader_preserves_relative_imports(tmp_path: Path) -> None:
    package = tmp_path / "segment_anything"
    package.mkdir()
    (package / "value.py").write_text("VALUE = 17\n", encoding="utf-8")
    (package / "__init__.py").write_text(
        "from .value import VALUE\n", encoding="utf-8"
    )
    loaded = SAMPredictor._load_segment_anything_alias(
        "btxrd_test_segment_anything_alias", package
    )
    assert loaded.VALUE == 17


def test_candidate_command_carries_backend_and_source(tmp_path: Path) -> None:
    command = common_generation_args(
        source_root=tmp_path / "source",
        data_root=tmp_path / "data",
        split_manifest=tmp_path / "split.csv",
        split="val",
        classifier=tmp_path / "classifier.pt",
        sam=tmp_path / "sam.pt",
        sam_model_type="sam2.1_hiera_large",
        sam_backend="sam2",
        sam_source_root=tmp_path / "official_sam2",
        output_dir=tmp_path / "output",
    )
    assert command[command.index("--sam-backend") + 1] == "sam2"
    assert command[command.index("--sam-model-type") + 1] == "sam2.1_hiera_large"
    assert command[command.index("--sam-source-root") + 1] == str(
        tmp_path / "official_sam2"
    )
