from __future__ import annotations

import json
from pathlib import Path

import pytest

from project.demo_final_pipeline import DemoConfig, _fresh
from project.generate_biomedclip_saliency import load_biomedclip


def test_demo_config_json_round_trip(tmp_path: Path) -> None:
    config = DemoConfig(
        repository_root=tmp_path / "repo",
        checkpoint_root=tmp_path / "weights",
        dataset_root=tmp_path / "data",
        split_manifest=tmp_path / "split.csv",
        split="val",
        image_id="IMG000001.jpeg",
        work_dir=tmp_path / "work",
    )
    path = config.write_json(tmp_path / "config.json")
    assert DemoConfig.from_json(path) == config


def test_fresh_refuses_work_root_or_escape(tmp_path: Path) -> None:
    work = tmp_path / "demo"
    work.mkdir()
    with pytest.raises(ValueError, match="child of work_dir"):
        _fresh(work, work_root=work)
    with pytest.raises(ValueError, match="child of work_dir"):
        _fresh(tmp_path / "outside", work_root=work)
    stage = work / "stage"
    _fresh(stage, work_root=work)
    assert stage.is_dir()


def test_local_biomedclip_loader_rewrites_text_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "biomedclip"
    model_dir.mkdir()
    config = {
        "model_cfg": {
            "text_cfg": {
                "hf_model_name": "remote/model",
                "hf_tokenizer_name": "remote/model",
            }
        },
        "preprocess_cfg": {},
    }
    (model_dir / "open_clip_config.json").write_text(json.dumps(config))
    for name in (
        "open_clip_pytorch_model.bin", "config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    ):
        (model_dir / name).write_bytes(b"x")

    class FakeOpenClip:
        def __init__(self) -> None:
            self.config_path: Path | None = None
            self.pretrained: str | None = None
            self.pretrained_hf: bool | None = None

        def add_model_config(self, path: Path) -> None:
            self.config_path = Path(path)

        def create_model_from_pretrained(self, name: str, **kwargs):
            assert name == "btxrd_biomedclip_local"
            assert self.config_path is not None
            local = json.loads(self.config_path.read_text())
            text = local["model_cfg"]["text_cfg"]
            assert text["hf_model_name"] == str(model_dir.resolve())
            assert text["hf_tokenizer_name"] == str(model_dir.resolve())
            self.pretrained = kwargs["pretrained"]
            self.pretrained_hf = kwargs["pretrained_hf"]
            return "model", "preprocess"

        def get_tokenizer(self, name: str):
            assert name == "btxrd_biomedclip_local"
            return "tokenizer"

    fake = FakeOpenClip()
    model, preprocess, tokenizer, weight = load_biomedclip(fake, model_dir=model_dir)
    assert (model, preprocess, tokenizer) == ("model", "preprocess", "tokenizer")
    assert weight == (model_dir / "open_clip_pytorch_model.bin").resolve()
    assert fake.pretrained == str(weight)
    assert fake.pretrained_hf is False
