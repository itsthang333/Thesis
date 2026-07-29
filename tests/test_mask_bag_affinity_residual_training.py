from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SOURCE = PROJECT / "models" / "mask_bag_affinity_residual_training.py"

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from models.mask_bag_affinity_residual_training import (
        AFFINITY_DIM,
        AffinityResidualTrainingConfig,
        attach_cached_affinity_features,
        score_affinity_residual_records,
        train_affinity_residual_adapter,
    )
    from models.mask_bag_residual_objective import ResidualObjectiveConfig
    from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


def test_r2_source_is_gt_free_and_has_no_confirmation_target() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "ground_truth",
        "candidate_quality",
        "size_group",
        "argmax",
        "self_guided_instance_loss",
    ):
        assert forbidden not in lowered
    assert "AFFINITY_DIM = 24" in source
    assert "auxiliary_dim=AFFINITY_DIM" in source
    assert "attach_cached_affinity_features(records)" in source


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_affinity_alignment_fails_closed() -> None:
    record = {
        "image_id": "a",
        "descriptors": np.zeros((2, 8), dtype=np.float32),
        "flipped_descriptors": np.zeros((2, 8), dtype=np.float32),
        "affinity_features": np.zeros((1, AFFINITY_DIM), dtype=np.float32),
        "flipped_affinity_features": np.zeros((2, AFFINITY_DIM), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="does not align"):
        attach_cached_affinity_features([record])


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable locally")
def test_r2_learns_only_residual_and_scores_all_candidates() -> None:
    torch.manual_seed(4)
    descriptor_dim = 8
    base = RadDinoMaskBagMIL(
        MaskBagMILConfig(
            token_dim=2,
            token_layers=1,
            hidden_dim=8,
            metadata_dim=2,
        )
    )
    records = []
    for index, label in enumerate((0, 1, 0, 1)):
        count = 2 + index % 2
        records.append(
            {
                "image_id": f"image_{index}",
                "label": label,
                "descriptors": np.random.default_rng(index).normal(
                    size=(count, descriptor_dim)
                ).astype(np.float32),
                "flipped_descriptors": np.random.default_rng(index + 20).normal(
                    size=(count, descriptor_dim)
                ).astype(np.float32),
                "affinity_features": np.random.default_rng(index + 40).normal(
                    size=(count, AFFINITY_DIM)
                ).astype(np.float32),
                "flipped_affinity_features": np.random.default_rng(
                    index + 60
                ).normal(size=(count, AFFINITY_DIM)).astype(np.float32),
            }
        )
    before = {key: value.detach().clone() for key, value in base.state_dict().items()}
    adapter, history = train_affinity_residual_adapter(
        records,
        base,
        descriptor_dim=descriptor_dim,
        objective_config=ResidualObjectiveConfig(),
        training_config=AffinityResidualTrainingConfig(
            epochs=2,
            batch_size=2,
            adapter_hidden_dim=8,
        ),
        device=torch.device("cpu"),
    )
    assert len(history) == 2
    assert all(torch.equal(before[key], value) for key, value in base.state_dict().items())
    scored = score_affinity_residual_records(
        records,
        base,
        adapter,
        bag_temperature=0.2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert [len(row["candidate_logits"]) for row in scored] == [2, 3, 2, 3]
    assert all(np.isfinite(row["candidate_logits"]).all() for row in scored)
