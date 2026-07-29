from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "models" / "mask_bag_normal_residual_training.py"
PROJECT = ROOT / "project"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _load_module():
    import sys

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location(
        "mask_bag_normal_residual_training", SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_has_no_inferred_positive_or_segmentation_path() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "self_guided_instance_loss",
        "argmax",
        "candidate_quality",
        "segmentation_dataset",
        "mask_tensor",
        "size_group",
    ):
        assert forbidden not in source
    assert re.search(r"\bdice\b", source) is None
    assert "frozen_base_scorer.requires_grad_(false).eval()" in source
    assert "residual_arm_objective(" in source


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_prototype_bank_weights_original_and_flip_as_one_image() -> None:
    import numpy as np

    module = _load_module()
    records = [
        {
            "image_id": "normal_a",
            "label": 0,
            "descriptors": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
            "flipped_descriptors": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
            "family_ids": np.asarray([0, 0]),
        },
        {
            "image_id": "tumor_b",
            "label": 1,
            "descriptors": np.asarray([[0.0, 1.0]]),
            "flipped_descriptors": np.asarray([[0.0, 1.0]]),
            "family_ids": np.asarray([0]),
        },
    ]
    prototypes, audit = module.fit_normal_prototype_bank(
        records, prototype_count=1, seed=42
    )

    assert prototypes.shape == (1, 2)
    assert audit["normal_images"] == 1
    assert audit["normal_candidate_views"] == 4
    assert audit["original_and_flip_share_image_family_weight"] is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_adapter_training_leaves_base_scorer_bit_identical() -> None:
    import copy
    import numpy as np
    import torch

    module = _load_module()
    from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
    from models.mask_bag_residual_objective import ResidualObjectiveConfig

    config = MaskBagMILConfig(
        token_dim=1, token_layers=1, hidden_dim=8, metadata_dim=1
    )
    base = RadDinoMaskBagMIL(config)
    before = copy.deepcopy(base.state_dict())
    records = []
    for index, label in enumerate((0, 1, 0, 1)):
        descriptors = np.asarray(
            [[1.0 + index, 0.2, 0.1, 0.0], [0.1, 1.0, 0.2, 0.0]],
            dtype=np.float32,
        )
        records.append(
            {
                "image_id": f"image_{index}",
                "label": label,
                "descriptors": descriptors,
                "flipped_descriptors": descriptors.copy(),
                "auxiliary_features": np.ones((2, 4), dtype=np.float32),
                "flipped_auxiliary_features": np.ones((2, 4), dtype=np.float32),
            }
        )
    adapter, history = module.train_normal_residual_adapter(
        records,
        base,
        descriptor_dim=4,
        objective_config=ResidualObjectiveConfig(
            bag_temperature=0.2,
            consistency_weight=0.1,
            residual_drift_weight=1.0e-3,
        ),
        training_config=module.NormalResidualTrainingConfig(
            epochs=2, batch_size=2, adapter_hidden_dim=8
        ),
        device=torch.device("cpu"),
    )

    assert len(history) == 2
    assert all(torch.equal(before[key], base.state_dict()[key]) for key in before)
    assert not any(parameter.requires_grad for parameter in base.parameters())
    assert torch.count_nonzero(adapter.residual[-1].weight).item() > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_scoring_returns_every_candidate_in_original_order() -> None:
    import numpy as np
    import torch

    module = _load_module()
    from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
    from models.mask_bag_descriptor_residual import AuxiliaryDescriptorResidual

    config = MaskBagMILConfig(
        token_dim=1, token_layers=1, hidden_dim=8, metadata_dim=1
    )
    base = RadDinoMaskBagMIL(config)
    adapter = AuxiliaryDescriptorResidual(
        base_descriptor_dim=4, auxiliary_dim=4, hidden_dim=8
    )
    records = [
        {
            "image_id": "a",
            "label": 1,
            "descriptors": np.ones((3, 4), dtype=np.float32),
            "flipped_descriptors": np.ones((3, 4), dtype=np.float32),
            "auxiliary_features": np.ones((3, 4), dtype=np.float32),
            "flipped_auxiliary_features": np.ones((3, 4), dtype=np.float32),
        }
    ]
    scored = module.score_normal_residual_records(
        records,
        base,
        adapter,
        bag_temperature=0.2,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert len(scored) == 1
    assert scored[0]["image_id"] == "a"
    assert scored[0]["candidate_count"] == 3
    assert scored[0]["candidate_logits"].shape == (3,)
