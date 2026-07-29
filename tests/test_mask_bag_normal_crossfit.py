from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
SOURCE = PROJECT / "models" / "mask_bag_normal_crossfit.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _load_module():
    import sys

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location("mask_bag_normal_crossfit", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_crossfit_source_is_group_excluded_and_gt_free() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    assert "audit_crossfit_training_exclusion(" in source
    assert "set(training_groups) & set(heldout_groups)" in source
    assert "validation_segmentation_quality_used" in source
    assert "initial_adapter_state=initial_adapter_state" in source
    for forbidden in (
        "segmentation_dataset",
        "mask_tensor",
        "candidate_quality",
        "size_group",
        "self_guided_instance_loss",
    ):
        assert forbidden not in source
    assert re.search(r"\bdice\b", source) is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_two_fold_oof_covers_every_group_once_without_base_drift() -> None:
    import copy
    import numpy as np
    import torch

    module = _load_module()
    from models.mask_bag_normal_residual_training import (
        NormalResidualTrainingConfig,
    )
    from models.mask_bag_residual_objective import ResidualObjectiveConfig
    from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL

    records = []
    for index, label in enumerate((0, 0, 0, 0, 1, 1, 1, 1)):
        descriptors = np.asarray(
            [
                [1.0 + 0.1 * index, 0.2, 0.1, 0.0],
                [0.1, 1.0 + 0.1 * label, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        records.append(
            {
                "image_id": f"image_{index}",
                "group_id": f"group_{index}",
                "label": label,
                "descriptors": descriptors,
                "flipped_descriptors": descriptors.copy(),
                "family_ids": np.asarray([0, 1], dtype=np.int32),
            }
        )
    fold_ids = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    config = MaskBagMILConfig(
        token_dim=1, token_layers=1, hidden_dim=8, metadata_dim=1
    )
    base = RadDinoMaskBagMIL(config)
    before = copy.deepcopy(base.state_dict())
    objective = ResidualObjectiveConfig()
    training = NormalResidualTrainingConfig(
        epochs=1, batch_size=2, adapter_hidden_dim=8
    )
    artifacts = [
        module.fit_normal_oof_fold(
            records,
            fold_ids,
            heldout_fold=fold,
            prototype_count=1,
            frozen_base_scorer=base,
            descriptor_dim=4,
            objective_config=objective,
            training_config=training,
            device=torch.device("cpu"),
        )
        for fold in (0, 1)
    ]
    assembled = module.assemble_normal_oof_candidate(
        records, fold_ids, artifacts, prototype_count=1
    )

    assert len(assembled["oof_predictions"]) == len(records)
    assert assembled["crossfit_exclusion"]["group_overlap"] == 0
    assert len(assembled["fold_image_bce"]) == 2
    assert all(torch.equal(before[key], base.state_dict()[key]) for key in before)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_assembler_rejects_missing_fold() -> None:
    import numpy as np

    module = _load_module()
    records = [
        {"image_id": "a", "group_id": "ga", "label": 0},
        {"image_id": "b", "group_id": "gb", "label": 1},
    ]
    with pytest.raises(ValueError, match="each fold"):
        module.assemble_normal_oof_candidate(
            records,
            np.asarray([0, 1]),
            [
                {
                    "heldout_fold": 0,
                    "prototype_count": 1,
                    "training_groups": ["gb"],
                    "heldout_predictions": [],
                }
            ],
            prototype_count=1,
        )
