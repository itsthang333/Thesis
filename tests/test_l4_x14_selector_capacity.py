from __future__ import annotations

import numpy as np
import torch
from torch import nn

from project.models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from project.run_l4_x14_selector_capacity import (
    ARCHITECTURES,
    SelectorCapacityMIL,
    _stable_argmax,
    architecture_parameter_count,
)


def _linear_layers(model: SelectorCapacityMIL) -> int:
    return sum(isinstance(module, nn.Linear) for module in model.scorer)


def test_capacity_arms_have_declared_depth_and_ordered_parameter_counts() -> None:
    config = MaskBagMILConfig(token_dim=4, token_layers=3, hidden_dim=8)
    models = {name: SelectorCapacityMIL(config, name) for name in ARCHITECTURES}
    assert [_linear_layers(models[name]) for name in ARCHITECTURES] == [1, 2, 3]
    counts = [architecture_parameter_count(config, name) for name in ARCHITECTURES]
    assert counts[0] < counts[1] < counts[2]


def test_two_hidden_arm_is_exact_current_g1_scorer() -> None:
    config = MaskBagMILConfig(token_dim=4, token_layers=3, hidden_dim=8)
    torch.manual_seed(123)
    current = SelectorCapacityMIL(config, "two_hidden")
    torch.manual_seed(123)
    deployed = RadDinoMaskBagMIL(config)
    assert list(current.scorer.state_dict()) == list(deployed.scorer.state_dict())
    for key, value in current.scorer.state_dict().items():
        assert torch.equal(value, deployed.scorer.state_dict()[key])

    current.eval()
    deployed.eval()
    descriptors = torch.randn(2, 5, config.descriptor_dim)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    current_logits, current_bag = current.score_descriptors(descriptors, valid)
    deployed_logits, deployed_bag = deployed.score_descriptors(descriptors, valid)
    assert torch.allclose(current_logits, deployed_logits)
    assert torch.allclose(current_bag, deployed_bag)


def test_stable_argmax_prefers_lower_frozen_candidate_index_on_tie() -> None:
    values = np.asarray([0.4, 0.9, 0.9, 0.2])
    indices = np.asarray([8, 11, 3, 5])
    assert _stable_argmax(values, indices) == 2
