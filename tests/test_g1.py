import torch

from btxrd_wsss.models.rad_dino_g1 import G1Scorer, g1_mil_loss, smooth_bag_logit


def test_normalized_smooth_bag_logit_does_not_reward_duplicate_candidates() -> None:
    singleton = smooth_bag_logit(torch.tensor([2.0]), 0.2)
    duplicates = smooth_bag_logit(torch.tensor([2.0, 2.0, 2.0]), 0.2)
    torch.testing.assert_close(singleton, duplicates)


def test_g1_negative_bag_penalizes_positive_instances() -> None:
    model = G1Scorer(16, hidden_dim=8, dropout=0)
    logits = model(torch.randn(4, 16))
    loss, parts = g1_mil_loss(
        logits,
        torch.tensor(0.0),
        temperature=0.2,
        negative_instance_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert parts["negative_instance"] > 0
