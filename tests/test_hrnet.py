import torch

from btxrd_wsss.models.hrnet_mil import (
    HRNetDenseMIL,
    hrnet_mil_loss,
    hrnet_tile_bag_loss,
)


def test_hrnet_uses_native_multiresolution_stage_features() -> None:
    model = HRNetDenseMIL(pretrained=False, gradient_checkpointing=False, dense_channels=64)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 96))
    assert output.features.shape == (1, 720, 16, 24)
    assert output.dense_logits.shape == (1, 10, 16, 24)
    assert output.class_logits.shape == (1, 10)


def test_hrnet_weak_label_loss_is_finite() -> None:
    model = HRNetDenseMIL(pretrained=False, gradient_checkpointing=False, dense_channels=64)
    output = model(torch.randn(1, 3, 64, 64))
    target = torch.tensor([2])
    multi_hot = torch.zeros(1, 10)
    multi_hot[0, 2] = 1
    loss, parts = hrnet_mil_loss(output, target, multi_hot_targets=multi_hot)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "loss",
        "classification",
        "binary",
        "normal_suppression",
        "map_consistency",
    }


def test_tile_supervision_is_bag_level() -> None:
    model = HRNetDenseMIL(pretrained=False, gradient_checkpointing=False, dense_channels=64)
    output = model(torch.randn(2, 3, 64, 64))
    target = torch.tensor([2])
    multi_hot = torch.zeros(1, 10)
    multi_hot[0, 2] = 1
    references = [torch.zeros_like(output.tumor_map[:1]) for _ in range(2)]
    loss, parts = hrnet_tile_bag_loss(output, target, multi_hot, references)
    assert torch.isfinite(loss)
    assert torch.isfinite(parts["classification"])
