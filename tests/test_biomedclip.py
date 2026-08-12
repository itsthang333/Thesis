from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from btxrd_wsss.models.biomedclip import FrozenBiomedCLIP


class DummyBiomedCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = SimpleNamespace(norm1=nn.LayerNorm(4))
        self.visual = SimpleNamespace(trunk=SimpleNamespace(blocks=[block]))
        self.calls = 0

    def encode_text(self, tokens: torch.Tensor, normalize: bool) -> torch.Tensor:
        values = torch.stack(
            (tokens.float(), 1 - tokens.float(), tokens.float(), tokens.float()), 1
        )
        return F.normalize(values, dim=1) if normalize else values

    def encode_image(self, images: torch.Tensor, normalize: bool) -> torch.Tensor:
        self.calls += 1
        base = images.mean((1, 2, 3))
        tokens = torch.stack([base + offset for offset in range(20)], 1).reshape(-1, 5, 4)
        values = self.visual.trunk.blocks[0].norm1(tokens).mean(1)
        return F.normalize(values, dim=1) if normalize else values


def test_biomedclip_batches_full_and_tile_views() -> None:
    model = DummyBiomedCLIP()

    def preprocess(image: Image.Image) -> torch.Tensor:
        image = image.resize((8, 8))
        return torch.from_numpy(np.asarray(image, np.float32)).permute(2, 0, 1) / 255

    def tokenizer(prompts: list[str]) -> torch.Tensor:
        return torch.arange(len(prompts)) % 2

    localizer = FrozenBiomedCLIP(model, preprocess, tokenizer, device="cpu")
    image = Image.fromarray(np.full((20, 30, 3), 127, np.uint8))
    result = localizer.localize(
        image,
        crop_fraction=0.5,
        positions_per_axis=3,
        top_k_tiles=3,
    )
    assert result.saliency.shape == (20, 30)
    assert len(result.tile_scores) == 9
    assert model.calls == 1
