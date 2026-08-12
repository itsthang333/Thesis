from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch import nn

from btxrd_wsss.types import CandidateMask

SOURCE_ORDER = ("hrnet_full", "hrnet_tile", "biomedclip")


def context_box(mask: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Candidate mask is empty")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = max(8, (x1 - x0) * scale), max(8, (y1 - y0) * scale)
    return (
        max(0, round(cx - width / 2)),
        max(0, round(cy - height / 2)),
        min(mask.shape[1], round(cx + width / 2)),
        min(mask.shape[0], round(cy + height / 2)),
    )


def geometry_features(candidate: CandidateMask) -> np.ndarray:
    mask = np.asarray(candidate.mask, bool)
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    area = mask.sum()
    perimeter = np.logical_and(mask, ~ndimage.binary_erosion(mask)).sum()
    return np.asarray(
        [
            area / mask.size,
            area / max(1, (x1 - x0) * (y1 - y0)),
            4 * np.pi * area / max(1, perimeter**2),
            xs.mean() / width,
            ys.mean() / height,
            candidate.predicted_iou,
            candidate.stability,
            candidate.roi_scale / 4,
        ],
        np.float32,
    )


@dataclass(frozen=True)
class DescriptorBatch:
    values: np.ndarray
    candidate_ids: tuple[str, ...]


class FrozenRadDINODescriptor:
    def __init__(
        self,
        model_id: str,
        *,
        input_size: int,
        selected_layers: list[int],
        projection_dim: int,
        batch_size: int = 16,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError("RAD-DINO descriptors require transformers") from exc
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device, self.input_size, self.batch_size = torch.device(device), input_size, batch_size
        self.selected_layers, self.projection_dim, self.seed = selected_layers, projection_dim, seed
        self._projections: dict[tuple[int, int], torch.Tensor] = {}

    def _projection(self, layer_index: int, width: int) -> torch.Tensor:
        key = (layer_index, width)
        if key not in self._projections:
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + 1009 * (layer_index + 100)
            )
            matrix = torch.randn(width, self.projection_dim, generator=generator) / np.sqrt(
                self.projection_dim
            )
            self._projections[key] = matrix.to(self.device)
        return self._projections[key]

    @staticmethod
    def _patch_grid(hidden: torch.Tensor) -> torch.Tensor:
        token_count = hidden.shape[1]
        side = int(np.sqrt(token_count))
        while token_count - side * side > 8:
            side -= 1
        prefix = token_count - side * side
        return hidden[:, prefix:].reshape(hidden.shape[0], side, side, hidden.shape[-1])

    def _encode_crops(self, crops: list[Image.Image]) -> tuple[torch.Tensor, ...]:
        collected: list[list[torch.Tensor]] = [[] for _ in self.selected_layers]
        for start in range(0, len(crops), self.batch_size):
            inputs = self.processor(
                images=crops[start : start + self.batch_size], return_tensors="pt"
            )
            pixels = inputs["pixel_values"].to(self.device)
            pixels = F.interpolate(
                pixels, (self.input_size, self.input_size), mode="bicubic", align_corners=False
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
            ):
                output = self.model(pixel_values=pixels, output_hidden_states=True)
            for target, index in zip(collected, self.selected_layers, strict=True):
                target.append(output.hidden_states[index].float())
        return tuple(torch.cat(values) for values in collected)

    def extract(
        self, image: Image.Image, candidates: list[CandidateMask], context_scales: list[float]
    ) -> DescriptorBatch:
        image = image.convert("RGB")
        crops: list[Image.Image] = []
        masks: list[np.ndarray] = []
        for candidate in candidates:
            for scale in context_scales:
                box = context_box(candidate.mask, scale)
                x0, y0, x1, y1 = box
                crops.append(image.crop(box))
                masks.append(candidate.mask[y0:y1, x0:x1])
        layers = self._encode_crops(crops)
        parts: list[torch.Tensor] = []
        for layer_number, hidden in zip(self.selected_layers, layers, strict=True):
            grid = self._patch_grid(hidden)
            side = grid.shape[1]
            local: list[torch.Tensor] = []
            for index, mask in enumerate(masks):
                weights = torch.from_numpy(mask.astype(np.float32))[None, None]
                weights = F.interpolate(weights, (side, side), mode="nearest")[0, 0].to(self.device)
                ring = F.max_pool2d(weights[None, None], 3, 1, 1)[0, 0] * (1 - weights)
                tokens = grid[index]
                inside = (tokens * weights[..., None]).sum((0, 1)) / weights.sum().clamp_min(1)
                outside = (tokens * ring[..., None]).sum((0, 1)) / ring.sum().clamp_min(1)
                projection = self._projection(layer_number, tokens.shape[-1])
                local.append(
                    torch.cat(
                        (inside @ projection, outside @ projection, (inside - outside) @ projection)
                    )
                )
            parts.append(torch.stack(local))
        # Reorder from [layer, candidate*context] to [candidate, context*layer].
        encoded = (
            torch.cat(parts, dim=1).reshape(len(candidates), len(context_scales), -1).flatten(1)
        )
        geometry = torch.from_numpy(np.stack([geometry_features(item) for item in candidates])).to(
            self.device
        )
        source = torch.zeros(len(candidates), len(SOURCE_ORDER), device=self.device)
        for index, candidate in enumerate(candidates):
            source[index, SOURCE_ORDER.index(candidate.proposal_source)] = 1
        values = torch.cat((encoded, geometry, source), dim=1).cpu().numpy().astype(np.float32)
        return DescriptorBatch(values, tuple(item.candidate_id for item in candidates))


class G1Scorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.network(descriptors).squeeze(-1)


def smooth_bag_logit(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return temperature * (torch.logsumexp(logits / temperature, dim=0) - np.log(len(logits)))


def g1_mil_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    negative_instance_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bag = smooth_bag_logit(logits, temperature)
    target = target.float().reshape(())
    bag_loss = F.binary_cross_entropy_with_logits(bag, target)
    negative = (1 - target) * F.softplus(logits).mean()
    total = bag_loss + negative_instance_weight * negative
    return total, {
        "loss": total.detach(),
        "bag": bag_loss.detach(),
        "negative_instance": negative.detach(),
        "bag_logit": bag.detach(),
    }
