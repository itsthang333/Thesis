from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


BIOMEDCLIP_MODEL_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
TUMOR_PROMPTS = (
    "A bone radiograph showing a bone tumor.",
    "An x-ray image with a bone neoplasm.",
    "An x-ray image showing an abnormal bone lesion.",
)
NORMAL_PROMPTS = (
    "A normal bone radiograph without a tumor.",
    "An x-ray image of healthy bone.",
    "An x-ray image without a bone lesion.",
)


@dataclass(frozen=True)
class TileSaliency:
    box: tuple[int, int, int, int]
    contrast_score: float
    saliency: np.ndarray


@dataclass(frozen=True)
class BiomedClipSaliencyResult:
    saliency: np.ndarray
    full_contrast_score: float
    selected_tiles: tuple[TileSaliency, ...]
    all_tile_scores: tuple[float, ...]


def _axis_positions(extent: int, crop_size: int, positions: int) -> list[int]:
    if extent <= 0 or crop_size <= 0 or crop_size > extent:
        raise ValueError("Invalid extent/crop_size")
    if positions <= 0:
        raise ValueError("positions must be positive")
    if extent == crop_size or positions == 1:
        return [0]
    values = np.linspace(0, extent - crop_size, num=positions)
    return sorted({int(round(value)) for value in values})


def square_crop_boxes(
    width: int,
    height: int,
    *,
    crop_fraction: float = 0.5,
    positions_per_axis: int = 3,
) -> list[tuple[int, int, int, int]]:
    """Return deterministic square crop boxes as half-open xyxy coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in (0,1]")
    side = max(1, min(width, height, int(round(min(width, height) * crop_fraction))))
    xs = _axis_positions(width, side, positions_per_axis)
    ys = _axis_positions(height, side, positions_per_axis)
    return [(x, y, x + side, y + side) for y in ys for x in xs]


def pad_to_square(
    image: Image.Image,
    *,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad symmetrically and return the original-content box in the square."""
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)
    return padded, (left, top, left + width, top + height)


def resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Saliency map must be two-dimensional")
    if height <= 0 or width <= 0:
        raise ValueError("Output dimensions must be positive")
    tensor = torch.from_numpy(values.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy().astype(np.float32, copy=False)


def project_padded_square_map(
    saliency: np.ndarray,
    *,
    padded_side: int,
    content_box: tuple[int, int, int, int],
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Map a saliency grid from a padded square view onto the original image."""
    square = resize_map(saliency, padded_side, padded_side)
    x0, y0, x1, y1 = content_box
    if not (0 <= x0 < x1 <= padded_side and 0 <= y0 < y1 <= padded_side):
        raise ValueError("Content box falls outside padded square")
    content = square[y0:y1, x0:x1]
    return resize_map(content, output_height, output_width)


def robust_normalize(
    values: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Saliency must be a finite two-dimensional array")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Invalid normalization percentiles")
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if float(high) - float(low) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values.astype(np.float32) - float(low)) / (float(high) - float(low))
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def aggregate_full_and_tiles(
    full_saliency: np.ndarray,
    tiles: Sequence[TileSaliency],
    *,
    output_height: int,
    output_width: int,
    top_k_tiles: int = 3,
) -> tuple[np.ndarray, tuple[TileSaliency, ...]]:
    """Fuse full-view evidence with the top-scoring image-only tile evidence."""
    if full_saliency.shape != (output_height, output_width):
        raise ValueError("Full saliency shape differs from output grid")
    if top_k_tiles <= 0:
        raise ValueError("top_k_tiles must be positive")
    indexed = list(enumerate(tiles))
    for index, tile in indexed:
        if not np.isfinite(tile.contrast_score):
            raise ValueError(f"Tile score {index} is non-finite")
        x0, y0, x1, y1 = tile.box
        if not (0 <= x0 < x1 <= output_width and 0 <= y0 < y1 <= output_height):
            raise ValueError(f"Tile box {index} falls outside output grid")
        if tile.saliency.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"Tile saliency {index} does not match its box")
    ranked = sorted(indexed, key=lambda item: (-item[1].contrast_score, item[0]))
    selected = tuple(tile for _, tile in ranked[: min(top_k_tiles, len(ranked))])
    local_canvas = np.zeros((output_height, output_width), dtype=np.float32)
    for tile in selected:
        x0, y0, x1, y1 = tile.box
        normalized = robust_normalize(tile.saliency)
        local_canvas[y0:y1, x0:x1] = np.maximum(
            local_canvas[y0:y1, x0:x1], normalized
        )
    fused = np.maximum(robust_normalize(full_saliency), local_canvas)
    return fused.astype(np.float32, copy=False), selected


class FrozenBiomedClipSaliency:
    """Frozen BiomedCLIP gradient×activation full-plus-tiled saliency."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
        tokenizer: Callable[[Sequence[str]], torch.Tensor],
        *,
        device: torch.device,
        crop_fraction: float = 0.5,
        positions_per_axis: int = 3,
        top_k_tiles: int = 3,
    ) -> None:
        self.model = model.eval().to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.crop_fraction = crop_fraction
        self.positions_per_axis = positions_per_axis
        self.top_k_tiles = top_k_tiles
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            tokens = tokenizer([*TUMOR_PROMPTS, *NORMAL_PROMPTS]).to(device)
            text = self.model.encode_text(tokens, normalize=True)
            tumor = text[: len(TUMOR_PROMPTS)].mean(dim=0)
            normal = text[len(TUMOR_PROMPTS) :].mean(dim=0)
            tumor = tumor / tumor.norm()
            normal = normal / normal.norm()
            self.contrast = (tumor - normal).detach()
        try:
            self.target_layer = self.model.visual.trunk.blocks[11].norm2
        except (AttributeError, IndexError) as error:
            raise ValueError("Unexpected BiomedCLIP visual-transformer structure") from error

    def _view_saliency(self, image: Image.Image) -> tuple[np.ndarray, float]:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                raise ValueError("BiomedCLIP target layer did not return a tensor")
            captured["activation"] = output
            output.retain_grad()

        handle = self.target_layer.register_forward_hook(hook)
        try:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            feature = self.model.encode_image(tensor, normalize=True)
            score = (feature @ self.contrast).sum()
            score.backward()
            activation = captured.get("activation")
            if activation is None or activation.grad is None or activation.ndim != 3:
                raise ValueError("BiomedCLIP target activation/gradient is unavailable")
            gradient = activation.grad
            token_count = int(activation.shape[1])
            side = int(round((token_count - 1) ** 0.5))
            if side * side == token_count - 1:
                activation = activation[:, 1:, :]
                gradient = gradient[:, 1:, :]
            else:
                side = int(round(token_count**0.5))
                if side * side != token_count:
                    raise ValueError(f"Unexpected BiomedCLIP token count: {token_count}")
            saliency = (
                (activation * gradient)
                .sum(dim=-1)
                .abs()
                .reshape(side, side)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            if not np.isfinite(saliency).all():
                raise ValueError("BiomedCLIP saliency is non-finite")
            return saliency.astype(np.float32, copy=False), float(score.detach().cpu())
        finally:
            handle.remove()

    def __call__(self, image: Image.Image) -> BiomedClipSaliencyResult:
        image = image.convert("RGB")
        width, height = image.size
        padded, content_box = pad_to_square(image, fill=(0, 0, 0))
        full_grid, full_score = self._view_saliency(padded)
        full_map = project_padded_square_map(
            full_grid,
            padded_side=padded.width,
            content_box=content_box,
            output_height=height,
            output_width=width,
        )

        tile_results = []
        for box in square_crop_boxes(
            width,
            height,
            crop_fraction=self.crop_fraction,
            positions_per_axis=self.positions_per_axis,
        ):
            crop = image.crop(box)
            grid, score = self._view_saliency(crop)
            x0, y0, x1, y1 = box
            tile_results.append(
                TileSaliency(
                    box=box,
                    contrast_score=score,
                    saliency=resize_map(grid, y1 - y0, x1 - x0),
                )
            )
        fused, selected = aggregate_full_and_tiles(
            full_map,
            tile_results,
            output_height=height,
            output_width=width,
            top_k_tiles=self.top_k_tiles,
        )
        return BiomedClipSaliencyResult(
            saliency=fused,
            full_contrast_score=full_score,
            selected_tiles=selected,
            all_tile_scores=tuple(tile.contrast_score for tile in tile_results),
        )
