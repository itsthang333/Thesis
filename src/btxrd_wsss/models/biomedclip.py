from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

TUMOR_PROMPTS = (
    "a bone radiograph showing a bone tumor",
    "an x-ray with a bone neoplasm",
    "an abnormal bone lesion on x-ray",
)
NORMAL_PROMPTS = (
    "a normal bone radiograph without a tumor",
    "a healthy bone x-ray",
    "an x-ray without a bone lesion",
)


@dataclass(frozen=True)
class BiomedCLIPResult:
    saliency: np.ndarray
    contrast_score: float
    tile_scores: tuple[float, ...]


def robust_normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)


def square_boxes(
    width: int, height: int, fraction: float, positions: int
) -> list[tuple[int, int, int, int]]:
    side = max(1, round(min(width, height) * fraction))
    xs = np.linspace(0, width - side, positions).round().astype(int)
    ys = np.linspace(0, height - side, positions).round().astype(int)
    return [(int(x), int(y), int(x + side), int(y + side)) for y in ys for x in xs]


class FrozenBiomedCLIP:
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable,
        tokenizer: Callable[[Sequence[str]], torch.Tensor],
        device: str = "cuda",
    ) -> None:
        self.model, self.preprocess, self.tokenizer = model.eval().to(device), preprocess, tokenizer
        self.device = torch.device(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            text = self.model.encode_text(
                tokenizer([*TUMOR_PROMPTS, *NORMAL_PROMPTS]).to(device), normalize=True
            )
            tumor = F.normalize(text[: len(TUMOR_PROMPTS)].mean(0), dim=0)
            normal = F.normalize(text[len(TUMOR_PROMPTS) :].mean(0), dim=0)
            self.contrast = tumor - normal
        self.target = self.model.visual.trunk.blocks[-1].norm1

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cuda") -> FrozenBiomedCLIP:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("BiomedCLIP requires open_clip_torch") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        return cls(model, preprocess, open_clip.get_tokenizer(model_id), device)

    def _views(self, images: list[Image.Image]) -> tuple[np.ndarray, tuple[float, ...]]:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["value"] = output
            output.retain_grad()

        handle = self.target.register_forward_hook(hook)
        try:
            tensor = torch.stack([self.preprocess(image) for image in images])
            tensor = tensor.to(self.device).requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            scores = self.model.encode_image(tensor, normalize=True) @ self.contrast
            scores.sum().backward()
            activation = captured["value"]
            gradient = activation.grad
            tokens = activation.shape[1]
            side = round((tokens - 1) ** 0.5)
            if side * side == tokens - 1:
                activation, gradient = activation[:, 1:], gradient[:, 1:]
            else:
                side = round(tokens**0.5)
                if side * side != tokens:
                    raise RuntimeError(f"Unexpected BiomedCLIP token count: {tokens}")
            saliency = (activation * gradient).abs().mean(-1).reshape(len(images), side, side)
            return (
                saliency.detach().float().cpu().numpy(),
                tuple(float(value) for value in scores.detach().float().cpu()),
            )
        finally:
            handle.remove()

    @staticmethod
    def _resize(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        tensor = torch.from_numpy(values.astype(np.float32))[None, None]
        return F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()

    def localize(
        self, image: Image.Image, *, crop_fraction: float, positions_per_axis: int, top_k_tiles: int
    ) -> BiomedCLIPResult:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        padded = ImageOps.expand(image, (left, top, side - width - left, side - height - top))
        boxes = square_boxes(width, height, crop_fraction, positions_per_axis)
        grids, scores = self._views([padded, *[image.crop(box) for box in boxes]])
        full_grid, full_score = grids[0], scores[0]
        full_square = self._resize(full_grid, (side, side))
        full = self._resize(full_square[top : top + height, left : left + width], (height, width))
        tiles: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
        for box, grid, score in zip(boxes, grids[1:], scores[1:], strict=True):
            x0, y0, x1, y1 = box
            tiles.append((score, box, self._resize(grid, (y1 - y0, x1 - x0))))
        canvas = np.zeros((height, width), np.float32)
        for _, (x0, y0, x1, y1), values in sorted(tiles, reverse=True, key=lambda item: item[0])[
            :top_k_tiles
        ]:
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], robust_normalize(values))
        return BiomedCLIPResult(
            np.maximum(robust_normalize(full), canvas),
            full_score,
            tuple(value[0] for value in tiles),
        )
