from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_labelme_mask(path: str | Path, *, height: int, width: int) -> np.ndarray:
    """Evaluation-only loader. Never import this module in training stages."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in payload.get("shapes", []):
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=np.uint8).astype(bool)
