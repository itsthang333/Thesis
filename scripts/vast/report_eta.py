from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", type=int, required=True)
    args = parser.parse_args()
    total_seconds = 0.0
    for stage in ("source_maps", "sam_gallery", "rad_dino"):
        path = Path(args.output) / "evaluation" / stage / "per_image.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        mean = sum(float(row["seconds"]) for row in rows) / len(rows)
        estimate = mean * args.images
        total_seconds += estimate
        print(f"{stage:>14}: {mean:7.2f}s/image, projected {estimate / 3600:6.2f}h")
    history = Path(args.output) / "checkpoints/hrnet_history.json"
    if history.exists():
        epochs = json.loads(history.read_text(encoding="utf-8"))
        if epochs:
            print(f"{'hrnet/epoch':>14}: {epochs[-1]['seconds'] / 3600:6.2f}h")
    print(f"{'supply total':>14}: {total_seconds / 3600:6.2f}h (training excluded)")


if __name__ == "__main__":
    main()
