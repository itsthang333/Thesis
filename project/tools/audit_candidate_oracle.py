from __future__ import annotations

"""Post-freeze GT audit for a completed candidate pool and selected masks."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.factory import build_segmentation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btxrd-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--selected-mask-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def dice(mask: np.ndarray, target: np.ndarray) -> float:
    mask = mask.astype(bool)
    target = target.astype(bool)
    denom = int(mask.sum()) + int(target.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(mask, target).sum() / denom)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    ) > 0


def main() -> None:
    args = parse_args()
    dataset = build_segmentation_dataset(
        "btxrd", root=args.btxrd_root, split=args.split, image_size=args.image_size, augment=False
    )
    rows: list[dict[str, object]] = []
    missing = 0
    tumor_oracle: list[float] = []
    tumor_selected: list[float] = []
    normal_specific: list[float] = []

    for _image, target_tensor, image_name in dataset:
        stem = Path(str(image_name)).stem
        target = target_tensor[0].numpy() > 0.5
        selected_path = args.selected_mask_root / f"{stem}.png"
        if not selected_path.exists():
            missing += 1
            rows.append({"image_name": image_name, "status": "missing"})
            continue
        selected = load_mask(selected_path, args.image_size)
        is_tumor = bool(target.any())
        if not is_tumor:
            specificity = float(not selected.any())
            normal_specific.append(specificity)
            rows.append({
                "image_name": image_name, "status": "ok", "group": "normal",
                "oracle_dice": "", "selected_dice": dice(selected, target),
                "selection_loss_dice": "", "specificity": specificity,
            })
            continue

        cache_path = args.candidate_cache_dir / f"{stem}.npz"
        if not cache_path.exists():
            missing += 1
            rows.append({"image_name": image_name, "status": "missing_candidate_cache", "group": "tumor"})
            continue
        with np.load(cache_path) as cache:
            candidates = cache["masks"].astype(bool)
        if len(candidates) == 0:
            oracle = 0.0
        else:
            oracle = max(dice(candidate, target) for candidate in candidates)
        selected_score = dice(selected, target)
        selection_loss = oracle - selected_score
        tumor_oracle.append(oracle)
        tumor_selected.append(selected_score)
        rows.append({
            "image_name": image_name, "status": "ok", "group": "tumor",
            "oracle_dice": oracle, "selected_dice": selected_score,
            "selection_loss_dice": selection_loss, "specificity": "",
        })

    summary = {
        "dataset": "btxrd",
        "split": args.split,
        "audit_only_gt_used": True,
        "tumor_images_evaluated": len(tumor_selected),
        "normal_images_evaluated": len(normal_specific),
        "oracle_dice": float(np.mean(tumor_oracle)) if tumor_oracle else 0.0,
        "selected_dice": float(np.mean(tumor_selected)) if tumor_selected else 0.0,
        "selection_loss_dice": (
            float(np.mean(np.asarray(tumor_oracle) - np.asarray(tumor_selected)))
            if tumor_selected else 0.0
        ),
        "normal_specificity": float(np.mean(normal_specific)) if normal_specific else 0.0,
        "missing_masks_or_caches": missing,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name", "status", "group", "oracle_dice", "selected_dice",
        "selection_loss_dice", "specificity",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
