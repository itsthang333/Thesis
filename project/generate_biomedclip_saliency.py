from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

if __package__:
    from .models.biomedclip_saliency import (
        BIOMEDCLIP_MODEL_ID,
        NORMAL_PROMPTS,
        TUMOR_PROMPTS,
        FrozenBiomedClipSaliency,
        resize_map,
    )
else:
    from models.biomedclip_saliency import (
        BIOMEDCLIP_MODEL_ID,
        NORMAL_PROMPTS,
        TUMOR_PROMPTS,
        FrozenBiomedClipSaliency,
        resize_map,
    )


EXPECTED_OPEN_CLIP_VERSION = "2.32.0"
EXPECTED_TRANSFORMERS_VERSION = "4.35.2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen BiomedCLIP saliency maps from images and binary "
            "image-level labels. This command never reads segmentation masks."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(
    split_manifest: Path,
    *,
    expected_sha256: str,
    split: str,
) -> list[dict[str, str]]:
    if sha256_file(split_manifest) != expected_sha256:
        raise ValueError("Split manifest SHA-256 mismatch")
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == split and row.get("eligible") == "1"
        ]
    if not rows:
        raise ValueError(f"No eligible rows for split {split}")
    names = [row["image_id"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image IDs in selected split")
    if any(int(row["tumor"]) not in (0, 1) for row in rows):
        raise ValueError("Selected split contains a non-binary tumor image label")
    return rows


def locate_image(dataset_root: Path, row: dict[str, str]) -> Path:
    candidates = [
        dataset_root / "images" / row["image_id"],
        dataset_root / row["image_id"],
    ]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Image not found uniquely: {row['image_id']}")
    path = matches[0]
    if sha256_file(path) != row["image_sha256"]:
        raise ValueError(f"Source image SHA-256 mismatch: {row['image_id']}")
    return path


def find_weight_file(expected_sha256: str) -> Path:
    if len(expected_sha256) != 64:
        raise ValueError("Expected model-weight SHA-256 must contain 64 hex characters")
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = [
        path
        for path in cache.glob("models--microsoft--BiomedCLIP*/**/*")
        if path.is_file()
        and path.suffix.lower() in {".bin", ".safetensors", ".pt", ".pth"}
        and path.stat().st_size > 1_000_000
    ]
    matches = [path.resolve() for path in candidates if sha256_file(path) == expected_sha256]
    unique = {str(path): path for path in matches}
    if not unique:
        raise FileNotFoundError("Frozen BiomedCLIP weight SHA-256 was not found in cache")
    # Hugging Face snapshot paths can be symlinks to the same blob.
    resolved = {str(path.resolve()): path.resolve() for path in unique.values()}
    if len(resolved) != 1:
        raise ValueError("More than one physical BiomedCLIP weight matched the frozen hash")
    return next(iter(resolved.values()))


def save_map(path: Path, saliency: np.ndarray) -> None:
    if saliency.ndim != 2 or not np.isfinite(saliency).all():
        raise ValueError("Cannot save a non-finite/non-2D saliency map")
    if float(saliency.min()) < 0.0 or float(saliency.max()) > 1.0:
        raise ValueError("Saliency map must be normalized to [0,1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, saliency.astype(np.float16, copy=False), allow_pickle=False)


def main() -> None:
    args = parse_args()
    if args.output_size <= 0:
        raise ValueError("--output-size must be positive")
    if (
        len(args.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in args.source_commit)
    ):
        raise ValueError("--source-commit must be a lowercase 40-character Git SHA")
    if args.split == "test":
        raise ValueError("Test saliency generation is locked")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Output directory is non-empty; pass --overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    rows = load_rows(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
    )
    dataset_root = args.dataset_root.resolve()
    if (dataset_root / "Annotations").exists():
        # The directory may exist in BTXRD, but this command never opens it.
        annotation_contract = "directory_present_but_never_enumerated_or_opened"
    else:
        annotation_contract = "directory_absent"

    import open_clip
    import transformers

    if open_clip.__version__ != EXPECTED_OPEN_CLIP_VERSION:
        raise ValueError(f"open_clip version drift: {open_clip.__version__}")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError(f"transformers version drift: {transformers.__version__}")

    device = torch.device(args.device)
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
    weight_path = find_weight_file(args.expected_model_weight_sha256)
    saliency_model = FrozenBiomedClipSaliency(
        model,
        preprocess,
        tokenizer,
        device=device,
        crop_fraction=0.5,
        positions_per_axis=3,
        top_k_tiles=3,
    )

    map_dir = args.output_dir / "maps"
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_path = locate_image(dataset_root, row)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        label = int(row["tumor"])
        if label == 0:
            saliency = np.zeros((args.output_size, args.output_size), dtype=np.float32)
            full_score = None
            selected_tiles: list[dict[str, Any]] = []
            all_tile_scores: list[float] = []
            generation = "known_normal_image_label_empty"
        else:
            generated = saliency_model(image)
            saliency = resize_map(
                generated.saliency,
                args.output_size,
                args.output_size,
            )
            saliency = np.clip(saliency, 0.0, 1.0)
            full_score = float(generated.full_contrast_score)
            selected_tiles = [
                {
                    "box_xyxy": list(tile.box),
                    "contrast_score": float(tile.contrast_score),
                }
                for tile in generated.selected_tiles
            ]
            all_tile_scores = [float(value) for value in generated.all_tile_scores]
            generation = "frozen_biomedclip_full_plus_top3_tiles"
        map_path = map_dir / f"{Path(row['image_id']).stem}.npy"
        save_map(map_path, saliency)
        dynamic_range = float(saliency.max() - saliency.min())
        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "split": args.split,
                "tumor_image_label": label,
                "source_image_sha256": row["image_sha256"],
                "source_width": int(row["width"]),
                "source_height": int(row["height"]),
                "map_path": str(map_path.relative_to(args.output_dir)).replace("\\", "/"),
                "map_sha256": sha256_file(map_path),
                "map_height": args.output_size,
                "map_width": args.output_size,
                "map_min": float(saliency.min()),
                "map_max": float(saliency.max()),
                "map_mean": float(saliency.mean()),
                "map_dynamic_range": dynamic_range,
                "full_contrast_score": full_score,
                "selected_tiles": json.dumps(selected_tiles, separators=(",", ":")),
                "all_tile_scores": json.dumps(all_tile_scores, separators=(",", ":")),
                "generation": generation,
            }
        )
        print(f"[{index + 1}/{len(rows)}] {row['image_id']} label={label}")

    manifest_path = args.output_dir / "saliency_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    tumor_rows = [row for row in manifest_rows if row["tumor_image_label"] == 1]
    normal_rows = [row for row in manifest_rows if row["tumor_image_label"] == 0]
    if any(float(row["map_dynamic_range"]) != 0.0 for row in normal_rows):
        raise RuntimeError("Known-normal maps are not exactly empty")
    if any(not math.isfinite(float(row["map_dynamic_range"])) for row in tumor_rows):
        raise RuntimeError("Tumor saliency manifest contains non-finite values")

    metadata = {
        "stage": "prediction-first BiomedCLIP saliency generation",
        "supervision": "images and binary image-level labels only",
        "source_commit": args.source_commit,
        "source_files": {
            "generate_biomedclip_saliency.py": sha256_file(Path(__file__).resolve()),
            "models/biomedclip_saliency.py": sha256_file(
                Path(__file__).resolve().parent / "models" / "biomedclip_saliency.py"
            ),
        },
        "split": args.split,
        "population": {
            "images": len(rows),
            "tumor": len(tumor_rows),
            "normal": len(normal_rows),
        },
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "model": {
            "id": BIOMEDCLIP_MODEL_ID,
            "weight_path_name": weight_path.name,
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
        },
        "prompts": {
            "tumor": list(TUMOR_PROMPTS),
            "normal": list(NORMAL_PROMPTS),
            "sha256": sha256_json(
                {"tumor": list(TUMOR_PROMPTS), "normal": list(NORMAL_PROMPTS)}
            ),
        },
        "view_contract": {
            "saliency_reduction": "channelwise mean absolute gradient-times-activation",
            "target_layer": "model.visual.trunk.blocks[11].norm1",
            "full_view": "black pad to square",
            "crop_fraction_of_short_side": 0.5,
            "positions_per_axis": 3,
            "top_k_tiles_by_contrast_score": 3,
            "normalization_percentiles": [1.0, 99.0],
            "fusion": "pixelwise maximum of full view and selected tiles",
            "output_size": args.output_size,
        },
        "manifest_sha256": sha256_file(manifest_path),
        "annotation_contract": annotation_contract,
        "validation_gt_read": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "open_clip": open_clip.__version__,
            "transformers": transformers.__version__,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
