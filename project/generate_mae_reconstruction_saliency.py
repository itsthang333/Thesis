from __future__ import annotations

"""Generate hash-locked ViT-MAE reconstruction-error maps without reading GT."""

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from .mae_reconstruction_io import (
        load_split_rows_without_annotations,
        locate_verified_image,
        save_float_map,
        sha256_file,
        verify_model_snapshot,
    )
    from .models.mae_reconstruction import (
        accumulate_masked_squared_error,
        make_noise_bank,
        noise_bank_sha256,
        pad_to_square,
        project_square_map,
        radiograph_foreground_mask,
        robust_foreground_normalize,
        validate_complete_mask_coverage,
    )
else:
    from mae_reconstruction_io import (
        load_split_rows_without_annotations,
        locate_verified_image,
        save_float_map,
        sha256_file,
        verify_model_snapshot,
    )
    from models.mae_reconstruction import (
        accumulate_masked_squared_error,
        make_noise_bank,
        noise_bank_sha256,
        pad_to_square,
        project_square_map,
        radiograph_foreground_mask,
        robust_foreground_normalize,
        validate_complete_mask_coverage,
    )


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--model-role", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--num-masks", type=int, default=10)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--mask-batch-size", type=int, default=2)
    return parser.parse_args()


def _tensor_from_square(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def main() -> None:
    args = parse_args()
    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers must be {EXPECTED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This heavy generation stage requires a Kaggle GPU")
    snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    device = torch.device("cuda")
    model = ViTMAEForPreTraining.from_pretrained(
        args.model_dir, local_files_only=True
    ).eval().to(device)
    patch_size = int(model.config.patch_size)
    if args.input_size % patch_size:
        raise ValueError("input-size must be divisible by patch size")
    grid = args.input_size // patch_size
    noise_bank = make_noise_bank(
        num_masks=args.num_masks,
        num_patches=grid * grid,
        seed=args.mask_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, object]] = []
    observed_masks: list[torch.Tensor] = []

    for row_index, row in enumerate(rows):
        image_path = locate_verified_image(args.dataset_root, row)
        original = Image.open(image_path).convert("RGB")
        square, projection = pad_to_square(original, fill=0)
        pixel_values = _tensor_from_square(square, args.input_size)
        error_sum = torch.zeros((args.input_size, args.input_size), device=device)
        coverage_sum = torch.zeros_like(error_sum)
        for start in range(0, args.num_masks, args.mask_batch_size):
            noise = noise_bank[start : start + args.mask_batch_size].to(device)
            batch = pixel_values.unsqueeze(0).repeat(noise.shape[0], 1, 1, 1).to(device)
            with torch.inference_mode():
                output = model(
                    pixel_values=batch,
                    noise=noise,
                    interpolate_pos_encoding=True,
                )
            errors, coverage = accumulate_masked_squared_error(
                prediction_patches=output.logits,
                pixel_values=batch,
                patch_mask=output.mask,
                patch_size=patch_size,
            )
            error_sum += errors.sum(0)
            coverage_sum += coverage.sum(0)
            if row_index == 0:
                observed_masks.extend(mask.detach().cpu() for mask in output.mask)
        if torch.any(coverage_sum <= 0):
            raise RuntimeError("Frozen mask bank leaves pixels without reconstruction evidence")
        raw_square = (error_sum / coverage_sum).detach().cpu().numpy()
        raw_map = project_square_map(
            raw_square,
            projection,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        foreground = radiograph_foreground_mask(
            original,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        normalized = robust_foreground_normalize(raw_map, foreground)
        relative = Path("maps") / f"{Path(row['image_id']).stem}.npy"
        save_float_map(args.output_dir / relative, normalized)
        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "map_path": relative.as_posix(),
                "map_sha256": sha256_file(args.output_dir / relative),
                "raw_mean": float(raw_map.mean()),
                "raw_p99": float(np.percentile(raw_map, 99)),
                "normalized_mean": float(normalized.mean()),
                "normalized_max": float(normalized.max()),
            }
        )

    coverage = validate_complete_mask_coverage(
        observed_masks, num_patches=grid * grid
    )
    manifest_path = args.output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "stage": "validation prediction generation before GT access",
        "model_role": args.model_role,
        "source_commit": args.source_commit,
        "source_files": {
            "generator": sha256_file(Path(__file__).resolve()),
            "mae_reconstruction_io": sha256_file(Path(__file__).with_name("mae_reconstruction_io.py")),
            "mae_reconstruction": sha256_file(Path(__file__).parent / "models" / "mae_reconstruction.py"),
        },
        "split": "val",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "cohort": len(rows),
        "model_snapshot": snapshot,
        "inference": {
            "input_size": args.input_size,
            "output_size": args.output_size,
            "patch_size": patch_size,
            "mask_ratio": float(model.config.mask_ratio),
            "num_masks": args.num_masks,
            "mask_seed": args.mask_seed,
            "noise_bank_sha256": noise_bank_sha256(noise_bank),
            "minimum_patch_coverage": int(coverage.min()),
            "normalization": "foreground pixelwise p5-p99",
        },
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "validation_gt_read": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
