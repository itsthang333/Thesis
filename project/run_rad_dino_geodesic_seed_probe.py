from __future__ import annotations

"""Generate prediction-first RAD-DINO geodesic seed-expansion maps.

This runner has no segmentation-dataset or annotation import.  It verifies a
frozen affinity-decoder map package, extracts frozen RAD-DINO patch features,
and refines only the validation maps.  A separate evaluator may open
validation masks after ``prediction_freeze.json`` has been written.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    save_float_map,
    sha256_file,
    verify_model_snapshot,
)
from models.mae_reconstruction import (
    SquareProjection,
    pad_to_square,
    radiograph_foreground_mask,
)
from models.nominal_patch_memory import (
    make_seeded_random_projection,
    project_features,
    projection_sha256,
)
from pseudo.affinity_selector_input import (
    load_affinity_selector_contract,
    load_affinity_selector_map,
)
from pseudo.geodesic_seed_expansion import geodesic_seed_expansion


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
RAD_DINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
RAD_DINO_STD = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--affinity-input-root", type=Path, required=True)
    parser.add_argument("--expected-affinity-manifest-sha256", required=True)
    parser.add_argument("--expected-affinity-package-sha256", required=True)
    parser.add_argument("--expected-affinity-freeze-sha256", required=True)
    parser.add_argument("--expected-affinity-source-commit", required=True)
    parser.add_argument("--expected-affinity-protocol-sha256", required=True)
    parser.add_argument("--expected-affinity-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--graph-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--foreground-fraction", type=float, default=0.01)
    parser.add_argument("--background-fraction", type=float, default=0.50)
    parser.add_argument("--geodesic-ratio", type=float, default=1.0)
    return parser.parse_args()


def _raw_normalized_square(
    image: Image.Image, *, input_size: int
) -> tuple[torch.Tensor, SquareProjection]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    raw = torch.from_numpy(values).permute(2, 0, 1)
    return (raw - RAD_DINO_MEAN) / RAD_DINO_STD, projection


def _extract_patch_tokens(
    encoder: nn.Module,
    pixels: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        hidden = encoder(
            pixel_values=pixels[None].to(device, non_blocking=True)
        ).last_hidden_state
    expected = grid_size * grid_size + 1
    if hidden.ndim != 3 or hidden.shape != (1, expected, 768):
        raise RuntimeError(f"Unexpected RAD-DINO token shape {tuple(hidden.shape)}")
    return (
        hidden[0, 1:]
        .reshape(grid_size, grid_size, 768)
        .float()
        .cpu()
        .numpy()
    )


def project_square_features(
    features: np.ndarray,
    projection: SquareProjection,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Sample square-grid features in the continuous unpadded content frame.

    Unlike prediction-map reprojection, structural features are not first
    rasterized to the native radiograph dimensions.  Sampling the continuous
    square grid directly avoids a large intermediate tensor while preserving
    the exact content-box coordinates and aspect ratio.
    """

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("features must have shape [C, H, W] and be finite")
    if values.shape[1] != values.shape[2]:
        raise ValueError("RAD-DINO feature grid must be square")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output dimensions must be positive")
    x0, y0, x1, y1 = projection.content_box
    side = float(projection.padded_side)
    if not (0 <= x0 < x1 <= side and 0 <= y0 < y1 <= side):
        raise ValueError("Content box lies outside the padded square")
    xs = x0 + (np.arange(output_width, dtype=np.float32) + 0.5) * (
        (x1 - x0) / float(output_width)
    )
    ys = y0 + (np.arange(output_height, dtype=np.float32) + 0.5) * (
        (y1 - y0) / float(output_height)
    )
    grid_x = np.broadcast_to(2.0 * xs[None, :] / side - 1.0, (output_height, output_width))
    grid_y = np.broadcast_to(2.0 * ys[:, None] / side - 1.0, (output_height, output_width))
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1))[None]
    sampled = F.grid_sample(
        torch.from_numpy(values)[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0].numpy().astype(np.float32, copy=False)


def _resize_map(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Map must be finite and two-dimensional")
    resized = F.interpolate(
        torch.from_numpy(array)[None, None],
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.input_size != 448 or args.input_size % 14:
        raise ValueError("Protocol requires 448-pixel RAD-DINO input")
    if args.graph_size != 64 or args.output_size != 320:
        raise ValueError("Protocol requires a 64-pixel graph and 320-pixel maps")
    if args.projection_dim != 16 or args.projection_seed != 42:
        raise ValueError("Protocol requires the frozen 768-to-16 seed-42 projection")
    if (
        args.foreground_fraction != 0.01
        or args.background_fraction != 0.50
        or args.geodesic_ratio != 1.0
    ):
        raise ValueError("Geodesic seed/fusion settings differ from protocol")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if (
        len(rows) != 371
        or sum(int(row["tumor"]) for row in rows) != 184
        or sum(1 - int(row["tumor"]) for row in rows) != 187
    ):
        raise ValueError("Frozen validation cohort differs from 371/184/187")

    affinity_root = args.affinity_input_root.resolve()
    affinity_records, affinity_contract = load_affinity_selector_contract(
        manifest_path=affinity_root / "prediction_manifest.csv",
        package_metadata_path=affinity_root / "selector_input_manifest.json",
        prediction_freeze_path=affinity_root / "prediction_freeze.json",
        expected_manifest_sha256=args.expected_affinity_manifest_sha256,
        expected_package_metadata_sha256=args.expected_affinity_package_sha256,
        expected_prediction_freeze_sha256=args.expected_affinity_freeze_sha256,
        expected_source_commit=args.expected_affinity_source_commit,
        expected_protocol_sha256=args.expected_affinity_protocol_sha256,
        expected_checkpoint_sha256=args.expected_affinity_checkpoint_sha256,
        split="val",
        split_manifest_sha256=args.expected_split_sha256,
        image_size=args.output_size,
    )
    if set(affinity_records) != {row["image_id"] for row in rows}:
        raise ValueError("Affinity input cohort differs from frozen validation split")

    model_audit = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers {transformers.__version__} differs from "
            f"{EXPECTED_TRANSFORMERS_VERSION}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("RAD-DINO geodesic probe requires Kaggle CUDA")
    device = torch.device("cuda")
    encoder = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True
    ).eval().to(device)
    if (
        int(getattr(encoder.config, "hidden_size", -1)) != 768
        or int(getattr(encoder.config, "patch_size", -1)) != 14
    ):
        raise RuntimeError("RAD-DINO architecture differs from protocol")

    random_projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    projection_hash = projection_sha256(random_projection)
    prediction_dir = args.output_dir / "predictions"
    map_dir = prediction_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    source_map_bytes = 0
    output_map_bytes = 0
    grid_size = args.input_size // 14

    for index, row in enumerate(rows):
        image_id = row["image_id"]
        image_label = int(row["tumor"])
        source_record = affinity_records[image_id]
        source_map = load_affinity_selector_map(
            source_record,
            root=affinity_root,
            expected_image_id=image_id,
            expected_group_id=row["group_id"],
            expected_image_label=image_label,
            image_size=args.output_size,
        )
        source_path = affinity_root / source_record["map_path"]
        source_map_bytes += source_path.stat().st_size
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        foreground = radiograph_foreground_mask(
            image,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        if not foreground.any():
            raise RuntimeError(f"Empty radiograph foreground: {image_id}")

        diagnostics: dict[str, float | int]
        if image_label == 0:
            output_map = np.zeros(
                (args.output_size, args.output_size), dtype=np.float32
            )
            diagnostics = {
                "valid_pixels": args.graph_size * args.graph_size,
                "foreground_seed_pixels": 0,
                "background_seed_pixels": args.graph_size * args.graph_size,
                "ambiguous_pixels": 0,
                "foreground_fraction": args.foreground_fraction,
                "background_fraction": args.background_fraction,
                "ratio": args.geodesic_ratio,
                "refined_min": 0.0,
                "refined_max": 0.0,
                "refined_mean": 0.0,
            }
        else:
            normalized, square_projection = _raw_normalized_square(
                image, input_size=args.input_size
            )
            patch_tokens = _extract_patch_tokens(
                encoder,
                normalized,
                grid_size=grid_size,
                device=device,
            )
            projected = project_features(patch_tokens, random_projection)
            structural = project_square_features(
                projected.transpose(2, 0, 1),
                square_projection,
                output_height=args.graph_size,
                output_width=args.graph_size,
            )
            grayscale = np.asarray(
                image.convert("L").resize(
                    (args.graph_size, args.graph_size),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            ) / 255.0
            graph_source = _resize_map(source_map, args.graph_size)
            result = geodesic_seed_expansion(
                graph_source,
                grayscale,
                structural,
                np.ones((args.graph_size, args.graph_size), dtype=bool),
                foreground_fraction=args.foreground_fraction,
                background_fraction=args.background_fraction,
                ratio=args.geodesic_ratio,
            )
            output_map = _resize_map(result.probability, args.output_size)
            output_map = np.clip(output_map, 0.0, 1.0)
            output_map[~foreground] = 0.0
            diagnostics = result.diagnostics

        relative = Path("maps") / f"{Path(image_id).stem}.npy"
        output_path = prediction_dir / relative
        save_float_map(output_path, output_map)
        output_map_bytes += output_path.stat().st_size
        records.append(
            {
                "image_id": image_id,
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "map_path": relative.as_posix(),
                "map_sha256": sha256_file(output_path),
                "source_map_sha256": source_record["map_sha256"],
                "raw_mean": float(output_map.mean()),
                "raw_p99": float(np.percentile(output_map[foreground], 99)),
                "raw_max": float(output_map.max()),
                **diagnostics,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"geodesic maps: {index + 1}/{len(rows)}", flush=True)

    manifest_path = prediction_dir / "prediction_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest_hash = sha256_file(manifest_path)
    generation = {
        "stage": "prediction-first RAD-DINO geodesic seed expansion",
        "scientific_role": "spatial mechanism probe; not a pseudo-mask consumer",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_manifest_sha256": args.expected_split_sha256,
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "source_affinity_contract": affinity_contract,
        "model_snapshot": model_audit,
        "projection": {
            "input_dim": 768,
            "output_dim": args.projection_dim,
            "seed": args.projection_seed,
            "sha256": projection_hash,
        },
        "graph": {
            "input_size": args.input_size,
            "graph_size": args.graph_size,
            "output_size": args.output_size,
            "connectivity": 8,
            "feature_branches": [
                "per-image robust-scaled grayscale",
                "per-image robust-scaled frozen RAD-DINO projected tokens",
            ],
            "branch_weighting": "equal energy; no fitted scalar weight",
            "foreground_fraction": args.foreground_fraction,
            "background_fraction": args.background_fraction,
            "geodesic_ratio": args.geodesic_ratio,
            "ambiguous_pixels": "continuous and unlabeled",
        },
        "maps": {
            "count": len(records),
            "source_bytes": source_map_bytes,
            "output_bytes": output_map_bytes,
            "manifest_sha256": manifest_hash,
        },
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    generation_path = prediction_dir / "generation_metadata.json"
    generation_path.write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "stage": generation["stage"],
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_manifest_sha256": args.expected_split_sha256,
        "prediction_manifest_sha256": manifest_hash,
        "generation_metadata_sha256": sha256_file(generation_path),
        "validation_predictions": len(records),
        "physical_map_bytes": output_map_bytes,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "prediction_manifest_sha256": manifest_hash,
                "generation_metadata_sha256": sha256_file(generation_path),
                "prediction_freeze_sha256": sha256_file(freeze_path),
                "projection_sha256": projection_hash,
                "maps": len(records),
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
