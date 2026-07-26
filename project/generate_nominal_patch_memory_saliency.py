from __future__ import annotations

"""Generate prediction-first RAD-DINO nominal patch-memory maps.

The memory and both calibrations are built exclusively from clean-train
normal radiographs selected by binary image-level labels. Segmentation
annotations are never enumerated or opened in this stage.
"""

import argparse
import csv
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
        pad_to_square,
        project_square_map,
        radiograph_foreground_mask,
    )
    from .models.nominal_patch_memory import (
        FrozenNormalCalibration,
        calibration_sha256,
        fixed_tile_layout,
        make_seeded_random_projection,
        merge_overlapping_tile_maps,
        projection_sha256,
        retrieve_normal_context,
        retrieve_normal_context_matrix,
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
        pad_to_square,
        project_square_map,
        radiograph_foreground_mask,
    )
    from models.nominal_patch_memory import (
        FrozenNormalCalibration,
        calibration_sha256,
        fixed_tile_layout,
        make_seeded_random_projection,
        merge_overlapping_tile_maps,
        projection_sha256,
        retrieve_normal_context,
        retrieve_normal_context_matrix,
    )


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
RAD_DINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
RAD_DINO_STD = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)
ARM_FORMULAS = {
    "single_scale": "frozen-normal ECDF of full-view spatial patch distance",
    "multiscale": (
        "0.5 * calibrated full view + 0.5 * overlap-averaged calibrated "
        "four-corner tile views"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--tile-size", type=int, default=280)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--spatial-radius", type=int, default=2)
    return parser.parse_args()


def _view_tensors(
    square: Image.Image,
    *,
    input_size: int,
    tile_size: int,
) -> tuple[torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    layout = fixed_tile_layout(image_size=input_size, tile_size=tile_size)
    images = [resized]
    for box in layout:
        images.append(
            resized.crop(box).resize(
                (input_size, input_size),
                Image.Resampling.BICUBIC,
            )
        )
    tensors: list[torch.Tensor] = []
    for image in images:
        values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1)
        tensors.append((tensor - RAD_DINO_MEAN) / RAD_DINO_STD)
    return torch.stack(tensors), layout


def _extract_projected_views(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    projection: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_values = pixel_values.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        hidden = model(pixel_values=pixel_values).last_hidden_state
    expected_tokens = grid_size * grid_size
    if hidden.ndim != 3 or hidden.shape[1] != expected_tokens + 1:
        raise RuntimeError(
            f"Unexpected RAD-DINO token shape {tuple(hidden.shape)}; "
            f"expected CLS + {expected_tokens} patches"
        )
    global_feature = F.normalize(hidden[0, 0].float(), dim=0)
    patches = hidden[:, 1:].float() @ projection
    patches = F.normalize(patches, dim=-1)
    grids = patches.reshape(
        hidden.shape[0],
        grid_size,
        grid_size,
        projection.shape[1],
    )
    return (
        global_feature.detach().cpu().numpy().astype(np.float32),
        grids.detach().cpu().numpy().astype(np.float16),
    )


def spatial_context_scores(
    query_grid: np.ndarray,
    context_grids: np.ndarray,
    *,
    radius: int,
    device: torch.device,
) -> np.ndarray:
    """GPU-vectorized same-anatomy nearest-neighbour patch distance."""
    query = torch.from_numpy(np.asarray(query_grid, dtype=np.float32)).to(device)
    context = torch.from_numpy(np.asarray(context_grids, dtype=np.float32)).to(device)
    if query.ndim != 3 or context.ndim != 4 or context.shape[1:] != query.shape:
        raise ValueError("Query/context grids are incompatible")
    query = F.normalize(query, dim=-1)
    context = F.normalize(context, dim=-1)
    count, height, width, dimension = context.shape
    kernel = 2 * radius + 1
    neighborhoods = F.unfold(
        context.permute(0, 3, 1, 2),
        kernel_size=kernel,
        padding=radius,
    )
    neighborhoods = neighborhoods.reshape(
        count,
        dimension,
        kernel * kernel,
        height * width,
    ).permute(3, 0, 2, 1)
    query_flat = query.reshape(height * width, dimension)
    similarities = torch.einsum(
        "pd,pknd->pkn",
        query_flat,
        neighborhoods,
    )

    # F.unfold zero-pads outside the grid. Zero vectors cannot beat a
    # nonnegative cosine match in practice, but mark them invalid explicitly
    # so the boundary contract does not depend on feature geometry.
    validity = torch.ones(
        (1, 1, height, width),
        dtype=torch.float32,
        device=device,
    )
    validity = F.unfold(validity, kernel_size=kernel, padding=radius)
    validity = validity.reshape(
        kernel * kernel,
        height * width,
    ).T[:, None, :]
    similarities = similarities.masked_fill(validity <= 0, -torch.inf)
    best = similarities.amax(dim=(1, 2))
    if not torch.isfinite(best).all():
        raise RuntimeError("Spatial patch matching produced an invalid score")
    scores = torch.clamp(1.0 - best, 0.0, 2.0)
    return scores.reshape(height, width).cpu().numpy().astype(np.float32)


def _resize_map(values: np.ndarray, *, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy().astype(np.float32, copy=False)


def _save_array(path: Path, values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers must be {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Nominal patch-memory generation requires a Kaggle GPU")
    if args.input_size % 14:
        raise ValueError("RAD-DINO input size must be divisible by patch size 14")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = args.output_dir / "memory_evidence"
    evidence_dir.mkdir()

    snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    train_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    normal_rows = [row for row in train_rows if row["tumor"] == "0"]
    validation_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(train_rows) != 2981 or len(normal_rows) != 1493:
        raise RuntimeError("Unexpected clean-train/normal cohort")
    if len(validation_rows) != 371:
        raise RuntimeError("Unexpected validation cohort")

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
    ).eval().to(device)
    if int(model.config.patch_size) != 14 or int(model.config.hidden_size) != 768:
        raise RuntimeError("RAD-DINO snapshot architecture differs from protocol")
    grid_size = args.input_size // int(model.config.patch_size)
    view_count = 5
    projection_np = make_seeded_random_projection(
        input_dim=int(model.config.hidden_size),
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    projection = torch.from_numpy(projection_np).to(device)

    patch_path = args.scratch_dir / "normal_patch_bank.npy"
    global_path = args.scratch_dir / "normal_global_bank.npy"
    patch_bank = np.lib.format.open_memmap(
        patch_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(normal_rows),
            view_count,
            grid_size,
            grid_size,
            args.projection_dim,
        ),
    )
    global_bank = np.lib.format.open_memmap(
        global_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(normal_rows), int(model.config.hidden_size)),
    )
    layout: tuple[tuple[int, int, int, int], ...] | None = None
    for index, row in enumerate(normal_rows):
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        square, _ = pad_to_square(image, fill=0)
        tensors, observed_layout = _view_tensors(
            square,
            input_size=args.input_size,
            tile_size=args.tile_size,
        )
        if layout is None:
            layout = observed_layout
        elif layout != observed_layout:
            raise RuntimeError("Tile layout drifted during normal-bank extraction")
        global_feature, patch_features = _extract_projected_views(
            model,
            tensors,
            projection,
            grid_size=grid_size,
            device=device,
        )
        global_bank[index] = global_feature
        patch_bank[index] = patch_features
        if (index + 1) % 25 == 0 or index + 1 == len(normal_rows):
            print(f"normal feature bank: {index + 1}/{len(normal_rows)}", flush=True)
    patch_bank.flush()
    global_bank.flush()
    if layout is None:
        raise RuntimeError("Normal bank is empty")

    normal_context, normal_context_similarity = retrieve_normal_context_matrix(
        np.asarray(global_bank),
        top_k=args.top_k,
    )
    calibration_path = args.scratch_dir / "normal_raw_scores.npy"
    normal_raw = np.lib.format.open_memmap(
        calibration_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(normal_rows), view_count, grid_size, grid_size),
    )
    for index in range(len(normal_rows)):
        context_indices = normal_context[index]
        for view in range(view_count):
            normal_raw[index, view] = spatial_context_scores(
                patch_bank[index, view],
                patch_bank[context_indices, view],
                radius=args.spatial_radius,
                device=device,
            ).astype(np.float16)
        if (index + 1) % 25 == 0 or index + 1 == len(normal_rows):
            print(f"normal calibration: {index + 1}/{len(normal_rows)}", flush=True)
    normal_raw.flush()
    full_calibration = FrozenNormalCalibration.fit(
        np.asarray(normal_raw[:, 0], dtype=np.float32)
    )
    tile_calibration = FrozenNormalCalibration.fit(
        np.asarray(normal_raw[:, 1:], dtype=np.float32)
    )

    evidence = {
        "projection": _save_array(
            evidence_dir / "projection.npy",
            projection_np.astype(np.float32),
        ),
        "normal_global_features": _save_array(
            evidence_dir / "normal_global_features.npy",
            np.asarray(global_bank, dtype=np.float32),
        ),
        "normal_context_indices": _save_array(
            evidence_dir / "normal_context_indices.npy",
            normal_context.astype(np.int32),
        ),
        "normal_context_similarities": _save_array(
            evidence_dir / "normal_context_similarities.npy",
            normal_context_similarity.astype(np.float32),
        ),
        "full_calibration": _save_array(
            evidence_dir / "full_calibration.npy",
            np.asarray(full_calibration.sorted_normal_scores, dtype=np.float32),
        ),
        "tile_calibration": _save_array(
            evidence_dir / "tile_calibration.npy",
            np.asarray(tile_calibration.sorted_normal_scores, dtype=np.float32),
        ),
    }
    scratch_hashes = {
        "normal_patch_bank_sha256": sha256_file(patch_path),
        "normal_global_bank_sha256": sha256_file(global_path),
        "normal_raw_scores_sha256": sha256_file(calibration_path),
    }

    arm_rows: dict[str, list[dict[str, object]]] = {
        arm: [] for arm in ARM_FORMULAS
    }
    for arm in ARM_FORMULAS:
        (args.output_dir / f"{arm}_prediction" / "maps").mkdir(parents=True)

    normal_global_values = np.asarray(global_bank, dtype=np.float32)
    for index, row in enumerate(validation_rows):
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        square, square_projection = pad_to_square(image, fill=0)
        tensors, observed_layout = _view_tensors(
            square,
            input_size=args.input_size,
            tile_size=args.tile_size,
        )
        if observed_layout != layout:
            raise RuntimeError("Validation tile layout differs from frozen normal layout")
        global_feature, patch_features = _extract_projected_views(
            model,
            tensors,
            projection,
            grid_size=grid_size,
            device=device,
        )
        context_indices, context_similarities = retrieve_normal_context(
            global_feature,
            normal_global_values,
            top_k=args.top_k,
        )
        raw_views = [
            spatial_context_scores(
                patch_features[view],
                patch_bank[context_indices, view],
                radius=args.spatial_radius,
                device=device,
            )
            for view in range(view_count)
        ]
        calibrated_full = full_calibration.transform(raw_views[0])
        full_square = _resize_map(calibrated_full, size=args.input_size)
        calibrated_tiles = [
            _resize_map(
                tile_calibration.transform(raw_views[view]),
                size=args.tile_size,
            )
            for view in range(1, view_count)
        ]
        tile_square = merge_overlapping_tile_maps(
            np.stack(calibrated_tiles),
            image_size=args.input_size,
            layout=layout,
        )
        square_maps = {
            "single_scale": full_square,
            "multiscale": 0.5 * full_square + 0.5 * tile_square,
        }
        foreground = radiograph_foreground_mask(
            image,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        for arm, square_map in square_maps.items():
            output_map = project_square_map(
                np.asarray(square_map, dtype=np.float32),
                square_projection,
                output_height=args.output_size,
                output_width=args.output_size,
            )
            output_map[~foreground] = 0.0
            output_map = np.clip(output_map, 0.0, 1.0)
            relative = Path("maps") / f"{Path(row['image_id']).stem}.npy"
            prediction_dir = args.output_dir / f"{arm}_prediction"
            save_float_map(prediction_dir / relative, output_map)
            arm_rows[arm].append(
                {
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "map_path": relative.as_posix(),
                    "map_sha256": sha256_file(prediction_dir / relative),
                    "raw_mean": float(output_map.mean()),
                    "raw_p99": float(np.percentile(output_map, 99)),
                    "raw_max": float(output_map.max()),
                    "context_indices": "|".join(str(value) for value in context_indices),
                    "context_image_ids": "|".join(
                        normal_rows[int(value)]["image_id"] for value in context_indices
                    ),
                    "context_similarities": "|".join(
                        f"{float(value):.9g}" for value in context_similarities
                    ),
                }
            )
        if (index + 1) % 25 == 0 or index + 1 == len(validation_rows):
            print(f"validation predictions: {index + 1}/{len(validation_rows)}", flush=True)

    memory_metadata = {
        "stage": "normal-only context-conditioned RAD-DINO patch memory",
        "scientific_role": "mechanism feasibility; not final pseudo masks",
        "source_commit": args.source_commit,
        "source_files": {
            "generator": sha256_file(Path(__file__).resolve()),
            "mae_reconstruction_io": sha256_file(
                Path(__file__).with_name("mae_reconstruction_io.py")
            ),
            "mae_reconstruction": sha256_file(
                Path(__file__).parent / "models" / "mae_reconstruction.py"
            ),
            "nominal_patch_memory": sha256_file(
                Path(__file__).parent / "models" / "nominal_patch_memory.py"
            ),
        },
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "population": {
            "all_train_images": len(train_rows),
            "normal_memory_images": len(normal_rows),
            "tumor_training_images_used": 0,
            "validation_images": len(validation_rows),
        },
        "model_snapshot": snapshot,
        "feature_contract": {
            "model_patch_size": 14,
            "input_size": args.input_size,
            "grid_size": grid_size,
            "hidden_size": int(model.config.hidden_size),
            "projection_dim": args.projection_dim,
            "projection_seed": args.projection_seed,
            "projection_semantic_sha256": projection_sha256(projection_np),
            "views": view_count,
            "tile_size": args.tile_size,
            "tile_layout": [list(box) for box in layout],
            "top_k_normal_images": args.top_k,
            "spatial_radius_patches": args.spatial_radius,
            "distance": "one minus maximum cosine similarity",
        },
        "calibration": {
            "method": "leave-one-image-out clean-train normal empirical CDF",
            "full": {
                **full_calibration.metadata(),
                "semantic_sha256": calibration_sha256(full_calibration),
            },
            "tiles": {
                **tile_calibration.metadata(),
                "semantic_sha256": calibration_sha256(tile_calibration),
            },
        },
        "evidence_files": evidence,
        "reconstructible_scratch_hashes": scratch_hashes,
        "annotation_contract": "segmentation annotation paths were never enumerated or opened",
        "validation_gt_read": False,
        "test_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    memory_metadata_path = args.output_dir / "memory_metadata.json"
    memory_metadata_path.write_text(
        json.dumps(memory_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    memory_metadata_sha = sha256_file(memory_metadata_path)

    for arm, rows in arm_rows.items():
        prediction_dir = args.output_dir / f"{arm}_prediction"
        manifest_path = prediction_dir / "prediction_manifest.csv"
        _write_manifest(manifest_path, rows)
        metadata = {
            "stage": "validation prediction generation before GT access",
            "scientific_role": "nominal patch-memory mechanism feasibility",
            "arm": arm,
            "formula": ARM_FORMULAS[arm],
            "source_commit": args.source_commit,
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "cohort": len(validation_rows),
            "memory_metadata_sha256": memory_metadata_sha,
            "prediction_manifest_sha256": sha256_file(manifest_path),
            "output_size": args.output_size,
            "normalization": (
                "frozen train-normal ECDF; no validation/per-image min-max"
            ),
            "validation_gt_read": False,
            "test_evaluated": False,
        }
        (prediction_dir / "generation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    # The 2 GB patch bank is deterministically reconstructible and its hash is
    # recorded above; retain only compact evidence plus frozen validation maps.
    del normal_raw, patch_bank, global_bank
    gc.collect()
    for path in (calibration_path, patch_path, global_path):
        path.unlink()
    args.scratch_dir.rmdir()
    print(json.dumps(memory_metadata, indent=2))


if __name__ == "__main__":
    main()
