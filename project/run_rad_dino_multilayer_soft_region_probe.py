from __future__ import annotations

"""Train and freeze a prediction-first multi-layer RAD-DINO decoder probe.

Only clean-training radiographs, their binary image labels, frozen RAD-DINO
features, and a train-normal-calibrated anomaly teacher are consumed.  This
runner has no segmentation-dataset import and never opens validation masks.
The separate evaluator may run only after prediction_freeze.json exists.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from generate_nominal_patch_memory_saliency import spatial_context_scores
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
    project_square_map,
    radiograph_foreground_mask,
)
from models.nominal_patch_memory import (
    FrozenNormalCalibration,
    calibration_sha256,
    make_seeded_random_projection,
    projection_sha256,
    retrieve_normal_context,
    retrieve_normal_context_matrix,
)
from models.rad_dino_multilayer_soft_region_decoder import (
    MultiLayerSoftRegionConfig,
    RadDinoMultiLayerSoftRegionDecoder,
    bidirectional_affinity_refinement,
    horizontal_flip_consistency_loss,
    image_level_loss,
    local_affinity,
    make_guidance,
    soft_affinity_pair_loss,
    soft_region_pseudo_loss,
)


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
RAD_DINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
RAD_DINO_STD = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)
SELECTED_HIDDEN_LAYERS = (4, 8, 12)


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
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--teacher-spatial-radius", type=int, default=2)
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--affinity-loss-weight", type=float, default=0.1)
    parser.add_argument("--flip-consistency-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return sha256_file(path)


def close_memmap(values: np.ndarray) -> None:
    if isinstance(values, np.memmap):
        values.flush()
        mmap_handle = getattr(values, "_mmap", None)
        if mmap_handle is not None and not mmap_handle.closed:
            mmap_handle.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _raw_and_normalized_square(
    image: Image.Image,
    *,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor, SquareProjection]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    raw = torch.from_numpy(values).permute(2, 0, 1)
    normalized = (raw - RAD_DINO_MEAN) / RAD_DINO_STD
    return raw, normalized, projection


def content_validity(
    projection: SquareProjection,
    *,
    grid_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = projection.content_box
    side = float(projection.padded_side)
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) * side / grid_size
    valid_x = (centers >= x0) & (centers < x1)
    valid_y = (centers >= y0) & (centers < y1)
    valid = valid_y[:, None] & valid_x[None, :]
    if not valid.any():
        raise RuntimeError("Image content covers no RAD-DINO patch center")
    return valid


def extract_projected_features(
    encoder: nn.Module,
    pixels: torch.Tensor,
    projection: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        output = encoder(
            pixel_values=pixels[None].to(device, non_blocking=True),
            output_hidden_states=True,
        )
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != 13:
        raise RuntimeError("RAD-DINO must expose embedding plus 12 hidden states")
    expected = grid_size * grid_size + 1
    selected: list[torch.Tensor] = []
    for layer_index in SELECTED_HIDDEN_LAYERS:
        hidden = hidden_states[layer_index]
        if hidden.shape != (1, expected, 768):
            raise RuntimeError(
                f"Unexpected layer-{layer_index} shape {tuple(hidden.shape)}"
            )
        patches = hidden[0, 1:].reshape(grid_size, grid_size, 768).float()
        selected.append(F.normalize(patches @ projection, dim=-1))
    final_global = F.normalize(
        hidden_states[SELECTED_HIDDEN_LAYERS[-1]][0, 0].float(),
        dim=0,
    )
    return (
        final_global.cpu().numpy().astype(np.float32),
        torch.stack(selected, dim=0).cpu().numpy().astype(np.float16),
    )


def _open_training_caches(
    scratch: Path,
    *,
    images: int,
    grid_size: int,
    guidance_size: int,
    projection_dim: int,
) -> dict[str, np.memmap]:
    return {
        "tokens": np.lib.format.open_memmap(
            scratch / "train_projected_tokens.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                images,
                len(SELECTED_HIDDEN_LAYERS),
                grid_size,
                grid_size,
                projection_dim,
            ),
        ),
        "globals": np.lib.format.open_memmap(
            scratch / "train_globals.npy",
            mode="w+",
            dtype=np.float32,
            shape=(images, 768),
        ),
        "guidance": np.lib.format.open_memmap(
            scratch / "train_guidance.npy",
            mode="w+",
            dtype=np.float16,
            shape=(images, 3, guidance_size, guidance_size),
        ),
        "validity": np.lib.format.open_memmap(
            scratch / "train_validity.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(images, grid_size, grid_size),
        ),
    }


def build_training_feature_cache(
    encoder: nn.Module,
    rows: list[dict[str, str]],
    projection: torch.Tensor,
    args: argparse.Namespace,
    *,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> tuple[dict[str, np.memmap], np.ndarray]:
    caches = _open_training_caches(
        args.scratch_dir,
        images=len(rows),
        grid_size=grid_size,
        guidance_size=guidance_size,
        projection_dim=args.projection_dim,
    )
    normal_indices: list[int] = []
    for index, row in enumerate(rows):
        image = Image.open(
            locate_verified_image(args.dataset_root, row)
        ).convert("RGB")
        raw, normalized, square_projection = _raw_and_normalized_square(
            image,
            input_size=args.input_size,
        )
        global_feature, projected_tokens = extract_projected_features(
            encoder,
            normalized,
            projection,
            grid_size=grid_size,
            device=device,
        )
        caches["tokens"][index] = projected_tokens
        caches["globals"][index] = global_feature
        caches["guidance"][index] = (
            make_guidance(raw[None], output_size=guidance_size)[0]
            .numpy()
            .astype(np.float16)
        )
        caches["validity"][index] = content_validity(
            square_projection,
            grid_size=grid_size,
        ).astype(np.uint8)
        if row["tumor"] == "0":
            normal_indices.append(index)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"multi-layer train features: {index + 1}/{len(rows)}", flush=True)
    for values in caches.values():
        values.flush()
    if len(normal_indices) != 1493:
        raise RuntimeError(f"Expected 1493 train normals, got {len(normal_indices)}")
    return caches, np.asarray(normal_indices, dtype=np.int32)


def build_teacher_cache(
    rows: list[dict[str, str]],
    caches: dict[str, np.memmap],
    normal_indices: np.ndarray,
    args: argparse.Namespace,
    *,
    grid_size: int,
    projection_np: np.ndarray,
    device: torch.device,
) -> tuple[np.memmap, dict[str, object]]:
    final_layer = len(SELECTED_HIDDEN_LAYERS) - 1
    normal_globals = np.asarray(
        caches["globals"][normal_indices],
        dtype=np.float32,
    )
    normal_context, normal_context_similarity = retrieve_normal_context_matrix(
        normal_globals,
        top_k=args.top_k,
    )
    normal_raw_path = args.scratch_dir / "normal_teacher_raw.npy"
    normal_raw = np.lib.format.open_memmap(
        normal_raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(normal_indices), grid_size, grid_size),
    )
    for normal_index in range(len(normal_indices)):
        query = np.asarray(
            caches["tokens"][normal_indices[normal_index], final_layer],
            dtype=np.float32,
        )
        contexts = np.asarray(
            caches["tokens"][
                normal_indices[normal_context[normal_index]],
                final_layer,
            ],
            dtype=np.float32,
        )
        normal_raw[normal_index] = spatial_context_scores(
            query,
            contexts,
            radius=args.teacher_spatial_radius,
            device=device,
        ).astype(np.float16)
        if (
            (normal_index + 1) % 25 == 0
            or normal_index + 1 == len(normal_indices)
        ):
            print(
                f"multi-layer normal calibration: "
                f"{normal_index + 1}/{len(normal_indices)}",
                flush=True,
            )
    normal_raw.flush()
    calibration = FrozenNormalCalibration.fit(
        np.asarray(normal_raw, dtype=np.float32)
    )
    teacher_path = args.scratch_dir / "train_teacher.npy"
    teachers = np.lib.format.open_memmap(
        teacher_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), grid_size, grid_size),
    )
    teachers[:] = 0.0
    positive_count = 0
    for index, row in enumerate(rows):
        if row["tumor"] != "1":
            continue
        context_indices, _similarities = retrieve_normal_context(
            np.asarray(caches["globals"][index], dtype=np.float32),
            normal_globals,
            top_k=args.top_k,
        )
        query = np.asarray(
            caches["tokens"][index, final_layer],
            dtype=np.float32,
        )
        contexts = np.asarray(
            caches["tokens"][normal_indices[context_indices], final_layer],
            dtype=np.float32,
        )
        raw = spatial_context_scores(
            query,
            contexts,
            radius=args.teacher_spatial_radius,
            device=device,
        )
        teachers[index] = calibration.transform(raw).astype(np.float16)
        positive_count += 1
        if (
            positive_count % 25 == 0
            or positive_count == len(rows) - len(normal_indices)
        ):
            print(
                f"multi-layer positive teachers: "
                f"{positive_count}/{len(rows) - len(normal_indices)}",
                flush=True,
            )
    teachers.flush()
    positive_values = np.asarray(
        teachers[[row["tumor"] == "1" for row in rows]],
        dtype=np.float32,
    )
    foreground_counts = (
        positive_values > MultiLayerSoftRegionConfig.foreground_start
    ).sum(axis=(1, 2))
    metadata = {
        "normal_images": len(normal_indices),
        "positive_images": positive_count,
        "selected_hidden_layers": list(SELECTED_HIDDEN_LAYERS),
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "projection_semantic_sha256": projection_sha256(projection_np),
        "normal_context_top_k": args.top_k,
        "normal_context_indices_sha256": hashlib.sha256(
            normal_context.astype(np.int32).tobytes()
        ).hexdigest(),
        "normal_context_similarities_sha256": hashlib.sha256(
            normal_context_similarity.astype(np.float32).tobytes()
        ).hexdigest(),
        "teacher_spatial_radius": args.teacher_spatial_radius,
        "calibration": {
            **calibration.metadata(),
            "semantic_sha256": calibration_sha256(calibration),
        },
        "soft_region_contract": {
            "foreground_start": MultiLayerSoftRegionConfig.foreground_start,
            "background_end": MultiLayerSoftRegionConfig.background_end,
            "variable_foreground_count_minimum": int(foreground_counts.min()),
            "variable_foreground_count_median": float(
                np.median(foreground_counts)
            ),
            "variable_foreground_count_maximum": int(foreground_counts.max()),
            "positive_images_without_foreground_weight": int(
                (foreground_counts == 0).sum()
            ),
        },
        "positive_teacher_distribution": {
            "minimum": float(positive_values.min()),
            "mean": float(positive_values.mean()),
            "p90": float(np.percentile(positive_values, 90)),
            "p99": float(np.percentile(positive_values, 99)),
            "maximum": float(positive_values.max()),
        },
        "teacher_cache_sha256": sha256(teacher_path),
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    close_memmap(normal_raw)
    del normal_raw
    gc.collect()
    normal_raw_path.unlink()
    return teachers, metadata


class FrozenMultiLayerCacheDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        caches: dict[str, np.memmap],
        teachers: np.memmap,
    ) -> None:
        self.rows = rows
        self.caches = caches
        self.teachers = teachers

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = np.asarray(self.caches["tokens"][index], dtype=np.float32)
        guidance = np.asarray(self.caches["guidance"][index], dtype=np.float32)
        teacher = np.asarray(self.teachers[index], dtype=np.float32)
        validity = np.asarray(
            self.caches["validity"][index],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(tokens)),
            torch.from_numpy(np.ascontiguousarray(guidance)),
            torch.from_numpy(np.ascontiguousarray(teacher))[None],
            torch.from_numpy(np.ascontiguousarray(validity))[None],
            torch.tensor(float(self.rows[index]["tumor"]), dtype=torch.float32),
        )


def train_decoder(
    rows: list[dict[str, str]],
    caches: dict[str, np.memmap],
    teachers: np.memmap,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[
    RadDinoMultiLayerSoftRegionDecoder,
    list[dict[str, object]],
]:
    config = MultiLayerSoftRegionConfig(
        input_dim=args.projection_dim,
        layer_count=len(SELECTED_HIDDEN_LAYERS),
    )
    decoder = RadDinoMultiLayerSoftRegionDecoder(config).to(device)
    dataset = FrozenMultiLayerCacheDataset(rows, caches, teachers)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals = {
            "total_loss": [],
            "image_loss": [],
            "soft_region_loss": [],
            "soft_affinity_loss": [],
            "flip_consistency_loss": [],
        }
        correct = 0
        image_count = 0
        for tokens, guidance, teacher, valid, labels in loader:
            tokens = tokens.to(device, non_blocking=True)
            guidance = guidance.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, affinity_features, layer_weights = decoder(tokens, guidance)
            filtered_weights, learned_affinity, pair_validity = local_affinity(
                affinity_features,
                tokens[:, -1],
                radius=config.affinity_radius,
                temperature=config.affinity_temperature,
                frozen_similarity_power=config.frozen_similarity_power,
            )
            refined_teacher = bidirectional_affinity_refinement(
                teacher,
                filtered_weights,
                radius=config.affinity_radius,
                steps=config.refinement_steps,
                residual=config.refinement_residual,
            )
            loss_image, pooled = image_level_loss(
                logits,
                labels,
                alpha=config.smoothmax_alpha,
            )
            loss_soft_region = soft_region_pseudo_loss(
                logits,
                refined_teacher,
                labels,
                valid_region=valid,
                foreground_start=config.foreground_start,
                background_end=config.background_end,
            )
            loss_affinity = soft_affinity_pair_loss(
                learned_affinity,
                pair_validity,
                teacher,
                labels,
                radius=config.affinity_radius,
                foreground_start=config.foreground_start,
                background_end=config.background_end,
                valid_region=valid,
            )
            flipped_logits, _flipped_features, _flipped_layer_weights = decoder(
                tokens.flip(3),
                guidance.flip(3),
            )
            loss_consistency = horizontal_flip_consistency_loss(
                logits,
                flipped_logits,
                valid_region=valid,
            )
            loss = (
                loss_image
                + args.pseudo_loss_weight * loss_soft_region
                + args.affinity_loss_weight * loss_affinity
                + args.flip_consistency_weight * loss_consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()
            for name, value in (
                ("total_loss", loss),
                ("image_loss", loss_image),
                ("soft_region_loss", loss_soft_region),
                ("soft_affinity_loss", loss_affinity),
                ("flip_consistency_loss", loss_consistency),
            ):
                totals[name].append(float(value.detach().cpu()))
            correct += int(((pooled >= 0.5) == (labels >= 0.5)).sum())
            image_count += int(labels.numel())
        record = {
            "epoch": epoch,
            **{
                name: float(np.mean(values))
                for name, values in totals.items()
            },
            "training_accuracy": float(correct / max(image_count, 1)),
            "layer_weights": [
                float(value)
                for value in layer_weights.detach().cpu().tolist()
            ],
        }
        history.append(record)
        print(
            f"multi-layer decoder epoch {epoch}/{args.epochs}: {record}",
            flush=True,
        )
    return decoder.eval(), history


def save_checkpoint(
    decoder: RadDinoMultiLayerSoftRegionDecoder,
    history: list[dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    checkpoint = args.output_dir / "rad_dino_multilayer_soft_region_decoder.pt"
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "config": decoder.config.__dict__,
            "selected_hidden_layers": SELECTED_HIDDEN_LAYERS,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pseudo_loss_weight": args.pseudo_loss_weight,
            "affinity_loss_weight": args.affinity_loss_weight,
            "flip_consistency_weight": args.flip_consistency_weight,
            "seed": args.seed,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "history": history,
        },
        checkpoint,
    )
    return checkpoint


def write_validation_maps(
    encoder: nn.Module,
    decoder: RadDinoMultiLayerSoftRegionDecoder,
    projection: torch.Tensor,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> str:
    prediction_dir = args.output_dir / "predictions"
    map_dir = prediction_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        image = Image.open(
            locate_verified_image(args.dataset_root, row)
        ).convert("RGB")
        raw, normalized, square_projection = _raw_and_normalized_square(
            image,
            input_size=args.input_size,
        )
        _global, projected_tokens = extract_projected_features(
            encoder,
            normalized,
            projection,
            grid_size=grid_size,
            device=device,
        )
        tokens = torch.from_numpy(
            projected_tokens.astype(np.float32)
        )[None].to(device)
        guidance = make_guidance(
            raw[None],
            output_size=guidance_size,
        ).to(device)
        with torch.inference_mode():
            logits, _features, _layer_weights = decoder(tokens, guidance)
            flipped_logits, _features, _layer_weights = decoder(
                tokens.flip(3),
                guidance.flip(3),
            )
            square_map = (
                0.5
                * (
                    torch.sigmoid(logits)
                    + torch.sigmoid(flipped_logits).flip(-1)
                )
            )[0, 0].cpu().numpy()
        output_map = project_square_map(
            square_map.astype(np.float32),
            square_projection,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        foreground = radiograph_foreground_mask(
            image,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        if not foreground.any():
            raise RuntimeError(f"Empty radiograph foreground: {row['image_id']}")
        output_map[~foreground] = 0.0
        output_map = np.clip(output_map, 0.0, 1.0)
        relative = Path("maps") / f"{Path(row['image_id']).stem}.npy"
        save_float_map(prediction_dir / relative, output_map)
        records.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "map_path": relative.as_posix(),
                "map_sha256": sha256(prediction_dir / relative),
                "raw_mean": float(output_map.mean()),
                "raw_p99": float(np.percentile(output_map[foreground], 99)),
                "raw_max": float(output_map.max()),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(
                f"multi-layer validation maps: {index + 1}/{len(rows)}",
                flush=True,
            )
    manifest = prediction_dir / "prediction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return sha256(manifest)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-layer decoder probe requires a Kaggle GPU")
    if (
        args.input_size != 448
        or args.output_size != 320
        or args.epochs != 12
        or args.batch_size != 8
        or args.seed != 42
    ):
        raise ValueError("Frozen geometry/budget is 448/320, 12 epochs, batch 8")
    if (
        args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.top_k != 8
        or args.teacher_spatial_radius != 2
        or args.pseudo_loss_weight != 1.0
        or args.affinity_loss_weight != 0.1
        or args.flip_consistency_weight != 0.2
    ):
        raise ValueError("Multi-layer decoder hyperparameters differ from protocol")
    if args.output_dir.exists() or args.scratch_dir.exists():
        raise FileExistsError("output-dir and scratch-dir must not exist")
    args.output_dir.mkdir(parents=True)
    args.scratch_dir.mkdir(parents=True)
    seed_everything(args.seed)
    started = datetime.now(timezone.utc)
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
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("Frozen train/validation cohort mismatch")
    if any(
        len({Path(row["image_id"]).stem for row in rows}) != len(rows)
        for rows in (train_rows, val_rows)
    ):
        raise RuntimeError("Image stems must be unique within each split")
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers must be {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    device = torch.device("cuda")
    encoder = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
    ).eval().to(device)
    if (
        int(encoder.config.patch_size) != 14
        or int(encoder.config.hidden_size) != 768
        or int(encoder.config.num_hidden_layers) != 12
    ):
        raise RuntimeError("RAD-DINO snapshot architecture differs from protocol")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    grid_size = args.input_size // 14
    guidance_size = grid_size * MultiLayerSoftRegionConfig.decoder_scale
    projection_np = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    projection = torch.from_numpy(projection_np).to(device)
    caches, normal_indices = build_training_feature_cache(
        encoder,
        train_rows,
        projection,
        args,
        grid_size=grid_size,
        guidance_size=guidance_size,
        device=device,
    )
    teachers, teacher_metadata = build_teacher_cache(
        train_rows,
        caches,
        normal_indices,
        args,
        grid_size=grid_size,
        projection_np=projection_np,
        device=device,
    )
    teacher_metadata_path = args.output_dir / "teacher_metadata.json"
    teacher_metadata_path.write_text(
        json.dumps(teacher_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    decoder, history = train_decoder(
        train_rows,
        caches,
        teachers,
        args,
        device=device,
    )
    checkpoint = save_checkpoint(decoder, history, args)
    checkpoint_sha = sha256(checkpoint)
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_manifest_sha = write_validation_maps(
        encoder,
        decoder,
        projection,
        val_rows,
        args,
        grid_size=grid_size,
        guidance_size=guidance_size,
        device=device,
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "teacher_metadata_sha256": sha256(teacher_metadata_path),
        "training_history_sha256": sha256(history_path),
        "prediction_manifest_sha256": prediction_manifest_sha,
        "validation_predictions": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2) + "\n",
        encoding="utf-8",
    )
    source_root = Path(__file__).resolve().parent
    run_manifest = {
        "run_id": args.output_dir.name,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": snapshot,
        "selected_hidden_layers": list(SELECTED_HIDDEN_LAYERS),
        "source_files": {
            "runner": sha256(Path(__file__).resolve()),
            "decoder": sha256(
                source_root
                / "models"
                / "rad_dino_multilayer_soft_region_decoder.py"
            ),
            "nominal_memory": sha256(
                source_root / "models" / "nominal_patch_memory.py"
            ),
            "mae_reconstruction": sha256(
                source_root / "models" / "mae_reconstruction.py"
            ),
        },
        "checkpoint_sha256": checkpoint_sha,
        "teacher_metadata_sha256": sha256(teacher_metadata_path),
        "training_history_sha256": sha256(history_path),
        "prediction_freeze_sha256": sha256(freeze_path),
        "cohort": {
            "train": 2981,
            "train_normal": 1493,
            "train_tumor": 1488,
            "validation": 371,
            "validation_tumor": 184,
            "validation_normal": 187,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "prediction_tta": "mean of original and aligned horizontal flip",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    close_memmap(teachers)
    del teachers
    for values in caches.values():
        close_memmap(values)
    caches.clear()
    gc.collect()
    for path in args.scratch_dir.iterdir():
        path.unlink()
    args.scratch_dir.rmdir()
    print(
        json.dumps(
            {
                "run_manifest": run_manifest,
                "prediction_freeze": freeze,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
