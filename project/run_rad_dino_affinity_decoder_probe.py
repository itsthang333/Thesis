from __future__ import annotations

"""Train a prediction-first RAD-DINO spatial affinity-decoder probe.

Only the clean training split's binary image labels are consumed before the
validation prediction freeze. A normal patch-memory teacher is reconstructed
from clean-train normal images, then used through fixed per-image confidence
ranks. Validation masks are opened only after every decoder map and its
manifest have been hash-frozen.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from compare_nominal_patch_memory_arms import METRICS, paired_group_bootstrap
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
from models.rad_dino_affinity_decoder import (
    AffinityDecoderConfig,
    RadDinoSpatialDecoder,
    affinity_pair_loss,
    image_level_loss,
    local_affinity,
    make_guidance,
    masked_pseudo_loss,
    propagate_seed_preserving,
)
from generate_nominal_patch_memory_saliency import spatial_context_scores


EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
RAD_DINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
RAD_DINO_STD = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)
GATE_THRESHOLDS = {
    "image_level_auroc_from_raw_p99": 0.65,
    "overall_pixel_auroc": 0.75,
    "small_pixel_auroc": 0.77,
    "overall_dice_p90": 0.10,
    "small_dice_p97": 0.03,
    "medium_dice_p90": 0.12,
    "large_dice_p90": 0.35,
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
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-baseline-per-image-sha256", required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return sha256_file(path)


def close_memmap(values: np.ndarray) -> None:
    """Flush and close an owned NumPy memmap without relying on GC timing."""
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
    """Mark patch centers that lie in the unpadded radiograph content."""
    x0, y0, x1, y1 = projection.content_box
    side = float(projection.padded_side)
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) * side / grid_size
    valid_x = (centers >= x0) & (centers < x1)
    valid_y = (centers >= y0) & (centers < y1)
    valid = valid_y[:, None] & valid_x[None, :]
    if not valid.any():
        raise RuntimeError("Image content does not cover any RAD-DINO patch center")
    return valid


def extract_frozen_features(
    encoder: nn.Module,
    pixels: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        hidden = encoder(
            pixel_values=pixels[None].to(device, non_blocking=True)
        ).last_hidden_state
    expected = grid_size * grid_size + 1
    if hidden.ndim != 3 or hidden.shape != (1, expected, 768):
        raise RuntimeError(
            f"Unexpected RAD-DINO token shape {tuple(hidden.shape)}"
        )
    global_feature = F.normalize(hidden[0, 0].float(), dim=0)
    patch_tokens = hidden[0, 1:].reshape(grid_size, grid_size, 768)
    return (
        global_feature.cpu().numpy().astype(np.float32),
        patch_tokens.cpu().numpy().astype(np.float16),
    )


def _open_training_caches(
    scratch: Path,
    *,
    images: int,
    grid_size: int,
    guidance_size: int,
) -> dict[str, np.memmap]:
    return {
        "tokens": np.lib.format.open_memmap(
            scratch / "train_tokens.npy",
            mode="w+",
            dtype=np.float16,
            shape=(images, grid_size, grid_size, 768),
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
    )
    normal_indices: list[int] = []
    for index, row in enumerate(rows):
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        raw, normalized, projection = _raw_and_normalized_square(
            image, input_size=args.input_size
        )
        global_feature, patch_tokens = extract_frozen_features(
            encoder,
            normalized,
            grid_size=grid_size,
            device=device,
        )
        caches["tokens"][index] = patch_tokens
        caches["globals"][index] = global_feature
        caches["guidance"][index] = (
            make_guidance(raw[None], output_size=guidance_size)[0]
            .numpy()
            .astype(np.float16)
        )
        caches["validity"][index] = content_validity(
            projection, grid_size=grid_size
        ).astype(np.uint8)
        if row["tumor"] == "0":
            normal_indices.append(index)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"training feature cache: {index + 1}/{len(rows)}", flush=True)
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
    device: torch.device,
) -> tuple[np.memmap, dict[str, object]]:
    projection_np = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    projection = torch.from_numpy(projection_np).to(device)
    normal_patch_path = args.scratch_dir / "normal_projected_patches.npy"
    normal_patches = np.lib.format.open_memmap(
        normal_patch_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(normal_indices),
            grid_size,
            grid_size,
            args.projection_dim,
        ),
    )
    projection_batch = 16
    for start in range(0, len(normal_indices), projection_batch):
        indices = normal_indices[start : start + projection_batch]
        tokens = torch.from_numpy(
            np.asarray(caches["tokens"][indices], dtype=np.float32)
        ).to(device)
        projected = F.normalize(tokens @ projection, dim=-1)
        normal_patches[start : start + len(indices)] = (
            projected.cpu().numpy().astype(np.float16)
        )
    normal_patches.flush()

    normal_globals = np.asarray(
        caches["globals"][normal_indices], dtype=np.float32
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
        normal_raw[normal_index] = spatial_context_scores(
            normal_patches[normal_index],
            normal_patches[normal_context[normal_index]],
            radius=args.teacher_spatial_radius,
            device=device,
        ).astype(np.float16)
        if (
            (normal_index + 1) % 25 == 0
            or normal_index + 1 == len(normal_indices)
        ):
            print(
                f"normal teacher calibration: "
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
        tokens = torch.from_numpy(
            np.asarray(caches["tokens"][index], dtype=np.float32)
        ).to(device)
        query = F.normalize(tokens @ projection, dim=-1).cpu().numpy()
        raw = spatial_context_scores(
            query,
            normal_patches[context_indices],
            radius=args.teacher_spatial_radius,
            device=device,
        )
        teachers[index] = calibration.transform(raw).astype(np.float16)
        positive_count += 1
        if positive_count % 25 == 0 or positive_count == len(rows) - len(normal_indices):
            print(
                f"positive training teachers: "
                f"{positive_count}/{len(rows) - len(normal_indices)}",
                flush=True,
            )
    teachers.flush()
    positive_values = np.asarray(
        teachers[[row["tumor"] == "1" for row in rows]], dtype=np.float32
    )
    metadata = {
        "normal_images": len(normal_indices),
        "positive_images": positive_count,
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
        "positive_teacher_distribution": {
            "minimum": float(positive_values.min()),
            "mean": float(positive_values.mean()),
            "p99": float(np.percentile(positive_values, 99)),
            "maximum": float(positive_values.max()),
        },
        "teacher_cache_sha256": sha256(teacher_path),
        "normal_patch_cache_retained": False,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    close_memmap(normal_raw)
    close_memmap(normal_patches)
    del normal_raw, normal_patches
    gc.collect()
    normal_raw_path.unlink()
    normal_patch_path.unlink()
    return teachers, metadata


class FrozenDecoderCacheDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        caches: dict[str, np.memmap],
        teachers: np.memmap,
        *,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.caches = caches
        self.teachers = teachers
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = np.asarray(self.caches["tokens"][index], dtype=np.float32)
        guidance = np.asarray(self.caches["guidance"][index], dtype=np.float32)
        teacher = np.asarray(self.teachers[index], dtype=np.float32)
        validity = np.asarray(self.caches["validity"][index], dtype=np.float32)
        if self.augment and random.random() < 0.5:
            tokens = np.flip(tokens, axis=1)
            guidance = np.flip(guidance, axis=2)
            teacher = np.flip(teacher, axis=1)
            validity = np.flip(validity, axis=1)
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
) -> tuple[RadDinoSpatialDecoder, list[dict[str, object]]]:
    config = AffinityDecoderConfig()
    decoder = RadDinoSpatialDecoder(config).to(device)
    dataset = FrozenDecoderCacheDataset(rows, caches, teachers, augment=True)
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
            "pseudo_loss": [],
            "affinity_loss": [],
        }
        correct = 0
        image_count = 0
        for tokens, guidance, teacher, valid, labels in loader:
            tokens = tokens.to(device, non_blocking=True)
            guidance = guidance.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, affinity_features = decoder(tokens, guidance)
            weights, learned, pair_validity = local_affinity(
                affinity_features,
                tokens,
                radius=config.affinity_radius,
                temperature=config.affinity_temperature,
                frozen_similarity_power=config.frozen_similarity_power,
            )
            refined_teacher = propagate_seed_preserving(
                teacher,
                weights,
                radius=config.affinity_radius,
                steps=config.propagation_steps,
                residual=config.propagation_residual,
            )
            loss_image, pooled = image_level_loss(
                logits,
                labels,
                alpha=config.smoothmax_alpha,
            )
            loss_pseudo = masked_pseudo_loss(
                logits,
                refined_teacher,
                labels,
                foreground_quantile=config.foreground_quantile,
                background_quantile=config.background_quantile,
                valid_region=valid,
            )
            loss_affinity = affinity_pair_loss(
                learned,
                pair_validity,
                teacher,
                labels,
                radius=config.affinity_radius,
                foreground_quantile=config.foreground_quantile,
                background_quantile=config.background_quantile,
                valid_region=valid,
            )
            loss = (
                loss_image
                + args.pseudo_loss_weight * loss_pseudo
                + args.affinity_loss_weight * loss_affinity
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()
            for name, value in (
                ("total_loss", loss),
                ("image_loss", loss_image),
                ("pseudo_loss", loss_pseudo),
                ("affinity_loss", loss_affinity),
            ):
                totals[name].append(float(value.detach().cpu()))
            correct += int(((pooled >= 0.5) == (labels >= 0.5)).sum())
            image_count += int(labels.numel())
        record = {
            "epoch": epoch,
            **{name: float(np.mean(values)) for name, values in totals.items()},
            "training_accuracy": float(correct / max(image_count, 1)),
        }
        history.append(record)
        print(f"affinity decoder epoch {epoch}/{args.epochs}: {record}", flush=True)
    return decoder.eval(), history


def save_checkpoint(
    decoder: RadDinoSpatialDecoder,
    history: list[dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    checkpoint = args.output_dir / "rad_dino_affinity_decoder.pt"
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "config": decoder.config.__dict__,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pseudo_loss_weight": args.pseudo_loss_weight,
            "affinity_loss_weight": args.affinity_loss_weight,
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
    decoder: RadDinoSpatialDecoder,
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
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        raw, normalized, projection = _raw_and_normalized_square(
            image, input_size=args.input_size
        )
        _global, patch_tokens = extract_frozen_features(
            encoder,
            normalized,
            grid_size=grid_size,
            device=device,
        )
        tokens = torch.from_numpy(
            patch_tokens.astype(np.float32)
        )[None].to(device)
        guidance = make_guidance(raw[None], output_size=guidance_size).to(device)
        with torch.inference_mode():
            logits, _features = decoder(tokens, guidance)
            square_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
        output_map = project_square_map(
            square_map.astype(np.float32),
            projection,
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
            print(f"validation decoder maps: {index + 1}/{len(rows)}", flush=True)
    manifest = prediction_dir / "prediction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return sha256(manifest)


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return (
        2.0
        * float(np.logical_and(prediction, target).sum())
        / float(denominator)
    )


def _subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def evaluate_frozen_predictions(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_dir = args.output_dir / "predictions"
    manifest_path = prediction_dir / "prediction_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if (
        len(manifest) != 371
        or len({row["image_id"] for row in manifest}) != 371
        or len({row["map_path"] for row in manifest}) != 371
    ):
        raise RuntimeError("Prediction manifest must contain 371 unique maps")
    expected_map_paths = {row["map_path"] for row in manifest}
    observed_map_paths = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "maps").glob("*.npy")
    }
    if observed_map_paths != expected_map_paths:
        raise RuntimeError("Prediction directory and frozen manifest differ")
    for row in manifest:
        path = prediction_dir / row["map_path"]
        if sha256(path) != row["map_sha256"]:
            raise RuntimeError(f"Prediction map hash mismatch: {row['image_id']}")

    from datasets.btxrd import BTXRDSegmentationDataset

    dataset = BTXRDSegmentationDataset(
        root=args.dataset_root,
        split="val",
        image_size=args.output_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, mask, name = dataset[index]
        gt_by_name[str(name)] = mask[0].numpy() > 0.5
    if set(gt_by_name) != {row["image_id"] for row in manifest}:
        raise RuntimeError("Frozen prediction and validation GT cohorts differ")
    evaluated: list[dict[str, object]] = []
    for row in manifest:
        if row["tumor"] != "1":
            continue
        values = np.load(
            prediction_dir / row["map_path"], allow_pickle=False
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        item: dict[str, object] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": float(target.mean()),
            "size_group": _subgroup(float(target.mean())),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(
                target.reshape(-1)[int(np.argmax(flat_values))]
            ),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1.0e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = _dice(
                values >= np.percentile(values, percentile),
                target,
            )
        evaluated.append(item)
    if len(evaluated) != 184:
        raise RuntimeError(f"Expected 184 tumor evaluations, got {len(evaluated)}")
    counts = {
        name: sum(row["size_group"] == name for row in evaluated)
        for name in ("small", "medium", "large")
    }
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"Subgroup contract drift: {counts}")
    evaluation_dir = prediction_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    per_image = evaluation_dir / "per_image.csv"
    with per_image.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    metrics = [
        "pixel_ap",
        "pixel_auroc",
        "argmax_hit",
        "saliency_mass_in_gt",
        "dice_p90",
        "dice_p95",
        "dice_p97",
        "dice_p99",
    ]
    image_labels = np.asarray(
        [int(row["tumor"]) for row in manifest], dtype=np.uint8
    )
    image_scores = np.asarray(
        [float(row["raw_p99"]) for row in manifest], dtype=np.float64
    )
    summary: dict[str, object] = {
        "arm": "decoder_prediction",
        "cohort": {"validation": 371, "tumor": 184, **counts},
        "image_level_auroc_from_raw_p99": float(
            roc_auc_score(image_labels, image_scores)
        ),
        "tumor_localization": {},
    }
    for name in ("overall", "small", "medium", "large"):
        selected = [
            row
            for row in evaluated
            if name == "overall" or row["size_group"] == name
        ]
        summary["tumor_localization"][name] = {
            "n": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in metrics
            },
        }
    summary.update(
        {
            "prediction_manifest_sha256": sha256(manifest_path),
            "per_image_sha256": sha256(per_image),
            "validation_gt_read_only_after_prediction_freeze": True,
            "complete_misses_included": True,
            "consumer_trained": False,
            "test_evaluated": False,
        }
    )
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return evaluated, summary


def compare_to_frozen_nominal(
    decoder_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    if sha256(args.baseline_per_image) != args.expected_baseline_per_image_sha256:
        raise RuntimeError("Frozen nominal baseline per-image hash mismatch")
    with args.baseline_per_image.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline = {str(row["image_id"]): row for row in baseline_rows}
    decoder = {str(row["image_id"]): row for row in decoder_rows}
    if baseline.keys() != decoder.keys() or len(decoder) != 184:
        raise RuntimeError("Decoder and nominal baseline cohorts differ")
    results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        results[metric] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in decoder.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            statistics = paired_group_bootstrap(
                [
                    (
                        str(decoder[name]["group_id"]),
                        float(decoder[name][metric])
                        - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                replicates=10_000,
                seed=20260727 + metric_index * 10 + len(stratum),
            )
            statistics["delta_decoder_minus_nominal"] = statistics.pop(
                "delta_multiscale_minus_single_scale"
            )
            results[metric][stratum] = statistics
    return {
        "comparison": "affinity decoder minus frozen nominal-memory single-scale",
        "baseline_per_image_sha256": args.expected_baseline_per_image_sha256,
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed_family": 20260727,
        "metrics": results,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def apply_gate(summary: dict[str, object]) -> dict[str, object]:
    localization = summary["tumor_localization"]
    observed = {
        "image_level_auroc_from_raw_p99": summary[
            "image_level_auroc_from_raw_p99"
        ],
        "overall_pixel_auroc": localization["overall"]["pixel_auroc"],
        "small_pixel_auroc": localization["small"]["pixel_auroc"],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p97": localization["small"]["dice_p97"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    checks = {
        name: {
            "observed": float(observed[name]),
            "minimum": float(minimum),
            "pass": bool(float(observed[name]) >= float(minimum)),
        }
        for name, minimum in GATE_THRESHOLDS.items()
    }
    passed = all(check["pass"] for check in checks.values())
    return {
        "gate_id": "rad_dino_affinity_decoder_prediction_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "all_checks_required": True,
        "checks": checks,
        "on_pass": (
            "mechanism may be retained; pseudo-mask selection or consumer "
            "training still requires a separate predeclared protocol"
        ),
        "on_fail": "reject this configuration without threshold fitting",
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Affinity decoder probe requires a Kaggle GPU")
    if (
        args.input_size != 448
        or args.output_size != 320
        or args.epochs != 12
        or args.batch_size != 8
        or args.seed != 42
    ):
        raise ValueError("Frozen geometry/budget is 448/320, 12 epochs, batch 8, seed 42")
    if (
        args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.top_k != 8
        or args.teacher_spatial_radius != 2
        or args.pseudo_loss_weight != 1.0
        or args.affinity_loss_weight != 0.1
    ):
        raise ValueError("Affinity decoder hyperparameters differ from protocol")
    if args.output_dir.exists():
        raise FileExistsError("output-dir must not exist")
    if args.scratch_dir.exists():
        raise FileExistsError("scratch-dir must not exist")
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
        raise RuntimeError("Image stems must be unique within each frozen split")

    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers must be {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    device = torch.device("cuda")
    encoder = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True
    ).eval().to(device)
    if int(encoder.config.patch_size) != 14 or int(encoder.config.hidden_size) != 768:
        raise RuntimeError("RAD-DINO snapshot architecture differs from protocol")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    grid_size = args.input_size // 14
    guidance_size = grid_size * AffinityDecoderConfig.decoder_scale

    caches, normal_indices = build_training_feature_cache(
        encoder,
        train_rows,
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
        device=device,
    )
    teacher_metadata_path = args.output_dir / "teacher_metadata.json"
    teacher_metadata_path.write_text(
        json.dumps(teacher_metadata, indent=2) + "\n", encoding="utf-8"
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
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    # Validation predictions are complete and hashed before any GT loader exists.
    prediction_manifest_sha = write_validation_maps(
        encoder,
        decoder,
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
        "prediction_manifest_sha256": prediction_manifest_sha,
        "validation_predictions": 371,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )

    # Only after the freeze above may the evaluator import validation masks.
    evaluated, summary = evaluate_frozen_predictions(args)
    comparison = compare_to_frozen_nominal(evaluated, args)
    comparison_path = args.output_dir / "paired_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    gate = apply_gate(summary)
    gate_path = args.output_dir / "gate_decision.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    source_root = Path(__file__).resolve().parent
    run_manifest = {
        "run_id": args.output_dir.name,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": snapshot,
        "source_files": {
            "runner": sha256(Path(__file__).resolve()),
            "affinity_decoder": sha256(
                source_root / "models" / "rad_dino_affinity_decoder.py"
            ),
            "nominal_memory": sha256(
                source_root / "models" / "nominal_patch_memory.py"
            ),
            "mae_reconstruction": sha256(
                source_root / "models" / "mae_reconstruction.py"
            ),
        },
        "checkpoint_sha256": checkpoint_sha,
        "prediction_freeze_sha256": sha256(freeze_path),
        "paired_comparison_sha256": sha256(comparison_path),
        "gate_decision_sha256": sha256(gate_path),
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
        "validation_gt_read_only_after_prediction_freeze": True,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
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
                "summary": summary,
                "comparison": comparison,
                "gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
