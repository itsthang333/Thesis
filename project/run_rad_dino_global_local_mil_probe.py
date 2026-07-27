"""Train a high-resolution local MIL branch from image labels only.

The runner freezes an audited global RAD-DINO decoder, uses its validation
maps and train-positive predictions only for ROI proposal, and trains one
shared local spatial decoder from bags of crops.  It never imports the
segmentation dataset and never opens validation annotations.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    save_float_map,
    sha256_file,
    verify_model_snapshot,
)
from models.mae_reconstruction import (
    project_square_map,
    radiograph_foreground_mask,
)
from models.nominal_patch_memory import make_seeded_random_projection
from models.rad_dino_global_local_mil import (
    GlobalLocalMILConfig,
    RadDinoGlobalLocalMILDecoder,
    confidence_gated_rank_fusion,
    greedy_saliency_windows,
    local_mil_loss,
    random_diverse_windows,
    stitch_local_maps,
    top_fraction_pool,
)
from models.rad_dino_multilayer_soft_region_decoder import (
    MultiLayerSoftRegionConfig,
    RadDinoMultiLayerSoftRegionDecoder,
    make_guidance,
)
from run_rad_dino_multilayer_soft_region_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    _raw_and_normalized_square,
    content_validity,
    extract_projected_features,
    seed_everything,
    seed_worker,
)


GLOBAL_CHECKPOINT_NAME = "rad_dino_multilayer_soft_region_decoder.pt"
EXPECTED_GLOBAL_CONFIG = {
    "input_dim": 128,
    "layer_count": 3,
    "hidden_dim": 128,
    "affinity_dim": 64,
    "decoder_scale": 2,
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
    parser.add_argument("--global-run-root", type=Path, required=True)
    parser.add_argument("--expected-global-checkpoint-sha256", required=True)
    parser.add_argument("--expected-global-freeze-sha256", required=True)
    parser.add_argument(
        "--expected-global-prediction-manifest-sha256", required=True
    )
    parser.add_argument("--expected-global-per-image-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--local-projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return sha256_file(path)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close_memmap(values: np.ndarray) -> None:
    if isinstance(values, np.memmap):
        values.flush()
        mmap_handle = getattr(values, "_mmap", None)
        if mmap_handle is not None and not mmap_handle.closed:
            mmap_handle.close()


def validate_global_run(
    root: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, dict[str, str]]]:
    checkpoint = root / GLOBAL_CHECKPOINT_NAME
    freeze_path = root / "prediction_freeze.json"
    manifest_path = root / "predictions/prediction_manifest.csv"
    per_image_path = root / "evaluation/per_image.csv"
    expected = {
        checkpoint: args.expected_global_checkpoint_sha256,
        freeze_path: args.expected_global_freeze_sha256,
        manifest_path: args.expected_global_prediction_manifest_sha256,
        per_image_path: args.expected_global_per_image_sha256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Frozen global artifact mismatch: {path}")
    freeze = read_json(freeze_path)
    if (
        freeze["validation_predictions"] != 371
        or freeze["checkpoint_sha256"]
        != args.expected_global_checkpoint_sha256
        or freeze["prediction_manifest_sha256"]
        != args.expected_global_prediction_manifest_sha256
        or freeze["validation_gt_read"] is not False
        or freeze["consumer_trained"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Frozen global prediction contract mismatch")
    records = read_csv(manifest_path)
    if len(records) != 371 or len({row["image_id"] for row in records}) != 371:
        raise ValueError("Frozen global prediction cohort mismatch")
    indexed = {row["image_id"]: row for row in records}
    for row in records:
        path = root / "predictions" / row["map_path"]
        if not path.is_file() or sha256(path) != row["map_sha256"]:
            raise ValueError(f"Frozen global map mismatch: {row['image_id']}")
    return checkpoint, indexed


def load_global_decoder(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> RadDinoMultiLayerSoftRegionDecoder:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Frozen global checkpoint has no decoder config")
    for name, expected in EXPECTED_GLOBAL_CONFIG.items():
        if raw_config.get(name) != expected:
            raise ValueError(f"Frozen global config mismatch: {name}")
    decoder = RadDinoMultiLayerSoftRegionDecoder(
        MultiLayerSoftRegionConfig(**raw_config)
    )
    decoder.load_state_dict(checkpoint["state_dict"], strict=True)
    decoder.eval().to(device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def predict_global_map(
    image: Image.Image,
    encoder: nn.Module,
    decoder: RadDinoMultiLayerSoftRegionDecoder,
    projection: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> np.ndarray:
    raw, normalized, square_projection = _raw_and_normalized_square(
        image,
        input_size=input_size,
    )
    _global, projected = extract_projected_features(
        encoder,
        normalized,
        projection,
        grid_size=grid_size,
        device=device,
    )
    tokens = torch.from_numpy(projected.astype(np.float32))[None].to(device)
    guidance = make_guidance(raw[None], output_size=guidance_size).to(device)
    with torch.inference_mode():
        logits, _features, _weights = decoder(tokens, guidance)
        flipped, _features, _weights = decoder(
            tokens.flip(3), guidance.flip(-1)
        )
        square_map = (
            0.5
            * (torch.sigmoid(logits) + torch.sigmoid(flipped).flip(-1))
        )[0, 0].cpu().numpy()
    output = project_square_map(
        square_map.astype(np.float32),
        square_projection,
        output_height=output_size,
        output_width=output_size,
    )
    foreground = radiograph_foreground_mask(
        image,
        output_height=output_size,
        output_width=output_size,
    )
    if not foreground.any():
        raise RuntimeError("Radiograph foreground is empty")
    output[~foreground] = 0.0
    return np.clip(output, 0.0, 1.0)


def crop_from_output_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    output_size: int,
) -> Image.Image:
    x0, y0, x1, y1 = box
    width, height = image.size
    left = int(np.floor(x0 * width / output_size))
    top = int(np.floor(y0 * height / output_size))
    right = int(np.ceil(x1 * width / output_size))
    bottom = int(np.ceil(y1 * height / output_size))
    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    right = min(max(right, left + 1), width)
    bottom = min(max(bottom, top + 1), height)
    return image.crop((left, top, right, bottom))


def local_patch_feature_batch(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    encoder: nn.Module,
    projection: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not boxes:
        raise ValueError("Local feature batch requires at least one box")
    raw_rows: list[torch.Tensor] = []
    normalized_rows: list[torch.Tensor] = []
    valid_rows: list[np.ndarray] = []
    for box in boxes:
        crop = crop_from_output_box(image, box, output_size=output_size)
        raw, normalized, square_projection = _raw_and_normalized_square(
            crop,
            input_size=input_size,
        )
        raw_rows.append(raw)
        normalized_rows.append(normalized)
        valid_tokens = content_validity(square_projection, grid_size=grid_size)
        valid_rows.append(
            np.repeat(
                np.repeat(
                    valid_tokens,
                    guidance_size // grid_size,
                    axis=0,
                ),
                guidance_size // grid_size,
                axis=1,
            )
        )
    pixels = torch.stack(normalized_rows).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        output = encoder(pixel_values=pixels, output_hidden_states=True)
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != 13:
        raise RuntimeError("RAD-DINO must expose embedding plus 12 hidden states")
    expected = grid_size * grid_size + 1
    selected: list[torch.Tensor] = []
    for layer_index in SELECTED_HIDDEN_LAYERS:
        hidden = hidden_states[layer_index]
        if hidden.shape != (len(boxes), expected, 768):
            raise RuntimeError(
                f"Unexpected local layer-{layer_index} shape {tuple(hidden.shape)}"
            )
        patches = hidden[:, 1:].reshape(
            len(boxes), grid_size, grid_size, 768
        ).float()
        selected.append(F.normalize(patches @ projection, dim=-1))
    tokens = torch.stack(selected, dim=1).cpu().numpy().astype(np.float16)
    guidance = make_guidance(
        torch.stack(raw_rows), output_size=guidance_size
    ).numpy().astype(np.float32)
    return tokens, guidance, np.stack(valid_rows)


def stable_image_seed(image_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def proposal_boxes(
    row: dict[str, str],
    *,
    global_map: np.ndarray | None,
    config: GlobalLocalMILConfig,
    output_size: int,
    seed: int,
) -> tuple[list[tuple[int, int, int, int]], str]:
    if row["tumor"] == "1":
        if global_map is None:
            raise ValueError("Positive training image requires global saliency")
        return (
            greedy_saliency_windows(
                global_map,
                window_size=config.proposal_size,
                count=config.train_patches,
                stride=config.proposal_stride,
                iou_limit=config.proposal_iou_limit,
            ),
            "frozen_global_top_mass",
        )
    return (
        random_diverse_windows(
            output_shape=(output_size, output_size),
            window_size=config.proposal_size,
            count=config.train_patches,
            stride=config.proposal_stride,
            iou_limit=config.proposal_iou_limit,
            seed=stable_image_seed(row["image_id"], seed),
        ),
        "seeded_random_negative",
    )


def open_local_cache(
    scratch: Path,
    *,
    images: int,
    config: GlobalLocalMILConfig,
    grid_size: int,
    guidance_size: int,
) -> dict[str, np.memmap]:
    return {
        "tokens": np.lib.format.open_memmap(
            scratch / "local_tokens.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                images,
                config.train_patches,
                len(SELECTED_HIDDEN_LAYERS),
                grid_size,
                grid_size,
                config.input_dim,
            ),
        ),
        "guidance": np.lib.format.open_memmap(
            scratch / "local_guidance.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                images,
                config.train_patches,
                3,
                guidance_size,
                guidance_size,
            ),
        ),
        "valid": np.lib.format.open_memmap(
            scratch / "local_valid.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(
                images,
                config.train_patches,
                guidance_size,
                guidance_size,
            ),
        ),
    }


def build_local_cache(
    encoder: nn.Module,
    global_decoder: RadDinoMultiLayerSoftRegionDecoder,
    train_rows: list[dict[str, str]],
    global_projection: torch.Tensor,
    local_projection: torch.Tensor,
    config: GlobalLocalMILConfig,
    args: argparse.Namespace,
    *,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> tuple[dict[str, np.memmap], Path]:
    caches = open_local_cache(
        args.scratch_dir,
        images=len(train_rows),
        config=config,
        grid_size=grid_size,
        guidance_size=guidance_size,
    )
    proposal_path = args.output_dir / "training_proposals.csv"
    fields = ["image_id", "group_id", "tumor", "source", "boxes"]
    with proposal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(train_rows):
            image = Image.open(locate_verified_image(args.dataset_root, row)).convert(
                "RGB"
            )
            global_map = None
            if row["tumor"] == "1":
                global_map = predict_global_map(
                    image,
                    encoder,
                    global_decoder,
                    global_projection,
                    input_size=args.input_size,
                    output_size=args.output_size,
                    grid_size=grid_size,
                    guidance_size=guidance_size,
                    device=device,
                )
            boxes, source = proposal_boxes(
                row,
                global_map=global_map,
                config=config,
                output_size=args.output_size,
                seed=args.seed,
            )
            tokens, guidance, valid = local_patch_feature_batch(
                image,
                boxes,
                encoder,
                local_projection,
                input_size=args.input_size,
                output_size=args.output_size,
                grid_size=grid_size,
                guidance_size=guidance_size,
                device=device,
            )
            caches["tokens"][index] = tokens
            caches["guidance"][index] = guidance
            caches["valid"][index] = valid.astype(np.uint8)
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "source": source,
                    "boxes": json.dumps(boxes, separators=(",", ":")),
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(train_rows):
                print(
                    f"global-local feature cache: {index + 1}/{len(train_rows)}",
                    flush=True,
                )
    for values in caches.values():
        values.flush()
    return caches, proposal_path


class LocalCacheDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        caches: dict[str, np.memmap],
    ) -> None:
        self.rows = rows
        self.caches = caches

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.caches["tokens"][index], np.float32)),
            torch.from_numpy(np.array(self.caches["guidance"][index], np.float32)),
            torch.from_numpy(np.array(self.caches["valid"][index], bool)),
            torch.tensor(float(self.rows[index]["tumor"]), dtype=torch.float32),
        )


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    epoch: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_local_decoder(
    train_rows: list[dict[str, str]],
    caches: dict[str, np.memmap],
    config: GlobalLocalMILConfig,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[RadDinoGlobalLocalMILDecoder, list[dict[str, float]]]:
    decoder = RadDinoGlobalLocalMILDecoder(config).to(device)
    dataset = LocalCacheDataset(train_rows, caches)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        tracked = {
            "total_loss": [],
            "mil_loss": [],
            "image_bce": [],
            "negative_dense": [],
            "positive_sparsity": [],
            "flip_consistency": [],
        }
        for tokens, guidance, valid, labels in make_loader(
            dataset, args, shuffle=True, epoch=epoch
        ):
            tokens = tokens.to(device, non_blocking=True)
            guidance = guidance.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _features = decoder(tokens, guidance)
            mil, parts = local_mil_loss(
                logits,
                valid,
                labels,
                top_fraction=config.top_fraction,
                negative_dense_weight=config.negative_dense_weight,
                positive_sparsity_weight=config.positive_sparsity_weight,
            )
            flipped_logits, _features = decoder(
                tokens.flip(4), guidance.flip(-1)
            )
            flip_error = (
                torch.sigmoid(logits)
                - torch.sigmoid(flipped_logits).flip(-1)
            ).square()
            flip = flip_error[valid.unsqueeze(2)].mean()
            loss = mil + config.flip_consistency_weight * flip
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()
            values = {
                "total_loss": loss,
                "mil_loss": mil,
                "image_bce": parts["image_bce"],
                "negative_dense": parts["negative_dense"],
                "positive_sparsity": parts["positive_sparsity"],
                "flip_consistency": flip,
            }
            for name, value in values.items():
                tracked[name].append(float(value.detach().cpu()))
        record = {"epoch": float(epoch)}
        record.update({name: float(np.mean(values)) for name, values in tracked.items()})
        if not all(np.isfinite(value) for value in record.values()):
            raise RuntimeError("Non-finite local training history")
        history.append(record)
        print(f"global-local epoch {epoch}/{args.epochs}: {record}", flush=True)
    return decoder.eval(), history


def calibrate_normal_confidence(
    decoder: RadDinoGlobalLocalMILDecoder,
    train_rows: list[dict[str, str]],
    caches: dict[str, np.memmap],
    config: GlobalLocalMILConfig,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, object]:
    normal_indices = [
        index for index, row in enumerate(train_rows) if row["tumor"] == "0"
    ]
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(normal_indices), args.batch_size):
            indices = normal_indices[start : start + args.batch_size]
            tokens = torch.from_numpy(
                np.array(caches["tokens"][indices], np.float32)
            ).to(device)
            guidance = torch.from_numpy(
                np.array(caches["guidance"][indices], np.float32)
            ).to(device)
            valid = torch.from_numpy(
                np.array(caches["valid"][indices], bool)
            ).to(device)
            logits, _features = decoder(tokens, guidance)
            pooled = top_fraction_pool(
                logits, valid, fraction=config.top_fraction
            )
            scores.extend(torch.sigmoid(pooled).cpu().tolist())
    if len(scores) != 1493 or not np.isfinite(scores).all():
        raise RuntimeError("Train-normal confidence calibration mismatch")
    values = np.asarray(scores, dtype=np.float64)
    return {
        "normal_images": len(scores),
        "score_minimum": float(values.min()),
        "score_median": float(np.median(values)),
        "score_p95": float(np.percentile(values, 95)),
        "score_p99": float(np.percentile(values, 99)),
        "score_maximum": float(values.max()),
        "validation_gt_read": False,
        "test_evaluated": False,
    }


def predict_local_bag(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    encoder: nn.Module,
    decoder: RadDinoGlobalLocalMILDecoder,
    projection: torch.Tensor,
    config: GlobalLocalMILConfig,
    args: argparse.Namespace,
    *,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    token_rows, guidance_rows, valid_rows = local_patch_feature_batch(
        image,
        boxes,
        encoder,
        projection,
        input_size=args.input_size,
        output_size=args.output_size,
        grid_size=grid_size,
        guidance_size=guidance_size,
        device=device,
    )
    tokens = torch.from_numpy(token_rows.astype(np.float32))[None].to(device)
    guidance = torch.from_numpy(guidance_rows.astype(np.float32))[None].to(device)
    valid = torch.from_numpy(valid_rows.astype(bool))[None].to(device)
    with torch.inference_mode():
        logits, _features = decoder(tokens, guidance)
        flipped, _features = decoder(tokens.flip(4), guidance.flip(-1))
        probabilities = 0.5 * (
            torch.sigmoid(logits) + torch.sigmoid(flipped).flip(-1)
        )
        probabilities = probabilities * valid.unsqueeze(2)
        confidence = torch.sigmoid(
            top_fraction_pool(logits, valid, fraction=config.top_fraction)
        ).item()
    patch_maps = probabilities[0, :, 0].cpu().numpy().astype(np.float32)
    local_map, coverage = stitch_local_maps(
        patch_maps,
        boxes,
        output_shape=(args.output_size, args.output_size),
    )
    return local_map, coverage, float(confidence)


def write_validation_predictions(
    encoder: nn.Module,
    decoder: RadDinoGlobalLocalMILDecoder,
    local_projection: torch.Tensor,
    val_rows: list[dict[str, str]],
    global_rows: dict[str, dict[str, str]],
    calibration: dict[str, object],
    config: GlobalLocalMILConfig,
    args: argparse.Namespace,
    *,
    grid_size: int,
    guidance_size: int,
    device: torch.device,
) -> str:
    prediction_dir = args.output_dir / "predictions"
    fused_dir = prediction_dir / "maps"
    local_dir = prediction_dir / "local_maps"
    fused_dir.mkdir(parents=True, exist_ok=False)
    local_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for index, row in enumerate(val_rows):
        global_row = global_rows[row["image_id"]]
        global_path = args.global_run_root / "predictions" / global_row["map_path"]
        global_map = np.load(global_path, allow_pickle=False).astype(np.float32)
        boxes = greedy_saliency_windows(
            global_map,
            window_size=config.proposal_size,
            count=config.inference_patches,
            stride=config.proposal_stride,
            iou_limit=config.proposal_iou_limit,
        )
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        local_map, coverage, local_confidence = predict_local_bag(
            image,
            boxes,
            encoder,
            decoder,
            local_projection,
            config,
            args,
            grid_size=grid_size,
            guidance_size=guidance_size,
            device=device,
        )
        foreground = radiograph_foreground_mask(
            image,
            output_height=args.output_size,
            output_width=args.output_size,
        )
        local_map[~foreground] = 0.0
        coverage &= foreground
        fused_map, gate = confidence_gated_rank_fusion(
            global_map,
            local_map,
            coverage,
            local_confidence=local_confidence,
            normal_confidence_p99=float(calibration["score_p99"]),
            keep_fraction=config.local_keep_fraction,
            residual_weight=config.residual_weight,
            temperature=config.confidence_temperature,
        )
        fused_map[~foreground] = 0.0
        stem = Path(row["image_id"]).stem
        local_relative = Path("local_maps") / f"{stem}.npy"
        fused_relative = Path("maps") / f"{stem}.npy"
        save_float_map(prediction_dir / local_relative, local_map)
        save_float_map(prediction_dir / fused_relative, fused_map)
        records.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "global_map_path": global_row["map_path"],
                "global_map_sha256": global_row["map_sha256"],
                "local_map_path": local_relative.as_posix(),
                "local_map_sha256": sha256(prediction_dir / local_relative),
                "map_path": fused_relative.as_posix(),
                "map_sha256": sha256(prediction_dir / fused_relative),
                "proposal_boxes": json.dumps(boxes, separators=(",", ":")),
                "local_confidence": local_confidence,
                "fusion_gate": gate,
                "raw_mean": float(fused_map.mean()),
                "raw_p99": float(np.percentile(fused_map[foreground], 99)),
                "raw_max": float(fused_map.max()),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(val_rows):
            print(
                f"global-local validation maps: {index + 1}/{len(val_rows)}",
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
    config = GlobalLocalMILConfig(input_dim=args.local_projection_dim)
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Global-local MIL probe requires a Kaggle GPU")
    if (
        args.input_size != 448
        or args.output_size != 320
        or args.epochs != 12
        or args.batch_size != 4
        or args.local_projection_dim != 64
        or args.projection_seed != 42
        or args.seed != 42
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
    ):
        raise ValueError("Global-local hyperparameters differ from protocol")
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
    global_checkpoint, global_rows = validate_global_run(
        args.global_run_root, args
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
    if set(global_rows) != {row["image_id"] for row in val_rows}:
        raise RuntimeError("Frozen global and validation cohorts differ")
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
    if (
        int(encoder.config.patch_size) != 14
        or int(encoder.config.hidden_size) != 768
        or int(encoder.config.num_hidden_layers) != 12
    ):
        raise RuntimeError("RAD-DINO snapshot architecture differs from protocol")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    global_decoder = load_global_decoder(global_checkpoint, device=device)
    grid_size = args.input_size // 14
    guidance_size = grid_size * config.decoder_scale
    global_projection = torch.from_numpy(
        make_seeded_random_projection(
            input_dim=768,
            output_dim=128,
            seed=args.projection_seed,
        )
    ).to(device)
    local_projection_np = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.local_projection_dim,
        seed=args.projection_seed,
    )
    local_projection = torch.from_numpy(local_projection_np).to(device)
    caches, proposal_path = build_local_cache(
        encoder,
        global_decoder,
        train_rows,
        global_projection,
        local_projection,
        config,
        args,
        grid_size=grid_size,
        guidance_size=guidance_size,
        device=device,
    )
    del global_decoder, global_projection
    torch.cuda.empty_cache()
    decoder, history = train_local_decoder(
        train_rows, caches, config, args, device=device
    )
    calibration = calibrate_normal_confidence(
        decoder,
        train_rows,
        caches,
        config,
        args,
        device=device,
    )
    calibration_path = args.output_dir / "normal_confidence_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_path = args.output_dir / "rad_dino_global_local_mil_decoder.pt"
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "config": config.__dict__,
            "normal_confidence_calibration": calibration,
            "selected_hidden_layers": SELECTED_HIDDEN_LAYERS,
            "local_projection_sha256": hashlib.sha256(
                local_projection_np.tobytes(order="C")
            ).hexdigest(),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "global_checkpoint_sha256": args.expected_global_checkpoint_sha256,
        },
        checkpoint_path,
    )
    prediction_manifest_sha = write_validation_predictions(
        encoder,
        decoder,
        local_projection,
        val_rows,
        global_rows,
        calibration,
        config,
        args,
        grid_size=grid_size,
        guidance_size=guidance_size,
        device=device,
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "global_checkpoint_sha256": args.expected_global_checkpoint_sha256,
        "global_freeze_sha256": args.expected_global_freeze_sha256,
        "global_prediction_manifest_sha256": (
            args.expected_global_prediction_manifest_sha256
        ),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_proposals_sha256": sha256(proposal_path),
        "training_history_sha256": sha256(history_path),
        "normal_confidence_calibration_sha256": sha256(calibration_path),
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
    source_root = Path(__file__).resolve().parent
    source_files = [
        source_root / "run_rad_dino_global_local_mil_probe.py",
        source_root / "models/rad_dino_global_local_mil.py",
        source_root / "run_rad_dino_multilayer_soft_region_probe.py",
        source_root / "models/rad_dino_multilayer_soft_region_decoder.py",
        source_root / "models/mae_reconstruction.py",
        source_root / "models/nominal_patch_memory.py",
        source_root / "mae_reconstruction_io.py",
    ]
    finished = datetime.now(timezone.utc)
    run_manifest = {
        "run_id": args.output_dir.name,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": snapshot,
        "global_bindings": {
            "checkpoint_sha256": args.expected_global_checkpoint_sha256,
            "freeze_sha256": args.expected_global_freeze_sha256,
            "prediction_manifest_sha256": (
                args.expected_global_prediction_manifest_sha256
            ),
            "per_image_sha256": args.expected_global_per_image_sha256,
        },
        "config": config.__dict__,
        "cohort": {"train": 2981, "validation": 371},
        "source_hashes": {
            path.relative_to(source_root).as_posix(): sha256(path)
            for path in source_files
        },
        "output_hashes": {
            "checkpoint": sha256(checkpoint_path),
            "training_proposals": sha256(proposal_path),
            "training_history": sha256(history_path),
            "normal_confidence_calibration": sha256(calibration_path),
            "prediction_manifest": prediction_manifest_sha,
            "prediction_freeze": sha256(freeze_path),
        },
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for values in caches.values():
        close_memmap(values)
    del caches
    gc.collect()
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
