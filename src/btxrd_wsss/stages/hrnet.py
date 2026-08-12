from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from btxrd_wsss.config import PipelineConfig
from btxrd_wsss.data.images import load_native_grayscale, make_hrnet_channels
from btxrd_wsss.data.manifest import ImageRecord, read_manifest
from btxrd_wsss.data.tiling import blend_tile_maps, extract_tiles
from btxrd_wsss.data.views import (
    pad_to_multiple,
    resize_long_side,
    resize_square,
    sample_native_tiles,
)
from btxrd_wsss.io import atomic_json, seed_everything
from btxrd_wsss.models.hrnet_mil import HRNetDenseMIL, hrnet_mil_loss, hrnet_tile_bag_loss


def _tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(make_hrnet_channels(image)).unsqueeze(0).to(device)


def _tile_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    channels = np.stack([make_hrnet_channels(image) for image in images])
    return torch.from_numpy(channels).to(device)


def _full_view(image: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, tuple[int, int]]:
    resized, _ = resize_long_side(image, config.hrnet.full_long_side)
    return pad_to_multiple(resized, config.data.pad_multiple)


def _native_reference(
    tumor_map: torch.Tensor,
    *,
    padded_shape: tuple[int, int],
    valid_shape: tuple[int, int],
    native_shape: tuple[int, int],
) -> torch.Tensor:
    pixels = F.interpolate(
        tumor_map[:, None].float(), padded_shape, mode="bilinear", align_corners=False
    )[:, 0]
    pixels = pixels[..., : valid_shape[0], : valid_shape[1]]
    return F.interpolate(pixels[:, None], native_shape, mode="bilinear", align_corners=False)[:, 0]


def _multi_hot(record: ImageRecord, classes: int, device: torch.device) -> torch.Tensor:
    target = torch.zeros(1, classes, device=device)
    for index in record.class_indices:
        target[0, index] = 1
    if record.is_tumor:
        target[0, 0] = 0
    return target


def _model(config: PipelineConfig, device: torch.device, *, pretrained: bool) -> HRNetDenseMIL:
    return HRNetDenseMIL(
        backbone_name=config.hrnet.backbone,
        pretrained=pretrained,
        classes=config.hrnet.output_classes,
        dense_channels=config.hrnet.dense_channels,
        dropout=config.hrnet.dropout,
        topk_fractions=tuple(config.hrnet.topk_fractions),
    ).to(device)


def _full_reference_crop(
    full_map: torch.Tensor,
    box: tuple[int, int, int, int],
    native_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    map_h, map_w = full_map.shape[-2:]
    native_h, native_w = native_shape
    mx0, mx1 = round(x0 * map_w / native_w), round(x1 * map_w / native_w)
    my0, my1 = round(y0 * map_h / native_h), round(y1 * map_h / native_h)
    crop = full_map[..., my0 : max(my0 + 1, my1), mx0 : max(mx0 + 1, mx1)]
    return F.interpolate(crop[:, None], output_shape, mode="bilinear", align_corners=False)[:, 0]


def _validation(
    model: HRNetDenseMIL, records: list[ImageRecord], config: PipelineConfig, device: torch.device
) -> dict[str, float]:
    labels, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for record in records:
            image, _ = _full_view(load_native_grayscale(record.image_path), config)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(_tensor(image, device))
            labels.append(float(record.is_tumor))
            probabilities.append(float(torch.sigmoid(output.tumor_logit)[0].cpu()))
    if len(set(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def train_hrnet(config: PipelineConfig) -> Path:
    seed_everything(config.experiment.seed)
    device = torch.device(config.runtime.device)
    records = read_manifest(config.data.manifest, data_root=config.data.root)
    train = [
        r
        for r in records
        if r.split == "train" and r.fold != config.experiment.selector_holdout_fold
    ]
    validation = [r for r in records if r.split == "val"]
    if not train or not validation:
        raise ValueError("HRNet requires non-empty inner-train and validation records")
    model = _model(config, device, pretrained=config.hrnet.pretrained)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(config.hrnet.backbone_lr, config.hrnet.head_lr),
        weight_decay=config.hrnet.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train) / config.hrnet.gradient_accumulation)
    total_steps = max(1, steps_per_epoch * config.hrnet.epochs)
    warmup_steps = steps_per_epoch * config.hrnet.warmup_epochs

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(config.experiment.output_dir) / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "hrnet_best.pt"
    last_checkpoint = output / "hrnet_last.pt"
    history: list[dict[str, float]] = []
    best, patience, update_step, start_epoch = -np.inf, 0, 0, 0
    if config.experiment.resume and last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, patience = (
            int(state["epoch"]),
            float(state["best"]),
            int(state["patience"]),
        )
        history = list(state["history"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.hrnet.epochs):
        epoch_started = time.perf_counter()
        model.train()
        epoch_records = list(train)
        random.Random(config.experiment.seed + epoch).shuffle(epoch_records)
        running = 0.0
        for sample_index, record in enumerate(epoch_records):
            native = load_native_grayscale(record.image_path)
            full, full_valid_shape = _full_view(native, config)
            tiles = sample_native_tiles(
                native,
                record.image_id,
                config.data.tile_sizes,
                config.data.tile_overlap,
                config.hrnet.tiles_per_image,
                random.Random(config.experiment.seed + epoch * len(train) + sample_index),
            )
            target = torch.tensor([record.class_index], device=device)
            binary = torch.tensor([float(record.is_tumor)], device=device)
            multi_hot = _multi_hot(record, config.hrnet.output_classes, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                full_output = model(_tensor(full, device))
                full_loss, _ = hrnet_mil_loss(
                    full_output,
                    target,
                    binary_targets=binary,
                    multi_hot_targets=multi_hot,
                    normal_weight=config.hrnet.normal_suppression_weight,
                    consistency_weight=0,
                )
            (full_loss / config.hrnet.gradient_accumulation).backward()
            running += float(full_loss.detach())
            full_reference = _native_reference(
                full_output.tumor_map.detach(),
                padded_shape=full.shape,
                valid_shape=full_valid_shape,
                native_shape=native.shape,
            )
            tile_pixels = [
                resize_square(tile.pixels, config.hrnet.network_tile_size) for tile in tiles
            ]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tile_output = model(_tile_tensor(tile_pixels, device))
                references = [
                    _full_reference_crop(
                        full_reference,
                        tile.box,
                        native.shape,
                        tuple(tile_output.tumor_map.shape[-2:]),
                    )
                    for tile in tiles
                ]
                tile_loss, _ = hrnet_tile_bag_loss(
                    tile_output,
                    target,
                    multi_hot,
                    references,
                    normal_weight=config.hrnet.normal_suppression_weight,
                    consistency_weight=config.hrnet.full_tile_consistency_weight,
                )
            (tile_loss / config.hrnet.gradient_accumulation).backward()
            running += float(tile_loss.detach())
            if (
                sample_index + 1
            ) % config.hrnet.gradient_accumulation == 0 or sample_index + 1 == len(epoch_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1
        metrics = _validation(model, validation, config, device)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": running / len(train),
            "seconds": time.perf_counter() - epoch_started,
            **metrics,
        }
        history.append(row)
        score = metrics["auprc"]
        if score > best:
            best, patience = score, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, checkpoint
            )
        else:
            patience += 1
        atomic_json(output / "hrnet_history.json", history)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "patience": patience,
                "history": history,
            },
            last_checkpoint,
        )
        if patience >= config.hrnet.early_stopping_patience:
            break
    return checkpoint


def load_trained_hrnet(
    config: PipelineConfig, checkpoint: str | Path | None = None
) -> HRNetDenseMIL:
    device = torch.device(config.runtime.device)
    model = _model(config, device, pretrained=False)
    path = Path(checkpoint or Path(config.experiment.output_dir) / "checkpoints/hrnet_best.pt")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    return model.eval()


@torch.inference_mode()
def infer_hrnet_sources(
    model: HRNetDenseMIL, image: np.ndarray, image_id: str, config: PipelineConfig
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    device = next(model.parameters()).device
    full, full_valid_shape = _full_view(image, config)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(_tensor(full, device))
    full_map = (
        _native_reference(
            output.tumor_map,
            padded_shape=full.shape,
            valid_shape=full_valid_shape,
            native_shape=image.shape,
        )[0]
        .cpu()
        .numpy()
    )
    full_confidence = float(torch.sigmoid(output.tumor_logit)[0].cpu())
    tile_maps, boxes, tile_probabilities = [], [], []
    tiles = [
        tile
        for size in config.data.tile_sizes
        for tile in extract_tiles(
            image, image_id=image_id, tile_size=size, overlap=config.data.tile_overlap
        )
    ]
    for start in range(0, len(tiles), config.hrnet.inference_tile_batch):
        chunk = tiles[start : start + config.hrnet.inference_tile_batch]
        pixels = [resize_square(tile.pixels, config.hrnet.network_tile_size) for tile in chunk]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            tile_output = model(_tile_tensor(pixels, device))
        for index, tile in enumerate(chunk):
            native_tile = (
                F.interpolate(
                    tile_output.tumor_map[index : index + 1, None].float(),
                    (tile.scale, tile.scale),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
            )
            tile_maps.append(native_tile)
            boxes.append(tile.box)
            tile_probabilities.append(float(torch.sigmoid(tile_output.tumor_logit[index]).cpu()))
    blended = blend_tile_maps(tile_maps, boxes, image.shape)
    maximum = np.zeros_like(blended)
    for values, (x0, y0, x1, y1) in zip(tile_maps, boxes, strict=True):
        maximum[y0:y1, x0:x1] = np.maximum(maximum[y0:y1, x0:x1], values[: y1 - y0, : x1 - x0])
    weight = config.hrnet.tile_fusion_max_weight
    tile_map = weight * maximum + (1 - weight) * blended
    return {"hrnet_full": full_map.astype(np.float32), "hrnet_tile": tile_map.astype(np.float32)}, {
        "hrnet_full": full_confidence,
        "hrnet_tile": float(max(tile_probabilities, default=0)),
    }


def empirical_cdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, np.float32).reshape(-1))
    if not len(reference):
        raise ValueError("Calibration reference is empty")
    return (np.searchsorted(reference, values, side="right") / len(reference)).astype(np.float32)


def calibrate_hrnet(config: PipelineConfig, model: HRNetDenseMIL) -> Path:
    records = read_manifest(config.data.manifest, data_root=config.data.root)
    normals = [
        r
        for r in records
        if r.split == "train"
        and not r.is_tumor
        and r.fold != config.experiment.selector_holdout_fold
    ]
    if not normals:
        raise ValueError("HRNet calibration requires normal inner-train images")
    random.Random(config.experiment.seed).shuffle(normals)
    normals = normals[: config.hrnet.calibration_normal_images]
    references = {"hrnet_full": [], "hrnet_tile": []}
    rng = np.random.default_rng(config.experiment.seed)
    for record in normals:
        maps, _ = infer_hrnet_sources(
            model, load_native_grayscale(record.image_path), record.image_id, config
        )
        for source, values in maps.items():
            flat = values.ravel()
            indices = rng.choice(len(flat), min(4096, len(flat)), replace=False)
            references[source].append(flat[indices])
    path = Path(config.experiment.output_dir) / "calibration/hrnet_normal_cdf.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.sort(np.concatenate(value)) for key, value in references.items()}
    )
    return path
