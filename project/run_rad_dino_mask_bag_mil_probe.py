from __future__ import annotations

"""Train/freeze a SAM-proposal MIL selector using image labels only.

Candidate proposal bags are generated and hash-frozen before this runner. The
runner consumes radiographs, frozen RAD-DINO tokens, class-agnostic proposal
masks and binary image labels. It has no segmentation-dataset import and never
opens validation annotations; a separate evaluator owns that boundary.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    save_float_map,
    sha256_file,
    verify_model_snapshot,
)
from models.mask_bag_affinity_features import affinity_summary_features
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    mask_pool_descriptors,
    proposal_context_grid_weights,
    project_direct_resize_masks_to_square,
    self_guided_instance_loss,
    smooth_mil_pool,
    winner_take_all_map,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from run_rad_dino_multilayer_soft_region_probe import _raw_and_normalized_square


SELECTED_HIDDEN_LAYERS = (4, 8, 12)
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-pseudo-manifest-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--instance-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--instance-warmup-epochs", type=int, default=2)
    parser.add_argument("--maximum-candidates", type=int, default=81)
    parser.add_argument("--rich-gallery-union", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ProjectedMultiLayerEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projection: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("projection", projection.float(), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != 13:
            raise RuntimeError("RAD-DINO must expose embedding plus 12 hidden states")
        batch = pixel_values.shape[0]
        expected_tokens = (pixel_values.shape[-1] // 14) ** 2 + 1
        grid_size = int(math.isqrt(expected_tokens - 1))
        selected: list[torch.Tensor] = []
        for layer_index in SELECTED_HIDDEN_LAYERS:
            hidden = hidden_states[layer_index]
            if hidden.shape != (batch, expected_tokens, 768):
                raise RuntimeError(
                    f"Unexpected RAD-DINO layer-{layer_index} shape {tuple(hidden.shape)}"
                )
            patches = hidden[:, 1:].reshape(batch, grid_size, grid_size, 768).float()
            selected.append(torch.nn.functional.normalize(patches @ self.projection, dim=-1))
        return torch.stack(selected, dim=1)


def _load_candidate_payload(
    candidate_root: Path,
    manifest_row: dict[str, str],
    *,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = candidate_root / manifest_row["diagnostic_path"]
    if sha256_file(path) != manifest_row["diagnostic_sha256"]:
        raise ValueError(f"Candidate payload hash mismatch: {manifest_row['image_name']}")
    with np.load(path, allow_pickle=False) as payload:
        masks = payload["sam_masks"].astype(np.float32)
        prompt_map = payload["prompt_map"].astype(np.float32)
        sam_scores = payload["sam_scores"].astype(np.float32).reshape(-1)
    if masks.ndim != 3 or prompt_map.ndim != 2:
        raise ValueError("Candidate payload has an invalid spatial layout")
    if masks.shape[1:] != prompt_map.shape or len(masks) != len(sam_scores):
        raise ValueError("Candidate payload arrays are not aligned")
    if len(masks) > maximum_candidates:
        raise RuntimeError(
            f"Candidate bag exceeds frozen cap {maximum_candidates}: "
            f"{manifest_row['image_name']} has {len(masks)}"
        )
    fallback = False
    if not len(masks):
        threshold = float(np.percentile(prompt_map, 90.0))
        candidate = (prompt_map >= threshold) & (prompt_map > 0)
        if not candidate.any():
            candidate = np.zeros_like(prompt_map, dtype=bool)
            height, width = candidate.shape
            y0, y1 = height // 4, height - height // 4
            x0, x1 = width // 4, width - width // 4
            candidate[y0:y1, x0:x1] = True
        masks = candidate[None].astype(np.float32)
        sam_scores = np.zeros(1, dtype=np.float32)
        fallback = True
    total = float(prompt_map.size)
    prompt_mass = max(float(prompt_map.sum()), 1.0e-8)
    metadata: list[list[float]] = []
    for mask, sam_score in zip(masks, sam_scores, strict=True):
        binary = mask > 0.5
        area = float(binary.sum())
        inside = prompt_map[binary]
        metadata.append(
            [
                float(np.clip(sam_score, 0.0, 1.0)),
                float(np.log1p(area) / np.log1p(total)),
                float((prompt_map * binary).sum() / prompt_mass),
                float(inside.mean()) if inside.size else 0.0,
            ]
        )
    fallback_flags = np.full(len(masks), int(fallback), dtype=np.uint8)
    return masks, np.asarray(metadata, dtype=np.float32), sam_scores, fallback_flags


def _audit_candidate_input(
    root: Path,
    rows: list[dict[str, str]],
    *,
    split: str,
    expected_manifest_sha256: str,
    expected_pseudo_manifest_sha256: str,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    indexed, audit = validate_candidate_diagnostics_manifest(
        root,
        expected_image_names=[row["image_id"] for row in rows],
        split=split,
        expected_pseudo_manifest_sha256=expected_pseudo_manifest_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if audit.get("cohort") != "all":
        raise ValueError("Mask-bag MIL requires a complete positive/normal proposal cohort")
    if len(indexed) != len(rows):
        raise RuntimeError("Candidate manifest count differs from frozen split")
    return indexed, audit


def _pool_one_bag(
    token_maps: torch.Tensor,
    masks: np.ndarray,
    metadata: np.ndarray,
    content_mask: np.ndarray,
    config: MaskBagMILConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mask_tensor = torch.from_numpy(masks)[None]
    metadata_tensor = torch.from_numpy(metadata)[None]
    validity = torch.ones((1, len(masks)), dtype=torch.bool)
    descriptors, pooled_valid = mask_pool_descriptors(
        token_maps[None].float(),
        mask_tensor,
        metadata_tensor,
        validity,
        config,
        content_masks=torch.from_numpy(content_mask)[None],
    )
    kept = torch.nonzero(pooled_valid[0], as_tuple=False).reshape(-1)
    if not len(kept):
        raise RuntimeError("Candidate bag contains no proposal with token-grid support")
    return (
        descriptors[0, kept].numpy().astype(np.float16),
        kept.numpy().astype(np.int32),
    )


def _affinity_one_bag(
    token_maps: torch.Tensor,
    masks: np.ndarray,
    content_mask: np.ndarray,
    kept_indices: np.ndarray,
    config: MaskBagMILConfig,
) -> np.ndarray:
    """Compute R2 summaries with the exact descriptor validity/context geometry."""

    feature_device = token_maps.device
    mask_tensor = torch.from_numpy(masks)[None].to(feature_device)
    candidate_valid = torch.ones(
        (1, len(masks)),
        dtype=torch.bool,
        device=feature_device,
    )
    proposal, context, valid = proposal_context_grid_weights(
        mask_tensor,
        candidate_valid,
        grid_height=int(token_maps.shape[-3]),
        grid_width=int(token_maps.shape[-2]),
        minimum_grid_mass=config.minimum_grid_mass,
        context_radius=config.context_radius,
        content_masks=torch.from_numpy(content_mask)[None].to(feature_device),
    )
    kept = torch.nonzero(valid[0], as_tuple=False).reshape(-1)
    expected = torch.from_numpy(
        np.asarray(kept_indices, dtype=np.int64)
    ).to(feature_device)
    if not torch.equal(kept, expected):
        raise RuntimeError("Affinity and descriptor candidate validity differ")
    features = affinity_summary_features(
        token_maps[None].float(),
        proposal,
        context,
        valid,
    )
    return features[0, kept].float().cpu().numpy().astype(np.float16)


def build_descriptor_cache(
    rows: list[dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
    encoder: nn.Module,
    config: MaskBagMILConfig,
    args: argparse.Namespace,
    device: torch.device,
    *,
    split: str,
    include_affinity_features: bool = False,
) -> list[dict[str, object]]:
    cache: list[dict[str, object]] = []
    for start in range(0, len(rows), args.encoder_batch_size):
        batch_rows = rows[start : start + args.encoder_batch_size]
        pixels: list[torch.Tensor] = []
        payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        projections = []
        for row in batch_rows:
            image_path = locate_verified_image(args.dataset_root, row)
            with Image.open(image_path) as image:
                _raw, normalized, projection = _raw_and_normalized_square(
                    image,
                    input_size=args.input_size,
                )
            pixels.append(normalized)
            projections.append(projection)
            payloads.append(
                _load_candidate_payload(
                    candidate_root,
                    candidate_rows[Path(row["image_id"]).stem],
                    maximum_candidates=args.maximum_candidates,
                )
            )
        original = torch.stack(pixels, dim=0)
        augmented = torch.cat([original, original.flip(-1)], dim=0)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            encoded_token_batch = encoder(augmented.to(device, non_blocking=True))
        token_batch = encoded_token_batch.float().cpu()
        count = len(batch_rows)
        for offset, (row, payload, projection) in enumerate(
            zip(batch_rows, payloads, projections, strict=True)
        ):
            masks, metadata, _sam_scores, fallback_flags = payload
            descriptor_masks = project_direct_resize_masks_to_square(
                torch.from_numpy(masks),
                padded_side=projection.padded_side,
                content_box=projection.content_box,
                output_size=int(token_batch.shape[-2]) * 4,
            ).numpy()
            descriptor_content = project_direct_resize_masks_to_square(
                torch.ones((1, masks.shape[-2], masks.shape[-1])),
                padded_side=projection.padded_side,
                content_box=projection.content_box,
                output_size=int(token_batch.shape[-2]) * 4,
            )[0].numpy()
            descriptors, kept = _pool_one_bag(
                token_batch[offset],
                descriptor_masks,
                metadata,
                descriptor_content,
                config,
            )
            flipped_descriptors, flipped_kept = _pool_one_bag(
                token_batch[count + offset],
                descriptor_masks[..., ::-1].copy(),
                metadata,
                descriptor_content[..., ::-1].copy(),
                config,
            )
            if not np.array_equal(kept, flipped_kept):
                raise RuntimeError("Original/flip candidate validity differs")
            record: dict[str, object] = {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "label": int(row["tumor"]),
                "descriptors": descriptors,
                "flipped_descriptors": flipped_descriptors,
                "kept_indices": kept,
                "fallback_count": int(fallback_flags[kept].sum()),
                "candidate_payload_sha256": candidate_rows[
                    Path(row["image_id"]).stem
                ][
                    "diagnostic_sha256"
                ],
            }
            if include_affinity_features:
                affinity = _affinity_one_bag(
                    encoded_token_batch[offset],
                    descriptor_masks,
                    descriptor_content,
                    kept,
                    config,
                )
                flipped_affinity = _affinity_one_bag(
                    encoded_token_batch[count + offset],
                    descriptor_masks[..., ::-1].copy(),
                    descriptor_content[..., ::-1].copy(),
                    kept,
                    config,
                )
                if affinity.shape != flipped_affinity.shape:
                    raise RuntimeError("Original/flip affinity shapes differ")
                record["affinity_features"] = affinity
                record["flipped_affinity_features"] = flipped_affinity
            cache.append(record)
        completed = min(start + len(batch_rows), len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"mask-bag {split} feature cache: {completed}/{len(rows)}", flush=True)
    return cache


def _padded_batch(
    cache: list[dict[str, object]],
    indices: np.ndarray,
    key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = [cache[int(index)] for index in indices]
    maximum = max(len(record[key]) for record in records)
    dimension = int(records[0][key].shape[1])
    values = np.zeros((len(records), maximum, dimension), dtype=np.float32)
    valid = np.zeros((len(records), maximum), dtype=bool)
    labels = np.zeros(len(records), dtype=np.float32)
    for row_index, record in enumerate(records):
        descriptor = np.asarray(record[key], dtype=np.float32)
        values[row_index, : len(descriptor)] = descriptor
        valid[row_index, : len(descriptor)] = True
        labels[row_index] = float(record["label"])
    return (
        torch.from_numpy(values).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(labels).to(device),
    )


def train_selector(
    cache: list[dict[str, object]],
    config: MaskBagMILConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RadDinoMaskBagMIL, list[dict[str, float]]]:
    model = RadDinoMaskBagMIL(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = np.random.default_rng(args.seed + epoch)
        order = generator.permutation(len(cache))
        sums = {"total": 0.0, "image": 0.0, "instance": 0.0, "consistency": 0.0}
        batches = 0
        for start in range(0, len(order), args.train_batch_size):
            indices = order[start : start + args.train_batch_size]
            descriptors, valid, labels = _padded_batch(
                cache, indices, "descriptors", device
            )
            flipped, flipped_valid, _ = _padded_batch(
                cache, indices, "flipped_descriptors", device
            )
            if not torch.equal(valid, flipped_valid):
                raise RuntimeError("Original/flip cached validity differs")
            logits, bag_logits = model.score_descriptors(descriptors, valid)
            flip_logits, flip_bag_logits = model.score_descriptors(flipped, valid)
            image_loss = 0.5 * (
                image_bag_loss(bag_logits, labels)
                + image_bag_loss(flip_bag_logits, labels)
            )
            if epoch > args.instance_warmup_epochs:
                instance_loss = 0.5 * (
                    self_guided_instance_loss(logits, valid, labels)
                    + self_guided_instance_loss(flip_logits, valid, labels)
                )
            else:
                instance_loss = logits.sum() * 0.0
            consistency = aligned_candidate_consistency_loss(
                logits, flip_logits, valid
            )
            total = (
                image_loss
                + args.instance_loss_weight * instance_loss
                + args.consistency_loss_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("instance", instance_loss),
                ("consistency", consistency),
            ):
                sums[key] += float(value.detach().item())
            batches += 1
        record = {"epoch": float(epoch)}
        record.update({key: value / batches for key, value in sums.items()})
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return model, history


def write_validation_predictions(
    model: RadDinoMaskBagMIL,
    cache: list[dict[str, object]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    map_dir = args.output_dir / "predictions" / "maps"
    map_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    model.eval()
    for index, record in enumerate(cache):
        descriptors = torch.from_numpy(
            np.asarray(record["descriptors"], dtype=np.float32)
        )[None].to(device)
        flipped_descriptors = torch.from_numpy(
            np.asarray(record["flipped_descriptors"], dtype=np.float32)
        )[None].to(device)
        if flipped_descriptors.shape != descriptors.shape:
            raise RuntimeError("Original/flip validation descriptor shapes differ")
        valid = torch.ones(descriptors.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            original_logits, _original_bag_logits = model.score_descriptors(
                descriptors, valid
            )
            flipped_logits, _flipped_bag_logits = model.score_descriptors(
                flipped_descriptors, valid
            )
            logits = 0.5 * (original_logits + flipped_logits)
            bag_logits = smooth_mil_pool(
                logits,
                valid,
                temperature=model.config.bag_temperature,
            )
        manifest_row = candidate_rows[Path(str(record["image_id"])).stem]
        masks, _metadata, _sam_scores, _fallback_flags = _load_candidate_payload(
            candidate_root,
            manifest_row,
            maximum_candidates=args.maximum_candidates,
        )
        kept = np.asarray(record["kept_indices"], dtype=np.int32)
        kept_masks = torch.from_numpy(masks[kept])[None].to(device)
        probability_map, winner = winner_take_all_map(
            logits, kept_masks, valid, bag_logits
        )
        output_path = map_dir / f"{Path(str(record['image_id'])).stem}.npy"
        save_float_map(output_path, probability_map[0].cpu().numpy())
        local_winner = int(winner.item())
        original_winner = int(kept[local_winner])
        records.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                "candidate_count": len(kept),
                "selected_candidate_index": original_winner,
                "selected_candidate_logit": float(logits[0, local_winner].item()),
                "candidate_logit_tta": "mean_original_aligned_horizontal_flip",
                "bag_logit": float(bag_logits[0].item()),
                "bag_probability": float(torch.sigmoid(bag_logits[0]).item()),
                "selected_area_ratio": float(masks[original_winner].mean()),
                "fallback_count": record["fallback_count"],
                "map_path": str(Path("maps") / output_path.name),
                "map_sha256": sha256_file(output_path),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(cache):
            print(f"mask-bag validation maps: {index + 1}/{len(cache)}", flush=True)
    manifest_path = args.output_dir / "predictions" / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return sha256_file(manifest_path)


def main() -> None:
    args = parse_args()
    if args.input_size != 448 or args.projection_dim != 128:
        raise ValueError("The v1 mask-bag protocol requires input 448/projection 128")
    if args.encoder_batch_size < 2:
        raise ValueError("Mask-bag v1 requires encoder batch >=2")
    if args.rich_gallery_union:
        if not 1 <= args.maximum_candidates <= 162:
            raise ValueError("Rich gallery union cap must be in [1, 162]")
    elif args.maximum_candidates != 81:
        raise ValueError("Mask-bag v1 requires candidate cap 81")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

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
        raise RuntimeError("Frozen BTXRD train/validation cohort mismatch")
    train_candidates, train_audit = _audit_candidate_input(
        args.train_candidate_root,
        train_rows,
        split="train",
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
    )
    val_candidates, val_audit = _audit_candidate_input(
        args.val_candidate_root,
        val_rows,
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    model_snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("Unexpected transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Mask-bag v1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"Mask-bag v1 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder: nn.Module = ProjectedMultiLayerEncoder(
        backbone,
        torch.from_numpy(projection),
    ).to(device)
    encoder = nn.DataParallel(encoder, device_ids=[0, 1], output_device=0).eval()
    config = MaskBagMILConfig(
        token_dim=args.projection_dim,
        token_layers=len(SELECTED_HIDDEN_LAYERS),
    )
    train_cache = build_descriptor_cache(
        train_rows,
        train_candidates,
        args.train_candidate_root,
        encoder,
        config,
        args,
        device,
        split="train",
    )
    val_cache = build_descriptor_cache(
        val_rows,
        val_candidates,
        args.val_candidate_root,
        encoder,
        config,
        args,
        device,
        split="val",
    )
    del encoder, backbone
    torch.cuda.empty_cache()

    model, history = train_selector(train_cache, config, args, device)
    checkpoint_path = args.output_dir / "rad_dino_mask_bag_mil.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "final_epoch": args.epochs,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    prediction_manifest_sha256 = write_validation_predictions(
        model,
        val_cache,
        val_candidates,
        args.val_candidate_root,
        args,
        device,
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "candidate_descriptor_geometry": {
            "candidate_frame": "direct-resize source-image coordinates",
            "encoder_frame": "centered square-padded source-image coordinates",
            "projection": "continuous content-box transform with bilinear sampling",
            "oversampling_per_token_axis": 4,
            "flip": "horizontal flip of the projected square mask",
            "padding_exclusion": "fractional content occupancy excludes square padding from proposal and local-context pooling",
            "rich_gallery_union": args.rich_gallery_union,
            "maximum_candidates": args.maximum_candidates,
        },
        "train_candidate_manifest_sha256": args.train_candidate_manifest_sha256,
        "train_pseudo_manifest_sha256": args.train_pseudo_manifest_sha256,
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history_sha256": sha256_file(history_path),
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "validation_predictions": len(val_cache),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "run_id": "btxrd_rad_dino_mask_bag_mil_probe_val_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "instance_loss_weight": args.instance_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "instance_warmup_epochs": args.instance_warmup_epochs,
            "final_epoch_only": True,
            "rich_gallery_union": args.rich_gallery_union,
            "maximum_candidates": args.maximum_candidates,
        },
        "candidate_inputs": {"train": train_audit, "validation": val_audit},
        "cohort": {"train": len(train_rows), "validation": len(val_rows)},
        "fallback_bags": {
            "train": sum(int(row["fallback_count"] > 0) for row in train_cache),
            "validation": sum(int(row["fallback_count"] > 0) for row in val_cache),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "encoder_data_parallel": True,
            "pid": os.getpid(),
        },
        "output_hashes": freeze,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
