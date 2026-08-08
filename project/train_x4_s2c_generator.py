from __future__ import annotations

"""Train the frozen X4 S2C generator using image labels and cached SAM regions."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.s2c import BTXRDS2CDataset, collate_s2c_batch
from config import BTXRD_S2C_PIPELINE
from models.s2c import (
    DenseNet121S2C,
    build_cached_cpm_targets,
    cpm_consistency_loss,
    normalize_positive_cam,
    score_cam_segments,
    score_cam_proposals,
    segment_contrastive_loss,
)
from pseudo.manifest import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-segment-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--image-size", type=int, default=BTXRD_S2C_PIPELINE.classifier_image_size)
    parser.add_argument("--batch-size", type=int, default=BTXRD_S2C_PIPELINE.classifier_batch_size)
    parser.add_argument("--epochs", type=int, default=BTXRD_S2C_PIPELINE.classifier_epochs)
    parser.add_argument("--lr", type=float, default=BTXRD_S2C_PIPELINE.classifier_lr)
    parser.add_argument("--weight-decay", type=float, default=BTXRD_S2C_PIPELINE.classifier_weight_decay)
    parser.add_argument("--fpn-channels", type=int, default=BTXRD_S2C_PIPELINE.fpn_channels)
    parser.add_argument("--embedding-dim", type=int, default=BTXRD_S2C_PIPELINE.embedding_dim)
    parser.add_argument("--dropout", type=float, default=BTXRD_S2C_PIPELINE.dropout)
    parser.add_argument(
        "--pool-top-fraction",
        type=float,
        default=BTXRD_S2C_PIPELINE.pool_top_fraction,
    )
    parser.add_argument("--seed", type=int, default=BTXRD_S2C_PIPELINE.classifier_seed)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--radimagenet-checkpoint", type=Path, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--ssc-loss-weight", type=float, default=BTXRD_S2C_PIPELINE.ssc_loss_weight)
    parser.add_argument("--cpm-loss-weight", type=float, default=BTXRD_S2C_PIPELINE.cpm_loss_weight)
    parser.add_argument(
        "--flip-equivariance-loss-weight",
        type=float,
        default=BTXRD_S2C_PIPELINE.flip_equivariance_loss_weight,
    )
    parser.add_argument("--ssc-temperature", type=float, default=BTXRD_S2C_PIPELINE.ssc_temperature)
    parser.add_argument(
        "--ssc-max-pixels-per-segment",
        type=int,
        default=BTXRD_S2C_PIPELINE.ssc_max_pixels_per_segment,
    )
    parser.add_argument("--cpm-start-epoch", type=int, default=BTXRD_S2C_PIPELINE.cpm_start_epoch)
    parser.add_argument(
        "--cpm-scales",
        type=str,
        default=",".join(str(value) for value in BTXRD_S2C_PIPELINE.cpm_scales),
    )
    parser.add_argument("--cpm-positive-threshold", type=float, default=BTXRD_S2C_PIPELINE.cpm_positive_threshold)
    parser.add_argument("--cpm-min-positive-score", type=float, default=BTXRD_S2C_PIPELINE.cpm_min_positive_score)
    parser.add_argument("--cpm-negative-threshold", type=float, default=BTXRD_S2C_PIPELINE.cpm_negative_threshold)
    parser.add_argument("--cpm-min-sam-quality", type=float, default=BTXRD_S2C_PIPELINE.cpm_min_sam_quality)
    parser.add_argument("--cpm-top-k", type=int, default=BTXRD_S2C_PIPELINE.cpm_top_k)
    parser.add_argument(
        "--cpm-positive-weight-max",
        type=float,
        default=BTXRD_S2C_PIPELINE.cpm_positive_weight_max,
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--cpm-scales must contain positive comma-separated values")
    return scales


def _git_provenance() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def _multiscale_cam(
    model: DenseNet121S2C,
    images: torch.Tensor,
    scales: tuple[float, ...],
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    target_size = tuple(images.shape[-2:])
    cams = []
    for scale in scales:
        if abs(scale - 1.0) < 1e-8:
            scaled = images
        else:
            scaled = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        logits = model(scaled)["tumor_cam_logits"]
        cams.append(torch.relu(F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)))
    if was_training:
        model.train()
    return normalize_positive_cam(torch.stack(cams).sum(dim=0))


def _binary_f1(targets: list[int], predictions: list[int]) -> float:
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


@torch.no_grad()
def _validate(
    model: DenseNet121S2C,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    min_sam_quality: float,
) -> dict[str, float]:
    model.eval()
    binary_targets: list[int] = []
    binary_predictions: list[int] = []
    tumor_weak_localization_scores: list[float] = []
    tumor_flip_equivariance_scores: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        output = model(images)
        flipped_output = model(torch.flip(images, dims=(-1,)))
        binary_targets.extend(int(value) for value in batch["tumor"].tolist())
        binary_predictions.extend(int(value) for value in (output["tumor_logit"].sigmoid() >= 0.5).cpu().tolist())
        segments = batch["segments"].to(device)
        cam = normalize_positive_cam(
            output["tumor_cam_logits"],
            size=tuple(segments.shape[-2:]),
        )
        flipped_cam = torch.flip(
            normalize_positive_cam(
                flipped_output["tumor_cam_logits"],
                size=tuple(segments.shape[-2:]),
            ),
            dims=(-1,),
        )
        for sample_index, quality_cpu in enumerate(batch["quality"]):
            quality = quality_cpu.to(device)
            proposal_masks = batch["proposal_masks"][sample_index]
            proposal_quality = batch["proposal_quality"][sample_index].to(device)
            if proposal_masks.shape[0] > 0:
                _means, joint = score_cam_proposals(
                    cam[sample_index], proposal_masks.to(device), proposal_quality
                )
                reliable = proposal_quality >= min_sam_quality
                top_joint = float(joint[reliable].max().item()) if bool(reliable.any()) else 0.0
            else:
                ids, _means, joint = score_cam_segments(
                    cam[sample_index], segments[sample_index], quality
                )
                if ids.numel():
                    reliable = quality[ids] >= min_sam_quality
                    top_joint = float(joint[reliable].max().item()) if bool(reliable.any()) else 0.0
                else:
                    top_joint = 0.0
            is_tumor = bool(batch["tumor"][sample_index].item() > 0.5)
            if is_tumor:
                native_has_evidence = float(cam[sample_index].amax().item() > 1e-6)
                flipped_has_evidence = float(flipped_cam[sample_index].amax().item() > 1e-6)
                evidence_gate = native_has_evidence * flipped_has_evidence
                tumor_weak_localization_scores.append(top_joint)
                tumor_flip_equivariance_scores.append(
                    evidence_gate
                    * (
                        1.0
                        - float(
                            (cam[sample_index] - flipped_cam[sample_index])
                            .abs()
                            .mean()
                            .item()
                        )
                    )
                )
    binary_f1 = _binary_f1(binary_targets, binary_predictions)
    weak_localization = (
        float(np.mean(tumor_weak_localization_scores))
        if tumor_weak_localization_scores
        else 0.0
    )
    flip_equivariance = (
        float(np.mean(tumor_flip_equivariance_scores))
        if tumor_flip_equivariance_scores
        else 0.0
    )
    return {
        "val_binary_f1": binary_f1,
        "val_weak_localization_score": weak_localization,
        "val_flip_equivariance": flip_equivariance,
        "selection_score": (
            0.5 * binary_f1 + 0.3 * weak_localization + 0.2 * flip_equivariance
        ),
        "val_images": float(len(binary_targets)),
    }


def _checkpoint_payload(
    model: DenseNet121S2C,
    *,
    epoch: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": 3,
        "method": "x4_cached_sam_s2c_style_wsss",
        "ground_truth_spatial_supervision": False,
        "student_trained": False,
        "test_evaluated": False,
        "epoch": epoch,
        "checkpoint_role": "fixed_epoch_snapshot",
        "model_config": {
            "architecture": "densenet121_binary_fpn_s2c",
            "output_stride": model.output_stride,
            "fpn_channels": model.fpn_channels,
            "embedding_dim": model.embedding_dim,
            "dropout": model.dropout_probability,
            "pool_mode": "top_percent",
            "top_fraction": model.top_fraction,
            "classes": ["normal", "tumor"],
        },
        "state_dict": model.state_dict(),
        "training_metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.image_size <= 0
        or args.fpn_channels <= 0
        or args.embedding_dim <= 0
    ):
        raise ValueError("Epoch, batch, image, FPN and embedding sizes must be positive")
    if not 0.0 < args.pool_top_fraction <= 1.0:
        raise ValueError("pool-top-fraction must be in (0,1]")
    if (
        args.cpm_start_epoch < 1
        or args.cpm_top_k < 1
        or args.cpm_positive_weight_max < 1.0
    ):
        raise ValueError("CPM start epoch/top-k must be positive and positive weight at least 1")
    scales = _parse_scales(args.cpm_scales)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    normalization = "radimagenet" if args.radimagenet_checkpoint else "imagenet"
    pretrained_source = (
        "radimagenet" if args.radimagenet_checkpoint else ("random" if args.no_pretrained else "imagenet")
    )

    train_dataset = BTXRDS2CDataset(
        root=args.data_root,
        split="train",
        split_manifest=args.split_manifest,
        segment_cache_dir=args.train_segment_cache,
        image_size=args.image_size,
        augment=args.augment,
        normalization=normalization,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_s2c_batch,
        pin_memory=args.device == "cuda",
    )

    model = DenseNet121S2C(
        fpn_channels=args.fpn_channels,
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        top_fraction=args.pool_top_fraction,
        radimagenet_checkpoint=args.radimagenet_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    binary_criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    git_commit, git_dirty = _git_provenance()
    metadata: dict[str, object] = {
        "schema_version": 1,
        "method": "x4_cached_sam_s2c_style_wsss",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "reference": "Kweon et al., From SAM to CAMs, CVPR 2024",
        "adaptation": (
            "offline automatic SAM proposals replace online SAM ViT-H CPM; "
            "binary stride-4 FPN replaces the rejected stride-32 GAP localizer; "
            "SSC prototype is detached"
        ),
        "architecture": "densenet121_binary_fpn_s2c",
        "classes": ["normal", "tumor"],
        "output_stride": 4,
        "wsss_supervision": "BTXRD binary image label plus class-agnostic SAM segments",
        "ground_truth_spatial_supervision": False,
        "annotation_files_opened": False,
        "outer_validation_images_opened": False,
        "validation_gt_accessed": False,
        "student_trained": False,
        "test_evaluated": False,
        "selection_rule": "fixed terminal epoch; no outer-validation selection",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "fpn_channels": args.fpn_channels,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "pool_mode": "top_percent",
        "pool_top_fraction": args.pool_top_fraction,
        "seed": args.seed,
        "augment": args.augment,
        "amp": amp_enabled,
        "loss_weights": {
            "binary": args.binary_loss_weight,
            "ssc": args.ssc_loss_weight,
            "cpm": args.cpm_loss_weight,
            "flip_equivariance": args.flip_equivariance_loss_weight,
        },
        "ssc_temperature": args.ssc_temperature,
        "ssc_max_pixels_per_segment": args.ssc_max_pixels_per_segment,
        "cpm_start_epoch": args.cpm_start_epoch,
        "cpm_scales": scales,
        "cpm_positive_threshold": args.cpm_positive_threshold,
        "cpm_min_positive_score": args.cpm_min_positive_score,
        "cpm_negative_threshold": args.cpm_negative_threshold,
        "cpm_min_sam_quality": args.cpm_min_sam_quality,
        "cpm_top_k": args.cpm_top_k,
        "cpm_positive_weight_max": args.cpm_positive_weight_max,
        "screening_only": bool(args.max_train_batches),
        "max_train_batches": args.max_train_batches,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "train_segment_cache": str(args.train_segment_cache.resolve()),
        "train_segment_summary_sha256": train_dataset.cache_summary["summary_sha256"],
        "radimagenet_checkpoint": str(args.radimagenet_checkpoint.resolve()) if args.radimagenet_checkpoint else None,
        "radimagenet_checkpoint_sha256": sha256_file(args.radimagenet_checkpoint) if args.radimagenet_checkpoint else None,
        "normalization": normalization,
        "pretrained_source": pretrained_source,
        "imagenet_pretrained": not args.no_pretrained and args.radimagenet_checkpoint is None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    metadata["configuration_sha256"] = _canonical_hash(metadata)
    metadata_path = args.output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "binary": 0.0,
            "ssc": 0.0,
            "cpm": 0.0,
            "flip_equivariance": 0.0,
        }
        batches = 0
        cpm_selected_images = 0
        cpm_selected_segments = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images = batch["image"].to(device)
            tumor = batch["tumor"].to(device)
            segments = batch["segments"].to(device)
            qualities = [quality.to(device) for quality in batch["quality"]]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                loss_binary = binary_criterion(output["tumor_logit"], tumor)
                loss_ssc = segment_contrastive_loss(
                    output["embedding"],
                    segments,
                    temperature=args.ssc_temperature,
                    max_pixels_per_segment=args.ssc_max_pixels_per_segment,
                )
                if args.flip_equivariance_loss_weight > 0:
                    flipped_output = model(torch.flip(images, dims=(-1,)))
                    cam_native = normalize_positive_cam(output["tumor_cam_logits"])
                    cam_flipped = torch.flip(
                        normalize_positive_cam(flipped_output["tumor_cam_logits"]),
                        dims=(-1,),
                    )
                    tumor_weight = tumor[:, None, None]
                    loss_flip = (
                        (cam_native - cam_flipped).abs() * tumor_weight
                    ).sum() / (
                        tumor_weight.sum().clamp_min(1.0)
                        * cam_native.shape[-2]
                        * cam_native.shape[-1]
                    )
                else:
                    loss_flip = output["tumor_cam_logits"].sum() * 0.0
                if epoch >= args.cpm_start_epoch:
                    # Training CPM reuses the current stride-4 CAM.  Running
                    # three additional DenseNet forwards per batch would not
                    # change the supervision source and would multiply the
                    # full-compute cost. Multi-scale CAM remains enabled for
                    # frozen pseudo-mask generation.
                    cam = normalize_positive_cam(
                        output["tumor_cam_logits"],
                        size=tuple(segments.shape[-2:]),
                    ).detach()
                    targets, cpm_weights, cpm_stats = build_cached_cpm_targets(
                        cam,
                        segments,
                        qualities,
                        tumor,
                        proposal_masks=batch["proposal_masks"],
                        proposal_qualities=batch["proposal_quality"],
                        positive_threshold=args.cpm_positive_threshold,
                        min_positive_score=args.cpm_min_positive_score,
                        negative_cam_threshold=args.cpm_negative_threshold,
                        min_sam_quality=args.cpm_min_sam_quality,
                        top_k=args.cpm_top_k,
                    )
                    loss_cpm = cpm_consistency_loss(
                        output["tumor_cam_logits"],
                        targets,
                        cpm_weights,
                        positive_weight_max=args.cpm_positive_weight_max,
                    )
                    cpm_selected_images += cpm_stats["selected_images"]
                    cpm_selected_segments += cpm_stats["selected_segments"]
                else:
                    loss_cpm = output["tumor_cam_logits"].sum() * 0.0
                loss = (
                    args.binary_loss_weight * loss_binary
                    + args.ssc_loss_weight * loss_ssc
                    + args.cpm_loss_weight * loss_cpm
                    + args.flip_equivariance_loss_weight * loss_flip
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.item())
            totals["binary"] += float(loss_binary.item())
            totals["ssc"] += float(loss_ssc.item())
            totals["cpm"] += float(loss_cpm.item())
            totals["flip_equivariance"] += float(loss_flip.item())
            batches += 1

        if batches == 0:
            raise RuntimeError("No training batches were processed")
        row: dict[str, object] = {
            "epoch": epoch,
            "train_batches": batches,
            **{f"train_{key}": value / batches for key, value in totals.items()},
            "cpm_selected_images": cpm_selected_images,
            "cpm_selected_segments": cpm_selected_segments,
        }
        log_rows.append(row)
        payload = _checkpoint_payload(model, epoch=epoch, metadata=metadata)
        torch.save(payload, args.output_dir / "last_s2c.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    run_manifest = {
        **metadata,
        "training_metadata_sha256": sha256_file(metadata_path),
        "training_log_sha256": sha256_file(log_path),
        "terminal_checkpoint": str((args.output_dir / "last_s2c.pt").resolve()),
        "last_checkpoint_sha256": sha256_file(args.output_dir / "last_s2c.pt"),
        "epochs_completed": len(log_rows),
        "outer_validation_selection": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
