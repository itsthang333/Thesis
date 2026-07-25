from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.factory import build_classification_dataset
from models.s2c_cpm import (
    DenseNet121S2CCPMClassifier,
    cpm_cross_entropy_loss,
    normalized_foreground_cam,
)
from models.sam_segment_contrastive import (
    SamSegmentMapStore,
    sam_segment_contrastive_loss,
)
from progress import should_disable_tqdm
from train_classifier import (
    classifier_epoch_budget_audit,
    confusion_counts,
    metrics_from_confusion,
)


EXPECTED_TARGET_COLUMNS = ["tumor"]
EXPECTED_IMAGE_SIZE = 320
EXPECTED_FEATURE_CHANNELS = 256
EXPECTED_FEATURE_STRIDE = 8
EXPECTED_EMBEDDING_SHAPE = (256, 64, 64)
EXPECTED_SAM_SOURCE_SIZE = 512
EXPECTED_SAM_ENCODER_SIZE = 1024
DEFAULT_CAM_SCALES = (0.5, 1.0, 1.5, 2.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the predeclared binary DenseNet S2C SSC+CPM classifier"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sam-segment-map-root", type=Path, required=True)
    parser.add_argument(
        "--sam-segment-map-manifest-sha256",
        type=str,
        required=True,
    )
    parser.add_argument("--sam-embedding-cache-root", type=Path, required=True)
    parser.add_argument(
        "--sam-embedding-cache-sha256",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sam-embedding-index-sha256",
        type=str,
        required=True,
    )
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint-sha256", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=EXPECTED_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-channels", type=int, default=256)
    parser.add_argument("--ssc-weight", type=float, default=1.0)
    parser.add_argument("--ssc-temperature", type=float, default=1.0)
    parser.add_argument("--cpm-weight", type=float, default=1.0)
    parser.add_argument("--cpm-warmup-epochs", type=int, default=2)
    parser.add_argument("--cpm-peak-threshold", type=float, default=0.5)
    parser.add_argument("--cpm-peak-min-distance", type=int, default=20)
    parser.add_argument("--cpm-max-peaks", type=int, default=8)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SamEmbeddingCache:
    def __init__(
        self,
        root: Path,
        train_samples: list[dict[str, object]],
        *,
        expected_cache_sha256: str,
        expected_index_sha256: str,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.cache_path = self.root / "sam_embeddings_fp16.npy"
        self.index_path = self.root / "embedding_index.csv"
        self.manifest_path = self.root / "run_manifest.json"
        for path in (self.cache_path, self.index_path, self.manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if len(expected_cache_sha256) != 64:
            raise ValueError("A frozen SAM embedding-cache SHA-256 is required")
        if len(expected_index_sha256) != 64:
            raise ValueError("A frozen SAM embedding-index SHA-256 is required")
        actual_cache_sha = sha256_file(self.cache_path)
        actual_index_sha = sha256_file(self.index_path)
        if actual_cache_sha != expected_cache_sha256:
            raise RuntimeError(
                "SAM embedding-cache SHA-256 mismatch: "
                f"{actual_cache_sha} != {expected_cache_sha256}"
            )
        if actual_index_sha != expected_index_sha256:
            raise RuntimeError(
                "SAM embedding-index SHA-256 mismatch: "
                f"{actual_index_sha} != {expected_index_sha256}"
            )

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("test_evaluated") is not False:
            raise RuntimeError("Embedding cache must keep test_evaluated=false")
        if manifest.get("polygons_or_masks_loaded") is not False:
            raise RuntimeError("Embedding cache may not load polygons or masks")
        if manifest.get("validation_images_processed") is not False:
            raise RuntimeError("Embedding cache may not process validation images")
        if manifest.get("processed_split") != "train":
            raise RuntimeError("Embedding cache must be train-only")

        with self.index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.index_by_image: dict[str, int] = {}
        for expected_index, row in enumerate(rows):
            index = int(row["embedding_index"])
            if index != expected_index:
                raise RuntimeError("Embedding index must be contiguous and ordered")
            image_id = row["image_id"]
            if image_id in self.index_by_image:
                raise RuntimeError(f"Duplicate embedding image ID: {image_id}")
            self.index_by_image[image_id] = index

        expected_tumors = {
            str(sample["image_id"])
            for sample in train_samples
            if int(sample["tumor"]) == 1
        }
        if set(self.index_by_image) != expected_tumors:
            missing = sorted(expected_tumors - set(self.index_by_image))
            extra = sorted(set(self.index_by_image) - expected_tumors)
            raise RuntimeError(
                "Embedding population differs from clean-train tumors: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        self.array = np.load(self.cache_path, mmap_mode="r")
        expected_shape = (len(rows), *EXPECTED_EMBEDDING_SHAPE)
        if self.array.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {self.array.shape} != {expected_shape}"
            )
        if self.array.dtype != np.float16:
            raise RuntimeError(f"Unexpected embedding dtype: {self.array.dtype}")
        self.cache_sha256 = actual_cache_sha
        self.index_sha256 = actual_index_sha
        self.manifest = manifest

    def load(self, image_id: str, *, device: torch.device) -> torch.Tensor:
        if image_id not in self.index_by_image:
            raise KeyError(f"No train-tumor SAM embedding for {image_id}")
        index = self.index_by_image[image_id]
        array = np.array(self.array[index], dtype=np.float32, copy=True)
        return torch.from_numpy(array).to(device=device, non_blocking=True)


def extract_cpm_peaks(
    cam: torch.Tensor,
    *,
    threshold: float,
    min_distance: int,
    max_peaks: int,
) -> torch.Tensor:
    """Return deterministic ``(x,y)`` peaks, always including the global max."""

    if cam.ndim != 2:
        raise ValueError("cam must have shape [H,W]")
    if min_distance <= 0 or max_peaks <= 0:
        raise ValueError("min_distance and max_peaks must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    height, width = cam.shape
    flat_index = int(torch.argmax(cam).item())
    global_row, global_column = divmod(flat_index, width)
    selected: list[tuple[int, int]] = [(global_row, global_column)]

    local_max = F.max_pool2d(
        cam[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    candidates = torch.nonzero(
        (cam >= local_max) & (cam > threshold),
        as_tuple=False,
    )
    ranked = sorted(
        (
            (float(cam[row, column]), int(row), int(column))
            for row, column in candidates.tolist()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    distance_squared = min_distance * min_distance
    for _score, row, column in ranked:
        if (row, column) == selected[0]:
            continue
        if any(
            (row - other_row) ** 2 + (column - other_column) ** 2
            < distance_squared
            for other_row, other_column in selected
        ):
            continue
        selected.append((row, column))
        if len(selected) >= max_peaks:
            break
    return torch.tensor(
        [(column, row) for row, column in selected],
        dtype=torch.float32,
        device=cam.device,
    )


class CachedSamPromptDecoder:
    def __init__(
        self,
        checkpoint: Path,
        *,
        expected_sha256: str,
        cache: SamEmbeddingCache,
        device: torch.device,
    ) -> None:
        if sha256_file(checkpoint) != expected_sha256:
            raise RuntimeError("SAM checkpoint SHA-256 mismatch")
        from segment_anything import sam_model_registry

        self.sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.sam.to(device=device)
        self.sam.eval()
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        if int(self.sam.image_encoder.img_size) != EXPECTED_SAM_ENCODER_SIZE:
            raise RuntimeError("Unexpected SAM encoder size")
        self.cache = cache
        self.device = device

    @torch.inference_mode()
    def decode(
        self,
        image_id: str,
        cam_320: torch.Tensor,
        *,
        peak_threshold: float,
        peak_min_distance: int,
        max_peaks: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        points_320 = extract_cpm_peaks(
            cam_320,
            threshold=peak_threshold,
            min_distance=peak_min_distance,
            max_peaks=max_peaks,
        )
        points_encoder = points_320 * (
            EXPECTED_SAM_ENCODER_SIZE / EXPECTED_IMAGE_SIZE
        )
        point_labels = torch.ones(
            (1, points_encoder.shape[0]),
            dtype=torch.int64,
            device=self.device,
        )
        sparse, dense = self.sam.prompt_encoder(
            points=(points_encoder[None], point_labels),
            boxes=None,
            masks=None,
        )
        embedding = self.cache.load(image_id, device=self.device)[None]
        low_resolution_masks, predicted_iou = self.sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        mask_logits = self.sam.postprocess_masks(
            low_resolution_masks,
            input_size=(EXPECTED_SAM_ENCODER_SIZE, EXPECTED_SAM_ENCODER_SIZE),
            original_size=(EXPECTED_SAM_SOURCE_SIZE, EXPECTED_SAM_SOURCE_SIZE),
        )[0]
        masks_320 = F.interpolate(
            (mask_logits > self.sam.mask_threshold).float()[:, None],
            size=(EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
            mode="nearest",
        )[:, 0] > 0.5
        cam_means = torch.stack(
            [
                cam_320[mask].mean()
                if bool(mask.any())
                else cam_320.new_zeros(())
                for mask in masks_320
            ]
        )
        reliability = predicted_iou[0].float() * cam_means
        best = int(torch.argmax(reliability).item())
        selected = masks_320[best]
        return selected, {
            "peaks": float(points_320.shape[0]),
            "sam_predicted_iou": float(predicted_iou[0, best].item()),
            "cam_mean_inside": float(cam_means[best].item()),
            "reliability": float(reliability[best].item()),
            "foreground_fraction": float(selected.float().mean().item()),
        }


@torch.inference_mode()
def multiscale_teacher_cam(
    model: DenseNet121S2CCPMClassifier,
    images: torch.Tensor,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        accumulated = None
        for scale in DEFAULT_CAM_SCALES:
            scaled = (
                images
                if scale == 1.0
                else F.interpolate(
                    images,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            _logits, _features, cam_logits = model(
                scaled,
                return_spatial=True,
            )
            cam = normalized_foreground_cam(
                cam_logits,
                output_size=(EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
            )
            accumulated = cam if accumulated is None else accumulated + cam
        assert accumulated is not None
        maxima = F.adaptive_max_pool2d(accumulated, output_size=1)
        return accumulated / (maxima + 1e-5)
    finally:
        model.train(was_training)


def build_cpm_targets(
    model: DenseNet121S2CCPMClassifier,
    decoder: CachedSamPromptDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    image_ids: list[str],
    *,
    peak_threshold: float,
    peak_min_distance: int,
    max_peaks: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    cams = multiscale_teacher_cam(model, images)[:, 0]
    targets = torch.zeros(
        images.shape[0],
        1,
        EXPECTED_IMAGE_SIZE,
        EXPECTED_IMAGE_SIZE,
        device=images.device,
    )
    diagnostics: list[dict[str, float]] = []
    for index, image_id in enumerate(image_ids):
        if float(labels[index].reshape(-1)[0].item()) <= 0.5:
            continue
        mask, local = decoder.decode(
            image_id,
            cams[index],
            peak_threshold=peak_threshold,
            peak_min_distance=peak_min_distance,
            max_peaks=max_peaks,
        )
        targets[index, 0] = mask.float()
        diagnostics.append(local)
    if not diagnostics:
        return targets, {
            "positive_images": 0.0,
            "mean_peaks": 0.0,
            "mean_sam_predicted_iou": 0.0,
            "mean_reliability": 0.0,
            "mean_foreground_fraction": 0.0,
        }
    return targets, {
        "positive_images": float(len(diagnostics)),
        "mean_peaks": sum(item["peaks"] for item in diagnostics)
        / len(diagnostics),
        "mean_sam_predicted_iou": sum(
            item["sam_predicted_iou"] for item in diagnostics
        )
        / len(diagnostics),
        "mean_reliability": sum(
            item["reliability"] for item in diagnostics
        )
        / len(diagnostics),
        "mean_foreground_fraction": sum(
            item["foreground_fraction"] for item in diagnostics
        )
        / len(diagnostics),
    }


def run_epoch(
    model: DenseNet121S2CCPMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    train: bool,
    epoch: int,
    region_store: SamSegmentMapStore | None = None,
    decoder: CachedSamPromptDecoder | None = None,
    ssc_weight: float = 1.0,
    ssc_temperature: float = 1.0,
    cpm_weight: float = 1.0,
    cpm_warmup_epochs: int = 2,
    peak_threshold: float = 0.5,
    peak_min_distance: int = 20,
    max_peaks: int = 8,
) -> tuple[float, dict[str, float], dict[str, int], dict[str, float]]:
    total_cls = 0.0
    total_ssc = 0.0
    total_cpm = 0.0
    total_optimization = 0.0
    total_positive_images = 0.0
    weighted_peaks = 0.0
    weighted_iou = 0.0
    weighted_reliability = 0.0
    weighted_area = 0.0
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batches = 0
    model.train(train)

    progress = tqdm(
        loader,
        desc="train" if train else "val",
        leave=False,
        disable=should_disable_tqdm(),
    )
    for images, targets, image_ids in progress:
        images = images.to(device)
        targets = targets.to(device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        use_cpm = train and epoch > cpm_warmup_epochs
        cpm_targets = None
        prompt_diagnostics = {
            "positive_images": 0.0,
            "mean_peaks": 0.0,
            "mean_sam_predicted_iou": 0.0,
            "mean_reliability": 0.0,
            "mean_foreground_fraction": 0.0,
        }
        if use_cpm:
            if decoder is None:
                raise RuntimeError("CPM decoder is required after warmup")
            cpm_targets, prompt_diagnostics = build_cpm_targets(
                model,
                decoder,
                images,
                targets,
                list(image_ids),
                peak_threshold=peak_threshold,
                peak_min_distance=peak_min_distance,
                max_peaks=max_peaks,
            )

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if train:
                    logits, features, cam_logits = model(
                        images,
                        return_spatial=True,
                    )
                else:
                    logits = model(images)
                    features = None
                    cam_logits = None
                cls_loss = criterion(logits, targets)
            if train:
                if region_store is None or features is None:
                    raise RuntimeError("SSC region store/features are required")
                region_maps = region_store.load_batch(image_ids, device=device)
                with torch.cuda.amp.autocast(enabled=False):
                    ssc_loss = sam_segment_contrastive_loss(
                        features,
                        region_maps,
                        temperature=ssc_temperature,
                    )
                    cpm_loss = (
                        cpm_cross_entropy_loss(
                            cam_logits,
                            cpm_targets,
                            targets,
                        )
                        if use_cpm
                        else cls_loss.new_zeros(())
                    )
                    loss = (
                        cls_loss.float()
                        + ssc_weight * ssc_loss
                        + cpm_weight * cpm_loss
                    )
            else:
                ssc_loss = cls_loss.new_zeros(())
                cpm_loss = cls_loss.new_zeros(())
                loss = cls_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite S2C CPM optimization loss")
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("Non-finite S2C CPM gradient norm")
                scaler.step(optimizer)
                scaler.update()

        batch_counts = confusion_counts(logits.detach(), targets.detach())
        for key in counts:
            counts[key] += batch_counts[key]
        total_cls += float(cls_loss.item())
        total_ssc += float(ssc_loss.item())
        total_cpm += float(cpm_loss.item())
        total_optimization += float(loss.item())
        positives = prompt_diagnostics["positive_images"]
        total_positive_images += positives
        weighted_peaks += positives * prompt_diagnostics["mean_peaks"]
        weighted_iou += positives * prompt_diagnostics["mean_sam_predicted_iou"]
        weighted_reliability += positives * prompt_diagnostics["mean_reliability"]
        weighted_area += positives * prompt_diagnostics["mean_foreground_fraction"]
        batches += 1
        progress.set_postfix(
            cls=float(cls_loss.item()),
            ssc=float(ssc_loss.item()),
            cpm=float(cpm_loss.item()),
        )

    if batches == 0:
        raise RuntimeError("No batches were processed")
    diagnostics = {
        "classification_loss": total_cls / batches,
        "ssc_loss": total_ssc / batches,
        "cpm_loss": total_cpm / batches,
        "optimization_loss": total_optimization / batches,
        "cpm_positive_images": total_positive_images,
        "mean_cpm_peaks": weighted_peaks / max(1.0, total_positive_images),
        "mean_sam_predicted_iou": weighted_iou
        / max(1.0, total_positive_images),
        "mean_cpm_reliability": weighted_reliability
        / max(1.0, total_positive_images),
        "mean_cpm_foreground_fraction": weighted_area
        / max(1.0, total_positive_images),
    }
    return (
        diagnostics["classification_loss"],
        metrics_from_confusion(counts),
        counts,
        diagnostics,
    )


def save_checkpoint(
    path: Path,
    model: DenseNet121S2CCPMClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_val_f1: float,
    config: dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_val_f1,
            "target_columns": EXPECTED_TARGET_COLUMNS,
            "task": "multi-label",
            "dataset": "btxrd",
            "num_classes": 1,
            "normalization": "imagenet",
            "train_augment": False,
            "pipeline_profile": "s2c_cpm_fpn_v1",
            "split_manifest_sha256": config["split_manifest_sha256"],
            "image_size": EXPECTED_IMAGE_SIZE,
            "seed": int(config["seed"]),
            "s2c_cpm": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError("The predeclared CPM recipe requires image size 320")
    if args.feature_channels != EXPECTED_FEATURE_CHANNELS:
        raise ValueError("The predeclared CPM recipe requires 256 FPN channels")
    if args.batch_size != 2:
        raise ValueError("The predeclared CPM recipe requires batch size 2")
    if args.epochs != 30 or args.early_stop_patience != 7:
        raise ValueError("The predeclared CPM recipe requires 30 epochs/patience 7")
    if args.cpm_warmup_epochs != 2:
        raise ValueError("The predeclared CPM recipe requires two warmup epochs")
    for name, value in (
        ("ssc_weight", args.ssc_weight),
        ("ssc_temperature", args.ssc_temperature),
        ("cpm_weight", args.cpm_weight),
    ):
        if value != 1.0:
            raise ValueError(f"The predeclared CPM recipe fixes {name}=1")
    if not torch.cuda.is_available():
        raise RuntimeError("S2C CPM training requires a CUDA GPU")
    seed_everything(args.seed)
    device = torch.device("cuda")

    train_dataset = build_classification_dataset(
        root=args.data_root,
        split="train",
        target_columns=EXPECTED_TARGET_COLUMNS,
        image_size=EXPECTED_IMAGE_SIZE,
        augment=False,
        preprocessing_mode="none",
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    val_dataset = build_classification_dataset(
        root=args.data_root,
        split="val",
        target_columns=EXPECTED_TARGET_COLUMNS,
        image_size=EXPECTED_IMAGE_SIZE,
        augment=False,
        preprocessing_mode="none",
        normalization="imagenet",
        split_manifest=args.split_manifest,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    region_store = SamSegmentMapStore(
        args.sam_segment_map_root,
        train_dataset.samples,
        expected_manifest_sha256=args.sam_segment_map_manifest_sha256,
    )
    embedding_cache = SamEmbeddingCache(
        args.sam_embedding_cache_root,
        train_dataset.samples,
        expected_cache_sha256=args.sam_embedding_cache_sha256,
        expected_index_sha256=args.sam_embedding_index_sha256,
    )
    decoder = CachedSamPromptDecoder(
        args.sam_checkpoint,
        expected_sha256=args.sam_checkpoint_sha256,
        cache=embedding_cache,
        device=device,
    )
    model = DenseNet121S2CCPMClassifier(
        pretrained=True,
        feature_channels=args.feature_channels,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    cpm_config: dict[str, object] = {
        "method": "BTXRD DenseNet FPN adaptation of S2C SSC plus CPM",
        "architecture": "DenseNet121 stride-8 FPN CAM classifier",
        "feature_sources": ["denseblock2", "final"],
        "feature_channels": EXPECTED_FEATURE_CHANNELS,
        "feature_stride": EXPECTED_FEATURE_STRIDE,
        "classification_head": "global average pooling of one-channel CAM",
        "image_size": EXPECTED_IMAGE_SIZE,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_manifest_sha256": sha256_file(args.split_manifest.resolve()),
        "augmentation": False,
        "preprocessing": "none",
        "normalization": "imagenet",
        "ssc_weight": args.ssc_weight,
        "ssc_temperature": args.ssc_temperature,
        "ssc_region_manifest_sha256": region_store.manifest_sha256,
        "cpm_weight": args.cpm_weight,
        "cpm_warmup_epochs": args.cpm_warmup_epochs,
        "cpm_cam_scales": list(DEFAULT_CAM_SCALES),
        "cpm_peak_threshold": args.cpm_peak_threshold,
        "cpm_peak_min_distance": args.cpm_peak_min_distance,
        "cpm_max_peaks": args.cpm_max_peaks,
        "sam_model": "vit_b",
        "sam_checkpoint_sha256": args.sam_checkpoint_sha256,
        "sam_embedding_cache_sha256": embedding_cache.cache_sha256,
        "sam_embedding_index_sha256": embedding_cache.index_sha256,
        "sam_embedding_cache_population": len(
            embedding_cache.index_by_image
        ),
        "sam_mask_selection": (
            "highest SAM predicted-IoU times mean normalized CAM inside mask"
        ),
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "maximum_epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_selection": "clean validation tumor F1 at threshold 0.5",
        "train_polygons_loaded": False,
        "validation_polygons_loaded": False,
        "test_evaluated": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "split_manifest": str(args.split_manifest.resolve()),
                "split_manifest_sha256": sha256_file(
                    args.split_manifest.resolve()
                ),
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "s2c_cpm": cpm_config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    history_path = args.output_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "train_classification_loss",
        "train_ssc_loss",
        "train_cpm_loss",
        "train_optimization_loss",
        "train_f1",
        "train_sensitivity",
        "train_specificity",
        "val_classification_loss",
        "val_f1",
        "val_sensitivity",
        "val_specificity",
        "cpm_positive_images",
        "mean_cpm_peaks",
        "mean_sam_predicted_iou",
        "mean_cpm_reliability",
        "mean_cpm_foreground_fraction",
    ]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    budget_records: list[dict[str, float | int]] = []
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _train_counts, train_diagnostics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            epoch=epoch,
            region_store=region_store,
            decoder=decoder,
            ssc_weight=args.ssc_weight,
            ssc_temperature=args.ssc_temperature,
            cpm_weight=args.cpm_weight,
            cpm_warmup_epochs=args.cpm_warmup_epochs,
            peak_threshold=args.cpm_peak_threshold,
            peak_min_distance=args.cpm_peak_min_distance,
            max_peaks=args.cpm_max_peaks,
        )
        val_loss, val_metrics, _val_counts, _val_diagnostics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
            epoch=epoch,
        )
        row = {
            "epoch": epoch,
            "train_classification_loss": train_loss,
            "train_ssc_loss": train_diagnostics["ssc_loss"],
            "train_cpm_loss": train_diagnostics["cpm_loss"],
            "train_optimization_loss": train_diagnostics[
                "optimization_loss"
            ],
            "train_f1": train_metrics["f1"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "val_classification_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "cpm_positive_images": train_diagnostics[
                "cpm_positive_images"
            ],
            "mean_cpm_peaks": train_diagnostics["mean_cpm_peaks"],
            "mean_sam_predicted_iou": train_diagnostics[
                "mean_sam_predicted_iou"
            ],
            "mean_cpm_reliability": train_diagnostics[
                "mean_cpm_reliability"
            ],
            "mean_cpm_foreground_fraction": train_diagnostics[
                "mean_cpm_foreground_fraction"
            ],
        }
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        budget_records.append(
            {
                "epoch": epoch,
                "val_f1": float(val_metrics["f1"]),
            }
        )
        print(
            f"Epoch {epoch:03d} | train_cls={train_loss:.4f} "
            f"ssc={train_diagnostics['ssc_loss']:.4f} "
            f"cpm={train_diagnostics['cpm_loss']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f}",
            flush=True,
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_classifier.pt",
                model,
                optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
                config=cpm_config,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            args.output_dir / "last_classifier.pt",
            model,
            optimizer,
            epoch=epoch,
            best_val_f1=best_val_f1,
            config=cpm_config,
        )
        if epochs_without_improvement >= args.early_stop_patience:
            stopped_early = True
            print(
                f"Early stopping after {epoch} epochs; "
                f"best validation F1={best_val_f1:.6f}",
                flush=True,
            )
            break

    budget_audit = classifier_epoch_budget_audit(
        budget_records,
        requested_epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        stopped_early=stopped_early,
    )
    budget_audit.update(
        {
            "metric": "audited-split validation F1",
            "split": "val",
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(
                args.split_manifest.resolve()
            ),
            "pipeline_profile": "s2c_cpm_fpn_v1",
        }
    )
    (args.output_dir / "classifier_epoch_budget_audit.json").write_text(
        json.dumps(budget_audit, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
