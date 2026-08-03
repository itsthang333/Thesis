"""Train, score, and physically freeze the S10 proposal-MIL arms.

Only radiographs, binary image labels, and class-agnostic proposal masks enter
this runner. Validation segmentation annotations and BTXRD test data are not
accepted arguments. A separate independent auditor must pass before evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_bas_candidate_descriptor_core as base
import run_rad_dino_mask_bag_mil_probe as legacy
from mae_reconstruction_io import locate_verified_image, save_float_map, sha256_file
from models.highres_candidate_pmil import (
    HighResProposalMIL,
    HighResProposalMILOutput,
    aligned_view_consistency,
    area_orthogonality_penalty,
    attention_union_consistency,
    candidate_capture_purity,
    dual_stream_bag_probability,
    image_label_proposal_loss,
    pareto_guarded_selection,
    top_instance_dropout_mask,
)
from models.mae_reconstruction import pad_to_square
from models.mask_bag_same_family_graph import (
    SameFamilyGraphConfig,
    score_same_family_graph_records,
)
from models.mask_bag_score_evidence import (
    save_candidate_score_evidence,
    write_candidate_score_manifest,
)


EXPERIMENT_ID = "EXP-20260803-codex-s10-highres-proposal-pmil-v1"
RUN_ID = "btxrd_highres_candidate_pmil_s10_v1"
CONTROL_ARM = "geometry_v3_plus_upstream_control"
CAPACITY_ARM = "control_plus_s10_identity_capacity"
PRIMARY_ARM = "s10_pareto_identity_capture_purity"
EXPECTED_TRAIN = 2981
EXPECTED_VALIDATION = 371
EXPECTED_NORMAL_TRAIN = 1493
EXPECTED_TUMOR_TRAIN = 1488
EXPECTED_IMAGE_SIZE = 640
EXPECTED_SUPPORT_SIZE = 160
EXPECTED_BATCH_SIZE = 4
EXPECTED_EPOCHS = 32
EXPECTED_BACKBONE_LR = 3.0e-5
EXPECTED_HEAD_LR = 3.0e-4
EXPECTED_WEIGHT_DECAY = 1.0e-4
EXPECTED_WARMUP_EPOCHS = 4
EXPECTED_TOP_DROPOUT_FRACTION = 0.2
EXPECTED_MAXIMUM_CANDIDATES = 81
EXPECTED_SEED = 42
EXPECTED_PRETRAINED_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
LOSS_WEIGHTS = {
    "bag": 1.0,
    "normal_candidate": 0.25,
    "normal_pixel": 0.25,
    "tumor_union": 0.25,
    "flip_consistency": 0.10,
    "identity_area_projection": 0.05,
    "detection_area_projection": 0.05,
}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-pseudo-manifest-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--backbone-lr", type=float, default=EXPECTED_BACKBONE_LR)
    parser.add_argument("--head-lr", type=float, default=EXPECTED_HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=EXPECTED_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=EXPECTED_WARMUP_EPOCHS)
    parser.add_argument(
        "--top-dropout-fraction",
        type=float,
        default=EXPECTED_TOP_DROPOUT_FRACTION,
    )
    parser.add_argument(
        "--maximum-candidates", type=int, default=EXPECTED_MAXIMUM_CANDIDATES
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    actual = (
        args.batch_size,
        args.epochs,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
        args.warmup_epochs,
        args.top_dropout_fraction,
        args.maximum_candidates,
        args.seed,
        args.expected_pretrained_sha256,
    )
    expected = (
        EXPECTED_BATCH_SIZE,
        EXPECTED_EPOCHS,
        EXPECTED_BACKBONE_LR,
        EXPECTED_HEAD_LR,
        EXPECTED_WEIGHT_DECAY,
        EXPECTED_WARMUP_EPOCHS,
        EXPECTED_TOP_DROPOUT_FRACTION,
        EXPECTED_MAXIMUM_CANDIDATES,
        EXPECTED_SEED,
        EXPECTED_PRETRAINED_SHA256,
    )
    if actual != expected:
        raise ValueError("S10 execution differs from the frozen one-shot recipe")
    if args.num_workers < 0:
        raise ValueError("S10 num_workers must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sha256_file(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty S10 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _array_sha256(values: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return sha256(stream.getvalue()).hexdigest()


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize(
        (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())
    values = values.permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _project_square_supports(
    masks: np.ndarray,
    *,
    projection: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.from_numpy(np.asarray(masks, dtype=np.float32))
    projected = legacy.project_direct_resize_masks_to_square(
        source,
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=EXPECTED_SUPPORT_SIZE,
    )
    content = legacy.project_direct_resize_masks_to_square(
        torch.ones((1, masks.shape[-2], masks.shape[-1]), dtype=torch.float32),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=EXPECTED_SUPPORT_SIZE,
    )[0]
    if projected.shape != (len(masks), EXPECTED_SUPPORT_SIZE, EXPECTED_SUPPORT_SIZE):
        raise RuntimeError("S10 candidate support shape changed")
    if content.shape != (EXPECTED_SUPPORT_SIZE, EXPECTED_SUPPORT_SIZE):
        raise RuntimeError("S10 content support shape changed")
    return projected.to(torch.float16), content.to(torch.float16)


@dataclass(frozen=True)
class S10InputRecord:
    image_id: str
    group_id: str
    label: int
    image_path: Path
    candidate_path: Path
    candidate_sha256: str
    candidate_indices: np.ndarray


def build_input_records(
    rows: list[dict[str, str]],
    accepted: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
    candidate_root: Path,
    dataset_root: Path,
) -> list[S10InputRecord]:
    if len(rows) != len(accepted):
        raise ValueError("S10 split/cache cohorts do not align")
    result: list[S10InputRecord] = []
    for row, record in zip(rows, accepted):
        if row["image_id"] != record["image_id"]:
            raise RuntimeError("S10 selector-cache order differs from frozen split")
        candidate_row = candidates[Path(row["image_id"]).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"]
            != record["candidate_payload_sha256"]
        ):
            raise RuntimeError("S10 physical/cache candidate provenance mismatch")
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        if (
            indices.ndim != 1
            or not len(indices)
            or len(indices) > EXPECTED_MAXIMUM_CANDIDATES
            or np.any(indices < 0)
            or len(np.unique(indices)) != len(indices)
        ):
            raise RuntimeError("S10 accepted candidate indices are invalid")
        result.append(
            S10InputRecord(
                image_id=row["image_id"],
                group_id=row["group_id"],
                label=int(row["tumor"]),
                image_path=locate_verified_image(dataset_root, row),
                candidate_path=candidate_path,
                candidate_sha256=candidate_row["diagnostic_sha256"],
                candidate_indices=indices,
            )
        )
    return result


class S10Dataset(Dataset[dict[str, object]]):
    def __init__(self, records: list[S10InputRecord], maximum_candidates: int) -> None:
        self.records = records
        self.maximum_candidates = maximum_candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            pixels, projection = _normalized_square(image)
        masks = _load_candidate_masks_without_rehash(
            record.candidate_path, maximum_candidates=self.maximum_candidates
        )
        if masks.ndim != 3 or np.any(record.candidate_indices >= len(masks)):
            raise RuntimeError(f"S10 candidate payload changed: {record.image_id}")
        selected = masks[record.candidate_indices]
        supports, content = _project_square_supports(selected, projection=projection)
        return {
            "image_id": record.image_id,
            "group_id": record.group_id,
            "label": record.label,
            "pixels": pixels,
            "candidate_masks": supports,
            "content_mask": content,
            "candidate_indices": torch.from_numpy(record.candidate_indices.copy()),
            "candidate_payload_sha256": record.candidate_sha256,
        }


def _load_candidate_masks_without_rehash(
    path: Path, *, maximum_candidates: int
) -> np.ndarray:
    """Load a once-audited payload and reproduce the frozen empty-bag fallback."""

    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["sam_masks"], dtype=np.float32)
        prompt_map = np.asarray(payload["prompt_map"], dtype=np.float32)
        sam_scores = np.asarray(payload["sam_scores"], dtype=np.float32).reshape(-1)
    if masks.ndim != 3 or prompt_map.ndim != 2:
        raise ValueError("S10 candidate payload has an invalid spatial layout")
    if masks.shape[1:] != prompt_map.shape or len(masks) != len(sam_scores):
        raise ValueError("S10 candidate payload arrays are not aligned")
    if len(masks) > maximum_candidates:
        raise RuntimeError("S10 candidate bag exceeds the frozen cap")
    if len(masks):
        return masks
    threshold = float(np.percentile(prompt_map, 90.0))
    fallback = (prompt_map >= threshold) & (prompt_map > 0)
    if not fallback.any():
        fallback = np.zeros_like(prompt_map, dtype=bool)
        height, width = fallback.shape
        y0, y1 = height // 4, height - height // 4
        x0, x1 = width // 4, width - width // 4
        fallback[y0:y1, x0:x1] = True
    return fallback[None].astype(np.float32)


def collate_s10(items: list[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("S10 cannot collate an empty batch")
    batch = len(items)
    maximum = max(int(item["candidate_masks"].shape[0]) for item in items)
    masks = torch.zeros(
        (batch, maximum, EXPECTED_SUPPORT_SIZE, EXPECTED_SUPPORT_SIZE),
        dtype=torch.float16,
    )
    valid = torch.zeros((batch, maximum), dtype=torch.bool)
    indices = torch.full((batch, maximum), -1, dtype=torch.int64)
    for row, item in enumerate(items):
        count = int(item["candidate_masks"].shape[0])
        masks[row, :count] = item["candidate_masks"]
        valid[row, :count] = True
        indices[row, :count] = item["candidate_indices"]
    return {
        "image_id": [str(item["image_id"]) for item in items],
        "group_id": [str(item["group_id"]) for item in items],
        "candidate_payload_sha256": [
            str(item["candidate_payload_sha256"]) for item in items
        ],
        "labels": torch.tensor([int(item["label"]) for item in items]),
        "pixels": torch.stack([item["pixels"] for item in items]),
        "candidate_masks": masks,
        "content_mask": torch.stack([item["content_mask"] for item in items]),
        "candidate_valid": valid,
        "candidate_indices": indices,
    }


def _tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("labels", "pixels", "candidate_masks", "content_mask", "candidate_valid")
    }


def _tumor_dropout_valid(
    detection_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    retained = candidate_valid.clone()
    tumor = labels.reshape(-1) == 1
    if bool(tumor.any()) and fraction > 0:
        retained[tumor] = top_instance_dropout_mask(
            detection_logits[tumor], candidate_valid[tumor], fraction=fraction
        )
    return retained


def _view_terms(
    output: HighResProposalMILOutput,
    labels: torch.Tensor,
    *,
    dropout_fraction: float,
) -> dict[str, torch.Tensor]:
    primitive = image_label_proposal_loss(
        output.classification_logits,
        output.detection_logits,
        output.dense_logits,
        labels,
        output.candidate_valid,
    )
    retained = _tumor_dropout_valid(
        output.detection_logits,
        output.candidate_valid,
        labels,
        dropout_fraction,
    )
    bag = dual_stream_bag_probability(
        output.classification_logits, output.detection_logits, retained
    )["bag_probability"]
    bag_loss = F.binary_cross_entropy(bag, labels.float())
    tumor = labels.reshape(-1) == 1
    zero = output.classification_logits[output.candidate_valid].sum() * 0.0
    union = zero
    if bool(tumor.any()):
        attention = dual_stream_bag_probability(
            output.classification_logits[tumor],
            output.detection_logits[tumor],
            output.candidate_valid[tumor],
        )["detection_attention"]
        union = attention_union_consistency(
            output.dense_logits[tumor],
            output.candidate_weights[tumor],
            attention,
            output.candidate_valid[tumor],
        )
    return {
        "bag": bag_loss,
        "normal_candidate": primitive["normal_candidate"],
        "normal_pixel": primitive["normal_pixel"],
        "tumor_union": union,
        "identity_area_projection": area_orthogonality_penalty(
            output.classification_logits,
            output.candidate_area,
            output.candidate_valid,
        ),
        "detection_area_projection": area_orthogonality_penalty(
            output.detection_logits,
            output.candidate_area,
            output.candidate_valid,
        ),
    }


def s10_training_objective(
    original: HighResProposalMILOutput,
    flipped: HighResProposalMILOutput,
    labels: torch.Tensor,
    *,
    dropout_fraction: float,
) -> dict[str, torch.Tensor]:
    first = _view_terms(original, labels, dropout_fraction=dropout_fraction)
    second = _view_terms(flipped, labels, dropout_fraction=dropout_fraction)
    terms = {key: 0.5 * (first[key] + second[key]) for key in first}
    terms["flip_consistency"] = aligned_view_consistency(
        original.classification_logits,
        flipped.classification_logits,
        original.dense_logits,
        flipped.dense_logits.flip(-1),
        original.candidate_valid,
    )
    total = sum(LOSS_WEIGHTS[key] * value for key, value in terms.items())
    return {"total": total, **terms}


def _forward_views(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
) -> tuple[HighResProposalMILOutput, HighResProposalMILOutput]:
    original = model(
        tensors["pixels"],
        tensors["candidate_masks"],
        tensors["content_mask"],
        tensors["candidate_valid"],
    )
    flipped = model(
        tensors["pixels"].flip(-1),
        tensors["candidate_masks"].flip(-1),
        tensors["content_mask"].flip(-1),
        tensors["candidate_valid"],
    )
    return original, flipped


def _parameter_groups(model: HighResProposalMIL, args: argparse.Namespace) -> list[dict[str, object]]:
    backbone_modules = (
        model.fpn.stem,
        model.fpn.layer1,
        model.fpn.layer2,
        model.fpn.layer3,
        model.fpn.layer4,
    )
    backbone = [parameter for module in backbone_modules for parameter in module.parameters()]
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if not backbone or not head or backbone_ids & {id(parameter) for parameter in head}:
        raise RuntimeError("S10 optimizer parameter partition is invalid")
    if len(backbone_ids) + len({id(parameter) for parameter in head}) != len(
        {id(parameter) for parameter in model.parameters()}
    ):
        raise RuntimeError("S10 optimizer parameter partition is incomplete")
    return [
        {"params": backbone, "lr": args.backbone_lr, "name": "backbone"},
        {"params": head, "lr": args.head_lr, "name": "fpn_and_heads"},
    ]


def train_model(
    model: nn.DataParallel,
    loader: DataLoader[dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], dict[str, torch.Tensor]]:
    labels = np.asarray([record.label for record in loader.dataset.records], dtype=np.int8)
    if (
        len(labels) != EXPECTED_TRAIN
        or int((labels == 0).sum()) != EXPECTED_NORMAL_TRAIN
        or int((labels == 1).sum()) != EXPECTED_TUMOR_TRAIN
    ):
        raise RuntimeError("S10 training-label cohort mismatch")
    groups = _parameter_groups(model.module, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        sums = {"total": 0.0, **{key: 0.0 for key in LOSS_WEIGHTS}}
        seen = 0
        normal_seen = 0
        tumor_seen = 0
        dropout_fraction = (
            0.0 if epoch < args.warmup_epochs else args.top_dropout_fraction
        )
        for batch in loader:
            tensors = _tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                original, flipped = _forward_views(model, tensors)
                output = s10_training_objective(
                    original,
                    flipped,
                    tensors["labels"],
                    dropout_fraction=dropout_fraction,
                )
            if not torch.isfinite(output["total"]):
                raise RuntimeError("S10 training loss became non-finite")
            scaler.scale(output["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            size = int(tensors["labels"].shape[0])
            seen += size
            normal_seen += int((tensors["labels"] == 0).sum().item())
            tumor_seen += int((tensors["labels"] == 1).sum().item())
            for key in sums:
                sums[key] += float(output[key].detach().cpu()) * size
        if (
            seen != EXPECTED_TRAIN
            or normal_seen != EXPECTED_NORMAL_TRAIN
            or tumor_seen != EXPECTED_TUMOR_TRAIN
        ):
            raise RuntimeError("S10 epoch cohort changed")
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{key: value / seen for key, value in sums.items()},
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "top_dropout_fraction": dropout_fraction,
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError("S10 training history became non-finite")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    state = {
        key: value.detach().cpu() for key, value in model.module.state_dict().items()
    }
    return history, state


@torch.inference_mode()
def score_validation(
    model: nn.DataParallel,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    result: list[dict[str, object]] = []
    for batch in loader:
        tensors = _tensor_batch(batch, device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            original, flipped = _forward_views(model, tensors)
        identity = 0.5 * (
            original.classification_logits.float()
            + flipped.classification_logits.float()
        )
        dense = 0.5 * (
            original.dense_logits.float() + flipped.dense_logits.float().flip(-1)
        )
        capture, purity = candidate_capture_purity(
            dense,
            original.candidate_weights.float(),
            original.ring_weights.float(),
            original.candidate_valid,
            tensors["content_mask"].float(),
        )
        for row, image_id in enumerate(batch["image_id"]):
            valid = tensors["candidate_valid"][row]
            count = int(valid.sum().item())
            indices = batch["candidate_indices"][row, :count].numpy().astype(np.int32)
            result.append(
                {
                    "image_id": image_id,
                    "group_id": batch["group_id"][row],
                    "label": int(tensors["labels"][row].item()),
                    "candidate_payload_sha256": batch["candidate_payload_sha256"][row],
                    "candidate_indices": indices,
                    "identity": identity[row, :count].cpu().numpy().astype(np.float32),
                    "capture": capture[row, :count].cpu().numpy().astype(np.float32),
                    "purity": purity[row, :count].cpu().numpy().astype(np.float32),
                    "dense_logits": dense[row].cpu().numpy().astype(np.float16),
                    "content_weights": tensors["content_mask"][row]
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                }
            )
    if len(result) != EXPECTED_VALIDATION:
        raise RuntimeError("S10 validation score cohort changed")
    return result


def compose_arms(
    output_dir: Path,
    accepted: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    learned: list[dict[str, object]],
    baseline_rows: list[dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float | int]]:
    if {len(accepted), len(base_scored), len(learned)} != {EXPECTED_VALIDATION}:
        raise RuntimeError("S10 validation evidence cohorts do not align")
    accepted_predictions = {row["image_id"]: row for row in baseline_rows}
    evidence_root = output_dir / "s10_candidate_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    arms: dict[str, list[dict[str, Any]]] = {
        CONTROL_ARM: [],
        CAPACITY_ARM: [],
        PRIMARY_ARM: [],
    }
    capacity_changed = 0
    pareto_changed = 0
    dominator_images = 0
    dominator_candidates = 0
    for ordinal, (record, baseline, semantic) in enumerate(
        zip(accepted, base_scored, learned)
    ):
        image_id = str(record["image_id"])
        if image_id != baseline["image_id"] or image_id != semantic["image_id"]:
            raise RuntimeError("S10 validation evidence order mismatch")
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        if not np.array_equal(indices.astype(np.int32), semantic["candidate_indices"]):
            raise RuntimeError("S10 learned candidate order differs from cache")
        base_logits = np.asarray(baseline["base_candidate_logits"], dtype=np.float32)
        identity = np.asarray(semantic["identity"], dtype=np.float32)
        capture = np.asarray(semantic["capture"], dtype=np.float32)
        purity = np.asarray(semantic["purity"], dtype=np.float32)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]
        ):
            raise RuntimeError("S10 upstream candidate provenance mismatch")
        with np.load(candidate_path, allow_pickle=False) as payload:
            upstream_all = np.asarray(payload["selection_scores"], dtype=np.float32)
        upstream = upstream_all[indices]
        valid = torch.ones((1, len(indices)), dtype=torch.bool)
        control = base.equal_rank_aggregate(
            (torch.from_numpy(base_logits)[None], torch.from_numpy(upstream)[None]),
            valid,
        )[0].numpy().astype(np.float32)
        capacity = base.equal_rank_aggregate(
            (
                torch.from_numpy(base_logits)[None],
                torch.from_numpy(upstream)[None],
                torch.from_numpy(identity)[None],
            ),
            valid,
        )[0].numpy().astype(np.float32)
        control_local = int(np.argmax(control))
        capacity_local = int(np.argmax(capacity))
        decision = pareto_guarded_selection(
            identity, capture, purity, indices, control_local
        )
        primary_local = int(np.flatnonzero(indices == decision.selected_index)[0])
        primary_scores = np.zeros(len(indices), dtype=np.float32)
        primary_scores[primary_local] = 1.0
        capacity_changed += int(capacity_local != control_local)
        pareto_changed += int(decision.switched)
        dominator_images += int(decision.dominator_count > 0)
        dominator_candidates += decision.dominator_count
        relative = Path(f"{ordinal:04d}_{Path(image_id).stem}.npz")
        evidence_path = evidence_root / relative
        np.savez_compressed(
            evidence_path,
            candidate_indices=indices.astype(np.int32),
            baseline_logits=base_logits,
            upstream_scores=upstream,
            identity=identity,
            capture=capture,
            purity=purity,
            dense_logits=np.asarray(semantic["dense_logits"], dtype=np.float16),
            content_weights=np.asarray(semantic["content_weights"], dtype=np.float16),
            control_scores=control,
            capacity_scores=capacity,
            primary_decision_scores=primary_scores,
            control_local_index=np.asarray(control_local, dtype=np.int32),
            capacity_local_index=np.asarray(capacity_local, dtype=np.int32),
            primary_local_index=np.asarray(primary_local, dtype=np.int32),
            dominator_count=np.asarray(decision.dominator_count, dtype=np.int32),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_count": len(indices),
                "evidence_path": str(relative),
                "evidence_sha256": sha256_file(evidence_path),
                "identity_sha256": _array_sha256(identity),
                "capture_sha256": _array_sha256(capture),
                "purity_sha256": _array_sha256(purity),
                "dense_logits_sha256": _array_sha256(semantic["dense_logits"]),
            }
        )
        accepted_row = accepted_predictions[image_id]
        common = {
            "image_id": image_id,
            "bag_logit": float(accepted_row["bag_logit"]),
            "bag_probability": float(accepted_row["bag_probability"]),
        }
        arms[CONTROL_ARM].append(
            {**common, "candidate_logits": control, "selected_local_index": control_local}
        )
        arms[CAPACITY_ARM].append(
            {**common, "candidate_logits": capacity, "selected_local_index": capacity_local}
        )
        arms[PRIMARY_ARM].append(
            {
                **common,
                "candidate_logits": primary_scores,
                "selected_local_index": primary_local,
            }
        )
    diagnostics: dict[str, float | int] = {
        "capacity_changed_selections": capacity_changed,
        "capacity_changed_selection_fraction": capacity_changed / EXPECTED_VALIDATION,
        "pareto_changed_selections": pareto_changed,
        "pareto_changed_selection_fraction": pareto_changed / EXPECTED_VALIDATION,
        "images_with_pareto_dominator": dominator_images,
        "total_pareto_dominators": dominator_candidates,
    }
    return arms, _write_csv(evidence_root / "evidence_manifest.csv", evidence_rows), diagnostics


def write_validation_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    *,
    recipe: str,
) -> tuple[str, str]:
    if len(records) != len(scored) or len(records) != EXPECTED_VALIDATION:
        raise ValueError("S10 output records/scores do not align")
    prediction_root = output_dir / "predictions"
    map_root = prediction_root / "maps"
    score_root = output_dir / "candidate_scores"
    score_payload_root = score_root / "scores"
    map_root.mkdir(parents=True, exist_ok=False)
    score_payload_root.mkdir(parents=True, exist_ok=False)
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for ordinal, (record, prediction) in enumerate(zip(records, scored)):
        if record["image_id"] != prediction["image_id"]:
            raise RuntimeError("S10 output order differs from selector cache")
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        scores = np.asarray(prediction["candidate_logits"], dtype=np.float32)
        local_winner = int(prediction["selected_local_index"])
        if (
            scores.shape != indices.shape
            or not np.isfinite(scores).all()
            or int(np.argmax(scores)) != local_winner
        ):
            raise RuntimeError("S10 output decision score is invalid")
        stem = f"{ordinal:04d}_{Path(str(record['image_id'])).stem}"
        score_relative = Path("scores") / f"{stem}.npz"
        saved_score = save_candidate_score_evidence(
            score_root / score_relative,
            candidate_indices=indices,
            candidate_logits=scores,
        )
        masks = base.unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        bag_probability = float(prediction["bag_probability"])
        map_path = map_root / f"{stem}.npy"
        save_float_map(map_path, masks[local_winner] * bag_probability)
        score_rows.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                **saved_score,
                "score_path": str(score_relative),
            }
        )
        prediction_rows.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                "candidate_count": len(indices),
                "selected_candidate_index": int(indices[local_winner]),
                "selected_candidate_logit": saved_score["selected_candidate_logit"],
                "candidate_logit_recipe": recipe,
                "bag_logit": prediction["bag_logit"],
                "bag_probability": bag_probability,
                "selected_area_ratio": float(masks[local_winner].mean()),
                "fallback_count": int(np.asarray(record["fallback_flags"]).sum()),
                "map_path": str(Path("maps") / map_path.name),
                "map_sha256": sha256_file(map_path),
            }
        )
    prediction_sha = _write_csv(prediction_root / "prediction_manifest.csv", prediction_rows)
    score_manifest = write_candidate_score_manifest(score_root, score_rows)
    return prediction_sha, str(score_manifest["manifest_sha256"])


def _loader(
    records: list[S10InputRecord],
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, object]]:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        S10Dataset(records, args.maximum_candidates),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_s10,
        generator=generator,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _seed_everything(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S10 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S10 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    if sha256_file(args.pretrained_checkpoint) != args.expected_pretrained_sha256:
        raise ValueError("S10 ResNet-50 pretrained checkpoint SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    split_rows = {
        split: base.load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != EXPECTED_TRAIN or len(split_rows["val"]) != EXPECTED_VALIDATION:
        raise RuntimeError("S10 frozen cohort mismatch")
    train_candidates, train_candidate_audit = legacy._audit_candidate_input(
        args.train_candidate_root,
        split_rows["train"],
        split="train",
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
    )
    val_candidates, val_candidate_audit = legacy._audit_candidate_input(
        args.val_candidate_root,
        split_rows["val"],
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    cache_freeze, cache_manifest_rows = base._verify_cache_freeze(args)
    accepted = base._load_cache_records(args, split_rows, cache_manifest_rows)
    train_records = build_input_records(
        split_rows["train"],
        accepted["train"],
        train_candidates,
        args.train_candidate_root,
        args.dataset_root,
    )
    val_records = build_input_records(
        split_rows["val"],
        accepted["val"],
        val_candidates,
        args.val_candidate_root,
        args.dataset_root,
    )
    input_manifest_rows = [
        {
            "split": split,
            "image_id": record.image_id,
            "group_id": record.group_id,
            "tumor": record.label,
            "candidate_count": len(record.candidate_indices),
            "candidate_payload_sha256": record.candidate_sha256,
            "candidate_indices_sha256": _array_sha256(record.candidate_indices),
        }
        for split, records in (("train", train_records), ("val", val_records))
        for record in records
    ]
    input_manifest_sha = _write_csv(
        args.output_dir / "input_manifest.csv", input_manifest_rows
    )
    input_gate_sha = _write_json(
        args.output_dir / "input_operational_gate.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "PASS_BEFORE_TRAINING",
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "input_manifest_sha256": input_manifest_sha,
            "cohort": {"train": len(train_records), "validation": len(val_records)},
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )

    state_dict = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    model = HighResProposalMIL(backbone_state_dict=state_dict).to(device)
    del state_dict
    parallel = nn.DataParallel(model, device_ids=(0, 1), output_device=0)
    train_loader = _loader(train_records, args, shuffle=True)
    history, final_state = train_model(parallel, train_loader, args, device)
    history_sha = _write_json(args.output_dir / "training_history.json", {"epochs": history})
    checkpoint_path = args.output_dir / "highres_candidate_pmil.pt"
    checkpoint = {
        "model_state_dict": final_state,
        "architecture": {
            "backbone": "torchvision_resnet50_imagenet1k_v2",
            "input_size": EXPECTED_IMAGE_SIZE,
            "support_size": EXPECTED_SUPPORT_SIZE,
            "fpn_channels": 128,
            "set_hidden_dim": 256,
            "set_heads": 4,
            "set_layers": 2,
            "set_dropout": 0.1,
            "ring_radius": 3,
        },
        "optimizer": {
            "name": "AdamW",
            "schedule": "cosine_to_zero",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "top_dropout_fraction": args.top_dropout_fraction,
            "loss_weights": LOSS_WEIGHTS,
            "seed": args.seed,
            "checkpoint_selection": "final_epoch_only_no_validation_selection",
        },
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "pretrained_sha256": args.expected_pretrained_sha256,
        "input_operational_gate_sha256": input_gate_sha,
        "input_manifest_sha256": input_manifest_sha,
        "training_history_sha256": history_sha,
        "training_labels": "image_level_normal_tumor_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    checkpoint_sha = sha256_file(checkpoint_path)

    val_loader = _loader(val_records, args, shuffle=False)
    learned = score_validation(parallel, val_loader, device)
    del parallel, model, train_loader, val_loader
    torch.cuda.empty_cache()

    baseline_freeze, baseline_rows = base._verify_baseline_freeze(args)
    baseline_model, baseline_config = base._load_baseline_model(args, device=device)
    base_scored = score_same_family_graph_records(
        accepted["val"],
        baseline_model,
        bag_temperature=baseline_config.bag_temperature,
        graph_config=SameFamilyGraphConfig(
            minimum_iou=1.0,
            minimum_containment=1.0,
            alpha=0.0,
            iterations=1,
        ),
        batch_size=16,
        device=device,
    )
    identity_rows = base._baseline_identity(accepted["val"], base_scored, baseline_rows)
    identity_sha = _write_csv(args.output_dir / "baseline_identity.csv", identity_rows)
    del baseline_model
    torch.cuda.empty_cache()

    arms, evidence_sha, diagnostics = compose_arms(
        args.output_dir,
        accepted["val"],
        base_scored,
        learned,
        baseline_rows,
        val_candidates,
        args.val_candidate_root,
    )
    recipes = {
        CONTROL_ARM: "tie_aware_equal_rank_geometry_v3_plus_upstream",
        CAPACITY_ARM: "tie_aware_equal_rank_geometry_v3_plus_upstream_plus_identity",
        PRIMARY_ARM: "pareto_identity_capture_purity_guarded_from_control",
    }
    arm_freezes: dict[str, str] = {}
    for arm_name, scored in arms.items():
        arm_root = args.output_dir / arm_name
        prediction_sha, score_sha = write_validation_outputs(
            arm_root,
            accepted["val"],
            scored,
            recipe=recipes[arm_name],
        )
        arm_freezes[arm_name] = _write_json(
            arm_root / "prediction_freeze.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "arm": arm_name,
                "source_commit": args.source_commit,
                "protocol_sha256": args.protocol_sha256,
                "split_sha256": args.expected_split_sha256,
                "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
                "selector_cache_manifest_sha256": cache_freeze["selector_cache_manifest_sha256"],
                "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
                "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
                "baseline_prediction_manifest_sha256": baseline_freeze["prediction_manifest_sha256"],
                "baseline_identity_sha256": identity_sha,
                "pretrained_sha256": args.expected_pretrained_sha256,
                "s10_checkpoint_sha256": checkpoint_sha,
                "training_history_sha256": history_sha,
                "input_operational_gate_sha256": input_gate_sha,
                "input_manifest_sha256": input_manifest_sha,
                "s10_candidate_evidence_manifest_sha256": evidence_sha,
                "prediction_manifest_sha256": prediction_sha,
                "candidate_score_manifest_sha256": score_sha,
                "validation_predictions": EXPECTED_VALIDATION,
                "training_labels": "image_level_normal_tumor_only",
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
        )
    triple_sha = _write_json(
        args.output_dir / "prediction_triple_freeze.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "arms": arm_freezes,
            "all_arms_physically_frozen_before_validation_gt": True,
            "collaborator_output_accessed": False,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    diagnostics_sha = _write_json(
        args.output_dir / "gt_blind_diagnostics.json",
        {
            "experiment_id": EXPERIMENT_ID,
            **diagnostics,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    run_manifest = {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device_count": 2,
            "cuda_device_names": device_names,
            "model_data_parallel": True,
            "mixed_precision": "float16_amp",
        },
        "cohort": {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION},
        "architecture": checkpoint["architecture"],
        "optimizer": checkpoint["optimizer"],
        "train_candidates": train_candidate_audit,
        "validation_candidates": val_candidate_audit,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "input_operational_gate_sha256": input_gate_sha,
        "input_manifest_sha256": input_manifest_sha,
        "s10_checkpoint_sha256": checkpoint_sha,
        "training_history_sha256": history_sha,
        "baseline_identity_sha256": identity_sha,
        "s10_candidate_evidence_manifest_sha256": evidence_sha,
        "gt_blind_diagnostics_sha256": diagnostics_sha,
        "prediction_triple_freeze_sha256": triple_sha,
        "arms": arm_freezes,
        "training_labels": "image_level_normal_tumor_only",
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
