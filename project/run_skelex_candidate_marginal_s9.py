"""Run the frozen S9 high-resolution candidate-marginal selector.

The runner accepts only radiographs, binary image labels and class-agnostic
candidate masks. Validation segmentation annotations are not an input. Both
prediction arms are physically frozen for a separate post-freeze evaluator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

import run_bas_candidate_descriptor_core as base
import run_rad_dino_mask_bag_mil_probe as legacy
from mae_reconstruction_io import (
    locate_verified_image,
    save_float_map,
    sha256_file,
    verify_model_snapshot,
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
from models.skelex_candidate_marginal import (
    NonlinearTokenEvidenceHead,
    SKELEX_GRID_SIZE,
    SKELEX_HEAD_HIDDEN_DIM,
    SKELEX_HIDDEN_LAYERS,
    SKELEX_HIDDEN_SIZE,
    SKELEX_INPUT_SIZE,
    SKELEX_PATCHES,
    SKELEX_RING_RADIUS,
    SKELEX_TOKEN_DIM,
    SkelexMultiLayerTokenEncoder,
    candidate_marginal_image_label_loss,
    candidate_spatial_log_likelihood,
    finite_readout,
    fractional_candidate_ring_supports,
)


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
RUN_ID = "btxrd_skelex_candidate_marginal_s9_v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_s9_likelihood_equal_rank"
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
EXPECTED_TRAIN = 2981
EXPECTED_VALIDATION = 371
EXPECTED_NORMAL_TRAIN = 1493
EXPECTED_TUMOR_TRAIN = 1488
EXPECTED_ENCODER_BATCH_SIZE = 2
EXPECTED_TRAIN_BATCH_SIZE = 8
EXPECTED_EPOCHS = 32
EXPECTED_LEARNING_RATE = 1.0e-3
EXPECTED_WEIGHT_DECAY = 1.0e-4
EXPECTED_MAXIMUM_CANDIDATES = 81
EXPECTED_SEED = 42
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--skelex-model-dir", type=Path, required=True)
    parser.add_argument("--expected-skelex-config-sha256", required=True)
    parser.add_argument("--expected-skelex-preprocessor-sha256", required=True)
    parser.add_argument("--expected-skelex-weight-sha256", required=True)
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
    parser.add_argument("--encoder-batch-size", type=int, default=EXPECTED_ENCODER_BATCH_SIZE)
    parser.add_argument("--train-batch-size", type=int, default=EXPECTED_TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=EXPECTED_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=EXPECTED_WEIGHT_DECAY)
    parser.add_argument("--maximum-candidates", type=int, default=EXPECTED_MAXIMUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    actual = (
        args.encoder_batch_size,
        args.train_batch_size,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.maximum_candidates,
        args.seed,
    )
    expected = (
        EXPECTED_ENCODER_BATCH_SIZE,
        EXPECTED_TRAIN_BATCH_SIZE,
        EXPECTED_EPOCHS,
        EXPECTED_LEARNING_RATE,
        EXPECTED_WEIGHT_DECAY,
        EXPECTED_MAXIMUM_CANDIDATES,
        EXPECTED_SEED,
    )
    if actual != expected:
        raise ValueError("S9 execution differs from the frozen one-shot recipe")


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((SKELEX_INPUT_SIZE, SKELEX_INPUT_SIZE), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _array_sha256(values: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return sha256(stream.getvalue()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sha256_file(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty S9 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _project_supports(
    masks: np.ndarray,
    *,
    projection: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = torch.from_numpy(np.asarray(masks, dtype=np.float32))
    square_candidates = legacy.project_direct_resize_masks_to_square(
        source,
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=SKELEX_INPUT_SIZE,
    )
    square_content = legacy.project_direct_resize_masks_to_square(
        torch.ones((1, masks.shape[-2], masks.shape[-1]), dtype=torch.float32),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=SKELEX_INPUT_SIZE,
    )[0]
    inside, ring, content_valid = fractional_candidate_ring_supports(
        square_candidates,
        square_content,
        grid_size=SKELEX_GRID_SIZE,
        ring_radius=SKELEX_RING_RADIUS,
    )
    return (
        inside.numpy().astype(np.float16),
        ring.numpy().astype(np.float16),
        content_valid.numpy().astype(np.uint8),
    )


def _validate_candidate_indices(record: dict[str, Any], count: int) -> np.ndarray:
    indices = np.asarray(record["candidate_indices"], dtype=np.int64)
    if (
        indices.ndim != 1
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise RuntimeError("S9 accepted candidate indices are invalid")
    return indices


def build_feature_cache(
    rows: list[dict[str, str]],
    accepted_records: list[dict[str, Any]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
    encoder: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    *,
    split: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float | int]]:
    if len(rows) != len(accepted_records):
        raise ValueError("S9 split/cache cohorts do not align")
    cached: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    inside_masses: list[float] = []
    ring_masses: list[float] = []
    physical_candidates = 0
    retained_candidates = 0
    for start in range(0, len(rows), args.encoder_batch_size):
        batch_rows = rows[start : start + args.encoder_batch_size]
        batch_records = accepted_records[start : start + args.encoder_batch_size]
        if len(batch_rows) != len(batch_records):
            raise RuntimeError("S9 final encoder batch differs")
        pixels: list[torch.Tensor] = []
        payloads: list[tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]] = []
        for row, record in zip(batch_rows, batch_records):
            if row["image_id"] != record["image_id"]:
                raise RuntimeError("S9 accepted cache order differs from frozen split")
            candidate_row = candidate_rows[Path(row["image_id"]).stem]
            if candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]:
                raise RuntimeError("S9 physical/cache candidate provenance mismatch")
            masks, _metadata, _scores, _fallback = legacy._load_candidate_payload(
                candidate_root,
                candidate_row,
                maximum_candidates=args.maximum_candidates,
            )
            indices = _validate_candidate_indices(record, len(masks))
            image_path = locate_verified_image(args.dataset_root, row)
            with Image.open(image_path) as image:
                normalized, projection = _normalized_square(image)
            pixels.append(normalized)
            payloads.append((masks[indices], indices, projection, record))
            physical_candidates += len(masks)
            retained_candidates += len(indices)
        pixel_batch = torch.stack(pixels)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            token_batch = encoder(pixel_batch.to(device, non_blocking=True))
        token_batch = token_batch.float().cpu().numpy().astype(np.float16)
        for offset, (row, payload) in enumerate(zip(batch_rows, payloads)):
            masks, indices, projection, record = payload
            inside, rings, content_valid = _project_supports(masks, projection=projection)
            tokens = token_batch[offset]
            if tokens.shape != (SKELEX_PATCHES, SKELEX_TOKEN_DIM):
                raise RuntimeError("S9 token cache shape mismatch")
            if inside.shape != rings.shape or inside.shape != (len(indices), SKELEX_PATCHES):
                raise RuntimeError("S9 support cache shape mismatch")
            inside_masses.extend(inside.astype(np.float32).sum(axis=1).tolist())
            ring_masses.extend(rings.astype(np.float32).sum(axis=1).tolist())
            item: dict[str, object] = {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "label": int(row["tumor"]),
                "tokens": tokens,
                "candidate_weights": inside,
                "ring_weights": rings,
                "content_valid": content_valid,
                "candidate_indices": indices.astype(np.int32),
                "candidate_payload_sha256": record["candidate_payload_sha256"],
            }
            cached.append(item)
            manifest_rows.append(
                {
                    "split": split,
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "tumor": int(row["tumor"]),
                    "candidate_count": len(indices),
                    "candidate_payload_sha256": record["candidate_payload_sha256"],
                    "candidate_indices_sha256": _array_sha256(item["candidate_indices"]),
                    "tokens_sha256": _array_sha256(tokens),
                    "candidate_weights_sha256": _array_sha256(inside),
                    "ring_weights_sha256": _array_sha256(rings),
                    "content_valid_sha256": _array_sha256(content_valid),
                }
            )
        completed = min(start + len(batch_rows), len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"S9 {split} feature cache: {completed}/{len(rows)}", flush=True)
    if len(cached) != len(rows) or retained_candidates != len(inside_masses):
        raise RuntimeError("S9 feature-cache count mismatch")
    diagnostics: dict[str, float | int] = {
        "images": len(cached),
        "physical_candidates": physical_candidates,
        "exact_retained_candidates": retained_candidates,
        "minimum_inside_mass": float(min(inside_masses)),
        "median_inside_mass": float(np.median(inside_masses)),
        "maximum_inside_mass": float(max(inside_masses)),
        "minimum_ring_mass": float(min(ring_masses)),
        "median_ring_mass": float(np.median(ring_masses)),
        "maximum_ring_mass": float(max(ring_masses)),
        "all_tensors_finite": 1,
        "exact_candidate_set_preserved": 1,
    }
    return cached, manifest_rows, diagnostics


def collate_feature_records(
    records: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("S9 cannot collate an empty batch")
    count = len(records)
    maximum = max(len(np.asarray(record["candidate_indices"])) for record in records)
    tokens = torch.from_numpy(np.stack([np.asarray(record["tokens"]) for record in records]))
    candidates = torch.zeros((count, maximum, SKELEX_PATCHES), dtype=torch.float32)
    rings = torch.zeros_like(candidates)
    candidate_valid = torch.zeros((count, maximum), dtype=torch.bool)
    content_valid = torch.from_numpy(
        np.stack([np.asarray(record["content_valid"], dtype=np.uint8) for record in records])
    ).bool()
    for index, record in enumerate(records):
        size = len(np.asarray(record["candidate_indices"]))
        candidates[index, :size] = torch.from_numpy(
            np.asarray(record["candidate_weights"], dtype=np.float32)
        )
        rings[index, :size] = torch.from_numpy(
            np.asarray(record["ring_weights"], dtype=np.float32)
        )
        candidate_valid[index, :size] = True
    return {
        "tokens": tokens,
        "tumor": torch.tensor([int(record["label"]) for record in records]),
        "candidate_weights": candidates,
        "ring_weights": rings,
        "candidate_valid": candidate_valid,
        "content_valid": content_valid,
    }


def train_head(
    records: list[dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[NonlinearTokenEvidenceHead, list[dict[str, float | int]]]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int8)
    if (
        len(records) != EXPECTED_TRAIN
        or int((labels == 0).sum()) != EXPECTED_NORMAL_TRAIN
        or int((labels == 1).sum()) != EXPECTED_TUMOR_TRAIN
    ):
        raise RuntimeError("S9 training-label cohort mismatch")
    model = NonlinearTokenEvidenceHead().to(device)
    with torch.inference_mode():
        probe = collate_feature_records(records[: args.train_batch_size])
        initial = model(probe["tokens"].to(device))
        if not torch.equal(initial, torch.zeros_like(initial)):
            raise RuntimeError("S9 output layer is not exactly zero initialized")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(records))
        total_sum = 0.0
        normal_sum = 0.0
        tumor_sum = 0.0
        normal_count = 0
        tumor_count = 0
        model.train()
        for start in range(0, len(records), args.train_batch_size):
            indices = order[start : start + args.train_batch_size]
            batch_records = [records[int(index)] for index in indices]
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in collate_feature_records(batch_records).items()
            }
            optimizer.zero_grad(set_to_none=True)
            token_logits = model(batch["tokens"])
            output = candidate_marginal_image_label_loss(
                token_logits,
                batch["tumor"],
                batch["candidate_weights"],
                batch["ring_weights"],
                batch["candidate_valid"],
                batch["content_valid"],
            )
            loss = output["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("S9 training loss became non-finite")
            loss.backward()
            optimizer.step()
            size = len(batch_records)
            batch_normal = int((batch["tumor"] == 0).sum().item())
            batch_tumor = size - batch_normal
            total_sum += float(loss.detach().cpu()) * size
            if batch_normal:
                normal_sum += float(output["normal_dense"].detach().cpu()) * batch_normal
                normal_count += batch_normal
            if batch_tumor:
                tumor_sum += (
                    float(output["tumor_candidate_marginal"].detach().cpu()) * batch_tumor
                )
                tumor_count += batch_tumor
        if normal_count != EXPECTED_NORMAL_TRAIN or tumor_count != EXPECTED_TUMOR_TRAIN:
            raise RuntimeError("S9 epoch label counts changed")
        parameters = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "total_loss": total_sum / len(records),
            "normal_dense_loss": normal_sum / normal_count,
            "tumor_candidate_marginal_loss": tumor_sum / tumor_count,
            "parameter_l2": float(torch.linalg.vector_norm(parameters)),
            "output_weight_l2": float(torch.linalg.vector_norm(model.output.weight.detach().cpu())),
            "output_bias": float(model.output.bias.detach().cpu()),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError("S9 training history became non-finite")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return model.eval(), history


@torch.inference_mode()
def score_likelihoods(
    model: NonlinearTokenEvidenceHead,
    records: list[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    result: list[dict[str, object]] = []
    for record in records:
        batch = collate_feature_records([record])
        token_logits = model(batch["tokens"].to(device))
        likelihood = candidate_spatial_log_likelihood(
            token_logits,
            batch["candidate_weights"].to(device),
            batch["ring_weights"].to(device),
            batch["candidate_valid"].to(device),
        )[0]
        result.append(
            {
                "image_id": record["image_id"],
                "token_logits": token_logits[0].float().cpu().numpy(),
                "candidate_likelihood": likelihood.float().cpu().numpy(),
            }
        )
    return result


def compose_pair(
    output_dir: Path,
    accepted_records: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    likelihood_scored: list[dict[str, object]],
    feature_records: list[dict[str, object]],
    baseline_rows: list[dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float | int]]:
    lengths = {
        len(accepted_records), len(base_scored), len(likelihood_scored), len(feature_records)
    }
    if lengths != {EXPECTED_VALIDATION}:
        raise RuntimeError("S9 validation evidence cohorts do not align")
    accepted_predictions = {row["image_id"]: row for row in baseline_rows}
    evidence_root = output_dir / "s9_likelihood_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    arms: dict[str, list[dict[str, Any]]] = {CONTROL_ARM: [], PRIMARY_ARM: []}
    correlations: list[float] = []
    changed = 0
    iterator = zip(accepted_records, base_scored, likelihood_scored, feature_records)
    for index, (record, baseline, semantic, feature) in enumerate(iterator):
        image_id = str(record["image_id"])
        if not (
            image_id == baseline["image_id"] == semantic["image_id"] == feature["image_id"]
        ):
            raise RuntimeError("S9 validation score order mismatch")
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        base_logits = np.asarray(baseline["base_candidate_logits"], dtype=np.float32)
        likelihood = np.asarray(semantic["candidate_likelihood"], dtype=np.float32)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]
        ):
            raise RuntimeError("S9 upstream candidate provenance mismatch")
        with np.load(candidate_path, allow_pickle=False) as payload:
            upstream_all = np.asarray(payload["selection_scores"], dtype=np.float32)
        upstream = upstream_all[indices]
        scores = finite_readout(base_logits, upstream, likelihood)
        valid = torch.ones((1, len(indices)), dtype=torch.bool)
        reference_control = base.equal_rank_aggregate(
            (
                torch.from_numpy(base_logits)[None],
                torch.from_numpy(upstream)[None],
            ),
            valid,
        )[0].numpy()
        if not np.array_equal(scores["control"].astype(np.float32), reference_control):
            raise RuntimeError("S9 control does not reproduce the accepted two-rank recipe")
        if len(indices) > 1:
            value = float(np.corrcoef(scores["control"], scores["primary"])[0, 1])
            if np.isfinite(value):
                correlations.append(value)
        changed += int(int(np.argmax(scores["control"])) != int(np.argmax(scores["primary"])))
        relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        evidence_path = evidence_root / relative
        np.savez_compressed(
            evidence_path,
            candidate_indices=indices.astype(np.int32),
            baseline_logits=base_logits,
            upstream_scores=upstream,
            token_logits=np.asarray(semantic["token_logits"], dtype=np.float32),
            candidate_weights=np.asarray(feature["candidate_weights"], dtype=np.float16),
            ring_weights=np.asarray(feature["ring_weights"], dtype=np.float16),
            content_valid=np.asarray(feature["content_valid"], dtype=np.uint8),
            candidate_likelihood=likelihood,
            control_rank=scores["control"].astype(np.float32),
            primary_rank=scores["primary"].astype(np.float32),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_count": len(indices),
                "evidence_path": str(relative),
                "evidence_sha256": sha256_file(evidence_path),
                "tokens_sha256": _array_sha256(np.asarray(feature["tokens"])),
                "candidate_likelihood_sha256": _array_sha256(likelihood),
            }
        )
        accepted_row = accepted_predictions[image_id]
        common = {
            "image_id": image_id,
            "bag_logit": float(accepted_row["bag_logit"]),
            "bag_probability": float(accepted_row["bag_probability"]),
        }
        arms[CONTROL_ARM].append(
            {**common, "candidate_logits": scores["control"].astype(np.float32)}
        )
        arms[PRIMARY_ARM].append(
            {**common, "candidate_logits": scores["primary"].astype(np.float32)}
        )
    if not correlations:
        raise RuntimeError("S9 primary/control correlation is undefined")
    diagnostics: dict[str, float | int] = {
        "mean_primary_control_rank_correlation": float(np.mean(correlations)),
        "correlation_images": len(correlations),
        "primary_changed_selections": changed,
        "primary_changed_selection_fraction": changed / EXPECTED_VALIDATION,
    }
    return arms, _write_csv(evidence_root / "evidence_manifest.csv", evidence_rows), diagnostics


def write_validation_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> tuple[str, str]:
    if len(records) != len(scored) or len(records) != EXPECTED_VALIDATION:
        raise ValueError("S9 output records/scores do not align")
    prediction_root = output_dir / "predictions"
    map_root = prediction_root / "maps"
    score_root = output_dir / "candidate_scores"
    score_payload_root = score_root / "scores"
    map_root.mkdir(parents=True, exist_ok=False)
    score_payload_root.mkdir(parents=True, exist_ok=False)
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index, (record, prediction) in enumerate(zip(records, scored)):
        if record["image_id"] != prediction["image_id"]:
            raise RuntimeError("S9 validation scoring order differs from cache")
        candidate_indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        logits = np.asarray(prediction["candidate_logits"], dtype=np.float32)
        if logits.shape != candidate_indices.shape or not np.isfinite(logits).all():
            raise RuntimeError("S9 validation score vector is invalid")
        stem = f"{index:04d}_{Path(str(record['image_id'])).stem}"
        score_relative = Path("scores") / f"{stem}.npz"
        saved_score = save_candidate_score_evidence(
            score_root / score_relative,
            candidate_indices=candidate_indices,
            candidate_logits=logits,
        )
        masks = base.unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        local_winner = int(np.argmax(logits))
        original_winner = int(candidate_indices[local_winner])
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
                "candidate_count": len(candidate_indices),
                "selected_candidate_index": original_winner,
                "selected_candidate_logit": saved_score["selected_candidate_logit"],
                "candidate_logit_recipe": "within_image_equal_percentile_rank_no_tta",
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


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    legacy.seed_everything(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S9 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S9 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    model_snapshot = verify_model_snapshot(
        args.skelex_model_dir,
        expected_config_sha256=args.expected_skelex_config_sha256,
        expected_preprocessor_sha256=args.expected_skelex_preprocessor_sha256,
        expected_weight_sha256=args.expected_skelex_weight_sha256,
    )
    split_rows = {
        split: base.load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != EXPECTED_TRAIN or len(split_rows["val"]) != EXPECTED_VALIDATION:
        raise RuntimeError("S9 frozen cohort mismatch")
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

    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("S9 transformers version mismatch")
    pretrained = ViTMAEForPreTraining.from_pretrained(
        args.skelex_model_dir,
        local_files_only=True,
    )
    pretrained.vit.config.mask_ratio = 0.0
    pretrained.vit.requires_grad_(False).eval()
    encoder: nn.Module = SkelexMultiLayerTokenEncoder(pretrained.vit).to(device)
    del pretrained
    encoder = nn.DataParallel(encoder, device_ids=(0, 1), output_device=0).eval()
    train_cache, train_manifest, train_gate = build_feature_cache(
        split_rows["train"],
        accepted["train"],
        train_candidates,
        args.train_candidate_root,
        encoder,
        args,
        device,
        split="train",
    )
    val_cache, val_manifest, val_gate = build_feature_cache(
        split_rows["val"],
        accepted["val"],
        val_candidates,
        args.val_candidate_root,
        encoder,
        args,
        device,
        split="val",
    )
    del encoder
    torch.cuda.empty_cache()
    feature_manifest_sha = _write_csv(
        args.output_dir / "feature_cache_manifest.csv",
        train_manifest + val_manifest,
    )
    feature_gate = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_BEFORE_HEAD_TRAINING",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "feature_cache_manifest_sha256": feature_manifest_sha,
        "train": train_gate,
        "validation": val_gate,
        "image_input_size": SKELEX_INPUT_SIZE,
        "token_grid_size": SKELEX_GRID_SIZE,
        "hidden_layers": list(SKELEX_HIDDEN_LAYERS),
        "token_dim": SKELEX_TOKEN_DIM,
        "training_labels": "image_level_normal_tumor_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    feature_gate_path = args.output_dir / "feature_cache_operational_gate.json"
    feature_gate_sha = _write_json(feature_gate_path, feature_gate)

    head, history = train_head(train_cache, args, device)
    history_path = args.output_dir / "training_history.json"
    history_sha = _write_json(history_path, {"epochs": history})
    checkpoint_path = args.output_dir / "skelex_candidate_marginal_head.pt"
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "architecture": {
            "input_size": SKELEX_INPUT_SIZE,
            "grid_size": SKELEX_GRID_SIZE,
            "hidden_layers": list(SKELEX_HIDDEN_LAYERS),
            "hidden_size": SKELEX_HIDDEN_SIZE,
            "token_dim": SKELEX_TOKEN_DIM,
            "head_hidden_dim": SKELEX_HEAD_HIDDEN_DIM,
            "ring_radius": SKELEX_RING_RADIUS,
        },
        "optimizer": {
            "name": "AdamW",
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "checkpoint_selection": "final_epoch_only_no_validation_selection",
        },
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "skelex_weight_sha256": args.expected_skelex_weight_sha256,
        "feature_cache_manifest_sha256": feature_manifest_sha,
        "training_history_sha256": history_sha,
        "training_labels": "image_level_normal_tumor_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    checkpoint_sha = sha256_file(checkpoint_path)
    likelihood_scored = score_likelihoods(head, val_cache, device)
    del head
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

    arms, evidence_sha, pair_diagnostics = compose_pair(
        args.output_dir,
        accepted["val"],
        base_scored,
        likelihood_scored,
        val_cache,
        baseline_rows,
        val_candidates,
        args.val_candidate_root,
    )
    arm_freezes: dict[str, str] = {}
    for arm_name, scores in arms.items():
        arm_root = args.output_dir / arm_name
        prediction_sha, score_sha = write_validation_outputs(arm_root, accepted["val"], scores)
        freeze = {
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
            "skelex_public_weight_sha256": args.expected_skelex_weight_sha256,
            "s9_checkpoint_sha256": checkpoint_sha,
            "training_history_sha256": history_sha,
            "feature_cache_operational_gate_sha256": feature_gate_sha,
            "feature_cache_manifest_sha256": feature_manifest_sha,
            "s9_likelihood_evidence_manifest_sha256": evidence_sha,
            "prediction_manifest_sha256": prediction_sha,
            "candidate_score_manifest_sha256": score_sha,
            "validation_predictions": EXPECTED_VALIDATION,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        arm_freezes[arm_name] = _write_json(freeze_path, freeze)
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_sha = _write_json(
        pair_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "arms": arm_freezes,
            "pair_physically_frozen_before_validation_gt": True,
            "collaborator_output_accessed": False,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
    )
    diagnostics_path = args.output_dir / "gt_blind_diagnostics.json"
    diagnostics_sha = _write_json(
        diagnostics_path,
        {
            "experiment_id": EXPERIMENT_ID,
            **pair_diagnostics,
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
            "transformers": EXPECTED_TRANSFORMERS_VERSION,
            "cuda_device_count": 2,
            "cuda_device_names": device_names,
            "encoder_data_parallel": True,
        },
        "cohort": {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION},
        "skelex_model_snapshot": model_snapshot,
        "architecture": checkpoint["architecture"],
        "optimizer": checkpoint["optimizer"],
        "train_candidates": train_candidate_audit,
        "validation_candidates": val_candidate_audit,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "feature_cache_operational_gate_sha256": feature_gate_sha,
        "feature_cache_manifest_sha256": feature_manifest_sha,
        "s9_checkpoint_sha256": checkpoint_sha,
        "training_history_sha256": history_sha,
        "baseline_identity_sha256": identity_sha,
        "s9_likelihood_evidence_manifest_sha256": evidence_sha,
        "gt_blind_diagnostics_sha256": diagnostics_sha,
        "prediction_pair_freeze_sha256": pair_sha,
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
