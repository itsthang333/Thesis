from __future__ import annotations

"""Train and freeze the matched rich-gallery G2 selector arms.

The runner consumes images, binary image labels and hash-frozen proposal bags.
It has no segmentation-dataset import and never opens validation annotations.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
    verify_model_snapshot,
)
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    self_guided_instance_loss,
    smooth_mil_pool,
)
from models.rich_gallery_g2_objective import (
    geometric_continuation_temperature,
    hierarchical_source_candidate_weights,
    hierarchical_source_smooth_pool,
    negative_bag_instance_loss,
    rank_fusion_scores,
    shared_source_validity,
    stable_select,
)
from run_rad_dino_mask_bag_mil_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    _audit_candidate_input,
    build_descriptor_cache,
    seed_everything,
)


ARM_NAMES = (
    "flat_shared_hardtop",
    "flat_shared_negative_only",
    "hierarchical_shared_negative_only",
)
MODEL_NAMES = ("g1_frozen", *ARM_NAMES)
SOURCE_TO_ID = {"classifier448": 0, "layercam320": 1, "external_saliency": 2}
ID_TO_SOURCE = {value: key for key, value in SOURCE_TO_ID.items()}


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
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--g1-prediction-manifest", type=Path, required=True)
    parser.add_argument("--expected-g1-prediction-manifest-sha256", required=True)
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
    parser.add_argument("--maximum-candidates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown proposal source: {value!r}")


def attach_rich_metadata(
    cache: list[dict[str, object]],
    candidate_rows: Mapping[str, Mapping[str, str]],
    candidate_root: Path,
) -> dict[str, object]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    presence: dict[str, Counter[str]] = defaultdict(Counter)
    bag_counts: dict[str, list[int]] = defaultdict(list)
    for record in cache:
        image_id = str(record["image_id"])
        row = candidate_rows[Path(image_id).stem]
        path = candidate_root / row["diagnostic_path"]
        if sha256_file(path) != row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(path, allow_pickle=False) as payload:
            upstream = payload["selection_scores"].astype(np.float32).reshape(-1)
            source_names = np.asarray(
                [canonical_source(value) for value in payload["proposal_source_ids"]],
                dtype="U32",
            )
            candidate_count = int(payload["sam_masks"].shape[0])
        if len(upstream) != candidate_count or len(source_names) != candidate_count:
            raise ValueError(f"rich proposal metadata misaligned: {image_id}")
        kept = np.asarray(record["kept_indices"], dtype=np.int64)
        if not len(kept) or int(kept.min()) < 0 or int(kept.max()) >= candidate_count:
            raise ValueError(f"kept indices invalid: {image_id}")
        kept_names = source_names[kept]
        source_ids = np.asarray([SOURCE_TO_ID[name] for name in kept_names], dtype=np.int16)
        kept_upstream = upstream[kept]
        if not np.isfinite(kept_upstream).all():
            raise ValueError(f"upstream scores are non-finite: {image_id}")
        record["source_ids"] = source_ids
        record["source_names"] = kept_names
        record["upstream_scores"] = kept_upstream
        label_name = "tumor" if int(record["label"]) else "normal"
        counts[label_name].update(kept_names.tolist())
        presence[label_name].update(set(kept_names.tolist()))
        bag_counts[label_name].append(len(kept))
    report: dict[str, object] = {}
    for label_name in ("normal", "tumor"):
        values = np.asarray(bag_counts[label_name], dtype=np.float64)
        report[label_name] = {
            "images": len(values),
            "candidate_count_mean": float(values.mean()),
            "candidate_count_median": float(np.median(values)),
            "candidate_counts_by_source": dict(sorted(counts[label_name].items())),
            "source_presence_images": dict(sorted(presence[label_name].items())),
        }
    external_normal = int(presence["normal"].get("external_saliency", 0))
    external_tumor = int(presence["tumor"].get("external_saliency", 0))
    report["external_source_label_shortcut"] = {
        "normal_presence": external_normal,
        "tumor_presence": external_tumor,
        "confirmed": external_normal == 0 and external_tumor > 0,
    }
    return report


def padded_rich_batch(
    cache: list[dict[str, object]],
    indices: np.ndarray,
    descriptor_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    records = [cache[int(index)] for index in indices]
    maximum = max(len(np.asarray(record[descriptor_key])) for record in records)
    dimension = int(np.asarray(records[0][descriptor_key]).shape[1])
    descriptors = np.zeros((len(records), maximum, dimension), dtype=np.float32)
    valid = np.zeros((len(records), maximum), dtype=bool)
    labels = np.zeros(len(records), dtype=np.float32)
    sources = np.full((len(records), maximum), -1, dtype=np.int64)
    for row_index, record in enumerate(records):
        values = np.asarray(record[descriptor_key], dtype=np.float32)
        source_ids = np.asarray(record["source_ids"], dtype=np.int64)
        if len(values) != len(source_ids):
            raise ValueError("descriptor/source arrays differ")
        descriptors[row_index, : len(values)] = values
        valid[row_index, : len(values)] = True
        sources[row_index, : len(values)] = source_ids
        labels[row_index] = float(record["label"])
    return (
        torch.from_numpy(descriptors).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(sources).to(device),
    )


def _bag_logits_for_arm(
    name: str,
    candidate_logits: torch.Tensor,
    shared_valid: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if name in {"flat_shared_hardtop", "flat_shared_negative_only"}:
        return smooth_mil_pool(
            candidate_logits,
            shared_valid,
            temperature=0.2,
        )
    if name == "hierarchical_shared_negative_only":
        values, _diagnostic = hierarchical_source_smooth_pool(
            candidate_logits,
            shared_valid,
            source_ids,
            temperature=temperature,
        )
        return values
    raise ValueError(f"unknown G2 arm: {name}")


def _effective_count(
    logits: torch.Tensor,
    valid: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    temperature: float,
    hierarchical: bool,
) -> float:
    if hierarchical:
        weights = hierarchical_source_candidate_weights(
            logits,
            valid,
            source_ids,
            temperature=temperature,
        )
    else:
        masked = (logits / temperature).masked_fill(~valid, -torch.inf)
        weights = torch.softmax(masked, dim=1)
    values = 1.0 / weights.square().sum(dim=1).clamp_min(1.0e-12)
    return float(values.mean().detach().item())


def train_arm(
    name: str,
    cache: list[dict[str, object]],
    config: MaskBagMILConfig,
    initial_state: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RadDinoMaskBagMIL, list[dict[str, float]]]:
    if name not in ARM_NAMES:
        raise ValueError(f"unknown G2 arm: {name}")
    seed_everything(args.seed)
    model = RadDinoMaskBagMIL(config).to(device)
    model.load_state_dict(initial_state, strict=True)
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
        temperature = (
            geometric_continuation_temperature(epoch, args.epochs)
            if name == "hierarchical_shared_negative_only"
            else 0.2
        )
        sums = {
            "total": 0.0,
            "image": 0.0,
            "instance": 0.0,
            "consistency": 0.0,
            "effective_candidates": 0.0,
        }
        batches = 0
        for start in range(0, len(order), args.train_batch_size):
            indices = order[start : start + args.train_batch_size]
            descriptors, valid, labels, sources = padded_rich_batch(
                cache, indices, "descriptors", device
            )
            flipped, flipped_valid, _flip_labels, flip_sources = padded_rich_batch(
                cache, indices, "flipped_descriptors", device
            )
            if not torch.equal(valid, flipped_valid) or not torch.equal(sources, flip_sources):
                raise RuntimeError("original/flip rich metadata differs")
            shared = shared_source_validity(valid, sources)
            logits, _unused = model.score_descriptors(descriptors, valid)
            flip_logits, _unused_flip = model.score_descriptors(flipped, valid)
            bag_logits = _bag_logits_for_arm(
                name, logits, shared, sources, temperature=temperature
            )
            flip_bag_logits = _bag_logits_for_arm(
                name, flip_logits, shared, sources, temperature=temperature
            )
            image_loss = 0.5 * (
                image_bag_loss(bag_logits, labels)
                + image_bag_loss(flip_bag_logits, labels)
            )
            if epoch > args.instance_warmup_epochs:
                if name == "flat_shared_hardtop":
                    instance_loss = 0.5 * (
                        self_guided_instance_loss(logits, shared, labels)
                        + self_guided_instance_loss(flip_logits, shared, labels)
                    )
                else:
                    instance_loss = 0.5 * (
                        negative_bag_instance_loss(logits, shared, labels)
                        + negative_bag_instance_loss(flip_logits, shared, labels)
                    )
            else:
                instance_loss = logits.sum() * 0.0
            consistency = aligned_candidate_consistency_loss(logits, flip_logits, shared)
            total = (
                image_loss
                + args.instance_loss_weight * instance_loss
                + args.consistency_loss_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["total"] += float(total.detach().item())
            sums["image"] += float(image_loss.detach().item())
            sums["instance"] += float(instance_loss.detach().item())
            sums["consistency"] += float(consistency.detach().item())
            sums["effective_candidates"] += _effective_count(
                logits.detach(),
                shared,
                sources,
                temperature=temperature,
                hierarchical=name == "hierarchical_shared_negative_only",
            )
            batches += 1
        record = {
            "epoch": float(epoch),
            "temperature": float(temperature),
            **{key: value / batches for key, value in sums.items()},
        }
        history.append(record)
        print(json.dumps({"arm": name, **record}, sort_keys=True), flush=True)
    return model, history


def score_model(
    name: str,
    model: RadDinoMaskBagMIL,
    cache: list[dict[str, object]],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    model.eval()
    scored: dict[str, dict[str, object]] = {}
    for record in cache:
        original = torch.from_numpy(
            np.asarray(record["descriptors"], dtype=np.float32)
        )[None].to(device)
        flipped = torch.from_numpy(
            np.asarray(record["flipped_descriptors"], dtype=np.float32)
        )[None].to(device)
        valid = torch.ones(original.shape[:2], dtype=torch.bool, device=device)
        sources = torch.from_numpy(
            np.asarray(record["source_ids"], dtype=np.int64)
        )[None].to(device)
        with torch.inference_mode():
            original_logits, _ = model.score_descriptors(original, valid)
            flipped_logits, _ = model.score_descriptors(flipped, valid)
            logits = 0.5 * (original_logits + flipped_logits)
            if name == "g1_frozen":
                bag_logits = smooth_mil_pool(logits, valid, temperature=0.2)
            else:
                shared = shared_source_validity(valid, sources)
                bag_logits = _bag_logits_for_arm(
                    name,
                    logits,
                    shared,
                    sources,
                    temperature=0.2,
                )
        values = logits[0].float().cpu().numpy().astype(np.float64)
        upstream = np.asarray(record["upstream_scores"], dtype=np.float64)
        raw_local = stable_select(values, values)
        fused = rank_fusion_scores(values, upstream)
        fused_local = stable_select(fused, values)
        kept = np.asarray(record["kept_indices"], dtype=np.int64)
        scored[str(record["image_id"])] = {
            "candidate_logits": values.astype(np.float32),
            "raw_local_index": raw_local,
            "fusion_local_index": fused_local,
            "raw_candidate_index": int(kept[raw_local]),
            "fusion_candidate_index": int(kept[fused_local]),
            "fusion_score": float(fused[fused_local]),
            "bag_logit": float(bag_logits.item()),
            "bag_probability": float(torch.sigmoid(bag_logits).item()),
        }
    return scored


def load_expected_g1_choices(path: Path, expected_sha256: str) -> dict[str, int]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("G1 prediction manifest SHA-256 mismatch")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: int(row["selected_candidate_index"]) for row in rows}
    if len(indexed) != 371:
        raise ValueError("G1 prediction manifest must contain 371 unique images")
    return indexed


def freeze_scores_and_choices(
    args: argparse.Namespace,
    val_cache: list[dict[str, object]],
    scored_models: Mapping[str, Mapping[str, Mapping[str, object]]],
    checkpoints: Mapping[str, Mapping[str, object]],
    histories: Mapping[str, list[dict[str, float]]],
    *,
    initial_state_sha256: str,
    g1_reproduction_max_index_delta: int,
    train_source_report: Mapping[str, object],
    val_source_report: Mapping[str, object],
    model_snapshot: Mapping[str, object],
    projection_hash: str,
) -> Path:
    score_root = args.output_dir / "stage_a_scores"
    score_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    score_hashes: list[str] = []
    for record in val_cache:
        image_id = str(record["image_id"])
        stem = Path(image_id).stem
        payload: dict[str, np.ndarray] = {
            "candidate_indices": np.asarray(record["kept_indices"], dtype=np.int32),
            "source_ids": np.asarray(record["source_ids"], dtype=np.int16),
            "upstream_scores": np.asarray(record["upstream_scores"], dtype=np.float32),
        }
        for name in MODEL_NAMES:
            payload[f"{name}_candidate_logits"] = np.asarray(
                scored_models[name][image_id]["candidate_logits"], dtype=np.float32
            )
        score_path = score_root / f"{stem}.npz"
        np.savez_compressed(score_path, **payload)
        score_hash = sha256_file(score_path)
        score_hashes.append(score_hash)
        sources = np.asarray(record["source_names"], dtype=str)
        for model_name in MODEL_NAMES:
            result = scored_models[model_name][image_id]
            for rule, local_key, candidate_key in (
                ("raw", "raw_local_index", "raw_candidate_index"),
                ("rank_fusion", "fusion_local_index", "fusion_candidate_index"),
            ):
                local_index = int(result[local_key])
                rows.append(
                    {
                        "image_id": image_id,
                        "group_id": record["group_id"],
                        "tumor": int(record["label"]),
                        "model": model_name,
                        "rule": rule,
                        "variant": f"{model_name}__{rule}",
                        "candidate_payload_sha256": record["candidate_payload_sha256"],
                        "candidate_count": len(payload["candidate_indices"]),
                        "selected_local_index": local_index,
                        "selected_candidate_index": int(result[candidate_key]),
                        "selected_source": str(sources[local_index]),
                        "selected_raw_logit": float(result["candidate_logits"][local_index]),
                        "selected_fusion_score": (
                            float(result["fusion_score"]) if rule == "rank_fusion" else ""
                        ),
                        "bag_logit": float(result["bag_logit"]),
                        "bag_probability": float(result["bag_probability"]),
                        "score_path": str(score_path.relative_to(args.output_dir)).replace("\\", "/"),
                        "score_sha256": score_hash,
                    }
                )
    manifest_path = args.output_dir / "stage_a_selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_hashes: dict[str, str] = {}
    history_hashes: dict[str, str] = {}
    checkpoint_root = args.output_dir / "checkpoints"
    history_root = args.output_dir / "training_history"
    checkpoint_root.mkdir()
    history_root.mkdir()
    for name in ARM_NAMES:
        checkpoint_path = checkpoint_root / f"{name}.pt"
        torch.save(checkpoints[name], checkpoint_path)
        checkpoint_hashes[name] = sha256_file(checkpoint_path)
        history_path = history_root / f"{name}.json"
        history_path.write_text(
            json.dumps(histories[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        history_hashes[name] = sha256_file(history_path)
    freeze = {
        "stage": "rich_gallery_g2_selector_pair_stage_a_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_hash,
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "g1_prediction_manifest_sha256": args.expected_g1_prediction_manifest_sha256,
        "g1_reproduction_max_selected_index_delta": g1_reproduction_max_index_delta,
        "train_candidate_manifest_sha256": args.train_candidate_manifest_sha256,
        "train_pseudo_manifest_sha256": args.train_pseudo_manifest_sha256,
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "initial_state_sha256": initial_state_sha256,
        "arm_checkpoint_sha256": checkpoint_hashes,
        "training_history_sha256": history_hashes,
        "selection_manifest_sha256": sha256_file(manifest_path),
        "score_set_sha256": (
            sha256_file(manifest_path)
            if not score_hashes
            else hashlib.sha256("\n".join(sorted(score_hashes)).encode()).hexdigest()
        ),
        "validation_images": 371,
        "variants": [f"{model}__{rule}" for model in MODEL_NAMES for rule in ("raw", "rank_fusion")],
        "selection_rows": len(rows),
        "train_source_report": train_source_report,
        "validation_source_report": val_source_report,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze_path


def main() -> None:
    args = parse_args()
    if (
        args.input_size != 448
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.encoder_batch_size != 4
        or args.train_batch_size != 16
        or args.epochs != 16
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.instance_loss_weight != 0.25
        or args.consistency_loss_weight != 0.10
        or args.instance_warmup_epochs != 2
        or args.seed != 42
        or not 1 <= args.maximum_candidates <= 243
    ):
        raise ValueError("G2 execution differs from the frozen contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("G2 output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("canonical train/validation cohort mismatch")
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
    if sha256_file(args.g1_checkpoint) != args.expected_g1_checkpoint_sha256:
        raise ValueError("G1 checkpoint hash mismatch")

    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("unexpected transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("G2 requires exactly T4 x2")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"G2 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    projection = make_seeded_random_projection(
        input_dim=768, output_dim=args.projection_dim, seed=args.projection_seed
    )
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder: nn.Module = ProjectedMultiLayerEncoder(
        backbone, torch.from_numpy(projection)
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
    train_source_report = attach_rich_metadata(
        train_cache, train_candidates, args.train_candidate_root
    )
    val_source_report = attach_rich_metadata(
        val_cache, val_candidates, args.val_candidate_root
    )
    if train_source_report["external_source_label_shortcut"]["confirmed"] is not True:
        raise RuntimeError("predeclared external-source shortcut was not reproduced")

    seed_everything(args.seed)
    initial_model = RadDinoMaskBagMIL(config)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_model.state_dict().items()
    }
    initial_path = args.output_dir / "matched_initial_state.pt"
    torch.save(initial_state, initial_path)
    initial_state_sha256 = sha256_file(initial_path)
    del initial_model

    trained_models: dict[str, RadDinoMaskBagMIL] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints: dict[str, dict[str, object]] = {}
    for name in ARM_NAMES:
        model, history = train_arm(
            name, train_cache, config, initial_state, args, device
        )
        trained_models[name] = model
        histories[name] = history
        checkpoints[name] = {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "config": asdict(config),
            "arm": name,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "final_epoch": args.epochs,
            "training_labels": "binary_image_labels_only",
            "external_candidates_used_in_training_loss": False,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }

    g1_payload = torch.load(args.g1_checkpoint, map_location="cpu", weights_only=False)
    if MaskBagMILConfig(**g1_payload["config"]) != config:
        raise ValueError("G1 checkpoint configuration differs from G2")
    g1_model = RadDinoMaskBagMIL(config).to(device)
    g1_model.load_state_dict(g1_payload["model_state_dict"], strict=True)
    scored_models: dict[str, dict[str, dict[str, object]]] = {
        "g1_frozen": score_model("g1_frozen", g1_model, val_cache, device)
    }
    for name in ARM_NAMES:
        scored_models[name] = score_model(name, trained_models[name], val_cache, device)
    expected_g1 = load_expected_g1_choices(
        args.g1_prediction_manifest,
        args.expected_g1_prediction_manifest_sha256,
    )
    index_deltas = [
        abs(int(scored_models["g1_frozen"][image_id]["raw_candidate_index"]) - expected)
        for image_id, expected in expected_g1.items()
    ]
    maximum_g1_index_delta = max(index_deltas)
    if maximum_g1_index_delta != 0:
        raise RuntimeError("G1 checkpoint/cache selected-index reproduction failed")

    freeze_path = freeze_scores_and_choices(
        args,
        val_cache,
        scored_models,
        checkpoints,
        histories,
        initial_state_sha256=initial_state_sha256,
        g1_reproduction_max_index_delta=maximum_g1_index_delta,
        train_source_report=train_source_report,
        val_source_report=val_source_report,
        model_snapshot=model_snapshot,
        projection_hash=projection_sha256(projection),
    )
    run_manifest = {
        "stage": "rich_gallery_g2_selector_pair_training_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "arms": list(ARM_NAMES),
        "matched_variables": [
            "descriptor_cache",
            "shared-source training cohort",
            "MLP architecture and initial state",
            "batch order",
            "optimizer",
            "epochs",
            "flip consistency",
            "validation cohort",
        ],
        "training": {
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "instance_loss_weight": args.instance_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "instance_warmup_epochs": args.instance_warmup_epochs,
            "final_epoch_only": True,
        },
        "candidate_inputs": {"train": train_audit, "validation": val_audit},
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_names": device_names,
            "encoder_data_parallel": True,
            "selector_training_device": device_names[0],
        },
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
