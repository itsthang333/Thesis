from __future__ import annotations

"""Run the frozen SKELEX same-gallery selector representation ablation.

Only full radiographs, binary image-level labels and class-agnostic proposals
are consumed. Validation segmentation annotations are not accepted by this
runner; both arms are physically frozen for a separate evaluator.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import run_bas_candidate_descriptor_core as base
import run_rad_dino_mask_bag_mil_probe as legacy
from mae_reconstruction_io import locate_verified_image, sha256_file, verify_model_snapshot
from models.mae_reconstruction import pad_to_square
from models.mask_bag_same_family_graph import (
    SameFamilyGraphConfig,
    score_same_family_graph_records,
)
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL, smooth_mil_pool
from models.skelex_mask_bag_descriptor import (
    SELECTED_HIDDEN_LAYERS,
    SkelexDescriptorConfig,
    SkelexProjectedMultiLayerEncoder,
    exact_fractional_mask_pool_descriptors,
)


EXPERIMENT_ID = "EXP-20260802-codex-s5-skelex-selector-v1"
RUN_ID = "btxrd_skelex_mask_bag_selector_s5_v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_skelex_equal_rank"
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
EXPECTED_INPUT_SIZE = 224
EXPECTED_PROJECTION_DIM = 128
EXPECTED_PROJECTION_SEED = 42
EXPECTED_EPOCHS = 16
EXPECTED_TRAIN_BATCH_SIZE = 16
EXPECTED_LEARNING_RATE = 3.0e-4
EXPECTED_WEIGHT_DECAY = 1.0e-4
EXPECTED_INSTANCE_WEIGHT = 0.25
EXPECTED_CONSISTENCY_WEIGHT = 0.10
EXPECTED_WARMUP_EPOCHS = 2
EXPECTED_MAXIMUM_CANDIDATES = 81
GEOMETRY_BRIDGE_SIZE = 128
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
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=EXPECTED_TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=EXPECTED_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=EXPECTED_WEIGHT_DECAY)
    parser.add_argument("--instance-loss-weight", type=float, default=EXPECTED_INSTANCE_WEIGHT)
    parser.add_argument("--consistency-loss-weight", type=float, default=EXPECTED_CONSISTENCY_WEIGHT)
    parser.add_argument("--instance-warmup-epochs", type=int, default=EXPECTED_WARMUP_EPOCHS)
    parser.add_argument("--maximum-candidates", type=int, default=EXPECTED_MAXIMUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    actual = (
        args.train_batch_size,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.instance_loss_weight,
        args.consistency_loss_weight,
        args.instance_warmup_epochs,
        args.maximum_candidates,
        args.seed,
    )
    expected = (
        EXPECTED_TRAIN_BATCH_SIZE,
        EXPECTED_EPOCHS,
        EXPECTED_LEARNING_RATE,
        EXPECTED_WEIGHT_DECAY,
        EXPECTED_INSTANCE_WEIGHT,
        EXPECTED_CONSISTENCY_WEIGHT,
        EXPECTED_WARMUP_EPOCHS,
        EXPECTED_MAXIMUM_CANDIDATES,
        42,
    )
    if actual != expected or args.encoder_batch_size < 2:
        raise ValueError("S5 execution differs from the frozen one-shot recipe")


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _pool_exact_bag(
    token_maps: torch.Tensor,
    projected_masks: np.ndarray,
    candidate_metadata: np.ndarray,
    content_mask: np.ndarray,
    descriptor_config: SkelexDescriptorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(projected_masks)
    descriptors, valid, mass = exact_fractional_mask_pool_descriptors(
        token_maps[None].float(),
        torch.from_numpy(projected_masks)[None],
        torch.from_numpy(candidate_metadata)[None],
        torch.ones((1, count), dtype=torch.bool),
        descriptor_config,
        content_masks=torch.from_numpy(content_mask)[None],
    )
    if not valid.all() or not (mass[valid] > descriptor_config.support_epsilon).all():
        raise RuntimeError("S5 did not preserve the exact immutable candidate set")
    return (
        descriptors[0].numpy().astype(np.float16),
        mass[0].numpy().astype(np.float32),
    )


def build_skelex_descriptor_cache(
    rows: list[dict[str, str]],
    accepted_records: list[dict[str, Any]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
    encoder: nn.Module,
    descriptor_config: SkelexDescriptorConfig,
    args: argparse.Namespace,
    device: torch.device,
    *,
    split: str,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    if len(rows) != len(accepted_records):
        raise ValueError("S5 split rows and accepted cache records do not align")
    result: list[dict[str, object]] = []
    support_masses: list[float] = []
    physical_candidates = 0
    retained_candidates = 0
    for start in range(0, len(rows), args.encoder_batch_size):
        batch_rows = rows[start : start + args.encoder_batch_size]
        batch_records = accepted_records[start : start + args.encoder_batch_size]
        pixels: list[torch.Tensor] = []
        payloads: list[tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]] = []
        for row, record in zip(batch_rows, batch_records, strict=True):
            if row["image_id"] != record["image_id"]:
                raise RuntimeError("S5 accepted cache order differs from frozen split")
            candidate_row = candidate_rows[Path(row["image_id"]).stem]
            if candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]:
                raise RuntimeError("S5 physical/cache candidate provenance mismatch")
            masks, _metadata, _scores, _fallback = legacy._load_candidate_payload(
                candidate_root,
                candidate_row,
                maximum_candidates=args.maximum_candidates,
            )
            indices = np.asarray(record["candidate_indices"], dtype=np.int64)
            if (
                indices.ndim != 1
                or not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(masks))
                or len(np.unique(indices)) != len(indices)
            ):
                raise RuntimeError("S5 accepted candidate indices are invalid")
            # The final four columns of the accepted Geometry-v3 descriptor are
            # its exact non-semantic SAM/prompt metadata. Reusing those columns
            # holds metadata fixed; cache ``shape_features`` are different
            # relational diagnostics and must not silently replace them.
            accepted_descriptors = np.asarray(record["descriptors"], dtype=np.float32)
            candidate_metadata = accepted_descriptors[:, -4:]
            if candidate_metadata.shape != (len(indices), 4):
                raise RuntimeError("S5 frozen candidate metadata shape mismatch")
            image_path = locate_verified_image(args.dataset_root, row)
            with Image.open(image_path) as image:
                normalized, projection = _normalized_square(image)
            pixels.append(normalized)
            payloads.append((masks[indices], candidate_metadata, projection, record))
            physical_candidates += len(masks)
            retained_candidates += len(indices)

        original = torch.stack(pixels)
        augmented = torch.cat((original, original.flip(-1)), dim=0)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            encoded = encoder(augmented.to(device, non_blocking=True))
        token_batch = encoded.float().cpu()
        count = len(batch_rows)
        for offset, (row, payload) in enumerate(zip(batch_rows, payloads, strict=True)):
            masks, candidate_metadata, projection, accepted = payload
            projected_masks = legacy.project_direct_resize_masks_to_square(
                torch.from_numpy(masks),
                padded_side=projection.padded_side,
                content_box=projection.content_box,
                # Reuse the accepted 128-square fractional geometry bridge
                # before area pooling to 14x14. Directly sampling at 56 could
                # erase a retained subpatch proposal before mass accounting.
                output_size=GEOMETRY_BRIDGE_SIZE,
            ).numpy()
            content = legacy.project_direct_resize_masks_to_square(
                torch.ones((1, masks.shape[-2], masks.shape[-1])),
                padded_side=projection.padded_side,
                content_box=projection.content_box,
                output_size=GEOMETRY_BRIDGE_SIZE,
            )[0].numpy()
            descriptors, masses = _pool_exact_bag(
                token_batch[offset],
                projected_masks,
                candidate_metadata,
                content,
                descriptor_config,
            )
            flipped, flipped_masses = _pool_exact_bag(
                token_batch[count + offset],
                projected_masks[..., ::-1].copy(),
                candidate_metadata,
                content[..., ::-1].copy(),
                descriptor_config,
            )
            if not np.allclose(masses, flipped_masses, rtol=0.0, atol=1.0e-6):
                raise RuntimeError("S5 original/flip support mass differs")
            support_masses.extend(masses.tolist())
            result.append(
                {
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "label": int(row["tumor"]),
                    "descriptors": descriptors,
                    "flipped_descriptors": flipped,
                    "kept_indices": np.asarray(accepted["candidate_indices"], dtype=np.int32),
                    "candidate_payload_sha256": accepted["candidate_payload_sha256"],
                }
            )
        completed = min(start + len(batch_rows), len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"S5 {split} descriptor cache: {completed}/{len(rows)}", flush=True)
    if len(result) != len(rows) or len(support_masses) != retained_candidates:
        raise RuntimeError("S5 descriptor cache cohort mismatch")
    return result, {
        "images": len(result),
        "physical_candidates": physical_candidates,
        "exact_retained_candidates": retained_candidates,
        "positive_support_candidates": len(support_masses),
        "minimum_fractional_grid_mass": float(min(support_masses)),
        "median_fractional_grid_mass": float(np.median(support_masses)),
        "maximum_fractional_grid_mass": float(max(support_masses)),
        "all_descriptors_finite": 1,
        "exact_candidate_set_preserved": 1,
    }


@torch.inference_mode()
def _score_skelex(
    model: RadDinoMaskBagMIL,
    records: list[dict[str, object]],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    scored: list[dict[str, Any]] = []
    for record in records:
        descriptors = torch.from_numpy(np.asarray(record["descriptors"], dtype=np.float32))[None].to(device)
        flipped = torch.from_numpy(np.asarray(record["flipped_descriptors"], dtype=np.float32))[None].to(device)
        valid = torch.ones(descriptors.shape[:2], dtype=torch.bool, device=device)
        logits_a, _ = model.score_descriptors(descriptors, valid)
        logits_b, _ = model.score_descriptors(flipped, valid)
        logits = 0.5 * (logits_a + logits_b)
        bag_logit = smooth_mil_pool(logits, valid, temperature=model.config.bag_temperature)
        scored.append(
            {
                "image_id": record["image_id"],
                "candidate_logits": logits[0].float().cpu().numpy(),
                "bag_logit": float(bag_logit.item()),
                "bag_probability": float(torch.sigmoid(bag_logit).item()),
            }
        )
    return scored


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _compose_pair(
    output_dir: Path,
    accepted_records: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    skelex_scored: list[dict[str, Any]],
    skelex_records: list[dict[str, object]],
    baseline_rows: list[dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float | int]]:
    accepted_predictions = {row["image_id"]: row for row in baseline_rows}
    evidence_root = output_dir / "skelex_score_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    arms: dict[str, list[dict[str, Any]]] = {CONTROL_ARM: [], PRIMARY_ARM: []}
    correlations: list[float] = []
    changed = 0
    for index, (record, baseline, semantic, descriptor_record) in enumerate(
        zip(accepted_records, base_scored, skelex_scored, skelex_records, strict=True)
    ):
        image_id = str(record["image_id"])
        if (
            image_id != baseline["image_id"]
            or image_id != semantic["image_id"]
            or image_id != descriptor_record["image_id"]
        ):
            raise RuntimeError("S5 validation score order mismatch")
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        base_logits = torch.from_numpy(np.asarray(baseline["base_candidate_logits"], dtype=np.float32))[None]
        skelex_logits = torch.from_numpy(np.asarray(semantic["candidate_logits"], dtype=np.float32))[None]
        valid = torch.ones((1, len(indices)), dtype=torch.bool)
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]
        ):
            raise RuntimeError("S5 upstream candidate provenance mismatch")
        with np.load(candidate_path, allow_pickle=False) as payload:
            upstream_all = np.asarray(payload["selection_scores"], dtype=np.float32)
        upstream = torch.from_numpy(upstream_all[indices])[None]
        control = base.equal_rank_aggregate((base_logits, upstream), valid)[0]
        primary = base.equal_rank_aggregate((base_logits, upstream, skelex_logits), valid)[0]
        skelex_rank = base.within_bag_percentile_ranks(skelex_logits, valid)[0]
        if len(indices) > 1:
            value = float(np.corrcoef(control.numpy(), skelex_rank.numpy())[0, 1])
            if np.isfinite(value):
                correlations.append(value)
        changed += int(int(control.argmax()) != int(primary.argmax()))
        relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        evidence_path = evidence_root / relative
        np.savez_compressed(
            evidence_path,
            candidate_indices=indices.astype(np.int32),
            baseline_logits=base_logits[0].numpy(),
            upstream_scores=upstream[0].numpy(),
            skelex_logits=skelex_logits[0].numpy(),
            skelex_rank=skelex_rank.numpy(),
            control_rank=control.numpy(),
            primary_rank=primary.numpy(),
            descriptors=np.asarray(descriptor_record["descriptors"], dtype=np.float16),
            flipped_descriptors=np.asarray(
                descriptor_record["flipped_descriptors"], dtype=np.float16
            ),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_count": len(indices),
                "evidence_path": str(relative),
                "evidence_sha256": sha256_file(evidence_path),
                "skelex_logits_sha256": sha256(skelex_logits.numpy().tobytes()).hexdigest(),
            }
        )
        common = {
            "image_id": image_id,
            "bag_logit": float(accepted_predictions[image_id]["bag_logit"]),
            "bag_probability": float(accepted_predictions[image_id]["bag_probability"]),
        }
        arms[CONTROL_ARM].append({**common, "candidate_logits": control.numpy().astype(np.float32)})
        arms[PRIMARY_ARM].append({**common, "candidate_logits": primary.numpy().astype(np.float32)})
    if not correlations:
        raise RuntimeError("S5 rank correlation is undefined")
    diagnostics: dict[str, float | int] = {
        "mean_skelex_control_rank_correlation": float(np.mean(correlations)),
        "correlation_images": len(correlations),
        "primary_changed_selections": changed,
        "primary_changed_selection_fraction": changed / 371.0,
    }
    return arms, _write_csv(evidence_root / "evidence_manifest.csv", evidence_rows), diagnostics


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    legacy.seed_everything(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S5 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S5 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    model_snapshot = verify_model_snapshot(
        args.skelex_model_dir,
        expected_config_sha256=args.expected_skelex_config_sha256,
        expected_preprocessor_sha256=args.expected_skelex_preprocessor_sha256,
        expected_weight_sha256=args.expected_skelex_weight_sha256,
    )
    cache_freeze, cache_manifest_rows = base._verify_cache_freeze(args)
    split_rows = {
        split: base.load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("S5 frozen cohort mismatch")
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
    accepted = base._load_cache_records(args, split_rows, cache_manifest_rows)

    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("S5 transformers version mismatch")
    pretrained = ViTMAEForPreTraining.from_pretrained(
        args.skelex_model_dir,
        local_files_only=True,
    )
    pretrained.vit.config.mask_ratio = 0.0
    pretrained.vit.requires_grad_(False).eval()
    projection = make_seeded_random_projection(
        input_dim=1024,
        output_dim=EXPECTED_PROJECTION_DIM,
        seed=EXPECTED_PROJECTION_SEED,
    )
    encoder: nn.Module = SkelexProjectedMultiLayerEncoder(
        pretrained.vit,
        torch.from_numpy(projection),
    ).to(device)
    del pretrained
    encoder = nn.DataParallel(encoder, device_ids=(0, 1), output_device=0).eval()
    descriptor_config = SkelexDescriptorConfig()
    train_cache, train_descriptor_gate = build_skelex_descriptor_cache(
        split_rows["train"], accepted["train"], train_candidates,
        args.train_candidate_root, encoder, descriptor_config, args, device,
        split="train",
    )
    val_cache, val_descriptor_gate = build_skelex_descriptor_cache(
        split_rows["val"], accepted["val"], val_candidates,
        args.val_candidate_root, encoder, descriptor_config, args, device,
        split="val",
    )
    descriptor_gate = {
        "status": "PASS_BEFORE_SELECTOR_TRAINING",
        "train": train_descriptor_gate,
        "validation": val_descriptor_gate,
        "exact_candidate_indices_source": "accepted_selector_cache",
        "minimum_mass_cutoff": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    descriptor_gate_path = args.output_dir / "descriptor_operational_gate.json"
    descriptor_gate_path.write_text(json.dumps(descriptor_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del encoder
    torch.cuda.empty_cache()

    selector_config = MaskBagMILConfig(
        token_dim=descriptor_config.token_dim,
        token_layers=descriptor_config.token_layers,
    )
    model, history = legacy.train_selector(train_cache, selector_config, args, device)
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_path = args.output_dir / "skelex_mask_bag_selector.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(selector_config),
            "descriptor_config": asdict(descriptor_config),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "skelex_weight_sha256": args.expected_skelex_weight_sha256,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    skelex_scored = _score_skelex(model, val_cache, device)
    classification = base._binary_metrics(
        np.asarray([row["label"] for row in val_cache], dtype=np.int8),
        np.asarray([row["bag_probability"] for row in skelex_scored], dtype=np.float64),
    )

    baseline_freeze, baseline_rows = base._verify_baseline_freeze(args)
    base_model, base_config = base._load_baseline_model(args, device=device)
    base_scored = score_same_family_graph_records(
        accepted["val"],
        base_model,
        bag_temperature=base_config.bag_temperature,
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
    identity_sha256 = _write_csv(args.output_dir / "baseline_identity.csv", identity_rows)
    arms, evidence_sha256, diagnostics = _compose_pair(
        args.output_dir,
        accepted["val"],
        base_scored,
        skelex_scored,
        val_cache,
        baseline_rows,
        val_candidates,
        args.val_candidate_root,
    )
    diagnostic_path = args.output_dir / "gt_blind_diagnostics.json"
    diagnostic = {
        **diagnostics,
        "skelex_image_label_metrics": classification,
        "diagnostics_block_prediction_freeze": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arm_freezes: dict[str, str] = {}
    for arm_name, scores in arms.items():
        arm_root = args.output_dir / arm_name
        prediction_sha, score_sha = base._write_validation_outputs(
            arm_root, accepted["val"], scores
        )
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
            "skelex_checkpoint_sha256": sha256_file(checkpoint_path),
            "skelex_public_weight_sha256": args.expected_skelex_weight_sha256,
            "training_history_sha256": sha256_file(history_path),
            "descriptor_operational_gate_sha256": sha256_file(descriptor_gate_path),
            "baseline_identity_sha256": identity_sha256,
            "skelex_score_evidence_manifest_sha256": evidence_sha256,
            "gt_blind_diagnostics_sha256": sha256_file(diagnostic_path),
            "prediction_manifest_sha256": prediction_sha,
            "candidate_score_manifest_sha256": score_sha,
            "validation_predictions": 371,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arm_freezes[arm_name] = sha256_file(freeze_path)
    pair = {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "arms": arm_freezes,
        "pair_physically_frozen_before_validation_gt": True,
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "cohort": {"train": 2981, "validation": 371},
        "skelex_model_snapshot": model_snapshot,
        "skelex_selected_layers": list(SELECTED_HIDDEN_LAYERS),
        "projection_sha256": projection_sha256(projection),
        "train_candidates": train_candidate_audit,
        "validation_candidates": val_candidate_audit,
        "descriptor_operational_gate_sha256": sha256_file(descriptor_gate_path),
        "pair_freeze_sha256": sha256_file(pair_path),
        "arms": arm_freezes,
        "training_labels": "image_level_normal_tumor_only",
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
