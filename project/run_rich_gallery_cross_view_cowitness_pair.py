from __future__ import annotations

"""Train matched cross-view co-witness full/control selector heads.

The runner is prediction-first and image-label-only.  It consumes frozen
proposal bags, frozen RAD-DINO descriptors, an immutable G1 checkpoint and a
hash-bound train-only pair manifest.  It never imports or opens segmentation
annotations.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
    verify_model_snapshot,
)
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from models.rich_gallery_cross_view_cowitness import (
    CrossViewCoWitnessConfig,
    CrossViewCoWitnessHead,
    co_witness_margin_loss,
    co_witness_score,
    dense_normal_candidate_loss,
    image_bag_loss,
    normalized_logmeanexp,
)
from models.rich_gallery_g2_objective import rank_fusion_scores, shared_source_validity, stable_select
from run_rad_dino_mask_bag_mil_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    _audit_candidate_input,
    build_descriptor_cache,
    seed_everything,
)
from run_rich_gallery_g2_selector_pair import attach_rich_metadata


ARM_NAMES = ("control", "full")
RESIDUAL_MULTIPLIERS = (0.25, 0.50, 1.00, 2.00)


def variant_name(arm: str, multiplier: float) -> str:
    if arm not in ARM_NAMES or multiplier not in RESIDUAL_MULTIPLIERS:
        raise ValueError("unknown cross-view residual variant")
    return f"{arm}__residual_x{multiplier:g}"


def frozen_variants() -> list[str]:
    return ["baseline"] + [
        variant_name(arm, multiplier)
        for arm in ARM_NAMES
        for multiplier in RESIDUAL_MULTIPLIERS
    ]


def variant_spec(name: str) -> tuple[str | None, float]:
    if name == "baseline":
        return None, 0.0
    for arm in ARM_NAMES:
        for multiplier in RESIDUAL_MULTIPLIERS:
            if name == variant_name(arm, multiplier):
                return arm, multiplier
    raise ValueError(f"unknown frozen variant: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--maximum-candidates", type=int, default=243)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _read_pair_manifest(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("cross-view pair manifest SHA-256 mismatch")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_id",
        "anchor_image_id",
        "anchor_group_id",
        "same_image_id",
        "same_group_id",
        "control1_image_id",
        "control1_group_id",
        "control2_image_id",
        "control2_group_id",
    }
    if len(rows) != 384 or not rows or not required.issubset(rows[0]):
        raise ValueError("cross-view pair manifest population/schema mismatch")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("cross-view pair IDs are duplicated")
    for row in rows:
        if row["anchor_group_id"] != row["same_group_id"]:
            raise ValueError("full-arm pair does not share a group")
        if len(
            {
                row["same_group_id"],
                row["control1_group_id"],
                row["control2_group_id"],
            }
        ) != 3:
            raise ValueError("control-arm groups are not distinct")
    return rows


def _appearance(record: Mapping[str, object]) -> np.ndarray:
    original = np.asarray(record["descriptors"], dtype=np.float32)
    flipped = np.asarray(record["flipped_descriptors"], dtype=np.float32)
    if original.shape != flipped.shape or original.ndim != 2 or original.shape[1] != 1156:
        raise ValueError("G1 candidate descriptor layout changed")
    # The last four fields are candidate metadata.  The learned residual may
    # use appearance only, not area, coordinates or source/count metadata.
    return (0.5 * (original[:, :1152] + flipped[:, :1152])).astype(np.float32)


def attach_immutable_baseline(
    cache: list[dict[str, object]],
    g1_model: RadDinoMaskBagMIL,
    device: torch.device,
) -> dict[str, object]:
    g1_model.eval()
    maximum_zero_residual_choice_delta = 0
    for record in cache:
        original = torch.from_numpy(np.asarray(record["descriptors"], dtype=np.float32))[None].to(device)
        flipped = torch.from_numpy(np.asarray(record["flipped_descriptors"], dtype=np.float32))[None].to(device)
        valid = torch.ones(original.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            original_logits, _ = g1_model.score_descriptors(original, valid)
            flipped_logits, _ = g1_model.score_descriptors(flipped, valid)
        g1 = (0.5 * (original_logits + flipped_logits))[0].float().cpu().numpy().astype(np.float64)
        upstream = np.asarray(record["upstream_scores"], dtype=np.float64)
        fusion = rank_fusion_scores(g1, upstream)
        centered = (2.0 * fusion - 1.0).astype(np.float32)
        record["appearance"] = _appearance(record)
        record["g1_logits"] = g1.astype(np.float32)
        record["baseline_fusion"] = fusion.astype(np.float32)
        record["baseline_scores"] = centered
        baseline_local = stable_select(centered.astype(np.float64), g1)
        repeated_local = stable_select((centered + 0.0).astype(np.float64), g1)
        maximum_zero_residual_choice_delta = max(
            maximum_zero_residual_choice_delta, abs(baseline_local - repeated_local)
        )
    return {
        "images": len(cache),
        "appearance_dim": 1152,
        "metadata_fields_excluded": 4,
        "zero_residual_maximum_local_choice_delta": maximum_zero_residual_choice_delta,
    }


def _tensor_record(record: Mapping[str, object], device: torch.device):
    appearance = torch.from_numpy(np.asarray(record["appearance"], dtype=np.float32))[None].to(device)
    baseline = torch.from_numpy(np.asarray(record["baseline_scores"], dtype=np.float32))[None].to(device)
    g1_logits = torch.from_numpy(np.asarray(record["g1_logits"], dtype=np.float32))[None].to(device)
    source_ids = torch.from_numpy(np.asarray(record["source_ids"], dtype=np.int64))[None].to(device)
    valid = torch.ones(baseline.shape, dtype=torch.bool, device=device)
    shared = shared_source_validity(valid, source_ids)
    return appearance, baseline, g1_logits, valid, shared


def _forward_record(
    model: CrossViewCoWitnessHead,
    record: Mapping[str, object],
    device: torch.device,
):
    appearance, baseline, g1_logits, valid, shared = _tensor_record(record, device)
    combined, residual, embedding = model(appearance, baseline, valid)
    residual = residual * shared.to(residual.dtype)
    selector_scores = baseline + residual
    image_scores = g1_logits + residual
    return selector_scores, image_scores, residual, embedding, valid, shared


def train_arm(
    arm: str,
    train_by_image: Mapping[str, dict[str, object]],
    pair_rows: list[dict[str, str]],
    normal_records: list[dict[str, object]],
    initial_state: Mapping[str, torch.Tensor],
    config: CrossViewCoWitnessConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CrossViewCoWitnessHead, list[dict[str, float]]]:
    if arm not in ARM_NAMES:
        raise ValueError("unknown cross-view arm")
    seed_everything(args.seed)
    model = CrossViewCoWitnessHead(config).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(pair_rows))
        normal_order = np.random.default_rng(args.seed + 1000 + epoch).permutation(len(normal_records))
        totals = {"total": 0.0, "pair": 0.0, "tumor_bag": 0.0, "normal_bag": 0.0, "normal_dense": 0.0, "drift": 0.0}
        for step, pair_index in enumerate(order.tolist()):
            row = pair_rows[pair_index]
            anchor = train_by_image[row["anchor_image_id"]]
            positive_id = row["same_image_id"] if arm == "full" else row["control1_image_id"]
            negative_id = row["control1_image_id"] if arm == "full" else row["control2_image_id"]
            positive = train_by_image[positive_id]
            negative = train_by_image[negative_id]
            normal = normal_records[int(normal_order[step % len(normal_order)])]

            anchor_values = _forward_record(model, anchor, device)
            positive_values = _forward_record(model, positive, device)
            negative_values = _forward_record(model, negative, device)
            normal_values = _forward_record(model, normal, device)
            _, anchor_image, anchor_residual, anchor_embedding, _, anchor_shared = anchor_values
            _, positive_image, positive_residual, positive_embedding, _, positive_shared = positive_values
            _, negative_image, negative_residual, negative_embedding, _, negative_shared = negative_values
            _, normal_image, normal_residual, _normal_embedding, _, normal_shared = normal_values

            positive_pair = co_witness_score(
                anchor_residual, anchor_embedding, anchor_shared,
                positive_residual, positive_embedding, positive_shared,
                temperature=config.pair_temperature, cosine_weight=config.cosine_weight,
            )
            negative_pair = co_witness_score(
                anchor_residual, anchor_embedding, anchor_shared,
                negative_residual, negative_embedding, negative_shared,
                temperature=config.pair_temperature, cosine_weight=config.cosine_weight,
            )
            pair_loss = co_witness_margin_loss(
                positive_pair, negative_pair, margin=config.pair_margin
            )
            tumor_bag = 0.5 * (
                image_bag_loss(anchor_image, anchor_shared, torch.ones(1, device=device), temperature=config.bag_temperature)
                + image_bag_loss(positive_image, positive_shared, torch.ones(1, device=device), temperature=config.bag_temperature)
            )
            normal_bag = image_bag_loss(
                normal_image, normal_shared, torch.zeros(1, device=device), temperature=config.bag_temperature
            )
            normal_dense = dense_normal_candidate_loss(
                normal_image, normal_shared, torch.zeros(1, device=device)
            )
            drift = torch.cat(
                [anchor_residual[anchor_shared], positive_residual[positive_shared], normal_residual[normal_shared]]
            ).square().mean()
            total = pair_loss + 0.50 * tumor_bag + 0.50 * normal_bag + 0.25 * normal_dense + 0.001 * drift
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total), ("pair", pair_loss), ("tumor_bag", tumor_bag),
                ("normal_bag", normal_bag), ("normal_dense", normal_dense), ("drift", drift),
            ):
                totals[key] += float(value.detach().item())
        record = {"epoch": float(epoch), "steps": float(len(order)), **{key: value / len(order) for key, value in totals.items()}}
        history.append(record)
        print(json.dumps({"arm": arm, **record}, sort_keys=True), flush=True)
    return model, history


def score_validation(
    model: CrossViewCoWitnessHead,
    cache: list[dict[str, object]],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    model.eval()
    output: dict[str, dict[str, object]] = {}
    for record in cache:
        with torch.inference_mode():
            selector, image_scores, residual, _embedding, valid, shared = _forward_record(model, record, device)
        selector_values = selector[0].float().cpu().numpy().astype(np.float64)
        raw = np.asarray(record["g1_logits"], dtype=np.float64)
        local = stable_select(selector_values, raw)
        kept = np.asarray(record["kept_indices"], dtype=np.int64)
        output[str(record["image_id"])] = {
            "selected_local_index": local,
            "selected_candidate_index": int(kept[local]),
            "selector_scores": selector_values.astype(np.float32),
            "residual": residual[0].float().cpu().numpy().astype(np.float32),
            "image_bag_logit": float(
                normalized_logmeanexp(
                    image_scores, shared, temperature=0.2
                )[0].item()
            ),
            "shared_candidates": int(shared.sum().item()),
            "all_candidates": int(valid.sum().item()),
        }
    return output


def freeze_output(
    args: argparse.Namespace,
    val_cache: list[dict[str, object]],
    scored: Mapping[str, Mapping[str, Mapping[str, object]]],
    histories: Mapping[str, list[dict[str, float]]],
    checkpoints: Mapping[str, dict[str, object]],
    *,
    pair_rows: list[dict[str, str]],
    cache_reports: Mapping[str, object],
    model_snapshot: Mapping[str, object],
    projection_hash: str,
) -> Path:
    score_root = args.output_dir / "stage_a_scores"
    score_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    score_hashes: list[str] = []
    for record in val_cache:
        image_id = str(record["image_id"])
        payload: dict[str, np.ndarray] = {
            "candidate_indices": np.asarray(record["kept_indices"], dtype=np.int32),
            "source_ids": np.asarray(record["source_ids"], dtype=np.int16),
            "g1_logits": np.asarray(record["g1_logits"], dtype=np.float32),
            "upstream_scores": np.asarray(record["upstream_scores"], dtype=np.float32),
            "baseline_fusion": np.asarray(record["baseline_fusion"], dtype=np.float32),
            "baseline_scores": np.asarray(record["baseline_scores"], dtype=np.float32),
        }
        baseline_local = stable_select(
            np.asarray(record["baseline_scores"], dtype=np.float64),
            np.asarray(record["g1_logits"], dtype=np.float64),
        )
        for arm in ARM_NAMES:
            payload[f"{arm}_selector_scores"] = np.asarray(scored[arm][image_id]["selector_scores"], dtype=np.float32)
            payload[f"{arm}_residual"] = np.asarray(scored[arm][image_id]["residual"], dtype=np.float32)
        score_path = score_root / f"{Path(image_id).stem}.npz"
        np.savez_compressed(score_path, **payload)
        score_hash = sha256_file(score_path)
        score_hashes.append(score_hash)
        sources = np.asarray(record["source_names"], dtype=str)
        selections: list[tuple[str, int]] = [("baseline", baseline_local)]
        for arm in ARM_NAMES:
            residual = np.asarray(scored[arm][image_id]["residual"], dtype=np.float64)
            for multiplier in RESIDUAL_MULTIPLIERS:
                variant = variant_name(arm, multiplier)
                local = stable_select(
                    np.asarray(record["baseline_scores"], dtype=np.float64)
                    + float(multiplier) * residual,
                    np.asarray(record["g1_logits"], dtype=np.float64),
                )
                selections.append((variant, local))
        for variant, local in selections:
            rows.append(
                {
                    "image_id": image_id,
                    "group_id": record["group_id"],
                    "tumor": int(record["label"]),
                    "variant": variant,
                    "candidate_payload_sha256": record["candidate_payload_sha256"],
                    "candidate_count": len(payload["candidate_indices"]),
                    "selected_local_index": local,
                    "selected_candidate_index": int(payload["candidate_indices"][local]),
                    "selected_source": str(sources[local]),
                    "score_path": str(score_path.relative_to(args.output_dir)).replace("\\", "/"),
                    "score_sha256": score_hash,
                }
            )
    selection_path = args.output_dir / "stage_a_selection_manifest.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_root = args.output_dir / "checkpoints"
    history_root = args.output_dir / "training_history"
    checkpoint_root.mkdir()
    history_root.mkdir()
    checkpoint_hashes: dict[str, str] = {}
    history_hashes: dict[str, str] = {}
    for arm in ARM_NAMES:
        checkpoint_path = checkpoint_root / f"{arm}.pt"
        torch.save(checkpoints[arm], checkpoint_path)
        checkpoint_hashes[arm] = sha256_file(checkpoint_path)
        history_path = history_root / f"{arm}.json"
        history_path.write_text(json.dumps(histories[arm], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        history_hashes[arm] = sha256_file(history_path)
    freeze = {
        "stage": "rich_gallery_cross_view_cowitness_pair_stage_a_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "pair_manifest_sha256": args.expected_pair_manifest_sha256,
        "pair_rows": len(pair_rows),
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_hash,
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "train_candidate_manifest_sha256": args.train_candidate_manifest_sha256,
        "train_pseudo_manifest_sha256": args.train_pseudo_manifest_sha256,
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "cache_reports": cache_reports,
        "checkpoint_sha256": checkpoint_hashes,
        "training_history_sha256": history_hashes,
        "selection_manifest_sha256": sha256_file(selection_path),
        "score_set_sha256": hashlib.sha256("\n".join(sorted(score_hashes)).encode()).hexdigest(),
        "validation_images": 371,
        "variants": frozen_variants(),
        "residual_multipliers": list(RESIDUAL_MULTIPLIERS),
        "selection_rows": len(rows),
        "zero_residual_baseline_reproduced": True,
        "candidate_choices_frozen_before_validation_gt": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
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
        or args.maximum_candidates != 243
        or args.epochs != 2
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.seed != 20260802
    ):
        raise ValueError("cross-view execution differs from frozen diagnostic")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("cross-view output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    pair_rows = _read_pair_manifest(args.pair_manifest, args.expected_pair_manifest_sha256)
    train_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("canonical train/validation cohort mismatch")
    train_candidates, train_audit = _audit_candidate_input(
        args.train_candidate_root, train_rows, split="train",
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
    )
    val_candidates, val_audit = _audit_candidate_input(
        args.val_candidate_root, val_rows, split="val",
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
        raise ValueError("G1 checkpoint SHA-256 mismatch")

    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("unexpected transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("cross-view diagnostic requires exactly T4 x2")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"cross-view diagnostic requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    projection = make_seeded_random_projection(input_dim=768, output_dim=128, seed=42)
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder: nn.Module = ProjectedMultiLayerEncoder(backbone, torch.from_numpy(projection)).to(device)
    encoder = nn.DataParallel(encoder, device_ids=[0, 1], output_device=0).eval()
    g1_config = MaskBagMILConfig(token_dim=128, token_layers=len(SELECTED_HIDDEN_LAYERS))
    train_cache = build_descriptor_cache(
        train_rows, train_candidates, args.train_candidate_root, encoder, g1_config, args, device, split="train"
    )
    val_cache = build_descriptor_cache(
        val_rows, val_candidates, args.val_candidate_root, encoder, g1_config, args, device, split="val"
    )
    del encoder, backbone
    torch.cuda.empty_cache()
    train_source_report = attach_rich_metadata(train_cache, train_candidates, args.train_candidate_root)
    val_source_report = attach_rich_metadata(val_cache, val_candidates, args.val_candidate_root)

    g1_payload = torch.load(args.g1_checkpoint, map_location="cpu", weights_only=False)
    if MaskBagMILConfig(**g1_payload["config"]) != g1_config:
        raise ValueError("G1 checkpoint config changed")
    g1_model = RadDinoMaskBagMIL(g1_config).to(device)
    g1_model.load_state_dict(g1_payload["model_state_dict"], strict=True)
    train_baseline = attach_immutable_baseline(train_cache, g1_model, device)
    val_baseline = attach_immutable_baseline(val_cache, g1_model, device)
    del g1_model
    torch.cuda.empty_cache()

    train_by_image = {str(record["image_id"]): record for record in train_cache}
    if len(train_by_image) != 2981:
        raise RuntimeError("train descriptor cache identities changed")
    required_pair_images = {
        row[key]
        for row in pair_rows
        for key in ("anchor_image_id", "same_image_id", "control1_image_id", "control2_image_id")
    }
    if not required_pair_images.issubset(train_by_image):
        raise ValueError("pair manifest references missing train cache images")
    normal_records = [record for record in train_cache if int(record["label"]) == 0]
    if len(normal_records) != 1493:
        raise RuntimeError("canonical train-normal cohort changed")
    head_config = CrossViewCoWitnessConfig()
    seed_everything(args.seed)
    initial_model = CrossViewCoWitnessHead(head_config)
    initial_state = {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()}
    if any(torch.count_nonzero(value).item() for key, value in initial_state.items() if key.startswith("residual_head.")):
        raise RuntimeError("cross-view residual head is not zero initialized")
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints: dict[str, dict[str, object]] = {}
    models: dict[str, CrossViewCoWitnessHead] = {}
    for arm in ARM_NAMES:
        model, history = train_arm(
            arm, train_by_image, pair_rows, normal_records, initial_state, head_config, args, device
        )
        models[arm] = model
        histories[arm] = history
        checkpoints[arm] = {
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "config": asdict(head_config),
            "arm": arm,
            "final_epoch": args.epochs,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "pair_manifest_sha256": args.expected_pair_manifest_sha256,
            "training_labels": "binary_image_labels_plus_heuristic_train_group_relation",
            "spatial_ground_truth_used": False,
            "test_evaluated": False,
        }
    scored = {arm: score_validation(models[arm], val_cache, device) for arm in ARM_NAMES}
    freeze_path = freeze_output(
        args, val_cache, scored, histories, checkpoints, pair_rows=pair_rows,
        cache_reports={
            "train_candidate_audit": train_audit,
            "val_candidate_audit": val_audit,
            "train_source_report": train_source_report,
            "val_source_report": val_source_report,
            "train_baseline": train_baseline,
            "val_baseline": val_baseline,
        },
        model_snapshot=model_snapshot,
        projection_hash=projection_sha256(projection),
    )
    run_manifest = {
        "run_id": "btxrd_rich_gallery_cross_view_cowitness_pair_v1",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "pair_manifest_sha256": args.expected_pair_manifest_sha256,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
