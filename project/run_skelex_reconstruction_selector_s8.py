"""Run the frozen S8 SKELEX decoder-reconstruction same-gallery reranker.

The runner consumes only radiographs, image-level labels, the immutable
candidate gallery/cache and a frozen public SKELEX checkpoint.  It creates the
validation prediction pair before any segmentation annotation path is opened.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
from mae_reconstruction_io import locate_verified_image, sha256_file, verify_model_snapshot
from models.mae_reconstruction import make_noise_bank, noise_bank_sha256, pad_to_square
from models.mask_bag_same_family_graph import (
    SameFamilyGraphConfig,
    score_same_family_graph_records,
)
from models.skelex_reconstruction_selector import (
    SkelexReconstructionConfig,
    masked_patch_squared_error,
    select_with_spatial_null,
)


EXPERIMENT_ID = "EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1"
RUN_ID = "btxrd_skelex_reconstruction_selector_s8_v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
PRIMARY_ARM = "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank"
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
EXPECTED_INPUT_SIZE = 224
EXPECTED_PATCH_SIZE = 16
EXPECTED_MASKS = 10
EXPECTED_MASK_RATIO = 0.75
EXPECTED_MASK_SEED = 42
EXPECTED_NULL_PERMUTATIONS = 255
EXPECTED_NULL_SEED = 20261203
EXPECTED_MAXIMUM_CANDIDATES = 81
EXPECTED_VALIDATION = 371
EXPECTED_TRAIN = 2981
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
    parser.add_argument("--maximum-candidates", type=int, default=EXPECTED_MAXIMUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    if args.maximum_candidates != EXPECTED_MAXIMUM_CANDIDATES or args.seed != 42:
        raise ValueError("S8 execution differs from the frozen one-shot recipe")


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _image_reconstruction_errors(
    model: nn.Module,
    pixels: torch.Tensor,
    noise_bank: torch.Tensor,
    expected_masks: torch.Tensor,
    config: SkelexReconstructionConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = noise_bank.shape[0]
    batch = pixels[None].expand(masks, -1, -1, -1).contiguous()
    noise = noise_bank.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            pixel_values=batch.to(device, non_blocking=True),
            noise=noise,
            interpolate_pos_encoding=True,
        )
    predicted_mask = output.mask.detach().to("cpu") > 0.5
    if predicted_mask.shape != expected_masks.shape or not torch.equal(predicted_mask, expected_masks):
        raise RuntimeError("S8 ViT-MAE mask order differs from frozen noise bank")
    errors, observed = masked_patch_squared_error(
        output.logits.detach().to("cpu"),
        batch,
        predicted_mask,
        patch_size=config.patch_size,
    )
    grid = config.grid_size
    return errors.reshape(masks, grid, grid).float(), observed.reshape(masks, grid, grid)


def _project_candidate_grid(
    masks: np.ndarray,
    *,
    projection: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = legacy.project_direct_resize_masks_to_square(
        torch.from_numpy(np.asarray(masks, dtype=np.float32)),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=14,
    )
    content_source = torch.ones((1, masks.shape[-2], masks.shape[-1]), dtype=torch.float32)
    content = legacy.project_direct_resize_masks_to_square(
        content_source,
        padded_side=projection.padded_side,
        content_box=projection.content_box,
        output_size=14,
    )[0]
    return projected.float(), content.float()


def _write_evidence(path: Path, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return sha256_file(path)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    legacy.seed_everything(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S8 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S8 requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    config = SkelexReconstructionConfig(
        input_size=EXPECTED_INPUT_SIZE,
        patch_size=EXPECTED_PATCH_SIZE,
        num_masks=EXPECTED_MASKS,
        mask_ratio=EXPECTED_MASK_RATIO,
        mask_seed=EXPECTED_MASK_SEED,
        null_permutations=EXPECTED_NULL_PERMUTATIONS,
        null_seed=EXPECTED_NULL_SEED,
    )
    noise_bank = make_noise_bank(
        num_masks=config.num_masks,
        num_patches=config.patch_count,
        seed=config.mask_seed,
    )
    expected_masks = torch.zeros_like(noise_bank, dtype=torch.bool)
    keep = int(config.patch_count * (1.0 - config.mask_ratio))
    expected_masks.scatter_(1, torch.argsort(noise_bank, dim=1, stable=True)[:, keep:], True)

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
        raise RuntimeError("S8 frozen cohort mismatch")
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
    baseline_freeze, baseline_rows = base._verify_baseline_freeze(args)
    baseline_model, baseline_config = base._load_baseline_model(args, device=device)
    base_scored = score_same_family_graph_records(
        accepted["val"],
        baseline_model,
        bag_temperature=baseline_config.bag_temperature,
        graph_config=SameFamilyGraphConfig(minimum_iou=1.0, minimum_containment=1.0, alpha=0.0, iterations=1),
        batch_size=16,
        device=device,
    )
    base._baseline_identity(accepted["val"], base_scored, baseline_rows)
    del baseline_model
    torch.cuda.empty_cache()

    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("S8 transformers version mismatch")
    pretrained = ViTMAEForPreTraining.from_pretrained(args.skelex_model_dir, local_files_only=True)
    if abs(float(pretrained.config.mask_ratio) - EXPECTED_MASK_RATIO) > 1.0e-12:
        raise RuntimeError("S8 public checkpoint mask_ratio differs from frozen recipe")
    if bool(getattr(pretrained.config, "norm_pix_loss", False)):
        raise RuntimeError("S8 public checkpoint enables forbidden normalized pixel loss")
    pretrained.requires_grad_(False).eval()
    model: nn.Module = nn.DataParallel(pretrained.to(device), device_ids=(0, 1), output_device=0).eval()

    evidence_root = args.output_dir / "reconstruction_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    control_scores: list[dict[str, object]] = []
    primary_scores: list[dict[str, object]] = []
    switched = 0
    family_pass = 0
    p_values: list[float] = []
    if len(accepted["val"]) != len(base_scored):
        raise RuntimeError("S8 baseline/cache validation cohorts do not align")
    for index, (record, baseline) in enumerate(zip(accepted["val"], base_scored)):
        image_id = str(record["image_id"])
        candidate_row = val_candidates[Path(image_id).stem]
        if candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]:
            raise RuntimeError(f"S8 candidate provenance mismatch: {image_id}")
        packed_masks = base.unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        image_path = locate_verified_image(args.dataset_root, split_rows["val"][index])
        with Image.open(image_path) as image:
            pixels, projection = _normalized_square(image)
        candidate_grid, content_grid = _project_candidate_grid(packed_masks, projection=projection)
        original_errors, original_observed = _image_reconstruction_errors(
            model, pixels, noise_bank, expected_masks, config, device
        )
        flipped_errors, flipped_observed = _image_reconstruction_errors(
            model, pixels.flip(-1), noise_bank, expected_masks, config, device
        )
        aligned_flip_errors = flipped_errors.flip(-1)
        aligned_flip_observed = flipped_observed.flip(-1)
        base_logits = torch.from_numpy(np.asarray(baseline["base_candidate_logits"], dtype=np.float32))
        families = tuple(str(value) for value in np.asarray(record["family_ids"]).tolist())
        selected = select_with_spatial_null(
            base_scores=base_logits,
            accepted_index=int(np.argmax(base_logits.numpy())),
            families=families,
            original_errors=original_errors,
            original_observed=original_observed,
            aligned_flip_errors=aligned_flip_errors,
            aligned_flip_observed=aligned_flip_observed,
            candidate_masks=candidate_grid,
            content_mask=content_grid,
            config=config,
        )
        p_values.append(float(selected["permutation_p_value"]))
        switched += int(bool(selected["switched"]))
        family_pass += int(bool(selected["family_consistent"]))
        selected_fused = np.asarray(selected["combined_fused"].detach().cpu(), dtype=np.float32)
        selected_fused = np.where(np.isfinite(selected_fused), selected_fused, -1.0e9)
        primary_logits = selected_fused if bool(selected["switched"]) else base_logits.numpy().astype(np.float32)
        common = {
            "image_id": image_id,
            "bag_logit": float(baseline["base_bag_logit"]),
            "bag_probability": float(baseline["base_bag_probability"]),
        }
        control_scores.append({**common, "candidate_logits": base_logits.numpy().astype(np.float32)})
        primary_scores.append({**common, "candidate_logits": primary_logits})
        evidence_path = evidence_root / f"{index:04d}_{Path(image_id).stem}.npz"
        evidence_sha = _write_evidence(
            evidence_path,
            {
                "candidate_indices": np.asarray(record["candidate_indices"], dtype=np.int32),
                "family_ids": np.asarray(record["family_ids"], dtype=np.int32),
                "base_scores": base_logits.numpy().astype(np.float32),
                "candidate_masks": candidate_grid.numpy().astype(np.float32),
                "content_mask": content_grid.numpy().astype(np.float32),
                "original_errors": original_errors.numpy().astype(np.float32),
                "original_observed": original_observed.numpy().astype(np.uint8),
                "aligned_flip_errors": aligned_flip_errors.numpy().astype(np.float32),
                "aligned_flip_observed": aligned_flip_observed.numpy().astype(np.uint8),
                "original_lcb": selected["original_lcb"].detach().cpu().numpy().astype(np.float32),
                "aligned_flip_lcb": selected["aligned_flip_lcb"].detach().cpu().numpy().astype(np.float32),
                "combined_lcb": selected["combined_lcb"].detach().cpu().numpy().astype(np.float32),
                "combined_fused": selected["combined_fused"].detach().cpu().numpy().astype(np.float32),
                "combined_candidate_valid": selected["combined_candidate_valid"].detach().cpu().numpy().astype(np.uint8),
                "selected_index": np.asarray(int(selected["selected_index"]), dtype=np.int32),
                "combined_winner": np.asarray(int(selected["combined_winner"]), dtype=np.int32),
                "original_winner": np.asarray(int(selected["original_winner"]), dtype=np.int32),
                "aligned_flip_winner": np.asarray(int(selected["aligned_flip_winner"]), dtype=np.int32),
                "family_consistent": np.asarray(int(bool(selected["family_consistent"])), dtype=np.uint8),
                "switched": np.asarray(int(bool(selected["switched"])), dtype=np.uint8),
                "permutation_p_value": np.asarray(float(selected["permutation_p_value"]), dtype=np.float64),
                "noise_bank": noise_bank.numpy().astype(np.float32),
            },
        )
        evidence_rows.append({
            "image_id": image_id,
            "group_id": record["group_id"],
            "candidate_count": len(base_logits),
            "evidence_path": str(evidence_path.relative_to(args.output_dir)),
            "evidence_sha256": evidence_sha,
            "selected_index": int(selected["selected_index"]),
            "switched": int(bool(selected["switched"])),
            "permutation_p_value": float(selected["permutation_p_value"]),
        })
        if (index + 1) % 25 == 0 or index + 1 == EXPECTED_VALIDATION:
            print(f"S8 reconstruction selector: {index + 1}/{EXPECTED_VALIDATION}", flush=True)
    del model
    torch.cuda.empty_cache()

    evidence_manifest = args.output_dir / "reconstruction_evidence" / "evidence_manifest.json"
    evidence_manifest_sha = _write_json(evidence_manifest, {
        "experiment_id": EXPERIMENT_ID,
        "rows": evidence_rows,
        "noise_bank_sha256": noise_bank_sha256(noise_bank),
        "null_permutations": config.null_permutations,
        "null_seed": config.null_seed,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    })
    arms = {CONTROL_ARM: control_scores, PRIMARY_ARM: primary_scores}
    arm_freezes: dict[str, str] = {}
    for arm_name, scores in arms.items():
        arm_root = args.output_dir / arm_name
        prediction_sha, score_sha = base._write_validation_outputs(arm_root, accepted["val"], scores)
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
            "skelex_public_weight_sha256": args.expected_skelex_weight_sha256,
            "noise_bank_sha256": noise_bank_sha256(noise_bank),
            "reconstruction_evidence_manifest_sha256": evidence_manifest_sha,
            "prediction_manifest_sha256": prediction_sha,
            "candidate_score_manifest_sha256": score_sha,
            "validation_predictions": EXPECTED_VALIDATION,
            "training_labels": "image_level_normal_tumor_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        _write_json(freeze_path, freeze)
        arm_freezes[arm_name] = sha256_file(freeze_path)
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    _write_json(pair_path, {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "arms": arm_freezes,
        "pair_physically_frozen_before_validation_gt": True,
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    })
    diagnostics_path = args.output_dir / "gt_blind_diagnostics.json"
    _write_json(diagnostics_path, {
        "experiment_id": EXPERIMENT_ID,
        "validation_predictions": EXPECTED_VALIDATION,
        "switch_count": switched,
        "switch_fraction": switched / EXPECTED_VALIDATION,
        "family_consistent_count": family_pass,
        "median_permutation_p_value": float(np.median(p_values)),
        "minimum_permutation_p_value": float(np.min(p_values)),
        "maximum_permutation_p_value": float(np.max(p_values)),
        "noise_bank_sha256": noise_bank_sha256(noise_bank),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    })
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
            "decoder_data_parallel": True,
        },
        "cohort": {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION},
        "skelex_model_snapshot": model_snapshot,
        "reconstruction_config": {
            "input_size": config.input_size,
            "patch_size": config.patch_size,
            "num_masks": config.num_masks,
            "mask_ratio": config.mask_ratio,
            "mask_seed": config.mask_seed,
            "noise_bank_sha256": noise_bank_sha256(noise_bank),
            "null_permutations": config.null_permutations,
            "null_seed": config.null_seed,
            "context_radius": config.context_radius,
            "lcb_z": config.lcb_z,
            "geometry_weight": config.geometry_weight,
            "reconstruction_weight": config.reconstruction_weight,
        },
        "train_candidates": train_candidate_audit,
        "validation_candidates": val_candidate_audit,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "prediction_pair_freeze_sha256": sha256_file(pair_path),
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
