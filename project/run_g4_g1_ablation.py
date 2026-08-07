from __future__ import annotations

"""G4 E6 matched G1 feature/loss ablation on one reusable RAD-DINO cache.

This stage is annotation-free.  It encodes the frozen rich gallery once, fits
all predeclared arms, and freezes every validation candidate choice.  A
separate evaluator may open validation polygons only after this script exits.
"""

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform

import numpy as np
import torch

from evaluate_g4_classifier_labels import _binary_metrics
from final_selector import select_candidate
from frozen_io import (
    load_split_rows_without_annotations,
    sha256_file,
    verify_model_snapshot,
)
from gpu_runtime import place_frozen_encoder, require_cuda_runtime
from models.nominal_patch_memory import (
    make_seeded_random_projection,
    projection_sha256,
)
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    negative_bag_instance_loss,
    self_guided_instance_loss,
    smooth_mil_pool,
)
from run_rad_dino_mask_bag_mil_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    _audit_candidate_input,
    _padded_batch,
    build_descriptor_cache,
    seed_everything,
)


FEATURE_ARMS = (
    "inside_only",
    "inside_ring",
    "inside_ring_contrast",
    "full",
)
LOSS_ARMS = (
    "bag_only",
    "bag_negative",
    "bag_selfguided",
    "full",
)


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
    parser.add_argument("--baseline-choice-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-choice-freeze-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
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
    parser.add_argument("--maximum-candidates", type=int, default=243)
    parser.add_argument("--seeds", default="42,43,44")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if values != (42, 43, 44):
        raise ValueError("G4 E6 requires the frozen seeds 42,43,44")
    return values


def descriptor_feature_mask(
    config: MaskBagMILConfig,
    arm: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a capacity-matched zeroing mask for one cumulative feature arm."""

    if arm not in FEATURE_ARMS:
        raise ValueError(f"unknown G1 feature arm: {arm}")
    per_block = config.token_layers * config.token_dim
    mask = torch.zeros(config.descriptor_dim, dtype=torch.float32, device=device)
    mask[:per_block] = 1.0
    if arm in {"inside_ring", "inside_ring_contrast", "full"}:
        mask[per_block : 2 * per_block] = 1.0
    if arm in {"inside_ring_contrast", "full"}:
        mask[2 * per_block : 3 * per_block] = 1.0
    if arm == "full":
        mask[3 * per_block :] = 1.0
    return mask


def _training_specs() -> list[dict[str, str]]:
    return [
        {"key": "feature_inside_only", "feature": "inside_only", "loss": "full"},
        {"key": "feature_inside_ring", "feature": "inside_ring", "loss": "full"},
        {
            "key": "feature_inside_ring_contrast",
            "feature": "inside_ring_contrast",
            "loss": "full",
        },
        {"key": "full", "feature": "full", "loss": "full"},
        {"key": "loss_bag_only", "feature": "full", "loss": "bag_only"},
        {"key": "loss_bag_negative", "feature": "full", "loss": "bag_negative"},
        {
            "key": "loss_bag_selfguided",
            "feature": "full",
            "loss": "bag_selfguided",
        },
    ]


def _reported_arm_names(seed: int) -> dict[str, str]:
    return {
        "feature_inside_only": f"E6F__inside_only__seed{seed}",
        "feature_inside_ring": f"E6F__inside_ring__seed{seed}",
        "feature_inside_ring_contrast": f"E6F__inside_ring_contrast__seed{seed}",
        "feature_full": f"E6F__full__seed{seed}",
        "loss_bag_only": f"E6L__bag_only__seed{seed}",
        "loss_bag_negative": f"E6L__bag_negative__seed{seed}",
        "loss_bag_selfguided": f"E6L__bag_selfguided__seed{seed}",
        "loss_full": f"E6L__full__seed{seed}",
    }


def _load_baseline_choices(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    freeze_path = args.baseline_choice_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_baseline_choice_freeze_sha256:
        raise ValueError("baseline choice freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = args.baseline_choice_root / "selection_manifest.csv"
    if (
        freeze.get("stage") != "final_rich_gallery_choice_freeze_v1"
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("candidate_manifest_sha256")
        != args.val_candidate_manifest_sha256
        or freeze.get("g1_checkpoint_sha256")
        != args.expected_g1_checkpoint_sha256
        or freeze.get("selection_manifest_sha256") != sha256_file(manifest_path)
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("baseline choices violate the frozen G4 contract")
    indexed = {row["image_id"]: row for row in _read_csv(manifest_path)}
    if len(indexed) != 371 or set(indexed) != {row["image_id"] for row in rows}:
        raise ValueError("baseline choice cohort mismatch")
    return indexed


def _apply_feature_mask(
    descriptors: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if descriptors.shape[-1] != mask.numel():
        raise ValueError("feature mask differs from descriptor layout")
    return descriptors * mask


def _instance_term(
    mode: str,
    logits: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if mode == "bag_only":
        return logits.sum() * 0.0
    if mode == "bag_negative":
        return negative_bag_instance_loss(logits, valid, labels)
    if mode in {"bag_selfguided", "full"}:
        return self_guided_instance_loss(logits, valid, labels)
    raise ValueError(f"unknown G1 loss arm: {mode}")


def train_arm(
    cache: list[dict[str, object]],
    config: MaskBagMILConfig,
    *,
    feature_arm: str,
    loss_arm: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RadDinoMaskBagMIL, list[dict[str, float]]]:
    seed_everything(seed)
    model = RadDinoMaskBagMIL(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    feature_mask = descriptor_feature_mask(config, feature_arm, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(cache))
        sums = {"total": 0.0, "image": 0.0, "instance": 0.0, "consistency": 0.0}
        batches = 0
        for start in range(0, len(order), args.train_batch_size):
            indices = order[start : start + args.train_batch_size]
            original, valid, labels = _padded_batch(cache, indices, "descriptors", device)
            flipped, flipped_valid, _ = _padded_batch(
                cache, indices, "flipped_descriptors", device
            )
            if not torch.equal(valid, flipped_valid):
                raise RuntimeError("original/flip candidate validity differs")
            original = _apply_feature_mask(original, feature_mask)
            flipped = _apply_feature_mask(flipped, feature_mask)
            logits, bag_logits = model.score_descriptors(original, valid)
            flip_logits, flip_bag_logits = model.score_descriptors(flipped, valid)
            image = 0.5 * (
                image_bag_loss(bag_logits, labels)
                + image_bag_loss(flip_bag_logits, labels)
            )
            if epoch > args.instance_warmup_epochs and loss_arm != "bag_only":
                instance = 0.5 * (
                    _instance_term(loss_arm, logits, valid, labels)
                    + _instance_term(loss_arm, flip_logits, valid, labels)
                )
            else:
                instance = logits.sum() * 0.0
            consistency = (
                aligned_candidate_consistency_loss(logits, flip_logits, valid)
                if loss_arm == "full"
                else logits.sum() * 0.0
            )
            total = image + args.instance_loss_weight * instance
            if loss_arm == "full":
                total = total + args.consistency_loss_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for name, value in (
                ("total", total),
                ("image", image),
                ("instance", instance),
                ("consistency", consistency),
            ):
                sums[name] += float(value.detach().item())
            batches += 1
        record = {"epoch": float(epoch)}
        record.update({name: value / batches for name, value in sums.items()})
        history.append(record)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "feature_arm": feature_arm,
                    "loss_arm": loss_arm,
                    **record,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model, history


def score_validation(
    model: RadDinoMaskBagMIL,
    cache: list[dict[str, object]],
    config: MaskBagMILConfig,
    *,
    feature_arm: str,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    feature_mask = descriptor_feature_mask(config, feature_arm, device=device)
    rows: list[dict[str, object]] = []
    for record in cache:
        original = torch.from_numpy(
            np.asarray(record["descriptors"], dtype=np.float32)
        )[None].to(device)
        flipped = torch.from_numpy(
            np.asarray(record["flipped_descriptors"], dtype=np.float32)
        )[None].to(device)
        valid = torch.ones(original.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits_a, _ = model.score_descriptors(
                _apply_feature_mask(original, feature_mask), valid
            )
            logits_b, _ = model.score_descriptors(
                _apply_feature_mask(flipped, feature_mask), valid
            )
            logits = 0.5 * (logits_a + logits_b)
            bag_logit = smooth_mil_pool(
                logits, valid, temperature=config.bag_temperature
            )[0]
        rows.append(
            {
                "image_id": str(record["image_id"]),
                "group_id": str(record["group_id"]),
                "tumor": int(record["label"]),
                "candidate_payload_sha256": str(record["candidate_payload_sha256"]),
                "candidate_indices": np.asarray(record["kept_indices"], dtype=np.int32),
                "candidate_logits": logits[0].float().cpu().numpy(),
                "bag_logit": float(bag_logit.item()),
                "bag_probability": float(torch.sigmoid(bag_logit).item()),
            }
        )
    return rows


def _candidate_metadata(
    root: Path,
    manifest_row: dict[str, str],
) -> dict[str, np.ndarray]:
    path = root / manifest_row["diagnostic_path"]
    if sha256_file(path) != manifest_row["diagnostic_sha256"]:
        raise ValueError("candidate payload changed before G1 choice freezing")
    with np.load(path, allow_pickle=False) as payload:
        result = {
            "upstream": payload["selection_scores"].astype(np.float64).reshape(-1),
            "sam": payload["sam_scores"].astype(np.float64).reshape(-1),
            "sources": payload["proposal_source_ids"].astype(str).reshape(-1),
            "prompt_modes": payload["prompt_modes"].astype(str).reshape(-1),
        }
    if len({len(value) for value in result.values()}) != 1:
        raise ValueError("candidate metadata arrays are not aligned")
    return result


def _choice_row(
    *,
    arm: str,
    split_row: dict[str, str],
    scored: dict[str, object],
    candidate_row: dict[str, str],
    metadata: dict[str, np.ndarray],
    selected_index: int,
    selected_logit: float,
) -> dict[str, object]:
    eligible = np.asarray(scored["candidate_indices"], dtype=np.int32)
    count = len(metadata["upstream"])
    if (
        selected_index < 0
        or selected_index >= count
        or selected_index not in set(eligible.tolist())
    ):
        raise ValueError(f"invalid frozen choice: {split_row['image_id']}/{arm}")
    return {
        "image_id": split_row["image_id"],
        "group_id": split_row["group_id"],
        "tumor": split_row["tumor"],
        "arm": arm,
        "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
        "gallery_candidate_count": count,
        "g1_eligible_candidate_count": len(eligible),
        "eligible_candidate_count": len(eligible),
        "eligible_candidate_indices": ";".join(str(int(index)) for index in eligible),
        "selected_candidate_index": selected_index,
        "selected_source": str(metadata["sources"][selected_index]),
        "selected_prompt_mode": str(metadata["prompt_modes"][selected_index]),
        "selected_sam_score": float(metadata["sam"][selected_index]),
        "selected_upstream_score": float(metadata["upstream"][selected_index]),
        "selected_g1_logit": selected_logit,
    }


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if (
        args.input_size != 448
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.maximum_candidates != 243
        or args.epochs != 16
        or args.train_batch_size != 16
        or args.instance_warmup_epochs != 2
    ):
        raise ValueError("G4 E6 runtime differs from the frozen G1 protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    if (
        len(train_rows) != 2981
        or sum(int(row["tumor"]) for row in train_rows) != 1488
        or len(val_rows) != 371
        or sum(int(row["tumor"]) for row in val_rows) != 184
    ):
        raise ValueError("G4 E6 requires the canonical train/validation cohort")
    baseline_choices = _load_baseline_choices(args, val_rows)
    train_candidates, train_candidate_audit = _audit_candidate_input(
        args.train_candidate_root,
        train_rows,
        split="train",
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
    )
    val_candidates, val_candidate_audit = _audit_candidate_input(
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
        raise RuntimeError("unexpected transformers version")
    runtime = require_cuda_runtime()
    device = runtime.primary_device
    projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder = place_frozen_encoder(
        ProjectedMultiLayerEncoder(backbone, torch.from_numpy(projection)), runtime
    )
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

    scored_by_arm: dict[str, list[dict[str, object]]] = {}
    histories: dict[str, object] = {}
    checkpoints: dict[str, str] = {}
    label_metrics: dict[str, object] = {}
    checkpoint_root = args.output_dir / "checkpoints"
    checkpoint_root.mkdir()
    for seed in seeds:
        unique_scores: dict[str, list[dict[str, object]]] = {}
        for spec in _training_specs():
            model, history = train_arm(
                train_cache,
                config,
                feature_arm=spec["feature"],
                loss_arm=spec["loss"],
                seed=seed,
                args=args,
                device=device,
            )
            scores = score_validation(
                model,
                val_cache,
                config,
                feature_arm=spec["feature"],
                device=device,
            )
            unique_scores[spec["key"]] = scores
            unique_name = f"{spec['key']}__seed{seed}"
            checkpoint_path = checkpoint_root / f"{unique_name}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "feature_arm": spec["feature"],
                    "loss_arm": spec["loss"],
                    "seed": seed,
                    "source_commit": args.source_commit,
                    "protocol_sha256": args.protocol_sha256,
                    "split_sha256": args.expected_split_sha256,
                    "validation_gt_read": False,
                    "test_evaluated": False,
                },
                checkpoint_path,
            )
            checkpoints[unique_name] = sha256_file(checkpoint_path)
            histories[unique_name] = history
            del model
            torch.cuda.empty_cache()

        names = _reported_arm_names(seed)
        aliases = {
            names["feature_inside_only"]: unique_scores["feature_inside_only"],
            names["feature_inside_ring"]: unique_scores["feature_inside_ring"],
            names["feature_inside_ring_contrast"]: unique_scores[
                "feature_inside_ring_contrast"
            ],
            names["feature_full"]: unique_scores["full"],
            names["loss_bag_only"]: unique_scores["loss_bag_only"],
            names["loss_bag_negative"]: unique_scores["loss_bag_negative"],
            names["loss_bag_selfguided"]: unique_scores["loss_bag_selfguided"],
            names["loss_full"]: unique_scores["full"],
        }
        for arm, scores in aliases.items():
            scored_by_arm[arm] = scores
            y_true = np.asarray([int(row["tumor"]) for row in scores], dtype=np.int64)
            probability = np.asarray(
                [float(row["bag_probability"]) for row in scores], dtype=np.float64
            )
            label_metrics[arm] = _binary_metrics(y_true, probability)

    history_path = args.output_dir / "training_histories.json"
    history_path.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    label_path = args.output_dir / "image_label_metrics.json"
    label_path.write_text(
        json.dumps(label_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    val_cache_by_id = {str(record["image_id"]): record for record in val_cache}
    scored_indexes = {
        arm: {str(row["image_id"]): row for row in scores}
        for arm, scores in scored_by_arm.items()
    }
    arm_names = ["E8__R7", *scored_by_arm]
    choice_rows: list[dict[str, object]] = []
    baseline_matches = 0
    for split_row in val_rows:
        image_id = split_row["image_id"]
        cache_record = val_cache_by_id[image_id]
        candidate_row = val_candidates[Path(image_id).stem]
        metadata = _candidate_metadata(args.val_candidate_root, candidate_row)
        eligible = np.asarray(cache_record["kept_indices"], dtype=np.int32)
        baseline = baseline_choices[image_id]
        baseline_index = int(baseline["selected_candidate_index"])
        if baseline["candidate_payload_sha256"] != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"baseline candidate changed: {image_id}")
        baseline_local = int(np.flatnonzero(eligible == baseline_index)[0])
        choice_rows.append(
            _choice_row(
                arm="E8__R7",
                split_row=split_row,
                scored={
                    "candidate_indices": eligible,
                },
                candidate_row=candidate_row,
                metadata=metadata,
                selected_index=baseline_index,
                selected_logit=float(baseline["selected_g1_logit"]),
            )
        )
        if baseline_local >= 0:
            baseline_matches += 1
        for arm in scored_by_arm:
            scored = scored_indexes[arm][image_id]
            kept = np.asarray(scored["candidate_indices"], dtype=np.int32)
            upstream = metadata["upstream"][kept]
            logits = np.asarray(scored["candidate_logits"], dtype=np.float64)
            local, _fused = select_candidate(logits, upstream)
            selected = int(kept[local])
            choice_rows.append(
                _choice_row(
                    arm=arm,
                    split_row=split_row,
                    scored=scored,
                    candidate_row=candidate_row,
                    metadata=metadata,
                    selected_index=selected,
                    selected_logit=float(logits[local]),
                )
            )
    if baseline_matches != 371:
        raise ValueError("baseline selected candidates do not lie in the cached G1 bags")
    choices_path = args.output_dir / "g4_choices.csv"
    choices_sha = _write_csv(choices_path, choice_rows)
    freeze = {
        "schema_version": 1,
        "stage": "g4_offline_ablation_choice_freeze_v1",
        "study": "G4 E6 matched G1 feature/loss ablations",
        "cohort_split": "val",
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "baseline_freeze_sha256": args.expected_baseline_choice_freeze_sha256,
        "g1_freeze_sha256": "retrained_matched_arms_bound_below",
        "protocol_sha256": args.protocol_sha256,
        "source_commit": args.source_commit,
        "choices_sha256": choices_sha,
        "training_histories_sha256": sha256_file(history_path),
        "image_label_metrics_sha256": sha256_file(label_path),
        "checkpoint_sha256": checkpoints,
        "images": 371,
        "tumor_images": 184,
        "arms": arm_names,
        "selection_rows": len(choice_rows),
        "seeds": list(seeds),
        "unique_models_per_seed": len(_training_specs()),
        "reported_learned_arms_per_seed": 8,
        "full_feature_and_full_loss_alias_exact": True,
        "baseline_r7_exact_matches": baseline_matches,
        "descriptor_cache_encoded_once": True,
        "descriptor_blocks": {
            "inside": [0, 384],
            "local_ring": [384, 768],
            "inside_minus_ring": [768, 1152],
            "metadata": [1152, 1156],
        },
        "candidate_inputs": {
            "train": train_candidate_audit,
            "validation": val_candidate_audit,
        },
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "eligible_candidate_indices_frozen_per_image_arm": True,
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "limitations": {
            "patient_identity": "group_id is the frozen filename/metadata heuristic, not a verified patient identifier",
            "feature_zeroing": "removed feature blocks are zeroed to keep scorer dimensionality and parameter count matched",
        },
    }
    freeze_path = args.output_dir / "g4_choice_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "study": freeze["study"],
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
            "seeds": list(seeds),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_devices": list(runtime.device_names),
            "encoder_data_parallel": runtime.encoder_data_parallel,
        },
        "choice_freeze_sha256": sha256_file(freeze_path),
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "choice_freeze_sha256": sha256_file(freeze_path),
                "arms": len(arm_names),
                "selection_rows": len(choice_rows),
                "validation_gt_read": False,
                "test_images_read": 0,
                "test_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
