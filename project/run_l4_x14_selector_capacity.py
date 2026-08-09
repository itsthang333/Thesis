from __future__ import annotations

"""L4 X14 matched selector-capacity diagnostic.

The frozen RAD-DINO candidate descriptors, MIL objective, group-aware inner
split, optimizer, epoch budget and random seeds are shared across a linear,
one-hidden-layer and current two-hidden-layer selector.  This stage uses only
image labels and freezes outer-validation choices before any polygon is read.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from build_x4_inner_split import assign_inner_roles
from evaluate_g4_classifier_labels import _binary_metrics
from final_selector import select_candidate
from frozen_io import load_split_rows_without_annotations, sha256_file, verify_model_snapshot
from gpu_runtime import place_frozen_encoder, require_cuda_runtime
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    aligned_candidate_consistency_loss,
    image_bag_loss,
    self_guided_instance_loss,
    smooth_mil_pool,
)
from run_g4_g1_ablation import (
    _candidate_metadata,
    _choice_row,
    _load_baseline_choices,
    _parse_seeds,
    _read_csv,
    _write_csv,
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


ARCHITECTURES = ("linear", "one_hidden", "two_hidden")


class SelectorCapacityMIL(nn.Module):
    """Capacity-only variants with the exact current G1 scoring interface."""

    def __init__(self, config: MaskBagMILConfig, architecture: str) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown selector architecture: {architecture}")
        self.config = config
        self.architecture = architecture
        if architecture == "linear":
            self.scorer = nn.Sequential(
                nn.LayerNorm(config.descriptor_dim),
                nn.Linear(config.descriptor_dim, 1),
            )
        elif architecture == "one_hidden":
            self.scorer = nn.Sequential(
                nn.LayerNorm(config.descriptor_dim),
                nn.Linear(config.descriptor_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(config.hidden_dim, 1),
            )
        else:
            # Construct the deployed two-hidden-layer scorer verbatim rather
            # than a similar approximation; tests compare state-dict layout.
            self.scorer = RadDinoMaskBagMIL(config).scorer

    def score_descriptors(
        self, descriptors: torch.Tensor, candidate_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.config.descriptor_dim:
            raise ValueError("descriptors must have shape [B,N,descriptor_dim]")
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid must align with descriptors")
        valid = candidate_valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every bag must contain a valid proposal")
        logits = self.scorer(descriptors).squeeze(-1).masked_fill(~valid, 0.0)
        bag_logits = smooth_mil_pool(
            logits, valid, temperature=self.config.bag_temperature
        )
        return logits, bag_logits


def architecture_parameter_count(config: MaskBagMILConfig, architecture: str) -> int:
    return sum(
        parameter.numel()
        for parameter in SelectorCapacityMIL(config, architecture).parameters()
        if parameter.requires_grad
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


def _train_model(
    cache: list[dict[str, object]],
    config: MaskBagMILConfig,
    architecture: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SelectorCapacityMIL, list[dict[str, float]]]:
    seed_everything(seed)
    model = SelectorCapacityMIL(config, architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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
            logits, bag_logits = model.score_descriptors(original, valid)
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
            consistency_loss = aligned_candidate_consistency_loss(
                logits, flip_logits, valid
            )
            total = (
                image_loss
                + args.instance_loss_weight * instance_loss
                + args.consistency_loss_weight * consistency_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for key, value in (
                ("total", total),
                ("image", image_loss),
                ("instance", instance_loss),
                ("consistency", consistency_loss),
            ):
                sums[key] += float(value.detach().item())
            batches += 1
        record = {"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}}
        history.append(record)
        print(json.dumps({"architecture": architecture, "seed": seed, **record}, sort_keys=True), flush=True)
    return model, history


def _score_cache(
    model: SelectorCapacityMIL,
    cache: list[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for record in cache:
        original = torch.from_numpy(np.asarray(record["descriptors"], dtype=np.float32))[None].to(device)
        flipped = torch.from_numpy(np.asarray(record["flipped_descriptors"], dtype=np.float32))[None].to(device)
        valid = torch.ones(original.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits_a, _bag_a = model.score_descriptors(original, valid)
            logits_b, _bag_b = model.score_descriptors(flipped, valid)
            logits = 0.5 * (logits_a + logits_b)
            bag_logit = smooth_mil_pool(
                logits, valid, temperature=model.config.bag_temperature
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


def _stable_argmax(values: np.ndarray, candidate_indices: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 1 or values.shape != candidate_indices.shape or len(values) == 0:
        raise ValueError("argmax inputs must be aligned non-empty vectors")
    return int(np.lexsort((candidate_indices, -values))[0])


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
        raise ValueError("L4 X14 runtime differs from the frozen matched protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if (
        len(train_rows) != 2981
        or sum(int(row["tumor"]) for row in train_rows) != 1488
        or len(val_rows) != 371
        or sum(int(row["tumor"]) for row in val_rows) != 184
    ):
        raise ValueError("L4 X14 requires the canonical train/validation cohort")

    inner_rows = assign_inner_roles(train_rows)
    inner_path = args.output_dir / "x14_inner_split.csv"
    inner_sha = _write_csv(inner_path, inner_rows)
    role_by_id = {str(row["image_id"]): str(row["inner_role"]) for row in inner_rows}
    if set(role_by_id) != {row["image_id"] for row in train_rows}:
        raise ValueError("inner split differs from canonical training images")

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
        input_dim=768, output_dim=args.projection_dim, seed=args.projection_seed
    )
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder = place_frozen_encoder(
        ProjectedMultiLayerEncoder(backbone, torch.from_numpy(projection)), runtime
    )
    config = MaskBagMILConfig(
        token_dim=args.projection_dim, token_layers=len(SELECTED_HIDDEN_LAYERS)
    )
    train_cache_all = build_descriptor_cache(
        train_rows, train_candidates, args.train_candidate_root, encoder, config, args, device, split="train"
    )
    val_cache = build_descriptor_cache(
        val_rows, val_candidates, args.val_candidate_root, encoder, config, args, device, split="val"
    )
    del encoder, backbone
    torch.cuda.empty_cache()
    inner_train_cache = [record for record in train_cache_all if role_by_id[str(record["image_id"])] == "inner_train"]
    inner_holdout_cache = [record for record in train_cache_all if role_by_id[str(record["image_id"])] == "inner_holdout"]
    if len(inner_train_cache) + len(inner_holdout_cache) != 2981:
        raise RuntimeError("inner cache partition differs")

    scored_outer: dict[tuple[str, int], list[dict[str, object]]] = {}
    histories: dict[str, object] = {}
    label_metrics: dict[str, object] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_root = args.output_dir / "checkpoints"
    checkpoint_root.mkdir()
    for seed in seeds:
        for architecture in ARCHITECTURES:
            model, history = _train_model(
                inner_train_cache, config, architecture, seed, args, device
            )
            holdout_scores = _score_cache(model, inner_holdout_cache, device)
            outer_scores = _score_cache(model, val_cache, device)
            key = f"{architecture}__seed{seed}"
            histories[key] = history
            y_true = np.asarray([int(row["tumor"]) for row in holdout_scores], dtype=np.int64)
            probability = np.asarray([float(row["bag_probability"]) for row in holdout_scores], dtype=np.float64)
            label_metrics[key] = _binary_metrics(y_true, probability)
            scored_outer[(architecture, seed)] = outer_scores
            checkpoint_path = checkpoint_root / f"{key}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "architecture": architecture,
                    "trainable_parameters": architecture_parameter_count(config, architecture),
                    "seed": seed,
                    "source_commit": args.source_commit,
                    "protocol_sha256": args.protocol_sha256,
                    "split_sha256": args.expected_split_sha256,
                    "inner_split_sha256": inner_sha,
                    "validation_gt_read": False,
                    "test_images_read": 0,
                    "test_evaluated": False,
                },
                checkpoint_path,
            )
            checkpoint_hashes[key] = sha256_file(checkpoint_path)
            del model
            torch.cuda.empty_cache()

    history_path = args.output_dir / "training_histories.json"
    history_path.write_text(json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_path = args.output_dir / "inner_holdout_image_label_metrics.json"
    label_path.write_text(json.dumps(label_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    val_cache_by_id = {str(record["image_id"]): record for record in val_cache}
    score_index = {
        key: {str(row["image_id"]): row for row in rows}
        for key, rows in scored_outer.items()
    }
    learned_arms = [
        f"X14__{architecture}_{mode}__seed{seed}"
        for seed in seeds
        for architecture in ARCHITECTURES
        for mode in ("only", "r7")
    ]
    arms = ["E8__R7", "X14__upstream", *learned_arms]
    choices: list[dict[str, object]] = []
    baseline_matches = 0
    for split_row in val_rows:
        image_id = split_row["image_id"]
        candidate_row = val_candidates[Path(image_id).stem]
        metadata = _candidate_metadata(args.val_candidate_root, candidate_row)
        eligible = np.asarray(val_cache_by_id[image_id]["kept_indices"], dtype=np.int32)
        baseline = baseline_choices[image_id]
        baseline_index = int(baseline["selected_candidate_index"])
        if baseline["candidate_payload_sha256"] != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"baseline candidate changed: {image_id}")
        if baseline_index not in set(eligible.tolist()):
            raise ValueError(f"baseline candidate absent from bag: {image_id}")
        choices.append(
            _choice_row(
                arm="E8__R7",
                split_row=split_row,
                scored={"candidate_indices": eligible},
                candidate_row=candidate_row,
                metadata=metadata,
                selected_index=baseline_index,
                selected_logit=float(baseline["selected_g1_logit"]),
            )
        )
        baseline_matches += 1
        upstream_local = _stable_argmax(metadata["upstream"][eligible], eligible)
        choices.append(
            _choice_row(
                arm="X14__upstream",
                split_row=split_row,
                scored={"candidate_indices": eligible},
                candidate_row=candidate_row,
                metadata=metadata,
                selected_index=int(eligible[upstream_local]),
                selected_logit=0.0,
            )
        )
        for seed in seeds:
            for architecture in ARCHITECTURES:
                scored = score_index[(architecture, seed)][image_id]
                kept = np.asarray(scored["candidate_indices"], dtype=np.int32)
                logits = np.asarray(scored["candidate_logits"], dtype=np.float64)
                only_local = _stable_argmax(logits, kept)
                r7_local, _ = select_candidate(logits, metadata["upstream"][kept])
                for mode, local in (("only", only_local), ("r7", int(r7_local))):
                    choices.append(
                        _choice_row(
                            arm=f"X14__{architecture}_{mode}__seed{seed}",
                            split_row=split_row,
                            scored=scored,
                            candidate_row=candidate_row,
                            metadata=metadata,
                            selected_index=int(kept[local]),
                            selected_logit=float(logits[local]),
                        )
                    )
    if baseline_matches != 371 or len(choices) != 371 * len(arms):
        raise RuntimeError("X14 frozen choice matrix is incomplete")
    choices_path = args.output_dir / "g4_choices.csv"
    choices_sha = _write_csv(choices_path, choices)
    freeze = {
        "schema_version": 1,
        "stage": "l4_x14_selector_capacity_choice_freeze_v1",
        "study": "L4 X14 matched selector-capacity diagnostic",
        "baseline_arm": "E8__R7",
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "baseline_freeze_sha256": args.expected_baseline_choice_freeze_sha256,
        "g1_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "protocol_sha256": args.protocol_sha256,
        "source_commit": args.source_commit,
        "inner_split_sha256": inner_sha,
        "choices_sha256": choices_sha,
        "training_histories_sha256": sha256_file(history_path),
        "inner_holdout_metrics_sha256": sha256_file(label_path),
        "checkpoint_sha256": checkpoint_hashes,
        "images": 371,
        "tumor_images": 184,
        "arms": arms,
        "selection_rows": len(choices),
        "architectures": list(ARCHITECTURES),
        "architecture_trainable_parameters": {
            architecture: architecture_parameter_count(config, architecture)
            for architecture in ARCHITECTURES
        },
        "seeds": list(seeds),
        "same_descriptor_cache": True,
        "same_inner_split": True,
        "same_mil_objective_optimizer_epochs": True,
        "no_best_epoch_selection": True,
        "candidate_inputs": {"train": train_candidate_audit, "validation": val_candidate_audit},
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "baseline_r7_exact_matches": baseline_matches,
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "limitations": {
            "patient_identity": "group_id is the frozen filename/metadata heuristic, not a verified patient identifier",
            "outer_validation": "used only after all architecture choices are frozen; no architecture or epoch is selected from polygons",
        },
    }
    freeze_path = args.output_dir / "g4_choice_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "choice_freeze_sha256": sha256_file(freeze_path),
                "architectures": list(ARCHITECTURES),
                "models": len(checkpoint_hashes),
                "arms": len(arms),
                "selection_rows": len(choices),
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
