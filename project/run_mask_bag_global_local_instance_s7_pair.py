from __future__ import annotations

"""Run the frozen Geometry-v3 identity versus S7 instance-selector pair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_global_local_instance import (
    GlobalLocalInstanceConfig,
    GlobalLocalInstanceResidual,
)
from models.mask_bag_global_local_instance_training import (
    GlobalLocalInstanceTrainingConfig,
    attach_frozen_base_logits,
    audit_zero_initialization,
    initial_global_local_state,
    score_global_local_instance,
    train_global_local_instance,
)
from run_mask_bag_critical_relation_arm import (
    _absolute_spearman,
    _validate_descriptor_cache,
)
from run_mask_bag_normal_prototype_arm import (
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)


EXPERIMENT_ID = "EXP-20260802-codex-s7-global-local-instance-v1"
RUN_ID = "btxrd_mask_bag_global_local_instance_s7_pair_v1"
ARMS = ("geometry_v3_identity", "global_local_instance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--bag-temperature", type=float, default=0.20)
    parser.add_argument("--start-positive-mass", type=float, default=0.50)
    parser.add_argument("--target-positive-mass", type=float, default=0.15)
    parser.add_argument("--mass-transition-epochs", type=int, default=20)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--residual-drift-weight", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_recipe(args: argparse.Namespace) -> None:
    actual = (
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.hidden_dim,
        args.dropout,
        args.bag_temperature,
        args.start_positive_mass,
        args.target_positive_mass,
        args.mass_transition_epochs,
        args.consistency_weight,
        args.residual_drift_weight,
        args.seed,
    )
    expected = (
        40,
        16,
        3.0e-4,
        1.0e-4,
        128,
        0.10,
        0.20,
        0.50,
        0.15,
        20,
        0.10,
        1.0e-3,
        42,
    )
    if actual != expected:
        raise ValueError("S7 execution differs from the frozen one-shot recipe")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("cannot write empty S7 evidence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _write_target_snapshot(
    root: Path,
    manifest_rows: list[dict[str, Any]],
    epoch_index: int,
    records: Sequence[Mapping[str, Any]],
    logits: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    diagnostics: Mapping[str, Any],
) -> None:
    if not (
        len(records) == len(logits) == len(targets) == len(weights) == 2981
    ):
        raise RuntimeError("S7 target snapshot does not cover the train cohort")
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for index, values in enumerate(logits):
        offsets[index + 1] = offsets[index] + len(values)
    flat_logits = np.concatenate(
        [np.asarray(values, dtype=np.float32) for values in logits]
    )
    flat_targets = np.concatenate(
        [np.asarray(values, dtype=np.float32) for values in targets]
    )
    flat_weights = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in weights]
    )
    if not (
        len(flat_logits) == len(flat_targets) == len(flat_weights) == offsets[-1]
    ):
        raise RuntimeError("S7 target snapshot arrays do not align")
    max_id_length = max(len(str(record["image_id"])) for record in records)
    path = root / f"epoch_{epoch_index:02d}.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        epoch_index=np.asarray([epoch_index], dtype=np.int64),
        image_ids=np.asarray(
            [str(record["image_id"]) for record in records],
            dtype=f"<U{max_id_length}",
        ),
        labels=np.asarray([int(record["label"]) for record in records], dtype=np.int8),
        offsets=offsets,
        current_logits=flat_logits,
        soft_targets=flat_targets,
        candidate_weights=flat_weights,
    )
    manifest_rows.append(
        {
            "epoch_index": epoch_index,
            "target_positive_mass": float(diagnostics["target_positive_mass"]),
            "target_sha256": str(diagnostics["target_sha256"]),
            "snapshot_path": path.name,
            "snapshot_sha256": sha256_file(path),
            "candidate_count": int(offsets[-1]),
            "projected_mass_before_local": float(
                diagnostics["projected_mass_before_local"]
            ),
            "realized_mass_after_local": float(
                diagnostics["realized_mass_after_local"]
            ),
            "locally_forced_candidates": int(
                diagnostics["locally_forced_candidates"]
            ),
        }
    )


def _score_device_shard(
    records: list[dict[str, Any]],
    state: Mapping[str, torch.Tensor],
    *,
    model_config: GlobalLocalInstanceConfig,
    bag_temperature: float,
    batch_size: int,
    device_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    model = GlobalLocalInstanceResidual(model_config).to(device)
    model.load_state_dict(state, strict=True)
    smoke = torch.arange(4096, dtype=torch.float32, device=device).reshape(64, 64)
    smoke_checksum = float((smoke @ smoke.T).sum().item())
    scored = score_global_local_instance(
        records,
        model,
        bag_temperature=bag_temperature,
        batch_size=batch_size,
        device=device,
    )
    runtime = {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "records": len(records),
        "cuda_matmul_checksum": smoke_checksum,
    }
    del model, smoke
    torch.cuda.empty_cache()
    return scored, runtime


def _diagnostics(
    records: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, output in zip(records, scored):
        base = np.asarray(output["base_candidate_logits"], dtype=np.float32)
        primary = np.asarray(output["candidate_logits"], dtype=np.float32)
        rows.append(
            {
                "image_id": str(record["image_id"]),
                "tumor": int(record["label"]),
                "candidate_count": len(base),
                "base_selected_local_index": int(np.argmax(base)),
                "primary_selected_local_index": int(np.argmax(primary)),
                "selection_changed": int(np.argmax(base) != np.argmax(primary)),
                "bag_probability": float(output["bag_probability"]),
                "original_flip_agreement": int(output["original_flip_agreement"]),
                "mean_absolute_residual": float(np.mean(np.abs(primary - base))),
                "maximum_absolute_residual": float(np.max(np.abs(primary - base))),
            }
        )
    counts = np.asarray([int(row["candidate_count"]) for row in rows])
    probabilities = np.asarray([float(row["bag_probability"]) for row in rows])
    summary = {
        "records": len(rows),
        "changed_selection_count": int(sum(row["selection_changed"] for row in rows)),
        "changed_selection_fraction": float(
            np.mean([row["selection_changed"] for row in rows])
        ),
        "original_flip_agreement": float(
            np.mean([row["original_flip_agreement"] for row in rows])
        ),
        "absolute_candidate_count_probability_spearman": _absolute_spearman(
            counts, probabilities
        ),
        "mean_absolute_residual": float(
            np.mean([row["mean_absolute_residual"] for row in rows])
        ),
        "maximum_absolute_residual": float(
            np.max([row["maximum_absolute_residual"] for row in rows])
        ),
        "bag_probability_changed_from_accepted_baseline": False,
        "diagnostics_block_prediction_freeze": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    _validate_recipe(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S7 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S7 requires Tesla T4 x2, got {device_names}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)
    split_rows = {
        split: load_split_rows_without_annotations(
            args.split_manifest,
            expected_sha256=args.expected_split_sha256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != 2981 or len(split_rows["val"]) != 371:
        raise RuntimeError("S7 frozen train/validation cohort mismatch")
    cache, validated_cache_rows = _load_cache_records(
        args, split_rows, cache_manifest_rows
    )
    train_records = cache["train"]
    val_records = cache["val"]
    descriptor_dim = _validate_descriptor_cache(train_records + val_records)
    if descriptor_dim != 1156:
        raise RuntimeError("S7 descriptor dimension differs from the frozen design")
    if sum(int(record["label"]) for record in train_records) != 1488:
        raise RuntimeError("S7 train image-label count differs from the frozen cohort")

    training_device = torch.device("cuda:0")
    base, baseline_config = _load_baseline_model(args, device=training_device)
    if not math.isclose(
        baseline_config.bag_temperature,
        args.bag_temperature,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("S7 bag temperature differs from the accepted baseline")
    attach_frozen_base_logits(
        train_records, base, batch_size=args.batch_size, device=training_device
    )
    attach_frozen_base_logits(
        val_records, base, batch_size=args.batch_size, device=training_device
    )
    del base
    torch.cuda.empty_cache()

    model_config = GlobalLocalInstanceConfig(
        descriptor_dim=descriptor_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        start_positive_mass=args.start_positive_mass,
        target_positive_mass=args.target_positive_mass,
        mass_transition_epochs=args.mass_transition_epochs,
        total_epochs=args.epochs,
        consistency_weight=args.consistency_weight,
        residual_drift_weight=args.residual_drift_weight,
    )
    training_config = GlobalLocalInstanceTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    initial_state = initial_global_local_state(model_config, seed=args.seed)
    pretraining_identity = {
        "train": audit_zero_initialization(
            train_records,
            model_config=model_config,
            initial_state=initial_state,
            bag_temperature=args.bag_temperature,
            batch_size=args.batch_size,
            device=training_device,
        ),
        "validation": audit_zero_initialization(
            val_records,
            model_config=model_config,
            initial_state=initial_state,
            bag_temperature=args.bag_temperature,
            batch_size=args.batch_size,
            device=training_device,
        ),
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    for cohort in ("train", "validation"):
        audit = pretraining_identity[cohort]
        if (
            audit["exact_candidate_score_records"] != audit["records"]
            or audit["exact_selected_index_records"] != audit["records"]
            or audit["maximum_candidate_score_delta"] != 0.0
        ):
            raise RuntimeError(f"S7 zero-initialization identity failed for {cohort}")
    identity_path = args.output_dir / "pretraining_identity_audit.json"
    identity_path.write_text(
        json.dumps(pretraining_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target_root = args.output_dir / "target_snapshots"
    target_root.mkdir(parents=True, exist_ok=False)
    target_manifest_rows: list[dict[str, Any]] = []

    def snapshot_callback(
        epoch_index: int,
        records: Sequence[Mapping[str, Any]],
        logits: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        diagnostics: Mapping[str, Any],
    ) -> None:
        _write_target_snapshot(
            target_root,
            target_manifest_rows,
            epoch_index,
            records,
            logits,
            targets,
            weights,
            diagnostics,
        )

    model, history = train_global_local_instance(
        train_records,
        model_config=model_config,
        training_config=training_config,
        device=training_device,
        initial_state=initial_state,
        target_snapshot_callback=snapshot_callback,
    )
    if len(history) != 40 or len(target_manifest_rows) != 40:
        raise RuntimeError("S7 did not complete the frozen 40-epoch target history")
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_manifest_path = target_root / "target_snapshot_manifest.json"
    target_manifest_path.write_text(
        json.dumps(target_manifest_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    checkpoint_path = args.output_dir / "global_local_instance_residual.pt"
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model_state_dict": state,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "training_labels": "binary_image_level_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    del model
    torch.cuda.empty_cache()

    shards = [val_records[::2], val_records[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_device_shard,
                shards[device_index],
                state,
                model_config=model_config,
                bag_temperature=args.bag_temperature,
                batch_size=args.batch_size,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        shard_outputs = [future.result() for future in futures]
    unordered = [item for scored, _runtime in shard_outputs for item in scored]
    by_image = {str(item["image_id"]): item for item in unordered}
    if len(by_image) != 371:
        raise RuntimeError("S7 T4x2 scoring does not cover 371 unique images")
    scored = [by_image[str(record["image_id"])] for record in val_records]
    baseline_scored = [
        {
            "image_id": output["image_id"],
            "candidate_logits": output["base_candidate_logits"],
            "bag_logit": output["bag_logit"],
            "bag_probability": output["bag_probability"],
        }
        for output in scored
    ]
    primary_scored = [
        {
            "image_id": output["image_id"],
            "candidate_logits": output["candidate_logits"],
            "bag_logit": output["bag_logit"],
            "bag_probability": output["bag_probability"],
        }
        for output in scored
    ]
    diagnostic_rows, diagnostic_summary = _diagnostics(val_records, scored)
    diagnostic_path = args.output_dir / "gt_blind_diagnostics.csv"
    diagnostic_sha = _write_csv(diagnostic_path, diagnostic_rows)
    diagnostic_summary_path = args.output_dir / "gt_blind_diagnostic_summary.json"
    diagnostic_summary_path.write_text(
        json.dumps(diagnostic_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arm_freezes: dict[str, str] = {}
    for arm, arm_scores in zip(ARMS, (baseline_scored, primary_scored)):
        arm_root = args.output_dir / arm
        prediction_sha, score_sha = _write_validation_outputs(
            argparse.Namespace(output_dir=arm_root), val_records, arm_scores
        )
        freeze = {
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "selector_cache_manifest_sha256": cache_freeze[
                "selector_cache_manifest_sha256"
            ],
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "primary_checkpoint_sha256": sha256_file(checkpoint_path),
            "training_history_sha256": sha256_file(history_path),
            "target_snapshot_manifest_sha256": sha256_file(target_manifest_path),
            "pretraining_identity_audit_sha256": sha256_file(identity_path),
            "gt_blind_diagnostics_sha256": diagnostic_sha,
            "gt_blind_diagnostic_summary_sha256": sha256_file(
                diagnostic_summary_path
            ),
            "candidate_score_manifest_sha256": score_sha,
            "prediction_manifest_sha256": prediction_sha,
            "validation_predictions": 371,
            "accepted_bag_probability_preserved": True,
            "training_labels": "binary_image_level_only",
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }
        freeze_path = arm_root / "prediction_freeze.json"
        freeze_path.write_text(
            json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        arm_freezes[arm] = sha256_file(freeze_path)

    pair = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "arms": arm_freezes,
        "sole_changed_variable": "global_local_all_instance_selector_residual",
        "accepted_bag_probability_preserved": True,
        "pair_physically_frozen_before_validation_gt": True,
        "diagnostics_block_prediction_freeze": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_path = args.output_dir / "prediction_pair_freeze.json"
    pair_path.write_text(
        json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "validated_cache_records": {
            "train": len(validated_cache_rows["train"]),
            "validation": len(validated_cache_rows["val"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "training_device": device_names[0],
            "validation_shards": [len(shards[0]), len(shards[1])],
            "scoring_device_evidence": [runtime for _scored, runtime in shard_outputs],
        },
        "target_snapshot_manifest_sha256": sha256_file(target_manifest_path),
        "prediction_pair_freeze_sha256": sha256_file(pair_path),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    run_path = args.output_dir / "run_manifest.json"
    run_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
