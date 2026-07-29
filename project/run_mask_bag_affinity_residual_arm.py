from __future__ import annotations

"""Run the isolated R2 RAD-DINO affinity residual selector arm."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_affinity_residual_training import (
    AFFINITY_DIM,
    AffinityResidualTrainingConfig,
    score_affinity_residual_records,
    train_affinity_residual_adapter,
)
from models.mask_bag_descriptor_residual import AuxiliaryDescriptorResidual
from models.mask_bag_residual_objective import ResidualObjectiveConfig
from run_mask_bag_normal_prototype_arm import (
    _initial_adapter_state,
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)


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
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--residual-drift-weight", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_affinity_cache(records: list[dict[str, Any]]) -> int:
    if not records:
        raise ValueError("R2 cache cannot be empty")
    descriptor_dim = int(np.asarray(records[0]["descriptors"]).shape[1])
    for record in records:
        count = len(record["candidate_indices"])
        expected_affinity = (count, AFFINITY_DIM)
        if (
            np.asarray(record["descriptors"]).shape != (count, descriptor_dim)
            or np.asarray(record["flipped_descriptors"]).shape
            != (count, descriptor_dim)
            or np.asarray(record["affinity_features"]).shape
            != expected_affinity
            or np.asarray(record["flipped_affinity_features"]).shape
            != expected_affinity
        ):
            raise ValueError("R2 cache descriptor/affinity alignment mismatch")
    return descriptor_dim


def _score_device_shard(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    adapter_state: Mapping[str, torch.Tensor],
    *,
    descriptor_dim: int,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, config = _load_baseline_model(args, device=device)
    adapter = AuxiliaryDescriptorResidual(
        base_descriptor_dim=descriptor_dim,
        auxiliary_dim=AFFINITY_DIM,
        hidden_dim=args.adapter_hidden_dim,
    ).to(device)
    adapter.load_state_dict(adapter_state, strict=True)
    scored = score_affinity_residual_records(
        records,
        base,
        adapter,
        bag_temperature=config.bag_temperature,
        batch_size=args.batch_size,
        device=device,
    )
    del base, adapter
    torch.cuda.empty_cache()
    return scored


def main() -> None:
    args = parse_args()
    if (
        args.epochs != 16
        or args.batch_size != 16
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.adapter_hidden_dim != 128
        or args.consistency_weight != 0.10
        or args.residual_drift_weight != 1.0e-3
        or args.seed != 42
    ):
        raise ValueError("R2 execution differs from the frozen finite contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("R2 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"R2 requires Tesla T4 x2, got {device_names}")
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
        raise RuntimeError("frozen train/validation cohort mismatch")
    cache, validated_cache_rows = _load_cache_records(
        args,
        split_rows,
        cache_manifest_rows,
    )
    train_records = cache["train"]
    val_records = cache["val"]
    descriptor_dim = _validate_affinity_cache(train_records + val_records)

    training_config = AffinityResidualTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adapter_hidden_dim=args.adapter_hidden_dim,
        seed=args.seed,
    )
    objective_config = ResidualObjectiveConfig(
        bag_temperature=0.20,
        consistency_weight=args.consistency_weight,
        residual_drift_weight=args.residual_drift_weight,
    )
    training_device = torch.device("cuda:0")
    base, baseline_config = _load_baseline_model(args, device=training_device)
    initial_state = _initial_adapter_state(
        descriptor_dim=descriptor_dim,
        auxiliary_dim=AFFINITY_DIM,
        hidden_dim=args.adapter_hidden_dim,
        seed=args.seed,
    )
    adapter, history = train_affinity_residual_adapter(
        train_records,
        base,
        descriptor_dim=descriptor_dim,
        objective_config=objective_config,
        training_config=training_config,
        device=training_device,
        initial_adapter_state=initial_state,
    )
    adapter_state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    del base, adapter
    torch.cuda.empty_cache()

    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = args.output_dir / "affinity_residual.pt"
    torch.save(
        {
            "adapter_state_dict": adapter_state,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_config": asdict(baseline_config),
            "affinity_dim": AFFINITY_DIM,
            "objective_config": asdict(objective_config),
            "training_config": asdict(training_config),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": (
                args.expected_selector_cache_freeze_sha256
            ),
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )

    shards = [val_records[::2], val_records[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_device_shard,
                args,
                shards[device_index],
                adapter_state,
                descriptor_dim=descriptor_dim,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        unordered_scored = [
            prediction for future in futures for prediction in future.result()
        ]
    by_image = {
        str(prediction["image_id"]): prediction
        for prediction in unordered_scored
    }
    if len(unordered_scored) != 371 or len(by_image) != 371:
        raise RuntimeError("R2 T4x2 scoring does not cover 371 unique images")
    scored_val = [by_image[str(record["image_id"])] for record in val_records]
    prediction_manifest_sha256, score_manifest_sha256 = (
        _write_validation_outputs(args, val_records, scored_val)
    )

    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": (
            args.expected_selector_cache_freeze_sha256
        ),
        "selector_cache_manifest_sha256": cache_freeze[
            "selector_cache_manifest_sha256"
        ],
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "affinity_dim": AFFINITY_DIM,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history_sha256": sha256_file(history_path),
        "candidate_score_manifest_sha256": score_manifest_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "validation_predictions": 371,
        "training_labels": "image_level_only",
        "epoch_selection": "fixed_final_epoch_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "run_id": "btxrd_mask_bag_affinity_residual_r2_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "objective_config": asdict(objective_config),
        "training_config": asdict(training_config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "training_device": device_names[0],
            "validation_scoring_workers": 2,
            "validation_shards": [len(shards[0]), len(shards[1])],
        },
        "validated_cache_records": {
            "train": len(validated_cache_rows["train"]),
            "validation": len(validated_cache_rows["val"]),
        },
        "output_hashes": freeze,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
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
