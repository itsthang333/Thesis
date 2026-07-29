from __future__ import annotations

"""Run the matched S1 standard-vs-family-balanced residual pair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_pooling_residual_training import (
    DescriptorOnlyResidual,
    POOL_MODES,
    PoolingResidualTrainingConfig,
    score_pooling_residual_records,
    train_pooling_residual_adapter,
)
from models.mask_bag_residual_objective import ResidualObjectiveConfig
from run_mask_bag_normal_prototype_arm import (
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--residual-drift-weight", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _initial_residual_state(
    *,
    descriptor_dim: int,
    hidden_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    adapter = DescriptorOnlyResidual(
        descriptor_dim=descriptor_dim,
        hidden_dim=hidden_dim,
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }


def _run_matched_arm(
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    initial_state: Mapping[str, torch.Tensor],
    *,
    descriptor_dim: int,
    pool_mode: str,
    device_index: int,
    objective_config: ResidualObjectiveConfig,
    training_config: PoolingResidualTrainingConfig,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, baseline_config = _load_baseline_model(args, device=device)
    probe_adapter = DescriptorOnlyResidual(
        descriptor_dim=descriptor_dim,
        hidden_dim=training_config.hidden_dim,
    ).to(device)
    probe_adapter.load_state_dict(initial_state, strict=True)
    initial_probe = score_pooling_residual_records(
        train_records[:8],
        base,
        probe_adapter,
        pool_mode=pool_mode,
        bag_temperature=objective_config.bag_temperature,
        batch_size=8,
        device=device,
    )
    del probe_adapter
    adapter, history = train_pooling_residual_adapter(
        train_records,
        base,
        descriptor_dim=descriptor_dim,
        pool_mode=pool_mode,
        objective_config=objective_config,
        training_config=training_config,
        device=device,
        initial_adapter_state=initial_state,
    )
    scored = score_pooling_residual_records(
        val_records,
        base,
        adapter,
        pool_mode=pool_mode,
        bag_temperature=objective_config.bag_temperature,
        batch_size=training_config.batch_size,
        device=device,
    )
    adapter_state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    del base, adapter
    torch.cuda.empty_cache()
    return {
        "pool_mode": pool_mode,
        "device_index": device_index,
        "baseline_config": asdict(baseline_config),
        "adapter_state_dict": adapter_state,
        "initial_probe_candidate_logits": [
            row["candidate_logits"] for row in initial_probe
        ],
        "training_history": history,
        "validation_scores": scored,
    }


def _freeze_arm(
    args: argparse.Namespace,
    result: dict[str, Any],
    val_records: list[dict[str, Any]],
    *,
    cache_freeze: dict[str, Any],
    objective_config: ResidualObjectiveConfig,
    training_config: PoolingResidualTrainingConfig,
    initial_state_sha256: str,
) -> dict[str, Any]:
    pool_mode = str(result["pool_mode"])
    arm_root = args.output_dir / pool_mode
    arm_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = arm_root / "descriptor_residual.pt"
    torch.save(
        {
            "adapter_state_dict": result["adapter_state_dict"],
            "pool_mode": pool_mode,
            "initial_state_sha256": initial_state_sha256,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_config": result["baseline_config"],
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
    history_path = arm_root / "training_history.json"
    history_path.write_text(
        json.dumps(result["training_history"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_args = SimpleNamespace(output_dir=arm_root)
    prediction_manifest_sha256, score_manifest_sha256 = (
        _write_validation_outputs(
            output_args,
            val_records,
            result["validation_scores"],
        )
    )
    freeze = {
        "arm": pool_mode,
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
        "initial_state_sha256": initial_state_sha256,
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
    freeze_path = arm_root / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **freeze,
        "prediction_freeze_sha256": sha256_file(freeze_path),
    }


def main() -> None:
    args = parse_args()
    if (
        args.epochs != 16
        or args.batch_size != 16
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.hidden_dim != 128
        or args.consistency_weight != 0.10
        or args.residual_drift_weight != 1.0e-3
        or args.seed != 42
    ):
        raise ValueError("S1 execution differs from the frozen matched contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S1 requires Tesla T4 x2, got {device_names}")
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
    descriptor_dim = int(np.asarray(train_records[0]["descriptors"]).shape[1])
    if any(
        np.asarray(record["descriptors"]).shape[1] != descriptor_dim
        for record in train_records + val_records
    ):
        raise ValueError("S1 cache descriptor dimensions are inconsistent")

    objective_config = ResidualObjectiveConfig(
        bag_temperature=0.20,
        consistency_weight=args.consistency_weight,
        residual_drift_weight=args.residual_drift_weight,
    )
    training_config = PoolingResidualTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    initial_state = _initial_residual_state(
        descriptor_dim=descriptor_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    initial_state_path = args.output_dir / "matched_initial_state.pt"
    torch.save(initial_state, initial_state_path)
    initial_state_sha256 = sha256_file(initial_state_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_matched_arm,
                args,
                train_records,
                val_records,
                initial_state,
                descriptor_dim=descriptor_dim,
                pool_mode=pool_mode,
                device_index=device_index,
                objective_config=objective_config,
                training_config=training_config,
            )
            for device_index, pool_mode in enumerate(POOL_MODES)
        ]
        results = [future.result() for future in futures]
    by_mode = {str(result["pool_mode"]): result for result in results}
    if set(by_mode) != set(POOL_MODES):
        raise RuntimeError("S1 matched pair is incomplete")
    probe_deltas = [
        float(np.max(np.abs(standard - balanced)))
        for standard, balanced in zip(
            by_mode["standard"]["initial_probe_candidate_logits"],
            by_mode["family_balanced"]["initial_probe_candidate_logits"],
            strict=True,
        )
    ]
    maximum_initial_probe_delta = max(probe_deltas, default=float("inf"))
    if maximum_initial_probe_delta > 5.0e-6:
        raise RuntimeError("S1 identical-T4 initial candidate logits differ")
    arm_freezes = {
        pool_mode: _freeze_arm(
            args,
            by_mode[pool_mode],
            val_records,
            cache_freeze=cache_freeze,
            objective_config=objective_config,
            training_config=training_config,
            initial_state_sha256=initial_state_sha256,
        )
        for pool_mode in POOL_MODES
    }
    pair_freeze = {
        "run_id": "btxrd_mask_bag_family_balanced_s1_pair_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": (
            args.expected_selector_cache_freeze_sha256
        ),
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "matched_variables": [
            "descriptor_cache",
            "frozen_baseline",
            "adapter_architecture",
            "adapter_initial_state",
            "batch_order",
            "optimizer",
            "epochs",
            "loss_weights",
            "validation_cohort",
        ],
        "sole_changed_variable": "standard_vs_family_balanced_bag_pool",
        "cross_device_initial_candidate_logit_max_delta": (
            maximum_initial_probe_delta
        ),
        "cross_device_initial_candidate_logit_tolerance": 5.0e-6,
        "initial_state_sha256": initial_state_sha256,
        "arms": arm_freezes,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    pair_freeze_path = args.output_dir / "pair_prediction_freeze.json"
    pair_freeze_path.write_text(
        json.dumps(pair_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        **pair_freeze,
        "pair_prediction_freeze_sha256": sha256_file(pair_freeze_path),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "parallel_training_workers": 2,
            "device_assignment": {
                "standard": device_names[0],
                "family_balanced": device_names[1],
            },
        },
        "validated_cache_records": {
            "train": len(validated_cache_rows["train"]),
            "validation": len(validated_cache_rows["val"]),
        },
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
