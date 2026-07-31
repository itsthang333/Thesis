from __future__ import annotations

"""Run the isolated R3 DSMIL-style critical-relation selector arm."""

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
from models.mask_bag_critical_relation_training import (
    CriticalRelationTrainingConfig,
    audit_zero_initialization_records,
    initial_critical_relation_state,
    score_critical_relation_records,
    train_critical_relation_adapter,
)
from models.mask_bag_relational_selector import CriticalRelationResidual
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
    parser.add_argument("--instance-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--instance-warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--count-probability-spearman-ceiling",
        type=float,
        default=0.5013777759365411,
    )
    parser.add_argument("--minimum-critical-agreement-delta", type=float, default=-0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_descriptor_cache(records: list[dict[str, Any]]) -> int:
    if not records:
        raise ValueError("R3 cache cannot be empty")
    descriptor_dim = int(np.asarray(records[0]["descriptors"]).shape[1])
    if descriptor_dim != 1156:
        raise ValueError("R3 requires the frozen 1,156-D descriptor")
    for record in records:
        count = len(record["candidate_indices"])
        if (
            count <= 0
            or np.asarray(record["descriptors"]).shape != (count, descriptor_dim)
            or np.asarray(record["flipped_descriptors"]).shape
            != (count, descriptor_dim)
        ):
            raise ValueError("R3 cache descriptor/candidate alignment mismatch")
    return descriptor_dim


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank values must be one-dimensional and finite")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _absolute_spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Spearman inputs must have equal nontrivial length")
    ranks_a = _rankdata_average(values_a)
    ranks_b = _rankdata_average(values_b)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        raise ValueError("Spearman input is constant")
    return abs(float(np.corrcoef(ranks_a, ranks_b)[0, 1]))


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
    adapter = CriticalRelationResidual(descriptor_dim, args.hidden_dim).to(device)
    adapter.load_state_dict(adapter_state, strict=True)
    scored = score_critical_relation_records(
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
        or args.hidden_dim != 128
        or args.instance_loss_weight != 0.25
        or args.consistency_weight != 0.10
        or args.instance_warmup_epochs != 2
        or args.count_probability_spearman_ceiling != 0.5013777759365411
        or args.minimum_critical_agreement_delta != -0.01
        or args.seed != 42
    ):
        raise ValueError("R3 execution differs from the frozen finite contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("R3 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"R3 requires Tesla T4 x2, got {device_names}")
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
        args, split_rows, cache_manifest_rows
    )
    train_records = cache["train"]
    val_records = cache["val"]
    descriptor_dim = _validate_descriptor_cache(train_records + val_records)

    training_config = CriticalRelationTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        instance_loss_weight=args.instance_loss_weight,
        consistency_weight=args.consistency_weight,
        instance_warmup_epochs=args.instance_warmup_epochs,
        seed=args.seed,
    )
    training_device = torch.device("cuda:0")
    base, baseline_config = _load_baseline_model(args, device=training_device)
    initial_state = initial_critical_relation_state(
        descriptor_dim=descriptor_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    initial_adapter = CriticalRelationResidual(descriptor_dim, args.hidden_dim).to(
        training_device
    )
    initial_adapter.load_state_dict(initial_state, strict=True)
    initial_audit = {
        "train": audit_zero_initialization_records(
            train_records,
            base,
            initial_adapter,
            batch_size=args.batch_size,
            device=training_device,
        ),
        "validation": audit_zero_initialization_records(
            val_records,
            base,
            initial_adapter,
            batch_size=args.batch_size,
            device=training_device,
        ),
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    initial_audit_path = args.output_dir / "pretraining_identity_audit.json"
    initial_audit_path.write_text(
        json.dumps(initial_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del initial_adapter

    adapter, history = train_critical_relation_adapter(
        train_records,
        base,
        descriptor_dim=descriptor_dim,
        bag_temperature=baseline_config.bag_temperature,
        training_config=training_config,
        device=training_device,
        initial_state=initial_state,
    )
    adapter_state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    del base, adapter
    torch.cuda.empty_cache()

    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_path = args.output_dir / "critical_relation_residual.pt"
    torch.save(
        {
            "adapter_state_dict": adapter_state,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_config": asdict(baseline_config),
            "descriptor_dim": descriptor_dim,
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
    by_image = {str(item["image_id"]): item for item in unordered_scored}
    if len(unordered_scored) != 371 or len(by_image) != 371:
        raise RuntimeError("R3 T4x2 scoring does not cover 371 unique images")
    scored_val = [by_image[str(record["image_id"])] for record in val_records]
    prediction_manifest_sha256, score_manifest_sha256 = _write_validation_outputs(
        args, val_records, scored_val
    )

    count_spearman = _absolute_spearman(
        np.asarray([item["candidate_count"] for item in scored_val]),
        np.asarray([item["bag_probability"] for item in scored_val]),
    )
    base_agreement = float(
        np.mean([item["base_critical_agreement"] for item in scored_val])
    )
    frozen_base_agreement = float(
        initial_audit["validation"]["base_flip_critical_agreement"]
    )
    if base_agreement != frozen_base_agreement:
        raise RuntimeError("R3 base critical agreement changed after training")
    final_agreement = float(
        np.mean([item["final_selected_agreement"] for item in scored_val])
    )
    gate = {
        "absolute_candidate_count_probability_spearman": count_spearman,
        "count_probability_spearman_ceiling": (
            args.count_probability_spearman_ceiling
        ),
        "count_probability_gate_pass": (
            count_spearman <= args.count_probability_spearman_ceiling
        ),
        "base_flip_critical_agreement": frozen_base_agreement,
        "final_flip_selected_agreement": final_agreement,
        "minimum_allowed_final_agreement": (
            frozen_base_agreement + args.minimum_critical_agreement_delta
        ),
        "critical_agreement_gate_pass": (
            final_agreement
            >= frozen_base_agreement + args.minimum_critical_agreement_delta
        ),
    }
    gate["gt_blind_gate_pass"] = bool(
        gate["count_probability_gate_pass"]
        and gate["critical_agreement_gate_pass"]
    )

    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "selector_cache_manifest_sha256": cache_freeze[
            "selector_cache_manifest_sha256"
        ],
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pretraining_identity_audit_sha256": sha256_file(initial_audit_path),
        "training_history_sha256": sha256_file(history_path),
        "candidate_score_manifest_sha256": score_manifest_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "validation_predictions": 371,
        "gt_blind_gate": gate,
        "training_labels": "image_level_only",
        "epoch_selection": "fixed_final_epoch_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "run_id": "btxrd_mask_bag_critical_relation_r3_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
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
