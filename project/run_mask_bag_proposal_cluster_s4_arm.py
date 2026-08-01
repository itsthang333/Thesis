from __future__ import annotations

"""Run the isolated S4 group-excluded OOF proposal-cluster selector arm."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_crossfit import (
    assign_group_stratified_folds,
    crossfit_assignment_manifest,
)
from models.mask_bag_proposal_cluster_training import (
    ProposalClusterResidual,
    ProposalClusterTrainingConfig,
    attach_teacher_clusters,
    audit_cluster_residual_identity,
    audit_oof_teacher_coverage,
    default_teacher_model_config,
    fit_proposal_teacher_oof_fold,
    initial_cluster_residual_state,
    initial_teacher_state,
    score_cluster_residual,
    score_proposal_teacher,
    train_cluster_residual,
    train_proposal_teacher,
)
from run_mask_bag_normal_prototype_arm import (
    _load_baseline_model,
    _load_cache_records,
    _verify_cache_freeze,
    _write_validation_outputs,
)


FOLD_COUNT = 5
EXPECTED_FOLD_SUMMARY = [
    {"fold": 0, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 1, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
    {"fold": 2, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 3, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
    {"fold": 4, "images": 597, "groups": 198, "normal_images": 299, "tumor_images": 298},
]


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
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--teacher-instance-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--instance-warmup-epochs", type=int, default=2)
    parser.add_argument("--maximum-clusters", type=int, default=4)
    parser.add_argument("--minimum-iou", type=float, default=0.50)
    parser.add_argument("--minimum-containment", type=float, default=0.75)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.20)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--count-probability-spearman-ceiling",
        type=float,
        default=0.5013777759365411,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _frozen_training_config(args: argparse.Namespace) -> ProposalClusterTrainingConfig:
    expected = {
        "fold_count": 5,
        "epochs": 16,
        "batch_size": 16,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "teacher_instance_loss_weight": 0.25,
        "consistency_weight": 0.10,
        "instance_warmup_epochs": 2,
        "maximum_clusters": 4,
        "minimum_iou": 0.50,
        "minimum_containment": 0.75,
        "start_temperature": 1.0,
        "end_temperature": 0.20,
        "residual_hidden_dim": 128,
        "seed": 42,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected or args.count_probability_spearman_ceiling != 0.5013777759365411:
        raise ValueError("S4 execution differs from the frozen finite contract")
    return ProposalClusterTrainingConfig(**expected)


def _validate_cache(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("S4 cache cannot be empty")
    for record in records:
        count = len(record["candidate_indices"])
        if count < 1:
            raise ValueError("S4 cache contains an empty bag")
        if np.asarray(record["descriptors"]).shape != (count, 1156):
            raise ValueError("S4 requires the frozen 1,156-D descriptor")
        if np.asarray(record["flipped_descriptors"]).shape != (count, 1156):
            raise ValueError("S4 flipped descriptor alignment mismatch")
        for key in ("pairwise_iou", "pairwise_containment"):
            matrix = np.asarray(record[key], dtype=np.float32)
            if matrix.shape != (count, count) or not np.isfinite(matrix).all():
                raise ValueError(f"S4 {key} alignment/finite check failed")


def _fit_oof_jobs(
    records: list[dict[str, Any]],
    fold_ids: np.ndarray,
    folds: Sequence[int],
    training_config: ProposalClusterTrainingConfig,
    initial_states: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    model_config = default_teacher_model_config()
    output = [
        fit_proposal_teacher_oof_fold(
            records,
            fold_ids,
            heldout_fold=int(fold),
            model_config=model_config,
            training_config=training_config,
            device=device,
            initial_state=initial_states[int(fold)],
        )
        for fold in folds
    ]
    torch.cuda.empty_cache()
    return output


def _audit_full_teacher_group_exclusion(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the all-train teacher sees no validation group."""

    train_groups = sorted({str(record["group_id"]) for record in train_records})
    val_groups = sorted({str(record["group_id"]) for record in val_records})
    overlap = sorted(set(train_groups) & set(val_groups))
    if overlap:
        raise RuntimeError("S4 full teacher would see a validation group")
    return {
        "training_groups": train_groups,
        "validation_groups": val_groups,
        "training_group_count": len(train_groups),
        "validation_group_count": len(val_groups),
        "group_overlap": 0,
        "validation_segmentation_quality_used": False,
    }


def _write_score_payloads(
    root: Path,
    scores: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        image_id = str(score["image_id"])
        record = records_by_id[image_id]
        path = root / f"{index:04d}_{Path(image_id).stem}.npz"
        np.savez_compressed(
            path,
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            original_logits=np.asarray(score["original_logits"], dtype=np.float32),
            flipped_logits=np.asarray(score["flipped_logits"], dtype=np.float32),
            conservative_seed_logits=np.asarray(
                score["conservative_seed_logits"], dtype=np.float32
            ),
        )
        rows.append(
            {
                "image_id": image_id,
                "group_id": score["group_id"],
                "image_label": score["image_label"],
                "heldout_fold": score.get("heldout_fold", -1),
                "selected_view_agreement": int(score["selected_view_agreement"]),
                "candidate_count": score["candidate_count"],
                "bag_probability": score["bag_probability"],
                "payload_path": path.name,
                "payload_sha256": sha256_file(path),
            }
        )
    manifest = root.parent / "score_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(manifest)


def _write_oof_artifacts(
    output_dir: Path,
    fold_artifacts: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    training_config: ProposalClusterTrainingConfig,
) -> dict[str, Any]:
    root = output_dir / "oof_teachers"
    root.mkdir(parents=True, exist_ok=False)
    records_by_id = {str(record["image_id"]): record for record in records}
    hashes: dict[str, Any] = {}
    for artifact in sorted(fold_artifacts, key=lambda row: int(row["heldout_fold"])):
        fold = int(artifact["heldout_fold"])
        fold_root = root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        checkpoint = fold_root / "teacher.pt"
        torch.save(
            {
                "model_state_dict": artifact["teacher_state_dict"],
                "model_config": asdict(default_teacher_model_config()),
                "training_config": asdict(training_config),
                "heldout_fold": fold,
                "training_groups": artifact["training_groups"],
                "heldout_groups": artifact["heldout_groups"],
                "group_overlap": 0,
                "source_commit": args.source_commit,
                "protocol_sha256": args.protocol_sha256,
                "split_sha256": args.expected_split_sha256,
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            checkpoint,
        )
        history = fold_root / "training_history.json"
        history.write_text(
            json.dumps(artifact["training_history"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        score_manifest_sha = _write_score_payloads(
            fold_root / "scores", artifact["heldout_scores"], records_by_id
        )
        audit = fold_root / "fold_audit.json"
        audit.write_text(
            json.dumps(
                {
                    "heldout_fold": fold,
                    "derived_seed": artifact["derived_seed"],
                    "training_groups": artifact["training_groups"],
                    "heldout_groups": artifact["heldout_groups"],
                    "group_overlap": 0,
                    "heldout_records": len(artifact["heldout_scores"]),
                    "teacher_checkpoint_sha256": sha256_file(checkpoint),
                    "training_history_sha256": sha256_file(history),
                    "score_manifest_sha256": score_manifest_sha,
                    "validation_segmentation_quality_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        hashes[f"fold_{fold}"] = {
            "teacher_checkpoint_sha256": sha256_file(checkpoint),
            "training_history_sha256": sha256_file(history),
            "score_manifest_sha256": score_manifest_sha,
            "fold_audit_sha256": sha256_file(audit),
        }
    return hashes


def _write_cluster_evidence(
    output_dir: Path,
    split: str,
    records: Sequence[Mapping[str, Any]],
    heldout_fold_by_id: Mapping[str, int] | None,
) -> str:
    root = output_dir / "clusters" / split
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        path = root / f"{index:04d}_{Path(image_id).stem}.npz"
        np.savez_compressed(
            path,
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            teacher_original_logits=np.asarray(
                record["teacher_original_logits"], dtype=np.float32
            ),
            teacher_flipped_logits=np.asarray(
                record["teacher_flipped_logits"], dtype=np.float32
            ),
            teacher_conservative_seed_logits=np.asarray(
                record["teacher_conservative_seed_logits"], dtype=np.float32
            ),
            clusters=np.asarray(record["clusters"], dtype=np.uint8),
            cluster_valid=np.asarray(record["cluster_valid"], dtype=np.uint8),
            seed_indices=np.asarray(record["seed_indices"], dtype=np.int32),
        )
        rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "image_label": record["label"],
                "heldout_fold": -1 if heldout_fold_by_id is None else heldout_fold_by_id[image_id],
                "teacher_selected_view_agreement": int(
                    record["teacher_selected_view_agreement"]
                ),
                "cluster_count": int(np.asarray(record["cluster_valid"]).sum()),
                "cluster_member_count": int(np.asarray(record["clusters"]).any(axis=0).sum()),
                "payload_path": path.name,
                "payload_sha256": sha256_file(path),
            }
        )
    manifest = output_dir / "clusters" / f"{split}_cluster_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(manifest)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _absolute_spearman(first: np.ndarray, second: np.ndarray) -> float:
    ranks_first = _rankdata(np.asarray(first, dtype=np.float64))
    ranks_second = _rankdata(np.asarray(second, dtype=np.float64))
    if len(first) < 2 or np.std(ranks_first) == 0 or np.std(ranks_second) == 0:
        raise ValueError("S4 Spearman inputs are invalid")
    return abs(float(np.corrcoef(ranks_first, ranks_second)[0, 1]))


def _write_gt_blind_diagnostics(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    *,
    ceiling: float,
) -> tuple[str, float]:
    rows: list[dict[str, Any]] = []
    for record, prediction in zip(records, scored, strict=True):
        if str(record["image_id"]) != str(prediction["image_id"]):
            raise RuntimeError("S4 diagnostic order mismatch")
        rows.append(
            {
                "image_id": record["image_id"],
                "candidate_count": prediction["candidate_count"],
                "bag_probability": prediction["bag_probability"],
                "teacher_selected_view_agreement": int(
                    record["teacher_selected_view_agreement"]
                ),
                "final_selected_view_agreement": int(
                    prediction["final_selected_view_agreement"]
                ),
                "cluster_count": prediction["cluster_count"],
                "cluster_member_count": prediction["cluster_member_count"],
                "outside_cluster_count": prediction["outside_cluster_count"],
                "outside_cluster_original_residual_exact_zero": int(
                    prediction["outside_cluster_original_residual_exact_zero"]
                ),
                "outside_cluster_flipped_residual_exact_zero": int(
                    prediction["outside_cluster_flipped_residual_exact_zero"]
                ),
            }
        )
    spearman = _absolute_spearman(
        np.asarray([row["candidate_count"] for row in rows], dtype=np.float64),
        np.asarray([row["bag_probability"] for row in rows], dtype=np.float64),
    )
    if spearman > ceiling:
        raise RuntimeError("S4 count/probability GT-blind gate failed")
    path = output_dir / "gt_blind_diagnostics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path), spearman


def _score_validation_shard(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    residual_state: Mapping[str, torch.Tensor],
    training_config: ProposalClusterTrainingConfig,
    *,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, model_config = _load_baseline_model(args, device=device)
    residual = ProposalClusterResidual(
        model_config.descriptor_dim, training_config.residual_hidden_dim
    ).to(device)
    residual.load_state_dict(residual_state, strict=True)
    output = score_cluster_residual(
        records,
        base,
        residual,
        training_config,
        batch_size=training_config.batch_size,
        device=device,
    )
    del base, residual
    torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    training_config = _frozen_training_config(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S4 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S4 requires Tesla T4 x2, got {device_names}")
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
        raise RuntimeError("S4 frozen cohort mismatch")
    cache, validated_cache_rows = _load_cache_records(
        args, split_rows, cache_manifest_rows
    )
    train_records = cache["train"]
    val_records = cache["val"]
    _validate_cache(train_records + val_records)
    full_teacher_group_audit = _audit_full_teacher_group_exclusion(
        train_records, val_records
    )
    full_teacher_group_audit_path = (
        args.output_dir / "full_teacher_group_exclusion_audit.json"
    )
    full_teacher_group_audit_path.write_text(
        json.dumps(full_teacher_group_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    labels = np.asarray([record["label"] for record in train_records], dtype=np.int8)
    groups = np.asarray([record["group_id"] for record in train_records], dtype="U128")
    fold_ids = assign_group_stratified_folds(
        labels, groups, fold_count=training_config.fold_count, seed=training_config.seed
    )
    assignment = crossfit_assignment_manifest(
        [record["image_id"] for record in train_records],
        groups,
        labels,
        fold_ids,
    )
    if assignment["fold_summary"] != EXPECTED_FOLD_SUMMARY:
        raise RuntimeError("S4 cross-fit assignment differs from frozen cohort")
    assignment_path = args.output_dir / "crossfit_assignment.json"
    assignment_path.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    teacher_config = default_teacher_model_config()
    oof_initial_states = {
        fold: initial_teacher_state(
            teacher_config,
            seed=training_config.seed + 1000 + fold,
        )
        for fold in range(FOLD_COUNT)
    }
    folds_by_device = [list(range(FOLD_COUNT))[::2], list(range(FOLD_COUNT))[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _fit_oof_jobs,
                train_records,
                fold_ids,
                folds_by_device[device_index],
                training_config,
                oof_initial_states,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        fold_artifacts = [artifact for future in futures for artifact in future.result()]
    coverage = audit_oof_teacher_coverage(train_records, fold_ids, fold_artifacts)
    oof_hashes = _write_oof_artifacts(
        args.output_dir, fold_artifacts, train_records, args, training_config
    )
    coverage_path = args.output_dir / "oof_coverage_audit.json"
    coverage_path.write_text(
        json.dumps(
            {key: value for key, value in coverage.items() if key != "ordered_scores"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if (
        coverage["absolute_candidate_count_probability_spearman"]
        > args.count_probability_spearman_ceiling
    ):
        raise RuntimeError("S4 OOF teacher count/probability gate failed")
    enriched_train = attach_teacher_clusters(
        train_records, coverage["ordered_scores"], training_config
    )
    heldout_fold_by_id = {
        str(record["image_id"]): int(fold)
        for record, fold in zip(train_records, fold_ids, strict=True)
    }
    train_cluster_manifest_sha = _write_cluster_evidence(
        args.output_dir, "train", enriched_train, heldout_fold_by_id
    )

    training_device = torch.device("cuda:0")
    torch.cuda.set_device(training_device)
    full_teacher, full_teacher_history = train_proposal_teacher(
        train_records,
        teacher_config,
        training_config,
        device=training_device,
        seed_offset=100000,
    )
    full_teacher_scores = score_proposal_teacher(
        val_records,
        full_teacher,
        batch_size=training_config.batch_size,
        device=training_device,
    )
    enriched_val = attach_teacher_clusters(
        val_records, full_teacher_scores, training_config
    )
    full_teacher_checkpoint = args.output_dir / "full_train_teacher.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in full_teacher.state_dict().items()
            },
            "model_config": asdict(teacher_config),
            "training_config": asdict(training_config),
            "training_groups": full_teacher_group_audit["training_groups"],
            "validation_groups": full_teacher_group_audit["validation_groups"],
            "group_overlap": 0,
            "full_teacher_group_exclusion_audit_sha256": sha256_file(
                full_teacher_group_audit_path
            ),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        full_teacher_checkpoint,
    )
    full_teacher_history_path = args.output_dir / "full_train_teacher_history.json"
    full_teacher_history_path.write_text(
        json.dumps(full_teacher_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    val_teacher_score_manifest_sha = _write_score_payloads(
        args.output_dir / "full_teacher_validation_scores" / "scores",
        full_teacher_scores,
        {str(record["image_id"]): record for record in val_records},
    )
    val_cluster_manifest_sha = _write_cluster_evidence(
        args.output_dir, "val", enriched_val, None
    )
    del full_teacher
    torch.cuda.empty_cache()

    base, baseline_config = _load_baseline_model(args, device=training_device)
    initial_state = initial_cluster_residual_state(
        baseline_config.descriptor_dim, training_config
    )
    initial_residual = ProposalClusterResidual(
        baseline_config.descriptor_dim, training_config.residual_hidden_dim
    ).to(training_device)
    initial_residual.load_state_dict(initial_state, strict=True)
    identity_audit = {
        "train": audit_cluster_residual_identity(
            enriched_train,
            base,
            initial_residual,
            batch_size=training_config.batch_size,
            device=training_device,
        ),
        "validation": audit_cluster_residual_identity(
            enriched_val,
            base,
            initial_residual,
            batch_size=training_config.batch_size,
            device=training_device,
        ),
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    identity_path = args.output_dir / "pretraining_identity_audit.json"
    identity_path.write_text(
        json.dumps(identity_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del initial_residual
    residual, residual_history = train_cluster_residual(
        enriched_train,
        base,
        baseline_config,
        training_config,
        device=training_device,
        initial_state=initial_state,
    )
    residual_state = {
        key: value.detach().cpu().clone() for key, value in residual.state_dict().items()
    }
    residual_checkpoint = args.output_dir / "proposal_cluster_residual.pt"
    torch.save(
        {
            "residual_state_dict": residual_state,
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_config": asdict(baseline_config),
            "training_config": asdict(training_config),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        residual_checkpoint,
    )
    residual_history_path = args.output_dir / "residual_training_history.json"
    residual_history_path.write_text(
        json.dumps(residual_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del base, residual
    torch.cuda.empty_cache()

    val_shards = [enriched_val[:186], enriched_val[186:]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_validation_shard,
                args,
                val_shards[device_index],
                residual_state,
                training_config,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        scored_val = [prediction for future in futures for prediction in future.result()]
    diagnostics_sha, count_probability_spearman = _write_gt_blind_diagnostics(
        args.output_dir,
        enriched_val,
        scored_val,
        ceiling=args.count_probability_spearman_ceiling,
    )
    prediction_manifest_sha, score_manifest_sha = _write_validation_outputs(
        args, enriched_val, scored_val
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "selector_cache_manifest_sha256": cache_freeze["selector_cache_manifest_sha256"],
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "crossfit_assignment_sha256": sha256_file(assignment_path),
        "oof_coverage_audit_sha256": sha256_file(coverage_path),
        "oof_teacher_hashes": oof_hashes,
        "train_cluster_manifest_sha256": train_cluster_manifest_sha,
        "full_teacher_group_exclusion_audit_sha256": sha256_file(
            full_teacher_group_audit_path
        ),
        "full_teacher_checkpoint_sha256": sha256_file(full_teacher_checkpoint),
        "full_teacher_history_sha256": sha256_file(full_teacher_history_path),
        "validation_teacher_score_manifest_sha256": val_teacher_score_manifest_sha,
        "validation_cluster_manifest_sha256": val_cluster_manifest_sha,
        "pretraining_identity_audit_sha256": sha256_file(identity_path),
        "residual_checkpoint_sha256": sha256_file(residual_checkpoint),
        "residual_training_history_sha256": sha256_file(residual_history_path),
        "gt_blind_diagnostics_sha256": diagnostics_sha,
        "absolute_candidate_count_probability_spearman": count_probability_spearman,
        "candidate_score_manifest_sha256": score_manifest_sha,
        "prediction_manifest_sha256": prediction_manifest_sha,
        "validation_predictions": 371,
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "run_id": "btxrd_mask_bag_proposal_cluster_s4_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "crossfit": assignment,
        "teacher_model_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "oof_parallel_workers": 2,
            "oof_folds_by_device": folds_by_device,
            "validation_shards": [186, 185],
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
    run_manifest_path = args.output_dir / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "S4_PREDICTIONS_FROZEN_GT_BLIND",
                "prediction_freeze_sha256": sha256_file(freeze_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "absolute_candidate_count_probability_spearman": count_probability_spearman,
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
