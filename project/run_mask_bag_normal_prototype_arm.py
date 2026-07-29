from __future__ import annotations

"""Run the R1 normal-prototype selector arm from a verified post-v3 cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    save_float_map,
    sha256_file,
)
from models.mask_bag_crossfit import (
    assign_group_stratified_folds,
    crossfit_assignment_manifest,
)
from models.mask_bag_descriptor_residual import AuxiliaryDescriptorResidual
from models.mask_bag_normal_crossfit import (
    assemble_normal_oof_candidate,
    fit_normal_oof_fold,
)
from models.mask_bag_normal_residual_training import (
    NormalResidualTrainingConfig,
    attach_normal_prototype_features,
    fit_normal_prototype_bank,
    score_normal_residual_records,
    train_normal_residual_adapter,
)
from models.mask_bag_oof_selection import select_prototype_count_one_standard_error
from models.mask_bag_residual_objective import ResidualObjectiveConfig
from models.mask_bag_score_evidence import (
    save_candidate_score_evidence,
    write_candidate_score_manifest,
)
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import (
    load_selector_cache_record,
    validate_selector_cache_manifest,
)
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


PROTOTYPE_COUNTS = (8, 16, 32)
FOLD_COUNT = 5


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
    parser.add_argument(
        "--baseline-absolute-count-probability-spearman",
        type=float,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--prototype-temperature", type=float, default=0.10)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--residual-drift-weight", type=float, default=1.0e-3)
    parser.add_argument("--count-association-tolerance", type=float, default=0.02)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _verify_cache_freeze(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    freeze_path = args.selector_cache_root / "selector_cache_freeze.json"
    if sha256_file(freeze_path) != args.expected_selector_cache_freeze_sha256:
        raise ValueError("selector cache freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("baseline_checkpoint_sha256")
        != args.expected_baseline_checkpoint_sha256
        or freeze.get("baseline_source_commit")
        != args.expected_baseline_source_commit
        or freeze.get("baseline_protocol_sha256")
        != args.expected_baseline_protocol_sha256
        or freeze.get("cohort") != {"train": 2981, "validation": 371}
        or freeze.get("validation_selected_indices_reproduced") != 371
        or freeze.get("validation_map_hashes_reproduced") != 371
        or freeze.get("train_masks_discarded") is not True
        or freeze.get("validation_masks_bitpacked") is not True
        or freeze.get("affinity_features_cached") is not True
        or freeze.get("affinity_feature_dim") != 24
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("selector cache freeze provenance/safety mismatch")
    manifest_path = args.selector_cache_root / "selector_cache_manifest.csv"
    if sha256_file(manifest_path) != freeze["selector_cache_manifest_sha256"]:
        raise ValueError("selector cache manifest differs from freeze")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return freeze, list(csv.DictReader(handle))


def _load_baseline_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[RadDinoMaskBagMIL, MaskBagMILConfig]:
    checkpoint_path = args.baseline_root / "rad_dino_mask_bag_mil.pt"
    if sha256_file(checkpoint_path) != args.expected_baseline_checkpoint_sha256:
        raise ValueError("baseline checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != args.expected_baseline_source_commit
        or checkpoint.get("protocol_sha256")
        != args.expected_baseline_protocol_sha256
        or checkpoint.get("split_sha256") != args.expected_split_sha256
        or checkpoint.get("validation_gt_read") is not False
        or checkpoint.get("consumer_trained") is not False
        or checkpoint.get("test_evaluated") is not False
    ):
        raise ValueError("baseline checkpoint provenance/safety mismatch")
    config = MaskBagMILConfig(**checkpoint["config"])
    model = RadDinoMaskBagMIL(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).to(device).eval()
    return model, config


def _load_cache_records(
    args: argparse.Namespace,
    split_rows: Mapping[str, list[dict[str, str]]],
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, str]]]]:
    indexed = {(row["split"], row["image_id"]): row for row in manifest_rows}
    expected: dict[str, dict[str, dict[str, object]]] = {"train": {}, "val": {}}
    for split in ("train", "val"):
        for row in split_rows[split]:
            manifest = indexed.get((split, row["image_id"]))
            if manifest is None:
                raise ValueError(f"selector cache omits {split}/{row['image_id']}")
            expected[split][row["image_id"]] = {
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "candidate_payload_sha256": manifest[
                    "candidate_payload_sha256"
                ],
            }
    validated = validate_selector_cache_manifest(
        args.selector_cache_root,
        expected_manifest_sha256=sha256_file(
            args.selector_cache_root / "selector_cache_manifest.csv"
        ),
        expected_images=expected,
    )
    loaded: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in ("train", "val"):
        validated_by_id = {row["image_id"]: row for row in validated[split]}
        for split_row in split_rows[split]:
            row = validated_by_id[split_row["image_id"]]
            payload = load_selector_cache_record(
                args.selector_cache_root / row["cache_path"],
                expected_sha256=row["cache_sha256"],
                require_packed_masks=split == "val",
            )
            loaded[split].append(
                {
                    "image_id": split_row["image_id"],
                    "group_id": split_row["group_id"],
                    "label": int(split_row["tumor"]),
                    "candidate_payload_sha256": row[
                        "candidate_payload_sha256"
                    ],
                    **payload,
                }
            )
    return loaded, validated


def _initial_adapter_state(
    *,
    descriptor_dim: int,
    hidden_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    adapter = AuxiliaryDescriptorResidual(
        base_descriptor_dim=descriptor_dim,
        auxiliary_dim=4,
        hidden_dim=hidden_dim,
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }


def _run_oof_device_jobs(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    fold_ids: np.ndarray,
    jobs: list[tuple[int, int, dict[str, torch.Tensor]]],
    objective_config: ResidualObjectiveConfig,
    training_config: NormalResidualTrainingConfig,
    *,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, config = _load_baseline_model(args, device=device)
    results: list[dict[str, Any]] = []
    for prototype_count, fold, initial_state in jobs:
        results.append(
            fit_normal_oof_fold(
                records,
                fold_ids,
                heldout_fold=fold,
                prototype_count=prototype_count,
                frozen_base_scorer=base,
                descriptor_dim=config.descriptor_dim,
                objective_config=objective_config,
                training_config=training_config,
                device=device,
                initial_adapter_state=initial_state,
            )
        )
    del base
    torch.cuda.empty_cache()
    return results


def _write_oof_artifacts(
    root: Path,
    artifacts: list[dict[str, Any]],
    assembled: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for artifact in sorted(
        artifacts,
        key=lambda row: (row["prototype_count"], row["heldout_fold"]),
    ):
        k = int(artifact["prototype_count"])
        fold = int(artifact["heldout_fold"])
        fold_root = root / f"k_{k}" / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        prototype_path = fold_root / "normal_prototypes.npz"
        np.savez_compressed(
            prototype_path,
            prototypes=artifact["prototype_bank"],
        )
        adapter_path = fold_root / "adapter.pt"
        torch.save(
            {
                "adapter_state_dict": artifact["adapter_state_dict"],
                "training_config": artifact["adapter_training_config"],
                "objective_config": artifact["objective_config"],
                "prototype_count": k,
                "heldout_fold": fold,
                "derived_seed": artifact["derived_seed"],
                "validation_gt_read": False,
                "consumer_trained": False,
                "test_evaluated": False,
            },
            adapter_path,
        )
        metadata = {
            key: artifact[key]
            for key in (
                "prototype_count",
                "heldout_fold",
                "derived_seed",
                "training_groups",
                "heldout_groups",
                "group_overlap",
                "prototype_audit",
                "adapter_training_config",
                "objective_config",
                "training_history",
                "heldout_mean_image_bce",
                "validation_segmentation_quality_used",
            )
        }
        metadata_path = fold_root / "fold_audit.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        predictions_path = fold_root / "heldout_predictions.csv"
        predictions = artifact["heldout_predictions"]
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        hashes[f"k_{k}_fold_{fold}"] = {
            "prototype_sha256": sha256_file(prototype_path),
            "adapter_sha256": sha256_file(adapter_path),
            "audit_sha256": sha256_file(metadata_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
    for k, result in assembled.items():
        k_root = root / f"k_{k}"
        predictions_path = k_root / "oof_predictions.csv"
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(result["oof_predictions"][0])
            )
            writer.writeheader()
            writer.writerows(result["oof_predictions"])
        summary = {
            key: result[key]
            for key in (
                "prototype_count",
                "fold_image_bce",
                "mean_oof_image_bce",
                "count_probability_spearman",
                "crossfit_exclusion",
                "validation_segmentation_quality_used",
            )
        }
        summary_path = k_root / "oof_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes[f"k_{k}_aggregate"] = {
            "oof_predictions_sha256": sha256_file(predictions_path),
            "oof_summary_sha256": sha256_file(summary_path),
        }
    return hashes


def _write_validation_outputs(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> tuple[str, str]:
    prediction_root = args.output_dir / "predictions"
    map_root = prediction_root / "maps"
    score_root = args.output_dir / "candidate_scores"
    score_payload_root = score_root / "scores"
    map_root.mkdir(parents=True, exist_ok=False)
    score_payload_root.mkdir(parents=True, exist_ok=False)
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index, (record, prediction) in enumerate(zip(records, scored, strict=True)):
        if record["image_id"] != prediction["image_id"]:
            raise RuntimeError("validation scoring order differs from cache")
        candidate_indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        logits = np.asarray(prediction["candidate_logits"], dtype=np.float32)
        if logits.shape != candidate_indices.shape:
            raise RuntimeError("validation score count differs from cache")
        stem = f"{index:04d}_{Path(str(record['image_id'])).stem}"
        score_relative = Path("scores") / f"{stem}.npz"
        saved_score = save_candidate_score_evidence(
            score_root / score_relative,
            candidate_indices=candidate_indices,
            candidate_logits=logits,
        )
        packed = record["packed_masks"]
        masks = unpack_candidate_masks(packed).astype(np.float32)
        local_winner = int(np.argmax(logits))
        original_winner = int(candidate_indices[local_winner])
        bag_probability = float(prediction["bag_probability"])
        map_path = map_root / f"{stem}.npy"
        save_float_map(map_path, masks[local_winner] * bag_probability)
        score_rows.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                **saved_score,
                "score_path": str(score_relative),
            }
        )
        prediction_rows.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                "candidate_count": len(candidate_indices),
                "selected_candidate_index": original_winner,
                "selected_candidate_logit": saved_score[
                    "selected_candidate_logit"
                ],
                "candidate_logit_tta": "mean_original_aligned_horizontal_flip",
                "bag_logit": prediction["bag_logit"],
                "bag_probability": bag_probability,
                "selected_area_ratio": float(masks[local_winner].mean()),
                "fallback_count": int(
                    np.asarray(record["fallback_flags"]).sum()
                ),
                "map_path": str(Path("maps") / map_path.name),
                "map_sha256": sha256_file(map_path),
            }
        )
    prediction_manifest = prediction_root / "prediction_manifest.csv"
    with prediction_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    score_manifest = write_candidate_score_manifest(score_root, score_rows)
    return sha256_file(prediction_manifest), str(score_manifest["manifest_sha256"])


def main() -> None:
    args = parse_args()
    if (
        args.epochs != 16
        or args.batch_size != 16
        or args.learning_rate != 3.0e-4
        or args.weight_decay != 1.0e-4
        or args.prototype_temperature != 0.10
        or args.adapter_hidden_dim != 128
        or args.consistency_weight != 0.10
        or args.residual_drift_weight != 1.0e-3
        or args.count_association_tolerance != 0.02
        or args.fold_count != FOLD_COUNT
        or not 0.0
        <= args.baseline_absolute_count_probability_spearman
        <= 1.0
    ):
        raise ValueError("R1 execution differs from the frozen finite contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("R1 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"R1 requires Tesla T4 x2, got {device_names}")
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
    descriptor_dim = int(np.asarray(train_records[0]["descriptors"]).shape[1])

    labels = np.asarray([record["label"] for record in train_records], dtype=np.int8)
    groups = np.asarray(
        [record["group_id"] for record in train_records], dtype="U128"
    )
    fold_ids = assign_group_stratified_folds(
        labels, groups, fold_count=FOLD_COUNT, seed=args.seed
    )
    assignment = crossfit_assignment_manifest(
        [record["image_id"] for record in train_records],
        groups,
        labels,
        fold_ids,
    )
    expected_fold_summary = [
        {"fold": 0, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
        {"fold": 1, "images": 596, "groups": 196, "normal_images": 298, "tumor_images": 298},
        {"fold": 2, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
        {"fold": 3, "images": 596, "groups": 197, "normal_images": 299, "tumor_images": 297},
        {"fold": 4, "images": 597, "groups": 198, "normal_images": 299, "tumor_images": 298},
    ]
    if assignment["fold_summary"] != expected_fold_summary:
        raise RuntimeError("R1 cross-fit assignment differs from frozen cohort")
    assignment_path = args.output_dir / "crossfit_assignment.json"
    assignment_path.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    objective_config = ResidualObjectiveConfig(
        bag_temperature=0.20,
        consistency_weight=args.consistency_weight,
        residual_drift_weight=args.residual_drift_weight,
    )
    training_config = NormalResidualTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        prototype_temperature=args.prototype_temperature,
        adapter_hidden_dim=args.adapter_hidden_dim,
        seed=args.seed,
    )
    jobs: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    for prototype_count in PROTOTYPE_COUNTS:
        for fold in range(FOLD_COUNT):
            derived_seed = args.seed + 1000 * prototype_count + fold
            jobs.append(
                (
                    prototype_count,
                    fold,
                    _initial_adapter_state(
                        descriptor_dim=descriptor_dim,
                        hidden_dim=args.adapter_hidden_dim,
                        seed=derived_seed,
                    ),
                )
            )
    jobs_by_device = [jobs[::2], jobs[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_oof_device_jobs,
                args,
                train_records,
                fold_ids,
                jobs_by_device[device_index],
                objective_config,
                training_config,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        fold_artifacts = [
            artifact for future in futures for artifact in future.result()
        ]
    assembled = {
        prototype_count: assemble_normal_oof_candidate(
            train_records,
            fold_ids,
            [
                artifact
                for artifact in fold_artifacts
                if int(artifact["prototype_count"]) == prototype_count
            ],
            prototype_count=prototype_count,
        )
        for prototype_count in PROTOTYPE_COUNTS
    }
    oof_hashes = _write_oof_artifacts(
        args.output_dir / "oof", fold_artifacts, assembled
    )
    selection = select_prototype_count_one_standard_error(
        [
            {
                "prototype_count": prototype_count,
                "fold_image_bce": assembled[prototype_count]["fold_image_bce"],
                "count_probability_spearman": assembled[prototype_count][
                    "count_probability_spearman"
                ],
            }
            for prototype_count in PROTOTYPE_COUNTS
        ],
        baseline_absolute_count_association=
        args.baseline_absolute_count_probability_spearman,
        maximum_absolute_count_association_increase=
        args.count_association_tolerance,
    )
    selection_path = args.output_dir / "prototype_count_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_k = int(selection["selected_prototype_count"])

    final_seed = args.seed + 100000 + selected_k
    final_prototypes, final_prototype_audit = fit_normal_prototype_bank(
        train_records,
        prototype_count=selected_k,
        seed=final_seed,
    )
    enriched_train = attach_normal_prototype_features(
        train_records,
        final_prototypes,
        temperature=args.prototype_temperature,
    )
    enriched_val = attach_normal_prototype_features(
        val_records,
        final_prototypes,
        temperature=args.prototype_temperature,
    )
    final_device = torch.device("cuda:0")
    final_base, baseline_config = _load_baseline_model(
        args, device=final_device
    )
    final_training_config = replace(training_config, seed=final_seed)
    final_adapter, final_history = train_normal_residual_adapter(
        enriched_train,
        final_base,
        descriptor_dim=descriptor_dim,
        objective_config=objective_config,
        training_config=final_training_config,
        device=final_device,
        initial_adapter_state=_initial_adapter_state(
            descriptor_dim=descriptor_dim,
            hidden_dim=args.adapter_hidden_dim,
            seed=final_seed,
        ),
    )
    scored_val = score_normal_residual_records(
        enriched_val,
        final_base,
        final_adapter,
        bag_temperature=objective_config.bag_temperature,
        batch_size=args.batch_size,
        device=final_device,
    )
    prototype_path = args.output_dir / "normal_prototypes.npz"
    np.savez_compressed(prototype_path, prototypes=final_prototypes)
    checkpoint_path = args.output_dir / "normal_prototype_residual.pt"
    torch.save(
        {
            "adapter_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in final_adapter.state_dict().items()
            },
            "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
            "baseline_config": asdict(baseline_config),
            "selected_prototype_count": selected_k,
            "prototype_sha256": sha256_file(prototype_path),
            "prototype_audit": final_prototype_audit,
            "objective_config": asdict(objective_config),
            "training_config": asdict(final_training_config),
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "selector_cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    history_path = args.output_dir / "final_training_history.json"
    history_path.write_text(
        json.dumps(final_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_manifest_sha256, score_manifest_sha256 = (
        _write_validation_outputs(args, val_records, scored_val)
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
        "crossfit_assignment_sha256": sha256_file(assignment_path),
        "prototype_count_selection_sha256": sha256_file(selection_path),
        "oof_artifact_hashes": oof_hashes,
        "selected_prototype_count": selected_k,
        "final_prototype_sha256": sha256_file(prototype_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "final_training_history_sha256": sha256_file(history_path),
        "candidate_score_manifest_sha256": score_manifest_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "validation_predictions": 371,
        "training_labels": "image_level_only",
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
        "run_id": "btxrd_mask_bag_normal_prototype_r1_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "crossfit": assignment,
        "prototype_counts": list(PROTOTYPE_COUNTS),
        "selection": selection,
        "objective_config": asdict(objective_config),
        "training_config": asdict(training_config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "oof_parallel_workers": 2,
            "oof_jobs_by_device": [len(jobs_by_device[0]), len(jobs_by_device[1])],
            "final_fit_device": device_names[0],
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
