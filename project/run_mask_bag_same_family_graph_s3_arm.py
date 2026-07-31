from __future__ import annotations

"""Run the fixed S3 same-family proposal-graph smoothing arm."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_same_family_graph import (
    SameFamilyGraphConfig,
    score_same_family_graph_records,
)
from models.mask_bag_selector_cache import unpack_candidate_masks
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--expected-selector-cache-freeze-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.25)
    parser.add_argument("--minimum-containment", type=float, default=0.50)
    parser.add_argument("--graph-alpha", type=float, default=0.50)
    parser.add_argument("--graph-iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--count-probability-spearman-ceiling",
        type=float,
        default=0.5013777759365411,
    )
    return parser.parse_args()


def _npy_sha256(values: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, values.astype(np.float16, copy=False), allow_pickle=False)
    return sha256(stream.getvalue()).hexdigest()


def _verify_baseline_freeze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    freeze_path = args.baseline_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("accepted baseline prediction-freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = args.baseline_root / "predictions" / "prediction_manifest.csv"
    if (
        sha256_file(manifest_path) != freeze.get("prediction_manifest_sha256")
        or freeze.get("source_commit") != args.expected_baseline_source_commit
        or freeze.get("protocol_sha256") != args.expected_baseline_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("accepted baseline prediction-freeze provenance mismatch")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 371 or len({row["image_id"] for row in rows}) != 371:
        raise ValueError("accepted baseline prediction cohort mismatch")
    return freeze, rows


def _base_vector_sha256(logits: np.ndarray) -> str:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("base candidate logits must be one finite vector")
    return sha256(values.tobytes(order="C")).hexdigest()


def _write_pregraph_identity_audit(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
) -> tuple[Path, dict[str, Any]]:
    indexed = {row["image_id"]: row for row in baseline_rows}
    rows: list[dict[str, object]] = []
    for record, prediction in zip(records, scored, strict=True):
        image_id = str(record["image_id"])
        baseline = indexed.get(image_id)
        if baseline is None or prediction["image_id"] != image_id:
            raise RuntimeError("S3/base identity order mismatch")
        candidate_indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        base_logits = np.asarray(prediction["base_candidate_logits"], dtype=np.float32)
        if base_logits.shape != candidate_indices.shape:
            raise RuntimeError("S3/base candidate-score alignment mismatch")
        winner = int(np.argmax(base_logits))
        original_winner = int(candidate_indices[winner])
        packed = record["packed_masks"]
        masks = unpack_candidate_masks(packed).astype(np.float32)
        probability = float(prediction["base_bag_probability"])
        map_sha256 = _npy_sha256(masks[winner] * probability)
        accepted = {
            "selected_candidate_index": int(baseline["selected_candidate_index"]),
            "selected_candidate_logit": float(baseline["selected_candidate_logit"]),
            "bag_logit": float(baseline["bag_logit"]),
            "bag_probability": float(baseline["bag_probability"]),
            "map_sha256": baseline["map_sha256"],
        }
        observed = {
            "selected_candidate_index": original_winner,
            "selected_candidate_logit": float(base_logits[winner]),
            "bag_logit": float(prediction["base_bag_logit"]),
            "bag_probability": probability,
            "map_sha256": map_sha256,
        }
        exact = observed == accepted
        rows.append(
            {
                "image_id": image_id,
                "candidate_count": len(candidate_indices),
                "base_candidate_logits_sha256": _base_vector_sha256(base_logits),
                "alpha_zero_identity_exact": int(
                    prediction["alpha_zero_identity_exact"]
                ),
                "accepted_selected_index_exact": int(
                    observed["selected_candidate_index"]
                    == accepted["selected_candidate_index"]
                ),
                "accepted_selected_logit_exact": int(
                    observed["selected_candidate_logit"]
                    == accepted["selected_candidate_logit"]
                ),
                "accepted_bag_logit_exact": int(
                    observed["bag_logit"] == accepted["bag_logit"]
                ),
                "accepted_bag_probability_exact": int(
                    observed["bag_probability"] == accepted["bag_probability"]
                ),
                "accepted_map_sha256_exact": int(
                    observed["map_sha256"] == accepted["map_sha256"]
                ),
                "accepted_row_exact": int(exact),
            }
        )
    path = args.output_dir / "pregraph_identity_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "records": len(rows),
        "candidate_vectors": len(rows),
        "alpha_zero_identity_exact_records": sum(
            int(row["alpha_zero_identity_exact"]) for row in rows
        ),
        "accepted_row_exact_records": sum(int(row["accepted_row_exact"]) for row in rows),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    if (
        summary["records"] != 371
        or summary["alpha_zero_identity_exact_records"] != 371
        or summary["accepted_row_exact_records"] != 371
    ):
        raise RuntimeError("S3 alpha-zero or accepted-baseline identity failed")
    return path, summary


def _write_gt_blind_diagnostics(
    output_dir: Path,
    scored: list[dict[str, Any]],
) -> Path:
    path = output_dir / "gt_blind_diagnostics.csv"
    rows = [
        {
            "image_id": item["image_id"],
            "candidate_count": item["candidate_count"],
            "bag_probability": item["bag_probability"],
            "view_swap_exact": int(item["view_swap_exact"]),
            "alpha_zero_identity_exact": int(item["alpha_zero_identity_exact"]),
            "graph_symmetric": int(item["graph_symmetric"]),
            "cross_family_edge_count": item["cross_family_edge_count"],
            "non_self_edge_count": item["non_self_edge_count"],
            "isolated_candidate_count": item["isolated_candidate_count"],
            "isolated_logits_exact": int(item["isolated_logits_exact"]),
        }
        for item in scored
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _score_device_shard(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    graph_config: SameFamilyGraphConfig,
    device_index: int,
) -> list[dict[str, Any]]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    base, baseline_config = _load_baseline_model(args, device=device)
    scored = score_same_family_graph_records(
        records,
        base,
        bag_temperature=baseline_config.bag_temperature,
        graph_config=graph_config,
        batch_size=args.batch_size,
        device=device,
    )
    del base
    torch.cuda.empty_cache()
    return scored


def main() -> None:
    args = parse_args()
    if (
        args.minimum_iou != 0.25
        or args.minimum_containment != 0.50
        or args.graph_alpha != 0.50
        or args.graph_iterations != 10
        or args.batch_size != 16
        or args.count_probability_spearman_ceiling != 0.5013777759365411
    ):
        raise ValueError("S3 execution differs from the frozen finite contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S3 requires exactly two visible CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S3 requires Tesla T4 x2, got {device_names}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    cache_freeze, cache_manifest_rows = _verify_cache_freeze(args)
    baseline_freeze, baseline_rows = _verify_baseline_freeze(args)
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
    _validate_descriptor_cache(cache["train"] + cache["val"])
    val_records = cache["val"]
    graph_config = SameFamilyGraphConfig(
        minimum_iou=args.minimum_iou,
        minimum_containment=args.minimum_containment,
        alpha=args.graph_alpha,
        iterations=args.graph_iterations,
    )

    shards = [val_records[::2], val_records[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _score_device_shard,
                args,
                shards[device_index],
                graph_config=graph_config,
                device_index=device_index,
            )
            for device_index in range(2)
        ]
        unordered_scored = [item for future in futures for item in future.result()]
    by_image = {str(item["image_id"]): item for item in unordered_scored}
    if len(unordered_scored) != 371 or len(by_image) != 371:
        raise RuntimeError("S3 T4x2 scoring does not cover 371 unique images")
    scored_val = [by_image[str(record["image_id"])] for record in val_records]

    identity_path, identity_summary = _write_pregraph_identity_audit(
        args,
        val_records,
        scored_val,
        baseline_rows,
    )
    prediction_manifest_sha256, score_manifest_sha256 = _write_validation_outputs(
        args,
        val_records,
        scored_val,
    )
    diagnostics_path = _write_gt_blind_diagnostics(args.output_dir, scored_val)
    count_spearman = _absolute_spearman(
        np.asarray([item["candidate_count"] for item in scored_val]),
        np.asarray([item["bag_probability"] for item in scored_val]),
    )
    graph_gate = {
        "absolute_candidate_count_probability_spearman": count_spearman,
        "count_probability_spearman_ceiling": args.count_probability_spearman_ceiling,
        "count_probability_gate_pass": count_spearman
        <= args.count_probability_spearman_ceiling,
        "view_swap_exact_records": sum(
            bool(item["view_swap_exact"]) for item in scored_val
        ),
        "alpha_zero_identity_exact_records": sum(
            bool(item["alpha_zero_identity_exact"]) for item in scored_val
        ),
        "graph_symmetric_records": sum(
            bool(item["graph_symmetric"]) for item in scored_val
        ),
        "cross_family_edge_count": sum(
            int(item["cross_family_edge_count"]) for item in scored_val
        ),
        "non_self_edge_count": sum(
            int(item["non_self_edge_count"]) for item in scored_val
        ),
        "isolated_candidate_count": sum(
            int(item["isolated_candidate_count"]) for item in scored_val
        ),
        "isolated_logits_exact_records": sum(
            bool(item["isolated_logits_exact"]) for item in scored_val
        ),
        "accepted_baseline_identity_exact_records": identity_summary[
            "accepted_row_exact_records"
        ],
    }
    graph_gate["gt_blind_gate_pass"] = bool(
        graph_gate["count_probability_gate_pass"]
        and graph_gate["view_swap_exact_records"] == 371
        and graph_gate["alpha_zero_identity_exact_records"] == 371
        and graph_gate["graph_symmetric_records"] == 371
        and graph_gate["cross_family_edge_count"] == 0
        and graph_gate["non_self_edge_count"] > 0
        and graph_gate["isolated_logits_exact_records"] == 371
        and graph_gate["accepted_baseline_identity_exact_records"] == 371
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
        "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
        "baseline_prediction_manifest_sha256": baseline_freeze[
            "prediction_manifest_sha256"
        ],
        "graph_config": asdict(graph_config),
        "pregraph_identity_audit_sha256": sha256_file(identity_path),
        "gt_blind_diagnostics_sha256": sha256_file(diagnostics_path),
        "candidate_score_manifest_sha256": score_manifest_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "validation_predictions": 371,
        "gt_blind_gate": graph_gate,
        "arm_fit": "none_fixed_operator",
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
        "run_id": "btxrd_mask_bag_same_family_graph_s3_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "cache_freeze_sha256": args.expected_selector_cache_freeze_sha256,
        "graph_config": asdict(graph_config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
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
