from __future__ import annotations

"""Independent physical/result audit for the MAE normality probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_COUNTS = {
    "validation": 371,
    "tumor": 184,
    "small": 94,
    "medium": 72,
    "large": 18,
}
METRICS = (
    "pixel_ap",
    "pixel_auroc",
    "argmax_hit",
    "saliency_mass_in_gt",
    "dice_p90",
    "dice_p95",
    "dice_p97",
    "dice_p99",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-base-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def _audit_wrapper(path: Path, expected_hash: str) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError("Wrapper SHA-256 mismatch")
    text = path.read_text(encoding="utf-8")
    markers = {
        "base_generation": 'prediction = OUTPUT / f"{role}_prediction"',
        "prediction_freeze": 'freeze_path = OUTPUT / "prediction_freeze.json"',
        "evaluation": 'evaluation = OUTPUT / f"{role}_evaluation"',
        "comparison": 'comparison = OUTPUT / "paired_comparison.json"',
    }
    positions = {name: text.index(marker) for name, marker in markers.items()}
    if not (
        positions["base_generation"]
        < positions["prediction_freeze"]
        < positions["evaluation"]
        < positions["comparison"]
    ):
        raise ValueError("Wrapper prediction-freeze/evaluation ordering drift")
    return {"sha256": actual_hash, "ordering_positions": positions}


def _audit_prediction_arm(
    *,
    run_root: Path,
    role: str,
    frozen: dict[str, Any],
    expected_base_weight: str,
) -> dict[str, Any]:
    prediction = run_root / f"{role}_prediction"
    manifest_path = prediction / "prediction_manifest.csv"
    metadata_path = prediction / "generation_metadata.json"
    if sha256_file(manifest_path) != frozen["prediction_manifest_sha256"]:
        raise ValueError(f"{role}: prediction manifest differs from freeze")
    if sha256_file(metadata_path) != frozen["generation_metadata_sha256"]:
        raise ValueError(f"{role}: generation metadata differs from freeze")
    rows = _read_csv(manifest_path)
    if len(rows) != EXPECTED_COUNTS["validation"]:
        raise ValueError(f"{role}: incomplete prediction cohort")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"{role}: duplicate prediction image")
    if sum(int(row["tumor"]) for row in rows) != EXPECTED_COUNTS["tumor"]:
        raise ValueError(f"{role}: tumor image-label cohort mismatch")
    total_bytes = 0
    for row in rows:
        map_path = (prediction / row["map_path"]).resolve()
        try:
            map_path.relative_to(prediction.resolve())
        except ValueError as error:
            raise ValueError(f"{role}: map escapes prediction root") from error
        if not map_path.is_file() or sha256_file(map_path) != row["map_sha256"]:
            raise ValueError(f"{role}: missing/hash-mismatched map {row['image_id']}")
        values = np.load(map_path, allow_pickle=False)
        if values.shape != (320, 320) or values.dtype != np.float16:
            raise ValueError(f"{role}: map shape/dtype drift")
        if not np.isfinite(values).all() or float(values.min()) < 0 or float(values.max()) > 1:
            raise ValueError(f"{role}: invalid normalized map")
        total_bytes += map_path.stat().st_size
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["cohort"] != EXPECTED_COUNTS["validation"]
        or metadata["validation_gt_read"] is not False
        or metadata["test_evaluated"] is not False
    ):
        raise ValueError(f"{role}: generation data-access contract failed")
    if metadata["prediction_manifest_sha256"] != frozen["prediction_manifest_sha256"]:
        raise ValueError(f"{role}: metadata manifest hash drift")
    model_weights = metadata["model_snapshot"]["model.safetensors"]["sha256"]
    if model_weights != frozen["model_hashes"]["model.safetensors"]:
        raise ValueError(f"{role}: model hash differs from freeze")
    if role == "base" and model_weights != expected_base_weight:
        raise ValueError("Base arm weight hash mismatch")
    return {
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "generation_metadata_sha256": sha256_file(metadata_path),
        "maps": len(rows),
        "map_bytes": total_bytes,
        "model_weight_sha256": model_weights,
        "noise_bank_sha256": metadata["inference"]["noise_bank_sha256"],
        "minimum_patch_coverage": metadata["inference"]["minimum_patch_coverage"],
    }


def _audit_evaluation_arm(
    run_root: Path,
    role: str,
    frozen_manifest_hash: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    evaluation = run_root / f"{role}_evaluation"
    per_image_path = evaluation / "per_image.csv"
    summary_path = evaluation / "summary.json"
    rows = _read_csv(per_image_path)
    by_name = {row["image_id"]: row for row in rows}
    if len(rows) != EXPECTED_COUNTS["tumor"] or len(by_name) != len(rows):
        raise ValueError(f"{role}: incomplete/duplicate tumor evaluation")
    counts = {
        group: sum(row["size_group"] == group for row in rows)
        for group in ("small", "medium", "large")
    }
    if counts != {key: EXPECTED_COUNTS[key] for key in counts}:
        raise ValueError(f"{role}: subgroup cohort mismatch {counts}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary["prediction_manifest_sha256"] != frozen_manifest_hash
        or summary["per_image_sha256"] != sha256_file(per_image_path)
        or summary["validation_gt_read_only_after_prediction_freeze"] is not True
        or summary["complete_misses_included"] is not True
        or summary["consumer_trained"] is not False
        or summary["test_evaluated"] is not False
    ):
        raise ValueError(f"{role}: evaluation contract mismatch")
    for stratum in ("overall", "small", "medium", "large"):
        selected = rows if stratum == "overall" else [
            row for row in rows if row["size_group"] == stratum
        ]
        reported = summary["tumor_localization"][stratum]
        if reported["n"] != len(selected):
            raise ValueError(f"{role}/{stratum}: summary count mismatch")
        for metric in METRICS:
            recomputed = float(np.mean([float(row[metric]) for row in selected]))
            _assert_close(
                recomputed,
                float(reported[metric]),
                f"{role}/{stratum}/{metric}",
            )
    return by_name, {
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "subgroups": counts,
    }


def _group_bootstrap(
    pairs: list[tuple[str, float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for group_id, delta in pairs:
        groups.setdefault(group_id, []).append(delta)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    results = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        values: list[float] = []
        for sampled_group in sampled_groups:
            values.extend(groups[str(sampled_group)])
        results[index] = np.mean(values)
    return {
        "delta_adapted_minus_base": float(np.mean([delta for _, delta in pairs])),
        "ci95": [float(value) for value in np.percentile(results, [2.5, 97.5])],
        "n_images": len(pairs),
        "n_groups": len(group_ids),
    }


def _audit_comparison(
    *,
    run_root: Path,
    base: dict[str, dict[str, str]],
    adapted: dict[str, dict[str, str]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if set(base) != set(adapted):
        raise ValueError("Evaluation arms are not paired")
    for name in base:
        for field in ("group_id", "size_group", "gt_area_ratio"):
            if base[name][field] != adapted[name][field]:
                raise ValueError(f"Paired field drift for {name}/{field}")
    comparison_path = run_root / "paired_comparison.json"
    reported = json.loads(comparison_path.read_text(encoding="utf-8"))
    if reported["replicates"] != replicates or reported["seed"] != seed:
        raise ValueError("Bootstrap configuration drift")
    recomputed: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        recomputed[metric] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name for name, row in base.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            pairs = [
                (
                    base[name]["group_id"],
                    float(adapted[name][metric]) - float(base[name][metric]),
                )
                for name in names
            ]
            result = _group_bootstrap(
                pairs,
                replicates=replicates,
                seed=seed + metric_index * 10 + len(stratum),
            )
            claimed = reported["metrics"][metric][stratum]
            _assert_close(
                result["delta_adapted_minus_base"],
                float(claimed["delta_adapted_minus_base"]),
                f"{metric}/{stratum}/delta",
            )
            for index in range(2):
                _assert_close(
                    result["ci95"][index],
                    float(claimed["ci95"][index]),
                    f"{metric}/{stratum}/ci95[{index}]",
                )
            if result["n_images"] != claimed["n_images"] or result["n_groups"] != claimed["n_groups"]:
                raise ValueError(f"{metric}/{stratum}: bootstrap cohort drift")
            recomputed[metric][stratum] = result
    return {
        "paired_comparison_sha256": sha256_file(comparison_path),
        "recomputed": recomputed,
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run_manifest["source_commit"] != args.expected_source_commit
        or run_manifest["protocol_sha256"] != args.expected_protocol_sha256
        or run_manifest["split_sha256"] != args.expected_split_sha256
        or run_manifest["test_evaluated"] is not False
        or run_manifest["consumer_trained"] is not False
    ):
        raise ValueError("Run manifest provenance/data contract mismatch")
    wrapper = _audit_wrapper(args.wrapper_source, args.expected_wrapper_sha256)
    if run_manifest["wrapper_sha256"] != wrapper["sha256"]:
        raise ValueError("Cloud run wrapper hash differs from locked wrapper")
    freeze_path = run_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != run_manifest["prediction_freeze_sha256"]:
        raise ValueError("Prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["validation_gt_read"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Prediction freeze contract mismatch")

    training_meta_path = run_root / "adapted_training/run_metadata.json"
    training = json.loads(training_meta_path.read_text(encoding="utf-8"))
    checkpoint_path = run_root / "adapted_training/model/model.safetensors"
    if (
        training["population"] != {
            "all_train_images": 2981,
            "normal_training_images": 1493,
            "tumor_training_images_used": 0,
        }
        or training["validation_gt_read"] is not False
        or training["test_evaluated"] is not False
        or sha256_file(checkpoint_path) != training["final_checkpoint"]["weight_sha256"]
        or run_manifest["adapted_checkpoint_sha256"]
        != training["final_checkpoint"]["weight_sha256"]
    ):
        raise ValueError("Adapted training/checkpoint contract mismatch")

    prediction_audit = {
        role: _audit_prediction_arm(
            run_root=run_root,
            role=role,
            frozen=freeze["arms"][role],
            expected_base_weight=args.expected_base_weight_sha256,
        )
        for role in ("base", "normal_adapted")
    }
    if (
        prediction_audit["base"]["noise_bank_sha256"]
        != prediction_audit["normal_adapted"]["noise_bank_sha256"]
    ):
        raise ValueError("Inference noise bank differs between arms")
    base_rows, base_eval = _audit_evaluation_arm(
        run_root, "base", freeze["arms"]["base"]["prediction_manifest_sha256"]
    )
    adapted_rows, adapted_eval = _audit_evaluation_arm(
        run_root,
        "normal_adapted",
        freeze["arms"]["normal_adapted"]["prediction_manifest_sha256"],
    )
    comparison = _audit_comparison(
        run_root=run_root,
        base=base_rows,
        adapted=adapted_rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    if comparison["paired_comparison_sha256"] != run_manifest["paired_comparison_sha256"]:
        raise ValueError("Run-manifest comparison hash mismatch")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "wrapper": wrapper,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "adapted_training_metadata_sha256": sha256_file(training_meta_path),
        "adapted_checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "predictions": prediction_audit,
        "evaluations": {
            "base": base_eval,
            "normal_adapted": adapted_eval,
        },
        "comparison": comparison,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
