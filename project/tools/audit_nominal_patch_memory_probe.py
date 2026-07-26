from __future__ import annotations

"""Independent physical/result audit for the nominal patch-memory probe."""

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
ARMS = ("single_scale", "multiscale")


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
    parser.add_argument("--expected-model-weight-sha256", required=True)
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
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError("Wrapper SHA-256 mismatch")
    source = path.read_text(encoding="utf-8")
    markers = {
        "prediction_generation": 'prediction_root = OUTPUT / "predictions"',
        "prediction_freeze": 'freeze_path = OUTPUT / "prediction_freeze.json"',
        "evaluation": 'evaluation = OUTPUT / f"{arm}_evaluation"',
        "comparison": 'comparison = OUTPUT / "paired_comparison.json"',
    }
    positions = {name: source.index(marker) for name, marker in markers.items()}
    if not (
        positions["prediction_generation"]
        < positions["prediction_freeze"]
        < positions["evaluation"]
        < positions["comparison"]
    ):
        raise ValueError("Wrapper freeze-before-GT ordering drift")
    return {"sha256": actual, "ordering_positions": positions}


def _semantic_array_hash(values: np.ndarray) -> str:
    values = np.asarray(values, dtype="<f4")
    payload = {
        "dtype": "float32",
        "shape": list(values.shape),
        "bytes_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _audit_memory(
    run_root: Path,
    *,
    expected_model_weight: str,
) -> dict[str, Any]:
    path = run_root / "predictions/memory_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata["population"]
        != {
            "all_train_images": 2981,
            "normal_memory_images": 1493,
            "tumor_training_images_used": 0,
            "validation_images": 371,
        }
        or metadata["annotation_contract"]
        != "segmentation annotation paths were never enumerated or opened"
        or metadata["validation_gt_read"] is not False
        or metadata["test_evaluated"] is not False
    ):
        raise ValueError("Normal-memory population/data-access contract mismatch")
    contract = metadata["feature_contract"]
    expected_contract = {
        "model_patch_size": 14,
        "input_size": 448,
        "grid_size": 32,
        "hidden_size": 768,
        "projection_dim": 128,
        "projection_seed": 42,
        "views": 5,
        "tile_size": 280,
        "tile_layout": [
            [0, 0, 280, 280],
            [168, 0, 448, 280],
            [0, 168, 280, 448],
            [168, 168, 448, 448],
        ],
        "top_k_normal_images": 8,
        "spatial_radius_patches": 2,
        "distance": "one minus maximum cosine similarity",
    }
    for key, expected in expected_contract.items():
        if contract[key] != expected:
            raise ValueError(f"Memory feature contract drift: {key}")
    weight_hash = metadata["model_snapshot"]["model.safetensors"]["sha256"]
    if weight_hash != expected_model_weight:
        raise ValueError("RAD-DINO model weight hash mismatch")

    evidence_root = path.parent / "memory_evidence"
    expected_shapes = {
        "projection": (768, 128),
        "normal_global_features": (1493, 768),
        "normal_context_indices": (1493, 8),
        "normal_context_similarities": (1493, 8),
        "full_calibration": (1493 * 32 * 32,),
        "tile_calibration": (1493 * 4 * 32 * 32,),
    }
    arrays: dict[str, Any] = {}
    for name, shape in expected_shapes.items():
        record = metadata["evidence_files"][name]
        evidence_path = evidence_root / record["path"]
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != record["sha256"]
            or evidence_path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"Memory evidence file mismatch: {name}")
        values = np.load(evidence_path, allow_pickle=False)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Memory evidence shape/value drift: {name}")
        arrays[name] = values
    indices = arrays["normal_context_indices"]
    if (
        indices.dtype != np.int32
        or np.any(indices < 0)
        or np.any(indices >= 1493)
        or np.any(indices == np.arange(1493, dtype=np.int32)[:, None])
    ):
        raise ValueError("Leave-one-image-out context evidence is invalid")
    if (
        _semantic_array_hash(arrays["projection"])
        != contract["projection_semantic_sha256"]
        or _semantic_array_hash(arrays["full_calibration"])
        != metadata["calibration"]["full"]["semantic_sha256"]
        or _semantic_array_hash(arrays["tile_calibration"])
        != metadata["calibration"]["tiles"]["semantic_sha256"]
    ):
        raise ValueError("Projection/calibration semantic hash mismatch")
    if (
        not np.all(np.diff(arrays["full_calibration"]) >= 0)
        or not np.all(np.diff(arrays["tile_calibration"]) >= 0)
    ):
        raise ValueError("Frozen normal calibration is not sorted")
    return {
        "memory_metadata_sha256": sha256_file(path),
        "model_weight_sha256": weight_hash,
        "evidence_files": {
            name: metadata["evidence_files"][name]["sha256"]
            for name in expected_shapes
        },
        "reconstructible_scratch_hashes": metadata[
            "reconstructible_scratch_hashes"
        ],
    }


def _audit_prediction_arm(
    run_root: Path,
    *,
    arm: str,
    frozen: dict[str, Any],
    memory_hash: str,
) -> dict[str, Any]:
    prediction = run_root / f"predictions/{arm}_prediction"
    manifest_path = prediction / "prediction_manifest.csv"
    metadata_path = prediction / "generation_metadata.json"
    if (
        sha256_file(manifest_path) != frozen["prediction_manifest_sha256"]
        or sha256_file(metadata_path) != frozen["generation_metadata_sha256"]
    ):
        raise ValueError(f"{arm}: prediction freeze hash mismatch")
    rows = _read_csv(manifest_path)
    if (
        len(rows) != EXPECTED_COUNTS["validation"]
        or len({row["image_id"] for row in rows}) != len(rows)
        or sum(int(row["tumor"]) for row in rows) != EXPECTED_COUNTS["tumor"]
    ):
        raise ValueError(f"{arm}: prediction cohort mismatch")
    total_bytes = 0
    for row in rows:
        if (
            len(row["context_indices"].split("|")) != 8
            or len(row["context_image_ids"].split("|")) != 8
            or len(row["context_similarities"].split("|")) != 8
        ):
            raise ValueError(f"{arm}: context evidence width mismatch")
        map_path = (prediction / row["map_path"]).resolve()
        try:
            map_path.relative_to(prediction.resolve())
        except ValueError as error:
            raise ValueError(f"{arm}: map escapes prediction root") from error
        if not map_path.is_file() or sha256_file(map_path) != row["map_sha256"]:
            raise ValueError(f"{arm}: missing/hash-mismatched map")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0
            or float(values.max()) > 1
        ):
            raise ValueError(f"{arm}: frozen map shape/dtype/value drift")
        total_bytes += map_path.stat().st_size
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["arm"] != arm
        or metadata["cohort"] != 371
        or metadata["memory_metadata_sha256"] != memory_hash
        or metadata["prediction_manifest_sha256"]
        != frozen["prediction_manifest_sha256"]
        or metadata["validation_gt_read"] is not False
        or metadata["test_evaluated"] is not False
    ):
        raise ValueError(f"{arm}: generation metadata contract mismatch")
    return {
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "generation_metadata_sha256": sha256_file(metadata_path),
        "maps": len(rows),
        "map_bytes": total_bytes,
    }


def _audit_evaluation(
    run_root: Path,
    *,
    arm: str,
    frozen_manifest_hash: str,
    memory_hash: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    evaluation = run_root / f"{arm}_evaluation"
    per_image_path = evaluation / "per_image.csv"
    summary_path = evaluation / "summary.json"
    rows = _read_csv(per_image_path)
    by_name = {row["image_id"]: row for row in rows}
    counts = {
        group: sum(row["size_group"] == group for row in rows)
        for group in ("small", "medium", "large")
    }
    if (
        len(rows) != 184
        or len(by_name) != len(rows)
        or counts != {"small": 94, "medium": 72, "large": 18}
    ):
        raise ValueError(f"{arm}: evaluation cohort/subgroup mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary["arm"] != arm
        or summary["prediction_manifest_sha256"] != frozen_manifest_hash
        or summary["memory_metadata_sha256"] != memory_hash
        or summary["per_image_sha256"] != sha256_file(per_image_path)
        or summary["validation_gt_read_only_after_prediction_freeze"] is not True
        or summary["complete_misses_included"] is not True
        or summary["consumer_trained"] is not False
        or summary["test_evaluated"] is not False
    ):
        raise ValueError(f"{arm}: evaluation contract mismatch")
    for stratum in ("overall", "small", "medium", "large"):
        selected = (
            rows
            if stratum == "overall"
            else [row for row in rows if row["size_group"] == stratum]
        )
        reported = summary["tumor_localization"][stratum]
        if reported["n"] != len(selected):
            raise ValueError(f"{arm}/{stratum}: count mismatch")
        for metric in METRICS:
            _assert_close(
                float(np.mean([float(row[metric]) for row in selected])),
                float(reported[metric]),
                f"{arm}/{stratum}/{metric}",
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
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values[index] = np.mean(
            [value for group in sampled for value in groups[str(group)]]
        )
    return {
        "delta_multiscale_minus_single_scale": float(
            np.mean([delta for _, delta in pairs])
        ),
        "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        "n_images": len(pairs),
        "n_groups": len(group_ids),
    }


def _audit_comparison(
    run_root: Path,
    *,
    single: dict[str, dict[str, str]],
    multiscale: dict[str, dict[str, str]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if set(single) != set(multiscale):
        raise ValueError("Evaluation arms are not paired")
    for name in single:
        for field in ("group_id", "size_group", "gt_area_ratio"):
            if single[name][field] != multiscale[name][field]:
                raise ValueError(f"Paired field drift for {name}/{field}")
    path = run_root / "paired_comparison.json"
    reported = json.loads(path.read_text(encoding="utf-8"))
    if (
        reported["replicates"] != replicates
        or reported["seed"] != seed
        or reported["test_evaluated"] is not False
    ):
        raise ValueError("Bootstrap configuration/data contract drift")
    recomputed: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        recomputed[metric] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in single.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            result = _group_bootstrap(
                [
                    (
                        single[name]["group_id"],
                        float(multiscale[name][metric])
                        - float(single[name][metric]),
                    )
                    for name in names
                ],
                replicates=replicates,
                seed=seed + metric_index * 10 + len(stratum),
            )
            claimed = reported["metrics"][metric][stratum]
            _assert_close(
                result["delta_multiscale_minus_single_scale"],
                float(claimed["delta_multiscale_minus_single_scale"]),
                f"{metric}/{stratum}/delta",
            )
            for index in range(2):
                _assert_close(
                    result["ci95"][index],
                    float(claimed["ci95"][index]),
                    f"{metric}/{stratum}/ci95[{index}]",
                )
            if (
                result["n_images"] != claimed["n_images"]
                or result["n_groups"] != claimed["n_groups"]
            ):
                raise ValueError(f"{metric}/{stratum}: bootstrap cohort drift")
            recomputed[metric][stratum] = result
    return {
        "paired_comparison_sha256": sha256_file(path),
        "recomputed": recomputed,
    }


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run_manifest["source_commit"] != args.expected_source_commit
        or run_manifest["protocol_sha256"] != args.expected_protocol_sha256
        or run_manifest["split_sha256"] != args.expected_split_sha256
        or run_manifest["consumer_trained"] is not False
        or run_manifest["test_evaluated"] is not False
    ):
        raise ValueError("Run manifest provenance/data contract mismatch")
    wrapper = _audit_wrapper(args.wrapper_source, args.expected_wrapper_sha256)
    if run_manifest["wrapper_sha256"] != wrapper["sha256"]:
        raise ValueError("Cloud wrapper differs from locally locked wrapper")
    memory = _audit_memory(
        root,
        expected_model_weight=args.expected_model_weight_sha256,
    )
    freeze_path = root / "prediction_freeze.json"
    if sha256_file(freeze_path) != run_manifest["prediction_freeze_sha256"]:
        raise ValueError("Prediction freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["memory_metadata_sha256"]
        != memory["memory_metadata_sha256"]
        or freeze["validation_gt_read"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Prediction freeze contract mismatch")
    predictions = {
        arm: _audit_prediction_arm(
            root,
            arm=arm,
            frozen=freeze["arms"][arm],
            memory_hash=memory["memory_metadata_sha256"],
        )
        for arm in ARMS
    }
    single_rows, single_eval = _audit_evaluation(
        root,
        arm="single_scale",
        frozen_manifest_hash=freeze["arms"]["single_scale"][
            "prediction_manifest_sha256"
        ],
        memory_hash=memory["memory_metadata_sha256"],
    )
    multi_rows, multi_eval = _audit_evaluation(
        root,
        arm="multiscale",
        frozen_manifest_hash=freeze["arms"]["multiscale"][
            "prediction_manifest_sha256"
        ],
        memory_hash=memory["memory_metadata_sha256"],
    )
    comparison = _audit_comparison(
        root,
        single=single_rows,
        multiscale=multi_rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    if comparison["paired_comparison_sha256"] != run_manifest[
        "paired_comparison_sha256"
    ]:
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
        "memory": memory,
        "predictions": predictions,
        "evaluations": {
            "single_scale": single_eval,
            "multiscale": multi_eval,
        },
        "comparison": comparison,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
