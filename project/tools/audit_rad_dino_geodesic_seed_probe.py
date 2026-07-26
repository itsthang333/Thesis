from __future__ import annotations

"""Independent physical/result audit for the RAD-DINO geodesic probe."""

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.btxrd import BTXRDSegmentationDataset


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
STRATA = ("overall", "small", "medium", "large")
EXPECTED_COHORT = {"validation": 371, "tumor": 184, "normal": 187}
EXPECTED_SUBGROUPS = {"small": 94, "medium": 72, "large": 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--source-map-root", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-checkout-commit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-baseline-lf-sha256", required=True)
    parser.add_argument("--expected-baseline-runtime-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_bytes(values: bytes) -> list[dict[str, str]]:
    text = values.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def assert_close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = 1.0e-12,
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def crlf_hash_from_file(path: Path) -> tuple[str, str]:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return (
        sha256_bytes(normalized),
        sha256_bytes(normalized.replace(b"\n", b"\r\n")),
    )


def audit_static_sources(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.wrapper_source) != args.expected_wrapper_sha256:
        raise ValueError("Downloaded wrapper SHA-256 mismatch")
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError("Downloaded protocol SHA-256 mismatch")
    wrapper = args.wrapper_source.read_text(encoding="utf-8")
    ast.parse(wrapper)
    markers = {
        "build_prediction_command": "generate = generation_command(",
        "run_prediction": "run(generate,",
        "audit_prediction_freeze": "frozen = audit_frozen_predictions(",
        "build_evaluation_command": "evaluate = evaluation_command(",
        "run_evaluation": "run(evaluate,",
    }
    positions = {name: wrapper.index(marker) for name, marker in markers.items()}
    if list(positions.values()) != sorted(positions.values()):
        raise ValueError("Wrapper prediction/freeze/evaluation order drift")

    protocol = read_json(args.protocol)
    expected_sources = protocol["source"]["files"]
    local_sources = {
        "project/run_rad_dino_geodesic_seed_probe.py": args.runner_source,
        "project/evaluate_rad_dino_geodesic_seed_probe.py": args.evaluator_source,
    }
    materialized: dict[str, str] = {}
    for relative, path in local_sources.items():
        _lf_hash, crlf_hash = crlf_hash_from_file(path)
        if crlf_hash != expected_sources[relative]:
            raise ValueError(f"Local source differs from protocol: {relative}")
        materialized[relative] = crlf_hash

    runner_source = args.runner_source.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    runner_imports: list[str] = []
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Import):
            runner_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            runner_imports.append(node.module or "")
    if any(name.startswith("datasets") for name in runner_imports):
        raise ValueError("Prediction runner imports a segmentation dataset")
    if "BTXRDSegmentationDataset" in runner_source or "Annotations" in runner_source:
        raise ValueError("Prediction runner contains validation-spatial-GT access")
    if 'split="test"' in runner_source or "split='test'" in runner_source:
        raise ValueError("Prediction runner contains test-split access")

    evaluator = args.evaluator_source.read_text(encoding="utf-8")
    ast.parse(evaluator)
    freeze_positions = {
        "manifest_hash": evaluator.index(
            "if sha256_file(manifest_path) != expected_manifest:"
        ),
        "freeze_hash": evaluator.index(
            "if sha256_file(freeze_path) != expected_freeze:"
        ),
        "physical_map_loop": evaluator.index("for row in manifest:"),
        "first_gt_marker": evaluator.index(
            "# First validation-spatial-GT access: all predictions and hashes are frozen."
        ),
        "dataset_instantiation": evaluator.index(
            "dataset = BTXRDSegmentationDataset("
        ),
    }
    if list(freeze_positions.values()) != sorted(freeze_positions.values()):
        raise ValueError("Evaluator freeze-before-GT order drift")
    return {
        "wrapper_sha256": sha256_file(args.wrapper_source),
        "protocol_sha256": sha256_file(args.protocol),
        "wrapper_order_offsets": positions,
        "evaluator_order_offsets": freeze_positions,
        "protocol_runtime_source_hashes": materialized,
        "prediction_runner_dataset_imports": False,
        "prediction_runner_annotation_access": False,
        "test_split_access": False,
    }


def audit_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(args.protocol)
    affinity = protocol["frozen_affinity_input"]
    baseline = protocol["evaluation"]["baseline"]
    if (
        protocol["status"]
        != "predeclared_before_any_geodesic_refinement_prediction"
        or protocol["source"]["commit"] != args.expected_source_commit
        or protocol["data_contract"]["split_sha256"] != args.expected_split_sha256
        or baseline["per_image_sha256"]
        != args.expected_baseline_runtime_sha256
        or affinity["contains_validation_gt_derived_metrics"] is not False
        or affinity["consumer_trained"] is not False
        or affinity["test_evaluated"] is not False
        or protocol["runtime"]["test_locked"] is not True
    ):
        raise ValueError("Predeclared protocol contract mismatch")
    return protocol


def audit_run_contract(
    root: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_path = root / "run_manifest.json"
    freeze_path = root / "prediction_freeze.json"
    generation_path = root / "predictions/generation_metadata.json"
    manifest_path = root / "predictions/prediction_manifest.csv"
    run = read_json(run_path)
    freeze = read_json(freeze_path)
    generation = read_json(generation_path)
    if (
        run["checkout_commit"] != args.expected_checkout_commit
        or run["scientific_source_commit"] != args.expected_source_commit
        or run["protocol_sha256"] != args.expected_protocol_sha256
        or run["wrapper_sha256"] != args.expected_wrapper_sha256
        or run["split_audit"] != EXPECTED_COHORT
        or run["validation_gt_read_only_after_prediction_freeze"] is not True
        or run["consumer_trained"] is not False
        or run["test_evaluated"] is not False
    ):
        raise ValueError("Run-manifest provenance/lock mismatch")
    if (
        run["source_audit"]["split_sha256"] != args.expected_split_sha256
        or run["source_audit"]["baseline_per_image_sha256"]
        != args.expected_baseline_runtime_sha256
        or run["model"]["hashes"]["model.safetensors"]
        != args.expected_model_weight_sha256
        or run["source_image_audit_before_prediction"]["images_verified"] != 371
        or run["source_image_audit_before_prediction"]["annotation_files_opened"] != 0
        or run["source_image_audit_before_prediction"]["validation_gt_read"]
        is not False
    ):
        raise ValueError("Run-manifest source/input audit mismatch")
    affinity_protocol = protocol["frozen_affinity_input"]
    affinity_run = run["affinity_input_audit"]
    for key, protocol_key in {
        "prediction_manifest_sha256": "prediction_manifest_sha256",
        "package_manifest_sha256": "package_manifest_sha256",
        "prediction_freeze_sha256": "prediction_freeze_sha256",
    }.items():
        if affinity_run[key] != affinity_protocol[protocol_key]:
            raise ValueError(f"Affinity direct-input hash mismatch: {key}")
    if (
        not affinity_run["direct_mount_root"].startswith("/kaggle/input/")
        or affinity_run["physical_map_hashes_verified"] != 371
        or affinity_run["consumer_trained"] is not False
        or affinity_run["test_evaluated"] is not False
    ):
        raise ValueError("Affinity direct-mount audit mismatch")

    manifest_hash = sha256_file(manifest_path)
    generation_hash = sha256_file(generation_path)
    freeze_hash = sha256_file(freeze_path)
    if (
        freeze["prediction_manifest_sha256"] != manifest_hash
        or freeze["generation_metadata_sha256"] != generation_hash
        or freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_manifest_sha256"] != args.expected_split_sha256
        or freeze["validation_predictions"] != 371
        or freeze["validation_gt_read"] is not False
        or freeze["consumer_trained"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Prediction-freeze contract mismatch")
    affinity_generation = generation["source_affinity_contract"]
    if (
        generation["stage"]
        != "prediction-first RAD-DINO geodesic seed expansion"
        or generation["source_commit"] != args.expected_source_commit
        or generation["protocol_sha256"] != args.expected_protocol_sha256
        or generation["split_manifest_sha256"] != args.expected_split_sha256
        or generation["cohort"] != EXPECTED_COHORT
        or generation["maps"]["count"] != 371
        or generation["maps"]["manifest_sha256"] != manifest_hash
        or generation["model_snapshot"]["model.safetensors"]["sha256"]
        != args.expected_model_weight_sha256
        or affinity_generation["manifest_sha256"]
        != affinity_protocol["prediction_manifest_sha256"]
        or generation["validation_gt_read"] is not False
        or generation["consumer_trained"] is not False
        or generation["test_evaluated"] is not False
    ):
        raise ValueError("Generation metadata contract mismatch")
    run_freeze = run["prediction_freeze_audit_before_gt"]
    if (
        run_freeze["prediction_manifest_sha256"] != manifest_hash
        or run_freeze["generation_metadata_sha256"] != generation_hash
        or run_freeze["prediction_freeze_sha256"] != freeze_hash
        or run_freeze["physical_map_hashes_verified"] != 371
        or run_freeze["source_map_hash_bindings_verified"] != 371
        or run_freeze["normal_nonzero_pixels"] != 0
        or run_freeze["validation_gt_read"] is not False
    ):
        raise ValueError("Wrapper pre-GT freeze audit mismatch")
    return run, freeze, generation


def audit_maps(
    root: Path,
    freeze: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prediction = root / "predictions"
    manifest_path = prediction / "prediction_manifest.csv"
    rows = read_csv(manifest_path)
    expected_paths = {row["map_path"] for row in rows}
    actual_paths = {
        path.relative_to(prediction).as_posix()
        for path in (prediction / "maps").glob("*.npy")
    }
    if (
        sha256_file(manifest_path) != freeze["prediction_manifest_sha256"]
        or len(rows) != 371
        or len({row["image_id"] for row in rows}) != 371
        or sum(int(row["tumor"]) for row in rows) != 184
        or len(expected_paths) != 371
        or expected_paths != actual_paths
    ):
        raise ValueError("Frozen prediction cohort/map set mismatch")
    total_bytes = 0
    normal_nonzero = 0
    for row in rows:
        relative = Path(row["map_path"])
        path = (prediction / relative).resolve()
        try:
            path.relative_to(prediction.resolve())
        except ValueError as error:
            raise ValueError("Prediction map escapes run root") from error
        if not path.is_file() or sha256_file(path) != row["map_sha256"]:
            raise ValueError(f"Physical map hash mismatch: {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if (
            values.dtype != np.float16
            or values.shape != (320, 320)
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Physical map schema mismatch: {row['image_id']}")
        assert_close(
            float(values.mean()),
            float(row["raw_mean"]),
            f"{row['image_id']}/raw_mean",
            atol=5.0e-4,
        )
        assert_close(
            float(values.max()),
            float(row["raw_max"]),
            f"{row['image_id']}/raw_max",
            atol=5.0e-4,
        )
        if int(row["tumor"]) == 0:
            normal_nonzero += int(np.count_nonzero(values))
        total_bytes += path.stat().st_size
    if total_bytes != freeze["physical_map_bytes"] or normal_nonzero:
        raise ValueError("Physical map bytes or normal-map zeros mismatch")
    return rows, {
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "maps": len(rows),
        "physical_map_hashes_verified": len(rows),
        "physical_map_bytes": total_bytes,
        "normal_nonzero_pixels": normal_nonzero,
    }


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def audit_source_map_relation(
    root: Path,
    candidate_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_root = args.source_map_root.resolve()
    manifest_path = source_root / "prediction_manifest.csv"
    if sha256_file(manifest_path) != args.expected_source_manifest_sha256:
        raise ValueError("Frozen source-map manifest SHA-256 mismatch")
    source_rows = read_csv(manifest_path)
    source = {row["image_id"]: row for row in source_rows}
    candidate = {row["image_id"]: row for row in candidate_rows}
    if (
        len(source_rows) != 371
        or len(source) != 371
        or set(source) != set(candidate)
    ):
        raise ValueError("Source/candidate map cohorts differ")
    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "maps").glob("*.npy")
    }
    if actual_paths != {row["map_path"] for row in source_rows}:
        raise ValueError("Frozen source physical map set mismatch")

    mean_absolute_difference: list[float] = []
    pearson_correlation: list[float] = []
    top10_jaccard: list[float] = []
    top3_jaccard: list[float] = []
    argmax_same: list[float] = []
    source_bytes = 0
    for image_id, source_row in source.items():
        source_path = source_root / source_row["map_path"]
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source_row["map_sha256"]
        ):
            raise ValueError(f"Frozen source map hash mismatch: {image_id}")
        values = np.load(source_path, allow_pickle=False)
        if (
            values.dtype != np.float16
            or values.shape != (320, 320)
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Frozen source map schema mismatch: {image_id}")
        source_bytes += source_path.stat().st_size
        if candidate[image_id]["source_map_sha256"] != source_row["map_sha256"]:
            raise ValueError(f"Candidate source-map binding mismatch: {image_id}")
        if int(source_row["tumor"]) == 0:
            continue
        refined = np.load(
            root / "predictions" / candidate[image_id]["map_path"],
            allow_pickle=False,
        ).astype(np.float32)
        values = values.astype(np.float32)
        mean_absolute_difference.append(float(np.mean(np.abs(refined - values))))
        pearson_correlation.append(
            float(np.corrcoef(values.reshape(-1), refined.reshape(-1))[0, 1])
        )
        for percentile, output in ((90, top10_jaccard), (97, top3_jaccard)):
            source_support = values >= float(np.percentile(values, percentile))
            refined_support = refined >= float(np.percentile(refined, percentile))
            union = int(np.logical_or(source_support, refined_support).sum())
            output.append(
                1.0
                if union == 0
                else float(np.logical_and(source_support, refined_support).sum())
                / union
            )
        argmax_same.append(
            float(int(np.argmax(values)) == int(np.argmax(refined)))
        )
    if source_bytes != 76028288:
        raise ValueError("Frozen source map-byte total mismatch")
    return {
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_maps_verified": len(source_rows),
        "source_map_bytes": source_bytes,
        "tumor_maps_compared": len(mean_absolute_difference),
        "mean_absolute_difference": distribution(mean_absolute_difference),
        "pixelwise_pearson_correlation": distribution(pearson_correlation),
        "top10_percent_support_jaccard": distribution(top10_jaccard),
        "top3_percent_support_jaccard": distribution(top3_jaccard),
        "argmax_identity_rate": float(np.mean(argmax_same)),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return (
        1.0
        if denominator == 0
        else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator
    )


def subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        **{
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
    }


def recompute_evaluation(
    root: Path,
    manifest: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # Deliberately called only after audit_maps verifies all physical maps.
    dataset = BTXRDSegmentationDataset(
        root=args.dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=args.split_manifest,
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, mask, name = dataset[index]
        gt_by_name[str(name)] = mask[0].numpy() > 0.5
    if set(gt_by_name) != {row["image_id"] for row in manifest}:
        raise ValueError("Validation GT and prediction cohorts differ")

    per_image_path = root / "evaluation/per_image.csv"
    reported_rows = read_csv(per_image_path)
    reported = {row["image_id"]: row for row in reported_rows}
    if len(reported_rows) != 184 or len(reported) != 184:
        raise ValueError("Reported per-image cohort mismatch")
    recomputed: dict[str, dict[str, Any]] = {}
    image_scores: list[float] = []
    image_labels: list[int] = []
    for row in manifest:
        image_scores.append(float(row["raw_p99"]))
        image_labels.append(int(row["tumor"]))
        if int(row["tumor"]) == 0:
            continue
        values = np.load(
            root / "predictions" / row["map_path"], allow_pickle=False
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        area = float(target.mean())
        item: dict[str, Any] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": area,
            "size_group": subgroup(area),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(
                target.reshape(-1)[int(np.argmax(flat_values))]
            ),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1.0e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = dice(
                values >= float(np.percentile(values, percentile)),
                target,
            )
        claimed = reported[row["image_id"]]
        if (
            claimed["group_id"] != item["group_id"]
            or claimed["size_group"] != item["size_group"]
        ):
            raise ValueError(f"Evaluation identity mismatch: {row['image_id']}")
        for metric in ("gt_area_ratio", *METRICS):
            assert_close(
                float(item[metric]),
                float(claimed[metric]),
                f"{row['image_id']}/{metric}",
            )
        recomputed[row["image_id"]] = item

    counts = {
        name: sum(row["size_group"] == name for row in recomputed.values())
        for name in ("small", "medium", "large")
    }
    if counts != EXPECTED_SUBGROUPS:
        raise ValueError(f"Subgroup counts differ: {counts}")
    localization = {
        "overall": summarize(list(recomputed.values())),
        **{
            name: summarize(
                [
                    row
                    for row in recomputed.values()
                    if row["size_group"] == name
                ]
            )
            for name in ("small", "medium", "large")
        },
    }
    summary_path = root / "evaluation/summary.json"
    summary = read_json(summary_path)
    if (
        summary["cohort"] != EXPECTED_COHORT
        or summary["subgroups"] != EXPECTED_SUBGROUPS
        or summary["per_image_sha256"] != sha256_file(per_image_path)
        or summary["complete_misses_included"] is not True
        or summary["validation_gt_read_only_after_prediction_freeze"] is not True
        or summary["consumer_trained"] is not False
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("Evaluation summary contract mismatch")
    image_auroc = float(roc_auc_score(image_labels, image_scores))
    assert_close(
        image_auroc,
        float(summary["image_level_auroc_from_raw_p99"]),
        "image_level_auroc_from_raw_p99",
    )
    for stratum in STRATA:
        for metric in METRICS:
            assert_close(
                float(localization[stratum][metric]),
                float(summary["tumor_localization"][stratum][metric]),
                f"{stratum}/{metric}",
            )
    complete_misses = {
        f"dice_p{percentile}": {
            stratum: sum(
                float(row[f"dice_p{percentile}"]) == 0.0
                for row in recomputed.values()
                if stratum == "overall" or row["size_group"] == stratum
            )
            for stratum in STRATA
        }
        for percentile in (90, 95, 97, 99)
    }
    return recomputed, {
        "per_image_sha256": sha256_file(per_image_path),
        "summary_sha256": sha256_file(summary_path),
        "image_level_auroc_from_raw_p99": image_auroc,
        "localization": localization,
        "subgroups": counts,
        "complete_misses": complete_misses,
    }


def paired_group_report(
    rows: list[tuple[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for group_id, delta in rows:
        groups.setdefault(group_id, []).append(float(delta))
    group_ids = sorted(groups)
    if not group_ids:
        raise ValueError("Bootstrap requires nonempty groups")
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group in sampled for value in groups[str(group)]]
        boot[index] = float(np.mean(values))
    return {
        "mean_delta": float(np.mean([delta for _, delta in rows])),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "images": len(rows),
        "groups": len(group_ids),
        "iterations": iterations,
        "seed": seed,
    }


def audit_comparison(
    root: Path,
    candidate: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_bytes = args.baseline_per_image.read_bytes().replace(b"\r\n", b"\n")
    if sha256_bytes(baseline_bytes) != args.expected_baseline_lf_sha256:
        raise ValueError("Tracked baseline LF hash mismatch")
    runtime_bytes = baseline_bytes.replace(b"\n", b"\r\n")
    if sha256_bytes(runtime_bytes) != args.expected_baseline_runtime_sha256:
        raise ValueError("Frozen baseline runtime hash mismatch")
    baseline_rows = read_csv_bytes(runtime_bytes)
    baseline = {row["image_id"]: row for row in baseline_rows}
    if len(baseline_rows) != 184 or set(baseline) != set(candidate):
        raise ValueError("Baseline/candidate comparison cohort mismatch")

    comparison_path = root / "evaluation/paired_comparison.json"
    reported = read_json(comparison_path)
    if (
        reported["method"] != "paired complete-group bootstrap"
        or reported["iterations"] != args.bootstrap_iterations
        or reported["seed_family"] != args.bootstrap_seed
        or reported["consumer_trained"] is not False
        or reported["test_evaluated"] is not False
    ):
        raise ValueError("Reported comparison configuration mismatch")
    recomputed: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        recomputed[metric] = {}
        for stratum_index, stratum in enumerate(STRATA):
            names = [
                name
                for name, row in candidate.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            result = paired_group_report(
                [
                    (
                        str(candidate[name]["group_id"]),
                        float(candidate[name][metric]) - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + metric_index * 10 + stratum_index,
            )
            claimed = reported["metrics"][metric][stratum]
            for key in ("mean_delta", "ci95_low", "ci95_high"):
                assert_close(
                    float(result[key]),
                    float(claimed[key]),
                    f"{metric}/{stratum}/{key}",
                )
            for key in ("images", "groups", "iterations", "seed"):
                if int(result[key]) != int(claimed[key]):
                    raise ValueError(f"{metric}/{stratum}/{key} mismatch")
            recomputed[metric][stratum] = result
    return {
        "paired_comparison_sha256": sha256_file(comparison_path),
        "baseline_lf_sha256": sha256_bytes(baseline_bytes),
        "baseline_runtime_sha256": sha256_bytes(runtime_bytes),
        "metrics": recomputed,
    }


def audit_gate(
    root: Path,
    protocol: dict[str, Any],
    evaluation: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    reported_path = root / "evaluation/gate_decision.json"
    reported = read_json(reported_path)
    localization = evaluation["localization"]
    observed = {
        "image_level_auroc_from_raw_p99": evaluation[
            "image_level_auroc_from_raw_p99"
        ],
        "overall_pixel_auroc": localization["overall"]["pixel_auroc"],
        "small_pixel_auroc": localization["small"]["pixel_auroc"],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p97": localization["small"]["dice_p97"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    minimums = protocol["predeclared_gate"]["absolute_minimum"]
    absolute_pass = True
    for name, minimum in minimums.items():
        check = reported["absolute_checks"][name]
        assert_close(float(check["observed"]), float(observed[name]), f"gate/{name}")
        assert_close(float(check["minimum"]), float(minimum), f"gate/{name}/minimum")
        expected = float(observed[name]) >= float(minimum)
        if check["pass"] is not expected:
            raise ValueError(f"Gate absolute check mismatch: {name}")
        absolute_pass = absolute_pass and expected

    p90 = comparison["metrics"]["dice_p90"]
    expected_overall = float(p90["overall"]["ci95_low"]) > 0.0
    expected_subgroups = all(
        float(p90[group]["mean_delta"]) >= 0.0
        for group in ("small", "medium", "large")
    )
    relative = reported["relative_checks"]
    assert_close(
        float(relative["overall_dice_p90_ci95_low_above_zero"]["observed"]),
        float(p90["overall"]["ci95_low"]),
        "gate/overall_dice_p90_ci95_low",
    )
    for group in ("small", "medium", "large"):
        assert_close(
            float(
                relative["no_subgroup_mean_dice_p90_decrease"]["observed"][group]
            ),
            float(p90[group]["mean_delta"]),
            f"gate/{group}_dice_p90_delta",
        )
    if (
        relative["overall_dice_p90_ci95_low_above_zero"]["pass"]
        is not expected_overall
        or relative["no_subgroup_mean_dice_p90_decrease"]["pass"]
        is not expected_subgroups
    ):
        raise ValueError("Gate relative-check mismatch")
    expected_status = (
        "PASS"
        if absolute_pass and expected_overall and expected_subgroups
        else "FAIL"
    )
    expected_decision = (
        "AUTHORIZE_SEPARATE_PARTIAL_LABEL_CONSUMER_PROTOCOL"
        if expected_status == "PASS"
        else "REJECT_FIXED_GEODESIC_CONFIGURATION"
    )
    if (
        reported["status"] != expected_status
        or reported["decision"] != expected_decision
        or reported["all_checks_required"] is not True
        or reported["consumer_trained"] is not False
        or reported["test_evaluated"] is not False
    ):
        raise ValueError("Final gate decision mismatch")
    return {
        "gate_decision_sha256": sha256_file(reported_path),
        "status": expected_status,
        "decision": expected_decision,
        "absolute_observed": {key: float(value) for key, value in observed.items()},
        "overall_dice_p90_ci95_low": float(p90["overall"]["ci95_low"]),
        "subgroup_dice_p90_mean_delta": {
            group: float(p90[group]["mean_delta"])
            for group in ("small", "medium", "large")
        },
    }


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("Local frozen split SHA-256 mismatch")
    static = audit_static_sources(args)
    protocol = audit_protocol(args)
    run, freeze, generation = audit_run_contract(root, args, protocol)
    manifest, map_audit = audit_maps(root, freeze)
    source_relation = audit_source_map_relation(root, manifest, args)
    # Validation spatial GT is reached only after the complete map audit.
    candidate, evaluation = recompute_evaluation(root, manifest, args)
    comparison = audit_comparison(root, candidate, args)
    gate = audit_gate(root, protocol, evaluation, comparison)
    run_evaluation = run["evaluation_audit"]
    if (
        run_evaluation["summary_sha256"] != evaluation["summary_sha256"]
        or run_evaluation["paired_comparison_sha256"]
        != comparison["paired_comparison_sha256"]
        or run_evaluation["gate_decision_sha256"] != gate["gate_decision_sha256"]
        or run_evaluation["gate_status"] != gate["status"]
        or run_evaluation["decision"] != gate["decision"]
    ):
        raise ValueError("Run-manifest evaluation audit mismatch")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "prediction_freeze_sha256": sha256_file(root / "prediction_freeze.json"),
        "generation_metadata_sha256": sha256_file(
            root / "predictions/generation_metadata.json"
        ),
        "source_commit": args.expected_source_commit,
        "checkout_commit": args.expected_checkout_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_weight_sha256": args.expected_model_weight_sha256,
        "static_order": static,
        "generation_contract": {
            "stage": generation["stage"],
            "projection": generation["projection"],
            "graph": generation["graph"],
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        },
        "predictions": map_audit,
        "source_map_relation_before_gt": source_relation,
        "evaluation": evaluation,
        "comparison": comparison,
        "gate": gate,
        "cohort": {**EXPECTED_COHORT, **EXPECTED_SUBGROUPS},
        "validation_gt_read_only_after_prediction_freeze": True,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "predictions": result["predictions"],
                "gate": result["gate"],
                "consumer_trained": False,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
