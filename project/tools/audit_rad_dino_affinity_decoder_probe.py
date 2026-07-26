from __future__ import annotations

"""Independent physical/result audit for the RAD-DINO affinity-decoder probe."""

import argparse
import ast
import csv
import hashlib
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
EXPECTED_COHORT = {
    "train": 2981,
    "train_normal": 1493,
    "train_tumor": 1488,
    "validation": 371,
    "validation_tumor": 184,
    "validation_normal": 187,
}
EXPECTED_SUBGROUPS = {"small": 94, "medium": 72, "large": 18}
RUN_MANIFEST_SOURCE_KEYS = {
    "runner": "run_rad_dino_affinity_decoder_probe.py",
    "affinity_decoder": "models/rad_dino_affinity_decoder.py",
    "nominal_memory": "models/nominal_patch_memory.py",
    "mae_reconstruction": "models/mae_reconstruction.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-checkout-commit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = 1.0e-12,
) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def audit_static_order(wrapper: Path, runner: Path, wrapper_hash: str) -> dict[str, Any]:
    if sha256_file(wrapper) != wrapper_hash:
        raise ValueError("Local wrapper SHA-256 mismatch")
    wrapper_source = wrapper.read_text(encoding="utf-8")
    ast.parse(wrapper_source)
    if "run_rad_dino_affinity_decoder_probe.py" not in wrapper_source:
        raise ValueError("Wrapper does not invoke the frozen runner")

    runner_source = runner.read_text(encoding="utf-8")
    tree = ast.parse(runner_source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls: list[tuple[int, str]] = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append((node.lineno, node.func.id))
        elif isinstance(node.func, ast.Attribute):
            calls.append((node.lineno, node.func.attr))

    def first(name: str) -> int:
        return min(line for line, called in calls if called == name)

    positions = {
        "training_cache": first("build_training_feature_cache"),
        "teacher_cache": first("build_teacher_cache"),
        "train_decoder": first("train_decoder"),
        "validation_maps": first("write_validation_maps"),
        "validation_evaluation": first("evaluate_frozen_predictions"),
        "paired_comparison": first("compare_to_frozen_nominal"),
        "gate": first("apply_gate"),
    }
    if list(positions.values()) != sorted(positions.values()):
        raise ValueError("Runner train/freeze/evaluate ordering drift")
    if runner_source.index("freeze_path.write_text") > runner_source.index(
        "evaluate_frozen_predictions(args)"
    ):
        raise ValueError("Validation GT evaluation precedes prediction freeze")
    before_evaluator = runner_source.split(
        "def evaluate_frozen_predictions", maxsplit=1
    )[0]
    if "BTXRDSegmentationDataset" in before_evaluator:
        raise ValueError("Validation GT dataset imported before evaluator")
    if 'split="test"' in runner_source or "split='test'" in runner_source:
        raise ValueError("Runner contains a test-split access")
    return {
        "wrapper_sha256": sha256_file(wrapper),
        "runner_sha256": sha256_file(runner),
        "ordering_lines": positions,
    }


def audit_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError("Protocol SHA-256 mismatch")
    protocol = read_json(args.protocol)
    supervision = protocol["weak_supervision_contract"]
    if (
        protocol["status"] != "predeclared_before_execution"
        or protocol["source"]["commit"] != args.expected_source_commit
        or supervision["consumer_trained"] is not False
        or supervision["test_evaluated"] is not False
    ):
        raise ValueError("Protocol supervision/provenance contract mismatch")
    if protocol["data_contract"]["split_sha256"] != args.expected_split_sha256:
        raise ValueError("Protocol split hash mismatch")
    if (
        protocol["evaluation"]["baseline"]["per_image_sha256"]
        != args.expected_baseline_sha256
    ):
        raise ValueError("Protocol baseline hash mismatch")
    return protocol


def audit_run_manifests(
    root: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path = root / "run_manifest.json"
    wrapper_path = root / "wrapper_provenance.json"
    freeze_path = root / "prediction_freeze.json"
    run_manifest = read_json(run_manifest_path)
    wrapper = read_json(wrapper_path)
    freeze = read_json(freeze_path)
    if (
        run_manifest["source_commit"] != args.expected_source_commit
        or run_manifest["protocol_sha256"] != args.expected_protocol_sha256
        or run_manifest["cohort"] != EXPECTED_COHORT
        or run_manifest["validation_gt_read_only_after_prediction_freeze"] is not True
        or run_manifest["complete_misses_included"] is not True
        or run_manifest["consumer_trained"] is not False
        or run_manifest["test_evaluated"] is not False
    ):
        raise ValueError("Run manifest contract mismatch")
    model_snapshot = run_manifest["model_snapshot"]
    if (
        model_snapshot["model.safetensors"]["sha256"]
        != args.expected_model_weight_sha256
    ):
        raise ValueError("RAD-DINO model weight mismatch")
    expected_sources = protocol["source"]["files"]
    if set(run_manifest["source_files"]) != set(RUN_MANIFEST_SOURCE_KEYS):
        raise ValueError("Run-manifest source-file set mismatch")
    for name, reported in run_manifest["source_files"].items():
        protocol_name = RUN_MANIFEST_SOURCE_KEYS[name]
        if reported != expected_sources[protocol_name]:
            raise ValueError(f"Run-manifest source hash mismatch: {name}")
    if (
        wrapper["checkout_commit"] != args.expected_checkout_commit
        or wrapper["scientific_source_commit"] != args.expected_source_commit
        or wrapper["protocol_sha256"] != args.expected_protocol_sha256
        or wrapper["wrapper_sha256"] != args.expected_wrapper_sha256
        or wrapper["split_sha256"] != args.expected_split_sha256
        or wrapper["baseline_per_image_sha256"] != args.expected_baseline_sha256
        or wrapper["model_hashes"]["model.safetensors"]
        != args.expected_model_weight_sha256
        or wrapper["run_manifest_sha256"] != sha256_file(run_manifest_path)
        or wrapper["prediction_maps"] != 371
        or wrapper["consumer_trained"] is not False
        or wrapper["test_evaluated"] is not False
    ):
        raise ValueError("Wrapper provenance mismatch")
    if (
        sha256_file(freeze_path) != run_manifest["prediction_freeze_sha256"]
        or freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["checkpoint_sha256"] != run_manifest["checkpoint_sha256"]
        or freeze["validation_predictions"] != 371
        or freeze["validation_gt_read"] is not False
        or freeze["consumer_trained"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Prediction freeze mismatch")
    return run_manifest, wrapper, freeze


def audit_training_evidence(
    root: Path,
    run_manifest: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = root / "rad_dino_affinity_decoder.pt"
    teacher_path = root / "teacher_metadata.json"
    history_path = root / "training_history.json"
    if sha256_file(checkpoint) != run_manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if sha256_file(teacher_path) != freeze["teacher_metadata_sha256"]:
        raise ValueError("Teacher metadata hash mismatch")
    teacher = read_json(teacher_path)
    if (
        teacher["normal_images"] != 1493
        or teacher["positive_images"] != 1488
        or teacher["projection_dim"] != 128
        or teacher["projection_seed"] != 42
        or teacher["normal_context_top_k"] != 8
        or teacher["teacher_spatial_radius"] != 2
        or teacher["normal_patch_cache_retained"] is not False
        or teacher["validation_gt_read"] is not False
        or teacher["test_evaluated"] is not False
    ):
        raise ValueError("Training teacher contract mismatch")
    distribution = teacher["positive_teacher_distribution"]
    if (
        not all(np.isfinite(float(value)) for value in distribution.values())
        or float(distribution["minimum"]) < 0
        or float(distribution["maximum"]) > 1
        or float(distribution["minimum"]) > float(distribution["maximum"])
    ):
        raise ValueError("Teacher distribution is invalid")
    history = read_json(history_path)
    if len(history) != 12 or [row["epoch"] for row in history] != list(range(1, 13)):
        raise ValueError("Training history does not contain fixed epochs 1..12")
    fields = (
        "total_loss",
        "image_loss",
        "pseudo_loss",
        "affinity_loss",
        "training_accuracy",
    )
    if any(
        not all(np.isfinite(float(row[field])) for field in fields)
        for row in history
    ):
        raise ValueError("Training history contains non-finite values")
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "teacher_metadata_sha256": sha256_file(teacher_path),
        "training_history_sha256": sha256_file(history_path),
        "final_epoch": history[-1],
    }


def audit_frozen_maps(
    root: Path,
    freeze: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prediction = root / "predictions"
    manifest_path = prediction / "prediction_manifest.csv"
    if sha256_file(manifest_path) != freeze["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest freeze mismatch")
    rows = read_csv(manifest_path)
    expected_paths = {row["map_path"] for row in rows}
    observed_paths = {
        path.relative_to(prediction).as_posix()
        for path in (prediction / "maps").glob("*.npy")
    }
    if (
        len(rows) != 371
        or len({row["image_id"] for row in rows}) != 371
        or len(expected_paths) != 371
        or sum(int(row["tumor"]) for row in rows) != 184
        or expected_paths != observed_paths
    ):
        raise ValueError("Frozen prediction cohort/map set mismatch")
    total_bytes = 0
    for row in rows:
        path = (prediction / row["map_path"]).resolve()
        try:
            path.relative_to(prediction.resolve())
        except ValueError as error:
            raise ValueError("Prediction map escapes run root") from error
        if not path.is_file() or sha256_file(path) != row["map_sha256"]:
            raise ValueError(f"Prediction map hash mismatch: {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0
            or float(values.max()) > 1
        ):
            raise ValueError(f"Prediction map value drift: {row['image_id']}")
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
        if not 0 <= float(row["raw_p99"]) <= 1:
            raise ValueError(f"Invalid foreground p99: {row['image_id']}")
        total_bytes += path.stat().st_size
    return rows, {
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "maps": len(rows),
        "map_bytes": total_bytes,
    }


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def recompute_validation_evaluation(
    root: Path,
    manifest: list[dict[str, str]],
    args: argparse.Namespace,
    freeze: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # This is deliberately called only after audit_frozen_maps has verified
    # the complete physical prediction freeze.
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
        raise ValueError("Validation GT and frozen prediction cohorts differ")

    reported_path = root / "predictions/evaluation/per_image.csv"
    summary_path = root / "predictions/evaluation/summary.json"
    reported_rows = read_csv(reported_path)
    reported = {row["image_id"]: row for row in reported_rows}
    if len(reported_rows) != 184 or len(reported) != 184:
        raise ValueError("Per-image evaluation cohort mismatch")

    recomputed: dict[str, dict[str, Any]] = {}
    for row in manifest:
        if row["tumor"] != "1":
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
                values >= np.percentile(values, percentile), target
            )
        claimed = reported[row["image_id"]]
        if (
            claimed["group_id"] != item["group_id"]
            or claimed["size_group"] != item["size_group"]
        ):
            raise ValueError(f"Evaluation identity drift: {row['image_id']}")
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
        raise ValueError(f"Evaluation subgroup contract drift: {counts}")
    summary = read_json(summary_path)
    if (
        summary["cohort"]
        != {"validation": 371, "tumor": 184, **EXPECTED_SUBGROUPS}
        or summary["prediction_manifest_sha256"]
        != freeze["prediction_manifest_sha256"]
        or summary["per_image_sha256"] != sha256_file(reported_path)
        or summary["validation_gt_read_only_after_prediction_freeze"] is not True
        or summary["complete_misses_included"] is not True
        or summary["consumer_trained"] is not False
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("Evaluation summary contract mismatch")
    image_labels = np.asarray([int(row["tumor"]) for row in manifest])
    image_scores = np.asarray([float(row["raw_p99"]) for row in manifest])
    assert_close(
        float(roc_auc_score(image_labels, image_scores)),
        float(summary["image_level_auroc_from_raw_p99"]),
        "image_level_auroc_from_raw_p99",
    )
    for name in ("overall", "small", "medium", "large"):
        selected = [
            row
            for row in recomputed.values()
            if name == "overall" or row["size_group"] == name
        ]
        claimed = summary["tumor_localization"][name]
        if claimed["n"] != len(selected):
            raise ValueError(f"{name}: summary count mismatch")
        for metric in METRICS:
            assert_close(
                float(np.mean([float(row[metric]) for row in selected])),
                float(claimed[metric]),
                f"{name}/{metric}",
            )
    return recomputed, {
        "per_image_sha256": sha256_file(reported_path),
        "summary_sha256": sha256_file(summary_path),
        "subgroups": counts,
        "summary": summary,
    }


def group_bootstrap(
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
    boot = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        boot[index] = np.mean(
            [value for group in sampled for value in groups[str(group)]]
        )
    return {
        "delta_decoder_minus_nominal": float(
            np.mean([delta for _, delta in pairs])
        ),
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "n_images": len(pairs),
        "n_groups": len(group_ids),
    }


def audit_comparison(
    root: Path,
    decoder: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if sha256_file(args.baseline_per_image) != args.expected_baseline_sha256:
        raise ValueError("Local frozen nominal baseline hash mismatch")
    baseline_rows = read_csv(args.baseline_per_image)
    baseline = {row["image_id"]: row for row in baseline_rows}
    if set(baseline) != set(decoder) or len(decoder) != 184:
        raise ValueError("Decoder and nominal comparison cohorts differ")
    for name in decoder:
        if (
            baseline[name]["group_id"] != decoder[name]["group_id"]
            or baseline[name]["size_group"] != decoder[name]["size_group"]
        ):
            raise ValueError(f"Paired comparison identity drift: {name}")
    path = root / "paired_comparison.json"
    reported = read_json(path)
    if (
        reported["baseline_per_image_sha256"] != args.expected_baseline_sha256
        or reported["replicates"] != args.bootstrap_replicates
        or reported["seed_family"] != args.bootstrap_seed
        or reported["consumer_trained"] is not False
        or reported["test_evaluated"] is not False
    ):
        raise ValueError("Paired comparison configuration mismatch")
    recomputed: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        recomputed[metric] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in decoder.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            result = group_bootstrap(
                [
                    (
                        str(decoder[name]["group_id"]),
                        float(decoder[name][metric])
                        - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + metric_index * 10 + len(stratum),
            )
            claimed = reported["metrics"][metric][stratum]
            assert_close(
                result["delta_decoder_minus_nominal"],
                float(claimed["delta_decoder_minus_nominal"]),
                f"{metric}/{stratum}/delta",
            )
            for index in range(2):
                assert_close(
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
    return {"sha256": sha256_file(path), "recomputed": recomputed}


def audit_gate(
    root: Path,
    summary: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gate_path = root / "gate_decision.json"
    reported = read_json(gate_path)
    localization = summary["tumor_localization"]
    observed = {
        "image_level_auroc_from_raw_p99": summary[
            "image_level_auroc_from_raw_p99"
        ],
        "overall_pixel_auroc": localization["overall"]["pixel_auroc"],
        "small_pixel_auroc": localization["small"]["pixel_auroc"],
        "overall_dice_p90": localization["overall"]["dice_p90"],
        "small_dice_p97": localization["small"]["dice_p97"],
        "medium_dice_p90": localization["medium"]["dice_p90"],
        "large_dice_p90": localization["large"]["dice_p90"],
    }
    minimums = protocol["predeclared_mechanistic_gate"]["minimum"]
    expected_status = (
        "PASS"
        if all(float(observed[name]) >= float(value) for name, value in minimums.items())
        else "FAIL"
    )
    if (
        reported["status"] != expected_status
        or reported["all_checks_required"] is not True
        or reported["consumer_trained"] is not False
        or reported["test_evaluated"] is not False
    ):
        raise ValueError("Gate status/contract mismatch")
    for name, minimum in minimums.items():
        check = reported["checks"][name]
        assert_close(float(check["observed"]), float(observed[name]), f"gate/{name}")
        assert_close(float(check["minimum"]), float(minimum), f"gate/{name}/minimum")
        if check["pass"] is not (float(observed[name]) >= float(minimum)):
            raise ValueError(f"Gate check differs: {name}")
    return {
        "sha256": sha256_file(gate_path),
        "status": expected_status,
        "observed": {name: float(value) for name, value in observed.items()},
    }


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("Local frozen split manifest SHA-256 mismatch")
    protocol = audit_protocol(args)
    static = audit_static_order(
        args.wrapper_source,
        args.runner_source,
        args.expected_wrapper_sha256,
    )
    if (
        static["runner_sha256"]
        != protocol["source"]["files"]["run_rad_dino_affinity_decoder_probe.py"]
    ):
        raise ValueError("Local runner differs from protocol-bound source")
    run_manifest, wrapper, freeze = audit_run_manifests(root, args, protocol)
    training = audit_training_evidence(root, run_manifest, freeze)
    manifest, maps = audit_frozen_maps(root, freeze)
    # Validation GT becomes reachable only after the full map freeze above.
    decoder, evaluation = recompute_validation_evaluation(
        root, manifest, args, freeze
    )
    comparison = audit_comparison(root, decoder, args)
    if comparison["sha256"] != run_manifest["paired_comparison_sha256"]:
        raise ValueError("Paired comparison run-manifest hash mismatch")
    gate = audit_gate(root, evaluation["summary"], protocol)
    if gate["sha256"] != run_manifest["gate_decision_sha256"]:
        raise ValueError("Gate run-manifest hash mismatch")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "wrapper_provenance_sha256": sha256_file(
            root / "wrapper_provenance.json"
        ),
        "source_commit": args.expected_source_commit,
        "checkout_commit": args.expected_checkout_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_weight_sha256": args.expected_model_weight_sha256,
        "baseline_per_image_sha256": args.expected_baseline_sha256,
        "static_order": static,
        "training": training,
        "prediction_freeze_sha256": sha256_file(root / "prediction_freeze.json"),
        "predictions": maps,
        "evaluation": {
            key: value for key, value in evaluation.items() if key != "summary"
        },
        "comparison": comparison,
        "gate": gate,
        "cohort": {
            **EXPECTED_COHORT,
            **EXPECTED_SUBGROUPS,
        },
        "validation_gt_read_only_after_prediction_freeze": True,
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
