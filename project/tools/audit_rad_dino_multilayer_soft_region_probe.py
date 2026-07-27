from __future__ import annotations

"""Independent physical and result audit for the multi-layer soft-region probe."""

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
GATE_THRESHOLDS = {
    "image_level_auroc_from_raw_p99": 0.65,
    "overall_pixel_auroc": 0.75,
    "small_pixel_auroc": 0.77,
    "overall_dice_p90": 0.10,
    "small_dice_p97": 0.03,
    "medium_dice_p90": 0.12,
    "large_dice_p90": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-checkout-commit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
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


def git_blob(repository: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
    )


def audit_static_and_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if sha256(args.wrapper_source) != args.expected_wrapper_sha256:
        raise ValueError("Wrapper SHA-256 mismatch")
    source = args.wrapper_source.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)
    ordering = [
        main_source.index('"run_rad_dino_multilayer_soft_region_probe.py"'),
        main_source.index(
            "pre_gt_audit = verify_prediction_freeze_without_gt(OUTPUT)"
        ),
        main_source.index(
            '"evaluate_rad_dino_multilayer_soft_region_probe.py"'
        ),
        main_source.index(
            "post_gt_audit = verify_post_gt_evaluation(OUTPUT)"
        ),
    ]
    if ordering != sorted(ordering) or len(set(ordering)) != 4:
        raise ValueError("Wrapper prediction/freeze/GT order differs")
    protocol = read_json(args.protocol)
    if (
        sha256(args.protocol) != args.expected_protocol_sha256
        or protocol["source"]["commit"] != args.expected_source_commit
        or protocol["weak_supervision_contract"]["consumer_trained"] is not False
        or protocol["weak_supervision_contract"]["test_evaluated"] is not False
        or protocol["runtime"]["test_locked"] is not True
    ):
        raise ValueError("Protocol provenance/supervision contract mismatch")
    for relative, expected in protocol["source"][
        "canonical_git_lf_sha256"
    ].items():
        actual = hashlib.sha256(
            git_blob(args.repository_root, args.expected_source_commit, relative)
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Canonical source binding mismatch: {relative}")
    runner = git_blob(
        args.repository_root,
        args.expected_source_commit,
        "project/run_rad_dino_multilayer_soft_region_probe.py",
    ).decode("utf-8")
    evaluator = git_blob(
        args.repository_root,
        args.expected_source_commit,
        "project/evaluate_rad_dino_multilayer_soft_region_probe.py",
    ).decode("utf-8")
    if (
        "BTXRDSegmentationDataset" in runner
        or 'split="test"' in runner
        or "split='test'" in runner
    ):
        raise ValueError("Prediction runner can access segmentation GT/test")
    if evaluator.index("def verify_prediction_freeze") > evaluator.index(
        "def evaluate_frozen_predictions"
    ) or evaluator.index("def evaluate_frozen_predictions") > evaluator.index(
        "from datasets.btxrd import BTXRDSegmentationDataset"
    ):
        raise ValueError("Evaluator GT import is not post-freeze scoped")
    return {
        "wrapper_sha256": sha256(args.wrapper_source),
        "protocol_sha256": sha256(args.protocol),
        "source_commit": args.expected_source_commit,
        "source_bindings": len(
            protocol["source"]["canonical_git_lf_sha256"]
        ),
        "wrapper_order_offsets": ordering,
    }


def audit_no_gt_artifacts(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = args.run_root
    paths = {
        "run_manifest": root / "run_manifest.json",
        "freeze": root / "prediction_freeze.json",
        "prediction_manifest": root / "predictions/prediction_manifest.csv",
        "checkpoint": root / "rad_dino_multilayer_soft_region_decoder.pt",
        "teacher": root / "teacher_metadata.json",
        "history": root / "training_history.json",
        "wrapper_pre_gt": root / "wrapper_pre_gt_audit.json",
        "wrapper_provenance": root / "wrapper_provenance.json",
        "runtime_wrapper": root / "wrapper.py",
        "runtime_protocol": root / "protocol.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required runtime artifacts missing: {missing}")
    if (
        sha256(paths["runtime_wrapper"]) != args.expected_wrapper_sha256
        or sha256(paths["runtime_protocol"]) != args.expected_protocol_sha256
        or sha256(args.split_manifest) != args.expected_split_sha256
        or sha256(args.baseline_per_image) != args.expected_baseline_sha256
    ):
        raise ValueError("Runtime wrapper/protocol/split/baseline hash mismatch")
    run = read_json(paths["run_manifest"])
    freeze = read_json(paths["freeze"])
    pre_gt = read_json(paths["wrapper_pre_gt"])
    provenance = read_json(paths["wrapper_provenance"])
    if (
        run["source_commit"] != args.expected_source_commit
        or run["protocol_sha256"] != args.expected_protocol_sha256
        or run["split_sha256"] != args.expected_split_sha256
        or run["cohort"] != EXPECTED_COHORT
        or run["selected_hidden_layers"] != [4, 8, 12]
        or run["validation_gt_read"] is not False
        or run["consumer_trained"] is not False
        or run["test_evaluated"] is not False
        or freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["validation_predictions"] != 371
        or freeze["validation_gt_read"] is not False
        or freeze["consumer_trained"] is not False
        or freeze["test_evaluated"] is not False
    ):
        raise ValueError("Run manifest or prediction-freeze contract mismatch")
    if (
        run["model_snapshot"]["model.safetensors"]["sha256"]
        != args.expected_model_weight_sha256
        or run["checkpoint_sha256"] != freeze["checkpoint_sha256"]
        or run["teacher_metadata_sha256"] != freeze["teacher_metadata_sha256"]
        or run["training_history_sha256"] != freeze["training_history_sha256"]
        or sha256(paths["checkpoint"]) != freeze["checkpoint_sha256"]
        or sha256(paths["teacher"]) != freeze["teacher_metadata_sha256"]
        or sha256(paths["history"]) != freeze["training_history_sha256"]
        or sha256(paths["prediction_manifest"])
        != freeze["prediction_manifest_sha256"]
    ):
        raise ValueError("Frozen training/prediction artifact hash mismatch")
    expected_run_sources = {
        "runner": protocol["source"]["canonical_git_lf_sha256"][
            "project/run_rad_dino_multilayer_soft_region_probe.py"
        ],
        "decoder": protocol["source"]["canonical_git_lf_sha256"][
            "project/models/rad_dino_multilayer_soft_region_decoder.py"
        ],
        "nominal_memory": protocol["source"]["canonical_git_lf_sha256"][
            "project/models/nominal_patch_memory.py"
        ],
        "mae_reconstruction": protocol["source"]["canonical_git_lf_sha256"][
            "project/models/mae_reconstruction.py"
        ],
    }
    if run["source_files"] != expected_run_sources:
        raise ValueError("Runtime scientific source hashes differ")
    history = read_json(paths["history"])
    teacher = read_json(paths["teacher"])
    if (
        len(history) != 12
        or [row["epoch"] for row in history] != list(range(1, 13))
        or teacher["normal_images"] != 1493
        or teacher["positive_images"] != 1488
        or teacher["selected_hidden_layers"] != [4, 8, 12]
        or teacher["projection_dim"] != 128
        or teacher["projection_seed"] != 42
        or teacher["soft_region_contract"]["foreground_start"] != 0.9
        or teacher["soft_region_contract"]["background_end"] != 0.5
        or teacher["validation_gt_read"] is not False
        or teacher["test_evaluated"] is not False
    ):
        raise ValueError("Training history or teacher contract mismatch")
    rows = read_csv(paths["prediction_manifest"])
    if (
        len(rows) != 371
        or len({row["image_id"] for row in rows}) != 371
        or len({row["map_path"] for row in rows}) != 371
        or sum(row["tumor"] == "1" for row in rows) != 184
        or sum(row["tumor"] == "0" for row in rows) != 187
    ):
        raise ValueError("Prediction-manifest cohort mismatch")
    prediction_dir = root / "predictions"
    expected_paths = {row["map_path"] for row in rows}
    observed_paths = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "maps").glob("*.npy")
    }
    if observed_paths != expected_paths:
        raise ValueError("Physical map set differs from prediction manifest")
    map_bytes = 0
    for row in rows:
        path = prediction_dir / row["map_path"]
        if sha256(path) != row["map_sha256"]:
            raise ValueError(f"Map hash mismatch: {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise ValueError(f"Invalid physical map: {row['image_id']}")
        map_bytes += path.stat().st_size
    if (
        pre_gt["prediction_maps"] != 371
        or pre_gt["prediction_map_bytes"] != map_bytes
        or pre_gt["prediction_manifest_sha256"]
        != freeze["prediction_manifest_sha256"]
        or pre_gt["validation_gt_read"] is not False
        or pre_gt["consumer_trained"] is not False
        or pre_gt["test_evaluated"] is not False
        or provenance["checkout_commit"] != args.expected_checkout_commit
        or provenance["scientific_source_commit"] != args.expected_source_commit
        or provenance["protocol_sha256"] != args.expected_protocol_sha256
        or provenance["wrapper_sha256"] != args.expected_wrapper_sha256
        or provenance["split_sha256"] != args.expected_split_sha256
        or provenance["baseline_per_image_sha256"]
        != args.expected_baseline_sha256
        or provenance["consumer_trained"] is not False
        or provenance["test_evaluated"] is not False
    ):
        raise ValueError("Wrapper pre-GT/provenance contract mismatch")
    return rows, {
        "prediction_maps": len(rows),
        "prediction_map_bytes": map_bytes,
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "teacher_metadata_sha256": sha256(paths["teacher"]),
        "training_history_sha256": sha256(paths["history"]),
        "prediction_manifest_sha256": sha256(paths["prediction_manifest"]),
        "prediction_freeze_sha256": sha256(paths["freeze"]),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else (
        2.0 * float(np.logical_and(prediction, target).sum()) / denominator
    )


def subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def paired_group_bootstrap(
    rows: list[tuple[str, float]], *, replicates: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for group_id, delta in rows:
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
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "n_images": len(rows),
        "n_groups": len(group_ids),
        "delta_candidate_minus_affinity": float(
            np.mean([delta for _, delta in rows])
        ),
    }


def evaluate_from_gt(
    args: argparse.Namespace, manifest: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from datasets.btxrd import BTXRDSegmentationDataset

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
    evaluated: list[dict[str, Any]] = []
    for row in manifest:
        if row["tumor"] != "1":
            continue
        values = np.load(
            args.run_root / "predictions" / row["map_path"],
            allow_pickle=False,
        ).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        item: dict[str, Any] = {
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "gt_area_ratio": float(target.mean()),
            "size_group": subgroup(float(target.mean())),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(target.reshape(-1)[int(np.argmax(flat_values))]),
            "saliency_mass_in_gt": float(values[target].sum())
            / max(float(values.sum()), 1.0e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = dice(
                values >= np.percentile(values, percentile), target
            )
        evaluated.append(item)
    counts = {
        name: sum(row["size_group"] == name for row in evaluated)
        for name in EXPECTED_SUBGROUPS
    }
    if len(evaluated) != 184 or counts != EXPECTED_SUBGROUPS:
        raise ValueError("Tumor evaluation/subgroup cohort mismatch")
    image_labels = np.asarray([int(row["tumor"]) for row in manifest])
    image_scores = np.asarray([float(row["raw_p99"]) for row in manifest])
    summary: dict[str, Any] = {
        "arm": "rad_dino_multilayer_soft_region_decoder",
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "subgroups": counts,
        "image_level_auroc_from_raw_p99": float(
            roc_auc_score(image_labels, image_scores)
        ),
        "tumor_localization": {},
        "complete_misses": {},
    }
    for name in ("overall", "small", "medium", "large"):
        selected = [
            row
            for row in evaluated
            if name == "overall" or row["size_group"] == name
        ]
        summary["tumor_localization"][name] = {
            "n": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in METRICS
            },
        }
    for percentile in (90, 95, 97, 99):
        metric = f"dice_p{percentile}"
        summary["complete_misses"][metric] = {
            name: sum(
                float(row[metric]) == 0.0
                for row in evaluated
                if name == "overall" or row["size_group"] == name
            )
            for name in ("overall", "small", "medium", "large")
        }
    baseline_rows = read_csv(args.baseline_per_image)
    baseline = {row["image_id"]: row for row in baseline_rows}
    candidate = {row["image_id"]: row for row in evaluated}
    if baseline.keys() != candidate.keys() or len(candidate) != 184:
        raise ValueError("Candidate and frozen affinity baseline differ")
    metric_results: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        metric_results[metric] = {}
        for stratum_index, stratum in enumerate(
            ("overall", "small", "medium", "large")
        ):
            names = [
                name
                for name, row in candidate.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            metric_results[metric][stratum] = paired_group_bootstrap(
                [
                    (
                        str(candidate[name]["group_id"]),
                        float(candidate[name][metric])
                        - float(baseline[name][metric]),
                    )
                    for name in names
                ],
                replicates=args.bootstrap_replicates,
                seed=20260827 + metric_index * 10 + stratum_index,
            )
    comparison = {
        "comparison": (
            "multi-layer soft-region decoder minus frozen RAD-DINO "
            "affinity-decoder v3"
        ),
        "baseline_per_image_sha256": args.expected_baseline_sha256,
        "method": "paired complete-group bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed_family": 20260827,
        "metrics": metric_results,
        "consumer_trained": False,
        "test_evaluated": False,
    }
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
    absolute = {
        name: {
            "observed": float(observed[name]),
            "minimum": float(minimum),
            "pass": bool(float(observed[name]) >= float(minimum)),
        }
        for name, minimum in GATE_THRESHOLDS.items()
    }
    dice_comparison = comparison["metrics"]["dice_p90"]
    relative = {
        "overall_dice_p90_ci95_low_above_zero": {
            "observed": float(dice_comparison["overall"]["ci95"][0]),
            "minimum_exclusive": 0.0,
            "pass": float(dice_comparison["overall"]["ci95"][0]) > 0.0,
        },
        "no_subgroup_mean_dice_p90_decrease": {
            "observed": {
                name: float(
                    dice_comparison[name]["delta_candidate_minus_affinity"]
                )
                for name in ("small", "medium", "large")
            },
            "minimum": 0.0,
            "pass": all(
                float(dice_comparison[name]["delta_candidate_minus_affinity"])
                >= 0.0
                for name in ("small", "medium", "large")
            ),
        },
    }
    passed = all(item["pass"] for item in absolute.values()) and all(
        item["pass"] for item in relative.values()
    )
    gate = {
        "gate_id": "rad_dino_multilayer_soft_region_prediction_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "all_checks_required": True,
        "absolute_checks": absolute,
        "relative_checks": relative,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    return evaluated, summary, comparison, gate


def assert_nested_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{label} dictionary schema differs")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list schema differs")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{label} differs: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{label} differs: {actual} != {expected}")


def audit_cloud_evaluation(
    args: argparse.Namespace,
    evaluated: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    root = args.run_root / "evaluation"
    cloud_rows = read_csv(root / "per_image.csv")
    if [row["image_id"] for row in cloud_rows] != [
        row["image_id"] for row in evaluated
    ]:
        raise ValueError("Cloud/local per-image order or cohort differs")
    for cloud, local in zip(cloud_rows, evaluated):
        for key in (
            "gt_area_ratio",
            "pixel_ap",
            "pixel_auroc",
            "argmax_hit",
            "saliency_mass_in_gt",
            "dice_p90",
            "dice_p95",
            "dice_p97",
            "dice_p99",
        ):
            if not np.isclose(float(cloud[key]), float(local[key]), rtol=0, atol=1e-12):
                raise ValueError(f"Cloud/local per-image metric differs: {key}")
        if cloud["group_id"] != local["group_id"] or cloud["size_group"] != local[
            "size_group"
        ]:
            raise ValueError("Cloud/local per-image metadata differs")
    cloud_summary = read_json(root / "summary.json")
    summary_core = {
        key: cloud_summary[key]
        for key in (
            "arm",
            "cohort",
            "subgroups",
            "image_level_auroc_from_raw_p99",
            "tumor_localization",
            "complete_misses",
        )
    }
    assert_nested_close(summary_core, summary, "summary")
    assert_nested_close(
        read_json(root / "paired_comparison.json"), comparison, "comparison"
    )
    cloud_gate = read_json(root / "gate_decision.json")
    gate_core = {
        key: cloud_gate[key]
        for key in (
            "gate_id",
            "status",
            "all_checks_required",
            "absolute_checks",
            "relative_checks",
            "consumer_trained",
            "test_evaluated",
        )
    }
    assert_nested_close(gate_core, gate, "gate")
    audit = read_json(root / "evaluation_audit.json")
    if (
        audit["cohort"]
        != {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            **EXPECTED_SUBGROUPS,
        }
        or audit["complete_misses_included"] is not True
        or audit["bootstrap_replicates"] != 10_000
        or audit["validation_gt_read_only_after_prediction_freeze"] is not True
        or audit["consumer_trained"] is not False
        or audit["test_evaluated"] is not False
    ):
        raise ValueError("Cloud evaluation audit contract mismatch")
    return {
        "summary_sha256": sha256(root / "summary.json"),
        "per_image_sha256": sha256(root / "per_image.csv"),
        "comparison_sha256": sha256(root / "paired_comparison.json"),
        "gate_sha256": sha256(root / "gate_decision.json"),
        "evaluation_audit_sha256": sha256(root / "evaluation_audit.json"),
        "gate_status": gate["status"],
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10_000:
        raise ValueError("Frozen audit requires 10,000 bootstrap replicates")
    static = audit_static_and_protocol(args)
    protocol = read_json(args.protocol)
    manifest, no_gt = audit_no_gt_artifacts(args, protocol)
    evaluated, summary, comparison, gate = evaluate_from_gt(args, manifest)
    cloud = audit_cloud_evaluation(
        args, evaluated, summary, comparison, gate
    )
    evidence = {
        "status": "PASS",
        "static_and_protocol": static,
        "no_gt_physical_audit": no_gt,
        "cohort": {
            "validation": 371,
            "tumor": 184,
            "normal": 187,
            **EXPECTED_SUBGROUPS,
        },
        "independent_summary": summary,
        "independent_comparison": comparison,
        "independent_gate": gate,
        "cloud_match": cloud,
        "complete_misses_included": True,
        "bootstrap_replicates": 10_000,
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
