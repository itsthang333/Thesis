from __future__ import annotations

"""Independent physical auditor for corrected RAD-DINO square-frame maps."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("single_scale", "multiscale")
RUN_NAMES = ("dense_mil", "insight")
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
SUBGROUPS = {"small": 94, "medium": 72, "large": 18}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path):
    values = Path(path).read_bytes()
    if values.startswith(b"\xef\xbb\xbf"):
        values = values[3:]
    values = values.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(values).hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual, expected, path="root"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                "{}: keys differ: {} != {}".format(
                    path, sorted(actual), sorted(expected)
                )
            )
        for key in actual:
            assert_close(actual[key], expected[key], "{}.{}".format(path, key))
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("{}: list lengths differ".format(path))
        for index, pair in enumerate(zip(actual, expected)):
            assert_close(
                pair[0],
                pair[1],
                "{}[{}]".format(path, index),
            )
        return
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if not np.isclose(
            float(actual), float(expected), rtol=0.0, atol=1.0e-12
        ):
            raise AssertionError(
                "{}: {} != {}".format(path, actual, expected)
            )
        return
    if actual != expected:
        raise AssertionError("{}: {!r} != {!r}".format(path, actual, expected))


def paired_group_bootstrap(rows, replicates, seed):
    groups = {}
    for group_id, delta in rows:
        groups.setdefault(group_id, []).append(delta)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [
            value
            for sampled_group in sampled
            for value in groups[str(sampled_group)]
        ]
        boot[index] = np.mean(values)
    return {
        "delta_multiscale_minus_single_scale": float(
            np.mean([delta for _, delta in rows])
        ),
        "ci95": [
            float(value) for value in np.percentile(boot, [2.5, 97.5])
        ],
        "n_images": len(rows),
        "n_groups": len(group_ids),
    }


def bootstrap_compare(left_rows, right_rows):
    left = {row["image_id"]: row for row in left_rows}
    right = {row["image_id"]: row for row in right_rows}
    if left.keys() != right.keys() or len(left) != 184:
        raise AssertionError("Paired evaluation cohorts differ")
    results = {}
    for metric_index, metric in enumerate(METRICS):
        strata = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in left.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            strata[stratum] = paired_group_bootstrap(
                [
                    (
                        left[name]["group_id"],
                        float(right[name][metric]) - float(left[name][metric]),
                    )
                    for name in names
                ],
                replicates=10_000,
                seed=20260726 + metric_index * 10 + len(stratum),
            )
        results[metric] = strata
    return {
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed": 20260726,
        "interpretation": (
            "mechanism feasibility only; no arm/threshold promotion and no "
            "downstream consumer without a separate predeclared protocol"
        ),
        "metrics": results,
        "test_evaluated": False,
    }


def original_arm_root(root, arm):
    candidates = (
        Path(root) / "predictions" / arm,
        Path(root) / arm,
    )
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise AssertionError(
            "Expected one original {} arm under {}".format(arm, root)
        )
    return matches[0]


def checked_map_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise AssertionError("Map path escapes prediction directory")
    return candidate


def audit_corrected_arm(
    corrected_run,
    original_root,
    arm,
    expected_original_manifest_sha,
    expected_derived_manifest_sha,
    expected_protocol_sha,
):
    corrected_arm = Path(corrected_run) / "predictions" / arm
    original_arm = original_arm_root(original_root, arm)
    corrected_manifest_path = corrected_arm / "prediction_manifest.csv"
    original_manifest_path = original_arm / "prediction_manifest.csv"
    if sha256(original_manifest_path) != expected_original_manifest_sha:
        raise AssertionError("{} original manifest hash mismatch".format(arm))
    if sha256(corrected_manifest_path) != expected_derived_manifest_sha:
        raise AssertionError("{} corrected manifest hash mismatch".format(arm))

    corrected_rows = read_csv(corrected_manifest_path)
    original_rows = read_csv(original_manifest_path)
    corrected = {row["image_id"]: row for row in corrected_rows}
    original = {row["image_id"]: row for row in original_rows}
    if (
        len(corrected_rows) != 371
        or len(corrected) != 371
        or corrected.keys() != original.keys()
        or sum(int(row["tumor"]) for row in corrected_rows) != 184
    ):
        raise AssertionError("{} prediction cohort mismatch".format(arm))

    map_bytes = 0
    aspect_ratios = []
    for image_id, row in corrected.items():
        if row["source_map_sha256"] != original[image_id]["map_sha256"]:
            raise AssertionError(
                "{} source map mismatch for {}".format(arm, image_id)
            )
        path = checked_map_path(corrected_arm, row["map_path"])
        if not path.is_file() or sha256(path) != row["map_sha256"]:
            raise AssertionError(
                "{} corrected map hash mismatch for {}".format(arm, image_id)
            )
        values = np.load(path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise AssertionError(
                "{} invalid corrected map for {}".format(arm, image_id)
            )
        width = int(row["original_width"])
        height = int(row["original_height"])
        ratio = min(width, height) / max(width, height)
        if not np.isclose(
            ratio, float(row["content_fraction"]), rtol=0.0, atol=1.0e-12
        ):
            raise AssertionError(
                "{} content fraction mismatch for {}".format(arm, image_id)
            )
        aspect_ratios.append(ratio)
        map_bytes += path.stat().st_size

    metadata_path = corrected_arm / "generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["protocol_sha256"] != expected_protocol_sha
        or metadata["source_prediction_manifest_sha256"]
        != expected_original_manifest_sha
        or metadata["prediction_manifest_sha256"]
        != expected_derived_manifest_sha
        or metadata["cohort"] != 371
        or metadata["parameters_fitted"]
        or metadata["threshold_selected"]
        or metadata["validation_gt_read"]
        or metadata["test_evaluated"]
    ):
        raise AssertionError("{} generation metadata mismatch".format(arm))

    evaluation = corrected_arm / "evaluation"
    per_image_path = evaluation / "per_image.csv"
    evaluation_rows = read_csv(per_image_path)
    if (
        len(evaluation_rows) != 184
        or len({row["image_id"] for row in evaluation_rows}) != 184
    ):
        raise AssertionError("{} evaluation cohort mismatch".format(arm))
    subgroup_counts = {
        name: sum(row["size_group"] == name for row in evaluation_rows)
        for name in SUBGROUPS
    }
    if subgroup_counts != SUBGROUPS:
        raise AssertionError(
            "{} subgroup mismatch: {}".format(arm, subgroup_counts)
        )
    summary_path = evaluation / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary["cohort"]
        != {"validation": 371, "tumor": 184, **SUBGROUPS}
        or not summary["validation_gt_read_only_after_prediction_freeze"]
        or not summary["complete_misses_included"]
        or summary["consumer_trained"]
        or summary["test_evaluated"]
    ):
        raise AssertionError("{} evaluation summary mismatch".format(arm))

    aspect_summary = {
        "square": int(sum(np.isclose(value, 1.0) for value in aspect_ratios)),
        "below_0_90": int(sum(value < 0.90 for value in aspect_ratios)),
        "below_0_75": int(sum(value < 0.75 for value in aspect_ratios)),
        "minimum": float(min(aspect_ratios)),
        "mean": float(np.mean(aspect_ratios)),
    }
    return {
        "manifest_sha256": sha256(corrected_manifest_path),
        "generation_metadata_sha256": sha256(metadata_path),
        "maps": len(corrected_rows),
        "map_bytes": map_bytes,
        "per_image_sha256": sha256(per_image_path),
        "summary_sha256": sha256(summary_path),
        "subgroups": subgroup_counts,
        "aspect_ratio": aspect_summary,
        "evaluation_rows": evaluation_rows,
        "summary": summary,
    }


def audit_run(
    name,
    corrected_run,
    original_root,
    protocol_run,
    expected_protocol_sha,
    expected_split_sha,
):
    corrected_run = Path(corrected_run)
    original_root = Path(original_root)
    original_manifest_path = original_root / "run_manifest.json"
    original_freeze_path = original_root / "prediction_freeze.json"
    if (
        sha256(original_manifest_path) != protocol_run["run_manifest_sha256"]
        or sha256(original_freeze_path)
        != protocol_run["prediction_freeze_sha256"]
    ):
        raise AssertionError("{} original run/freeze hash mismatch".format(name))
    original_manifest = json.loads(
        original_manifest_path.read_text(encoding="utf-8")
    )
    original_freeze = json.loads(
        original_freeze_path.read_text(encoding="utf-8")
    )
    if (
        original_manifest["checkpoint_sha256"]
        != protocol_run["checkpoint_sha256"]
        or original_freeze["checkpoint_sha256"]
        != protocol_run["checkpoint_sha256"]
        or original_manifest["split_sha256"] != expected_split_sha
        or original_freeze["split_sha256"] != expected_split_sha
        or original_manifest["test_evaluated"]
        or original_freeze["validation_gt_read"]
        or original_freeze["test_evaluated"]
    ):
        raise AssertionError("{} original frozen contract mismatch".format(name))

    corrected_freeze_path = corrected_run / "prediction_freeze.json"
    corrected_freeze = json.loads(
        corrected_freeze_path.read_text(encoding="utf-8")
    )
    if (
        corrected_freeze["protocol_sha256"] != expected_protocol_sha
        or corrected_freeze["split_sha256"] != expected_split_sha
        or corrected_freeze["source_run_manifest_sha256"]
        != protocol_run["run_manifest_sha256"]
        or corrected_freeze["source_prediction_freeze_sha256"]
        != protocol_run["prediction_freeze_sha256"]
        or corrected_freeze["source_checkpoint_sha256"]
        != protocol_run["checkpoint_sha256"]
        or corrected_freeze["source_prediction_manifests"]
        != original_freeze["prediction_manifests"]
        or corrected_freeze["validation_gt_read"]
        or corrected_freeze["test_evaluated"]
    ):
        raise AssertionError("{} corrected freeze mismatch".format(name))

    arm_evidence = {}
    for arm in ARMS:
        arm_evidence[arm] = audit_corrected_arm(
            corrected_run,
            original_root,
            arm,
            protocol_run[
                "{}_manifest_sha256".format(
                    "single_scale" if arm == "single_scale" else "multiscale"
                )
            ],
            corrected_freeze["derived_prediction_manifests"][arm],
            expected_protocol_sha,
        )
        original_rows = read_csv(
            original_arm_root(original_root, arm)
            / "evaluation"
            / "per_image.csv"
        )
        recomputed_effect = bootstrap_compare(
            original_rows, arm_evidence[arm]["evaluation_rows"]
        )
        effect_path = corrected_run / "{}_geometry_effect.json".format(arm)
        stored_effect = json.loads(effect_path.read_text(encoding="utf-8"))
        assert_close(
            stored_effect,
            recomputed_effect,
            "{}.{}.geometry_effect".format(name, arm),
        )
        arm_evidence[arm]["geometry_effect_sha256"] = sha256(effect_path)
        arm_evidence[arm]["recomputed_geometry_effect"] = recomputed_effect

    if (
        arm_evidence["single_scale"]["aspect_ratio"]
        != arm_evidence["multiscale"]["aspect_ratio"]
    ):
        raise AssertionError("{} arm geometry summaries differ".format(name))
    recomputed_paired = bootstrap_compare(
        arm_evidence["single_scale"]["evaluation_rows"],
        arm_evidence["multiscale"]["evaluation_rows"],
    )
    paired_path = corrected_run / "paired_comparison.json"
    stored_paired = json.loads(paired_path.read_text(encoding="utf-8"))
    assert_close(
        stored_paired,
        recomputed_paired,
        "{}.corrected_multiscale_minus_single".format(name),
    )

    run_manifest_path = corrected_run / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run_manifest["protocol_sha256"] != expected_protocol_sha
        or run_manifest["split_sha256"] != expected_split_sha
        or run_manifest["source_checkpoint_sha256"]
        != protocol_run["checkpoint_sha256"]
        or run_manifest["source_run_manifest_sha256"]
        != protocol_run["run_manifest_sha256"]
        or run_manifest["prediction_freeze_sha256"]
        != sha256(corrected_freeze_path)
        or run_manifest["cohort"]
        != {"validation": 371, "tumor": 184, "normal": 187}
        or run_manifest["aspect_ratio"]
        != arm_evidence["single_scale"]["aspect_ratio"]
        or run_manifest["parameters_fitted"]
        or run_manifest["threshold_selected"]
        or run_manifest["consumer_trained"]
        or run_manifest["test_evaluated"]
    ):
        raise AssertionError("{} correction run manifest mismatch".format(name))
    assert_close(
        run_manifest["corrected_multiscale_minus_single"],
        recomputed_paired,
        "{}.run_manifest.corrected_multiscale_minus_single".format(name),
    )
    for arm in ARMS:
        assert_close(
            run_manifest["summaries"][arm],
            arm_evidence[arm]["summary"],
            "{}.run_manifest.summaries.{}".format(name, arm),
        )
        assert_close(
            run_manifest["geometry_effect"][arm],
            arm_evidence[arm]["recomputed_geometry_effect"],
            "{}.run_manifest.geometry_effect.{}".format(name, arm),
        )
        del arm_evidence[arm]["evaluation_rows"]
        del arm_evidence[arm]["summary"]
    return {
        "run_manifest_sha256": sha256(run_manifest_path),
        "prediction_freeze_sha256": sha256(corrected_freeze_path),
        "checkpoint_sha256": protocol_run["checkpoint_sha256"],
        "original_run_manifest_sha256": sha256(original_manifest_path),
        "original_prediction_freeze_sha256": sha256(original_freeze_path),
        "arms": arm_evidence,
        "paired_comparison_sha256": sha256(paired_path),
        "recomputed_corrected_multiscale_minus_single": recomputed_paired,
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--correction-source", type=Path, required=True)
    parser.add_argument("--dense-original-root", type=Path, required=True)
    parser.add_argument("--insight-original-root", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-correction-source-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        canonical_text_sha256(args.wrapper_source)
        != args.expected_wrapper_sha256
        or canonical_text_sha256(args.protocol)
        != args.expected_protocol_sha256
        or canonical_text_sha256(args.correction_source)
        != args.expected_correction_source_sha256
    ):
        raise AssertionError("Wrapper/protocol/correction-source hash mismatch")

    wrapper = args.wrapper_source.read_text(encoding="utf-8-sig")
    wrapper = wrapper.replace("\r\n", "\n").replace("\r", "\n")
    correction = args.correction_source.read_text(encoding="utf-8-sig")
    correction = correction.replace("\r\n", "\n").replace("\r", "\n")
    wrapper_constants = {}
    for node in ast.parse(wrapper).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                wrapper_constants[node.targets[0].id] = ast.literal_eval(
                    node.value
                )
            except (ValueError, TypeError):
                continue
    wrapper_positions = {
        "direct_mount_filter": wrapper.index(
            'if "thesis_source" in manifest_path.parts:'
        ),
        "regression_test": wrapper.index(
            "tests/test_square_probe_reprojection.py"
        ),
        "corrections": wrapper.index("results = {"),
        "run_index": wrapper.index(
            '(OUTPUT / "run_index.json").write_text('
        ),
    }
    correction_main = correction[correction.index("def main()") :]
    correction_positions = {
        "source_validation": correction_main.index("validate_original_run("),
        "derived_generation": correction_main.index("manifest_hashes = {"),
        "prediction_freeze": correction_main.index("freeze_path.write_text("),
        "gt_evaluation": correction_main.index("base.evaluate_arm("),
        "comparison": correction_main.index("base.bootstrap_compare("),
    }
    if (
        list(wrapper_positions.values())
        != sorted(wrapper_positions.values())
        or list(correction_positions.values())
        != sorted(correction_positions.values())
        or args.expected_source_commit not in wrapper
        or args.expected_protocol_sha256 not in wrapper
        or args.expected_correction_source_sha256 not in wrapper
    ):
        raise AssertionError("Wrapper/correction ordering or source lock mismatch")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        wrapper_constants["SOURCE_COMMIT"] != args.expected_source_commit
        or wrapper_constants["PROTOCOL_SHA256"]
        != args.expected_protocol_sha256
        or wrapper_constants["FROZEN_SPLIT_SHA256"]
        != args.expected_split_sha256
        or wrapper_constants["SOURCE_HASHES"] != protocol["source"]["files"]
        or protocol["source"]["files"][
            "tools/reproject_frozen_square_probe_maps.py"
        ]
        != args.expected_correction_source_sha256
        or protocol["data_contract"]["split_sha256"]
        != args.expected_split_sha256
        or protocol["data_contract"]["test_evaluated"]
    ):
        raise AssertionError("Protocol source/data contract mismatch")

    run_index_path = args.run_root / "run_index.json"
    run_index = json.loads(run_index_path.read_text(encoding="utf-8"))
    if (
        run_index["protocol_sha256"] != args.expected_protocol_sha256
        or run_index["source_commit"] != args.expected_source_commit
        or run_index["wrapper_sha256"] != args.expected_wrapper_sha256
        or run_index["consumer_trained"]
        or run_index["test_evaluated"]
        or set(run_index["results"]) != set(RUN_NAMES)
    ):
        raise AssertionError("Top-level run index mismatch")

    execution_log_path = args.run_root / "execution.log"
    execution_log = execution_log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    test_position = execution_log.index(
        "tests/test_square_probe_reprojection.py"
    )
    command_positions = [
        execution_log.index(
            "--run-root /kaggle/input/notebooks/itsthang333/"
            "btxrd-rad-dino-dense-mil-probe-v1/"
        ),
        execution_log.index(
            "--run-root /kaggle/input/notebooks/itsthang333/"
            "btxrd-rad-dino-insight-mechanism-probe-v1/"
        ),
    ]
    if (
        "3 passed" not in execution_log
        or test_position >= min(command_positions)
        or "/thesis_source/artifacts/" in execution_log
    ):
        raise AssertionError("Regression test/direct kernel-output evidence mismatch")

    original_roots = {
        "dense_mil": args.dense_original_root,
        "insight": args.insight_original_root,
    }
    protocol_runs = protocol["affected_frozen_runs"]
    protocol_keys = {
        "dense_mil": "rad_dino_dense_mil_probe_val_v1",
        "insight": "rad_dino_insight_probe_val_v1",
    }
    for name in RUN_NAMES:
        frozen = protocol_runs[protocol_keys[name]]
        wrapper_run = wrapper_constants["RUNS"][name]
        if (
            wrapper_run["checkpoint_sha256"] != frozen["checkpoint_sha256"]
            or wrapper_run["run_manifest_sha256"]
            != frozen["run_manifest_sha256"]
            or wrapper_run["prediction_freeze_sha256"]
            != frozen["prediction_freeze_sha256"]
        ):
            raise AssertionError(
                "{} wrapper/original-run source lock mismatch".format(name)
            )
    runs = {}
    for name in RUN_NAMES:
        corrected_run = args.run_root / run_index["results"][name]["path"]
        runs[name] = audit_run(
            name,
            corrected_run,
            original_roots[name],
            protocol_runs[protocol_keys[name]],
            args.expected_protocol_sha256,
            args.expected_split_sha256,
        )
        if (
            run_index["results"][name]["run_manifest_sha256"]
            != runs[name]["run_manifest_sha256"]
            or run_index["results"][name]["prediction_freeze_sha256"]
            != runs[name]["prediction_freeze_sha256"]
        ):
            raise AssertionError("{} run-index hashes mismatch".format(name))

    result = {
        "schema_version": 1,
        "status": "PASS",
        "auditor_sha256": canonical_text_sha256(Path(__file__)),
        "run_index_sha256": sha256(run_index_path),
        "execution_log_sha256": sha256(execution_log_path),
        "wrapper_sha256": args.expected_wrapper_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "source_commit": args.expected_source_commit,
        "correction_source_sha256": args.expected_correction_source_sha256,
        "split_sha256": args.expected_split_sha256,
        "wrapper_ordering_positions": wrapper_positions,
        "correction_ordering_positions": correction_positions,
        "regression_test": "3 passed before correction commands",
        "source_discovery": "direct Kaggle kernel-output mounts only",
        "runs": runs,
        "cohort": {"validation": 371, "tumor": 184, "normal": 187},
        "subgroups": SUBGROUPS,
        "bootstrap_replicates": 10_000,
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
