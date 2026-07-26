from __future__ import annotations

"""Independent physical auditor for the RAD-DINO dense-MIL probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_nominal_patch_memory_arms import METRICS, paired_group_bootstrap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-probe-source-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(left: object, right: object, *, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in left:
            assert_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: list length mismatch")
        for index, (lvalue, rvalue) in enumerate(zip(left, right)):
            assert_close(lvalue, rvalue, path=f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12):
            raise AssertionError(f"{path}: {left} != {right}")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def evaluate_arm(root: Path, arm: str) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    arm_root = root / "predictions" / arm
    manifest_path = arm_root / "prediction_manifest.csv"
    manifest = load_csv(manifest_path)
    if (
        len(manifest) != 371
        or len({row["image_id"] for row in manifest}) != 371
        or sum(int(row["tumor"]) for row in manifest) != 184
    ):
        raise AssertionError(f"{arm}: prediction cohort mismatch")
    map_bytes = 0
    for row in manifest:
        path = arm_root / row["map_path"]
        if not path.is_file() or sha256(path) != row["map_sha256"]:
            raise AssertionError(f"{arm}: map hash mismatch for {row['image_id']}")
        values = np.load(path, allow_pickle=False)
        if values.shape != (320, 320) or values.dtype != np.float16 or not np.isfinite(values).all():
            raise AssertionError(f"{arm}: invalid map for {row['image_id']}")
        map_bytes += path.stat().st_size
    metadata = json.loads((arm_root / "generation_metadata.json").read_text(encoding="utf-8"))
    if metadata["validation_gt_read"] or metadata["test_evaluated"]:
        raise AssertionError(f"{arm}: prediction metadata violates lock")
    evaluation_root = arm_root / "evaluation"
    per_image_path = evaluation_root / "per_image.csv"
    rows = load_csv(per_image_path)
    if len(rows) != 184 or len({row["image_id"] for row in rows}) != 184:
        raise AssertionError(f"{arm}: tumor evaluation cohort mismatch")
    counts = {
        name: sum(row["size_group"] == name for row in rows)
        for name in ("small", "medium", "large")
    }
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise AssertionError(f"{arm}: subgroup mismatch {counts}")
    summary = json.loads((evaluation_root / "summary.json").read_text(encoding="utf-8"))
    if (
        not summary["validation_gt_read_only_after_prediction_freeze"]
        or not summary["complete_misses_included"]
        or summary["consumer_trained"]
        or summary["test_evaluated"]
    ):
        raise AssertionError(f"{arm}: evaluation contract mismatch")
    return (
        {row["image_id"]: row for row in rows},
        {
            "manifest_sha256": sha256(manifest_path),
            "generation_metadata_sha256": sha256(arm_root / "generation_metadata.json"),
            "per_image_sha256": sha256(per_image_path),
            "summary_sha256": sha256(evaluation_root / "summary.json"),
            "maps": len(manifest),
            "map_bytes": map_bytes,
            "subgroups": counts,
        },
    )


def recompute_comparison(
    single: dict[str, dict[str, str]],
    multi: dict[str, dict[str, str]],
) -> dict[str, object]:
    if single.keys() != multi.keys():
        raise AssertionError("Paired arm cohorts differ")
    metric_results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        strata: dict[str, object] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in single.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            strata[stratum] = paired_group_bootstrap(
                [
                    (
                        single[name]["group_id"],
                        float(multi[name][metric]) - float(single[name][metric]),
                    )
                    for name in names
                ],
                replicates=10_000,
                seed=20260726 + metric_index * 10 + len(stratum),
            )
        metric_results[metric] = strata
    return {
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed": 20260726,
        "interpretation": (
            "mechanism feasibility only; no arm/threshold promotion and no "
            "downstream consumer without a separate predeclared protocol"
        ),
        "metrics": metric_results,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if sha256(args.wrapper_source) != args.expected_wrapper_sha256:
        raise AssertionError("Wrapper SHA-256 mismatch")
    if sha256(args.probe_source) != args.expected_probe_source_sha256:
        raise AssertionError("Probe source SHA-256 mismatch")
    source = args.probe_source.read_text(encoding="utf-8")
    main_start = source.index("def main()")
    main_source = source[main_start:]
    positions = {
        "training": main_source.index("train_head("),
        "prediction_generation": main_source.index("write_maps("),
        "prediction_freeze": main_source.index("freeze_path.write_text("),
        "evaluation": main_source.index("evaluate_arm("),
        "comparison": main_source.index("bootstrap_compare("),
    }
    if list(positions.values()) != sorted(positions.values()):
        raise AssertionError(f"Probe execution ordering mismatch: {positions}")
    run_manifest = json.loads((args.run_root / "run_manifest.json").read_text(encoding="utf-8"))
    expected_fields = {
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "wrapper_sha256": args.expected_wrapper_sha256,
    }
    for key, expected in expected_fields.items():
        if run_manifest.get(key) != expected:
            raise AssertionError(f"Run manifest mismatch for {key}")
    if (
        run_manifest["model_hashes"]["model.safetensors"]["sha256"]
        != args.expected_model_weight_sha256
        or not run_manifest["validation_gt_read_only_after_prediction_freeze"]
        or run_manifest["consumer_trained"]
        or run_manifest["test_evaluated"]
    ):
        raise AssertionError("Run manifest model/data contract mismatch")
    checkpoint = args.run_root / "dense_mil_head.pt"
    if sha256(checkpoint) != run_manifest["checkpoint_sha256"]:
        raise AssertionError("Dense-MIL checkpoint hash mismatch")
    freeze = json.loads((args.run_root / "prediction_freeze.json").read_text(encoding="utf-8"))
    if (
        freeze["source_commit"] != args.expected_source_commit
        or freeze["protocol_sha256"] != args.expected_protocol_sha256
        or freeze["split_sha256"] != args.expected_split_sha256
        or freeze["checkpoint_sha256"] != run_manifest["checkpoint_sha256"]
        or freeze["validation_gt_read"]
        or freeze["test_evaluated"]
    ):
        raise AssertionError("Prediction freeze contract mismatch")
    single, single_evidence = evaluate_arm(args.run_root, "single_scale")
    multi, multi_evidence = evaluate_arm(args.run_root, "multiscale")
    if freeze["prediction_manifests"] != {
        "single_scale": single_evidence["manifest_sha256"],
        "multiscale": multi_evidence["manifest_sha256"],
    }:
        raise AssertionError("Prediction-freeze manifest hashes differ")
    recomputed = recompute_comparison(single, multi)
    paired_path = args.run_root / "paired_comparison.json"
    stored = json.loads(paired_path.read_text(encoding="utf-8"))
    assert_close(stored, recomputed)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_manifest_sha256": sha256(args.run_root / "run_manifest.json"),
        "wrapper_sha256": args.expected_wrapper_sha256,
        "probe_source_sha256": args.expected_probe_source_sha256,
        "source_commit": args.expected_source_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_weight_sha256": args.expected_model_weight_sha256,
        "checkpoint_sha256": run_manifest["checkpoint_sha256"],
        "prediction_freeze_sha256": sha256(args.run_root / "prediction_freeze.json"),
        "ordering_positions": positions,
        "arms": {
            "single_scale": single_evidence,
            "multiscale": multi_evidence,
        },
        "comparison": {
            "paired_comparison_sha256": sha256(paired_path),
            "recomputed": recomputed,
        },
        "complete_misses_included": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
