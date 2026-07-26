from __future__ import annotations

"""Independent physical auditor for the RAD-DINO INSIGHT mechanism probe."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_rad_dino_dense_mil_probe as dense_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, required=True)
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-probe-source-sha256", required=True)
    parser.add_argument("--expected-head-source-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    args = parse_args()
    expected_hashes = {
        args.wrapper_source: args.expected_wrapper_sha256,
        args.probe_source: args.expected_probe_source_sha256,
        args.head_source: args.expected_head_source_sha256,
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise AssertionError(f"Source SHA-256 mismatch: {path}")

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
        run_manifest.get("model_hashes", {})
        .get("model.safetensors", {})
        .get("sha256")
        != args.expected_model_weight_sha256
        or not run_manifest["validation_gt_read_only_after_prediction_freeze"]
        or run_manifest["consumer_trained"]
        or run_manifest["test_evaluated"]
        or run_manifest.get("head", {}).get("name")
        != "INSIGHT-style local detector + context suppression"
    ):
        raise AssertionError("Run manifest model/data/head contract mismatch")

    checkpoint = args.run_root / "insight_mil_head.pt"
    if not checkpoint.is_file() or sha256(checkpoint) != run_manifest["checkpoint_sha256"]:
        raise AssertionError("INSIGHT checkpoint hash mismatch")
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

    single, single_evidence = dense_audit.evaluate_arm(args.run_root, "single_scale")
    multi, multi_evidence = dense_audit.evaluate_arm(args.run_root, "multiscale")
    if freeze["prediction_manifests"] != {
        "single_scale": single_evidence["manifest_sha256"],
        "multiscale": multi_evidence["manifest_sha256"],
    }:
        raise AssertionError("Prediction-freeze manifest hashes differ")

    recomputed = dense_audit.recompute_comparison(single, multi)
    paired_path = args.run_root / "paired_comparison.json"
    stored = json.loads(paired_path.read_text(encoding="utf-8"))
    assert_close(stored, recomputed)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_manifest_sha256": sha256(args.run_root / "run_manifest.json"),
        "wrapper_sha256": args.expected_wrapper_sha256,
        "probe_source_sha256": args.expected_probe_source_sha256,
        "head_source_sha256": args.expected_head_source_sha256,
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
