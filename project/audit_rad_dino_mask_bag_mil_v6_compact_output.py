from __future__ import annotations

"""Audit the direct compact output of the frozen mask-bag MIL v6 kernel.

This tool is deliberately dataset-free. It verifies only files already emitted
by the Kaggle wrapper and never opens a BTXRD image or segmentation annotation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


WRAPPER_SHA256 = "6293b040ce25109e6c4bb167a32fcc635bfbeef38e82a23c18c28ba9d8aaf1ff"
CHECKOUT_COMMIT = "689e96616ef04a692193b5e253d0c0c91450822b"
SCIENTIFIC_SOURCE_COMMIT = "1567c2a05e77fcb5f514f9094f9e12791d8dd882"
PROTOCOL_SHA256 = "a8f3101be461a1bdc007f442f60e8e3b50ccd6abf015f81f084c004829b7c4b9"
CORRECTION_HASHES = {
    "rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v1.json":
        "cbcb28c2ac2e4b1f61e18b28c01868aa16f9177d96db9b3fc5d6d1acf3867cad",
    "rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v2.json":
        "5c611c993893ae0957e99e0dde5df33d73e5199dfaad4bf80aa1591aa477f6e1",
    "rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v3.json":
        "b655fb806e69138684aa26b02a255acacf91a6b2e145e26320a8275aeffdfc30",
    "rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v4.json":
        "dab9f073db1223938ab61b1f8bc5efdc29f331ae100f2a9bce1cff4f68a5b4a3",
    "rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v5.json":
        "036d4c7adadd6551367783f90377d3424dc42a3f1b974122b646018e648abd40",
}
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_SHA256 = "f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5"
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
BASELINE_SHA256 = "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540"
RAD_DINO_HASHES = {
    "config.json": "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede",
    "preprocessor_config.json":
        "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56",
    "model.safetensors":
        "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae",
}
OPERATIONAL_GOALS = {
    "overall": 0.34024039,
    "small": 0.17895493,
    "medium": 0.51244178,
    "large": 0.49370336,
}
EXPECTED_COHORT = {
    "validation": 371,
    "tumor": 184,
    "normal": 187,
    "small": 94,
    "medium": 72,
    "large": 18,
}
EXPECTED_SPLIT_COUNTS = {
    "train": 2981,
    "train_normal": 1493,
    "train_tumor": 1488,
    "validation": 371,
    "validation_normal": 187,
    "validation_tumor": 184,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_provenance(root: Path, independent: dict[str, Any]) -> None:
    provenance = root / "provenance"
    _require(
        sha256_file(provenance / "wrapper.py") == WRAPPER_SHA256,
        "Direct wrapper hash mismatch",
    )
    protocol_candidates = [
        path
        for path in provenance.glob("*.json")
        if sha256_file(path) == PROTOCOL_SHA256
    ]
    _require(len(protocol_candidates) == 1, "Frozen parent protocol is missing/ambiguous")
    for name, expected in CORRECTION_HASHES.items():
        _require(
            sha256_file(provenance / name) == expected,
            f"Wrapper correction hash mismatch: {name}",
        )
    _require(independent["protocol_sha256"] == PROTOCOL_SHA256, "Protocol mismatch")
    for index, expected in enumerate(CORRECTION_HASHES.values(), start=1):
        _require(
            independent[f"wrapper_correction_v{index}_sha256"] == expected,
            f"Independent correction-v{index} hash mismatch",
        )


def _verify_candidate_evidence(
    root: Path,
    independent: dict[str, Any],
) -> dict[str, dict[str, str]]:
    audit_path = root / "candidate_input_audit.json"
    _require(
        sha256_file(audit_path) == independent["candidate_input_audit_sha256"],
        "Candidate-input audit hash mismatch",
    )
    audit = _json(audit_path)
    _require(
        audit.get("candidate_payloads_verified_before_training") == 3352,
        "Physical candidate verification count mismatch",
    )
    for key in ("validation_gt_read", "consumer_trained", "test_evaluated"):
        _require(audit.get(key) is False, f"Candidate evidence lock violated: {key}")

    hashes: dict[str, dict[str, str]] = {}
    expected_images = {"train": 2981, "validation": 371}
    filename_prefix = {"train": "train", "validation": "val"}
    for split, count in expected_images.items():
        entry = audit.get(split)
        _require(isinstance(entry, dict), f"Candidate audit missing split: {split}")
        _require(entry.get("images") == count, f"Candidate image count mismatch: {split}")
        _require(
            entry.get("physical_payload_hashes_verified") == count,
            f"Physical payload verification mismatch: {split}",
        )
        _require(entry.get("maximum_candidates", 82) <= 81, "Candidate cap exceeded")
        _require(
            entry.get("normal_pseudo_masks_nonempty") == 0,
            f"Non-empty normal pseudo mask: {split}",
        )
        _require(
            entry.get("ground_truth_loaded_during_generation") is False,
            f"GT generation lock violated: {split}",
        )
        prefix = filename_prefix[split]
        candidate_manifest = (
            root / "candidate_evidence" / f"{prefix}_candidate_diagnostics_manifest.csv"
        )
        pseudo_manifest = (
            root / "candidate_evidence" / f"{prefix}_pseudo_mask_manifest.csv"
        )
        run_metadata = root / "candidate_evidence" / f"{prefix}_run_metadata.json"
        _require(
            sha256_file(candidate_manifest) == entry["candidate_manifest_sha256"],
            f"Candidate manifest hash mismatch: {split}",
        )
        _require(
            sha256_file(pseudo_manifest) == entry["pseudo_manifest_sha256"],
            f"Pseudo manifest hash mismatch: {split}",
        )
        _require(
            sha256_file(run_metadata) == entry["run_metadata_sha256"],
            f"Candidate run-metadata hash mismatch: {split}",
        )
        candidate_rows = _csv(candidate_manifest)
        pseudo_rows = _csv(pseudo_manifest)
        _require(
            len(candidate_rows) == count
            and len({row["image_name"] for row in candidate_rows}) == count,
            f"Candidate compact cohort mismatch: {split}",
        )
        _require(len(pseudo_rows) == count, f"Pseudo compact cohort mismatch: {split}")
        hashes[split] = {
            "candidate_manifest_sha256": entry["candidate_manifest_sha256"],
            "pseudo_manifest_sha256": entry["pseudo_manifest_sha256"],
        }
    _require(
        independent.get("candidate_inputs") == audit,
        "Independent candidate audit payload differs from direct artifact",
    )
    return hashes


def _verify_predictions(root: Path, independent: dict[str, Any]) -> dict[str, Any]:
    probe = root / "probe"
    freeze_path = probe / "prediction_freeze.json"
    freeze = _json(freeze_path)
    _require(
        sha256_file(freeze_path) == independent["prediction_freeze_sha256"],
        "Prediction-freeze hash mismatch",
    )
    manifest_path = probe / "predictions" / "prediction_manifest.csv"
    _require(
        sha256_file(manifest_path) == independent["prediction_manifest_sha256"],
        "Prediction-manifest hash mismatch",
    )
    rows = _csv(manifest_path)
    _require(
        len(rows) == 371 and len({row["image_id"] for row in rows}) == 371,
        "Prediction cohort mismatch",
    )
    for row in rows:
        path = probe / "predictions" / row["map_path"]
        _require(path.is_file(), f"Prediction map missing: {row['image_id']}")
        _require(
            sha256_file(path) == row["map_sha256"],
            f"Prediction map hash mismatch: {row['image_id']}",
        )
    _require(
        independent.get("physical_prediction_map_hashes_verified") == 371,
        "Independent prediction-map count mismatch",
    )
    _require(freeze.get("validation_predictions") == 371, "Freeze cohort mismatch")
    for key in ("validation_gt_read", "consumer_trained", "test_evaluated"):
        _require(freeze.get(key) is False, f"Prediction freeze lock violated: {key}")
    _require(
        sha256_file(probe / "rad_dino_mask_bag_mil.pt")
        == freeze["checkpoint_sha256"],
        "Checkpoint hash mismatch",
    )
    _require(
        sha256_file(probe / "training_history.json")
        == freeze["training_history_sha256"],
        "Training-history hash mismatch",
    )
    _require(
        freeze["prediction_manifest_sha256"] == independent["prediction_manifest_sha256"],
        "Freeze/independent prediction manifest mismatch",
    )
    return freeze


def _verify_runtime(root: Path, independent: dict[str, Any]) -> None:
    _require(independent["checkout_commit"] == CHECKOUT_COMMIT, "Checkout mismatch")
    _require(
        independent["scientific_source_commit"] == SCIENTIFIC_SOURCE_COMMIT,
        "Scientific source commit mismatch",
    )
    _require(independent["split_sha256"] == SPLIT_SHA256, "Split hash mismatch")
    _require(independent["split_counts"] == EXPECTED_SPLIT_COUNTS, "Split counts mismatch")
    _require(
        independent["classifier_sha256"] == CLASSIFIER_SHA256,
        "Classifier hash mismatch",
    )
    _require(independent["sam_sha256"] == SAM_SHA256, "SAM hash mismatch")
    _require(
        independent["baseline_per_image_sha256"] == BASELINE_SHA256,
        "Baseline hash mismatch",
    )
    _require(independent["rad_dino_hashes"] == RAD_DINO_HASHES, "RAD-DINO hash mismatch")
    gpu = independent.get("gpu", {})
    _require(gpu.get("torch") == "2.5.1+cu121", "Torch runtime mismatch")
    _require(gpu.get("cuda") == "12.1", "CUDA runtime mismatch")
    _require(gpu.get("device_count") == 2, "Two physical GPUs were not proved")
    names = gpu.get("device_names", [])
    _require(len(names) == 2 and all("T4" in name for name in names), "T4x2 mismatch")
    _require(
        gpu.get("real_convolution_sums") == [324.0, 324.0],
        "Real two-device convolution proof mismatch",
    )

    run_manifest = _json(root / "probe" / "run_manifest.json")
    _require(
        run_manifest.get("source_commit") == SCIENTIFIC_SOURCE_COMMIT,
        "Probe source commit mismatch",
    )
    _require(run_manifest.get("protocol_sha256") == PROTOCOL_SHA256, "Probe protocol mismatch")
    _require(run_manifest.get("split_sha256") == SPLIT_SHA256, "Probe split mismatch")
    runtime = run_manifest.get("runtime", {})
    _require(runtime.get("cuda_device_count") == 2, "Probe CUDA count mismatch")
    _require(
        len(runtime.get("cuda_device_names", [])) == 2
        and all("T4" in name for name in runtime["cuda_device_names"]),
        "Probe T4x2 runtime mismatch",
    )
    _require(runtime.get("encoder_data_parallel") is True, "DataParallel was not active")
    for key in ("validation_gt_read", "consumer_trained", "test_evaluated"):
        _require(run_manifest.get(key) is False, f"Probe runtime lock violated: {key}")


def _verify_evaluation(root: Path, independent: dict[str, Any]) -> dict[str, Any]:
    evaluation = root / "evaluation"
    audit_path = evaluation / "evaluation_audit.json"
    gate_path = evaluation / "gate_decision.json"
    _require(
        sha256_file(audit_path) == independent["evaluation_audit_sha256"],
        "Evaluation-audit hash mismatch",
    )
    _require(
        sha256_file(gate_path) == independent["gate_decision_sha256"],
        "Gate-decision hash mismatch",
    )
    audit = _json(audit_path)
    gate = _json(gate_path)
    summary = _json(evaluation / "summary.json")
    per_image = _csv(evaluation / "per_image.csv")
    _require(audit.get("cohort") == EXPECTED_COHORT, "Evaluation cohort mismatch")
    _require(audit.get("bootstrap_replicates") == 10000, "Bootstrap count mismatch")
    _require(len(per_image) == 184, "Positive per-image cohort mismatch")
    subgroup_counts = {
        name: sum(row["size_group"] == name for row in per_image)
        for name in ("small", "medium", "large")
    }
    _require(
        subgroup_counts == {"small": 94, "medium": 72, "large": 18},
        "Per-image subgroup counts mismatch",
    )
    _require(summary.get("complete_misses_included") is True, "Misses were excluded")
    for payload in (audit, gate, summary):
        _require(payload.get("consumer_trained") is False, "Consumer lock violated")
        _require(payload.get("test_evaluated") is False, "Test lock violated")
    _require(gate.get("all_checks_required") is True, "Gate is not all-checks-required")
    _require(gate.get("status") == independent["gate_status"], "Gate status mismatch")
    _require(
        sha256_file(evaluation / "per_image.csv") == audit["per_image_sha256"],
        "Evaluated per-image hash mismatch",
    )
    _require(
        sha256_file(evaluation / "summary.json") == audit["summary_sha256"],
        "Evaluation summary hash mismatch",
    )
    _require(
        sha256_file(evaluation / "paired_comparison.json")
        == audit["paired_comparison_sha256"],
        "Paired comparison hash mismatch",
    )
    selected = {
        subgroup: float(summary["tumor_localization"][subgroup]["dice"])
        for subgroup in OPERATIONAL_GOALS
    }
    oracle = {
        subgroup: float(summary["candidate_oracle"][subgroup])
        for subgroup in OPERATIONAL_GOALS
    }
    return {
        "selected_dice": selected,
        "candidate_oracle_dice": oracle,
        "selected_operational_goal_checks": {
            subgroup: selected[subgroup] >= threshold
            for subgroup, threshold in OPERATIONAL_GOALS.items()
        },
        "oracle_operational_goal_checks": {
            subgroup: oracle[subgroup] >= threshold
            for subgroup, threshold in OPERATIONAL_GOALS.items()
        },
        "complete_misses": {
            subgroup: int(summary["tumor_localization"][subgroup]["complete_misses"])
            for subgroup in OPERATIONAL_GOALS
        },
        "prediction_gate_status": gate["status"],
    }


def audit_compact_output(root: Path) -> dict[str, Any]:
    completed_path = root / "completed.json"
    independent_path = root / "independent_audit.json"
    completed = _json(completed_path)
    independent = _json(independent_path)
    _require(completed.get("status") == "COMPLETE", "Kernel output is not COMPLETE")
    _require(
        sha256_file(independent_path) == completed["independent_audit_sha256"],
        "Independent-audit hash mismatch",
    )
    for payload in (completed, independent):
        _require(payload.get("consumer_trained") is False, "Consumer lock violated")
        _require(payload.get("test_evaluated") is False, "Test lock violated")
    _require(independent.get("complete_misses_included") is True, "Miss contract missing")
    _require(
        independent.get("validation_gt_read_only_after_prediction_freeze") is True,
        "Prediction-first validation contract missing",
    )
    _verify_provenance(root, independent)
    _verify_runtime(root, independent)
    candidate_hashes = _verify_candidate_evidence(root, independent)
    freeze = _verify_predictions(root, independent)
    evaluation = _verify_evaluation(root, independent)
    log_text = (root / "execution.log").read_text(encoding="utf-8")
    _require("206 passed, 1 skipped" in log_text, "Whole-suite evidence missing")
    return {
        "audit": "rad_dino_mask_bag_mil_v6_compact_output_v1",
        "status": "PASS",
        "compact_root": str(root.resolve()),
        "wrapper_sha256": WRAPPER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "gpu": independent["gpu"],
        "candidate_payloads_verified_before_training": 3352,
        "candidate_hashes": candidate_hashes,
        "prediction_freeze_sha256": independent["prediction_freeze_sha256"],
        "prediction_manifest_sha256": freeze["prediction_manifest_sha256"],
        "prediction_maps_verified": 371,
        "cohort": EXPECTED_COHORT,
        "bootstrap_replicates": 10000,
        **evaluation,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_compact_output(args.compact_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
