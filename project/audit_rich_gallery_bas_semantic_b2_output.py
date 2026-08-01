"""Independent no-GT physical auditor for rich-gallery BAS B2 output."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from audit_rich_gallery_stage_a_transport import (
    EXPECTED_BASELINE_VARIANT,
    audit as audit_transport,
    audit_g1_baseline_row,
    find_forbidden_transport_paths,
    load_npz_mapping,
    safe_transport_path,
)
from models.rich_gallery_bas_residual import (
    average_percentile_rank,
    bas_candidate_scores,
    canonical_source,
    score_rich_gallery_bas_pair,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from pseudo.manifest import sha256_file


EXPERIMENT_ID = "EXP-20260801-codex-b2-rich-gallery-bas-residual-v1"
CONTROL_ARM = "g1_upstream_control"
SEMANTIC_ARM = "g1_upstream_bas_semantic"
EXPECTED_PRETRAINED_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-pair-freeze-sha256", required=True)
    parser.add_argument("--expected-b2-source-commit", required=True)
    parser.add_argument("--expected-b2-protocol-sha256", required=True)
    parser.add_argument("--expected-b2-runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _indexed(rows: list[dict[str, str]], *, name: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row.get("image_id", "")
        if not image_id or image_id in result:
            raise ValueError(f"{name} contains duplicate/empty image id: {image_id!r}")
        result[image_id] = row
    if len(result) != 371:
        raise ValueError(f"{name} must contain 371 images")
    return result


def unpack_prediction_payload(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"packed_mask", "shape"}:
            raise ValueError("B2 prediction payload schema mismatch")
        dimensions = tuple(int(value) for value in payload["shape"].reshape(-1))
        if len(dimensions) != 2 or min(dimensions) <= 0:
            raise ValueError("B2 prediction shape is invalid")
        bits = np.unpackbits(
            np.asarray(payload["packed_mask"], dtype=np.uint8),
            bitorder="little",
        )
    required = int(np.prod(dimensions))
    if len(bits) < required:
        raise ValueError("B2 packed prediction is truncated")
    return bits[:required].reshape(dimensions).astype(bool)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_percentile_rank(left)
    right_rank = average_percentile_rank(right)
    if float(np.std(left_rank)) <= 1.0e-12 or float(np.std(right_rank)) <= 1.0e-12:
        return 1.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("B2 independent rank correlation is non-finite")
    return value


def _require_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
    atol: float = 2.0e-6,
) -> None:
    if np.asarray(actual).shape != np.asarray(expected).shape or not np.allclose(
        actual,
        expected,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(f"B2 independent evidence mismatch: {name}")


def _verify_no_runner_import(auditor_path: Path) -> None:
    tree = ast.parse(auditor_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        if any("run_rich_gallery_bas_semantic_b2" in module for module in modules):
            raise ValueError("B2 auditor must not import the scientific producer")


def _verify_arm(
    prediction_root: Path,
    pair: Mapping[str, Any],
    arm: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    arm_root = prediction_root / arm
    freeze_path = arm_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != pair["arm_freezes"][arm]:
        raise ValueError(f"B2 {arm} freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("experiment_id") != EXPERIMENT_ID
        or freeze.get("arm") != arm
        or freeze.get("validation_predictions") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError(f"B2 {arm} freeze safety contract mismatch")
    manifest_path = arm_root / "prediction_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("prediction_manifest_sha256"):
        raise ValueError(f"B2 {arm} prediction manifest changed")
    return _indexed(_load_csv(manifest_path), name=f"B2 {arm} manifest"), freeze


def audit(args: argparse.Namespace) -> dict[str, Any]:
    _verify_no_runner_import(Path(__file__))
    forbidden = find_forbidden_transport_paths(args.prediction_root)
    if forbidden:
        raise ValueError(f"GT/Stage-B paths present in B2 output: {forbidden[:5]}")
    runner_path = Path(__file__).with_name("run_rich_gallery_bas_semantic_b2.py")
    if sha256_file(runner_path) != args.expected_b2_runner_sha256:
        raise ValueError("B2 scientific producer source hash mismatch")

    transport_result = audit_transport(args)
    transport_audit_path = args.prediction_root / "transport_audit.json"
    stored_transport = json.loads(transport_audit_path.read_text(encoding="utf-8"))
    if stored_transport != transport_result or stored_transport.get("audit_pass") is not True:
        raise ValueError("B2 stored transport audit differs from independent audit")

    pair_path = args.prediction_root / "prediction_pair_freeze.json"
    if sha256_file(pair_path) != args.expected_pair_freeze_sha256:
        raise ValueError("B2 pair-freeze SHA-256 mismatch")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("source_commit") != args.expected_b2_source_commit
        or pair.get("protocol_sha256") != args.expected_b2_protocol_sha256
        or pair.get("split_sha256") != args.expected_split_sha256
        or pair.get("transport_prediction_freeze_sha256")
        != args.expected_freeze_sha256
        or pair.get("transport_audit_sha256") != sha256_file(transport_audit_path)
        or pair.get("g1_checkpoint_sha256") != args.expected_g1_checkpoint_sha256
        or set(pair.get("arm_freezes", {})) != {CONTROL_ARM, SEMANTIC_ARM}
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("training_labels") != "image_level_normal_tumor_only"
        or pair.get("validation_gt_read") is not False
        or pair.get("consumer_trained") is not False
        or pair.get("test_images_read") != 0
        or pair.get("test_evaluated") is not False
    ):
        raise ValueError("B2 pair-freeze provenance/safety contract mismatch")

    checkpoint_path = args.prediction_root / "bas_localizer_final_fp16.pt"
    if sha256_file(checkpoint_path) != pair.get("bas_checkpoint_sha256"):
        raise ValueError("B2 BAS checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != args.expected_b2_source_commit
        or checkpoint.get("protocol_sha256") != args.expected_b2_protocol_sha256
        or checkpoint.get("split_sha256") != args.expected_split_sha256
        or checkpoint.get("pretrained_sha256") != EXPECTED_PRETRAINED_SHA256
        or checkpoint.get("input_size") != 448
        or checkpoint.get("epochs") != 100
        or checkpoint.get("training_labels") != "image_level_normal_tumor_only"
        or checkpoint.get("validation_gt_read") is not False
        or checkpoint.get("consumer_trained") is not False
        or checkpoint.get("test_images_read") != 0
        or checkpoint.get("test_evaluated") is not False
    ):
        raise ValueError("B2 BAS checkpoint provenance/safety mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state or any(
        str(name).startswith("background_") for name in state
    ):
        raise ValueError("B2 BAS active checkpoint state is invalid")

    history_path = args.prediction_root / "training_history.csv"
    if sha256_file(history_path) != pair.get("training_history_sha256"):
        raise ValueError("B2 training-history hash mismatch")
    history = _load_csv(history_path)
    if len(history) != 100 or [int(row["epoch"]) for row in history] != list(
        range(1, 101)
    ):
        raise ValueError("B2 fixed 100-epoch history is incomplete")
    gate_path = args.prediction_root / "operational_gate.json"
    if sha256_file(gate_path) != pair.get("operational_gate_sha256"):
        raise ValueError("B2 operational-gate hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        not isinstance(gate.get("classification_gate_pass"), bool)
        or not isinstance(gate.get("complementarity_gate_pass"), bool)
        or gate.get("descriptor_diagnostic_gate_pass")
        != bool(
            gate.get("classification_gate_pass")
            and gate.get("complementarity_gate_pass")
        )
        or gate.get("operational_gate_pass") is not True
        or gate.get("spatial_evaluation_authorized_after_pair_freeze") is not True
        or gate.get("descriptor_diagnostics_do_not_block_spatial_evaluation")
        is not True
        or gate.get("consumer_authorized") is not False
        or gate.get("validation_gt_read") is not False
        or gate.get("consumer_trained") is not False
        or gate.get("test_images_read") != 0
        or gate.get("test_evaluated") is not False
    ):
        raise ValueError("B2 operational gate did not pass safely")

    run_path = args.prediction_root / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("experiment_id") != EXPERIMENT_ID
        or run.get("source_commit") != args.expected_b2_source_commit
        or run.get("protocol_sha256") != args.expected_b2_protocol_sha256
        or run.get("pair_freeze_sha256") != args.expected_pair_freeze_sha256
        or run.get("runtime", {}).get("cuda_device_count") != 2
        or not all("T4" in name for name in run.get("runtime", {}).get("cuda_device_names", []))
        or run.get("cohort") != {"train": 2981, "validation": 371}
        or run.get("validation_gt_read") is not False
        or run.get("consumer_trained") is not False
        or run.get("test_images_read") != 0
        or run.get("test_evaluated") is not False
    ):
        raise ValueError("B2 run manifest provenance/safety mismatch")

    control_rows, _control_freeze = _verify_arm(args.prediction_root, pair, CONTROL_ARM)
    semantic_rows, _semantic_freeze = _verify_arm(
        args.prediction_root, pair, SEMANTIC_ARM
    )
    score_manifest_path = args.prediction_root / "candidate_score_manifest.csv"
    activation_manifest_path = args.prediction_root / "activation_manifest.csv"
    if (
        sha256_file(score_manifest_path) != pair.get("candidate_score_manifest_sha256")
        or sha256_file(activation_manifest_path)
        != pair.get("activation_manifest_sha256")
    ):
        raise ValueError("B2 evidence manifest hash mismatch")
    score_rows = _indexed(_load_csv(score_manifest_path), name="B2 score manifest")
    activation_rows = _indexed(
        _load_csv(activation_manifest_path), name="B2 activation manifest"
    )

    transport_rows = [
        row
        for row in _load_csv(args.transport_root / "stage_a_selection_manifest.csv")
        if row.get("variant") == EXPECTED_BASELINE_VARIANT
    ]
    transport_index = _indexed(transport_rows, name="transport G1 baseline")
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=transport_index,
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )

    g1_correlations: list[float] = []
    upstream_correlations: list[float] = []
    positive_changed = 0
    physical_predictions = 0
    for image_id, transport_row in transport_index.items():
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        stage_path = safe_transport_path(args.transport_root, transport_row["score_path"])
        candidate_payload = load_npz_mapping(candidate_path)
        stage_payload = load_npz_mapping(stage_path)
        aligned = audit_g1_baseline_row(
            transport_row,
            candidate_payload,
            stage_payload,
        )

        activation_row = activation_rows[image_id]
        activation_path = safe_transport_path(
            args.prediction_root, activation_row["activation_path"]
        )
        if sha256_file(activation_path) != activation_row["activation_sha256"]:
            raise ValueError(f"B2 activation evidence changed: {image_id}")
        activation = np.load(activation_path, allow_pickle=False)
        if activation.ndim != 2 or not np.isfinite(activation).all():
            raise ValueError(f"B2 activation evidence is invalid: {image_id}")
        coverage, purity, harmonic = bas_candidate_scores(
            activation,
            aligned.candidate_masks,
        )
        pair_scores = score_rich_gallery_bas_pair(
            aligned.g1_logits,
            aligned.upstream_scores,
            harmonic,
        )

        score_row = score_rows[image_id]
        score_path = safe_transport_path(args.prediction_root, score_row["score_path"])
        if (
            sha256_file(score_path) != score_row["score_sha256"]
            or score_row["candidate_payload_sha256"]
            != transport_row["candidate_payload_sha256"]
            or int(score_row["candidate_count"]) != len(aligned.candidate_indices)
        ):
            raise ValueError(f"B2 candidate score evidence changed: {image_id}")
        stored = load_npz_mapping(score_path)
        expected_keys = {
            "candidate_indices",
            "source_ids",
            "g1_logits",
            "upstream_scores",
            "coverage",
            "purity",
            "bas_scores",
            "baseline_rank",
            "semantic_rank",
        }
        if set(stored) != expected_keys:
            raise ValueError(f"B2 score evidence schema mismatch: {image_id}")
        if not np.array_equal(stored["candidate_indices"], aligned.candidate_indices):
            raise ValueError(f"B2 candidate indices changed: {image_id}")
        if not np.array_equal(stored["source_ids"], aligned.source_ids):
            raise ValueError(f"B2 source ids changed: {image_id}")
        _require_close(stored["g1_logits"], aligned.g1_logits, name="g1 logits")
        _require_close(
            stored["upstream_scores"], aligned.upstream_scores, name="upstream"
        )
        _require_close(stored["coverage"], coverage, name="coverage")
        _require_close(stored["purity"], purity, name="purity")
        _require_close(stored["bas_scores"], harmonic, name="BAS harmonic")
        _require_close(
            stored["baseline_rank"], pair_scores.baseline_rank, name="baseline rank"
        )
        _require_close(
            stored["semantic_rank"], pair_scores.bas_residual_rank, name="semantic rank"
        )

        source_names = np.asarray(candidate_payload["proposal_source_ids"]).reshape(-1)
        for arm, row, local_index in (
            (CONTROL_ARM, control_rows[image_id], pair_scores.baseline_local_index),
            (SEMANTIC_ARM, semantic_rows[image_id], pair_scores.bas_residual_local_index),
        ):
            original_index = int(aligned.candidate_indices[local_index])
            if (
                int(row["selected_local_index"]) != local_index
                or int(row["selected_candidate_index"]) != original_index
                or row["selected_source"] != canonical_source(source_names[original_index])
                or row["candidate_payload_sha256"]
                != transport_row["candidate_payload_sha256"]
            ):
                raise ValueError(f"B2 frozen choice mismatch: {arm}/{image_id}")
            arm_root = args.prediction_root / arm
            prediction_path = safe_transport_path(arm_root, row["prediction_path"])
            if sha256_file(prediction_path) != row["prediction_sha256"]:
                raise ValueError(f"B2 prediction hash mismatch: {arm}/{image_id}")
            physical = unpack_prediction_payload(prediction_path)
            if not np.array_equal(physical, aligned.candidate_masks[local_index]):
                raise ValueError(f"B2 physical prediction mismatch: {arm}/{image_id}")
            physical_predictions += 1

        if int(transport_row["tumor"]) == 1:
            g1_correlations.append(_rank_correlation(harmonic, aligned.g1_logits))
            upstream_correlations.append(
                _rank_correlation(harmonic, aligned.upstream_scores)
            )
            positive_changed += int(
                pair_scores.baseline_local_index != pair_scores.bas_residual_local_index
            )

    if len(g1_correlations) != 184 or physical_predictions != 742:
        raise RuntimeError("B2 independent physical audit cohort mismatch")
    independent = {
        "mean_bas_g1_rank_correlation_positive_bags": float(np.mean(g1_correlations)),
        "mean_bas_upstream_rank_correlation_positive_bags": float(
            np.mean(upstream_correlations)
        ),
        "correlation_positive_bags": 184,
        "semantic_changed_positive_selections": positive_changed,
        "semantic_changed_positive_selection_fraction": positive_changed / 184.0,
    }
    for key, value in independent.items():
        if isinstance(value, float):
            if abs(float(gate[key]) - value) > 1.0e-10:
                raise ValueError(f"B2 independent gate diagnostic mismatch: {key}")
        elif int(gate[key]) != value:
            raise ValueError(f"B2 independent gate diagnostic mismatch: {key}")

    return {
        "stage": "rich_gallery_bas_semantic_b2_independent_no_gt_audit_v1",
        "audit_pass": True,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": args.expected_b2_source_commit,
        "protocol_sha256": args.expected_b2_protocol_sha256,
        "pair_freeze_sha256": args.expected_pair_freeze_sha256,
        "transport_freeze_sha256": args.expected_freeze_sha256,
        "transport_audit_sha256": sha256_file(transport_audit_path),
        "candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "candidate_summary_sha256": candidate_audit["summary_sha256"],
        "bas_checkpoint_sha256": sha256_file(checkpoint_path),
        "run_manifest_sha256": sha256_file(run_path),
        "operational_gate_sha256": sha256_file(gate_path),
        "physical_predictions_reproduced": 742,
        "candidate_score_payloads_reproduced": 371,
        "activation_payloads_verified": 371,
        "control_choices_reproduced_from_trusted_transport": 371,
        "semantic_choices_reproduced": 371,
        "independent_complementarity": independent,
        "descriptor_diagnostic_gate_pass": gate["descriptor_diagnostic_gate_pass"],
        "spatial_evaluation_authorized_after_pair_freeze": True,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("B2 independent audit output already exists")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "audit_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
