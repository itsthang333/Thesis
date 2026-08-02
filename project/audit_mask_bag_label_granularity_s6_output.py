from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen S6 prediction pair."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_label_granularity import (
    LabelGranularityConfig,
    LabelGranularityResidual,
)
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import load_selector_cache_record
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL


EXPERIMENT_ID = "EXP-20260802-codex-s6-label-granularity-mil-v1"
RUN_ID = "btxrd_mask_bag_label_granularity_s6_pair_v1"
ARMS = ("coarse_control", "hierarchical_entropy_routed")
EXPECTED_SUBTYPE_COUNTS = [598, 211, 164, 74, 35, 41, 92, 237, 36]
REPRODUCTION_ATOL = 5.0e-5


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"S6 output path is absolute: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"S6 output path escapes root: {relative}")
    return resolved


def _safety(payload: Mapping[str, object], name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S6 {name} safety boundary failed")


def _smooth_pool(values: np.ndarray, temperature: float = 0.20) -> float:
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or len(logits) == 0 or not np.isfinite(logits).all():
        raise ValueError("S6 candidate logits must be finite and nonempty")
    scaled = logits / temperature
    maximum = float(scaled.max())
    return float(
        temperature
        * (
            maximum
            + math.log(float(np.exp(scaled - maximum).sum()))
            - math.log(len(logits))
        )
    )


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - logits.max()
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _entropy_route(values: np.ndarray) -> tuple[int, float, np.ndarray]:
    probabilities = _softmax(values)
    entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))
    )
    route = max(0.0, min(1.0, 1.0 - entropy / math.log(9.0)))
    return int(np.argmax(values)), route, probabilities


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _absolute_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = _average_ranks(np.asarray(first, dtype=np.float64))
    right = _average_ranks(np.asarray(second, dtype=np.float64))
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("S6 diagnostic Spearman input is constant")
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _close(actual: object, expected: object, name: str, atol: float = REPRODUCTION_ATOL) -> None:
    left = float(actual)
    right = float(expected)
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > atol:
        raise ValueError(f"S6 {name} differs: {left} versus {right}")


def _load_residual(
    path: Path,
    *,
    arm: str,
    source_commit: str,
    protocol_sha256: str,
    split_sha256: str,
    cache_freeze_sha256: str,
    baseline_checkpoint_sha256: str,
) -> tuple[LabelGranularityResidual, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, f"{arm} checkpoint")
    if (
        checkpoint.get("experiment_id") != EXPERIMENT_ID
        or checkpoint.get("arm") != arm
        or checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("split_sha256") != split_sha256
        or checkpoint.get("selector_cache_freeze_sha256") != cache_freeze_sha256
        or checkpoint.get("baseline_checkpoint_sha256")
        != baseline_checkpoint_sha256
        or checkpoint.get("training_labels") != "image_level_only"
    ):
        raise ValueError(f"S6 {arm} checkpoint provenance mismatch")
    config = LabelGranularityConfig(**checkpoint["model_config"])
    model = LabelGranularityResidual(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval(), checkpoint


def _load_baseline(
    path: Path,
    *,
    expected_sha256: str,
    source_commit: str,
    protocol_sha256: str,
    split_sha256: str,
) -> RadDinoMaskBagMIL:
    if sha256_file(path) != expected_sha256:
        raise ValueError("S6 baseline checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "baseline checkpoint")
    if (
        checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("split_sha256") != split_sha256
    ):
        raise ValueError("S6 baseline checkpoint provenance mismatch")
    model = RadDinoMaskBagMIL(MaskBagMILConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval()


@torch.inference_mode()
def _reproduce_scores(
    payload: Mapping[str, Any],
    baseline: RadDinoMaskBagMIL,
    control: LabelGranularityResidual,
    hierarchy: LabelGranularityResidual,
) -> dict[str, Any]:
    descriptors = torch.from_numpy(
        np.asarray(payload["descriptors"], dtype=np.float32)
    )[None]
    flipped = torch.from_numpy(
        np.asarray(payload["flipped_descriptors"], dtype=np.float32)
    )[None]
    valid = torch.ones(descriptors.shape[:2], dtype=torch.bool)
    base_original, _ = baseline.score_descriptors(descriptors, valid)
    base_flipped, _ = baseline.score_descriptors(flipped, valid)
    base = (0.5 * (base_original + base_flipped))[0].numpy().astype(np.float64)
    control_residual = (
        0.5 * (control(descriptors, valid) + control(flipped, valid))
    )[0].numpy().astype(np.float64)
    hierarchy_residual = (
        0.5 * (hierarchy(descriptors, valid) + hierarchy(flipped, valid))
    )[0].numpy().astype(np.float64)
    control_scores = base + control_residual.mean(axis=1)
    hierarchy_coarse = base + hierarchy_residual.mean(axis=1)
    subtype_bag = np.asarray(
        [
            _smooth_pool(base + hierarchy_residual[:, subtype])
            for subtype in range(9)
        ],
        dtype=np.float64,
    )
    predicted, route, probabilities = _entropy_route(subtype_bag)
    fine = hierarchy_residual[:, predicted]
    coarse = hierarchy_residual.mean(axis=1)
    hierarchy_scores = base + coarse + route * (fine - coarse)
    return {
        "coarse_control": control_scores.astype(np.float32),
        "hierarchical_entropy_routed": hierarchy_scores.astype(np.float32),
        "control_bag_logit": _smooth_pool(control_scores),
        "hierarchy_bag_logit": _smooth_pool(hierarchy_coarse),
        "predicted_tumor_type": predicted + 1,
        "predicted_subtype_probability": float(probabilities[predicted]),
        "entropy_route_strength": route,
    }


def audit_output(
    output_root: Path,
    protocol_path: Path,
    binding_path: Path,
    split_path: Path,
    cache_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    protocol_sha256 = sha256_file(protocol_path)
    protocol = _json(protocol_path)
    binding = _json(binding_path)
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != protocol_sha256
        or binding.get("scientific_source_commit")
        != protocol.get("scientific_source", {}).get("commit")
    ):
        raise ValueError("S6 protocol/binding mismatch")
    source_commit = str(binding["scientific_source_commit"])
    inputs = protocol.get("frozen_inputs", {})
    split_sha256 = str(inputs.get("split_sha256"))
    cache_freeze_sha256 = str(inputs.get("selector_cache_freeze_sha256"))
    cache_manifest_sha256 = str(inputs.get("selector_cache_manifest_sha256"))
    baseline_checkpoint_sha256 = str(inputs.get("baseline_checkpoint_sha256"))
    baseline_source_commit = str(inputs.get("baseline_source_commit"))
    baseline_protocol_sha256 = str(inputs.get("baseline_protocol_sha256"))
    if sha256_file(split_path) != split_sha256:
        raise ValueError("S6 split SHA-256 mismatch")
    if sha256_file(cache_root / "selector_cache_freeze.json") != cache_freeze_sha256:
        raise ValueError("S6 selector-cache freeze mismatch")
    if sha256_file(cache_root / "selector_cache_manifest.csv") != cache_manifest_sha256:
        raise ValueError("S6 selector-cache manifest mismatch")
    cache_freeze = _json(cache_root / "selector_cache_freeze.json")
    _safety(cache_freeze, "selector cache")

    label_path = output_root / "image_label_evidence.json"
    label_evidence = _json(label_path)
    _safety(label_evidence, "image-label evidence")
    if (
        label_evidence.get("train_subtype_counts_1_to_9") != EXPECTED_SUBTYPE_COUNTS
        or label_evidence.get("validation_labels_used_for_training") is not False
    ):
        raise ValueError("S6 image-label evidence mismatch")
    history_path = output_root / "training_histories.json"
    histories = _json(history_path)
    if set(histories) != set(ARMS):
        raise ValueError("S6 history arms mismatch")
    for arm in ARMS:
        history = histories[arm]
        if (
            not isinstance(history, list)
            or len(history) != 16
            or [int(row["epoch"]) for row in history] != list(range(1, 17))
            or not all(np.isfinite(list(row.values())).all() for row in history)
        ):
            raise ValueError(f"S6 {arm} training history mismatch")
    identity_path = output_root / "pretraining_identity_audit.json"
    identity = _json(identity_path)
    _safety(identity, "pretraining identity")
    for cohort, expected in (("train", 2981), ("validation", 371)):
        section = identity.get(cohort, {})
        if (
            section.get("records") != expected
            or section.get("exact_control_candidate_score_records") != expected
            or section.get("exact_hierarchy_candidate_score_records") != expected
            or float(section.get("maximum_candidate_score_delta", 1.0)) != 0.0
            or float(section.get("maximum_zero_init_entropy_route_strength", 1.0))
            > 1.0e-6
        ):
            raise ValueError(f"S6 {cohort} zero-initialization identity mismatch")

    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    _safety(pair, "prediction pair")
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("run_id") != RUN_ID
        or pair.get("source_commit") != source_commit
        or pair.get("protocol_sha256") != protocol_sha256
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("validation_subtype_label_used_for_routing") is not False
        or pair.get("diagnostics_block_prediction_freeze") is not False
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S6 prediction-pair contract mismatch")
    run_manifest = _json(output_root / "run_manifest.json")
    _safety(run_manifest, "run manifest")
    runtime = run_manifest.get("runtime", {})
    if (
        run_manifest.get("run_id") != RUN_ID
        or run_manifest.get("prediction_pair_freeze_sha256") != sha256_file(pair_path)
        or runtime.get("cuda_device_count") != 2
        or len(runtime.get("scoring_device_evidence", [])) != 2
        or {int(row["device_index"]) for row in runtime["scoring_device_evidence"]}
        != {0, 1}
        or not all("T4" in str(name) for name in runtime.get("cuda_device_names", []))
    ):
        raise ValueError("S6 runtime/T4x2 evidence mismatch")

    checkpoint_paths = {
        arm: output_root / f"{arm}_residual.pt" for arm in ARMS
    }
    control, control_checkpoint = _load_residual(
        checkpoint_paths["coarse_control"],
        arm="coarse_control",
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        split_sha256=split_sha256,
        cache_freeze_sha256=cache_freeze_sha256,
        baseline_checkpoint_sha256=baseline_checkpoint_sha256,
    )
    hierarchy, hierarchy_checkpoint = _load_residual(
        checkpoint_paths["hierarchical_entropy_routed"],
        arm="hierarchical_entropy_routed",
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        split_sha256=split_sha256,
        cache_freeze_sha256=cache_freeze_sha256,
        baseline_checkpoint_sha256=baseline_checkpoint_sha256,
    )
    if control_checkpoint["model_config"] != hierarchy_checkpoint["model_config"]:
        raise ValueError("S6 matched pair model configuration differs")
    baseline = _load_baseline(
        baseline_root / "rad_dino_mask_bag_mil.pt",
        expected_sha256=baseline_checkpoint_sha256,
        source_commit=baseline_source_commit,
        protocol_sha256=baseline_protocol_sha256,
        split_sha256=split_sha256,
    )

    val_rows = load_split_rows_without_annotations(
        split_path, expected_sha256=split_sha256, split="val"
    )
    if len(val_rows) != 371:
        raise ValueError("S6 validation cohort mismatch")
    cache_rows = {
        row["image_id"]: row
        for row in _rows(cache_root / "selector_cache_manifest.csv")
        if row["split"] == "val"
    }
    if len(cache_rows) != 371:
        raise ValueError("S6 validation cache cohort mismatch")
    diagnostic_csv_path = output_root / "gt_blind_diagnostics.csv"
    diagnostic_summary_path = output_root / "gt_blind_diagnostic_summary.json"
    diagnostic_rows = {row["image_id"]: row for row in _rows(diagnostic_csv_path)}
    diagnostic_summary = _json(diagnostic_summary_path)
    _safety(diagnostic_summary, "diagnostic summary")
    if len(diagnostic_rows) != 371 or diagnostic_summary.get(
        "diagnostics_block_prediction_freeze"
    ) is not False:
        raise ValueError("S6 diagnostic cohort/contract mismatch")

    arm_freezes: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    score_rows: dict[str, dict[str, dict[str, str]]] = {}
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arms"][arm]:
            raise ValueError(f"S6 {arm} freeze SHA-256 mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"{arm} freeze")
        if (
            freeze.get("arm") != arm
            or freeze.get("validation_predictions") != 371
            or freeze.get("checkpoint_sha256") != sha256_file(checkpoint_paths[arm])
            or freeze.get("training_histories_sha256") != sha256_file(history_path)
            or freeze.get("image_label_evidence_sha256") != sha256_file(label_path)
            or freeze.get("pretraining_identity_audit_sha256")
            != sha256_file(identity_path)
            or freeze.get("gt_blind_diagnostics_sha256")
            != sha256_file(diagnostic_csv_path)
            or freeze.get("gt_blind_diagnostic_summary_sha256")
            != sha256_file(diagnostic_summary_path)
            or freeze.get("validation_subtype_label_used_for_routing") is not False
        ):
            raise ValueError(f"S6 {arm} freeze provenance mismatch")
        arm_freezes[arm] = freeze
        prediction_rows[arm] = {
            row["image_id"]: row
            for row in _rows(
                output_root / arm / "predictions" / "prediction_manifest.csv"
            )
        }
        score_rows[arm] = {
            row["image_id"]: row
            for row in _rows(
                output_root
                / arm
                / "candidate_scores"
                / "candidate_score_manifest.csv"
            )
        }
        if len(prediction_rows[arm]) != 371 or len(score_rows[arm]) != 371:
            raise ValueError(f"S6 {arm} physical output cohort mismatch")

    maximum_logit_delta = {arm: 0.0 for arm in ARMS}
    maximum_map_delta = {arm: 0.0 for arm in ARMS}
    physical_bytes = 0
    counts: list[int] = []
    probabilities = {arm: [] for arm in ARMS}
    changed = 0
    for split_row in val_rows:
        image_id = split_row["image_id"]
        cache_row = cache_rows[image_id]
        payload = load_selector_cache_record(
            cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        reproduced = _reproduce_scores(payload, baseline, control, hierarchy)
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        masks = unpack_candidate_masks(payload["packed_masks"]).astype(np.float32)
        diagnostic = diagnostic_rows[image_id]
        if (
            int(diagnostic["tumor"]) != int(split_row["tumor"])
            or int(diagnostic["tumor_type"]) != int(split_row["tumor_type"])
            or int(diagnostic["candidate_count"]) != len(indices)
            or int(diagnostic["predicted_tumor_type"])
            != reproduced["predicted_tumor_type"]
        ):
            raise ValueError(f"S6 diagnostic identity mismatch: {image_id}")
        _close(
            diagnostic["predicted_subtype_probability"],
            reproduced["predicted_subtype_probability"],
            f"subtype probability {image_id}",
        )
        _close(
            diagnostic["entropy_route_strength"],
            reproduced["entropy_route_strength"],
            f"entropy route {image_id}",
        )
        counts.append(len(indices))
        selected_positions: dict[str, int] = {}
        for arm in ARMS:
            score_root = output_root / arm / "candidate_scores"
            score_row = score_rows[arm][image_id]
            score_path = _safe_child(score_root, score_row["score_path"])
            if sha256_file(score_path) != score_row["score_sha256"]:
                raise ValueError(f"S6 {arm} score hash mismatch: {image_id}")
            with np.load(score_path, allow_pickle=False) as score_payload:
                if set(score_payload.files) != {
                    "schema_version",
                    "candidate_indices",
                    "candidate_logits",
                } or int(score_payload["schema_version"]) != 1:
                    raise ValueError(f"S6 {arm} score schema mismatch: {image_id}")
                saved_indices = score_payload["candidate_indices"]
                saved_logits = score_payload["candidate_logits"]
            if not np.array_equal(saved_indices, indices):
                raise ValueError(f"S6 {arm} candidate order mismatch: {image_id}")
            expected_logits = np.asarray(reproduced[arm], dtype=np.float32)
            delta = float(np.max(np.abs(saved_logits - expected_logits)))
            maximum_logit_delta[arm] = max(maximum_logit_delta[arm], delta)
            if delta > REPRODUCTION_ATOL:
                raise ValueError(f"S6 {arm} score reproduction mismatch: {image_id}")
            selected = int(np.argmax(saved_logits))
            selected_positions[arm] = selected
            prediction = prediction_rows[arm][image_id]
            if (
                int(score_row["selected_candidate_index"]) != int(indices[selected])
                or int(prediction["selected_candidate_index"]) != int(indices[selected])
                or prediction["candidate_payload_sha256"]
                != cache_row["candidate_payload_sha256"]
            ):
                raise ValueError(f"S6 {arm} selected candidate mismatch: {image_id}")
            bag_logit = float(
                reproduced[
                    "control_bag_logit"
                    if arm == "coarse_control"
                    else "hierarchy_bag_logit"
                ]
            )
            bag_probability = _sigmoid(bag_logit)
            _close(prediction["bag_logit"], bag_logit, f"{arm} bag logit {image_id}")
            _close(
                prediction["bag_probability"],
                bag_probability,
                f"{arm} bag probability {image_id}",
            )
            probabilities[arm].append(float(prediction["bag_probability"]))
            map_path = _safe_child(
                output_root / arm / "predictions", prediction["map_path"]
            )
            if sha256_file(map_path) != prediction["map_sha256"]:
                raise ValueError(f"S6 {arm} map hash mismatch: {image_id}")
            saved_map = np.load(map_path, allow_pickle=False)
            expected_map = (masks[selected] * bag_probability).astype(np.float16)
            if saved_map.dtype != np.float16 or saved_map.shape != expected_map.shape:
                raise ValueError(f"S6 {arm} map schema mismatch: {image_id}")
            map_delta = float(
                np.max(
                    np.abs(saved_map.astype(np.float32) - expected_map.astype(np.float32))
                )
            )
            maximum_map_delta[arm] = max(maximum_map_delta[arm], map_delta)
            if map_delta != 0.0:
                raise ValueError(f"S6 {arm} map reproduction mismatch: {image_id}")
            physical_bytes += score_path.stat().st_size + map_path.stat().st_size
        changed += int(
            selected_positions["coarse_control"]
            != selected_positions["hierarchical_entropy_routed"]
        )

    recomputed_spearman = {
        arm: _absolute_spearman(counts, probabilities[arm]) for arm in ARMS
    }
    _close(
        diagnostic_summary["control_absolute_candidate_count_probability_spearman"],
        recomputed_spearman["coarse_control"],
        "control count/probability Spearman",
        atol=1.0e-12,
    )
    _close(
        diagnostic_summary[
            "hierarchy_absolute_candidate_count_probability_spearman"
        ],
        recomputed_spearman["hierarchical_entropy_routed"],
        "hierarchy count/probability Spearman",
        atol=1.0e-12,
    )
    if (
        diagnostic_summary.get("changed_selection_count") != changed
        or abs(
            float(diagnostic_summary.get("changed_selection_fraction"))
            - changed / 371.0
        )
        > 1.0e-12
    ):
        raise ValueError("S6 changed-selection diagnostics mismatch")
    return {
        "audit_id": "independent_mask_bag_label_granularity_s6_output_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_PASS",
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha256,
        "prediction_pair_freeze_sha256": sha256_file(pair_path),
        "validation_records": 371,
        "physical_candidate_score_payloads": 742,
        "physical_prediction_maps": 742,
        "verified_score_and_map_bytes": physical_bytes,
        "maximum_candidate_logit_reproduction_delta": maximum_logit_delta,
        "maximum_prediction_map_reproduction_delta": maximum_map_delta,
        "changed_selection_count": changed,
        "changed_selection_fraction": changed / 371.0,
        "absolute_candidate_count_probability_spearman": recomputed_spearman,
        "validation_subtype_label_used_for_routing": False,
        "diagnostics_used_for_model_selection": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_output(
        args.output_root,
        args.protocol,
        args.binding,
        args.split,
        args.cache_root,
        args.baseline_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
