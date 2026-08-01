from __future__ import annotations

"""Independent GT-blind physical auditor for the frozen S5 prediction pair."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_bas_candidate_descriptor_core as common
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import load_selector_cache_record
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPERIMENT_ID = "EXP-20260802-codex-s5-skelex-selector-v1"
ARMS = (
    "geometry_v3_plus_upstream_equal_rank",
    "geometry_v3_plus_upstream_plus_skelex_equal_rank",
)
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
BASELINE_CHECKPOINT_SHA256 = "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
BASELINE_PROTOCOL_SHA256 = "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
VAL_CANDIDATE_MANIFEST_SHA256 = "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
VAL_PSEUDO_MANIFEST_SHA256 = "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
SKELEX_WEIGHT_SHA256 = "81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rank32(values: np.ndarray) -> np.ndarray:
    """Reproduce the generator's torch.float32 percentile-rank arithmetic."""

    return np.asarray(common._rank(values), dtype=np.float32)


def _safety(payload: dict[str, Any], name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S5 {name} safety boundary failed")


def _load_skelex_selector(path: Path) -> RadDinoMaskBagMIL:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "selector checkpoint")
    if (
        checkpoint.get("skelex_weight_sha256") != SKELEX_WEIGHT_SHA256
        or checkpoint.get("training_labels") != "image_level_normal_tumor_only"
    ):
        raise ValueError("S5 selector checkpoint provenance mismatch")
    model = RadDinoMaskBagMIL(MaskBagMILConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval()


@torch.inference_mode()
def _skelex_logits(model: RadDinoMaskBagMIL, evidence: Any) -> np.ndarray:
    descriptors = torch.from_numpy(np.asarray(evidence["descriptors"], dtype=np.float32))[None]
    flipped = torch.from_numpy(np.asarray(evidence["flipped_descriptors"], dtype=np.float32))[None]
    valid = torch.ones(descriptors.shape[:2], dtype=torch.bool)
    first, _ = model.score_descriptors(descriptors, valid)
    second, _ = model.score_descriptors(flipped, valid)
    return (0.5 * (first + second))[0].numpy().astype(np.float32)


def audit_output(
    output_root: Path,
    protocol_path: Path,
    binding_path: Path,
    split_path: Path,
    cache_root: Path,
    baseline_root: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    protocol_sha = sha256_file(protocol_path)
    protocol = _json(protocol_path)
    binding = _json(binding_path)
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != protocol_sha
        or binding.get("scientific_source_commit")
        != protocol.get("scientific_source", {}).get("commit")
    ):
        raise ValueError("S5 protocol/binding mismatch")

    descriptor_gate_path = output_root / "descriptor_operational_gate.json"
    descriptor_gate = _json(descriptor_gate_path)
    _safety(descriptor_gate, "descriptor gate")
    for split, images in (("train", 2981), ("validation", 371)):
        section = descriptor_gate.get(split, {})
        if (
            section.get("images") != images
            or section.get("exact_candidate_set_preserved") != 1
            or section.get("positive_support_candidates")
            != section.get("exact_retained_candidates")
            or not float(section.get("minimum_fractional_grid_mass", 0.0)) > 0.0
        ):
            raise ValueError(f"S5 {split} descriptor gate mismatch")
    if descriptor_gate.get("status") != "PASS_BEFORE_SELECTOR_TRAINING":
        raise ValueError("S5 descriptor gate did not precede selector training")

    history_path = output_root / "training_history.json"
    history = _json(history_path)
    if (
        not isinstance(history, list)
        or len(history) != 16
        or [int(row["epoch"]) for row in history] != list(range(1, 17))
        or not all(np.isfinite(list(row.values())).all() for row in history)
    ):
        raise ValueError("S5 training history mismatch")
    selector_path = output_root / "skelex_mask_bag_selector.pt"
    selector = _load_skelex_selector(selector_path)
    checkpoint = torch.load(selector_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != binding["scientific_source_commit"]
        or checkpoint.get("protocol_sha256") != protocol_sha
        or checkpoint.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("S5 selector checkpoint binding mismatch")

    val_rows = load_split_rows_without_annotations(
        split_path, expected_sha256=SPLIT_SHA256, split="val"
    )
    if len(val_rows) != 371:
        raise ValueError("S5 validation split mismatch")
    cache_freeze_path = cache_root / "selector_cache_freeze.json"
    if sha256_file(cache_freeze_path) != CACHE_FREEZE_SHA256:
        raise ValueError("S5 selector cache freeze mismatch")
    cache_freeze = _json(cache_freeze_path)
    _safety(cache_freeze, "selector cache")
    cache_rows = {
        row["image_id"]: row
        for row in _rows(cache_root / "selector_cache_manifest.csv")
        if row["split"] == "val"
    }
    if len(cache_rows) != 371:
        raise ValueError("S5 validation cache cohort mismatch")
    candidate_rows, _ = validate_candidate_diagnostics_manifest(
        candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=VAL_CANDIDATE_MANIFEST_SHA256,
        expected_pseudo_manifest_sha256=VAL_PSEUDO_MANIFEST_SHA256,
    )
    baseline_freeze_path = baseline_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != BASELINE_FREEZE_SHA256:
        raise ValueError("S5 accepted baseline freeze mismatch")
    baseline_freeze = _json(baseline_freeze_path)
    _safety(baseline_freeze, "accepted baseline")
    baseline_predictions = {
        row["image_id"]: row
        for row in _rows(baseline_root / "predictions" / "prediction_manifest.csv")
    }
    if len(baseline_predictions) != 371:
        raise ValueError("S5 accepted baseline cohort mismatch")

    # Reuse only the independently audited baseline reconstruction primitives.
    common.BASELINE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256
    common.BASELINE_FREEZE_SHA256 = BASELINE_FREEZE_SHA256
    common.BASELINE_SOURCE_COMMIT = BASELINE_SOURCE_COMMIT
    common.BASELINE_PROTOCOL_SHA256 = BASELINE_PROTOCOL_SHA256
    common.SPLIT_SHA256 = SPLIT_SHA256
    baseline_model = common._load_baseline(baseline_root)

    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    _safety(pair, "prediction pair")
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("source_commit") != binding["scientific_source_commit"]
        or pair.get("protocol_sha256") != protocol_sha
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("collaborator_output_accessed") is not False
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S5 prediction-pair contract mismatch")

    diagnostics_path = output_root / "gt_blind_diagnostics.json"
    diagnostics = _json(diagnostics_path)
    _safety(diagnostics, "GT-blind diagnostics")
    identity_rows = {
        row["image_id"]: row for row in _rows(output_root / "baseline_identity.csv")
    }
    evidence_rows = {
        row["image_id"]: row
        for row in _rows(output_root / "skelex_score_evidence" / "evidence_manifest.csv")
    }
    if len(identity_rows) != 371 or len(evidence_rows) != 371:
        raise ValueError("S5 identity/evidence cohort mismatch")

    arm_freezes: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    score_rows: dict[str, dict[str, dict[str, str]]] = {}
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arms"][arm]:
            raise ValueError(f"S5 {arm} freeze hash mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"{arm} freeze")
        if (
            freeze.get("arm") != arm
            or freeze.get("validation_predictions") != 371
            or freeze.get("skelex_checkpoint_sha256") != sha256_file(selector_path)
            or freeze.get("descriptor_operational_gate_sha256")
            != sha256_file(descriptor_gate_path)
            or freeze.get("gt_blind_diagnostics_sha256") != sha256_file(diagnostics_path)
        ):
            raise ValueError(f"S5 {arm} freeze provenance mismatch")
        arm_freezes[arm] = freeze
        prediction_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "predictions" / "prediction_manifest.csv")
        }
        score_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "candidate_scores" / "candidate_score_manifest.csv")
        }
        if len(prediction_rows[arm]) != 371 or len(score_rows[arm]) != 371:
            raise ValueError(f"S5 {arm} output cohort mismatch")

    correlations: list[float] = []
    changes = 0
    maximum_base_delta = 0.0
    maximum_skelex_delta = 0.0
    physical_maps = 0
    physical_scores = 0
    for split_row in val_rows:
        image_id = split_row["image_id"]
        cache_row = cache_rows[image_id]
        payload = load_selector_cache_record(
            cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        masks = unpack_candidate_masks(payload["packed_masks"]).astype(np.float32)
        evidence_row = evidence_rows[image_id]
        evidence_path = output_root / "skelex_score_evidence" / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"S5 evidence hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as evidence:
            if not np.array_equal(evidence["candidate_indices"], indices.astype(np.int32)):
                raise ValueError(f"S5 evidence candidate indices differ: {image_id}")
            stored_base = np.asarray(evidence["baseline_logits"], dtype=np.float32)
            stored_skelex = np.asarray(evidence["skelex_logits"], dtype=np.float32)
            observed_base = common._base_logits(baseline_model, payload)
            observed_skelex = _skelex_logits(selector, evidence)
            maximum_base_delta = max(
                maximum_base_delta, float(np.max(np.abs(observed_base - stored_base)))
            )
            maximum_skelex_delta = max(
                maximum_skelex_delta, float(np.max(np.abs(observed_skelex - stored_skelex)))
            )
            candidate_row = candidate_rows[Path(image_id).stem]
            candidate_path = candidate_root / candidate_row["diagnostic_path"]
            if sha256_file(candidate_path) != cache_row["candidate_payload_sha256"]:
                raise ValueError(f"S5 candidate payload differs: {image_id}")
            with np.load(candidate_path, allow_pickle=False) as candidate:
                upstream = np.asarray(candidate["selection_scores"], dtype=np.float32)[indices]
            base_rank = _rank32(stored_base)
            upstream_rank = _rank32(upstream)
            skelex_rank = _rank32(stored_skelex)
            control = np.asarray(0.5 * (base_rank + upstream_rank), dtype=np.float32)
            primary = np.asarray(
                (base_rank + upstream_rank + skelex_rank) / 3.0, dtype=np.float32
            )
            if (
                not np.allclose(evidence["skelex_rank"], skelex_rank, atol=1e-7, rtol=0)
                or not np.allclose(evidence["control_rank"], control, atol=1e-7, rtol=0)
                or not np.allclose(evidence["primary_rank"], primary, atol=1e-7, rtol=0)
            ):
                raise ValueError(f"S5 rank evidence does not reproduce: {image_id}")
        if len(indices) > 1:
            correlation = float(np.corrcoef(control, skelex_rank)[0, 1])
            if np.isfinite(correlation):
                correlations.append(correlation)
        changes += int(int(np.argmax(control)) != int(np.argmax(primary)))
        expected = {ARMS[0]: control, ARMS[1]: primary}
        for arm, values in expected.items():
            score_row = score_rows[arm][image_id]
            score_path = output_root / arm / "candidate_scores" / score_row["score_path"]
            if sha256_file(score_path) != score_row["score_sha256"]:
                raise ValueError(f"S5 {arm} score hash mismatch: {image_id}")
            with np.load(score_path, allow_pickle=False) as stored:
                if (
                    not np.array_equal(stored["candidate_indices"], indices.astype(np.int32))
                    or not np.allclose(stored["candidate_logits"], values, atol=1e-7, rtol=0)
                ):
                    raise ValueError(f"S5 {arm} candidate scores differ: {image_id}")
            prediction = prediction_rows[arm][image_id]
            winner = int(np.argmax(values))
            if (
                int(prediction["selected_candidate_index"]) != int(indices[winner])
                or abs(
                    float(prediction["bag_probability"])
                    - float(baseline_predictions[image_id]["bag_probability"])
                )
                > 1.0e-12
            ):
                raise ValueError(f"S5 {arm} prediction identity mismatch: {image_id}")
            map_path = output_root / arm / "predictions" / prediction["map_path"]
            saved_map = np.load(map_path, allow_pickle=False).astype(np.float32)
            expected_map = masks[winner] * float(prediction["bag_probability"])
            if (
                sha256_file(map_path) != prediction["map_sha256"]
                or not np.allclose(saved_map, expected_map, atol=5.0e-4, rtol=0)
            ):
                raise ValueError(f"S5 {arm} physical map mismatch: {image_id}")
            physical_scores += 1
            physical_maps += 1
        if identity_rows[image_id].get("identity_pass") != "1":
            raise ValueError(f"S5 baseline identity failed: {image_id}")

    mean_correlation = float(np.mean(correlations))
    change_fraction = changes / 371.0
    if (
        maximum_base_delta > 5.0e-5
        or maximum_skelex_delta > 5.0e-5
        or abs(
            mean_correlation
            - float(diagnostics["mean_skelex_control_rank_correlation"])
        )
        > 1.0e-12
        or abs(
            change_fraction
            - float(diagnostics["primary_changed_selection_fraction"])
        )
        > 1.0e-12
    ):
        raise ValueError("S5 independent GT-blind diagnostics do not reproduce")
    return {
        "audit_id": "independent_skelex_mask_bag_selector_s5_output_v1",
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED",
        "protocol_sha256": protocol_sha,
        "source_commit": binding["scientific_source_commit"],
        "pair_freeze_sha256": sha256_file(pair_path),
        "descriptor_operational_gate_sha256": sha256_file(descriptor_gate_path),
        "selector_checkpoint_sha256": sha256_file(selector_path),
        "validation_predictions_per_arm": 371,
        "physical_prediction_maps_verified": physical_maps,
        "physical_candidate_scores_verified": physical_scores,
        "physical_descriptor_evidence_verified": len(evidence_rows),
        "maximum_base_logit_reproduction_delta": maximum_base_delta,
        "maximum_skelex_logit_reproduction_delta": maximum_skelex_delta,
        "mean_skelex_control_rank_correlation": mean_correlation,
        "primary_changed_selection_fraction": change_fraction,
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch-binding", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_output(
        args.output_root.resolve(),
        args.protocol.resolve(),
        args.launch_binding.resolve(),
        args.split_manifest.resolve(),
        args.selector_cache_root.resolve(),
        args.baseline_root.resolve(),
        args.val_candidate_root.resolve(),
    )
    args.audit_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
