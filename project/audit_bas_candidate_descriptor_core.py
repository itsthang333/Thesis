from __future__ import annotations

"""Independent GT-blind physical auditor for B1 BAS pair outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.mask_bag_selector_cache_io import load_selector_cache_record
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPERIMENT_ID = "EXP-20260801-codex-b1-bas-candidate-descriptor-v1"
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
BASELINE_CHECKPOINT_SHA256 = "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
BASELINE_PROTOCOL_SHA256 = "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
VAL_CANDIDATE_MANIFEST_SHA256 = "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
VAL_PSEUDO_MANIFEST_SHA256 = "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
PRETRAINED_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
ARMS = ("transferred_geometry_upstream", "three_way_geometry_upstream_bas")
AUDIT_ID = "independent_bas_candidate_descriptor_b1_output_v1"
AUDIT_PASS_STATUS = "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS"
REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE = True
CORRELATION_KEY = "mean_bas_upstream_rank_correlation"
CHANGE_FRACTION_KEY = "three_way_changed_selection_fraction"
MAXIMUM_CORRELATION = 0.80
MINIMUM_CHANGE_FRACTION = 0.05
EXPECTED_EXTRA_PROVENANCE: dict[str, object] = {}


def _extra_provenance(payload: dict[str, Any], name: str) -> None:
    if any(payload.get(key) != value for key, value in EXPECTED_EXTRA_PROVENANCE.items()):
        raise ValueError(f"{name} variant provenance mismatch")


def _expected_arm_scores(
    base_rank: np.ndarray,
    upstream_rank: np.ndarray,
    bas_rank: np.ndarray,
) -> dict[str, np.ndarray]:
    transferred = 0.5 * (base_rank + upstream_rank)
    three_way = (base_rank + upstream_rank + bas_rank) / 3.0
    return {
        ARMS[0]: transferred.astype(np.float32),
        ARMS[1]: three_way.astype(np.float32),
    }


def _correlation_reference(
    base_rank: np.ndarray,
    upstream_rank: np.ndarray,
) -> np.ndarray:
    del base_rank
    return upstream_rank


def _audit_extra_evidence(evidence: Any, image_id: str) -> None:
    """Variant hook for additional no-GT evidence identities."""

    del evidence, image_id


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safety(payload: dict[str, Any], name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"{name} safety boundary failed")


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank input must be one finite vector")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + 0.5 * (stop - start - 1)) / (len(values) - 1)
        start = stop
    return result


def _activation_scores(activation: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, ...]:
    activation = np.asarray(activation, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if activation.ndim != 2 or masks.ndim != 3 or not np.isfinite(activation).all():
        raise ValueError("B1 activation/mask shapes are invalid")
    lower = float(activation.min())
    normalized = (activation - lower) / max(float(activation.max()) - lower, 1.0e-10)
    resized = F.interpolate(
        torch.from_numpy(masks)[:, None],
        size=activation.shape,
        mode="area",
    )[:, 0].numpy().clip(0.0, 1.0)
    overlap = (resized * normalized[None]).sum(axis=(1, 2))
    coverage = overlap / max(float(normalized.sum()), 1.0e-8)
    purity = overlap / np.maximum(resized.sum(axis=(1, 2)), 1.0e-8)
    harmonic = 2.0 * coverage * purity / np.maximum(coverage + purity, 1.0e-8)
    return coverage, purity, harmonic, _rank(harmonic)


def _load_baseline(root: Path) -> RadDinoMaskBagMIL:
    path = root / "rad_dino_mask_bag_mil.pt"
    if sha256_file(path) != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("B1 auditor baseline checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != BASELINE_SOURCE_COMMIT
        or checkpoint.get("protocol_sha256") != BASELINE_PROTOCOL_SHA256
        or checkpoint.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("B1 auditor baseline checkpoint provenance mismatch")
    _safety(checkpoint, "baseline checkpoint")
    model = RadDinoMaskBagMIL(MaskBagMILConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval()


def _base_logits(model: RadDinoMaskBagMIL, payload: dict[str, Any]) -> np.ndarray:
    original = torch.from_numpy(np.asarray(payload["descriptors"], dtype=np.float32))[None]
    flipped = torch.from_numpy(np.asarray(payload["flipped_descriptors"], dtype=np.float32))[None]
    valid = torch.ones(original.shape[:2], dtype=torch.bool)
    with torch.inference_mode():
        first, _ = model.score_descriptors(original, valid)
        second, _ = model.score_descriptors(flipped, valid)
    return (0.5 * (first + second))[0].numpy().astype(np.float32)


def audit_b1_output(
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
        or binding.get("protocol_sha256") != protocol_sha
        or binding.get("scientific_source_commit") != protocol.get("scientific_source", {}).get("commit")
    ):
        raise ValueError("B1 protocol/binding mismatch")
    gate_path = output_root / "operational_gate.json"
    history_path = output_root / "training_history.csv"
    checkpoint_path = output_root / "bas_localizer_final_fp16.pt"
    gate = _json(gate_path)
    _safety(gate, "B1 gate")
    _extra_provenance(gate, "B1 gate")
    history = _rows(history_path)
    if len(history) != 100 or [int(row["epoch"]) for row in history] != list(range(1, 101)):
        raise ValueError("B1 training history mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "B1 checkpoint")
    _extra_provenance(checkpoint, "B1 checkpoint")
    if (
        checkpoint.get("source_commit") != binding["scientific_source_commit"]
        or checkpoint.get("protocol_sha256") != protocol_sha
        or checkpoint.get("split_sha256") != SPLIT_SHA256
        or checkpoint.get("pretrained_sha256") != PRETRAINED_SHA256
        or gate.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or gate.get("training_history_sha256") != sha256_file(history_path)
    ):
        raise ValueError("B1 checkpoint/gate provenance mismatch")
    if (
        REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE
        and gate.get("operational_gate_pass") is not True
    ):
        if (output_root / "prediction_pair_freeze.json").exists():
            raise ValueError("B1 gate-fail output must not contain predictions")
        return {
            "audit_id": AUDIT_ID,
            "status": "GT_BLIND_OPERATIONAL_GATE_FAIL",
            "operational_gate_sha256": sha256_file(gate_path),
            "validation_gt_read": False,
            "consumer_trained": False,
            "test_evaluated": False,
        }

    val_rows = load_split_rows_without_annotations(
        split_path,
        expected_sha256=SPLIT_SHA256,
        split="val",
    )
    if len(val_rows) != 371:
        raise ValueError("B1 validation split mismatch")
    cache_freeze_path = cache_root / "selector_cache_freeze.json"
    if sha256_file(cache_freeze_path) != CACHE_FREEZE_SHA256:
        raise ValueError("B1 cache freeze hash mismatch")
    cache_freeze = _json(cache_freeze_path)
    _safety(cache_freeze, "selector cache")
    cache_rows = {
        row["image_id"]: row
        for row in _rows(cache_root / "selector_cache_manifest.csv")
        if row["split"] == "val"
    }
    if len(cache_rows) != 371:
        raise ValueError("B1 validation cache cohort mismatch")
    candidate_rows, _candidate_audit = validate_candidate_diagnostics_manifest(
        candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=VAL_CANDIDATE_MANIFEST_SHA256,
        expected_pseudo_manifest_sha256=VAL_PSEUDO_MANIFEST_SHA256,
    )
    baseline_freeze_path = baseline_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != BASELINE_FREEZE_SHA256:
        raise ValueError("B1 baseline freeze mismatch")
    baseline_freeze = _json(baseline_freeze_path)
    _safety(baseline_freeze, "baseline freeze")
    baseline_rows = {
        row["image_id"]: row
        for row in _rows(baseline_root / "predictions" / "prediction_manifest.csv")
    }
    if len(baseline_rows) != 371:
        raise ValueError("B1 baseline cohort mismatch")

    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    _safety(pair, "B1 pair freeze")
    _extra_provenance(pair, "B1 pair freeze")
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("source_commit") != binding["scientific_source_commit"]
        or pair.get("protocol_sha256") != protocol_sha
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("B1 pair-freeze provenance mismatch")
    activation_rows = {
        row["image_id"]: row
        for row in _rows(output_root / "activation_evidence" / "activation_manifest.csv")
    }
    identity_rows = {row["image_id"]: row for row in _rows(output_root / "baseline_identity.csv")}
    if len(activation_rows) != 371 or len(identity_rows) != 371:
        raise ValueError("B1 evidence cohort mismatch")
    arm_freezes: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    score_rows: dict[str, dict[str, dict[str, str]]] = {}
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arms"][arm]:
            raise ValueError(f"B1 {arm} freeze hash mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"B1 {arm} freeze")
        _extra_provenance(freeze, f"B1 {arm} freeze")
        if freeze.get("arm") != arm or freeze.get("validation_predictions") != 371:
            raise ValueError(f"B1 {arm} freeze contract mismatch")
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
            raise ValueError(f"B1 {arm} output cohort mismatch")

    model = _load_baseline(baseline_root)
    correlations: list[float] = []
    changes = 0
    maximum_base_delta = 0.0
    for row in val_rows:
        image_id = row["image_id"]
        cache_row = cache_rows[image_id]
        payload = load_selector_cache_record(
            cache_root / cache_row["cache_path"],
            expected_sha256=cache_row["cache_sha256"],
            require_packed_masks=True,
        )
        masks = unpack_candidate_masks(payload["packed_masks"]).astype(np.float32)
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        evidence_row = activation_rows[image_id]
        evidence_path = output_root / "activation_evidence" / evidence_row["evidence_path"]
        if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
            raise ValueError(f"B1 evidence hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as evidence:
            activation = evidence["activation"].astype(np.float32)
            stored_base = evidence["baseline_logits"].astype(np.float32)
            candidate_path = candidate_root / candidate_rows[Path(image_id).stem]["diagnostic_path"]
            with np.load(candidate_path, allow_pickle=False) as candidate:
                upstream = candidate["selection_scores"].astype(np.float32)[indices]
            coverage, purity, harmonic, bas_rank = _activation_scores(activation, masks)
            observed_base = _base_logits(model, payload)
            maximum_base_delta = max(maximum_base_delta, float(np.max(np.abs(observed_base - stored_base))))
            if (
                not np.allclose(evidence["coverage"], coverage, atol=2e-6, rtol=0)
                or not np.allclose(evidence["purity"], purity, atol=2e-6, rtol=0)
                or not np.allclose(evidence["harmonic"], harmonic, atol=2e-6, rtol=0)
                or not np.allclose(evidence["activation_rank"], bas_rank, atol=1e-7, rtol=0)
                or not np.array_equal(evidence["candidate_indices"], indices.astype(np.int32))
            ):
                raise ValueError(f"B1 activation evidence does not reproduce: {image_id}")
            _audit_extra_evidence(evidence, image_id)
            base_rank = _rank(stored_base)
            upstream_rank = _rank(upstream)
            expected_scores = _expected_arm_scores(
                base_rank,
                upstream_rank,
                bas_rank,
            )
        if len(indices) > 1:
            correlation = float(
                np.corrcoef(
                    bas_rank,
                    _correlation_reference(base_rank, upstream_rank),
                )[0, 1]
            )
            if np.isfinite(correlation):
                correlations.append(correlation)
        changes += int(
            int(np.argmax(expected_scores[ARMS[0]]))
            != int(np.argmax(expected_scores[ARMS[1]]))
        )
        for arm, scores in expected_scores.items():
            score_row = score_rows[arm][image_id]
            score_path = output_root / arm / "candidate_scores" / score_row["score_path"]
            with np.load(score_path, allow_pickle=False) as stored:
                if not np.array_equal(stored["candidate_indices"], indices.astype(np.int32)) or not np.allclose(stored["candidate_logits"], scores, atol=1e-7, rtol=0):
                    raise ValueError(f"B1 {arm} candidate scores differ: {image_id}")
            prediction = prediction_rows[arm][image_id]
            winner = int(np.argmax(scores))
            if (
                int(prediction["selected_candidate_index"]) != int(indices[winner])
                or abs(float(prediction["bag_probability"]) - float(baseline_rows[image_id]["bag_probability"])) > 1e-12
            ):
                raise ValueError(f"B1 {arm} prediction identity mismatch: {image_id}")
            map_path = output_root / arm / "predictions" / prediction["map_path"]
            saved_map = np.load(map_path, allow_pickle=False).astype(np.float32)
            expected_map = masks[winner] * float(prediction["bag_probability"])
            if sha256_file(map_path) != prediction["map_sha256"] or not np.allclose(saved_map, expected_map, atol=5e-4, rtol=0):
                raise ValueError(f"B1 {arm} prediction map mismatch: {image_id}")

    mean_correlation = float(np.mean(correlations))
    change_fraction = changes / 371.0
    if (
        maximum_base_delta > 5e-5
        or abs(mean_correlation - float(gate[CORRELATION_KEY])) > 1e-12
        or abs(change_fraction - float(gate[CHANGE_FRACTION_KEY])) > 1e-12
        or (
            REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE
            and mean_correlation > MAXIMUM_CORRELATION
        )
        or (
            REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE
            and change_fraction < MINIMUM_CHANGE_FRACTION
        )
    ):
        raise ValueError("B1 independent operational gate does not reproduce")
    return {
        "audit_id": AUDIT_ID,
        "status": AUDIT_PASS_STATUS,
        "protocol_sha256": protocol_sha,
        "source_commit": binding["scientific_source_commit"],
        "pair_freeze_sha256": sha256_file(pair_path),
        "operational_gate_sha256": sha256_file(gate_path),
        "validation_predictions_per_arm": 371,
        "maximum_base_logit_reproduction_delta": maximum_base_delta,
        CORRELATION_KEY: mean_correlation,
        CHANGE_FRACTION_KEY: change_fraction,
        "diagnostic_gate_pass": gate.get("operational_gate_pass") is True,
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
    result = audit_b1_output(
        args.output_root.resolve(),
        args.protocol.resolve(),
        args.launch_binding.resolve(),
        args.split_manifest.resolve(),
        args.selector_cache_root.resolve(),
        args.baseline_root.resolve(),
        args.val_candidate_root.resolve(),
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
