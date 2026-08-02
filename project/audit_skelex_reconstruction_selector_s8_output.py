"""Independent GT-blind arithmetic/physical audit for S8 output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mae_reconstruction_io import sha256_file


EXPERIMENT_ID = "EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1"
ARMS = (
    "geometry_v3_plus_upstream_equal_rank",
    "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros_like(values, dtype=np.float32)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return result
    source = values[indices]
    if len(indices) == 1:
        result[indices[0]] = np.float32(1.0)
        return result
    less = (source[:, None] > source[None, :]).sum(axis=1).astype(np.float32)
    equal = (source[:, None] == source[None, :]).sum(axis=1).astype(np.float32)
    result[indices] = (less + np.float32(0.5) * (equal - np.float32(1.0))) / np.float32(len(indices) - 1)
    return result


def _lcb(
    errors: np.ndarray,
    observed: np.ndarray,
    candidates: np.ndarray,
    content: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.asarray(errors, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    candidates = np.asarray(candidates, dtype=np.float32)
    content = np.asarray(content, dtype=np.float32)
    maps, height, width = errors.shape
    dilated = np.zeros_like(candidates)
    for y in range(height):
        for x in range(width):
            y0, y1 = max(0, y - 2), min(height, y + 3)
            x0, x1 = max(0, x - 2), min(width, x + 3)
            dilated[:, y, x] = candidates[:, y0:y1, x0:x1].max(axis=(1, 2))
    inside = candidates * content[None]
    ring = np.maximum(dilated - candidates, 0.0) * content[None]
    contrasts = np.full((maps, len(candidates)), np.nan, dtype=np.float32)
    for m in range(maps):
        inside_w = inside * observed[m][None]
        ring_w = ring * observed[m][None]
        inside_mass = inside_w.reshape(len(candidates), -1).sum(axis=1)
        ring_mass = ring_w.reshape(len(candidates), -1).sum(axis=1)
        good = (inside_mass > 1.0e-8) & (ring_mass > 1.0e-8)
        err = errors[m]
        inside_sum = (inside_w * err[None]).reshape(len(candidates), -1).sum(axis=1)
        ring_sum = (ring_w * err[None]).reshape(len(candidates), -1).sum(axis=1)
        contrasts[m, good] = inside_sum[good] / inside_mass[good] - ring_sum[good] / ring_mass[good]
    count = np.isfinite(contrasts).sum(axis=0)
    safe = np.nan_to_num(contrasts, nan=0.0)
    mean = safe.sum(axis=0) / np.maximum(count, 1)
    centered = np.where(np.isfinite(contrasts), (safe - mean[None]) ** 2, 0.0)
    variance = centered.sum(axis=0) / np.maximum(count - 1, 1)
    lcb = mean - np.float32(1.96) * np.sqrt(variance / np.maximum(count, 1))
    valid = count >= 2
    lcb = np.where(valid, lcb, -np.inf).astype(np.float32)
    return lcb, valid


def _load_score_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "candidate_scores" / "candidate_score_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        score_path = root / "candidate_scores" / row["score_path"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise ValueError(f"S8 score hash mismatch: {row['image_id']}")
        with np.load(score_path, allow_pickle=False) as payload:
            result[row["image_id"]] = {
                "indices": np.asarray(payload["candidate_indices"], dtype=np.int64),
                "logits": np.asarray(payload["candidate_logits"], dtype=np.float32),
                "selected": int(np.argmax(payload["candidate_logits"])),
            }
    if len(result) != 371:
        raise ValueError("S8 score manifest cohort mismatch")
    return result


def audit_output(output_root: Path, protocol_path: Path, audit_output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("status") != "FROZEN_PRELAUNCH" or protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("S8 protocol provenance mismatch")
    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S8 prediction pair freeze contract mismatch")
    for payload in (pair, _json(output_root / "run_manifest.json"), _json(output_root / "gt_blind_diagnostics.json")):
        if payload.get("validation_gt_read") is not False or payload.get("consumer_trained") is not False or payload.get("test_evaluated") is not False:
            raise ValueError("S8 safety boundary failed")
    evidence_manifest = _json(output_root / "reconstruction_evidence" / "evidence_manifest.json")
    if evidence_manifest.get("validation_gt_read") is not False or len(evidence_manifest.get("rows", [])) != 371:
        raise ValueError("S8 evidence manifest contract mismatch")
    control_scores = _load_score_manifest(output_root / ARMS[0])
    primary_scores = _load_score_manifest(output_root / ARMS[1])
    checked = 0
    switches = 0
    for row in evidence_manifest["rows"]:
        image_id = str(row["image_id"])
        evidence_path = output_root / str(row["evidence_path"])
        if sha256_file(evidence_path) != row["evidence_sha256"]:
            raise ValueError(f"S8 evidence hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as evidence:
            candidates = np.asarray(evidence["candidate_masks"], dtype=np.float32)
            content = np.asarray(evidence["content_mask"], dtype=np.float32)
            original_lcb, original_valid = _lcb(evidence["original_errors"], evidence["original_observed"], candidates, content)
            flip_lcb, flip_valid = _lcb(evidence["aligned_flip_errors"], evidence["aligned_flip_observed"], candidates, content)
            combined_errors = np.concatenate((evidence["original_errors"], evidence["aligned_flip_errors"]), axis=0)
            combined_observed = np.concatenate((evidence["original_observed"], evidence["aligned_flip_observed"]), axis=0)
            combined_lcb, combined_valid = _lcb(combined_errors, combined_observed, candidates, content)
            base_scores = np.asarray(evidence["base_scores"], dtype=np.float32)
            base_rank = _rank(base_scores, np.ones(len(base_scores), dtype=bool))
            combined_rank = _rank(combined_lcb, combined_valid)
            fused = np.float32(0.75) * base_rank + np.float32(0.25) * combined_rank
            fused = np.where(combined_valid, fused, -np.inf).astype(np.float32)
            for observed, expected, name in (
                (original_lcb, evidence["original_lcb"], "original LCB"),
                (flip_lcb, evidence["aligned_flip_lcb"], "flip LCB"),
                (combined_lcb, evidence["combined_lcb"], "combined LCB"),
                (fused, evidence["combined_fused"], "fused score"),
            ):
                if not np.allclose(observed, expected, atol=2.0e-5, rtol=0.0, equal_nan=True):
                    raise ValueError(f"S8 {name} arithmetic mismatch: {image_id}")
            selected_index = int(np.asarray(evidence["selected_index"]))
            switches += int(np.asarray(evidence["switched"]))
            for arm_root in (output_root / ARMS[0], output_root / ARMS[1]):
                manifest = control_scores if arm_root.name == ARMS[0] else primary_scores
                if image_id not in manifest:
                    raise ValueError(f"S8 arm omits image: {image_id}")
                values = manifest[image_id]["logits"]
                if arm_root.name == ARMS[0]:
                    if not np.array_equal(values, base_scores):
                        raise ValueError(f"S8 control is not baseline-identical: {image_id}")
                elif bool(np.asarray(evidence["switched"])):
                    if not np.allclose(values, np.where(np.isfinite(fused), fused, -1.0e9), atol=2.0e-5, rtol=0.0):
                        raise ValueError(f"S8 switched primary score mismatch: {image_id}")
                else:
                    if not np.array_equal(values, base_scores):
                        raise ValueError(f"S8 fallback is not byte-identical: {image_id}")
                expected_position = selected_index if arm_root.name != ARMS[0] else int(np.argmax(base_scores))
                if int(np.argmax(values)) != expected_position:
                    raise ValueError(f"S8 selected score mismatch: {image_id}")
            checked += 1
    result = {
        "audit_id": "independent_skelex_reconstruction_selector_s8_output_v1",
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED",
        "experiment_id": EXPERIMENT_ID,
        "validation_predictions_per_arm": checked,
        "switched_predictions": switches,
        "pair_freeze_sha256": sha256_file(pair_path),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_output(args.output_root, args.protocol, args.audit_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
