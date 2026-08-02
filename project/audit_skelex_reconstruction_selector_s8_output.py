"""Independent GT-blind arithmetic/physical audit for S8 output."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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


def _rank_serialized_lcb(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reproduce producer tie ranks from its serialized float32 LCB payload."""

    return _rank(np.asarray(values, dtype=np.float32), np.asarray(valid, dtype=bool))


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


def _unpack_masks(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    count = packed.shape[0]
    values = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, count=height * width)
    return values.reshape(count, height, width).astype(np.float32)


def _project_grid(
    masks: np.ndarray,
    *,
    padded_side: int,
    content_box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = (int(value) for value in content_box)
    coordinates = (torch.arange(14, dtype=torch.float32) + 0.5) * (float(padded_side) / 14.0)
    source_x = (coordinates - float(x0)) / float(x1 - x0)
    source_y = (coordinates - float(y0)) / float(y1 - y0)
    grid_y, grid_x = torch.meshgrid(source_y, source_x, indexing="ij")
    grid = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)

    def sample(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[:, None]
        expanded = grid[None].expand(tensor.shape[0], -1, -1, -1)
        return F.grid_sample(
            tensor,
            expanded,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)

    projected = sample(masks)
    content = sample(np.ones((1, masks.shape[-2], masks.shape[-1]), dtype=np.float32))[0]
    return projected.numpy(), content.numpy()


@lru_cache(maxsize=8)
def _full_permutation_orders(
    maps: int,
    cells: int,
    permutations: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.stack(
        [
            torch.stack(
                [torch.randperm(cells, generator=generator) for _ in range(permutations)],
                dim=0,
            )
            for _ in range(maps)
        ],
        dim=0,
    )


def _null_improvements(
    errors: np.ndarray,
    observed: np.ndarray,
    candidates: np.ndarray,
    content: np.ndarray,
    base_scores: np.ndarray,
    accepted_index: int,
) -> np.ndarray:
    # The frozen producer intentionally moves reconstruction errors, candidate
    # grids and base logits to CPU before select_with_spatial_null. Replaying
    # the rank-discontinuous null statistic on CUDA can split an exact CPU tie.
    # Keep the independent arithmetic on the producer's selector device.
    device = torch.device("cpu")
    errors_t = torch.from_numpy(np.asarray(errors, dtype=np.float32)).to(device)
    observed_t = torch.from_numpy(np.asarray(observed, dtype=bool))
    candidate_t = torch.from_numpy(np.asarray(candidates, dtype=np.float32)).to(device)
    content_cpu = torch.from_numpy(np.asarray(content, dtype=np.float32))
    maps, height, width = observed_t.shape
    cells = height * width
    valid_cells = (observed_t & (content_cpu[None] > 0.0)).reshape(maps, cells)
    orders = _full_permutation_orders(maps, cells, 255, 20261203)
    source_bank = torch.full((maps, 255, cells), -1, dtype=torch.long)
    for map_index in range(maps):
        indices = torch.nonzero(valid_cells[map_index], as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        for permutation in range(255):
            order = orders[map_index, permutation]
            source_bank[map_index, permutation, : indices.numel()] = order[
                valid_cells[map_index][order]
            ]
    flat_errors = errors_t.reshape(maps, cells)
    permuted = torch.zeros((255, maps, cells), dtype=torch.float32, device=device)
    for map_index in range(maps):
        targets = torch.nonzero(valid_cells[map_index], as_tuple=False).flatten()
        if targets.numel() < 2:
            continue
        source = source_bank[map_index, :, : targets.numel()].to(device)
        permuted[:, map_index, targets.to(device)] = flat_errors[map_index][source]

    content_t = content_cpu.to(device)
    inside = candidate_t * content_t[None]
    dilated = F.max_pool2d(candidate_t[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
    ring = (dilated - candidate_t).clamp_min(0.0) * content_t[None]
    observed_device = observed_t.to(device).float()
    inside_w = (observed_device[:, None] * inside[None]).reshape(maps, len(candidates), cells)
    ring_w = (observed_device[:, None] * ring[None]).reshape(maps, len(candidates), cells)
    inside_mass = inside_w.sum(dim=-1)
    ring_mass = ring_w.sum(dim=-1)
    finite = (inside_mass > 1.0e-8) & (ring_mass > 1.0e-8)
    inside_sum = torch.einsum("kmp,mnp->kmn", permuted, inside_w)
    ring_sum = torch.einsum("kmp,mnp->kmn", permuted, ring_w)
    contrast = inside_sum / inside_mass.clamp_min(1.0e-8)
    contrast = contrast - ring_sum / ring_mass.clamp_min(1.0e-8)
    contrast = torch.where(finite[None], contrast, torch.full_like(contrast, float("nan")))
    count = torch.isfinite(contrast).sum(dim=1)
    safe = torch.nan_to_num(contrast, nan=0.0)
    mean = safe.sum(dim=1) / count.clamp_min(1).to(safe.dtype)
    centered = torch.where(
        torch.isfinite(contrast),
        (safe - mean[:, None]).square(),
        torch.zeros_like(safe),
    )
    variance = centered.sum(dim=1) / (count - 1).clamp_min(1).to(safe.dtype)
    lcb = mean - 1.96 * torch.sqrt(variance / count.clamp_min(1).to(safe.dtype))
    candidate_valid = count >= 2
    lcb = torch.where(candidate_valid, lcb, torch.full_like(lcb, float("-inf")))

    values_i = lcb[:, :, None]
    values_j = lcb[:, None, :]
    valid_i = candidate_valid[:, :, None]
    valid_j = candidate_valid[:, None, :]
    less = ((values_i > values_j) & valid_i & valid_j).sum(dim=2).float()
    equal = ((values_i == values_j) & valid_i & valid_j).sum(dim=2).float()
    valid_count = candidate_valid.sum(dim=1, keepdim=True)
    ranks = (less + 0.5 * (equal - 1.0)) / (valid_count - 1).clamp_min(1).float()
    ranks = torch.where(candidate_valid, ranks, torch.zeros_like(ranks))
    ranks = torch.where(
        candidate_valid & (valid_count == 1),
        torch.ones_like(ranks),
        ranks,
    )
    base_rank = torch.from_numpy(
        _rank(base_scores, np.ones(len(base_scores), dtype=bool))
    ).to(device)
    fused = 0.75 * base_rank[None] + 0.25 * ranks
    fused = torch.where(candidate_valid, fused, torch.full_like(fused, float("-inf")))
    return (fused.max(dim=1).values - fused[:, accepted_index]).detach().cpu().numpy()


def _load_prediction_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "predictions" / "prediction_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        map_path = root / "predictions" / row["map_path"]
        if sha256_file(map_path) != row["map_sha256"]:
            raise ValueError(f"S8 prediction map hash mismatch: {row['image_id']}")
        values = np.load(map_path, allow_pickle=False)
        if values.ndim != 2 or values.dtype != np.float16 or not np.isfinite(values).all():
            raise ValueError(f"S8 prediction map content mismatch: {row['image_id']}")
        result[row["image_id"]] = {"row": row, "map": values}
    if len(result) != 371:
        raise ValueError("S8 prediction manifest cohort mismatch")
    return result


def audit_output(output_root: Path, protocol_path: Path, audit_output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("status") != "FROZEN_PRELAUNCH" or protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("S8 protocol provenance mismatch")
    pair_path = output_root / "prediction_pair_freeze.json"
    pair = _json(pair_path)
    if (
        pair.get("experiment_id") != EXPERIMENT_ID
        or pair.get("protocol_sha256") != sha256_file(protocol_path)
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
    control_predictions = _load_prediction_manifest(output_root / ARMS[0])
    primary_predictions = _load_prediction_manifest(output_root / ARMS[1])
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        freeze = _json(freeze_path)
        if sha256_file(freeze_path) != pair["arms"][arm] or freeze.get("arm") != arm:
            raise ValueError(f"S8 arm freeze mismatch: {arm}")
        if freeze.get("validation_gt_read") is not False or freeze.get("consumer_trained") is not False or freeze.get("test_evaluated") is not False:
            raise ValueError(f"S8 arm safety boundary failed: {arm}")
    checked = 0
    switches = 0
    nonconstant_error_banks = 0
    for row in evidence_manifest["rows"]:
        image_id = str(row["image_id"])
        evidence_path = output_root / str(row["evidence_path"])
        if sha256_file(evidence_path) != row["evidence_sha256"]:
            raise ValueError(f"S8 evidence hash mismatch: {image_id}")
        with np.load(evidence_path, allow_pickle=False) as evidence:
            height = int(np.asarray(evidence["mask_height"]))
            width = int(np.asarray(evidence["mask_width"]))
            source_masks = _unpack_masks(evidence["packed_candidate_masks"], height, width)
            candidates, content = _project_grid(
                source_masks,
                padded_side=int(np.asarray(evidence["projection_padded_side"])),
                content_box=np.asarray(evidence["projection_content_box"], dtype=np.int32),
            )
            if not np.allclose(candidates, evidence["candidate_masks"], atol=2.0e-6, rtol=0.0) or not np.allclose(content, evidence["content_mask"], atol=2.0e-6, rtol=0.0):
                raise ValueError(f"S8 independent candidate projection mismatch: {image_id}")
            noise = np.asarray(evidence["noise_bank"], dtype=np.float32)
            if noise.shape != (10, 196) or not np.isfinite(noise).all():
                raise ValueError(f"S8 noise bank mismatch: {image_id}")
            keep = 49
            order = np.argsort(noise, axis=1, kind="stable")
            expected_observed = np.zeros_like(noise, dtype=bool)
            np.put_along_axis(expected_observed, order[:, keep:], True, axis=1)
            expected_grid = expected_observed.reshape(10, 14, 14)
            if not np.array_equal(np.asarray(evidence["original_observed"], dtype=bool), expected_grid) or not np.array_equal(np.asarray(evidence["aligned_flip_observed"], dtype=bool), expected_grid[..., ::-1]):
                raise ValueError(f"S8 mask/noise arithmetic mismatch: {image_id}")
            original_errors = np.asarray(evidence["original_errors"], dtype=np.float32)
            flip_errors = np.asarray(evidence["aligned_flip_errors"], dtype=np.float32)
            nonconstant_error_banks += int(float(np.std(np.concatenate((original_errors, flip_errors)))) > 0.0)
            original_lcb, original_valid = _lcb(evidence["original_errors"], evidence["original_observed"], candidates, content)
            flip_lcb, flip_valid = _lcb(evidence["aligned_flip_errors"], evidence["aligned_flip_observed"], candidates, content)
            combined_errors = np.concatenate((evidence["original_errors"], evidence["aligned_flip_errors"]), axis=0)
            combined_observed = np.concatenate((evidence["original_observed"], evidence["aligned_flip_observed"]), axis=0)
            combined_lcb, combined_valid = _lcb(combined_errors, combined_observed, candidates, content)
            base_scores = np.asarray(evidence["base_scores"], dtype=np.float32)
            base_rank = _rank(base_scores, np.ones(len(base_scores), dtype=bool))
            # The producer applies tie-aware ranking to its float32 tensors before
            # serializing evidence. Re-ranking a freshly recomputed NumPy LCB can
            # split/merge ties at ~1e-8 and falsely reject an otherwise exact run.
            # Arithmetic remains independently checked above; rank from the
            # producer's serialized float32 LCB values to reproduce its contract.
            original_rank = _rank_serialized_lcb(evidence["original_lcb"], original_valid)
            flip_rank = _rank_serialized_lcb(evidence["aligned_flip_lcb"], flip_valid)
            combined_rank = _rank_serialized_lcb(evidence["combined_lcb"], combined_valid)
            original_fused = np.float32(0.75) * base_rank + np.float32(0.25) * original_rank
            flip_fused = np.float32(0.75) * base_rank + np.float32(0.25) * flip_rank
            fused = np.float32(0.75) * base_rank + np.float32(0.25) * combined_rank
            original_fused = np.where(original_valid, original_fused, -np.inf).astype(np.float32)
            flip_fused = np.where(flip_valid, flip_fused, -np.inf).astype(np.float32)
            fused = np.where(combined_valid, fused, -np.inf).astype(np.float32)
            for observed, expected, name in (
                (original_lcb, evidence["original_lcb"], "original LCB"),
                (flip_lcb, evidence["aligned_flip_lcb"], "flip LCB"),
                (combined_lcb, evidence["combined_lcb"], "combined LCB"),
                (fused, evidence["combined_fused"], "fused score"),
            ):
                if not np.allclose(observed, expected, atol=2.0e-5, rtol=0.0, equal_nan=True):
                    raise ValueError(f"S8 {name} arithmetic mismatch: {image_id}")
            accepted_index = int(np.argmax(base_scores))
            original_winner = int(np.argmax(original_fused)) if original_valid.any() else -1
            flip_winner = int(np.argmax(flip_fused)) if flip_valid.any() else -1
            combined_winner = int(np.argmax(fused)) if combined_valid.any() else -1
            families = np.asarray(evidence["family_ids"], dtype=np.int32)
            family_consistent = bool(
                combined_winner >= 0
                and original_winner >= 0
                and flip_winner >= 0
                and families[combined_winner] == families[original_winner] == families[flip_winner]
            )
            accepted_value = float(fused[accepted_index]) if combined_valid[accepted_index] else float("-inf")
            observed_improvement = float(fused[combined_winner] - accepted_value) if combined_winner >= 0 else float("-inf")
            null_improvement = _null_improvements(
                combined_errors,
                combined_observed,
                candidates,
                content,
                base_scores,
                accepted_index,
            )
            if not np.allclose(null_improvement, evidence["null_max_improvements"], atol=2.0e-5, rtol=0.0, equal_nan=True):
                raise ValueError(f"S8 independent 255-null distribution mismatch: {image_id}")
            exceedances = int(np.sum(null_improvement >= observed_improvement - 1.0e-12)) if np.isfinite(observed_improvement) else 255
            p_value = (1.0 + exceedances) / 256.0
            switched = bool(
                family_consistent
                and combined_winner != accepted_index
                and combined_winner >= 0
                and observed_improvement > 0.0
                and p_value <= 0.05
            )
            expected_selected = combined_winner if switched else accepted_index
            scalar_checks = (
                (int(np.asarray(evidence["original_winner"])), original_winner, "original winner"),
                (int(np.asarray(evidence["aligned_flip_winner"])), flip_winner, "flip winner"),
                (int(np.asarray(evidence["combined_winner"])), combined_winner, "combined winner"),
                (int(np.asarray(evidence["family_consistent"])), int(family_consistent), "family gate"),
                (int(np.asarray(evidence["permutation_exceedances"])), exceedances, "null exceedances"),
                (int(np.asarray(evidence["switched"])), int(switched), "switch decision"),
                (int(np.asarray(evidence["selected_index"])), expected_selected, "selected index"),
            )
            for observed_value, expected_value, name in scalar_checks:
                if observed_value != expected_value:
                    raise ValueError(f"S8 {name} mismatch: {image_id}")
            if abs(float(np.asarray(evidence["observed_improvement"])) - observed_improvement) > 2.0e-5 or abs(float(np.asarray(evidence["permutation_p_value"])) - p_value) > 1.0e-12:
                raise ValueError(f"S8 randomization scalar mismatch: {image_id}")
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
            control_probability = float(control_predictions[image_id]["row"]["bag_probability"])
            primary_probability = float(primary_predictions[image_id]["row"]["bag_probability"])
            if control_probability != primary_probability:
                raise ValueError(f"S8 arm bag probability mismatch: {image_id}")
            expected_control_map = (source_masks[int(np.argmax(base_scores))] * control_probability).astype(np.float16)
            expected_primary_map = (source_masks[selected_index] * primary_probability).astype(np.float16)
            if not np.array_equal(control_predictions[image_id]["map"], expected_control_map) or not np.array_equal(primary_predictions[image_id]["map"], expected_primary_map):
                raise ValueError(f"S8 physical prediction map mismatch: {image_id}")
            checked += 1
    if nonconstant_error_banks != checked:
        raise ValueError("S8 reconstruction evidence is constant for at least one image")
    result = {
        "audit_id": "independent_skelex_reconstruction_selector_s8_output_v1",
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED",
        "experiment_id": EXPERIMENT_ID,
        "validation_predictions_per_arm": checked,
        "switched_predictions": switches,
        "candidate_projections_reproduced": checked,
        "spatial_null_distributions_reproduced": checked,
        "physical_prediction_maps_verified": checked * 2,
        "nonconstant_reconstruction_banks": nonconstant_error_banks,
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
