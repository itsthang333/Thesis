"""Independent GT-blind auditor for S10 checkpoint, evidence, and three arms."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import audit_bas_candidate_descriptor_core as common
import run_bas_candidate_descriptor_core as base
import run_rad_dino_mask_bag_mil_probe as legacy
from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from models.highres_candidate_pmil import HighResProposalMIL
from models.mae_reconstruction import pad_to_square
from models.mask_bag_selector_cache import unpack_candidate_masks
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPERIMENT_ID = "EXP-20260803-codex-s10-highres-proposal-pmil-v1"
ARMS = (
    "geometry_v3_plus_upstream_control",
    "control_plus_s10_identity_capacity",
    "s10_pareto_identity_capture_purity",
)
SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CACHE_FREEZE_SHA256 = "2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c"
BASELINE_CHECKPOINT_SHA256 = "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
BASELINE_FREEZE_SHA256 = "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
BASELINE_SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
BASELINE_PROTOCOL_SHA256 = "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
TRAIN_CANDIDATE_MANIFEST_SHA256 = "ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1"
TRAIN_PSEUDO_MANIFEST_SHA256 = "5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21"
VAL_CANDIDATE_MANIFEST_SHA256 = "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
VAL_PSEUDO_MANIFEST_SHA256 = "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
PRETRAINED_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
EXPECTED_TRAIN = 2981
EXPECTED_VALIDATION = 371
EXPECTED_EPOCHS = 32
EXPECTED_BATCH_SIZE = 4
EXPECTED_IMAGE_SIZE = 640
EXPECTED_SUPPORT_SIZE = 160
MAXIMUM_INFERENCE_DELTA = 5.0e-4
MAXIMUM_SCORE_DELTA = 5.0e-5
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _array_sha256(values: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return sha256(stream.getvalue()).hexdigest()


def _safety(payload: dict[str, Any], name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"S10 {name} safety boundary failed")


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("S10 rank input is invalid")
    if len(values) == 1:
        return np.ones(1, dtype=np.float32)
    result = np.empty(len(values), dtype=np.float32)
    for index, value in enumerate(values):
        less = np.float32(np.sum(values < value))
        equal = np.float32(np.sum(values == value))
        numerator = less + np.float32(0.5) * (equal - np.float32(1.0))
        result[index] = numerator / np.float32(len(values) - 1)
    return result


def _pareto_local(
    identity: np.ndarray,
    capture: np.ndarray,
    purity: np.ndarray,
    indices: np.ndarray,
    control_local: int,
) -> tuple[int, int]:
    components = np.stack(tuple(_rank(values) for values in (identity, capture, purity)), axis=1)
    baseline = components[control_local]
    eligible = np.flatnonzero(
        np.all(components >= baseline[None], axis=1)
        & np.any(components > baseline[None], axis=1)
    )
    if not len(eligible):
        return control_local, 0
    best = max(
        eligible.tolist(),
        key=lambda row: (
            float(components[row].min()),
            float(components[row, 0]),
            -int(indices[row]),
        ),
    )
    return int(best), int(len(eligible))


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize(
        (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())
    values = values.permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _reference_direct_square(
    masks: torch.Tensor,
    *,
    padded_side: int,
    content_box: tuple[int, int, int, int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (int(value) for value in content_box)
    coordinates = (torch.arange(EXPECTED_SUPPORT_SIZE, dtype=torch.float32) + 0.5) * (
        float(padded_side) / EXPECTED_SUPPORT_SIZE
    )
    source_x = (coordinates - float(x0)) / float(x1 - x0)
    source_y = (coordinates - float(y0)) / float(y1 - y0)
    grid_y, grid_x = torch.meshgrid(source_y, source_x, indexing="ij")
    grid = torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)
    grid = grid[None].expand(masks.shape[0], -1, -1, -1)
    return F.grid_sample(
        masks[:, None].float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].clamp(0.0, 1.0)


def _candidate_masks(candidate_root: Path, row: dict[str, str]) -> np.ndarray:
    masks, _metadata, _scores, _fallback = legacy._load_candidate_payload(
        candidate_root, row, maximum_candidates=81
    )
    return masks


def _reference_input(
    image_path: Path,
    masks: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(image_path) as image:
        pixels, projection = _normalized_square(image)
    supports = _reference_direct_square(
        torch.from_numpy(masks),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
    ).to(torch.float16)
    content = _reference_direct_square(
        torch.ones((1, masks.shape[-2], masks.shape[-1]), dtype=torch.float32),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
    )[0].to(torch.float16)
    return pixels, supports, content


def _collate_reference(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(item[1].shape[0] for item in items)
    masks = torch.zeros(
        len(items), maximum, EXPECTED_SUPPORT_SIZE, EXPECTED_SUPPORT_SIZE,
        dtype=torch.float16,
    )
    valid = torch.zeros(len(items), maximum, dtype=torch.bool)
    for row, (_pixels, supports, _content) in enumerate(items):
        masks[row, : len(supports)] = supports
        valid[row, : len(supports)] = True
    return (
        torch.stack([item[0] for item in items]),
        masks,
        torch.stack([item[2] for item in items]),
        valid,
    )


def _capture_purity_numpy(
    dense_logits: np.ndarray,
    candidates: np.ndarray,
    rings: np.ndarray,
    content: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dense = np.asarray(dense_logits, dtype=np.float64)
    evidence = (1.0 / (1.0 + np.exp(-dense))) * np.asarray(content, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    rings = np.asarray(rings, dtype=np.float64)
    inside = (candidates * evidence[None]).sum(axis=(1, 2))
    outside = (rings * evidence[None]).sum(axis=(1, 2))
    capture = inside / max(float(evidence.sum()), 1.0e-12)
    purity = inside / np.maximum(candidates.sum(axis=(1, 2)), 1.0e-12)
    purity -= outside / np.maximum(rings.sum(axis=(1, 2)), 1.0e-12)
    return capture.astype(np.float32), purity.astype(np.float32)


def _load_s10_checkpoint(
    path: Path,
    *,
    source_commit: str,
    protocol_sha: str,
) -> tuple[dict[str, Any], HighResProposalMIL]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "checkpoint")
    architecture = checkpoint.get("architecture", {})
    optimizer = checkpoint.get("optimizer", {})
    if (
        checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha
        or checkpoint.get("split_sha256") != SPLIT_SHA256
        or checkpoint.get("pretrained_sha256") != PRETRAINED_SHA256
        or checkpoint.get("training_labels") != "image_level_normal_tumor_only"
        or architecture.get("input_size") != EXPECTED_IMAGE_SIZE
        or architecture.get("support_size") != EXPECTED_SUPPORT_SIZE
        or optimizer.get("epochs") != EXPECTED_EPOCHS
        or optimizer.get("batch_size") != EXPECTED_BATCH_SIZE
        or optimizer.get("checkpoint_selection")
        != "final_epoch_only_no_validation_selection"
    ):
        raise ValueError("S10 checkpoint provenance/config mismatch")
    state = checkpoint.get("model_state_dict", {})
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("S10 checkpoint state is missing or non-finite")
    model = HighResProposalMIL()
    model.load_state_dict(state, strict=True)
    return checkpoint, model.requires_grad_(False).eval()


@torch.inference_mode()
def _reexecute_validation(
    model: nn.DataParallel,
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
    candidate_root: Path,
    dataset_root: Path,
    evidence_rows: dict[str, dict[str, str]],
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    reproduced: dict[str, dict[str, np.ndarray]] = {}
    maximum = {"identity": 0.0, "dense": 0.0, "capture": 0.0, "purity": 0.0}
    for start in range(0, len(rows), EXPECTED_BATCH_SIZE):
        batch_rows = rows[start : start + EXPECTED_BATCH_SIZE]
        batch_records = records[start : start + EXPECTED_BATCH_SIZE]
        inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        indices_by_row: list[np.ndarray] = []
        for row, record in zip(batch_rows, batch_records):
            if row["image_id"] != record["image_id"]:
                raise ValueError("S10 auditor split/cache order mismatch")
            candidate_row = candidates[Path(row["image_id"]).stem]
            if candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]:
                raise ValueError("S10 auditor candidate provenance mismatch")
            all_masks = _candidate_masks(candidate_root, candidate_row)
            indices = np.asarray(record["candidate_indices"], dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= len(all_masks)):
                raise ValueError("S10 auditor candidate index is invalid")
            inputs.append(
                _reference_input(
                    locate_verified_image(dataset_root, row), all_masks[indices]
                )
            )
            indices_by_row.append(indices)
        pixels, masks, content, valid = _collate_reference(inputs)
        pixels = pixels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        content = content.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            original = model(pixels, masks, content, valid)
            flipped = model(
                pixels.flip(-1), masks.flip(-1), content.flip(-1), valid
            )
        identity = 0.5 * (
            original.classification_logits.float()
            + flipped.classification_logits.float()
        )
        dense = 0.5 * (
            original.dense_logits.float() + flipped.dense_logits.float().flip(-1)
        )
        for offset, (row, indices) in enumerate(zip(batch_rows, indices_by_row)):
            count = len(indices)
            evidence_row = evidence_rows[row["image_id"]]
            evidence_path = output_root / "s10_candidate_evidence" / evidence_row["evidence_path"]
            if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
                raise ValueError("S10 evidence hash mismatch")
            with np.load(evidence_path, allow_pickle=False) as evidence:
                observed_identity = np.asarray(evidence["identity"], dtype=np.float32)
                observed_dense = np.asarray(evidence["dense_logits"], dtype=np.float16)
                reference_identity = identity[offset, :count].cpu().numpy().astype(np.float32)
                reference_dense = dense[offset].cpu().numpy().astype(np.float16)
                candidates_np = original.candidate_weights[offset, :count].float().cpu().numpy()
                rings_np = original.ring_weights[offset, :count].float().cpu().numpy()
                content_np = content[offset].float().cpu().numpy()
                capture, purity = _capture_purity_numpy(
                    dense[offset].float().cpu().numpy(),
                    candidates_np,
                    rings_np,
                    content_np,
                )
                observed_capture = np.asarray(evidence["capture"], dtype=np.float32)
                observed_purity = np.asarray(evidence["purity"], dtype=np.float32)
                deltas = {
                    "identity": float(np.max(np.abs(reference_identity - observed_identity))),
                    "dense": float(np.max(np.abs(reference_dense.astype(np.float32) - observed_dense.astype(np.float32)))),
                    "capture": float(np.max(np.abs(capture - observed_capture))),
                    "purity": float(np.max(np.abs(purity - observed_purity))),
                }
                for key, value in deltas.items():
                    maximum[key] = max(maximum[key], value)
                if any(value > MAXIMUM_INFERENCE_DELTA for value in deltas.values()):
                    raise ValueError(f"S10 validation inference mismatch: {row['image_id']}")
                reproduced[row["image_id"]] = {
                    "candidate_indices": indices,
                    "identity": observed_identity,
                    "capture": observed_capture,
                    "purity": observed_purity,
                }
    return reproduced, maximum


def audit_output(
    *,
    output_root: Path,
    protocol_path: Path,
    binding_path: Path,
    dataset_root: Path,
    split_path: Path,
    train_candidate_root: Path,
    val_candidate_root: Path,
    cache_root: Path,
    baseline_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    protocol_sha = sha256_file(protocol_path)
    protocol = _json(protocol_path)
    binding = _json(binding_path)
    source_commit = str(binding.get("scientific_source_commit"))
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != "FROZEN_PRELAUNCH"
        or binding.get("protocol_sha256") != protocol_sha
        or source_commit != protocol.get("scientific_source", {}).get("commit")
    ):
        raise ValueError("S10 protocol/binding mismatch")
    run_manifest = _json(output_root / "run_manifest.json")
    _safety(run_manifest, "run manifest")
    if (
        run_manifest.get("experiment_id") != EXPERIMENT_ID
        or run_manifest.get("source_commit") != source_commit
        or run_manifest.get("protocol_sha256") != protocol_sha
        or run_manifest.get("cohort")
        != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
    ):
        raise ValueError("S10 run manifest mismatch")
    split_rows = {
        split: load_split_rows_without_annotations(
            split_path, expected_sha256=SPLIT_SHA256, split=split
        )
        for split in ("train", "val")
    }
    train_candidates, _ = validate_candidate_diagnostics_manifest(
        train_candidate_root,
        expected_image_names=[row["image_id"] for row in split_rows["train"]],
        split="train",
        expected_manifest_sha256=TRAIN_CANDIDATE_MANIFEST_SHA256,
        expected_pseudo_manifest_sha256=TRAIN_PSEUDO_MANIFEST_SHA256,
    )
    val_candidates, _ = validate_candidate_diagnostics_manifest(
        val_candidate_root,
        expected_image_names=[row["image_id"] for row in split_rows["val"]],
        split="val",
        expected_manifest_sha256=VAL_CANDIDATE_MANIFEST_SHA256,
        expected_pseudo_manifest_sha256=VAL_PSEUDO_MANIFEST_SHA256,
    )
    cache_args = SimpleNamespace(
        selector_cache_root=cache_root,
        expected_selector_cache_freeze_sha256=CACHE_FREEZE_SHA256,
        expected_split_sha256=SPLIT_SHA256,
        expected_baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA256,
        expected_baseline_source_commit=BASELINE_SOURCE_COMMIT,
        expected_baseline_protocol_sha256=BASELINE_PROTOCOL_SHA256,
    )
    cache_freeze, cache_manifest_rows = base._verify_cache_freeze(cache_args)
    accepted = base._load_cache_records(cache_args, split_rows, cache_manifest_rows)
    _safety(cache_freeze, "selector cache")

    input_gate = _json(output_root / "input_operational_gate.json")
    input_manifest_path = output_root / "input_manifest.csv"
    _safety(input_gate, "input gate")
    if (
        input_gate.get("status") != "PASS_BEFORE_TRAINING"
        or input_gate.get("source_commit") != source_commit
        or input_gate.get("protocol_sha256") != protocol_sha
        or input_gate.get("input_manifest_sha256") != sha256_file(input_manifest_path)
        or input_gate.get("cohort")
        != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
    ):
        raise ValueError("S10 input gate mismatch")
    input_rows = _rows(input_manifest_path)
    if len(input_rows) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise ValueError("S10 input manifest cohort mismatch")
    input_by_key = {(row["split"], row["image_id"]): row for row in input_rows}
    for split in ("train", "val"):
        candidates = train_candidates if split == "train" else val_candidates
        for row, record in zip(split_rows[split], accepted[split]):
            manifest = input_by_key[(split, row["image_id"])]
            candidate_row = candidates[Path(row["image_id"]).stem]
            indices = np.asarray(record["candidate_indices"], dtype=np.int64)
            if (
                manifest["candidate_payload_sha256"] != candidate_row["diagnostic_sha256"]
                or manifest["candidate_indices_sha256"] != _array_sha256(indices)
                or int(manifest["candidate_count"]) != len(indices)
            ):
                raise ValueError("S10 independent input manifest mismatch")

    history_path = output_root / "training_history.json"
    history = _json(history_path).get("epochs", [])
    if (
        len(history) != EXPECTED_EPOCHS
        or [int(row["epoch"]) for row in history]
        != list(range(1, EXPECTED_EPOCHS + 1))
        or not all(np.isfinite([float(value) for value in row.values()]).all() for row in history)
    ):
        raise ValueError("S10 training history mismatch")
    checkpoint_path = output_root / "highres_candidate_pmil.pt"
    checkpoint, model = _load_s10_checkpoint(
        checkpoint_path, source_commit=source_commit, protocol_sha=protocol_sha
    )
    if (
        checkpoint.get("training_history_sha256") != sha256_file(history_path)
        or checkpoint.get("input_manifest_sha256") != sha256_file(input_manifest_path)
    ):
        raise ValueError("S10 checkpoint input/history binding mismatch")

    evidence_manifest_path = output_root / "s10_candidate_evidence" / "evidence_manifest.csv"
    evidence_rows = {row["image_id"]: row for row in _rows(evidence_manifest_path)}
    if len(evidence_rows) != EXPECTED_VALIDATION:
        raise ValueError("S10 evidence cohort mismatch")
    parallel = nn.DataParallel(model.to(device), device_ids=(0, 1), output_device=0)
    reproduced, inference_deltas = _reexecute_validation(
        parallel,
        split_rows["val"],
        accepted["val"],
        val_candidates,
        val_candidate_root,
        dataset_root,
        evidence_rows,
        output_root,
        device,
    )
    del parallel, model
    torch.cuda.empty_cache()

    common.BASELINE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256
    common.BASELINE_FREEZE_SHA256 = BASELINE_FREEZE_SHA256
    common.BASELINE_SOURCE_COMMIT = BASELINE_SOURCE_COMMIT
    common.BASELINE_PROTOCOL_SHA256 = BASELINE_PROTOCOL_SHA256
    common.SPLIT_SHA256 = SPLIT_SHA256
    baseline_model = common._load_baseline(baseline_root)
    baseline_freeze_path = baseline_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != BASELINE_FREEZE_SHA256:
        raise ValueError("S10 accepted baseline freeze mismatch")
    baseline_freeze = _json(baseline_freeze_path)
    _safety(baseline_freeze, "accepted baseline")
    baseline_predictions = {
        row["image_id"]: row
        for row in _rows(baseline_root / "predictions" / "prediction_manifest.csv")
    }

    triple_path = output_root / "prediction_triple_freeze.json"
    triple = _json(triple_path)
    _safety(triple, "prediction triple")
    if (
        triple.get("experiment_id") != EXPERIMENT_ID
        or triple.get("source_commit") != source_commit
        or triple.get("protocol_sha256") != protocol_sha
        or triple.get("all_arms_physically_frozen_before_validation_gt") is not True
        or triple.get("collaborator_output_accessed") is not False
        or set(triple.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S10 prediction triple contract mismatch")
    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    score_rows: dict[str, dict[str, dict[str, str]]] = {}
    checkpoint_sha = sha256_file(checkpoint_path)
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != triple["arms"][arm]:
            raise ValueError("S10 arm freeze hash mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"{arm} freeze")
        if (
            freeze.get("arm") != arm
            or freeze.get("s10_checkpoint_sha256") != checkpoint_sha
            or freeze.get("s10_candidate_evidence_manifest_sha256")
            != sha256_file(evidence_manifest_path)
        ):
            raise ValueError("S10 arm freeze provenance mismatch")
        prediction_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "predictions" / "prediction_manifest.csv")
        }
        score_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "candidate_scores" / "candidate_score_manifest.csv")
        }
        if len(prediction_rows[arm]) != EXPECTED_VALIDATION or len(score_rows[arm]) != EXPECTED_VALIDATION:
            raise ValueError("S10 arm output cohort mismatch")

    maximum_base_delta = 0.0
    capacity_changes = 0
    pareto_changes = 0
    total_dominators = 0
    physical_maps = 0
    physical_scores = 0
    for row, record in zip(split_rows["val"], accepted["val"]):
        image_id = row["image_id"]
        indices = np.asarray(record["candidate_indices"], dtype=np.int64)
        masks = unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        evidence_path = output_root / "s10_candidate_evidence" / evidence_rows[image_id]["evidence_path"]
        with np.load(evidence_path, allow_pickle=False) as evidence:
            stored_base = np.asarray(evidence["baseline_logits"], dtype=np.float32)
            observed_base = common._base_logits(baseline_model, record)
            maximum_base_delta = max(
                maximum_base_delta,
                float(np.max(np.abs(stored_base - observed_base))),
            )
            candidate_row = val_candidates[Path(image_id).stem]
            candidate_path = val_candidate_root / candidate_row["diagnostic_path"]
            if sha256_file(candidate_path) != record["candidate_payload_sha256"]:
                raise ValueError("S10 candidate payload differs")
            with np.load(candidate_path, allow_pickle=False) as payload:
                upstream = np.asarray(payload["selection_scores"], dtype=np.float32)[indices]
            if not np.array_equal(upstream, evidence["upstream_scores"]):
                raise ValueError("S10 upstream evidence differs")
            learned = reproduced[image_id]
            identity = learned["identity"]
            capture = learned["capture"]
            purity = learned["purity"]
            control = np.stack(
                (_rank(stored_base), _rank(upstream)), axis=0
            ).mean(axis=0, dtype=np.float32)
            capacity = np.stack(
                (_rank(stored_base), _rank(upstream), _rank(identity)), axis=0
            ).mean(axis=0, dtype=np.float32)
            control_local = int(np.argmax(control))
            capacity_local = int(np.argmax(capacity))
            primary_local, dominators = _pareto_local(
                identity, capture, purity, indices, control_local
            )
            primary_scores = np.zeros(len(indices), dtype=np.float32)
            primary_scores[primary_local] = 1.0
            if (
                not np.array_equal(control, evidence["control_scores"])
                or not np.array_equal(capacity, evidence["capacity_scores"])
                or not np.array_equal(primary_scores, evidence["primary_decision_scores"])
                or int(evidence["dominator_count"]) != dominators
            ):
                raise ValueError("S10 decision evidence does not reproduce")
        capacity_changes += int(capacity_local != control_local)
        pareto_changes += int(primary_local != control_local)
        total_dominators += dominators
        expected = {
            ARMS[0]: (control, control_local),
            ARMS[1]: (capacity, capacity_local),
            ARMS[2]: (primary_scores, primary_local),
        }
        for arm, (scores, winner) in expected.items():
            score_row = score_rows[arm][image_id]
            score_path = output_root / arm / "candidate_scores" / score_row["score_path"]
            if sha256_file(score_path) != score_row["score_sha256"]:
                raise ValueError("S10 candidate score hash mismatch")
            with np.load(score_path, allow_pickle=False) as stored:
                if (
                    not np.array_equal(stored["candidate_indices"], indices.astype(np.int32))
                    or not np.array_equal(stored["candidate_logits"], scores)
                ):
                    raise ValueError("S10 candidate score payload differs")
            prediction = prediction_rows[arm][image_id]
            bag_probability = float(baseline_predictions[image_id]["bag_probability"])
            if (
                int(prediction["selected_candidate_index"]) != int(indices[winner])
                or abs(float(prediction["bag_probability"]) - bag_probability) > 5.0e-7
            ):
                raise ValueError("S10 physical winner/bag scalar differs")
            map_path = output_root / arm / "predictions" / prediction["map_path"]
            if sha256_file(map_path) != prediction["map_sha256"]:
                raise ValueError("S10 physical map hash mismatch")
            observed_map = np.load(map_path, allow_pickle=False)
            expected_map = (masks[winner] * bag_probability).astype(np.float16)
            if not np.array_equal(observed_map, expected_map):
                raise ValueError("S10 physical map content differs")
            physical_scores += 1
            physical_maps += 1
    if maximum_base_delta > MAXIMUM_SCORE_DELTA:
        raise ValueError("S10 baseline logit reproduction exceeds tolerance")
    diagnostics = _json(output_root / "gt_blind_diagnostics.json")
    _safety(diagnostics, "GT-blind diagnostics")
    if (
        int(diagnostics["capacity_changed_selections"]) != capacity_changes
        or int(diagnostics["pareto_changed_selections"]) != pareto_changes
        or int(diagnostics["total_pareto_dominators"]) != total_dominators
    ):
        raise ValueError("S10 GT-blind diagnostics do not reproduce")
    return {
        "audit_id": "independent_highres_candidate_pmil_s10_output_v1",
        "status": "PREDICTION_TRIPLE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha,
        "run_manifest_sha256": sha256_file(output_root / "run_manifest.json"),
        "prediction_triple_freeze_sha256": sha256_file(triple_path),
        "validation_inference_reproduced": EXPECTED_VALIDATION,
        "physical_candidate_scores_verified": physical_scores,
        "physical_prediction_maps_verified": physical_maps,
        "maximum_inference_deltas": inference_deltas,
        "maximum_baseline_logit_delta": maximum_base_delta,
        "capacity_changed_selections": capacity_changes,
        "pareto_changed_selections": pareto_changes,
        "total_pareto_dominators": total_dominators,
        "collaborator_output_accessed": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S10 auditor requires exactly two visible CUDA devices")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"S10 auditor requires T4 x2, got {names}")
    audit = audit_output(
        output_root=args.output_root,
        protocol_path=args.protocol,
        binding_path=args.binding,
        dataset_root=args.dataset_root,
        split_path=args.split_manifest,
        train_candidate_root=args.train_candidate_root,
        val_candidate_root=args.val_candidate_root,
        cache_root=args.selector_cache_root,
        baseline_root=args.baseline_root,
        device=torch.device("cuda:0"),
    )
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output_audit.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
