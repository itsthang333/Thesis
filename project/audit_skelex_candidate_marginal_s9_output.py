"""Independent GT-blind physical auditor for the frozen S9 prediction pair."""

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
    verify_model_snapshot,
)
from models.mae_reconstruction import pad_to_square
from models.mask_bag_selector_cache import unpack_candidate_masks
from models.skelex_candidate_marginal import (
    NonlinearTokenEvidenceHead,
    SKELEX_GRID_SIZE,
    SKELEX_HEAD_HIDDEN_DIM,
    SKELEX_HIDDEN_LAYERS,
    SKELEX_HIDDEN_SIZE,
    SKELEX_INPUT_SIZE,
    SKELEX_RING_RADIUS,
    SKELEX_TOKEN_DIM,
    SkelexMultiLayerTokenEncoder,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


EXPERIMENT_ID = "EXP-20260803-codex-s9-skelex-candidate-marginal-v1"
ARMS = (
    "geometry_v3_plus_upstream_equal_rank",
    "geometry_v3_plus_upstream_plus_s9_likelihood_equal_rank",
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
SKELEX_WEIGHT_SHA256 = "81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8"
EXPECTED_TRANSFORMERS_VERSION = "4.50.2"
EXPECTED_TRAIN = 2981
EXPECTED_VALIDATION = 371
EXPECTED_ENCODER_BATCH_SIZE = 2
EXPECTED_EPOCHS = 32
EXPECTED_HEAD_PARAMETERS = 524_801
MAXIMUM_HEAD_LOGIT_DELTA = 5.0e-5
MAXIMUM_LIKELIHOOD_DELTA = 5.0e-5
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
        raise ValueError(f"S9 {name} safety boundary failed")


def _normalized_square(image: Image.Image) -> tuple[torch.Tensor, Any]:
    square, projection = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((SKELEX_INPUT_SIZE, SKELEX_INPUT_SIZE), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD, projection


def _reference_direct_square(
    masks: torch.Tensor,
    *,
    padded_side: int,
    content_box: tuple[int, int, int, int],
) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError("S9 reference masks must be NHW")
    x0, y0, x1, y1 = (int(value) for value in content_box)
    coordinates = (torch.arange(SKELEX_INPUT_SIZE, dtype=torch.float32) + 0.5) * (
        float(padded_side) / SKELEX_INPUT_SIZE
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
    )[:, 0].clamp_(0.0, 1.0)


def _reference_supports(
    masks: np.ndarray,
    *,
    projection: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = torch.from_numpy(np.asarray(masks, dtype=np.float32))
    square = _reference_direct_square(
        source,
        padded_side=projection.padded_side,
        content_box=projection.content_box,
    )
    content_square = _reference_direct_square(
        torch.ones((1, masks.shape[-2], masks.shape[-1]), dtype=torch.float32),
        padded_side=projection.padded_side,
        content_box=projection.content_box,
    )[0]
    candidates = F.interpolate(
        square[:, None], size=(SKELEX_GRID_SIZE, SKELEX_GRID_SIZE), mode="area"
    )[:, 0].clamp_(0.0, 1.0)
    content = F.interpolate(
        content_square[None, None],
        size=(SKELEX_GRID_SIZE, SKELEX_GRID_SIZE),
        mode="area",
    )[0, 0].clamp_(0.0, 1.0)
    candidates = candidates * content[None]
    rings = (
        F.max_pool2d(
            candidates[:, None],
            kernel_size=2 * SKELEX_RING_RADIUS + 1,
            stride=1,
            padding=SKELEX_RING_RADIUS,
        )[:, 0]
        - candidates
    ).clamp_min(0.0) * content[None]
    if bool((candidates.sum(dim=(-2, -1)) <= 1.0e-8).any()):
        raise ValueError("S9 reference candidate has empty inside support")
    if bool((rings.sum(dim=(-2, -1)) <= 1.0e-8).any()):
        raise ValueError("S9 reference candidate has empty ring support")
    return (
        candidates.reshape(len(masks), -1).numpy().astype(np.float16),
        rings.reshape(len(masks), -1).numpy().astype(np.float16),
        (content.reshape(-1) > 1.0e-8).numpy().astype(np.uint8),
    )


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("S9 reference rank input is invalid")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    result = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        less = int(np.sum(values < value))
        equal = int(np.sum(values == value))
        result[index] = (less + 0.5 * (equal - 1)) / (len(values) - 1)
    return result


def _reference_head_logits(
    tokens: np.ndarray,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(tokens, dtype=np.float16))[None].to(device)
    if values.ndim != 3 or values.shape[-1] != SKELEX_TOKEN_DIM:
        raise ValueError("S9 reference token shape mismatch")
    grouped = values.float().reshape(1, values.shape[1], -1, SKELEX_HIDDEN_SIZE)
    normalized = F.normalize(grouped, dim=-1, eps=1.0e-6).flatten(start_dim=-2)
    hidden = F.gelu(
        F.linear(
            normalized,
            state["projection.weight"].to(device),
            state["projection.bias"].to(device),
        )
    )
    logits = F.linear(
        hidden,
        state["output.weight"].to(device),
        state["output.bias"].to(device),
    )[..., 0]
    return logits[0].float().cpu().numpy()


def _reference_likelihood(
    token_logits: np.ndarray,
    candidates: np.ndarray,
    rings: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(token_logits, dtype=np.float64)
    inside_weights = np.asarray(candidates, dtype=np.float64)
    ring_weights = np.asarray(rings, dtype=np.float64)
    log_positive = -np.logaddexp(0.0, -logits)
    log_negative = -np.logaddexp(0.0, logits)
    inside = (inside_weights * log_positive[None]).sum(axis=1) / inside_weights.sum(axis=1)
    outside = (ring_weights * log_negative[None]).sum(axis=1) / ring_weights.sum(axis=1)
    return np.asarray(0.5 * (inside + outside), dtype=np.float32)


def _load_checkpoint(path: Path, *, source_commit: str, protocol_sha: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _safety(checkpoint, "head checkpoint")
    architecture = checkpoint.get("architecture", {})
    optimizer = checkpoint.get("optimizer", {})
    if (
        checkpoint.get("source_commit") != source_commit
        or checkpoint.get("protocol_sha256") != protocol_sha
        or checkpoint.get("split_sha256") != SPLIT_SHA256
        or checkpoint.get("skelex_weight_sha256") != SKELEX_WEIGHT_SHA256
        or checkpoint.get("training_labels") != "image_level_normal_tumor_only"
        or architecture
        != {
            "input_size": SKELEX_INPUT_SIZE,
            "grid_size": SKELEX_GRID_SIZE,
            "hidden_layers": list(SKELEX_HIDDEN_LAYERS),
            "hidden_size": SKELEX_HIDDEN_SIZE,
            "token_dim": SKELEX_TOKEN_DIM,
            "head_hidden_dim": SKELEX_HEAD_HIDDEN_DIM,
            "ring_radius": SKELEX_RING_RADIUS,
        }
        or optimizer.get("epochs") != EXPECTED_EPOCHS
        or optimizer.get("checkpoint_selection")
        != "final_epoch_only_no_validation_selection"
    ):
        raise ValueError("S9 head checkpoint provenance/config mismatch")
    model = NonlinearTokenEvidenceHead()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_HEAD_PARAMETERS:
        raise ValueError("S9 trainable parameter count changed")
    if not all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()):
        raise ValueError("S9 head checkpoint contains non-finite state")
    return checkpoint


def audit_output(
    *,
    output_root: Path,
    protocol_path: Path,
    binding_path: Path,
    dataset_root: Path,
    split_path: Path,
    skelex_model_dir: Path,
    skelex_config_sha256: str,
    skelex_preprocessor_sha256: str,
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
        raise ValueError("S9 protocol/binding mismatch")

    run_manifest_path = output_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    _safety(run_manifest, "run manifest")
    if (
        run_manifest.get("experiment_id") != EXPERIMENT_ID
        or run_manifest.get("source_commit") != source_commit
        or run_manifest.get("protocol_sha256") != protocol_sha
        or run_manifest.get("cohort")
        != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
    ):
        raise ValueError("S9 run manifest mismatch")

    verify_model_snapshot(
        skelex_model_dir,
        expected_config_sha256=skelex_config_sha256,
        expected_preprocessor_sha256=skelex_preprocessor_sha256,
        expected_weight_sha256=SKELEX_WEIGHT_SHA256,
    )
    split_rows = {
        split: load_split_rows_without_annotations(
            split_path,
            expected_sha256=SPLIT_SHA256,
            split=split,
        )
        for split in ("train", "val")
    }
    if len(split_rows["train"]) != EXPECTED_TRAIN or len(split_rows["val"]) != EXPECTED_VALIDATION:
        raise ValueError("S9 independent split cohort mismatch")
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

    feature_gate_path = output_root / "feature_cache_operational_gate.json"
    feature_gate = _json(feature_gate_path)
    _safety(feature_gate, "feature cache gate")
    feature_manifest_path = output_root / "feature_cache_manifest.csv"
    if (
        feature_gate.get("status") != "PASS_BEFORE_HEAD_TRAINING"
        or feature_gate.get("source_commit") != source_commit
        or feature_gate.get("protocol_sha256") != protocol_sha
        or feature_gate.get("feature_cache_manifest_sha256") != sha256_file(feature_manifest_path)
        or feature_gate.get("hidden_layers") != list(SKELEX_HIDDEN_LAYERS)
        or feature_gate.get("token_grid_size") != SKELEX_GRID_SIZE
    ):
        raise ValueError("S9 feature-cache gate mismatch")
    feature_rows_list = _rows(feature_manifest_path)
    feature_rows = {(row["split"], row["image_id"]): row for row in feature_rows_list}
    if len(feature_rows) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise ValueError("S9 feature-cache manifest cohort mismatch")

    history_path = output_root / "training_history.json"
    history = _json(history_path)
    epoch_rows = history.get("epochs", [])
    if (
        not isinstance(epoch_rows, list)
        or len(epoch_rows) != EXPECTED_EPOCHS
        or [int(row["epoch"]) for row in epoch_rows] != list(range(1, EXPECTED_EPOCHS + 1))
        or not all(np.isfinite([float(value) for value in row.values()]).all() for row in epoch_rows)
    ):
        raise ValueError("S9 training history mismatch")
    checkpoint_path = output_root / "skelex_candidate_marginal_head.pt"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        source_commit=source_commit,
        protocol_sha=protocol_sha,
    )
    if (
        checkpoint.get("feature_cache_manifest_sha256") != sha256_file(feature_manifest_path)
        or checkpoint.get("training_history_sha256") != sha256_file(history_path)
    ):
        raise ValueError("S9 checkpoint input/history binding mismatch")

    evidence_manifest_path = output_root / "s9_likelihood_evidence" / "evidence_manifest.csv"
    evidence_rows = {row["image_id"]: row for row in _rows(evidence_manifest_path)}
    if len(evidence_rows) != EXPECTED_VALIDATION:
        raise ValueError("S9 likelihood evidence cohort mismatch")

    import transformers
    from transformers import ViTMAEForPreTraining

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError("S9 auditor transformers version mismatch")
    pretrained = ViTMAEForPreTraining.from_pretrained(skelex_model_dir, local_files_only=True)
    pretrained.vit.config.mask_ratio = 0.0
    pretrained.vit.requires_grad_(False).eval()
    encoder: nn.Module = SkelexMultiLayerTokenEncoder(pretrained.vit).to(device)
    del pretrained
    encoder = nn.DataParallel(encoder, device_ids=(0, 1), output_device=0).eval()
    head_state = checkpoint["model_state_dict"]
    maximum_head_delta = 0.0
    maximum_likelihood_delta = 0.0
    feature_hashes_verified = 0
    validation_reproduced: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val"):
        rows = split_rows[split]
        records = accepted[split]
        candidates_by_stem = train_candidates if split == "train" else val_candidates
        candidate_root = train_candidate_root if split == "train" else val_candidate_root
        for start in range(0, len(rows), EXPECTED_ENCODER_BATCH_SIZE):
            batch_rows = rows[start : start + EXPECTED_ENCODER_BATCH_SIZE]
            batch_records = records[start : start + EXPECTED_ENCODER_BATCH_SIZE]
            pixels: list[torch.Tensor] = []
            payloads: list[tuple[np.ndarray, np.ndarray, Any]] = []
            for row, record in zip(batch_rows, batch_records):
                if row["image_id"] != record["image_id"]:
                    raise ValueError("S9 auditor split/cache order mismatch")
                candidate_row = candidates_by_stem[Path(row["image_id"]).stem]
                if candidate_row["diagnostic_sha256"] != record["candidate_payload_sha256"]:
                    raise ValueError("S9 auditor candidate provenance mismatch")
                masks, _metadata, _scores, _fallback = legacy._load_candidate_payload(
                    candidate_root,
                    candidate_row,
                    maximum_candidates=81,
                )
                indices = np.asarray(record["candidate_indices"], dtype=np.int64)
                if (
                    indices.ndim != 1
                    or not len(indices)
                    or np.any(indices < 0)
                    or np.any(indices >= len(masks))
                    or len(np.unique(indices)) != len(indices)
                ):
                    raise ValueError("S9 auditor candidate indices invalid")
                image_path = locate_verified_image(dataset_root, row)
                with Image.open(image_path) as image:
                    normalized, projection = _normalized_square(image)
                pixels.append(normalized)
                payloads.append((masks[indices], indices, projection))
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                token_batch = encoder(torch.stack(pixels).to(device, non_blocking=True))
            token_batch = token_batch.float().cpu().numpy().astype(np.float16)
            for offset, (row, payload) in enumerate(zip(batch_rows, payloads)):
                masks, indices, projection = payload
                tokens = token_batch[offset]
                inside, rings, content_valid = _reference_supports(masks, projection=projection)
                manifest = feature_rows[(split, row["image_id"])]
                expected_hashes = {
                    "candidate_indices_sha256": _array_sha256(indices.astype(np.int32)),
                    "tokens_sha256": _array_sha256(tokens),
                    "candidate_weights_sha256": _array_sha256(inside),
                    "ring_weights_sha256": _array_sha256(rings),
                    "content_valid_sha256": _array_sha256(content_valid),
                }
                if any(manifest[key] != value for key, value in expected_hashes.items()):
                    raise ValueError(f"S9 feature hash mismatch: {split}/{row['image_id']}")
                feature_hashes_verified += 1
                if split != "val":
                    continue
                evidence_row = evidence_rows[row["image_id"]]
                evidence_path = output_root / "s9_likelihood_evidence" / evidence_row["evidence_path"]
                if sha256_file(evidence_path) != evidence_row["evidence_sha256"]:
                    raise ValueError(f"S9 evidence hash mismatch: {row['image_id']}")
                with np.load(evidence_path, allow_pickle=False) as evidence:
                    if (
                        not np.array_equal(evidence["candidate_indices"], indices.astype(np.int32))
                        or not np.array_equal(evidence["candidate_weights"], inside)
                        or not np.array_equal(evidence["ring_weights"], rings)
                        or not np.array_equal(evidence["content_valid"], content_valid)
                    ):
                        raise ValueError(f"S9 physical support evidence mismatch: {row['image_id']}")
                    observed_logits = np.asarray(evidence["token_logits"], dtype=np.float32)
                    reference_logits = _reference_head_logits(tokens, head_state, device)
                    reference_likelihood = _reference_likelihood(reference_logits, inside, rings)
                    observed_likelihood = np.asarray(evidence["candidate_likelihood"], dtype=np.float32)
                    maximum_head_delta = max(
                        maximum_head_delta,
                        float(np.max(np.abs(reference_logits - observed_logits))),
                    )
                    maximum_likelihood_delta = max(
                        maximum_likelihood_delta,
                        float(np.max(np.abs(reference_likelihood - observed_likelihood))),
                    )
                    if (
                        maximum_head_delta > MAXIMUM_HEAD_LOGIT_DELTA
                        or maximum_likelihood_delta > MAXIMUM_LIKELIHOOD_DELTA
                        or not np.array_equal(_rank(reference_likelihood), _rank(observed_likelihood))
                    ):
                        raise ValueError(f"S9 likelihood reproduction mismatch: {row['image_id']}")
                    validation_reproduced[row["image_id"]] = {
                        "candidate_indices": indices,
                        "likelihood": observed_likelihood,
                    }
    del encoder
    torch.cuda.empty_cache()
    if feature_hashes_verified != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise ValueError("S9 feature reproduction count mismatch")

    cache_freeze_path = cache_root / "selector_cache_freeze.json"
    if sha256_file(cache_freeze_path) != CACHE_FREEZE_SHA256:
        raise ValueError("S9 selector-cache freeze mismatch")
    _safety(cache_freeze, "selector cache")
    baseline_freeze_path = baseline_root / "prediction_freeze.json"
    if sha256_file(baseline_freeze_path) != BASELINE_FREEZE_SHA256:
        raise ValueError("S9 accepted baseline freeze mismatch")
    baseline_freeze = _json(baseline_freeze_path)
    _safety(baseline_freeze, "accepted baseline")
    baseline_predictions = {
        row["image_id"]: row
        for row in _rows(baseline_root / "predictions" / "prediction_manifest.csv")
    }
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
        or pair.get("source_commit") != source_commit
        or pair.get("protocol_sha256") != protocol_sha
        or pair.get("pair_physically_frozen_before_validation_gt") is not True
        or pair.get("collaborator_output_accessed") is not False
        or set(pair.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("S9 prediction-pair contract mismatch")
    diagnostics_path = output_root / "gt_blind_diagnostics.json"
    diagnostics = _json(diagnostics_path)
    _safety(diagnostics, "GT-blind diagnostics")
    identity_rows = {row["image_id"]: row for row in _rows(output_root / "baseline_identity.csv")}
    if len(identity_rows) != EXPECTED_VALIDATION:
        raise ValueError("S9 baseline identity cohort mismatch")

    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    score_rows: dict[str, dict[str, dict[str, str]]] = {}
    checkpoint_sha = sha256_file(checkpoint_path)
    for arm in ARMS:
        freeze_path = output_root / arm / "prediction_freeze.json"
        if sha256_file(freeze_path) != pair["arms"][arm]:
            raise ValueError(f"S9 {arm} freeze hash mismatch")
        freeze = _json(freeze_path)
        _safety(freeze, f"{arm} freeze")
        if (
            freeze.get("arm") != arm
            or freeze.get("source_commit") != source_commit
            or freeze.get("protocol_sha256") != protocol_sha
            or freeze.get("validation_predictions") != EXPECTED_VALIDATION
            or freeze.get("s9_checkpoint_sha256") != checkpoint_sha
            or freeze.get("feature_cache_manifest_sha256") != sha256_file(feature_manifest_path)
            or freeze.get("s9_likelihood_evidence_manifest_sha256")
            != sha256_file(evidence_manifest_path)
        ):
            raise ValueError(f"S9 {arm} freeze provenance mismatch")
        prediction_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "predictions" / "prediction_manifest.csv")
        }
        score_rows[arm] = {
            row["image_id"]: row
            for row in _rows(output_root / arm / "candidate_scores" / "candidate_score_manifest.csv")
        }
        if len(prediction_rows[arm]) != EXPECTED_VALIDATION or len(score_rows[arm]) != EXPECTED_VALIDATION:
            raise ValueError(f"S9 {arm} output cohort mismatch")

    correlations: list[float] = []
    changes = 0
    maximum_base_delta = 0.0
    physical_maps = 0
    physical_scores = 0
    val_cache_rows = {
        row["image_id"]: row
        for row in cache_manifest_rows
        if row["split"] == "val"
    }
    for split_row, payload in zip(split_rows["val"], accepted["val"]):
        image_id = split_row["image_id"]
        indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        masks = unpack_candidate_masks(payload["packed_masks"]).astype(np.float32)
        evidence_path = output_root / "s9_likelihood_evidence" / evidence_rows[image_id]["evidence_path"]
        with np.load(evidence_path, allow_pickle=False) as evidence:
            stored_base = np.asarray(evidence["baseline_logits"], dtype=np.float32)
            stored_upstream = np.asarray(evidence["upstream_scores"], dtype=np.float32)
            observed_base = common._base_logits(baseline_model, payload)
            maximum_base_delta = max(
                maximum_base_delta,
                float(np.max(np.abs(observed_base - stored_base))),
            )
            candidate_row = val_candidates[Path(image_id).stem]
            candidate_path = val_candidate_root / candidate_row["diagnostic_path"]
            cache_row = val_cache_rows[image_id]
            if sha256_file(candidate_path) != cache_row["candidate_payload_sha256"]:
                raise ValueError(f"S9 candidate payload differs: {image_id}")
            with np.load(candidate_path, allow_pickle=False) as candidate:
                upstream = np.asarray(candidate["selection_scores"], dtype=np.float32)[indices]
            if not np.array_equal(upstream, stored_upstream):
                raise ValueError(f"S9 upstream evidence differs: {image_id}")
            likelihood = validation_reproduced[image_id]["likelihood"]
            rank_inputs = tuple(
                torch.from_numpy(values.astype(np.float32, copy=False))[None]
                for values in (stored_base, upstream, likelihood)
            )
            rank_valid = torch.ones_like(rank_inputs[0], dtype=torch.bool)
            control = base.equal_rank_aggregate(rank_inputs[:2], rank_valid)[0].numpy()
            primary = base.equal_rank_aggregate(rank_inputs, rank_valid)[0].numpy()
            if (
                not np.array_equal(evidence["control_rank"], control.astype(np.float32))
                or not np.array_equal(evidence["primary_rank"], primary.astype(np.float32))
            ):
                raise ValueError(f"S9 rank evidence does not reproduce: {image_id}")
        if len(indices) > 1:
            correlation = float(np.corrcoef(control, primary)[0, 1])
            if np.isfinite(correlation):
                correlations.append(correlation)
        changes += int(int(np.argmax(control)) != int(np.argmax(primary)))
        expected = {ARMS[0]: control.astype(np.float32), ARMS[1]: primary.astype(np.float32)}
        for arm, values in expected.items():
            score_row = score_rows[arm][image_id]
            score_path = output_root / arm / "candidate_scores" / score_row["score_path"]
            if sha256_file(score_path) != score_row["score_sha256"]:
                raise ValueError(f"S9 {arm} score hash mismatch: {image_id}")
            with np.load(score_path, allow_pickle=False) as stored:
                if (
                    not np.array_equal(stored["candidate_indices"], indices.astype(np.int32))
                    or not np.array_equal(stored["candidate_logits"], values)
                ):
                    raise ValueError(f"S9 {arm} candidate scores differ: {image_id}")
            prediction = prediction_rows[arm][image_id]
            winner = int(np.argmax(values))
            if (
                int(prediction["selected_candidate_index"]) != int(indices[winner])
                or prediction.get("candidate_logit_recipe")
                != "within_image_equal_percentile_rank_no_tta"
                or abs(
                    float(prediction["bag_probability"])
                    - float(baseline_predictions[image_id]["bag_probability"])
                )
                > 1.0e-12
            ):
                raise ValueError(f"S9 {arm} prediction identity mismatch: {image_id}")
            map_path = output_root / arm / "predictions" / prediction["map_path"]
            saved_map = np.load(map_path, allow_pickle=False).astype(np.float32)
            expected_map = masks[winner] * float(prediction["bag_probability"])
            if (
                sha256_file(map_path) != prediction["map_sha256"]
                or not np.allclose(saved_map, expected_map, atol=5.0e-4, rtol=0)
            ):
                raise ValueError(f"S9 {arm} physical map mismatch: {image_id}")
            physical_scores += 1
            physical_maps += 1
        if identity_rows[image_id].get("identity_pass") != "1":
            raise ValueError(f"S9 baseline identity failed: {image_id}")
    del baseline_model
    torch.cuda.empty_cache()

    mean_correlation = float(np.mean(correlations))
    change_fraction = changes / EXPECTED_VALIDATION
    if (
        maximum_base_delta > 5.0e-5
        or abs(mean_correlation - float(diagnostics["mean_primary_control_rank_correlation"]))
        > 1.0e-12
        or abs(change_fraction - float(diagnostics["primary_changed_selection_fraction"]))
        > 1.0e-12
    ):
        raise ValueError("S9 independent GT-blind diagnostics do not reproduce")
    return {
        "audit_id": "independent_skelex_candidate_marginal_s9_output_v1",
        "status": "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_REPRODUCTION_PASS",
        "protocol_sha256": protocol_sha,
        "source_commit": source_commit,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "pair_freeze_sha256": sha256_file(pair_path),
        "feature_cache_operational_gate_sha256": sha256_file(feature_gate_path),
        "feature_cache_manifest_sha256": sha256_file(feature_manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "validation_predictions_per_arm": EXPECTED_VALIDATION,
        "feature_hashes_verified": feature_hashes_verified,
        "physical_likelihood_evidence_verified": len(evidence_rows),
        "physical_prediction_maps_verified": physical_maps,
        "physical_candidate_scores_verified": physical_scores,
        "maximum_base_logit_reproduction_delta": maximum_base_delta,
        "maximum_head_logit_reproduction_delta": maximum_head_delta,
        "maximum_likelihood_reproduction_delta": maximum_likelihood_delta,
        "mean_primary_control_rank_correlation": mean_correlation,
        "primary_changed_selection_fraction": change_fraction,
        "training_reexecuted": False,
        "checkpoint_selection": "final_epoch_only_no_validation_selection",
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--skelex-model-dir", type=Path, required=True)
    parser.add_argument("--skelex-config-sha256", required=True)
    parser.add_argument("--skelex-preprocessor-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S9 independent auditor requires exactly two CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"S9 independent auditor requires T4 x2, got {device_names}")
    result = audit_output(
        output_root=args.output_root.resolve(),
        protocol_path=args.protocol.resolve(),
        binding_path=args.launch_binding.resolve(),
        dataset_root=args.dataset_root.resolve(),
        split_path=args.split_manifest.resolve(),
        skelex_model_dir=args.skelex_model_dir.resolve(),
        skelex_config_sha256=args.skelex_config_sha256,
        skelex_preprocessor_sha256=args.skelex_preprocessor_sha256,
        train_candidate_root=args.train_candidate_root.resolve(),
        val_candidate_root=args.val_candidate_root.resolve(),
        cache_root=args.selector_cache_root.resolve(),
        baseline_root=args.baseline_root.resolve(),
        device=torch.device("cuda:0"),
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
