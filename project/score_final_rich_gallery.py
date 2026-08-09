from __future__ import annotations

"""Score every frozen rich-gallery candidate without opening spatial GT."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import torch

from evaluation.frozen_test_guard import verify_frozen_test_config
from gpu_runtime import place_frozen_encoder, require_cuda_runtime
from frozen_io import load_split_rows_without_annotations, sha256_file, verify_model_snapshot
from models.nominal_patch_memory import make_seeded_random_projection, projection_sha256
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL, smooth_mil_pool
from run_rad_dino_mask_bag_mil_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    _audit_candidate_input,
    build_descriptor_cache,
    seed_everything,
)


EXPECTED_COUNTS = {"train": 2981, "val": 371, "test": 373}
EVIDENCE_FIELDS = (
    "image_id", "group_id", "tumor", "candidate_payload_sha256",
    "candidate_count", "selected_candidate_index", "selected_candidate_logit",
    "bag_logit", "bag_probability", "evidence_path", "evidence_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--pseudo-manifest-sha256", required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-g1-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--maximum-candidates", type=int, default=243)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _candidate_metadata(root: Path, row: dict[str, str], candidate_count: int) -> dict[str, np.ndarray]:
    path = root / row["diagnostic_path"]
    if sha256_file(path) != row["diagnostic_sha256"]:
        raise ValueError("candidate payload changed during final scoring")
    with np.load(path, allow_pickle=False) as payload:
        if len(payload["sam_masks"]):
            result = {
                "selection_scores": payload["selection_scores"].astype(np.float32).reshape(-1),
                "proposal_source_ids": payload["proposal_source_ids"].astype("U96").reshape(-1),
            }
        else:
            result = {
                "selection_scores": np.zeros(1, dtype=np.float32),
                "proposal_source_ids": np.asarray(["fallback"], dtype="U96"),
            }
    if any(len(values) != candidate_count for values in result.values()):
        raise ValueError("candidate metadata count differs from G1 candidate bag")
    return result


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> str:
    expected = EXPECTED_COUNTS[rows[0]["split"]]
    if len(rows) != expected or len({str(row["image_id"]) for row in rows}) != expected:
        raise ValueError("final G1 evidence cohort is incomplete")
    path = root / "descriptor_evidence_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVIDENCE_FIELDS))
        writer.writeheader()
        writer.writerows([{key: row[key] for key in EVIDENCE_FIELDS} for row in rows])
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    if args.input_size != 448 or args.projection_dim != 128 or args.projection_seed != 42 or not 1 <= args.maximum_candidates <= 567:
        raise ValueError("scoring geometry differs from the frozen G1 contract")
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_artifacts={"g1_checkpoint": args.g1_checkpoint},
    )
    if sha256_file(args.g1_checkpoint) != args.expected_g1_checkpoint_sha256:
        raise ValueError("G1 checkpoint SHA-256 mismatch")
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
        allow_test=args.split == "test",
    )
    if len(rows) != EXPECTED_COUNTS[args.split]:
        raise ValueError("canonical scoring cohort count differs")
    candidates, candidate_audit = _audit_candidate_input(
        args.candidate_root,
        rows,
        split=args.split,
        expected_manifest_sha256=args.candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.pseudo_manifest_sha256,
    )
    model_snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_artifacts={
            "g1_checkpoint": args.g1_checkpoint,
            "rad_dino_weight": args.model_dir / "model.safetensors",
        },
    )
    runtime = require_cuda_runtime()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import transformers
    from transformers import AutoModel
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("unexpected transformers version")
    device = runtime.primary_device
    projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder = place_frozen_encoder(
        ProjectedMultiLayerEncoder(backbone, torch.from_numpy(projection)),
        runtime,
    )
    config = MaskBagMILConfig(token_dim=args.projection_dim, token_layers=len(SELECTED_HIDDEN_LAYERS))
    cache = build_descriptor_cache(rows, candidates, args.candidate_root, encoder, config, args, device, split=args.split)
    del encoder, backbone
    torch.cuda.empty_cache()

    checkpoint = torch.load(args.g1_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = MaskBagMILConfig(**checkpoint["config"])
    if asdict(checkpoint_config) != asdict(config):
        raise ValueError("G1 checkpoint descriptor configuration differs")
    model = RadDinoMaskBagMIL(checkpoint_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).to(device).eval()

    evidence_root = args.output_dir / "descriptor_evidence"
    evidence_rows: list[dict[str, object]] = []
    for index, record in enumerate(cache):
        kept = np.asarray(record["kept_indices"], dtype=np.int32)
        descriptors = np.asarray(record["descriptors"], dtype=np.float32)
        flipped = np.asarray(record["flipped_descriptors"], dtype=np.float32)
        valid = torch.ones((1, len(kept)), dtype=torch.bool, device=device)
        with torch.inference_mode():
            original, _ = model.score_descriptors(torch.from_numpy(descriptors)[None].to(device), valid)
            mirrored, _ = model.score_descriptors(torch.from_numpy(flipped)[None].to(device), valid)
            logits = 0.5 * (original + mirrored)
            bag_logit = smooth_mil_pool(logits, valid, temperature=model.config.bag_temperature)[0]
        logits_np = logits[0].cpu().numpy().astype(np.float32)
        image_id = str(record["image_id"])
        candidate_row = candidates[Path(image_id).stem]
        extras = _candidate_metadata(
            args.candidate_root,
            candidate_row,
            int(candidate_row["candidate_count"]),
        )
        relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        path = evidence_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int32),
            candidate_indices=kept,
            candidate_logits=logits_np,
            selection_scores=extras["selection_scores"][kept],
            proposal_source_ids=extras["proposal_source_ids"][kept],
        )
        selected_local = int(np.argmax(logits_np))
        evidence_rows.append({
            "split": args.split,
            "image_id": image_id,
            "group_id": record["group_id"],
            "tumor": record["label"],
            "candidate_payload_sha256": record["candidate_payload_sha256"],
            "candidate_count": len(kept),
            "selected_candidate_index": int(kept[selected_local]),
            "selected_candidate_logit": float(logits_np[selected_local]),
            "bag_logit": float(bag_logit.item()),
            "bag_probability": float(torch.sigmoid(bag_logit).item()),
            "evidence_path": str(relative),
            "evidence_sha256": sha256_file(path),
        })
    evidence_manifest_sha = _write_manifest(args.output_dir, evidence_rows)
    freeze = {
        "stage": "rich_gallery_g1_all_candidate_score_freeze_v1",
        "cohort_split": args.split,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "baseline_checkpoint_sha256": args.expected_g1_checkpoint_sha256,
        "baseline_prediction_manifest_sha256": None,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "pseudo_manifest_sha256": args.pseudo_manifest_sha256,
        "descriptor_evidence_manifest_sha256": evidence_manifest_sha,
        "images": len(rows),
        "validation_images": len(rows) if args.split == "val" else 0,
        "maximum_candidates": args.maximum_candidates,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": len(rows) if args.split == "test" else 0,
        "test_evaluated": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": list(runtime.device_names),
            "encoder_data_parallel": runtime.encoder_data_parallel,
        },
        "candidate_input_audit": candidate_audit,
    }
    freeze_path = args.output_dir / "diagnostic_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "diagnostic_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
