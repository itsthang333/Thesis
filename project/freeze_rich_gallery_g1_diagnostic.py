from __future__ import annotations

"""Freeze every G1 candidate score/descriptor before spatial-GT diagnosis."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from build_mask_bag_selector_cache import (
    _compare_reproduction,
    _load_baseline_model,
)
from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
    verify_model_snapshot,
)
from models.mask_bag_score_evidence import (
    save_candidate_score_evidence,
    write_candidate_score_manifest,
)
from models.mask_bag_selector_cache import candidate_shape_features
from models.nominal_patch_memory import (
    make_seeded_random_projection,
    projection_sha256,
)
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, smooth_mil_pool
from run_rad_dino_mask_bag_mil_probe import (
    EXPECTED_TRANSFORMERS_VERSION,
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    _audit_candidate_input,
    _load_candidate_payload,
    build_descriptor_cache,
    seed_everything,
    write_validation_predictions,
)


EVIDENCE_FIELDS = (
    "image_id",
    "group_id",
    "tumor",
    "candidate_payload_sha256",
    "candidate_count",
    "selected_candidate_index",
    "selected_candidate_logit",
    "bag_logit",
    "bag_probability",
    "evidence_path",
    "evidence_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-baseline-freeze-sha256", required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--expected-baseline-source-commit", required=True)
    parser.add_argument("--expected-baseline-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--maximum-candidates", type=int, default=243)
    parser.add_argument("--logit-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _verify_baseline(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    freeze_path = args.baseline_root / "prediction_freeze.json"
    checkpoint_path = args.baseline_root / "rad_dino_mask_bag_mil.pt"
    manifest_path = args.baseline_root / "predictions" / "prediction_manifest.csv"
    if sha256_file(freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("G1 prediction freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != args.expected_baseline_source_commit
        or freeze.get("protocol_sha256")
        != args.expected_baseline_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("checkpoint_sha256")
        != args.expected_baseline_checkpoint_sha256
        or freeze.get("prediction_manifest_sha256") != sha256_file(manifest_path)
        or freeze.get("val_candidate_manifest_sha256")
        != args.val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.val_pseudo_manifest_sha256
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 prediction freeze provenance mismatch")
    if sha256_file(checkpoint_path) != args.expected_baseline_checkpoint_sha256:
        raise ValueError("G1 checkpoint SHA-256 mismatch")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in val_rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != set(expected):
        raise ValueError("G1 prediction cohort mismatch")
    ordered = []
    for image_id, expected_row in expected.items():
        row = indexed[image_id]
        if (
            row["group_id"] != expected_row["group_id"]
            or row["tumor"] != expected_row["tumor"]
        ):
            raise ValueError(f"G1 prediction identity mismatch: {image_id}")
        ordered.append(row)
    return freeze, ordered


def _candidate_arrays(
    root: Path,
    row: dict[str, str],
    *,
    candidate_count: int,
) -> dict[str, np.ndarray]:
    path = root / row["diagnostic_path"]
    if sha256_file(path) != row["diagnostic_sha256"]:
        raise ValueError("candidate payload changed during G1 diagnosis")
    with np.load(path, allow_pickle=False) as payload:
        raw_count = len(payload["sam_masks"])
        if raw_count:
            result = {
                "sam_scores": payload["sam_scores"].astype(np.float32).reshape(-1),
                "selection_scores": payload["selection_scores"].astype(np.float32).reshape(-1),
                "classifier_causal_scores": payload["classifier_causal_scores"].astype(np.float32).reshape(-1),
                "component_ids": payload["component_ids"].astype(np.int32).reshape(-1),
                "prompt_modes": payload["prompt_modes"].astype("U64").reshape(-1),
                "proposal_source_ids": payload["proposal_source_ids"].astype("U96").reshape(-1),
            }
        else:
            result = {
                "sam_scores": np.zeros(1, dtype=np.float32),
                "selection_scores": np.zeros(1, dtype=np.float32),
                "classifier_causal_scores": np.zeros(1, dtype=np.float32),
                "component_ids": np.asarray([-1], dtype=np.int32),
                "prompt_modes": np.asarray(["fallback"], dtype="U64"),
                "proposal_source_ids": np.asarray(["fallback"], dtype="U96"),
            }
    if any(len(values) != candidate_count for values in result.values()):
        raise ValueError("candidate metadata count differs from G1 candidate bag")
    return result


def _write_evidence_manifest(
    root: Path,
    rows: list[dict[str, object]],
) -> str:
    if len(rows) != 371 or len({str(row["image_id"]) for row in rows}) != 371:
        raise ValueError("G1 evidence manifest must contain 371 unique images")
    path = root / "descriptor_evidence_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVIDENCE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    if (
        args.input_size != 448
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.maximum_candidates != 243
        or args.logit_tolerance != 5.0e-6
    ):
        raise ValueError("G1 diagnostic differs from the frozen rich-gallery contract")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(val_rows) != 371:
        raise RuntimeError("frozen BTXRD validation cohort mismatch")
    baseline_freeze, baseline_rows = _verify_baseline(args, val_rows)
    val_candidates, candidate_audit = _audit_candidate_input(
        args.val_candidate_root,
        val_rows,
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    model_snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("unexpected transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("G1 diagnostic requires exactly two visible GPUs")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"G1 diagnostic requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    if projection_sha256(projection) != baseline_freeze["projection_sha256"]:
        raise ValueError("G1 random projection SHA-256 mismatch")
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder: nn.Module = ProjectedMultiLayerEncoder(
        backbone,
        torch.from_numpy(projection),
    ).to(device)
    encoder = nn.DataParallel(encoder, device_ids=[0, 1], output_device=0).eval()
    config = MaskBagMILConfig(
        token_dim=args.projection_dim,
        token_layers=len(SELECTED_HIDDEN_LAYERS),
    )
    cache = build_descriptor_cache(
        val_rows,
        val_candidates,
        args.val_candidate_root,
        encoder,
        config,
        args,
        device,
        split="val",
    )
    del encoder, backbone
    torch.cuda.empty_cache()
    model, checkpoint_config = _load_baseline_model(args, device)
    if asdict(checkpoint_config) != asdict(config):
        raise ValueError("G1 descriptor config differs from checkpoint")

    reproduction_root = args.output_dir / "baseline_reproduction"
    write_validation_predictions(
        model,
        cache,
        val_candidates,
        args.val_candidate_root,
        SimpleNamespace(
            output_dir=reproduction_root,
            maximum_candidates=args.maximum_candidates,
        ),
        device,
    )
    reproduction_audit = _compare_reproduction(
        args,
        baseline_rows,
        reproduction_root,
    )
    reproduction_audit_path = args.output_dir / "baseline_reproduction_audit.json"
    reproduction_audit_path.write_text(
        json.dumps(reproduction_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    score_root = args.output_dir / "candidate_scores"
    descriptor_root = args.output_dir / "descriptor_evidence"
    score_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    baseline_by_id = {row["image_id"]: row for row in baseline_rows}
    for index, record in enumerate(cache):
        descriptors = np.asarray(record["descriptors"], dtype=np.float16)
        flipped = np.asarray(record["flipped_descriptors"], dtype=np.float16)
        kept = np.asarray(record["kept_indices"], dtype=np.int32)
        valid = torch.ones((1, len(kept)), dtype=torch.bool, device=device)
        with torch.inference_mode():
            original_logits, _ = model.score_descriptors(
                torch.from_numpy(descriptors.astype(np.float32))[None].to(device),
                valid,
            )
            flipped_logits, _ = model.score_descriptors(
                torch.from_numpy(flipped.astype(np.float32))[None].to(device),
                valid,
            )
            logits = 0.5 * (original_logits + flipped_logits)
            bag_logit = smooth_mil_pool(
                logits,
                valid,
                temperature=model.config.bag_temperature,
            )[0]
        original_np = original_logits[0].cpu().numpy().astype(np.float32)
        flipped_np = flipped_logits[0].cpu().numpy().astype(np.float32)
        logits_np = logits[0].cpu().numpy().astype(np.float32)
        image_id = str(record["image_id"])
        score_relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        saved_score = save_candidate_score_evidence(
            score_root / score_relative,
            candidate_indices=kept,
            candidate_logits=logits_np,
        )
        candidate_row = val_candidates[Path(image_id).stem]
        masks, metadata, _sam_scores, fallback = _load_candidate_payload(
            args.val_candidate_root,
            candidate_row,
            maximum_candidates=args.maximum_candidates,
        )
        extras = _candidate_arrays(
            args.val_candidate_root,
            candidate_row,
            candidate_count=len(masks),
        )
        evidence_relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        evidence_path = descriptor_root / evidence_relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            evidence_path,
            schema_version=np.asarray(1, dtype=np.int32),
            descriptors=descriptors,
            flipped_descriptors=flipped,
            candidate_indices=kept,
            original_logits=original_np,
            flipped_logits=flipped_np,
            candidate_logits=logits_np,
            descriptor_metadata=metadata[kept].astype(np.float32),
            shape_features=candidate_shape_features(masks[kept]),
            fallback_flags=fallback[kept],
            **{name: values[kept] for name, values in extras.items()},
        )
        selected = int(saved_score["selected_candidate_index"])
        baseline = baseline_by_id[image_id]
        probability = float(torch.sigmoid(bag_logit).item())
        if (
            selected != int(baseline["selected_candidate_index"])
            or abs(float(saved_score["selected_candidate_logit"]) - float(baseline["selected_candidate_logit"])) > args.logit_tolerance
            or abs(float(bag_logit.item()) - float(baseline["bag_logit"])) > args.logit_tolerance
            or abs(probability - float(baseline["bag_probability"])) > args.logit_tolerance
        ):
            raise ValueError(f"G1 all-candidate evidence does not reproduce winner: {image_id}")
        common = {
            "image_id": image_id,
            "group_id": record["group_id"],
            "tumor": record["label"],
            "candidate_payload_sha256": record["candidate_payload_sha256"],
            "candidate_count": len(kept),
            "selected_candidate_index": selected,
            "selected_candidate_logit": saved_score["selected_candidate_logit"],
        }
        score_rows.append(
            {
                **common,
                "score_path": str(score_relative),
                "score_sha256": saved_score["score_sha256"],
            }
        )
        evidence_rows.append(
            {
                **common,
                "bag_logit": float(bag_logit.item()),
                "bag_probability": probability,
                "evidence_path": str(evidence_relative),
                "evidence_sha256": sha256_file(evidence_path),
            }
        )
    score_manifest = write_candidate_score_manifest(score_root, score_rows)
    evidence_manifest_sha256 = _write_evidence_manifest(
        args.output_dir,
        evidence_rows,
    )
    freeze = {
        "stage": "rich_gallery_g1_all_candidate_score_freeze_v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "baseline_source_commit": args.expected_baseline_source_commit,
        "baseline_protocol_sha256": args.expected_baseline_protocol_sha256,
        "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "baseline_prediction_manifest_sha256": baseline_freeze["prediction_manifest_sha256"],
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "candidate_score_manifest_sha256": score_manifest["manifest_sha256"],
        "descriptor_evidence_manifest_sha256": evidence_manifest_sha256,
        "baseline_reproduction_audit_sha256": sha256_file(reproduction_audit_path),
        "validation_images": 371,
        "maximum_candidates": 243,
        "validation_gt_read": False,
        "spatial_ground_truth_used": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "diagnostic_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        **freeze,
        "diagnostic_freeze_sha256": sha256_file(freeze_path),
        "candidate_input_audit": candidate_audit,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_names": device_names,
        },
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
