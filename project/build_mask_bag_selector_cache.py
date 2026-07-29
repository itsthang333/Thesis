from __future__ import annotations

"""Build a GT-blind selector cache only after reproducing frozen v3 outputs."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    sha256_file,
    verify_model_snapshot,
)
from models.mask_bag_selector_cache import (
    candidate_shape_features,
    encode_candidate_families,
    pack_candidate_masks,
    pairwise_overlap_geometry,
)
from models.mask_bag_selector_cache_io import (
    save_selector_cache_record,
    write_selector_cache_manifest,
)
from models.nominal_patch_memory import (
    make_seeded_random_projection,
    projection_sha256,
)
from models.rad_dino_mask_bag_mil import MaskBagMILConfig, RadDinoMaskBagMIL
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--train-candidate-root", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-pseudo-manifest-sha256", required=True)
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
    parser.add_argument("--maximum-candidates", type=int, default=81)
    parser.add_argument("--logit-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _verify_frozen_baseline(
    args: argparse.Namespace,
    val_rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    freeze_path = args.baseline_root / "prediction_freeze.json"
    if sha256_file(freeze_path) != args.expected_baseline_freeze_sha256:
        raise ValueError("baseline prediction-freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("source_commit") != args.expected_baseline_source_commit
        or freeze.get("protocol_sha256")
        != args.expected_baseline_protocol_sha256
        or freeze.get("split_sha256") != args.expected_split_sha256
        or freeze.get("checkpoint_sha256")
        != args.expected_baseline_checkpoint_sha256
        or freeze.get("train_candidate_manifest_sha256")
        != args.train_candidate_manifest_sha256
        or freeze.get("train_pseudo_manifest_sha256")
        != args.train_pseudo_manifest_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != args.val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256")
        != args.val_pseudo_manifest_sha256
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("baseline prediction-freeze provenance mismatch")
    checkpoint_path = args.baseline_root / "rad_dino_mask_bag_mil.pt"
    if sha256_file(checkpoint_path) != args.expected_baseline_checkpoint_sha256:
        raise ValueError("baseline checkpoint SHA-256 mismatch")

    manifest_path = args.baseline_root / "predictions" / "prediction_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("prediction_manifest_sha256"):
        raise ValueError("baseline prediction manifest differs from freeze")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    expected = {row["image_id"]: row for row in val_rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != set(expected):
        raise ValueError("baseline prediction manifest cohort mismatch")
    ordered: list[dict[str, str]] = []
    for image_id in expected:
        row = indexed[image_id]
        if (
            row["group_id"] != expected[image_id]["group_id"]
            or row["tumor"] != expected[image_id]["tumor"]
        ):
            raise ValueError(f"baseline prediction identity mismatch: {image_id}")
        map_path = args.baseline_root / "predictions" / row["map_path"]
        if not map_path.is_file() or sha256_file(map_path) != row["map_sha256"]:
            raise ValueError(f"baseline prediction map hash mismatch: {image_id}")
        values = np.load(map_path, allow_pickle=False)
        if (
            values.shape != (320, 320)
            or values.dtype != np.float16
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"baseline prediction map content mismatch: {image_id}")
        ordered.append(row)
    return freeze, ordered


def _load_candidate_provenance(
    candidate_root: Path,
    manifest_row: dict[str, str],
    *,
    returned_candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = candidate_root / manifest_row["diagnostic_path"]
    if sha256_file(path) != manifest_row["diagnostic_sha256"]:
        raise ValueError("candidate payload changed during cache construction")
    with np.load(path, allow_pickle=False) as payload:
        raw_count = len(payload["sam_masks"])
        if raw_count:
            components = payload["component_ids"].astype(np.int32).reshape(-1)
            modes = payload["prompt_modes"].astype("U64").reshape(-1)
            sources = payload["proposal_source_ids"].astype("U64").reshape(-1)
        else:
            components = np.asarray([-1], dtype=np.int32)
            modes = np.asarray(["fallback"], dtype="U64")
            sources = np.asarray(["fallback"], dtype="U64")
    if not (
        len(components)
        == len(modes)
        == len(sources)
        == returned_candidate_count
    ):
        raise ValueError("candidate provenance differs from reproduced bag")
    return components, modes, sources


def _load_baseline_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RadDinoMaskBagMIL, MaskBagMILConfig]:
    checkpoint_path = args.baseline_root / "rad_dino_mask_bag_mil.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != args.expected_baseline_source_commit
        or checkpoint.get("protocol_sha256")
        != args.expected_baseline_protocol_sha256
        or checkpoint.get("split_sha256") != args.expected_split_sha256
        or checkpoint.get("validation_gt_read") is not False
        or checkpoint.get("consumer_trained") is not False
        or checkpoint.get("test_evaluated") is not False
    ):
        raise ValueError("baseline checkpoint provenance mismatch")
    config = MaskBagMILConfig(**checkpoint["config"])
    if (
        config.token_dim != 128
        or config.token_layers != 3
        or config.minimum_grid_mass != 0.25
    ):
        raise ValueError("baseline checkpoint configuration mismatch")
    model = RadDinoMaskBagMIL(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).to(device).eval()
    return model, config


def _compare_reproduction(
    args: argparse.Namespace,
    baseline_rows: list[dict[str, str]],
    reproduction_root: Path,
) -> dict[str, object]:
    manifest_path = reproduction_root / "predictions" / "prediction_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reproduced = list(csv.DictReader(handle))
    if len(reproduced) != 371:
        raise RuntimeError("reproduction manifest must contain 371 rows")
    indexed = {row["image_id"]: row for row in reproduced}
    maximum_logit_delta = 0.0
    maximum_bag_logit_delta = 0.0
    maximum_probability_delta = 0.0
    map_hashes = 0
    for baseline in baseline_rows:
        current = indexed.get(baseline["image_id"])
        if current is None:
            raise ValueError("reproduction omits a baseline image")
        for field in (
            "group_id",
            "tumor",
            "candidate_payload_sha256",
            "candidate_count",
            "selected_candidate_index",
            "candidate_logit_tta",
            "fallback_count",
            "map_path",
            "map_sha256",
        ):
            if current[field] != baseline[field]:
                raise ValueError(
                    f"baseline reproduction mismatch: {baseline['image_id']} {field}"
                )
        maximum_logit_delta = max(
            maximum_logit_delta,
            abs(
                float(current["selected_candidate_logit"])
                - float(baseline["selected_candidate_logit"])
            ),
        )
        maximum_bag_logit_delta = max(
            maximum_bag_logit_delta,
            abs(float(current["bag_logit"]) - float(baseline["bag_logit"])),
        )
        maximum_probability_delta = max(
            maximum_probability_delta,
            abs(
                float(current["bag_probability"])
                - float(baseline["bag_probability"])
            ),
        )
        map_path = reproduction_root / "predictions" / current["map_path"]
        if sha256_file(map_path) != current["map_sha256"]:
            raise ValueError("reproduced physical map hash mismatch")
        map_hashes += 1
    if (
        maximum_logit_delta > args.logit_tolerance
        or maximum_bag_logit_delta > args.logit_tolerance
        or maximum_probability_delta > args.logit_tolerance
    ):
        raise ValueError("baseline reproduction numerical tolerance exceeded")
    return {
        "validation_images": 371,
        "selected_indices_exact": 371,
        "map_hashes_exact": map_hashes,
        "maximum_selected_logit_delta": maximum_logit_delta,
        "maximum_bag_logit_delta": maximum_bag_logit_delta,
        "maximum_bag_probability_delta": maximum_probability_delta,
        "logit_tolerance": args.logit_tolerance,
        "reproduction_manifest_sha256": sha256_file(manifest_path),
    }


def _serialize_cache_split(
    args: argparse.Namespace,
    records: list[dict[str, object]],
    candidate_rows: dict[str, dict[str, str]],
    candidate_root: Path,
    *,
    split: str,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        manifest_row = candidate_rows[Path(str(record["image_id"])).stem]
        masks, _metadata, _scores, fallback_flags = _load_candidate_payload(
            candidate_root,
            manifest_row,
            maximum_candidates=args.maximum_candidates,
        )
        components, modes, sources = _load_candidate_provenance(
            candidate_root,
            manifest_row,
            returned_candidate_count=len(masks),
        )
        kept = np.asarray(record["kept_indices"], dtype=np.int32)
        kept_masks = masks[kept]
        families, _family_table = encode_candidate_families(
            components,
            modes,
            sources,
            kept_indices=kept,
            fallback_flags=fallback_flags,
        )
        iou, containment, distance = pairwise_overlap_geometry(kept_masks)
        relative = (
            Path("records")
            / split
            / f"{index:04d}_{Path(str(record['image_id'])).stem}.npz"
        )
        saved = save_selector_cache_record(
            args.output_dir / relative,
            descriptors=np.asarray(record["descriptors"]),
            flipped_descriptors=np.asarray(record["flipped_descriptors"]),
            candidate_indices=kept,
            family_ids=families,
            component_ids=components[kept],
            prompt_modes=modes[kept],
            proposal_source_ids=sources[kept],
            fallback_flags=fallback_flags[kept],
            shape_features=candidate_shape_features(kept_masks),
            pairwise_iou=iou,
            pairwise_containment=containment,
            pairwise_distance=distance,
            packed_masks=pack_candidate_masks(kept_masks) if split == "val" else None,
        )
        output_rows.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "tumor": record["label"],
                "split": split,
                "candidate_payload_sha256": record["candidate_payload_sha256"],
                **saved,
                "cache_path": str(relative),
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    if (
        args.input_size != 448
        or args.projection_dim != 128
        or args.projection_seed != 42
        or args.maximum_candidates != 81
        or args.logit_tolerance != 5.0e-6
    ):
        raise ValueError("selector cache builder differs from the frozen contract")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    train_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="train",
    )
    val_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split="val",
    )
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("frozen train/validation cohort mismatch")
    baseline_freeze, baseline_rows = _verify_frozen_baseline(args, val_rows)
    train_candidates, train_candidate_audit = _audit_candidate_input(
        args.train_candidate_root,
        train_rows,
        split="train",
        expected_manifest_sha256=args.train_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.train_pseudo_manifest_sha256,
    )
    val_candidates, val_candidate_audit = _audit_candidate_input(
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
        raise RuntimeError("selector cache builder requires exactly two CUDA devices")
    device_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in device_names):
        raise RuntimeError(f"selector cache builder requires T4 x2, got {device_names}")
    device = torch.device("cuda:0")
    projection = make_seeded_random_projection(
        input_dim=768,
        output_dim=args.projection_dim,
        seed=args.projection_seed,
    )
    if projection_sha256(projection) != baseline_freeze["projection_sha256"]:
        raise ValueError("RAD-DINO projection differs from frozen baseline")
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
    train_cache = build_descriptor_cache(
        train_rows,
        train_candidates,
        args.train_candidate_root,
        encoder,
        config,
        args,
        device,
        split="train",
    )
    val_cache = build_descriptor_cache(
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

    baseline_model, baseline_config = _load_baseline_model(args, device)
    if asdict(baseline_config) != asdict(config):
        raise ValueError("reconstructed descriptor config differs from checkpoint")
    reproduction_root = args.output_dir / "baseline_reproduction"
    reproduction_args = SimpleNamespace(
        output_dir=reproduction_root,
        maximum_candidates=args.maximum_candidates,
    )
    write_validation_predictions(
        baseline_model,
        val_cache,
        val_candidates,
        args.val_candidate_root,
        reproduction_args,
        device,
    )
    reproduction_audit = _compare_reproduction(
        args,
        baseline_rows,
        reproduction_root,
    )
    del baseline_model
    torch.cuda.empty_cache()

    cache_rows = _serialize_cache_split(
        args,
        train_cache,
        train_candidates,
        args.train_candidate_root,
        split="train",
    )
    cache_rows.extend(
        _serialize_cache_split(
            args,
            val_cache,
            val_candidates,
            args.val_candidate_root,
            split="val",
        )
    )
    cache_manifest = write_selector_cache_manifest(args.output_dir, cache_rows)
    if (
        cache_manifest["train_records"] != 2981
        or cache_manifest["validation_records"] != 371
    ):
        raise RuntimeError("selector cache manifest cohort mismatch")

    reproduction_path = args.output_dir / "baseline_reproduction_audit.json"
    reproduction_path.write_text(
        json.dumps(reproduction_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_snapshot": model_snapshot,
        "projection_sha256": projection_sha256(projection),
        "baseline_source_commit": args.expected_baseline_source_commit,
        "baseline_protocol_sha256": args.expected_baseline_protocol_sha256,
        "baseline_prediction_freeze_sha256": args.expected_baseline_freeze_sha256,
        "baseline_checkpoint_sha256": args.expected_baseline_checkpoint_sha256,
        "baseline_prediction_manifest_sha256": baseline_freeze[
            "prediction_manifest_sha256"
        ],
        "train_candidate_manifest_sha256": args.train_candidate_manifest_sha256,
        "train_pseudo_manifest_sha256": args.train_pseudo_manifest_sha256,
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "selector_cache_manifest_sha256": cache_manifest["manifest_sha256"],
        "baseline_reproduction_audit_sha256": sha256_file(reproduction_path),
        "cohort": {"train": 2981, "validation": 371},
        "validation_selected_indices_reproduced": 371,
        "validation_map_hashes_reproduced": 371,
        "train_masks_discarded": True,
        "validation_masks_bitpacked": True,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "selector_cache_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "run_id": "btxrd_mask_bag_selector_cache_v1",
        "cache_freeze_sha256": sha256_file(freeze_path),
        "cache": cache_manifest,
        "baseline_reproduction": reproduction_audit,
        "candidate_inputs": {
            "train": train_candidate_audit,
            "validation": val_candidate_audit,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": device_names,
            "encoder_data_parallel": True,
        },
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
