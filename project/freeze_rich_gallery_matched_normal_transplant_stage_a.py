from __future__ import annotations

"""Freeze matched-normal transplant scores and layerwise diagnostics.

This Stage-A runner is prediction-only.  It reads canonical image labels and
raw radiographs, but never imports or opens a spatial annotation.  Every
candidate score and selected mask index is serialized before Stage B.
"""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets.btxrd import resolve_btxrd_root
from mae_reconstruction_io import load_split_rows_without_annotations, sha256_file
from models.classifier import DenseNet121AnatomyClassifier
from models.matched_normal_candidate_transplant import (
    DENSENET_DIAGNOSTIC_STAGES,
    frozen_selector_panel,
    matched_transplant_layerwise_scores,
    select_normal_reference_pairs,
    select_random_normal_reference_pairs,
)
from models.rich_gallery_g2_objective import average_percentile_rank, stable_select
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


VARIANTS = (
    "g1_upstream_baseline",
    "transplant_only",
    "baseline_transplant_equal",
    "baseline_transplant_three_to_one",
    "baseline_random_control_three_to_one",
)
IMAGE_SIZE = 448
PAIR_COUNT = 2
FEATHER_KERNEL = 7
RANDOM_CONTROL_SEED = 20260802


def canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    raise ValueError(f"unknown rich-gallery source: {value!r}")


def _verify_g1_stage_a(
    root: Path,
    *,
    expected_freeze_sha256: str,
    expected_split_sha256: str,
    expected_val_manifest_sha256: str,
    expected_val_pseudo_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    freeze_path = root / "prediction_freeze.json"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise ValueError("G1/G2 Stage-A freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "rich_gallery_g2_selector_pair_stage_a_v1"
        or freeze.get("split_sha256") != expected_split_sha256
        or freeze.get("val_candidate_manifest_sha256") != expected_val_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256") != expected_val_pseudo_sha256
        or freeze.get("g1_reproduction_max_selected_index_delta") != 0
        or freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1/G2 Stage-A safety/provenance mismatch")
    manifest_path = root / "stage_a_selection_manifest.csv"
    if sha256_file(manifest_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError("G1/G2 selection manifest changed")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["image_id"]: row
        for row in rows
        if row["variant"] == "g1_frozen__rank_fusion"
    }
    if len(selected) != 371:
        raise ValueError("G1 rank-fusion Stage-A cohort mismatch")
    return freeze, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--g1-stage-a-root", type=Path, required=True)
    parser.add_argument("--expected-g1-stage-a-freeze-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--expected-val-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-chunk-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_classifier(path: Path, expected_sha256: str, device: torch.device):
    if sha256_file(path) != expected_sha256:
        raise ValueError("classifier448 checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("task") != "multi-label"
        or payload.get("target_columns") != ["tumor"]
        or int(payload.get("image_size", 0)) != IMAGE_SIZE
        or int(payload.get("num_classes", 0)) != 1
        or payload.get("normalization") != "imagenet"
    ):
        raise ValueError("classifier448 checkpoint contract mismatch")
    model = DenseNet121AnatomyClassifier(
        num_classes=1,
        pretrained=False,
        dropout=0.2,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def _tensor_payload(
    prefix: str,
    result: dict[str, torch.Tensor | tuple[str, ...]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if isinstance(value, tuple):
            payload[f"{prefix}_{key}"] = np.asarray(value, dtype="U32")
        elif isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite layerwise output: {prefix}_{key}")
            payload[f"{prefix}_{key}"] = array.astype(np.float32)
        else:
            raise TypeError(f"unsupported layerwise output: {key}")
    return payload


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("matched-normal Stage-A output must not exist")
    if args.batch_size <= 0 or args.candidate_chunk_size <= 0:
        raise ValueError("batch/chunk sizes must be positive")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("canonical split SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("matched-normal Stage A requires a CUDA device")
    device = torch.device("cuda:0")
    model, checkpoint = _load_classifier(
        args.classifier_checkpoint,
        args.expected_classifier_sha256,
        device,
    )
    dataset_root = resolve_btxrd_root(args.dataset_root)
    image_root = dataset_root / "images"
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
    normal_rows = [row for row in train_rows if row["tumor"] == "0"]
    if len(train_rows) != 2981 or len(normal_rows) != 1493 or len(val_rows) != 371:
        raise ValueError("canonical train/normal/validation counts mismatch")
    all_rows = {row["image_id"]: row for row in train_rows + val_rows}
    verified_images: set[str] = set()

    @lru_cache(maxsize=24)
    def load_image(image_id: str) -> torch.Tensor:
        row = all_rows[image_id]
        path = image_root / image_id
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical image: {path}")
        if image_id not in verified_images:
            if sha256_file(path) != row["image_sha256"]:
                raise ValueError(f"canonical image SHA-256 mismatch: {image_id}")
            verified_images.add(image_id)
        return _image_tensor(path)

    g1_freeze, g1_rows = _verify_g1_stage_a(
        args.g1_stage_a_root,
        expected_freeze_sha256=args.expected_g1_stage_a_freeze_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_val_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_val_pseudo_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    candidate_rows, candidate_audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[row["image_id"] for row in val_rows],
        split="val",
        expected_manifest_sha256=args.expected_val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.expected_val_pseudo_manifest_sha256,
    )
    if candidate_audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("candidate-diagnostic validation cohort mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_root = args.output_dir / "stage_a_scores"
    score_root.mkdir()
    reference_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    baseline_reproduced = 0
    start_time = time.time()
    for image_number, row in enumerate(val_rows, start=1):
        image_id = row["image_id"]
        stem = Path(image_id).stem
        matched_pairs = select_normal_reference_pairs(row, normal_rows, pair_count=PAIR_COUNT)
        random_pairs = select_random_normal_reference_pairs(
            row,
            normal_rows,
            pair_count=PAIR_COUNT,
            seed=RANDOM_CONTROL_SEED,
        )
        for arm, pairs in (("matched", matched_pairs), ("random", random_pairs)):
            for pair_index, pair in enumerate(pairs):
                reference_rows.append(
                    {
                        "image_id": image_id,
                        "group_id": row["group_id"],
                        "arm": arm,
                        "pair_index": pair_index,
                        "recipient_image_id": pair.recipient_image_id,
                        "recipient_group_id": pair.recipient_group_id,
                        "recipient_sha256": all_rows[pair.recipient_image_id]["image_sha256"],
                        "sham_image_id": pair.sham_image_id,
                        "sham_group_id": pair.sham_group_id,
                        "sham_sha256": all_rows[pair.sham_image_id]["image_sha256"],
                    }
                )
        frozen = g1_rows[image_id]
        g1_path = args.g1_stage_a_root / frozen["score_path"]
        if sha256_file(g1_path) != frozen["score_sha256"]:
            raise ValueError(f"G1 Stage-A score payload changed: {image_id}")
        candidate_row = candidate_rows[stem]
        candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
        if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(g1_path, allow_pickle=False) as scored:
            candidate_indices = scored["candidate_indices"].astype(np.int64)
            g1_logits = scored["g1_frozen_candidate_logits"].astype(np.float64)
            upstream = scored["upstream_scores"].astype(np.float64)
        with np.load(candidate_path, allow_pickle=False) as candidate:
            masks_320 = candidate["sam_masks"].astype(np.float32)[candidate_indices]
            sources = candidate["proposal_source_ids"].astype(str)[candidate_indices]
        if not (len(candidate_indices) == len(g1_logits) == len(upstream) == len(masks_320)):
            raise ValueError(f"G1/candidate arrays are misaligned: {image_id}")
        masks = F.interpolate(
            torch.from_numpy(masks_320)[:, None],
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        )[:, 0].to(device)
        source = load_image(image_id).to(device)

        def references(pairs):
            return [
                (
                    load_image(pair.recipient_image_id).to(device),
                    load_image(pair.sham_image_id).to(device),
                )
                for pair in pairs
            ]

        with torch.inference_mode():
            matched = matched_transplant_layerwise_scores(
                model,
                source,
                masks,
                references(matched_pairs),
                batch_size=args.batch_size,
                candidate_chunk_size=args.candidate_chunk_size,
                feather_kernel=FEATHER_KERNEL,
            )
            random_control = matched_transplant_layerwise_scores(
                model,
                source,
                masks,
                references(random_pairs),
                batch_size=args.batch_size,
                candidate_chunk_size=args.candidate_chunk_size,
                feather_kernel=FEATHER_KERNEL,
            )
        baseline_fusion = 0.5 * (
            average_percentile_rank(g1_logits) + average_percentile_rank(upstream)
        )
        panel = frozen_selector_panel(
            baseline_fusion,
            matched["score"].numpy(),
            random_control["score"].numpy(),
        )
        baseline_local = stable_select(panel["g1_upstream_baseline"], g1_logits)
        if int(candidate_indices[baseline_local]) != int(frozen["selected_candidate_index"]):
            raise ValueError(f"immutable baseline does not reproduce: {image_id}")
        baseline_reproduced += 1
        score_path = score_root / f"{stem}.npz"
        np.savez_compressed(
            score_path,
            candidate_indices=candidate_indices.astype(np.int32),
            g1_logits=g1_logits.astype(np.float32),
            upstream_scores=upstream.astype(np.float32),
            proposal_sources=sources.astype("U32"),
            **{name: values.astype(np.float32) for name, values in panel.items()},
            **_tensor_payload("matched", matched),
            **_tensor_payload("random", random_control),
        )
        score_sha = sha256_file(score_path)
        for variant in VARIANTS:
            local = stable_select(panel[variant], g1_logits)
            selection_rows.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "tumor": int(row["tumor"]),
                    "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                    "candidate_count": len(candidate_indices),
                    "selected_local_index": local,
                    "selected_candidate_index": int(candidate_indices[local]),
                    "selected_source": canonical_source(sources[local]),
                    "selected_g1_logit": float(g1_logits[local]),
                    "selected_upstream_score": float(upstream[local]),
                    "selected_matched_score": float(matched["score"][local]),
                    "selected_random_score": float(random_control["score"][local]),
                    "selected_variant_score": float(panel[variant][local]),
                    "score_path": str(score_path.relative_to(args.output_dir)).replace("\\", "/"),
                    "score_sha256": score_sha,
                }
            )
        del masks, source, matched, random_control
        torch.cuda.empty_cache()
        if image_number % 10 == 0 or image_number == len(val_rows):
            print(
                json.dumps(
                    {
                        "validation_images_frozen": image_number,
                        "elapsed_seconds": round(time.time() - start_time, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if baseline_reproduced != 371 or len(selection_rows) != 371 * len(VARIANTS):
        raise RuntimeError("matched-normal Stage-A cohort mismatch")
    reference_sha = _write_csv(args.output_dir / "reference_manifest.csv", reference_rows)
    selection_sha = _write_csv(args.output_dir / "selection_manifest.csv", selection_rows)
    freeze = {
        "stage": "rich_gallery_matched_normal_transplant_stage_a_v1",
        "experiment_id": "EXP-20260802-codex-rich-gallery-matched-normal-transplant-stage-a-v1",
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "g1_stage_a_freeze_sha256": args.expected_g1_stage_a_freeze_sha256,
        "g1_checkpoint_sha256": g1_freeze["g1_checkpoint_sha256"],
        "val_candidate_manifest_sha256": args.expected_val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.expected_val_pseudo_manifest_sha256,
        "classifier_checkpoint_sha256": args.expected_classifier_sha256,
        "classifier_checkpoint_epoch": int(checkpoint["epoch"]),
        "reference_manifest_sha256": reference_sha,
        "selection_manifest_sha256": selection_sha,
        "validation_images": 371,
        "tumor_validation_images": 184,
        "train_normal_references": 1493,
        "reference_rows": 371 * 2 * PAIR_COUNT,
        "selection_rows": 371 * len(VARIANTS),
        "variants": list(VARIANTS),
        "baseline_reproduction_images": baseline_reproduced,
        "image_size": IMAGE_SIZE,
        "pair_count": PAIR_COUNT,
        "feather_kernel": FEATHER_KERNEL,
        "random_control_seed": RANDOM_CONTROL_SEED,
        "layerwise_stages": list(DENSENET_DIAGNOSTIC_STAGES),
        "layerwise_full_tensors_saved": False,
        "candidate_choices_frozen_before_validation_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": time.time() - start_time,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
