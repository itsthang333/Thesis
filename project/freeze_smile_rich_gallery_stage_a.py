from __future__ import annotations

"""Freeze SMILE maps and rich-gallery choices before validation polygons."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from project.datasets.smile_reference import (
    BTXRDSMILEReferenceDataset,
    collate_smile_batch,
    imagenet_normalize_grayscale,
    sha256_file,
)
from project.models.smile_local_evidence import (
    SMILE_METHOD,
    SMILELocalEvidence,
    score_gallery_candidates_from_evidence,
)
from project.pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


VARIANTS = ("baseline", "identity_only", "identity_extent")
EXPECTED_BASELINE_DICE = 0.2887294867


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-summary-sha256", required=True)
    parser.add_argument("--g1-stage-a-root", type=Path, required=True)
    parser.add_argument("--g1-freeze-sha256", required=True)
    parser.add_argument("--g1-split-sha256", required=True)
    parser.add_argument("--val-candidate-root", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest-sha256", required=True)
    parser.add_argument("--val-pseudo-manifest-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _stable_select(score: np.ndarray, g1: np.ndarray) -> int:
    return max(range(len(score)), key=lambda index: (float(score[index]), float(g1[index]), -index))


def _canonical_source(value: object) -> str:
    lowered = str(value).lower()
    if "classifier448" in lowered:
        return "classifier448"
    if "external" in lowered or "biomed" in lowered:
        return "external_saliency"
    if "layer" in lowered or "anchor" in lowered:
        return "layercam320"
    if "fallback" in lowered:
        return "fallback"
    raise ValueError(f"unknown proposal source: {value!r}")


def _verify_g1(
    root: Path,
    *,
    freeze_sha256: str,
    split_sha256: str,
    candidate_sha256: str,
    pseudo_sha256: str,
) -> dict[str, dict[str, str]]:
    freeze_path = root / "prediction_freeze.json"
    if sha256_file(freeze_path) != freeze_sha256:
        raise ValueError("G1 freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("split_sha256") != split_sha256
        or freeze.get("val_candidate_manifest_sha256") != candidate_sha256
        or freeze.get("val_pseudo_manifest_sha256") != pseudo_sha256
        or freeze.get("validation_images") != 371
        or freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("G1 freeze contract mismatch")
    manifest_path = root / "stage_a_selection_manifest.csv"
    rows = _read_csv(manifest_path)
    selected = {
        row["image_id"]: row
        for row in rows
        if row["variant"] == "g1_frozen__rank_fusion"
    }
    if len(selected) != 371:
        raise ValueError("G1 baseline selection cohort mismatch")
    return selected


def _load_model(
    training_root: Path,
    *,
    training_summary_sha256: str,
    arm: str,
    protocol_sha256: str,
    source_sha256: str,
    device: torch.device,
) -> SMILELocalEvidence:
    summary_path = training_root / "training_summary.json"
    if sha256_file(summary_path) != training_summary_sha256:
        raise ValueError("training summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = training_root / "smile_terminal.pt"
    if summary.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("terminal checkpoint differs from summary")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("method") != SMILE_METHOD
        or checkpoint.get("arm") != arm
        or checkpoint.get("global_step") != 2986
        or checkpoint.get("terminal_epoch") != 1
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("source_sha256") != source_sha256
        or checkpoint.get("spatial_ground_truth_used") is not False
        or checkpoint.get("test_evaluated") is not False
    ):
        raise ValueError("terminal checkpoint contract mismatch")
    config = checkpoint["model_config"]
    model = SMILELocalEvidence(
        arm=arm,
        fpn_channels=int(config["fpn_channels"]),
        dropout=float(config["dropout"]),
        match_temperature=float(config["match_temperature"]),
        query_chunk_size=int(config["query_chunk_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _map_to_candidate_grid(
    evidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    locations = torch.nonzero(valid > 0.5, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("evidence map has no valid image cells")
    y_min, x_min = locations.amin(dim=0)
    y_max, x_max = locations.amax(dim=0)
    crop = evidence[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1]
    restored = F.interpolate(
        crop[None, None].float(), size=output_size, mode="bilinear", align_corners=False
    )[0, 0]
    return restored.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model: nn.Module = _load_model(
        args.training_root,
        training_summary_sha256=args.training_summary_sha256,
        arm=args.arm,
        protocol_sha256=args.protocol_sha256,
        source_sha256=args.source_sha256,
        device=device,
    )
    dataset = BTXRDSMILEReferenceDataset(
        root=args.dataset_root,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_sha256,
        reference_manifest=args.reference_manifest,
        reference_manifest_sha256=args.reference_sha256,
        split="val",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_smile_batch,
        pin_memory=device.type == "cuda",
    )
    g1_rows = _verify_g1(
        args.g1_stage_a_root,
        freeze_sha256=args.g1_freeze_sha256,
        split_sha256=args.g1_split_sha256,
        candidate_sha256=args.val_candidate_manifest_sha256,
        pseudo_sha256=args.val_pseudo_manifest_sha256,
    )
    candidate_rows, audit = validate_candidate_diagnostics_manifest(
        args.val_candidate_root,
        expected_image_names=[sample.image_id for sample in dataset.samples],
        split="val",
        expected_manifest_sha256=args.val_candidate_manifest_sha256,
        expected_pseudo_manifest_sha256=args.val_pseudo_manifest_sha256,
    )
    if audit.get("cohort") != "all" or len(candidate_rows) != 371:
        raise ValueError("candidate validation cohort mismatch")

    args.output_dir.mkdir(parents=True)
    score_root = args.output_dir / "scores"
    score_root.mkdir()
    selection_rows: list[dict[str, object]] = []
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            query = imagenet_normalize_grayscale(raw["query"].to(device))
            query_valid = raw["query_valid"].to(device)
            subtype = raw["subtype"].to(device)
            if args.arm == "full":
                primary = model(
                    query,
                    query_valid,
                    imagenet_normalize_grayscale(raw["primary_references"].to(device)),
                    raw["primary_reference_valid"].to(device),
                    conditioning_subtype=subtype,
                )
                swap = model(
                    query,
                    query_valid,
                    imagenet_normalize_grayscale(raw["swap_references"].to(device)),
                    raw["swap_reference_valid"].to(device),
                    conditioning_subtype=subtype,
                )
                dense = 0.5 * (
                    primary["conditioned_evidence_logits"]
                    + swap["conditioned_evidence_logits"]
                )
            else:
                primary = model(
                    query,
                    query_valid,
                    conditioning_subtype=subtype,
                )
                dense = primary["conditioned_evidence_logits"]
            for index, image_id_value in enumerate(raw["image_id"]):
                image_id = str(image_id_value)
                candidate_row = candidate_rows[Path(image_id).stem]
                candidate_path = args.val_candidate_root / candidate_row["diagnostic_path"]
                if sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]:
                    raise ValueError(f"candidate payload changed: {image_id}")
                frozen = g1_rows[image_id]
                g1_path = args.g1_stage_a_root / frozen["score_path"]
                if sha256_file(g1_path) != frozen["score_sha256"]:
                    raise ValueError(f"G1 score payload changed: {image_id}")
                with np.load(g1_path, allow_pickle=False) as payload:
                    candidate_indices = payload["candidate_indices"].astype(np.int64)
                    g1_key = "g1_frozen_candidate_logits" if "g1_frozen_candidate_logits" in payload else "g1_logits"
                    g1 = payload[g1_key].astype(np.float64)
                    upstream = payload["upstream_scores"].astype(np.float64)
                with np.load(candidate_path, allow_pickle=False) as payload:
                    proposals = payload["sam_masks"].astype(bool)
                    sources = payload["proposal_source_ids"].astype(str)
                eligible = proposals[candidate_indices]
                evidence_320 = _map_to_candidate_grid(
                    dense[index, 0],
                    primary["evidence_valid"][index, 0],
                    output_size=eligible.shape[-2:],
                )
                panel = score_gallery_candidates_from_evidence(
                    evidence_320, eligible, g1, upstream
                )
                baseline_local = _stable_select(panel["baseline"], g1)
                if int(candidate_indices[baseline_local]) != int(frozen["selected_candidate_index"]):
                    raise ValueError(f"immutable baseline does not reproduce: {image_id}")
                score_path = score_root / f"{Path(image_id).stem}.npz"
                np.savez_compressed(
                    score_path,
                    evidence_logit=evidence_320,
                    candidate_indices=candidate_indices.astype(np.int32),
                    proposal_sources=sources[candidate_indices].astype("U32"),
                    g1_logits=g1.astype(np.float32),
                    upstream_scores=upstream.astype(np.float32),
                    **{name: values.astype(np.float32) for name, values in panel.items()},
                )
                score_sha = sha256_file(score_path)
                for variant in VARIANTS:
                    local = _stable_select(panel[variant], g1)
                    selection_rows.append(
                        {
                            "arm": args.arm,
                            "variant": variant,
                            "image_id": image_id,
                            "group_id": str(raw["group_id"][index]),
                            "tumor": int(raw["tumor"][index].item()),
                            "subtype": int(raw["subtype"][index].item()),
                            "selected_local_index": local,
                            "selected_candidate_index": int(candidate_indices[local]),
                            "selected_source": _canonical_source(sources[candidate_indices[local]]),
                            "score_path": str(score_path.relative_to(args.output_dir)).replace("\\", "/"),
                            "score_sha256": score_sha,
                        }
                    )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "validation_images_frozen": len(selection_rows) // len(VARIANTS),
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(selection_rows) != 371 * len(VARIANTS):
        raise RuntimeError("Stage-A selection cohort mismatch")
    selection_sha = _write_csv(args.output_dir / "selection_manifest.csv", selection_rows)
    freeze = {
        "stage": "smile_rich_gallery_stage_a_v1",
        "arm": args.arm,
        "variants": list(VARIANTS),
        "selection_manifest_sha256": selection_sha,
        "validation_images": 371,
        "selection_rows": len(selection_rows),
        "training_summary_sha256": args.training_summary_sha256,
        "split_sha256": args.split_sha256,
        "reference_sha256": args.reference_sha256,
        "g1_freeze_sha256": args.g1_freeze_sha256,
        "g1_split_sha256": args.g1_split_sha256,
        "val_candidate_manifest_sha256": args.val_candidate_manifest_sha256,
        "val_pseudo_manifest_sha256": args.val_pseudo_manifest_sha256,
        "protocol_sha256": args.protocol_sha256,
        "source_sha256": args.source_sha256,
        "dense_map_reference_sets_averaged": args.arm == "full",
        "candidate_choices_frozen_before_validation_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
