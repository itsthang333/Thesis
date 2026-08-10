from __future__ import annotations

"""Freeze HR-CBPMIL-IE+ validation choices before spatial annotations are opened."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.hr_cbpmil_bags import HRCBPMILBagDataset, collate_hr_cbpmil_bags, load_cluster_cache
from frozen_io import load_split_rows_without_annotations, sha256_file
from models.hr_cbpmil_ie_plus import HRCBPMILIEPlus
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from hr_selectors.hr_cbpmil_ie_plus import select_ie_plus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--pseudo-manifest-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-training-checkpoint-sha256", required=True)
    parser.add_argument("--cluster-cache", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Prediction freeze requires CUDA")
    for path, expected, label in (
        (args.classifier_checkpoint, args.expected_classifier_sha256, "classifier"),
        (args.training_checkpoint, args.expected_training_checkpoint_sha256, "training"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} checkpoint SHA-256 mismatch")
    rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    candidate_rows, audit = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=[row["image_id"] for row in rows],
        split="val",
        expected_pseudo_manifest_sha256=args.pseudo_manifest_sha256,
        expected_manifest_sha256=args.candidate_manifest_sha256,
    )
    if audit.get("cohort") != "all":
        raise ValueError("Prediction requires the full normal/tumor candidate cohort")
    clusters = load_cluster_cache(args.cluster_cache, candidate_rows)
    dataset = HRCBPMILBagDataset(
        rows,
        dataset_root=args.dataset_root,
        candidate_root=args.candidate_root,
        candidate_rows=candidate_rows,
        cluster_cache=clusters,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_hr_cbpmil_bags)
    device = torch.device("cuda:0")
    model = HRCBPMILIEPlus(args.classifier_checkpoint).to(device)
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("stage") != "hr_cbpmil_ie_plus_training_checkpoint_v1"
        or int(checkpoint.get("epoch", -1)) != 30
        or checkpoint.get("protocol_sha256") != args.protocol_sha256
        or checkpoint.get("spatial_ground_truth_read") is not False
        or int(checkpoint.get("test_images_read", -1)) != 0
    ):
        raise ValueError("Training checkpoint is not the final annotation-free epoch-30 EMA source")
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_dir = args.output_dir / "scores"
    mask_dir = args.output_dir / "masks"
    score_dir.mkdir()
    mask_dir.mkdir()
    manifest: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            image_id = batch["image_id"][0]
            image = batch["image"].to(device)
            masks = batch["candidate_masks"].to(device)
            valid = batch["candidate_valid"].to(device)
            cluster_ids = batch["cluster_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                original = model(image, masks, valid, cluster_ids)
                flipped = model(
                    torch.flip(image, dims=(-1,)),
                    torch.flip(masks, dims=(-1,)),
                    valid,
                    cluster_ids,
                )
            count = int(valid.sum())
            a = 0.5 * (
                original["classification_logits"][0, :count].float()
                + flipped["classification_logits"][0, :count].float()
            )
            b = 0.5 * (
                original["detection_logits"][0, :count].float()
                + flipped["detection_logits"][0, :count].float()
            )
            dense = 0.5 * (
                original["dense_logits"][0].float()
                + torch.flip(flipped["dense_logits"][0].float(), dims=(-1,))
            )
            stem = Path(image_id).stem
            candidate_row = candidate_rows[stem]
            candidate_path = args.candidate_root / candidate_row["diagnostic_path"]
            with np.load(candidate_path, allow_pickle=False) as payload:
                candidate_masks = payload["sam_masks"].astype(np.uint8)
                sources = payload["proposal_source_ids"].astype(str)
            result = select_ie_plus(
                candidate_masks,
                a.cpu().numpy(),
                b.cpu().numpy(),
                dense.cpu().numpy(),
                clusters[stem],
            )
            selected = candidate_masks[result.selected_index]
            if int(batch["binary_label"][0]) == 0:
                selected = np.zeros_like(selected, dtype=np.uint8)
            score_path = score_dir / f"{stem}.npz"
            np.savez_compressed(
                score_path,
                classification_logits=a.cpu().numpy().astype(np.float32),
                detection_logits=b.cpu().numpy().astype(np.float32),
                dense_logits=dense.cpu().numpy().astype(np.float32),
                cluster_ids=clusters[stem].astype(np.int32),
                selected_index=np.asarray([result.selected_index], dtype=np.int32),
                top3_clusters=np.asarray(result.top3_clusters, dtype=np.int32),
                cluster_identity=result.cluster_identity.astype(np.float32),
                candidate_extent=result.candidate_extent.astype(np.float32),
            )
            mask_path = mask_dir / f"{stem}.npy"
            np.save(mask_path, selected, allow_pickle=False)
            manifest.append({
                "image_id": image_id,
                "binary_label": int(batch["binary_label"][0]),
                "selected_candidate_index": result.selected_index,
                "selected_source": str(sources[result.selected_index]),
                "selected_cluster": result.selected_cluster,
                "candidate_count": count,
                "candidate_payload_sha256": candidate_row["diagnostic_sha256"],
                "score_path": str(score_path.relative_to(args.output_dir)),
                "score_sha256": sha256_file(score_path),
                "mask_path": str(mask_path.relative_to(args.output_dir)),
                "mask_sha256": sha256_file(mask_path),
            })
    manifest_path = args.output_dir / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    freeze = {
        "stage": "hr_cbpmil_ie_plus_prediction_freeze_v1",
        "images": len(manifest),
        "tumor_images": sum(int(row["binary_label"]) for row in manifest),
        "split_sha256": args.expected_split_sha256,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "training_checkpoint_sha256": args.expected_training_checkpoint_sha256,
        "protocol_sha256": args.protocol_sha256,
        "selection_manifest_sha256": sha256_file(manifest_path),
        "candidate_choices_frozen_before_spatial_gt": True,
        "spatial_ground_truth_used": False,
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "prediction_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
