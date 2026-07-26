from __future__ import annotations

"""Train and audit an image-label-only INSIGHT-style RAD-DINO heatmap probe.

The frozen RAD-DINO encoder and the validation-first execution contract are
shared conceptually with ``run_rad_dino_dense_mil_probe.py``.  This runner
keeps its own head, checkpoint metadata and prediction source so the
mechanism comparison remains hashable and independently auditable.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import run_rad_dino_dense_mil_probe as base
from models.rad_dino_insight import (
    InsightDenseMILHead,
    InsightMILConfig,
    insight_mil_loss,
    resize_heatmap,
)


def train_head(
    encoder: torch.nn.Module,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[InsightDenseMILHead, list[dict[str, object]]]:
    dataset = base.LabelOnlyRadiographDataset(
        rows, args.dataset_root.resolve(), args.input_size, augment=True
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=base.seed_worker,
        generator=generator,
    )
    config = InsightMILConfig(
        input_dim=768,
        hidden_dim=128,
        detection_kernel=3,
        context_kernel=9,
        smoothmax_alpha=12.0,
        spectral_lambda=1.0e-4,
    )
    head = InsightDenseMILHead(config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        bce_losses: list[float] = []
        spectral_losses: list[float] = []
        correct = 0
        total = 0
        for pixels, labels, _image_ids in loader:
            tokens = base.extract_patch_tokens(
                encoder, pixels, device=device, grid_size=args.input_size // 14
            )
            _heatmap, fused_logits, _detector, _context = head(tokens)
            loss, bce, spectral, pooled = insight_mil_loss(
                fused_logits,
                labels.to(device),
                alpha=config.smoothmax_alpha,
                spectral_lambda=config.spectral_lambda,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce.detach().cpu()))
            spectral_losses.append(float(spectral.detach().cpu()))
            correct += int(((pooled >= 0.5) == labels.to(device)).sum())
            total += int(labels.numel())
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "bce": float(np.mean(bce_losses)),
            "spectral_decoupling": float(np.mean(spectral_losses)),
            "accuracy": float(correct / max(total, 1)),
        }
        history.append(record)
        print(f"INSIGHT-MIL epoch {epoch}/{args.epochs}: {record}", flush=True)
    return head.eval(), history


def save_checkpoint(
    head: InsightDenseMILHead,
    args: argparse.Namespace,
    history: list[dict[str, object]],
    *,
    output: Path,
) -> Path:
    checkpoint = output / "insight_mil_head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head": "INSIGHT-style local detector + context suppression",
            "input_dim": 768,
            "hidden_dim": 128,
            "detection_kernel": 3,
            "context_kernel": 9,
            "smoothmax_alpha": 12.0,
            "spectral_lambda": 1.0e-4,
            "input_size": args.input_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "split_sha256": args.expected_split_sha256,
            "history": history,
        },
        checkpoint,
    )
    return checkpoint


def map_for_views(
    encoder: torch.nn.Module,
    head: InsightDenseMILHead,
    views: torch.Tensor,
    *,
    device: torch.device,
    input_size: int,
    output_size: int,
    tile_boxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        tokens = base.extract_patch_tokens(
            encoder, views, device=device, grid_size=input_size // 14
        )
        heatmap, _fused, _detector, _context = head(tokens)
        probabilities = resize_heatmap(heatmap, output_size=input_size)
        full = probabilities[0:1]
        tiles = torch.stack(
            [
                F.interpolate(
                    probabilities[index + 1 : index + 2],
                    size=(y1 - y0, x1 - x0),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                for index, (x0, y0, x1, y1) in enumerate(tile_boxes)
            ]
        )
        merged = base.merge_full_and_tiles(
            full,
            tiles,
            tile_boxes=tile_boxes,
            image_size=input_size,
            full_weight=0.5,
        )
        full_out = F.interpolate(
            full, size=(output_size, output_size), mode="bilinear", align_corners=False
        )[0, 0]
        multi_out = F.interpolate(
            merged,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return (
        full_out.detach().cpu().numpy().astype(np.float16),
        multi_out.detach().cpu().numpy().astype(np.float16),
    )


def write_maps(
    encoder: torch.nn.Module,
    head: InsightDenseMILHead,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    device: torch.device,
    output: Path,
) -> dict[str, str]:
    arms = {"single_scale": output / "single_scale", "multiscale": output / "multiscale"}
    for path in arms.values():
        (path / "maps").mkdir(parents=True, exist_ok=False)
    manifests: dict[str, list[dict[str, object]]] = {}
    boxes: tuple[tuple[int, int, int, int], ...] | None = None
    for row_index, row in enumerate(rows):
        image = Image.open(base.locate_verified_image(args.dataset_root, row)).convert("RGB")
        views, observed = base.fixed_views(
            image, input_size=args.input_size, tile_size=args.tile_size
        )
        if boxes is None:
            boxes = observed
        elif boxes != observed:
            raise RuntimeError("Fixed tile layout drifted")
        full_map, multi_map = map_for_views(
            encoder,
            head,
            views,
            device=device,
            input_size=args.input_size,
            output_size=args.output_size,
            tile_boxes=observed,
        )
        for arm, values in (("single_scale", full_map), ("multiscale", multi_map)):
            rel = Path("maps") / f"{Path(row['image_id']).stem}.npy"
            destination = arms[arm] / rel
            np.save(destination, values, allow_pickle=False)
            manifests.setdefault(arm, []).append(
                {
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "tumor": row["tumor"],
                    "map_path": rel.as_posix(),
                    "map_sha256": base.sha256(destination),
                    "raw_p99": float(np.percentile(values.astype(np.float32), 99)),
                    "raw_max": float(values.max()),
                }
            )
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(rows):
            print(f"validation INSIGHT maps: {row_index + 1}/{len(rows)}", flush=True)
    if boxes is None:
        raise RuntimeError("No validation maps generated")
    manifest_hashes: dict[str, str] = {}
    for arm, records in manifests.items():
        path = arms[arm] / "prediction_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifest_hashes[arm] = base.sha256(path)
        (arms[arm] / "generation_metadata.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "source": "frozen RAD-DINO INSIGHT-style local/context MIL head",
                    "cohort": len(rows),
                    "input_size": args.input_size,
                    "output_size": args.output_size,
                    "tile_size": args.tile_size,
                    "validation_gt_read": False,
                    "test_evaluated": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return manifest_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-preprocessor-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--output-size", type=int, default=320)
    parser.add_argument("--tile-size", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("INSIGHT probe requires a Kaggle GPU")
    if (args.input_size, args.output_size, args.tile_size) != (448, 320, 280):
        raise ValueError("The INSIGHT geometry is frozen at 448/320/280")
    if (args.epochs, args.batch_size, args.seed) != (12, 8, 42):
        raise ValueError("The INSIGHT budget is frozen at 12 epochs, batch 8, seed 42")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output-dir must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    base.seed_everything(args.seed)
    snapshot = base.verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    train_rows = base.load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="train"
    )
    val_rows = base.load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=args.expected_split_sha256, split="val"
    )
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("Frozen train/validation cohort mismatch")
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != base.TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers must be {base.TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval().cuda()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda")
    head, history = train_head(encoder, train_rows, args, device=device)
    checkpoint = save_checkpoint(head, args, history, output=args.output_dir)
    checkpoint_sha = base.sha256(checkpoint)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    arms_root = args.output_dir / "predictions"
    manifests = write_maps(
        encoder, head, val_rows, args, device=device, output=arms_root
    )
    freeze = {
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "prediction_manifests": manifests,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    # Only now is validation GT opened.
    single, single_summary = base.evaluate_arm(
        arms_root / "single_scale", args.dataset_root, args.split_manifest
    )
    multi, multi_summary = base.evaluate_arm(
        arms_root / "multiscale", args.dataset_root, args.split_manifest
    )
    paired = base.bootstrap_compare(single, multi)
    paired_path = args.output_dir / "paired_comparison.json"
    paired_path.write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": args.output_dir.name,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "model_hashes": snapshot,
        "checkpoint_sha256": checkpoint_sha,
        "cohort": {"train": 2981, "val": 371, "val_tumor": 184},
        "head": {
            "name": "INSIGHT-style local detector + context suppression",
            "input_dim": 768,
            "hidden_dim": 128,
            "detection_kernel": 3,
            "context_kernel": 9,
            "smoothmax_alpha": 12.0,
            "spectral_lambda": 1.0e-4,
        },
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_manifest": run_manifest,
                "single": single_summary,
                "multiscale": multi_summary,
                "paired": paired,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
