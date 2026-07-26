from __future__ import annotations

"""Train and audit a frozen-RAD-DINO dense MIL localization probe.

The head sees only patch tokens and binary image-level labels from the clean
training split.  Validation masks are opened only after both prediction arms
and their manifests have been hash-frozen.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
    verify_model_snapshot,
)
from compare_nominal_patch_memory_arms import METRICS, paired_group_bootstrap
from models.mae_reconstruction import pad_to_square
from models.rad_dino_dense_mil import (
    DenseMILConfig,
    DenseMILHead,
    dense_mil_loss,
    resize_probability_map,
    merge_full_and_tiles,
)


TRANSFORMERS_VERSION = "4.50.2"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return sha256_file(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tensor_from_image(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    return (torch.from_numpy(values).permute(2, 0, 1) - MEAN) / STD


def fixed_views(
    image: Image.Image,
    *,
    input_size: int,
    tile_size: int,
) -> tuple[torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
    square, _ = pad_to_square(image.convert("RGB"), fill=0)
    resized = square.resize((input_size, input_size), Image.Resampling.BICUBIC)
    end = input_size - tile_size
    boxes = ((0, 0, tile_size, tile_size), (end, 0, input_size, tile_size),
             (0, end, tile_size, input_size), (end, end, input_size, input_size))
    views = [resized] + [
        resized.crop(box).resize((input_size, input_size), Image.Resampling.BICUBIC)
        for box in boxes
    ]
    return torch.stack([tensor_from_image(view, input_size) for view in views]), boxes


class LabelOnlyRadiographDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        dataset_root: Path,
        image_size: int,
        *,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image = Image.open(locate_verified_image(self.dataset_root, row)).convert("RGB")
        square, _ = pad_to_square(image, fill=0)
        if self.augment and random.random() < 0.5:
            square = square.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return (
            tensor_from_image(square, self.image_size),
            torch.tensor(float(row["tumor"]), dtype=torch.float32),
            row["image_id"],
        )


def extract_patch_tokens(
    encoder: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    grid_size: int,
) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = encoder(pixel_values=pixels.to(device, non_blocking=True)).last_hidden_state
    expected = grid_size * grid_size + 1
    if hidden.ndim != 3 or hidden.shape[1] != expected:
        raise RuntimeError(f"Unexpected token shape {tuple(hidden.shape)}; expected {expected}")
    return hidden[:, 1:].float().reshape(-1, grid_size, grid_size, hidden.shape[-1])


def train_head(
    encoder: torch.nn.Module,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[DenseMILHead, list[dict[str, object]]]:
    dataset = LabelOnlyRadiographDataset(
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
        worker_init_fn=lambda worker_id: seed_everything(args.seed + worker_id + 1),
        generator=generator,
    )
    grid = args.input_size // 14
    head = DenseMILHead(input_dim=768).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        correct = 0
        total = 0
        for pixels, labels, _image_ids in loader:
            tokens = extract_patch_tokens(encoder, pixels, device=device, grid_size=grid)
            logits = head(tokens)
            loss, pooled = dense_mil_loss(logits, labels.to(device), temperature=args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int(((torch.sigmoid(pooled) >= 0.5) == labels.to(device)).sum())
            total += int(labels.numel())
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "accuracy": float(correct / max(total, 1)),
        }
        history.append(record)
        print(f"dense-MIL epoch {epoch}/{args.epochs}: {record}", flush=True)
    return head.eval(), history


def save_checkpoint(
    head: DenseMILHead,
    args: argparse.Namespace,
    history: list[dict[str, object]],
    *,
    output: Path,
) -> Path:
    checkpoint = output / "dense_mil_head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": 768,
            "temperature": args.temperature,
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
    head: DenseMILHead,
    views: torch.Tensor,
    *,
    device: torch.device,
    input_size: int,
    output_size: int,
    tile_boxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    grid = input_size // 14
    tokens = extract_patch_tokens(encoder, views, device=device, grid_size=grid)
    with torch.inference_mode():
        logits = head(tokens)
        probabilities = resize_probability_map(logits, output_size=input_size)
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
        merged = merge_full_and_tiles(
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
            merged, size=(output_size, output_size), mode="bilinear", align_corners=False
        )[0, 0]
    return (
        full_out.detach().cpu().numpy().astype(np.float16),
        multi_out.detach().cpu().numpy().astype(np.float16),
    )


def write_maps(
    encoder: torch.nn.Module,
    head: DenseMILHead,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    device: torch.device,
    output: Path,
) -> dict[str, str]:
    arms = {"single_scale": output / "single_scale", "multiscale": output / "multiscale"}
    for path in arms.values():
        (path / "maps").mkdir(parents=True, exist_ok=False)
    manifests: dict[str, str] = {}
    boxes: tuple[tuple[int, int, int, int], ...] | None = None
    for row_index, row in enumerate(rows):
        image = Image.open(locate_verified_image(args.dataset_root, row)).convert("RGB")
        views, observed = fixed_views(image, input_size=args.input_size, tile_size=args.tile_size)
        if boxes is None:
            boxes = observed
        elif boxes != observed:
            raise RuntimeError("Fixed tile layout drifted")
        full_map, multi_map = map_for_views(
            encoder, head, views, device=device, input_size=args.input_size,
            output_size=args.output_size, tile_boxes=observed,
        )
        for arm, values in (("single_scale", full_map), ("multiscale", multi_map)):
            rel = Path("maps") / f"{Path(row['image_id']).stem}.npy"
            destination = arms[arm] / rel
            np.save(destination, values, allow_pickle=False)
            manifests.setdefault(arm, [])
            manifests[arm].append({
                "image_id": row["image_id"], "group_id": row["group_id"],
                "tumor": row["tumor"], "map_path": rel.as_posix(),
                "map_sha256": sha256(destination),
                "raw_p99": float(np.percentile(values.astype(np.float32), 99)),
                "raw_max": float(values.max()),
            })
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(rows):
            print(f"validation dense-MIL maps: {row_index + 1}/{len(rows)}", flush=True)
    for arm, records in manifests.items():
        path = arms[arm] / "prediction_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifests[arm] = sha256(path)
        (arms[arm] / "generation_metadata.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "source": "frozen RAD-DINO dense MIL head",
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
    if boxes is None:
        raise RuntimeError("No validation maps generated")
    return manifests


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def evaluate_arm(
    arm_dir: Path,
    dataset_root: Path,
    split_manifest: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_path = arm_dir / "prediction_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    # All prediction files are hash-checked before opening any annotation.
    for row in manifest:
        path = arm_dir / row["map_path"]
        if sha256(path) != row["map_sha256"]:
            raise RuntimeError(f"Prediction map hash mismatch: {row['image_id']}")
    from datasets.btxrd import BTXRDSegmentationDataset

    dataset = BTXRDSegmentationDataset(
        root=dataset_root, split="val", image_size=320, augment=False, split_manifest=split_manifest
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, mask, name = dataset[index]
        gt_by_name[str(name)] = mask[0].numpy() > 0.5
    evaluated: list[dict[str, object]] = []
    for row in manifest:
        if row["tumor"] != "1":
            continue
        values = np.load(arm_dir / row["map_path"], allow_pickle=False).astype(np.float32)
        target = gt_by_name[row["image_id"]]
        flat_target = target.reshape(-1).astype(np.uint8)
        flat_values = values.reshape(-1)
        item: dict[str, object] = {
            "image_id": row["image_id"], "group_id": row["group_id"],
            "gt_area_ratio": float(target.mean()), "size_group": subgroup(float(target.mean())),
            "pixel_ap": float(average_precision_score(flat_target, flat_values)),
            "pixel_auroc": float(roc_auc_score(flat_target, flat_values)),
            "argmax_hit": float(target.reshape(-1)[int(np.argmax(flat_values))]),
            "saliency_mass_in_gt": float(values[target].sum()) / max(float(values.sum()), 1e-12),
        }
        for percentile in (90, 95, 97, 99):
            item[f"dice_p{percentile}"] = dice(values >= np.percentile(values, percentile), target)
        evaluated.append(item)
    if len(evaluated) != 184:
        raise RuntimeError(f"Expected 184 tumor evaluations, got {len(evaluated)}")
    counts = {name: sum(row["size_group"] == name for row in evaluated) for name in ("small", "medium", "large")}
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise RuntimeError(f"Subgroup contract drift: {counts}")
    out = arm_dir / "evaluation"
    out.mkdir(exist_ok=False)
    per_image = out / "per_image.csv"
    with per_image.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    metrics = ["pixel_ap", "pixel_auroc", "argmax_hit", "saliency_mass_in_gt", "dice_p90", "dice_p95", "dice_p97", "dice_p99"]
    image_labels = np.asarray([int(row["tumor"]) for row in manifest], dtype=np.uint8)
    image_scores = np.asarray([float(row["raw_p99"]) for row in manifest], dtype=np.float64)
    overall_metrics = {
        metric: float(np.mean([row[metric] for row in evaluated]))
        for metric in metrics
    }
    summary = {
        "arm": arm_dir.name,
        "cohort": {"validation": 371, "tumor": 184, **counts},
        "image_level_auroc_from_raw_p99": float(
            roc_auc_score(image_labels, image_scores)
        ),
        "tumor_localization": {"overall": {"n": 184, **overall_metrics}},
    }
    for name in ("small", "medium", "large"):
        rows = [r for r in evaluated if r["size_group"] == name]
        summary["tumor_localization"][name] = {"n": len(rows), **{m: float(np.mean([r[m] for r in rows])) for m in metrics}}
    summary.update({"prediction_manifest_sha256": sha256(manifest_path), "validation_gt_read_only_after_prediction_freeze": True,
                    "complete_misses_included": True, "consumer_trained": False, "test_evaluated": False})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return evaluated, summary


def bootstrap_compare(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
) -> dict[str, object]:
    left_by = {str(row["image_id"]): row for row in left}
    right_by = {str(row["image_id"]): row for row in right}
    if left_by.keys() != right_by.keys() or len(left_by) != 184:
        raise RuntimeError("Dense-MIL paired cohorts differ")
    metric_results: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        strata: dict[str, object] = {}
        for stratum in ("overall", "small", "medium", "large"):
            names = [
                name
                for name, row in left_by.items()
                if stratum == "overall" or row["size_group"] == stratum
            ]
            strata[stratum] = paired_group_bootstrap(
                [
                    (
                        str(left_by[name]["group_id"]),
                        float(right_by[name][metric]) - float(left_by[name][metric]),
                    )
                    for name in names
                ],
                replicates=10_000,
                seed=20260726 + metric_index * 10 + len(stratum),
            )
        metric_results[metric] = strata
    return {
        "method": "paired complete-group bootstrap",
        "replicates": 10_000,
        "seed": 20260726,
        "interpretation": (
            "mechanism feasibility only; no arm/threshold promotion and no "
            "downstream consumer without a separate predeclared protocol"
        ),
        "metrics": metric_results,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dense MIL probe requires a Kaggle GPU")
    if args.input_size != 448 or args.output_size != 320 or args.tile_size != 280:
        raise ValueError("The dense-MIL geometry is frozen at 448/320/280")
    if args.epochs != 12 or args.batch_size != 8 or args.seed != 42:
        raise ValueError("The frozen dense-MIL budget is 12 epochs, batch 8, seed 42")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output-dir must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    snapshot = verify_model_snapshot(
        args.model_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_preprocessor_sha256=args.expected_preprocessor_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    train_rows = load_split_rows_without_annotations(args.split_manifest, expected_sha256=args.expected_split_sha256, split="train")
    val_rows = load_split_rows_without_annotations(args.split_manifest, expected_sha256=args.expected_split_sha256, split="val")
    if len(train_rows) != 2981 or len(val_rows) != 371:
        raise RuntimeError("Frozen train/validation cohort mismatch")
    from transformers import AutoModel
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"transformers must be {TRANSFORMERS_VERSION}, got {transformers.__version__}")
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval().cuda()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda")
    started = datetime.now(timezone.utc)
    head, history = train_head(encoder, train_rows, args, device=device)
    checkpoint = save_checkpoint(head, args, history, output=args.output_dir)
    checkpoint_sha = sha256(checkpoint)
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    # The full validation prediction set is produced before any GT evaluator is called.
    arms_root = args.output_dir / "predictions"
    manifests = write_maps(encoder, head, val_rows, args, device=device, output=arms_root)
    freeze = {
        "source_commit": args.source_commit, "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256, "checkpoint_sha256": checkpoint_sha,
        "prediction_manifests": manifests, "validation_gt_read": False, "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    # Only now is validation GT opened.
    single, single_summary = evaluate_arm(arms_root / "single_scale", args.dataset_root, args.split_manifest)
    multi, multi_summary = evaluate_arm(arms_root / "multiscale", args.dataset_root, args.split_manifest)
    paired = bootstrap_compare(single, multi)
    paired_path = args.output_dir / "paired_comparison.json"
    paired_path.write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": args.output_dir.name, "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256, "split_sha256": args.expected_split_sha256,
        "model_hashes": snapshot, "checkpoint_sha256": checkpoint_sha,
        "cohort": {"train": 2981, "val": 371, "val_tumor": 184},
        "validation_gt_read_only_after_prediction_freeze": True,
        "consumer_trained": False, "test_evaluated": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_manifest": run_manifest, "single": single_summary, "multiscale": multi_summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
