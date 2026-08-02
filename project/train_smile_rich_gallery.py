from __future__ import annotations

"""Two-pass, image-label-only SMILE control/full trainer."""

import argparse
import hashlib
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
    SMILE_SCHEMA_VERSION,
    SMILELocalEvidence,
    smile_image_label_objective,
)
from project.smile_training import (
    MixedSubtypeBalancedBatchSampler,
    SMILE_BATCH_SIZE,
    SMILE_CONSISTENCY_EVERY,
    SMILE_FLIP_STYLE_WEIGHT,
    SMILE_LR,
    SMILE_PASSES,
    SMILE_REFERENCE_SWAP_WEIGHT,
    SMILE_SEED,
    SMILE_STEPS_PER_PASS,
    SMILE_TERMINAL_STEP,
    SMILE_WEIGHT_DECAY,
    label_safe_summary,
    seed_smile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--densenet-weights", type=Path, required=True)
    parser.add_argument("--densenet-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp"),
        default="fp32",
        help="Numerical execution mode; fp32 is the fail-closed default.",
    )
    parser.add_argument("--verify-image-hashes", action="store_true")
    return parser.parse_args()


def _core(model: nn.Module) -> SMILELocalEvidence:
    value = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(value, SMILELocalEvidence):
        raise TypeError("unexpected SMILE model wrapper")
    return value


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _inputs(
    batch: dict[str, object],
    *,
    arm: str,
    reference_set: str = "primary",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    query = imagenet_normalize_grayscale(batch["query"])
    valid = batch["query_valid"]
    subtype = batch["subtype"]
    if arm == "control":
        return query, valid, None, None, subtype
    references = imagenet_normalize_grayscale(batch[f"{reference_set}_references"])
    reference_valid = batch[f"{reference_set}_reference_valid"]
    return query, valid, references, reference_valid, subtype


def _forward(
    model: nn.Module,
    batch: dict[str, object],
    *,
    arm: str,
    reference_set: str = "primary",
) -> dict[str, torch.Tensor]:
    query, valid, references, reference_valid, subtype = _inputs(
        batch, arm=arm, reference_set=reference_set
    )
    output = model(
        query,
        valid,
        references,
        reference_valid,
        conditioning_subtype=subtype,
    )
    if not isinstance(output, dict):
        raise TypeError("SMILE forward did not return a tensor dictionary")
    return output


@torch.inference_mode()
def validate_label_safe(
    model: nn.Module,
    loader: DataLoader,
    *,
    arm: str,
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    model.eval()
    labels: list[int] = []
    binary_logits: list[float] = []
    subtypes: list[int] = []
    subtype_logits: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for raw in loader:
        batch = _to_device(raw, device)
        output = _forward(model, batch, arm=arm)
        for index, image_id in enumerate(raw["image_id"]):
            label = int(batch["tumor"][index].item())
            subtype = int(batch["subtype"][index].item())
            binary = float(output["binary_image_logits"][index].float().item())
            local_subtype = output["subtype_image_logits"][index].float().cpu().numpy()
            labels.append(label)
            binary_logits.append(binary)
            subtypes.append(subtype)
            subtype_logits.append(local_subtype)
            rows.append(
                {
                    "image_id": str(image_id),
                    "group_id": str(raw["group_id"][index]),
                    "tumor": label,
                    "subtype": subtype,
                    "binary_logit": binary,
                    "subtype_prediction": int(local_subtype.argmax()),
                }
            )
    if len(rows) != 371:
        raise RuntimeError("label-safe validation must contain 371 images")
    return (
        label_safe_summary(labels, binary_logits, subtypes, np.stack(subtype_logits)),
        rows,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_parameter_summary(model: nn.Module) -> tuple[bool, float]:
    finite = True
    maximum = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            current = parameter.detach()
            if not bool(torch.isfinite(current).all()):
                finite = False
            if current.numel():
                maximum = max(maximum, float(current.abs().max().float().cpu()))
    return finite, maximum


def _write_recovery_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    arm: str,
    precision: str,
    epoch: int,
    global_step: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "method": SMILE_METHOD,
            "schema_version": SMILE_SCHEMA_VERSION,
            "arm": arm,
            "precision": precision,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": _core(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "spatial_ground_truth_used": False,
            "test_images_read": 0,
            "test_evaluated": False,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.densenet_weights) != args.densenet_sha256:
        raise ValueError("DenseNet weight SHA-256 mismatch")
    seed_smile()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True)
    train_dataset = BTXRDSMILEReferenceDataset(
        root=args.dataset_root,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_sha256,
        reference_manifest=args.reference_manifest,
        reference_manifest_sha256=args.reference_sha256,
        split="train",
        verify_image_hashes=args.verify_image_hashes,
    )
    val_dataset = BTXRDSMILEReferenceDataset(
        root=args.dataset_root,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_sha256,
        reference_manifest=args.reference_manifest,
        reference_manifest_sha256=args.reference_sha256,
        split="val",
        verify_image_hashes=args.verify_image_hashes,
    )
    train_tumor = [sample.tumor for sample in train_dataset.samples]
    train_subtype = [sample.tumor_type for sample in train_dataset.samples]
    val_loader = DataLoader(
        val_dataset,
        batch_size=SMILE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_smile_batch,
        pin_memory=device.type == "cuda",
    )
    model: nn.Module = SMILELocalEvidence(
        arm=args.arm,
        pretrained_checkpoint=args.densenet_weights,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SMILE_LR, weight_decay=SMILE_WEIGHT_DECAY
    )
    amp = device.type == "cuda" and args.precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    global_step = 0
    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(SMILE_PASSES):
        sampler = MixedSubtypeBalancedBatchSampler(
            train_tumor, train_subtype, epoch=epoch
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_smile_batch,
            pin_memory=amp,
        )
        model.train()
        totals: dict[str, float] = {}
        for raw in loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = _forward(model, batch, arm=args.arm)
                losses = smile_image_label_objective(
                    output, batch["tumor"], batch["subtype"]
                )
                total = losses["total"]
                swap_loss = total * 0.0
                flip_loss = total * 0.0
                if global_step % SMILE_CONSISTENCY_EVERY == 0:
                    if args.arm == "full":
                        with torch.no_grad():
                            swap = _forward(model, batch, arm=args.arm, reference_set="swap")
                        swap_loss = F.mse_loss(
                            torch.sigmoid(output["conditioned_evidence_logits"]),
                            torch.sigmoid(swap["conditioned_evidence_logits"].detach()),
                        )
                    flipped_batch = dict(batch)
                    flipped_batch["query"] = torch.flip(batch["query"], dims=(-1,))
                    flipped_batch["query_valid"] = torch.flip(batch["query_valid"], dims=(-1,))
                    with torch.no_grad():
                        flipped = _forward(model, flipped_batch, arm=args.arm)
                    flip_target = torch.flip(
                        flipped["conditioned_evidence_logits"].detach(), dims=(-1,)
                    )
                    flip_loss = F.mse_loss(
                        torch.sigmoid(output["conditioned_evidence_logits"]),
                        torch.sigmoid(flip_target),
                    )
                total = (
                    total
                    + SMILE_REFERENCE_SWAP_WEIGHT * swap_loss
                    + SMILE_FLIP_STYLE_WEIGHT * flip_loss
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % 64 == 0:
                parameters_finite, parameter_abs_max = _finite_parameter_summary(model)
                if not parameters_finite:
                    raise FloatingPointError(
                        f"model parameters became non-finite at global_step={global_step}"
                    )
                telemetry = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "precision": args.precision,
                    "loss": float(total.detach().float().cpu()),
                    "grad_norm": float(grad_norm.detach().float().cpu()),
                    "parameter_abs_max": parameter_abs_max,
                    "grad_scale": float(scaler.get_scale()),
                }
                with (args.output_dir / "step_telemetry.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(telemetry, sort_keys=True) + "\n")
            if global_step % 256 == 0:
                _write_recovery_checkpoint(
                    args.output_dir / "smile_recovery.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    arm=args.arm,
                    precision=args.precision,
                    epoch=epoch,
                    global_step=global_step,
                )
            for name, value in {
                **losses,
                "reference_swap": swap_loss,
                "flip_consistency": flip_loss,
                "optimized_total": total,
            }.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        if global_step != (epoch + 1) * SMILE_STEPS_PER_PASS:
            raise RuntimeError("terminal step differs from frozen protocol")
        validation, validation_rows = validate_label_safe(
            model, val_loader, arm=args.arm, device=device
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": time.time() - started,
            **{f"train_{name}": value / SMILE_STEPS_PER_PASS for name, value in totals.items()},
            **{f"val_{name}": value for name, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        _write_rows(args.output_dir / f"validation_epoch{epoch}.csv", validation_rows)
        _write_json(args.output_dir / "history.json", history)

    if global_step != SMILE_TERMINAL_STEP:
        raise RuntimeError("SMILE training did not reach the terminal step")
    checkpoint = {
        "method": SMILE_METHOD,
        "schema_version": SMILE_SCHEMA_VERSION,
        "arm": args.arm,
        "precision": args.precision,
        "model_config": _core(model).checkpoint_model_config(),
        "model_state_dict": _core(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "terminal_epoch": SMILE_PASSES - 1,
        "split_sha256": args.split_sha256,
        "reference_sha256": args.reference_sha256,
        "densenet_sha256": args.densenet_sha256,
        "protocol_sha256": args.protocol_sha256,
        "source_sha256": args.source_sha256,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    checkpoint_path = args.output_dir / "smile_terminal.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "method": SMILE_METHOD,
        "arm": args.arm,
        "precision": args.precision,
        "terminal_epoch": SMILE_PASSES - 1,
        "global_step": global_step,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
        "split_sha256": args.split_sha256,
        "reference_sha256": args.reference_sha256,
        "densenet_sha256": args.densenet_sha256,
        "protocol_sha256": args.protocol_sha256,
        "source_sha256": args.source_sha256,
        "training_images": 2981,
        "validation_images": 371,
        "spatial_ground_truth_used": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "training_summary.json", summary)


if __name__ == "__main__":
    main()
