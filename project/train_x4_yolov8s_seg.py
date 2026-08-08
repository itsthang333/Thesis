from __future__ import annotations

"""Train the frozen X4 YOLOv8s-seg fully supervised upper bound."""

import argparse
import json
import shutil
import time
from pathlib import Path

from frozen_io import sha256_file, validate_sha256


ULTRALYTICS_VERSION = "8.4.0"


def json_float_metrics(values: dict[str, object]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-export-report-sha256", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.epochs != 300 or args.imgsz != 600:
        raise ValueError("X4 protocol fixes YOLOv8s-seg at 300 epochs and 600 pixels")
    export_report_path = args.dataset_root / "export_report.json"
    if sha256_file(export_report_path) != validate_sha256(
        args.expected_export_report_sha256, name="export report SHA-256"
    ):
        raise ValueError("X4 YOLO export report changed")
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    if (
        export_report.get("stage") != "x4_yolov8s_seg_dataset_export_v1"
        or export_report.get("split_counts") != {
            "train": {"images": 2981, "normal": 1493, "tumor": 1488},
            "val": {"images": 371, "normal": 187, "tumor": 184},
        }
        or export_report.get("test_images_read") != 0
        or export_report.get("test_annotations_read") != 0
        or export_report.get("test_evaluated") is not False
    ):
        raise ValueError("X4 YOLO export report violates the frozen cohort")
    pretrained_sha = validate_sha256(args.expected_pretrained_sha256, name="pretrained SHA-256")
    if sha256_file(args.pretrained_checkpoint) != pretrained_sha:
        raise ValueError("Official YOLOv8s-seg checkpoint hash mismatch")

    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"X4 requires ultralytics=={ULTRALYTICS_VERSION}, got {ultralytics.__version__}"
        )
    dataset_yaml = args.dataset_root / "btxrd_x4.yaml"
    started = time.perf_counter()
    model = YOLO(str(args.pretrained_checkpoint))
    model.train(
        data=str(dataset_yaml),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        patience=0,
        pretrained=True,
        cache=False,
        amp=True,
        save=True,
        # Only best/last are required; periodic checkpoints add no scientific
        # evidence and can exhaust Kaggle working storage during 300 epochs.
        save_period=-1,
        val=True,
        plots=False,
        project=str(args.output_dir),
        name="train",
        exist_ok=False,
        verbose=True,
    )
    run_dir = args.output_dir / "train"
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    if not best_source.is_file() or not last_source.is_file():
        raise FileNotFoundError("Ultralytics training did not produce best.pt and last.pt")
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    shutil.copy2(best_source, best_path)
    shutil.copy2(last_source, last_path)
    best_model = YOLO(str(best_path))
    validation = best_model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=False,
        project=str(args.output_dir),
        name="native_val",
        exist_ok=False,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    native = {
        "mask_map50_95": float(validation.seg.map),
        "mask_map50": float(validation.seg.map50),
        "mask_map75": float(validation.seg.map75),
        "results_dict": json_float_metrics(validation.results_dict),
    }
    report = {
        "schema_version": 1,
        "stage": "x4_yolov8s_seg_training_v1",
        "model": "YOLOv8s-seg",
        "ultralytics_version": ULTRALYTICS_VERSION,
        "official_pretrained_sha256": pretrained_sha,
        "export_report_sha256": args.expected_export_report_sha256,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "patience": 0,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "native_ultralytics_validation": native,
        "elapsed_seconds_including_native_validation": elapsed,
        "validation_used_for_standard_fully_supervised_model_selection": True,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "training_report_sha256": sha256_file(report_path)}, indent=2))


if __name__ == "__main__":
    main()
