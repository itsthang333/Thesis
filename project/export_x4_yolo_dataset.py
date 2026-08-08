from __future__ import annotations

"""Export the locked BTXRD train/validation split to Ultralytics segmentation format."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image

from datasets.btxrd import resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
    validate_sha256,
)


EXPECTED_COUNTS = {
    "train": {"images": 2981, "normal": 1493, "tumor": 1488},
    "val": {"images": 371, "normal": 187, "tumor": 184},
}


def polygons_to_yolo_rows(payload: dict[str, object], *, width: int, height: int) -> list[str]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source image geometry")
    rows: list[str] = []
    shapes = payload.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("LabelMe shapes must be a list")
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) < 3:
            continue
        coordinates: list[float] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Invalid LabelMe polygon point")
            x = min(max(float(point[0]), 0.0), float(width - 1)) / float(width)
            y = min(max(float(point[1]), 0.0), float(height - 1)) / float(height)
            coordinates.extend((x, y))
        rows.append("0 " + " ".join(f"{value:.10f}" for value in coordinates))
    return rows


def materialize_image(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown image materialization mode: {mode}")
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    split_sha = validate_sha256(args.expected_split_sha256, name="split SHA-256")
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    manifest_rows: list[dict[str, object]] = []
    annotation_files_opened = 0
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        rows = load_split_rows_without_annotations(
            args.split_manifest, expected_sha256=split_sha, split=split, allow_test=False
        )
        counts = {
            "images": len(rows),
            "normal": sum(int(row["tumor"]) == 0 for row in rows),
            "tumor": sum(int(row["tumor"]) == 1 for row in rows),
        }
        if counts != EXPECTED_COUNTS[split]:
            raise ValueError(f"X4 YOLO {split} counts differ: {counts}")
        split_counts[split] = counts
        for row in rows:
            image_id = row["image_id"]
            source_image = locate_verified_image(btxrd_root, row)
            with Image.open(source_image) as handle:
                width, height = handle.size
            label_rows: list[str] = []
            annotation_sha = ""
            if int(row["tumor"]):
                annotation = btxrd_root / "Annotations" / f"{Path(image_id).stem}.json"
                annotation_sha = sha256_file(annotation)
                payload = json.loads(annotation.read_text(encoding="utf-8"))
                annotation_files_opened += 1
                label_rows = polygons_to_yolo_rows(payload, width=width, height=height)
                if not label_rows:
                    raise ValueError(f"Tumor image has no valid polygon: {image_id}")
            destination_image = args.output_dir / "images" / split / image_id
            materialize_image(source_image, destination_image, mode=args.image_mode)
            destination_label = args.output_dir / "labels" / split / f"{Path(image_id).stem}.txt"
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "group_id": row["group_id"],
                    "split": split,
                    "tumor": row["tumor"],
                    "width": width,
                    "height": height,
                    "instances": len(label_rows),
                    "source_image_sha256": row["image_sha256"],
                    "source_annotation_sha256": annotation_sha,
                    "label_path": destination_label.relative_to(args.output_dir).as_posix(),
                    "label_sha256": sha256_file(destination_label),
                    "image_mode": args.image_mode,
                }
            )

    if annotation_files_opened != 1488 + 184:
        raise ValueError("X4 YOLO did not open exactly the canonical train/val tumor annotations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "export_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    dataset_yaml = args.output_dir / "btxrd_x4.yaml"
    escaped_root = args.output_dir.resolve().as_posix().replace("'", "''")
    dataset_yaml.write_text(
        f"path: '{escaped_root}'\ntrain: images/train\nval: images/val\nnames:\n  0: tumor\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "stage": "x4_yolov8s_seg_dataset_export_v1",
        "split_sha256": split_sha,
        "split_counts": split_counts,
        "images": len(manifest_rows),
        "instance_rows": sum(int(row["instances"]) for row in manifest_rows),
        "annotation_files_opened": annotation_files_opened,
        "export_manifest_sha256": sha256_file(manifest_path),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "format": "Ultralytics polygon segmentation: class x1 y1 ... xn yn, normalized",
        "image_materialization": args.image_mode,
        "test_images_read": 0,
        "test_annotations_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "export_report_sha256": sha256_file(report_path)}, indent=2))


if __name__ == "__main__":
    main()
