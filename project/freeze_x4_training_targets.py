from __future__ import annotations

"""Freeze one complete native-resolution X4 WSSS training-target bundle.

Two upstream representations are accepted:

* ``mask_manifest`` for CAM, PuzzleCAM and S2C generators; and
* ``rich_gallery`` for a frozen candidate bank plus frozen G1/R7 choices.

The output schema is deliberately identical for all four WSSS student arms.
Normal images are always materialized as explicit empty masks.  Native geometry
is taken from the immutable canonical split manifest.  An optional dataset root
can additionally verify the image bytes against that manifest, but polygons and
test data are never read.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from datasets.btxrd import resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest
from x4_contract import CANONICAL_SPLIT_SHA256, PSEUDO_STUDENT_ARMS, load_x4_protocol


EXPECTED = {"images": 2981, "tumor": 1488, "normal": 1493}
MASK_PATH_FIELDS = ("mask_path", "mask_file", "mask_filename", "prediction_path")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source-mask relative path: {value}")
    return relative


def binary_at_native(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("X4 source target must be a 2-D mask")
    binary = array > 0
    if binary.shape == (height, width):
        return binary
    return np.asarray(
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


def source_mask_path(row: dict[str, str], source_root: Path) -> Path:
    values = [row.get(field, "").strip() for field in MASK_PATH_FIELDS]
    populated = [value for value in values if value]
    if len(populated) != 1:
        raise ValueError("mask manifest must define exactly one recognized mask-path field")
    path = source_root / safe_relative(populated[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("mask_sha256", "").strip()
    if expected and sha256_file(path) != expected:
        raise ValueError(f"source mask SHA-256 mismatch: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=PSEUDO_STUDENT_ARMS, required=True)
    parser.add_argument(
        "--source-kind", choices=("mask_manifest", "rich_gallery"), required=True
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional BTXRD root for an additional image-byte geometry check.",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument("--expected-source-freeze-sha256")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--candidate-pseudo-manifest-sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_options(args: argparse.Namespace) -> None:
    if (args.source_freeze is None) != (args.expected_source_freeze_sha256 is None):
        raise ValueError("source freeze path and SHA-256 must be provided together")
    rich_fields = (
        args.candidate_root,
        args.candidate_manifest_sha256,
        args.candidate_pseudo_manifest_sha256,
    )
    if args.source_kind == "rich_gallery" and not all(rich_fields):
        raise ValueError("rich-gallery target freeze requires all candidate locks")
    if args.source_kind == "mask_manifest" and any(rich_fields):
        raise ValueError("mask-manifest target freeze cannot receive candidate inputs")


def main() -> None:
    args = parse_args()
    _validate_options(args)
    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    if sha256_file(args.source_manifest) != args.expected_source_manifest_sha256:
        raise ValueError("X4 source manifest SHA-256 mismatch")
    source_freeze_sha = None
    if args.source_freeze is not None:
        source_freeze_sha = sha256_file(args.source_freeze)
        if source_freeze_sha != args.expected_source_freeze_sha256:
            raise ValueError("X4 source freeze SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(args.repo_root)
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split="train",
        allow_test=False,
    )
    counts = {
        "images": len(split_rows),
        "tumor": sum(int(row["tumor"]) for row in split_rows),
        "normal": sum(1 - int(row["tumor"]) for row in split_rows),
    }
    if counts != EXPECTED:
        raise ValueError(f"canonical X4 train counts differ: {counts}")

    source_rows = read_csv(args.source_manifest)
    source_by_id = {row["image_id"]: row for row in source_rows}
    expected_ids = {row["image_id"] for row in split_rows}
    if len(source_rows) != len(source_by_id) or set(source_by_id) != expected_ids:
        raise ValueError("X4 source manifest differs from canonical train")

    candidate_rows = None
    candidate_summary = None
    if args.source_kind == "rich_gallery":
        assert args.candidate_root is not None
        candidate_rows, candidate_summary = validate_candidate_diagnostics_manifest(
            args.candidate_root,
            expected_image_names=expected_ids,
            split="train",
            expected_pseudo_manifest_sha256=args.candidate_pseudo_manifest_sha256,
            expected_manifest_sha256=args.candidate_manifest_sha256,
        )

    root = resolve_btxrd_root(args.dataset_root) if args.dataset_root else None
    mask_root = args.output_dir / "masks"
    mask_root.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, object]] = []
    tumor_empty = 0
    total_foreground = 0
    for canonical in split_rows:
        image_id = canonical["image_id"]
        source = source_by_id[image_id]
        if source.get("group_id", canonical["group_id"]) != canonical["group_id"]:
            raise ValueError(f"source group differs: {image_id}")
        if int(source.get("tumor", canonical["tumor"])) != int(canonical["tumor"]):
            raise ValueError(f"source label differs: {image_id}")
        native_width = int(canonical["width"])
        native_height = int(canonical["height"])
        if native_width <= 0 or native_height <= 0:
            raise ValueError(f"invalid canonical native geometry: {image_id}")
        if root is not None:
            image_path = locate_verified_image(root, canonical)
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            if (image_width, image_height) != (native_width, native_height):
                raise ValueError(f"canonical/image geometry differs: {image_id}")

        selected_index = None
        selected_source = "explicit_empty_normal"
        copy_source: Path | None = None
        foreground: int
        if args.source_kind == "mask_manifest":
            path = source_mask_path(source, args.source_root)
            with Image.open(path) as image:
                mask = image.convert("L")
                if mask.size != (native_width, native_height):
                    raise ValueError(f"source mask native geometry differs: {image_id}")
                histogram = mask.histogram()
            if sum(histogram[1:255]) != 0:
                raise ValueError(f"source mask is not binary: {image_id}")
            foreground = int(histogram[255])
            expected_foreground = source.get("mask_foreground_pixels", "").strip()
            if expected_foreground and foreground != int(expected_foreground):
                raise ValueError(f"source mask foreground differs: {image_id}")
            if int(canonical["tumor"]) == 0 and foreground != 0:
                raise ValueError(f"normal source mask is non-empty: {image_id}")
            copy_source = path
            selected_source = str(source.get("source", args.arm))
        elif int(canonical["tumor"]) == 0:
            native = np.zeros((native_height, native_width), dtype=bool)
        else:
            assert candidate_rows is not None and args.candidate_root is not None
            candidate = candidate_rows[Path(image_id).stem]
            payload_path = args.candidate_root / candidate["diagnostic_path"]
            if sha256_file(payload_path) != candidate["diagnostic_sha256"]:
                raise ValueError(f"candidate payload changed: {image_id}")
            selected_index = int(source["selected_candidate_index"])
            with np.load(payload_path, allow_pickle=False) as payload:
                masks = payload["sam_masks"]
                sources = payload["proposal_source_ids"].astype(str)
                if not 0 <= selected_index < len(masks):
                    raise ValueError(f"selected candidate is out of range: {image_id}")
                native = binary_at_native(
                    masks[selected_index], width=native_width, height=native_height
                )
                selected_source = str(sources[selected_index])

        if copy_source is None:
            foreground = int(native.sum())
        tumor_empty += int(int(canonical["tumor"]) == 1 and foreground == 0)
        total_foreground += foreground
        relative = Path("masks") / f"{Path(image_id).stem}.png"
        output_path = args.output_dir / relative
        if copy_source is not None:
            shutil.copyfile(copy_source, output_path)
        else:
            Image.fromarray(native.astype(np.uint8) * 255, mode="L").save(
                output_path, optimize=True
            )
        manifest_rows.append(
            {
                "image_id": image_id,
                "group_id": canonical["group_id"],
                "tumor": canonical["tumor"],
                "mask_path": relative.as_posix(),
                "mask_height": native_height,
                "mask_width": native_width,
                "mask_foreground_pixels": foreground,
                "mask_sha256": sha256_file(output_path),
                "source_kind": args.source_kind,
                "source": selected_source,
                "selected_candidate_index": (
                    "" if selected_index is None else selected_index
                ),
            }
        )

    manifest_path = args.output_dir / "x4_target_manifest.csv"
    manifest_sha = write_csv(manifest_path, manifest_rows)
    output_bytes = int(
        sum(item.stat().st_size for item in args.output_dir.rglob("*") if item.is_file())
    )
    freeze = {
        "schema_version": 1,
        "stage": "x4_train_target_freeze_v1",
        "study": protocol["study"],
        "arm": args.arm,
        "source_kind": args.source_kind,
        "source_commit": args.source_commit,
        "protocol_sha256": protocol_sha,
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "source_freeze_sha256": source_freeze_sha,
        "candidate_manifest_sha256": (
            candidate_summary["manifest_sha256"] if candidate_summary else None
        ),
        "candidate_pseudo_manifest_sha256": (
            args.candidate_pseudo_manifest_sha256 if candidate_summary else None
        ),
        "manifest_sha256": manifest_sha,
        "images": counts["images"],
        "tumor_images": counts["tumor"],
        "normal_images": counts["normal"],
        "tumor_empty_targets": tumor_empty,
        "total_foreground_pixels": total_foreground,
        "native_resolution_masks": True,
        "native_geometry_reference": (
            "canonical_manifest_plus_image_bytes"
            if root is not None
            else "canonical_manifest"
        ),
        "normal_targets_explicitly_empty": True,
        "train_spatial_annotations_read": 0,
        "targets_frozen_before_outer_validation_gt": True,
        "outer_validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_bytes_before_freeze": output_bytes,
    }
    freeze_path = args.output_dir / "x4_target_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**freeze, "x4_target_freeze_sha256": sha256_file(freeze_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
