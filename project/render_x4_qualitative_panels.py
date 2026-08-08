from __future__ import annotations

"""Render the protocol-frozen X4 X10 qualitative panel.

Case identities are frozen by ``select_x4_qualitative_cases.py``.  This
renderer verifies that freeze, the direct E5 choices, and every candidate
payload before it opens a validation annotation.  Optional student/fully
prediction bundles are hash-checked from a JSON specification, so the final
figure can be regenerated as the remaining X4 arms complete without changing
the selected cases.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.btxrd import _decode_labelme_polygon_mask, resolve_btxrd_root
from frozen_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
    validate_sha256,
)


METHOD_ORDER = (
    "cam_student",
    "puzzlecam_student",
    "s2c_student",
    "rich_gallery_student",
    "fully_supervised",
)
METHOD_TITLES = {
    "cam_student": "CAM student",
    "puzzlecam_student": "PuzzleCAM student",
    "s2c_student": "S2C student",
    "rich_gallery_student": "Rich student",
    "fully_supervised": "Fully supervised",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def normalize_radiograph(image: Image.Image) -> Image.Image:
    values = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        normalized = np.zeros_like(values, dtype=np.uint8)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        normalized = np.rint(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def normalize_score_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Qualitative score map must be finite and two-dimensional")
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    binary = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L")
    binary = binary.resize(base.size, Image.Resampling.NEAREST)
    layer = Image.new("RGB", base.size, color)
    return Image.composite(Image.blend(base, layer, 0.48), base, binary)


def overlay_heatmap(image: Image.Image, values: np.ndarray) -> Image.Image:
    base = image.convert("RGB")
    score = normalize_score_map(values)
    # A small deterministic blue-yellow map, avoiding a runtime matplotlib dependency.
    red = np.clip(2.0 * score - 0.15, 0.0, 1.0)
    green = np.clip(1.7 * score - 0.35, 0.0, 1.0)
    blue = np.clip(1.0 - 1.25 * score, 0.0, 1.0)
    rgb = np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    heat = Image.fromarray(rgb, mode="RGB").resize(base.size, Image.Resampling.BILINEAR)
    alpha = Image.fromarray(np.rint(score * 180.0).astype(np.uint8), mode="L").resize(
        base.size, Image.Resampling.BILINEAR
    )
    return Image.composite(Image.blend(base, heat, 0.62), base, alpha)


def title_tile(content: Image.Image, title: str, *, tile_size: int) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + 34), "white")
    canvas.paste(content.resize((tile_size, tile_size), Image.Resampling.BILINEAR), (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 9), title, fill="black", font=ImageFont.load_default())
    return canvas


def candidate_montage(
    image: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    tile_size: int,
    count: int = 9,
) -> Image.Image:
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("Candidate montage masks/scores differ")
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    if len(masks) == 0:
        return canvas
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=np.float64)))
    indices = order[: min(count, len(order))]
    side = 3
    cell = tile_size // side
    for position, index in enumerate(indices):
        row, column = divmod(position, side)
        tile = overlay_mask(image, masks[index], (0, 220, 255)).resize(
            (cell, cell), Image.Resampling.BILINEAR
        )
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 38, 12), fill="black")
        draw.text((2, 1), f"#{int(index)}", fill="white", font=ImageFont.load_default())
        canvas.paste(tile, (column * cell, row * cell))
    return canvas


def load_prediction_bundles(spec_path: Path | None) -> dict[str, dict[str, object]]:
    if spec_path is None:
        return {}
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) - set(METHOD_ORDER):
        raise ValueError("X10 prediction specification has an unknown method")
    bundles: dict[str, dict[str, object]] = {}
    for method, raw in payload.items():
        if not isinstance(raw, dict):
            raise ValueError(f"X10 prediction specification for {method} is invalid")
        root = Path(str(raw["root"]))
        manifest = root / str(raw.get("manifest", "prediction_manifest.csv"))
        freeze = root / str(raw.get("freeze", "prediction_freeze.json"))
        expected_freeze = validate_sha256(
            str(raw["expected_freeze_sha256"]), name=f"{method} freeze SHA-256"
        )
        if sha256_file(freeze) != expected_freeze:
            raise ValueError(f"X10 {method} freeze hash mismatch")
        rows = read_csv(manifest)
        indexed = {row["image_id"]: row for row in rows}
        if len(rows) != 371 or len(indexed) != 371:
            raise ValueError(f"X10 {method} prediction cohort must contain 371 images")
        for image_id, row in indexed.items():
            mask_path = root / safe_relative_path(row["mask_path"])
            if sha256_file(mask_path) != row["mask_sha256"]:
                raise ValueError(f"X10 {method} mask changed: {image_id}")
        bundles[method] = {"root": root, "rows": indexed, "freeze_sha256": expected_freeze}
    return bundles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--expected-selection-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--choices-csv", type=Path, required=True)
    parser.add_argument("--expected-choices-sha256", required=True)
    parser.add_argument("--direct-per-image", type=Path, required=True)
    parser.add_argument("--expected-direct-per-image-sha256", required=True)
    parser.add_argument("--direct-arm", default="E5_exact__cap243")
    parser.add_argument("--prediction-spec-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.tile_size < 96:
        raise ValueError("X10 tile size is too small")
    expected_split = validate_sha256(args.expected_split_sha256, name="split SHA-256")
    selection_freeze_path = args.selection_root / "selection_freeze.json"
    selection_manifest_path = args.selection_root / "selection_manifest.csv"
    if sha256_file(selection_freeze_path) != validate_sha256(
        args.expected_selection_freeze_sha256, name="selection freeze SHA-256"
    ):
        raise ValueError("X10 selection freeze hash mismatch")
    selection_freeze = json.loads(selection_freeze_path.read_text(encoding="utf-8"))
    if (
        selection_freeze.get("selection_before_image_or_gt_rendering") is not True
        or selection_freeze.get("selection_by_visual_appeal") is not False
        or selection_freeze.get("selection_manifest_sha256") != sha256_file(selection_manifest_path)
        or selection_freeze.get("test_images_read") != 0
        or selection_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("X10 case selection is not a valid pre-render freeze")
    selections = [row for row in read_csv(selection_manifest_path) if row["available"] == "1"]
    selected_ids = [row["image_id"] for row in selections]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("X10 selected cases must be unique")

    split_rows = load_split_rows_without_annotations(
        args.split_manifest, expected_sha256=expected_split, split="val", allow_test=False
    )
    split_by_id = {row["image_id"]: row for row in split_rows}
    if not set(selected_ids).issubset(split_by_id):
        raise ValueError("X10 selected cases differ from the canonical validation split")

    if sha256_file(args.direct_per_image) != validate_sha256(
        args.expected_direct_per_image_sha256, name="direct per-image SHA-256"
    ):
        raise ValueError("X10 direct per-image table changed")
    direct_rows = {
        row["image_id"]: row
        for row in read_csv(args.direct_per_image)
        if row.get("arm") == args.direct_arm
    }
    if len(direct_rows) != 371:
        raise ValueError("X10 direct arm must contain 371 rows")

    if sha256_file(args.choices_csv) != validate_sha256(
        args.expected_choices_sha256, name="choices SHA-256"
    ):
        raise ValueError("X10 frozen choice table changed")
    choices = {
        row["image_id"]: row
        for row in read_csv(args.choices_csv)
        if row.get("arm") == args.direct_arm
    }
    if len(choices) != 371:
        raise ValueError("X10 frozen direct choices must contain 371 rows")

    candidate_manifest_path = args.candidate_root / "candidate_diagnostics_manifest.csv"
    if sha256_file(candidate_manifest_path) != validate_sha256(
        args.expected_candidate_manifest_sha256, name="candidate manifest SHA-256"
    ):
        raise ValueError("X10 candidate manifest changed")
    candidates = {row["image_name"]: row for row in read_csv(candidate_manifest_path)}
    if len(candidates) != 371:
        raise ValueError("X10 candidate manifest must contain 371 images")
    candidate_paths: dict[str, Path] = {}
    for image_id in selected_ids:
        row = candidates[image_id]
        path = args.candidate_root / safe_relative_path(row["diagnostic_path"])
        if sha256_file(path) != row["diagnostic_sha256"]:
            raise ValueError(f"X10 candidate payload changed: {image_id}")
        if choices[image_id]["candidate_payload_sha256"] != row["diagnostic_sha256"]:
            raise ValueError(f"X10 choice/candidate binding differs: {image_id}")
        if int(choices[image_id]["selected_candidate_index"]) != int(
            direct_rows[image_id]["selected_candidate_index"]
        ):
            raise ValueError(f"X10 direct selected index differs: {image_id}")
        candidate_paths[image_id] = path

    prediction_bundles = load_prediction_bundles(args.prediction_spec_json)
    btxrd_root = resolve_btxrd_root(args.dataset_root)
    # Verify source image bytes before the first annotation is opened.
    image_paths = {
        image_id: locate_verified_image(btxrd_root, split_by_id[image_id])
        for image_id in selected_ids
    }

    args.output_dir.mkdir(parents=True)
    panel_rows: list[dict[str, object]] = []
    annotations_opened = 0
    for selection in selections:
        image_id = selection["image_id"]
        with Image.open(image_paths[image_id]) as handle:
            radiograph = normalize_radiograph(handle)
            width, height = handle.size
        with np.load(candidate_paths[image_id], allow_pickle=False) as payload:
            masks = np.asarray(payload["sam_masks"], dtype=bool)
            prompt_map = np.asarray(payload["prompt_map"], dtype=np.float32)
            scores = np.asarray(payload["selection_scores"], dtype=np.float32)
        selected_index = int(direct_rows[image_id]["selected_candidate_index"])
        oracle_index = int(direct_rows[image_id]["full_gallery_oracle_candidate_index"])
        selected_mask = masks[selected_index] if selected_index >= 0 else np.zeros(masks.shape[1:], bool)
        oracle_mask = masks[oracle_index] if oracle_index >= 0 else np.zeros(masks.shape[1:], bool)

        panels: list[tuple[str, Image.Image]] = [
            ("X-ray", radiograph),
            ("CAM/localization", overlay_heatmap(radiograph, prompt_map)),
            (
                "Proposal gallery",
                candidate_montage(radiograph, masks, scores, tile_size=args.tile_size),
            ),
            ("Selected mask", overlay_mask(radiograph, selected_mask, (255, 70, 40))),
            ("Oracle mask", overlay_mask(radiograph, oracle_mask, (40, 220, 80))),
        ]
        method_availability: dict[str, bool] = {}
        for method in METHOD_ORDER:
            bundle = prediction_bundles.get(method)
            if bundle is None:
                placeholder = Image.new("RGB", radiograph.size, (235, 235, 235))
                ImageDraw.Draw(placeholder).text(
                    (10, 10), "awaiting frozen output", fill="black", font=ImageFont.load_default()
                )
                panels.append((METHOD_TITLES[method], placeholder))
                method_availability[method] = False
                continue
            prediction_row = bundle["rows"][image_id]
            prediction_path = bundle["root"] / safe_relative_path(prediction_row["mask_path"])
            with Image.open(prediction_path) as handle:
                prediction = np.asarray(handle.convert("L")) > 0
            panels.append((METHOD_TITLES[method], overlay_mask(radiograph, prediction, (255, 140, 0))))
            method_availability[method] = True

        if int(split_by_id[image_id]["tumor"]):
            target = _decode_labelme_polygon_mask(
                btxrd_root / "Annotations" / f"{Path(image_id).stem}.json",
                height=height,
                width=width,
            )
            annotations_opened += 1
        else:
            target = np.zeros((height, width), dtype=bool)
        panels.append(("Ground truth", overlay_mask(radiograph, target, (30, 255, 30))))

        tiles = [title_tile(content, title, tile_size=args.tile_size) for title, content in panels]
        panel = Image.new("RGB", (args.tile_size * len(tiles), args.tile_size + 34), "white")
        for index, tile in enumerate(tiles):
            panel.paste(tile, (index * args.tile_size, 0))
        panel_path = args.output_dir / f"{selection['category']}__{Path(image_id).stem}.png"
        panel.save(panel_path, format="PNG", optimize=True)
        panel_rows.append(
            {
                **selection,
                "panel_path": panel_path.name,
                "panel_sha256": sha256_file(panel_path),
                "selected_candidate_index": selected_index,
                "oracle_candidate_index": oracle_index,
                "direct_dice": direct_rows[image_id]["dice"],
                **{f"{method}_available": int(method_availability[method]) for method in METHOD_ORDER},
            }
        )

    panel_manifest = args.output_dir / "panel_manifest.csv"
    with panel_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(panel_rows[0]))
        writer.writeheader()
        writer.writerows(panel_rows)
    report = {
        "schema_version": 1,
        "stage": "x4_qualitative_panel_render_v1",
        "selected_cases": len(panel_rows),
        "selection_freeze_sha256": args.expected_selection_freeze_sha256,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "candidate_manifest_sha256": args.expected_candidate_manifest_sha256,
        "choices_sha256": args.expected_choices_sha256,
        "direct_per_image_sha256": args.expected_direct_per_image_sha256,
        "prediction_bundle_freezes": {
            method: prediction_bundles[method]["freeze_sha256"] for method in prediction_bundles
        },
        "panel_manifest_sha256": sha256_file(panel_manifest),
        "all_provenance_verified_before_annotations": True,
        "selection_frozen_before_image_or_gt_rendering": True,
        "selection_by_visual_appeal": False,
        "validation_annotations_opened": annotations_opened,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    report_path = args.output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "render_report_sha256": sha256_file(report_path)}, indent=2))


if __name__ == "__main__":
    main()
