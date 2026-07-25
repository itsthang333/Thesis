from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SOURCE_COMMIT = "8a997c87170538f897e6aa3b13b0f6c13e39f32f"
EXPECTED_SOURCE_HASHES = {
    "generate_biomedclip_saliency.py": (
        "c475f3b8bd16b3b2fd85add21cf35e4b631943ab50be9c4217302dba763ed46c"
    ),
    "models/biomedclip_saliency.py": (
        "f07718a47d71c0aa05c6e110c243d3dfc7197064cca89d5e1dfb44612fd28d5d"
    ),
}
EXPECTED_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
EXPECTED_MODEL_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
EXPECTED_PROMPTS = {
    "tumor": [
        "A bone radiograph showing a bone tumor.",
        "An x-ray image with a bone neoplasm.",
        "An x-ray image showing an abnormal bone lesion.",
    ],
    "normal": [
        "A normal bone radiograph without a tumor.",
        "An x-ray image of healthy bone.",
        "An x-ray image without a bone lesion.",
    ],
}
EXPECTED_VIEW_CONTRACT = {
    "saliency_reduction": "channelwise mean absolute gradient-times-activation",
    "target_layer": "model.visual.trunk.blocks[11].norm1",
    "full_view": "black pad to square",
    "crop_fraction_of_short_side": 0.5,
    "positions_per_axis": 3,
    "top_k_tiles_by_contrast_score": 3,
    "normalization_percentiles": [1.0, 99.0],
    "fusion": "pixelwise maximum of full view and selected tiles",
    "output_size": 320,
}
EXPECTED_POPULATIONS = {
    "train": {"images": 2981, "tumor": 1488, "normal": 1493},
    "val": {"images": 371, "tumor": 184, "normal": 187},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_expected_rows(split_manifest: Path, split: str) -> list[dict[str, str]]:
    if split not in EXPECTED_POPULATIONS:
        raise ValueError("Only train/val saliency can be audited")
    if sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen split SHA-256 mismatch")
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == split and row.get("eligible") == "1"
        ]
    population = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if population != EXPECTED_POPULATIONS[split]:
        raise ValueError(f"Frozen {split} population mismatch: {population}")
    return rows


def parse_json_list(value: str, name: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list")
    return parsed


def finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def validate_manifest_rows(
    root: Path,
    manifest_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    *,
    output_size: int = 320,
) -> dict[str, Any]:
    expected_names = [row["image_id"] for row in expected_rows]
    actual_names = [row.get("image_id", "") for row in manifest_rows]
    if actual_names != expected_names:
        raise ValueError("Saliency manifest identities/order differ from frozen split")
    expected_by_name = {row["image_id"]: row for row in expected_rows}
    observed_paths: set[Path] = set()
    tumor_dynamic_ranges = []
    for index, row in enumerate(manifest_rows):
        name = row["image_id"]
        expected = expected_by_name[name]
        label = int(row["tumor_image_label"])
        if label != int(expected["tumor"]):
            raise ValueError(f"Image-label mismatch at manifest row {index}")
        for key in ("group_id", "split"):
            if row[key] != expected[key]:
                raise ValueError(f"{key} mismatch at manifest row {index}")
        if row["source_image_sha256"] != expected["image_sha256"]:
            raise ValueError(f"Source-image hash mismatch at manifest row {index}")
        if int(row["source_width"]) != int(expected["width"]) or int(
            row["source_height"]
        ) != int(expected["height"]):
            raise ValueError(f"Source dimensions mismatch at manifest row {index}")
        if int(row["map_height"]) != output_size or int(row["map_width"]) != output_size:
            raise ValueError(f"Recorded map dimensions mismatch at manifest row {index}")
        expected_relative = Path("maps") / f"{Path(name).stem}.npy"
        relative = Path(row["map_path"])
        if relative != expected_relative:
            raise ValueError(f"Unexpected map path at manifest row {index}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"Map path escapes root at manifest row {index}") from error
        if not path.is_file() or sha256_file(path) != row["map_sha256"]:
            raise ValueError(f"Map file/hash mismatch at manifest row {index}")
        if path in observed_paths:
            raise ValueError(f"Duplicate map path at manifest row {index}")
        observed_paths.add(path)
        values = np.load(path, allow_pickle=False)
        if values.dtype != np.float16 or values.shape != (output_size, output_size):
            raise ValueError(f"Map dtype/shape mismatch at manifest row {index}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite map at manifest row {index}")
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.astype(np.float32).mean())
        dynamic_range = maximum - minimum
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"Map range mismatch at manifest row {index}")
        recorded = {
            "map_min": finite_float(row["map_min"], f"map_min {index}"),
            "map_max": finite_float(row["map_max"], f"map_max {index}"),
            "map_mean": finite_float(row["map_mean"], f"map_mean {index}"),
            "map_dynamic_range": finite_float(
                row["map_dynamic_range"], f"map_dynamic_range {index}"
            ),
        }
        # Generator summaries are computed before float16 serialization.
        if abs(recorded["map_min"] - minimum) > 5e-4:
            raise ValueError(f"Map minimum mismatch at manifest row {index}")
        if abs(recorded["map_max"] - maximum) > 5e-4:
            raise ValueError(f"Map maximum mismatch at manifest row {index}")
        if abs(recorded["map_mean"] - mean) > 5e-4:
            raise ValueError(f"Map mean mismatch at manifest row {index}")
        if abs(recorded["map_dynamic_range"] - dynamic_range) > 1e-3:
            raise ValueError(f"Map dynamic range mismatch at manifest row {index}")
        selected = parse_json_list(row["selected_tiles"], f"selected_tiles {index}")
        scores = parse_json_list(row["all_tile_scores"], f"all_tile_scores {index}")
        if label == 0:
            if (
                row["generation"] != "known_normal_image_label_empty"
                or np.count_nonzero(values) != 0
                or recorded["map_dynamic_range"] != 0.0
                or row["full_contrast_score"] not in ("", None)
                or selected
                or scores
            ):
                raise ValueError(f"Known-normal empty-map contract failed at row {index}")
        else:
            if row["generation"] != "frozen_biomedclip_full_plus_top3_tiles":
                raise ValueError(f"Tumor generation contract failed at row {index}")
            finite_float(row["full_contrast_score"], f"full contrast score {index}")
            if len(selected) != 3 or len(scores) != 9:
                raise ValueError(f"Tiled-view count mismatch at manifest row {index}")
            if any(not math.isfinite(float(score)) for score in scores):
                raise ValueError(f"Non-finite tile score at manifest row {index}")
            if recorded["map_dynamic_range"] <= 1e-6:
                raise ValueError(f"Constant tumor saliency at manifest row {index}")
            selected_scores = [finite_float(item["contrast_score"], "selected score") for item in selected]
            expected_top = sorted((float(score) for score in scores), reverse=True)[:3]
            if any(abs(a - b) > 1e-12 for a, b in zip(selected_scores, expected_top)):
                raise ValueError(f"Selected tiles are not top-three at row {index}")
            tumor_dynamic_ranges.append(recorded["map_dynamic_range"])

    disk_maps = {path.resolve() for path in (root / "maps").glob("*.npy")}
    if disk_maps != observed_paths:
        raise ValueError("Map directory contains missing or unmanifested .npy files")
    return {
        "images": len(manifest_rows),
        "tumor": sum(int(row["tumor_image_label"]) for row in manifest_rows),
        "normal": sum(1 - int(row["tumor_image_label"]) for row in manifest_rows),
        "tumor_dynamic_range_min": min(tumor_dynamic_ranges),
        "map_files": len(observed_paths),
    }


def audit(
    root: Path,
    split_manifest: Path,
    local_project: Path,
    *,
    split: str,
    expected_model_weight_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    metadata_path = root / "run_metadata.json"
    manifest_path = root / "saliency_manifest.csv"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("BiomedCLIP saliency evidence is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("BiomedCLIP saliency source commit mismatch")
    if metadata.get("source_files") != EXPECTED_SOURCE_HASHES:
        raise ValueError("Cloud saliency source hashes mismatch")
    for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
        if sha256_file(local_project / relative) != expected_hash:
            raise ValueError(f"Local source hash mismatch: {relative}")
    if metadata.get("split") != split:
        raise ValueError("Saliency metadata split mismatch")
    if metadata.get("population") != EXPECTED_POPULATIONS[split]:
        raise ValueError("Saliency metadata population mismatch")
    if metadata.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Saliency metadata split hash mismatch")
    model = metadata.get("model", {})
    if model.get("id") != EXPECTED_MODEL_ID:
        raise ValueError("BiomedCLIP model ID mismatch")
    if model.get("weight_sha256") != expected_model_weight_sha256:
        raise ValueError("BiomedCLIP physical weight hash mismatch")
    if int(model.get("weight_bytes", 0)) <= 1_000_000:
        raise ValueError("BiomedCLIP physical weight size is invalid")
    prompts = metadata.get("prompts", {})
    if prompts.get("tumor") != EXPECTED_PROMPTS["tumor"] or prompts.get(
        "normal"
    ) != EXPECTED_PROMPTS["normal"]:
        raise ValueError("BiomedCLIP prompt text drift")
    if prompts.get("sha256") != sha256_json(EXPECTED_PROMPTS):
        raise ValueError("BiomedCLIP prompt hash drift")
    if metadata.get("view_contract") != EXPECTED_VIEW_CONTRACT:
        raise ValueError("BiomedCLIP view contract drift")
    if metadata.get("validation_gt_read") is not False:
        raise ValueError("Validation GT was not locked during saliency generation")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Test was not locked during saliency generation")
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Saliency manifest SHA-256 mismatch")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    expected_rows = load_expected_rows(split_manifest, split)
    population = validate_manifest_rows(root, manifest_rows, expected_rows)
    return {
        "status": "PASS",
        "role": "prediction-first BiomedCLIP saliency audit",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "split": split,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "population": population,
        "model_weight_sha256": expected_model_weight_sha256,
        "prompt_sha256": sha256_json(EXPECTED_PROMPTS),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "validation_gt_read": False,
        "test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--local-project", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--expected-model-weight-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.root,
        args.split_manifest,
        args.local_project,
        split=args.split,
        expected_model_weight_sha256=args.expected_model_weight_sha256,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
