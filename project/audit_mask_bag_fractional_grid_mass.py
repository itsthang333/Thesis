from __future__ import annotations

"""GT-blind audit of proposal mass after square-frame projection.

This tool consumes only the frozen split rows, source radiographs, image
labels, and prediction-first candidate payloads. It never imports the
segmentation dataset or reads a BTXRD annotation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mae_reconstruction_io import (
    load_split_rows_without_annotations,
    locate_verified_image,
    sha256_file,
)
from models.rad_dino_mask_bag_mil import project_direct_resize_masks_to_square
from pseudo.candidate_diagnostics import validate_candidate_diagnostics_manifest


MASS_BINS = (
    ("mass_0p25_to_0p5", 0.25, 0.5),
    ("mass_0p5_to_1", 0.5, 1.0),
    ("mass_1_to_2", 1.0, 2.0),
    ("mass_ge_2", 2.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-pseudo-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-grid-size", type=int, default=32)
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument("--minimum-grid-mass", type=float, default=0.25)
    parser.add_argument("--maximum-candidates", type=int, default=81)
    return parser.parse_args()


def _load_payload(
    root: Path,
    manifest_row: dict[str, str],
    *,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = root / manifest_row["diagnostic_path"]
    if sha256_file(path) != manifest_row["diagnostic_sha256"]:
        raise ValueError(f"Candidate payload hash mismatch: {manifest_row['image_name']}")
    with np.load(path, allow_pickle=False) as payload:
        masks = payload["sam_masks"].astype(np.float32)
        prompt_map = payload["prompt_map"].astype(np.float32)
        modes = (
            payload["prompt_modes"].astype("U32")
            if "prompt_modes" in payload
            else np.full(len(masks), "unassigned", dtype="U32")
        )
        sources = (
            payload["proposal_source_ids"].astype("U32")
            if "proposal_source_ids" in payload
            else np.full(len(masks), "unassigned", dtype="U32")
        )
    if masks.ndim != 3 or prompt_map.ndim != 2:
        raise ValueError("Candidate payload has invalid spatial dimensions")
    if len(masks) > maximum_candidates:
        raise RuntimeError("Candidate bag exceeds the frozen maximum")
    if len(modes) != len(masks) or len(sources) != len(masks):
        raise ValueError("Candidate provenance arrays are not aligned")
    if len(masks):
        fallback = np.zeros(len(masks), dtype=np.uint8)
        return masks, modes, sources, fallback

    threshold = float(np.percentile(prompt_map, 90.0))
    candidate = (prompt_map >= threshold) & (prompt_map > 0)
    if not candidate.any():
        candidate = np.zeros_like(prompt_map, dtype=bool)
        height, width = candidate.shape
        candidate[height // 4 : height - height // 4, width // 4 : width - width // 4] = True
    return (
        candidate[None].astype(np.float32),
        np.asarray(["fallback"], dtype="U32"),
        np.asarray(["fallback"], dtype="U32"),
        np.ones(1, dtype=np.uint8),
    )


def _projected_grid_mass(
    masks: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    token_grid_size: int,
    oversampling: int,
) -> tuple[np.ndarray, np.ndarray]:
    side = max(image_width, image_height)
    left = (side - image_width) // 2
    top = (side - image_height) // 2
    output_size = token_grid_size * oversampling
    projected = project_direct_resize_masks_to_square(
        torch.from_numpy(masks),
        padded_side=side,
        content_box=(left, top, left + image_width, top + image_height),
        output_size=output_size,
    )
    grid = F.interpolate(
        projected[:, None],
        size=(token_grid_size, token_grid_size),
        mode="area",
    )[:, 0]
    flipped_grid = F.interpolate(
        projected.flip(-1)[:, None],
        size=(token_grid_size, token_grid_size),
        mode="area",
    )[:, 0]
    masses = grid.sum(dim=(-2, -1)).cpu().numpy().astype(np.float64)
    flipped_masses = (
        flipped_grid.sum(dim=(-2, -1)).cpu().numpy().astype(np.float64)
    )
    if (
        not np.isfinite(masses).all()
        or not np.isfinite(flipped_masses).all()
        or (masses < 0).any()
        or (flipped_masses < 0).any()
    ):
        raise RuntimeError("Projected grid masses are invalid")
    return masses, flipped_masses


def _summarize(values: np.ndarray, *, minimum_grid_mass: float) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    retained = values[values >= minimum_grid_mass]
    summary: dict[str, object] = {
        "candidates": int(values.size),
        "retained": int(retained.size),
        "rejected_below_minimum": int(values.size - retained.size),
        "minimum_grid_mass": float(minimum_grid_mass),
    }
    if retained.size:
        summary["retained_quantiles"] = {
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"),
                np.quantile(retained, (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)),
                strict=True,
            )
        }
    else:
        summary["retained_quantiles"] = {}
    for name, low, high in MASS_BINS:
        summary[name] = int(((retained >= low) & (retained < high)).sum())
    summary["retained_below_one"] = int((retained < 1.0).sum())
    summary["retained_below_one_fraction"] = (
        float((retained < 1.0).mean()) if retained.size else 0.0
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.token_grid_size != 32 or args.oversampling != 4:
        raise ValueError("This audit is frozen to a 32x32 grid and fourfold oversampling")
    if args.minimum_grid_mass != 0.25 or args.maximum_candidates != 81:
        raise ValueError("This audit must match the parent mask-bag configuration")
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
    )
    indexed, candidate_audit = validate_candidate_diagnostics_manifest(
        args.candidate_root,
        expected_image_names=[row["image_id"] for row in rows],
        split=args.split,
        expected_pseudo_manifest_sha256=args.expected_pseudo_manifest_sha256,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
    )

    output_rows: list[dict[str, object]] = []
    for row in rows:
        stem = Path(row["image_id"]).stem
        masks, modes, sources, fallback = _load_payload(
            args.candidate_root,
            indexed[stem],
            maximum_candidates=args.maximum_candidates,
        )
        image_path = locate_verified_image(args.dataset_root, row)
        with Image.open(image_path) as image:
            width, height = image.size
        masses, flipped_masses = _projected_grid_mass(
            masks,
            image_width=width,
            image_height=height,
            token_grid_size=args.token_grid_size,
            oversampling=args.oversampling,
        )
        retained = masses >= args.minimum_grid_mass
        flipped_retained = flipped_masses >= args.minimum_grid_mass
        if not np.array_equal(retained, flipped_retained):
            raise RuntimeError(
                f"Original/flip candidate validity differs for {row['image_id']}"
            )
        if not np.allclose(masses, flipped_masses, rtol=0.0, atol=1.0e-5):
            raise RuntimeError(
                f"Original/flip grid mass differs for {row['image_id']}"
            )
        for index, (
            mask,
            mode,
            source,
            is_fallback,
            mass,
            flipped_mass,
        ) in enumerate(
            zip(
                masks,
                modes,
                sources,
                fallback,
                masses,
                flipped_masses,
                strict=True,
            )
        ):
            output_rows.append(
                {
                    "image_id": row["image_id"],
                    "group_id": row["group_id"],
                    "image_label": int(row["tumor"]),
                    "candidate_index": index,
                    "prompt_mode": str(mode),
                    "proposal_source": str(source),
                    "fallback": int(is_fallback),
                    "raw_area_pixels_320": int((mask > 0.5).sum()),
                    "grid_mass": float(mass),
                    "flip_grid_mass": float(flipped_mass),
                    "absolute_flip_mass_delta": float(abs(mass - flipped_mass)),
                    "retained": int(mass >= args.minimum_grid_mass),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / f"{args.split}_fractional_grid_mass.csv"
    fields = list(output_rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    masses = np.asarray([float(row["grid_mass"]) for row in output_rows])
    labels = np.asarray([int(row["image_label"]) for row in output_rows])
    modes = np.asarray([str(row["prompt_mode"]) for row in output_rows])
    sources = np.asarray([str(row["proposal_source"]) for row in output_rows])
    fallback = np.asarray([int(row["fallback"]) for row in output_rows])
    summary = {
        "audit": "mask_bag_fractional_grid_mass_v1",
        "split": args.split,
        "prediction_first": True,
        "ground_truth_loaded": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "images": len(rows),
        "candidate_manifest_sha256": candidate_audit["manifest_sha256"],
        "candidate_summary_sha256": candidate_audit["summary_sha256"],
        "csv_sha256": sha256_file(csv_path),
        "maximum_absolute_flip_mass_delta": float(
            max(float(row["absolute_flip_mass_delta"]) for row in output_rows)
        ),
        "original_flip_validity_aligned": True,
        "overall": _summarize(masses, minimum_grid_mass=args.minimum_grid_mass),
        "by_image_label": {
            str(label): _summarize(
                masses[labels == label],
                minimum_grid_mass=args.minimum_grid_mass,
            )
            for label in (0, 1)
        },
        "by_prompt_mode": {
            mode: _summarize(
                masses[modes == mode],
                minimum_grid_mass=args.minimum_grid_mass,
            )
            for mode in sorted(set(modes.tolist()))
        },
        "by_proposal_source": {
            source: _summarize(
                masses[sources == source],
                minimum_grid_mass=args.minimum_grid_mass,
            )
            for source in sorted(set(sources.tolist()))
        },
        "by_fallback": {
            str(value): _summarize(
                masses[fallback == value],
                minimum_grid_mass=args.minimum_grid_mass,
            )
            for value in (0, 1)
        },
    }
    summary_path = args.output_dir / f"{args.split}_fractional_grid_mass_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
