from __future__ import annotations

"""Merge disjoint prediction-first candidate-supply shards without GT."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from frozen_io import load_split_rows_without_annotations, sha256_file
from pseudo.candidate_diagnostics import write_candidate_diagnostics_manifest
from pseudo.manifest import write_pseudo_mask_manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=args.expected_split_sha256,
        split=args.split,
    )
    expected_names = [str(row["image_id"]) for row in rows]
    expected_stems = {Path(name).stem for name in expected_names}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "masks").mkdir()
    (args.output_dir / "candidate_diagnostics").mkdir()
    pseudo_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    source_hashes: list[dict[str, str]] = []
    seen: set[str] = set()
    image_size: int | None = None
    for shard in args.shard_root:
        pseudo_summary = json.loads((shard / "pseudo_mask_summary.json").read_text(encoding="utf-8"))
        diagnostic_summary = json.loads((shard / "candidate_diagnostics_summary.json").read_text(encoding="utf-8"))
        if pseudo_summary.get("split") != args.split or diagnostic_summary.get("split") != args.split:
            raise ValueError("Shard split differs")
        local_size = int(pseudo_summary["image_size"])
        image_size = local_size if image_size is None else image_size
        if local_size != image_size or int(diagnostic_summary["image_size"]) != image_size:
            raise ValueError("Shard image geometry differs")
        local_pseudo = read_csv(shard / "pseudo_mask_manifest.csv")
        local_diagnostic = read_csv(shard / "candidate_diagnostics_manifest.csv")
        by_stem = {Path(row["image_name"]).stem: row for row in local_diagnostic}
        if len(by_stem) != len(local_diagnostic) or len(local_pseudo) != len(local_diagnostic):
            raise ValueError("Shard manifests differ or contain duplicate IDs")
        for pseudo_row in local_pseudo:
            stem = Path(pseudo_row["image_name"]).stem
            if stem in seen or stem not in expected_stems or stem not in by_stem:
                raise ValueError(f"Unexpected/duplicate shard image {stem}")
            seen.add(stem)
            mask_source = shard / pseudo_row["mask_path"]
            diagnostic_row = by_stem[stem]
            diagnostic_source = shard / diagnostic_row["diagnostic_path"]
            if sha256_file(mask_source) != pseudo_row["mask_sha256"]:
                raise ValueError(f"Shard mask hash mismatch: {stem}")
            if sha256_file(diagnostic_source) != diagnostic_row["diagnostic_sha256"]:
                raise ValueError(f"Shard diagnostic hash mismatch: {stem}")
            shutil.copy2(mask_source, args.output_dir / "masks" / f"{stem}.png")
            shutil.copy2(diagnostic_source, args.output_dir / "candidate_diagnostics" / f"{stem}.npz")
            pseudo_rows.append(pseudo_row)
            diagnostic_rows.append(diagnostic_row)
        source_hashes.append({
            "root": str(shard),
            "pseudo_manifest_sha256": sha256_file(shard / "pseudo_mask_manifest.csv"),
            "candidate_manifest_sha256": sha256_file(shard / "candidate_diagnostics_manifest.csv"),
        })
    if seen != expected_stems:
        raise ValueError(f"Merged shards incomplete: {len(seen)}/{len(expected_stems)}")
    run_metadata = {
        "stage": "dsll_candidate_supply_merged_shards_v1",
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "split": args.split,
        "shards": source_hashes,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    run_path = args.output_dir / "run_metadata.json"
    run_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert image_size is not None
    pseudo_summary = write_pseudo_mask_manifest(
        args.output_dir,
        pseudo_rows,
        expected_image_names=expected_names,
        split=args.split,
        image_size=image_size,
        run_metadata_sha256=sha256_file(run_path),
    )
    candidate_summary = write_candidate_diagnostics_manifest(
        args.output_dir,
        diagnostic_rows,
        expected_image_names=expected_names,
        split=args.split,
        image_size=image_size,
        pseudo_manifest_sha256=str(pseudo_summary["manifest_sha256"]),
        selection_method="dsll_source_specific_coverage_mass_sam",
        support_clip_kernel=5,
        cam_percentile=90.0,
        cohort="all",
    )
    run_metadata["pseudo_manifest_sha256"] = pseudo_summary["manifest_sha256"]
    run_metadata["candidate_manifest_sha256"] = candidate_summary["manifest_sha256"]
    run_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The pseudo summary binds run_metadata; update once after final metadata.
    pseudo_summary = write_pseudo_mask_manifest(
        args.output_dir,
        pseudo_rows,
        expected_image_names=expected_names,
        split=args.split,
        image_size=image_size,
        run_metadata_sha256=sha256_file(run_path),
    )
    candidate_summary = write_candidate_diagnostics_manifest(
        args.output_dir,
        diagnostic_rows,
        expected_image_names=expected_names,
        split=args.split,
        image_size=image_size,
        pseudo_manifest_sha256=str(pseudo_summary["manifest_sha256"]),
        selection_method="dsll_source_specific_coverage_mass_sam",
        support_clip_kernel=5,
        cam_percentile=90.0,
        cohort="all",
    )
    print(json.dumps({
        "images": len(seen),
        "pseudo_manifest_sha256": pseudo_summary["manifest_sha256"],
        "candidate_manifest_sha256": candidate_summary["manifest_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
