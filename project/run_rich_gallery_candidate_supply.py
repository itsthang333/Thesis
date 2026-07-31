from __future__ import annotations

"""Generate full-cohort GT-blind candidate galleries for rich-gallery G0/G1."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_COUNTS = {
    "train": {"images": 2981, "tumor": 1488, "normal": 1493},
    "val": {"images": 371, "tumor": 184, "normal": 187},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_counts(split_manifest: Path, split: str) -> dict[str, int]:
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == split and row.get("eligible") == "1"
        ]
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image IDs in {split}")
    result = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if result != EXPECTED_COUNTS[split]:
        raise ValueError(f"Canonical {split} counts differ: {result}")
    return result


def common_generation_args(
    *,
    source_root: Path,
    data_root: Path,
    split_manifest: Path,
    split: str,
    classifier: Path,
    sam: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(source_root / "project" / "generate_pseudo_masks.py"),
        "--pipeline-profile",
        "default",
        "--data-root",
        str(data_root),
        "--split",
        split,
        "--split-manifest",
        str(split_manifest),
        "--classifier-checkpoint",
        str(classifier),
        "--sam-checkpoint",
        str(sam),
        "--classifier-device",
        "cuda",
        "--sam-device",
        "cuda",
        "--target-columns",
        "tumor",
        "--sam-image-size",
        "512",
        "--batch-size",
        "1",
        "--num-workers",
        "2",
        "--output-dir",
        str(output_dir),
        "--process-all",
        "--save-visuals-limit",
        "0",
        "--confidence-threshold",
        "0.5",
        "--cam-tta-flip",
        "--cam-percentile",
        "90",
        "--cam-percentile-ensemble",
        "--cam-percentile-values",
        "85,90,95",
        "--max-points",
        "5",
        "--mask-score-threshold",
        "0.4",
        "--seed-percentile",
        "82",
        "--support-percentile",
        "55",
        "--morphology-fusion-mode",
        "components",
        "--sam-prompt-mode",
        "box_point",
        "--sam-prompt-ensemble",
        "--max-components",
        "3",
        "--all-cam-components",
        "--points-per-component",
        "5",
        "--bbox-padding-ratio",
        "0.02",
        "--negative-points-per-component",
        "4",
        "--max-box-area-ratio",
        "0.35",
        "--selection-method",
        "coverage_mass_sam",
        "--fusion-topk",
        "1",
        "--component-topk",
        "3",
        "--closing-kernel",
        "0",
        "--opening-kernel",
        "0",
        "--max-hole-area",
        "0",
        "--guidance-threshold",
        "0.4",
        "--preprocessing-mode",
        "none",
        "--low-score-policy",
        "empty",
        "--cam-target-class",
        "ground_truth",
        "--save-candidate-diagnostics",
        "--candidate-diagnostics-cohort",
        "all",
        "--force-normal-candidate-gallery",
    ]


def build_generation_command(
    *,
    mode: str,
    source_root: Path,
    data_root: Path,
    split_manifest: Path,
    split: str,
    classifier: Path,
    sam: Path,
    output_dir: Path,
    external_root: Path | None = None,
    external_manifest_sha256: str | None = None,
    external_metadata_sha256: str | None = None,
    external_source_commit: str | None = None,
    external_weight_sha256: str | None = None,
) -> list[str]:
    command = common_generation_args(
        source_root=source_root,
        data_root=data_root,
        split_manifest=split_manifest,
        split=split,
        classifier=classifier,
        sam=sam,
        output_dir=output_dir,
    )
    if mode == "anchor":
        if not all(
            (
                external_root,
                external_manifest_sha256,
                external_metadata_sha256,
                external_source_commit,
                external_weight_sha256,
            )
        ):
            raise ValueError("Anchor mode requires the complete external-saliency lock")
        command.extend(
            [
                "--image-size",
                "320",
                "--min-component-area",
                "100",
                "--prompt-border-margin",
                "2",
                "--support-clip-kernel",
                "5",
                "--min-size",
                "40",
                "--external-saliency-manifest",
                str(external_root / "saliency_manifest.csv"),
                "--external-saliency-run-metadata",
                str(external_root / "run_metadata.json"),
                "--external-saliency-expected-manifest-sha256",
                str(external_manifest_sha256),
                "--external-saliency-expected-metadata-sha256",
                str(external_metadata_sha256),
                "--external-saliency-expected-source-commit",
                str(external_source_commit),
                "--external-saliency-expected-model-weight-sha256",
                str(external_weight_sha256),
                "--external-saliency-role",
                "proposal_gallery",
            ]
        )
    elif mode == "addition":
        if any(
            value is not None
            for value in (
                external_root,
                external_manifest_sha256,
                external_metadata_sha256,
                external_source_commit,
                external_weight_sha256,
            )
        ):
            raise ValueError("Addition mode cannot consume external saliency")
        command.extend(
            [
                "--image-size",
                "448",
                "--min-component-area",
                "196",
                "--prompt-border-margin",
                "3",
                "--support-clip-kernel",
                "7",
                "--min-size",
                "78",
            ]
        )
    else:
        raise ValueError(f"Unsupported rich-gallery supply mode: {mode}")
    return command


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("anchor", "addition"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sam-sha256", required=True)
    parser.add_argument("--external-saliency-supply-root", type=Path)
    parser.add_argument("--expected-external-supply-manifest-sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("Canonical split SHA-256 mismatch")
    if sha256_file(args.classifier_checkpoint) != args.expected_classifier_sha256:
        raise ValueError("Classifier checkpoint SHA-256 mismatch")
    if sha256_file(args.sam_checkpoint) != args.expected_sam_sha256:
        raise ValueError("SAM checkpoint SHA-256 mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Candidate supply output must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: split_counts(args.split_manifest, split) for split in ("train", "val")}

    supply = None
    if args.mode == "anchor":
        if args.external_saliency_supply_root is None:
            raise ValueError("Anchor mode requires --external-saliency-supply-root")
        supply_manifest = (
            args.external_saliency_supply_root / "saliency_supply_manifest.json"
        )
        if (
            not supply_manifest.is_file()
            or sha256_file(supply_manifest)
            != args.expected_external_supply_manifest_sha256
        ):
            raise ValueError("External saliency supply manifest hash mismatch")
        supply = json.loads(supply_manifest.read_text(encoding="utf-8"))
        if (
            supply.get("spatial_ground_truth_read") is not False
            or supply.get("test_images_read") != 0
            or supply.get("test_evaluated") is not False
        ):
            raise ValueError("External saliency supply violates no-GT/no-test")
    elif args.external_saliency_supply_root is not None:
        raise ValueError("Addition mode cannot receive external saliency")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(args.source_root / "project"),
            "PYTHONUNBUFFERED": "1",
            "BTXRD_DISABLE_TQDM": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    log = args.output_dir / "kernel.log"
    results: dict[str, dict[str, object]] = {}
    for split in ("train", "val"):
        external = (
            args.external_saliency_supply_root / split
            if args.external_saliency_supply_root is not None
            else None
        )
        external_lock = supply.get("splits", {}).get(split, {}) if supply else {}
        command = build_generation_command(
            mode=args.mode,
            source_root=args.source_root,
            data_root=args.data_root,
            split_manifest=args.split_manifest,
            split=split,
            classifier=args.classifier_checkpoint,
            sam=args.sam_checkpoint,
            output_dir=args.output_dir / split,
            external_root=external,
            external_manifest_sha256=external_lock.get("manifest_sha256"),
            external_metadata_sha256=external_lock.get("metadata_sha256"),
            external_source_commit=(supply.get("source_commit") if supply else None),
            external_weight_sha256=(
                supply.get("biomedclip_weight_sha256") if supply else None
            ),
        )
        run(command, cwd=args.source_root, env=env, log=log)
        stage = args.output_dir / split
        candidate_summary = json.loads(
            (stage / "candidate_diagnostics_summary.json").read_text(encoding="utf-8")
        )
        pseudo_summary = json.loads(
            (stage / "pseudo_mask_summary.json").read_text(encoding="utf-8")
        )
        metadata = json.loads((stage / "run_metadata.json").read_text(encoding="utf-8"))
        if (
            candidate_summary.get("complete") is not True
            or candidate_summary.get("prediction_first") is not True
            or candidate_summary.get("ground_truth_loaded_during_generation") is not False
            or candidate_summary.get("cohort") != "all"
            or int(candidate_summary.get("manifest_rows", -1))
            != EXPECTED_COUNTS[split]["images"]
            or int(pseudo_summary.get("manifest_rows", -1))
            != EXPECTED_COUNTS[split]["images"]
            or metadata.get("split") != split
            or metadata.get("force_normal_candidate_gallery") is not True
            or metadata.get("candidate_diagnostics_cohort") != "all"
        ):
            raise ValueError(f"Frozen {split} candidate supply audit failed")
        results[split] = {
            "counts": counts[split],
            "image_size": int(candidate_summary["image_size"]),
            "candidate_manifest_sha256": candidate_summary["manifest_sha256"],
            "candidate_summary_sha256": sha256_file(
                stage / "candidate_diagnostics_summary.json"
            ),
            "pseudo_manifest_sha256": pseudo_summary["manifest_sha256"],
            "pseudo_summary_sha256": sha256_file(stage / "pseudo_mask_summary.json"),
            "run_metadata_sha256": sha256_file(stage / "run_metadata.json"),
        }
    manifest = {
        "schema_version": 1,
        "stage": "rich_gallery_candidate_supply",
        "mode": args.mode,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": args.expected_split_sha256,
        "classifier_checkpoint_sha256": args.expected_classifier_sha256,
        "sam_checkpoint_sha256": args.expected_sam_sha256,
        "external_saliency_supply_manifest_sha256": (
            args.expected_external_supply_manifest_sha256
            if args.mode == "anchor"
            else None
        ),
        "splits": results,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    path = args.output_dir / "candidate_supply_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
