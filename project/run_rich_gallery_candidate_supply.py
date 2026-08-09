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
    "test": {"images": 373, "tumor": 187, "normal": 186},
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
    sam_model_type: str = "vit_b",
    sam_backend: str = "sam_v1",
    sam_source_root: Path | None = None,
    output_dir: Path,
    attribution_method: str = "layercam",
    prompt_mode: str = "box_point",
    prompt_ensemble: bool = True,
    target_columns: str = "tumor",
    cam_aggregation: str = "class",
    sam_single_mask: bool = False,
    classifier_split_manifest: Path | None = None,
    frozen_config: Path | None = None,
) -> list[str]:
    if target_columns == "tumor" and cam_aggregation != "class":
        raise ValueError("binary tumor supply requires class CAM aggregation")
    if target_columns == "tumor_type" and cam_aggregation != "tumor_log_odds":
        raise ValueError(
            "ten-class G4 supply requires exact collapsed tumor_log_odds"
        )
    command = [
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
        "--sam-backend",
        sam_backend,
        "--sam-model-type",
        sam_model_type,
        "--classifier-device",
        "cuda",
        "--sam-device",
        "cuda",
        "--target-columns",
        target_columns,
        "--attribution-method",
        attribution_method,
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
        prompt_mode,
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
        "--cam-aggregation",
        cam_aggregation,
        "--save-candidate-diagnostics",
        "--candidate-diagnostics-cohort",
        "all",
    ]
    if sam_source_root is not None:
        command.extend(["--sam-source-root", str(sam_source_root)])
    # Both matched E1 arms require proposal bags for normal images so that the
    # downstream negative-bag objective sees the same cohort.  For ten-class
    # checkpoints, ``tumor_log_odds`` is the exact collapsed tumor/normal event.
    command.append("--force-normal-candidate-gallery")
    if prompt_ensemble:
        command.append("--sam-prompt-ensemble")
    else:
        if split != "val":
            raise ValueError("single-mode prompt ablation is validation-only")
        command.extend([
            "--disable-sam-prompt-ensemble",
            "--allow-validation-prompt-ablation",
        ])
    if sam_single_mask:
        if split != "val":
            raise ValueError("single-mask SAM ablation is validation-only")
        command.extend(
            ["--sam-single-mask", "--allow-validation-sam-single-mask-ablation"]
        )
    if classifier_split_manifest is not None:
        command.extend(
            ["--classifier-split-manifest", str(classifier_split_manifest)]
        )
    if split == "test":
        if frozen_config is None:
            raise ValueError("test candidate generation requires --frozen-config")
        command.extend(["--frozen-config", str(frozen_config)])
    return command


def build_generation_command(
    *,
    mode: str,
    source_root: Path,
    data_root: Path,
    split_manifest: Path,
    split: str,
    classifier: Path,
    sam: Path,
    sam_model_type: str = "vit_b",
    sam_backend: str = "sam_v1",
    sam_source_root: Path | None = None,
    output_dir: Path,
    classifier_split_manifest: Path | None = None,
    external_root: Path | None = None,
    external_manifest_sha256: str | None = None,
    external_metadata_sha256: str | None = None,
    external_source_commit: str | None = None,
    external_weight_sha256: str | None = None,
    frozen_config: Path | None = None,
    attribution_method: str = "layercam",
    prompt_mode: str = "box_point",
    prompt_ensemble: bool = True,
    target_columns: str = "tumor",
    cam_aggregation: str = "class",
    sam_single_mask: bool = False,
) -> list[str]:
    command = common_generation_args(
        source_root=source_root,
        data_root=data_root,
        split_manifest=split_manifest,
        classifier_split_manifest=classifier_split_manifest,
        split=split,
        classifier=classifier,
        sam=sam,
        sam_model_type=sam_model_type,
        sam_backend=sam_backend,
        sam_source_root=sam_source_root,
        output_dir=output_dir,
        frozen_config=frozen_config,
        attribution_method=attribution_method,
        prompt_mode=prompt_mode,
        prompt_ensemble=prompt_ensemble,
        target_columns=target_columns,
        cam_aggregation=cam_aggregation,
        sam_single_mask=sam_single_mask,
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
    parser.add_argument("--classifier-split-manifest", type=Path)
    parser.add_argument("--expected-classifier-split-sha256")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam-backend",
        choices=("sam_v1", "sam2", "sam_med2d", "medsam"),
        default="sam_v1",
    )
    parser.add_argument("--sam-source-root", type=Path)
    parser.add_argument(
        "--sam-model-type",
        default="vit_b",
    )
    parser.add_argument("--expected-sam-sha256", required=True)
    parser.add_argument("--external-saliency-supply-root", type=Path)
    parser.add_argument("--expected-external-supply-manifest-sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated subset of train,val,test. Test requires --frozen-config.",
    )
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--attribution-method",
        choices=("layercam", "cam", "gradcam", "gradcam_plus_plus"),
        default="layercam",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("point", "box", "box_point"),
        default="box_point",
    )
    parser.add_argument(
        "--prompt-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-columns",
        choices=("tumor", "tumor_type"),
        default="tumor",
    )
    parser.add_argument(
        "--cam-aggregation",
        choices=("class", "tumor_log_odds"),
        default="class",
    )
    parser.add_argument(
        "--sam-single-mask",
        action="store_true",
        help="Run the predeclared validation-only G4 E5 single-mask SAM arm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("Canonical split SHA-256 mismatch")
    if args.classifier_split_manifest is not None:
        if args.expected_classifier_split_sha256 is None:
            raise ValueError("Classifier split manifest requires its expected SHA-256")
        if (
            sha256_file(args.classifier_split_manifest)
            != args.expected_classifier_split_sha256
        ):
            raise ValueError("Classifier split manifest SHA-256 mismatch")
    elif args.expected_classifier_split_sha256 is not None:
        raise ValueError("Unexpected classifier split SHA-256 without a manifest")
    if sha256_file(args.classifier_checkpoint) != args.expected_classifier_sha256:
        raise ValueError("Classifier checkpoint SHA-256 mismatch")
    if sha256_file(args.sam_checkpoint) != args.expected_sam_sha256:
        raise ValueError("SAM checkpoint SHA-256 mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Candidate supply output must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits or len(splits) != len(set(splits)) or any(item not in EXPECTED_COUNTS for item in splits):
        raise ValueError("--splits must be a unique subset of train,val,test")
    if "test" in splits and args.frozen_config is None:
        raise ValueError("test candidate generation requires --frozen-config")
    counts = {split: split_counts(args.split_manifest, split) for split in splits}

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
        expected_test_reads = counts["test"]["images"] if "test" in splits else 0
        if (
            supply.get("spatial_ground_truth_read") is not False
            or int(supply.get("test_images_read", -1)) != expected_test_reads
            or supply.get("test_evaluated") is not False
        ):
            raise ValueError(
                "External saliency supply violates the no-spatial-GT or "
                "declared test-image-read contract"
            )
    elif args.external_saliency_supply_root is not None:
        raise ValueError("Addition mode cannot receive external saliency")

    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH", "")
    python_paths = [str(args.source_root / "project")]
    if inherited_pythonpath:
        python_paths.append(inherited_pythonpath)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONUNBUFFERED": "1",
            "BTXRD_DISABLE_TQDM": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    log = args.output_dir / "kernel.log"
    results: dict[str, dict[str, object]] = {}
    for split in splits:
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
            classifier_split_manifest=args.classifier_split_manifest,
            split=split,
            classifier=args.classifier_checkpoint,
            sam=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            sam_backend=args.sam_backend,
            sam_source_root=args.sam_source_root,
            output_dir=args.output_dir / split,
            external_root=external,
            external_manifest_sha256=external_lock.get("manifest_sha256"),
            external_metadata_sha256=external_lock.get("metadata_sha256"),
            external_source_commit=(supply.get("source_commit") if supply else None),
            external_weight_sha256=(
                supply.get("biomedclip_weight_sha256") if supply else None
            ),
            frozen_config=args.frozen_config,
            attribution_method=args.attribution_method,
            prompt_mode=args.prompt_mode,
            prompt_ensemble=args.prompt_ensemble,
            target_columns=args.target_columns,
            cam_aggregation=args.cam_aggregation,
            sam_single_mask=args.sam_single_mask,
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
            or metadata.get("sam_model_type") != args.sam_model_type
            or metadata.get("sam_backend") != args.sam_backend
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
            "resource_metrics_sha256": sha256_file(stage / "resource_metrics.json"),
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
        "sam_backend": args.sam_backend,
        "sam_model_type": args.sam_model_type,
        "sam_single_mask": args.sam_single_mask,
        "attribution_method": args.attribution_method,
        "prompt_mode": args.prompt_mode,
        "prompt_ensemble": args.prompt_ensemble,
        "external_saliency_supply_manifest_sha256": (
            args.expected_external_supply_manifest_sha256
            if args.mode == "anchor"
            else None
        ),
        "splits": results,
        "spatial_ground_truth_read": False,
        "test_images_read": EXPECTED_COUNTS["test"]["images"] if "test" in splits else 0,
        "test_evaluated": False,
    }
    path = args.output_dir / "candidate_supply_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
