from __future__ import annotations

"""Build one hash-bound dataset/kernel payload for X4 YOLO Stage-A/B."""

import argparse
import json
import shutil
from pathlib import Path

from frozen_io import sha256_file


SOURCE_FILES = (
    "project/run_x4_yolo_evaluation_kaggle.py",
    "project/run_x4_yolo_kaggle.py",
    "project/freeze_x4_yolo_predictions.py",
    "project/evaluate_x4_yolo_predictions.py",
    "project/train_x4_yolov8s_seg.py",
    "project/frozen_io.py",
    "project/x4_contract.py",
    "project/config.py",
    "project/datasets/__init__.py",
    "project/datasets/btxrd.py",
    "project/datasets/common.py",
    "project/evaluation/__init__.py",
    "project/evaluation/segmentation_metrics.py",
    "artifacts/final_pipeline/x4/x4_protocol.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--yolo-input-source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-slug", required=True)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    bundle_sha = sha256_file(args.training_bundle)
    if (
        receipt.get("stage") != "x4_yolov8s_seg_kaggle_wrapper_v1"
        or receipt.get("seed") != args.seed
        or receipt.get("split_sha256") != args.split_sha256
        or receipt.get("training_archive_sha256") != bundle_sha
        or receipt.get("test_images_read") != 0
        or receipt.get("test_evaluated") is not False
    ):
        raise RuntimeError("Training receipt/bundle differs from evaluation payload")

    dataset_dir = args.output_dir / "dataset"
    source_root = dataset_dir / "x4_yolo_evaluation_source"
    kernel_dir = args.output_dir / "kernel"
    source_root.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    for relative in SOURCE_FILES:
        source = args.repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )

    hashes = {row["path"]: row["sha256"] for row in manifest}
    contract = {
        "schema_version": 1,
        "stage": "x4_yolo_kaggle_evaluation_contract_v1",
        "source_commit": args.source_commit,
        "seed": args.seed,
        "runtime_manifest_sha256": args.runtime_manifest_sha256,
        "split_sha256": args.split_sha256,
        "training_slug": args.training_slug,
        "training_bundle_name": args.training_bundle.name,
        "training_bundle_sha256": bundle_sha,
        "training_receipt_name": args.training_receipt.name,
        "training_receipt_sha256": sha256_file(args.training_receipt),
        "runner_sha256": hashes["project/run_x4_yolo_evaluation_kaggle.py"],
        "freeze_runner_sha256": hashes["project/freeze_x4_yolo_predictions.py"],
        "evaluator_sha256": hashes["project/evaluate_x4_yolo_predictions.py"],
        "output_prefix": f"x4_yolov8s_seg_seed{args.seed}",
        "evaluation_batch": 1,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    contract_path = dataset_dir / "evaluation_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = dataset_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset_slug = f"btxrd-x4-yolo-eval-source-s{args.seed}-20260809"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": f"{args.owner}/{dataset_slug}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper_source = args.repo_root / "project/run_x4_yolo_evaluation_kernel.py"
    wrapper_name = f"btxrd-x4-yolo-eval-seed{args.seed}.py"
    shutil.copy2(wrapper_source, kernel_dir / wrapper_name)
    kernel_slug = f"btxrd-x4-yolo-eval-seed{args.seed}"
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{args.owner}/{kernel_slug}",
                "title": kernel_slug,
                "code_file": wrapper_name,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": False,
                "dataset_sources": [
                    args.data_source,
                    args.yolo_input_source,
                    f"{args.owner}/{dataset_slug}",
                ],
                "competition_sources": [],
                "kernel_sources": [args.training_slug],
                "model_sources": [],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "stage": "x4_yolo_evaluation_payload_v1",
        "source_commit": args.source_commit,
        "owner": args.owner,
        "seed": args.seed,
        "dataset_slug": f"{args.owner}/{dataset_slug}",
        "kernel_slug": f"{args.owner}/{kernel_slug}",
        "contract_sha256": sha256_file(contract_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "wrapper_sha256": sha256_file(kernel_dir / wrapper_name),
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "payload_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
