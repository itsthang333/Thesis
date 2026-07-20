from __future__ import annotations

"""Create/verify an immutable, checksum-bearing BTXRD pipeline configuration."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BTXRD_BEST_PIPELINE, BTXRD_HYBRID_PIPELINE
from evaluation.frozen_test_guard import verify_frozen_test_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("btxrd_best", "btxrd_hybrid"), default="btxrd_best")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--supervised-unet-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=("candidate", "final"), default="candidate")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        document = verify_frozen_test_config(args.output, split="test")
        print(json.dumps({"verified": True, "freeze_sha256": document["freeze_sha256"]}, indent=2))
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen config: {args.output}")
    profile = BTXRD_HYBRID_PIPELINE if args.profile == "btxrd_hybrid" else BTXRD_BEST_PIPELINE
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    if args.status == "final":
        if dirty:
            raise ValueError("A final configuration cannot be frozen from a dirty working tree")
        for label, path in (
            ("split manifest", args.split_manifest),
            ("classifier checkpoint", args.classifier_checkpoint),
            ("SAM checkpoint", args.sam_checkpoint),
            ("U-Net checkpoint", args.unet_checkpoint),
            ("supervised U-Net checkpoint", args.supervised_unet_checkpoint),
        ):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Final freeze requires a local {label}: {path}")
    document: dict[str, object] = {
        "schema_version": 2,
        "status": args.status,
        "profile": asdict(profile),
        "source": {"branch": "pipeline", "git_commit": commit, "git_dirty": dirty},
        "split_manifest": (
            {"path": str(args.split_manifest.resolve()), "sha256": sha256_file(args.split_manifest)}
            if args.split_manifest and args.split_manifest.is_file() else None
        ),
        "classifier_checkpoint": (
            {"path": str(args.classifier_checkpoint.resolve()), "sha256": sha256_file(args.classifier_checkpoint)}
            if args.classifier_checkpoint and args.classifier_checkpoint.is_file() else None
        ),
        "sam_checkpoint": (
            {"path": str(args.sam_checkpoint.resolve()), "sha256": sha256_file(args.sam_checkpoint)}
            if args.sam_checkpoint and args.sam_checkpoint.is_file() else None
        ),
        "unet_checkpoint": (
            {"path": str(args.unet_checkpoint.resolve()), "sha256": sha256_file(args.unet_checkpoint)}
            if args.unet_checkpoint and args.unet_checkpoint.is_file() else None
        ),
        "supervised_unet_checkpoint": (
            {"path": str(args.supervised_unet_checkpoint.resolve()), "sha256": sha256_file(args.supervised_unet_checkpoint)}
            if args.supervised_unet_checkpoint and args.supervised_unet_checkpoint.is_file() else None
        ),
        "protocols": {
            "canonical_train_pseudo_masks": "known image-level label (ground_truth class target)",
            "diagnostic_end_to_end": "predicted classifier class",
            "locked_test_usage": "one final run after configuration status=final",
            "final_inference": "U-Net only",
        },
    }
    document["freeze_sha256"] = hashlib.sha256(canonical_bytes(document)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "freeze_sha256": document["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
