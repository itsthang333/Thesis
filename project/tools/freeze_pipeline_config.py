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

from config import BTXRD_BEST_PIPELINE
from evaluation.frozen_test_guard import sha256_source_file, verify_frozen_test_config


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
    parser.add_argument("--profile", choices=("btxrd_best",), default="btxrd_best")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--threshold-selection", type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=("candidate", "final"), default="candidate")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        document = verify_frozen_test_config(
            args.output,
            split="test",
            validate_document_only=True,
        )
        print(json.dumps({"verified": True, "freeze_sha256": document["freeze_sha256"]}, indent=2))
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen config: {args.output}")
    profile = BTXRD_BEST_PIPELINE
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    if args.status == "final":
        if dirty:
            raise ValueError("A final configuration cannot be frozen from a dirty working tree")
        for label, path in (
            ("split manifest", args.split_manifest),
            ("U-Net checkpoint", args.unet_checkpoint),
            ("validation summary", args.validation_summary),
            ("validation threshold selection", args.threshold_selection),
        ):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Final freeze requires a local {label}: {path}")
        if args.threshold is None or not 0.0 <= args.threshold <= 1.0:
            raise ValueError("Final freeze requires an explicit --threshold in [0,1]")
        if args.image_size is None or args.image_size <= 0:
            raise ValueError("Final freeze requires a positive explicit --image-size")
        threshold_selection = json.loads(
            args.threshold_selection.read_text(encoding="utf-8")
        )
        selected = threshold_selection.get("selected") or {}
        if threshold_selection.get("selection_split") != "val":
            raise ValueError("Final threshold selection must come from validation")
        if float(selected.get("threshold", -1.0)) != float(args.threshold):
            raise ValueError("--threshold differs from validation threshold-selection evidence")
        validation_summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
        if validation_summary.get("split") != "val":
            raise ValueError("Final validation summary must describe split='val'")

    source_files = []
    for source_path in sorted(PROJECT_ROOT.rglob("*.py")):
        relative = source_path.relative_to(REPO_ROOT)
        source_files.append({
            "path": relative.as_posix(),
            "sha256": sha256_source_file(source_path),
            "bytes": source_path.stat().st_size,
        })

    def portable_artifact(path: Path | None) -> dict[str, object] | None:
        if path is None or not path.is_file():
            return None
        return {
            "path_hint": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    document: dict[str, object] = {
        "schema_version": 4,
        "status": args.status,
        "profile": asdict(profile),
        "source": {
            "branch": "main",
            "git_commit": commit,
            "git_dirty_at_freeze": dirty,
            "files": source_files,
        },
        "split_manifest": portable_artifact(args.split_manifest),
        "unet_checkpoint": portable_artifact(args.unet_checkpoint),
        "validation_evidence": {
            "summary": portable_artifact(args.validation_summary),
            "threshold_selection": portable_artifact(args.threshold_selection),
        },
        "evaluation": {
            "split": "test",
            "threshold": args.threshold,
            "image_size": args.image_size,
            "threshold_selection_partition": "val",
            "threshold_sweep_forbidden": True,
            "test_evaluated": False,
        },
        "allowed_test_stages": ["official_wsss_segmenter"],
        "protocols": {
            "canonical_train_pseudo_masks": "known image-level label (ground_truth class target)",
            "supervision": "WSSS: binary image-level labels -> LayerCAM/SAM pseudo masks -> U-Net",
            "locked_test_usage": "exactly one final segmenter evaluation after status=final",
            "final_inference": "U-Net only",
            "fully_supervised_checkpoint_allowed": False,
        },
    }
    document["freeze_sha256"] = hashlib.sha256(canonical_bytes(document)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "freeze_sha256": document["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
