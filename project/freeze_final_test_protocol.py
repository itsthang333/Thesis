from __future__ import annotations

"""Create the immutable one-shot test protocol for the thesis pipeline."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def parse_named(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path or not name.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("named artifacts must use NAME=/path/to/file")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=parse_named, action="append", default=[])
    parser.add_argument("--test-run-id", required=True)
    parser.add_argument(
        "--source-commit",
        help=(
            "Explicit immutable source commit for an exported source snapshot that "
            "does not contain .git metadata (for example a private Kaggle dataset)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source_commit is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
        if dirty:
            raise ValueError("final test protocol can only be frozen from a clean committed tree")
    else:
        commit = args.source_commit.strip().lower()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("--source-commit must be a lowercase 40-character Git SHA")
    named: dict[str, object] = {}
    for name, path in args.artifact:
        if name in named:
            raise ValueError(f"duplicate named artifact: {name}")
        named[name] = artifact(path)
    document: dict[str, object] = {
        "schema_version": 4,
        "status": "final",
        "method": "G1 + fixed equal percentile-rank fusion",
        "test_run_id": args.test_run_id,
        "source": {"git_commit": commit, "git_dirty": False},
        "split_manifest": artifact(args.split_manifest),
        "classifier_checkpoints": [artifact(path) for path in args.classifier_checkpoint],
        "sam_checkpoint": artifact(args.sam_checkpoint),
        "g1_checkpoint": artifact(args.g1_checkpoint),
        "artifacts": named,
        "profile": {
            "candidate_sources": ["layercam320", "classifier448", "external_saliency"],
            "maximum_candidates": 243,
            "fusion": "0.5*percentile_rank(G1)+0.5*percentile_rank(upstream)",
            "tie_break": "raw G1 logit, then lower immutable candidate index",
            "image_label_protocol": "known binary image label; no spatial annotation",
            "test_policy": "one final execution after validation configuration freeze"
        },
    }
    document["freeze_sha256"] = hashlib.sha256(canonical_bytes(document)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "freeze_sha256": document["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
