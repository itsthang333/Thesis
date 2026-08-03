from __future__ import annotations

"""Bind one or more frozen BiomedCLIP split outputs into a supply manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits or len(splits) != len(set(splits)) or any(item not in {"train", "val", "test"} for item in splits):
        raise ValueError("invalid --splits")
    records: dict[str, object] = {}
    source_commit = None
    weight_sha = None
    for split in splits:
        root = args.root / split
        manifest = root / "saliency_manifest.csv"
        metadata_path = root / "run_metadata.json"
        if not manifest.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"BiomedCLIP output incomplete: {root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("split") != split
            or metadata.get("validation_gt_read") is not False
            or metadata.get("test_evaluated") is not False
            or metadata.get("manifest_sha256") != sha256_file(manifest)
        ):
            raise ValueError(f"BiomedCLIP output contract mismatch: {split}")
        current_commit = metadata["source_commit"]
        current_weight = metadata["model"]["weight_sha256"]
        source_commit = current_commit if source_commit is None else source_commit
        weight_sha = current_weight if weight_sha is None else weight_sha
        if current_commit != source_commit or current_weight != weight_sha:
            raise ValueError("BiomedCLIP source/weight differs between splits")
        records[split] = {
            "manifest_sha256": sha256_file(manifest),
            "metadata_sha256": sha256_file(metadata_path),
            "images": metadata["population"]["images"],
        }
    payload = {
        "schema_version": 1,
        "source_commit": source_commit,
        "biomedclip_weight_sha256": weight_sha,
        "splits": records,
        "spatial_ground_truth_read": False,
        "test_images_read": records.get("test", {}).get("images", 0),
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "manifest_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
