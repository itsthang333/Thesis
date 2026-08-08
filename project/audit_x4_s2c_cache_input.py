from __future__ import annotations

"""Adopt and fully verify a class-agnostic SAM cache for X4 S2C.

The expensive SAM cache may predate X4, but it is reusable only when every
canonical image id, image hash, binary image label, cached payload hash and
no-GT contract matches the current X4 split.  The emitted audit binds that
legacy image-derived artifact to the current immutable X4 protocol without
claiming it was regenerated.
"""

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pseudo.manifest import sha256_file
from pseudo.sam_segment_cache import validate_sam_segment_cache
from x4_contract import CANONICAL_SPLIT_SHA256, load_x4_protocol


EXPECTED = {"train": (2981, 1493, 1488), "val": (371, 187, 184)}


def _samples(path: Path, split: str) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "image_id": str(row["image_id"]),
            "image_sha256": str(row["image_sha256"]),
            "tumor": int(row["tumor"]),
        }
        for row in rows
        if row.get("split") == split
        and str(row.get("eligible", "1")).strip().lower() not in {"0", "false", "no"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    protocol, protocol_sha = load_x4_protocol(REPO_ROOT)
    metadata_path = args.cache_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("ground_truth_spatial_supervision") is not False:
        raise ValueError("S2C cache does not certify no spatial GT")
    if metadata.get("split") != args.split:
        raise ValueError("S2C cache split differs")
    metadata_text = json.dumps(metadata, sort_keys=True).lower()
    if any(token in metadata_text for token in ("test_evaluated\": true", "spatial_ground_truth\": true")):
        raise ValueError("S2C cache metadata contains forbidden GT/test state")

    samples = _samples(args.split_manifest, args.split)
    count, normals, tumors = EXPECTED[args.split]
    observed = (
        len(samples),
        sum(int(not bool(row["tumor"])) for row in samples),
        sum(int(bool(row["tumor"])) for row in samples),
    )
    if observed != (count, normals, tumors):
        raise ValueError(f"X4 canonical cohort differs: {observed}")
    validated = validate_sam_segment_cache(args.cache_dir, samples, split=args.split)
    result = {
        "status": "pass",
        "study": "X4 W2 reusable class-agnostic SAM cache adoption",
        "split": args.split,
        "images": count,
        "normal_images": normals,
        "tumor_images": tumors,
        "cache_manifest_sha256": sha256_file(args.cache_dir / "sam_segment_manifest.csv"),
        "cache_summary_sha256": validated["summary_sha256"],
        "cache_run_metadata_sha256": sha256_file(metadata_path),
        "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
        "x4_protocol_sha256": protocol_sha,
        "source_cache_split_manifest_sha256": metadata.get("split_manifest_sha256"),
        "source_cache_reused": True,
        "top_level_cache_files_and_all_payload_hashes_verified": True,
        "ground_truth_spatial_supervision": False,
        "outer_validation_selection": False,
        "student_trained": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
