from __future__ import annotations

"""Bounded Kaggle runner for one DSLL candidate-supply shard."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from run_rich_gallery_candidate_supply import build_generation_command


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
CLASSIFIER_SHA = "e043f43655798f3389005be89cb85605ef09a87977a74733ea07bf9abc0ed7ec"
SAM_SHA = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
EXTERNAL_WEIGHT_SHA = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
EXTERNAL_SOURCE_COMMIT = "e8683c3bf42cde99c781cb3bb528a6ab1333b327"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_hash(root: Path, digest: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and sha256(path) == digest]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one input with SHA {digest}, found {matches}")
    return matches[0]


def unique_source(root: Path) -> Path:
    matches = []
    for path in root.rglob("generate_pseudo_masks.py"):
        if (path.parent / "dsll_top3.py").is_file() and (path.parent / "run_rich_gallery_candidate_supply.py").is_file():
            matches.append(path.parent.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DSLL source tree, found {matches}")
    return matches[0]


def unique_data(root: Path) -> Path:
    matches = [path for path in root.rglob("BTXRD") if path.is_dir() and (path / "images").is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one BTXRD root, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--image-list-name", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/dsll_supply"))
    args = parser.parse_args()
    source = unique_source(args.input_root)
    split_manifest = unique_hash(args.input_root, SPLIT_SHA)
    classifier = unique_hash(args.input_root, CLASSIFIER_SHA)
    sam = unique_hash(args.input_root, SAM_SHA)
    data = unique_data(args.input_root)
    image_lists = list(args.input_root.rglob(args.image_list_name))
    if len(image_lists) != 1:
        raise RuntimeError(f"Expected one image list {args.image_list_name}, found {image_lists}")
    supply_manifests = list(args.input_root.rglob("saliency_supply_manifest.json"))
    valid_supplies = []
    for path in supply_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("split_sha256") == SPLIT_SHA
            and payload.get("biomedclip_weight_sha256") == EXTERNAL_WEIGHT_SHA
            and payload.get("source_commit") == EXTERNAL_SOURCE_COMMIT
            and payload.get("spatial_ground_truth_read") is False
            and int(payload.get("test_images_read", -1)) == 0
        ):
            valid_supplies.append((path.parent, payload))
    if len(valid_supplies) != 1:
        raise RuntimeError(f"Expected one exact external supply, found {len(valid_supplies)}")
    supply_root, supply = valid_supplies[0]
    split_supply = supply_root / args.split
    split_lock = supply["splits"][args.split]
    command = build_generation_command(
        mode="anchor",
        source_root=source,
        data_root=data,
        split_manifest=split_manifest,
        classifier_split_manifest=split_manifest,
        split=args.split,
        classifier=classifier,
        sam=sam,
        output_dir=args.output_dir,
        external_root=split_supply,
        external_manifest_sha256=str(split_lock["manifest_sha256"]),
        external_metadata_sha256=str(split_lock["metadata_sha256"]),
        external_source_commit=EXTERNAL_SOURCE_COMMIT,
        external_weight_sha256=EXTERNAL_WEIGHT_SHA,
        target_columns="tumor_type",
        cam_aggregation="dsll_top3_gallery",
    )
    command.extend(["--image-list", str(image_lists[0])])
    if args.split == "val":
        command.append("--save-dsll-cam-maps")
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": "/kaggle/working/pycache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    subprocess.run(command, cwd=source, env=env, check=True)
    receipt = {
        "stage": "dsll_top3_candidate_shard_v1",
        "split": args.split,
        "image_list": args.image_list_name,
        "source_commit": args.source_commit,
        "protocol_sha256": args.protocol_sha256,
        "split_sha256": SPLIT_SHA,
        "classifier_sha256": CLASSIFIER_SHA,
        "sam_sha256": SAM_SHA,
        "spatial_ground_truth_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    (args.output_dir / "dsll_shard_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
