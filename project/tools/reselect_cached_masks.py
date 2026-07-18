from __future__ import annotations

"""Re-run Stage 5/6 from cached CAM/SAM candidates without any GT access."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.factory import build_classification_dataset
from pseudo.mask_selection import SELECTION_METHODS, select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--btxrd-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument(
        "--methods",
        default="consensus_vote",
        help="Comma-separated selection methods to materialize from one shared cache.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--skip-missing-candidates", action="store_true",
        help="Pilot subsets only: omit tumor images whose candidate NPZ is absent.",
    )
    parser.add_argument(
        "--normal-limit", type=int, default=0,
        help="Maximum normal images to materialize; 0 keeps every normal image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(dict.fromkeys(method.strip() for method in args.methods.split(",") if method.strip()))
    unknown = sorted(set(methods) - set(SELECTION_METHODS))
    if unknown:
        raise ValueError(f"Unknown selection methods {unknown}; choose from {SELECTION_METHODS}")
    dataset = build_classification_dataset(
        "btxrd",
        root=args.btxrd_root,
        split=args.split,
        target_columns=["tumor_type"],
        image_size=args.image_size,
    )
    output_dirs = {method: args.output_root / method / "masks" for method in methods}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    empty = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
    tumor_count = 0
    normal_count = 0
    skipped_missing_tumor = 0
    for sample in dataset.samples:
        image_name = str(sample["image_id"])
        output_name = f"{Path(image_name).stem}.png"
        if int(sample["tumor_type"]) == 0:
            if args.normal_limit > 0 and normal_count >= args.normal_limit:
                continue
            normal_count += 1
            for directory in output_dirs.values():
                save_mask(empty, directory / output_name)
            continue
        cache_path = args.candidate_cache_dir / f"{Path(image_name).stem}.npz"
        if not cache_path.exists():
            if args.skip_missing_candidates:
                skipped_missing_tumor += 1
                continue
            raise FileNotFoundError(f"Missing tumor candidate cache: {cache_path}")
        tumor_count += 1
        with np.load(cache_path) as data:
            for method in methods:
                selected = select_and_fuse_masks(
                    data["masks"],
                    data["fused_cam"],
                    mask_score_threshold=0.4,
                    selection_method=method,
                    fusion_topk=3,
                    bone_likelihood=data["bone_likelihood"],
                    bone_support=data["bone_support"],
                    sam_scores=data["sam_scores"],
                    component_ids=data["component_ids"],
                    component_masks=data["component_masks"],
                    prompt_modes=data["prompt_modes"],
                    best_per_component=True,
                    component_topk=1,
                    support_clip_kernel=5,
                )
                final = morphological_refinement(
                    selected,
                    closing_kernel=0,
                    opening_kernel=0,
                    min_size=40,
                    guidance_map=data["bone_likelihood"],
                    guidance_threshold=0.4,
                    max_hole_area=0,
                )
                save_mask(final, output_dirs[method] / output_name)
    manifest = {
        "dataset": "btxrd",
        "split": args.split,
        "image_size": args.image_size,
        "methods": list(methods),
        "tumor_images": tumor_count,
        "normal_images": normal_count,
        "skipped_missing_tumor": skipped_missing_tumor,
        "ground_truth_used": False,
        "candidate_cache_dir": str(args.candidate_cache_dir.resolve()),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "reselection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
