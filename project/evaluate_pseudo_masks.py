from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET, SUPPORTED_DATASETS, SegmentationConfig
from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
from datasets.factory import build_segmentation_dataset
from progress import should_disable_tqdm
from evaluation.segmentation_metrics import (
    bootstrap_group_confidence_intervals,
    json_safe,
    segmentation_metrics,
    subgroup_summaries,
    summarize_segmentation_rows,
)
from evaluation.frozen_test_guard import verify_frozen_test_config
from pseudo.manifest import validate_pseudo_mask_manifest
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BTXRD pseudo masks against polygon ground truth")
    parser.set_defaults(dataset="btxrd")
    parser.add_argument("--data-root", type=Path, required=True, help="BTXRD dataset root")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--frozen-config", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Immutable derived split manifest. Its assignments are authoritative for BTXRD.",
    )
    parser.add_argument("--pred-mask-root", type=Path, default=ROOT / "outputs" / "pseudo_masks" / "masks")
    parser.add_argument("--image-size", type=int, default=SegmentationConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "eval.csv")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Optional text file of image names to evaluate (one per line).")
    parser.add_argument("--skipped-list", type=Path, default=None,
                        help="Path to skipped_low_confidence.txt from generate_pseudo_masks.py "
                        "(defaults to <pred-mask-root>/../skipped_low_confidence.txt if present). "
                        "Tumor images the classifier skipped for low confidence get an all-zero "
                        "pseudo-mask by construction; reporting their Dice alongside images that "
                        "actually ran CAM/SAM would blame the wrong pipeline stage for the failure.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Diagnostic-only opt-in to report missing predictions instead of failing final evaluation.",
    )
    return parser.parse_args()


def load_skipped_names(path: Path | None, pred_mask_root: Path) -> set[str]:
    candidate = path if path is not None else pred_mask_root.parent / "skipped_low_confidence.txt"
    if not candidate.exists():
        return set()
    return {Path(line.strip()).stem for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_run_metadata(pred_mask_root: Path) -> dict[str, object]:
    metadata_path = pred_mask_root.parent / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_run_protocol(metadata: dict[str, object]) -> str:
    return str(metadata.get("cam_target_class", "unknown"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pred_mask(mask_root: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = mask_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_pred_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    mask = Image.open(mask_path).convert("L")
    return (transform(mask) > 0.5).float()


def binary_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def main() -> None:
    args = parse_args()
    verify_frozen_test_config(args.frozen_config, split=args.split, split_manifest=args.split_manifest)
    dataset = build_segmentation_dataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        split_manifest=args.split_manifest,
    )
    if args.dataset == "btxrd":
        pseudo_manifest_info = validate_pseudo_mask_manifest(
            args.pred_mask_root,
            dataset.samples,
            split=args.split,
            image_size=args.image_size,
        )
        print(f"Validated pseudo-mask manifest: {pseudo_manifest_info['manifest_sha256']}")
    if args.image_list is not None:
        requested_names = {
            line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        original_count = len(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples if str(sample["image_id"]) in requested_names
        ]
        if not dataset.samples:
            raise ValueError(f"--image-list {args.image_list} matched no images in split '{args.split}'")
        print(f"Image-list filter: {len(dataset.samples)}/{original_count} samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_names = load_skipped_names(args.skipped_list, args.pred_mask_root)
    run_metadata = load_run_metadata(args.pred_mask_root)
    run_protocol = load_run_protocol(run_metadata)
    if args.split_manifest is not None:
        manifest_path = args.split_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
        expected_manifest_hash = sha256_file(manifest_path)
        actual_manifest_hash = run_metadata.get("split_manifest_sha256")
        if actual_manifest_hash != expected_manifest_hash:
            raise ValueError(
                "Pseudo-mask run split manifest hash does not match the requested evaluation manifest"
            )
    print(f"Pseudo-mask protocol: {run_protocol}")
    if skipped_names:
        print(f"Loaded {len(skipped_names)} low-confidence-skipped image names")

    rows: list[dict[str, object]] = []
    metadata_by_name = {str(sample["image_id"]): sample for sample in dataset.samples}
    # Split metrics by whether the GT mask actually has a lesion/bone region,
    # and further split tumor images by whether the classifier skipped them
    # for low confidence. A pooled mean over all three conflates three
    # different failure modes: normal images trivially score Dice=1 when both
    # prediction and GT are empty (a correct *detection*, not evidence of good
    # *segmentation*); classifier-skipped tumor images always get Dice≈0 by
    # construction (an all-zero mask was saved before CAM/SAM ever ran) and
    # blame the classifier threshold, not CAM/SAM/morphology; only tumor
    # images that actually ran the full pipeline reflect CAM/SAM quality.
    tumor_skipped_count = 0
    missing = 0

    for _, gt_masks, image_names in tqdm(
        loader, desc="evaluate", disable=should_disable_tqdm()
    ):
        pred_masks = []
        valid_names = []
        valid_gt = []
        for index, image_name in enumerate(image_names):
            pred_path = resolve_pred_mask(args.pred_mask_root, image_name)
            if pred_path is None:
                missing += 1
                rows.append({"image_name": str(image_name), "status": "missing"})
                continue
            pred_masks.append(load_pred_mask(pred_path, args.image_size))
            valid_gt.append(gt_masks[index])
            valid_names.append(image_name)

        if not pred_masks:
            continue

        for image_name, pred_mask, gt_mask in zip(valid_names, pred_masks, valid_gt):
            was_skipped = Path(image_name).stem in skipped_names
            metrics = segmentation_metrics(
                pred_mask[0].numpy() > 0.5,
                gt_mask[0].numpy() > 0.5,
            )
            metadata = metadata_by_name.get(str(image_name), {})
            tumor_type = int(metadata.get("tumor_type", 0) or 0)
            rows.append({
                "image_name": str(image_name),
                "status": "ok",
                "group": "tumor" if metrics["gt_positive"] else "normal",
                "skipped_low_confidence": was_skipped,
                "group_id": str(metadata.get("group_id", "")),
                "group_source": str(metadata.get("group_source", "")),
                "center": str(metadata.get("center", "")),
                "anatomy": str(metadata.get("anatomy", "")),
                "view": str(metadata.get("view", "")),
                "tumor_type": tumor_type,
                "tumor_type_name": (
                    TUMOR_TYPE_CLASS_NAMES[tumor_type]
                    if 0 <= tumor_type < len(TUMOR_TYPE_CLASS_NAMES) else "unknown"
                ),
                **metrics,
            })
            if metrics["gt_positive"]:
                # End-to-end metrics include detection/classification
                # failures. If predicted protocol maps a tumor image to the
                # normal class, generation deliberately writes an empty mask;
                # that Dice/IoU belongs in the end-to-end result even though
                # it is excluded from the conditional CAM/SAM-only metric.
                if was_skipped:
                    tumor_skipped_count += 1

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"{missing} predictions are missing under {args.pred_mask_root}. "
            "Final evaluation refuses an incomplete prediction set; use --allow-missing only for diagnostics."
        )

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    conditional_rows = [
        row for row in valid_rows
        if bool(row.get("gt_positive")) and not bool(row.get("skipped_low_confidence"))
    ]
    end_to_end = summarize_segmentation_rows(valid_rows)
    conditional = summarize_segmentation_rows(conditional_rows)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "cam_target_class_protocol": run_protocol,
        "pred_mask_root": str(args.pred_mask_root.resolve()),
        "boundary_distance_unit": "pixels on the resized evaluation grid",
        "missing": missing,
        "tumor_images_skipped_by_image_gate": tumor_skipped_count,
        # Keep the end-to-end aggregate fields at the top level as the stable
        # report API used by the notebook's component-top-k audit, while also
        # retaining the explicitly named nested block below.  Omitting these
        # aliases made the notebook fail only after both full validation
        # pseudo-mask runs had completed (KeyError on
        # gt_component_count_histogram).
        **end_to_end,
        "end_to_end": end_to_end,
        "cam_sam_conditional_tumor_only": conditional,
    }
    bootstrap = bootstrap_group_confidence_intervals(
        valid_rows,
        group_key="group_id",
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    subgroup_rows = subgroup_summaries(valid_rows)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(json_safe(rows))

    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    subgroup_path = args.output_csv.with_name(args.output_csv.stem + "_subgroups.csv")
    with subgroup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]) if subgroup_rows else ["subgroup_field"])
        writer.writeheader()
        writer.writerows(json_safe(subgroup_rows))
    bootstrap_path = args.output_csv.with_name(args.output_csv.stem + "_bootstrap.json")
    bootstrap_path.write_text(json.dumps(json_safe(bootstrap), indent=2) + "\n", encoding="utf-8")
    confusion_path = args.output_csv.with_name(args.output_csv.stem + "_pixel_confusion.json")
    confusion = {key: end_to_end[key] for key in ("tp_pixels", "fp_pixels", "fn_pixels", "tn_pixels")}
    confusion_path.write_text(json.dumps(confusion, indent=2) + "\n", encoding="utf-8")
    run_manifest_path = args.output_csv.with_name(args.output_csv.stem + "_run_manifest.json")
    run_manifest_path.write_text(
        json.dumps(
            {
                "entrypoint": str(Path(__file__).resolve()),
                "dataset": args.dataset,
                "split": args.split,
                "protocol": run_protocol,
                "run_metadata_sha256": (
                    sha256_file(args.pred_mask_root.parent / "run_metadata.json")
                    if (args.pred_mask_root.parent / "run_metadata.json").is_file() else None
                ),
                "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
                "split_manifest_sha256": sha256_file(args.split_manifest) if args.split_manifest else None,
                "artifacts": {
                    "per_image": str(args.output_csv.resolve()),
                    "summary": str(output_json.resolve()),
                    "subgroups": str(subgroup_path.resolve()),
                    "bootstrap": str(bootstrap_path.resolve()),
                    "pixel_confusion": str(confusion_path.resolve()),
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        f"{args.dataset} {args.split}: "
        f"tumor images={end_to_end['tumor_images']} (image-gate skips={tumor_skipped_count}) "
        f"conditional Dice={conditional['mean_tumor_dice']:.4f} "
        f"end-to-end Dice={end_to_end['mean_tumor_dice']:.4f} | "
        f"normal empty_prediction_rate={end_to_end['normal_empty_prediction_rate']:.4f} | missing={missing}"
    )
    print(f"Saved separate per-image, summary, subgroup, bootstrap, confusion and run-manifest artifacts")


if __name__ == "__main__":
    main()
