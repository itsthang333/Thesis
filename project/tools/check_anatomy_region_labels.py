from __future__ import annotations

"""Verify the anatomy-region columns (upper limb / lower limb / pelvis) are
clean enough to use for anatomy-matched contrastive learning, before any
training code depends on them.

Checks:
  1. Every image has EXACTLY one region set (not zero, not more than one).
  2. Both tumor and normal images are represented in every region (this is
     what makes the columns safe to use -- unlike the specific-bone columns
     tibia/femur/humerus/etc., which are only ever set for tumor images and
     would leak the tumor label if used directly).
  3. Reports the per-region tumor/normal counts so they can be checked
     against the numbers already sanity-checked by hand (upper limb
     672/452, lower limb 1095/1311, pelvis 112/104 normal/tumor).

Run directly on the machine that has the real dataset.csv (Kaggle/Colab),
independent of any other pipeline run:
    python3 tools/check_anatomy_region_labels.py --ram-root /path/to/BTXRD
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.btxrd import ANATOMY_REGION_COLUMNS, load_btxrd_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify BTXRD anatomy-region column quality")
    parser.add_argument("--ram-root", type=Path, required=True, help="BTXRD dataset root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_btxrd_records(args.ram_root)
    print(f"Total records: {len(records)}")

    unknown = [r for r in records if r["anatomy_region"] == -1]
    print(f"\nRecords with anatomy_region == -1 (zero or multiple region columns set): {len(unknown)}")
    if unknown:
        print("First 10 offending image_ids:", [r["image_id"] for r in unknown[:10]])
        print(
            "WARNING: anatomy-matched sampling cannot use these images as-is -- "
            "either exclude them or investigate why more/fewer than one region column is set."
        )

    print("\nPer-region tumor/normal breakdown:")
    print(f"{'region':<12} {'normal':>8} {'tumor':>8} {'total':>8}")
    for i, region_name in enumerate(ANATOMY_REGION_COLUMNS):
        region_records = [r for r in records if r["anatomy_region"] == i]
        normal_count = sum(1 for r in region_records if r["tumor"] == 0)
        tumor_count = sum(1 for r in region_records if r["tumor"] == 1)
        print(f"{region_name:<12} {normal_count:>8} {tumor_count:>8} {len(region_records):>8}")
        if normal_count == 0 or tumor_count == 0:
            print(
                f"  WARNING: region '{region_name}' has no {'normal' if normal_count == 0 else 'tumor'} "
                "images -- anatomy-matched pairing is impossible for this region."
            )

    accounted = len(records) - len(unknown)
    print(f"\nSanity check: {accounted}/{len(records)} records have exactly one region set "
          f"({accounted / len(records) * 100:.2f}%).")


if __name__ == "__main__":
    main()
