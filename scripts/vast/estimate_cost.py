from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dollars-per-hour", type=float, required=True)
    parser.add_argument("--hours", type=float, nargs="+", default=[12, 24, 30])
    parser.add_argument("--storage-gb", type=float, default=300)
    parser.add_argument("--storage-per-gb-month", type=float, default=0.10)
    args = parser.parse_args()
    for hours in args.hours:
        compute = args.dollars_per_hour * hours
        storage = args.storage_gb * args.storage_per_gb_month * hours / (30 * 24)
        print(
            f"{hours:>6.1f}h  compute=${compute:>9,.2f}  "
            f"storage~${storage:>7,.2f}  total~${compute + storage:>9,.2f}"
        )


if __name__ == "__main__":
    main()
