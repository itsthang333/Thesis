from __future__ import annotations

"""Independent fail-closed audit for the G4 E1 CAM-only completion."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from run_g4_e1_cam_only_completion import ARMS, E1_PROTOCOL_SHA, SEEDS, SPLIT_SHA


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(root: Path) -> dict[str, object]:
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    if (
        summary.get("study") != "G4 E1 CAM-only localization completion"
        or summary.get("e1_protocol_sha256") != E1_PROTOCOL_SHA
        or summary.get("split_sha256") != SPLIT_SHA
        or summary.get("threshold_selected_from_spatial_gt") is not False
        or int(summary.get("images_per_seed", -1)) != 371
        or int(summary.get("tumor_images_per_seed", -1)) != 184
        or int(summary.get("test_images_read", -1)) != 0
        or summary.get("test_evaluated") is not False
    ):
        raise ValueError("E1 CAM-only root contract differs")

    arms = summary.get("arms", {})
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("E1 CAM-only arm set differs")
    receipts: dict[str, object] = {}
    for arm in ARMS:
        seed_results = arms[arm].get("seed_results", [])
        if [int(item["seed"]) for item in seed_results] != list(SEEDS):
            raise ValueError(f"E1 CAM-only seed set differs for {arm}")
        arm_receipts: list[dict[str, object]] = []
        for item in seed_results:
            seed = int(item["seed"])
            evaluation = root / arm / f"seed_{seed}"
            report_path = evaluation / "summary.json"
            audit_path = evaluation / "audit.json"
            cam_path = evaluation / "cam_only_per_image.csv"
            per_image_path = evaluation / "per_image.csv"
            if not all(path.is_file() for path in (report_path, audit_path, cam_path, per_image_path)):
                raise FileNotFoundError(f"incomplete E1 CAM-only output for {arm}/{seed}")
            report = read_json(report_path)
            report_audit = read_json(audit_path)
            cam_rows = csv_rows(cam_path)
            selected_rows = csv_rows(per_image_path)
            if (
                report_audit.get("pass") is not True
                or report_audit.get("cam_only_per_image_sha256") != sha256(cam_path)
                or item.get("cam_only_per_image_sha256") != sha256(cam_path)
                or item.get("completion_summary_sha256") != sha256(report_path)
                or len(cam_rows) != 371
                or len(selected_rows) != 371
                or len({row["image_id"] for row in cam_rows}) != 371
                or sum(row["gt_positive"] == "True" for row in cam_rows) != 184
                or int(report.get("validation_annotations_opened", -1)) != 184
                or report.get("test_evaluated") is not False
            ):
                raise ValueError(f"E1 CAM-only receipt differs for {arm}/{seed}")
            arm_receipts.append({
                "seed": seed,
                "summary_sha256": sha256(report_path),
                "cam_only_per_image_sha256": sha256(cam_path),
            })
        receipts[arm] = arm_receipts
    return {
        "schema_version": 1,
        "study": "independent G4 E1 CAM-only completion audit",
        "pass": True,
        "summary_sha256": sha256(summary_path),
        "split_sha256": SPLIT_SHA,
        "arms": 2,
        "seeds_per_arm": 3,
        "images_per_seed": 371,
        "tumor_images_per_seed": 184,
        "receipts": receipts,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E1 CAM-only audit output already exists")
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "audit_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
