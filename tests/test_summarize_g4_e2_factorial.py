from __future__ import annotations

import csv
import json
from pathlib import Path

from project.summarize_g4_e2_factorial import EXPECTED_ARMS, summarize


def test_factorial_summary_is_paired_and_finds_best(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    for arm_index, arm in enumerate(EXPECTED_ARMS):
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "audit.json").write_text(
            json.dumps({"pass": True}) + "\n", encoding="utf-8"
        )
        (arm_dir / "summary.json").write_text(
            json.dumps({"variant": arm}) + "\n", encoding="utf-8"
        )
        with (arm_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("image_id", "group_id", "gt_positive", "dice")
            )
            writer.writeheader()
            for image_index in range(184):
                writer.writerow({
                    "image_id": f"IMG{image_index:06d}.jpeg",
                    "group_id": f"group-{image_index // 2}",
                    "gt_positive": "True",
                    "dice": 0.1 + arm_index / 100.0,
                })
            for image_index in range(184, 371):
                writer.writerow({
                    "image_id": f"IMG{image_index:06d}.jpeg",
                    "group_id": f"normal-{image_index}",
                    "gt_positive": "False",
                    "dice": 1.0,
                })

    result = summarize([root], iterations=100, seed=7)
    assert result["best_arm"] == EXPECTED_ARMS[-1]
    assert result["images"] == 184
    assert result["groups"] == 92
    key = f"{EXPECTED_ARMS[-1]}_minus_{EXPECTED_ARMS[0]}"
    assert result["paired_complete_group_bootstrap"][key]["delta_mean_dice"] > 0
    assert result["test_evaluated"] is False
