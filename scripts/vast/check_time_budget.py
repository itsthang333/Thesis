from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _mean_seconds(records_path: Path) -> float:
    if not records_path.exists():
        return 0.0
    values = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line).get("seconds")
            if isinstance(value, int | float):
                values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--main-output", required=True)
    parser.add_argument("--benchmark-output", required=True)
    parser.add_argument("--images", type=int, required=True)
    parser.add_argument("--max-hours", type=float, default=22.0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    history_path = Path(args.main_output) / "checkpoints/hrnet_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    completed_epochs = len(history)
    mean_epoch_seconds = sum(float(row["seconds"]) for row in history) / completed_epochs
    remaining_epochs = max(0, int(config["hrnet"]["epochs"]) - completed_epochs)

    benchmark = Path(args.benchmark_output) / "evaluation"
    stage_seconds = {
        stage: _mean_seconds(benchmark / stage / "per_image.jsonl")
        for stage in ("source_maps", "sam_gallery", "rad_dino")
    }
    projected_seconds = mean_epoch_seconds * remaining_epochs + sum(
        seconds * args.images for seconds in stage_seconds.values()
    )
    payload = {
        "completed_epochs": completed_epochs,
        "remaining_epochs": remaining_epochs,
        "mean_epoch_seconds": mean_epoch_seconds,
        "stage_seconds_per_image": stage_seconds,
        "projected_hours_before_g1_selection_evaluation": projected_seconds / 3600,
        "max_hours": args.max_hours,
        "within_budget": projected_seconds <= args.max_hours * 3600,
    }
    output_path = Path(args.benchmark_output) / "time_budget.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["within_budget"] else 2)


if __name__ == "__main__":
    main()
