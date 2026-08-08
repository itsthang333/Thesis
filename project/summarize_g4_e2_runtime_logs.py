from __future__ import annotations

"""Recover exact G4 E2 arm wall times from immutable Kaggle event logs."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ATTRIBUTIONS = ("cam", "gradcam", "gradcam_plus_plus", "layercam")
PROMPTS = ("point", "box", "box_point")
EXPECTED_ARMS = tuple(f"{method}__{prompt}" for method in ATTRIBUTIONS for prompt in PROMPTS)
GENERATOR = "generate_pseudo_masks.py"
EVALUATOR = "evaluate_g4_pseudo_mask_variant.py"
GENERATION_DONE = "Prediction-first candidate diagnostics frozen: 371/371 all cases"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Kaggle log is not a non-empty event array: {path}")
    events: list[dict[str, Any]] = []
    previous = -float("inf")
    for item in value:
        if not isinstance(item, dict) or not {"time", "data"} <= set(item):
            raise ValueError(f"Kaggle log event schema differs: {path}")
        timestamp = float(item["time"])
        if timestamp < previous:
            raise ValueError(f"Kaggle log timestamps are not monotonic: {path}")
        previous = timestamp
        events.append({**item, "time": timestamp, "data": str(item["data"])})
    return events


def _arm_from_command(data: str, stage: str) -> str | None:
    if stage == "generation" and GENERATOR not in data:
        return None
    if stage == "evaluation" and EVALUATOR not in data:
        return None
    if '"command"' not in data:
        return None
    patterns = (
        r"/predictions/([a-z_]+)",
        r'--variant-name[^a-z_]+([a-z_]+)',
    )
    for pattern in patterns:
        match = re.search(pattern, data)
        if match and match.group(1) in EXPECTED_ARMS:
            return match.group(1)
    raise ValueError(f"could not recover E2 arm from {stage} command")


def summarize_logs(paths: list[Path]) -> dict[str, Any]:
    starts_generation: dict[str, tuple[float, str]] = {}
    ends_generation: dict[str, float] = {}
    starts_evaluation: dict[str, tuple[float, str]] = {}
    ends_evaluation: dict[str, float] = {}
    receipts: list[dict[str, Any]] = []

    for path in paths:
        events = read_events(path)
        active_generation: str | None = None
        for event in events:
            data = event["data"]
            timestamp = float(event["time"])
            generation_arm = _arm_from_command(data, "generation")
            if generation_arm is not None:
                if generation_arm in starts_generation:
                    raise ValueError(f"duplicate generation start for {generation_arm}")
                starts_generation[generation_arm] = (timestamp, path.name)
                active_generation = generation_arm
                continue
            if GENERATION_DONE in data:
                if active_generation is None or active_generation in ends_generation:
                    raise ValueError("generation completion has no unique active arm")
                ends_generation[active_generation] = timestamp
                active_generation = None
                continue

            evaluation_arm = _arm_from_command(data, "evaluation")
            if evaluation_arm is not None:
                if evaluation_arm in starts_evaluation:
                    raise ValueError(f"duplicate evaluation start for {evaluation_arm}")
                starts_evaluation[evaluation_arm] = (timestamp, path.name)
                continue
            variant = re.search(r'"variant"\s*:\s*"([a-z_]+)"', data)
            if variant and variant.group(1) in EXPECTED_ARMS:
                arm = variant.group(1)
                if arm not in starts_evaluation or arm in ends_evaluation:
                    raise ValueError(f"evaluation completion is not unique for {arm}")
                ends_evaluation[arm] = timestamp

        receipts.append(
            {
                "path": path.name,
                "sha256": sha256(path),
                "events": len(events),
                "first_event_seconds": float(events[0]["time"]),
                "last_event_seconds": float(events[-1]["time"]),
            }
        )

    expected = set(EXPECTED_ARMS)
    for label, mapping in (
        ("generation starts", starts_generation),
        ("generation ends", ends_generation),
        ("evaluation starts", starts_evaluation),
        ("evaluation ends", ends_evaluation),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{label} differ: missing={sorted(expected-set(mapping))}, extra={sorted(set(mapping)-expected)}")

    arms: dict[str, dict[str, Any]] = {}
    for arm in EXPECTED_ARMS:
        generation_start, generation_log = starts_generation[arm]
        evaluation_start, evaluation_log = starts_evaluation[arm]
        if generation_log != evaluation_log:
            raise ValueError(f"generation/evaluation logs differ for {arm}")
        generation_seconds = ends_generation[arm] - generation_start
        evaluation_seconds = ends_evaluation[arm] - evaluation_start
        if generation_seconds <= 0 or evaluation_seconds <= 0:
            raise ValueError(f"non-positive E2 duration for {arm}")
        arms[arm] = {
            "log": generation_log,
            "generation_start_seconds": generation_start,
            "generation_frozen_seconds": ends_generation[arm],
            "candidate_generation_wall_seconds": generation_seconds,
            "candidate_generation_seconds_per_validation_image": generation_seconds / 371.0,
            "evaluation_start_seconds": evaluation_start,
            "evaluation_complete_seconds": ends_evaluation[arm],
            "evaluation_wall_seconds": evaluation_seconds,
            "evaluation_seconds_per_validation_image": evaluation_seconds / 371.0,
            "validation_images": 371,
        }

    generation_values = [item["candidate_generation_wall_seconds"] for item in arms.values()]
    evaluation_values = [item["evaluation_wall_seconds"] for item in arms.values()]
    return {
        "schema_version": 1,
        "study": "G4 E2 attribution-prompt runtime recovery",
        "measurement": "Kaggle event-clock wall time from launcher command emission through prediction-first candidate diagnostic freeze; evaluator measured separately through final variant report",
        "hardware": "Kaggle NvidiaTeslaT4 kernel metadata; one process per six-arm kernel",
        "arms": arms,
        "aggregate": {
            "arms": 12,
            "candidate_generation_total_seconds": sum(generation_values),
            "candidate_generation_mean_seconds_per_arm": sum(generation_values) / 12.0,
            "candidate_generation_min_seconds_per_arm": min(generation_values),
            "candidate_generation_max_seconds_per_arm": max(generation_values),
            "evaluation_total_seconds": sum(evaluation_values),
            "evaluation_mean_seconds_per_arm": sum(evaluation_values) / 12.0,
        },
        "logs": receipts,
        "spatial_ground_truth_used_for_runtime_selection": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize_logs([path.resolve() for path in args.log])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arms": 12, "output_sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
