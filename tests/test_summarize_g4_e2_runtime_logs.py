from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from summarize_g4_e2_runtime_logs import EXPECTED_ARMS, summarize_logs  # noqa: E402


def write_log(path: Path, arms: tuple[str, ...], start: float) -> None:
    events: list[dict[str, object]] = []
    time = start
    for arm in arms:
        events.append({"stream_name": "stdout", "time": time, "data": json.dumps({"command": ["python", "generate_pseudo_masks.py", "--output-dir", f"/kaggle/working/g4_e2/predictions/{arm}"]})})
        time += 10.0
        events.append({"stream_name": "stdout", "time": time, "data": "Prediction-first candidate diagnostics frozen: 371/371 all cases"})
        time += 1.0
    for arm in arms:
        events.append({"stream_name": "stdout", "time": time, "data": json.dumps({"command": ["python", "evaluate_g4_pseudo_mask_variant.py", "--variant-name", arm]})})
        time += 2.0
        events.append({"stream_name": "stdout", "time": time, "data": json.dumps({"variant": arm})})
        time += 1.0
    path.write_text(json.dumps(events), encoding="utf-8")


def test_recovers_all_twelve_exact_durations(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    write_log(first, EXPECTED_ARMS[:6], 1.0)
    write_log(second, EXPECTED_ARMS[6:], 2.0)
    report = summarize_logs([first, second])
    assert report["aggregate"]["arms"] == 12
    assert report["aggregate"]["candidate_generation_total_seconds"] == 120.0
    assert report["aggregate"]["evaluation_total_seconds"] == 24.0
    assert set(report["arms"]) == set(EXPECTED_ARMS)
    assert all(item["validation_images"] == 371 for item in report["arms"].values())


def test_missing_arm_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.log"
    write_log(path, EXPECTED_ARMS[:-1], 1.0)
    with pytest.raises(ValueError, match="differ"):
        summarize_logs([path])


def test_non_monotonic_log_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.log"
    path.write_text(json.dumps([
        {"time": 2.0, "data": "a"},
        {"time": 1.0, "data": "b"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="not monotonic"):
        summarize_logs([path])
