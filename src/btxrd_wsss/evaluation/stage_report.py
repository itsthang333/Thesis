from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class StageReportWriter:
    """Append-only per-image metrics plus atomically replaced summaries."""

    def __init__(self, output_dir: str | Path, stage: str) -> None:
        self.directory = Path(output_dir) / "evaluation" / stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_path = self.directory / "per_image.jsonl"
        self.summary_path = self.directory / "summary.json"

    def append(self, record: dict[str, Any]) -> None:
        if "image_id" not in record:
            raise ValueError("Stage record requires image_id")
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def completed_ids(self) -> set[str]:
        if not self.records_path.exists():
            return set()
        with self.records_path.open("r", encoding="utf-8") as handle:
            return {str(json.loads(line)["image_id"]) for line in handle if line.strip()}

    def write_summary(self, summary: dict[str, Any]) -> None:
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.summary_path)

    def records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def write_numeric_summary(self, extra: dict[str, Any] | None = None) -> None:
        rows = self.records()
        numeric: dict[str, list[float]] = {}

        def collect(prefix: str, payload: dict[str, Any]) -> None:
            for key, value in payload.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    collect(name, value)
                elif (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and np.isfinite(value)
                ):
                    numeric.setdefault(name, []).append(float(value))

        for row in rows:
            collect("", row)
        summary = {
            "image_count": len(rows),
            "mean": {key: float(np.mean(values)) for key, values in numeric.items()},
        }
        if extra:
            summary.update(extra)
        self.write_summary(summary)
