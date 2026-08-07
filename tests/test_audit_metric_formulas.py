from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metric_formula_audit_passes_and_is_test_free(tmp_path: Path) -> None:
    output = tmp_path / "metric_audit.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "project" / "audit_metric_formulas.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["checks_passed"] == report["checks_total"]
    assert report["checks_total"] >= 15
    assert report["spatial_annotations_read"] == 0
    assert report["test_images_read"] == 0
    assert report["test_evaluated"] is False
