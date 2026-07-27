from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "compare_mask_bag_evaluated_arms.py"


def test_evaluated_arm_comparator_never_reopens_segmentation_data() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "datasets.btxrd" not in imported
    assert "PIL" not in imported
    assert "Annotations" not in text
    assert '"ground_truth_reopened": False' in text
    assert '"consumer_trained": False' in text
    assert '"test_evaluated": False' in text


def test_evaluated_arm_comparator_freezes_cohort_metric_and_bootstrap() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "len(rows) != 184" in text
    assert 'EXPECTED_COUNTS = {"small": 94, "medium": 72, "large": 18}' in text
    assert "args.bootstrap_replicates != 10000" in text
    assert "args.bootstrap_seed != 20261101" in text
    assert "Frozen paired field" in text
    assert '"oracle_best_single_dice"' in text
    assert '"complete_misses_included": True' in text
    assert '"misses_recovered"' in text
    assert '"overlaps_lost"' in text


def test_evaluated_arm_comparator_runs_complete_paired_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "project"))
    module = importlib.import_module("compare_mask_bag_evaluated_arms")
    fields = [
        "image_id",
        "group_id",
        "gt_area_ratio",
        "size_group",
        "dice",
        "oracle_best_single_dice",
        "complete_miss",
        "selected_area_ratio",
    ]
    subgroup_rows = (("small", 94, 0.005), ("medium", 72, 0.02), ("large", 18, 0.08))
    reference_rows = []
    candidate_rows = []
    index = 0
    for subgroup, count, area in subgroup_rows:
        for _ in range(count):
            common = {
                "image_id": f"IMG{index:06d}.jpeg",
                "group_id": f"group-{index // 2:04d}",
                "gt_area_ratio": area,
                "size_group": subgroup,
                "oracle_best_single_dice": 0.6,
            }
            reference_rows.append(
                {
                    **common,
                    "dice": 0.2,
                    "complete_miss": int(index < 12),
                    "selected_area_ratio": 0.03,
                }
            )
            candidate_rows.append(
                {
                    **common,
                    "dice": 0.3,
                    "complete_miss": int(6 <= index < 8),
                    "selected_area_ratio": 0.03,
                }
            )
            index += 1

    def write(name: str, rows: list[dict[str, object]]) -> tuple[Path, str]:
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    candidate_path, candidate_sha = write("candidate.csv", candidate_rows)
    reference_path, reference_sha = write("reference.csv", reference_rows)
    output = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--candidate-per-image",
            str(candidate_path),
            "--expected-candidate-sha256",
            candidate_sha,
            "--candidate-name",
            "candidate",
            "--reference-per-image",
            str(reference_path),
            "--expected-reference-sha256",
            reference_sha,
            "--reference-name",
            "reference",
            "--output-dir",
            str(output),
        ],
    )
    module.main()
    payload = json.loads((output / "paired_comparison.json").read_text(encoding="utf-8"))
    assert payload["cohort"] == {"tumor": 184, "small": 94, "medium": 72, "large": 18}
    for subgroup in ("overall", "small", "medium", "large"):
        assert payload["metrics"][subgroup]["delta_candidate_minus_reference"] == pytest.approx(0.1)
    assert payload["metrics"]["overall"]["reference_complete_misses"] == 12
    assert payload["metrics"]["overall"]["candidate_complete_misses"] == 2
    assert payload["metrics"]["overall"]["misses_recovered"] == 10
    assert payload["metrics"]["overall"]["overlaps_lost"] == 0
    assert payload["ground_truth_reopened"] is False
