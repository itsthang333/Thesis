from __future__ import annotations

import ast
from pathlib import Path

from project.summarize_g4_e1_downstream_pair import SEEDS, mean_sd


ROOT = Path(__file__).resolve().parents[1]


def test_e1_pair_summary_contract_is_three_seed_paired_and_validation_only() -> None:
    source = (ROOT / "project" / "summarize_g4_e1_downstream_pair.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert SEEDS == (42, 43, 44)
    assert "paired_group_bootstrap_deltas" in source
    assert '"ten_class_minus_binary"' in source
    assert '"small", "medium", "large"' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source


def test_mean_sd_uses_sample_standard_deviation() -> None:
    result = mean_sd([1.0, 2.0, 3.0])
    assert result == {"mean": 2.0, "sample_sd": 1.0}
