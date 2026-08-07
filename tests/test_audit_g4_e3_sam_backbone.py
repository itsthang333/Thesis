from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e3_independent_auditor_is_validation_only_and_checks_all_backbones() -> None:
    source = (ROOT / "project" / "audit_g4_e3_sam_backbone.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert '"vit_b"' in source
    assert '"vit_l"' in source
    assert '"vit_h"' in source
    assert 'int(summary.get("small", {}).get("n", -1)) != 94' in source
    assert 'int(summary.get("medium", {}).get("n", -1)) != 72' in source
    assert 'int(summary.get("large", {}).get("n", -1)) != 18' in source
    assert 'choice_freeze.get("candidate_choices_frozen_before_spatial_gt") is not True' in source
    assert 'result.get("test_evaluated")' not in source or '_assert_no_test' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source
