from __future__ import annotations

import ast
from pathlib import Path

from project.audit_g4_e3_sam_backbone import (
    PROTOCOL_SHA,
    SAM_SHA,
    _assert_no_test,
)


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
    assert 'evaluation.get("validation_ablation") is not True' in source
    assert 'evaluation_audit.get("overall_dice_reproduced") is not False' in source
    assert 'choice_freeze.get("candidate_choices_frozen_before_spatial_gt") is not True' in source
    assert 'result.get("test_evaluated")' not in source or '_assert_no_test' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source


def test_e3_auditor_constants_match_the_frozen_runner() -> None:
    assert PROTOCOL_SHA == (
        "c65e6771cc6e68fe51de39c19374cffab35180259e8eed40eead7eed4ff6fb74"
    )
    assert SAM_SHA == {
        "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
        "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    }


def test_no_test_contract_allows_only_redundant_receipt_to_omit_read_count() -> None:
    _assert_no_test(
        {"test_evaluated": False},
        name="redundant evaluation receipt",
        require_images_read=False,
    )
    try:
        _assert_no_test({"test_evaluated": False}, name="primary artifact")
    except ValueError as error:
        assert "zero test images read" in str(error)
    else:
        raise AssertionError("primary artifact without read count was accepted")

    for invalid in (
        {"test_evaluated": True, "test_images_read": 0},
        {"test_evaluated": False, "test_images_read": 1},
        {"test_images_read": 0},
    ):
        try:
            _assert_no_test(invalid, name="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid no-test receipt was accepted: {invalid}")
