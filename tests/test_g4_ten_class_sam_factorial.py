from __future__ import annotations

from pathlib import Path

from project.run_g4_e1_downstream import E1_SHA
from project.run_g4_e3_sam_backbone import SAM_SHA, SPLIT_SHA
from project.run_g4_ten_class_sam_factorial import PROTOCOL_SHA, SUPPORTED_SAM


ROOT = Path(__file__).resolve().parents[1]


def test_factorial_contract_is_validation_only_and_three_seed() -> None:
    source = (ROOT / "project" / "run_g4_ten_class_sam_factorial.py").read_text(
        encoding="utf-8"
    )
    assert set(E1_SHA["ten_class"]) == {42, 43, 44}
    assert SUPPORTED_SAM == ("vit_l",)
    assert SAM_SHA["vit_l"] == (
        "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
    )
    assert SPLIT_SHA == "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
    assert PROTOCOL_SHA == (
        "949a6f9441fa2f1964a9f2e133e95a871b8701291e5c2fd5507b0bcac9a96df6"
    )
    assert '"--splits", "val"' in source or '"--split", "val"' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source
    assert '"--target-columns", "tumor_type"' in source
    assert '"--cam-aggregation", "tumor_log_odds"' in source
