from __future__ import annotations

from project.audit_x4_s2c_training_output import _json_normalized


def test_s2c_audit_normalizes_tuple_to_json_list_without_changing_values() -> None:
    checkpoint = {"cpm_scales": (0.75, 1.0, 1.25), "nested": {"flag": False}}
    sidecar = {"cpm_scales": [0.75, 1.0, 1.25], "nested": {"flag": False}}
    assert _json_normalized(checkpoint) == _json_normalized(sidecar)
    assert _json_normalized({"cpm_scales": (0.75, 1.0)}) != _json_normalized(sidecar)
