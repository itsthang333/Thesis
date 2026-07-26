import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from project.tools.audit_mae_reconstruction_probe import (
    _assert_close,
    _audit_wrapper,
    _group_bootstrap,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_wrapper_enforces_prediction_freeze_order(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        'prediction = OUTPUT / f"{role}_prediction"\n'
        'freeze_path = OUTPUT / "prediction_freeze.json"\n'
        'evaluation = OUTPUT / f"{role}_evaluation"\n'
        'comparison = OUTPUT / "paired_comparison.json"\n',
        encoding="utf-8",
    )
    result = _audit_wrapper(wrapper, _sha256(wrapper))
    assert result["ordering_positions"]["base_generation"] == 0


def test_audit_wrapper_rejects_evaluation_before_freeze(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        'prediction = OUTPUT / f"{role}_prediction"\n'
        'evaluation = OUTPUT / f"{role}_evaluation"\n'
        'freeze_path = OUTPUT / "prediction_freeze.json"\n'
        'comparison = OUTPUT / "paired_comparison.json"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordering"):
        _audit_wrapper(wrapper, _sha256(wrapper))


def test_group_bootstrap_is_complete_group_and_deterministic() -> None:
    pairs = [("a", 1.0), ("a", 3.0), ("b", -2.0)]
    first = _group_bootstrap(pairs, replicates=100, seed=7)
    second = _group_bootstrap(pairs, replicates=100, seed=7)
    assert first == second
    assert first["n_images"] == 3
    assert first["n_groups"] == 2
    assert np.isclose(first["delta_adapted_minus_base"], 2.0 / 3.0)


def test_assert_close_is_fail_closed() -> None:
    _assert_close(0.25, 0.25 + 1e-13, "close")
    with pytest.raises(ValueError, match="differs"):
        _assert_close(0.25, 0.250001, "drift")
