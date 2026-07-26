import hashlib
from pathlib import Path

import numpy as np
import pytest

from project.tools.audit_nominal_patch_memory_probe import (
    _assert_close,
    _audit_wrapper,
    _group_bootstrap,
    _semantic_array_hash,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wrapper_audit_enforces_freeze_before_evaluation(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        'prediction_root = OUTPUT / "predictions"\n'
        'freeze_path = OUTPUT / "prediction_freeze.json"\n'
        'evaluation = OUTPUT / f"{arm}_evaluation"\n'
        'comparison = OUTPUT / "paired_comparison.json"\n',
        encoding="utf-8",
    )
    result = _audit_wrapper(wrapper, _sha256(wrapper))
    assert result["ordering_positions"]["prediction_generation"] == 0


def test_wrapper_audit_rejects_evaluation_before_freeze(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        'prediction_root = OUTPUT / "predictions"\n'
        'evaluation = OUTPUT / f"{arm}_evaluation"\n'
        'freeze_path = OUTPUT / "prediction_freeze.json"\n'
        'comparison = OUTPUT / "paired_comparison.json"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordering"):
        _audit_wrapper(wrapper, _sha256(wrapper))


def test_group_bootstrap_is_deterministic_and_complete_group() -> None:
    pairs = [("a", 1.0), ("a", 3.0), ("b", -2.0)]
    first = _group_bootstrap(pairs, replicates=100, seed=7)
    second = _group_bootstrap(pairs, replicates=100, seed=7)
    assert first == second
    assert first["n_images"] == 3
    assert first["n_groups"] == 2
    assert np.isclose(
        first["delta_multiscale_minus_single_scale"],
        2.0 / 3.0,
    )


def test_semantic_array_hash_and_close_checks_fail_closed() -> None:
    values = np.array([0.1, 0.2], dtype=np.float32)
    assert _semantic_array_hash(values) == _semantic_array_hash(values.copy())
    assert _semantic_array_hash(values) != _semantic_array_hash(values[::-1])
    _assert_close(0.25, 0.25 + 1e-13, "close")
    with pytest.raises(ValueError, match="differs"):
        _assert_close(0.25, 0.250001, "drift")
