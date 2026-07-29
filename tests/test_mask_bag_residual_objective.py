from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "models" / "mask_bag_residual_objective.py"
PROJECT = ROOT / "project"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _load_module():
    if str(PROJECT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location("mask_bag_residual_objective", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_contains_no_argmax_or_instance_target_path() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    assert "argmax" not in source
    assert "self_guided_instance_loss" not in source
    assert "candidate_quality" not in source
    assert "segmentation" not in source


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_zero_residual_objective_is_finite_and_drift_free() -> None:
    torch = __import__("torch")
    module = _load_module()
    logits = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float32)
    flipped = torch.tensor([[0.2, -0.1], [0.0, 0.5]], dtype=torch.float32)
    residual = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])
    total, details = module.residual_arm_objective(
        logits,
        flipped,
        residual,
        residual,
        valid,
        labels,
        module.ResidualObjectiveConfig(),
    )

    assert torch.isfinite(total)
    assert details["residual_drift"].item() == 0.0
    assert details["consistency"].item() > 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_only_valid_residuals_contribute_to_drift() -> None:
    torch = __import__("torch")
    module = _load_module()
    original = torch.tensor([[0.2, 100.0]], dtype=torch.float32)
    flipped = torch.tensor([[0.2, -100.0]], dtype=torch.float32)
    original_residual = torch.tensor([[2.0, 1000.0]], dtype=torch.float32)
    flipped_residual = torch.tensor([[4.0, -1000.0]], dtype=torch.float32)
    valid = torch.tensor([[True, False]])
    _total, details = module.residual_arm_objective(
        original,
        flipped,
        original_residual,
        flipped_residual,
        valid,
        torch.tensor([1.0]),
        module.ResidualObjectiveConfig(),
    )

    assert details["residual_drift"].item() == pytest.approx(10.0)
    assert details["consistency"].item() == 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch unavailable locally")
def test_gradients_flow_to_valid_residual_logits() -> None:
    torch = __import__("torch")
    module = _load_module()
    base = torch.tensor([[0.1, -0.3]], dtype=torch.float32, requires_grad=True)
    original_residual = torch.zeros_like(base, requires_grad=True)
    flipped_residual = torch.zeros_like(base, requires_grad=True)
    valid = torch.ones_like(base, dtype=torch.bool)
    total, _details = module.residual_arm_objective(
        base + original_residual,
        base + flipped_residual,
        original_residual,
        flipped_residual,
        valid,
        torch.tensor([1.0]),
        module.ResidualObjectiveConfig(),
    )
    total.backward()

    assert base.grad is None
    assert original_residual.grad is not None
    assert flipped_residual.grad is not None
    assert torch.count_nonzero(original_residual.grad) > 0
    assert torch.count_nonzero(flipped_residual.grad) > 0
