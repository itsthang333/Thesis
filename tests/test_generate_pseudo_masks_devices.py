from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


pytest.importorskip("torch")

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import generate_pseudo_masks as generator


def test_parser_accepts_explicit_dual_gpu_routing() -> None:
    argv = [
        "generate_pseudo_masks.py",
        "--data-root",
        "dataset",
        "--sam-checkpoint",
        "sam.pth",
        "--classifier-device",
        "cuda:0",
        "--sam-device",
        "cuda:1",
    ]
    with patch.object(sys, "argv", argv):
        args = generator.parse_args()
    assert args.classifier_device == "cuda:0"
    assert args.sam_device == "cuda:1"


@pytest.mark.parametrize("value", ["cuda:-1", "cuda:x", "cuda:", "mps"])
def test_parser_rejects_invalid_device_specs(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="device must be"):
        generator.parse_device_spec(value)


def test_runtime_device_validation_rejects_missing_index() -> None:
    with (
        patch.object(generator.torch.cuda, "is_available", return_value=True),
        patch.object(generator.torch.cuda, "device_count", return_value=1),
        pytest.raises(RuntimeError, match=r"only 1 CUDA device\(s\)"),
    ):
        generator.validate_runtime_device(
            generator.torch.device("cuda:1"), "--sam-device"
        )
