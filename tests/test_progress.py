from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from progress import should_disable_tqdm


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_progress_can_be_disabled_explicitly(monkeypatch, value):
    monkeypatch.setenv("BTXRD_DISABLE_TQDM", value)
    assert should_disable_tqdm() is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_progress_can_be_enabled_explicitly(monkeypatch, value):
    monkeypatch.setenv("BTXRD_DISABLE_TQDM", value)
    assert should_disable_tqdm() is False


def test_invalid_progress_setting_fails_closed(monkeypatch):
    monkeypatch.setenv("BTXRD_DISABLE_TQDM", "sometimes")
    with pytest.raises(ValueError, match="BTXRD_DISABLE_TQDM"):
        should_disable_tqdm()
