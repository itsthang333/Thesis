from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from progress import should_disable_tqdm


class ProgressPolicyTests(unittest.TestCase):
    def test_progress_can_be_disabled_explicitly(self) -> None:
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"BTXRD_DISABLE_TQDM": value}
            ):
                self.assertTrue(should_disable_tqdm())

    def test_progress_can_be_enabled_explicitly(self) -> None:
        for value in ("0", "false", "NO", "off"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"BTXRD_DISABLE_TQDM": value}
            ):
                self.assertFalse(should_disable_tqdm())

    def test_invalid_progress_setting_fails_closed(self) -> None:
        with patch.dict("os.environ", {"BTXRD_DISABLE_TQDM": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "BTXRD_DISABLE_TQDM"):
                should_disable_tqdm()
