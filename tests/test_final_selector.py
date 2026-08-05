from __future__ import annotations

import sys
import os
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from final_selector import average_percentile_rank, fixed_rank_fusion, select_candidate
from evaluation.frozen_test_guard import verify_frozen_test_config


class FinalSelectorTests(unittest.TestCase):
    def test_average_ties_are_deterministic(self) -> None:
        actual = average_percentile_rank(np.asarray([4.0, 1.0, 4.0, 2.0]))
        np.testing.assert_allclose(actual, [5 / 6, 0.0, 5 / 6, 1 / 3])

    def test_fixed_rule_is_equal_rank_fusion(self) -> None:
        g1 = np.asarray([1.0, 3.0, 2.0])
        upstream = np.asarray([3.0, 1.0, 2.0])
        np.testing.assert_allclose(fixed_rank_fusion(g1, upstream), [0.5, 0.5, 0.5])

    def test_tie_breaks_by_g1_then_lower_index(self) -> None:
        selected, _ = select_candidate(
            np.asarray([2.0, 3.0, 3.0]),
            np.asarray([3.0, 2.0, 2.0]),
        )
        self.assertEqual(selected, 1)

    def test_nonfinite_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            select_candidate(np.asarray([1.0, np.nan]), np.asarray([0.0, 1.0]))

    def test_test_access_requires_final_lock_but_validation_does_not(self) -> None:
        self.assertIsNone(verify_frozen_test_config(None, split="val"))
        with self.assertRaisesRegex(ValueError, "frozen-config is required"):
            verify_frozen_test_config(None, split="test")

    def test_exported_snapshot_commit_environment_is_accepted(self) -> None:
        commit = "a" * 40
        payload = {
            "schema_version": 4,
            "status": "final",
            "source": {"git_commit": commit, "git_dirty": False},
        }
        payload["freeze_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch(
                "evaluation.frozen_test_guard.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(128, "git"),
            ), mock.patch.dict(os.environ, {"BTXRD_SOURCE_COMMIT": commit}):
                document = verify_frozen_test_config(path, split="test")
        self.assertEqual(document["source"]["git_commit"], commit)


if __name__ == "__main__":
    unittest.main()
