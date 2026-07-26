from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_gt_reproduction_v4_under_test",
    TOOLS / "audit_gt_reproduction_v4.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class GtReproductionV4AuditTests(unittest.TestCase):
    def test_line_content_comparison_ignores_transport_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.py"
            right = root / "right.py"
            left.write_bytes(b"print('a')\nprint('b')\n")
            right.write_bytes(b"\xef\xbb\xbfprint('a')\r\nprint('b')\r\n")
            self.assertTrue(AUDIT._line_content_equal(left, right))

    def test_selected_means_reads_all_fixed_subgroups(self) -> None:
        root = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "gt_reference_independent_reproduction_v3"
            / "btxrd_gt_reference_independent_reproduction_v3"
        )
        means = AUDIT._selected_means(root)
        self.assertEqual(set(means), {"overall", *AUDIT.SIZE_ORDER})
        self.assertAlmostEqual(means["overall"], 0.4994012363463918)
        self.assertAlmostEqual(means["large_ge_5pct"], 0.7482716392488015)


if __name__ == "__main__":
    unittest.main()
