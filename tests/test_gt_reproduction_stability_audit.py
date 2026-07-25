from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_gt_reproduction_stability_under_test",
    TOOLS / "audit_gt_reproduction_stability.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class GtReproductionStabilityAuditTests(unittest.TestCase):
    def test_pearson_identical_vectors_is_one(self) -> None:
        self.assertTrue(
            math.isclose(
                AUDIT._pearson([0.1, 0.4, 0.9], [0.1, 0.4, 0.9]),
                1.0,
                abs_tol=1e-12,
            )
        )

    def test_identical_per_image_file_has_zero_repeat_delta(self) -> None:
        path = (
            ROOT
            / "artifacts"
            / "kaggle"
            / "gt_reference_independent_reproduction_v2"
            / "btxrd_gt_reference_independent_reproduction_v2"
            / "evaluation"
            / "selected_per_image.csv"
        )
        result = AUDIT.per_image_stability(path, path)
        for subgroup in ("overall", *AUDIT.SIZE_ORDER):
            self.assertEqual(
                result[subgroup]["mean_absolute_per_image_delta"],
                0.0,
            )
            self.assertTrue(
                math.isclose(
                    result[subgroup]["pearson_per_image_dice"],
                    1.0,
                    abs_tol=1e-12,
                )
            )


if __name__ == "__main__":
    unittest.main()
