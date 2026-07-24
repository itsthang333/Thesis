from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BTXRD_BEST_PIPELINE


class OfficialWsssConfigTests(unittest.TestCase):
    def test_binary_cam_sam_tta_recipe_is_canonical(self) -> None:
        self.assertEqual(BTXRD_BEST_PIPELINE.name, "btxrd_best")
        self.assertEqual(BTXRD_BEST_PIPELINE.target_columns, ("tumor",))
        self.assertTrue(BTXRD_BEST_PIPELINE.cam_tta_flip)
        self.assertFalse(BTXRD_BEST_PIPELINE.cam_contrast_normal)
        self.assertEqual(BTXRD_BEST_PIPELINE.cam_target_class, "ground_truth")
        self.assertEqual(
            BTXRD_BEST_PIPELINE.cam_percentile_values,
            (85.0, 90.0, 95.0),
        )
        self.assertEqual(BTXRD_BEST_PIPELINE.selection_method, "coverage_mass_sam")
        self.assertEqual(BTXRD_BEST_PIPELINE.support_clip_kernel, 5)


if __name__ == "__main__":
    unittest.main()
