from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "models" / "mask_bag_oof_selection.py"
SPEC = importlib.util.spec_from_file_location("mask_bag_oof_selection", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(k: int, losses: list[float], association: float) -> dict[str, object]:
    return {
        "prototype_count": k,
        "fold_image_bce": losses,
        "count_probability_spearman": association,
    }


class OOFSelectionTests(unittest.TestCase):
    def test_smallest_k_wins_inside_best_one_se_band(self) -> None:
        result = MODULE.select_prototype_count_one_standard_error(
            [
                _candidate(8, [0.40, 0.43, 0.39, 0.44, 0.41], -0.30),
                _candidate(16, [0.39, 0.40, 0.38, 0.41, 0.39], -0.31),
                _candidate(32, [0.35, 0.42, 0.36, 0.43, 0.36], -0.32),
            ],
            baseline_absolute_count_association=0.30,
            maximum_absolute_count_association_increase=0.02,
        )

        self.assertEqual(result["best_mean_prototype_count"], 32)
        self.assertEqual(result["selected_prototype_count"], 16)
        self.assertFalse(result["validation_segmentation_quality_used"])

    def test_count_guard_rejects_lower_bce_shortcut(self) -> None:
        result = MODULE.select_prototype_count_one_standard_error(
            [
                _candidate(8, [0.50, 0.50, 0.50, 0.50, 0.50], -0.20),
                _candidate(16, [0.30, 0.30, 0.30, 0.30, 0.30], -0.60),
            ],
            baseline_absolute_count_association=0.20,
        )

        self.assertEqual(result["selected_prototype_count"], 8)
        indexed = {row["prototype_count"]: row for row in result["candidates"]}
        self.assertFalse(indexed[16]["count_guard_pass"])

    def test_clear_oof_improvement_selects_larger_k(self) -> None:
        result = MODULE.select_prototype_count_one_standard_error(
            [
                _candidate(8, [0.60, 0.61, 0.59, 0.60, 0.60], 0.10),
                _candidate(16, [0.40, 0.41, 0.39, 0.40, 0.40], 0.11),
            ],
            baseline_absolute_count_association=0.10,
        )

        self.assertEqual(result["selected_prototype_count"], 16)

    def test_all_count_shortcuts_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "all prototype counts"):
            MODULE.select_prototype_count_one_standard_error(
                [_candidate(8, [0.4] * 5, 0.5)],
                baseline_absolute_count_association=0.1,
            )

    def test_selection_is_input_order_independent(self) -> None:
        rows = [
            _candidate(8, [0.42, 0.43, 0.41, 0.42, 0.42], -0.2),
            _candidate(16, [0.40, 0.41, 0.39, 0.40, 0.40], -0.2),
            _candidate(32, [0.39, 0.40, 0.38, 0.39, 0.39], -0.2),
        ]
        first = MODULE.select_prototype_count_one_standard_error(
            rows, baseline_absolute_count_association=0.2
        )
        second = MODULE.select_prototype_count_one_standard_error(
            list(reversed(rows)), baseline_absolute_count_association=0.2
        )

        self.assertEqual(first, second)

    def test_source_api_is_label_only_and_subgroup_free(self) -> None:
        source = SOURCE.read_text(encoding="utf-8").lower()
        self.assertNotIn("mask_tensor", source)
        self.assertNotIn("dice", source)
        self.assertNotIn("subgroup", source)
        self.assertNotIn("lesion", source)
