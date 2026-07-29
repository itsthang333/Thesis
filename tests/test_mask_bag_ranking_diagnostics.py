from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "models" / "mask_bag_ranking_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("mask_bag_ranking_diagnostics", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RankingDiagnosticsTests(unittest.TestCase):
    def test_exact_ranking_reports_regret_top_k_and_spearman(self) -> None:
        result = MODULE.candidate_ranking_diagnostics(
            [0.9, 0.8, 0.7, 0.6],
            [0.1, 0.4, 0.9, 0.2],
            top_k=(1, 2, 3),
        )

        self.assertEqual(result["selected_candidate_index"], 0)
        self.assertEqual(result["oracle_candidate_index"], 2)
        self.assertEqual(result["oracle_best_rank"], 3)
        self.assertAlmostEqual(result["selected_to_oracle_regret"], 0.8)
        self.assertEqual(
            result["top_k_oracle_reach"], {"1": False, "2": False, "3": True}
        )
        self.assertEqual(result["top_k_best_quality"], {"1": 0.1, "2": 0.4, "3": 0.9})
        self.assertAlmostEqual(result["score_quality_spearman"], -0.4)

    def test_oracle_ties_use_the_best_frozen_rank(self) -> None:
        result = MODULE.candidate_ranking_diagnostics(
            [0.1, 0.8, 0.7],
            [1.0, 0.2, 1.0],
            top_k=(1, 2),
        )

        self.assertEqual(result["oracle_candidate_index"], 2)
        self.assertEqual(result["oracle_best_rank"], 2)
        self.assertEqual(result["top_k_oracle_reach"], {"1": False, "2": True})

    def test_invalid_candidates_are_excluded_and_score_ties_are_stable(self) -> None:
        result = MODULE.candidate_ranking_diagnostics(
            [0.5, 100.0, 0.5],
            [0.2, 1.0, 0.8],
            valid=[True, False, True],
            top_k=(1, 5),
        )

        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["selected_candidate_index"], 0)
        self.assertEqual(result["oracle_candidate_index"], 2)
        self.assertEqual(result["oracle_best_rank"], 2)
        self.assertIsNone(result["score_quality_spearman"])
        self.assertEqual(result["top_k_oracle_reach"], {"1": False, "5": True})

    def test_empty_bag_is_an_explicit_complete_ranking_case(self) -> None:
        result = MODULE.candidate_ranking_diagnostics([], [], top_k=(1, 3))

        self.assertEqual(result["candidate_count"], 0)
        self.assertIsNone(result["selected_candidate_index"])
        self.assertEqual(result["selected_quality"], 0.0)
        self.assertEqual(result["selected_to_oracle_regret"], 0.0)
        self.assertEqual(result["top_k_oracle_reach"], {"1": False, "3": False})

    def test_summary_keeps_recovered_and_lost_misses_separate(self) -> None:
        rows = [
            {
                "size_group": "small",
                "selected_to_oracle_regret": 0.4,
                "score_quality_spearman": 0.5,
                "top_k_oracle_reach": {"1": False, "3": True},
                "top_k_regret": {"1": 0.4, "3": 0.0},
                "selected_complete_miss": False,
                "baseline_complete_miss": True,
            },
            {
                "size_group": "small",
                "selected_to_oracle_regret": 0.2,
                "score_quality_spearman": None,
                "top_k_oracle_reach": {"1": False, "3": False},
                "top_k_regret": {"1": 0.2, "3": 0.1},
                "selected_complete_miss": True,
                "baseline_complete_miss": False,
            },
        ]

        summary = MODULE.summarize_ranking_diagnostics(
            rows, subgroup_names=("overall", "small"), top_k=(1, 3)
        )

        self.assertEqual(summary["overall"]["n"], 2)
        self.assertAlmostEqual(
            summary["small"]["mean_selected_to_oracle_regret"], 0.3
        )
        self.assertEqual(
            summary["small"]["top_k_oracle_reach_rate"], {"1": 0.0, "3": 0.5}
        )
        self.assertEqual(summary["small"]["complete_misses"], 1)
        self.assertEqual(summary["small"]["recovered_baseline_misses"], 1)
        self.assertEqual(summary["small"]["lost_baseline_hits"], 1)

    def test_evaluation_only_module_has_no_dataset_or_training_dependency(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("datasets.factory", source)
        self.assertNotIn("torch", source)
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn("RESEARCH_LOG", source)
