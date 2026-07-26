from __future__ import annotations

import unittest

from evaluate_rad_dino_geodesic_seed_probe import paired_group_report


class EvaluateRadDinoGeodesicSeedProbeTests(unittest.TestCase):
    def test_complete_group_bootstrap_is_deterministic(self) -> None:
        rows = [
            ("a", 0.1),
            ("a", 0.3),
            ("b", -0.2),
            ("c", 0.4),
        ]
        first = paired_group_report(rows, iterations=1000, seed=42)
        second = paired_group_report(rows, iterations=1000, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 0.15)
        self.assertEqual(first["images"], 4)
        self.assertEqual(first["groups"], 3)

    def test_bootstrap_rejects_empty_rows(self) -> None:
        with self.assertRaises(ValueError):
            paired_group_report([], iterations=100, seed=42)


if __name__ == "__main__":
    unittest.main()
