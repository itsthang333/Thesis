from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run_g4_e1_cam_only_completion import aggregate_seed_results  # noqa: E402


def test_aggregate_seed_results_reports_mean_and_sample_sd() -> None:
    rows = [
        {
            "cam_only": {"mean_tumor_dice": cam},
            "final_selected": {
                "dice": selected,
                "candidate_oracle_dice": oracle,
            },
        }
        for cam, selected, oracle in (
            (0.1, 0.2, 0.4),
            (0.2, 0.3, 0.5),
            (0.3, 0.4, 0.6),
        )
    ]
    result = aggregate_seed_results(rows)
    assert result["seeds"] == 3
    assert abs(result["cam_only_mean_dice"] - 0.2) < 1e-12
    assert abs(result["cam_only_sample_sd_dice"] - 0.1) < 1e-12
    assert abs(result["final_selected_mean_dice"] - 0.3) < 1e-12
    assert abs(result["final_candidate_oracle_mean_dice"] - 0.5) < 1e-12
