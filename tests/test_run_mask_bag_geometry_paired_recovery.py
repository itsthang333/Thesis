from __future__ import annotations

import csv
from pathlib import Path

import pytest

from project.run_mask_bag_geometry_paired_recovery import (
    semantic_reference_audit,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "project" / "run_mask_bag_geometry_paired_recovery.py"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(*, digest: str, size: int) -> dict[str, object]:
    return {
        "image_name": "IMG000001.jpeg",
        "candidate_count": 3,
        "box_count": 1,
        "positive_point_count": 2,
        "negative_point_count": 1,
        "generation_status": "ok",
        "diagnostic_path": "candidate_diagnostics/IMG000001.npz",
        "diagnostic_sha256": digest,
        "diagnostic_bytes": size,
    }


def _pseudo_row(*, mask_sha256: str = "mask") -> dict[str, object]:
    return {
        "image_name": "IMG000001.jpeg",
        "mask_sha256": mask_sha256,
        "status": "ok",
        "sam_candidate_count": 3,
        "mask_foreground_pixels": 100,
        "selected_candidates": 1,
        "selected_components": 1,
        "cam_max": 1.0,
        "cam_mean": 0.25,
        "cam_std": 0.1,
        "selection_score_min": 0.2,
        "selection_score_mean": 0.3,
        "selection_score_max": 0.4,
        "support_area_ratio": 0.5,
        "selected_area_ratio": 0.25,
        "unique_prompt_points": 3,
        "unique_positive_prompt_points": 2,
        "above_threshold_candidates": 2,
    }


def test_recovery_source_freezes_both_predictions_before_gt_evaluation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "build_segmentation_dataset" not in source
    assert "find_recovered_train_root()" in source
    assert 'RECOVERED_TRAIN_ROOT_ENV = "BTXRD_RECOVERED_TRAIN_ROOT"' in source
    assert '"legacy_direct_resize"' in source
    assert '"square_corrected_v3"' in source
    assert source.index("train_root = find_recovered_train_root()") < source.index(
        "val_root = TEMP /"
    )
    assert source.count("runner_command(") == 3
    assert source.index('geometry="legacy_direct_resize"') < source.index(
        'geometry="square_corrected_v3"'
    )
    assert source.index("write_json(pair_freeze_path, pair_freeze)") < source.index(
        "legacy_evaluation = OUTPUT /"
    )
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
    assert '"overall": 0.34024039' in source
    assert '"small": 0.17895493' in source
    assert '"medium": 0.51244178' in source
    assert '"large": 0.49370336' in source
    assert '"test": 373' in source
    assert 'f"{split}_fractional_grid_mass_summary.json"' in source
    assert '"train_fractional_grid_mass_summary.json"' in source
    assert '"val_fractional_grid_mass_summary.json"' in source
    assert 'output_dir / "summary.json"' not in source


def test_semantic_reference_accepts_byte_and_bounded_float_differences(
    tmp_path: Path,
) -> None:
    root = tmp_path / "current"
    reference = tmp_path / "reference"
    current_candidate = _candidate_row(digest="new", size=101)
    reference_candidate = _candidate_row(digest="old", size=99)
    current_pseudo = _pseudo_row()
    reference_pseudo = _pseudo_row()
    current_pseudo["cam_mean"] = 0.2500001
    _write_csv(
        root / "candidate_diagnostics_manifest.csv",
        [current_candidate],
    )
    _write_csv(root / "pseudo_mask_manifest.csv", [current_pseudo])
    _write_csv(reference / "candidate.csv", [reference_candidate])
    _write_csv(reference / "pseudo.csv", [reference_pseudo])

    audit = semantic_reference_audit(
        split="train",
        current_root=root,
        reference_candidate=reference / "candidate.csv",
        reference_pseudo=reference / "pseudo.csv",
    )
    assert audit["candidate_differing_columns"] == {
        "diagnostic_sha256": 1,
        "diagnostic_bytes": 1,
    }
    assert audit["pseudo_unlisted_field_mismatches"] == 0
    assert audit["maximum_absolute_numeric_delta"]["cam_mean"] == pytest.approx(
        1.0e-7
    )


def test_semantic_reference_rejects_final_mask_change(tmp_path: Path) -> None:
    root = tmp_path / "current"
    reference = tmp_path / "reference"
    _write_csv(
        root / "candidate_diagnostics_manifest.csv",
        [_candidate_row(digest="new", size=101)],
    )
    _write_csv(
        reference / "candidate.csv",
        [_candidate_row(digest="old", size=99)],
    )
    _write_csv(root / "pseudo_mask_manifest.csv", [_pseudo_row(mask_sha256="new")])
    _write_csv(reference / "pseudo.csv", [_pseudo_row(mask_sha256="old")])

    with pytest.raises(RuntimeError, match="Unlisted pseudo fields differ"):
        semantic_reference_audit(
            split="val",
            current_root=root,
            reference_candidate=reference / "candidate.csv",
            reference_pseudo=reference / "pseudo.csv",
        )
