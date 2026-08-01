from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "freeze_rich_gallery_g1_diagnostic.py"


def test_stage_a_is_spatial_annotation_and_test_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "mask_tensor",
        'split="test"',
        "test_loader",
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": False' in source
    assert '"spatial_ground_truth_used": False' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source


def test_stage_a_reproduces_g1_before_freezing_all_candidate_evidence() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reproduction = source.index("reproduction_audit = _compare_reproduction(")
    score_freeze = source.index("saved_score = save_candidate_score_evidence(")
    final_freeze = source.index('freeze_path = args.output_dir / "diagnostic_freeze.json"')
    assert reproduction < score_freeze < final_freeze
    assert "candidate_indices=kept" in source
    assert "candidate_logits=logits_np" in source
    assert "descriptors=descriptors" in source
    assert "flipped_descriptors=flipped" in source


def test_stage_a_binds_exact_rich_gallery_and_g1_checkpoint() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "args.maximum_candidates != 243" in source
    assert "expected_baseline_checkpoint_sha256" in source
    assert "expected_baseline_freeze_sha256" in source
    assert "val_candidate_manifest_sha256" in source
    assert "val_pseudo_manifest_sha256" in source

