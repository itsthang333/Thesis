from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_fractional_grid_mass.py"


def test_fractional_mass_audit_is_gt_blind_and_fail_closed() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "datasets.btxrd" not in imported
    assert "Annotations" not in text
    assert 'choices=["train", "val"]' in text
    assert "validate_candidate_diagnostics_manifest(" in text
    assert "locate_verified_image(" in text
    assert '"ground_truth_loaded": False' in text
    assert '"consumer_trained": False' in text
    assert '"test_evaluated": False' in text
    assert "args.minimum_grid_mass != 0.25" in text
    assert "args.maximum_candidates != 81" in text


def test_fractional_mass_audit_uses_v3_projection_contract() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "project_direct_resize_masks_to_square(" in text
    assert "output_size = token_grid_size * oversampling" in text
    assert 'mode="area"' in text
    assert "args.token_grid_size != 32" in text
    assert "args.oversampling != 4" in text
    assert "retained_below_one_fraction" in text
    assert '"flip_grid_mass"' in text
    assert '"absolute_flip_mass_delta"' in text
    assert "np.array_equal(retained, flipped_retained)" in text
    assert "np.allclose(masses, flipped_masses" in text
    assert '"original_flip_validity_aligned": True' in text
