from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from audit_mask_bag_same_family_graph_s3_output import _verify_identity_rows
from audit_mask_bag_normal_prototype_r1_output import sha256_file
from run_mask_bag_same_family_graph_s3_arm import _float32_scalar_identity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_same_family_graph_s3_output.py"


def test_s3_auditor_is_gt_blind_and_pins_physical_helper() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "candidate_quality",
        "size_group",
        "oracle_best",
        "ground_truth",
    ):
        assert forbidden not in lowered
    assert "PHYSICAL_HELPER_SHA256" in source
    assert "_verify_validation_evidence" in source
    assert '"validation_gt_read": False' in source


def test_s3_auditor_recomputes_every_gt_blind_graph_gate() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        "pregraph_identity_audit.csv",
        "gt_blind_diagnostics.csv",
        "absolute_candidate_count_probability_spearman",
        "COUNT_SPEARMAN_CEILING = 0.5013777759365411",
        'binary_sums["view_swap_exact"] == expected_validation',
        'binary_sums["alpha_zero_identity_exact"] == expected_validation',
        'binary_sums["graph_symmetric"] == expected_validation',
        "cross_family_edges == 0",
        "non_self_edges > 0",
        'binary_sums["isolated_logits_exact"] == expected_validation',
        "gt_blind_gate_pass",
    ):
        assert required in source


def test_s3_auditor_requires_exact_operator_runtime_and_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        '"minimum_iou": 0.25',
        '"minimum_containment": 0.5',
        '"alpha": 0.5',
        '"iterations": 10',
        'binding.get("schema_version") != 2',
        "NUMERIC_IDENTITY_ADDENDUM_SHA256",
        "SCALAR_IDENTITY_MAX_FLOAT32_ULPS = 4",
        '"accepted_row_identity_pass"',
        'runtime.get("validation_shards") != [186, 185]',
        'not all("T4" in name for name in runtime["cuda_device_names"])',
        "physical_cache_records_verified",
        "physical_pregraph_identity_rows_verified",
    ):
        assert required in source


def _identity_row(observed: float, accepted: float) -> dict[str, object]:
    row: dict[str, object] = {
        "image_id": "IMG1.jpeg",
        "candidate_count": 2,
        "base_candidate_logits_sha256": "a" * 64,
        "alpha_zero_identity_exact": 1,
        "accepted_selected_index_exact": 1,
        "accepted_map_sha256_exact": 1,
    }
    all_exact = True
    all_pass = True
    for prefix in (
        "accepted_selected_logit",
        "accepted_bag_logit",
        "accepted_bag_probability",
    ):
        evidence = _float32_scalar_identity(observed, accepted)
        row.update(
            {
                f"{prefix}_observed": evidence["observed"],
                f"{prefix}_reference": evidence["accepted"],
                f"{prefix}_abs_delta": evidence["absolute_delta"],
                f"{prefix}_float32_spacing": evidence["float32_spacing"],
                f"{prefix}_tolerance": evidence["tolerance"],
                f"{prefix}_exact": evidence["exact"],
                f"{prefix}_within_tolerance": evidence["within_tolerance"],
            }
        )
        all_exact &= bool(evidence["exact"])
        all_pass &= bool(evidence["within_tolerance"])
    row["accepted_row_exact"] = int(all_exact)
    row["accepted_row_identity_pass"] = int(all_pass)
    return row


def _write_identity(root: Path, row: dict[str, object]) -> dict[str, str]:
    path = root / "pregraph_identity_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return {"pregraph_identity_audit_sha256": sha256_file(path)}


def test_s3_auditor_independently_enforces_four_ulp_bound(tmp_path: Path) -> None:
    accepted = float(np.float32(16.0))
    spacing = abs(float(np.spacing(np.float32(accepted))))
    freeze = _write_identity(tmp_path, _identity_row(accepted + 4 * spacing, accepted))
    result = _verify_identity_rows(tmp_path, freeze, expected_validation=1)
    assert result["accepted_row_identity_pass_records"] == 1

    outside = tmp_path / "outside"
    outside.mkdir()
    freeze = _write_identity(outside, _identity_row(accepted + 8 * spacing, accepted))
    with pytest.raises(ValueError, match="exceeds frozen tolerance"):
        _verify_identity_rows(outside, freeze, expected_validation=1)
