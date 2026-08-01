from __future__ import annotations

"""Independent no-GT auditor configuration for same-gallery BAS B4."""

import numpy as np

import audit_bas_candidate_descriptor_core as base


EXPERIMENT_ID = "EXP-20260801-codex-b4-same-gallery-bas-semantic-v1"
ARMS = (
    "geometry_v3_plus_upstream_equal_rank",
    "geometry_v3_plus_upstream_plus_class_contrast_bas",
)
EXPECTED_EXTRA_PROVENANCE = {
    "input_size": 448,
    "semantic_map": "tumor_over_tumor_plus_normal",
    "control_formula": "mean_rank_geometry_v3_upstream",
    "semantic_formula": "mean_rank_geometry_v3_upstream_class_contrast_bas",
}


def _expected_arm_scores(
    base_rank: np.ndarray,
    upstream_rank: np.ndarray,
    bas_rank: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        ARMS[0]: np.asarray(0.5 * (base_rank + upstream_rank), dtype=np.float32),
        ARMS[1]: np.asarray(
            (base_rank + upstream_rank + bas_rank) / 3.0,
            dtype=np.float32,
        ),
    }


def _correlation_reference(
    base_rank: np.ndarray,
    upstream_rank: np.ndarray,
) -> np.ndarray:
    return 0.5 * (base_rank + upstream_rank)


def _audit_extra_evidence(evidence: object, image_id: str) -> None:
    names = set(evidence.files)
    required = {
        "activation",
        "normal_activation",
        "tumor_activation",
        "class_contrast_activation",
    }
    if not required.issubset(names):
        raise ValueError(f"B4 class-contrast evidence is incomplete: {image_id}")
    normal = np.asarray(evidence["normal_activation"], dtype=np.float32)
    tumor = np.asarray(evidence["tumor_activation"], dtype=np.float32)
    contrast = np.asarray(evidence["class_contrast_activation"], dtype=np.float32)
    expected = tumor / np.maximum(tumor + normal, 1.0e-8)
    if (
        normal.shape != tumor.shape
        or tumor.shape != contrast.shape
        or not np.isfinite(normal).all()
        or not np.isfinite(tumor).all()
        or not np.allclose(contrast, expected, atol=2.0e-7, rtol=0)
        or not np.array_equal(np.asarray(evidence["activation"]), contrast)
    ):
        raise ValueError(f"B4 class-contrast identity mismatch: {image_id}")


def main() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.ARMS = ARMS
    base.AUDIT_ID = "independent_same_gallery_bas_semantic_b4_output_v1"
    base.AUDIT_PASS_STATUS = (
        "PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED"
    )
    base.REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE = False
    base.CORRELATION_KEY = "mean_contrast_baseline_rank_correlation"
    base.CHANGE_FRACTION_KEY = "semantic_changed_selection_fraction"
    base.MAXIMUM_CORRELATION = 0.95
    base.MINIMUM_CHANGE_FRACTION = 0.05
    base.EXPECTED_EXTRA_PROVENANCE = EXPECTED_EXTRA_PROVENANCE
    base._expected_arm_scores = _expected_arm_scores
    base._correlation_reference = _correlation_reference
    base._audit_extra_evidence = _audit_extra_evidence
    base.main()


if __name__ == "__main__":
    main()
