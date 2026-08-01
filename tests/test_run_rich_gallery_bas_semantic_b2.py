from __future__ import annotations

import numpy as np

from run_rich_gallery_bas_semantic_b2 import (
    CONTROL_ARM,
    SEMANTIC_ARM,
    _materialize_pair,
    _rank_correlation,
    pack_binary_mask,
    score_one_image,
    unpack_binary_mask,
)


def _candidate_payload() -> dict[str, np.ndarray]:
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    masks[1, :2, :2] = 1
    masks[2, 5:, 5:] = 1
    return {
        "sam_masks": masks,
        "selection_scores": np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        "proposal_source_ids": np.asarray(
            ["classifier448", "layercam320", "external_saliency"]
        ),
    }


def _stage_a_payload() -> dict[str, np.ndarray]:
    return {
        "candidate_indices": np.asarray([0, 1, 2], dtype=np.int32),
        "source_ids": np.asarray([0, 1, 2], dtype=np.int16),
        "upstream_scores": np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        "g1_frozen_candidate_logits": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
    }


def _baseline_row() -> dict[str, str]:
    return {
        "variant": "g1_frozen__rank_fusion",
        "image_id": "example.png",
        "group_id": "group-a",
        "tumor": "1",
        "candidate_payload_sha256": "a" * 64,
        "selected_local_index": "1",
        "selected_candidate_index": "1",
    }


def test_score_one_image_keeps_exact_control_and_adds_semantic_choice() -> None:
    activation = np.zeros((8, 8), dtype=np.float32)
    activation[2:6, 2:6] = 1.0
    scored = score_one_image(
        activation,
        _baseline_row(),
        _candidate_payload(),
        _stage_a_payload(),
    )
    assert scored.baseline_local_index == 1
    assert scored.semantic_local_index == 0
    assert scored.baseline_source == "layercam320"
    assert scored.semantic_source == "classifier448"
    np.testing.assert_array_equal(scored.semantic_mask, _candidate_payload()["sam_masks"][0])


def test_prediction_pack_round_trip_is_exact() -> None:
    mask = np.zeros((9, 11), dtype=bool)
    mask[2:7, 3:8] = True
    packed, shape = pack_binary_mask(mask)
    np.testing.assert_array_equal(unpack_binary_mask(packed, shape), mask)


def test_constant_bas_rank_fails_closed_as_maximally_redundant() -> None:
    assert _rank_correlation(np.ones(4), np.arange(4)) == 1.0


def test_materialized_pair_contains_two_physical_frozen_arms(tmp_path) -> None:
    activation = np.zeros((8, 8), dtype=np.float32)
    activation[2:6, 2:6] = 1.0
    base = score_one_image(
        activation,
        _baseline_row(),
        _candidate_payload(),
        _stage_a_payload(),
    )
    scored = [
        type(base)(
            **{
                **base.__dict__,
                "image_id": f"image-{index}.png",
                "group_id": f"group-{index}",
                "tumor": int(index < 184),
            }
        )
        for index in range(371)
    ]
    freezes = _materialize_pair(tmp_path, scored)
    assert set(freezes) == {CONTROL_ARM, SEMANTIC_ARM}
    for arm in freezes:
        manifest = (tmp_path / arm / "prediction_manifest.csv").read_text()
        assert len(manifest.splitlines()) == 372
        assert (tmp_path / arm / "prediction_freeze.json").is_file()
        assert len(list((tmp_path / arm / "predictions").glob("*.npz"))) == 371
