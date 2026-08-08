from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from g4_e5_exact import (  # noqa: E402
    attach_exact_multimask_provenance,
    concatenate_payloads,
    first_unique_mask_indices,
    normalized_payload,
    project_payload_masks_to_grid,
    verify_post_dedup_reproduction,
)
from freeze_g4_e5_exact_choices import (  # noqa: E402
    _derive_post_dedup,
    _optional_g1_value,
    _select_indexed,
    _top_upstream_index,
    _validate_g1_alignment,
)


def payload(
    masks: np.ndarray,
    *,
    source: str = "layercam",
    components: list[int] | None = None,
    modes: list[str] | None = None,
    exact: bool = False,
) -> dict[str, np.ndarray]:
    count = len(masks)
    components = components or [0] * count
    modes = modes or ["point"] * count
    result = {
        "sam_masks": masks.astype(np.uint8),
        "sam_scores": np.linspace(0.1, 0.9, count, dtype=np.float32),
        "selection_scores": np.linspace(0.9, 0.1, count, dtype=np.float32),
        "classifier_causal_scores": np.zeros(count, dtype=np.float32),
        "component_ids": np.asarray(components, dtype=np.int32),
        "prompt_modes": np.asarray(modes),
        "proposal_source_ids": np.asarray([source] * count),
    }
    if exact:
        result.update(
            {
                "cam_levels": np.full(count, 90.0, dtype=np.float32),
                "prompt_ids": np.asarray(
                    [f"{source}|p90|c{component}|{mode}" for component, mode in zip(components, modes)]
                ),
                "multimask_indices": np.zeros(count, dtype=np.int16),
            }
        )
    return result


def test_exact_single_prompt_binds_three_multimasks() -> None:
    masks = np.zeros((3, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[1, 1, 1] = 1
    masks[2, 2, 2] = 1
    multi = payload(masks)
    single = payload(masks[:1], exact=True)
    bound = attach_exact_multimask_provenance(multi, single)
    assert bound["prompt_ids"].tolist() == ["layercam|p90|c0|point"] * 3
    assert bound["cam_levels"].tolist() == [90.0] * 3
    assert bound["multimask_indices"].tolist() == [0, 1, 2]


def test_prompt_group_cardinality_is_fail_closed() -> None:
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="one single mask and three multimasks"):
        attach_exact_multimask_provenance(payload(masks), payload(masks[:1], exact=True))


def test_namespace_applies_to_source_and_prompt_id() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    namespaced = normalized_payload(payload(masks, exact=True), namespace="classifier448")
    assert namespaced["proposal_source_ids"].tolist() == ["classifier448:layercam"]
    assert namespaced["prompt_ids"].tolist() == [
        "classifier448:layercam|p90|c0|point"
    ]


def test_raw_union_reproduces_post_dedup_exactly() -> None:
    masks = np.zeros((3, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[1] = masks[0]
    masks[2, 2, 2] = 1
    raw = payload(masks)
    post = {field: np.asarray(value)[[0, 2]] for field, value in raw.items()}
    kept = verify_post_dedup_reproduction(raw, post)
    assert kept.tolist() == [0, 2]
    assert first_unique_mask_indices(masks).tolist() == [0, 2]


def test_concatenation_preserves_exact_provenance() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    left = payload(masks, exact=True)
    right = normalized_payload(payload(masks, exact=True), namespace="classifier448")
    joined = concatenate_payloads(left, right)
    assert len(joined["sam_masks"]) == 2
    assert joined["prompt_ids"].tolist()[1].startswith("classifier448:")


def test_addition_grid_projection_replays_frozen_merge_geometry() -> None:
    source = np.zeros((1, 7, 7), dtype=np.uint8)
    source[0, 2:6, 3:7] = 1
    original = payload(source, source="classifier448:layercam", exact=True)
    projected = project_payload_masks_to_grid(original, (4, 4))

    y = np.floor(np.arange(4) * 7 / 4).astype(np.int64)
    x = np.floor(np.arange(4) * 7 / 4).astype(np.int64)
    expected = source[:, y[:, None], x[None, :]]
    assert projected["sam_masks"].shape == (1, 4, 4)
    assert np.array_equal(projected["sam_masks"], expected)
    assert np.array_equal(projected["prompt_ids"], original["prompt_ids"])
    assert np.array_equal(projected["selection_scores"], original["selection_scores"])


def test_projected_raw_union_reproduces_post_dedup() -> None:
    anchor_masks = np.zeros((1, 4, 4), dtype=np.uint8)
    anchor_masks[0, 1:3, 1:3] = 1
    addition_masks = np.zeros((2, 8, 8), dtype=np.uint8)
    addition_masks[0, 2:6, 2:6] = 1  # exact duplicate after 8 -> 4 replay
    addition_masks[1, 0:2, 0:2] = 1
    anchor = payload(anchor_masks)
    addition = project_payload_masks_to_grid(payload(addition_masks), (4, 4))
    raw = concatenate_payloads(anchor, addition)
    keep = first_unique_mask_indices(raw["sam_masks"])
    post = {field: np.asarray(raw[field])[keep] for field in raw}
    assert verify_post_dedup_reproduction(raw, post).tolist() == [0, 2]


def test_sparse_g1_indices_are_joined_by_candidate_identity() -> None:
    full_upstream = np.asarray([0.9, 0.7, 0.8, 0.1], dtype=np.float32)
    indices, logits, upstream = _validate_g1_alignment(
        np.asarray([0, 2, 3]),
        np.asarray([-1.0, 2.0, 0.0]),
        full_upstream[[0, 2, 3]],
        full_upstream,
        image_id="example.jpeg",
        label="pre-dedup",
    )
    selected, g1, selected_upstream, _fused = _select_indexed(
        indices,
        logits,
        upstream,
        np.asarray([1, 2, 3]),
    )
    assert selected == 2
    assert g1 == 2.0
    assert selected_upstream == pytest.approx(0.8)
    assert _optional_g1_value(indices, logits, 1) == ""
    assert _optional_g1_value(indices, logits, 2) == 2.0


def test_upstream_top1_uses_full_bank_not_sparse_g1_positions() -> None:
    full_upstream = np.asarray([0.1, 0.95, 0.8], dtype=np.float32)
    # Candidate 1 is intentionally absent from the hypothetical G1 bag.
    sparse_indices = np.asarray([0, 2], dtype=np.int64)
    assert _top_upstream_index(full_upstream) == 1
    assert sparse_indices[_top_upstream_index(full_upstream[sparse_indices])] == 2


def test_recovery_derives_exact_first_occurrence_post_bank() -> None:
    masks = np.zeros((3, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[1] = masks[0]
    masks[2, 0, 0] = 1
    raw = normalized_payload(payload(masks, exact=True))
    post, raw_first = _derive_post_dedup(raw)
    assert raw_first.tolist() == [0, 2]
    assert len(post["sam_masks"]) == 2
    assert np.array_equal(post["sam_masks"], raw["sam_masks"][[0, 2]])


def test_sparse_g1_alignment_rejects_misaligned_upstream() -> None:
    with pytest.raises(ValueError, match="upstream scores differ"):
        _validate_g1_alignment(
            np.asarray([0, 2]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.2, 0.3]),
            np.asarray([0.2, 0.4, 0.5]),
            image_id="example.jpeg",
            label="post-dedup",
        )
