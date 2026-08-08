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
    verify_post_dedup_reproduction,
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

