from __future__ import annotations

from argparse import Namespace
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from project.models.highres_candidate_pmil import HighResProposalMILOutput
from project.models.mask_bag_selector_cache import pack_candidate_masks
import project.run_highres_candidate_pmil_s10 as runner


def _args() -> Namespace:
    return Namespace(
        batch_size=runner.EXPECTED_BATCH_SIZE,
        epochs=runner.EXPECTED_EPOCHS,
        backbone_lr=runner.EXPECTED_BACKBONE_LR,
        head_lr=runner.EXPECTED_HEAD_LR,
        weight_decay=runner.EXPECTED_WEIGHT_DECAY,
        warmup_epochs=runner.EXPECTED_WARMUP_EPOCHS,
        top_dropout_fraction=runner.EXPECTED_TOP_DROPOUT_FRACTION,
        maximum_candidates=runner.EXPECTED_MAXIMUM_CANDIDATES,
        num_workers=0,
        seed=runner.EXPECTED_SEED,
        expected_pretrained_sha256=runner.EXPECTED_PRETRAINED_SHA256,
    )


def _output(classification: torch.Tensor, detection: torch.Tensor) -> HighResProposalMILOutput:
    batch, candidates = classification.shape
    masks = torch.zeros(batch, candidates, 4, 4)
    rings = torch.zeros_like(masks)
    for index in range(candidates):
        masks[:, index, index : index + 2, index : index + 2] = 1
        rings[:, index, :, :] = 1
        rings[:, index] -= masks[:, index]
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    return HighResProposalMILOutput(
        classification_logits=classification,
        detection_logits=detection,
        dense_logits=torch.zeros(batch, 4, 4, requires_grad=True),
        candidate_weights=masks,
        ring_weights=rings,
        candidate_area=masks.sum(dim=(-2, -1)),
        candidate_valid=valid,
    )


def test_frozen_recipe_rejects_runtime_sweep() -> None:
    runner._validate_recipe(_args())
    changed = _args()
    changed.epochs -= 1
    with pytest.raises(ValueError, match="frozen one-shot recipe"):
        runner._validate_recipe(changed)


def test_square_support_projection_keeps_fractional_content_geometry() -> None:
    masks = np.zeros((2, 64, 32), dtype=np.uint8)
    masks[0, 20:28, 10:18] = 1
    masks[1, 32:50, 12:25] = 1
    projection = SimpleNamespace(padded_side=64, content_box=(16, 0, 48, 64))
    supports, content = runner._project_square_supports(masks, projection=projection)
    assert supports.shape == (
        2,
        runner.EXPECTED_SUPPORT_SIZE,
        runner.EXPECTED_SUPPORT_SIZE,
    )
    assert content.shape == (
        runner.EXPECTED_SUPPORT_SIZE,
        runner.EXPECTED_SUPPORT_SIZE,
    )
    assert supports.dtype == content.dtype == torch.float16
    assert 0 < torch.count_nonzero(content) < content.numel()


def test_collate_pads_only_candidate_axis_and_keeps_identity() -> None:
    items = []
    for count, label in ((1, 0), (3, 1)):
        items.append(
            {
                "image_id": f"image-{count}",
                "group_id": f"group-{count}",
                "candidate_payload_sha256": f"sha-{count}",
                "label": label,
                "pixels": torch.zeros(3, runner.EXPECTED_IMAGE_SIZE, runner.EXPECTED_IMAGE_SIZE),
                "candidate_masks": torch.ones(
                    count,
                    runner.EXPECTED_SUPPORT_SIZE,
                    runner.EXPECTED_SUPPORT_SIZE,
                    dtype=torch.float16,
                ),
                "content_mask": torch.ones(
                    runner.EXPECTED_SUPPORT_SIZE,
                    runner.EXPECTED_SUPPORT_SIZE,
                    dtype=torch.float16,
                ),
                "candidate_indices": torch.arange(count) + 7,
            }
        )
    batch = runner.collate_s10(items)
    assert batch["candidate_masks"].shape[:2] == (2, 3)
    assert batch["candidate_valid"].tolist() == [
        [True, False, False],
        [True, True, True],
    ]
    assert batch["candidate_indices"].tolist() == [[7, -1, -1], [7, 8, 9]]


def test_top_dropout_only_changes_tumor_bags() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    retained = runner._tumor_dropout_valid(
        logits, valid, torch.tensor([0, 1]), fraction=0.5
    )
    assert retained[0].tolist() == [True, True, True]
    assert retained[1].tolist() == [False, True, True]


def test_empty_candidate_payload_reproduces_frozen_fallback(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npz"
    np.savez_compressed(
        path,
        sam_masks=np.empty((0, 8, 6), dtype=np.float32),
        prompt_map=np.zeros((8, 6), dtype=np.float32),
        sam_scores=np.empty(0, dtype=np.float32),
    )
    masks = runner._load_candidate_masks_without_rehash(path, maximum_candidates=81)
    assert masks.shape == (1, 8, 6)
    assert masks.sum() == 16


def test_two_view_objective_is_finite_and_backpropagates() -> None:
    classification = torch.tensor(
        [[0.1, -0.2, 0.4], [0.4, 0.3, -0.1]], requires_grad=True
    )
    detection = torch.tensor(
        [[0.2, 0.0, -0.1], [0.1, 0.5, -0.2]], requires_grad=True
    )
    original = _output(classification, detection)
    flipped = _output(classification + 0.1, detection - 0.1)
    output = runner.s10_training_objective(
        original, flipped, torch.tensor([0, 1]), dropout_fraction=0.2
    )
    assert set(output) == {"total", *runner.LOSS_WEIGHTS}
    assert torch.isfinite(output["total"])
    output["total"].backward()
    assert classification.grad is not None and torch.count_nonzero(classification.grad)
    assert detection.grad is not None and torch.count_nonzero(detection.grad)


def test_output_manifest_freezes_explicit_physical_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_VALIDATION", 2)
    records = []
    scored = []
    for index in range(2):
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[0, 1:3, 1:3] = 1
        masks[1, 4:7, 4:7] = 1
        records.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "group_id": f"g{index}",
                "label": index,
                "candidate_payload_sha256": f"payload-{index}",
                "candidate_indices": np.asarray([4, 9], dtype=np.int32),
                "fallback_flags": np.asarray([0, 0], dtype=np.uint8),
                "packed_masks": pack_candidate_masks(masks),
            }
        )
        scored.append(
            {
                "image_id": f"IMG{index:06d}.jpeg",
                "candidate_logits": np.asarray([0.0, 1.0], dtype=np.float32),
                "selected_local_index": 1,
                "bag_logit": 0.5,
                "bag_probability": 0.6,
            }
        )
    prediction_sha, score_sha = runner.write_validation_outputs(
        tmp_path, records, scored, recipe="frozen_test_recipe"
    )
    assert len(prediction_sha) == len(score_sha) == 64
    with (tmp_path / "predictions" / "prediction_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["selected_candidate_index"] for row in rows] == ["9", "9"]
    assert {row["candidate_logit_recipe"] for row in rows} == {"frozen_test_recipe"}


def test_highres_output_is_namedtuple_for_data_parallel_gather() -> None:
    output = _output(torch.zeros(1, 2), torch.zeros(1, 2))
    assert isinstance(output, tuple)
    assert output._fields[0] == "classification_logits"
