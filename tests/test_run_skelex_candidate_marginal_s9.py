from __future__ import annotations

from argparse import Namespace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from project.models.bas_candidate_localizer import equal_rank_aggregate
from project.models.mask_bag_selector_cache import pack_candidate_masks
from project.models.skelex_candidate_marginal import finite_readout
import project.run_skelex_candidate_marginal_s9 as runner


def _args() -> Namespace:
    return Namespace(
        encoder_batch_size=runner.EXPECTED_ENCODER_BATCH_SIZE,
        train_batch_size=runner.EXPECTED_TRAIN_BATCH_SIZE,
        epochs=runner.EXPECTED_EPOCHS,
        learning_rate=runner.EXPECTED_LEARNING_RATE,
        weight_decay=runner.EXPECTED_WEIGHT_DECAY,
        maximum_candidates=runner.EXPECTED_MAXIMUM_CANDIDATES,
        seed=runner.EXPECTED_SEED,
    )


def test_frozen_recipe_rejects_runtime_sweep() -> None:
    runner._validate_recipe(_args())
    changed = _args()
    changed.epochs += 1
    with pytest.raises(ValueError, match="frozen one-shot recipe"):
        runner._validate_recipe(changed)


def test_project_supports_is_deterministic_and_keeps_fractional_mass() -> None:
    masks = np.zeros((2, 64, 32), dtype=np.uint8)
    masks[0, 20:28, 10:18] = 1
    masks[1, 32:50, 12:25] = 1
    projection = SimpleNamespace(padded_side=64, content_box=(16, 0, 48, 64))
    first = runner._project_supports(masks, projection=projection)
    second = runner._project_supports(masks, projection=projection)
    for observed, repeated in zip(first, second):
        np.testing.assert_array_equal(observed, repeated)
    inside, ring, content = first
    assert inside.shape == ring.shape == (2, runner.SKELEX_PATCHES)
    assert content.shape == (runner.SKELEX_PATCHES,)
    assert inside.dtype == ring.dtype == np.float16
    assert float(inside.sum()) > 0 and float(ring.sum()) > 0
    assert int(content.sum()) < runner.SKELEX_PATCHES


def test_collate_pads_only_candidate_axis() -> None:
    records = []
    for count, label in ((1, 0), (3, 1)):
        records.append(
            {
                "label": label,
                "tokens": np.zeros(
                    (runner.SKELEX_PATCHES, runner.SKELEX_TOKEN_DIM), dtype=np.float16
                ),
                "candidate_weights": np.ones(
                    (count, runner.SKELEX_PATCHES), dtype=np.float16
                ),
                "ring_weights": np.ones(
                    (count, runner.SKELEX_PATCHES), dtype=np.float16
                ),
                "content_valid": np.ones(runner.SKELEX_PATCHES, dtype=np.uint8),
                "candidate_indices": np.arange(count, dtype=np.int32),
            }
        )
    batch = runner.collate_feature_records(records)
    assert batch["tokens"].shape == (2, runner.SKELEX_PATCHES, runner.SKELEX_TOKEN_DIM)
    assert batch["candidate_weights"].shape == (2, 3, runner.SKELEX_PATCHES)
    assert batch["candidate_valid"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["tumor"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "first,second",
    [
        ([3.0], [9.0]),
        ([1.0, 1.0, 2.0], [3.0, 2.0, 2.0]),
        ([0.2, -1.0, 4.0, 0.2], [8.0, 7.0, 6.0, 5.0]),
    ],
)
def test_control_rank_is_byte_identical_to_accepted_implementation(
    first: list[float],
    second: list[float],
) -> None:
    observed = finite_readout(
        np.asarray(first, dtype=np.float32),
        np.asarray(second, dtype=np.float32),
        np.arange(len(first), dtype=np.float32),
    )["control"].astype(np.float32)
    expected = equal_rank_aggregate(
        (
            torch.tensor(first, dtype=torch.float32)[None],
            torch.tensor(second, dtype=torch.float32)[None],
        ),
        torch.ones((1, len(first)), dtype=torch.bool),
    )[0].numpy()
    np.testing.assert_array_equal(observed, expected)


def test_json_writer_is_lf_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    runner._write_json(path, {"validation_gt_read": False})
    assert b"\r" not in path.read_bytes()
    assert json.loads(path.read_text(encoding="utf-8"))["validation_gt_read"] is False
    with pytest.raises(FileExistsError):
        runner._write_json(path, {"validation_gt_read": True})


def test_output_manifest_declares_no_tta_and_physical_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                "candidate_logits": np.asarray([0.2, 0.8], dtype=np.float32),
                "bag_logit": 0.5,
                "bag_probability": 0.6,
            }
        )
    prediction_sha, score_sha = runner.write_validation_outputs(tmp_path, records, scored)
    assert len(prediction_sha) == len(score_sha) == 64
    with (tmp_path / "predictions" / "prediction_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["selected_candidate_index"] for row in rows] == ["9", "9"]
    assert {row["candidate_logit_recipe"] for row in rows} == {
        "within_image_equal_percentile_rank_no_tta"
    }
    assert all(Path(tmp_path / "predictions" / row["map_path"]).exists() for row in rows)
