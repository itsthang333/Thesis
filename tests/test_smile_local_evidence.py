from __future__ import annotations

import numpy as np
import torch

from project.models.smile_local_evidence import (
    SMILELocalEvidence,
    average_percentile_rank,
    matched_normal_counterparts,
    score_gallery_candidates_from_evidence,
    smile_image_label_objective,
    soft_intra_class_discrimination,
    target_subtype_margin,
)


def test_reference_matching_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    query = torch.randn(1, 8, 6, 6)
    references = torch.randn(1, 4, 8, 3, 3)
    query_valid = torch.ones(1, 6, 6, dtype=torch.bool)
    reference_valid = torch.ones(1, 4, 3, 3, dtype=torch.bool)
    first = matched_normal_counterparts(
        query, references, query_valid, reference_valid, query_chunk_size=7
    )
    order = torch.tensor([2, 0, 3, 1])
    second = matched_normal_counterparts(
        query,
        references[:, order],
        query_valid,
        reference_valid[:, order],
        query_chunk_size=7,
    )
    assert torch.allclose(first[0], second[0], atol=1e-6)
    assert torch.allclose(first[1], second[1], atol=1e-6)


def test_subtype_margin_uses_target_channel_without_input_leakage() -> None:
    logits = torch.zeros(2, 10, 2, 2)
    logits[0, 3] = 2.0
    logits[1, 7] = 4.0
    logits[:, 0] = 0.5
    margin = target_subtype_margin(logits, torch.tensor([3, 7]))
    assert torch.allclose(margin[0], torch.full((1, 2, 2), 1.5))
    assert torch.allclose(margin[1], torch.full((1, 2, 2), 3.5))


def test_soft_intra_class_discrimination_is_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 10, 8, 8, requires_grad=True)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    foreground, background = soft_intra_class_discrimination(
        logits,
        valid,
        torch.tensor([1.0, 0.0]),
        torch.tensor([4, 0]),
    )
    loss = foreground + background
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_image_label_objective_has_all_components_and_gradients() -> None:
    torch.manual_seed(11)
    binary = torch.randn(2, 1, 8, 8, requires_grad=True)
    subtype = torch.randn(2, 10, 8, 8, requires_grad=True)
    output = {
        "binary_image_logits": torch.tensor([0.4, -0.3], requires_grad=True),
        "subtype_image_logits": torch.randn(2, 10, requires_grad=True),
        "binary_evidence_logits": binary,
        "subtype_local_logits": subtype,
        "evidence_valid": torch.ones(2, 1, 8, 8),
    }
    losses = smile_image_label_objective(
        output, torch.tensor([1.0, 0.0]), torch.tensor([2, 0])
    )
    assert set(losses) == {
        "total",
        "binary_bag",
        "subtype_bag",
        "normal_binary_dense",
        "normal_subtype_dense",
        "foreground_subtype",
        "background_normal",
        "binary_subtype_alignment",
    }
    losses["total"].backward()
    assert binary.grad is not None and float(binary.grad.abs().sum()) > 0
    assert subtype.grad is not None and float(subtype.grad.abs().sum()) > 0


def test_control_model_ignores_reference_arguments() -> None:
    model = SMILELocalEvidence(arm="control", fpn_channels=8, dropout=0.0).eval()
    query = torch.rand(1, 3, 64, 64)
    valid = torch.ones(1, 1, 64, 64)
    with torch.inference_mode():
        first = model(query, valid, conditioning_subtype=torch.tensor([1]))
        second = model(
            query,
            valid,
            torch.rand(1, 4, 3, 64, 64),
            torch.ones(1, 4, 1, 64, 64),
            conditioning_subtype=torch.tensor([1]),
        )
    for key in first:
        assert torch.equal(first[key], second[key])
    assert first["binary_evidence_logits"].shape[-2:] == (16, 16)
    assert first["subtype_local_logits"].shape == (1, 10, 16, 16)


def test_candidate_readout_separates_identity_and_extent() -> None:
    evidence = np.full((32, 32), -5.0, dtype=np.float32)
    evidence[12:20, 13:19] = 5.0
    exact = np.zeros_like(evidence, dtype=bool)
    exact[12:20, 13:19] = True
    too_large = np.zeros_like(exact)
    too_large[5:28, 5:28] = True
    wrong = np.zeros_like(exact)
    wrong[2:8, 23:29] = True
    candidates = np.stack((wrong, too_large, exact))
    result = score_gallery_candidates_from_evidence(
        evidence,
        candidates,
        np.asarray([0.9, 0.2, 0.1]),
        np.asarray([0.8, 0.3, 0.2]),
    )
    assert int(np.argmax(result["identity"])) in {1, 2}
    assert int(np.argmax(result["extent"])) == 2
    residual = result["identity_extent"] - result["baseline"]
    assert int(np.argmax(residual)) == 2
    assert residual[2] > residual[1] > residual[0]
    assert np.isfinite(np.concatenate(list(result.values()))).all()


def test_zero_residual_baseline_and_average_ties() -> None:
    values = np.asarray([3.0, 1.0, 3.0, 2.0])
    ranks = average_percentile_rank(values)
    assert np.allclose(ranks, np.asarray([2.5, 0.0, 2.5, 1.0]) / 3.0)
    candidates = np.zeros((3, 8, 8), dtype=bool)
    candidates[0, :2, :2] = True
    candidates[1, 3:5, 3:5] = True
    candidates[2, 6:, 6:] = True
    g1 = np.asarray([0.3, 0.1, 0.2])
    upstream = np.asarray([0.1, 0.3, 0.2])
    result = score_gallery_candidates_from_evidence(
        np.zeros((8, 8), dtype=np.float32), candidates, g1, upstream
    )
    expected = 0.5 * (average_percentile_rank(g1) + average_percentile_rank(upstream))
    assert np.array_equal(result["baseline"], expected)
