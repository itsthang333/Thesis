from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"


def test_runner_has_image_only_surface_and_t4x2_encoder_parallelism() -> None:
    source = (PROJECT / "run_rad_dino_mask_bag_mil_probe.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "datasets.factory" not in imported
    assert "build_segmentation_dataset" not in source
    assert "Annotations" not in source
    assert 'torch.cuda.device_count() != 2' in source
    assert 'all("T4" in name for name in device_names)' in source
    assert "nn.DataParallel(encoder, device_ids=[0, 1]" in source
    assert 'default=81' in source
    assert 'args.maximum_candidates != 81' in source
    assert '"--rich-gallery-union"' in source
    assert "1 <= args.maximum_candidates <= 243" in source
    assert "logits = 0.5 * (original_logits + flipped_logits)" in source
    assert '"candidate_logit_tta": "mean_original_aligned_horizontal_flip"' in source
    assert "project_direct_resize_masks_to_square(" in source
    assert "descriptor_masks[..., ::-1].copy()" in source
    assert '"candidate_descriptor_geometry"' in source
    assert '"padding_exclusion"' in source
    assert "content_masks=torch.from_numpy(content_mask)[None]" in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_runner_calls_keyword_only_random_projection_contract() -> None:
    source = (PROJECT / "run_rad_dino_mask_bag_mil_probe.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_seeded_random_projection"
    ]
    assert len(calls) == 1
    assert calls[0].args == []
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "input_dim",
        "output_dim",
        "seed",
    }


def test_evaluator_verifies_every_prediction_input_before_gt_loader() -> None:
    scorer = (PROJECT / "score_final_rich_gallery.py").read_text(
        encoding="utf-8"
    )
    evaluator = (PROJECT / "evaluate_final_rich_gallery.py").read_text(
        encoding="utf-8"
    )
    assert "build_segmentation_dataset" not in scorer
    assert scorer.index("verify_frozen_test_config(") < scorer.index("_audit_candidate_input(")
    assert "candidate_manifest_sha256" in scorer
    assert "pseudo_manifest_sha256" in scorer
    gt_loader = evaluator.index("from datasets.factory import build_segmentation_dataset")
    assert evaluator.index("verify_frozen_test_config(") < gt_loader
    assert evaluator.index("candidate_choices_frozen_before_spatial_gt") < gt_loader


def test_mask_bag_gate_is_prediction_first_and_all_checks_required() -> None:
    source = (PROJECT / "evaluate_final_rich_gallery.py").read_text(
        encoding="utf-8"
    )
    assert 'choices=("val", "test")' in source
    assert 'args.split == "test" and args.expected_overall_dice is not None' in source
    assert '"candidate_choices_frozen_before_spatial_gt": True' in source
    assert '"candidate_choices_frozen_before_test_gt": args.split == "test"' in source
    assert '"test_evaluated": args.split == "test"' in source


def test_candidate_generator_defaults_remain_backward_compatible() -> None:
    source = (PROJECT / "generate_pseudo_masks.py").read_text(encoding="utf-8")
    assert 'choices=["tumor", "all"]' in source
    assert 'default="tumor"' in source
    assert '"--force-normal-candidate-gallery"' in source
    assert "is_tumor or args.force_normal_candidate_gallery" in source
    assert '"force_normal_candidate_gallery_semantics"' in source
    assert "final_mask = np.zeros_like(final_mask, dtype=bool)" in source
