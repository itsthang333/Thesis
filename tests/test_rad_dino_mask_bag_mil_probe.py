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
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_evaluator_verifies_every_prediction_input_before_gt_loader() -> None:
    source = (PROJECT / "evaluate_rad_dino_mask_bag_mil_probe.py").read_text(
        encoding="utf-8"
    )
    gt_loader = source.index("from datasets.factory import build_segmentation_dataset")
    assert source.index("_load_and_verify_predictions(args, val_rows)") < gt_loader
    assert source.index("validate_candidate_diagnostics_manifest(") < gt_loader
    assert source.index("sha256_file(args.baseline_per_image)") < gt_loader
    assert 'parser.add_argument("--dataset-root"' in source
    assert "choices=[\"test\"" not in source
    assert '"bootstrap_replicates": args.bootstrap_replicates' in source
    assert '"complete_misses_included": True' in source


def test_mask_bag_gate_is_prediction_first_and_all_checks_required() -> None:
    source = (PROJECT / "evaluate_rad_dino_mask_bag_mil_probe.py").read_text(
        encoding="utf-8"
    )
    assert '"overall": 0.250' in source
    assert '"small": 0.130' in source
    assert '"medium": 0.370' in source
    assert '"large": 0.380' in source
    assert '"all_checks_required": True' in source
    assert "overall_ci95_low_above_zero" in source
    assert "no_subgroup_mean_decrease" in source
    assert "no_complete_miss_increase" in source
    assert "authorize only a separately predeclared pseudo-mask consumer" in source


def test_candidate_generator_defaults_remain_backward_compatible() -> None:
    source = (PROJECT / "generate_pseudo_masks.py").read_text(encoding="utf-8")
    assert 'choices=["tumor", "all"]' in source
    assert 'default="tumor"' in source
    assert '"--force-normal-candidate-gallery"' in source
    assert "is_tumor or args.force_normal_candidate_gallery" in source
    assert '"force_normal_candidate_gallery_semantics"' in source
    assert "final_mask = np.zeros_like(final_mask, dtype=bool)" in source
