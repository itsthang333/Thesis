from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "build_mask_bag_selector_cache.py"


def test_cache_builder_source_is_annotation_and_test_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "build_segmentation_dataset",
        "mask_tensor",
        "annotation_name",
        'split="test"',
        "test_loader",
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_every_frozen_input_is_verified_before_descriptor_extraction() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    extraction = source.index("train_cache = build_descriptor_cache(")
    for required in (
        "baseline_freeze, baseline_rows = _verify_frozen_baseline(args, val_rows)",
        "train_candidates, train_candidate_audit = _audit_candidate_input(",
        "val_candidates, val_candidate_audit = _audit_candidate_input(",
        "model_snapshot = verify_model_snapshot(",
    ):
        assert source.index(required) < extraction


def test_cache_freeze_occurs_only_after_exact_baseline_reproduction() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reproduce = source.index("reproduction_audit = _compare_reproduction(")
    serialize = source.index("cache_rows = _serialize_cache_split(")
    freeze = source.index('freeze_path = args.output_dir / "selector_cache_freeze.json"')
    assert reproduce < serialize < freeze
    assert '"selected_candidate_index"' in source
    assert '"map_sha256"' in source
    assert '"validation_selected_indices_reproduced": 371' in source
    assert '"validation_map_hashes_reproduced": 371' in source


def test_cache_preserves_complete_candidate_provenance() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for required in (
        'payload["component_ids"]',
        'payload["prompt_modes"]',
        'payload["proposal_source_ids"]',
        "family_ids=families",
        "component_ids=components[kept]",
        "prompt_modes=modes[kept]",
        "proposal_source_ids=sources[kept]",
        "fallback_flags=fallback_flags[kept]",
    ):
        assert required in source


def test_cache_extracts_aligned_affinity_in_the_same_encoder_pass() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("include_affinity_features=True") == 2
    assert "affinity_features=np.asarray(record[\"affinity_features\"])" in source
    assert "record[\"flipped_affinity_features\"]" in source
    assert '"affinity_features_cached": True' in source
    assert '"affinity_feature_dim": 8 * len(SELECTED_HIDDEN_LAYERS)' in source


def test_train_masks_are_discarded_and_validation_masks_are_bitpacked() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert (
        "packed_masks=pack_candidate_masks(kept_masks) if split == \"val\" else None"
        in source
    )
    assert '"train_masks_discarded": True' in source
    assert '"validation_masks_bitpacked": True' in source


def test_runtime_is_resource_adaptive_and_geometry_remains_frozen() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "require_cuda_runtime()" in source
    assert "place_frozen_encoder(" in source
    assert '"encoder_data_parallel": runtime.encoder_data_parallel' in source
    assert 'all("T4" in name for name in device_names)' not in source
    assert "args.input_size != 448" in source
    assert "args.projection_dim != 128" in source
    assert "expected_maximum_candidates = 243 if args.rich_gallery_union else 81" in source
    assert "args.maximum_candidates != expected_maximum_candidates" in source
    assert '"maximum_candidates": args.maximum_candidates' in source
    assert '"rich_gallery_union": args.rich_gallery_union' in source
    assert "projection_sha256(projection) != baseline_freeze" in source
