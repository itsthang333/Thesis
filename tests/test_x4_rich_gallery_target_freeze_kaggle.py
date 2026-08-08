from pathlib import Path


def test_rich_gallery_target_wrapper_is_fail_closed_and_cpu_bounded() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "run_x4_rich_gallery_target_freeze_kaggle.py"
    ).read_text(encoding="utf-8")
    assert '"--arm", "rich_gallery"' in source
    assert '"--source-kind", "rich_gallery"' in source
    assert "validate_x4_target_bundle" in source
    assert '"images": 2981' in source
    assert '"tumor_images": 1488' in source
    assert '"normal_images": 1493' in source
    assert '"train_spatial_annotations_read": 0' in source
    assert '"outer_validation_annotations_read": 0' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source
    assert "torch" not in source
