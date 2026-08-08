from pathlib import Path


def test_train_score_wrapper_is_train_only_and_fail_closed() -> None:
    source = (
        Path(__file__).parents[1]
        / "project"
        / "run_x4_rich_gallery_train_score_kaggle.py"
    ).read_text(encoding="utf-8")
    assert '"--split", "train"' in source
    assert '"images": 2981' in source
    assert '"spatial_annotations_read": 0' in source
    assert '"test_images_read": 0' in source
    assert '"test_evaluated": False' in source
    assert "TRAIN_CANDIDATE_MANIFEST_SHA256" in source
    assert "G1_CHECKPOINT_SHA256" in source
    assert "load_supplies" in source
    assert '"--addition-namespace", "classifier448"' in source
    assert "reconstructed train candidate manifest differs" in source
