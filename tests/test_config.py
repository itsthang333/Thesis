from pathlib import Path

from btxrd_wsss.config import load_config


def test_reference_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "pipeline.yaml")
    assert config.hrnet.output_classes == 10
    assert config.hrnet.full_long_side == 1536
    assert config.sam.backend == "sam_med2d"
    assert config.sam.image_size == 256
    assert config.sam.encoder_adapter is True
    assert config.sam.maximum_selected_candidates == 48
    assert config.sam.gallery_minimum_quotas["hrnet_tile"] == 16
