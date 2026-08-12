from pathlib import Path

from btxrd_wsss.config import load_config


def test_reference_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "pipeline.yaml")
    assert config.hrnet.output_classes == 10
    assert config.hrnet.full_long_side == 1536
    assert config.sam.maximum_selected_candidates == 48
    assert config.sam.gallery_minimum_quotas["hrnet_tile"] == 16
