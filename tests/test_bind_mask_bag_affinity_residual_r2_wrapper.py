from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project/bind_mask_bag_affinity_residual_r2_wrapper.py"


def test_r2_binder_has_exact_finite_replacement_surface() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'b"KERNEL_VERSION = 0"' in source
    assert 'b"LAUNCH_BINDING_READY = False"' in source
    assert "CHECKOUT_COMMIT" in source and "UNBOUND" in source
    assert '"replacement_count": len(replacements)' in source
    assert '"inverse_reconstruction_matches_template": True' in source


def test_r2_binder_refuses_existing_outputs(tmp_path: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("r2_binder", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "bound.py"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.bind(
            tmp_path / "template.py",
            output,
            tmp_path / "binding.json",
            repository_root=ROOT,
            checkout_commit="a" * 40,
            kernel_version=1,
        )


def test_r2_binder_keeps_safety_fields_in_launch_binding() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
