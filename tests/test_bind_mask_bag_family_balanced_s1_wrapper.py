from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project/bind_mask_bag_family_balanced_s1_wrapper.py"


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("s1_binder", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s1_binder_has_exact_finite_replacement_surface() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'b"KERNEL_VERSION = 0"' in source
    assert 'b"LAUNCH_BINDING_READY = False"' in source
    assert "CHECKOUT_COMMIT" in source and "UNBOUND" in source
    assert '"replacement_count": len(replacements)' in source
    assert '"inverse_reconstruction_matches_template": True' in source


def test_s1_binder_refuses_existing_outputs(tmp_path: Path) -> None:
    module = _load_module()
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


def test_s1_binder_pins_claim_protocol_and_safety() -> None:
    module = _load_module()
    source = SOURCE.read_text(encoding="utf-8")
    assert module.CLAIM_COMMIT == "97db17c16938a8f842546076a26a52e58928b07b"
    assert module.PROTOCOL_SHA256 == "62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413"
    assert '"claim_commit": CLAIM_COMMIT' in source
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source
