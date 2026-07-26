import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load_generator():
    path = (
        Path(__file__).resolve().parents[1]
        / "project"
        / "generate_nominal_patch_memory_saliency.py"
    )
    project_dir = str(path.parent)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    spec = importlib.util.spec_from_file_location(
        "nominal_patch_generator_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_spatial_context_matches_expected_local_geometry() -> None:
    module = _load_generator()
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[..., 0] = 1.0
    query[0, 0] = [0.0, 1.0]
    context = np.zeros((1, 2, 2, 2), dtype=np.float32)
    context[..., 0] = 1.0
    context[0, 1, 1] = [0.0, 1.0]
    radius_zero = module.spatial_context_scores(
        query,
        context,
        radius=0,
        device=torch.device("cuda"),
    )
    radius_one = module.spatial_context_scores(
        query,
        context,
        radius=1,
        device=torch.device("cuda"),
    )
    assert np.isclose(radius_zero[0, 0], 1.0, atol=1e-5)
    assert np.isclose(radius_one[0, 0], 0.0, atol=1e-5)
    assert np.isfinite(radius_zero).all()
    assert np.isfinite(radius_one).all()
