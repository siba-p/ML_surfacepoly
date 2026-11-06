import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from mlsurfpoly.models import (  # noqa: E402
    build_backext_model,
    build_DNN_forward_model,
    build_HybridCNN_backward_model,
    build_HybridCNN_forward_model,
)


def test_forward_models_map_440_inputs_to_100_point_pmf():
    x = np.random.default_rng(0).integers(0, 2, (4, 440)).astype("float32")
    for build in (build_DNN_forward_model, build_HybridCNN_forward_model):
        assert build().predict(x, verbose=0).shape == (4, 100)


def test_backward_model_outputs_400_surface_sites():
    out = build_HybridCNN_backward_model()(np.zeros((2, 100), "float32"))
    assert out.shape[-1] == 400


def test_backext_model_outputs_pmf_and_delta_f():
    pmf, delta_f = build_backext_model()(np.zeros((2, 41), "float32"))
    assert pmf.shape == (2, 100) and delta_f.shape == (2, 1)
