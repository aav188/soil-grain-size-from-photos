import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soilgrain.gradation import gradation_indicators, soil_class
from soilgrain.metrics import SIEVES, emd, make_valid
from soilgrain.validation import blend, simplex_grid

SAND = [0, 2, 5, 12, 30, 55, 75, 90, 98, 100, 100]


def test_emd_zero_and_max():
    curve = np.linspace(0, 100, len(SIEVES))
    assert emd(curve, curve)[0] == 0
    assert emd(np.zeros(11), np.full(11, 100.0))[0] == pytest.approx(500)


def test_emd_triangle_inequality_for_mean():
    rng = np.random.default_rng(0)
    a, b, t = (make_valid(rng.uniform(0, 100, (20, 11))) for _ in range(3))
    assert (emd(t, (a + b) / 2) <= (emd(t, a) + emd(t, b)) / 2 + 1e-9).all()


def test_make_valid_is_monotone_and_bounded():
    v = make_valid(np.random.default_rng(1).uniform(-20, 120, (50, 11)))
    assert (np.diff(v, axis=1) >= 0).all() and v.min() >= 0 and (v[:, -1] == 100).all()


def test_indicators_and_class():
    ind = gradation_indicators(SIEVES, SAND)
    assert 0.063 < ind["D50"] < 2 and ind["Cu"] > 1
    assert soil_class(SIEVES, SAND) == "arena"


def test_monotone_head():
    torch = pytest.importorskip("torch")
    from soilgrain.model import MonotoneHead
    out = MonotoneHead(8)(torch.randn(4, 8))
    assert (out.diff(dim=1) >= 0).all() and torch.allclose(out[:, -1], torch.full((4,), 100.0))


def test_simplex_grid_sums_to_one():
    g = simplex_grid(3, 0.25)
    assert np.allclose(g.sum(axis=1), 1) and len(g) == 15


def test_blend_of_valid_curves_is_valid():
    a, b = make_valid(np.random.default_rng(2).uniform(0, 100, (2, 11)))
    assert (np.diff(blend([a, b], (0.3, 0.7)), axis=1) >= 0).all()
