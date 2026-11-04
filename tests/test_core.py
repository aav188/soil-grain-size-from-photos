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


def test_center_by_camera_zero_mean_per_family():
    from soilgrain.validation import center_by_camera
    X = np.arange(12, dtype=float).reshape(6, 2)
    phones = ["Motorola Edge", "Motorola Edge 60 Fusion", "Samsung A52", "Samsung A52", "iPhone 16", "iPhone 14"]
    Xc = center_by_camera(X, phones)
    for rows in ([0, 1], [2, 3], [4, 5]):
        assert np.allclose(Xc[rows].mean(axis=0), 0)


def test_isotonic_projection_is_monotone_and_not_biased_to_fines():
    curve = np.array([[10, 30, 20, 40, 60, 50, 70, 80, 90, 95, 100.0]])
    iso, cum = make_valid(curve, method="isotonic"), make_valid(curve)
    assert (np.diff(iso, axis=1) >= 0).all() and iso[0, -1] == 100
    assert np.allclose(iso[0, 1:3], 25) and np.allclose(iso[0, 4:6], 55)  # media de cada par invertido
    assert iso.sum() < cum.sum()  # cummax solo sube; la isotónica no sesga hacia finos
    ok = make_valid(np.sort(np.random.default_rng(3).uniform(0, 100, (5, 11)), axis=1))
    assert np.allclose(make_valid(ok, method="isotonic"), ok)
