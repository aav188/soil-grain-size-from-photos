import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soilgrain.leaderboard import KNOWN_SCORES, SUBMISSIONS, direction_table, extrapolate, signed_area
from soilgrain.metrics import emd, make_valid


def test_extrapolate_is_valid_and_moves_away():
    rng = np.random.default_rng(0)
    ref, away = (make_valid(rng.uniform(0, 100, (5, 11))) for _ in range(2))
    out = extrapolate(ref, away, 0.5)
    assert (np.diff(out, axis=1) >= 0).all() and out.min() >= 0 and (out[:, -1] == 100).all()
    assert np.allclose(extrapolate(ref, away, 0.0), ref)


def test_signed_area_sign():
    fine, coarse = np.full((1, 11), 80.0), np.full((1, 11), 20.0)
    assert signed_area(fine, coarse)[0] > 0 and signed_area(coarse, fine)[0] < 0
    assert abs(signed_area(fine, coarse)[0]) == emd(fine, coarse)[0]


def test_direction_table_ratio_within_triangle_bounds():
    assert all((SUBMISSIONS / f"{n}.csv").exists() for n in KNOWN_SCORES)
    t = direction_table()
    assert ((t["ratio"] >= -1) & (t["ratio"] <= 1)).all()


def test_reconstructed_truths_reproduce_known_scores():
    from soilgrain.leaderboard import load, reconstruct_truths
    truths = reconstruct_truths(n_samples=3, seed=0)
    assert len(truths) > 0
    for T in truths:
        assert (np.diff(T, axis=1) >= -1e-9).all() and np.allclose(T[:, -1], 100)
        for name, lb in KNOWN_SCORES.items():
            assert abs(emd(T, load(name).loc[load("ensemble_agg").index].values).mean() - lb) <= 4.0
