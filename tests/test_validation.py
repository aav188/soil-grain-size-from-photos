"""Fiabilidad entre cámaras y Ridge sobre un subconjunto de features, con datos sintéticos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soilgrain.metrics import SIEVES
from soilgrain.validation import Context, camera_reliability, ridge_aggregated


def fake_context(n_samples=12, seed=0):
    """Cada muestra con 2 fotos de Motorola y 2 de Samsung.

    - suelo: depende solo de la muestra (lo que debería medir una buena feature).
    - camara: un desplazamiento fuerte por móvil más ruido, sin relación con el suelo.
    """
    rng = np.random.default_rng(seed)
    ids = [f"H{100 + i}" for i in range(n_samples)]
    soil = dict(zip(ids, rng.normal(size=n_samples)))
    rows, X = [], []
    for s in ids:
        for phone, offset in (("Motorola Edge", 0.0), ("Samsung A52", 3.0)):
            for _ in range(2):
                rows.append({"sample_id": s, "phone": phone, "split": "train"})
                X.append([soil[s] + rng.normal(scale=0.05), offset + rng.normal()])
    photos = pd.DataFrame(rows)
    # curva creciente cuyo nivel depende del suelo
    base = np.linspace(5, 95, len(SIEVES))
    labels = pd.DataFrame([np.clip(base + 10 * soil[s], 0, 100) for s in ids], index=ids, columns=SIEVES)
    labels[SIEVES[-1]] = 100.0
    sid = photos["sample_id"].values
    return Context(photos, labels, np.array(X), ["suelo", "camara"], [], labels.reindex(sid).values, sid)


def test_camera_reliability_separates_soil_from_camera():
    rel = camera_reliability(fake_context())
    assert rel.loc["suelo", "r_entre_camaras"] > 0.95
    assert abs(rel.loc["suelo", "desplaz_cam_sd"]) < 0.5
    assert abs(rel.loc["camara", "r_entre_camaras"]) < 0.8
    assert rel.loc["camara", "desplaz_cam_sd"] > 2


def test_ridge_aggregated_on_subset_uses_only_those_columns():
    ctx = fake_context(n_samples=20)
    held = ctx.sid == "H100"
    tr, ev = np.where(~held)[0], np.where(held)[0]
    pred = ridge_aggregated(ctx, tr, ev, cols=["suelo"])
    assert pred.shape == (len(SIEVES),) and pred[-1] == 100
    assert np.abs(pred - ctx.labels.loc["H100"].values).mean() < 5
    # cambiar la feature de cámara de la muestra evaluada no cambia nada si no se usa
    ctx.tex[ev, 1] += 100
    assert np.allclose(pred, ridge_aggregated(ctx, tr, ev, cols=["suelo"]))


def test_knn_curves_returns_valid_curve_within_train_range():
    from soilgrain.validation import knn_curves
    ctx = fake_context()
    tr = np.where(ctx.sid != "H100")[0]
    ev = np.where(ctx.sid == "H100")[0]
    pred = knn_curves(ctx, tr, ev)
    Y = ctx.labels.drop("H100").values
    assert (np.diff(pred) >= 0).all() and pred[-1] == 100
    assert (pred >= Y.min(0) - 1e-9).all() and (pred <= Y.max(0) + 1e-9).all()


def test_ridge_aggregated_iso_is_valid():
    from soilgrain.validation import ridge_aggregated_iso
    ctx = fake_context()
    tr, ev = np.where(ctx.sid != "H101")[0], np.where(ctx.sid == "H101")[0]
    pred = ridge_aggregated_iso(ctx, tr, ev)
    assert (np.diff(pred) >= 0).all() and pred.min() >= 0 and pred[-1] == 100
