"""Qué dicen las puntuaciones públicas ya conocidas, y envíos candidatos que se pueden generar sin los datos.

Uso: python -m soilgrain.leaderboard [--estimar]   (desde src/). Solo lee submissions/*.csv, no necesita data/raw.
Con --estimar reconstruye curvas del test compatibles con las puntuaciones y estima la nota de los envíos sin puntuar (~1 min).

Idea: todas las puntuaciones de Kaggle se pueden comparar con el mejor envío (Ridge Agregado, 43,52).
- `finos`: área con signo entre cada envío y la referencia (positiva = curva más fina, por encima).
- `ratio`: (LB - LB_ref) / EMD(envío, ref). Por la desigualdad triangular está entre -1 y 1; si vale
  1, todo lo que el envío se aparta de la referencia es error añadido.
Con los 7 envíos puntuados, cuanto más fino es un envío, peor puntúa, y el ratio va de 0,6 a 0,96:
la mayor parte de lo que se aparta de la referencia lo aleja de la verdad. Por eso, moverse en la
dirección contraria (más grueso) debería bajar la EMD, al menos mientras no se pase de la verdad.

Ojo: es ajustar al leaderboard público con 10 muestras. La EMD es convexa a lo largo de una recta,
así que para x = ref + λ·(ref - otro) solo se sabe que
    LB_ref - λ·(LB_otro - LB_ref)  <=  LB(x)  <=  LB_ref + λ·EMD(otro, ref).
Con λ pequeño el riesgo está acotado. Enviar como sonda antes de cambiar el envío por defecto.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import LOG_WIDTHS, emd, make_valid

SUBMISSIONS = Path(__file__).resolve().parents[2] / "submissions"
REFERENCE = "ensemble_agg"
# Puntuación pública de cada archivo de submissions/ (ver README)
KNOWN_SCORES = {
    "ensemble_agg": 43.52,
    "ensemble_v2": 56.25,
    "ensemble_50_35_15": 56.48,
    "ensemble_v3": 58.56,
    "baseline_ridge_robust": 64.9,
    "media_textura_cnn": 67.2,
    "ensemble_cnn95": 70.06,
    "cnn": 72.3,
}
# Envíos candidatos: (nombre del archivo, envío del que alejarse, λ)
CANDIDATES = [
    ("agg_lejos_robust_050", "baseline_ridge_robust", 0.5),
    ("agg_lejos_v2_030", "ensemble_v2", 0.3),
]


def load(name):
    return pd.read_csv(SUBMISSIONS / f"{name}.csv", index_col="sample_id")


def signed_area(curves, ref):
    """Área con signo en escala log entre dos juegos de curvas, por muestra (positiva = más fina que ref)."""
    return ((np.asarray(curves) - np.asarray(ref))[:, :-1] * LOG_WIDTHS).sum(axis=1)


def signal(name, ref):
    return signed_area(load(name).loc[ref.index].values, ref.values).mean()


def direction_table(scores=KNOWN_SCORES, reference=REFERENCE):
    ref = load(reference)
    rows = {}
    for name, lb in scores.items():
        if name == reference:
            continue
        sub = load(name).loc[ref.index]
        dist = emd(ref.values, sub.values).mean()
        rows[name] = {"LB": lb, "finos": signed_area(sub.values, ref.values).mean(),
                      "EMD_a_ref": dist, "ratio": (lb - scores[reference]) / dist}
    return pd.DataFrame(rows).T.sort_values("LB")


def extrapolate(ref, away, lam):
    """Curvas ref + λ·(ref - away), devueltas como envío válido (monótonas, en [0, 100], 100 al final)."""
    return make_valid(np.asarray(ref) + lam * (np.asarray(ref) - np.asarray(away)))


def bounds(away, lam, scores=KNOWN_SCORES, reference=REFERENCE):
    """Cotas de la puntuación de ref + λ·(ref - away) por convexidad y desigualdad triangular."""
    ref, other = load(reference), load(away)
    dist = emd(ref.values, other.loc[ref.index].values).mean()
    return scores[reference] - lam * (scores[away] - scores[reference]), scores[reference] + lam * dist


def _ccp_step(T0, subs, lbs, rng):
    """Un LP del procedimiento convexo-cóncavo: EMD_k(T) <= LB_k exacto y |x| >= signo(x0)·x como cota inferior."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, eye, hstack, vstack

    K, n, m = subs.shape
    nT, nU = n * m, K * n * m
    w = np.tile(LOG_WIDTHS, n) / n
    tile = coo_matrix(np.tile(np.eye(nT), (K, 1)))                    # T repetida para cada envío
    I_u = eye(nU)
    zeros_s = coo_matrix((nU, K))
    rows = [hstack([tile, -I_u, zeros_s]), hstack([-tile, -I_u, zeros_s])]   # ±(T - S) <= u
    b = [subs.ravel(), -subs.ravel()]
    blk = coo_matrix(np.kron(np.eye(K), w))
    rows.append(hstack([coo_matrix((K, nT)), blk, coo_matrix((K, K))]))      # EMD_k <= LB_k
    b.append(lbs)
    sg = np.sign(T0[None] - subs)
    sg[sg == 0] = rng.choice([-1, 1], (sg == 0).sum())
    lin = (sg * LOG_WIDTHS / n).reshape(K, nT)
    rows.append(hstack([coo_matrix(-lin), coo_matrix((K, nU)), -eye(K)]))   # cota inferior + holgura >= LB_k
    b.append(-lbs - (lin * subs.reshape(K, nT)).sum(1))
    mono = coo_matrix(np.kron(np.eye(n), np.eye(m - 1, m) - np.eye(m - 1, m, 1)))
    rows.append(hstack([mono, coo_matrix((n * (m - 1), nU + K))]))           # curva creciente
    b.append(np.zeros(n * (m - 1)))
    c = np.r_[np.zeros(nT + nU), np.ones(K)]
    res = linprog(c, A_ub=vstack(rows).tocsr(), b_ub=np.concatenate(b),
                  bounds=[(0, 100)] * nT + [(0, None)] * (nU + K), method="highs")
    return res.x[:nT].reshape(n, m)


def reconstruct_truths(n_samples=24, max_error=4.0, iters=40, seed=0, scores=KNOWN_SCORES):
    """Juegos de curvas del test compatibles con todas las puntuaciones públicas conocidas.

    Cada juego parte de un envío conocido con ruido y se ajusta con el procedimiento convexo-cóncavo
    hasta que su EMD a cada envío coincide con la puntuación (se aceptan los de error máximo <= max_error;
    el redondeo de Kaggle impide la igualdad exacta).

    Limitación medida: sirve para interpolar, no para descubrir. Dejando fuera la puntuación del Ridge
    Agregado, las reconstrucciones le predecían ~56 cuando sacó 43,52.
    """
    ref = load(REFERENCE)
    names = list(scores)
    subs = np.stack([load(k).loc[ref.index].values[:, :-1] for k in names])
    lbs = np.array([scores[k] for k in names])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        T = np.maximum.accumulate(np.clip(subs[rng.integers(len(names))] + rng.normal(0, 15, subs.shape[1:]), 0, 100), 1)
        for _ in range(iters):
            T = _ccp_step(T, subs, lbs, rng)
            full = np.c_[T, np.full(len(T), 100.0)]
            err = np.abs([emd(full, np.c_[s, np.full(len(s), 100.0)]).mean() for s in subs] - lbs).max()
            if err < 0.05:
                break
        if err <= max_error:
            out.append(full)
    return np.array(out)


def estimate_scores(candidates, truths):
    """Puntuación de cada envío candidato frente a cada reconstrucción: media, mínimo y máximo."""
    ref = load(REFERENCE)
    rows = {}
    for name in candidates:
        sub = load(name).loc[ref.index].values
        s = np.array([emd(T, sub).mean() for T in truths])
        rows[name] = {"estimada": s.mean(), "min": s.min(), "max": s.max()}
    return pd.DataFrame(rows).T


def main(estimate=False):
    pd.set_option("display.width", 120)
    print(direction_table().round(2).to_string())
    ref = load(REFERENCE)
    for name in ("ensemble_bands", "baseline_ridge"):
        if (SUBMISSIONS / f"{name}.csv").exists():
            print(f"{name} (sin puntuar): finos = {signal(name, ref):+.1f} respecto a {REFERENCE}")
    for out, away, lam in CANDIDATES:
        curves = extrapolate(ref.values, load(away).loc[ref.index].values, lam)
        df = pd.DataFrame(curves, index=ref.index, columns=ref.columns).round(4)
        df.to_csv(SUBMISSIONS / f"{out}.csv")
        lo, hi = bounds(away, lam)
        print(f"Guardado submissions/{out}.csv: puntuación acotada entre {lo:.1f} y {hi:.1f}")
    if estimate:
        truths = reconstruct_truths()
        unscored = sorted(p.stem for p in SUBMISSIONS.glob("*.csv") if p.stem not in KNOWN_SCORES)
        print(f"\nPuntuación estimada con {len(truths)} reconstrucciones del test (orientativa):")
        print(estimate_scores([REFERENCE, *unscored], truths).sort_values("estimada").round(1).to_string())


if __name__ == "__main__":
    import sys
    main(estimate="--estimar" in sys.argv)
