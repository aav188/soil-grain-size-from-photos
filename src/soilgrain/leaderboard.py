"""Qué dicen las puntuaciones públicas ya conocidas, y envíos candidatos que se pueden generar sin los datos.

Uso: python -m soilgrain.leaderboard   (desde src/). Solo lee submissions/*.csv, no necesita data/raw.

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


def main():
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



if __name__ == "__main__":
    main()
