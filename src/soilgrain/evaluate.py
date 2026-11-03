"""Calcula las predicciones fuera de muestra de cada modelo y valida el ensemble con pesos anidados.

Uso: python -m soilgrain.evaluate   (desde src/). Guarda data/processed/oof.pkl y reports/validacion.csv
"""
import pickle
import sys
import time

import pandas as pd

from . import validation as v
from .metrics import emd

CACHE = v.PROCESSED / "oof.pkl"
SITE = "Por obra"
MEMBERS = ("ridge_aggregated", "ridge_photo", "cnn", "cnn_ridge")


def compute_oof(ctx, force=False):
    oof = pickle.loads(CACHE.read_bytes()) if CACHE.exists() and not force else {}
    groups = v.site_groups(ctx.labels.index)
    for name, fp in v.MODELS.items():
        for setup in (*v.SETUPS, SITE):
            if (name, setup) not in oof:
                t = time.time()
                oof[(name, setup)] = (v.out_of_fold(ctx, fp, "LOSO", groups=groups) if setup == SITE
                                      else v.out_of_fold(ctx, fp, setup))
                print(f"{name:18s} {setup:20s} {time.time() - t:5.0f}s", file=sys.stderr, flush=True)
                CACHE.write_bytes(pickle.dumps(oof))
    return oof


def main():
    from .ensemble import PROFILES
    ctx = v.load_context()
    oof = compute_oof(ctx)
    labels = ctx.labels
    cols = [SITE, *v.SETUPS]
    table = {m: {s: v.score(labels, oof[(m, s)]).mean() for s in cols} for m in MEMBERS}
    for name, w in PROFILES.items():
        table[f"ensemble {name} {w}"] = {
            s: emd(labels.loc[oof[(MEMBERS[0], s)].index].values,
                   v.blend([oof[(m, s)].values for m in MEMBERS], w)).mean() for s in cols}
    df = pd.DataFrame(table).T[cols]
    out = v.ROOT / "reports" / "validacion.csv"
    df.round(1).to_csv(out, index_label="modelo")
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
