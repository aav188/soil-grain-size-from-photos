"""Predicciones fuera de muestra de cada modelo y de cada perfil de ensemble, con todos los esquemas de validación.

Uso: python -m soilgrain.evaluate   (desde src/). Guarda data/processed/oof.pkl, reports/validacion.csv
y reports/fiabilidad_camaras.csv.
"""
import pickle
import sys
import time

import pandas as pd

from . import validation as v
from .metrics import emd

CACHE = v.PROCESSED / "oof.pkl"
SITE = "Por obra"


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
    from .ensemble import CENTERED, PROFILES
    ctx = v.load_context()
    oof = compute_oof(ctx)
    labels = ctx.labels
    cols = [SITE, *v.SETUPS]
    table = {m: {s: v.score(labels, oof[(m, s)]).mean() for s in cols} for m in v.MODELS}
    for name, weights in PROFILES.items():
        if name in CENTERED or len(weights) == 1:
            continue  # v3 necesita otras features; los perfiles de un modelo ya están en la tabla
        table[f"ensemble {name}"] = {s: emd(labels.loc[oof[(next(iter(weights)), s)].index].values,
                                            v.blend([oof[(m, s)].values for m in weights], list(weights.values()))).mean()
                                     for s in cols}
    df = pd.DataFrame(table).T[cols]
    df.round(1).to_csv(v.ROOT / "reports" / "validacion.csv", index_label="modelo")
    print(df.round(1).to_string())
    rel = v.camera_reliability(ctx).sort_values("r_entre_camaras", ascending=False)
    rel.round(2).to_csv(v.ROOT / "reports" / "fiabilidad_camaras.csv", index_label="feature")


if __name__ == "__main__":
    main()
