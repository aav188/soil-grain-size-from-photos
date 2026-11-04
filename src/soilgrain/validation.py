"""Validación común a todos los modelos: LOSO y cambio de cámara, con las mismas funciones.

Cada modelo es una función `fit_predict(ctx, tr, ev) -> curva (11,)` que recibe los índices de
las fotos de entrenamiento y de evaluación. Los hiperparámetros se eligen siempre dentro de
`tr` (validación anidada), así que nada de la muestra evaluada influye en el modelo.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_labels, photo_index
from .features import TARGET_PPM
from .metrics import emd, make_valid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CAMERA_SENSITIVE = ("lum_mean", "lum_std", "lum_p10", "lum_p90", "chroma_r", "chroma_g")
ALPHAS = np.logspace(0, 5, 11)
# Energía por banda de escala a 1, 2 y 4 mm: las features más estables entre cámaras (r ≥ 0,92 entre
# Motorola y Samsung sobre las mismas muestras, desplazamiento ≤ 0,9 sd) y muy ligadas a la etiqueta.
BANDS = ("band_1mm", "band_2mm", "band_4mm")
SETUPS = ("LOSO", "Motorola → Samsung", "Samsung → Motorola")


@dataclass
class Context:
    photos: pd.DataFrame
    labels: pd.DataFrame
    tex: np.ndarray            # features de textura sin color ni brillo absolutos, una fila por foto
    tex_cols: list             # nombre de cada columna de `tex`
    patch_emb: list            # embeddings de la CNN por recorte, una lista por foto
    y: np.ndarray              # curva de cada foto (NaN en test), sin la última columna
    sid: np.ndarray

    @property
    def train_mask(self):
        return (self.photos["split"] == "train").values


def camera_family(phones):
    """Motorola, Samsung o iPhone: los dos modelos de cada marca comparten procesado de imagen."""
    return pd.Series(phones).str.split().str[0].values


def center_by_camera(X, phones):
    """Resta a cada familia de cámara la media de sus propias fotos (en test, las del iPhone sin etiqueta).

    Quita el desplazamiento de color/nitidez de cada móvil. Ojo: si el test tiene de verdad suelos
    distintos de media, también borra esa parte de la señal.
    """
    X = X.copy()
    fam = camera_family(phones)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(axis=0)
    return X


def load_context(center_tex=False):
    photos, labels = photo_index(), load_labels()
    tex = pd.read_csv(PROCESSED / f"features_ppm{TARGET_PPM}.csv").set_index("file")
    tex = tex.loc[photos["path"].map(lambda p: p.name)]
    cols = [c for c in tex.columns if c != "n_patches" and not c.startswith(CAMERA_SENSITIVE)]
    z = np.load(PROCESSED / f"emb_convnext_tiny_color_ppm{TARGET_PPM}.npz")
    emb = [z[str(i)] for i in range(len(photos))]
    sid = photos["sample_id"].values
    X = tex[cols].values
    if center_tex:
        X = center_by_camera(X, photos["phone"])
    return Context(photos, labels, X, cols, emb, labels.reindex(sid).values, sid)


def camera_reliability(ctx):
    """Cuánto mide cada feature el suelo y cuánto la cámara, con las muestras de train fotografiadas por ambos móviles.

    - r_entre_camaras: correlación, entre muestras, de la media con Motorola y la media con Samsung.
      Una feature que mide suelo ordena igual las muestras con las dos cámaras (r cerca de 1).
    - desplaz_cam_sd: diferencia media Samsung - Motorola, en desviaciones típicas entre suelos con una misma cámara.
    No usa etiquetas ni el test.
    """
    fam = camera_family(ctx.photos["phone"])
    df = pd.DataFrame(ctx.tex, columns=ctx.tex_cols)
    df["fam"], df["sid"] = fam, ctx.sid
    means = df[ctx.train_mask].groupby(["fam", "sid"])[ctx.tex_cols].mean()
    both = means.loc["Motorola"].index.intersection(means.loc["Samsung"].index)
    mot, sam = means.loc["Motorola"].loc[both], means.loc["Samsung"].loc[both]
    within_sd = np.sqrt((mot.var() + sam.var()) / 2)
    return pd.DataFrame({
        "r_entre_camaras": [np.corrcoef(mot[c], sam[c])[0, 1] for c in ctx.tex_cols],
        "desplaz_cam_sd": ((sam - mot).mean() / within_sd).values,
    }, index=ctx.tex_cols)


def _ridge(X, y, groups=None):
    """Ridge con alpha elegido por validación cruzada agrupada por muestra."""
    cv = GroupKFold(n_splits=5) if groups is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                         cv=cv, scoring="neg_mean_absolute_error")
    return model.fit(X, y, groups=groups) if groups is not None else model.fit(X, y)


def _mean_by_sample(X, sid):
    df = pd.DataFrame(X).groupby(sid).mean()
    return df.values, df.index.values


def ridge_photo(ctx, tr, ev):
    """Un Ridge entrenado por foto; la predicción de la muestra es la media de sus fotos."""
    m = _ridge(ctx.tex[tr], ctx.y[tr][:, :-1], groups=ctx.sid[tr])
    return np.r_[m.predict(ctx.tex[ev]).mean(axis=0), 100]


def ridge_aggregated(ctx, tr, ev, cols=None):
    """Un Ridge sobre las features medias por muestra (menos ruido de orientación y sombras).

    Con `cols`, solo usa esas columnas de textura.
    """
    T = ctx.tex if cols is None else ctx.tex[:, [ctx.tex_cols.index(c) for c in cols]]
    X, ids = _mean_by_sample(T[tr], ctx.sid[tr])
    y = ctx.labels.loc[ids].values[:, :-1]
    m = _ridge(X, y)
    return np.r_[m.predict(T[ev].mean(axis=0, keepdims=True))[0], 100]


def ridge_bands(ctx, tr, ev):
    """Ridge agregado solo con la energía por banda a 1, 2 y 4 mm (BANDS), las features más estables entre cámaras."""
    return ridge_aggregated(ctx, tr, ev, cols=BANDS)


def cnn_head(ctx, tr, ev, seeds=3, epochs=40):
    from .train_final import train_heads, predict_heads
    X = np.concatenate([ctx.patch_emb[i] for i in tr])
    sids = np.concatenate([[ctx.sid[i]] * len(ctx.patch_emb[i]) for i in tr])
    bundle = train_heads(X, ctx.labels.loc[sids].values, sids, seeds=seeds, epochs=epochs)
    return predict_heads(bundle, np.concatenate([ctx.patch_emb[i] for i in ev]))


def _emb_mean(ctx, idx):
    return np.stack([ctx.patch_emb[i].mean(axis=0) for i in idx])


def cnn_ridge(ctx, tr, ev):
    """Ridge agregado por muestra sobre la media de los embeddings de la CNN (sin cabeza entrenada)."""
    X, ids = _mean_by_sample(_emb_mean(ctx, tr), ctx.sid[tr])
    m = _ridge(X, ctx.labels.loc[ids].values[:, :-1])
    return np.r_[m.predict(_emb_mean(ctx, ev).mean(axis=0, keepdims=True))[0], 100]


def ridge_aggregated_iso(ctx, tr, ev):
    """Ridge agregado con proyección isotónica en vez de cummax (que solo corrige hacia finos)."""
    return make_valid(ridge_aggregated(ctx, tr, ev), method="isotonic")[0]


KNN_K = (3, 5, 8)


def _knn_predict(Xtr, Ytr, Xev, k):
    """Media de las curvas de los k vecinos más cercanos, con peso inverso a la distancia."""
    d = np.linalg.norm(Xev[:, None] - Xtr[None], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    w = 1 / (np.take_along_axis(d, nn, axis=1) + 1e-6)
    return (Ytr[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


def knn_curves(ctx, tr, ev, cols=None):
    """Curva media ponderada de las muestras de train más parecidas en textura (features medias por muestra).

    Una mezcla de curvas reales siempre es una curva válida y nunca sale del rango visto en train,
    así que no puede dar los 0 % o 100 % en bloque que da el Ridge al extrapolar. k se elige con
    leave-one-out dentro de `tr`.
    """
    T = ctx.tex if cols is None else ctx.tex[:, [ctx.tex_cols.index(c) for c in cols]]
    X, ids = _mean_by_sample(T[tr], ctx.sid[tr])
    Y = ctx.labels.loc[ids].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z, zev = (X - mu) / sd, (T[ev].mean(axis=0, keepdims=True) - mu) / sd
    loo = lambda k: np.mean([emd(Y[i], _knn_predict(np.delete(Z, i, 0), np.delete(Y, i, 0), Z[i:i + 1], k))[0]
                             for i in range(len(Z))])
    k = min((k for k in KNN_K if k < len(Z)), key=loo)
    return make_valid(_knn_predict(Z, Y, zev, k))[0]


MODELS = {"ridge_photo": ridge_photo, "ridge_aggregated": ridge_aggregated, "ridge_bands": ridge_bands,
          "ridge_aggregated_iso": ridge_aggregated_iso, "knn_curves": knn_curves,
          "cnn": cnn_head, "cnn_ridge": cnn_ridge}


def setup_masks(ctx, setup):
    n = len(ctx.photos)
    moto = ctx.photos["phone"].str.startswith("Motorola").values
    sams = (ctx.photos["phone"] == "Samsung A52").values
    return {"LOSO": (np.ones(n, bool),) * 2,
            "Motorola → Samsung": (moto, sams),
            "Samsung → Motorola": (sams, moto)}[setup]


def site_groups(ids, gap=10):
    """Agrupa ids consecutivos (H366, H367, H371…) como una misma obra: ids a menos de `gap` de distancia."""
    key = pd.Series({i: (i[0], int(i[1:])) for i in ids}).sort_values()
    group, prev, out = 0, None, {}
    for i, (letter, n) in key.items():
        if prev is not None and (letter != prev[0] or n - prev[1] > gap):
            group += 1
        out[i], prev = group, (letter, n)
    return pd.Series(out)


def out_of_fold(ctx, fit_predict, setup, groups=None):
    """Predicción de cada muestra de train entrenando sin ella (o sin todo su grupo, si se da `groups`)."""
    trf, evf = setup_masks(ctx, setup)
    rows = {}
    for s in ctx.labels.index:
        is_s = ctx.sid == s if groups is None else pd.Series(ctx.sid).map(groups).values == groups[s]
        tr = np.where(ctx.train_mask & ~is_s & trf)[0]
        ev = np.where(is_s & evf)[0]
        if len(ev):
            rows[s] = fit_predict(ctx, tr, ev)
    return pd.DataFrame(make_valid(np.vstack(list(rows.values()))), index=list(rows), columns=ctx.labels.columns)


def score(labels, pred):
    return pd.Series(emd(labels.loc[pred.index].values, pred.values), index=pred.index)


def simplex_grid(k, step=0.05):
    n = round(1 / step)
    def rec(left, parts):
        if parts == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in rec(left - i, parts - 1):
                yield (i, *rest)
    return np.array(list(rec(n, k))) / n


def blend(preds, w):
    return make_valid(sum(wi * p for wi, p in zip(w, preds)))
