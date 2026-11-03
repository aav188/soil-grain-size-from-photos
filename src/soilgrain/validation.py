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
SETUPS = ("LOSO", "Motorola → Samsung", "Samsung → Motorola")


@dataclass
class Context:
    photos: pd.DataFrame
    labels: pd.DataFrame
    tex: np.ndarray            # features de textura robustas a la cámara, una fila por foto
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
    return Context(photos, labels, X, emb, labels.reindex(sid).values, sid)


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


def ridge_aggregated(ctx, tr, ev):
    """Un Ridge sobre las features medias por muestra (menos ruido de orientación y sombras)."""
    X, ids = _mean_by_sample(ctx.tex[tr], ctx.sid[tr])
    y = ctx.labels.loc[ids].values[:, :-1]
    m = _ridge(X, y)
    return np.r_[m.predict(ctx.tex[ev].mean(axis=0, keepdims=True))[0], 100]


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


def tex_cnn_ridge(ctx, tr, ev):
    """Ridge agregado sobre textura y embeddings concatenados."""
    f = lambda idx: np.hstack([ctx.tex[idx], _emb_mean(ctx, idx)])
    X, ids = _mean_by_sample(f(tr), ctx.sid[tr])
    m = _ridge(X, ctx.labels.loc[ids].values[:, :-1])
    return np.r_[m.predict(f(ev).mean(axis=0, keepdims=True))[0], 100]


MODELS = {"ridge_photo": ridge_photo, "ridge_aggregated": ridge_aggregated, "cnn": cnn_head, "cnn_ridge": cnn_ridge}


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


def fit_weights(labels, preds_by_setup, exclude, grid):
    """Pesos que minimizan la EMD media entre cámaras, sin usar la muestra `exclude`.

    Cada esquema de cambio de cámara cubre solo las muestras fotografiadas con ambos móviles.
    """
    cross = [s for s in preds_by_setup if s != "LOSO"]
    data = []
    for s in cross:
        ids = preds_by_setup[s][0].index.drop(exclude, errors="ignore")
        data.append((labels.loc[ids].values, [p.loc[ids].values for p in preds_by_setup[s]]))
    vals = [np.mean([emd(y, blend(ps, w)).mean() for y, ps in data]) for w in grid]
    return grid[int(np.argmin(vals))]


def nested_blend(labels, preds_by_setup, grid):
    """Los pesos de cada muestra se eligen sin ella: la estimación honesta de un ensemble ajustado."""
    out = {}
    for setup, preds in preds_by_setup.items():
        rows = []
        for s in preds[0].index:
            w = fit_weights(labels, preds_by_setup, s, grid)
            rows.append(blend([p.loc[[s]].values for p in preds], w)[0])
        out[setup] = pd.DataFrame(np.vstack(rows), index=preds[0].index, columns=labels.columns)
    return out
