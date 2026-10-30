"""Generación del modelo Ridge agregado y del ensemble calibrado de test.

Mejora sobre el baseline:
1. Ridge Aggregated Robust: agrega las features de textura a nivel de muestra
   promediando las fotos de cada muestra antes de ajustar Ridge, eliminando el ruido
   de orientación y sombras individuales y reduciendo el error inter-cámara.
2. Ensemble calibrado: 50 % Ridge Agregado + 35 % Ridge Robusto + 15 % CNN,
   evitando el sesgo hacia finos de la CNN sobre las fotos de iPhone y minimizando la EMD.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import photo_index, load_labels
from .features import TARGET_PPM
from .metrics import make_valid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
TEMPLATE = ROOT / "data" / "raw" / "sample_submission.csv"

CAMERA_SENSITIVE = ("lum_mean", "lum_std", "lum_p10", "lum_p90", "chroma_r", "chroma_g")


def compute_submissions():
    photos = photo_index()
    labels = load_labels()
    template = pd.read_csv(TEMPLATE)
    tex = pd.read_csv(PROCESSED / f"features_ppm{TARGET_PPM}.csv").set_index("file").loc[photos["path"].map(lambda p: p.name)]

    features = [c for c in tex.columns if c != "n_patches"]
    robust = [f for f in features if not f.startswith(CAMERA_SENSITIVE)]
    target_cols = labels.columns[:-1]

    is_test = (photos["split"] == "test").values
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]

    # 1. Ridge robust original (entrenado por foto)
    pipe_orig = make_pipeline(StandardScaler(), Ridge())
    cv_orig = GroupKFold(n_splits=5)
    model_orig = GridSearchCV(pipe_orig, {"ridge__alpha": np.logspace(-1, 4, 11)}, cv=cv_orig, scoring="neg_mean_absolute_error")
    model_orig.fit(tex.iloc[train_idx][robust].values, labels.loc[photos.iloc[train_idx]["sample_id"], target_cols].values,
                   groups=photos.iloc[train_idx]["sample_id"].values)
    pred_orig_raw = model_orig.predict(tex.iloc[test_idx][robust].values)
    pred_orig_raw = np.c_[pred_orig_raw, np.full(len(pred_orig_raw), 100.0)]
    df_orig = pd.DataFrame(pred_orig_raw, index=photos.iloc[test_idx]["sample_id"].values)
    pred_orig = pd.DataFrame(make_valid(df_orig.groupby(level=0).mean().loc[template["sample_id"]].values),
                             index=template["sample_id"], columns=template.columns[1:])

    # 2. Ridge aggregated (entrenado y predicho sobre medias por muestra)
    train_df = photos.iloc[train_idx].copy()
    for c in robust:
        train_df[c] = tex.iloc[train_idx][c].values
    tr_grouped = train_df.groupby("sample_id")[robust].mean()
    X_tr = tr_grouped.values
    y_tr = labels.loc[tr_grouped.index, target_cols].values

    pipe_agg = make_pipeline(StandardScaler(), Ridge())
    model_agg = GridSearchCV(pipe_agg, {"ridge__alpha": np.logspace(-1, 4, 11)}, cv=5, scoring="neg_mean_absolute_error")
    model_agg.fit(X_tr, y_tr)

    test_df = photos.iloc[test_idx].copy()
    for c in robust:
        test_df[c] = tex.iloc[test_idx][c].values
    te_grouped = test_df.groupby("sample_id")[robust].mean().loc[template["sample_id"]]

    pred_agg_raw = model_agg.predict(te_grouped.values)
    pred_agg_raw = np.c_[pred_agg_raw, np.full(len(pred_agg_raw), 100.0)]
    pred_agg = pd.DataFrame(make_valid(pred_agg_raw), index=template["sample_id"], columns=template.columns[1:])

    # Guardar Ridge Agregado
    SUBMISSIONS.mkdir(exist_ok=True)
    pred_agg.round(4).to_csv(SUBMISSIONS / "ridge_aggregated_robust.csv")

    # 3. Ensemble calibrado con CNN (si existe submission de CNN)
    cnn_file = SUBMISSIONS / "cnn.csv"
    if cnn_file.exists():
        cnn = pd.read_csv(cnn_file, index_col="sample_id")
        blend = 0.50 * pred_agg.values + 0.35 * pred_orig.values + 0.15 * cnn.loc[template["sample_id"]].values
        ensemble = pd.DataFrame(make_valid(blend), index=template["sample_id"], columns=template.columns[1:])
        ensemble.round(4).to_csv(SUBMISSIONS / "ensemble_mejorado.csv")
        print(f"Guardado {SUBMISSIONS / 'ensemble_mejorado.csv'}")

    print(f"Guardado {SUBMISSIONS / 'ridge_aggregated_robust.csv'}")


if __name__ == "__main__":
    compute_submissions()
