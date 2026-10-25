"""Carga de etiquetas e índice de fotos con su escala real (píxeles por mm)."""
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
TRAIN_DIR = RAW / "Training-All_Photos_updated" / "Training-All_Photos_updated"
TEST_DIR = RAW / "Test_All_Photos" / "Test_All_Photos"

# Prefijo del nombre de archivo -> fila de ppm_updated.csv (más largos primero)
PHONE_PREFIXES = [
    ("Motorola_Edge_60_fusion", "Motorola Edge 60 Fusion"),
    ("Motorola_Edge", "Motorola Edge"),
    ("Samsung_A52", "Samsung A52"),
    ("iPhone_16", "iPhone 16"),
    ("iPhone16", "iPhone 16"),
    ("iPhone14", "iPhone 14"),
]


def load_labels():
    """Curvas de entrenamiento: una fila por muestra, columnas = tamiz en mm."""
    df = pd.read_csv(RAW / "Training_labels_updated.csv", index_col="sample_id")
    df.columns = df.columns.astype(float)
    return df


def load_test_ids():
    return pd.read_csv(RAW / "sample_submission.csv")["sample_id"].tolist()


def _normalize(text):
    # Los archivos usan "Münster_BS6_9,0" y el CSV "Muenster_BS6_9_0"
    text = unicodedata.normalize("NFC", text)
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", ",": "_"}.items():
        text = text.replace(a, b)
    return text.lower()


def _match(name, candidates):
    hits = [c for c in candidates if _normalize(c) in _normalize(name)]
    if not hits:
        raise ValueError(f"No se reconoce el archivo {name}")
    return max(hits, key=len)


def photo_index():
    """Una fila por foto con móvil, muestra, tamaño real y ppm efectivo.

    Las fotos de entrenamiento vienen reducidas respecto a la resolución nativa del
    móvil, así que el ppm de la tabla se escala por (lado largo real / lado largo nativo).
    """
    ppm = pd.read_csv(RAW / "ppm_updated.csv").set_index("phone")
    splits = {"train": (TRAIN_DIR, load_labels().index.tolist()), "test": (TEST_DIR, load_test_ids())}
    rows = []
    for split, (folder, ids) in splits.items():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() != ".jpg":
                continue
            phone = next(p for prefix, p in PHONE_PREFIXES if f.name.startswith(prefix))
            with Image.open(f) as im:
                w, h = im.size
            nominal = ppm.loc[phone]
            scale = max(w, h) / max(nominal.width, nominal.height)
            rows.append({
                "path": f,
                "split": split,
                "sample_id": _match(f.name, ids),
                "phone": phone,
                "width": w,
                "height": h,
                "ppm_nominal": nominal.ppm,
                "ppm": nominal.ppm * scale,
            })
    df = pd.DataFrame(rows)
    df["fov_mm"] = df[["width", "height"]].max(axis=1) / df["ppm"]
    return df


# Las fotos de train (Motorola y Samsung) llegaron reducidas a ~4,6 px/mm y recomprimidas en
# JPEG de calidad ~75; las de test vienen del iPhone a resolución completa y calidad ~93
TRAIN_PIPELINE_PPM = 4.6
TRAIN_PIPELINE_QUALITY = 75


def emulate_train_pipeline(im, ppm):
    """Reproduce en una foto de alta resolución la reducción y recompresión que traen las de train."""
    if ppm <= TRAIN_PIPELINE_PPM * 1.2:
        return im, ppm
    factor = TRAIN_PIPELINE_PPM / ppm
    im = im.resize((round(im.width * factor), round(im.height * factor)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=TRAIN_PIPELINE_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), TRAIN_PIPELINE_PPM


def load_photo(path, target_ppm, ppm, emulate_train=False):
    """Abre la foto con la orientación EXIF aplicada y la reescala a target_ppm."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if emulate_train:
            im, ppm = emulate_train_pipeline(im, ppm)
        factor = target_ppm / ppm
        size = (round(im.width * factor), round(im.height * factor))
        return np.asarray(im.resize(size, Image.LANCZOS))
