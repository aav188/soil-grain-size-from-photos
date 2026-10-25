"""Demo: sube una foto de la superficie de un suelo y obtén su curva granulométrica.

Ejecutar desde la raíz del repo: streamlit run app/app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soilgrain.gradation import gradation_indicators, soil_class
from soilgrain.plotting import CLASS_COLOR, GRID, INK_2, set_style
from soilgrain.predict import Predictor, effective_ppm, phone_table

MODEL = ROOT / "models" / "cnn_head.pt"
MANUAL = "Otro (indicar px/mm)"

st.set_page_config(page_title="Granulometría desde una foto", page_icon="🪨", layout="wide")
set_style()


@st.cache_resource
def get_predictor():
    return Predictor(MODEL)


st.title("Granulometría de un suelo a partir de una foto")
st.caption("CNN congelada (ConvNeXt-Tiny) + cabeza monótona entrenada con la EMD · "
           "24 muestras de entrenamiento: tómalo como una estimación, no como un ensayo de laboratorio.")

if not MODEL.exists():
    st.error("No hay modelo entrenado. Ejecuta antes `python -m soilgrain.train_final` desde `src/`.")
    st.stop()

with st.sidebar:
    st.header("Foto")
    upload = st.file_uploader("Superficie del suelo, en vertical y a unos 30 cm", type=["jpg", "jpeg", "png"])
    phone = st.selectbox("Móvil con el que se hizo", [*phone_table().index, MANUAL])
    manual_ppm = st.number_input("Píxeles por mm", 1.0, 100.0, 14.0, 0.5) if phone == MANUAL else None

if upload is None:
    st.info("Sube una foto en la barra lateral. La escala importa: elige el móvil o indica los píxeles por mm.")
    st.stop()

with Image.open(upload) as im:
    width, height = im.size
ppm = manual_ppm or effective_ppm(phone, width, height)
upload.seek(0)

with st.spinner("Analizando la foto…"):
    curve, img = get_predictor().predict(upload, ppm)
predictor = get_predictor()
sieves = predictor.sieves
ind = gradation_indicators(sieves, curve)
cls = soil_class(sieves, curve)

left, right = st.columns([1, 1.3])
with left:
    st.image(img, caption=f"Reescalada a {predictor.target_ppm} px/mm (escala original: {ppm:.1f} px/mm)",
             width="stretch")
with right:
    st.metric("Clase dominante (EN ISO 14688)", cls.capitalize())
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for b in (0.063, 2):
        ax.axvline(b, color=INK_2, lw=1, ls="--")
    ax.plot(sieves, curve, color=CLASS_COLOR[cls], marker="o", ms=4)
    ax.set(xscale="log", ylim=(-3, 103), xlabel="Tamaño de partícula (mm)", ylabel="% que pasa")
    for x, txt in [(0.01, "finos"), (0.35, "arena"), (12, "grava")]:
        ax.text(x, 105, txt, ha="center", color=INK_2, fontsize=8)
    ax.set_title("Curva granulométrica estimada", loc="left")
    fig.tight_layout()
    st.pyplot(fig)

    c = st.columns(6)
    for col, key, fmt in zip(c, ["D10", "D30", "D50", "D60", "Cu", "Cc"], ["{:.3f} mm"] * 4 + ["{:.1f}"] * 2):
        col.metric(key, "—" if pd.isna(ind[key]) else fmt.format(ind[key]))

with st.expander("Tabla de la curva"):
    st.dataframe(pd.DataFrame({"Tamiz (mm)": sieves, "% que pasa": curve.round(1)}), hide_index=True)
