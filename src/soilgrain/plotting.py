"""Estilo común de los gráficos del proyecto."""
import matplotlib.pyplot as plt

# Paleta categórica validada para 3 clases (incluido daltonismo); el verde va siempre con etiqueta
CLASS_COLOR = {"finos": "#2a78d6", "arena": "#eb6834", "grava": "#1baf7a"}
CLASS_MARKER = {"finos": "o", "arena": "s", "grava": "^"}
CLASSES = list(CLASS_COLOR)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "lines.linewidth": 2, "figure.dpi": 110,
    })
