"""Indicadores geotécnicos a partir de una curva granulométrica acumulada.

La curva se describe con tamaños de tamiz (mm, crecientes) y el porcentaje
que pasa por cada uno (0-100, no decreciente).
"""
import numpy as np


def d_value(sizes_mm, percent_passing, p):
    """Tamaño Dp (mm) por el que pasa el p % de la muestra.

    Interpola en escala logarítmica de tamaño, como en el gráfico semilog estándar.
    Devuelve NaN si p queda fuera del rango de la curva.
    """
    sizes = np.asarray(sizes_mm, dtype=float)
    passing = np.asarray(percent_passing, dtype=float)
    if p < passing.min() or p > passing.max():
        return np.nan
    # np.interp necesita x creciente; las mesetas se resuelven por el primer punto
    passing_u, idx = np.unique(passing, return_index=True)
    return float(10 ** np.interp(p, passing_u, np.log10(sizes[idx])))


def gradation_indicators(sizes_mm, percent_passing):
    d10, d30, d50, d60 = (d_value(sizes_mm, percent_passing, p) for p in (10, 30, 50, 60))
    return {
        "D10": d10,
        "D30": d30,
        "D50": d50,
        "D60": d60,
        "Cu": d60 / d10,
        "Cc": d30**2 / (d10 * d60),
    }


def soil_class(sizes_mm, percent_passing):
    """Fracción dominante según EN ISO 14688: finos < 0.063 mm, arena 0.063-2 mm, grava 2-63 mm.

    Son los mismos cortes que los tamices DIN de los datos de la competición.
    """
    sizes = np.log10(np.asarray(sizes_mm, dtype=float))
    passing = np.asarray(percent_passing, dtype=float)
    fines = np.interp(np.log10(0.063), sizes, passing)
    sand = np.interp(np.log10(2.0), sizes, passing) - fines
    gravel = 100 - fines - sand
    return ("finos", "arena", "grava")[int(np.argmax([fines, sand, gravel]))]
