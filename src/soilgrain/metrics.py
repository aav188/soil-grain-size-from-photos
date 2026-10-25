"""Métrica oficial y post-proceso para que una curva sea un envío válido."""
import numpy as np

SIEVES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
LOG_WIDTHS = np.diff(np.log10(SIEVES))


def emd(y_true, y_pred):
    """EMD con peso logarítmico por muestra (0-500, menor es mejor).

    Sigue la fórmula de la competición: |F_i - F̂_i| en el punto izquierdo de cada
    uno de los 10 intervalos, ponderado por su anchura en décadas.
    """
    y_true, y_pred = np.atleast_2d(y_true), np.atleast_2d(y_pred)
    return (np.abs(y_true - y_pred)[:, :-1] * LOG_WIDTHS).sum(axis=1)


def mean_emd(y_true, y_pred):
    return float(emd(y_true, y_pred).mean())


def make_valid(curves):
    """Recorta a [0, 100], fuerza la monotonía y fija 100 en el tamiz de 200 mm."""
    curves = np.clip(np.atleast_2d(curves), 0, 100)
    curves = np.maximum.accumulate(curves, axis=1)
    curves[:, -1] = 100.0
    return curves
