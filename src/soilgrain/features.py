"""Features de textura explicables, calculadas sobre recortes a escala física común."""
import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

TARGET_PPM = 4.5  # la resolución más baja del train: así nadie se reescala hacia arriba
PATCH_MM = 40
SCALES_MM = (0.5, 1, 2, 4, 8)
GLCM_DIST_MM = (0.5, 1, 2, 4)


def luminance(img):
    return img @ np.array([0.299, 0.587, 0.114]) / 255


def patches(img, patch_px, stride_px=None):
    stride_px = stride_px or patch_px
    h, w = img.shape[:2]
    for y in range(0, h - patch_px + 1, stride_px):
        for x in range(0, w - patch_px + 1, stride_px):
            yield img[y:y + patch_px, x:x + patch_px]


def patch_features(patch, ppm=TARGET_PPM):
    lum = luminance(patch)
    feats = {
        "lum_mean": lum.mean(),
        "lum_std": lum.std(),
        "lum_p10": np.percentile(lum, 10),
        "lum_p90": np.percentile(lum, 90),
    }
    rgb = patch.reshape(-1, 3).astype(float) + 1
    chroma = rgb / rgb.sum(axis=1, keepdims=True)
    feats["chroma_r"], feats["chroma_g"] = chroma[:, 0].mean(), chroma[:, 1].mean()

    # El resto se calcula sobre la luminancia normalizada, para restar el efecto de la cámara
    z = (lum - lum.mean()) / (lum.std() + 1e-6)
    gy, gx = np.gradient(z)
    grad = np.hypot(gx, gy)
    feats["grad_mean"], feats["grad_std"] = grad.mean(), grad.std()

    # Energía por banda de escala (diferencia de gaussianas): dónde está la variación
    prev = z
    for s in SCALES_MM:
        blurred = ndimage.gaussian_filter(z, sigma=s * ppm)
        feats[f"band_{s}mm"] = (prev - blurred).var()
        prev = blurred
    feats["band_rest"] = prev.var()

    q = np.clip((z + 3) / 6 * 31, 0, 31).astype(np.uint8)
    dists = [max(1, round(d * ppm)) for d in GLCM_DIST_MM]
    glcm = graycomatrix(q, dists, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "correlation", "energy"):
        values = graycoprops(glcm, prop).mean(axis=1)
        for d, v in zip(GLCM_DIST_MM, values):
            feats[f"glcm_{prop}_{d}mm"] = v
    return feats


def photo_features(img, ppm=TARGET_PPM):
    """Media y desviación de las features de todos los recortes de una foto ya reescalada."""
    rows = [patch_features(p, ppm) for p in patches(img, int(PATCH_MM * ppm))]
    keys = rows[0].keys()
    arr = np.array([[r[k] for k in keys] for r in rows])
    out = {f"{k}": v for k, v in zip(keys, arr.mean(axis=0))}
    out |= {f"{k}_sd": v for k, v in zip(keys, arr.std(axis=0))}
    out["n_patches"] = len(rows)
    return out
