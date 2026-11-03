"""Envío final: ensemble de modelos de textura y de CNN, con perfiles de pesos nombrados.

Uso: python -m soilgrain.ensemble [agg|v1|v2|v3]   (desde src/; por defecto agg, el mejor en Kaggle)

- agg: solo el Ridge agregado por muestra. Kaggle: 43,52, el mejor con diferencia. Los miembros con
  CNN empujan las curvas hacia los finos y el test parece más grueso.

- v1: 50 % Ridge agregado + 35 % Ridge + 15 % cabeza CNN. Puntuación pública en Kaggle: 56,48.
- v2: 30 % Ridge agregado + 30 % Ridge + 10 % cabeza CNN + 30 % Ridge sobre embeddings de la CNN.
  Mejora a v1 en los tres proxies de validación (por obra y entre cámaras). Kaggle: 56,25.
- v3: los pesos de v2, con las features de textura centradas por familia de cámara
  (validation.center_by_camera). Proxies: 49,2 por obra, 48,5 / 43,2 entre cámaras (v2: 50,6 / 57,9 / 45,4).
  Kaggle: 58,56, PEOR que v2: centrar borra señal real si los suelos del test difieren de los de train.

Los pesos NO se ajustan con la validación entre cámaras: el envío con esos pesos (95 % CNN) sacó 70,06
en Kaggle frente a 56,48 de v1.
"""
import sys

import pandas as pd

from . import validation as v
from .metrics import make_valid

SUBMISSIONS = v.ROOT / "submissions"
TEMPLATE = v.ROOT / "data" / "raw" / "sample_submission.csv"
MEMBERS = ("ridge_aggregated", "ridge_photo", "cnn", "cnn_ridge")
PROFILES = {"v1": (0.50, 0.35, 0.15, 0.0), "v2": (0.30, 0.30, 0.10, 0.30), "v3": (0.30, 0.30, 0.10, 0.30),
            "agg": (1.0, 0.0, 0.0, 0.0)}
CENTERED = {"v3"}  # perfiles cuyas features de textura se centran por familia de cámara


def main(profile="agg"):
    weights = PROFILES[profile]
    ctx = v.load_context(center_tex=profile in CENTERED)
    template = pd.read_csv(TEMPLATE)
    tr = (ctx.photos["split"] == "train").values.nonzero()[0]
    cols = template.columns[1:]
    used = [m for m, w in zip(MEMBERS, weights) if w > 0]
    preds = {m: [] for m in used}
    for sid in template["sample_id"]:
        ev = (ctx.sid == sid).nonzero()[0]
        for m in used:
            preds[m].append(v.MODELS[m](ctx, tr, ev))
    frames = [make_valid(preds[m]) for m in used]
    ens = v.blend(frames, [w for w in weights if w > 0])
    out = pd.DataFrame(ens, index=template["sample_id"], columns=cols).round(4)
    assert (out.diff(axis=1).iloc[:, 1:] >= 0).all().all() and (out.iloc[:, -1] == 100).all()
    path = SUBMISSIONS / f"ensemble_{profile}.csv"
    SUBMISSIONS.mkdir(exist_ok=True)
    out.to_csv(path)
    print(f"Guardado {path} con pesos {dict(zip(MEMBERS, weights))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "agg")  # agg es el mejor en Kaggle
