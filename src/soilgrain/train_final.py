"""Entrena la cabeza monótona con todas las muestras y la guarda para la demo.

Uso: python -m soilgrain.train_final   (desde src/, tras ejecutar el notebook 03 para tener los embeddings)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_labels, photo_index
from .embeddings import PATCH_PX
from .features import TARGET_PPM
from .model import MonotoneHead, emd_loss

BACKBONE = "convnext_tiny.fb_in22k"
ROOT = Path(__file__).resolve().parents[2]
EMB_CACHE = ROOT / "data" / "processed" / f"emb_convnext_tiny_color_ppm{TARGET_PPM}.npz"
OUT = ROOT / "models" / "cnn_head.pt"


def train_heads(X, y, sample_ids, seeds=3, epochs=40, hidden=128, lr=1e-3, wd=1e-2, device="cpu"):
    """Mismos hiperparámetros que en el notebook 03; cada muestra pesa lo mismo."""
    w = pd.Series(sample_ids).map(1 / pd.Series(sample_ids).value_counts()).values
    mu, sd = X.mean(0), X.std(0) + 1e-6
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_t, y_t, w_t = to_t((X - mu) / sd), to_t(y), to_t(w / w.mean())
    states = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        head = MonotoneHead(X.shape[1], hidden=hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            for idx in torch.randperm(len(X_t), device=device).split(256):
                loss = (emd_loss(head(X_t[idx]), y_t[idx], reduce=False) * w_t[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        states.append({k: v.cpu() for k, v in head.state_dict().items()})
    return {"states": states, "mu": mu, "sd": sd, "hidden": hidden, "in_features": X.shape[1]}


def main():
    photos = photo_index()
    labels = load_labels()
    z = np.load(EMB_CACHE)
    train = np.where(photos["split"] == "train")[0]
    X = np.concatenate([z[str(i)] for i in train])
    sids = np.concatenate([[photos["sample_id"].iat[i]] * len(z[str(i)]) for i in train])
    y = labels.loc[sids].values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = train_heads(X, y, sids, device=device)
    bundle |= {"backbone": BACKBONE, "target_ppm": TARGET_PPM, "patch_px": PATCH_PX,
               "sieves": labels.columns.tolist()}
    OUT.parent.mkdir(exist_ok=True)
    torch.save(bundle, OUT)
    print(f"Guardado en {OUT.relative_to(ROOT)} ({len(X)} recortes de {len(train)} fotos)")


if __name__ == "__main__":
    main()
