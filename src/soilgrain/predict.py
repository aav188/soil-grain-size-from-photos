"""Inferencia de una foto nueva: reescalado a escala común, CNN congelada y cabeza monótona."""
import numpy as np
import pandas as pd
import torch

from .data import RAW, load_photo
from .embeddings import load_backbone, patch_embeddings
from .model import MonotoneHead


def phone_table():
    return pd.read_csv(RAW / "ppm_updated.csv").set_index("phone")


def effective_ppm(phone, width, height):
    """Corrige el ppm de la tabla si la foto no está a la resolución nativa del móvil."""
    row = phone_table().loc[phone]
    return row.ppm * max(width, height) / max(row.width, row.height)


class Predictor:
    def __init__(self, bundle_path, device=None):
        bundle = torch.load(bundle_path, weights_only=False)
        self.backbone = load_backbone(bundle["backbone"], device)
        self.device = self.backbone[3]
        self.target_ppm = bundle["target_ppm"]
        self.sieves = np.array(bundle["sieves"], dtype=float)
        self.mu = torch.tensor(bundle["mu"], dtype=torch.float32, device=self.device)
        self.sd = torch.tensor(bundle["sd"], dtype=torch.float32, device=self.device)
        self.heads = []
        for state in bundle["states"]:
            head = MonotoneHead(bundle["in_features"], hidden=bundle["hidden"])
            head.load_state_dict(state)
            self.heads.append(head.to(self.device).eval())

    @torch.no_grad()
    def predict(self, source, ppm):
        """Devuelve (curva en % para cada tamiz, imagen reescalada)."""
        img = load_photo(source, self.target_ppm, ppm)
        emb = torch.tensor(patch_embeddings(img, self.backbone), device=self.device)
        x = (emb - self.mu) / self.sd
        curve = torch.stack([h(x).mean(0) for h in self.heads]).mean(0).cpu().numpy()
        return curve, img
