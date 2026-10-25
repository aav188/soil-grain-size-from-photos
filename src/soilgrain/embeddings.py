"""Features de una CNN preentrenada y congelada sobre recortes a escala física común."""
import numpy as np
import timm
import torch

from .features import TARGET_PPM, patches

PATCH_PX = 224  # a 4,5 px/mm son ~50 mm de suelo


def load_backbone(name="convnext_tiny.fb_in22k", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std, device


@torch.no_grad()
def patch_embeddings(img, backbone, gray=False, stride_px=PATCH_PX // 2, batch=64):
    """Una fila por recorte. Con gray=True se quita el color, que es lo que más cambia entre cámaras.

    Cada recorte se promedia con su volteo horizontal y vertical (TTA barato).
    """
    model, mean, std, device = backbone
    crops = np.stack(list(patches(img, PATCH_PX, stride_px))).astype(np.float32) / 255
    if gray:
        lum = crops @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        crops = np.repeat(lum[..., None], 3, axis=-1)
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(crops[i:i + batch]).permute(0, 3, 1, 2).to(device)
        x = (x - mean) / std
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            emb = (model(x) + model(x.flip(3)) + model(x.flip(2))) / 3
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out)
