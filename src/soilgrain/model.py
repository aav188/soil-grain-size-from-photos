"""Redes con salida de curva acumulada siempre monótona y pérdida EMD."""
import timm
import torch
from torch import nn

from .metrics import LOG_WIDTHS


class MonotoneHead(nn.Module):
    """Predice la masa de cada intervalo con softmax y la acumula.

    Con 11 tamices hay 11 intervalos: lo que pasa por el primero y lo que queda entre
    cada par consecutivo. La suma acumulada termina en 1, así que la curva es monótona
    y vale 100 % en el tamiz de 200 mm por construcción.
    """

    def __init__(self, in_features, n_sieves=11, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_sieves),
        )

    def forward(self, x):
        return 100 * torch.cumsum(torch.softmax(self.net(x), dim=1), dim=1)


class GradationCNN(nn.Module):
    """Backbone preentrenado + cabeza monótona, para entrenar de extremo a extremo o para la demo."""

    def __init__(self, n_sieves=11, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = MonotoneHead(self.backbone.num_features, n_sieves)

    def forward(self, x):
        return self.head(self.backbone(x))


def emd_loss(pred, target, reduce=True):
    """La métrica de la competición como pérdida diferenciable (curvas en %).

    Con reduce=False devuelve una pérdida por fila, para poder ponderarlas.
    """
    w = torch.as_tensor(LOG_WIDTHS, dtype=pred.dtype, device=pred.device)
    per_row = ((pred - target).abs()[:, :-1] * w).sum(dim=1)
    return per_row.mean() if reduce else per_row
