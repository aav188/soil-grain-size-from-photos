"""CNN preentrenada que predice una curva acumulada siempre monótona."""
import timm
import torch
from torch import nn


class GradationCNN(nn.Module):
    """Predice incrementos por intervalo con softmax y los acumula.

    La salida va de ~0 a 1 y nunca decrece, así que siempre es una curva físicamente válida.
    """

    def __init__(self, n_sieves, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # Un intervalo extra para la fracción mayor que el último tamiz
        self.head = nn.Linear(self.backbone.num_features, n_sieves + 1)

    def forward(self, x):
        increments = torch.softmax(self.head(self.backbone(x)), dim=1)
        return torch.cumsum(increments, dim=1)[:, :-1]
