from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import DenseNet121_Weights, densenet121
except Exception:  # pragma: no cover - torchvision version differences
    from torchvision.models import densenet121

    DenseNet121_Weights = None


class DenseNet121AnatomyClassifier(nn.Module):
    """DenseNet121 multi-label anatomy classifier with explicit feature extraction for Grad-CAM.

    Input:  [B, 3, H, W]
    Output: logits [B, C] and final feature maps [B, 1024, H/32, W/32]
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        if pretrained and DenseNet121_Weights is not None:
            backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            backbone = densenet121(weights=None)

        self.features = backbone.features
        self.classifier_input_features = backbone.classifier.in_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.classifier_input_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return logits, features
        return logits
