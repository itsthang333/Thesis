from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from torchvision.models import DenseNet121_Weights, densenet121
except Exception:  # pragma: no cover - torchvision version differences
    from torchvision.models import densenet121

    DenseNet121_Weights = None


def load_radimagenet_densenet121_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    remapped = {}
    for key, value in raw_state_dict.items():
        if key.startswith("backbone.0."):
            remapped[key[len("backbone.0."):]] = value
    return remapped


class DenseNet121AnatomyClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        radimagenet_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if radimagenet_checkpoint is not None:
            backbone = densenet121(weights=None)
            state_dict = load_radimagenet_densenet121_state_dict(radimagenet_checkpoint)
            missing, unexpected = backbone.features.load_state_dict(state_dict, strict=False)
            if missing:
                raise RuntimeError(f"RadImageNet checkpoint missing expected keys: {missing[:5]}")
            print(f"Loaded RadImageNet backbone from {radimagenet_checkpoint} "
                  f"({len(state_dict)} tensors, {len(unexpected)} unexpected keys ignored)")
        elif pretrained and DenseNet121_Weights is not None:
            backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            backbone = densenet121(weights=None)

        self.features = backbone.features
        self.classifier_input_features = backbone.classifier.in_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.classifier_input_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = self.features(x.float())
            x = torch.relu(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return logits, features
        return logits
