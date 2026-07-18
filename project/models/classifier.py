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
    """Load a RadImageNet DenseNet121 checkpoint (e.g. Lab-Rasool/RadImageNet's
    DenseNet121.pt) and remap its keys to match torchvision's densenet121.

    The checkpoint is a raw state_dict prefixed with "backbone.0." (its
    saved model wraps the DenseNet121 body under `backbone[0]`), while
    torchvision's densenet121().features uses no such prefix. Only the
    prefix differs -- the rest of the key structure lines up with
    torchvision's DenseNet121 exactly -- so we strip "backbone.0." and drop
    any keys torchvision's `features` submodule doesn't have (e.g. a
    classification head trained on RadImageNet's own label set).
    """
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    remapped = {}
    for key, value in raw_state_dict.items():
        if key.startswith("backbone.0."):
            remapped[key[len("backbone.0."):]] = value
    return remapped


class DenseNet121AnatomyClassifier(nn.Module):
    """DenseNet121 classifier with explicit feature extraction for LayerCAM.

    Input:  [B, 3, H, W]
    Output: logits [B, C] and final feature maps [B, 1024, H/32, W/32]
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        radimagenet_checkpoint: str | Path | None = None,
        anatomy_num_classes: int = 0,
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
        self.anatomy_num_classes = int(anatomy_num_classes)
        self.anatomy_classifier = (
            nn.Linear(self.classifier_input_features, self.anatomy_num_classes)
            if self.anatomy_num_classes > 0 else None
        )
        self.region_tumor_classifier = (
            nn.Linear(self.classifier_input_features, self.anatomy_num_classes * 2)
            if self.anatomy_num_classes > 0 else None
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Force fp32 through the backbone regardless of an enclosing
        # autocast context. Found empirically: a RadImageNet-pretrained
        # backbone can push intermediate activations for a rare pathological
        # input (e.g. a converted-from-grayscale X-ray with all 3 channels
        # identical) past fp16's ~65504 max mid-forward-pass through the
        # dense blocks, well before the final feature map -- clamping the
        # output afterward is too late once inf/nan has already propagated
        # through later dense layers. fp32 has enough headroom (this
        # backbone's worst observed activation was ~2.6e5, still far under
        # fp32's ~3.4e38 max) that the same input never overflows here.
        with torch.cuda.amp.autocast(enabled=False):
            x = self.features(x.float())
            x = torch.relu(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_anatomy: bool = False,
    ):
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        if return_anatomy:
            if self.anatomy_classifier is None or self.region_tumor_classifier is None:
                raise RuntimeError("This model has no anatomy-aware heads")
            anatomy_logits = self.anatomy_classifier(dropped)
            region_tumor_logits = self.region_tumor_classifier(dropped).view(
                -1, self.anatomy_num_classes, 2
            )
            return logits, anatomy_logits, region_tumor_logits, features
        if return_features:
            return logits, features
        return logits
