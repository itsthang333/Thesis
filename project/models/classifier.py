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
        num_anatomy_regions: int | None = None,
        region_conditioned_tumor_head: bool = False,
    ) -> None:
        """num_anatomy_regions: when set, adds a second linear head
        (self.region_classifier) predicting the coarse anatomy region
        (upper limb / lower limb / pelvis, see datasets/btxrd.py's
        ANATOMY_REGION_COLUMNS) from the same pooled backbone features as
        the main tumor_type head. This auxiliary head has no effect on the
        main classifier (self.classifier, self.features) that LayerCAM and
        PuzzleCAM/Teacher-Student hook into directly by attribute name --
        it is purely additive. None (default) disables it entirely,
        matching prior behavior exactly.

        region_conditioned_tumor_head: when True (requires
        num_anatomy_regions to also be set), adds one binary
        tumor-vs-normal linear head PER anatomy region
        (self.region_tumor_classifiers[r]), instead of a single global
        normal class. This lets a sample's loss and CAM evidence be
        anchored to "normal for its own region" rather than one shared
        normal concept spanning upper limb/lower limb/pelvis anatomy --
        see anatomy_conditioned_cam_score() in models/layercam.py for how
        this feeds anatomy-conditioned CAM. False (default) disables it
        entirely, matching prior behavior exactly.
        """
        super().__init__()
        if region_conditioned_tumor_head and num_anatomy_regions is None:
            raise ValueError("region_conditioned_tumor_head=True requires num_anatomy_regions to be set")
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
        self.region_classifier = (
            nn.Linear(self.classifier_input_features, num_anatomy_regions)
            if num_anatomy_regions is not None
            else None
        )
        self.region_tumor_classifiers = (
            nn.ModuleList(
                [nn.Linear(self.classifier_input_features, 1) for _ in range(num_anatomy_regions)]
            )
            if region_conditioned_tumor_head
            else None
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

    def forward(self, x: torch.Tensor, return_features: bool = False, return_embedding: bool = False):
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        outputs: tuple = (logits,)
        if return_features:
            outputs = outputs + (features,)
        if return_embedding:
            # Undropped pooled features, for anatomy-matched contrastive loss
            # (models/anatomy_contrastive.py) -- dropout is a training-time
            # regularizer for the classification head, not something that
            # should perturb the embedding space two samples are compared in.
            outputs = outputs + (pooled,)
        return outputs[0] if len(outputs) == 1 else outputs

    def forward_region_logits(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Auxiliary region-classification logits from pooled backbone
        features already computed by forward(..., return_embedding=True).
        Raises if num_anatomy_regions was not set at construction time.
        """
        if self.region_classifier is None:
            raise RuntimeError(
                "forward_region_logits() called but this model was constructed with "
                "num_anatomy_regions=None -- no region_classifier head exists."
            )
        return self.region_classifier(self.dropout(pooled_features))

    def forward_region_tumor_logits(
        self, pooled_features: torch.Tensor, anatomy_region: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample binary tumor-vs-normal logit, using each sample's OWN
        region's binary head (see region_conditioned_tumor_head in __init__).

        anatomy_region: [B] long tensor, index into ANATOMY_REGION_COLUMNS
        (datasets/btxrd.py). Every entry must be >= 0 (a known region) --
        callers should filter out anatomy_region == -1 samples first, same
        as forward_region_logits's caller does for the region CE loss.

        Returns: [B] logits (one per sample, from that sample's region head).
        """
        if self.region_tumor_classifiers is None:
            raise RuntimeError(
                "forward_region_tumor_logits() called but this model was constructed with "
                "region_conditioned_tumor_head=False -- no region_tumor_classifiers exist."
            )
        dropped = self.dropout(pooled_features)
        logits = torch.zeros(pooled_features.shape[0], device=pooled_features.device, dtype=pooled_features.dtype)
        for region_index, head in enumerate(self.region_tumor_classifiers):
            mask = anatomy_region == region_index
            if mask.any():
                # Under an enclosing torch.cuda.amp.autocast (as in
                # train_classifier.py's run_epoch_multiclass), nn.Linear
                # casts its output to fp16 even though pooled_features (and
                # therefore this method's zero-initialized `logits`) is fp32
                # -- forward_features forces fp32 through the backbone, see
                # its own docstring. Index-assigning a fp16 tensor into a
                # fp32 tensor raises "Index put requires the source and
                # destination dtypes match" -- cast the head's output back
                # to logits' dtype before assigning.
                logits[mask] = head(dropped[mask]).squeeze(-1).to(dtype=logits.dtype)
        return logits
