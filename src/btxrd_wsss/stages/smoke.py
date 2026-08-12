from __future__ import annotations

import gc

import numpy as np
import torch
from PIL import Image

from btxrd_wsss.config import PipelineConfig
from btxrd_wsss.models.biomedclip import FrozenBiomedCLIP
from btxrd_wsss.models.hrnet_mil import HRNetDenseMIL
from btxrd_wsss.models.rad_dino_g1 import FrozenRadDINODescriptor
from btxrd_wsss.pipeline.sam_gallery import create_sam_backend
from btxrd_wsss.types import CandidateMask, Proposal


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def smoke_models(config: PipelineConfig) -> dict[str, object]:
    """Load every external checkpoint and exercise its exact adapter once."""
    device = torch.device(config.runtime.device)
    report: dict[str, object] = {}
    hrnet = (
        HRNetDenseMIL(
            backbone_name=config.hrnet.backbone,
            pretrained=config.hrnet.pretrained,
            classes=config.hrnet.output_classes,
            dense_channels=config.hrnet.dense_channels,
            dropout=config.hrnet.dropout,
            topk_fractions=tuple(config.hrnet.topk_fractions),
            gradient_checkpointing=False,
        )
        .eval()
        .to(device)
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        output = hrnet(torch.zeros(1, 3, 128, 128, device=device))
    report["hrnet"] = {"dense_shape": list(output.dense_logits.shape)}
    del hrnet, output
    _release()

    pixels = np.full((96, 128), 0.5, np.float32)
    image = Image.fromarray(np.full((96, 128, 3), 127, np.uint8))
    biomed = FrozenBiomedCLIP.from_pretrained(config.biomedclip.model_id, config.runtime.device)
    semantic = biomed.localize(
        image,
        crop_fraction=config.biomedclip.crop_fraction,
        positions_per_axis=config.biomedclip.positions_per_axis,
        top_k_tiles=config.biomedclip.top_k_tiles,
    )
    report["biomedclip"] = {"saliency_shape": list(semantic.saliency.shape)}
    del biomed, semantic
    _release()

    component = np.zeros_like(pixels, bool)
    component[32:48, 48:64] = True
    proposal = Proposal(
        proposal_id="smoke",
        source="hrnet_tile",
        source_view="smoke",
        native_box=(44, 28, 68, 52),
        positive_points=((56, 40),),
        negative_points=((40, 24),),
        score=0.8,
        component_mask=component,
        metadata={"peak_x": 56, "peak_y": 40, "source_confidence": 0.8},
    )
    sam = create_sam_backend(config.sam, config.runtime.device)
    predictions = sam.predict_roi(
        pixels, proposal, roi_scale=config.sam.initial_roi_scale, multimask=False
    )
    mask, predicted_iou, stability = predictions[0]
    if not mask.any():
        mask = component.copy()
    sam_name = sam.name
    report["sam"] = {
        "mask_shape": list(mask.shape),
        "predicted_iou": predicted_iou,
        "stability": stability,
    }
    del sam, predictions
    _release()

    candidate = CandidateMask(
        candidate_id="smoke",
        mask=mask,
        proposal_id="smoke",
        proposal_source="hrnet_tile",
        sam_backend=sam_name,
        prompt_type="box+point",
        predicted_iou=predicted_iou,
        stability=stability,
        roi_scale=config.sam.initial_roi_scale,
        metadata={"source_component": component},
    )
    rad_dino = FrozenRadDINODescriptor(
        config.rad_dino.model_id,
        input_size=config.rad_dino.input_size,
        selected_layers=config.rad_dino.selected_layers,
        projection_dim=config.g1.projection_dim,
        batch_size=1,
        device=config.runtime.device,
        seed=config.experiment.seed,
    )
    descriptor = rad_dino.extract(image, [candidate], config.rad_dino.context_scales)
    report["rad_dino"] = {"descriptor_shape": list(descriptor.values.shape)}
    del rad_dino, descriptor
    _release()
    return report
