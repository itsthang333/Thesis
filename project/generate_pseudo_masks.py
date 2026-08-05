from __future__ import annotations

"""Stage 2 — Generate pseudo bone masks using LayerCAM + SAM.

Pipeline per pipeline.md:
  Image
  → DenseNet121 → logits → sigmoid weights
  → LayerCAM (denseblock2/3/4, confidence-filtered, weighted-average fused)
  → Adaptive threshold → connected components → peak extraction (SAM prompts)
  → SAM ViT-B → candidate masks
  → CAM-guided mask selection (score = mean CAM inside mask)
  → Morphological refinement (closing → opening → fill_holes → remove_small)
  → Final pseudo mask saved as PNG
"""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    BTXRD_BEST_PIPELINE,
    BTXRD_HYBRID_PIPELINE,
    DATASET_TARGET_COLUMNS,
)
from progress import should_disable_tqdm
from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.s2c_cpm import (
    DenseNet121S2CCPMClassifier,
    S2CCPMDirectCAM,
)
from models.layercam import LayerCAM
from models.unet import architecture_name_from_metadata, build_segmentation_model
from evaluation.frozen_test_guard import verify_frozen_test_config
from pseudo.generate_layercam import generate_fused_cam
from pseudo.extract_prompts import extract_point_prompts
from pseudo import tumor_morphology as morphology
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import score_masks, select_and_fuse_masks
from pseudo.manifest import sha256_file as manifest_sha256_file, write_pseudo_mask_manifest
from pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    write_candidate_diagnostics_manifest,
)
from pseudo.affinity_selector_input import (
    load_affinity_selector_contract,
    load_affinity_selector_map,
)
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def external_saliency_test_authorized(
    split: str,
    frozen_test_document: dict[str, object] | None,
) -> bool:
    """Allow final-test proposal generation only under the WSSS-only lock."""
    return split != "test" or (
        frozen_test_document is not None
        and frozen_test_document.get("scope") == "wsss_prediction_only"
    )


def parse_device_spec(value: str) -> str:
    """Accept generic or explicitly indexed CUDA devices."""
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return normalized
    raise argparse.ArgumentTypeError(
        "device must be one of auto, cpu, cuda, or an indexed CUDA device "
        "such as cuda:0"
    )


def validate_runtime_device(device: torch.device, option: str) -> None:
    """Fail clearly when an explicitly selected CUDA device is unavailable."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"{option} {device} requested but CUDA is unavailable")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{option} {device} requested but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BTXRD pseudo masks via LayerCAM + SAM")
    parser.set_defaults(dataset="btxrd")
    parser.add_argument(
        "--pipeline-profile",
        type=str,
        default="default",
        choices=["default", "btxrd_best", "btxrd_hybrid"],
        help=(
            "Use a reproducible tested downstream configuration. btxrd_best and btxrd_hybrid select the same "
            "BTXRD validation configuration (320px CAM, 512px SAM, CAM contrast/"
            "percentile ensemble, prompt ensemble and coverage_mass_sam). It never "
            "enables polygon/bbox inputs."
        ),
    )
    parser.add_argument(
        "--allow-validation-component-topk-ablation",
        action="store_true",
        help=(
            "Allow an explicit --component-topk override under a canonical BTXRD profile only on split=val. "
            "This is a diagnostic ablation and must never be used for canonical train/test artifacts."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True, help="BTXRD dataset root")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--frozen-config", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Immutable derived split manifest. Its assignments are authoritative for BTXRD.",
    )
    parser.add_argument(
        "--classifier-split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional provenance-only split manifest bound to an older classifier "
            "checkpoint. Dataset selection and external saliency remain bound to "
            "--split-manifest."
        ),
    )
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument(
        "--external-saliency-manifest",
        type=Path,
        default=None,
        help=(
            "Optional prediction-first image-label-only saliency manifest. Its "
            "role is controlled by --external-saliency-role."
        ),
    )
    parser.add_argument(
        "--external-saliency-role",
        choices=["replace", "proposal_gallery"],
        default="replace",
        help=(
            "replace reproduces the original external-saliency ablation. "
            "proposal_gallery keeps LayerCAM as the selector/support map and "
            "unconditionally appends external-saliency component/SAM proposals."
        ),
    )
    parser.add_argument("--external-saliency-run-metadata", type=Path, default=None)
    parser.add_argument("--external-saliency-expected-manifest-sha256", default=None)
    parser.add_argument("--external-saliency-expected-metadata-sha256", default=None)
    parser.add_argument("--external-saliency-expected-source-commit", default=None)
    parser.add_argument("--external-saliency-expected-model-weight-sha256", default=None)
    parser.add_argument("--affinity-selector-manifest", type=Path, default=None)
    parser.add_argument("--affinity-selector-package-metadata", type=Path, default=None)
    parser.add_argument("--affinity-selector-prediction-freeze", type=Path, default=None)
    parser.add_argument("--affinity-selector-expected-manifest-sha256", default=None)
    parser.add_argument("--affinity-selector-expected-package-metadata-sha256", default=None)
    parser.add_argument("--affinity-selector-expected-prediction-freeze-sha256", default=None)
    parser.add_argument("--affinity-selector-expected-source-commit", default=None)
    parser.add_argument("--affinity-selector-expected-protocol-sha256", default=None)
    parser.add_argument("--affinity-selector-expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--proposal-teacher-segmentation-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional frozen pseudo-mask-trained U-Net used to add SAM prompt "
            "components. Its probability map is also available only to the explicit "
            "source_consensus selector; CAM support clipping remains unchanged."
        ),
    )
    parser.add_argument(
        "--proposal-teacher-expected-sha256",
        type=str,
        default=None,
        help="Required immutable SHA-256 when a proposal-teacher checkpoint is supplied.",
    )
    parser.add_argument(
        "--proposal-teacher-threshold",
        type=float,
        default=0.85,
        help="Absolute probability threshold for frozen teacher proposal components.",
    )
    parser.add_argument(
        "--proposal-teacher-min-component-area",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--proposal-teacher-max-components",
        type=int,
        default=3,
    )
    parser.add_argument("--auxiliary-binary-checkpoint", type=Path, default=None,
                        help="Optional one-logit tumor checkpoint whose CAM is blended with the main CAM.")
    parser.add_argument("--auxiliary-binary-weight", type=float, default=0.35,
                        help="Blend weight for --auxiliary-binary-checkpoint CAM (0..1).")
    parser.add_argument("--sam-checkpoint", type=Path, required=True,
                        help="Path to an attached local SAM checkpoint. Runtime downloads are disabled "
                        "so that the run is reproducible and works with Kaggle Internet off.")
    parser.add_argument(
        "--sam-device",
        type=parse_device_spec,
        default="auto",
        help=(
            "Device for SAM. 'auto' follows the classifier; indexed CUDA devices "
            "such as cuda:1 are supported for multi-GPU execution."
        ),
    )
    parser.add_argument(
        "--classifier-device",
        type=parse_device_spec,
        default="auto",
        help=(
            "Device for DenseNet/LayerCAM. Use cpu for constrained GPUs or an "
            "indexed CUDA device such as cuda:0 for multi-GPU execution."
        ),
    )
    parser.add_argument("--target-columns", type=str, default=None,
                        help="BTXRD target column; defaults to 'tumor'")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-image-size", type=int, default=0,
                        help="Square input resolution for SAM, loaded from the original image. "
                             "0 keeps the classifier/CAM resolution (default); e.g. 512 preserves "
                             "more X-ray detail while prompts are scaled and masks returned to CAM size.")
    parser.add_argument("--sam-preserve-aspect", action="store_true",
                        help="When --sam-image-size is set, preserve radiograph aspect ratio; prompt "
                             "geometry is scaled independently in x/y before SAM inference.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pseudo_masks")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly allow replacing masks already present in --output-dir. Default is fail-safe.",
    )
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Optional text file of image names to process (one per line), for deterministic stratified validation smoke tests.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Preview mode limit; 0 processes all images (the notebook preview passes 10)")
    parser.add_argument("--process-all", action="store_true",
                        help="Process the full dataset when generating pseudo masks for segmentation training")
    parser.add_argument("--save-visuals-limit", type=int, default=10,
                        help="Maximum number of images for which CAM overlays/debug visuals are saved")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Min sigmoid score for a class CAM to participate in fusion")
    parser.add_argument("--cam-percentile", type=float, default=85.0,
                        help="Percentile threshold for foreground extraction (85/90/95)")
    parser.add_argument("--cam-percentile-ensemble", action="store_true",
                        help="A/B: build prompt components at the values supplied by --cam-percentile-values and union the proposals.")
    parser.add_argument("--disable-cam-percentile-ensemble", action="store_true",
                        help="A/B override for the default profile; rejected by btxrd_best.")
    parser.add_argument("--cam-percentile-values", type=str, default="70,85,90",
                        help="Comma-separated CAM percentiles used by --cam-percentile-ensemble.")
    parser.add_argument("--max-points", type=int, default=5,
                        help="Max SAM prompt points per image")
    parser.add_argument("--min-component-area", type=int, default=100,
                        help="Min pixels for a CAM component to generate a prompt")
    parser.add_argument("--mask-score-threshold", type=float, default=0.4,
                        help="Min mean-CAM score for a SAM mask to be kept")
    parser.add_argument("--closing-kernel", type=int, default=0)
    parser.add_argument("--opening-kernel", type=int, default=0,
                        help="0 disables opening; recommended for thin hand/wrist bones")
    parser.add_argument("--min-size", type=int, default=40,
                        help="Minimum final component size; kept small for phalanges/carpal bones")
    parser.add_argument("--max-hole-area", type=int, default=0,
                        help="Only fill enclosed holes up to this area; preserves gaps between bones")
    parser.add_argument("--guidance-threshold", type=float, default=0.40,
                        help="Minimum mean bone-likelihood for keeping a final mask component")
    parser.add_argument("--seed-percentile", type=float, default=None,
                        help="BTXRD default: 82.0")
    parser.add_argument("--support-percentile", type=float, default=None,
                        help="BTXRD default: 55.0")
    parser.add_argument("--morphology-fusion-mode", type=str, default="components",
                        choices=["components", "weighted"])
    parser.add_argument("--sam-prompt-mode", type=str, default="box_point",
                        choices=["point", "joint_points", "box", "box_point"])
    parser.add_argument("--sam-prompt-ensemble", action="store_true",
                        help="A/B: generate candidates from box_point, point, and box prompts for each CAM component.")
    parser.add_argument(
        "--sam-grid-gallery",
        action="store_true",
        help=(
            "Pro2SAM-style diagnostic: replace component prompts with the official "
            "SAM dense grid mask gallery. No annotation or lesion-size input is used."
        ),
    )
    parser.add_argument("--sam-grid-points-per-side", type=int, default=32)
    parser.add_argument("--sam-grid-points-per-batch", type=int, default=64)
    parser.add_argument("--sam-grid-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--sam-grid-stability-thresh", type=float, default=0.95)
    parser.add_argument("--sam-grid-box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--disable-sam-prompt-ensemble", action="store_true",
                        help="A/B override for the default profile; rejected by btxrd_best.")
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--all-cam-components", action="store_true",
                        help="A/B option for BTXRD: keep up to --max-components CAM components instead of largest-only.")
    parser.add_argument("--disable-all-cam-components", action="store_true",
                        help="A/B override for the default profile; rejected by btxrd_best.")
    parser.add_argument("--points-per-component", type=int, default=3)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.02)
    parser.add_argument("--negative-points-per-component", type=int, default=4)
    parser.add_argument("--prompt-border-margin", type=int, default=2,
                        help="Drop positive SAM points this many pixels from image borders")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.35,
                        help="Drop SAM box prompts larger than this fraction of the image; <=0 disables")
    parser.add_argument("--sam-single-mask", action="store_true")
    parser.add_argument("--include-cam-candidate", action="store_true",
                        help="A/B: append each image-level CAM component itself as a fallback candidate "
                             "alongside SAM masks; no segmentation annotation is used.")
    parser.add_argument("--disable-morphology", action="store_true",
                        help="Run the original CAM-only baseline without pre-SAM bone morphology")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--preprocessing-mode", type=str, default="none",
                        choices=["none", "clahe", "contrast", "gamma", "foreground_crop"],
                        help="Optional X-ray preprocessing before classifier/CAM")
    parser.add_argument(
        "--cam-backend",
        choices=["layercam", "s2c_cpm"],
        default="layercam",
        help=(
            "layercam uses gradient-weighted DenseNet blocks; s2c_cpm uses "
            "the checkpoint's directly trained stride-8 multiscale CAM."
        ),
    )
    parser.add_argument(
        "--layercam-weights",
        type=str,
        default="0.20,0.30,0.50",
        help="Three comma-separated LayerCAM weights for denseblock2/3/4.",
    )
    parser.add_argument("--layercam-gradient-mode", type=str, default="positive",
                        choices=["positive", "absolute"],
                        help="LayerCAM gradient evidence: standard positive ReLU or absolute-gradient ablation.")
    parser.add_argument("--selection-method", type=str, default="bone_hybrid",
                        choices=[
                            "mean", "sum", "mean_area", "coverage", "coverage_mass",
                            "coverage_mass_sam", "coverage_mass_sam_causal", "hybrid",
                            "bone_hybrid", "simple_hybrid", "prompt_hybrid",
                            "consistency_hybrid", "source_consensus",
                            "prompt_source_graph", "affinity_rank_single",
                        ],
                        help="CAM-guided mask scoring method")
    parser.add_argument(
        "--prompt-score-weights",
        type=str,
        default="0.30,0.20,0.15,0.15,0.20",
        help="Five comma-separated prompt_hybrid weights: CAM coverage, CAM density, "
             "within-prompt SAM rank, support-relative area, prompt consistency.",
    )
    parser.add_argument(
        "--prompt-area-target",
        type=float,
        default=2.0,
        help="prompt_hybrid expected candidate-area / CAM-support-area expansion ratio.",
    )
    parser.add_argument(
        "--prompt-area-log-sigma",
        type=float,
        default=1.0,
        help="Width of prompt_hybrid's log-space area prior; larger is less restrictive.",
    )
    parser.add_argument("--fusion-topk", type=int, default=3,
                        help="0=OR all above-thresh, 1=top-1 only, k>1=union top-k, k<0=intersect top-|k|. "
                        "Only used when --disable-best-per-component is set or components are unavailable "
                        "-- best_per_component (on by default whenever component_ids exist) takes priority "
                        "over fusion_topk in select_and_fuse_masks and returns before fusion_topk's branch runs.")
    parser.add_argument("--disable-best-per-component", action="store_true",
                        help="Disable per-component best-candidate selection (which unions the single best "
                        "SAM candidate from every kept morphology component, up to --max-components) "
                        "and fall back to fusion_topk's global top-k selection across all candidates instead. "
                        "Per-component selection can union in a non-lesion component's mask alongside the "
                        "real lesion's, which is invisible to fusion_topk and can dilute Dice more than "
                        "fusion_topk ever could -- this flag isolates that effect for A/B testing.")
    parser.add_argument("--component-topk", type=int, default=0,
                        help="When best-per-component is enabled, keep only the top K component proposals by image-only score; 0 keeps all.")
    parser.add_argument("--support-clip-kernel", type=int, default=5,
                        help="Clip fused SAM masks to dilated bone support; 0/1 means no dilation, -1 disables")
    parser.add_argument("--cam-tta-flip", action="store_true",
                        help="Average LayerCAM from the original and horizontally flipped image (inference-only consistency A/B test).")
    parser.add_argument("--cam-multiscale-sizes", type=str, default="",
                        help="A/B: comma-separated classifier input sizes for a contrastive CAM ensemble, "
                             "e.g. '224,256,288'. Maps are normalized before averaging; empty disables it.")
    parser.add_argument("--cam-aggregation", type=str, default="class",
                        choices=["class", "tumor_union", "tumor_union_contrast", "tumor_union_contrast_class_max"],
                        help="Class-conditioned CAM (default), aggregate CAM for all non-normal logits, "
                             "or aggregate tumor evidence contrasted against the normal logit; "
                             "the *_class_max variant also retains the selected-class contrastive peaks.")
    parser.add_argument("--cam-contrast-normal", action="store_true",
                        help="A/B: for tumor_type, condition LayerCAM on logit(class)-logit(normal) to suppress normal/background evidence.")
    parser.add_argument("--disable-cam-contrast-normal", action="store_true",
                        help="A/B override for the default profile; rejected by btxrd_best.")
    parser.add_argument("--cam-contrast-weight", type=float, default=1.0,
                        help="Blend weight for contrastive CAM when --cam-contrast-normal is enabled (0=original, 1=contrastive).")
    parser.add_argument("--cam-target-class", type=str, default="predicted",
                        choices=["predicted", "ground_truth"],
                        help="Which class LayerCAM is conditioned on. 'predicted' (default) uses the "
                        "classifier's own argmax/top-confidence class, matching real inference where "
                        "the true label is unknown. 'ground_truth' instead conditions LayerCAM on the "
                        "dataset's true image-level label (still just the image-level class -- never "
                        "polygon/bbox annotations, so this stays within WSSS) when generating pseudo "
                        "masks for a labeled split. For --target-columns tumor_type it selects the "
                        "known 10-class target; for --target-columns tumor it selects the known positive "
                        "tumor logit and emits an empty mask for normal images. Only meaningful when the split actually has GT labels (i.e. not "
                        "at real deployment time on unlabeled images).")
    parser.add_argument("--debug", action="store_true",
                        help="Save per-image debug outputs (SAM masks, prompt overlays, scores)")
    parser.add_argument("--evaluate-prompt-quality", action="store_true",
        help="Removed leakage-prone legacy mode. Use --save-candidate-diagnostics, "
             "freeze the generated manifests, then use the final prediction-first evaluator.")
    parser.add_argument(
        "--save-candidate-diagnostics",
        action="store_true",
        help=(
            "Save prediction-only candidates/prompts/stage masks for every image-level-positive "
            "case. No segmentation GT is loaded; evaluate them only after the manifests freeze."
        ),
    )
    parser.add_argument(
        "--candidate-diagnostics-cohort",
        choices=["tumor", "all"],
        default="tumor",
        help=(
            "Save proposal bags for only image-level-positive cases (legacy) or "
            "for the complete split. The all mode is intended for image-label MIL "
            "and still never loads segmentation annotations."
        ),
    )
    parser.add_argument(
        "--force-normal-candidate-gallery",
        action="store_true",
        help=(
            "With binary ground-truth CAM targeting, generate tumor-class CAM/SAM "
            "proposals for image-level-normal cases instead of an empty bag. This "
            "is only for negative bags in image-label MIL; resulting pseudo masks "
            "must not be used as foreground supervision."
        ),
    )
    parser.add_argument("--force-non-normal-cam", action="store_true",
                        help="Predicted-protocol A/B: if argmax is normal, condition CAM on the strongest "
                             "non-normal class instead of skipping. Default keeps normal images empty.")
    parser.add_argument(
        "--low-score-policy",
        choices=["empty", "keep-best"],
        default="empty",
        help="What to do when every SAM candidate scores below --mask-score-threshold. "
        "Use 'empty' for production; 'keep-best' is retained only for debug ablations.",
    )
    args = parser.parse_args()
    args._explicit_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args


def apply_pipeline_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the tested pipeline configuration without using segmentation GT.

    The default CLI remains available for diagnostics, while
    ``--pipeline-profile btxrd_best`` or ``btxrd_hybrid`` makes the exact BTXRD
    configuration used in the best validation run reproducible.  Recipe-
    critical CLI changes are rejected under this profile; only dataset paths,
    protocol selection, output paths, checkpoint paths, and device placement
    remain machine/run inputs.
    """
    if args.pipeline_profile == "default":
        return args
    if args.dataset != "btxrd":
        raise ValueError(f"--pipeline-profile {args.pipeline_profile} requires BTXRD")

    explicit = getattr(args, "_explicit_options", set())

    # A checkpoint is deliberately required for the frozen profile.  The
    # checkpoint is the artifact produced by the separate training run, so
    # silently selecting a stale local model would make a cross-machine run
    # irreproducible.
    if "--classifier-checkpoint" not in explicit:
        raise ValueError(
            f"--pipeline-profile {args.pipeline_profile} requires an explicit "
            "--classifier-checkpoint from the current training run"
        )

    best_sam = ROOT.parent / "sam_vit_b_01ec64.pth"
    if "--sam-checkpoint" not in explicit and best_sam.exists():
        args.sam_checkpoint = best_sam

    profile = (
        BTXRD_HYBRID_PIPELINE
        if args.pipeline_profile == BTXRD_HYBRID_PIPELINE.name
        else BTXRD_BEST_PIPELINE
    )

    def require_or_set(option: str, attribute: str, expected: object) -> None:
        if (
            option == "--component-topk"
            and args.allow_validation_component_topk_ablation
            and args.split == "val"
            and option in explicit
        ):
            return
        if option in explicit and getattr(args, attribute) != expected:
            raise ValueError(
                f"--pipeline-profile {args.pipeline_profile} fixes {option}={expected!r}; "
                f"received {getattr(args, attribute)!r}"
            )
        setattr(args, attribute, expected)

    if args.allow_validation_component_topk_ablation and args.split != "val":
        raise ValueError("component_topk ablation is allowed only on the validation split")

    require_or_set("--target-columns", "target_columns", ",".join(profile.target_columns))
    require_or_set("--image-size", "image_size", profile.classifier_image_size)
    require_or_set("--sam-image-size", "sam_image_size", profile.sam_image_size)
    require_or_set("--cam-percentile", "cam_percentile", profile.cam_percentile)
    require_or_set(
        "--cam-percentile-values",
        "cam_percentile_values",
        ",".join(str(int(value)) for value in profile.cam_percentile_values),
    )
    require_or_set("--max-components", "max_components", profile.max_bone_components)
    require_or_set("--component-topk", "component_topk", profile.component_topk)
    require_or_set("--points-per-component", "points_per_component", profile.points_per_component)
    require_or_set("--bbox-padding-ratio", "bbox_padding_ratio", profile.bbox_padding_ratio)
    require_or_set("--negative-points-per-component", "negative_points_per_component", profile.negative_points_per_component)
    require_or_set("--prompt-border-margin", "prompt_border_margin", profile.prompt_border_margin)
    require_or_set("--max-box-area-ratio", "max_box_area_ratio", profile.max_box_area_ratio)
    require_or_set("--selection-method", "selection_method", profile.selection_method)
    require_or_set("--fusion-topk", "fusion_topk", profile.fusion_topk)
    require_or_set("--support-clip-kernel", "support_clip_kernel", profile.support_clip_kernel)
    require_or_set("--confidence-threshold", "confidence_threshold", profile.confidence_threshold)
    require_or_set("--max-points", "max_points", profile.max_points)
    require_or_set("--min-component-area", "min_component_area", profile.min_component_area)
    require_or_set("--mask-score-threshold", "mask_score_threshold", profile.mask_score_threshold)
    require_or_set("--low-score-policy", "low_score_policy", "empty")
    require_or_set("--morphology-fusion-mode", "morphology_fusion_mode", profile.morphology_fusion_mode)
    require_or_set("--layercam-gradient-mode", "layercam_gradient_mode", profile.layercam_gradient_mode)
    require_or_set("--cam-aggregation", "cam_aggregation", profile.cam_aggregation)
    require_or_set("--cam-contrast-weight", "cam_contrast_weight", profile.cam_contrast_weight)
    require_or_set("--closing-kernel", "closing_kernel", profile.closing_kernel)
    require_or_set("--opening-kernel", "opening_kernel", profile.opening_kernel)
    require_or_set("--min-size", "min_size", profile.min_size)
    require_or_set("--max-hole-area", "max_hole_area", profile.max_hole_area)
    require_or_set("--guidance-threshold", "guidance_threshold", profile.guidance_threshold)
    require_or_set("--preprocessing-mode", "preprocessing_mode", "none")
    require_or_set("--cam-multiscale-sizes", "cam_multiscale_sizes", "")
    require_or_set(
        "--layercam-weights",
        "layercam_weights",
        ",".join(str(value) for value in profile.layercam_weights),
    )
    require_or_set("--sam-prompt-mode", "sam_prompt_mode", profile.sam_prompt_mode)

    # Follow the available accelerator on the new workstation.  This is the
    # only profile field intentionally left hardware-dependent.
    if "--sam-device" not in explicit:
        args.sam_device = "auto"

    locked_true = {
        "--cam-percentile-ensemble": "cam_percentile_ensemble",
        "--sam-prompt-ensemble": "sam_prompt_ensemble",
        "--all-cam-components": "all_cam_components",
        "--cam-contrast-normal": "cam_contrast_normal",
    }
    for option, attribute in locked_true.items():
        if option in explicit and not getattr(args, attribute):
            raise ValueError(f"--pipeline-profile {args.pipeline_profile} requires {option}")
        setattr(args, attribute, True)

    forbidden_disable_flags = {
        "--disable-cam-percentile-ensemble",
        "--disable-sam-prompt-ensemble",
        "--disable-all-cam-components",
        "--disable-cam-contrast-normal",
    }
    for option in forbidden_disable_flags:
        if option in explicit:
            raise ValueError(f"--pipeline-profile {args.pipeline_profile} rejects {option}")

    locked_false = {
        "--sam-preserve-aspect": "sam_preserve_aspect",
        "--sam-single-mask": "sam_single_mask",
        "--include-cam-candidate": "include_cam_candidate",
        "--disable-morphology": "disable_morphology",
        "--cam-tta-flip": "cam_tta_flip",
        "--disable-best-per-component": "disable_best_per_component",
        "--force-non-normal-cam": "force_non_normal_cam",
        "--use-clahe": "use_clahe",
    }
    for option, attribute in locked_false.items():
        if option in explicit and getattr(args, attribute):
            raise ValueError(f"--pipeline-profile {args.pipeline_profile} fixes {option} off")
        setattr(args, attribute, False)

    if "--auxiliary-binary-checkpoint" in explicit:
        raise ValueError(f"--pipeline-profile {args.pipeline_profile} does not use an auxiliary binary checkpoint")

    return args


def load_classifier(
    checkpoint_path: Path,
    fallback_num_classes: int,
    device: torch.device,
    expected_target_columns: list[str] | None = None,
    expected_task: str | None = None,
    expected_num_classes: int | None = None,
    expected_split_manifest: Path | None = None,
    expected_pipeline_profile: str | None = None,
) -> tuple[torch.nn.Module, str, str]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_target_columns = state.get("target_columns")
    checkpoint_task = state.get("task", "multi-label")
    checkpoint_num_classes = state.get("num_classes", fallback_num_classes)
    checkpoint_pipeline_profile = state.get("pipeline_profile")
    if (
        expected_pipeline_profile is not None
        and checkpoint_pipeline_profile != expected_pipeline_profile
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} has pipeline_profile={checkpoint_pipeline_profile!r}; "
            f"the selected pipeline requires {expected_pipeline_profile!r}."
        )
    if expected_target_columns is not None and checkpoint_target_columns != expected_target_columns:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has target_columns={checkpoint_target_columns!r}; "
            f"the selected pipeline requires {expected_target_columns!r}."
        )
    if expected_task is not None and checkpoint_task != expected_task:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has task={checkpoint_task!r}; "
            f"the selected pipeline requires {expected_task!r}."
        )
    if expected_num_classes is not None and checkpoint_num_classes != expected_num_classes:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has num_classes={checkpoint_num_classes!r}; "
            f"the selected pipeline requires {expected_num_classes}."
        )
    if expected_split_manifest is not None:
        manifest_path = expected_split_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
        expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        checkpoint_hash = state.get("split_manifest_sha256")
        if checkpoint_hash != expected_hash:
            raise ValueError(
                f"Classifier checkpoint {checkpoint_path} split manifest hash does not match "
                "the requested manifest"
            )
    num_classes = checkpoint_num_classes
    if checkpoint_pipeline_profile == "s2c_cpm_fpn_v1":
        cpm_config = state.get("s2c_cpm")
        if not isinstance(cpm_config, dict):
            raise ValueError("S2C CPM checkpoint is missing its frozen configuration")
        if (
            checkpoint_target_columns != ["tumor"]
            or checkpoint_task != "multi-label"
            or checkpoint_num_classes != 1
        ):
            raise ValueError("S2C CPM checkpoint must be a one-logit tumor classifier")
        model = DenseNet121S2CCPMClassifier(
            pretrained=False,
            feature_channels=int(cpm_config["feature_channels"]),
        )
    else:
        model = DenseNet121AnatomyClassifier(
            num_classes=num_classes,
            pretrained=False,
        )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_task, state.get("normalization", "imagenet")


def load_proposal_teacher(
    checkpoint_path: Path,
    *,
    expected_sha256: str | None,
    expected_split_manifest: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load an image-label-only segmentation teacher under a fail-closed contract."""
    checkpoint_path = checkpoint_path.resolve()
    expected_hash = str(expected_sha256 or "").strip().lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError(
            "--proposal-teacher-expected-sha256 must be a 64-character hexadecimal digest"
        )
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Proposal-teacher checkpoint SHA-256 mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0 compatibility for local provenance checks
        state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model_state_dict"), dict):
        raise ValueError("Proposal-teacher checkpoint is missing model_state_dict")
    if state.get("dataset") != "btxrd":
        raise ValueError("Proposal teacher must be a BTXRD checkpoint")
    if int(state.get("image_size", -1)) != 448:
        raise ValueError("Proposal teacher must use the frozen 448px WSL consumer contract")
    architecture = architecture_name_from_metadata(state.get("architecture"))
    if architecture != "resnet18_unet":
        raise ValueError("Proposal teacher must be the frozen ResNet18UNet consumer")
    if not state.get("train_pred_mask_root"):
        raise ValueError("Proposal teacher is not pseudo-mask trained")
    pseudo_manifest_hash = str(
        state.get("train_pseudo_mask_manifest_sha256") or ""
    ).strip().lower()
    if len(pseudo_manifest_hash) != 64:
        raise ValueError("Proposal teacher lacks a valid train pseudo-mask manifest hash")
    if state.get("val_pred_mask_root") is not None:
        raise ValueError("Proposal teacher validation must use GT masks, not pseudo masks")
    if expected_split_manifest is None:
        raise ValueError("Proposal teacher requires --split-manifest")
    expected_split_hash = sha256_file(expected_split_manifest.resolve())
    if state.get("split_manifest_sha256") != expected_split_hash:
        raise ValueError(
            "Proposal-teacher split hash differs from the requested frozen split"
        )

    resolved = state.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("Proposal teacher lacks its resolved training configuration")
    required_contract = {
        "dataset": "btxrd",
        "image_size": 448,
        "model_architecture": "resnet18_unet",
        "seed": 42,
        "val_pred_mask_root": None,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in required_contract.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Proposal-teacher consumer contract mismatch: {mismatches}")

    model = build_segmentation_model(architecture, pretrained=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "architecture": architecture,
        "image_size": 448,
        "split_manifest_sha256": expected_split_hash,
        "train_pseudo_mask_manifest_sha256": pseudo_manifest_hash,
        "supervision": "image_labels_via_pseudo_masks_only",
        "test_evaluated": False,
        "proposal_role": "add_prompt_components_only",
    }


def classifier_class_weights(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "single-label":
        return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def parse_prompt_score_weights(value: str) -> tuple[float, float, float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 5:
        raise ValueError(
            "--prompt-score-weights requires exactly five comma-separated values: "
            "CAM coverage, CAM density, SAM rank, area, prompt consistency."
        )
    weights = tuple(float(part) for part in parts)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("--prompt-score-weights must be non-negative and sum to > 0.")
    return weights  # type: ignore[return-value]


def parse_layercam_weights(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--layercam-weights requires three comma-separated values: denseblock2,3,4")
    weights = tuple(float(part) for part in parts)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("--layercam-weights must be non-negative and sum to > 0.")
    return weights  # type: ignore[return-value]


def parse_cam_percentile_values(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--cam-percentile-values requires at least one percentile")
    percentiles = tuple(float(part) for part in parts)
    if any(percentile <= 0.0 or percentile >= 100.0 for percentile in percentiles):
        raise ValueError("--cam-percentile-values must be strictly between 0 and 100")
    return percentiles


def parse_cam_multiscale_sizes(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    sizes = tuple(int(part) for part in parts)
    if any(size < 32 for size in sizes):
        raise ValueError("--cam-multiscale-sizes values must be >= 32")
    return sizes


def should_skip_tumor_type(
    class_weights: np.ndarray,
    use_ground_truth_class: bool,
    ground_truth_class: int | None,
) -> bool:
    """Return True when the selected image-level class is ``normal``.

    Both localization protocols must map class 0 to an empty pseudo-mask:
    ground_truth mode uses the known image-level class, while predicted mode
    uses the classifier argmax. Generating a LayerCAM for the semantic class
    ``normal`` and treating it as a lesion CAM would invalidate end-to-end
    specificity and mix detection errors with segmentation errors.
    """
    selected_class = (
        int(ground_truth_class)
        if use_ground_truth_class and ground_truth_class is not None
        else int(np.argmax(class_weights))
    )
    return selected_class == 0


def write_or_validate_run_metadata(
    output_dir: Path,
    metadata: dict[str, object],
    *,
    allow_existing: bool = False,
) -> None:
    """Prevent predicted/ground-truth protocol outputs from sharing a mask directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    mask_dir = output_dir / "masks"
    has_existing_masks = mask_dir.exists() and any(mask_dir.glob("*.png"))
    if has_existing_masks and not allow_existing:
        raise FileExistsError(
            f"Refusing to overwrite existing pseudo masks under {mask_dir}. "
            "Use a fresh --output-dir, or pass --overwrite-existing deliberately."
        )
    if has_existing_masks and not metadata_path.exists():
        raise RuntimeError(
            f"Refusing to reuse {output_dir}: it already contains masks with unknown protocol "
            "provenance (no run_metadata.json). Use a fresh --output-dir."
        )
    if metadata_path.exists() and has_existing_masks:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in metadata.items()
            if existing.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: existing={old!r}, requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                "Refusing to mix pseudo masks from different protocols/configurations in "
                f"{output_dir}. Use a fresh --output-dir. Differences: {details}"
            )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_saliency_contract(
    *,
    manifest_path: Path | None,
    metadata_path: Path | None,
    expected_manifest_sha256: str | None,
    expected_metadata_sha256: str | None,
    expected_source_commit: str | None,
    expected_model_weight_sha256: str | None,
    split: str,
    split_manifest_sha256: str | None,
    image_size: int,
) -> tuple[dict[str, dict[str, str]], dict[str, object] | None]:
    supplied = [
        manifest_path,
        metadata_path,
        expected_manifest_sha256,
        expected_metadata_sha256,
        expected_source_commit,
        expected_model_weight_sha256,
    ]
    if not any(value is not None for value in supplied):
        return {}, None
    if any(value is None for value in supplied):
        raise ValueError(
            "External saliency requires manifest, run metadata and all four "
            "expected manifest/metadata/source/model hashes"
        )
    assert manifest_path is not None and metadata_path is not None
    assert expected_manifest_sha256 is not None and expected_metadata_sha256 is not None
    assert expected_source_commit is not None and expected_model_weight_sha256 is not None
    manifest_path = manifest_path.resolve()
    metadata_path = metadata_path.resolve()
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("External saliency manifest SHA-256 mismatch")
    if sha256_file(metadata_path) != expected_metadata_sha256:
        raise ValueError("External saliency metadata SHA-256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("stage") != "prediction-first BiomedCLIP saliency generation":
        raise ValueError("External saliency stage is not the frozen prediction-first stage")
    if metadata.get("supervision") != "images and binary image-level labels only":
        raise ValueError("External saliency supervision contract mismatch")
    if metadata.get("source_commit") != expected_source_commit:
        raise ValueError("External saliency source commit mismatch")
    if metadata.get("split") != split:
        raise ValueError("External saliency split mismatch")
    if metadata.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("External saliency split-manifest hash mismatch")
    if metadata.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("External saliency metadata does not lock the manifest")
    if metadata.get("model", {}).get("weight_sha256") != expected_model_weight_sha256:
        raise ValueError("External saliency model-weight hash mismatch")
    if metadata.get("validation_gt_read") is not False:
        raise ValueError("External saliency generation accessed validation GT")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("External saliency generation accessed test")
    if metadata.get("view_contract", {}).get("output_size") != image_size:
        raise ValueError("External saliency grid differs from --image-size")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("External saliency manifest contains duplicate image IDs")
    contract = {
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "metadata": str(metadata_path),
        "metadata_sha256": expected_metadata_sha256,
        "source_commit": expected_source_commit,
        "source_files": metadata.get("source_files"),
        "model": metadata.get("model"),
        "prompts": metadata.get("prompts"),
        "view_contract": metadata.get("view_contract"),
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    return indexed, contract


def load_external_saliency_map(
    record: dict[str, str],
    *,
    root: Path,
    expected_image_id: str,
    expected_image_label: int,
    image_size: int,
) -> np.ndarray:
    if record.get("image_id") != expected_image_id:
        raise ValueError("External saliency image identity mismatch")
    if int(record.get("tumor_image_label", "-1")) != expected_image_label:
        raise ValueError(f"External saliency image-label mismatch: {expected_image_id}")
    expected_relative = Path("maps") / f"{Path(expected_image_id).stem}.npy"
    relative = Path(record.get("map_path", ""))
    if relative != expected_relative:
        raise ValueError(f"External saliency path mismatch: {expected_image_id}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("External saliency map escapes artifact root") from error
    if sha256_file(path) != record.get("map_sha256"):
        raise ValueError(f"External saliency map SHA-256 mismatch: {expected_image_id}")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float16 or values.shape != (image_size, image_size):
        raise ValueError(f"External saliency map dtype/shape mismatch: {expected_image_id}")
    values = values.astype(np.float32)
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError(f"External saliency map value range mismatch: {expected_image_id}")
    if expected_image_label == 0 and np.count_nonzero(values) != 0:
        raise ValueError(f"Known-normal external saliency is nonempty: {expected_image_id}")
    if expected_image_label == 1 and float(values.max() - values.min()) <= 1e-6:
        raise ValueError(f"Tumor external saliency is constant: {expected_image_id}")
    return values


def build_external_saliency_proposal_gallery(
    *,
    image_rgb: np.ndarray,
    saliency_map: np.ndarray,
    prompt_percentiles: tuple[float, ...],
    min_component_area: int,
    max_components: int,
    all_cam_components: bool,
    points_per_component: int,
    bbox_padding_ratio: float,
    negative_points_per_component: int,
) -> list:
    """Build the same component gallery used by external-saliency replacement.

    Component IDs are made contiguous across percentile views. The caller
    applies a second offset before appending this gallery to the LayerCAM
    gallery, so existing LayerCAM proposals and source ordering are preserved.
    """
    components = []
    for prompt_percentile in prompt_percentiles:
        _, _, local_components = morphology.build_class_conditioned_components(
            image_rgb,
            [saliency_map],
            [1.0],
            cam_percentile=prompt_percentile,
            min_component_area=max(20, min_component_area // 2),
            max_components=(max_components if all_cam_components else 1),
            points_per_component=points_per_component,
            bbox_padding_ratio=bbox_padding_ratio,
            negative_points_per_component=negative_points_per_component,
            debug_dir=None,
        )
        offset = len(components)
        components.extend(
            replace(
                component,
                component_id=offset + int(component.component_id),
            )
            for component in local_components
        )
    maximum = max_components * len(prompt_percentiles)
    if all_cam_components and len(components) > maximum:
        components = components[:maximum]
    return components


def tensor_to_rgb_numpy(image_tensor: torch.Tensor, normalization: str = "imagenet") -> np.ndarray:
    """Convert a [3,H,W] normalised tensor to [H,W,3] uint8 RGB numpy for SAM."""
    pil = tensor_to_pil(image_tensor.detach().cpu(), normalization=normalization)
    return np.array(pil, dtype=np.uint8)


def classifier_candidate_causal_scores(
    classifier: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_logits: torch.Tensor,
    candidate_masks: np.ndarray,
    *,
    candidate_batch_size: int = 8,
    blur_kernel: int = 31,
    feather_kernel: int = 7,
) -> np.ndarray:
    """Measure proposal-specific deletion and insertion evidence.

    Only the frozen image-level classifier is used. Candidate pixels are
    replaced with a deterministic blurred version of the same radiograph for
    deletion; insertion keeps the candidate over that blurred baseline.
    ``mask_selection`` later converts these values to within-component ranks,
    so raw classifier calibration never crosses images.
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Causal candidate scoring expects one image tensor")
    if original_logits.ndim != 2 or tuple(original_logits.shape) != (1, 1):
        raise ValueError(
            "coverage_mass_sam_causal requires a one-logit binary classifier"
        )
    if candidate_masks.ndim != 3:
        raise ValueError("candidate_masks must have shape [N,H,W]")
    if candidate_masks.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    for name, value in (("blur_kernel", blur_kernel), ("feather_kernel", feather_kernel)):
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")

    device = image_tensor.device
    target_size = tuple(int(value) for value in image_tensor.shape[-2:])
    mask_tensor = torch.from_numpy(candidate_masks.astype(np.float32))[:, None]
    if tuple(mask_tensor.shape[-2:]) != target_size:
        mask_tensor = F.interpolate(mask_tensor, size=target_size, mode="nearest")
    mask_tensor = mask_tensor.to(device)
    if feather_kernel > 1:
        mask_tensor = F.avg_pool2d(
            mask_tensor,
            kernel_size=feather_kernel,
            stride=1,
            padding=feather_kernel // 2,
        ).clamp_(0.0, 1.0)
    blur_pad = blur_kernel // 2
    padded = F.pad(image_tensor, (blur_pad,) * 4, mode="reflect")
    blurred = F.avg_pool2d(padded, kernel_size=blur_kernel, stride=1)
    original_logit = original_logits.detach()[0, 0]
    with torch.no_grad():
        baseline_logit = classifier(blurred)[0, 0]
        causal_scores: list[torch.Tensor] = []
        for start in range(0, mask_tensor.shape[0], candidate_batch_size):
            masks = mask_tensor[start:start + candidate_batch_size]
            image_batch = image_tensor.expand(masks.shape[0], -1, -1, -1)
            blurred_batch = blurred.expand_as(image_batch)
            removed = image_batch * (1.0 - masks) + blurred_batch * masks
            inserted = image_batch * masks + blurred_batch * (1.0 - masks)
            paired_logits = classifier(torch.cat([removed, inserted], dim=0))[:, 0]
            batch_size = masks.shape[0]
            deletion = torch.relu(original_logit - paired_logits[:batch_size])
            insertion = torch.relu(paired_logits[batch_size:] - baseline_logit)
            causal_scores.append(0.5 * (deletion + insertion))
    return torch.cat(causal_scores).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = apply_pipeline_profile(parse_args())
    if args.evaluate_prompt_quality:
        raise ValueError(
            "--evaluate-prompt-quality was removed because it loaded validation GT before "
            "predictions were frozen. Generate with --save-candidate-diagnostics, freeze "
            "the final choices, and evaluate only against the locked manifests."
        )
    frozen_test_document = verify_frozen_test_config(
        args.frozen_config,
        split=args.split,
        split_manifest=args.split_manifest,
        requested_artifacts={
            "classifier_checkpoint": args.classifier_checkpoint,
            "sam_checkpoint": args.sam_checkpoint,
        },
    )
    prompt_score_weights = parse_prompt_score_weights(args.prompt_score_weights)
    layercam_weights = parse_layercam_weights(args.layercam_weights)
    cam_percentile_values = parse_cam_percentile_values(args.cam_percentile_values)
    cam_multiscale_sizes = parse_cam_multiscale_sizes(args.cam_multiscale_sizes)
    if args.target_columns is None:
        target_columns = list(DATASET_TARGET_COLUMNS[args.dataset])
    else:
        target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]

    if args.force_normal_candidate_gallery and (
        not args.save_candidate_diagnostics
        or args.candidate_diagnostics_cohort != "all"
        or args.cam_target_class != "ground_truth"
        or target_columns != ["tumor"]
    ):
        raise ValueError(
            "--force-normal-candidate-gallery is restricted to prediction-first "
            "full-cohort diagnostics with binary image-label ground-truth CAM "
            "targeting"
        )

    if target_columns == ["tumor_type"]:
        from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
        class_names = list(TUMOR_TYPE_CLASS_NAMES)
    else:
        class_names = target_columns

    default_seed_percentile, default_support_percentile = (82.0, 55.0)
    seed_percentile = (
        args.seed_percentile if args.seed_percentile is not None else default_seed_percentile
    )
    support_percentile = (
        args.support_percentile if args.support_percentile is not None else default_support_percentile
    )

    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = default_device if args.classifier_device == "auto" else torch.device(args.classifier_device)
    sam_device = device if args.sam_device == "auto" else torch.device(args.sam_device)
    validate_runtime_device(device, "--classifier-device")
    validate_runtime_device(sam_device, "--sam-device")
    canonical_profile = {
        BTXRD_BEST_PIPELINE.name: BTXRD_BEST_PIPELINE,
        BTXRD_HYBRID_PIPELINE.name: BTXRD_HYBRID_PIPELINE,
    }.get(args.pipeline_profile)
    expected_profile_columns = (
        list(canonical_profile.target_columns) if canonical_profile is not None else None
    )
    classifier, classifier_task, classifier_normalization = load_classifier(
        args.classifier_checkpoint,
        len(target_columns),
        device,
        expected_target_columns=expected_profile_columns,
        expected_task="single-label" if expected_profile_columns is not None else None,
        expected_num_classes=10 if expected_profile_columns is not None else None,
        expected_split_manifest=(args.classifier_split_manifest or args.split_manifest),
        expected_pipeline_profile=(
            canonical_profile.name if canonical_profile is not None else None
        ),
    )
    print(f"Loaded classifier checkpoint task={classifier_task} normalization={classifier_normalization}")
    proposal_teacher = None
    proposal_teacher_info = None
    if args.proposal_teacher_segmentation_checkpoint is not None:
        if args.sam_grid_gallery:
            raise ValueError("Proposal-teacher components are incompatible with --sam-grid-gallery")
        if args.disable_morphology or args.morphology_fusion_mode != "components":
            raise ValueError(
                "Proposal teacher requires enabled component-mode morphology"
            )
        if target_columns != ["tumor"] or args.cam_target_class != "ground_truth":
            raise ValueError(
                "The frozen proposal-teacher experiment requires tumor image labels "
                "and --cam-target-class ground_truth"
            )
        if classifier_normalization != "imagenet":
            raise ValueError("Proposal teacher requires ImageNet-normalized inputs")
        if not 0.0 <= args.proposal_teacher_threshold <= 1.0:
            raise ValueError("--proposal-teacher-threshold must be in [0,1]")
        if (
            args.proposal_teacher_min_component_area <= 0
            or args.proposal_teacher_max_components <= 0
        ):
            raise ValueError("Proposal-teacher component limits must be positive")
        proposal_teacher, proposal_teacher_info = load_proposal_teacher(
            args.proposal_teacher_segmentation_checkpoint,
            expected_sha256=args.proposal_teacher_expected_sha256,
            expected_split_manifest=(args.classifier_split_manifest or args.split_manifest),
            device=device,
        )
        print(
            "Loaded frozen pseudo-trained proposal teacher "
            f"sha256={proposal_teacher_info['checkpoint_sha256']}"
        )
    elif args.proposal_teacher_expected_sha256 is not None:
        raise ValueError(
            "--proposal-teacher-expected-sha256 requires "
            "--proposal-teacher-segmentation-checkpoint"
        )
    external_saliency_rows, external_saliency_contract = load_external_saliency_contract(
        manifest_path=args.external_saliency_manifest,
        metadata_path=args.external_saliency_run_metadata,
        expected_manifest_sha256=args.external_saliency_expected_manifest_sha256,
        expected_metadata_sha256=args.external_saliency_expected_metadata_sha256,
        expected_source_commit=args.external_saliency_expected_source_commit,
        expected_model_weight_sha256=args.external_saliency_expected_model_weight_sha256,
        split=args.split,
        split_manifest_sha256=(
            sha256_file(args.split_manifest.resolve()) if args.split_manifest else None
        ),
        image_size=args.image_size,
    )
    external_saliency_root = (
        args.external_saliency_run_metadata.resolve().parent
        if external_saliency_contract is not None
        else None
    )
    affinity_selector_rows, affinity_selector_contract = (
        load_affinity_selector_contract(
            manifest_path=args.affinity_selector_manifest,
            package_metadata_path=args.affinity_selector_package_metadata,
            prediction_freeze_path=args.affinity_selector_prediction_freeze,
            expected_manifest_sha256=(
                args.affinity_selector_expected_manifest_sha256
            ),
            expected_package_metadata_sha256=(
                args.affinity_selector_expected_package_metadata_sha256
            ),
            expected_prediction_freeze_sha256=(
                args.affinity_selector_expected_prediction_freeze_sha256
            ),
            expected_source_commit=args.affinity_selector_expected_source_commit,
            expected_protocol_sha256=args.affinity_selector_expected_protocol_sha256,
            expected_checkpoint_sha256=(
                args.affinity_selector_expected_checkpoint_sha256
            ),
            split=args.split,
            split_manifest_sha256=(
                sha256_file(args.split_manifest.resolve())
                if args.split_manifest
                else None
            ),
            image_size=args.image_size,
        )
    )
    affinity_selector_root = (
        args.affinity_selector_manifest.resolve().parent
        if affinity_selector_contract is not None
        else None
    )
    if external_saliency_contract is not None:
        if not external_saliency_test_authorized(args.split, frozen_test_document):
            raise ValueError("External-saliency test generation remains locked")
        if target_columns != ["tumor"] or args.cam_target_class != "ground_truth":
            raise ValueError(
                "External saliency requires binary image labels and "
                "--cam-target-class ground_truth"
            )
        if args.disable_morphology or args.morphology_fusion_mode != "components":
            raise ValueError("External saliency requires component-mode morphology")
        if proposal_teacher is not None or args.auxiliary_binary_checkpoint is not None:
            raise ValueError(
                "The first external-saliency diagnostic isolates one localization "
                "source and cannot mix proposal-teacher/auxiliary CAM evidence"
            )
    elif args.external_saliency_role != "replace":
        raise ValueError(
            "--external-saliency-role proposal_gallery requires the complete "
            "hash-locked external-saliency contract"
        )
    if args.selection_method == "prompt_source_graph":
        if (
            external_saliency_contract is None
            or args.external_saliency_role != "proposal_gallery"
        ):
            raise ValueError(
                "prompt_source_graph requires the hash-locked external saliency "
                "proposal gallery"
            )
        if not args.sam_prompt_ensemble or args.disable_sam_prompt_ensemble:
            raise ValueError(
                "prompt_source_graph requires the fixed multi-prompt SAM ensemble"
            )
        if args.component_topk <= 0:
            raise ValueError(
                "prompt_source_graph requires a positive component_topk cluster cap"
            )
    if args.selection_method == "affinity_rank_single":
        if affinity_selector_contract is None:
            raise ValueError(
                "affinity_rank_single requires the complete frozen affinity "
                "selector input contract"
            )
        if (
            external_saliency_contract is None
            or args.external_saliency_role != "proposal_gallery"
        ):
            raise ValueError(
                "affinity_rank_single requires the frozen LayerCAM plus "
                "external-saliency proposal gallery"
            )
        if args.split == "test":
            raise ValueError("Affinity-selector test generation remains locked")
        if target_columns != ["tumor"] or args.cam_target_class != "ground_truth":
            raise ValueError(
                "affinity_rank_single requires supplied binary image labels"
            )
        if (
            args.fusion_topk != 1
            or not args.disable_best_per_component
            or args.support_clip_kernel >= 0
            or abs(float(args.mask_score_threshold)) > 1e-12
            or args.closing_kernel != 0
            or args.opening_kernel != 0
            or args.min_size != 1
            or args.max_hole_area != 0
            or abs(float(args.guidance_threshold)) > 1e-12
        ):
            raise ValueError(
                "affinity_rank_single requires one raw global candidate with "
                "no clipping or morphology"
            )
    elif affinity_selector_contract is not None:
        raise ValueError(
            "Frozen affinity selector input is valid only with "
            "selection_method=affinity_rank_single"
        )

    run_metadata = {
            "pipeline_profile": args.pipeline_profile,
            "dataset": args.dataset,
            "split": args.split,
            "target_columns": target_columns,
            "cam_target_class": args.cam_target_class,
            "force_non_normal_cam": args.force_non_normal_cam,
            "candidate_diagnostics_cohort": args.candidate_diagnostics_cohort,
            "force_normal_candidate_gallery": args.force_normal_candidate_gallery,
            "force_normal_candidate_gallery_semantics": (
                "tumor-class proposals are retained only as negative MIL bag "
                "instances; saved normal pseudo masks remain empty"
                if args.force_normal_candidate_gallery
                else None
            ),
            "low_score_policy": args.low_score_policy,
            "classifier_task": classifier_task,
            "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
            "classifier_checkpoint_sha256": sha256_file(args.classifier_checkpoint.resolve()),
            "proposal_teacher": proposal_teacher_info,
            "proposal_teacher_threshold": (
                args.proposal_teacher_threshold if proposal_teacher is not None else None
            ),
            "proposal_teacher_min_component_area": (
                args.proposal_teacher_min_component_area if proposal_teacher is not None else None
            ),
            "proposal_teacher_max_components": (
                args.proposal_teacher_max_components if proposal_teacher is not None else None
            ),
            "proposal_teacher_semantics": (
                (
                    "proposal_components_plus_source_consensus_scoring; "
                    "CAM support clipping unchanged"
                    if args.selection_method == "source_consensus"
                    else "proposal_components_only; CAM scoring and support clipping unchanged"
                )
                if proposal_teacher is not None else None
            ),
            "external_saliency": external_saliency_contract,
            "external_saliency_role": (
                args.external_saliency_role
                if external_saliency_contract is not None else None
            ),
            "external_saliency_semantics": (
                (
                    "replace_layercam"
                    if args.external_saliency_role == "replace"
                    else (
                        (
                            "append_component_sam_proposals; prompt-stable "
                            "source-consensus graph selector; layercam support "
                            "and post-processing unchanged"
                        )
                        if args.selection_method == "prompt_source_graph"
                        else (
                            (
                                "append_component_sam_proposals_only; frozen "
                                "affinity-rank raw single-candidate selector"
                            )
                            if args.selection_method == "affinity_rank_single"
                            else (
                                "append_component_sam_proposals_only; layercam selector, "
                                "support and post-processing unchanged"
                            )
                        )
                    )
                )
                if external_saliency_contract is not None else None
            ),
            "affinity_selector": affinity_selector_contract,
            "affinity_selector_semantics": (
                (
                    "fixed top-20/15/10/5/3/1-percent rank supports; "
                    "lexicographic variable-area raw single-candidate selection; "
                    "no validation-GT input, scalar-weight fitting, union, support "
                    "clipping or boundary-changing post-processing"
                )
                if affinity_selector_contract is not None
                else None
            ),
            "auxiliary_binary_checkpoint": (
                str(args.auxiliary_binary_checkpoint.resolve())
                if args.auxiliary_binary_checkpoint else None
            ),
            "auxiliary_binary_weight": args.auxiliary_binary_weight,
            "image_size": args.image_size,
            "sam_image_size": args.sam_image_size,
            "sam_preserve_aspect": args.sam_preserve_aspect,
            "image_list": str(args.image_list.resolve()) if args.image_list else None,
            "sam_backend": "sam_v1_vit_b",
            "sam_device": str(sam_device),
            "classifier_device": str(device),
            "layercam_weights": list(layercam_weights),
            "layercam_gradient_mode": args.layercam_gradient_mode,
            "sam_checkpoint": str(args.sam_checkpoint.resolve()) if args.sam_checkpoint else None,
            "sam_checkpoint_sha256": sha256_file(args.sam_checkpoint.resolve()) if args.sam_checkpoint else None,
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
            "split_manifest_sha256": sha256_file(args.split_manifest.resolve()) if args.split_manifest else None,
            "classifier_split_manifest": (
                str(args.classifier_split_manifest.resolve())
                if args.classifier_split_manifest else None
            ),
            "classifier_split_manifest_sha256": (
                sha256_file(args.classifier_split_manifest.resolve())
                if args.classifier_split_manifest else None
            ),
            "sam_prompt_mode": args.sam_prompt_mode,
            "sam_prompt_ensemble": args.sam_prompt_ensemble,
            "sam_grid_gallery": args.sam_grid_gallery,
            "sam_grid_points_per_side": args.sam_grid_points_per_side,
            "sam_grid_points_per_batch": args.sam_grid_points_per_batch,
            "sam_grid_pred_iou_thresh": args.sam_grid_pred_iou_thresh,
            "sam_grid_stability_thresh": args.sam_grid_stability_thresh,
            "sam_grid_box_nms_thresh": args.sam_grid_box_nms_thresh,
            "sam_single_mask": args.sam_single_mask,
            "include_cam_candidate": args.include_cam_candidate,
            "cam_percentile": args.cam_percentile,
            "cam_percentile_ensemble": args.cam_percentile_ensemble,
            "cam_percentile_values": list(cam_percentile_values),
            "confidence_threshold": args.confidence_threshold,
            "cam_tta_flip": args.cam_tta_flip,
            "cam_multiscale_sizes": list(cam_multiscale_sizes),
            "cam_backend": args.cam_backend,
            "cam_aggregation": args.cam_aggregation,
            "cam_contrast_normal": args.cam_contrast_normal,
            "cam_contrast_weight": args.cam_contrast_weight,
            "morphology_fusion_mode": args.morphology_fusion_mode,
            "min_component_area": args.min_component_area,
            "max_components": args.max_components,
            "all_cam_components": args.all_cam_components,
            "points_per_component": args.points_per_component,
            "bbox_padding_ratio": args.bbox_padding_ratio,
            "negative_points_per_component": args.negative_points_per_component,
            "max_box_area_ratio": args.max_box_area_ratio,
            "selection_method": args.selection_method,
            "classifier_causal_scoring": (
                {
                    "evidence": "equal-weight positive deletion and insertion logit deltas",
                    "replacement": "same-image 31px mean blur",
                    "mask_feather_kernel": 7,
                    "candidate_batch_size": 8,
                    "normalization": "within-component percentile rank",
                    "score_weight": 0.20,
                }
                if args.selection_method == "coverage_mass_sam_causal"
                else None
            ),
            "mask_score_threshold": args.mask_score_threshold,
            "fusion_topk": args.fusion_topk,
            "best_per_component": not args.disable_best_per_component,
            "component_topk": args.component_topk,
            "support_clip_kernel": args.support_clip_kernel,
            "prompt_score_weights": list(prompt_score_weights),
            "prompt_area_target": args.prompt_area_target,
            "prompt_area_log_sigma": args.prompt_area_log_sigma,
            "closing_kernel": args.closing_kernel,
            "opening_kernel": args.opening_kernel,
            "min_size": args.min_size,
            "max_hole_area": args.max_hole_area,
            "guidance_threshold": args.guidance_threshold,
        }
    write_or_validate_run_metadata(
        args.output_dir,
        run_metadata,
        allow_existing=args.overwrite_existing,
    )

    dataset = build_classification_dataset(
        root=args.data_root,
        split=args.split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        preprocessing_mode=args.preprocessing_mode,
        normalization=classifier_normalization,
        split_manifest=args.split_manifest,
    )
    if args.image_list is not None:
        requested_names = {
            line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        original_count = len(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples if str(sample["image_id"]) in requested_names
        ]
        if not dataset.samples:
            raise ValueError(f"--image-list {args.image_list} matched no images in split '{args.split}'")
        print(f"Image-list filter: {len(dataset.samples)}/{original_count} samples")
    if external_saliency_contract is not None:
        dataset_names = [str(sample["image_id"]) for sample in dataset.samples]
        if set(dataset_names) != set(external_saliency_rows) or len(
            dataset_names
        ) != len(external_saliency_rows):
            raise ValueError(
                "External saliency cohort differs from the exact pseudo-mask dataset cohort"
            )
    if affinity_selector_contract is not None:
        dataset_names = [str(sample["image_id"]) for sample in dataset.samples]
        if set(dataset_names) != set(affinity_selector_rows) or len(
            dataset_names
        ) != len(affinity_selector_rows):
            raise ValueError(
                "Affinity selector cohort differs from the exact pseudo-mask "
                "dataset cohort"
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    tumor_type_by_name: dict[str, str] = {}
    if args.dataset == "btxrd":
        from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
        for sample in dataset.samples:
            tumor_type_by_name[str(sample["image_id"])] = TUMOR_TYPE_CLASS_NAMES[int(sample["tumor_type"])]

    if args.cam_backend == "s2c_cpm":
        if not isinstance(classifier, DenseNet121S2CCPMClassifier):
            raise ValueError(
                "--cam-backend s2c_cpm requires an s2c_cpm_fpn_v1 checkpoint"
            )
        if target_columns != ["tumor"] or classifier_task != "multi-label":
            raise ValueError("The direct CPM CAM backend requires binary tumor labels")
        if args.auxiliary_binary_checkpoint is not None:
            raise ValueError("The direct CPM CAM backend does not use an auxiliary CAM")
        layercam = S2CCPMDirectCAM(classifier)
    else:
        if isinstance(classifier, DenseNet121S2CCPMClassifier):
            raise ValueError(
                "An s2c_cpm_fpn_v1 checkpoint must use --cam-backend s2c_cpm"
            )
        layercam = LayerCAM(
            classifier,
            device=device,
            layer_weights=layercam_weights,
            gradient_mode=args.layercam_gradient_mode,
        )
    auxiliary_classifier = None
    auxiliary_layercam = None
    if args.auxiliary_binary_checkpoint is not None:
        if not 0.0 <= args.auxiliary_binary_weight <= 1.0:
            raise ValueError("--auxiliary-binary-weight must be in [0,1]")
        auxiliary_classifier, auxiliary_task, _ = load_classifier(
            args.auxiliary_binary_checkpoint, 1, device,
            expected_split_manifest=args.split_manifest,
        )
        if auxiliary_classifier.classifier.out_features != 1:
            raise ValueError("--auxiliary-binary-checkpoint must contain a one-logit classifier")
        auxiliary_layercam = LayerCAM(
            auxiliary_classifier,
            device=device,
            layer_weights=layercam_weights,
            gradient_mode=args.layercam_gradient_mode,
        )
        print(f"Loaded auxiliary binary checkpoint task={auxiliary_task}")

    sam_predictor = SAMPredictor(
        checkpoint_path=args.sam_checkpoint,
        device=str(sam_device),
    )

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    candidate_diagnostic_dir = args.output_dir / "candidate_diagnostics"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_candidate_diagnostics:
        candidate_diagnostic_dir.mkdir(parents=True, exist_ok=True)

    candidate_diagnostic_rows: list[dict[str, object]] = []
    pseudo_manifest_rows: list[dict[str, object]] = []
    skipped_image_names: list[str] = []
    samples_by_name = {str(sample["image_id"]): sample for sample in dataset.samples}

    skipped = 0
    processed = 0
    visualized = 0
    process_limit = None if args.process_all or args.max_images <= 0 else args.max_images
    use_ground_truth_class = args.cam_target_class == "ground_truth"
    if use_ground_truth_class and target_columns not in (["tumor_type"], ["tumor"]):
        raise ValueError(
            "--cam-target-class ground_truth requires target_columns=['tumor_type'] "
            "or ['tumor'] so the image-level label has a defined CAM target."
        )
    try:
        for images, targets, image_names in tqdm(
            loader, desc="pseudo-masks", disable=should_disable_tqdm()
        ):
            images = images.to(device)

            for idx, image_name in enumerate(image_names):
                if process_limit is not None and processed >= process_limit:
                    break
                image_tensor = images[idx : idx + 1]  # [1,3,H,W]
                mask_path = mask_dir / f"{Path(image_name).stem}.png"
                save_visuals = visualized < max(0, args.save_visuals_limit)

                # ── 1. Classifier forward ─────────────────────────────────────
                gt_class: int | None = None
                with torch.no_grad():
                    logits = classifier(image_tensor)
                    predicted_class_weights = classifier_class_weights(logits, classifier_task)
                    if use_ground_truth_class and target_columns == ["tumor_type"]:
                        gt_class = int(targets[idx].item())
                        class_weights = np.zeros(logits.shape[1], dtype=np.float32)
                        class_weights[gt_class] = 1.0
                    elif use_ground_truth_class and target_columns == ["tumor"]:
                        is_tumor = bool(float(targets[idx].item()) > 0.5)
                        gt_class = 0
                        class_weights = np.asarray(
                            [
                                1.0
                                if is_tumor or args.force_normal_candidate_gallery
                                else 0.0
                            ],
                            dtype=np.float32,
                        )
                    else:
                        class_weights = predicted_class_weights.copy()

                selected_class = (
                    int(gt_class)
                    if use_ground_truth_class and gt_class is not None
                    else int(np.argmax(class_weights))
                )
                classifier_confidence = (
                    float(predicted_class_weights[selected_class])
                    if 0 <= selected_class < len(predicted_class_weights)
                    else float(np.max(predicted_class_weights))
                )
                sample_record = samples_by_name[str(image_name)]
                true_tumor = int(bool(sample_record.get("tumor", 0)))
                true_tumor_type = int(sample_record.get("tumor_type", true_tumor))
                affinity_selector_record = None
                affinity_selector_map = None
                if affinity_selector_contract is not None:
                    assert affinity_selector_root is not None
                    affinity_selector_record = affinity_selector_rows[str(image_name)]
                    affinity_selector_map = load_affinity_selector_map(
                        affinity_selector_record,
                        root=affinity_selector_root,
                        expected_image_id=str(image_name),
                        expected_group_id=str(sample_record.get("group_id", "")),
                        expected_image_label=true_tumor,
                        image_size=args.image_size,
                    )

                should_skip = (
                    (
                        target_columns == ["tumor_type"]
                        and should_skip_tumor_type(class_weights, use_ground_truth_class, gt_class)
                        and not (args.force_non_normal_cam and not use_ground_truth_class)
                    )
                    or (
                        use_ground_truth_class
                        and target_columns == ["tumor"]
                        and float(class_weights[0]) < 0.5
                        and not args.force_normal_candidate_gallery
                    )
                    or (
                        not use_ground_truth_class
                        and target_columns != ["tumor_type"]
                        and classifier_task != "single-label"
                        and float(class_weights.max()) < args.confidence_threshold
                    )
                )
                if should_skip:
                    empty_mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
                    save_mask(empty_mask, mask_path)
                    skip_reason = (
                        "known_image_label_normal"
                        if use_ground_truth_class
                        else "classifier_predicted_normal_or_below_confidence"
                    )
                    pseudo_manifest_rows.append(
                        {
                            "image_name": str(image_name),
                            "group_id": str(sample_record.get("group_id", "")),
                            "affinity_selector_map_sha256": (
                                affinity_selector_record["map_sha256"]
                                if affinity_selector_record is not None
                                else ""
                            ),
                            "true_tumor": true_tumor,
                            "true_tumor_type": true_tumor_type,
                            "selected_class": selected_class,
                            "classifier_confidence": classifier_confidence,
                            "cam_target_protocol": args.cam_target_class,
                            "status": "empty_by_image_gate",
                            "reason": skip_reason,
                            "cam_min": "",
                            "cam_max": "",
                            "cam_mean": "",
                            "cam_std": "",
                            "cam_nonzero_ratio": "",
                            "morphology_components": 0,
                            "sam_prompt_calls": 0,
                            "unique_positive_prompt_points": 0,
                            "unique_negative_prompt_points": 0,
                            "unique_prompt_points": 0,
                            "box_prompt_calls": 0,
                            "sam_candidate_count": 0,
                            "selection_score_min": "",
                            "selection_score_max": "",
                            "selection_score_mean": "",
                            "classifier_causal_score_min": "",
                            "classifier_causal_score_max": "",
                            "classifier_causal_score_mean": "",
                            "above_threshold_candidates": 0,
                            "selected_candidates": 0,
                            "selected_components": 0,
                            "support_area_ratio": 0.0,
                            "selected_area_ratio": 0.0,
                            "final_area_ratio": 0.0,
                        }
                    )
                    if args.save_candidate_diagnostics and (
                        true_tumor or args.candidate_diagnostics_cohort == "all"
                    ):
                        diagnostic_path = candidate_diagnostic_dir / f"{Path(image_name).stem}.npz"
                        diagnostic_row = save_candidate_diagnostics(
                            diagnostic_path,
                            sam_masks=np.zeros(
                                (0, args.image_size, args.image_size), dtype=np.uint8
                            ),
                            refined_mask=empty_mask,
                            final_mask=empty_mask,
                            bone_support=None,
                            prompt_map=np.zeros_like(empty_mask, dtype=np.float32),
                            positive_points=[],
                            negative_points=[],
                            boxes=[],
                            sam_scores=np.zeros(0, dtype=np.float32),
                            selection_scores=np.zeros(0, dtype=np.float32),
                            classifier_causal_scores=None,
                            component_ids=None,
                            prompt_modes=[],
                        )
                        candidate_diagnostic_rows.append(
                            {
                                "image_name": str(image_name),
                                "affinity_selector_map_sha256": (
                                    affinity_selector_record["map_sha256"]
                                    if affinity_selector_record is not None
                                    else ""
                                ),
                                "tumor_type": tumor_type_by_name.get(str(image_name), ""),
                                "generation_status": "empty_by_image_gate",
                                **diagnostic_row,
                            }
                        )
                    skipped += 1
                    skipped_image_names.append(str(image_name))
                    processed += 1
                    continue

                if (
                    args.force_non_normal_cam
                    and not use_ground_truth_class
                    and target_columns == ["tumor_type"]
                    and int(np.argmax(class_weights)) == 0
                ):
                    non_normal = np.asarray(class_weights, dtype=np.float32).copy()
                    non_normal[0] = -np.inf
                    selected = int(np.argmax(non_normal))
                    class_weights = np.zeros_like(class_weights, dtype=np.float32)
                    class_weights[selected] = 1.0
                    selected_class = selected
                    classifier_confidence = float(predicted_class_weights[selected])

                # ── 2. LayerCAM fusion ────────────────────────────────────────
                if args.cam_aggregation in {
                    "tumor_union", "tumor_union_contrast", "tumor_union_contrast_class_max"
                } and target_columns == ["tumor_type"]:
                    union_output = (
                        layercam.cam_for_tumor_union_contrast(image_tensor)
                        if args.cam_aggregation in {"tumor_union_contrast", "tumor_union_contrast_class_max"}
                        else layercam.cam_for_tumor_union(image_tensor)
                    )
                    fused_cam = union_output.cam[0].detach().cpu().numpy()
                    if args.cam_aggregation == "tumor_union_contrast_class_max":
                        selected_class = (
                            int(gt_class)
                            if use_ground_truth_class and gt_class is not None
                            else int(np.argmax(class_weights))
                        )
                        if selected_class != 0:
                            class_output = layercam.cam_for_class_contrast(
                                image_tensor, selected_class, reference_index=0
                            )
                            class_cam = class_output.cam[0].detach().cpu().numpy()
                            fused_cam = np.maximum(fused_cam, class_cam).astype(np.float32)
                            fused_cam = (fused_cam - float(fused_cam.min())) / (
                                float(fused_cam.max()) - float(fused_cam.min()) + 1e-8
                            )
                    per_class_cams = [fused_cam]
                    # The single aggregate CAM is stored as local index 0;
                    # it is not a semantic tumor_type class index.
                    active_indices = [0]
                    class_weights = np.asarray([1.0], dtype=np.float32)
                else:
                    fused_cam, per_class_cams, active_indices = generate_fused_cam(
                        layercam,
                        image_tensor,
                        class_weights=class_weights,
                        confidence_threshold=args.confidence_threshold,
                    )
                if (
                    args.cam_contrast_normal
                    and target_columns == ["tumor_type"]
                    and args.cam_aggregation == "class"
                ):
                    if not 0.0 <= args.cam_contrast_weight <= 1.0:
                        raise ValueError("--cam-contrast-weight must be in [0,1]")
                    selected_class = (
                        int(gt_class)
                        if use_ground_truth_class and gt_class is not None
                        else int(np.argmax(class_weights))
                    )
                    if selected_class != 0:
                        contrast_output = layercam.cam_for_class_contrast(
                            image_tensor, selected_class, reference_index=0
                        )
                        contrast_cam = contrast_output.cam[0].detach().cpu().numpy()
                        contrast_weight = float(args.cam_contrast_weight)
                        fused_cam = (
                            (1.0 - contrast_weight) * fused_cam
                            + contrast_weight * contrast_cam
                        ).astype(np.float32, copy=False)
                        fused_cam = (fused_cam - float(fused_cam.min())) / (
                            float(fused_cam.max()) - float(fused_cam.min()) + 1e-8
                        )
                        per_class_cams = [fused_cam]
                        active_indices = [selected_class]
                if cam_multiscale_sizes:
                    if target_columns != ["tumor_type"] or args.cam_aggregation != "class":
                        raise ValueError(
                            "--cam-multiscale-sizes currently requires target_columns=tumor_type and "
                            "--cam-aggregation class"
                        )
                    selected_class = (
                        int(gt_class)
                        if use_ground_truth_class and gt_class is not None
                        else int(np.argmax(class_weights))
                    )
                    if selected_class != 0:
                        multiscale_maps: list[np.ndarray] = []
                        for scale in (args.image_size, *cam_multiscale_sizes):
                            scaled_tensor = image_tensor
                            if scale != args.image_size:
                                scaled_tensor = F.interpolate(
                                    image_tensor, size=(scale, scale), mode="bilinear", align_corners=False
                                )
                            scaled_output = layercam.cam_for_class_contrast(
                                scaled_tensor, selected_class, reference_index=0
                            )
                            scaled_cam = scaled_output.cam[0]
                            scaled_cam = F.interpolate(
                                scaled_cam[None, None], size=(args.image_size, args.image_size),
                                mode="bilinear", align_corners=False,
                            )[0, 0].detach().cpu().numpy()
                            scaled_cam = (scaled_cam - float(scaled_cam.min())) / (
                                float(scaled_cam.max()) - float(scaled_cam.min()) + 1e-8
                            )
                            multiscale_maps.append(scaled_cam.astype(np.float32))
                        fused_cam = np.mean(np.stack(multiscale_maps, axis=0), axis=0).astype(np.float32)
                        fused_cam = (fused_cam - float(fused_cam.min())) / (
                            float(fused_cam.max()) - float(fused_cam.min()) + 1e-8
                        )
                        per_class_cams = [fused_cam]
                        active_indices = [selected_class]

                if auxiliary_layercam is not None:
                    auxiliary_output = auxiliary_layercam.cam_for_class(image_tensor, 0)
                    auxiliary_cam = auxiliary_output.cam[0].detach().cpu().numpy()
                    blend = float(args.auxiliary_binary_weight)
                    fused_cam = (1.0 - blend) * fused_cam + blend * auxiliary_cam
                    per_class_cams = [
                        (1.0 - blend) * cam + blend * auxiliary_cam
                        for cam in per_class_cams
                    ]
                    fused_cam = (fused_cam - float(fused_cam.min())) / (
                        float(fused_cam.max()) - float(fused_cam.min()) + 1e-8
                    )
                    per_class_cams = [
                        (cam - float(cam.min())) / (float(cam.max()) - float(cam.min()) + 1e-8)
                        for cam in per_class_cams
                    ]
                if args.cam_tta_flip:
                    flipped_tensor = torch.flip(image_tensor, dims=[3])
                    if args.cam_aggregation in {
                        "tumor_union", "tumor_union_contrast", "tumor_union_contrast_class_max"
                    } and target_columns == ["tumor_type"]:
                        flipped_output = (
                            layercam.cam_for_tumor_union_contrast(flipped_tensor)
                            if args.cam_aggregation in {"tumor_union_contrast", "tumor_union_contrast_class_max"}
                            else layercam.cam_for_tumor_union(flipped_tensor)
                        )
                        flipped_cam = flipped_output.cam[0].detach().cpu().numpy()
                        flipped_class_cams = [flipped_cam]
                        flipped_indices = [0]
                    else:
                        flipped_cam, flipped_class_cams, flipped_indices = generate_fused_cam(
                            layercam,
                            flipped_tensor,
                            class_weights=class_weights,
                            confidence_threshold=args.confidence_threshold,
                        )
                    if active_indices != flipped_indices:
                        raise RuntimeError("CAM TTA changed active class indices; cannot fuse maps safely")
                    fused_cam = 0.5 * (fused_cam + np.fliplr(flipped_cam))
                    per_class_cams = [
                        0.5 * (cam + np.fliplr(flipped_cam_item))
                        for cam, flipped_cam_item in zip(per_class_cams, flipped_class_cams)
                    ]
                    fused_cam = (fused_cam - float(fused_cam.min())) / (
                        float(fused_cam.max()) - float(fused_cam.min()) + 1e-8
                    )

                external_saliency_map = None
                if external_saliency_contract is not None:
                    assert external_saliency_root is not None
                    external_saliency_map = load_external_saliency_map(
                        external_saliency_rows[str(image_name)],
                        root=external_saliency_root,
                        expected_image_id=str(image_name),
                        expected_image_label=true_tumor,
                        image_size=args.image_size,
                    )
                    if args.external_saliency_role == "replace":
                        fused_cam = external_saliency_map
                        per_class_cams = [fused_cam]
                        active_indices = [0]
                        class_weights = np.asarray([1.0], dtype=np.float32)

                image_pil = tensor_to_pil(image_tensor[0].detach().cpu(), normalization=classifier_normalization)
                image_rgb = tensor_to_rgb_numpy(image_tensor[0], normalization=classifier_normalization)
                teacher_probability = None
                teacher_support = None
                teacher_components = []
                external_saliency_components = []
                cam_component_count = 0
                if proposal_teacher is not None:
                    with torch.no_grad():
                        teacher_input = F.interpolate(
                            image_tensor,
                            size=(448, 448),
                            mode="bilinear",
                            align_corners=False,
                        )
                        teacher_logits = proposal_teacher(teacher_input)
                        teacher_probability_tensor = torch.sigmoid(teacher_logits)
                        teacher_probability_tensor = F.interpolate(
                            teacher_probability_tensor,
                            size=(args.image_size, args.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        teacher_probability = (
                            teacher_probability_tensor[0, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                if save_visuals:
                    for local_i, cls_i in enumerate(active_indices):
                        cls_name = class_names[cls_i]
                        save_overlay(
                            image_pil,
                            per_class_cams[local_i],
                            overlay_dir / f"{Path(image_name).stem}_{cls_name}.png",
                        )
                    save_overlay(
                        image_pil,
                        fused_cam,
                        overlay_dir / f"{Path(image_name).stem}_fused_layercam.png",
                    )
                    visualized += 1

                # ── 3. Prompt extraction ──────────────────────────────────────
                debug_dir = (
                    args.output_dir / "debug" / Path(image_name).stem
                    if args.debug and save_visuals else None
                )
                bone_likelihood = None
                bone_support = None
                bone_components = []
                prompt_map = fused_cam
                if not args.disable_morphology:
                    if args.morphology_fusion_mode == "components":
                        active_weights = [float(class_weights[i]) for i in active_indices]
                        prompt_percentiles = (
                            cam_percentile_values if args.cam_percentile_ensemble else (args.cam_percentile,)
                        )
                        bone_likelihood = None
                        bone_support = None
                        bone_components = []
                        for prompt_percentile in prompt_percentiles:
                            local_likelihood, local_support, local_components = morphology.build_class_conditioned_components(
                                image_rgb,
                                per_class_cams,
                                active_weights,
                                cam_percentile=prompt_percentile,
                                min_component_area=max(20, args.min_component_area // 2),
                                max_components=(args.max_components if args.all_cam_components else 1),
                                points_per_component=args.points_per_component,
                                bbox_padding_ratio=args.bbox_padding_ratio,
                                negative_points_per_component=args.negative_points_per_component,
                                debug_dir=debug_dir,
                            )
                            if bone_likelihood is None:
                                bone_likelihood = local_likelihood
                            if bone_support is None:
                                bone_support = local_support.copy()
                            else:
                                bone_support = np.maximum(bone_support, local_support)
                            offset = len(bone_components)
                            bone_components.extend(
                                replace(component, component_id=offset + int(component.component_id))
                                for component in local_components
                            )
                        max_prompt_components = args.max_components * len(prompt_percentiles)
                        if args.all_cam_components and len(bone_components) > max_prompt_components:
                            bone_components = bone_components[:max_prompt_components]
                        if bone_likelihood is None:
                            bone_likelihood = np.zeros_like(fused_cam, dtype=np.float32)
                        if bone_support is None:
                            bone_support = np.zeros_like(fused_cam, dtype=np.uint8)
                        cam_component_count = len(bone_components)
                        if (
                            external_saliency_map is not None
                            and args.external_saliency_role == "proposal_gallery"
                        ):
                            external_components = (
                                build_external_saliency_proposal_gallery(
                                    image_rgb=image_rgb,
                                    saliency_map=external_saliency_map,
                                    prompt_percentiles=tuple(prompt_percentiles),
                                    min_component_area=args.min_component_area,
                                    max_components=args.max_components,
                                    all_cam_components=args.all_cam_components,
                                    points_per_component=args.points_per_component,
                                    bbox_padding_ratio=args.bbox_padding_ratio,
                                    negative_points_per_component=(
                                        args.negative_points_per_component
                                    ),
                                )
                            )
                            offset = len(bone_components)
                            external_saliency_components = [
                                replace(
                                    component,
                                    component_id=(
                                        offset + int(component.component_id)
                                    ),
                                )
                                for component in external_components
                            ]
                            bone_components.extend(external_saliency_components)
                        if teacher_probability is not None:
                            teacher_support, teacher_components = (
                                morphology.build_probability_components(
                                    teacher_probability,
                                    threshold=args.proposal_teacher_threshold,
                                    min_component_area=args.proposal_teacher_min_component_area,
                                    max_components=args.proposal_teacher_max_components,
                                    points_per_component=args.points_per_component,
                                    bbox_padding_ratio=args.bbox_padding_ratio,
                                    negative_points_per_component=(
                                        args.negative_points_per_component
                                    ),
                                )
                            )
                            offset = len(bone_components)
                            bone_components.extend(
                                replace(
                                    component,
                                    component_id=offset + int(component.component_id),
                                )
                                for component in teacher_components
                            )
                    else:
                        bone_likelihood, bone_support = morphology.build_tumor_guidance(
                            image_rgb,
                            fused_cam,
                            seed_percentile=seed_percentile,
                            support_percentile=support_percentile,
                            min_component_area=max(20, args.min_component_area // 2),
                            debug_dir=debug_dir,
                        )
                        prompt_map = morphology.fuse_cam_with_bone_guidance(
                            fused_cam,
                            bone_likelihood,
                            bone_support,
                        )

                # ── 4. SAM candidate masks ────────────────────────────────────
                component_ids = None
                sam_image_rgb = image_rgb
                sam_image_pil = image_pil
                sam_components = bone_components
                point_prompts: list[tuple[int, int]] = []
                candidate_prompt_modes: list[str] = []
                candidate_proposal_sources: list[str] = []
                prompt_stats = {
                    "sam_prompt_calls": 0,
                    "unique_positive_prompt_points": 0,
                    "unique_negative_prompt_points": 0,
                    "unique_prompt_points": 0,
                    "box_prompt_calls": 0,
                }
                sam_scale = float(args.sam_image_size) / float(args.image_size) if args.sam_image_size > 0 else 1.0
                sam_scale_x = sam_scale
                sam_scale_y = sam_scale
                if args.sam_image_size > 0 and args.sam_image_size != args.image_size:
                    original_path = dataset.images_dir / str(image_name)
                    original = Image.open(original_path).convert("RGB")
                    if args.sam_preserve_aspect:
                        original_width, original_height = original.size
                        resize_scale = min(
                            float(args.sam_image_size) / max(1, original_width),
                            float(args.sam_image_size) / max(1, original_height),
                        )
                        sam_width = max(1, int(round(original_width * resize_scale)))
                        sam_height = max(1, int(round(original_height * resize_scale)))
                        sam_image_pil = original.resize(
                            (sam_width, sam_height), Image.Resampling.BILINEAR
                        )
                    else:
                        sam_image_pil = original.resize(
                            (args.sam_image_size, args.sam_image_size), Image.Resampling.BILINEAR
                        )
                    sam_image_rgb = np.asarray(sam_image_pil, dtype=np.uint8)
                    sam_height, sam_width = sam_image_rgb.shape[:2]
                    sam_scale_x = float(sam_width) / float(args.image_size)
                    sam_scale_y = float(sam_height) / float(args.image_size)
                    if bone_components:
                        sam_components = []
                        for component in bone_components:
                            mask = torch.from_numpy(component.mask.astype(np.float32))[None, None]
                            scaled_mask = F.interpolate(
                                mask,
                                size=(sam_height, sam_width),
                                mode="nearest",
                            )[0, 0].numpy().astype(np.uint8)
                            h_sam, w_sam = scaled_mask.shape
                            x0, y0, x1, y1 = component.bbox
                            bbox = (
                                max(0, min(w_sam - 1, int(round(x0 * sam_scale_x)))),
                                max(0, min(h_sam - 1, int(round(y0 * sam_scale_y)))),
                                max(0, min(w_sam - 1, int(round((x1 + 1) * sam_scale_x) - 1))),
                                max(0, min(h_sam - 1, int(round((y1 + 1) * sam_scale_y) - 1))),
                            )
                            positive_points = tuple(
                                (
                                    max(0, min(h_sam - 1, int(round(row * sam_scale_y)))),
                                    max(0, min(w_sam - 1, int(round(col * sam_scale_x)))),
                                )
                                for row, col in component.positive_points
                            )
                            negative_points = tuple(
                                (
                                    max(0, min(h_sam - 1, int(round(row * sam_scale_y)))),
                                    max(0, min(w_sam - 1, int(round(col * sam_scale_x)))),
                                )
                                for row, col in getattr(component, "negative_points", ())
                            )
                            values = {
                                "mask": scaled_mask,
                                "bbox": bbox,
                                "positive_points": positive_points,
                            }
                            if hasattr(component, "negative_points"):
                                values["negative_points"] = negative_points
                            sam_components.append(replace(component, **values))

                if args.sam_grid_gallery:
                    sam_masks, sam_scores = sam_predictor.predict_grid_gallery(
                        sam_image_rgb,
                        points_per_side=args.sam_grid_points_per_side,
                        points_per_batch=args.sam_grid_points_per_batch,
                        pred_iou_thresh=args.sam_grid_pred_iou_thresh,
                        stability_score_thresh=args.sam_grid_stability_thresh,
                        box_nms_thresh=args.sam_grid_box_nms_thresh,
                    )
                    component_ids = None
                    candidate_prompt_modes.extend(["grid"] * len(sam_scores))
                    prompt_stats.update(sam_predictor.last_prompt_stats)
                elif sam_components:
                    prompt_modes = [args.sam_prompt_mode]
                    if args.sam_prompt_ensemble:
                        prompt_modes = list(dict.fromkeys(prompt_modes + ["point", "box"]))
                    mask_batches: list[np.ndarray] = []
                    score_batches: list[np.ndarray] = []
                    component_batches: list[np.ndarray] = []
                    for prompt_mode in prompt_modes:
                        mode_masks, mode_scores, mode_components = sam_predictor.predict_from_components(
                            sam_image_rgb,
                            sam_components,
                            prompt_mode=prompt_mode,
                            multimask_output=not args.sam_single_mask,
                            negative_points_per_component=args.negative_points_per_component,
                            prompt_border_margin=args.prompt_border_margin,
                            max_box_area_ratio=(
                                args.max_box_area_ratio
                                if args.max_box_area_ratio and args.max_box_area_ratio > 0
                                else None
                            ),
                            debug_dir=debug_dir,
                            image_pil=sam_image_pil,
                        )
                        mask_batches.append(mode_masks)
                        score_batches.append(mode_scores)
                        component_batches.append(mode_components)
                        candidate_prompt_modes.extend([prompt_mode] * len(mode_scores))
                        mode_stats = sam_predictor.last_prompt_stats
                        prompt_stats["sam_prompt_calls"] += int(mode_stats["sam_prompt_calls"])
                        prompt_stats["box_prompt_calls"] += int(mode_stats["box_prompt_calls"])
                        for key in (
                            "unique_positive_prompt_points",
                            "unique_negative_prompt_points",
                            "unique_prompt_points",
                        ):
                            prompt_stats[key] = max(prompt_stats[key], int(mode_stats[key]))
                    sam_masks = np.concatenate(mask_batches, axis=0) if mask_batches else np.zeros((0, *sam_image_rgb.shape[:2]), dtype=bool)
                    sam_scores = np.concatenate(score_batches, axis=0) if score_batches else np.zeros(0, dtype=np.float32)
                    component_ids = np.concatenate(component_batches, axis=0) if component_batches else np.zeros(0, dtype=np.int32)
                else:
                    point_prompts = extract_point_prompts(
                        prompt_map,
                        cam_percentile=args.cam_percentile,
                        max_points=args.max_points,
                        min_component_area=args.min_component_area,
                        support_mask=bone_support,
                        debug_dir=debug_dir,
                        image_pil=image_pil,
                    )
                    if sam_scale_x != 1.0 or sam_scale_y != 1.0:
                        point_prompts = [
                            (
                                max(0, min(sam_image_rgb.shape[0] - 1, int(round(row * sam_scale_y)))),
                                max(0, min(sam_image_rgb.shape[1] - 1, int(round(col * sam_scale_x)))),
                            )
                            for row, col in point_prompts
                        ]
                    sam_masks, sam_scores = sam_predictor.predict_from_points(
                        sam_image_rgb, point_prompts,
                        debug_dir=debug_dir,
                        image_pil=sam_image_pil,
                    )
                    prompt_stats.update(sam_predictor.last_prompt_stats)

                sam_candidate_count = int(len(sam_masks))

                if sam_masks.shape[-2:] != (args.image_size, args.image_size):
                    sam_masks = F.interpolate(
                        torch.from_numpy(sam_masks.astype(np.float32))[:, None],
                        size=(args.image_size, args.image_size),
                        mode="nearest",
                    )[:, 0].numpy() > 0.5

                if args.include_cam_candidate and bone_components:
                    cam_masks = np.stack(
                        [component.mask.astype(bool) for component in bone_components], axis=0
                    )
                    cam_component_ids = np.asarray(
                        [int(component.component_id) for component in bone_components], dtype=np.int32
                    )
                    sam_masks = np.concatenate([sam_masks.astype(bool), cam_masks], axis=0)
                    if component_ids is None:
                        component_ids = cam_component_ids
                    else:
                        component_ids = np.concatenate([component_ids, cam_component_ids], axis=0)
                    sam_scores = np.concatenate([
                        np.asarray(sam_scores, dtype=np.float32),
                        np.zeros(len(cam_masks), dtype=np.float32),
                    ], axis=0)
                    candidate_prompt_modes.extend(["cam"] * len(cam_masks))

                if component_ids is not None:
                    external_component_end = (
                        cam_component_count + len(external_saliency_components)
                    )
                    for component_id in component_ids:
                        component_id_int = int(component_id)
                        if (
                            external_saliency_contract is not None
                            and args.external_saliency_role == "replace"
                        ):
                            candidate_proposal_sources.append("external_saliency")
                        elif component_id_int < cam_component_count:
                            candidate_proposal_sources.append("layercam")
                        elif component_id_int < external_component_end:
                            candidate_proposal_sources.append("external_saliency")
                        else:
                            candidate_proposal_sources.append("proposal_teacher")
                else:
                    candidate_proposal_sources.extend(["unassigned"] * len(sam_masks))
                proposal_source_ids = np.asarray(
                    candidate_proposal_sources, dtype="U32"
                )
                prompt_mode_array = np.asarray(candidate_prompt_modes, dtype="U32")
                if (
                    len(proposal_source_ids) != len(sam_masks)
                    or len(prompt_mode_array) != len(sam_masks)
                ):
                    raise RuntimeError(
                        "Candidate source/prompt provenance is not aligned with SAM masks"
                    )

                classifier_causal_scores = None
                if args.selection_method == "coverage_mass_sam_causal":
                    if target_columns != ["tumor"] or classifier.classifier.out_features != 1:
                        raise ValueError(
                            "coverage_mass_sam_causal requires target_columns=['tumor'] "
                            "and a one-logit binary classifier"
                        )
                    classifier_causal_scores = classifier_candidate_causal_scores(
                        classifier,
                        image_tensor,
                        logits,
                        sam_masks,
                    )

                if debug_dir is not None:
                    np.savez_compressed(
                        Path(debug_dir) / "candidate_diagnostics.npz",
                        masks=sam_masks.astype(np.uint8),
                        sam_scores=np.asarray(sam_scores, dtype=np.float32),
                        classifier_causal_scores=np.asarray(
                            classifier_causal_scores
                            if classifier_causal_scores is not None
                            else np.zeros(len(sam_masks), dtype=np.float32),
                            dtype=np.float32,
                        ),
                        component_ids=np.asarray(component_ids if component_ids is not None else np.zeros(len(sam_masks), dtype=np.int32)),
                        fused_cam=fused_cam.astype(np.float32),
                        component_masks=(
                            np.stack([component.mask for component in bone_components]).astype(np.uint8)
                            if bone_components else np.zeros((0, args.image_size, args.image_size), dtype=np.uint8)
                        ),
                        prompt_modes=np.asarray(candidate_prompt_modes, dtype="U16"),
                        proposal_source_ids=proposal_source_ids,
                        selector_map=np.asarray(
                            affinity_selector_map
                            if affinity_selector_map is not None
                            else fused_cam,
                            dtype=np.float32,
                        ),
                    )

                # ── 4b. Prompt-quality metrics (optional, pre-SAM diagnostics) ──
                # ── 5. CAM-guided mask selection ──────────────────────────────
                component_mask_array = (
                    np.stack([component.mask for component in bone_components])
                    if bone_components else None
                )
                positive_points_by_component = (
                    {
                        int(component.component_id): tuple(component.positive_points)
                        for component in bone_components
                    }
                    if bone_components else None
                )
                negative_points_by_component = (
                    {
                        int(component.component_id): tuple(getattr(component, "negative_points", ()))
                        for component in bone_components
                    }
                    if bone_components else None
                )
                selection_map = (
                    affinity_selector_map
                    if affinity_selector_map is not None
                    else fused_cam
                )
                selection_scores = score_masks(
                    sam_masks,
                    selection_map,
                    method=args.selection_method,
                    bone_likelihood=bone_likelihood,
                    bone_support=bone_support,
                    sam_scores=sam_scores,
                    classifier_causal_scores=classifier_causal_scores,
                    component_ids=component_ids,
                    component_masks=component_mask_array,
                    positive_points_by_component=positive_points_by_component,
                    negative_points_by_component=negative_points_by_component,
                    proposal_teacher_probability=teacher_probability,
                    proposal_teacher_component_start=(
                        cam_component_count if teacher_probability is not None else None
                    ),
                    prompt_modes=prompt_mode_array,
                    proposal_source_ids=proposal_source_ids,
                    graph_component_topk=args.component_topk,
                    prompt_hybrid_weights=prompt_score_weights,
                    prompt_area_target=args.prompt_area_target,
                    prompt_area_log_sigma=args.prompt_area_log_sigma,
                )
                refined, selection_details = select_and_fuse_masks(
                    sam_masks,
                    selection_map,
                    mask_score_threshold=args.mask_score_threshold,
                    selection_method=args.selection_method,
                    fusion_topk=args.fusion_topk,
                    bone_likelihood=bone_likelihood,
                    bone_support=bone_support,
                    sam_scores=sam_scores,
                    classifier_causal_scores=classifier_causal_scores,
                    component_ids=component_ids,
                    component_masks=component_mask_array,
                    positive_points_by_component=positive_points_by_component,
                    negative_points_by_component=negative_points_by_component,
                    proposal_teacher_probability=teacher_probability,
                    proposal_teacher_component_start=(
                        cam_component_count if teacher_probability is not None else None
                    ),
                    prompt_modes=prompt_mode_array,
                    proposal_source_ids=proposal_source_ids,
                    prompt_hybrid_weights=prompt_score_weights,
                    prompt_area_target=args.prompt_area_target,
                    prompt_area_log_sigma=args.prompt_area_log_sigma,
                    best_per_component=component_ids is not None and not args.disable_best_per_component,
                    component_topk=args.component_topk,
                    support_clip_kernel=args.support_clip_kernel,
                    low_score_policy=args.low_score_policy,
                    return_details=True,
                )

                # ── 5b. SAM-vs-selection oracle diagnostic (optional) ───────────
                # ── 6. Morphological refinement ───────────────────────────────
                final_mask = morphological_refinement(
                    refined,
                    closing_kernel=args.closing_kernel,
                    opening_kernel=args.opening_kernel,
                    min_size=args.min_size,
                    guidance_map=(
                        selection_map
                        if args.selection_method == "affinity_rank_single"
                        else bone_likelihood
                    ),
                    guidance_threshold=args.guidance_threshold,
                    max_hole_area=args.max_hole_area,
                )

                # ── 7. Save ───────────────────────────────────────────────────
                if args.save_candidate_diagnostics and (
                    true_tumor or args.candidate_diagnostics_cohort == "all"
                ):
                    all_points = (
                        [
                            point
                            for component in bone_components
                            for point in component.positive_points
                        ]
                        if bone_components
                        else point_prompts
                    )
                    all_negative_points = (
                        [
                            point
                            for component in bone_components
                            for point in getattr(component, "negative_points", ())
                        ]
                        if bone_components
                        else []
                    )
                    all_boxes: list[tuple[int, int, int, int]] = []
                    if bone_components and args.sam_prompt_mode in {"box", "box_point"}:
                        image_height, image_width = image_rgb.shape[:2]
                        for component in bone_components:
                            x0, y0, x1, y1 = component.bbox
                            box_area_ratio = (
                                max(1, x1 - x0 + 1) * max(1, y1 - y0 + 1)
                                / float(image_height * image_width)
                            )
                            if (
                                args.max_box_area_ratio <= 0
                                or box_area_ratio <= args.max_box_area_ratio
                            ):
                                all_boxes.append(tuple(component.bbox))
                    diagnostic_path = (
                        candidate_diagnostic_dir / f"{Path(image_name).stem}.npz"
                    )
                    diagnostic_row = save_candidate_diagnostics(
                        diagnostic_path,
                        sam_masks=sam_masks,
                        refined_mask=refined,
                        final_mask=final_mask,
                        bone_support=bone_support,
                        prompt_map=prompt_map,
                        positive_points=all_points,
                        negative_points=all_negative_points,
                        boxes=all_boxes,
                        sam_scores=sam_scores,
                        selection_scores=selection_scores,
                        classifier_causal_scores=classifier_causal_scores,
                        component_ids=component_ids,
                        prompt_modes=candidate_prompt_modes,
                        proposal_source_ids=proposal_source_ids,
                    )
                    candidate_diagnostic_rows.append(
                        {
                            "image_name": str(image_name),
                            "affinity_selector_map_sha256": (
                                affinity_selector_record["map_sha256"]
                                if affinity_selector_record is not None
                                else ""
                            ),
                            "tumor_type": tumor_type_by_name.get(str(image_name), ""),
                            "generation_status": (
                                "ok" if final_mask.any() else "empty_after_localization"
                            ),
                            **diagnostic_row,
                        }
                    )

                # Normal-image proposals are negative MIL instances, never
                # foreground pseudo-labels. Preserve the frozen diagnostic bag
                # above while keeping the ordinary pseudo-mask artifact empty.
                if args.force_normal_candidate_gallery and not true_tumor:
                    final_mask = np.zeros_like(final_mask, dtype=bool)

                save_mask(final_mask, mask_path)
                pseudo_manifest_rows.append(
                    {
                        "image_name": str(image_name),
                        "group_id": str(sample_record.get("group_id", "")),
                        "affinity_selector_map_sha256": (
                            affinity_selector_record["map_sha256"]
                            if affinity_selector_record is not None
                            else ""
                        ),
                        "true_tumor": true_tumor,
                        "true_tumor_type": true_tumor_type,
                        "selected_class": selected_class,
                        "classifier_confidence": classifier_confidence,
                        "cam_target_protocol": args.cam_target_class,
                        "status": "ok" if final_mask.any() else "empty_after_localization",
                        "reason": "" if final_mask.any() else "no_candidate_survived_or_postprocess_empty",
                        "cam_min": float(np.min(fused_cam)),
                        "cam_max": float(np.max(fused_cam)),
                        "cam_mean": float(np.mean(fused_cam)),
                        "cam_std": float(np.std(fused_cam)),
                        "cam_nonzero_ratio": float(np.count_nonzero(fused_cam) / fused_cam.size),
                        "morphology_components": len(bone_components),
                        "cam_morphology_components": cam_component_count,
                        "external_saliency_components": len(
                            external_saliency_components
                        ),
                        "proposal_teacher_components": len(teacher_components),
                        "proposal_teacher_support_area_ratio": (
                            float(np.count_nonzero(teacher_support) / teacher_support.size)
                            if teacher_probability is not None and teacher_support is not None
                            else 0.0
                        ),
                        **prompt_stats,
                        "sam_candidate_count": sam_candidate_count,
                        "selection_score_min": float(selection_scores.min()) if len(selection_scores) else "",
                        "selection_score_max": float(selection_scores.max()) if len(selection_scores) else "",
                        "selection_score_mean": float(selection_scores.mean()) if len(selection_scores) else "",
                        "classifier_causal_score_min": (
                            float(classifier_causal_scores.min())
                            if classifier_causal_scores is not None and len(classifier_causal_scores)
                            else ""
                        ),
                        "classifier_causal_score_max": (
                            float(classifier_causal_scores.max())
                            if classifier_causal_scores is not None and len(classifier_causal_scores)
                            else ""
                        ),
                        "classifier_causal_score_mean": (
                            float(classifier_causal_scores.mean())
                            if classifier_causal_scores is not None and len(classifier_causal_scores)
                            else ""
                        ),
                        **selection_details,
                        "support_area_ratio": (
                            float(np.count_nonzero(bone_support) / bone_support.size)
                            if bone_support is not None else 0.0
                        ),
                        "selected_area_ratio": float(np.count_nonzero(refined) / refined.size),
                        "final_area_ratio": float(np.count_nonzero(final_mask) / final_mask.size),
                    }
                )
                processed += 1
            if process_limit is not None and processed >= process_limit:
                break
    finally:
        layercam.close()
        if auxiliary_layercam is not None:
            auxiliary_layercam.close()

    mode = "full dataset" if args.process_all else f"preview ({processed} images)"
    print(f"\nDone: {mode}. Masks saved to {mask_dir} (skipped {skipped} normal/low-confidence images)")

    skipped_path = args.output_dir / "skipped_low_confidence.txt"
    if skipped_image_names:
        skipped_path.write_text("\n".join(skipped_image_names) + "\n", encoding="utf-8")
        print(f"Saved list of {len(skipped_image_names)} skipped image names to {skipped_path}")
    elif skipped_path.exists():
        # Do not let a previous run's predicted-normal/low-confidence list
        # leak into evaluation after a clean rerun in the same output dir.
        skipped_path.unlink()

    expected_names = [str(sample["image_id"]) for sample in dataset.samples]
    if process_limit is not None:
        expected_names = expected_names[:process_limit]
    run_metadata_path = args.output_dir / "run_metadata.json"
    pseudo_summary = write_pseudo_mask_manifest(
        args.output_dir,
        pseudo_manifest_rows,
        expected_image_names=expected_names,
        split=args.split,
        image_size=args.image_size,
        run_metadata_sha256=manifest_sha256_file(run_metadata_path),
    )
    print(
        f"Pseudo-mask completeness: {pseudo_summary['manifest_rows']}/"
        f"{pseudo_summary['expected_images']} rows; manifest_sha256="
        f"{pseudo_summary['manifest_sha256']}"
    )

    if args.save_candidate_diagnostics:
        expected_diagnostic_names = (
            list(expected_names)
            if args.candidate_diagnostics_cohort == "all"
            else [
                str(sample["image_id"])
                for sample in dataset.samples[: len(expected_names)]
                if bool(sample.get("tumor", 0))
            ]
        )
        diagnostic_summary = write_candidate_diagnostics_manifest(
            args.output_dir,
            candidate_diagnostic_rows,
            expected_image_names=expected_diagnostic_names,
            split=args.split,
            image_size=args.image_size,
            pseudo_manifest_sha256=str(pseudo_summary["manifest_sha256"]),
            selection_method=args.selection_method,
            support_clip_kernel=args.support_clip_kernel,
            cam_percentile=args.cam_percentile,
            cohort=args.candidate_diagnostics_cohort,
        )
        print(
            "Prediction-first candidate diagnostics frozen: "
            f"{diagnostic_summary['manifest_rows']}/"
            f"{diagnostic_summary['expected_images']} "
            f"{args.candidate_diagnostics_cohort} cases; "
            f"manifest_sha256={diagnostic_summary['manifest_sha256']}"
        )

if __name__ == "__main__":
    main()
