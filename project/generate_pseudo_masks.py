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
    DATASET_TARGET_COLUMNS,
    DEFAULT_DATASET,
    SUPPORTED_DATASETS,
)
from datasets.factory import build_classification_dataset, build_segmentation_dataset
from models.classifier import DenseNet121AnatomyClassifier
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.cam_refine import extract_feature_map, refine_cam_with_feature_affinity
from pseudo.extract_prompts import extract_point_prompts
from pseudo.morphology_factory import get_morphology_module
from pseudo.prompt_metrics import (
    binary_mask_localization_metrics,
    box_prompt_localization_metrics,
    cam_localization_metrics,
    negative_point_rejection_rate,
    point_prompt_hit_rate,
)
from pseudo.oracle_diagnostics import binary_overlap_metrics, oracle_vs_selected_metrics
from pseudo.sam_refine import SAMPredictor
from pseudo.mask_selection import SELECTION_METHODS, select_and_fuse_masks
from pseudo.morphology import morphological_refinement
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAM-H1200/BTXRD pseudo masks via LayerCAM + SAM")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--pipeline-profile",
        type=str,
        default="default",
        choices=["default", "btxrd_best"],
        help=(
            "Use a reproducible tested configuration. btxrd_best selects the current "
            "BTXRD validation configuration (320px CAM, 512px SAM, CAM contrast/"
            "percentile ensemble, prompt ensemble and coverage_mass_sam). It never "
            "enables polygon/bbox inputs."
        ),
    )
    parser.add_argument(
        "--research-overrides", action="store_true",
        help="With btxrd_best, keep profile defaults but allow explicitly supplied research ablations.",
    )
    parser.add_argument("--ram-root", type=Path, default=ROOT.parent / "RAM-H1200-v1",
                        help="Dataset root (RAM-H1200 root or BTXRD root, depending on --dataset)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        default=ROOT / "outputs" / "classifier" / "best_classifier.pt")
    parser.add_argument("--auxiliary-binary-checkpoint", type=Path, default=None,
                        help="Optional one-logit tumor checkpoint whose CAM is blended with the main CAM.")
    parser.add_argument("--auxiliary-binary-weight", type=float, default=0.35,
                        help="Blend weight for --auxiliary-binary-checkpoint CAM (0..1).")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="Path to sam_vit_b_01ec64.pth (v1) or sam2.1_hiera_tiny.pt (v2); "
                        "auto-downloaded if absent")
    parser.add_argument("--sam-version", type=str, default="v1", choices=["v1", "v2", "medsam2"],
                        help="v1=original SAM (ViT-B, SamPredictor API); "
                        "v2=SAM2 (Hiera-tiny by default, SAM2ImagePredictor — same "
                        "point/box prompt API, different checkpoint/package); "
                        "medsam2=SAM2 fine-tuned on medical imagery (bowang-lab/MedSAM2), "
                        "same API, its own vendored sam2 package/checkpoint/config")
    parser.add_argument("--sam-model-type", type=str, default="auto",
                        choices=["auto", "vit_b", "vit_l", "vit_h"],
                        help="SAM v1 backbone. 'auto' infers vit_b/vit_l/vit_h from the checkpoint filename.")
    parser.add_argument("--sam-device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for SAM. 'auto' follows the classifier; use 'cpu' on 4GB GPUs so DenseNet and SAM do not compete for VRAM.")
    parser.add_argument("--classifier-device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for DenseNet/LayerCAM. Set cpu when SAM must use the GPU on a 4GB card.")
    parser.add_argument("--sam2-model-cfg", type=str, default=None,
                        help="Overrides the default config for --sam-version v2/medsam2 "
                        "(v2 default: configs/sam2.1/sam2.1_hiera_t.yaml; "
                        "medsam2 default: configs/sam2.1_hiera_t512.yaml)")
    parser.add_argument("--target-columns", type=str, default=None,
                        help="Defaults to 'hand' for ramh1200 or 'tumor' for btxrd")
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
    parser.add_argument("--image-list", type=Path, default=None,
                        help="Optional text file of image names to process (one per line), for deterministic stratified validation smoke tests.")
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
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
    parser.add_argument("--bone-seed-percentile", type=float, default=None,
                        help="Defaults to 88.0 for ramh1200 or 82.0 for btxrd")
    parser.add_argument("--bone-support-percentile", type=float, default=None,
                        help="Defaults to 68.0 for ramh1200 or 55.0 for btxrd")
    parser.add_argument("--morphology-fusion-mode", type=str, default="components",
                        choices=["components", "weighted"])
    parser.add_argument("--sam-prompt-mode", type=str, default="box_point",
                        choices=["point", "joint_points", "box", "box_point"])
    parser.add_argument("--sam-prompt-ensemble", action="store_true",
                        help="A/B: generate candidates from box_point, point, and box prompts for each CAM component.")
    parser.add_argument("--disable-sam-prompt-ensemble", action="store_true",
                        help="A/B override for the default profile; rejected by btxrd_best.")
    parser.add_argument("--max-bone-components", type=int, default=12)
    parser.add_argument("--all-cam-components", action="store_true",
                        help="A/B option for BTXRD: keep up to --max-bone-components CAM components instead of largest-only.")
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
    parser.add_argument("--disable-bone-morphology", action="store_true",
                        help="Run the original CAM-only baseline without pre-SAM bone morphology")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--preprocessing-mode", type=str, default="none",
                        choices=["none", "clahe", "contrast", "gamma", "foreground_crop"],
                        help="Optional X-ray preprocessing before classifier/CAM")
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
                        choices=SELECTION_METHODS,
                        help="CAM-guided mask scoring method")
    parser.add_argument(
        "--selection-ablation-methods",
        type=str,
        default="",
        help="Comma-separated extra selection methods evaluated on the exact same SAM candidate pool. "
             "Their masks are saved under ablation_masks/<method>/ and diagnostics under "
             "selection_ablation.csv; this does not change the primary pipeline output.",
    )
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
                        "SAM candidate from every kept morphology component, up to --max-bone-components) "
                        "and fall back to fusion_topk's global top-k selection across all candidates instead. "
                        "Per-component selection can union in a non-lesion component's mask alongside the "
                        "real lesion's, which is invisible to fusion_topk and can dilute Dice more than "
                        "fusion_topk ever could -- this flag isolates that effect for A/B testing.")
    parser.add_argument("--component-topk", type=int, default=0,
                        help="When best-per-component is enabled, keep only the top K component proposals by image-only score; 0 keeps all.")
    parser.add_argument("--support-clip-kernel", type=int, default=5,
                        help="Clip fused SAM masks to dilated bone support; 0/1 means no dilation, -1 disables")
    parser.add_argument("--cam-refine", action="store_true",
                        help="Refine the fused LayerCAM via feature-similarity propagation "
                        "(pseudo/cam_refine.py) before morphology/SAM. Disabled by default so the "
                        "existing CAM path is unchanged unless explicitly enabled -- lets refined vs. "
                        "raw CAM be A/B tested through the same prompt_quality.csv/oracle diagnostics "
                        "already used elsewhere in this project.")
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
    parser.add_argument("--cam-refine-layer", type=str, default="denseblock3",
                        choices=["denseblock2", "denseblock3", "denseblock4"],
                        help="DenseNet121 layer to source features from for --cam-refine")
    parser.add_argument("--cam-refine-high-percentile", type=float, default=90.0,
                        help="Percentile defining seed pixels for --cam-refine")
    parser.add_argument("--cam-refine-low-percentile", type=float, default=40.0,
                        help="Percentile defining refinement targets for --cam-refine")
    parser.add_argument("--cam-refine-strength", type=float, default=0.6,
                        help="Blend strength (0=no change, 1=fully replaced) for --cam-refine")
    parser.add_argument("--cam-target-class", type=str, default="predicted",
                        choices=["predicted", "image_label", "ground_truth"],
                        help="Which class LayerCAM is conditioned on. 'predicted' (default) uses the "
                        "classifier's own argmax/top-confidence class, matching real inference where "
                        "the true label is unknown. 'image_label' instead conditions LayerCAM on the "
                        "dataset's true image-level label (still just the image-level class -- never "
                        "polygon/bbox annotations, so this stays within WSSS) when generating pseudo "
                        "masks for a labeled split. Only affects --target-columns tumor_type (single-"
                        "label multi-class); for the binary tumor/normal task this is a no-op since "
                        "'not tumor' vs '10 possible tumor types' isn't a single ground-truth CAM target "
                        "the same way. Only meaningful when the split actually has GT labels (i.e. not "
                        "at real deployment time on unlabeled images). 'ground_truth' remains a "
                        "backward-compatible alias for 'image_label'.")
    parser.add_argument("--debug", action="store_true",
                        help="Save per-image debug outputs (SAM masks, prompt overlays, scores)")
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=None,
        help="Optional diagnostics-only directory for compact per-image CAM/SAM candidate pools. "
             "The cache contains no ground-truth masks and never affects pseudo-mask generation.",
    )
    parser.add_argument(
        "--skip-existing-candidate-cache", action="store_true",
        help="Resume full generation by skipping tumor images whose candidate NPZ already exists.",
    )
    parser.add_argument("--evaluate-prompt-quality", action="store_true",
        help="Log CAM localization and point-prompt hit-rate against ground-truth "
                        "masks to prompt_quality.csv. Isolates CAM/prompt failure from SAM/mask-"
             "selection failure, unlike the final pseudo-mask Dice/IoU. Only meaningful "
             "on images that actually have a lesion/bone GT mask.")
    parser.add_argument("--force-non-normal-cam", action="store_true",
                        help="Predicted-protocol A/B: if argmax is normal, condition CAM on the strongest "
                             "non-normal class instead of skipping. Default keeps normal images empty.")
    args = parser.parse_args()
    # Keep raw option names so the canonical profile can distinguish allowed
    # run/hardware inputs from recipe changes and reject the latter explicitly.
    args._explicit_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args


def apply_pipeline_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the tested pipeline configuration without using segmentation GT.

    The profile is deliberately opt-in.  RAM-H1200 keeps its historical
    defaults, while ``--pipeline-profile btxrd_best`` makes the exact BTXRD
    configuration used in the best validation run reproducible.  Recipe-
    critical CLI changes are rejected under this profile; only dataset paths,
    protocol selection, output paths, checkpoint paths, and device placement
    remain machine/run inputs.
    """
    if args.pipeline_profile == "default":
        return args
    if args.dataset != "btxrd":
        raise ValueError("--pipeline-profile btxrd_best requires --dataset btxrd")

    explicit = getattr(args, "_explicit_options", set())
    research = bool(args.research_overrides)

    # A checkpoint is deliberately required for the frozen profile.  The
    # checkpoint is the artifact produced by the separate training run, so
    # silently selecting a stale local model would make a cross-machine run
    # irreproducible.
    if "--classifier-checkpoint" not in explicit:
        raise ValueError(
            "--pipeline-profile btxrd_best requires an explicit "
            "--classifier-checkpoint from the current training run"
        )

    best_sam = ROOT.parent / "sam_vit_b_01ec64.pth"
    if "--sam-checkpoint" not in explicit and best_sam.exists():
        args.sam_checkpoint = best_sam

    profile = BTXRD_BEST_PIPELINE

    def require_or_set(option: str, attribute: str, expected: object) -> None:
        if research and option in explicit:
            return
        if option in explicit and getattr(args, attribute) != expected:
            raise ValueError(
                f"--pipeline-profile btxrd_best fixes {option}={expected!r}; "
                f"received {getattr(args, attribute)!r}"
            )
        setattr(args, attribute, expected)

    require_or_set("--target-columns", "target_columns", ",".join(profile.target_columns))
    require_or_set("--image-size", "image_size", profile.classifier_image_size)
    require_or_set("--sam-image-size", "sam_image_size", profile.sam_image_size)
    require_or_set("--sam-version", "sam_version", profile.sam_version)
    require_or_set("--sam-model-type", "sam_model_type", profile.sam_model_type)
    require_or_set("--cam-percentile", "cam_percentile", profile.cam_percentile)
    require_or_set(
        "--cam-percentile-values",
        "cam_percentile_values",
        ",".join(str(int(value)) for value in profile.cam_percentile_values),
    )
    require_or_set("--max-bone-components", "max_bone_components", profile.max_bone_components)
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
        if research and (option in explicit or f"--disable-{option[2:]}" in explicit):
            continue
        if option in explicit and not getattr(args, attribute):
            raise ValueError(f"--pipeline-profile btxrd_best requires {option}")
        setattr(args, attribute, True)

    forbidden_disable_flags = {
        "--disable-cam-percentile-ensemble",
        "--disable-sam-prompt-ensemble",
        "--disable-all-cam-components",
        "--disable-cam-contrast-normal",
    }
    for option in forbidden_disable_flags:
        if option in explicit and not research:
            raise ValueError(f"--pipeline-profile btxrd_best rejects {option}")

    locked_false = {
        "--sam-preserve-aspect": "sam_preserve_aspect",
        "--sam-single-mask": "sam_single_mask",
        "--include-cam-candidate": "include_cam_candidate",
        "--disable-bone-morphology": "disable_bone_morphology",
        "--cam-refine": "cam_refine",
        "--cam-tta-flip": "cam_tta_flip",
        "--disable-best-per-component": "disable_best_per_component",
        "--force-non-normal-cam": "force_non_normal_cam",
        "--use-clahe": "use_clahe",
    }
    for option, attribute in locked_false.items():
        if research and option in explicit:
            continue
        if option in explicit and getattr(args, attribute):
            raise ValueError(f"--pipeline-profile btxrd_best fixes {option} off")
        setattr(args, attribute, False)

    if "--auxiliary-binary-checkpoint" in explicit and not research:
        raise ValueError("--pipeline-profile btxrd_best does not use an auxiliary binary checkpoint")

    # These fields are metadata only; keep them out of the downstream API so
    # profile selection cannot accidentally become a data source.
    return args


def load_classifier(
    checkpoint_path: Path,
    fallback_num_classes: int,
    device: torch.device,
    expected_target_columns: list[str] | None = None,
    expected_task: str | None = None,
    expected_num_classes: int | None = None,
) -> tuple[DenseNet121AnatomyClassifier, str, str]:
    state = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_target_columns = state.get("target_columns")
    checkpoint_task = state.get("task", "multi-label")
    checkpoint_num_classes = state.get("num_classes", fallback_num_classes)
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
    # num_classes must come from the checkpoint, not be inferred from
    # len(target_columns) at the call site -- a checkpoint trained with
    # target_columns=["tumor_type"] has 1 target column but a 10-class model
    # (see train_classifier.py's save_checkpoint). fallback_num_classes only
    # covers checkpoints saved before this field existed.
    num_classes = checkpoint_num_classes
    model = DenseNet121AnatomyClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_task, state.get("normalization", "imagenet")


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


def write_or_validate_run_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    """Prevent predicted/ground-truth protocol outputs from sharing a mask directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    mask_dir = output_dir / "masks"
    has_existing_masks = mask_dir.exists() and any(mask_dir.glob("*.png"))
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


def tensor_to_rgb_numpy(image_tensor: torch.Tensor, normalization: str = "imagenet") -> np.ndarray:
    """Convert a [3,H,W] normalised tensor to [H,W,3] uint8 RGB numpy for SAM."""
    pil = tensor_to_pil(image_tensor.detach().cpu(), normalization=normalization)
    return np.array(pil, dtype=np.uint8)


def main() -> None:
    args = apply_pipeline_profile(parse_args())
    ablation_methods = tuple(dict.fromkeys(
        method.strip() for method in args.selection_ablation_methods.split(",") if method.strip()
    ))
    unknown_ablation_methods = sorted(set(ablation_methods) - set(SELECTION_METHODS))
    if unknown_ablation_methods:
        raise ValueError(
            f"Unknown --selection-ablation-methods {unknown_ablation_methods}; "
            f"choose from {list(SELECTION_METHODS)}"
        )
    prompt_score_weights = parse_prompt_score_weights(args.prompt_score_weights)
    layercam_weights = parse_layercam_weights(args.layercam_weights)
    cam_percentile_values = parse_cam_percentile_values(args.cam_percentile_values)
    cam_multiscale_sizes = parse_cam_multiscale_sizes(args.cam_multiscale_sizes)
    if args.target_columns is None:
        target_columns = list(DATASET_TARGET_COLUMNS[args.dataset])
    else:
        target_columns = [c.strip() for c in args.target_columns.split(",") if c.strip()]

    # class_names is what active_indices/cls_i actually index into when
    # saving debug overlays: for the binary/multi-label "tumor" task, that's
    # target_columns itself (1 element == 1 class). For "tumor_type" (single-
    # label multi-class), target_columns is just ["tumor_type"] (1 element,
    # naming the classification head, not the classes), while the model has
    # 10 real classes -- target_columns[cls_i] would IndexError for any
    # cls_i >= 1. Use the real 10-class names in that case instead.
    if target_columns == ["tumor_type"]:
        from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
        class_names = list(TUMOR_TYPE_CLASS_NAMES)
    else:
        class_names = target_columns

    morphology = get_morphology_module(args.dataset)
    # NOTE: tumor_morphology.py's build_tumor_guidance() used to hard-cap the
    # support threshold at percentile-55 regardless of the support_percentile
    # value passed in (that cap has been removed -- see tumor_morphology.py).
    # BTXRD's default here is kept at 55.0 (matching the old, always-applied
    # cap) while a better value is investigated: raising it to 78.0 was tried
    # and tested worse overall (higher selection_loss_dice on this project's
    # own oracle diagnostic) and caused overcorrection (support cutting into
    # good candidates) on some images, so a single fixed percentile is not
    # yet a clear improvement -- an adaptive per-image threshold may be
    # needed instead. Revisit before changing this default again.
    default_seed_percentile, default_support_percentile = (
        (88.0, 68.0) if args.dataset == "ramh1200" else (82.0, 55.0)
    )
    bone_seed_percentile = (
        args.bone_seed_percentile if args.bone_seed_percentile is not None else default_seed_percentile
    )
    bone_support_percentile = (
        args.bone_support_percentile if args.bone_support_percentile is not None else default_support_percentile
    )

    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = default_device if args.classifier_device == "auto" else torch.device(args.classifier_device)
    sam_device = device if args.sam_device == "auto" else torch.device(args.sam_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--classifier-device cuda requested but CUDA is unavailable")
    if sam_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--sam-device cuda requested but CUDA is unavailable")
    expected_profile_columns = (
        list(BTXRD_BEST_PIPELINE.target_columns)
        if args.pipeline_profile == BTXRD_BEST_PIPELINE.name
        else None
    )
    classifier, classifier_task, classifier_normalization = load_classifier(
        args.classifier_checkpoint,
        len(target_columns),
        device,
        expected_target_columns=expected_profile_columns,
        expected_task="single-label" if expected_profile_columns is not None else None,
        expected_num_classes=10 if expected_profile_columns is not None else None,
    )
    print(f"Loaded classifier checkpoint task={classifier_task} normalization={classifier_normalization}")

    write_or_validate_run_metadata(
        args.output_dir,
        {
            "pipeline_profile": args.pipeline_profile,
            "research_overrides": args.research_overrides,
            "dataset": args.dataset,
            "split": args.split,
            "target_columns": target_columns,
            "cam_target_class": args.cam_target_class,
            "force_non_normal_cam": args.force_non_normal_cam,
            "classifier_task": classifier_task,
            "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
            "auxiliary_binary_checkpoint": (
                str(args.auxiliary_binary_checkpoint.resolve())
                if args.auxiliary_binary_checkpoint else None
            ),
            "auxiliary_binary_weight": args.auxiliary_binary_weight,
            "image_size": args.image_size,
            "sam_image_size": args.sam_image_size,
            "sam_preserve_aspect": args.sam_preserve_aspect,
            "image_list": str(args.image_list.resolve()) if args.image_list else None,
            "sam_version": args.sam_version,
            "sam_model_type": args.sam_model_type,
            "sam_device": str(sam_device),
            "classifier_device": str(device),
            "layercam_weights": list(layercam_weights),
            "layercam_gradient_mode": args.layercam_gradient_mode,
            "sam_checkpoint": str(args.sam_checkpoint.resolve()) if args.sam_checkpoint else None,
            "sam_prompt_mode": args.sam_prompt_mode,
            "sam_prompt_ensemble": args.sam_prompt_ensemble,
            "sam_single_mask": args.sam_single_mask,
            "include_cam_candidate": args.include_cam_candidate,
            "cam_percentile": args.cam_percentile,
            "cam_percentile_ensemble": args.cam_percentile_ensemble,
            "cam_percentile_values": list(cam_percentile_values),
            "confidence_threshold": args.confidence_threshold,
            "cam_refine": args.cam_refine,
            "cam_tta_flip": args.cam_tta_flip,
            "cam_multiscale_sizes": list(cam_multiscale_sizes),
            "cam_aggregation": args.cam_aggregation,
            "cam_contrast_normal": args.cam_contrast_normal,
            "cam_contrast_weight": args.cam_contrast_weight,
            "cam_refine_layer": args.cam_refine_layer,
            "cam_refine_high_percentile": args.cam_refine_high_percentile,
            "cam_refine_low_percentile": args.cam_refine_low_percentile,
            "cam_refine_strength": args.cam_refine_strength,
            "morphology_fusion_mode": args.morphology_fusion_mode,
            "min_component_area": args.min_component_area,
            "max_bone_components": args.max_bone_components,
            "all_cam_components": args.all_cam_components,
            "points_per_component": args.points_per_component,
            "bbox_padding_ratio": args.bbox_padding_ratio,
            "negative_points_per_component": args.negative_points_per_component,
            "max_box_area_ratio": args.max_box_area_ratio,
            "selection_method": args.selection_method,
            "selection_ablation_methods": list(ablation_methods),
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
        },
    )

    dataset = build_classification_dataset(
        args.dataset,
        root=args.ram_root,
        split=args.split,
        target_columns=target_columns,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        preprocessing_mode=args.preprocessing_mode,
        normalization=classifier_normalization,
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
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        unsharded_count = len(dataset.samples)
        dataset.samples = [
            sample for index, sample in enumerate(dataset.samples)
            if index % args.num_shards == args.shard_index
        ]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(dataset.samples)}/{unsharded_count} samples"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Maps image_name -> human-readable tumor_type class name, for the
    # per-tumor-type oracle breakdown in prompt_quality.csv (only meaningful
    # for --dataset btxrd; empty dict elsewhere, and empty for
    # target_columns=["tumor"] samples since load_btxrd_records always
    # populates "tumor_type" on every record regardless of which head is
    # trained, so this works even if this run's classifier is a binary one).
    tumor_type_by_name: dict[str, str] = {}
    if args.dataset == "btxrd":
        from datasets.btxrd import TUMOR_TYPE_CLASS_NAMES
        for sample in dataset.samples:
            tumor_type_by_name[str(sample["image_id"])] = TUMOR_TYPE_CLASS_NAMES[int(sample["tumor_type"])]

    gt_masks_by_name: dict[str, np.ndarray] = {}
    if args.evaluate_prompt_quality:
        seg_dataset = build_segmentation_dataset(
            args.dataset,
            root=args.ram_root,
            split=args.split,
            image_size=args.image_size,
            augment=False,
        )
        for index in range(len(seg_dataset)):
            _, mask_tensor, image_name = seg_dataset[index]
            gt_masks_by_name[str(image_name)] = (mask_tensor[0].numpy() > 0.5)
        print(f"Loaded {len(gt_masks_by_name)} ground-truth masks for prompt-quality evaluation")

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
            args.auxiliary_binary_checkpoint, 1, device
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
        auto_download=(args.sam_checkpoint is None),
        device=str(sam_device),
        sam_version=args.sam_version,
        sam2_model_cfg=args.sam2_model_cfg,
        sam_model_type=args.sam_model_type,
    )

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    ablation_mask_dirs = {
        method: args.output_dir / "ablation_masks" / method
        for method in ablation_methods
    }
    for directory in ablation_mask_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    prompt_quality_rows: list[list[object]] = []
    ablation_rows: list[list[object]] = []
    skipped_image_names: list[str] = []

    skipped = 0
    processed = 0
    visualized = 0
    process_limit = None if args.process_all or args.max_images <= 0 else args.max_images
    use_ground_truth_class = (
        args.cam_target_class in {"image_label", "ground_truth"}
        and target_columns == ["tumor_type"]
    )
    try:
        for images, targets, image_names in tqdm(loader, desc="pseudo-masks"):
            images = images.to(device)

            for idx, image_name in enumerate(image_names):
                if process_limit is not None and processed >= process_limit:
                    break
                image_tensor = images[idx : idx + 1]  # [1,3,H,W]
                mask_path = mask_dir / f"{Path(image_name).stem}.png"
                save_visuals = visualized < max(0, args.save_visuals_limit)
                if (
                    args.skip_existing_candidate_cache
                    and args.candidate_cache_dir is not None
                    and int(targets[idx].item()) != 0
                    and (args.candidate_cache_dir / f"{Path(image_name).stem}.npz").exists()
                ):
                    processed += 1
                    continue

                # ── 1. Classifier forward ─────────────────────────────────────
                gt_class: int | None = None
                with torch.no_grad():
                    logits = classifier(image_tensor)
                    if use_ground_truth_class:
                        # One-hot the true tumor_type class instead of the
                        # classifier's own prediction -- LayerCAM/generate_
                        # fused_cam's confidence-filtering path always fires
                        # for exactly this one class (weight 1.0 >= any
                        # confidence_threshold <= 1.0), so this always
                        # conditions CAM on the GT class, never the
                        # prediction. Still an image-level label only (no
                        # polygon/bbox), so this stays within WSSS.
                        gt_class = int(targets[idx].item())
                        class_weights = np.zeros(logits.shape[1], dtype=np.float32)
                        class_weights[gt_class] = 1.0
                    else:
                        class_weights = classifier_class_weights(logits, classifier_task)

                # For multi-label checkpoints, low confidence can mean no reliable anatomy class.
                # For single-label checkpoints, LayerCAM will fall back to the top softmax class.
                # For ground_truth-conditioned CAM, gt_class==0 means "normal" (no tumor at all,
                # see TUMOR_TYPE_CLASS_NAMES[0]) -- there is no lesion class to condition LayerCAM
                # on, so this must still skip exactly like the low-confidence path does, or a
                # normal image would get a CAM/pseudo-mask generated for the "normal" class as if
                # it were a lesion type.
                should_skip = (
                    (
                        target_columns == ["tumor_type"]
                        and should_skip_tumor_type(class_weights, use_ground_truth_class, gt_class)
                        and not (args.force_non_normal_cam and not use_ground_truth_class)
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
                    for directory in ablation_mask_dirs.values():
                        save_mask(empty_mask, directory / mask_path.name)
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
                    # Explicit end-to-end detection-recall ablation: retain
                    # the strongest non-normal class when normal wins the
                    # softmax, without using any polygon/bbox information.
                    non_normal = np.asarray(class_weights, dtype=np.float32).copy()
                    non_normal[0] = -np.inf
                    selected = int(np.argmax(non_normal))
                    class_weights = np.zeros_like(class_weights, dtype=np.float32)
                    class_weights[selected] = 1.0

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
                    # Contrastive evidence is conditioned on the selected
                    # image-level class only.  The normal class is skipped
                    # above, so this never creates a lesion CAM for a normal
                    # image.  No polygon/bbox information enters this path.
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

                # ── 2b. Optional feature-guided CAM refinement ─────────────────
                if args.cam_refine:
                    feature_map = extract_feature_map(
                        classifier, image_tensor, layer_name=args.cam_refine_layer
                    )
                    fused_cam = refine_cam_with_feature_affinity(
                        fused_cam,
                        feature_map,
                        high_conf_percentile=args.cam_refine_high_percentile,
                        low_conf_percentile=args.cam_refine_low_percentile,
                        propagation_strength=args.cam_refine_strength,
                    )

                image_pil = tensor_to_pil(image_tensor[0].detach().cpu(), normalization=classifier_normalization)
                image_rgb = tensor_to_rgb_numpy(image_tensor[0], normalization=classifier_normalization)
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
                if not args.disable_bone_morphology:
                    if args.morphology_fusion_mode == "components":
                        active_weights = [float(class_weights[i]) for i in active_indices]
                        if args.dataset == "btxrd":
                            # tumor_morphology.build_class_conditioned_components is the
                            # single-CAM-threshold + largest-component implementation;
                            # bone_morphology's is a different (RAM-H1200-specific) signature.
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
                                    max_components=(args.max_bone_components if args.all_cam_components else 1),
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
                            if args.all_cam_components and len(bone_components) > args.max_bone_components * len(prompt_percentiles):
                                bone_components = bone_components[: args.max_bone_components * len(prompt_percentiles)]
                            if bone_likelihood is None:
                                bone_likelihood = np.zeros_like(fused_cam, dtype=np.float32)
                            if bone_support is None:
                                bone_support = np.zeros_like(fused_cam, dtype=np.uint8)
                        else:
                            bone_likelihood, bone_support, bone_components = morphology.build_class_conditioned_components(
                                image_rgb,
                                per_class_cams,
                                active_weights,
                                seed_percentile=bone_seed_percentile,
                                support_percentile=bone_support_percentile,
                                min_component_area=max(20, args.min_component_area // 2),
                                max_components=args.max_bone_components,
                                points_per_component=args.points_per_component,
                                bbox_padding_ratio=args.bbox_padding_ratio,
                                debug_dir=debug_dir,
                            )
                    else:
                        bone_likelihood, bone_support = morphology.build_bone_guidance(
                            image_rgb,
                            fused_cam,
                            seed_percentile=bone_seed_percentile,
                            support_percentile=bone_support_percentile,
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
                # The classifier/CAM grid can be much smaller than the
                # original radiograph (BTXRD images are commonly 500--2500px
                # wide).  Optionally run SAM on a higher-resolution square
                # copy loaded from disk, then map its masks back to the CAM
                # grid before scoring.  All component/prompt geometry remains
                # derived from image-level CAM only.
                sam_image_rgb = image_rgb
                sam_image_pil = image_pil
                sam_components = bone_components
                candidate_prompt_modes: list[str] = []
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

                if sam_components:
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

                if sam_masks.shape[-2:] != (args.image_size, args.image_size):
                    sam_masks = F.interpolate(
                        torch.from_numpy(sam_masks.astype(np.float32))[:, None],
                        size=(args.image_size, args.image_size),
                        mode="nearest",
                    )[:, 0].numpy() > 0.5

                # The CAM component is already a valid weakly-supervised
                # candidate, not a ground-truth-derived mask.  Keeping it in
                # the candidate pool gives selection a safe fallback when
                # SAM's mask decoder fails on radiographic texture.  It is
                # opt-in because an over-diffuse CAM should not silently
                # change the historical SAM-only baseline.
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

                if debug_dir is not None:
                    np.savez_compressed(
                        Path(debug_dir) / "candidate_diagnostics.npz",
                        masks=sam_masks.astype(np.uint8),
                        sam_scores=np.asarray(sam_scores, dtype=np.float32),
                        component_ids=np.asarray(component_ids if component_ids is not None else np.zeros(len(sam_masks), dtype=np.int32)),
                        fused_cam=fused_cam.astype(np.float32),
                        component_masks=(
                            np.stack([component.mask for component in bone_components]).astype(np.uint8)
                            if bone_components else np.zeros((0, args.image_size, args.image_size), dtype=np.uint8)
                        ),
                        prompt_modes=np.asarray(candidate_prompt_modes, dtype="U16"),
                    )
                if args.candidate_cache_dir is not None:
                    args.candidate_cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        args.candidate_cache_dir / f"{Path(image_name).stem}.npz",
                        masks=sam_masks.astype(np.uint8),
                        sam_scores=np.asarray(sam_scores, dtype=np.float32),
                        component_ids=np.asarray(
                            component_ids
                            if component_ids is not None
                            else np.zeros(len(sam_masks), dtype=np.int32),
                            dtype=np.int32,
                        ),
                        fused_cam=fused_cam.astype(np.float32),
                        bone_support=(
                            bone_support.astype(np.uint8)
                            if bone_support is not None
                            else np.zeros_like(fused_cam, dtype=np.uint8)
                        ),
                        bone_likelihood=(
                            bone_likelihood.astype(np.float32)
                            if bone_likelihood is not None
                            else np.zeros_like(fused_cam, dtype=np.float32)
                        ),
                        component_masks=(
                            np.stack([component.mask for component in bone_components]).astype(np.uint8)
                            if bone_components
                            else np.zeros((0, args.image_size, args.image_size), dtype=np.uint8)
                        ),
                        component_ids_ordered=np.asarray(
                            [int(component.component_id) for component in bone_components], dtype=np.int32
                        ),
                        prompt_modes=np.asarray(candidate_prompt_modes, dtype="U16"),
                    )

                # ── 4b. Prompt-quality metrics (optional, pre-SAM diagnostics) ──
                gt_mask = gt_masks_by_name.get(str(image_name)) if args.evaluate_prompt_quality else None
                prompt_quality_entry: list[object] | None = None
                if gt_mask is not None:
                    all_points = (
                        [point for component in bone_components for point in component.positive_points]
                        if bone_components
                        else point_prompts
                    )
                    all_negative_points = (
                        [
                            point
                            for component in bone_components
                            for point in getattr(component, "negative_points", ())
                        ]
                        if bone_components else []
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
                            if args.max_box_area_ratio <= 0 or box_area_ratio <= args.max_box_area_ratio:
                                all_boxes.append(tuple(component.bbox))
                    # bone_components/bone_support come from morphological
                    # reconstruction (seed+support thresholds, not a single
                    # percentile cut), so compare that concrete mask
                    # directly rather than recomputing a percentile cut on
                    # prompt_map, which would not reflect what SAM actually
                    # receives in this mode.
                    if bone_support is not None:
                        fg_metrics = binary_mask_localization_metrics(bone_support, gt_mask)
                    else:
                        fg_metrics = cam_localization_metrics(prompt_map, gt_mask, percentile=args.cam_percentile)
                        fg_metrics = {
                            "iou": fg_metrics["cam_iou"],
                            "recall": fg_metrics["cam_recall"],
                            "precision": fg_metrics["cam_precision"],
                        }
                    hit_metrics = point_prompt_hit_rate(all_points, gt_mask)
                    negative_metrics = negative_point_rejection_rate(all_negative_points, gt_mask)
                    box_metrics = box_prompt_localization_metrics(all_boxes, gt_mask)
                    prompt_quality_entry = [
                        image_name,
                        tumor_type_by_name.get(str(image_name), ""),
                        fg_metrics["iou"],
                        fg_metrics["recall"],
                        fg_metrics["precision"],
                        hit_metrics["point_hit_rate"],
                        hit_metrics["num_points"],
                        hit_metrics["num_hits"],
                        negative_metrics["negative_rejection_rate"],
                        negative_metrics["num_negative_points"],
                        negative_metrics["num_false_negatives"],
                        box_metrics["box_recall"],
                        box_metrics["box_precision"],
                    ]

                # ── 5. CAM-guided mask selection ──────────────────────────────
                component_masks_for_selection = (
                    np.stack([component.mask for component in bone_components])
                    if bone_components else None
                )
                positive_points_for_selection = (
                    {
                        int(component.component_id): tuple(component.positive_points)
                        for component in bone_components
                    }
                    if bone_components else None
                )
                negative_points_for_selection = (
                    {
                        int(component.component_id): tuple(getattr(component, "negative_points", ()))
                        for component in bone_components
                    }
                    if bone_components else None
                )

                def _select(selection_method: str) -> np.ndarray:
                    return select_and_fuse_masks(
                        sam_masks,
                        fused_cam,
                        mask_score_threshold=args.mask_score_threshold,
                        selection_method=selection_method,
                        fusion_topk=args.fusion_topk,
                        bone_likelihood=bone_likelihood,
                        bone_support=bone_support,
                        sam_scores=sam_scores,
                        component_ids=component_ids,
                        component_masks=component_masks_for_selection,
                        prompt_modes=np.asarray(candidate_prompt_modes, dtype="U16"),
                        positive_points_by_component=positive_points_for_selection,
                        negative_points_by_component=negative_points_for_selection,
                        prompt_hybrid_weights=prompt_score_weights,
                        prompt_area_target=args.prompt_area_target,
                        prompt_area_log_sigma=args.prompt_area_log_sigma,
                        best_per_component=component_ids is not None and not args.disable_best_per_component,
                        component_topk=args.component_topk,
                        support_clip_kernel=args.support_clip_kernel,
                    )

                refined = _select(args.selection_method)
                ablation_refined = {
                    method: _select(method)
                    for method in ablation_methods
                }

                # ── 5b. SAM-vs-selection oracle diagnostic (optional) ───────────
                if prompt_quality_entry is not None:
                    oracle_metrics = oracle_vs_selected_metrics(
                        sam_masks,
                        refined,
                        gt_mask,
                        bone_support=bone_support,
                        selection_method=args.selection_method,
                        support_clip_kernel=args.support_clip_kernel,
                    )
                    prompt_quality_entry.extend([
                        oracle_metrics["best_single_dice"],
                        oracle_metrics["best_single_dice_clipped"],
                        oracle_metrics["selected_dice"],
                        oracle_metrics["gap_dice"],
                        oracle_metrics["support_loss_dice"],
                        oracle_metrics["selection_loss_dice"],
                    ])

                # ── 6. Morphological refinement ───────────────────────────────
                final_mask = morphological_refinement(
                    refined,
                    closing_kernel=args.closing_kernel,
                    opening_kernel=args.opening_kernel,
                    min_size=args.min_size,
                    guidance_map=bone_likelihood,
                    guidance_threshold=args.guidance_threshold,
                    max_hole_area=args.max_hole_area,
                )
                ablation_final_masks = {
                    method: morphological_refinement(
                        candidate_mask,
                        closing_kernel=args.closing_kernel,
                        opening_kernel=args.opening_kernel,
                        min_size=args.min_size,
                        guidance_map=bone_likelihood,
                        guidance_threshold=args.guidance_threshold,
                        max_hole_area=args.max_hole_area,
                    )
                    for method, candidate_mask in ablation_refined.items()
                }

                if gt_mask is not None:
                    for method, candidate_mask in ablation_refined.items():
                        metrics = oracle_vs_selected_metrics(
                            sam_masks,
                            candidate_mask,
                            gt_mask,
                            bone_support=bone_support,
                            selection_method=method,
                            support_clip_kernel=args.support_clip_kernel,
                        )
                        final_metrics_for_method = binary_overlap_metrics(
                            ablation_final_masks[method], gt_mask
                        )
                        ablation_rows.append([
                            image_name,
                            tumor_type_by_name.get(str(image_name), ""),
                            method,
                            metrics["best_single_dice"],
                            metrics["best_single_dice_clipped"],
                            metrics["selected_dice"],
                            metrics["support_loss_dice"],
                            metrics["selection_loss_dice"],
                            final_metrics_for_method["dice"],
                        ])

                if prompt_quality_entry is not None:
                    final_metrics = binary_overlap_metrics(final_mask, gt_mask)
                    selected_dice = float(prompt_quality_entry[15])
                    prompt_quality_entry.extend([
                        final_metrics["dice"],
                        final_metrics["iou"],
                        final_metrics["dice"] - selected_dice,
                    ])
                    prompt_quality_rows.append(prompt_quality_entry)

                # ── 7. Save ───────────────────────────────────────────────────
                save_mask(final_mask, mask_path)
                for method, candidate_mask in ablation_final_masks.items():
                    save_mask(candidate_mask, ablation_mask_dirs[method] / mask_path.name)
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

    if args.evaluate_prompt_quality:
        import csv

        quality_csv = args.output_dir / "prompt_quality.csv"
        with quality_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image_name", "tumor_type", "foreground_iou", "foreground_recall", "foreground_precision",
                "point_hit_rate", "num_points", "num_hits",
                "negative_rejection_rate", "num_negative_points", "num_false_negative_points",
                "box_recall", "box_precision",
                "oracle_best_single_dice", "oracle_best_single_dice_clipped", "selected_dice",
                "oracle_gap_dice", "support_loss_dice", "selection_loss_dice",
                "final_dice", "final_iou", "postprocess_delta_dice",
            ])
            writer.writerows(prompt_quality_rows)

        def _mean(column_index: int) -> float:
            values = [row[column_index] for row in prompt_quality_rows if row[column_index] == row[column_index]]
            return sum(values) / len(values) if values else float("nan")

        print(
            f"Prompt quality ({len(prompt_quality_rows)} images with GT): "
            f"mean foreground_iou={_mean(2):.4f} mean foreground_recall={_mean(3):.4f} "
            f"mean foreground_precision={_mean(4):.4f} mean point_hit_rate={_mean(5):.4f} "
            f"mean negative_rejection={_mean(8):.4f} mean box_recall={_mean(11):.4f}"
        )
        print(
            f"SAM-vs-selection oracle diagnostic (total gap decomposed into support-clip loss "
            f"vs. mask-selection loss): "
            f"mean oracle_best_single_dice={_mean(13):.4f} "
            f"mean oracle_best_single_dice_clipped={_mean(14):.4f} "
            f"mean selected_dice={_mean(15):.4f} mean total_gap={_mean(16):.4f} "
            f"mean support_loss={_mean(17):.4f} mean selection_loss={_mean(18):.4f} "
            f"mean final_dice={_mean(19):.4f} mean postprocess_delta={_mean(21):.4f} "
            "(large support_loss => bone_support under-covers the lesion, fix morphology "
            "seed/support percentiles, not mask_selection.py; large selection_loss => "
            "bone_hybrid scoring is discarding a good clipped candidate, fix mask_selection.py)"
        )
        if any(row[1] for row in prompt_quality_rows):
            print("\nOracle Dice by tumor_type (breaks down where CAM localization fails):")
            by_type: dict[str, list[float]] = {}
            for row in prompt_quality_rows:
                tumor_type_name = row[1]
                oracle_dice = row[13]
                if tumor_type_name and oracle_dice == oracle_dice:  # skip empty/NaN
                    by_type.setdefault(tumor_type_name, []).append(oracle_dice)
            for tumor_type_name, values in sorted(by_type.items(), key=lambda item: -len(item[1])):
                print(f"  {tumor_type_name:<28} n={len(values):>4}  mean_oracle_dice={sum(values)/len(values):.4f}")
        print(f"Saved per-image prompt-quality metrics to {quality_csv}")

    if ablation_methods:
        import csv

        ablation_csv = args.output_dir / "selection_ablation.csv"
        with ablation_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image_name", "tumor_type", "selection_method",
                "oracle_best_single_dice", "oracle_best_single_dice_clipped",
                "selected_dice", "support_loss_dice", "selection_loss_dice", "final_dice",
            ])
            writer.writerows(ablation_rows)
        print("Selection ablation on the shared candidate pool:")
        for method in ablation_methods:
            rows = [row for row in ablation_rows if row[2] == method and row[5] == row[5]]
            mean = lambda index: sum(float(row[index]) for row in rows) / len(rows) if rows else float("nan")
            print(
                f"  {method}: n={len(rows)} oracle_clipped={mean(4):.4f} "
                f"selected_dice={mean(5):.4f} selection_loss={mean(7):.4f} "
                f"final_dice={mean(8):.4f}"
            )
        print(f"Saved selection ablation metrics to {ablation_csv}")


if __name__ == "__main__":
    main()
