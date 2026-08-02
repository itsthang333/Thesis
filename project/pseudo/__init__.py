"""Pseudo-label utilities with dependency-safe lazy public imports."""

from importlib import import_module


_EXPORTS = {
    "generate_fused_cam": ("generate_layercam", "generate_fused_cam"),
    "extract_point_prompts": ("extract_prompts", "extract_point_prompts"),
    "SAMPredictor": ("sam_refine", "SAMPredictor"),
    "select_and_fuse_masks": ("mask_selection", "select_and_fuse_masks"),
    "constrain_to_bone_support": ("mask_selection", "constrain_to_bone_support"),
    "morphological_refinement": ("morphology", "morphological_refinement"),
    "overlay_heatmap": ("visualization", "overlay_heatmap"),
    "save_mask": ("visualization", "save_mask"),
    "save_overlay": ("visualization", "save_overlay"),
    "tensor_to_pil": ("visualization", "tensor_to_pil"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value

__all__ = [
    "SAMPredictor",
    "constrain_to_bone_support",
    "extract_point_prompts",
    "generate_fused_cam",
    "morphological_refinement",
    "overlay_heatmap",
    "save_mask",
    "save_overlay",
    "select_and_fuse_masks",
    "tensor_to_pil",
]
