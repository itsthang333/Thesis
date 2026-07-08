from __future__ import annotations

"""Select bone- vs tumor-specific morphology by dataset name.

RAM-H1200 (hand-only label) needs bone_morphology.py's radiopaque-intensity
prior because its CAM only anchors the whole-hand silhouette. BTXRD trains
directly on a tumor-vs-normal label, so its CAM is a much stronger cue and
tumor_morphology.py weights it accordingly (see that module's docstring).
Both modules expose the same function names/signatures so callers only need
to swap which module they call into.
"""

from types import ModuleType

from config import SUPPORTED_DATASETS


def get_morphology_module(dataset: str) -> ModuleType:
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}.")
    if dataset == "ramh1200":
        from pseudo import bone_morphology
        return bone_morphology
    from pseudo import tumor_morphology
    return tumor_morphology
