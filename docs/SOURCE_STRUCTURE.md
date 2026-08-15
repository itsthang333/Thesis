# Source structure

The repository is organized around the data flow of the final WSSS method.
Generated datasets and experiment outputs remain external; the frozen final
model assets are committed through Git LFS under `checkpoints/final_method/`.

## Root

| Path | Purpose |
|---|---|
| `README.md` | Method scope and documentation entry points. |
| `requirements.txt` | Shared Python dependencies. |
| `pytest.ini` | Test discovery configuration. |
| `project/` | Executable pipeline source. |
| `tests/` | Unit, contract, and numerical tests for the retained method. |
| `docs/` | Source map and operating instructions only. |
| `notebooks/` | Interactive, annotation-free inference demonstration. |
| `checkpoints/final_method/` | Hash-verified final checkpoints and local model snapshots stored with Git LFS. |

## Pipeline entry points

| File | Responsibility |
|---|---|
| `train_classifier.py` | Train the image-level DenseNet classifier used by LayerCAM. |
| `evaluate_classifier.py` | Evaluate image-level classification without spatial annotations. |
| `run_classifier448_supply.py` | Reproducible high-resolution classifier training wrapper. |
| `generate_biomedclip_saliency.py` | Produce frozen external saliency maps from images and image labels. |
| `package_biomedclip_saliency_supply.py` | Bind saliency outputs and their hashes into one supply manifest. |
| `run_rich_gallery_candidate_supply.py` | Generate the anchor and high-resolution proposal supplies. |
| `generate_pseudo_masks.py` | Core LayerCAM, prompt, SAM, and candidate-diagnostic engine used by the supply wrapper. |
| `merge_frozen_candidate_galleries.py` | Merge and deduplicate candidate supplies without annotations. |
| `run_rad_dino_mask_bag_mil_probe.py` | Train G1 from frozen candidate bags and binary image labels. |
| `score_final_rich_gallery.py` | Apply a frozen G1 checkpoint to a candidate gallery. |
| `freeze_final_rich_gallery.py` | Apply the final rank-fusion selector and freeze mask choices. |
| `evaluate_final_rich_gallery.py` | Open spatial annotations only after freezing and compute metrics. |
| `freeze_final_test_protocol.py` | Bind source, split, checkpoints, and auxiliary assets before test access. |
| `capture_final_run_environment.py` | Record the software and accelerator environment. |
| `demo_final_pipeline.py` | Orchestrate the exact pipeline on one image for the notebook demo. |

## Packages

### `project/datasets`

BTXRD parsing, preprocessing, image-label loading, and split-manifest-backed
dataset factories. Training and inference paths use image labels. Polygon
decoding is called only by the spatial evaluator.

### `project/models`

- DenseNet classifier and LayerCAM feature extraction;
- frozen BiomedCLIP saliency;
- RAD-DINO preprocessing and candidate descriptor construction;
- mask-bag MIL, candidate geometry, and deterministic projection;
- support classes used by the shared proposal-generation engine.

### `project/pseudo`

Prompt extraction, SAM invocation, mask scoring, proposal diagnostics,
morphology, manifests, and visualization helpers. These modules convert soft
localization evidence into a frozen candidate gallery; they do not select the
final candidate using spatial annotations.

### `project/evaluation`

Classification and segmentation metrics plus the fail-closed test guard. The
guard verifies the immutable protocol before any test-aware command can run.

### `project/tools`

Utility for building the group-aware BTXRD split manifest.

## Environment files

- `project/requirements.txt`: shared packages;
- `project/requirements-candidate.txt`: candidate and BiomedCLIP environment;
- `project/requirements-g1.txt`: RAD-DINO and G1 environment.
- `project/requirements-demo.txt`: Jupyter and visualization packages layered on the candidate environment.

Candidate generation and G1 use separate environments because their frozen
model stacks require different Transformers releases.
