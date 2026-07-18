# RAM-H1200 / BTXRD Segmentation Pipeline

## 0. Two Datasets, One Pipeline

This document describes the pipeline as implemented for **RAM-H1200**
(sections 1-13 below). The same DenseNet121 -> LayerCAM -> SAM architecture
also runs on **BTXRD** (bone tumor segmentation) via `--dataset btxrd`, with
two deliberate differences:

The canonical `btxrd_best` profile uses the 10-class `tumor_type`
image-level head. The older `tumor` binary head remains available only for
legacy/default runs; neither branch passes polygon or bounding-box GT into
the WSSS stages.

- **Classifier target**: RAM-H1200 uses `hand` (whole-hand presence); BTXRD
  uses `tumor` (tumor vs normal). Both are still single binary image-level
  labels — the WSSS setup (SAM/morphology never see ground-truth
  polygons/boxes) is unchanged.
- **Morphology prior before SAM**: RAM-H1200's `pseudo/bone_morphology.py`
  assumes the target is radiopaque bone, since a hand-only label makes CAM
  behave like a hand-silhouette map (see Stage 3 below). BTXRD's classifier is
  trained directly on tumor presence, so its CAM is a much stronger
  localization signal; `pseudo/tumor_morphology.py` instead weights CAM more
  heavily (55% vs bone_morphology's 10%) and looks for local intensity
  *anomalies* — both lytic (locally darker) and sclerotic (locally brighter)
  lesions — rather than assuming the target is always the brightest tissue.
  `pseudo/morphology_factory.py` selects between the two by `--dataset`.

Everything downstream of the morphology stage (SAM prompting, `bone_hybrid`
mask selection, Stage 6 morphological refinement) is dataset-agnostic and
unchanged between the two datasets.

BTXRD also differs in one structural way: it ships with no predefined
train/val/test split, so `project/datasets/btxrd.py` derives an 80/10/10
split stratified by normal/benign/malignant with a fixed seed (42).

The selected reproducible BTXRD implementation is available with
`project/generate_pseudo_masks.py --pipeline-profile btxrd_best`. This profile
uses the current best validation configuration: 320 px CE classifier/CAM,
normal-logit contrast, 85/90/95 CAM percentile ensemble, up to three CAM
components, 512 px SAM ViT-B, box+point/point/box prompt ensemble, and
`coverage_mass_sam` candidate selection with one component proposal retained.
The profile is image-level only; polygon masks are constructed only by the
optional `--evaluate-prompt-quality` diagnostics path.
The matching classifier is trained with
`project/train_classifier.py --pipeline-profile btxrd_best`; this fixes
`tumor_type` CrossEntropy training at 320 px, batch size 4, six epochs, seed
42, and disables PuzzleCAM/teacher-attention losses for this selected run.

See `btxrd_kaggle.ipynb` for the BTXRD equivalent of the walkthrough below.

## 1. Objective (RAM-H1200)

This document's sections 1-13 describe the RAM-H1200 branch in detail.

The target is binary visible-bone segmentation on hand radiographs. RAM-H1200
provides full-hand X-ray images and COCO RLE bone instance annotations, which
the loader merges into one binary bone mask per image.

The core research path is still:

```text
RAM-H1200 hand X-ray
    |
    +--> DenseNet121 hand checkpoint
    |        |
    |        +--> LayerCAM from denseblock2/3/4
    |
    +--> X-ray bone morphology
             |
             +--> intensity response
             +--> cortical-edge response
             +--> constrained reconstruction
                         |
LayerCAM --------------> CAM-selected bone components
                                      |
                                      +--> component bounding boxes
                                      +--> structured positive points
                                      |
                                      v
                             SAM ViT-B candidates
                                      |
                             best mask per component
                                      |
                         conservative post-processing
                                      |
                               pseudo bone mask
                                      |
                         Dice/IoU against RAM-H1200 GT
```

RAM-H1200 GT masks also support a supervised U-Net baseline:

```text
RAM-H1200 image + RAM-H1200 GT bone mask -> U-Net -> segmentation mask
```

## 2. Dataset

Expected layout:

```text
RAM-H1200-v1/
`-- Segmentation/
    |-- train/
    |   |-- *.bmp
    |   `-- _annotations_bone_rle.coco.json
    |-- val/ or validation/
    |   |-- *.bmp
    |   `-- _annotations_bone_rle.coco.json
    `-- test/
        |-- *.bmp
        `-- _annotations_bone_rle.coco.json
```

The dataset loader accepts either `val` or `validation`.

The annotation loader excludes non-bone categories by keyword:

```text
soft, tissue, implant, intravenous, cannula, ring, artifact, marker, label, ruler
```

## 3. Stage 1: DenseNet121 Hand Checkpoint

RAM-H1200 is hand-only in this project, so the classifier target is:

```python
["hand"]
```

The DenseNet checkpoint is retained for LayerCAM feature and gradient
extraction. It is not the old multi-anatomy model anymore.

Command:

```bash
python project/train_classifier.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --target-columns hand \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir project/outputs/classifier
```

Checkpoint:

```text
project/outputs/classifier/best_classifier.pt
```

## 4. Stage 2: LayerCAM

Target layers:

```python
model.features.denseblock2
model.features.denseblock3
model.features.denseblock4
```

Layer fusion:

```python
cam = 0.2 * denseblock2_cam + 0.3 * denseblock3_cam + 0.5 * denseblock4_cam
```

The result is resized to the input resolution and normalized to `[0, 1]`.

## 5. Stage 3: Bone-Specific Morphology

This keeps the morphology-enhanced CAM-guided SAM idea, but adapts it to
projected X-ray bone structures.

The breast-lesion assumptions are not used:

- bone candidates are not required to be oval or compact;
- elongated components are not removed;
- no breast parenchymal-layer prior is used;
- fixed BUSI thresholds are not transferred to RAM-H1200 hand radiographs.

Bone likelihood:

```python
bone_likelihood =
    0.58 * enhanced_intensity
  + 0.32 * cortical_edge
  + 0.10 * cam
```

Default thresholds:

```python
bone_seed_percentile = 88
bone_support_percentile = 68
```

Because RAM-H1200 uses a hand-level classifier label, CAM can behave like a
hand-silhouette map. The morphology stage therefore uses CAM as a weak semantic
anchor and gives more weight to bright radiopaque structures plus cortical edge
response. High-confidence seeds are reconstructed only through stricter
radiographic support, which reduces soft-tissue spread before SAM prompting.

## 6. Stage 4: Component-Wise SAM Prompts

Each selected `BoneComponent` stores:

```python
component.mask
component.score
component.bbox
component.positive_points
```

Default prompt configuration:

```text
sam_prompt_mode = box_point
points_per_component = 3
bbox_padding_ratio = 0.02
max_bone_components = 12
```

Available prompt modes:

```text
point
joint_points
box
box_point
```

## 7. Stage 5: SAM and Mask Selection

SAM is called once per selected bone component. With `multimask_output=True`,
each component usually returns three candidates.

Default mask selection:

```python
selection_method = "bone_hybrid"
```

The score combines:

```text
mean bone likelihood
mean CAM activation
bone-support recall (does the candidate contain the seed support region?)
SAM predicted quality
large-mask penalty (soft ceiling on plausible lesion size)
border-touch penalty
```

An earlier version also penalized candidates for extending *outside* the bone-
support region (`support precision` / `outside-support ratio` / `soft-tissue
penalty`). That assumed the support region should bound the lesion
(`lesion ⊆ support`), but on BTXRD the opposite holds: `bone_support` is a
narrow CAM-percentile seed strictly inside the true lesion (`support ⊆
lesion`), so a correct candidate routinely extends beyond it. Penalizing that
extension discarded good candidates in favor of tiny ones that stayed fully
inside the narrow seed — confirmed via an oracle diagnostic that decomposed
the pseudo-mask Dice gap into a support-clipping term (near zero) and a
mask-selection term (the dominant one). Those three terms were removed; CAM/
bone-likelihood weighting was increased to compensate as the primary
"is this the lesion" signal, and `support recall` (candidate contains the
support seed) was kept since that direction of the assumption still holds.

For component-wise prompting, the best candidate is selected per component and
the selected masks are unioned. In `bone_hybrid` mode, the fused SAM mask is
also constrained by a dilated bone-support map so that SAM can refine local
shape but cannot freely expand to the full hand silhouette.

## 8. Stage 6: Conservative Post-Processing

Default post-processing:

```text
small closing
optional opening
selective hole filling
small-component filtering
bone-guidance component filtering
```

Defaults:

```python
closing_kernel = 5
opening_kernel = 0
max_hole_area = 500
min_size = 40
```

The goal is to remove noise without destroying phalanges, carpal bones, or
normal spaces between separate bones.

## 9. Pseudo-Mask Generation

Preview ten images:

```bash
python project/generate_pseudo_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --sam-checkpoint D:/thesis/sam_vit_b_01ec64.pth \
  --target-columns hand \
  --image-size 384 \
  --max-images 10 \
  --output-dir project/outputs/pseudo_masks
```

Full split:

```bash
python project/generate_pseudo_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --sam-checkpoint D:/thesis/sam_vit_b_01ec64.pth \
  --target-columns hand \
  --image-size 384 \
  --process-all \
  --save-visuals-limit 10 \
  --output-dir project/outputs/pseudo_masks
```

Outputs:

```text
project/outputs/pseudo_masks/masks/<image_stem>.png
project/outputs/pseudo_masks/overlays/<image_stem>_fused_layercam.png
```

## 10. Pseudo-Mask Evaluation

Evaluate generated masks against RAM-H1200 GT:

```bash
python project/evaluate_ramh1200_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --pred-mask-root project/outputs/pseudo_masks/masks \
  --image-size 384
```

Metrics:

```text
Dice
IoU
```

Results are written to:

```text
project/outputs/ramh1200_eval.csv
```

## 11. Supervised U-Net Baseline

Train U-Net directly on RAM-H1200 GT masks:

```bash
python project/train_segmentation.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --train-split train \
  --val-split val \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir project/outputs/segmentation
```

Checkpoint:

```text
project/outputs/segmentation/best_unet.pt
```

## 12. Source Structure

```text
project/
|-- datasets/
|   |-- common.py
|   `-- ramh1200.py
|-- models/
|   |-- classifier.py
|   |-- layercam.py
|   |-- losses.py
|   `-- unet.py
|-- pseudo/
|   |-- bone_morphology.py
|   |-- extract_prompts.py
|   |-- generate_layercam.py
|   |-- mask_selection.py
|   |-- morphology.py
|   |-- sam_refine.py
|   `-- visualization.py
|-- train_classifier.py
|-- generate_pseudo_masks.py
|-- evaluate_ramh1200_masks.py
|-- train_segmentation.py
|-- inference.py
`-- visualize_pipeline.py
```

## 13. Implementation Status

Implemented:

- RAM-H1200 COCO RLE bone-mask loader;
- DenseNet121 hand checkpoint path for LayerCAM;
- multi-layer LayerCAM;
- bone-specific morphology and constrained reconstruction;
- component-wise box/point SAM prompting;
- bone-aware SAM mask selection;
- conservative post-processing;
- pseudo-mask Dice/IoU evaluation against RAM-H1200 GT;
- supervised U-Net baseline on RAM-H1200 GT;
- local and Kaggle notebooks updated for RAM-H1200.

Verified locally:

- Python syntax compilation;
- notebook JSON validity;
- stale dataset flag/reference scan.

Not verified locally:

- end-to-end GPU execution with PyTorch and SAM;
- qualitative ten-image preview;
- final quantitative Dice/IoU.
