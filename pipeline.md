# Weakly-Supervised Bone Segmentation with LayerCAM, Bone Morphology, and SAM

## 1. Objective

The project generates pseudo bone masks from image-level anatomy labels and
uses those pseudo masks to train a U-Net segmentation model.

The target is the visible bone region in an X-ray image, not the complete
hand, leg, hip, shoulder, or surrounding soft-tissue silhouette.

## 2. Current Pipeline

```text
X-ray image
    |
    +--> Multi-label DenseNet121 anatomy classifier
    |        |
    |        +--> sigmoid scores: hand, leg, hip, shoulder
    |        |
    |        +--> per-class multi-layer LayerCAM
    |
    +--> X-ray bone enhancement
             |
             +--> intensity response
             +--> cortical-edge response
             +--> morphological reconstruction
                         |
Per-class CAM ---------> CAM-selected full bone components
                                      |
                                      +--> component bounding box
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
                                U-Net training
                                      |
                          final bone segmentation mask
```

The central design principle is:

> Bone morphology proposes complete candidate structures, LayerCAM selects the
> anatomically relevant candidates, and SAM refines their boundaries.

CAM is not treated as a bone boundary mask. SAM is not expected to infer the
bone target from one unconstrained hotspot.

---

## 3. Stage 1: Multi-Label Anatomy Classification

### Objective

Train a classifier that:

1. predicts which anatomical regions are present;
2. supplies semantic features and class-specific gradients for LayerCAM.

### Input

```python
image.shape = [3, image_size, image_size]
```

The notebook currently uses:

```python
image_size = 384
```

### Labels

```python
[hand, leg, hip, shoulder]
```

These are multi-label targets. More than one value may be positive for mixed or
multiscan images.

Examples:

```python
[1, 0, 0, 0]
[0, 1, 0, 0]
[1, 1, 0, 0]
```

### Model and loss

```python
DenseNet121(pretrained=True)
criterion = BCEWithLogitsLoss()
probabilities = sigmoid(logits)
```

The main pipeline does not use single-label CrossEntropy training.

### Checkpoint

```text
outputs/classifier/best_classifier.pt
```

---

## 4. Stage 2: Per-Class LayerCAM Generation

### Target layers

```python
model.features.denseblock2
model.features.denseblock3
model.features.denseblock4
```

### Multi-layer fusion

For each active anatomy class:

```python
class_cam =
    0.2 * denseblock2_cam
  + 0.3 * denseblock3_cam
  + 0.5 * denseblock4_cam
```

Each class CAM is resized to the input image resolution and normalized to
`[0, 1]`.

### Active classes

```python
class_probability >= confidence_threshold
```

Default:

```python
confidence_threshold = 0.5
```

The per-class CAMs are preserved for class-conditioned morphology. A weighted
fused CAM is also retained for visualization, fallback prompting, and final
mask scoring.

---

## 5. Stage 3: Bone-Specific Morphology

This stage adapts the morphology-CAM fusion idea from MorSeg-CAM-SAM to X-ray
bone segmentation.

The breast-lesion assumptions from the original paper are not used:

- bone candidates are not required to be oval or compact;
- elongated components are not removed;
- no breast parenchymal-layer prior is used;
- fixed BUSI thresholds are not transferred to FracAtlas.

### 5.1 Bone likelihood

For each active class CAM:

```text
X-ray
  -> grayscale
  -> percentile normalization
  -> optional CLAHE
  -> cortical edge response
```

The initial likelihood is:

```python
bone_likelihood =
    0.45 * enhanced_intensity
  + 0.25 * cortical_edge
  + 0.30 * class_cam
```

This map is normalized to `[0, 1]`.

### 5.2 Seed and support masks

Default thresholds:

```python
bone_seed_percentile = 88
bone_support_percentile = 62
```

High-confidence seed pixels must also pass a CAM gate. The lower-confidence
support mask must satisfy:

- bone-likelihood threshold;
- relaxed CAM gate;
- intensity or cortical-edge evidence.

### 5.3 Morphological reconstruction

High-confidence bone seeds are expanded only through connected pixels inside
the support mask.

```text
seed pixels
    -> constrained flood fill inside support
    -> reconstructed bone candidates
```

This is safer than unconstrained dilation because growth cannot freely spread
into surrounding soft tissue.

### 5.4 CAM-selected full components

Each reconstructed connected component is scored using:

```text
CAM recall
CAM precision
mean CAM activation
mean bone likelihood
```

The complete component is retained. CAM selects the component but does not crop
the component to the CAM hotspot.

Default:

```python
max_bone_components = 6
```

### 5.5 Per-class processing

Morphology is performed separately for every active anatomy CAM:

```text
hand CAM     -> hand-supported bone candidates
leg CAM      -> leg-supported bone candidates
hip CAM      -> hip-supported bone candidates
shoulder CAM -> shoulder-supported bone candidates
```

Candidates are merged only after class-specific processing. Components with
IoU greater than or equal to `0.65` are treated as duplicates.

This prevents a strong class CAM from suppressing a weaker class in mixed
images.

---

## 6. Stage 4: Component-Wise Prompt Generation

Every selected `BoneComponent` contains:

```python
component.mask
component.score
component.bbox
component.positive_points
```

### Bounding box

The smallest component bounding box is expanded by:

```python
bbox_padding_ratio = 0.05
```

### Structured positive points

Points are selected deterministically from:

1. the strongest bone-likelihood/CAM response;
2. the component centroid;
3. interior points along the major axis for elongated bones.

Default:

```python
points_per_component = 3
```

Random contour points are not used.

### Prompt modes

```text
point        : strongest point only
joint_points : all structured points in one SAM call
box          : component box only
box_point    : component box and all structured points
```

Default:

```python
sam_prompt_mode = "box_point"
```

Optional deterministic negative points can be placed on background positions
inside the expanded box:

```python
negative_points_per_component = 0
```

They are disabled by default and reserved for ablation.

---

## 7. Stage 5: SAM Candidate Generation

### Model

```text
SAM ViT-B
```

SAM is called once for each selected bone component.

Default behavior:

```python
multimask_output = True
```

Therefore, each component normally produces three SAM candidates. The
`--sam-single-mask` option requests only one mask per component.

Each generated mask retains its `component_id`, allowing candidate selection to
be performed within the morphology proposal that generated it.

---

## 8. Stage 6: Bone-Aware Mask Selection

The default scoring method is:

```python
selection_method = "bone_hybrid"
```

For a SAM mask, the score combines:

```text
mean bone likelihood
mean CAM activation
bone-component recall
bone-component precision
SAM predicted quality
large-mask penalty
```

Approximate implemented weighting:

```python
score =
    0.25 * bone_mean
  + 0.25 * cam_mean
  + 0.20 * component_recall
  + 0.20 * component_precision
  + 0.10 * sam_quality
  - 0.40 * large_mask_penalty
```

### Best mask per component

For component-wise prompting:

```text
three SAM candidates for component 0 -> keep the best candidate
three SAM candidates for component 1 -> keep the best candidate
...
union the selected component masks
```

This replaces global top-3 union as the default behavior. It prevents several
nearly identical candidates from one prompt from dominating the final mask.

The original global ranking and `fusion_topk` behavior remains available when
the component pipeline is disabled.

---

## 9. Stage 7: Conservative Morphological Refinement

The post-processing objective is to remove noise without destroying small bones
or filling normal spaces between bones.

Default sequence:

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

Important constraints:

- opening is disabled by default because erosion may remove phalanges or thin
  wrist structures;
- only enclosed holes up to `max_hole_area` are filled;
- spaces between separate bones should remain background;
- small components are retained more conservatively than in the old pipeline;
- components with weak bone-likelihood support can be discarded.

The result is saved as:

```text
outputs/pseudo_masks/masks/<image_stem>.png
```

---

## 10. Stage 8: U-Net Training

### Training pair

```text
input  = X-ray image
target = generated pseudo bone mask
```

### Model

```text
U-Net
encoder channels: 64 -> 128 -> 256 -> 512 -> 1024
```

### Loss

```python
loss =
    0.5 * BCEWithLogitsLoss()
  + 0.5 * DiceLoss()
```

### Metrics

```text
Dice
IoU
```

When a ground-truth mask directory is supplied, validation should use ground
truth rather than pseudo masks.

### Checkpoint

```text
outputs/segmentation/best_unet.pt
```

---

## 11. Preview and Full Generation Modes

Pseudo-mask generation is expensive because it runs LayerCAM and SAM.

### Preview

The notebook defaults to:

```python
RUN_FULL_PSEUDO_MASKS = False
```

This passes:

```text
--max-images 10
```

Only ten images are processed for visual inspection.

### Full pseudo-mask generation

Before U-Net training:

```python
RUN_FULL_PSEUDO_MASKS = True
```

This passes:

```text
--process-all
```

Even in full mode, overlays and debug images are limited by:

```text
--save-visuals-limit 10
```

The notebook checks that a sufficiently large pseudo-mask set exists before
starting segmentation training.

---

## 12. Baseline and Ablation Modes

### Original CAM-only point baseline

```text
--disable-bone-morphology
--selection-method mean
```

### Previous weighted bone-guidance baseline

```text
--morphology-fusion-mode weighted
--sam-prompt-mode point
```

### Current proposed configuration

```text
--morphology-fusion-mode components
--sam-prompt-mode box_point
--selection-method bone_hybrid
--max-bone-components 6
--points-per-component 3
```

### Prompt ablation

Compare:

```text
point
joint_points
box
box_point
```

### Additional ablations

```text
per-class morphology vs fused-CAM morphology
multimask vs --sam-single-mask
negative points 0 vs 1 vs 2
opening kernel 0 vs 3
different seed/support percentiles
```

---

## 13. Directory Structure

```text
project/
|-- datasets/
|   `-- fracatlas.py
|-- models/
|   |-- classifier.py
|   |-- layercam.py
|   |-- unet.py
|   `-- losses.py
|-- pseudo/
|   |-- generate_layercam.py
|   |-- bone_morphology.py
|   |-- extract_prompts.py
|   |-- sam_refine.py
|   |-- mask_selection.py
|   |-- morphology.py
|   `-- visualization.py
|-- train_classifier.py
|-- generate_pseudo_masks.py
|-- train_segmentation.py
|-- inference.py
`-- visualize_pipeline.py
```

---

## 14. Research Contribution

The proposed method adapts morphology-enhanced CAM-guided SAM from compact
breast-lesion segmentation to projected X-ray bone structures.

The adaptation consists of:

1. bone-specific intensity and cortical-edge likelihood;
2. seed-constrained morphological reconstruction;
3. per-class CAM selection of complete morphology components;
4. component-wise structured box and point prompts;
5. best-mask-per-component selection;
6. conservative post-processing that preserves spaces between bones.

The expected benefit is improved pseudo-mask coverage of complete bone
structures while reducing SAM masks that follow surrounding soft-tissue
silhouettes.

---

## 15. Implementation Status

Implemented:

- multi-label anatomy classifier path;
- multi-layer and per-class LayerCAM;
- bone-specific morphology and reconstruction;
- class-conditioned full-component selection;
- component deduplication;
- structured points and padded bounding boxes;
- point, joint-point, box, and box-point SAM modes;
- optional negative points;
- best SAM mask per morphology component;
- bone-aware scoring;
- conservative post-processing;
- preview/full generation controls;
- visualization of bone guidance, boxes, and points.

Verified locally:

- Python syntax compilation;
- notebook JSON validity;
- Git whitespace checks.

Not yet verified locally:

- end-to-end GPU execution with PyTorch and SAM;
- qualitative quality on the ten-image preview;
- quantitative Dice/IoU improvement.

The next required step is to run the ten-image preview on the GPU environment
and tune the morphology thresholds before generating all pseudo masks.
