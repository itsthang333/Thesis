# Weakly-Supervised Bone Segmentation with LayerCAM and SAM

## 1. Overview

### Baseline Pipeline

```text
Image
↓
DenseNet121
↓
Grad-CAM
↓
Pseudo Mask
↓
U-Net
↓
Bone Segmentation
```

### Proposed Pipeline

```text
Image
↓
DenseNet121
↓
LayerCAM
↓
CAM Aggregation
↓
CAM Normalization
↓
Adaptive Threshold
↓
Connected Components
↓
Peak Extraction
↓
SAM ViT-B
↓
CAM-guided Mask Selection
↓
Morphological Refinement
↓
Final Pseudo Mask
↓
U-Net
↓
Bone Segmentation Mask
```

---

# 2. Stage 1 — Anatomy Classification

## Objective

Train an anatomy classifier using image-level labels.

The classifier serves two purposes:

1. Anatomy prediction.
2. Feature extraction for LayerCAM generation.

---

## Input

```python
image.shape = [3, 512, 512]
```

---

## Labels

Multi-label anatomy labels:

```python
[hand, leg, hip, shoulder]
```

Examples:

```python
[1,0,0,0]
[0,1,0,0]
[0,0,1,0]
[0,0,0,1]
```

---

## Model

```python
DenseNet121(pretrained=True)
```

---

## Output

```python
logits =
[
    l_hand,
    l_leg,
    l_hip,
    l_shoulder
]
```

```python
probs = sigmoid(logits)
```

Example:

```python
[0.92, 0.05, 0.01, 0.02]
```

---

## Loss

```python
BCEWithLogitsLoss()
```

---

## Saved Model

```text
outputs/classifier/best_classifier.pt
```

---

# 3. Stage 2 — Bone Activation Map Generation using LayerCAM

## Objective

Generate localization maps highlighting bone regions.

LayerCAM is selected instead of Grad-CAM because it can utilize information from multiple feature layers and reduce discriminative-region bias.

---

## Input

```text
Image
+
best_classifier.pt
```

---

## Feature Layers

```python
model.features.denseblock2
model.features.denseblock3
model.features.denseblock4
```

---

## LayerCAM Generation

For each anatomy class:

```python
cam2_hand
cam2_leg
cam2_hip
cam2_shoulder
```

```python
cam3_hand
cam3_leg
cam3_hip
cam3_shoulder
```

```python
cam4_hand
cam4_leg
cam4_hip
cam4_shoulder
```

---

## Multi-layer Fusion

Example:

```python
cam_hand =
0.2 * cam2_hand +
0.3 * cam3_hand +
0.5 * cam4_hand
```

Similarly for:

```python
cam_leg
cam_hip
cam_shoulder
```

---

## Resize

Upsample to:

```python
[512,512]
```

using bilinear interpolation.

---

## CAM Aggregation

Classifier confidence is used as weights.

```python
weights = sigmoid(logits)
```

Example:

```python
weights =
[0.92, 0.05, 0.01, 0.02]
```

---

### Weighted Aggregation

```python
bone_cam =
Σ(weights_i * cam_i)
/
Σ(weights_i)
```

---

## CAM Normalization

```python
bone_cam =
(bone_cam - bone_cam.min())
/
(bone_cam.max() - bone_cam.min())
```

---

## Output

```python
bone_cam.shape = [512,512]
bone_cam ∈ [0,1]
```

---

# 4. Stage 3 — Prompt Extraction from Bone CAM

## Objective

Generate point prompts for SAM.

---

## Adaptive Threshold

Threshold is configurable.

```python
cam_percentile = 85
```

or

```python
cam_percentile ∈ {85,90,95}
```

---

```python
threshold =
np.percentile(
    bone_cam,
    cam_percentile
)

fg =
bone_cam > threshold
```

---

## Connected Components

```python
components =
connected_components(fg)
```

---

## Peak Extraction

For each component:

```python
peak =
argmax(
    bone_cam
    inside component
)
```

Example:

```python
[
 (220,145),
 (310,278),
 (180,400)
]
```

---

## Prompt Selection

One peak point is selected from each connected component.

Optionally:

```python
max_points = 5
```

to prevent excessive prompts.

---

## Output

```python
point_prompts
```

---

# 5. Stage 4 — SAM Candidate Mask Generation

## Objective

Recover object boundaries using SAM.

---

## Input

```python
image
point_prompts
```

---

## Model

```python
SAM ViT-B
```

Reason:

* Lightweight
* Easier deployment
* Suitable for thesis-scale experiments

---

## Prediction

```python
masks,
scores,
logits =
sam.predict(
    point_coords=points,
    point_labels=[1]
)
```

---

## Output

```python
Mask_1
Mask_2
...
Mask_N
```

Each mask:

```python
shape = [512,512]
```

---

# 6. Stage 5 — CAM-guided Mask Selection

## Objective

Select masks most consistent with LayerCAM activations.

---

## Mask Scoring

```python
score(mask)
=
mean(
    bone_cam[mask == 1]
)
```

---

## Example

```python
Mask A = 0.84
Mask B = 0.72
Mask C = 0.18
Mask D = 0.05
```

---

## Selection Rule

```python
mask_score_threshold = 0.4
```

```python
selected =
score(mask)
>
mask_score_threshold
```

The threshold will be selected using validation experiments.

---

## Mask Fusion

```python
pseudo_mask =
logical_or(
    selected_masks
)
```

---

## Output

```python
refined_pseudo_mask
```

---

# 7. Stage 6 — Morphological Refinement

## Objective

Remove noise and improve pseudo-label quality.

---

## Binary Closing

```python
binary_closing(
    mask,
    disk(5)
)
```

---

## Binary Opening

```python
binary_opening(
    mask,
    disk(3)
)
```

---

## Fill Holes

```python
binary_fill_holes()
```

---

## Remove Small Objects

```python
remove_small_objects(
    mask,
    min_size=200
)
```

---

## Output

```python
final_pseudo_mask
```

---

# 8. Stage 7 — Segmentation Training

## Objective

Train a segmentation network using refined pseudo labels.

---

## Dataset

Input:

```python
image
```

Target:

```python
final_pseudo_mask
```

---

## Model

```python
U-Net
```

Encoder:

```text
64
128
256
512
1024
```

Decoder:

```text
1024
512
256
128
64
```

---

## Output

```python
pred_mask
```

```python
shape = [1,512,512]
```

---

## Loss

```python
loss =
0.5 * BCEWithLogitsLoss()
+
0.5 * DiceLoss()
```

---

## Metrics

```python
Dice
IoU
Precision
Recall
```

---

## Saved Model

```text
outputs/segmentation/best_unet.pt
```

---

# 9. Inference

## Deployment Inference

After training, only U-Net is required.

```text
Image
↓
U-Net
↓
Bone Segmentation Mask
```

---

## Visualization Pipeline

For qualitative analysis and thesis figures:

```text
Image
↓
DenseNet121
↓
LayerCAM
↓
CAM Aggregation
↓
Peak Extraction
↓
SAM ViT-B
↓
CAM-guided Selection
↓
Pseudo Mask
```

---

# 10. Directory Structure

```text
datasets/
└── fracatlas.py

models/
├── classifier.py
├── layercam.py
├── unet.py
└── losses.py

pseudo/
├── generate_layercam.py
├── extract_prompts.py
├── sam_refine.py
├── mask_selection.py
└── morphology.py

outputs/
├── classifier/
├── pseudo_masks/
└── segmentation/

train_classifier.py
generate_pseudo_masks.py
train_segmentation.py

inference.py
visualize_pipeline.py
```

---

# 11. Experimental Plan

## Experiment 1 — CAM Comparison

Compare:

```text
Grad-CAM
Grad-CAM++
LayerCAM
```

Metrics:

```text
Dice
IoU
CAM Coverage
```

---

## Experiment 2 — Effect of SAM

Compare:

```text
LayerCAM
```

vs

```text
LayerCAM + SAM
```

---

## Experiment 3 — Effect of Morphological Refinement

Compare:

```text
LayerCAM + SAM
```

vs

```text
LayerCAM + SAM + Morphology
```

---

## Experiment 4 — Final Segmentation Performance

Baseline:

```text
DenseNet121
↓
Grad-CAM
↓
Pseudo Mask
↓
U-Net
```

Proposed:

```text
DenseNet121
↓
LayerCAM
↓
SAM
↓
Pseudo Mask
↓
U-Net
```

Metrics:

```text
Dice
IoU
Precision
Recall
```

---

# Expected Research Contribution

## Baseline

```text
DenseNet121
↓
Grad-CAM
↓
Pseudo Mask
↓
U-Net
```

## Proposed

```text
DenseNet121
↓
LayerCAM
↓
CAM-guided SAM Refinement
↓
Morphological Refinement
↓
U-Net
```

## Main Idea

LayerCAM improves bone localization coverage and reduces discriminative-region bias, while SAM refines object boundaries. Their combination generates higher-quality pseudo labels, leading to improved weakly-supervised bone segmentation performance on FracAtlas X-ray images.
