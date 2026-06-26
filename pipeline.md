# RAM-H1200 Bone Segmentation Pipeline

## 1. Objective

This project now uses RAM-H1200 as the only dataset.

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
bone-component recall
bone-component precision
SAM predicted quality
large-mask penalty
soft-tissue / low-precision penalty
```

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
