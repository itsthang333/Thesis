# FracAtlas Anatomy-Aware Weakly Supervised Segmentation Pipeline

This project implements a clean PyTorch baseline for weakly supervised medical image segmentation on the FracAtlas X-ray dataset.

Pipeline:

`X-ray -> DenseNet121 multi-label anatomy classifier -> per-class Grad-CAM -> CAM aggregation -> pseudo bone mask generation -> U-Net segmentation -> final bone mask`

Current target:

- bone vs background
- anatomy-aware localization, not fracture localization

## What Is Included

- `train_classifier.py` trains a DenseNet121 multi-label anatomy classifier.
- `generate_pseudo_masks.py` runs per-class Grad-CAM and aggregates anatomy CAMs into pseudo segmentation masks.
- `train_segmentation.py` trains a U-Net on the pseudo masks.
- `inference.py` runs the full pipeline on one image and saves CAM, pseudo mask, and final segmentation outputs.

Core modules:

- `datasets/fracatlas.py` FracAtlas dataset loading, indexing, CLAHE preprocessing, and train/val splits.
- `models/classifier.py` DenseNet121 classifier with explicit feature extraction for Grad-CAM.
- `models/gradcam.py` manual Grad-CAM implementation using forward and backward hooks plus CAM aggregation helpers.
- `models/unet.py` binary U-Net segmentation model.
- `models/losses.py` Dice, IoU, and combined BCE + Dice losses.
- `pseudo/cam_to_mask.py` CAM normalization, thresholding, morphology, and component filtering.
- `pseudo/visualization.py` CAM overlay and mask saving helpers.

## Data Layout

Expected workspace layout:

```text
project/
FracAtlas/
  dataset.csv
  images/
  Annotations/
    COCO JSON/
    PASCAL VOC/
    YOLO/
```

The code expects `dataset.csv` to contain at least the `image_id` column and one or more anatomy label columns such as `hand`, `leg`, `hip`, and `shoulder`.

By default, the scripts look for images under:

- `FracAtlas/images`

If your images live elsewhere, pass `--image-root` explicitly.

## Environment Setup

The repository was validated with a local virtual environment at:

- `d:/thesis/.venv/Scripts/python.exe`

Install dependencies:

```powershell
cd D:\thesis\project
d:/thesis/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Optional dependency:

- `opencv-python` is only needed if you enable CLAHE preprocessing.

## Stage 1: Train the Classifier

Train the DenseNet121 classifier on FracAtlas anatomy labels.

Default target columns:

- `hand`
- `leg`
- `hip`
- `shoulder`

Example:

```powershell
cd D:\thesis\project
d:/thesis/.venv/Scripts/python.exe train_classifier.py `
  --data-root D:\thesis\FracAtlas `
  --target-columns hand,leg,hip,shoulder `
  --batch-size 8 `
  --image-size 512 `
  --epochs 25 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --output-dir D:\thesis\project\outputs\classifier
```

Useful flags:

- `--no-pretrained` disables ImageNet weights.
- `--use-clahe` applies CLAHE-like contrast enhancement before resizing and normalization.
- `--target-columns hand,leg,hip,shoulder,frontal,lateral,oblique` enables multi-label anatomy and view classification.

What the classifier returns:

- logits with shape `[B, C]`
- final convolutional feature maps with shape `[B, 1024, H/32, W/32]`

Grad-CAM target layer:

- `model.features.denseblock4`

Outputs saved in:

- `outputs/classifier/best.pt`
- `outputs/classifier/last.pt`
- `outputs/classifier/training_log.csv`

## Stage 2: Generate Pseudo Masks

Run Grad-CAM independently for each anatomy class, aggregate the class CAMs, and convert the result into a binary pseudo bone mask.

Example:

```powershell
cd D:\thesis\project
d:/thesis/.venv/Scripts/python.exe generate_pseudo_masks.py `
  --data-root D:\thesis\FracAtlas `
  --checkpoint D:\thesis\project\outputs\classifier\best.pt `
  --target-columns hand,leg,hip,shoulder `
  --percentile 80 `
  --min-area 200 `
  --kernel-size 5 `
  --output-dir D:\thesis\project\outputs\pseudo_masks
```

What happens in this stage:

- Grad-CAM feature activations and gradients are captured from `features.denseblock4`.
- CAMs are produced separately for each anatomy class.
- The per-class CAMs are normalized and fused with weighted averaging or max fusion.
- A percentile threshold turns the aggregated CAM into a binary foreground mask.
- Morphology cleanup removes noise.
- Hole filling restores enclosed bone regions.
- Connected-component filtering removes tiny fragments and keeps the largest component by default.

Saved outputs:

- `outputs/pseudo_masks/masks/*.png`
- `outputs/pseudo_masks/overlays/*.png`

Pseudo mask encoding:

- `0 = background`
- `1 = bone`

## Stage 3: Train the Segmentation Model

Train the U-Net using the pseudo masks as supervision.

Important:

- The segmentation loader expects image/mask files to share the same stem, for example `IMG0000019.jpg` and `IMG0000019.png`.
- Masks should be stored in the directory passed via `--mask-root`.

Example:

```powershell
cd D:\thesis\project
d:/thesis/.venv/Scripts/python.exe train_segmentation.py `
  --data-root D:\thesis\FracAtlas `
  --mask-root D:\thesis\project\outputs\pseudo_masks\masks `
  --batch-size 8 `
  --image-size 512 `
  --epochs 25 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --output-dir D:\thesis\project\outputs\segmentation
```

U-Net tensor flow:

- input image: `[B, 3, H, W]`
- output logits: `[B, 1, H, W]`
- sigmoid probability map: `[B, 1, H, W]`

Loss:

- `BCEWithLogits + Dice`

Metrics:

- Dice
- IoU

Saved outputs:

- `outputs/segmentation/best.pt`
- `outputs/segmentation/last.pt`
- `outputs/segmentation/training_log.csv`

## Stage 4: Full Inference

Run the complete pipeline on a single X-ray:

`image -> anatomy classifier -> per-class Grad-CAM -> CAM aggregation -> pseudo mask -> U-Net -> final mask`

Example:

```powershell
cd D:\thesis\project
d:/thesis/.venv/Scripts/python.exe inference.py `
  --image-path D:\thesis\FracAtlas\images\IMG0000019.jpg `
  --classifier-checkpoint D:\thesis\project\outputs\classifier\best.pt `
  --segmentation-checkpoint D:\thesis\project\outputs\segmentation\best.pt `
  --image-size 512 `
  --percentile 80 `
  --min-area 200 `
  --kernel-size 5 `
  --output-dir D:\thesis\project\outputs\inference
```

Inference outputs:

- `*_cam_overlay.png`
- `*_pseudo_mask.png`
- `*_segmentation_mask.png`
- `*_final_overlay.png`

## Grad-CAM Notes

The Grad-CAM implementation is manual and does not depend on external Grad-CAM libraries.

The tensor flow is:

```text
input image [B, 3, H, W]
  -> DenseNet121 backbone
  -> target layer: features.denseblock4
  -> activations [B, 1024, H/32, W/32]
  -> gradients from selected anatomy class score
  -> channel weights via global average pooling
  -> CAM [B, 1, H/32, W/32]
  -> upsampled CAM [B, H, W]
```

For multi-label anatomy supervision, the classifier predicts several anatomy classes at once, and each class gets its own CAM. These CAMs are then fused into a single attention map that better covers the full bone extent than a fracture-only CAM.

For binary classification, the code uses the single output logit. For multi-label classification, it selects the requested class index or the highest-scoring class.

## Practical Tips

- Start with `--image-size 512` if GPU memory allows.
- If you run out of memory, try `--image-size 384` or `--batch-size 4`.
- If pseudo masks are too sparse, lower `--threshold` slightly.
- If pseudo masks are too noisy, raise `--threshold` or increase `--min-area`.
- Use the pseudo-mask overlays before segmentation training to sanity-check the CAM quality.

## Extending The Baseline

This codebase is intentionally modular so you can add later research ideas without rewriting the pipeline:

- multi-scale CAM
- AdvCAM
- SAM refinement
- S2C refinement

The best insertion points are:

- CAM generation: `models/gradcam.py`
- CAM-to-mask logic: `pseudo/cam_to_mask.py`
- Segmentation model/losses: `models/unet.py` and `models/losses.py`

## Notes

- The project currently assumes the FracAtlas images and labels are available locally.
- The dataset split is a simple seeded random split; if you need a stratified split, add it in `datasets/fracatlas.py` or in the training scripts.
- The code compiles cleanly under the workspace Python environment.
