# RAM-H1200 Bone Segmentation Pipeline

This thesis project now uses only the RAM-H1200 hand radiograph dataset.

The target is binary visible-bone segmentation on hand X-ray images. RAM-H1200
provides full-hand radiographs and COCO RLE bone instance masks, which this
code merges into one binary bone mask per image.

## Dataset Layout

Expected local layout:

```text
D:/thesis/RAM-H1200-v1/
`-- Segmentation/
    |-- train/
    |   |-- *.bmp
    |   `-- _annotations_bone_rle.coco.json
    |-- val/              # or validation/
    |   |-- *.bmp
    |   `-- _annotations_bone_rle.coco.json
    `-- test/
        |-- *.bmp
        `-- _annotations_bone_rle.coco.json
```

The loader also accepts `--split validation` and falls back between `val` and
`validation` when one of them exists.

## Install

```bash
pip install -r project/requirements.txt
```

SAM is installed from the requirement file. If automatic checkpoint download is
not desired, provide `--sam-checkpoint /path/to/sam_vit_b_01ec64.pth`.

## Main Commands

Train the DenseNet checkpoint used by LayerCAM:

```bash
python project/train_classifier.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --target-columns hand \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir project/outputs/classifier
```

Generate a quick pseudo-mask preview:

```bash
python project/generate_pseudo_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --max-images 10 \
  --output-dir project/outputs/pseudo_masks
```

Generate pseudo masks for a full split:

```bash
python project/generate_pseudo_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --process-all \
  --save-visuals-limit 10 \
  --output-dir project/outputs/pseudo_masks
```

Evaluate generated pseudo masks against RAM-H1200 GT:

```bash
python project/evaluate_ramh1200_masks.py \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --pred-mask-root project/outputs/pseudo_masks/masks
```

Train supervised U-Net on RAM-H1200 GT masks:

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

Run inference on one image:

```bash
python project/inference.py \
  --image-path D:/thesis/RAM-H1200-v1/Segmentation/val/example.bmp \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --segmentation-checkpoint project/outputs/segmentation/best_unet.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --output-dir project/outputs/inference
```

## Source Structure

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

## Notes

- RAM-H1200 is hand-only for this project, so the default classifier target is
  `hand`.
- The classifier is retained only to provide features and gradients for
  LayerCAM. The final quantitative benchmark should use RAM-H1200 ground-truth
  masks via `evaluate_ramh1200_masks.py`.
- `pycocotools` is required because the RAM-H1200 masks are stored as COCO RLE.
