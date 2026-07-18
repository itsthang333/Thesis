# Weakly-Supervised Bone/Tumor Segmentation Pipeline

This thesis project supports two datasets through the same DenseNet121 +
LayerCAM + SAM pipeline, selected via `--dataset`:

- **RAM-H1200** (`--dataset ramh1200`, default): binary visible-bone
  segmentation on hand X-ray images. RAM-H1200 provides full-hand radiographs
  and COCO RLE bone instance masks, which this code merges into one binary
  bone mask per image. The classifier is trained on a `hand` image-level
  label.
- **BTXRD** (`--dataset btxrd`): binary bone-**tumor** segmentation on hand,
  limb, and pelvis X-ray images. BTXRD provides LabelMe polygon annotations
  for tumor lesions plus a `dataset.csv`/`dataset.xlsx` with a `tumor`
  image-level label (tumor vs normal), which drives the classifier/LayerCAM
  stage. BTXRD has no predefined split; this project derives an 80/10/10
  stratified split (by normal/benign/malignant) with a fixed seed.

Because a hand-only weak label makes CAM behave like a hand-silhouette map,
but a tumor-vs-normal label makes CAM a much stronger localization cue, the
two datasets use different morphology priors before SAM prompting:
`pseudo/bone_morphology.py` (radiopaque-intensity prior) for RAM-H1200 and
`pseudo/tumor_morphology.py` (CAM-dominant, local-anomaly prior) for BTXRD.
`pseudo/morphology_factory.py` selects between them based on `--dataset`.

## Dataset Layout — RAM-H1200

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

## Dataset Layout — BTXRD

Download from [kaggle.com/datasets/thanhngan123/btxrd-data](https://www.kaggle.com/datasets/thanhngan123/btxrd-data).
Expected local layout:

```text
BTXRD/
|-- images/
|   `-- IMG######.jpeg
|-- Annotations/
|   `-- IMG######.json        # LabelMe format; tumor images only
`-- dataset.csv                # or dataset.xlsx
```

`dataset.csv`/`dataset.xlsx` has one row per image with a `tumor` column
(1 = tumor present, 0 = normal) plus benign/malignant/subtype/anatomical-site
metadata. Images without a tumor have no `Annotations/*.json` file; the
loader gives them an all-zero ground-truth mask. Reading `dataset.xlsx`
requires `pandas` + `openpyxl` (`pip install pandas openpyxl`); exporting to
`dataset.csv` avoids that extra dependency.

BTXRD ships with no train/val/test split. `project/datasets/btxrd.py` derives
one locally: an 80/10/10 split stratified by normal/benign/malignant, with a
fixed seed (42) so it's reproducible across runs and machines.

## Install

```bash
pip install -r project/requirements.txt
```

SAM is installed from the requirement file. If automatic checkpoint download is
not desired, provide `--sam-checkpoint /path/to/sam_vit_b_01ec64.pth`.

## Main Commands

All commands default to `--dataset ramh1200`. Pass `--dataset btxrd` (with
`--ram-root` pointing at the BTXRD folder) to run the same pipeline on BTXRD
tumor segmentation instead.

Train the DenseNet checkpoint used by LayerCAM:

```bash
python project/train_classifier.py \
  --dataset ramh1200 \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --target-columns hand \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir project/outputs/classifier

# BTXRD equivalent:
python project/train_classifier.py \
  --dataset btxrd \
  --ram-root D:/thesis/BTXRD \
  --target-columns tumor \
  --image-size 384 \
  --batch-size 8 \
  --epochs 25 \
  --output-dir project/outputs/btxrd_classifier
```

Generate a quick pseudo-mask preview:

```bash
python project/generate_pseudo_masks.py \
  --dataset ramh1200 \
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
  --dataset ramh1200 \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --process-all \
  --save-visuals-limit 10 \
  --output-dir project/outputs/pseudo_masks
```

Evaluate generated pseudo masks against ground truth:

```bash
python project/evaluate_ramh1200_masks.py \
  --dataset ramh1200 \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --split val \
  --pred-mask-root project/outputs/pseudo_masks/masks
```

Train supervised U-Net on ground-truth masks:

```bash
python project/train_segmentation.py \
  --dataset ramh1200 \
  --ram-root D:/thesis/RAM-H1200-v1 \
  --train-split train \
  --val-split val \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir project/outputs/segmentation
```

For the independent fully supervised BTXRD comparison branch, run U-Net with
the LabelMe polygon masks. This does not replace or modify the pseudo-mask
pipeline:

```bash
# Fully supervised U-Net trained with BTXRD LabelMe polygon masks
python project/train_segmentation.py \
  --dataset btxrd --ram-root D:/thesis/dataset/BTXRD \
  --train-split train --val-split val \
  --output-dir project/outputs/btxrd_unet_ground_truth
```

`evaluate_unet.py` evaluates the best checkpoint on a locked ground-truth
split and reports tumor-only Dice/IoU separately from normal specificity.
The expanded walkthrough is in `btxrd_kaggle_vi_debug.ipynb`.

Run inference on one image:

```bash
python project/inference.py \
  --dataset ramh1200 \
  --image-path D:/thesis/RAM-H1200-v1/Segmentation/val/example.bmp \
  --classifier-checkpoint project/outputs/classifier/best_classifier.pt \
  --segmentation-checkpoint project/outputs/segmentation/best_unet.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --output-dir project/outputs/inference
```

For a full guided run on BTXRD (environment setup, dataset resolution,
training, pseudo-mask generation, evaluation, visualization), see
`btxrd_kaggle.ipynb`.

### Reproducible BTXRD WSSS profile

The currently selected BTXRD configuration is exposed as an opt-in profile
in `generate_pseudo_masks.py`. It uses the CE DenseNet checkpoint trained at
320 px, LayerCAM contrast against the normal logit, CAM thresholds 85/90/95,
up to three CAM components, SAM ViT-B at 512 px, box+point/point/box prompt
ensemble, and `coverage_mass_sam` selection. Polygon annotations are loaded
only when `--evaluate-prompt-quality` is explicitly requested for diagnostics.

Train the paired classifier on the new machine first:

```bash
python project/train_classifier.py \
  --dataset btxrd \
  --pipeline-profile btxrd_best \
  --ram-root D:/thesis/BTXRD \
  --output-dir project/outputs/btxrd_classifier
```

```bash
python project/generate_pseudo_masks.py \
  --dataset btxrd \
  --pipeline-profile btxrd_best \
  --ram-root D:/thesis/BTXRD \
  --split val \
  --classifier-checkpoint project/outputs/btxrd_classifier/best_classifier.pt \
  --sam-checkpoint D:/thesis/sam_vit_b_01ec64.pth \
  --process-all \
  --output-dir project/outputs/btxrd_best
```

The profile requires an explicit classifier checkpoint so a stale local model
cannot be selected accidentally. It follows the available accelerator for
DenseNet and SAM; pass `--sam-device cpu` if GPU memory is insufficient. The
profile defaults to the predicted-class protocol. For the separate
image-level localization protocol, add `--cam-target-class ground_truth` and
report it separately; this still never supplies a polygon or bounding box to
CAM, prompts, candidate selection, or post-processing.

## Source Structure

```text
project/
|-- datasets/
|   |-- common.py
|   |-- factory.py            # picks RAM-H1200/BTXRD loader by --dataset
|   |-- ramh1200.py
|   `-- btxrd.py
|-- models/
|   |-- classifier.py
|   |-- layercam.py
|   |-- losses.py
|   `-- unet.py
|-- pseudo/
|   |-- bone_morphology.py    # RAM-H1200 prior: radiopaque intensity + edges
|   |-- tumor_morphology.py   # BTXRD prior: CAM-dominant, local anomaly
|   |-- morphology_factory.py # picks the prior above by --dataset
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

- RAM-H1200 is hand-only for this project, so its classifier target is
  `hand`; BTXRD's classifier target is `tumor` (tumor vs normal).
- The classifier is retained only to provide features and gradients for
  LayerCAM. The final quantitative benchmark should use ground-truth masks
  via `evaluate_ramh1200_masks.py --dataset {ramh1200,btxrd}`.
- `pycocotools` is required because the RAM-H1200 masks are stored as COCO
  RLE. `pandas`/`openpyxl` are required only if BTXRD's `dataset.xlsx` is used
  instead of a `dataset.csv` export.
- For BTXRD, SAM and the morphology stage never see the ground-truth
  polygon/bbox annotations — only the image-level `tumor` label drives
  localization, keeping this a weakly-supervised setup consistent with the
  RAM-H1200 branch.
