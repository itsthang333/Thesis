# Reproduction commands

All GPU-heavy commands must run on Kaggle. Paths below are placeholders; hashes
in `run_manifest.json` and `checkpoints/checkpoint_pointer.json` are mandatory.

## Generate checksum-bound train pseudo masks

```bash
python project/generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root /kaggle/input/datasets/itsthang333/btxrd-raw/BTXRD \
  --split train \
  --split-manifest /kaggle/input/source/split_manifest.csv \
  --classifier-checkpoint /kaggle/input/models/best_classifier.pt \
  --sam-checkpoint /kaggle/input/models/sam_vit_b_01ec64.pth \
  --process-all \
  --save-visuals-limit 0 \
  --output-dir /kaggle/working/pseudo_train
```

`btxrd_best` locks the binary `tumor` target, 320 px LayerCAM, horizontal-flip
TTA, percentiles 85/90/95, SAM at 512 px, `coverage_mass_sam`,
`component_topk=3`, and support clipping at 5 px.

## Train the official WSSS segmenter

```bash
python project/train_segmentation.py \
  --pipeline-profile btxrd_best \
  --data-root /kaggle/input/datasets/itsthang333/btxrd-raw/BTXRD \
  --train-split train --val-split val \
  --split-manifest /kaggle/input/source/split_manifest.csv \
  --train-pred-mask-root /kaggle/input/pseudo-train/masks \
  --image-size 448 --model-architecture resnet18_unet \
  --batch-size 8 --lr 0.0001 --weight-decay 0.0001 \
  --epochs 35 --seed 42 --num-workers 4 \
  --early-stop-patience 10 --checkpoint-dice-tolerance 0.0001 \
  --pos-weight-mode manual --pos-weight-value 10 \
  --output-dir /kaggle/working/wsss_segmenter
```

## Archived freeze and one-time final-test command

The final test evaluation below has already completed once. It is retained for
provenance and independent replication, not as authorization to rerun or tune
on the current test split.

Create the frozen config only from a clean committed tree:

```bash
python project/tools/freeze_pipeline_config.py \
  --profile btxrd_best \
  --split-manifest artifacts/data_audit/split_manifest.csv \
  --unet-checkpoint /path/to/hash-verified/best_unet.pt \
  --validation-summary artifacts/official_wsss/segmenter/evaluation/summary.json \
  --threshold-selection artifacts/official_wsss/segmenter/evaluation/threshold_selection.json \
  --threshold 0.85 --image-size 448 --status final \
  --output configs/official_wsss_frozen_test.json
```

The final Kaggle command must contain no threshold grid:

```bash
python project/evaluate_unet.py \
  --data-root /kaggle/input/datasets/itsthang333/btxrd-raw/BTXRD \
  --split test \
  --split-manifest /kaggle/input/source/split_manifest.csv \
  --checkpoint /kaggle/input/models/best_unet.pt \
  --frozen-config /kaggle/input/source/official_wsss_frozen_test.json \
  --image-size 448 --threshold 0.85 \
  --batch-size 8 --num-workers 4 \
  --bootstrap-iterations 10000 --bootstrap-seed 42 \
  --prediction-dir /kaggle/working/official_wsss_test/prediction_masks \
  --qualitative-dir /kaggle/working/official_wsss_test/qualitative \
  --output-csv /kaggle/working/official_wsss_test/evaluation/per_image.csv \
  --output-json /kaggle/working/official_wsss_test/evaluation/summary.json
```
