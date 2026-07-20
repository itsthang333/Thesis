# BTXRD Weakly Supervised Bone-Tumor Segmentation

This branch contains one production pipeline for segmenting bone tumors on
BTXRD radiographs from image-level labels. The canonical executable workflow
is [thesis_final.ipynb](thesis_final.ipynb); it calls the Python entrypoints
instead of duplicating model logic inside notebook cells.

## Canonical pipeline

```text
audited group split manifest
  -> DenseNet121 image classifier (tumor_type, btxrd_best)
  -> multi-layer LayerCAM + class-vs-normal contrast
  -> tumor morphology + positive/negative point and box prompts
  -> local SAM v1 ViT-B
  -> strict coverage_mass_sam selection
  -> pseudo-mask manifest with hashes/provenance
  -> U-Net trained on train pseudo-masks
  -> validation selection and one locked test report
  -> U-Net-only deployment inference
```

The known image-level class is the canonical WSSS training protocol.
Predicted-class CAM is reported separately as a deployment-oriented
diagnostic. Polygon masks are used only for explicit evaluation, the
fully-supervised oracle baseline, and held-out U-Net validation/test; they do
not enter pseudo-label generation.

## Immutable inputs

Attach these to Kaggle with Internet disabled:

- the repository at one committed revision of branch `pipeline`;
- the BTXRD dataset containing `images/`, `Annotations/`, and its metadata
  spreadsheet/CSV;
- the audited `split_manifest.csv`;
- official `sam_vit_b_01ec64.pth`;
- a local installation artifact for the pinned Segment Anything commit.

Set `BTXRD_GIT_COMMIT`, `BTXRD_SPLIT_MANIFEST`, and `SAM_CHECKPOINT`.
Optionally set `BTXRD_ROOT`, `BTXRD_RUN_ID`, `BTXRD_OUTPUT`, and
`BTXRD_NUM_WORKERS`. The output directory must be new.

Install the locked environment from
[project/requirements.txt](project/requirements.txt), then run the notebook
top-to-bottom. The preflight cell checks dependency versions, repository
revision, dataset/manifest integrity, checkpoint presence, and output
isolation before creating a run.

## Command-line entrypoints

Run commands from `project/`. Representative canonical commands are:

```bash
python tools/build_btxrd_split_manifest.py \
  --dataset-root /path/to/btxrd \
  --output-csv /path/to/split_manifest.csv \
  --report-json /path/to/split_report.json

python train_classifier.py \
  --pipeline-profile btxrd_best \
  --data-root /path/to/btxrd \
  --split-manifest /path/to/split_manifest.csv \
  --output-dir /path/to/classifier_run

python generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root /path/to/btxrd \
  --split-manifest /path/to/split_manifest.csv \
  --split train \
  --classifier-checkpoint /path/to/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --cam-target-class ground_truth \
  --process-all --output-dir /path/to/pseudo_train

python train_segmentation.py \
  --data-root /path/to/btxrd \
  --split-manifest /path/to/split_manifest.csv \
  --train-pred-mask-root /path/to/pseudo_train/masks \
  --output-dir /path/to/unet_run

python evaluate_unet.py \
  --data-root /path/to/btxrd \
  --split-manifest /path/to/split_manifest.csv \
  --split val --checkpoint /path/to/best_unet.pt \
  --output-csv /path/to/unet_val.csv \
  --output-json /path/to/unet_val.json

python inference.py \
  --image-path /path/to/radiograph.png \
  --segmentation-checkpoint /path/to/best_unet.pt \
  --output-dir /path/to/inference
```

Use `evaluate_classifier.py` for the complete classifier and binary
tumor-vs-normal gate report, and `evaluate_pseudo_masks.py` for pseudo-mask
Dice/IoU, boundary metrics, normal-case specificity, and bootstrap confidence
intervals.

## Scientific and reproducibility guards

- The split manifest is authoritative and rejects group overlap or changed
  image hashes.
- Degenerate/non-finite CAMs fail closed and cannot create corner prompts.
- Missing or tampered pseudo-masks fail before U-Net training.
- Candidate thresholds are strict: no candidate below threshold is silently
  promoted.
- Checkpoints store preprocessing, architecture, optimizer/resume state,
  dataset and manifest provenance.
- Final inference restores U-Net preprocessing and decision threshold, checks
  architecture/dataset compatibility, returns original-resolution outputs,
  and records the checkpoint SHA-256.
- Test-set execution is gated by `BTXRD_RUN_LOCKED_TEST=1` after the final
  configuration is frozen.

## Tests

```bash
python -m compileall -q project tests
python -m unittest discover -s tests -v
```

GPU/PyTorch integration tests must pass in the locked Kaggle environment.
A lightweight local environment without PyTorch will explicitly skip those
tests; that is not equivalent to a successful Kaggle smoke run.
