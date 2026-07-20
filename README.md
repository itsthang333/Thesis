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

Supervision for classifier/CAM generation is the 10-class image-level
`tumor_type` label (normal plus nine tumor types), not a binary mask label.
The known image-level class is the canonical WSSS training protocol.
Predicted-class CAM is reported separately as a deployment-oriented
diagnostic. Polygon masks are used only for explicit evaluation, the
fully-supervised oracle baseline, and held-out U-Net validation/test; they do
not enter pseudo-label generation. Held-out validation polygons are explicitly
used for U-Net checkpoint/model selection, so they are development labels even
though the U-Net training targets remain weak pseudo-masks.

## Kaggle bootstrap

Enable Kaggle Internet and attach the BTXRD dataset containing `images/`,
`Annotations/`, and its metadata spreadsheet/CSV. The first notebook cell then:

- clones the public `pipeline` branch and records its exact commit;
- installs [project/requirements.txt](project/requirements.txt);
- downloads official `sam_vit_b_01ec64.pth`;
- uses a dataset-provided `split_manifest.csv` or deterministically creates the
  group-aware audited manifest under `/kaggle/working`.

The default dataset path is
`/kaggle/input/datasets/wanwin/data-btxrd/BTXRD`. `BTXRD_ROOT`,
`BTXRD_SPLIT_MANIFEST`, and `SAM_CHECKPOINT` remain optional overrides.
`BTXRD_RUN_ID`, `BTXRD_OUTPUT`, and `BTXRD_NUM_WORKERS` can also be overridden;
the output directory must be new. After bootstrap, the preflight verifies exact
dependency versions, repository revision, manifest integrity, SAM presence,
and output isolation before training begins.

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
  --pipeline-profile btxrd_best \
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
  configuration is frozen. Frozen-config schema v3 also requires a classifier
  epoch-budget audit showing a plateau/decline on the audited validation split.
  Every test-split CLI now requires `--frozen-config`;
  its checksum, Git commit, split manifest, SAM, classifier, and both U-Net
  checkpoint hashes are verified before dataset construction.
- HD95/ASSD are reported only as `*_conditional_defined` in pixels on the
  resized evaluation grid (not millimetres), with the eligible,
  excluded, and complete-miss counts beside them. They are not presented as
  unconditional end-to-end means.
- Lesion detection includes maximum one-to-one component matching at IoU >=
  0.10, 0.25, and 0.50, alongside explicitly named any-overlap diagnostics and the GT
  component-count/multifocal distribution.
- Classifier macro-F1 and tumor-gate AUROC, AUPRC, sensitivity, and specificity
  include percentile CIs from complete heuristic-group bootstrap resampling.
  These groups are inferred from filenames/stable metadata and are not verified
  patient identifiers.
- U-Net training writes `pos_weight_audit.json` containing foreground ratio,
  empty-mask rate, raw/clamped/fixed candidate weights, and both raw and
  effective selected weights. Checkpoint selection keeps positive-lesion Dice
  primary and uses normal empty-case specificity as a tolerance-based tie-breaker.
  Set `BTXRD_RUN_POS_WEIGHT_ABLATION=1` to run raw and fixed-10 comparisons
  against the canonical clamped run.
- Pseudo-mask manifest schema v2 distinguishes above-threshold candidates,
  selected candidates/components, SAM calls, and unique prompt points.
- Any WSSS-versus-fully-supervised performance difference is reported as the
  observed gap of this complete pipeline (classifier/CAM/SAM selection,
  pseudo-label noise, optimization, and weak supervision together), not as a
  causal estimate of the cost of weak labels alone.
- Before any full pseudo-mask run, the notebook executes a mandatory one-epoch
  U-Net preflight; its supervised smoke checkpoint is never reported as WSSS.
  The SAM smoke cell processes five deterministic validation images with the
  full prompt ensemble and requires CUDA.

## Tests

```bash
python -m compileall -q project tests
python -m unittest discover -s tests -v
```

GPU/PyTorch integration tests must pass in the locked Kaggle environment.
A lightweight local environment without PyTorch will explicitly skip those
tests; that is not equivalent to a successful Kaggle smoke run.
