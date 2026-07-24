# BTXRD weakly supervised bone-tumor segmentation

This `main` branch contains one thesis pipeline:

```text
binary image-level tumor label
  -> DenseNet121 classifier
  -> multi-layer LayerCAM + horizontal-flip TTA
  -> SAM ViT-B pseudo masks selected by coverage_mass_sam
  -> ResNet18 U-Net trained on checksum-bound train pseudo masks
  -> validation-selected checkpoint and threshold
  -> one locked test evaluation
```

The official validation result is WSSS mean tumor Dice **0.230020** at threshold
**0.85**. See [FINAL_RESULTS.md](FINAL_RESULTS.md) and
[artifacts/official_wsss/SELECTION.json](artifacts/official_wsss/SELECTION.json).
The fully supervised Dice **0.495132** snapshot trained directly on polygon
masks and is only an upper-bound diagnostic. It is not the official pipeline.

## Frozen evidence

- Clean split: 2,981 train / 371 validation / 373 test; SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`.
- Binary classifier: F1 0.783333, AUROC 0.865453.
- Validation pseudo masks: mean tumor Dice 0.234339.
- WSSS segmenter: ResNet18 U-Net, selected epoch 13, image size 448,
  checkpoint SHA-256
  `02d3af8feede3c3e650cb76d664185c59092697c1c8306ea67613b89f8407fb4`.
- WSSS validation Dice 95% group-bootstrap CI:
  [0.185729, 0.275031].
- Test status: not evaluated until the schema-v4 frozen config is committed.

Large checkpoints, BTXRD data, caches, temporary Kaggle payloads, and secrets
are excluded from Git. Resolve checkpoints only through
[checkpoint_pointer.json](artifacts/official_wsss/checkpoints/checkpoint_pointer.json).

## Reproduce on Kaggle

GPU-heavy work must run on Kaggle. Attach BTXRD with `images/`, `Annotations/`,
and metadata, then use a clean checkout of the exact commit.

The canonical profile is `btxrd_best`. It locks:

- target `tumor` (binary image-level label);
- classifier image size 320 and no augmentation;
- LayerCAM weights 0.20/0.30/0.50, horizontal-flip TTA, percentiles 85/90/95;
- CAM target from the known image-level label for train pseudo-mask generation;
- SAM image size 512 and box-point/point/box prompt ensemble;
- `coverage_mass_sam`, `component_topk=3`, support clipping 5;
- fail-closed low-score behavior.

Exact generation, training, freeze, and final-evaluation commands are in
[COMMANDS.md](artifacts/official_wsss/COMMANDS.md). The shorter stage sequence is:

```bash
# 1. Generate all train pseudo masks with the locked btxrd_best profile.
python project/generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root /kaggle/input/.../BTXRD \
  --split train \
  --split-manifest artifacts/data_audit/split_manifest.csv \
  --classifier-checkpoint /kaggle/input/.../best_classifier.pt \
  --sam-checkpoint /kaggle/input/.../sam_vit_b_01ec64.pth \
  --process-all --save-visuals-limit 0 \
  --output-dir /kaggle/working/pseudo_train

# 2. Train the U-Net only from pseudo masks; validation polygons select it.
python project/train_segmentation.py \
  --pipeline-profile btxrd_best \
  --data-root /kaggle/input/.../BTXRD \
  --split-manifest artifacts/data_audit/split_manifest.csv \
  --train-pred-mask-root /kaggle/working/pseudo_train/masks \
  --image-size 448 --model-architecture resnet18_unet \
  --batch-size 8 --epochs 35 --early-stop-patience 10 \
  --pos-weight-mode manual --pos-weight-value 10 \
  --output-dir /kaggle/working/wsss_segmenter
```

Do not run a new experiment merely to reproduce the thesis result: the
validation-selected checkpoint already exists by hash. No threshold sweep,
model selection, or qualitative cherry-picking is allowed on test.

## Test lock

`project/tools/freeze_pipeline_config.py` creates a portable schema-v4 record
that binds:

- every `project/**/*.py` source hash;
- split-manifest hash and size;
- official WSSS checkpoint hash and size;
- validation summary and threshold-selection evidence;
- image size 448 and threshold 0.85;
- the sole permitted test stage: official WSSS segmenter evaluation.

`project/evaluate_unet.py` validates that record before constructing the test
dataset. It rejects a threshold grid, rejects the fully supervised checkpoint,
requires fresh prediction/qualitative directories, and writes:

- summary and per-image metrics;
- subgroup metrics, bootstrap confidence intervals, and pixel confusion;
- one prediction-mask PNG per test image;
- deterministic best/median/worst/failure overlays;
- exact command, environment/provenance, hashes, and `test_evaluated=true`.

## Repository layout

```text
artifacts/data_audit/                       clean split and leakage audit
artifacts/official_wsss/                    official compact thesis evidence
artifacts/diagnostics/fully_supervised.../  upper-bound diagnostic only
configs/                                    portable frozen test config
project/                                    executable pipeline and guards
tests/                                      unit and integration guards
```

Expanded experiment history and rejected causal-selector implementation are
preserved on `codex/research-wsss-improvement`. They are intentionally absent
from the `main` execution path.

## Verification

```bash
python -m compileall -q project tests
python -m unittest discover -s tests -v
```

The local lightweight runtime may lack PyTorch/SciPy; that can only yield
explicit skips/import errors, not a full pass. The final verification and smoke
test must run in the locked Kaggle environment before test evaluation.
