# Installation and usage

This document describes the complete training and inference sequence. Paths
shown in angle brackets are deployment-specific and must be replaced by the
operator. Every output directory must be new or empty.

## Inputs kept outside the repository

Prepare the following external resources:

- BTXRD image directory and its image-level labels;
- a group-aware split manifest created by the included split tool;
- ImageNet DenseNet initialization weights;
- a local SAM ViT-B checkpoint;
- a frozen BiomedCLIP snapshot available in the Hugging Face cache;
- a frozen RAD-DINO model directory;
- writable output directories for every stage.

Record SHA-256 hashes for the split manifest and every model artifact. Commands
fail closed when a supplied artifact differs from its declared hash.

## Environments

Create a shared environment and install the platform-matched PyTorch and
torchvision builds first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a candidate environment for classifiers, BiomedCLIP, LayerCAM, and SAM:

```bash
python -m venv .venv-candidate
source .venv-candidate/bin/activate
python -m pip install --upgrade pip
python -m pip install -r project/requirements-candidate.txt
```

Create a separate G1 environment for RAD-DINO and mask-bag MIL:

```bash
python -m venv .venv-g1
source .venv-g1/bin/activate
python -m pip install --upgrade pip
python -m pip install -r project/requirements-g1.txt
```

On Windows, replace `source <environment>/bin/activate` with the corresponding
`<environment>\\Scripts\\activate` command.

## Interactive end-to-end demonstration

The repository includes
`notebooks/BTXRD_Rich_Gallery_G1_Demo.ipynb`. It runs one manifest-selected
image through every inference stage and displays prediction-time artifacts
without opening spatial annotations.

Install Jupyter and plotting support in the candidate environment:

```bash
python -m pip install -r project/requirements-demo.txt
```

Keep the separate G1 environment described above. The notebook invokes
`.venv-candidate` for BiomedCLIP, LayerCAM, and SAM, then invokes `.venv-g1`
for RAD-DINO and G1. This preserves the two frozen Transformers versions while
still presenting one sequential notebook.

Before running the first cell, provide:

- the external checkpoint package root;
- the BTXRD root;
- the canonical split manifest;
- a validation image ID, or a final frozen test lock when using a test image.

The supplied default image ID is a validation demonstration case. Every heavy
stage writes to `demo_outputs/<image_stem>/`; rerunning that stage deliberately
replaces only its own demo output. The final cell shows the input radiograph,
BiomedCLIP map, fused prompt evidence, and selected native-resolution mask.
No Dice, IoU, polygon, bounding box, or lesion-size information is consumed.

## Build the immutable split manifest

```bash
python project/tools/build_btxrd_split_manifest.py --help
```

Run the tool with the BTXRD root and an empty output path. Keep the resulting
manifest unchanged for all subsequent stages and calculate its SHA-256.

## Train the image-level classifiers

The method uses binary tumor/normal image labels. No polygon or mask directory
is needed in this stage.

Train the anchor classifier:

```bash
python project/train_classifier.py \
  --pipeline-profile default \
  --data-root <BTXRD_ROOT> \
  --train-split train \
  --val-split val \
  --split-manifest <SPLIT_MANIFEST> \
  --target-columns tumor \
  --image-size <ANCHOR_IMAGE_SIZE> \
  --output-dir <CLASSIFIER_ANCHOR_OUTPUT>
```

Train the high-resolution classifier through the reproducible wrapper:

```bash
python project/run_classifier448_supply.py \
  --source-root <REPOSITORY_ROOT> \
  --data-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --densenet-weight <DENSENET_WEIGHT> \
  --expected-densenet-sha256 <DENSENET_SHA256> \
  --source-commit <SOURCE_COMMIT> \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --output-dir <CLASSIFIER_HIGH_RES_OUTPUT>
```

Each checkpoint stores its architecture and normalization metadata. Candidate
generation validates that metadata before loading the classifier.

## Generate and package BiomedCLIP saliency

Run the saliency generator independently for the training and validation
splits. Add `--frozen-config` only for a locked test run.

```bash
python project/generate_biomedclip_saliency.py \
  --dataset-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --split <SPLIT> \
  --model-dir <BIOMEDCLIP_SNAPSHOT_DIR> \
  --output-dir <SALIENCY_ROOT>/<SPLIT> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --expected-model-weight-sha256 <BIOMEDCLIP_WEIGHT_SHA256> \
  --source-commit <SOURCE_COMMIT>
```

Bind the completed split outputs:

```bash
python project/package_biomedclip_saliency_supply.py \
  --root <SALIENCY_ROOT> \
  --splits train,val \
  --output <SALIENCY_ROOT>/saliency_supply_manifest.json
```

## Generate the two candidate supplies

The anchor supply combines LayerCAM at the anchor scale with the packaged
BiomedCLIP saliency. The addition supply uses the high-resolution classifier.
Both invoke SAM and write full candidate diagnostics without opening spatial
annotations.

Anchor supply:

```bash
python project/run_rich_gallery_candidate_supply.py \
  --mode anchor \
  --source-root <REPOSITORY_ROOT> \
  --data-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --classifier-checkpoint <ANCHOR_CHECKPOINT> \
  --expected-classifier-sha256 <ANCHOR_CHECKPOINT_SHA256> \
  --sam-checkpoint <SAM_CHECKPOINT> \
  --expected-sam-sha256 <SAM_SHA256> \
  --external-saliency-supply-root <SALIENCY_ROOT> \
  --expected-external-supply-manifest-sha256 <SALIENCY_SUPPLY_SHA256> \
  --source-commit <SOURCE_COMMIT> \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --output-dir <ANCHOR_SUPPLY_OUTPUT>
```

High-resolution addition supply:

```bash
python project/run_rich_gallery_candidate_supply.py \
  --mode addition \
  --source-root <REPOSITORY_ROOT> \
  --data-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --classifier-checkpoint <HIGH_RES_CHECKPOINT> \
  --expected-classifier-sha256 <HIGH_RES_CHECKPOINT_SHA256> \
  --sam-checkpoint <SAM_CHECKPOINT> \
  --expected-sam-sha256 <SAM_SHA256> \
  --source-commit <SOURCE_COMMIT> \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --output-dir <ADDITION_SUPPLY_OUTPUT>
```

Use `--splits train,val` during development. A test split is accepted only
with a valid frozen protocol.

## Merge the Rich Gallery

Run once for each required split. Supply the hashes written by both preceding
stages:

```bash
python project/merge_frozen_candidate_galleries.py \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --split <SPLIT> \
  --anchor-root <ANCHOR_SPLIT_OUTPUT> \
  --anchor-candidate-manifest-sha256 <ANCHOR_CANDIDATE_SHA256> \
  --anchor-pseudo-manifest-sha256 <ANCHOR_PSEUDO_SHA256> \
  --addition-root <ADDITION_SPLIT_OUTPUT> \
  --addition-candidate-manifest-sha256 <ADDITION_CANDIDATE_SHA256> \
  --addition-pseudo-manifest-sha256 <ADDITION_PSEUDO_SHA256> \
  --addition-namespace classifier448 \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --output-dir <MERGED_GALLERY_OUTPUT>
```

The merger preserves candidate indices, namespaces every source, performs
exact mask deduplication, and writes new immutable manifests.

## Train G1

Activate the G1 environment. G1 consumes the training and validation candidate
bags, their upstream scores, verified X-rays, and binary image labels. It does
not import the polygon evaluator.

```bash
python project/run_rad_dino_mask_bag_mil_probe.py \
  --dataset-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --model-dir <RAD_DINO_MODEL_DIR> \
  --expected-config-sha256 <RAD_DINO_CONFIG_SHA256> \
  --expected-preprocessor-sha256 <RAD_DINO_PREPROCESSOR_SHA256> \
  --expected-weight-sha256 <RAD_DINO_WEIGHT_SHA256> \
  --train-candidate-root <MERGED_TRAIN_GALLERY> \
  --train-candidate-manifest-sha256 <TRAIN_CANDIDATE_SHA256> \
  --train-pseudo-manifest-sha256 <TRAIN_PSEUDO_SHA256> \
  --val-candidate-root <MERGED_VAL_GALLERY> \
  --val-candidate-manifest-sha256 <VAL_CANDIDATE_SHA256> \
  --val-pseudo-manifest-sha256 <VAL_PSEUDO_SHA256> \
  --source-commit <SOURCE_COMMIT> \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --rich-gallery-union \
  --output-dir <G1_OUTPUT>
```

Training emits the final G1 checkpoint, image-level history, candidate logits,
and provenance receipts. Checkpoint selection must use the declared protocol;
spatial validation annotations are not available to this command.

## Score and freeze inference choices

Score a frozen gallery with G1:

```bash
python project/score_final_rich_gallery.py \
  --split <SPLIT> \
  --dataset-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --model-dir <RAD_DINO_MODEL_DIR> \
  --expected-config-sha256 <RAD_DINO_CONFIG_SHA256> \
  --expected-preprocessor-sha256 <RAD_DINO_PREPROCESSOR_SHA256> \
  --expected-weight-sha256 <RAD_DINO_WEIGHT_SHA256> \
  --candidate-root <MERGED_GALLERY> \
  --candidate-manifest-sha256 <CANDIDATE_SHA256> \
  --pseudo-manifest-sha256 <PSEUDO_SHA256> \
  --g1-checkpoint <G1_CHECKPOINT> \
  --expected-g1-checkpoint-sha256 <G1_CHECKPOINT_SHA256> \
  --source-commit <SOURCE_COMMIT> \
  --protocol-sha256 <PROTOCOL_SHA256> \
  --output-dir <G1_SCORE_OUTPUT>
```

Freeze one mask choice per image:

```bash
python project/freeze_final_rich_gallery.py \
  --split <SPLIT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --g1-diagnostic-root <G1_SCORE_OUTPUT> \
  --expected-g1-freeze-sha256 <G1_SCORE_FREEZE_SHA256> \
  --candidate-root <MERGED_GALLERY> \
  --expected-candidate-manifest-sha256 <CANDIDATE_SHA256> \
  --expected-pseudo-manifest-sha256 <PSEUDO_SHA256> \
  --output-dir <FROZEN_CHOICES_OUTPUT>
```

The final selector converts G1 logits and upstream scores into within-image
average-tie percentile ranks, gives both ranks equal weight, then applies a
stable argmax. No image, annotation, lesion size, or hand-crafted test rule is
available to the selector.

## Spatial evaluation

Evaluation is a separate command and must be run only after the choice-freeze
hash has been recorded:

```bash
python project/evaluate_final_rich_gallery.py \
  --split <SPLIT> \
  --dataset-root <BTXRD_ROOT> \
  --split-manifest <SPLIT_MANIFEST> \
  --expected-split-sha256 <SPLIT_SHA256> \
  --selection-root <FROZEN_CHOICES_OUTPUT> \
  --expected-selection-freeze-sha256 <CHOICE_FREEZE_SHA256> \
  --candidate-root <MERGED_GALLERY> \
  --output-dir <EVALUATION_OUTPUT>
```

For test inference and evaluation, first create a lock with
`freeze_final_test_protocol.py`:

```bash
python project/freeze_final_test_protocol.py \
  --split-manifest <SPLIT_MANIFEST> \
  --classifier-checkpoint <ANCHOR_CHECKPOINT> \
  --classifier-checkpoint <HIGH_RES_CHECKPOINT> \
  --sam-checkpoint <SAM_CHECKPOINT> \
  --g1-checkpoint <G1_CHECKPOINT> \
  --artifact biomedclip_weight=<BIOMEDCLIP_WEIGHT> \
  --artifact rad_dino_weight=<RAD_DINO_WEIGHT> \
  --test-run-id <IMMUTABLE_RUN_ID> \
  --output <LOCK_FILE>
```

Pass `--frozen-config <LOCK_FILE>` to every test-aware stage. Do not modify
source, split assignments, checkpoints, candidate configuration, or selector
settings after creating the lock.

## Output contracts

Every major stage writes a JSON receipt or freeze document and one or more CSV
manifests. Downstream stages require both the path and expected SHA-256. Keep
the complete directory tree because relative paths inside manifests are part
of the contract.

The final evaluation directory contains per-image metrics, an aggregate
summary, and an audit document. Those generated outputs are not source code
and must not be committed to this repository.
