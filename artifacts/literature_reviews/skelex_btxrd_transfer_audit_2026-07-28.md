# SKELEX transfer audit for image-label-only BTXRD WSSS

Date: 2026-07-28  
Status: literature/provenance audit only; no checkpoint downloaded or job authorized

## Why SKELEX is unusually relevant

Kim et al. introduced SKELEX, a ViT-MAE foundation model pretrained with
self-supervised reconstruction on 1,296,540 musculoskeletal radiographs from
Seoul National University Hospital. Pretraining used no manual labels or
anatomical supervision. The authors explicitly excluded public datasets from
pretraining to avoid leakage in external evaluation.

The model is more anatomically aligned with BTXRD than a chest-dominant
radiology encoder. The paper reports:

- bone-tumor classification evaluation on BTXRD;
- zero-shot reconstruction-error maps on BTXRD, FracAtlas and OAI;
- public release of the pretrained model.

Primary paper:
Kim et al., *A large-scale vision foundation model for musculoskeletal
radiographs*, npj Digital Medicine 2026:
https://doi.org/10.1038/s41746-026-02826-9

Open manuscript:
https://arxiv.org/pdf/2602.03076

Public checkpoint:
https://huggingface.co/skhoha/SKELEX

## Public artifact provenance

The Hugging Face repository was inspected read-only on 2026-07-28:

- repository revision:
  `368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`;
- architecture: `ViTMAEForPreTraining`;
- encoder: ViT-Large, 24 layers, hidden size 1024, 16 attention heads;
- decoder: 8 layers, hidden size 512;
- input: 224x224 RGB, patch size 16, mask ratio 0.75;
- serialized checkpoint size: 1,318,230,232 bytes;
- checkpoint LFS SHA-256:
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`;
- license metadata: `cc-by-nc-nd-4.0`;
- the README contains only license metadata and no inference recipe.

The config identifies the released object as the MAE pretraining model, not
the BTXRD-fine-tuned classifier. That makes frozen feature/reconstruction use
scientifically plausible. The sparse model card and NoDerivatives license
still require conservative use: do not redistribute modified weights, claim
an unsupported intended use, or silently substitute a fine-tuned checkpoint.

## Critical BTXRD evaluation caveat

The paper's full BTXRD anomaly-map procedure is **not deployable evidence for
our setting**. The Methods state that BTXRD images were cropped to the
anatomical region containing the tumor to suppress text, clothing and metallic
artifacts, then compared with normal bone regions matched by anatomical
location. A crop “containing the tumor” supplies localization information that
our image-label-only pipeline does not have.

Therefore:

- do not quote the paper's BTXRD anomaly localization as a full-image WSSS
  result;
- do not reproduce its tumor-containing crop;
- do not use BTXRD boxes/masks, validation subgroup or per-image oracle to
  choose a crop;
- do not use the paper's downstream BTXRD-fine-tuned classifier;
- do not treat aggregate classification AUROC as segmentation evidence.

The clean transferable component is the externally pretrained MAE checkpoint,
not the paper's localized BTXRD evaluation setup.

## Two distinct admissible arms

### A. Frozen descriptor replacement — selector-support branch

Eligibility: geometry-v3 candidate oracle passes all goals, but corrected
RAD-DINO selection remains below entry.

Use the immutable SKELEX encoder on the exact same square-padded full images
and proposal gallery. Pool proposal/context features from its 14x14 token grid
with exact geometry. Train only the existing lightweight image-label MIL
selector. Compare:

1. RAD-DINO v3 descriptors;
2. SKELEX descriptors;
3. a separately predeclared late fusion only if one-source results justify it.

This is not bundled with relational MIL, weighted-mean pooling or local tiles.
The 14x14 grid is substantially coarser than RAD-DINO's 32x32 grid, so SKELEX
may improve musculoskeletal semantics while harming small spatial resolution.
That trade-off must be measured with complete subgroup misses after prediction
freeze.

### B. Full-image reconstruction map — proposal-support branch

Eligibility: the immutable candidate oracle fails at least one operational
goal.

Reproduce only the paper's generic zero-shot mechanism:

1. full square-padded image only; no tumor/anatomy crop;
2. ten deterministic random 75%-mask patterns fixed by image hash and protocol
   seed;
3. per-pixel squared reconstruction error accumulated only where a pixel was
   masked and normalized by its mask count;
4. exact inverse geometry to 320x320;
5. freeze every dense map and hash before GT.

First perform a train/validation GT-blind map audit: finite range, coverage
count, flip behavior, image-label separation of mean error, border/text
concentration, component counts and normal-image false-activation burden.
Only a separately frozen protocol may then convert the anomaly map into
additional SAM prompts/proposals. Existing candidates must be retained and
only bit-exact duplicates removed, so union-oracle support cannot decrease.

The full-image arm is hypothesis-generating because the published BTXRD
evidence used tumor-containing crops. It is not promoted on visualization or
image-label separation; the frozen proposal union still requires the complete
oracle and miss audit.

## Small-object and artifact risks

- At 224 input with 16-pixel patches, the 14x14 reconstruction grid can erase
  subtle small lesions.
- Reconstruction error highlights any out-of-distribution signal, including
  text labels, clothing, implants, positioning and acquisition artifacts.
- SKELEX learned both normal and pathological images; if a tumor remains
  partly visible outside masked patches, the decoder may reconstruct it,
  reducing anomaly contrast.
- Ten random masks do not guarantee uniform finite-sample coverage. The wrapper
  must fail if any pixel receives zero masked observations before inverse
  geometry.
- Full-image tiling could improve resolution, but it is a later separate arm
  because combining new checkpoint, anomaly generation and tiling would
  destroy causal attribution.

## T4x2 feasibility

The checkpoint is 1.23 GiB serialized and the model uses 224 input, so inference
on a 16-GB T4 is plausible but unproven. ViT-Large plus decoder activations,
ten reconstruction passes, Transformers overhead and allocator behavior must
be benchmarked.

Preferred execution:

- shard image IDs deterministically across two independent T4 workers;
- load one complete immutable checkpoint replica per T4;
- use inference mode and mixed precision only after proving numerical map
  stability against an FP32 train-only subset;
- record peak allocated/reserved memory, map throughput and exact rerun hashes;
- enforce the existing 14-GiB peak-reserved ceiling per T4.

No checkpoint fine-tuning, DDP or gradient storage is required for map
generation.

## Decision

SKELEX is a high-value conditional direction, but not evidence that the current
goal is already reachable. Its best role depends on terminal evidence:

- oracle pass / selection fail: frozen SKELEX proposal descriptors as a
  one-variable semantic-source ablation;
- oracle fail: full-image reconstruction maps as a GT-blind proposal-support
  diagnostic;
- small-only failure: do not assume the 14x14 model helps; geometry,
  fractional pooling and higher-resolution local evidence remain higher
  priority.

The paper's tumor-containing BTXRD crop is explicitly forbidden.
