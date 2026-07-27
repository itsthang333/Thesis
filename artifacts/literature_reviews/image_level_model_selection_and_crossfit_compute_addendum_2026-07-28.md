# Image-level-only model selection and cross-fit compute addendum

Date: 2026-07-28

Status: conditional feasibility analysis only. It neither changes the running
mask-bag MIL v6 protocol nor authorizes a new Kaggle job, consumer, validation
mask read, or BTXRD test access.

## Question

If a prediction-first source passes, how can BTXRD create out-of-fold train
teachers and choose consumer checkpoints without quietly using spatial
validation labels, while remaining practical on Kaggle T4x2?

## Model-selection boundary

Junsuk Choe et al., *Evaluating Weakly Supervised Object Localization Methods
Right*, CVPR 2020,
https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html,
show that localization hyperparameters and model selection are often tuned
with forbidden full localization supervision. Their task and metric differ
from BTXRD, but the protocol lesson transfers directly: image-label-only claims
must not hide spatial supervision in checkpoint or threshold selection.

An image-level proxy also cannot be treated as a substitute for localization
quality. In this project, classifiers and local MIL decoders have previously
reduced image BCE or achieved high confidence while localizing the wrong
region. SmoothMax AUROC can detect empty/all-positive collapse, but maximizing
it may favor the smallest discriminative tumor fragment and hurt Dice.

Therefore the first consumer protocol should use:

- literature/source-audit-fixed architecture and loss coefficients;
- a fixed training horizon;
- the final EMA checkpoint, not the epoch with best validation Dice;
- image-level AUROC, positive/normal separation, output area, view
  equivariance and teacher agreement only as fail-closed collapse diagnostics;
- one spatial validation evaluation after all 371 maps and hashes are frozen.

If a later experiment wants checkpoint selection by an image-label proxy, its
proxy, tie rule and direction must be predeclared and independently justified.
It remains an image-level selection claim, not proof that the chosen epoch has
the best spatial Dice.

## Cross-fit source contract

Ramazan Gokberk Cinbis, Jakob Verbeek and Cordelia Schmid, *Multi-fold MIL
Training for Weakly Supervised Object Localization*, CVPR 2014,
https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html,
prevents premature localization lock-in by re-localizing each fold with a
detector trained on other folds. Jihye Kim et al., *CrossSplit: Mitigating
Label Noise Memorization through Data Splitting*, ICML 2023,
https://proceedings.mlr.press/v202/kim23a.html, independently supports the
principle that a peer trained on a disjoint partition cannot simply return the
same memorized noisy example-label pair.

For BTXRD, the minimum admissible implementation is:

1. Assign complete `group_id` units to five deterministic folds, stratifying
   only by the allowed binary image label. No lesion size, polygon, anatomy
   mask, validation result or test record enters folding.
2. Extract frozen original/flip RAD-DINO proposal descriptors once for all
   train and validation candidates and hash-freeze the cache.
3. Train five selector heads. Fold `k` receives image-label loss from the other
   four folds and scores only fold `k` for the train teacher.
4. Train a sixth selector on the complete clean-train cohort for validation
   inference. It uses the same fixed epoch horizon and does not inherit a
   spatially selected fold checkpoint.
5. Save every candidate's original/flip logits, kept index and
   component/prompt/source provenance. Freeze all 2,981 out-of-fold train
   records before consumer training.
6. Freeze the sixth model's 371 validation predictions before the sole
   validation GT evaluation. Test stays locked.

The fold-specific heads may share immutable descriptor bytes but no optimizer
state. Candidate ordering and gallery hashes must remain identical across
folds.

## Exact storage feasibility

The audited galleries contain approximately `174,657` train and `20,965`
validation candidates, each represented by a 1,156-dimensional descriptor.
At float16:

- train one view: `385.10 MiB`;
- train original plus flip: `770.20 MiB`;
- validation original plus flip: `92.45 MiB`;
- complete two-view descriptor cache: `862.65 MiB`.

Six models' original/flip float16 logits for every train and validation
candidate require only about `4.48 MiB`, before CSV/NPZ metadata. Masks already
belong to the immutable proposal galleries and should not be copied into every
fold artifact.

The current selector has about `331,529` trainable parameters:

- LayerNorm over 1,156 dimensions;
- `1156 -> 256 -> 128 -> 1` MLP.

Its weights occupy about `1.27 MiB` in float32; optimizer state and a padded
`16 x 81 x 1156` descriptor batch remain small relative to either T4. Thus
RAD-DINO descriptor extraction and disk serialization dominate; five-fold head
training is not five repetitions of the encoder.

## Fastest admissible T4x2 execution

Descriptor extraction:

- load the same hash-verified frozen RAD-DINO snapshot on each T4;
- deterministically shard whole images, including all their candidates,
  between devices;
- write two disjoint memory-mappable cache shards with per-record hashes;
- merge only after exact `image_id`, candidate count, kept-index and
  provenance coverage checks.

Selector training:

- schedule three of the six independent heads per T4;
- do not use DistributedDataParallel, because heads optimize different
  folds/models and must not synchronize gradients;
- reuse the read-only descriptor cache;
- record the actual device and nonzero sample count for every head.

Consumer training, if later authorized, is different: one shared model should
use two-process DistributedDataParallel with synchronized gradients. Calling
independent fold jobs "data parallel" would be technically false.

## Predeclared no-GT diagnostics

For each out-of-fold selector and the full-data selector, record without using
segmentation masks:

- held-out image-label AUROC/AUPRC and confusion at a predeclared threshold;
- original/flip candidate rank correlation and winner agreement;
- positive/normal bag-probability distributions;
- selected-support area distribution and empty/fallback counts;
- fold-to-fold dispersion of all metrics;
- candidate-count correlation, because the current normalized LogSumExp bag
  still has measured count shortcut risk.

These diagnostics may reject collapse or distributional shortcut. They cannot
promote a spatial mechanism, alter the operational Dice goals, or select a
post-GT threshold.

## Decision

Cross-fitting is computationally practical on T4x2 if descriptors are frozen
once and reused. It should be required before a proposal selector supplies
train pseudo labels to a dense consumer. The first consumer should retain a
fixed horizon and final EMA checkpoint; image-level diagnostics protect the
contract but do not masquerade as spatial model selection.

No implementation or Kaggle launch occurs until terminal v6 evidence selects
this branch.

