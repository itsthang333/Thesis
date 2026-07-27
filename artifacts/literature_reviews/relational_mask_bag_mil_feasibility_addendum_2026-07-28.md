# Relational mask-bag MIL implementation feasibility addendum

Date: 2026-07-28

Status: implementation-readiness research only. This addendum does not modify
or supersede the running mask-bag MIL version-6 protocol. It was prepared
without segmentation annotations or BTXRD test access.

## Why a second selector is technically justified only conditionally

The current v6 head has descriptor dimension
`3 * 3 * 128 + 4 = 1,156`, a maximum of 81 proposals, and a shared MLP that
scores each proposal independently. Its only candidate interaction is
normalized LogSumExp at bag level. After a two-epoch warm-up, the model's own
detached argmax becomes the sole positive instance target for each positive
bag.

The candidate NPZ schema already stores aligned `component_ids`,
`prompt_modes`, and `proposal_source_ids`. The generator typically obtains
three SAM multimask variants per prompt call and uses the modes
`box_point`, `point`, and `box` for each morphology component. The observed
27/36/.../81 candidate-count pattern is therefore structured multiplicity,
not an arbitrary set size. The v6 loader discards this provenance.

If v6 localization fails while its oracle passes, a family-aware relational
head tests a new causal hypothesis:

> selection fails because independent self-training cannot distinguish
> repeated prompt variants and locks onto its own early winner.

If the oracle fails, this hypothesis is not sufficient and the head must not be
launched.

## Proposed hierarchy

Each raw candidate remains eligible for final winner-take-all prediction. The
hierarchy changes only how bag evidence and instance targets are learned.

1. **Variant level:** candidates sharing
   `(component_id, prompt_mode, proposal_source_id)` form one SAM multimask
   family. Aggregate their evidence with a normalized operation so three
   variants do not contribute three times the evidence of a single CAM
   candidate.
2. **Prompt level:** combine prompt-mode families within
   `(component_id, proposal_source_id)`. Retain mode identity as a small
   categorical embedding and expose mode agreement/entropy as diagnostics.
3. **Component level:** combine components within the image using normalized
   attention. Component IDs are local identifiers, not semantic categories;
   their raw integer values are never embedded.
4. **Relational level:** an instance stream scores each proposal. A DSMIL-like
   bag stream compares every proposal representation with the critical
   candidate representation. This lets a proposal be judged relative to other
   explanations in the same radiograph.
5. **Output level:** prediction still selects exactly one original frozen
   proposal. No learned pixel decoder, union, threshold expansion, or
   validation-fitted post-processing is introduced.

This hierarchy is a project-specific synthesis. DSMIL supplies the
critical-instance relational principle, while the family normalization follows
from the project's frozen candidate-generation schema and measured count
shortcut.

Primary source:

- Bin Li, Yin Li, and Kevin W. Eliceiri, *Dual-stream Multiple Instance
  Learning Network for Whole Slide Image Classification with Self-supervised
  Contrastive Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html

## Candidate representation

Retain the existing frozen features:

- RAD-DINO layer 4/8/12 proposal mean;
- local-ring mean;
- proposal-minus-ring contrast;
- SAM score, log area, prompt-map mass coverage, and prompt-map mean;
- aligned horizontal-flip descriptor.

Add only prediction-derived features:

- normalized centroid `(cx, cy)`;
- normalized bounding-box width/height and aspect;
- fractional area and a compactness/extent statistic;
- prompt-mode/source categorical embedding;
- per-proposal nominal-memory mean, maximum, mass coverage, ring mean, and
  proposal-minus-ring contrast from a previously frozen, hash-bound map.

The nominal-memory map is not thresholded into a segmentation. Earlier
corrected evidence showed that it ranks tumor pixels better than rejected
dense heads but lacks usable standalone shape. Pooling it over a SAM proposal
lets a boundary prior and an anomaly prior contribute complementary evidence.

Position is treated cautiously. Krishnamoorthy and Wiens found positional
encodings helpful in radiograph MIL, but BTXRD includes heterogeneous anatomy.
Position features therefore require a no-position ablation, exact
flip-equivariance, and reporting of winner-location distributions. No anatomy,
true tumor size, or subgroup ID may enter the model.

Primary source:

- Meera Krishnamoorthy and Jenna Wiens, *Multiple Instance Learning with
  Absolute Position Information*, CHIL/PMLR 248:88-104, 2024:
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html

## Cross-fitted instance supervision

The proposed replacement for same-model argmax supervision is:

1. partition the training cohort into deterministic group-preserving,
   label-stratified folds using only group IDs and image labels;
2. fit the bag-level relational selector on all but one fold;
3. predict a complete candidate probability distribution for the held-out
   fold;
4. repeat until every training image has out-of-fold candidate probabilities;
5. freeze and hash the OOF manifest;
6. train one final full-train head with image-level BCE, reliable
   normal-candidate negatives, flip consistency, and a soft positive-instance
   target derived only from the corresponding OOF distribution.

No fold sees its own inferred instance label during the model that generates
that label. Ambiguous positive candidates remain soft/ignored rather than being
declared background.

This transfers the anti-lock-in principle of:

- Ramazan Gokberk Cinbis, Jakob Verbeek, and Cordelia Schmid,
  *Multi-fold MIL Training for Weakly Supervised Object Localization*, CVPR
  2014:
  https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html

The number of folds, soft-target temperature, loss weights, epoch count, seeds
and final-fit rule must be frozen before validation predictions. Train-only
bag AUROC cannot be the sole selection criterion because the measured count
shortcut can inflate it. Candidate-count/source association and flip
consistency are mandatory train-only diagnostics.

## Count-robustness controls

Family-normalized pooling is the first control because it removes the known
prompt multiplicity structurally without dropping proposal support.

MIL-Dropout is optional and later:

- activate only after the relational head's warm-up;
- start with the smallest non-zero top-k candidate count;
- disable at inference;
- report winner concentration and whether score/count association falls;
- reject it if small localization decreases after the frozen validation
  evaluation.

The reason for this ordering is specific to BTXRD: a positive small-tumor bag
may contain only one useful candidate. Removing that candidate too early can
force the model to learn background evidence.

Primary source:

- Wenhui Zhu et al., *How Effective Can Dropout Be in Multiple Instance
  Learning?*, ICML 2025, PMLR 267:80090-80106:
  https://proceedings.mlr.press/v267/zhu25q.html

IBMIL and probability-space MIL remain second-line. They are justified only if
family normalization plus cross-fitting still leaves measurable bag-context or
selection-drift failure.

Primary sources:

- Tiancheng Lin et al., *Interventional Bag Multi-Instance Learning on
  Whole-Slide Pathological Images*, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Interventional_Bag_Multi-Instance_Learning_on_Whole-Slide_Pathological_Images_CVPR_2023_paper.html
- Zhaolong Du et al., *Rethinking Multiple-Instance Learning From Feature
  Space to Probability Space*, ICLR 2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html

## T4x2 feasibility

Using the frozen manifest means:

- estimated train candidates: about `174,657`;
- estimated validation candidates: about `20,965`;
- one fp16 1,156-D descriptor view: about `431.33 MiB` for train plus
  validation;
- original and flip cache together: about `862.66 MiB`, before small
  provenance/nominal feature additions.

For a 256-D relational space, batch size 16 and 81 proposals:

- proposal embeddings contain `331,776` fp32 values, about `1.27 MiB`;
- a four-head `81 x 81` attention tensor contains `419,904` fp32 values,
  about `1.60 MiB`.

A `1,156 -> 256` projection plus a small DSMIL/attention head remains below
roughly one million trainable parameters. Therefore RAD-DINO descriptor
extraction, not relational fitting, dominates runtime. Cache original/flip
descriptors once with RAD-DINO data-parallelized across the two T4s; then run
the fold heads on cached descriptors. Five-fold selector fitting is
computationally plausible without repeating the frozen encoder.

All heavy execution remains on Kaggle. These estimates are architectural
arithmetic, not a runtime claim.

## Required implementation tests before launch

1. Candidate provenance arrays align exactly with retained candidate indices
   after minimum-grid-mass filtering.
2. Family normalization is invariant to duplicating an identical proposal
   within a family.
3. Adding a genuinely different component changes bag evidence.
4. Invalid/padded candidates and empty families fail closed.
5. Candidate coordinate features transform exactly under horizontal flip.
6. Original/flip candidates retain the same local-to-original index mapping.
7. Group IDs never cross OOF train/held-out folds.
8. Every training row receives exactly one OOF probability vector whose length
   matches its frozen candidate payload.
9. Normal-bag candidates receive negative targets; non-selected positive-bag
   candidates are not automatically negative.
10. Nominal map/checkpoint/geometry/source hashes are verified before feature
    construction.
11. All 371 validation maps and the checkpoint/OOF manifest are frozen before
    the evaluator can import segmentation data.
12. `consumer_trained=false` and `test_evaluated=false` remain enforced.

## Go/no-go rule

- **Go:** version-6 candidate oracle meets every operational target, while its
  selected localization fails at least one required gate.
- **No-go:** oracle support fails; build high-resolution proposals instead.
- **No-go:** version 6 passes the complete gate; do not add selector complexity.
- **No-go:** required provenance or nominal-memory inputs cannot be physically
  hash-bound.

Only after the terminal v6 audit may these implementation details be converted
into a new immutable experimental protocol.
