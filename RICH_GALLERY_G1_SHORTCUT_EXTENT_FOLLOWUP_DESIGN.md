# Rich-gallery G1 shortcut/extent follow-up

Status: post-Stage-B exploratory diagnostic. This file is frozen before the
new selector choices below are evaluated against validation polygons. Results
are diagnostic only because Stage-B evidence motivated the choices; they are
not an independently predeclared promotion claim.

## Evidence motivating this diagnostic

- The rich gallery raises oracle Dice to about `0.528` while G1 reaches only
  `0.206`.
- Source-choice regret is `0.08236`, but within-selected-source regret is
  `0.23951`; extent/ranking inside a source is therefore the larger failure.
- The external source is present for `184/184` tumor bags and `0/187` normal
  bags. Candidate count alone has image-label AUROC `0.89449`.
- SmoothMIL has median effective candidate count `1.63`; the selected candidate
  receives median weight `0.756`, while the oracle receives median weight
  `8.14e-15`. The detached top-1 instance target is low-quality on `60.9%` of
  tumors (`83.0%` for lesions below one percent area).
- Removing descriptor metadata improves actual Dice from `0.20603` to
  `0.22569`; the frozen upstream purity/coverage score reaches `0.22587`.

## Frozen GT-blind choices

All choices are computed from Stage-A frozen arrays before importing the
validation segmentation dataset:

1. `g1_shared_only`: original G1 score with external-source candidates made
   ineligible. This isolates the external-source shortcut; it is not a proposed
   final system because it can discard useful external oracle masks.
2. `rank_fusion_g1_upstream`: equal-weight mean of within-image percentile
   ranks from G1 and the upstream coverage/purity score. This has no fitted
   coefficient and tests complementary localization versus extent evidence.
3. `rank_fusion_nometa_upstream`: the same rank fusion after replacing the four
   metadata columns by their within-bag mean through the exact frozen G1 MLP.
4. `rank_fusion_nometa_upstream_shared_only`: item 3 restricted to the two
   sources represented in both normal and tumor bags.
5. `purity_completeness_harmonic`: harmonic mean of anchor prompt purity and
   prompt-mass completeness. It instantiates TS2C's purity/completeness idea
   without a validation-fitted weight or threshold.

Ties are resolved by higher original G1 score, then lower frozen candidate
index. Candidate scores and choices are frozen before any polygon is opened.
No per-image GT area, oracle routing, threshold sweep, morphology, SAM rerun,
consumer training, or test data is allowed.

## Decision use

- If `g1_shared_only` improves materially, generate external proposals on
  normal images and use source-balanced candidate sampling in the next train
  arm rather than permanently deleting the source.
- If rank fusion improves but remains below the accepted `0.24548`, use its
  purity/completeness signal as a fixed residual in a matched retraining arm.
- Irrespective of this exploratory result, the next training objective must
  remove singleton hard-positive confirmation: source/family-balanced pooling,
  source dropout, and delayed self-paced overlap-cluster targets are the bounded
  mechanism changes supported by the failure evidence.

