# CAM-conditioned Geometry-v4 selector design

Status: **STATIC DESIGN ONLY — NO CLAIM, NO REAL INPUT, NO PREDICTION, NO GT**.

## 1. Scope and reason for returning to the original pipeline

This design deliberately returns to the original BTXRD chain:

`binary DenseNet121 -> flip-TTA LayerCAM -> SAM candidate gallery -> frozen RAD-DINO Geometry-v3 -> image-label MIL selector`.

It does not introduce a new backbone, FPN, dense decoder, new proposal source,
consumer, segmentation loss, validation-size router or post-hoc morphology. The
immutable same-gallery candidate supply and every physical candidate mask remain
unchanged.

The accepted Geometry-v3 result is
`0.24548239 / 0.11708058 / 0.37713552 / 0.38941265` Dice for
overall/small/medium/large. Equal-rank fusion with the frozen CAM-derived
upstream score reaches
`0.25520289 / 0.12563547 / 0.39380626 / 0.37741925`, but increases complete
misses from 53 to 70. The same-gallery oracle remains
`0.40907553 / 0.22274949 / 0.59414708 / 0.64182537`, so the problem is still
candidate representation/ranking, not proposal availability.

## 2. Uncovered representation bottleneck

Geometry-v3 pools, at each frozen RAD-DINO layer, an unweighted candidate
interior mean, an unweighted local-ring mean and their contrast. Its only CAM
inputs are four scalar metadata fields, including total CAM-mass capture and
mean CAM inside the mask. Therefore two candidates with similar scalar
capture/purity can still have very different spatial-semantic layouts while
receiving no CAM-conditioned feature representation.

The terminal extent evidence is signed and cohort-dependent: selected/GT area
ratio medians are approximately `14.603 / 1.098 / 0.382` for
small/medium/large. A scalar global shrink/expand rule is therefore invalid.
The missing observable must distinguish:

- a large mask whose low-CAM interior contains unrelated anatomy;
- a tight mask whose exterior ring still contains tumor-class evidence; and
- a mask whose CAM core and local RAD-DINO semantics agree.

## 3. Single scientific delta

For each already frozen candidate, use the already serialized flip-TTA
`prompt_map` from its candidate payload. Project the prompt map, candidate mask,
content-validity map and radius-two exterior ring through the exact accepted
Geometry-v3 square-coordinate contract.

Let `p` be the projected LayerCAM prompt value in `[0,1]`, `m` the fractional
candidate support, `r` its accepted exterior ring, and `f_l` the frozen
128-dimensional projected RAD-DINO token at layer `l in {4,8,12}`. Append three
threshold-free feature means per layer:

1. CAM core: `mean(f_l; m * p)`;
2. low-CAM candidate interior: `mean(f_l; m * (1-p))`;
3. CAM-positive exterior ring: `mean(f_l; r * p)`.

Every weighted mean uses the accepted fractional-mass denominator contract;
zero/near-zero weight is represented deterministically and never drops or
reorders a candidate. No CAM threshold, raw coordinate, source, family,
subgroup or validation-derived area enters the scorer.

The original 1,156-D Geometry-v3 descriptor is retained byte-for-byte and the
three new 128-D means at each of three layers append 1,152 dimensions, giving a
2,308-D descriptor. The scorer keeps the accepted form
`LayerNorm -> Linear(256) -> GELU -> Dropout(0.1) -> Linear(128) -> GELU -> Linear(1)`;
only its input width changes. Frozen RAD-DINO layers/projection, normalized
SmoothMax temperature `0.2`, image BCE, self-guided instance weight `0.25`,
two-epoch warm-up, aligned-flip consistency `0.1`, AdamW, seed 42 and exactly
16 final-only epochs stay unchanged.

## 4. Why this is not a repetition

- `coverage_mass_sam`, BAS-B1/B4 and the rich-gallery G1 fusion use scalar
  activation coverage/purity ranks; they do not condition RAD-DINO feature
  pooling on the spatial CAM distribution.
- R1/R2/R3/R4/S1-S8 change prototypes, affinity, aggregation, relational
  residuals, graph/cluster structure or another representation; none append
  CAM-core/low-interior/exterior feature means to Geometry-v3.
- S2C SSC/CPM changed classifier training/CAM generation and was terminally
  rejected on BTXRD. This design freezes the promoted LayerCAM and does not
  rerun S2C.
- S9 adds a frozen SKELEX likelihood; S10 adds a trainable ResNet50-FPN. Neither
  is inherited. S10 remains paused.

## 5. Finite matched arms if the design later becomes a claim

All arms must be frozen physically before validation segmentation GT:

1. accepted Geometry-v3 identity;
2. exact Geometry-v3 plus frozen upstream equal-rank control;
3. CAM-conditioned Geometry-v4 direct scorer;
4. primary: equal tie-aware within-image percentile ranks of Geometry-v4 and
   the exact frozen upstream score.

The primary changes only the learned descriptor score inside the already known
two-rank architecture. There is no weight, threshold, layer, ring-radius,
epoch, temperature, mask-count or subgroup sweep.

Before GT, the run must prove exact control reconstruction, complete cohort and
hash closure, finite/nonconstant new descriptors, original/flip alignment and
at least 16 changed tumor-image winners versus the `0.25520289` control. Fewer
than 16 changes cannot close the current overall goal gap even under an
impossible per-changed-image Dice gain of 1, so GT evaluation would have no
operational value.

The post-freeze mechanism gate requires primary overall and small Dice to
strictly improve over both controls, medium and large not to regress, and
complete misses not to increase over the stricter 53-miss Geometry-v3
reference. The operational gate remains simultaneous Dice
`0.34024039 / 0.17895493 / 0.51244178 / 0.49370336` with paired overall CI95
lower bound above zero. A failure receives no post-hoc rescue or sweep.

## 6. Safety and execution boundary

- Training supervision: binary image-level normal/tumor labels only.
- Validation candidate predictions and physical maps freeze before GT.
- No consumer before operational pass; BTXRD test stays locked.
- Heavy cache construction/training/inference only on Kaggle T4x2 or P100.
- This file authorizes no real candidate/cache/image read and no Kaggle launch.
  A new unique `DANG LAM` claim, exact protocol/source hashes, independent
  producer-free auditor and full `KAGGLE_PREFLIGHT_CHECKLIST.md` audit are
  required first.

## 7. Literature boundary

- Kweon and Yoon, *From SAM to CAMs*, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html
  motivates joint CAM/SAM reliability, but its SSC/CPM implementation is not
  reused because the controlled BTXRD family was already rejected.
- Chen et al., *C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation
  on Medical Image*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Chen_C-CAM_Causal_CAM_for_Weakly_Supervised_Semantic_Segmentation_on_Medical_CVPR_2022_paper.html
  motivates separating tumor evidence from recurrent anatomy; no C-CAM model
  or reported performance is transferred.
- Rong et al., *Boundary-Enhanced Co-Training for Weakly Supervised Semantic
  Segmentation*, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.html
  identifies pseudo-label boundary noise. Its downstream co-training consumer
  is explicitly not authorized; only the need to preserve boundary-local
  evidence motivates the exterior-ring descriptor.
- Chen et al., *Segment Anything Model (SAM) Enhanced Pseudo Labels for Weakly
  Supervised Semantic Segmentation*, 2023:
  https://arxiv.org/abs/2305.05803
  supports the class-aware CAM plus class-agnostic SAM-mask decomposition; its
  dataset-specific aggregation is not copied.

