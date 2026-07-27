# Paired diagnosis of the medium/large gain and small-tumor regression

Date: 2026-07-28

Status: post-freeze validation diagnostic of already rejected models. It does
not retune either model, change the running mask-bag MIL v6 protocol, authorize
a consumer, or access BTXRD test.

## Frozen evidence

Candidate:

- multi-layer soft-region decoder per-image SHA-256
  `84b3dca01096a504cd1f9725752c314f0e63c69ddbfaee39e58c8d130e35eb2e`;
- frozen prediction-manifest SHA-256
  `3647fefb9046f1982d20510eb2a0d5a8ce8ecaa91c85e0922c4b9a66e6b2fc63`.

Baseline:

- affinity-decoder v3 per-image SHA-256
  `facc4405234d33ebb3eeda697d6d232ac35170112fc829bebde48fc955aed8ea`.

Both prediction sets were previously frozen and independently audited before
validation GT was opened. The present analysis joins only those 184 positive
validation rows to the already-frozen split metadata. It reads no train
segmentation annotation and no test record.

## What changed

The multi-layer decoder improved p90 Dice by
`+0.05204674/+0.10180003/+0.13504283` overall/medium/large, but small changed
by `-0.00195504`. Small p97 Dice was `0.02912824`, below the frozen `0.03`
gate and below the affinity baseline `0.03445384`.

This was not merely one or two outliers:

- small p90 Dice improved on 31 images, decreased on 43 and tied on 20;
- small p97 Dice improved on 21, decreased on 39 and tied on 34;
- the candidate recovered ten baseline p90 complete misses but lost overlap on
  fifteen previously non-missing images;
- small p90 complete misses therefore increased from `30/94` to `35/94`;
- argmax hit was recovered on four images and lost on five.

The new representation changes which region is selected. It is not a monotonic
refinement of the affinity map.

## Small-area quartiles

The 94 small tumors were sorted by their already-known post-freeze GT area and
divided deterministically into consecutive quartiles. Candidate-minus-baseline
means are:

| Small quartile | n | GT area range | Delta p90 Dice | Delta p97 Dice | Delta pixel AP | Delta pixel AUROC | p90 misses candidate/base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 23 | 0.000068--0.000498 | -0.000607 | -0.001793 | -0.034558 | -0.016216 | 10/10 |
| Q2 | 24 | 0.000527--0.000938 | -0.003753 | -0.002556 | +0.006028 | -0.046348 | 14/9 |
| Q3 | 23 | 0.000947--0.002021 | -0.002139 | -0.006628 | -0.020910 | -0.016614 | 3/4 |
| Q4 | 24 | 0.002256--0.009473 | -0.001273 | -0.010233 | -0.013752 | +0.035246 | 8/7 |

All four quartiles regress in mean p90 and p97 Dice. Q2 causes the largest
complete-miss increase, while Q4 improves AUROC but loses p97 overlap. Thus
better global foreground/background ranking can coexist with worse shape and
support calibration.

The median small lesion covers about `96.5` pixels at `320x320`, but only
`3.86` cells on the decoder's native `64x64` output. The minimum is below one
native cell. This explains why added multi-layer semantics help larger lesions
but do not reliably preserve a small spatial maximum.

## Fixed-percentile support inflation

The frozen evaluator uses `map >= percentile(map)`. The float16/interpolated
maps contain tied plateaus, so the selected support can exceed the nominal
percentile area:

| Small diagnostic | Nominal support | Mean actual support | Median | Maximum | Images above 1.5x nominal |
|---|---:|---:|---:|---:|---:|
| p90 | 0.10 | 0.10291 | 0.10016 | 0.16271 | 1 |
| p97 | 0.03 | 0.03697 | 0.03019 | 0.10606 | 14 |
| p99 | 0.01 | 0.01915 | 0.01008 | 0.10606 | 17 |

This does not invalidate the frozen comparison: the same predeclared operator
was used consistently. It does show that `small p97 = 0.02913` is a
non-deployable diagnostic tied to a sometimes variable support size. Future
continuous-map reports should include actual support fraction and tie count,
but the rejected decoder cannot be rescued by changing `>=` to `>` or choosing
another percentile after GT.

## Anatomy is a diagnostic, not a router

Among exact anatomy labels with at least four small validation images:

- tibia (`n=31`) has delta p97 `-0.01199` and pixel AP `-0.06089`;
- femur (`n=16`) has delta p90 `-0.00307` and p90 misses increase `4 -> 6`;
- foot (`n=6`) has delta p90 `-0.00320` and misses `2 -> 3`;
- fibula (`n=12`) improves p90/p97 by `+0.00297/+0.01117`;
- radius (`n=5`) improves p90/p97 by `+0.00358/+0.00961`.

These counts are too small and post-GT for an anatomy-specific decision rule.
They indicate that a single global dense decoder is not scale/anatomy
invariant; anatomy must not be used to route validation predictions.

## Literature-grounded interpretation

1. Cheolhyun Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
   Segmentation*, WACV 2024:
   https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

   Their experiments show that aggregate WSSS metrics hide small-object
   failure and motivate size-balanced learning. For BTXRD, true size is never
   available to training; only frozen train proposal area may define balance.

2. Kangning Liu et al., *Weakly-supervised High-resolution Segmentation of
   Mammography Images for Breast Cancer Diagnosis*, MIDL/PMLR 2021:
   https://proceedings.mlr.press/v143/liu21b.html

   GLAM addresses lesions small relative to high-resolution medical images by
   combining coarse global context with local high-resolution analysis. The
   prior BTXRD learned local-decoder attempt failed, so the transferable part
   is original-resolution proposal generation, not free local saliency fusion.

3. Ekaterina Redekop et al., *Attention-Guided Prostate Lesion Localization
   and Grade Group Classification with Multiple Instance Learning*, MIDL/PMLR
   2022:
   https://proceedings.mlr.press/v172/redekop22a.html

   Their image-label-only medical MIL combines whole-image and patch features.
   It supports retaining global context while scoring local instances, but its
   MRI sensitivity/FP metrics cannot be converted to BTXRD Dice.

4. Constantin Seibold et al., *Self-Guided Multiple Instance Learning for
   Weakly Supervised Thoracic Disease Classification and Localization in Chest
   Radiographs*, ACCV 2020:
   https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html

   This radiograph MIL work treats uncertain instances differently rather than
   making every positive-bag patch positive. It supports ignoring ambiguous
   candidates and cross-fitting pseudo-instance assignments.

## Mechanism that addresses the measured failure

The solution must be adaptive in shape and scale rather than another global
percentile:

1. Preserve the coarse RAD-DINO/global map only for context and deterministic
   proposal windows.
2. Constrain output geometry to class-agnostic SAM candidates, whose areas
   vary by image; evaluate candidate oracle before learning a selector.
3. Normalize duplicate prompt variants within component/source families.
4. Score candidates relationally and out of fold. Do not make the current
   same-model argmax the sole positive instance target.
5. Represent candidate area through train-only proposal strata. Pool within
   each stratum before combining strata, so large masks and high candidate
   counts cannot dominate merely by multiplicity. A positive image still does
   not reveal which stratum is correct; normal bags remain negative in all
   strata.
6. Add exact full-view/zoom-view candidate consistency only after the base
   relational selector. A high-resolution crop may strengthen a small
   candidate, but cannot inherit the full-image positive label blindly.
7. If the frozen candidate oracle lacks small headroom, append
   original-resolution crop SAM proposals while retaining every old candidate;
   do not train another free local pixel decoder.
8. Only after a deployable proposal source passes may a partial-consensus
   consumer learn confident pixels; ambiguous positive-image background stays
   ignored.

This sequence is already compatible with the conditional version-6 decision
tree: pass selects a cross-fitted consumer source, oracle-pass/selector-fail
selects family-balanced relational MIL, and oracle-fail selects
high-resolution proposal expansion.

## Decision

The apparent medium/large success and small regression are explained by three
joint effects: native-grid undersampling, fixed-area percentile mismatch/ties,
and instance-rank replacement. The appropriate repair is an adaptive
proposal-level, scale-balanced, cross-fitted selector with optional
high-resolution proposal expansion—not a post-hoc small threshold, anatomy
router, or another hard pseudo-mask U-Net.

