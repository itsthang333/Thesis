# Rich-gallery G1 fixed-rank-fusion bottleneck dossier

## Scope and decision boundary

This dossier analyzes the current best observed prompt-free, binary,
image-label-only BTXRD WSSS validation pipeline:

`0.5 * percentile_rank(G1 selector logit) + 0.5 * percentile_rank(frozen upstream score)`.

The rule selects one already-generated binary proposal per image.  Its choices
were frozen before validation polygons were opened, it uses no spatial ground
truth for selection, and it does not read test.  However, the equal-rank rule
was proposed after inspecting earlier validation evidence.  Its Dice is
therefore an **exploratory validation best**, not an independent confirmatory
estimate.  This distinction must be preserved in every paper or handoff.

The primary goal of this analysis is not to explain loss/AUROC proxies.  It is
to identify which part of the actual binary-mask Dice gap can still be changed
by the existing proposal gallery and which mechanism must change next.

## Current best actual result

| Cohort | n | Dice | IoU | Complete misses | Gallery oracle Dice | Selector regret |
|---|---:|---:|---:|---:|---:|---:|
| All tumors | 184 | **0.288729** | **0.216839** | 49 | 0.528298 | 0.239569 |
| `<1%` | 94 | **0.157723** | 0.118576 | 35 | 0.331876 | 0.174153 |
| `1-<5%` | 72 | **0.435229** | 0.323558 | 13 | 0.730251 | 0.295022 |
| `>=5%` | 18 | **0.386874** | 0.303117 | 1 | 0.746247 | 0.359374 |

The same gallery already contains substantially better proposals.  The gap is
therefore dominated by selection, not by a hard proposal-supply ceiling.

The fully supervised comparison remains approximately `0.492765` overall.
The weak pipeline is 0.204 Dice below that reference, but its gallery oracle is
already 0.0355 above the fully supervised mean.  A new proposal generator is
not the first intervention justified by this evidence.

## Exact regret decomposition

For image `i`, let `s_i` be the source chosen by the frozen fusion, `c_i` its
chosen proposal, and `d(i,c)` the proposal Dice.  The total selector regret is

`max_{all gallery c} d(i,c) - d(i,c_i)`.

It can be decomposed into:

1. **cross-source regret**:
   `max_all d - max_{c in s_i} d`, and
2. **within-selected-source regret**:
   `max_{c in s_i} d - d(i,c_i)`.

The observed mean decomposition is:

| Component | Mean Dice regret | Share of total selector regret |
|---|---:|---:|
| Cross-source choice | 0.070796 | 29.55% |
| Within-selected-source ranking | **0.168376** | **70.29%** |
| Candidate truncation | 0.000396 | 0.17% |

The selected source equals the global-oracle source on only 43.48% of tumor
images, so source routing is imperfect.  Nevertheless, replacing the source
router alone cannot close most of the gap: approximately seven tenths of the
recoverable error remains after conditioning on the source already selected.

The candidate-truncation term is negligible: `0.0007757` in the small group
and zero in medium/large.  Regenerating or simply retaining more of the same
candidates is not supported as the next primary experiment.

## Rank depth and two distinct failure branches

The gallery-oracle proposal has median fusion rank 32, mean rank 51 and 90th
percentile rank 138.  Exact oracle recall under the frozen fusion ordering is:

| Search depth | Oracle is present within depth |
|---|---:|
| 1 | 4.89% |
| 3 | 13.04% |
| 5 | 19.02% |
| 10 | 28.26% |
| 20 | 42.39% |
| 50 | 61.96% |

If an ideal reranker could choose only among the current top candidates, the
actual upper bounds would be:

| Candidate set available to reranker | Mean best Dice |
|---|---:|
| top 1 | 0.288729 |
| top 3 | 0.341837 |
| top 5 | 0.364894 |
| top 10 | 0.399326 |
| top 20 | 0.426336 |
| top 50 | 0.475500 |

This separates the next problem into two branches that must not be conflated:

- **top-rank refinement:** a bounded top-10 reranker has about `+0.1106` mean
  Dice headroom without changing proposal supply;
- **deep-rank miss recovery:** the 49 complete misses have median oracle rank
  95, so a top-10-only method cannot rescue most of them.

All 49 misses are gallery-recoverable.  Their median oracle Dice is 0.242165;
31 have oracle Dice at least 0.1, 20 at least 0.3, and 12 at least 0.5.  The
misses are 35 small, 13 medium and 1 large.  This is strong evidence for a
ranking failure rather than missing masks, but it also shows why a single
local top-k adjustment cannot solve the whole cohort.

## Extent failure is size-dependent and bidirectional

The selected/GT area-ratio medians are:

| Cohort | Median selected/GT area ratio | Interpretation |
|---|---:|---|
| All tumors | 2.045 | aggregate over-segmentation |
| `<1%` | **14.603** | severe over-segmentation |
| `1-<5%` | 1.098 | approximately calibrated extent |
| `>=5%` | **0.382** | under-segmentation |

The corresponding area-error bins are:

| Selected/GT ratio | n | Mean Dice | Misses | Dominant failure |
|---|---:|---:|---:|---|
| `<0.5` | 31 | 0.147686 | 10 | mostly medium/large under-segmentation |
| `0.5-2` | 60 | **0.620033** | 6 | well-calibrated extent |
| `2-10` | 39 | 0.261961 | 12 | moderate over-segmentation |
| `>=10` | 54 | **0.020917** | 21 | almost entirely small-lesion over-segmentation |

Thus one global threshold, area prior or morphology rule is structurally
wrong: it would need to shrink masks for small lesions and expand masks for
large lesions.  Lesion size is unavailable at inference under image-label-only
WSSS, so the correction must be inferred from tumor-specific local evidence,
not routed by validation GT subgroup.

## Source complementarity and calibration

### Frozen selected choices

| Selected source | n | Mean selected Dice | Misses | Median selected/GT ratio |
|---|---:|---:|---:|---:|
| classifier448 | 88 | 0.265953 | 27 | 1.949 |
| LayerCAM320 | 72 | 0.252272 | 21 | 2.935 |
| external saliency | 24 | **0.481614** | 1 | 1.006 |

External saliency is selected infrequently but is strong when selected.  G2
showed that its source presence is a label shortcut during training; this does
not imply that its individual masks should be removed at inference.

### Per-source oracle Dice

| Source | Overall | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|
| classifier448 | 0.429787 | **0.255548** | 0.615126 | 0.598343 |
| LayerCAM320 | 0.409076 | 0.222750 | 0.594148 | 0.641828 |
| external saliency | 0.387303 | 0.134276 | **0.637033** | **0.709747** |
| source union | **0.528298** | **0.331876** | **0.730251** | **0.746247** |

No source dominates every subgroup.  Classifier448 supplies the best small
lesion ceiling; external saliency is strongest for medium/large lesions; the
union is substantially better than each source.  Removing a source or forcing
uniform source usage would discard useful complementarity.

## Which signals work and which have been ruled out

Mean within-image Spearman correlation between candidate score and actual
candidate Dice is:

| Signal | Overall | `<1%` | Interpretation |
|---|---:|---:|---|
| G1 model score | **0.563249** | **0.459710** | strongest individual rank signal |
| frozen upstream score | 0.332841 | 0.263354 | weaker but complementary near the top |
| fixed rank fusion | 0.529201 | 0.441554 | best top-1 Dice, not best global ordering |
| SAM predicted IoU | 0.015965 | -0.061468 | no tumor-identity signal |
| mask area | 0.330869 | subgroup-dependent | confounded by lesion size/anatomy |
| cached causal score | 0.000000 | 0.000000 | identically zero in this gallery |

The fusion improves G1 raw on 79 images, hurts on 63 and ties on 42, producing
mean `+0.082703` Dice.  It also beats upstream alone by `+0.062861`.  This
confirms that G1 and upstream carry complementary information, but their equal
rank fusion concentrates that gain near selected candidates rather than
improving the ordering of the whole bag.

Prompt mode is diagnostic, not a fixed prior.  The selector chooses
`box/box_point/point = 68/71/45`, while the eligible oracle uses
`73/44/67`.  Box-point is overselected and point underselected, but prompt mode
alone cannot identify which individual proposal is tumor.

The following mechanisms have now been directly ruled out as primary fixes:

- SAM confidence alone: Dice 0.098983, small Dice 0.020622, 125 misses;
- source-presence or source-count learning: G2 removes the shortcut but loses
  the complementarity that made G1 fusion useful;
- global geometric agreement: the matched consensus diagnostic below selects
  repeated anatomy rather than tumor;
- another global area/threshold rule: extent error changes direction by lesion
  size;
- simple candidate deletion/insertion scalar: the earlier causal selector did
  not improve its matched baseline;
- more resolution, CAM/PCAM/SAM-IoU selection, affinity expansion,
  teacher-pseudo-student, reconstruction/diffusion residual, healthy-density,
  NRCE, SynRad and OLV: prior controlled runs failed their localization gates.

## Cross-source consensus experiment and failure mechanism

The fixed diagnostic scored candidates by cross-source mask agreement, then
tested consensus alone and two equal/factorized combinations with the frozen
G1/upstream ranks.  All 371 choices for all four variants were frozen before
validation polygons.

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1 + upstream baseline | **0.288729** | **0.216839** | **0.157723** | 0.435229 | 0.386874 | 49 |
| consensus only | 0.160453 | 0.117087 | 0.041551 | 0.252279 | 0.414075 | 78 |
| equal G1/upstream/consensus | 0.267073 | 0.202981 | 0.106249 | **0.438749** | **0.420227** | 44 |
| product fusion | 0.253993 | 0.192212 | 0.104109 | 0.412932 | 0.400964 | 41 |

Consensus reduces complete misses (`49 -> 44/41`) and slightly improves
medium/large extent, so overlap carries some anatomical stability signal.  It
nevertheless reduces overall Dice by 0.021657-0.034736 because it strongly
prefers repeated large bone/anatomy masks.  Small-lesion median selected/GT
area ratio rises from 14.60 to 38.99 and 53.10.  The consensus-only selected
masks have median agreement near 0.966 but Dice only 0.160453: high agreement
is evidence of common anatomy, not tumor identity.

Decision: retire global cross-source consensus.  Do not sweep its weight,
neighbor count, IoU threshold or mask scale.  Spatial relations remain
potentially useful only when conditioned on tumor-specific evidence and
evaluated inside a bounded candidate neighborhood.

## Why previous G2 failure still helps this baseline

G2 is not a replacement baseline.  Its best Dice is 0.254326, below 0.288729.
Its role was causal diagnosis:

- removing the external-source training shortcut improved raw G1
  `0.206026 -> 0.248502`;
- but negative-only MIL could not identify a positive tumor proposal;
- low-temperature pooling collapsed to about 1.7-1.9 effective candidates;
- G2 model/upstream rank correlation rose, causing the fusion gain to collapse
  from `+0.082703` to at most `+0.030551`.

The transferable lesson is precise: retain the frozen G1/upstream baseline and
source union, but replace neither with another bag classifier that learns the
same coverage/purity score.  Any learned refinement must add candidate-level
positive evidence that remains complementary to both ranks.

## Mapping the evidence to primary literature

- **OICR** iteratively converts an inferred top proposal into labels for nearby
  overlapping proposals.  It provides a mechanism for within-source
  refinement, but our 49 misses show that blindly anchoring every image to the
  current top-1 would reinforce wrong locations.
- **PCL** and **C-MIL** reduce proposal ambiguity through proposal clusters,
  spatial/class-related subsets and continuation.  They support relational
  candidate learning, but the consensus failure shows that BTXRD clusters must
  be conditioned on tumor evidence and normal-image negatives rather than
  raw overlap frequency.
- The chest-X-ray CAM-to-affinity pipeline shows that affinity propagation can
  improve extent only after a usable disease seed exists.  Our global
  consensus experiment is the corresponding negative control: affinity
  without tumor identity expands stable anatomy.
- Correct WSOL evaluation practice requires separating localization-based
  development from confirmatory evaluation.  Therefore 0.288729 remains an
  exploratory validation result until a fully frozen pipeline is evaluated on
  an untouched cohort.

Primary references:

- OICR: https://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_Instance_Detection_CVPR_2017_paper.html
- C-MIL: https://openaccess.thecvf.com/content_CVPR_2019/html/Wan_C-MIL_Continuation_Multiple_Instance_Learning_for_Weakly_Supervised_Object_Detection_CVPR_2019_paper.html
- PCL: https://arxiv.org/abs/1807.03342
- WSOL evaluation: https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html
- Chest X-ray WSSS: https://arxiv.org/abs/2007.00748

## Next mechanism: exact hypothesis and falsification plan

The next experiment must target **within-source positive-instance ranking**.
It must not be another proposal generator, source-balancing loss, global
consensus, threshold sweep, or whole-bag top-1 MIL objective.

The strongest evidence-backed candidate is a matched, group-cross-fitted,
relational proposal reranker with these constraints:

1. keep the immutable G1 and upstream ranks as two explicit baseline features;
2. learn only a residual score, so the new head must demonstrate information
   not already encoded by the baseline;
3. use train-normal proposal bags as dense negative evidence and exclude the
   external-source presence shortcut from the loss;
4. use source, prompt mode, component identity and pairwise proposal relations
   only as relational context, never as label-perfect routing variables;
5. use group-preserving out-of-fold train predictions for any inferred
   positive proposal targets, preventing same-group self-training leakage;
6. separate a bounded top-rank refinement diagnostic from a deep-rank recovery
   diagnostic, because the former can reach 0.3993 within top-10 while the
   latter must address misses whose median oracle rank is 95;
7. freeze every candidate choice before validation polygons and compare paired
   against 0.288729 overall and 0.157723/0.435229/0.386874 by subgroup.

Minimum mechanism gates before a costly full run:

- a cheap matched diagnostic must improve top-10 oracle capture or candidate
  Dice rank within classifier448 and LayerCAM separately;
- it must not increase small-lesion selected/GT area inflation;
- it must retain external masks as eligible inference candidates without
  allowing source presence into image-label training;
- improvement must remain after removing source identity and candidate count
  from the diagnostic report;
- actual frozen binary-mask Dice is the final criterion; image AUROC, loss and
  rank correlation are explanatory metrics only.

If this diagnostic fails, the failure dossier must report per-image paired
changes, source transition, oracle rank, miss recovery/loss, extent ratio and
the exact share of cross-source versus within-source regret remaining.  It is
not scientifically acceptable to respond with a seed, epoch, resolution,
threshold or fusion-weight sweep.

## Provenance

- Bottleneck summary SHA-256:
  `e0228fef2b4360d766e9188e66178be5365788411a8ef6878d058850b77641a1`
- Bottleneck per-image table SHA-256:
  `c58a6f1afd6173ac1e99ef3b66fa290a97dc01bcf36380b6f02b0ff032574b75`
- Bottleneck analysis audit SHA-256:
  `0b005822b605461dba41bbdaed22fdca71bb8cb68341d1a24ad0f15f386bbe6d`
- Consensus Stage-B summary SHA-256:
  `1a1e8020f7fb9a9290e8e253ed854f7cad2d862ba69e0a7f5a4ceef590c59d50`
- Consensus Stage-B per-image SHA-256:
  `4b4a5c3c11159812037f14b0f3d4ef0b42f95feeecbf81a9939d8a1ad4641e7d`
- Consensus Stage-A prediction-freeze SHA-256:
  `30fd0bf766743a3ae9fc5bc2a2710aaa779808e9456ea7011925ae2613216641`
- Validation cohort: 371 images, 184 tumors, subgroup 94/72/18.
- Test evaluated: false; test images read: 0.
