# Rich-gallery top-10 relational diagnostic failure dossier

## Question and immutable rule

The previous global-consensus diagnostic failed because masks representing
common bone anatomy agree strongly across proposal sources.  This follow-up
asked a narrower question: does cross-source overlap become tumor-specific if
it is allowed to act only inside the top-10 candidates of the current best G1
+ upstream rank fusion?

The fixed annotation-free rule was:

1. compute the immutable equal G1/upstream percentile-rank baseline;
2. retain its top 10 candidates;
3. for candidate `c`, compute the mean across other sources of
   `max_d IoU(c,d) * baseline_rank(d)`;
4. select by the geometric mean of baseline rank and that support;
5. fall back exactly to baseline when the top-10 contains only one source or
   no positive cross-source overlap.

There was no fitted parameter, GT area, lesion subgroup, threshold, source
weight, morphology or test input.  The 742 choices for 371 images were frozen
before validation polygons.

## Actual binary-mask result

| Variant | Overall Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1 + upstream baseline | **0.288729** | **0.216839** | **0.157723** | 0.435229 | 0.386874 | 49 |
| top-10 relational product | 0.285647 | 0.214737 | 0.122765 | **0.463765** | **0.423777** | **45** |
| Paired delta | **-0.003083** | -0.002102 | **-0.034958** | +0.028536 | +0.036903 | -4 |

Overall complete-group bootstrap CI95 is
`[-0.024322, 0.016844]`.  The overall change is inconclusive around zero, but
the small-lesion loss is not: small CI95 is
`[-0.063430, -0.011129]`.  The predeclared subgroup noninferiority condition
therefore fails and the rule cannot replace the 0.288729 baseline.

## Paired mass and miss transitions

The relational rule changes the selected candidate on 123/184 images (66.85%).
Among all tumors it produces 53 wins, 51 losses and 80 ties.  The positive Dice
mass is `+5.66698`; the negative mass is `-6.23419`.  It rescues 12 complete
misses but turns 8 prior hits into misses:

| Transition | Images |
|---|---:|
| hit -> hit | 127 |
| hit -> miss | 8 |
| miss -> hit | 12 |
| miss -> miss | 37 |

This distinction matters.  Cross-source relation improves hit recall, but the
quality lost on already-correct images is larger than the recovered mass.  A
miss-count-only gate would have promoted a worse Dice selector.

The small group contains only 17 wins versus 31 losses and 46 ties.  Its
positive mass is `+0.67277`, while negative mass is `-3.95881`.  It recovers
four small misses but creates seven new ones.  In contrast, medium recovers
seven misses while creating one, and large recovers its only baseline miss.

## Extent explains the subgroup divergence

| Cohort | Baseline median selected/GT area | Relational median | Direction |
|---|---:|---:|---|
| all tumors | 2.045 | 2.769 | larger |
| `<1%` | **14.603** | **27.094** | much larger/worse |
| `1-<5%` | 1.098 | 1.313 | modestly larger |
| `>=5%` | 0.382 | 0.666 | larger/better |

The same operation moves medium/large masks toward the desired extent but
moves small masks away from it.  This is the same bidirectional extent problem
found in the baseline dossier, now causally reproduced by a fixed intervention.
Cross-source overlap rewards anatomical structures with stable, larger extent.
For large tumors this corrects under-segmentation; for small tumors it doubles
an already severe over-segmentation ratio.

## Why support cannot safely gate the intervention

Relational support has Spearman correlation `0.43660` with the Dice of the
candidate it selects.  It therefore identifies generally plausible anatomical
masks.  However, its Spearman correlation with the paired Dice improvement
over baseline is `-0.00016`.  By subgroup, support-versus-delta is approximately
`0.0012/-0.0089/-0.5054` for small/medium/large.

Thus high support does not tell us whether switching away from baseline helps.
There is no evidence-backed label-free support threshold that can retain the
medium/large gains while avoiding small losses.  Sweeping a support threshold
on these validation polygons would only overfit the same cohort.

## Source-transition analysis

The largest favorable transition is `classifier448 -> LayerCAM320`: 22 images,
`+1.29387` total Dice.  Almost all of this comes from medium lesions: 10 images,
`+1.24657`; the 12 small transitions contribute only `+0.04730`.

Several transitions are consistently harmful:

- `external -> LayerCAM`: 8 images, `-0.63126` total;
- `LayerCAM -> external`: 12 images, `-0.70678` total;
- `LayerCAM -> classifier`: 17 images, `-0.45778` total.

Even same-source changes can fail.  On small lesions,
`classifier -> classifier` contributes `-1.02717` total Dice over 26 images.
Therefore the error is not just a source-calibration mistake; relation also
ranks the wrong extent within a source, consistent with the 70.29% within-source
regret measured previously.

It would be tempting to keep only the favorable classifier-to-LayerCAM
transition.  That router is not authorized: its rule was discovered from
validation polygon outcomes, its benefit is concentrated in a true-size
subgroup unavailable at inference, and G2 has already shown that source
identity is a dangerous label shortcut.  It is recorded as a training
hypothesis, not a deployable selector.

## What has been learned, rather than merely failed

1. **Relation contains real signal.** Medium and large Dice improve and 12
   misses are recovered.  Proposal interaction should not be discarded as a
   feature family.
2. **Geometry is not tumor identity.** Relation alone cannot decide when to
   override baseline; its support-versus-delta correlation is zero.
3. **Small lesions require semantic specificity.** The dominant small failure
   is selecting a large stable anatomical mask, even within classifier448.
4. **A global post-hoc router is exhausted.** Global consensus and bounded
   top-10 consensus fail for the same causal reason at different scales.
5. **The baseline must be preserved.** Any next learned head must start as the
   exact 0.288729 scorer and demonstrate a residual gain rather than replacing
   its complementary G1/upstream evidence.

## Decision: retire post-hoc geometric reranking

Do not sweep top-K, IoU threshold, support weight, product/equal fusion, source
weight, mask resolution or morphology.  These would tune a non-tumor-specific
signal on the same validation polygons and cannot solve the observed
size-direction conflict.

The next justified mechanism is a **baseline-preserving learned relational
residual**, not the existing critical-instance primitive unchanged.  The old
primitive anchors every relation to the current top-1; the baseline misses 49
images and the miss oracle has median rank 95, so that design would reinforce
the same confirmation error.

The replacement design must:

- retain G1 and upstream ranks explicitly and zero-initialize the residual;
- use multiple candidate hypotheses rather than one detached winner;
- use all train-normal candidates as reliable negatives;
- generate any positive proposal targets out of group-preserving train-only
  folds, never from the same image/model fit;
- remove external-source presence and candidate count from the training loss,
  while retaining external masks as inference candidates;
- use pairwise relation as context, not as a direct score;
- measure within-source ranking and deep-rank recovery separately;
- freeze validation choices before polygon evaluation and compare actual Dice
  directly with 0.288729 and all three subgroups.

A matched control is mandatory: the same residual capacity using independent
candidate descriptors without relational context.  Only a full-minus-control
gain can be attributed to relation.  Image BCE/AUROC are diagnostics, not the
promotion result.

## Provenance

- Protocol SHA-256:
  `6cb03b673b2aaf2b333253685b634ff1860d60260b5acc05a5afd23079683541`
- Stage-A prediction freeze SHA-256:
  `c208be23d74b80c749a5d6f711befeafc9a8f04c4804d612f7bfbf82f0d38597`
- Stage-A selection manifest SHA-256:
  `07ac2517b9b294f0467b4d84ea8b0cc1bd207067c44e1ecca955b6c0658e25b2`
- Stage-B summary SHA-256:
  `2965361276b8e04007034c47872eabfac7bf4692c2a8eaea79135b03097749aa`
- Stage-B per-image SHA-256:
  `e5e97f364b2edfd33fb0f2d3e4734a9e50e4a2e241b1133c4cd7531d88552e70`
- Failure-decomposition summary SHA-256:
  `e48ee11e9854ea0aaf87d77c2ebbf88726b3e81c7c15b8f93848bb441bc067a5`
- Failure-decomposition per-image SHA-256:
  `944e2f63d37ba9d819ddeca24b9a83a226ce5d7c4c8b43ae5beae8bcfbc22540`
- Failure-decomposition audit SHA-256:
  `6c81cceb65b98faae15e6bb0f9f7a62e8cae6254441f2984c44a0e1985de79a6`
- Validation cohort: 371 images, 184 tumors, subgroup 94/72/18.
- Test evaluated: false; test images read: 0.
