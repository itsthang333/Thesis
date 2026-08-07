# Metric and defense-evidence audit

The Vietnamese defense-facing execution matrix is maintained in
[`DEFENSE_EXPERIMENT_PLAN_VI.md`](DEFENSE_EXPERIMENT_PLAN_VI.md).

This audit answers two different questions:

1. Is a metric established and appropriate for this problem?
2. Is the exact pipeline formula externally established, or is it a thesis
   design choice that must be justified by ablation?

Those questions must not be conflated. Dice, IoU, HD95, AUROC and Brier score
are established metrics. The exact upstream score and equal percentile-rank
fusion are project-specific choices, even though their ingredients have
published precedent.

The executable formula audit is
[`artifacts/final_pipeline/g4/metric_formula_audit.json`](../artifacts/final_pipeline/g4/metric_formula_audit.json):
26/26 independent fixtures pass, including tied-score AP/AUROC and MONAI 1.5.1
HD95/ASSD reference values. This verifies the implementation; it does not turn
project-specific pipeline weights into literature-standard formulae.

## 1. Problem fingerprint

- **Primary task:** binary semantic segmentation of tumor pixels, conditional
  on a supplied image-level tumor/normal label.
- **Secondary task:** lesion/object detection because an image can contain more
  than one disconnected lesion.
- **Important structure properties:** very small and highly variable lesion
  area, occasional multifocality, imprecise or incomplete predictions, and no
  trustworthy physical pixel spacing.
- **Important output properties:** one frozen binary mask per image plus, for
  candidate analysis, a finite proposal gallery and continuous selector scores.
- **Statistical unit:** image; the available `group_id` is a deterministic
  grouping heuristic, not a verified patient ID.

This follows the problem-fingerprint principle of Metrics Reloaded:
<https://www.nature.com/articles/s41592-023-02151-z>.

## 2. Segmentation metrics

| Metric | Formula / convention | Role | Status |
|---|---|---|---|
| Per-image Dice | `2TP/(2TP+FP+FN)` | Primary overlap endpoint; macro mean over tumor images | Implemented/tested |
| Per-image IoU/Jaccard | `TP/(TP+FP+FN)` | Secondary overlap endpoint and connection to standard WSSS reporting | Implemented/tested |
| Precision | `TP/(TP+FP)` | Diagnoses over-segmentation | Implemented/tested |
| Recall/sensitivity | `TP/(TP+FN)` | Diagnoses under-segmentation/missed extent | Implemented/tested |
| Micro Dice/IoU | Pool TP/FP/FN before applying formula | Pixel-weighted complement to macro metrics | Implemented/tested |
| Predicted/GT area ratio | `|P|/|G|` | Multiplicative extent bias | Implemented/tested |
| Relative volume difference | `(|P|-|G|)/|G|` | Signed extent error; called area difference in 2-D | Implemented/tested |
| Empty prediction | `|P|=0` | Complete absence of a prediction | Implemented/tested |
| Zero overlap | `|G|>0 and |P intersect G|=0` | Includes non-empty masks at the wrong location | Implemented/tested |
| HD95 | Maximum of the two directed 95th-percentile surface distances | Robust worst-boundary error | Implemented; MONAI 1.5.1 cross-check |
| ASSD | Mean of concatenated directed surface-distance samples | Average boundary error | Implemented; MONAI 1.5.1 cross-check |
| Lesion matching | Maximum-cardinality one-to-one component matching at IoU 0.10/0.25/0.50 | Multifocal lesion detection diagnostics | Implemented; thresholds explicitly exploratory |

Dice and surface-aware evaluation have strong medical-segmentation precedent:
the Medical Segmentation Decathlon used Dice and normalized surface Dice
<https://www.nature.com/articles/s41467-022-30695-9>; Taha and Hanbury review
20 medical segmentation measures and warn that multiple definitions coexist
<https://doi.org/10.1186/s12880-015-0068-x>. Recent medical SAM evaluation also
uses Dice and NSD following Metrics Reloaded
<https://www.nature.com/articles/s41467-024-44824-z>.

### Empty-set policy

- Both prediction and GT empty: Dice/IoU = 1.
- Tumor GT non-empty and prediction empty: Dice/IoU = 0; HD95/ASSD undefined.
- Tumor GT and prediction non-empty but disjoint: Dice/IoU = 0; HD95/ASSD remain
  defined and the case is counted as zero overlap.
- The primary macro Dice is computed only over the 184 tumor images. Normal
  images are reported separately by false-positive case rate and predicted
  area, so the 187 easy empty/empty cases cannot inflate tumor Dice.

### Boundary-distance limitation

BTXRD has no reliable physical spacing. HD95 and ASSD are therefore reported in
pixels on a named grid, never in millimetres. Native-grid values are primary;
320/448 values are sensitivity analyses. NSD is not computed until a tolerance
is justified independently of validation performance. An arbitrary tolerance
chosen to improve the result would be scientifically weaker than omitting NSD.

## 3. Aggregation and uncertainty

- **Primary point estimate:** macro mean Dice over tumor images. This gives a
  tiny lesion case the same case-level weight as a large lesion case.
- **Required complement:** micro Dice/IoU. It reveals whether a method performs
  well mainly because large lesions dominate pixel counts.
- **Distribution:** median and IQR of per-image Dice and extent ratio.
- **Subgroups:** native lesion area `<1%`, `1–<5%`, `>=5%`, with exact `n`.
- **Confidence intervals:** nonparametric bootstrap of complete canonical
  groups, not pixels.
- **Method comparisons:** paired group bootstrap of per-image deltas, with
  point delta, 95% percentile interval, positive-delta probability and
  win/tie/loss counts.
- **Multiple comparisons:** one primary contrast per experiment family;
  exploratory p-values, if shown, require Holm correction.
- **Random training:** three frozen seeds, reporting every seed and mean +/- SD.
  Deterministic replay arms do not receive artificial seed variance.

The bootstrap cannot manufacture independent patients: because `group_id` is a
heuristic, the thesis must call it a grouped sensitivity analysis rather than a
patient-clustered confidence interval.

## 4. Classification metrics required for E1

### Binary/collapsed tumor task

- AUROC and average precision/AUPRC from continuous tumor scores.
- Sensitivity, specificity, precision, F1, balanced accuracy and MCC at the
  predeclared operating threshold.
- Brier score, negative log-likelihood and a reliability diagram.
- ECE may be reported, but binning and bin count must be declared; Brier/NLL are
  less dependent on an arbitrary binning choice.
- Confusion matrix and class prevalence.

### Ten-class task

- Macro and weighted one-vs-rest AUROC and average precision.
- Macro F1, balanced accuracy, top-1 accuracy and per-class precision/recall/F1.
- Full 10x10 confusion matrix.
- Multiclass Brier score and NLL.
- The collapsed C10-to-B2 tumor probability is `1 - p(normal)` for softmax
  outputs. Summing nine probabilities is equivalent but less numerically
  concise.

Metrics Reloaded explicitly separates discrimination and calibration for image
classification. Calibration of modern neural networks and ECE/temperature
scaling are documented by Guo et al.
<https://proceedings.mlr.press/v70/guo17a.html>. The E1 comparison must report
downstream mask Dice as the decisive endpoint; better image AUROC alone does
not prove better weak localization.

## 5. Candidate-supply and selector metrics

These metrics answer where the segmentation error enters:

- **Oracle Dice:** maximum candidate Dice within the exact eligible gallery.
- **Candidate recall@tau:** fraction of tumor images whose oracle Dice is at
  least `tau`, for tau = 0.10/0.30/0.50.
- **Selected Dice:** actual Dice of the frozen selected candidate.
- **Selector regret:** `oracle Dice - selected Dice`.
- **Within-source regret:** best Dice in selected source minus selected Dice.
- **Cross-source regret:** full eligible oracle minus best Dice in selected
  source.
- **Source-choice confusion:** selected source versus oracle source.
- **Oracle@K:** oracle after the predeclared, upstream-ranked balanced cap K.
- **Runtime/storage:** seconds/image, peak VRAM, candidate count and artifact
  bytes.

Oracle and regret are diagnostic constructs, not deployable performance. They
may use validation polygons only after the gallery and all selections have been
frozen. They must never select a mask or tune a threshold.

## 6. Which pipeline formulas are project-specific?

| Formula | Scientific position | Required evidence |
|---|---|---|
| CAM, Grad-CAM, Grad-CAM++, LayerCAM | Published attribution methods | E2 matched comparison |
| Ten-class to binary attribution | Exact softmax identity `logsumexp(z_1..z_9)-z_0` | E1 three-seed downstream Dice; no fitted coefficients |
| Point/box prompts and SAM multimask | Published SAM interface; exact conversion is project-specific | E2/E5 prompt ablation |
| Rich proposal union | Motivated by incomplete CAMs and proposal-set methods, but exact three-source union is project-specific | E4 all source subsets and E5 budget curve |
| Upstream `0.60D+0.25M+0.15R` | Entirely project-specific coefficients | E7 U0–U6; do not cite as a standard formula |
| G1 MIL | MIL and negative-bag supervision have published precedent; exact feature blocks, pooling and coefficients are project-specific | E6 cumulative feature/loss ablations, matched capacity and three seeds |
| Equal percentile-rank fusion | Rank normalization is generic; exact equal weighting is project-specific | E8 versus score and alternative rank fusions |
| Reciprocal-rank fusion | Published by Cormack et al. | Included as an E8 comparator: <https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/> |

Image-label MIL itself is established by Ilse et al.
<https://proceedings.mlr.press/v80/ilse18a.html>. SAM/CAM use in WSSS is
supported by S2C <https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html>,
but S2C also shows that noisy CAM seeds can remain a bottleneck. Therefore these
papers justify testing the components, not claiming that this thesis's exact
composition is already proven.

## 7. Missing evidence, ordered by defense value

### Tier A: mandatory before the defense

1. E0 native/320/448 reconciliation for frozen WSSS and fully masks.
2. E4 seven localization-source subsets with fixed selector.
3. E6 selector controls: random, SAM score, upstream, G1, final fusion, oracle.
4. E8 all frozen fusion rules.
5. E1 B2 versus C10/C10-to-B2 using three identical seeds, including actual
   downstream segmentation.
6. E2 CAM family x prompt-type factorial.
7. E3 SAM ViT-B/L/H accuracy-resource comparison.
8. E5 exact gallery construction/budget/deduplication curve.
9. E6 G1 feature and loss ablations with three seeds.
10. E7 source-correct upstream-score ablation.

E2 selected-mask endpoint status (2026-08-07): complete for all 12 factorial
arms. LayerCAM+point is best at tumor Dice 0.205224. The paired attribution
main effect supports LayerCAM, while prompt contrasts are less certain. This
evidence justifies LayerCAM as the attribution family used by the final system;
it does not by itself justify the later rich-gallery or selector modules.

### Tier B: strongly recommended

- Anatomy, view, center and tumor-type subgroup results with sample sizes and
  bootstrap intervals; label them exploratory where `n` is small.
- Parameter count, peak VRAM, wall-clock seconds/image and disk per major stage.
- Reliability diagram and calibration metrics for B2/C10.
- Failure taxonomy: empty prediction, non-empty wrong location, over-extent,
  under-extent, fragmentation and multifocal miss.
- Qualitative panels selected by a predeclared rule: median case, lower quartile,
  largest positive delta and largest negative delta, never hand-picked only for
  visual success.

### Tier C: only if defensible metadata become available

- NSD with an externally justified boundary tolerance.
- Patient-level bootstrap if verified patient IDs replace heuristic groups.
- External-site generalization if an independent radiograph cohort can legally
  be obtained.
- Inter-observer variability if a second annotation exists.

## 8. What must not be claimed

- Do not call 0.288729 a test result; it is validation-selected performance.
- Do not call the supplied-label setting prompt-free deployment.
- Do not claim binary supervision is superior until E1 downstream Dice exists.
- Do not claim ViT-B is optimal until E3 exists; call it the frozen resource
  choice.
- Do not claim cap 243 or weights 0.60/0.25/0.15 and 0.5/0.5 came from a paper.
- Do not use oracle Dice as actual pipeline Dice.
- Do not interpret normal specificity as tumor segmentation quality.
- Do not report boundary distance in millimetres.

## 9. Exact formula registry

The defense should use the following registry verbatim.  “Established” means
that the metric or learning principle has a published definition.  It does not
mean that this thesis's aggregation, threshold, coefficient, or empty-case
policy is universal.  Every project-specific convention is named explicitly.

| Quantity | Exact implementation used here | Aggregation / exceptional cases | Scientific status |
|---|---|---|---|
| Tumor Dice | `2TP/(2TP+FP+FN)` | Primary: arithmetic mean of 184 per-tumor-image values. Tumor GT non-empty and prediction empty gives 0. Normal images are excluded from this mean. | Established overlap metric; primary endpoint is a predeclared thesis choice. |
| Tumor IoU | `TP/(TP+FP+FN)` | Same 184-image macro aggregation as Dice. | Established overlap metric. |
| Micro Dice/IoU | Apply the Dice/IoU formula after pooling TP/FP/FN over the cohort. | Large lesions receive more pixel weight; never substitutes for macro Dice. | Established aggregation, reported as a sensitivity estimand. |
| Pixel precision | `TP/(TP+FP)` | Macro mean on tumor images. Empty tumor prediction gives 0. | Established; diagnoses excess extent. |
| Pixel recall | `TP/(TP+FN)` | Macro mean on tumor images. Empty tumor prediction gives 0. | Established; diagnoses missing extent. |
| Area ratio | `|P|/|G|` | Median and IQR on tumor images; undefined for normal GT. | Standard volume/area diagnostic; in 2-D it is an area ratio. |
| Relative area difference | `(|P|-|G|)/|G|` | Median and IQR; negative means under-segmentation. | Standard relative volume-difference idea applied to 2-D area. |
| HD95 | `max(Q95(d(partial P,partial G)), Q95(d(partial G,partial P)))` | Pixels at native or named resized grid. Undefined when exactly one surface is empty; report eligible/excluded counts. | Established robust surface-distance family; exact directed-percentile convention is declared and tested. |
| ASSD | Mean of the concatenated two directed nearest-surface-distance samples. | Same unit and empty policy as HD95. Surface pixels, rather than directions, receive equal weight. | Established surface-distance family; this precise weighting convention is declared because libraries vary. |
| Lesion TP at IoU `t` | Maximum-cardinality one-to-one matching of 8-connected GT and predicted components with component IoU `>=t`. | Report precision/recall/F1 at `t=0.10,0.25,0.50`; thresholds are exploratory, not clinical. | One-to-one object matching is established; these three thresholds are project-specific diagnostics. |
| Normal false-positive case rate | Number of normal images with any predicted-positive pixel divided by number of normal images. | Report predicted-area median/p95 as severity. Do not mix empty/empty normal cases into tumor Dice. | Established case-level error rate; gating protocol must be stated. |
| Candidate oracle Dice | `max_j Dice(C_ij,G_i)` over the exact frozen eligible gallery. | Validation diagnostic after all candidates and choices are frozen; never deployable. | Analysis construct, not a performance claim or training target. |
| Selector regret | `oracle Dice - selected Dice`. | Report overall and by lesion-size group; split into within-source and cross-source regret. | Project diagnostic with a transparent algebraic definition. |
| Candidate Recall@Dice `t` | Fraction of tumor images with oracle Dice `>=t`, `t in {0.10,0.30,0.50}`. | Uses the complete frozen gallery. | Proposal-recall analogue; thresholds are project-specific. |
| Image AUROC | Probability interpretation: a random tumor receives a higher continuous score than a random normal, with 0.5 credit for ties. | Report each seed and mean +/- SD; use paired comparisons on the same validation images. | Established rank-discrimination metric. |
| Image average precision/AUPRC | Area under the stepwise precision-recall curve using the declared implementation. | Report prevalence beside AP; do not use trapezoidal PR AUC interchangeably. | Established, particularly informative under class imbalance. |
| Brier score | `N^-1 sum_i (p_i-y_i)^2` for the collapsed binary event. | Lower is better. | Established strictly proper scoring rule. |
| Binary NLL | `-N^-1 sum_i[y_i log p_i+(1-y_i)log(1-p_i)]`, with numerical clipping only. | Lower is better. | Established strictly proper scoring rule. |
| ECE-15 | `sum_b |B_b|/N * |accuracy(B_b)-confidence(B_b)|` with 15 equal-width bins. | Always state binning; reliability diagram, Brier and NLL accompany it. | Widely used but bin-dependent; never the sole calibration result. |
| G1 normalized LogSumExp | `tau[log(sum_j exp(s_j/tau))-log N_i]` over valid candidates, `tau=0.20`. | Smooth maximum with correction for bag size. Every bag must be non-empty. | LogSumExp and MIL are established; normalization and `tau=0.20` are project-specific and require E6. |
| Negative-bag instance loss | BCE-to-zero for every valid proposal in a normal image bag. | Tumor-bag non-winners remain unlabeled. | Follows the standard MIL premise that a negative bag has no positive instance. |
| Positive-winner loss | BCE-to-one only for the detached current argmax proposal in a tumor bag. | Remaining tumor candidates are ignored. | Project-specific self-training assumption; it needs its own cumulative E6 arm and must not be presented as a standard MIL theorem. |
| Percentile rank | For sorted values, tied items receive the average zero-based rank, divided by `max(N-1,1)`. | Computed within each image's eligible candidates. Stable lower frozen index is the final tie-break. | Midranks are established; this normalization and tie policy are declared implementation choices. |
| Final fusion | `0.5*r_G1 + 0.5*r_upstream`. | One candidate per image; then G1 and frozen index tie-breaks. | Project-specific. E8, not a citation alone, is the evidence for the equal weights. |
| Upstream score | `0.60D+0.25M+0.15R`, where `D` is in-mask CAM-positive density, `M` is captured CAM mass, and `R` is component-local SAM predicted-IoU percentile rank. | Must use the prompt/saliency map belonging to the candidate's own source. | Entire coefficient vector is project-specific. E7 is mandatory. |
| Ten-class collapsed target | `logsumexp(z_1,...,z_9)-z_0 = log(P(any tumor)/P(normal))`. | No subtype label is needed at inference. | Exact probability identity for a softmax model, not a learned fusion heuristic. |

The published anchors are CAM/Grad-CAM/Grad-CAM++/LayerCAM, SAM, attention
MIL, S2C and the metric papers already linked above.  The exact thesis scores
are novel engineering hypotheses.  Calling them “standard formulas used by
many papers” would weaken, rather than strengthen, the defense.

## 10. Defense experiment matrix and decision rules

Every row below answers one foreseeable objection with a matched experiment.
The decisive endpoint is always the frozen **actual binary-mask Dice**, unless
the row explicitly studies classification or computation.  Oracle, AUROC and
loss are explanatory outcomes and cannot replace actual Dice.

| ID | Defense question | Arms and control | Held fixed | Required output | Current status / allowed conclusion |
|---|---|---|---|---|---|
| E0 | Is the result an artifact of resize coordinates? | Native, 320, 448 replay for WSSS and fully supervised | Exact frozen masks/checkpoints and cohort | Macro/micro Dice/IoU, subgroups, migration matrix | Complete. WSSS varies by at most 0.000505 and subgroup counts remain 94/72/18. Coordinate choice does not explain the result. |
| E1 | Why binary rather than 10-class supervision? | B2 versus matched C10, seeds 42/43/44; collapse C10 by exact tumor log-odds; then identical downstream pipeline | Backbone, optimizer, epochs, checkpoint endpoint, split, SAM-B, gallery, G1, fusion | Image discrimination/calibration **and downstream selected/oracle Dice** | Image stage complete; C10 is better calibrated and slightly higher in mean discrimination, but downstream Dice is pending. No binary-superiority claim yet. |
| E2 | Why LayerCAM and which prompt? | 4 attribution methods x point/box/point+box; CAM-only threshold control | Classifier, source, SAM-B, candidate/evaluator protocol | Actual Dice/oracle/subgroups, paired main effects, runtime | 12 selected-mask arms complete. LayerCAM has supported positive main effects; point is numerically best but prompt contrasts cross zero. CAM-only/oracle enrichment pending. |
| E3 | Why SAM ViT-B? | ViT-B/L/H end-to-end | Inputs, prompts, multimask, gallery merge/dedup/cap, G1, fusion | Selected/oracle Dice, subgroups, seconds/image, peak VRAM, disk | B/L running; H queued. Until complete, ViT-B is only a frozen resource choice, not proven optimal. |
| E4 | Why three localization sources? | All seven non-empty subsets of LayerCAM-320, LayerCAM-448 and external saliency | Exact selector, cap policy and cohort | Candidate count/oracle/Recall@Dice, selected Dice, regret, runtime/storage | Fixed-selector replay complete. It supports complementarity only where paired selected/oracle deltas and source regrets agree. Retrained-per-subset G1 is a separate optional question. |
| E5 | Why a rich gallery and why cap 243? | Upstream top-1; one exact prompt; three multimasks; full pre-dedup; post-dedup; caps 27/81/162/243 | Source generation and selector | Oracle@stage/K, selected Dice, truncation regret, candidate count, resource curve | Cap curve complete and monotonic to 0.288224 at 243. Exact prompt/dedup necessity still missing because old payload lacks prompt IDs; regeneration is mandatory for the exact claim. |
| E6a | Does learned G1 add information? | random, SAM-IoU, upstream, G1, final fusion, oracle | Identical eligible candidate set | Actual Dice, paired deltas, regret/source error | Complete: final 0.288224 native versus upstream 0.225306, G1 0.205545, random 0.101890, SAM 0.098902. G1 alone is not superior; its complementarity under fusion is the defensible claim. |
| E6b | Which G1 features/losses are necessary? | Four cumulative feature arms and four cumulative loss arms, 3 seeds, matched capacity | Cached RAD-DINO descriptors, scorer size, train budget, split | Selected Dice/oracle/regret plus AUROC/AUPRC/calibration and seed variation | Code/protocol complete; GPU execution pending. Exact self-guided winner term remains unproven until this finishes. |
| E7 | Why `0.60D+0.25M+0.15R`? | U0--U6: raw SAM, D, M, equal combinations, current local rank, global-rank variant | Candidates, G1, fusion/evaluator | Actual Dice, paired CI, subgroup/regret | Pending source-correct replay. Cannot use the merged anchor prompt map for other sources. Current coefficients must be called frozen empirical weights, not literature constants. |
| E8 | Why equal percentile-rank fusion? | upstream, G1, z-score, robust-z, min-max, RRF, percentile weights .25/.50/.75 | Same candidates and scores | Actual Dice and paired CIs | Complete: equal percentile rank is best of declared rules at 0.288224 native/0.288729 common-320; alternatives z-score 0.286375 and RRF 0.285523 are close, so report effect size rather than “universal optimum.” |
| E9 | Is the selector or proposal supply the dominant ceiling? | Selected versus within-source oracle versus all-source oracle | Frozen gallery | Regret decomposition, oracle-source confusion, Recall@Dice, misses | Required synthesis after E2/E3/E5. Existing all-gallery oracle 0.528298 versus selected 0.288729 already shows large selector regret, but absence of a good proposal for some cases remains measurable via Recall@Dice. |
| E10 | Does the method fail differently by lesion burden? | Predeclared `<1%`, `1-<5%`, `>=5%` | Native subgroup definition | Dice/IoU/precision/recall/RVD, zero-overlap, lesion recall, paired deltas with exact n | Mandatory for every final/ablation comparison. Never choose a subgroup-specific rule using validation GT burden. |
| E11 | Is the gain practically worth the complexity? | Baseline stages and B/L/H resource arms | Named hardware, batch and precision | Parameters, peak VRAM, wall time/image, total stage time, disk/candidates | Partially implemented; E3 supplies backbone telemetry. Final end-to-end table is still required. |

### Fixed interpretation rule

For every deterministic ablation report `(delta Dice, 95% paired grouped
bootstrap CI, win/tie/loss)` against its immediate reference.  A numerical gain
whose interval crosses zero is an observed improvement, not established
superiority.  For stochastic learned arms, show all seeds and compare matched
seed deltas; never select the most favourable seed for the thesis table.

## 11. Metrics intentionally not added by default

| Candidate metric | Decision | Reason |
|---|---|---|
| Pixel accuracy / pixel specificity as headline | Do not use as headline | The overwhelmingly normal background can make a poor tiny-lesion segmenter look excellent. Pixel specificity may appear only as a diagnostic. |
| MCC on all pixels | Optional diagnostic only | Although established, it mixes the vast background with lesion overlap and does not answer the primary case-level tumor-boundary question better than Dice plus precision/recall. |
| NSD / surface Dice | Do not run yet | It requires a distance tolerance with clinical or inter-observer justification. BTXRD has no reliable spacing or second annotation. Validation-tuning a tolerance is not defensible. |
| FROC | Conditional | Add only if the thesis makes a lesion-detection claim and each predicted component has a frozen continuous confidence. The current final output is one semantic mask, so one-to-one lesion precision/recall/F1 is the honest object diagnostic. |
| Per-image best threshold, GT-area matching, oracle routing | Prohibited | These use validation spatial truth to alter the prediction and would inflate actual Dice. They may appear only as labelled oracle analyses. |
| Calibration of G1 rank scores | Do not report as probability calibration | G1 is used as a within-image ranker, not a calibrated tumor probability. Ranking regret and selected Dice are the relevant endpoints. |
| Millimetre HD95/ASSD | Prohibited | No trustworthy pixel spacing. Report pixels on the named grid. |

## 12. Minimum final result package

The thesis result is not complete until one immutable package contains:

1. exact split, source, checkpoint, protocol and evaluator hashes;
2. 371 frozen masks and 371 per-image rows for WSSS and fully supervised;
3. primary macro tumor Dice/IoU with 95% grouped sensitivity intervals;
4. micro overlap, precision/recall, RVD/area ratio, empty and zero-overlap;
5. conditional HD95/ASSD with units and excluded counts;
6. one-to-one lesion precision/recall/F1 and multifocal counts;
7. 94/72/18 lesion-size subgroups and exploratory metadata subgroups with `n`;
8. candidate oracle, Recall@Dice and regret decomposition;
9. the E0--E8 ablation table with actual Dice, not proxy-only metrics;
10. end-to-end runtime, peak VRAM, storage, parameter and candidate counts;
11. three-seed classifier/G1 tables where training randomness exists; and
12. predeclared qualitative failure panels, including failures rather than only
    successful examples.

Until E1-downstream, E3, exact E5, E6b and source-correct E7 finish, the safest
defense wording is: the current final system is the best **validation-selected
configuration among the completed declared arms**, while several component
optimality claims remain under experimental verification.
