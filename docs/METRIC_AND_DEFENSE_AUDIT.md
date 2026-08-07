# Metric and defense-evidence audit

This audit answers two different questions:

1. Is a metric established and appropriate for this problem?
2. Is the exact pipeline formula externally established, or is it a thesis
   design choice that must be justified by ablation?

Those questions must not be conflated. Dice, IoU, HD95, AUROC and Brier score
are established metrics. The exact upstream score and equal percentile-rank
fusion are project-specific choices, even though their ingredients have
published precedent.

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
| G1 MIL | MIL has published precedent; exact feature blocks, pooling and losses are project-specific | E6 feature/loss ablations and seeds |
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
