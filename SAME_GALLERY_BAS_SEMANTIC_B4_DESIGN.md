# B4 same-gallery class-contrast BAS semantic selector

Status: static preparation only. No claim, real-data fit, prediction, validation
segmentation-GT access, consumer training, or BTXRD-test access is authorized by
this document.

## Evidence and bottleneck

The accepted same-gallery Geometry-v3 selector reaches Dice
overall/small/medium/large
`0.24548239/0.11708058/0.37713552/0.38941265`, while the exact immutable
candidate gallery has oracle Dice
`0.40907553/0.22274949/0.59414708/0.64182537`. Candidate supply therefore
exceeds all four active goals. R1/R2/R3/R4/S1/S3/S4/T1 showed that normality,
local affinity, flip/orbit relations, family balancing, graph smoothing,
clustering, and count control can improve proxy behavior without improving
candidate identity.

The trusted collaborator result is used as architecture evidence without
re-audit or output access: rich-gallery G1 plus immutable upstream equal-rank
fusion reaches `0.28872949/0.15772330/0.43522933/0.38687353`. Its subsequent
identifiability analysis shows that 57.42% of eligible-selector regret is
irrecoverable by every monotone combination of those two scores; 30/31 severe
over-extent cases are small, while 25/26 under-extent cases are medium/large.
The two-score architecture is worth retaining, but it needs a third
source-agnostic tumor observable rather than another weight.

This execution uses only central same-gallery artifacts owned by `itsthang333`.
Collaborator outputs, checkpoints, predictions and Kaggle access are not read;
the published metrics and architecture are trusted as requested.

The B4 runtime uses a dedicated image-label-only dataset adapter. It reads the
frozen manifest, verified train/validation radiographs and binary image labels,
but never resolves, opens or hashes an `Annotations` file. This is required
because the shared historical classification constructor also verifies
annotation hashes as a general dataset-integrity check, which would cross the
validation-GT boundary before prediction freeze even though mask values never
enter its loss.

## One new scientific variable

B4 trains an ImageNet-initialized ResNet-50 Background Activation Suppression
(BAS) localizer with the frozen train split and binary image-level normal/tumor
labels only. At inference it converts the two learned localization maps into a
parameter-free class-competition map
`tumor / max(tumor + normal, 1e-8)`. This fixed contrast is motivated by
ReCAM's cross-class disentanglement and FPR's finding that absent-class
activations encode co-occurring background. It is intended to suppress normal
bone anatomy without imposing one global area prior. The gallery, masks,
candidate indices, accepted Geometry-v3 checkpoint, upstream scores, bag
logits, and bag probabilities remain immutable.

The fixed recipe is:

- input `448 x 448`, chosen once from prior BTXRD evidence that 448 improves
  small-lesion spatial fidelity; there is no 224/320/448 sweep;
- localization output stride 8 and classification output stride 16;
- final epoch 100, batch 32 on exactly two Tesla T4 devices;
- SGD backbone LR `0.001`, momentum `0.9`, weight decay `0.0005`, official BAS
  head LR multipliers, horizontal-flip train augmentation;
- full-image CE, foreground CE weight `0.5`, erased-background activation ratio,
  and fixed area weight `1.2`;
- original/aligned-horizontal-flip mean class maps at validation inference.

For each immutable candidate, class-contrast BAS evidence is the harmonic mean
of normalized contrast-map coverage and purity. Every score is converted to a
tie-aware within-image percentile rank. Exactly two arms are frozen:

1. `geometry_v3_plus_upstream_equal_rank`: arithmetic mean of accepted
   Geometry-v3 rank and immutable upstream rank, transferring the trusted
   two-score architecture onto the accessible same gallery.
2. `geometry_v3_plus_upstream_plus_class_contrast_bas`: arithmetic mean of the
   two control ranks and class-contrast BAS evidence. There is no coefficient,
   threshold, area, source, subgroup, epoch, resolution, or morphology search.

## Prediction-first boundary

Image AUROC, sensitivity/specificity, raw/contrast activation nondegeneracy,
contrast/control rank correlation, and changed-choice fraction are GT-blind
diagnostics. They may lock adoption or consumer training, but they do not
suppress the primary endpoint. After transport, provenance, cohort, finiteness,
and T4x2 checks pass, both 371-image arms must be physically frozen even when a
diagnostic is weak. An independent auditor that imports neither the B4 producer
nor any segmentation dataset must reproduce activation evidence, ranks,
choices, score payloads, maps, hashes, and pair freeze before validation
polygons can be opened.

Only then may the common evaluator compare both arms to the exact corrected
Geometry-v3 per-image table using 10,000 complete-group bootstrap replicates.
It must report overall/small/medium/large Dice, complete misses, selected-to-
oracle regret, oracle rank/top-k depth, count/miss association, and signed
gain/loss mass. B4 is an improvement only if actual Dice and regret improve;
proxy improvements alone are insufficient.

## Decision and safety

The operational goal remains Dice at least
`0.34024039/0.17895493/0.51244178/0.49370336`, with a positive overall paired
CI95 lower bound, no subgroup decrease, no complete-miss increase, and image
AUROC at least `0.75`. No consumer is trained before every operational check
passes. BTXRD test remains locked. A terminal failure rejects this exact
class-contrast BAS arm; it does not authorize a rescue sweep. Full
prototype-based FPR remains a separate successor only if B4 shows useful
semantic signal but residual anatomy false positives remain.

Primary sources:

- Wu et al., *Background Activation Suppression for Weakly Supervised Object
  Localization*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Background_Activation_Suppression_for_Weakly_Supervised_Object_Localization_CVPR_2022_paper.html
- Official BAS implementation: https://github.com/wpy1999/BAS
- Chen et al., *Class Re-Activation Maps for Weakly-Supervised Semantic
  Segmentation*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Class_Re-Activation_Maps_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2022_paper.html
- Chen et al., *FPR: False Positive Rectification for Weakly Supervised Semantic
  Segmentation*, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html
