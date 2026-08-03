# S10 — high-resolution proposal-conditioned MIL with Pareto extent guard

## Status and research boundary

This document is static preparation. It is not an experiment registration and
does not authorize opening BTXRD radiographs/candidate payloads, fitting on the
real cohort, producing validation predictions, or reading validation
segmentation ground truth.

S10 is a same-gallery selector experiment. It retains every immutable accepted
candidate and the accepted Geometry-v3/upstream control. Training may use only
clean-train radiographs, binary image-level normal/tumor labels and
class-agnostic candidate masks. It must not use train/validation polygons,
validation size groups, validation-derived thresholds, BTXRD test, collaborator
Kaggle/output, or a downstream pseudo-mask consumer.

## Failure mechanism being addressed

S9's image-label-trained likelihood contained real overall/medium signal, but
it changed 100/184 tumor winners and 112/187 normal winners, increased complete
misses from 70 to 72, and remained below the accepted comparator. Its candidate
likelihood was strongly anti-correlated with area (median Spearman -0.444 on
tumor bags), while high likelihood margin did not identify beneficial switches.
The scalar 32x32 token field therefore learned image class evidence without a
reliable proposal identity or extent criterion.

The broader evidence localizes three coupled bottlenecks:

1. selector regret dominates proposal truncation regret;
2. most regret is within-source candidate identity/ranking;
3. selected/GT extent has opposite signs: small lesions are over-segmented,
   medium lesions are mostly an identity/location problem, and large lesions
   are under-segmented. A global area coefficient or GT-size router is invalid.

S10 therefore changes both the representation and the decision structure. It
does not reuse S9 likelihood, S7 pseudo-instance targets, S6 subtype routing,
S8 reconstruction significance, or any validation-derived area rule.

## Frozen scientific hypothesis

A trainable high-resolution CNN-FPN can preserve local lesion evidence that a
32x32 frozen token field loses. Proposal-conditioned dual-stream MIL can align
the image-label objective with the proposal-ranking unit used at inference.
Separating tumor identity, evidence capture and evidence purity, then allowing
a switch only under component-wise Pareto dominance over the accepted control
winner, can improve small/medium/large cases without a global signed extent
correction.

## Intended representation

- Full square-padded radiograph at 640x640; no tumor crop, box or spatial label.
- ImageNet-1K ResNet-50 initialization with exact public weight SHA frozen in a
  later protocol. The backbone is trainable; this is deliberately heavier than
  the frozen low-resolution S5/S9 heads.
- A 128-channel FPN combines C2/C3/C4/C5 and exposes a stride-4 160x160 map.
- Each immutable candidate is area-projected to the stride-4 grid. The model
  pools candidate interior, a radius-3 exterior ring, global context and the
  interior-minus-ring contrast. Raw area, coordinates, family, source,
  accepted score and candidate count are not inputs to the identity head.
- A two-layer, four-head candidate-set transformer models relative candidates
  within the image. No candidate is dropped at inference.
- A dense 1x1 evidence head over the stride-4 FPN supplies continuous tumor
  evidence for extent diagnostics and the frozen readout.

## Image-label-only objective

The two proposal streams follow the proposal-level alignment principle of
WSDDN: a classification logit estimates candidate tumor identity and a
detection logit normalizes competition within a bag. Bag probability is the
sum of candidate classification probabilities weighted by detection softmax.

- Known-normal image: binary bag loss plus dense candidate-negative loss and
  dense pixel-negative loss. Every proposal/pixel is a legitimate negative.
- Tumor image: binary bag loss only; no hard/detached winner and no candidate
  pseudo-label.
- MIL-specific top-instance dropout is applied only while training after a
  fixed warm-up, forcing evidence beyond the current most discriminative
  proposal. It never removes candidates from the frozen inference gallery.
- Original/flip proposal-score and dense-map equivariance are trained with a
  fixed symmetric consistency loss.
- Within each bag, the squared projection of centered identity/detection logits
  onto centered log candidate area is penalized. Unlike squared correlation,
  this retains a nonzero correction gradient at a perfectly linear shortcut.
  Area remains available only for this nuisance penalty, never as a predictive
  feature.
- On tumor bags only, a soft attention-weighted union of candidate masks is
  matched to the dense evidence map. This ties proposal ranking to a spatial map
  without conflicting with dense normal negatives and without a polygon, hard
  mask target or current-control winner.

The later protocol must freeze the exact loss weights, warm-up, dropout rate,
optimizer, epochs and seed before any real input. No validation checkpoint
selection or hyperparameter sweep is permitted.

## Frozen extent decomposition and finite arms

For each candidate and dense evidence probability `e`:

```text
identity = mean(original/flip proposal identity logit)
capture  = sum(mask * e) / sum(content * e)
purity   = mean(mask * e) - mean(local_ring * e)
```

All three become tie-aware within-image percentile ranks. Capture rewards a
candidate that contains the image's evidence; purity rejects masks diluted by
normal surrounding anatomy. A candidate may replace the accepted control
winner only if it is no worse in all three ranks and strictly better in at
least one. Among Pareto dominators, maximize the minimum of the three ranks,
then identity rank, then the immutable candidate index. With no dominator, keep
the control winner exactly. This threshold-free conjunctive guard is the same
for every image and never observes a lesion-size group.

The future producer freezes three arms before validation GT:

1. accepted Geometry-v3 + upstream control;
2. capacity arm: control plus proposal-identity equal-rank evidence;
3. primary S10: Pareto-guarded identity/capture/purity selector.

The capacity arm separates representation value from the extent guard. The
primary must beat both the exact control and capacity arm to attribute a gain to
the full mechanism. No post-hoc threshold, margin, area, subgroup, source,
morphology or fusion rescue is allowed.

## Predeclared gates and safety

- Static tests must prove permutation equivariance, candidate-order invariant
  bag probability, finite gradients, all-normal dense negatives, area-nuisance
  behavior, flip alignment, soft-union arithmetic, tie handling and exact
  control fallback/Pareto dominance.
- Before real-data execution: fetch/read/collision check, register a unique
  `ĐANG LÀM` claim centrally, commit/push it, read the full Kaggle preflight
  checklist and freeze a hash-complete protocol.
- Heavy training/inference only on Kaggle T4x2 or P100.
- An independent GT-blind auditor must reproduce model/input hashes, all
  candidate scores, three prediction arms and physical maps. Predictions must
  be frozen and the readiness artifact committed/pushed before validation GT.
- Mechanism gate: primary strictly improves overall and small over both
  controls, does not reduce medium/large, and does not increase complete misses.
- Operational gate: Dice overall/small/medium/large at least
  0.34024039/0.17895493/0.51244178/0.49370336, paired overall CI95 lower bound
  strictly positive, no subgroup regression and no miss increase.
- No downstream consumer before operational pass. BTXRD test stays locked.

## Primary sources and transfer limits

- GLAM motivates coarse/global plus high-resolution local processing for small
  lesions under image-level medical supervision; S10 uses a shared FPN and
  proposal masks rather than reproducing GLAM's mammography ROI pipeline:
  https://proceedings.mlr.press/v143/liu21b.html
- WSDDN aligns learning and inference at the proposal unit through separate
  classification and detection streams:
  https://openaccess.thecvf.com/content_cvpr_2016/papers/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.pdf
- Proposal-based MIL motivates surrounding contrast, proposal completeness and
  rank consistency; S10 uses continuous capture/purity and flip consistency,
  not temporal pseudo-labels:
  https://arxiv.org/abs/2305.17861
- MIL-Dropout reports that dropping the most important training instances can
  reduce lazy reliance on noisy features; S10 freezes one bounded dropout rule:
  https://proceedings.mlr.press/v267/zhu25q.html
- Instance learnability analysis cautions that bag accuracy does not establish
  instance localization, motivating explicit normal proposal negatives and
  proposal-unit evaluation:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html
- Single-stage WSSS identifies local consistency, semantic fidelity and
  completeness as distinct requirements:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Araslanov_Single-Stage_Semantic_Segmentation_From_Image_Labels_CVPR_2020_paper.html
- The small-object WSSS study supports explicit size-stratified reporting but
  does not authorize using validation size labels for routing:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

## Non-collision statement

The collaborator's unproven SMILE preparation uses a DenseNet-FPN, subtype
heads, matched-normal references, rich-gallery training and no proposal masks
in representation learning. S10 instead uses ResNet-FPN proposal-conditioned
dual-stream MIL, binary labels, no matched normals, central same-gallery masks,
an area-nuisance loss and a Pareto fallback. It copies no collaborator code,
protocol or Kaggle output and is not promoted as an improvement before a
terminal audited result.
