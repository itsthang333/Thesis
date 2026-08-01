# Matched-normal candidate transplant: bounded causal diagnostic

## Why another mask-bag learner is not authorized

The current frozen reference is G1+upstream equal rank fusion at validation
Dice/IoU `0.2887294867/0.2168391813`, subgroup Dice
`0.1577232964/0.4352293348/0.3868735327`. Gallery oracle is `0.5282983322`, so
selection—not proposal supply—is the active bottleneck.

An inside-mask/local-ring MIL learner initially appeared to target the missing
candidate identity. A complete historical comparison rejects that launch:

- mask-bag v6 already pooled frozen RAD-DINO inside/context/difference
  descriptors, used all normal candidates as negatives, and invented a
  detached positive winner in tumor bags;
- it reached image AUROC `0.813386` but Dice only `0.217899`, proving that bag
  classification did not identify the positive instance;
- Geometry-v3 corrected descriptor coordinates but reached Dice `0.245482`;
- G2 negative-only MIL removed a source/count shortcut but had no positive
  instance signal and reached only `0.254326` after fusion;
- B2.2 supplied broad foreground evidence, but its score correlated with area
  at median Spearman `0.950055` and reduced Dice to `0.191726`.

Therefore the next experiment must create **candidate-specific positive causal
evidence without assigning the current winner as positive**, and must cancel
mask-shape, paste-boundary, source-count and recipient-background shortcuts.

## Hypothesis

If candidate `c` contains tumor-specific content, that content should retain a
tumor logit after it is moved from the original tumor radiograph into multiple
matched normal radiographs. Normal content copied through the exact same mask
and blending operation should not. This difference is a candidate-level
intervention rather than a CAM, area score, frozen-feature rarity score, or
latent MIL winner.

For tumor image `x`, candidate mask `m_c`, matched normal recipient `n_r`, and
independent matched normal sham donor `n_s`, define

`T_pos(c,r)  = paste(style(x -> n_r), m_c, n_r)`

`T_sham(c,r) = paste(style(n_s -> n_r), m_c, n_r)`.

The same mask, feathering, recipient, position, and area are used in both.
Only the copied content changes. With frozen binary classifier logit `f`,

`q(c,r) = f(T_pos(c,r)) - f(T_sham(c,r))`,

`q(c)   = mean_r q(c,r)`.

The sham difference cancels a large class of artifacts: mask geometry,
candidate area, boundary seam, recipient anatomy, and paste position cannot by
themselves increase `q`, because they are identical in both terms.

## Fixed matching and transforms

Recipients and sham donors come only from the 1,493 canonical train-normal
images. They must have a different group from the query and from one another.
The deterministic match order is:

1. exact anatomy;
2. exact view;
3. exact acquisition center;
4. minimum absolute log-aspect-ratio difference;
5. SHA-256 tie break on query/donor IDs.

Two recipient/sham pairs are frozen per query. Every image is transformed by
the classifier's existing 448-pixel letterbox. Candidate masks are projected
from their frozen square grid by nearest-neighbor interpolation. Paste masks
use one fixed seven-pixel feather. Source content is matched to the recipient
with one fixed robust affine intensity transform estimated from full valid
image content, never from a polygon or lesion area.

No anatomy/view/center value is used as a prediction feature; it only prevents
obvious mismatched-bone interventions. A matched ablation deliberately uses
deterministic random normal donors while preserving all other computation.

## Frozen selector variants

The first run is a no-training mechanism diagnostic using the immutable
classifier448 checkpoint and rich gallery. It freezes, for every candidate:

- positive and sham logits for both recipients;
- `q(c)` and recipient disagreement;
- candidate area and source for shortcut diagnostics;
- five selections before any validation polygon:
  1. immutable G1+upstream baseline;
  2. transplant-only rank;
  3. equal rank of baseline and transplant;
  4. 3:1 baseline/transplant rank;
  5. matched-random-donor control using the same 3:1 rule.

These are a finite mechanistic panel, not a validation-selected weight sweep.
The baseline is the only promoted endpoint before Stage B. Any other variant
is promotable only if its formula was frozen in Stage A and its actual Dice is
better than the baseline.

## Mandatory actual-Dice and failure decomposition

After independent Stage-A verification, Stage B opens exactly the canonical
184 validation polygons and reports binary-mask Dice/IoU overall and for
94/72/18 lesion-size subgroups. Proxy classification metrics cannot replace or
block this endpoint.

The diagnostic is promising only if at least one predeclared transplant fusion
beats Dice `0.2887294867`, beats its random-donor control, and does not improve
only the 18-image large subgroup while materially damaging small/medium.

Whether it passes or fails, report:

- within-selected-source and cross-source regret;
- rank of each gallery oracle under `q` and fused scores;
- candidate-area/source dependence of `q`;
- positive-versus-sham logit differences on tumor and normal queries;
- recipient-pair sign agreement;
- hit/miss and selected-area transitions by subgroup;
- paired complete-group bootstrap against the immutable baseline.

## Predeclared failure branches

- `q_pos ~= q_sham` or matched ~= random donors: the frozen classifier does
  not carry transplantable tumor evidence; do not train a transplant model.
- `q` rises monotonically with area: paste artifacts or total transferred mass
  dominate; inspect positive/sham cancellation and retire without an area
  normalization sweep.
- good image-level transplant separation but poor candidate rank: the
  classifier again uses nonlocal/background evidence; a learned head is not
  authorized.
- large-only gain: the score is another expansion prior; inspect
  selected/GT-area transitions and retire.
- material within-source rank gain with stable small/medium: only then design a
  train-time recipient-swap consistency head, zero-initialized on the baseline.

No threshold, alpha, epoch, seed, resolution, SAM, pseudo-label, or student
rescue follows a failed diagnostic. Test remains locked.

## Academic boundary

Training supervision remains image-level only; class-agnostic candidate masks
are generated without spatial GT. Train-normal images act as counterfactual
background/reference data. Validation polygons are evaluation-only after all
candidate scores and selections are frozen. This is WSSS, not fully supervised
segmentation and not validation-mask training.

The design transfers only general principles from primary work:

- CutMix shows that region transplantation can create localizable classifier
  evidence instead of blank/noise masking;
- progressive proposal mining shows that mask-out interventions can test
  class-specific proposal evidence, while also warning about noisy latent
  proposals;
- WSOL evaluation protocol work motivates freezing all formulas before spatial
  validation labels.

Unlike ordinary CutMix, the proposed score uses a same-mask normal-to-normal
sham and multiple matched recipients so that region shape and paste artifacts
cannot serve as the positive label.

## Primary sources

- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers
  With Localizable Features*, ICCV 2019:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html
- Li et al., *Weakly Supervised Object Localization With Progressive Domain
  Adaptation*, CVPR 2016:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Li_Weakly_Supervised_Object_CVPR_2016_paper.html
- Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html
