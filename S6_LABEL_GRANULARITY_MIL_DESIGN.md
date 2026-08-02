# S6 label-granularity MIL selector design

## Scientific question

The immutable same-gallery candidate set has enough oracle support to exceed all
four thesis goals, while repeated binary-image-label selector variants fail to
identify the right candidate.  The accepted RAD-DINO mask-bag model reduces all
BTXRD pathology to `normal` versus `tumor`, although the frozen split contains
image-level benign/malignant and nine tumor-subtype labels.  S6 asks whether
this unused image-level taxonomy can resolve within-bag candidate ambiguity
without changing proposal supply, mask geometry, encoder descriptors, or the
evaluation protocol.

## Evidence motivating the bounded test

- Geometry-v3 has validation Dice
  `0.24548239 / 0.11708058 / 0.37713552 / 0.38941265`, while the exact
  candidate oracle is
  `0.40907553 / 0.22274949 / 0.59414708 / 0.64182537`.
- R1/R2/R3/R4/S1/S3/S4/T1 show that normal prototypes, local affinity,
  critical-instance relations, orbit averaging, family balancing, proposal
  clusters, and count control do not close selector regret.  The original
  binary self-guided objective also assigns its current detached winner as the
  only positive instance, creating a self-reinforcing latent target.
- S5 changes the representation and gives local medium/large gains, but hurts
  small lesions and fails every operational goal.  Therefore S6 retains the
  finer 32x32 RAD-DINO descriptors and changes only the image-label semantics
  and uncertainty-aware routing.
- The training split contains 2,981 eligible images and all nine tumor
  subtypes.  Subtype labels are image-level metadata already present in the
  immutable split; no annotation polygon or candidate quality is needed.

## Matched pair

Both arms use the exact accepted Geometry-v3 checkpoint and selector-cache
records.  Both have the same nine-output, zero-initialized residual network,
optimizer, seed, epoch budget, batch order, normalized SmoothMax, flip
consistency, and residual-drift regularization.  The frozen base scorer is never
updated.

1. `coarse_control`: all nine output columns remain exchangeable because the
   model receives only the binary normal/tumor bag loss.  Its candidate residual
   is the mean of those columns.
2. `hierarchical_entropy_routed`: the same initialization also receives
   benign/malignant and exact tumor-subtype bag losses on tumor images.  At
   inference, the predicted subtype residual is shrunk continuously toward the
   coarse residual by
   `1 - entropy(subtype posterior) / log(9)`.  It never consumes a supplied
   validation subtype label.

Before scoring, each proposal descriptor is centered by the valid-candidate
mean of its own image.  The residual outputs are centered again across valid
candidates.  Consequently, the learned branch cannot improve its bag loss by
adding an image-global anatomy/view offset to every candidate, and both arms are
exactly identical to Geometry-v3 at zero initialization.

The loss contains no hard or soft candidate winner, no pseudo mask, no
segmentation target, and no validation-derived target.  Subtype cross entropy
uses a deterministic inverse-square-root class weight derived only from training
image-label counts.  Benign/malignant logits use normalized log-mean-exp within
their seven/two subtype groups, avoiding group-cardinality bias.

## Frozen execution and evaluation contract

- Static work and synthetic tests may run locally.  Real cache loading, fitting,
  and validation prediction run once on private Kaggle T4x2 (or P100 if the
  final wrapper is explicitly bound to it).
- Exact inputs are split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  selector-cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  cache manifest
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`,
  and baseline checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`.
- The runner writes both complete candidate-score vectors, maps, checkpoints,
  training histories, label-count evidence, image-level diagnostics, and one
  physical pair freeze before any validation polygon can be opened.
- The independent GT-blind auditor must reproduce the input/source hashes,
  2,981/371 cohorts, subtype taxonomy, zero-initialization identity, all 742
  score/map outputs, exact selected indices, entropy route, count/probability
  association, original/flip agreement, and every GT/consumer/test lock.
- Count association, binary AUROC, subtype macro metrics, entropy, route strength,
  and changed-selection fraction are diagnostics, not post-hoc model-selection
  gates.  Only integrity/safety failures block evaluation, so the bounded pair
  always yields scientific Dice information once physically frozen.
- After the audit passes, the unchanged evaluator reports actual Dice and
  10,000 complete-group paired bootstrap intervals for both arms versus each
  other and accepted Geometry-v3.  The mechanism is useful only if the
  hierarchical arm improves overall and small-lesion mean Dice over its matched
  control without decreasing medium/large mean Dice or increasing complete
  misses.  Consumer authorization still requires all thesis goals, a positive
  overall paired-CI lower bound, no subgroup regression, no miss increase, and
  the existing safety gates.
- Validation polygons remain closed before pair freeze; no consumer is trained
  before operational pass; BTXRD test stays locked.

## Primary literature

- Cole et al., *On Label Granularity and Object Localization*, ECCV 2022:
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7044_ECCV_2022_paper.php
- Wang et al., *Multiple Granularity Descriptors for Fine-Grained
  Categorization*, ICCV 2015:
  https://openaccess.thecvf.com/content_iccv_2015/html/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.html
- Jang and Kwon, *Are Multiple Instance Learning Algorithms Learnable for
  Instances?*, NeurIPS 2024:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html
- Liu and Ji, *Weakly-Supervised Residual Evidential Learning for
  Multi-Instance Uncertainty Estimation*, ICML 2024:
  https://proceedings.mlr.press/v235/liu24ac.html
- Li, *A Multiclass Multiple Instance Learning Method with Exact Likelihood*:
  https://arxiv.org/abs/1811.12346
- Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html

These works motivate the label-granularity, multiclass-MIL, uncertainty-shrinkage
and prediction-first protocol principles only.  S6 does not claim that any of
their benchmark gains transfer to BTXRD before the terminal audited pair.
