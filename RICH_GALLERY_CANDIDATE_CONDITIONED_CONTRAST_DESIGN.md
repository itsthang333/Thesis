# Candidate-conditioned contrast residual: bounded successor design

## Decision target

The immutable reference is G1 plus upstream equal percentile-rank fusion:

- validation Dice/IoU: `0.2887294867/0.2168391813`;
- subgroup Dice `<1% / 1-<5% / >=5%`:
  `0.1577232964/0.4352293348/0.3868735327`;
- rich-gallery oracle Dice: `0.5282983322`;
- candidate-truncation regret: `0.0003963063`.

The goal is not to create more proposals. It is to reduce selector regret by
adding candidate evidence absent from G1 and upstream while exactly preserving
the baseline when the new residual is zero.

## Evidence forcing this design

1. `57.42%` of eligible selector regret is jointly dominated under both old
   scores; 154/184 oracle candidates are outside their Pareto frontier.
2. Geometry/G2/consensus/relational selectors model regularity and context but
   do not establish which candidate contains tumor-specific content.
3. BAS-B2.2 changed 80.86% of choices but behaved as an expansion prior. It
   improved large lesions by `+0.152591` while losing `-0.129408/-0.117096` on
   small/medium lesions.
4. A GT-only oracle switch between baseline and B2.2 reaches `0.322338`, but a
   group-separated router using all current label-safe observables reaches only
   `0.287786`. Therefore another metadata gate cannot isolate the benefit.
5. The required correction changes sign with scale: tiny lesions need less
   normal bone inside the candidate, while fragmented medium/large lesions need
   abnormal evidence beyond the current boundary.

Primary literature supports three transferable principles, not a promise of
BTXRD success:

- patch- and image-level MIL supervision can train localized CXR predictions
  from image labels (Seibold et al., ACCV 2020);
- explicit foreground/background contrast is needed because activation
  expansion can overrun background (Ki et al., ACCV 2020);
- high-resolution feature maps are necessary for small medical findings, but
  resolution alone does not solve the classification/localization mismatch
  (PYLON, 2020).

The BTXRD failure evidence adds the missing safeguards: exact normal-candidate
negatives, no global/coordinate bypass, area-dependence measurement, and an
immutable baseline residual.

## Mechanism

For image `x`, candidate mask `m_c`, and a local ring
`r_c = dilate(m_c) - m_c`, a shared high-resolution encoder produces feature
map `F(x)`. It has no absolute coordinate channels and no image-global shortcut
into the candidate score.

The candidate descriptor contains separately pooled evidence:

`z_c = [topq(F|m_c), mean(F|m_c), lowq(F|m_c),`
`       topq(F|r_c), mean(F|r_c), topq(F|m_c)-topq(F|r_c)]`.

The three interior statistics separate a tiny abnormal core from normal tissue
included by an oversized proposal. The ring statistics make the extent signal
signed: a high abnormal ring penalizes an under-extended candidate, whereas a
normal-like interior tail penalizes an over-extended candidate.

A small shared head outputs residual `r_theta(c)`. The deployed selector is

`s(c) = b(c) + eta*tanh(r_theta(c))`,

where `b(c)` is the frozen centered G1/upstream equal-rank score and `eta` is a
single predeclared constant. The last residual layer is initialized to zero,
so the first and fallback prediction exactly reproduce Dice `0.2887294867`.

## Image-label-only training

All candidates from each of the 1,493 canonical train-normal images are direct
tumor-negative instances:

`L_normal = mean_c softplus(r_theta(c))`.

For a tumor image, only the bag is positive:

`L_tumor = softplus(-tau*logsumexp_c(s(c)/tau))`.

Additional label-safe terms are:

- candidate score consistency under horizontal flip/intensity style change;
- source-balanced bag pooling so one proposal generator cannot dominate;
- within-normal-bag covariance penalty between residual and log candidate area;
- local ring consistency under small candidate-preserving transforms.

No candidate is declared positive from a validation polygon, GT area, SAM
prompt, test image, or per-image threshold. The combined objective supplies
the two pieces split across prior failures: B2.2 lacked direct tumor-channel
normal negatives, while G2 negative-only selection lacked candidate-specific
positive evidence. Here the negative term teaches the tumor channel to reject
normal candidate content, and the positive MIL bag forces at least one tumor
candidate to remain.

## Matched diagnostic before a full run

Run two fixed canonical passes with two arms:

- **control:** same encoder, capacity, optimizer, batches, and losses, but the
  candidate mask/ring assignment is deterministically permuted within each
  image;
- **full:** correct candidate interior/ring assignment.

Both arms freeze one score per existing gallery candidate and 371 final masks
before validation polygons. Stage B always reports actual binary-mask Dice,
IoU and 94/72/18 subgroups; proxy metrics cannot block or replace Dice.

The diagnostic supports a full run only if all hold:

1. full Dice exceeds both immutable baseline and matched control;
2. improvement is not large-only: small and medium do not each regress by more
   than `0.01`;
3. within-selected-source regret decreases materially;
4. residual-versus-area dependence on tumor and normal bags is below B2.2 and
   no source supplies more than 70% of all positive residual mass;
5. exact zero-residual reproduction, no-GT/no-test provenance, and frozen-mask
   audits pass.

The research goal remains the subgroup contract
`0.195607621/0.479674337/0.513613009`; a cheap diagnostic need not reach it,
but it must beat `0.2887294867` with the correct causal signature before more
epochs are authorized.

## Predeclared failure branches

- If full equals control, candidate content is not being used: retire this
  representation, not the gallery.
- If residual correlates strongly with area, the mechanism has repeated BAS;
  retire without an area/weight sweep.
- If only large improves, ring/interior pooling is an expansion prior; inspect
  core-versus-tail tensors and retire before full training.
- If classification improves but Dice does not, candidate-positive MIL remains
  non-identifiable; inspect candidate ranks and consider cross-image positive
  prototypes, not another threshold.
- If within-source regret falls but cross-source regret does not, preserve the
  candidate head and study source calibration separately with a matched arm.

No threshold, fusion-alpha, epoch, seed, resolution, SAM, pseudo-label, or
student rescue is authorized after a failed diagnostic. Test remains locked.

## Primary sources

- Seibold et al., *Self-Guided Multiple Instance Learning for Weakly
  Supervised Disease Classification and Localization in Chest Radiographs*,
  ACCV 2020:
  https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html
- Ki et al., *In-sample Contrastive Learning and Consistent Attention for
  Weakly Supervised Object Localization*, ACCV 2020:
  https://openaccess.thecvf.com/content/ACCV2020/html/Ki_In-sample_Contrastive_Learning_and_Consistent_Attention_for_Weakly_Supervised_Object_ACCV_2020_paper.html
- Preechakul et al., *High resolution weakly supervised localization
  architectures for medical images*:
  https://arxiv.org/abs/2010.11475
- Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html
