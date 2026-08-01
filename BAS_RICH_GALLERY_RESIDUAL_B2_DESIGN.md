# BAS rich-gallery residual B2 design

Status: static design and synthetic transport audit only. No experiment claim,
real-data access, fitting, prediction, validation segmentation-GT access,
consumer training, or BTXRD test access.

## Evidence inherited because it is actually better

The collaborator result at commit
`d849155e49372c2027a1168fbe4a0b68e199470d` established the current best
observed weak pipeline on the frozen 371-image validation cohort:

- fixed equal percentile-rank fusion of rich-gallery G1 and immutable upstream
  score: Dice `0.28872949`, subgroup Dice
  `0.15772330/0.43522933/0.38687353`, 49 complete misses;
- gallery oracle: `0.52829833/0.33187635/0.73025092/0.74624721`;
- within-selected-source ranking accounts for 70.29% of selector regret;
- top-10 oracle-restricted Dice is `0.399326`, while the current misses have
  oracle-rank median 95.

The rich proposal union and fixed G1/upstream rank pair are therefore carried
forward. G2 and global cross-source consensus are not carried forward because
their audited Dice (`0.25432565` best G2 and at most `0.26707272` with
consensus) is worse. The old static same-gallery B1 protocol SHA-256
`5b9a80c63331551ff2c4ba0140096c14fa27076e141b83129e774843a7a7fde8`
is superseded before claim or launch.

The `0.28872949` result remains exploratory rather than confirmatory because
the equal-rank rule was designed after earlier validation Stage-B evidence.
B2 is consequently another development experiment, not a claim of independent
generalization.

The later collaborator commit
`8b1a38d459e3b4681f7ef5722451cc38deb7d67f` also rejected a fixed top-10
cross-source relational product: Dice `0.28564683`, subgroup
`0.12276538/0.46376513/0.42377677`, 45 misses. Its overall delta was
`-0.00308265` with CI95 `[-0.024322,0.016844]`, and its small-lesion delta
`-0.03495792` had strictly negative CI95 `[-0.063430,-0.011129]`. Relation
expanded stable anatomy and doubled the already excessive small-lesion area
ratio. B2 therefore contains no geometric relation, top-k or consensus score.

The term "residual" in the B2 identifier means one extra semantic evidence
rank in a fixed comparison; it does **not** claim to implement the later
recommended zero-initialized learned relational residual. Scientifically, B2
is the required independent-descriptor control: only if BAS alone has a
terminal paired gain may a future relational head use it as the matched control.
If B2 fails, relation may not be combined with BAS as a rescue.

## New scientific variable

B2 keeps every rich-gallery proposal, G1 logit and upstream score immutable. It
trains one ResNet-50 Background Activation Suppression (BAS) localizer using
only frozen train-split normal/tumor image labels. At validation, the aligned
original/flip tumor activation map supplies candidate-level activation
coverage and purity; their harmonic mean is converted to a within-image
average-tie percentile rank.

The finite prediction pair is:

1. `g1_upstream_control`: exact mean of frozen G1 and upstream percentile ranks;
2. `g1_upstream_bas_residual`: exact unweighted mean of G1, upstream and BAS
   percentile ranks.

There is no rank weight, source weight, threshold, candidate deletion, top-k
restriction, area prior or lesion-size router. Stable ties are resolved by the
frozen G1 raw logit and then the lower local candidate index, exactly matching
the collaborator baseline. Because BAS scores every retained candidate, it can
change both the bounded top-rank and deep-rank miss branches; post-freeze
diagnostics must report them separately.

## Resolution and training recipe

Input resolution is fixed once at 448 pixels rather than carrying the earlier
224-pixel static probe forward. This is not a sweep: prior BTXRD evidence found
that 448-pixel training improved small-lesion localization, classifier448 has
the strongest small-source oracle in the rich gallery, and a 224-pixel
output-stride-8 activation grid would reduce already tiny lesions to very few
cells. No 224/320 comparator is authorized.

The remaining BAS recipe is fixed from the official implementation:

- ImageNet-initialized ResNet-50;
- output stride 8 localization and output stride 16 classification;
- full-image CE + `0.5` foreground CE + background/full activation ratio +
  `1.2` activation area constraint;
- horizontal flip only, fixed final epoch 100, batch 32 across exactly T4 x2;
- SGD Nesterov, backbone LR `0.001`, body-bias/head-weight/head-bias multipliers
  `2/10/20`, momentum `0.9`, weight decay `0.0005`;
- no segmentation metric, early stopping, epoch selection, validation mask,
  lesion size, candidate Dice or oracle rank during training/prediction.

## Required GT-blind collaborator transport

B2 must not regenerate the rich gallery or rerun G0/G1/G2. Before a claim can
be registered, an exact no-GT transport from the `wanwin` workstream must be
available and independently accepted. It contains only:

- `prediction_freeze.json`, `stage_a_selection_manifest.csv` and all 371
  `stage_a_scores/*.npz` from G2 Stage-A;
- the validation rich-gallery candidate manifest/summary and 371 physical
  candidate payloads;
- exact G1 checkpoint/source/protocol/split/input hashes needed for provenance.

It must not contain Stage-B evaluation, `per_image.csv`, validation polygons,
annotation/ground-truth material or test data. The independent auditor
`project/audit_rich_gallery_stage_a_transport.py` rejects those paths, verifies
the collaborator freeze and score-set hashes, aligns every kept index/source/
upstream score against its physical candidate payload, and reproduces all 371
G1+upstream selections. Current Kaggle credentials receive HTTP 403 for the
private `wanwin` kernel, so no output was downloaded and no GT boundary was
crossed.

## GT-blind operational gates

Before either arm can be emitted as a prediction:

- transport independent audit passes and reproduces 371/371 control choices;
- final BAS image AUROC is at least 0.75, sensitivity and specificity at 0.5
  are each at least 0.60;
- at least 95% of the 184 image-label-positive validation maps have activation
  range above `1e-4`;
- BAS mean within-bag rank correlation is at most 0.80 against each of G1 and
  upstream, and B2 changes at least 5% of the control selections;
- exact T4 x2, split, ImageNet initialization, source, protocol and no-test
  contracts pass.

Failure freezes a GT-blind negative result and forbids Dice evaluation. On
pass, both 371-choice arms, candidate scores and selected immutable masks are
physically frozen as one pair before a separate evaluator can import validation
polygons.

## Post-freeze decision

The primary paired comparator is B2 minus exact rich-gallery G1+upstream, not
the old same-gallery Geometry-v3 result. A useful mechanism requires a positive
lower 95% complete-group bootstrap bound overall, no negative subgroup mean,
and no increase in complete misses. It is a full goal pass only at Dice at
least `0.34024039/0.17895493/0.51244178/0.49370336`. A consumer remains locked
unless the full operational goal gate passes. Regardless of outcome, report
source transitions, changed-case Dice, top-rank gains, deep-rank miss
recoveries/losses and remaining within/cross-source regret. No rescue sweep is
authorized.

## Primary sources and transfer boundary

- Wu, Zhai and Cao, *Background Activation Suppression for Weakly Supervised
  Object Localization*, CVPR 2022, DOI `10.1109/CVPR52688.2022.01385`,
  https://arxiv.org/abs/2112.00580 and https://github.com/wpy1999/BAS.
  B2 transfers only image-label BAS activation as candidate evidence.
- Tang et al., *Multiple Instance Detection Network With Online Instance
  Classifier Refinement*, CVPR 2017,
  https://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_Instance_Detection_CVPR_2017_paper.html.
- Tang et al., *PCL: Proposal Cluster Learning for Weakly Supervised Object
  Detection*, TPAMI/arXiv `1807.03342`, https://arxiv.org/abs/1807.03342.
- Wan et al., *C-MIL*, CVPR 2019,
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wan_C-MIL_Continuation_Multiple_Instance_Learning_for_Weakly_Supervised_Object_Detection_CVPR_2019_paper.html.
  OICR/PCL/C-MIL motivate instance refinement, but their inferred-positive
  propagation is deferred because current top-1 and raw overlap are often
  wrong; B2 first tests independent tumor evidence without pseudo-positive
  propagation.
- Xu et al., *CREAM*, CVPR 2022,
  https://openaccess.thecvf.com/content/CVPR2022/html/Xu_CREAM_Weakly_Supervised_Object_Localization_via_Class_RE-Activation_Mapping_CVPR_2022_paper.html.
  Foreground/background re-activation is relevant, but is deferred rather than
  combined because global consensus already showed that expanding stable
  anatomy can harm small lesions. It may be reconsidered only after BAS has an
  audited positive result.
- Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020, https://arxiv.org/abs/2001.07437. This is why development
  metrics and untouched confirmatory evidence remain explicitly separated.
