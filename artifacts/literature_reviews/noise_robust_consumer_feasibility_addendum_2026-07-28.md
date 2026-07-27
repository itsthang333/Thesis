# Noise-robust consumer feasibility addendum

Date: 2026-07-28

Status: conditional research design only. This document does not authorize
consumer training, validation-mask access, or BTXRD test access. The running
mask-bag MIL version-6 prediction-first gate remains the boundary.

## Measured failure to solve

The prior image-label-only pipeline trained a 448-pixel ResNet18 U-Net on hard
CAM/SAM pseudo masks. It reached train positive Dice `0.73817187` against those
pseudo targets but only `0.23001987` frozen validation Dice; subgroup Dice was
`0.07100142/0.41340304/0.32691688` for small/medium/large. Its source pseudo
masks themselves reached `0.23433922` validation Dice. In contrast, the same
consumer family trained with real masks reached `0.49513170` overall.

This is evidence of pseudo-label memorization and spatial-noise propagation,
not evidence that another hard-mask training schedule will recover the missing
shape. In particular, hard background on a positive radiograph turns every
missed tumor pixel into a high-confidence false negative. A global Dice term
then couples those errors across the whole image and can be especially harmful
when the true lesion is small.

## Primary literature and limits of transfer

1. Jia Fu et al., *UM-CAM: Uncertainty-weighted multi-resolution class
   activation maps for weakly-supervised segmentation*, Pattern Recognition
   160 (2025), 111204:
   https://doi.org/10.1016/j.patcog.2024.111204

   This is the closest supervision regime: medical segmentation from
   image-level labels. Its Random-View Consensus trains a final segmenter from
   noisy pseudo labels by suppressing unreliable pixels and enforcing
   random-view agreement. The transferable mechanism is view-aligned
   reliability, not its fetal-brain thresholds or geodesic parameters.

2. Yude Wang et al., *Self-Supervised Equivariant Attention Mechanism for
   Weakly Supervised Semantic Segmentation*, CVPR 2020:
   https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html

   SEAM establishes that spatially aligned predictions should be equivariant
   under image transforms. For BTXRD, every resize, pad, crop and flip must be
   inverted exactly before a consistency loss is computed.

3. Antti Tarvainen and Harri Valpola, *Mean teachers are better role models:
   Weight-averaged consistency targets improve semi-supervised deep learning
   results*, NeurIPS 2017:
   https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html

   An exponential-moving-average model provides a temporally ensembled target.
   It is useful only as a stabilizer here; without an independent trustworthy
   spatial seed it can preserve the student's own mistakes.

4. Lequan Yu et al., *Uncertainty-aware Self-ensembling Model for
   Semi-supervised 3D Left Atrium Segmentation*, MICCAI 2019:
   https://arxiv.org/abs/1907.07034

   UA-MT estimates teacher uncertainty through stochastic passes and restricts
   consistency to reliable voxels, with a gradual ramp-up. Its experiments use
   real pixel labels for part of training, unlike BTXRD image-level-only WSSS.
   Therefore its numerical gains cannot be transferred, but its rule
   "uncertain teacher output is not a target" is applicable.

5. Xiaokang Chen et al., *Semi-Supervised Semantic Segmentation With Cross
   Pseudo Supervision*, CVPR 2021:
   https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.html

   CPS reduces dependence on one initialization through two-way pseudo
   supervision. It also assumes real dense labels for a labeled subset.
   Running two decoders from the same rejected BTXRD masks would not create
   independent evidence, so full CPS is not the first consumer.

6. Wenyu Li et al., *Pseudo-mask Matters in Weakly-supervised Semantic
   Segmentation*, ICCV 2021:
   https://openaccess.thecvf.com/content/ICCV2021/html/Li_Pseudo-Mask_Matters_in_Weakly-Supervised_Semantic_Segmentation_ICCV_2021_paper.html

   PMM explicitly suppresses high-loss pseudo pixels after warm-up and studies
   cyclic pseudo-mask updates. The robust-loss idea transfers; automatic cyclic
   replacement does not, because the BTXRD consumer has already demonstrated
   strong confirmation bias and has no spatial ground truth during training.

7. Cheolhyun Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
   Segmentation*, WACV 2024:
   https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

   The paper shows that aggregate WSSS evaluation hides small-object failure
   and that size-balanced training can improve multiple baselines. BTXRD may
   use only train-time pseudo/proposal area for balancing; true lesion size and
   validation subgroup identity remain forbidden inputs.

8. Qing En and Yuhong Guo, *Cross-model Mutual Learning for Exemplar-based
   Medical Image Segmentation*, AISTATS 2024:
   https://proceedings.mlr.press/v238/en24a.html

   Weak-to-strong cross-model perturbation and multi-resolution feature
   consistency motivate complementary views. This method uses exemplar masks
   and synthetic data, so its result is not evidence for the achievable BTXRD
   Dice. A second model is justified only after a single EMA consumer shows
   residual confirmation lock-in.

## What can be combined safely

The smallest coherent combination is:

1. a frozen prediction-first proposal source that has passed every localization
   gate;
2. family/view agreement to define pixel reliability;
3. partial soft supervision rather than a complete hard mask;
4. exact transform consistency with a single EMA teacher;
5. image-level tumor/normal BCE as the only semantic ground truth;
6. train-only proposal-area-balanced sampling to prevent large supports from
   dominating.

These mechanisms address different measured failures. Proposal agreement
controls label noise; EMA/random views control training instability;
image-level BCE prevents an empty positive solution; proposal-area balancing
protects small predicted supports. They should not be mixed with a new encoder,
cyclic relabeling, dual decoders, and post-processing in the first experiment,
because such a bundle would make attribution impossible.

## Conditional source eligibility

A consumer is eligible only if the upstream prediction-first protocol passes
all of its predeclared checks. Image AUROC, candidate oracle, or a point gain
alone is insufficient. The selected deployable map must pass the frozen
localization gate, show no subgroup regression or complete-miss increase, and
have a positive overall paired confidence-interval lower bound.

The source checkpoint, train and validation candidate galleries, per-candidate
scores/provenance, and every resulting teacher map must be frozen and hashed
before consumer construction. Validation masks are still unopened at this
point.

## Consumer C1: partial-consensus EMA U-Net

### Teacher reliability map

For every clean-train image, construct reliability only from frozen,
non-GT signals:

- align original/flip predictions through exact inverse geometry;
- aggregate candidate evidence by proposal family before combining families,
  so nine near-duplicate prompts do not count as nine independent votes;
- define confident foreground where independently generated family/view
  evidence agrees;
- define explicit confident background everywhere on image-level-normal
  images;
- on positive images, label background only where all eligible sources agree
  on absence and no source marks the pixel uncertain;
- leave every other positive-image pixel ignored.

The exact agreement statistic and train-only cutoffs must be declared after
the terminal source audit but before generating any validation prediction. A
winner-take-all proposal multiplied by one bag probability is not a calibrated
pixel probability; its constant interior score must not be misrepresented as
boundary uncertainty.

### Training objective

Use the already proven 448-pixel ResNet18 U-Net for the first experiment:

`L = L_partial_soft + lambda_img L_smoothmax_image +`
`lambda_view L_ema_equivariance + lambda_edge L_local_affinity`.

- `L_partial_soft`: balanced BCE only on reliable foreground/background
  pixels. Ignore ambiguous pixels. Do not apply a whole-image hard Dice loss
  to incomplete positive masks.
- `L_smoothmax_image`: binary image-label loss on the full output map. Normals
  must be empty; tumor images must retain some positive evidence.
- `L_ema_equivariance`: student and EMA teacher predictions from weak/strong
  views are compared only after exact spatial alignment and only where
  uncertainty is below a fixed train-only threshold. Ramp its weight from zero.
- `L_local_affinity`: encourage nearby pixels with similar radiograph and
  frozen RAD-DINO features to agree, but do not let it cross a strong image
  edge. This term cannot invent foreground labels.

No validation-mask checkpoint selection is allowed. Use a fixed horizon and
final EMA checkpoint, or a predeclared image-level-only holdout proxy.

### Small-lesion protection without GT size

- Form train sampling strata from frozen proposal-support area quantiles,
  computed on clean-train predictions only.
- Balance positive images across those strata and retain positive cases with
  empty/low-confidence support; do not silently discard probable misses.
- Add deterministic high-resolution crops centered on frozen candidate
  geometry, but do not assign the full-image positive label to each crop.
  Crops receive only aligned partial-teacher and full-to-crop consistency.
- Apply the identical crop-construction algorithm to normal images to avoid a
  label-correlated geometric shortcut.
- Report final Dice separately for the fixed `94/72/18` validation subgroups;
  never feed those subgroup identities to training.

This directly targets the observed pattern in which medium/large localization
improves while small objects disappear: large supports can dominate dense loss,
whereas a missed small lesion becomes almost entirely false background.

## Staged ablation and stop rules

Do not launch all ideas at once. If source eligibility passes, predeclare:

1. `C1a`: partial-consensus soft BCE plus image-level SmoothMax;
2. `C1b`: `C1a` plus EMA random-view equivariance;
3. `C1c`: only if `C1b` improves the no-GT training proxies without collapse,
   add proposal-area-balanced sampling and aligned high-resolution crops.

The expensive execution may share frozen maps and run the two T4 devices in
parallel, but each arm needs a distinct seed/hash/checkpoint and must not share
optimizer state. Promotion requires the unchanged operational Dice goals:
`overall >= 0.34024039`, `small >= 0.17895493`,
`medium >= 0.51244178`, `large >= 0.49370336`, with complete misses included.

If `C1a` cannot improve on the frozen deployable source, stop: consistency
cannot repair missing spatial information. If `C1b` increases medium/large but
decreases small, reject it rather than hiding the loss in overall Dice. A
cyclic self-training round or dual-decoder CPS is not automatically authorized.

## T4x2 execution design

The consumer is the first stage for which synchronized two-GPU data parallel
training is useful. Use PyTorch DistributedDataParallel with one process and
one model replica per T4; split batches through a distributed sampler and
synchronize gradients. This is different from the proposal-generation design,
where independent image shards are appropriate.

Freeze effective global batch size, learning-rate scaling, sampler seed,
mixed-precision mode, and BatchNorm behavior in the protocol. Record both real
device names and prove both ranks process nonzero disjoint samples. Map
generation remains a separate deterministic preprocessing job and can also be
sharded, but must finish and hash-freeze before DDP training.

## Decision

Do not implement or launch this consumer while mask-bag MIL version 6 is
unresolved. At terminal audit:

- if the deployable source passes, convert this design into a narrow immutable
  protocol beginning with `C1a/C1b`;
- if the oracle passes but selection fails, repair the relational selector
  first;
- if proposal oracle fails, improve high-resolution proposal support first;
- if the source fails, do not use EMA, robust loss, or a second decoder to
  disguise rejected pseudo supervision.

