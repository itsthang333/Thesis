# S9 — SKELEX candidate-marginalized spatial likelihood

## Research boundary

S9 is a same-gallery selector experiment for BTXRD WSSS. Training may use only
the frozen train radiographs, image-level normal/tumor labels, and the existing
class-agnostic candidate masks. It must not use train or validation polygons,
validation size groups, validation-derived thresholds, BTXRD test, or a
consumer trained from the resulting masks.

This document is a static design, not an experiment registration or launch.

## Why S8 failed and what S9 changes

S8 proved that a SKELEX reconstruction winner can be non-random and consistent
under horizontal flip while still selecting the wrong lesion identity or
extent. Tumor and normal switch rates were nearly identical, every tumor switch
crossed family, and most selected masks collapsed in area. The randomization
p-value therefore measured structured reconstruction anomaly, not tumor
relevance.

S9 changes the learning signal rather than the randomization threshold:

1. The SKELEX encoder remains frozen, but an image-label-trained tumor direction
   replaces decoder reconstruction error.
2. The tumor direction is learned from all tokens in known-normal images and a
   differentiable marginal likelihood over all candidate masks in tumor images.
3. No current selector winner, detached argmax, pseudo-positive candidate, or
   validation metric appears in the target.
4. Candidate likelihood requires positive evidence inside a mask and negative
   evidence in its local ring. A mask that is too small leaves tumor evidence in
   the ring; a mask that is too large dilutes its inside likelihood with normal
   anatomy. This is one shared spatial likelihood, not a GT-size router.

## Frozen intended representation

- Public SKELEX revision:
  `skhoha/SKELEX@368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`.
- Exact public weights remain external and unmodified; expected SHA-256:
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`.
- Input is the existing square-padded radiograph at `512x512`, with checkpoint
  normalization and positional interpolation, yielding a `32x32`
  patch grid. No crop, tile, anatomy box, or tumor prompt is used.
- Use the fixed pair of encoder hidden layers `8` and `16`. The final layer is
  intentionally excluded because ViT final-token over-smoothing is a known
  WSSS failure mode. Each layer token is L2-normalized, concatenated, and passed
  through one shared `2048 -> 256 -> 1` GELU head. The head has exactly
  `524,801` trainable parameters and a zero-initialized output layer; the
  SKELEX encoder is frozen. There is no layer or width sweep.
- Candidate masks are area-projected from the immutable square-corrected grid to
  `32x32`. A radius-2 token dilation minus the candidate defines the local ring.
  Empty inside or ring mass fails closed.

The higher resolution and bounded nonlinear head supersede the original static
`320x320`, layer-16 affine draft before any real-data action. The change responds
to the measured small-lesion pixel scale and the repeated failure of low-capacity
global residuals; additional runtime is accepted because efficacy, not short
runtime, is the objective.

## Frozen intended objective

For token logit `e_p`, candidate fractional support `m_cp`, and ring `r_cp`:

```text
inside_c = sum_p m_cp log(sigmoid(e_p)) / sum_p m_cp
ring_c   = sum_p r_cp log(sigmoid(-e_p)) / sum_p r_cp
q_c      = 0.5 * inside_c + 0.5 * ring_c
```

For a tumor image, the latent candidate objective is the normalized marginal
negative log-likelihood:

```text
L_pos = -(logsumexp_c(q_c) - log(number_of_valid_candidates))
```

For a known-normal image, every content token is a valid negative:

```text
L_neg = mean_p softplus(e_p)
```

The batch objective is the mean over image losses, so candidate multiplicity
cannot reweight images. Candidate order must be exactly permutation invariant.
No candidate family, source, area, coordinates, accepted score, or subtype is
fed to the trainable head.

The one-shot optimizer is AdamW for exactly `32` epochs, batch size `8`, learning
rate `1e-3`, weight decay `1e-4`, seed `42`, no early stopping and final-epoch
checkpoint only. Encoder extraction batch size is `2` across two T4s. These are
not tunable from validation results.

At inference, the new evidence is `q_c`. The finite primary readout is the
unweighted mean of within-image percentile ranks for Geometry-v3, immutable
upstream score, and `q_c`; the control is the existing two-rank mean. There is
no fusion-weight, temperature, layer, ring-radius, resolution, threshold,
subgroup, or morphology sweep.

## Difference from prior and collaborator work

- S5 pooled 224px projected frozen SKELEX candidate/context descriptors into the
  existing MIL residual; S9 learns a shared nonlinear 512px token head through
  an exact latent mask likelihood.
- S7 trained against targets derived monotonically from current logits and
  forced the current argmax; S9 never constructs an instance target.
- S8 used class-agnostic reconstruction anomaly; S9 is explicitly
  image-label-conditioned and penalizes evidence left in the local ring.
- The rejected Gate-C deletion/insertion selector perturbed classifier inputs;
  S9 performs no deletion, insertion, occlusion, or counterfactual image edit.
- Collaborator SMILE uses a trainable DenseNet-FPN at 512 px, subtype heads,
  matched normal references, and no candidate masks during representation
  training on the rich gallery. S9 uses a frozen SKELEX encoder, binary labels
  only, candidate-marginal likelihood during training, and the central
  same-gallery artifacts. It neither copies nor launches SMILE.

## Predeclared safeguards and gates before any future launch

- Register `ĐANG LÀM` centrally and push before the first real input is opened.
- Heavy extraction/training/inference only on Kaggle T4x2 or P100.
- Static tests must prove candidate-order invariance, exact normalized
  log-mean-exp, zero/current-logit independence, nonzero gradients, fractional
  mask behavior, ring fail-closed behavior, and control identity.
- A GT-blind operational audit must reproduce frozen encoder/token hashes,
  candidate projections, checkpoint, all candidate likelihoods, both prediction
  arms, and physical maps before validation GT can be read.
- Predictions must freeze physically before validation GT. Evaluation uses the
  existing corrected evaluator and exact Geometry-v3 baseline table.
- Mechanism gate: strict overall and small Dice improvement, no medium/large
  regression, and no complete-miss increase. Operational gate additionally
  requires the thesis goals and positive overall paired-CI lower bound.
- Failure closes the arm; no post-hoc objective, layer, resolution, radius,
  fusion, threshold, subgroup, area, or morphology rescue.

## Primary sources and constraints they support

- SKELEX domain foundation model and reconstruction/encoder provenance:
  https://www.nature.com/articles/s41746-026-02826-9
  and https://arxiv.org/abs/2602.03076 .
- WSDDN establishes proposal-level latent learning from image labels, while its
  proposal-softmax construction motivates explicit candidate normalization:
  https://openaccess.thecvf.com/content_cvpr_2016/papers/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.pdf .
- ToCo reports that intermediate ViT tokens retain semantic diversity while
  final tokens over-smooth:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html .
- Feature-direction alignment identifies class-vector/token-direction mismatch
  as a localization bottleneck:
  https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Bridging_the_Gap_Between_Classification_and_Localization_for_Weakly_Supervised_CVPR_2022_paper.html .
- Jang and Kwon show that bag accuracy alone does not ensure instance
  learnability; S9 therefore adds explicit normal-token negatives and candidate
  spatial likelihood rather than trusting a bag score as a localization proxy:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html .
- Choe et al. show why localization hyperparameters chosen with localization GT
  can create illusory WSOL gains; S9 freezes the finite arm before GT:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html .
