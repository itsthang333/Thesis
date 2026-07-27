# Count-robust MIL and small-object localization addendum

Date: 2026-07-28

Scope: literature-derived contingency only. This document does not modify the
frozen RAD-DINO mask-bag MIL version-6 protocol, candidate manifests, model,
gate, operational goals, or validation predictions. No BTXRD pixel mask or test
sample was read to prepare it.

## Evidence motivating a conditional mechanism

The frozen no-GT manifest audit found a strong candidate-count shortcut:
direction-corrected count-only AUROC is `0.86726993` on train and `0.71207277`
on validation. Consequently, a high image AUROC can coexist with poor proposal
selection. Independently, the BTXRD operational goal is most difficult for
small tumors, so a failed small subgroup can arise either because the correct
proposal is absent or because a present proposal is ranked poorly. These are
different causal failures and must not be repaired with one undiagnosed method.

## Primary sources and transferable mechanisms

1. Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras,
   Abolfazl Razi, and Yalin Wang, *How Effective Can Dropout Be in Multiple
   Instance Learning?*, Proceedings of the 42nd International Conference on
   Machine Learning, PMLR 267:80090-80106, 2025.
   Official page: https://proceedings.mlr.press/v267/zhu25q.html

   The paper reports that dropping the top-k most important instances, and
   related instances selected by feature similarity, improves generalization
   and robustness across MIL benchmarks. The mechanism is relevant because a
   mask-bag selector may otherwise lock onto one easy proposal or a bag-level
   shortcut. It is not direct evidence for radiographic segmentation and does
   not prove that dropout improves Dice.

2. Tiancheng Lin, Zhimiao Yu, Hongyu Hu, Yi Xu, and Chang-Wen Chen,
   *Interventional Bag Multi-Instance Learning on Whole-Slide Pathological
   Images*, CVPR 2023, pp. 19830-19839.
   Official page:
   https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Interventional_Bag_Multi-Instance_Learning_on_Whole-Slide_Pathological_Images_CVPR_2023_paper.html

   IBMIL treats bag-context priors as confounders and uses a learned
   confounder dictionary with interventional training. This directly supports
   the diagnosis that bag construction can yield correct image predictions for
   the wrong reason. Its whole-slide pathology setting and multi-stage causal
   machinery are materially different from BTXRD, so it is a second-line
   option rather than the first implementation.

3. Cheolhyun Mun, Sanghuk Lee, Youngjung Uh, Junsuk Choe, and Hyeran Byun,
   *Small Objects Matters in Weakly-Supervised Semantic Segmentation*, WACV
   2024, pp. 413-422, DOI 10.1109/WACV57701.2024.00048.
   Official paper:
   https://openaccess.thecvf.com/content/WACV2024/papers/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.pdf

   The authors show that aggregate WSSS metrics can conceal systematic
   small-object failures and propose size-balanced evaluation and loss. This
   supports retaining BTXRD's explicit small/medium/large gate. Their
   size-balanced cross-entropy acts at dense segmentation training, so it is
   not an admissible repair for the current prediction-first proposal selector
   and must be deferred until a consumer is authorized.

4. Dongjun Hwang, Seong Joon Oh, and Junsuk Choe, *Small object matters in
   weakly supervised object localization*, Neurocomputing 648 (2025), 130494,
   DOI 10.1016/j.neucom.2025.130494.
   Publisher page:
   https://www.sciencedirect.com/science/article/pii/S092523122501166X

   This image-label-only WSOL study reports a zoomed
   foreground/background-consistency mechanism designed specifically for
   small objects, with improvements reported without sacrificing medium and
   large localization. The transferable principle is prediction-driven
   high-resolution re-observation with cross-view consistency. Its natural
   image bounding-box localization task is not equivalent to BTXRD Dice, so
   any adaptation requires a separately frozen protocol and direct subgroup
   evaluation.

## Frozen diagnosis and action order after version 6

1. **Oracle passes all operational goals, selector fails:** treat this as
   ranking failure. The first candidate is count-robust attention/relational
   MIL with train-time MIL-Dropout. Dropout acts on proposal descriptors only,
   is disabled at inference, and its schedule must be selected solely by
   group-preserving train-image-label cross-validation. The study must report
   count-only AUROC, predicted bag score versus candidate-count association,
   and localization gate metrics. A small initial `K` is necessary because
   dropping the sole true small-lesion proposal can otherwise strengthen
   background shortcuts. Exact `K`, similarity-neighbour count, probability,
   folds, seed, and stopping rule must be predeclared before validation
   predictions are generated.

2. **Oracle fails, especially for small:** do not train a more elaborate
   selector over inadequate support. Predeclare an image-label-only
   prediction-first zoom-consistency/high-resolution proposal probe. Crop
   coordinates must come from frozen model evidence, never GT or subgroup
   identity; every image follows the same deterministic routing; candidate
   maps are frozen before validation GT evaluation. Compare oracle coverage,
   complete misses, and all subgroup Dice against the existing proposal set.

3. **Oracle and selected gate both pass:** do not add either mechanism. Move
   to a separately predeclared pseudo-mask consumer protocol. Size-balanced
   dense loss may then be studied using pseudo-mask-derived sizes only; it may
   not use validation GT sizes or subgroup labels during training.

4. **IBMIL/deconfounding:** defer unless the simpler count-robust selector
   still shows strong score/count dependence after train-only cross-fitting.
   Its added complexity is not justified before that evidence exists.

## Non-adoption constraints

- Version 6 remains unchanged while running.
- Image AUROC alone cannot pass a localization experiment.
- No validation GT may select dropout, crop, threshold, architecture, or loss
  hyperparameters.
- No true size subgroup may be used for train-time routing or weighting.
- A method that improves overall/medium/large but decreases small still fails
  the existing no-subgroup-decrease requirement.
- BTXRD test remains locked and no pseudo-mask consumer is trained before the
  full prediction-first gate passes.
