# Proposal-bag selector contingency for image-label-only BTXRD WSSS

Date reviewed: 2026-07-27

Status: literature and mechanism analysis only. This document does not change
the running mask-bag v1 protocol, its validation gate, operational targets, or
the test lock. No new experiment is authorized before the version-6 terminal
audit identifies whether the limiting factor is proposal support or ranking.

## Project evidence available before validation GT

- Frozen train/validation bags: 2,981 / 371.
- Image-level populations: train 1,493 normal and 1,488 tumor; validation 187
  normal and 184 tumor.
- Candidate count per bag:
  - train mean/median/p90/max 58.59 / 63 / 81 / 81;
  - validation mean/median/p90/max 56.51 / 60 / 81 / 81.
- The running selector uses normalized SmoothMax bag pooling. After a
  two-epoch warm-up, its self-guided instance loss assigns a positive target
  only to the scorer's current highest-logit proposal in each positive bag.
  All other positive-bag proposals remain uncertain; all normal-bag proposals
  are negatives.

The current rule is supervision-valid, but its main statistical risk is
confirmation bias: an initially wrong winner can receive the only positive
instance target and reinforce itself. The risk is material when a typical bag
contains roughly sixty overlapping proposals.

## Primary literature

1. Ilse, Tomczak and Welling, *Attention-based Deep Multiple Instance
   Learning*, ICML 2018.

   Official record:
   https://proceedings.mlr.press/v80/ilse18a.html

   The paper learns a permutation-invariant, gated-attention distribution over
   instances directly from bag labels. Unlike a detached hard winner, every
   instance can contribute differentiably to the bag representation.

   BTXRD transfer: replace the hard positive-winner auxiliary target with a
   soft attention aggregator over the same frozen proposal descriptors.
   Preserve an explicit per-proposal class logit for localization; do not
   reinterpret attention alone as calibrated segmentation probability.

2. Cinbis, Verbeek and Schmid, *Weakly Supervised Object Localization with
   Multi-fold Multiple Instance Learning*, IEEE TPAMI 2017; preprint 2015.

   Stable preprint:
   https://arxiv.org/abs/1503.00949

   The study identifies premature locking onto erroneous locations as a
   central MIL failure and prevents the model that assigns a positive instance
   from immediately fitting that same example by using multi-fold training.

   BTXRD transfer: if version 6 shows adequate oracle support but poor
   selection, form deterministic group-preserving folds within the clean train
   split. Out-of-fold proposal scores or attention weights may supply
   cross-fitted soft targets. Validation images and masks must never
   participate in this stage.

3. Li, Li and Eliceiri, *Dual-Stream Multiple Instance Learning Network for
   Whole Slide Image Classification with Self-supervised Contrastive
   Learning*, CVPR 2021.

   Official paper:
   https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html

   DSMIL combines an instance classifier with a second stream measuring
   relations between every instance and a critical instance. This addresses
   large, unbalanced bags and retains an interpretable instance score.

   BTXRD transfer: use the highest-scoring proposal descriptor only as a query
   for relational aggregation, not as a detached positive label. Proposal
   selection remains constrained to the original SAM masks. The WSI
   multiscale/pretraining recipe is not copied.

4. Seibold, Kleesiek, Schlemmer and Stiefelhagen, *Self-Guided Multiple
   Instance Learning for Weakly Supervised Disease Classification and
   Localization in Chest Radiographs*, ACCV 2020.

   Official record:
   https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html

   This radiograph-specific work motivates customized targets for uncertain
   patch instances rather than treating every non-winner as background.

   BTXRD transfer already present: normal-bag proposals are reliable negatives
   and non-winning positive-bag proposals are ignored. If the current hard
   winner fails, uncertainty treatment should be retained while replacing the
   self-labeling mechanism.

## Recommended post-terminal experiment only if ranking fails

The preferred next probe is a train-only-selected attention/relational MIL
selector:

1. Keep candidate generation, RAD-DINO snapshot, mask pooling, image labels,
   split, prediction-first evaluator and test lock unchanged.
2. Compare the existing SmoothMax scorer against a gated-attention/DSMIL-style
   scorer using deterministic group-preserving cross-validation on the 2,981
   clean-train images only.
3. Choose architecture and stopping epoch solely by out-of-fold image-level
   AUROC/log loss plus a predeclared normal-bag false-positive statistic.
   Segmentation masks, lesion sizes and validation images are forbidden.
4. Retrain the chosen fixed configuration on all clean-train images, freeze all
   371 validation predictions, then run the existing separate GT evaluator.
5. Retain the operational all-subgroup gate. A classification improvement
   without selected-proposal Dice improvement is a rejection.

Multi-fold cross-fitting is more expensive than a one-model ablation, but it
directly targets premature winner lock-in and provides train-only selection.
It should be preferred over a post-hoc sweep of instance-loss weights on
validation GT.

## Rejection/branch rule

- If version-6 candidate oracle fails an operational subgroup target, do not
  run this selector study: no ranking model can select a mask absent from its
  bag. Return to prediction-first proposal generation, especially
  high-resolution local proposals for small tumors.
- If oracle passes but selected Dice fails, this contingency is admissible
  after a new protocol is frozen.
- If the complete version-6 gate passes, do not launch the contingency; move
  to a separately predeclared pseudo-mask consumer.

