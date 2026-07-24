# Improvement plan

The experiment order is chosen to maximize information gain under the 20-hour limit while preserving a locked test protocol.

Promotion rule: an experimental technique is kept in production source only after a full-validation gain or a correctness benefit is demonstrated. Rejected techniques must be removed together with their CLI/config surface, tests, dependencies, checkpoints, caches, and debug outputs; only compact evidence remains in the research log.

## Gate A — credible supervised upper bound

Train the pretrained ResNet-18 U-Net on clean train polygons and select by clean validation tumor-only Dice. Sweep the decision threshold only on validation. If this cannot exceed 0.50, architecture/loss/optimization is the immediate bottleneck and pseudo-label work cannot rescue the final model reliably.

Primary run: 320 px, AdamW, BCE + soft Dice, foreground weight clamped to 20, seed 42, up to 50 epochs, patience 10.

### Compute-aware learning-curve protocol

Short runs are used only for screening and are never reported as final evidence. Each candidate first receives a common minimum budget. Promotion to the next budget rung uses all of: (1) best validation tumor-only Dice, (2) robust slope over the most recent epochs for tumor Dice and validation loss, (3) train-validation gap, (4) normal-image specificity and its stability, and (5) validation-only threshold-sweep gain. A run may be rejected early only when it has no meaningful recent gain **and** shows saturation, divergence, or domination by another otherwise comparable run. A run stopped before that evidence exists is labelled `paused_inconclusive` and its resumable checkpoint is retained. Final model comparisons use reasonably converged checkpoints under the same split and evaluator; projected Dice is used only to allocate compute, never as the claimed result.

The initial budget rungs are 5, 10, and 20 epochs, with extension toward 50 epochs for the surviving configuration. Because an epoch can contain a transient calibration phase, the minimum 5-epoch rung cannot by itself reject a pretrained segmentation configuration unless it is invalid (NaN, broken masks, or protocol violation).

Operationally, epoch 5 is observation-only. At epoch 10, a candidate is extended when its best tumor Dice is within 0.03 of the incumbent, its robust recent slope is positive, or it has gained at least 0.02 during the latest five epochs. It is eligible for early rejection only when it trails the incumbent by more than 0.05, has gained less than 0.01 over five epochs, and has a non-positive Dice slope or a persistently worsening validation loss. Borderline curves advance rather than being discarded. At the selected checkpoint, per-image bootstrap confidence intervals and threshold-sweep behavior provide additional uncertainty/calibration evidence; these do not replace measured Dice.

This is a conservative adaptation of [Hyperband/successive halving](https://www.jmlr.org/papers/v18/16-558.html) for resource allocation and [Freeze-Thaw Bayesian optimization](https://arxiv.org/abs/1406.3896) for pause/resume decisions. Learning-curve extrapolation follows the safety principle in [Domhan et al., IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/487.pdf): terminate only when the observed partial curve provides sufficiently strong evidence that it cannot beat the incumbent. Here, the dataset is too small and the number of completed curves too low to justify a calibrated Bayesian predictor, so extrapolation remains a qualitative allocation aid rather than a reported metric.


If the 320 px survivor remains limited specifically on the predeclared small-lesion subgroup, the next Gate-A fidelity rung increases spatial resolution while keeping split, seed, architecture, loss, and evaluator fixed. A later ROI experiment is valid only when its proposal is available from weak-label/CAM inference at deployment; an ROI derived from ground-truth masks is oracle-only and cannot support the final claim.

The bounded tumor-only training ablation has been completed and rejected: at epoch 10 it reduced selected-threshold overall Dice from 0.38501 to 0.37715 and small-lesion Dice from 0.17463 to 0.14445. Its implementation and artifacts were purged. The 448 px full-population run is promoted because it raised overall Dice to 0.41832 and small-lesion Dice to 0.24904. Its curve is still improving, so it receives a convergence budget through epoch 35 with patience 10 before another training variable is introduced.

Only if higher fidelity still leaves a precision-recall failure on small lesions, run one bounded loss ablation based on [Focal Tversky loss](https://arxiv.org/abs/1810.07842), which was proposed specifically for class-imbalanced small lesion structures. It must use a predeclared parameter setting and the same full validation protocol; it is not combined with another new change in the same comparison.

## Gate B — semantic seed quality

Train a binary tumor-vs-normal DenseNet121 on the clean manifest. This removes unnecessary ten-way tumor-type confusion for a binary segmentation endpoint and avoids consuming the one subtype conflict resolved with polygon metadata. Evaluate classifier recall/F1 and CAM localization on validation. Compare predicted-class and ground-truth image-label protocols explicitly; the latter is allowed only for generating training pseudo labels from supplied weak binary image labels.

Status: classifier training is complete and promoted to Gate C. The selected epoch-11 checkpoint reached validation F1 `0.78333`, sensitivity `0.76630`, specificity `0.81283`, AUROC `0.86545`, and used no polygon/mask supervision. Checkpoint SHA-256 is `f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5`; test remains locked.

## Gate C — CAM/SAM candidate selection

On validation tumors, log CAM Dice, prompt hit rate, SAM oracle candidate Dice, selected Dice, and oracle-selection loss. Prioritize bounded ablations already supported by the code:

1. canonical contrastive LayerCAM + percentile ensemble;
2. horizontal-flip and multiscale CAM consistency;
3. global top-1 versus best-per-component selection;
4. CAM-component fallback candidate;
5. selection score / support clipping.

Choose configurations by validation tumor Dice and normal false-positive behavior, never by test feedback.

Status: the first predeclared binary LayerCAM/SAM diagnostic completed. Known-image-label pseudo-mask Dice is `0.20230`; classifier-predicted end-to-end Dice is `0.17043` because 43/184 tumors are gated out. In the legal known-label protocol, mean SAM oracle candidate Dice is `0.38092`, selected Dice `0.20108`, support loss `0.02094`, and selection loss `0.15889`. Global top-1 was rejected with a paired Dice delta of `-0.02457` and fully negative 95% CI. The predeclared `prompt_hybrid` candidate was also rejected: Dice `0.11182`, paired delta `-0.09049`, 95% CI `[-0.12424, -0.05586]`; oracle quality was unchanged but selection loss rose to `0.24956`. The original best-per-component `coverage_mass_sam` selector is retained. Horizontal-flip CAM TTA is promoted after raising Dice to `0.23434`, paired delta `+0.03204`, 95% CI `[+0.00854, +0.05739]`, with gains in all lesion-size subgroups. Adding raw CAM components was rejected at Dice `0.23066`, paired delta versus TTA `-0.00368`, 95% CI `[-0.00848, +0.00057]`. Gate C is frozen.

## Gate D — final weakly supervised segmenter

Generate clean-train pseudo masks using only images and their image-level tumor labels, record a checksum-bearing pseudo-mask manifest, and train the pretrained segmentation model against those masks. Validate exclusively against clean validation polygons. Threshold is fixed from validation and stored in the frozen configuration.

Status: pseudo generation completed with manifest SHA-256 `7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6`, 2,981 verified masks, 1,463 non-empty/25 empty tumor masks, and 1,493 explicit empty normal masks. The manifest-bound ResNet18-U-Net 448 px run is active on Kaggle; test remains locked.

If the validation endpoint exceeds 0.50, freeze all inputs and evaluate the clean test once. If it does not, report the best honest validation result, pseudo-mask upper/oracle diagnostics, and remaining bottleneck without changing the metric.

## Research basis

- AdvCAM expands discriminative regions with a post-hoc anti-adversarial attribution process, making it a lower-cost fallback if standard CAM remains too sparse: https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Anti-Adversarially_Manipulated_Attributions_for_Weakly_and_Semi-Supervised_Semantic_Segmentation_CVPR_2021_paper.html
- ToCo shows why transformer CAMs can improve object completeness but also identifies token over-smoothing; its full training cost makes it a secondary option in this hardware/time budget: https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- S2C directly targets the observed CAM→SAM semantic-selection gap through SAM-derived supervision and CAM-based prompting: https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html

These papers motivate the experiments; none of their benchmark results are treated as evidence for BTXRD performance.
