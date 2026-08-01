# B1 BAS candidate descriptor design

Status: static design only; no claim, real-data fit, prediction, or validation
segmentation-GT access.

## Evidence carried forward

The collaborator G2 v3 audit physically reproduced all 371 frozen G1 choices.
The strongest observed rich-gallery diagnostic is equal percentile-rank fusion
of the G1 model score and its frozen upstream selection score:

- Dice overall/small/medium/large:
  `0.28872949/0.15772330/0.43522933/0.38687353`;
- gain over G1 raw: `+0.082703` overall;
- complete misses: `29 -> 49`.

This is useful mechanism evidence: model and upstream ranks can be
complementary. It is not treated as an independently promoted baseline because
the fusion was formulated after inspecting the same validation Stage-B result,
large Dice decreased, and all operational goals remain unmet. The B1 experiment
does not reuse the collaborator's private rich-gallery output or copy G2.

## New scientific variable

B1 trains an ImageNet-initialized ResNet-50 Background Activation Suppression
localizer using only frozen train-split binary image labels. BAS contributes a
third, class-aware positive-instance rank on the already accepted same-gallery
candidate masks. Proposal generation, masks, candidate indices, Geometry-v3
checkpoint, and upstream selection scores remain immutable.

The BAS implementation follows the official ResNet resolution path and loss:

- localization output stride 8 and classification output stride 16;
- full-image cross-entropy;
- foreground cross-entropy weight `0.5`;
- erased-background activation ratio plus area constraint `1.2`;
- per-forward current-weight background branch;
- fixed-final 100 epochs, batch 32 over T4x2, official SGD parameter-group
  multipliers from backbone learning rate `0.001`, momentum `0.9`, weight decay
  `0.0005`;
- 224-pixel resize and horizontal-flip train augmentation;
- original/aligned-horizontal-flip mean activation at validation inference.

Primary sources:

- https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Background_Activation_Suppression_for_Weakly_Supervised_Object_Localization_CVPR_2022_paper.html
- https://github.com/wpy1999/BAS
- https://raw.githubusercontent.com/wpy1999/BAS/main/CUB/Model/resnet.py
- https://raw.githubusercontent.com/wpy1999/BAS/main/CUB/train.py
- https://raw.githubusercontent.com/wpy1999/BAS/main/CUB/utils/optimizer.py

## Frozen finite comparison

Every candidate score is converted to a tie-aware within-image percentile rank.
There is no coefficient search.

1. `transferred_geometry_upstream` is the transfer control:
   `(Geometry-v3 rank + frozen upstream selection-score rank) / 2`.
2. `three_way_geometry_upstream_bas` is the sole B1 arm:
   `(Geometry-v3 rank + upstream rank + BAS activation-evidence rank) / 3`.

BAS activation evidence is the harmonic mean of activation-mass coverage and
activation purity inside a candidate after official per-image min-max map
normalization. The equal three-way mean tests whether independently trained
class-aware localization adds positive-instance evidence while retaining the
two signals that produced the collaborator's best diagnostic.

## GT-blind operational gates

Before any validation polygon or GT-derived table may be opened:

- final classifier AUROC `>=0.75`;
- sensitivity and specificity each `>=0.60` at probability `0.5`;
- at least `95%` of 184 image-label-tumor validation maps have activation range
  above `1e-4`;
- mean BAS/upstream within-bag rank correlation `<=0.80`;
- the three-way arm changes at least `5%` of selections relative to the transfer
  control;
- all 371 cache/baseline/candidate identities and physical output hashes pass;
- T4x2, split, ImageNet initialization, source, protocol and no-test safety
  contracts pass.

Failure of any gate freezes a GT-blind failure result and forbids Dice
evaluation. If all gates pass, both 371-prediction arms are physically frozen as
one pair before the evaluator imports validation masks.

## Decision rule

The three-way arm is useful only if it beats the transfer control and accepted
Geometry-v3 through predeclared paired complete-group comparisons, does not
trade away a lesion-size subgroup, and improves positive-hit/miss diagnostics.
All goal metrics are reported regardless. A point gain alone does not authorize
a rank-weight, epoch, resolution, threshold, or morphology sweep. Consumer
training and BTXRD test remain locked.
