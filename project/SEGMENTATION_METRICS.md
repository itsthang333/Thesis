# BTXRD segmentation loss and metric policy

This policy deliberately separates three quantities that must not be mixed:

1. **Training objective:** computed from noisy pseudo masks and image-level
   tumor/normal labels only in the WSSS run.
2. **Checkpoint selection:** computed on validation pseudo masks only in the
   WSSS run. Polygon segmentation ground truth is not loaded.
3. **Locked audit:** computed after training against polygon ground truth by
   `evaluate_unet.py`. These values are never fed back into training,
   threshold selection, early stopping, or model selection.

## WSSS training loss

For a tumor image with a non-empty pseudo mask, the supervised loss is

`L_tumor = alpha * weighted_BCE + (1 - alpha) * soft_Dice_loss`.

For a known normal image, only all-background BCE is used. A known tumor image
whose pseudo mask is empty is treated as **unknown**, not as reliable
background. Tumor-image BCE and normal-image BCE are averaged as two groups so
the more numerous/easier group cannot dominate solely through sample count.

The default notebook uses `sqrt(background_pixels / foreground_pixels)` as
positive BCE weight. The former full ratio was about 75 on the audited run and
overweighted noisy positive pixels. A one-pixel soft boundary band receives
weight 0.25; unlike the former three-pixel hard ignore band, it cannot delete a
small lesion from supervision.

## Checkpoint metric without segmentation GT

For every non-empty tumor pseudo reference, metrics are computed per image and
then macro-averaged:

`Dice = 2 TP / (2 TP + FP + FN)`

`IoU = TP / (TP + FP + FN)`

`Precision = TP / (TP + FP)` and `Recall = TP / (TP + FN)`.

No additive epsilon is placed in the numerator: an empty prediction for a
non-empty tumor reference scores zero. Empty normal references are never mixed
into Dice/IoU. They are reported separately as:

- image specificity: fraction of normal images with no predicted tumor pixel;
- false-positive pixel rate: predicted tumor pixels / all normal-image pixels.

The threshold is swept on pseudo validation only. Checkpoint score is the
harmonic mean of macro tumor pseudo-Dice and normal-image specificity:

`score = 2 * Dice_tumor * Specificity_normal / (Dice_tumor + Specificity_normal)`.

Thus both degenerate predictors score zero: all-background has zero tumor Dice,
and all-foreground has zero normal specificity. Early stopping and LR plateau
monitor this score. The selected threshold is stored in the checkpoint.

## Locked polygon-GT audit

`evaluate_unet.py` loads the threshold stored in the checkpoint unless an
explicit CLI override is supplied. Tumor/normal grouping comes from BTXRD's
image-level label, not from the existence of a polygon. Tumor images missing a
usable polygon are counted and excluded from overlap means.

The audit reports macro tumor Dice, IoU, precision and recall; micro tumor Dice;
overlap-hit rate; Dice >= 0.1 and Dice >= 0.4 rates; normal image specificity;
and normal false-positive pixel fraction. Macro tumor Dice is the primary
segmentation result. The complementary metrics expose size imbalance and the
failure modes hidden by any single aggregate.
