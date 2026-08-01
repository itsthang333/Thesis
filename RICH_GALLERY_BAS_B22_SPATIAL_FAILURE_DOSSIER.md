# Rich-gallery BAS-B2.2 spatial failure dossier

## Frozen endpoint

The B2.2 maps, five candidate-score variants and all 1,855 choices were frozen
before validation polygons. Stage B reproduced the G1+upstream baseline on
371/371 validation images and evaluated 184 tumors split 94/72/18. Test was
not opened.

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1+upstream baseline | 0.288729 | 0.216839 | 0.157723 | 0.435229 | 0.386874 | 49 |
| B2.2 only | 0.061326 | 0.033995 | 0.007550 | 0.081066 | 0.263200 | 8 |
| G1+B2.2 | 0.107847 | 0.068888 | 0.011793 | 0.170168 | 0.360180 | 10 |
| upstream+B2.2 | 0.179844 | 0.124628 | 0.019122 | 0.297480 | 0.548631 | 38 |
| G1+upstream+B2.2 | 0.191726 | 0.137812 | 0.028316 | 0.318133 | 0.539464 | 25 |

The primary delta is `-0.097003` with complete-group bootstrap
CI95 `[-0.128452, -0.047141]`. B2.2 is
therefore decisively worse overall, not a noisy tie.

## What B2.2 changes

B2.2 changes 80.86% of tumor selections. It wins on 53 images, ties on 47 and
loses on 84. It recovers only 3 baseline misses while turning 27 previous hits
into misses. Positive Dice mass is `+6.1840`; negative mass is `-24.0327`.

The effect is scale-opposite:

- small: `-0.129408`;
- medium: `-0.117096`;
- large: `+0.152591`.

The large gain is real, but the candidate extent reveals why it cannot be
applied globally. Median selected/GT area changes from
`14.60/1.10/0.38` in the baseline to `158.24/4.93/1.43` for
small/medium/large. B2.2 is a strong expansion prior: it roughly repairs the
baseline's large-lesion under-extent while catastrophically amplifying the
existing small-lesion over-extent.

## No usable label-safe router was found

Choosing the better of baseline and B2.2 per image using GT would reach Dice
`0.322338` (`+0.033609`), so B2.2 contains limited
complementary choices. That is an oracle diagnostic, not an algorithm.

A deterministic five-fold group-separated ridge diagnostic used only
observable frozen quantities: baseline/B2.2 area and border, image tumor
probability, activation mean/std, candidate-area dependence, candidate count
and source identities. Its out-of-fold routed Dice is
`0.287786` (`-0.000944`), with rank correlation
`0.182482` and non-tie sign accuracy
`0.599`. It improves large lesions but harms both
small and medium, reproducing the overall baseline rather than exceeding it.
This diagnostic is itself validation-GT-trained and non-promotable; its failure
shows that the observed B2.2 benefit cannot be isolated reliably from current
label-safe metadata.

## Exact bottleneck update

The gallery remains adequate: oracle Dice is 0.528298 and truncation regret is
0.000396. B2.2 increases selector regret from 0.239569 to 0.336572. Its
within-selected-source regret rises from 0.168376 to 0.230754 and cross-source
regret from 0.070796 to 0.105422. Thus it worsens both major selector terms.

The missing observable is not another global area, source, confidence, CAM or
anatomy score. It is **candidate-conditioned tumor identity plus signed extent
calibration**:

1. Does the candidate contain tumor-specific content rather than mostly normal
   bone (critical for small lesions)?
2. Does removing the candidate erase tumor evidence, and does keeping only the
   candidate preserve it (critical for medium/large extent)?
3. Is the evidence stronger inside the candidate than in a matched local ring,
   and absent on candidates from train-normal images?

## Research decision

1. Retire BAS-B2.2 and all BAS epoch/weight/threshold/seed routing sweeps.
2. Preserve G1+upstream fixed fusion at Dice 0.2887294867.
3. The next bounded mechanism must operate per candidate, not per image:
   inside-versus-local-ring evidence with exact candidate masking, train-normal
   hard negatives, and a zero-initialized residual on the immutable baseline.
4. Before any full selector training, require a cheap matched diagnostic to
   reduce within-source oracle rank without increasing candidate-area
   dependence; large-only gains or GT-size routing are insufficient.
5. Keep test locked.
