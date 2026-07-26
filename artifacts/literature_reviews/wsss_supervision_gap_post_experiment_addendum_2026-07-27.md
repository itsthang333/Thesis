# Post-experiment addendum: attainable image-label WSSS gaps

Date: 2026-07-27

This addendum was written only after the RAD-DINO geodesic seed-expansion
probe had reached terminal status and its predictions had been independently
audited. It does not mutate the frozen goal protocol or use BTXRD test data.

## Additional primary evidence

### Medical CAM ensembles

Ostrowski et al., *Exploring Weakly Supervised Semantic Segmentation
Ensembles for Medical Imaging Systems*:

https://arxiv.org/abs/2303.07896

The paper uses image-level labels and reports matched fully supervised
references:

| Dataset | Image-label WSSS Dice | Fully supervised Dice | Gap |
|---|---:|---:|---:|
| BraTS | 0.703 | 0.818 | 0.115 |
| Prostate Decathlon | 0.793 | 0.868 | 0.075 |

The authors also report that medical CAMs struggle because of limited sample
size and image complexity. On BraTS, an empty-mask predictor scores 0.646,
SEAM scores 0.561, and WSS-CMER scores 0.597. This is a warning that a high
aggregate score can partly reflect empty images and that generic natural-image
WSSS mechanisms may transfer poorly. BTXRD therefore continues to report
positive-image Dice with complete tumor misses included rather than relying on
an aggregate containing normal-image true negatives.

### Morphology-guided CAM and SAM

Yue et al., *Morphology-Enhanced CAM-Guided SAM for Weakly Supervised Breast
Lesion Segmentation*:

https://arxiv.org/abs/2311.11176

On BUSI, the proposed image-label pipeline reports Dice 0.7439 versus 0.7831
for a fully supervised U-Net, a gap of 0.0392. This is strong evidence that a
gap below 0.10 is possible when modality-specific morphology provides useful
contours and a foundation model can refine a localized lesion. It is not a
generic guarantee: the method encodes ultrasound-specific layered anatomy,
lesion aspect-ratio priors, clustering thresholds, CAM thresholds, SAM, and
post-processing. Its AffinityNet comparison scores only 0.1614 Dice, showing
how sharply performance can depend on domain-matched priors.

### Frozen foundation-model decoder and dynamic refinement

Zhang et al., *Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic
Segmentation (WeCLIP)*:

https://arxiv.org/abs/2406.11189

WeCLIP uses a frozen CLIP encoder, a lightweight trainable spatial decoder,
intermediate features, decoder affinity, and dynamic pseudo-label refinement.
On PASCAL VOC, the decoder-only ablation scores 68.7 mIoU and the refinement
module raises it to 74.9 (+6.2 points); the full weakly supervised result with
CRF is 76.4, while the paper's fully supervised frozen-backbone decoder scores
81.6. This supports the mechanism family chosen for the next BTXRD probe, but
the approximately 0.052 same-paper gap cannot be transplanted as a BTXRD Dice
target because CLIP uses web-scale image-text pretraining, language prompts,
natural RGB objects, and a different metric.

### Survey synthesis

Chen and Sun, *Weakly-Supervised Semantic Segmentation with Image-Level
Labels: from Traditional Models to Foundation Models*:

https://arxiv.org/abs/2310.13026

The survey calls image-level supervision the most challenging WSSS form. It
attributes the central error to CAMs highlighting discriminative parts rather
than complete object extent and organizes remedies into pixel-wise,
image-wise, cross-image, and external/foundation-model mechanisms.

## Decision after the geodesic failure

The literature supports two conclusions simultaneously:

1. A Dice gap of 0.10 or less is possible in a favorable task with enough data
   or strong domain priors.
2. A uniform `fully supervised - 0.10` hard gate is not justified for BTXRD
   today, especially for small lesions and the 18-image large subgroup.

The failed geodesic probe reinforces the second conclusion. Its median
pixel-wise correlation with the frozen source affinity map is 0.9944 and its
median top-10-percent support Jaccard is 0.9486; it therefore does not create
the representation-to-shape change needed to approach the consumer goal.

The frozen operational minimum remains:

| Population | Dice target | Gap from frozen full | Gain over current |
|---|---:|---:|---:|
| Overall | 0.34024039 | 0.15489131 | 0.11022052 |
| Small | 0.17895493 | 0.15000000 | 0.10795352 |
| Medium | 0.51244178 | 0.15000000 | 0.09903874 |
| Large | 0.49370336 | 0.20000000 | 0.16678648 |

All four conditions are required. The frozen `full - 0.10` thresholds remain
a stretch tier. The validation cohort, positive-image mean Dice, subgroup
definitions, complete-miss accounting, image-label-only training restriction,
and test lock are unchanged.
