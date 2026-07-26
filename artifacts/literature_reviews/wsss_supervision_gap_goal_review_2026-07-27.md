# Image-label-only WSSS versus fully supervised segmentation:
# a BTXRD goal-feasibility review

Date frozen: 2026-07-27

Scope: training may use only image-level labels from BTXRD. Validation masks
remain evaluation/selection data under the frozen prediction-first protocol.
The test split remains locked. Results from methods using boxes, points,
scribbles, any task-specific training mask, or test-time oracle filtering are
not treated as evidence for the attainable image-label-only target.

## Question

Is a Dice gap of 0.10 from the frozen fully supervised BTXRD reference a
reasonable hard goal for every lesion-size subgroup, and what operational
targets remain ambitious but feasible relative to the current WSL consumer?

## Evidence reviewed

### Surveys

1. Chen and Sun, *Weakly-supervised Semantic Segmentation with Image-level
   Labels: From Traditional Models to Foundation Models*, ACM Computing
   Surveys 57(5), 2025.
   https://doi.org/10.1145/3707447

   The survey identifies image-level supervision as the most challenging WSSS
   form and separates single-stage from two-stage/pseudo-mask pipelines. Its
   central message is that mask quality, refinement, and target-domain
   annotation conventions remain limiting factors; foundation models do not
   remove the need for target-domain adaptation and protocol control.

2. Peng and Wang, *Medical Image Segmentation with Limited Supervision: A
   Review of Deep Network Models*, 2021.
   https://arxiv.org/abs/2103.00429

   The review treats limited supervision as intrinsically difficult and notes
   that specialized learning strategies are required because dense medical
   labels cannot simply be recovered from weak labels without additional
   inductive bias.

3. Rao et al., *A Review of Non-Fully Supervised Deep Learning for Medical
   Image Segmentation*, Information 16(6), 2025.
   https://doi.org/10.3390/info16060433

   This medical review separates pure image-label methods from stronger weak
   labels such as points, boxes, and scribbles. It explicitly characterizes
   image-level CAM/MIL supervision as the lowest-cost but commonly less
   accurate setting.

### Primary studies with directly useful quantitative evidence

1. Liu et al., *Weakly-supervised High-resolution Segmentation of Mammography
   Images for Breast Cancer Diagnosis*, MIDL/PMLR 2021.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8791642/

   This is the closest modality/problem analogue in the reviewed literature:
   high-resolution radiographs, small cancer lesions, image-level training
   labels, Dice evaluation, and a fully supervised U-Net upper bound on the
   same study.

   - Malignant Dice: GLAM 0.390 versus fully supervised U-Net 0.504; gap 0.114.
   - Benign Dice: GLAM 0.335 versus fully supervised U-Net 0.412; gap 0.077.
   - The study trained/evaluated on a dataset containing more than one million
     mammograms and selected models using validation segmentation performance.

   Interpretation: a gap near 0.10 is possible, but this result is supported by
   orders of magnitude more images than BTXRD. It is evidence for a stretch
   target, not a universal minimum.

2. Yang et al., *Anomaly-guided weakly supervised lesion segmentation on
   retinal OCT images*, Medical Image Analysis 2024.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11016376/

   This image-label-only medical WSSS study reports a matched downstream
   DeepLabV3+ comparison on RESC:

   - pseudo-label-trained ResNet-101 DeepLabV3+: mIoU 0.5387;
   - pixel-mask-trained ResNet-101 DeepLabV3+ upper bound: mIoU 0.7165;
   - absolute mIoU gap: 0.1778.

   The metric is mIoU, not the BTXRD per-image tumor Dice, so the number must
   not be converted or transplanted directly. It nevertheless shows that an
   approximately 0.15--0.18 gap can remain in a strong medical image-label
   pipeline even with 8,960 RESC training images.

3. Zhang et al., *Frozen CLIP: A Strong Backbone for Weakly Supervised
   Semantic Segmentation (WeCLIP)*, CVPR 2024.
   https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html

   WeCLIP reaches 76.4%/77.2% mIoU on PASCAL VOC val/test and 47.1% on COCO
   val. Its dynamic refinement module improves the no-CRF VOC ablation from
   68.7% to 74.9% mIoU (+6.2 points). This supports the selected mechanism of
   a frozen strong encoder, trainable spatial decoder, and learned affinity
   refinement. Transfer is limited because CLIP was pretrained on 400 million
   image-text pairs, the benchmarks are natural RGB images, language prompts
   are used, and the reported metric is class mIoU rather than lesion Dice.

4. Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
   Segmentation*, WACV 2024.
   https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

   The paper demonstrates that existing image-label WSSS methods struggle to
   capture small instances and that aggregate mIoU can conceal those failures.
   This directly supports retaining an explicit small-lesion gate rather than
   relying only on BTXRD overall Dice.

5. Zhang et al., *INSIGHT: Explainable Weakly-Supervised Medical Image
   Analysis*, MLHC/PMLR 2025.
   https://proceedings.mlr.press/v298/zhang25a.html

   INSIGHT shows that local detection, context suppression, and SmoothMax can
   produce useful weak heatmaps, but its outcomes vary substantially by
   modality: Dice 42.7% on MosMed and 74.6% on CAMELYON16. Its MosMed fully
   supervised comparator was trained on an external dataset, so the 42.7%
   versus 40.5% values are not a matched weak-versus-full gap. This is
   mechanism evidence, not threshold evidence.

6. Yoo et al., *Deep superpixel generation and clustering for weakly
   supervised segmentation of brain tumors in MR images*, BMC Medical
   Imaging 2024.
   https://pubmed.ncbi.nlm.nih.gov/39695438/

   Binary image-label training reaches mean Dice 0.745 on an external BraTS
   2023 cohort. This demonstrates that strong image-level tumor segmentation
   is possible when MRI appearance and learned superpixel structure provide
   favorable boundary cues. It does not report a matched fully supervised
   comparator under the same protocol and is not a radiograph result.

7. Yao et al., *A Radiograph Dataset for the Classification, Localization,
   and Segmentation of Primary Bone Tumors*, Scientific Data 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11739492/

   BTXRD contains 3,746 images, including 1,867 tumor radiographs. The released
   fully supervised YOLOv8s-seg baseline reports mask mAP@0.5, not mean
   per-image Dice, so it cannot be used to set the present Dice target. It
   confirms the dataset's heterogeneity and the absence of a published,
   protocol-matched image-label-only Dice benchmark.

## BTXRD evidence that controls the decision

Frozen validation population:

- 371 images: 184 tumor and 187 normal;
- lesion-size subgroups: small 94, medium 72, large 18;
- mean per-image tumor Dice includes complete misses;
- subgroup size is computed from validation GT only after prediction freeze;
- test has not been evaluated.

Frozen fully supervised reference:

| Population | Dice |
|---|---:|
| Overall | 0.49513170 |
| Small | 0.32895493 |
| Medium | 0.66244178 |
| Large | 0.69370336 |

Current image-label-only WSL consumer:

| Population | Dice |
|---|---:|
| Overall | 0.23001987 |
| Small | 0.07100142 |
| Medium | 0.41340304 |
| Large | 0.32691688 |

The former `FS - 0.10` goal requires Dice
0.39513170/0.22895493/0.56244178/0.59370336 for
overall/small/medium/large. Relative to the current consumer, that is an
absolute gain of +0.16511/+0.15795/+0.14904/+0.26679. The large requirement
is particularly brittle: only 18 validation images are present, and the
four-run fully supervised sensitivity audit spans 0.60722--0.74827, a range
of 0.14105, despite the same recorded training contract.

The geometry-corrected RAD-DINO probes also show that ranking/localization
quality does not yet translate into usable mask shape. Corrected INSIGHT
single-scale has overall/small p90 Dice only 0.02864/0.00760 and small
argmax hit 0/94. No reviewed mechanism has yet produced internal evidence
that a +0.267 large-subgroup consumer gain is an appropriate minimum gate.

## Decision

The 0.10 weak-versus-full gap is scientifically plausible as a stretch goal,
but it is not a defensible hard minimum for every BTXRD subgroup at the
present data scale and evidence level.

The revised operational target uses:

- a 0.15 gap from the frozen fully supervised anchor for small and medium;
- a 0.20 gap for large because n=18 and the supervised reference itself has
  a 0.14105 four-run range;
- an overall target equal to the subgroup-count-weighted operational floor.

| Population | Revised operational Dice | Gain over current | Fraction of frozen FS |
|---|---:|---:|---:|
| Small | 0.17895493 | +0.10795352 | 54.4% |
| Medium | 0.51244178 | +0.09903874 | 77.4% |
| Large | 0.49370336 | +0.16678648 | 71.2% |
| Overall | 0.34024039 | +0.11022052 | 68.7% |

The exact overall floor is

`(94*0.17895493248574226 + 72*0.5124417783635557 +
18*0.4937033565801355) / 184 = 0.340240391925425`.

These thresholds remain deliberately demanding: every subgroup must improve,
overall Dice must rise by 0.11022 absolute (47.9% relative to the current
consumer), and small Dice must rise by 0.10795 absolute (152.0% relative).
They are more consistent with the 0.114 Dice gap seen in million-scale
mammography, the 0.178 mIoU gap seen in medical OCT, the known small-object
penalty, and BTXRD's subgroup uncertainty.

The former `FS - 0.10` thresholds remain frozen as a stretch tier:

| Population | Stretch Dice |
|---|---:|
| Small | 0.22895493 |
| Medium | 0.56244178 |
| Large | 0.59370336 |
| Overall | 0.39513170 |

No metric, cohort, subgroup definition, complete-miss policy, or test lock is
changed by this decision. Meeting the operational tier is not evidence of
clinical equivalence to fully supervised training; it is the next
protocol-valid research success gate.
