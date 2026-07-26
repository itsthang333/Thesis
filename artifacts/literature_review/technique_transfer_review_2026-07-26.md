# Technique transfer review for BTXRD image-label-only WSSS

Date: 2026-07-26

## Research question

This review asks a broader question than "which checkpoint can be plugged into
the pipeline?": what mechanism did each paper use to solve under-activation,
over-expansion, small lesions, ambiguous boundaries, domain shift, or noisy
pseudo-labels, and can that mechanism be adapted to BTXRD under image-level
supervision?

The transfer does not require the source paper to use radiographs, bone tumors,
the same architecture, or a public checkpoint. It does require that the BTXRD
adaptation never consume BTXRD masks, boxes, points, true tumor area, validation
subgroup identity, or test feedback during learning or prediction.

## What the project evidence says must be solved

The literature is filtered through the completed experiments in
`RESEARCH_LOG.md`:

- Small tumors below 1% area are the dominant bottleneck.
- Higher resolution and adversarial CAM expansion increased small-lesion
  support recall and SAM oracle quality, but the no-GT selector chose masks
  that were too broad.
- The BiomedCLIP gallery again improved the raw oracle in every subgroup, while
  direct/scalar selectors failed.
- The pseudo-mask consumer fitted its training targets strongly but
  generalized poorly. Therefore pseudo-label noise and consumer robustness are
  a second bottleneck, not merely a mask-generation problem.
- Post-hoc AdvCAM variants, scalar candidate gates, direct BiomedCLIP
  replacement, generic grid proposals, and the tested S2C SSC/CPM adaptations
  should not be repeated under new names.

This points to three useful classes of technique:

1. learn a spatial representation whose scores are meaningful at small scale;
2. expand/refine seeds using image structure and uncertainty rather than a
   scalar candidate score;
3. prevent the final segmenter from memorizing noisy pseudo-label pixels.

## Technique 1 - masked reconstruction anomaly maps

### What the papers do

SKELEX uses a two-stage masked autoencoder. It initializes a ViT-Large MAE from
ImageNet-1K, then self-supervises it on 1,296,540 unlabeled musculoskeletal
radiographs. The radiograph-specific changes are important:

- 75% random patch masking;
- normalized pixel loss disabled because large black X-ray backgrounds would
  distort it;
- random resized crop over 20%-100% of image area and horizontal flip;
- no manual label or anatomical supervision in pretraining.

For zero-shot localization, it reconstructs each image under ten independent
random masks, subtracts each reconstruction from the original, and averages the
ten pixel-wise error maps. When a subtle tumor remains partly visible, it may be
reconstructed; when the lesion is completely masked, the model tends to
reconstruct normal-appearing anatomy, causing the lesion to appear in the error
map. The paper explicitly demonstrates this mechanism on BTXRD as well as
fracture and osteoarthritis data.

### Transfer to this project

The mechanism can be tested without SKELEX weights:

1. initialize an open ViT-MAE from ImageNet;
2. adapt only on the clean BTXRD training images, preferably with a controlled
   comparison of all-train versus normal-image-only reconstruction;
3. generate an ensemble reconstruction-error map with ten frozen mask patterns;
4. remove background/acquisition error using a no-GT radiograph foreground
   mask and robust per-image normalization;
5. use the error map as an independent seed/proposal source, not as an
   automatically trusted final mask;
6. freeze all maps before validation GT is loaded.

The normal-only version is still weakly supervised: it uses only the image-level
normal/tumor label to choose reconstruction training images. It may sharpen
tumor residuals, but anatomical heterogeneity can become false anomaly.
Therefore a no-label anatomy cluster or view-conditioned normal bank should be
audited. True tumor size must never choose a branch.

### Why it may help small tumors

A classifier is rewarded once it finds any discriminative tumor patch, which
causes incomplete CAMs. Reconstruction instead asks whether a local patch is
predictable from surrounding bone structure. A tiny lesion can create a strong
local residual even when it contributes little to a global class score.

### Risks and ablation

- Black borders, text markers, implants, fractures, and acquisition artifacts
  may dominate reconstruction error.
- Adapting on tumor images may teach the MAE to reconstruct tumors; adapting
  only on normals may confuse anatomy with pathology.
- ViT-Large pretraining is heavy; a ViT-Base feasibility run should establish
  whether the signal exists before scaling.

Required ablation: image error vs feature error; one vs ten masks; all-train vs
normal-only; global vs anatomy-conditioned normalization. The first gate is
small/medium/large seed recall and oracle proposal Dice, not final Dice.

Primary source:

- https://arxiv.org/abs/2602.03076
- https://doi.org/10.1038/s41746-026-02826-9

## Technique 2 - uncertainty-weighted multi-resolution CAM

### What the papers do

UM-CAM attacks low-resolution, incomplete medical CAMs at three levels:

1. fuse CAMs from several resolutions, weighting spatial evidence by
   uncertainty rather than averaging every view equally;
2. expand reliable foreground/background seeds with exponential geodesic
   distance, so propagation follows image similarity and stops at boundaries;
3. train the final segmenter with Random-View Consensus (RVC), suppressing
   unreliable pixels and enforcing agreement across random transformed views.

It was evaluated for 2D fetal brain and extended to 3D brain-tumor
segmentation, both with image-level supervision.

### Transfer to this project

This is directly compatible with existing infrastructure:

- generate aligned LayerCAM or dense-token maps at fixed scales such as
  320/448/640 plus horizontal flip;
- estimate pixel reliability from cross-view mean/variance or entropy;
- retain high-confidence tumor seeds and high-confidence background seeds;
- run grayscale geodesic propagation on the original radiograph, optionally
  with gradient magnitude so cortical/lesion boundaries are costly to cross;
- feed soft confidence and ignore regions to the pseudo-mask consumer;
- enforce prediction consistency only after inverting the known random
  transform.

This differs from the rejected scalar selectors: uncertainty is spatial and
geodesic expansion uses local image structure. It also differs from rejected
AdvCAM: it does not indiscriminately enlarge the high-score region.

### Risks and ablation

X-rays superimpose anatomy, so raw intensity similarity is not always a valid
boundary cue. Test UM-CAM fusion, geodesic expansion, and RVC as separate
single-variable stages. A valid improvement must preserve small-lesion
precision, not merely increase support recall.

Primary sources:

- https://arxiv.org/abs/2306.11490
- https://github.com/HiLab-git/UM-CAM
- https://doi.org/10.1016/j.patcog.2024.111204

## Technique 3 - progressive confidence expansion with prototypes

### What the papers do

Progressive Confidence Region Expansion (PCRE) addresses ViT CAM
over-expansion. It starts from a small, highest-confidence region and expands
gradually instead of propagating all affinities at once. Dataset-level class
prototypes are compared with patch features to supervise the expansion.

MCTformer extracts a complementary mechanism from transformer attention:
class-to-patch attention provides class-specific localization and
patch-to-patch attention provides pairwise affinity for refinement.

### Transfer to this project

For the binary BTXRD task:

- derive a per-image tumor prototype only from frozen high-confidence positive
  seeds;
- derive robust background prototypes from normal training radiographs and
  low-score areas of positive images;
- expand in small fixed iterations using feature similarity plus local
  connectivity;
- stop expansion when confidence/feature similarity falls, not when a
  validation-fitted area is reached;
- compare CNN LayerCAM features with RAD-DINO/DINO patch features.

This can replace candidate-level scalar ranking with pixel/patch-level evidence.
It is especially attractive after the current graph selector because both use
connectivity, but prototype expansion can recover the correct extent within a
component rather than choose one complete SAM mask.

### Risks and ablation

A single tumor prototype may be dominated by medium/large lesions and miss
heterogeneous small tumors. Keep per-image and dataset prototypes as distinct
ablations; do not route by true size. ViT affinities can over-smooth into
background, the exact failure PCRE was designed to control.

Primary sources:

- https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Weakly_Supervised_Semantic_Segmentation_via_Progressive_Confidence_Region_Expansion_CVPR_2025_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Class_Token_Transformer_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.html

## Technique 4 - token contrast instead of ordinary CAM

### What the paper does

ToCo observes that final ViT patch tokens become over-smoothed while
intermediate layers preserve semantic diversity. Patch Token Contrast derives
pseudo-relations from intermediate tokens and uses them to supervise final
tokens. Class Token Contrast then makes uncertain local crops consistent with
the global object through contrastive class-token learning.

### Transfer to this project

Use an X-ray/dense transformer encoder and:

- build positive/negative patch relations only from confident, view-consistent
  seeds;
- use intermediate-layer similarity to prevent final tokens from becoming a
  uniform broad lesion region;
- contrast uncertain high-resolution crops with the global tumor image
  embedding;
- treat normal images as strong negative bags.

This is a learned spatial representation change, which prior evidence favors
over another post-hoc threshold. It could be added to a RAD-DINO or ViT-MAE
branch without copying a full natural-image WSSS pipeline.

Risk: local crops from a positive image do not necessarily contain a tiny
tumor. They must not inherit a positive label blindly. Only confident seed
overlap or teacher consistency may designate a positive crop; otherwise the
crop is ignored, not labeled negative.

Primary source:

- https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html

## Technique 5 - direct dense multi-instance learning

### What the papers do

Multi-resolution medical localization treats every pixel/patch as an instance
and aggregates dense probabilities into an image score. Learnable
lower-bounded Log-Sum-Exp pooling provides a sharpness prior while allowing
abnormalities of different sizes. Probabilistic-CAM pooling similarly avoids
the rigid behavior of global average or global max pooling. WSUnet trains a
U-Net dense output using only global radiological tumor labels.

### Transfer to this project

Train a dense U-Net/HRNet/FPN head with:

- image-level BCE after learnable LSE or probabilistic pooling;
- normal-image dense suppression;
- high-resolution multi-scale features;
- cross-view consistency;
- an explicit but predeclared sparsity/total-variation prior to avoid
  whole-image activation;
- uncertainty masks rather than hard pseudo-labeling every pixel.

This removes the discrete SAM-mask selector entirely. It should be treated as
an independent architecture branch, not mixed immediately with all proposal
techniques.

Risks are trivial all-zero maps, single-pixel shortcuts under max pooling, and
whole-image maps under average pooling. Report image classification, positive
map area, connected components, normal false-positive area, and localization
metrics together.

Primary sources:

- https://arxiv.org/abs/1803.07703
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10657919/
- https://github.com/jfhealthcare/Chexpert

## Technique 6 - learned affinity and structure-aware boundary recovery

### What the papers do

The image-label-only chest X-ray pipeline of Viniavskyi et al. first regularizes
a classifier, then trains an Inter-pixel Relation Network from confident CAM
seeds, performs affinity propagation, and finally trains a segmenter. Their
ablation improves substantially from raw CAM to IRNet-refined masks.

UM-CAM's geodesic seed expansion is the non-learned analogue. Recent CLIP WSSS
also combines semantic patch distributions with DINO structural knowledge to
add boundary supervision.

### Transfer to this project

Two staged options:

- cheap: geodesic/random-walk propagation over grayscale gradient and dense
  feature similarity;
- learned: train pairwise same/different affinity only on high-confidence
  foreground/background pairs, ignore ambiguous pairs, then propagate.

Use bone-aware edge channels such as gradient magnitude only as input
structure, never as a hand-tuned GT surrogate. This stage is useful only when
seed recall is adequate; it cannot recover a completely missed tumor.

Primary sources:

- https://arxiv.org/abs/2007.00748
- https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Leveraging_Class_Distributions_in_CLIP_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2026_paper.html

## Technique 7 - noise-robust pseudo-mask consumer

### What the papers do

Medical self-training methods distinguish uncertainty at multiple levels:

- checkpoint disagreement estimates sample difficulty;
- dual decoders with different upsampling paths estimate per-pixel uncertainty;
- easy-to-hard curricula add reliable images first;
- Random-View Consensus ignores pixels that are unstable under transforms;
- cross-supervision uses reliable pseudo-labels for segmentation and lets
  unreliable pixels participate only in contrastive representation learning.

### Transfer to this project

The current pseudo consumer reached high train Dice but poor validation
localization, so a noise-robust consumer is justified:

1. preserve a soft pseudo probability and source/view agreement map;
2. supervise hard foreground/background only above frozen confidence bounds;
3. ignore ambiguous pixels for BCE/Dice but use transform consistency there;
4. use two decoder heads or EMA teacher disagreement as uncertainty;
5. add samples easy-to-hard based on no-GT agreement, not validation Dice;
6. keep checkpoint selection on the locked image-level/consumer validation
   contract, never pseudo-mask training fit alone.

This should be tested after or jointly predeclared with a stronger seed source.
It cannot turn systematically misplaced masks into correct tumors, but it can
prevent memorization of uncertain edges and false components.

Primary sources:

- https://doi.org/10.1016/j.patcog.2024.111204
- https://arxiv.org/abs/2304.04441
- https://link.springer.com/article/10.1007/s42979-026-05041-1

## Technique 8 - adversarial classifier/reconstructor decomposition

### What the paper does

ACR treats a CAM as an image decomposition. If one predicted segment can be
reconstructed easily from the other, the split is considered imprecise.
A classifier generates CAM segments while a reconstructor adversarially
measures cross-segment inferability. Alternating optimization discourages both
incomplete object coverage and irrelevant activation.

### Transfer to this project

For grayscale radiographs, train the classifier and a lightweight masked
reconstructor so that predicted tumor and background regions carry distinct,
non-inferable information. Combine this with normal images, where the tumor
segment should be empty. This is conceptually related to reconstruction anomaly
maps but uses reconstruction as a training constraint on the localizer.

Risk: bone texture is highly spatially correlated, so both sides may remain
inferable; the adversarial objective is expensive and unstable. It is a later
experiment after the simpler MAE error-map feasibility test.

Primary source:

- https://openaccess.thecvf.com/content/CVPR2023/html/Kweon_Weakly_Supervised_Semantic_Segmentation_via_Adversarial_Learning_of_Classifier_and_CVPR_2023_paper.html

## Technique 9 - style/content decoupling for multi-center X-rays

### What the paper does

D-CAM separates Fourier amplitude (treated as style) from phase (treated as
content), applies instance normalization to obtain a domain-invariant amplitude,
then recombines it with content phase before Grad-CAM generation. The goal is
stable pseudo-labels under unseen hospital/style shifts.

### Transfer to this project

Use acquisition/style perturbations during classifier training or CAM
generation:

- Fourier amplitude mixing between same-label training images;
- phase-preserving intensity/contrast changes;
- consistency between original and style-perturbed dense maps;
- optional instance-normalized feature branch.

This may reduce center/scanner shortcuts, but it does not directly solve small
lesion selection. It becomes high priority only if per-center or acquisition
audit shows localization drift. Center information may be used for reporting,
not for validation-fitted routing.

Primary source:

- https://papers.miccai.org/miccai-2025/paper/0830_paper.pdf

## Technique 10 - normal-patch memory and synthetic anomaly learning

This family is important because it does not require the source task to be
bone-tumor segmentation. Its transferable assumption is simply that a local
pathology patch should differ from the distribution of normal local patches.
That is compatible with the project contract: clean-train normal images can be
identified from image-level labels, and no tumor mask is needed.

### PatchCore: compare local features with normal memory

PatchCore keeps mid-level patch embeddings from normal training images and
scores a query patch by distance to its nearest nominal patch. Mid-level
features retain more spatial detail than a classifier's final feature map.
Greedy coreset subsampling makes a broad normal memory feasible without keeping
every patch.

A direct BTXRD transfer should not use one unconditioned memory bank blindly:
normal bones, projections and acquisition styles are highly heterogeneous, so
anatomy differences could dominate tumor anomaly. A safer image-only design is:

1. extract a global embedding and multi-level patch grid from every normal
   clean-train radiograph;
2. retrieve the nearest normal images to each query using only global visual
   embeddings;
3. calculate local patch distance against this context-conditioned normal
   memory, with the fixed global coreset as a fallback;
4. aggregate aligned distances across two feature depths/resolutions;
5. freeze the continuous anomaly maps before GT evaluation.

No anatomy/view metadata is required for routing. The retrieval is inferred
from image pixels and therefore remains within the image-label-only contract.
The first backbone should be the radiograph-adapted MAE from the current probe
if its representation is stable; a radiograph foundation encoder is a separate
ablation, not an automatic replacement.

Main risk: k-nearest-neighbour distance can highlight cortical edges, image
markers or rare normal anatomy. Report false-positive area on all 187 normal
validation images and inspect whether global retrieval reduces it. Do not
normalize each heatmap in a way that forces every normal image to contain an
anomaly.

Primary source:

- https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html

### Reverse distillation: learn only the normal teacher manifold

Reverse Distillation uses a frozen pretrained teacher to emit multi-scale
features. A bottleneck plus student decoder is trained on normal images to
recover those teacher features. Normal structure is reconstructed well, while
unseen anomalous regions create teacher-student discrepancies at several
scales. Unlike pixel reconstruction, feature discrepancy is less sensitive to
minor intensity noise and can retain semantic structural deviations.

For BTXRD, the teacher should be fixed and the student trained only on normal
clean-train images. The transferable comparison is pixel MAE residual versus
feature residual, using the same prediction-first validation audit. This is
particularly useful if the current MAE produces high error on exposure,
markers or bone boundaries but little tumor contrast.

Risk: a high-capacity student can generalize to anomalies and reconstruct their
features too. The bottleneck, feature depths and training duration must be
fixed before validation evaluation.

Primary source:

- https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html

### DRAEM/NSA: learn a local anomaly discriminator from synthetic masks

DRAEM does more than subtract a reconstruction. It trains a reconstructive
network to restore a synthetically corrupted normal image, then gives both the
input and reconstruction to a U-Net-like discriminator. The discriminator
learns the anomaly-specific distance function and is trained with the exact
synthetic corruption mask. Natural Synthetic Anomalies (NSA) improves the
corruption mechanism by Poisson-blending scaled patches from other normal
images, creating more natural local irregularities. Neither method needs a
real lesion mask.

The most defensible BTXRD adaptation is not to paste arbitrary colourful
textures. It should synthesize radiograph-compatible deviations:

- Poisson-blend a patch from a different normal radiograph inside the
  radiograph foreground;
- apply local contrast/density shift, blur/sharpen or elastic displacement
  within an irregular multi-scale mask;
- preserve the surrounding black background and avoid text/marker regions;
- sample many small masks so the discriminator cannot solve the task only from
  large obvious corruptions;
- optionally concatenate the frozen MAE normal reconstruction with the
  corrupted/original image, allowing the discriminator to learn where the
  residual is meaningful.

This remains weakly/self-supervised because its pixel masks describe synthetic
corruptions generated from normal images; they are not BTXRD tumor GT.
Attention-conditioned augmentation offers a medical-X-ray-tested improvement:
use self-attention to place/mask corruptions on foreground structure rather
than destroying irrelevant background.

Risks are substantial. Synthetic artifacts may be easier than true bone tumors,
causing shortcut learning; copy-paste seams may dominate; and a synthetic
segmenter can overestimate large regions. Before a consumer experiment, require
prediction-first localization on real validation tumors and false-positive
audit on normal validation images. Synthetic-mask training Dice is not evidence
of real localization.

Primary sources:

- https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7519_ECCV_2022_paper.php
- https://ojs.aaai.org/index.php/AAAI/article/view/26720

## Techniques already tested or partially tested here

| Literature mechanism | Project evidence | Decision |
|---|---|---|
| AdvCAM / adversarial expansion | Increased recall/oracle, harmed small final Dice through broad mask selection; regularized, dual-map, and split-selector variants failed gates. | Closed as a post-hoc family. Reuse only the causal lesson. |
| S2C SAM-Segment Contrasting | Final and stride-8 feature taps were audited; region partitions did not yield the required localization. | Closed for the tested SSC adaptation. |
| S2C CAM-based Prompting | Faithful bounded adaptation and Gate-C were rejected. | Do not copy the S2C pipeline verbatim or tune it on validation. |
| Generic SAM grid gallery | Large candidate set did not solve localization/selection. | Closed. |
| BiomedCLIP direct localization | Direct result failed, while raw gallery oracle improved. | Retain only as an independent proposal source. |
| Higher resolution / blind tiling | Higher resolution improved small oracle/support; blind tiling without matching training was not credible. | Use multi-resolution learning/uncertainty, not blind inference tiling alone. |
| Scalar candidate selectors | Multiple independent variants failed. | Replace with spatial uncertainty, affinity, reconstruction, or learned pixel evidence. |

## Revised priority order

The order below ranks transferable mechanisms, not named models:

1. Run the already launched bounded SKELEX-inspired MAE reconstruction-error feasibility audit,
   because it is exact-modality, demonstrated on BTXRD, independent of CAM, and
   theoretically favorable to tiny local anomalies.
2. If MAE pixel residual is weak or dominated by acquisition artifacts, test a
   normal-patch memory branch using context-conditioned PatchCore-style
   distances. It can reuse the frozen MAE/radiograph features and directly
   targets small local deviations without learning another mask selector.
3. Implement UM-CAM's uncertainty-weighted multi-resolution fusion. If its
   seeds improve, test geodesic expansion separately.
4. Add RVC/uncertainty-ignore training to the pseudo-mask consumer, addressing
   the measured train-fit/validation-generalization gap.
5. Train a dense MIL localizer with probabilistic/LSE pooling as an independent
   route that bypasses candidate selection.
6. If transformer dense features are useful but over-smoothed, add ToCo-style
   intermediate-token contrast or PCRE-style progressive prototype expansion.
7. Add learned affinity only after seed recall is high enough.
8. Evaluate Fourier style/content consistency if a center/acquisition subgroup
   audit shows domain drift.
9. Test radiograph-compatible NSA/DRAEM synthetic anomaly learning only after a
   real-tumor prediction-first gate; synthetic training accuracy is not a gate.
10. Keep reverse distillation and ACR adversarial reconstruction as higher-cost
    fallbacks when pixel residuals fail but feature-level anomaly remains
    plausible.

Each numbered item must be predeclared as a controlled experiment. Techniques
may be composed only after their individual prediction-first audits show
complementary errors; validation GT may assess a frozen output but may not
choose per-image methods, thresholds, or subgroup routes.
