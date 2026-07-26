# Medical/X-ray model landscape for BTXRD image-label-only WSSS

Date: 2026-07-26

> This file is the deployment/model availability audit. The broader,
> technique-first transfer analysis requested afterward is in
> `technique_transfer_review_2026-07-26.md`. A method does not need to match
> BTXRD exactly or publish a checkpoint to contribute a transferable mechanism.

## Scope and decision rule

The target is tumor segmentation on BTXRD radiographs while adapting to BTXRD
with image-level labels only. External pretrained weights may have been trained
with richer annotations, as is already true for SAM, but no BTXRD train mask,
box, point, true tumor size, validation subgroup, or validation/test GT may
enter training, proposal generation, selection, checkpoint selection, or
threshold selection. Every external checkpoint, license, source revision, and
physical hash must be frozen before evaluation.

Candidates are ranked by:

1. compatibility with 2D radiographs and small/low-contrast lesions;
2. availability of spatial features or promptable masks rather than only a
   global embedding;
3. compliance with image-label-only BTXRD adaptation;
4. reproducible checkpoint/code/license availability;
5. feasibility on Kaggle T4/P100-class hardware;
6. whether the candidate addresses a bottleneck already demonstrated in
   `RESEARCH_LOG.md`.

The current evidence matters more than model popularity. The BiomedCLIP
LayerCAM gallery contains substantially better raw candidates for every
subgroup, but direct/scalar selection has failed repeatedly. Therefore a
medical model is useful only if it improves independent localization,
cross-prompt stability, small-lesion recall, or selection—not merely because it
has "medical" in its name.

## Ranked candidates

### 1. RAD-DINO multi-resolution dense MIL/localization branch

**Decision: highest-priority new model branch after the in-flight graph
selector is evaluated.**

- RAD-DINO is an 86.6M-parameter DINOv2-base encoder self-supervised on chest
  radiographs. The official checkpoint is MIT licensed and directly exposes
  dense patch embeddings; its example output is `768 x 37 x 37`. The model card
  explicitly supports a segmentation decoder over patch tokens.
- It is not musculoskeletal-specific, so chest-to-bone domain shift is a real
  risk. However, it has the strongest combination currently available of
  X-ray-specific pretraining, open weights, dense tokens, manageable size, and
  standard Transformers integration.
- Proposed use is not another ordinary CAM. Train an image-label-only
  multi-resolution MIL head on patch tokens, using learnable
  Log-Sum-Exp/probabilistic pooling and scale consistency. High-resolution
  crops are necessary because the frozen validation audit identified
  `<1%`-area tumors as the dominant failure group.
- Its dense score map becomes an independent proposal source. It can be tested
  alone and then as prompts/support for the existing proposal gallery, keeping
  the consumer and evaluation contracts unchanged.

Primary sources:

- https://huggingface.co/microsoft/rad-dino
- https://www.nature.com/articles/s42256-024-00965-w
- https://arxiv.org/abs/1803.07703

### 2. SAM-Med2D controlled candidate-refiner swap

**Decision: first promptable medical segmenter to test; ahead of MedSAM.**

- SAM-Med2D was adapted on 4.6M 2D medical images and 19.7M masks across ten
  modalities. Unlike MedSAM's released inference path, it was explicitly tuned
  with point, box, and mask prompts, matching the project's prompt ensemble.
- The official repository provides a ViT-B checkpoint, 256-resolution path,
  ONNX export, mixed precision, and Apache-2.0 code. This is substantially more
  Kaggle-friendly than a mandatory 1024-resolution pass.
- The strict experiment is a one-variable candidate-generator/refiner swap:
  same saliency, proposal sources, prompts, selector, morphology, split, and
  Gate-C. No medical-model-specific threshold may be tuned after GT is seen.
- It does not solve proposal selection by itself. Its value is improved
  cross-prompt consistency and boundary quality, which should make a no-GT
  graph selector more reliable.

Primary sources:

- https://github.com/OpenGVLab/SAM-Med2D
- https://arxiv.org/abs/2308.16184

### 3. Direct multi-resolution dense MIL segmenter

**Decision: high-priority architectural branch; it bypasses the fragile
pseudo-mask selector.**

- A U-Net-like dense probability field can be aggregated to the image label
  with learnable Log-Sum-Exp or probabilistic CAM pooling. This makes every
  pixel/patch an MIL instance and trains the dense output using image labels
  only.
- Multi-resolution fusion plus a learnable sharpness prior was designed for
  abnormalities of different sizes under image-level supervision. A
  radiological tumor study (WSUnet) likewise trains a dense U-Net output from
  global labels.
- For BTXRD, the branch should use normal-image suppression, positive
  area/sparsity priors declared without GT fitting, cross-scale consistency,
  and high-resolution positive crops. It must be audited against trivial
  all-background and whole-image solutions.
- This is more structurally different than changing a foundation checkpoint,
  and therefore has higher scientific value if prompt-selection remains the
  bottleneck.

Primary sources:

- https://arxiv.org/abs/1803.07703
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10657919/
- https://github.com/jfhealthcare/Chexpert

### 4. Anatomy-conditioned normal-only anomaly localization

**Decision: promising small-lesion branch, but requires a carefully controlled
feasibility audit before a full run.**

- Normal radiographs provide valid negative supervision. A memory-bank or
  reconstruction model can learn normal local appearance and produce a
  per-patch residual/anomaly map without tumor masks.
- This targets a different signal from discriminative CAM: unusual local bone
  texture rather than only the most classification-discriminative region.
  That is especially relevant to tiny lesions.
- A single normal bank across all anatomical sites would mostly detect body
  part/acquisition differences. The bank must therefore be conditioned on an
  anatomy grouping obtained without tumor masks (existing metadata if
  permitted, otherwise frozen image-only clustering). Tumor size and validation
  subgroup may never route the model.
- SKELEX provides strong external motivation: it was self-supervised on 1.2M
  musculoskeletal radiographs, performed bone-tumor classification, and used
  reconstruction error for zero-shot abnormality localization. No reproducible
  public checkpoint/code was found in this audit, so SKELEX itself is a
  watchlist item, not a runnable dependency. We may test the principle
  independently; we must not claim to reproduce SKELEX.

Primary sources:

- https://doi.org/10.1038/s41746-026-02826-9
- https://arxiv.org/abs/2602.03076
- https://pubmed.ncbi.nlm.nih.gov/38952749/

### 5. MedSAM strict refiner comparator

**Decision: runnable secondary comparator, below SAM-Med2D for this project.**

- MedSAM was trained on 1,570,263 medical image-mask pairs spanning ten
  modalities and is released with Apache-2.0 code/checkpoint.
- It is a credible box-promptable medical refiner, but the public inference
  recipe is box-centered at 1024 resolution. The project's candidate ensemble
  depends on point/box combinations and must process many proposals, so cost and
  prompt mismatch are worse than SAM-Med2D.
- Small target size and low contrast are known failure axes for SAM-family
  models. MedSAM must therefore be evaluated by complete small/medium/large
  groups, not only overall Dice.

Primary sources:

- https://www.nature.com/articles/s41467-024-44824-z
- https://github.com/bowang-lab/MedSAM
- https://arxiv.org/abs/2304.09324

### 6. Learned inter-pixel affinity/boundary propagation

**Decision: refinement experiment after a stronger seed map is available.**

- A published image-label-only chest-radiograph pipeline uses classification
  regularization, an Inter-pixel Relation Network, and a final segmentation
  consumer. The reported ablation shows a large improvement from CAM to IRNet
  before the final segmenter.
- This directly addresses incomplete boundaries, but an affinity network cannot
  recover a lesion absent from all seed maps. It should follow RAD-DINO/dense
  MIL seeds rather than precede them.
- Any affinity labels must be derived only from frozen high-confidence
  foreground/background seeds; validation GT cannot tune propagation radius,
  thresholds, or loss weights.

Primary source:

- https://arxiv.org/abs/2007.00748

## Deferred or rejected as immediate experiments

| Candidate | Decision | Reason |
|---|---|---|
| SKELEX | Watchlist, very high scientific relevance | Directly targets musculoskeletal radiographs and bone tumors, but this audit found no public weights/code suitable for a hash-locked Kaggle reproduction. |
| XR-0 | Watchlist | Multi-anatomy X-ray foundation model with localization/segmentation claims, but no reproducible public checkpoint/code was found. |
| MedImageInsight | Defer | Broad X-ray performance is attractive, but current official deployment is Azure/Foundry-oriented and the premium model is closed-weight; this conflicts with a self-contained Kaggle protocol. |
| BiomedParse | Defer/reject direct use | The official X-ray vocabulary does not list bone tumor; the checkpoint is gated and CC-BY-NC-SA-4.0. It is large/Detectron2-heavy and likely to create out-of-vocabulary masks. |
| MedicoSAM | Low priority | Its own paper reports gains for interactive segmentation but no benefit from medical pretraining for automatic semantic segmentation, the relevant mode here. |
| BioViL-T / CXR Foundation | Low priority | Useful chest-X-ray embeddings/grounding, but anatomy is chest-specific and less aligned than RAD-DINO's directly accessible patch tokens. |
| Medical SAM Adapter / MedSAM2 / 3D models | Reject for now | The data are independent 2D radiographs; 3D/video memory and 3D prompt mechanisms add complexity without matching the acquisition. |
| Medical SAM3 | Watchlist | New, large, text-prompted model; checkpoint maturity, license, and T4 feasibility need independent verification before it can enter the protocol. |
| Generic SAM automatic mask grid | Reject repetition | Already evaluated in the project; dense grid proposals did not solve selection/localization and repeating them would not test a new mechanism. |
| BiomedCLIP direct replacement | Reject repetition | Already failed as a direct selector/localizer. Only its demonstrated raw proposal diversity remains useful. |

Primary sources for the deferred candidates:

- https://arxiv.org/abs/2509.12146
- https://www.microsoft.com/en-us/research/publication/medimageinsight-an-open-source-embedding-model-for-general-domain-medical-imaging/
- https://learn.microsoft.com/en-us/azure/foundry/how-to/healthcare-ai/deploy-medimageinsight-premium
- https://github.com/microsoft/BiomedParse
- https://huggingface.co/microsoft/BiomedParse
- https://arxiv.org/abs/2501.11734
- https://github.com/computational-cell-analytics/medico-sam

## Evidence-driven execution order

The earlier prompt/source graph selector is now complete and rejected: its
whole-mask selection harmed small lesions even though the frozen proposal
gallery retained positive oracle gains in all size groups. The current order is
therefore:

1. Finish the already launched SKELEX-inspired MAE normality-reconstruction
   probe. This compares the same ImageNet MAE before and after normal-only
   radiograph adaptation; it is a mechanism gate, not a final model.
2. If reconstruction error contains coherent small-tumor signal, predeclare a
   separate fusion/proposal protocol before combining it with existing CAMs.
   No threshold or per-image branch may be chosen from validation GT.
3. If pixel residual is weak or dominated by acquisition effects, use
   mid-level frozen radiograph features in a context-conditioned
   PatchCore-style nominal memory. This tests feature anomaly rather than
   repeating reconstruction with another checkpoint.
4. Implement UM-CAM spatial uncertainty across aligned resolutions/views and
   geodesic expansion as separate ablations. This directly targets the
   measured over-broad selector failure.
5. Build the RAD-DINO multi-resolution dense MIL branch. First freeze
   image-label classifier/localizer checkpoints and no-GT maps; then evaluate
   localization/pseudo-mask Dice and subgroup deltas.
6. If a new seed source improves recall but boundaries remain weak, apply either
   SAM-Med2D refinement or learned inter-pixel affinity as a separate
   one-variable experiment.
7. Add Random-View Consensus/uncertainty-ignore supervision to the paired
   pseudo-mask consumer only after a pseudo source passes its prediction-first
   gate.
8. Keep radiograph-compatible NSA/DRAEM synthetic anomaly learning as a
   controlled fallback. Real validation localization, not synthetic training
   accuracy, is its gate.

At no point may validation outcomes retroactively change the frozen GT
reference, split, subgroup definitions, prediction manifests, or thresholds.
Test remains locked until a complete final pipeline is frozen.
