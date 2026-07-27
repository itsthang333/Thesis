# External medical foundation models as proposal sources for BTXRD WSSS

Date: 2026-07-28  
Status: literature/feasibility analysis only; no experiment authorized by this note

## Question and supervision boundary

The narrow question is whether a frozen medical segmentation foundation model
can repair the weak small-lesion proposal support observed in the BTXRD
image-label-only pipeline.

Using such a model without any BTXRD mask for training, tuning, prompting or
checkpoint selection remains compatible with **image-label-only adaptation on
BTXRD**. It is not equivalent to a method that has never seen dense
supervision: MedSAM, SAM-Med2D and BiomedParse were pretrained with external
image-mask pairs. Any result must therefore be described as:

> image-label-only BTXRD WSSS using a frozen, externally mask-pretrained
> proposal foundation model.

This arm must be reported separately from both (a) a pure generic-SAM proposal
arm and (b) fully supervised BTXRD training. Frozen external dense pretraining
does not authorize reading BTXRD validation masks before predictions are
frozen, reading BTXRD test, or fine-tuning any model with BTXRD masks.

## Primary-source audit

### 1. MedSAM: first-priority proposal source

Ma et al., *Segment Anything in Medical Images*, Nature Communications 2024:
https://www.nature.com/articles/s41467-024-44824-z

Official code and checkpoint instructions:
https://github.com/bowang-lab/MedSAM

- MedSAM was trained using 1,570,263 image-mask pairs spanning ten imaging
  modalities and more than 30 cancer types; its corpus includes grayscale
  X-ray images.
- It is a SAM ViT-B derivative and accepts box prompts. The official repository
  is Apache-2.0 and publishes `medsam_vit_b` checkpoint instructions.
- This is the closest drop-in replacement for the present ViT-B SAM proposal
  stage. It can reuse the same frozen prompt boxes/windows, inverse geometry,
  exact-duplicate rule and downstream RAD-DINO candidate scoring.
- The published breadth is not evidence of bone-tumor performance. A broad
  study of vanilla SAM in medical imaging found large dataset-dependent
  variation, including a reported IoU range from 0.1135 on one spine MRI task
  to 0.865 on a hip X-ray task:
  https://www.sciencedirect.com/science/article/pii/S1361841523001780
  This supports an oracle-first diagnostic, not an assumed gain.

### 2. SAM-Med2D: second-priority, computationally attractive source

Cheng et al., *SAM-Med2D: Bridging the Gap between Natural Image Segmentation
and Medical Image Segmentation*, arXiv 2023:
https://arxiv.org/abs/2308.16184

Official code:
https://github.com/openmedlab/SAM-Med2D

- The project reports 4.6 million training images and 19.7 million masks across
  ten modalities, anatomical structures and lesions.
- It freezes the original image encoder, inserts trainable adapters, and trains
  the prompt encoder/mask decoder with points, boxes and masks.
- The official implementation defaults to 256x256 inference and is
  Apache-2.0. It is therefore a plausible lighter second proposal source on a
  T4, but its aggregate prompt metrics cannot be converted into a BTXRD Dice
  expectation.
- Its medical adaptation may improve boundary/proposal geometry, while the
  lower default resolution may still erase the smallest tumors. The two
  effects must be measured by the frozen proposal oracle and complete-miss
  counts.

### 3. BiomedParse: useful zero-shot diagnostic, not the first implementation

Zhao et al., *A foundation model for joint segmentation, detection and
recognition of biomedical objects across nine modalities*, Nature Methods:
https://www.nature.com/articles/s41592-024-02499-w

Official project, repository and model card:
https://microsoft.github.io/BiomedParse/
https://github.com/microsoft/BiomedParse
https://huggingface.co/microsoft/BiomedParse

- The 2D model covers nine modalities including X-ray, uses text prompts, and
  was trained on more than six million image-mask-text triples covering 82
  learned object types.
- The public examples include chest-X-ray infection, not primary bone tumors.
  A text prompt such as `bone tumor` may be outside its learned ontology.
- Prompt sweeping after validation-mask inspection would be a hidden
  hyperparameter search. A valid diagnostic would freeze a short,
  semantically justified prompt list before inference and retain every
  complete miss.
- Detectron2/CUDA dependencies and ontology mismatch make it lower priority
  than the two box-prompt models. It should only enter if a source manifest
  confirms a relevant target vocabulary or if the box-prompt sources fail.

### 4. FluoroSAM: low-priority transfer source

Seibold et al., *FluoroSAM: A Language-Promptable Foundation Model for X-Ray
Image Segmentation*, MICCAI 2025:
https://papers.miccai.org/miccai-2025/0344-Paper5042.html

- FluoroSAM was trained from scratch on three million synthetic X-rays derived
  from 1,621 CT volumes, with organ and tool classes, and evaluated on real
  X-rays.
- Its radiographic inductive bias is attractive, but the synthetic projection
  domain and organ/tool vocabulary are poorly matched to subtle primary bone
  lesions. It is not the next rational BTXRD proposal source.

## Dataset-overlap and provenance risk

The BTXRD data paper is Scientific Data 2025:
Yao et al., *A Radiograph Dataset for the Classification, Localization, and
Segmentation of Primary Bone Tumors*,
https://www.nature.com/articles/s41597-024-04311-y

MedSAM (2024 publication) and SAM-Med2D (2023 release) predate the public BTXRD
paper/release. This is strong chronological evidence against intentional
BTXRD inclusion, but it is not proof that no source image was duplicated in
any upstream public corpus. Before use, freeze:

1. repository revision, model URL, license and checkpoint SHA-256;
2. the model's available training-corpus manifest and publication date;
3. a statement that no BTXRD image or mask was supplied during adaptation;
4. perceptual-hash duplicate screening if the external training image list is
   actually obtainable.

Absence of a complete upstream image manifest must be reported as a limitation,
not silently interpreted as proof of non-overlap.

## Conditional experiment design

This experiment is **not launched while mask-bag MIL v6 is unresolved**. It is
eligible only if the terminal decision enters the frozen
`oracle-fail -> high-resolution proposal support` branch, or if oracle
headroom is too narrow specifically for small lesions.

### Frozen proposal-union diagnostic

1. Preserve the current proposal gallery and all old proposal identities.
2. Add MedSAM and SAM-Med2D arms using exactly the same predeclared boxes,
   prompt windows, square-pad inverse geometry and image resolution contract.
3. Generate proposals for all clean-train and all 371 validation images,
   including normal images. Do not use labels to suppress proposal generation.
4. Keep model/source/prompt/component provenance for every mask. Remove only
   bit-exact duplicate masks; do not select by validation GT.
5. Freeze all masks and hashes before the single validation-GT oracle audit.
6. Report proposal-oracle Dice, complete misses and paired changes for
   overall/small/medium/large. In particular, require fewer small complete
   misses and positive paired small change; an overall-only gain is
   insufficient.
7. If the union oracle improves, selection remains image-level-only:
   cross-fitted RAD-DINO relational scoring, family-normalized logits and
   area-stratified candidate handling. The oracle is a diagnostic upper bound,
   never a deployable prediction.
8. If the union oracle does not improve, reject the exact medical proposal
   transfer. Do not train a consumer to compensate for missing spatial
   support.

No numeric oracle threshold is invented here after observing prior validation
GT. Any new gate must be frozen in a separate protocol before the job is
launched and must retain the already frozen operational goals and
consumer-entry policy.

## T4x2 execution plan

- Heavy inference runs only on Kaggle.
- Benchmark a small fixed, GT-blind train subset for memory/throughput, then
  freeze the execution choice.
- Preferred scheduling is one immutable source per T4 (MedSAM on GPU 0,
  SAM-Med2D on GPU 1) when runtimes coexist safely. If one arm dominates
  runtime, shard that arm over both T4s in a separate stage.
- Do not use DDP for independent source models. Write disjoint shards,
  synchronize only after completion, verify count/hash/provenance, then build
  the union.
- MedSAM ViT-B is expected to fit similarly to the existing SAM ViT-B
  inference. SAM-Med2D's 256 default is lighter, but this is an engineering
  expectation to benchmark, not a claimed runtime.

## Decision

The best combined direction is not to replace RAD-DINO with a medical
segmenter. It is:

`frozen medical proposal geometry -> provenance-preserving proposal union ->
cross-fitted image-label relational selector -> optional robust consumer only
after the frozen entry gate`.

This combination assigns each model the role supported by its training:
medical SAM variants supply boundary priors, RAD-DINO supplies radiographic
semantic descriptors, image-level MIL chooses candidates, and a consumer is
considered only after a strong prediction source exists. MedSAM is priority 1,
SAM-Med2D priority 2, BiomedParse a conditional diagnostic, and FluoroSAM is
deferred. The proposal-union arm is scientifically admissible but should run
only after v6 evidence selects it.
