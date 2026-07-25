# Improvement plan — paired WSL-to-GT supervision gap

## Objective

Build a bone-tumor segmentation pipeline under image-level-only weak
supervision. The experimental ResNet18 U-Net trained from pseudo-masks must
come within `0.05` mean tumor-only Dice of the frozen GT-trained reference in
each validation lesion-size subgroup:

- small: `gt_area_ratio < 0.01`;
- medium: `0.01 <= gt_area_ratio < 0.05`;
- large: `gt_area_ratio >= 0.05`.

Subgroup membership is a post-prediction validation diagnostic only. It is
never available to pseudo-mask generation, training-time routing or inference.
Complete misses remain in every tumor mean. Test stays locked until the final
WSL pipeline has been fixed.

There is no restriction on the model family or weakly supervised technique,
provided that the experimental pipeline uses only radiographs and image-level
labels from BTXRD. External pretrained models are allowed when their training
does not consume BTXRD masks/polygons.

## Frozen reference and fair-pair contract

Reference `gt_resnet18_unet_448_v1` is hash-locked under
`artifacts/reference/gt_resnet18_unet_448_v1/`.

Both arms use the same:

- clean split and group isolation;
- ImageNet-pretrained ResNet18 U-Net at 448 px;
- seed 42 and batch size 8;
- ImageNet normalization and paired horizontal-flip augmentation (`p=0.5`);
- `0.5*BCEWithLogits(pos_weight=10) + 0.5*soft-Dice`;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- maximum 35 epochs and early-stop patience 10;
- fixed-threshold-0.5 checkpoint selection with normal specificity used only
  as the frozen tie-breaker;
- validation threshold grid `0.20:0.05:0.85` and selection rule;
- evaluator, cohort, subgroup definitions and paired group bootstrap.

Only the train mask source differs:

- reference: train GT masks;
- experimental: pseudo-masks generated from images and image-level labels.

The reference checkpoint, logits and predictions may not initialize, distill,
guide, filter or select the WSL arm. Train GT may not influence WSL method
selection. Validation GT is read only after predictions exist.

If a future experiment changes the consumer architecture, loss, augmentation,
training schedule or self-training procedure, the current pair is no longer
valid. A new GT reference must be trained and hash-locked under the identical
new consumer contract before the corresponding WSL result can be judged.

## Starting point

The GT reference reaches:

| Population | GT-reference Dice |
|---|---:|
| Overall tumor | 0.4951316963 |
| Small | 0.3289549325 |
| Medium | 0.6624417784 |
| Large | 0.6937033566 |

The historical image-label-only pseudo-mask consumer reaches:

| Population | WSL Dice | Absolute gap | Goal |
|---|---:|---:|---:|
| Overall tumor | 0.2300198701 | 0.2651118262 | diagnostic |
| Small | 0.0710014171 | 0.2579535154 | <=0.05 |
| Medium | 0.4134030430 | 0.2490387353 | <=0.05 |
| Large | 0.3269168774 | 0.3667864792 | <=0.05 |

All paired bootstrap intervals are wholly below zero. Required inclusive WSL
Dice intervals are:

- small: `[0.2789549325, 0.3789549325]`;
- medium: `[0.6124417784, 0.7124417784]`;
- large: `[0.6437033566, 0.7437033566]`.

Small tumors remain the named research priority, while the large subgroup
currently has the largest measured supervision gap. A method cannot be
declared successful by improving only the overall mean or only one subgroup.

## Experiment order

### Stage 1 — close the existing proposal-gallery diagnostic

**Completed and rejected.** The independently audited 32-by-32 grid run
achieved only `0.0278769051` tumor Dice and reduced clipped single-candidate
oracle Dice from `0.3413734584` for the same-CPM component-prompt control to
`0.1130855663`. Direct CAM/prompt diagnostics were identical, so the failure
is localized to the automatic SAM proposal pool. Dense-grid and multiscale-grid
follow-ups are closed under this SAM/filtering contract; no train masks are
generated from this candidate.

Finish the already-predeclared 32-by-32 SAM grid-gallery validation run.
Freeze the CPM CAM, `coverage_mass_sam` scoring formula, support clipping,
post-processing, cohort and metrics. Compare it against both:

1. the promoted LayerCAM/flip-TTA pseudo-mask baseline;
2. the same CPM Gate-C run with component point/box prompts.

Record overall and subgroup Dice, paired group-bootstrap intervals, direct-CAM
support, prompt statistics, unclipped/clipped single-candidate oracle Dice,
selection loss and final Dice. The run is diagnostic until all hashes,
371/184/187 coverage, complete misses and `test_evaluated=false` pass.

Decision:

- if the dense gallery materially improves small-lesion oracle quality, retain
  it as a proposal mechanism and next address selection;
- if oracle quality does not improve, reject dense grid prompting and return
  to the CAM/localization source;
- if oracle improves but selected/final Dice does not, do not generate 2,981
  train masks yet; repair the GT-free selector first.

### Stage 2 — improve pseudo-mask supervision, one bottleneck at a time

Candidate families are not restricted, but every run must change a declared
mechanism and remain image-label-only.

1. **Proposal omission**
   - pseudo-trained segmentation-teacher components added to the retained
     CAM-component SAM gallery;
   - high-resolution/pyramidal image-label CAM;
   - CAM peak and connected-region prompts;
   - consensus proposals from independently pretrained foundation models.

2. **Selector loss**
   - coarse-mask-to-SAM pixel matching using only the predicted CAM/support;
   - rank consistency across flip/scale views;
   - image-label classification consistency after retaining/removing a
     candidate;
   - global versus component-local ranking, predeclared and evaluated without
     per-image GT routing.

3. **Pseudo-label boundary/noise**
   - agreement masks or confidence-weighted ignore regions derived only from
     model predictions;
   - teacher-student refinement and self-training only after defining a new
     paired consumer contract and training the matching GT reference;
   - no morphology or area threshold selected from GT lesion size.

4. **Foundation-model localization fallback**
   - This branch is activated only if the frozen proposal-teacher
     `source_consensus` selector is rejected. It does not repeat the already
     rejected 448px DenseNet, AdvCAM, S2C/CPM or automatic SAM-grid families.
   - Use a frozen biomedical vision-language encoder (BiomedCLIP) to produce a
     tumor-vs-normal saliency map from image-label-derived text prompts. The
     prompt templates and model/checkpoint hash must be fixed before reading
     validation segmentation metrics. Tumor-type text may be used only when it
     is an available image-level label.
   - First run a validation-only zero-shot diagnostic: BiomedCLIP
     saliency/gScoreCAM -> deterministic CRF or confidence seeds -> the existing
     hash-locked SAM candidate/evaluator. Report the same support, prompt,
     oracle, selector, post-processing and subgroup chain. Do not generate
     train masks unless the candidate passes a predeclared promotion gate.
   - If zero-shot localization adds a credible proposal oracle, a later
     train-only prompt-tuning stage may use BTXRD image labels but no polygons,
     masks, validation GT routing or test images. Any pseudo-mask consumer still
     uses the frozen paired GT-reference contract.
   - This is a BTXRD-specific, independently audited adaptation motivated by
     FMA-WSSS and MedCLIP-SAM, not a claim of copying or reproducing either
     complete method. External pretrained weights, licenses, source commits and
     deviations must be recorded explicitly.

Every validation candidate must report the complete error chain:

`CAM support -> proposal oracle -> selector -> support clip -> post-process`.

Scalar gates derived from validation GT size, per-image oracle choice, GT crop,
or any polygon feature are prohibited even as implementation shortcuts.

### Stage 3 — regenerate train pseudo-masks only after a valid promotion

A proposal/selector candidate advances to train generation only when:

- all 371 validation predictions and all 184 tumors are present;
- source, split, classifier/foundation-model and SAM hashes verify;
- test is untouched;
- overall and subgroup behavior addresses the diagnosed bottleneck;
- any point gain is accompanied by paired complete-group uncertainty;
- no hidden GT-dependent routing or parameter appears in generation.

Generate all 2,981 clean-train pseudo-masks from image-level labels, write a
checksum-bearing manifest, and verify each mask on its immutable source grid.
Known normal image labels may produce explicit empty masks. Train polygons are
not loaded.

### Stage 4 — paired consumer training

Train the WSL consumer with the frozen reference-v1 consumer contract. Do not
tune loss, seed, augmentation, budget or checkpoint selection separately for
the WSL arm.

After prediction:

1. select threshold using the common validation grid;
2. audit exact 371/184/187 cohort equality;
3. run `project/tools/audit_wsl_gt_pair.py`;
4. report per-image rows, all subgroup means, absolute gaps and 10,000 paired
   complete-group bootstrap intervals;
5. mark success only if small, medium and large each satisfy `abs(gap)<=0.05`.

Failure is evidence. If the consumer fits pseudo-targets but the GT gap stays
large, decompose pseudo-mask omission, commission, boundary and subgroup errors
instead of extending identical training.

## Promotion, logging and artifact policy

- Record every scientific launch, implementation failure, completed result,
  hash audit and accept/reject decision in `RESEARCH_LOG.md`.
- Keep compact JSON/CSV/log evidence for rejected methods; remove heavy masks,
  caches and checkpoints after their hashes and conclusions are preserved.
- Implementation failures are not model evidence and never justify weakening
  a scientific gate.
- Validation decisions never read test.
- All GPU-heavy generation, training and full-cohort inference runs execute on
  Kaggle.

## Research basis

- S2C transfers SAM region structure into CAM training under weak supervision:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html
- AdvCAM expands image-label discriminative evidence but requires explicit
  localization/selection audits:
  https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Anti-Adversarially_Manipulated_Attributions_for_Weakly_and_Semi-Supervised_Semantic_Segmentation_CVPR_2021_paper.html
- ToCo motivates token-level consistency for more complete WSSS localization:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- The dense-grid proposal diagnostic follows the mask-gallery idea in the
  Pro2SAM ECCV paper while retaining this project's independently frozen
  selector and protocol:
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08795.pdf
- FMA-WSSS motivates frozen CLIP/SAM coarse-to-fine seeds and learnable
  task-specific prompts under image-level supervision:
  https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html
- MedCLIP-SAM motivates biomedical CLIP saliency, CRF seed refinement and SAM
  prompting across medical modalities, including X-ray:
  https://papers.miccai.org/miccai-2024/paper/2311_paper.pdf
- BiomedCLIP is an open biomedical vision-language model pretrained on PMC-15M
  and is used only as an external frozen prior unless a separate image-label
  prompt-tuning protocol is predeclared:
  https://arxiv.org/abs/2303.00915

External papers motivate mechanisms only. Their benchmark scores are never
treated as BTXRD evidence.
