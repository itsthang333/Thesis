# BTXRD image-level WSSS deep research synthesis

Date: 2026-07-28

Status: research/contingency artifact. It does not alter the running
RAD-DINO mask-bag MIL version-6 protocol, predictions, gate, or operational
goals. The BTXRD test split and all segmentation annotations were excluded from
this review.

## Research question

What is the most defensible next mechanism, or staged combination of
mechanisms, for reaching the frozen validation goals
`0.34024039/0.17895493/0.51244178/0.49370336`
(overall/small/medium/large) using only image-level labels during BTXRD
training?

The answer depends on a diagnostic that version 6 will supply:

- **proposal-support failure:** the best frozen proposal per image cannot meet
  the goals;
- **proposal-ranking failure:** adequate proposals exist but the deployable
  selector chooses the wrong one;
- **consumer/generalization failure:** a prediction-first pseudo source passes,
  but a downstream dense model later memorizes its noise.

Combining methods before separating these failures would obscure attribution
and consume validation evidence inefficiently.

## New source-level findings about version 6

The current runner and payload schema were inspected without reading any mask
ground truth.

1. Each proposal is scored independently by the same MLP. The only interaction
   between candidates is normalized LogSumExp at bag pooling and argmax at
   prediction. The scorer cannot compare one candidate with its neighbours,
   detect near-duplicate proposals, or model agreement among prompt sources.
2. The frozen payload already stores `component_ids`, `prompt_modes`, and
   `proposal_source_ids`, but `_load_candidate_payload` discards all three.
   It retains only SAM score, log area, prompt-map mass coverage, and mean
   prompt-map value as metadata.
3. Candidate counts are largely generated in prompt families: several SAM
   masks are returned for each component/prompt mode. The no-GT audit found
   count-only AUROC `0.86726993` train and `0.71207277` validation. Normalized
   LogSumExp removes the exact mathematical advantage of duplicating identical
   logits, but it does not remove different component/source distributions or
   multiple correlated high-logit candidates.
4. After a two-epoch warm-up, the positive instance target is the model's own
   detached argmax. There is no out-of-fold prediction. A wrong early winner
   can therefore be reinforced on the same image, matching the premature
   localization lock-in identified by multi-fold MIL.
5. The original/flip consistency term is scientifically useful and should be
   retained. The RAD-DINO backbone is frozen and the small-mask area
   interpolation preserves fractional token occupancy; neither is the first
   component to replace if proposal support is adequate.
6. Earlier corrected experiments provide complementary frozen evidence:
   standalone dense maps had useful pixel ranking but poor shape. In
   particular, the corrected nominal-memory single-scale map was much stronger
   than corrected INSIGHT as a ranker, while neither was an acceptable final
   mask. Such evidence is better reused as proposal-level features than
   thresholded again as another standalone pseudo mask.

These findings make a relational, family-aware selector a materially different
experiment from the rejected hand-written prompt/source graph: the earlier
graph used fixed heuristics, whereas the proposed model learns proposal
semantics from image labels while explicitly controlling prompt multiplicity.

## Primary literature and what actually transfers

### Relational and cross-fitted MIL

- Ramazan Gokberk Cinbis, Jakob Verbeek, and Cordelia Schmid,
  *Multi-fold MIL Training for Weakly Supervised Object Localization*, CVPR
  2014, pp. 2409-2416:
  https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html

  Multi-fold MIL prevents a learner from repeatedly assigning and training on
  its own erroneous positive location. The direct transfer is group-preserving
  train-only cross-fitting: instance targets for one training fold must come
  from a selector fitted without that fold.

- Bin Li, Yin Li, and Kevin W. Eliceiri, *Dual-stream Multiple Instance
  Learning Network for Whole Slide Image Classification with Self-supervised
  Contrastive Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html

  DSMIL combines an instance classifier with a bag stream that measures each
  instance against a critical instance. The transferable mechanism is
  candidate-to-candidate relational evidence, not the pathology encoder or
  slide tiling.

- Ming Y. Lu et al., *Data-efficient and weakly supervised computational
  pathology on whole-slide images*, Nature Biomedical Engineering 5,
  555-570 (2021):
  https://www.nature.com/articles/s41551-020-00682-w

  CLAM constrains the feature space using representative high/low-attention
  instances under bag labels. For BTXRD, reliable normal-bag candidates are
  unusually strong negatives; a conservative contrastive constraint can use
  them while leaving ambiguous positive-bag candidates unlabeled.

- Zhaolong Du et al., *Rethinking Multiple-Instance Learning From Feature
  Space to Probability Space*, ICLR 2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html

  PSMIL explicitly targets selection drift using self-training alignment and
  attention in probability space. It supports the diagnosis of unstable
  instance selection, but its full framework is second-line: cross-fitting is
  simpler, easier to audit, and better matched to the current source.

### Position and radiographic alignment

- Meera Krishnamoorthy and Jenna Wiens, *Multiple Instance Learning with
  Absolute Position Information*, CHIL/PMLR 248:88-104, 2024:
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html

  Adding positional encodings improved standard MIL on chest radiographs from
  AUROC `0.782` to `0.799` and matched a transformer while remaining much
  faster. BTXRD covers heterogeneous anatomy and views, so absolute position
  must not become a new shortcut. The admissible low-risk transfer is normalized
  candidate centroid/bounding-box geometry coupled to flip-equivariance, with
  an ablation against no position. True anatomy, lesion size, or subgroup may
  not route the model.

### Avoiding dominant-instance and bag-context shortcuts

- Wenhui Zhu et al., *How Effective Can Dropout Be in Multiple Instance
  Learning?*, ICML 2025, PMLR 267:80090-80106:
  https://proceedings.mlr.press/v267/zhu25q.html

  MIL-Dropout removes top-ranked instances and similar neighbours during
  training, encouraging alternative evidence. The authors' implementation
  computes non-parametric importance, masks the top instances plus
  feature-similar neighbours, rescales the surviving features, and disables
  the operation at inference. For BTXRD it is a regularizer after a relational
  selector is stable, not the first repair: aggressively dropping the only
  true small-lesion proposal can be harmful.

- Tiancheng Lin et al., *Interventional Bag Multi-Instance Learning on
  Whole-Slide Pathological Images*, CVPR 2023, pp. 19830-19839:
  https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Interventional_Bag_Multi-Instance_Learning_on_Whole-Slide_Pathological_Images_CVPR_2023_paper.html

  IBMIL treats bag context as a confounder. This supports auditing proposal
  count/source dependence, but its confounder dictionary and interventional
  training are deferred until a simpler family-balanced/cross-fitted model
  demonstrably retains shortcut dependence.

- JuneHyoung Kwon et al., *Learning to Detour: Shortcut Mitigating
  Augmentation for Weakly Supervised Semantic Segmentation*, WACV 2024,
  pp. 819-828:
  https://openaccess.thecvf.com/content/WACV2024/html/Kwon_Learning_to_Detour_Shortcut_Mitigating_Augmentation_for_Weakly_Supervised_Semantic_WACV_2024_paper.html

  SMA disentangles and recombines object/background representations to weaken
  contextual shortcuts. Its full feature synthesis is not a first-line
  transfer, but it motivates a simpler measurable invariant: a candidate
  should retain its score under aligned view/intensity changes and should not
  be identifiable merely from prompt-family/count context.

### Small-object support and global-to-local consistency

- Kangning Liu et al., *Weakly-supervised High-resolution Segmentation of
  Mammography Images for Breast Cancer Diagnosis*, MIDL/PMLR 143:451-472,
  2021:
  https://proceedings.mlr.press/v143/liu21b.html

  GLAM uses coarse localization to select high-resolution regions and fuses
  global/local saliency. The medical setting and small lesion-to-image ratio
  closely match the BTXRD failure, though mammography and bone radiography
  remain different modalities.

- Peng-Tao Jiang et al., *L2G: A Simple Local-to-Global Knowledge Transfer
  Framework for Weakly Supervised Semantic Segmentation*, CVPR 2022,
  pp. 16886-16896:
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html

  Local crops reveal details missed in the global view; a global network learns
  their complementary attention online. For BTXRD, a positive image label
  cannot be copied blindly to every crop because most crops may miss a tiny
  tumor. Local views need a frozen global-evidence intersection, teacher
  consistency, or an ignore state.

- Yude Wang et al., *Self-Supervised Equivariant Attention Mechanism for
  Weakly Supervised Semantic Segmentation*, CVPR 2020, pp. 12275-12284:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html

  SEAM enforces spatial equivariance across transformed views and uses
  pixel-correlation refinement. The project already applies flip consistency
  at candidate-score level; a high-resolution branch should extend consistency
  to aligned local/global spatial evidence rather than add arbitrary crops.

- Dongjun Hwang, Seong Joon Oh, and Junsuk Choe, *Small object matters in
  weakly supervised object localization*, Neurocomputing 648 (2025), 130494:
  https://doi.org/10.1016/j.neucom.2025.130494

  Their image-label-only zoomed foreground/background consistency directly
  targets small-object localization and reports preservation of medium/large
  performance. The transferable principle is deterministic
  prediction-conditioned zoom with cross-view consistency.

- Cheolhyun Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
  Segmentation*, WACV 2024, pp. 413-422:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

  This work shows why aggregate scores hide small-object failure. It validates
  the project's subgroup gate and complete-miss reporting. Its size-balanced
  dense loss is relevant only after a consumer is authorized and may use
  pseudo-derived, never GT-derived, size.

### Foundation-model and boundary transfer

- Bingfeng Zhang et al., *Frozen CLIP: A Strong Backbone for Weakly
  Supervised Semantic Segmentation*, CVPR 2024, pp. 3796-3806:
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html

  WeCLIP supports a frozen foundation backbone, light spatial decoder, and
  dynamic refinement. Project evidence already showed that a free dense
  decoder can learn an image-classification shortcut, so the transferable part
  is refinement/affinity as proposal evidence, not another unconstrained mask.

- Hyeokjun Kweon and Kuk-Jin Yoon, *From SAM to CAMs: Exploring Segment
  Anything Model for Weakly Supervised Semantic Segmentation*, CVPR 2024,
  pp. 19499-19509:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html

  SAM masks can regularize CAM learning rather than serve only as final
  post-processing. For BTXRD, the corresponding low-risk transfer is to use
  prompt-family/source agreement as a learned proposal relation, because SAM
  boundaries alone have already failed to solve semantic selection.

- Ostap Viniavskyi, Mariia Dobko, and Oles Dobosevych, *Weakly-Supervised
  Segmentation for Disease Localization in Chest X-Ray Images*, 2020:
  https://arxiv.org/abs/2007.00748

  Their image-label chest-X-ray pipeline refines confident seeds with learned
  inter-pixel relations before consumer training. It supports affinity after
  seed/proposal recall is adequate, not as a mechanism for a completely missed
  lesion.

## Ranked staged designs

### Design R: family-balanced cross-fitted relational proposal MIL

**Trigger:** current proposal oracle passes the operational goals but version-6
selected localization fails.

This is the highest-priority ranking repair.

1. Keep the exact frozen SAM masks and RAD-DINO token maps.
2. Preserve the existing inside/ring/contrast descriptor and flip consistency.
3. Load the already-frozen `component_ids`, `prompt_modes`, and
   `proposal_source_ids`; add normalized centroid, bounding-box dimensions and
   compact shape geometry. None requires GT.
4. Add frozen nominal-normality evidence per proposal: inside mean/max/mass,
   ring mean, and inside-minus-ring contrast from the already audited
   single-scale nominal-memory map. This reuses a ranking signal without
   thresholding it into another poor shape.
5. Replace independent scoring with a small dual-stream relational head:
   instance logits plus attention conditioned on a critical candidate and
   proposal-family/source embeddings.
6. Aggregate first within each component/prompt family, then across families,
   so producing more correlated SAM variants cannot automatically create more
   bag evidence. Retain an explicit normalized formulation and fail closed on
   empty families.
7. Generate positive-instance soft targets out-of-fold using
   group-preserving train-only folds. A fold never trains on its own inferred
   winner. Normal-bag candidates remain reliable negatives; ambiguous
   positive candidates remain ignored or softly weighted.
8. Fit one final model on all training images using the frozen out-of-fold
   rule. Freeze all validation candidate scores/maps before GT evaluation.

**Required no-GT diagnostics:** train/validation candidate-count-only AUROC,
bag-score versus count/source/component-count association, flip agreement,
attention entropy, fraction of winners by prompt mode/source, and normal-image
false-positive score. Image AUROC alone remains insufficient.

**Ablation order:** family/source/geometry metadata; then relational
cross-fitting; then nominal-memory features. MIL-Dropout (`K=1` initially) is
added only if winner concentration/count dependence persists. IBMIL is later.

**Why the combination is coherent:** source/family balancing attacks the
measured shortcut, cross-fitting attacks confirmation bias, relational
attention attacks independent-candidate blindness, and normality features add
an orthogonal medical signal. All four operate at proposal ranking and leave
candidate geometry untouched.

### Design P: prediction-conditioned high-resolution proposal expansion

**Trigger:** proposal oracle fails, especially on small tumors, or Design R
proves that the existing oracle margin is too narrow to make selection robust.

1. Use a frozen global map to select the same fixed number of windows for every
   image, including normal images. No GT, subgroup, or true size routing.
2. Re-encode windows at high resolution with frozen RAD-DINO and create local
   class-agnostic SAM proposals.
3. Map every local proposal back through exact recorded crop/pad geometry.
4. Require aligned global/local and original/flip consistency. A positive
   image label is not assigned blindly to every crop; unsupported crops are
   ignored.
5. Append the local candidates to a separately frozen gallery and audit the
   new oracle before training a selector.
6. If support improves, apply Design R to the augmented gallery rather than
   invent a second hand-written selector.

This combines GLAM's coarse-to-fine resolution, L2G's local detail transfer,
SEAM/Hwang consistency, and the existing SAM proposal boundary prior. The
combination is scientifically coherent because every component acts on missing
small-object support.

### Design C: uncertainty-aware dense consumer

**Trigger:** only after a prediction-first source passes its full gate.

- keep soft proposal/localizer confidence and cross-view agreement;
- supervise only stable foreground/background pixels and ignore uncertain
  boundaries;
- use random-view consistency on ignored pixels;
- study pseudo-size-balanced loss without true size labels;
- evaluate the consumer against the same complete subgroup/miss gate.

This stage must not be used to rescue a failed proposal or selector.

## Candidate combinations rejected as first action

- **Another standalone continuous RAD-DINO decoder:** corrected dense-MIL,
  INSIGHT, nominal-memory threshold maps, and affinity-decoder evidence already
  show rank signal without usable shape.
- **SAM/MedSAM swap alone:** a boundary model cannot identify which proposal is
  the tumor; semantic selection is the current likely bottleneck.
- **Blind random crops or tiles:** a tiny positive lesion is absent from most
  crops, so inheriting the image label creates false-positive instance labels.
- **Large transformer MIL immediately:** only 2,981 training bags are
  available; a small relational head with explicit provenance is more
  auditable and T4x2-friendly.
- **Using true subgroup/size at training or routing:** this is leakage even if
  it improves small Dice.
- **Selecting combinations from validation Dice repeatedly:** each design must
  be predeclared and predictions frozen. Validation GT may evaluate a protocol,
  not tune it interactively.

## Expected-value decision

Before version 6 is known, the best conditional portfolio is:

1. **If gate passes:** stop altering localization and predeclare Design C.
2. **If oracle passes but selection fails:** Design R has the highest expected
   value because it directly fixes three concrete source defects and reuses the
   strongest frozen normality signal.
3. **If oracle fails:** Design P has the highest expected value for small
   tumors because it increases effective spatial resolution without
   GT-selected crops.
4. **If Design R still fails while oracle remains strong:** measure whether
   score/count dependence or winner lock-in remains. Use MIL-Dropout for the
   former concentration problem, PSMIL for persistent probability drift, or
   IBMIL only for demonstrated bag-context confounding.
5. Combine Design P and R only after P independently increases proposal
   support; combine R and C only after R independently passes the
   prediction-first gate.

This staged combination is more likely to generalize than adding all losses at
once, and it preserves a falsifiable explanation for every result.
