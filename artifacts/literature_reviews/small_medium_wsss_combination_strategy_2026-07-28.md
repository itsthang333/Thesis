# Small/medium WSSS combination strategy for BTXRD

Date: 2026-07-28  
Status: literature and mechanism synthesis only; no new experiment launched

## Question

Why can an operating point improve overall, medium and large Dice while small
Dice falls, and which image-level-only WSSS mechanisms can address that
without sacrificing the other groups?

The current evidence supports two distinct bottlenecks rather than one global
threshold problem:

1. **Small:** masks can occupy substantially less than one RAD-DINO token
   after exact square projection. The current descriptor divides a weighted
   feature sum by `max(grid_mass, 1)`, so accepted sub-token proposals are
   attenuated even though log area is already an explicit descriptor. Small
   lesions also suffer most from any coordinate, resize or boundary error.
2. **Medium:** the frozen candidate oracle has sufficient headroom, but the
   selector must recover 66.85% of the current-to-oracle gap to reach the
   operational goal. This is chiefly a proposal-ranking/relational-evidence
   problem.
3. **Large:** the required gap recovery is only 34.57%; aggressive
   small-object expansion can create false-positive area and erase this
   advantage.

This interpretation is consistent with Mun et al., who show that image-level
WSSS methods systematically struggle on small objects and that aggregate
metrics hide complete small-object misses. Their size-balanced loss is useful
only as a transfer hypothesis: BTXRD validation subgroup identity or mask area
must never be exposed to training.

Source: Mun et al., *Small Objects Matters in Weakly-supervised Semantic
Segmentation*, WACV 2024:
https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

## Ordered combination, not a bundled ablation

### Stage A — repair representation before changing learning

Run the already frozen geometry-v3 correction, including its GT-blind
fractional grid-mass audit. It is the only scientifically valid next arm if v6
is terminal, misses a goal and retains oracle feasibility. The geometry fix
must remain the sole causal change.

If sub-token mass is material and v3 still misses small, compare the frozen
descriptor against a true weighted mean:

`inside = sum(token_feature * mask_weight) / sum(mask_weight)`

after the same `grid_mass >= 0.25` filter. Keep log-area metadata, proposals,
selector, loss, seeds and evaluator unchanged. This removes duplicated size
attenuation without dilating masks or introducing subgroup labels.

### Stage B — add local resolution only if Stage A is insufficient

For each already-generated proposal, extract a deterministic padded crop and
encode it at the frozen RAD-DINO input resolution. Fuse:

- global square-frame proposal descriptor;
- crop-view descriptor;
- crop/global consistency and proposal metadata.

The crop box, padding factor and fusion rule must be predeclared from
train-only/image-level evidence. A flip-consistency and payload-hash audit is
required before optimization. This adapts the zoom-in principle reported for
small-object weak localization, but must not use ground-truth size or
validation subgroup membership to choose crops.

Transfer source: Hwang, Oh and Choe, *Small object matters in weakly
supervised object localization*, Neurocomputing 2025:
https://doi.org/10.1016/j.neucom.2025.130494

The stronger image-label-WSSS precedent is L2G: local patches reveal finer
object details and transfer them to a global network. Directly assigning a
positive image label to every BTXRD crop would, however, create false-positive
supervision whenever the crop misses the lesion. The safe adaptation is
proposal-instance based: a crop contributes only a descriptor for its
corresponding candidate, and image-level BCE is applied after MIL bag
aggregation. No crop receives an independent positive target.

Source: Jiang et al., *L2G: A Simple Local-to-Global Knowledge Transfer
Framework for Weakly Supervised Semantic Segmentation*, CVPR 2022:
https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html

The current original/flip loss already implements one equivariance axis.
Scale/crop consistency is a later, separate ablation—not part of geometry v3.
SEAM provides the image-label-only precedent for transformed-view attention
consistency and context-based pixel correlation.

Source: Wang et al., *Self-Supervised Equivariant Attention Mechanism for
Weakly Supervised Semantic Segmentation*, CVPR 2020:
https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html

### Stage C — relational selection for medium

If the corrected gallery oracle passes all goals but WTA remains below the
medium goal, use relational bag evidence rather than more threshold search:

- nominate a critical high-scoring proposal;
- score other proposals by similarity/complementarity to it;
- retain inner-versus-context contrast and overlap/diversity metadata;
- aggregate with normalized attention/SmoothMax under image-level BCE.

This targets proposal choice, not proposal generation. It follows the
critical-instance plus relational-evidence idea of DSMIL and arbitrary-mask
proposal pooling, without importing their datasets, metrics or instance
labels.

Sources:

- Li et al., *Dual-stream Multiple Instance Learning Network for
  Whole Slide Image Classification with Self-supervised Contrastive
  Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_CVPR_2021_paper.html
- Shen et al., *Toward Joint Thing-and-Stuff Mining for Weakly Supervised
  Panoptic Segmentation*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html

### Stage D — contour refinement only after selector entry

If the prediction-first selector passes the frozen consumer-entry gate, a
consumer may use confidence-weighted pseudo masks, uncertain-pixel ignore
regions and multi-scale features. A superpixel/affinity boundary head is a
plausible radiograph-specific refinement because it propagates local context
and can contour lesions, but it must not train before the entry gate.

Medical transfer sources:

- Saeed et al., *Deep superpixel generation and clustering for weakly
  supervised segmentation of brain tumors in MR images*, Medical Image
  Analysis 2025, trained with binary image-level labels:
  https://pubmed.ncbi.nlm.nih.gov/39695438/
- Gu et al., *Chest L-Transformer: Local Features With Position Attention
  for Weakly Supervised Chest Radiograph Segmentation and Classification*,
  IEEE TMI 2022, using image-level labels and local prediction aggregation:
  https://pubmed.ncbi.nlm.nih.gov/35721071/
- Zhang et al., *Frozen CLIP: A Strong Backbone for Weakly Supervised
  Semantic Segmentation (WeCLIP)*, CVPR 2024, for frozen foundation features,
  a lightweight decoder and dynamic affinity refinement:
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html

## Rejected combinations

- Do not combine geometry v3, weighted-mean pooling, crop views and relational
  MIL in one run; attribution would be impossible.
- Do not use validation GT area, subgroup identity or the small/medium/large
  thresholds in training, routing or crop selection.
- Do not compensate for small misses by lowering a global threshold if that
  expands medium/large false positives.
- Do not train a U-Net/consumer from proposal masks before the frozen
  prediction-first and consumer-entry gates pass.
- Do not adopt SAM/CLIP natural-image benchmark gains as expected BTXRD Dice.

## Decision tree after v6

1. Audit v6 and freeze all evaluated predictions.
2. If all operational goals pass, stop mechanism search and test only when the
   protocol permits.
3. If any oracle subgroup fails, selector work cannot solve it; consider a
   separately protocolled medical-foundation proposal generator.
4. If oracle passes and v6 misses, run geometry v3 only.
5. If v3 small still misses and GT-blind mass audit is material, test
   weighted-mean pooling only.
6. If small remains the isolated failure, test crop-view/global-view
   consistency.
7. If medium remains below goal with oracle headroom, test relational MIL.
8. Only after entry, train a confidence-aware multi-scale consumer with a
   boundary/affinity refinement ablation.

This ladder can improve small without surrendering medium/large because
resolution repair is isolated from ranking repair, and every promotion is
judged on all frozen subgroups with complete misses included.

## Audited source insertion points

No source was changed in this literature pass. The smallest future changes
have precise boundaries:

- `project/models/rad_dino_mask_bag_mil.py::mask_pool_descriptors` currently
  computes `mask_denominator = grid_mass.clamp_min(1.0)`. A weighted-mean arm
  changes only that denominator to `clamp_min(epsilon)` after the unchanged
  `minimum_grid_mass` validity test. It must be implemented as a new
  protocol-bound mode, not silently replace the frozen v3 behavior.
- `project/run_rad_dino_mask_bag_mil_probe.py::build_descriptor_cache`
  projects each direct-resize candidate into the square encoder frame and
  caches original/flip descriptors. A crop-view arm belongs here: derive the
  crop solely from each frozen candidate mask, encode it with frozen RAD-DINO,
  restore candidate order, and concatenate/fuse descriptors before the
  selector. Candidate masks and final WTA maps remain unchanged.
- `RadDinoMaskBagMIL.score_descriptors` is an independent selector boundary.
  Relational MIL belongs here or in a separate selector class operating on
  the same immutable descriptor cache. It must not regenerate proposals or
  alter geometry.

This separation permits reuse of one hash-bound candidate gallery and makes
the causal questions auditable:

1. geometry changes coordinates only;
2. weighted mean changes sub-token feature normalization only;
3. crop view changes spatial evidence resolution only;
4. relational MIL changes proposal interaction/ranking only.
