# High-resolution proposal-only repair feasibility addendum

Date: 2026-07-28

Status: conditional research design only. It does not modify the running
mask-bag MIL version-6 protocol, generate new validation predictions, or access
BTXRD test data or segmentation annotations.

## Trigger

This branch is admissible only if either:

1. the frozen candidate oracle fails an operational subgroup goal; or
2. a family-balanced relational selector later fails and the existing oracle
   margin is judged too narrow under a separately predeclared protocol.

It is not launched merely because small Dice is difficult, and it is not
needed if version 6 passes the complete localization gate.

## Prior project evidence that constrains the design

The rejected RAD-DINO global-local MIL v4 already tested a superficially
similar idea:

- the frozen global map's top-six 160-pixel ROIs intersected `93/94` small
  tumors and fully contained `88/94`;
- the local branch nevertheless reached small pixel AP `0.00516770`, p90 Dice
  `0.00507460`, and argmax hit `0/94`;
- median small local confidence was `0.98055`, and training image BCE fell from
  `0.64210` to `0.17340`;
- fused prediction improved only `3/94` small cases and degraded `23/94`.

Thus high ROI coverage did not make an image-label local dense decoder spatially
faithful. Classification confidence was a shortcut and cannot be reused as a
fusion gate.

The current full-image proposal gallery uses a 320-pixel working map and
512-pixel SAM image. Its old frozen single-candidate oracle is already
`0.40907629/0.22274968/0.59414817/0.64182777`
(overall/small/medium/large), but selection is much lower. Therefore
high-resolution generation is a headroom mechanism, not a replacement for
ranking unless a new oracle audit proves it useful.

## Distinct scientific hypothesis

> A frozen coarse map can identify a neighborhood containing a small tumor,
> while a class-agnostic boundary model operating on the original-resolution
> crop can propose a more precise raw mask without learning a local
> classification shortcut.

The output of this branch is only an expanded immutable proposal gallery. It
does not produce a deployable mask, learn a pixel decoder, or use a local bag
probability.

## Primary literature

1. Kangning Liu et al., *Weakly-supervised High-resolution Segmentation of
   Mammography Images for Breast Cancer Diagnosis*, MIDL/PMLR 143:451-472,
   2021:
   https://proceedings.mlr.press/v143/liu21b.html

   GLAM motivates coarse global ROI selection followed by high-resolution
   local analysis for lesions small relative to the image. The transferable
   element is the resolution hierarchy, not GLAM's exact learned local
   classifier.

2. Peng-Tao Jiang et al., *L2G: A Simple Local-to-Global Knowledge Transfer
   Framework for Weakly Supervised Semantic Segmentation*, CVPR 2022,
   pp. 16886-16896:
   https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html

   Local crops can expose object details absent from global CAMs. For BTXRD,
   positive image labels cannot be inherited by arbitrary crops; frozen global
   evidence supplies proposals only.

3. Yude Wang et al., *Self-Supervised Equivariant Attention Mechanism for
   Weakly Supervised Semantic Segmentation*, CVPR 2020, pp. 12275-12284:
   https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html

   SEAM motivates exact spatial alignment across transforms. The new gallery
   must record crop, resize, pad and flip transforms and compare aligned
   candidate evidence rather than rely on scalar consistency.

4. Dongjun Hwang, Seong Joon Oh, and Junsuk Choe, *Small object matters in
   weakly supervised object localization*, Neurocomputing 648 (2025), 130494:
   https://doi.org/10.1016/j.neucom.2025.130494

   Their image-label-only zoom consistency specifically targets small objects.
   The transferable element is deterministic prediction-driven zoom with
   cross-view consistency, not their natural-image localization thresholds.

5. Hyeokjun Kweon and Kuk-Jin Yoon, *From SAM to CAMs: Exploring Segment
   Anything Model for Weakly Supervised Semantic Segmentation*, CVPR 2024,
   pp. 19499-19509:
   https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html

   SAM masks can regularize weak localization rather than merely post-process
   a final CAM. For BTXRD, prompt/source agreement is retained as proposal
   provenance and later learned by the selector.

6. Xiaobo Yang and Xiaojin Gong, *Foundation Model Assisted Weakly
   Supervised Semantic Segmentation*, WACV 2024, pp. 523-532:
   https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html

   Their coarse-to-fine CLIP/SAM seeding supports using frozen semantic
   evidence to invoke a class-agnostic boundary model. The BTXRD adaptation
   uses RAD-DINO/global evidence and does not import natural-image CLIP prompts.

## Prediction-only generation contract

### Global window selection

- Use one frozen, hash-bound global map for every train and validation image.
- Apply the same deterministic top-K diverse-window algorithm to tumor and
  normal images. Do not use random windows for normals, because that creates a
  label-correlated generation path.
- Select a fixed literature/prior-protocol-motivated set of window fractions;
  do not route by true size or validation subgroup.
- Record window coordinates in content-aligned 320-space and the exact mapping
  to the original radiograph.
- If saliency is flat, use a deterministic image-independent fallback and
  report it; never silently reduce candidate count for one label.

### Local proposal generation

- Crop from the original radiograph, preserving native pixels before fixed
  resize/pad.
- Project the frozen global component boxes and positive/negative prompt points
  into the crop when they intersect it.
- Run the same class-agnostic SAM checkpoint with fixed prompt modes and
  multimask output. A fixed local grid-prompt fallback may be included, but its
  parameters must be predeclared.
- Map every local mask back to the full image through the exact inverse
  crop/resize/pad transform. Save both local and full-frame hashes.
- Keep component, prompt mode, source, ROI ID, scale and transform provenance
  aligned with every candidate.
- Apply identical generation to normal images. Normal candidates are negative
  MIL instances later; their ordinary pseudo mask remains empty.

### Gallery union

- Retain every candidate from the old hash-bound gallery.
- Append high-resolution candidates; never replace old candidates based on
  validation results.
- Exact duplicate removal is allowed only through a predeclared byte-identical
  mask rule. Similar-mask NMS or truncation is forbidden unless frozen before
  GT and scientifically justified.
- Predeclare a hard maximum with fail-closed overflow; do not rank candidates
  to fit the cap.
- Freeze the complete train and validation galleries, payload hashes,
  provenance manifests and generation metadata before importing any
  segmentation dataset.

Because the old gallery is retained, the raw GT oracle of the union cannot
decrease for any image. This monotonicity is a post-freeze diagnostic property,
not permission to use the oracle for prediction.

## Two-stage gate

### Gate P1: proposal support only

After the expanded gallery is frozen:

- verify all 371 validation payloads and exact cohort `371/184/187`;
- evaluate best-single-candidate oracle including every complete miss;
- report overall/small/medium/large oracle Dice and paired per-image delta
  versus the old gallery;
- report small complete misses and the fraction whose best candidate comes
  from the new high-resolution source;
- use paired complete-group bootstrap with 10,000 replicates.

P1 requires a predeclared material small-oracle improvement and no corruption
of old-candidate hashes. Passing P1 does not yield a deployable result.

### Gate P2: deployable selection

Only after P1 passes may the family-balanced relational selector consume the
expanded gallery. Its prediction manifest/checkpoint must again be frozen
before GT evaluation. The complete operational goals, no-subgroup-decrease,
complete-miss and paired-bootstrap requirements remain decisive.

No high-resolution candidate oracle may be reported as a deployed pipeline.

## Difference from rejected global-local v4

| Rejected v4 | Proposal-only repair |
|---|---|
| Learned a free local pixel decoder from image BCE | Learns no local pixel decoder |
| Used local bag confidence as a fusion gate | Has no local bag confidence |
| Added sparse local residual values to global map | Appends raw class-agnostic masks only |
| Could place hot pixels anywhere inside an ROI | Output is constrained to SAM proposal shapes |
| Evaluated fused/local continuous maps | First evaluates immutable proposal support |
| Normal training used label-dependent random ROIs | Uses identical deterministic window logic for both labels |

The new branch is therefore not a retuned version of the rejected experiment.

## T4x2 execution strategy

The full-image gallery routed the classifier to `cuda:0` and SAM to `cuda:1`.
This branch consumes frozen global maps and needs no classifier forward pass.
The fastest admissible design on the user's available hardware is:

- load one identical hash-verified SAM checkpoint on each T4;
- shard images, including all their ROIs, deterministically across devices;
- keep one image's proposal order deterministic independent of worker timing;
- write per-device temporary manifests and merge only after exact
  image/provenance/hash coverage checks;
- record two real T4 device names and run a convolution/proposal smoke test on
  each;
- never describe two model replicas as data parallel unless the implementation
  actually synchronizes them.

Candidate generation remains heavy and must run on Kaggle. CPU-only manifest
and source audits may remain local.

## Required tests before any launch

1. Same image/global map yields identical ROI order across runs.
2. Tumor and normal rows execute the same window-selection code path.
3. Crop/inverse mapping round-trips corners and single-pixel masks exactly.
4. Horizontal flip maps ROI coordinates and restored masks exactly.
5. Local prompt points/boxes remain inside the crop after projection.
6. Candidate/provenance arrays have identical lengths and stable ordering.
7. Old candidate payload hashes are unchanged in the union.
8. Exact-duplicate removal cannot remove two non-identical masks.
9. Overflow fails before prediction freeze and before GT access.
10. Both T4 devices perform real work and output disjoint deterministic image
    shards.
11. All train/validation payloads and maps are finite, bounded and complete.
12. The evaluator cannot import GT until gallery freeze verification passes.
13. `consumer_trained=false` and `test_evaluated=false` are recorded.

## Decision

Prepare no implementation or Kaggle launch until version-6 terminal evidence
selects this branch. If activated, build the smallest proposal-only experiment
that tests resolution headroom; do not combine it immediately with dropout,
affinity, a consumer, or a new foundation checkpoint.
