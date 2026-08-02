# Cross-view co-witness candidate representation

Status: **hypothesis and bounded diagnostic design frozen for implementation
review**.  This document does not authorize a Kaggle launch until candidate
payload availability, code, tests, memory/runtime preflight, provenance and
no-GT/no-test audits pass.

## Decision target

The immutable deployed reference is G1 plus upstream equal percentile-rank
fusion at validation Dice/IoU `0.2887294867/0.2168391813`.  The new model may
only add a zero-initialized residual to this score.  Actual binary-mask Dice,
not image AUROC or a similarity proxy, is the promotion endpoint.

## Why a new representation is required

The rich gallery already supplies eligible oracle Dice `0.5279020259`; proposal
supply loses only `0.0003963063`.  Conditional analysis of all 32,519 frozen
candidates found no material tumor-identity residual after controlling G1,
upstream, area and source:

- transition2 matched relative contrast: partial correlation `0.039274` and
  oracle-above-baseline `0.491429`;
- transition2 matched-minus-random contrast: `-0.010044`;
- matched-minus-random terminal logit: `0.003547`.

Therefore this design does not append cross-view cosine or another frozen
descriptor rank.  It introduces a new training relation that earlier binary
MIL did not have: two radiographs believed to show the same lesion from
different views.

## Available annotation-free relation

The canonical train split contains:

- 1,488 tumor images;
- 443 heuristic tumor groups with at least two distinct views;
- 1,027 tumor images in those groups.

Every such group has one anatomy and one tumor-type value in the canonical
manifest.  A visual audit of representative pairs confirms that the two views
can depict the same bone and gross lesion morphology.  The limitation is
explicit: BTXRD publishes no patient/case identifier; `group_id` is a
consecutive-image and stable-metadata heuristic.  Pair supervision is therefore
noisy and must be compared with a matched different-group control.

The existing frozen RAD-DINO candidate descriptors do **not** solve the task:

- 13 distinct-view validation groups, 26 oriented query images;
- same-group contrast-descriptor quality correlation median `0.415869`, but
  eligible oracle is above baseline only `46.15%`;
- among 16 queries with same-anatomy/type/view controls, matched-minus-control
  partial correlation is only `0.014121` and the oracle beats baseline on only
  `18.75%`.

Thus direct cross-view similarity fusion is retired.  The relation is retained
only as potential supervision for learning a different representation.

## Proposed mechanism

For image `x_a` and candidate `c`, a shared high-resolution encoder produces
candidate descriptor `z_a(c)` from separately pooled candidate interior,
local ring and their difference.  There are no absolute coordinate channels
and no source/count metadata in the learned residual.

A unary head outputs `r_a(c)`.  The deployed candidate score is

`s_a(c) = b_a(c) + eta*tanh(r_a(c))`,

where `b_a(c)` is the immutable centered G1/upstream score.  The last residual
layer is exactly zero initialized, so the initial/fallback selector reproduces
Dice `0.2887294867` bit-for-bit.

For two different views `a,b` in one heuristic group, define a co-witness bag:

`P(a,b) = LSE_(c,d) [r_a(c) + r_b(d) + beta*cos(q(z_a(c)),q(z_b(d)))]`.

Training uses four fixed terms:

1. **Normal dense negatives:** every candidate from a normal image must have a
   non-positive tumor residual.
2. **Tumor image bag:** each tumor image must retain positive bag evidence; no
   detached top-1 winner becomes a pseudo-label.
3. **Same-group co-witness margin:** `P(a,b)` must exceed `P(a,b_ctrl)`, where
   `b_ctrl` is a different group with the same anatomy, tumor type and target
   view whenever available.  This is the new positive-instance constraint.
4. **Style/source controls:** horizontal flip/intensity transforms preserve
   unary order; source-balanced pooling and residual-area covariance prevent
   candidate count, source and mask mass from becoming shortcuts.

Cross-view input is training-only.  Inference remains prompt-free and operates
on one image using `s_a(c)`.

## Why this is not a repeated failed family

- **Mask-bag/G2:** those objectives could satisfy a positive image bag by
  reinforcing one self-selected wrong candidate.  Here a positive co-witness
  must survive in a second real projection and beat a matched different-case
  pairing.
- **Consensus/top-k relation:** those selected repeated mask geometry within
  one image.  Cross-view co-witness uses candidate appearance across independent
  acquisitions; raw overlap/area is not a pair feature.
- **Matched-normal transplant/NRCE/density:** those were post-hoc interventions
  on frozen representations.  This design retrains the representation and does
  not use rarity, reconstruction residual or a frozen classifier response.
- **OLV:** no absolute coordinate channels, border crop identity or local-only
  image prediction is available.

Adversarial erasing literature establishes that classification features often
cover only discriminative parts and that feature erasure can expose
complementary regions.  PYLON establishes that medical weak localization needs
high spatial resolution but resolution alone is insufficient.  These are
mechanism precedents, not BTXRD performance guarantees:

- ACoL, CVPR 2018:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.html
- EIL, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Mai_Erasing_Integrated_Learning_A_Simple_Yet_Effective_Approach_for_Weakly_CVPR_2020_paper.html
- PYLON:
  https://arxiv.org/abs/2010.11475
- Correct WSOL evaluation, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html

## Bounded matched diagnostic

Run exactly two canonical passes in two equal-capacity arms:

- **full:** true heuristic different-view partner;
- **control:** different-group partner matched on anatomy, tumor type and view.

Both arms freeze all validation candidate residuals and every candidate choice
before polygons.  To avoid an unnecessarily strict single residual scale, the
prediction-first stage predeclares multipliers `0.25/0.5/1.0/2.0` for both
arms.  Stage B reports every variant and may select one *global* development
multiplier; this is explicitly exploratory validation tuning, never a
confirmatory test estimate.  No per-image, subgroup-specific or GT-area
routing is allowed.  Stage B always reports actual Dice/IoU, 94/72/18
subgroups, misses, source regret, within-source regret and selected/GT extent.

The primary development decision is deliberately not an AND gate over proxy
metrics.  A longer run is justified when a pre-frozen full-arm variant beats
both `0.2887294867` and its multiplier-matched control in actual Dice.  The
following remain diagnostics and safeguards rather than automatic vetoes:

1. subgroup Dice and any small/medium regression;
2. within-selected-source regret versus complete-miss changes;
3. full-control same-group co-witness margin, residual-area correlation and
   source concentration.

Only the academic validity checks remain hard: zero-residual baseline
reproduction, exact group/split hashes, prediction-before-GT, no spatial GT in
training/selection and no test access.

If full equals control, or only paired validation images improve, group
supervision did not produce a generalizable tumor representation and this
family is retired without epoch, weight, seed, resolution, source or threshold
sweeps.

## Provenance of the feasibility decision

- Rich-gallery byte-exact split SHA-256:
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`.
- Later canonical split SHA-256:
  `7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c`.
- The semantic-bridge audit compares all 3,746 IDs and 32 shared scientific
  fields exactly.  Only the volatile dataset-table hash differs and the later
  file adds `dataset_table_semantic_sha256`; audit SHA-256:
  `c711116aeafae2d76fe0f4c9e25efe16216526327357c7fdba47ac96188fadc9`.
- Train-only pair manifest SHA-256:
  `0950ed5063e932e3988b97fff590945ecaf37d27fbce84a50ba216e2045ead8a`.
- G1 diagnostic freeze SHA-256:
  `c4e80a0c9bd8a1d4e5ef6204d23123d2d4f7b4deabb4c4b38aa4578b8b899e1c`.
- Cross-view feasibility JSON SHA-256:
  `2ff77b11c0acc9e1d215bd246aaa80a29467a2bc3785243d7b97fa05db9f40be`.
- Per-query feasibility table SHA-256:
  `6d3f471916d77ca1fd58fa3356fd62a268a36f7c5bd4d142d2a376022c243e14`.
- Validation GT was used only retrospectively after both source freezes; test
  was not opened.
