# Post-v6 mechanism decision matrix

## Scope

This note orders the next BTXRD image-label-only WSSS experiments. It does not
authorize a launch, change the frozen validation protocol, open test data, or
permit validation subgroup identity in training. The terminal v6 audit remains
the decision point.

## Evidence that determines the branch

Four quantities must be separated:

1. **Candidate support:** immutable-gallery oracle Dice and complete misses.
2. **Selector quality:** winning-mask Dice relative to that oracle.
3. **Consumer readiness:** the frozen prediction-first entry gate.
4. **Failure morphology:** GT-blind proposal/token statistics first, followed
   by the already-declared post-freeze validation diagnostics.

The current frozen evidence gives current/oracle Dice
overall/small/medium/large of
`0.23366822/0.11152529/0.34768577/0.41545552` and
`0.40907629/0.22274968/0.59414817/0.64182777`. Therefore medium currently has
the largest selector-recovery burden, while small has the least absolute oracle
headroom. An aggregate improvement is not enough to choose a branch.

## Mechanism matrix

| Terminal evidence | Smallest causal next arm | Why it matches | Why other arms wait |
|---|---|---|---|
| Any subgroup oracle fails after the exact geometry correction | Proposal-support arm only | A selector cannot choose a mask absent from its gallery | Relational MIL and a consumer cannot repair missing support |
| Oracle passes all goals, selector misses, and fractional projected mass is materially common | True weighted-mean descriptor pooling only | Removes the current `clamp_min(1)` attenuation without changing masks or supervision | Tiling and relational scoring would confound the pooling test |
| Oracle passes; small alone remains below goal after corrected pooling | Fixed global + four overlapping local views | Local evidence raises effective spatial sampling while preserving image-level bag supervision | Per-proposal crops are computationally unbounded; size-balanced loss would require forbidden subgroup/size knowledge |
| Oracle passes; medium remains below goal with clear selector headroom | Zero-initialized relational MIL residual | Candidate-to-critical-instance relations target ranking rather than proposal recall | Activation expansion changes support and is not a clean selector test |
| Predictions pass the frozen consumer-entry gate | Confidence-aware segmentation consumer | Boundary/coverage refinement is then supported by prediction-first evidence | Training it earlier would hide whether the selector itself works |

## Transfer assessment of newer literature

### Small-object evidence

Mun et al. show that image-label WSSS methods systematically underperform on
small objects and that aggregate scores conceal the failure. Their
size-balanced loss is not directly admissible here because BTXRD size subgroup
membership is derived from dense validation truth and cannot supervise
training. The transferable parts are subgroup-complete evaluation and the
motivation for a GT-blind local-resolution arm.

Primary source:
https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

### Activation expansion

FSAE expands reliable activations toward weakly activated and boundary regions
using pseudo-labels plus weak/strong consistency. This addresses CAM
under-coverage, but uncontrolled expansion is poorly matched to a candidate
gallery whose large subgroup must not regress and whose immediate uncertainty
is selection versus support. It is therefore a later proposal/consumer
contingency, not the next selector arm.

Primary source:
https://openaccess.thecvf.com/content/WACV2025/html/Choi_Feature-Level_and_Spatial-Level_Activation_Expansion_for_Weakly-Supervised_Semantic_Segmentation_WACV_2025_paper.html

### Dataset-level bias modelling

BRNF models dataset-wide pixel-feature distributions with normalizing flows
and a Gaussian-mixture classifier to counter both under- and over-activation.
The general lesson—model positive and background feature distributions rather
than trusting a discriminative peak—is relevant. The full method is not a
minimal transfer to the current binary radiograph candidate selector and adds
substantial trainable machinery. It remains a later representation arm only
if geometry, pooling and relational selection fail under oracle feasibility.

Primary source:
https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html

### Foundation-model assistance

FMA-WSSS couples frozen CLIP with learned task prompts and SAM-based coarse and
fine seeds. It supports the general proposal-first use of a frozen foundation
model, which the current SAM/RAD-DINO gallery already instantiates. It does not
justify adding CLIP before measuring whether the present gallery oracle or
selector is the bottleneck.

Primary sources:
https://arxiv.org/abs/2312.03585 and
https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html

ExCEL uses patch-text rather than only global image-text alignment and adds
static/learnable visual calibration. Dense alignment is a useful design
principle, but the binary label "bone tumor" has less natural-language
specificity than the natural-image categories used in the paper. RAD-DINO and
SKELEX are better domain-matched first choices. ExCEL is therefore a
domain-gap contingency, not a priority arm.

Primary sources:
https://arxiv.org/abs/2503.20826 and
https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Exploring_CLIPs_Dense_Knowledge_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.html

### Medical image-level precedents

Kuang et al. use multi-level features, semantic affinity and cross-guidance in
image-label-only medical segmentation. The transferable point is that
affinity-guided feature decomposition can supply information absent from a
single discriminative classifier. Their multi-class label-symbiosis problem
does not directly match binary BTXRD, so their complete architecture and
reported metrics are not transferred.

Primary source:
https://pubmed.ncbi.nlm.nih.gov/38422964/

Viniavskyi et al. combine classification maps, an inter-pixel relation network
and a downstream segmentation model for chest radiographs. This supports the
ordered seed-to-relation-to-consumer pattern, but the consumer remains
ineligible here until the frozen entry gate passes.

Primary source:
https://arxiv.org/abs/2007.00748

## Frozen decision order

After the terminal v6 evidence is preserved:

1. run the already-frozen exact descriptor geometry correction alone if v6
   misses and the corrected candidate oracle remains feasible;
2. use the GT-blind fractional-mass audit to decide whether weighted-mean
   pooling is warranted;
3. use local views only for a remaining isolated small-support failure;
4. use relational MIL only for a remaining oracle-feasible medium-selection
   failure;
5. consider SKELEX under its already-audited support/selection branch;
6. train no segmentation consumer until its entry gate passes.

BRNF, FSAE and CLIP/ExCEL are reserve hypotheses. They must not be bundled with
the above arms merely because they are newer. Each future experiment must
freeze predictions before validation GT evaluation, include complete misses,
report all subgroups, and keep BTXRD test locked.
