# G4 thesis evidence protocol

This document is the executable evidence plan for defending the final BTXRD
pipeline.  It separates published methodological precedent from project-specific
engineering choices.  No coefficient, candidate cap, source combination, or
fusion rule is described as universally optimal unless the corresponding
ablation supports that claim.

## 1. Scientific boundary

- Training supervision for WSSS is image-level only. Spatial polygons are never
  opened by training, candidate generation, G1 scoring, or choice-freezing code.
- All candidate scores and choices are frozen before the validation evaluator
  opens polygons.
- The final test was already opened once under the frozen thesis protocol. All
  new G4 ablations are validation-only; test is not used for model or
  hyperparameter selection.
- The primary task is supplied-label conditional segmentation: the known
  tumor/normal image label gates segmentation. A separate predicted-gate
  deployment analysis must report classifier errors and normal-image false
  positives.
- The primary endpoint is macro Dice over the 184 canonical validation tumor
  images. Size subgroups are computed only from native polygons and native image
  area: `<1%` (94), `1–<5%` (72), and `>=5%` (18), subject to E0 verification.

## 2. Why these metrics are scientifically defensible

### 2.1 Overlap and error-mode metrics

For image `i`, with prediction `P_i` and reference `G_i`:

`Dice_i = 2 TP_i / (2 TP_i + FP_i + FN_i)`

`IoU_i = TP_i / (TP_i + FP_i + FN_i)`

`Precision_i = TP_i / (TP_i + FP_i)`

`Recall_i = TP_i / (TP_i + FN_i)`

`RVD_i = (|P_i| - |G_i|) / |G_i|`

Macro Dice/IoU average the per-image values, giving every tumor image equal
weight. Micro Dice/IoU first pool TP/FP/FN, so large lesions contribute more
pixels. Both are reported because the two estimands answer different questions.
Precision and recall distinguish over-segmentation from under-segmentation;
median RVD and median predicted/GT area ratio quantify extent bias.

These are established segmentation measures, not project inventions. Taha and
Hanbury review medical-segmentation metrics and their competing definitions:
<https://doi.org/10.1186/s12880-015-0068-x>. The Medical Segmentation Decathlon
uses Dice and a surface metric across ten biomedical tasks:
<https://doi.org/10.1038/s41467-022-30695-9>. Metrics Reloaded recommends
problem-aware metric selection rather than relying on one score:
<https://doi.org/10.1038/s41592-023-02151-z>.

Two failure events are deliberately separate:

- `empty_prediction`: `|P_i| = 0`.
- `zero_overlap`: `|G_i| > 0` and `|P_i intersection G_i| = 0`.

A non-empty mask in the wrong anatomical location is a zero-overlap failure but
not an empty prediction. `overlap_detection_rate = 1 - zero_overlap_rate` on
tumor images.

### 2.2 Object/lesion metrics

Reference and predicted masks are decomposed with 8-connectivity. Maximum-cardinality
one-to-one matching is reported at predeclared IoU thresholds 0.10, 0.25 and
0.50. Lesion precision, recall and F1 diagnose missed or fragmented multifocal
lesions. These thresholds are diagnostics, not claimed clinical operating
points.

### 2.3 Boundary metrics

For directed surface distances `d(partial P, partial G)`:

`HD95 = max(Q95[d(partial P, partial G)], Q95[d(partial G, partial P)])`

`ASSD = mean(concat(d(partial P, partial G), d(partial G, partial P)))`

This exact convention is cross-checked against MONAI 1.5.1. It weights every
sampled surface pixel equally; it is therefore not numerically identical to the
alternative convention that gives the two directed means equal weight. The
thesis records the convention and library version because the literature and
software ecosystem contain both definitions.

Both-empty masks receive zero. Cases where exactly one mask is empty have
undefined surface distance and are excluded from conditional boundary means,
while their counts are reported. Distances are pixels on an explicitly named
grid, never millimetres: BTXRD does not provide trustworthy physical spacing.
NSD is not reported until a clinically or annotation-variability justified
tolerance is predeclared; choosing a tolerance from validation would be
post-hoc optimization.

### 2.4 Uncertainty and comparisons

- Report point estimates and 95% nonparametric bootstrap intervals, resampling
  complete canonical groups rather than individual pixels.
- For two methods on the same cohort, bootstrap the paired per-image difference
  (`comparison - reference`). Report the interval, probability of positive
  delta, and win/tie/loss counts.
- For trained arms, report all three seeds plus mean and standard deviation.
  Candidate-generation-only arms are deterministic and do not gain artificial
  “seed variance”.
- If formal p-values are requested for many ablations, predeclare the primary
  contrasts and apply Holm correction. The thesis should not turn 20 exploratory
  comparisons into 20 independent superiority claims.
- Current `group_id` is a deterministic grouping heuristic, not a verified
  patient identifier. Confidence intervals must be described accordingly unless
  patient IDs become available.

## 3. Published precedent versus project-specific design

| Component | Published precedent | What remains project-specific and therefore needs ablation |
|---|---|---|
| DenseNet-121 classifier | DenseNet, Huang et al., CVPR 2017: <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html> | Binary vs ten-class target, 320/448 input and checkpoint selection |
| CAM family | CAM, Zhou et al., CVPR 2016: <https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf>; Grad-CAM, Selvaraju et al., ICCV 2017: <https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf>; Grad-CAM++, Chattopadhyay et al., WACV 2018: <https://arxiv.org/abs/1710.11063>; LayerCAM, Jiang et al., TIP 2021, DOI 10.1109/TIP.2021.3089943 | Layer choice, attribution method and how a heatmap becomes prompts |
| Promptable segmentation | Segment Anything, Kirillov et al., ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf> | ViT-B/L/H trade-off, point/box/both prompts and medical-domain transfer |
| CAM-to-SAM WSSS bridge | S2C demonstrates a SAM/CAM WSSS relationship: <https://openaccess.thecvf.com/content/CVPR2024/papers/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.pdf> | The thesis pipeline is not S2C; its three-source gallery and exact scoring rule require its own controls |
| External biomedical evidence | BiomedCLIP: <https://arxiv.org/abs/2303.00915> | Turning saliency into one of three candidate sources |
| Frozen radiology descriptors | RAD-DINO: <https://arxiv.org/abs/2401.10815> | Inside/ring/contrast/metadata descriptor and the G1 loss |
| G1 bag learning | Attention MIL, Ilse et al., ICML 2018: <https://proceedings.mlr.press/v80/ilse18a.html> | Exact smooth pooling, auxiliary losses and candidate score interpretation |
| Rank fusion | Reciprocal Rank Fusion, Cormack et al., SIGIR 2009: <https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/> | Equal percentile-rank fusion and its 0.5/0.5 weight |

The rich gallery cap 243 is explicitly an engineering factorial upper bound:
`3 sources × 3 CAM levels × 3 components × 3 prompt modes × 3 SAM masks`.
It is not attributed to a paper.

## 4. Required experiment matrix

### E0 — evaluator correction and coordinate sensitivity

Evaluate the already frozen final WSSS and fully supervised masks on:

1. native image/polygon resolution (new primary result),
2. one common 320×320 grid,
3. one common 448×448 grid.

Report macro/micro Dice and IoU, precision, recall, RVD, empty and zero-overlap
rates, lesion metrics, conditional HD95/ASSD, native size subgroups and group
bootstrap CIs. Emit a migration table from old resized-GT subgroup to native
subgroup. Recompute and bind the official candidate oracle to candidate manifest,
evaluator commit/hash, grid, candidate count and the 184 tumor IDs, resolving
the historical 0.527902 versus 0.528298 discrepancy.

### E1 — binary versus ten-class image supervision

Use the same DenseNet-121, transforms, epochs, optimizer, split and seeds for:

- B2: normal versus any tumor.
- C10: normal plus nine disease classes.
- C10→B2: collapse the ten-class probabilities/logits into normal versus tumor
  for deployment and attribution.

The downstream C10→B2 attribution target is fixed to the exact softmax binary
log-odds identity

`log(P(any tumor)/P(normal)) = logsumexp(z_1,...,z_9) - z_0`.

It therefore does not require the true tumor subtype at inference and is
semantically matched to the one-logit binary classifier. This is probability
algebra, not a tuned project score. The comparison still uses the known binary
image label to emit empty final masks for normal cases, exactly as the final
supplied-label WSSS protocol does.

Run three fixed seeds. Report per-seed and mean±SD image AUROC, AUPRC,
sensitivity, specificity, balanced accuracy, F1, Brier score and calibration;
then run the same downstream attribution→SAM→frozen G1/rank selector and report
actual segmentation metrics. The claim “binary is better” is allowed only if the
matched downstream evidence supports it.

### E2 — attribution and prompt ablation

With one frozen B2 classifier, compare CAM, Grad-CAM, Grad-CAM++, and LayerCAM.
For each attribution map compare point, box, and point+box prompting. Include a
CAM-only thresholded-mask control. Hold SAM, gallery budget, deduplication and
selector fixed. Report classifier localization maps frozen before GT, actual
Dice, oracle, source/candidate count and runtime.

Executable status: the 4x3 attribution/prompt factorial is implemented by
`run_g4_e2_cam_prompt.py`. `evaluate_g4_pseudo_mask_variant.py` independently
verifies the 371 frozen masks and, when given the frozen candidate-summary
hash, additionally reports all-candidate oracle/Recall@Dice and the CAM-only
control. The CAM-only rule is predeclared as `prompt_map >= per-image p90`; a
constant map yields an empty mask. No morphology, SAM output, or GT-dependent
threshold is used in that control.

Result (2026-08-07): all 12 frozen-mask arms completed on the exact 371-image
validation split (184 tumor images). LayerCAM+point is best in this controlled
single-source/single-prompt experiment: tumor Dice/IoU 0.205224/0.155143,
with subgroup Dice 0.118534/0.274250/0.381836. The attribution marginal Dice
is 0.143837/0.137324/0.149826/0.193677 for CAM/Grad-CAM/Grad-CAM++/LayerCAM.
LayerCAM exceeds every other attribution method under paired complete-group
bootstrap (all corresponding 95% CIs exclude zero). Prompt marginal Dice is
0.167533/0.145091/0.155874 for point/box/box+point; the prompt contrasts do
not yet exclude zero. Box-conditioned arms systematically over-expand masks:
median predicted/GT area is about 6--10x, versus 1.145x for LayerCAM+point.
The machine-readable paired analysis is
`artifacts/final_pipeline/g4/e2_cam_prompt_factorial_results.json`. CAM-only
and proposal-oracle replay are still being added from the frozen candidate
payloads; they cannot change the selected-mask results above.

### E3 — SAM backbone ablation

Run ViT-B, ViT-L and ViT-H with identical images, prompt coordinates, multimask
setting, gallery construction and selector. Report selected and oracle Dice by
subgroup, seconds/image, peak VRAM and disk. ViT-B is defended by the Pareto
trade-off only if L/H do not yield a material improvement.

Executable status: `run_g4_e3_sam_backbone.py` changes only the explicit
official SAM-v1 model type/checkpoint. For every arm it regenerates both the
LayerCAM-320+BiomedCLIP anchor and classifier-448 addition supply, then applies
the same exact deduplication, cap 243, frozen G1 checkpoint and 0.5/0.5 rank
fusion. The spatial evaluator reports selected Dice, all-candidate oracle,
selector regret, Recall@Dice 0.10/0.30/0.50 and subgroup values. Candidate
generation also writes measured seconds/image, CUDA peak allocated/reserved
bytes and output bytes. Official B/L/H checkpoints are SHA-256 locked, and the
ablation-only evaluator rejects test.

### E4 — localization-source ablation

Run all seven non-empty subsets of LayerCAM-320, classifier-448 LayerCAM and
external BiomedCLIP saliency. For each subset report:

- candidate count median/IQR,
- candidate oracle and recall at Dice 0.10/0.30/0.50,
- fixed-G1-plus-rank selected Dice,
- subgroup Dice and selector regret,
- runtime and storage.

The first tier is fixed-selector replay. Retraining G1 per subset is optional and
must be labelled separately.

### E5 — gallery richness and budget

Balanced per-source caps are 9/27/54/81, giving total caps 27/81/162/243.
Candidates are retained by the pre-frozen upstream order, with lower original
index breaking ties. Report Oracle@K, selected Dice, truncation regret,
candidate-count distribution and resource cost.

The exact necessity study compares upstream top-1, one exact prompt,
three multimasks for that prompt, full gallery before deduplication, after exact
deduplication, and after cap 243. It requires regenerated payload metadata:
`source_id`, `CAM_level`, `component_id`, `prompt_id`, `prompt_mode`, and
`multimask_index`. The old merged artifact lacks exact `prompt_id`; a
prompt-mode-only replay must not be misreported as the exact single-prompt arm.

### E6 — G1 selector and G1 design

Selector controls on one identical eligible candidate set:

- deterministic random candidate,
- SAM predicted-IoU only,
- upstream only,
- G1 only,
- final G1 + upstream rank fusion,
- oracle (diagnostic only; never deployable).

Feature arms: inside only; inside+local ring; plus contrast; plus four metadata
features. Keep descriptor dimensionality/model capacity matched by zeroing the
removed blocks. Loss arms: bag BCE; plus negative-instance loss; plus flip
consistency. Train every learned arm for three fixed seeds and report selected
Dice, oracle/regret, image-label metrics and seed variation.

### E7 — upstream-score ablation

Recompute `D` (fraction of in-mask CAM pixels above 0.5), `M` (fraction of total
CAM mass captured), component-local SAM percentile rank `R`, and global SAM
rank. Compare U0–U6 exactly as declared in G4, including current
`0.60D + 0.25M + 0.15R`.

Critical provenance requirement: every source needs its own prompt/saliency map.
The old merged gallery stores the anchor prompt map for all sources, so E7 must
use the three original source payloads or regenerate them. Recomputing E7 from
only the merged prompt map is invalid and is rejected by design.

### E8 — fusion ablation

On frozen G1 and upstream scores compare upstream, G1, z-score sum, robust
median/MAD sum, min-max sum, reciprocal-rank fusion (k=60), and percentile-rank
weights 0.25/0.75, 0.50/0.50, 0.75/0.25. Candidate order and tie rules are
immutable. R7 must reproduce all 371 final frozen choices before evaluation.

## 5. Execution order and stopping rules

1. Run E0 first. No scientific comparison is interpreted before coordinate and
   subgroup consistency are known.
2. Run replayable E4, selector-level E6 and E8 from frozen artifacts. These are
   CPU/validation jobs and give the fastest strong evidence.
3. Recover/regenerate source-specific payloads, then run E7 and exact E5.
4. Run E1 and E2 on the same DenseNet checkpoint protocol.
5. Run E3 B/L/H, recording resources.
6. Build one reusable RAD-DINO descriptor cache, then run matched G1 feature/loss
   arms and three seeds without re-encoding images.

No arm is rescued with validation-GT thresholds, per-image area, oracle choices,
or test selection. Unexpected technical errors may be fixed without changing the
scientific configuration; efficacy failures are reported, not silently swept.

## 6. Implementation status (2026-08-07)

- Metric definitions corrected; empty and zero-overlap are distinct; symmetric
  HD95/ASSD, macro/micro overlap, extent and paired group bootstrap are tested.
  HD95 and ASSD were independently matched to MONAI 1.5.1 on synthetic cases.
- E0 completed on 371 validation images. Native/320/448 Dice differs by at most
  0.000505 for WSSS; subgroup membership remains exactly 94/72/18. The matched
  final-retrain fully checkpoint obtains Dice 0.490149 at 448.
- E4, cap replay, selector-level E6 and E8 completed for 27 predeclared arms
  (10,017 frozen choices). R7 reproduced the official common-320 result exactly.
  Random/SAM/upstream/G1-only controls obtain native Dice 0.101890/0.098902/
  0.225306/0.205545 versus 0.288224 for R7.
- E1 matched binary and ten-class runners are implemented with three seeds,
  identical DenseNet-121/320/optimizer budgets, and a shared binary-F1
  checkpoint endpoint. They report discrimination and calibration without
  spatial GT before downstream segmentation.
  Both arms completed and passed an independent output audit. Ten-class-to-
  binary improves mean NLL in every seed; paired intervals for AUROC/F1 still
  include zero. No claim about WSSS superiority is allowed until downstream
  frozen-mask Dice is available. The downstream runner now freezes the exact
  collapsed log-odds LayerCAM target and holds SAM-B, the external and
  classifier-448 supplies, G1 and fusion fixed across all six runs.
- E2 CAM/Grad-CAM/Grad-CAM++/LayerCAM attribution and point/box/point+box
  single-prompt factorial completed for all 12 arms. Every arm froze all 371
  binary masks before the evaluator opened 184 validation polygons. The best
  arm is LayerCAM+point at Dice 0.205224; LayerCAM has a positive paired main
  effect over all three alternatives. CAM-only/proposal-oracle enrichment of
  the same frozen payloads remains in progress.
- E3 end-to-end B/L/H support, candidate oracle reporting and measured resource
  telemetry are implemented and locally tested. Matched ViT-B reproduction and
  ViT-L arms are running on private/offline Kaggle T4 kernels; ViT-H follows in
  the first free slot. No E3 result is claimed before the frozen output exists.
- E7 core formulas are implemented, but execution is blocked intentionally until
  source-specific prompt maps are available.
- Exact E5, downstream E1 localization, and learned G1 feature/loss runners
  remain to be completed.
