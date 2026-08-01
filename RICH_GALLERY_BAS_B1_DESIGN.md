# Rich-gallery BAS-B1 candidate-evidence design

Status: static design; no real fit, frozen prediction, validation polygon read,
or BTXRD test access under this protocol yet.

## Scientific question

Can an image-label-only Background Activation Suppression (BAS) localizer add
tumor-specific positive-instance evidence to the current best rich-gallery
selector without changing proposal supply?

The immutable reference is G1 plus equal within-image percentile-rank fusion
with upstream proposal evidence:

- Dice/IoU `0.2887294867/0.2168391813`;
- subgroup Dice `0.1577232964/0.4352293348/0.3868735327`;
- 49 complete misses;
- gallery oracle `0.5282983322`.

The primary bottleneck is within-selected-source ranking regret (`0.1683762053`,
70.29% of total selector regret), not candidate truncation (`0.0003963`). The
49 misses are all gallery-recoverable but their oracle mask has median baseline
rank 95. Therefore the experiment keeps the complete rich gallery and changes
only candidate evidence.

## Why this is different from failed work

- R2 used frozen local-affinity summaries; BAS learns a class-aware spatial map
  end to end from image labels.
- R3/R4 used candidate-to-critical-instance relations; BAS does not anchor to a
  current winner or relation graph.
- S3 and the two rich-gallery consensus diagnostics used geometric agreement;
  BAS has no proposal-overlap term.
- S4/T1 used inferred positive candidates; BAS uses no pseudo-instance target.
- Dense normality, reconstruction and counterfactual pipelines model anomaly;
  BAS optimizes tumor-class foreground/background activation directly.

Thus the experiment tests a missing signal, not a renamed parameter sweep.

## Frozen representation and training

Adopt the tested BAS primitive from the central branch:

- ImageNet ResNet-50 initialization;
- localization output stride 8 and classification output stride 16;
- full-image cross entropy;
- foreground cross entropy weight `0.5`;
- erased-background/full activation ratio plus area penalty `1.2`;
- 224-pixel input, horizontal flip only;
- SGD Nesterov, backbone LR `0.001`, official body/head multipliers,
  momentum `0.9`, weight decay `0.0005`;
- fixed final epoch 100, total batch 32 on T4x2, seed 42;
- binary image-level normal/tumor labels only.

No validation segmentation metric, pseudo mask, candidate Dice, oracle rank,
lesion size or early stopping enters training.

For each validation image, average original and aligned horizontal-flip tumor
activation maps. For each immutable rich-gallery candidate, compute:

1. activation coverage: candidate-contained activation mass / total activation;
2. activation purity: candidate-contained activation mass / candidate area;
3. BAS evidence: harmonic mean of coverage and purity after per-image min-max
   activation normalization.

## Frozen finite selectors

Every component is converted to the same tie-aware within-image percentile
rank. There is no weight, threshold, top-K, source or subgroup search.

Primary comparison:

1. `g1_upstream_baseline`:
   `(rank(G1) + rank(upstream)) / 2`;
2. `g1_upstream_bas_three_way`:
   `(rank(G1) + rank(upstream) + rank(BAS)) / 3`.

The following are mechanism diagnostics only and cannot replace the primary arm
post hoc on this validation set:

- `bas_only`;
- `g1_bas_two_way`;
- `upstream_bas_two_way`.

All five candidate choices and all component score vectors must freeze before
validation polygons. G1/upstream baseline reproduction must be exact on all 371
images and reproduce Dice `0.2887294867` after evaluation.

## Metrics and decision rule

Actual binary candidate-mask Dice/IoU is primary. Image AUROC, sensitivity,
specificity, activation range, BAS/upstream rank correlation and selection
change are diagnostics; they do not suppress Dice evaluation because the user
explicitly requires the spatial endpoint.

Report overall and `94/72/18` subgroups, complete misses, hit/miss transitions,
source transitions, selected/GT area ratio, precision/recall, selector regret,
oracle-rank movement, wins/losses/ties, signed Dice mass and paired complete-
group bootstrap intervals.

Promotion requires the three-way arm to improve overall Dice over `0.2887294867`
without reducing any subgroup mean or increasing complete misses. A positive
point estimate with a negative paired-CI lower bound remains exploratory.

## Mandatory failure branches

If the primary arm fails, complete
`BTXRD_WSSS_FAILURE_ANALYSIS_CONTRACT.md` before another experiment and identify
one of these evidence-backed branches:

1. **BAS map lacks lesion hit signal:** BAS-only overlap/argmax and score-quality
   rank are near chance although gallery oracle is high. Retire BAS on this
   architecture; do not sweep fusion weights.
2. **BAS duplicates upstream extent:** high BAS/upstream rank correlation,
   minimal independent wins, and unchanged oracle-rank depth. Retire the BAS
   addition; do not tune its weight.
3. **BAS detects tumor but expands anatomy:** improved hit count with worse
   precision/area ratio, especially small lesions. The unresolved component is
   candidate extent, not localization; a future method must learn foreground
   versus background within candidate evidence rather than use global area.
4. **BAS detects discriminative fragments only:** improved precision but low
   recall and large-lesion under-extent. The representation needs multi-scale
   foreground expansion, not a new selector.
5. **Fusion dilution:** BAS-only/g1-BAS evidence improves cases, but equal
   three-way fusion loses stronger baseline cases. Record this as evidence for
   a train-only learned calibration or independent confirmatory fusion—not a
   validation-tuned coefficient.
6. **Source-specific failure:** benefits are confined to a proposal source or
   true lesion-size subgroup. This may inform representation training, but no
   GT-derived source/subgroup router is permitted.

BTXRD test stays locked.
