# BTXRD WSSS mandatory failure-analysis contract

## Purpose

This contract is a hard gate for every experiment derived from the current
best rich-gallery baseline, G1 plus equal within-image percentile-rank fusion
with the immutable upstream score (validation tumor Dice `0.2887294867`). A
failed run may not be followed by a parameter sweep, renamed mechanism, or new
GPU launch until the analysis below is complete and identifies a falsifiable
next intervention.

The test split remains locked. Validation polygons are evaluator-only and may
be opened only after predictions are frozen. Any rule discovered using those
polygons is exploratory evidence and cannot be applied to the same validation
predictions as a confirmatory correction.

## 1. Separate technical failure from scientific failure

A technical failure is a failure before a scientifically interpretable output:
packaging, path resolution, hash mismatch, read-only filesystem, odd batch,
OOM, runtime limit, corrupted output, or evaluator schema. It may be rerun only
after the exact root cause is reproduced, fixed, covered by a regression test,
and the scientific protocol is shown unchanged.

A scientific failure is an audited/frozen prediction whose metric or mechanism
gate fails. It may not be rerun by changing seed, epoch, resolution, threshold,
fusion weight, top-K, morphology, or source rule unless a new analysis predicts
which failure component the change will alter and defines a matched control.

## 2. Mandatory per-image evidence table

For every one of the 184 validation tumors, preserve at least:

- image and group identity;
- lesion-area subgroup (`<1%`, `1–<5%`, `>=5%`);
- baseline and proposed selected candidate indices and sources;
- baseline, proposed, source-oracle, eligible-oracle and gallery-oracle Dice;
- candidate rank of each oracle under baseline and proposed scoring;
- hit/miss state and transition (`hit->hit`, `hit->miss`, `miss->hit`,
  `miss->miss`);
- predicted/GT area ratio, intersection, precision, recall and Dice;
- candidate count, source presence/count, selected upstream/G1/model scores;
- flip/view agreement and every method-specific diagnostic used during fit.

Aggregates without this paired table are insufficient to authorize a next run.

## 3. Required bottleneck decomposition

### Candidate supply

Report gallery oracle, selector-eligible oracle and truncation loss. If
truncation regret is negligible, proposal generation or a larger cap is ruled
out. For misses, report oracle-Dice distribution and oracle-rank depth.

### Selector regret

Decompose mean regret exactly into:

1. wrong-source regret: selected-source oracle versus global gallery oracle;
2. within-selected-source regret: selected mask versus selected-source oracle;
3. candidate truncation regret.

Report the fraction of images whose selected source matches the oracle source,
plus source-transition counts and signed Dice mass. A source transition is not
promotable merely because it helps one validation subgroup.

### Localization versus extent

Report complete misses, point/overlap hit rate, precision, recall and
predicted/GT area-ratio quantiles. Interpret the four primary cases separately:

- no overlap: localization/ranking failure;
- overlap with very high area ratio: over-extent/background inclusion;
- overlap with very low area ratio: under-extent/discriminative-fragment bias;
- plausible area but low Dice: boundary/shape or wrong-location failure.

Small and large lesions must not be averaged into one extent diagnosis. The
known baseline ratios (`14.603` small versus `0.382` large) already falsify a
single global area correction.

### Ranking depth and recoverability

Report top-`1/3/5/10/20/50` restricted-oracle Dice, oracle-rank median/p90 and
the number of recoverable misses at Dice `>=0.1/0.3/0.5`. This decides whether
the next mechanism is a shallow reranker, deep-rank retrieval, or new
representation. Current evidence (miss oracle rank median `95`) forbids treating
a top-10-only rule as a complete solution.

### Shortcut and identifiability tests

Measure, where available, association with candidate count, source presence,
area, border, anatomy/view, flip inconsistency, and upstream score. Report image
classification separately from instance localization: high AUROC/F1 cannot be
used as evidence of good segmentation when per-image classifier score is
uncorrelated with pixel localization.

For a learned selector, compare score-quality rank correlation, effective
candidate count, original/flip selected-index agreement, and residual drift
from the frozen baseline. A shortcut reduction is only diagnostic; it does not
override worse Dice/regret.

## 4. Paired statistical accounting

Every promoted or rejected method must report overall and `94/72/18` subgroup
Dice/IoU, wins/losses/ties, positive and negative Dice mass, hit/miss transition
counts, and complete-group paired bootstrap confidence intervals. A lower miss
count does not imply improvement if lost Dice mass is larger, as demonstrated
by the top-10 relational diagnostic.

## 5. Falsification tree before the next mechanism

The dossier must answer all of the following before another GPU run:

1. What exact component improved?
2. What exact component regressed?
3. Which hypothesis is falsified, including nearby parameter sweeps that are
   therefore retired?
4. Which residual bottleneck still dominates the gap to the gallery oracle?
5. What new observable signal is absent from the failed method?
6. Why should the next method provide that signal under image-label-only
   supervision?
7. What matched control isolates that signal?
8. What numeric outcome would falsify the new method?

If these questions cannot be answered, the correct action is further analysis
or literature study, not a new run.

## 6. Current consequence for the next experiment

The central research log now contains terminal evidence that critical-instance
and orbit-averaged relational residuals (R3/R4), local affinity (R2),
same-family graph smoothing (S3), OOF clustering (S4), and count-controlled
self-paced confirmation (T1) do not improve the accepted selector. Therefore a
new relational residual would be a duplicate and is retired before compute.

The remaining justified mechanism is a genuinely class-aware localization
representation: adapt the predeclared BAS foreground-capture/background-
suppression descriptor to the rich gallery, keep the exact G1/upstream baseline
as an immutable arm, and add BAS only as a matched candidate-evidence arm.
Actual frozen binary-mask Dice against `0.2887294867` is primary. If that arm
fails, its dossier must determine whether BAS lacks lesion hit signal, duplicates
upstream extent, or introduces source/area bias before any successor is allowed.
