# G1 + upstream two-score identifiability dossier

## Scope and academic status

This validation-only diagnostic asks whether the remaining gap of the frozen
best baseline can be repaired by changing only the relative weight or the
monotone fusion of its two existing inputs:

`S_0.5(c) = 0.5 rank(G1(c)) + 0.5 rank(upstream(c))`.

The candidate gallery, G1 logits and upstream scores were immutable before
validation polygons were opened. The analysis reproduces baseline Dice
`0.2887294867` exactly on 184 tumors. Its alpha sweep and oracle quantities are
post-hoc explanatory bounds, not a promotable selector. No test data was read.

## Exact score-space decomposition

For each image let `O` be the best Dice among selector-eligible candidates,
`P` the best-Dice candidate on the Pareto frontier of `(G1, upstream)`, `L` the
best candidate reachable by a nonnegative linear percentile-rank fusion when
alpha ranges from 0 to 1, and `B` the frozen alpha-0.5 baseline. Then:

`O - B = (O - P) + (P - L) + (L - B)`.

- `O-P` is the score-identifiability barrier. A better mask is jointly
  dominated by another candidate in both existing scores, so no monotone
  reweighting of those scores can recover it.
- `P-L` is the residual nonlinear-frontier gap.
- `L-B` is the theoretical image-specific weighting gap. It is not realizable
  without an independent annotation-free signal that chooses the weight.

| Group | Baseline | Eligible oracle | Pareto oracle | Per-image alpha oracle | Dominance gap | Frontier gap | Weight gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 0.288729 | 0.527902 | 0.390564 | 0.381358 | 0.137338 | 0.009206 | 0.092628 |
| `<1%` n=94 | 0.157723 | 0.331101 | 0.227702 | 0.221039 | 0.103398 | 0.006664 | 0.063315 |
| `1-<5%` n=72 | 0.435229 | 0.730251 | 0.563658 | 0.557002 | 0.166593 | 0.006656 | 0.121773 |
| `>=5%` n=18 | 0.386874 | 0.746247 | 0.548689 | 0.516005 | 0.197559 | 0.032684 | 0.129131 |

Of the total eligible-selector regret `0.239173`:

- `0.137338` (57.42%) is irrecoverable by every monotone combination of G1
  and upstream because both signals jointly rank a wrong candidate higher;
- `0.009206` (3.85%) is nonlinear Pareto-frontier selection;
- `0.092628` (38.73%) is recoverable only with an oracle image-specific
  weight.

This dominance barrier is present in every lesion-size group rather than being
an artifact of the 94 small lesions.

## Image-level evidence

- The eligible oracle is absent from the two-score Pareto frontier on 154/184
  tumors; 104/184 lose at least 0.05 Dice specifically to this dominance.
- The Pareto frontier contains only 6.57 candidates per image on average, yet
  its oracle is just 0.390564. Most of the gallery's useful Dice is therefore
  outside the region visible to both current scores.
- Of the 49 frozen-baseline complete misses, only 38 have any overlapping
  candidate on the Pareto frontier and only 37 can be hit by the linear alpha
  family. At least 11-12 misses require an independent candidate signal.
- A per-image oracle alpha reaches 0.381358, but this is not a deployable rule.
  It only quantifies conditional heterogeneity that a valid image-label-only
  mechanism would need to explain.

## Shared-alpha falsification

A dense 1001-point alpha sweep is diagnostic only. The best shared alpha is
`0.632`, with Dice `0.291473`: just `+0.002743` over the frozen baseline. At
that alpha, subgroup Dice is `0.160367/0.439451/0.384220`; the large group
regresses from `0.386874`.

Therefore ordinary global weight tuning is both academically non-confirmatory
and scientifically negligible. The large per-image weight upper bound cannot
be converted into a pipeline without a new causal routing observable. The G2
failure already showed that negative-only MIL did not learn such an observable
and instead destroyed G1 complementarity.

## Consequence for improvement

The dominant bottleneck is not gallery size, candidate truncation, global
fusion weight or a mildly nonlinear transformation of the same two scores. It
is missing tumor-specific candidate identity: G1 and upstream often agree on
the wrong anatomy. This explains why consensus, relational smoothing and
metadata ablations repeatedly move masks without reliably increasing Dice.

The current BAS-B2 experiment is therefore correctly targeted: it keeps the
gallery and frozen baseline fixed and adds a third class-aware spatial signal
trained from image labels. Its decisive test is not classification AUROC but
whether BAS moves high-Dice masks onto the score frontier, reduces dominance
and within-source regret, recovers misses without worsening small-lesion
over-extent, and raises actual binary-mask Dice above 0.288729.

If BAS-B2 fails, the next analysis must determine whether BAS lacks lesion-hit
signal, duplicates the two existing rankings, detects tumor but assigns the
wrong extent, or supplies useful evidence that equal three-way fusion dilutes.
No alpha/weight sweep is authorized from this validation analysis.

