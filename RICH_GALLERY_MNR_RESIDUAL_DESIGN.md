# Matched-normal local-evidence residual for the rich gallery

Status: **PAUSED / superseded before GPU launch** by
`BTXRD_DEEP_BOTTLENECK_SYNTHESIS_20260802.md`.

The original MNR-v1 experiment must not be launched unchanged.  The deep
dataset/output review found two unresolved repetitions of prior failure modes:

1. binary positive-bag MIL still does not identify which local instance is the
   lesion, the exact mechanism that limited G2;
2. the top-17-inside-minus-ring candidate readout measures local identity but
   is largely insensitive to candidate extent, although extent errors have
   opposite signs for small and large lesions.

The reusable reference cache, stride-4 implementation and control/full
infrastructure remain valid.  The successor must add subtype-conditioned
intra-class local discrimination and expose separate identity and normalized
extent-compatibility readouts.  This document is retained as an historical
design record, not launch authorization.

## Decision question

Can a newly learned stride-4 map of query-versus-normal local evidence reduce
candidate-selection regret when added as a baseline-preserving residual to G1
plus fixed percentile-rank fusion?

This is an improvement experiment, not another attempt to verify the already
reproduced `0.2887294867` baseline.

## Why this mechanism remains scientifically distinct

The retired mechanisms all read the old representation or let proposal/source
geometry dominate:

- G2 negative-only MIL had no positive-instance identifiability;
- mask-bag/Geometry used frozen proposal descriptors and detached winners;
- matched-normal transplant measured a frozen classifier response and mainly
  transferred mask area;
- cross-view co-witness learned a proposal-source shortcut;
- latent-size gates could not infer true burden from annotation-free bag
  statistics;
- raw consensus exposes lesion coverage but is far too imprecise to supervise
  a consumer.

The successor instead learns a new dense representation end to end.  Every
query cell is compared with locally soft-matched cells from four appearance-
matched, same-view, train-normal radiographs.  The image prediction has no
global bypass: it is aggregated from sparse local evidence.  Normal images
provide clean negative cells, while tumor images provide only a positive bag
label.  Candidate masks and proposal source are absent from representation
training, preventing direct source and proposal-area shortcuts.

This is the previously implemented MNR-MIL mechanism, now used for the exact
measured bottleneck: it supplies a new candidate-local residual rather than
replacing the rich gallery with a thresholded dense map.

## Reused work — no rebuild

- exact canonical train/validation population: `2,981/371`;
- `1,493` train-normal reference pool;
- frozen `3,352 x 1,024` descriptor cache and `26,816` top-eight reference
  assignments;
- ranks 1-4 primary, ranks 5-8 reference-swap control;
- DenseNet121-FPN input 512, output stride 4 (`128 x 128` evidence);
- fixed two-pass protocol, terminal step `2,986`;
- query-only capacity-matched control and matched-normal full arm.

The existing reference cache was already independently reproduced exactly and
contains no spatial GT or test access.  It will not be regenerated.

## Candidate residual

Let `b(c)` be the immutable centered G1/upstream percentile-rank score of
candidate `c`, and let `E(x)` be the learned dense MNR evidence map.  Candidate
evidence is computed without source identity:

1. invert the fixed letterbox transform to the gallery grid;
2. take the mean of the highest `min(17, |c|)` valid evidence cells inside
   candidate `c`;
3. subtract the median evidence in a fixed local ring around `c`;
4. convert the resulting values to a within-image percentile rank.

The candidate selector is

`s(c) = b(c) + 0.25 * centered_rank(E_c)`.

The coefficient is single and fixed before validation polygons.  A zero MNR
map exactly reproduces the baseline.  There is no source embedding, proposal
area term, subgroup router, per-image threshold, SAM rescue or morphology.

The query-only control uses the identical readout and coefficient.  Therefore
an improvement by the full arm over both baseline and control is attributable
to matched-normal local evidence rather than extra capacity or another fusion
sweep.

## Streamlined execution

Only four checks block the run:

1. canonical split and exact reference-cache hashes match;
2. training reads image labels but no spatial GT/test;
3. a one-batch real-data forward/backward is finite and fits GPU memory;
4. zero evidence reproduces all 184 tumor baseline choices exactly.

After those checks, run control and full arms in parallel on private/offline
Kaggle T4 GPUs for exactly two passes.  Freeze 371 continuous maps and gallery
choices, then open validation polygons once and report actual binary-mask
Dice/IoU overall and `94/72/18` subgroups.  AUROC, F1, border and swap
correlation remain explanations only; they cannot veto Dice evaluation.

Promotion is deliberately simple:

- report every result, regardless of proxy metrics;
- promote only if full actual Dice exceeds `0.2887294867` and exceeds the
  matched query-only control;
- subgroup changes and paired uncertainty are reported, not used to hide an
  overall gain;
- test remains locked.

If this bounded run fails, do not extend epochs or sweep the residual weight.
The interpretation will distinguish map localization failure from gallery
readout failure; the normal-reference representation family is then retired.
