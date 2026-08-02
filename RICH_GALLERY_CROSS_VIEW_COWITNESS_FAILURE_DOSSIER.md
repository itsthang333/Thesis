# Rich-gallery cross-view co-witness failure dossier

## Decision

The cross-view co-witness residual is retired as a standalone selector and as
an annotation-free scale-gate feature in its current form.  The best valid
prompt-free image-label-only validation result remains the immutable
G1+upstream fixed percentile-rank fusion:

- Dice/IoU `0.2887294867/0.2168391813`;
- subgroup Dice `<1% / 1-<5% / >=5%`
  `0.1577232964/0.4352293348/0.3868735327`;
- gallery/eligible oracle Dice `0.5282983322/0.5279020259`.

The Kaggle run completed and the independent Stage-A audit passed exact 371
validation score payloads, 3,339 choices, 384 pair rows, two 384-step epochs
per arm, exact G1 reproduction, no spatial-GT training/selection and no test.

## Actual binary-mask endpoint

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| Immutable baseline | **0.288729** | **0.216839** | **0.157723** | **0.435229** | 0.386874 | 49 |
| Control residual x0.25 | 0.287912 | 0.215879 | 0.149947 | 0.437334 | 0.410709 | 48 |
| Full residual x0.25 | 0.287911 | 0.215879 | 0.149945 | 0.437334 | 0.410709 | 48 |
| Full residual x0.5 | 0.271524 | 0.204147 | 0.120298 | 0.432841 | 0.415987 | 45 |
| Full residual x1 | 0.239372 | 0.180161 | 0.072177 | 0.411341 | 0.424625 | 41 |
| Full residual x2 | 0.235867 | 0.178325 | 0.061764 | 0.415979 | 0.424625 | 38 |

The exploratory best full arm loses `0.000819` overall and is effectively
identical to the multiplier-matched control (`-0.00000072`).  Full x0.25
versus baseline has complete-group bootstrap 95% CI
`[-0.017278, 0.016311]`; no subgroup establishes improvement.  A longer run
is not supported.

## Root cause 1: the head learned source escape, not candidate identity

The residual head saturates negatively on every trainable proposal source:

| Arm/source | Mean residual | Negative fraction | `<=-0.45` fraction |
|---|---:|---:|---:|
| Control classifier448 | -0.497172 | 1.000 | 1.000 |
| Control LayerCAM320 | -0.496343 | 1.000 | 1.000 |
| Full classifier448 | -0.494875 | 1.000 | 0.972 |
| Full LayerCAM320 | -0.491549 | 1.000 | 0.946 |
| External saliency, both arms | 0 | 0 | 0 |

External-saliency candidates were intentionally held at zero residual because
their appearance descriptors were unavailable.  The optimizer therefore found
a cheaper solution: suppress almost every classifier/LayerCAM candidate and
escape to the unpenalized external source.  External selections rise from
`24/184` at baseline to `67`, `112`, `173` and `184` as the residual multiplier
increases.

This is not a harmless source shift.  For `<1%` lesions the median selected/GT
area ratio grows from `14.603x` to `16.916x`, `23.226x`, `52.709x` and
`64.760x`; Dice consequently falls from `0.157723` to `0.061764`.  For large
lesions, whose baseline masks are under-extent (`0.382x` GT), the same broad
external masks raise Dice to `0.424625`.  The residual therefore reproduces
the already-known opposite scale effect rather than learning tumor identity.

## Root cause 2: the positive relation barely separates from the control

For margin loss

`softplus(negative - positive + 0.2)`,

the epoch-2 control loss `0.797691` implies positive-minus-negative separation
only `0.000814`; the full loss `0.775446` implies `0.041663`.  Thus the
same-group/different-view relation contains a weak relational signal, but not
enough candidate-local information to overcome the common bag/normal losses.

At candidate level:

- full/control residual Pearson correlation is `0.998657` over all candidates;
- mean absolute full-control difference is only `0.003738` versus residual
  range `[-0.5,0]`;
- full and control choose the same tumor candidate in `182/184` images at
  x0.25, `183/184` at x0.5 and `184/184` at x1/x2;
- the full residual has mean within-source Spearman `-0.182606` with candidate
  Dice; the control-subtracted difference improves this only to `+0.064174`.

The relation can distinguish paired bags slightly, but it does not identify
which proposal within a bag is the tumor.  Normalized pair LogMeanExp over up
to `243 x 243` candidate pairs also permits shared anatomy/background pairs to
carry the bag relation.  This is precisely the instance-identifiability gap
that image-level MIL alone cannot resolve.

## Causal-control subtraction salvage

Because the matched control exposed the shortcut, a post-freeze diagnostic
tested

`delta_causal = residual_full - residual_control`.

All choices were frozen before validation GT.  Two policies were evaluated:

1. add a shared raw multiplier of `delta_causal` to the centered baseline;
2. keep the baseline-selected source fixed and rerank only within that source
   by the percentile rank of `delta_causal`.

Neither beats the baseline.  The closest raw policies x1/x2 reach `0.287737`;
the closest source-locked policy x0.05 reaches `0.287508`.  Larger weights
degrade monotonically, reaching `0.213495` and `0.234829`.  Therefore the weak
`+0.064` conditional rank signal is real but too small/noisy to be a selector
or a scale gate.

## What this adds to the bottleneck model

The candidate gallery is not the bottleneck: proposal-supply regret remains
only `0.000396`.  The unresolved gap is now separated into three problems:

1. **candidate identity:** eligible selector regret is `0.239173`, of which
   `0.168376` (70.4%) is inside the selected source;
2. **source choice:** `104/184` baseline choices use a different source from
   the eligible oracle, contributing `0.070796` regret;
3. **signed extent:** small needs shrinkage, medium mainly needs identity, and
   large needs expansion.

Separate group-dependent corrections are mathematically supported.  The
retrospective fixed experts reach `0.181215/0.435229/0.501086` and true-group
routing would reach `0.311904`; their per-image oracle is `0.350692`.  But GT
group routing is forbidden, and proposal-area gates score only
`0.268845-0.278291`.  The next bottleneck is therefore an annotation-free
latent-burden gate **plus** candidate-specific identity evidence.  One global
correction is ruled out; three experts without a reliable gate are also
insufficient.

The next representation must not pool a whole candidate-pair Cartesian
product into one bag relation.  It must expose sparse local correspondence or
independent evidence support at high resolution, use dense normal-image
patches as negatives, and train the scale gate separately from the candidate
residual.  Any successor must retain zero-residual reproduction of the
`0.288729` baseline and prove added information beyond a capacity-matched
control before a full run.

## Provenance

- Original prediction freeze SHA-256:
  `0a5c229390c9bf0a4d2f8c96db20580d8acf4ef3497309d677c63e376eac9f4d`.
- Independent Stage-A audit SHA-256:
  `0364ba4970bb6363366fcc2eb1e62cc852d6df866d5500b5e3cecbe5d5ab3134`.
- Original Stage-B summary/per-image SHA-256:
  `3d5f285e9f1c6940f556d0987343a2101bc58fb6c81c7b77201d71c871854798` /
  `66ea8a1429afdb9520cb206ecb6c750a6d41c97640d5d24a4e4fb3bb802be982`.
- Failure-analysis summary/per-image/audit SHA-256:
  `3dee5cc1af9c484a9be173a2da5e2a9a7c7fdd5bde704af7c74e1d42034ffb94` /
  `791fe3b83792a20aaa7f8f44d08c12265bdf3bb50d69b5d492bd410cf187b58e` /
  `841996fc5897ef7bf6e0611508482eec699fc3b5d07e4772b64be30297fc2a5f`.
- Control-contrast Stage-A freeze SHA-256:
  `9c43b48bce02b61fc047b81b3293554ee8615fe44edb3c637d81079e31b8ba5b`.
- Control-contrast Stage-B summary/per-image/audit SHA-256:
  `b3f28c489eba437a597afad27fc9eaf01f94309ca3d75f406ab943b2945d8d2a` /
  `1c0efc2c9642d95d687c376ec18ffa5efecc2e2ed35b47811a5187df07d19b65` /
  `411bf2203169f738416d8475478ceb1792eb3298016def8f6fade80187247b11`.
- Exact cohort: 371 validation images, 184 tumors, subgroups 94/72/18.
  Test was not read or evaluated.
