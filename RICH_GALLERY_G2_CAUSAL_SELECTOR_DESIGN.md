# Rich-gallery G2 causal selector correction

Status: frozen scientific design before G2 training or G2 validation Dice.
The only spatial validation evidence used to motivate this design is the
already-audited G1 Stage-B/follow-up evidence listed below. No G2 choice,
coefficient, checkpoint or epoch may be selected from validation polygons.

## Question

Why did proposal oracle Dice rise from about `0.4091` to `0.5279` while the
trained selector fell from `0.2455` to `0.2060`, and can a correction recover
the new proposal support without using spatial supervision?

## Measured failure, not an architectural guess

The rich-gallery result is explained by four jointly measured mechanisms.

1. **The gallery changed by much more than its oracle.** The old gallery has
   about `56` retained proposals per validation image; the rich gallery has
   about `150`, and tumor bags average `176.7`. The positive witness rate is
   therefore diluted even though the best available proposal is better.
2. **One source leaks the image label through bag composition.** External
   BiomedCLIP proposals occur in `184/184` tumor bags and `0/187` normal bags
   because known-normal external maps were deliberately written as zero. A
   candidate-count-only classifier reaches image AUROC `0.89449`, above the
   trained selector's approximately `0.865`. The selector can solve image
   classification without identifying a lesion.
3. **Normalized SmoothMIL still has a nearly max-pool gradient.** For
   temperature `tau=0.2`,

       z_b = tau * (log sum_i exp(z_i/tau) - log n)
       d z_b / d z_i = softmax(z_i/tau).

   The median effective candidate count is only `1.63`; the selected proposal
   receives median gradient weight `0.756`, while the oracle receives median
   weight `8.14e-15`. Normalizing by `log n` corrects the logit offset but not
   gradient concentration.
4. **Detached top-1 confirmation amplifies the first wrong winner.** From
   epoch three onward the old objective labels only its current argmax as a
   positive instance. The selected proposal has Dice below `0.1` in `60.9%`
   of tumor images and `83.0%` of the `<1%` subgroup. Thus the pseudo target is
   usually wrong exactly where localization is hardest.

The regret decomposition excludes a source-only explanation: total
selected-to-oracle regret is `0.32188`; source choice contributes `0.08236`
(`25.6%`) and wrong rank/extent inside the selected source contributes
`0.23951` (`74.4%`). Median selected/GT area ratio is `13.0` overall and
`46.6` for `<1%` lesions. Dice versus absolute log-area error has Spearman
`-0.705` overall, `-0.865` for medium and `-0.957` for large lesions.

The post-freeze GT-blind follow-up supplies a positive mechanism signal:
equal percentile-rank fusion of G1 and the frozen upstream coverage/purity
score raises actual Dice from `0.20603` to `0.28873` (`0.15772/0.43523/
0.38687` by the fixed `94/72/18` subgroups). It simultaneously shrinks median
selected area from `4.77%` to `2.12%`, but complete misses rise from `29` to
`49`. G1 therefore provides useful hit/localization evidence, while upstream
provides complementary extent/purity evidence. Replacing either one is
incorrect; they must be combined.

## Immutable supervision boundary

- Training inputs: canonical train images, binary image labels, frozen
  class-agnostic proposal masks, proposal-source identities, frozen RAD-DINO
  descriptors, horizontal-flip descriptors and upstream GT-blind scores.
- Validation Stage A: freeze all candidate logits, indices and binary selected
  masks for exactly `371` images before importing any polygon reader.
- Validation Stage B: evaluate those immutable choices on exactly `184` tumor
  masks and the fixed `94/72/18` subgroups.
- No train/validation polygon, lesion area, oracle rank, per-image router,
  morphology, threshold sweep, SAM rerun, pseudo-mask consumer or test input is
  available to the trainer.

## Matched arms

All arms use the same 1,156-D descriptors, MLP capacity, initial weights,
train order, optimizer, 16 final-only epochs and flip consistency. External
proposals are excluded only from the training loss because their source
presence is a perfect label shortcut. They remain eligible at inference and
are scored by the source-agnostic shared MLP.

1. `flat_shared_hardtop`: source-safe control. It retains the old `tau=0.2`
   flat SmoothMIL and delayed detached top-1 instance loss. This isolates the
   effect of removing the external bag-composition leak.
2. `flat_shared_negative_only`: same flat pooling, but removes all invented
   positive instance labels. Every proposal in a normal image remains a
   reliable negative; candidates in positive bags are supervised only through
   the bag label. Difference from arm 1 isolates hard top-1 confirmation.
3. `hierarchical_shared_negative_only`: safe negative-only objective plus
   equal source-mass pooling. For shared source `s`,

       u_s = tau * (log sum_{i in s} exp(z_i/tau) - log n_s)
       z_b = tau * (log sum_s exp(u_s/tau) - log |S|).

   `tau` follows a fixed geometric continuation from `1.0` to `0.2` over 16
   epochs. Early gradients are distributed across plausible candidates; the
   final objective regains selectivity. Difference from arm 2 tests pooling
   and gradient concentration.

For every trained arm, Stage A freezes two inference rules:

- `raw`: argmax of mean original/flip candidate logit;
- `rank_fusion`: argmax of

      0.5 * percentile_rank(model_logit)
    + 0.5 * percentile_rank(upstream_score).

The coefficient is exactly equal weight, candidate ranks are computed within
the image, and ties use higher raw model logit then lower frozen candidate
index. It is the already-frozen successful rule, not a G2 validation fit.

## Scientific expectation and gates

Primary evidence is actual selected binary-mask Dice. AUROC, bag F1,
effective candidate count and source/count correlations are explanatory only.

- Reproduction gate: the frozen old G1 checkpoint must reproduce all 371
  Stage-A choices and its `0.206026` Stage-B Dice.
- Mechanism gate: removing hard top-1 must increase effective candidate count
  and must not reduce fused overall Dice by more than `0.01` versus the
  source-safe control.
- Correction gate: hierarchical fused Dice must exceed the frozen exploratory
  fusion `0.288729`; no subgroup may fall by more than `0.02`.
- Operational goal remains Dice at least `0.195608/0.479674/0.513613` for
  small/medium/large, with the stretch targets retained in the research log.
- If the corrected representation does not beat the fixed fusion baseline,
  this exact objective is retired. No seed, temperature, threshold, epoch,
  resolution or morphology sweep is permitted on the same validation cohort.

## Paper-to-mechanism mapping

- OICR propagates a seed to spatially related proposals rather than treating a
  singleton top proposal as unquestioned truth. This supports retiring the
  current detached singleton target.
- PCL groups related proposals to reduce part/extent ambiguity; hierarchical
  source pooling is the minimal low-risk precursor before any pseudo cluster
  refinement.
- ItS2CLR shows that pseudo-instance refinement in medical MIL needs
  reliability-aware self pacing; it contradicts immediate hard top-1 labels.
- TS2C explicitly separates proposal purity and completeness, matching the
  measured complementarity of G1 localization and upstream extent evidence.
- *Small Objects Matters* requires size-stratified evidence because aggregate
  WSSS metrics hide small-object collapse.
- Shortcut Mitigating Augmentation supports removing label-correlated context;
  here the shortcut is even more direct and observable: proposal-source
  presence itself encodes the image label.

This experiment tests a bounded causal correction. It does not claim that an
architecture name, a higher oracle or a lower training loss implies improved
segmentation.
