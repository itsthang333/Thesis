# Rich-gallery partial-consensus seed diagnostic

Status: **COMPLETE — useful mechanism evidence, not a confirmed replacement
for the baseline**.

## Question

Can agreement among the already frozen rich-gallery candidates expose enough
reliable foreground/background supervision to train a better segmentation
consumer, without spatial ground truth?

The immutable comparator is G1 plus fixed percentile-rank fusion:

- Dice/IoU: `0.2887294867/0.2168391813`;
- subgroup Dice `<1% / 1-<5% / >=5%`:
  `0.1577232964/0.4352293348/0.3868735327`;
- complete misses: `49/184` tumor images.

All candidate choices and consensus rules were fixed before validation
polygons were opened.  The diagnostic covers the exact 371-image validation
cohort, including 184 tumors (`94/72/18`), and reads no test image.

## Fixed rules and actual Dice

| Frozen rule | Overall Dice | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|
| Baseline hard mask | 0.288729 | 0.157723 | 0.435229 | 0.386874 | 49 |
| Best-per-source union | 0.242142 | 0.065425 | 0.410185 | 0.492825 | 18 |
| Best-per-source majority | 0.284572 | 0.119169 | 0.470421 | 0.404942 | 43 |
| Best-per-source intersection | 0.227714 | 0.112627 | 0.373816 | 0.244314 | 90 |
| Baseline plus cross-source agreement | 0.289080 | 0.152415 | 0.442524 | 0.388996 | 58 |
| Top-three rank-fusion majority | **0.293436** | 0.153564 | **0.458480** | 0.363708 | 47 |

The top-three majority is the highest **exploratory point estimate** in this
diagnostic, `+0.004707` over baseline.  It is not promoted as a confirmed new
baseline: its paired group-bootstrap 95% interval is
`[-0.008610, +0.017426]`, and it lowers both small and large subgroup means.
The best confirmed comparator therefore remains `0.2887294867`.

## What the result identifies

### Proposal supply is not the limiting resource

The source union recovers overlap on `31/49` baseline misses and raises mean
tumor recall to `0.7481`.  Useful lesion support is therefore present in the
gallery even when the selected mask completely misses it.  This agrees with
the independently measured gallery oracle Dice `0.528298` and near-zero
candidate-truncation regret.

### The union is too noisy to be a training target

Union micro precision is only `0.0663` overall and `0.00575` in the small
subgroup.  Pixels outside the union still contain `25.2%` of target pixels on
average (`30.3%/14.9%/39.8%` for small/medium/large).  Consequently:

- union foreground cannot be treated as a reliable positive mask;
- outside-union pixels cannot be treated as reliable background;
- the source-majority rule remains especially unsafe for small tumors
  (micro precision `0.0181`);
- a hard- or partial-label consumer would receive both severe false-positive
  supervision and systematic false-negative supervision.

This directly explains why another pseudo-mask U-Net is not authorized.  The
earlier consumer already demonstrated pseudo-label memorization: high train
pseudo-mask agreement but only about `0.230` validation Dice.

### Consensus contains a narrow medium-lesion signal

Top-three majority improves medium Dice by `+0.023251`, but changes small and
large by `-0.004160/-0.023166` and rescues only `3/49` baseline misses.  Thus
agreement can suppress some medium-lesion selector noise, but it does not
identify the subtle small-lesion candidate or restore the missing extent of
large lesions.  A single consensus rule is not the missing all-subgroup
mechanism.

## Decision

1. Do not train a segmentation consumer from these exact union/majority
   labels.
2. Keep `0.293436` as a disclosed exploratory point estimate, not a confirmed
   headline baseline.
3. Retain the gallery and immutable G1+fixed-rank score; the remaining problem
   is candidate-specific tumor evidence and signed extent, not proposal count.
4. Stop post-hoc scoring on the old frozen representation.  Matched-normal
   transplant, conditional-feature ranking, cross-view residual, latent-size
   routing and consensus now jointly show that its observables do not identify
   the useful candidate reliably.
5. The next bounded experiment must learn a new high-resolution local evidence
   map from image labels and matched normal references, then use that map only
   as a zero-initialized residual over the exact baseline selector.

## Provenance

- canonical split SHA-256:
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
- frozen candidate table SHA-256:
  `ea3fd29d5fac7de46bd846bfcf65e87be0041875a0c11e8a9303c6f4fe95c73c`;
- per-image output SHA-256:
  `16587600d8a27c5645f9b99cedb54fbbc3dce2e202676386d9f3ed25e1125250`;
- summary SHA-256:
  `ef1d943aa98f65054e0a4a48968e2b2c44e0e20b6d989418a5bcef1f1aeeba20`;
- `consumer_trained=false`, `test_evaluated=false`, `test_images_read=0`.

