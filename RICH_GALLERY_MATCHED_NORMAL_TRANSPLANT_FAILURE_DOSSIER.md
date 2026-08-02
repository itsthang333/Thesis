# Rich-gallery matched-normal transplant failure dossier

## Decision

Matched-normal candidate transplant is retired as a selector signal.  The
immutable best validation pipeline remains G1 plus upstream fixed
percentile-rank fusion at Dice/IoU `0.2887294867/0.2168391813`.  The fixed
matched-normal 3:1 fusion reaches only `0.2810666203/0.2090362808`; it does not
promote.  No weight, threshold, seed, resolution or morphology sweep is
authorized from this result.  Test remains locked.

## Actual binary-mask endpoint

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1 + upstream baseline | 0.288729 | 0.216839 | 0.157723 | 0.435229 | 0.386874 | 49 |
| Transplant only | 0.088128 | 0.055086 | 0.018099 | 0.145750 | 0.223343 | 37 |
| Equal baseline/transplant | 0.226236 | 0.166112 | 0.056999 | 0.372878 | 0.523463 | 31 |
| Fixed 3:1 baseline/transplant | 0.281067 | 0.209036 | 0.096454 | 0.472267 | 0.480353 | 31 |
| Fixed 3:1 random-recipient control | 0.263178 | 0.193958 | 0.099006 | 0.437539 | 0.423072 | 34 |

The complete-group paired bootstrap for 3:1 matched versus baseline is
`-0.007763`, 95% CI `[-0.038285, 0.021947]`.  The subgroup effects expose the
mechanism: small `-0.062685` with CI `[-0.107285,-0.025327]`, medium
`+0.043854` with CI `[0.002485,0.088362]`, and large `+0.101229` with CI
`[0.018082,0.198543]`.  Matched versus random control is not established
overall: `+0.014998`, CI `[-0.010515,0.040384]`.

## The baseline gap is selection, not proposal supply

- Gallery oracle Dice: `0.5282983322`.
- Eligible-gallery oracle Dice: `0.5279020259`.
- Candidate-supply regret: only `0.0003963063`.
- Remaining eligible selector regret: `0.2391725392`.
- Within-selected-source regret: `0.1683762053` (about 70.4% of the eligible
  selector gap).
- Cross-source regret: `0.0707963339` (about 29.6%).
- Baseline selects a different source from the eligible oracle in `104/184`
  tumor images and completely misses overlap in `49/184`.

Adding more proposals cannot materially solve this gap.  Both choosing the
source and ranking candidates inside the chosen source remain wrong, with the
larger mean loss occurring inside source.

## Extent error is signed and lesion-size dependent

- Small lesions: Dice `0.157723`; `35/94` complete misses; `77/94` baseline
  choices are over-extent; median selected/GT area ratio `14.603`.
- Medium lesions: Dice `0.435229`; median ratio `1.098`, but selector regret is
  still `0.295022`.
- Large lesions: Dice `0.386874`; `10/18` choices are under-extent; median ratio
  `0.382` and selector regret is `0.359374`.

The transplant fusion changes `143/184` candidates.  It recovers 22 baseline
misses but creates four new misses.  Among small lesions it changes 70 choices,
improves 20 and worsens 32 while multiplying selected area by median `1.722`;
small Dice therefore collapses by `0.061270`.  The same expansion helps medium
and large lesions, explaining their gains without demonstrating better tumor
identity.

## Candidate identity evidence

Median within-image rank statistics are:

| Signal | Quality correlation | Area correlation | Oracle percentile |
|---|---:|---:|---:|
| G1 logit | 0.587220 | 0.342029 | 0.805825 |
| Upstream | 0.349794 | 0.013163 | 0.783922 |
| Fixed G1/upstream fusion | 0.585058 | 0.202131 | 0.815248 |
| Matched transplant logit | 0.218359 | 0.238147 | 0.549586 |
| Random transplant logit | 0.236125 | 0.260950 | 0.598820 |
| Matched minus random | 0.005788 | 0.020520 | 0.489923 |
| Matched final class response inside mask | 0.203832 | 0.168133 | 0.560208 |
| Norm5 relative feature contrast | 0.049537 | 0.429383 | 0.313421 |
| Fixed 3:1 fusion | 0.584175 | 0.252320 | 0.812737 |

The added score has much less relation to candidate overlap than G1 and is no
better than its random-recipient control.  Subtracting the random control
removes almost all residual quality signal.  Therefore it is not a new
candidate-specific tumor observable.

## Layer-by-layer failure localization

| Layer | Matched oracle percentile | Random oracle percentile | Matched-random gain | Quality corr | Area corr | Recipient CV |
|---|---:|---:|---:|---:|---:|---:|
| `pool0` | 0.405731 | 0.390201 | 0.011112 | -0.055331 | 0.011891 | 0.212585 |
| `transition1` | 0.511597 | 0.488071 | 0.014368 | 0.159896 | 0.192265 | 0.096764 |
| `transition2` | 0.572229 | 0.554774 | 0.024098 | 0.233164 | 0.290897 | 0.078084 |
| `transition3` | 0.489459 | 0.517278 | 0.000000 | 0.078551 | 0.090664 | 0.118339 |
| `norm5` | 0.313421 | 0.352712 | 0.000048 | 0.049537 | 0.429383 | 0.525054 |

The copied candidate is not distinguishable as tumor content at the stem after
sham cancellation.  A weak mid-level signal appears at transition2, but the
random arm follows almost the same trajectory.  It collapses at transition3;
by norm5, candidate quality correlation is nearly zero while area correlation
is strong, and recipient instability rises sharply.  The frozen image
classifier converts the intervention mainly into transferred mass/extent,
not stable local tumor identity.  Global tumor-class response cannot recover
identity that the spatial representation no longer contains.

The effect is especially poor for small lesions: their norm5 oracle percentile
is about `0.231`, versus `0.357` for medium and `0.673` for large.  This is the
layer-level counterpart of the observed extent expansion.

## What is now ruled out

- More candidates or gallery truncation.
- Another global fixed fusion weight for the transplant score.
- Treating lower miss count as proof of better segmentation.
- Frozen-classifier transplantation, whether anatomy matched or random.
- A morphology/threshold rescue after selecting the wrong candidate.
- Repeating mask-bag/self-winner MIL, G2 negative-only MIL, broad foreground
  expansion, density rarity or post-hoc scoring on the same frozen
  representation.

## Requirement for the successor

The successor must preserve the G1/upstream baseline and add two missing
observables rather than replace it:

1. candidate-specific tumor identity that remains predictive after controlling
   for candidate area and source;
2. a signed, scale-dependent extent cue that shrinks over-extent for small
   lesions but expands under-extent for large lesions.

The next bounded analysis must first test whether any frozen transition2
statistic adds conditional information after controlling G1, upstream, area
and source.  If it does not, the representation—not fusion—must be retrained at
high spatial resolution with image-label-only objectives and normal-image
candidate negatives.  No test data were accessed.

## Provenance

- Stage-A prediction freeze SHA-256:
  `3e9760d3b98ac5dbe1d909db74968483eaf0fff0d2bc6c70c46668a7479ab765`.
- Independent Stage-A audit SHA-256:
  `67c20c4b9d9c140b1d91bb89d4d7e4f3368346fe9815a4043a412729ef555580`.
- Stage-B summary SHA-256:
  `ebbfb74527513d2411bb4ce110d24875ad7749c928264f1a18425d4c73a5dffc`.
- Per-image/per-candidate/per-layer SHA-256:
  `a0f291d155c0bd389d81a1e4cc27e8be6c208af1eabde20b13fc7a7fc02f286f`,
  `ea3fd29d5fac7de46bd846bfcf65e87be0041875a0c11e8a9303c6f4fe95c73c`,
  `001263495dda1b02047c3c73e2f8aee14761ddc01d2ef937c9e73f62ca4d9903`.
- Failure analysis/dossier SHA-256:
  `6dc76e13212851c6522036d40afd92082578d48e0c0b814ee87538391c82f2e9`,
  `4b848354cba0482d0c80965e7adb8303d68c682b757cec90e5f4676c8f44ca7d`.
- Validation predictions were frozen before polygons; test was not opened.
