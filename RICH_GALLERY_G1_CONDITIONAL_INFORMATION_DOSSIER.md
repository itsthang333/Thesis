# G1 fixed-fusion conditional-information dossier

## Bound result

The immutable prompt-free image-label-only validation baseline remains
`0.5*rank(G1)+0.5*rank(upstream)`:

- Dice/IoU: `0.2887294867/0.2168391813`;
- subgroup Dice `<1% / 1-<5% / >=5%`:
  `0.1577232964/0.4352293348/0.3868735327`;
- gallery/eligible oracle Dice: `0.5282983322/0.5279020259`;
- proposal-supply regret: `0.0003963063`;
- eligible selector regret: `0.2391725392`.

The purpose of this analysis is not to fit another validation selector.  It
asks whether any signal already frozen by the matched-normal Stage-A run
contains candidate-quality information beyond G1, upstream, candidate area and
proposal source.  The analysis consumes 32,519 evaluated candidate rows from
184 tumor images.  It creates no score, choice or binary mask.

## Conditional calculation

Within each image, candidate Dice and a frozen candidate signal are converted
to average percentile ranks.  Let

`C = [rank(G1), rank(upstream), rank(log(1+area)), source one-hot]`.

For target rank `y` and signal rank `s`, the diagnostic computes

`y_res = y - Proj_C(y)` and `s_res = s - Proj_C(s)`,

then reports `corr(y_res, s_res)`.  Every image contributes one correlation, so
large candidate bags cannot dominate the result.  The second endpoint asks how
often the signal ranks the eligible oracle above the immutable baseline choice.

This is a retrospective validation-GT information diagnostic only.  A signal
discovered here is not promotable without a new prediction-first protocol.

## Key results

| Signal | Overall partial | Oracle > baseline | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|---:|
| matched transplant logit | 0.053716 | 0.462857 | 0.061345 | 0.049311 | -0.032400 |
| random transplant logit | 0.015407 | 0.514286 | 0.076762 | -0.000926 | -0.041444 |
| matched minus random logit | **0.003547** | 0.491429 | 0.002705 | 0.033715 | -0.001238 |
| transition2 matched relative contrast | 0.039274 | 0.491429 | **-0.013838** | 0.074742 | 0.200759 |
| transition2 random relative contrast | 0.021533 | 0.440000 | -0.009387 | 0.057670 | 0.249899 |
| transition2 matched minus random | **-0.010044** | 0.565714 | -0.003974 | -0.018746 | 0.000975 |
| norm5 matched relative contrast | -0.044498 | 0.462857 | -0.052845 | -0.022785 | -0.119811 |

The strongest apparent conditional signals are ring-mass terms at pool0,
transition1 and transition2 (`0.112-0.136`).  But matched and random values are
identical and rank the eligible oracle above the baseline only `45.7-46.9%` of
the time.  They are geometry/support variables, not tumor identity.

The transition2 signal previously looked weakly useful before conditioning.
After controlling the immutable scores, area and source, its overall residual
is only `0.039274`; in the `<1%` subgroup it becomes negative.  Subtracting the
random-recipient arm removes rather than strengthens it.  The matched-specific
causal signal is therefore absent.

## Mechanistic decision

The frozen representation is exhausted as a source of candidate-specific tumor
identity.  Do not run another frozen fusion, feature layer, transplant weight,
threshold, morphology, source rule or area rule.  Such a run would combine
geometry proxies while leaving the oracle below both original ranks.

The successor must preserve the G1/upstream score exactly at zero residual and
learn a new representation from an additional weak-supervision relation.  It
must also separate candidate identity from signed extent:

1. small lesions need removal of normal tissue from masks that are median
   `14.603x` the GT area;
2. large lesions need recovery of abnormal tissue beyond masks that are median
   `0.382x` the GT area;
3. medium lesions already have plausible area but remain candidate-ranking
   limited.

## Provenance

- Stage-B summary SHA-256:
  `ebbfb74527513d2411bb4ce110d24875ad7749c928264f1a18425d4c73a5dffc`.
- Per-candidate table SHA-256:
  `ea3fd29d5fac7de46bd846bfcf65e87be0041875a0c11e8a9303c6f4fe95c73c`.
- Conditional-information JSON SHA-256:
  `d970aed42fd2f676b50c147075c1fef64c69f889acccb60b5e5ae0f0e4c526f7`.
- Generated dossier SHA-256:
  `124246792b5875122f86021a9c3471598a900f02a23bb44a8a6cf4b58dfd50a5`.
- Focused conditional/matched-normal regression: `8/8` pass.
- Validation predictions were frozen before polygons; test was not opened.
