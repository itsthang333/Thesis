# G1 fixed-rank fusion after BAS-B2: causal bottleneck dossier

## Immutable reference and actual endpoint

The reference remains `0.5*rank(G1)+0.5*rank(upstream)`. Its actual frozen
validation Dice/IoU is `0.2887294867/0.2168391813`, with subgroup Dice
`0.1577232964/0.4352293348/0.3868735327` and 49 complete misses. The
validation-tuned alpha-0.632 result `0.2914726136` remains an explanatory upper
bound, not a promotable result.

BAS-B2 added one predeclared rank:

`S_B2(c) = (rank(G1(c)) + rank(upstream(c)) + rank(BAS(c))) / 3`.

All five selector variants were frozen before validation polygons. Test was
not opened.

| Variant | Dice | IoU | `<1%` | `1-<5%` | `>=5%` | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1 + upstream baseline | **0.288729** | **0.216839** | **0.157723** | **0.435229** | 0.386874 | 49 |
| BAS only | 0.048777 | 0.026720 | 0.004869 | 0.063854 | 0.217761 | 27 |
| G1 + BAS | 0.099668 | 0.062829 | 0.011057 | 0.151667 | 0.354415 | 9 |
| upstream + BAS | 0.159352 | 0.109432 | 0.014652 | 0.257524 | **0.522317** | 45 |
| G1 + upstream + BAS | 0.181106 | 0.127894 | 0.027455 | 0.301711 | 0.501086 | 22 |

The primary delta is `-0.107623`. The paired complete-group bootstrap CI95 is
`[-0.140313,-0.057568]`. BAS-B2 therefore fails decisively; it is not a noisy
tie with the baseline.

## The complete failure chain

The result can be traced without guessing:

`optimization collapse -> label-prior activation -> area-rank candidate score
-> over-expansion -> small/medium Dice loss`.

### 1. The two-logit classifier died

From epoch 2 through epoch 100:

- full-image CE is exactly `0.693359375`;
- foreground CE is exactly `0.693359375`;
- training accuracy is exactly `0.5008386448`;
- validation AUROC is `0.5`;
- validation sensitivity/specificity is `1/0` at the fixed probability rule.

The canonical train set contains 1,493 normal and 1,488 tumor images. A tied
two-logit classifier has `argmax=normal`, hence accuracy

`1493/2981 = 0.5008386448`,

which matches the observed value exactly. CE is the AMP representation of
`log(2)`. This is not merely weak classification: the classifier is constant.

The local architecture copied the official BAS ResNet head, including the
terminal class-map ReLU. In the final checkpoint, a label-safe real-image
probe found both class pre-activation maps strictly negative
(`min=-4.6393`, `max=-0.2311` on the probe), so the terminal ReLU produces
exact zero maps and blocks the CE gradient. This is the proximal optimization
failure. It does not by itself falsify BAS on its original 200/1000-class
benchmarks; it exposes an unstable binary transfer.

The official implementation and paper support the architectural provenance:

- the ResNet classifier ends with `Conv2d(...,200) -> ReLU`;
- the localizer is a sigmoid map;
- the objective combines full/foreground classification and background
  activation suppression.

Primary sources:

- BAS paper: https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Background_Activation_Suppression_for_Weakly_Supervised_Object_Localization_CVPR_2022_paper.html
- BAS reference code: https://github.com/wpy1999/BAS/blob/main/CUB/Model/resnet.py

### 2. The localization branch became a label prior, not a spatial map

The epoch-2-to-100 BAS term is `0.5989936437`. For a degenerate solution in
which the selected normal map is zero and selected tumor map is one, the
background-ratio term vanishes and the area penalty is

`lambda_area * N_tumor / N_train = 1.2 * 1488/2981 = 0.5989936263`.

The residual to the observed BAS loss is only `1.74e-8`. Independently, all
371 frozen tumor maps are spatially degenerate: tumor nondegenerate fraction
is `0`, mean activation range is `9.70e-7`, and the tumor maps have mean near
one. Thus the exact training scalar and Stage-A tensors agree: the network
encoded the image label as an almost constant map instead of learning tumor
location.

### 3. The candidate score consequently became an area rank

For a nearly constant normalized activation `A(x)=mu` and candidate mask `M`:

`coverage(M) = sum_M A / sum_Omega A = |M|/|Omega|`,

`purity(M) = sum_M A / |M| = mu`.

The frozen BAS score was the harmonic mean

`H(M) = 2*coverage*purity/(coverage+purity)`.

For fixed positive `mu`, `dH/d|M| > 0`; therefore ranking by `H` is ranking by
mask area. The empirical check across all 184 tumor bags confirms the
derivation:

- mean/median within-image Spearman(`BAS score`, candidate area):
  `0.999902/0.999922`;
- fraction of images with correlation above `0.9`: `1.0`;
- BAS-only selected candidate area percentile: mean/median `1.0/1.0`.

BAS did not supply tumor identity. It supplied the largest-mask prior.

### 4. Area expansion explains every subgroup transition

| Frozen regime | n | Baseline Dice | BAS-3way Dice | Delta | Baseline/BAS misses | BAS area/GT median |
|---|---:|---:|---:|---:|---:|---:|
| all tumors | 184 | 0.288729 | 0.181106 | -0.107623 | 49/22 | 28.756 |
| two-score dominance gap `>=0.05` | 104 | 0.205704 | 0.170230 | -0.035474 | 34/12 | 26.334 |
| baseline complete miss | 49 | 0 | 0.030601 | +0.030601 | 49/19 | 118.323 |
| baseline area/GT `<0.5` | 31 | 0.147686 | 0.261491 | +0.113805 | 10/3 | 5.837 |
| baseline area/GT `0.5-2` | 60 | 0.620033 | 0.325037 | -0.294995 | 6/2 | 8.293 |
| baseline area/GT `2-10` | 39 | 0.261961 | 0.133297 | -0.128664 | 12/3 | 45.237 |
| baseline area/GT `>=10` | 54 | 0.020917 | 0.009565 | -0.011352 | 21/14 | 338.621 |

The broad mask prior recovers 30 previous misses but loses three previous
hits. More importantly, it contributes only `+5.8176` total Dice mass while
destroying `-25.6202`. Miss count is therefore misleading: overlap recall
rises while precision and calibrated extent collapse. This is also why large
lesions improve by `+0.114212`, whereas small and medium lesions lose
`-0.130268/-0.133518`.

## What the failure adds to the baseline diagnosis

The rich gallery is not the limiting factor:

- gallery oracle Dice: `0.5282983322`;
- eligible oracle Dice: `0.5279020259`;
- candidate truncation regret: `0.0003963063`.

For the eligible gap

`O - B = 0.5279020259 - 0.2887294867 = 0.2391725392`,

the exact decomposition is:

| Component | Dice gap | Share |
|---|---:|---:|
| jointly dominated under G1 and upstream | 0.137338 | 57.42% |
| nonlinear two-score frontier | 0.009206 | 3.85% |
| image-specific two-score weight | 0.092628 | 38.73% |

The oracle candidate is absent from the two-score Pareto frontier in 154/184
tumors. In 104 severe cases, 64 choose the wrong proposal source and 40 fail
within the correct source. Thirty-one severe cases are extreme over-extent
(30 small), while 26 are under-extent (17 medium, 8 large, 1 small).

The precise missing observable is therefore **candidate-level tumor identity
with scale-dependent extent calibration**. It must answer two different
questions:

1. for tiny lesions, does the candidate include mostly normal bone around a
   small abnormal focus?
2. for medium/large lesions, is the candidate only a discriminative fragment
   of a larger abnormal region?

One scalar area prior cannot answer both because the required correction has
opposite signs.

## Cross-experiment elimination table

| Tested evidence | What it did | Why it cannot close the current gap |
|---|---|---|
| more/rescaled proposals | oracle already saturated | truncation regret is 0.000396 |
| G1/upstream weight sweep | best post-hoc Dice 0.291473 | only +0.002743; 57.42% of gap is jointly dominated |
| G2 negative-only MIL | reduced source shortcut but lost G1 complementarity | no positive-instance signal; more misses |
| global consensus | fewer misses, better medium/large | selected repeated anatomy; small over-extent rose |
| top-10 relation | Dice 0.285647 | support expands stable masks and harms small lesions |
| SAM confidence | Dice 0.098983, 125 misses | generic object confidence is not tumor identity |
| anomaly/reconstruction/counterfactual families | near-zero spatial Dice | image difference/rarity was not lesion-specific |
| BAS-B2 as run | Dice 0.181106 in three-way fusion | binary optimization collapsed; BAS score became area |

These failures narrow the search. Another threshold, seed, fusion alpha,
source rule, consensus statistic, global CAM, or morphology operation would
repeat an eliminated mechanism.

## Scientifically justified correction

B2 is not eligible for a full hyperparameter sweep. It has one bounded
technical follow-up because its intended mechanism was never reached:

1. replace only the dead terminal class-map ReLU by a strictly positive,
   non-zero-gradient mapping (Softplus), preserving the nonnegative BAS
   activation semantics;
2. retain the same data, 448 resolution, backbone, losses, gallery and seed;
3. stop after a cheap label-safe prefix unless all conditions hold:
   classification departs materially from the tied-logit solution, both class
   maps remain nondegenerate, the tumor activation is spatially nonconstant,
   and the BAS-candidate correlation with raw area is materially below one;
4. only after passing those mechanics may the run continue and freeze
   candidate selections;
5. actual Dice, IoU and all `94/72/18` subgroups—not AUROC—remain the final
   promotion endpoint against `0.2887294867`.

This correction is a root-cause test, not a new pipeline family. If it still
collapses or its viable BAS score remains an extent duplicate, binary BAS is
retired without weight, epoch, resolution or threshold sweeps.

If BAS becomes mechanically valid but does not beat the baseline, the next
addition must be candidate-conditioned rather than image-global: an
inside-versus-local-ring positive-evidence residual with train-normal hard
negatives, zero initialized on top of the immutable G1/upstream score. Its
purpose would be to identify lesion content while penalizing normal anatomy
inside oversized candidates, not to reward area or generic proposal quality.

## Provenance

- corrected Stage-B summary SHA-256:
  `da3c8a4d2fa3d920321b2fb703166f75428919580d90eb370e44fd21159a3283`;
- corrected Stage-B per-image SHA-256:
  `7fb1fb5874c2ca40749faf5189f9e150e634f1faed06af2cfb0208eebdb57936`;
- post-BAS analysis summary SHA-256:
  `490a387d8042ae7b31c43706604e11970270d1c24a8562577c4ae2506ad7cd4f`;
- post-BAS per-image SHA-256:
  `926005147f4c888357f0aef6ac1cdaf5fdb72cb0c37752e6d67f0a18a31e3e92`;
- generated dossier SHA-256:
  `b784a31e23e403cfb9a8cc0d233f607dfbfc5c4e675f0cae71a738e7041ff533`;
- validation: 371 images, 184 tumors, subgroups 94/72/18;
- validation polygons used only after prediction freeze: true;
- test evaluated: false.
