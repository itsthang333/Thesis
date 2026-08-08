# Results

## Primary validation result

| Cohort | n | Dice | IoU | Precision | Recall | Median predicted/GT area |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 184 | **0.288729** | **0.216839** | 0.336581 | 0.489305 | 2.0448x |
| `<1%` | 94 | 0.157723 | 0.118576 | 0.147795 | 0.470738 | 14.6026x |
| `1-<5%` | 72 | 0.435229 | 0.323558 | 0.488862 | 0.545975 | 1.0981x |
| `>=5%` | 18 | 0.386874 | 0.303117 | 0.713342 | 0.359586 | 0.3824x |

## Fully-supervised validation comparison

| Method | Overall Dice | Overall IoU | `<1%` Dice | `1-<5%` Dice | `>=5%` Dice |
|---|---:|---:|---:|---:|---:|
| WSSS Rich Gallery G1 + fixed rank fusion | 0.288729 | 0.216839 | 0.157723 | 0.435229 | 0.386874 |
| Fully supervised ResNet18-U-Net, historical checkpoint | 0.492765 | 0.400359 | 0.345608 | 0.629674 | 0.713613 |
| Fully supervised ResNet18-U-Net, final-retrain checkpoint | 0.490149 | 0.394966 | 0.338842 | 0.622616 | 0.750434 |

All rows are validation results. The historical fully checkpoint is retained as
an audited historical upper bound. The final-retrain checkpoint SHA-256
`becd752d0df2b0adfe0ea0c099117435f5b82da8fe2726eebecfb7af6322f33f`
is the matched model actually carried into the locked test comparison. Neither
fully-supervised row may be presented as WSSS performance.

E0 evaluates the final-retrain checkpoint on three grids:

| Grid | WSSS Dice | Fully Dice |
|---|---:|---:|
| 320 | 0.288729 | 0.489308 |
| 448 | 0.288218 | 0.490149 |
| Native | 0.288224 | 0.489581 |

The native subgroup definition yields exactly the same membership counts as the
old 320 definition (94/72/18). Thus the headline conclusions are not an artifact
of coordinate resolution.

## G4 E1: binary versus ten-class image supervision

Both arms use the same DenseNet-121, 320 input, ImageNet initialization,
AdamW budget, canonical split, checkpoint endpoint (validation binary F1), and
seeds 42/43/44. The ten-class tumor score is `1 - p(normal)`.

| Image-level endpoint | Binary, mean +/- SD | Ten-class collapsed to binary, mean +/- SD |
|---|---:|---:|
| F1 at 0.5 | 0.780882 +/- 0.007753 | **0.795082 +/- 0.007123** |
| MCC at 0.5 | 0.560627 +/- 0.007211 | **0.592005 +/- 0.028990** |
| AUROC | 0.848727 +/- 0.009683 | **0.860100 +/- 0.007996** |
| Average precision/AUPRC | 0.860190 +/- 0.006253 | **0.869750 +/- 0.006818** |
| Brier (lower is better) | 0.196481 +/- 0.004550 | **0.179059 +/- 0.013914** |
| NLL (lower is better) | 1.514750 +/- 0.115097 | **1.033681 +/- 0.077392** |
| ECE-15 (lower is better) | 0.192456 +/- 0.008731 | **0.164921 +/- 0.022648** |

The paired grouped-bootstrap intervals for F1, MCC, AUROC and AUPRC include zero
for every matched seed. NLL is the stable classification-side exception:
ten-class minus binary is negative for all seeds, with 95% intervals
`[-1.058,-0.294]`, `[-0.723,-0.045]`, and `[-0.738,-0.078]`. The decisive
downstream experiment below is therefore required; image-level discrimination
alone is not used as evidence of spatial improvement.

The AP values above supersede the first audit's 0.864460/0.871859 values. The
first implementation processed equal-score samples one row at a time, so AP
could depend on stable row order. The corrected implementation treats all
equal-score samples as one threshold, is row-order invariant, and was replayed
from the immutable prediction CSVs. AUROC, F1, calibration metrics and all
segmentation results are unchanged. The corrected independent audit SHA-256 is
`967b588537d4a2f2814a68e2a660c5b35fc87dd559482efb97069a9f58d4b215`.

The ten-class task itself is difficult and seed-sensitive: top-1 accuracy is
0.7062/0.6253/0.6146 and macro F1 is 0.4743/0.3533/0.3528. This prevents the
collapsed result from being misreported as uniformly strong nine-disease
classification.

The binary downstream arm has now completed the full CAM -> SAM gallery -> G1
-> rank-fusion path for all three classifier seeds and passed an independent
output audit:

| Binary seed | Overall Dice | `<1%` | `1-<5%` | `>=5%` | Gallery oracle |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.283706 | 0.143147 | 0.440155 | 0.391938 | 0.532662 |
| 43 | 0.273176 | 0.124510 | 0.444978 | 0.362334 | 0.516062 |
| 44 | 0.231815 | 0.128050 | 0.361174 | 0.256265 | 0.517571 |
| Mean +/- sample SD | **0.262899 +/- 0.027429** | -- | -- | -- | -- |

The matched ten-class downstream arm has also completed and passed the same
independent output audit:

| Ten-class seed | Overall Dice | `<1%` | `1-<5%` | `>=5%` | Gallery oracle |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.294779 | 0.149702 | 0.459865 | 0.392053 | 0.524402 |
| 43 | 0.306235 | 0.127777 | 0.497273 | 0.474034 | 0.534792 |
| 44 | 0.295848 | 0.119748 | 0.482090 | 0.470517 | 0.539855 |
| Mean +/- sample SD | **0.298954 +/- 0.006328** | **0.132409 +/- 0.015505** | **0.479743 +/- 0.018814** | **0.445535 +/- 0.046350** | -- |

The mean ten-class-minus-binary Dice delta is `+0.036055` (sample SD across
three matched seeds `0.026607`), with per-seed deltas `+0.011073`, `+0.033059`
and `+0.064033`. The paired grouped-bootstrap Dice interval crosses zero for
seed 42 (`[-0.013630,0.035355]`) but is positive for seed 43
(`[0.003513,0.063614]`) and seed 44 (`[0.029156,0.099021]`). The gain is
concentrated in medium and large lesions: mean subgroup deltas are `+0.064307`
and `+0.108689`, whereas the small-lesion mean delta is only `+0.000507` and
changes sign across seeds. Thus ten-class supervision is supported as the
better overall downstream configuration under this matched protocol, but it
does **not** resolve the small-lesion bottleneck.

Independent audit SHA-256 values are
`64479f6c8ff330324e74219e6073f2f8541cd96699b344d7c43fd2a5f8a9e0ca`
for the ten-class arm and
`c49b8cfbaeb4a3c97df313e61e19d9152dc4fe97010828dfbe88cfbf78a05176`
for the paired report.

## G4 E3: matched SAM ViT-B versus ViT-L (validation, interim)

Both completed arms keep the image evidence, prompts, multimask setting,
gallery merge/cap, G1 selector and fixed rank fusion identical; only the frozen
SAM-v1 backbone/checkpoint changes. Both independent output audits pass. ViT-H
is still running, so this is an interim B/L comparison and not the final E3
backbone conclusion.

| SAM backbone | Dice | IoU | Oracle Dice | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B | 0.288729 | 0.216839 | 0.528298 | 0.157723 | 0.435229 | **0.386874** |
| ViT-L | **0.291185** | **0.219708** | **0.546000** | **0.165224** | **0.446083** | 0.329389 |

The paired heuristic-group bootstrap gives B-to-L Dice delta `+0.002456`
(95% CI `[-0.020021, 0.024824]`, `P(delta>0)=0.5866`), so the point estimate
does not establish a real overall improvement. Subgroup deltas are `+0.007501`
for small (CI `[-0.029231, 0.043982]`), `+0.010854` for medium (CI
`[-0.013151, 0.036579]`) and `-0.057485` for large (CI
`[-0.137646, 0.004692]`). ViT-L changes the selected source for 49/184 tumor
images and increases the gallery oracle, but the selector does not reliably
convert that extra proposal supply into Dice.

| Measured resource | ViT-L / ViT-B ratio |
|---|---:|
| Candidate-generation wall time | 2.061x |
| Total arm wall time | 1.841x |
| Peak allocated VRAM | 1.570x |
| Peak reserved VRAM | 1.496x |
| Merged gallery bytes | 1.005x |

Therefore ViT-B remains the defensible resource/accuracy choice pending ViT-H:
ViT-L costs materially more, its overall CI includes zero, and it likely harms
the scarce large-lesion subgroup. This is an accuracy-cost conclusion, not a
claim that ViT-B is universally superior.

These are development results on 184 tumor validation images. There are 49
complete misses.

### G4 E2 attribution/prompt decomposition

The independent 4-attribution x 3-prompt decomposition completed on the same
184 validation tumor images.  The table below reports marginals over the three
prompt modes; `selected` uses the frozen upstream selector, while `oracle` is
diagnostic proposal supply and is never an inference result.

| Attribution | CAM-only Dice | Selected SAM Dice | Proposal-oracle Dice | Selector regret |
|---|---:|---:|---:|---:|
| CAM | 0.132644 | 0.143837 | 0.231095 | 0.087258 |
| Grad-CAM | 0.129019 | 0.137324 | 0.218027 | 0.080704 |
| Grad-CAM++ | 0.134681 | 0.149826 | 0.236080 | 0.086254 |
| **LayerCAM** | **0.152326** | **0.193677** | **0.330514** | **0.136837** |

The best individual arm is LayerCAM + point prompt: CAM-only Dice 0.152326,
selected Dice 0.205224, proposal-oracle Dice 0.339441, and Recall@Dice
0.1/0.3/0.5 of 0.586957/0.483696/0.347826.  Its subgroup decomposition is:

| Subgroup | CAM-only Dice | Selected Dice | Proposal-oracle Dice | Selector regret |
|---|---:|---:|---:|---:|
| `<1%`, n=94 | 0.019024 | 0.118534 | 0.168793 | 0.050259 |
| `1-<5%`, n=72 | 0.260831 | 0.274250 | 0.496441 | 0.222191 |
| `>=5%`, n=18 | 0.414433 | 0.381836 | 0.602605 | 0.220769 |

Thus LayerCAM is selected by BTXRD evidence, not citation alone. SAM materially
improves the point estimate, especially for small lesions, but cannot repair
the weak small-lesion supply ceiling. For medium and large lesions, the much
larger oracle-minus-selected gaps identify candidate selection as the dominant
remaining error.

## Locked final test result

The independently frozen WSSS and fully-supervised predictions were evaluated
together in one annotation-opening pass. The test cohort contains 373 images,
including 187 tumor images used for the tumor-only metrics below.

| Method | Overall Dice | Overall IoU | `<1%` Dice, n=111 | `1-<5%` Dice, n=51 | `>=5%` Dice, n=25 |
|---|---:|---:|---:|---:|---:|
| **WSSS Rich Gallery G1 + fixed rank fusion** | **0.260881** | **0.190837** | 0.153196 | 0.432402 | 0.389095 |
| Fully supervised ResNet18-U-Net, threshold 0.20 | 0.524423 | 0.421245 | 0.428977 | 0.616287 | 0.760801 |

The WSSS-to-fully-supervised Dice gap is `-0.263542`. WSSS retains 49.7%
of the fully-supervised test Dice. Its validation-to-test Dice change is
`-0.027849` (0.288729 to 0.260881).

## Why all three candidate sources remain

The same fixed fusion rule was used for every source-subset comparison. No
two-source subset preserved the full validation result:

| Available sources | Overall Dice |
|---|---:|
| LayerCAM + classifier-448 + external saliency | **0.288729** |
| LayerCAM + classifier-448 | 0.283344 |
| Classifier-448 + external saliency | 0.282440 |
| LayerCAM + external saliency | 0.280697 |

The gallery complexity is therefore supported by an ablation: each source
rescues a different subset of validation images.

The independent G4 replay (evaluation SHA-256
`1ae1b1f20bed1b4f403efd28e74b6c58769e4db0d56c3f9baf3de93ba5a471fb`)
also evaluated the same frozen choices at native image resolution. R7 reproduces
the official common-320 Dice exactly (`0.288729487`) and obtains native Dice
`0.288224022`. The coordinate change is therefore only `-0.000505465` Dice.

### G4 selector and fusion replay

| Selector/fusion | Native Dice | Common-320 Dice | Native delta vs R7 | Paired 95% CI |
|---|---:|---:|---:|---:|
| Random candidate | 0.101890 | 0.101866 | -0.186334 | diagnostic |
| SAM score only | 0.098902 | 0.098983 | -0.189322 | diagnostic |
| Upstream only / R0 | 0.225306 | 0.225868 | -0.062918 | diagnostic |
| G1 only / R1 | 0.205545 | 0.206026 | -0.082679 | diagnostic |
| z-score sum, R2 | 0.286375 | 0.287017 | -0.001849 | [-0.011902, 0.008911] |
| RRF(k=60), R5 | 0.285523 | 0.286009 | -0.002701 | [-0.013806, 0.006348] |
| **equal percentile rank, R7** | **0.288224** | **0.288729** | reference | [0, 0] |
| G1-heavy percentile rank, R8 | 0.283986 | 0.284431 | -0.004238 | diagnostic |

R7 is the best point estimate among the nine predeclared frozen fusion rules,
but its paired interval versus R2 includes zero. The defensible conclusion is
that rank fusion materially beats either raw score alone; the evidence does not
prove that 0.50/0.50 is uniquely optimal.

### G4 source-correct upstream-score replay

The source-correct E7 study froze 371 choices for each of 16 predeclared arms
before opening the 184 validation polygons. Legacy U5+R7 reproduces native Dice
0.288224 and common-320 Dice 0.288729. Recomputing density, mass and component
rank from each candidate's own source changes U5+R7 only slightly to native
Dice 0.289358. The global-rank U6+R7 arm has the largest point estimate:
native Dice 0.294956, common-320 Dice 0.295568, and small/medium/large Dice
0.149495/0.450780/0.431288. Its paired native delta is +0.006732, but the 95%
group-bootstrap interval [-0.010248, 0.024597] includes zero. This supports a
possible advantage of cross-source global rank; it does not establish a
statistically reliable replacement for the locked R7 baseline.

### G4 gallery-cap replay

| Total balanced cap | Native Dice | Native delta vs 243 | Paired 95% CI |
|---:|---:|---:|---:|
| 27 | 0.269759 | -0.018465 | [-0.036690, -0.004077] |
| 81 | 0.281168 | -0.007056 | [-0.018453, 0.001543] |
| 162 | 0.283835 | -0.004389 | [-0.013324, 0.000764] |
| 243 | **0.288224** | reference | [0, 0] |

Cap 27 is demonstrably insufficient on this validation cohort. The point
estimate increases monotonically through 243, but the paired intervals for 81
and 162 versus 243 include zero. Thus 243 is retained as the frozen accuracy
choice, not claimed as a universal optimum.

### G4 source-subset replay at native resolution

| Sources | Native Dice | Small | Medium | Large |
|---|---:|---:|---:|---:|
| All three | **0.288224** | **0.156724** | 0.435221 | 0.386958 |
| L320 + C448 | 0.282742 | 0.152888 | 0.428980 | 0.375910 |
| L320 + external | 0.280381 | 0.126035 | **0.453269** | 0.394858 |
| C448 + external | 0.282053 | 0.141390 | 0.428097 | 0.432454 |
| L320 only | 0.275183 | 0.125092 | 0.436259 | 0.414691 |
| C448 only | 0.257660 | 0.131406 | 0.410257 | 0.306601 |
| External only | 0.234452 | 0.061876 | 0.406385 | **0.447945** |

All three sources give the best overall and small-lesion point estimates, but
individual subsets can be better in medium or large lesions. Therefore the
three-source result supports complementary supply, while also exposing scale-
dependent selector error; it does not imply every source is best for every case.

## Remaining bottleneck

Candidate supply is not the main validation ceiling: the all-source per-image
oracle Dice is 0.527902. The dominant gap is selecting the correct candidate
and its extent. Small lesions are strongly over-segmented (median area ratio
14.6x), medium lesions are mainly identity/localization limited, and large
lesions are under-segmented (median area ratio 0.382x). A single global area
correction was therefore not included.

## Reporting status

The 0.288729 result is the validation-selected configuration; 0.260881 is its
locked final test Dice. They must remain visibly separate. WSSS choices and
fully-supervised masks were frozen before spatial test annotations were opened.
The final evaluator opened exactly 187 tumor polygons in one joint pass.
