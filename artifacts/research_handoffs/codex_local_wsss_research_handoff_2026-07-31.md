# Codex local BTXRD WSSS research handoff — 2026-07-31

## Scope and status

This is a documentation-only handoff from the parallel local `pipeline`
workspace into `research-wsss-improvement`. It does not register or launch a
new experiment, alter the active selector-cache/R1 claim, authorize a consumer,
or access BTXRD test.

The source workspace is dirty and is therefore identified by both its base
commit and the hashes of its research records rather than represented as a
reproducible source checkout:

- source branch/base commit: `pipeline` at
  `980722ac4b3f673dd09a9b2156d78b6ad334d0d9`;
- local research-log SHA-256:
  `489ea357e4d019aeb9e1661ee93e7a7de050ce3dc2e825e48f0a97a0f688e446`;
- experiment-registry SHA-256:
  `d1f68db99adac1a29daaacf5c9836bc25450323221c9ed80dcf83c5637277330`;
- ceiling-analysis SHA-256:
  `2f96f77e61b79edfb7e97c853493622cf799275d21e34739acb24b8cb1bca71d`;
- failure-analysis protocol SHA-256:
  `ad484d1d47621c2721520d726adbc94793f078bb8c341730f6b140378d1eec16`;
- counterfactual mechanism audit SHA-256:
  `9ecce5a9fdad5f6f5e419f9861e7ce8b9b926f51abe027407b6301bfdb419082`;
- BiomedCLIP failure-analysis SHA-256:
  `74d7a58fe9a9396b7562dc09b282dd65fda6c716ba1403cf5ccf60f26d6b9dc6`.

Only independently audited, full-validation efficacy results are called Dice
results below. Ranking-only or GT-oracle diagnostics are labeled separately.

## Split comparability audit

The two workspaces use byte-different manifests:

- this branch: `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
- local pipeline: `7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c`.

They are scientifically equivalent. A row-by-row audit found:

- `3,746/3,746` identical image IDs;
- `32` shared scientific columns;
- `0` differences across shared values, including group, split, eligibility,
  labels, image/annotation hashes and image geometry;
- no ID missing in either manifest.

The byte difference is caused by the later provenance schema adding
`dataset_table_semantic_sha256` and changing dataset-table hash metadata, not
by a cohort or split change. The validation population is the same
`371/184/187`, with tumor subgroups `94/72/18`.

## Completed comparable results

The machine-readable companion table is
`codex_local_wsss_results_2026-07-31.csv`. The most important comparable
results are:

| Method | Role | Tumor Dice | IoU | `<1%` | `1–<5%` | `>=5%` |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 U-Net 448, GT masks | fully supervised comparison | 0.492765 | 0.400359 | 0.345608 | 0.629674 | 0.713613 |
| Binary LayerCAM-320 + SAM | local best deployable WSSS teacher | 0.234339 | 0.173206 | 0.112163 | 0.348604 | 0.415311 |
| PCAM-512 | WSSS teacher | 0.203785 | 0.142124 | 0.026549 | 0.354184 | 0.527755 |
| Binary high-resolution-512 | WSSS teacher | 0.187435 | 0.135514 | 0.119516 | 0.281805 | 0.164643 |
| PCAM + SAM-IoU selector | WSSS teacher | 0.110730 | 0.079941 | 0.008768 | 0.168039 | 0.413969 |
| Dense normal-contrast anomaly | direct WSSS | 0.017385 | 0.009439 | 0.021934 | 0.012919 | 0.011493 |
| Global-local binary S2C | direct WSSS | 0.015063 | 0.010237 | 0.020035 | 0.012279 | 0.000232 |
| SynRad consensus | direct WSSS | 0.003838 | 0.002176 | 0.005836 | 0.000117 | 0.008286 |
| Masked healthy inpainting | direct WSSS | 0.000539 | 0.000277 | 0.001047 | 0.000000 | 0.000042 |
| Overlapping local-view MIL | direct WSSS | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The geometry-v3 selector result already on this branch,
`0.245482/0.117081/0.377136/0.389413`, is therefore the best completed WSSS
pseudo-mask result across the two synchronized workstreams. It is a real
improvement over LayerCAM overall and on small/medium tumors, but it is not a
final consumer result and it remains below its operational gate.

## Root-cause conclusions that survived multiple families

### 1. Resolution is necessary for small lesions but is not sufficient

The mean `<1%` lesion occupies only about `0.175/0.447/0.699` cells on
stride-32 grids at 320/512/640. Higher resolution improved support in some
cases but did not consistently improve selected masks. Binary 512 improved
small Dice only to `0.119516`, while PCAM-512 improved large Dice to
`0.527755` but collapsed small Dice to `0.026549`.

### 2. CAM-family selection and extent are separate bottlenecks

A GT-only per-image best-of-three oracle over LayerCAM-320, high-resolution
512 and PCAM-512 reached only `0.335916` overall and `0.164664` on small
tumors; `28/184` tumors were zero in all three. Small-mask median predicted/GT
area was `12.55x–53.27x`. More selector work on the same maps cannot recover
the fully supervised gap.

A wider 13-family audit reached only `0.384107` with a non-deployable GT
family oracle. On `<1%`, oracle overlap was `0.989362`, but oracle Dice was
only `0.219994` and median predicted/GT area was `18.19x`. This is direct
evidence that candidate extent remains limiting even when proposal recall is
nearly perfect.

### 3. A pseudo-mask consumer cannot repair missing spatial information

The historical pseudo-mask U-Net reached `0.230020`, below its LayerCAM
teacher at `0.234339`; small Dice fell by about `0.0412`. Training Dice on
pseudo-targets was high, so the consumer mainly copied pseudo-label noise.
Consumer work should remain gated on a material teacher improvement.

### 4. Generic anomaly/normal rarity is not tumor-specific evidence

Normal reconstruction, dense normal contrast, K=32 healthy-feature density
and normal-feature replacement all reduced some shortcuts or exposed weak
rank signal, but failed overlap:

- K=32 density: pixel AUC `0.563712`, true-area Dice `0.024037`, small
  `0.005725`, image AUROC `0.559143`;
- feature-normal replacement: pixel AUC `0.510931`, true-area Dice `0.023373`,
  small `0.002463`, image AUROC `0.419960`;
- causal-patch full arm: pixel AUC `0.518862`, true-area Dice `0.014293`,
  argmax hit `3/184`.

This is the most relevant negative transfer for the active R1 normal-prototype
arm. Distance from normal prototypes can be useful, but it must be validated
as image-label-conditioned candidate evidence rather than assumed to be tumor
evidence.

R1 is not a duplicate of these rejected diagnostics: it operates on the
stronger geometry-v3 SAM candidate gallery, retains the complete frozen base
scorer, learns only a zero-initialized residual with train image labels, and
selects K by group-held-out image-level OOF. It remains scientifically
admissible. The local evidence nevertheless justifies the existing strict
requirements on image AUROC, count shortcuts, selected-to-oracle regret and
subgroup Dice; failure should retire R1 without a broader K/threshold sweep.

### 5. Shortcut-resistant classification is not localization

- SynRad achieved strong synthetic Dice but placed `92.91%` of predicted
  pixels in the outer border on real images.
- Overlapping local-view MIL produced non-empty masks on `123/184` tumors but
  zero overlap on every subgroup; lesion-vs-background pixel AUC was
  `0.247154` and `81.95%` of lesion-area-matched top-score pixels were in the
  outer border.
- Raw BiomedCLIP dense scores were systematically inverted: pixel AUROC
  `0.2127–0.2489`, while anatomy prompts improved image AUROC without fixing
  localization. Local crop fusion increased border reliance.

Image F1/AUROC, flip consistency and normal suppression are useful safety
diagnostics, not substitutes for frozen post-prediction Dice.

## Recommendations for the active branch

1. Finish the exact selector-cache v5 transport and independent audit; do not
   accept the current partial 617-file download.
2. Run only the already frozen R1 arm if every cache gate passes.
3. Report both selector/pseudo-mask Dice and the fact that no new consumer has
   been trained. Do not compare a teacher Dice directly with final U-Net Dice
   without labeling the stage.
4. Preserve the gallery-oracle ceiling as the causal bound. If R1 fails,
   proceed to R2 affinity and S1 family-balanced pooling as predeclared; do not
   reopen resolution, normal-prototype K, threshold, morphology or SAM-grid
   sweeps.
5. If the finite selector campaign cannot materially close selected-to-oracle
   regret, the next representation must learn tumor-specific local evidence
   from positive-versus-negative image bags. Generic healthy rarity or
   acquisition-style invariance alone has already been falsified.

## Safety statement

All local efficacy rows above were validation-only and used spatial GT only
after predictions were frozen. Fully supervised results are comparison-only.
No train polygon was used by a WSSS arm, no local result authorized a student,
and BTXRD test remained locked.
