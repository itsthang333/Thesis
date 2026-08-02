# BTXRD WSSS deep bottleneck synthesis — 2026-08-02

Status: **analysis complete; no successor training launched**.  Test remains
sealed.  All spatial evidence below is retrospective validation-only.

## 1. Authoritative result ledger

Two statements must not be conflated:

1. The best reproduced rich-gallery comparator is **G1 plus equal
   percentile-rank fusion**, Dice/IoU `0.2887294867/0.2168391813`.
2. A fixed top-three majority has the highest exploratory point estimate,
   `0.2934364815`, but its paired group-bootstrap interval
   `[-0.008610,+0.017426]` crosses zero and it lowers both small and large
   subgroup means.  It is not a confirmed new baseline.

| Method | Academic status | Dice | IoU | `<1%` | `1-<5%` | `>=5%` |
|---|---|---:|---:|---:|---:|---:|
| Fully supervised reference A | comparison | 0.492765 | 0.400359 | 0.345608 | 0.629674 | 0.713613 |
| Fully supervised 448 continuation | clean-val FS winner | 0.495132 | 0.393262 | 0.328955 | 0.662442 | 0.693703 |
| LayerCAM-320 flip-TTA | promoted pre-gallery WSSS | 0.234339 | 0.173206 | 0.112163 | 0.348604 | 0.415311 |
| Rich gallery G1 raw | frozen diagnostic | 0.206026 | 0.151407 | 0.109585 | 0.300705 | 0.330950 |
| **Rich gallery G1 + fixed fusion** | **confirmed comparator** | **0.288729** | **0.216839** | **0.157723** | **0.435229** | **0.386874** |
| Top-three majority | exploratory only | 0.293436 | — | 0.153564 | 0.458480 | 0.363708 |
| G2 best arm | rejected | 0.254326 | 0.190661 | 0.137925 | 0.376147 | 0.374906 |
| Matched-normal transplant | rejected | 0.281067 | 0.209036 | 0.096454 | 0.472267 | 0.480353 |
| Cross-view co-witness best | rejected | 0.287911 | 0.215879 | 0.149945 | 0.437334 | 0.410709 |
| Pseudo-mask U-Net consumer | rejected | 0.230020 | 0.165471 | 0.071001 | 0.4134 | 0.3269 |

The two fully supervised rows are different checkpoint/evaluator lineages, not
two names for one number.  Reference A is retained for the historical subgroup
contract; the 448 continuation is the clean-validation FS winner recorded in
`EXPERIMENT_COMPARISON.csv`.

Against fully reference A, the confirmed WSSS gap is `0.204035` overall and
`0.187884/0.194445/0.326739` for small/medium/large.  The problem is not only
small lesions: the absolute gap is largest for the 18 large cases, although
the failure mechanism differs by group.

## 2. Original dataset facts that constrain the method

The canonical cohort has `2,981` train images (`1,493` normal, `1,488` tumor)
and `371` validation images (`187` normal, `184` tumor).  Validation tumors are
`94/72/18` small/medium/large.  The train/validation groups are `984/242`, with
`730/71` multi-image groups.  BTXRD exposes no patient/case ID, so heuristic
grouping remains a limitation.

### 2.1 Tiny lesions are absent from late CAM grids

| Group | Median GT area | Median bbox at 320 px | Expected positive cells at 10×10 | 32×32 | 128×128 |
|---|---:|---:|---:|---:|---:|
| `<1%` | 0.0952% | 11.55×15.45 px | **0.095** | 0.975 | 15.60 |
| `1-<5%` | 2.2212% | 53.24×60.40 px | 2.22 | 22.75 | 363.92 |
| `>=5%` | 9.0867% | 125.97×136.35 px | 9.09 | 93.05 | 1,488.76 |

At the small-lesion p10, a 10×10 grid contains only `0.023` expected lesion
cells; even 32×32 contains only `0.240`.  A final DenseNet CAM grid cannot
represent most small lesions geometrically.  A 512 px stride-4 map is therefore
necessary.  It is not sufficient: the 512/640 and OLV runs showed that a model
can preserve pixels while still learn image-level shortcuts.

### 2.2 Lesion burden is confounded with the supplied subtype label

- all `94/94` small tumors are benign;
- medium is `43` benign and `29` malignant;
- large is `13` benign and `5` malignant;
- `69/94` small cases are osteochondroma;
- osteosarcoma appears only in medium/large (`25/5`).

This matters in two opposite ways.  First, binary tumor-vs-normal training
discards legal image-level subtype information that may make the local task
less heterogeneous.  Second, a model can exploit subtype/anatomy without
learning the lesion.  Fine labels must condition local evidence, not serve as a
hard surrogate for true lesion size.

A new group-separated diagnostic confirms the distinction.  A subtype-routed
extent expert raises small/large Dice but damages medium, giving only
`0.277330` overall versus `0.288729`.  Benign/malignant routing gives
`0.277628`.  Thus subtype explains cohort structure but is not a valid extent
gate by itself.

### 2.3 The validation distribution is statistically fragile

The large subgroup contains only 18 images.  Reusing 184 tumors for many
exploratory selectors makes point estimates optimistic.  This is why
`0.293436` is disclosed but not promoted.  Future validation can be relaxed
for speed, but every headline must distinguish frozen/reproduced from
same-validation exploratory selection, consistent with the protocol warning
in [Choe et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html).

## 3. Where the current Dice is lost

### 3.1 Proposal supply is already sufficient

| Quantity | Overall | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|
| Selected G1+fusion Dice | 0.288729 | 0.157723 | 0.435229 | 0.386874 |
| Eligible gallery oracle | 0.527902 | 0.331101 | 0.730251 | 0.746247 |
| Full gallery oracle | 0.528298 | 0.331876 | 0.730251 | 0.746247 |
| Proposal truncation regret | 0.000396 | 0.000776 | 0 | 0 |
| Selector regret | **0.239173** | 0.173377 | 0.295022 | 0.359374 |

The gallery oracle exceeds fully reference A overall, medium and large; only
small is lower by `0.013731`.  The next experiment should not spend GPU on
adding proposals, SAM prompt modes or higher proposal resolution.  The useful
mask already exists in almost every bag.

### 3.2 Candidate identity is the primary global bottleneck

- the oracle candidate has median/mean/P90 fusion rank `32/51/138`;
- top-3/top-10/top-50 oracle bounds are `0.341837/0.399326/0.475500`;
- `49/184` baseline choices have zero overlap, yet all 49 have a recoverable
  gallery candidate; 31 have oracle Dice at least 0.1 and 12 at least 0.5;
- G1 candidate quality Spearman is only `0.563` overall and `0.460` for small;
- upstream rank is only `0.333`; SAM predicted-IoU rank is `0.016` overall and
  `-0.061` for small.

Selector regret decomposes into `0.168376` within the selected source and
`0.070796` from choosing the wrong source.  Therefore **70.4% of the missing
Dice is within-source identity/ranking**, not source routing.  A source prior
cannot solve the dominant term.

The frozen scores often actively suppress the oracle: it ranks below the
selected mask in `84.2%/87.0%/70.7%` of images under G1/upstream/SAM ranks.
Changing global fusion coefficients cannot rescue a candidate that all
available signals rank poorly.

### 3.3 Extent has opposite signs and must be separated from identity

| Group | Median selected/GT area | Mean precision | Mean recall | Mechanism |
|---|---:|---:|---:|---|
| `<1%` | **14.603×** | 0.148 | 0.471 | correct vicinity mixed with much normal tissue; 35 misses |
| `1-<5%` | 1.098× | 0.489 | 0.546 | area roughly plausible, identity/location wrong; 13 misses |
| `>=5%` | **0.382×** | 0.713 | 0.360 | discriminative fragment only; 1 miss |

Among all tumors, 54 severe-over cases have Dice only `0.0209` and are almost
entirely small; 31 under-extent cases have Dice `0.1477` and are mostly
medium/large.  One threshold, dilation, erosion or global area prior must harm
one end while helping the other.

The scale-expert feasibility result proves that conditional correction is
useful: fixed betas `-1/0/+0.5` would yield
`0.181215/0.435229/0.501086`, and true-group routing would reach `0.311904`.
But proposal-area gates reach only `0.268845-0.278291`; over-segmented small
masks already look large, so area routing is a self-reinforcing error.

### 3.4 Consensus shows coverage, not trustworthy supervision

The best-per-source union rescues 31/49 misses and reaches recall `0.7481`, but
micro precision is only `0.0663` overall and `0.00575` for small.  Pixels
outside the union still contain `25.2%` of lesion pixels.  Neither union
foreground nor outside-union background is a safe pseudo label.  This explains
why the pseudo-mask U-Net memorized train pseudo masks (`0.738` agreement) but
stopped near `0.230` validation Dice.

## 4. What each failed family actually ruled out

| Family | Observation | Mechanistic lesson |
|---|---|---|
| LayerCAM/PCAM/high-res CAM | best 0.234; PCAM loses tiny lesions; 512 alone lower | final-grid resolution and discriminative-part bias are real; resize alone does not create local supervision |
| SAM selector | SAM-IoU rank is uncorrelated with true Dice | SAM is useful proposal supply, not an image-label lesion selector |
| G2 negative-only/hard-top MIL | shortcut removal improves raw result, final best only 0.254 | normal bags supervise negatives; positive tumor bag still does not identify its lesion instance |
| BAS/softplus foreground controls | collapse, area/activation shortcuts | global classifier response is not candidate-local tumor identity |
| Matched-normal transplant | 0.281; small falls to 0.096; matched≈random | frozen DenseNet transplant measures area/style; tumor-specific signal is absent even at early layers after sham cancellation |
| Cross-view co-witness | full≈control, residual correlation 0.9987 | Cartesian bag pooling learns shared anatomy/source escape, not the responsible proposal |
| Top-10 relational/consensus | helps medium/large, significantly harms small | cross-source agreement often represents common bone anatomy and broad masks |
| Density/NRCE/causal post-hoc scores | weak pixel rank, near-zero true-area Dice | rarity or classifier causal effect on the old embedding is not tumor-specific local evidence |
| Counterfactual/SynRad/OLV | near-zero masks or border shortcut | direct image-label objectives can solve shortcuts while spatial Dice remains zero |
| Hard pseudo consumer | train fit high, validation unchanged | a downstream segmenter cannot repair biased seeds by itself |
| Subtype/area hard routing | 0.2773/0.2688-0.2783 | subtype and proposal area are confounded proxies, not a deployable burden gate |

This leaves a narrow target.  Another proposal, threshold, source weight,
post-hoc frozen score, pseudo consumer or longer training of the same objective
is unsupported.

## 5. Alignment with primary research

The local evidence agrees with, rather than merely resembles, several primary
results:

- [Small Objects Matters in WSSS (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html)
  shows systematic small-object failure and gains from size-balanced training.
- [ICD (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Learning_Integral_Objects_With_Intra-Class_Discriminator_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2020_paper.html)
  identifies the classification-boundary mismatch: inter-class recognition
  does not separate foreground from background within a positive image.  G2
  and every near-zero direct teacher reproduce this exact gap.
- [GLAM (MIDL 2021)](https://proceedings.mlr.press/v143/liu21b.html) uses
  coarse ROI localization plus fine high-resolution local segmentation because
  lesions are small relative to the full image.  BTXRD's 0.095-cell median on
  a 10×10 grid quantitatively requires the same local-resolution principle.
- [PYLON](https://arxiv.org/abs/2010.11475) explicitly shows that increasing
  feature-map resolution is insufficient because the weak classification task
  and pooling/normalization can still damage localization.
- [SEAM (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html)
  supplies spatial equivariance missing from image labels; it is relevant as a
  representation constraint, not as a rescue for bad frozen CAM seeds.
- [Sub-category Exploration (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Chang_Weakly-Supervised_Semantic_Segmentation_via_Sub-Category_Exploration_CVPR_2020_paper.html)
  supports using fine image labels to make the classifier attend beyond one
  discriminative part.  BTXRD already supplies nine tumor subtypes, so ignoring
  them in a binary representation is unnecessarily restrictive.
- [INSIGHT (MLHC 2025)](https://proceedings.mlr.press/v298/zhang25a.html)
  combines small-kernel local detection with broader context to suppress local
  false positives.  This directly matches the small-lesion precision problem.

The papers do not imply that copying a named module will solve BTXRD.  Together
with our outputs, they specify the missing inductive biases: fine local
resolution, within-positive foreground/background discrimination, subtype-
conditioned evidence, context suppression and spatial consistency.

## 6. Causal bottleneck statement

The present ceiling is caused by a conjunction, not one scalar hyperparameter:

1. **representation bottleneck:** late binary classifier features erase tiny
   lesions and retain anatomy/style;
2. **positive-instance identifiability:** normal bags give dense negatives,
   but a tumor image label does not identify which of ~177 proposals is causal;
3. **signed extent:** small requires precision/shrinkage, large requires
   recall/expansion, and medium requires location;
4. **calibration:** source identity and candidate area are easier shortcuts
   than lesion evidence;
5. **evaluation confounding:** lesion size is correlated with subtype and the
   large group has only 18 cases.

The decisive evidence is the combination of gallery oracle `0.528298`,
near-zero truncation regret, within-source regret `0.168376`, opposite extent
signs and failed conditional signals from every frozen representation.

## 7. Successor recommendation — revise, do not launch MNR-v1 as written

The existing MNR-v1 design is directionally right about stride-4 matched-normal
evidence, but it has two unresolved defects:

- it uses a binary positive-bag MIL objective, so it can repeat G2's
  positive-instance ambiguity;
- its candidate readout takes only the top 17 inside cells minus a ring, which
  is largely insensitive to whether a same-location candidate has the correct
  full extent.

The recommended successor is **Subtype-conditioned Matched-Normal
Intra-class Local Evidence (SMILE) + immutable rich gallery**:

### Representation

1. DenseNet/FPN or equivalent at 512 px, stride 4 (`128×128` map), no absolute
   coordinate channel and no global-classifier bypass.
2. Four appearance-/view-matched train-normal references; query-vs-normal
   soft local matching is trained end to end, not applied to a frozen classifier.
3. Two heads sharing the encoder:
   - binary tumor/normal for robust normal rejection;
   - ten-class subtype local evidence using the supplied image label.
   Subtype is conditioning, not a hard size gate.
4. Dense negative-cell loss on all normal images, sparse multi-pool positive
   loss on tumors, subtype-balanced batches, flip/scale equivariance and
   reference-swap consistency.
5. An intra-class foreground/background objective: local evidence must be high
   inside a softly selected candidate cluster and lower in its ring, while the
   cluster posterior remains soft across the whole bag.  No detached hard
   winner and no source ID may enter the head.

### Separate identity from extent

For each existing gallery candidate compute two frozen annotation-free scores:

- **identity:** top local subtype/abnormal evidence inside minus local ring;
- **extent compatibility:** evidence mass captured inside the candidate minus
  positive-evidence leakage immediately outside it, normalized by total local
  evidence rather than raw candidate area.

Use them as a zero-initialized residual over the exact
`0.5*rank(G1)+0.5*rank(upstream)` baseline.  Report an identity-only readout and
an identity+extent readout fixed before polygons.  This exposes whether failure
comes from localization or extent instead of hiding both behind one Dice.

### Fast matched run

Run two capacity-matched T4 arms in parallel for exactly two canonical passes:

1. query-only subtype-conditioned local model (control);
2. matched-normal subtype-conditioned local model (full).

Reuse the existing reference cache.  The only prelaunch checks are exact
split/cache hashes, no-GT/no-test inspection, one real forward/backward and
zero-residual baseline reproduction.  Do not rebuild the gallery or run a
threshold/resolution/epoch sweep.

After training, freeze 371 maps and candidate choices, then always report
actual Dice/IoU overall and `94/72/18`.  Also report within-source rank gain,
49-miss recovery, small precision and large recall.  Proxies explain the Dice;
they do not veto evaluation.

Stop the family after this run if the full arm does not exceed both the
`0.288729` baseline and capacity-matched control, or if any gain is explained
only by source/area shift.  If identity improves but extent does not, retain
the encoder and revise only the extent readout; do not retrain another global
selector.

## 8. Reproducible local evidence and provenance

- Dataset analysis script:
  `project/analyze_btxrd_original_dataset_bottleneck.py`.
- Dataset summary/audit SHA-256:
  `0a23152fd7a342be9c0e0d8a1a78d2c8413156ebc8955172d29670fdbd0c2a1d` /
  `audit_pass=true`; exact 184 validation tumors; test images opened `0`.
- Subtype gate script:
  `project/analyze_rich_gallery_label_conditioned_scale_gate.py`.
- Subtype summary SHA-256:
  `48383c9090cf5704d28ef24073a027d27e900667334890c5ea7882b754189ae6`;
  group-separated subtype routing Dice `0.277330`; test read `0`.
- Candidate table SHA-256:
  `ea3fd29d5fac7de46bd846bfcf65e87be0041875a0c11e8a9303c6f4fe95c73c`.
- Canonical split SHA-256:
  `7b16771a634e423d2d4ce7d5a835e6ea5ff6d1a422f124aab8019ed53512529c`.

No successor kernel was launched during this synthesis.
