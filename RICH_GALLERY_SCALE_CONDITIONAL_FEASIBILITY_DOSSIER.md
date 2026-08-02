# Rich-gallery scale-conditional feasibility dossier

## Question

The immutable G1+upstream selector has opposite extent errors:

- `<1%`: selected masks are usually too large;
- `1-<5%`: area is approximately right but candidate identity/location is
  often wrong;
- `>=5%`: selected masks are usually too small.

This diagnostic asks whether separate global area preferences can improve all
three groups, and whether a gate derived from frozen proposal areas can choose
the correct preference without spatial ground truth.

## Bound input and calculation

The analysis consumes the already frozen `32,519`-row validation candidate
table SHA-256
`ea3fd29d5fac7de46bd846bfcf65e87be0041875a0c11e8a9303c6f4fe95c73c`.
No score or candidate is regenerated.  For each image, let

`b_i = 0.5 rank(G1_i) + 0.5 rank(upstream_i)`

and define a scale expert

`s_i(beta) = b_i + beta * (rank(log area_i) - 0.5)`.

The fixed retrospective grid is
`beta in {-2,-1,-0.5,-0.25,0,0.25,0.5,1,2}`.  Negative beta prefers compact
candidates, zero exactly reproduces the baseline and positive beta prefers
broad candidates.  Candidate Dice and true size groups are read only after the
candidate table is frozen, so best-beta and oracle results are feasibility
bounds, not deployable selectors.

## Result: three experts can improve all groups

The best beta is exactly signed as the failure analysis predicts:

| True group | Best beta | Baseline Dice | Expert Dice | Delta |
|---|---:|---:|---:|---:|
| `<1%` | `-1.0` | `0.157723` | `0.181215` | `+0.023492` |
| `1-<5%` | `0.0` | `0.435229` | `0.435229` | `0` |
| `>=5%` | `+0.5` | `0.386874` | `0.501086` | `+0.114212` |

True-group routing would reach overall Dice `0.311904`, above the immutable
`0.288729` baseline.  The per-image oracle over the same three experts is
`0.350692`.  Thus the expert family has real supply; the opposite small/large
extent errors are not mutually irreconcilable once the policy is conditional.

## Result: proposal area cannot provide the gate

Three annotation-free area-only gates were evaluated with the expert settings
above:

| Gate feature | Size accuracy | Routed Dice |
|---|---:|---:|
| Baseline-selected area | `44.57%` | `0.278291` |
| Top-five median area | `45.11%` | `0.271403` |
| Median best-per-source area | `42.93%` | `0.268845` |

Their Spearman correlations with the true size ordinal are only
`0.0761/0.0964/0.0428`.  The baseline-selected median predicted areas are
`2.02%/2.64%/4.08%` for true small/medium/large; the distributions overlap too
strongly.  This is a direct consequence of the original bias: an over-segmented
small lesion looks medium/large if its erroneous proposal area is used as the
gate.  Therefore hard routing on predicted mask area creates a feedback loop
and underperforms the baseline.

## Mechanistic decision

The user's conditional-policy hypothesis is supported, but the next problem is
not another morphology/area sweep.  It is **latent lesion-burden
identification** without spatial labels.  A successor should keep the three
signed experts but learn a soft gate from signals that are less downstream of
the erroneous mask:

1. high-resolution candidate appearance and tumor-evidence density;
2. cross-source spatial/score consensus and disagreement;
3. image-level tumor confidence and normal-reference rejection;
4. cross-view co-witness evidence when available.

Proposal area may be one calibrated input, but cannot be the gate by itself.
The gate must be trained/frozen without polygon or true subgroup labels and
must route by a soft mixture rather than an oracle hard branch.  The currently
running cross-view diagnostic is therefore useful twice: it tests a candidate
identity residual now, and its learned evidence can become a gate feature if it
contains signal even when its standalone Dice does not promote.

## Provenance

- Summary SHA-256:
  `47ec2c711a42a5ca2efe6e2d4419bdc71bd8f4287df00297a2c2d414a70c5430`.
- Per-image feature SHA-256:
  `2ed22cc34a64232a08380acd8136cebdae0e24fb758659c42ef0a8112ed98ac0`.
- Audit SHA-256:
  `970fbd271a771b20c048fd6736487c44ce4c9a13e55cb9027a84ce0fcf97300c`.
- Exact cohort: `184` tumor images, subgroups `94/72/18`.
- Validation GT was used retrospectively; true-group routing is explicitly not
  deployable.  Test was not read or evaluated.
