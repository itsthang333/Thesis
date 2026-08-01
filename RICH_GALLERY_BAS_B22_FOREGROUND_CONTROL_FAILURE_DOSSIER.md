# BAS-B2.2 foreground-control failure dossier

## Audited outcome

The matched five-epoch probe changed only the B2.1 localization objective.
Independent audit verified all 371 maps and predictions, checkpoint/source
hashes, and no validation polygons or test access.

- final train accuracy: `0.765850`;
- validation image AUROC: `0.735294`;
- final foreground CE: `0.463459`;
- final full-image CE: `0.985558`;
- mechanics gate: **FAIL**, solely because full-image CE exceeds `0.69`.

B2.2 repairs the exact zero-map optimum of B2.1, but it does not identify
tumor-specific spatial evidence.

## Exact objective geometry

For a fixed nonnegative target-class map `C`, full score `S = mean(C)`,
localization map `M`, and foreground score `S_fg = mean(C*M)`, B2.2 uses

`L(M) = 1.5 * (0.5 - S_fg / stopgrad(S)) + 1.2 * mean(M)`.

The reference ratio `0.5` is an additive constant: it changes neither the
gradient nor the optimum. For each cell `i`,

`dL/dM_i = (1.2 - 1.5*C_i/S) / N`.

Ignoring the sigmoid parameterization, the box-constrained optimum is

`M_i = 1 iff C_i/mean(C) > 1.2/1.5 = 0.8`.

This condition detects above-average class evidence, not tumor. Common
anatomy, acquisition style, borders, and other image-label shortcuts satisfy
the same inequality. Foreground CE encourages retaining enough class evidence
but provides no pixel-negative constraint.

The implementation also gathers only the ground-truth image-label channel.
Normal images train the normal map and tumor images train the tumor map, so the
tumor channel receives no direct dense-negative supervision from the 1,493
normal training images. High image AUROC can therefore coexist with a
non-specific tumor map.

## Tensor evidence: empty collapse became diffuse saturation

Across 184 tumor validation images:

- median activation mean: `0.496241`;
- median fraction of cells `>=0.90`: `0.352838`;
- median fraction of cells `>=0.99`: `0.255740`;
- median effective support: `0.576708`;
- median top-1%-mass: `0.020563`;
- argmax on the outer 10% border: `0.467391`;
- mean border-minus-interior activation: `-0.271584`.

The map is no longer empty or a numerical spike. It is a broad, strongly
bimodal anatomy map: `0.726` of cells are `<=0.1` or `>=0.9`. Tumor-channel
maps on normal images are also broad (median mean `0.418861`), independently
confirming failure of dense tumor/background discrimination.

## Why candidate scoring is an area proxy

The BAS candidate score min-max normalizes each activation and computes the
harmonic mean of coverage and purity. With a broad centered activation field,
coverage grows with mask area while purity mostly measures whether a mask lies
on common anatomy. The within-image Spearman correlation between BAS score and
candidate area is mean/median `0.933174/0.950055`; `81.52%` of tumor bags exceed
`0.90`. This is not the independent candidate-identity observable missing from
G1/upstream.

Full CE reached its minimum `0.850467` at epoch 4 and worsened to `0.985558`
at epoch 5 while train accuracy increased. More epochs or validation-selected
epochs cannot repair the objective's identifiability defect.

## Consequence

1. Retire B2.2 after the one frozen exploratory spatial evaluation; do not
   sweep epoch, weight, threshold, resolution, or seed.
2. Preserve G1+upstream at validation Dice `0.2887294867`.
3. Require the successor to score each candidate using tumor-specific content,
   direct tumor-channel negatives from normal images, and signed extent
   evidence rather than image-global anatomy or area.
4. Test remains locked.

## Provenance

- checkpoint SHA-256:
  `353d1f2113b0c7a1a30e3d1fa8d0b16bb459d282eba2e6e222c421266bcfe22c`;
- probe summary SHA-256:
  `13b5f2123414cd631115bd2b879c90bdbe361ea50b38c0e0f27dbb70620a1c5d`;
- independent audit SHA-256:
  `a506dc6ccd9994a227d1f04d927c96a4880ddc047121cf64bb570346aa3ad46e`;
- validation: 371 images, 184 tumors, subgroups 94/72/18;
- validation polygons used during probe: false;
- test evaluated: false.
