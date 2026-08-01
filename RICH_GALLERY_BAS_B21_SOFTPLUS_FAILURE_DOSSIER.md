# BAS-B2.1 Softplus failure dossier

## Audited outcome

The five-epoch matched probe changed only the terminal classifier-map
activation from ReLU to Softplus. Independent output audit passed for all 371
validation images; validation polygons and test were never opened.

- final train accuracy: `0.839651`;
- validation image AUROC: `0.743199`;
- final full-image CE: `0.378285`;
- final foreground-guidance CE: `0.693147182`;
- mechanics gate: **FAIL** (`activation_range` and `nondegenerate_fraction`).

Softplus repairs the dead classifier but does not repair spatial localization.

## Tensor-level localization collapse

Across 184 tumor images:

- mean maximum localization probability: `2.237e-07`;
- mean implied maximum sigmoid preactivation: `-15.501`;
- mean maximum sigmoid derivative: `2.237e-07`;
- median effective support: `0.001338` of the 56x56 grid;
- median mass inside the largest 1% of cells: `0.931514`;
- argmax at the outer 10% border: `1.000`.

The map is not a weak tumor map. It is a sigmoid-saturated near-zero map plus a
padding-border numerical spike. At preactivation about -15.5, the sigmoid
gradient is attenuated by roughly four to seven million times; continuing
epochs cannot plausibly recover spatial structure.

## Exact objective loophole

Let `M` be the localization map, `S` the target-class full-image activation and
`S_bg(M)` the activation after erasing `M`. The transferred loss is

`L_BAS = gate(S_bg/S) + 1.2*mean(M)`,

where the official implementation sets the ratio to zero whenever
`S_bg >= S`. At `M=0`, the erased and full feature maps are identical, so
`S_bg(0)=S`. The hard gate therefore returns zero and the area term is also
zero: `L_BAS(M=0)=0`.

Thus the empty map is an exact global minimizer of the localization objective.
The area term initially pushes sigmoid logits downward; once saturated, every
remaining localization gradient is multiplied by `M(1-M)`. Meanwhile the
full-image classifier can independently reduce CE and reach AUROC 0.743. The
fixed foreground CE at `log(2)=0.69314718` confirms that no class information
survives through the near-zero map.

## Cross-experiment meaning

B2 failed at the classifier head (`all-zero class logits`, tumor map near one).
B2.1 fixes that exact defect and exposes the next independent defect: the
hard-gated background ratio plus area loss admits a zero-map optimum. Neither
failure tests useful BAS candidate evidence, and neither score may be fused
into the `0.2887294867` baseline.

## Research decision

1. Retire the hard-gated background-ratio objective; do not extend epochs,
   sweep area weight, threshold numerical spikes, or run spatial GT.
2. A successor is allowed only if its foreground objective has a nonzero,
   spatially selective gradient at `M=0` and includes border/support gates.
3. The supported correction is a continuous foreground-control ratio
   `R - S_fg/S` plus area constraint. Its cell derivative is proportional to
   `lambda_area - lambda_fgc*C_i/S`; strong class-evidence cells are pushed up
   while weak cells are pushed down.
4. This correction must first pass a bounded label-safe mechanics probe; only
   then may it score candidates and report actual Dice against `0.2887294867`.

Primary references: [BAS (CVPR 2022)](https://arxiv.org/abs/2112.00580) and
[foreground control for image-label-only chest-X-ray localization (Scientific
Reports 2024)](https://www.nature.com/articles/s41598-024-79701-8).
