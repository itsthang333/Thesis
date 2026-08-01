# Rich Gallery BAS-B2.2 foreground-control mechanics probe

## Decision being tested

B2.1 proved that Softplus repairs the binary classifier but its localization
map still collapses: all 184 tumor-map maxima lie on the outer border, median
effective support is 0.001338 and the maximum sigmoid derivative is only about
2.24e-7.  The cause is not insufficient training.  The transferred BAS term
sets its background ratio to zero when background activation is not below the
full activation.  At an empty map this condition is exact and the area term is
also zero, so the empty map is a global minimizer.

B2.2 changes one scientific component only: replace that hard-gated
background ratio with the continuous foreground-control ratio described for
image-label-only chest-X-ray localization by Wang et al. (Scientific Reports,
2024).  Backbone, image size, initialization, optimizer, seed, five train
passes, Softplus classifier, foreground CE, data and validation-label firewall
remain matched to B2.1.

## Frozen objective

For the target class, let `S` be the full-image activation, `S_fg` the
activation retained by localization map `M`, and `A(M)=mean(M)`:

`L = CE_full + 0.5 CE_fg + 1.5 (0.5 - S_fg/(stopgrad(S)+1e-8)) + 1.2 A(M)`.

The constant 0.5 does not affect the gradient.  At `M=0`, for class-evidence
cell `C_i` on a grid of `N` cells,

`dL_spatial/dM_i = (1.2 - 1.5*C_i/S)/N`.

Therefore cells with relative target evidence above 0.8 receive an expansion
gradient, while weak cells receive a shrinking gradient.  This directly closes
the all-cells-shrink loophole observed in B2.1.

## Predeclared mechanics gates

This run does not open validation polygons and cannot claim a Dice gain.  Full
candidate scoring is authorized only if every gate passes:

- final full-image CE <= 0.69;
- final foreground CE <= 0.68 (must improve materially over log(2));
- final accuracy differs from the majority prior by >= 0.01;
- validation image AUROC >= 0.55;
- mean activation range >= 1e-3;
- >= 50% of tumor maps have range > 1e-4;
- tumor argmax outer-border fraction <= 0.50;
- tumor median top-1%-mass fraction <= 0.75;
- tumor median effective-support fraction >= 0.003;
- mean candidate-score/area Spearman <= 0.98.

If any gate fails, the foreground-control BAS family is retired without epoch,
threshold, seed, resolution or loss-weight sweeping.  If all pass, the exact
checkpoint may be used to score the immutable rich gallery and a separate
frozen Stage-B evaluator must report actual binary-mask Dice against the
0.2887294867 validation baseline.

Validation polygons and test data are prohibited from this mechanics probe.
