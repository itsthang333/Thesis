# Rich-gallery G2 terminal failure dossier

## Scope and validity

This dossier analyzes the completed private/offline validation-only run
`wanwin/btxrd-rich-gallery-g2-selector-pair` version 3.  The run froze eight
selector variants for all 371 validation images before opening validation
polygons.  The in-kernel Stage-A and Stage-B audits passed, reproduced the G1
choice for 371/371 images, recorded 184 tumors split 94/72/18, and report zero
test reads.  The result therefore answers the scientific question even though
it does not meet the promotion gate.

The immutable protocol SHA-256 is
`6dd6c66e396054157eb498cbca46635d1058c73d1139541d385bb769088801be`.
The prediction-freeze and evaluation-summary SHA-256 values are
`78970d417de20dc884958dfb3fd9cb2bad9f2cda53240ae2207607ba827cd167`
and `813b5a0c9506d2052508f1a8ffe2a401e947183bef452eaef3330b3a48cdd9b6`.

## Actual binary-mask result

| Variant | Overall Dice | IoU | <1% | 1-<5% | >=5% | Misses |
|---|---:|---:|---:|---:|---:|---:|
| G1 frozen raw | 0.206026 | 0.151407 | 0.109585 | 0.300705 | 0.330950 | 29 |
| G1 frozen + rank fusion | **0.288729** | **0.216839** | **0.157723** | **0.435229** | **0.386874** | 49 |
| G2 flat hard-top raw | 0.248502 | 0.183757 | 0.135015 | 0.362974 | 0.383267 | 44 |
| G2 flat hard-top + fusion | 0.251741 | 0.188475 | 0.136681 | 0.373768 | 0.364498 | 64 |
| G2 flat negative-only raw | 0.240116 | 0.176914 | 0.135806 | 0.348201 | 0.352502 | 43 |
| G2 flat negative-only + fusion | 0.250350 | 0.185825 | 0.139743 | 0.372380 | 0.339840 | 62 |
| G2 hierarchical negative-only raw | 0.223774 | 0.163702 | 0.132132 | 0.315446 | 0.335664 | 47 |
| G2 hierarchical negative-only + fusion | **0.254326** | **0.190661** | **0.137925** | **0.376147** | **0.374906** | 62 |

No G2 arm beats the frozen G1 rank-fusion reference.  The best G2 arm is
hierarchical negative-only plus rank fusion at 0.254326, a paired mean loss of
0.034404 Dice; its complete-group bootstrap 95% interval is
[-0.058333, -0.014022].  All three primary subgroup gates fail.  The rich
gallery oracle remains 0.528298 overall (0.331876/0.730251/0.746247), so this
is a selector failure, not a proposal-supply failure.

There is nevertheless one real positive result: removing the external-source
shortcut from the training loss raises flat hard-top raw Dice from 0.206026 to
0.248502, a paired gain of 0.042476 with bootstrap 95% interval
[0.010932, 0.075269].  This confirms that the shortcut diagnosis was causal,
but it is not sufficient to solve the selector.

## Why the oracle rose while selected Dice fell

Oracle Dice is a maximum over candidates.  Appending candidates can only keep
or increase it:

`D_oracle(B union C) = max(max_{b in B} D(b), max_{c in C} D(c))`.

The learned selector must instead identify one mask from a much larger bag.
The rich tumor bags average 176.7 candidates, while the old gallery had about
56.  The best mask has median rank 35.5 under G1 and is top-1 in only 3.26% of
tumors.  Thus the gallery improved coverage but made the latent-instance
identification problem much harder.  Oracle growth is not evidence that a
weak image-label objective can identify the added good mask.

The original G1 objective also had a perfect source-presence shortcut:
external proposals occur in 184/184 tumor validation bags and 0/187 normal
bags.  Candidate count alone has image AUROC 0.89449.  G1 can lower image BCE
without learning lesion localization, explaining the simultaneous high image
AUROC and low Dice.

## What G2 fixed

G2 excludes the external source from every training loss and compares matched
pooling/instance-label mechanisms.  This reduces gross over-segmentation:

- G1 raw median selected/GT area ratio: 13.02, with 29 misses.
- Flat hard-top raw: 3.05, with 44 misses.
- Hierarchical negative-only raw: 2.64, with 47 misses.

The raw hard-top gain is therefore real: the model relies less on huge
external masks and improves all three subgroup means.  This is why G2 raw
reaches 0.248502, slightly above the published Geometry-v3 mean 0.245482,
despite using the harder rich gallery.

## Why G2 still fails

### 1. Positive-instance non-identifiability remains

For a positive bag, normalized LogMeanExp only requires at least one candidate
to receive a high logit.  Negative-only instance BCE gives exact supervision
only to normal bags; it gives no information about which candidate in a tumor
bag is the lesion.  Source normalization removes multiplicity bias, but it
does not create a positive spatial label.

The effective candidate count confirms collapse.  The flat hard-top arm falls
from 22.39 to 1.87; flat negative-only from 22.39 to 1.74; hierarchical
negative-only from 61.76 to 1.74.  Ending every schedule at temperature 0.2
therefore recreates an almost hard top-1 objective.  The hierarchy spreads
early gradients but converges to the same identifiability failure.

Hard-top is the strongest raw G2 arm because its noisy detached winner at
least supplies a positive-instance signal.  Negative-only arms are safer but
cannot learn a within-positive-bag ordering.  This is the central trade-off:
hard-top is biased; negative-only is under-identified.

### 2. The extent correction trades overlap for misses

G1 rank fusion reduces median selected/GT area ratio from 13.02 to 2.04 and
raises Dice to 0.288729, but misses rise from 29 to 49.  G2 fusion produces a
nearly identical median area ratio (1.96-1.99) yet raises misses further to
62-64.  Its masks are not too large anymore; they are more often on the wrong
location.  The bottleneck after shortcut removal is therefore lesion-hit
recall/positive-instance ranking, especially for <1% lesions, not another
global extent or threshold adjustment.

### 3. G2 destroys the complementarity that made fixed fusion useful

Rank fusion adds 0.082703 Dice to G1 raw, but only 0.003239 to G2 hard-top,
0.010234 to G2 flat negative-only, and 0.030551 to G2 hierarchical.  Mean
within-image Spearman correlation between model and upstream candidate ranks
rises from 0.1868 for G1 to 0.2454/0.2437/0.2273 for the three G2 arms.  G2 has
partly learned the same coverage/purity preference already carried by the
upstream score, so averaging the two ranks is redundant rather than
complementary.

The G2 fusion choices equal the G1-fusion choice on only 47.8-51.1% of tumor
images.  On the changed half, mean Dice falls by about 0.070.  The largest
concentrated loss is a G1-fusion classifier448 choice being replaced by a
LayerCAM choice: 30 images and -3.665 total Dice for hard-top fusion; 28 images
and -4.206 for hierarchical fusion.  Same-source classifier448 selections are
not the problem--they improve slightly.  The failure is cross-source
calibration plus within-source ranking, not simply external-source leakage.

### 4. Why collaborator Geometry-v3 improved while rich G1/G2 did not

Geometry-v3 repaired a deterministic coordinate error: RAD-DINO descriptors
were mapped through direct resize even though the encoder used centered square
padding.  Correcting the coordinate frame aligned the descriptor with the
same physical proposal mask on the same gallery.  It did not enlarge the bag,
change source composition, or ask image labels to resolve more latent
instances.  It therefore raised Dice from 0.219492 to 0.245482 through a
causal representation correction.

Rich G1 changed the statistical problem: approximately tripled candidates,
introduced a label-perfect source-presence feature, and increased the oracle
rank depth.  G2 removed the easiest shortcut, but its weak objective still has
no reliable positive-instance target.  The two outcomes are therefore not a
contradiction: the collaborator fixed observed geometry; our first rich run
added latent ambiguity faster than it added selector information.

## Bottleneck decision and next research constraint

The active bottleneck is now **positive-instance ranking and hit recall inside
the rich bag**, followed by source calibration.  Proposal coverage, global
threshold, raw resolution, and source-count normalization are not the primary
remaining limits.

The next mechanism must add annotation-free candidate-level positive evidence
rather than another temperature/threshold/source-weight sweep.  A valid cheap
diagnostic should test cross-source spatial consensus and causal candidate
necessity on the frozen gallery, with these predeclared requirements:

1. preserve or improve the 49-miss G1-fusion hit rate, especially the 35 small
   misses;
2. improve candidate ordering within classifier448 and LayerCAM separately;
3. retain complementary information to upstream rank rather than reproduce it;
4. freeze all choices before validation polygons and compare paired against
   0.288729;
5. reject the mechanism if its gain comes only from source identity, candidate
   count, per-image area, or validation-GT routing.

No further sweep of G2 temperature, epoch, threshold, resolution, or rank
weight is justified by this result.
