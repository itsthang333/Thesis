# SMILE + immutable rich gallery failure dossier

## Decision

SMILE is retired in its current readout form. Frozen validation Dice/IoU is
`0.249431770/0.183282792` versus the exact G1 + fixed-rank baseline
`0.288729487/0.216839181`. Subgroup Dice changes from
`0.157723/0.435229/0.386874` to `0.071768/0.417140/0.506398`. The paired
complete-group bootstrap interval for the overall Dice delta is
`[-0.071630,-0.000764]`.

## What changed

- Complete misses fall from `49` to `35`: the local evidence is not empty.
- Precision/recall shifts from `0.3366/0.4893` to `0.2448/0.6286`. SMILE buys
  recall at a larger precision cost.
- Median selected/GT area becomes `81.13x/2.06x/0.89x` for
  small/medium/large, versus `14.60x/1.10x/0.38x` for the baseline.
- Baseline-to-primary choices improve/harm `13/34` small, `20/23` medium and
  `8/5` large images.
- The full arm is only `+0.00672` Dice above the query-only control on average;
  this matched-normal signal is too weak to overcome the readout bias.

## Root cause at candidate level

- The fixed-top-17 identity statistic is positively associated with candidate
  area within an image: median Spearman `0.6503`, positive in `99.5%` of
  evaluable images. A larger mask has more chances to contain 17 extreme
  evidence cells, while the surrounding-ring median is not an equal-size null.
- The soft extent score is even more area coupled: median within-image
  Spearman `0.7663`, positive in `100%` of images. Image-label MIL provides
  sparse discriminative evidence, not a calibrated lesion-occupancy map, so
  soft Dice between `sigmoid(evidence)` and a proposal cannot identify extent.
- Median within-image rank correlation with true candidate Dice is `0.5851`
  for the baseline and `0.6048` for the combined residual. This small global
  ordering gain does not transfer to top-1 selection: near the top of the
  baseline ranking, the scale-biased residual promotes larger wrong masks.
- Selected sources move from classifier/external/LayerCAM `88/24/72` to
  `75/38/71`. Increased external-source selection is a secondary shortcut;
  selector regret rises from `0.239569` to `0.278867`.

## Mechanistic interpretation

The evidence map contains useful presence signal: recall, recovered misses and
large-lesion Dice improve. The failure is the conversion from presence to
candidate identity and extent. Both residual statistics reward scale, so the
same correction expands masks in every subgroup. That is directionally correct
for under-segmented large lesions and catastrophic for tiny lesions that are
already over-segmented.

No additional successor is launched. Research is closed for the thesis and the
final method remains G1 plus equal within-image percentile-rank fusion. BTXRD
test was not opened.
