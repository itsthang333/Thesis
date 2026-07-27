# Fractional mask-pooling contingency for small BTXRD proposals

Date: 2026-07-28  
Status: source/method audit only; not part of geometry-correction v3 and not
authorized for execution

## Finding

After a proposal is area-resampled to the RAD-DINO token grid, the current
inside descriptor is:

`inside = sum_p(w_p f_p) / max(sum_p(w_p), 1)`.

The proposal is accepted when `sum_p(w_p) >= 0.25`. Therefore, for accepted
proposals whose token-grid mass `s` is between 0.25 and 1, the descriptor is
not a weighted mean. It is:

`inside = s * weighted_mean(f)`.

The same floor is used for the context denominator, although a radius-two
context normally has mass above one. The deterministic effect for the
proposal descriptor is:

| Grid mass `s` | Appearance multiplier |
| ---: | ---: |
| 0.25 | 0.25 |
| 0.50 | 0.50 |
| 0.75 | 0.75 |
| >=1.00 | 1.00 |

This does not invalidate the running experiment. It may have been intended as
a confidence penalty for sub-token support. But it also entangles proposal
appearance with proposal size even though log area is already included as an
explicit metadata feature. For a small lesion, a semantically strong proposal
can thus have all three layer-wise inside/context/contrast blocks altered by
support size before the MLP sees the explicit area feature.

This is a plausible small-only failure mechanism. It is **not** corrected in
`rad_dino_mask_bag_mil_descriptor_geometry_correction_v3`, because changing
the denominator is a scientific pooling ablation rather than a coordinate
implementation correction.

## Evidence required before an ablation

The regenerated, hash-frozen physical candidate payloads permit a GT-blind
diagnostic before optimizer construction:

1. after v3 coordinate projection, save each retained proposal's token-grid
   mass;
2. report counts/quantiles in `[0.25,0.5)`, `[0.5,1)`, `[1,2)` and `>=2`;
3. stratify only by allowed train/validation image label, fallback status,
   prompt mode and proposal source—not by lesion size or validation mask;
4. save the same values for original and flip and require aligned validity;
5. freeze every row and hash before training and before any validation GT
   import.

After predictions are frozen, the evaluator may report whether selected
small-group candidates disproportionately occupy the sub-token strata and
whether complete misses concentrate there. This is diagnosis, not permission
to change v3.

If almost no retained candidate has mass below one, reject this hypothesis
without a training ablation. If a material fraction does, a separate protocol
may compare:

- **parent:** divide by `max(s,1)`;
- **true weighted mean:** divide by `max(s, epsilon)` after the unchanged
  minimum-mass filter, with one predeclared numerical epsilon.

The explicit log-area metadata must remain in both arms. Proposal geometry,
candidate validity threshold, model, losses, seed, epochs, TTA, WTA output and
evaluation gate remain identical. No size-balanced loss is admissible because
BTXRD lesion-size identity is derived from segmentation GT and cannot be used
for image-label-only training.

## Why this is technically plausible

Shen et al. introduce Mask-of-Interest pooling to extract proposal-aligned
features for MIL over arbitrary-shaped masks:

- Shen et al., *Toward Joint Thing-and-Stuff Mining for Weakly Supervised
  Panoptic Segmentation*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html

The transferable principle is proposal-aligned feature pooling. Their reported
natural-image metrics do not establish the correct BTXRD denominator.

Ilse et al. formulate attention MIL as a permutation-invariant weighted
average over transformed instances:

- Ilse, Tomczak and Welling, *Attention-based Deep Multiple Instance
  Learning*, ICML 2018:
  https://proceedings.mlr.press/v80/ilse18a.html

This supports separating normalized instance representation from learned bag
importance. It does not prove a localization gain for BTXRD.

Ren et al. classify proposals directly and construct surrounding contrastive
features from inner and outer regions:

- Ren et al., *Proposal-Based Multiple Instance Learning for Weakly-Supervised
  Temporal Action Localization*, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html

The relevant transfer is that proposal appearance and surrounding contrast
should be represented explicitly rather than being implicitly rescaled by
proposal extent. Temporal-action results are not converted into image Dice.

Mun et al. show that conventional WSSS performance can conceal pronounced
small-object failure and advocate size-specific auditing:

- Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
  Segmentation*, WACV 2024:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html

Their size-balanced loss uses information unavailable under the BTXRD
image-label-only training contract; only the subgroup audit principle is
transferred.

## Decision

Run geometry-correction v3 first. Add a GT-blind grid-mass audit to its wrapper
without changing predictions. Only if sub-token candidates are materially
present and v3 still fails small may a separately frozen true-weighted-mean
ablation be launched. It must not be bundled with relational MIL, new
proposals or a consumer, so its causal effect remains identifiable.

## Prepared GT-blind audit

Commit `ec4a773b10d9a51b8cb56977fc9e560709ae8a30` adds:

- `project/audit_mask_bag_fractional_grid_mass.py`, canonical-LF SHA-256
  `25a694fb2a38fa0cf8c7e4601493a1d55d4dd5d40c1a1f1d820968e9a55dfa44`;
- `tests/test_mask_bag_fractional_grid_mass_audit.py`, canonical-LF SHA-256
  `0c8eb0b8cdc7e00705c21b6c0bc26f882619f357ee304c8d0da7e0f576a1e047`.

The tool verifies the split, every source-image hash, candidate manifest,
summary, pseudo-manifest binding and every physical NPZ hash. It reproduces
fallback bags, projects proposals using the v3 content-box transform, and
writes every candidate's grid mass plus a hash-bound summary. Fixed summaries
cover overall, image label, prompt mode, proposal source and fallback strata.
It accepts only train/validation, fixes grid/oversampling/minimum mass/candidate
cap to `32/4/0.25/81`, and records
`ground_truth_loaded=false`, `consumer_trained=false`, and
`test_evaluated=false`.

Local `py_compile` passed and the static GT-boundary suite reported `7 passed`.
The numerical Torch path remains a required Kaggle preflight because local
Python has no Torch.
