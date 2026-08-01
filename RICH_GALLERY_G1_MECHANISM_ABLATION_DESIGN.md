# G1 selector mechanism ablation

This post-freeze diagnostic uses only the Stage-A descriptors, candidate
metadata and exact G1 checkpoint. Every selected index below is frozen for all
371 validation images before spatial validation annotations are imported.

## Fixed selector variants

1. `g1`: original averaged original/flip Geometry-v3 logits.
2. `upstream_selection`: source-specific `coverage_mass_sam` score produced by
   each candidate generator before gallery merging.
3. `sam_score`: frozen SAM predicted mask quality.
4. `classifier_causal_score`: frozen candidate mask-out classifier score; this
   is expected to be uninformative when the generating profile did not enable
   the causal selector, and is retained as a negative control.
5. `anchor_prompt_mass` and `anchor_prompt_inside`: the two prompt features
   currently passed to Geometry-v3. For candidates added from the 448 source,
   these are calculated against the 320 anchor prompt map by the current merge
   contract, not their own source prompt.
6. `source_zscore_g1`: normalize G1 logits within each proposal source in each
   image, then select the largest standardized score. This tests cross-source
   score-scale imbalance without fitting a parameter.
7. Four block interventions on the 1,156-D descriptor. Replace one candidate
   block by its within-bag mean while leaving every other block and the frozen
   nonlinear scorer unchanged:
   - inside features `[0:384]`;
   - context features `[384:768]`;
   - inside-minus-context features `[768:1152]`;
   - metadata `[1152:1156]`.

Replacing a block with its within-bag mean removes only that block's ability to
rank candidates in the current image. It is an inference intervention, not a
claim that the modified descriptor lies on the training distribution.

## Interpretation

- If `upstream_selection` materially exceeds G1, the merge discarded a useful
  source-conditioned ordering signal and the improved selector should consume
  or distill it.
- If `source_zscore_g1` improves, cross-source logit calibration contributes to
  the failure. If it does not, source identity alone is insufficient.
- If removing metadata improves Dice, shared-anchor prompt features are a
  shortcut. If removing contrast/inside/context improves Dice, the implicated
  representation block is driving the wrong extent ranking.
- No variant is a final method. These are causal diagnostics on one frozen
  validation cohort. A final selector must be trained image-label-only on train
  data with its rule fixed before actual validation Dice is opened.

The evaluator reports actual Dice and complete misses for overall and fixed
94/72/18 subgroups for every frozen variant. Test remains locked.
