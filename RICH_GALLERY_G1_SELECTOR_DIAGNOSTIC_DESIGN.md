# Rich-gallery G1 selector diagnostic

## Question

Why did the three-source gallery raise validation tumor oracle Dice from
`0.409076` to `0.528298`, while the retrained Geometry-v3 selector fell from
`0.245482` to `0.206026`?

There is no collaborator checkpoint. The comparison is therefore between the
same Geometry-v3 algorithm trained on two different proposal distributions,
not an exact checkpoint transfer.

## Frozen evidence

- Same-gallery Geometry-v3: Dice `0.245482`, oracle `0.409076`, 53 complete
  misses.
- Rich-gallery G1: Dice `0.206026`, oracle `0.528298`, 29 complete misses.
- Rich-gallery selected-to-oracle regret: `0.322272`.
- Median selected/GT area ratio: about `13.02`; selected Dice versus absolute
  log-area error Spearman: about `-0.705`.
- Rich gallery: 55,814 retained validation candidates, mean 150.44/image,
  maximum 243/image. Candidate count versus per-image regret Spearman is only
  about `0.059`.

These observations reject proposal scarcity and raw bag length as sufficient
explanations. They do not distinguish cross-source calibration from incorrect
extent ranking within a source.

## Stage A: annotation-free score freeze

Use the exact audited G1 checkpoint and exact merged validation gallery. Run
the frozen RAD-DINO projection and Geometry-v3 scorer once. Before importing
or opening validation polygons, freeze for all 371 validation images:

- every retained candidate index;
- original, horizontally flipped and averaged candidate logits;
- original and flipped 1,156-D descriptors;
- descriptor metadata, shape features, SAM score, upstream selection score,
  classifier-causal score, component, prompt mode and proposal source;
- image bag logit/probability and the exact selected index.

The freeze is valid only if it reproduces all 371 G1 selected indices, logits,
bag logits, probabilities and output-map hashes within `5e-6`, and binds the
checkpoint, split, gallery and model-snapshot hashes. No spatial annotation or
test image may be read.

## Stage B: post-freeze ranking diagnosis

After Stage A is independently hash-verified, open exactly the 184 validation
tumor polygons and compute candidate Dice. For image `i`, let `s_i` be the
frozen selected candidate, `o_i` the global oracle and `S(s_i)` the selected
candidate's proposal source. Decompose regret exactly as

`D(o_i)-D(s_i) = [D(o_i)-max_{j in S(s_i)}D(j)] + [max_{j in S(s_i)}D(j)-D(s_i)]`.

The first term is source-choice regret; the second is within-source ranking and
extent regret. Report both overall and for the fixed `94/72/18` lesion-size
subgroups.

Also report:

- oracle rank and oracle reach/best Dice at top `1/3/5/10/20/50`;
- score-Dice correlation per image and pooled within each source;
- selected/oracle source matrix and source-specific score correlations with
  area, prompt mass, SAM, upstream selection and classifier-causal scores;
- selected/GT area ratio and Dice/log-area-error relationship;
- original/flip score stability;
- the smooth-MIL weight of the selected and oracle candidates and effective
  candidate count, where
  `w_j = softmax(z_j / 0.2)` and `N_eff = 1 / sum_j w_j^2`;
- the fraction of positive bags whose hard self-guided target has Dice below
  `0.1`.

## Predeclared interpretation

- Source-choice regret at least half of total regret supports source-conditional
  calibration/family-balanced pooling.
- Within-source regret at least half supports boundary/extent discrimination
  and minimal-sufficiency ranking; a source-only correction is insufficient.
- High top-k oracle reach with poor top-1 supports a bounded shortlist plus a
  second, independently trained ranking signal.
- Low oracle MIL weight, very small effective candidate count, or hard-positive
  low-quality rate above 0.5 supports confirmation bias from smooth-LSE plus
  epoch-3 argmax self-training.
- Strong score association with prompt mass or source but weak score-Dice
  association supports descriptor/objective shortcut rather than proposal
  failure.

No improved selector is authorized from intuition alone. The next objective
must be chosen from this decomposition, trained with image labels only, and
evaluated on actual binary-mask Dice. Test remains locked.
