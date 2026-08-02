# S7 global-local constrained instance self-training

## Scientific question

Can a direct candidate classifier trained from all image-level-negative
instances and globally/local-constrained soft labels on image-level-positive
bags reduce Geometry-v3 selector regret without another lazy bag-level MIL
objective?

This is a same-gallery selector experiment. It does not generate proposals,
change mask geometry, use validation subtype labels, or use spatial ground
truth for training.

## Failure mechanism addressed

S6 learned pathology/subtype labels at bag level but subtype identification was
only 46.74%, the auxiliary arm paid a binary/consistency/drift cost, and the
entropy/confidence observables did not predict whether a changed selection
would improve Dice. The experiment therefore failed because bag-level taxonomy
did not identify positive candidates reliably.

Earlier mechanisms also rule out a simple pseudo-winner implementation:

- mask-bag v6 used true-negative bags plus a detached latent winner and reached
  Dice 0.21789918;
- T1 used OOF top-1 self-paced supervised contrast and reached 0.24282104;
- S4 clustering and S6 hierarchy did not repair candidate identity.

S7 differs by assigning soft labels to every candidate in every positive bag,
with both a global positive-mass constraint and the MIL local constraint that
each positive bag contains at least one positive instance. No bag-classification
loss or attention/SmoothMax loss trains the residual.

## Source and bounded transfer

The primary mechanism is adapted from Ma et al., *Rethinking Multiple Instance
Learning: Developing an Instance-Level Classifier via Weakly-Supervised
Self-Training* (arXiv:2408.04813):

https://arxiv.org/html/2408.04813

The paper uses entropy-regularized global pseudo-label projection, a local
one-positive-per-bag constraint, and iterative direct instance-classifier
training. It reports that soft labels, constraints, and an adaptive positive
mass are important on CAMELYON16. S7 transfers only that mechanism. It does not
transfer external metrics, instance annotations, a tuned BTXRD hyperparameter,
or the paper's image encoder.

INS supplies supporting rationale that true-negative bags can stabilize direct
instance learning, but its prototype/self-paced contrastive implementation is
not copied because it overlaps T1:

https://arxiv.org/abs/2307.02249

## Frozen candidate method before real input

- Input: exact accepted selector cache, Geometry-v3 scorer, gallery, ordering,
  family identifiers, original/flip descriptors and packed masks.
- Training labels: binary train image label only.
- Residual: descriptor dimension 1156, hidden dimension 128, GELU, dropout 0.1,
  scalar output initialized exactly to zero.
- Candidate input: valid-candidate-centered descriptors. The scalar output is
  not centered, because true-negative instances must be able to shift below
  zero; a bagwise constant cannot affect the final argmax.
- Initial instance logits: exact frozen Geometry-v3 candidate logits.
- Pseudo-label schedule: weighted global positive mass linearly decreases from
  0.50 to 0.15 over epochs 0-20, then stays at 0.15 through epoch 40. The final
  value 0.15 is fixed from the external CAMELYON16 result, not selected on
  BTXRD validation.
- Global projection: a single additive logit bias is solved so the weighted
  sigmoid mean equals the scheduled mass. Candidate weights are equal image,
  then equal family, then equal candidate within family.
- Local constraint: after global projection, the highest current instance
  logit in every positive bag is set to target one. The global mass before and
  realized mass after this constraint are recorded.
- Negative bags: every candidate target is exactly zero with the same
  image/family/candidate weighting.
- Loss: soft instance BCE on original and flip logits, original/flip Smooth-L1
  consistency weight 0.10, residual drift weight 0.001.
- Optimizer plan: AdamW, learning rate 3e-4, weight decay 1e-4, batch 16,
  exactly 40 epochs, seed 42, no early stopping or sweep.
- Inference: average original/flip base-plus-residual candidate logits and take
  one argmax. Use the exact accepted Geometry-v3 bag probability and selected
  immutable mask. This isolates candidate ranking from image classification and
  candidate-count probability shortcuts.

## Matched output and decision

The future runner must freeze an exact Geometry-v3 identity control and the S7
primary prediction, all candidate scores, pseudo-label histories, checkpoint,
and physical maps before validation polygons are available. An independent
GT-blind auditor must reproduce zero-initialized identity, every projection and
local constraint, the complete 40-epoch source/input contract, candidate
scores, selected indices, frozen baseline probabilities and maps.

Only after that audit is committed may the frozen evaluator read validation GT.
The mechanism passes only if S7 improves overall and small mean Dice over the
identity control, does not reduce medium or large, and does not increase
complete misses. Consumer authorization additionally requires all four thesis
goals, a positive complete-group overall CI95 lower bound, no subgroup
regression, and no miss increase.

No post-hoc mass, epoch, seed, fusion, threshold, subgroup or per-image router
is allowed after the result. BTXRD test remains locked.
