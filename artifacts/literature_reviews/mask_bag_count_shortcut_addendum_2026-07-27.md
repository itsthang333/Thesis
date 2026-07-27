# Mask-bag candidate-count shortcut addendum

Date: 2026-07-27

This is a no-GT diagnostic addendum to
`mask_bag_selector_contingency_2026-07-27.md`. It does not modify the running
version-6 protocol, model, gate, operational targets or test lock.

## Evidence

Normal images have more tumor-targeted proposals than tumor images:

- train candidate count mean with p10/median/p90:
  normal `71.745 [54,78,81]`, tumor `45.399 [27,45,72]`;
- validation:
  normal `63.674 [36,69,81]`, tumor `49.239 [27,45,81]`.

Using only the negative candidate count as a score gives direction-corrected
image-label AUROC `0.86726993` on train and `0.71207277` on validation. Box,
positive-point and negative-point counts give nearly identical AUROC because
they are coupled by the prompt-ensemble construction. Diagnostic payload byte
size alone gives `0.75822026/0.64566380`.

This uses only the prediction-first candidate manifests and permitted binary
image labels. It reads no segmentation mask, subgroup, lesion area or test
data. The proposal generator explicitly targets the tumor logit for both
positive and normal bags; normal bags are reliable negative MIL instances.

## Interpretation

This is not segmentation-label leakage, but it is a localization shortcut. A
selector may obtain useful image AUROC from the number or distribution of
proposals without learning which proposal overlaps the lesion. Normalized
SmoothMax removes the exact log-count offset for identical logits, but cannot
make different descriptor distributions identical.

Therefore image AUROC remains only one gate component. Candidate Dice,
subgroup preservation, paired confidence intervals and complete misses remain
decisive. A high image AUROC cannot authorize a pseudo-mask consumer.

## Requirement for any post-version-6 ranking study

If version 6 has adequate candidate oracle support but poor selected Dice, a
new protocol should:

1. report the frozen count-only AUROC values above;
2. report train-only and validation-image-label association between learned
   bag probability and candidate count;
3. choose architecture and stopping only from clean-train image-label
   cross-validation;
4. prefer a predeclared count-robust mechanism, such as train-time candidate
   dropout or instance-level normal-prototype contrast, when the learned bag
   score merely reproduces the construction shortcut;
5. retain the unchanged prediction-first segmentation gate.

No count threshold, candidate cap or sampling rate may be chosen from
validation segmentation GT.
