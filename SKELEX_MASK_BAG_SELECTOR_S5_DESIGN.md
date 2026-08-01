# S5: frozen SKELEX same-gallery selector representation

Status: static design; no claim, no real-data execution, no prediction and no
validation-GT access at the time this file was written.

## Evidence inherited

This successor trusts the terminal result recorded by the collaborator at
commit `f30e37b063682c4e79c50b58c53dd3fbadd64478`; it does not download or audit
that experiment's output. The retained architecture is the rich-gallery
control: immutable class-agnostic proposals followed by equal within-image
percentile ranks from a learned Geometry-v3 selector and the upstream proposal
score. Its recorded Dice is `0.2887294867`, while the gallery oracle is
`0.52829833`, leaving approximately `0.23917` selector regret.

The rejected BAS arm showed why another activation-map variant is not the next
experiment: its terminal ReLU head was constant, its proposal score was almost
an area rank, and Dice fell to `0.18110635`. Earlier normal prototypes,
RAD-DINO local affinity, proposal relations, count control, family balancing,
clustering and dense saliency arms were also negative. S5 therefore changes
the frozen representation source, not the gallery, loss family or rank fusion.

## Single scientific change

S5 replaces the frozen generic RAD-DINO proposal representation in the new
selector branch with the public musculoskeletal-radiograph SKELEX ViT-MAE
encoder. The exact checkpoint revision is
`368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`, and the exact
`model.safetensors` SHA-256 is
`81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`.
The checkpoint remains frozen and is never redistributed in repository or
experiment output. Its CC-BY-NC-ND-4.0 license is retained.

Full square-padded radiographs are resized to 224 and normalized with the
checkpoint's ImageNet mean/std. No tumor-containing crop, anatomical crop,
box, mask or validation statistic is used. The encoder mask ratio is set to
zero and deterministic monotone MAE noise preserves the native patch order.
After removing the non-spatial CLS token, patch tokens from fixed layers 8, 16 and 24 are projected from 1024 to 128
dimensions with the existing seed-42 Gaussian projection and L2-normalized.

For every image, S5 consumes the exact candidate indices frozen in selector
cache `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`.
It must reproduce every index; it cannot silently discard a small proposal on
the coarser 14x14 grid. The already accepted 128-square fractional geometry
bridge is retained before area interpolation to 14x14; this avoids a new
coarse direct-sampling loss. Area interpolation preserves fractional support and
inside/context means divide by their exact positive fractional mass (epsilon
only). This support-preservation rule is necessary to keep the gallery
immutable and is not a second proposed performance mechanism.

The lightweight MIL scorer uses the already accepted fixed recipe: 16 epochs,
batch 16, AdamW learning rate `3e-4`, weight decay `1e-4`, SmoothMax
temperature `0.20`, self-guided instance weight `0.25` after two warm-up
epochs, aligned flip consistency weight `0.10`, seed 42. Training sees only the
binary image-level tumor label.

## Frozen pair

Exactly two validation arms are physically frozen before validation GT:

1. `geometry_v3_plus_upstream_equal_rank` (identity control);
2. `geometry_v3_plus_upstream_plus_skelex_equal_rank` (primary S5).

The primary score is the unweighted mean of the three within-image percentile
ranks. There is no weight, temperature, layer, source, subgroup, resolution,
epoch, threshold or morphology sweep. GT-blind correlation and changed-winner
statistics are diagnostics only and cannot prevent pair freezing.

Because the user prohibited access to collaborator outputs, S5 is first run on
the exact same-gallery artifacts owned by the central workstream. It is a
transferable representation ablation, not a claim that the physical rich
gallery `0.2887294867` output has already been improved. Promotion to that
gallery requires a later GT-blind scorer interface and a separate claim.

## Decision and safety

The primary gate is corrected paired Dice after both arms and the independent
GT-blind audit are frozen. Promote only if overall Dice is strictly above the
identity control and no mandatory subgroup regresses under the predeclared
comparator. Record a negative result without a rescue sweep.

- Training labels: image-level tumor only.
- Validation predictions: physical pair freeze before GT.
- Consumer training: forbidden before the operational gate passes.
- BTXRD test: locked.
- Heavy compute: Kaggle T4x2 only.
- Collaborator outputs/Kaggle: not accessed.

Primary sources:

- https://www.nature.com/articles/s41746-026-02826-9
- https://arxiv.org/abs/2602.03076
- https://huggingface.co/skhoha/SKELEX/tree/368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e
