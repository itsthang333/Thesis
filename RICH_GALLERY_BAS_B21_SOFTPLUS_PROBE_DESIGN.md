# BAS-B2.1 binary-stability probe

Status: frozen before execution; image-label-only mechanics diagnostic; no
validation polygons and no test access.

## Question

Did BAS-B2 fail because the intended class-aware localization mechanism is
inapplicable to BTXRD, or because the official terminal ReLU class-map head
entered a dead state after being reduced from 200/1000 classes to two?

B2 produced exact tied logits and constant label maps. Its candidate score was
therefore an area rank, so its Dice result cannot answer the intended BAS
mechanism question. B2.1 changes only the terminal nonnegative activation from
ReLU to Softplus:

`ReLU(z)=max(0,z)` becomes `Softplus(z)=log(1+exp(z))`.

Both preserve nonnegative class maps required by BAS, but for finite `z`,

`d Softplus(z)/dz = sigmoid(z) > 0`,

whereas `d ReLU(z)/dz=0` for the all-negative state observed in B2.

## Immutable matched controls

Unchanged from B2:

- canonical 2,981 train and 371 validation images;
- binary normal/tumor image labels only;
- ImageNet ResNet-50 initialization;
- 448-pixel input and horizontal flip policy;
- output strides, localization sigmoid, BAS/foreground/full-image objectives;
- BAS area weight 1.2 and foreground CE weight 0.5;
- SGD Nesterov, LR multipliers, momentum, weight decay, batch 32, T4x2;
- seed 42;
- immutable rich gallery, G1 scores and upstream scores.

Changed: terminal class-map activation only.

## Bounded execution

Train exactly five canonical passes and freeze:

- training loss components and accuracy;
- the checkpoint;
- 371 continuous tumor activation maps;
- validation image-label AUROC/F1;
- activation range/nondegeneracy;
- within-image Spearman correlation of BAS candidate score with candidate
  area for all 184 tumor bags.

No candidate choice or spatial metric is optimized in this probe. Validation
polygons remain unopened. This avoids another 100-epoch run when the exact
technical pathology can be rejected after five passes.

## Predeclared mechanics continuation gate

All conditions must hold:

| Quantity | Gate | Why |
|---|---:|---|
| final full-image CE | `<=0.69` | departs from tied-logit `log(2)` |
| distance from exact class-prior accuracy | `>=0.01` | rejects constant argmax |
| validation image AUROC | `>=0.55` | rejects constant score without demanding final efficacy |
| mean activation range | `>=1e-3` | rejects spatially constant maps |
| tumor nondegenerate-map fraction | `>=0.50` | rejects isolated numerical variation |
| mean BAS-area Spearman | `<=0.98` | rejects the proven disguised area rank |

These gates authorize only a full scientific run. They are not substitutes for
Dice and cannot be reported as a segmentation improvement.

If every gate passes, run the same Softplus configuration to its frozen final
epoch, freeze all five selector choices, and evaluate actual binary-mask Dice
against `0.2887294867` overall and `0.1577232964/0.4352293348/0.3868735327`
by subgroup. If any gate fails, retire the binary BAS transfer without a seed,
threshold, loss-weight, resolution, epoch or fusion sweep.

## Safety and provenance

- training spatial GT: forbidden;
- validation polygons during mechanics probe: forbidden;
- test images/labels: forbidden;
- private/offline Kaggle only;
- every output binds source, protocol, split, checkpoint and input hashes.
