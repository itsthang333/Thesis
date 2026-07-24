# Final results

## Official WSSS pipeline

Status on 2026-07-24: validation selection frozen and the one permitted final
test evaluation completed on Kaggle.

The official pipeline is binary image-level supervision → LayerCAM with
horizontal-flip TTA → SAM pseudo masks → ResNet18 U-Net. It trained on 2,981
checksum-bound pseudo masks and was selected only on the clean validation
partition.

| Stage / metric | Validation result |
|---|---:|
| Binary classifier F1 (threshold 0.5) | 0.783333 |
| Binary classifier AUROC | 0.865453 |
| Pseudo-mask mean tumor Dice | 0.234339 |
| Final WSSS segmenter mean tumor Dice | 0.230020 |
| Final WSSS segmenter mean tumor IoU | 0.165471 |
| Final WSSS normal empty-mask specificity | 0.513369 |
| WSSS Dice group-bootstrap 95% CI | [0.185729, 0.275031] |
| Frozen segmentation threshold | 0.85 |

The clean validation cohort contains 371 images: 184 tumor and 187 normal.
Subgroup Dice is 0.071001 for lesions smaller than 1% of the resized image,
0.413403 for 1–5%, and 0.326917 for at least 5%. This small-lesion weakness is
the principal limitation. The epoch-wise evidence and rendered curve are in
`artifacts/official_wsss/segmenter/training/`.

## Locked final test

The schema-v4 config, checkpoint, split, image size, and threshold were frozen
before test access. Kaggle kernel
`itsthang333/btxrd-official-wsss-test-run-8796411`, version 3, then executed
the scientific evaluator exactly once. Versions 1 and 2 stopped in preflight
with zero evaluator invocations. No threshold sweep or post-test model
selection occurred.

| Metric | Test result |
|---|---:|
| Images (tumor / normal) | 373 (187 / 186) |
| Mean tumor Dice | 0.203289 |
| Mean tumor Dice group-bootstrap 95% CI | [0.162691, 0.245949] |
| Mean tumor IoU | 0.145002 |
| Mean tumor precision | 0.216612 |
| Mean tumor recall | 0.447688 |
| Normal empty-mask specificity | 0.478495 |
| Pixel specificity | 0.982539 |
| Complete misses on tumor images | 27 / 187 |
| Lesion recall, one-to-one IoU ≥ 0.10 | 0.373333 |
| Lesion precision, one-to-one IoU ≥ 0.10 | 0.162791 |

Test Dice is 0.026731 below validation Dice (an 11.6% relative decrease).
Performance remains strongly lesion-size dependent: test Dice is 0.097937 for
lesions smaller than 1% of the resized image, 0.333540 for 1–5%, and 0.405339
for at least 5%. The bootstrap resamples 244 heuristic case groups because
verified patient identifiers are unavailable; it must not be described as a
patient-level interval. Boundary distances are reported only for the 160 tumor
images with non-empty ground truth and prediction, in pixels on the resized
448×448 grid; 27 complete misses are excluded from those conditional distance
means and counted separately.

The complete evidence is in `artifacts/official_wsss/test/`: summary and
per-image metrics, group-bootstrap intervals, subgroups, pixel confusion,
373 prediction masks, 12 deterministic qualitative overlays, environment,
exact command, kernel logs, and a hash-bound final audit.

## Fully supervised diagnostic

The `fs_resnet18_pw10_full_448_e20` snapshot reaches validation mean tumor Dice
0.495132 at threshold 0.2, but it trained directly on ground-truth polygon
masks. It is an upper-bound diagnostic, not a WSSS result, and must not be
reported as the thesis pipeline. Its observed gap from the official WSSS
segmenter is 0.265112 Dice points on validation; this is a descriptive pipeline
gap, not a causal estimate of weak supervision.

The fully supervised validation result must not be compared with the WSSS test
result as if both came from the same partition. It remains an upper-bound
validation diagnostic only.
