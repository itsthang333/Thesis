# Final results

## Official WSSS pipeline

Status on 2026-07-24: validation frozen; test not yet evaluated.

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

## Fully supervised diagnostic

The `fs_resnet18_pw10_full_448_e20` snapshot reaches validation mean tumor Dice
0.495132 at threshold 0.2, but it trained directly on ground-truth polygon
masks. It is an upper-bound diagnostic, not a WSSS result, and must not be
reported as the thesis pipeline. Its observed gap from the official WSSS
segmenter is 0.265112 Dice points on validation; this is a descriptive pipeline
gap, not a causal estimate of weak supervision.

## Test claim

No test metric is reported yet. The one final test evaluation is permitted only
after `configs/official_wsss_frozen_test.json` is generated and committed. It
must run on Kaggle with checkpoint SHA-256
`02d3af8feede3c3e650cb76d664185c59092697c1c8306ea67613b89f8407fb4`,
image size 448, and threshold 0.85, without a threshold grid.
