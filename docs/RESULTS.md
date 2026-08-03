# Results

## Primary validation result

| Cohort | n | Dice | IoU | Precision | Recall | Median predicted/GT area |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 184 | **0.288729** | **0.216839** | 0.336581 | 0.489305 | 2.0448x |
| `<1%` | 94 | 0.157723 | 0.118576 | 0.147795 | 0.470738 | 14.6026x |
| `1-<5%` | 72 | 0.435229 | 0.323558 | 0.488862 | 0.545975 | 1.0981x |
| `>=5%` | 18 | 0.386874 | 0.303117 | 0.713342 | 0.359586 | 0.3824x |

## Fully-supervised validation comparison

| Method | Overall Dice | Overall IoU | `<1%` Dice | `1-<5%` Dice | `>=5%` Dice |
|---|---:|---:|---:|---:|---:|
| WSSS Rich Gallery G1 + fixed rank fusion | 0.288729 | 0.216839 | 0.157723 | 0.435229 | 0.386874 |
| Fully supervised ResNet18-U-Net, threshold 0.20 | 0.492765 | 0.400359 | 0.345608 | 0.629674 | 0.713613 |

Both rows are validation results. The fully-supervised row is a comparison
upper bound and must not be presented as WSSS performance.

These are development results on 184 tumor validation images. There are 49
complete misses. The final test result is not yet available and must not be
filled from exploratory outputs.

## Why all three candidate sources remain

The same fixed fusion rule was used for every source-subset comparison. No
two-source subset preserved the full validation result:

| Available sources | Overall Dice |
|---|---:|
| LayerCAM + classifier-448 + external saliency | **0.288729** |
| LayerCAM + classifier-448 | 0.283344 |
| Classifier-448 + external saliency | 0.282440 |
| LayerCAM + external saliency | 0.280697 |

The gallery complexity is therefore supported by an ablation: each source
rescues a different subset of validation images.

## Remaining bottleneck

Candidate supply is not the main validation ceiling: the all-source per-image
oracle Dice is 0.527902. The dominant gap is selecting the correct candidate
and its extent. Small lesions are strongly over-segmented (median area ratio
14.6x), medium lesions are mainly identity/localization limited, and large
lesions are under-segmented (median area ratio 0.382x). A single global area
correction was therefore not included.

## Reporting status

The 0.288729 result is the final validation-selected configuration. A separate
test row may be added only after the locked one-time evaluation described in
`TEST_PROTOCOL.md`. Validation and test numbers must remain visibly separate.
