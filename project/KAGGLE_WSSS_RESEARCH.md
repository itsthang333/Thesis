# BTXRD WSSS — final Kaggle pipeline

The single supported research pipeline is:

1. DenseNet121 10-class `tumor_type` classifier trained with image-level labels.
2. Class-vs-normal LayerCAM with scales 256/320 plus horizontal-flip TTA.
3. SAM ViT-B prompt ensemble.
4. `simple_hybrid` candidate selector.
5. U-Net trained only from complete train/val pseudo masks, with a 3-pixel
   boundary ignore band and confidence-masked weak/strong consistency.

No LabelMe polygon, decoded segmentation mask or GT-derived statistic enters
steps 1–5. `--cam-target-class image_label` means the allowed image-level
`tumor_type`, not a segmentation target. Polygon GT is opened only after masks
or checkpoints are frozen for final comparison. The fully supervised branch is
separate and disabled by default in the notebook.

Use `D:\thesis\btxrd_kaggle_vi_debug.ipynb`. It intentionally keeps the
detailed debug structure: CAM traces, prompt components, SAM candidates,
selector diagnostics, shard counts, checkpoint metadata and final audits.

## GPU behavior

- Classifier and U-Net: pass `--num-gpus 1` or `--num-gpus 2`. Checkpoints are
  always saved without a `module.` prefix, so they are portable across 1/2 GPU
  and CPU evaluation.
- Pseudo masks: `tools/generate_pseudo_multigpu.py` launches one independent
  classifier+SAM worker per GPU, deterministically shards the split, then merges
  masks and candidate caches. This is more appropriate for per-image SAM
  inference than replicating SAM with `DataParallel`.
- In the notebook, `BTXRD_NUM_GPUS=0` auto-selects up to two visible GPUs.

## Convergence control

Both training stages use `ReduceLROnPlateau` before early stopping. The locked
notebook requires an improvement of at least `0.002`; the classifier monitors
image-level validation macro-F1, while WSSS U-Net monitors pseudo-validation
Dice. Neither convergence signal reads polygon GT. LR is reduced after two flat
epochs, and training stops after five flat classifier epochs or six flat WSSS
epochs. Every epoch prints LR and convergence state for debugging.

## Locked local smoke evidence

Five tumor images were generated without segmentation GT and evaluated only
after masks were frozen:

| Configuration | Mean Dice |
|---|---:|
| Existing `simple_hybrid` baseline | 0.4815 |
| Multi-scale + flip | 0.4962 |
| Affinity only | 0.4795 |
| Multi-view + affinity | 0.4963 |
| Symmetric contrast multi-view | 0.4502 |

The five-image sample is too small for a performance claim, but it is enough to
reject affinity and symmetric fusion from the final default. The full validation
audit and independent Dice target remain `>= 0.4` on Kaggle.
