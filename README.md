# BTXRD Rich Gallery G1

This repository contains only the source code required to train, run, freeze,
and evaluate the thesis method:

```text
X-ray + binary image label
  -> LayerCAM localization at two input scales
  -> frozen BiomedCLIP saliency
  -> SAM ViT-B proposal generation
  -> merged and deduplicated rich proposal gallery
  -> RAD-DINO candidate descriptors
  -> image-label-only G1 multiple-instance learning
  -> equal percentile-rank fusion of G1 and upstream evidence
  -> one frozen output mask per image
```

Spatial annotations are not inputs to classifier training, saliency
generation, proposal generation, G1 training, scoring, or mask selection.
They are opened only by the separate evaluator after predictions have been
frozen.

No dataset copy, checkpoint, experiment output, performance table, research
archive, or qualitative test case is stored in this repository.

## Documentation

- [Source structure](docs/SOURCE_STRUCTURE.md)
- [Installation, training, inference, and evaluation](docs/USAGE.md)
- [Interactive one-image inference demo](notebooks/BTXRD_Rich_Gallery_G1_Demo.ipynb)

## Quick verification

```bash
python -m compileall -q project
python -m pytest -q tests
```
