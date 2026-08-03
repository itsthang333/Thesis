# Reproducibility

## External assets

- BTXRD images, metadata, and annotations.
- DenseNet121 checkpoints for the 320 and 448 candidate supplies.
- SAM v1 ViT-B checkpoint.
- Frozen BiomedCLIP and RAD-DINO snapshots.
- Frozen G1 checkpoint.

The final test lock records the path, byte size, and SHA-256 of every asset.
Paths may differ between machines only before the lock is created.

## Development reproduction

The canonical stages are:

1. Verify the canonical split manifest.
2. Generate the LayerCAM-320, classifier-448, and BiomedCLIP supplies.
3. Run `run_rich_gallery_candidate_supply.py` in `anchor` and `addition`
   modes.
4. Merge the two galleries with `merge_frozen_candidate_galleries.py`.
5. Build the G1 selector cache and train G1 using image labels only.
6. Score every candidate with `score_final_rich_gallery.py`.
7. Freeze equal-rank choices with `freeze_final_rich_gallery.py`.
8. Evaluate the frozen validation choices with
   `evaluate_final_rich_gallery.py --split val`.

The retained CLIs expose exact arguments through `--help`; hashes link every
stage. The refactored code reproduces validation Dice exactly:
`0.28872948670665205`.

## Final A100 test

The final test is not another experiment. It reuses the validation-selected
method without tuning. Follow [TEST_PROTOCOL.md](TEST_PROTOCOL.md): commit the
clean tree, create the schema-v4 lock, generate and score test candidates
without polygons, freeze all 373 choices, then invoke the spatial evaluator
once. Save the lock, stdout/stderr, manifests, hashes, and result together.

## Preflight

```bash
python -m py_compile project/*.py project/evaluation/*.py
python -m pytest -q tests
```

The preflight uses no test images or annotations.
