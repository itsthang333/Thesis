# Qualitative figure contract

The locked evaluator deterministically selects best, median, worst, complete
miss, and normal false-positive cases after inference. Overlays use:

- green: prediction only;
- red: ground truth only;
- yellow: overlap.

The final test overlays and `case_manifest.csv` will be stored here after the
single permitted Kaggle evaluation. No validation overlays are fabricated
because the retained validation artifact did not include probability maps.
