# Qualitative figure contract

The locked evaluator deterministically selects best, median, worst, complete
miss, and normal false-positive cases after inference. Overlays use:

- green: prediction only;
- red: ground truth only;
- yellow: overlap.

The 12 final-test overlays and `case_manifest.csv` are committed under
`artifacts/official_wsss/test/qualitative/`. They were selected
deterministically by the locked evaluator. No validation overlays are
fabricated because the retained validation artifact did not include
probability maps.
