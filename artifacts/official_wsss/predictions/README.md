# Prediction-mask contract

Validation per-image metrics are committed, but the original validation run did
not export all 371 probability-derived PNG masks. This absence is explicit; it
must not be reconstructed from ground truth.

The locked test evaluator requires a fresh output directory and writes exactly
one binary PNG per test image at the frozen threshold. The downloaded Kaggle
output will be stored here after the single permitted final test run.
