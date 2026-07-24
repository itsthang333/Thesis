# Prediction-mask contract

Validation per-image metrics are committed, but the original validation run did
not export all 371 probability-derived PNG masks. This absence is explicit; it
must not be reconstructed from ground truth.

The completed locked test evaluator wrote exactly one binary PNG per test image
at threshold 0.85. All 373 masks are committed under
`artifacts/official_wsss/test/prediction_masks/`; their byte counts and SHA-256
hashes are bound by `artifacts/official_wsss/test/FINAL_TEST_AUDIT.json`.
