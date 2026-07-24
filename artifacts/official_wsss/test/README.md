# Locked final test output

Status: complete on Kaggle kernel version 3. The scientific test evaluator ran
exactly once, at image size 448 and the validation-selected threshold 0.85.
There was no threshold sweep, post-test selection, or rerun.

The official WSSS result on 373 test images (187 tumor, 186 normal) is:

- mean tumor Dice 0.203289, group-bootstrap 95% CI [0.162691, 0.245949];
- mean tumor IoU 0.145002;
- normal empty-prediction rate 0.478495;
- pixel specificity 0.982539;
- 27 complete misses among tumor images.

`evaluation/` contains the summary, 373-row per-image table, subgroup table,
10,000-iteration group bootstrap, pixel confusion, and run manifest.
`prediction_masks/` contains exactly 373 masks. `qualitative/` contains 12
deterministically selected best/median/worst/complete-miss overlays and their
case manifest.

`FINAL_TEST_AUDIT.json` binds every Kaggle-produced output by byte count and
SHA-256. `TEST_EVALUATION_LEDGER.json` records the two preflight-only failures
and the single successful scientific evaluation. The frozen pre-test record
remains unchanged at `configs/official_wsss_frozen_test.json`; its
`test_evaluated=false` field describes the state at freeze time.
