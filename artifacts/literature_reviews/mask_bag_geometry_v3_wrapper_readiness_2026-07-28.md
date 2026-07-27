# Mask-bag geometry-correction v3 wrapper readiness

Date: 2026-07-28  
Status: execution design only; wrapper is intentionally not finalized or
launched before the running v6 terminal audit

## Audited starting point

The current version-6 wrapper is:

- `tmp/kaggle/rad_dino_mask_bag_mil_probe_val_v1/run_rad_dino_mask_bag_mil_probe_val_v1.py`;
- canonical-LF SHA-256
  `6293b040ce25109e6c4bb167a32fcc635bfbeef38e82a23c18c28ba9d8aaf1ff`;
- Kaggle kernel `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1`,
  version 6.

It already performs the correct high-level sequence:

1. clone and hash-audit source/protocol/split/checkpoints;
2. prove two real T4 devices;
3. run focused and whole-repository tests;
4. generate complete train and validation candidate payloads;
5. validate every physical payload before training;
6. train/freeze predictions without segmentation GT;
7. invoke a separate evaluator after prediction freeze;
8. compact evidence and delete temporary payloads only at the end.

Its candidate generator includes the accumulated device-routing fixes:
`project/generate_pseudo_masks.py`, canonical-LF SHA-256
`028ca4b8c0f1445178043bf9726c3ef4092df1ee39669938ed7f475c1bfa0ba7`.
The unchanged evaluator is
`project/evaluate_rad_dino_mask_bag_mil_probe.py`, canonical-LF SHA-256
`090a7e8e8e34704a7002af0f90c50498437520c0a28efe2b62f095eb8bc44f8b`.

## Required changes for a v3 correction wrapper

The new wrapper should be derived mechanically from the audited v6 wrapper,
but it may not be finalized until terminal v6 candidate hashes are known.

### Provenance

- Bind final geometry protocol
  `rad_dino_mask_bag_mil_descriptor_geometry_correction_v3`, SHA-256
  `3248b37b8f60da2b3c7b6e4009f76967876bef6d498ac781e29f779a57351a91`.
- Bind source commit
  `cb2c269394959ce04377a48b51971f1304f96c05` for the corrected runner/model.
- Bind:
  - corrected runner
    `1bf56d0d9bc238aafd37988bcd767352065f89a160744b3a837d68127c7e9a71`;
  - corrected model
    `44cd6ff052a38f9e87c1d93ce71d2aad2105c4b11083354d06b835409c517407`;
  - model test
    `b563dd7c91aa47c939dce1c95db23344548799ba85af1dd53be13bf30643df0c`;
  - runner test
    `7aab3c2073dbb170c08aaeef931442f97e930c2f67e509b3a976e91fe74a0001`.
- Retain the parent scientific protocol and all five v6 wrapper-correction
  artifacts as historical provenance. Do not rewrite them.

### Candidate invariance

Candidate generation must remain byte-identical to v6. The corrected wrapper
must require the regenerated train/validation candidate manifests and
pseudo-manifests to equal the direct terminal v6 hashes. Those four hashes are
intentionally **not guessed here**; they are the unresolved fields that block
final wrapper freeze.

Manifests alone cannot be mounted as candidate evidence. Either:

- regenerate every physical NPZ with the unchanged generator and prove the
  four terminal hashes; or
- mount an immutable v6 payload dataset and verify every NPZ hash against its
  manifest.

The current compact v6 design copies only manifests/summaries and deletes
temporary candidate payloads after success, so regeneration is the expected
path unless the terminal output proves otherwise.

### GT-blind fractional-mass audit

Before optimizer construction, run:

- `project/audit_mask_bag_fractional_grid_mass.py`, canonical-LF SHA-256
  `aa684de20407d0934bb8c4d32f5293eac1ed56e341e28eab0ecce78fd2757c79`;
- static contract test
  `15cd00183d6e6ef38228863a513e17b3c67a4cd84dbfc48c3e4b7312a73c77e6`.

Run it separately for train and validation against the same physical
candidate roots used by the probe. Copy both CSV/summary pairs to compact
output and hash-bind them. This diagnostic cannot change v3 training or
predictions.

### T4x2 and tests

- Preserve candidate routing: classifier/LayerCAM on `cuda:0`, SAM on
  `cuda:1`.
- Preserve RAD-DINO `DataParallel` over both T4s.
- Add the geometry and fractional-mass tests to the focused suite.
- Require the whole suite with exactly the already documented checkpoint skip;
  any new skip or failure stops before candidate generation.
- Preserve a real nontrivial convolution on both physical devices.

### Prediction and comparison

- Output directory/kernel identity must be new; do not overwrite v6.
- Freeze checkpoint/history/371 maps/prediction manifest before importing the
  segmentation dataset.
- Use the unchanged evaluator and parent gate.
- Add a post-freeze paired comparison:
  1. v3 corrected minus v6;
  2. v3 corrected minus promoted flip-TTA baseline;
  3. v6 minus promoted baseline.
- Report overall/small/medium/large means, complete misses and 10,000
  complete-group paired bootstrap intervals.
- Consumer and BTXRD test remain locked.

Commit `5e548fc61cdf50d7b7774e6001849b247ba0eee6` prepares the
post-freeze v3-minus-v6 comparator:

- `project/compare_mask_bag_evaluated_arms.py`, canonical-LF SHA-256
  `24c625cfc50740d9cb633906d60ae81089e3960d3eec4b3ead6f3ce89ebaffad`;
- `tests/test_compare_mask_bag_evaluated_arms.py`, canonical-LF SHA-256
  `e761c249da8b36445b28fb73b7578f9d5c7e2b728d1edb3e2730c2b393373661`.

It accepts only two already evaluated and hash-bound `per_image.csv` files; it
does not import a dataset, open an image or reopen a segmentation mask. It
requires identical image/group/subgroup/GT-area/oracle fields, the frozen
184/94/72/18 cohort, 10,000 group-bootstrap replicates and seed family
20261101. Outputs include paired Dice CI, complete misses, recovered misses
and lost overlaps. The unchanged evaluator remains responsible for each arm
versus the promoted baseline.

## Finalization gate

The wrapper is ready to finalize only after the separate monitor delivers a
terminal v6 artifact and the main task audits:

- wrapper/source/protocol/split/checkpoint hashes;
- two real T4s and exact focused/whole-test counts;
- complete 2981/371 candidate evidence;
- exact four candidate/pseudo-manifest hashes;
- checkpoint/history/371 maps/prediction-freeze hashes;
- cohort 371/184/187 and subgroup 94/72/18;
- complete misses, paired bootstrap, `consumer_trained=false`,
  `test_evaluated=false`.

If v6 already meets every operational goal, correction v3 is not launched.
If its candidate oracle fails any goal, correction v3 is insufficient and is
not launched. Otherwise, fill only the terminal evidence hashes, freeze a new
wrapper-correction record, run complete preflight, and launch one T4x2 job with
one separate ten-minute monitor.
