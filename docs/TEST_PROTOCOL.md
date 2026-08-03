# Locked one-time test protocol (A100)

## Purpose

This protocol converts the validation-selected pipeline into an untouched
final test estimate. Test is not used for model selection, threshold choice,
source selection, checkpoint selection, or debugging scientific behavior.

## Cohort and endpoint

- Canonical test images: 373 (187 tumor, 186 normal).
- Primary endpoint: mean per-image Dice over the 187 tumor images.
- Secondary endpoint: mean per-image IoU over the same images.
- The candidate cap, three sources, checkpoints, equal 0.5/0.5 rank fusion,
  and tie-break are identical to validation.
- The known binary image label is part of the declared inference setting.

## Step 0 - clean commit and environment

Use the exact `codex/thesis-final` commit on the A100 machine. Record Python,
PyTorch, CUDA, cuDNN, GPU name, and package versions. Run the unit tests before
mounting or opening the test image directory.

```bash
python project/capture_final_run_environment.py \
  --output /results/a100_environment.json
```

## Step 1 - freeze the protocol

From a clean committed tree, create the immutable lock. Include both
classifier checkpoints and every additional frozen weight as a named artifact.

```bash
python project/freeze_final_test_protocol.py \
  --split-manifest /data/split_manifest.csv \
  --classifier-checkpoint /models/classifier320.pt \
  --classifier-checkpoint /models/classifier448.pt \
  --sam-checkpoint /models/sam_vit_b.pth \
  --g1-checkpoint /models/g1.pt \
  --validation-result artifacts/final_pipeline/result_summary.json \
  --artifact environment=/results/a100_environment.json \
  --artifact biomedclip_weight=/models/biomedclip.bin \
  --artifact rad_dino_weight=/models/rad_dino.bin \
  --test-run-id thesis-final-a100-001 \
  --output /results/final_test_protocol.json
```

Do not edit the repository, lock, or bound artifacts after this command.

## Step 2 - annotation-free test prediction

Every test-aware command must receive:

```text
--split test --frozen-config /results/final_test_protocol.json
```

Run these stages in order:

1. `generate_biomedclip_saliency.py` for the test split.
2. `package_biomedclip_saliency_supply.py`.
3. `run_rich_gallery_candidate_supply.py --mode anchor --splits test`.
4. `run_rich_gallery_candidate_supply.py --mode addition --splits test`.
5. `merge_frozen_candidate_galleries.py --split test`.
6. `score_final_rich_gallery.py --split test`.
7. `freeze_final_rich_gallery.py --split test`.

Check the stage manifests and SHA-256 values after each command. Step 2 may
read test images and their binary image labels, but must not open polygons.
The final freeze must report exactly 373 choices, 187 tumor images, 186 normal
images, and `candidate_choices_frozen_before_test_gt=true`.

## Step 3 - one-time spatial evaluation

Only after Step 2 is complete, execute:

```bash
python project/evaluate_final_rich_gallery.py \
  --split test \
  --frozen-config /results/final_test_protocol.json \
  --dataset-root /data/BTXRD \
  --split-manifest /data/split_manifest.csv \
  --expected-split-sha256 <locked-split-sha256> \
  --selection-root /results/final_test_choices \
  --expected-selection-freeze-sha256 <step-2-freeze-sha256> \
  --candidate-root /results/final_test_gallery \
  --output-dir /results/final_test_evaluation
```

Do not pass `--expected-overall-dice` for test. Archive the complete output
directory immediately and report the result regardless of whether it is high
or low.

## Failure policy

- A technical failure before any test polygon is opened may be resumed with
  the same commit, lock, assets, and parameters; record the failure log.
- Once spatial test GT has been opened, no scientific configuration may be
  changed. A deterministic technical recovery must preserve the exact frozen
  choices and be disclosed.
- Test performance never authorizes tuning or a second method comparison.
