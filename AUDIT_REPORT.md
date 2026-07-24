# AUDIT REPORT — BTXRD bone-tumor segmentation

Status: research in progress (2026-07-24). The clean test partition remains locked and has not been used for model or threshold selection.

## 1. Protocol decision

The primary endpoint is **mean Dice across every tumor image with valid polygon ground truth**. Complete misses, empty predictions, and inference failures remain in the denominator. Normal images are excluded from the primary Dice mean but are reported separately through empty-prediction specificity / false-positive case rate. Median Dice, mean IoU, per-image rows, bootstrap confidence intervals, subgroup summaries, lesion detection, and pixel confusion are retained as secondary evidence.

Checkpoint selection uses validation tumor-only Dice. Threshold selection is validation-only. Test may be evaluated only after the checkpoint, preprocessing, threshold, and post-processing configuration are frozen. No result from the legacy split is eligible as a final claim.

## 2. Dataset inventory and clean split

- Source: 3,746 images, 1,867 LabelMe annotation files, and 3,746 metadata rows.
- Integrity: no missing tumor annotations, unreadable images, or orphan annotation files were found.
- Exact duplicates: 21 duplicate groups; 21 redundant images were excluded. Nine groups contain conflicting metadata, so silent retention would be unsafe.
- Eligible clean cohort: 3,725 images in 1,470 inferred case groups.
- Clean split (seed 42): train 2,981 images, validation 371 images, test 373 images.
- Tumor counts: train 1,488; validation 184; test 187.
- Clean split overlap checks: zero group overlap and zero exact-image-hash overlap.
- Split manifest SHA-256: `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`.
- One source subtype conflict (`IMG001276.jpeg`) was resolved to synovial osteochondroma using its matching polygon class. Its source metadata already has `tumor=1` and `benign=1`; therefore the official binary tumor/normal weak label is unchanged and does not use polygon information.
- The polygon-resolved subtype must not be used as a weak multiclass training target in the official result. The planned official classifier is binary; any multiclass diagnostic must exclude this record or remain non-official.

The public release has no patient, study, lesion, or accession identifier. The audit therefore groups consecutive image IDs with stable metadata while excluding view, which reduces obvious multi-view leakage but **cannot prove patient independence**. This limitation must remain attached to all reported results.

Artifacts: `artifacts/data_audit/dataset_audit_summary.json`, `split_manifest.csv`, `image_hash_manifest.csv`, `exact_duplicate_report.json`, and `near_duplicate_candidates.json`.

## 3. Leakage in the legacy protocol

Reconstructing the earlier image-level stratified split (seed 42) produced 2,996 train / 374 validation / 376 test images. It contains:

- 6 exact duplicate groups crossing partitions;
- 426 inferred case groups crossing partitions, involving 2,019 images;
- only 45 shared image IDs between the old validation set and the clean validation set.

Consequently, old validation/test numbers are development history only. They cannot be compared as if they came from the new locked cohort, and they cannot support a final performance claim.

## 4. Legacy checkpoint re-evaluation

Both available later checkpoints were evaluated on the exact legacy validation cohort with the same tumor-only formula:

| Artifact | Supervision | Tumor Dice | Tumor IoU | Tumor non-empty sensitivity | Normal specificity |
|---|---:|---:|---:|---:|---:|
| `btxrd_hybrid/unet_from_pseudo` | CAM + SAM pseudo masks | 0.17756 | 0.11198 | 0.99462 | 0.00000 |
| `btxrd_hybrid/supervised_unet_oracle` | Ground-truth masks | 0.07401 | 0.04519 | 0.91398 | 0.20745 |

The later run selected checkpoints with legacy all-image `val_dice`; normal empty masks could dominate that value. The pseudo-mask U-Net therefore predicts a non-empty mask on every normal validation image. The supervised run stopped after epoch 8 with its best checkpoint at epoch 1, whereas an older supervised run improved only around epoch 24. These facts make checkpoint selection and inadequate training budget primary confounders, not evidence that full supervision is intrinsically worse.

An older notebook reports pseudo-mask mean tumor Dice 0.24813 and a separately evaluated supervised checkpoint at 0.32459 on the legacy validation cohort. A single-image SAM diagnostic reached oracle candidate Dice 0.752 but selected Dice 0.0177, exposing a large candidate-selection loss. These notebook values have incomplete run manifests and remain diagnostic only.

Artifacts: `artifacts/legacy_eval/hybrid_wsss_val.csv/json` and `hybrid_supervised_val.csv/json`.

## 5. Implementation and environment audit

- Repository baseline: commit `980722ac4b3f673dd09a9b2156d78b6ad334d0d9`.
- Local runtime: Python 3.9, PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118; RTX 3050 Ti Laptop GPU (4 GB). The driver at the crashes was 462.62 dated 2021-05-11.
- Three training attempts were interrupted by full Windows reboots. The failures have direct evidence including `nvlddmkm` events 14/153, bugcheck `0x133`, and one PyTorch `CUDA illegal memory access` traceback. Resumable checkpoints load correctly on CPU. These interruptions are infrastructure failures and are not used as evidence for or against an experimental method.
- With user authorization, NVIDIA Studio 610.47 WHQL was downloaded from NVIDIA, Authenticode-verified, and installed after exporting the old signed package for rollback. Post-reboot CUDA and short ResNet18 U-Net stress tests passed, but sustained batch-8 training reproduced `nvlddmkm` plus bugcheck `0x133`; a driver update alone did not resolve the local instability.
- Sustained local GPU training is therefore unsafe. The controlled ablations should move to a private remote GPU using the same split hash, checkpoint hash, source, and metric protocol; the test partition remains locked.
- The repository requirements pin a newer PyTorch stack than the installed environment. Exact runtime versions must be written to every final run manifest.
- A local SAM ViT-B checkpoint is available and runtime downloads are disabled.
- The existing evaluator correctly computes per-image tumor Dice and retains complete misses. It also produces subgroup, bootstrap, pixel-confusion, and run-manifest artifacts.
- The initial unit suite had four failures caused by tests assuming notebook `cell.source` was always a string; nbformat also permits a list of lines. The test harness was corrected without altering notebook semantics.

## 6. Confirmed bottlenecks

1. **Invalid legacy partitioning:** exact duplicate and inferred case leakage invalidate prior test-like claims.
2. **Wrong historical selection metric:** all-image Dice rewards empty normal masks and is not the requested endpoint.
3. **Pseudo-mask selection loss:** a good SAM candidate can exist while the image-only selector chooses a poor candidate.
4. **Weak semantic localization:** multiclass classifier macro-F1 was approximately 0.60–0.63; class-conditioned CAMs can miss or mislocalize tumors.
5. **Severe pixel imbalance:** clean training masks contain only 1.2116% foreground pixels; the raw BCE positive weight is 81.53 and requires controlled weighting.
6. **Underpowered segmentation baseline:** scratch U-Net and premature early stopping do not establish a credible upper bound.

## 7. Changes under evaluation

- Added an ImageNet-pretrained ResNet-18 U-Net option with checkpoint architecture provenance.
- Checkpoint selection is tumor-positive validation Dice, with normal empty-case specificity only as a tie-breaker.
- Added validation-only threshold sweeping; the evaluator rejects test threshold sweeps.
- Started a fully supervised clean-validation upper-bound run using the same final evaluation metric. This diagnoses model capacity before spending the remaining budget on pseudo-label refinement.

No test result is reported in this document while research is in progress.

## 8. Actual execution flow

The production weak-supervision path is:

1. checksum-bearing group-aware split manifest;
2. image-level tumor classifier trained on clean-train images;
3. class-conditioned CAM from the classifier;
4. image/CAM-only morphology and prompts;
5. SAM candidate masks;
6. image-only candidate scoring and pseudo-mask manifest;
7. segmentation model trained on clean-train pseudo masks;
8. direct segmentation inference, followed by the common evaluator.

Polygon masks are outside this path. Clean-train polygons are used only in the explicitly labelled supervised upper-bound diagnostic. Clean-validation polygons are used for evaluation and bounded hyperparameter selection. Clean-test polygons remain inaccessible until the pipeline is frozen.

## 9. First clean upper-bound ablation

The ResNet-18 U-Net with BCE positive weight 20 reached 0.31840 tumor Dice at threshold 0.5. A validation-only threshold sweep selected 0.65 and improved it to 0.32627 (mean IoU 0.24688, normal specificity 0.65241), so calibration alone did not close the observed gap at that checkpoint. The run was paused after only five epochs when specificity deteriorated. It is **inconclusive, not rejected**: the learning curve is too short to establish convergence. An otherwise identical positive-weight-10 run completed three epochs and is resume-ready, but is paused after a confirmed NVIDIA driver/GPU crash; both runs remain governed by the learning-curve protocol in `IMPROVEMENT_PLAN.md`.

Subgroup analysis localizes the dominant error: 94/184 validation tumor images (51.1%) have a lesion occupying less than 1% of the resized image, and their mean Dice is only 0.08644. Medium lesions (1-5%) reach 0.52043 and large lesions (at least 5%) reach 0.72161 at the same checkpoint. The architecture can segment adequately sized lesions; small-lesion localization/resolution is the primary Gate-A bottleneck.

At the 320x320 evaluation grid, the 94 small-lesion masks have a median area of only 96.5 pixels (approximately 11.1-pixel equivalent circular diameter); the 25th percentile is 54 pixels and the minimum is 7 pixels. Scaling to 448 px would raise the median to about 189 pixels / 15.5-pixel equivalent diameter. This supports a controlled resolution ablation but is not itself performance evidence.

## 10. Clean-cloud small-lesion screening

The private Kaggle T4 screening run completed three controlled epoch-10 configurations on the same clean split and evaluator. Test was not accessed.

| Configuration | Validation-selected threshold | Mean tumor Dice | Small <1% Dice | Normal specificity | Decision |
|---|---:|---:|---:|---:|---|
| Full train, 320 px | 0.45 | 0.38501 | 0.17463 | 0.51872 | Superseded |
| Tumor-only train, 320 px | 0.60 | 0.37715 | 0.14445 | 0.58289 | Rejected and purged |
| Full train, 448 px | 0.75 | **0.41832** | **0.24904** | 0.62567 | Promoted |

Resolution is the only promoted change. Relative to full-320, 448 px gains +0.03331 overall and +0.07441 on the 94-image small subgroup. At threshold 0.5, complete tumor misses fall from 30 to 13 and tumor overlap detection rises from 70.7% to 77.2%. Medium-lesion Dice remains approximately unchanged, which localizes the benefit to the intended bottleneck.

The promoted 448 px run resumed exactly from epoch 10 and converged under the predeclared patience rule. Its best checkpoint is epoch 20; no later epoch through early stopping at epoch 30 exceeded the validation positive-Dice criterion. The full evaluator gives 0.489941 mean tumor Dice at threshold 0.5 and 0.495132 at validation-selected threshold 0.2. The latter has a group-bootstrap 95% CI of 0.442827-0.546114. This is the current clean-validation winner but remains below the strict `>0.50` target.

Tumor-only training is removed from the current source, CLI, dataset factory, tests, and active cloud bundle. After the new-schema epoch-20 checkpoint was saved and verified, the final inactive legacy resume shim and its assertions were also removed. A full search finds no tumor-only training token in source/tests, and the current suite passes 49/49 tests.

## 11. Frozen current-best pipeline audit

The winner is frozen at `artifacts/best_pipeline/fs_resnet18_pw10_full_448_e20/`, with `artifacts/best_pipeline/CURRENT.json` as the only promotion pointer. The snapshot contains clean source, tests, both notebooks required by the test suite, immutable split manifest, checkpoint, epochs 1-30 log, threshold selection, per-image outputs, subgroup tables, grouped bootstrap, and a per-file SHA-256 manifest. Its fail-closed verifier checks all 68 manifested files, split counts/isolation, metric semantics, all 184 tumor rows including misses, best epoch, threshold provenance, checkpoint identity, and `test_evaluated=false`. The verifier passes; the frozen source exactly matches the worktree across 40 project files and five test files; 49/49 snapshot-local tests pass. The 2026-07-24 deployment-only repair makes `inference.py` instantiate the checkpoint-declared `ResNet18UNet`; it changed no model checkpoint, training path, evaluator, metric, threshold, validation evidence, or test state.

Candidate experiments run outside this directory. `CURRENT.json` may change only for a strictly better clean-validation pipeline using the unchanged endpoint and locked test; rejected candidate code/artifacts must be purged.

No test result is reported while later small-lesion and weak-supervision stages remain in progress.

## 12. Blind-tiling candidate decision

The private version-2 blind-tiling kernel completed on all 371 clean-validation images and reproduced the frozen full-image result exactly (`0.4951316963`, selected threshold `0.20`). It recorded the frozen split/checkpoint hashes and `test_evaluated=false`.

The predeclared `blend50` candidate reached `0.4953371473` at validation-selected threshold `0.40`, but its paired improvement over the current winner was only `+0.0002054510` with a 10,000-resample group-bootstrap 95% CI of `[-0.0132936291, +0.0149442548]`. Normal empty-mask specificity declined from `0.5561497326` to `0.4705882353`; medium- and large-lesion Dice also declined. The candidate is rejected, the `CURRENT` model/pipeline is unchanged, and only compact validation evidence is retained. The test partition remains locked.

## 13. Gate-B binary classifier audit

The official weak-label path now uses a one-logit tumor-vs-normal DenseNet121 rather than the legacy ten-class tumor-type classifier. The private Kaggle run used only the clean-train image and binary `tumor` label; polygons and segmentation masks were not loaded by classifier training or evaluation. Checkpoint selection was clean-validation tumor F1 at the fixed probability threshold `0.5`.

The epoch-11 checkpoint reached F1 `0.7833333333`, sensitivity `0.7663043478`, specificity `0.8128342246`, precision `0.8011363636`, AUROC `0.8654528017`, and AUPRC `0.8772278795` on all 371 clean-validation images (141 TP, 35 FP, 43 FN, 152 TN). Training stopped at epoch 18 after the predeclared patience of seven epochs. The checkpoint SHA-256 is `f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5`; the split SHA-256 matches the frozen manifest, the per-image table has 371 rows, and the run manifest states `test_evaluated=false`.

This result establishes semantic gate quality but is not a segmentation result. Gate C subsequently measured binary LayerCAM localization, prompt quality, SAM oracle candidate Dice, support-clipping loss, candidate-selection loss, final pseudo-mask Dice, classifier false-negative impact, and normal false-positive behavior. The supplied-image-label protocol and classifier-predicted protocol were evaluated separately. Validation polygons were used only after predictions were fixed for these diagnostic metrics.

## 14. Foreground-crop/tiling supervised diagnostic

The corrected private Kaggle version completed 15 fine-tuning epochs and all four full-validation inference variants. The source full-image baseline reproduced `CURRENT` exactly. The crop-trained checkpoint was selected at epoch 12 (fixed-0.5 validation positive Dice `0.5070608605`, checkpoint SHA-256 `ec2cdd1f11bc50be1aea172e8c6f5285c98b909fc569c1f2f99224e634723eaf`). Clean-train polygons placed only training crops; validation inference used the full image or a blind dimension-only grid. This remains a fully supervised upper-bound diagnostic.

At validation-selected thresholds, candidate full-image inference reached Dice `0.5155799970`, and blend50 reached `0.5268559713`. The latter improved small/medium/large Dice to `0.3746071130/0.6791475124/0.7127671778`, but normal specificity fell to `0.4171122995`. A 10,000-resample paired group bootstrap gave `CURRENT`-relative deltas of `+0.0204483007` (95% CI `-0.0123973802` to `+0.0527633776`) for candidate-full and `+0.0317242750` (95% CI `-0.0008387065` to `+0.0648922460`) for blend50. The local audit independently reproduced both paired intervals from the 371-row per-image files.

Decision: this demonstrates supervised capacity above 0.50 but does not credibly replace the frozen winner because both paired intervals include zero, blend50 materially worsens normal false-positive behavior, and polygon-supervised training is outside the official WSSS claim. `CURRENT` and the locked test remain unchanged.

## 15. Gate-C binary CAM/SAM decomposition

The supplied-image-label protocol produced complete manifests and evaluations for 371 images. It localized all 184 tumors, emitted explicit empty masks for all 187 known normals, and reached pseudo-mask Dice `0.2023038352` (group-bootstrap 95% CI `0.1588296298-0.2467844435`). Foreground-support IoU was `0.0703875111`, foreground recall `0.6058501891`, point-hit rate `0.2705575974`, SAM oracle single-candidate Dice `0.3809186551`, selected Dice `0.2010835193`, support loss `0.0209441901`, and selection loss `0.1588909457`. The selector discards substantially more recoverable quality than support clipping does.

The separate predicted-gate diagnostic skipped 43/184 tumor images. Conditional downstream Dice on the 141 classifier true positives was `0.2224065422`; including the 43 required zero-mask failures reduced end-to-end Dice to `0.1704311003` (95% CI `0.1289388966-0.2110567873`). Normal specificity was `0.8181818182`. Both protocols match the frozen split and classifier hashes, have 371 per-image and pseudo-manifest rows, and record `test_evaluated=false`.

The first bounded selector experiment replaced best-per-component union with a single global top-1 candidate while keeping CAM, morphology, prompts, SAM, score method, thresholds, split, seed, and checkpoint unchanged. It is rejected: Dice fell from `0.2023038352` to `0.1777293956`, with paired delta `-0.0245744396` and 95% CI `[-0.0493509738, -0.0012575416]`. Oracle candidate Dice stayed exactly `0.3809186551`, selected Dice fell to `0.1868790157`, and selection loss increased to `0.1730954493`. Best-per-component union is therefore retained.

The second bounded selector-family run used the fixed `prompt_hybrid` ranker while retaining best-per-component/top-3 and every upstream CAM/SAM setting. Its predeclared component-local support semantics disabled the baseline global support clip. It is decisively rejected: Dice fell to `0.1118186396`, a paired delta of `-0.0904851956` with independently reproduced 10,000-resample group-bootstrap 95% CI `[-0.1242405321, -0.0558622391]`. Oracle single-candidate Dice remained exactly `0.3809186551`, but selected Dice fell to `0.1313552106` and selection loss increased to `0.2495634445`. All candidate hashes and row counts verify, and `test_evaluated=false`.

Selector experiments therefore returned to the original best-per-component `coverage_mass_sam` baseline. The isolated horizontal-flip CAM TTA experiment is promoted within Gate C: pseudo-mask Dice rose from `0.2023038352` to `0.2343392222`, a paired delta of `+0.0320353870` with independently reproduced 95% CI `[+0.0085405304, +0.0573916861]`. Foreground IoU/recall rose to `0.0767800857/0.6510582698`, oracle single-candidate Dice to `0.4090762905`, and selected Dice to `0.2336682193`. Small/medium/large Dice all improved to `0.1121634039/0.3486037144/0.4153105265`. All hashes and row counts verify, normal specificity remains `1.0`, and `test_evaluated=false`.

The last bounded Gate-C pool experiment retained the TTA winner and changed only `include_cam_candidate=true`. It is rejected: Dice decreased to `0.2306616510`, paired delta versus TTA `-0.0036775712`, independently reproduced 95% CI `[-0.0084839943, +0.0005739561]`; small, medium, and large Dice all decreased. Multiscale CAM was not launched because the current implementation explicitly rejects the binary classifier path. Gate C is frozen as binary LayerCAM with horizontal-flip TTA, SAM-only candidates, and the original best-per-component `coverage_mass_sam` selector.

Gate-D train pseudo-mask generation completed on Kaggle for all 2,981 clean-train images. It used only the supplied binary image label, explicitly disabled polygon-based prompt diagnostics, and verified every generated PNG against its manifest hash. The schema-v2 manifest SHA-256 is `7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6`; 1,463/1,488 tumors have non-empty pseudo masks, 25 are empty after localization, and all 1,493 known normals are empty. Split/classifier/SAM/source hashes match, while `train_polygons_loaded=false`, `validation_evaluated=false`, and `test_evaluated=false`.

The first manifest-bound ResNet18-U-Net launch stopped before training because the pseudo-manifest validator conflated the immutable 320 px mask grid with the 448 px consumer transform. The narrow correctness repair validates every pseudo mask and hash at its source grid, records source/consumer sizes, and then uses the already-existing nearest-neighbor training resize. A new immutable source bundle preserves the old one; critical source hashes and the pseudo-manifest hash are fail-closed. Unit and payload tests pass.

Gate-D version 2 is now executing on Kaggle at 448 px, batch 8, positive weight 10, seed 42, at most 35 epochs, and patience 10. No model, pseudo mask, loss, split, metric, checkpoint rule, or threshold protocol changed. Checkpoint selection remains fixed-threshold validation tumor Dice with normal specificity only as a tolerance tie-breaker; the post-training threshold grid is validation-only. Test remains locked.
