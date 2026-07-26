# BTXRD Research Continuation Handoff

Updated: 2026-07-26, Asia/Bangkok.

This document is the starting point for a new Codex task or account. Its purpose is to continue the work without reconstructing the protocol, opening the test set early, losing the current winner, or leaving rejected-ablation debris in the source tree.

## 1. Objective and immutable rules

Primary objective:

- Build an image-label-only weakly supervised bone-tumor segmentation pipeline.
- Under the locked paired consumer protocol, the pseudo-mask-trained U-Net must
  have absolute mean tumor-only Dice gap `<=0.10` from the GT-trained reference
  separately for small (`<1%`), medium (`1%` to `<5%`) and large (`>=5%`)
  validation tumors. This is the user-approved 2026-07-26 goal revision;
  required minimum WSL Dice is `0.22895493/0.56244178/0.59370336` for
  small/medium/large.
- Architecture and WSL technique are unrestricted, but the experimental arm
  may use only radiographs and image-level labels from BTXRD. Train GT masks,
  GT-derived crops/size/location, reference weights/predictions and
  pre-prediction validation annotations are prohibited.
- The authoritative reference and pair contracts are
  `artifacts/reference/gt_resnet18_unet_448_v1/reference_lock.json` and
  `paired_protocol_v2.json`. The earlier `paired_protocol_v1.json` and all
  evidence evaluated at tolerance `0.05` remain immutable historical records.
  The reference, consumer invariants and allowed supervision difference did
  not change. Protocol v2 canonical-LF SHA-256 is
  `2f7965b2ece0c00e9db6441562c489f84b5ccb942619a3c6a3d08ca2328359d0`;
  future pair audits must supply the protocol and this expected hash.
- Do not use data leakage, redefine metrics, change cohorts, remove complete
  misses, or select on test. Test stays locked until the final WSL pipeline is
  frozen.
- If the three gap criteria are not reached, report the best result, paired
  uncertainty, subgroup gaps, remaining bottlenecks and experimental evidence
  honestly.
- Run all heavy generation, training and full-cohort inference on Kaggle.

Source-cleanliness rules:

- Run every candidate outside the frozen winner snapshot.
- If an ablation is rejected, remove its runtime code, CLI/configuration options, tests, dependencies, and heavy artifacts.
- Keep only compact evidence required for auditability.
- Do not modify the frozen model, data, evaluation evidence, or source of the winner snapshot.
- Change the `CURRENT` pointer only when a candidate is a valid winner under the same clean-validation protocol.

## 2. Workspace and environment

Workspace:

```text
D:\thesis
```

Local Python used for audit and CPU work:

```text
C:\Users\USER\miniconda3\envs\btxrd-pseudomask\python.exe
```

Local versions:

- Python 3.9.23
- NumPy 1.23.5
- PyTorch 2.1.2+cu118

Important local-GPU warning:

- The machine has an RTX 3050 Ti Laptop GPU with 4 GB VRAM.
- NVIDIA Studio 610.47 WHQL was installed with user authorization.
- Short CUDA stress tests passed, but sustained training still caused `nvlddmkm` events 14/153 and bugcheck `0x133`.
- Do not use the local GPU for long training.
- Use a private Kaggle T4 runtime for heavy training and evaluation.

The winner checkpoint was produced in this Kaggle runtime:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA 12.8
- Tesla T4

Local PyTorch 2.1.2 may exit abnormally while loading the PyTorch 2.10 checkpoint. This is not evidence that the checkpoint is corrupt: its SHA-256 matches and Kaggle loaded and evaluated it successfully. Use a compatible Kaggle runtime to load the model, and use the local standard-library verifier for hashes, CSV files, and JSON files.

## 3. Data audit and clean split

Local dataset:

```text
D:\thesis\BTXRD\BTXRD
```

Immutable split manifest:

```text
D:\thesis\artifacts\data_audit\split_manifest.csv
```

Split-manifest SHA-256:

```text
85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c
```

Eligible split counts:

- Train: 2,981 images; 1,488 tumor and 1,493 normal.
- Validation: 371 images; 184 tumor and 187 normal.
- Test: 373 images; 187 tumor and 186 normal.

Verified properties:

- Zero cross-split group overlap.
- Zero cross-split exact-image SHA overlap.
- Twenty-one redundant exact-duplicate images were excluded.
- The legacy split has clear leakage through exact duplicates and inferred case groups crossing partitions.

Limitation that must remain in every report:

- BTXRD does not provide patient, case, study, or accession identifiers.
- `group_id` uses a heuristic based on consecutive image IDs and stable metadata while excluding view.
- This heuristic reduces obvious multi-view leakage but cannot prove patient-independent splitting.

## 4. Metric contract

Primary endpoint:

- Mean per-image Dice over every tumor image in the evaluated partition.
- Validation contains exactly 184 tumor images.
- A complete miss receives Dice 0 and remains in the denominator.
- Normal images are not mixed into the primary Dice mean.
- Normal empty-mask case specificity is reported separately.
- Threshold selection is allowed only on validation.
- The evaluator rejects a threshold sweep on test.
- Boundary metrics are conditional diagnostics only and never replace the primary endpoint.

Canonical implementation:

```text
project\evaluation\segmentation_metrics.py
project\evaluate_unet.py
project\evaluation\frozen_test_guard.py
```

## 5. Result chronology

Legacy results are diagnostic only and are invalid or unverified because of split or provenance problems:

- Legacy hybrid WSSS: tumor Dice 0.17756, invalid split.
- Legacy supervised model: tumor Dice 0.07401, invalid split.
- Notebook pseudo-mask model: tumor Dice 0.24813, incomplete manifest.
- Notebook supervised model: tumor Dice 0.32459, incomplete manifest.

Clean supervised upper-bound results:

1. ResNet18 U-Net, `pos_weight=20`, 320 px, 5 epochs:
   - Validation-selected threshold: 0.65.
   - Tumor Dice: 0.32627.
   - Inconclusive because the run had not converged.

2. Full training set, `pos_weight=10`, 320 px, epoch 10:
   - Validation-selected threshold: 0.45.
   - Overall Dice: 0.385008.
   - Small-lesion Dice below 1% area: 0.174631.
   - Medium-lesion Dice: 0.588980.
   - Large-lesion Dice: 0.667757.
   - Superseded.

3. Tumor-only training, 320 px, epoch 10:
   - Validation-selected threshold: 0.60.
   - Overall Dice: 0.377150.
   - Small-lesion Dice: 0.144453.
   - Rejected and purged from the active source.

4. Full training set, 448 px, epoch 10:
   - Validation-selected threshold: 0.75.
   - Overall Dice: 0.418316.
   - Small-lesion Dice: 0.249039.
   - Promoted for a longer learning curve.

5. Full training set, 448 px, convergence run:
   - Resumed from epoch 10.
   - Best epoch: 20.
   - Early stopped at epoch 30 with patience 10.
   - Dice at fixed threshold 0.5: 0.4899413582.
   - Dice at validation-selected threshold 0.2: 0.4951316963.
   - Group-bootstrap 95% CI: 0.4428268900 to 0.5461137726.
   - Small subgroup, 94 images: 0.3289549325.
   - Medium subgroup, 72 images: 0.6624417784.
   - Large subgroup, 18 images: 0.6937033566.
   - Normal empty-mask specificity at threshold 0.2: 0.5561497326.
   - `test_evaluated=false`.
   - This is the `CURRENT` winner, but it remains below the strict `>0.50` target.

Paired epoch-10 to epoch-20 winner comparison:

- Overall selected-threshold delta: +0.076816.
- Group-bootstrap 95% CI: +0.047631 to +0.106773.
- Small-lesion delta: +0.079916.
- Sixteen small-lesion overlaps were recovered and two were lost.

## 6. Small-lesion bottleneck

Validation evidence:

- 94 of 184 tumor images, or 51.1%, have a lesion below 1% of image area.
- The current winner reaches only 0.328955 Dice on this subgroup, while medium and large lesions reach about 0.66 and 0.69.
- Small-lesion overlap detection is 0.712766, with 14 complete misses at threshold 0.2.

Training-only geometry:

- 754 of 1,488 tumor training images have a lesion below 1% of image area.
- Median lesion bounding-box width is about 5.2% of image width.
- Median lesion bounding-box height is about 6.4% of image height.
- Small lesions are not globally scarce in the training set.
- Hand and foot cases remain weak despite the presence of training examples.

Research-allocation conclusion:

- Longer training produced a real gain but saturated at best epoch 20.
- Simple small-versus-large oversampling has less potential than spatial treatment.
- Prioritize blind tiling or foreground-aware crop training paired with blind tiled inference.
- If crop-aware training is evaluated, ground truth may locate a crop only for clean-train examples. Inference must use a blind grid and must never use ground truth or a validation target to place an ROI.

## 7. Frozen and audited winner pipeline

Canonical pointer:

```text
D:\thesis\artifacts\best_pipeline\CURRENT.json
```

Snapshot:

```text
D:\thesis\artifacts\best_pipeline\fs_resnet18_pw10_full_448_e20
```

The snapshot contains:

- Clean project source.
- Four test files.
- Two notebooks required by the test bootstrap.
- Split manifest.
- Winner checkpoint.
- Epochs 1-30 training log.
- Cloud training console log.
- Fixed-threshold and selected-threshold per-image CSV files.
- Threshold sweep and selection evidence.
- Subgroup CSV files.
- Grouped-bootstrap JSON evidence.
- Convergence summary.
- `pipeline_lock.json`.
- `FILE_MANIFEST.csv`.
- `verify_pipeline.py`.
- `AUDIT_VERIFICATION.json`.

Winner checkpoint:

```text
model\best_unet.pt
```

Checkpoint SHA-256:

```text
05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31
```

Checkpoint size:

```text
230,924,939 bytes
```

`FILE_MANIFEST.csv` SHA-256 after the English documentation repair:

```text
c7b6a8a720dbaf26514f22dda6ed18a5b5913afcafef884b1b730d50a288b898
```

Audit status: PASS.

- 68 manifested files.
- 41 project files in the snapshot match the cleaned worktree, diff 0.
- Four test files in the snapshot match the cleaned worktree, diff 0.
- 49 of 49 tests pass inside the snapshot.
- No `train_tumor_only`, `train-tumor-only`, or rejected legacy token remains.
- Split isolation and counts pass.
- Checkpoint hash and byte-size checks pass.
- Best epoch and training-log checks pass.
- Evaluation contains 371 rows: 184 tumor and 187 normal.
- Recomputed per-image mean equals 0.4951316963.
- The threshold was selected on validation.
- `test_evaluated=false`.

Verification commands:

```powershell
cd D:\thesis\artifacts\best_pipeline\fs_resnet18_pw10_full_448_e20
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Users\USER\miniconda3\envs\btxrd-pseudomask\python.exe verify_pipeline.py
C:\Users\USER\miniconda3\envs\btxrd-pseudomask\python.exe -m unittest discover -s tests -p 'test_*.py'
```

The model, data, evidence, and source inside this snapshot are frozen. New candidates must use separate `tmp` and candidate-artifact directories. Documentation-only encoding repairs must be reflected in `FILE_MANIFEST.csv` and followed by a verifier run.

## 8. Kaggle external state

The Kaggle CLI is authenticated to the user's account.

Private research dataset:

```text
itsthang333/btxrd-small-lesion-research-v1
```

Dataset ID:

```text
11311114
```

Latest uploaded payload:

- `project.zip`: audited clean source.
- `split_manifest.csv`.
- `winner_448_epoch20.zip`: checkpoint and training log.
- Dataset is private and ready.
- The latest payload contains no radiographs, credentials, caches, or rejected checkpoints.

Current winner convergence kernel:

```text
itsthang333/btxrd-448-convergence
version 3
status COMPLETE
```

Candidate kernel at the 5% usage stop:

```text
itsthang333/btxrd-small-lesion-tiling
version 2
private T4
status RUNNING at the recorded stop checkpoint
```

Version 1 failed after about five seconds because Kaggle exposed the research bundle as ZIP archives instead of extracted folders. Version 2 adds archive-mount extraction only; the model, tiling method, metric, and protocol are unchanged. Do not push another version until version 2 status and logs have been checked.

Candidate local source:

```text
D:\thesis\tmp\kaggle\tiling_kernel_v1\cloud_small_lesion_tiling.py
D:\thesis\tmp\kaggle\tiling_kernel_v1\kernel-metadata.json
```

Candidate local payload:

```text
D:\thesis\tmp\kaggle\small_lesion_bundle_v2
```

The user granted standing authorization for Kaggle operations. Do not ask for permission at each Kaggle step.

## 9. Running candidate: blind tiling

Predeclared method:

- Run full-image inference at 448 px.
- Take nine blind crops directly from the original radiograph before resizing.
- Tile scale is 60% of image width and height.
- Start positions are 0.0, 0.2, and 0.4 on both axes.
- Merge tile predictions with a uniform mean in overlapping regions.
- `tile_only` is a diagnostic candidate.
- `blend50 = 0.5 * full_probability + 0.5 * tiled_probability` is the primary candidate.
- Tile coordinates depend only on image size.
- Validation targets are used only after inference to compute metrics.
- No annotation is used to place a tile.
- Keep the threshold grid unchanged at 0.20 through 0.85, with 14 values.
- The full-image baseline must reproduce Dice 0.49513 within tolerance; otherwise the kernel must fail.
- `test_evaluated=false`.

First status command:

```powershell
kaggle kernels status itsthang333/btxrd-small-lesion-tiling
```

After status is `COMPLETE`, download compact outputs first:

```powershell
kaggle kernels output itsthang333/btxrd-small-lesion-tiling --path D:\thesis\artifacts\kaggle\tiling_v1 --file-pattern '(tiling_ablation_summary\.json|evaluation/.*\.(json|csv))$' --page-size 200
```

Inspect:

- `tiling_ablation_summary.json`.
- Full-image baseline reproduction.
- `tile_only` selected Dice and threshold.
- `blend50` selected Dice and threshold.
- Small, medium, and large subgroups.
- Normal specificity.
- Group bootstrap.
- Paired delta and confidence interval.
- Recovered and lost overlaps.
- `test_evaluated` must be `false`.

Decision rules:

- If candidate Dice is not greater than 0.4951316963, reject it and purge candidate runtime code, source, and heavy artifacts. Keep only a compact summary in the reports.
- If it is higher but not above 0.50, it may replace `CURRENT` only if the protocol passes and paired evidence is credible; research must continue.
- If it exceeds 0.50, perform a full audit, freeze a new snapshot, and update `CURRENT`. Do not open the test set if the weak-supervision or final-pipeline scope is not genuinely frozen.
- If Dice rises through excessive mask expansion, report the normal-specificity and precision tradeoff honestly.
- Never select a variant or threshold on test.

## 10. If blind tiling is rejected

The highest-potential next controlled ablation is:

```text
Foreground-aware crop training plus blind tiled inference
```

Recommended design:

- Start from the winner checkpoint or a clearly separated training branch; do not modify `CURRENT`.
- Train on a mix of full images and 60% image crops.
- For tumor training examples only, jitter the crop center around the ground-truth lesion.
- For normal training examples, sample random crops from the same crop-size distribution.
- Retain full-image examples so the model does not lose global context.
- Validation and test inference must use the same fixed 3-by-3 blind grid and must never use ground truth or a target to place an ROI.
- Change only this technique family; keep the loss, architecture, seed, and split unchanged.
- Use a learning-curve gate for screening; do not decide from only a few epochs.
- Predeclare the promotion and rejection rule.
- If rejected, purge every crop flag, transform, test, kernel source, and temporary checkpoint.

Only after that ablation, and not simultaneously with it, consider:

- A predeclared Focal Tversky loss with the same data and inference.
- Do not combine multiple changes in one run because the gain would not be attributable.

## 11. Remaining weak-supervision and final-pipeline scope

The current winner is a fully supervised upper bound trained with polygon masks. It shows that the segmentation backbone can approach the target, but the original overall mission requires a weak-supervision pipeline.

Planned production weak-supervision path:

1. Freeze the clean split.
2. Train an image-level tumor classifier.
3. Generate class-conditioned CAM or LayerCAM.
4. Derive morphology and prompts from only the image and CAM.
5. Generate SAM candidate masks.
6. Score or fuse candidates using image-only information.
7. Produce a pseudo-mask manifest for all clean-training images.
8. Train the segmentation model on pseudo masks.
9. Freeze the final pipeline.
10. Evaluate test exactly once.

Clean-training polygons may be used only for the explicitly labelled supervised upper-bound diagnostic or a clearly labelled crop-training experiment. Validation and test polygons must never generate pseudo masks or place an ROI during inference.

Before marking the goal complete, verify against the mission PDF whether a fully supervised winner can be the final model or is only an upper bound. Do not silently equate the upper bound with the final weakly supervised pipeline.

## 12. Main reports and evidence

Files:

```text
D:\thesis\AUDIT_REPORT.md
D:\thesis\IMPROVEMENT_PLAN.md
D:\thesis\RESEARCH_LOG.md
D:\thesis\EXPERIMENT_COMPARISON.csv
D:\thesis\artifacts\kaggle\convergence_448_v3
D:\thesis\artifacts\small_lesion_analysis
D:\thesis\artifacts\best_pipeline
```

`RESEARCH_LOG.md` records the convergence result and frozen-winner audit.

`AUDIT_REPORT.md` records:

- Convergence at best epoch 20.
- Removal of the final legacy compatibility shim.
- Current-best snapshot audit.
- All 49 tests.
- The still-locked test set.

## 13. Cleanup policy and history

Already purged:

- Tumor-only CLI, filter, factory, tests, and runtime.
- Tumor-only checkpoint artifact.
- Obsolete kernel branch.
- Final legacy `train_tumor_only` resume shim.
- `project/tests/__pycache__` before the snapshot was created.

Do not delete:

- The `CURRENT` winner snapshot.
- Convergence evidence.
- Split-audit evidence.
- Compact experiment-comparison rows.

After every candidate:

- Decide immediately whether to promote or purge it.
- Delete temporary kernel source if rejected.
- Delete large candidate checkpoints if rejected.
- Remove remote kernel or dataset history only after the latest clean dataset version and winner are verified; do not risk deleting the current dataset.
- Keep compact summary and per-image evidence only when needed for audit.

The worktree remains dirty and contains untracked research work. Do not hard-reset it and do not use checkout to remove user changes.

## 14. Checklist for a new task or account

1. Read this document completely.
2. Read `CURRENT.json` and `pipeline_lock.json`.
3. Run `verify_pipeline.py` and all 49 tests.
4. Check the Kaggle tiling-kernel status.
5. If it is `COMPLETE`, download compact outputs and verify `test_evaluated=false`.
6. Compare the candidate with `CURRENT` using paired group bootstrap.
7. Promote or purge the candidate.
8. If Dice is still not above 0.50, run exactly one crop-aware and tiling ablation as defined in section 10.
9. Keep test locked until the final pipeline is genuinely frozen.
10. When usage falls below 5%, do not launch another job. Update this document with kernel status, output paths, hashes, metrics, and the exact next command, then stop.

## 15. Goal status

The goal remains active.

Do not mark it complete because:

- Frozen `CURRENT` clean-validation Dice is 0.49513. A supervised crop diagnostic crossed 0.50 only as an uncertain polygon-trained upper bound and is not the final WSSS model.
- The frozen Gate-C validation pseudo-mask Dice is 0.23434 and the full train pseudo-mask/segmenter stages are not complete yet.
- The final weak-supervision scope was not complete or audited.
- The test set remains correctly locked.

## 16. Five-percent usage stop checkpoint

The user reported 5% usage remaining. Work stopped without launching another job. The already-running private T4 kernel was intentionally left running so completed inference would not be discarded.

First commands for the next task or account:

```powershell
kaggle kernels status itsthang333/btxrd-small-lesion-tiling
kaggle kernels logs itsthang333/btxrd-small-lesion-tiling
```

If status is `COMPLETE`, download only `tiling_ablation_summary.json` and evaluation CSV/JSON files first. If status is `ERROR`, read the traceback before changing anything. `CURRENT` remains `fs_resnet18_pw10_full_448_e20` at Dice 0.4951316963 and is fully audited.

## 17. Continuation after the five-percent checkpoint

The blind-tiling kernel completed successfully and its compact outputs were downloaded to:

```text
D:\thesis\artifacts\kaggle\tiling_v1\btxrd_small_lesion_tiling
```

Decision: rejected and purged.

- Baseline reproduced: `0.4951316962732512`, threshold `0.20`.
- Tile-only Dice: `0.43346991734240276`.
- Predeclared blend50 Dice: `0.49533714732323847`, threshold `0.40`.
- Paired delta: `+0.00020545104998727208`.
- Paired group-bootstrap 95% CI: `-0.01329362905419054` to `+0.014944254845535158`.
- Small-lesion delta: `+0.015205901893624767`; six overlaps recovered, one lost.
- Normal specificity fell from `0.5561497326203209` to `0.47058823529411764`.
- `test_evaluated=false`.
- Summary SHA-256: `55c7a85c2e1ad4011f410d30e08effb18a31cb129b705f019a66d172ae4e4b7c`.

The raw point estimate was only trivially higher, the paired CI crossed zero widely, and specificity plus medium/large performance worsened. Therefore it does not satisfy the handoff's credible-evidence condition for replacing `CURRENT`.

The frozen winner was re-verified after the deployment-loader correctness repair: verifier PASS and 49/49 tests PASS. `FILE_MANIFEST.csv` now has SHA-256 `c7b6a8a720dbaf26514f22dda6ed18a5b5913afcafef884b1b730d50a288b898`, and `CURRENT.json` carries the same value. The loader now supports the checkpoint-declared `ResNet18UNet`; no checkpoint, training path, data, evaluator, metric, threshold, validation evidence, or test state changed.

Next experiment: run exactly one foreground-aware 60% crop-training ablation paired with the same blind 3-by-3 inference, as specified in section 10. Keep `CURRENT` and the test partition locked.

## 18. Binary WSSS continuation

Gate B completed successfully on private Kaggle kernel `itsthang333/btxrd-clean-binary-classifier`.

- Best checkpoint: epoch 11, SHA-256 `f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5`.
- Full clean validation: 371 images, 184 tumor, 187 normal.
- Fixed threshold 0.5: F1 `0.7833333333`, sensitivity `0.7663043478`, specificity `0.8128342246`, precision `0.8011363636`, AUROC `0.8654528017`, AUPRC `0.8772278795`.
- Confusion: TP 141, FP 35, FN 43, TN 152.
- Early stop: epoch 18, patience 7, best not at budget boundary.
- Split SHA matches `85511ee1...`; `test_evaluated=false`.
- Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_classifier_v1\btxrd_clean_binary_classifier`.

Gate C completed at `itsthang333/btxrd-binary-cam-sam-gate-c`.

- Known-image-label protocol: Dice `0.2023038352`, 95% CI `0.1588296298-0.2467844435`, 184/184 tumors processed, normal specificity `1.0` by explicit known-normal empty masks.
- Predicted-gate protocol: conditional Dice `0.2224065422` on 141 reached tumors; 43 classifier false negatives reduce end-to-end Dice to `0.1704311003`; normal specificity `0.8181818182`.
- Known-label decomposition: foreground IoU/recall `0.0703875111/0.6058501891`, oracle candidate Dice `0.3809186551`, selected Dice `0.2010835193`, support loss `0.0209441901`, selection loss `0.1588909457`.
- Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_cam_sam_gate_c_v1\btxrd_binary_cam_sam_gate_c`.

Foreground-crop/tiling version 2 also completed. Candidate-full Dice is `0.5155799970`; blend50 is `0.5268559713`, but paired 95% CIs versus `CURRENT` include zero and blend50 specificity falls to `0.4171122995`. It remains a supervised upper-bound diagnostic and does not replace `CURRENT` or satisfy the WSSS goal. Compact outputs: `D:\thesis\artifacts\kaggle\foreground_crop_tiling_v2\btxrd_foreground_crop_tiling`.

Global top-1 completed and is rejected: Dice `0.1777293956` versus baseline `0.2023038352`, paired delta `-0.0245744396`, 95% CI `-0.0493509738` to `-0.0012575416`. Oracle candidate Dice was unchanged, while selected Dice and selection loss worsened. Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_cam_sam_global_top1_v1\btxrd_binary_cam_sam_global_top1`.

Prompt-hybrid completed and is rejected: Dice `0.1118186396` versus baseline `0.2023038352`, paired delta `-0.0904851956`, independently reproduced 95% CI `-0.1242405321` to `-0.0558622391` across 167 tumor groups. Foreground support and oracle candidate quality were unchanged, but selected Dice fell to `0.1313552106` and selection loss rose to `0.2495634445`. Hashes, 371 evaluation/pseudo-manifest rows, the 184/187 tumor/normal counts, and `test_evaluated=false` all verify. Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_cam_sam_prompt_hybrid_v1\btxrd_binary_cam_sam_prompt_hybrid`.

Horizontal-flip TTA completed and is promoted within Gate C: Dice `0.2343392222` versus baseline `0.2023038352`, paired delta `+0.0320353870`, independently reproduced 95% CI `+0.0085405304` to `+0.0573916861` across 167 tumor groups. Oracle Dice rose to `0.4090762905`, selected Dice to `0.2336682193`, and all small/medium/large subgroups improved. Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_cam_sam_tta_flip_v1\btxrd_binary_cam_sam_tta_flip`. Comparison SHA-256: `aa945e434b38a07df568841c3feb93d615a8aac757e701138613de5c9cbf74fd`; test remains locked.

CAM-component fallback completed and is rejected: Dice `0.2306616510` versus TTA `0.2343392222`, paired delta `-0.0036775712`, independently reproduced 95% CI `-0.0084839943` to `+0.0005739561`; small, medium, and large Dice all decreased. Compact outputs: `D:\thesis\artifacts\kaggle\wsss_binary_cam_sam_tta_flip_cam_candidate_v1\btxrd_binary_cam_sam_tta_flip_cam_candidate`. Comparison SHA-256: `29704cbef73d5481cb4d33811a6775a972a3445ba424b3597d8e50b6e8a246a5`.

Gate C is frozen at the TTA-only winner. Train pseudo generation completed at `itsthang333/btxrd-wsss-train-pseudo-v1`: 2,981 masks were hash-verified, manifest SHA-256 is `7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6`, 1,463/1,488 tumor masks are non-empty, 25 tumor masks are empty after localization, and all 1,493 normals are empty. Train polygon diagnostics, validation evaluation, and test evaluation were disabled.

Heavy training/inference must run on Kaggle. Local work is limited to lightweight tests, audit, packaging, and artifact verification.

## 19. User-requested shutdown pause

The user requested a local pause so the computer can be shut down. Local polling and editing stopped. The independent Kaggle kernel `itsthang333/btxrd-wsss-train-pseudo-v1` was still `RUNNING` at the last successful status check and must be left alone; shutting down the local computer does not stop it.

The next segmentation-training payload is prepared under `D:\thesis\tmp\kaggle\wsss_segmenter_v1`, but it has **not** been pushed. Its source intentionally contains `EXPECTED_PSEUDO_MANIFEST_SHA256 = "__FILL_AFTER_PSEUDO_GENERATION__"`, and a static test rejects launch until the completed pseudo-manifest hash is frozen. No local GPU training is active. Test remains unopened.

First commands after restart:

```powershell
$env:PYTHONUTF8='1'
kaggle kernels status itsthang333/btxrd-wsss-train-pseudo-v1
```

If `COMPLETE`, download compact JSON/CSV outputs first and verify `generation_summary.json`, `run_manifest.json`, 2,981 manifest rows, 1,488 tumor/1,493 normal rows, 2,981 hash-verified masks, `train_polygons_loaded=false`, `validation_evaluated=false`, `test_evaluated=false`, frozen split/classifier/SAM hashes, TTA enabled, CAM fallback disabled, and selector `coverage_mass_sam`. Then place the verified pseudo-manifest SHA-256 into `tmp/kaggle/wsss_segmenter_v1/cloud_converge_448.py`, update the predeclared protocol, run the payload tests, and only then push `itsthang333/btxrd-wsss-segmenter-v1`.

If `ERROR`, fetch the Kaggle log and repair only the exact runtime/correctness fault; do not change the frozen Gate-C recipe or open test.

## 20. Resume after shutdown

The train pseudo kernel completed successfully. Compact outputs are under `D:\thesis\artifacts\kaggle\wsss_train_pseudo_v1\btxrd_wsss_train_pseudo_v1`. All population, provenance, recipe, manifest, metadata, source, and test-lock checks passed. The verified manifest SHA-256 `7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6` now binds the downstream checkpoint.

Kaggle kernel `itsthang333/btxrd-wsss-segmenter-v1` version 1 stopped before epoch 1: the loader correctly found the manifest but incorrectly required its 320 px source grid to equal the 448 px training consumer grid. The dataset already resizes pseudo masks with nearest-neighbor in `_build_mask`; the repair now validates masks on the recorded source grid and explicitly records the consumer resize. Worktree and immutable `btxrd-small-lesion-research-v2` source hashes match; the old bundle was not overwritten. Targeted integration and payload tests pass.

Version 2 is `RUNNING`. It retains pretrained ResNet18-U-Net 448 px, batch 8, AdamW `1e-4`, positive weight 10, seed 42, maximum 35 epochs, patience 10, the identical pseudo-mask manifest, validation checkpoint/threshold rules, and locked test.

## 21. Gate-C2 pseudo-mask morphology audit

Kaggle CPU kernel `itsthang333/btxrd-wsss-pseudo-morphology-v1` completed a validation-only fixed morphology ablation on the promoted flip-TTA pseudo masks. The unmodified source reproduced Dice `0.2343392222`. Erosion widths 3/5/7/9 px were all worse at `0.2246593401/0.2071840015/0.1938446124/0.1824582669`; the smallest erosion already had paired delta `-0.0096798820` with 95% CI `-0.0172828969` to `-0.0031285875` across 167 tumor groups. The result covers 371 unique rows with 184 tumors and 187 normals; provenance hashes match, train polygons were not loaded, and test was not evaluated. Compact outputs are under `D:\thesis\artifacts\kaggle\wsss_pseudo_morphology_v1\btxrd_wsss_pseudo_morphology_v1`; summary SHA-256 is `9bcca010618c2370f1024ec87ada2bf0df8921b5de562160bd5172bc39443d03`. Decision: retain the original pseudo masks and reject erosion.

The prepared Gate-E inference payload is under `D:\thesis\tmp\kaggle\wsss_inference_refinement_v1`. It must not be pushed until Gate-D version 2 is `COMPLETE`, because its kernel source must resolve to the completed WSSS checkpoint rather than the earlier runtime-error version. It predeclares full-image reproduction, flip-TTA, blind tile blend, and TTA-plus-tile blend on validation only; test remains locked.

## 22. Gate-D version-2 result and causal-selector follow-up

Gate-D version 2 trained through epoch 23 and early-stopped correctly after the epoch-13 validation winner. It then completed fixed and threshold-selected validation, but the outer wrapper raised after computation because it required a `missing` key that the current evaluator does not emit. The population evidence is complete: both output CSVs have 371 unique rows, 184 tumors, and 187 normals. The wrapper now accepts the current schema while still rejecting a nonzero `missing` field if one is present; 6/6 payload tests pass.

The recovered scientific result is below the promoted pseudo-mask baseline: fixed-0.5 Dice `0.2172072031`; selected threshold `0.85` gives Dice `0.2300198701`, 95% CI `0.1857288351-0.2750307254`, and normal specificity `0.5133689840`. Small/medium/large Dice is `0.0710014171/0.4134030430/0.3269168774`. Training positive Dice reached `0.7381718686`, so longer identical training is not justified: the model fitted noisy pseudo targets without improving true localization. Compact artifacts are under `D:\thesis\artifacts\kaggle\wsss_segmenter_v2_error\btxrd_wsss_segmenter_v1`; selected summary SHA-256 is `d5ccc8acb5a8b9c35341c7bb0d7232ced406dc9f2884f525e28335aa44b2d2ca`. Test was not evaluated.

The Gate-E inference payload remains prepared but is deprioritized because a small inference-only gain cannot plausibly close the observed gap to 0.50, and the failed wrapper version is not a usable downstream Kaggle kernel source without repeating training or repackaging the checkpoint.

The active next experiment is `itsthang333/btxrd-binary-cam-sam-causal-selector`. It retains the promoted flip-TTA candidate pool and adds only within-component classifier-causal ranking: equal positive deletion/insertion logit evidence using a deterministic blurred same-image replacement, weighted 20% beside CAM density/mass and SAM rank. Source bundle `itsthang333/btxrd-small-lesion-research-v3` contains 41/41 worktree-matching project files and the unchanged split SHA. It evaluates validation only and compares against the frozen TTA per-image CSV by paired tumor-group bootstrap; test remains locked.

## 23. Research resumed: adversarial-climbing CAM result

The causal selector was rejected as recorded above. Research then resumed after the thesis-ready `main` branch was finalized.

Kaggle kernel `itsthang333/btxrd-binary-adversarial-climbing-cam` completed a validation-only AdvCAM-inspired LayerCAM expansion test:

- Candidate tumor-only Dice: `0.2489722614`.
- Promoted flip-TTA baseline: `0.2343392222`.
- Paired delta: `+0.0146330392`.
- Independently reproduced paired tumor-group 95% CI: `-0.0187006140` to `+0.0476416375` over 167 groups / 184 tumor images.
- Small-lesion delta: `-0.0475064946` on 94 images; medium `+0.0730616899`; large `+0.1054248913`.
- Population: 371 unique validation rows, 184 tumor, 187 normal, zero missing.
- Provenance: split/classifier/SAM/source hashes all match; 11/11 Kaggle Torch preflight tests pass; `test_evaluated=false`.
- Comparison SHA-256: `3a387161a649b3240fe86b7af9c10c714d491de66cb00ea2f10d29d758c1aadc`.
- Compact artifacts: `D:\thesis\artifacts\kaggle\wsss_adversarial_climbing_cam_v1`.

Decision: **reject**. The point estimate rose, but the uncertainty interval crosses zero and the dominant small-lesion subgroup deteriorated substantially. The runtime implementation was removed after preserving commit history and evidence.

Mechanism audit:

- Small-lesion CAM support recall improves by `+0.190074`, SAM oracle Dice by `+0.027705`, and clipped-oracle Dice by `+0.036647`.
- The regression occurs in ranking/fusion: selection loss rises by `+0.077056`, selected Dice falls by `-0.040409`, predicted area expands by `+0.024175`, and precision falls by `-0.062437`.
- Medium and large lesions gain at both oracle and selected stages. Therefore the expanded proposal pool contains useful information; using the expanded map as its own ranking signal is the specific failure.
- Simple GT-free scalar gates are not adequate (best tested single-feature benefit AUC approximately `0.55`; fixed one-percent predicted-area gate Dice `0.243597`). GT-size and per-image-oracle hybrids are diagnostic only and must not be implemented.

Next controlled experiment: generate components/prompts/SAM candidates/support with the expanded AdvCAM map, but score the candidates with the original promoted flip-TTA CAM using the unchanged `coverage_mass_sam` formula. This dual-map design uses no GT size or polygon and introduces no learned validation gate. Freeze all other settings. Promotion requires a positive lower bound in the overall paired group bootstrap versus `0.2343392222` and a non-negative small-lesion point delta. Do not train the downstream segmenter unless that rule passes. The 448 px classifier ablation is deferred until this cheaper causal test is resolved.

## 24. Dual-map result and eligibility/ranking split

The dual-map run `itsthang333/btxrd-binary-advcam-dual-selection` completed:

- Dice `0.2424091257` versus baseline `0.2343392222`.
- Paired delta `+0.0080699035`, 95% CI `-0.0177663368` to `+0.0336182788`.
- Small Dice `0.1004806976`, delta `-0.0116827063`.
- Compared with expanded-map ranking, precision improves to `0.2480668260` but recall falls to `0.4582456568`; complete misses increase from one to nine.
- All 371/184/187 population checks, source/split/checkpoint hashes, 11 Torch tests, and `test_evaluated=false` verify.
- Comparison SHA-256: `ed3e2d7d2aca53e432297523d26b6da8dd40a5258490e7531751fe8ab78bf6b7`.
- Compact artifacts: `D:\thesis\artifacts\kaggle\wsss_advcam_dual_selection_v1`.

Decision: reject. It partially rescues small-lesion precision but applies the baseline score too broadly: eligibility thresholding and component priority lose useful expanded proposals.

Next controlled experiment: expanded-CAM score determines whether a candidate/component is eligible and ranks the top three components; baseline-CAM score selects the best SAM mask only within each eligible component. Keep threshold, score formula, proposal pool, support clipping and morphology fixed. This is not a weight sweep or GT-size gate.

## 25. Split-selector result and technique-family closure

The split-selector run `itsthang333/btxrd-binary-advcam-split-selector` completed on the locked validation split:

- Dice `0.2387746569` versus baseline `0.2343392222`.
- Paired delta `+0.0044354347`, 95% CI `-0.0271092059` to `+0.0354617169`.
- Small Dice `0.0673641053`, delta `-0.0447992986` across the fixed 94-image subgroup.
- Medium delta `+0.0476357232`; large delta `+0.0887489992`.
- Mean precision/recall `0.2289799200/0.5518265541`; one complete tumor miss.
- Population and provenance pass: 371 unique rows, 184 tumors, 187 normals, zero missing, frozen split/classifier/SAM/source hashes, 11 Torch tests plus 30 audit tests, and `test_evaluated=false`.
- Comparison SHA-256: `41ecb86cb423c929993d8197968191155981fc00c5ec4c69a437a3a0dd6e61d6`.
- Compact artifacts: `D:\thesis\artifacts\kaggle\wsss_advcam_split_selector_v1`.

Decision: reject. The three controlled variants establish that expanded CAM contains useful medium/large evidence and improves the candidate oracle, but none of the available ground-truth-free post-hoc scoring variants can preserve that gain without harming small lesions. Do not add another selector threshold, size gate, or validation-derived mixture.

Next experiment: train the same clean binary DenseNet121 classifier at 448 px on Kaggle, then evaluate the unchanged promoted flip-TTA LayerCAM plus SAM Gate-C protocol. Treat classifier input resolution as the scientific variable; preserve relative morphology geometry by predeclared proportional scaling where a pixel constant is resolution-dependent. Promotion still requires an overall positive paired-bootstrap lower bound and no decrease in the fixed small-lesion subgroup. Test remains locked.

## 26. Classifier 448 result and regularized-expansion follow-up

The controlled 448 px classifier completed at epoch 12. Its fixed-threshold
validation F1 is `0.7959183673`, sensitivity `0.8478260870`, specificity
`0.7219251337`, AUROC `0.8665717275`, and checkpoint SHA-256
`bb70912c179bda8cc32498b6bf8c405f11d46a37143cb4577a9986a838cec45c`.
All 371 rows, checkpoint metadata, split SHA and `test_evaluated=false` verify.

Gate-C with the 448 classifier is rejected:

- Dice `0.2296019980`; paired delta `-0.0047372242`.
- Paired group-bootstrap 95% CI `-0.0415031583` to `+0.0320557694`.
- Small/medium deltas `+0.0018854426/+0.0015835381`; large delta
  `-0.0646053111`.
- Small foreground recall, SAM oracle and clipped-oracle improve
  `+0.207594/+0.041537/+0.048980`, but small selected Dice changes
  `-0.000195` because selection loss increases `+0.049174`.
- Across all tumors oracle improves `+0.025554`, selected Dice falls
  `-0.005526`; large proposal quality itself deteriorates.
- Comparison SHA-256:
  `98f5925c884a0b84b27aeb4a06ed4ebd6b80a748d77767edf402c7e7378a1daa`.
  Nine Torch plus 29 audit tests pass; 371/184/187 rows are complete; test is
  untouched.

Interpretation: 448 and unregularized AdvCAM both add useful candidate evidence,
especially for small lesions, but neither provides a calibrated ranking signal.
Do not add an image-size gate or another post-hoc score threshold.

Next experiment: return to the frozen 320 px classifier and add the
activation-difference regularizer omitted by the first bounded AdvCAM
adaptation. Keep ten steps and step size `0.08`; use the official repository
defaults `AD_coeff=7` and activation threshold `0.5` on the differentiable
final LayerCAM layer. Everything else remains the promoted flip-TTA Gate-C
recipe. This isolates whether preserving the original discriminative core
while expanding new evidence can retain the measured medium/large gain without
destroying small-lesion selection. Test remains locked.

## 27. Activation-regularized AdvCAM result

The regularized run `itsthang333/btxrd-binary-advcam-regularized` is rejected:

- Dice `0.2307861310`; paired delta `-0.0035530912`.
- Paired group-bootstrap 95% CI `-0.0289307324` to `+0.0207423190`.
- Small/medium/large deltas versus baseline:
  `-0.0399829922/+0.0275501896/+0.0622788244`.
- Compared with unregularized climbing, small selection loss improves
  `0.037925` and small Dice improves `0.007524`, but small oracle Dice falls
  `0.027753`; medium/large Dice also lose `0.045512/0.043146`.
- Compared with baseline, small oracle Dice is unchanged (`-0.000048`) and
  selected Dice still falls `-0.033557`.
- Population 371/184/187, 184 prompt rows, 11 Torch tests, hashes and test lock
  all verify. Comparison SHA-256:
  `0daec4744259e766b2e949fe4cbc9de21e8f4f81a5df35a5c605ca7c947e4e54`.

Interpretation: the official `L_AD` defaults suppress part of the harmful
expansion but also discard useful oracle coverage. Do not sweep its coefficient
on validation. Inference-time climbing and post-hoc selector variants are now
closed.

Next experiment is a controlled S2C SAM-Segment Contrasting adaptation:

1. On Kaggle, run SAM ViT-B Segment-Everything on the 2,981 clean-train images
   only and save compact 80x80 region-index maps. Do not load polygons,
   validation, or test.
2. Train the same 320 px binary DenseNet121 recipe with BCE plus SSC, loss
   weight 1 and temperature 1 from the official S2C code. Augmentation remains
   off so the maps are spatially aligned.
3. Evaluate the frozen flip-TTA LayerCAM+SAM Gate-C recipe. Promotion requires
   positive paired-CI lower bound and no small-subgroup decrease.

This transfers SAM boundary/region knowledge into classifier features, directly
targeting the measured oracle-to-selection gap instead of adding another
inference heuristic.

## 28. Active S2C experiment

The AdvCAM/selector family is closed. The active direction is S2C
SAM-Segment Contrasting because the 448/AdvCAM diagnostics improved small-lesion
support recall and oracle candidates but lost the gain at feature
calibration/ranking.

- `itsthang333/btxrd-s2c-segment-everything-v1` is the heavy train-only
  preprocessing kernel. It must finish with 2,981 unique 80x80 `uint16` maps,
  exact split/SAM/source hashes, `polygons_or_masks_loaded=false`,
  `validation_images_processed=false`, and `test_evaluated=false`.
- Project source includes opt-in
  `project/models/sam_segment_contrastive.py` plus classifier CLI/integrity
  wiring. The frozen experiment is binary DenseNet121 320, BCE + SSC weight 1,
  temperature 1, no augmentation, identical optimizer/early stopping and
  clean-validation checkpoint rule.
- Source bundle v9 failed its smoke test before training because the cosine
  matrix needed one transpose before pixel-wise cross-entropy. Do not reuse or
  overwrite v9; publish the repaired source as the next immutable bundle.
  Smoke kernel `itsthang333/btxrd-s2c-ssc-torch-test-v1` must pass a real
  DenseNet optimizer step before full training.
- Repaired immutable source v10 passed 4/4 Torch tests and the faithful
  batch-8/320px/map-80 AMP DenseNet step. Compact result:
  `artifacts/kaggle/s2c_ssc_torch_test_v1/test_result.json`. The long classifier
  wrapper is prepared under `tmp/kaggle/s2c_binary_classifier_v1` and its
  static test must continue to fail until the completed region-manifest SHA
  replaces `__FILL_AFTER_S2C_PRECOMPUTE__`.
- Once preprocessing completes, download JSON/CSV/log first, verify the output,
  freeze `region_map_manifest.csv` SHA-256 in the classifier wrapper, then use
  the completed preprocessing kernel as an immutable Kaggle kernel source.
- After classifier training, run the unchanged promoted flip-TTA LayerCAM+SAM
  Gate-C recipe. Promotion requires a positive paired group-bootstrap lower
  bound over Dice `0.2343392222` and a non-negative delta on the fixed 94
  small-lesion cases. Do not evaluate test or train a downstream segmenter
  unless that rule passes.

Current status:

- The precompute is complete and audited: 2,981/2,981 unique train maps,
  80x80 `uint16`, map-manifest SHA-256
  `677dbb16ca0824a03bb75c991a01e6a83de5819d1db1f6f6d9f02765375c8f8e`.
  It did not load polygon/mask supervision, validation images, or test.
  Compact evidence is in
  `artifacts/kaggle/s2c_segment_everything_v1/`.
- The hash placeholder in `tmp/kaggle/s2c_binary_classifier_v1` has been
  frozen, 3/3 payload tests pass, and
  `itsthang333/btxrd-s2c-binary-classifier-v1` version 1 is running.
- The prepared `tmp/kaggle/s2c_gate_c_v1` payload already binds the region
  manifest and promoted baseline per-image hash, but intentionally retains
  `__FILL_AFTER_S2C_CLASSIFIER__`. Freeze the classifier checkpoint SHA only
  after the training kernel completes, then run its payload tests and push
  Gate C.

## 29. S2C final-feature result and stride-8 continuation

The first S2C classifier and Gate-C evaluation are complete.

Classifier:

- Kernel: `itsthang333/btxrd-s2c-binary-classifier-v1`, version 1, COMPLETE.
- Valid early stop at epoch 17; selected checkpoint epoch 10.
- Validation F1 `0.8201058201`, sensitivity `0.8423913043`, specificity
  `0.7914438503`, AUROC `0.8762206464`, AUPRC `0.8832401125`.
- Baseline F1 was `0.7833333333`; the SSC classifier improved F1 by
  `+0.0367724868`.
- Checkpoint SHA-256:
  `6c1562a4f6b2e789d71749c20fb9e8e3304e74c5f10da4f77cbd3657d820ee48`.
- All 371 rows, fixed confusion `155/39/29/148`, source/split/region hashes,
  polygon flags, and `test_evaluated=false` verify.

Gate C:

- Kernel: `itsthang333/btxrd-s2c-gate-c-v1`, version 1, COMPLETE.
- Candidate Dice `0.2334851256` versus promoted baseline `0.2343392222`.
- Paired delta `-0.0008540966`; tumor-group bootstrap 95% CI
  `-0.0312535770` to `+0.0291092193`.
- Small/medium/large deltas:
  `-0.0487671999/+0.0476879964/+0.0551904038`.
- Promotion is false. Compact evidence is under
  `artifacts/kaggle/s2c_binary_classifier_v1/` and
  `artifacts/kaggle/s2c_gate_c_v1/`. Test stayed locked.

Mechanism:

- Overall foreground recall, box recall, and SAM oracle Dice improve by
  `+0.138044/+0.169903/+0.012096`.
- On the fixed 94 small lesions, predicted area rises from `0.03105` to
  `0.07195`, precision falls `-0.066781`, point-hit rate falls `-0.017469`,
  oracle Dice falls `-0.024367`, selected Dice falls `-0.041551`, and
  post-processing costs another `-0.007216`.
- The first adaptation used final DenseNet features at 10x10/stride 32.
  Official S2C uses a high-resolution, stride-8 classifier feature map. At
  320 px the median small lesion is only about 11 px across, so final-grid
  interpolation cannot restore the missing spatial degrees of freedom.

Next controlled experiment:

- Move only the SSC feature tap to DenseNet `denseblock2`, which is
  `[B,512,40,40]`/stride 8 at 320 px.
- Add no projection layer, so the DenseNet state dictionary and inference
  architecture remain unchanged.
- Freeze the same 2,981 region maps, BCE/SSC weight and temperature,
  initialization, optimizer, split, early stopping, Gate-C recipe, metric, and
  promotion rule.
- Do not sweep SSC weight, CAM thresholds, or selectors. If the stride-8
  feature tap fails Gate C, close this SSC family and move to a separately
  predeclared CAM-logit supervision/CPM hypothesis.
- Source bundle v11 and the real T4 optimizer smoke test are complete. The
  latter passed 6/6 tests with a `[8,512,40,40]` tap, finite total loss and
  gradient norm, unchanged inference state dict, and no test evaluation.
  Compact evidence is
  `artifacts/kaggle/s2c_stride8_torch_test_v1/`.
- `itsthang333/btxrd-s2c-stride8-binary-classifier-v1` version 2 is the active
  Kaggle run. Version 2 changes only the fail-closed wrapper key from the
  generic projection name to the exact checkpoint key
  `feature_projection`; the scientific recipe is unchanged.
- Prepared Gate-C payload:
  `tmp/kaggle/s2c_stride8_gate_c_v1`. It intentionally fails the frozen-hash
  test until the completed classifier checkpoint SHA-256 is independently
  verified. Do not launch it early.

## 30. Stride-8 classifier audit and partition-oracle rejection

The stride-8 classifier kernel version 2 is COMPLETE. The wrapper-only
metadata fix is present and the scientific recipe is unchanged.

- Valid early stop: 19 completed epochs, checkpoint epoch 12.
- Validation F1 `0.7965616046`; fixed confusion
  `TP/FP/FN/TN = 139/26/45/161`.
- Relative to the clean BCE baseline, F1 is `+0.0132282713`, sensitivity
  `-0.0108695652`, specificity `+0.0481283422`, AUROC `-0.0041850732`, and
  AUPRC `+0.0043807906`.
- Checkpoint SHA-256:
  `8bb4a9136291bffba9d2e70752f5b03c003730ff5365ab7e1033f8dc222331d6`.
- The checkpoint/run metadata binds `denseblock2`, stride 8, 512 channels,
  post-ReLU, no projection, SSC weight/temperature 1, the 2,981 verified maps,
  and region-manifest SHA. Both polygon-loaded flags and
  `test_evaluated` are false.
- Compact evidence, excluding the 84 MB reconstructible checkpoint, is under
  `artifacts/kaggle/s2c_stride8_binary_classifier_v1/`.

A separate train-only CPU oracle rules out direct region selection over the
stored SSC maps. On 1,488 tumors the exact optimal subset of disjoint 80x80
partitions reaches mean Dice only `0.1126347` (best single `0.1112537`);
648 targets have zero best-single overlap. Small-lesion optimal-subset Dice is
`0.0759493`, with 355/752 zero-overlap cases and 11 targets disappearing at
80x80. Evidence is in `artifacts/kaggle/s2c_train_region_oracle_v1/`, row SHA
`3cc3d41d2cafd842361a382089270506e00fd55fec592708f3cd13a9b0b6fffa`.
Do not launch the prepared validation oracle: the geometric ceiling is already
decisive and validation GPU use would add no decision value. This result does
not test official online point-prompted CPM.

The frozen classifier hash is now inserted into
`tmp/kaggle/s2c_stride8_gate_c_v1`; its 6/6 fail-closed tests and Python
compilation pass. The next action is the unchanged Gate-C localization run.

## 31. Stride-8 Gate-C rejection and CPM transition

`itsthang333/btxrd-s2c-stride8-gate-c-v1` is COMPLETE and rejected.

- Candidate tumor-only Dice `0.1842404724`, promoted baseline
  `0.2343392222`.
- Paired delta `-0.0500987498`; complete-tumor-group bootstrap 95% CI
  `[-0.0818340900, -0.0195896270]`.
- Frozen small/medium/large subgroup deltas:
  `-0.0597196529/-0.0289336577/-0.0845166242`.
- All 371 rows, 184 tumors, 187 normals and 167 tumor groups verify.
  Checkpoint, region-map, baseline, split, SAM and 11 source hashes match;
  `test_evaluated=false`.
- Mechanism deltas versus the promoted baseline: support recall
  `+0.0093599`, support precision `-0.0121927`, point hit `-0.0514334`,
  SAM oracle `-0.0310159`, clipped oracle `-0.0325518`, selected Dice
  `-0.0499680`, final Dice `-0.0500988`.

Decision: close SSC tap/weight/temperature variations. The failure now affects
the proposal oracle and all three lesion-size groups, so neither another
selector nor a regularizer sweep is justified.

The next bounded direction is CAM-based Prompting Module supervision. The
prepared model is a BTXRD/DenseNet adaptation, not a copy of S2C's ResNet38:
it fuses projected `denseblock2` stride-8 detail with projected final
DenseNet semantics, creates a 256x40x40 feature, emits a differentiable
one-channel CAM, and obtains the image logit by global average pooling that
CAM. `project/models/s2c_cpm.py` contains the model and binary CPM loss.

Private T4 smoke kernel `itsthang333/btxrd-s2c-cpm-torch-test-v1` version 2
passed 3/3 tests and a real batch-2, 320px backward/optimizer step:

- feature `[2,256,40,40]`, CAM `[2,1,40,40]`;
- BCE/SSC/CPM/total `0.678304/4.011762/0.379352/5.069419`;
- finite pre-clip gradient norm `13.012649`;
- gradients reach both FPN projections and the CAM head;
- no test evaluation.

Before full training, create a train-tumor-only cache of frozen SAM ViT-B
image embeddings. With augmentation disabled, caching the encoder result is
mathematically equivalent to rerunning the frozen encoder each epoch; CPM
still recomputes CAM peaks and the SAM prompt decoder online. Bind the cache
to the clean split/image hashes and do not process validation/test images.

## 32. MAE normality probe rejection and nominal-memory transition

`itsthang333/btxrd-mae-normality-reconstruction-probe-v1` is COMPLETE and its
independent physical auditor passes.

- Source/wrapper/protocol/split hashes match the frozen declarations.
- All 371 base plus 371 adapted maps and the 447,670,680-byte adapted
  checkpoint were re-hashed; prediction freeze precedes every validation-GT
  read; cohort/subgroups are 371/184/187 and 94/72/18; complete misses are
  included and test is untouched.
- Normal-only adaptation is neutral on small lesions: AP delta
  `-0.00001345`, saliency-mass delta `-0.00008150`, p90-Dice delta
  `-0.00012162`, all CIs crossing zero; small argmax hit is `0/94`.
- Statistically coherent gains appear only in large lesions, including AP
  `+0.00897512` and p90 Dice `+0.01686208`, but absolute large p90 Dice is
  still only `0.13453951`.
- Reject MAE residuals as a standalone or automatically fused source; no
  threshold and no consumer are authorized.
- Compact evidence:
  `artifacts/kaggle/mae_normality_reconstruction_probe_val_v1/`.

The next experiment is a separately frozen, context-conditioned nominal patch
memory probe. Infrastructure in `project/models/nominal_patch_memory.py`
already provides deterministic global-context retrieval, global/spatial patch
distance, normal-only calibration, fixed fusion and hash-locked Gaussian
projection. It must be wired into a Kaggle prediction-first probe and pass a
new protocol/auditor before any pseudo-mask fusion.

The probe is now launched as
`itsthang333/btxrd-rad-dino-nominal-memory-probe-v1` version 1. Locked source
commit is `30f62d9475949dd43c9ad19c0590a0cbc854d440`, protocol SHA is
`458fee51...94a35`, wrapper SHA is `cfe2f5ef...c0662`, and RAD-DINO weight SHA
is `dbfb9f54...fb91ae`. One heartbeat,
`theo-d-i-rad-dino-nominal-memory-kaggle`, checks every five minutes. On
completion, download both 371-map arms plus compact memory evidence, run
`project/tools/audit_nominal_patch_memory_probe.py`, and only then interpret
small/medium/large results. No threshold or consumer is authorized by this
probe.

