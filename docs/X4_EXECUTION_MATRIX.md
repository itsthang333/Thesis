# X4 execution matrix

This file is the requirement-by-requirement execution ledger for `X4.pdf`.
`Complete` means that both an implementation and a result artifact exist.
`Implemented` means that the code is ready but the required run/result is not
yet complete.  Validation is the canonical 371-image outer cohort; test remains
locked throughout X4 development.

Last updated: 2026-08-09.

## Current GPU allocation

| Account | Kernel | Purpose | State at update |
|---|---|---|---|
| `wanwin` | `btxrd-x4-rich-gallery-train-score` v3 | W3 reconstruct gallery once, score 2,981 train images with locked G1 and freeze targets inline | Running |
| `wanwin` | `btxrd-x4-fully-student-seed42` v2 | X1 matched fully-supervised U-Net, seed 42 | Running after pre-data wrapper fix |
| `qwinwan` | `btxrd-x4-puzzlecam-student-seed44` | W1 PuzzleCAM-pseudo-mask matched U-Net student, seed 44 | Running |
| `qwinwan` | `btxrd-x4-fully-student-seed43` v2 | X1 matched fully-supervised U-Net, seed 43 | Running after pre-data wrapper fix |
| `itsthang333` | `btxrd-x4-fully-student-seed44` v1 | X1 matched fully-supervised U-Net, seed 44 | Running |
| `itsthang333` | `btxrd-x4-s2c-student-seed42` v1 | W2 S2C-pseudo-mask matched U-Net, seed 42 | Running; required ablation despite near-empty S2C targets |

## Audited native-resolution validation results available

All prediction archives below were frozen before any outer-validation spatial
annotation was opened.  The common Stage-B evaluator then opened exactly 184
tumor polygons and reported the same native-resolution endpoints.

| Arm | Seed | Mean tumor Dice | Mean tumor IoU | `<1%` | `1–<5%` | `>=5%` | Normal FP case rate | Macro predicted/GT area ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAM-to-U-Net | 42 | 0.141918 | 0.088047 | 0.020283 | 0.249093 | 0.348419 | 0.550802 | 90.9166 |
| CAM-to-U-Net | 43 | 0.123913 | 0.073606 | 0.017688 | 0.210529 | 0.332185 | 0.679144 | 92.0087 |
| CAM-to-U-Net | 44 | 0.142643 | 0.087459 | 0.029516 | 0.242359 | 0.334548 | 0.572193 | 83.9119 |
| PuzzleCAM-to-U-Net | 42 | 0.116542 | 0.070714 | 0.013698 | 0.188556 | 0.365558 | 0.978610 | 90.0120 |
| PuzzleCAM-to-U-Net | 43 | 0.116134 | 0.069532 | 0.016581 | 0.185026 | 0.360460 | 0.983957 | 98.6048 |

These results establish an efficacy failure, not an execution failure: all five
students strongly over-segment, especially in the `<1%` subgroup.  PuzzleCAM
also predicts a non-empty mask on almost every normal image.  Direct Rich
Gallery remains the current validation baseline at Dice 0.2887294867.

## Requirement ledger

| ID | Required evidence | Implementation/evidence | Status | Next blocking action |
|---|---|---|---|---|
| X1 | Direct Rich Gallery versus Rich-Gallery pseudo-U-Net versus matched fully-U-Net | Direct Rich Gallery validation evidence exists in the frozen final/G4 artifacts. Common student trainer, prediction freezer and native evaluator are implemented. Kaggle renames every Python entrypoint to `script.py`, so the fully-supervised wrapper now binds seed 42/43/44 explicitly in each staged payload; the exact failure is covered by 10/10 focused tests. All three fully-supervised seeds are running. | Partial | Freeze Rich Gallery targets inline with G1 scoring; evaluate both arms on the same native grid. |
| X2 | Known binary label, binary predicted gate, ten-class predicted gate and label-free student | `project/freeze_x4_gate_predictions.py` freezes all four 371-image arms before GT. It binds the audited G4 binary/ten-class probabilities at threshold 0.5, the frozen direct Rich-Gallery choices and the matched Rich-Gallery student. `project/evaluate_x4_gate_predictions.py` reports the common native-grid tumor/subgroup and normal-image endpoints. | Implemented | Run seeds 42/43/44 after the Rich-Gallery student prediction freezes exist. |
| W0 | CAM-to-U-Net | CAM train/validation mask freezes and audits are complete. The standardized 2,981-image target bundle is frozen. Seeds 42, 43 and 44 completed and passed training-output checks. All three 371-image prediction freezes passed archive/freeze SHA verification and common Stage-B evaluation. Dice is 0.141918/0.123913/0.142643 for seeds 42/43/44. | Complete | Aggregate mean/sample SD and include the over-segmentation/normal-FP evidence in X8/X9. |
| W1 | PuzzleCAM-to-U-Net | PuzzleCAM generator training and train/validation mask audits are complete. The standardized target bundle is frozen. Seeds 42 and 43 passed training and prediction-freeze checks; Stage-B Dice is 0.116542/0.116134. Seed 44 is running. | Running | Audit/freeze/evaluate seed 44, then aggregate all three seeds. |
| W2 | S2C-to-U-Net | Generator/checkpoint/cache provenance passed, and both mask archives plus the common target bundle completed. Independent native-mask audit exposed an efficacy failure: only 1/1,488 tumor train images contains any selected S2C segment (432 foreground pixels), while 0/184 tumor validation images contains one; all 1,493/187 normal masks are correctly empty. Seed-42 matched student is now running. | Running required ablation | Complete seeds 42/43/44 for protocol comparability, while reporting the near-empty supervision explicitly. |
| W3 | Rich-Gallery+G1+R7-to-U-Net | V1 exposed the deleted temporary train gallery. V2 reconstructed exactly the same semantic supply (468,315 masks; identical per-source/duplicate totals) but its manifest byte SHA differed because `np.savez_compressed` embeds ZIP timestamps. V3 verifies the immutable source hashes and full semantic totals, then scores G1 and freezes targets inline against the same reconstruction, avoiding a false byte-hash failure and a second 26-minute merge. Wrapper/target tests pass 12/12. | Running v3 | Audit both score and inline target receipts, then train/evaluate three matched students. |
| X3 | YOLOv8s-seg upper bound, 600 px, 300 epochs, native mAP and common binary-union evaluation | `project/export_x4_yolo_dataset.py` exports the exact canonical train/val polygons in Ultralytics instance-seg format; `project/train_x4_yolov8s_seg.py` fixes official YOLOv8s-seg, Ultralytics 8.4.0, 600 px and 300 epochs; Stage A freezes native union masks and the separate evaluator reports native mAP plus all common X4 metrics. The minimal offline payload contains 17 exact files, binds source commit `4124fb2`, and passed server download-back verification with manifest SHA `b04df84cff383b55d1c26c59f309842894cb5c4171285cb50bc60b72dde840a4`. Seed-42 v1 and unchanged v2 on `itsthang333` both failed before producing a worker log or output artifact, so the failure has not yet localized to model code. | Implemented; runtime unresolved | Retry once on a different account/worker after a P0 slot frees; do not repeat blindly on the same account. |
| X4 | Normal-image specificity/FPR, predicted area distribution/threshold rates, false components and examples | `project/evaluation/segmentation_metrics.py` implements all numeric endpoints; the student evaluator uses all 187 normal validation images. | Implemented | Produce results after each prediction freeze; add protocol-selected normal examples under X10. |
| X5 | 8-neighbour lesion/multifocal metrics and one-to-one matching at IoU 0.10/0.25/0.50 | Implemented in `project/evaluation/segmentation_metrics.py`, including precision/recall/F1, missed/excess lesions, partial multifocal miss and component-count error. Exact fixtures now verify the expected 2/1/1 matches and F1 1.0/0.5/0.5 at the three thresholds. | Implemented and tested | Run on every arm/seed after each prediction freeze. |
| X6 | Macro/micro overlap, pixel precision/recall, extent, HD95/ASSD, empty and zero-overlap cases | Implemented in the common native-grid evaluator. HD95/ASSD are explicitly pixels and conditional-defined counts are reported. | Implemented | Run on every arm/seed and aggregate the declared subgroup tables. |
| X7 | Seeds 42/43/44 for CAM, PuzzleCAM, S2C, Rich Gallery and fully supervised | Seed contract is frozen and common trainer supports all arms/seeds. CAM 42/43/44 is fully evaluated; PuzzleCAM 42/43 is fully evaluated and seed44 runs; S2C42 runs; fully-supervised 42/43/44 run after explicit per-payload seed binding fixed Kaggle's `script.py` rename. | Running | Complete the remaining matched students and report mean plus sample SD. |
| X8 | 10,000 paired heuristic-group bootstrap replicates for five frozen contrasts | `project/summarize_x4_student_study.py` enforces the complete 15-run matrix, aggregates all seeds and executes the five frozen paired heuristic-group contrasts; deterministic/fail-closed regression coverage exists. | Implemented, awaiting inputs | Execute after all required per-image tables exist. |
| X9 | Ten-class failure taxonomy | `project/analyze_x4_error_taxonomy.py` implements all ten X4 classes as non-exclusive flags plus a deterministic primary label. The direct cap-243 baseline result is frozen for 371 images; it records 29 supply failures, 78 selector failures, 48 complete misses, 93 over-segmented and 31 under-segmented tumor cases. | Partial | Re-run the same fixed rules with the matched student tables after their prediction evaluations complete. |
| X10 | Protocol-selected qualitative panel | `project/select_x4_qualitative_cases.py` freezes cases before image/GT rendering using Dice quantiles, size groups and deterministic failure-category rules. `project/render_x4_qualitative_panels.py` verifies that freeze plus the direct choices/candidate payloads, then renders X-ray, localization, proposal gallery, selected/oracle, five optional model outputs and GT. The direct freeze supplies 11/12 categories without visual cherry-picking. | Implemented, awaiting model outputs | Add the normal-FP case after predicted-gate/student evaluation and rerun the renderer with all five audited prediction bundles. |
| X11 | Risk-coverage using frozen G1/fusion confidence | `project/analyze_x4_risk_coverage.py` and the frozen result under `artifacts/final_pipeline/x4/x11_risk_coverage` are complete. Spearman confidence-vs-Dice is 0.313141; Dice<0.10 and complete-miss AUROC are 0.658570/0.656916. Mean Dice at 100/80/60/40% coverage is 0.288729/0.335050/0.361465/0.367326. | Complete | Interpret as moderate failure-detection evidence, not calibrated uncertainty or a new routing rule. |
| X12 | Runtime, peak VRAM, storage; offline separated from online | X4 student Stage A now performs three warm-up forwards and measures all 371 images, emitting median/IQR latency, elapsed time, per-device peak allocated/reserved VRAM and storage while explicitly excluding offline pseudo generation. Generator stages already retain separate runtime/device/storage metadata. | Implemented | Aggregate the emitted same-GPU measurements after prediction freezes; add direct-pipeline and YOLO rows when their common benchmarks are available. |

## P0/P1 readiness

| Priority | Requirement | Current state |
|---|---|---|
| P0 | Rich-Gallery pseudo-U-Net, matched fully, CAM, S2C, X2, normal/lesion/subgroup, three seeds, bootstrap/CI, error analysis | In progress; common metrics and X2 code are ready, but student runs, X2 result execution and taxonomy remain incomplete. |
| P1 | PuzzleCAM, YOLO, efficiency, qualitative panel, risk-coverage | PuzzleCAM 42/43 is evaluated and seed44 runs; risk-coverage is complete; YOLO worker startup, efficiency aggregation and the final qualitative panel remain. |

## Slot-reuse order

1. A completed CAM/PuzzleCAM-pseudo-mask student slot is reused for the next missing seed
   of the same already-frozen target arm.
2. The S2C generator slot first freezes/audits S2C masks and then launches the
   S2C student seeds.
3. Once W0-W2 seed coverage is no longer the critical path, slots launch Rich
   Gallery and fully-supervised matched students in pairs.
4. YOLO is launched only after the P0 student critical path is fully occupied;
   CPU/local work continues on X2 and X8-X12 without consuming a GPU slot.

No run may use outer-validation polygons for training, checkpoint selection or
threshold selection, and no X4 development run may read test images.
