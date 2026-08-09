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
| `wanwin` | `btxrd-x4-rich-gallery-student-seed42` v1 | W3 Rich-Gallery-pseudo-mask matched U-Net, seed 42 | Running |
| `wanwin` | `btxrd-x4-yolov8s-seg-seed42` v4 | X3 YOLOv8s-seg seed42 after batch-two/background-only/writable-copy fixes | Running |
| `qwinwan` | `btxrd-x4-yolov8s-seg-seed44` v1 | X3 YOLOv8s-seg seed44 with the audited v4 compatibility source | Running |
| `qwinwan` | `btxrd-x4-rich-gallery-seed43-prediction-freeze` v1 | W3 freeze 371 Rich-student seed43 predictions before spatial GT | Running |
| `itsthang333` | `btxrd-x4-rich-gallery-seed44-prediction-freeze` v1 | W3 freeze 371 Rich-student seed44 predictions before spatial GT | Running |
| `itsthang333` | `btxrd-x4-yolov8s-seg-seed43` v4 | X3 YOLOv8s-seg seed43 after batch-two/background-only/writable-copy fixes | Running |

The two YOLO v2 jobs stopped after a deterministic Ultralytics
8.4.0 compatibility failure on a final batch of size two.  The upstream loss
code mistakes a four-dimensional tensor whose batch dimension is two for a
two-item tuple, unpacks it to three dimensions and then fails.  The exact
type-guard compatibility patch has 6/6 focused tests passing; no scientific
setting is changed.  V3 then proved that fix by reaching epoch two, where it
exposed a separate upstream background-only-batch bug (`None * 0`) and showed
that read-only symlinks caused Ultralytics to reject 11 train and 3 validation
JPEGs that it attempted to repair.  V4 conditionally applies the unused-
gradient term only when a semantic head exists and materializes writable byte
copies without changing source images.  Seven focused tests pass; rebuilt
private payloads and server-returned runner/exporter hashes match on all three
accounts.  Corrected seed42/43 v4 jobs are running.  Seed44 was launched on
`qwinwan` after its S2C and Rich-student training slots completed.

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
| PuzzleCAM-to-U-Net | 44 | 0.116389 | 0.071055 | 0.012323 | 0.187477 | 0.375487 | 0.855615 | 88.8728 |
| S2C-to-U-Net | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| S2C-to-U-Net | 44 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| Fully supervised U-Net | 42 | 0.456580 | 0.367609 | 0.286350 | 0.624306 | 0.674657 | 0.288770 | 4.2848 |
| Fully supervised U-Net | 43 | 0.456626 | 0.363456 | 0.293991 | 0.607704 | 0.701623 | 0.406417 | 5.3053 |
| Fully supervised U-Net | 44 | 0.449361 | 0.356357 | 0.296252 | 0.598555 | 0.652155 | 0.294118 | 4.5662 |

These results establish an efficacy failure, not an execution failure: all five
CAM and PuzzleCAM students strongly over-segment, especially in the `<1%`
subgroup, and PuzzleCAM predicts a non-empty mask on almost every normal image.
In contrast, both evaluated S2C students collapse to the all-background
solution: 184/184 tumor predictions are empty.  This is the expected downstream
consequence of the audited S2C targets (only 1/1,488 tumor training targets has
foreground), not an evaluator failure.  Direct Rich Gallery remains the current
validation baseline at Dice 0.2887294867.

## Requirement ledger

| ID | Required evidence | Implementation/evidence | Status | Next blocking action |
|---|---|---|---|---|
| X1 | Direct Rich Gallery versus Rich-Gallery pseudo-U-Net versus matched fully-U-Net | Direct Rich Gallery validation evidence exists in the frozen final/G4 artifacts. All three fully-supervised trainings, freezes and evaluations passed with Dice 0.456580/0.456626/0.449361. Rich target audit passed; Rich seeds43/44 training-output audits passed and their prediction freezes run, while seed42 still trains. | Partial | Freeze/evaluate all three Rich students on the same native grid. |
| X2 | Known binary label, binary predicted gate, ten-class predicted gate and label-free student | `project/freeze_x4_gate_predictions.py` freezes all four 371-image arms before GT. It binds the audited G4 binary/ten-class probabilities at threshold 0.5, the frozen direct Rich-Gallery choices and the matched Rich-Gallery student. `project/evaluate_x4_gate_predictions.py` reports the common native-grid tumor/subgroup and normal-image endpoints. | Implemented | Run seeds 42/43/44 after the Rich-Gallery student prediction freezes exist. |
| W0 | CAM-to-U-Net | CAM train/validation mask freezes and audits are complete. The standardized 2,981-image target bundle is frozen. Seeds 42, 43 and 44 completed and passed training-output checks. All three 371-image prediction freezes passed archive/freeze SHA verification and common Stage-B evaluation. Dice is 0.141918/0.123913/0.142643 for seeds 42/43/44. | Complete | Aggregate mean/sample SD and include the over-segmentation/normal-FP evidence in X8/X9. |
| W1 | PuzzleCAM-to-U-Net | PuzzleCAM generator training and train/validation mask audits are complete. The standardized target bundle is frozen. Seeds 42/43/44 passed training, prediction-freeze and Stage-B checks; Dice is 0.116542/0.116134/0.116389 (mean 0.116355, sample SD 0.000206). | Complete | Include the stable over-segmentation and normal-FP failure in X8/X9. |
| W2 | S2C-to-U-Net | Generator/checkpoint/cache provenance passed, and both mask archives plus the common target bundle completed. Independent native-mask audit exposed an efficacy failure: only 1/1,488 tumor train images contains any selected S2C segment (432 foreground pixels), while 0/184 tumor validation images contains one; all 1,493/187 normal masks are correctly empty. Seeds42/44 produce empty masks and Dice/IoU 0. Seed43 training completed and its independent 30-epoch/checkpoint/provenance audit passed (best checkpoint SHA `fc82c1d8...`). | Three trainings audited; two seeds evaluated | Freeze/evaluate seed43 to close the required three-seed ablation, then report the deterministic all-background collapse. |
| W3 | Rich-Gallery+G1+R7-to-U-Net | V3 reconstructed the exact semantic supply, scored all 2,981 train images with locked G1 and froze targets inline. Both archives matched receipt SHA. Independent target validation passed exact canonical IDs/hashes: 2,981 targets, 1,488/1,488 tumor targets non-empty, zero outer-validation annotations and zero test reads. Seeds43/44 completed all 30 epochs and independent training audits passed; their 371-image prediction freezes run. Seed42 remains in training. | Two trainings audited; prediction freezes running | Complete seed42, then audit/freeze/evaluate all three Rich students. |
| X3 | YOLOv8s-seg upper bound, 600 px, 300 epochs, native mAP and common binary-union evaluation | Export, fixed protocol and Stage-A/B code are implemented. V2 exposed the batch-size-two tensor/tuple bug; V3 passed it and exposed the independent background-only `None` dereference plus read-only JPEG-repair exclusions. V4 fixes those compatibility faults without changing the scientific protocol; 7/7 focused tests and three-account server source hashes pass. | Seeds42/43 v4 and seed44 v1 running | Confirm zero canonical image exclusions and passage beyond epoch two, then audit/freeze/evaluate all three seeds. |
| X4 | Normal-image specificity/FPR, predicted area distribution/threshold rates, false components and examples | `project/evaluation/segmentation_metrics.py` implements all numeric endpoints; the student evaluator uses all 187 normal validation images. | Implemented | Produce results after each prediction freeze; add protocol-selected normal examples under X10. |
| X5 | 8-neighbour lesion/multifocal metrics and one-to-one matching at IoU 0.10/0.25/0.50 | Implemented in `project/evaluation/segmentation_metrics.py`, including precision/recall/F1, missed/excess lesions, partial multifocal miss and component-count error. Exact fixtures now verify the expected 2/1/1 matches and F1 1.0/0.5/0.5 at the three thresholds. | Implemented and tested | Run on every arm/seed after each prediction freeze. |
| X6 | Macro/micro overlap, pixel precision/recall, extent, HD95/ASSD, empty and zero-overlap cases | Implemented in the common native-grid evaluator. HD95/ASSD are explicitly pixels and conditional-defined counts are reported. | Implemented | Run on every arm/seed and aggregate the declared subgroup tables. |
| X7 | Seeds 42/43/44 for CAM, PuzzleCAM, S2C, Rich Gallery and fully supervised | 11/15 required seed results are evaluated: CAM 3/3, PuzzleCAM 3/3 and Fully 3/3; S2C is 2/3 evaluated with seed43 training audited; Rich is 0/3 evaluated, with seeds43/44 training audited and prediction freezes running while seed42 trains. | Running/evaluation | Freeze/evaluate S2C43 and all three Rich seeds, then report the complete mean plus sample SD table. |
| X8 | 10,000 paired heuristic-group bootstrap replicates for five frozen contrasts | `project/summarize_x4_student_study.py` enforces the complete 15-run matrix, aggregates all seeds and executes the five frozen paired heuristic-group contrasts; deterministic/fail-closed regression coverage exists. | Implemented, awaiting inputs | Execute after all required per-image tables exist. |
| X9 | Ten-class failure taxonomy | `project/analyze_x4_error_taxonomy.py` implements all ten X4 classes as non-exclusive flags plus a deterministic primary label. The direct cap-243 baseline result is frozen for 371 images; it records 29 supply failures, 78 selector failures, 48 complete misses, 93 over-segmented and 31 under-segmented tumor cases. | Partial | Re-run the same fixed rules with the matched student tables after their prediction evaluations complete. |
| X10 | Protocol-selected qualitative panel | `project/select_x4_qualitative_cases.py` freezes cases before image/GT rendering using Dice quantiles, size groups and deterministic failure-category rules. `project/render_x4_qualitative_panels.py` verifies that freeze plus the direct choices/candidate payloads, then renders X-ray, localization, proposal gallery, selected/oracle, five optional model outputs and GT. The direct freeze supplies 11/12 categories without visual cherry-picking. | Implemented, awaiting model outputs | Add the normal-FP case after predicted-gate/student evaluation and rerun the renderer with all five audited prediction bundles. |
| X11 | Risk-coverage using frozen G1/fusion confidence | `project/analyze_x4_risk_coverage.py` and the frozen result under `artifacts/final_pipeline/x4/x11_risk_coverage` are complete. Spearman confidence-vs-Dice is 0.313141; Dice<0.10 and complete-miss AUROC are 0.658570/0.656916. Mean Dice at 100/80/60/40% coverage is 0.288729/0.335050/0.361465/0.367326. | Complete | Interpret as moderate failure-detection evidence, not calibrated uncertainty or a new routing rule. |
| X12 | Runtime, peak VRAM, storage; offline separated from online | X4 student Stage A now performs three warm-up forwards and measures all 371 images, emitting median/IQR latency, elapsed time, per-device peak allocated/reserved VRAM and storage while explicitly excluding offline pseudo generation. Generator stages already retain separate runtime/device/storage metadata. | Implemented | Aggregate the emitted same-GPU measurements after prediction freezes; add direct-pipeline and YOLO rows when their common benchmarks are available. |

## P0/P1 readiness

| Priority | Requirement | Current state |
|---|---|---|
| P0 | Rich-Gallery pseudo-U-Net, matched fully, CAM, S2C, X2, normal/lesion/subgroup, three seeds, bootstrap/CI, error analysis | In progress; Fully and CAM are 3/3 complete, S2C is 2/3, while Rich, X2 execution and matched taxonomy remain incomplete. |
| P1 | PuzzleCAM, YOLO, efficiency, qualitative panel, risk-coverage | PuzzleCAM is 3/3 evaluated and risk-coverage is complete; YOLO compatibility reruns, efficiency aggregation and the final qualitative panel remain. |

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
