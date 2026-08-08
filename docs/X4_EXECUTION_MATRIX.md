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
| `wanwin` | `btxrd-x4-rich-gallery-train-score` v2 | W3 reconstruct exact frozen gallery, then score 2,981 train images with locked G1 | Running |
| `wanwin` | `btxrd-x4-fully-student-seed42` | X1 matched fully-supervised U-Net, seed 42 | Running |
| `qwinwan` | `btxrd-x4-puzzlecam-student-seed44` | W1 PuzzleCAM-pseudo-mask matched U-Net student, seed 44 | Running |
| `qwinwan` | `btxrd-x4-fully-student-seed43` | X1 matched fully-supervised U-Net, seed 43 | Running |
| `itsthang333` | `btxrd-x4-cam-student-seed44` | W0 CAM-pseudo-mask matched U-Net student, seed 44 | Running |
| `itsthang333` | `btxrd-x4-puzzlecam-student-seed43` | W1 PuzzleCAM-pseudo-mask matched U-Net student, seed 43 | Running |

## Requirement ledger

| ID | Required evidence | Implementation/evidence | Status | Next blocking action |
|---|---|---|---|---|
| X1 | Direct Rich Gallery versus Rich-Gallery pseudo-U-Net versus matched fully-U-Net | Direct Rich Gallery validation evidence exists in the frozen final/G4 artifacts. Common student trainer, prediction freezer and native evaluator are implemented. One fail-closed offline wrapper now stages fully-supervised seeds 42/43/44 from identical bytes and infers the seed only from the kernel filename; 13/13 focused matched-training tests pass. | Partial | Freeze Rich Gallery train targets; launch the staged fully-supervised kernels when slots free; evaluate both arms on the same native grid. |
| X2 | Known binary label, binary predicted gate, ten-class predicted gate and label-free student | `project/freeze_x4_gate_predictions.py` freezes all four 371-image arms before GT. It binds the audited G4 binary/ten-class probabilities at threshold 0.5, the frozen direct Rich-Gallery choices and the matched Rich-Gallery student. `project/evaluate_x4_gate_predictions.py` reports the common native-grid tumor/subgroup and normal-image endpoints. | Implemented | Run seeds 42/43/44 after the Rich-Gallery student prediction freezes exist. |
| W0 | CAM-to-U-Net | CAM train/validation mask freezes and audits are complete. The standardized 2,981-image target bundle is frozen. Seeds 42 and 43 completed; seed 43 training audit passed with checkpoint `cd3e4d82...0cb8`, best epoch 8 and inner-only threshold 0.65. Seed 44 is running. | Running | Audit seed 42, then freeze/evaluate 371 predictions for seeds 42/43 while seed 44 runs. |
| W1 | PuzzleCAM-to-U-Net | PuzzleCAM generator training and train/validation mask audits are complete. The standardized target bundle is frozen. Seed 42 completed; seeds 43/44 are running. | Running | Download/audit seed 42, then freeze/evaluate all completed predictions. |
| W2 | S2C-to-U-Net | Generator/checkpoint/cache provenance passed, and both mask archives plus the common target bundle completed. Independent native-mask audit exposed an efficacy failure: only 1/1,488 tumor train images contains any selected S2C segment (432 foreground pixels), while 0/184 tumor validation images contains one; all 1,493/187 normal masks are correctly empty. This is not a technical/audit failure. | Target complete; efficacy failure established | Preserve the required matched-student arm for X4 comparability, but report the near-empty supervision explicitly and avoid interpreting it as a competitive pseudo-label source. |
| W3 | Rich-Gallery+G1+R7-to-U-Net | Version 1 failed before scoring because the old Geometry-v3 output retained the G1 checkpoint but deleted its temporary merged gallery. The root-cause fix reconstructs the exact train gallery from the immutable `anchor` and `classifier448` source kernels and requires the merged manifest/pseudo binding to equal the published Geometry-v3 hashes before scoring. Train-score v2 is running; target-freeze reconstructs the same gallery independently. Combined merge/wrapper/target regression is 10/10 pass. | Running | Audit train-score v2, immediately chain the CPU target-freeze job, then train/evaluate three matched students. |
| X3 | YOLOv8s-seg upper bound, 600 px, 300 epochs, native mAP and common binary-union evaluation | `project/export_x4_yolo_dataset.py` exports the exact canonical train/val polygons in Ultralytics instance-seg format; `project/train_x4_yolov8s_seg.py` fixes official YOLOv8s-seg, Ultralytics 8.4.0, 600 px and 300 epochs; Stage A freezes native union masks and the separate evaluator reports native mAP plus all common X4 metrics. The minimal offline payload contains 17 exact files, binds source commit `4124fb2`, and passed server download-back verification with manifest SHA `b04df84cff383b55d1c26c59f309842894cb5c4171285cb50bc60b72dde840a4`. | Implemented and staged, not launched | Launch seed 42 after the P0 student critical path has occupied all reusable slots; add seeds 43/44 only if budget permits. |
| X4 | Normal-image specificity/FPR, predicted area distribution/threshold rates, false components and examples | `project/evaluation/segmentation_metrics.py` implements all numeric endpoints; the student evaluator uses all 187 normal validation images. | Implemented | Produce results after each prediction freeze; add protocol-selected normal examples under X10. |
| X5 | 8-neighbour lesion/multifocal metrics and one-to-one matching at IoU 0.10/0.25/0.50 | Implemented in `project/evaluation/segmentation_metrics.py`, including precision/recall/F1, missed/excess lesions, partial multifocal miss and component-count error. Exact fixtures now verify the expected 2/1/1 matches and F1 1.0/0.5/0.5 at the three thresholds. | Implemented and tested | Run on every arm/seed after each prediction freeze. |
| X6 | Macro/micro overlap, pixel precision/recall, extent, HD95/ASSD, empty and zero-overlap cases | Implemented in the common native-grid evaluator. HD95/ASSD are explicitly pixels and conditional-defined counts are reported. | Implemented | Run on every arm/seed and aggregate the declared subgroup tables. |
| X7 | Seeds 42/43/44 for CAM, PuzzleCAM, S2C, Rich Gallery and fully supervised | Seed contract is frozen and common trainer supports all arms/seeds. CAM 42/43 and PuzzleCAM 42 completed; CAM44 and PuzzleCAM43/44 run now. Fully-supervised seeds 42/43 also run; seed44 is staged. | Running | Complete 15 matched student runs and report mean plus sample SD. |
| X8 | 10,000 paired heuristic-group bootstrap replicates for five frozen contrasts | `project/summarize_x4_student_study.py` enforces the complete 15-run matrix, aggregates all seeds and executes the five frozen paired heuristic-group contrasts; deterministic/fail-closed regression coverage exists. | Implemented, awaiting inputs | Execute after all required per-image tables exist. |
| X9 | Ten-class failure taxonomy | `project/analyze_x4_error_taxonomy.py` implements all ten X4 classes as non-exclusive flags plus a deterministic primary label. The direct cap-243 baseline result is frozen for 371 images; it records 29 supply failures, 78 selector failures, 48 complete misses, 93 over-segmented and 31 under-segmented tumor cases. | Partial | Re-run the same fixed rules with the matched student tables after their prediction evaluations complete. |
| X10 | Protocol-selected qualitative panel | `project/select_x4_qualitative_cases.py` freezes cases before image/GT rendering using Dice quantiles, size groups and deterministic failure-category rules. `project/render_x4_qualitative_panels.py` verifies that freeze plus the direct choices/candidate payloads, then renders X-ray, localization, proposal gallery, selected/oracle, five optional model outputs and GT. The direct freeze supplies 11/12 categories without visual cherry-picking. | Implemented, awaiting model outputs | Add the normal-FP case after predicted-gate/student evaluation and rerun the renderer with all five audited prediction bundles. |
| X11 | Risk-coverage using frozen G1/fusion confidence | `project/analyze_x4_risk_coverage.py` and the frozen result under `artifacts/final_pipeline/x4/x11_risk_coverage` are complete. Spearman confidence-vs-Dice is 0.313141; Dice<0.10 and complete-miss AUROC are 0.658570/0.656916. Mean Dice at 100/80/60/40% coverage is 0.288729/0.335050/0.361465/0.367326. | Complete | Interpret as moderate failure-detection evidence, not calibrated uncertainty or a new routing rule. |
| X12 | Runtime, peak VRAM, storage; offline separated from online | X4 student Stage A now performs three warm-up forwards and measures all 371 images, emitting median/IQR latency, elapsed time, per-device peak allocated/reserved VRAM and storage while explicitly excluding offline pseudo generation. Generator stages already retain separate runtime/device/storage metadata. | Implemented | Aggregate the emitted same-GPU measurements after prediction freezes; add direct-pipeline and YOLO rows when their common benchmarks are available. |

## P0/P1 readiness

| Priority | Requirement | Current state |
|---|---|---|
| P0 | Rich-Gallery pseudo-U-Net, matched fully, CAM, S2C, X2, normal/lesion/subgroup, three seeds, bootstrap/CI, error analysis | In progress; common metrics and X2 code are ready, but student runs, X2 result execution and taxonomy remain incomplete. |
| P1 | PuzzleCAM, YOLO, efficiency, qualitative panel, risk-coverage | PuzzleCAM is running and risk-coverage is complete; YOLO, efficiency and qualitative panel remain. |

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
