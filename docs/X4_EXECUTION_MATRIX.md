# X4 execution matrix

This file is the requirement-by-requirement execution ledger for `X4.pdf`.
`Complete` means that both an implementation and a result artifact exist.
`Implemented` means that the code is ready but the required run/result is not
yet complete.  Validation is the canonical 371-image outer cohort; test remains
locked throughout X4 development.

Last updated: 2026-08-08.

## Current GPU allocation

| Account | Kernel | Purpose | State at update |
|---|---|---|---|
| `wanwin` | `btxrd-x4-s2c-generator-training` | W2 S2C generator | Running |
| `wanwin` | `btxrd-x4-cam-student-seed42` | W0 CAM student, seed 42 | Running |
| `qwinwan` | `btxrd-x4-cam-student-seed43` | W0 CAM student, seed 43 | Running |
| `qwinwan` | `btxrd-x4-puzzlecam-student-seed42` | W1 PuzzleCAM student, seed 42 | Running |
| `itsthang333` | `btxrd-x4-cam-student-seed44` | W0 CAM student, seed 44 | Running |
| `itsthang333` | `btxrd-x4-puzzlecam-student-seed43` | W1 PuzzleCAM student, seed 43 | Running |

## Requirement ledger

| ID | Required evidence | Implementation/evidence | Status | Next blocking action |
|---|---|---|---|---|
| X1 | Direct Rich Gallery versus Rich-Gallery pseudo-U-Net versus matched fully-U-Net | Direct Rich Gallery validation evidence exists in the frozen final/G4 artifacts. Common student trainer, prediction freezer and native evaluator are implemented. | Partial | Freeze Rich Gallery train targets; train Rich Gallery and fully-supervised seeds 42/43/44; evaluate all on the same native grid. |
| X2 | Known binary label, binary predicted gate, ten-class predicted gate and label-free student | Gate names are frozen in `project/x4_contract.py`. G4 contains the matched binary/ten-class classifiers and downstream evidence. | Partial | Implement the four-arm deployment evaluator using frozen predictions; report actual mask Dice and normal-image behavior. |
| W0 | CAM-to-U-Net | CAM train/validation mask freezes and audits are complete. The standardized 2,981-image target bundle is frozen. Seeds 42/43/44 are running across three accounts. | Running | Audit completed students, then freeze/evaluate 371 predictions for all seeds. |
| W1 | PuzzleCAM-to-U-Net | PuzzleCAM generator training and train/validation mask audits are complete. The standardized target bundle is frozen. Seeds 42/43 are running. | Running | Audit completed students, launch seed 44 in the first reusable slot, then freeze/evaluate all predictions. |
| W2 | S2C-to-U-Net | Canonical train/validation S2C cache adoption audits pass. Generator training is running. Mask freezer and auditors are implemented. | Running | Audit generator; freeze/audit train and validation masks; freeze standardized target; train/evaluate three seeds. |
| W3 | Rich-Gallery+G1+R7-to-U-Net | Direct frozen Rich Gallery/G1/R7 choices and candidate diagnostics exist. The common target freezer accepts `rich_gallery`. | Implemented | Materialize and audit the 2,981-image Rich Gallery target bundle, then train/evaluate three matched students. |
| X3 | YOLOv8s-seg upper bound, 600 px, 300 epochs, native mAP and common binary-union evaluation | Protocol requirement documented only. | Missing | Implement canonical train-only YOLO export/training wrapper and a common validation-mask export/evaluator; run at least seed 42. |
| X4 | Normal-image specificity/FPR, predicted area distribution/threshold rates, false components and examples | `project/evaluation/segmentation_metrics.py` implements all numeric endpoints; the student evaluator uses all 187 normal validation images. | Implemented | Produce results after each prediction freeze; add protocol-selected normal examples under X10. |
| X5 | 8-neighbour lesion/multifocal metrics and one-to-one matching at IoU 0.10/0.25/0.50 | Implemented in `project/evaluation/segmentation_metrics.py`, including precision/recall/F1, missed/excess lesions, partial multifocal miss and component-count error. | Implemented | Run on every arm/seed and add formula regression fixtures for all three one-to-one thresholds. |
| X6 | Macro/micro overlap, pixel precision/recall, extent, HD95/ASSD, empty and zero-overlap cases | Implemented in the common native-grid evaluator. HD95/ASSD are explicitly pixels and conditional-defined counts are reported. | Implemented | Run on every arm/seed and aggregate the declared subgroup tables. |
| X7 | Seeds 42/43/44 for CAM, PuzzleCAM, S2C, Rich Gallery and fully supervised | Seed contract is frozen and common trainer supports all arms/seeds. | Running | Complete 15 matched student runs and report mean plus sample SD. |
| X8 | 10,000 paired heuristic-group bootstrap replicates for five frozen contrasts | Group and paired-group bootstrap functions are implemented; protocol freezes the five contrasts. | Partial | Implement the cross-run contrast aggregator and execute it after all required per-image tables exist. |
| X9 | Ten-class failure taxonomy | Requirement documented only. | Missing | Implement deterministic classification from selected/oracle/extent/lesion fields and emit counts plus per-image assignments. |
| X10 | Protocol-selected qualitative panel | Selection categories are frozen in the protocol document. | Missing | Implement deterministic selection and renderer after per-image X4 results exist. |
| X11 | Risk-coverage using frozen G1/fusion confidence | Required endpoints are documented. | Missing | Implement score-Dice Spearman, failure AUROC and Dice/miss at 100/80/60/40% coverage on frozen Rich Gallery rows. |
| X12 | Runtime, peak VRAM, storage; offline separated from online | Generators/student metadata already retain elapsed time, device and output bytes in several stages, but no common benchmark runner/summary exists. | Partial | Add one common warm-up plus >=100-image inference benchmark and aggregate median/IQR, VRAM and storage by stage. |

## P0/P1 readiness

| Priority | Requirement | Current state |
|---|---|---|
| P0 | Rich-Gallery pseudo-U-Net, matched fully, CAM, S2C, X2, normal/lesion/subgroup, three seeds, bootstrap/CI, error analysis | In progress; common metrics are ready, but student runs, X2, cross-run statistics and taxonomy remain incomplete. |
| P1 | PuzzleCAM, YOLO, efficiency, qualitative panel, risk-coverage | PuzzleCAM is running; the other four items remain to be implemented/run. |

## Slot-reuse order

1. A completed CAM/PuzzleCAM student slot is reused for the next missing seed
   of the same already-frozen target arm.
2. The S2C generator slot first freezes/audits S2C masks and then launches the
   S2C student seeds.
3. Once W0-W2 seed coverage is no longer the critical path, slots launch Rich
   Gallery and fully-supervised matched students in pairs.
4. YOLO is launched only after the P0 student critical path is fully occupied;
   CPU/local work continues on X2 and X8-X12 without consuming a GPU slot.

No run may use outer-validation polygons for training, checkpoint selection or
threshold selection, and no X4 development run may read test images.
