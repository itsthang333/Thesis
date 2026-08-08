# X4 experiment protocol

X4 is the post-G4 experiment family that tests whether frozen WSSS masks are
useful training supervision and whether the resulting image-only students are
competitive with matched fully supervised systems. All model selection occurs
inside the canonical 2,981-image training split. The 371-image validation split
is an outer evaluation cohort only; test remains unopened.

Live implementation/run status is tracked in `X4_EXECUTION_MATRIX.md`.

## Frozen requirements

| ID | Experiment | Required arms/output |
|---|---|---|
| X1 | Direct mask vs pseudo-U-Net vs fully-U-Net | Direct R7, Rich-Gallery student, matched fully student |
| X2 | Inference-label assumption | Known label, binary predicted gate, ten-class predicted gate, label-free student |
| W0-W3 | Common WSSS baselines | CAM, PuzzleCAM, S2C, Rich Gallery; direct mask and common-student quality |
| X3 | Official BTXRD upper bound | YOLOv8s-seg, canonical split, native mAP plus common binary-union evaluator |
| X4 | Normal-image evaluation | Specificity/FPR, area median/IQR and >0.1/1/5%, false components, examples |
| X5 | Lesion/multifocal | One-to-one component precision/recall/F1 at IoU 0.10/0.25/0.50 |
| X6 | Boundary/extent | Macro/micro overlap, precision/recall, area ratio/RVD, HD95/ASSD pixels, misses |
| X7 | Seed robustness | Matched U-Net seeds 42/43/44 for CAM-, PuzzleCAM-, S2C- and Rich-Gallery-pseudo-mask supervision plus fully supervised GT; these are standardized downstream students, not the direct original methods |
| X8 | Paired statistics | 10,000 heuristic-group bootstrap replicates and five frozen contrasts |
| X9 | Error taxonomy | Ten frozen failure classes using selected/oracle/extent/lesion evidence |
| X10 | Qualitative panel | Protocol-selected quantiles, sizes, extent, misses, wrong site, normal FP, multifocal |
| X11 | Risk-coverage | Score-Dice Spearman, failure AUROC, Dice/miss at 100/80/60/40% coverage |
| X12 | Efficiency | Median/IQR time, peak VRAM and storage; offline generation separated from inference |

## Matched student contract

All five student arms use the exact architecture, input size, preprocessing,
paired augmentation, loss, optimizer, batch size, 30-epoch budget, threshold
grid, seeds and native-grid evaluator in `x4_protocol.json`. A deterministic
15% group-aware inner holdout is derived only from canonical train. WSSS
students select against their own frozen pseudo targets; the fully supervised
student selects against GT and is explicitly declared fully supervised.

Normal train images receive an explicit empty target in every WSSS arm. No
validation polygon may be opened before a checkpoint and threshold are frozen.

## Generator definitions

- W0 CAM: frozen binary DenseNet-121/320 LayerCAM and label-safe threshold.
- W1 PuzzleCAM: clean binary-label PuzzleCAM generator, no SAM.
- W2 S2C: clean binary S2C classifier/CAM plus SAM-enhanced pseudo masks.
- W3 Rich Gallery: frozen multi-source SAM gallery plus G1 and equal
  percentile-rank R7 selector.
- Fully: polygon union masks from canonical train only.

One frozen generator plus three student seeds is the declared resource-aware
level. This measures student uncertainty but not full generator uncertainty.

## Interpretation safeguards

- Direct known-label R7 is image-label-aware localization, not label-free
  deployment. Its normal specificity is not credited.
- YOLO native mAP is reported only for YOLO; no pseudo-mAP is invented for WSSS.
- HD95/ASSD are reported in pixels because BTXRD has no trustworthy spacing.
- `group_id` is heuristic and bootstrap output must not be called patient-level.
- Qualitative examples are selected by the frozen protocol rather than visual appeal.
- Test remains excluded from X4 development and evaluation.
