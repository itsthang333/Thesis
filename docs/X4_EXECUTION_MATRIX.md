# X4 execution matrix

This file is the requirement-by-requirement execution ledger for `X4.pdf`.
`Complete` means that both an implementation and a result artifact exist.
`Implemented` means that the code is ready but the required run/result is not
yet complete.  Validation is the canonical 371-image outer cohort; test remains
locked throughout X4 development.

Last updated: 2026-08-10.

## Paused GPU state

All new experiments were paused on 2026-08-10 at the researcher's request so
that the next available GPU budget can be assigned to one highest-priority
experiment. No X4 test image was opened during this pause.

| Account | Kernel | Purpose | State at update |
|---|---|---|---|
| `wanwin` | `btxrd-x4-yolo-eval-seed42` v6 | Freeze/evaluate completed YOLO seed42 checkpoint | Technical CUDA OOM in the YOLO prototype head; no efficacy result |
| `wanwin` | `btxrd-g4-ten-medsam-screen` v1 | Validation-only ten-class + MedSAM matched foundation-model screen | Complete: Dice 0.232962 |
| `qwinwan` | `btxrd-x4-yolo-eval-seed44` v6 | Freeze/evaluate completed YOLO seed44 checkpoint | Technical CUDA OOM in the YOLO prototype head; no efficacy result |
| `qwinwan` | `btxrd-g4-ten-sam2l-screen` v2 | Validation-only ten-class + SAM2.1-L matched foundation-model screen | Cancelled while running; no final Dice result |
| `itsthang333` | `btxrd-x4-yolo-eval-seed43` v6 | Freeze/evaluate completed YOLO seed43 checkpoint | Technical CUDA OOM in the YOLO prototype head; no efficacy result |
| `itsthang333` | `btxrd-g4-ten-sammed2d` | Validation-only ten-class + SAM-Med2D matched screen | Not launched; complete input payload restored and preserved |

Three second-account slots became available after X12 completion. L4 adds X13
and X14 as non-duplicate supplementary work. X13 is complete and audited;
X14 v1 stopped before training because its wrapper retained a stale expected
gallery SHA-256; this is a technical provenance failure, not an efficacy
result. Cross-account inputs were server-download verified
against their bound local SHA-256 values before either run.

The three YOLO training jobs are complete. Their native Ultralytics mask
`mAP50/mAP50-95` values are `0.561096/0.277663` (seed42),
`0.533550/0.267338` (seed43), and `0.541830/0.272754` (seed44). These are not
substitutes for the common binary-union Dice endpoint: checkpoint download,
371-image prediction freeze and Stage-B Dice/IoU evaluation remain pending.
All three receipts bind the canonical split and report zero test access.

The matched v6 YOLO evaluation retries also produced no binary-mask Dice.
All three independently reached the same CUDA OOM in the segmentation
prototype head on the first inference pass: a 4.09-GiB allocation request on
GPU 0 after 12.95 GiB was already in use. This occurs before the low-resolution
instance-mask union and therefore is not evidence about YOLO segmentation
efficacy. Partial prediction files are not treated as a valid 371-image result.

All five bounded X12 student/fully jobs are
complete and audited, so their former slots are intentionally not filled with
duplicate runs. These jobs do not retrain models: they reuse exact seed42
checkpoints and measure 371-image online inference latency, T4 peak memory and
prediction storage with three warm-up passes while explicitly excluding
offline pseudo-label generation.

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
accounts. Corrected seed42/43 v4 jobs and seed44 v1 have now completed
training.

The first Rich-student prediction-freeze versions failed before inference
because one archive basename in the reused wrapper remained hard-coded as
`x4_s2c_...`.  Version two changes only that basename to the arm-qualified
`x4_{ARM}_...`; checkpoint hashes, thresholds and the scientific protocol are
unchanged.  Static archive/checkpoint hash checks and syntax checks pass.

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
| S2C-to-U-Net | 43 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| S2C-to-U-Net | 44 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| Rich-Gallery-to-U-Net | 42 | 0.071517 | 0.041128 | 0.009421 | 0.099037 | 0.285710 | 0.951872 | 323.3770 |
| Rich-Gallery-to-U-Net | 43 | 0.095067 | 0.056616 | 0.015964 | 0.149795 | 0.289250 | 0.850267 | 198.9174 |
| Rich-Gallery-to-U-Net | 44 | 0.078052 | 0.045134 | 0.009427 | 0.122887 | 0.257086 | 0.770053 | 238.5777 |
| Fully supervised U-Net | 42 | 0.456580 | 0.367609 | 0.286350 | 0.624306 | 0.674657 | 0.288770 | 4.2848 |
| Fully supervised U-Net | 43 | 0.456626 | 0.363456 | 0.293991 | 0.607704 | 0.701623 | 0.406417 | 5.3053 |
| Fully supervised U-Net | 44 | 0.449361 | 0.356357 | 0.296252 | 0.598555 | 0.652155 | 0.294118 | 4.5662 |

These results establish efficacy failures, not execution failures. CAM,
PuzzleCAM and Rich-Gallery students strongly over-segment, especially in the
`<1%` subgroup. Rich students have a three-seed mean Dice of 0.081545 (sample
SD 0.012158) and extremely large macro predicted/GT area ratios. All three S2C
students collapse to all-background: 184/184 tumor predictions are empty for
every seed. This follows the audited S2C targets (only 1/1,488 tumor training
targets has foreground), not an evaluator fault. Direct Rich Gallery remains
the current validation WSSS baseline at Dice 0.2887294867.

## X2 gate ablation

All four arms were frozen before any polygon was opened, then evaluated on the
same 371-image native-resolution endpoint. The table reports the three-seed
mean; the known-label arm is deterministic and is repeated only to preserve the
matched seed protocol.

| Gate arm | Mean Dice | Sample SD | Mean IoU | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|---:|---:|
| Known binary label | 0.288224 | 0.000000 | 0.216295 | 0.156725 | 0.435221 | 0.386958 |
| Binary predicted gate | 0.251309 | 0.007822 | 0.190746 | 0.129155 | 0.376875 | 0.386958 |
| Ten-class predicted gate | 0.253956 | 0.014822 | 0.191749 | 0.123362 | 0.398274 | 0.358680 |
| Label-free Rich student | 0.081545 | 0.012158 | 0.047626 | 0.011604 | 0.123906 | 0.277349 |

The ten-class gate does not outperform the binary gate materially and both
remain below the known binary label reference. This supplies the requested
empirical answer: replacing the binary task with ten disease classes does not
improve final segmentation in this matched experiment.

## X12 matched T4 student efficiency

| Arm, seed42 | Median ms/image | IQR ms/image | Peak allocated MiB | Prediction storage MiB |
|---|---:|---:|---:|---:|
| CAM student | 10.731 | 10.643-10.847 | 614.185 | 96.187 |
| PuzzleCAM student | 10.623 | 10.598-10.778 | 614.185 | 89.113 |
| S2C student | 10.785 | 10.668-10.852 | 614.185 | 57.946 |
| Rich-Gallery student | 10.771 | 10.608-10.811 | 614.185 | 86.613 |
| Fully supervised U-Net | 10.662 | 10.591-10.781 | 614.185 | 104.436 |

Every row uses the same Tesla T4 x2 environment, three warm-ups and all 371
validation images. The results measure online inference only; offline
pseudo-label generation is excluded by design and must be reported separately.

## X13 equal-budget source complementarity

Version three completed after two fail-fast technical diagnoses. Version one
missed the source root in `PYTHONPATH`; version two correctly exposed that the
external source is generated only for the 184 known-tumor images. The final
protocol therefore applies identical per-image budgets on those 184 cases and
records explicit empty-mask abstention for all 187 known-normal cases. The
independent audit passes with 371 frozen choices, 1,288 tumor/subset rows and
zero test reads. Mean candidate budget is 26.897 (range bounded by 27).

| Equal-budget source subset | Dice | IoU | Oracle Dice | Regret | `<1%` | `1-<5%` | `>=5%` |
|---|---:|---:|---:|---:|---:|---:|---:|
| L320 + C448 | **0.275508** | 0.205566 | 0.410035 | 0.134526 | 0.147928 | 0.421661 | 0.357148 |
| C448 + external | 0.274591 | **0.207330** | **0.412641** | 0.138049 | **0.151048** | 0.417439 | 0.348371 |
| L320 + C448 + external | 0.271698 | 0.202496 | 0.411649 | 0.139951 | 0.145263 | **0.423016** | 0.326698 |
| C448 | 0.258230 | 0.190881 | 0.384003 | 0.125773 | 0.135765 | 0.409702 | 0.291882 |
| L320 | 0.255826 | 0.192786 | 0.369000 | **0.113175** | 0.117984 | 0.396001 | **0.414964** |
| L320 + external | 0.254619 | 0.192893 | 0.379548 | 0.124929 | 0.127964 | 0.396047 | 0.350320 |
| external | 0.228535 | 0.172433 | 0.361730 | 0.133195 | 0.061102 | 0.400756 | 0.414028 |

At a matched proposal count, the external source does not improve selected
Dice over the L320+C448 pair. Full-minus-L320+C448 is -0.005366 with grouped
bootstrap CI [-0.019132, 0.005643], so there is no evidence that adding the
external source improves the selector endpoint. Full does beat external-only
by 0.042861, CI [0.008315, 0.078053], showing complementarity of the two
internal sources rather than a universal benefit from every source. This is an
ablation result; it does not replace the unrestricted Direct-Rich baseline
Dice 0.288729.

## DSLL Top-3 plus early-collapse diagnostic

The final `wanwin/btxrd-dsll-top3-g1-validation` version 2 run completed in
11,589 seconds. Its receipt, archive, checkpoint, score freeze, choice freeze,
selection manifest and per-image table were independently hash-checked. It
contains exactly 2,981 training and 371 validation images; the 371 choices were
frozen before opening the 184 validation polygons, and no test image was read.
Independent aggregation of all per-image rows reproduces the published result.

| Cohort | n | Selected Dice | Selected IoU | Generic oracle | Generic + DSLL oracle | Full seven-source oracle | Selector regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 184 | 0.272935 | 0.207377 | 0.412356 | 0.493133 | **0.560114** | 0.287179 |
| `<1%` | 94 | 0.130710 | 0.099219 | 0.205497 | 0.302469 | 0.369284 | 0.238574 |
| `1-<5%` | 72 | 0.424694 | 0.318341 | 0.630832 | 0.694114 | 0.759284 | 0.334590 |
| `>=5%` | 18 | **0.408625** | 0.328345 | 0.618719 | 0.684896 | 0.759988 | 0.351363 |

DSLL therefore improves candidate supply rather than the deployed endpoint:
the seven-source oracle rises above the prior 0.528298 ceiling to 0.560114,
while selected Dice falls from 0.288729 to 0.272935. The small and medium
subgroups decline by 0.027013 and 0.010535; only the large subgroup improves by
0.021751. This is direct evidence that selector regret, not proposal absence,
is the immediate bottleneck after adding DSLL. The DSLL gallery remains useful
as a candidate source, but this G1/R7 consumer does not replace the baseline.

## Requirement ledger

| ID | Required evidence | Implementation/evidence | Status | Next blocking action |
|---|---|---|---|---|
| X1 | Direct Rich Gallery versus Rich-Gallery pseudo-U-Net versus matched fully-U-Net | All matched runs are complete. Direct Rich is 0.288729. Rich students are 0.071517/0.095067/0.078052; fully supervised is 0.456580/0.456626/0.449361. | Complete | Report the direct-vs-student conclusion and paired uncertainty in the thesis. |
| X2 | Known binary label, binary predicted gate, ten-class predicted gate and label-free student | Complete for all three seeds. Four arms x 371 images per seed were frozen before polygons; Stage B opened exactly 184 validation annotations. Mean Dice is 0.288224 known-label, 0.251309 binary gate, 0.253956 ten-class gate and 0.081545 label-free Rich student. | Complete | Report that ten-class supervision did not improve the matched final segmentation endpoint. |
| W0 | CAM-to-U-Net | CAM train/validation mask freezes and audits are complete. The standardized 2,981-image target bundle is frozen. Seeds 42, 43 and 44 completed and passed training-output checks. All three 371-image prediction freezes passed archive/freeze SHA verification and common Stage-B evaluation. Dice is 0.141918/0.123913/0.142643 for seeds 42/43/44. | Complete | Aggregate mean/sample SD and include the over-segmentation/normal-FP evidence in X8/X9. |
| W1 | PuzzleCAM-to-U-Net | PuzzleCAM generator training and train/validation mask audits are complete. The standardized target bundle is frozen. Seeds 42/43/44 passed training, prediction-freeze and Stage-B checks; Dice is 0.116542/0.116134/0.116389 (mean 0.116355, sample SD 0.000206). | Complete | Include the stable over-segmentation and normal-FP failure in X8/X9. |
| W2 | S2C-to-U-Net | All three trainings/freezes/evaluations passed provenance checks and produce Dice/IoU 0. Only 1/1,488 tumor train targets has foreground, explaining the deterministic all-background collapse. | Complete | Report as target-generation efficacy failure. |
| W3 | Rich-Gallery+G1+R7-to-U-Net | All three trainings/freezes/evaluations passed. Dice is 0.071517/0.095067/0.078052 (mean 0.081545, SD 0.012158); massive over-segmentation and normal FP explain the gap to direct Rich. | Complete | Retain direct Rich inference; do not replace it with pseudo-mask distillation. |
| X3 | YOLOv8s-seg upper bound, 600 px, 300 epochs, native mAP and common binary-union evaluation | Export, fixed protocol and Stage-A/B code are implemented. V2 exposed the batch-size-two tensor/tuple bug; V3 passed it and exposed the independent background-only `None` dereference plus read-only JPEG-repair exclusions. V4 fixes those compatibility faults without changing the scientific protocol; 7/7 focused tests and three-account server source hashes pass. | Seeds42/43 v4 and seed44 v1 running | Confirm zero canonical image exclusions and passage beyond epoch two, then audit/freeze/evaluate all three seeds. |
| X4 | Normal-image specificity/FPR, predicted area distribution/threshold rates, false components and examples | `project/evaluation/segmentation_metrics.py` implements all numeric endpoints; the student evaluator uses all 187 normal validation images. | Implemented | Produce results after each prediction freeze; add protocol-selected normal examples under X10. |
| X5 | 8-neighbour lesion/multifocal metrics and one-to-one matching at IoU 0.10/0.25/0.50 | Implemented in `project/evaluation/segmentation_metrics.py`, including precision/recall/F1, missed/excess lesions, partial multifocal miss and component-count error. Exact fixtures now verify the expected 2/1/1 matches and F1 1.0/0.5/0.5 at the three thresholds. | Implemented and tested | Run on every arm/seed after each prediction freeze. |
| X6 | Macro/micro overlap, pixel precision/recall, extent, HD95/ASSD, empty and zero-overlap cases | Implemented in the common native-grid evaluator. HD95/ASSD are explicitly pixels and conditional-defined counts are reported. | Implemented | Run on every arm/seed and aggregate the declared subgroup tables. |
| X7 | Seeds 42/43/44 for CAM, PuzzleCAM, S2C, Rich Gallery and fully supervised | All 15/15 required student seed results are trained, frozen and evaluated on the same native grid. | Complete | Transfer the mean/SD table to the thesis. |
| X8 | 10,000 paired heuristic-group bootstrap replicates for five frozen contrasts | Complete artifact `artifacts/final_pipeline/x4/x8_multiseed_student_study.json`. Rich student minus Direct Rich mean delta is -0.207184; every seed CI is below zero. Rich also trails CAM, PuzzleCAM and fully supervised, but beats collapsed S2C. | Complete | Report paired CIs and avoid claiming pseudo-mask distillation improvement. |
| X9 | Ten-class failure taxonomy | Complete for Direct Rich and all 15 students under `artifacts/final_pipeline/x4/x9_error_taxonomy_all_students`. Rich students over-segment 165-176/184 tumors and trigger normal FP on 144-178/187 normals; S2C misses/under-segments 184/184 tumors. | Complete | Use taxonomy to explain mechanism-specific failures. |
| X10 | Protocol-selected qualitative panel | Complete. The pre-render freeze supplies 12/12 categories without visual cherry-picking, including normal FP. Twelve panels were rendered with exact Direct Rich choices/gallery plus CAM, PuzzleCAM, S2C, Rich-student and fully-supervised seed42 bundles. All provenance was verified before 11 tumor annotations were opened; no test read. | Complete | Transfer the frozen panels and selection protocol to the thesis. |
| X11 | Risk-coverage using frozen G1/fusion confidence | `project/analyze_x4_risk_coverage.py` and the frozen result under `artifacts/final_pipeline/x4/x11_risk_coverage` are complete. Spearman confidence-vs-Dice is 0.313141; Dice<0.10 and complete-miss AUROC are 0.658570/0.656916. Mean Dice at 100/80/60/40% coverage is 0.288729/0.335050/0.361465/0.367326. | Complete | Interpret as moderate failure-detection evidence, not calibrated uncertainty or a new routing rule. |
| X12 | Runtime, peak VRAM, storage; offline separated from online | Same-T4 student/fully benchmark complete for all five seed42 checkpoints: 371 timed images, three warm-ups, 0 validation annotations and 0 test images read. Median online latency is 10.623-10.785 ms/image and peak allocation is 614.185 MiB. Raw receipts and aggregate CSV are under `artifacts/final_pipeline/x4/x12_efficiency_seed42`. | Student/fully complete | Add Direct Rich and YOLO online rows separately; report offline pseudo-label/generator resources outside this table. |
| X13 | Equal-budget source complementarity | Complete and independently audited: 371 freezes, 184 tumor cases x seven subsets, 187 known-normal abstentions and zero test reads. Best matched-budget subset is L320+C448 Dice 0.275508; full is 0.271698 and external-only 0.228535. Full-minus-L320+C448 CI crosses zero, while full significantly beats external-only. | Complete | Report that candidate-count control supports L320+C448 complementarity but not an added selected-Dice benefit from external proposals. |
| X14 | Selector depth/capacity baseline | Design and implementation frozen: upstream, linear, one-hidden-layer and exact current two-hidden-layer G1; selector-only and identical R7 fusion; seeds42/43/44. All learned arms share one RAD-DINO descriptor cache, one group-aware inner split, the full MIL objective, optimizer and fixed 16 epochs with no best-epoch selection. Stage-A output auditor and Stage-B evaluator support are implemented; 31/31 focused/regression tests pass. Source/protocol payload server-download hashes match. V1 failed its pre-training gallery hash guard because the wrapper expected a stale SHA-256. | Technical pre-training error; no efficacy result | Rebind the exact server-returned gallery SHA, rerun provenance checks, then launch once when a slot is available. |
| X15 | Optional four-fold robustness | Four locked endpoints only; group-aware train+validation folds; test excluded. | Deferred | Run only if GPU remains after X3/X13/X14. |
| X8b | Locked-test uncertainty | Requires only already-frozen test predictions, not retraining. | Deferred by test lock | Do not access during X4 development; perform only under explicit final-analysis authorization. |
| DSLL | Ten-class Top-3/late-fusion proposal supply plus early-collapse and G1/R7 selection | Version 2 COMPLETE and hash-audited. Full seven-source oracle is 0.560114, but actual selected Dice is 0.272935 versus the 0.288729 baseline. | Complete; supply gain, selector efficacy failure | Retain DSLL as evidence/candidate supply; do not promote this selector configuration. |

## P0/P1 readiness

| Priority | Requirement | Current state |
|---|---|---|
| P0 | Rich-Gallery pseudo-U-Net, matched fully, CAM, S2C, X2, normal/lesion/subgroup, three seeds, bootstrap/CI, error analysis | Complete: all 15 student runs, X2, bootstrap/CI and taxonomy have audited result artifacts. |
| P1 | PuzzleCAM, YOLO, efficiency, qualitative panel, risk-coverage | PuzzleCAM, student/fully efficiency, risk-coverage and the qualitative panel are complete; YOLO and separate Direct-Rich/YOLO efficiency rows remain. |

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
