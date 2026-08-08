# G4 validation evidence

This directory indexes the independently frozen G4 validation replays. Raw
per-image payloads remain in the private run archive because they include
dataset identifiers and are much larger than the thesis source repository.

## E0 common-grid replay

- Validation only; no test access.
- Split: 371 images, including 184 tumor images.
- WSSS Dice: 0.288729487 at 320, 0.288217723 at 448, and
  0.288224022 after native-grid inversion.
- Matched final-retrain fully-supervised Dice: 0.489308200 at 320,
  0.490148598 at 448, and 0.489581371 after native-grid inversion.
- The lesion-area subgroup membership remains 94/72/18 under the native and
  historical 320 definitions.
- All-candidate common-320 oracle Dice: 0.528298332.
- Corrected MONAI-compatible summary SHA-256:
  `f7cc3b8e5ba1ac60df6dbbc03b8222c38a8f2485247666224feac6d75fbfb082`.
- Corrected evaluator-audit file SHA-256:
  `ba08538672a0101851d8bfa4d212f509900ea82e763a11c3c7df19779e96ca92`.

## E4/E5/E6/E8 offline replay

- Validation only; no test access.
- 27 predeclared arms, 371 images per arm, 10,017 frozen selections.
- Exact R7 reproduction: common-320 Dice 0.288729487.
- Native R7 Dice: 0.288224022.
- G1-eligible candidate oracle: common-320 Dice 0.527902026.
- The official complete-gallery oracle is 0.528298332. The 0.000396 gap to the
  G1-eligible oracle is caused by candidates excluded from G1 scoring; these
  are two named candidate populations, not conflicting metric implementations.
- Random/SAM/upstream/G1 native Dice: 0.101890 / 0.098902 / 0.225306 /
  0.205545.
- Evaluation summary SHA-256:
  `1ae1b1f20bed1b4f403efd28e74b6c58769e4db0d56c3f9baf3de93ba5a471fb`.
- Evaluation audit SHA-256:
  `c4c7055c6f0bfea342e35c02184f19b28c6fb740ac1431ddbbe2ffd33d9270b8`.

### Exact current-split E4 source subsets

- All seven non-empty subsets were replayed with the same frozen G1 plus equal
  percentile-rank selector on exactly 371 validation images (184 tumor,
  subgroups 94/72/18), with zero test access.
- Full selected/oracle Dice: 0.288729487 / 0.528298332.
- Two-source selected Dice: 0.283344376 (L320+C448), 0.282440158
  (C448+external), and 0.280697091 (L320+external).
- Single-source selected Dice: 0.275498639 / 0.258021296 / 0.234635977 for
  L320 / C448 / external.
- Stage-A plus Stage-B time for all seven replays: 47.523953 s after reuse of
  the frozen 1615.225732 s candidate-generation stage.
- Artifacts: `e4_source_subset_results.json` and
  `e4_source_subset_audit.json`; report/audit SHA-256
  `865366ad6ae2077d1c5f43d76ca571941ceb4fc369b5ee9e06a97997f86aafad` /
  `f7db51ea53bc1441963e2cd761b00ad87aaecfd79327b2aa7e28ad8adc23f973`.

## E1 label-granularity image-level stage

- Matched binary and ten-class DenseNet-121 classifiers completed for seeds
  42/43/44; all six runs use the same binary-F1 checkpoint endpoint and no
  spatial GT/test.
- Binary versus ten-class-to-binary mean AUROC: 0.848727 versus 0.860100.
- Binary versus ten-class-to-binary mean F1: 0.780882 versus 0.795082.
- Binary versus ten-class-to-binary mean NLL: 1.514750 versus 1.033681.
- Discrimination paired intervals include zero; NLL favours ten-class in all
  three matched seeds. The matched downstream study is also complete: binary
  and ten-class mean mask Dice are 0.262899 and 0.298954, respectively.
- The CAM-only completion is also frozen for all six checkpoints. Binary and
  ten-class-to-binary mean CAM-only Dice is 0.133468 and 0.172241. Artifacts:
  `e1_cam_only_completion.json` and `e1_cam_only_completion_audit.json`;
  independent audit SHA-256
  `4efd081d16e02a660271e82a7087aeb90174ba1e72d58a4845dba955fd3a7c3f`.
- Independent audit artifact: `e1_label_granularity_audit.json`.
- Audit SHA-256:
  `2792af5f4bb61cddb1329f670e1b94df85adc161caa9d8db4b56759caf7c52c6`.
- Corrected tie-invariant step-wise AP audit:
  `e1_metric_reaudit/e1_audit.json`, SHA-256
  `967b588537d4a2f2814a68e2a660c5b35fc87dd559482efb97069a9f58d4b215`.
  Corrected binary/ten-class-collapsed AP is 0.860190/0.869750; no Dice,
  AUROC, F1 or calibration value changed.

## E2 attribution/prompt runtime

- The 12 factorial arms all have an exact launcher-start and 371/371
  candidate-diagnostics-freeze event in the two original Kaggle T4 logs.
- Candidate generation ranges from 262.901 to 342.988 seconds per 371-image
  arm, totals 3560.736 seconds, and averages 296.728 seconds/arm.
- The separately measured evaluator totals 194.159 seconds across 12 arms.
- Artifact: `e2_runtime.json`, SHA-256
  `7d80a394f01db8403ed70f343cc16bab965bc18d2799c0a896cdd92e7fe6dc7f`.
- Bound raw-log SHA-256 values:
  `65f8d7bd6098ed34842ef2b8e62a1a7c6dc2f9005c58c5ee175573bae4e45fcf` and
  `b0164e07b76a7deaba658afa98ead2fbda82e5bf04fa2ce3c49eb78ebb573221`.

## E7 source-correct upstream-score ablation

- Validation only; 371 frozen choices per arm, including 184 tumor images;
  exact subgroup counts 94/72/18 and zero test access.
- Sixteen predeclared upstream-only and upstream-plus-G1 arms were evaluated
  after the source-specific saliency/component evidence was frozen.
- Legacy U5+R7 reproduces common-320 Dice 0.288729487 and obtains native Dice
  0.288224022.
- Source-correct U5+R7 obtains native Dice 0.289357584.
- The largest point estimate is source-correct global-rank U6+R7: native Dice
  0.294955925 (small/medium/large 0.149495/0.450780/0.431288) and common-320
  Dice 0.295568298. Its paired native delta is +0.006732, but the 95% interval
  [-0.010248, 0.024597] includes zero; it is therefore evidence about the
  formula, not proof of superiority.
- Frozen artifacts: `e7_source_correct_summary.json`,
  `e7_source_correct_evaluation_audit.json`, and
  `e7_source_correct_run_manifest.json`.
- Summary/audit SHA-256:
  `6a57c67d07ac92be68918e79bee2f5ac2855da6379592dc43c0e15c01e41181c` /
  `d36e359e972ff76acfce2fc6eca780801711d93f27f7e7e149c60999075034af`.

## E3 SAM-v1 backbone ablation

- ViT-B/L/H all completed and passed the same independent 371-image,
  validation-only output audit.
- Dice is 0.288729/0.291185/0.279212 and candidate-oracle Dice is
  0.528298/0.546000/0.510446 for B/L/H.
- H minus B paired Dice delta is -0.009517 with 95% CI
  [-0.034714, 0.015280]; H requires 3.17x total wall time and 2.02x peak
  allocated VRAM. H minus L is -0.011973 with CI
  [-0.029889, 0.005554].
- ViT-L's small overall point-estimate gain over B is also uncertain while
  costing 1.84x total wall time. ViT-B is therefore retained as the frozen
  accuracy/resource choice for this pipeline and hardware, not asserted as a
  universally superior SAM backbone.
- Frozen independent artifacts: `e3_vit_h_audit.json`,
  `e3_vit_b_vs_h.json`, and `e3_vit_l_vs_h.json` (plus the existing B/L
  audits and B-vs-L comparison).
- H audit/B-vs-H/L-vs-H SHA-256:
  `1c35747c6352a8b0fb99c7ea28784568cea6ed6cd078160e8e83c8ccb670181f` /
  `ec86b2d257342bf242ef488b7be83a82f64022566ebb056fcf528efe1e58484b` /
  `1b3c6f58722352368f5f01fc1e43d04120a8757bceeb518f426b547f3a9a1d95`.

The exact formulas, arm definitions, safeguards, and interpretation limits are
documented in `docs/G4_EXPERIMENT_PROTOCOL.md`,
`docs/METRIC_AND_DEFENSE_AUDIT.md`, and `docs/RESULTS.md`.
