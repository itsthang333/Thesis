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
- Random/SAM/upstream/G1 native Dice: 0.101890 / 0.098902 / 0.225306 /
  0.205545.
- Evaluation summary SHA-256:
  `1ae1b1f20bed1b4f403efd28e74b6c58769e4db0d56c3f9baf3de93ba5a471fb`.
- Evaluation audit SHA-256:
  `c4c7055c6f0bfea342e35c02184f19b28c6fb740ac1431ddbbe2ffd33d9270b8`.

The exact formulas, arm definitions, safeguards, and interpretation limits are
documented in `docs/G4_EXPERIMENT_PROTOCOL.md`,
`docs/METRIC_AND_DEFENSE_AUDIT.md`, and `docs/RESULTS.md`.
