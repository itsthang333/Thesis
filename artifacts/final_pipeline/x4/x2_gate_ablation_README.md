# X2 binary-versus-ten-class gate ablation

For each seed, Stage A freezes 371 masks for four arms before spatial
validation annotations are opened: known binary label, predicted binary gate,
predicted ten-class gate and the label-free Rich-Gallery student. Stage B then
evaluates exactly 184 tumor polygons on the common native-resolution endpoint.
Test data are not read.

Three-seed mean tumor Dice is 0.288224 for the known binary label, 0.251309 for
the predicted binary gate, 0.253956 for the predicted ten-class gate and
0.081545 for the label-free student. The ten-class gate therefore does not
provide a material improvement over binary gating in this matched experiment.

Full per-image, subgroup, summary and provenance files are stored in
`x2_gate_seed42`, `x2_gate_seed43` and `x2_gate_seed44`.
