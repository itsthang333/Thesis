# Thesis inference demo

`demo_rich_gallery_inference.ipynb` is the defense-ready replay of the final
Rich Gallery G1 inference pipeline. It uses the real locked test case
`IMG001598.jpeg` and reproduces the selector from frozen candidate scores.

The notebook deliberately separates inference from evaluation:

1. cells under **Inference** use only the X-ray, image-level tumor label,
   frozen localization maps, SAM proposals, G1 logits, and upstream scores;
2. the ground-truth mask is first opened under **Evaluation boundary**;
3. Dice and IoU are recomputed from the selected mask and the ground truth.

## Required local evidence

The default paths expect the archived evidence already present in this
workspace:

- `outputs/analysis/round2_test_evidence_20260812/cases/three_source_complementarity`
- `outputs/private/round2_test_evidence_20260812/targeted/gallery/final_test_gallery/candidate_diagnostics/IMG001598.npz`
- `outputs/private/round2_test_evidence_20260812/targeted/scores/final_test_scores/descriptor_evidence/0152_IMG001598.npz`

These outputs are intentionally not committed because they contain dataset
images and large experiment artifacts. Set `REPO_ROOT` in the configuration
cell if the notebook is copied elsewhere.

## Recommended presentation flow

- Run all cells before the defense and keep the outputs visible.
- Present the inference cells first without scrolling to the ground truth.
- Pause at the selected mask and ask the audience to note that no spatial
  annotation has been used.
- Then cross the evaluation boundary and reveal the Dice/IoU verification.
- Expected result for the selected case: Dice `0.8574706364`, IoU
  `0.7505020560`, selected candidate `87/144`.

