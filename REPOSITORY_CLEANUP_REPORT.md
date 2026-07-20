# Repository Cleanup Report

Date: 2026-07-20  
Branch audited: `pipeline`  
Cleanup authority: `Repository Cleanup Plan.pdf`

## A. Verdict

**CONDITIONAL PASS.** The repository has been reduced to one BTXRD pipeline
and one canonical notebook, and all checks executable in the local lightweight
environment pass. Full readiness is not yet proven because this machine has no
PyTorch/CUDA, attached BTXRD corpus, SAM checkpoint, or Kaggle runtime.
Consequently six GPU/PyTorch integration tests and the required Kaggle
single-image smoke run could not execute.

## B. Canonical map

1. `tools/build_btxrd_split_manifest.py`: immutable group-aware split.
2. `train_classifier.py`: image-level DenseNet121 classifier.
3. `evaluate_classifier.py`: multiclass and tumor-gate report.
4. `generate_pseudo_masks.py`: LayerCAM, tumor morphology, SAM v1 ViT-B,
   strict mask selection, pseudo-mask manifest.
5. `evaluate_pseudo_masks.py`: polygon-only diagnostic evaluation.
6. `train_segmentation.py`: U-Net on WSSS pseudo-masks; optional supervised
   oracle uses the same trainer.
7. `evaluate_unet.py`: held-out U-Net evaluation.
8. `inference.py`: U-Net-only final deployment.
9. `thesis_final.ipynb`: sole end-to-end orchestration notebook.

## C. Files removed

Dataset/model/backend legacy:

- `project/datasets/ramh1200.py`
- `project/pseudo/bone_morphology.py`
- `project/pseudo/morphology_factory.py`
- `project/pseudo/cam_refine.py`
- `project/tools/evaluate_cam_methods.py`
- `project/tools/fit_linear_head.py`
- `project/tools/rebuild_btxrd_debug_notebook.py`
- `project/visualize_pipeline.py`

Historical notebooks:

- `btxrd_colab.ipynb`
- `btxrd_kaggle.ipynb`
- `btxrd_kaggle_vi_debug.ipynb`
- `btxrd_pipeline_colab.ipynb`
- `thesis_experiment_debug.ipynb`
- `thesis_experiment_kaggle.ipynb`
- `thesis_smoke_test.ipynb`

Stale documentation/assets:

- `pipeline.md`
- `documents/EXPERIMENT_TRACKING.xlsx`
- `documents/FracAtlas_WSSS_Research_Roadmap_V1.pdf`

The old `project/evaluate_ramh1200_masks.py` path was replaced by
`project/evaluate_pseudo_masks.py`.

## D. Symbols and branches removed

- RAM-H1200 dataset dispatch and COCO annotation argument.
- SAM2 and MedSAM2 checkpoint URLs, downloads, Hydra fallback, model config,
  model-type inference, and CLI branches.
- Runtime SAM download and implicit checkpoint defaults.
- Optional feature-affinity CAM-refinement implementation and flags.
- Dataset-selecting morphology factory.
- Final-inference pseudo-diagnostic/classifier/CAM/SAM path.
- Duplicate classifier confusion, AUROC/AP, binary, and multiclass metric
  implementations.
- Public `--dataset`, `--ram-root`, `--annotation-name`, and legacy
  bone-named morphology options.

## E. Files merged or rewritten

- `project/datasets/factory.py`: BTXRD-only constructors.
- `project/pseudo/sam_refine.py`: strict official SAM v1 ViT-B wrapper.
- `project/generate_pseudo_masks.py`: direct tumor morphology, canonical
  backend, BTXRD CLI, updated provenance keys.
- `project/evaluate_classifier.py`: uses
  `evaluation/classification_metrics.py` as the sole metric implementation.
- `project/inference.py`: U-Net-only deployment with checkpoint architecture,
  dataset, preprocessing, threshold, original-size output, and SHA-256
  metadata checks.
- `README.md`: current BTXRD/Kaggle workflow only.
- `FINAL_AUDIT_KAGGLE_READINESS.md`: current filenames and CLI.
- `project/config.py`: BTXRD-only dataset constants; SAM v1 is now structural,
  not a selectable profile field.

## F. Notebook cleanup

Exactly one notebook remains: `thesis_final.ipynb`.

- 40 cells, 25 code cells.
- Zero stored outputs.
- Every code cell compiles.
- Commands use `--data-root` and the renamed pseudo-mask evaluator.
- Default profile is now consistently `btxrd_best`.
- Internet downloads remain forbidden.
- Predicted, known-label, oracle, supervised-baseline, frozen-config, resume,
  and locked-test protocols remain separated.

## G. Dependencies

`project/requirements.txt` remains version locked. `pycocotools` was
removed because it served only the deleted RAM-H1200 COCO loader.
`segment-anything` remains pinned to an immutable SAM v1 commit.
SAM2/MedSAM2 are not dependencies. Kaggle must install dependencies from
attached local artifacts before preflight; no mid-run upgrades are allowed.

## H. Verification executed

- `python -m compileall -q project tests`: **PASS**
- `python -m unittest discover -s tests -v`: **PASS**
  - 25 discovered
  - 19 passed
  - 6 explicitly skipped because PyTorch is absent
  - 0 failed
- Notebook JSON parse and all code-cell compilation: **PASS**
- Notebook inventory: exactly one notebook and zero outputs: **PASS**
- Reverse lookup for deleted paths/backends: no live-code/doc references.
  One occurrence remains only inside `audit/local_readiness.json`, an
  immutable historical pre-cleanup status snapshot.

Not executable locally:

- model construction/backpropagation;
- optimizer/resume equivalence integration;
- CUDA inference;
- SAM single-image generation;
- end-to-end Kaggle smoke run.

## I. Remaining inventory

The production repository retains:

- 1 notebook;
- 1 BTXRD dataset implementation plus shared transforms/factory;
- classifier, LayerCAM, PuzzleCAM/teacher-student training support, U-Net,
  losses, differentiable morphology;
- tumor morphology, prompts, SAM v1, selection, manifests, diagnostics,
  visualization helpers;
- centralized classification and segmentation metrics;
- split, freeze, and Kaggle-readiness tools;
- classifier, pseudo-label, U-Net training/evaluation, and final inference
  entrypoints;
- two regression test modules.

PuzzleCAM/EMA code remains because it is still referenced by the retained
classifier experimental profile and its regression tests; it is not enabled
by canonical `btxrd_best`.

## J. Artifact compatibility

- Existing DenseNet121 and U-Net state dictionaries remain structurally
  compatible; model architectures were not changed.
- Current checkpoints with stored dataset, split-manifest, preprocessing, and
  architecture metadata remain supported.
- Final inference now rejects non-BTXRD or incompatible-architecture
  checkpoints instead of silently loading them.
- SAM v1 ViT-B checkpoints remain compatible. SAM2/MedSAM2 checkpoints are
  intentionally unsupported.
- Old command lines are breaking: replace `--ram-root` with `--data-root`,
  remove `--dataset btxrd`, and use `evaluate_pseudo_masks.py`.
- New pseudo-run metadata uses `sam_backend=sam_v1_vit_b` and
  `max_components`; consumers that depended on old descriptive keys must be
  updated. Mask PNGs and model state dictionaries are unchanged.
- Existing output/checkpoint/data directories were not deleted.

## K. Git diff status

The cleanup is intentionally uncommitted. At the final review, tracked diff
statistics were approximately 1,500 insertions and 19,800 deletions across 40
tracked paths, plus the earlier audit additions already present in the dirty
working tree. No reset, checkout, staging, commit, push, dataset deletion,
checkpoint deletion, or output deletion was performed.

## L. Final confirmation

The source tree and notebook are ready for a **controlled Kaggle smoke run**,
not yet for an evidence-free claim of full end-to-end success. Before the
thesis full run, Kaggle must produce all of the following at one exact commit:

1. strict preflight PASS with attached dataset/manifest/SAM/dependencies;
2. all 25 tests passing with no PyTorch skips;
3. one deterministic validation-image classifier -> CAM -> SAM -> pseudo-mask
   smoke output and manifest;
4. short U-Net train/resume/evaluate smoke;
5. only then, a new full run directory and the locked final protocol.

Until those runtime checks pass, the correct release verdict remains
**CONDITIONAL PASS**.

