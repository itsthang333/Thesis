# BTXRD WSSS - thesis-final pipeline

This branch contains the locked thesis method: **Rich Gallery G1 + fixed equal
percentile-rank fusion**. Spatial annotations are never used to create
candidates, train G1, score candidates, or choose a final mask.

## Development result

The method was selected on the canonical validation cohort. These numbers are
not presented as test performance.

| Cohort | n | Dice | IoU | Complete misses |
|---|---:|---:|---:|---:|
| Tumor overall | 184 | **0.288729** | **0.216839** | 49 |
| Lesion area `<1%` | 94 | 0.157723 | 0.118576 | 35 |
| Lesion area `1-<5%` | 72 | 0.435229 | 0.323558 | 13 |
| Lesion area `>=5%` | 18 | 0.386874 | 0.303117 | 1 |

The historical fully-supervised ResNet18-U-Net comparison reached validation
Dice 0.492765. The model retrained on the final manifest and subsequently used
for the locked test comparison reached validation Dice 0.490149 at the same
threshold 0.20 (0.489581 after native-grid inversion). The latter is the matched
final-checkpoint comparison. Fully supervised output is never an input to WSSS.

The final test was evaluated exactly once after source, assets and method were
frozen. WSSS reached Dice 0.260881 and the independently frozen fully-supervised
comparison reached 0.524423 on the 187 tumor test images. No G4 ablation uses
test images or test polygons.

## End-to-end method

```text
BTXRD train images + binary tumor/normal image labels
  -> DenseNet121 classifiers at 320 and 448
  -> LayerCAM-320 and classifier-448 proposal evidence
  -> frozen BiomedCLIP external saliency
  -> SAM ViT-B proposals from all three sources
  -> exact union and duplicate removal (maximum 243 masks/image)
  -> G1 Geometry-v3 candidate encoder + image-label MIL training
  -> freeze each candidate's G1 logit and upstream score
  -> 0.5 * percentile_rank(G1) + 0.5 * percentile_rank(upstream)
  -> stable argmax and immutable mask choices without spatial GT
  -> spatial evaluator (validation during development; test once at the end)
```

The three proposal sources are retained because every two-source ablation
reduced validation Dice. See [docs/RESULTS.md](docs/RESULTS.md).

## Matched final comparison

The repository contains two isolated executable tracks:

- `project/run_fully_supervised_comparison.py`: train/validation only,
  polygon-supervised ResNet18-U-Net at 448 px;
- the WSSS stages above: binary image labels only before evaluation.

For final test, both tracks freeze predictions before spatial test GT.
`evaluate_final_rich_gallery.py` then reads each tumor annotation once and
writes WSSS and fully-supervised per-image results plus `comparison.csv`.
On Kaggle the tracks may run as two private jobs. On one A100 they should run
as independent sequential jobs to avoid resource contention; this scheduling
choice does not change either scientific protocol.

## Scientific test policy

- Validation chose the method and all hyperparameters.
- The final source commit and every external asset are SHA-256 locked.
- Test images may be processed only after that lock exists.
- The setting assumes the binary image label is available when localizing a
  test image. It is therefore image-label-aware WSSS localization, not
  label-free deployment segmentation.
- Test polygons remain unavailable until all 373 choices have been frozen.
- The 187 tumor test polygons are opened by the final evaluator only once.
- Test output cannot be used to modify the method or rerun a tuned variant.

See [docs/TEST_PROTOCOL.md](docs/TEST_PROTOCOL.md) for the A100 procedure.
The stage-by-stage T4x2/A100 support matrix is in
[docs/EXECUTION_MATRIX.md](docs/EXECUTION_MATRIX.md).

## Repository layout

- `project/`: executable final pipeline and its dependency closure.
- `tests/`: focused fail-closed and numerical tests.
- `artifacts/final_pipeline/`: validation result and immutable provenance.
- `docs/`: method, results, test protocol, limitations, and archive pointer.

The complete research history and all retired experiments remain preserved at
commit `aca685f` on branch `codex/research-sync-20260731`.

## Installation

```bash
python -m venv .venv
python -m pip install --upgrade pip
# Install the platform-specific PyTorch/CUDA wheel first, then choose one:
python -m pip install -r project/requirements-candidate.txt
python -m pip install -r project/requirements-g1.txt
python -m pip install -r project/requirements-fully.txt
```

Candidate/BiomedCLIP and G1 use separate environments because the frozen
validation run used `transformers` 4.35.2 and 4.50.2 respectively. Installing
both stage files into one environment is incorrect.

BTXRD, SAM ViT-B, BiomedCLIP/RAD-DINO snapshots, and trained checkpoints are
not committed. Their authoritative hashes and parameters are recorded in
`artifacts/final_pipeline/final_run_config.json` and frozen again before test.

## Verification

```bash
python -m py_compile project/*.py project/evaluation/*.py
python -m pytest -q tests
```

No test image or test annotation is needed for this preflight.
