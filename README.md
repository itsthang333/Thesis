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

The final test result is intentionally pending. It will be measured exactly
once after the source commit, asset hashes, method, and validation result have
been frozen in `final_test_protocol.json`.

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
python -m pip install -r project/requirements.txt
```

BTXRD, SAM ViT-B, BiomedCLIP/RAD-DINO snapshots, and trained checkpoints are
not committed. Their hashes are frozen before the test run.

## Verification

```bash
python -m py_compile project/*.py project/evaluation/*.py
python -m pytest -q tests
```

No test image or test annotation is needed for this preflight.
