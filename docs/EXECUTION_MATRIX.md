# Execution matrix

| Stage | Environment | Kaggle T4x2 | Server A100 48 GB | Spatial GT |
|---|---|---|---|---|
| Classifier/LayerCAM candidate supply | candidate | Runs | Runs on one A100 | No |
| BiomedCLIP saliency | candidate | Runs on one T4 | Runs on one A100 | No |
| SAM galleries and merge | candidate | Runs | Runs on one A100 | No |
| RAD-DINO descriptor cache | G1 | DataParallel across T4x2 | One A100 | No |
| G1 image-label MIL training | G1 | Runs | Runs with same method | No |
| Final G1 scoring and rank fusion | G1 | DataParallel encoder | One A100 | No |
| Fully-supervised ResNet18-U-Net train/val | fully | DataParallel across T4x2 | One A100 | Train/val only |
| WSSS choice freeze | common | Runs | Runs | No |
| Fully prediction freeze | fully | Runs | Runs | No |
| Joint Dice/IoU evaluator | common/CPU | Runs | Runs | Yes, once after both freezes |

## Scheduling

On Kaggle, WSSS and fully supervised may be separate private/offline jobs and
run concurrently when two accelerator allocations are available. They do not
share checkpoints, predictions, or supervision.

On a single A100, run them as independent sequential jobs. Literal same-GPU
concurrency adds VRAM contention and does not improve the scientific design.
The two frozen output bundles are joined only by the final evaluator.

## Hardware invariance

The scientific configurations do not change with hardware. T4x2 uses
DataParallel where supported; one A100 uses one device. Full retraining may
not be bitwise identical across GPU architectures, so the untouched test must
use the validation-selected checkpoints bound into the final lock.

## External assets

The repository contains source, fixed configuration, hashes, tests, and the
execution protocol. The operator supplies BTXRD and exact external weights
matching `artifacts/final_pipeline/final_run_config.json`.
