# Reproducibility

## External assets

- BTXRD images, metadata, and annotations.
- DenseNet121 checkpoints for the 320 and 448 candidate supplies.
- SAM v1 ViT-B checkpoint.
- Frozen BiomedCLIP and RAD-DINO snapshots.
- Frozen G1 checkpoint.

The final test lock records the path, byte size, and SHA-256 of every asset.
Paths may differ between machines only before the lock is created.
The authoritative expected hashes and fixed hyperparameters are in
`artifacts/final_pipeline/final_run_config.json`.

## Stage environments

Install the CUDA build of PyTorch first. The reference validation environment
was Python 3.12.13, PyTorch 2.10.0+cu128, and CUDA 12.8. Then use separate
environments:

```bash
# Candidate generation, BiomedCLIP and SAM
python -m pip install -r project/requirements-candidate.txt

# RAD-DINO cache, G1 training and final candidate scoring
python -m pip install -r project/requirements-g1.txt

# Fully-supervised ResNet18-U-Net comparison
python -m pip install -r project/requirements-fully.txt
```

Kaggle Internet=Off must receive Segment Anything commit
`6fdee8f2727f4506cfbbe553e23b895e27956588` as an attached Dataset/source
tree and add it to `PYTHONPATH`; it must not attempt the online Git install.

## Development reproduction

The canonical stages are:

1. Verify the canonical split manifest.
2. Generate the LayerCAM-320, classifier-448, and BiomedCLIP supplies.
3. Run `run_rich_gallery_candidate_supply.py` in `anchor` and `addition`
   modes.
4. Merge the two galleries with `merge_frozen_candidate_galleries.py`.
5. Build the G1 selector cache and train G1 using image labels only.
6. Score every candidate with `score_final_rich_gallery.py`.
7. Freeze equal-rank choices with `freeze_final_rich_gallery.py`.
8. Evaluate the frozen validation choices with
   `evaluate_final_rich_gallery.py --split val`.

The matched fully-supervised track is run independently with:

```bash
python project/run_fully_supervised_comparison.py \
  --dataset-root /data/BTXRD \
  --split-manifest /data/split_manifest.csv \
  --resnet18-weight /models/torch_home/hub/checkpoints/resnet18-f37072fd.pth \
  --output-dir /results/fully_supervised
```

The ResNet18 file must have SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Kaggle may run this as a second private job while a WSSS job is active. A
single A100 should run the jobs independently rather than sharing VRAM.

After both validation tracks complete, run
`freeze_fully_supervised_predictions.py --split val` and invoke
`evaluate_final_rich_gallery.py --split val` with
`--fully-prediction-root` and `--expected-fully-freeze-sha256`. The evaluator
writes `per_image.csv`, `fully_supervised_per_image.csv`, `summary.json`, and
`comparison.csv`. Thus the comparison artifact is produced automatically from
the two frozen tracks rather than assembled manually.

The retained CLIs expose exact arguments through `--help`; hashes link every
stage. The refactored code reproduces validation Dice exactly:
`0.28872948670665205`.

Resource placement is automatic: one visible A100 uses a single encoder,
whereas two visible T4 GPUs use DataParallel for the frozen encoder. No method
hyperparameter changes with hardware.

## Final A100 test

The final test is not another experiment. It reuses the validation-selected
method without tuning. Follow [TEST_PROTOCOL.md](TEST_PROTOCOL.md): commit the
clean tree, create the schema-v4 lock, generate and score test candidates
without polygons, freeze all 373 choices, then invoke the spatial evaluator
once. Save the lock, stdout/stderr, manifests, hashes, and result together.

## Preflight

```bash
python -m py_compile project/*.py project/evaluation/*.py
python -m pytest -q tests
```

The preflight uses no test images or annotations.
