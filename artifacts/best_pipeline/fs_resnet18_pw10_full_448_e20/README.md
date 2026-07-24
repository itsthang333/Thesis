# Current Best Pipeline

Locked pipeline: `fs_resnet18_pw10_full_448_e20`.

This is the complete snapshot of the pipeline with the best validated mean tumor-only Dice currently available: **0.4951316963** at threshold 0.2, selected only on validation. This does not meet the strict target of `>0.50`. The test partition has not been evaluated.

The snapshot contains source code, tests, the immutable split manifest, the epoch-20 checkpoint, the epochs 1-30 training log, all validation evidence, the locked configuration, and a per-file hash manifest. The checkpoint is retained as an exact hard link or copy with SHA-256 `05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31`.

Usage rules:

- Run every new candidate outside this snapshot directory.
- Promote a candidate only when it uses the same metric and split protocol, keeps test locked, and achieves a credibly higher validation mean tumor-only Dice.
- Purge code, runtime state, and heavy artifacts for rejected candidates; do not alter the frozen winner model or evidence.
- Never choose a threshold or redefine a metric on test.
- Documentation-only repairs must be recorded in `FILE_MANIFEST.csv` and followed by a verifier run.

Post-freeze correctness repair on 2026-07-24: the deployment loader now
instantiates the checkpoint-declared `ResNet18UNet` and records its architecture
in inference metadata. This changed no checkpoint, training path, evaluator,
metric, validation evidence, threshold, or test state. The repair is covered by
three fail-closed architecture tests and is recorded in `pipeline_lock.json`.

Verify the snapshot:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python verify_pipeline.py
python -m unittest discover -s tests -p 'test_*.py'
```

`verify_pipeline.py` fails closed if a file hash, sample count, split-isolation check, best epoch, metric, threshold-selection provenance, checkpoint identity, or test lock is wrong.
