# Post-affinity BTXRD consumer design (draft, not a protocol)

Status: conditional engineering design prepared while
`rad_dino_affinity_decoder_probe_val_v1` is running. It does not authorize
pseudo-mask generation, consumer training, validation selection, or test
access. A new immutable protocol may be created only after the probe output is
independently audited and its all-required mechanistic gate passes.

## Evidence that the old consumer recipe should not simply be repeated

The current image-label-only pipeline trained a 448 px ResNet18 U-Net on hard
CAM/SAM pseudo masks. Its train positive Dice against those pseudo targets
reached `0.73817187`, but frozen validation Dice was only `0.23001987`;
small/medium/large were `0.07100142/0.41340304/0.32691688`. The pseudo masks
themselves reached only `0.23433922` validation Dice. This is direct evidence
of fitting a noisy hard target without learning transferable tumor shape.

The same consumer family trained with real masks reaches
`0.49513170` overall and
`0.32895493/0.66244178/0.69370336` by subgroup. Therefore the principal gap
is spatial supervision quality and its use, not an established inability of
the consumer architecture to represent the task.

Repeating binary-mask resizing plus weighted BCE/Dice would preserve the
measured bottleneck. Changing only threshold, loss weight, early stopping, or
post-processing after seeing validation GT would not be an admissible
solution.

## Conditional primary experiment

If and only if the affinity-decoder probe passes its frozen gate:

1. Freeze its final checkpoint and generate one continuous `320x320` map for
   every one of the 2,981 clean-train images. Use the exact inverse-square
   geometry already audited. Image-level normal cases receive an explicit
   zero teacher. No train polygon, validation image, validation label, or test
   record may enter this stage.
2. Hash every continuous map and its manifest before consumer construction.
   Preserve probabilities rather than converting the teacher immediately to
   one hard mask. Record score distributions separately for image-level
   normal and tumor training cohorts.
3. Train the proven ImageNet-initialized 448 px ResNet18 U-Net as a
   partial/soft-label consumer:
   - balanced soft BCE on teacher probabilities in confident regions;
   - explicit all-background supervision for image-level normal images;
   - image-label SmoothMax BCE on every output map, preventing a positive
     image from collapsing to empty;
   - edge-aware local consistency using only radiograph intensity and frozen
     RAD-DINO/decoder affinity, never spatial GT;
   - weak/strong-view consistency with an EMA teacher, with geometric
     transforms applied identically to images and soft targets.
4. Treat ambiguous teacher pixels as unlabeled rather than false background.
   Foreground/background confidence rules must be fixed from clean-train
   normal/tumor score distributions or deterministic ranks before validation
   evaluation. Do not choose them from validation Dice.
5. Use a fixed optimization horizon and final checkpoint, or an
   image-label-only training/holdout proxy declared in advance. Do not inspect
   validation masks every epoch for checkpoint selection. Generate and hash
   all 371 validation predictions before the sole spatial-GT evaluation.
6. Evaluate the unchanged mean per-positive-image Dice with complete misses
   and fixed `94/72/18` subgroups. Test remains locked. Operational success
   still requires all four thresholds:
   `overall >= 0.34024039`, `small >= 0.17895493`,
   `medium >= 0.51244178`, `large >= 0.49370336`.

## Why soft/partial supervision is the primary change

The affinity probe is trained to produce a continuous shape hypothesis, while
its p90/p95/p97/p99 masks are only nonselective diagnostics. Turning one of
those diagnostics into a dense hard target would declare every omitted tumor
pixel background and repeat the old consumer's confirmation bias. A soft
teacher retains ranking information; confidence masking prevents uncertain
regions from dominating; image-level BCE preserves the only task label
actually supplied; EMA consistency lets the student learn image statistics
without inventing spatial ground truth.

The first consumer should keep the proven ResNet18 U-Net so the experimental
change is the supervision interface. A stronger decoder/transformer consumer
may follow only under a separate predeclared arm; it should not be introduced
simultaneously with unvalidated pseudo-label rules.

## Fail branch

If the affinity probe fails any required gate, do not generate train pseudo
labels from it and do not train this consumer. Retain the failure as mechanism
evidence. The next architecture must improve learned spatial representation
or obtain a stronger image-label-only teacher; another validation-time
threshold/selector sweep is excluded by the accumulated CAM/SAM, AdvCAM,
S2C, dense-MIL, INSIGHT, and nominal-memory evidence.

## Required implementation work after a pass

- prediction-first train-map generator with checkpoint/protocol/source hashes;
- soft-map manifest schema distinct from the existing binary PNG schema;
- dataset loader that returns probabilities plus confidence masks without
  opening train annotations;
- partial soft BCE, image-level BCE, affinity consistency, and EMA tests;
- source-order audit proving train/validation prediction freeze precedes any
  validation GT loader;
- a new protocol frozen before Kaggle execution and a new independent output
  auditor.

