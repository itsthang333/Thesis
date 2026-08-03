# Method

## Learning and inference setting

The method is prompt-free binary image-label-only weakly supervised semantic
segmentation. Training uses only whether an image is tumor or normal. Spatial
polygon annotations are isolated behind the validation or final-test
evaluator. At inference, the binary image label is assumed available for the
CAM target and normal/tumor handling; this assumption must be stated in the
thesis and any comparison table.

## Candidate supply

For each image, an offline stage constructs a rich proposal gallery from three
complementary sources:

1. **LayerCAM-320**: multi-layer DenseNet121 LayerCAM components converted to
   SAM point/box proposals.
2. **Classifier-448**: higher-resolution binary-classifier evidence and SAM
   proposals.
3. **External saliency**: frozen prediction-first BiomedCLIP saliency converted
   to SAM proposals.

The gallery batches are merged without using spatial GT. Exact duplicate masks
are removed, while immutable source and upstream-score metadata remain bound to
each mask. The fixed cap is 243 candidates per image.

## G1 candidate selector

G1 encodes each candidate using frozen RAD-DINO features aligned between image
and mask coordinates. Its representation combines inside, local-context,
contrast, and geometry features. The mask-bag MIL objective uses only image
labels: normal images provide certain negative bags and tumor images provide
positive bags. The chosen checkpoint is frozen by SHA-256.

Before any polygon is opened, G1 produces a logit for every candidate.

## Fixed fusion

Raw G1 and upstream scores have unrelated scales. Within each image, each score
vector is therefore converted to an average-tie percentile rank:

\[
r_i(s)=\frac{\operatorname{rank}_{avg}(s_i)}{N-1}.
\]

The fixed candidate score is

\[
S_i=0.5\,r_i(s_i^{G1})+0.5\,r_i(s_i^{upstream}).
\]

The maximum is selected. Ties are resolved first by the raw G1 logit and then
by the lower immutable candidate index. There is no per-image GT area,
test-tuned threshold, GT-derived routing, or post-test morphological rescue.

## Evaluation boundary

`freeze_final_rich_gallery.py` freezes the complete cohort before spatial GT.
For validation this means 371 choices; for final test it means 373 choices.
`evaluate_final_rich_gallery.py` then verifies all frozen hashes and opens only
the tumor polygons (184 validation or 187 test) to compute mean per-image Dice
and IoU. A test run additionally requires the immutable schema-v4 lock created
by `freeze_final_test_protocol.py` from a clean committed tree.

## A100 execution

An A100 changes only execution resources. It does not change image sizes,
candidate cap, checkpoints, scores, fusion weights, tie-breaking, or the
evaluation definition. Frozen RAD-DINO encoding uses all visible GPUs through
DataParallel when T4x2 is available and a single device on a 48 GB A100. The
selector itself remains on the primary device. This placement is a runtime
choice, not an A100-specific scientific variant.

The untouched final test uses the already frozen validation-selected G1
checkpoint. Retraining on different GPU architectures is supported for method
reproduction, but bitwise-identical training weights are not promised across
T4 and A100 kernels.

## Fully-supervised comparison

The comparison track is a ResNet18-U-Net trained at 448 px from train-split
polygon masks. It uses AdamW (learning rate and weight decay 1e-4), batch 8,
positive-class weight 10, seed 42, at most 30 epochs, and validation-only early
stopping with patience 10. Its final binary threshold 0.20 was locked on
validation. Its checkpoint is explicitly marked `comparison_only=true` and
`wsss_eligible=false`; neither its weights nor predictions can enter candidate
generation or G1 selection.

Before final evaluation, `freeze_fully_supervised_predictions.py` creates one
immutable mask per test image without opening polygons. The joint evaluator
loads the already-frozen WSSS choice and fully mask, opens each tumor polygon
once, and calculates the two metric families from that same annotation pass.
