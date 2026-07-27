# Relational MIL design for the BTXRD medium-selection bottleneck

Date: 2026-07-28  
Status: conditional design audit only; no source change or experiment

## Eligibility

This arm is eligible only after:

1. running version 6 reaches a terminal state and is audited;
2. geometry correction v3 is authorized, executed and audited;
3. the immutable candidate oracle still meets all operational goals;
4. corrected independent scoring remains below the consumer-entry tier,
   especially medium.

It reuses the exact hash-bound candidate gallery, geometry-v3 descriptors,
split, image labels, seeds, epochs, evaluator and prediction-first gates. It
does not regenerate proposals, change WTA masks, use subgroup identity, train
a consumer or open test.

## Why bag classification is insufficient

The deployable output is an instance decision: which candidate mask wins.
Improving only image-level AUROC can leave instance ranking unchanged or make
it worse. Jang and Kwon formalize that instance-level learnability does not
automatically follow from bag-level learnability in deep MIL. Accordingly,
BTXRD relational MIL must expose a score for every candidate, freeze its WTA
prediction before GT, and be promoted by localization—not bag AUROC alone.

Source: Jang and Kwon, *Are Multiple Instance Learning Algorithms Learnable
for Instances?*, NeurIPS 2024:
https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html

## Smallest causal architecture

Let `d_i` be the unchanged geometry-v3 descriptor of candidate `i`.

1. Embed each proposal independently:
   `h_i = GELU(Linear(LayerNorm(d_i)))`.
2. Produce a base instance logit `s_i` with the current independent head.
3. Select the critical instance `m = argmax_i(s_i)` over valid candidates.
   The index is detached; candidate ordering is unchanged.
4. Project `h_i` to a 128-dimensional query `q_i`. Compute DSMIL-style
   critical-instance affinity:
   `a_i = softmax_i(<q_i,q_m>/sqrt(128))`.
5. For every candidate form relational features:
   `[h_i, h_m, abs(h_i-h_m), h_i*h_m, log(N_valid*a_i + eps)]`.
6. A lightweight MLP predicts a residual `delta_i`; its final linear layer is
   zero-initialized.
7. The deployable score is `r_i = s_i + delta_i`.

Zero initialization makes the relational arm begin exactly at the independent
selector rather than changing score scale at initialization. The residual can
learn to promote candidates supported by a coherent proposal family or
suppress a critical false positive using evidence learned from image-level
negative bags.

The critical-instance relation follows DSMIL, which compares every instance
with the highest-scoring instance and uses a normalized, permutation-invariant
attention distribution:

Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning Network for
Whole Slide Image Classification with Self-supervised Contrastive Learning*,
CVPR 2021:
https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html

## Frozen training semantics

The causal arm changes the selector architecture only:

- apply the existing normalized SmoothMax to `r_i`;
- retain image-level BCE;
- retain the existing self-guided rule: every valid instance in a negative
  image is negative, while only the detached current winner in a positive bag
  receives a positive instance target;
- retain original/flip candidate alignment and consistency on final logits;
- retain optimizer, learning rate, batch construction, loss weights, epochs
  and seed family from geometry v3.

No auxiliary DSMIL bag head, contrastive pretraining, pseudo instance label,
mask overlap target or validation-derived weight is added. These would confound
the causal question.

## Required diagnostics

Before optimizer construction, save a GT-blind audit of:

- valid candidate counts by train/validation and image label;
- base critical-index agreement between original and aligned flip;
- critical-affinity entropy and effective neighbor count;
- descriptor/query finiteness and exact candidate order;
- zero-initialized equality `r_i == s_i` to numerical tolerance.

After prediction freeze and only then GT evaluation, report:

- all 371 predictions and all 184 positive cases;
- overall/small/medium/large Dice with complete misses;
- paired 10,000-bootstrap deltas versus geometry v3;
- recovered and lost overlaps;
- candidate-oracle identity;
- image-level AUROC only as a secondary diagnostic.

Promotion cannot rely on higher image-level AUROC. At minimum it must satisfy
the inherited selector/consumer-entry gate, show positive overall paired
evidence, not decrease any subgroup mean or increase subgroup misses, and
materially improve medium. Exact numerical thresholds must be frozen in an
execution protocol before the job and cannot be revised after GT.

## Separated contingencies

- If the GT-blind candidate-count audit shows strong bag-size imbalance, a
  per-bag-normalized negative instance loss is a separate future arm. Do not
  bundle it with relational scoring.
- If critical-instance flip agreement is poor before training, reject this
  exact hard-critical design and predeclare a soft-critical alternative;
  do not tune it using validation GT.
- If geometry v3 already passes entry, relational MIL is unnecessary before
  the separately gated consumer.
- If the oracle fails a subgroup, no selector architecture can repair absent
  proposal support; return to proposal generation instead.

## Compute

With at most 81 candidates and a 128-dimensional query, critical-instance
affinity is linear in bag size and negligible beside RAD-DINO caching. The
heavy descriptor cache remains Kaggle T4x2 work. Selector training may run on
one T4 after the two-GPU cache is complete; DDP would add overhead without
changing scientific output.
