# Rich-gallery false-positive rectification B3 design

Status: evidence-backed successor hypothesis only. It is not an experiment
claim, protocol, binding or launch. It may not run before B2 has a terminal,
independently audited result and the central coordination log is reread.

## Why this is downstream of the 0.288729 pipeline

The inherited pipeline is the immutable rich proposal union plus the frozen
equal-rank G1/upstream selector. Its exploratory validation Dice is
`0.28872949`, while the same gallery oracle is `0.52829833`. The collaborator
decomposition attributes `70.29%` of selector regret to within-selected-source
ranking. Its 49 complete misses are all gallery-recoverable, with median oracle
rank 95. Therefore B3 must not regenerate proposals or rerun G0/G1/G2.

The dominant failure is not one global area error. Small lesions are already
over-segmented by a median factor `14.603`, while large lesions are
under-segmented at `0.382`. Global and top-10 geometric consensus both enlarged
stable bone anatomy: the latter reduced misses but lowered small Dice by
`0.034958` with a strictly negative confidence interval. This identifies a
semantic false-positive problem: repeated anatomy is plausible foreground but
is not tumor.

B2 is the necessary first semantic control. It adds a class-aware BAS map to
the frozen G1/upstream ranks. B3 must not be interpreted as proven better and
must not be launched in parallel. Its only purpose at this stage is to freeze
the strongest distinct response if B2 later shows that BAS contains useful but
anatomically confounded tumor evidence.

## Transfer from FPR, and what is not transferred

Chen et al., *FPR: False Positive Rectification for Weakly Supervised Semantic
Segmentation*, ICCV 2023, observes that activations for an absent class expose
the background cues that co-occur with that class. FPR forms positive and
negative feature prototypes, then uses region contrast and pixel rectification
to suppress foreground pixels closer to negative prototypes. Paper and
official implementation:

- https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html
- https://github.com/mt-cly/FPR
- https://raw.githubusercontent.com/mt-cly/FPR/master/step/train_fpr.py

The binary BTXRD analogue is unusually direct: every image-label-normal train
image is an absent-tumor image. Its highest tumor activations are therefore
task-specific false-positive bone/anatomy evidence. This differs from rejected
R1/N1 normal anomaly scoring: R1 learned a bag residual that amplified proposal
count, while N1 proposed generic candidate distance to normal prototypes. B3
would use BAS task features only to identify and suppress false-positive
activated pixels before candidate scoring; it would never use count, source or
generic anomaly distance as a tumor score.

The official repository's validation-mIoU model selection and threshold loop
are explicitly forbidden here. No FPR coefficient, cluster count, activation
threshold, epoch, rank weight or validation-mask selection may be copied or
tuned post hoc. The paper supplies a mechanism hypothesis, not BTXRD evidence.

## Conditional fixed mechanism

If and only if a terminal B2 audit shows non-degenerate, complementary BAS
evidence but its losses remain associated with large anatomical activation,
the B3 protocol may freeze the following single correction before launch:

1. Reuse the exact final B2 BAS checkpoint; do not retrain or select an epoch.
2. On train images only, extract one L2-normalized stage-3 regional feature per
   original/flip view from pixels where the per-image BAS tumor activation is
   above the fixed FPR value `0.1`.
3. Tumor-label train regions form the positive pool. Normal-label train regions
   form the absent-tumor false-positive pool. Give every train image and view
   equal total mass; source identity and proposal count are absent.
4. Form exactly ten deterministic spherical prototypes in each pool, matching
   FPR's fixed `num_cluster=10`; seed 42, no K sweep.
5. For each validation feature pixel, find its closest positive and negative
   prototype. A pixel is retained exactly when positive distance is not larger
   than negative distance, matching FPR's false-positive criterion. The
   rectified map is the frozen BAS activation multiplied by that binary retain
   mask; there is no learned calibration temperature.
6. Score every immutable rich-gallery candidate by the same coverage/purity
   harmonic used in B2. The sole B3 selection is the equal percentile-rank mean
   of G1, upstream and rectified-BAS evidence.

The matched comparator is the raw BAS semantic arm from the same B2 checkpoint,
not a regenerated 0.288729 control. This isolates rectification from BAS itself.
No relation, consensus, top-k restriction, morphology, lesion-size inference,
source router or candidate deletion is allowed.

## Gates and falsification

Before validation segmentation GT, an independent auditor must reproduce all
train-region pools, equal image/view weights, 20 prototypes, pixel retain maps,
candidate scores, choices and physical masks. It must also show that the
rectifier is neither identity nor collapse on image-label-positive validation
maps. Exact non-degeneracy bounds must be frozen from train-only diagnostics in
the future protocol, not chosen from validation Dice.

Only then may a post-freeze evaluator compare rectified B3 against raw B2 and
the inherited control. A mechanism gain requires a positive overall paired
complete-group bootstrap lower bound, no subgroup Dice decrease and no miss
increase. Full promotion still requires Dice
`0.34024039/0.17895493/0.51244178/0.49370336`. Failure retires the BAS/FPR
semantic family; it does not authorize an FPR threshold/K/loss-weight sweep or
a geometric rescue.

## Safety and coordination

All fitting uses train image labels only. Validation choices are physically
frozen before segmentation GT. Consumer training and BTXRD test remain locked.
Heavy compute is Kaggle T4x2/P100 only. Before any B3 real-data step, fetch both
branches, read the complete central log, verify B2 terminal status, register a
distinct `ĐANG LÀM` claim and push it to the coordination branch.
