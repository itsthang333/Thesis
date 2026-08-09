# X12 matched online-inference efficiency, seed 42

These rows measure the online U-Net student inference and prediction-freeze
path on the same Kaggle `Tesla T4 x2` environment. Each run uses three warm-up
iterations followed by all 371 canonical validation images. Spatial validation
annotations and test images were not read.

The figures intentionally exclude offline pseudo-label generation. Therefore,
they compare the deployed students and the matched fully supervised U-Net, not
the total training or pseudo-mask-generation cost. Direct Rich Gallery and YOLO
are reported separately because their online computation graphs differ.

The five raw JSON receipts bind the exact checkpoint, prediction manifest,
canonical split, X4 protocol, device names, timing and CUDA memory counters.
