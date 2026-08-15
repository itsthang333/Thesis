BTXRD FINAL METHOD CHECKPOINT PACKAGE

This folder contains the complete frozen model assets used by the final
Rich Gallery G1 inference pipeline and the initialization weight needed to
retrain its DenseNet121 classifiers.

Contents
--------
classifiers/classifier_320_binary.pt
  Binary DenseNet121 classifier used by the 320 px LayerCAM branch.

classifiers/classifier_448_binary.pt
  Binary DenseNet121 classifier used by the 448 px LayerCAM branch.

g1_selector/rad_dino_mask_bag_mil.pt
  Trained G1 candidate selector.

sam_vit_b/sam_vit_b_01ec64.pth
  SAM ViT-B proposal generator checkpoint.

biomedclip/
  Complete local BiomedCLIP snapshot: model weight, OpenCLIP configuration,
  BiomedBERT architecture configuration, tokenizer files, and license.

rad_dino/
  Complete local RAD-DINO snapshot: model weight, model configuration, and
  image preprocessor configuration.

training_initialization/densenet121-a639ec97.pth
  ImageNet DenseNet121 initialization used when retraining classifiers.

Integrity
---------
Run VERIFY_SHA256.ps1 from PowerShell. Every line must report OK.

This package intentionally excludes SAM ViT-L, SAM ViT-H, MedSAM, SAM2,
SAM-Med2D, experimental students, and retired research checkpoints because
they are not part of the finalized method.
