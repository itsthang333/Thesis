# Research log

## 2026-07-22

- Read and visually verified the 30-page mission specification.
- Inventoried repository, dataset, historical notebooks/checkpoints, GPU, Python environment, and cached pretrained weights.
- Built a checksum-bearing, group-aware clean split. Excluded 21 exact duplicate images and verified zero group/hash overlap across clean partitions.
- Reconstructed the legacy image-level split and found definitive leakage: six cross-partition exact duplicate groups and 426 cross-partition inferred case groups.
- Re-evaluated later pseudo-supervised and fully supervised checkpoints with tumor-only Dice on their exact legacy validation cohort. Results: 0.17756 and 0.07401 respectively; both are invalid as final evidence.
- Identified historical all-image checkpoint selection and early stopping as major confounders.
- Added and smoke-tested a pretrained ResNet-18 U-Net with architecture-aware checkpoint loading.
- Added validation-only threshold sweep support and an explicit guard against test-set sweeping.
- Corrected notebook bootstrap tests to accept both valid nbformat source encodings.
- Started clean fully supervised upper-bound run `artifacts/runs/fs_resnet18_clean_seed42`.

Decision rule: continue to binary classifier and CAM/SAM ablations only after the supervised upper bound establishes that the segmentation backbone can reach the target on the clean validation cohort.

- Supervised ResNet-18 U-Net with BCE positive weight 20 peaked at epoch 4: validation tumor Dice 0.31840 at threshold 0.5.
- A validation-only sweep selected threshold 0.65, but improved tumor Dice only to 0.32627 (normal specificity 0.65241); therefore calibration alone does not explain the gap.
- Paused positive weight 20 after epoch 5 because validation specificity deteriorated, but reclassified the run as **inconclusive rather than rejected**: five epochs cannot establish convergence. Started an otherwise identical positive-weight-10 ablation and retained the resumable checkpoint.
- Adopted a promote-or-purge rule: rejected experimental implementations and their CLI/config/test/dependency/debug surface are removed before final hand-off; compact metric evidence remains.
- Adopted a successive-halving learning-curve gate. A short run is a screening result only. Promotion uses best tumor-only Dice, recent Dice/loss slopes, train-validation gap, calibration/specificity stability, and a validation-only threshold sweep. Early rejection requires both no recent improvement and evidence of saturation/divergence; otherwise the run is paused/inconclusive and may receive more epochs. Final claims require a reasonably converged checkpoint, not an extrapolation.

- Quantitative potential check for the paused weight-20 run: train positive Dice rose monotonically from 0.1980 to 0.3998 over epochs 1-5; validation best rose from 0.2021 to 0.3184 and occurred at epoch 4, followed by only one worse epoch. Training therefore had not saturated and the original early rejection was not supported.
- The first weight-10 attempt was interrupted before epoch 1 by a Windows reboot at 21:58:09 (confirmed from OS boot time), with no Python traceback or metric row. The incomplete artifacts were removed and the identical run was restarted from scratch at 22:03; the interruption is not treated as model evidence.
- The restarted weight-10 run completed epochs 1-3 (tumor Dice 0.22867, 0.29270, 0.25668) before a second system failure. The training traceback records `CUDA error: an illegal memory access was encountered`; Windows records `nvlddmkm` event 14 at 22:31:12 followed by bugcheck `0x133` and reboot at 22:33:40. This is a hardware/driver interruption, not a learning-curve rejection.
- The epoch-3 resume checkpoint was loaded successfully on CPU: epoch 3, best epoch 2, best tumor Dice 0.292699, 182 model tensors, optimizer state present, SHA-256 `28710ef9a1210f38580e75c4ec0960fddc099d009f5b3fd6abd9296e900ef3c0`. The run is paused as `hardware_interrupted_resume_ready` until the GPU stack is remediated or a safer CPU/GPU configuration is authorized.
- Environment evidence at the interruption: RTX 3050 Ti Laptop GPU, NVIDIA driver 462.62 (2021-05-11), driver-reported CUDA 11.2, versus the installed PyTorch 2.1.2 CUDA 11.8 build. Driver 462.62 satisfies CUDA 11.x minimum minor-version compatibility, so the causal claim is limited to observed instability of this machine's driver/GPU stack; it is not described as a formal incompatibility.
- With explicit user authorization, downloaded NVIDIA Studio 610.47 WHQL from `us.download.nvidia.com` (978,481,008 bytes; SHA-256 `59ac4a1659664aad0a6fc525e5df99b3fa76887bde663f9e36e0e7ebb5dba937`; Authenticode `Valid`, NVIDIA Corporation). Exported signed rollback package `oem37.inf` for driver 462.62, then installed clean with automatic reboot disabled. Before the planned reboot, PnP reported signed driver `32.0.16.1047` / `oem49.inf` and `nvidia-smi` reported KMD 610.47 with the GPU healthy.
- After reboot, CUDA smoke tests passed, including 80 ResNet18 U-Net batch-8 320 px AMP forward/backward iterations. The weight-10 run then resumed from epoch 3 and completed epochs 4-5: tumor Dice 0.317795 and 0.313586; validation loss continued down to 0.529034 and normal specificity rose to 0.540107.
- Sustained training still triggered `nvlddmkm` events 14/153 at 23:21:09, followed by the same bugcheck `0x133` and reboot at 23:26:57 despite Studio 610.47. This falsifies the hypothesis that a current driver alone resolves the local machine instability. Local batch-8 GPU training is no longer considered safe.
- The epoch-5 checkpoint loads successfully on CPU: epoch 5, best epoch 4, best tumor Dice 0.317795, global step 1865, 182 tensors, optimizer state present, SHA-256 `d0190cd5181e0bcbbc93a58ac9f2ab6004739cd40ed8bef893d28f290596ac97`.
- Kaggle CLI is authenticated and reports 20.25 GPU-hours remaining. A private T4 kernel is prepared locally for controlled full-320, tumor-only-320, and winner-448 experiments; external upload/run awaits explicit authorization because the bundle contains local source, split manifest, checkpoint, and logs.

- The full-validation subgroup audit of the weight-20 checkpoint identified a size bottleneck: small lesions below 1% area comprise 94/184 tumor images and score 0.08644 Dice, versus 0.52043 for 72 medium lesions and 0.72161 for 18 large lesions. This prioritizes resolution/valid ROI localization after the weight and tumor-only controls.

- Pixel-scale audit of the small-lesion subgroup at 320 px: area min 7, P25 54, median 96.5, P75 207, max 970 pixels; median equivalent diameter 11.1 px. At 448 px the same proportional median would be about 189 pixels / 15.5 px diameter. This justifies, but does not prejudge, a controlled 448 px experiment.

## 2026-07-23

- The private Kaggle screening kernel completed on the locked clean validation split; no test record was loaded or evaluated. The first cloud attempt failed before training because PyTorch 2.6+ defaults `torch.load` to `weights_only=True`; trusted pipeline-owned checkpoints now opt into `weights_only=False`, restoring the prior behavior without weakening manifest/checkpoint provenance checks. The full suite passed after the compatibility fix.
- Epoch-10 validation results using the unchanged mean-per-image tumor Dice endpoint:

| Run | Selected threshold | Overall Dice | Small <1% Dice | Medium 1-5% Dice | Large >=5% Dice | Normal specificity |
|---|---:|---:|---:|---:|---:|---:|
| Full train, 320 px | 0.45 | 0.38501 | 0.17463 | 0.58898 | 0.66776 | 0.51872 |
| Tumor-only train, 320 px | 0.60 | 0.37715 | 0.14445 | 0.59476 | 0.72188 | 0.58289 |
| Full train, 448 px | 0.75 | **0.41832** | **0.24904** | 0.58973 | 0.61668 | 0.62567 |

- The 448 px change improved the predeclared small-lesion subgroup by +0.07441 Dice over full-320 while leaving medium-lesion performance essentially stable. At threshold 0.5 it also increased tumor overlap detection from 0.7065 to 0.7717 and reduced complete misses from 30 to 13. This is direct evidence that spatial fidelity improves localization rather than merely inflating predicted masks.
- The 448 learning curve remains non-saturated: validation tumor Dice at threshold 0.5 rose from 0.34137 (epoch 5) to 0.41068 (epoch 8) and 0.41613 (epoch 10), while validation loss fell to 0.49360. It is promoted to a convergence run from epoch 10 with maximum epoch 35 and patience 10; no result is projected or claimed before measurement.
- Tumor-only training is rejected: it trails full-320 overall by 0.00786 and the small subgroup by 0.03018. Its CLI option, dataset/factory filter, dedicated test, source bundle checkpoint, and obsolete kernel branch were removed. Only the compact rows above and evaluator artifacts remain as audit evidence. The source suite now has 46 passing tests.
- The verified 448 resume checkpoint contains epoch 10, best epoch 10, best validation tumor Dice 0.4161256199 at threshold 0.5, 182 model tensors, optimizer/scaler and RNG states, and split SHA `85511ee1...`; file SHA-256 is `8278865eba7e07e0f1e327e7f6a853e3ee770a30a5a57e35438bfe5f6c458bd3`.
- A clean private-Kaggle payload is prepared locally: 45 files / 233,357,821 bytes, containing project source, the immutable split manifest, the verified 448 checkpoint, and its 10-row training log; it contains no radiographs, credentials, caches, or rejected 320 checkpoint. Upload is pending an explicit payload-specific confirmation required by the execution safety layer.
- A 10,000-resample paired per-image bootstrap confirms the resolution gain: selected-threshold overall delta is +0.03331 (95% CI +0.00206 to +0.06492), and the small-lesion delta is +0.07441 (95% CI +0.02491 to +0.12531). At fixed threshold 0.5, the corresponding deltas are +0.03128 and +0.05906, so the conclusion is not an artifact of choosing different global thresholds.
- At fixed 0.5, 448 recovers overlap on 24 tumor images that had zero overlap at 320, while losing overlap on 12; within the small subgroup it recovers 19 and loses 9. The selected 0.75 threshold improves precision/specificity but leaves 41/94 small cases with zero overlap, so localization remains the dominant residual failure.
- The small-lesion gain is heterogeneous: the third area quartile gains +0.18361, whereas the smallest two quartiles have uncertain +0.04429 and +0.03526 gains. Hand and foot remain particularly weak, despite 68 and 47 small-lesion train examples respectively. Thus pure resolution scaling is insufficient by itself.
- Train contains 754/1,488 tumor images below 1% area (50.7%); the fast polygon-area estimator agrees 100% with the validation raster subgroup assignment. Small lesions are not globally scarce, and the current soft Dice is already averaged per image. If convergence remains below target, higher-potential controlled candidates are foreground-aware zoom/crop training paired with inference-time fixed tiling, or the predeclared Focal Tversky loss; naive small-vs-large oversampling is not preferred.
- If medium and large subgroup performance stayed fixed at the epoch-10 448 values, small-lesion Dice would need approximately 0.409 to reach overall 0.50. This is a leverage calculation for experiment allocation, not a projected result. Compact evidence: `artifacts/small_lesion_analysis/paired_320_448.json` and `train_small_distribution.json`.
- Test remains locked. All numbers in this section are clean-validation diagnostics, not final test claims.

- The promoted 448 px continuation resumed exactly from epoch 10 and early-stopped at epoch 30 after 10 non-improving epochs. The best checkpoint is epoch 20, with validation positive-Dice 0.490171 at the fixed 0.5 training criterion. Full evaluator results are 0.489941 tumor-only Dice at threshold 0.5 (group-bootstrap 95% CI 0.437213-0.540560) and **0.495132** at the validation-selected threshold 0.2 (95% CI 0.442827-0.546114). This is the current best clean-validation result, but it remains below the strict `>0.50` target.
- At selected threshold 0.2, subgroup Dice is 0.328955 for 94 small lesions below 1% area, 0.662442 for 72 medium lesions, and 0.693703 for 18 large lesions. Small-lesion overlap detection is 0.712766 with 14 complete misses. Normal-image empty-mask specificity is 0.556150; the lower threshold trades specificity for recall and is therefore reported explicitly rather than hidden.
- Relative to the epoch-10 448 checkpoint, the selected-threshold paired gain is +0.076816 Dice (10,000-resample group bootstrap 95% CI +0.047631 to +0.106773). The small subgroup gains +0.079916, recovering overlap in 16 cases and losing it in 2. This confirms that longer training produced a real gain, while best-at-20 followed by patience exhaustion at 30 is evidence of practical saturation.
- The promoted checkpoint was downloaded and verified at 230,924,939 bytes with SHA-256 `05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31`. Its complete result bundle is saved under `artifacts/kaggle/convergence_448_v3/`; `convergence_summary.json` records the immutable split SHA, resume SHA, timestamps, environment, metrics, and `test_evaluated: false`.
- Decision: preserve epoch 20 as the current validation winner and proceed to a single controlled small-lesion inference ablation (full-image plus fixed blind tiling). Test remains locked.
- The complete winner pipeline is frozen at `artifacts/best_pipeline/fs_resnet18_pw10_full_448_e20/` and selected by `artifacts/best_pipeline/CURRENT.json`. After the deployment-loader correctness repair, a 68-file SHA manifest and fail-closed verifier pass; split isolation/counts, checkpoint, 184-tumor metric population, threshold provenance, and `test_evaluated=false` are all verified. Frozen source matches the worktree exactly across 40 project and 5 test files, no rejected tumor-only training code remains, and 49/49 tests pass inside the snapshot.

### Blind-tiling ablation

- Private Kaggle kernel `itsthang333/btxrd-small-lesion-tiling`, version 2, completed on the clean validation partition. The full-image baseline reproduced the frozen winner exactly: Dice `0.4951316963` at threshold `0.20`. The split SHA and checkpoint SHA matched the frozen contracts, and `test_evaluated=false`.
- The predeclared primary `blend50` candidate (50% full-image probability plus 50% uniform 3-by-3 blind-tile probability) selected threshold `0.40` and reached Dice `0.4953371473`, a paired delta of only `+0.0002054510`. The 10,000-resample paired group-bootstrap 95% CI was `-0.0132936291` to `+0.0149442548`, so the apparent gain is not credible.
- Small-lesion Dice rose from `0.3289549325` to `0.3441608344`, with six overlaps recovered and one lost, but medium Dice fell from `0.6624417784` to `0.6506621348`, large Dice fell from `0.6937033566` to `0.6635134985`, and normal empty-mask specificity fell from `0.5561497326` to `0.4705882353`.
- Diagnostic tile-only inference was clearly worse at Dice `0.4334699173`. Blind tiling without crop-aware training is rejected. `CURRENT` remains the epoch-20 full-image winner. Compact evidence is retained under `artifacts/kaggle/tiling_v1/`; the ablation summary SHA-256 is `55c7a85c2e1ad4011f410d30e08effb18a31cb129b705f019a66d172ae4e4b7c`. Candidate runtime source and metadata were purged after the decision.
- The next and only predeclared controlled ablation in this technique family is foreground-aware 60% crop training paired with the same blind 3-by-3 validation inference. Clean-train polygons may place/jitter training crops; validation targets never place tiles or ROIs. Test remains locked.

### Gate B — clean binary image classifier

- Private Kaggle T4 kernel `itsthang333/btxrd-clean-binary-classifier` completed on the frozen clean split. It used only the binary image-level `tumor` label, DenseNet121 with hash-checked ImageNet initialization, image size 320, AdamW at `1e-4`, fixed probability threshold `0.5`, and no polygon or mask input.
- Early stopping fired validly at epoch 18 after seven epochs without improving the epoch-11 checkpoint. The best clean-validation tumor F1 is `0.7833333333`; sensitivity `0.7663043478`, specificity `0.8128342246`, precision `0.8011363636`, AUROC `0.8654528017`, and AUPRC `0.8772278795`.
- The full 371-image validation confusion matrix is TP `141`, FP `35`, FN `43`, TN `152`. The group-bootstrap 95% intervals are `0.7042861410-0.8277822581` for sensitivity and `0.7405353143-0.8793969849` for specificity.
- The selected checkpoint SHA-256 is `f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5`; split SHA-256 is `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`. The run manifest records Kaggle Tesla T4, PyTorch 2.10.0+cu128, source hashes, `test_evaluated=false`, and all 371 per-image rows.
- Gate B is promoted to localization diagnosis, not directly to a segmentation claim. Gate-C kernel `itsthang333/btxrd-binary-cam-sam-gate-c` completed two predeclared validation protocols: supplied binary image label for training-pseudo-mask semantics, and classifier-predicted gating as a separate deployment diagnostic.
- In the supplied-image-label protocol, all 184 tumors are localized and all 187 known normals are explicit empty masks. End-to-end pseudo-mask Dice is `0.2023038352` (group-bootstrap 95% CI `0.1588296298-0.2467844435`). Mean foreground-support IoU/recall are `0.0703875111/0.6058501891`; point-hit rate `0.2705575974`; SAM oracle single-candidate Dice `0.3809186551`; selected Dice `0.2010835193`. Support loss is only `0.0209441901`, but image-only candidate-selection loss is `0.1588909457`, identifying the selector as the largest immediate bottleneck.
- In the predicted-gate diagnostic, classifier false negatives skip 43/184 tumors. Conditional CAM/SAM Dice on the 141 reached tumors is `0.2224065422`, but end-to-end Dice falls to `0.1704311003` (95% CI `0.1289388966-0.2110567873`); normal specificity is `0.8181818182`. This separates classifier-gate loss from downstream localization loss.
- Per-image Gate-C audit finds 78/184 tumors with an oracle SAM candidate Dice at least 0.50, but only 38/184 selected masks reach 0.50; 24 images have oracle Dice at least 0.50 while the selected mask remains below 0.20. Selection loss is at least 0.20 on 51 images, whereas support-clipping loss is at least 0.20 on only six. Small-lesion oracle/selected Dice is `0.2067609233/0.0810218360`; medium is `0.5519975194/0.3084612064`; large is `0.6060935752/0.3985615620`. Thus selector repair has immediate leverage, while small lesions will still require a later CAM/prompt improvement even under a perfect selector.
- The controlled global-top-1 ablation is rejected. It preserved the oracle pool exactly but reduced supplied-label pseudo-mask Dice from `0.2023038352` to `0.1777293956`; paired delta `-0.0245744396`, 95% CI `-0.0493509738` to `-0.0012575416`. Selected Dice fell to `0.1868790157` and selection loss rose to `0.1730954493`. Best-per-component union therefore contributes useful coverage and remains the baseline.
- The predeclared `prompt_hybrid` selector-family candidate is rejected. Its fixed component-local scoring/no-global-clip method reduced pseudo-mask Dice from `0.2023038352` to `0.1118186396`; paired delta `-0.0904851956`, independently reproduced 10,000-resample group-bootstrap 95% CI `-0.1242405321` to `-0.0558622391` across 167 tumor groups. Foreground support and oracle single-candidate Dice were identical to baseline (`0.0703875111` and `0.3809186551`), while selected Dice fell to `0.1313552106` and selection loss rose to `0.2495634445`. Compact artifacts are under `artifacts/kaggle/wsss_binary_cam_sam_prompt_hybrid_v1/`; comparison SHA-256 is `1d3416023e8d6e749a2131a3f694f374fef530e5aa600b42469a3aa984c4d5a8`; test was untouched.
- Horizontal-flip CAM TTA is promoted within Gate C. It raised supplied-label pseudo-mask Dice from `0.2023038352` to `0.2343392222`; paired delta `+0.0320353870`, independently reproduced 10,000-resample group-bootstrap 95% CI `+0.0085405304` to `+0.0573916861` across 167 tumor groups. Foreground IoU/recall improved to `0.0767800857/0.6510582698`, point-hit rate to `0.2957003457`, box recall to `0.7043623912`, oracle single-candidate Dice to `0.4090762905`, and selected Dice to `0.2336682193`. Small/medium/large final Dice improved from `0.0824283044/0.3097778654/0.3984243755` to `0.1121634039/0.3486037144/0.4153105265`. Comparison SHA-256 is `aa945e434b38a07df568841c3feb93d615a8aac757e701138613de5c9cbf74fd`; test was untouched.
- The current source's multiscale branch explicitly requires the legacy `tumor_type` classifier, so it was not run under the binary protocol. The final bounded Gate-C pool candidate retained TTA and added only `include_cam_candidate=true`. It is rejected: Dice `0.2306616510` versus TTA `0.2343392222`, paired delta `-0.0036775712`, independently reproduced 95% CI `-0.0084839943` to `+0.0005739561`; all lesion-size subgroups decreased. Comparison SHA-256 is `29704cbef73d5481cb4d33811a6775a972a3445ba424b3597d8e50b6e8a246a5`; test was untouched.
- Gate C is frozen as binary LayerCAM plus horizontal-flip TTA, SAM-only candidate pool, best-per-component/top-3 `coverage_mass_sam`, and support clip kernel 5. Kaggle kernel `itsthang333/btxrd-wsss-train-pseudo-v1` completed the full clean-train pseudo set. Manifest SHA-256 is `7b0b133e7bbff8fecb102159b1be41801b6c51199de549a3420978b13ea7c7e6`; all 2,981 PNGs were verified on Kaggle, with 1,463 non-empty/25 empty tumor masks and all 1,493 normals empty. Polygon prompt diagnostics, validation evaluation, and test evaluation were absent.
- Foreground-crop/tiling version 2 completed successfully after the narrow `pred_area_pixels` repair. Full-image candidate inference reached Dice `0.5155799970` at threshold `0.20`; blend50 reached `0.5268559713` at threshold `0.30`. However, paired deltas versus `CURRENT` were `+0.0204483007` (95% CI `-0.0123973802` to `+0.0527633776`) and `+0.0317242750` (95% CI `-0.0008387065` to `+0.0648922460`). Blend50 also reduced normal specificity from `0.5561497326` to `0.4171122995`. This is valuable supervised upper-bound evidence that the architecture can cross 0.50, but it is not a credible replacement for `CURRENT` and cannot count as the final WSSS result.
- Heavy training and inference remain on Kaggle. Local execution is limited to source tests, audit, artifact verification, and packaging.
- User-requested shutdown pause: local polling stopped while `itsthang333/btxrd-wsss-train-pseudo-v1` remained `RUNNING` on Kaggle. The downstream WSSS segmenter payload is prepared but deliberately unlaunchable until the completed pseudo-manifest SHA-256 replaces its fail-closed placeholder. Test remains locked.
- Resume after shutdown: pseudo generation was `COMPLETE`; every local audit check passed. The verified manifest SHA replaced the fail-closed placeholder, 4/4 payload tests and Python compilation passed, and `itsthang333/btxrd-wsss-segmenter-v1` version 1 was launched on Kaggle. Test remains locked.
- Gate-D version 1 failed before epoch 1 because `validate_pseudo_mask_manifest` required the pseudo source grid (320) to equal the model consumer grid (448), despite the dataset's existing nearest-neighbor mask resize. The repair makes source-grid validation explicit and records `source_image_size`, `consumer_image_size`, and `resized_for_consumer`; evaluator callers can still require an exact grid. A dedicated integration test passes, 27 audit tests pass with two optional-SciPy skips, and the new `btxrd-small-lesion-research-v2` bundle is hash-locked. Version 2 is running with the unchanged scientific protocol.
- The validation-only Gate-C2 morphology ablation rejected pseudo-mask erosion. The promoted unmodified flip-TTA masks reproduced Dice `0.2343392222`; square erosion widths 3/5/7/9 px reduced Dice to `0.2246593401/0.2071840015/0.1938446124/0.1824582669`. Even 3 px had paired delta `-0.0096798820` with tumor-group bootstrap 95% CI `-0.0172828969` to `-0.0031285875` across 167 tumor groups. All 371 rows, 184 tumors, 187 normals, split/source/kernel hashes, `train_polygons_loaded=false`, and `test_evaluated=false` passed audit. The summary SHA-256 is `9bcca010618c2370f1024ec87ada2bf0df8921b5de562160bd5172bc39443d03`; erosion is not carried into training.
- Gate-D version 2 completed all 23 training epochs and both validation passes before a wrapper-only schema check failed: `evaluate_unet.py` omits the optional `missing` key after already proving 371/184/187 coverage, while the wrapper incorrectly treated the absent key as `-1`. The contract is repaired to reject nonzero `missing` only when the field exists; 6/6 payload tests pass. The scientific result is nevertheless fully recoverable from the emitted artifacts: best epoch 13 fixed-threshold tumor Dice `0.2172072031`; validation-selected threshold `0.85` yields Dice `0.2300198701` (95% CI `0.1857288351-0.2750307254`) and normal specificity `0.5133689840`. Small/medium/large Dice is `0.0710014171/0.4134030430/0.3269168774`. Train positive Dice reached `0.7381718686`, demonstrating pseudo-target fitting without localization generalization. The selected summary SHA-256 is `d5ccc8acb5a8b9c35341c7bb0d7232ced406dc9f2884f525e28335aa44b2d2ca`; test stayed locked.
- Decision: do not spend another T4 run repeating the unchanged Gate-D recipe merely to produce the wrapper summary. The next bounded hypothesis directly targets the measured SAM selection loss: a classifier-causal candidate rank using deletion/insertion evidence, while retaining promoted flip-TTA CAM, candidate pool, prompts, support clipping, split, and metric. Its source is immutable Kaggle dataset `itsthang333/btxrd-small-lesion-research-v3`; validation kernel `itsthang333/btxrd-binary-cam-sam-causal-selector` is running.
- The classifier-causal selector completed and is rejected. It reached validation pseudo-mask Dice `0.2267174872` versus the promoted flip-TTA baseline `0.2343392222`, a paired delta of `-0.0076217350` with tumor-group bootstrap 95% CI `-0.0270798054` to `+0.0117573222`. The predeclared promotion rule is false, `test_evaluated=false`, and the compact comparison SHA-256 is `a6254cd855aef5115c2c25e098b5c7962b64c5e7314650d23abdb37b37c5fdde`.
- Research improvement work is now paused by user instruction. Repository normalization and thesis-ready WSSS selection take priority; no new Kaggle experiment is authorized until the user explicitly resumes research.

## 2026-07-24

- Research resumed on branch `research-wsss-improvement` after the thesis-ready pipeline was placed on `main`. Heavy experiments remain Kaggle-only. Model selection continues to use the clean validation partition; the final test result is not used as experimental feedback.
- A bounded adaptation of AdvCAM's adversarial-climbing mechanism was evaluated as a single-variable Gate-C ablation. The existing binary DenseNet121 checkpoint, flip TTA, CAM thresholds, SAM prompts/candidates, selector, post-processing, split, and metric were frozen. The candidate added ten target-tumor-logit input-gradient updates per TTA orientation with step size `0.08`, then averaged the normalized LayerCAMs from steps 0-10.
- Private Kaggle T4 kernel `itsthang333/btxrd-binary-adversarial-climbing-cam` completed in 616.7 seconds. Its PyTorch preflight ran 11 real Torch tests successfully. The output covers 371 unique clean-validation images (184 tumor, 187 normal), with zero missing images; all split, classifier, SAM, source, run-metadata, pseudo-manifest, and per-image hashes match their frozen contracts; `test_evaluated=false`.
- Candidate tumor-only Dice is `0.2489722614` versus the promoted flip-TTA baseline `0.2343392222`, a point delta of `+0.0146330392`. An independently reproduced 10,000-resample paired tumor-group bootstrap over 167 groups gives 95% CI `-0.0187006140` to `+0.0476416375`. The predeclared rule requires a positive lower bound, so the candidate is **rejected**, despite its higher point estimate.
- The aggregate point increase is misleading for the primary failure mode. Small-lesion Dice on 94 tumors fell from `0.1121634039` to `0.0646569093` (`-0.0475064946`), while medium and large lesions improved by `+0.0730616899` and `+0.1054248913`. Foreground-support recall increased to `0.8017557388`, but precision remained only `0.1004775162`, and selection loss increased to `0.1927557476`. Adversarial climbing expanded evidence mainly for larger lesions while further diluting small-lesion localization.
- Compact evidence is retained under `artifacts/kaggle/wsss_adversarial_climbing_cam_v1/`. The comparison SHA-256 is `3a387161a649b3240fe86b7af9c10c714d491de66cb00ea2f10d29d758c1aadc`, candidate per-image SHA-256 is `afa7dfeb700b4c20fcb3567fe2cf9fb695bba9d6fe15d7f8e3ca3efc34f595cc`, and baseline per-image SHA-256 is `fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540`. The rejected runtime/CLI/test surface was purged from the final worktree; its implementation and real-GPU test evidence remain in commit history and the compact artifact.
- A deeper paired decomposition shows that the expanded CAM is useful even for small lesions: small-lesion foreground recall rises by `+0.190074`, SAM oracle Dice by `+0.027705`, and clipped-oracle Dice by `+0.036647`. The loss occurs after proposal generation: small-lesion selection loss rises by `+0.077056`, selected Dice falls by `-0.040409`, and the final predicted area expands by `+0.024175` of the image while precision falls by `-0.062437`. Medium/large selection does not show this failure. Therefore the correct next target is the coupling between proposal expansion and candidate ranking, not discarding the expanded proposal pool.
- Observable scalar metadata from the two completed runs does not reliably identify which images benefit: the best tested single-feature benefit AUC is only about `0.55`, and a fixed one-percent predicted-area gate reaches only `0.243597`. A ground-truth-size hybrid would reach `0.273242`, and a per-image oracle between the two final masks would reach `0.301075`, but both are diagnostic ceilings and are forbidden as deployment/model-selection rules.
- The next predeclared experiment is a dual-map proposal/ranking ablation. The adversarially expanded flip-TTA CAM generates components, prompts, SAM candidates, and the support constraint; the unmodified promoted flip-TTA CAM is retained as a high-precision **selection CAM** for the unchanged `coverage_mass_sam` ranker. No GT size, polygon, or validation-derived per-image gate is available to the algorithm. This isolates the measured failure while preserving the expanded pool whose oracle quality improved in every size subgroup. Promotion requires overall paired 95% CI lower bound above zero versus `0.2343392222` and no decrease in the predeclared 94-image small-lesion mean. Higher-resolution classifier training is deferred until this cheaper causal test is resolved.
- The dual-map run completed and is rejected by both predeclared conditions. Dice is `0.2424091257`, delta `+0.0080699035` versus promoted flip TTA, with independently reproduced paired 95% CI `-0.0177663368` to `+0.0336182788`. Small-lesion Dice is `0.1004806976`, still `-0.0116827063` below baseline. The dual map partially rescues small lesions relative to expanded-map ranking (`0.0646569093`) and raises overall mean precision from `0.2370598035` to `0.2480668260`, but recall falls from `0.5582865493` to `0.4582456568`, complete misses rise from one to nine, and medium/large gains shrink.
- This identifies a second coupling inside the selector: the same CAM score currently controls candidate eligibility/thresholding, component priority, and within-component mask ranking. Replacing all three with the baseline CAM recovers precision but incorrectly rejects useful expanded components. The next single-change ablation therefore uses expanded-CAM scores only for eligibility and component priority, while baseline-CAM scores choose the best SAM mask among eligible candidates within each retained component. The score formula, threshold `0.4`, top-three components, candidate pool, support, and every other setting remain fixed.
- The split-selector run `itsthang333/btxrd-binary-advcam-split-selector` completed with Dice `0.2387746569`, only `+0.0044354347` over the promoted flip-TTA baseline. The independently reproduced paired group-bootstrap 95% CI is `-0.0271092059` to `+0.0354617169`; the predeclared promotion rule is false.
- Medium and large lesions retain gains of `+0.0476357232` and `+0.0887489992`, but the 94-image small-lesion subgroup falls from `0.1121634039` to `0.0673641053` (`-0.0447992986`). Mean recall is `0.5518265541`, precision `0.2289799200`, and only one tumor is a complete miss. This confirms that preserving expanded eligibility does not solve the over-broad selection failure on small lesions.
- All 371 validation images (184 tumor, 187 normal) are present exactly once; split/checkpoint/SAM/source hashes match, 11 Torch tests and 30 audit tests passed on Kaggle, and `test_evaluated=false`. Compact evidence is retained under `artifacts/kaggle/wsss_advcam_split_selector_v1/`; comparison SHA-256 is `41ecb86cb423c929993d8197968191155981fc00c5ec4c69a437a3a0dd6e61d6`.
- Decision: reject and close the post-hoc AdvCAM selector family. Across expanded ranking, baseline ranking, and split eligibility/ranking, the medium/large signal survives but no ground-truth-free selector protects small lesions. The next controlled hypothesis must change the learned spatial representation rather than tune another validation-time gate. A 448 px binary classifier is predeclared as the next experiment; the split, objective, initialization, optimizer, augmentation, stopping rule, CAM/SAM recipe, and evaluation endpoint remain fixed, and only input resolution plus proportional geometry constants may change.
- The controlled 448 px binary-classifier run completed at epoch 12 with validation F1 `0.7959183673` versus `0.7833333333` at 320 px. Sensitivity rose from `0.7663043478` to `0.8478260870`, while specificity fell from `0.8128342246` to `0.7219251337`. Checkpoint SHA-256 is `bb70912c179bda8cc32498b6bf8c405f11d46a37143cb4577a9986a838cec45c`; all 371 validation rows and the frozen split verify, and `test_evaluated=false`.
- Gate-C evaluation with that checkpoint is rejected. Dice is `0.2296019980`, delta `-0.0047372242` versus the promoted 320 px flip-TTA baseline, with independently reproduced paired group-bootstrap 95% CI `-0.0415031583` to `+0.0320557694`. Small and medium change only `+0.0018854426/+0.0015835381`; large falls `-0.0646053111`.
- Resolution nevertheless improves the candidate pool for small lesions: foreground recall rises `+0.207594`, box recall `+0.222049`, oracle SAM Dice `+0.041537`, and clipped-oracle Dice `+0.048980`. Selected Dice is flat (`-0.000195`) because small selection loss grows `+0.049174`. Across all tumors, oracle Dice rises `+0.025554` but selected Dice falls `-0.005526`. Large-lesion foreground IoU, oracle Dice, and selected Dice fall `-0.039186/-0.042893/-0.062787`, explaining the aggregate rejection.
- This reproduces the same structural bottleneck as unregularized adversarial climbing: additional useful candidates do not help when the CAM used for expansion is poorly calibrated for candidate ranking. Post-hoc scalar gates have already failed, so the next bounded experiment implements AdvCAM's missing activation-difference regularizer rather than another selector. Freeze the 320 px promoted classifier and Gate-C recipe, keep ten climbing steps at `0.08`, and add only the paper/repository default high-activation preservation term (`AD_coeff=7`, score threshold `0.5`) on the differentiable final LayerCAM layer. Promotion still requires a positive paired-CI lower bound and no small-lesion decrease.
- Compact evidence is stored under `artifacts/kaggle/wsss_binary_classifier_448_v1/` and `artifacts/kaggle/wsss_binary_cam_sam_classifier448_v1/`. Gate-C comparison SHA-256 is `98f5925c884a0b84b27aeb4a06ed4ebd6b80a748d77767edf402c7e7378a1daa`; candidate per-image SHA-256 is `dc086958f8b89747eecedfda5268119b1f3008f931fe4a5fe1809d24c2db0c6a`. Nine Torch tests and 29 audit tests passed on Kaggle; population is 371/184/187 with zero missing; test stayed locked.
- Activation-regularized AdvCAM completed and is rejected. Dice is `0.2307861310`, delta `-0.0035530912`, and the independently reproduced paired group-bootstrap 95% CI is `-0.0289307324` to `+0.0207423190`. Small remains below baseline by `-0.0399829922`; medium and large retain real point gains of `+0.0275501896/+0.0622788244`.
- Relative to unregularized climbing, `L_AD` partially rescues small selection: selection loss improves by `0.037925` and final small Dice by `0.007524`. The cost is weaker expansion: small oracle Dice falls `0.027753`, and medium/large final Dice lose `0.045512/0.043146` relative to the unregularized version. Relative to baseline, regularized small oracle is essentially unchanged (`-0.000048`) while selected Dice remains `-0.033557`. The official coefficient/threshold therefore trade coverage for core preservation but do not resolve the calibration problem; no coefficient sweep is justified on validation.
- All 371 rows (184 tumor, 187 normal), 184 prompt diagnostics, split/checkpoint/SAM/source hashes, 11 real Torch tests, comparison hash, and `test_evaluated=false` verify. Compact evidence is under `artifacts/kaggle/wsss_advcam_regularized_v1/`; comparison SHA-256 is `0daec4744259e766b2e949fe4cbc9de21e8f4f81a5df35a5c605ca7c947e4e54`, per-image SHA-256 `573f429077cdd0151f55e8fcef56b19cfe14cbae0be669ac096a7b291557edd9`.
- Decision: close inference-time climbing. The repeated pattern is now causal: expanding evidence improves the SAM oracle, while the classifier features/CAM score remain insufficiently region-aware to choose the right mask. The next high-potential experiment follows S2C's learned feature transfer rather than another post-hoc rule. Precompute SAM ViT-B Segment-Everything region maps on the 2,981 clean-train images only (no labels/polygons/validation/test), then train the frozen 320 px binary DenseNet recipe with BCE plus SAM-Segment Contrasting loss. Use the paper/code default loss weight and temperature `1`; keep augmentation off so region maps remain aligned. Gate-C and promotion rules remain unchanged.
- The official S2C implementation was inspected before adaptation. Its Segment-Everything maps initialize uncovered pixels to `-1`, assign `enumerate(reversed(masks))`, then add one at load time so uncovered pixels become cross-entropy ignore ID `0`. SSC normalizes classifier features, averages detached features inside each SAM region to form prototypes, and classifies covered pixels by prototype-feature cosine similarity.
- Private Kaggle kernel `itsthang333/btxrd-s2c-segment-everything-v1` was launched with official Segment Anything commit `6fdee8f...` and verified SAM ViT-B checkpoint `ec2df627...`. It processes exactly 2,981 clean-train images at 320 px, saves contiguous 80x80 `uint16` maps, and does not load polygons, validation images, or test. Every source/map hash and the complete population are checked before acceptance.
- The classifier now has an opt-in S2C SSC path that is inert at weight zero. When enabled, it requires a frozen region-manifest SHA, verifies every train image/map hash, locks 320 px with augmentation off and preprocessing `none`, uses detached per-image segment prototypes, ignores ID 0, and records its full configuration in the checkpoint. The frozen ablation is BCE + SSC weight 1, temperature 1, with the otherwise unchanged clean binary DenseNet121 recipe.
- The first real-PyTorch smoke test correctly stopped before training because
  the per-image cosine matrix was passed to cross-entropy as
  `[regions,pixels]` rather than `[pixels,regions]`. This is an implementation
  error, not model evidence. The transpose was repaired and will be published
  as a new immutable source bundle rather than overwriting v9. A separate real
  DenseNet optimizer step must pass before the long classifier job. Test
  remains locked.
- Repaired source bundle `itsthang333/btxrd-small-lesion-research-v10` passed
  all four SSC/map-integrity Torch tests and a faithful T4 optimizer step at
  batch 8, image size 320, map size 80 and AMP. DenseNet feature shape was
  `[8,1024,10,10]`; BCE/SSC/total loss was
  `0.692756/3.825942/4.518698`, and the pre-clip gradient norm was finite at
  `3.762442`. Compact evidence is
  `artifacts/kaggle/s2c_ssc_torch_test_v1/test_result.json`.
- The full classifier wrapper is prepared but deliberately fails its payload
  test while the region-manifest hash placeholder remains. It may be launched
  only after the train-only precompute completes and the exact manifest SHA is
  frozen.
- The train-only SAM Segment-Everything precompute completed after
  `15104.39` seconds. It produced exactly 2,981 unique 80x80 `uint16` region
  maps for the 2,981 clean-train images; region count was min/mean/median/max
  `1/7.7246/7/51`, and uncovered fraction was
  `0/0.41980/0.42828/0.99344`. The split population remained
  2,981/371/373 with no group overlap, every source/map hash was populated,
  and the final map-manifest SHA-256 is
  `677dbb16ca0824a03bb75c991a01e6a83de5819d1db1f6f6d9f02765375c8f8e`.
  `polygons_or_masks_loaded=false`, `validation_images_processed=false`, and
  `test_evaluated=false`; compact evidence is under
  `artifacts/kaggle/s2c_segment_everything_v1/`.
- That manifest hash is now frozen in the classifier wrapper, whose 3/3
  fail-closed payload tests pass and whose four bound project-source hashes
  match the repaired v10 bundle. Kaggle kernel
  `itsthang333/btxrd-s2c-binary-classifier-v1` version 1 is running the
  predeclared binary DenseNet121 320 BCE+SSC experiment. Gate C remains
  unchanged and fail-closed until the completed classifier checkpoint hash is
  available.
- The S2C SSC classifier completed 17 epochs and early-stopped validly after
  seven epochs without improving its epoch-10 checkpoint. Clean-validation F1
  was `0.8201058201`, versus `0.7833333333` for the BCE baseline; sensitivity
  rose by `+0.0760869565`, specificity fell by `-0.0213903743`, AUROC rose by
  `+0.0107678447`, and AUPRC rose by `+0.0060122330`. The checkpoint SHA-256 is
  `6c1562a4f6b2e789d71749c20fb9e8e3304e74c5f10da4f77cbd3657d820ee48`.
  All 371 validation rows, 17 training rows, fixed-threshold confusion
  `155/39/29/148`, early-stop record, split/region/source hashes, and
  `test_evaluated=false` were independently reproduced.
- The unchanged Gate-C run `itsthang333/btxrd-s2c-gate-c-v1` is rejected.
  Tumor-only Dice is `0.2334851256` versus the promoted flip-TTA baseline
  `0.2343392222`; paired delta is `-0.0008540966`, with complete-group
  bootstrap 95% CI `-0.0312535770` to `+0.0291092193`. The frozen 94-image
  small-lesion subgroup falls by `-0.0487671999`, while medium and large rise
  by `+0.0476879964/+0.0551904038`; therefore every promotion condition is
  false. All 371 images, 184 tumors, 187 normals, 42 runtime tests, source and
  artifact hashes verify; test stayed locked.
- Mechanism decomposition explains why classification improvement did not
  transfer to small-lesion localization. Across all tumors, foreground recall
  rises `+0.138044`, box recall `+0.169903`, and SAM single-candidate oracle
  Dice `+0.012096`. For small lesions, however, predicted area rises from
  `0.03105` to `0.07195` of the image, precision falls `-0.066781`, point-hit
  rate falls `-0.017469`, oracle Dice falls `-0.024367`, selected Dice falls
  `-0.041551`, and post-processing costs another `-0.007216`. SSC therefore
  learned broader tumor evidence useful to medium/large cases but not a
  sufficiently fine spatial representation for sub-one-percent lesions.
- This result exposes a controlled fidelity gap rather than grounds for an
  SSC weight or threshold sweep. Official S2C applies SSC to a 256-channel
  high-resolution feature map from a stride-8 dilated classifier, whereas the
  first BTXRD adaptation applied the same prototype/cosine loss to the final
  DenseNet map `[B,1024,10,10]` at 320 px (stride 32) and then interpolated it
  to 80x80. A median small lesion is only about 11 px across and is therefore
  substantially below one final-map cell.
- The next predeclared single-change experiment moves SSC from final
  DenseNet features to the existing `denseblock2` output
  `[B,512,40,40]` (stride 8 at 320 px), without adding a projection head or
  changing the model state dictionary. BCE, SSC weight/temperature, SAM maps,
  initialization, optimizer, split, checkpoint selection, flip-TTA LayerCAM,
  SAM recipe, selector, metric, and promotion rule remain fixed. This directly
  tests spatial granularity; no validation-derived threshold or per-image
  gate is introduced. If this stride-8 candidate fails Gate C, close the SSC
  feature-tap family rather than tuning loss weights on validation.
- Source bundle `itsthang333/btxrd-small-lesion-research-v11` passed 58/58
  local tests. Private T4 smoke kernel
  `itsthang333/btxrd-s2c-stride8-torch-test-v1` then passed all six targeted
  tests and one real batch-8, 320 px AMP optimizer step. The captured feature
  is `[8,512,40,40]`; BCE/SSC/total loss is
  `0.692756/4.124758/4.817514`; gradient norm is finite at `3.758309`; and the
  inference state dictionary is unchanged. Compact evidence is under
  `artifacts/kaggle/s2c_stride8_torch_test_v1/`.
- Classifier kernel
  `itsthang333/btxrd-s2c-stride8-binary-classifier-v1` version 2 is running.
  Version 2 corrects only a wrapper metadata-key mismatch
  (`feature_projection`); the predeclared training recipe and source bundle
  are unchanged. Gate C remains fail-closed until the completed checkpoint
  SHA-256 replaces its placeholder.
- The WACV 2024 paper *Small Objects Matters in Weakly-supervised Semantic
  Segmentation* was reviewed as a bounded downstream option. Its
  size-weighted cross-entropy upweights pixels in small connected components
  of pseudo masks, while an EWC second stage preserves the preceding
  large-object objective. This is relevant only after pseudo-label/localization
  quality improves: the existing Gate-D experiment already showed that merely
  fitting the current Dice-0.234 pseudo masks does not generalize. It is
  therefore not allowed to displace the current stride-8 Gate-C test or to be
  reported as a pseudo-mask improvement.
- A train-only CPU oracle tested whether the stored 80x80 Segment-Everything
  partitions could instead be learned or selected directly. Across all 1,488
  clean-train tumors, the mean target coverage ceiling was only `0.4480473`,
  the best-single-region Dice was `0.1112537`, and the exact optimal subset
  Dice was `0.1126347`; 648 tumors had zero best-single overlap. For the 752
  small tumors, optimal-subset Dice was only `0.0759493` and 355 had zero
  overlap. Eleven small targets vanished entirely after 80x80 downsampling.
  The audited row SHA-256 is
  `3cc3d41d2cafd842361a382089270506e00fd55fec592708f3cd13a9b0b6fffa`;
  no validation/test images were processed. Direct MIL/selection over these
  overwritten partitions is therefore rejected without spending a validation
  GPU run. This does not reject official S2C CPM, which obtains separate
  point-prompted SAM masks online instead of treating the SSC partition map as
  a proposal inventory.
- The stride-8 SSC classifier version 2 completed and passed independent
  audit. It validly early-stopped at epoch 19 and selected epoch 12.
  Clean-validation F1 is `0.7965616046` versus baseline `0.7833333333`
  (`+0.0132282713`); sensitivity changes by `-0.0108695652`, specificity by
  `+0.0481283422`, AUROC by `-0.0041850732`, and AUPRC by `+0.0043807906`.
  The 371-row confusion matrix recomputes exactly as
  `TP/FP/FN/TN = 139/26/45/161`. Checkpoint SHA-256 is
  `8bb4a9136291bffba9d2e70752f5b03c003730ff5365ab7e1033f8dc222331d6`.
  Its metadata freezes `denseblock2`, stride 8, 512 channels, post-ReLU,
  no projection, SSC weight/temperature 1, all 2,981 maps and the region
  manifest hash. Train/validation polygons were not loaded and test remained
  locked. Classification alone cannot promote the model; the unchanged
  Gate-C localization run is now authorized.
- The unchanged stride-8 Gate-C run is decisively rejected. Mean tumor-only
  Dice is `0.1842404724` versus `0.2343392222`; paired delta is
  `-0.0500987498`, with predeclared complete-group bootstrap 95% CI
  `-0.0818340900` to `-0.0195896270`. The frozen small/medium/large deltas
  are all negative: `-0.0597196529/-0.0289336577/-0.0845166242`.
  Independent reconstruction from the two 371-row files reproduced 184
  tumors, 167 tumor groups, the exact point delta and a 10,000-resample CI
  within `3e-5` of the recorded interval. All source/checkpoint/split hashes
  verify and test remained locked.
- The mechanism also closes the SSC feature-tap family rather than motivating
  another weight/temperature sweep. Relative to the promoted baseline,
  foreground recall changes only `+0.0093599`, while foreground precision,
  point-hit rate, SAM single-candidate oracle Dice, clipped oracle Dice,
  selected Dice and final Dice change by
  `-0.0121927/-0.0514334/-0.0310159/-0.0325518/-0.0499680/-0.0500988`.
  The stride-8 regularizer therefore does not merely expose a selector issue;
  it degrades the candidate evidence itself on this domain.
- The next method is a separately bounded CPM adaptation, not another SSC
  tap. A DenseNet FPN-style CAM classifier fuses the existing stride-8
  `denseblock2` tensor with projected final semantics, produces a
  256-channel 40x40 feature and a differentiable one-channel CAM, and couples
  classification to that CAM through global average pooling. This preserves
  fine spatial degrees of freedom while avoiding the shallow-feature-only
  classifier that a direct `denseblock2` head would create. A T4 smoke run
  passed 3/3 tensor tests plus a real 320px batch-2 BCE+SSC+CPM optimizer
  step; losses were `0.678304/4.011762/0.379352`, all three head branches
  received gradients, and test was not evaluated. The full CPM training
  remains gated on a train-only, hash-audited SAM embedding cache and a frozen
  no-polygon protocol.
- The frozen SAM ViT-B embedding cache completed on Kaggle in `896.28`
  seconds. It contains exactly the 1,488 clean-train tumor images as
  `[256,64,64]` float16 tensors (3,120,562,304 bytes); normal-image CPM
  targets remain deterministic all-background and therefore require no cached
  embedding. Cache/index SHA-256 values are
  `1853602fb36913c049010461057014465780ba06f37e5a562a4aae2f0a6436be`
  and
  `4c838827c02be65f3c1793b4c7036b2c7d0625d920d19c56666251a237ad7d06`.
  The run used only raw train radiographs and binary image labels, loaded no
  polygon or mask, processed no validation image, and kept test locked.
  Compact evidence is under
  `artifacts/kaggle/s2c_cpm_sam_embedding_cache_v1/`; the 3 GB cache remains
  on Kaggle.
- A pre-launch state audit found that multiscale teacher-CAM generation left
  the classifier in evaluation mode after the first CPM batch. This would
  unintentionally freeze BatchNorm behavior from epoch 3 onward. The function
  now restores its incoming train/eval state in a `finally` block, and a
  dedicated regression test checks the invariant. This is an implementation
  correction made before any long CPM training or validation localization
  result; it does not change the predeclared objective or use validation/test
  feedback.
- The corrected source at Git commit
  `f4a17a41d9e66d2e23f1c7b079d882a5e89e7ca4` then passed all six CPM
  tensor/state tests and a real T4 batch-2 optimizer step. Fused feature/CAM
  shapes were `[2,256,40,40]` and `[2,1,40,40]`; BCE/SSC/CPM/total losses
  were `0.678304/4.011762/0.379352/5.069419`; the finite pre-clip gradient
  norm was `13.012649`, and both lateral projections plus the CAM head
  received gradients. Compact evidence is under
  `artifacts/kaggle/s2c_cpm_torch_test_v2/`. The long classifier run is
  authorized only with this commit and the frozen cache/index hashes.
- A contingent post-CPM diagnostic is pre-scoped from Pro2SAM
  (`https://arxiv.org/abs/2505.04905`), which addresses SAM prompt ambiguity
  by generating a grid-point mask gallery and matching masks to a coarse
  semantic map at pixel level. It is not authorized as an unconditional next
  run. It may be tested only if the completed CPM Gate-C mechanism audit shows
  that small-lesion loss is dominated by missing SAM proposals (low
  single-candidate oracle/point-hit despite usable direct-CAM support). The
  only changed variable would be the proposal gallery; the frozen CPM CAM,
  selector score, support clipping, morphology, split, metric, and promotion
  rule would remain unchanged. If CPM instead has poor CAM support or high
  selection loss with an adequate oracle pool, grid prompting is rejected
  without a GPU run because it does not address the observed bottleneck.
- The full S2C SSC+CPM classifier completed on Kaggle and passed the frozen
  artifact audit. It validly early-stopped after 10 epochs and selected epoch
  3 with clean-validation F1 `0.7473684211`. Independent reconstruction from
  all 371 per-image rows gives `TP/FP/FN/TN = 142/54/42/133` and the exact
  recorded F1. Relative to the baseline classifier, F1/AUROC/AUPRC change by
  `-0.0359649123/-0.0889764008/-0.1243164066`; sensitivity changes by
  `+0.0054347826`, while specificity changes by `-0.1016042781`. These are
  diagnostics, not localization evidence. The downloaded 89,100,000-byte
  checkpoint independently hashes to
  `7da19e9c0537501c4c919200ed65b2bf6992383af70aa91c18b312a5d6204043`.
  Split/source hashes match commit
  `f4a17a41d9e66d2e23f1c7b079d882a5e89e7ca4`; train/validation polygons were
  not loaded and test remained locked. Compact evidence is under
  `artifacts/kaggle/s2c_cpm_classifier_v1/`.
- The checkpoint hash is now frozen into the predeclared Gate-C wrapper. All
  six local fail-closed protocol tests pass. Kernel
  `itsthang333/btxrd-s2c-cpm-gate-c-v1` version 1 was launched with the
  unchanged promoted CAM-percentile/SAM selector and the three-part promotion
  rule: improve overall tumor Dice above `0.2343392222`, paired complete-group
  bootstrap 95% lower bound above zero, and no decrease on the frozen 94-case
  small-lesion subgroup. All 184 validation tumors remain in the endpoint and
  complete misses remain zero; test remains inaccessible to model selection.
- The completed CPM Gate-C run is decisively rejected. Mean tumor-only Dice is
  `0.1643982371` versus the promoted `0.2343392222`; the paired complete-group
  delta is `-0.0699409850` with 10,000-resample 95% CI
  `[-0.1088206332, -0.0322901766]`. Small/medium/large deltas are
  `-0.0825044780/-0.0632940569/-0.0309193456`. Independent reconstruction
  verifies 371 validation images, 184 tumors, 187 normals, 167 complete tumor
  groups, zero missing predictions, the frozen split/checkpoint hashes, and
  `test_evaluated=false`.
- The mechanism audit separates the bottlenecks. On the 94 small tumors, CAM
  support recall remains usable (`+0.02737` versus the promoted baseline), but
  point-hit falls `-0.0617` and clipped single-candidate oracle Dice is only
  `0.12942` (`-0.08397`), so proposal omission/quality is dominant. Medium and
  large clipped oracle Dice remains `0.56444/0.55598`, while selection loses
  `0.26923/0.17487` Dice, so those groups are mainly selector-limited. Overall
  clipped oracle Dice is only `0.34137`, which also proves that selector repair
  alone cannot meet the former `>0.50` endpoint.
- The predeclared dense grid SAM gallery diagnostic is therefore authorized,
  but only to test the small-lesion proposal hypothesis. It changes the SAM
  proposal gallery to the official 32-by-32 automatic-mask grid while freezing
  the CPM CAM, `coverage_mass_sam` ranking score, support clipping,
  morphology, split, metric and promotion rule. It does not import Pro2SAM's
  full training method or replace the pipeline selector. Test remains locked.
- A fresh execution audit of the frozen GT reference exposed a portability
  defect in its old snapshot verifier: text files were recorded with LF byte
  sizes/hashes, while the Windows working tree uses CRLF. The verifier therefore
  rejects `convergence_summary.json` before reaching the otherwise intact
  checkpoint. The 230,924,939-byte checkpoint still independently hashes to
  `05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31`.
  This is an audit-tool defect, not permission to waive verification; the new
  reference lock will use canonical-LF hashes for text and exact byte hashes
  for binary checkpoints.
- Grid-gallery smoke version 1 failed closed before loading data or a model.
  Its source audit used hashes calculated from the CRLF Windows checkout,
  whereas Kaggle correctly cloned canonical LF Git blobs at commit
  `ab5f7cca1036b60a8b225288f14e20a70097234a`. No experiment ran and test was
  not read. Version 2 replaces only those two expected source hashes with the
  canonical Git-blob SHA-256 values; the scientific protocol is unchanged.
- Grid-gallery smoke version 2 passes on a Kaggle T4. The official SAM commit
  `6fdee8f...` generated a 32-by-32 dense point gallery for the frozen
  validation tumor `IMG000001.jpeg`, retained 16 independent masks at the
  predeclared predicted-IoU/stability filters, returned aligned
  `[16,320,320]` candidates and 16 scores in `[0.88088,0.98121]`, and preserved
  prompt mode `grid` for every candidate. CPM classifier, SAM checkpoint and
  split hashes match the frozen values; source is commit `ab5f7cca...`;
  `test_evaluated=false`. This is execution evidence only, not a performance
  or promotion claim. Full validation is now authorized with the same CPM CAM,
  `coverage_mass_sam` score, support clipping and post-processing; global
  top-1 is required because dense gallery candidates have no CAM-component
  identity.
- The new goal's fully supervised reference arm is now independently audited
  and hash-locked as `gt_resnet18_unet_448_v1`. It is the existing 448px
  ImageNet-ResNet18 U-Net trained on all 2,981 clean-train images with seed 42,
  paired horizontal-flip augmentation, batch size 8, AdamW at `1e-4`,
  weight decay `1e-4`, and
  `0.5*BCEWithLogits(pos_weight=10)+0.5*soft-Dice`. The frozen budget is 35
  epochs with patience 10; checkpoint selection uses validation tumor Dice at
  threshold 0.5 and normal empty-mask specificity only as a tie-breaker.
  The exact 230,924,939-byte epoch-20 checkpoint hashes to
  `05606a0ace6c845ca52a26e8c4a5269bf8e03350dd31d27bbd5e80d55df70c31`.
- Independent reconstruction from the frozen 371-row validation file verifies
  184 tumors, 187 normals, 167 tumor groups, and size counts `94/72/18`.
  With the predeclared validation threshold grid and selection rule, reference
  Dice is `0.4951316963` overall and
  `0.3289549325/0.6624417784/0.6937033566` for small/medium/large. Complete
  misses remain included. The authoritative lock and reusable paired auditor
  are under `artifacts/reference/gt_resnet18_unet_448_v1/` and
  `project/tools/audit_wsl_gt_pair.py`; all three reference/pair audit tests
  pass. The lock explicitly forbids checkpoint/prediction reuse or any
  train-GT influence in the WSL arm, permits validation GT only after
  prediction, and records `test_evaluated=false`.
- The paired WSL-vs-GT protocol is frozen before any new WSL consumer
  training. Both arms must use the exact reference consumer contract and the
  common validation threshold grid `0.20:0.05:0.85`; only the train mask
  source may differ. The historical WSL segmenter already satisfies the
  consumer contract. Its older sweep included four extra endpoints, but its
  selected `0.85` threshold and the reference's selected `0.20` are both in
  the frozen common grid, so restricting the comparison does not change either
  result.
- The first authoritative paired audit quantifies the starting gap. Historical
  image-label-only WSL Dice is `0.2300198701` overall and
  `0.0710014171/0.4134030430/0.3269168774` for small/medium/large. Absolute
  gaps to the GT reference are respectively
  `0.2651118262` overall and
  `0.2579535154/0.2490387353/0.3667864792`; all fail the `<=0.05` criterion,
  with paired group-bootstrap intervals wholly below zero. The required WSL
  Dice intervals are now explicit: small
  `[0.2789549325,0.3789549325]`, medium
  `[0.6124417784,0.7124417784]`, and large
  `[0.6437033566,0.7437033566]`. This makes clear that small tumors remain the
  priority named by the goal, while large tumors currently have the largest
  supervision gap and cannot be ignored.
- Future paired WSL consumer runs are now fail-closed at the training entry
  point. Supplying `--paired-reference-lock` verifies the frozen split hash and
  requires the exact GT-reference consumer contract: ResNet18 U-Net with
  ImageNet initialization, 448px input, batch size 8, seed 42, AdamW at
  `1e-4`, weight decay `1e-4`, manual `pos_weight=10`, no CLAHE, maximum 35
  epochs, patience 10, and Dice tie tolerance `1e-4`. It also requires a train
  pseudo-mask root and forbids a validation pseudo-mask override, so the sole
  permitted arm difference is the train mask source and validation GT remains
  post-prediction only. The resolved reference-lock identity and hashes are
  embedded in training configuration and checkpoints. Seven contract and
  paired-audit tests pass; test remains locked.
- An exact GT-reference reproduction was launched as private Kaggle kernel
  `itsthang333/btxrd-gt-reference-exact-reproduction-v1`, version 1, after an
  independent run by a collaborator was reported to differ. This is not a new
  model or hyperparameter experiment: it reuses the original historical
  wrapper byte-for-byte (SHA-256
  `900d0594d593e9bb980b5cb46401be164bce9e3a495d8aedeebb2c7f3da90123`),
  immutable clean split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  and epoch-10 resume checkpoint SHA-256
  `8278865eba7e07e0f1e327e7f6a853e3ee770a30a5a57e35438bfe5f6c458bd3`.
  The resume checkpoint restores model, optimizer, AMP scaler and saved
  PyTorch/CUDA/NumPy/Python RNG state before continuing the frozen 448px,
  seed-42, `pos_weight=10` contract. The new run is isolated from the
  authoritative reference and will be compared by epoch, checkpoint hash,
  threshold selection, per-image predictions and lesion-size subgroup after
  completion. Validation only is authorized; test remains locked.
- GT reproduction version 1 failed before loading the dataset, model or any
  training state. The historical wrapper expected the old direct bundle with
  `resume_448_epoch10/last_unet.pt`, but the current immutable dataset version
  contains compressed source/split/winner artifacts and no longer exposes that
  epoch-10 resume directory. The traceback was
  `Expected one research bundle, found []`; no epoch ran, no validation metric
  was produced, and this is not scientific evidence about reproducibility.
- Because the unavailable epoch-10 state makes an exact continuation
  impossible, the active replacement is explicitly an **independent epoch-1
  reproduction**, not a continuation. Kaggle renamed the private kernel to
  `itsthang333/btxrd-gt-reference-independent-reproduction-v2`, version 2.
  It extracts the frozen source archive and fails closed against the six
  canonical-LF source hashes in the GT reference lock, verifies the same clean
  split hash, and trains from epoch 1 with the frozen seed-42, 448px,
  ResNet18-U-Net, batch-8, AdamW `1e-4`/`1e-4`, manual `pos_weight=10`,
  35-epoch/patience-10 contract. Its wrapper SHA-256 is
  `afa5d4be59a062c2822f467a0dfcb1019842e81dc369646653abd859df8bfcb1`.
  After validation it will record checkpoint/epoch/threshold and
  small/medium/large Dice deltas against the authoritative reference. This
  independent run is the appropriate comparison for a collaborator's
  independently trained model. Test remains locked.
- The GT reproduction will be audited independently with
  `project/tools/audit_gt_reproduction.py`, rather than accepting the cloud
  summary at face value. The auditor reselects the checkpoint directly from
  contiguous epoch-1 training rows using the frozen Dice/specificity
  tie-breaker, verifies the 35-epoch/patience-10 budget, frozen source/split
  hashes, validation-only threshold grid, exact 371/184/187 cohort, per-image
  identity/group/GT-area equality, downloaded artifact hashes and test lock.
  It then computes paired complete-group bootstrap differences against the
  hash-locked reference for overall and all three lesion-size groups. Ten
  combined reference, paired-contract and reproduction-auditor tests pass.
- The full 32-by-32 official SAM automatic-mask grid-gallery diagnostic
  completed on Kaggle and is decisively rejected. Independent reconstruction
  verifies all 371 validation images, 184 tumors, 187 normals, 167 tumor
  groups, frozen `94/72/18` size counts, source commit `ab5f7cca...`, split,
  CPM-classifier and SAM hashes, and `test_evaluated=false`. Downloaded
  comparison/per-image/run-metadata/pseudo-manifest hashes all match the cloud
  manifest; no mask PNGs were retained locally.
- Candidate mean tumor Dice is only `0.0278769051`. Relative to the promoted
  flip-TTA baseline `0.2343392222`, paired complete-group delta is
  `-0.2064623171` with 95% CI
  `[-0.2524417116,-0.1607865654]`; relative to the same-CPM component-prompt
  control `0.1643982371`, delta is `-0.1365213321` with 95% CI
  `[-0.1764958606,-0.0986665204]`. Small/medium/large Dice is
  `0.0072473952/0.0579301833/0.0153956770`; all are below both controls.
  There are 110/184 complete misses, only 74 non-empty tumor predictions and
  only 16 tumor images with any overlap.
- The mechanism audit falsifies the intended proposal hypothesis. Direct
  CPM-CAM support, point and box diagnostics are bit-for-bit identical to the
  component-prompt control, but automatic-grid single-candidate oracle Dice
  falls from `0.3693434433` to `0.1193872516`, and clipped oracle from
  `0.3413734584` to `0.1130855663`. Clipped-oracle drops are
  `-0.0802427974/-0.3769592013/-0.4067270392` for small/medium/large. Thus
  dense SAM objectness proposals mainly capture radiographic anatomy rather
  than the tumor and do not repair small-lesion omission. The smaller reported
  selection loss is an artifact of the impoverished oracle pool, not selector
  improvement.
- Decision: close the dense automatic-grid family under this SAM/checkpoint
  and filtering contract; do not generate train masks or spend GPU time on a
  multiscale grid. This is a bounded grid-gallery adaptation, not a claim that
  the complete Pro2SAM method was reproduced. Dense masks lack CAM-component
  identity, so the predeclared global-top-1 aggregation is also a structural
  limitation. Compact evidence and the independent auditor are under
  `artifacts/kaggle/pro2sam_grid_full_v1/` and
  `project/tools/audit_grid_gallery.py`; eleven combined protocol tests pass.
- The next bounded WSL proposal experiment will retain the promoted flip-TTA
  CAM component-prompt gallery exactly and add proposals from the frozen
  pseudo-mask-trained WSL U-Net as a second, GT-free spatial teacher. Its
  checkpoint was trained only from image-label-derived pseudo-masks. The
  existing CAM, score, support clipping, SAM and baseline candidates remain
  available; the added teacher components are ranked only with frozen
  image-derived signals. This tests whether learned spatial generalization can
  add proposals that automatic SAM objectness missed. No train/validation
  polygon, GT size gate or per-image oracle routing is permitted. Promotion
  still requires a positive overall paired-CI lower bound versus
  `0.2343392222` and no decrease on the fixed 94-case small subgroup.
- That validation-only experiment is now running as private Kaggle kernel
  `itsthang333/btxrd-wsl-proposal-teacher-val-v1`, version 1. Source is pinned
  to commit `ef4cd71290e9aa40f6f66983e8f0aba05d8fd4a8`; the cloud wrapper SHA-256 is
  `2a8d10543c7c158c56a67d15a07b63b018c3984a8087ca0e44dca80f32a5e549`.
  The proposal teacher is independently packaged and frozen at checkpoint
  SHA-256
  `02d3af8feede3c3e650cb76d664185c59092697c1c8306ea67613b89f8407fb4`.
  Loading fails closed unless its BTXRD split, 448px ResNet18-U-Net consumer,
  seed-42 and pseudo-mask-only training provenance match. The sole experimental
  change is up to three teacher probability components at threshold `0.85`
  and minimum area 20 added to the exact promoted flip-TTA component/SAM
  gallery; CAM scoring and support clipping are unchanged. The full 371-case
  validation run will be compared to the frozen baseline with 10,000 paired
  complete-group bootstrap iterations and fixed `94/72/18` size groups. Test
  remains locked and generated masks will not be retained in compact evidence.
- Proposal-teacher kernel version 1 failed before loading a radiograph or
  running any model inference. Its cloud smoke-test command imported
  `tests.test_proposal_teacher_components`, but the repository's `tests/`
  directory is not a Python package and Kaggle resolves an unrelated installed
  `tests` namespace first. The direct traceback is
  `ModuleNotFoundError: No module named 'tests.test_proposal_teacher_components'`.
  No validation prediction or scientific metric was produced, so version 1 is
  implementation evidence only. Version 2 replaces that invocation with
  explicit `unittest discover -s tests -p
  test_proposal_teacher_components.py`; every source/model/split hash and
  scientific parameter remains unchanged. The repaired wrapper SHA-256 is
  `a08edf88efacb2d6ca339c0920a142049c36ae411cdac1174c1234af38f92d58`.
- Proposal-teacher version 2 completed and its compact evidence passed the
  independent auditor. Provenance is locked to source commit
  `ef4cd71290e9aa40f6f66983e8f0aba05d8fd4a8`, split SHA-256
  `85511ee1...`, classifier `f62d3702...`, SAM `ec2df627...`, frozen
  pseudo-trained teacher checkpoint `02d3af8f...`, and the teacher's
  pseudo-mask training manifest `7b0b133e...`. The full validation population
  is unchanged at 371 images/184 tumors/187 normals, 167 tumor groups and
  `94/72/18` small/medium/large; test was not evaluated. Downloaded candidate,
  run-metadata and pseudo-manifest hashes all match the cloud manifest.
- The proposal-teacher hypothesis is rejected. Candidate mean tumor Dice is
  `0.2336592113` versus promoted flip-TTA baseline `0.2343392222`, paired
  complete-group delta `-0.0006800109` with 95% CI
  `[-0.0093284022,+0.0084660375]`. Small and medium change only
  `+0.0012430942/+0.0014239183`, while large decreases
  `-0.0191386101` with paired 95% CI
  `[-0.0426347225,-0.0025369182]`; the predeclared promotion rule is false.
  No 2,981-image train pseudo-mask generation is authorized.
- The mechanism result is informative despite rejection. Teacher components
  are available on 154/184 tumors and increase unclipped single-candidate
  oracle Dice by `+0.0400704867`; clipped oracle still increases
  `+0.0220265163`. However selected Dice changes `-0.0009038856`.
  Added proposals increase support loss by `+0.0180439704` and selection loss
  by `+0.0229304020`. On large tumors specifically, oracle improves
  `+0.0399334479` but clipping retains only `+0.0037785493`, selected Dice
  falls `-0.0189320037`, and final Dice falls `-0.0191386101`. Therefore the
  next proposal-teacher experiment must change source-aware candidate
  selection/support semantics rather than adding more teacher components under
  the unchanged CAM-only selector and clip.
- A lightweight post-hoc gate diagnostic was used only to localize the
  mechanism and is not promotion evidence. Candidate masks differ from the
  baseline on 97/184 tumors. Changed small/medium cases average
  `+0.003246/+0.002092` Dice, whereas 12 changed large cases average
  `-0.028708`. Teacher-support upper gates produce at most about `+0.0022`
  overall in this diagnostic and do not robustly protect large tumors.
  Consequently no validation-GT-derived area gate will be implemented.
- The independent epoch-1 GT reproduction completed on Kaggle with the exact
  frozen code, split, ImageNet initialization, seed-42, 448 px, batch-8,
  AdamW `1e-4`/`1e-4`, `pos_weight=10`, 35-epoch/patience-10 contract and
  validation-only threshold grid. The independently recomputed checkpoint
  selection is epoch 33, last epoch 35, validation positive Dice at fixed 0.5
  `0.5106942783`, and selected inference threshold `0.25`. Source and split
  hashes, wrapper SHA-256 `afa5d4be...`, all 371/184/187 identities, all
  GT-area/subgroup assignments, training-log continuity, cloud artifact
  hashes and `test_evaluated=false` pass. The frozen authoritative reference
  is not overwritten. The 230,923,915-byte reproduction checkpoint was also
  downloaded independently and its physical SHA-256 verifies as
  `ad0605a97e716a195b9981ca4f22170048c6a6e82d751f8d783846a400bd0824`;
  the final auditor records `checkpoint_file_verified=true`.
- Reproduction Dice is `0.5113904441` overall versus frozen reference
  `0.4951316963`, a paired delta `+0.0162587478` with 95% CI
  `[-0.0161203857,+0.0495478497]`. Subgroup reproduction is not within 0.05
  everywhere: small is `0.3998633760` versus `0.3289549325`
  (`+0.0709084435`, CI `[+0.0249550436,+0.1193881516]`); medium is
  `0.6330372531` versus `0.6624417784` (`-0.0294045253`, CI
  `[-0.0801662047,+0.0181789871]`); large is `0.6072223413` versus
  `0.6937033566` (`-0.0864810152`, CI
  `[-0.1658684068,-0.0219844196]`). Thus overall Dice is reproducible within
  0.05, but seed-42 independent training is not subgroup-stable; this directly
  explains why a collaborator can obtain a different result under an
  apparently similar protocol.
- Both runs used Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8 and Tesla T4,
  and the same source/data hashes. The historical reference lineage resumed
  an epoch-10 checkpoint and selected epoch 20, whereas the reproduction
  started independently at epoch 1 and selected epoch 33. Their training
  trajectories already differ at epoch 1 despite seeding and cuDNN
  deterministic/benchmark flags; `torch.use_deterministic_algorithms` was not
  enabled. The evidence therefore localizes the discrepancy to independent
  optimization trajectory/GPU algorithm reproducibility rather than cohort,
  metric, threshold leakage, source drift or split drift. The reference stays
  hash-locked as the paired-study anchor, while the reproduction result must
  be reported as sensitivity evidence.
- The first GT-auditor pass exposed a path-schema mismatch: the reference lock
  names source files with a `project/` prefix while the cloud wrapper stores
  the same six paths relative to that directory. The auditor now normalizes
  only that optional prefix, rejects normalized duplicates, and still requires
  exact key/hash equality. Five focused tests pass. This was an audit
  implementation repair and did not alter any result or scientific gate.
- The next validation-only WSL experiment is predeclared from the rejected
  proposal-teacher mechanism audit. It keeps the exact teacher/CAM/SAM
  proposal pool, teacher threshold `0.85`, minimum area 20, maximum three
  teacher components, CAM-only support clipping (kernel 5), component-top-3,
  post-processing, cohort and promotion rule. The sole scientific variable is
  selector `source_consensus` in place of `coverage_mass_sam`.
  Each candidate receives fixed weights: CAM density `0.25`, global CAM-mass
  coverage `0.15`, within-prompt SAM rank `0.15`, component-local source-map
  coverage `0.25`, source-map density `0.10`, and maximum IoU with a candidate
  from the opposite proposal source `0.10`. CAM components use the frozen CAM
  as their source map; teacher components use the frozen teacher probability
  map. The implementation fails closed unless the teacher map, component
  boundary and component IDs are aligned. It never reads GT, lesion size or a
  per-image oracle. Three focused selector tests and 31 existing
  proposal/pipeline tests pass. Promotion still requires the paired
  complete-group overall delta CI95 lower bound above zero versus
  `0.2343392222` and no decrease on all 94 small tumors; test remains locked.
- The source-consensus experiment was launched as private Kaggle kernel
  `itsthang333/btxrd-wsl-source-consensus-val-v1`, version 1, and is RUNNING.
  Source is pinned to commit
  `80443fddec1cea8333905dff8650f5a2eeacff5d`; wrapper SHA-256 is
  `ed75e59158e2014532cecba50909c3eca0e72fe8c0c96be0abf6cda76f79b3dc`.
  It uses the same frozen classifier, SAM, proposal-teacher, split and promoted
  baseline artifacts as the audited proposal-teacher experiment. Heavy
  inference executes only on Kaggle; validation polygons remain
  post-prediction diagnostics and test is not accessed.
- Independent source-consensus auditing is prepared before completion in
  `project/tools/audit_source_consensus.py`. It verifies the cloud wrapper,
  source commit and changed-file hashes, exact six fixed selector weights,
  teacher/CAM/SAM/split provenance, generation command, unchanged CAM support,
  cohort/subgroups, compact artifact hashes, mechanism chain and recomputed
  paired promotion decision. Eight combined selector/auditor tests pass.
- Source-consensus validation version 1 completed and the independent audit
  passed. Source commit `80443fd...`, wrapper `ed75e591...`, split
  `85511ee1...`, classifier `f62d3702...`, SAM `ec2df627...`, frozen teacher
  `02d3af8f...`, generation parameters, exact six selector weights, downloaded
  evidence hashes and the complete 371/184/187 cohort all verify; test was not
  evaluated. Candidate per-image SHA-256 is
  `da42b92526320564f28cb5f08ca028b4ba93b2efc0439873108d347e1db0c85d`.
- The source-consensus selector is rejected. Final tumor Dice is
  `0.2125053135` versus promoted baseline `0.2343392222`, paired delta
  `-0.0218339086` with group-bootstrap CI95
  `[-0.0515454335,+0.0064740245]`. Small falls from `0.1121634055` to
  `0.0472296920`, delta `-0.0649337119`, CI95
  `[-0.1063383543,-0.0283010629]`, violating the explicit small-lesion gate.
  Medium improves `+0.0182890292` and large improves `+0.0427510904`, but these
  gains do not authorize train pseudo-mask generation.
- Mechanism decomposition confirms that added teacher proposals remain useful
  while the selector is harmful. Overall single-proposal oracle improves
  `+0.0400704867` and clipped oracle `+0.0220265163`, but selected Dice falls
  `-0.0220649034`. For small tumors oracle improves `+0.0419288505`, while
  selected Dice falls `-0.0648511286`; source-consensus increases the small
  selection loss by `+0.0905868438`. The next branch therefore changes the
  localization evidence itself through a frozen biomedical vision-language
  prior, rather than tuning these validation-derived selector weights or
  generating 2,981 train masks from a rejected candidate.
- Before any biomedical-VLM validation diagnostic, an implementation-only
  BiomedCLIP saliency smoke test is predeclared. It loads frozen
  `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` through
  `open_clip_torch==2.32.0`, with three fixed tumor and three fixed normal
  bone-radiograph prompts. Samples come only from train: within each binary
  image-label class, rows are sorted by SHA-256 of image ID; the first 32 are
  used for score diagnostics and the first four for saliency. Every source
  image hash must match the frozen split manifest.
- The smoke saliency is predeclared as absolute gradient-times-activation over
  channels at visual transformer block 11 `norm2`, targeted by mean frozen
  tumor text embedding minus mean frozen normal text embedding. The gate
  requires a physical pretrained-weight hash, finite scores/maps, dynamic
  range above `1e-6` on all eight maps and deterministic repeat delta at most
  `1e-5`. It reads no annotations, validation or test images and cannot promote
  train pseudo-mask generation. Source is pinned to
  `de02acb59900cc64f7bdf649d20286d6219af82c`; wrapper SHA-256 is
  `8aa182ade040082ac40be8c40ade8ea8af41c32245018a03c8bb0075b014d5af`.
- The private GPU kernel
  `itsthang333/btxrd-biomedclip-saliency-smoke-v1`, version 1, was launched and
  is RUNNING. The existing single 15-minute monitor was retargeted to this
  kernel; no duplicate monitor was created.
- Independent smoke auditing was prepared before completion in
  `project/tools/audit_biomedclip_smoke.py`. It fails closed on wrapper,
  source, split, model, runtime-version or prompt hash drift; reconstructs the
  exact 32+32 score and 4+4 saliency train-image selection from the frozen
  split; verifies image-label assignments, physical model-weight hashes,
  recomputed score summaries, finite/nonconstant maps and repeatability; and
  requires zero validation/test access. Five focused mutation tests plus three
  source-consensus auditor tests pass.
- BiomedCLIP smoke version 1 ended in an implementation error before any image
  was processed. Source checkout and model download succeeded, but
  `open_clip_torch==2.32.0` called `BertTokenizer.batch_encode_plus` against
  Kaggle's newer incompatible `transformers` runtime, raising
  `AttributeError: BertTokenizer has no attribute batch_encode_plus` during
  fixed-prompt tokenization. No score, saliency, validation/test access or
  scientific metric was produced.
- Version 2 changes only the environment contract by pinning
  `transformers==4.35.2`, the tokenizer version used by the inspected
  BiomedCLIP/MedCLIP-SAM compatibility stack. Model ID, pretrained weights,
  source commit, prompts, deterministic train sample identities, saliency
  definition and all gates are unchanged. The repaired wrapper SHA-256 is
  `3a881de74c7f2cadd152e67d079f29e39f92e14480fa84f374c76e19e51e372d`;
  the auditor now also verifies the predeclared and runtime transformers
  version. All five focused auditor tests pass after the repair.
- The repaired private GPU kernel
  `itsthang333/btxrd-biomedclip-saliency-smoke-v2`, version 1, was launched and
  is RUNNING. The same single 15-minute monitor was retargeted from the failed
  version 1 kernel to version 2; no additional monitor was created.
- While version 2 runs, the data-independent small-lesion saliency geometry was
  implemented in `project/models/biomedclip_saliency.py`: a padded full view
  plus fixed 3-by-3 half-short-side square crops, top-three local views selected
  only by frozen tumor-minus-normal text contrast, fixed 1st/99th percentile
  normalization, exact projection to the source grid and pixelwise-max fusion.
  No image, label, mask, subgroup or validation metric was used to implement or
  test this geometry. Four synthetic geometry/aggregation tests and five smoke
  auditor tests pass. The module remains prospective until the train-only smoke
  gate passes.
- A prediction-first saliency generator was added while the smoke run remains
  pending. `project/generate_biomedclip_saliency.py` permits only train/val,
  requires exact split and physical BiomedCLIP-weight hashes, verifies every
  source-image hash, creates exact empty maps for known-normal image labels,
  and stores float16 maps plus per-file hashes and fixed-view diagnostics before
  any GT evaluator is allowed. It never enumerates or opens `Annotations` and
  has no test-split CLI choice. Four focused manifest/map tests, four saliency
  geometry tests and five smoke-auditor tests pass. This is implementation
  preparation only, not validation evidence.
- The generator now also requires a full lowercase Git source commit and writes
  hashes for both the CLI and saliency implementation into its immutable run
  metadata. This closes the remaining source-provenance gap before a validation
  launch; it does not change the saliency method.
- Independent map auditing is prepared in
  `project/tools/audit_biomedclip_saliency.py`, pinned to source commit
  `34484b3e28cb22e97ae76c6569b42465b3a7fd25`. It reconstructs the exact
  train/val cohort from the frozen split; verifies source/model/prompt/view
  contracts; opens every map with pickle disabled; recomputes dtype, shape,
  range and hash checks; requires normal maps to be exactly empty and every
  tumor map to be nonconstant; verifies top-three tile ranking; and rejects
  missing or unmanifested maps. Four focused mutation tests bring the current
  BiomedCLIP preparation/auditor suite to 17 passing tests.
- BiomedCLIP smoke version 2 completed its train-only computation but failed
  the nonconstant-saliency implementation gate. The physical 783,705,670-byte
  BiomedCLIP weight hashes to
  `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`;
  source, split, prompt, 32+32 score cohort, 4+4 saliency cohort and
  `validation_masks_read=false`/`test_evaluated=false` all match the
  predeclaration. Frozen contrast scores are finite and show a small train-only
  tumor-minus-normal mean difference `+0.0048933686` with overall standard
  deviation `0.0257473165`. Repeat delta is exactly zero, but all eight 14x14
  maps have zero dynamic range, so no validation diagnostic is authorized.
- The zero-map failure is localized to the saliency reduction, not model
  loading or text/image scoring: summing `activation*gradient` over channels at
  LayerNorm `norm2` cancels to zero before the outer absolute value. Version 3
  replaces only that invalid reduction with channelwise
  `mean(abs(activation*gradient))`, and disables flash/memory-efficient SDP in
  favor of deterministic math SDP. Model, weight, source, prompt, train sample
  identities, target layer and all implementation gates remain unchanged.
  This repair uses no validation image, mask or metric. Version 3 wrapper
  SHA-256 is
  `e70928fc44485b02cd74874ae9398cfcb7f846e28baf82be9d2231f0403cc0c1`;
  the smoke auditor now fails closed on reduction drift. All 17 focused tests
  pass.
- The prospective prediction-first saliency source/auditor lock is advanced to
  repair commit `8b265e9c4cb75604eeb8612c8f750d31d0b3f0da`; generator and saliency-module
  SHA-256 values are `2366fd8c...` and `f07718a4...`, respectively. This lock
  applies only to a future full diagnostic after smoke version 3 passes.
- The repaired private GPU kernel
  `itsthang333/btxrd-biomedclip-saliency-smoke-v3`, version 1, was launched and
  is RUNNING. The existing single 15-minute monitor was retargeted to version 3;
  no duplicate monitor was created.
- Downstream ingestion was prepared without activating validation. The
  prediction-first map grid is fixed at 320 to match the promoted CAM grid.
  `generate_pseudo_masks.py` now accepts external saliency only when manifest,
  metadata, source commit, model weight, split, cohort, image-label and per-map
  hashes all verify; maps must be pickle-free float16, aligned, finite and in
  range. Known-normal maps must be exactly empty and tumor maps nonconstant.
  The first diagnostic forbids test, proposal-teacher and auxiliary-CAM mixing;
  after map replacement it retains the promoted morphology, SAM 512, prompt
  ensemble, selector, support clip and post-processing contract. Three focused
  ingestion tests, the 17 existing BiomedCLIP tests and 37 broader
  proposal/pipeline regressions pass. This remains inactive pending smoke V3.
- The prospective full-diagnostic source lock is advanced to commit
  `8a997c87170538f897e6aa3b13b0f6c13e39f32f`; generator, saliency module and
  external-map pseudo generator hash to `c475f3b8...`, `f07718a4...` and
  `35462180...`, respectively. No validation map or metric has been produced
  under this source lock yet.
- The conditional full-validation protocol is frozen before smoke V3 returns
  and before any BiomedCLIP validation prediction or metric. Protocol
  `artifacts/research_protocols/biomedclip_tiled_val_v1.json` has SHA-256
  `d20712790808dc14fc5c6d882502872a04edb4b2d516fa986a1be57122217e1c`.
  It fixes the 371/184/187 cohort, 94/72/18 subgroups, physical model/source/
  prompt hashes, tiled-view geometry, exact promoted downstream settings and
  prediction-before-GT order. Passing the small-lesion oracle gate authorizes
  only selector/support research; direct train-mask generation still requires
  positive overall final-Dice CI95 lower bound and no small-lesion decrease.
- The independent pre-launch protocol audit passes. It verifies the exact
  protocol file hash, frozen CRLF paired split `85511ee1...`, three source-file
  hashes, the physical BiomedCLIP weight evidence from smoke V2, and both
  promoted baseline evidence hashes. It records
  `validation_predictions_generated=false`, `validation_gt_read=false` and
  `test_evaluated=false`. The saliency-manifest auditor was corrected from the
  repository LF serialization to this authoritative paired split hash before
  any launch. Four protocol mutation tests plus four saliency-auditor tests
  pass.
- A saliency-only full-validation Kaggle wrapper is prepared but not launched
  before smoke V3 passes. Wrapper SHA-256 is
  `dfade1d3598d8d19d74d56d0be0ff1e66191ff0f54d19bfdba27de67c492183e`.
  It writes the predeclaration first, checks out the locked source, reconstructs
  the paired split, generates/hashes all 371 maps, and runs the independent map
  auditor. It contains no segmentation evaluator or pseudo-mask stage and
  records zero validation-GT/test access.
- BiomedCLIP smoke version 3 also fails the nonconstant-saliency gate with the
  same physical weight, cohort and finite contrast scores; repeat delta remains
  zero and all eight patch maps remain exactly zero. No validation/test access
  occurred. Channelwise mean-absolute reduction therefore fixed cancellation
  but exposed a second implementation issue: block 11 `norm2` is after the last
  self-attention, so its patch-token MLP outputs have no path to the pooled CLS
  token and their gradients are zero.
- Version 4 moves only the hook from block 11 `norm2` to block 11 `norm1`,
  immediately before final self-attention where patch keys/values affect CLS.
  Reduction, deterministic math-SDP, model, weight, prompts, train cohort and
  gates stay fixed. This is the standard causal location for last-block ViT
  patch gradients and was selected without validation data. V4 wrapper
  SHA-256 is
  `ef0521b9a9080bce722d72c71ae4a773fe4d446008eedea88907c5b118a99f86`.
  The previously prepared full-validation wrapper is invalidated and remains
  unlaunched until V4 passes and the amended protocol/source hashes re-audit.
- The conditional validation protocol was amended before any validation
  prediction to target `model.visual.trunk.blocks[11].norm1`, activate only on
  smoke V4, and lock repaired implementation commit `95fc1c24...`. Amended
  protocol SHA-256 is `9f5b2250...15ea8`; generator/module hashes are
  `a160a5a7...`/`18858511...`. The full independent pre-launch protocol audit
  passes again and still records no validation prediction, validation-GT read
  or test access.
- The repaired private GPU kernel
  `itsthang333/btxrd-biomedclip-saliency-smoke-v4`, version 1, was launched and
  is RUNNING. The same single 15-minute monitor was retargeted to V4; the
  saliency-only full-validation kernel remains unlaunched.
- The saliency-only full-validation wrapper was rebuilt against the independently
  re-audited `norm1` protocol and current repository snapshot. Its new SHA-256
  is `e34171b67f0bca069d488eaa47610ae6cbc6415e2ac6a9f45f306eb1a8b02a30`.
  It remains conditional and unlaunched pending an audited V4 smoke pass.
- Static wrapper audit passes and verifies the exact wrapper, repository,
  implementation, protocol, protocol-audit, split and physical model-weight
  hashes. It rejects annotation/segmentation-evaluator tokens, requires the
  `norm1` target and both zero-access flags, and has four passing mutation
  tests. The audited wrapper remains unlaunched.
- BiomedCLIP smoke V4 completed and independently passes the implementation
  gate. Exact wrapper/source/model/prompt and deterministic train-sample
  identities verify; the physical weight remains `52cc993c...`. Recomputed
  32+32 train contrast difference is `+0.0048933701`, all eight saliency maps
  are finite and nonconstant, minimum dynamic range is `3.1789790e-05`, and
  exact repeat delta is zero. Population is train-only with zero validation/test
  images; `validation_masks_read=false` and `test_evaluated=false`.
- The first local audit invocation exposed only checkout serialization drift:
  cloud/source split is canonical LF `43662d5d...`, while Windows materializes
  the same file as CRLF `85511ee1...`. The smoke auditor now accepts only that
  exact CRLF-to-LF canonicalization and rejects any remaining carriage return
  or content drift; one focused test covers the repair. Scientific evidence,
  samples and gates are unchanged. The conditional activation gate for the
  independently audited saliency-only full-validation kernel is now satisfied.
- The private GPU kernel
  `itsthang333/btxrd-biomedclip-tiled-saliency-val-v1`, version 1, was launched
  and is RUNNING with audited wrapper `e34171b6...`. This stage generates and
  hashes all 371 saliency maps and runs the independent manifest auditor only;
  it contains no pseudo-mask/segmentation evaluator and cannot read validation
  GT. The same single 15-minute monitor was retargeted from smoke V4 to this
  kernel; no duplicate monitor was created.
- A third independent fully supervised GT reproduction is predeclared after an
  external similar experiment reported a different score. It starts from epoch
  1 and preserves the frozen `gt_resnet18_unet_448_v1` consumer contract:
  ResNet18UNet, ImageNet encoder, 448 px, seed 42, batch 8, fixed
  `pos_weight=10`, AdamW `1e-4`, 35 epochs, patience 10 and the frozen
  validation threshold grid. The run is a reference-stability audit only, uses
  no test data and cannot replace the hash-locked reference automatically.
  Protocol is
  `artifacts/research_protocols/gt_reference_reproduction_v3.json`; wrapper
  SHA-256 is `4080d04f...f0193`. The post-run audit will compare v3 pairwise
  against both the frozen reference and reproduction v2 at image, complete
  validation-group and lesion-size-subgroup levels.
- Static determinism audit of the frozen GT training source passes its recorded
  seed controls but identifies explicit limitations relevant to the reported
  collaborator discrepancy. Python/NumPy/Torch/CUDA RNGs are seeded, cuDNN
  deterministic mode is enabled, benchmarking is disabled, and resume restores
  model/optimizer/AMP plus global RNG states. However,
  `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG` are not
  enabled; DataLoaders have no explicit generator or worker-init function;
  persistent worker RNG states are neither checkpointed nor restored; and AMP
  remains active. Reference and independent v2 trajectories already differ at
  epoch 1. These facts identify plausible reproducibility gaps but do not prove
  a specific CUDA/DataLoader mechanism. The frozen reference remains valid and
  unchanged; v3 will distinguish current fresh-run repeatability from
  historical-lineage sensitivity. Evidence:
  `artifacts/reference/gt_resnet18_unet_448_v1/reproducibility_static_audit.json`.
- The physical ImageNet ResNet-18 encoder input was independently downloaded
  as the single 46,830,571-byte
  `resnet18-f37072fd.pth` file from the exact Kaggle dataset attached by GT
  reproduction v2/v3. Its SHA-256 is
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
  Wrapper hashes prove both runs locate that unique filename and copy it into
  the Torch cache, but the wrappers do not assert the physical weight hash
  inside the cloud process. The evidence is therefore marked
  `PASS_WITH_PROVENANCE_LIMITATION`, not overstated as an in-kernel proof.
  Every future paired consumer wrapper must fail closed on this weight hash.
  Evidence:
  `artifacts/reference/gt_resnet18_unet_448_v1/pretrained_weight_audit.json`.
- The first full-validation BiomedCLIP saliency kernel failed after 4.07
  seconds, before model loading or any radiograph/validation prediction. Direct
  Kaggle logs show `RuntimeError: Frozen protocol-audit SHA-256 mismatch`.
  Root cause is cross-platform text serialization only: the wrapper had locked
  the Windows CRLF byte hash `b009cad4...`, whereas the same committed
  `biomedclip_tiled_val_v1_audit.json` is checked out by Git on Kaggle as LF
  with canonical hash `844cd93c...`. Protocol, source, model, split and
  scientific settings were not reached; validation GT/test access remains
  false and no metric exists for this failed version.
- The repaired wrapper normalizes only CRLF/CR to LF for this exact text audit
  file and fails closed on canonical SHA-256
  `844cd93cd5240c917e15b4c3dbce011514ea6d1bb4847c6095e8c91617e54225`.
  Its new wrapper SHA-256 is
  `c3b5088dfbe1ac7713af440ef541395d9968f03d371b75f91e5129c8428476cf`.
  Static wrapper audit passes again with the unchanged repository commit,
  implementation, protocol, split, physical BiomedCLIP weight, `norm1` target,
  no-GT/no-test flags and saliency-only stage; four mutation tests pass. The
  repaired heavy rerun is held until it will not interfere with the active GT
  reproduction priority.
- Static paired-consumer source audit now proves the current WSL consumer has
  not silently drifted from the frozen GT consumer. `models/unet.py`,
  `models/losses.py`, `evaluate_unet.py` and
  `evaluation/segmentation_metrics.py` are canonical-LF byte-identical.
  The exact reviewed `train_segmentation.py` diff adds only fail-closed
  paired-reference validation and provenance; optimizer, AMP, DataLoaders,
  seed, model, loss and train/validation loops are unchanged. The exact
  `datasets/btxrd.py` diff only permits a pseudo-mask source grid different
  from consumer 448 px and records resize provenance; GT construction, image
  and mask transforms and paired flip are unchanged. Both complete diff hashes
  are fail-closed, so any further drift breaks the audit. The sole scientific
  arm difference remains the training mask source, validation GT remains
  evaluation/model-selection only, and test remains locked. Evidence:
  `artifacts/reference/gt_resnet18_unet_448_v1/paired_consumer_source_equivalence_audit.json`.
- Independent fully supervised GT reproduction v3 completed on Kaggle in
  3,186.30 seconds and passed the fail-closed reproduction auditor. It started
  at epoch 1, completed all 35 epochs, independently reselects epoch 33 at
  fixed-threshold validation positive Dice `0.4952337807`, and selects
  inference threshold `0.20` from the frozen grid. Exact source/split/wrapper
  hashes, the 371/184/187 validation cohort, 167 tumor groups, `94/72/18`
  lesion-size groups, contiguous training log, threshold selection, per-image
  artifact hashes and `test_evaluated=false` all pass.
- V3 selected mean tumor Dice is `0.4994012363` overall and
  `0.3521898858/0.6293762322/0.7482716392` for small/medium/large. Signed gaps
  to the hash-locked reference are
  `+0.0042695401/+0.0232349533/-0.0330655462/+0.0545682827`.
  Overall, small and medium fall within absolute `0.05`; large exceeds the
  bound by `0.0045682827`. Paired complete-group CI95 values are
  `[-0.0304210,+0.0403183]` overall,
  `[-0.0279916,+0.0735878]` small,
  `[-0.0890728,+0.0204204]` medium and
  `[-0.0222736,+0.1468771]` large. This is GT reproducibility evidence, not a
  WSL success claim.
- Physical v3 checkpoint verification passes: the downloaded
  230,923,915-byte selected checkpoint hashes to
  `65bb34613f53134ce3b0b9469278e5e5ec1da87d4bbed01e0b2807b21a99ad95`.
  The 46,830,571-byte ResNet-18 weight copied and used inside the kernel hashes
  to `f37072fd...e07ec`, resolving the earlier provenance limitation. Neither
  large binary is retained in Git; compact logs, per-image results, summaries,
  audits, wrapper and metadata are retained.
- The three-run stability audit confirms the collaborator discrepancy is real
  run-to-run training instability, not cohort/metric/source drift. Fresh v2
  and v3 differ from epoch 1 and select different checkpoint bytes despite the
  same seed/source/split/environment contract. V3 minus v2 mean Dice is
  `-0.0119892077` overall,
  `-0.0476734902/-0.0036610209/+0.1410492979` for
  small/medium/large; the large paired CI95
  `[+0.0355631,+0.2661626]` excludes zero. Across the frozen reference, v2 and
  v3, subgroup ranges are `0.0709084` small, `0.0330655` medium and
  `0.1410493` large, versus only `0.0162587` overall. Per-image v2-v3 mean
  absolute Dice difference is `0.1508196`; 102/184 tumors differ by more than
  0.05 and 74/184 by more than 0.10.
- Decision: retain `gt_resnet18_unet_448_v1` as the predeclared hash-locked
  paired-study anchor. Replacing it after observing repeated validation
  results would invalidate comparability. The thesis must report v2/v3 as
  sensitivity evidence and explicitly warn that the 18-image large subgroup
  is unstable. A stricter deterministic GT run may be a separately
  predeclared sensitivity arm, never a silent replacement. Compact evidence,
  report-ready comparison and decision are under
  `artifacts/kaggle/gt_reference_independent_reproduction_v3/`.
- With the requested GT rerun/audit complete, WSL research resumed. The
  repaired saliency-only kernel
  `itsthang333/btxrd-biomedclip-tiled-saliency-val-v1`, version 2, was
  launched and is RUNNING. Static audit immediately before launch passes
  wrapper SHA-256 `c3b5088d...`, unchanged protocol `9f5b2250...`,
  canonical-LF protocol-audit `844cd93c...`, source/model/split locks,
  prediction-first ordering, `validation_gt_read=false` and
  `test_evaluated=false`. This kernel only generates/hashes 371 BiomedCLIP
  saliency maps and runs the independent map auditor; downstream pseudo-mask
  generation remains unauthorized until the map audit passes. The single
  existing 15-minute monitor was retargeted from GT v3 to this version; no
  duplicate monitor was created.
- BiomedCLIP full-validation saliency v1 version 2 completed on Kaggle in
  164.16 seconds and passed both the in-kernel independent audit and a local
  compact-evidence audit. The frozen cohort is exactly 371 validation images
  (`184` tumor, `187` normal, 371 unique IDs); all 371 manifest rows contain
  valid map hashes. Every tumor map has positive dynamic range and every known
  normal map is exactly zero under the predeclared image-label policy.
  Saliency-manifest SHA-256 is
  `72360237a12802c06ea5da8cecde7dbb87d4fef7a9dbf358f09e080384267bf1`;
  metadata SHA-256 is
  `06109965042b9c433126101f7e32c609c8069312af58b2d0e6693ad4e77ccc4b`.
  Local source hashes exactly match the Kaggle run for the generator, model
  and independent auditor. Validation polygon GT was not read and test was not
  evaluated. Maps were intentionally not downloaded locally; the next Kaggle
  stage must consume the hash-locked kernel output directly.
- The former inline `--evaluate-prompt-quality` path was retired because it
  loaded validation GT before prediction generation had finished. The
  replacement is a two-phase, prediction-first diagnostic contract:
  generation saves pickle-free candidate/prompt/support/selected/final NPZ
  artifacts for all 184 image-level-positive cases (including classifier-gate
  complete misses), then freezes their hashes and cohort against the
  pseudo-mask manifest. Only the separate
  `evaluate_saved_candidate_diagnostics.py` process may load validation GT,
  and only after both caller-locked manifests pass. Test is unsupported.
  This preserves error decomposition into saliency support, prompts, proposal
  oracle, selector and post-processing without allowing GT to influence the
  predictions being diagnosed.
- Predeclared downstream protocol
  `biomedclip_tiled_downstream_val_v2` was hash-locked and launched as Kaggle
  kernel `itsthang333/btxrd-biomedclip-tiled-downstream-val-v2`, version 1.
  Protocol SHA-256 is
  `8f1fbbe273ed07b05f22401a651df42cde301540e2bda58267792b25bed84b56`;
  wrapper SHA-256 is
  `d70ae1568b7da3a00df89eef4c9fc1408c4eab6e6712c22f0466444c21e55d7b`.
  The wrapper first verifies all 371 physical saliency-map hashes, then
  generates all pseudo masks and all 184 tumor candidate diagnostics without
  segmentation GT, writes an explicit prediction-freeze artifact, and passes
  both frozen manifest hashes to the separate validation-GT evaluator. It then
  evaluates the final 371-mask cohort and performs 10,000-iteration paired
  complete-group bootstrap overall and for the fixed `94/72/18`
  small/medium/large subgroups. The scientific parameters are unchanged from
  the parent predeclared protocol; this version changes only the formerly
  leakage-prone execution ordering. Test remains locked. Kernel status after
  launch is RUNNING, and the single 15-minute monitor was retargeted without
  creating a duplicate.
- The prediction-first BiomedCLIP downstream kernel completed in 550.1
  seconds and passed the local compact-evidence audit: all declared wrapper,
  protocol, split, source, saliency, prediction-freeze, pseudo-mask and
  candidate-manifest hashes match; the exact `371/184/187` cohort is present;
  complete misses are retained; validation GT is first read by the separate
  post-freeze evaluator; and `test_evaluated=false`.
- Direct BiomedCLIP replacement is rejected. Final tumor Dice falls from
  `0.2343392222` to `0.1760893904`, paired complete-group CI95 for the
  `-0.0582498318` delta is `[-0.09982265,-0.01839313]`. Small falls
  `0.1121634039 -> 0.02578231`; medium `0.3486037144 -> 0.31764299`; large
  `0.4153105265 -> 0.39481197`. Neither the oracle research gate nor the
  direct-train-mask gate passes.
- Mechanism decomposition explains why the arm still contains useful signal.
  Raw best-single oracle Dice changes by `-0.08847388/+0.04288522/+0.06791886`
  for small/medium/large, while selection loss worsens strongly for medium and
  large. Small-lesion failure occurs before support/morphology: foreground
  recall falls `-0.042835`, positive-point hit rate falls `-0.071061`, and raw
  oracle falls `-0.088474`, whereas its support loss improves.
- A post-freeze feasibility diagnostic takes the maximum raw single-candidate
  Dice over the two already frozen galleries. This is not an executable
  GT-oracle router. The unconditional gallery has oracle Dice `0.4837261732`,
  `+0.0746498827` over baseline with 10,000-resample complete-group CI95
  `[+0.05277334,+0.09832107]`. Gains are positive in every fixed subgroup:
  small `+0.03701584` (`[+0.02287461,+0.05389866]`), medium `+0.11669329`
  (`[+0.07082705,+0.16560901]`), and large `+0.10300960`
  (`[+0.03699213,+0.18664071]`).
- Decision: preserve the promoted flip-TTA LayerCAM proposals, selector map,
  support and post-processing; append the frozen BiomedCLIP component/SAM
  proposal gallery unconditionally. Pixel blending, replacement, GT-oracle
  routing, lesion-size routing and subgroup-specific tuning remain forbidden.
  This bounded arm tests whether the existing selector can exploit a strictly
  richer candidate pool; it does not authorize train pseudo masks.
- Protocol `biomedclip_layercam_proposal_gallery_val_v1` is predeclared before
  any gallery-union prediction. It locks implementation commit
  `1f6847d9326d9e8a789e1924a3a2a7735f5a1f0f`, protocol SHA-256
  `8b5883780df881ecb654ee9a6d431a56a175d560c41f34f0ca6170f8a54a8c1d`,
  the exact saliency/checkpoint/split/baseline hashes and the unchanged
  `coverage_mass_sam` selector/support/post-processing recipe. Prediction
  manifests must be frozen before validation GT is loaded. Promotion to train
  pseudo masks requires a positive overall paired-CI lower bound and no mean
  final-Dice decrease in any fixed size subgroup; test remains locked.
- Kaggle kernel
  `itsthang333/btxrd-biomedclip-layercam-proposal-gallery-val-v1`, version 1,
  was launched and is RUNNING. Static wrapper audit passes SHA-256
  `fd70b742ed8d6758222a1a83109344aba4018433ba48f4889ef35412b60658cf`;
  protocol SHA-256 remains `8b588378...`. The wrapper additionally compares
  every tumor image against the frozen baseline manifest before GT access:
  LayerCAM component counts must be identical, SAM candidate counts cannot
  decrease, proposal-source counts must balance, and all 184 tumor candidate
  diagnostics must freeze successfully. The single existing monitor was
  retargeted to this kernel at the unchanged 15-minute interval.
- Proposal-gallery kernel version 1 failed after 501.47 seconds with
  `KeyError: 'parameters'` in the wrapper-only post-generation audit. The
  no-GT generation subprocess had completed, but the wrapper assumed a nested
  `run_metadata["parameters"]` object while the hash-locked generator writes a
  flat metadata schema. Prediction manifests were not frozen, the separate
  validation-GT evaluator was never invoked, and test remained untouched; no
  scientific result exists for version 1.
- The repair changes only that wrapper lookup to use the flat `run_metadata`
  object. Source commit, implementation, protocol, split, checkpoints,
  saliency maps, generation command, selector, support, metric and promotion
  gates remain unchanged. Version 2 is held until the repaired wrapper hash
  and static audit are updated. The repaired wrapper compiles and passes the
  static ordering/contract audit at SHA-256
  `767af8ddde1edbdf5d94e12f053e5ee8df51f380bab20e092869c3148730ec8c`.
- Repaired kernel version 2 was launched and is RUNNING. The single existing
  15-minute monitor was updated in place to version 2 and the repaired wrapper
  hash; no duplicate monitor or competing heavy job was created.
- A fourth fully supervised GT reproducibility run was predeclared after a
  collaborator reported a different result from a similar experiment. This is
  an exact byte-identical rerun of the v3 Kaggle wrapper (SHA-256
  `4080d04f...f0193`) as kernel version 2, with the same frozen source, split
  SHA-256 `85511ee1...c8c`, seed 42, 448-pixel ResNet18UNet consumer contract,
  371-image validation cohort and `94/72/18` lesion-size subgroups. The purpose
  is to isolate stochastic/runtime variation from configuration or metric
  drift; it cannot replace the hash-locked reference and test remains locked.
- Proposal-gallery version 2 completed and passed the independent and local
  compact-evidence audits. All retained artifact hashes match the run
  manifest; prediction artifacts were frozen before validation GT; the exact
  `371/184/187` cohort and `94/72/18` subgroups include complete misses;
  LayerCAM component counts are preserved, SAM candidate counts never decrease,
  every tumor receives external proposals, and test remains untouched.
- The direct promotion gate fails. Final mean tumor Dice changes only
  `0.2343392 -> 0.2373441` overall (`+0.0030049`, complete-group CI95
  `[-0.0049258,+0.0106602]`); small and medium improve by `+0.0020985` and
  `+0.0141854`, but large declines by `-0.0369836` with CI95 entirely below
  zero `[-0.0789123,-0.0072707]`. These masks are therefore not authorized
  for train-pseudo generation.
- The selector-research gate passes because the unchanged unconditional
  proposal gallery retains the previously frozen positive oracle gains:
  `+0.0746499` overall, `+0.0370158/+0.1166933/+0.1030096` for
  small/medium/large, with every complete-group CI95 lower bound positive.
  The evidence isolates proposal selection—not source availability—as the
  remaining failure, especially for the 18-image large subgroup. Next WSL work
  may study a no-GT selector only; no size routing or GT-oracle routing is
  authorized.
- Fully supervised GT kernel version 2 (scientific role v4) was launched and
  is RUNNING. The single 15-minute monitor was retargeted to this exact rerun;
  the completed proposal-gallery kernel no longer has a monitor.
- Fully supervised GT reproduction v4 completed on Kaggle after 3,215.81
  seconds. The run is kernel version 2 at
  `itsthang333/btxrd-gt-reference-independent-reproduction-v3`; its scientific
  role remains `gt_reference_independent_reproduction_v4`, not a WSL result.
  Test was not evaluated.
- The post-download physical audit passes. The selected checkpoint is
  230,923,915 bytes with SHA-256
  `7336b466213550b3bb27fd099c1e5986a60399452204318ad7105dc9abb4373c`.
  The downloaded ImageNet ResNet-18 encoder weight is 46,830,571 bytes with
  SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
  Both large binaries were verified from the kernel output and omitted from
  compact Git evidence.
- The exact locked upload wrapper is retained at SHA-256
  `4080d04f6042ea03ba872b5bbee7ac6c0060f64ce08c3719e8a102e5d25f0193`.
  Pulling the current Kaggle source changes its byte representation, producing
  raw SHA-256 `3ba582ea...50ed3`, but a BOM-aware, line-ending-independent
  comparison proves identical line content. This is recorded as a transport
  byte limitation, not contract drift.
- A monitor-text typo contained a 65-character alleged protocol digest
  (`...b3ddbddcdd247`), which cannot be a SHA-256. The authoritative protocol
  file was committed before launch in commit `9a2890b` and has the valid
  64-character SHA-256
  `f06aad247d505c81d595ea474c960e7ef32816383e4f2656fd50b3ddbdcdd247`.
  The audit fails closed against this physical pre-launch file.
- Frozen-source hashes and split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`
  match. Independent recomputation from the retained training log selects
  epoch 23, with early stopping at epoch 33; independent threshold
  recomputation selects 0.25 from the frozen grid. The exact validation cohort
  is `371/184/187`, fixed size subgroups are `94/72/18`, all 184 tumor images
  remain in the primary mean, and the 17 selected-threshold complete misses
  equal the subgroup sum `11+5+1`.
- V4 selected mean tumor-only Dice is `0.5030822743` overall:
  small `0.3588136129`, medium `0.6587714893`, and large `0.6337284231`.
  Against the frozen reference, paired deltas are
  `+0.00795058/+0.02985868/-0.00367029/-0.05997493` for
  overall/small/medium/large. Complete-group 10,000-bootstrap CI95 values are
  `[-0.02091496,+0.03749890]`, `[-0.01739264,+0.07713623]`,
  `[-0.04280468,+0.03569585]`, and `[-0.14201474,+0.00639340]`.
  Thus v4 is within 0.05 for overall, small and medium, but not large; it is
  sensitivity evidence only and does not alter the WSL success target.
- The four audited GT results are:
  frozen `0.49513170/0.32895493/0.66244178/0.69370336`,
  v2 `0.51139044/0.39986338/0.63303725/0.60722234`,
  v3 `0.49940124/0.35218989/0.62937623/0.74827164`, and
  v4 `0.50308227/0.35881361/0.65877149/0.63372842`, ordered as
  overall/small/medium/large. Four-run ranges are
  `0.01625875/0.07090844/0.03306555/0.14104930`; restricting to the three
  fresh epoch-1 runs gives
  `0.01198921/0.04767349/0.02939526/0.14104930`.
- V3 and v4 have no audited environment difference and use the same
  wrapper/source/split/seed/consumer contract, yet their numeric trajectories
  diverge at epoch 1 (`val_positive_dice 0.21135136` versus `0.22591704`) and
  select different checkpoint bytes. V4 minus v3 is
  `+0.00368104/+0.00662373/+0.02939526/-0.11454322`; the large-subgroup
  complete-group CI95 is entirely negative
  `[-0.23037363,-0.03069912]`. This confirms stochastic/runtime optimization
  instability rather than contract drift. It does not isolate one CUDA, AMP,
  DataLoader-worker or other numeric kernel as the sole cause.
- Decision: retain the hash-locked GT reference
  `gt_resnet18_unet_448_v1`. Replacing it after observing validation
  sensitivity would invalidate paired WSL comparability. Report v2/v3/v4 as
  sensitivity analysis, emphasizing that large contains only 18 images.
  Compact evidence and the four-run decision are stored under
  `artifacts/kaggle/gt_reference_independent_reproduction_v4/`.

## 2026-07-26 — Prompt/source graph selector predeclaration

- The complete prior log was re-audited before choosing the next direction.
  Repeated scalar selectors (`prompt_hybrid`, classifier-causal,
  `source_consensus` and related CAM/source weighting) did not recover richer
  proposal pools. The frozen LayerCAM plus BiomedCLIP gallery is different:
  its unchanged final selector changes Dice by only `+0.0030049` overall and
  harms large lesions by `-0.0369836`, while its raw single-candidate oracle
  improves overall/small/medium/large by
  `+0.0746499/+0.0370158/+0.1166933/+0.1030096`, with every paired CI95 lower
  bound positive. This locks proposal selection, not proposal availability, as
  the immediate bottleneck.
- A no-GT selector was implemented and isolated in commit
  `abdcaca482d676b45a902ad3d927c832877a39d4`. It first selects a robust SAM
  medoid per morphology component using lexicographic cross-prompt agreement,
  then clusters component medoids at fixed IoU `0.50`, prioritizes clusters
  supported by both independent proposal sources, and retains single-source
  clusters as a fallback for lesions missed by one source. No learned or
  validation-fitted score weights, true lesion size, subgroup identity,
  segmentation GT or GT-oracle routing are available to the selector.
- Candidate diagnostics now preserve aligned per-candidate proposal-source
  provenance in a backward-compatible schema-v2 payload. The frozen LayerCAM
  support clip and all proposal generation, prompt, SAM, morphology and
  post-processing settings remain unchanged.
- Thirteen focused tests pass, including cross-source priority,
  single-source small-lesion fallback, support clipping, provenance
  fail-closed behavior and the existing prediction-first/gallery contracts.
  The full local suite passes 135 tests; two unrelated pre-existing GT
  stability-audit tests require Python's `zip(..., strict=True)` and cannot run
  in the available Python 3.9 research environment. This is an environment
  compatibility issue outside the selector and will be rechecked in Kaggle's
  runtime before generation.
- Protocol `prompt_source_graph_selector_val_v1` is physically predeclared
  before any new prediction at SHA-256
  `b0a4f49755a3bf3cf316823addb9bf4d7a9cc7109f192bdb23ac69abea5c0bb0`.
  It locks commit/file/split/checkpoint/saliency/baseline hashes, the
  `371/184/187` cohort and `94/72/18` fixed subgroups. Predictions and
  schema-v2 diagnostics must be hash-frozen before validation GT is loaded.
  Direct promotion requires an overall paired final-Dice CI95 lower bound
  above zero and no mean degradation in any subgroup. Train pseudo masks and
  paired U-Net consumer training remain forbidden unless this Gate-C passes;
  test remains locked.
- Kaggle kernel
  `itsthang333/btxrd-prompt-source-graph-selector-val-v1`, version 1, was
  launched from source commit `370f635fbe74f89d2450dfe94d7f88545167f8b3`
  and is RUNNING. The standalone wrapper compiles and passes its static
  source/command/freeze-order/schema audit at SHA-256
  `8c99a03bb46ebc0ae2f510b8be847a80368e4a1368c9060b7e51b053fc8e12ef`.
  It verifies both proposal sources and required prompt-mode provenance in
  every schema-v2 tumor diagnostic before freezing prediction hashes and
  invoking either validation-GT evaluator. A single 15-minute monitor
  `theo-d-i-prompt-source-graph-kaggle` follows this kernel; no competing
  heavy local task or duplicate monitor was created.
- Kernel version 1 failed after 8.6 seconds in the pre-generation source audit:
  `generate_pseudo_masks.py` did not match the wrapper's expected SHA-256.
  Direct console evidence confirms checkout commit `370f635`; no generation
  subprocess, prediction manifest, validation-GT evaluator or test access
  occurred. The cause is that four expected hashes were taken from the
  Windows CRLF working-tree representation rather than the canonical LF Git
  blobs that Kaggle checks out.
- The repair changes only those transport representations in the protocol and
  wrapper: canonical Git-blob hashes are now
  `c958a38f...be185` for `generate_pseudo_masks.py`,
  `21f48976...a965126` for `pseudo/mask_selection.py`,
  `acae150b...db2d` for `evaluate_pseudo_masks.py`, and
  `48467c99...d7e404` for `pseudo/manifest.py`. Selector implementation,
  proposal inputs, parameters, execution order, cohort, metric and promotion
  gates are unchanged. The repaired protocol SHA-256 is
  `f6fbc130ebd353ac8ba59552dff87ceed04a70895d936bdae648afe45cf8c50e`.
- Repaired kernel version 2 was launched from source commit
  `e8233ae48f62a526647ea5dba13a482d50f2c111` and is RUNNING. Before upload,
  the wrapper independently compared every expected digest with bytes returned
  by `git show` at that exact commit, rechecked the prediction-freeze ordering,
  compiled successfully and passed at SHA-256
  `99a3e058b19286cc99487a4df210773329151bcad31cc30985bd4200acbaffec`.
  The existing 15-minute monitor was updated in place to version 2; no second
  monitor or competing heavy run was created.
- Repaired version 2 completed in 653.70 seconds on a Tesla T4. Local
  acceptance re-hashed the protocol and wrapper, compared every frozen source
  digest with `git show` bytes at commit `e8233ae...`, verified schema-v2
  provenance for all 184 tumor diagnostics, and verified that the 371-row
  prediction manifest was frozen before validation GT was loaded. Cohort is
  exactly `371/184/187`; subgroups are `94/72/18`; complete misses are
  included; `test_evaluated=false`.
- Independent complete-group bootstrap with 10,000 resamples reproduces the
  kernel result exactly. Prompt/source graph final Dice is `0.20341410` versus
  the frozen flip-TTA baseline `0.23433922`, delta `-0.03092512`, CI95
  `[-0.06408515,+0.00145140]`. Small falls by `-0.08184462`, while medium and
  large change by `+0.01191864/+0.06361279`. Direct Gate-C therefore fails and
  paired U-Net consumer training is forbidden.
- The richer gallery remains mechanistically valuable: raw best-single
  candidate oracle improves overall/small/medium/large by
  `+0.07464988/+0.03701584/+0.11669329/+0.10300960`, with every oracle CI95
  lower bound positive. The selector failure is more severe for small tumors:
  choosing whole SAM masks via prompt/source consensus still favors
  over-broad components. This rejects the graph selector, not the independent
  proposal sources.
- Compact evidence and the local acceptance audit are stored under
  `artifacts/kaggle/prompt_source_graph_selector_val_v1/`. Prediction-freeze
  SHA-256 is `3310bc59...f159c`; pseudo/candidate manifests are
  `4c8c6992...67fd` and `c96dc766...2ee`; independent local comparison SHA-256
  is `d0d45135...a78a`.

## 2026-07-26 - Technique-first literature transfer audit

- The literature search was broadened beyond deployable medical checkpoints.
  Methods were decomposed into transferable mechanisms and cross-checked
  against every prior experiment in this log. The resulting audit is stored in
  `artifacts/literature_review/technique_transfer_review_2026-07-26.md`;
  checkpoint/model availability remains separately documented in
  `medical_xray_wsss_candidate_landscape_2026-07-26.md`.
- Highest-priority new mechanism is a SKELEX-inspired masked-reconstruction
  anomaly map. The full paper, not only its abstract, was inspected. SKELEX
  initializes a ViT-Large MAE from ImageNet, adapts it to 1,296,540 unlabeled
  musculoskeletal radiographs with 75% patch masking and normalized pixel loss
  disabled, then averages pixel reconstruction errors from ten random masks.
  The paper explicitly demonstrates this zero-shot localization mechanism on
  BTXRD. The transferable hypothesis is that a completely masked small tumor
  is reconstructed as normal bone, creating a local residual independent of a
  global classification CAM.
- The proposed bounded adaptation does not claim to reproduce SKELEX and does
  not require its unpublished checkpoint. It will first test an open
  ImageNet-pretrained ViT-MAE adapted only on the clean BTXRD training
  partition, with all heavy learning/inference on Kaggle. A normal-only
  reconstruction arm may use the binary image label but no polygon/mask.
  Reconstruction maps must be hash-frozen before validation GT is loaded.
- Other high-potential transferable mechanisms are UM-CAM's spatial
  uncertainty-weighted multi-resolution fusion, geodesic seed expansion, and
  Random-View Consensus for a noise-robust consumer; progressive
  prototype/affinity expansion; token contrast using intermediate ViT
  features; and direct dense MIL with learnable LSE/probabilistic pooling.
  AdvCAM, tested S2C SSC/CPM, generic SAM grids, direct BiomedCLIP localization,
  blind tiling, and scalar selection are explicitly marked closed to prevent
  repetition under a new name.

## 2026-07-26 - Predeclared MAE normality-reconstruction mechanism probe

- Protocol `mae_normality_reconstruction_probe_val_v1` was frozen before
  execution. This is a bounded mechanism probe, not a SKELEX reproduction,
  pseudo-mask candidate, final segmentation result, or permission to train a
  downstream U-Net.
- Two arms use the identical immutable `facebook/vit-mae-base` snapshot:
  (A) the untouched ImageNet MAE and (B) a fixed-final-epoch checkpoint adapted
  for 20 epochs on clean-train normal radiographs only. The normal/tumor
  image-level flag is the only task label used; annotation paths are not
  enumerated or opened in adaptation or prediction generation.
- Both arms use the same ten seed-42 masks per validation image at 448 px.
  Masked-pixel reconstruction error maps for all 371 validation images must be
  physically present and hash-frozen before any validation GT is opened.
  Diagnostics are continuous pixel AP/AUROC, argmax hit and saliency mass in
  GT, plus non-selective fixed percentile Dice at p90/p95/p97/p99. Adapted
  minus base is compared with complete-group bootstrap 10,000 for
  overall/small/medium/large. Test remains locked.
- The decision target is specifically whether normality modeling supplies
  complementary localization for the 94 small-tumor cases. Even a positive
  probe will require a new predeclared fusion/pseudo-mask experiment; this run
  cannot promote a threshold or consumer.

## 2026-07-26 - Normal-only anomaly-localization transfer review

- While the MAE probe runs, the technique audit was extended beyond named
  medical checkpoints to PatchCore, Reverse Distillation, DRAEM, Natural
  Synthetic Anomalies, and attention-conditioned augmentation. Detailed
  mechanism/transfer/risk analysis is recorded in
  `artifacts/literature_review/technique_transfer_review_2026-07-26.md`.
- The strongest follow-up if pixel reconstruction is weak is a
  context-conditioned nominal patch memory: retrieve visually similar normal
  train radiographs, then score each query patch against their mid-level
  features. This uses only images plus the normal/tumor train label and avoids
  the rejected whole-mask selector.
- Synthetic-anomaly segmentation is lower priority. It remains protocol-valid
  when synthetic masks are generated from normal images, but it must pass
  prediction-first localization on real tumors; synthetic training Dice is
  explicitly disallowed as promotion evidence.

## 2026-07-26 - User-approved paired-goal revision v2

- The user reduced the required minimum WSL Dice by another `0.05` in each
  lesion-size subgroup. Operationally, the allowed absolute mean-Dice gap to
  the unchanged frozen GT reference is now `<=0.10` per subgroup.
- With reference Dice
  `0.3289549325/0.6624417784/0.6937033566`, the revised minimum WSL Dice is
  `0.2289549325/0.5624417784/0.5937033566` for
  small/medium/large. The corresponding absolute-gap intervals are
  `[0.2289549325,0.4289549325]`,
  `[0.5624417784,0.7624417784]`, and
  `[0.5937033566,0.7937033566]`.
- This revision changes only the success tolerance. The frozen GT checkpoint,
  split, subgroup definitions, consumer architecture/training/evaluation
  contract, WSL supervision restrictions, complete-miss policy and test lock
  remain unchanged. Historical protocol v1 and results evaluated at `0.05`
  are not rewritten.
- Future paired consumers must cite
  `artifacts/reference/gt_resnet18_unet_448_v1/paired_protocol_v2.json` and run
  the authoritative auditor with the physically locked protocol, canonical-LF
  SHA-256
  `2f7965b2ece0c00e9db6441562c489f84b5ccb942619a3c6a3d08ca2328359d0`.
  An unlocked CLI tolerance is insufficient for future promotion. The
  in-flight MAE localization probe is unaffected because it trains no paired
  consumer and has its own already-frozen mechanism gate.

## 2026-07-26 - MAE normality-reconstruction probe completed and rejected

- Kaggle kernel
  `itsthang333/btxrd-mae-normality-reconstruction-probe-v1` completed under
  the predeclared protocol. The run binds source commit
  `7292c5d2f7722d273c27eb147b19cbe7b25c9709`, wrapper SHA-256
  `5c250532045fe4344b4f29046f0b49a038c0e7162240445d32d5e30a099ceb6b`,
  protocol SHA-256
  `06055700e48980ddab0ab87c9e36dace7c7aab103d5b6fc004fa5dd742f8da06`
  and split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`.
  It used 1,493 clean-train normal images, zero tumor images and zero
  segmentation annotations for the 20-epoch adaptation arm.
- Prediction generation physically produced 371 base maps and 371
  normal-adapted maps, each arm totaling 76,028,288 bytes. Both arms share
  the same ten seed-42 masks and noise-bank SHA-256
  `22020f1b455daca7bc1b88391225374daadf659c99c16bb1282bfa3d0f741b50`.
  The adapted checkpoint is 447,670,680 bytes with SHA-256
  `f64a74ca802bf9c5e97797e1c8d03db82cd39496884a3ebad1be38b502a8b020`.
  Prediction-freeze SHA-256 is
  `997d325db94dafed3913c4af984ce2e7aebac225f9faf053103d61f49ca3b894`;
  the wrapper order independently verifies base/adapted generation before
  freeze, evaluation, then paired comparison.
- The independent local auditor returned `PASS`. It re-hashed all 742 maps,
  the adapted checkpoint, metadata/manifests/evaluations, and exactly
  reproduced the complete-group bootstrap with 10,000 resamples. Cohorts are
  371 validation images, 184 tumors, 187 normals and fixed subgroups
  94/72/18. Complete misses are included, no consumer was trained, validation
  GT was read only after prediction freeze, and `test_evaluated=false`.
- Absolute localization quality is far below a useful pseudo-mask mechanism.
  Base versus adapted overall pixel AP is `0.0338572 -> 0.0348382`, pixel
  AUROC `0.6392444 -> 0.6423162`, saliency mass in GT
  `0.0291108 -> 0.0297702`, and fixed-p90 Dice
  `0.0409011 -> 0.0422803`. Image-level p99 AUROC worsens
  `0.4418449 -> 0.4392002`.
- The target 94-case small subgroup does not improve coherently. Base versus
  adapted pixel AP is `0.00334690 -> 0.00333345`, saliency mass in GT
  `0.00239732 -> 0.00231582`, argmax hit remains `0/94`, and fixed-p90 Dice
  is `0.00439892 -> 0.00427730`. Paired deltas and 95% CIs are respectively
  AP `-0.00001345 [-0.00024674,+0.00023024]`, saliency mass
  `-0.00008150 [-0.00027920,+0.00008434]`, and p90 Dice
  `-0.00012162 [-0.00083533,+0.00058581]`.
- The only coherent gain is on the 18 large tumors: pixel AP delta
  `+0.00897512 [95% CI +0.00248353,+0.01623120]`, saliency-mass delta
  `+0.00845876 [+0.00204250,+0.01483683]`, p90 Dice delta
  `+0.01686208 [+0.00686339,+0.02769057]`, and p95 Dice delta
  `+0.01064547 [+0.00347519,+0.01831977]`. This repeats the earlier failure
  pattern where coarse/global anomaly evidence helps conspicuous large
  lesions but does not resolve sub-1% tumors.
- Decision: reject both raw MAE reconstruction residual and normal-only MAE
  adaptation as standalone pseudo-mask or fusion arms. Do not select a
  percentile threshold and do not train a consumer from this result. The
  failure is mechanistic, not a selector failure: a 16x16-patch reconstruction
  loss plus ten random masks provides insufficient spatial and contrast
  sensitivity for tiny lesions, while exposure/anatomy reconstruction error
  dominates the map.
- Compact evidence is stored under
  `artifacts/kaggle/mae_normality_reconstruction_probe_val_v1/`; the
  independent audit is `local_acceptance_audit.json`, and the paired
  comparison SHA-256 is
  `eb1d20f8115beaa6a2bcf414802e3b56e71f37f69c3de9c552472e386b9b70b7`.
  The reconstructible 447 MB checkpoint and 152 MB map payload remain only in
  the ignored temporary audit directory.
- Next bounded direction: context-conditioned nominal patch memory. It will
  compare mid-level local features against visually similar clean-train
  normal radiographs, use a frozen normal-only calibration rather than
  per-image normalization, freeze all validation anomaly maps before GT, and
  use the same continuous/fixed diagnostic gate before any fusion or
  consumer is permitted.

## 2026-07-26 - Predeclared context-conditioned nominal patch-memory probe

- Protocol `nominal_patch_memory_probe_val_v1` is frozen before execution.
  This is a RAD-DINO/PatchCore-inspired transfer experiment, not a PatchCore
  reproduction, final pseudo-mask pipeline, threshold-selection run or
  authorization to train a U-Net.
- The frozen feature extractor is `microsoft/rad-dino` at immutable revision
  `110cbc18d5133582e320b43d53bf5c44e410c936`; model SHA-256 is
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`.
  RAD-DINO is selected because its official model card explicitly exposes
  dense patch tokens for radiograph segmentation/retrieval; no fine-tuning is
  performed.
- The normal memory uses all 1,493 eligible clean-train normal radiographs
  selected by the binary image label, zero tumor training images and zero
  segmentation annotations. Images are black-square-padded at 448 px. Frozen
  final-layer 32x32 patch tokens are projected 768 to 128 by a seed-42
  Gaussian matrix and L2-normalized.
- Every query first retrieves the top eight normal images by CLS cosine
  similarity. Each query patch is then scored as one minus its best cosine
  match within a fixed plus/minus-two-patch coordinate window in those
  context images. This prevents anatomically unrelated regions from becoming
  nominal matches.
- Calibration is a leave-one-image-out empirical CDF fitted only on normal
  train distances; the source normal is explicitly excluded from its own
  context. Full-view and pooled-tile calibrations are separate and frozen.
  There is no validation fit and no per-image min-max normalization.
- Two and only two predeclared prediction arms are generated for all 371
  validation images before GT access: `single_scale` uses the full 448 view;
  `multiscale` adds four fixed overlapping 280 px corner crops, overlap-averages
  their maps, then uses the fixed formula `0.5*full + 0.5*tiles`. No parameter
  sweep is permitted.
- After the 742 maps and manifests are hash-frozen, both arms are evaluated on
  the unchanged 184-tumor cohort and 94/72/18 subgroups using continuous
  localization metrics plus fixed p90/p95/p97/p99 diagnostics. Multiscale
  minus single-scale is compared by complete-group bootstrap 10,000.
- Promotion is deliberately non-automatic. A coherent improvement on small
  lesions without material overall/medium/large regression can only justify
  a new frozen fusion/pseudo-mask protocol. Test stays locked and no consumer
  is trained in this probe.
- Prelaunch verification passed 14 CPU tests, Python compilation, source/blob
  hash checks and wrapper ordering audit. Wrapper SHA-256 is
  `cfe2f5ef9c52f0c3ba22df0470efda53ec487607305a55b1735470a5247c0662`;
  protocol SHA-256 is
  `458fee51bef0fa5754be11566db7c5ea7d08cbd8c0d97c2477d267f626494a35`.
  The CUDA spatial-matching smoke test is required to pass on the T4 before
  the heavy feature-bank stage begins.
- Kaggle kernel
  `itsthang333/btxrd-rad-dino-nominal-memory-probe-v1` version 1 was launched
  and entered `RUNNING`. A single five-minute heartbeat monitor
  `theo-d-i-rad-dino-nominal-memory-kaggle` tracks it; no duplicate monitor
  was created. Prelaunch evidence is stored in
  `artifacts/research_protocols/nominal_patch_memory_probe_val_v1_wrapper_audit.json`.

## 2026-07-26 - RAD-DINO nominal patch-memory probe completed

- Kernel `itsthang333/btxrd-rad-dino-nominal-memory-probe-v1` completed on a
  Tesla T4. The independent physical auditor returned `PASS` and re-hashed
  the wrapper, source commit, protocol, split, RAD-DINO snapshot, memory
  evidence, both prediction manifests, all 742 maps, both evaluation files
  and the paired comparison. Wrapper SHA-256 is
  `cfe2f5ef9c52f0c3ba22df0470efda53ec487607305a55b1735470a5247c0662`;
  source commit is `30f62d9475949dd43c9ad19c0590a0cbc854d440`;
  protocol SHA-256 is
  `458fee51bef0fa5754be11566db7c5ea7d08cbd8c0d97c2477d267f626494a35`;
  split SHA-256 is
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  RAD-DINO weights SHA-256 is
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`.
- The image-only contract is intact: all 1,493 clean-train normal images
  formed the memory; zero tumor train images and zero segmentation
  annotations were used; leave-one-image-out normal calibration excluded the
  source normal from its own context; prediction maps for all 371 validation
  images were frozen before validation GT access; complete misses remain
  included; `consumer_trained=false` and `test_evaluated=false`.
  Cohorts are `371/184/187` with fixed subgroups `94/72/18`.
- The single-scale arm's absolute localization metrics are
  overall pixel AP/AUROC `0.09333895/0.76755187`, small
  `0.01796705/0.78714348`, medium `0.10891928/0.72357058`, and large
  `0.42462648/0.84116532`. Fixed p90 Dice is
  `0.08465377/0.01298239/0.10669106/0.37078849` for
  overall/small/medium/large. These are mechanism diagnostics, not a
  pseudo-mask or segmentation score.
- The fixed multiscale arm adds four overlapping 280px views with the
  predeclared `0.5*full + 0.5*tiles` merge. Its absolute pixel AP/AUROC is
  `0.09828856/0.78238737` overall,
  `0.02459231/0.80165948` small,
  `0.11203034/0.73843316` medium, and
  `0.42817970/0.85756097` large. Fixed p90 Dice is
  `0.08596578/0.01330257/0.10614308/0.38472003` for
  overall/small/medium/large. Argmax hit on small remains only `2.13%`
  (`2/94`), so the map is a ranking signal rather than a usable shape.
- Independent complete-group bootstrap with exactly 10,000 resamples
  reproduces the kernel result. Multiscale minus single-scale deltas on the
  small subgroup are pixel AP `+0.00662526`
  (95% CI `[-0.00207647,+0.01629728]`), pixel AUROC `+0.01451601`
  (`[+0.00072478,+0.02888615]`), saliency mass in GT
  `+0.00007170` (`[+0.00003725,+0.00011346]`), p90 Dice
  `+0.00032018` (`[-0.00048671,+0.00115653]`) and p99 Dice
  `+0.00789809` (`[+0.00229802,+0.01439491]`). The AUROC, saliency-mass
  and p99 signals are coherent, while AP and ordinary fixed-percentile
  overlap remain weak.
- Medium and large show no material regression: pixel AUROC improves by
  `+0.01486258` and `+0.01639565` with CIs excluding zero; medium p90/p95/
  p97/p99 Dice deltas all have CIs crossing zero, while large p90/p95 gains
  are positive but imprecise because `n=18`. Image-level raw-p99 AUROC is
  `0.61500233 -> 0.61352011`, a reminder that the memory is a local
  localization cue, not a better binary classifier.
- Decision: do not promote either arm, choose no threshold from validation,
  and do not train a consumer from this probe. The mechanism is not rejected
  outright: it supplies complementary small-lesion ranking evidence, but its
  coarse 32x32 map and low argmax-hit rate cannot form a stand-alone
  pseudo-mask. A separate predeclared image-label-only fusion experiment is
  justified. It will use the fixed normal-calibrated RAD-DINO score as an
  auxiliary local-evidence channel for an already frozen CAM/SAM seed source,
  with all train pseudo-masks generated before validation GT access and no
  validation-fitted threshold, selector or per-image routing.
- Compact audited evidence is stored under
  `artifacts/kaggle/nominal_patch_memory_probe_val_v1/`; reconstructible
  742-map payloads remain in the ignored temporary audit directory. The
  five-minute monitor was deleted after the terminal result. Test remains
  locked.

## 2026-07-26 - Predeclared RAD-DINO dense-MIL localization probe

- The nominal-memory result retained statistically coherent small-lesion
  ranking signal but failed as a shape source. The next bounded experiment
  therefore changes the learning mechanism rather than choosing a
  validation threshold: a trainable dense MIL head is applied to frozen
  RAD-DINO patch tokens and supervised only by the clean-train binary
  image-level tumor flag.
- Protocol `rad_dino_dense_mil_probe_val_v1` was frozen before execution at
  SHA-256
  `785da0fce22f40ad4863c69368292b0c45b78c7506f386f87fdc73be1154f438`.
  The frozen encoder is the same immutable RAD-DINO snapshot and weight hash
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`.
  The only trainable module is `LayerNorm(768) -> Linear(768,1)` on the
  32x32 patch grid. Image logits use temperature-`0.20` Log-Sum-Exp pooling,
  optimized for exactly 12 epochs, batch 8, AdamW `1e-3`, weight decay
  `1e-4`, seed 42 and training-only horizontal flip.
- Training uses all 2,981 clean-train images and their binary image labels;
  no annotation path, validation label, validation image or test image enters
  optimization/checkpoint selection. The fixed final epoch is used, so no
  validation-fit early stopping is possible.
- Two and only two validation prediction arms are declared. `single_scale`
  uses the full 448 view. `multiscale` adds the same four fixed overlapping
  280px corner views and merges them by the fixed
  `0.5*full + 0.5*overlap-averaged tiles` formula. Maps are sigmoid
  probabilities with no per-image min-max, no threshold selection and no
  validation calibration.
- All 371 maps in both arms must be physically present and hash-frozen before
  validation GT is opened. Evaluation is limited to continuous AP/AUROC,
  argmax hit and saliency mass plus fixed p90/p95/p97/p99 diagnostics on the
  unchanged 184 tumors and 94/72/18 subgroups. Every metric is compared by
  paired complete-group bootstrap with 10,000 resamples. The run cannot
  promote a threshold, generate train pseudo-masks or train a consumer.
- Source commit is
  `2ce616ccd90650869fb7f820836d236213f8f1f9`. Prelaunch CPU/Torch smoke and
  Python compilation pass. The wrapper independently matches every canonical
  Git source hash and verifies the execution order train -> predictions ->
  freeze -> validation-GT evaluation -> comparison. Wrapper SHA-256 is
  `a15ec0304e7ec12e8c9c80dd92638cb89bf0816fc7aabdf4d72ddf22af3c55a5`;
  the audit is stored in
  `artifacts/research_protocols/rad_dino_dense_mil_probe_val_v1_wrapper_audit.json`.
- Private Kaggle kernel
  `itsthang333/btxrd-rad-dino-dense-mil-probe-v1` version 1 was launched on
  a Tesla T4 and entered `RUNNING` at 2026-07-26 17:51 ICT. A single
  five-minute heartbeat monitor
  `theo-d-i-rad-dino-dense-mil-kaggle` follows it. No duplicate monitor or
  competing heavy local job was created.

## 2026-07-26 - RAD-DINO dense-MIL probe v1 environment error

- Kaggle kernel `itsthang333/btxrd-rad-dino-dense-mil-probe-v1` version 1
  terminated before training. The direct wrapper log reports
  `RuntimeError: transformers must be 4.50.2, got 5.0.0` at the explicit
  environment-version guard in `run_rad_dino_dense_mil_probe.py`.
- The failure is an implementation/environment packaging omission in the
  Kaggle wrapper: unlike the already audited RAD-DINO nominal-memory and MAE
  wrappers, it did not install the predeclared `transformers==4.50.2`.
  Therefore no dense-MIL checkpoint, validation prediction, validation-GT
  evaluation or test access occurred in version 1.
- The admissible correction is limited to installing
  `transformers==4.50.2` before model loading and execution, with
  `TOKENIZERS_PARALLELISM=false`. The frozen source commit, protocol, split,
  RAD-DINO weights, training schedule, validation arms and evaluation
  contract remain unchanged.
- The corrected wrapper has SHA-256
  `7464a7c13a2187bed40ffe5c40f16fad4bad66d2e1341e6609e991fb927a6c10`;
  the prelaunch audit record was updated without changing the scientific
  source/protocol hashes. Kernel version 2 was pushed to the same private
  Kaggle kernel and the existing heartbeat monitor was updated in place to
  use the new wrapper hash at its five-minute interval.

## 2026-07-26 - RAD-DINO dense-MIL probe v2 frozen-encoder autograd error

- Kernel version 2 successfully installed `transformers==4.50.2` and then
  failed at the first dense-MIL training batch. The direct traceback was
  `RuntimeError: Inference tensors cannot be saved for backward` at
  `DenseMILHead.norm`, because frozen RAD-DINO tokens produced inside
  `torch.inference_mode()` were passed directly to the trainable head.
  No validation prediction, validation-GT evaluation or test access occurred.
- The implementation correction clones encoder tokens after leaving
  `inference_mode`, producing ordinary non-gradient tensors that can safely
  be consumed by the trainable head while keeping the encoder frozen. The
  DataLoader worker seeder was also hardened to seed Python/NumPy from
  `torch.initial_seed()` without invoking CUDA inside forked workers.
  Scientific parameters and the image-label-only supervision contract are
  unchanged.
- The corrected source is pinned at commit
  `e0741a9fddca7bf3fdac93e21dee3c8dfb4b6cc1`; the canonical source hash for
  `run_rad_dino_dense_mil_probe.py` is
  `88084d9bfb8ec9bae14dfa558d06d113c926a94d2ef7851c0d254e05050f08fe`.
- The corrected wrapper SHA-256 is
  `58f68c087a43a6cba3cf128ac0f8d31fee15504d7544a04e0d17f415e1f2e37e`.
  Kernel version 3 was pushed to the same private Kaggle kernel, and the
  existing five-minute monitor was updated in place with the new source and
  wrapper hashes.

## 2026-07-26 - INSIGHT mechanism transfer predeclared

- While RAD-DINO dense-MIL version 3 is running, the full INSIGHT paper was
  read from the primary PMLR version (Zhang, Chen and Kanan, MLHC 2025,
  PMLR 298) rather than relying on a checkpoint or an abstract. Its useful
  transferable mechanism is architectural: retain a spatial feature map,
  use a small-kernel detection branch for fine detail, use a broad-context
  branch to suppress local false positives, fuse them as
  `sigmoid((1-sigmoid(context_logits))*detector_logits)`, and train the
  resulting heatmap with image-level SmoothMax pooling. The paper also
  reports spectral-decoupling regularization and an Otsu visualization
  threshold.
- This is not a reproduction of INSIGHT and no external code, checkpoint or
  dataset is copied. The BTXRD transfer is predeclared as a separate
  image-label-only mechanism probe: frozen RAD-DINO 32x32 patch tokens,
  1x1 projection, a 3x3 local detector, a depthwise-separable 9x9 context
  suppression branch, bounded SmoothMax pooling, and fixed spectral
  decoupling. The depthwise context implementation is an explicit
  parameter-efficient deviation for a T4, and the diagnostic keeps
  continuous heatmaps rather than selecting an Otsu/validation threshold.
- The reason for this bounded follow-up is the measured bottleneck: the
  current dense-MIL probe uses only `LayerNorm -> Linear` per patch, while
  the literature's strongest small-lesion mechanism adds locality and
  context suppression inside the trainable head. This changes the learned
  representation rather than repeating rejected post-hoc selectors.
- The local implementation is `project/models/rad_dino_insight.py` with
  CPU-independent shape/backward tests in
  `tests/test_rad_dino_insight.py`. The probe must not launch until dense-MIL
  version 3 is physically audited; it will use the same 448/320/280 geometry,
  2,981 image-level train labels, 371 validation predictions, fixed
  94/72/18 subgroup evaluation, complete-group bootstrap with 10,000
  replicates, prediction freeze before validation GT, and
  `test_evaluated=false`. It cannot promote a threshold, create train
  pseudo-masks, or train a consumer on its own.
- The prelaunch wrapper audit is `PRELAUNCH_PASS` with wrapper SHA-256
  `5708c75603f838a9b9d753a9cc8a893eb295a2ae8a74a459f80939e766251a42`,
  source commit
  `813d6848941ac6a3ebe77538f4c0e34a0ddf7f4a`, protocol SHA-256
  `12289c77839a2c24d6e9ad92165fb1d95f46cf9c6b51ec8c6b693eba6aded83f`,
  and head source SHA-256
  `4897971f74a85f8181e28891d36a2efd9a3449d41ca60e63f95ea6bad66a7a03`.
  The complete static evidence is
  `artifacts/research_protocols/rad_dino_insight_probe_val_v1_wrapper_audit.json`.
- Primary literature links for the transfer record:
  `https://proceedings.mlr.press/v298/zhang25a.html` and
  `https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html`.
  The latter independently reinforces the project decision to report
  small-lesion performance explicitly and to avoid letting large lesions
  dominate the research conclusion.

## 2026-07-26 - Frozen CLIP-DINO / WeCLIP+ transfer review

- The primary WeCLIP paper and ablations were inspected from the authors'
  arXiv record and official code link
  `https://arxiv.org/abs/2406.11189`. The transferable idea is not merely
  using a stronger encoder: a lightweight decoder learns spatial features
  from a frozen backbone, while a refinement module builds decoder affinity
  (`sigmoid(F_u^T F_u)`) and uses it to filter/refine the frozen-backbone
  relationships. The decoder and refinement supervise each other through
  online pseudo-label refinement, avoiding a permanently frozen CAM.
- The reported ablation isolates the mechanism: decoder-only mIoU is 68.7,
  decoder plus refinement is 74.9, and the refinement map's affinity,
  global-evidence and selected-attention terms all contribute. The paper
  also uses multi-scale inference and DenseCRF, but those components are
  not evidence for BTXRD and will not be copied without a separate
  predeclared contract.
- Transfer assessment: this is a credible second-stage candidate if
  INSIGHT-style local/context heatmaps remain too coarse. A BTXRD version
  would use frozen RAD-DINO patch tokens, a small trainable spatial decoder,
  image-level BCE/SmoothMax, and a fixed token-affinity consistency/refinement
  loss derived only from the decoder and image labels. It would need a
  prediction-first localization probe before any pseudo-mask or consumer.
  No implementation or Kaggle launch is authorized yet; the current
  dense-MIL result and the predeclared INSIGHT probe must resolve first.

## 2026-07-26 - RAD-DINO dense-MIL localization probe completed and rejected

- Kaggle kernel `itsthang333/btxrd-rad-dino-dense-mil-probe-v1` version 3
  completed on a Tesla T4 after the two implementation-only corrections
  recorded above. The independent physical auditor returned `PASS`.
  Wrapper SHA-256 is
  `58f68c087a43a6cba3cf128ac0f8d31fee15504d7544a04e0d17f415e1f2e37e`;
  source commit is `e0741a9fddca7bf3fdac93e21dee3c8dfb4b6cc1`;
  probe source SHA-256 is
  `88084d9bfb8ec9bae14dfa558d06d113c926a94d2ef7851c0d254e05050f08fe`;
  protocol SHA-256 is
  `785da0fce22f40ad4863c69368292b0c45b78c7506f386f87fdc73be1154f438`;
  split SHA-256 is
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  RAD-DINO weights SHA-256 is
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`;
  checkpoint SHA-256 is
  `945cff3221190014437a0a34dda88935477d6ad6ea07fb03cec54e39c5801d3e`.
- The image-only contract is intact: 2,981 clean-train image labels train
  the head; no segmentation annotation, validation label or test image enters
  optimization; all 742 validation maps are physically hash-checked and
  frozen before GT access; complete misses are included; cohorts are
  `371/184/187` with subgroups `94/72/18`; `consumer_trained=false` and
  `test_evaluated=false`. Compact evidence and the audit are stored under
  `artifacts/kaggle/rad_dino_dense_mil_probe_val_v1/`.
- Absolute single-scale localization is weak: overall pixel AP/AUROC
  `0.01722582/0.32162069`, small `0.00161379/0.32844999`, medium
  `0.02193040/0.30773659`, large `0.07993705/0.34149299`; small-lesion
  argmax hit is `0/94`. Fixed p90 Dice is only
  `0.00855865/0.00089750/0.01289031/0.03124031` for
  overall/small/medium/large. These are continuous mechanism diagnostics,
  not segmentation Dice.
- The fixed multiscale arm is not an improvement: overall pixel AP/AUROC
  `0.01628222/0.31358044`, small `0.00159403/0.31381836`, medium
  `0.01966358/0.30227604`, large `0.07946175/0.35755558`; small argmax hit
  remains `0/94`. Fixed p90 Dice is
  `0.00654377/0.00061396/0.00885131/0.02828032`.
- Independent paired complete-group bootstrap with exactly 10,000
  replicates gives multiscale minus single-scale overall pixel AP
  `-0.00094361` (95% CI `[-0.00285399,+0.00039675]`) and pixel AUROC
  `-0.00804025` (CI `[-0.02607627,+0.00994470]`). On small lesions the
  deltas are pixel AP `-0.00001976` (CI
  `[-0.00018363,+0.00014127]`), pixel AUROC `-0.01463163` (CI
  `[-0.04587535,+0.01670747]`), saliency mass `-0.00005748` (CI
  `[-0.00022129,+0.00011235]`) and p90 Dice `-0.00028353` (CI
  `[-0.00088184,+0.00025407]`). No small-lesion metric improves coherently;
  medium is also slightly down and large gains are too imprecise at `n=18`.
- Decision: reject the plain dense-MIL `LayerNorm -> Linear` head and both
  scales as standalone pseudo-mask sources; no validation threshold,
  consumer U-Net, or downstream pseudo-mask is authorized from this run.
  This is a mechanism failure rather than a contract failure. The result
  supports moving to the already predeclared INSIGHT-style local-detector
  plus broad-context-suppression head, while keeping the same frozen
  RAD-DINO encoder, image-label-only supervision and prediction-first audit.

## 2026-07-26 - INSIGHT mechanism probe launched after dense-MIL audit

- The dense-MIL probe reached a terminal result and passed its independent
  physical audit, satisfying the prelaunch condition for the separate
  INSIGHT transfer probe. No dense-MIL threshold, pseudo-mask or consumer was
  promoted.
- Kaggle kernel
  `itsthang333/btxrd-rad-dino-insight-mechanism-probe-v1` version 1 was
  pushed as a private Tesla T4 run and entered `RUNNING`. The actual Kaggle
  slug follows the URL returned by the push; no duplicate dense-MIL job was
  created.
- The probe remains mechanism-only: frozen RAD-DINO patch tokens, a local
  3x3 detector branch, depthwise-separable 9x9 context-suppression branch,
  fixed fusion, SmoothMax image-level pooling and spectral-decoupling
  regularization. It uses 2,981 clean-train binary image labels only,
  produces fixed single/multiscale validation maps, freezes all maps before
  validation-GT evaluation, and cannot select a threshold, create training
  pseudo-masks, train a consumer or access test.
- Prelaunch evidence remains `PRELAUNCH_PASS` at wrapper SHA-256
  `5708c75603f838a9b9d753a9cc8a893eb295a2ae8a74a459f80939e766251a42`,
  source commit `813d6848941ac6a3ebe77538f4c0e34a0ddf7f4a`, protocol SHA-256
  `12289c77839a2c24d6e9ad92165fb1d95f46cf9c6b51ec8c6b693eba6aded83f`,
  split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c` and
  RAD-DINO weight SHA-256
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`.
  Status polling must remain on the exact URL slug above; no test is read
  while the kernel is non-terminal.

## 2026-07-26 - INSIGHT probe v1 canonical-source hash error

- Kernel version 1 terminated after 3.21 seconds in the wrapper's pre-run
  source audit, before dependency installation, model loading, radiograph
  access, training, validation prediction or validation-GT evaluation.
  Direct Kaggle log reports
  `RuntimeError: Source hash mismatch: datasets/btxrd.py`.
- Independent comparison against Git bytes at source commit
  `813d6848941ac6a3ebe77538f4c0e34a0ddf7f4a` shows that all eight new
  INSIGHT/RAD-DINO sources match. Only the three reused dataset files were
  recorded from the Windows CRLF worktree rather than canonical LF Git blobs.
  Their corrected canonical hashes are
  `d8f0804b...156001`, `1927eb35...3b98a` and
  `6b478bbb...32784` for `datasets/btxrd.py`, `datasets/common.py` and
  `datasets/__init__.py`.
- This is a transport-representation/packaging failure, not a scientific
  result. The admissible repair changes only those three expected hashes in
  the frozen protocol and wrapper, then rebinds the wrapper to a new source
  commit. Model, split, image-label-only supervision, architecture, training
  schedule, prediction arms, evaluation and test lock remain unchanged.
- The corrected protocol is bound to source commit
  `da676c9e862d36792fcbe8fdea268a557ecafe2c` with canonical protocol
  SHA-256
  `a7aa1adfd99237193f1cb7ea61a049e7195d60fb73994bcd71a51cdadab0918a`.
  An independent local comparison confirms all 11 declared project-source
  hashes exactly match `git show` bytes at that commit. The corrected wrapper
  compiles and hashes to
  `839a1256e6a1c602d28dca9379f861e567d8deafa0adaab04b52f3c5323afd1e`;
  the prelaunch audit remains `PRELAUNCH_PASS`.
- Kernel version 2 was pushed to the same private Kaggle kernel and entered
  `RUNNING`. The existing single five-minute monitor was updated in place
  with the corrected hashes; no duplicate monitor or competing heavy job was
  created.

## 2026-07-26 - INSIGHT mechanism probe completed and rejected

- Kaggle kernel
  `itsthang333/btxrd-rad-dino-insight-mechanism-probe-v1` version 2
  completed. The independent physical auditor returned `PASS`, re-hashing
  the wrapper, probe/head sources, source commit, protocol, split, RAD-DINO
  snapshot, checkpoint, prediction freeze, all 742 maps, both manifests,
  evaluation files and the stored paired comparison. Wrapper SHA-256 is
  `839a1256e6a1c602d28dca9379f861e567d8deafa0adaab04b52f3c5323afd1e`;
  source commit is `da676c9e862d36792fcbe8fdea268a557ecafe2c`;
  protocol SHA-256 is
  `a7aa1adfd99237193f1cb7ea61a049e7195d60fb73994bcd71a51cdadab0918a`;
  split SHA-256 is
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  RAD-DINO weight SHA-256 is
  `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`;
  checkpoint SHA-256 is
  `35bc926bb5768e1f2879dd7ae8ce37ac1e8538ed1223bcd65e81424f2f950286`.
- The weak-supervision and evaluation contract is intact: training uses
  2,981 clean-train binary image labels and no segmentation annotation or
  validation input; every validation prediction is frozen before GT access;
  cohorts are `371/184/187`, subgroups are `94/72/18`, complete misses are
  included, `consumer_trained=false` and `test_evaluated=false`.
- Single-scale absolute localization improves substantially over the rejected
  linear dense-MIL head but remains weak. Overall pixel AP/AUROC is
  `0.03207112/0.53918555`; small `0.00553422/0.54775549`; medium
  `0.03905904/0.54029862`; large `0.14270105/0.48997917`.
  Overall/small/medium/large fixed-p90 Dice is
  `0.03117199/0.00513944/0.04241465/0.12214912`; small argmax hit remains
  `0/94`.
- Relative to dense-MIL single-scale, INSIGHT raises overall pixel AP/AUROC
  by `+0.01484530/+0.21756486` and p90 Dice by `+0.02261334`; on small it
  raises AP/AUROC by `+0.00392043/+0.21930550` and p90 Dice by
  `+0.00424195`. Thus local detection plus context suppression is a real
  architectural improvement over a per-patch linear scorer, but it still
  fails to identify the tumor maximum in every small case.
- The stronger normal-memory single-scale arm remains clearly better:
  INSIGHT versus nominal memory is lower by `-0.06126783` overall pixel AP,
  `-0.22836631` overall pixel AUROC and `-0.05348178` overall p90 Dice.
  On small it is lower by `-0.01243283/-0.23938799/-0.00784295`, and its
  argmax hit is `0/94` versus `2/94`. Therefore INSIGHT does not replace the
  retained nominal-memory ranking signal.
- Fixed multiscale fusion degrades the INSIGHT map. Overall pixel AP/AUROC
  changes by `-0.00425899/-0.02512013`, with AUROC CI95 entirely negative
  `[-0.04218326,-0.00756706]`; p90 Dice changes by `-0.00907089`
  `[-0.01454480,-0.00393037]`. Small AP rises only `+0.00175532` with CI
  crossing zero while small AUROC, saliency mass and p90/p95 Dice fall.
  Medium and large show statistically coherent AP/AUROC or overlap
  regressions. The multiscale arm is rejected.
- Decision: reject both INSIGHT arms as standalone pseudo-mask sources and
  do not select a threshold or train a consumer. The experiment validates
  the benefit of learning local/context features but shows that a
  classification-pooled 32x32 heatmap remains too coarse and poorly
  calibrated for sub-1% tumors. The next admissible mechanism is the already
  reviewed WeCLIP-style dynamic affinity/refinement direction: retain a
  frozen radiograph encoder, learn a lightweight spatial decoder from image
  labels, and regularize/refine it using token affinity rather than another
  post-hoc selector. It requires a new predeclared prediction-first probe
  before any pseudo-mask or U-Net consumer.
- Compact audited evidence is stored under
  `artifacts/kaggle/rad_dino_insight_probe_val_v1/`; reconstructible maps and
  the 1.10 MB checkpoint remain in the ignored temporary audit directory.
  Test remains locked.

## 2026-07-26 - RAD-DINO square-frame evaluation defect predeclared for correction

- A source-level audit found that both the dense-MIL and INSIGHT runners pad
  every radiograph to a centered square before RAD-DINO, but save the
  resulting square-frame heatmap directly as 320x320. Their evaluator then
  compares that square-frame map against a mask resized directly from the
  original rectangular radiograph. The prediction and GT therefore use
  different coordinate systems whenever the source image is not square.
- This is material to the frozen validation cohort: only 9/371 images are
  square; 322/371 have minor-to-major padding with aspect ratio below 0.90,
  183/371 are below 0.75, the minimum is 0.35037 and the mean is 0.72415.
  These counts were obtained from source image dimensions, without consulting
  corrected GT metrics.
- Protocol `rad_dino_square_geometry_correction_val_v1` was frozen before
  corrected evaluation at SHA-256
  `9fb897734fe416b56f3757d8d0973edcbe53e163e4a8f60d37cd2a86c7045977`.
  The only admissible transformation is the exact inverse geometry already
  used by the nominal-memory runner: crop the original-image content box from
  each frozen square map, then bilinear-resize it to 320x320. No foreground
  mask, normalization, threshold, learned parameter or fitted parameter is
  added.
- The correction applies identically to all 371 frozen maps in both arms of
  both affected runs. Every original run/checkpoint/freeze/manifest/map hash
  must pass before derivation; all corrected maps and their manifests must be
  hash-frozen before validation GT is opened. Corrected-minus-original and
  corrected multiscale-minus-single comparisons use paired complete-group
  bootstrap 10,000 with the existing metric contract.
- Correction source commit is
  `bc3d48fa21c8027721f05635532b279b88fdca3f`; canonical correction source
  SHA-256 is
  `ee84e2c188792c97c29d8014594bf138b42e27a6bd8af79d7254912aff5347b9`.
  The protocol-bound repository commit is
  `e9a2201168af7ac45775e7749f3fec92bb7e9872`. The Kaggle wrapper compiles
  and hashes to
  `0c73f33a2598335a431c9fb272c141e1ffa858322c427de12434298e8bc81ab4`;
  prelaunch evidence is
  `artifacts/research_protocols/rad_dino_square_geometry_correction_val_v1_wrapper_audit.json`.
- The previous absolute dense-MIL/INSIGHT localization conclusions are now
  provisional until this deterministic correction is audited. No threshold,
  pseudo-mask, consumer or test access is authorized.

## 2026-07-26 - Geometry-correction wrapper v1 duplicate-source error

- Kernel
  `itsthang333/btxrd-rad-dino-square-geometry-correction-v1` version 1
  stopped after source checkout and before any map derivation or GT access.
  Direct log shows that source discovery found two dense-MIL manifests: the
  real direct kernel output and an identical compact evidence copy embedded
  under the INSIGHT source checkout at `thesis_source/artifacts/`.
- This is an implementation-only source-discovery ambiguity. The admissible
  repair excludes every candidate path containing `thesis_source`, so only
  direct immutable Kaggle kernel-output mounts can satisfy the already pinned
  run-manifest/freeze/checkpoint hashes. Geometry, cohorts, metrics, bootstrap,
  prediction freeze and test lock are unchanged.
- Corrected wrapper SHA-256 is
  `76a18a6bc3c2c2f445a670a526d5f928d029008962d6ba7a1b807695732cfba3`;
  it compiles locally and retains `PRELAUNCH_PASS`.

## 2026-07-27 - Geometry-correction v2 JSON serialization error

- Kernel
  `itsthang333/btxrd-rad-dino-square-geometry-correction-v1` version 2
  terminated after deriving and evaluating the dense-MIL corrected maps.
  The direct nested execution log reports
  `TypeError: Object of type int64 is not JSON serializable` while writing
  the dense-MIL correction `run_manifest.json`. The INSIGHT correction was
  not started, the top-level run index was not written, and no terminal
  scientific result is accepted from this version.
- The failure is confined to the audit summary: `sum(np.isclose(...))`
  returned a NumPy integer for the count of square images. Map reprojection,
  prediction freezing, GT evaluation, cohorts, metrics and bootstrap had
  already executed, but their partial output cannot be promoted without the
  complete fail-closed manifests and independent audit.
- The admissible repair converts every aspect-ratio count and statistic to a
  JSON-native `int` or `float` through a validated helper. A regression test
  requires exact counts and successful `json.dumps`. The corrected source is
  commit `63139fd5168f3114420674b629d1a83caee4fa1e`; canonical source SHA-256 is
  `aad0a03404721e32513b8675226e8586003aca493fe548ffe9ce9bd8837bdc1a`.
  Python compilation passes. The local environment lacks NumPy and Torch, so
  the real regression test remains mandatory in the Kaggle wrapper before
  rerunning the correction.
- The scientific contract is unchanged: exact inverse square geometry only,
  no parameter fit or threshold selection, prediction freeze before GT,
  complete misses, paired complete-group bootstrap 10,000,
  `consumer_trained=false` and `test_evaluated=false`. Only the correction
  source/protocol binding and wrapper preflight may change for version 3.
- The implementation-only protocol amendment is now frozen at canonical
  SHA-256
  `391c02cead32cb5708dfcac484478e37fbc499912ff993c0b41ab9885686e109`
  and bound to repository commit
  `07fc153a2924e998f4e9cbbd2fca7cf22f8fcf12`. The version-3 wrapper requires
  the focused Kaggle regression test before either correction arm, retains
  direct-kernel-output-only discovery and exact original hashes, compiles,
  passes the static source/protocol/order audit, and has canonical-LF
  SHA-256
  `a878d675b659951509c92be78123392a9b029722b2ca05e12d7cacb34a7825dd`.
- The corrected repository state was pushed through commit
  `9df71e139b7e43c539c8e45a876fb739eb1e5eac`. Kernel version 3 was pushed
  to the same private Kaggle kernel and entered `RUNNING`; no competing
  heavy run was launched. The existing single five-minute monitor
  `theo-d-i-rad-dino-geometry-correction` was updated in place with the
  version-3 wrapper/protocol/source hashes and the post-terminal literature
  review requested by the user; no duplicate monitor was created.

## 2026-07-27 - RAD-DINO square-geometry correction completed and audited

- Kaggle kernel
  `itsthang333/btxrd-rad-dino-square-geometry-correction-v1` version 3
  completed. The independent physical auditor returned `PASS` after
  re-hashing all 1,484 corrected maps plus every compact artifact and exactly
  reproducing all six paired complete-group bootstrap comparisons with
  10,000 replicates. Run-index SHA-256 is
  `2b6e3f8cab6e568a51d126c5a0e2c95e0ec9364ca276a5d0466fc3ad3e761faa`;
  execution-log SHA-256 is
  `f079ef678a0632c144fbc46febc0f155406304f5888f083e6e1ba87df475d3bd`;
  local-auditor SHA-256 is
  `1ce11fbfd68a8be24245352bd7c153623bfb280e8f411fecfd1b8ce4078bb473`.
- Provenance and execution order are intact: wrapper
  `a878d675...a7825dd`, protocol `391c02ce...6e109`,
  protocol-bound source commit `07fc153a...fcf12`, correction source
  `aad0a034...bdc1a`, split `85511ee1...193c8c`, dense checkpoint
  `945cff32...01d3e` and INSIGHT checkpoint `35bc926b...50286` all match.
  The focused JSON-serialization regression suite passed 3/3 before either
  correction command. Source discovery used only direct immutable Kaggle
  kernel-output mounts; every original run/freeze/manifest/map hash was
  checked before derivation. Every derived arm contains exactly 371 unique
  maps frozen before GT; cohorts are `371/184/187`, subgroups `94/72/18`,
  complete misses are included, `consumer_trained=false` and
  `test_evaluated=false`.
- The geometry defect was quantitatively material but did not reverse the
  dense-MIL decision. Corrected dense-MIL single-scale pixel AP/AUROC is
  `0.01923622/0.37916193` overall and
  `0.00251040/0.38752858` on small lesions, versus original
  `0.01722582/0.32162069` and `0.00161379/0.32844999`.
  Corrected-minus-original AUROC is `+0.05754124`
  (CI95 `[+0.02934631,+0.08627798]`) overall and `+0.05907859`
  (`[+0.01119626,+0.10588987]`) on small. However fixed-p90 Dice remains
  only `0.00762012` overall and `0.00166766` small, with small argmax hit
  still `0/94`.
- Corrected INSIGHT single-scale also improves as a ranking map: pixel
  AP/AUROC is `0.03498934/0.59483216` overall and
  `0.00958331/0.63345801` small, versus original
  `0.03207112/0.53918555` and `0.00553422/0.54775549`.
  Corrected-minus-original AUROC is `+0.05564661`
  (CI95 `[+0.02452230,+0.08937409]`) overall and `+0.08570252`
  (`[+0.03247362,+0.14159730]`) small. Shape localization remains weak:
  fixed-p90 Dice is `0.02864239` overall,
  `0.00759779/0.03779363/0.10193701` for small/medium/large, and small
  argmax hit remains `0/94`.
- The corrected comparison strengthens the rejection of fixed multiscale
  INSIGHT. Multiscale minus single-scale pixel AP is `-0.00841763`
  (CI95 `[-0.01251007,-0.00491510]`) overall and `-0.00554783`
  (`[-0.01084837,-0.00141125]`) small; pixel AUROC is `-0.03764003`
  (`[-0.05306757,-0.02166112]`) overall and `-0.03796247`
  (`[-0.06384525,-0.01095316]`) small; fixed-p90 Dice is
  `-0.01134452` (`[-0.01657536,-0.00637741]`) overall. Corrected dense-MIL
  multiscale is essentially tied on AP/AUROC but has lower p90 Dice
  `-0.00332222` (CI95 `[-0.00686331,-0.00031792]`); it is not promoted.
- Even after correction, INSIGHT single-scale remains below the already
  geometry-correct nominal-memory single-scale signal. Overall
  AP/AUROC/p90 Dice is `0.03498934/0.59483216/0.02864239` versus
  `0.09333895/0.76755187/0.08465377`; small is
  `0.00958331/0.63345801/0.00759779` versus
  `0.01796705/0.78714348/0.01298239`, with small argmax hit
  `0/94` versus `2/94`.
- Decision: corrected values supersede the prior absolute dense-MIL and
  INSIGHT evaluations, but both heads and their multiscale arms remain
  rejected as standalone pseudo-mask sources. Geometry repair recovers
  ranking quality, not usable tumor shape. No threshold, pseudo mask or
  consumer is promoted. Compact evidence, wrapper provenance and the
  independent audit are retained under
  `artifacts/kaggle/rad_dino_square_geometry_correction_val_v1/`; the
  304.1 MB reconstructible map payload remains only in the ignored temporary
  audit directory. Test remains locked.
- Per the user's explicit follow-up, the next action before launching the
  prepared WeCLIP-inspired decoder is a primary-paper/survey review of
  image-label-only WSSS versus fully supervised segmentation, followed by a
  dataset-specific feasibility analysis of the current `0.10` subgroup gap
  and a predeclared revision of subgroup and overall Dice goals.

## 2026-07-27 - Literature-calibrated WSSS validation goal revision

- The requested post-experiment review is frozen in
  `artifacts/literature_reviews/wsss_supervision_gap_goal_review_2026-07-27.md`
  (canonical-LF SHA-256
  `e9c34f65ddb638affbb24786427c5c7d9a9c2b9e245ac942eea9964e062197b7`).
  It covers the 2025 ACM image-label WSSS survey, two medical limited/non-full
  supervision reviews, WeCLIP, the WACV small-object analysis, GLAM
  mammography, anomaly-guided retinal OCT WSSS, INSIGHT, image-label brain
  tumor segmentation and the BTXRD dataset paper. Evidence using boxes,
  points, scribbles, task-specific training masks or test-time oracle
  filtering was excluded from target setting.
- A `0.10` weak-versus-full Dice gap is feasible in principle but is not a
  defensible hard minimum for every BTXRD subgroup. The closest
  modality/problem comparison, GLAM, reports image-label-only versus fully
  supervised Dice gaps of `0.114` malignant and `0.077` benign, but it uses
  more than one million mammograms. A matched medical OCT pseudo-label
  consumer remains `0.1778` mIoU below its fully supervised upper bound
  (`0.5387` versus `0.7165`); its metric and modality are not converted into
  BTXRD Dice. WeCLIP shows a `+0.062` mIoU contribution from dynamic
  refinement on VOC, but uses natural RGB images, language supervision and a
  CLIP encoder pretrained on 400 million image-text pairs.
- BTXRD-specific evidence makes the old uniform gap especially brittle.
  Reaching the previous `FS - 0.10` tier from the current consumer requires
  gains of `+0.16511/+0.15795/+0.14904/+0.26679` for
  overall/small/medium/large. The large subgroup contains only 18 images and
  its four fully supervised sensitivity runs span `0.60722--0.74827`, a
  `0.14105` range. The corrected dense-MIL/INSIGHT probes recover ranking
  quality but still do not provide usable tumor shape, and the small-object
  literature independently confirms that aggregate WSSS metrics can hide
  small-instance failure.
- Protocol `wsss_feasible_validation_goal_v1` is therefore frozen before the
  next architecture probe at canonical-LF SHA-256
  `d180dfa25d01515ca7d36902a5e495754ed85ff6b8e3c9bebace99aabc6d5ccf`.
  The revised operational minimum Dice is:
  `overall >= 0.34024039`, `small >= 0.17895493`,
  `medium >= 0.51244178`, and `large >= 0.49370336`.
  Small/medium use a `0.15` gap from the frozen fully supervised anchor;
  large uses `0.20` because of its sample size and observed reference
  instability; overall is the exact `94/72/18` subgroup-count-weighted floor.
  All four gates must pass.
- The operational tier remains materially above the current image-label-only
  consumer: required absolute gains are
  `+0.11022052/+0.10795352/+0.09903874/+0.16678648` for
  overall/small/medium/large. The former `FS - 0.10` values
  `0.39513170/0.22895493/0.56244178/0.59370336` are retained unchanged as a
  stretch tier, not discarded.
- No scientific evaluation contract changes: training remains image-label
  only, validation GT is available only after candidate prediction freeze,
  complete misses remain in mean per-image tumor Dice, subgroup GT is never
  a model input, all 371 validation images and `94/72/18` positive subgroups
  remain fixed, and test remains locked. Meeting the operational tier will
  not be reported as clinical equivalence to fully supervised training.

## 2026-07-27 - RAD-DINO affinity-decoder prediction probe predeclaration

- The two user-prepared untracked files
  `project/models/rad_dino_affinity_decoder.py` and
  `tests/test_rad_dino_affinity_decoder.py` were audited before use and
  preserved as the basis of the next mechanism probe. The audit found a
  material implementation bug in the original absolute
  `foreground_threshold=0.99`: bilinear teacher resize from `32x32` to
  `64x64` reduced an isolated `1.0` seed to a maximum of `0.5625`, leaving
  zero positive pseudo pixels. In addition, `58/184` frozen nominal
  validation maps have raw maximum below `0.99`, so an absolute threshold
  was not robust to empirical calibration.
- The implementation now mines deterministic top/bottom confidence ranks at
  the native teacher resolution (`top 1%` foreground, `bottom 50%`
  background), excludes known square padding, guarantees disjoint ranks, and
  resizes foreground seeds with nearest-neighbor preservation. A regression
  test confirms that one isolated `4x4` seed produces four foreground
  gradient pixels at `8x8`, rather than disappearing. Explicit memmap closure
  and exact prediction-manifest/map-set checks were added for Kaggle
  reproducibility.
- Lightweight local validation passed: all three Python files compile;
  decoder, image BCE, pseudo BCE and affinity BCE jointly backpropagate with
  finite gradients; propagation never reduces a source seed; padding is
  excluded from confidence ranks; the mechanistic gate passes exactly at
  every frozen boundary and fails immediately below one boundary. Static AST
  order checks confirm all validation maps are generated before the
  validation GT evaluator is called, the GT dataset import occurs only
  inside that post-freeze evaluator, and no test split occurs in the runner.
  Local pytest collected the suite but skipped it because the default local
  environment lacks Torch; the equivalent Torch checks passed in the
  existing `btxrd-pseudomask` environment. No heavy compute ran locally.
- Source was frozen in commit
  `38b5bb4b9d7a846862443b442ff406f0ab41d3bd`. Protocol
  `rad_dino_affinity_decoder_probe_val_v1` is predeclared before execution
  at SHA-256
  `260f47fbd733f173f1d003ad46c0c5c245bd6bd8612a0549444c0a6a2ff62ee2`.
  Training uses only the 2,981 clean-train images, binary image labels,
  frozen RAD-DINO tokens, deterministic image guidance, and a
  clean-train-normal nominal-memory teacher. The fixed configuration is 12
  epochs, batch 8, AdamW `3e-4`, image SmoothMax BCE plus `1.0` pseudo loss
  and `0.1` local-affinity loss; final epoch only, with no validation
  selection.
- This is prediction-first and cannot itself meet the final segmentation
  consumer goal. Its all-required gate is frozen at image AUROC `>=0.65`;
  overall/small pixel AUROC `>=0.75/0.77`; overall p90 Dice `>=0.10`;
  small p97 Dice `>=0.03`; medium/large p90 Dice `>=0.12/0.35`. Passing only
  authorizes a separately predeclared pseudo-mask or consumer experiment;
  failing rejects this fixed configuration without validation threshold
  fitting. Evaluation includes all 184 tumor images, fixed `94/72/18`
  subgroups, paired complete-group bootstrap `10,000` against frozen nominal
  single-scale evidence, `consumer_trained=false`, and
  `test_evaluated=false`.
- Prelaunch wrapper audit
  `rad_dino_affinity_decoder_probe_val_v1_wrapper_audit.json` passed and is
  committed. The private Tesla-T4 Kaggle kernel
  `itsthang333/btxrd-rad-dino-affinity-decoder-probe-v1` version 1 was
  launched and entered `RUNNING`. Wrapper SHA-256 is
  `3756a0b42a5624b4861b8c64b4f5dcb42fddb65b317ef94d0217498c74eee211`;
  it checks out protocol commit
  `2255361986304d047aa1438ee81b6af5dc9fa044`, separately binds scientific
  source commit `38b5bb4b9d7a846862443b442ff406f0ab41d3bd`, verifies all source/model/
  split/baseline hashes, runs the Torch regression suite before the heavy
  stage, and keeps the approximately 4.7 GB feature cache under
  `/kaggle/temp`. A single five-minute heartbeat monitor
  `theo-d-i-rad-dino-affinity-decoder` follows the kernel; no duplicate
  monitor or competing local heavy job was created.

## 2026-07-27 - Affinity-decoder probe v1 source-binding error

- Kaggle kernel version 1 terminated approximately three seconds after
  checkout, before package installation, Torch tests, dataset discovery,
  model download, feature extraction, training, validation prediction, GT
  access or test access. The direct Kaggle traceback is
  `RuntimeError: Source hash mismatch: datasets/btxrd.py`.
- Root cause is limited to provenance serialization: the prelaunch audit
  recorded Windows CRLF working-tree SHA-256 values for
  `project/datasets/btxrd.py` and `project/datasets/common.py`, while Kaggle
  correctly checked out the canonical LF Git blobs. The canonical blob
  hashes are respectively
  `d8f0804be4e81cdb4d58e4673708c1067eb7d9b49b42bb78cb6051188c156001`
  and
  `1927eb358a9db1a0e9c2571be5e222c3edd9d69814dfb4bc3375bd3f8593b98a`.
- The admissible correction changes only those two source-hash bindings in
  the protocol/wrapper. The scientific source commit, architecture, teacher,
  losses, schedule, gate, split, baseline, model, prediction geometry and
  evaluation contract are unchanged. Corrected protocol SHA-256 is
  `53e2bb82ef35862b6c3e20387edbe60776f9d1ba46da516b9d5116db3fa2e7cf`.
  `consumer_trained=false`, validation GT was not read, and test remains
  locked. Version 2 may rerun only after the corrected wrapper is re-audited.
- Corrected wrapper re-audit passed against all nine canonical Git blobs.
  Wrapper SHA-256 is
  `1892c20f421cb4f183654a3d0e27f700f2b3192794793aba73a510f087722392`;
  protocol checkout commit is
  `a716d059648924b5bb7ccf76f41549d4715ec89c`. Kernel version 2 was pushed
  to the same private kernel and entered `RUNNING`. The existing
  `theo-d-i-rad-dino-affinity-decoder` five-minute monitor remains the only
  monitor; no duplicate was created.

## 2026-07-27 - Affinity-decoder probe v2 baseline-binding error

- Kernel version 2 passed the corrected source hash checks, then terminated
  before package installation, data/model access, training or validation GT
  with `RuntimeError: Frozen nominal baseline hash mismatch`.
- This is the same line-ending provenance class, now isolated to the frozen
  baseline CSV: its canonical LF Git blob is SHA-256
  `f685e85b22ff5e3e48ecdf659d8f1c0f9f60cf13e9ffa69783305d4819aff8c9`
  (`31,987` bytes), while the already frozen/evaluated local CRLF artifact is
  SHA-256
  `c7bd20412913157b8d6f17b69ce4ed01495645a1c2a91b17a0a37166737f844c`
  (`32,172` bytes). Parsed CSV values are unchanged.
- The wrapper now verifies the canonical Git blob, deterministically creates
  the CRLF frozen baseline in `/kaggle/temp`, verifies the original frozen
  baseline hash, and passes only that copy to the runner. This mirrors the
  already audited split-manifest conversion. No protocol, source,
  architecture, parameter, metric, gate or cohort changes. Wrapper v3
  SHA-256 is
  `e8a90a1a1a87affba9ff10fcd8ec6bf66de086485bd3bed3505073ec26724a51`.
  Validation GT and test were not read; consumer training did not begin.
- The corrected wrapper re-audit passed and kernel version 3 was pushed to
  the same private kernel; it entered `RUNNING`. The existing single
  five-minute monitor continues unchanged.
- While version 3 runs, an independent post-run auditor was prepared at
  `project/tools/audit_rad_dino_affinity_decoder_probe.py` (SHA-256
  `14331ca1128566fa0e0ef08abe11ef0c995ad41b449e0e27744c7b446874f64c`).
  It independently verifies wrapper/runner ordering, protocol/source/split/
  model/baseline bindings, checkpoint and 12-epoch history, teacher
  metadata, the complete set and hashes of 371 float16 maps, then and only
  then opens validation GT to recompute every per-image metric and the
  `94/72/18` subgroups. It also reproduces the complete-group bootstrap
  `10,000` and mechanistic gate. Compile, static-order, deterministic
  bootstrap and metric smoke checks pass locally; it does not access test.
- A conditional post-gate consumer design is recorded at
  `artifacts/research_plans/post_affinity_consumer_design_v1.md` (SHA-256
  `a9237d4c117d4593724396487cd814724d292e05f6b31cef8d51bb44347acc8e`).
  This is explicitly a draft, not an executable protocol. It avoids repeating
  the measured Gate-D failure in which the U-Net fit hard pseudo targets to
  train Dice `0.73817` but generalized to only `0.23002`. If the affinity gate
  passes, the primary next hypothesis is a fixed-horizon ResNet18-U-Net
  consumer using frozen continuous train maps, confident partial soft BCE,
  image-level SmoothMax BCE, affinity/edge consistency and EMA view
  consistency; ambiguous pixels stay unlabeled instead of becoming false
  background. All train maps must be frozen before consumer training and all
  validation predictions before the single spatial-GT evaluation. If the gate
  fails, the design is not authorized and no decoder pseudo-mask consumer may
  run.

## 2026-07-27 - RAD-DINO affinity-decoder probe v3 terminal audit and rejection

- Kaggle kernel
  `itsthang333/btxrd-rad-dino-affinity-decoder-probe-v1` version 3 completed.
  The first local CLI download was interrupted after 117 maps, so it was not
  used for acceptance. A fresh download of the immutable version-3 output
  produced all 371 float16 maps plus checkpoint, teacher metadata, training
  history, prediction/run manifests, freeze record and wrapper provenance.
  The downloader emitted a Windows console `charmap` message only after those
  artifacts were present; no scientific output was regenerated or changed.
- The independent auditor ran on that fresh direct download and passed at
  runtime SHA-256
  `7312982bf8283cead0fdf961937048e505ad7e8f84fb802b8a6f01418677aa4d`.
  It verified wrapper SHA-256
  `e8a90a1a1a87affba9ff10fcd8ec6bf66de086485bd3bed3505073ec26724a51`,
  checkout/scientific commits
  `a716d059648924b5bb7ccf76f41549d4715ec89c` /
  `38b5bb4b9d7a846862443b442ff406f0ab41d3bd`, protocol, split,
  RAD-DINO and frozen-baseline hashes, the 12-epoch final checkpoint
  (`c5f9278de813396628fffe8360f09c786f1b74750861c23192d568405535b0d3`),
  371 maps (`76,028,288` bytes), and cohort
  `371/184/187` with `94/72/18` small/medium/large positives. Validation GT
  was opened only after prediction freeze; complete misses are included,
  paired complete-group bootstrap used 10,000 replicates, and
  `consumer_trained=false`, `test_evaluated=false`.
- Threshold-free localization improved over the frozen nominal-memory map.
  Decoder-minus-nominal pixel AUROC was `+0.03743012` overall
  (95% CI `[0.01795775, 0.05664618]`), `+0.05433193` small
  (`[0.02462228, 0.08363946]`) and `+0.03054294` medium
  (`[0.00389523, 0.05671270]`); large was `-0.02328623` with a CI crossing
  zero. Pixel AP improved `+0.03164879` overall and `+0.04910153` small with
  positive CIs. Saliency mass inside GT also improved with positive CIs in
  every subgroup. This supports the spatial-decoder/affinity mechanism as a
  useful localization signal, but not this fixed configuration as a
  pseudo-label source.
- The predeclared all-required gate failed exactly two of seven checks.
  Image AUROC `0.81725180`, overall/small pixel AUROC
  `0.80498199/0.84147541`, small Dice@p97 `0.03445384` and large Dice@p90
  `0.38334541` passed. Overall Dice@p90 was `0.09112394 < 0.10`, and medium
  Dice@p90 was `0.11561626 < 0.12`; therefore gate status is `FAIL`.
  Thresholds are not relaxed after seeing validation GT. Per the frozen
  contract, this exact configuration is rejected and the conditional
  soft-label consumer is not authorized.
- The measured bottleneck is now sharper: ranking/localization is credible,
  especially for small lesions, but the decoder's fixed-percentile support
  does not recover sufficient variable lesion extent and boundary shape.
  The next admissible hypothesis must add a prediction-time, GT-independent
  variable-area shape mechanism rather than merely fit another consumer to
  the rejected maps or tune percentiles on validation GT.
- Compact evidence is stored under
  `artifacts/kaggle/rad_dino_affinity_decoder_probe_val_v1/`. The dense maps
  and checkpoint remain only in the isolated audit download. The
  `compact_evidence_manifest.json` records direct-runtime hashes and the
  canonical LF hashes for the three text files normalized for Git, so the
  line-ending transformation is explicit rather than hidden.

## 2026-07-27 - Affinity-guided raw single-proposal selector predeclaration

- The rejected affinity decoder is not promoted as a mask source. Its
  statistically credible ranking signal is instead assigned one narrower,
  prediction-time role: select exactly one raw SAM proposal from the already
  frozen LayerCAM plus BiomedCLIP proposal gallery. This targets the observed
  selection bottleneck without fitting another consumer to rejected masks.
  The existing gallery's best-single oracle Dice is
  `0.48372617/0.25976551/0.71084146/0.74483737` for
  overall/small/medium/large, above the operational consumer goals
  `0.34024039/0.17895493/0.51244178/0.49370336` in every subgroup.
- Source commit
  `56c01f241bda4b80183918517999f7ddbb37fc55` adds
  `affinity_rank_single`: six fixed top-rank supports
  `20/15/10/5/3/1%`, followed by a deterministic lexicographic selector over
  support overlap, captured affinity mass, inside/outside contrast, frozen
  SAM rank, candidate area and stable index. It retains exactly one
  variable-area raw proposal. Candidate union, support clipping, fitted
  weights, morphology and every other boundary-changing operation are
  disabled. The LayerCAM, BiomedCLIP and SAM proposal-generation contract is
  unchanged and will fail closed unless every per-image component and
  candidate count matches the frozen same-gallery baseline.
- The sanitized private Kaggle dataset
  `itsthang333/btxrd-rad-dino-affinity-selector-input-v1`, version 1, contains
  exactly 374 files: 371 frozen float16 `320x320` affinity maps plus the
  direct prediction manifest, prediction freeze and sanitized package
  manifest. Their SHA-256 values are respectively
  `c066744a2acf3df3a078f4b25a973ea21bf140adeb88f713c90de8886a53fc42`,
  `ed8b323dfaddf8fc9b5f7061a49dbc74f67187a45fb55c73ebdb46f36f9ff4ad`
  and
  `f9fb008a19c9f0ed1cb5ffb3f04f4f35227562614ca1dd6655fdaff6373167e9`.
  A local package audit reverified all 371 manifest-indexed map hashes,
  schemas, finite `[0,1]` range, cohort `371/184/187` and total map bytes
  `76,028,288`. The owner-authenticated Kaggle page independently displays
  version 1, private visibility, 374 files and the same three top-level
  artifacts. Kaggle CLI `status` reports `ready`, but its current OAuth
  context returns HTTP 403 for private file listing/download; this is treated
  as a tooling-auth read error, not scientific evidence. The kernel wrapper
  therefore audits the direct Kaggle dataset mount byte-for-byte before any
  prediction and fails closed if access or any artifact differs.
- Protocol `affinity_guided_proposal_selector_val_v1` is predeclared before
  any affinity-guided selector prediction. Its scientific baseline is the
  frozen same-gallery `coverage_mass_sam` run, with final per-image,
  prompt-quality and pseudo-manifest hashes
  `59bead9162ff90851087c67ecac7f1bc8d9133e7c6a8aebb2f3db6e6606d7b05`,
  `200dbc4172dcd7e5bd7c2c0a23734725925e3153d73b52a2796c4f3fcda5ab9a`
  and
  `6962bb4b92a0311f2f8d0f5e68e14a4aac0e4199994868c6eb0d5626f08790ba`.
  All 371 masks and all 184 complete candidate galleries must be frozen
  before validation GT is imported. Evaluation uses the unchanged
  `94/72/18` subgroups, includes complete misses and uses 10,000 paired
  complete-group bootstrap replicates.
- A future consumer protocol is authorized only if all three frozen gates
  pass: the regenerated gallery oracle remains above every operational
  consumer target; overall final-Dice improvement over the same-gallery
  baseline has paired CI95 lower bound above zero with no subgroup mean
  decrease; and pseudo-mask Dice reaches the halfway-to-goal floors
  `0.28879224/0.14660840/0.43761543/0.43601515`. Passing does not train a
  consumer automatically. Failing does not permit post-hoc adjustment of
  rank supports, thresholds, weights, routing or morphology.
- Protocol JSON, wrapper compile, full sanitized-input audit and the four
  focused selector/input/gallery/diagnostic test modules pass locally in the
  existing `btxrd-pseudomask` environment (16 tests). Heavy computation has
  not run locally, validation GT has not been read for the new selector,
  `consumer_trained=false`, and `test_evaluated=false`.
- Protocol was frozen in commit
  `ca9462f13588243c0e490c2f18564038e49fd857` at canonical Git-blob
  SHA-256
  `07bd490d309a850daeba1d00590d36968360aa7a70bebe3d418feb7c44ffadf7`.
  The prelaunch wrapper audit
  `affinity_guided_proposal_selector_val_v1_wrapper_audit.json` passes.
  Wrapper SHA-256 is
  `526bf6b32cdbee15a872290ffaa85e0b92950559c0aac63e97fafe2addaf7e47`;
  it binds all 12 canonical source blobs, the direct private dataset mount,
  same-gallery baseline evidence, both frozen proposal sources, checkpoints
  and split. Static order verification places generation before manifest
  freeze, both validation-GT evaluators after freeze, and even the
  GT-derived baseline metric artifacts after freeze. Kernel metadata
  SHA-256 is
  `183dc06f9cbb853531ae7a6f9786b0b8637e2bf8e10d7d02fd3bc144c51a5f5c`.
- Private Kaggle T4 kernel
  `itsthang333/btxrd-affinity-guided-proposal-selector-val-v1` version 1 was
  launched from that audited wrapper and entered `RUNNING`. Kaggle rejected
  only the optional hyphenated tags; the kernel itself was accepted without
  metadata or scientific-contract substitution. Exactly one five-minute
  heartbeat, `theo-d-i-affinity-selector`, follows this kernel. No competing
  local heavy job and no duplicate monitor were created.
- Before version 1 reached terminal acceptance, a provenance-only output
  retention issue was found: compact cleanup would delete the selected final
  masks after cloud evaluation, preventing a local independent auditor from
  recomputing Dice directly from masks. Version 2 therefore preserves only
  `pseudo_masks/masks`; large candidate-diagnostic arrays are still removed.
  No selector, proposal, prediction, metric, gate, source, data or protocol
  setting changed. Re-audited v2 wrapper SHA-256 is
  `7a736598a9dd9791dc3271c9176c64ab3cf7f5da59b7c399ae39c4c979ad1190`.
  The existing single heartbeat remains attached to the same kernel slug.
- While v2 runs, an independent post-freeze auditor was added at
  `project/tools/audit_affinity_guided_proposal_selector.py` (SHA-256
  `26c0b22fdf91d4ce1d115bb899f0fa5380518e492b44e801aba86ad302e064a8`).
  It first audits the runtime wrapper/order, protocol, all 371 source affinity
  maps, physical final-mask hashes and both frozen prediction manifests
  without GT. Only after that complete freeze passes does it instantiate the
  validation segmentation dataset, recompute Dice directly from all 371
  retained masks, verify `371/184/187` and `94/72/18`, compare the regenerated
  proposal gallery per image with the frozen same-gallery baseline, reproduce
  the 10,000-replicate complete-group bootstrap and independently recompute
  all three promotion gates. Its five focused unit tests plus the 16 selector
  regression tests pass locally. It contains no test-dataset access.

## 2026-07-27 - Affinity-guided selector v2 post-generation audit error

- Kaggle kernel version 2 failed after `583.1` seconds. Direct Kaggle
  traceback is
  `RuntimeError: LayerCAM component gallery drift: IMG000001.jpeg` at wrapper
  line 594. Before the error, all runtime source/input hashes and 16 focused
  tests passed; generation completed all 371 pseudo-mask rows and all 184
  tumor candidate galleries with manifest SHA-256 values
  `8799c2e712889f8a396f61dc854fb863a4bcefedbf6dd60a9fcfea604ebcd1b7`
  and
  `35cc3d0275956a4901244ef5815e229ac197afd0ca8a403b42cc64a22f87a99e`.
  The exception occurred in the wrapper's no-GT post-generation audit before
  `prediction_freeze.json`, validation evaluation, consumer training or any
  test access.
- Root cause is an implementation-only invariant mismatch introduced when
  the comparison baseline was strengthened from the original LayerCAM run to
  the same LayerCAM-plus-BiomedCLIP proposal gallery. The inherited check
  compared candidate `cam_morphology_components` with baseline total
  `morphology_components`; for `IMG000001.jpeg` these represent different
  quantities. The same-gallery manifest exposes the correct
  `cam_morphology_components` field, while separate frozen checks already
  compare external-component, total-component and SAM-candidate counts.
  Compact direct evidence is stored in
  `artifacts/kaggle/affinity_guided_proposal_selector_val_v1/version2_error.json`.
- Version 3 changes only that field-to-field audit comparison. Proposal
  generation, affinity selector, masks, source checkout, protocol, metric,
  gate, cohort and GT/test ordering are unchanged. Compile, JSON, unit and
  same-gallery schema/accounting re-audits pass. Wrapper v3 SHA-256 is
  `022369ae92c7818da0e6443fac97b7f2b832d335ee8ad8d5dd84611ed6987192`.
  The independent auditor is rebound only to that wrapper hash and now has
  SHA-256
  `35bb8313c72a881d0ce3cb07cf52af3b0463976edd7cf0bed96f5d462f66a612`.
  The existing single heartbeat remains the only monitor.
- Corrected version 3 was pushed to the same private kernel and entered
  `RUNNING`. No new monitor or local heavy job was created.

## 2026-07-27 - Affinity-guided selector v3 terminal audit and rejection

- Kaggle kernel
  `itsthang333/btxrd-affinity-guided-proposal-selector-val-v1` version 3
  completed successfully in `713.63` wrapper seconds (`720.5` seconds in the
  Kaggle UI). The first two bulk downloads hit transient
  `kaggleusercontent.com` connection timeouts after partial progress. No
  partial output was accepted: the immutable version-3 output was downloaded
  in bounded filename-prefix chunks and reconciled against the frozen split
  before audit. It contains exactly 371 final PNG masks with no missing or
  extra validation ID.
- The independent local verifier passed. Before opening validation GT it
  verified wrapper SHA-256
  `022369ae92c7818da0e6443fac97b7f2b832d335ee8ad8d5dd84611ed6987192`,
  protocol SHA-256
  `07bd490d309a850daeba1d00590d36968360aa7a70bebe3d418feb7c44ffadf7`,
  checkout/implementation commits
  `ca9462f13588243c0e490c2f18564038e49fd857` /
  `56c01f241bda4b80183918517999f7ddbb37fc55`, the frozen split,
  all 371 affinity-map hashes, both prediction freezes, both prediction
  manifests and all 371 physical final-mask hashes. The same-gallery oracle
  and candidate accounting are invariant with maximum absolute metric delta
  `0.0`. Only then did it recompute Dice from validation masks. Cohort is
  `371/184/187`, positive subgroups are `94/72/18`, complete misses are
  included, paired bootstrap uses 10,000 complete-group resamples with seed
  42, `consumer_trained=false`, and `test_evaluated=false`.
- Cloud and independent local results match. Affinity-rank-single final Dice
  is `0.09742941/0.02661902/0.11328910/0.40377819` for
  overall/small/medium/large, versus the same-gallery baseline
  `0.23734410/0.11426187/0.36278908/0.37832694`. Paired deltas and CI95 are
  `-0.13991469 [-0.19146040,-0.08744420]` overall,
  `-0.08764285 [-0.13623080,-0.04357934]` small,
  `-0.24949998 [-0.33670396,-0.15904919]` medium, and
  `+0.02545126 [-0.22047091,+0.28599635]` large.
- The regenerated best-single oracle remains high at
  `0.48372617/0.25976551/0.71084146/0.74483737`, so proposal availability
  passes its gate. Statistical improvement and halfway-to-operational-quality
  both fail, making the all-required consumer-authorization gate fail. No
  pseudo-mask consumer is launched and the thresholds are not altered after
  seeing validation GT.
- The failure mode is global-support scale, not gallery drift. The selector
  chose the broad p80/top-20-percent affinity support on 122/184 tumors.
  Median predicted-to-GT area ratios are `24.44x` overall, `169.83x` small,
  `10.14x` medium and `1.19x` large; 50/184 tumors are complete misses.
  Thus a diffuse global affinity rank can overlap the correct location yet
  reward anatomy-scale SAM masks. This explains the slight large-lesion point
  gain and severe small/medium collapse. A further scalar reranker, global
  percentile sweep, source-consensus selector or classifier-causal selector
  is not justified: those families have now failed repeatedly under frozen
  validation gates.
- Decision: reject `affinity_rank_single` and retain the RAD-DINO decoder only
  as localization/affinity research evidence. The next admissible hypothesis
  must change representation-to-shape transfer: use high-confidence
  image-label-derived seeds, local RAD-DINO token affinities and explicit
  boundary/region propagation with ambiguous pixels unlabeled, followed by a
  train-only partial-label/consistency decoder if and only if a new
  prediction-first spatial gate passes. It must be separately predeclared;
  current validation metrics cannot tune its thresholds or weights.
- Compact terminal evidence is stored at
  `artifacts/kaggle/affinity_guided_proposal_selector_val_v1/version3_compact_evidence.json`.
  The direct comparison, run, freeze, kernel-log and independent-audit hashes
  are recorded there; the 371 PNG masks remain only in the isolated audit
  download.

## 2026-07-27 - RAD-DINO geodesic seed-expansion probe predeclaration

- The next experiment changes representation-to-shape transfer rather than
  adding another proposal-level scalar selector. The primary UM-CAM paper,
  journal record and official code were re-inspected at
  `https://arxiv.org/abs/2306.11490`,
  `https://doi.org/10.1016/j.patcog.2024.111204`, and official repository
  commit `1389eddb3858fc1c5793329e68f9794245f8c3a4`. The transferable mechanism
  is foreground/background seeding plus exponential geodesic-distance fusion.
  This is not claimed as an exact reproduction: the official script reads GT
  in its loop and searches a CAM threshold, whereas the BTXRD runner contains
  no segmentation-dataset/annotation import and performs no threshold search.
- Source commit `1b5d0cc151530d71b49e1088e20a94a42c25e08a` implements a
  deterministic 64x64 eight-neighbour feature graph. Foreground and
  background seeds are the already-frozen exact top 1% and bottom 50%
  affinity ranks; the remaining 49% stays continuous/ambiguous. Graph costs
  combine grayscale with a seed-42 frozen 768-to-16 projection of spatial
  RAD-DINO tokens. Each branch is normalized by its own within-image median
  local edge magnitude and receives equal energy, so there is no fitted
  scalar weight. Exact multi-source Dijkstra distances feed a ratio-1
  exponential foreground/background cue, which is confidence-blended with
  the source map and reprojected to 320x320. Image-level normal cases remain
  exact zero maps.
- This design directly targets the measured failure. The rejected global
  affinity selector chose top-20-percent support on 122/184 tumors and
  produced median predicted-to-GT area ratios of `169.83x` small and `10.14x`
  medium. In the new graph, p99 seeds stay fixed while expansion is local and
  boundary-sensitive; no whole SAM proposal, union, morphology, true size,
  subgroup routing, validation-tuned percentile or fitted propagation ratio
  is available.
- Protocol `rad_dino_geodesic_seed_expansion_val_v1` is physically
  predeclared before any new geodesic prediction at SHA-256
  `fbf49a452664e52e6efe07059282e91f1819cf2fab66baa0d4a1d3025fcfe1b2`.
  It binds the immutable 371-map affinity input package, RAD-DINO snapshot,
  split, source commit and canonical Git-blob hashes. All maps, manifests and
  generation metadata must be frozen before the evaluator can instantiate
  the validation segmentation dataset. Cohort remains `371/184/187`,
  subgroup counts `94/72/18`, complete misses are included, paired
  complete-group bootstrap uses 10,000 replicates, and test remains locked.
- The all-required gate retains the seven absolute affinity-decoder checks and
  adds shape-transfer evidence: overall p90-Dice improvement must have paired
  CI95 lower bound above zero and mean p90 Dice cannot decrease in any
  subgroup. Passing authorizes only a separate train-map/partial-label
  consumer protocol; failing rejects the exact graph without changing ranks,
  ratio, branch weights, graph cost or thresholds after GT. No consumer is
  trained automatically.
- Module, prediction runner, post-freeze evaluator, continuous projection,
  rank seeding, feature normalization, boundary-respecting shortest path,
  exponential fusion and complete-group bootstrap compile and pass 11 focused
  tests locally. Heavy inference has not run, validation GT has not been read
  for this experiment, `consumer_trained=false`, and `test_evaluated=false`.

## 2026-07-27 - RAD-DINO geodesic seed-expansion Kaggle wrapper freeze

- The private Kaggle wrapper for
  `itsthang333/btxrd-rad-dino-geodesic-seed-expansion-val-v1` is frozen at
  SHA-256
  `97de7ce759525b45b3186eb7f9c5829f5ca4b9597ea81a8c5abc64d3260b09d5`;
  kernel metadata SHA-256 is
  `96787e700be9b5f39e2bd4f30532cb56429896d6a442ab5d3042153fe82b96c4`.
  Static audit evidence is stored at
  `artifacts/research_protocols/rad_dino_geodesic_seed_expansion_val_v1_wrapper_audit.json`.
- The source protocol stores the exact Windows CRLF runtime bytes while the
  canonical Kaggle Git checkout materializes LF. The wrapper first verifies
  the 12 canonical LF hashes from source commit
  `1b5d0cc151530d71b49e1088e20a94a42c25e08a`, deterministically materializes
  CRLF, and then verifies every executed file against the already-predeclared
  protocol hashes. No scientific text changes during this conversion.
- Before prediction the wrapper requires the direct frozen affinity-input
  mount, audits all 371 physical affinity maps, verifies all 371 raw-image
  hashes from the split, reconstructs the frozen CRLF split and baseline
  evidence, pins RAD-DINO revision
  `110cbc18d5133582e320b43d53bf5c44e410c936`, and reruns all 11 focused tests.
- Execution order is statically fixed as prediction, prediction freeze,
  independent verification of all 371 generated maps and source-map bindings,
  then evaluator construction and validation-GT access. Evaluation must retain
  cohort `371/184/187`, subgroups `94/72/18`, complete misses, paired
  complete-group bootstrap with 10,000 replicates and seed family 20260727,
  `consumer_trained=false`, and `test_evaluated=false`.
- The wrapper and metadata compile/audit PASS locally. No heavy inference,
  consumer training or test evaluation occurred during this freeze; the
  kernel had not been launched when this evidence was written.
- After the prelaunch evidence was committed and pushed at `a790067`, private
  Kaggle kernel
  `itsthang333/btxrd-rad-dino-geodesic-seed-expansion-val-v1` version 1 was
  launched successfully and entered `RUNNING`. Exactly one five-minute
  heartbeat is active with ID `theo-d-i-rad-dino-geodesic`; no duplicate
  monitor was created. The monitor is required to remove itself on terminal
  status before output audit or any implementation-only rerun.

## 2026-07-27 - RAD-DINO geodesic seed-expansion terminal audit and rejection

- Private Kaggle kernel
  `itsthang333/btxrd-rad-dino-geodesic-seed-expansion-val-v1` version 1
  completed successfully on a Tesla T4 in `240.8093` seconds. The sole
  five-minute monitor `theo-d-i-rad-dino-geodesic` removed itself at terminal
  status, so no Kaggle monitor remains. The direct run contains 371/371
  output maps. Eleven focused tests passed in-kernel before prediction.
- Direct provenance and outputs were independently frozen and audited:
  protocol SHA-256
  `fbf49a452664e52e6efe07059282e91f1819cf2fab66baa0d4a1d3025fcfe1b2`,
  wrapper
  `97de7ce759525b45b3186eb7f9c5829f5ca4b9597ea81a8c5abc64d3260b09d5`,
  run manifest
  `dbb18d163f79b2b367c2c6b0010be286eafd5da05b1ced6f13db45f5f488824d`,
  prediction freeze
  `3f7c7437347b6587593ef348c307721ef3e02d978405817f04a40874dc432565`,
  prediction manifest
  `0e3f02e6a938c8dbeeb2dab7a7eb22e69b8b126bbd4d161f21523bdccfc7ed4d`,
  summary
  `1c8a9af13225614fa227d6c1affca988f2cf7bbc26778292af87ab4bdee6b942`,
  paired comparison
  `756005b2dd2905324e54e12af2024662200c5494b8a5619729892b4a3bae8de6`
  and gate
  `59bd1efa0ea98755c4b779fb8b50fc7b07d05dfb11555dbf98046c8498f8ffca`.
- A new independent local auditor at
  `project/tools/audit_rad_dino_geodesic_seed_probe.py` (SHA-256
  `d66cdb737a80c592fce335a02b822141c51802fd7e325f8e92188d257e0c9ce8`)
  verified source order and the absence of dataset/annotation/test access in
  generation; rehashed all 371 source and 371 output maps; then recomputed
  every metric, complete miss, complete-group paired bootstrap and gate from
  local validation GT only after the prediction freeze. Its evidence SHA-256
  is
  `13d777d19f85f7a9d538b83030623f013c57513fb98bb5de5e5cbd6edc4da8fd`;
  cohort `371/184/187`, subgroups `94/72/18`, 10,000 bootstrap replicates and
  every cloud value match.
- Geodesic overall/small/medium/large p90 Dice is
  `0.09167708/0.01623404/0.11702251/0.38427568`; pixel AUROC is
  `0.79697640/0.83400862/0.74582948/0.80817356`. Complete misses at p90
  are `40/28/11/1`. Against the frozen affinity source, overall pixel AP
  decreases by `-0.00960391` (CI95
  `[-0.01907460,-0.00256564]`) and pixel AUROC by `-0.00800559`
  (`[-0.01157337,-0.00452495]`). Overall p90 Dice changes only
  `+0.00055315` (`[-0.00020901,+0.00146991]`); small p90 Dice decreases
  `-0.00017251`.
- The no-GT mechanistic comparison explains the failure: across 184 tumor
  maps, median pixel correlation with the source is `0.99437678`, median
  top-10-percent support Jaccard is `0.94861551`, and median absolute map
  change is only `0.00771322`. The graph mostly preserves the same support
  ranking, so it cannot correct the source affinity map's shape error.
- The predeclared all-required gate fails: overall p90 Dice is below `0.10`,
  medium p90 Dice is below `0.12`, the overall improvement CI lower bound is
  not positive, and small mean p90 Dice decreases. Decision:
  `REJECT_FIXED_GEODESIC_CONFIGURATION`. No pseudo-mask consumer is
  authorized, `consumer_trained=false`, and `test_evaluated=false`.
  Compact evidence is stored at
  `artifacts/kaggle/rad_dino_geodesic_seed_expansion_val_v1/version1_compact_evidence.json`.

## 2026-07-27 - Operational goal retained after post-experiment literature check

- After the geodesic experiment completed, the frozen literature review
  `artifacts/literature_reviews/wsss_supervision_gap_goal_review_2026-07-27.md`
  and goal protocol
  `artifacts/research_protocols/wsss_feasible_validation_goal_v1.json` were
  rechecked against additional direct comparisons. Chen and Sun's ACM survey
  identifies image-label-only training as the most challenging WSSS setting
  and emphasizes incomplete CAM extent. In medical image-label-only studies,
  EnsembleCAM reports BraTS Dice `0.703` versus fully supervised `0.818`
  (gap `0.115`) and prostate Decathlon `0.793` versus `0.868` (gap `0.075`);
  morphology-guided CAM+SAM reports BUSI Dice `0.7439` versus fully
  supervised U-Net `0.7831` (gap `0.0392`), but depends on strong
  modality-specific morphology and foundation-model priors.
- Therefore a weak-versus-full gap of `0.10` is achievable in favorable
  settings but is not a defensible hard minimum for every BTXRD subgroup.
  BTXRD has only 3,746 images, small lesions are the weakest current group,
  and large validation has only 18 images with a `0.14105` supervised-run
  range. The already-frozen feasible operational goal remains unchanged:
  overall/small/medium/large Dice at least
  `0.34024039/0.17895493/0.51244178/0.49370336`. These require absolute
  gains of `+0.11022052/+0.10795352/+0.09903874/+0.16678648` over the
  current consumer. The former fully-supervised-minus-0.10 tier remains a
  stretch goal only. Metric, cohort, complete-miss policy and test lock are
  unchanged. The post-experiment quantitative addendum is stored at
  `artifacts/literature_reviews/wsss_supervision_gap_post_experiment_addendum_2026-07-27.md`.

## 2026-07-27 - Multi-layer soft-region RAD-DINO source preparation

- The rejected fixed geodesic configuration is not reused. The next
  prediction-first source changes the trainable representation and supervision
  together: frozen RAD-DINO patch tokens from layers `4/8/12` are projected
  by the seed-42 frozen `768-to-128` map and fused by a lightweight spatial
  decoder. This transfers the intermediate-token mechanism motivated by ToCo
  and the frozen multi-layer decoder mechanism from WeCLIP without importing
  CLIP text supervision or natural-image pseudo masks.
- The previous affinity decoder used exactly the top 1% teacher ranks as
  foreground for every positive image. The new soft-region loss instead uses
  the absolute empirical train-normal calibration: teacher evidence above
  `0.90` receives a linearly increasing foreground weight, evidence below
  `0.50` receives a linearly increasing background weight, and the interval
  remains unlabeled. Foreground support is therefore variable-area and tied
  to normal-tail evidence rather than validation lesion size or a per-image
  rank quota. Normal training images retain exact dense background
  supervision.
- Decoder affinity is filtered by frozen final-layer token cosine and refines
  foreground and background evidence symmetrically for three steps before a
  soft ratio is formed. Training combines image-level alpha-12 SmoothMax BCE,
  soft-region BCE, confident soft-pair affinity BCE and aligned horizontal
  flip consistency with fixed weights `1.0/1.0/0.1/0.2`. Prediction averages
  original and aligned horizontal-flip decoder probabilities. No validation
  epoch, threshold, support fraction or loss weight is selected.
- Focused local checks pass in the Torch research environment: all five new
  files compile; the full loss backpropagates through every decoder parameter;
  absolute calibration produces different foreground counts for different
  maps; bidirectional refinement is bounded; flip alignment is exact; and
  normal-image gradients are dense background gradients. AST/order checks
  confirm that the runner has no segmentation-dataset or test access and the
  evaluator verifies the physical prediction freeze before importing
  validation GT. Heavy training has not run, `consumer_trained=false`, and
  `test_evaluated=false`. The source will be committed before an immutable
  protocol binds its hashes and scientific gate.

## 2026-07-27 - Multi-layer soft-region prediction probe predeclaration

- Source commit `5cb7e81434742f7bf1b796018239c97ca1731d87` is now
  immutably bound by protocol
  `rad_dino_multilayer_soft_region_probe_val_v1` at SHA-256
  `5d0c5f8c96f32aa9e49b5bef96c2aa026ed3559a6dc639dc20929608d3a475c9`.
  The protocol verifies canonical Git hashes for the runner, separate
  post-freeze evaluator, decoder, all imported scientific helpers, dataset
  loader and focused tests. No prediction from this mechanism existed when
  the protocol was frozen.
- The frozen scientific configuration is RAD-DINO layers `4/8/12`, a shared
  seed-42 `768-to-128` projection, 12 final-epoch-only training epochs,
  batch 8 and AdamW `3e-4`. The total loss is image-level SmoothMax BCE plus
  `1.0` soft-region BCE, `0.1` soft affinity-pair BCE and `0.2` aligned flip
  consistency. Absolute train-normal-calibrated evidence thresholds are
  `0.90` foreground start and `0.50` background end; no per-image support
  quota or validation-tuned scalar exists.
- All 371 candidate maps and their manifest must be physically hash-frozen
  before the separate evaluator can import validation GT. The comparison arm
  is the frozen affinity-decoder v3 per-image evidence. The all-required gate
  retains the seven absolute localization checks and additionally requires
  the candidate-minus-affinity overall p90-Dice complete-group CI95 lower
  bound to be positive with no negative mean p90-Dice delta in small, medium
  or large. Passing authorizes only a separately predeclared consumer;
  failure rejects the exact configuration. Complete misses remain included,
  bootstrap uses 10,000 replicates, `consumer_trained=false`, and test remains
  locked.

## 2026-07-27 - Multi-layer soft-region Kaggle wrapper freeze

- The protocol was committed and pushed at
  `149c32d9cd747e1cf43528add6d7dea4268ab965`; its canonical Git-blob
  SHA-256 remains
  `5d0c5f8c96f32aa9e49b5bef96c2aa026ed3559a6dc639dc20929608d3a475c9`.
  All 12 declared scientific-source hashes were independently recomputed from
  Git bytes at source commit `5cb7e81434742f7bf1b796018239c97ca1731d87`.
- The private Kaggle wrapper compiles and has SHA-256
  `d92df6a8d398f6067e2024ceba23b63a7fe27b0f51afc25a0ed1882f695af98b`;
  kernel metadata SHA-256 is
  `2424aa7110116343d8a9d1211be3891e9fc55ebf39a35d9ba78a808a0fc88c68`.
  Static prelaunch evidence is stored at
  `artifacts/research_protocols/rad_dino_multilayer_soft_region_probe_val_v1_wrapper_audit.json`.
- The wrapper reconstructs and verifies the frozen CRLF split and affinity-v3
  baseline, pins the RAD-DINO revision and three model-file hashes, and reruns
  all nine focused Torch/AST tests before heavy execution. Its fixed order is
  prediction runner, independent physical verification of the checkpoint,
  teacher/history, manifest and all 371 maps without GT, separate validation-GT
  evaluator, then post-GT cohort/gate verification. The evaluator cannot run
  until the no-GT verification passes.
- The prelaunch audit is `PRELAUNCH_PASS`. Heavy training has not run yet,
  no validation GT has been read for this candidate, `consumer_trained=false`,
  and `test_evaluated=false`. The kernel may launch only after this wrapper
  audit is committed and pushed.
- After the wrapper evidence was committed and pushed at `f9d6fe0`, private
  Tesla-T4 Kaggle kernel
  `itsthang333/btxrd-rad-dino-multi-layer-soft-region-probe-v1` version 1 was
  launched and entered `RUNNING`. Kaggle normalized `multilayer` in the
  requested metadata ID to `multi-layer` in the authoritative URL slug and
  rejected only the optional hyphenated tags; neither changed the uploaded
  wrapper or scientific contract. Exactly one five-minute heartbeat,
  `theo-d-i-rad-dino-multilayer-soft-region`, follows this exact slug. No
  duplicate monitor, competing heavy local job, consumer training or test
  evaluation was created.

## 2026-07-27 - Multi-layer soft-region v1 gate-schema defect found in flight

- While kernel version 1 was still `RUNNING`, preparation of the independent
  result auditor exposed a deterministic evaluator-only schema mismatch. The
  frozen `paired_group_bootstrap` helper returns confidence bounds as
  `ci95: [low, high]`, but `apply_gate` attempted to read a nonexistent
  `ci95_low` field. A direct local reproduction raises `KeyError: 'ci95_low'`.
  The original focused gate test had hand-built the nonexistent field and
  therefore failed to cover the real helper-to-gate interface.
- The correction reads element zero from the validated two-element `ci95`
  list and changes the focused test to the helper's physical output schema.
  All nine focused Torch/AST tests pass. Prediction generation, checkpoint,
  training inputs, architecture, losses, validation metrics, 10,000-replicate
  bootstrap values, gate thresholds and decision logic are unchanged.
- Corrected evaluator/test source is committed and pushed at
  `bb767b05f665886d27d7fb50abd8701fa44d2da6`; canonical hashes are
  `9eae91da9753e20d5f84fc5575b31ce0b6b2c95ad54f902fcf0ea96e02775f08`
  and
  `f54b12b8d68029df283a21f13ad530624d20d602e17595033ad7bcbe721b14a9`.
  The implementation-only amended protocol is frozen before any version-2
  prediction at SHA-256
  `4fc754c9b9046aff0220d94b54022facbaa679924590bb119e4957af4ff1bef5`.
  No version-1 result will be accepted; version 2 cannot launch while version
  1 is still the active heavy job. Test remains locked.
- The corrected version-2 wrapper is prepared but not launched. It binds
  checkout commit `648a439a74c7f5087eb9812d98f00380c2455e62`, corrected scientific source
  `bb767b05f665886d27d7fb50abd8701fa44d2da6` and amended protocol
  `4fc754c9...1bef5`. Static source/order/metadata audit remains
  `PRELAUNCH_PASS`. After version 1 exposed that its nested execution log was
  kept only under ephemeral `/kaggle/temp`, the wrapper-only observability
  path was hardened to stream subprocess output to the direct Kaggle log and
  retain the nested log, wrapper and protocol in partial output on failure.
  The final version-2 wrapper SHA-256 is
  `40b87ecc5b7838efd996fb13a7ca9e8b12aa951d8012a5e6e59d4368639dd69f`.
  Evidence is stored in
  `rad_dino_multilayer_soft_region_probe_val_v1_wrapper_audit_v2.json`.
  The existing heartbeat remains attached to version 1 until it is terminal;
  no duplicate monitor or second heavy job is created.
- Version 1 reached terminal `ERROR` after `614.23` seconds and the sole
  heartbeat was deleted before output inspection. Direct Kaggle logs show the
  prediction runner subprocess returned exit code 1; the nested traceback was
  unavailable because the v1 wrapper redirected it only to ephemeral
  `/kaggle/temp`. The retained partial output contains only
  `teacher_metadata.json`, proving failure occurred after feature/teacher cache
  construction but before any checkpoint, validation map, prediction freeze,
  validation-GT evaluator or gate.
- Partial teacher evidence is internally coherent: `1493` normal and `1488`
  positive training images, layers `4/8/12`, projection dimension/seed
  `128/42`, variable positive foreground counts `0/154/951` for
  minimum/median/maximum, exactly one positive image without foreground
  weight, `validation_gt_read=false` and `test_evaluated=false`. A full-shape
  CPU batch-8 forward/backward smoke test passes, so the exact Kaggle runner
  failure remains unobserved rather than guessed. No version-1 scientific
  result is accepted.
- The first version-2 upload attempt returned HTTP `409 Conflict` before a new
  version was created because metadata still contained the originally
  requested `multilayer` ID while Kaggle's authoritative v1 URL uses
  `multi-layer`. Metadata alone was corrected to the existing exact slug;
  metadata SHA-256 is now
  `5db776ac222209c112beddd135a97a6bd91b485d7838b56fa0839cf7da5b95ef`.
  Wrapper, protocol, source and scientific settings are unchanged, and no
  concurrent heavy job or monitor resulted from the rejected upload.
- Corrected kernel version 2 was then pushed successfully to the same private
  authoritative slug and entered `RUNNING`. It uses wrapper
  `40b87ecc...9dd69f`, amended protocol `4fc754c9...1bef5`, corrected source
  `bb767b05...2da6` and unchanged prediction/training contract. Exactly one
  five-minute heartbeat, `theo-d-i-rad-dino-multilayer-soft-region`, follows
  version 2; no duplicate monitor or heavy job exists.

## 2026-07-27 - Multi-layer soft-region v2 ambiguous-only teacher error

- Kernel version 2 terminated after `622.30` wrapper seconds. The single
  heartbeat was deleted before inspection. The hardened direct log provides
  the exact nested traceback: after all `2981` feature caches, `1493` normal
  calibration maps and `1488` positive teacher maps completed, the first
  training epoch raised
  `RuntimeError: Positive image has no calibrated soft-region evidence` in
  `soft_region_pseudo_loss`. No checkpoint, validation prediction, prediction
  freeze, validation-GT evaluation, consumer or test access occurred.
- The retained teacher metadata had already declared exactly one positive
  image without foreground weight. Its teacher contains neither foreground
  evidence above `0.90` nor background evidence below `0.50`; all available
  pixels are ambiguous. The protocol explicitly says such an image retains
  image-level SmoothMax supervision with no artificial foreground rank, so
  rejecting it is an implementation error, not a scientific failure.
- The corrected loss omits only the unavailable pseudo-region term for an
  ambiguous-only positive; image-level BCE remains active and no pseudo pixel
  is injected. If an entire batch has no pseudo-region evidence, it returns a
  differentiable zero for that loss only. A regression test verifies zero
  pseudo-region gradient for this case, while the existing normal dense
  background and positive variable-region tests remain intact. All ten focused
  tests pass.
- Corrected source commit is
  `0ea383e48bcd91f7235d1e1130e77d80f099cbc4`; decoder/test canonical hashes
  are `d93986255d31e3b65e7bb1f2176079a67fcda35d10629737decf916a5048d481`
  and `14e8660b1c89e557af9a8d031237534d02d2968b804f893d8b253e63087b33b3`.
  The implementation-only amended protocol is frozen before version-3
  prediction at SHA-256
  `4093521cacdabea003273c3ec1274bc6c169fbf647a6b5eafe96925b1e36202e`.
  Architecture, train data, thresholds, loss weights, prediction, metric,
  bootstrap and gate are unchanged; no version-2 result is accepted.
- Version-3 wrapper static audit is `PRELAUNCH_PASS`. It binds checkout
  `2b5e541b7c5239e34de4500975e21d2986ff3e42`, scientific source
  `0ea383e48bcd91f7235d1e1130e77d80f099cbc4`, amended protocol
  `4093521c...36202e` and all 12 canonical source hashes. Wrapper SHA-256 is
  `1a2cd414ba15a798cc7c595882b50bafcde09c4560973ee0c891d0e41613dec3`;
  metadata remains `5db776ac...5b95ef`. Ten focused tests pass and execution
  order remains prediction, no-GT physical freeze verification, separate GT
  evaluator, post-GT verification. Version 3 has not launched at the time of
  this freeze.
- After the version-3 audit was committed and pushed at `ec00214`, kernel
  version 3 was launched on the same private Tesla-T4 slug and entered
  `RUNNING`. Exactly one five-minute heartbeat,
  `theo-d-i-rad-dino-multilayer-soft-region`, follows version 3. No competing
  heavy job, duplicate monitor, consumer training or test evaluation exists.
- An independent post-download auditor is committed at
  `0fea848256b8286ab80ad6eb7aa0d584c4c7c0ed`. Auditor SHA-256 is
  `e36d3a661e0e495877dc583f070f037d71a2f8be2f0f3383cd83c1f5c8062ca4`;
  its focused test SHA-256 is
  `12d905040ede17fdfce8dcc8c256b05251f6b8f35cb9061bc623ce7c3547eb54`.
  It verifies wrapper/protocol/Git-source bindings, training evidence and all
  371 physical map hashes/ranges before importing validation GT, then
  independently recomputes every per-image metric, complete miss, 10,000
  complete-group bootstrap comparison and gate and requires exact agreement
  with cloud artifacts. Eleven combined decoder/evaluator/auditor tests pass;
  the auditor contains no test-split call.

## 2026-07-27 - Multi-layer soft-region v3 terminal audit and gate decision

- Kernel version 3 reached terminal `COMPLETE`; the sole five-minute
  heartbeat was deleted before artifact inspection, leaving no active or
  duplicate monitor. Bulk Kaggle output transport timed out once and a later
  bounded retry was interrupted by Windows socket error 10053. Completeness
  was recovered by downloading only validation IDs missing from the frozen
  split. Two previously downloaded maps (`IMG002099.npy` and
  `IMG003204.npy`) were zero-byte transport artifacts; both were re-fetched
  with `--force` and matched their frozen prediction-manifest SHA-256 values.
  No prediction was regenerated or altered.
- The independent auditor then verified all `371` physical maps and hashes,
  `76,028,288` prediction bytes, wrapper/protocol/Git-source/split/baseline
  bindings, 12-epoch history, teacher metadata, checkpoint, prediction freeze,
  direct Kaggle output provenance, cohort `371/184/187`, subgroup counts
  `94/72/18`, complete misses and exact cloud/local agreement. Its initial
  local invocation lacked `PYTHONPATH` and stopped at the dataset import; the
  corrected invocation used `PYTHONPATH=D:\thesis\project` without source or
  protocol changes. Validation GT was read only after the no-GT physical
  audit; `consumer_trained=false` and `test_evaluated=false` throughout.
- Integrity/protocol audit status is `PASS`, while the frozen scientific gate
  is `FAIL`. The candidate achieved image-level AUROC `0.81024762`; p90 Dice
  overall/small/medium/large was
  `0.14317068 / 0.01445152 / 0.21741629 / 0.51838823`. Relative to frozen
  affinity-decoder v3, p90 Dice changed by
  `+0.05204674 / -0.00195504 / +0.10180003 / +0.13504283`; the paired
  10,000-replicate overall 95% CI was `[+0.03570832, +0.06939177]`.
  Medium and large localization improved materially, but the small subgroup
  did not: small Dice p97 was `0.02912824`, below the predeclared `0.03`, and
  the no-subgroup-decrease check also failed. Complete p90 misses were
  overall/small/medium/large `39/35/4/0`.
- Therefore this exact multi-layer soft-region mechanism is rejected for
  pseudo-mask/consumer training despite its overall, medium and large gains.
  It does not meet the operational Dice goals and cannot be promoted under the
  all-checks-required gate. Compact auditable evidence is frozen under
  `artifacts/kaggle/rad_dino_multilayer_soft_region_probe_val_v1/`, including
  prediction/evaluation manifests, cloud outputs and the independent audit;
  dense maps and the checkpoint remain outside Git in the isolated temporary
  download. Test remains locked.

## 2026-07-27 - Small-tumor bottleneck diagnosis and literature-grounded next mechanism

- Post-freeze validation diagnostics explain why the multi-layer decoder can
  improve medium/large while failing small. Small tumors have mean/median GT
  area ratios `0.001748 / 0.000942` (median approximately 96 pixels at
  320x320), with range `0.000068--0.009473`. The frozen RAD-DINO input is
  448x448 with 14-pixel patches, and the decoder predicts first on a 64x64
  grid before interpolation. A median small tumor therefore occupies only a
  few decoder cells; the smallest tumors are sub-cell at this resolution.
  This resolution mismatch is a project inference from frozen BTXRD geometry,
  not a claim copied from the papers below.
- A fixed post-freeze percentile diagnostic confirms that thresholding alone
  cannot solve the missing spatial signal. Small mean Dice rises from
  `0.01445152` at p90 to `0.05822649` at p99.75, but complete misses worsen
  from `35/94` to `71/94`. Even an oracle choosing the best member of the
  precomputed p90/p95/p97/p98/p99/p99.25/p99.5/p99.75/p99.9 grid per image
  reaches only `0.07040377` mean Dice and retains `35` complete misses. These
  values are diagnostic only and are not adopted as a new metric or tuned
  prediction rule.
- Coarse saliency is nevertheless useful for ROI proposal. A deterministic
  GLAM-style greedy window diagnostic on the already frozen maps, without
  using GT to choose the windows, found that three 160x160 top-mass windows
  intersect a small tumor in `92/94` cases and fully contain it in `85/94`;
  six windows reach `93/94` and `88/94`. Thus the next mechanism should reuse
  the global map for proposal/context but perform local high-resolution weak
  learning inside the selected regions.
- Literature sources consulted on 2026-07-27 and their concrete influence:
  - Chen and Sun, *Weakly-supervised Semantic Segmentation with Image-level
    Labels: From Traditional Models to Foundation Models*, ACM Computing
    Surveys 57(5), 2025, DOI `10.1145/3707447`,
    https://doi.org/10.1145/3707447. This survey supports organizing the
    design as image-label localization, seed/refinement and optional
    pseudo-mask consumer stages; it does not justify reading training masks.
  - Mun, Lee, Uh, Choe and Byun, *Small Objects Matters in
    Weakly-supervised Semantic Segmentation*, WACV 2024, pp. 413-422, DOI
    `10.1109/WACV57701.2024.00048`,
    https://openaccess.thecvf.com/content/WACV2024/papers/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.pdf.
    The paper shows that aggregate WSSS metrics hide poor small-instance
    behavior. It weights pixels inversely by pseudo-mask connected-component
    size and then uses elastic weight consolidation (EWC) to preserve the
    previously learned large-object task. For BTXRD, the directly transferable
    principle is explicit per-size evaluation and preservation of the frozen
    global path; component weighting is deferred until trustworthy local
    pseudo-components exist, because applying it to current noisy components
    could amplify false positives.
  - Liu, Shen, Wu, Chledowski, Fernandez-Granda and Geras,
    *Weakly-supervised High-resolution Segmentation of Mammography Images for
    Breast Cancer Diagnosis*, PMLR 143:268-285, 2021, PMID `35088055`,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8791642/ and
    https://arxiv.org/abs/2106.07049. Their GLAM framework uses a coarse global
    saliency map to select high-resolution patches, trains a local module with
    bag-level image labels/top-fraction pooling, and fuses global/local maps.
    It explicitly targets medical ROIs at or below 1% of image area. GLAM used
    six patches during training for recall and fewer at inference to limit
    false positives; this is the primary basis for the proposed BTXRD 6-train,
    3-inference ROI contract.
  - Zhang, Yu, Wei, Zhao and Xiao, *Frozen CLIP: A Strong Backbone for
    Weakly Supervised Semantic Segmentation (WeCLIP)*, CVPR 2024,
    pp. 3796-3806,
    https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf.
    WeCLIP supports retaining a frozen foundation backbone, training a light
    spatial decoder and refining frozen guidance dynamically. The BTXRD design
    transfers that mechanism to frozen RAD-DINO rather than copying CLIP or
    using text prompts.
  - Yang et al., *Anomaly-guided weakly supervised lesion segmentation on
    retinal OCT images*, Medical Image Analysis 94:103139, 2024, DOI
    `10.1016/j.media.2024.103139`,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11016376/. The method combines
    normal/anomaly evidence, self-attention and iterative refinement to improve
    small medical-lesion localization. It motivates retaining normal-image
    dense suppression and confidence calibration in the local MIL branch,
    while avoiding its modality-specific GAN synthesis.
- The selected next probe is therefore a controlled global-local transfer:
  freeze the audited v3 global checkpoint, use its saliency only to propose
  content-aligned 160x160 ROIs, extract frozen RAD-DINO local tokens, and train
  one shared local decoder with image-level MIL BCE/top-fraction pooling,
  dense normal suppression, positive sparsity and flip consistency. Training
  uses six ROI instances per bag; prediction uses three. A confidence-gated,
  sparse local residual is fused into the unchanged global map. No size label,
  segmentation mask or validation GT enters proposal, training or prediction;
  prediction must freeze before the separate GT evaluator. Test remains
  locked and no pseudo-mask consumer is authorized unless the predeclared
  small-improvement and medium/large-preservation gate passes.

## 2026-07-27 - Global-local MIL v1 protocol and prelaunch audit

- The literature-grounded small-tumor mechanism is implemented in
  `project/models/rad_dino_global_local_mil.py`,
  `project/run_rad_dino_global_local_mil_probe.py` and
  `project/evaluate_rad_dino_global_local_mil_probe.py`. Source/evaluator
  commit `9a62bb7b91632f624b3d188d1fba499c3170e046` is pushed. The full suite had
  `232` passing tests before protocol freeze; the final focused model/runner/
  evaluator suite has `13` passing tests, and all new files pass `py_compile`
  and Ruff.
- Protocol
  `artifacts/research_protocols/rad_dino_global_local_mil_probe_val_v1.json`
  was frozen before any new prediction at SHA-256
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`
  and committed at `0c7d371ab81f4fa0e4ea325ff8f1a3bbc4a90bd3`. It includes the full
  bibliographic/URL basis already recorded above, source hashes, the
  image-level-only supervision contract, fixed 6-train/3-inference 160x160 ROI
  geometry, losses, fusion, metrics, bootstrap and all-checks-required gate.
- The gate requires absolute fused Dice p90 overall/small/medium/large of at
  least `0.145/0.025/0.217/0.518`, small p99 at least `0.060`, a strictly
  positive lower 95% paired-bootstrap bound for small p90, no negative mean
  p90 delta in any overall/small/medium/large stratum, and fewer than `35`
  small p90 complete misses. Local-only maps are diagnostic and cannot replace
  the predeclared fused primary arm after GT is seen.
- The Kaggle wrapper binds checkout `0c7d371...`, scientific source
  `9a62bb7...`, protocol `f5941a20...a572`, and the direct v3 global
  checkpoint/freeze/manifest/per-image hashes
  `33ff0188...642f / 8d6225af...c0c2 / 3647fefb...fc63 /
  84b3dca0...eb2e`. It explicitly excludes any `thesis_source` evidence
  duplicate when discovering the one direct kernel-output mount. Wrapper and
  metadata SHA-256 are
  `00022389d6bd24bf4732a5e0dcf893fa281b97312adc2a3d0e9c7204d3477dc8`
  and
  `3a96403dbc825f0a8a6f731cb5b345095ff90a87f99a92ab6c28b61bd3c79399`.
  A real no-GT local smoke audit loaded all 371 global maps and the 364,324-
  parameter frozen global checkpoint. Static wrapper audit is
  `PRELAUNCH_PASS`; no new prediction, consumer training or test evaluation
  has occurred at this point.

## 2026-07-27 - Global-local MIL v1 Kaggle launch

- Private Kaggle kernel
  `itsthang333/btxrd-rad-dino-global-local-mil-probe-v1` version 1 was
  launched from the pre-audited wrapper. Its status after launch was
  `KernelWorkerStatus.RUNNING`. The launch did not change the scientific
  contract: protocol SHA-256
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`,
  scientific source commit
  `9a62bb7b91632f624b3d188d1fba499c3170e046`, wrapper SHA-256
  `00022389d6bd24bf4732a5e0dcf893fa281b97312adc2a3d0e9c7204d3477dc8`,
  and metadata SHA-256
  `3a96403dbc825f0a8a6f731cb5b345095ff90a87f99a92ab6c28b61bd3c79399`
  remain frozen.
- Exactly one five-minute heartbeat monitor is active:
  `theo-d-i-rad-dino-global-local-mil`. It watches only this kernel and must
  be deleted before terminal output download/audit. No duplicate monitor or
  heavy local job was created. Test remains locked, and no consumer may be
  trained before the predeclared validation gate is audited.
- Research-log citation policy is now explicit for all subsequent work: each
  investigated or adopted technique must be recorded with its primary paper
  or authoritative survey, bibliographic identity, DOI and/or stable official
  URL, plus a concise statement of which mechanism was transferred, adapted,
  deferred or rejected for BTXRD. This preserves traceability for the thesis
  report and does not by itself justify a method unless the frozen protocol
  and leakage constraints are also satisfied.

## 2026-07-27 - Global-local MIL v1 Kaggle version 1 implementation error

- Kernel version 1 terminated with `KernelWorkerStatus.ERROR`; the sole
  five-minute heartbeat `theo-d-i-rad-dino-global-local-mil` was deleted
  before terminal artifacts were downloaded. Direct Kaggle output was saved
  under the isolated local audit directory
  `tmp/kaggle/rad_dino_global_local_mil_probe_val_v1_output_v1_error_20260727`.
- This is an implementation/runtime error, not a scientific result. Kaggle
  allocated a Tesla P100 with CUDA capability `sm_60`, whereas the current
  base-image PyTorch binary reported support only for `sm_70` and newer. The
  first RAD-DINO convolution raised `cudaErrorNoKernelImageForDevice` while
  building the cache. Focused tests had already passed `13/13`, but there were
  zero rows in `training_proposals.csv`, no prediction freeze, no validation
  GT evaluation, no consumer training and no test access.
- Direct error-evidence hashes are: execution log
  `8f779535fddee494a26348bc632ffe82e0353c4362de59d8a0773a9fa8063bc0`,
  exact frozen protocol
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`,
  failed wrapper
  `00022389d6bd24bf4732a5e0dcf893fa281b97312adc2a3d0e9c7204d3477dc8`,
  and empty proposal manifest
  `9212f3c02dcf9e712caa978bd4f5fe850e0886e81244906883ef1d0ee783ec3e`.
- The version 2 wrapper correction pins the official PyTorch `2.5.1`,
  TorchVision `0.20.1` and TorchAudio `2.5.1` CUDA 12.1 wheel family, then
  requires the allocated device capability to occur in
  `torch.cuda.get_arch_list()` and executes a real CUDA convolution before
  tests or training. The official version/install matrix is PyTorch,
  *Previous PyTorch Versions*,
  https://pytorch.org/get-started/previous-versions/. This operational change
  does not alter the model, data, supervision, seed, hyperparameters,
  scientific source, protocol or gate.
- Corrected wrapper SHA-256 is
  `156a72fe8e8422d602bac701d1dac64ecfe22e0fcf0f29239d8bd5041578f995`;
  metadata remains
  `3a96403dbc825f0a8a6f731cb5b345095ff90a87f99a92ab6c28b61bd3c79399`.
  The only wrapper diff is the pinned official CUDA runtime installation and
  the architecture/convolution preflight. It passes `py_compile`, Ruff and
  JSON/hash audit. The protocol remains byte-identical at
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`.

## 2026-07-27 - Global-local MIL v1 H100 accelerator selection

- The official Kaggle CLI supports an explicit accelerator argument and lists
  `NvidiaTeslaP100`, `NvidiaTeslaT4`, `NvidiaTeslaA100`, `NvidiaL4`,
  `NvidiaH100` and other permission-dependent shapes. Source: Kaggle,
  *Kernels Commands - kaggle kernels push*,
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md. The installed
  CLI also maps `--accelerator` directly to the API `machine_shape` field.
- For this exact workload, `NvidiaH100` was selected ahead of T4 x2. The
  dominant workload is frozen RAD-DINO Transformer inference, and the runner
  already executes encoder forwards under CUDA FP16 autocast. A single H100
  can exploit that path without introducing data-parallel replication,
  inter-device gathers or a new numerical/source branch. T4 x2 would require
  source changes to distribute each six-ROI batch, while all single-image
  global proposal forwards would still occupy only one device.
- Kernel version 2 was superseded while still `RUNNING` solely to apply the
  user-requested faster accelerator; it is not a scientific result and no
  partial output will be used. The same staged wrapper was pushed with
  `--accelerator NvidiaH100`; Kaggle accepted kernel version 3, which entered
  `KernelWorkerStatus.RUNNING`. Exactly one existing five-minute heartbeat was
  retargeted from version 2 to version 3; no second kernel or monitor was
  created.
- No code or scientific setting changed between versions 2 and 3. Wrapper,
  metadata and protocol SHA-256 remain respectively
  `156a72fe8e8422d602bac701d1dac64ecfe22e0fcf0f29239d8bd5041578f995`,
  `3a96403dbc825f0a8a6f731cb5b345095ff90a87f99a92ab6c28b61bd3c79399`
  and
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`.
  Terminal audit must still verify that the runtime device is actually H100
  from the wrapper's CUDA preflight/provenance before making any speed claim.

## 2026-07-27 - Global-local MIL v1 version 3 terminal error and T4 x2 correction

- Terminal evidence disproved the earlier hardware assumption: although the
  CLI accepted `--accelerator NvidiaH100`, version 3 actually ran on
  `Tesla P100-PCIE-16GB`. The account owner confirmed that only Tesla P100 and
  T4 x2 are available. Therefore no H100 speed claim is retained. Direct
  execution log SHA-256 is
  `d8e6771ae53789a6d3cc6d047ed8755803a17ce4eb72bc7656a6e83c007bae90`;
  its CUDA preflight records PyTorch `2.5.1+cu121`, capability `sm_60`, the
  expected architecture support and a successful real convolution.
- Version 3 then failed before the first proposal row, prediction or GT read:
  the one-pass greedy selector could not extend its highest-mass 160x160
  starting window to six windows with pairwise IoU at most `0.25`. A complete
  geometry audit of all `441` stride-8 candidates found that the six-window
  contract is feasible, but `367/441` individual first positions cannot occur
  in any complete six-window set. Thus the protocol geometry was not
  impossible; the one-pass algorithm was incomplete.
- Scientific source commit
  `bc34a892cbdbe2eb0a8b3df9a365426d54e07af1` corrects selection without
  relaxing any scientific value. Positive candidates retain descending
  frozen-global saliency-mass order; normal candidates retain their seeded
  random order. Deterministic exact backtracking now returns the
  lexicographically first complete feasible set in that order, retaining a
  candidate only if the requested count can still be completed. Counts remain
  six train/three validation, with the same 160 window, stride 8 and IoU 0.25.
- The same source is prepared for the account-available T4 x2 shape. Frozen
  RAD-DINO local forwards return only projected layer-4/8/12 32x32x64 tokens,
  which `torch.nn.DataParallel` splits across both GPUs and gathers on device
  0. Six training ROIs therefore split 3+3; the single-image global proposal
  branch remains on device 0, and the small trainable decoder remains
  scientifically unchanged. Float32 projection is explicitly preserved after
  FP16 encoder inference.
- The correction is frozen before any new prediction in
  `artifacts/research_protocols/rad_dino_global_local_mil_probe_val_v1_t4x2_correction_v1.json`
  at SHA-256
  `3d948dd11ac46c09ce84e8de04034255a11be0182bfb17b44216c8d4c2172bf8`.
  It inherits the immutable base protocol
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`
  and changes neither supervision, losses, seed, fusion, metrics nor gate.
  Tests are `35` focused and `256` full-suite passes; Ruff and `py_compile`
  pass. Validation GT and test were not read, and no consumer was trained.
- Runtime/tool references retained for reporting are Kaggle,
  *Kernels Commands - kaggle kernels push*,
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md, for explicit
  `NvidiaTeslaT4` machine-shape selection, and PyTorch,
  *Previous PyTorch Versions*,
  https://pytorch.org/get-started/previous-versions/, for the official
  PyTorch 2.5.1 CUDA 12.1 wheel family. Version 4 must fail before prediction
  unless exactly two visible CUDA device names contain `T4` and a real
  convolution succeeds on each device.

## 2026-07-27 - Global-local MIL version 4 T4 x2 launch

- Kaggle accepted kernel version 4 with both metadata `machine_shape` and CLI
  override set to `NvidiaTeslaT4`; initial status is
  `KernelWorkerStatus.RUNNING`. Wrapper/metadata SHA-256 are
  `fb62f8c3254ed20ec3c4d6042510ca17c5c4bced2b0dc4a267fb661bf6066add`
  and
  `8d0d416e2ef3a9f3467de49000f9ca8cbfc5da39375f432b29b7b440c70df29d`.
  Checkout commit is `2148c76c2fb38283be7c2a1550cc315d904c6ca4`,
  scientific source is
  `bc34a892cbdbe2eb0a8b3df9a365426d54e07af1`, and correction protocol is
  `3d948dd11ac46c09ce84e8de04034255a11be0182bfb17b44216c8d4c2172bf8`.
- Exactly one five-minute heartbeat
  `theo-d-i-rad-dino-global-local-mil` watches version 4. The terminal audit
  must treat API acceptance as insufficient and verify two actual T4 names,
  two successful device convolutions, `cuda_device_count=2` and
  `local_encoder_data_parallel=true`. Test remains locked and no consumer is
  authorized before the unchanged all-checks-required gate.

## 2026-07-27 - Global-local MIL version 4 terminal audit and rejection

- Kaggle kernel
  `itsthang333/btxrd-rad-dino-global-local-mil-probe-v1` version 4 completed.
  The sole five-minute heartbeat `theo-d-i-rad-dino-global-local-mil` was
  deleted before output download. Terminal artifacts were downloaded into the
  isolated directory
  `tmp/kaggle/rad_dino_global_local_mil_probe_val_v1_output_v4_complete_20260727_0915`.
  The Kaggle CLI ended its local download command with a Windows console
  encoding error only after writing the artifacts; an independent file audit
  established that the scientific output itself is complete.
- Runtime evidence passes the T4 x2 correction contract. Execution log
  SHA-256 is
  `c685c232d3632e4c10c85fcf01a2f1519197cd15fc260484fa5cef7b4b5b39db`.
  Its real convolution preflight records PyTorch `2.5.1+cu121`, CUDA `12.1`,
  exactly two `Tesla T4` devices at capability `sm_75`, and convolution sums
  `324.0/324.0`; focused tests pass `35/35`. `run_manifest.json` SHA-256
  `0be4e34f9fe60210185ab33f161737f68007681964837a76421226b8e6d614c8`
  independently records two T4 devices and
  `local_encoder_data_parallel=true`.
- Provenance is exact. Downloaded wrapper/protocol SHA-256 are
  `fb62f8c3254ed20ec3c4d6042510ca17c5c4bced2b0dc4a267fb661bf6066add`
  and
  `3d948dd11ac46c09ce84e8de04034255a11be0182bfb17b44216c8d4c2172bf8`,
  byte-identical to the staged wrapper and frozen correction protocol. The
  inherited base protocol remains
  `f5941a203a7f003b9f534ede793ca9ce07dffee9a0e9c74049f68ee02f26a572`.
  All correction-protocol source hashes match the current checkout and those
  files are unchanged from scientific commit
  `bc34a892cbdbe2eb0a8b3df9a365426d54e07af1`. The wrapper found exactly one
  direct global kernel-output mount under `/kaggle/input/notebooks`, excluding
  `thesis_source`; its checkpoint/freeze/manifest/per-image hashes match the
  frozen v3 evidence
  `33ff0188...642f / 8d6225af...c0c2 / 3647fefb...fc63 /
  84b3dca0...eb2e`. Split SHA-256 remains
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`.
- Prediction freeze is valid before GT access. The independent audit found
  exactly `371` unique validation rows, `371` fused maps and `371` local-only
  maps. Every map hash matches `prediction_manifest.csv`; all global reference
  map hashes also match the independently downloaded v3 maps. All `1,113`
  global/local/fused arrays are finite `320x320` float16 arrays bounded in
  `[0,1]`. Important freeze hashes are: checkpoint
  `3cf2ba2f...56d3`, training proposals `f285a922...460`, training history
  `83ba9684...3798`, train-normal calibration `27381e0d...3018`, prediction
  manifest `01048705...82c`, prediction freeze `467e1ba3...fb5d`, and pre-GT
  audit `2394c899...8e38`. The pre-GT records state
  `validation_gt_read=false`, `consumer_trained=false`, and
  `test_evaluated=false`.
- Post-freeze evaluation also passes the audit contract: cohort
  validation/tumor/normal is `371/184/187`, tumor subgroup counts are
  small/medium/large `94/72/18`, complete misses are included, and paired
  complete-group bootstrap uses `10,000` replicates. Summary, paired
  comparison, gate and evaluation-audit SHA-256 are respectively
  `2c46d93b...94c36`, `e3487740...ddf7`, `92b0fa0c...91156`, and
  `e2628c4b...a5ffc`. Consumer and test flags remain false.
- The predeclared fused arm **fails**. Frozen-global v3 Dice p90
  overall/small/medium/large was
  `0.14317068 / 0.01445152 / 0.21741629 / 0.51838823`; fused v4 is
  `0.14308632 / 0.01438807 / 0.21727784 / 0.51841118`; local-only diagnostic
  is `0.04528690 / 0.00507460 / 0.06639630 / 0.17084688`. Fused image-level
  AUROC is `0.80847477` versus global `0.81024762`. Fused-minus-global p90
  deltas are `-0.00008435 / -0.00006346 / -0.00013846 / +0.00002295`, and
  small p90 95% paired-bootstrap CI is
  `[-0.00012262, -0.00001797]`. Absolute overall p90, small p90 and small p99
  (`0.04528470`) fail; the required positive small CI and no-decrease checks
  fail. Although small p90 complete misses decrease from `35` to `34`, this
  isolated count does not override the all-checks-required gate.
- The bottleneck is spatial shortcut learning, not ROI proposal recall or
  accelerator execution. On small tumors the local branch has pixel AP
  `0.00516770`, Dice p90 `0.00507460`, and local argmax hit exactly `0/94`,
  even though its median local confidence is `0.98055` and `62/94` small cases
  have confidence above `0.5`. Median small fusion gate is `0.53670`.
  Training image BCE falls from `0.64210` to `0.17340`, so the decoder learns
  the bag label while its hottest pixels are systematically not the lesion.
  The fixed fusion promotes only the top `2%` local values; it improves only
  `3/94` small p90 cases, degrades `23/94`, and is float16-identical in the
  remaining `68/94`. Thus local classification confidence is not a valid
  surrogate for localization fidelity. The exact global-local configuration
  is rejected without post-GT retuning. No pseudo-mask consumer is authorized,
  test remains locked, and the operational Dice goal remains active.

### Literature follow-up after the failed local-residual mechanism

- Jiang, Yang, Hou and Wei, *L2G: A Simple Local-to-Global Knowledge Transfer
  Framework for Weakly Supervised Semantic Segmentation*, CVPR 2022,
  pp. 16886-16896,
  https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.pdf.
  L2G uses attention from multiple local crops to teach a global network
  online. The transferable mechanism is coordinate-aligned local-to-global
  distillation; the failed BTXRD practice of independently classifying local
  bags and directly adding their peaks is rejected.
- Wang, Zhang, Kan, Shan and Chen, *Self-Supervised Equivariant Attention
  Mechanism for Weakly Supervised Semantic Segmentation (SEAM)*, CVPR 2020,
  pp. 12275-12284,
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf.
  SEAM constrains CAMs from transformed views and refines them through pixel
  correlation. BTXRD should transfer overlap-coordinate cross-scale/view
  equivariance; the current same-grid horizontal-flip penalty is empirically
  too weak and is not retained as the sole spatial constraint.
- Ru, Zheng, Zhan and Du, *Token Contrast for Weakly-Supervised Semantic
  Segmentation (ToCo)*, CVPR 2023, pp. 3093-3102,
  https://openaccess.thecvf.com/content/CVPR2023/papers/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf.
  ToCo uses diverse intermediate ViT tokens to supervise final-token relations
  and contrasts uncertain local regions with global semantics. The relevant
  adaptation is contrastive relational guidance from frozen RAD-DINO layers
  4/8/12, rather than treating a saturated MIL scalar as spatial evidence.
- Ru, Zhan, Yu and Du, *Learning Affinity From Attention: End-to-End
  Weakly-Supervised Semantic Segmentation With Transformers (AFA)*, CVPR 2022,
  pp. 16846-16855,
  https://openaccess.thecvf.com/content/CVPR2022/papers/Ru_Learning_Affinity_From_Attention_End-to-End_Weakly-Supervised_Semantic_Segmentation_With_Transformers_CVPR_2022_paper.pdf.
  AFA learns symmetric semantic affinity from transformer attention using
  reliable foreground/background/ignore pseudo-relations, then propagates
  seeds. For BTXRD, raw frozen attention or unrestricted propagation remains
  rejected; only confidence-masked, local-window affinity anchored to frozen
  global seeds is a candidate.
- Together with the previously recorded Chen--Sun survey, WACV small-object
  analysis, GLAM, WeCLIP and medical anomaly-guidance evidence, these papers
  support a distinct next probe: frozen-global-seed-anchored, coordinate-aware
  local-to-global transfer with cross-view equivariance and intermediate-token
  affinity/contrast. It must receive a new prediction-first protocol and gate
  before any validation prediction. It must not reuse the failed local scalar
  confidence as a fusion gate, retune v4 after GT, or train a consumer before
  passing that gate.

## 2026-07-27 - Pre-protocol decision: class-agnostic mask-bag MIL

- Source audit after the rejected global-local residual shows that another
  free pixel decoder would repeat the same shortcut: image BCE can become very
  good while the hottest location is unrelated to the lesion. The next
  prediction-first probe therefore constrains every spatial output to a
  class-agnostic SAM proposal. Frozen RAD-DINO layers `4/8/12` provide
  inside-mask, local-context and inside-minus-context descriptors; a small MIL
  scorer learns which proposal explains the binary image label. All proposals
  from normal images are reliable negative instances, while only the detached
  current winner in a positive bag is treated as positive and the uncertain
  remainder is ignored. A horizontally flipped, coordinate-aligned version of
  the same proposal receives a score-consistency penalty. At frozen validation
  inference, candidate logits are the arithmetic mean of original and aligned
  horizontal-flip logits, followed by the same SmoothMax MIL pool. The final
  map is the winner-take-all SAM shape multiplied by the image-bag probability.
  This is a selector probe only; it cannot launch a pseudo-mask consumer
  automatically.
- Seibold et al., *Self-Guided Multiple Instance Learning for Weakly
  Supervised Thoracic Disease Classification and Localization in Chest
  Radiographs*, ACCV 2020, official CVF record:
  https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html.
  The paper explicitly uses image- and patch-level predictions to construct
  auxiliary supervision while accounting for uncertain instances. BTXRD
  adapts that uncertainty rule to arbitrary SAM-mask instances: normal-bag
  masks are negative, the positive winner is positive, and other positive-bag
  masks are ignored rather than falsely labelled background.
- Shen et al., *Toward Joint Thing-and-Stuff Mining for Weakly Supervised
  Panoptic Segmentation*, CVPR 2021, official CVF record:
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html.
  Its MoIPool mechanism pools features from arbitrary-shape masks and applies
  MIL under image-level labels. BTXRD transfers the arbitrary-mask pooling
  principle, but uses area-resampled SAM masks over frozen radiology tokens
  and does not copy its panoptic branches or self-training consumer.
- Yang and Gong, *Foundation Model Assisted Weakly Supervised Semantic
  Segmentation*, WACV 2024, official CVF record:
  https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html.
  The supported transfer is using SAM only for boundary-aware class-agnostic
  proposal shapes and learning lesion semantics from image labels. Its
  CLIP/prompt architecture and direct downstream pseudo-label training are not
  copied, because RAD-DINO is the frozen domain encoder and the BTXRD consumer
  remains locked behind a separate gate.
- Liu et al., *Weakly-supervised High-resolution Segmentation of Mammography
  Images for Breast Cancer Diagnosis (GLAM)*, MIDL/PMLR 143, 2021, official
  PMLR record: https://proceedings.mlr.press/v143/liu21b.html. GLAM identifies
  low-resolution saliency as especially harmful when lesions are small and
  uses coarse ROI localization followed by fine-grained analysis. BTXRD
  transfers the coarse-candidate/fine-descriptor separation; it rejects the
  v4 practice of adding unconstrained high-confidence local pixels.
- Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
  Segmentation*, WACV 2024, official CVF record:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html.
  This is retained as direct literature support for a separate small-object
  failure mode. The mask-bag design preserves fractional occupancy with area
  interpolation onto the token grid instead of thresholding small masks away.
- Feasibility was checked against the already frozen flip-TTA SAM gallery,
  before this scorer is trained. Recomputing `oracle_best_single_dice` with
  the immutable subgroup rule `<1% / 1--5% / >=5%` gives candidate-oracle Dice
  overall/small/medium/large
  `0.40907629 / 0.22274968 / 0.59414817 / 0.64182777` on
  `184/94/72/18` positive images. These exceed the active operational goals
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336`; therefore proposal
  support does not make the gate impossible. Same-gallery selected Dice is
  only `0.23366822 / 0.11152529 / 0.34768577 / 0.41545552`, confirming that
  selection, especially for small lesions, has measurable headroom. These
  frozen prior values justify protocol design only and will not be used for
  post-result retuning.
- The same immutable validation pseudo-mask manifest shows a maximum of `81`
  SAM candidates per image (`21/371` images reach that maximum). The initial
  draft cap `64` is therefore rejected before protocol freeze because it would
  cause an implementation-only failure or silently require truncation. V1
  fail-closes at exactly `81`, which is the complete-gallery bound produced by
  the frozen three-percentile/three-component prompt ensemble; no proposal is
  ranked or discarded to satisfy memory limits.
- Supervision/safety boundary: candidate generation may use the binary image
  label to target the tumor CAM for both positive and normal images, but it
  never opens segmentation GT. Normal-image candidate masks are retained only
  as negative MIL instances and their ordinary pseudo-mask artifact is forced
  empty. Candidate manifests for all `2,981` train and `371` validation images
  must be hash-frozen before scorer training; validation annotations can be
  opened only after all `371` prediction-map hashes and the prediction freeze
  are verified. Test remains locked and every gate check is required.
- Runtime policy is fail-closed on Kaggle T4 x2: exactly two visible Tesla T4
  devices are required, the frozen RAD-DINO feature pass uses
  `torch.nn.DataParallel` on device IDs `0/1`, and the wrapper must prove real
  convolutions on both devices. The SAM/classifier proposal stage may place
  the classifier on T4-0 and SAM on T4-1, but this is device separation rather
  than a claim of synchronous data parallelism. No five-minute monitor is
  created until a kernel is actually launched, and only one monitor may exist.

### Mask-bag MIL protocol freeze and Kaggle launch

- Scientific source was committed and pushed in two pre-protocol commits:
  `b1a93b5805857a2abbd5ac92bef7ae75b793857d` adds the mask-bag scorer,
  prediction-first runner/evaluator, full-cohort candidate provenance and
  tests; `1567c2a05e77fcb5f514f9094f9e12791d8dd882` adds aligned
  original/flip candidate-logit TTA before SmoothMax pooling. Local
  `compileall`, diff-check and four static safety tests pass. Seven Torch
  model tests and candidate-diagnostic dynamic tests are deferred to the
  Kaggle preflight because both local Python environments lack Torch; the
  wrapper also runs the full repository test suite before the experiment.
- Protocol
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1.json`
  was frozen in separate commit
  `03b97e7200a646e890c80e20d759099f4b375696`; its SHA-256 is
  `a8f3101be461a1bdc007f442f60e8e3b50ccd6abf015f81f084c004829b7c4b9`.
  An independent pre-commit audit reproduced all `11` canonical Git source
  hashes from scientific commit `1567c2a...`. The all-checks-required gate is
  image AUROC `>=0.75`; selected-proposal Dice
  overall/small/medium/large `>=0.25/0.13/0.37/0.38`; proposal-oracle support
  at least the operational goals; overall paired CI95 lower bound positive;
  no subgroup mean decrease; and no complete-miss increase. Passing only
  authorizes a separately predeclared consumer protocol.
- Private Kaggle kernel
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1` version `1` was accepted
  at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1.
  Wrapper and metadata SHA-256 are
  `e819c46339d4f04f6feebd8d85bf17c2ddb57b240693e6aba4cad0b970e1dc2c`
  and
  `3086b5fa9bc1598831c45efd48a42dc8447447b290f5bc0c48d5b300e221caed`.
  The kernel requests `NvidiaTeslaT4`, fails unless exactly two actual T4s and
  real `324.0/324.0` device convolutions are observed, separates classifier
  and SAM onto `cuda:0/1`, and data-parallelizes the RAD-DINO encoder across
  both devices. Proposal generation itself is device-separated, not falsely
  reported as synchronous data parallelism.
- Exactly one heartbeat `theo-d-i-rad-dino-mask-bag-mil` was created only
  after version 1 was accepted. It checks once every five minutes and must not
  poll between heartbeats or create a duplicate. At terminal state it must be
  deleted before direct output/log download. Test remains locked and no
  consumer is authorized.

### Mask-bag MIL version 1 implementation-only preflight error

- Version 1 reached terminal `ERROR`; the sole five-minute heartbeat was
  deleted before direct output retrieval. Compact error evidence is isolated
  under
  `tmp/kaggle/rad_dino_mask_bag_mil_probe_val_v1_error_v1_20260727`.
  Execution log, downloaded wrapper and protocol SHA-256 are respectively
  `5b12b77086213e628a620756a85e216a337ecafff108e002d21b9869e400dc45`,
  `e819c46339d4f04f6feebd8d85bf17c2ddb57b240693e6aba4cad0b970e1dc2c`
  and
  `a8f3101be461a1bdc007f442f60e8e3b50ccd6abf015f81f084c004829b7c4b9`.
  The wrapper installed official PyTorch `2.5.1+cu121`, torchvision
  `0.20.1+cu121`, transformers `4.50.2` and pinned Segment Anything, then the
  exact mask-bag/candidate focused suite passed `17/17`.
- Failure occurred in the subsequent whole-repository test command, before
  RAD-DINO download, candidate generation, optimizer construction, validation
  prediction or GT access. The result was `196 passed`, `4 failed`, `4 errors`.
  Four errors came from a stale BiomedCLIP wrapper test whose only input was an
  ignored `tmp/kaggle/.../run_biomedclip_tiled_saliency_val.py` file already
  removed by the authorized cleanup. Three failures used a canonical LF Git
  split where the historical lock correctly expects the frozen Windows CRLF
  byte hash. The remaining failure required the fully supervised reference
  `best_unet.pt`, a `230,924,939`-byte file deliberately excluded by
  `.gitignore`. These are fixture portability/obsolete-test failures, not
  evidence about mask-bag learning, T4 execution or validation performance.
- Implementation-only repair: delete the orphaned BiomedCLIP wrapper auditor
  and its test because no tracked wrapper remains to audit; make the two split
  tests construct an exact temporary CRLF copy; and skip only the physical
  GT-checkpoint lock test when that explicitly untracked checkpoint is absent.
  All other GT-reference logic tests remain. No model, input, candidate,
  training, prediction, evaluator, bootstrap, gate or operational goal changes.
  A correction addendum must be frozen before rerunning version 2.

- The repair was verified locally with `10/10` affected tests, `compileall`
  and diff-check, then committed/pushed as
  `15efcc7c4feb44ec841e7bb22ebebe97c0f6ff10`. Correction addendum
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v1.json`
  was frozen separately in commit
  `7040f33dee9b6b53689eab673035cd3584418b10`; SHA-256 is
  `cbcb28c2ac2e4b1f61e18b28c01868aa16f9177d96db9b3fc5d6d1acf3867cad`.
  It requires the unchanged focused suite plus an unchanged whole-repository
  `pytest -q` result of exactly `199 passed, 1 skipped`; any other failure,
  error or skip stops before candidate generation.
- Kernel version `2` was accepted with corrected wrapper SHA-256
  `a18f5ea4ab406e9b7eee9cb0aa01907c2cdf04d796f637e39a2fe8140bf00c61`;
  metadata SHA-256 remains
  `3086b5fa9bc1598831c45efd48a42dc8447447b290f5bc0c48d5b300e221caed`.
  Base scientific protocol/hash, model, candidates, training, prediction,
  evaluation and gate are unchanged. Exactly one new five-minute heartbeat
  `theo-d-i-rad-dino-mask-bag-mil` was created after acceptance; no duplicate
  monitor exists.

### Mask-bag MIL version 2 second fixture-only preflight error

- Version 2 reached `ERROR`; the sole heartbeat was again deleted before
  direct log retrieval. Compact evidence is isolated under
  `tmp/kaggle/rad_dino_mask_bag_mil_probe_val_v1_error_v2_20260727`.
  Kaggle kernel log, execution log, downloaded wrapper, base protocol and
  correction-v1 SHA-256 are respectively
  `8f84b17692f8aef4e15a1ace2f43ff4e70608068a498a08550d6da8eb95a9605`,
  `0dd3e31498868333d5c7fbaf44a20ddbc4f398f97defc99985104afe89f1a5d8`,
  `a18f5ea4ab406e9b7eee9cb0aa01907c2cdf04d796f637e39a2fe8140bf00c61`,
  `a8f3101be461a1bdc007f442f60e8e3b50ccd6abf015f81f084c004829b7c4b9`
  and
  `cbcb28c2ac2e4b1f61e18b28c01868aa16f9177d96db9b3fc5d6d1acf3867cad`.
- The focused suite again passed `17/17`. Whole-repository tests improved to
  `198 passed, 1 skipped, 1 failed`. The only remaining failure is the same
  historical grid-gallery integration test: the split CSV was correctly
  reconstructed as CRLF, but its promoted-baseline, CPM-control, candidate
  per-image and pseudo-manifest CSVs still came from canonical LF Git bytes
  while their immutable runtime manifests require CRLF byte hashes. This is a
  second incomplete portability repair, not a repeated scientific failure.
  The error again occurred before RAD-DINO download, candidate generation,
  optimizer construction, validation prediction or GT access.
- Final fixture repair for a version-3 addendum: copy the compact grid-gallery
  evidence tree to a temporary test directory and convert every CSV to exact
  CRLF bytes; likewise copy the external baseline/CPM CSV inputs as CRLF.
  JSON files and scientific hashes remain unchanged. No production auditor,
  model, candidate generation, training, metric, gate or goal is modified.

### Mask-bag MIL version 3 launch after correction addendum v2

- The final fixture repair passed 10 focused historical-contract tests,
  `compileall`, and `git diff --check`. A local whole-suite attempt could not
  collect because the bundled Windows runtime lacks `torch` and `sklearn`;
  this is an already-known local environment limitation, so the unchanged
  whole-repository suite remains a fail-closed Kaggle preflight rather than a
  locally claimed pass. Repair commit
  `ea7376ba155c164ea6520d944927b42c82d1c333` is pushed.
- Correction addendum v2 was frozen before the rerun at
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v2.json`,
  SHA-256
  `5c611c993893ae0957e99e0dde5df33d73e5199dfaad4bf80aa1591aa477f6e1`,
  commit `d6c60665acf3671e099e45a3bdb82988885bf43d`. It binds the exact
  version-2 error evidence, correction-v1 hash, repair commit and expected
  version-3 full-suite result `199 passed, 1 skipped`; the base scientific
  protocol remains authoritative and unchanged.
- Kaggle accepted private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1`, version 3, at
  `2026-07-27T13:56:50Z`. Wrapper SHA-256 is
  `3f902bb9f7c26992bc8aaa00e093483266f06300e15e82d886a12688336b5c5e`;
  unchanged metadata SHA-256 is
  `3086b5fa9bc1598831c45efd48a42dc8447447b290f5bc0c48d5b300e221caed`.
  The wrapper independently verifies the base protocol plus both correction
  addenda and checks out `d6c60665...`; the scientific source remains pinned
  to `1567c2a0...`. The accelerator contract remains T4 x2, including a real
  convolution on each device and RAD-DINO encoder `DataParallel` over both.
- Exactly one five-minute heartbeat was created only after Kaggle acceptance:
  `theo-d-i-rad-dino-mask-bag-mil`. No status poll was performed after launch.
  Heavy compute remains Kaggle-only; validation GT is still inaccessible until
  prediction freeze, the BTXRD test split remains locked, and no consumer is
  authorized before the predeclared gate passes.

### Mask-bag MIL version 3 indexed-device CLI error

- Version 3 reached terminal `ERROR` after the prescribed one-check-per-five-
  minute monitoring cadence. The sole heartbeat
  `theo-d-i-rad-dino-mask-bag-mil` was deleted before logs were retrieved.
  Downloaded evidence is isolated under
  `tmp/kaggle/rad_dino_mask_bag_mil_probe_val_v1_error_v3_20260727`.
- Provenance passed exactly: wrapper SHA-256
  `3f902bb9f7c26992bc8aaa00e093483266f06300e15e82d886a12688336b5c5e`,
  base protocol `a8f3101b...`, correction v1 `cbcb28c2...`, correction v2
  `5c611c99...`, checkout commit `d6c60665...`, reconstructed split
  `85511ee1...`, and all three RAD-DINO files including physical weights
  `dbfb9f54...`. Execution-log SHA-256 is
  `206726cfd3239132708e7c653760040928b1b134826551146f62db7494f4157b`;
  a compact direct-console traceback excerpt hashes to
  `e2ec7710fc7a2078e67f392713d5a5597c5648c11cd677e6c20b1f1741ee225e`.
- Both preflights now pass: `17/17` focused tests and the exact expected whole
  suite `199 passed, 1 skipped`. Failure occurs at argument parsing of the
  first train candidate-generation command: the wrapper intentionally routes
  DenseNet/LayerCAM to `cuda:0` and SAM to `cuda:1`, but
  `generate_pseudo_masks.py` restricted each CLI option to the literal choices
  `auto`, `cpu`, or `cuda`. `argparse` rejected `cuda:0` before dataset/model
  construction with exit status 2.
- Classification: implementation-only multi-GPU device-routing interface
  error. RAD-DINO download completed, but no `train_candidates`,
  `val_candidates`, optimizer, prediction freeze, validation prediction,
  evaluation or scientific metric exists. Validation GT and BTXRD test were
  not read; no consumer was trained.
- The bounded repair accepts only `auto`, `cpu`, `cuda`, or `cuda:<nonnegative
  integer>` for the existing two device arguments, converts them through the
  already-used `torch.device`, and fails clearly if an explicit index is beyond
  `torch.cuda.device_count()`. It does not change candidate construction,
  model weights, image preprocessing, training, prediction, evaluation, gate,
  cohort or supervision semantics. A parser integration test locks the exact
  `cuda:0`/`cuda:1` route, invalid specifications are rejected, and unavailable
  indices fail closed.

### Mask-bag MIL version 4 launch after indexed-device correction

- The bounded implementation repair is commit
  `f66f585318d54ace73b6602dd895a608c0d68c79`. Correction addendum v3 was
  frozen before rerun at
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v3.json`,
  SHA-256
  `b655fb806e69138684aa26b02a255acacf91a6b2e145e26320a8275aeffdfc30`,
  commit `d8baabc0ef3cc919af847fc89a68f46b850969c0`. It authorizes one and
  only one base-source hash override, `project/generate_pseudo_masks.py` =
  `2adc57cd...`, and binds the new device-routing test `d38f9999...`.
- Kaggle accepted private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1`, version 4, at
  `2026-07-27T14:08:41Z`. Wrapper SHA-256 is
  `00a438637b98bc9541379f99263a3caa2d8245d142cd67542add721ff5699c3b`;
  metadata remains `3086b5fa...`. The expected focused/full preflights are
  `23 passed` and `205 passed, 1 skipped`; any mismatch fails before candidate
  generation. Classifier/LayerCAM remains explicitly on `cuda:0`, SAM on
  `cuda:1`, and later RAD-DINO encoding remains `DataParallel([0,1])`.
- Exactly one new five-minute heartbeat was created after version-4 acceptance:
  `theo-d-i-rad-dino-mask-bag-mil`. No launch-time status poll was performed.
  Validation GT remains inaccessible before prediction freeze, BTXRD test is
  locked, and no consumer is authorized before all prediction-first gate checks
  pass.

### Mask-bag MIL version 4 canonical-Git hash error

- Version 4 reached terminal `ERROR` on the first five-minute check. The sole
  heartbeat was deleted before evidence retrieval. Direct Kaggle console log
  and wrapper were downloaded under
  `tmp/kaggle/rad_dino_mask_bag_mil_probe_val_v1_error_v4_20260727`; their
  SHA-256 values are respectively
  `c06592460806fef523daed6d492188c59231c4074a2e388ec57844691c61e6d9`
  and `00a438637b98bc9541379f99263a3caa2d8245d142cd67542add721ff5699c3b`.
- Failure occurred 2.52 seconds after checkout, inside the source-hash guard:
  `Scientific source SHA-256 mismatch: project/generate_pseudo_masks.py`.
  No runtime installation, unit test, model download, dataset access,
  candidate generation, optimizer, prediction, GT evaluation or scientific
  metric occurred. Consumer/test flags therefore remain false.
- Root cause is a correction-provenance line-ending mistake. Correction v3
  recorded SHA-256 `2adc57cd...` from the Windows CRLF working-tree bytes.
  The immutable Git blob checked out on Kaggle is LF and hashes to
  `028ca4b8c0f1445178043bf9726c3ef4092df1ee39669938ed7f475c1bfa0ba7`
  (140,329 bytes). The device-routing test already used its correct canonical
  Git hash `d38f9999...`. The source commit and source bytes are not changed by
  this repair; only the wrapper's expected canonical hash will be corrected in
  a new immutable addendum.

### Mask-bag MIL version 5 launch after canonical-hash correction

- Correction addendum v4 was frozen at
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v4.json`,
  SHA-256
  `dab9f073db1223938ab61b1f8bc5efdc29f331ae100f2a9bce1cff4f68a5b4a3`,
  commit `9c5b2902a1638fe9b5d7fa0a2f5508753ec153be`. It records both the
  erroneous CRLF hash and corrected canonical LF hash; source and test bytes
  remain unchanged from version 4.
- Before launch, wrapper constants were independently checked against the
  exact bytes from `git show HEAD:<path>` for the base protocol, correction
  v1-v4, corrected generator and device test: all seven canonical hashes pass.
  The wrapper compiles and hashes to
  `975abd8262a57885962ccf73c7731381ab2cf20740fa42401c42f9e2390b427a`.
- Kaggle accepted version 5 of
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1` at
  `2026-07-27T14:13:41Z`. Exactly one five-minute heartbeat was created after
  acceptance: `theo-d-i-rad-dino-mask-bag-mil`; no launch-time status poll was
  performed. T4 x2 routing, image-level-only supervision, prediction-first GT
  boundary, locked BTXRD test and no-consumer-before-gate rules are unchanged.

### Mask-bag MIL version 5 keyword-only projection-call error

- Version 5 reached terminal `ERROR`; the separate five-minute monitor was
  deleted before audit and is confirmed absent. Direct compact output is
  isolated under
  `tmp/rad_dino_mask_bag_mil_v5_error_20260727-232828`. Execution-log and
  downloaded-wrapper SHA-256 are
  `f8f0c8a99404febf36a5f5cfd723fe9865a66dd8887349287b22969058fe38f1`
  and
  `975abd8262a57885962ccf73c7731381ab2cf20740fa42401c42f9e2390b427a`.
  The downloaded base protocol and correction v1-v4 exactly reproduce
  `a8f3101b...`, `cbcb28c2...`, `5c611c99...`, `b655fb80...` and
  `dab9f073...`. Reaching candidate generation proves that the wrapper's
  checkout/source/split/checkpoint guards, exact two-T4 device preflight and a
  real `324.0` convolution on each device passed. Focused and full preflights
  passed exactly `23/23` and `205 passed, 1 skipped`.
- Candidate generation completed prediction-first without GT for all `2981`
  train and `371` validation images. Frozen train candidate/pseudo hashes are
  `e43c06e...` / `0890aff5...`; validation hashes are
  `f391b7dc...` / `ef72225c...`. The independent input audit verified all
  `3352` physical candidate payloads, zero empty bags, zero nonempty ordinary
  pseudo masks for normal images, `validation_gt_read=false`,
  `consumer_trained=false`, and `test_evaluated=false`.
- The run then stopped before descriptor caching, optimizer construction,
  training, prediction freeze or GT evaluation. Root cause is a Python
  interface error: `make_seeded_random_projection` declares `input_dim` and
  `output_dim` keyword-only, but the new mask-bag runner passed both
  positionally. The traceback is
  `TypeError: make_seeded_random_projection() takes 0 positional arguments
  but 2 positional arguments ... were given`. Therefore no Dice/AUROC/gate
  result exists and this is an implementation-only failure, not evidence
  against the scientific hypothesis.
- The bounded repair changes only those two arguments to explicit keywords and
  adds an AST regression test that requires an argument-free positional list
  and the exact `input_dim`, `output_dim`, `seed` keyword set. The local
  mask-bag static suite passes `5/5`, `compileall` and diff-check pass. The
  NumPy-dependent primitive test remains deferred because the bundled local
  Python lacks NumPy; the unchanged Kaggle whole-suite preflight covers it.
  Model, candidates, frozen candidate hashes, training hyperparameters,
  prediction, evaluator, bootstrap, gate, goals, split and supervision
  semantics are unchanged. A new correction addendum must bind this repair
  before any rerun.

### Mask-bag MIL version 6 launch after projection-call correction

- Correction addendum v5 was frozen before execution at
  `artifacts/research_protocols/rad_dino_mask_bag_mil_probe_val_v1_wrapper_correction_v5.json`,
  SHA-256
  `036d4c7adadd6551367783f90377d3424dc42a3f1b974122b646018e648abd40`,
  commit `689e96616ef04a692193b5e253d0c0c91450822b`. It binds the exact
  version-5 error evidence, repair commit `c0228af...`, corrected canonical
  runner/test hashes and expected version-6 preflights `24 passed` /
  `206 passed, 1 skipped`; the base scientific protocol remains authoritative.
- Kaggle accepted private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1` version `6`. Wrapper and
  unchanged metadata SHA-256 are
  `6293b040ce25109e6c4bb167a32fcc635bfbeef38e82a23c18c28ba9d8aaf1ff`
  and
  `3086b5fa9bc1598831c45efd48a42dc8447447b290f5bc0c48d5b300e221caed`.
  The wrapper checks out `689e966...`, validates correction v1-v5, preserves
  T4 x2 routing and requires regenerated candidate manifests to exactly match
  the already frozen version-5 hashes. No launch-time status poll was made;
  validation GT, consumer training and BTXRD test remain locked.

### Mask-bag terminal decision tree and deferred cleanup audit

- A read-only source/workspace audit was performed while version 6 is
  monitored in its separate five-minute task; the main task did not poll
  Kaggle. The removable local bulk is concentrated in ignored runtime
  material: approximately `457 MB` under `tmp/kaggle`, `118 MB` under
  `tmp/literature`, Python bytecode/cache directories, and the isolated
  version-5 download. None was deleted before terminal audit because the
  staged version-6 wrapper and historical source/hash evidence still depend
  on exact files there. No tracked `.pyc` exists. Cleanup remains deferred
  until terminal evidence is secured, with frozen compact artifacts and Git
  history retained.
- Static review of the repaired mask-bag runner found no second occurrence of
  the positional/keyword-only error. Its direct project calls now match the
  declared keyword-only interfaces for split loading, model verification,
  candidate-manifest validation, square preprocessing, random projection and
  SmoothMax pooling. The local environment still lacks NumPy/Torch/Ruff, so
  this is not claimed as a dynamic GPU proof; version 6 remains fail-closed on
  its `24` focused and `206 passed, 1 skipped` Kaggle preflights.
- The post-terminal next action is frozen as a diagnosis tree, not as a new
  experiment or a way to change the current gate:
  1. If the complete proposal oracle meets the operational goals but selected
     Dice fails, the bottleneck is proposal ranking. The next admissible
     hypothesis is cross-image/normal-prototype relational scoring over the
     same frozen masks, not more unconstrained pixel expansion. Ahn and Kwak,
     *Learning Pixel-Level Semantic Affinity With Image-Level Supervision for
     Weakly Supervised Semantic Segmentation*, CVPR 2018, pp. 4981-4990,
     https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html,
     supports learning local affinity from reliable weak seeds; Seibold et
     al., *Self-Guided Multiple Instance Learning for Weakly Supervised
     Thoracic Disease Classification and Localization in Chest Radiographs*,
     ACCV 2020,
     https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html,
     supports treating uncertain positive-bag instances differently from
     reliable normal-bag negatives. For BTXRD these mechanisms must remain
     proposal-constrained and image-label-only.
  2. If the oracle itself fails, especially on small tumors, the bottleneck is
     proposal support. The next admissible mechanism is prediction-first
     high-resolution local proposal generation driven by frozen global
     evidence, following the already recorded GLAM coarse-to-fine principle,
     without GT-derived crop choice or size routing.
  3. Only a full gate pass authorizes design of a separately predeclared
     pseudo-mask consumer. Neither a high oracle nor image AUROC alone is a
     pass. Operational targets, complete misses, bootstrap, cohort and locked
     test remain unchanged.
- A further no-GT audit of the frozen version-5 candidate manifests confirms
  the selector's actual search burden before version-6 results are known.
  Train has `2,981` bags with mean/median/p90/max candidate counts
  `58.59/63/81/81`; validation has `371` bags with
  `56.51/60/81/81`. Candidate counts range from `27` to `81`. Image-label
  populations reproduce `1,493/1,488` normal/tumor in train and `187/184` in
  validation. This audit reads neither masks nor validation subgroup/GT.
  It supports retaining normalized SmoothMax, which subtracts the log of the
  valid bag count, and cautions that a high oracle with low selected Dice
  would be a genuine approximately-one-of-sixty ranking problem rather than
  proof that the proposal shapes are absent.
- A focused selector literature review is frozen at
  `artifacts/literature_reviews/mask_bag_selector_contingency_2026-07-27.md`.
  It records Ilse et al. gated-attention MIL (ICML 2018), Cinbis et al.
  multi-fold MIL against premature localization lock-in, Li et al. DSMIL
  relational critical-instance aggregation (CVPR 2021), and Seibold et al.
  uncertainty-aware radiograph MIL (ACCV 2020), with official/stable URLs and
  explicit adapt/reject decisions. If and only if version 6 proves adequate
  oracle support but poor ranking, the preferred next study is deterministic
  group-preserving train-only cross-validation of a soft attention/relational
  selector, followed by one fixed full-train model. Validation GT cannot choose
  its architecture, epoch or loss weight. If oracle support fails, the study
  is rejected in advance and proposal generation must be repaired instead.
- A no-GT shortcut audit is frozen separately from the prior selector review
  at
  `artifacts/literature_reviews/mask_bag_count_shortcut_addendum_2026-07-27.md`,
  SHA-256
  `eff23e9f0b0bd888f759b7e28f0a70e7769d2084655b6050dd4f087b5a61f6b9`.
  It adds an important interpretation constraint. Normal
  bags contain substantially more tumor-targeted proposals than tumor bags:
  train means `71.745` versus `45.399`, and validation means `63.674` versus
  `49.239`. Candidate count alone, with lower count interpreted as tumor,
  reaches direction-corrected image AUROC `0.86726993` on train and
  `0.71207277` on validation; box/positive-point/negative-point counts yield
  approximately `0.86--0.87` train and `0.70--0.71` validation AUROC.
  Diagnostic bytes alone yield `0.75822026/0.64566380`. This uses only frozen
  manifests and image labels, not validation masks or test.
- The difference is permitted image-level supervision rather than leakage:
  all bags use the tumor classifier target and normal bags are valid negative
  instances. However, it is a real bag-construction shortcut that can inflate
  image AUROC without identifying a lesion proposal. Normalized SmoothMax
  removes an exact equal-logit count offset but cannot erase distributional
  differences. Therefore version-6 image AUROC will not be interpreted as
  localization evidence on its own; the already frozen proposal-Dice,
  subgroup, CI and complete-miss checks remain decisive. Any ranking fallback
  must additionally report count-only AUROC and bag-score/count association,
  and should use predeclared count-robust training such as proposal dropout or
  normal-prototype instance contrast rather than tune against validation GT.

### Count-robust MIL and small-object contingency literature addendum

- A new literature-only addendum is frozen at
  `artifacts/literature_reviews/mask_bag_count_robust_small_object_addendum_2026-07-28.md`,
  SHA-256
  `741af83833b2492a9d6e493f3ab3a5d9f46099f4be61d2c765c62980d988bd61`.
  It does not modify the running version-6 protocol or inspect BTXRD masks/test.
  It separates two causal branches that aggregate Dice or image AUROC cannot
  distinguish: adequate proposals ranked incorrectly versus missing small
  lesion support.
- Zhu et al., *How Effective Can Dropout Be in Multiple Instance Learning?*,
  ICML 2025, PMLR 267:80090-80106,
  https://proceedings.mlr.press/v267/zhu25q.html, report that dropping top-k
  important MIL instances and feature-similar neighbours can improve
  generalization and robustness. If version 6 proves that proposal oracle
  coverage passes but selected localization fails, this motivates a
  train-image-label-only, cross-fitted attention/relational selector with
  train-time-only MIL-Dropout. It is not treated as proof of Dice improvement:
  exact dropout parameters must be predeclared, the small-lesion risk of
  dropping the sole true proposal must be controlled, and score/count
  association plus the complete localization gate remain mandatory.
- Lin et al., *Interventional Bag Multi-Instance Learning on Whole-Slide
  Pathological Images*, CVPR 2023, pp. 19830-19839,
  https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Interventional_Bag_Multi-Instance_Learning_on_Whole-Slide_Pathological_Images_CVPR_2023_paper.html,
  support treating bag-context priors as potential confounders. IBMIL is
  deferred as a second-line mechanism because it is designed for whole-slide
  pathology and adds a confounder dictionary/interventional stage; it becomes
  justified only if a simpler cross-fitted count-robust selector retains
  material score/count dependence.
- Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
  Segmentation*, WACV 2024, pp. 413-422,
  DOI 10.1109/WACV57701.2024.00048,
  https://openaccess.thecvf.com/content/WACV2024/papers/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.pdf,
  show that aggregate metrics conceal small-object WSSS failures and motivate
  size-balanced evaluation/loss. This supports retaining the explicit BTXRD
  subgroup gate; their dense loss is deferred until a pseudo-mask consumer is
  authorized and cannot use true validation sizes.
- Hwang, Oh, and Choe, *Small object matters in weakly supervised object
  localization*, Neurocomputing 648 (2025), 130494,
  DOI 10.1016/j.neucom.2025.130494,
  https://www.sciencedirect.com/science/article/pii/S092523122501166X,
  report image-label-only zoomed foreground/background consistency targeted at
  small objects. If the version-6 oracle itself fails on small tumors, the
  admissible next branch is therefore a separately frozen, prediction-driven
  high-resolution/zoom-consistency proposal probe, with identical routing for
  all images and no GT-derived crop or size selection. If oracle support is
  adequate, this branch is rejected in advance in favor of ranking repair.

### Deep WSSS research synthesis while version 6 is monitored separately

- The broader research synthesis requested during the ten-minute monitor
  intervals is recorded at
  `artifacts/literature_reviews/btxrd_wsss_deep_research_synthesis_2026-07-28.md`,
  SHA-256
  `34138778897d66b98b504fb0ed2830f55d7d5a4dd23dc2237dcbd040fcb31d7f`.
  It reviews proposal MIL, cross-fitting, radiographic position, shortcut
  control, high-resolution local/global consistency, foundation-model
  refinement and consumer robustness. It is a contingency artifact only:
  version 6, its frozen protocol, validation predictions and gate are
  unchanged; no segmentation annotation or BTXRD test sample was read.
- Source inspection exposed a concrete limitation in the running model.
  `RadDinoMaskBagMIL` scores every candidate independently; normalized
  LogSumExp is their only bag interaction. The frozen NPZ schema already stores
  `component_ids`, `prompt_modes`, and `proposal_source_ids`, but the runner
  discards them and retains only SAM score, log area, prompt-map mass coverage
  and prompt-map mean. Because candidates are produced in correlated
  component/prompt families, this explains how the measured candidate-count
  shortcut can survive exact LogSumExp count normalization. After warm-up,
  the same model also trains on its own detached positive-bag argmax, creating
  a plausible confirmation-lock-in route.
- Cinbis, Verbeek and Schmid, *Multi-fold MIL Training for Weakly Supervised
  Object Localization*, CVPR 2014,
  https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html,
  directly motivate group-preserving train-only cross-fitting so an image's
  positive instance target is generated by a selector that did not train on
  that image. Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning
  Network*, CVPR 2021,
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html,
  motivate comparing every proposal with a critical instance. Lu et al.,
  *Data-efficient and weakly supervised computational pathology on whole-slide
  images*, Nature Biomedical Engineering 5 (2021),
  https://www.nature.com/articles/s41551-020-00682-w, motivate attention plus
  conservative instance-feature constraints. These mechanisms are transferred,
  not their pathology encoders or datasets.
- Krishnamoorthy and Wiens, *Multiple Instance Learning with Absolute Position
  Information*, CHIL/PMLR 248 (2024),
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html, report that adding
  positional encoding improved standard MIL on chest radiographs from AUROC
  `0.782` to `0.799` while matching a transformer at much lower cost. A BTXRD
  transfer may add normalized candidate centroid/bounding-box geometry with
  flip equivariance, but must ablate it because heterogeneous bone anatomy can
  turn absolute position into another shortcut.
- The preferred post-terminal ranking design is now specified more precisely.
  If the proposal oracle passes but selected Dice fails, retain the exact
  gallery, RAD-DINO mask/ring descriptors and flip consistency; restore frozen
  component/prompt/source provenance; aggregate within proposal families
  before across families; use a small DSMIL-like relational head; generate
  positive-instance targets with group-preserving out-of-fold models; and add
  per-proposal statistics from the already frozen nominal-memory map as an
  orthogonal normality signal. This is distinct from the rejected hand-written
  prompt-source graph because the semantics are learned from image labels and
  the family structure controls multiplicity rather than choosing a mask by a
  fixed rule.
- Zhu et al., *How Effective Can Dropout Be in Multiple Instance Learning?*,
  ICML 2025, https://proceedings.mlr.press/v267/zhu25q.html, remains a
  conditional regularizer, not the first repair. MIL-Dropout is added only if
  cross-fitted relational learning retains winner concentration or count
  dependence; begin conservatively because dropping the sole true small-lesion
  proposal can be destructive. Du et al., *Rethinking Multiple-Instance
  Learning From Feature Space to Probability Space*, ICLR 2025,
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html,
  is a second-line option if selection drift persists. IBMIL remains deferred
  to demonstrated residual bag-context confounding.
- If oracle support fails instead, the synthesis combines Liu et al. GLAM
  (MIDL/PMLR 143, https://proceedings.mlr.press/v143/liu21b.html), Jiang et al.
  L2G (CVPR 2022,
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html),
  Wang et al. SEAM (CVPR 2020,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html),
  and Hwang et al. zoom consistency into one prediction-conditioned
  high-resolution proposal design. A fixed number of global-evidence windows
  is processed for every tumor/normal image; local proposals are mapped back
  through exact geometry and must agree across local/global and flipped views.
  A positive image label is never copied blindly to unsupported crops.
- The composition order is frozen at the conceptual level: independently prove
  proposal support, then ranking, then consumer robustness. A successful
  high-resolution proposal branch may feed the family-balanced relational
  selector; a successful selector may feed an uncertainty-aware consumer.
  Methods are not combined before their individual prediction-first evidence
  establishes complementary errors, and validation GT cannot select their
  internal hyperparameters.

### Relational mask-bag implementation-feasibility audit

- The conditional ranking design is refined into an implementation-readiness
  addendum at
  `artifacts/literature_reviews/relational_mask_bag_mil_feasibility_addendum_2026-07-28.md`,
  SHA-256
  `f6dfa2d73ecd4da5270db3543723b65f4c98b50dd207ad8a9cf01fd3359cfadd`.
  It does not change or launch version 6 and reads no segmentation annotation
  or BTXRD test data.
- The proposal hierarchy follows the frozen generator rather than treating the
  bag as unstructured. Multimask variants sharing component/prompt/source are
  normalized first, prompt modes within a component second, and components at
  bag level third; every raw proposal remains eligible for final WTA output.
  A DSMIL-like critical-instance stream then supplies learned
  candidate-to-candidate comparison. Source: Li, Li and Eliceiri, *Dual-stream
  Multiple Instance Learning Network*, CVPR 2021,
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html.
- Same-model detached argmax supervision is replaced conditionally by a
  deterministic group-preserving train-only cross-fitting contract. Each
  training image's soft instance target comes from a model fitted without its
  group, the complete OOF manifest is frozen, and one final full-train model is
  then fitted. This directly transfers the anti-lock-in mechanism of Cinbis,
  Verbeek and Schmid, *Multi-fold MIL Training for Weakly Supervised Object
  Localization*, CVPR 2014,
  https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html.
- Candidate representation retains the existing RAD-DINO mask/ring/contrast
  descriptor and adds only prediction-derived provenance/geometry plus
  proposal-pooled statistics from the hash-bound nominal-memory map. Absolute
  position is explicitly ablated and made flip-equivariant because
  Krishnamoorthy and Wiens, *Multiple Instance Learning with Absolute Position
  Information*, CHIL/PMLR 248 (2024),
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html, show radiograph
  MIL can benefit from position, while heterogeneous BTXRD anatomy creates a
  shortcut risk.
- Static capacity arithmetic confirms T4x2 feasibility. The frozen manifests
  imply about `174,657` train and `20,965` validation candidates. One fp16
  1,156-D descriptor view is approximately `431.33 MiB`; original plus flip is
  approximately `862.66 MiB`. At batch 16, 81 proposals and 256 dimensions,
  embeddings are approximately `1.27 MiB` fp32 and a four-head attention
  tensor approximately `1.60 MiB`. The relational head is below roughly one
  million trainable parameters, so RAD-DINO caching remains the dominant work;
  fold heads reuse one cache. These are arithmetic estimates, not a runtime
  claim, and heavy execution remains Kaggle-only.
- The addendum enumerates fail-closed provenance alignment, duplicate-family
  invariance, flip-coordinate mapping, OOF group isolation, complete OOF
  coverage, nominal-input hash, prediction freeze, consumer/test lock and
  cohort tests. Its go/no-go remains terminal-evidence-dependent: launch only
  if version-6 oracle passes but selected localization fails; use proposal
  repair if oracle fails; add nothing if the full gate passes.

### High-resolution proposal-only repair feasibility audit

- A separate implementation-readiness addendum for the proposal-support branch
  is recorded at
  `artifacts/literature_reviews/high_resolution_proposal_repair_feasibility_addendum_2026-07-28.md`,
  SHA-256
  `8d9ad6e76298dae16510c0b0a72a3c06396df25ccbe11005ca3e41a7bd78b20a`.
  It changes neither version 6 nor any current prediction and reads no BTXRD
  segmentation annotation/test data.
- The design explicitly incorporates the rejected global-local v4 evidence
  rather than repeating it. Its frozen global ROIs intersected `93/94` small
  tumors, but the learned local branch achieved small AP `0.00516770`, p90
  Dice `0.00507460` and argmax hit `0/94` despite median small confidence
  `0.98055` and training BCE falling from `0.64210` to `0.17340`. Therefore the
  new branch forbids a learned local pixel decoder, local bag-confidence gate
  and residual fusion.
- The conditional hypothesis is narrower: use the frozen global map only to
  select deterministic high-resolution windows, crop original radiograph
  pixels, generate class-agnostic SAM multimask candidates, map them back
  through exact inverse geometry, and append them to the old immutable
  gallery. The same window/prompt path is applied to tumor and normal images;
  random normal-only windows are rejected because they create a
  label-correlated generation shortcut.
- This transfers resolution hierarchy from Liu et al., GLAM,
  https://proceedings.mlr.press/v143/liu21b.html; local-detail transfer from
  Jiang et al., L2G,
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html;
  transform alignment from Wang et al., SEAM,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html;
  small-object zoom consistency from Hwang, Oh and Choe,
  https://doi.org/10.1016/j.neucom.2025.130494; and SAM-based weak seeding from
  Kweon and Yoon, *From SAM to CAMs*, CVPR 2024,
  https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html,
  plus Yang and Gong, *Foundation Model Assisted WSSS*, WACV 2024,
  https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html.
- Evaluation is deliberately two-stage. P1 freezes the expanded gallery before
  GT and tests raw best-single-candidate support, complete misses and 10,000
  paired bootstrap replicates; it is never a deployed result. Only a P1 pass
  can authorize the family-balanced relational selector as P2. Retaining every
  old candidate makes union-oracle Dice mathematically nondecreasing per image,
  while old payload hashes must remain exact.
- For T4x2, the branch no longer needs a classifier forward pass. The proposed
  execution replicates one hash-verified SAM checkpoint on each T4 and
  deterministically shards whole images/ROIs across devices, with real-work
  preflights and deterministic manifest merge. Heavy generation remains
  Kaggle-only. The branch is not implemented or launched before terminal v6
  evidence activates it.

## 2026-07-28 - Conditional noise-robust consumer research

- During the ten-minute mask-bag MIL monitor interval, the prior downstream
  evidence and consumer source were audited. The measured failure remains
  pseudo-label memorization: hard-mask train positive Dice reached
  `0.73817187`, while frozen validation Dice was only `0.23001987`
  (`0.07100142/0.41340304/0.32691688` small/medium/large). The same U-Net
  family reaches `0.49513170` with real masks, so repeating hard resized masks
  plus BCE/Dice is not a justified next experiment.
- A conditional design is recorded in
  `artifacts/literature_reviews/noise_robust_consumer_feasibility_addendum_2026-07-28.md`
  (canonical-LF SHA-256
  `22bb188591ab933d7a9927c8e8bc9a2e5a652dc39f281a8d463a8c3a468d843d`).
  It combines family/view consensus, partial soft labels, image-level
  SmoothMax, uncertainty-masked EMA equivariance, local affinity, and
  train-only proposal-area-balanced sampling. On tumor images, ambiguous
  pixels remain unlabeled instead of being converted into false background;
  a whole-image hard Dice loss is excluded from the first arm. High-resolution
  crops may receive aligned partial supervision and full-to-crop consistency,
  but never inherit the full-image positive label automatically.
- Primary sources and transfer boundaries are recorded in that addendum:
  Fu et al.'s image-label medical UM-CAM/Random-View Consensus
  (Pattern Recognition 2025, DOI
  `10.1016/j.patcog.2024.111204`), SEAM (CVPR 2020), Mean Teacher
  (NeurIPS 2017), uncertainty-aware Mean Teacher for medical segmentation
  (MICCAI 2019), Cross Pseudo Supervision (CVPR 2021), Pseudo-mask Matters
  (ICCV 2021), Small Objects Matters (WACV 2024), and cross-model mutual
  learning for medical segmentation (AISTATS 2024). Semi-supervised papers
  that use some real masks are treated as mechanistic inspiration only, not
  evidence that their reported gains transfer to image-label-only BTXRD.
- The staged contingency is `C1a` partial-consensus soft BCE plus image BCE,
  then `C1b` adds EMA random-view consistency; proposal-area balancing and
  aligned high-resolution crops are deferred to `C1c`. A consumer is still
  forbidden unless the prediction-first source passes every frozen gate. If
  the oracle passes but selection fails, the relational selector comes first;
  if the oracle fails, proposal support comes first. No validation mask, test
  record, new consumer implementation, or heavy local computation was used.
- A subsequent source audit found that version 6 deliberately persists only
  the WTA candidate, its mean-TTA logit, bag probability and output map. Its
  loader discards the frozen `component_ids`, `prompt_modes` and
  `proposal_source_ids`; it does not save every original/flip candidate logit
  or any train prediction. Therefore a passing v6 compact output is mechanism
  evidence, not yet a calibrated reliability teacher.
- Any authorized consumer source must first run a separately predeclared
  group-preserving multi-fold freeze: each clean-train image is scored only by
  a selector whose training excluded that group, while a distinct full-data
  selector produces validation predictions. Every candidate logit,
  provenance array, fold/checkpoint and physical map is frozen before consumer
  training. This transfers the anti-lock-in principle from Cinbis, Verbeek and
  Schmid, *Multi-fold MIL Training for Weakly Supervised Object Localization*,
  CVPR 2014,
  https://openaccess.thecvf.com/content_cvpr_2014/html/Cinbis_Multi-fold_MIL_Training_2014_CVPR_paper.html,
  and the disjoint-partition noise-memorization argument from Kim et al.,
  *CrossSplit*, ICML 2023,
  https://proceedings.mlr.press/v202/kim23a.html. Both are mechanistic
  transfers; neither reported metric is converted into a BTXRD Dice claim.

## 2026-07-28 - Image-level model selection and cross-fit compute feasibility

- A conditional compute/protocol analysis is recorded at
  `artifacts/literature_reviews/image_level_model_selection_and_crossfit_compute_addendum_2026-07-28.md`
  (canonical-LF SHA-256
  `b0da8fd8a71844a100c4add7aa41d96caa55f79a34a50a82ceb32398b2a23d7f`).
  It addresses a second leakage path: choosing a segmentation checkpoint from
  spatial validation Dice would silently exceed the image-label-only training
  claim. Choe et al., *Evaluating Weakly Supervised Object Localization
  Methods Right*, CVPR 2020,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html,
  is cited for this protocol failure mode.
- The first conditional consumer must therefore use a fixed horizon and final
  EMA checkpoint. Image-level AUROC, normal/positive separation, output area,
  transform consistency and teacher agreement are collapse diagnostics only;
  they cannot be presented as evidence that an epoch has the best spatial
  Dice. All 371 predictions are frozen before the single spatial evaluation.
- Five-fold out-of-fold train-teacher generation is practical if the frozen
  encoder runs once. For approximately `174,657` train and `20,965` validation
  candidates, the complete original/flip float16 1,156-D descriptor cache is
  about `862.65 MiB`; logits from six heads require only about `4.48 MiB`.
  The current selector has about `331,529` parameters (`~1.27 MiB` float32).
  Therefore descriptor extraction/serialization dominates, not fold-head
  training.
- The fastest admissible T4x2 plan shards immutable RAD-DINO descriptor
  extraction across two identical model replicas, then schedules three of the
  six independent fold/full heads per GPU over the shared read-only cache.
  This stage is not DDP because the heads must not synchronize. A later single
  consumer model may use two-process DDP. No implementation, Kaggle launch,
  validation mask or test access occurred.

## 2026-07-28 - Paired diagnosis of the small-only regression

- The user's cited pattern—large overall/medium/large p90 gains but a small
  decrease and small p97 `0.02912824 < 0.03`—was re-audited from the frozen
  multi-layer decoder and affinity-decoder per-image rows. The detailed
  diagnostic is recorded at
  `artifacts/literature_reviews/small_failure_paired_diagnostic_addendum_2026-07-28.md`
  (canonical-LF SHA-256
  `331f8a81107634616e7d83b3b36c46d5e58bd0bd6129462286b2e822ae51beba`).
  This is post-freeze analysis of already rejected outputs, not model
  selection or retuning; test remains untouched.
- The small decrease is structurally distributed. P90 improves on `31/94`,
  decreases on `43/94` and ties on 20; p97 improves on 21, decreases on 39
  and ties on 34. The candidate recovers ten p90 complete misses but loses
  fifteen previous overlaps, increasing misses `30 -> 35`; argmax hits change
  by four recovered versus five lost. All four within-small GT-area quartiles
  have negative mean p90 and p97 deltas, so the result is not caused only by
  the very smallest outliers.
- The median small lesion is about `96.5` pixels at `320x320` but `3.86`
  cells at the decoder's native `64x64` grid. Fixed-percentile ties add a
  second distortion: small p97 selects mean support `0.03697`, not nominal
  `0.03`, and 14/94 images exceed 4.5%; p99 selects mean `0.01915`, and 17/94
  exceed 1.5%. The frozen metrics remain valid for the rejected gate, but
  changing the tie operator or percentile after GT is forbidden.
- Sources and transfer limits are recorded: Mun et al., WACV 2024, for
  small-object size balance; Liu et al., GLAM/MIDL 2021, for global-to-local
  high-resolution medical localization; Redekop et al., MIDL 2022, for
  image-label global/patch medical MIL; and Seibold et al., ACCV 2020, for
  uncertainty-aware radiograph instances. Their numerical results are not
  converted to BTXRD Dice.
- The repair remains adaptive proposal geometry rather than a post-hoc small
  threshold: family-normalized, area-stratified, cross-fitted relational
  candidate scoring; full/zoom consistency only after the base selector; and
  original-resolution SAM proposal expansion only if frozen oracle support is
  inadequate. Anatomy-specific routing and another free local dense decoder
  are rejected. No new experiment is launched before version-6 terminal
  evidence selects the branch.

## 2026-07-28 - Predeclared consumer-entry tier

- Before any mask-bag version-6 terminal result was delivered to the main
  task, protocol `post_prediction_consumer_entry_gate_v1` was frozen at
  `artifacts/research_protocols/post_prediction_consumer_entry_gate_v1.json`
  (canonical-LF SHA-256
  `3eda9b91a367ebc1b5b97733959a8b256c4a435afcc64fefbce98c478d8a262e`).
  It does not alter the running v6 gate or automatically authorize another
  job. It prevents a barely passing prediction source from launching a dense
  consumer that would need to invent most of the remaining spatial gap.
- The all-subgroup entry tier is the arithmetic midpoint from the frozen
  current WSL consumer to the operational goal:
  `overall 0.28513013`, `small 0.12497817`,
  `medium 0.46292241`, `large 0.41031012`. This closes exactly half of each
  frozen subgroup deficit before consumer training and leaves remaining gaps
  of approximately `0.05511/0.05398/0.04952/0.08339`. It is a pragmatic risk
  boundary, not a claim that a consumer is guaranteed to recover the other
  half.
- Entry additionally requires the upstream all-checks gate, group-preserving
  out-of-fold train teacher freeze, a distinct full-data validation teacher
  frozen before GT, overall paired CI95 lower bound above zero versus the
  current WSL consumer, no subgroup mean decrease, no complete-miss increase,
  and image AUROC at least `0.75`. Candidate oracle or image AUROC alone cannot
  authorize training.
- The decision is now fixed before outcome inspection: a source already at
  the operational goals is directly reportable; an entry-tier pass may
  authorize separately predeclared C1a/C1b; oracle-pass but entry-fail returns
  to relational selection; oracle-fail returns to high-resolution proposal
  support. Sources recorded in the protocol are Li et al., *Pseudo-mask
  Matters* (ICCV 2021), Fu et al., UM-CAM/Random-View Consensus (Pattern
  Recognition 2025), and Choe et al., correct WSOL evaluation (CVPR 2020).
  Validation masks, test data, consumer training and new Kaggle compute were
  not used.

## 2026-07-28 - External medical foundation proposal-source feasibility

- While mask-bag MIL version 6 remained under the separate ten-minute
  monitor, a wider proposal-source review was recorded at
  `artifacts/literature_reviews/external_medical_foundation_proposal_feasibility_addendum_2026-07-28.md`
  (canonical-LF SHA-256
  `e5476f913e8c9cdf5add3f43d84003025d3c52423ff64860cd5fbd570188aea6`).
  This is literature analysis only; no Kaggle job, checkpoint download,
  validation-mask access or test access occurred.
- The supervision claim is bounded explicitly. A frozen MedSAM/SAM-Med2D
  proposal source that never sees a BTXRD mask is compatible with
  image-label-only **BTXRD adaptation**, but these models used external dense
  masks. Results must be disclosed as image-label-only WSSS with externally
  mask-pretrained proposal foundations and reported separately from a
  generic-SAM-only arm and fully supervised BTXRD training.
- Priority is MedSAM, then SAM-Med2D. MedSAM is an Apache-2.0 SAM ViT-B
  derivative trained on 1,570,263 image-mask pairs across ten modalities,
  including X-ray; source:
  Ma et al., *Segment Anything in Medical Images*, Nature Communications
  2024, https://www.nature.com/articles/s41467-024-44824-z, and official
  repository https://github.com/bowang-lab/MedSAM. SAM-Med2D reports 4.6
  million images/19.7 million masks, ten modalities, box/point prompting and a
  256x256 default; sources:
  https://arxiv.org/abs/2308.16184 and
  https://github.com/openmedlab/SAM-Med2D.
- BiomedParse is deferred because its text vocabulary contains general
  biomedical targets and an X-ray infection example but no established
  primary-bone-tumor target; post-GT prompt sweeping would be invalid. Sources:
  https://microsoft.github.io/BiomedParse/,
  https://github.com/microsoft/BiomedParse, and
  https://huggingface.co/microsoft/BiomedParse. FluoroSAM is lower priority
  because its three-million synthetic-X-ray pretraining targets organs/tools,
  not bone lesions; source: Seibold et al., MICCAI 2025,
  https://papers.miccai.org/miccai-2025/0344-Paper5042.html.
- Medical adaptation is not assumed to transfer. A broad SAM medical study
  reported strongly dataset-dependent performance, supporting an oracle-first
  audit rather than an expected BTXRD gain:
  https://www.sciencedirect.com/science/article/pii/S1361841523001780.
- The BTXRD Scientific Data paper is from 2025
  (https://www.nature.com/articles/s41597-024-04311-y), whereas MedSAM and
  SAM-Med2D were published/released in 2024 and 2023. This is chronological
  evidence against intentional BTXRD pretraining overlap, not proof of no
  source-image duplication. Repository revision, license, checkpoint SHA,
  available corpus manifest and any possible duplicate audit must be frozen
  before use; missing corpus provenance remains a stated limitation.
- The combined conditional branch is frozen conceptually as
  `medical proposal geometry -> provenance-preserving union -> cross-fitted
  RAD-DINO relational selector -> optional robust consumer after entry gate`.
  It is eligible only after v6 selects the oracle-fail/narrow-small-headroom
  branch. MedSAM and SAM-Med2D use identical frozen boxes and geometry;
  bit-exact duplicates only are removed; all 371 maps and proposal identities
  freeze before a single oracle audit. A positive overall result cannot rescue
  unchanged/worse small complete misses. T4x2 runs one independent proposal
  source per GPU when feasible; no DDP is used for independent models.

## 2026-07-28 - Mask-bag descriptor square-geometry defect and correction freeze

- A source audit performed while v6 was still isolated behind its separate
  ten-minute monitor found a second coordinate-frame defect. The candidate
  gallery is generated from the classification transform
  `Resize((320,320))` in `project/datasets/common.py`; SAM masks are likewise
  resampled to that direct-resize frame in `project/generate_pseudo_masks.py`.
  RAD-DINO instead consumes the original radiograph centered in a
  `max(width,height)` square and then resized to 448, through
  `_raw_and_normalized_square`. The v6 runner passed each direct-resize mask
  directly to `mask_pool_descriptors`, treating it as if it occupied the
  complete square token grid.
- This mismatch affects far more than edge cases: only 9/371 validation images
  are square, 322/371 have aspect ratio below 0.90 and 183/371 below 0.75.
  Small lesions are especially sensitive because their median support was
  already only about 3.86 cells at 64x64; v6 uses a still coarser 32x32 token
  grid. The WTA output is nevertheless an original frozen 320x320 SAM mask, so
  v6's post-freeze Dice remains a valid deployable prediction result. The
  defect invalidates a clean causal interpretation of v6 as correctly
  mask-conditioned RAD-DINO evidence; it does not authorize hiding v6.
- Implementation commit
  `3cec3bac021088a036482854a64e4619b7af384f` adds the continuous coordinate
  transform. Proposal sample centers are mapped through the exact
  `content_box/padded_side`, sampled bilinearly with zeros in square padding at
  four samples per token axis, and then area-pooled by the unchanged
  descriptor code. Flip descriptors now flip the already projected square
  mask, preserving possible one-pixel asymmetric padding. Candidate masks,
  ordering, metadata, model, losses, 16-epoch final-only rule, WTA output,
  evaluator and gates are unchanged.
- Before any corrected prediction or v6 terminal result was delivered to the
  main task, correction protocol
  `artifacts/research_protocols/rad_dino_mask_bag_mil_descriptor_geometry_correction_v1.json`
  was frozen (canonical-LF SHA-256
  `1e2fa9904583d487114e1c5b028781946ec3c94e2a10d079b536cd4a7f097ecd`).
  It allows a corrected T4x2 rerun only after the v6 audit, only if v6 has not
  already met all operational goals and the immutable proposal oracle still
  meets every goal. Corrected-minus-v6 and both arms versus baseline use the
  same complete-group paired bootstrap and unchanged all-checks gate.
- Local `py_compile` passed. Focused pytest reported `5 passed, 1 skipped`;
  the tensor geometry test module was skipped because the local Python
  environment lacks Torch, so the complete Torch test must pass on Kaggle
  before any corrected heavy run.
- Literature context recorded in the protocol: Choe et al., *Evaluating
  Weakly Supervised Object Localization Methods Right*, CVPR 2020,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html;
  Yang and Gong, *Foundation Model Assisted Weakly Supervised Semantic
  Segmentation*, WACV 2024,
  https://openaccess.thecvf.com/content/WACV2024/html/Yang_Foundation_Model_Assisted_Weakly_Supervised_Semantic_Segmentation_WACV_2024_paper.html;
  and Mun et al., *Small Objects Matters in Weakly-Supervised Semantic
  Segmentation*, WACV 2024,
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html.
  Their metrics are not transferred to BTXRD.

### Geometry-correction v2: exclude square padding from local context

- A second source-level review before any corrected heavy run found that v1
  aligned proposal support but still allowed its radius-two local-context ring
  to include black square-padding tokens. This can create an aspect/border
  shortcut and contaminate proposal-minus-context contrast, particularly for a
  small candidate near the edge of the real radiograph.
- Commit `1a6a9ae3b481b2ecc022bd0dc3b53520ca9ada83` projects an all-one source
  mask through the same content-box transform to obtain fractional real-image
  occupancy. Both proposal occupancy and its dilated context are multiplied by
  this validity map. Original/flip paths flip both projected proposal and
  content validity. A tensor test injects value `100` only in padding tokens
  and requires the inside/context means to remain at the real-content value
  `2`, with zero contrast.
- Future execution is now governed by
  `artifacts/research_protocols/rad_dino_mask_bag_mil_descriptor_geometry_correction_v2.json`
  (canonical-LF SHA-256
  `9931a349195f25b1968c2d735bec203a9fe1a5b110eccb91f2c8a51d94979756`).
  It supersedes correction v1 only for future execution; v1 remains immutable
  provenance. V2 still changes only descriptor coordinate handling relative
  to running v6. Proposals, model/loss/seed/epochs, evaluator, gates, consumer
  lock and test lock remain unchanged. Local `py_compile` and the non-Torch
  focused suite pass (`5 passed, 1 skipped`); the complete Torch suite remains
  a mandatory Kaggle preflight.

### Geometry-correction v3: preserve fractional proposal boundary once

- A final mathematical audit before any corrected execution found that v2
  multiplied the projected proposal by fractional content occupancy even
  though bilinear projection with zero padding had already applied that
  boundary support. At the image boundary a weight of `0.5` could therefore
  become `0.25`, disproportionately threatening the small candidates the
  correction is intended to preserve.
- Commit `cb2c269394959ce04377a48b51971f1304f96c05` removes this double
  attenuation. The projected proposal is used exactly once; fractional
  content validity masks only the dilated context ring. A regression test
  requires a projected boundary proposal of mass `0.5` to remain valid and
  retain weight `0.5`. The parent's minimum-grid-mass and denominator rules
  remain unchanged, so this is still a coordinate-only correction.
- Future execution is governed by final correction protocol
  `artifacts/research_protocols/rad_dino_mask_bag_mil_descriptor_geometry_correction_v3.json`
  (canonical-LF SHA-256
  `3248b37b8f60da2b3c7b6e4009f76967876bef6d498ac781e29f779a57351a91`).
  V1/v2 remain immutable provenance but are forbidden for a future job. The
  protocol also records that candidate manifests without physical NPZ
  payloads are insufficient for reuse; absent a direct hash-verifiable
  payload mount, the unchanged gallery must be regenerated.

## 2026-07-28 - Fractional mask-pooling small-object contingency

- A separate method audit is recorded at
  `artifacts/literature_reviews/fractional_mask_pooling_small_object_contingency_2026-07-28.md`
  (canonical-LF SHA-256
  `9a5c6943949becf6f40ee3057a36e82265951c7ffc0ab31d522a057f6360aee1`).
  It identifies a non-coordinate small-proposal hypothesis and deliberately
  does not modify geometry-correction v3.
- The current proposal descriptor divides its weighted token sum by
  `max(grid_mass,1)` while retaining candidates down to grid mass `0.25`.
  Thus an accepted proposal with mass `0.25/0.50/0.75` scales its apparent
  feature mean by `0.25/0.50/0.75`. This may be an intentional confidence
  penalty, but it duplicates size information already present in log-area
  metadata and may suppress semantically useful sub-token small proposals.
- The next wrapper should save a GT-blind, hash-frozen grid-mass audit before
  optimizer construction, with fixed mass bins and only image-label,
  fallback/prompt/source strata. If almost no proposal has mass below one, the
  hypothesis is rejected without training. If the fraction is material and v3
  still fails small, a separate future protocol may compare the unchanged
  floor with a true weighted mean after the same minimum-mass filter. It may
  not be bundled with relational MIL, proposal changes or a consumer.
- Sources and transfer boundaries: Shen et al., MoIPool for arbitrary-shaped
  proposal MIL, CVPR 2021,
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html;
  Ilse et al., normalized attention MIL, ICML 2018,
  https://proceedings.mlr.press/v80/ilse18a.html; Ren et al., proposal
  inner/outer contrast, CVPR 2023,
  https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html;
  and Mun et al., explicit small-object WSSS audit, WACV 2024,
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html.
  Their reported metrics and size-aware training labels are not transferred to
  BTXRD.
- Commits `ec4a773b10d9a51b8cb56977fc9e560709ae8a30` and
  `057ba9f1b819d440aaf0d3eb8fc1ac58e9740744` prepare and harden the GT-blind
  diagnostic. `project/audit_mask_bag_fractional_grid_mass.py`
  (canonical-LF SHA-256
  `aa684de20407d0934bb8c4d32f5293eac1ed56e341e28eab0ecce78fd2757c79`)
  verifies split/source/candidate/pseudo-manifest and every physical NPZ hash,
  reproduces fallback bags, applies the exact v3 projection, and saves every
  candidate mass plus fixed overall/image-label/prompt/source/fallback
  summaries. It is hard-limited to train/validation and the frozen
  `32/4/0.25/81` grid/oversampling/minimum/cap contract.
- It now saves original and square-frame-flipped mass per candidate, fails if
  retained indices differ, and requires absolute mass agreement within
  `1e-5`, matching the runner's original/flip validity invariant.
- Static boundary tests at
  `tests/test_mask_bag_fractional_grid_mass_audit.py` (canonical-LF SHA-256
  `15cd00183d6e6ef38228863a513e17b3c67a4cd84dbfc48c3e4b7312a73c77e6`)
  passed with the existing probe tests (`7 passed`); `py_compile` passed. The
  numerical path remains a mandatory Kaggle preflight. A full local pytest
  collection is unavailable because the current Python lacks NumPy and Torch
  and stops on missing-dependency imports, not source-test assertions. No
  candidate payload is currently retained locally, so no mass result was
  fabricated.

## 2026-07-28 - Geometry-v3 wrapper readiness audit

- The conditional wrapper plan is recorded at
  `artifacts/literature_reviews/mask_bag_geometry_v3_wrapper_readiness_2026-07-28.md`
  (canonical-LF SHA-256
  `265c4ca254fbf68f175781189674cdf9f67723ef6a18c5b97ec04a1dab94e3ae`).
  No wrapper was finalized, no Kaggle kernel was changed/launched and no extra
  status poll occurred.
- The audited v6 wrapper canonical-LF SHA-256 is
  `6293b040ce25109e6c4bb167a32fcc635bfbeef38e82a23c18c28ba9d8aaf1ff`.
  Its sequencing, T4x2 routing, candidate generation, prediction freeze and
  separate evaluator are retained. V3 adds only final geometry source/protocol
  binding and the GT-blind fractional-mass diagnostic before optimizer
  construction.
- A future corrected wrapper must require regenerated candidate/pseudo
  manifests to equal the four direct terminal v6 hashes. Those hashes are not
  guessed. The current compact wrapper keeps manifests/summaries but deletes
  physical NPZ payloads, so regeneration is expected unless terminal evidence
  exposes an immutable full-payload mount. Manifest-only reuse is forbidden.
- Finalization remains conditional: do not launch if v6 already meets every
  operational goal or if its immutable candidate oracle fails any goal.
  Otherwise, fill only terminal evidence hashes, run complete Torch and
  repository tests, freeze all corrected predictions, and compare
  corrected-v3/v6/baseline with complete-group bootstrap 10,000. Consumer and
  test remain locked.
- Commit `5e548fc61cdf50d7b7774e6001849b247ba0eee6` adds a post-freeze
  v3-minus-v6 comparator at `project/compare_mask_bag_evaluated_arms.py`
  (canonical-LF SHA-256
  `24c625cfc50740d9cb633906d60ae81089e3960d3eec4b3ead6f3ce89ebaffad`)
  with static boundary test SHA-256
  `1466be0ebe917e93a04914914d0f09e6f065257b95b32a4b9f9f5111a553298d`.
  It reads only two hash-bound evaluator CSVs and never imports the dataset or
  reopens GT. It fails unless image/group/subgroup/GT-area/oracle fields and
  184/94/72/18 cohorts agree, then reports Dice CI, complete misses, recovered
  misses and lost overlaps with fixed complete-group bootstrap
  `10000/20261101`. The unchanged evaluator still provides each arm versus the
  promoted baseline.
- Commit `011f479a4668dced2f943f85cb5929d850a68172` adds a numerical
  end-to-end comparator fixture using the complete 184/94/72/18 cohort,
  repeated group IDs, a known `+0.1` paired Dice delta and 12-to-2 miss
  transition. The bundled NumPy runtime reports `3 passed`; all four subgroup
  deltas, ten recovered misses and zero lost overlaps match the constructed
  truth. This verifies execution rather than only source structure and still
  does not open BTXRD GT.

## 2026-07-28 - Proposal-oracle recovery burden

- Frozen selector/oracle, consumer-entry and operational-goal values were
  combined arithmetically at
  `artifacts/literature_reviews/proposal_oracle_recovery_burden_2026-07-28.md`
  (canonical-LF SHA-256
  `b5b9c3392b07a9b1483c95d3c8c808263717cfd2227fc415489e377ff1b8ade3`).
  No per-image file, validation mask or test record was opened.
- Required current-to-oracle gap recovery for the operational goals is
  overall/small/medium/large
  `60.76% / 60.62% / 66.85% / 34.57%`; goal as a fraction of oracle is
  `83.17% / 80.34% / 86.25% / 76.92%`. Small has the narrowest absolute
  oracle headroom (`0.04379`), but medium is the hardest WTA selection burden.
- Entry-tier recovery is
  `29.34% / 12.10% / 46.76% / -2.27%`. This supports the frozen staged
  decision: geometry correction first; relational selection only after an
  oracle-pass/entry-fail; robust consumer only after entry. A small-only gain
  cannot compensate for medium failure, and subgroup identities remain
  forbidden for training/routing.
- The note cites Choe et al. (CVPR 2020) for correct WSOL evaluation, Shen et
  al. (CVPR 2021) for arbitrary-mask proposal MIL, and Li et al. (DSMIL,
  CVPR 2021) for relational critical-instance evidence. Their metrics are not
  transferred to BTXRD.

## 2026-07-28 - T4x2 candidate-generation concurrency audit

- A source/performance audit is recorded at
  `artifacts/literature_reviews/t4x2_candidate_generation_parallelism_audit_2026-07-28.md`
  (canonical-LF SHA-256
  `2a46e64b64bad3edf75dcf773508610ddfebafe2843b60d64cb72ff796bf91ae`).
  No benchmark, wrapper mutation or Kaggle launch occurred.
- Version 6 assigns classifier/LayerCAM to T4-0 and SAM to T4-1, but one
  process invokes them sequentially for each image at batch size 1. This is
  device separation, not evidence of concurrent throughput. The later
  RAD-DINO stage does genuinely shard encoder batches with DataParallel.
- If correction v3 is authorized and must regenerate payloads, the preferred
  bounded optimization is two spawned independent processes, each loading the
  full classifier+SAM stack on one T4 and processing deterministic even/odd
  image-ID shards. A frozen 32-train-image, no-GT benchmark must prove exact
  per-image and merged-manifest byte identity, no provenance/count change,
  safe memory and at least 30% steady-state speedup. Otherwise the sequential
  generator remains.
- PyTorch primary documentation is recorded for DataParallel batch sharding
  and CUDA multiprocessing:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html and
  https://docs.pytorch.org/docs/stable/notes/multiprocessing.html. The latter
  requires spawn/forkserver for CUDA and warns against CPU oversubscription.
  No speedup or utilization number is claimed without measurement.

## 2026-07-28 - Small/medium mechanism synthesis during v6 wait

- The T4x2 audit was extended with a static memory lower bound; its new
  canonical-LF SHA-256 is
  `05cbe82188489390f90412bc71e860818af71f63e29a8be2586636e8d9c9606a`.
  NVIDIA specifies 16 GB GDDR6 per T4. TorchVision reports DenseNet-121 as
  7,978,856 parameters and a 30.8 MB weight file. Meta's official SAM
  repository binds the employed ViT-B checkpoint, whose official URL
  returned `Content-Length: 375042383` bytes on a direct HEAD request.
  Serialized classifier+SAM weights are therefore well below 1 GiB per
  replica, making one full replica per T4 plausible but not proving peak
  runtime fit. Activations, CAM hooks, CUDA workspaces and allocator
  fragmentation remain unmeasured. The frozen 32-image benchmark must record
  peak allocated/reserved memory and enforce a 14 GiB peak-reserved ceiling
  per T4 before the parallel path can be promoted.
- Primary hardware/model sources:
  https://www.nvidia.com/en-us/data-center/tesla-t4/;
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html;
  https://github.com/facebookresearch/segment-anything; and
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth.
  No runtime-memory or speed claim is inferred from checkpoint size.
- A broader mechanism synthesis is frozen at
  `artifacts/literature_reviews/small_medium_wsss_combination_strategy_2026-07-28.md`
  (canonical-LF SHA-256
  `f61b58f820a3e4e4a96e26560f5fa47721f41d34385d7c559ecd969ebb96d4ec`).
  It separates the likely bottlenecks: small is most exposed to geometry,
  sub-token proposal-mass attenuation and insufficient local resolution;
  medium has the highest selector recovery burden and is primarily a
  proposal-ranking problem; large must be protected from indiscriminate mask
  expansion.
- The staged research order is now explicit: terminal v6 audit; geometry-v3
  alone; true weighted-mean pooling only if the GT-blind fractional-mass audit
  is material and v3 still misses small; global-plus-deterministic-crop
  RAD-DINO descriptors only for an isolated small failure; relational MIL for
  an oracle-pass/medium-selection failure; and confidence-aware
  multi-scale/boundary consumer only after the frozen entry gate. These
  mechanisms may not be bundled into one experiment.
- Literature and transfer boundaries recorded in the synthesis:
  Mun et al., WACV 2024, size-specific WSSS evaluation and size-balanced
  training,
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html;
  Hwang et al., Neurocomputing 2025, zoom/global-local consistency for
  small-object weak localization,
  https://doi.org/10.1016/j.neucom.2025.130494;
  Li et al., DSMIL, CVPR 2021, relational critical-instance evidence,
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_CVPR_2021_paper.html;
  Shen et al., CVPR 2021, arbitrary-mask proposal pooling,
  https://openaccess.thecvf.com/content/CVPR2021/html/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.html;
  Saeed et al., Medical Image Analysis 2025, image-level-guided tumor
  superpixels, https://pubmed.ncbi.nlm.nih.gov/39695438/;
  Gu et al., IEEE TMI 2022, local-feature aggregation for weakly supervised
  chest-radiograph segmentation,
  https://pubmed.ncbi.nlm.nih.gov/35721071/; and Zhang et al., WeCLIP,
  CVPR 2024, frozen foundation features with a lightweight decoder and
  affinity refinement,
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html.
  None of their reported natural-image/other-modality metrics, size labels or
  dense annotations are transferred to BTXRD.
- The synthesis was then extended with an audited source-boundary map and two
  direct image-label-WSSS precedents. Jiang et al., L2G, CVPR 2022, show that
  local crops can reveal finer object evidence and transfer it to a global
  view:
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html.
  Wang et al., SEAM, CVPR 2020, use transformed-view equivariance and
  context correlation:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html.
  The safe BTXRD transfer does not label every positive-image crop as
  positive, because many crops need not contain the lesion. Instead, each
  proposal-defined crop supplies only its candidate descriptor and the image
  label is applied after MIL bag aggregation. Weighted-mean normalization,
  crop-view evidence and relational selection have separate audited insertion
  points and may not silently alter the frozen v3 arm.
- Proposal-specific RAD-DINO crops were rejected as the preferred local-view
  implementation because the frozen 81-candidate cap would require up to 82
  encoder views per image before flip augmentation. The bounded alternative
  is a fixed global-plus-four-overlapping-tile bank: every proposal uses the
  local tile or equal average of symmetrically tied tiles retaining the
  greatest projected mask mass, and retains the global descriptor. This is
  exactly five views per image, 16.4x fewer than the worst-case
  proposal-specific view count.
  Tile geometry and local-validity rules remain deliberately unfrozen until a
  train-only/image-label and compute preflight; they cannot be tuned using
  subgroup Dice. The future arm must preserve gallery/order/WTA masks and
  audit square-frame coverage plus original/flip tile alignment.
- No Kaggle status poll, wrapper mutation, GT read, consumer training or new
  experiment occurred during this literature pass. The operational goal and
  all v6/v3 gates remain unchanged pending terminal evidence.

## 2026-07-28 - Conditional relational-MIL design for medium selection

- A conditional design audit is recorded at
  `artifacts/literature_reviews/relational_mil_medium_selection_design_2026-07-28.md`
  (canonical-LF SHA-256
  `456014a1d05c8d8b2ef7bfed7f37037088c1bce1926a79c99801b1295253fab1`).
  It is not an execution protocol and authorizes no job. Eligibility requires
  terminal v6 audit, completed geometry v3, oracle support for every goal and
  corrected independent selection still below entry, especially medium.
- The smallest causal arm reuses immutable v3 descriptors. It preserves the
  independent score, selects its detached critical instance, computes
  DSMIL-style normalized affinity to that critical instance, and predicts a
  zero-initialized per-candidate residual from self/critical/difference/product
  features plus normalized affinity. Final WTA logits remain instance-level;
  all existing BCE, self-guided negative/positive handling, flip consistency,
  optimizer and epochs remain unchanged.
- This design explicitly rejects bag-AUROC-only promotion. Jang and Kwon,
  NeurIPS 2024, show that deep-MIL bag learnability does not automatically
  imply instance learnability:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html.
  The critical-instance relation is transferred from Li, Li and Eliceiri,
  DSMIL, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html.
  Their WSI metrics and contrastive pipeline are not transferred.
- Required GT-blind diagnostics include candidate-count distribution,
  original/flip critical agreement, affinity entropy/effective neighbors,
  finite/order checks and proof that zero-initialized relational logits equal
  the independent logits. Post-freeze evaluation must report complete
  localization cohorts, subgroup Dice/misses, paired bootstrap and
  recovered/lost overlaps against v3. Candidate-count loss normalization and
  soft-critical selection remain separate contingencies, not bundled changes.
- No Kaggle status poll beyond the single separate-task snapshot, source
  mutation, GT read, test access or experiment launch occurred.

## 2026-07-28 - SKELEX musculoskeletal-foundation transfer audit

- A new source/provenance audit is recorded at
  `artifacts/literature_reviews/skelex_btxrd_transfer_audit_2026-07-28.md`
  (canonical-LF SHA-256
  `6acec816f1f0aab0d42b3445856fce87744ef82fbc8363d1868c35ee8c0d4056`).
  SKELEX is a ViT-MAE model pretrained without manual labels on 1,296,540
  musculoskeletal radiographs. The authors explicitly excluded public
  datasets from pretraining to avoid external-evaluation leakage. Primary
  sources:
  https://doi.org/10.1038/s41746-026-02826-9 and
  https://arxiv.org/pdf/2602.03076.
- The public Hugging Face artifact
  https://huggingface.co/skhoha/SKELEX was inspected read-only at revision
  `368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`. It exposes
  `ViTMAEForPreTraining`, ViT-Large `24x1024`, 224 input, patch 16, 75% masking
  and a 1,318,230,232-byte safetensors checkpoint with LFS SHA-256
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`.
  The model card is nearly empty and the license metadata is
  CC-BY-NC-ND-4.0, so use must not redistribute modified weights or infer an
  unsupported intended use.
- A crucial transfer limitation was identified in the paper's Methods. Its
  BTXRD anomaly-map study cropped the anatomical region containing the tumor
  to suppress text/metal/clothing artifacts and compared it with
  anatomy-matched normal regions. That tumor-containing crop is localization
  information unavailable to image-label-only WSSS and is forbidden here.
  Consequently, the published BTXRD anomaly visualization is not deployable
  evidence for our full-image pipeline.
- Two non-bundled conditional arms remain admissible. Under
  oracle-pass/selection-fail, use the frozen SKELEX encoder as a descriptor
  replacement on the immutable geometry-v3 gallery. Under oracle-fail, test
  the paper's generic ten-mask reconstruction error only on the complete
  square-padded image, with deterministic masks, no anatomy/tumor crop, exact
  inverse geometry and map freeze before GT. The latter must first audit
  artifact/border concentration, mask coverage and normal-image false
  activation without GT.
- SKELEX's 14x14 token grid is coarser than RAD-DINO's 32x32 grid, so its
  musculoskeletal semantics may help medium selection while harming small
  spatial support. Descriptor replacement, reconstruction proposals, tiling,
  relational MIL and pooling changes must remain separate causal arms.
- No checkpoint was downloaded, source/protocol was mutated, Kaggle job was
  launched, validation GT was opened or test was accessed.

## 2026-07-29 - Monitor recovery and post-v6 mechanism decision matrix

- The complete research log was reread before resuming. The repository was
  clean on branch `research-wsss-improvement` at commit `0080ac9`; the active
  operational goal remains overall/small/medium/large
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336`.
- The former separate monitor task had entered a task-level `systemError`.
  This is not evidence that Kaggle version 6 failed. Without polling Kaggle
  from the main task, the existing heartbeat automation
  `theo-d-i-rad-dino-mask-bag-mil-v6-m-i-10-ph-t` was retargeted in place to
  task `019fadba-49a7-7112-b3a0-f11481e10cd8`; its cadence remains exactly
  `10` minutes, status `ACTIVE`, and notifications remain failed-runs-only.
  The errored monitor task was archived. No duplicate automation was created.
- The post-v6 literature-bound decision matrix is recorded directly here.
  It keeps candidate support, selector quality and consumer readiness
  separate:
  - if any corrected-gallery oracle subgroup fails, use a proposal-support
    arm because no selector can recover a mask absent from its gallery;
  - if the oracle passes, the selector misses and fractional projected mass is
    materially common, test true weighted-mean descriptor pooling alone;
  - if the oracle passes and small alone remains below goal after corrected
    pooling, test the fixed global-plus-four-local-view descriptor bank;
  - if the oracle passes and medium retains selector headroom, test the
    zero-initialized relational-MIL residual alone;
  - train a confidence-aware segmentation consumer only after the frozen
    prediction-first consumer-entry gate passes.
  The current frozen current/oracle Dice are respectively
  `0.23366822/0.11152529/0.34768577/0.41545552` and
  `0.40907629/0.22274968/0.59414817/0.64182777`; medium has the largest
  selector-recovery burden while small has the least absolute oracle
  headroom. Aggregate improvement alone cannot choose a branch.
- Additional primary literature was screened rather than copied wholesale:
  Mun et al. (WACV 2024) confirms the need for size-complete WSSS evaluation,
  but its size-balanced loss is inadmissible because BTXRD size groups come
  from dense validation truth;
  Choi et al. (FSAE, WACV 2025) motivates activation/boundary expansion but is
  deferred because it can confound support and regress large masks;
  Qiu et al. (BRNF, ICCV 2025) motivates modelling dataset-level foreground
  and background feature distributions but is not a minimal candidate-ranking
  change;
  Yang and Gong (FMA-WSSS, WACV 2024) supports frozen-foundation proposal
  assistance already represented by the current SAM/RAD-DINO design; and
  Yang et al. (ExCEL, CVPR 2025) supports dense patch alignment but is a
  lower-priority domain-gap contingency than radiology/MSK encoders.
  Medical precedents from Kuang et al. (Computers in Biology and Medicine,
  2024) and Viniavskyi et al. (2020) support affinity/relation refinement and
  an ordered seed-to-relation-to-consumer pipeline.
- Primary sources:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html;
  https://openaccess.thecvf.com/content/WACV2025/html/Choi_Feature-Level_and_Spatial-Level_Activation_Expansion_for_Weakly-Supervised_Semantic_Segmentation_WACV_2025_paper.html;
  https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html;
  https://arxiv.org/abs/2312.03585;
  https://arxiv.org/abs/2503.20826;
  https://pubmed.ncbi.nlm.nih.gov/38422964/; and
  https://arxiv.org/abs/2007.00748.
  No external metric is transferred to BTXRD.
- No Kaggle status poll, wrapper/protocol mutation, validation GT read, test
  access, consumer training or new heavy experiment occurred in this pass.
  Cleanup remains deferred until terminal v6 evidence is preserved.

## 2026-07-29 - Centralized log policy and geometry-v3 readiness re-audit

- Per user direction, subsequent research notes, source citations, decisions
  and experiment/error records are written into `RESEARCH_LOG.md`. If this
  file becomes impractically long, continuation moves to
  `RESEARCH_LOG_v1.md`. Separate JSON protocols, machine-readable evidence and
  executable audit artifacts remain separate because they are reproducibility
  inputs rather than narrative logs. No new standalone literature-note file
  is created under this policy.
- Geometry-v3 readiness was re-audited without polling Kaggle. Canonical-LF
  SHA-256 values exactly match the frozen protocol:
  - protocol:
    `3248b37b8f60da2b3c7b6e4009f76967876bef6d498ac781e29f779a57351a91`;
  - model:
    `44cd6ff052a38f9e87c1d93ce71d2aad2105c4b11083354d06b835409c517407`;
  - runner:
    `1bf56d0d9bc238aafd37988bcd767352065f89a160744b3a837d68127c7e9a71`;
  - model/runner tests:
    `b563dd7c91aa47c939dce1c95db23344548799ba85af1dd53be13bf30643df0c`
    and
    `7aab3c2073dbb170c08aaeef931442f97e930c2f67e509b3a976e91fe74a0001`;
  - fractional-mass audit/test:
    `aa684de20407d0934bb8c4d32f5293eac1ed56e341e28eab0ecce78fd2757c79`
    and
    `15cd00183d6e6ef38228863a513e17b3c67a4cd84dbfc48c3e4b7312a73c77e6`;
  - evaluated-arm comparator/test:
    `24c625cfc50740d9cb633906d60ae81089e3960d3eec4b3ead6f3ce89ebaffad`
    and
    `1466be0ebe917e93a04914914d0f09e6f065257b95b32a4b9f9f5111a553298d`.
- Source inspection confirms v3 projects direct-resize proposals once into
  the exact square content box, supplies the independently projected content
  occupancy only to the context ring, flips both maps in square coordinates,
  and requires identical original/flip retained-candidate indices. This
  avoids v2's double multiplication of fractional proposal boundaries.
- The bundled local runtime has NumPy and pytest but no Torch. The GT-free
  static runner/fractional-audit tests and the numerical complete-cohort
  comparator fixture ran successfully: `10 passed in 2.08s`. Torch geometry
  tests remain mandatory in the future Kaggle preflight; their local absence
  is not reported as execution evidence.
- The geometry-v3 wrapper is still intentionally unfinalized: terminal v6
  must first supply audited candidate/pseudo-manifest hashes and prove both
  that v6 misses at least one operational goal and that its immutable-gallery
  oracle passes every goal. No new job is authorized before those conditions.

## 2026-07-29 - Dataset-free v6 compact-output auditor

- A fail-closed terminal auditor was added at
  `project/audit_rad_dino_mask_bag_mil_v6_compact_output.py` (canonical-LF
  SHA-256
  `7f4855d409dd456f7066ae79df58856b77627e9babe15ea3ed0bf8fff67313b0`).
  It uses only the direct compact Kaggle output and Python's standard library;
  it does not import Torch/NumPy, locate the BTXRD dataset, open an image or
  load a segmentation annotation.
- The auditor independently verifies:
  - exact v6 wrapper, parent protocol and correction-v1 through v5 hashes;
  - checkout/scientific-source/split/classifier/SAM/baseline/RAD-DINO hashes;
  - two real Tesla T4 devices, pinned Torch/CUDA and the recorded nontrivial
    convolution result on both devices;
  - direct candidate/pseudo manifests, 2981/371 identities, the evidence that
    all 3352 physical payload hashes were verified before training, candidate
    cap, empty normal pseudo masks and generation GT lock;
  - checkpoint/history/prediction freeze, all 371 physical prediction-map
    hashes and complete prediction identity;
  - evaluation hashes, cohort `371/184/187`, subgroup `94/72/18`, 10,000
    bootstrap replicates, complete misses, all-checks-required gate and
    consumer/test locks;
  - selected-prediction and candidate-oracle pass/fail against the operational
    goals separately. This prevents the lower consumer-entry gate from being
    mistaken for achievement of the final goal.
- Tests are at
  `tests/test_audit_rad_dino_mask_bag_mil_v6_compact_output.py`
  (canonical-LF SHA-256
  `6d406471220599ec73dc3049b6aea7b29aee6924b10c0c313172e09db5ad13c5`).
  A complete synthetic compact tree contains 2981/371 candidate rows, 371
  hashed maps and the exact `184=94/72/18` evaluation cohort. The test proves a
  valid artifact passes and a one-map byte mutation fails closed.
- The first focused test run reported `2 failed, 11 passed`. Both were
  test-fixture-only mistakes: a source-text assertion rejected the word
  `BTXRD` in the safety docstring, and the hand-entered fixture digest was
  incorrect. The assertions were corrected without weakening the auditor:
  the safety test now forbids dataset-loader/annotation-path code, and the
  byte hash uses the independently observed SHA-256. The complete rerun
  reports `14 passed in 4.62s`; `py_compile` also passed.
- This tool does not claim that deleted candidate NPZ files are present in the
  compact artifact. It verifies the direct manifests and wrapper-emitted
  pre-training physical-verification count honestly. A future geometry-v3 run
  must regenerate or immutably mount and rehash the physical payloads as its
  separate protocol already requires.
- No Kaggle status poll, validation GT read, test access, consumer training or
  heavy local computation occurred. The v6 result remains pending delivery
  from the separate ten-minute monitor.

## 2026-07-29 - Mask-bag MIL v6 terminal audit and geometry-v3 eligibility

- Kaggle kernel
  `itsthang333/btxrd-rad-dino-mask-bag-mil-probe-v1`, version `6`, completed.
  The external monitor deleted its heartbeat before retrieval. At the user's
  direction, the external monitor task was subsequently canceled and archived;
  future active-kernel monitoring will occur in this main task at ten-minute
  intervals, with no waiting messages. Because v6 is already terminal, no
  status automation is needed before the next launch.
- Direct compact evidence is summarized at
  `artifacts/kaggle/rad_dino_mask_bag_mil_probe_val_v1/version6_compact_evidence.json`.
  Its canonical-LF SHA-256 is
  `8fc23707f217e1b2df1ace8b4b17d0d683b8784c5ff57be8845f8a8cd209f5b4`.
  The fail-closed local auditor passed after verifying the exact wrapper,
  protocol/source/split/checkpoints, two real Tesla T4s, all 3352 pre-training
  candidate payload checks, 371 physical prediction maps, cohort
  `371/184/187`, subgroup `94/72/18`, complete misses, 10,000 group-bootstrap
  replicates and `consumer_trained=false`, `test_evaluated=false`.
- Retrieval exposed two implementation-only issues, neither scientific:
  - two Kaggle-CLI downloads had briefly written the same temp directory; the
    older duplicate process tree was stopped and only the paginated download
    was retained;
  - the first local auditor revision expected `probe/checkpoint.pt`, while the
    frozen runner emits `probe/rad_dino_mask_bag_mil.pt`. The real checkpoint
    SHA `28f7248939bf70cac0834e6676b5e2f7aeadc34fe60747663a8c9a3557865f9a`
    matched `prediction_freeze.json`; the auditor and synthetic test were
    corrected.
  A full 371-map rescan found exactly one hash-corrupted partial download,
  `IMG002729.npy`. The overwritten partial file's complete bad hash was not
  preserved (only prefix `e3b` was displayed), so no full digest is inferred.
  Targeted recovery produced the manifest-bound hash
  `451cb22181ad43a20fca3b291297467eb22ba2754806e18e03ad3cdf4b493003`.
  The complete auditor then passed. No result was interpreted before that pass.
- Exact selected Dice overall/small/medium/large is
  `0.21789918 / 0.07819678 / 0.34699070 / 0.43109012`; complete misses are
  `51 / 34 / 16 / 1`. All four operational goals fail. Image-level AUROC is
  `0.81338642`, so bag classification works but does not imply instance
  localization.
- The immutable candidate oracle is
  `0.40907553 / 0.22274949 / 0.59414708 / 0.64182537` and exceeds all four
  goals. Thus candidate support is adequate and the causal bottleneck is
  proposal descriptor/ranking rather than absent proposal masks.
- Versus the promoted flip-TTA selector, paired complete-group Dice deltas
  overall/small/medium/large are
  `-0.01644004 / -0.03396663 / -0.00161302 / +0.01577959`; 95% CIs are
  `[-0.05208244,+0.01796097]`,
  `[-0.08207393,+0.01232199]`,
  `[-0.05986178,+0.05523774]`, and
  `[-0.08060216,+0.14332592]`. The exact v6 configuration is rejected; it
  reduces complete misses but does not improve Dice reliably and harms small
  on mean.
- The predeclared geometry-v3 launch condition is now satisfied: v6 misses the
  operational goals while the immutable oracle passes every goal.
  Geometry-v3 remains a coordinate-only correction with unchanged gallery,
  scoring, loss, optimizer, epochs, seed, evaluator and WTA output. Consumer
  training remains forbidden.
- A supplementary GT-blind position-prior audit found 29 anatomy metadata
  values across the complete 3352 train/validation images, with 1680 missing
  anatomy values and frontal/lateral/oblique view counts
  `1932/1150/270`. Aspect ratio is below `0.90` for 2887 images and below
  `0.75` for 1749. Together with Krishnamoorthy and Wiens (CHIL/PMLR 2024),
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html, this argues
  against prioritizing absolute-position MIL: their benefit assumes common
  alignment, whereas BTXRD is multi-anatomy/multi-view and anatomy/view
  metadata cannot be used as training supervision. Explicit position remains
  a separate low-priority contingency, not bundled with geometry-v3 or
  relational MIL.
- Corrected local auditor/test canonical-LF SHA-256 values are respectively
  `efc29bb07d6ead3e687ca7348deec336e4951cf9b24351fa8fa2eeaebc1b85be`
  and
  `764099c3e59bc92eb66bde3b546fdfde8b593b4d025acd73c9b582e389b94a73`;
  its focused synthetic suite reports `4 passed in 1.76s`.
- No BTXRD test access occurred. Validation GT was read only by the frozen
  evaluator after prediction freeze. No consumer was trained and no metric,
  threshold, subgroup definition or goal was changed.

## 2026-07-29 - Selector/descriptor anti-loop research policy and geometry-v3 execution freeze

- The terminal v6 mechanism conclusion is now treated as a persistent research
  constraint, not a one-experiment guess: every immutable-gallery oracle
  subgroup exceeds its operational goal, whereas selected Dice fails all four.
  Therefore the active bottleneck remains proposal descriptor/selector. A
  failed selector ablation does not by itself authorize relabelling proposal
  support, the consumer or another stage as the main cause. Such a change
  requires new mechanism evidence, especially an oracle regression or proof
  that the gallery contract changed.
- The improvement program is a campaign rather than a loop of isolated guesses:
  1. correct the descriptor coordinate frame with geometry-v3;
  2. improve proposal representation while keeping the gallery immutable,
     including fractional pooling and normal/tumor contrast;
  3. model candidate-to-candidate and local spatial relations rather than score
     each proposal independently;
  4. control count, winner-concentration and same-model confirmation shortcuts;
  5. combine only mechanisms whose prediction-first errors are complementary.
  Each arm must report selected-minus-oracle gap, rank of the oracle proposal,
  selected/oracle Dice and misses by subgroup, count association, flip
  agreement and paired complete-group bootstrap. Image AUROC alone cannot
  promote an instance selector.
- The design draws on complementary strengths rather than copying one method.
  Ilse et al.'s attention MIL supplies permutation-invariant soft instance
  weighting; DSMIL supplies a critical-instance relational stream; CLAM
  supplies feature-space constraints around representative instances; ACMIL
  and MIL-Dropout address excessive winner concentration; SmMIL adds explicit
  local-dependency smoothing for medical-image localization; PSMIL addresses
  feature-space selection drift through probability-space alignment. Jang and
  Kwon's instance-learnability analysis is the reason bag classification
  performance is never accepted as localization evidence.
- Primary sources:
  https://proceedings.mlr.press/v80/ilse18a.html;
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html;
  https://www.nature.com/articles/s41551-020-00682-w;
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6954_ECCV_2024_paper.php;
  https://proceedings.mlr.press/v267/zhu25q.html;
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/8db9279f593652ee9bb2223b4a2c43fa-Abstract-Conference.html;
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html;
  and
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html.
  These sources motivate mechanisms only; their pathology/natural-image
  encoders, metrics and performance numbers are not transferred to BTXRD.
- Geometry-v3 is the first campaign arm because it changes the descriptor
  coordinate frame and nothing else. Execution protocol
  `artifacts/research_protocols/rad_dino_mask_bag_mil_descriptor_geometry_v3_execution_v1.json`
  (canonical-LF SHA-256
  `24a967744f63b0bd0e3b24342b04d7c43ae532e58943c257e0b568c58c4932ee`)
  binds checkout `6e0b50c`, wrapper canonical-LF SHA-256
  `216bcbd236978b2678b3b2aa8909a3eb9e56006e783a4048bfed462c8a5d806b`,
  metadata SHA-256
  `f1aef4c1e0ee32b17a2daaf0979faf5617e90d92654d7a5e61004f91ab8d24c2`,
  the exact four terminal-v6 candidate/pseudo-manifest hashes, the terminal v6
  evaluation hashes and new kernel identity
  `itsthang333/btxrd-rad-dino-mask-bag-geometry-v3`.
- The wrapper regenerates and physically verifies all 3352 candidate payloads,
  runs train/validation GT-blind fractional-grid-mass audits before optimizer
  construction, requires two real T4 devices and a nontrivial convolution on
  each, freezes 371 maps before GT evaluation, and computes v3-minus-v6 plus
  both arms versus the promoted baseline. Static wrapper/source/hash audit and
  `py_compile` pass locally. The whole Kaggle preflight is fixed at
  `219 passed, 1 skipped`, derived from the audited v6 count plus exactly 13
  newly added tests.
- Per the user's monitoring direction, the next running kernel will be observed
  only in this main task every ten minutes; RUNNING/QUEUED heartbeats produce
  `DONT_NOTIFY`. No external monitor task will be created. Consumer training
  and BTXRD test remain locked.
- Kaggle accepted private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-geometry-v3`, version `1`, using
  `NvidiaTeslaT4` (`T4 x2` required by fail-closed runtime preflight). Invalid
  free-form Kaggle tags were ignored by Kaggle, which does not affect code,
  inputs or scientific execution. The sole same-task heartbeat is
  `theo-d-i-geometry-v3-m-i-10-ph-t-trong-task-ch-nh`, active at exactly ten
  minutes with failed-run-only notifications and exact `DONT_NOTIFY` behavior
  while queued/running. No launch-time status poll or duplicate monitor was
  created.

### Selector campaign mechanism audit and ordered post-v3 arms

- A new read-only join of the terminal v6 prediction manifest and frozen
  post-freeze evaluation rows quantified why the campaign must address both
  representation and selection. On tumor images, selected-minus-oracle regret
  is `0.14455/0.24716/0.21074` for small/medium/large. Medium therefore has the
  largest pure ranking loss, while small has less oracle headroom and needs a
  descriptor that can resolve approximately one-token lesions. Candidate count
  still correlates with bag probability across all 371 images at Pearson
  `-0.48118` and Spearman `-0.48864`; within tumors, count correlates with
  complete miss at `+0.37808/+0.37607`. Within medium the count-versus-selected
  Dice Spearman correlation is `-0.34799`, and within small the
  count-versus-miss correlation is `+0.43403`. Thus normalized LogSumExp removes
  the exact log-count offset but not the learned bag-construction shortcut or
  the harder one-of-many ranking problem.
- Candidate order is not the dominant shortcut: the normalized selected-index
  median is `0.61414`, with `21.74%` in the first and `28.80%` in the last
  candidate quartile; index zero is selected in only `4.89%` of tumor bags.
  The campaign should preserve order audits but focus on feature separability,
  proposal relations and winner concentration.
- The ordered post-v3 campaign is pre-scoped as follows. These are distinct
  causal arms; none is launched before terminal geometry-v3 audit.
  1. **Fractional weighted-mean normalization.** If the GT-blind v3 audit shows
     a material retained population with projected grid mass below one, replace
     the current denominator floor of `1.0` with the exact retained fractional
     mass (epsilon only for numerical safety). This directly tests small-mask
     attenuation without changing proposals, token maps or the selector head.
  2. **Normal-prototype plus DINO-affinity representation.** Keep the corrected
     means and append train-normal-only prototype distance, within-mask token
     affinity/cohesion and across-boundary affinity contrast. Negative bags
     provide reliable instance negatives; no positive proposal is declared
     correct. DINO/ECA motivates pairwise semantic affinity, while CLAM
     motivates constraining the proposal feature space. This arm changes
     descriptors, not gallery support.
  3. **Family-balanced relational selector.** Restore immutable component,
     prompt-mode and proposal-source IDs; normalize within a proposal family
     before the bag; add a zero-initialized DSMIL-like residual comparing each
     proposal with the detached critical proposal; and add a proposal graph
     whose local edges use mask IoU/containment and normalized centroid
     distance rather than absolute anatomy. DSMIL supplies global critical
     relations and SmMIL supplies localization-oriented local dependency.
  4. **Confirmation/count robustness.** Generate positive-instance soft targets
     by group-preserving train-only out-of-fold selectors instead of the same
     model's detached argmax. If winner concentration/count association remains,
     add one conservative stochastic top-k/family dropout schedule based on
     ACMIL/MIL-Dropout. It is training-only and may not drop the sole family or
     sole retained small proposal.
  5. **Complementary composition.** Combine the best representation and
     relational selector only if their frozen per-image corrections are
     complementary; add robustness only if its own mechanism gate passes.
     There is no unbounded architecture search and no switch to consumer or
     proposal generation while the immutable oracle remains above every goal.
- Every future selector arm must freeze additional GT-blind outputs before
  evaluation: all candidate logits, proposal family/geometry, attention or
  affinity entropy, effective candidate count, original/flip ranks, score/count
  association and prediction maps. The post-freeze evaluator must add oracle
  candidate index, oracle rank under the selector, top-`1/3/5/10` oracle reach,
  selected-to-oracle regret, recovered/lost misses and score-versus-candidate
  Dice rank correlation for the complete `184/94/72/18` cohort. These
  diagnostics may explain a result but cannot tune the already frozen arm.
- Primary affinity source added for the representation arm: Wu et al.,
  *DINO is Also a Semantic Guider: Exploiting Class-aware Affinity for Weakly
  Supervised Semantic Segmentation*, ACM Multimedia 2024,
  https://openreview.net/forum?id=qipYQAcvVG. The transferable idea is
  class-agnostic DINO affinity seeded/refined by weak evidence; its natural
  image benchmark, CAM architecture and metric are not transferred. No
  validation mask, BTXRD test sample or new prediction was read for this
  design.

### Technique-strength audit and reusable selector-development cache

- The literature mechanisms are not interchangeable, and their failure modes
  are now explicit before implementation:
  - exact fractional normalization can restore the amplitude of sub-token
    proposals, but it can amplify a noisy `0.25`-mass candidate; it is tested
    alone and remains protected by the existing minimum-mass threshold;
  - a train-normal prototype uses genuinely reliable negative instances and
    directly attacks tumor-versus-normal separability, but normal bags contain
    more proposals. Prototype estimation must therefore give equal total weight
    to every normal image and cap every proposal family, rather than allowing
    large bags to dominate;
  - DINO affinity captures pairwise semantic cohesion that the current
    inside/outside means discard, but class-agnostic affinity may strongly
    connect healthy bone. It is appended as an instance feature and cannot
    expand or replace the frozen candidate mask;
  - DSMIL's critical-instance relation can recover medium candidates that are
    individually ambiguous, but an incorrect early critical instance can
    spread error. Its residual is zero-initialized and the independent v3 logit
    remains an explicit skip connection;
  - SmMIL directly optimizes localization-relevant smooth instance scores and
    can complement global relations. Its original “neighboring patches share a
    label” assumption does not automatically hold for alternative SAM masks.
    The BTXRD graph may connect only same-component/source proposals with
    sufficient mask overlap/containment; the normalized-Laplacian smoothing
    acts on candidate logits themselves, not merely on a bag embedding;
  - ACMIL/MIL-Dropout can break winner concentration, but their WSI setting has
    many truly positive patches. BTXRD can have one useful small-tumor proposal,
    so top-k masking is forbidden until measured within-family redundancy is
    adequate, and it must preserve at least one proposal in every family;
  - PSMIL's probability-space alignment is relevant to the observed
    same-model selection drift, but its paper also reports benchmark code where
    the untrimmed input contains an instance-label feature and provides a
    trimmed sensitivity version. No headline performance or label-bearing
    feature is transferable. Only augmentation-consistent probabilities from
    out-of-fold image-label-only models are admissible here.
- Primary details supporting those limits:
  SmMIL derives a fidelity-preserving normalized-Laplacian smoother from
  Dirichlet energy and applies it directly to attention/localization values:
  https://proceedings.neurips.cc/paper_files/paper/2024/file/8db9279f593652ee9bb2223b4a2c43fa-Paper-Conference.pdf.
  ACMIL uses multiple attention branches plus stochastic top-k masking to
  counter attention concentration:
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06954.pdf.
  PSMIL moves attention/alignment into probability space but documents both
  untrimmed label-feature and trimmed benchmark variants:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf.
- To make the campaign broad enough without paying for candidate generation
  and RAD-DINO extraction for every selector, the first post-v3 infrastructure
  job will be a reusable **GT-blind selector-development cache**, launched only
  after terminal v3 audit. It will mechanically reproduce the exact four
  candidate-manifest hashes and corrected v3 descriptors, then store:
  original/flip fp16 descriptors, retained indices, metadata, component/prompt/
  source IDs, normalized proposal geometry, family IDs, sparse proposal-graph
  edges and bit-packed validation WTA masks. Train masks are discarded after
  graph construction. The cache may read radiographs and image labels but no
  segmentation annotation.
- The cache is valid only if the frozen v3 checkpoint rescored from it
  reproduces all 371 v3 selected indices, logits within a predeclared numerical
  tolerance, and exact final map hashes. This makes subsequent selector arms
  inexpensive and causally comparable while preserving the immutable gallery.
  Each arm receives an independent prediction directory and protocol; one arm
  cannot overwrite another.
- The relational family is further separated for causal evidence:
  `family-balanced SmoothMax`, `DSMIL-like global critical relation`, and
  `SmMIL-like local proposal-graph smoothing` are three individual arms. Their
  combinations are authorized only after individual frozen maps show
  complementary recoveries. The same rule separates normal-prototype and DINO
  affinity descriptors before combining them. This yields a finite campaign,
  not an unbounded hyperparameter search, while ensuring a failure of one
  technique does not terminate work on the established selector bottleneck.

### Relational-selector primitive preparation while geometry-v3 runs

- Lightweight, dataset-agnostic primitives were prepared at
  `project/models/mask_bag_relational_selector.py`, canonical-LF SHA-256
  `3cae8adbdf0a11384a891926570069ceefb0cda2de6dc9b5476ae19a1f17f790`.
  This is source readiness only: it is not imported by the running v3 pipeline,
  does not alter its frozen wrapper/checkout and authorizes no additional job.
- The module implements three deliberately separable mechanisms:
  - hierarchical normalized SmoothMax, first within immutable proposal family
    and then across families, making an identical within-family duplicate
    neutral to the bag score;
  - a symmetric local graph that connects only same-family proposals satisfying
    fixed mask IoU or containment, followed by fidelity-preserving normalized
    graph smoothing directly on candidate logits;
  - a DSMIL-inspired critical-proposal relation residual with a detached
    independent critical index and exactly zero-initialized final layer, so its
    initial logits equal the independent selector bit-for-bit.
  The code accepts only descriptors, masks, validity and family IDs; it has no
  dataset, annotation, lesion-size, subgroup or GT API.
- Tests at `tests/test_mask_bag_relational_selector.py`, canonical-LF SHA-256
  `b3bb46a7e37562c2bc780e10480f1bcc358072adf9ac0037497a2fb06b30e1d8`,
  verify family-duplicate invariance, cross-family edge exclusion, isolated-node
  fidelity, decreased connected contrast, `alpha=0` identity, critical-index
  selection and zero-residual identity. Local `py_compile` passed; the local
  no-Torch runtime reports `1 passed, 4 skipped`. All four numerical Torch
  tests remain mandatory on Kaggle before any arm can execute.
- The primitives are intentionally not combined into one opaque model. The
  terminal v3 audit will determine the fixed baseline cache, after which
  family balance, global critical relation and local smoothing receive separate
  prediction freezes before any complementary combination. No Kaggle status
  poll, validation GT read, test access or consumer training occurred.

### Radiograph-specific selector research and persistent-bottleneck commitment

- The selector conclusion is now a **campaign-level falsifiable hypothesis**,
  not a label that can be changed after one or two negative experiments. Under
  the immutable v6 gallery, every subgroup oracle passes its operational goal
  while the learned selection fails it. Consequently, all post-v3 work remains
  inside descriptor/selector improvement until either (a) the operational goals
  pass, or (b) the finite representation, relation and confirmation campaign
  below is exhausted. Reassigning the main bottleneck requires new gallery
  evidence, such as an oracle falling below a goal after a contractually valid
  gallery change; a failed selector arm alone is insufficient.
- Seibold et al.'s *Self-Guided Multiple Instance Learning for Weakly
  Supervised Disease Classification and Localization in Chest Radiographs*
  trains with both image and patch predictions and deliberately avoids making
  every uncertain patch a hard binary target. Its transferable lesson is that
  the current same-model detached winner is an unsafe teacher: initialization
  errors and uncertain small findings can be reinforced. The admissible BTXRD
  adaptation is a soft target produced by group-preserving out-of-fold
  selectors and original/flip agreement. The model generating a target may not
  have trained on that image. Source:
  https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html.
- Liu et al.'s *Iterative Self-Paced Supervised Contrastive Learning* improves
  MIL instance representations by admitting only sufficiently reliable
  pseudo-labelled instances at each stage and reports both bag- and
  instance-level evaluation on medical datasets. This supports a late
  **cross-fitted self-paced contrastive** arm, not immediate hard pseudo-mask
  training. All candidates from image-level-normal training bags are reliable
  negatives; positive candidates enter the contrastive set only when an
  out-of-fold selector is confident and original/flip ranks agree. The
  curriculum starts with the most reliable subset and never treats all
  candidates in a positive bag as tumor. Its weakness is confirmation bias, so
  it follows rather than precedes independently tested prototype and relational
  arms. Source:
  https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html.
- Yang et al.'s *Trainable Prototype Enhanced Multiple Instance Learning*
  softly assigns all instances to prototypes instead of refining only a few
  selected patches and uses prototype distance as an alternative to raw
  attention for instance interpretation. The BTXRD transfer will be more
  conservative: fit a **normal-only multi-prototype bank** from corrected
  candidate descriptors using clean-train image labels, give every normal
  image equal total weight, every family within an image equal weight and every
  candidate within a family equal weight, then append nearest/soft-min normal
  distance, assignment entropy and prototype margin to each candidate. This is
  not the rejected nominal pixel-memory/INSIGHT map: it neither generates a new
  dense map nor changes candidate support; it changes only the representation
  used to rank the already frozen gallery. Source:
  https://proceedings.mlr.press/v227/yang24d.html.
- Qi et al.'s GREN shows that intra-image and inter-image relationships can
  improve weak X-ray localization. The inter-image comparison motivates a
  normal prototype bank, but its pretrained lung-lobe U-Net and lobe graph are
  not transferable to BTXRD's many anatomies without auxiliary localization
  supervision. Only descriptor-space cross-image relation is retained; no lung
  mask, anatomical segmenter or extra annotation is introduced. Source:
  https://pubmed.ncbi.nlm.nih.gov/35895637/ and DOI
  `10.1109/JBHI.2022.3193108`.
- Krishnamoorthy and Wiens show that absolute position can help MIL when
  medical images share a stable global alignment. BTXRD does not meet that
  premise: it contains many bone anatomies and views, highly variable aspect
  ratios and incomplete anatomy metadata. Therefore absolute coordinates or a
  chest-specific positional prior will not be a primary arm. The relational
  selector may use only normalized candidate size, centroid difference and
  overlap within an image, with no anatomy label. Source:
  https://proceedings.mlr.press/v248/krishnamoorthy24a.html.
- Choe et al. formalize why localization from image labels alone is
  underdetermined and why localization evaluation must be separated from
  classification/model fitting. This reinforces the existing prediction-first
  contract: validation masks remain unavailable to descriptor fitting,
  prototype fitting, selector training, arm choice and prediction freezing;
  they are read only by the frozen evaluator. Source:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html.

### Finite selector resolution matrix after geometry-v3

- The campaign is explicitly partitioned so a negative result identifies which
  selector submechanism failed without changing the main bottleneck:
  1. **R0 geometry/fractional pooling:** finish v3; if its GT-blind mass audit
     activates the predeclared condition, test exact fractional weighted means.
  2. **R1 normal representation:** candidate-level normal multi-prototype
     features alone. A finite train-only choice `K in {8,16,32}` is made by
     group-preserving out-of-fold image-label loss, with count association as a
     diagnostic and the smaller `K` winning a tie. Validation masks cannot
     select `K`.
  3. **R2 local representation:** DINO within-mask cohesion and across-boundary
     affinity contrast alone. It cannot expand, erode or introduce a proposal.
  4. **S1/S2/S3 selection:** family-balanced SmoothMax, zero-initialized
     critical-relation residual, and same-family overlap-graph logit smoothing
     are frozen as separate arms against the identical descriptor cache.
  5. **S4 proposal-cluster refinement:** a group-excluded OOF teacher seeds
     mask-overlap/containment clusters, which are optimized as normalized small
     bags with a smooth-to-sharp continuation schedule. Tumor-bag candidates
     outside a cluster remain unlabeled.
  6. **T1 confirmation:** out-of-fold original/flip soft targets followed by a
     conservative ItS2CLR-style self-paced contrastive stage. This arm is
     allowed only after its target producer and consumer folds are disjoint and
     every candidate target/provenance is serialized.
  7. **C1 composition:** combine the best representation arm with the best
     relational arm only when their frozen per-image recovered/lost cases show
     complementary corrections. Add T1 only if it independently reduces
     selected-to-oracle regret without increasing count dependence.
- Each arm keeps the same gallery, candidate maps, validation cohort, evaluator
  and goals. The mechanism gate is predeclared as: reduce complete-group
  selected-to-oracle regret in at least two tumor subgroups, do not worsen
  overall selected Dice, and do not increase the absolute count/miss
  association; promotion still requires the already frozen paired bootstrap
  and subgroup reporting. Failing this gate rejects that mechanism but advances
  to the next row of this matrix. It does not authorize switching to a new
  proposal generator, dense support model or consumer.
- The reusable post-v3 cache is what makes this matrix practical: RAD-DINO
  extraction and candidate construction occur once on T4x2, while the
  lightweight selector arms reuse hash-bound descriptors and masks. This
  supports enough controlled selector trials to resolve the observed cause
  without repeated expensive end-to-end jobs or an unbounded hyperparameter
  search.
- These additions are research/protocol preparation only. The running
  geometry-v3 kernel was not polled or modified, no validation segmentation
  annotation or BTXRD test sample was read, and no consumer was trained.

### Normal-prototype descriptor primitive preparation

- A dataset-agnostic source primitive was added at
  `project/models/mask_bag_normal_prototypes.py`, canonical-LF SHA-256
  `d20be292be6a4828f337c6f3348998e56d7c51245bb4e6d8a2d193b5e914c876`.
  It implements:
  - hierarchical weights with equal total mass per image, then per family,
    then per candidate, preventing large normal bags or duplicated proposals
    from dominating the bank;
  - seeded weighted spherical k-means for a finite, hashable prototype bank;
  - four candidate features: nearest normal-prototype cosine distance,
    temperature-softened distance, assignment entropy and top-two prototype
    margin.
  The module accepts only descriptor arrays and opaque image/family IDs. It has
  no dataset, image-label lookup, GT, subgroup or evaluation interface.
- Tests at `tests/test_mask_bag_normal_prototypes.py`, canonical-LF SHA-256
  `076827d1f7ff34f4cd95c75307b435381fc4bdf83aaf333ad77fd56c0b75a713`,
  check source isolation, equal image/family weights, invariance of the
  weighted mean to an identical within-family duplicate, deterministic
  normalized prototypes and increased normal-prototype distance for an unseen
  direction. `py_compile` passes. The default local Python lacks NumPy and the
  bundled dependency Python lacks pytest, so the five test functions were
  executed directly under the bundled NumPy runtime and all five passed.
- This is readiness code only. It is not imported by geometry-v3 and has not
  fitted a BTXRD prototype. After the terminal v3 cache is frozen, the caller
  must prove that its inputs contain clean-train normal bags selected only by
  image labels, serialize the hierarchical weights/prototypes/assignments and
  bind their hashes before producing validation logits. No Kaggle status poll,
  validation GT access, test access or consumer training occurred.

### Selector-cache provenance and geometry preparation

- A read-only source audit of `run_rad_dino_mask_bag_mil_probe.py` and the
  candidate-diagnostic schema found that the runner currently retains masks,
  four numeric metadata values and kept indices, but discards already frozen
  `component_ids`, `prompt_modes` and `proposal_source_ids`. Those arrays are
  present in every schema-v2 candidate NPZ and are integrity-covered by the
  existing per-payload and manifest hashes. Therefore family-balanced and
  proposal-graph arms need neither a new gallery nor new proposal generation:
  the post-v3 cache can restore provenance mechanically from the exact
  candidate payloads and slice it with the same `kept_indices`.
- Dataset-agnostic cache helpers were added at
  `project/models/mask_bag_selector_cache.py`, canonical-LF SHA-256
  `f13984043953c61befe658f87ae2859f4bdb7c9e9f3155ec0e5f25c1034ac1a5`.
  They define a family as the exact immutable tuple
  `(proposal_source_id, prompt_mode, component_id)`, with a separate explicit
  fallback family; IDs are assigned from sorted unique tuples and cannot depend
  on candidate order. The same module derives position-free proposal shape
  features, pairwise IoU/containment/normalized centroid distance and an exact
  bit-packed mask representation. Absolute centroid coordinates are not
  exposed as candidate features.
- Tests at `tests/test_mask_bag_selector_cache.py`, canonical-LF SHA-256
  `8e57b638cca17a0e282464ce8be0b9c83c795d678ef99d7fab69e2da514bb3f5`,
  verify source isolation, all-three-field family identity, same-family
  duplicates, position-free shape features, symmetric overlap geometry and
  exact bit-pack round trip. `py_compile` and all six numerical/source tests
  pass under the bundled NumPy runtime.
- The eventual cache acceptance gate remains stricter than these unit tests:
  it must bind the terminal v3 source/protocol/candidate hashes, cover exactly
  all train/validation rows, and rescore the frozen v3 checkpoint to reproduce
  every validation selected index, logit tolerance and map hash before any new
  selector is trained. This preparation did not poll or modify geometry-v3,
  load validation GT/test, fit a prototype or train a consumer.

### Group-preserving cross-fit preparation for confirmation control

- The present mask-bag runner confirms a positive instance with the argmax of
  the same selector being optimized. This is precisely the confirmation path
  identified by the radiograph self-guided-MIL and ItS2CLR audit. A source-only
  cross-fit primitive was added at `project/models/mask_bag_crossfit.py`,
  canonical-LF SHA-256
  `4f7b27d98250c00db5ed42bbd707d02bb2ae81e7d19beed74070189228e52bd6`.
  It assigns complete `group_id` units to folds while greedily balancing image
  counts separately for image-label normal/tumor, creates an order-independent
  assignment hash summary and fails closed if a target-producing fold
  checkpoint reports any held-out/training group overlap.
- A label-only read of the frozen training split found `2981` images in `984`
  heuristic groups, with no mixed tumor label in a group. Normal has
  `1493` images in `203` groups and tumor has `1488` images in `781` groups;
  balancing group counts would therefore badly imbalance images. The
  predeclared five-fold, seed-42 row-balanced assignment produces:
  `(596,298,298,196)`, `(596,298,298,196)`,
  `(596,299,297,197)`, `(596,299,297,197)` and
  `(597,299,298,198)`, where each tuple is
  `(images,normal,tumor,groups)`. Thus the requested OOF separation is feasible
  without changing the frozen outer train/validation split.
- Tests at `tests/test_mask_bag_crossfit.py`, canonical-LF SHA-256
  `ec2c026882fe565dc38e60fccb6e464865e206a7c2f57c82ab4b0a633b6536f1`,
  verify source isolation, deterministic group integrity, per-class image
  balance, order-independent manifest hashing and rejection of training/
  held-out overlap. `py_compile` and all five numerical/source tests pass.
- Only the fold/provenance layer is frozen now. Candidate soft-target
  confidence and curriculum thresholds are intentionally not chosen before the
  terminal v3 cache exposes GT-blind original/flip score distributions.
  Whatever train-only rule is later predeclared must serialize every admitted
  target, weight, source checkpoint, excluded fold and original/flip rank.
  Validation masks cannot select that rule. No Kaggle status poll, validation
  GT/test access, prototype fit, selector training or consumer training
  occurred.

### Proposal-ranking research from weakly supervised object detection

- BTXRD mask-bag selection is also structurally a weakly supervised object
  detection problem: an image contains a frozen set of region proposals and
  only an image class is available for learning which region is positive. This
  literature supplies mechanisms more directly matched to the observed
  selected-versus-oracle gap than another dense CAM generator.
- Bilen and Vedaldi's WSDDN jointly learns region classification and region
  selection streams from image labels. Its strength is end-to-end proposal
  scoring, but its documented failure modes include selecting discriminative
  object parts and merging instances. A raw detection-softmax stream would
  also reintroduce proposal-count competition already observed in BTXRD, so
  WSDDN is not copied as a new primary arm. The transferable element is the
  separation between candidate evidence and within-bag selection, which is
  already represented by descriptor arms plus family-balanced pooling. Source:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.html.
- Tang et al.'s OICR refines an instance classifier by propagating the inferred
  label of a top proposal to spatially overlapping proposals. This directly
  addresses proposal ranking, but its online preceding-stream teacher can
  reinforce a wrong early winner--the exact same-model confirmation path seen
  in the current runner. The admissible transfer therefore forbids online
  same-model labels: only a group-excluded OOF teacher with original/flip
  agreement may seed an overlap cluster. Source:
  https://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_Instance_Detection_CVPR_2017_paper.html.
- Tang et al.'s PCL improves OICR by grouping spatially adjacent proposals
  associated with one hypothesized object and treating a proposal cluster as a
  smaller bag, reducing the ambiguity of assigning hard labels to every
  proposal. This is highly compatible with the immutable BTXRD gallery and
  complements, rather than replaces, DSMIL: DSMIL relates candidates in
  descriptor space, whereas PCL restricts the positive ambiguity spatially.
  Its weakness remains dependence on the initial cluster center and its natural
  image box-overlap assumptions. The BTXRD transfer will use mask IoU or
  containment, retain the seed candidate even when isolated, and never label
  every out-of-cluster candidate in a tumor image as background. Source:
  https://arxiv.org/abs/1807.03342 and DOI
  `10.1109/TPAMI.2018.2876304`.
- Wan et al.'s C-MIL views early hard instance selection as a non-convex local
  minimum and optimizes a sequence of smoother losses over spatially and
  class-related proposal subsets. This supports a continuation schedule inside
  the PCL-like arm: begin with normalized soft pooling over the OOF-seeded
  overlap cluster and sharpen only after original/flip cluster stability,
  rather than introduce a fixed hard-positive threshold at epoch one. It is not
  a separate unconstrained architecture search. Source:
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Wan_C-MIL_Continuation_Multiple_Instance_Learning_for_Weakly_Supervised_Object_Detection_CVPR_2019_paper.pdf.
- Kosugi et al. show that naive top-proposal propagation can select only an
  object part and can incorrectly label other objects as negative; their
  context-based and spatially restricted labeling improves OICR. For BTXRD,
  this justifies preserving uncertain tumor-bag candidates instead of treating
  them as background. A proposal-versus-context completeness descriptor is a
  bounded contingency, but no natural-image box context loss or unverified
  claim of complete tumor coverage is transferred. Source:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Kosugi_Object-Aware_Instance_Labeling_for_Weakly_Supervised_Object_Detection_ICCV_2019_paper.html.
- The selector matrix gains one causal row after the independent relational
  arms and before general self-paced contrastive learning:
  **S4 OOF proposal-cluster refinement**. A fold-excluded teacher seeds at most
  the train-only predeclared number of clusters; membership uses frozen mask
  overlap/containment and teacher-view agreement; each cluster is a normalized
  small bag with a smooth-to-sharp continuation schedule. S4 must be tested
  alone on the same cache. Outside-cluster candidates in positive bags remain
  unlabeled. Only if S4 and a descriptor/relational arm show complementary
  frozen recoveries may they be combined.
- S4 remains subject to the same mechanism gate: selected-to-oracle regret must
  fall in at least two tumor subgroups, overall selected Dice cannot regress,
  and absolute count/miss association cannot increase. A failure rejects S4
  but leaves descriptor/selector as the active bottleneck and advances to the
  next finite row. No proposal, prediction, validation GT/test sample or
  consumer was produced during this literature audit, and geometry-v3 was not
  polled or changed.

### OOF proposal-cluster primitive preparation

- Dataset-agnostic S4 primitives were added at
  `project/models/mask_bag_proposal_clusters.py`, canonical-LF SHA-256
  `3fe2cab3054156256c03f1887fb3d38c9daf74dc7dc631192afd228c1a637721`.
  Given externally audited OOF teacher logits, validity and a frozen symmetric
  overlap matrix, the module greedily seeds a bounded number of disjoint
  overlap clusters, pools candidates with normalized SmoothMax inside each
  small bag and then across clusters, and supplies a geometric
  smooth-to-sharp temperature schedule. Every isolated seed remains a valid
  singleton cluster; no out-of-cluster negative label is generated.
- Tests at `tests/test_mask_bag_proposal_clusters.py`, canonical-LF SHA-256
  `b523bb0a6ce76c8590e86ed0319aef2ec9646dbb7476f7e1382744f315fe8f2d`,
  specify score/overlap seed behavior, invariance to an identical member
  duplicate, monotonic continuation and a GT/subgroup-free API.
  `py_compile` passes; the local no-Torch runtime reports
  `1 passed, 3 skipped`. These three numerical Torch tests are mandatory on
  Kaggle before S4 may run.
- The primitive deliberately cannot assert that teacher logits are cross-fitted
  by itself. The future runner must first pass
  `audit_crossfit_training_exclusion`, bind every teacher checkpoint and fold
  manifest, and serialize cluster seeds/members before optimizer construction.
  It is not imported by geometry-v3 and no BTXRD prototype, selector,
  validation prediction or consumer was trained here.

### Flip-equivariant global-plus-four-tile descriptor preparation

- Small remains a selector/descriptor case rather than a support case under
  the immutable oracle, but its proposal can occupy approximately one or fewer
  cells on the global `32x32` RAD-DINO grid. L2G reports that local crops expose
  finer object details than a full image, while SEAM shows that weak
  localization should remain equivariant under transformations. These
  mechanisms support a bounded higher-resolution descriptor arm, not a new
  proposal map. Primary sources:
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html
  and
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html.
  Mun et al. independently document that aggregate WSSS evaluation obscures
  poor small-object behavior:
  https://openaccess.thecvf.com/content/WACV2024/papers/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.pdf.
- A geometry-only source primitive was added at
  `project/models/mask_bag_multiview.py`, canonical-LF SHA-256
  `211d0a365056ab7b7af52454ecc342f88a6f2d71cc99ba4214a75aa0b57225db`.
  It creates four overlapping corner rectangles in the unpadded source-image
  frame, measures retained candidate mass in every tile, averages all tiles
  tied for maximum retention and verifies the exact horizontal-flip tile
  permutation. Per-tile descriptors are fused with those weights; the global
  descriptor and WTA candidate masks remain unchanged.
- The equal-tie average corrects an issue in the earlier prose design.
  Selecting the lowest-index tile in a tie is deterministic but not
  flip-equivariant for a centered candidate: the flipped candidate can again
  choose the lowest index instead of the mirrored view. Equal averaging across
  all maximal-retention tiles preserves the left/right permutation exactly and
  costs no additional encoder views because all four tiles are already
  encoded.
- The crop fraction is deliberately not selected yet. After the terminal v3
  cache is accepted, a finite GT-blind train-candidate audit will compare
  `{0.625,0.75,0.875}` and choose the smallest fraction satisfying both:
  at least `99%` of all retained training candidates keep at least `50%` of
  their mask mass in their best tile, and at least `99%` of training
  candidates whose own class-agnostic area is at most `1%` keep at least
  `90%`. If none passes, the local-view arm is rejected rather than tuning
  against validation Dice. Each rectangular tile is independently
  square-padded before the frozen encoder; its mask uses the exact same
  continuous content-box projection as geometry-v3.
- Tests at `tests/test_mask_bag_multiview.py`, canonical-LF SHA-256
  `714f41f4ce02ce854b94f6ad1c79fc909e16accbd314413700abd2612e17b992`,
  verify complete rectangular coverage, corner and centered mass behavior,
  horizontal-flip equivariance of retention/weights, symmetric descriptor
  fusion and a GT/subgroup-free API. `py_compile` and all five numerical/source
  tests pass under the bundled NumPy runtime.
- This remains a separate descriptor arm: it may not be bundled with
  fractional normalization, normal prototypes, DINO affinity, relational
  selection or S4 in its first frozen evaluation. No geometry-v3 poll or
  mutation, validation GT/test access, BTXRD feature extraction, selector
  training or consumer training occurred.

### Compact DINO-affinity descriptor preparation

- AffinityNet demonstrates that local semantic affinity can propagate weak
  discriminative evidence using only image-level supervision, and DINO-ECA
  shows that self-supervised ViT affinity can guide WSSS. TokenCut provides a
  complementary unsupervised result: cosine relations between self-supervised
  transformer tokens form a useful object graph, but it also documents a bias
  toward the largest salient object. These strengths and weaknesses motivate
  using affinity as a **candidate descriptor only**, not allowing it to grow or
  replace a mask. Primary sources:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html;
  https://openreview.net/forum?id=qipYQAcvVG; and
  https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.html.
- A dataset-agnostic source primitive was added at
  `project/models/mask_bag_affinity_features.py`, canonical-LF SHA-256
  `9334d41a8d426fadad8281cd5a6e6af70fa9a8b9cd3a47c2d6334013faa9b38a`.
  For every frozen candidate and RAD-DINO layer it computes eight explicit
  statistics: inclusive and off-diagonal proposal cohesion, the corresponding
  two context statistics, proposal-context cosine affinity, cohesion-minus-
  boundary affinity, and normalized effective token counts for proposal and
  context. The output is `24` values for the current three layers.
- The pairwise mean cosine is computed exactly from the squared norm of a
  weighted token sum and the squared weights, so the implementation does not
  materialize a `1024x1024` affinity matrix for each image. Off-diagonal
  cohesion removes self-similarity; it is zero when a candidate contains only
  one effective token rather than falsely claiming perfect multi-token
  coherence. Inclusive cohesion and cross-boundary affinity remain available
  for precisely that small-candidate case.
- This is not a reproduction of DINO-ECA. ECA derives a class-aware graph from
  a natural-image WSSS pipeline, whereas this arm uses frozen RAD-DINO hidden
  token cosine statistics inside the immutable BTXRD proposal/context
  geometry. Healthy bone may also be highly coherent, so the affinity arm must
  be evaluated alone first and may combine with normal prototypes only after
  complementary frozen recoveries are proven.
- Tests at `tests/test_mask_bag_affinity_features.py`, canonical-LF SHA-256
  `829a0d78c7e929046119f3158a02af6bbbc646c9bdc62127b4854c0053260a58`,
  specify maximal coherent-token affinity, reduced cohesion for orthogonal
  tokens, empty-context/invalid-candidate behavior and a GT/subgroup-free API.
  `py_compile` passes; the local no-Torch runtime reports
  `1 passed, 3 skipped`. These three numerical Torch tests are mandatory on
  Kaggle before R2 execution.
- R2 will append these statistics to the accepted v3 descriptor cache without
  changing candidate validity, ordering, WTA masks or pooling. Its first
  evaluation is separate from local tiles, prototypes and relational
  mechanisms. No geometry-v3 poll or mutation, validation GT/test access,
  descriptor extraction, selector training or consumer training occurred.

### Baseline-preserving auxiliary-descriptor residual

- To prevent a descriptor arm from becoming an uncontrolled full-selector
  retrain, a common residual adapter was added at
  `project/models/mask_bag_descriptor_residual.py`, canonical-LF SHA-256
  `9bb062d4ee1a100ab3bd108dfe52e846bf0df69aca29d58f2cdd2d8adb9eade9`.
  It independently projects the accepted v3 descriptor and one auxiliary
  descriptor, models their self/auxiliary/difference/product/cosine relation,
  and adds a scalar residual to the frozen baseline candidate logit.
  The last layer is exactly zero-initialized, so initial valid logits are
  bit-identical to the v3 scorer and invalid candidates remain zero.
- The first causal runs for normal prototypes (`4` auxiliary values),
  DINO-affinity (`24`) and local-view evidence must use separate instances of
  this adapter with the v3 scorer frozen. This both protects the large subgroup
  from gratuitous scorer drift and makes any learned correction attributable
  to the new representation. Unfreezing or composing adapters is a later arm
  requiring independent complementary evidence.
- Tests at `tests/test_mask_bag_descriptor_residual.py`, canonical-LF SHA-256
  `b8f36b70f679a419b089e529657e1e6f9436dfa2537c2160e1084e02c32ef6ac`,
  specify exact initialization identity, the expected zero gradient to earlier
  layers on the first step while the final layer becomes trainable, invalid
  candidate masking and a GT/subgroup-free API. `py_compile` passes; the local
  no-Torch runtime reports `1 passed, 3 skipped`. The three numerical Torch
  tests are mandatory on Kaggle before any auxiliary arm.
- Adapter training still uses only clean-train image labels and the frozen MIL
  loss contract; validation GT cannot choose an epoch, adapter size or
  optimizer. No geometry-v3 poll/mutation, BTXRD descriptor fit, validation
  GT/test access, selector training or consumer training occurred.

### Selector-bottleneck persistence rule and frozen ranking diagnostics

- The candidate oracle still exceeds every operational subgroup goal while
  the selected proposal fails every goal. Therefore candidate support is not
  reopened as the primary bottleneck. A failure of one or two descriptor
  mechanisms is evidence against those mechanisms, not evidence that the
  bottleneck has moved. The descriptor/selector diagnosis remains active until
  the finite campaign `R0-R2`, `S1-S4`, `T1` has either (a) produced the
  required regret reduction or (b) supplied a frozen all-arm pattern showing
  that proposal support, rather than ranking, has become limiting. This rule
  prevents cycling from selector to another speculative cause after each
  isolated failure.
- A fresh primary-source check supports this ordering. SelfPatch improves
  dense ViT features by enforcing patch-neighbour consistency, so it supports
  the already prepared affinity/local-descriptor arms rather than a new mask
  generator:
  https://arxiv.org/abs/2206.07990. Tang et al.'s weakly supervised RPN learns
  proposal objectness from image labels, but it addresses proposal generation
  and its natural-image objectness assumptions are poorly matched to subtle
  radiographic lesions:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Peng_Tang_Weakly_Supervised_Region_ECCV_2018_paper.html.
  Because the immutable BTXRD gallery oracle already passes all goals, that
  mechanism is deferred. SEAM and the small-object WSSS analysis continue to
  justify equivariant/local evidence specifically for the small subgroup:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html
  and
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html.
- An evaluation-only primitive was added at
  `project/models/mask_bag_ranking_diagnostics.py`, canonical-LF SHA-256
  `c8b95cde03006c78aad624f060684d2f4d3acad2bdfeb6fc42bfc05762abefe8`.
  Given candidate logits already frozen without validation masks, the
  post-freeze evaluator can report deterministic/tie-aware oracle rank,
  top-`1/3/5/10` oracle reach, top-k best achievable Dice, selected-to-oracle
  and top-k regret, and score-versus-candidate-Dice Spearman correlation.
  Complete misses remain in every subgroup, while recovered baseline misses
  and newly lost baseline hits are counted separately.
- These diagnostics distinguish mechanisms without changing the metric or
  tuning on validation GT. High top-3/5 reach with poor top-1 points to
  calibration or relational reranking; poor top-10 reach despite a high
  gallery oracle points to descriptor failure; a deficit concentrated in
  small supports the predeclared local-view arm; and a negative/near-zero
  score-quality correlation across sizes rejects the scoring representation
  even if one aggregate Dice moves by chance. Future arm prediction freezes
  must therefore serialize every valid candidate logit and its immutable
  gallery index, not only the winning logit. The evaluator may read candidate
  Dice only after that manifest is hash-frozen.
- Tests at `tests/test_mask_bag_ranking_diagnostics.py`, canonical-LF SHA-256
  `cfcd7855360ee3c4b3904fbdd0ddc8a5af6ff27f72de9bbd99841d17055b9cc8`,
  cover exact ranking/regret, oracle ties, invalid candidates, empty bags,
  recovered-versus-lost misses and the evaluation-only dependency boundary.
  All six tests pass under the bundled NumPy runtime. The current geometry-v3
  runner was not changed because its prediction protocol was already frozen;
  the richer score manifest is mandatory for the next selector arm. No Kaggle
  poll/mutation, validation GT/test access, training or consumer run occurred.
- That requirement is now executable rather than prose-only. The GT-free
  evidence contract at `project/models/mask_bag_score_evidence.py`,
  canonical-LF SHA-256
  `02371b4dda5e4b76947457c8265a449f361faa05ff4425d64777bf4675df046a`,
  stores every finite valid candidate logit with its immutable ascending
  gallery index in a hash-bound NPZ. Its manifest binds each payload to
  `image_id`, `group_id`, image label, candidate-gallery payload hash and exact
  candidate count. Validation fails closed on a cohort, provenance, physical
  hash, dtype/shape/order or argmax mismatch. Equal logits deterministically
  choose the first gallery candidate.
- Tests at `tests/test_mask_bag_score_evidence.py`, canonical-LF SHA-256
  `206d3fecd2688b244a7ba8a1124731df3b8fe436e9559068da60a6ed8ebcddf1`,
  cover round-trip provenance, stable ties, invalid/non-finite scores, physical
  tampering and independence from annotation/training loaders. All five tests
  pass under the bundled NumPy runtime; both new modules also pass
  `py_compile`. The current frozen geometry-v3 output contract remains
  untouched.

### R1 objective and train-only finite-selection contract

- The source audit confirmed that the baseline runner's positive-instance term
  uses the current model's detached argmax as its own positive target. Reusing
  it to train the R1 adapter would preserve the confirmation mechanism that R1
  is intended to escape. R1 therefore freezes the complete accepted v3 scorer
  and trains only a zero-initialized auxiliary residual with image-level
  SmoothMax BCE, aligned original/flip consistency and a small residual-drift
  penalty. It generates no positive candidate target. This remains standard
  bag-label MIL: Ilse et al. formulate MIL learning from a single label assigned
  to a bag of instances, while the already logged self-guided radiograph and
  ItS2CLR work explain why uncertain inferred instances require particular
  caution. Sources:
  https://proceedings.mlr.press/v80/ilse18a.html,
  https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html,
  and
  https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html.
- The executable objective is
  `project/models/mask_bag_residual_objective.py`, canonical-LF SHA-256
  `7b176c955408fb14b59e0c901243be12791f1d68540484ab06e99f2ba14b92df`.
  It detaches both original and flip base logits inside the objective, so the
  v3 freeze is enforced even if a caller accidentally includes the base scorer
  in an optimizer. Invalid candidates contribute to neither pooling,
  consistency nor drift. Its source contains no argmax, inferred instance
  target, segmentation loader or candidate-quality input.
- Tests at `tests/test_mask_bag_residual_objective.py`, canonical-LF SHA-256
  `a439ff2a1c26e926160beb2b3354e0bbda59bb88c05f154366ec82fc97f0d826`,
  verify the source boundary, zero-residual identity term, invalid-candidate
  exclusion, residual gradients and absence of a base-logit gradient. Local
  `py_compile` passes and the no-Torch runtime reports
  `1 passed, 3 skipped`; the three numerical Torch tests are mandatory in the
  post-v3 Kaggle preflight.
- R1 keeps the already frozen finite set `K={8,16,32}` and selects it using
  five group-held-out clean-train folds only. To reduce model-selection
  overfitting, the chosen value is the smallest K whose mean OOF image BCE is
  within one standard error of the lowest-BCE K. Cawley and Talbot show that
  model-selection overfitting can be comparable to apparent differences
  between algorithms, motivating this conservative finite rule:
  https://www.jmlr.org/papers/v11/cawley10a.html. The one-standard-error rule
  explicitly prefers the simpler model when cross-validated errors are not
  clearly separated. It is used here only for image-label training selection,
  never for reporting validation localization.
- Because the terminal v6 audit exposed a candidate-count shortcut, K is
  ineligible if the magnitude of its all-OOF candidate-count versus
  bag-probability Spearman correlation exceeds the frozen baseline magnitude
  by more than `0.02`. If all K values fail, R1 is rejected without reading a
  validation mask. The rule is implemented at
  `project/models/mask_bag_oof_selection.py`, canonical-LF SHA-256
  `4f3bacb75ad10b6751f90fbe8b4321d48db7ec03192382de0fe7eff82b460ff9`.
  Tests at `tests/test_mask_bag_oof_selection.py`, canonical-LF SHA-256
  `bcc106f96f5392931896bdfb8422699abe200fe516d7056d067d2db53628f856`,
  cover the one-SE choice, count guard, clear improvement, fail-closed
  behavior, order independence and absence of segmentation/subgroup inputs;
  all six tests pass under the bundled NumPy runtime.
- This prepares R1's causal loss and model-selection boundary but does not
  select K or fit a prototype before the accepted post-v3 cache exists. It
  changes neither the running geometry-v3 checkout nor its frozen protocol.
  No Kaggle poll, validation GT/test access, selector fit or consumer training
  occurred.

### Hash-bound selector-development cache format

- A reusable selector campaign is scientifically useful only if every arm sees
  the exact accepted v3 descriptors, valid-candidate indices and gallery
  geometry. Otherwise a changed cache could masquerade as a better selector.
  The cache format is now implemented at
  `project/models/mask_bag_selector_cache_io.py`, canonical-LF SHA-256
  `24f6f505c2fca39b989b9e6ff35548ad143e3ec03e732911f1c65b840ee42d93`.
  This is provenance/infrastructure rather than a new scientific mechanism; it
  operationalizes the already sourced representation/relational campaign.
- Each physical NPZ contains aligned original/flip fp16 descriptors, ascending
  immutable candidate-gallery indices, exact family IDs, four position-free
  shape features, the original component/prompt/source/fallback provenance,
  and symmetric float32 IoU/containment/relative-distance matrices. Training
  records are forbidden from retaining candidate masks
  after this graph geometry has been computed. Validation records retain only
  exact bit-packed immutable WTA masks, because future frozen candidate logits
  must still be converted into prediction maps before post-freeze evaluation.
- The manifest binds every record to split, image/group identity, image label,
  physical candidate-payload SHA-256, kept candidate count, descriptor
  dimension, mask-retention flag and physical cache hash. The loader rechecks
  hashes, schema, dtype, finite values, candidate order, family validity,
  geometry symmetry/range and packed-mask byte count. Thus even a hash-valid
  but structurally malformed record fails closed.
- The same module now supplies a full manifest validator for arm training. It
  requires independently expected train/validation image, group, image-label
  and candidate-payload identities, then opens and structurally validates
  every physical record rather than trusting the manifest hash alone.
- Tests at `tests/test_mask_bag_selector_cache_io.py`, canonical-LF SHA-256
  `00907cd4570205df8fa49cbc7df87dbb1322a97377e83bfddef17d461182b351`,
  cover validation round-trip, mandatory train mask deletion, physical
  tampering, structurally malformed payloads, full physical-manifest
  validation, split-specific boundaries, candidate ordering and absence of
  GT/subgroup APIs. All eight
  tests pass under the bundled NumPy runtime and both files pass
  `py_compile`.
- This format does not by itself authorize a cache. The future post-v3 builder
  must still bind the terminal prediction freeze/checkpoint and reproduce all
  `371` selected indices and final map hashes before writing the cache freeze.
  No cache was produced, Kaggle status polled, validation GT/test accessed or
  selector/consumer trained during this source preparation.

### Post-v3 cache-builder source readiness

- The cache production runner is now prepared at
  `project/build_mask_bag_selector_cache.py`, canonical-LF SHA-256
  `a17a7fcf1d9849c65fd2bda14e0ecd23947bfa3cdcd49618367ae2e961fc4992`.
  Before RAD-DINO extraction it verifies the split, every physical candidate
  payload through both manifests, the frozen model snapshot, the direct
  geometry-v3 prediction freeze, checkpoint, prediction manifest and all `371`
  physical prediction-map hashes. It is hard-bound to input `448`, projection
  `128/seed42`, three frozen layers, candidate cap `81`, the baseline random
  projection hash and two actual Tesla T4 devices.
- After extracting the corrected descriptors once, the runner loads the
  accepted v3 checkpoint and invokes the same prediction writer on the rebuilt
  validation cache. Cache serialization cannot begin until all `371` selected
  gallery indices and final fp16 map SHA-256 values match exactly; selected
  candidate logits, bag logits and probabilities must additionally agree
  within the fixed `5e-6` numerical tolerance. This detects changes in
  geometry, candidate validity/order, TTA alignment, scorer state or WTA map
  construction before selector development can proceed.
- Only after reproduction does it save train/validation cache records.
  Component IDs, prompt modes, proposal-source IDs and fallback flags are
  reopened from the already hash-verified candidate NPZs and sliced by the
  exact reproduced kept indices. Train masks are discarded after graph/shape
  construction; validation masks are bit-packed. The final cache freeze binds
  the terminal v3 freeze/checkpoint, source/protocol/split/model/projection,
  four candidate/pseudo manifests, all cache records and reproduction audit.
- Static tests at `tests/test_build_mask_bag_selector_cache.py`, canonical-LF
  SHA-256
  `610c52771e761571efe7d237bac80f45536000aa2fb4048b09029fd15c810cb5`,
  verify annotation/test isolation, pre-extraction input verification,
  reproduce-before-serialize ordering, complete provenance, split-specific
  mask handling and T4x2/frozen-geometry constraints. All six static tests and
  eight cache-I/O numerical tests pass; all four files pass `py_compile`.
- The runner remains source-only until the monitor reports terminal
  geometry-v3 and that output passes its independent audit. No cache was
  generated, Kaggle status polled, validation GT/test accessed, selector fit or
  consumer trained here.

### Common prediction-first evaluator for selector arms

- The campaign now has a common evaluator at
  `project/evaluate_mask_bag_selector_arm.py`, canonical-LF SHA-256
  `ccc3a4931907f0cafcf62adb0fef09db82108c6551db959c5926b8148d47b084`.
  Before importing the segmentation dataset or opening the previous
  GT-derived baseline rows, it verifies the split, selector-cache freeze and
  every physical validation cache record, the arm freeze, all `371` prediction
  maps, all-candidate score manifest/payloads, exact score-to-cache gallery
  indices, selected winner/logit/TTA semantics, and the direct accepted
  geometry-v3 baseline freeze/manifest/maps. This preserves the
  prediction-first boundary for every R/S/T arm.
- After the freeze boundary, candidate Dice is computed from the immutable
  bit-packed gallery and is never fed back to training or arm choice. For the
  complete `184/94/72/18` tumor cohort, the evaluator reports selected Dice,
  candidate oracle, deterministic oracle index/rank, top-`1/3/5/10` oracle
  reach and best Dice, selected/top-k regret, score-versus-candidate-Dice
  Spearman correlation, complete misses, recovered baseline misses and newly
  lost hits. It additionally confirms that the recomputed gallery oracle
  matches the frozen geometry-v3 per-image oracle.
- The evaluator distinguishes two outcomes:
  - `MECHANISM_PASS` requires selected-to-oracle regret reduction in at least
    two tumor-size subgroups, no overall selected-Dice regression, and no
    increase in the absolute overall candidate-count/miss association. Such an
    arm may be retained only for the predeclared complementary composition.
  - `OPERATIONAL_PASS` additionally requires all four current Dice goals, all
    oracle goals, overall paired complete-group bootstrap CI95 lower bound
    above zero, no tumor-subgroup mean decrease, no complete-miss increase and
    image AUROC at least `0.75`. Only this status sets
    `consumer_authorized=true`; `MECHANISM_PASS` alone cannot train a consumer.
  A failed arm advances to the next finite selector row without changing the
  selector/descriptor bottleneck.
- Static tests at `tests/test_evaluate_mask_bag_selector_arm.py`,
  canonical-LF SHA-256
  `21fafe74740588ae02a897e387a639a7723575334ce2ff4fe6df4fcd36aa8e29`,
  verify the single GT boundary, all-candidate/cache binding, gallery-index
  mapping, complete ranking diagnostics, distinct mechanism/operational
  statuses, fixed goals/cohort/bootstrap and consumer/test locks. All six
  tests pass; the evaluator and tests pass `py_compile`. Numerical paths that
  require Torch and frozen dataset artifacts remain mandatory in the future
  Kaggle preflight.
- No selector arm was evaluated, Kaggle status polled, validation GT/test
  accessed or consumer trained while preparing this evaluator.

### R1 normal-prototype residual training core

- R1 now has one shared fit/score implementation at
  `project/models/mask_bag_normal_residual_training.py`, canonical-LF SHA-256
  `8a2ca06585a2e718a29cda8c93f54f8c42743fb56f45a5ea134cad92cfcf5703`.
  The same functions will be used inside every group-held-out K-selection fold
  and for the final clean-train fit, preventing an implementation change
  between model selection and the frozen arm.
- The prototype bank uses only image-label-normal training bags. Original and
  horizontal-flip descriptors share the same image/family identity in the
  hierarchical weights, so adding the second view does not double an image's
  influence. Every image has equal total mass, then every immutable proposal
  family within that image, then every candidate/view within the family. This
  implements the conservative normal-prototype transfer already motivated by
  TPMIL rather than copying its full pathology-specific architecture:
  https://proceedings.mlr.press/v227/yang24d.html.
- Each candidate receives only the four predeclared normality features:
  nearest and soft-min normal-prototype distance, assignment entropy and
  top-two margin. The zero-initialized residual adapter is trained for fixed
  final-only epochs with image-level normalized SmoothMax BCE, original/flip
  consistency and residual-drift regularization. The complete v3 scorer is
  put in eval mode, marked `requires_grad=false`, evaluated under inference
  mode and detached again inside the objective. No argmax, hard positive
  candidate, segmentation quality or subgroup enters the fit. The bag-label
  supervision remains the MIL setting formalized by Ilse et al.:
  https://proceedings.mlr.press/v80/ilse18a.html.
- The scoring path returns every candidate logit in its original aligned cache
  order plus the normalized SmoothMax bag logit/probability. It performs the
  exact mean of original and aligned-flip logits required by the all-candidate
  evidence and common evaluator contracts.
- Tests at `tests/test_mask_bag_normal_residual_training.py`, canonical-LF
  SHA-256
  `ae36fe80615571ddecad18aed09a1d6a05e31bf69a013c59f109b162101d046f`,
  cover the no-confirmation/no-GT source boundary, flip-symmetric prototype
  weighting, bit-identical frozen scorer parameters with a learned residual,
  and complete ordered candidate scoring. Local `py_compile` passes and the
  no-Torch runtime reports `1 passed, 3 skipped`; all three numerical Torch
  tests are mandatory in the future Kaggle preflight.
- This is source readiness only. No prototype bank or adapter was fitted,
  Kaggle status polled, validation GT/test accessed or consumer trained.

### R1 group-excluded OOF orchestration core

- The fold-level R1 orchestration is implemented at
  `project/models/mask_bag_normal_crossfit.py`, canonical-LF SHA-256
  `30ea542734f9223942faf9f8ba29a4629089af2c2c3fb22e42b43e1adc1cb9f4`.
  Each invocation fits exactly one `(K, heldout_fold)` unit: it first proves
  that the training and held-out group sets are disjoint, fits the normal bank
  only on the training folds, enriches training and held-out descriptors using
  that bank, trains the residual only on training image labels and scores every
  held-out image once. The derived deterministic seed is
  `base_seed + 1000*K + fold`.
- The fold artifact includes training/held-out group inventories, prototype
  bank/audit, adapter state, exact objective/training config, final-only
  history, and held-out image label/logit/probability/BCE/candidate-count rows.
  It contains no candidate target or localization quality. Keeping a
  fold-level callable also allows the future Kaggle runner to dispatch
  independent folds across the two T4 devices without changing the scientific
  operation.
- The assembler requires every frozen fold exactly once, calls the existing
  `audit_crossfit_training_exclusion`, rejects a duplicate/missing image or
  identity/fold mismatch, and produces the five held-out fold BCE values plus
  the all-OOF candidate-count versus bag-probability Spearman correlation
  consumed by the already frozen one-standard-error/count-guard K selector.
  Thus no in-fold prediction can enter model selection.
- Tests at `tests/test_mask_bag_normal_crossfit.py`, canonical-LF SHA-256
  `15636584eea0b8e651931858f66b3666b3f6b3ba997bbd372e08d3bb532c37f1`,
  cover the group-exclusion/no-GT source boundary, complete two-fold OOF
  coverage with a bit-identical base scorer, and missing-fold rejection.
  Local `py_compile` passes and the no-Torch runtime reports
  `1 passed, 2 skipped`; both numerical Torch tests are mandatory in the
  future Kaggle preflight.
- This module only prepares the group-safe computation. No fold, prototype,
  adapter or OOF prediction was fitted; geometry-v3 was not polled and no
  validation GT/test or consumer was accessed.

### Complete R1 T4x2 runner readiness

- The complete R1 execution runner is prepared at
  `project/run_mask_bag_normal_prototype_arm.py`, canonical-LF SHA-256
  `5c11ae92eacc63d15d3838836267ed00a10485ea48bbedffd3dcf9092989692f`.
  It verifies the terminal selector-cache freeze, baseline checkpoint, frozen
  train/validation split and every physical cache record before assigning a
  fold or fitting any prototype. The exact known five-fold cohort
  `596/596/596/596/597` with normal/tumor counts
  `298/298`, `298/298`, `299/297`, `299/297`, `299/298` and group counts
  `196/196/197/197/198` is required.
- All `15` finite `(K in {8,16,32}, fold in {0..4})` jobs are assigned
  deterministically across two real T4 devices as `8/7` independent jobs.
  Adapter initial states are created sequentially on CPU with the exact
  derived seeds before two worker threads start; workers only load those states
  and therefore cannot race through the global RNG. CUDA deterministic
  algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are required. Each worker
  owns an independent frozen v3 scorer on its device.
- Every fold writes its prototype bank, adapter checkpoint, train/held-out
  group inventories, fit history and held-out image predictions with physical
  hashes. The assembled all-OOF evidence for each K is passed to the frozen
  one-standard-error and `baseline absolute count-probability Spearman + 0.02`
  guard. The baseline magnitude is a required protocol-bound scalar from the
  accepted geometry-v3 GT-blind prediction manifest; it cannot be recomputed
  from validation masks.
- Only the selected K is refit on the complete clean-train split. The runner
  then freezes all `371` validation all-candidate logits, exact gallery
  indices, prediction maps and manifests. It saves the final prototype and
  residual checkpoints plus a prediction freeze binding source/protocol,
  split/cache/baseline, every OOF artifact, K selection, final fit and score/
  map manifests. It imports no segmentation dataset and contains no validation
  target, candidate quality, consumer or test path.
- Static/core tests at
  `tests/test_run_mask_bag_normal_prototype_arm.py`, canonical-LF SHA-256
  `2655563f47214a598837762af237bf6bcc8e23f5b837e13ec9e10704bda76470`,
  plus the two updated core suites report `8 passed, 5 skipped` locally.
  The skipped tests require Torch and are mandatory on Kaggle before execution;
  all six changed/new files pass `py_compile`.
- The runner is not launched until geometry-v3 is terminal, independently
  audited, and its reproducible selector cache is accepted. No R1 fit,
  validation prediction, Kaggle poll, validation GT/test access or consumer
  training occurred here.

### Persistent descriptor/selector campaign contract

- The descriptor/selector diagnosis is now a campaign-level conclusion, not a
  disposable explanation for one arm. The frozen gallery oracle reaches
  `0.4090755342/0.2227494852/0.5941470844/0.6418253674`
  overall/small/medium/large and exceeds every current goal, whereas the v6
  selected result is only
  `0.2178991820/0.0781967787/0.3469906970/0.4310901171`.
  Therefore candidate support is sufficient under the present goal and the
  active causal target remains candidate representation, scoring, aggregation
  and selection. A failed selector arm rejects only its mechanism; it does not
  reopen proposal generation or replace the bottleneck hypothesis.
- The common endpoint for every remaining arm is the same selected-to-oracle
  regret on the immutable gallery, accompanied by top-k oracle reach,
  score-versus-candidate-Dice rank correlation, complete misses and
  candidate-count/miss association. This makes the experiments cumulative:
  each row must explain which part of the same regret it can reduce. An arm
  with a small-lesion regression is never accepted as a standalone solution;
  at most it can be retained as an orthogonal component for the predeclared C1
  composition, which must itself prove no subgroup decrease.
- The finite mechanism sequence remains fixed unless an earlier arm achieves
  the full operational gate:
  1. **R1 normal prototypes** exploits the reliable MIL fact that every
     instance in an image-label-negative bag is negative. Soft assignment to
     several normal prototypes can expose candidates that resemble normal
     anatomy without inventing positive instance labels. Its strength is
     conservative negative evidence; its weakness is that normal anatomy is
     diverse and a prototype bank can still encode acquisition/anatomy
     shortcuts. TPMIL supports soft assignment of all instances to prototypes
     and distance-based interpretability:
     https://proceedings.mlr.press/v227/yang24d.html.
  2. **R2 frozen RAD-DINO local affinity** changes representation rather than
     proposals. Local token correspondence and within-mask/near-boundary
     contrast are intended to separate a small lesion mask from a
     visually-similar anatomical mask. Its strength is dense self-supervised
     structure; its weakness is natural-image-to-radiograph domain shift, so
     the v3 scorer remains a frozen residual baseline rather than being
     replaced. DINOv2 reports transferable patch-level features:
     https://arxiv.org/abs/2304.07193. WeCLIP independently supports a frozen
     foundation backbone plus a lightweight trainable decoder/refinement
     design in image-label-only WSSS:
     https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html.
  3. **S1 family-balanced SmoothMax** changes aggregation only. Equalizing
     proposal-family mass directly targets the measured candidate-count
     shortcut and prevents a prolific prompt family from winning merely by
     multiplicity. Its strength is causal specificity and low capacity; its
     weakness is that it cannot repair an inseparable descriptor.
  4. **S2 critical-instance relation** uses both an instance-scoring stream and
     similarity from the critical instance to all other candidates. This can
     recover support distributed over related proposals, but an incorrect
     critical instance can propagate confirmation error. The mechanism is
     adapted from DSMIL, not treated as an architecture-name guarantee:
     https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html.
  5. **S3 same-family graph smoothing** tests whether overlap/containment and
     descriptor agreement provide useful relational consensus. Its strength
     is explicit candidate dependence; its weakness is over-smoothing between
     nested anatomical distractors. TransMIL motivates correlated rather than
     independent-instance MIL, while the BTXRD arm stays a smaller,
     provenance-constrained graph:
     https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html.
  6. **S4 group-excluded OOF proposal clusters** learns recurring proposal
     types from clean-train descriptors only. It can distinguish stable lesion
     and anatomy modes that one image cannot reveal; its main risk is learning
     group/acquisition shortcuts, hence all cluster features and selection are
     group-held-out before the final fit.
  7. **T1 self-paced confirmation** is delayed until R/S evidence yields a
     stable, cross-fitted high-confidence seed. It may refine the descriptor
     beyond fixed negative prototypes, but has the highest confirmation-bias
     risk and therefore cannot be the first repair. ItS2CLR reports
     self-paced pseudo-instance refinement for medical MIL and explicitly
     motivates controlling pseudo-label reliability:
     https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html.
- Small lesions remain a required selector diagnostic rather than a reason to
  jump back to proposal generation. The WACV study *Small Objects Matters in
  Weakly-Supervised Semantic Segmentation* independently finds that standard
  aggregate WSSS evaluation hides pronounced small-object failures and
  motivates size-stratified evaluation:
  https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html.
  In this project the immutable gallery already proves small-goal support
  (`0.2227494852 > 0.17895493`); the next question is specifically whether the
  selector ranks those supported small masks, not whether another gallery can
  be generated.
- `C1` may combine only mechanisms that independently satisfy the common
  mechanism gate and improve different regret slices; there is no unbounded
  architecture search. Proposal generation is reconsidered only if a future
  predeclared goal exceeds the frozen gallery oracle, or an independently
  audited gallery change raises the oracle under a separate protocol. Neither
  condition currently holds. This contract prevents one- or two-arm failures
  from causing another diagnostic loop.

### R2 affinity-cache integration and residual-training readiness

- Audit of the prepared selector cache exposed an execution gap: the compact
  affinity-statistic primitive existed, but the cache builder retained only
  mean-pooled descriptors. Running R2 from that cache would either require a
  second complete RAD-DINO pass or risk rebuilding proposal/context geometry
  differently from v3. The cache contract is therefore upgraded before any
  cache exists; this changes no running geometry-v3 artifact.
- `project/models/rad_dino_mask_bag_mil.py`, canonical-LF SHA-256
  `0c028a652ddf7710a9af3b3533a00ca796d2a76adea85ab5bec592f43ee7ee67`,
  now exposes one shared `proposal_context_grid_weights` primitive. Both the
  accepted mean descriptor and R2 affinity summaries use the same area
  projection, `minimum_grid_mass`, content mask, dilation radius, validity and
  padding exclusion. The refactor preserves the original arithmetic order;
  the future cache job must still reproduce all 371 v3 winners/map hashes
  before serialization, so source equivalence is not accepted on inspection
  alone.
- `project/run_rad_dino_mask_bag_mil_probe.py`, canonical-LF SHA-256
  `0a220933343212dd3f4cc349f459db54b2b492d04240a7aa28fb85c475d54ae8`,
  can now compute the fixed 24-value affinity summary during the same encoder
  pass that creates each original/flip descriptor. It asserts exact equality
  of descriptor and affinity kept indices and applies the corresponding
  square-frame flip to both candidate and content masks. This avoids a second
  3,352-image RAD-DINO extraction and makes T4x2 useful at the actual
  extraction bottleneck. Baseline descriptors retain the original CPU
  arithmetic required for exact v3 reproduction, while the additional
  affinity reductions run on the encoder output device rather than adding a
  large CPU-only matrix workload.
- Selector-cache schema v2 stores aligned fp16
  `affinity_features/flipped_affinity_features` for every candidate and binds
  `affinity_dim=24` in every manifest row. The loader physically validates
  dtype, finiteness, original/flip shape and candidate alignment. R1 now also
  fails closed unless the shared accepted cache contains these fields; this
  ensures R1 and R2 consume one identical gallery/cache rather than
  experiment-specific caches.
- The R2 fit/score core is prepared at
  `project/models/mask_bag_affinity_residual_training.py`, canonical-LF
  SHA-256
  `d7c3054f80bc89780eb2c9c56a3c6fb43655c12dd717c5ee223f7f9adf81d95f`.
  It requires exactly 24 aligned cached features, retains the complete frozen
  v3 scorer and trains only the zero-initialized auxiliary residual using
  image-level SmoothMax BCE, aligned original/flip consistency and residual
  drift. There is no argmax, inferred positive candidate, segmentation target,
  subgroup or validation-quality input. R2 therefore tests whether local
  token cohesion/boundary contrast improves ranking, not whether a second
  selector can relearn the gallery.
- This mechanism follows the already frozen AffinityNet/DINO-ECA/TokenCut
  rationale:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html,
  https://openreview.net/forum?id=qipYQAcvVG, and
  https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.html.
  Its anticipated strength is candidate-internal and across-boundary structure
  absent from mean descriptors; its explicit weakness is that healthy bone can
  also be highly coherent. Hence R2 remains separate from R1 normality,
  relational pooling and proposal generation, and must pass the common
  selected-to-oracle regret gate on its own.
- New R2 tests at
  `tests/test_mask_bag_affinity_residual_training.py`, canonical-LF SHA-256
  `0b231a2495c4e54a013ec1749872735ae9852c9feb1a858b0fa622d76bab0012`,
  plus updated geometry/cache/R1 suites report `26 passed, 9 skipped` under
  the local bundled NumPy runtime. The skips require Torch and remain mandatory
  in the post-v3 Kaggle preflight. All changed/new Python files pass
  `py_compile` and `git diff --check` has no error.
- This is source/cache readiness, not an R2 result. No cache, affinity feature,
  adapter, validation prediction or consumer was produced; validation GT and
  BTXRD test were not accessed, and Kaggle status was not polled outside the
  scheduled monitor.

### Complete R2 prediction-first runner readiness

- The isolated R2 runner is prepared at
  `project/run_mask_bag_affinity_residual_arm.py`, canonical-LF SHA-256
  `cfec06dd97d18156347fcea7079be6c9b9d9246d1b2647751e721119bfbe1d9d`.
  It reuses the physically verified cache/baseline loading and all-candidate
  evidence writer from the R1 runner, but it does not fit a normal prototype
  or import any R1 scientific feature. Before training it verifies the cache
  freeze, split, baseline checkpoint, all physical train/validation records and
  exact 24-value affinity alignment for every candidate.
- R2 has one predeclared final-only fit: 16 epochs, batch 16, AdamW
  `3e-4/1e-4`, hidden dimension 128, seed 42, bag temperature 0.20,
  original/flip consistency 0.10 and residual drift `1e-3`. There is no epoch
  search, early stopping, validation loss or alternate hyperparameter arm.
  The complete v3 scorer is frozen; the only learned parameters are the
  zero-initialized affinity residual. This avoids converting one scientific
  mechanism into an implicit validation architecture search.
- Runtime requires exactly two real Tesla T4 devices and deterministic CUDA
  controls. The small adapter fit uses one T4, where multi-device gradient
  synchronization would add more overhead and nondeterministic surface than
  useful throughput. After the fixed fit, two independent frozen
  baseline/adapter replicas score interleaved validation shards of `186/185`
  images concurrently. Results are joined back to the immutable 371-row cache
  order and fail closed on a missing or duplicated image.
- The arm writes a hash-bound adapter checkpoint, final-only history, all 371
  candidate-logit payloads, score manifest, WTA probability maps, prediction
  manifest and prediction freeze before the common evaluator may import
  validation segmentation. The freeze explicitly records
  `training_labels=image_level_only`,
  `epoch_selection=fixed_final_epoch_only`,
  `validation_gt_read=false`, `consumer_trained=false` and
  `test_evaluated=false`.
- Static tests at
  `tests/test_run_mask_bag_affinity_residual_arm.py`, canonical-LF SHA-256
  `4ab40011239ffc18c9972e969152b6cc38cda5f85ad76330f3c26a050d8ed38a`,
  verify source isolation, cache-before-fit ordering, the single finite fit,
  T4x2 complete scoring and prediction-before-evaluation freeze. Together with
  the R1/R2 core suites they report `13 passed, 5 skipped`; skipped numerical
  Torch paths remain mandatory in Kaggle preflight. All files pass
  `py_compile`.
- The runner is not launched before terminal geometry-v3 audit and acceptance
  of the shared reproducible cache. Its eventual result is judged only by the
  common selected-to-oracle regret mechanism gate; a failure rejects affinity
  cohesion as a mechanism and advances to S1/S2, without changing the
  descriptor/selector bottleneck. No Kaggle status poll, validation GT/test
  access or consumer training occurred here.

### S1 matched family-balanced pooling readiness

- A causal audit refined the S1 design. Recomputing only a family-balanced bag
  probability cannot change candidate ranking, while training a new residual
  head can. Attributing a resulting Dice change directly to family balance
  would therefore confound the new head/additional epochs with the pooling
  mechanism. S1 is now a matched two-arm experiment: an ordinary normalized
  SmoothMax residual control and a family-balanced SmoothMax residual receive
  the exact same frozen descriptors/base scorer, architecture, initial state,
  batches, optimizer, epochs and loss weights. The sole scientific change is
  the bag pooling operation during training and final bag scoring.
- The paired training core is
  `project/models/mask_bag_pooling_residual_training.py`, canonical-LF
  SHA-256
  `8deaee980bd639fddc309eab9942147c8b0be8538951298e6e3c4fecb38c063f`.
  A descriptor-only residual MLP has an exactly zero-initialized final layer,
  so both arms begin with candidate logits identical to v3. The base logits
  are detached inside the objective. Both arms use only image-level BCE,
  aligned original/flip consistency and residual drift; there is no argmax,
  positive-instance target, segmentation quality or subgroup input.
- The standard arm uses the existing normalized SmoothMax across all valid
  candidates. The family arm first applies the same normalized SmoothMax
  within each immutable `(proposal source, prompt mode, component)` family,
  then across families. An identical duplicate within one family is therefore
  neutral instead of increasing that family's effective representation.
  Ilse et al. motivate permutation-invariant learned aggregation for
  image-label MIL:
  https://proceedings.mlr.press/v80/ilse18a.html.
  Deep Sets provides the general invariant-set-function foundation:
  https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html.
  The BTXRD hierarchy is a bounded adaptation to the measured proposal-count
  shortcut, not a claim that proposal families are latent tumor classes.
- The complete paired runner is
  `project/run_mask_bag_family_balanced_pair.py`, canonical-LF SHA-256
  `e8bc06a7e2e3bd75de2f34fe32823e3d92e6237d6fdbcb1498ff4b8f991d6f59`.
  It verifies the shared cache/baseline and every physical record before
  training. One CPU-created initial state is physically frozen and supplied to
  both workers. Standard trains on T4:0 and family-balanced on the identical
  T4:1 concurrently. Before fitting, each device scores the same eight-image
  probe from that shared zero-residual state; execution fails if maximum
  candidate-logit disagreement exceeds `5e-6`, preventing a device difference
  from masquerading as the pooling effect.
- Both arms use the same fixed final-only contract: 16 epochs, batch 16,
  AdamW `3e-4/1e-4`, hidden 128, seed 42, bag temperature 0.20,
  consistency 0.10 and drift `1e-3`. Each independently freezes all 371
  candidate-score payloads, maps, manifests, checkpoint and history before a
  pair-level freeze is written. The pair freeze records all matched variables,
  the initial-state hash, cross-device probe delta and
  `sole_changed_variable=standard_vs_family_balanced_bag_pool`.
- S1 promotion is stricter than comparing family-balanced directly with v3.
  The family arm must first improve the frozen ranking/mechanism diagnostics
  over its matched standard residual control; otherwise any improvement shared
  by both is residual-head/extra-training evidence, not family/count evidence.
  It must still satisfy the common no-overall-regression, subgroup-regret and
  count/miss guard relative to v3. Its strength is direct control of proposal
  multiplicity; its weakness is that it cannot repair an affinity/descriptor
  separability failure.
- Tests at
  `tests/test_mask_bag_pooling_residual_training.py`, canonical-LF SHA-256
  `c1858d832b83a7f1a148cb2bc57a1dd5fb8c4c0f8a6ba046635ddea5eaa39e71`,
  and
  `tests/test_run_mask_bag_family_balanced_pair.py`, canonical-LF SHA-256
  `4aba83a50434cc35481937e1a673a84650efc9505d608cfff9117d7ecf395bb7`,
  verify zero-residual identity, standard/family objective separation, frozen
  base and complete candidate scoring, matched variables, T4x2 execution,
  cross-device guard and prediction-first freezes. The focused suites report
  `7 passed, 7 skipped`; numerical Torch paths remain mandatory on Kaggle.
- S1 remains source-only until geometry-v3 and the common cache pass their
  independent audits. No S1 model/prediction was produced, Kaggle was not
  polled outside its scheduled monitor, validation GT/test were not accessed
  and no consumer was trained.

## 2026-07-29 - Geometry-v3 version 1 preflight test-fixture error

- The single scheduled status check found Kaggle kernel
  `itsthang333/btxrd-rad-dino-mask-bag-geometry-v3`, version `1`, at terminal
  `ERROR`. Direct output was downloaded to a separate ignored temporary
  directory before diagnosis. The wrapper stopped in the focused preflight
  with `32 passed, 1 failed`; candidate generation, RAD-DINO extraction,
  optimizer construction, validation prediction, validation-GT evaluation and
  test access were never reached.
- The only failure was
  `test_project_then_square_flip_preserves_asymmetric_padding_geometry`.
  Its fixture created a wide `3x6` source with vertical square padding, then
  changed the vertical content box while applying a horizontal mask flip.
  Horizontal flipping does not mirror vertical padding, so the expected tensor
  encoded the wrong transform. The production geometry is unchanged:
  `content_box=(x0,y0,x1,y1)` is verified by the projection implementation and
  callers, and the runner projects once into the square frame before applying
  its horizontal flip.
- The fixture is corrected to a tall `6x3` source with real asymmetric
  horizontal padding: original box `(1,0,4,6)` and mirrored box
  `(2,0,5,6)`. This now tests the intended odd-padding horizontal
  flip-equivariance contract. The production model, projection arithmetic,
  gallery, loss, optimizer, epoch budget, seed, evaluator, goals and subgroup
  definitions are unchanged. In the isolated geometry-v3 execution lineage,
  canonical-LF test SHA-256 changes from
  `b563dd7c91aa47c939dce1c95db23344548799ba85af1dd53be13bf30643df0c`
  to
  `fed558ea04298bbafdcf922d36f43c176ca733d688815231ca90900d79dc9428`.
  The current research branch also contains later R2 cache tests, so its
  complete combined test-file hash is intentionally different and is not used
  by this isolated rerun.
- Local `git diff --check` passes. The local environment still lacks Torch, so
  the corrected numerical fixture remains mandatory in the next Kaggle
  preflight rather than being claimed as locally executed evidence. A
  hash-bound implementation-only correction addendum and wrapper revision
  must be frozen before version `2` can launch. No scientific result exists
  for version `1`, no consumer was trained and BTXRD test remains locked.
- To avoid importing later R1/R2/S1 source changes into this causal geometry
  arm, the rerun checkout is isolated from the original execution commit
  `6e0b50c`. Repair commit `3c852f0a3a3847f1190f452cc2b7ace5558a0312`
  changes only the invalid fixture; checkout
  `55a2305ee143ae7f9232ac8542103475e890fb60` adds only its correction
  addendum. The production model/runner hashes remain exactly
  `44cd6ff0...` and `1bf56d0d...`.
- Correction protocol
  `rad_dino_mask_bag_mil_descriptor_geometry_v3_execution_correction_v1`
  is frozen at canonical-LF SHA-256
  `da2037a57769435b0dc74bb74c08c9c0de90051b2a421c76a9a9ea99c018086e`.
  Version-2 wrapper and metadata hashes are
  `4592bffeecd3064f6d7a8eb699555af913de91fd12fc2ba942bc00ca2fe67a80`
  and
  `ed2a289cdcf028038451a6fe256249608486f891f5867c042057cc1a11782a2d`.
  The wrapper verifies that the repair is an ancestor, binds the corrected
  fixture hash, copies the addendum to output provenance and records its own
  physical hash. Prelaunch audit
  `rad_dino_mask_bag_mil_descriptor_geometry_v3_execution_v2_wrapper_audit.json`
  reports `PRELAUNCH_PASS`.
- Per the user's updated direction, the ten-minute heartbeat
  `theo-d-i-geometry-v3-m-i-10-ph-t-trong-task-ch-nh` was deleted. The
  version-2 preflight will be followed at a short cadence so an implementation
  error can be diagnosed immediately; this monitoring change does not alter
  the experiment.

### Geometry-v3 version 2 duplicate source-guard error

- Rapid monitoring found version `2` at terminal `ERROR` and the one-minute
  heartbeat was deleted before retrieval. The output directory was empty
  because the wrapper stopped at `3.06` seconds. Direct Kaggle execution logs
  prove checkout `55a2305` succeeded and then raised
  `Geometry-v3 source SHA-256 mismatch:
  tests/test_rad_dino_mask_bag_mil.py`.
- Root cause is a second wrapper guard, not the corrected fixture or
  production geometry. The wrapper had already authorized and verified the
  repaired test hash while iterating the parent protocol, but later iterated
  the immutable geometry-v3 protocol and compared the same file against its
  original pre-repair hash. Runtime installation, candidate generation,
  RAD-DINO extraction, optimizer construction, validation prediction and GT
  evaluation were never reached; no consumer or test access occurred.
- Correction-v2 protocol is frozen at canonical-LF SHA-256
  `9527deb8c6507da73a35d933224ab130e1303c310b897bd3e7a6a3204bd5360d`.
  The sole wrapper repair applies the already authorized corrected test hash
  in that second source loop. Production model, runner, fixture content and
  every scientific setting remain unchanged. Isolated execution checkout is
  `ae035311b7e7e080c7454a5f2e06b3f6f96d060b`; version-3 wrapper SHA-256
  is `02309d49385d518dd64c0e233d336abc6101f78af52a7232e03b09df06301bf1`.
  The prelaunch wrapper audit reports `PRELAUNCH_PASS`; numerical Torch
  execution remains mandatory on Kaggle.

### Geometry-v3 version 3 byte-hash reproducibility failure and paired recovery

- After the user resumed the paused task, one status check found version `3`
  at terminal `ERROR`. Direct logs and compact output were downloaded into a
  new ignored directory. Source checkout `ae03531`, the version-3 wrapper hash
  `02309d49...`, focused preflight `33 passed` and whole-repository preflight
  `219 passed, 1 skipped` all passed. The run then completed all `2,981`
  train candidate payloads and stopped at `5,709.73` seconds with
  `Regenerated train candidate/pseudo manifests differ from terminal v6`.
  Validation candidate generation, fractional-mass audit, optimizer
  construction, selector fitting, prediction, validation-GT evaluation,
  consumer training and BTXRD test access were never reached. Version `3`
  therefore has no scientific result.
- Direct terminal-v6 manifests were downloaded separately and joined by exact
  image name against the regenerated manifests. Candidate structure is
  unchanged for every image: `2,981/2,981` image rows, candidate count, box
  count, positive/negative point count and generation status agree exactly.
  The only candidate-manifest differences are `diagnostic_sha256` on all
  `2,981` rows and compressed `diagnostic_bytes` on `2,926`. The terminal-v6
  versus version-3 candidate-manifest hashes are `7dfe43cc...` versus
  `ad3b52d...`.
- The final pseudo mask is bit-identical for all `2,981` train images:
  `mask_sha256`, status and SAM candidate count have zero mismatches. The
  pseudo-manifest hashes nevertheless differ (`5d50bd39...` versus
  `5aec58ce...`) because `1,305` rows contain small summary differences.
  Maximum absolute CAM deltas are at most `1.20e-7`; the largest
  selection-score delta is `0.002014`; only six unique prompt counts, two
  selected-area summaries and one above-threshold count differ. Every one of
  those rows still has the same final pseudo-mask hash.
- This evidence rejects the wrapper's whole-file hash equality as a sufficient
  *causal equivalence test*, but it does not authorize silently replacing the
  four frozen v6 hashes. NumPy documents NPZ as a ZIP archive of named NPY
  members, so an NPZ byte hash binds serialization as well as array values:
  https://numpy.org/doc/stable/reference/generated/numpy.savez.html.
  PyTorch's reproducibility note separately states that seeded CUDA execution
  is not universally bit-identical unless deterministic implementations are
  available and enforced:
  https://docs.pytorch.org/docs/stable/notes/randomness.html. More
  importantly, terminal v6 deleted its physical NPZ payloads after compact
  export; its manifests cannot prove that every non-selected SAM candidate is
  bit-identical. A `v3-minus-v6` result on a regenerated gallery would
  therefore retain a small gallery confound even though the observed discrete
  evidence is extremely stable.
- The valid recovery is predeclared conceptually before any retry: generate
  and physically freeze one gallery, run the legacy direct-resize descriptor
  geometry and square-corrected geometry against that exact same gallery,
  freeze both complete `371`-map predictions before either evaluator imports
  validation segmentation, and compare them with the same paired
  complete-group bootstrap. Both arms must have identical labels, gallery,
  ordering, RAD-DINO tokens, scorer, loss, seed, optimizer and final-only
  epochs; descriptor coordinate geometry is the sole changed scientific
  variable. The corrected geometry remains canonical because it repairs a
  proven coordinate error, not because validation GT selects it. This paired
  control also avoids reopening the already established
  descriptor/selector—not candidate-support—bottleneck.
- Machine-readable error evidence is frozen at
  `artifacts/kaggle/rad_dino_mask_bag_mil_descriptor_geometry_v3/version3_error_audit.json`,
  SHA-256
  `db4b29054719960abfed97d6380ff7cb8388d0b86f3f574e662f9f1311110702`.
  It records the direct log/wrapper hashes, field-level manifest comparison,
  maximum numeric deltas, execution boundary and locks:
  `consumer_trained=false`, `test_evaluated=false`.

### Same-gallery paired geometry recovery freeze

- The recovery is now fully frozen before launch rather than weakening the
  failed equality assertion. Protocol
  `rad_dino_mask_bag_mil_descriptor_geometry_v3_execution_correction_v3`
  has canonical-LF SHA-256
  `4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555`.
  It requires a direct immutable mount of the `2,981` physical train payloads
  already produced by version `3` (`ad3b52d...` candidate manifest,
  `5aec58ce...` pseudo manifest); the job fails before further heavy work if
  that failed-version output cannot be mounted. Validation candidates are
  generated exactly once with the unchanged recipe, and both geometry arms
  consume the same physical train/validation paths and hashes.
- Scientific runner commit
  `fda732941664e67d4b87a8c3cba071b6979b2214` adds one mandatory,
  explicit execution argument only:
  `legacy_direct_resize` or `square_corrected_v3`. The legacy arm supplies the
  original direct-resize mask and all-valid content map to the same pooling
  implementation; a new Torch test proves this is exactly equal to the
  original no-content-mask contract. The corrected arm retains the frozen
  continuous content-box projection. Both isolated processes reseed to `42`
  and otherwise use identical RAD-DINO snapshot, random projection, scorer,
  image-level BCE/self-guided MIL/flip losses, optimizer and 16 final-only
  epochs. Canonical-LF runner/model hashes are `6e0280e6...` and
  `44cd6ff0...`.
- Implementation commit
  `75efd2b77776d15ce07cee52e14122b182390167` adds a fail-closed paired
  orchestrator. It verifies every candidate payload and also records a
  deterministic semantic hash of the boolean SAM-mask tensor stream. It
  freezes and physically checks `371` legacy plus `371` corrected maps, then
  writes a pair freeze before either evaluator imports validation
  segmentation. Required post-freeze comparisons are corrected-minus-legacy,
  each arm versus the promoted baseline, and descriptive-only comparisons
  versus terminal v6, all with complete misses and `10,000` paired
  complete-group bootstrap replicates. Current operational goal checks use
  `0.34024039/0.17895493/0.51244178/0.49370336`; the corrected coordinate
  contract remains canonical independently of validation performance.
- The new recovery/static suites report `8 passed`; the broader focused local
  group reports `12 passed, 1 skipped`. Torch numerical coverage remains
  mandatory on Kaggle and is bound to expected preflight counts `34 passed`
  and whole repository `223 passed, 1 skipped`. The execution checkout
  verifies two real Tesla T4 devices and uses RAD-DINO DataParallel in both
  arms. No consumer or BTXRD test path is available to the orchestrator.
- Kernel wrapper SHA-256 is
  `d1d7d9de819f5f49967d526f7fa0f1058938f1cabc2882078e51c4e094536b8d`;
  metadata SHA-256 is
  `d7f720d3bfe33fc46e4a2e4fcfe533f8b3a63ebf46cca9581f4592344c491790`.
  Prelaunch audit
  `rad_dino_mask_bag_mil_descriptor_geometry_paired_v1_wrapper_audit.json`,
  SHA-256
  `56632559c69683d5f2b47352c90443c4d2d4a3d392d5d0192a7a1c3def1f09f2`,
  reports `PRELAUNCH_PASS`. It binds the recovery kernel
  `itsthang333/btxrd-rad-dino-mask-bag-geometry-paired-v1` to the prior
  geometry-v3 output as a kernel source; no monitor is created by this
  protocol.

## 2026-07-30 - Goal recovery and paired geometry dataset-source correction

- The complete `6,644`-line research log was re-audited from the first clean
  split and supervised reference work through the latest same-gallery recovery
  freeze. Git was clean on `research-wsss-improvement` at
  `3cd43a126667fd1b54d9a1ba21243f0d1b9876f5`. The task runtime contained no
  active goal even though the research contract remained intact in this log
  and its protocols. The active goal was therefore restored verbatim in
  substance: image-level-only WSSS; Dice
  overall/small/medium/large at least
  `0.34024039/0.17895493/0.51244178/0.49370336`; persistent focus on the
  descriptor/selector bottleneck; prediction freeze before validation GT; no
  consumer before the operational gate; BTXRD test locked; heavy compute only
  on Kaggle T4x2/P100.
- One direct status check found paired recovery kernel
  `itsthang333/btxrd-rad-dino-mask-bag-geometry-paired-v1` version `1` at
  terminal `ERROR`. Direct logs and the only output files were downloaded to a
  new ignored temporary directory. Checkout `75efd2b...`, runtime installation
  and frozen split hash `85511ee1...` passed, then the runner stopped at
  `87.19` seconds because its fail-closed constant expected `353` test rows
  while the immutable split correctly contains `373`. T4x2 verification,
  gallery discovery, preflight tests, validation generation, optimizer
  construction, prediction, validation-GT evaluation, consumer training and
  test access were never reached. This is an implementation-only error and has
  no scientific result.
- Machine-readable evidence is
  `artifacts/kaggle/rad_dino_mask_bag_mil_descriptor_geometry_v3/paired_version1_error_audit.json`.
  The direct output-log SHA-256 is
  `a5f249719babd263f639857e0261690e55437ea04381509270e789fb8ea40d1a`;
  the separately retrieved CLI-log JSON SHA-256 is
  `5f4c7ddd4bed53a24ac388d1acda341e709f9d2cf45dbc58c910539fbdac41a9`.
- The split guard was corrected from `test=353` to `test=373` and bound by an
  assertion in the existing recovery test. `py_compile` and the focused suite
  report `3 passed`. Isolated source commit
  `f3ad49d657e50b5bc0da3572d98f3376800f2099` was pushed to
  `codex/geometry-v3-v2`; runner/test canonical-LF SHA-256 values at that
  correction were `f9fecd9d.../156250a1...`. Correction-v4 protocol SHA-256
  is `9f64d5fcfa420e0e9f09953c0db2688c0654711db29cce7779cc527cda855093`.
- The version-3 train gallery recovery finished locally without regenerating
  any candidate. Exactly `5,967` files / `1,172,344,928` bytes were retrieved
  from the direct error output. All `2,981` candidate NPZ payloads
  (`1,169,292,707` bytes) and all `2,981` pseudo masks (`1,331,655` bytes)
  were physically rehashed against manifests
  `ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1`
  and
  `5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21`.
  `consumer_trained=false` and `test_evaluated=false`.
- Kaggle does not accept a terminal-ERROR kernel as an input source, so the
  exact recovered gallery is being uploaded as private dataset
  `itsthang333/btxrd-mask-bag-geometry-v3-train-gallery-v1`. Directory
  transport uses `train_candidates.zip`; the wrapper safely extracts it and
  supplies an explicit recovered-root path. The runner still verifies both
  manifest hashes and every manifest-bound NPZ/PNG before heavy work. This is
  transport only: neither arm's gallery, descriptor, loss, optimizer, seed,
  epoch budget, prediction writer, evaluator or goal changes.
- Transport-root support is isolated at source commit
  `260388aee9a0d264982cbad6b8ded353c9fa3eaf`; final runner/test canonical-LF
  SHA-256 values are `72e1226d.../50231367...`. Correction-v5 protocol is
  frozen at SHA-256
  `b858c2e196c4a364c7c33b319fb3f3c00062091ddd5ddf8110b853fe6b5cbb39`
  and committed in execution checkout
  `f03c16c8543ac242849a2acd2b55cd9820d1d492`. The next valid action is to
  finish the private dataset upload, freeze/audit the revised wrapper and
  launch paired version `2`. No monitor is created; descriptor/selector arms
  R1/R2/S1 remain gated on the terminal paired audit and accepted shared cache.
- This correction uses no new scientific technique. The scientific rationale
  remains the same-gallery causal control already sourced in the preceding
  section; the relevant serialization/reproducibility references remain NumPy
  NPZ documentation
  (https://numpy.org/doc/stable/reference/generated/numpy.savez.html) and
  PyTorch reproducibility guidance
  (https://docs.pytorch.org/docs/stable/notes/randomness.html). No external
  metric or performance claim is transferred to BTXRD.
- A final prelaunch source audit found that extracting the archive beneath the
  runner's own runtime directory would pre-create a path that the runner
  intentionally creates with `exist_ok=false`. No kernel version used this
  wrapper. Correction-v6 first froze unconditional deletion of the extracted
  transport copy; correction-v7 then assigned it the distinct staging root
  `/kaggle/working/mask_bag_geometry_gallery_transport`. Their canonical-LF
  SHA-256 values are `b4f6fb0e...` and
  `a1dd596585e1eb196fe229b5d108373ff8999983ee79361f3703e2c4dbc98051`.
  Final isolated checkout is
  `217b68da81096b47dc9b82ee517171a205b9b4ac`; the revised wrapper/metadata
  SHA-256 values are `fb82857a54ef2baedb7a996cbe887ad37dcdd685f00ed8ee335395d9c80a987d`
  and `1c00d8ba7547ac4009f0bcb3a9e59588590752b73dbb0d337dfc7ea64d2a44dd`.
  `py_compile`, safe-archive extraction, path-traversal rejection, distinct
  staging/runtime roots and exact Git/protocol/source bindings pass locally.
  The private dataset upload remains the only unfinished prelaunch step; one
  upload process is active and no status monitor or duplicate upload exists.
- The single upload completed after `1:04:57`. Kaggle reports private dataset
  `itsthang333/btxrd-mask-bag-geometry-v3-train-gallery-v1` version `1` as
  `ready`, dataset ID `11426522`, `isPrivate=true`, and remote size
  `1,146,249,052` bytes. The uploaded files are the 599-byte physical audit
  plus `train_candidates.zip`; no source candidate was regenerated.
  Prelaunch audit
  `rad_dino_mask_bag_mil_descriptor_geometry_paired_v2_wrapper_audit.json`
  reports `PRELAUNCH_PASS` at canonical-LF SHA-256
  `d314d6ce4c17cf2190db9cd517644530a461bf4abfd98aee4146d89678db99bc`.
  It verifies the private dataset, exact source/protocol/wrapper/metadata
  bindings, safe extraction and path-traversal rejection, distinct transport
  and runner-runtime roots, removal of the rejected error-kernel source,
  T4x2 metadata and the unchanged prediction-first scientific contract.
- Kaggle accepted paired recovery kernel version `2` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-geometry-paired-v1.
  Its metadata requests `NvidiaTeslaT4`; the runner still fails closed unless
  exactly two physical T4 devices and a nontrivial convolution on each pass.
  Invalid free-form tags were ignored by Kaggle and do not affect inputs,
  source or execution. No monitor was created and no launch-time status poll
  was made. Validation GT, consumer training and BTXRD test remain locked.
- The next single status check found paired version `2` at terminal `ERROR`.
  Direct logs and the sole compact output log were retrieved into a new
  ignored directory. Source checkout `217b68d...` and all wrapper-bound source
  hashes passed, then execution stopped at `20.20` seconds before runtime
  installation because the wrapper found no literal `train_candidates.zip`.
  The private dataset was attached in the remote kernel metadata, so the error
  is the wrapper's archive-name/mount-representation assumption, not missing
  scientific input. T4x2 verification, runner runtime creation, gallery
  opening, preflights, validation generation, optimizer construction,
  prediction, validation GT, consumer and test were never reached.
- Direct output-log and CLI-log JSON SHA-256 values are
  `fa2ce84a6861f3ebfb2cd447835a0cf073264393fe0a54699607edd5b5a879c5`
  and `e4f35aedf93d69025fe02107fff394555e31f9e91ec6d68f43b5bd7296de7eb7`.
  Machine-readable error audit
  `paired_version2_error_audit.json` has canonical-LF SHA-256
  `26c4d7d5a612946b2038521d5869548372bbe797a594a1edc6fc7eec2f6f2ef6`.
- Correction-v8, frozen before another push at source commit
  `c3d06ea8f4835f32617831fc20e3c089e542b425`, first accepts exactly one
  direct Kaggle root whose candidate and pseudo manifests match
  `ad3b52d.../5aec58ce...`; only if none exists does it use the already frozen
  safe archive extraction. Multiple matches fail closed, and a missing match
  emits only a bounded file-path/size inventory. No image, label, candidate
  tensor or GT is opened for discovery. Protocol canonical-LF SHA-256 is
  `e457d09016eb9cb4fde19a90881d279ebe323ac429ace23da46dad020d4e2bfd`.
  Wrapper v3 SHA-256 is
  `d373171fc204d80e9c6f0b97e965bbc55ab0c1be42568173c2d652d65168f1bf`.
  Local direct-mount, safe-archive, traversal-rejection, distinct-root,
  `py_compile` and exact Git/source/protocol binding checks pass. Prelaunch
  audit SHA-256 is
  `ad4d5a18de0e13b6d8bc8a3f3054e47776efe3d3f72a7560e6436025dbe5f226`.
  The paired scientific contract is unchanged.
- Kaggle accepted paired recovery version `3` from the correction-v8 wrapper.
  Invalid free-form tags were ignored again; the attached private dataset,
  T4x2 machine shape and all scientific inputs remain unchanged. No monitor or
  immediate status poll was created. Validation GT, consumer and BTXRD test
  remain locked.

## 2026-07-31 - Paired geometry version 3 float32 audit correction

- A single terminal status check found paired recovery version `3` at
  `ERROR`. Direct compact output and logs were retrieved into a new ignored
  temporary directory. Checkout `c3d06ea...`, the T4x2 guard, focused
  preflight (`34 passed`), whole-repository preflight
  (`223 passed, 1 skipped`), physical recovered-train-gallery checks and
  prediction-first generation of all `371/371` validation candidate payloads
  passed. The candidate and pseudo manifests are
  `6fa02c3716b96609227677056c6effa439212d4cf61bf970f0aa9301dcc38cd6`
  and
  `78c19b7600d895059ab45168ae337e5d6f8c660a1e2468e6b55151ed1365ea9c`.
  The job then stopped in the GT-blind train fractional-grid-mass audit at
  `IMG000004.jpeg`, before optimizer construction, either geometry arm,
  prediction-map freeze or validation-GT import. There is no scientific
  result; `consumer_trained=false` and `test_evaluated=false`.
- The exact physical 27-candidate NPZ and frozen source dimensions
  `2561x2817` reproduce the cause. Horizontal reversal changes only float32
  reduction order: maximum mass difference is
  `1.52587890625e-5`, four candidates exceed the old fixed absolute
  tolerance `1e-5`, and the maximum difference is only two float32 ULPs.
  Original/flip retained vectors remain exactly equal at the scientific
  minimum-grid-mass threshold `0.25`. Thus the failed condition is a
  scale-independent numerical assertion, not a candidate-retention,
  descriptor or geometry disagreement.
- Correction-v9 keeps exact retained-vector equality and replaces only that
  diagnostic assertion with the frozen per-pair bound
  `max(4 float32 ULP at the mass scale, 4*float32 epsilon)`. It records each
  tolerance and delta-to-tolerance ratio and still fails closed above the
  bound. On the failing sample, the former check has four failures, the new
  check has zero, the maximum delta/tolerance ratio is `0.5`, and retention
  remains identical. Candidate tensors, `0.25` filter, descriptors, losses,
  optimizer, seeds, epochs, pair freeze, evaluator and gates are unchanged.
- This repair follows PyTorch's numerical-accuracy guidance: floating-point
  addition is non-associative, so mathematically identical reductions are not
  guaranteed to be bitwise identical
  (https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
  NumPy defines `spacing(x)` as the distance to the nearest adjacent
  representable value, which is the scale-aware ULP used by the audit
  (https://numpy.org/doc/stable/reference/generated/numpy.spacing.html).
  These sources justify only the diagnostic tolerance; they supply no BTXRD
  performance claim or model-selection signal.
- Error evidence
  `paired_version3_error_audit.json` has canonical-LF SHA-256
  `6ede5a65b965fb36f640e6b2c3739c5da1848d697d3180cefda6ec55f1319073`.
  Correction-v9 is frozen in isolated source commit
  `911f853bac788d0c9ac1cd63e5a4408b2d6dae1a`; protocol, corrected audit
  source and test canonical-LF hashes are
  `79a4d076cb0a48ef082385897c01059b7a16cfadcdaffca0090ba4c3a4583a06`,
  `a4841ee11f7cb61bde56b7681caea8caf0fa4ffd085e9b9ddff908c8d95bebe0`
  and
  `2d1d4efeee9688be28a4fc0d2dca4515eff9fc45090a4c4dc5c10a1374ac9d57`.
  The isolated branch is pushed.
- Version-4 wrapper SHA-256 is
  `b404f85b68fba8267bf8916f7fc14a84796b372a8219465049f163933ad46a49`;
  metadata remains
  `1c00d8ba7547ac4009f0bcb3a9e59588590752b73dbb0d337dfc7ea64d2a44dd`.
  Exact remote Git/protocol/eight-source binding, wrapper `py_compile`,
  direct mount, safe archive fallback, traversal rejection, two changed
  static tests and the failing-sample numerical reproduction pass.
  Prelaunch audit
  `rad_dino_mask_bag_mil_descriptor_geometry_paired_v4_wrapper_audit.json`
  has canonical-LF SHA-256
  `f6a915be443ec9f5e6d88798569eea37f8cf28f37efeaeb516ffed2e153c1b50`.
  The next valid action is to commit/push this error evidence, then submit
  paired version `4`; R1/R2/S1 and any consumer remain gated on its terminal
  paired audit.
- Error evidence and the prelaunch audit were committed and pushed on
  `research-wsss-improvement` at
  `0bbc98fa1ddb67353d6fdd8fb9254a5c80e2d8a4`. Kaggle then accepted paired
  recovery version `4` from the correction-v9 wrapper at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-geometry-paired-v1.
  Invalid free-form tags were ignored; metadata still requests
  `NvidiaTeslaT4`, and the runner still requires exactly two physical T4
  devices with a real convolution on both. No monitor was created and no
  launch-time status poll was made. Validation GT, consumer training and
  BTXRD test remain locked.

### Paired geometry version 4 split-qualified audit-artifact correction

- One terminal status check found paired recovery version `4` at `ERROR`.
  Direct output and Kaggle console logs were retrieved into a new ignored
  temporary directory before diagnosis. Checkout `911f853...`, the T4x2
  guard, focused preflight (`34 passed`), whole-repository preflight
  (`223 passed, 1 skipped`), physical recovered-train-gallery verification
  and prediction-first generation of all `371/371` validation candidate
  payloads passed. Validation candidate/pseudo manifest SHA-256 values are
  `e2f6f30033ca8a3b71349ac2377116555173a20b4eb06be731b4fc6ff59a8d04`
  and
  `38d7ae19ef57a2abd068bd0baa5f89600a4fbb54254fc47df3826dd7b87f3793`.
- The complete GT-blind train fractional-grid-mass audit also passed across
  `2,981` images and `174,669` candidates. It retained `173,376` candidates,
  verified exact original/flip retention-vector equality, observed maximum
  absolute mass delta `0.0001220703125` and maximum delta/tolerance ratio
  `0.75`. The summary and CSV SHA-256 values are
  `7611849bf3ab049ab8de0da06f06e27101557e0107cb7259c95269f0ae8b023b`
  and
  `f105005e193de33016aa262c3cf8b6167e994e5d8963f6cb79e0142d0e64d6fc`.
  This independently confirms that correction-v9's ULP-bound numerical audit
  works as intended.
- Failure occurred only after that audit returned: the auditor writes
  `train_fractional_grid_mass_summary.json`, while the paired orchestrator
  tried to read the nonexistent legacy name `summary.json`. Validation
  fractional audit, optimizer construction, both geometry arms, prediction
  freeze and validation-GT import were never reached. There is no scientific
  result; `consumer_trained=false` and `test_evaluated=false`.
  Machine-readable error evidence is
  `artifacts/kaggle/rad_dino_mask_bag_mil_descriptor_geometry_v3/paired_version4_error_audit.json`,
  SHA-256
  `e235b1de0ce4650e8aa161fac0244f9268d70f5402df74462f81e7fda35a843e`.
- Correction-v10 changes only artifact wiring: the runner now reads
  `<split>_fractional_grid_mass_summary.json` and hashes the exact
  `train_...`/`val_...` filenames. A static regression test rejects the
  obsolete path; `py_compile`, `git diff --check` and the two focused suites
  report `5 passed`. The isolated source branch is pushed at
  `4758701b5a8d8bb6b24d1d376ff75ff12d000379`; correction protocol, runner
  and test SHA-256 values are
  `b1a64a9e9ff4fddf7fa0f0251b12874d632bad1b4fbad3b4e2f777dd60e98d92`,
  `3be46915557634363972805d9253fc00288d6cdf27cf8bd7f0b372a5534af314`
  and
  `656736b3fe54bca9be8c63ee7bd93d25a5cd4f017e4dca6ad2a06c17e362603a`.
- Version-5 wrapper and unchanged metadata SHA-256 values are
  `1f0a6dcda07ca03dcdb8556bea8fb1251fcc846732209e739cb46d7c95ed0d2b`
  and
  `1c00d8ba7547ac4009f0bcb3a9e59588590752b73dbb0d337dfc7ea64d2a44dd`.
  Exact remote commit/protocol/eight-source binding and wrapper
  `py_compile` pass; prelaunch audit
  `rad_dino_mask_bag_mil_descriptor_geometry_paired_v5_wrapper_audit.json`
  reports `PRELAUNCH_PASS`. This correction introduces no new scientific
  technique. The numerical rationale remains the already cited PyTorch
  numerical-accuracy guidance
  (https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) and
  NumPy `spacing` definition
  (https://numpy.org/doc/stable/reference/generated/numpy.spacing.html);
  neither source supplies a BTXRD performance claim.
- Error evidence, correction-v10 prelaunch audit and this log were committed
  and pushed on `research-wsss-improvement` at
  `d47b9504e6431ccf50bfec707d16560a1359c6c2`. Kaggle then accepted paired
  recovery version `5` from checkout `4758701...` and wrapper
  `1f0a6dcd...`. Invalid free-form tags were ignored; the attached private
  gallery dataset, T4x2 metadata and scientific contract remain unchanged.
  No launch-time status poll or monitor was created. Validation GT, consumer
  training and BTXRD test remain locked.

### Paired geometry version 5 scientific recovery and selector-campaign activation

- Version `5` reached terminal `ERROR`, but unlike versions `1`-`4` it had
  already completed every scientific operation. Direct Kaggle logs bind
  checkout `4758701b5a8d8bb6b24d1d376ff75ff12d000379`, focused preflight
  `34 passed`, whole-repository preflight `223 passed, 1 skipped`, two real
  Tesla T4 devices, RAD-DINO DataParallel, the exact `2,981+371` physical
  candidate payloads and both 16-epoch arms. Both `371`-map prediction cohorts
  were frozen before either evaluator imported validation segmentation, and
  both complete `371/184/187` evaluations finished. The direct-log and
  execution-log SHA-256 values are `2fd4791c...` and `e9659939...`;
  `consumer_trained=false` and `test_evaluated=false` throughout.
- The only failure was the first final comparison call. The runner supplied
  bootstrap seed `20261201`, while the comparator already failed closed unless
  the predeclared `10,000/20261101` contract was used. Correction-v11 therefore
  changes all three orchestration calls to `20261101` and authorizes only
  post-hoc comparison of exact hash-bound frozen per-image tables. It does not
  rerun training, prediction or evaluation, does not reopen GT, and changes no
  metric, replicate count or selector. The source fix and regression test
  report `6 passed`; isolated source commit `d9263f7` is pushed on
  `codex/geometry-v3-v2`. Correction-v11 canonical-LF SHA-256 is
  `34cfde33248487cde270f8be46219d01038c7b0c5879909f43783365f3409308`.
- All `371+371` physical prediction maps were retrieved and rehashed against
  their immutable manifests with zero mismatch. The pair-freeze SHA-256 is
  `610a675e...`; the local ordered 742-map hash-manifest SHA-256 is
  `24f644dc...`. Legacy and corrected checkpoint SHA-256 values are
  `b6d6803b.../58b82642...`; per-image evaluation SHA-256 values are
  `f6933bd4.../a26143d0...`. The train and validation fractional-mass audits
  cover `174,669/20,967` candidates, retain `173,376/20,778`, preserve exact
  original/flip retention vectors and have maximum delta/tolerance ratios
  `0.75/0.75`.
- The same-gallery legacy selected Dice is
  `0.21949219/0.08043121/0.34656449/0.43741027`
  overall/small/medium/large. Square-corrected geometry-v3 reaches
  `0.24548239/0.11708058/0.37713552/0.38941265`, so corrected-minus-legacy is
  `+0.02599020/+0.03664936/+0.03057103/-0.04799762`. The exact paired
  complete-group bootstrap CI95 values are respectively
  `[-0.00181951,0.05505616]`, `[0.00114346,0.07770157]`,
  `[-0.01280067,0.07157508]` and `[-0.18019991,0.06424610]`. Thus the rare
  small-lesion improvement is statistically positive under the frozen paired
  analysis; overall and medium means improve, while the large mean decreases
  with a very wide interval over only `18` images.
- The causal interpretation is now fixed. Square-corrected geometry-v3 becomes
  canonical because it repairs a proven coordinate-frame error and has a
  positive small-lesion paired interval; validation performance did not choose
  the correction. The large decrease is retained as a required selector
  diagnostic, but its `n=18` interval does not justify reintroducing wrong
  geometry. A plausible mechanism, explicitly treated as an inference rather
  than a measured fact, is that removing square-padding distortion restores
  precise token/proposal alignment most strongly for small masks, whereas
  large masks pool more heterogeneous anatomy and remain sensitive to which
  proposal the descriptor scorer ranks first.
- The exact corrected mask-bag configuration still fails its operational gate:
  goal Dice remains
  `0.34024039/0.17895493/0.51244178/0.49370336`, and no consumer is
  authorized. Crucially, the unchanged single-candidate gallery oracle is
  `0.40907553/0.22274949/0.59414708/0.64182537`, above every goal. This
  simultaneously preserves the small gain and confirms that large support was
  not destroyed. The active causal problem remains descriptor/aggregation/
  selector regret, not candidate generation.
- The finite post-geometry campaign is therefore activated rather than reset:
  build one hash-bound shared cache that bit-reproduces all corrected winners
  and maps, then run `R1` normal-prototype residual, `R2` frozen RAD-DINO local
  affinity residual and matched `S1` standard-versus-family-balanced SmoothMax.
  Each arm sees only clean-train image labels, keeps the complete v3 scorer
  frozen and writes all-candidate scores/maps before validation GT. A failed
  arm advances to the next selector mechanism; it does not reopen proposal
  generation. An arm may be composed later only if it independently reduces
  selected-to-oracle regret without sacrificing the small subgroup.
- Research mechanisms and sources supporting this decision are recorded for
  the report:
  - Ilse, Tomczak and Welling, *Attention-based Deep Multiple Instance
    Learning*, ICML 2018, formalize trainable permutation-invariant bag
    aggregation under bag/image labels. This supports the residual MIL
    selector but not any inferred pixel target:
    https://proceedings.mlr.press/v80/ilse18a.html.
  - Zaheer et al., *Deep Sets*, NeurIPS 2017, give the invariant-set-function
    basis for treating an unordered candidate gallery and motivate S1's
    within-family then across-family aggregation:
    https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html.
  - Ahn and Kwak, *Learning Pixel-Level Semantic Affinity With Image-Level
    Supervision*, CVPR 2018, and Ru et al., *Learning Affinity From Attention*,
    CVPR 2022, show that local semantic affinity can recover spatial support
    missed by discriminative activation. BTXRD transfers only local
    token/proposal cohesion and boundary contrast into R2; it does not copy
    their natural-image propagation or use segmentation labels:
    https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html,
    https://openaccess.thecvf.com/content/CVPR2022/html/Ru_Learning_Affinity_From_Attention_End-to-End_Weakly-Supervised_Semantic_Segmentation_With_Transformers_CVPR_2022_paper.html.
  - Zhang et al., *Frozen CLIP: A Strong Backbone for WSSS (WeCLIP)*, CVPR
    2024, supports a frozen foundation backbone plus a lightweight learned
    decoder/refinement path. The bounded BTXRD adaptation is a zero-initialized
    residual on frozen radiology descriptors, not a CLIP/text transfer:
    https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html.
  - Roth et al., *Towards Total Recall in Industrial Anomaly Detection
    (PatchCore)*, CVPR 2022, motivate representative nominal patch memory for
    localization. R1 conservatively transfers normal-only prototype distance;
    industrial anomaly accuracy is not assumed to transfer to BTXRD:
    https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html.
  - Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning Network*,
    CVPR 2021, motivate the predeclared S2 fallback in which a critical
    candidate is related to every other candidate. S2 remains later than
    R1/R2/S1 because a wrong critical instance can amplify confirmation error:
    https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html.
- Complete machine-readable evidence is frozen at
  `artifacts/kaggle/rad_dino_mask_bag_mil_descriptor_geometry_v3/paired_version5_posthoc_audit.json`.
  Its canonical-LF SHA-256 is
  `f7e48abc9f377c41010def9fcca1ffed06c2caebd699c75756ecbc8bff2807a1`.
  The exact corrected arm is rejected, geometry-v3 is canonical, selector
  cache packaging is in progress, and validation GT, consumer and BTXRD test
  remain locked.

### Shared selector-cache v1 freeze and T4x2 prelaunch

- The exact version-5 validation gallery and canonical corrected baseline were
  recovered from direct Kaggle output rather than regenerated. All `371`
  candidate NPZ hashes, `371` pseudo-mask hashes and `371` corrected map hashes
  match their frozen manifests. They are packaged only for lossless transport
  as `val_candidates.zip` and `square_corrected_baseline.zip`, SHA-256
  `426fbe9c.../8857eb6d...`, in private dataset
  `itsthang333/btxrd-mask-bag-selector-baseline-v1` version `1`. Kaggle reports
  the dataset `ready`. Its transport-audit SHA-256 is `9377ce5b...`; no
  validation GT or BTXRD test artifact is included.
- Protocol `rad_dino_mask_bag_selector_cache_v1` is frozen before cache
  execution at canonical-LF SHA-256
  `ee810fd8b4e2253533b2fe145046f1ad6349f69e09745bde46b3ee4612e34489`.
  The cache must first reproduce all `371` canonical selected indices,
  candidate-TTA decisions and fp16 map hashes, with selected/bag-logit and
  probability deltas no greater than `5e-6`; any failure rejects the entire
  cache before R1/R2/S1. Only then may it serialize aligned original/flip mean
  descriptors, 24 local-affinity values, immutable candidate/family identity,
  shape and pairwise geometry. Training masks are discarded; only validation
  candidate masks are bit-packed for later post-freeze evaluation.
- A prelaunch source-closure audit expanded the initial direct hash list to all
  `18` transitive local modules imported by the cache path. No kernel used the
  narrower draft. Execution checkout
  `dd3e9f4689beb113fc7fff523a21e0bb7d5ca384` is a descendant of scientific
  source commit `61e64b38992db5e4b7414d3fb4c64b7edc39c6aa`; every remote Git blob and the
  protocol blob match the wrapper constants. The wrapper requires two Tesla
  T4 devices, performs a real convolution on each, uses RAD-DINO DataParallel
  for the heavy `2,981+371` extraction, runs focused and whole-source tests,
  and independently rehashes all `3,352` cache records after construction.
- Wrapper and metadata canonical-LF SHA-256 values are
  `4f072183b17f8c4557e550b2e5cb57ed99e42618e8387460e9721ecff6e9e79b`
  and `85a88e98...`. Local `py_compile`, JSON parsing, direct-root discovery,
  safe extraction, path-traversal rejection, exact remote source/protocol
  binding and T4x2/static contract checks pass. Prelaunch audit
  `rad_dino_mask_bag_selector_cache_v1_wrapper_audit.json` has canonical-LF
  SHA-256
  `9333c887e3783ac528c124e11a0d8e07dcc75288ad61466406684c92b7599e63`
  and reports `PRELAUNCH_PASS`.
- This cache adds no scientific mechanism and reads no segmentation GT. Its
  purpose is to make the already sourced R1 normal-memory, R2 local-affinity
  and S1 invariant family-pooling experiments consume one identical,
  geometry-corrected representation. Heavy extraction will run only on Kaggle
  T4x2; consumer training and BTXRD test remain locked.
- Kaggle accepted selector-cache kernel version `1` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1.
  Metadata requests `NvidiaTeslaT4`, and runtime still fails closed unless two
  physical Tesla T4 devices each complete a real convolution. No launch-time
  status poll or monitor was created. R1/R2/S1 remain gated on terminal cache
  reproduction audit; validation GT, consumer and BTXRD test remain locked.

### Shared selector-cache version 1 split-serialization correction

- A single terminal status check found selector-cache kernel version `1` at
  `ERROR`. Direct Kaggle output contained only the console log; its SHA-256 is
  `c07d5ed0f3eda11d475e98ff8b456172639e67883186b98ad1a7e8009473f049`.
  Exact checkout `dd3e9f4...`, scientific-source ancestry, package installation,
  the two-T4 real-convolution guard, transport safety checks, validation and
  baseline archive extraction, and train-gallery discovery all passed.
- Failure occurred at split discovery, before dataset-root verification, model
  download, pytest, image opening, descriptor extraction, baseline
  reproduction or cache serialization. The wrapper searched Kaggle inputs for
  the frozen Windows-CRLF split hash `85511ee1...`, but the Git-derived input
  exposes the same manifest in canonical LF form `43662d5d...`. No radiograph,
  validation GT or BTXRD test was opened; no scientific result exists and
  R1/R2/S1 remain unauthorized. Machine-readable evidence is
  `artifacts/kaggle/rad_dino_mask_bag_selector_cache_v1/kernel_version1_error_audit.json`.
- The version-2 correction is wrapper-only: read the split from the already
  hash-bound checkout, require exact LF hash `43662d5d...` and zero carriage
  returns, reconstruct CRLF exactly once, then require frozen hash
  `85511ee1...`. Local reconstruction reproduces that hash exactly and wrapper
  `py_compile` passes. Protocol, 18-module scientific source closure, gallery,
  baseline, cache schema, image-label-only contract and selector campaign are
  unchanged. Corrected wrapper SHA-256 is `25c8a98a...`; prelaunch audit
  `rad_dino_mask_bag_selector_cache_v1_kernel_v2_wrapper_audit.json` reports
  `PRELAUNCH_PASS`.
- This is cross-platform byte-serialization repair, not a research mechanism;
  it adds no technique/paper claim. The scientific sources supporting R1/R2/S1
  remain the Attention MIL, Deep Sets, AffinityNet/AFA, WeCLIP, PatchCore and
  DSMIL references recorded immediately above. The only valid next step is the
  same cache rerun; validation GT, consumer training and BTXRD test remain
  locked.

- After a complete-log reread and a clean-worktree audit, branch
  `research-wsss-improvement` was confirmed synchronized with
  `origin/research-wsss-improvement` at
  `ea2e1e0b1c3b2d4d45d9110f4916b8569bd1f333`. The exact corrected wrapper,
  unchanged metadata, frozen protocol, version-2 prelaunch audit and
  version-1 error audit re-matched SHA-256 values
  `25c8a98ae8f50744a8810d74dc5d66b91801042fe05687f599b13bc640015649`,
  `85a88e983bfc4707e7b3abfb1222ebc44a80c026739cdb8d757c0a4b006f6d54`,
  `ee810fd8b4e2253533b2fe145046f1ad6349f69e09745bde46b3ee4612e34489`,
  `ee203d9aaaa43fbb8d8d09323b2bb3643c488d0e293c5ab1f8ac6869ab927b1a`
  and
  `430c05804ae413d50f096073823be69cc25a330659a6dd75b67ec4bc0df8873e`.
  Kaggle then accepted selector-cache kernel version `2` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1.
  No launch-time status poll or monitor was created. The scientific source,
  cache protocol, gallery, baseline and selector campaign remain unchanged;
  R1/R2/S1, validation GT, consumer training and BTXRD test remain locked
  pending the terminal cache reproduction audit.

### Shared selector-cache version 2 Kaggle-mount correction

- A single continuation status check found selector-cache kernel version `2`
  at terminal `ERROR`. Direct Kaggle output was downloaded into a new ignored
  directory and read in full; the sole console log has SHA-256
  `dd980b03fef452563da2800d8a8744df3a8f03c61b49fb7acd9baeed88f59a2c`.
  Exact checkout `dd3e9f4...`, scientific-source ancestry, runtime installation,
  the T4x2 real-convolution guard, transport safety/unpacking, train-gallery
  discovery, canonical LF split verification and exact CRLF reconstruction
  all passed.
- Failure was `RuntimeError: BTXRD dataset root is missing:
  /kaggle/input/btxrd-raw/BTXRD` at about 97.49 seconds. It occurred before
  model download, pytest, image opening, descriptor extraction, baseline
  reproduction or cache serialization. Thus no radiograph, validation GT or
  BTXRD test was opened; no cache or scientific result exists, no consumer was
  trained, and R1/R2/S1 remain unauthorized. The machine-readable error audit
  is
  `artifacts/kaggle/rad_dino_mask_bag_selector_cache_v1/kernel_version2_error_audit.json`,
  canonical-LF SHA-256
  `d3da6b9c783652a2e752ff4db55f2b91d0ec51354c6c302d8c0ef655eb0bbf46`.
- Root cause is an implementation-only Kaggle mount convention mismatch. The
  attached dataset is still exactly `itsthang333/btxrd-raw`, whose file listing
  begins under `BTXRD/Annotations/...`, but the current runtime mounts it at
  `/kaggle/input/datasets/itsthang333/btxrd-raw/BTXRD`; earlier successful
  paired-geometry and independent-reproduction logs record that exact path.
  The version-2 wrapper instead required the legacy short path
  `/kaggle/input/btxrd-raw/BTXRD`.
- The authorized version-3 repair changes only wrapper dataset-root discovery:
  enumerate `BTXRD` directories beneath `/kaggle/input`, resolve/deduplicate
  them, require both `images/` and `Annotations/`, accept exactly one valid
  root, and fail closed otherwise. A synthetic current-mount test and an
  ambiguous-two-root rejection test pass, as do `py_compile`, JSON parsing and
  `git diff --check`. Corrected wrapper SHA-256 is
  `0ac60591248d385bddf8aff6f26450feb6f40e0013b3c36f3aa993e1c1632e5e`;
  metadata remains `85a88e98...`, protocol remains `ee810fd8...`, and the
  scientific checkout/source remain `dd3e9f4.../61e64b3...`. Prelaunch audit
  `rad_dino_mask_bag_selector_cache_v1_kernel_v3_wrapper_audit.json` reports
  `PRELAUNCH_PASS` and has canonical-LF SHA-256
  `beb136fa01d5dafa6ad9b91b32932092b657eadf03bb5a745edcc474d1b9f391`.
- This path repair adds no research mechanism or new literature claim. The
  frozen Attention MIL, Deep Sets, AffinityNet/AFA, WeCLIP, PatchCore and DSMIL
  basis, candidate gallery, corrected baseline, cache schema and finite
  R1/R2/S1 campaign are unchanged. The only authorized next action is the same
  cache rerun on Kaggle T4x2; validation GT, consumer training and BTXRD test
  remain locked.
- From a clean, origin-synchronized worktree at commit
  `e7eb6c4c77f60f85144a521e8c475d2135a1cbe0`, the wrapper, metadata,
  protocol and version-3 prelaunch audit re-matched exact SHA-256 values
  `0ac60591248d385bddf8aff6f26450feb6f40e0013b3c36f3aa993e1c1632e5e`,
  `85a88e983bfc4707e7b3abfb1222ebc44a80c026739cdb8d757c0a4b006f6d54`,
  `ee810fd8b4e2253533b2fe145046f1ad6349f69e09745bde46b3ee4612e34489`
  and
  `beb136fa01d5dafa6ad9b91b32932092b657eadf03bb5a745edcc474d1b9f391`.
  Kaggle accepted selector-cache kernel version `3` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1.
  No launch-time status poll or monitor was created. R1/R2/S1, validation GT,
  consumer training and BTXRD test remain locked pending terminal cache
  reproduction audit.

### R1 readiness audit while selector-cache version 3 runs

- The single scheduled continuation check found selector-cache kernel version
  `3` still `RUNNING`. No repeat poll or monitor was created, and the running
  kernel was not modified. R1/R2/S1 remain locked because no terminal cache
  freeze or reproduction audit exists yet.
- A local source-only R1 readiness run exposed one contradictory unit-test
  assertion in `tests/test_mask_bag_residual_objective.py`. The implemented
  and documented objective applies original/aligned-flip consistency to the
  final candidate logits after adding the learned residual. The old test used
  different valid residuals (`2` versus `4`) but incorrectly required zero
  consistency; the exact Smooth-L1 probability discrepancy is positive. This
  is a test-oracle error, not evidence that the objective should ignore the
  residual signal.
- The correction changes only that test. It now requires positive consistency
  for the deliberately different valid residuals, then changes every invalid
  base/residual value and proves both consistency and residual drift remain
  identical. The scientific objective source remains byte-identical at
  SHA-256
  `7b176c955408fb14b59e0c901243be12791f1d68540484ab06e99f2ba14b92df`;
  the corrected test SHA-256 is
  `65f7831c99c5acd2cf619d4f37e2540114ceb1214981095e00e00974adca3fcc`.
  No loss, feature, cohort, K candidate, selector rule or numerical
  hyperparameter changed.
- Under the available NumPy/Torch runtime, the corrected objective suite is
  `4 passed`. The complete focused R1/cache suite is `43 passed, 4 failed`
  under Python 3.9 solely because `zip(strict=...)` was introduced in Python
  3.10; the four tracebacks are all that interpreter boundary. Re-running with
  a local compatibility shim only for the unsupported `strict` keyword gives
  `47 passed`, and the whole repository gives `333 passed`. This shim is local
  diagnostic evidence only: the unmodified whole suite remains mandatory in
  Kaggle's Python 3.12 runtime before any cache or arm execution is accepted.
- The running version-3 cache still uses its already frozen checkout and is
  unaffected by this later test correction. Its eventual terminal boundary
  must be audited directly before deciding whether a new implementation-only
  checkout is needed. No cache record, prototype, adapter or validation
  prediction was created locally; validation GT, consumer training and BTXRD
  test remained untouched.
- Machine-readable R1 source-readiness evidence is frozen at
  `artifacts/research_protocols/r1_normal_prototype_readiness_audit_v1.json`,
  canonical-LF SHA-256
  `d70573d44ff71c6207acaf7030cf0eed595b16e123cb8b8935099178d654bc9c`.
  It binds the corrected source commit `d66c899...`, exact R1 source/test
  hashes, the finite `K={8,16,32}` five-fold contract, baseline GT-blind
  count-correlation magnitude `0.48137777593654113`, the `+0.02` shortcut
  guard and all safety locks. Its status is
  `SOURCE_READINESS_PASS_PENDING_SELECTOR_CACHE_GATE`, not launch approval.

### Shared selector-cache version 3 preflight-test failure

- The next single status check found selector-cache kernel version `3` at
  terminal `ERROR`. Direct Kaggle output was downloaded once into a new
  ignored directory and read completely. Its sole console log has SHA-256
  `de0bcb43ab2a64514de9296648f3a8c5fc79cab56044d6cffcf74f8e9508a591`.
- Exact checkout `dd3e9f4...`, scientific ancestry, runtime installation, the
  T4x2 guard, transport safety/unpacking, train-gallery discovery, LF-to-CRLF
  split reconstruction, unique current-mount BTXRD-root discovery and exact
  RAD-DINO snapshot preparation all passed. The focused cache/geometry suite
  then reported `44 passed` under Python 3.12 and PyTorch 2.5.1+cu121.
- The whole-repository preflight reported `331 passed, 1 failed, 1 skipped`
  with four warnings. Its sole failure was the already identified
  `test_only_valid_residuals_contribute_to_drift`: observed consistency
  `0.0036104973405599594` versus the contradictory expected `0.0`. The wrapper
  stopped before invoking `build_mask_bag_selector_cache.py`, BTXRD image
  opening, descriptor extraction, baseline reproduction or serialization.
  Consequently no cache or scientific result exists; validation GT, consumer
  and BTXRD test remain untouched, and every selector arm stays locked.
- Machine-readable evidence is
  `artifacts/kaggle/rad_dino_mask_bag_selector_cache_v1/kernel_version3_error_audit.json`.
  Its canonical-LF SHA-256 is
  `16364b1679c2b957c76a135bd49ea11f6ed17d18384953775d5f9d315bd1cee5`.
  The only authorized correction is the test-only commit `d66c899...`, whose
  corrected test SHA-256 is `65f7831c...`; the objective source remains
  byte-identical at `7b176c95...`. The next wrapper must bind both that exact
  checkout/test hash and the unchanged cache source/protocol, then rerun the
  full preflight before the unchanged builder. No scientific mechanism or
  literature claim changes.
- The version-4 wrapper now checks out exact correction commit `d66c899...`
  and adds the corrected test SHA-256 `65f7831c...` to its fail-closed source
  table. The original cache builder and all cache/scientific modules remain
  byte-identical, as do protocol `ee810fd8...`, metadata `85a88e98...`, gallery,
  baseline and schema. Wrapper SHA-256 is
  `c4157072b4bf3ff74aa1793c8aef5ed0ad6e649b2ad557359d85a30199312eb1`.
  Local `py_compile`, JSON parsing, Git ancestry/blob equality, canonical-LF
  hashes and cache-surface identity pass. Prelaunch audit
  `rad_dino_mask_bag_selector_cache_v1_kernel_v4_wrapper_audit.json`, SHA-256
  `72dd6b47a361dff265d07081e77521bc55d23a8bc8c9f5038d91a6bef6def9d0`,
  reports `PRELAUNCH_PASS`. The wrapper will still rerun the unmodified Python
  3.12 whole suite before cache construction; R1 remains unauthorized until
  terminal reproduction evidence passes.
- From clean, origin-synchronized commit
  `26bc44d530e0c6627676934bfa471874c451ae49`, the wrapper, metadata, protocol
  and version-4 prelaunch audit re-matched exact SHA-256 values
  `c4157072b4bf3ff74aa1793c8aef5ed0ad6e649b2ad557359d85a30199312eb1`,
  `85a88e983bfc4707e7b3abfb1222ebc44a80c026739cdb8d757c0a4b006f6d54`,
  `ee810fd8b4e2253533b2fe145046f1ad6349f69e09745bde46b3ee4612e34489`
  and
  `72dd6b47a361dff265d07081e77521bc55d23a8bc8c9f5038d91a6bef6def9d0`.
  Kaggle accepted selector-cache kernel version `4` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1.
  No launch-time poll or monitor was created. Validation GT, R1/R2/S1,
  consumer training and BTXRD test remain locked pending terminal cache
  reproduction audit.

### R1 normal-prototype arm protocol freeze while cache v4 runs

- The single scheduled continuation check found selector-cache version `4`
  still `RUNNING`; no repeat poll or monitor was created. R1 was not launched
  and consumed no cache record. To avoid post-cache scientific choices, the
  complete R1 arm contract is now frozen before the cache result at
  `artifacts/research_protocols/rad_dino_mask_bag_normal_prototype_r1_v1.json`,
  canonical-LF SHA-256
  `dded9c638e142576fedf0ae4c8102fdf64198744a4949707865e50b7081f312b`.
- The protocol fixes the already declared normal-only hierarchical spherical
  prototypes, four normality features, finite `K={8,16,32}`, five whole-group
  held-out folds, one-standard-error image-BCE rule, frozen absolute
  candidate-count/probability Spearman `0.48137777593654113` plus `0.02`
  shortcut guard, 16 final-only epochs, optimizer/loss weights, all seeds,
  exact fold census and T4x2 `8/7` job allocation. Only image-level labels may
  enter prototype/residual fitting or K selection.
- Cache binding is intentionally a one-time execution input rather than an
  unfrozen scientific choice: a launch wrapper may bind only a terminal direct
  cache output whose freeze and wrapper audit pass all `3,352` physical-record,
  `371` baseline-index/map reproduction, cohort and safety checks. Until that
  SHA exists, the protocol status remains
  `PREDECLARED_WAITING_FOR_SELECTOR_CACHE_BINDING` and launch is forbidden.
- All `371` validation candidate scores/maps must be physically hashed and
  prediction-frozen before the separate evaluator can receive the GT-derived
  baseline table `a26143d0...` or open validation annotations. Evaluation then
  uses the fixed `371/184/187`, `94/72/18`, complete misses and 10,000
  complete-group bootstrap contract; consumer authorization requires all four
  operational goals simultaneously. BTXRD test remains locked.
- This protocol adds no new technique or source. It materializes the R1 row
  already supported and logged by TPMIL and Attention MIL, and binds the exact
  source/test hashes already covered by the R1 readiness audit. No prototype,
  adapter, prediction, validation GT access or consumer training occurred.
- A complete but deliberately non-launchable R1 Kaggle wrapper is prepared in
  ignored staging at
  `tmp/kaggle/mask_bag_normal_prototype_r1_v1/`. Wrapper/metadata SHA-256 are
  `4312a1410c1cf83894359895907c94840f55273f78b8593bc728328f1cb4688e`
  and `e3144006ca18afdc7376b11545d8dcd9675c8c2eecc12545b3c2bf5393bad472`.
  It attaches only the sanitized baseline dataset and selector-cache kernel
  output—no raw BTXRD, validation GT or test source.
- `CACHE_BINDING_READY=False` and two explicit pending cache hashes cause
  `main()` to fail before filesystem creation, Git clone, input discovery or
  CUDA access; this dynamic fail-closed test passes. After a cache gate pass,
  the only permitted edits are those two audited cache hashes and the boolean.
  Scientific source, protocol, K grid, losses, seeds and runtime controls may
  not change.
- The staged wrapper binds checkout `3647997...`, scientific source
  `d66c899...`, protocol `dded9c63...`, all 24 protocol source/test hashes,
  reconstructed split, baseline transport/checkpoint, exact cache freeze plus
  wrapper audit, 3,352-record and 371-map reproduction gates, T4x2 real work,
  focused/whole tests, 15 OOF folds, three aggregates, 371 all-candidate score
  payloads/maps and the final prediction freeze. It never invokes the
  evaluator.
- Machine-readable readiness audit
  `rad_dino_mask_bag_normal_prototype_r1_v1_wrapper_readiness_audit.json`,
  SHA-256
  `9f90668a84ba7320b42a3e29d48de98c3c81b83337cc5400d1f24b96d39ff56d`,
  has status `WAITING_FOR_ACCEPTED_SELECTOR_CACHE_BINDING`, not prelaunch
  approval. No R1 kernel was pushed; cache, GT, consumer and test locks remain.

### Shared selector-cache version 4 affinity-axis failure

- The next single status check found selector-cache version `4` at terminal
  `ERROR`. Direct output was downloaded once into a new ignored directory and
  read completely; its sole log has SHA-256
  `625aedc2bf442d5aaedb6096353844bc0fcd830d733f6a59efbf127db4e4460a`.
- Checkout/source/protocol, T4x2, transport, galleries, split, modern BTXRD
  root and RAD-DINO snapshot all passed. Kaggle Python 3.12 reported `44`
  focused tests passing and the unmodified whole repository reported
  `332 passed, 1 skipped` with four warnings. This closes the version-3 test
  correction and proves the builder reached real T4x2 cache execution.
- The builder then failed during the first clean-train descriptor-cache pass,
  inside the additional 24-value affinity summary. `vector_norm_squared` has
  shape `[B,N,L]`, but inclusive affinity divided it by `[B,N]` mass squared.
  PyTorch therefore aligned the trailing candidate count `27` against the
  layer count `3` and raised the exact `3 versus 27` dimension error. A local
  synthetic tensor with `B=2,L=3,N=27` reproduces the same traceback.
- Existing tests used `N=L=1` or `N=L=2`, hiding the missing layer-axis
  singleton and, in the latter case, failing to detect candidate-denominator
  mixing. The mathematically intended exact weighted statistic requires
  `mass.square()[:, :, None]`, consistent with the already explicit
  off-diagonal denominator immediately below it. A new test must use multiple
  batches and `N != L` with candidate-specific weights.
- The failure occurred before a complete train cache, validation extraction,
  baseline reproduction, cache serialization or freeze. Clean-train
  radiographs were opened/encoded, but no segmentation GT was read; validation
  radiographs/GT and BTXRD test were not opened, no consumer was trained, and
  no scientific result or selector authorization exists. Machine-readable
  evidence is
  `artifacts/kaggle/rad_dino_mask_bag_selector_cache_v1/kernel_version4_error_audit.json`.
  Its canonical-LF SHA-256 is
  `2f82d7dcfbe2f8df2a778dd812469e579ae8ff7f88c1576e83d928f2c5cf04af`.
- Because this changes one source byte path covered by the frozen cache
  protocol, the correction must not masquerade under the old source/protocol
  hash. It will be recorded as an immutable implementation correction that
  references parent protocol `ee810fd8...`, binds the corrected source/test,
  and leaves gallery, geometry, feature definition, baseline and all selector
  science unchanged. The prepared R1 wrapper remains unbound and unlaunched.
- The source correction adds exactly the missing singleton in the inclusive
  denominator: `[B,N] -> [B,N,1]`; the off-diagonal formula and every feature
  definition remain unchanged. Corrected affinity source SHA-256 is
  `400f018f6181c740d429dbf0e7f2d1de501e3845232f59a7862d4b779c673348`.
  The new regression uses `B=2,L=3,N=4`, candidate-specific weights and an
  explicit weighted-token manual reference, so neither equal dimension sizes
  nor broadcasting can hide candidate/layer mixing. Corrected test SHA-256 is
  `3650cb415b81ddaea84b8be8284ebb8b26539c8decc593e7271afc8a38cf7215`.
- The six affinity tests pass. The exact former Kaggle shape
  `B=2,L=3,N=27` now returns finite `[2,27,24]` features. The entire local
  repository reports `334 passed` with the previously documented Python-3.9
  compatibility shim only for `zip(strict=...)`; Kaggle Python 3.12 must still
  rerun the unmodified focused and whole suites. No heavy local extraction,
  cache, validation prediction, GT access, consumer or test evaluation was
  performed.
- The correction is frozen separately at
  `artifacts/research_protocols/rad_dino_mask_bag_selector_cache_v1_affinity_broadcast_correction_v1.json`,
  SHA-256
  `be9c91b53926eda2f8bf8dba894385f03dc6accd15d3fa9646da0d1a17a635f2`.
  It references parent protocol `ee810fd8...`, error audit `2f82d7dc...`,
  corrected scientific source commit `c0e3862...` and both new source/test
  hashes. A corrected cache must record this correction protocol/source in its
  freeze; it may not claim the parent bytes unchanged.
- The R1 scientific protocol remains frozen because no R1 mechanism or input
  selection rule changes. Its earlier unbound wrapper-readiness audit is now
  explicitly superseded only for upstream cache provenance: after a corrected
  cache passes, the staged wrapper must bind the corrected cache source,
  correction protocol, freeze and wrapper-audit hashes before launch. R1 is
  still unlaunched and no cache binding exists.
- The version-5 cache wrapper now checks out `5aed03f...`, records corrected
  scientific source `c0e3862...` and correction protocol `be9c91b5...`, and
  replaces exactly the affinity source/test entries in its fail-closed hash
  table. Wrapper SHA-256 is
  `db6b461dbc9e9afe1837ecee90df68b9423ad9d60c56f1120b9da1693b658a7a`;
  metadata remains `85a88e98...`. Git comparison confirms those are the only
  changed project/test files relative to the v4 checkout.
- Version-5 prelaunch audit
  `rad_dino_mask_bag_selector_cache_v1_kernel_v5_wrapper_audit.json`, SHA-256
  `c71cdc0cec4879bff66ae74dd4ce455a49089966726a8e09fcda0465560e708e`,
  reports `PRELAUNCH_PASS`. The wrapper still requires the exact T4x2 guard,
  runs focused and unmodified Python-3.12 whole tests before compute, and must
  pass all baseline reproduction/physical-cache checks. R1 remains locked.
- From clean, origin-synchronized commit
  `9f00ffe98f5be74fe4d5d8fb81d333c8fd469348`, wrapper, metadata, correction
  protocol and version-5 audit re-matched exact SHA-256 values
  `db6b461dbc9e9afe1837ecee90df68b9423ad9d60c56f1120b9da1693b658a7a`,
  `85a88e983bfc4707e7b3abfb1222ebc44a80c026739cdb8d757c0a4b006f6d54`,
  `be9c91b53926eda2f8bf8dba894385f03dc6accd15d3fa9646da0d1a17a635f2`
  and
  `c71cdc0cec4879bff66ae74dd4ce455a49089966726a8e09fcda0465560e708e`.
  Kaggle accepted selector-cache kernel version `5` at
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1.
  No launch-time poll or monitor was created. R1/R2/S1, validation GT,
  consumer training and BTXRD test remain locked pending terminal corrected
  cache reproduction audit.

### Independent selector-cache output auditor readiness

- The single permitted status check in this continuation found selector-cache
  kernel version `5` still `RUNNING`. No second poll and no monitor was created.
  This is not a cache result, cache acceptance or selector authorization; R1,
  R2 and S1 remain unlaunched and unbound.
- A new independent, GT-blind post-download verifier is frozen at
  `project/audit_mask_bag_selector_cache_output.py`, SHA-256
  `c4d27e89d66254b4b6677230c3ed7ff08f5c5da2d78acc17d90f37d5880a7966`.
  It does not call a dataset factory, open annotations, train a model or access
  BTXRD test. It is deliberately separate from the version-5 wrapper's own
  output audit and will only run after a terminal direct output is downloaded.
- The verifier binds caller-supplied exact cache-freeze and wrapper-audit
  hashes, corrected scientific source/protocol, canonical-LF to frozen-CRLF
  split reconstruction, RAD-DINO snapshot, deterministic projection, exact
  candidate/pseudo manifests and the frozen baseline source/protocol,
  checkpoint, prediction manifest and maps. It rejects any cohort or safety
  flag mismatch and independently checks the wrapper's two-T4 names plus its
  two finite real-convolution checksums.
- For a production cache it will load and hash all `3,352` physical records,
  verify schema/dtypes/dimensions/provenance/strict candidate indices and
  independently reconstruct family IDs. For all `371` validation records it
  unpacks masks and independently recomputes the four shape features, pairwise
  IoU, containment and normalized centroid distance. It also independently
  checks all `371` baseline selections, logits/probabilities within the frozen
  `5e-6` tolerance and exact hashes of both baseline and reproduced physical
  maps. The emitted audit file is fail-if-present so a prior audit cannot be
  silently overwritten.
- Test source
  `tests/test_audit_mask_bag_selector_cache_output.py`, SHA-256
  `dedb9d160218c433ba54e417e55c15a345e5924408ece31af7fa8b9fb836cb24`,
  includes a complete synthetic train/validation cache, baseline reproduction,
  freeze, run manifest, wrapper audit, packed-mask geometry and T4x2 contract.
  `py_compile` passes; the auditor plus existing cache/cache-I/O suites report
  `20 passed`; `git diff --check` passes. The Python-3.9 compatibility shim was
  used only to ignore unsupported `zip(strict=...)` keywords in the existing
  tests, as previously documented; production Kaggle remains Python 3.12.
  The complete local repository regression then reports `339 passed` under
  the same compatibility shim.
- Machine-readable readiness evidence is
  `artifacts/research_protocols/independent_mask_bag_selector_cache_output_audit_v1_readiness.json`,
  SHA-256
  `a704b19eeaf828d52033a6d599b7e342d8222956d171ab245455af6d769b80ea`,
  with status `SOURCE_READINESS_PASS_WAITING_FOR_V5_OUTPUT`. It explicitly sets
  `cache_accepted=false` and `r1_binding_authorized=false`. A terminal cache
  mismatch will be recorded as rejection/error before any correction; a clean
  pass is required before binding the exact cache hashes into the R1 wrapper.
- No new scientific technique, paper or external URL was introduced in this
  preparation step. No heavy local compute, validation GT read, consumer
  training or BTXRD test access occurred.

### R1 corrected-cache provenance readiness

- The one status check in the next bounded continuation again found
  selector-cache kernel version `5` at `RUNNING`. No second poll and no monitor
  was created. There is still no terminal cache result or R1 authorization.
- Static re-audit of the ignored R1 staging wrapper found that its two upstream
  cache provenance constants still named the parent cache source/protocol
  `61e64b3...` and `ee810fd8...`. A successful version-5 freeze must instead
  name corrected source `c0e3862...` and correction protocol `be9c91b5...`;
  otherwise `find_cache_root()` would correctly reject the corrected cache
  before fitting. This is the upstream-provenance supersession already required
  by the correction protocol, not a change to the R1 mechanism or search space.
- Only those two known constants were updated in the ignored wrapper:
  `CACHE_SCIENTIFIC_SOURCE_COMMIT=c0e3862...` and
  `CACHE_PROTOCOL_SHA256=be9c91b5...`. The new unbound wrapper SHA-256 is
  `4b886f91aa01b15c18a1a0105db11a31f62233635e214e9bbd406e5712d05044`;
  inversely replacing exactly those two values reconstructs the previous hash
  `4312a1410c1cf83894359895907c94840f55273f78b8593bc728328f1cb4688e`.
  Metadata is unchanged at `e3144006...`.
- `py_compile` passes. Dynamic import confirms both corrected provenance
  constants, both pending terminal SHA placeholders and
  `CACHE_BINDING_READY=False`; calling `main()` raises at the first statement,
  before environment mutation, filesystem creation, clone, input discovery or
  CUDA access. No cache was consumed and the kernel was not launched.
- Successor readiness evidence
  `artifacts/research_protocols/rad_dino_mask_bag_normal_prototype_r1_v1_wrapper_readiness_audit_v2.json`,
  SHA-256
  `6287d15c250cb78761d6f4e4c1adf647b0ea252bf9f171008f60a2b1b5132854`,
  supersedes the earlier readiness audit only for upstream cache provenance.
  It binds the independent-auditor readiness SHA `a704b19e...` and permits only
  the two terminal output hashes plus `CACHE_BINDING_READY=True` after a full
  cache audit pass, followed by a final wrapper prelaunch audit.
- No R1 scientific code, protocol, hyperparameter, prediction or metric was
  changed. No new paper/URL was used; no heavy local compute, validation GT,
  consumer training or BTXRD test access occurred.

### EXP-20260731-codex-r1-normal-prototype-v1

- **Owner:** Codex main task on `research-wsss-improvement`.
- **Registered:** `2026-07-31T13:15:29.6440302Z`; registration base commit
  `72d92b812682760b1759608f02117040d4c8c1f3`; exact claim commit
  `62f6f4c96aeefb0c1977eb237bcec1fcbcbd0f55`.
- **Status:** `HOÀN THÀNH — BỊ LOẠI TẠI OOF COUNT-SHORTCUT GATE`; Kaggle
  version 3 kết thúc `ERROR` do fail-closed predeclared, không phải lỗi transport.
- **Objective/hypothesis:** accept or reject the already-running corrected
  selector cache version 5 with the frozen independent audit, then—only after
  every cache gate passes—run the predeclared R1 normal-prototype residual arm.
  R1 tests whether train-normal multi-prototype distance supplies candidate
  evidence missing from the frozen same-gallery baseline selector.
- **Non-duplicate scope:** this claim does not regenerate proposals, repeat a
  selector-cache launch, alter geometry-v3, or claim R2/S1. It takes ownership
  only of the terminal version-5 audit and the single R1 arm already frozen in
  the sections `Shared selector-cache v1 freeze and T4x2 prelaunch`,
  `R1 normal-prototype arm protocol freeze while cache v4 runs`, and their
  version-4/version-5 correction successors. No other `ĐANG LÀM` claim for
  this scope was present in the complete log at registration time.
- **Inherited evidence:** corrected geometry-v3 Dice
  `0.24548239/0.11708058/0.37713552/0.38941265`; gallery oracle
  `0.40907553/0.22274949/0.59414708/0.64182537`; cache correction source
  `c0e38628069ff3bedd4493c4ff004b75bd32e008`; cache correction protocol
  SHA-256
  `be9c91b53926eda2f8bf8dba894385f03dc6accd15d3fa9646da0d1a17a635f2`;
  cache-v5 wrapper SHA-256
  `db6b461dbc9e9afe1837ecee90df68b9423ad9d60c56f1120b9da1693b658a7a`;
  R1 protocol SHA-256
  `dded9c638e142576fedf0ae4c8102fdf64198744a4949707865e50b7081f312b`;
  independent-auditor readiness SHA-256
  `a704b19eeaf828d52033a6d599b7e342d8222956d171ab245455af6d769b80ea`;
  corrected unbound R1 wrapper SHA-256
  `4b886f91aa01b15c18a1a0105db11a31f62233635e214e9bbd406e5712d05044`.
- **Inputs/protocol:** frozen clean split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  sanitized dataset `itsthang333/btxrd-mask-bag-selector-baseline-v1` version 1;
  existing kernel `itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1`
  version 5; candidate gallery and baseline archives remain exact and contain
  no validation GT/test. R1 uses only the accepted cache plus image-level train
  labels and its frozen finite `K={8,16,32}`/15-fold group-OOF contract.
- **Compute:** terminal status/download and independent hash audit are local
  read-only work. Any R1 fitting/inference is heavy compute and may run only on
  Kaggle T4x2 under the frozen wrapper; no local heavy compute is authorized.
- **Expected output/gates:** cache acceptance requires all 3,352 record hashes,
  371 packed-mask geometry reconstructions, 371 baseline selected-index/map
  reproductions, corrected source/protocol, split/input/model/projection and
  T4x2 evidence to pass. R1 must freeze all 371 validation candidate scores,
  maps and predictions before the evaluator can receive validation GT. The
  final operational gate remains simultaneous Dice at least
  `0.34024039/0.17895493/0.51244178/0.49370336` for
  overall/small/medium/large; otherwise report rejection honestly.
- **Safety locks:** training supervision is image-level only; validation GT is
  unavailable before prediction freeze; consumer training is forbidden before
  the full operational gate passes; BTXRD test remains locked; no polling
  monitor is authorized.
- **Progress — terminal cache/download boundary:** after the claim became
  visible, the single status check found version 5 `COMPLETE`. The first direct
  output download request was terminated only by the local client timeout after
  604 seconds. Its new temp directory contains 617 files/125,689,801 bytes but
  lacks `selector_cache_freeze.json`, `wrapper_output_audit.json`,
  `selector_cache_manifest.csv`, and `run_manifest.json`; it is therefore an
  explicitly incomplete transport, not cache evidence. No incomplete file was
  accepted, no independent audit/binding ran, no GT/test was read and R1 stayed
  unlaunched. The bounded transport correction is one retry of the same frozen
  remote output into a new temp directory with Kaggle output page size 200 and
  a longer client timeout; scientific source/protocol and every gate remain
  unchanged.
- **Progress — transport retry diagnosis:** at the user's request the registered
  page-size-200 retry was stopped for inspection rather than left waiting. It
  had downloaded 1,094 files/227,858,692 bytes over about 16 minutes and still
  lacked the same root freeze/audit/manifest files, so it also remains partial
  and unaccepted. Direct inspection of the installed official Kaggle CLI shows
  that `kernels output` paginates the listing but performs one blocking
  `requests.get(...).content` download per file in a serial loop; the observed
  delay is therefore thousands-of-small-files overhead, not output size or a
  kernel failure. The bounded transport completion will reuse the exact signed
  URLs returned by the same authenticated Kaggle session-output API, validate
  existing files against remote Content-Length, download missing files with 16
  parallel workers to `.part` paths, and atomically rename each completed file.
  The full selector manifest plus independent per-record hashes remain the
  acceptance authority, so transport parallelism cannot relax a scientific or
  provenance gate. No GT/test was read and R1 remains unlaunched.
- **Progress — independent-auditor descriptor-dimension failure:** parallel
  transport completed the exact 3,731-file Kaggle output listing in 215.8
  seconds (2,415 downloaded atomically, 1,316 pre-existing files validated by
  Content-Length), for 998,064,727 remote bytes plus the direct log. Root
  freeze/wrapper/manifest/run files are now present. The first independent
  audit attempt then failed closed on the first train record
  `IMG000002.jpeg` before producing an audit output. The cache row and payload
  agree on descriptor shape `[27,1156]`; candidate count/index bounds, 24-value
  affinity shape, no-train-mask policy and independently reconstructed family
  IDs all pass. The auditor incorrectly hard-coded `128`—the per-layer random
  projection dimension—as the final descriptor dimension. The frozen model
  contract is `3 summaries × 3 layers × 128 + 4 metadata = 1156`, exactly as
  implemented by `MaskBagMILConfig.descriptor_dim`. This is an independent
  audit-tool defect, not cache evidence. The bounded correction is to freeze
  `1156` from that already-existing formula and update the synthetic end-to-end
  regression so `128` can no longer mask the error; no cache/scientific bytes
  or protocol change. No validation packed mask/GT/test was read in the failed
  attempt, no cache was accepted and R1 remains unlaunched.
- **Progress — descriptor-dimension auditor correction frozen:** the auditor now
  freezes projection width 128, three token layers, three proposal/context/
  difference summaries per layer and four metadata values, yielding exact
  descriptor dimension 1,156. Corrected auditor SHA-256 is
  `0c7ab38315dd29cb71b70bdc9dbebb9f4bc8d2d810a56a394c8ba74a47b10d23`;
  corrected regression SHA-256 is
  `a49d67a7b116783dd7e91f6cba78995e2b452d8cca339790cf77cc4b85638ee6`.
  `py_compile`, five auditor tests, the 20-test auditor/cache-I/O suite and the
  complete 343-test worktree regression pass; the synthetic end-to-end cache
  now uses 1,156-dimensional descriptors. The complete regression uses only the
  previously documented Python-3.9 `zip(strict=...)` compatibility shim.
  Machine-readable correction evidence is
  `artifacts/research_protocols/independent_mask_bag_selector_cache_output_audit_v1_descriptor_dim_correction.json`,
  SHA-256
  `714fdee6bbb90f53c255ee48fe13b712e6748304075427ef2b2cec07480a3d85`.
  It changes no cache/scientific bytes or protocol and authorizes only rerunning
  the same independent audit against the immutable downloaded version-5 output.
  Cache acceptance, R1 launch, validation GT, consumer and test remain locked.
- **Progress — selector-cache v5 independently accepted:** the corrected
  fail-closed audit completed successfully against the immutable version-5
  output. Independent audit JSON SHA-256 is
  `7d9f693dd5d1d9206b01cc2c8a0ed4aed497f9f17d9dedf670a97771b0f78334`;
  cache freeze/wrapper audit/cache manifest/baseline reproduction audit/run
  manifest SHA-256 values are respectively
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  `cc2528131003d8b579fd0b0fd0529df8fdd7b0e4e4c92d0a747a6bee5629eafd`,
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`,
  `9eb6280570bdb2568db6b574dbc24c6af2eb63f272f9a63818e09761e1d1875e`
  and
  `90c4e5b231453cfe6afe199c01e9d575c76b78d7f11554cece3d60249ec15a2b`.
  The audit independently verified all 3,352 physical records and
  921,137,925 cache bytes, cohorts train/validation `2,981/371`, candidate
  counts `173,376/20,778`, all 371 validation packed-mask geometries, and exact
  reproduction of all 371 baseline selected indices and maps with maximum
  logit/probability delta `0.0`. Corrected scientific source/protocol remain
  `c0e38628069ff3bedd4493c4ff004b75bd32e008` and
  `be9c91b53926eda2f8bf8dba894385f03dc6accd15d3fa9646da0d1a17a635f2`.
  The direct Kaggle log SHA-256 is
  `f9ce546212d4e64076366e253d94b35ea0fe43ae2c5b925495460b17914abb20`;
  it records two Tesla T4 devices, `DataParallel=True`, 45 focused tests and
  the complete Kaggle regression (`333 passed, 1 skipped`). Training labels
  remain image-level only; `validation_gt_read=false`,
  `consumer_trained=false`, and `test_evaluated=false`. The cache gate is now
  operationally passed, but R1 remains unbound and unlaunched until the newly
  announced collaborator log branch is fully synchronized and checked for an
  overlapping `ĐANG LÀM` claim.
- **Progress — fail-closed R1 cache binder ready:** collaborator synchronization
  found no overlapping active R1/cache claim. The binder at
  `project/bind_mask_bag_normal_prototype_r1_cache.py`, SHA-256
  `f949d56227f015518192ab536a6258a3e8226140cc38e0fd2a3b066d5cda9981`,
  accepts only the exact independent audit SHA, re-verifies both physical cache
  root hashes and their safety/cohort/reproduction contracts, verifies the
  exact unbound wrapper template, and permits exactly the three predeclared
  constant replacements with byte-exact inverse reconstruction. It refuses
  pre-existing or aliased output paths and re-reads the written wrapper/audit.
  Regression SHA-256 is
  `f8f953218edaef60fefac413be4cd9fb7e90d0561f986b80199b353301814c2b`;
  `py_compile` and all five binder tests pass. The 25-test binder/cache/auditor
  suite passes with the already documented local Python-3.9 diagnostic shim;
  without that shim, only the two known `zip(strict=...)` calls fail at the
  interpreter boundary. Kaggle Python 3.12 must still pass the unmodified
  tests before fitting. This is technical binding preparation only: no cache
  record was consumed for fitting, no prediction/GT/test was opened, and R1
  remains unbound and unlaunched pending commit/push of this binder evidence.
- **Progress — exact cache binding and final prelaunch pass:** after the binder
  evidence was pushed at `adec78c`, the accepted cache was bound once into the
  R1 wrapper. Binding audit
  `artifacts/research_protocols/rad_dino_mask_bag_normal_prototype_r1_v1_cache_binding_audit.json`
  has SHA-256
  `c17d7ed89adcb1f2ba86f0a21faacec83d62baf79521e0941cc6cae8d79551ae`.
  Bound wrapper SHA-256 is
  `2ee94b622fdaf02c4cf646be29a1651748cd134461df844ecf67217842269f5b`;
  its inverse reconstruction is the exact unbound template
  `4b886f91aa01b15c18a1a0105db11a31f62233635e214e9bbd406e5712d05044`,
  with only `CACHE_BINDING_READY` and the accepted cache freeze/audit SHA values
  changed. Final metadata SHA-256 is
  `fc290c04a913fd17dfded8ee71a933233af1b047afacdbceecc02d7571f108b2`.
  The first static order check failed closed because a whole-file string search
  matched a test filename in the source-hash table before `main()`; restricting
  order validation to the physical `main()` region passed without any wrapper,
  protocol or scientific change. Final prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_normal_prototype_r1_v1_final_prelaunch_audit.json`
  SHA-256
  `f3921d1102f1b090d57dbf90ff3a3504336a60488d5e935c9733fb778bc5bfd9`
  verifies metadata, `py_compile`, all 21 frozen source hashes at checkout
  `3647997d...`, source ancestry, cache/binding provenance and execution order:
  clone/source/hash/T4x2/cache gates, unmodified focused and whole tests,
  image-label-only R1 fit, physical prediction freeze/audit. No evaluator or
  validation GT path exists in the wrapper; `consumer_trained=false`,
  `test_evaluated=false`, and the kernel remains not launched until this final
  evidence is committed and pushed.
- **Progress — R1 version 1 launched:** final prelaunch evidence was committed
  and pushed at `5a3c659`. The direct prelaunch `kernels status` anti-duplicate
  check returned Kaggle permission/slug denial because no accessible private
  kernel yet existed; an authenticated `kernels list --mine` search then
  returned `Not found`, so no running or prior R1 job was duplicated. Kaggle
  accepted private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-normal-prototype-r1-v1` version `1` from
  bound wrapper SHA-256
  `2ee94b622fdaf02c4cf646be29a1651748cd134461df844ecf67217842269f5b`
  and metadata SHA-256
  `fc290c04a913fd17dfded8ee71a933233af1b047afacdbceecc02d7571f108b2`.
  URL:
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-normal-prototype-r1-v1.
  The wrapper fails before fitting unless exactly two Tesla T4 devices execute
  real convolutions, both unmodified focused and whole-repository suites pass,
  and the exact accepted cache/baseline/source/protocol gates pass. It generates
  and physically audits all 371 validation predictions and candidate-score
  payloads without any evaluator or validation GT. No launch-time status poll,
  recurring polling or monitor was created. Consumer training and BTXRD test
  remain locked; status remains `ĐANG LÀM` pending a later bounded terminal
  check and direct-output audit.
- **Progress — pause/resume and independent output-auditor readiness:** the
  single bounded post-launch check before the user's pause found R1 version 1
  still `RUNNING`. Work then stopped immediately: no repeat polling, monitor,
  output download, evaluator, GT/test access or new launch occurred. On explicit
  resume, central HEAD/origin and the complete `RESEARCH_LOG.md` blob remained
  unchanged at `c3ab3b7`/`961f6cd1...`; this R1 claim is still the only active
  overlapping claim. The GT-blind independent auditor is now frozen at
  `project/audit_mask_bag_normal_prototype_r1_output.py`, SHA-256
  `75a89ad885e8201a565a2a26aa9a401a76a9a0c5c89245cdcb29c411c19a38ba`.
  It independently verifies all 66 OOF artifacts, group exclusion, the finite
  `K={8,16,32}` count guard and one-standard-error choice, all 371 candidate
  score payloads/maps, SmoothMax/bag probabilities/winners, selected-mask map
  construction, final prototype geometry, source/protocol/cache/baseline hashes
  and T4x2 evidence before authorizing any GT evaluator. Regression SHA-256 is
  `1a1a8ef366ff46a1f91add69e04ded68598ffa9e429b52b57761ef48638afe86`;
  nine focused tests, the 62-test R1/cache suite and the complete 353-test
  worktree pass (the local Python-3.9 run uses only the documented
  `zip(strict=...)` shim). Three pre-output technical corrections are preserved
  in readiness evidence: two invalid synthetic fixtures; a string-level
  no-evaluator test that mistook provenance filenames for imports; and draft
  expectations for a nonexistent crossfit key/all-24 source set rather than the
  producer's physical crossfit schema/21 runtime sources. None consumed R1
  output or changed science. Readiness artifact
  `artifacts/research_protocols/independent_mask_bag_normal_prototype_r1_output_audit_v1_readiness.json`
  has SHA-256
  `42e836045827310967f42e566c2f855b7ca0ccbcb2724a537601304d10e17528`.
  Safety remains image-label-only, `validation_gt_read=false`,
  `consumer_trained=false`, and `test_evaluated=false`.
- **Progress — R1 kernel version 1 `LỖI` before scientific execution:** the
  single post-readiness terminal check found version 1 `ERROR`. The standard
  Windows Kaggle output command created a zero-byte log and then failed only
  because its default `charmap` could not encode Unicode progress glyphs; the
  same authenticated session-output API retrieved the complete 11,913-byte log
  explicitly as UTF-8, SHA-256
  `4edc8eac7db4a388957264540c8ec6feda8bf7a1319d5c56e6f4aaf980979dde`.
  The wrapper verified checkout/source/protocol, installed the frozen runtime,
  passed its two-T4 real-convolution guard, reconstructed the split and found
  the exact sanitized `transport_audit.json`, then failed in
  `prepare_baseline()` because no Kaggle input file matched
  `square_corrected_baseline.zip` SHA-256
  `8857eb6d1393683a21efaed3e3f33dee763e32203ac7665a76ff9fd809eee0c4`.
  It failed before baseline extraction, cache discovery, tests, fitting,
  inference or prediction freeze. There are zero kernel output files and no
  scientific result. Error audit
  `artifacts/kaggle/rad_dino_mask_bag_normal_prototype_r1_v1/kernel_version1_error_audit.json`
  has SHA-256
  `6afdb72a0d2fab2caf1bae44357344c45e3185f518e96c1f8a3e1324b55cf311`.
  The log proves only that the exact archive was absent from this input mount,
  not why it was absent; the bounded correction is a dedicated small private
  transport dataset containing the already frozen 1,784,924-byte archive and
  its existing audit, with no scientific source/protocol/wrapper change.
  Version 1 is permanently retained as `LỖI`; the R1 claim remains `ĐANG LÀM`
  solely for that transport-only successor. No validation GT/test was opened,
  no consumer was trained and no prediction was created.
- **Progress — version-2 baseline transport correction frozen:** private
  dataset `itsthang333/btxrd-mask-bag-r1-baseline-transport-v1` version 1 was
  created from exactly two already-frozen files and reached Kaggle status
  `ready`: `square_corrected_baseline.zip` (1,784,924 bytes, SHA-256
  `8857eb6d...`) and `transport_audit.json` (SHA-256 `9377ce5b...`). The
  authenticated CLI's independent dataset-download endpoint returned `403
  Forbidden` on `GetDatasetMetadata`, so no round-trip hash is claimed. Instead,
  the unchanged wrapper remains the launch gate: it must find exactly one
  mounted archive/audit with those physical hashes before cache discovery or
  fitting. Kernel metadata replaces only the old dataset source with this
  dedicated source; selector-cache kernel input and every scientific setting
  are unchanged. New metadata SHA-256 is
  `22d87a060b7d43952371ae11b2fc2db5c4f8f4a070af4ee58cab4cbf11a3dd9d`;
  wrapper remains byte-identical `2ee94b62...` and `py_compile` passes.
  Transport-correction audit
  `artifacts/research_protocols/rad_dino_mask_bag_normal_prototype_r1_v1_kernel_v2_transport_correction.json`
  has SHA-256
  `6ca2c8b2603b85ce160764be3819e74b7567da1ae3a24e8ca2353ce133a56a21`.
  Version 2 remains unlaunched until this evidence is committed/pushed; no GT,
  consumer, test or prediction access occurred.
- **Progress — R1 version 2 launched:** after the transport correction was
  committed and pushed at `279fc57`, Kaggle accepted version 2 of the same
  private kernel. It uses unchanged bound wrapper SHA-256 `2ee94b62...`, new
  metadata SHA-256 `22d87a06...`, the dedicated baseline-only dataset and the
  existing accepted selector-cache kernel source. No launch-time status poll,
  recurring polling or monitor was created. The runtime must still prove the
  exact archive/audit hashes, T4x2, source/protocol/cache/baseline contracts and
  pass unmodified tests before any image-label-only fitting. Validation GT,
  consumer and BTXRD test remain locked; the experiment remains `ĐANG LÀM`.
- **Progress — bounded version-2 status check:** the single planned continuation
  check found R1 version 2 still `RUNNING`. HEAD/origin and the complete central
  log blob were unchanged at `1306c1c`/`647e2ff3...`. No repeat poll, monitor,
  output access, local heavy compute, GT/test read or competing launch occurred.
  The independent output auditor remains frozen and the evaluator remains
  locked until a later terminal direct output passes its physical freeze audit.

### EXP-20260731-codex-local-research-handoff-v1

- **Owner/status:** Codex parallel local workstream; `HOÀN THÀNH` as a
  documentation-only retrospective. It does not claim the active selector
  cache or R1 scope and launches no experiment.
- **Source:** dirty local `pipeline` workspace based at
  `980722ac4b3f673dd09a9b2156d78b6ad334d0d9`; exact source-document hashes
  are recorded in
  `artifacts/research_handoffs/codex_local_wsss_research_handoff_2026-07-31.md`.
- **Split audit:** the local split SHA `7b16771a...` and this branch's
  `85511ee1...` manifest contain the same 3,746 image IDs and have zero
  differences across 32 shared scientific fields. The difference is
  provenance schema/serialization, not cohort assignment. Validation remains
  `371/184/187` with fixed tumor subgroups `94/72/18`.
- **Transferred evidence:** completed comparable Dice rows, diagnostic-only
  upper bounds, retired-family root causes and implications for R1 are frozen
  in the handoff document and machine-readable companion CSV. The best local
  WSSS remains LayerCAM `0.234339`; this branch's audited geometry-v3
  `0.245482` is therefore the best completed synchronized WSSS pseudo-mask
  result, but it is not a final consumer result and still fails its gate.
- **R1-specific negative evidence:** four separate normal/anomaly mechanisms
  reduced shortcuts or exposed weak rank signal but did not produce
  tumor-specific overlap. K=32 healthy density reached pixel AUC `0.563712`
  but true-area Dice only `0.024037`; feature-normal replacement reached
  `0.023373`; causal-patch reached `0.014293`. This does not duplicate or
  invalidate R1, because R1 uses the stronger geometry-v3 candidate gallery
  and image-label OOF residual learning. It does require honest retirement
  without K/threshold sweeps if the frozen R1 gates fail.
- **Artifacts:**
  `artifacts/research_handoffs/codex_local_wsss_research_handoff_2026-07-31.md`
  (canonical-LF SHA-256
  `31822e9389d3cb5f22f36bac47c4746afc44b971aa0aeffe040fc2d445fab74a`)
  and `artifacts/research_handoffs/codex_local_wsss_results_2026-07-31.csv`
  (canonical-LF SHA-256
  `f1a1090947c201e449e65beb887cb22a1e534b7215e28ada9a4868e5d18d20e8`).
- **Safety:** no code, model, cache, prediction or test artifact was imported;
  no validation GT was opened for this documentation step; consumer and test
  remain locked. The active incomplete selector-cache download remains
  explicitly unaccepted.
- **Central synchronization:** collaborator commit
  `bd3bbc27da6799a41c665ff7461762f088e658c0` was fetched and merged from
  `origin/codex/research-sync-20260731` on top of the current central history.
  Its complete `RESEARCH_LOG.md` consists of the already-read shared blob at
  merge base `be9974ec9738330ae2a161482e3ecabc51b50c38` plus the exact 40-line
  retrospective above; both new handoff artifacts were also read in full.
  There is no competing `ĐANG LÀM` claim for selector cache or R1. The final
  sentence of the historical safety record refers to the partial download at
  that branch point and is superseded chronologically by the independently
  accepted version-5 cache recorded under
  `EXP-20260731-codex-r1-normal-prototype-v1`.
- **Transferred decision/insight:** proceed with exactly the frozen R1 arm,
  because its image-label OOF residual on the stronger geometry-v3 gallery is
  not a duplicate of the retired direct anomaly maps. Interpret normality
  distance only as a candidate-ranking residual, never as tumor localization
  evidence by itself. If R1 fails its predeclared AUROC/count-shortcut/regret
  or Dice gates, retire it without K/threshold/morphology expansion and proceed
  only to predeclared R2 then S1. If the finite selector campaign still cannot
  close gallery regret, the next representation hypothesis must learn
  tumor-specific positive-versus-negative local evidence from image-label
  bags; generic healthy rarity, resolution-only changes and consumer retraining
  are already contradicted by the synchronized evidence. This synchronization
  launched no model, prediction or evaluation and opened no validation GT or
  BTXRD test.

### STATIC-DESIGN-20260731-rich-gallery-geometry-v3-pair-v1

- **Owner/status:** Codex local static preparation; `THIET KE TINH - CHUA
  CLAIM/CHUA LAUNCH`. This work does not overlap or modify the active
  `EXP-20260731-codex-r1-normal-prototype-v1` run.
- **Question:** the accepted geometry-v3 selector has actual Dice
  `0.24548239/0.11708058/0.37713552/0.38941265` while its current-gallery
  oracle is `0.40907553/0.22274949/0.59414708/0.64182537`. Before another
  selector change, determine whether already-frozen proposal sources can raise
  the proposal-support ceiling above the frozen fully-supervised reference in
  overall and every `94/72/18` lesion-size subgroup.
- **Hash-locked post-freeze result:** the independent analyzer validates the
  frozen prompt-quality and `test_evaluated=false` contracts for the
  LayerCAM+BiomedCLIP, classifier-448 and AdvCAM-split galleries. Anchoring the
  current LayerCAM+BiomedCLIP gallery and unconditionally appending the
  classifier-448 gallery gives oracle
  `0.53100361/0.33695530/0.73052286/0.74628997` for
  overall/small/medium/large. It exceeds the frozen fully reference
  `0.49513170/0.32895493/0.66244178/0.69370336` in all four metrics. The best
  overall anchored pair, LayerCAM+BiomedCLIP plus AdvCAM-split, is
  `0.53150194/0.33031714/0.73775319/0.75712864`; the classifier-448 pair is
  chosen for the next design because the predeclared bottleneck is the small
  subgroup. The three-source union is a later ceiling only:
  `0.56365361/0.38542822/0.74796800/0.75712864`.
- **Interpretation:** this is not achieved Dice and is not a legal inference
  router. It proves that proposal availability need not remain the ceiling if
  sources are appended to every image before the image-label-only selector.
  The deployable bottleneck remains selector regret and source/count shortcut
  risk.
- **Prepared implementation:** `project/merge_frozen_candidate_galleries.py`
  performs an unconditional, exact-mask-deduplicated union, retains namespaced
  proposal provenance and evaluates every added candidate against the anchor
  LayerCAM prompt map. It contains no polygon/test loader. The planned G0 arm
  transports the immutable geometry-v3 checkpoint to the merged gallery with
  no fit; G1 then retrains the unchanged square-corrected geometry-v3 model on
  the merged train gallery with image labels only. Both must freeze all 371
  predictions before Stage-B Dice/IoU evaluation.
- **Evidence:**
  `artifacts/research_handoffs/rich_gallery_union_oracle_20260731.json`
  SHA-256
  `a0c42250b65266c0a25f573cb40aaedb55ad249d5aa8e755498887db9ffcf4d6`;
  static execution design
  `artifacts/research_protocols/rich_gallery_geometry_v3_pair_candidate_v1.json`.
  Six new unit tests and the focused candidate/cache regression suite pass
  (`21/21`).
- **Coordination/safety:** no Kaggle kernel, candidate generation, training or
  evaluation was launched. After R1 reaches a terminal audited result, the
  complete central log must be synchronized again and a unique gallery-supply
  claim must be registered and made visible before G0/G1. No validation GT is
  an algorithm input; consumer training and BTXRD test remain locked.

### EXP-20260731-codex-rich-gallery-g0g1-v1

- **Owner/status:** Codex independent Kaggle workstream; `ĐANG LÀM`.
- **Scope:** execute the already frozen richer-gallery hypothesis from
  `rich_gallery_geometry_v3_pair_candidate_v1` on the private `wanwin`
  account. This claim unconditionally merges the LayerCAM+BiomedCLIP anchor
  gallery with the 448-pixel binary-classifier gallery, then measures G0
  frozen-selector transport and G1 matched geometry-v3 retraining. It does not
  read, modify, cancel, rerun or interpret the collaborator-owned R1 kernel.
- **Why this is a distinct claim:** R1 changes the candidate-ranking residual
  on the immutable old gallery. This claim changes proposal supply and then
  measures transport/retraining on the enlarged gallery. The scientific
  variable, Kaggle owner, payload and output namespace are separate.
- **Reproduction requirement:** private collaborator checkpoints are not
  assumed accessible. The 448-pixel DenseNet121 is therefore retrained from
  the published image-label-only recipe (seed 42, 448 px, AdamW
  `1e-4/1e-4`, no augmentation, fixed 0.5 validation-F1 checkpoint rule).
  The local BiomedCLIP, SAM ViT-B and RAD-DINO physical weights independently
  match the historical SHA-256 values `52cc993c...`, `ec2df627...` and
  `dbfb9f54...`. The local binary LayerCAM anchor checkpoint is hash-bound in
  the prelaunch payload audit.
- **Required reporting:** actual binary-mask Dice/IoU must be reported for all
  184 validation tumors and separately for `<1%` (`n=94`), `1–<5%` (`n=72`)
  and `>=5%` (`n=18`). Each Dice is accompanied by its signed gap to the
  frozen fully-supervised reference `0.49513170/0.32895493/0.66244178/
  0.69370336`. Oracle values are reported only as proposal-support ceilings,
  never as achieved Dice.
- **Safety/order:** train/validation only; image labels are the only spatially
  weak training supervision; candidate galleries and all 371 Stage-A outputs
  are frozen and independently audited before the Stage-B polygon evaluator;
  BTXRD test remains locked. No per-image oracle/source/lesion-size router,
  validation-area selection, threshold rescue or hidden GT input is allowed.
- **Execution update:** the user explicitly authorized this independent
  `wanwin` workstream without waiting for collaborator R1. Private/offline T4
  kernels `wanwin/btxrd-rich-gallery-classifier448-supply` and
  `wanwin/btxrd-rich-gallery-biomedclip-saliency` were launched after source,
  split, model-weight, Internet-off and no-GT/no-test audits. The server-side
  model bundle was download-backed with exact hashes for LayerCAM, SAM,
  RAD-DINO, OpenCLIP 2.32 and its manifest.
- **Pre-generation implementation audit:** the frozen anchor candidate grid is
  320 px while the published classifier-448 gallery is 448 px. The first
  merger implementation rejected this scientifically intended pair instead of
  defining the required alignment. The merger now performs one fixed,
  deterministic nearest-neighbor 448-to-320 mask projection before exact-mask
  deduplication and records the two input grids plus the number of resized
  images. The anchor prompt map remains the only prompt map used by the
  geometry-v3 descriptor. Focused merger/geometry tests pass `9/9`; no
  polygon, lesion size or source-dependent routing enters this transform.
- **Fast-path decision:** the user prioritized the earliest actual-Dice answer
  over matched reproduction overhead. The accepted collaborator Geometry-v3
  checkpoint and private selector-cache dataset are inaccessible from the
  independent `wanwin` account (Kaggle returns permission denied), so B0
  reproduction and G0 transport are removed from the critical path. The first
  result will be G1 only: the same square-corrected Geometry-v3 architecture
  and fixed image-label training recipe on the merged gallery, compared with
  the already audited published Geometry-v3 Dice. G0 remains optional only if
  that checkpoint later becomes accessible or G1 is ambiguous. Mandatory
  audit is narrowed to exact split/model hashes, complete gallery/mask counts,
  no-GT/no-test and prediction freeze before the polygon evaluator.

### Central rich-gallery synchronization and exact R2 static protocol freeze

- The collaborator branch `origin/codex/research-sync-20260731` advanced by one
  unique commit, `29d2d2871a2dcca663883ae8700f7ec3337e6a63`. Its complete log
  history through the common parent was already identical to the fully read
  central history; the 51 appended lines and all seven referenced
  code/protocol/evidence files were read in full. Merge commit
  `9c432d771d64f77de091701b5e926f5f09336c90` preserves the branch provenance,
  the active R1 version-2 heartbeat and every collaborator entry, and was
  pushed to `origin/research-wsss-improvement` without force-push or conflict.
- The synchronized post-freeze oracle evidence adds a real but separate
  proposal-supply hypothesis. Unconditionally appending classifier-448 to the
  current BiomedCLIP+LayerCAM anchor yields oracle
  `0.53100361/0.33695530/0.73052286/0.74628997`; the three-source ceiling is
  `0.56365361/0.38542822/0.74796800/0.75712864`. These are not achieved Dice
  and cannot be used as a per-image router. Inspection confirms the merger
  deduplicates exact masks, preserves source provenance only for audit and
  scores all appended candidates with the anchor prompt map. The evidence JSON
  hash is `a0c42250b65266c0a25f573cb40aaedb55ad249d5aa8e755498887db9ffcf4d6`;
  its six focused tests pass unchanged in the project environment. A remaining
  operational gap is that the repository evidence contains frozen oracle CSVs
  but not a complete classifier-448 candidate-diagnostics payload for both
  train and validation. Consequently G0/G1 remains a static design requiring a
  later distinct claim and candidate-transport/generation gate; it does not
  replace or tune the finite same-gallery R1/R2/S1 campaign.
- To prevent post-R1 design choices, the exact same-gallery R2 protocol is now
  frozen before the R1 result is known at
  `artifacts/research_protocols/rad_dino_mask_bag_affinity_residual_r2_v1.json`,
  SHA-256
  `3f28cc7187ad64f3755ae4c7a10bb380a0085d1733807dcf667c44d92d9f593d`.
  It binds scientific commit `c0e3862`, accepted cache freeze `2f6290cd...`,
  split `85511ee1...`, baseline checkpoint `58b82642...`, the fixed
  24-dimensional original/flip local-affinity mechanism and exactly one
  16-epoch image-label-only residual fit. It also fixes T4x2 execution,
  371-prediction/all-candidate-score physical freeze, the evaluator-only
  baseline CSV `a26143d...`, all four operational goals and the rule that the
  separate rich-gallery oracle/size information is never a training or
  inference input. All 21 canonical-LF source/test hashes match physically;
  `py_compile`, JSON parsing, `git diff --check` and all 14 focused R2 tests
  pass. Static readiness evidence SHA-256 is
  `f0cc043cc4dc9dfe3b4f4b4be42f39740bb0dc223d23440a92238082a1e5ee4c`.
- Two local test commands initially failed before scientific execution: the
  default Miniconda Python lacked NumPy, and the bundled Codex Python lacked
  PyTorch for the gallery-merger import. The identical tests were rerun in the
  existing `btxrd-pseudomask` environment and passed (`6/6` collaborator tests,
  `14/14` R2 tests). An unsupported PowerShell `Get-Date -AsUTC` spelling was
  likewise replaced by the read-only `DateTime.UtcNow` equivalent. These are
  environment/tooling errors only; no scientific input or output was opened.
- This is static preparation under the active
  `EXP-20260731-codex-r1-normal-prototype-v1` claim, not an R2 claim or result.
  R1 remains last known `RUNNING`; no Kaggle status poll, monitor, R2 launch,
  prediction, validation GT/test access or consumer training occurred. R2 may
  be claimed and launched only after R1 is terminal and independently audited,
  followed by another complete central-log synchronization.

### EXP-20260731-codex-r1-normal-prototype-v1 — version-2 transport error

- One bounded check after the intervening static work found kernel version 2
  `ERROR`. The user-requested standard `kaggle kernels output` command and one
  stdout-UTF8 retry each left a zero-byte log because the Kaggle client's
  Windows text writer still used the default `charmap`. The authenticated
  official session-output API was therefore called with an explicit UTF-8
  writer and returned exactly one 11,999-byte log and zero scientific output
  files. Direct log SHA-256 is
  `5a517847f495a5adbe4bd66731e4be6ffdd387783855233a5bb763d883a8c730`.
- The direct pull of the terminal kernel proves that version 2 used the intended
  dataset source `itsthang333/btxrd-mask-bag-r1-baseline-transport-v1`, the
  accepted selector-cache kernel source, T4 shape and the exact bound wrapper:
  pulled CRLF SHA-256 `44fb9a62...`, canonical-LF SHA-256
  `2ee94b622fdaf02c4cf646be29a1651748cd134461df844ecf67217842269f5b`.
  Runtime passed checkout/source/protocol, frozen dependency installation, the
  two-T4 real-convolution guard, split reconstruction and exact
  `transport_audit.json` discovery/safety. It then failed at the next statement
  in `prepare_baseline()` because no mounted file retained the exact name
  `square_corrected_baseline.zip`; cache discovery, all tests, fitting,
  inference and prediction freeze were never reached.
- Since the transport audit from the same dataset was mounted while only the
  ZIP container was absent, the supported transport inference is that Kaggle
  dataset ingestion/mount normalization expanded the `.zip` content instead of
  retaining that filename. The dataset is `ready` and listed at 1,844,965
  bytes, but `ListDatasetFiles` still returns 403, so no independent remote file
  inventory is claimed. Full version-2 error audit SHA-256 is
  `6bddeb1fde87bd740eb5296d6a74f00b7d12484e3091668e62ad87a6465dcc05`.
- A bounded version-3 transport correction is prepared but not launched: copy
  the same 1,784,924 archive bytes under the non-archive mount name
  `square_corrected_baseline.zip.bin`, require the unchanged archive SHA-256
  `8857eb6d...`, then use the unchanged safe ZIP extractor and all unchanged
  inner baseline gates. The wrapper changes exactly one literal and inverse
  reconstruction returns exact version 2. New wrapper canonical-LF SHA-256 is
  `a896d45dad1efb46d3f0b7544f01ca152e18a5e0f3eabd95f5f00414efe33114`;
  the payload copy is byte-identical, contains 378 members/77,479,804
  uncompressed bytes and passes `ZipFile.testzip`. Static correction audit
  SHA-256 is
  `7637e1542a65dccecd9ffb0047d8185b8474ba9bcae68cde6c9b52faaa091a01`.
  Dataset version 2 and kernel version 3 remain uncreated/unlaunched until this
  error and correction evidence are committed and pushed. Scientific
  source/protocol/cache/split/baseline/fit settings remain byte-for-byte or
  hash-for-hash unchanged. No validation prediction/GT/test access or consumer
  training occurred; no monitor or repeated status polling was used.
- **Progress — version-3 transport prelaunch gate passed:** dataset version 2
  accepted both the original archive and the byte-identical
  `square_corrected_baseline.zip.bin` payload and reached `ready`. The dataset
  slug and kernel metadata remain unchanged; metadata SHA-256 is `22d87a06...`.
  Because the private dataset file-list/download endpoint still denies 403, no
  remote roundtrip hash is claimed: the version-3 wrapper must find exactly one
  mounted `.zip.bin` and audit its frozen `8857eb6d...` bytes before safe
  extraction, cache discovery or fitting.
- The independent output auditor was rebound before any version-3 output exists
  to report kernel version 3 and corrected wrapper SHA-256 `a896d45d...`.
  Physical cache/OOF/selection/all-candidate-score/map checks and all GT locks
  are unchanged. Auditor source/test SHA-256 values are respectively
  `3cc5feeed7fd8fddc2b630448e6bdbd7e18d9020770de850b1e580a40c173a17`
  and `f16cfcad5de0e815bfd8f0487fe4cc02fb338b6e2cf10461fe6c309f09c8c984`;
  all ten auditor tests and the combined 63-test R1/cache suite pass (the latter
  uses only the already documented local Python-3.9 `zip(strict=...)` shim;
  Kaggle 3.12 must pass producer tests unmodified). Final version-3 prelaunch
  audit SHA-256 is
  `7beb631b8248e39602742b2664f84495da61ddaaa70241cf14a94259888e578b`.
  Kernel version 3 remains unlaunched until this evidence and auditor binding
  are committed and pushed. No scientific setting, prediction, GT/test access,
  consumer training, status poll or monitor was added.
- **Progress — R1 version 3 launched:** prelaunch evidence and the rebased
  independent auditor were committed and pushed at `339904b`. Kaggle accepted
  version 3 of the same private kernel with transport-corrected wrapper
  canonical-LF SHA-256 `a896d45d...`, unchanged metadata SHA-256
  `22d87a06...`, baseline transport dataset version 2 and the accepted
  selector-cache kernel source. The only wrapper delta from version 2 is the
  exact non-archive payload filename; the runtime still gates identical archive
  bytes, safe extraction, baseline/cache/source/protocol/T4x2 contracts and
  unmodified tests before the same image-label-only R1 fit. No launch-time
  status check, recurring poll or monitor was created. Validation GT, consumer
  and BTXRD test remain locked; the existing R1 claim stays `ĐANG LÀM` pending
  a later bounded terminal check and direct-output audit.
- **Progress — post-launch static regression:** without querying Kaggle again,
  the complete current worktree regression passes `360/360` in 18.25 seconds
  under the same documented local Python-3.9 `zip(strict=...)` diagnostic shim.
  This covers the synchronized rich-gallery utilities, frozen R2 protocol,
  version-3-bound independent R1 auditor and all existing repository tests.
  It used no scientific input, prediction, GT/test path or local heavy compute
  and changes no launch state.
- **Progress — exact transport-function simulation:** still without polling
  the launched kernel, the exact version-3 wrapper was imported against a new
  temporary Kaggle-like input root containing only the frozen transport audit
  and `.zip.bin` payload. Its unchanged `prepare_baseline()` passed safe
  extraction, found one baseline root and reproduced freeze/checkpoint/manifest
  SHA-256 values `ec346276...` / `58b82642...` / `a810e1fc...` exactly.
  Machine-readable evidence SHA-256 is
  `76d687a491c42bc2889d6fedac2918c5139fde1db51e7187d1a685852f140b5e`.
  This closes the local code-path question but does not substitute for the
  runtime mount/hash gate. No cache, model fit, prediction, GT/test or consumer
  was opened.
- **Progress — bounded version-3 heartbeat:** after the complete regression and
  exact transport-function simulation, one status check found kernel version 3
  `RUNNING`. No output was requested, no repeat poll/monitor was created and the
  independent auditor/evaluator remain locked until a later terminal check.
- **Progress — continuation heartbeat and static R2 output-auditor readiness:**
  after fetching and rereading the complete unchanged central log, one bounded
  continuation check again found R1 version 3 `RUNNING`; no output, repeat poll
  or monitor followed. During the wait, a GT-blind R2 physical-output auditor
  was prepared at
  `project/audit_mask_bag_affinity_residual_r2_output.py`, SHA-256
  `2d772ebca7c7b332e5c5d607a4dfb31fe7fb3cac40bbdbf1394d5bc503e85b64`.
  It pins the exact R2 protocol/source/split/cache/baseline, fixed 16-epoch fit,
  T4x2 `186/185` validation shards, adapter/history hashes and all 371 physical
  candidate-score/map/SmoothMax/winner contracts. It imports only the already
  audited generic physical-map helper and fails if that helper differs from
  SHA-256 `3cc5feee...`; it has no evaluator or GT loader.
- The future kernel/version/wrapper/checkout cannot be chosen after output:
  the auditor requires a separate `FROZEN_PRELAUNCH` launch-binding whose
  runtime hashes are already present in protocol `3f28cc71...`. Auditor test
  SHA-256 is
  `cec3307f2a205650a2589ef05d1dea4ce9db88f96965f05b29c61da65683d9f1`;
  14 focused auditor/helper tests and the combined 28-test R2 runner/features/
  training/auditor suite pass, as do `py_compile` and `git diff --check`.
  Readiness evidence SHA-256 is
  `0c10163009277c479afe596a113758614f4f0969ba172f7c749964fedd04be31`.
  This is static preparation only: R2 remains unclaimed/unlaunched behind the
  active R1 claim, and no prediction, validation GT/test or consumer was
  opened.
- The complete repository regression after adding the R2 auditor passes
  `364/364` in 16.24 seconds under the same documented Python-3.9
  `zip(strict=...)` diagnostic shim. No producer source/protocol or running
  kernel was changed, and Kaggle was not queried during this regression.
- **Progress — later R1 heartbeat and unbound R2 Kaggle wrapper readiness:** one
  later bounded check found R1 version 3 still `RUNNING`; polling stopped and no
  output was requested. A tracked, reviewable R2 Kaggle wrapper template is now
  prepared at `project/kaggle_wrappers/run_mask_bag_affinity_residual_r2_v1.py`,
  canonical-LF SHA-256
  `ee506cdcd2c7cfd7f914694e9dcdf3008d50d2c96159e483cbc8232219c73802`.
  It is physically fail-closed with kernel version `0`, binding false and
  wrapper/checkout `UNBOUND`; therefore it cannot launch under the active R1
  claim. Before fit it requires source/protocol hashes, T4x2 real convolutions,
  exact split, `.zip.bin` baseline, accepted cache/24D affinity and focused/full
  tests. After fit it audits all 371 maps and candidate-score payloads and emits
  the wrapper evidence required by the independent R2 auditor.
- A draft direct self-hash comparison was rejected during static testing:
  embedding the final whole-file hash inside that same file changes the hash
  and has no stable one-pass binding. The committed template follows the proven
  R1 boundary instead: an external one-time binder/final prelaunch audit verifies
  exact wrapper bytes, runtime validates bound fields, and the downloaded-output
  auditor verifies the declared binding. Three wrapper tests and the combined
  seven wrapper/auditor tests pass with `py_compile` and `git diff --check`;
  the complete worktree regression passes `367/367` in 15.82 seconds under the
  documented local Python-3.9 `zip(strict=...)` diagnostic shim.
  Readiness evidence SHA-256 is
  `cf6d6dc01e912a57e0cf05d0a99837666ae3c17f900332e930b18864f1337d31`.
  No R2 claim, binding, kernel launch, scientific input, prediction, GT/test or
  consumer access occurred.
- **Correction — R2 wrapper self-hash binding made non-circular:** a later
  binder audit found that the first tracked wrapper still carried an unused
  `BOUND_WRAPPER_SHA256` literal. Even without comparing it at runtime, filling
  that literal would change the bytes it purported to name. The corrected
  template removes the literal entirely. The future binder has exactly three
  finite replacements (`LAUNCH_BINDING_READY`, `KERNEL_VERSION`,
  `CHECKOUT_COMMIT`), then writes the resulting canonical wrapper hash to an
  external prelaunch binding. Runtime computes the same hash directly from
  `__file__` for `wrapper_output_audit.json`; the independent output auditor
  compares it to the pre-output binding. Corrected wrapper/test SHA-256 values
  are `000843edda2593cba96e9a310462670cfd4be7a561862c5a29ac48624eafd629`
  and `66cc6096a0b7f39437f411cf23079f9860a72c4fd3767cd19ad8d904a0704c4e`;
  seven wrapper/auditor tests and `py_compile` pass. Superseding readiness JSON
  SHA-256 is
  `407020467251cfafa5cdd27cf086ee715eff58460abda3f9df5092d3ce95dc85`.
- One bounded R1 check preceding this static correction still found version 3
  `RUNNING`; no repeat poll or output request followed. R2 remains unbound,
  unclaimed and unlaunched; GT/test and consumer remain locked.
- **Progress — finite R2 wrapper binder ready:** the external one-time binder
  is prepared at `project/bind_mask_bag_affinity_residual_r2_wrapper.py`,
  SHA-256
  `0e744edd20784ac1d56beb33bbe84c1e41f740b00304eb870050c300b3dd1aad`.
  It accepts only corrected template `000843ed...`, replaces exactly launch
  readiness, positive kernel version and a 40-hex execution checkout, verifies
  exact inverse reconstruction, source ancestry, protocol and all 21 frozen
  source/test hashes at that checkout, refuses existing output paths and
  rereads both written files. Its launch-binding JSON has the exact schema
  required by the already frozen independent output auditor.
- A synthetic ignored-path smoke against checkout `4b9b284` and version `1`
  produced wrapper/binding hashes `05e69e9f...` / `93610cde...`, three exact
  replacements and a byte-exact inverse. These smoke outputs are explicitly
  not an authorized R2 binding. Ten binder/wrapper/auditor tests and
  `py_compile` pass; the complete repository regression passes `370/370` in
  13.98 seconds under the documented local Python-3.9 `zip(strict=...)` shim.
  Test SHA-256 is
  `5e5683c767ee9986c51315f299fa2bebc2eae6534fc0332d09e3b68483dd331a`.
  Binder-readiness evidence SHA-256 is
  `a2586adb08c3366c681b512cff1095d171bd86a57bdec8e25649be15d2b5a858`.
  Real binding remains forbidden until R1 is terminal/audited, the central log
  is synchronized and a unique R2 claim is pushed; no scientific data,
  prediction, GT/test, consumer, R2 binding or launch occurred.

### Đồng bộ claim rich-gallery từ nhánh cộng tác (2026-07-31)

- Đã fetch `origin/research-wsss-improvement` và
  `origin/codex/research-sync-20260731`; nhánh trung tâm đang ở
  `1edfa37e9996d98650ee7ada7678f918dfc02a6b`, nhánh cộng tác mới nhất ở
  `65bae85a90c1ee0ca9968eec4fa615c9fcbf9b38`, merge-base là
  `84f7b1fa15423823a6e0295b0551b394fc573287`. Toàn bộ phần lịch sử chung đã
  được đọc trước đó; toàn bộ delta log duy nhất sau merge-base (33 dòng) đã
  được đọc và nhập nguyên nội dung claim
  `EXP-20260731-codex-rich-gallery-g0g1-v1` ở trên. Không nhập ngược các
  heartbeat R1 v2 cũ vì nhánh trung tâm đã giữ đầy đủ bằng chứng và trạng thái
  kế nhiệm R1 v3.
- **Thông tin điều phối mới:** nhánh cộng tác đã nhận quyền G0/G1 với trạng thái
  `ĐANG LÀM` trên tài khoản Kaggle `wanwin`: tái tạo classifier DenseNet121
  448 px bằng nhãn cấp ảnh, hợp nhất vô điều kiện gallery đó với anchor
  LayerCAM+BiomedCLIP, sau đó đo G0 frozen-selector transport và G1 matched
  geometry-v3 retraining. Vì vậy không được launch, tái tạo hoặc đánh giá một
  G0/G1 cạnh tranh trong workstream hiện tại. R1/R2/S1 same-gallery vẫn là biến
  khoa học khác nhưng mọi quyết định kế tiếp phải kế thừa mã claim G0/G1 này.
- **Bằng chứng/insight mới:** nhánh cộng tác chưa ghi metric hay kết quả khoa
  học mới; oracle rich-gallery vẫn chỉ là proposal-support ceiling, không phải
  achieved Dice. Commit kế tiếp `65bae85` chỉ thêm source/test cho supply stage
  và không thay đổi `RESEARCH_LOG.md`, do đó không suy diễn rằng kernel đã
  launch hoặc Stage-A đã freeze. Giá trị mới hiện tại là ownership và protocol:
  checkpoint classifier riêng không được giả định có thể chia sẻ, nên recipe
  tái huấn luyện image-label-only cùng hash vật lý BiomedCLIP/SAM/RAD-DINO phải
  là ranh giới tái lập; tất cả 371 Stage-A outputs phải freeze và audit độc lập
  trước Stage-B GT.
- Lần đồng bộ này chỉ sửa `RESEARCH_LOG.md`; không cherry-pick code của nhánh
  cộng tác, không chạy compute, không truy vấn Kaggle, không mở validation GT
  hoặc BTXRD test và không train consumer.

### Đồng bộ rich-gallery fast path và quy tắc học hỏi có gate (2026-07-31)

- Nhánh cộng tác `origin/codex/research-sync-20260731` tiến từ `65bae85` đến
  `e8683c3bf42cde99c781cb3bb528a6ab1333b327`. Bản log trung tâm tại
  `bdbaec192d5899467082f9817ee0f198ca27765b` không đổi so với bản đầy đủ đã
  đọc trước khi tạm dừng; toàn bộ delta log mới 27 dòng đã được đọc và nhập
  nguyên văn vào `EXP-20260731-codex-rich-gallery-g0g1-v1`. Protocol cập nhật,
  merger, candidate-supply runner và hai test file tại `e8683c3` cũng được đọc
  trực tiếp để hiểu ranh giới kỹ thuật, không cherry-pick hay launch lại.
- **Thông tin mới nhưng chưa phải cải tiến đã chứng minh:** hai kernel supply
  riêng trên `wanwin` đã launch; G1-only là critical path vì checkpoint/cache
  Geometry-v3 riêng không truy cập chéo tài khoản. Phép chiếu mask cố định
  nearest-neighbor `448→320` trước exact dedup giải quyết bất tương thích grid và
  giữ anchor prompt map cho mọi candidate. Đây là contract GT-blind cần thiết để
  G1 chạy đúng, nhưng chưa có terminal actual Dice nên không được xem là kỹ thuật
  cải thiện và chưa được đưa vào same-gallery R1/R2/S1.
- **Quy tắc học hỏi được người dùng làm rõ:** log nhánh cộng tác được dùng cả để
  tránh duplicate và để hai workstream tiếp nối nhau. Tuy nhiên chỉ kế thừa một
  kỹ thuật vào hướng cải tiến vì hiệu năng sau khi có kết quả terminal, audit
  đúng prediction-freeze/GT boundary và tốt hơn baseline liên quan trên metric/
  gate chung đã predeclare. Code/protocol/oracle ceiling/kết quả đang chạy không
  đủ. Kết quả âm và error chỉ dùng để loại hướng, thu hẹp không gian hoặc thiết
  kế đối chứng. Khi G1 có kết quả, workstream hiện tại chỉ được kế thừa phần tốt
  nếu actual Dice thực sự tăng; mọi successor phải nêu mã nguồn, exact artifact,
  phần giữ/thay đổi và một hypothesis chưa được chạy, không sao chép G1.
- Rule bền vững tương ứng đã được thêm vào `AGENTS.md`. Claim G1 vẫn thuộc
  workstream cộng tác và `ĐANG LÀM`; không có G0/G1 cạnh tranh, status poll hay
  monitor nào được tạo từ workstream hiện tại.

### S1 exact matched-pooling static predeclaration (2026-07-31)

- Exact protocol tĩnh
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1.json`
  có SHA-256
  `225edfdb4d66c2147b482ffc3857bf17024a9c683db45b1559c4bccd87b10008`.
  Nó khóa hai arm standard normalized SmoothMax và hierarchical
  family-balanced SmoothMax với cùng cache/baseline, CPU-created zero-residual
  initial state, batches, optimizer, 16 epoch fixed-final, loss, validation
  cohort và serialization; sole changed variable là bag pool. Standard chạy
  T4:0, family-balanced chạy T4:1 và probe tám ảnh phải khớp candidate logits
  trong `5e-6`. Mỗi arm phải freeze 371 prediction/candidate-score payload trước
  pair freeze và trước mọi validation GT.
- Protocol đặt S1 sau terminal audited R1 rồi R2, không claim/bind/launch và
  tham chiếu claim riêng `EXP-20260731-codex-rich-gallery-g0g1-v1` để không
  duplicate. Family/source balancing từ rich gallery chỉ có thể trở thành
  successor nếu G1 sau này chứng minh actual Dice tốt hơn; oracle hoặc supply
  readiness không đủ để thay đổi S1 same-gallery đã khóa.
- JSON parse và cả 12 canonical source/test hash đều pass. `py_compile` pass;
  focused runner/pooling/relational suite pass `14/14` trong existing
  `btxrd-pseudomask` Python 3.9 environment. Lần gọi đầu bằng terminal-default
  Python dừng ở pytest collection với `ModuleNotFoundError: numpy`; không test,
  dữ liệu hay kết quả khoa học nào được mở. Không cài package hoặc sửa source;
  rerun cùng suite bằng environment đã có. Readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_readiness.json`
  có SHA-256
  `45f71a00a26633dfb37721ba7680f515d48a4890a23f05491463ba8f7756f7dd`.
- R1 version 3 vẫn ở heartbeat cuối `RUNNING`; trong static preparation này
  không poll lặp, không tải output, không mở scientific input/validation GT/test,
  không train consumer và không chạy compute nặng local. S1 tiếp tục bị khóa sau
  R1→R2; artifact này chỉ ngăn lựa chọn hậu nghiệm nếu finite campaign tiến tới
  S1.

### EXP-20260731-codex-r1-normal-prototype-v1 — terminal OOF rejection

- Một bounded status check sau khi tiếp tục task thấy kernel
  `itsthang333/btxrd-rad-dino-mask-bag-normal-prototype-r1-v1` version 3 ở
  terminal `ERROR`. Không có repeat poll hoặc monitor. CLI direct-output đầu
  tiên bị local timeout sau 124 giây ở 48 file/12,318,159 bytes; không suy luận
  từ download dở. Official `KaggleApi.kernels_output` với UTF-8 text-open
  workaround đã tải lại bằng `force=True` và hoàn tất 68 file/16,785,943 bytes
  vào ignored temp root
  `tmp/kaggle/normal_prototype_r1_v3_error_20260731_2305`. Direct log SHA-256 là
  `bf81d1fed07af9a0a3a0103994874c6c342490b159c1206ce921dba0d0cc3a3a`.
- **Provenance/runtime gate:** log xác nhận checkout
  `3647997d0c18ad31057709462fd8c922d939fb4f`, scientific source
  `d66c89958baa3344dbbaae6030a9ccd8ecec7b3a`, protocol
  `dded9c638e142576fedf0ae4c8102fdf64198744a4949707865e50b7081f312b`, split
  `85511ee1...`, selector-cache freeze `2f6290cd...`, baseline checkpoint
  `58b82642...` và exact frozen hyperparameters trong duy nhất một runner call.
  Focused tests pass `53/53`; full Kaggle suite pass `332`, skip `1`. Exact bound
  wrapper SHA-256 `a896d45d...` đặt two-T4 real-convolution guard trước
  baseline/cache/tests/runner; vì log đi tới runner nên guard đó đã pass, dù tên
  device/checksum không được serialize do wrapper audit chỉ được viết sau
  prediction.
- **Independent GT-blind output audit:** cả 15 five-fold group-OOF fits cho
  `K={8,16,32}` hoàn tất trên 2,981 train images/984 groups, group overlap bằng
  0; 66 OOF files và toàn bộ aggregate rows/probability/BCE/Spearman/prototype
  payload được tái tạo. Crossfit assignment SHA-256 là
  `84ed881e982fbc70fa50129dafabf638dc6e2a864c0ebb60d5e3c2b512d9eeb8`.
  Audit JSON
  `artifacts/kaggle/rad_dino_mask_bag_normal_prototype_r1_v1/kernel_version3_oof_count_guard_error_audit.json`
  có SHA-256
  `833c4814eee0891df1cd55a01fa008e0708ad3485dd30b730657f52508317719`.
  Auditor source/test SHA-256 lần lượt là
  `04375745d3993e38fbd894fef854b306fdc9256e8d1702da2ebba9a7ff46ae2d` /
  `cd1147bd6fdc67f969b9ae9a7dfefb678df1f94faaa851f2a0949bba3ff6800e`;
  focused complete/error auditors pass `12/12`, full local regression pass
  `372/372` bằng documented Python-3.9 `zip(strict=...)` diagnostic shim.
- **Kết quả âm đã predeclare:** baseline absolute candidate-count/probability
  Spearman là `0.48137777593654113`, tolerance `0.02`, nên trần là
  `0.5013777759365411`. Với `K=8/16/32`, absolute OOF Spearman lần lượt là
  `0.6036591097469204/0.6029977890495820/0.6028243037056212`, vượt trần
  `+0.1022813338103793/+0.1016200131130409/+0.1014465277690801`.
  Mean OOF image BCE tương ứng là
  `0.1067681700602822/0.1046994421741868/0.1089559329557645`; cả ba K đều fail
  count guard, nên one-standard-error selector cố ý raise
  `all prototype counts increase the frozen count shortcut`.
- **Error boundary/kết luận:** R1 dừng sau toàn bộ OOF aggregates nhưng trước
  prototype-count selection, final fit, validation inference, prediction freeze
  và evaluator. Không có validation Dice để báo cáo và không được cứu bằng đổi
  K/tolerance/epoch. Normal-prototype residual làm mạnh hơn shortcut proposal
  count thay vì giảm selector regret; cơ chế R1 bị loại. Validation GT không mở,
  không tạo prediction validation, không train consumer và BTXRD test vẫn khóa.
  Finite campaign chuyển đúng predeclaration sang R2 frozen local-affinity.

### EXP-20260731-codex-r2-affinity-residual-v1

- **Owner/status:** Codex main task trên `research-wsss-improvement`;
  `HOÀN THÀNH — TERMINAL COMPLETE, GATE FAIL`.
- **Registered:** `2026-07-31T16:13:39.6180285Z`; registration base commit
  `f514f1bc8467f8ef05dfa4f9b2edf758a300c8f2`; exact claim commit
  `569bff4c4fb5552a039cad9ad08a260294a48069` đã push lên branch điều phối trước
  mọi binding/launch.
- **Objective/hypothesis:** kiểm tra liệu zero-initialized residual chỉ dùng 24
  frozen RAD-DINO local-affinity summaries (within proposal, local context và
  across-boundary contrast trên original/aligned flip) có giảm selected-to-oracle
  regret mà không tăng shortcut count, giữ nguyên complete Geometry-v3 scorer và
  candidate gallery hay không.
- **Non-duplicate scope/inheritance:** kế thừa kết quả terminal bị loại của
  `EXP-20260731-codex-r1-normal-prototype-v1`, accepted geometry-v3/cache và exact
  R2 static protocol. R2 thay representation evidence bằng local affinity, không
  dùng normal prototypes, không regenerate proposal và không trùng
  `EXP-20260731-codex-rich-gallery-g0g1-v1` đang thay proposal supply. G1 chưa có
  actual Dice terminal nên không có kỹ thuật hiệu năng nào từ G1 được đưa vào R2.
- **Exact inputs/protocol:** protocol
  `artifacts/research_protocols/rad_dino_mask_bag_affinity_residual_r2_v1.json`
  SHA-256
  `3f28cc7187ad64f3755ae4c7a10bb380a0085d1733807dcf667c44d92d9f593d`;
  split `85511ee1...`; selector-cache freeze `2f6290cd...` với 3,352 physical
  records và 24D affinity; baseline checkpoint `58b82642...`/prediction freeze
  `ec346276...`; source closure tại
  `c0e38628069ff3bedd4493c4ff004b75bd32e008`. Corrected fail-closed wrapper
  template SHA-256
  `000843edda2593cba96e9a310462670cfd4be7a561862c5a29ac48624eafd629`,
  one-time binder SHA-256
  `0e744edd20784ac1d56beb33bbe84c1e41f740b00304eb870050c300b3dd1aad`, và
  independent output auditor SHA-256
  `2d772ebca7c7b332e5c5d607a4dfb31fe7fb3cac40bbdbf1394d5bc503e85b64`
  đã static-ready; real binding chỉ được tạo sau khi exact claim commit xuất hiện
  trên origin.
- **Compute:** duy nhất Kaggle Nvidia Tesla T4x2; 16 epoch fixed-final image-label
  fit trên T4:0, two-device validation scoring/real CUDA guard theo protocol.
  Không heavy compute local.
- **Expected output/gates:** freeze vật lý đủ 371 validation maps và mọi candidate
  score theo original gallery order trước evaluator. Sau independent GT-blind
  output audit mới được mở validation GT và baseline per-image SHA-256
  `a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f`.
  `MECHANISM_PASS` và `OPERATIONAL_PASS` dùng common evaluator; operational Dice
  tối thiểu là `0.34024039/0.17895493/0.51244178/0.49370336`, cùng CI/miss/AUROC
  gates đã khóa. Chỉ full operational pass mới cho phép consumer.
- **Safety:** train supervision chỉ image-level; không validation GT/subgroup/
  candidate Dice/oracle rank trong fit; không post-hoc feature/epoch/threshold;
  BTXRD test khóa. Không launch trước khi claim được push/sync và bound wrapper
  cuối cùng pass source/protocol/cache/baseline/T4x2/tests gates.
- **Progress — version-1 binding/prelaunch frozen:** sau khi claim commit
  `569bff4c4fb5552a039cad9ad08a260294a48069` và commit xác nhận `5905b4f` đều
  hiện trên `origin/research-wsss-improvement`, one-time binder khóa wrapper vào
  execution checkout `5905b4f9549a03dde5f4dba1c9279ef7146e3244` và kernel version 1. Bound wrapper
  canonical SHA-256 là
  `37724fa4082db58459c65a89394e704ee0f4a60900aad2278de1f08364e91c88`;
  launch-binding SHA-256 là
  `93f7d8887fc4ee93d0e379fca753ff613223bbcb436b8d1e8c1820318efebb7b`;
  metadata SHA-256 là
  `c3d81f3afecd398514f7e95629e662a1cfa58cc12aa832140f583e8963772bd5`.
  Binder xác nhận đúng ba replacement, inverse về exact template và 21/21
  runtime source/test hash tại checkout.
- Bound wrapper `py_compile` pass; binder/wrapper/auditor/runner/feature/training
  suites pass `24/24`. Full repository ngay trước binding pass `372/372` bằng
  documented Python-3.9 `zip(strict=...)` diagnostic shim. Final prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_affinity_residual_r2_v1_kernel_v1_final_prelaunch_audit.json`
  có SHA-256
  `ff8e4a2d6ea81a466b57ef22795e3426a9a08fc0f1b80535d33bee1ff1462fc2`.
  Nó khóa đúng baseline transport dataset, selector-cache kernel input, private
  T4 machine shape, protocol/source/cache/baseline và independent auditor.
- Đây vẫn là prelaunch kỹ thuật: chưa push kernel, chưa status poll, chưa mở dữ
  liệu khoa học, chưa tạo prediction/validation GT, chưa train consumer và không
  truy cập test. Exact binding và audit phải commit/push trước launch; sau đó
  package hash được kiểm tra lại và chỉ một version 1 được push, không launch-time
  poll hoặc monitor.
- **Progress — R2 version 1 launched:** exact launch binding/prelaunch evidence
  được commit và push tại `d8661fe0264ede64e5a68f1ce937d1ea10740ee8`.
  Ngay trước launch, HEAD/origin khớp, worktree sạch, remote log chứa claim và
  bound hash; wrapper `37724fa4...` và metadata `c3d81f3a...` được hash lại và
  khớp binding/audit. Kaggle chấp nhận version 1 của private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-affinity-residual-r2-v1` lúc
  `2026-07-31T16:17:48Z`. Chỉ một push được thực hiện; không launch-time status
  poll, repeat poll hay monitor. Runtime vẫn phải tự chứng minh source/protocol,
  T4x2 real convolution, reconstructed split, exact baseline/cache, focused/full
  tests và freeze đủ 371 maps/candidate scores trước khi có thể audit/evaluate.
  Validation GT, consumer và BTXRD test tiếp tục khóa.

- **Terminal retrieval/audit:** one bounded status check found version 1
  `COMPLETE`. The direct CLI timed out locally after 124 seconds with only a
  partial download, and one official `force=True` API retry met a
  kaggleusercontent connection timeout; neither partial state was interpreted.
  A bounded official-inventory resume with atomic `.part` replacement produced
  the complete compact result root: 749 files/77,494,000 bytes, exactly 371
  candidate-score payloads, 371 maps and seven freeze/checkpoint/manifest files.
  The remaining official inventory was only the redundant `r2_source` checkout
  and legacy artifacts, so its helper was stopped to avoid wasting network/time.
  Direct log SHA-256 is
  `94406cf25328a3e574a761e087d10af15da3e368778a1eaf884768ae175eaf11`.
- Independent GT-blind audit passed before any validation GT and physically
  verified 77,497,309 bytes. Prediction-freeze SHA-256 is
  `3f557d9618169f292eb5c6a23fa77d050b8c37803f523bcb3e12f2a63684fa40`;
  score-manifest SHA-256 is
  `783069184f26c709eba2f428449bc18a3b4aea5544d9aa2a19669ca74f270e44`;
  wrapper-output-audit SHA-256 is
  `1a3707921a03a5f7b039ddea258e3233587b1beb0c54638aa1dbbe154ad8e467`.
  Tracked independent audit
  `artifacts/kaggle/rad_dino_mask_bag_affinity_residual_r2_v1/kernel_version1_complete_gt_blind_output_audit.json`
  has SHA-256
  `740616136bb56e4a356085117e771b0f7006b2f1f7dfa0680aab86bc8c0fea13`.
  Absolute validation count/probability Spearman decreased from frozen baseline
  `0.48137777593654113` to `0.4699654612347809`; this isolated diagnostic did
  not authorize promotion before Dice evaluation.
- **Post-freeze evaluation:** the first local Python 3.9 invocation opened GT
  only after audit pass, then stopped before output serialization because
  `zip(strict=True)` is unsupported. The identical evaluator/source/input/seed
  was rerun with the already documented diagnostic builtins-zip shim; no model,
  threshold, protocol or prediction changed. Evaluator SHA-256 is
  `ccc3a4931907f0cafcf62adb0fef09db82108c6551db959c5926b8148d47b084`.
  R2 Dice overall/small/medium/large is
  `0.23620168299249675 / 0.10976054406519861 / 0.3624019287552691 /
  0.39170442545063067`, versus geometry-v3
  `0.2454823867797678 / 0.11708057891440651 / 0.37713551529480416 /
  0.3894126471276201`. Deltas are
  `-0.009280703787271038 / -0.007320034849207898 /
  -0.014733586539534972 / +0.0022917783230105293`; CI95 are respectively
  `[-0.019035002847131204,-0.002001549645557231]`,
  `[-0.022427420643466426,+0.0004934592793643835]`,
  `[-0.025606165607094063,-0.005663656265479824]`, and
  `[-0.02085098649430166,+0.030689685502147086]`. Image AUROC is
  `0.8186177633108579`.
- **Gate/conclusion:** mechanism and operational gates both fail. Regret is
  reduced only for large (`1/4` subgroups), overall Dice regresses, and absolute
  candidate-count/miss association rises from `0.3135074102409047` to
  `0.32002927972401846`. Overall/small/medium regret worsens; only large Dice
  rises slightly with a wide CI. Therefore local-affinity R2 is rejected and is
  not learned/adopted as an improvement. Consumer remains unauthorized; no
  consumer was trained and BTXRD test remains untouched.
- Evaluator output hashes are: `evaluation_audit.json`
  `7544bbfcbf3c4d840180b7a83a47bded7e247634066b1f12269fa31b56411272`,
  `gate_decision.json`
  `8af7812c0a65a4dcd3fdc12b778be63faaf07dec921d085999777fc368a9a63e`,
  `paired_comparison.json`
  `7e2ab19f90a1cc17bff558bc1f54181e32d61e7f8031b4af1efcf19b0e7f4c7d`,
  `per_image.csv`
  `1013e0094861fcce417902245472087156ff921254ed3df09f43806e39542b4b`,
  and `summary.json`
  `cc273551adf84a4ba716927caeb9a203b6163995588a55f85096aa5a1050fec0`.
  Consolidated terminal audit
  `artifacts/kaggle/rad_dino_mask_bag_affinity_residual_r2_v1/kernel_version1_terminal_result_audit.json`
  has SHA-256
  `8e1687bbddc513d2d51472049f55951608d0405a0d676d9138689db228f33eae`.
- Immediately before the successor claim, central HEAD/origin were synchronized
  at `43fd32d9629c182350bbf6a6672229c3f5ce5102`; the entire central log was read.
  Collaborator branch advanced to
  `b1da28db0c18dd983f8ed5e080c745869935bd8c`, but its `RESEARCH_LOG.md` is
  unchanged since `e8683c3` and contains no terminal audited G1 metric. New
  rich-gallery evaluator/source code is therefore useful only as preparation,
  not performance evidence, and no G1 technique is adopted into the successor.

### S1 static audit-gap correction — no claim/no launch (2026-07-31)

- Static review found that the original S1 prediction contract froze candidate
  logits but did not physically freeze the immutable `family_ids`. Therefore an
  independent auditor could reproduce standard normalized SmoothMax but could
  not reproduce hierarchical family-balanced SmoothMax. This is an auditability
  defect before launch, not a scientific result and not evidence that S1 is
  better.
- The source-only correction was committed and pushed at
  `f3da1817ee3491f04e8c86335556762ebc675d8d`. Before parallel training, the S1
  runner now writes one shared 371-row candidate-family manifest plus exact
  candidate-index/nonnegative-family-ID payloads, then binds the manifest SHA-256
  into both checkpoints, both arm freezes, the pair freeze and run manifest.
  Runner/test SHA-256 are
  `0102f79220d3dfb417eeb70c7dadf6b846044a6c881a3653110f33ecfd526b74` and
  `50566421f5d1b2db564cfba0da8f46f8d391f91c5c87a504de1248385e15d010`.
- The corrected exact protocol supersedes, but does not erase, protocol SHA-256
  `225edfdb4d66c2147b482ffc3857bf17024a9c683db45b1559c4bccd87b10008`.
  New protocol SHA-256 is
  `62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413`;
  JSON parse and all 12 canonical runtime/test source hashes pass. It was pushed
  at `e6af4318078ecac8f35451fa05f395285350e46a`.
- A new GT-blind independent physical auditor at
  `project/audit_mask_bag_family_balanced_s1_pair_output.py` was committed and
  pushed at `8ffd04c6ca7aabbfd0c4ba1175b1fb49b0f02b48`. Auditor/test SHA-256 are
  `d1f6bac4bbffc71df37d609c5fc92dd65d33d905f14bb93cf33cbb949c11e5d6` and
  `86ad491583ed6766e71ebc188dea03bcc23f934351ac29adf804017096ef82c7`.
  It independently verifies the common family payloads, exact candidate order,
  standard and hierarchical family-balanced pooling, selected argmax,
  probabilities, maps, 16-epoch histories, pair/arm freezes, T4x2 runtime,
  cache/protocol/source/launch binding and safety flags before any evaluator or
  validation GT. Focused runner/pooling/relational/auditor suite passes `22/22`
  in 4.13 s; `py_compile` passes.
- Superseding readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_readiness.json`
  has SHA-256
  `eda5f84a841aa5bb228d9a5a5ee2b032a4b1082ab728153baa2de663cc7dba5d`.
  The previous readiness SHA-256
  `45f71a00a26633dfb37721ba7680f515d48a4890a23f05491463ba8f7756f7dd`
  remains recorded as superseded evidence.
- Full local regression passes `380/380` in 13.88 s. As in earlier audits, the
  Python 3.9 command used only the documented diagnostic shim that ignores the
  unsupported `strict` keyword for built-in `zip`; it did not change scientific
  source or the Kaggle runtime contract.
- Coordination remains unchanged: R1 is terminal and rejected at its frozen OOF
  count-shortcut gate; `EXP-20260731-codex-r2-affinity-residual-v1` version 1 is
  still the active experiment and no post-launch status check or monitor was
  performed during this S1 preparation. S1 remains unclaimed, unbound and
  unlaunched until the terminal independently audited R2 decision satisfies the
  predeclared transition. No scientific input, prediction, validation GT,
  consumer or BTXRD test was opened and no heavy compute ran locally.
- Result-gated learning remains binding: this correction improves auditability
  only. Neither S1 nor collaborator G1 may be adopted as a performance
  improvement until a terminal audited output is actually better on the common
  predeclared metrics/gates; running code, oracle support and implementation
  availability are insufficient.

### EXP-20260731-codex-s1-family-balanced-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`;
  `HOÀN THÀNH — TERMINAL REJECTED`.
- **Registered:** `2026-07-31T16:55:41Z`; registration base commit
  `43fd32d9629c182350bbf6a6672229c3f5ce5102`; exact registration commit
  `97db17c16938a8f842546076a26a52e58928b07b` was pushed to the central branch
  before any binding or launch.
- **Objective/hypothesis:** run the already predeclared matched causal pair to
  test whether hierarchical normalized SmoothMax within immutable proposal
  families and then across families reduces proposal-multiplicity shortcut and
  selected-to-oracle regret relative to an otherwise identical standard
  normalized-SmoothMax residual control.
- **Non-duplicate scope/inheritance:** inherits terminal rejected
  `EXP-20260731-codex-r1-normal-prototype-v1` and
  `EXP-20260731-codex-r2-affinity-residual-v1`, the accepted geometry-v3 baseline
  and selector cache. It does not reuse either rejected mechanism: both arms use
  the same descriptor-only residual and differ only in the predeclared bag pool.
  It does not duplicate active
  `EXP-20260731-codex-rich-gallery-g0g1-v1`, does not change candidate supply and
  adopts no collaborator technique because no terminal better G1 result exists.
- **Exact inputs/protocol:** corrected protocol
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1.json`
  SHA-256
  `62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413`;
  static readiness SHA-256
  `eda5f84a841aa5bb228d9a5a5ee2b032a4b1082ab728153baa2de663cc7dba5d`;
  source commit `f3da1817ee3491f04e8c86335556762ebc675d8d`; split
  `85511ee1...`; selector-cache freeze `2f6290cd...`; baseline checkpoint
  `58b82642...` and baseline prediction freeze `ec346276...`. The shared
  candidate-family manifest/payloads must be written before training and bound
  into both arms and pair freeze.
- **Compute/output:** one Kaggle T4x2 run, standard arm on T4:0 and
  family-balanced arm on T4:1 with identical initial state, batch order,
  optimizer, 16 fixed-final epochs and image-label-only losses. Both arms must
  freeze 371 maps and every candidate score; the pair must pass the independent
  GT-blind physical auditor before the evaluator may open validation GT.
- **Predeclared gates:** causal promotion requires family-balanced to improve
  the frozen mechanism diagnostics over the matched standard arm; improvement
  shared by both arms is residual-head evidence, not family-balance evidence.
  Operational Dice minimum remains
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336` plus CI/miss/AUROC gates.
  Only a terminal audited result that is actually better may be learned as an
  improvement; failure/negative evidence is recorded but not adopted.
- **Safety:** no validation segmentation GT, subgroup, candidate Dice or oracle
  rank during fit; no post-hoc pool/epoch/threshold alternative; no consumer
  before full operational pass; BTXRD test locked. No binding, kernel push,
  training, inference or prediction has occurred under this claim yet.
- **Progress — version-1 binding frozen, not launched:** compact fail-closed
  wrapper template and one-time binder were committed/pushed at
  `04ca6602743e411b28bec329df6b287d72b5608e` and
  `7ad0462f94e4cc0109fb7db605c0388ff7164f05`. Exact execution checkout is
  `7ad0462f94e4cc0109fb7db605c0388ff7164f05`; binder verified the S1 claim and
  scientific-source commits are ancestors, the corrected protocol hash and all
  12 canonical source/test hashes. Template/binder SHA-256 are
  `f613b259546552c226f0197e3878442fb6891ed76655b5030b251d24a087a4ba` /
  `f03f2a2ac7663cfcca6a208de332b45a7a875ef7241281eac86057c6a1f70bb1`.
  Bound wrapper SHA-256 is
  `ad5cfe595820215353ebd2513d6f6f71bacbe9121ba67a28ac28605cb44c5462`;
  launch-binding SHA-256 is
  `9e77ef03d77162674bc1305440f05a09257ad4d6ac1c0157987c36a6030fe442`;
  metadata SHA-256 is
  `36137acf5ced95c9e4a92f6cb73c157527ff2acd65d553be61882e1cc4418f7a`.
- Bound wrapper/binder/independent-auditor `py_compile` passes; focused closure
  passes `29/29`, and full repository passes `387/387` in 15.60 s under the
  documented local Python-3.9 `zip(strict)` diagnostic shim. The actual launch
  binding also passes the independent auditor's prelaunch source check. Final
  prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_kernel_v1_final_prelaunch_audit.json`
  has SHA-256
  `95d3adf4e74d84d681030e27da0387065e91f03cf28304fbb03d1d177c04631c`.
- To prevent the retrieval waste observed for R2, the wrapper removes only the
  redundant cloned source and extracted runtime after the complete pair output
  and wrapper audit have been written. Frozen predictions, scores, family
  payloads, checkpoints, histories and pair freeze remain. No kernel was pushed,
  no scientific data/prediction was opened locally, validation GT/consumer/test
  remain locked, and no monitor was created in this binding step.
- **Progress — S1 version 1 launched:** launch binding/final prelaunch audit were
  committed and pushed at `55c9baa2a2153319e04952bce5c774be9553c7ea`.
  Immediately before launch, HEAD matched origin with a clean worktree and the
  package hashes were reverified as bound wrapper `ad5cfe59...`, metadata
  `36137acf...`, launch binding `9e77ef03...` and final audit `95d3adf4...`.
  Kaggle accepted private kernel version 1
  `itsthang333/btxrd-rad-dino-mask-bag-family-balanced-s1-pair-v1`; local
  post-push timestamp was `2026-07-31T17:05:58Z`. Exactly one kernel push was
  performed. No immediate status poll, repeat poll or monitor was created.
  Validation GT, consumer and BTXRD test remain locked until terminal compact
  output passes the independent matched-pair physical audit.
- **Metadata-time correction:** the immutable prelaunch audit's manually entered
  `created_utc` value `2026-07-31T17:08:00Z` is about two minutes later than the
  actual commit/launch ordering. This display-only timestamp is inaccurate;
  exact Git commits, protocol/source/package hashes, tests and launch binding are
  unaffected. The frozen artifact is preserved unchanged and this successor note
  is the correction, rather than rewriting prelaunch evidence.
- **Overnight coordination/status:** on 2026-08-01 local time, the entire
  `AGENTS.md` and central `RESEARCH_LOG.md` were reread; central HEAD/origin were
  synchronized and clean at `6b2aeddb750e774f45352e962abcac8769255732`.
  Collaborator branch advanced to
  `797f191fcae8a5bb0b4ccb920d1adf635af1eff8`, but its log remains unchanged and
  contains no terminal G1 result; its classifier-bound supply provenance code is
  not performance evidence and is not adopted. One bounded S1 status check
  returned `RUNNING`; no second poll or monitor was created.
- **Post-freeze evaluation decision predeclared while RUNNING:** before seeing
  any terminal output, prediction freeze, validation GT or metric, a result-gated
  matched decision was committed/pushed at
  `9779e80a285741879fa5fce7e188f4dc67e0557c`. Decision source/test SHA-256 are
  `90bf0d92e6a53e3773418810fb81a12c0069a944e47d6c25a8a5c5956e8aaff7` /
  `71dcf4604fa11d19f15b9439a2560a15bf4b216164f92670581ec606d8f0161c`.
  After the independent GT-blind pair audit passes, standard and family-balanced
  will each be evaluated against the same geometry-v3 baseline using 10,000
  complete-group bootstrap replicates and the same seed `20261021`; then the
  already frozen comparator performs `family_balanced minus standard` with
  10,000 replicates/seed `20261101` without reopening GT.
- The final decision independently binds pair freeze/audit, both evaluation
  audits and physical output hashes, then requires family-balanced vs standard:
  regret improvement in at least two tumor subgroups, no overall Dice regression
  and no increase in absolute count/miss association. Consumer authorization and
  learning/adoption additionally require the family-balanced arm's complete
  operational pass against geometry-v3. Thus code availability or a partial
  causal gain cannot be called an improvement. Static suite passes `19/19` in
  1.29 s under the documented local Python-3.9 `zip(strict)` shim; no scientific
  source was changed for that runtime issue. Full repository regression passes
  `390/390` in 13.83 s under the same diagnostic shim.
- Machine-readable readiness
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_postfreeze_evaluation_readiness.json`
  has SHA-256
  `4497ab6e9a602a328c8a4d626e5741e138238cc2490159774117db7a4b68f8dd`.
  This preparation opened no scientific data/prediction/GT, trained no consumer,
  accessed no test and ran no heavy local compute.
- **Bounded status after readiness push:** at `2026-07-31T17:13:15Z`, after the
  static decision code/readiness and full regression were committed to central
  branch `b66699a228edd5a8b0213222aed915ff5348d2a7`, one additional bounded check
  still returned `RUNNING`. This is not a scientific result. No further poll or
  monitor was created; terminal output, prediction freeze, validation GT,
  consumer and BTXRD test remain unopened/locked.
- **Compact retrieval preparation without polling:** the next goal continuation
  occurred only about one minute after that bounded check, so no new status query
  was made. To avoid repeating R2's many-small-file timeout/waste when S1 is
  terminal, a generic official-inventory resume utility was committed/pushed at
  `817788d78da472c011d2e61f47aae2742c560b67`. Source/test SHA-256 are
  `049115f63e45eb984a6c9fd908363a92ae840cbc7b2f12869cc5aaf2ccee19a3` /
  `299499ebea6c512a6d8067593c88d6177ec0b9cf6be1b428ec157db84626b791`.
  It supports a compact-output regex, skips existing complete files, uses
  bounded workers/retries and atomically renames `.part` files only after a full
  response; it retrieves the official log but never queries kernel status or
  creates a monitor. `py_compile` and focused tests pass `3/3`. The utility has
  not yet been run on S1 output and opened no scientific data/prediction/GT.
  Full repository regression after adding it passes `393/393` in 14.14 s under
  the documented local Python-3.9 `zip(strict)` diagnostic shim.

### S1 terminal retrieval and GT-blind precision-audit boundary (2026-08-01)

- A user-authorized 20-minute heartbeat performed one bounded status query at
  `2026-07-31T17:46:23Z`; private Kaggle kernel version 1
  `itsthang333/btxrd-rad-dino-mask-bag-family-balanced-s1-pair-v1` was
  `COMPLETE`. No repeated status poll was made. Central HEAD/origin remained
  clean and synchronized at `e9c8fb9466fef24a8ef0d4feba6d9d1b15256544`;
  collaborator HEAD remained `797f191fcae8a5bb0b4ccb920d1adf635af1eff8`
  with no newer terminal G1 evidence.
- Official-inventory compact retrieval selected all `1,870/1,870` files under
  `btxrd_mask_bag_family_balanced_s1_pair_v1/`: `155,399,027` bytes, zero
  remaining `.part` files, 372 shared candidate-family files, 747 standard-arm
  files, 747 family-balanced-arm files and four pair-root files. Two initial
  local shell timeouts left one downloader child process alive; concurrent
  resume then hit Windows file locking on one `.part`. The exact stale
  downloader PID was identity-checked and stopped, after which one bounded
  resume downloaded the final file and independently confirmed the complete
  official inventory. This was a transport/process-management error only; no
  output was interpreted while incomplete. Direct Kaggle log SHA-256 is
  `4f5cda2e390c1d1445f3849a7fcad2c9ed890f213ee2a22f6c4cfd3d352e6432`.
- Frozen root hashes before any validation GT are: pair prediction freeze
  `b393174c6cf8ce8a6aaea551ab36a1893280140529336a702d9b7aceb9b6af16`,
  run manifest
  `e79e335e79f6fa9913e06286cfc433e90b673c5c0d90253d3a8b601b8ea3b252`,
  wrapper output audit
  `ea1bc6626ceabfba428755cca824eed573f24a9b723edad4c47e1915efa1460c`,
  and matched initial state
  `babcd9e31c795680ed47038ab7563e8cb8ce754dc80685b039314d37009f729a`.
  Protocol and launch-binding hashes remain exactly
  `62684fc7e01474ab64701c31a0a7d2fa1c802ffb2b5c4e8896848b94bc7e8413`
  and
  `9e77ef03d77162674bc1305440f05a09257ad4d6ac1c0157987c36a6030fe442`.
- The frozen independent GT-blind auditor stopped before writing a pass audit:
  one family-balanced bag logit for `IMG001934.jpeg` was stored as
  `-10.748274803161621` and independently reconstructed in NumPy float64 as
  `-10.748272689228461`, absolute delta `2.113933160075021e-06`, narrowly above
  the prelaunch absolute tolerance `2.0e-06`. No validation GT, baseline
  per-image table, evaluator, matched comparator, decision tool, consumer or
  BTXRD test was opened.
- A full 371-row output-only numerical diagnostic found no structural/hash or
  cohort exception before this boundary. Standard-arm maximum float64
  reconstruction delta is `1.2513269211922307e-06`. For family-balanced, only
  the single row above exceeds `2.0e-06`; the second-largest delta is
  `1.3723830765144385e-06`, and no row exceeds `3.0e-06`. Reproduction with the
  exact PyTorch float32 pooling operator gives maximum stored-vs-CPU deltas
  `1.9073486328125e-06` standard and `9.5367431640625e-07`
  family-balanced. Maximum sigmoid-probability deltas from the independent
  float64 logits are `8.53041076664951e-08` and
  `8.157106268225078e-08`, both inside the separately frozen `1.0e-07` bound.
  The evidence localizes the failure to the auditor's fixed absolute tolerance
  across GPU-float32 hierarchical reductions versus CPU-float64 reconstruction,
  not to a scientific prediction change; nevertheless the audit remains failed
  until a fail-closed, ULP-grounded correction is frozen and rerun.
- `EXP-20260731-codex-s1-family-balanced-v1` therefore remains `ĐANG LÀM` at a
  GT-blind audit boundary. Before any auditor correction or rerun, this exact
  error record must be committed/pushed. Any correction must retain the original
  failure, change no producer/prediction/protocol/scientific bytes, derive its
  tolerance from float32 representation rather than the observed metric, add
  regression tests and a machine-readable addendum, and pass the full physical
  audit before validation GT may be opened.
- **GT-blind auditor correction frozen before rerun:** after the error record was
  pushed at commit `b0818a7`, the sole numerical comparison change replaces the
  fixed bag-logit tolerance with
  `max(2e-6, 4*abs(spacing(float32(expected))))`. This retains the original
  `2e-6` bound for small logits and allows at most four representable float32
  steps for GPU-reduction versus CPU-float64 reconstruction; a regression test
  explicitly rejects an eight-ULP perturbation. Corrected auditor/test SHA-256
  are
  `6ee254d184e062927d45ff1355df998adc0caba20147acd59cb4629687ce66ce` /
  `1c8042416c282f4d72889d3e82bdc8848a95c1b4df3d97b625f3a9dea843515b`.
  The auditor now self-records its source hash and exact tolerance contract;
  every structural/hash/map/probability/provenance/safety check is unchanged.
  Machine-readable post-output, pre-GT addendum
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_postoutput_numeric_auditor_addendum.json`
  has SHA-256
  `095100c0d5b487ac0302e0b34574bc181eff8cdeba03b4c05904e987aef09237`.
  `py_compile`, focused tests `8/8`, JSON parse, `git diff --check` and full
  repository regression `394/394` in 16.60 seconds pass under the documented
  local Python-3.9 `zip(strict)` diagnostic shim. Producer, predictions,
  protocol, launch binding, GT boundary, consumer and test remain unchanged;
  the corrected physical audit has not yet been rerun at this point.
- **Corrected full physical audit pass before GT:** the same immutable compact
  output then passed the corrected independent auditor with status
  `MATCHED_PAIR_PREDICTIONS_PHYSICALLY_VERIFIED_GT_BLIND`. It physically
  verified 371 shared family payloads plus 371 candidate-score payloads and 371
  prediction maps for each arm, `155,401,323` bound evidence bytes, exact
  protocol/source/split/cache/baseline/initial-state/history/checkpoint/T4x2
  provenance and all safety flags. Tracked audit
  `artifacts/kaggle/rad_dino_mask_bag_family_balanced_s1_pair_v1/kernel_version1_complete_gt_blind_output_audit.json`
  has SHA-256
  `fb0f92706432e36d856ccbe6beff7ef6bde2063ec293810b665db3ed85b74b3a`;
  it self-binds corrected auditor SHA-256
  `6ee254d184e062927d45ff1355df998adc0caba20147acd59cb4629687ce66ce`
  and pair prediction-freeze SHA-256
  `b393174c6cf8ce8a6aaea551ab36a1893280140529336a702d9b7aceb9b6af16`.
  Absolute candidate-count/probability Spearman is
  `0.4674388986415078` standard and `0.46716896371012795`
  family-balanced, both below frozen geometry-v3 baseline
  `0.48137777593654113`; this GT-blind diagnostic alone does not authorize
  promotion. Validation GT, baseline per-image table, consumer and BTXRD test
  still had not been opened at audit freeze time. The audit artifact/hash must
  be committed/pushed before the predeclared post-freeze evaluator may run.
- **Both arm evaluations frozen after the audited GT boundary:** after audit
  commit `6ccfe1a`, the predeclared evaluator opened validation GT separately
  for each already frozen arm with identical 10,000 complete-group bootstrap
  replicates and seed `20261021`. The first standard invocation stopped at
  Python import before data access because `runpy` did not add `project/` to
  `sys.path`; the successful retry added only that module-search path plus the
  documented Python-3.9 `zip(strict)` shim, with no input/hash/prediction/source
  or seed change.
- Standard Dice overall/small/medium/large is
  `0.23328658248335798 / 0.1062023689932446 / 0.3580649939947074 /
  0.39783494021966315`; family-balanced is
  `0.2332033020034355 / 0.1062023689932446 / 0.3578521661015722 /
  0.39783494021966315`. Both arm gates fail against geometry-v3: each reduces
  regret only for large, regresses overall/small/medium, raises complete misses
  from 53 to 57 and remains below all four operational goals. Image AUROC is
  `0.8233550337130899` standard and `0.8230353406184608`
  family-balanced.
- The exact five evaluator outputs for each arm were copied byte-for-byte into
  tracked `standard_evaluation/` and `family_balanced_evaluation/` evidence.
  Standard evaluation-audit/per-image SHA-256 are
  `f5d6f44f4dd02fa7b12cf71df2638d0fcc6b912626f80694b17e048e5f86cc1b` /
  `57407f5304c6ed6f3ae1c37d9d40ed4a2a07b798f7b3a2e5d014d3bf85b14157`;
  family-balanced values are
  `6bd402ba01825b4a685cd7f13efa4e109dbdb5abf08c7e32cd7fd5a539032ee1` /
  `6f81ea1a1c2f0ec0668cb058462ce66182fa0f63dd47ee724da2f126bab1894d`.
  Evaluation-freeze artifact
  `kernel_version1_postfreeze_evaluation_freeze.json` has SHA-256
  `3daaa571c193388d89d4f1a1bd6a1e4fd5ba3f2c21ae22ca0f72c77f86573f1a`.
  It is frozen before the matched comparator: comparator/decision have not run,
  consumer remains unauthorized and BTXRD test remains untouched.
- **Matched-comparator integration error before result:** the first invocation
  of predeclared comparator source SHA-256
  `24c625cfc50740d9cb633906d60ae81089e3960d3eec4b3ead6f3ce89ebaffad`
  failed closed while reading the hash-locked family-balanced CSV because it
  required a `selected_area_ratio` field that
  `evaluate_mask_bag_selector_arm.py` does not serialize. No comparison output
  or metric was created and the comparator did not reopen GT. Static source
  inspection localizes this to a vestigial schema requirement: the field is
  only range-validated in `_read` and is never used in identity matching,
  paired Dice/miss rows, subgroup metrics, bootstrap, decision inputs or output.
  The already frozen evaluation CSVs must not be rewritten. This boundary is
  recorded and pushed before any comparator correction; a correction may only
  remove that unused required-field/range check, add an exact evaluator-schema
  integration regression test and freeze a machine-readable addendum. All
  per-image hashes, comparator seed/replicates, paired fields, calculations,
  decision gates, consumer lock and BTXRD test lock must remain unchanged.
- **Comparator correction frozen before rerun:** after pushing the error record
  at `40c16c4`, the comparator removed only `selected_area_ratio` from its
  required-field and range-validation lists. The frozen evaluator CSVs and all
  used paired fields/calculations remain byte-identical. Corrected comparator
  source/test SHA-256 are
  `2b868f93890f76a93f528f79f219a9d3caac4389af098e671e8021203481f119` /
  `f29650ad9139bc2c894274cfe338bb69890ecb77133286e5e90ce0825324c7b6`.
  Machine-readable no-GT-reopen addendum
  `artifacts/research_protocols/rad_dino_mask_bag_family_balanced_s1_pair_v1_postfreeze_comparator_schema_addendum.json`
  has SHA-256
  `54db1347668407f84d2f989e046fd206825ee5b6f863de52688fa1b6bf28d4e7`.
  `py_compile`, JSON parse, `git diff --check`, focused comparator/decision tests
  `6/6` and full repository regression `394/394` in 14.99 seconds pass under the
  documented Python-3.9 `zip(strict)` shim. The corrected comparator has not yet
  been rerun; GT remains closed to it, no consumer was trained and test remains
  locked.
- **Terminal matched comparison/decision:** the corrected comparator read only
  the two hash-locked evaluation CSVs and did not reopen GT. Family-balanced
  minus standard Dice is `-8.328047992246669e-05 / 0 /
  -0.00021282789313519266 / 0` for overall/small/medium/large; overall CI95 is
  `[-0.0002526218377948036, 0]`, with zero recovered misses and zero lost
  overlaps. Comparator output/per-image SHA-256 are
  `cfd360ac7b760b799b07276e88c8f2cc05a2f11f8e2fbdefcfd9711be2196124` /
  `0c0ecdb103f526d7922e2c4b9672b12cdbf52d1333c46922a8a6d0b6dbcce54d`.
- The predeclared decision source SHA-256
  `90bf0d92e6a53e3773418810fb81a12c0069a944e47d6c25a8a5c5956e8aaff7`
  returned `FAIL`; decision output SHA-256 is
  `7ff448b3783b796a7899481fad3d31a943bf05581ccedf2a00bce037ae871404`.
  Family balancing reduces regret in zero required subgroups, slightly regresses
  overall against standard, and the family arm itself fails every operational
  Dice goal. Therefore neither the shared descriptor-only residual nor
  hierarchical family-balanced SmoothMax is learned/adopted as a performance
  improvement. S1 is terminal rejected; consumer remains unauthorized, no
  consumer was trained, and BTXRD test remains untouched.
- Consolidated terminal result audit
  `artifacts/kaggle/rad_dino_mask_bag_family_balanced_s1_pair_v1/kernel_version1_terminal_result_audit.json`
  has SHA-256
  `6733eef1b9e2ad302285395c7fdb897b3035659d2d46314a8f29643d0286687b`.
  It preserves the official inventory/log, pair/audit/addendum/evaluation/
  comparator/decision hashes, all four technical error boundaries, exact Dice,
  paired CIs and safety flags. The next scientific claim must first fetch and
  read the collaborator branch for any now-terminal G1 result; only a terminal
  audited improvement may be inherited, and a successor must change an
  untested selector bottleneck rather than rerun R1/R2/S1 or proposal supply.

### EXP-20260801-codex-r3-critical-relation-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`;
  `HOÀN THÀNH — TERMINAL REJECTED AT GT-BLIND AGREEMENT GATE`.
- **Registered:** `2026-07-31T18:24:00Z` (`2026-08-01` ICT); registration base
  commit `93c7c13c3b1c8f27bdd7a992a1de11adc814562e`. The exact registration
  commit will be recorded in the immediate successor note after push; no R3
  runner, protocol, binding, kernel launch, training or prediction exists yet.
- **Coordination/non-duplicate audit:** immediately before registration, the
  complete central and collaborator logs were read after fetching both branches.
  Collaborator head remains
  `797f191fcae8a5bb0b4ccb920d1adf635af1eff8`; active
  `EXP-20260731-codex-rich-gallery-g0g1-v1` still has no terminal audited
  actual-Dice result. R3 does not alter proposal supply and therefore does not
  duplicate G0/G1. It inherits terminal rejected
  `EXP-20260731-codex-r1-normal-prototype-v1`,
  `EXP-20260731-codex-r2-affinity-residual-v1` and
  `EXP-20260731-codex-s1-family-balanced-v1` only as negative constraints: no
  normal-prototype, cached token-affinity or family-balanced-pooling mechanism
  is adopted as an improvement.
- **Objective/hypothesis:** test the previously unexecuted DSMIL-style
  critical-instance relation on the same proven-oracle gallery. For each bag,
  the frozen accepted geometry-v3 scorer supplies independent candidate logits;
  its detached argmax candidate is the critical instance. A zero-initialized
  residual compares every 1,156-D candidate descriptor with that critical
  descriptor using `[h_i, h_m, h_i-h_m, h_i*h_m, cosine(h_i,h_m)]`, then adds a
  per-candidate correction. This directly targets within-image candidate
  ranking that R1/R2/S1 did not model, while starting exactly at geometry-v3.
- **Exact inherited inputs:** selector cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`
  and manifest
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`;
  split
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  accepted baseline checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`
  and prediction freeze
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`;
  geometry-v3 Dice
  `0.2454823867797678 / 0.11708057891440651 / 0.37713551529480416 /
  0.3894126471276201`; immutable oracle
  `0.4090755342486002 / 0.2227494852063559 / 0.5941470844279589 /
  0.6418253674184405`. R3 changes no candidate, candidate order, family/source
  identity, mask geometry, baseline parameters, map construction or evaluator.
- **Frozen fit intent before implementation:** train only the zero-initialized
  critical-relation residual; keep the geometry-v3 base scorer frozen. Use the
  original geometry-v3 normalized SmoothMax (`temperature=0.2`), image BCE,
  self-guided instance loss (`weight=0.25`, warm-up 2 epochs: all candidates in
  negative bags are negative and only the detached current winner in positive
  bags is positive), aligned original/flip consistency (`weight=0.10`), AdamW
  `lr=3e-4`, weight decay `1e-4`, batch 16, seed 42 and exactly 16 fixed-final
  epochs. Hidden dimension is 128. No early stopping, validation loss,
  architecture/weight/epoch alternative, candidate target, pseudo mask or
  segmentation/subgroup input is permitted.
- **GT-blind gates before evaluation:** before optimization, freeze exact
  zero-residual equality and base original/flip critical-index agreement. After
  16 epochs, all 371 validation candidate-score vectors and WTA maps must be
  physically frozen and independently audited. Absolute candidate-count versus
  bag-probability Spearman must not exceed the inherited R1 ceiling
  `0.5013777759365411`; final original/flip critical-index agreement may not
  drop more than `0.01` below its frozen base value. Failure blocks GT
  evaluation and rejects R3 without a rerun or sweep.
- **Post-freeze promotion/output:** only after the GT-blind audit passes may the
  unchanged evaluator read validation GT. Report Dice, paired 10,000-group
  bootstrap CIs, selected-to-oracle regret, candidate-score/quality rank
  correlation, misses and image AUROC for overall/small/medium/large. Mechanism
  pass requires medium regret reduction, regret reduction in at least two tumor
  subgroups, no overall Dice regression and no increase in absolute
  candidate-count/miss association. Adoption/consumer authorization additionally
  requires simultaneous Dice at least
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336`, overall CI95 lower bound
  above zero, no subgroup mean decrease, no miss increase and AUROC at least
  `0.75`. Otherwise R3 is terminal rejected and not learned as an improvement.
- **Compute/safety:** prepare and statically test locally, but run scientific
  training/inference on one private Kaggle T4x2 kernel (training on T4:0,
  validation shards on both T4s with real CUDA work). Training supervision is
  binary image-level labels only. Validation GT remains closed until audited
  prediction freeze; no consumer before full operational pass; BTXRD test is
  locked.
- **Literature/technical basis:** the critical-instance relation is transferred
  from Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning Network for
  Whole Slide Image Classification with Self-supervised Contrastive Learning*,
  CVPR 2021,
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html .
  The requirement to judge instance ranking rather than infer it from bag
  classification follows Jang and Kwon, *Are Multiple Instance Learning
  Algorithms Learnable for Instances?*, NeurIPS 2024,
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html .
  The project-specific frozen design source is
  `artifacts/literature_reviews/relational_mil_medium_selection_design_2026-07-28.md`;
  R3 transfers only the relational principle, not pathology tiling,
  self-supervised pretraining, instance labels or an auxiliary bag head.
- **Static implementation progress (2026-08-01 ICT):** the image-label-only R3
  fitter and two-T4 runner now exist at
  `project/models/mask_bag_critical_relation_training.py` and
  `project/run_mask_bag_critical_relation_arm.py`, SHA-256 respectively
  `b29ffc4b77680f9e9b0046238506f738cf1c855cfc2f33769726b034d6d67042` and
  `a00bc05f289a7afe6653264fc09053ff5ed59e372be75de01024ea9447932686`.
  The fitter freezes the accepted base scorer, creates one deterministic
  zero-residual state, physically audits exact base equality and original/flip
  base-critical agreement before optimization, then performs the single fixed
  16-epoch fit declared above. The runner requires exactly T4x2, trains only on
  T4:0, scores disjoint `186/185` validation shards on both devices, freezes
  all 371 score vectors/maps, and records the count/probability and final
  original/flip selected-index gates without importing an evaluator or GT.
- Tests
  `tests/test_mask_bag_critical_relation_training.py` and
  `tests/test_run_mask_bag_critical_relation_arm.py` have SHA-256
  `a0b4e639f736e24419e76daef09d09f2415009c1d1cef09a6f77b414fb8bb47e` /
  `5a5dade7de8c2beab1aa4e59f7279fda517da8c609306df5aec7d43faebbf4fa`.
  The focused relational/R3 suite passes `12/12`; full repository regression
  passes `401/401` in 14.80 seconds under the already documented Python-3.9
  `zip(strict=...)` compatibility shim. The first default full-regression call
  produced the known environment-only boundary `7 failed, 394 passed` because
  Python 3.9 rejects the Python-3.10 `zip(strict=...)` keyword. One initial R3
  static test also matched substring `dice` inside `indices`; it was corrected
  to a word-boundary assertion before the passing runs. Neither error opened
  scientific input or changed the R3 mechanism.
- This is static preparation only. No R3 protocol/binding/kernel launch,
  training, validation prediction, GT/test access or consumer training has
  occurred. Independent output-auditor closure and exact protocol/source hash
  freeze remain required before launch.
- **R3 protocol and independent-auditor freeze (2026-08-01 ICT):** static review
  found that the first runner froze only aggregate agreement values, which an
  independent auditor could not reconstruct physically. Before protocol freeze
  or scientific execution, the runner was corrected to write one
  `gt_blind_diagnostics.csv` row per validation image with candidate count, bag
  probability, frozen-base critical agreement and final selected-index
  agreement. Corrected runner/test SHA-256 are
  `bd8bca7d02802ce5881193f4ba0b1286a216bc4f4d077b34de3da5d14c9ae05c` /
  `00fcd4fd20f7847a906918eed989659f4491b5be700f5dbb858c5a37ef3ccbe6`;
  exact scientific source commit `84867698dbc652957bd1a1430f4b9d32fa399119`
  was pushed before protocol creation. This is an auditability correction, not
  a mechanism or gate change.
- Exact protocol
  `artifacts/research_protocols/rad_dino_mask_bag_critical_relation_r3_v1.json`
  SHA-256
  `f7253df27444c1b56706ac19646441a7b2d2a7374a4d6888b8e97d26c2c3fd03`
  now freezes the source closure, critical relation, one 16-epoch fit, T4x2
  execution, physical predictions/diagnostics, both GT-blind gates and all
  post-freeze mechanism/adoption gates. Independent auditor
  `project/audit_mask_bag_critical_relation_r3_output.py` SHA-256
  `3a1078ba21e9215890b50bee8c057d44c59cf95ea7536f2cef763dddadc1d953`
  pins the generic physical-output helper `3cc5feee...`, requires a separate
  frozen prelaunch binding, verifies cache/source/T4x2/checkpoint/history and all
  371 physical scores/maps, and independently recomputes count/probability plus
  base/final flip-agreement gates without an evaluator or GT. Auditor test
  SHA-256 is
  `792d581686201c68fdfb975587a3962aa2717e11a7e0bd67eac54c95d1eb0de9`.
- Focused R3 suite passes `15/15`; full repository regression passes `404/404`
  in 14.80 seconds under the documented Python-3.9 `zip(strict=...)` shim.
  Static-readiness JSON SHA-256 is
  `f85a68fd6a627ccfa8cb6b95e423776847d0732f8ec1ea06d0334525296cc5e3`.
  Wrapper/binder/final binding remain unfinished, so no Kaggle kernel was bound
  or launched; validation GT, consumer and BTXRD test remain locked.
- **R3 fail-closed wrapper/binder readiness (2026-08-01 ICT):** unbound wrapper
  `project/kaggle_wrappers/run_mask_bag_critical_relation_r3_v1.py` canonical-LF
  SHA-256
  `c350238ad69231e6714c12d8c7cd8b7fc38b5d2a1702c22a55a0ed7ac6b6aa2e`
  retains `KERNEL_VERSION=0`, `LAUNCH_BINDING_READY=False` and checkout
  `UNBOUND`. It reconstructs the exact frozen split, verifies the transported
  accepted baseline and cache, requires real T4x2 CUDA work, runs the frozen
  source/tests/one R3 fit, and checks 371 physical maps, score payloads and
  per-image GT-blind diagnostic rows before writing its wrapper audit. Wrapper
  test SHA-256 is
  `076aff8e42a681c2bddbb32f8af852442785688782d129c95b23e0db2f27d48a`.
- One-time binder
  `project/bind_mask_bag_critical_relation_r3_wrapper.py` SHA-256
  `0bd3b2be99049eff89a36c375089221863e5cdee25e657b9ae3a12895d1a93d7`
  performs exactly three external replacements, requires exact inverse
  reconstruction, verifies protocol and every frozen source hash at the chosen
  execution checkout, refuses existing outputs and emits the schema already
  required by the independent auditor. Binder test SHA-256 is
  `ad6b0685df2ce88aa482abc8e83460e0c518ac6de5bb9d6b7c0cafb4475b1de1`.
  Focused wrapper/binder/auditor tests pass `7/7`; full regression passes
  `408/408` in 15.27 seconds under the documented Python-3.9 compatibility
  shim. Readiness JSON SHA-256 is
  `3ded95b79fb9a5f4822d8c873a00f4fddd963f95864e2be636f58349e720b952`.
  No real binding or launch exists yet; GT/test and consumer remain locked.
- **R3 kernel-v1 real binding, prelaunch only (2026-08-01 ICT):** after verifying
  clean synchronized central/collaborator state and unchanged active G1 scope,
  the one-time binder froze kernel version 1 to execution checkout
  `2e958f1bf7f5c0290cb54f6f45f8b0c9432c3168`. Bound wrapper SHA-256 is
  `21a43a06e8e6c7c7558f77c74df71465219758c9c85a431c2e7c8db1a12e27ef`;
  launch-binding SHA-256 is
  `4b49a0fca65c4e433446e752460aad90cc2153a2fef63faf553ccb255ca4e93d`.
  Exact inverse reconstruction, protocol/source closure and scientific-source
  ancestry passed. Kaggle metadata SHA-256 is
  `23f80f0342027f30a8e881f36df69e39d1657f16b9e7e6183c755c7b86f45e99`
  and retains the accepted baseline transport dataset, accepted selector-cache
  kernel source and `NvidiaTeslaT4` shape.
- Final prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_critical_relation_r3_v1_kernel_v1_final_prelaunch_audit.json`
  SHA-256
  `ee0678755581a49792c8ac4cb46da97915ea088a5c29cad845dadae4f4d0ec6d`
  records all binding/input/safety/test hashes. The binding and audit must be
  committed and visible on central origin before the first Kaggle push. At this
  note no Kaggle version exists yet; no training/prediction, validation GT/test
  access or consumer training occurred.
- **R3 launch:** after binding/audit commit `e0ef4d6` was visible on
  `origin/research-wsss-improvement`, the exact bound payload was pushed once as
  private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-critical-relation-r3-v1` version 1 on
  Kaggle `NvidiaTeslaT4` (expected T4x2). Push succeeded; exactly one immediate
  bounded status check returned `KernelWorkerStatus.RUNNING`. No output was
  requested and no repeat poll/extra monitor was created. Launch-record
  artifact SHA-256 is
  `b3d5a331c59045fcb9896b1f1ff7a8a9bb8e91c2700ecf385678fa443092beb4`. Validation
  GT, consumer and BTXRD test remain locked until terminal physical audit and
  both GT-blind gates pass.
- **R3 terminal COMPLETE and independent GT-blind rejection:** one later bounded
  status check found kernel version 1 `COMPLETE`. The official-output downloader
  was resumed to a complete inventory of 2,566 Kaggle files plus its direct log
  (2,567 local files, 196,064,064 bytes, zero `.part`). Direct log SHA-256 is
  `f5735b88eb1ab7807ba5cb9e2038921e929e5cca82c24a219000866c1533f055`.
  The first local invocation used the wrong existing Python environment and
  stopped before Kaggle API access because `kaggle` was absent. The first
  correct-interpreter invocation exceeded the local 124-second shell timeout
  while its identity-checked child continued; a same-destination resume found
  2,556 existing files, downloaded the remaining 10 and wrote the direct log.
  These are download/tooling boundaries only; no incomplete tree was audited.
- Runtime evidence reproduces exact wrapper `21a43a06...`, checkout `2e958f1`,
  protocol `f7253df...`, split `85511ee1...`, accepted cache freeze
  `2f6290cd...`/3,352 records, baseline archive `8857eb6d...`, and two real Tesla
  T4 convolutions. All 371 candidate-score payloads, maps and GT-blind
  diagnostic rows were physically frozen. Prediction-freeze/run/wrapper audit
  SHA-256 are `11ed9eb9aa4c5b97b0f14cae7458bbffbb7a91a844b9441b71cb5b9153fece98` /
  `64fa8dc4e6aee56b6d5de3f46189c04af069bfdba762ab459f1cdb536ce25cdf` /
  `e1be8fdee31b1fc0a219a2f45541ae46ee2f21ed526a1d8176f8950b625b6ca7`.
- Independent auditor output SHA-256
  `683ef9efe1993614770f58966f1f5427d2ceab51f831321cd51c0c1f6292394f`
  has status `PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_FAIL` and
  verifies 77,499,181 physical bytes. Absolute candidate-count/probability
  Spearman is `0.46925209554046915`, below the frozen ceiling
  `0.5013777759365411`, so the count shortcut gate passes. However frozen-base
  original/flip critical agreement `0.6522911051212938` falls after fitting to
  final selected-index agreement `0.6118598382749326`, delta
  `-0.0404312668463612`, below the minimum allowed `0.6422911051212938`.
  The predeclared agreement gate therefore fails.
- Per protocol, R3 is terminal rejected before validation segmentation GT. No
  Dice, subgroup, regret or oracle-quality metric was read; there is no rescue,
  rerun or sweep and the critical-relation mechanism is not adopted as a
  performance improvement. Consumer remains unauthorized and BTXRD test stays
  locked. Consolidated terminal audit SHA-256 is
  `d74d28ca66430524764fb5168955bcb0e710665d25caa5b65357190f08ef9632`.

### EXP-20260801-codex-r4-orbit-averaged-relation-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`;
  `HOÀN THÀNH — GATE FAIL`.
- **Registered:** `2026-07-31T19:57:24Z` (`2026-08-01` ICT); registration base
  commit `dc4222fb78b0e6f6c055c90fe83be6400ea923ac`. Exact claim commit will be
  recorded after this note is pushed. No R4 implementation, protocol, binding,
  kernel launch, training or prediction exists at registration.
- **Coordination/non-duplicate audit:** the complete central and collaborator
  logs were reread after fetching both branches. Collaborator head remains
  `797f191fcae8a5bb0b4ccb920d1adf635af1eff8`; active
  `EXP-20260731-codex-rich-gallery-g0g1-v1` still changes proposal supply and has
  no terminal audited actual-Dice result. R4 changes only same-gallery selector
  view aggregation and does not create/merge proposals, so it does not
  duplicate G0/G1. R1/R2/S1/R3 remain terminal negative evidence; none of their
  mechanisms is claimed or adopted as a demonstrated improvement.
- **GT-blind predecessor evidence and hypothesis:** R3 independently verified
  371 rows show base/final flip-agreement transitions `0→0:102`, `0→1:27`,
  `1→0:42`, `1→1:200`. Thus R3 created 15 net new disagreements; the 69 changed
  rows have mean candidate count `56.5652` versus `55.8775` for unchanged rows,
  so the failure is not explained by a simple high-count cohort. R4 tests a
  distinct correction: before selecting a critical instance or applying any
  relation, compute one aligned horizontal-flip orbit average for every
  candidate descriptor and frozen-base logit. Select the detached critical
  candidate from the averaged frozen-base logits and apply one zero-initialized
  relation residual to the averaged descriptor. This makes view swapping
  invariant by construction while testing whether relational ranking has value
  after the R3 instability is removed.
- **Exact inherited inputs:** selector-cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  manifest `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`,
  split `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  accepted checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`
  and prediction freeze
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`.
  Geometry-v3 Dice remains `0.2454823867797678 / 0.11708057891440651 /
  0.37713551529480416 / 0.3894126471276201`; immutable oracle remains
  `0.4090755342486002 / 0.2227494852063559 / 0.5941470844279589 /
  0.6418253674184405`. Candidate identity/order, masks, family/source IDs, map
  construction and evaluator are unchanged.
- **Frozen fit intent before implementation:** create averaged descriptor
  `0.5*(original+aligned_flip)` and averaged independent logit
  `0.5*(base_original+base_flip)`; use the same 1,156→128 embedding and
  `[h_i,h_m,h_i-h_m,h_i*h_m,cosine]` zero-initialized residual as an explicit
  falsifiable variant, not as an inherited improvement. Keep the complete base
  scorer frozen. Fit exactly 16 fixed-final epochs with normalized SmoothMax
  temperature `0.2`, binary image BCE, self-guided instance loss weight `0.25`
  after 2 warm-up epochs, AdamW `3e-4/1e-4`, batch 16, hidden 128 and seed 42.
  There is no separate flip-consistency term because the orbit-averaged branch
  is algebraically identical under view swap. No early stopping, validation
  loss, alternative epoch/weight/architecture, candidate target, segmentation
  pseudo-target or subgroup input is allowed.
- **GT-blind gates before evaluation:** before optimization, freeze bit-exact
  equality to accepted Geometry-v3 averaged TTA candidate logits and exact
  equality after swapping original/flip inputs. After the fixed fit, physically
  freeze all 371 candidate-score vectors/maps and independently verify exact
  view-swap equality. Absolute candidate-count/bag-probability Spearman may not
  exceed `0.5013777759365411`. Any failure rejects R4 before GT with no rerun,
  sweep or rescue.
- **Post-freeze decision:** only after independent physical audit and both
  GT-blind gates pass may the unchanged evaluator read validation GT. Mechanism
  pass still requires medium regret reduction, regret reduction in at least two
  tumor subgroups, no overall Dice regression and no increase in absolute
  candidate-count/miss association. Adoption/consumer additionally requires
  simultaneous Dice at least `0.34024039 / 0.17895493 / 0.51244178 /
  0.49370336`, overall CI95 lower bound above zero, no subgroup mean decrease,
  no miss increase and image AUROC at least `0.75`.
- **Compute/safety:** static local tests only; one future private Kaggle T4x2
  kernel, training on T4:0 and disjoint validation shards on both T4s. Training
  supervision is image-level labels only. Validation GT remains closed until
  prediction freeze and independent gate pass; consumer and BTXRD test remain
  locked.
- **Technical basis:** R4 reuses only the relational-principle reference from
  Li, Li and Eliceiri, DSMIL, CVPR 2021,
  https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_With_Self-Supervised_CVPR_2021_paper.html ,
  and the project’s already frozen aligned-flip TTA contract. It does not claim
  DSMIL or R3 is beneficial; this new arm tests the failure-derived
  orbit-averaging correction under the instance-ranking caution of Jang and
  Kwon, NeurIPS 2024,
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html .
- **Claim/source progress:** exact claim commit is
  `0c2085ace6166057bcd94e2abb427747a16f4182`. Static image-label-only fitter and
  T4x2 runner are now implemented at
  `project/models/mask_bag_orbit_relation_training.py` and
  `project/run_mask_bag_orbit_relation_arm.py`, SHA-256 respectively
  `44ecd9c6ad1779942e9cc0df31634fd5d83204a97c7e89c75c64507d66fecf71` /
  `68888d7d81cbe35cb0b5b1b1f46881469fe8949524c2371b948308e13591184c`.
  The fitter constructs one averaged descriptor/base-logit branch, verifies
  exact zero identity and exact input-swap invariance, freezes the base scorer,
  and exposes only the declared image BCE/self-guided instance objective. The
  runner freezes per-image swap diagnostics, count/probability gate, all 371
  score vectors/maps and T4x2 shard evidence before any evaluator could run.
- Tests `tests/test_mask_bag_orbit_relation_training.py` and
  `tests/test_run_mask_bag_orbit_relation_arm.py` have SHA-256
  `7a074178f6efc87f74bd631106d27adbe964c81ffe2f31012a63d6b98417b9e9` /
  `314492cb61098dbacbffb22102b2e90e85e4133d4fcefed1dfc6274c412ed2c9`.
  Focused R4/relational suite passes `12/12`; full repository regression passes
  `415/415` in 17.39 seconds under the documented Python-3.9
  `zip(strict=...)` shim. This is static preparation only: no protocol,
  binding, kernel launch, scientific training/prediction, validation GT/test or
  consumer access occurred.
- **R4 protocol/auditor freeze:** exact protocol
  `artifacts/research_protocols/rad_dino_mask_bag_orbit_relation_r4_v1.json`
  SHA-256
  `00d1ba1b7611c8320e87948d88dac06af87c0b29d1e62f875e79105fc41c0aae`
  freezes source commit `84f3dca`, the averaged branch, one fixed 16-epoch fit,
  T4x2 execution, physical outputs, exact swap gate, count ceiling and unchanged
  post-freeze promotion/adoption rules. Independent auditor
  `project/audit_mask_bag_orbit_relation_r4_output.py` SHA-256
  `f7d6e2823bda726e3f13cc7a87fe3465ee9fce78335f4873be755c32f7c234f6`
  requires an external frozen launch binding, pins generic physical helper
  `3cc5feee...`, verifies source/cache/baseline/T4x2/checkpoint/history and all
  371 maps/scores, then independently reconstructs both GT-blind gates from the
  physical diagnostic CSV. Auditor test SHA-256 is
  `e0530316386848780d04672bb3798ade7002df2c51de8b7f1afac7ef4add1837`.
- Focused R4 suite passes `15/15`; full repository regression passes `418/418`
  in 15.03 seconds under the documented Python-3.9 shim. Static-readiness JSON
  SHA-256 is
  `b6d545cdc7520db50c5c2fd4598899be3f4da9aecec0f5aa5c86c714648e8351`.
  Wrapper/binder remain unfinished, so no binding or launch exists and
  validation GT/test plus consumer remain locked.
- **R4 wrapper/binder readiness:** fail-closed wrapper
  `project/kaggle_wrappers/run_mask_bag_orbit_relation_r4_v1.py` canonical-LF
  SHA-256
  `4924779ad3ce01cdd0c410f5dfbfdad78f1ba4c2650012f11503517df2f67a19`
  remains unbound at kernel version 0/checkout `UNBOUND`, verifies exact source,
  split, accepted baseline/cache, T4x2 real work and all 371 physical outputs.
  Wrapper test SHA-256 is
  `437af08d550dacb3da83b10fe63a834e1a06697e089e3d82433941864620aa19`.
  One-time binder SHA-256
  `a72e4d1612fad1f4f65d1fb31b2cfba968821308b436b8201e8540e46e87fec3`
  performs exactly three external replacements, inverse reconstruction and
  checkout protocol/source verification; binder test SHA-256 is
  `1e9c498aa5fcac319aad0e70b34e1bfe50cba08bbd4d83df1aa4e263c669a255`.
- Focused wrapper/binder/auditor tests pass `7/7`; full regression passes
  `422/422` in 16.05 seconds under the documented Python-3.9 shim. Readiness
  JSON SHA-256 is
  `181d4fed5f91f72aa4f8da94eb4069bde193bc04e7a57f06bf5220be1955d140`.
  No real binding/launch or scientific output exists; GT/test and consumer stay
  locked.
- **R4 real binding, prelaunch only:** one-time binder froze kernel version 1 to
  execution checkout `2360a032b51f3e1f9b0b0ef9c9aa7ad9497385d3` after wrapper/binder
  source commit `2360a03` was visible on central origin. Bound wrapper,
  launch-binding and metadata SHA-256 are
  `4ff92612131ba461d9201b48a3d271f26079ef51afc78a71edc5dcf3a0d9bde9` /
  `4315ec022fad086ac9ce179314d7a1fdfc59e53252a0573c0c96dffacbcac4fb` /
  `5834705b38130979148ea69ab842ce496c5e67320c98952bab89f3313ec33ca1`.
  Exact inverse reconstruction, source closure and ancestry passed. Final
  prelaunch audit SHA-256 is
  `a1bec7958b0ba51bf25b3376c6dfd5230d90583c00a4eafb67d283f86e65f76d`.
  Binding/audit must be pushed centrally before Kaggle; at this note no kernel
  version exists and no GT/test/consumer/scientific output was opened.
- **R4 launch:** after binding/audit commit `a5f4b6d` was visible on central
  origin, the exact payload was pushed once as private Kaggle kernel
  `itsthang333/btxrd-rad-dino-mask-bag-orbit-relation-r4-v1` version 1 on
  `NvidiaTeslaT4` (expected T4x2). Push succeeded; one immediate bounded status
  check returned `KernelWorkerStatus.RUNNING`. No output request, repeat poll or
  extra monitor followed. Launch-record SHA-256 is
  `482b78e909d1b461f0ec7711b75e1816a0c65419aa2741c0dba3c0ec607d2f61`.
  Validation GT/test and consumer remain locked.
- **R4 terminal retrieval and local tooling boundaries:** one later bounded
  status check found kernel version 1 `COMPLETE`. The official-inventory
  downloader ultimately retained all `2,591` official files plus its direct
  log (`2,592` files, `196,271,163` bytes, zero `.part`); direct-log SHA-256 is
  `9b352edb62d30c1cbaed45d9f19d7420184abc5fd379161c90b4347b258162ef`.
  Its first invocation used a relative destination: after `2,584` atomic files
  had completed, the resolved-path guard rejected one valid output path. No
  partial file or scientific audit was produced. The same destination was
  identity-checked and resumed once with an absolute path, retaining those
  files and downloading only the remaining seven without another status query.
  The first independent-auditor invocation then received the inventory root
  instead of the nested scientific root and failed closed on a missing
  top-level `prediction_freeze.json`; it opened no GT and serialized no audit.
  The identical auditor/protocol/binding was rerun on the correct immutable
  nested root. These are retrieval/invocation errors only, not scientific
  retries or prediction changes.
- **Independent GT-blind physical pass before evaluation:** audit
  `artifacts/kaggle/rad_dino_mask_bag_orbit_relation_r4_v1/kernel_version1_complete_gt_blind_output_audit.json`
  SHA-256
  `4ca0980554a9869b27f581c274108701aeba956df627ee07ba8f3482e5e9b1b7`
  has status `PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS`.
  It reproduces exact wrapper `4ff92612...`, launch binding `4315ec02...`,
  protocol `00d1ba1b...`, source commit `84f3dca`, split `85511ee1...`, cache
  freeze `2f6290cd...` and T4x2 evidence, and physically verifies `371` score
  payloads plus `371` maps (`77,497,835` bytes). All `371/371` diagnostic rows
  are exactly invariant under view swap. Absolute candidate-count/probability
  Spearman is `0.47015715758272214`, below the frozen ceiling
  `0.5013777759365411`. Prediction-freeze and score-manifest SHA-256 are
  `2cebdc7a9bd1f20a836c7f2452dd03ad8ee3365bb429faa75c5eba2b28a80855` /
  `fad32a9d0b109aef9532e28dfc769ad67024678ce5db850228ea5d4e988006c0`.
  Only after this audit passed did the evaluator open validation GT.
- **Post-freeze R4 result:** the unchanged evaluator SHA-256 `ccc3a493...`
  used the hash-locked corrected geometry-v3 rows `a26143d0...`, `10,000`
  complete-group bootstrap replicates and seed `20261012`. R4 Dice
  overall/small/medium/large is
  `0.23316225547412966 / 0.1065303155319818 / 0.361635169733516 /
  0.3805707292455786`, versus geometry-v3
  `0.2454823867797678 / 0.11708057891440651 / 0.37713551529480416 /
  0.3894126471276201`. Deltas are
  `-0.012320131305638133 / -0.010550263382424716 /
  -0.015500345561288141 / -0.008841917882041484`; CI95 values are
  `[-0.02490438,-0.00266649] / [-0.02957012,+0.00200733] /
  [-0.03680769,-0.00137143] / [-0.02381435,0]`. Thus overall and medium
  regressions are independently nonzero at the paired-CI gate.
- Selected-to-oracle regret worsens in every corresponding cohort from
  `0.16359315/0.10566891/0.21701157/0.25241272` to
  `0.17591328/0.11621917/0.23251191/0.26125464`; complete misses rise
  `53→56` overall (`33→35` small, `18→19` medium, large unchanged at two).
  Candidate-count/miss association improves `0.31350741→0.29424646` and image
  AUROC `0.82018717` passes, but these isolated gates cannot rescue the failed
  regret, Dice, subgroup, miss or operational-goal checks. Evaluation output
  SHA-256 values for `evaluation_audit.json`, `gate_decision.json`,
  `paired_comparison.json`, `per_image.csv` and `summary.json` are respectively
  `9e9cbd4421b4bf9b322b7a48d3939fa83a52a5193ead5158b9c104323ce9e509`,
  `e9c4e79962294931819be1c78f36ab350178d0bb6c93b65ded7ddb2fbf826552`,
  `1bc4ae80b7669fdeda3db0a742053598e12e823840f9f62f59b12d62d06b1d53`,
  `c584c817b779b7c6bba8d829440084c8cd7d84d5b1719fb0de80e3da7d86251f`
  and `0b1535c7ec239c07ab66ec4c97ae671d7c171a76dd0479e95d0f3091507900be`.
- **Terminal decision:** R4 is rejected. Exact orbit averaging successfully
  removes R3's view-swap instability but does not make the relational residual
  a useful selector; no part of R3/R4 relation fitting is adopted as a proven
  performance improvement. Consolidated terminal audit is
  `artifacts/kaggle/rad_dino_mask_bag_orbit_relation_r4_v1/kernel_version1_terminal_result_audit.json`,
  SHA-256
  `a18b6382a39161b0d3ffa0edb56a7930aca720226bbae5a68dcad903e5092bcc`.
  Consumer authorization remains false, no consumer was trained, and BTXRD
  test remains locked. The goal is not reached; the next claim must be a
  non-relational, non-duplicate successor grounded in terminal evidence and
  must not compete with the collaborator's active proposal-supply G0/G1 arm.

### EXP-20260801-codex-s3-same-family-graph-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`; `HOÀN THÀNH`
  with terminal gate failure.
- **Registered:** `2026-07-31T21:17:00Z` (`2026-08-01` ICT), after terminal R4
  evidence commit `28c53c6862e8c8025329fdbf6bf4924581a7a2c0` was pushed centrally.
  Exact claim commit is `fad5ba66e72365ea12674aec41256f2e3bf582e9`.
  No S3 source, protocol, kernel binding, launch, prediction or metric exists
  at registration.
- **Coordination/non-duplicate audit:** full central coordination state was
  reread and both remote branches were fetched. Collaborator head remains
  `797f191fcae8a5bb0b4ccb920d1adf635af1eff8`; active
  `EXP-20260731-codex-rich-gallery-g0g1-v1` changes proposal supply by merging a
  classifier-448 gallery and fitting G1 on it. S3 keeps the immutable old
  candidate gallery and accepted Geometry-v3 scorer, changes no proposal or
  descriptor, and applies only a fixed same-family graph operator to frozen
  candidate logits. It therefore does not duplicate or compete with G0/G1.
  R1, R2, S1, R3 and R4 are terminal negative evidence and none of their
  mechanisms is adopted as a demonstrated improvement.
- **Inherited terminal evidence and hypothesis:** R4 terminal audit SHA-256
  `a18b6382a39161b0d3ffa0edb56a7930aca720226bbae5a68dcad903e5092bcc`
  shows exact view-swap stability but changes only `25/184` tumor winners; among
  those changes, four improve Dice, 14 worsen and seven tie, with four
  baseline-hit→miss versus one miss→hit transition. Thus a learned global
  critical relation is rejected. Yet the same frozen R4 ranking diagnostic
  reduces top-1 to top-3 regret from `0.17591328→0.12929660` overall,
  `0.23251191→0.17323013` medium and `0.26125464→0.15571134` large. S3 tests the
  narrower predeclared campaign question: can strictly local consensus among
  geometrically overlapping proposals from the same immutable family move a
  better supported candidate upward without propagating evidence across
  unrelated families?
- **Exact inherited inputs:** selector-cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  manifest `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`,
  split `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  accepted Geometry-v3 checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`
  and prediction freeze
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`.
  Geometry-v3 Dice remains `0.2454823867797678 / 0.11708057891440651 /
  0.37713551529480416 / 0.3894126471276201`; immutable oracle remains
  `0.4090755342486002 / 0.2227494852063559 / 0.5941470844279589 /
  0.6418253674184405`. Candidate identity/order, masks, family/source IDs,
  descriptor bytes, map construction and evaluator remain unchanged.
- **Frozen mechanism before implementation:** reproduce the accepted
  original/aligned-flip averaged candidate logits `s` exactly from the shared
  cache and checkpoint. Build a symmetric graph only between distinct valid
  candidates with the same immutable `family_id` and either IoU at least
  `0.25` or containment at least `0.50`; cross-family edges are forbidden and
  isolated candidates receive only a self-loop. With symmetric normalized
  adjacency `P`, apply exactly ten fidelity iterations
  `z_(t+1) = 0.5 * P * z_t + 0.5 * s`, initialized by `z_0=s`, then select the
  maximum final logit. These constants are the already prepared S3 primitive
  defaults/test coefficient, not a validation-fitted sweep. There is no fit,
  learnable parameter, alternate alpha/threshold/iteration, family balancing,
  critical relation, affinity residual, pseudo-instance label, subgroup input,
  GT router or rescue.
- **GT-blind gates before evaluation:** independently verify bit-exact
  `alpha=0` reproduction of all accepted base candidate logits/indices/maps;
  graph symmetry, zero cross-family edges and exact preservation of every
  isolated candidate logit; at least one non-self edge so the arm is
  nontrivial; `371/371` exact view-swap records; all `371` physical candidate
  score vectors/maps frozen; and absolute candidate-count/bag-probability
  Spearman no greater than `0.5013777759365411`. Any failure rejects S3 before
  validation GT with no parameter change, rerun, sweep or rescue.
- **Post-freeze decision/gates:** only after a separate physical-output audit
  passes may the unchanged evaluator read validation GT. Mechanism pass still
  requires medium regret reduction, regret reduction in at least two tumor
  subgroups, no overall Dice regression and no increase in absolute
  candidate-count/miss association. Adoption and consumer authorization
  additionally require Dice at least `0.34024039 / 0.17895493 / 0.51244178 /
  0.49370336`, paired overall CI95 lower bound above zero, no subgroup mean
  decrease, no miss increase and image AUROC at least `0.75`.
- **Compute/safety:** static local synthetic tests only; one future private
  Kaggle T4x2 kernel will rescore disjoint validation shards on both T4s and
  freeze the graph outputs. The inherited scorer was trained with image-level
  labels only and S3 performs no new fit. Validation GT remains closed until
  prediction freeze and independent gate pass; consumer and BTXRD test remain
  locked.
- **Technical basis:** this is the already predeclared S3 campaign row. SmMIL
  motivates fidelity-preserving normalized-Laplacian smoothing of localization
  values while warning that local-dependency assumptions must match the graph:
  https://proceedings.neurips.cc/paper_files/paper/2024/file/8db9279f593652ee9bb2223b4a2c43fa-Paper-Conference.pdf .
  TransMIL motivates correlated rather than independent MIL instances:
  https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html .
  S3 transfers only the local-consensus principle; neither paper nor the
  prepared primitive is claimed beneficial on BTXRD before a terminal audited
  result.
- **Static implementation progress:** scientific source commit
  `c11b891c7063e202e2c93a87b4d971542eb305e1` is pushed centrally. The shared
  relational primitive now constructs the exact same-family graph directly
  from the hash-frozen pairwise IoU/containment cache; canonical-LF SHA-256 is
  `0b5eb685e474ad4a0480abe97c5f30db5825bb101326b13c7fe1a9d616ec48a2`.
  New fit-free scorer `project/models/mask_bag_same_family_graph.py` and T4x2
  runner `project/run_mask_bag_same_family_graph_s3_arm.py` have SHA-256
  `5d2eef32bcfc971457e6cd6a2eac2b68c4dc7bd4d4eace654dd0121e486f7b0b` /
  `5f882090b1d8ce8192d207047afe16f0aa572394b94fee55b4d6b56e410ec6a1`.
  The runner verifies the full `2,981/371` cache, accepted baseline freeze and
  manifest; rescores disjoint validation shards on both T4s; freezes per-image
  hashes of all alpha-zero base score vectors; requires exact accepted
  selected index/logit/bag-logit/bag-probability/map reproduction; then writes
  the distinct S3 score vectors/maps and graph diagnostics before its physical
  prediction freeze. No optimizer or fitting surface exists.
- Synthetic graph/scorer and static runner tests have SHA-256
  `9d0270289b21343a3b08cee47e2a6f94439604383227ca6301805ec53d422904` /
  `8d834ec85be2b760a0edba3687f7458cd7f6d5ecc236d87c06f9b2b05643394e`.
  `py_compile` passes; the focused S3/relational/R4 compatibility suite passes
  `14/14`, and full repository regression passes `428/428` in 17.96 seconds
  under the documented local Python-3.9 `zip(strict=...)` diagnostic shim.
  This is static preparation only: no S3 protocol/binding/kernel launch,
  scientific input, prediction, validation GT/test or consumer access has
  occurred.
- **Static source-isolation correction:** the first full regression after the
  draft cached-pairwise helper reported `2 failed, 429 passed`: the S3 helper
  had been added to shared `mask_bag_relational_selector.py`, so the
  historical R3 and R4 binder tests correctly rejected current HEAD because
  their protocols hash-lock the earlier shared file. This happened entirely
  in static tests, before any scientific input, prediction or launch. Commit
  `293b013cd036d8346fea3852ec3025772172f32d` moves the helper into the isolated
  S3 module and restores the shared source byte/hash exactly to
  `3cae8adbdf0a11384a891926570069ceefb0cda2de6dc9b5476ae19a1f17f790`.
  The S3 graph arithmetic/constants/hypothesis are unchanged. Corrected S3
  module/test SHA-256 values are
  `0c65a2bfa2ad9a190f860bf51bfdc28c8af2b03b7a4e8ec9f2327e3cfed67287` /
  `ec0137a518d0def488a4f8a61fca368675cd797aa1d5d8dec20387c2a308993c`.
- **S3 protocol and independent-auditor freeze:** exact protocol
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1.json`
  SHA-256
  `7d7636176fc05d407b51a913170ad780e2d43d328d9437b2d9d2656e191471ca`
  freezes source commit `293b013`, the no-fit graph, one T4x2 execution, all
  alpha-zero/baseline/graph/count gates and unchanged post-freeze adoption
  rules. Independent auditor
  `project/audit_mask_bag_same_family_graph_s3_output.py` SHA-256
  `787b68b144ec0759f75f186a5566e716f02db02f72689f674294b51b089d9913`
  pins the generic physical-output helper `3cc5feee...`, requires a separate
  frozen launch binding, verifies source/cache/baseline/T4x2 plus all `371`
  maps/scores, and independently recomputes every alpha-zero, accepted-row,
  graph, view-swap and count gate without an evaluator or GT. Auditor test
  SHA-256 is
  `7f174dab7b4fa9515217158d9b2950aab1c3a798d49751425ddc4f679884a26f`.
- Focused S3 plus historical-binder closure passes `18/18`; full repository
  regression now passes `431/431` in 16.05 seconds under the documented
  Python-3.9 `zip(strict=...)` shim. Static-readiness artifact is
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_static_readiness.json`,
  SHA-256
  `23bd41b07d2e99ccf33c4175eac291f5e1d5bc49e68d1a7613a010916bea4fc2`.
  Wrapper/binder remain unfinished; no binding, launch, scientific prediction,
  validation GT/test access or consumer training exists.
- **Fail-closed wrapper/binder readiness:** unbound wrapper
  `project/kaggle_wrappers/run_mask_bag_same_family_graph_s3_v1.py` canonical-LF
  SHA-256
  `c4e41f81d4f874e137cefc9a80875ced07a92555289be7c7fbba9fe58f8a7b2d`
  fails closed at kernel version `0`, `LAUNCH_BINDING_READY=False` and
  `CHECKOUT_COMMIT=UNBOUND`. It verifies the exact Git checkout/protocol/source,
  T4x2, split reconstruction, cache and accepted baseline transport; runs the
  focused plus full repository suites before inference; then physically checks
  all 371 maps, score payloads, pregraph identity rows and graph diagnostics.
- One-time binder
  `project/bind_mask_bag_same_family_graph_s3_wrapper.py` SHA-256
  `29e0fc57ca78564865136a65e89c92e7db0380b7b41227aa1708df8bb77d3636`
  pins wrapper template SHA-256
  `c4e41f81d4f874e137cefc9a80875ced07a92555289be7c7fbba9fe58f8a7b2d`,
  changes exactly the three external launch fields, proves exact inverse
  reconstruction, refuses existing output and re-verifies every protocol-bound
  source hash at the future execution checkout. Wrapper/binder tests have
  SHA-256
  `304daa77c983e7f52f98e069ad6a8f64726fe13ee759eb6aa43682ae834efdfb` /
  `7dcba537ceca01380271a7cf3a8542c9d37b37eb12e633cc471aafb6f8072a50`.
- Focused S3/auditor/historical-binder closure passes `17/17`; full repository
  regression passes `435/435` in 17.01 seconds under the documented Python-3.9
  `zip(strict=...)` compatibility shim. `py_compile`, JSON parse and
  `git diff --check` pass. Wrapper/binder readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_wrapper_binder_readiness.json`
  is frozen before source commit/binding with SHA-256
  `92ffcbbac0a8604c989ef51f83db9bdbc1418dff2027843572a8e62d5e324e38`.
  No binding, kernel push, scientific input/prediction, validation GT/test
  access or consumer training occurred in this static step.
- **Version-1 launch binding frozen, not launched:** after wrapper/binder source
  and readiness were committed/pushed at
  `56c658af1062c95d8ee4c4eae62ea20557bf49b9`, the one-time binder bound that
  exact execution checkout and positive kernel version `1`. Scientific source
  `293b013...` and claim `fad5ba6...` are verified ancestors. Bound wrapper
  SHA-256 is
  `0b5ca61ba9170bba6301efad378a03acaf412f002eed3436c04ed9ac7c8d5a0b`;
  launch-binding SHA-256 is
  `0d280d4b56843366c18a1c370543c85a985415a4112019ee9f8b6e966f85ff47`;
  metadata SHA-256 is
  `3a4943f7f91ffc8641074c49f59eafa60701e0ec7d1b12b6a611a30801e26850`.
- Bound-wrapper `py_compile`, metadata/binding JSON parse, exact three-field
  inverse reconstruction, protocol/all-runtime-source verification and
  claim/source ancestry checks pass. Final prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_kernel_v1_final_prelaunch_audit.json`
  is frozen before Kaggle push with SHA-256
  `ee627c066f1036e199805f7eb33c6d80bc1cfe24cb8d085a85b0d6dda7edaf31`.
  At this point no kernel push, scientific input/prediction, validation GT/test
  access or consumer training has occurred.
- **S3 version 1 launched:** after launch binding/final audit were committed and
  pushed at `3870c37e504c593ad3bef91b95ee93cb9414bbab`, central HEAD/origin were
  synchronized with a clean tracked worktree and the ignored package was
  reverified as bound wrapper
  `0b5ca61ba9170bba6301efad378a03acaf412f002eed3436c04ed9ac7c8d5a0b`
  plus metadata
  `3a4943f7f91ffc8641074c49f59eafa60701e0ec7d1b12b6a611a30801e26850`.
  Kaggle accepted exactly one private kernel push as version `1` of
  `itsthang333/btxrd-rad-dino-mask-bag-same-family-graph-s3-v1` at about
  `2026-07-31T21:49:19Z`. The wrapper must physically observe two T4 devices or
  fail closed. Prediction freeze, validation GT, consumer and BTXRD test remain
  locked pending a terminal independent output audit.
- **Single bounded post-launch status:** at `2026-07-31T21:49:52Z`, exactly one
  status query returned `RUNNING`. This is not a scientific result. No repeat
  poll or additional monitor was created; validation GT, consumer and BTXRD
  test remain locked.
- **S3-specific post-freeze decision gap closed while RUNNING:** a static audit
  found that the generic evaluator enforces regret reduction in any two tumor
  subgroups, whereas the frozen S3 protocol additionally requires medium regret
  reduction explicitly. No terminal output or metric had been seen and no new
  Kaggle status query was made in this step. New decision source
  `project/decide_mask_bag_same_family_graph_s3.py` SHA-256
  `871b370a00a73c4266874b8c5abd04fc39f0319e1ae89656ff182ff779bb1ec9`
  reads only the hash-locked independent GT-blind audit and generic evaluation
  artifacts; it never opens GT, and requires both the complete generic mechanism
  gate and strictly positive medium regret improvement. Consumer authorization
  still requires the complete operational/adoption gate.
- Decision regression test SHA-256 is
  `d96353f59c1109f898b699a80ed4fd1144b7ba4de6ef4c48bb4514bfb081b6d4`;
  it proves a hypothetical generic `OPERATIONAL_PASS` with zero medium gain is
  rejected, a full pass with positive medium gain is accepted, and any mutated
  evaluation output is rejected. Focused closure passes `12/12`; full repository
  passes `438/438` in 16.88 seconds under the documented Python-3.9
  `zip(strict=...)` shim. Readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_postfreeze_decision_readiness.json`
  is frozen before terminal retrieval/GT with SHA-256
  `630a81372fc9cccace09c92715217b7ac85bf22d9234edff27bee792ba1aa08c`.
  Prediction bytes/protocol/producer remain unchanged; GT, consumer and BTXRD
  test remain locked.

### S4 feasibility boundary while S3 is running (2026-08-01)

- A static inventory found locally retained group-OOF artifacts from terminal
  rejected `EXP-20260731-codex-r1-normal-prototype-v1`, including all 15 folds
  for `K={8,16,32}`. Independent error audit SHA-256
  `833c4814eee0891df1cd55a01fa008e0708ad3485dd30b730657f52508317719`
  already proves those fits are group-excluded, but also proves absolute OOF
  candidate-count/probability Spearman
  `0.60365911/0.60299779/0.60282430`, all far above the frozen `0.50137778`
  ceiling. R1 is terminal negative evidence, not an accepted teacher.
- Therefore the predeclared S4 proposal-cluster arm must not inherit or relabel
  any R1 OOF logits/checkpoints as a useful teacher. If S3 is terminal rejected
  and S4 is claimed, its runner must create a distinct group-excluded teacher
  under the S4 protocol, verify every held-out/training-group exclusion before
  cluster construction, and serialize teacher/seed/member provenance. Using the
  accepted all-train Geometry-v3 checkpoint as an OOF teacher is also forbidden
  because it has seen the held-out train groups.
- This audit changes no S3/S4 scientific source or protocol, registers no S4
  claim and launches no competing experiment. No Kaggle status query,
  scientific input execution, validation prediction/GT, consumer training or
  BTXRD test access occurred. S3 remains the sole active central selector claim.

### EXP-20260801-codex-s3-same-family-graph-v1 — version-1 pregraph float-identity error

- After waiting the full bounded monitor interval, exactly one status query at
  about `2026-07-31T22:10:16Z` returned terminal `ERROR`. Direct Kaggle log was
  retrieved separately with SHA-256
  `d56049a8fe1f6615191413539f361a8fde2fdaa9afa2d9e2fc24ea36e0814ac4`
  (`19,408` bytes). An initial official CLI output download timed out locally
  after 124 seconds and left 147 files/29,752,885 bytes; its child downloader
  was confirmed absent afterward. The incomplete baseline/runtime inventory is
  not interpreted as a scientific result.
- Log verifies checkout `56c658af...`, scientific source `293b013...`, protocol
  `7d763617...`, focused tests `26/26`, full Kaggle tests `434 passed, 1 skipped`
  and runner entry after the fail-closed T4x2 guard. The runner stopped in
  `_write_pregraph_identity_audit` with
  `RuntimeError: S3 alpha-zero or accepted-baseline identity failed`, before
  graph candidate-score/prediction manifests, prediction freeze, wrapper audit
  or evaluator.
- Partial pregraph audit CSV SHA-256
  `08ebf0ec7c53f9355c85c736b963d1b9a4a530b90511f89d08dc673d09f036ce`
  contains all 371 validation identities. Crucially, alpha-zero candidate-vector
  identity, accepted selected index and accepted map SHA each pass `371/371`.
  Exact scalar equality passes only `104/371` selected logits, `158/371` bag
  logits and `247/371` bag probabilities, leaving `81/371` entire rows exact.
  Thus version 1 failed on strict `==` comparison of floating-point display
  scalars across two GPU executions, while the complete candidate vectors,
  discrete selector result and physical map identity were exact.
- Tracked error audit
  `artifacts/kaggle/rad_dino_mask_bag_same_family_graph_s3_v1/kernel_version1_error_audit.json`
  is frozen before any correction with SHA-256
  `8962636754fc06f7c0097de6a8866b91483e611279ec9c7d22a8bcc3f3aa0f78`.
  Version 1 remains `LỖI` with no scientific Dice. A permitted
  successor must be implementation-only: use a predeclared float32/ULP-grounded
  scalar identity tolerance, serialize observed/accepted/delta/tolerance per
  row, retain exact candidate-vector/index/map checks and change no graph,
  candidate, protocol gate or scientific hyperparameter. Validation GT,
  consumer and BTXRD test remain locked.
- **Version-2 numeric-identity correction frozen, not bound/launched:** after the
  version-1 error audit was committed/pushed at `6570a39`, implementation-only
  addendum
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_posterror_numeric_identity_addendum.json`
  was frozen with SHA-256
  `41e88ae7011c3f994f7d47a6a9216730ba9448ccb6f9fc8599d277a0679f0d51`.
  It transfers the already audited S1 numerical contract without observing a
  validation metric: scalar identity tolerance is exactly
  `max(2e-6, 4*abs(spacing(float32(accepted))))`; no post-hoc widening is
  allowed.
- Corrected runner/auditor SHA-256 values are
  `30e3048a706127e0cea52892d0e682d97e0c81dc8aee2bd05c4254674fabf6db` /
  `2ee49080f80b2f651f80f11be79bf663bc062cc288f6828d3bd8c427f94a39a6`.
  Every row now serializes observed/reference/delta/float32-spacing/tolerance/
  exact/pass for selected logit, bag logit and probability. Exact 371-vector,
  selected-index and physical-map checks remain mandatory. The independent
  auditor recomputes every scalar and regression tests accept four ULP but
  reject eight ULP.
- Runner/auditor test SHA-256 values are
  `66a8f81a0dbb7c150a03d95f27693c4d96544c1858f2621f49591655a98d8440` /
  `9c21b982834047986a407b5253d565272edb45fea7d8b0d67dd6d9f232fe8654`.
  Historical v1 binder regression is fixed to the immutable v1 execution
  checkout `56c658af...`, rather than pretending corrected HEAD is v1 source.
  Focused correction/binder tests pass `11/11`; full repository passes
  `441/441` in 16.61 seconds under the documented Python-3.9
  `zip(strict=...)` shim. Readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_kernel_v2_numeric_correction_readiness.json`
  is frozen before binding with SHA-256
  `9a8c512049b30fbc1af844a3021da55e29a628be82445bf90510925a14af321f`.
  Graph/candidates/hyperparameters/protocol gates are unchanged; no rerun,
  prediction, validation GT, consumer or test access occurred.
- **Version-2 wrapper/binder readiness:** fail-closed wrapper template SHA-256
  is `774cae093545632ab71de07800fd3642669cd04c713bbcbb4168370e6e30f42d`;
  schema-2 binder SHA-256 is
  `655b3bc9b2e1835a7b7650eac96f99d27eb524b2b53aaaf16d20d5d9e6848872`.
  The binder verifies the immutable parent protocol and addendum, and allows
  exactly two implementation overrides: corrected runner/test hashes
  `30e3048a...` / `66a8f81a...`; every other scientific source remains pinned to
  the original protocol. Updated independent auditor SHA-256 is
  `79591dff304dbe90621b226ff301af3094b8979827108eeb4d514875c22ceed4`
  and requires the same schema-2 binding/addendum/override map.
- Wrapper now removes only exact `/kaggle/working/s3_source` and `s3_runtime`
  paths in `finally`, after path-parent verification, so either terminal success
  or error keeps compact S3 evidence without exporting redundant cloned source
  and 371 baseline maps. Wrapper/binder test hashes are
  `f67af2ae036485a59bbd98e608e936f57a79ec964740335d44eaec6639675025` /
  `a8d8c5dec7e3ecd96863fe688fc9643c02f849d01ee7bc0ea458dfd63f6e139c`.
  Focused closure passes `13/13`; full repository passes `441/441` in 16.65
  seconds. Wrapper/binder readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_kernel_v2_wrapper_binder_readiness.json`
  is frozen before source commit/binding with SHA-256
  `2669f50858648f0616158225739ad38ab459a249a045a0adceacd72e1d4b6352`.
  No version-2 binding/launch, prediction, validation GT,
  consumer or BTXRD test access occurred.
- **Version-2 binding/final prelaunch frozen, not launched:** after corrected
  wrapper/binder source was committed/pushed at
  `271e3f7b476671e6719c26c6263bc9624f9cd8d3`, the one-time binder created a
  schema-2 version-2 binding to that exact checkout. Bound wrapper SHA-256 is
  `1b07e044b068170a24e06c63095ad3867094a40ff2f0f0d4640bed44ef7cfa33`;
  launch-binding SHA-256 is
  `f4592f31b6173f2b8d5fe1abd8c4166b57ca3bd0e0fa6d14c97b56bd3c98d9dc`;
  unchanged metadata SHA-256 is
  `3a4943f7f91ffc8641074c49f59eafa60701e0ec7d1b12b6a611a30801e26850`.
- Bound-wrapper `py_compile`, metadata/binding JSON parse and independent
  auditor schema-2 static verification pass; all 17 effective runtime hashes
  reproduce from parent protocol plus the exact two addendum overrides. Final
  prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_same_family_graph_s3_v1_kernel_v2_final_prelaunch_audit.json`
  is frozen before Kaggle push with SHA-256
  `4c39b310680cbbe79c07a4dfcaf72cf1558a570878d499605bca23811da74502`.
  No version-2 launch/prediction, validation GT, consumer or test access has
  occurred.
- **S3 version 2 launched:** after binding/final audit were committed/pushed at
  `8a949358f2774a60615d3c3614ff8d7c67ffe684`, central HEAD/origin were clean and
  synchronized. Because local policy refused removal of a generated
  `__pycache__` directory, no destructive workaround was used; instead a fresh
  ignored launch directory was created containing exactly two files. Their
  hashes reverified as bound wrapper
  `1b07e044b068170a24e06c63095ad3867094a40ff2f0f0d4640bed44ef7cfa33`
  and metadata
  `3a4943f7f91ffc8641074c49f59eafa60701e0ec7d1b12b6a611a30801e26850`.
  Kaggle accepted exactly one corrected private push as kernel version `2` at
  about `2026-07-31T22:38:27Z`. Prediction freeze, validation GT, consumer and
  BTXRD test remain locked pending terminal schema-2 independent audit.
- **Single bounded version-2 post-launch status:** one query immediately after
  the launch record was pushed returned `RUNNING`. This is not a scientific
  result. No repeat poll or additional monitor was created; all GT/consumer/test
  locks remain in force.
- **Bounded status-check access error:** after the predeclared approximately
  20-minute quiet interval, exactly one status command at about
  `2026-07-31T22:57Z` failed locally with Kaggle permission
  `kernels.get was denied` for the unchanged private kernel slug. This response
  does not establish `RUNNING`, `COMPLETE`, or kernel `ERROR`, so no scientific
  or terminal inference is made and no output/GT was opened. Per the one-check
  heartbeat constraint, it was not retried in this interval and no additional
  poll/monitor was created. The S3 claim remains `ĐANG LÀM`; prediction-freeze,
  validation-GT, consumer, and BTXRD-test locks remain in force pending a later
  bounded authenticated terminal check.
- **S3 version-2 terminal and GT-blind freeze gate:** a later single bounded
  authenticated API check again failed before returning kernel state (HTTP 401
  `kernels.get`), and the predeclared compact downloader likewise failed before
  listing any file (HTTP 403 `ListKernelSessionOutput`). Neither response was
  interpreted as kernel failure. The already authenticated Kaggle notebook page
  at
  `https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-same-family-graph-s3-v1`
  then provided the authoritative read-only terminal evidence: `Version 2 of
  2`, successful run, `4m 30s`, `GPU T4 x2`, and 500-file UI listing. The
  user-authorized web `Download output` action produced `results.zip`, SHA-256
  `1a4cf54c14c5cc93c33d9e64a24958cadf742456ffa79e667c3a0bb505e2b02e`,
  1,920,599 bytes, with 749 ZIP members and full CRC pass. It was copied to a
  new ignored experiment directory and expanded to 749 files; BTXRD test and
  validation GT were not opened.
- The independent schema-2 auditor then verified 371/371 candidate-score
  payloads and 371/371 frozen validation maps, exact protocol
  `7d763617...`, launch binding `f4592f31...`, bound wrapper `1b07e044...`,
  source commit `293b013...`, split `85511ee1...`, baseline/cache freezes,
  image-level-only training, no consumer and no test evaluation. It also
  verified zero cross-family edges, 18,441 non-self edges, 1,012 isolated
  candidates, exact graph symmetry/view-swap/isolated-logit invariants and all
  371 tolerance-bounded pregraph identity records. Prediction-freeze SHA-256 is
  `2e737e356b2f8785c61665099235873415d6f9a165d999578e7a53206a37a91f`.
  Independent audit
  `artifacts/kaggle/rad_dino_mask_bag_same_family_graph_s3_v1/kernel_version2_complete_gt_blind_output_audit.json`
  has SHA-256
  `d8fb1f74ff40bcc419566e7eaa2dd74ebe36cfd3f82ab4468028bc1fd7e6f627`
  and terminal status
  `PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS`. Validation GT
  remains unread at this commit boundary; only after this evidence is pushed
  may the frozen S3 pair be evaluated by the predeclared evaluator.
- **First post-freeze evaluator input-copy error before GT:** after audit commit
  `fff0be1` was pushed, the unchanged evaluator was invoked with the frozen S3
  arm/cache/split/seed inputs but pointed at the baseline copy retained by the
  incomplete S3 version-1 error download. That copy has the correct baseline
  freeze but lacks `predictions/prediction_manifest.csv`; evaluator verification
  therefore raised `FileNotFoundError` in `_verify_baseline` before line 349's
  GT boundary, before importing the segmentation dataset, and before creating
  any evaluation output. No metric or scientific result exists from this
  attempt. The authoritative paired-geometry-v5 baseline root is independently
  present with the same freeze SHA-256 `ec346276...` and expected manifest
  SHA-256 `a810e1fc...`. A bounded retry may change only the physical baseline
  root to that complete, hash-identical copy; arm predictions, cache, split,
  evaluator, protocol, baseline per-image SHA, 10,000 replicates and seed
  `20261012` remain unchanged. This error/correction is recorded and pushed
  before retry; BTXRD test and consumer remain locked.
- **Post-freeze S3 result:** the bounded corrected evaluator changed only the
  physical baseline root to the complete paired-geometry-v5 copy; all expected
  hashes, frozen arm/cache/split, evaluator SHA-256 `ccc3a493...`, baseline
  per-image SHA-256 `a26143d0...`, 10,000 complete-group bootstrap replicates
  and seed `20261012` remained fixed. It opened validation GT only after the
  committed independent freeze audit. S3 Dice overall/small/medium/large is
  `0.2424364550818621 / 0.10337489174368669 / 0.3819824010665648 /
  0.4104630574646339`, versus accepted geometry-v3
  `0.2454823867797678 / 0.11708057891440651 / 0.37713551529480416 /
  0.3894126471276201`. Deltas are
  `-0.0030459316979057254 / -0.013705687170719816 /
  +0.004846885771760589 / +0.02105041033701373`; paired CI95 values are
  `[-0.01946796,+0.01180367] / [-0.03852680,+0.00383695] /
  [-0.01696367,+0.03144844] / [-0.03105989,+0.07408482]`.
- Selected-to-oracle regret changes (baseline minus S3; positive is better) are
  `-0.003045931697905735 / -0.013705687170719821 /
  +0.004846885771760595 / +0.021050410337013742` for
  overall/small/medium/large. Thus the explicit medium-regret condition passes
  and large also improves, but overall and small regress. Complete misses fall
  from `53/33/18/2` to `49/30/17/2`; image AUROC is
  `0.8103638688677052`. Output hashes are evaluation audit `220ff04d...`, gate
  `97cdb393...`, paired comparison `7fd17c1a...`, per-image `27ca3c3e...`, and
  summary `7765d198...`.
- **Terminal decision:** the generic evaluator and S3-specific hash-locked
  decision both return `FAIL`: despite real medium/large gains, S3 misses every
  operational Dice goal and violates overall/small no-regression. Decision
  `artifacts/kaggle/rad_dino_mask_bag_same_family_graph_s3_v1/kernel_version2_postfreeze_decision.json`
  has SHA-256
  `077fc8d563c5a2f42d3f5fd9c6e409bf5147f94a6a4dcb2d51fcef2f88fc079c`.
  Consolidated terminal audit
  `artifacts/kaggle/rad_dino_mask_bag_same_family_graph_s3_v1/kernel_version2_terminal_result_audit.json`
  has SHA-256
  `6a03b7294c1cd168bf1967c30fa4f16fc8a72e983bd128b854a23c274a2349eb`.
  The graph is not adopted as a validated selector improvement and no consumer
  is trained. BTXRD test remains locked. The goal is not reached; any successor
  must be separately claimed after a fresh collaboration-log sync and may use
  the positive medium/large observation only as bounded terminal evidence, not
  as permission to relabel S3 as successful or to tune on validation GT.

### EXP-20260801-codex-s4-oof-proposal-cluster-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`;
  `HOÀN THÀNH — TERMINAL REJECTED AT GT-BLIND COUNT GATE`.
- **Registered:** `2026-07-31T23:17:34Z` (`2026-08-01` ICT), after terminal S3
  commit `1488a99daa2e2e40a2fee734eaf4899b2e2468f1` was pushed centrally.
  Exact registration commit is
  `fa8b0d64ea5571dedf3df2b1a98d221f4179a3f8`; it was pushed centrally before
  any scientific launch.
- **Hypothesis:** a distinct group-excluded OOF image-label-only teacher can
  identify recurring candidate modes without S3's per-image over-smoothing;
  using its stable original/flip seeds to form mask-IoU/containment proposal
  clusters, then training normalized cluster bags with a predeclared
  smooth-to-sharp continuation, may reduce selected-to-oracle regret while
  leaving out-of-cluster candidates unlabeled rather than falsely negative.
- **Scope/non-duplicate boundary:** this is the already finite-predeclared S4
  row and inherits terminal evidence from
  `EXP-20260801-codex-s3-same-family-graph-v1` and
  `EXP-20260731-codex-r1-normal-prototype-v1`. It is not S3/R4 graph smoothing,
  does not reuse rejected R1 OOF logits/checkpoints, and may not use the
  all-train geometry-v3 checkpoint as an OOF teacher. It keeps the immutable
  old same-gallery candidate supply and therefore does not launch, reproduce or
  evaluate the collaborator-owned active
  `EXP-20260731-codex-rich-gallery-g0g1-v1` proposal-supply/G1 scope.
- **Frozen inherited inputs:** selector cache kernel
  `itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1`, freeze SHA-256
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  independent audit SHA-256
  `7d9f693dd5d1d9206b01cc2c8a0ed4aed497f9f17d9dedf670a97771b0f78334`,
  split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  accepted geometry-v3 baseline freeze `ec346276...` and per-image evaluator
  table SHA-256
  `a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f`.
  S3 terminal audit SHA-256 is
  `6a03b7294c1cd168bf1967c30fa4f16fc8a72e983bd128b854a23c274a2349eb`.
- **Required new provenance before launch:** a fresh teacher must be trained in
  fully group-excluded folds using only clean-train image labels; every teacher
  checkpoint, fold membership, held-out/training-group exclusion, original/flip
  logit, seed and cluster member must be serialized and independently audited
  before optimizer construction. Teacher selection, cluster count,
  IoU/containment thresholds and continuation schedule must be frozen using
  train-only/image-label-only criteria and synthetic invariants before launch;
  any material scope change requires another pushed log update.
- **Compute:** static/synthetic preparation locally; all real BTXRD teacher
  fitting, cluster-arm fitting and validation prediction only on one Kaggle
  T4x2 or P100 job after fail-closed protocol/source/binding gates pass.
- **Output/gate:** physically freeze all 371 validation predictions plus
  all-candidate scores and cluster/teacher provenance before validation GT.
  After independent GT-blind audit pass and a pushed audit commit, evaluate with
  the common hash-locked selector evaluator. Mechanism pass still requires
  regret reduction in at least two tumor subgroups, no overall Dice regression
  and no increase in absolute candidate-count/miss association. Full adoption
  additionally requires Dice overall/small/medium/large at least
  `0.34024039/0.17895493/0.51244178/0.49370336`, positive overall paired CI95
  lower bound, no tumor-subgroup mean decrease, no miss increase and image AUROC
  at least `0.75`.
- **Safety locks:** validation GT remains closed until physical prediction
  freeze plus independent output audit; no consumer may be trained before the
  full operational gate; BTXRD test remains locked; no validation Dice/subgroup
  signal may select a teacher, cluster, threshold, schedule or checkpoint.
- **Static S4 teacher/cluster training preparation:** the new dataset-agnostic
  module `project/models/mask_bag_proposal_cluster_training.py`, canonical-LF
  SHA-256
  `374e1f03505c4b25bfb3f705b931e075f2794e07bfae0124b9e9a6de3bb40296`,
  implements a genuinely new five-fold group-excluded teacher rather than
  importing R1 outputs or the accepted all-train scorer as teacher weights.
  Each fold teacher starts deterministically from scratch, uses the accepted
  1,156-D scorer architecture and only image-bag labels, and exposes aligned
  original/flip candidate logits. Conservative cluster-seed evidence is the
  per-candidate minimum of the two view logits, so there is no fitted view
  threshold.
- Fixed draft controls now represented in code are five folds, 16 final-epoch
  teacher/student epochs, batch 16, AdamW `3e-4`/`1e-4`, teacher instance and
  view-consistency weights `0.25/0.10`, two-epoch instance warm-up, at most four
  disjoint clusters, mask IoU at least `0.50` or containment at least `0.75`,
  and normalized within/between cluster continuation from temperature `1.0` to
  `0.20`. The cluster student is a zero-initialized residual over the frozen
  accepted baseline; residuals are identically zero outside the OOF-seeded
  cluster union, preserving those candidates exactly rather than labeling them
  negative. The runner/protocol must freeze these exact values or update and
  push the claim before launch; validation metrics cannot choose alternatives.
- New tests
  `tests/test_mask_bag_proposal_cluster_training.py`, SHA-256
  `efb619481f310c1fd359859fcf7bdc652ed9be74e3a771ad5b5b252f7d206e67`,
  verify the GT/subgroup-free API, conservative view scoring, fixed
  IoU/containment membership, disjoint clustering, exact zero initialization,
  exact outside-cluster fallback, OOF coverage and rejection of any held-out
  group overlap. Together with the earlier cluster primitive/tests
  `3fe2cab3...` / `b523bb0a...`, `py_compile`, focused tests and the full
  repository regression pass (`446/446` in 18.80 seconds) under the documented
  local Python-3.9 `zip(strict=...)` shim. This step uses only synthetic/static
  inputs: no real BTXRD cache was opened by the new code, no teacher or selector
  was fit, no validation prediction/GT/test was accessed and no consumer was
  trained.
- **Runner-integrated S4 preparation:** the training module above is extended,
  not rewritten historically; its new canonical-LF SHA-256 is
  `8f13560192055de0e9fbb68cf9a7dadddd9e9b11d19b0f244984cfc6fbc10ac9`
  and test SHA-256 is
  `8b843b5879a483b6041f0f5525bac7b9c2603913d3c85027490c8c9a185c5cb5`.
  It now fits each teacher fold from scratch, scores every held-out train image
  exactly once, audits full group exclusion/coverage and computes the OOF
  absolute candidate-count/probability Spearman before any student fit. It also
  supplies an exact all-record zero-initialization audit and final scoring that
  proves outside-cluster residuals remain zero in both aligned views.
- The fail-closed real-run source
  `project/run_mask_bag_proposal_cluster_s4_arm.py`, canonical-LF SHA-256
  `be11aed8fc114c20ccca6eba82d4e2bb8fe634827aa87df2179255fa84f9d7bc`,
  fixes the complete five-fold assignment to the already audited cohort, splits
  folds `3/2` across T4x2, and requires the OOF teacher count/probability value
  to stay at or below `0.5013777759365411`. Before constructing the residual
  optimizer it serializes all five teacher checkpoints/training-group lists,
  every OOF candidate score, the exclusion audit, all train cluster
  seeds/members, a distinct all-train teacher used only on unseen validation
  groups, all validation teacher scores and all validation cluster
  seeds/members. The accepted all-train geometry-v3 scorer is loaded only as a
  frozen zero-residual student baseline, never as cluster teacher.
- After the student fit, the runner scores validation in fixed `186/185` shards
  on both T4s, requires the final GT-blind count/probability ceiling and exact
  outside-cluster fallback, then writes all-candidate scores, 371 maps and the
  prediction freeze. Runner tests SHA-256
  `35b7f36814564026d3caced2a1b6460f90b082398cbeecb7622a3247e5d6729c`
  prove serialization/gate ordering, frozen controls, T4x2/safety contracts and
  failure before prediction serialization when the count gate is exceeded.
  A final precommit fail-closed audit found two implementation risks before any
  scientific execution: the two OOF GPU threads could call global
  `manual_seed_all` concurrently and perturb each other's dropout stream, and
  the distinct full-train teacher recorded `validation_groups_seen=False`
  without proving the train/validation group sets were disjoint. The corrected
  runner now constructs all five initial teacher states serially, then each
  worker seeds only its own current CUDA device; it also writes and hash-binds
  an exact train/validation group-exclusion audit before the full-teacher
  optimizer is constructed. These are implementation/provenance corrections;
  the teacher architecture, loss, folds, cluster thresholds, continuation,
  residual and every scientific gate are unchanged.
  `py_compile`, focused closure `13/13` and full repository regression
  `450/450` in 24.57 seconds pass under the documented Python-3.9 compatibility
  shim. This remains static/synthetic work only: no real cache execution,
  teacher/student fit, validation prediction/GT/test or consumer training has
  occurred. Protocol/auditor/wrapper binding remain required before launch.
- **Collaboration sync before the S4 protocol freeze:** central
  `origin/research-wsss-improvement` was exact at local HEAD before each source
  commit. `origin/codex/research-sync-20260731` advanced from `797f191...` to
  `32b73fc4e7c98711bf028ca618443811dd2bfd45` through three code-only commits:
  support for the collaborator's three-source rich-gallery cap and a staged G1
  all-candidate ranking/extent diagnostic plus post-freeze mechanism
  ablations. Its `RESEARCH_LOG.md` did not change and still marks G0/G1
  `ĐANG LÀM`. The new design note mentions G1 Dice `0.206026` versus
  same-gallery `0.245482` and a higher rich-gallery oracle `0.528298`, but
  supplies no new log-bound terminal audit/artifact hash; achieved Dice is also
  worse, not better. Therefore none of that code/mechanism is adopted as a
  demonstrated improvement, and S4 remains confined to the old immutable
  same-gallery cache without duplicating proposal supply.
- **Physical residual-evidence closure:** pre-protocol independent review found
  that the runner's Boolean outside-cluster fallback flags were not sufficient
  for a separate auditor to prove the claim from downloaded bytes. Evidence-only
  source commit `95c4a3378eaf8463c57d57a0dd4e4cac6c69021f` is pushed centrally.
  It does not change any teacher/student logit, cluster, loss, threshold,
  schedule or gate. For every validation image it now freezes candidate order,
  cluster-member flags and original/flip base, residual and combined logits;
  exact base-plus-residual, averaged-logit and zero-outside-cluster identities
  must pass before the ordinary prediction manifest can be written.
- Final protocol-bound runner/training/test SHA-256 values are
  `52e870bdc0c669131072a3f7a28680004f412df2873dc1f78bce917591520ef7` /
  `75fc8efc128d110a0030ceb062cd589f839f1483331b2461eee6f828088d1a88` /
  `9fc0b7b0f08f117a45943936b192228541cd39f56e05b7aa899e4fb0c7c4f41e`.
  These supersede only the unlaunched preparation hashes above.
- **S4 protocol and independent-auditor freeze:** exact protocol
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1.json`
  has SHA-256
  `040227de1347c45bc1823bd5aef5d9614b8005619ecc35d9dceb45bb7eba71e8`.
  It binds source commit `95c4a33`, all fixed five-fold teacher, conservative
  seed, mask-IoU/containment cluster, continuation, zero-residual, count-gate,
  T4x2, prediction-freeze and post-freeze operational rules, plus all 24 runtime
  source/test hashes.
- Independent GT-blind auditor
  `project/audit_mask_bag_proposal_cluster_s4_output.py` SHA-256
  `1058f5f8c69e9906576d7803eb70665f199d690348b6b80427f60ba86f8e0014`
  requires a future exact launch binding and terminal cache freeze. It
  independently reconstructs the 2,981-row group-stratified fold assignment,
  every OOF exclusion/score payload, all train/validation clusters from the
  hash-locked cache IoU/containment matrices, the final cluster SmoothMax,
  physical residual fallback, candidate winners and 371 selected-mask maps.
  Auditor test SHA-256 is
  `faf88a9cce59f0efac13bc263cb10ae30b72ba4f6d0d727a4257dd4e98c452a5`.
- Metadata-only verification against the accepted cache manifest reproduces
  exactly 2,981 train rows, the frozen five-fold summary and row-payload
  SHA-256 `407be430a6aa4408e1baf961ce0cd8eb55e6fe06b34640ceecb3bdfe0cb67ec5`;
  no cache record payload, radiograph or annotation was opened in that check.
  Focused closure passes `17/17`; the full repository passes `454/454` in
  18.47 seconds under the documented local Python-3.9 `zip(strict=...)` shim.
  Static-readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_static_readiness.json`
  has SHA-256
  `d0fe3a1a94fe36d7e401cbb5364da6b77ec3f913614750c7dc63f3efb36521cc`.
  Wrapper/binder/final prelaunch audit remain absent; no real S4 cache record
  execution, teacher/student fit, Kaggle launch, validation prediction/GT/test
  access or consumer training has occurred.
- **Fail-closed wrapper/binder readiness:** unbound Kaggle wrapper
  `project/kaggle_wrappers/run_mask_bag_proposal_cluster_s4_v1.py` has
  canonical-LF SHA-256
  `af550929841b305f4612d9ba6581e214d9964ec3454c86fbd7c770f6dcefe4db`.
  It remains deliberately impossible to launch at kernel version `0`,
  `LAUNCH_BINDING_READY=False` and `CHECKOUT_COMMIT=UNBOUND`. After a future
  one-time binding, it must verify the exact Git checkout, protocol and all 24
  source hashes, reconstruct the CRLF split, verify accepted cache/baseline
  transport, require two physical T4 devices, pass focused plus full tests,
  execute S4 once and physically verify all 2,981 OOF scores, 2,981 train
  clusters, 371 validation-teacher scores/clusters/residual payloads,
  all-candidate scores and maps before writing its wrapper audit.
- One-time binder
  `project/bind_mask_bag_proposal_cluster_s4_wrapper.py` SHA-256
  `190c889d6ed3b05c270c880161af38620ea9a2281f68630febea7cf780a5b2b5`
  permits exactly three external launch-field replacements, proves inverse
  reconstruction to template
  `af550929841b305f4612d9ba6581e214d9964ec3454c86fbd7c770f6dcefe4db`,
  verifies source/protocol bytes at the future execution checkout and refuses
  any existing bound output. Wrapper/binder test SHA-256 values are
  `eab917457c02ad7df0b9db9e62dcec155f31796196ac7e67160491688fa27e4a` /
  `a8c6b609b6c9dcd1c70d81d8caab0aefbcb95686368b7995eabae039f664c7e2`.
- Focused closure passes `22/22` and the full repository passes `459/459` in
  18.82 seconds under the documented Python-3.9 shim. Wrapper/binder readiness
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_wrapper_binder_readiness.json`
  has SHA-256
  `1c2dafed3fe2d04c18b9dae3d26c547bd6eb3a72fe83bd7d0dd203faafa36961`.
  No binding, kernel package/push, scientific cache record, fit, prediction,
  validation GT/test access or consumer training occurred.
- **Prelaunch provenance error and binding abort:** the first local version-1
  binding was created but not committed or launched. Final ancestry checking
  then proved that the full claim commit copied into the protocol,
  `fa8b0d64ea5571dedf3df2b1a98d221f4179a3f8`, does not resolve; the actual
  commit identified by short hash `fa8b0d6` is
  `fa8b0d6de4d1ad9d1372fd2c61ee8b78feec91a4`. The failure occurred after
  local binding but before binding/final-audit commit, kernel package commit or
  Kaggle push, so it has no scientific output or metric.
- The aborted protocol/binding/bound-wrapper/final-draft hashes are
  `040227de1347c45bc1823bd5aef5d9614b8005619ecc35d9dceb45bb7eba71e8` /
  `c47e9a63767ec8039fa5fb074812464e5b7686529b7b955e2b8c9d2f47c1a4d0` /
  `8fe514c354661472eb99acebcb834d6e14df13e160f0ad4164f0c1625b6a30a5` /
  `362f84f871141085072ddae4e105f97610a336e29adaddc70829411a7af1b9ac`.
  Error audit
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_prelaunch_invalid_claim_hash_audit.json`
  has SHA-256
  `b9a87e0c51b371832abd34ca4a75f47819887283ce06e77bb91de51d9afcc72d`.
- **Implementation-only correction in progress:** the protocol claim field is
  corrected to the actual existing commit and the resulting protocol SHA-256
  is `fb39234a03890d7201531066e3ca7a11f2379eaa120bd503fe4b92e6de30a2a6`.
  Only this provenance byte chain changes; source commit, teacher/student
  algorithm, inputs, folds, thresholds, schedule and gates are unchanged.
  Corrected independent auditor/wrapper-template/binder SHA-256 values are
  `a93bc7a237d0fc62275f7bd6230c3e06f3d1def94ca4a560b0b58ac305b689ff` /
  `fdaa86c7e71ea50664d268b8e98458114a1d72871c42940ff1532b7315782ec7` /
  `186bb7472f1339df1f52a95c711e8729c55acd688c502c56d8908557665ebbce`.
  The earlier static/wrapper readiness artifacts remain immutable evidence for
  the aborted protocol chain and are superseded for launch. A corrected binding
  may be created only after these bytes are committed/pushed and the complete
  tests pass at that exact checkout. No scientific input, prediction, GT/test
  or consumer access occurred.
- **Corrected-chain readiness pass:** correction commit
  `da6c82ce646ece7a60d99c2ce41f31895ea5b07d` is pushed centrally. Both actual
  claim `fa8b0d6de4d1ad9d1372fd2c61ee8b78feec91a4` and scientific source
  `95c4a3378eaf8463c57d57a0dd4e4cac6c69021f` are verified ancestors.
  Corrected auditor/wrapper/binder focused closure passes `9/9` and the full
  repository passes `459/459` in 18.36 seconds. Correction-readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_claim_hash_correction_readiness.json`
  has SHA-256
  `17546d299db02f34ea6b6c8a85a609cb022be334dd059ced60b8e01e08b21d1f`.
  This authorizes only creation of a fresh local version-1 binding at a pushed
  checkout; it is not a launch or scientific result.
- **Corrected version-1 binding and final prelaunch freeze:** the fresh
  one-time binder targets exact pushed checkout
  `1ee46bb184ce63612baaaf922522265438d39d8d`. Corrected bound-wrapper,
  launch-binding and unchanged metadata SHA-256 values are
  `7abbb0b569f0df610fd1d4e84027242a92fe4ad6f70b5c9a4f346fba6ff1011c` /
  `2c4c60673218dcc63f1f58b11ac9932dff73656aca2046fe4f01bbd7358a766a` /
  `39ce8e274bcf293f540599bcf24ad7dc13060118d79ba67fa9d587ceb3c92666`.
  The independent binding schema, exact three-field inverse reconstruction,
  all 24 protocol source hashes, both claim/scientific-source ancestry checks,
  metadata parse and bound-wrapper `py_compile` pass.
- Final prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_kernel_v1_final_prelaunch_audit.json`
  is frozen before any Kaggle push with SHA-256
  `d395d0f800f22b9e6f2cd64666c1aacb549c35f6db99b54a19af363c98b858c1`.
  The invalid binding/draft were moved intact to an ignored error-evidence
  directory rather than deleted. At this boundary there is still no kernel
  push, scientific input execution, fit, prediction, validation GT/test access
  or consumer training.
- **S4 version 1 launched:** after corrected binding/final audit were committed
  and pushed at `975345919faf245a1c71cc11efa4316260a19eef`, central HEAD/origin
  were synchronized with a clean tracked worktree. The ignored launch package
  contained exactly the bound wrapper
  `7abbb0b569f0df610fd1d4e84027242a92fe4ad6f70b5c9a4f346fba6ff1011c`
  and metadata
  `39ce8e274bcf293f540599bcf24ad7dc13060118d79ba67fa9d587ceb3c92666`.
  Kaggle accepted exactly one private push as version `1` of
  `itsthang333/btxrd-rad-dino-mask-bag-proposal-cluster-s4-v1` at about
  `2026-08-01T05:56:48Z`. The wrapper must observe two physical T4 devices and
  every GT-blind gate or fail closed. Prediction freeze, validation GT,
  consumer and BTXRD test remain locked pending terminal output plus independent
  physical audit.
- **Single bounded post-launch status:** at `2026-08-01T05:57:19Z`, exactly one
  status query returned `RUNNING`. This is not a scientific result. No repeat
  poll or additional monitor was created; prediction freeze, validation GT,
  consumer and BTXRD test remain locked.
- **Post-freeze decision layer frozen while version 1 remains running:** no new
  Kaggle status query was made in this static-preparation step. The fail-closed
  decision source `project/decide_mask_bag_proposal_cluster_s4.py` and its test
  have SHA-256
  `389d77b0eaede89e4f194b8af94947e45c29ec3a940e9317511de9f97db4d8e1` /
  `303adb45bc56cef73950e4c94f5f82e8e8a9ce5693f01f27cd44b407007ace77`.
  They require the exact independent GT-blind output audit and evaluator audit,
  all physical output hashes, 10,000 bootstrap replicates and fixed seed family
  `20261013`; they independently recompute the generic mechanism, operational,
  oracle and final-safety gates. A mechanism-only pass explicitly keeps the
  consumer locked; authorization is possible only when every gate passes.
- `py_compile`, focused decision/evaluator/auditor tests `14/14` and the full
  repository regression `463/463` in 19.52 seconds pass under the documented
  local Python-3.9 compatibility shim. Static readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_proposal_cluster_s4_v1_postfreeze_decision_readiness.json`
  has SHA-256
  `e72e53b38e60ec9c10d8c85eca144f341a190a6db57910c52010f9595c995dc7`.
  This step opened no scientific input or validation GT, created no prediction,
  trained no consumer and evaluated no BTXRD test. S4 remains `ĐANG LÀM` until
  one later bounded terminal check and the predeclared audit/evaluation chain.
- **S4 version-1 terminal GT-blind count-gate rejection:** after a fresh complete
  coordination read/fetch, exactly one bounded status query at about
  `2026-08-01T06:08Z` returned Kaggle terminal `ERROR`. No repeat status poll or
  monitor was made. The UTF-8 direct log has SHA-256
  `952251f769a5e54003e6dda95f068d86670085d2fb3f526afa0cbbd415d9c52a`
  and proves exact checkout `1ee46bb...`, source/protocol binding, T4x2 wrapper
  progression, focused tests `37/37`, full checkout tests `458 passed, 1
  skipped`, and one runner invocation ending at line 603 with
  `RuntimeError: S4 OOF teacher count/probability gate failed`.
- The compact official inventory contains `3,004` files; all `18` JSON/CSV
  boundary files plus the direct log were downloaded without another status
  query. Independent local recomputation against the frozen selector cache and
  exact five-fold assignment verifies `2,981/2,981` OOF rows, fold sizes
  `596/596/596/596/597`, 16 training epochs per fold and zero held-out/training
  group overlap. Absolute candidate-count/bag-probability Spearman is exactly
  `0.5715729179616584`, versus frozen baseline `0.48137777593654113` and maximum
  allowed `0.5013777759365411`; it fails by `0.0701951420251173`.
- Tracked error audit
  `artifacts/kaggle/rad_dino_mask_bag_proposal_cluster_s4_v1/kernel_version1_oof_count_gate_error_audit.json`
  has SHA-256
  `248663692942b2fdf996f3fd8c76674b8b4287529230a0a0d161018e577b2366`.
  The direct bulk CLI download timed out locally after 124 seconds with 300
  small partial files, and the first parallel compact-inventory attempt hit a
  Windows resolved-parent race; a sequential resume completed all compact
  evidence. These are retrieval-only boundaries and did not change the result.
- **Error boundary/conclusion:** all five distinct group-excluded OOF teachers
  completed using image-level labels, but the predeclared scientific shortcut
  gate failed before train-cluster attachment, cluster-student optimizer, full
  teacher fitting, validation scoring, prediction freeze or evaluator. There is
  no Dice to report and no tolerance/threshold rescue or version-1 rerun is
  permitted. S4 is terminal rejected rather than an implementation error;
  validation GT, consumer and BTXRD test remain locked. A successor must be a
  new non-duplicate registered hypothesis and may use this result only as
  negative evidence against unconstrained image-label teacher probabilities.
- **Full S4 error-output preservation after compact conclusion:** responding to
  the observation that the output itself is small, the generic official
  inventory downloader was corrected for a Windows concurrent parent-resolve
  race by validating and creating all parent directories sequentially before
  workers start. Downloader/test SHA-256 values are
  `82d62afd036c5a6f63b773848ebf0df103cf5e0efe0b9e0500997effb4a4a96a` /
  `80c506edd0985fe72efd803dd270217b80f9467ef333217692abe1ecead54916`;
  focused tests pass `4/4` and full regression passes `471/471` in 19.73 seconds.
- Without another Kaggle status query, the corrected downloader resumed and a
  final inventory-only pass confirmed exact `3,004/3,004` official files,
  `12,506,968` bytes, zero partial files and `0` remaining downloads. Independent
  physical verification opens all five OOF checkpoints/histories/manifests and
  all `2,981` candidate-score NPZ payloads (`12,469,175` verified OOF bytes),
  checks every locked hash/schema/dtype, reconstructs conservative logits as the
  exact original/flip minimum, proves zero group overlap and recomputes the same
  failed Spearman `0.5715729179616584`.
- Full physical retrieval audit
  `artifacts/kaggle/rad_dino_mask_bag_proposal_cluster_s4_v1/kernel_version1_full_oof_physical_retrieval_audit.json`
  has SHA-256
  `de4c1f07cffc094a548df7e2b04b79de07d0bc4ced68b0b96aeb65051be38d5e`.
  This strengthens preservation/auditability only: the S4 terminal rejection,
  no-Dice boundary and all GT/consumer/test locks are unchanged.

### EXP-20260801-codex-t1-count-controlled-self-paced-v1

- **Owner/status:** Codex main task on `research-wsss-improvement`; `ĐANG LÀM`.
- **Registered:** `2026-08-01T06:24Z` (`2026-08-01` ICT). Exact first
  registration commit `d8cd1608b6886f51cc5240ddcc81c1dc2a5e1e2a` was pushed
  to the central branch before any T1 scientific execution.
- **Inheritance and coordination:** this is the predeclared T1 row after terminal
  `EXP-20260801-codex-s4-oof-proposal-cluster-v1`, whose GT-blind error audit
  SHA-256 is
  `248663692942b2fdf996f3fd8c76674b8b4287529230a0a0d161018e577b2366`.
  It also inherits terminal negative evidence from R1/R2/S1/R3/R4/S3 and the
  accepted Geometry-v3/cache, but inherits none of those rejected mechanisms as
  a demonstrated improvement. The collaborator-owned
  `EXP-20260731-codex-rich-gallery-g0g1-v1` remains a separate candidate-supply
  scope with no log-bound terminal result better than same-gallery Geometry-v3;
  T1 changes no proposal, candidate order or gallery and does not duplicate it.
- **Hypothesis:** S4's fresh group-excluded teacher retained strong OOF image
  classification (`AUROC 0.92145064`) and modest original/flip top-1 stability
  (`0.62328078`) but failed because bag probability remained strongly tied to
  candidate count (`|rho|=0.57157292`). A fresh producer trained with an explicit
  differentiable count-independence penalty may retain image-label signal while
  removing this shortcut. Only if that producer passes a frozen OOF gate may
  conservative cross-fitted targets drive the already predeclared ItS2CLR-style
  self-paced confirmation residual.
- **Non-duplicate/scientific delta:** this is not an S4 rerun or threshold rescue.
  It does not load S4 checkpoints, candidate logits, seeds or clusters and uses
  no mask-IoU/containment cluster objective. Five fresh group-excluded producers
  use the accepted `1,156`-D candidate representation and image labels, with the
  new project-specific nuisance loss equal to squared Pearson correlation
  between bag probability and standardized `log1p(candidate_count)` in
  deterministic group- and label-balanced batches. The coefficient is fixed at
  `1.0`; it is not selected by validation GT or a post-hoc lambda search.
- **Frozen producer controls:** exact five-fold assignment row-payload SHA-256
  `407be430a6aa4408e1baf961ce0cd8eb55e6fe06b34640ceecb3bdfe0cb67ec5`;
  each producer starts from a separately serialized deterministic initialization;
  16 final-epoch fits, batch 16, AdamW learning rate `3e-4`, weight decay `1e-4`,
  image BCE plus original/flip consistency weight `0.10` and count-independence
  weight `1.0`. The producer may read only clean-train descriptors, image labels,
  group IDs and candidate counts. Every checkpoint, training/held-out group,
  history, original/flip candidate logit and scalar diagnostic must be frozen.
- **Producer-only operational gate before any target/consumer:** independently
  verify exact `2,981/2,981` OOF coverage, zero held-out/training group overlap,
  absolute candidate-count/bag-probability Spearman no greater than
  `0.5013777759365411`, OOF image AUROC at least `0.75`, and original/flip top-1
  agreement at least `0.60`. Failure terminates T1 before target construction;
  the count ceiling is not widened and no consumer is fit. This gate uses only
  image labels and GT-blind metadata.
- **Frozen consumer controls, conditional on the complete producer gate:** for
  every image-label-positive training bag, only an OOF producer's original/flip
  agreeing top candidate is eligible; conservative original/flip margin orders
  eligible positives through fixed `20%→40%→60%` self-paced stages. All candidates
  in image-label-negative bags are admissible negatives but receive equal total
  mass per image, family and candidate. The consumer is a zero-initialized
  residual over the frozen accepted scorer, trained for 12 epochs with AdamW
  `1e-4`, supervised-contrastive weight `0.25`, image BCE weight `1.0` and the
  same count-independence weight `1.0`; contrastive temperature is fixed at
  `0.10`. Each target row binds the producer fold and proves that producer did
  not train on its group.
- **Inputs/provenance:** selector-cache kernel
  `itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1`, freeze SHA-256
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`,
  accepted Geometry-v3 checkpoint SHA-256
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`
  and baseline per-image evaluator SHA-256
  `a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f`.
- **Sources and transfer limit:** ItS2CLR motivates reliability-ordered
  pseudo-instance contrastive refinement
  (https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html),
  while self-guided radiograph MIL motivates soft/cross-fitted rather than
  same-model hard targets
  (https://openaccess.thecvf.com/content/ACCV2020/html/Seibold_Self-Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Thoracic_DiseaseClassification_and_ACCV_2020_paper.html).
  The count-correlation penalty is a transparent BTXRD-specific response to the
  terminal S4 diagnostic, not a claimed reproduction of either paper.
- **Compute/output/gates:** static and synthetic work locally; every real OOF
  producer fit, conditional consumer fit and validation prediction only in one
  registered Kaggle T4x2/P100 run. If the producer passes, all targets and 371
  validation predictions/all-candidate scores must freeze physically and pass
  an independent GT-blind audit before the common evaluator opens validation
  GT. Mechanism pass remains regret reduction in at least two tumor subgroups,
  no overall Dice regression and no increase in count/miss association. Full
  adoption additionally requires Dice overall/small/medium/large at least
  `0.34024039/0.17895493/0.51244178/0.49370336`, positive overall paired CI95
  lower bound, no tumor-subgroup decrease, no miss increase and image AUROC at
  least `0.75`.
- **Safety:** training remains image-label-only. No validation segmentation GT
  may select a producer, penalty, target, pace, epoch, residual or checkpoint.
  No target/consumer is created before the full producer gate; no downstream
  consumer is authorized before the full operational gate. BTXRD test remains
  locked. This registration launches nothing and opens no new scientific input.
- **Static producer/target primitive readiness:** dataset-agnostic source
  `project/models/mask_bag_count_controlled_self_paced.py` and tests have
  SHA-256
  `e01f6ab1b2790689420bb244c284157a5709cccc7b10959d4c55d27cba7ac6b1` /
  `1662d405b6cd6dbf8e72b7cc14a22224c6d1e6293fd28d48928b2d72958f8290`.
  The source implements fresh serializable producer initialization,
  deterministic label/group-balanced batches, image BCE plus aligned-view and
  differentiable probability/count-correlation losses, complete OOF
  exclusion/coverage and count/AUROC/view gates, a hard consumer lock, nested
  `20/40/60%` positive stages, equal negative image/family/candidate mass and a
  zero-initialized accepted-baseline residual. A final singleton batch keeps its
  image-label loss and receives a differentiable zero count penalty rather than
  dropping a training image.
- `py_compile`, focused synthetic tests `7/7` and full repository regression
  `470/470` in 16.22 seconds pass under the documented local Python-3.9
  compatibility shim. Primitive-readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_primitive_readiness.json`
  has SHA-256
  `79319ceb1c677544ca291a41a77725ec643a9402e5024016e4c830014df7f5ae`.
  Runner, protocol, independent auditor and wrapper remain absent, so this does
  not authorize Kaggle launch. No real cache record, producer fit, target,
  consumer, validation prediction/GT or BTXRD test was opened or created.
- **Conditional consumer-core readiness:** the same source is extended, without
  changing producer controls, to SHA-256
  `3b59930f7d048d20ac3218790e5257a46f8b05a39b6ad0c33adb188bed5c9ad4`;
  updated test SHA-256 is
  `0d1b35ea82c45a800ba449c51475c965e65498261e89b07a7ac16e787c44a48a`.
  The consumer refuses non-GT-blind/failed producer targets, uses exactly four
  epochs per `20/40/60%` stage, keeps the accepted scorer frozen, and learns a
  zero-initialized scalar residual plus projection embedding. Each trusted
  candidate contributes separate original/flip contrastive views, guaranteeing
  a cross-view positive even when a batch contains only one target of a class;
  final scoring retains base and residual evidence for every candidate.
- `py_compile`, focused tests `8/8` and full regression `472/472` in 17.39
  seconds pass. Consumer-core readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_consumer_core_readiness.json`
  has SHA-256
  `86989a26646a547ede66513b636de963e2bd0fd2d82193152422c9c0c67bd95d`.
  This is still synthetic/static: runner/protocol/auditor/wrapper are absent and
  no real cache, target, fit, prediction, validation GT/test or consumer was
  opened or created.
- **Fail-closed T1 runner readiness:** final model/runner/test SHA-256 values are
  `b8acab98594c55526e6aa13b7dca02ba751f790e83126594ae62762214d19d28` /
  `9fb05756bc44c6b9c12063d18706df76eab1c99137425540006c264292b2a5c3` /
  `0d1b35ea82c45a800ba449c51475c965e65498261e89b07a7ac16e787c44a48a` /
  `28cbc21694a26389e4fc69cba2d9d9ceeaa2a2744c1a4d49ae5ee3532f9cad75`.
  The runner validates the exact cache/split/accepted baseline and T4x2 runtime,
  builds five initial states serially, fits folds `0/2/4` and `1/3` on separate
  GPUs, writes every producer checkpoint/history/OOF score before the producer
  gate, and raises before target construction if any count/AUROC/view/exclusion
  check fails.
- Only after producer pass does it freeze negative plus all three positive target
  manifests, audit accepted-base identity on train and validation, fit the
  confirmation residual, and score validation in fixed `186/185` shards. It
  freezes every base/residual/combined candidate logit, independently checks
  exact arithmetic and final count Spearman, then writes all-candidate scores,
  371 maps and prediction freeze. No validation evaluator or GT import exists.
  The confirmation residual is part of this selector arm; the separate downstream
  consumer remains untrained and unauthorized.
- `py_compile`, focused model/runner tests `12/12` and full regression `476/476`
  in 17.81 seconds pass. Runner-readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_runner_readiness.json`
  has SHA-256
  `0840750f331accdd02a9e1905a4bccca1eeb51107d305f7bdf79523869371213`.
  Protocol, independent auditor, wrapper/binder and launch remain absent; no
  real cache execution, fit, target, prediction, validation GT/test or downstream
  consumer access occurred.

### Đồng bộ terminal G1 và readiness G2 từ nhánh cộng tác (2026-08-01)

- Toàn bộ `AGENTS.md`, log trung tâm và toàn bộ `RESEARCH_LOG.md` 8,107 dòng tại
  `origin/codex/research-sync-20260731` commit
  `f22700be9f74aaa560e0da95326d318f29c9a59b` đã được đọc sau khi fetch cả hai
  nhánh. Nhánh cộng tác giữ `EXP-20260731-codex-rich-gallery-g0g1-v1` ở trạng
  thái `ĐANG LÀM`; phần mới được ghi dưới mã
  `EXP-20260801-codex-rich-gallery-g1-root-cause-and-g2-v1`. Central không chạy
  cạnh tranh hoặc copy implementation G2.
- **G1 terminal evidence từ workstream cộng tác:** rich-gallery Geometry-v3
  đạt Dice validation overall/small/medium/large
  `0.20602633/0.10958465/0.30070545/0.33094975`, thấp hơn accepted same-gallery
  Geometry-v3 `0.24548239/0.11708058/0.37713552/0.38941265` ở cả bốn cohort,
  trong khi immutable gallery oracle là
  `0.52790203/0.33110060/0.73025092/0.74624721`. G1 có 29 complete misses
  (`27/2/0`). Checkpoint, independent Stage-A freeze và Stage-B summary/per-image
  SHA-256 được branch cộng tác ghi lần lượt là
  `634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c`,
  `c4e80a0c9bd8a1d4e5ef6204d23123d2d4f7b4deabb4c4b38aa4578b8b899e1c`,
  `3be34a5765c14c68a7be68773e37ac66b03abf4951a165d83c8229048621da98`
  và `8c8c12f9129351e842587c80f91fdd368de326f06029adc83d2ceb4e73b92d21`.
  Đây là terminal result xấu hơn baseline, nên rich-gallery G1 scorer không được
  kế thừa như một cải tiến hiệu năng.
- **Transferable negative evidence:** merged gallery tăng proposal khoảng
  `56 -> 150` mỗi validation image; external proposal xuất hiện ở `184/184`
  tumor và `0/187` normal, làm candidate count riêng đạt image AUROC `0.89449`.
  Với normalized LogSumExp temperature `0.2`, median effective candidate count
  chỉ `1.63`; detached hard winner có Dice dưới `0.1` ở `60.9%` tumor và `83.0%`
  small. Tổng regret `0.32187569` gồm wrong-source `0.08236474` (`25.6%`) và
  within-source candidate/extent `0.23951095` (`74.4%`). Bằng chứng âm này củng
  cố việc phải kiểm soát shortcut count/composition và extent, nhưng không chứng
  minh một kỹ thuật G2 cụ thể tốt hơn.
- **Exploratory post-freeze diagnostic:** loại external candidates tăng overall
  Dice lên `0.24062584`. Equal percentile-rank fusion của G1 với frozen upstream
  coverage/purity đạt
  `0.28872949/0.15772330/0.43522933/0.38687353`, tương ứng chênh so với accepted
  Geometry-v3 khoảng
  `+0.04324710/+0.04064272/+0.05809381/-0.00253912`; complete misses tăng từ G1
  `29` lên `49`. Choice/summary/audit SHA-256 do branch cộng tác báo là
  `403d290b2b9582ec52eb75831fa621918d329e8d0aa26125aa9c900faf942bd9`,
  `2ac9beb4a6d77fc339f7dc5b5bb06879bdb42c6bd25341d2f08329d4ead52b02`
  và `44974376a6652af4a5992ab5ebb4f7a30932c043ee98e9ffb6dbf0f644983a48`.
  Chính log nguồn gọi đây là exploratory vì rule được thiết kế sau Stage-B
  analysis; large còn giảm, bốn operational goal đều chưa đạt, và các output
  audit vật lý được tham chiếu không có trong Git tree của branch để central
  kiểm tra độc lập. Do đó fusion chưa được adopt như validated improvement và
  không được dùng để chọn/tune T1 trên validation GT.
- **G2 boundary:** commit `f22700b` chỉ thêm thiết kế, source-safe pooling,
  negative-only/source-balanced objectives, runner/auditor/evaluator và tests;
  focused regression `40/40`, nhưng log xác nhận chưa launch G2 GPU. Đây là
  readiness của claim cộng tác đang hoạt động, không phải terminal performance
  evidence. Central không chạy lại G2 hoặc dùng code/protocol đó trong T1.
- **Quyết định cho T1:** `EXP-20260801-codex-t1-count-controlled-self-paced-v1`
  vẫn là same-gallery producer/selector scope đã đăng ký, không đổi proposal
  supply, không dùng G1/G2 logits/checkpoints/fusion và không trùng G2. Nó tiếp
  tục đúng predeclared count-independence gate từ bằng chứng S4; phát hiện G1 chỉ
  được giữ như negative constraint rằng image-label fit phải fail closed trước
  target construction nếu còn khai thác count/composition shortcut. Không có
  prediction, validation GT, consumer hoặc BTXRD test nào được mở trong bước
  đồng bộ này.

### EXP-20260801-codex-t1-count-controlled-self-paced-v1 — protocol/auditor readiness

- Sau collaboration sync commit `2bfd46fb8e4897f16149bb67c968f3699a150ddc`,
  scientific model/runner tại commit `c7f0937d515ded9bbd8928a2236cbe44b7a25f79`
  được audit tĩnh lại và không cần thay đổi khoa học. Protocol fail-closed
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1.json`
  có SHA-256
  `6a4379e896f3ea3862dce1edcdea20af09a90ec8f9cbbd6eb25bf8eca1306a7c`.
  Nó khóa exact source/claim/cache/split/baseline, năm OOF producers, loss
  image-BCE + view consistency `0.10` + squared count correlation `1.0`, gate
  `|rho|<=0.5013777759365411`, AUROC `>=0.75`, top-1 view agreement `>=0.60`,
  target stages `20/40/60%`, 12-epoch confirmation residual và mọi post-freeze
  evaluation/adoption gate. Collaborator G1/G2/fusion không phải input.
- Independent GT-blind physical auditor
  `project/audit_mask_bag_count_controlled_self_paced_t1_output.py` SHA-256
  `31ab993baa2f641355aa024212360f157a6ebab9f8839a02ffd95d024490ead6`
  không import T1 scientific model/runner hoặc evaluator/GT. Nó độc lập tái tạo
  exact cross-fit assignment/group exclusion; kiểm tra năm checkpoint/history/
  fold audit và 2,981 OOF score payloads; tái tính count Spearman, image AUROC,
  view agreement; dựng lại hierarchical negative weights cùng nested positive
  targets; kiểm tra target freeze trước confirmation checkpoint; và kiểm tra
  exact base+residual arithmetic, 371 candidate scores/residual payloads/maps và
  final validation count gate. Auditor test SHA-256 là
  `4f39f6660442488940497d38d800629c4baff60eecc801a219fe430a9f2af1e8`.
- Focused T1 model/runner/auditor closure pass `17/17` trong 3.78 giây; full
  repository regression pass `481/481` trong 17.74 giây bằng documented local
  Python-3.9 strict-zip diagnostic shim. Protocol JSON parse, `py_compile` và
  `git diff --check` đều pass. Readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_protocol_auditor_readiness.json`
  có SHA-256
  `24583c3221c1d22ad1ac5f56285b44e1189ee9ce3cdbe59cf3ea8d019a893126`.
- **Static test error boundary:** Python mặc định 3.13.5 không có NumPy/PyTorch
  nên ba test module dừng tại import trước test body. Lượt đầu trong đúng env
  Python 3.9.23 còn ba known `zip(strict=True)` compatibility failure và một
  assertion auditor quá rộng vì bắt cả evaluator path được hash-pin dù không
  import. Chỉ assertion được thu hẹp về import lines và established strict-zip
  shim được dùng lại; model, runner, protocol, metric, gate và dữ liệu không đổi.
- Đây vẫn là static readiness: wrapper/binder/final prelaunch chưa tồn tại nên
  chưa authorize Kaggle launch. Không real cache execution/fit/target/prediction,
  không validation GT, không downstream consumer và không BTXRD test access.

- **Wrapper/binder static readiness:** fail-closed Kaggle wrapper template
  `project/kaggle_wrappers/run_mask_bag_count_controlled_t1_v1.py` có
  canonical-LF SHA-256
  `6bbe6a5dda3bb35db88e27b2b1c117b7c8c199ad7a9b004450727ddec9f2c4a0`.
  Nó còn khóa `KERNEL_VERSION=0`, `LAUNCH_BINDING_READY=False` và checkout
  `UNBOUND`; yêu cầu exact T4x2 cùng real convolution trên từng GPU trước
  split/baseline/cache/tests/runner, tái tạo exact CRLF split, kiểm tra transport
  không GT/test, chạy focused/full checkout tests, và chỉ viết wrapper audit sau
  khi đủ năm OOF inventories, target freeze, residual evidence, 371 score
  payloads và 371 maps đã được kiểm tra vật lý.
- One-time binder `project/bind_mask_bag_count_controlled_t1_wrapper.py` SHA-256
  `e8c842ee888b1a780701fa7cdb3d6af6dbdbc0224ea705b7bb51925f65f5df98`
  chỉ thay ba trường version/readiness/checkout, bắt buộc inverse-reconstruct exact
  template, kiểm tra protocol và toàn bộ source hashes ở checkout, source
  ancestry và từ chối output đã tồn tại. Binder/static wrapper test SHA-256 là
  `0615069fa17c0a0aca004bf5e6a48fcf21fc916adf26d3b49740f2d7b261367b`.
  Focused closure pass `20/20` trong 4.68 giây; full regression pass `484/484`
  trong 18.37 giây bằng documented Python-3.9 shim.
- Một static assertion đầu tiên sai phạm vi vì tìm evaluator path trên toàn file,
  dù path đó chỉ nằm trong post-freeze hash-exclusion set và không được import.
  Assertion đã được giới hạn về import lines; wrapper/scientific source/protocol
  không đổi. Wrapper/binder readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_wrapper_binder_readiness.json`
  có SHA-256
  `f35e8fab59799f713dee550bda1890b35e6d42bf5079b15d967db1b7a5d36523`.
  Chưa có binding/package/Kaggle push; mọi GT/consumer/test lock giữ nguyên.

- **Version-1 binding/final prelaunch frozen, not launched:** sau khi wrapper/
  binder readiness được commit/push tại
  `97c4cf322a7ad329e62338cc8e7a2bbfa6e368d5`, one-time binder khóa version 1
  nhắm đúng checkout đó. Bound-wrapper canonical SHA-256 là
  `662217040edecfcad1d7187dc7cc40daa3f1007a1f9344821076df0623d3d85e`;
  launch-binding SHA-256 là
  `588901fb4640d0f0854867daf7858a826c1a1821e43d65dc1001b9f4b9ef9be1`;
  metadata SHA-256 là
  `67f4c4b1fb3bf30cb8da2ee2559bd437bb448de1185e9b7072ac0ef602c426d3`.
  Binding schema, exact three-field inverse reconstruction, protocol/source
  hashes, scientific-source/claim ancestry, metadata parse và bound-wrapper
  `py_compile` đều pass. Ignored launch package chứa đúng hai file.
- Full repository tại exact execution checkout tiếp tục pass `484/484` trong
  18.90 giây (wall 20.34 giây) bằng documented Python-3.9 shim. Final prelaunch
  audit
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_kernel_v1_final_prelaunch_audit.json`
  có SHA-256
  `6dea35315516842fca620b4783ea10b32d219c63d47e8cb1750960b324bc34bf`.
  Nó authorize duy nhất một private Kaggle push sau khi binding/audit được
  commit/push; chưa có kernel version hay scientific execution ở boundary này.
- Local binding verifier đầu tiên dùng nhầm Python 3.13.5 không có NumPy và dừng
  khi import independent auditor trước khi đọc binding. Cùng verifier trong env
  `btxrd-pseudomask` trả đúng bound-wrapper SHA; không source/protocol/binding
  nào thay đổi. Validation GT, downstream consumer và BTXRD test vẫn khóa.

- **T1 version 1 launched:** sau khi binding/final prelaunch audit được
  commit/push tại `c86da4c3bfeac58f369ce8cc5ab7ba8e9f7b07a3`, ignored
  package được kiểm tra lại đúng bound-wrapper `66221704...` và metadata
  `67f4c4b1...`. Kaggle chấp nhận đúng một private push làm version `1` của
  `itsthang333/btxrd-rad-dino-mask-bag-count-controlled-t1-v1` lúc khoảng
  `2026-08-01T07:24:23Z`. Không status poll ngay sau push và không monitor mới
  được tạo. Launch audit
  `artifacts/kaggle/rad_dino_mask_bag_count_controlled_self_paced_t1_v1/kernel_version1_launch_audit.json`
  có SHA-256
  `61428bbd9e3d4b430d0111a9c0256c71042a22210f7ccc7ebd761919ede46858`.
  Prediction freeze chưa được quan sát; validation GT, downstream consumer và
  BTXRD test tiếp tục khóa cho tới terminal output và independent audit.

- **Post-freeze decision layer frozen while version 1 runs:** không Kaggle
  status query trong bước static này. Decision source
  `project/decide_mask_bag_count_controlled_t1.py` và test có SHA-256
  `617f2da17aa4c4f77113e271879cbd0046647159b0061c14f03a560754e6c981` /
  `45901aed4f68154db17a6c33421b61e56a5e31bb48db682ab78b1737327413c1`.
  Tool bắt buộc exact independent GT-blind output-audit pass, exact evaluator
  audit/output hashes, 10,000 complete-group bootstrap replicates và seed
  `20261014`; nó tái kiểm tra generic mechanism/operational/oracle/safety gates.
  `MECHANISM_PASS` vẫn khóa consumer; chỉ `OPERATIONAL_PASS` với mọi check true
  mới authorize consumer.
- Focused decision/auditor/evaluator tests pass `15/15`; full repository pass
  `488/488` trong 20.13 giây bằng documented Python-3.9 shim. Readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_count_controlled_self_paced_t1_v1_postfreeze_decision_readiness.json`
  có SHA-256
  `511b2aec036785f3477b3c88b7db04f73a90a444ebc936f984a5ffc85d61cdc1`.
  Không prediction/output/GT được mở, không consumer được train và test vẫn khóa.

- **Kernel version 1 COMPLETE; independent audit dừng fail-closed trước GT:**
  một bounded status query trả `KernelWorkerStatus.COMPLETE`. Compact downloader
  lấy đủ official inventory `4,130/4,130` file và direct log vào ignored
  `tmp/kaggle/count_controlled_t1_v1_complete_20260801`; scientific root có
  `4,130` file / `94,506,200` byte, toàn thư mục có `4,131` file /
  `94,523,514` byte, không còn `.part`. Direct-log SHA-256 là
  `ca126401509ba33facfece74772660a6a993c260c12373d92b73ff761b877d14`;
  observed immutable `prediction_freeze.json` và candidate-score manifest có
  SHA-256 `4f317fc7af45804aff0b084398bebe35582a7ae2039a29982586e90358ea096d` /
  `8412868756b8cb64e6032ee215f2e9081f9e8c8564bf93bada49ddc443245927`.
  Các hash này mới là observed output, chưa được chấp nhận khoa học.
- Auditor source frozen `31ab993b...` dừng tại validation payload đầu tiên
  với `ValueError: T1 candidate-score schema mismatch: IMG000001.jpeg`. Audit
  GT-blind của toàn bộ `371/371` score payload cho thấy chúng có cùng exact
  schema `schema_version:int32 scalar=1`, `candidate_indices:int64`,
  `candidate_logits:float32`, đúng contract của shared writer
  `project/models/mask_bag_score_evidence.py` SHA-256 `02371b4d...`; auditor lại
  sai khi chỉ chấp nhận hai field cuối. Sample payload SHA-256 là
  `63a315e0611024d8263294ddefa2c30d708d19b7a434b66aa49b0583e567e6f5`.
  Default Python 3.13 không có NumPy dừng trước auditor; Python 3.9 đúng
  env cần strict-zip shim đã ghi trong readiness. Hai lỗi runtime này không
  phải kết quả khoa học và không làm thay đổi output.
- Error-boundary artifact
  `artifacts/kaggle/rad_dino_mask_bag_count_controlled_self_paced_t1_v1/kernel_version1_gt_blind_auditor_schema_error.json`
  SHA-256 `2a3fefed9cdbb410b4d47c802cc3e46ef242621cce7470d6de52e2f57df47224`
  đã được ghi trước mọi correction. Cho phép duy nhất auditor-only
  correction để chấp nhận và kiểm tra `schema_version==1`, kèm regression
  test; không đổi scientific source/protocol/prediction, không rerun kernel.
  Cho tới khi corrected independent audit pass, không được mở validation GT
  hay chạy evaluator/decision. Trạng thái experiment vẫn `ĐANG LÀM`;
  validation GT chưa đọc, downstream consumer chưa train, BTXRD test vẫn khóa.

- **Auditor-only schema correction ready, chưa mở GT:** sau khi error boundary
  đã được commit/push, auditor chỉ được sửa để yêu cầu exact ba field
  của shared evidence và kiểm tra scalar `schema_version:int32 == 1` trước
  khi đọc indices/logits. Corrected auditor/test SHA-256 là
  `cc66b53ce88bdc16b70ab39d0eaa874b94a10463fbd5bf058d43af02486c1f88` /
  `7438edb2f927c3bf5c5e2d0c9510e0f466524db317274112fe4424ca93f9b0c0`.
  Regression mới chấp nhận contract versioned và fail nếu thiếu schema; focused
  `7/7`, full repository `490/490` trong 21.86 giây pass bằng documented
  Python-3.9 strict-zip shim. Readiness artifact SHA-256 là
  `21071c267228224eb6f7369c8bfb0094bfea70c0b21faf62cf471464e81f0007`.
  Scientific source/protocol/prediction không đổi, kernel không rerun; chỉ
  corrected independent audit trên exact immutable output được phép tiếp theo.

- **Corrected independent GT-blind audit PASS:** exact immutable version-1
  output và accepted selector cache đã được audit vật lý độc lập; artifact
  `artifacts/kaggle/rad_dino_mask_bag_count_controlled_self_paced_t1_v1/kernel_version1_gt_blind_output_audit.json`
  có SHA-256
  `5f37c9d8b6b38a1c64293295ba51f64ae0fd8e55f0d7eac7c543f12edf974b29`
  và status `PREDICTION_FREEZE_PHYSICALLY_VERIFIED_GT_BLIND_GATE_PASS`.
  Auditor tái lập đủ `2,981` OOF record/5 fold, group exclusion zero;
  OOF `|rho(count,prob)|=0.3125463730294633`, image AUROC
  `0.8335525462912948`, original/flip top-1 agreement
  `0.6205971150620597`, nên cả bốn producer gate pass. Target freeze có
  `106,160` negative candidates, `964` eligible positive bags và nested stage
  counts `193/386/579`; confirmation residual chỉ train sau producer pass.
- Validation freeze độc lập xác minh đủ `371` candidate-score payload,
  `371` residual-evidence payload và `371` physical map; final
  `|rho(count,prob)|=0.4771398920620165 <= 0.5013777759365411` pass.
  Tổng `95,252,346` byte output+cache được hash/content verify. Training label
  chỉ image-level; validation GT chưa đọc, downstream consumer chưa train,
  BTXRD test chưa mở. Exact common post-freeze evaluator/decision đã
  predeclare nay được phép chạy; không được sửa prediction hay hyperparameter.

- **Trạng thái cuối: HOÀN THÀNH (terminal decision `FAIL`).** Sau khi
  independent freeze audit đã pass và được push, common evaluator mới mở
  validation GT với exact corrected baseline per-image SHA-256 `a26143d0...`,
  `10,000` complete-group bootstrap replicates và seed `20261014`. Lần gọi
  đầu bằng `runpy` dừng ngay tại import `mae_reconstruction_io` do chưa
  thêm `project` vào `sys.path`; nó chưa đọc input/GT, chưa tạo output.
  Lần kế nhiệm giữ nguyên mọi argument/hash, chỉ bổ sung module path và pass.
- Corrected T1 Dice validation overall/small/medium/large là
  `0.24282104 / 0.11700560 / 0.37493570 / 0.37139856`, so với accepted
  Geometry-v3 `0.24548239 / 0.11708058 / 0.37713552 / 0.38941265` chênh
  `-0.00266135 / -0.00007498 / -0.00219982 / -0.01801409`.
  Paired complete-group CI95 cho chênh lệch overall/small/medium/large là
  `[-0.00663235,-0.00003937] / [-0.00019516,0] / [-0.00669240,0] /
  [-0.05147943,0]`. Complete misses tăng `53 -> 54`; không baseline miss nào
  được recover và một baseline hit bị mất. Image AUROC là
  `0.81606022`.
- T1 có một kết quả cơ chế hẹp đúng mong đợi: absolute
  candidate-count/miss Spearman giảm `0.31350741 -> 0.29721823`. Tuy nhiên
  selected-to-oracle regret xấu hơn trong cả bốn cohort:
  overall/small/medium/large `+0.00266135 / +0.00007498 / +0.00219982 /
  +0.01801409`; overall score-quality Spearman chỉ `0.44337370`. Do đó
  count-control/self-paced residual không được adopt như performance
  improvement; nó chỉ là negative evidence rằng gỡ shortcut count chưa đủ
  để giảm selector regret, đặc biệt large extent.
- Evaluation hashes: `evaluation_audit.json`
  `70b86f9a8955da1fb7618ad77a3c0789711a1f88daabdda70248e18329c1f638`,
  `gate_decision.json` `1162729bf629d037b0b87cf3c01db4e03bec793e75a1d396f7a571d5f888fe75`,
  `paired_comparison.json` `3d85a1d3b51d2ba5f176380374c0f8cef88984e4446a850bbc989ab85d0c8fc4`,
  `per_image.csv` `d03c8ac27337bce6dbc7a1b9cd005aac1b9a0ffea3488eeb1ef597842f17c3bd`,
  `summary.json` `df8640ede7e4ec5d5eac9c8f1916e2d793f33a70e63ce18326e59764c09d9d2d`.
  Final decision SHA-256 là
  `71e6e10381e97f7fdeafe36bbba3954d6b64f52b83a3876d7097380fe1bcf93f`;
  mechanism/full-adoption/operational gates đều fail, bốn oracle goal vẫn pass.
  Consumer không được authorize/trained; BTXRD test không đọc.
  Bước tiếp theo là sync terminal evidence với nhánh cộng tác và
  chọn successor selector/aggregation không trùng, không rerun T1.

### EXP-20260801-codex-n1-normal-only-direct-anomaly-v1

- **Owner/registered/status:** Codex main task trên `research-wsss-improvement`;
  đăng ký `2026-08-01T14:57:11+07:00`; trạng thái `ĐANG LÀM`.
  Registration base là terminal-T1 commit
  `a496bf1c184beece7c794e1f504a568cc175b755`; exact claim commit sẽ được
  ghi sau khi note này được push. Chưa có N1 source/protocol/fit/prediction.
- **Coordination audit:** đã fetch central và collaborator. Collaborator head
  vẫn là `f22700be9f74aaa560e0da95326d318f29c9a59b`; G1 terminal official
  `0.20602633/0.10958465/0.30070545/0.33094975` xấu hơn same-gallery
  Geometry-v3, exploratory fusion `0.28872949/...` không đủ audit/adoption,
  còn G2 là claim rich-gallery source-balanced MIL đang làm và chưa có
  terminal metric. N1 không dùng rich gallery, external proposals, G1/G2
  logits/checkpoints/fusion/objective và không chạy cạnh tranh G2.
- **Inherited evidence, not adopted success:** N1 tham chiếu terminal rejected
  `EXP-20260731-codex-r1-normal-prototype-v1` error-audit SHA-256
  `833c4814eee0891df1cd55a01fa008e0708ad3485dd30b730657f52508317719`
  và `EXP-20260801-codex-t1-count-controlled-self-paced-v1` decision SHA-256
  `71e6e10381e97f7fdeafe36bbba3954d6b64f52b83a3876d7097380fe1bcf93f`.
  R1 cho thấy learned MIL residual trên normal-prototype features làm count
  shortcut tăng; T1 chứng minh gỡ shortcut có thể pass nhưng selector regret
  vẫn xấu hơn. Không mechanism nào được relabel là improvement.
- **New hypothesis/scope:** tách hẳn classification khỏi candidate ranking.
  Fit duy nhất một weighted spherical `K=32` nominal bank trên original +
  aligned-flip 1,156-D candidate descriptors của **train image-label-normal**;
  mỗi normal image có equal mass, sau đó equal family mass và equal
  candidate/view mass. Candidate score là mean original/flip nearest-prototype
  cosine distance; không BCE, positive pseudo-instance, residual, SmoothMax,
  learned selector hay threshold. Candidate có anomaly distance cao nhất được
  chọn. `K=32` là fixed largest bank trong finite R1 design để cover normal
  heterogeneity; không sweep/chọn bằng validation GT hoặc Dice.
- **Classification/selection decoupling:** all 371 bag logits/probabilities giữ
  exact accepted Geometry-v3 values; chỉ selected candidate/map thay đổi. Map
  là selected immutable bit-packed mask nhân exact accepted bag probability.
  Vì vậy image AUROC và count/probability Spearman phải reproduce baseline
  exact, còn all-candidate anomaly scores có contract riêng và không được
  diễn giải như classification logits. Đây là causal test mới của
  normal-only direct anomaly ranking, không lặp learned residual R1.
- **Primary technical source:** Roth et al., *Towards Total Recall in Industrial
  Anomaly Detection* (PatchCore), CVPR 2022,
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html.
  Chỉ transfer nguyên tắc nominal-feature memory/prototype bank và nearest
  feature-distance anomaly scoring; không transfer industrial benchmark,
  ImageNet encoder, pixel threshold, coreset ratio hay claimed performance.
- **Exact inputs:** selector-cache kernel
  `itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1`, freeze SHA-256
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  manifest `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`;
  split `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  accepted baseline checkpoint/freeze
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069` /
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`.
  Gallery/order/family/mask geometry/evaluator không đổi; oracle vẫn
  `0.40907553/0.22274949/0.59414708/0.64182537`.
- **Compute/protocol/gates:** static/synthetic implementation và tests local;
  mọi fit/scoring trên real cache chỉ chạy trong một Kaggle T4x2 hoặc P100
  job sau khi source/protocol/wrapper/binding fail-closed được push. Trước
  GT, independent auditor phải prove normal-only fit cohort/weights, exact
  K/prototypes, original/flip arithmetic, all 371 score payloads/maps, exact
  baseline probabilities/AUROC/count association và prediction freeze.
  Common evaluator sau freeze dùng exact corrected baseline per-image
  `a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f`,
  10,000 complete-group bootstrap và fixed seed mới. Mechanism pass vẫn yêu
  cầu regret giảm trong ít nhất hai tumor subgroup, overall Dice không giảm
  và count/miss association không tăng. Full adoption/consumer cần đủ
  `0.34024039/0.17895493/0.51244178/0.49370336`, positive overall CI95 lower
  bound, không subgroup decrease/miss increase và AUROC `>=0.75`.
- **Safety:** training chỉ image-level normal label; không validation GT,
  candidate Dice/oracle rank/subgroup/lesion-size routing trong fit/scoring;
  prediction phải freeze vật lý trước GT. Không train downstream consumer
  trước full operational pass; BTXRD test khóa. Fail thì reject N1, không
  sweep K/blend/threshold hoặc post-hoc rescue.

- **Primitive static/synthetic readiness:** exact claim commit
  `0e1b5daa895d28809e13ba106fbe60236b2ac909` đã push trung tâm trước
  implementation. Dataset-agnostic primitive
  `project/models/mask_bag_normal_anomaly.py` và test có SHA-256
  `e74fb687168620d84317d57b240b1007af75586ca3c16a7dd5052e35aa1f6cbd` /
  `c1a7d432192b2445d51ced3794bdd7b4802ab47b8c7cf1b68f4372c6d0a7fc3f`.
  Nó fail closed nếu có positive image label, duplicate image, sai view/family/
  dimension, nonfinite/zero-norm descriptor hoặc khác frozen `K=32, seed=42`;
  audit equal image→family→candidate/view mass và exact averaged-view score.
- `py_compile`, focused synthetic tests `4/4` trong 2.50 giây và full repository
  regression `494/494` trong 20.35 giây pass bằng documented Python-3.9
  strict-zip shim. Primitive-readiness artifact SHA-256 là
  `5d4b07c8b902960fcce689d5702a07da198b43008cd8f7b670a8cf4d73c1fe86`.
  Đây chỉ là static/synthetic readiness: chưa fit real cache, chưa tạo
  validation prediction, chưa mở GT/consumer/test. Bước kế là runner,
  protocol và independent auditor fail-closed; chưa authorize Kaggle launch.

### Repository cleanup trước khi tiếp tục N1 (2026-08-01)

- Theo yêu cầu người dùng, N1 tạm dừng trước protocol/launch để
  audit dọn code/rác. Worktree chỉ có hai file runner/test N1 mới đang
  chuẩn bị; chúng được giữ lại vì là dependency của claim active,
  không phải rác. Toàn bộ Kaggle output, exact source snapshot, baseline
  transport, selector cache và tracked source của experiment terminal cũng được
  giữ nguyên để không xóa provenance/evidence.
- Audit `259` tracked file trong `project/` + `tests/` không phát hiện
  duplicate-content hash group hay zero-byte file. Không xóa tracked experiment
  code chỉ vì nó cũ: các source/protocol/auditor đó là bằng chứng tái
  lập của kết quả âm/lỗi và được `AGENTS.md` yêu cầu bảo toàn.
- Dọn `72` generated `__pycache__/.pytest_cache/.mypy_cache/.ruff_cache/
  .ipynb_checkpoints` directory, ban đầu gồm `1,422` file /
  `18,284,927` byte; thêm package ignored S4 `aborted_invalid_claim` không có
  reference (`2` file / `6,584` byte) và hai temp directory rỗng. Sau regression,
  `7` cache directory tái sinh (`228` file / `1,973,589` byte) cũng được dọn.
  Tổng `1,652` file / `20,265,100` byte được chuyển vào **Windows
  Recycle Bin**, nên có thể phục hồi; workspace còn `0` generated-cache
  directory và không có `.part/.tmp` download.
- Direct `Remove-Item` bị execution safety policy chặn trước khi xóa; correction
  chỉ dùng Recycle Bin API sau khi resolve mọi target nằm trong `D:\thesis`
  và prove cache target được git-ignore. Focused N1 runner test lần đầu
  có `1` assertion-only failure do match tên imported function trước `main()`;
  test được thu hẹp vào main body, không đổi scientific runner.
- Sau correction, focused N1 primitive/runner `7/7` trong 2.41 giây và full
  repository `497/497` trong 22.58 giây pass. Cleanup audit
  `artifacts/maintenance/repository_cleanup_audit_20260801.json` có SHA-256
  `46e015409b8ade253003daf2ba701d3276895b7b461ab623bea8c088b19d565d`.
  Cleanup không mở validation GT/BTXRD test, không tạo prediction, không train
  consumer. Sau commit/push audit này, N1 mới được tiếp tục static readiness.

- **N1 fail-closed runner static readiness sau cleanup:** runner
  `project/run_mask_bag_normal_only_direct_anomaly_n1.py` và test có SHA-256
  `6a279ea568c2bcbf2a6ea89998e24141e9ebaf33b7d7a15a7056f778aa9ed85d` /
  `a3f90444f8e2f94f8d220122b91c5a98d2c7d9dd4a1c5b2bfbfc6c5528c27fb5`.
  Runner fail closed trên exact split/cache/baseline freeze/checkpoint/source/
  protocol và `1493` normal training images trước khi tạo output directory;
  yêu cầu exact T4x2, `K=32/seed42`, fit không positive bag/optimizer,
  giữ exact accepted bag logits/probabilities, và freeze `371` all-candidate
  anomaly-score payloads, `371` original/flip evidence payloads cùng `371` maps.
- Candidate score được khai báo rõ là normal-distance ranking, không phải
  classification logit; map là selected immutable mask nhân accepted baseline
  probability. Runner tự yêu cầu final count/probability Spearman reproduce exact
  `0.48137777593654113`. Nó không import segmentation dataset/evaluator/consumer.
  Focused primitive+runner `7/7` trong 2.78 giây và full repository `497/497`
  trong 20.93 giây pass; cache sinh ra bởi test đã được dọn lại.
  Runner-readiness artifact SHA-256 là
  `fba7f143aefa9cbe0e698e0333f1620b90e0d10048c5c62332dc58247d4583ee`.
  Vẫn chưa có protocol/auditor/wrapper/binding hay real-cache fit/prediction;
  validation GT, consumer và BTXRD test giữ khóa.

- **N1 runner source-closure refinement (static, supersedes initial runner
  readiness):** trước khi đóng băng protocol, audit dependency phát hiện runner
  N1 đang import toàn bộ scientific runner R1 chỉ để dùng hai helper đọc/kiểm
  tra selector cache. N1 nay tự kiểm tra freeze/manifest/cohort/safety và chỉ gọi
  primitive I/O `models/mask_bag_selector_cache_io`; hypothesis, `K=32/seed42`,
  equal image→family→candidate/view weighting, score, output contract và mọi gate
  khoa học không đổi. Runner mới có SHA-256
  `c1c34a5ef6c64a0f46339ea435a8566fc8ae4e327d5fb6ff243a5052b8ddcc9a`;
  test giữ SHA-256
  `a3f90444f8e2f94f8d220122b91c5a98d2c7d9dd4a1c5b2bfbfc6c5528c27fb5`.
- Focused suite `7/7` trong 2.82 giây và full repository `497/497` trong
  22.18 giây pass bằng pinned `btxrd-pseudomask` environment cùng documented
  Python-3.9 strict-zip shim. Lần gọi full regression đầu tiên dùng nhầm system
  Python không có NumPy nên dừng tại collection; không mở real cache/dữ liệu và
  không có scientific result. Rerun trên đúng environment pass toàn bộ, cache
  test phát sinh đã chuyển vào Recycle Bin. Refinement-readiness artifact SHA-256
  là `4fe530618fffb91444b8c5fe01b0b215e986719719be5cc51f707facdabb8573`.
  Chưa fit real cache, chưa tạo prediction, chưa mở validation GT/consumer/test;
  initial runner-readiness được supersede chỉ ở source closure.

- **N1 protocol + independent auditor static readiness:** protocol duy nhất
  `artifacts/research_protocols/rad_dino_mask_bag_normal_only_direct_anomaly_n1_v1.json`
  đã freeze với SHA-256
  `1112515d00ed9db80a05670404ad16127109844788a44018834ff82f452d9b7d`
  trên scientific-source commit
  `c7ba620ce4492485ba0faa6dd42998e267be872d`. Nó pin normal-only cohort
  `1493`, descriptor `1156-D`, weighted spherical `K=32/seed42`, tối đa 100
  iteration/tolerance `1e-6`, score mean original/flip nearest-normal distance,
  accepted Geometry-v3 logits/probabilities, T4x2, `371` freeze records và
  post-freeze bootstrap seed `20261015`; không K/seed/blend/threshold sweep.
- Auditor độc lập
  `project/audit_mask_bag_normal_only_direct_anomaly_n1_output.py` có SHA-256
  `5b8e239793e2816839e2b74e80b077d619eb9650f49073df1038ce5b648b56ee`.
  Nó không import N1 runner, scientific normal-anomaly primitive hay evaluator;
  tự đọc physical hash-bound cache, tái dựng toàn bộ equal
  image→family→candidate/view weights và spherical bank, rồi tái tính mọi
  original/flip distance, score, first-argmax candidate, bit-packed mask × exact
  baseline probability map và count/probability Spearman trước GT. Binding phải
  khớp protocol/source hashes và exact self-hash của auditor.
- Protocol/auditor/runner focused suite `14/14` trong 2.69 giây và full
  repository `504/504` trong 23.93 giây pass; generated test caches đã chuyển
  lại vào Recycle Bin. Test SHA-256 là
  `3b892c46aa4e204c3e996b6c935a8412d778b3e7927e7ef7c963298c8db21751`;
  readiness artifact SHA-256 là
  `137d84b52d73a3e1e2426767ce7a3154f7726d1c9b9a9d6bc6efa9ae7599f070`.
  Đây vẫn chỉ là static/synthetic readiness: chưa mở real cache để fit, chưa
  tạo validation prediction, chưa mở GT/consumer/test. Bước kế là wrapper +
  launch binding fail-closed và phải push trước một Kaggle T4x2 launch duy nhất.

- **N1 wrapper/binder static readiness:** fail-closed Kaggle template
  `project/kaggle_wrappers/run_mask_bag_normal_only_direct_anomaly_n1_v1.py`
  có SHA-256
  `6228baf22e5b6a101ca9a9642cd404f2ec4ff944cb66bc9fb224eb66cea84868`;
  binder SHA-256 là
  `ed459a7f4725ffbb595b2c1455f3266aaf3c7b0ec4551c9b2750bf2ae84da25d`.
  Template mặc định `KERNEL_VERSION=0`, binding false và checkout unbound;
  binder chỉ được đổi đúng ba field này và phải inverse-reconstruct exact template.
- Wrapper phải prove exact checkout/protocol/all canonical source/auditor hashes và
  real T4 convolution trên cả hai GPU trước input preparation; sau đó dựng exact
  CRLF split, xác minh safe baseline transport/cache, chạy tests, chạy runner một
  lần, physical-freeze prediction, rồi mới gọi independent auditor. Runtime
  launch binding được tạo trước scientific runner; independent audit phải pass
  full bank/371 score/view/map recomputation trước wrapper audit cuối. Không có
  evaluator/GT/consumer/test import hay execution trong wrapper.
- Focused suite lần đầu có `1` assertion-only failure vì test match auditor
  filename constant trước `main()`; correction giới hạn order assertion vào thân
  `main()`, không đổi wrapper/protocol/mechanism. Sau correction focused `17/17`
  trong 3.38 giây và full repository `507/507` trong 21.88 giây pass; cache test
  đã chuyển vào Recycle Bin. Binder test chứng minh exact 3-field replacement,
  inverse reconstruction và launch binding được independent auditor chấp nhận.
  Test SHA-256 là
  `7a2927e3901aa70ed58aa30d9648a680349305b872cef29c798025466da7f00f`;
  readiness artifact SHA-256 là
  `5b1730fe5785cdf5610e2240d4fda88c15b8350976d6d884d174c22a0fb88792`.
  Vẫn chưa bind package, fit real cache, tạo prediction hay mở GT/consumer/test.

- **N1 kernel-v1 bound prelaunch gate:** ngay trước binding đã fetch central và
  collaborator; central/HEAD cùng ở
  `a6106e6e6fd35040ad2f82e2128c012462c40ac1`, collaborator vẫn
  `f22700be9f74aaa560e0da95326d318f29c9a59b`. Claim collaborator duy nhất liên
  quan vẫn là rich-gallery G2; N1 giữ old same-gallery cache nên không overlap.
  Worktree sạch trước binding và không có claim mới cạnh tranh.
- Binder đã đóng băng kernel version 1 với execution checkout `a6106e6e...`.
  Bound wrapper SHA-256 là
  `fda247c98409311abcdd645d2c4e9f531893d9e1e3fff08d799ddd1a961fd8c2`;
  launch binding SHA-256
  `4d77c0d0cd14ae48b77bf22263bf3660c48e7cf51a69a7e0060633ebf2d6a9c3`;
  metadata SHA-256
  `c8ec60d9777c6bcf6c8520d17c1a0d11dee5f0bf05f07cc04b011d273475f8fa`.
  `py_compile`, exact inverse reconstruction và independent-auditor acceptance
  pass. One-off static runtime test cũng prove wrapper tái dựng exact launch-
  binding core từ checkout/protocol/source/auditor/bound-wrapper hashes.
- Final prelaunch audit SHA-256 là
  `db9a414e1a9bac234d6c8850226b7cea8509ba45fe78be09f0acc3b9d5b54b76`;
  status `READY_FOR_SINGLE_KAGGLE_T4X2_PUSH_NOT_YET_LAUNCHED`. Exact inputs vẫn
  là selector-cache freeze `2f6290...`, cache audit `cc2528...`, baseline archive
  `8857eb...`, baseline freeze `ec3462...` và split `85511e...`. Chưa fit cache,
  chưa prediction/GT/consumer/test. Phải push binding/audit central thành công
  trước một launch duy nhất; không launch nếu push hoặc final status gate fail.

- **N1 kernel-create ERROR trước scientific execution:** sau khi binding/audit
  commit `dfe9e6b` đã push central, lệnh duy nhất `kaggle kernels push` trả
  `400 Bad Request` từ `KernelsApiService/SaveKernel`. Server không tạo
  kernel/version/job; một bounded status check chỉ trả `kernels.get permission
  denied / most likely wrong slug`, không có RUNNING/terminal version. Vì lỗi ở
  request-creation boundary nên không mở Kaggle input, không fit real cache,
  không prediction/GT/consumer/test và không có scientific result.
- Exact intended wrapper/binding/metadata vẫn là `fda247c9...` / `4d77c0d0...` /
  `c8ec60d9...`. Kaggle CLI `2.2.3` không expose server error body. Static
  localization ghi nhận slug/title dài `56` ký tự, trong khi working T1 slug
  dài `46`; metadata identifier length là root-cause hypothesis kỹ thuật duy
  nhất hiện tại nhưng **chưa** được coi là confirmed hay sửa trước khi note này
  push. Error audit SHA-256 là
  `bcca2138a6793729266ca679b48cf7de2d13facc7f61164dc8db621e71e977e5`.
  Official Kaggle CLI kernel-push reference đã xem:
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md; installed CLI
  request source cũng xác nhận 400 xảy ra tại SaveKernel sau local metadata parse.
  Package `__pycache__` phát sinh do prelaunch compile đã chuyển vào Recycle Bin.
  Phải push error boundary này trước mọi identifier correction hoặc rerun.

- **N1 short-slug transport correction ready, chưa rebind/rerun:** sau error
  commit `560aee4` đã push, kernel identifier được rút gọn từ slug 56 ký tự
  `btxrd-rad-dino-mask-bag-normal-only-direct-anomaly-n1-v1` xuống 44 ký tự
  `btxrd-rad-dino-mask-bag-normal-anomaly-n1-v1`. Đây chỉ là transport metadata/
  binding identity correction; scientific source commit, protocol SHA
  `1112515d...`, cache/baseline/split, K32/seed42, output và evaluation gates
  không đổi. Không tuyên bố root cause confirmed cho tới khi SaveKernel accept.
- Vì kernel identity nằm trong independent auditor và wrapper binding, exact
  auditor/template/binder hashes mới lần lượt là
  `562a3d4bce2ed8e27cc11147577fbca9600c2b19273c16ce1970d1f3faddfdf1` /
  `b5d7c6b90796aeb9e338ae2a1559d7fedcd7a06d23cb883f518c4da8cbc113ec` /
  `5ae8a052e2fc4d07b687fd140bc590112e224edb95bff6b4d55a2c3242dbc603`.
  `py_compile` và 16/16 test tĩnh không cần execution-checkout mới pass trong
  3.20 giây. Correction-readiness SHA-256 là
  `e45b15417272c03c9a027163d4cfc22b5f036df2860661ff1af5c45af03d1d98`.
  Chưa rebind package, chưa gửi SaveKernel lần hai, chưa fit/prediction/GT/
  consumer/test. Phải commit/push correction source trước full binder regression.

### Đồng bộ G2 terminal và deep-search sau chuỗi selector reject (2026-08-01)

- Đã fetch lại `origin/research-wsss-improvement` và
  `origin/codex/research-sync-20260731`, đọc lại toàn bộ trạng thái điều phối trên
  nền hai log đầy đủ đã audit. Central/HEAD ở `ec2e6b26d1b4016b8b06c6cacb9a518e34a5ed7b`;
  nhánh cộng tác mới tăng từ `f22700b...` lên
  `53ac916c13525946d8cfa7662857694bb6c33dde`. Delta log duy nhất là kết quả
  terminal G2 dưới đây; commit cộng tác không được merge code vào central và
  không ghi đè bằng chứng của workstream nào.
- **Kết quả terminal cộng tác —
  `EXP-20260801-codex-rich-gallery-g2-terminal-v3`:** private/offline T4x2
  kernel `wanwin/btxrd-rich-gallery-g2-selector-pair` version 3 hoàn tất, audit
  Stage-A/Stage-B pass, freeze đủ `2,968` lựa chọn (`8 x 371`) trước khi mở
  validation polygons, cohort `371/184` với subgroup `94/72/18`, zero test read.
  Protocol SHA-256 là
  `6dd6c66e396054157eb498cbca46635d1058c73d1139541d385bb769088801be`;
  prediction-freeze/evaluation-summary SHA-256 là
  `78970d417de20dc884958dfb3fd9cb2bad9f2cda53240ae2207607ba827cd167` /
  `813b5a0c9506d2052508f1a8ffe2a401e947183bef452eaef3330b3a48cdd9b6`.
- G2 tốt nhất (hierarchical shared-source negative-only + rank fusion) chỉ đạt
  Dice overall/small/medium/large
  `0.25432565/0.13792543/0.37614745/0.37490625`, `62` complete misses; thấp
  hơn frozen G1 rank-fusion reference `0.28872949` một lượng `-0.03440384`,
  paired complete-group CI95 `[-0.058333,-0.014022]`. Do đó G2 fail mọi primary
  subgroup gate; không adopt negative-only/hierarchy/temperature continuation
  như cải tiến.
- Một causal partial result dương được giữ làm bằng chứng, không đổi thành full
  promotion: loại external-source shortcut khỏi training loss giúp flat hard-top
  raw tăng G1 raw `0.20602633 -> 0.24850185`, delta `+0.04247552`, CI95
  `[0.010932,0.075269]`, đồng thời median selected/GT area giảm `13.02 -> 3.05`.
  Tuy nhiên misses tăng `29 -> 44`; mọi arm cuối cùng vẫn collapse về khoảng
  `1.74-1.87` effective candidates. Bỏ shortcut là cần thiết nhưng không sinh ra
  positive-instance identity trong tumor bag.
- Failure dossier cộng tác chứng minh bottleneck kế tiếp là hit/positive-instance
  ranking hơn là gross area: G2 fusion có median area ratio gần G1 fusion
  (`1.96-1.99` so với `2.04`) nhưng có `62-64` misses thay vì `49`; trên các ca
  đổi lựa chọn mất trung bình khoảng `0.070` Dice. Fusion G1 từng thêm
  `+0.082703` trên raw G1, còn chỉ thêm `+0.003239/+0.010234/+0.030551` trên ba
  raw arm G2 vì scorer G2 đã trở nên trùng thông tin upstream. Kết luận: phải có
  candidate-level positive evidence mới, đồng thời giữ complementarity/hit
  recall; không sweep tiếp temperature/epoch/threshold/resolution/rank weight.
- **N1 chuyển `TẠM DỪNG` trước scientific execution:** claim
  `EXP-20260801-codex-n1-normal-only-direct-anomaly-v1` chưa từng qua
  `SaveKernel`, chưa mở real cache, chưa fit hay sinh prediction. Deep audit phát
  hiện hypothesis nominal-only bị bằng chứng terminal cũ phản đối trực tiếp:
  healthy K32 density pixel AUC chỉ `0.563712`, true-area Dice `0.024037`,
  feature-normal replacement `0.023373`, causal-patch `0.014293`; R1 normal
  residual còn làm count shortcut tăng. Vì vậy không dùng short-slug correction
  để chi thêm T4x2 cho N1. Toàn bộ source/protocol/error/readiness vẫn được giữ
  như bằng chứng; trạng thái pause này không phải scientific reject và không có
  Dice N1 để báo cáo.
- **Loại các ý tưởng trùng trước khi chọn successor:** classifier-causal
  deletion/insertion đã được chạy ở Gate-C và bị reject (`0.22671749` so với
  `0.23433922`, delta `-0.00762174`, CI95
  `[-0.02707981,+0.01175732]`); fixed source-consensus cũng bị reject
  (`0.21250531` so với `0.23433922`). Do đó không được đổi tên rồi chạy lại hai
  selector này. R2/R4/T1 còn chứng minh AUROC, flip invariance hoặc count-control
  có thể pass trong khi Dice/regret xấu hơn; chúng chỉ là operational guards,
  không phải surrogate localization gate.
- **Deep-search primary evidence:** BAS chỉ ra CE có thể bão hòa khi mask mới phủ
  phần discriminative, trong khi foreground/background activation tiếp tục thay
  đổi tới biên; phương pháp dùng activation suppression + foreground guidance +
  area constraint:
  https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Background_Activation_Suppression_for_Weakly_Supervised_Object_Localization_CVPR_2022_paper.html
  và official code https://github.com/wpy1999/BAS. Kim et al. phân rã CAM thành
  feature norm và cosine alignment với class vector, rồi dùng feature-direction
  alignment + attentive-dropout consistency để đưa evidence từ classification
  sang localization:
  https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Bridging_the_Gap_Between_Classification_and_Localization_for_Weakly_Supervised_CVPR_2022_paper.pdf.
  Extremal Perturbations định nghĩa preservation/deletion theo fixed mask area và
  tìm vùng nhỏ nhất còn giữ đủ activation, phù hợp cho discrete candidate curve:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Fong_Understanding_Deep_Networks_via_Extremal_Perturbations_and_Smooth_Masks_ICCV_2019_paper.html.
- ToCo cho thấy intermediate ViT tokens giữ semantic diversity tốt hơn final
  oversmoothed tokens và dùng local-global class-token contrast cho vùng không
  chắc chắn:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html.
  L2G cho thấy local crops có thể lộ chi tiết object mà global classification bỏ
  qua và transfer local attention về global:
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.html.
  MIL-Dropout báo cáo top-instance dropout có thể giảm winner concentration,
  nhưng BTXRD chỉ có một proposal hữu ích ở nhiều small bag nên chỉ là
  regularizer bậc hai, không phải successor chính:
  https://proceedings.mlr.press/v267/zhu25q.html.
- **Synthesis cho successor:** proposal supply same-gallery vẫn đủ vì oracle
  `0.40907553/0.22274949/0.59414708/0.64182537` vượt cả bốn goal. Hướng có expected
  value cao nhất không phải thêm một MIL residual mà là tạo một class-aware
  localization descriptor bằng image-label-only activation training, sau đó
  score chính các immutable candidate masks bằng foreground capture/background
  suppression và fuse theo within-image rank với accepted Geometry-v3/upstream
  evidence. Candidate gallery, split và masks không đổi; không hard pseudo-positive,
  validation-size routing hay GT-tuned area/threshold. Một arm BAS-like/fixed-area
  candidate rerank phải được thiết kế và test tĩnh riêng; chỉ đăng ký `ĐANG LÀM`
  và launch sau khi exact architecture/default losses, OOF image-label gates,
  prediction-freeze contract và independent auditor được freeze. Hiện chưa có
  experiment successor, real-data training, prediction, validation-GT read,
  consumer training hay BTXRD-test access trong mục synthesis này.

### B1 BAS candidate descriptor — chuẩn bị primitive tĩnh (2026-08-01)

- Đã hiện thực primitive image-label-only trong
  `project/models/bas_candidate_localizer.py` (SHA-256
  `f285a69aaec47ab19adf26a9b979bd838eb1959b69252756be8005dcc85e37c9`).
  Kiến trúc giữ đúng resolution path của BAS ResNet chính thức: layer3 stride 1
  tạo localization map output-stride 8, max-pool 2x rồi layer4 stride 1 tạo
  classification map output-stride 16. Nhánh erased-background được đồng bộ từ
  trọng số hiện tại ở **mỗi forward**, frozen parameter nhưng vẫn truyền gradient
  tới localization map; như vậy không còn sai khác epoch-stale so với semantic
  per-forward deep copy của source chính thức. Loss primitive là background/full
  activation ratio cùng area weight cố định `1.2`; candidate descriptor là
  harmonic mean của activation coverage và purity sau per-image min-max
  normalization. Hai primitive selector chỉ gồm raw harmonic rank và fixed `1:1`
  within-bag rank fusion với score Geometry-v3; không có area/subgroup/GT weight.
- Test `tests/test_bas_candidate_localizer.py` (SHA-256
  `51c83144866b84ab7f081fdda36b3870b6180e9e2200952cd67ea41682149ceb`) kiểm
  tra công thức BAS, normalization constant-safe, coverage/purity, tie-aware rank,
  fixed fusion và fail-closed shape. `py_compile` pass; focused suite pass `6/6`
  trong 2.51 giây bằng pinned `btxrd-pseudomask` environment. Synthetic forward
  `1x3x64x64` xác nhận localization `1x1x8x8`, classification `1x2x4x4` và BAS
  loss finite. Không có real image/cache/GT/test được mở trong các kiểm tra này.
- Readiness tĩnh
  `artifacts/research_protocols/bas_candidate_descriptor_b1_primitive_readiness.json`
  có SHA-256
  `e840853dfdcdae11843d006788c933cf3bb81a6a897933626b6f2a960f858385`; nó ghi
  exact mechanism, primary source URLs, file hashes và scientific boundary.
  Đây **chưa phải** đăng ký thực nghiệm: chưa fit real data, chưa tạo prediction,
  chưa đọc validation segmentation GT, chưa train consumer và chưa mở BTXRD test.
  Trước khi mở claim B1 phải fetch lại hai branch điều phối, kiểm tra collision và
  push `ĐANG LÀM` theo `AGENTS.md`.

### Đồng bộ audit `0.28872949` và hiệu chính thiết kế tĩnh B1 (2026-08-01)

- Theo yêu cầu người dùng, đã fetch/`ls-remote` lại branch cộng tác, đọc toàn bộ
  `RESEARCH_LOG.md` 8,154 dòng và toàn bộ
  `RICH_GALLERY_G2_FAILURE_DOSSIER.md` tại
  `origin/codex/research-sync-20260731` commit
  `53ac916c13525946d8cfa7662857694bb6c33dde`. GitHub remote vẫn trỏ đúng commit
  này ở hai lần fetch độc lập; chưa có commit mới hơn bị bỏ sót. G2 v3 đã audit
  lại 371/371 G1 choices và xác nhận rich-gallery G1 + upstream equal-rank
  diagnostic đạt Dice `0.28872949/0.15772330/0.43522933/0.38687353`, tốt hơn
  same-gallery Geometry-v3 ở overall/small/medium nhưng large giảm; complete
  misses tăng `29 -> 49`. G2 tốt nhất vẫn chỉ `0.25432565` và thua reference này
  có CI95 hoàn toàn âm. Vì vậy insight được kế thừa là **complementarity của hai
  rank**, không phải G1/G2 model hay proposal-supply implementation.
- Ranh giới thống kê vẫn được giữ: fusion `0.28872949` được thiết kế sau khi xem
  Stage-B trên cùng validation nên là evidence thích nghi, không phải endpoint
  độc lập để quảng bá. Nó không được dùng để tune weight/threshold. B1 không tải
  private rich-gallery output và không chạy lại G2; candidate gallery vẫn là
  same-gallery cache đã chứng minh oracle support.
- Thiết kế B1 ban đầu đã được hiệu chính **trước claim/real-data execution** vì
  raw BAS coverage/purity có nguy cơ chỉ sao chép upstream extent signal. Frozen
  finite comparison mới gồm transfer control
  `(Geometry-v3 rank + immutable upstream selection-score rank)/2` và duy nhất
  một B1 arm không weight sweep
  `(Geometry-v3 rank + upstream rank + BAS rank)/3`. GT-blind gate mới yêu cầu
  mean BAS/upstream rank correlation `<=0.80` và ít nhất `5%` lựa chọn thay đổi;
  nếu fail thì cấm đọc validation polygons. Thiết kế đầy đủ nằm ở
  `BAS_CANDIDATE_DESCRIPTOR_B1_DESIGN.md` SHA-256
  `5b8884f08209a9293cc75535d80106e83746a465430cae4cf3a92a3ade16cda2`.
- Runner tĩnh `project/run_bas_candidate_descriptor_b1.py` SHA-256
  `d62c4b3b2d388927e5f7c297850a43b7af97db3574f6b77b7e7b50de173f43b0`
  khóa official-style 100-epoch/224px/SGD BAS recipe, exact official ImageNet
  weight `resnet50-11ad3fa6.pth` 102,540,417 bytes SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`,
  baseline/upstream/cache identity, T4x2 và pair prediction freeze. Model
  primitive mới SHA-256
  `8683339833ffb682152a45e1964d2031a67da18c7134efcd8d48e4abb844ad6c`;
  AMP BAS ratio được ép float32 để `1e-8` không underflow và output đổi sang
  namedtuple tương thích DataParallel. Focused primitive+runner suite pass
  `10/10` trong 3.11 giây; test đã phát hiện và sửa một Python-3.9
  `zip(strict=True)` boundary trước launch. Full repository regression pass
  `517/517` trong 21.93 giây bằng pinned environment và documented strict-zip
  compatibility shim.
- Correction-readiness
  `artifacts/research_protocols/bas_candidate_descriptor_b1_transfer_correction_readiness.json`
  có SHA-256
  `97cb666ed502e18d324f5f50ba267a9e68980325256d1790cfaad7b21c235179`.
  Đây vẫn chỉ là chuẩn bị tĩnh: không real radiograph/gallery/cache được mở,
  không training/prediction/validation segmentation GT/consumer/test và không
  compute nặng local. B1 chưa có claim; protocol độc lập, auditor, wrapper và
  launch binding vẫn phải hoàn tất trước đăng ký/chạy.

### B1 source-closure và independent-auditor readiness (2026-08-01)

- Static dependency audit sau commit `6d026d6` phát hiện runner B1 còn import
  private helper từ toàn bộ R1/S3 scientific runners chỉ để verify cache,
  baseline và ghi output. Coupling này đã được loại: runner nay tự thực hiện các
  kiểm tra provenance/output tương ứng và chỉ dùng primitive cache/score/model
  I/O chung. Activation evidence được lưu float32 thay vì float16 để auditor có
  thể tái tính coverage/purity/rank đúng tolerance trước GT. Runner mới SHA-256
  là `65366f42080a44d0f1f678e37f917922d0938d8c9c428173958b3bddd8d2e6bc`;
  recipe/arms/gates khoa học không đổi.
- Independent GT-blind auditor mới
  `project/audit_bas_candidate_descriptor_b1_output.py` SHA-256
  `81713f74a854ae5cd681302a71be266e271e64b831940a4582efe90dd7a5f55c`
  không import runner/BAS scientific source. Nó independently reload scorer
  Geometry-v3 từ checkpoint, tái tính original/aligned-flip base logits, đọc
  immutable upstream score, min-max BAS coverage/purity/harmonic, tie-aware
  ranks, hai arm Borda, selected indices/maps, pair freeze và complementarity
  gate trên đủ 371 ảnh. Auditor không nhận dataset root/annotation/test path nên
  không thể mở segmentation GT trong Stage-A.
- Test runner/auditor SHA-256
  `4b7df9282ae1053ea00133520ef22c236db8516ff8a594ba16df0fefad262d4b`;
  focused suite pass `11/11` trong 3.19 giây, bao gồm independent tie-rank và
  activation-evidence reproduction. `py_compile`/`git diff --check` pass. Chưa
  mở real image/gallery/cache, chưa fit/prediction/GT/consumer/test và chưa có
  heavy local compute. Đây vẫn là static readiness; protocol/claim/launch chưa
  tồn tại.

- **Source-closure correction before protocol:** runner còn một import private
  `_audit_candidate_input` từ RAD-DINO probe. Nó nay gọi trực tiếp primitive
  `validate_candidate_diagnostics_manifest` và tự khóa `cohort=all/371`; không
  còn import scientific runner nào khác. Runner SHA-256 cuối là
  `7ba49e38cb829fd8cdf7d20bd142713f5a9887d3011ec3e2a2b62fbe2295e20e`;
  focused `11/11` vẫn pass trong 3.19 giây. Không scientific boundary nào đổi.

### B1 frozen static protocol readiness (2026-08-01)

- Protocol `artifacts/research_protocols/bas_candidate_descriptor_b1_v1.json`
  đã freeze với SHA-256
  `5b9a80c63331551ff2c4ba0140096c14fa27076e141b83129e774843a7a7fde8`
  trên scientific-source commit
  `c2ef1322b6e28436b29d45b2d47459ea4b1e2dd9`. Nó khóa exact split/cache/
  baseline/candidate/ImageNet hashes, official-style 100-epoch BAS recipe,
  transfer control, duy nhất một three-way B1 arm, GT-blind classification +
  complementarity gates, pair prediction freeze, independent auditor, paired
  post-freeze evaluator và no-sweep/no-consumer/no-test rules.
- Protocol-source closure test tái hash 17 canonical file; focused primitive +
  runner + auditor + protocol suite pass `12/12` trong 3.16 giây. Protocol hiện
  ghi rõ `STATIC_PREDECLARED_NO_CLAIM_NO_BINDING_NO_LAUNCH`: chưa có experiment
  registration, wrapper, Kaggle job, real training/prediction hay validation GT.

### Đồng bộ rich-gallery `0.28872949` và supersede B1 trước claim (2026-08-01)

- Đã fetch và đọc toàn bộ `RESEARCH_LOG.md` của
  `origin/codex/research-sync-20260731` tại commit
  `d849155e49372c2027a1168fbe4a0b68e199470d`, gồm toàn bộ dossier
  `RICH_GALLERY_G1_FUSION_BOTTLENECK_DOSSIER.md`, handoff máy đọc được và
  protocol consensus mới. Không có claim B1 cạnh tranh; workstream cộng tác đã
  hoàn tất thêm hai phân tích terminal sau G2.
- Kết quả tốt nhất đã quan sát vẫn là lựa chọn rich-gallery theo trung bình hai
  percentile-rank G1 và upstream: Dice/IoU
  `0.28872949/0.21683918`, subgroup Dice
  `0.15772330/0.43522933/0.38687353`, `49` complete misses. Toàn bộ `371`
  choice đã freeze trước validation polygon và tái lập được, test read/evaluate
  bằng `0`; tuy nhiên đây vẫn là exploratory validation best vì rule equal-rank
  được thiết kế sau một Stage-B validation trước đó, không phải confirmatory
  endpoint độc lập.
- Gallery oracle đạt `0.52829833/0.33187635/0.73025092/0.74624721`. Selector
  regret `0.23956885` được phân rã thành cross-source `0.07079633` (`29.55%`),
  within-selected-source `0.16837621` (`70.29%`) và truncation chỉ `0.0003963`.
  Oracle trong top-3/5/10/20/50 của ordering hiện tại cho upper-bound Dice
  `0.341837/0.364894/0.399326/0.426336/0.475500`; cả `49` miss đều có proposal
  tốt hơn trong gallery nhưng oracle rank median là `95`. Vì vậy next mechanism
  phải thêm tumor-specific positive-instance evidence; tạo/giữ thêm cùng loại
  proposal hoặc chỉ rerank top-k không giải quyết toàn bộ miss branch.
- Global cross-source consensus đã terminal-reject: consensus-only/equal/product
  đạt Dice `0.16045251/0.26707272/0.25399301`, đều thấp hơn `0.28872949`.
  Equal/product giảm miss `49 -> 44/41` nhưng small Dice giảm
  `0.157723 -> 0.106249/0.104109`; small selected/GT area ratio tăng
  `14.60 -> 38.99/53.10`. Agreement gần `0.966` chủ yếu đo giải phẫu xương lặp
  lại, không phải tumor identity. Không adopt và không sweep consensus.
- Bằng chứng dương được adopt là **rich proposal union + fixed G1/upstream rank
  pair**; bằng chứng âm G2/consensus chỉ dùng để loại hypothesis. Protocol tĩnh
  `bas_candidate_descriptor_b1_v1` SHA-256
  `5b9a80c63331551ff2c4ba0140096c14fa27076e141b83129e774843a7a7fde8`
  vì thế được supersede trước claim/launch: không chạy nguyên trạng trên old
  gallery/baseline thấp hơn. Successor B1 sẽ giữ BAS image-label-only như một
  tumor-specific residual descriptor nhưng phải score chính rich gallery và so
  trực tiếp với fixed G1/upstream baseline `0.28872949`; không regenerate/copy
  G0/G1/G2, không source-presence routing và không dùng GT-derived per-image
  table trước prediction freeze.
- Artifact boundary để successor có thể launch mà không lặp compute là một
  transport GT-blind từ workstream `wanwin`: exact train/validation candidate
  payloads+manifests, G1 checkpoint, G1 frozen selection manifest, upstream
  scores, source/protocol/split hashes và audit xác nhận không chứa validation
  segmentation GT/test. Chỉ chuẩn bị adapter/unit test tĩnh được phép trước khi
  transport này tồn tại và được audit; không mở claim hoặc launch trên dữ liệu
  thật trong entry này.
- Cơ sở phương pháp đã đối chiếu lại: Wu et al., *Background Activation
  Suppression for Weakly Supervised Object Localization*, CVPR 2022,
  https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Background_Activation_Suppression_for_Weakly_Supervised_Object_Localization_CVPR_2022_paper.html;
  Tang et al., OICR, CVPR 2017,
  https://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_Instance_Detection_CVPR_2017_paper.html;
  Tang et al., PCL, TPAMI/arXiv `1807.03342`,
  https://arxiv.org/abs/1807.03342; Wan et al., C-MIL, CVPR 2019,
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wan_C-MIL_Continuation_Multiple_Instance_Learning_for_Weakly_Supervised_Object_Detection_CVPR_2019_paper.html;
  và Choe et al., *Evaluating Weakly Supervised Object Localization Methods
  Right*, CVPR 2020,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html.
  OICR/PCL/C-MIL ủng hộ refinement theo instance/cluster nhưng consensus âm và
  miss rank sâu cấm neo mù vào top-1/overlap; BAS được giữ vì trực tiếp thêm bản
  đồ class-aware, còn mọi choice vẫn phải freeze trước evaluator.

### B2 rich-gallery BAS residual static transport readiness (2026-08-01)

- Thiết kế kế nhiệm được ghi tại
  `BAS_RICH_GALLERY_RESIDUAL_B2_DESIGN.md`, canonical-LF SHA-256
  `b5506df5ba2f66c5f05a6b9f5bed9d287367146e9d82032b794384b4c3adfb7d`.
  B2 kế thừa đúng hai phần đã terminal tốt hơn ở
  `EXP-20260801-codex-rich-gallery-g1-fusion-bottleneck-v3`: immutable rich
  proposal union và fixed G1/upstream equal percentile-rank control
  `0.28872949`. Nó không adopt G2/consensus âm và không tạo lại G0/G1/G2.
- Primitive độc lập `project/models/rich_gallery_bas_residual.py` SHA-256
  `9639c47f3f0f8aeb235f65e800adec0100612a9b57b003195daf6b47a973d582`
  tái hiện exact average-tie percentile rank/tie-break của collaborator, align
  kept candidate index/upstream/source với physical payload và chỉ thêm một
  arm duy nhất: mean rank G1+upstream+BAS. Source identity được audit provenance
  nhưng không được dùng làm score/router. BAS coverage/purity/harmonic vẫn dùng
  primitive image-label-only đã audit ở B1.
- Independent transport auditor
  `project/audit_rich_gallery_stage_a_transport.py` SHA-256
  `300812291a339ac7e096257fd1c328dc44f8c58a1e8802da8716bec1f74a2476`
  không import runner G1/G2 của collaborator. Nó fail nếu có path Stage-B,
  evaluation, per-image GT, ground-truth, polygon/annotation; kiểm tra exact
  freeze/manifest/score-set/candidate hashes, cohort `371/184/187`, alignment
  của từng score payload và tái lập 371/371 G1+upstream choice trước khi B2 được
  phép dùng transport.
- Input BAS được cố định một lần ở 448 px, không có 224/320 arm hoặc resolution
  sweep. Lý do là bằng chứng BTXRD trước đó cho thấy 448 cải thiện small lesion,
  classifier448 đang có source-oracle small tốt nhất, còn output-stride-8 tại
  224 sẽ làm mất thêm tế bào kích hoạt của tổn thương rất nhỏ. Recipe BAS còn
  lại giữ official-style 100 epoch/final checkpoint/T4x2/image-label-only.
- Readiness JSON
  `artifacts/research_protocols/rich_gallery_bas_residual_b2_transport_readiness.json`
  SHA-256
  `817968984aac300a97fdd3b83d6b94938bc1c108c562d796d3801a0eef4afde2`
  khóa collaborator commit `d849155...`, G1 checkpoint
  `634e1200...1bfe8c`, G2 Stage-A freeze `78970d41...cd167`, G2 protocol
  `6dd6c66e...01be`, split `85511ee1...3c8c`, ImageNet weight
  `11ad3fa6...9ca` và 7 canonical source hash. Các binding chưa biết được liệt
  kê fail-closed, không đoán hash.
- Focused primitive+auditor+BAS regression pass `23/23` trong 3.26 giây; full
  repository pass `530/530` trong 22.82 giây trên pinned
  `btxrd-pseudomask` Python 3.9 với documented local strict-zip diagnostic
  shim; `py_compile`, Ruff, JSON parse, source-hash reproduction và
  `git diff --check` pass. Lần focused đầu tiên dùng nhầm system Miniconda thiếu
  NumPy/Ruff nên dừng ở collection trước data access; đây là environment error,
  không phải scientific result.
- Một read-only `kaggle kernels files
  wanwin/btxrd-rich-gallery-g2-selector-pair` nhận HTTP 403 Forbidden. Không file
  nào được list/download, không Stage-B/GT/test được mở và không status poll hay
  job được tạo. Vì thế exact GT-blind transport vẫn là boundary còn thiếu:
  `prediction_freeze.json`, `stage_a_selection_manifest.csv`, 371
  `stage_a_scores`, validation candidate manifest/summary/payloads và một
  inventory SHA-256 không chứa evaluation/annotation/test.
- Nguồn bổ sung đã đọc nhưng chưa adopt là Xu et al., *CREAM: Weakly
  Supervised Object Localization via Class RE-Activation Mapping*, CVPR 2022,
  https://openaccess.thecvf.com/content/CVPR2022/html/Xu_CREAM_Weakly_Supervised_Object_Localization_via_Class_RE-Activation_Mapping_CVPR_2022_paper.html.
  Foreground/background EM re-activation có thể tăng object extent, nhưng bị
  defer vì consensus vừa chứng minh stable anatomy làm over-segment small
  lesion; không combine trước khi BAS riêng có terminal positive result.
- Đây chưa phải claim `ĐANG LÀM`: chưa mở real radiograph/candidate transport,
  chưa fit, chưa prediction, chưa validation segmentation GT, chưa consumer và
  chưa BTXRD test. Khi transport GT-blind xuất hiện, phải fetch/read lại hai log,
  audit transport, freeze exact protocol/source rồi mới đăng ký/push claim trước
  launch Kaggle.

### Đồng bộ top-10 relational failure và hiệu chính vai trò B2 (2026-08-01)

- Sau pre-push fetch, nhánh cộng tác tiến từ `d849155...` lên
  `8b1a38d459e3b4681f7ef5722451cc38deb7d67f`. Toàn bộ log mới,
  `RICH_GALLERY_TOP10_RELATIONAL_FAILURE_DOSSIER.md` và protocol
  `rich_gallery_top10_relational_diagnostic_v1` đã được đọc. Đây là terminal
  audited result mới, không phải code/readiness chưa chạy.
- Fixed top-10 cross-source relational product đạt Dice/IoU
  `0.28564683/0.21473672`, subgroup
  `0.12276538/0.46376513/0.42377677`, 45 misses; thấp hơn baseline
  `0.28872949` overall `-0.00308265`, CI95 `[-0.024322,0.016844]`. Small giảm
  `-0.03495792` với CI95 strictly negative `[-0.063430,-0.011129]`, dù medium,
  large và miss-count tốt hơn. Rule đổi 123/184 choice, recover 12 miss nhưng
  làm mất 8 hit; positive Dice mass `+5.66698` nhỏ hơn loss `-6.23419`.
- Root cause tái lập consensus failure ở neighborhood hẹp: median selected/GT
  area tăng `2.045 -> 2.769` overall và `14.603 -> 27.094` small, trong khi
  tăng extent lại có ích cho large `0.382 -> 0.666`. Relation support tương quan
  `0.4366` với selected Dice nhưng gần bằng 0 (`-0.00016`) với paired
  improvement, nên không có label-free confidence threshold hợp lệ để router.
  Protocol/freeze/Stage-B summary/per-image SHA-256 lần lượt là
  `6cb03b67...3541 / c208be23...8597 / 29653612...49aa /
  e5e97f36...2e70`; test vẫn khóa.
- Quyết định được adopt là **retire mọi post-hoc geometric reranker** và không
  sweep top-K/IoU/weight/source/resolution/morphology. B2 không dùng relation,
  consensus hoặc top-k nên không trùng/falsified bởi experiment này. Tuy nhiên
  vai trò được hiệu chính trước claim: B2 chỉ là matched independent semantic
  descriptor control cho BAS, chưa phải zero-initialized baseline-preserving
  learned relational residual. Relation chỉ có thể được xem xét sau nếu B2
  riêng terminal positive; B2 fail thì không combine relation để rescue.
- Design sau addendum có canonical-LF SHA-256
  `562f001cd9f8db1ffab7d97a044d49c943d1f0b78e94e73acf16a6472bb8114d`;
  readiness JSON mới SHA-256
  `8efe1858b62bbbd943cd770d8d87fa6db34ee30e2c8e92671c9443cb44de5e7c`
  và ghi cả commit/result top-10. Seven source hashes, JSON parse và
  `git diff --check` pass. Scientific primitive/auditor bytes không đổi; không
  real-data access, fit, prediction, GT, consumer, test hay Kaggle launch.

### Tin cậy pipeline cộng tác và B2 semantic execution closure (2026-08-01)

- Theo chỉ đạo người dùng, kết quả terminal của workstream cộng tác được tin cậy
  và kế thừa, không chạy lại để xác nhận. Pipeline nguồn chính xác là rich
  proposal union + equal percentile-rank G1/upstream tại
  `0.2887294867/0.1577232964/0.4352293348/0.3868735327`, oracle
  `0.52829833/0.33187635/0.73025092/0.74624721`. B2 không train/evaluate lại
  G0/G1/G2/top-10; việc tái lập 371 control choice từ Stage-A transport chỉ là
  kiểm tra byte/schema/provenance. Câu hỏi khoa học mới duy nhất là BAS semantic
  arm trừ fixed control này.
- Phân rã bottleneck được giữ nguyên từ terminal dossier: selector regret
  `0.239569`, trong đó `70.29%` nằm trong selected source; oracle top-10 đã
  `0.399326` nhưng 49 complete miss có oracle-rank median `95`. Small có median
  selected/GT area `14.603` trong khi large là `0.382`. Top-10 relation đổi
  123/184 choice, recover 12 miss nhưng mất 8 hit và làm small Dice giảm
  `-0.03495792` với CI95 strictly negative. Vì vậy không proposal regeneration,
  geometric reranker, top-k/IoU/area/source/resolution/morphology sweep; tín hiệu
  mới phải nhận diện tumor, score cả deep rank và không đồng nhất anatomy với
  foreground.
- B2 static producer `project/run_rich_gallery_bas_semantic_b2.py` canonical-LF
  SHA-256 `03457dd079b388c454a436dd2162f1ce56f3638592c13fa56bddd302c48efda5`
  nay khóa một ResNet-50 BAS 448px/100 epoch/image-label-only/T4x2, exact
  ImageNet weight `11ad3fa6...9ca`, fixed control và duy nhất một semantic arm.
  Nó yêu cầu classification + complementarity gate GT-blind, lưu activation/
  candidate-score evidence và chỉ materialize 742 binary prediction file sau
  operational pass. Constant semantic rank fail-closed như redundant thay vì
  gây runtime error.
- Independent no-GT physical auditor
  `project/audit_rich_gallery_bas_semantic_b2_output.py` SHA-256
  `c2ddac26cfdf4ccc6c762e4b7484cc2ccf064366490d3b3fb3ba9b446fd872a2`
  không import producer và không nhận dataset/GT path. Nó tái tính BAS evidence,
  ranks, 371 control/semantic choice, 742 physical mask, exact checkpoint/
  history/gate/T4x2/source/protocol/input hashes trước khi evaluator được phép
  chạy.
- Post-freeze evaluator
  `project/evaluate_rich_gallery_bas_semantic_b2.py` SHA-256
  `98bf92fb186f5682e17f1bfb052a202d7a5a8b34abc1c03cc31cfb6ced39dfc7`
  chỉ import segmentation dataset sau pair freeze và no-GT audit pass. Nó dùng
  trusted `0.28872949` như integrity anchor, không như experiment mới; báo paired
  group-bootstrap 10,000, subgroup, miss recover/loss, positive/negative Dice
  mass, source transition và selected/GT-area ratio. Auditor hậu-GT độc lập
  `project/audit_rich_gallery_bas_semantic_b2_evaluation.py` SHA-256
  `3c0ada81f602b1c71e79206406f538cd21e20036773141fee6a389fc085282b7`
  không import producer/evaluator và tái tính raw physical-mask metrics/gate.
- B2 design/readiness mới có SHA-256
  `4775f69c12bf6269823e01c47aaac9969f205467056b7904c5b858ea45c30969` /
  `aefa5fb44bbe2cb91ccc7023da0d22a0ac29f4b8a280cac60251ca5d5d76f4bd`.
  Focused BAS/rich-gallery/producer/two-auditor/evaluator suite pass `38/38`;
  full repository regression pass `548/548` trong 21.53 giây bằng pinned
  Python-3.9 environment và documented strict-zip diagnostic shim;
  `py_compile`, Ruff, JSON parse và `git diff --check` pass. Đây vẫn là static preparation:
  chưa mở real transport/radiograph, chưa claim/fit/prediction/validation GT,
  chưa consumer/test và không heavy compute local. Boundary còn thiếu là exact
  GT-blind collaborator Stage-A/candidate transport; Kaggle access trước đó trả
  HTTP 403 nên không được phép tự tái tạo pipeline đã tin cậy.

### Deep-search successor nếu BAS còn bị anatomy false positive (2026-08-01)

- Primary evidence mới được đọc: Chen et al., *FPR: False Positive
  Rectification for Weakly Supervised Semantic Segmentation*, ICCV 2023,
  https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html;
  official repo https://github.com/mt-cly/FPR và exact training implementation
  https://raw.githubusercontent.com/mt-cly/FPR/master/step/train_fpr.py. FPR dùng
  activation của class vắng mặt để thu co-occurring-background prototype, sau
  đó region contrast/pixel rectification suppress pixel gần negative prototype.
  Với BTXRD binary, train image-label-normal là absent-tumor cohort trực tiếp,
  nên high tumor activation trên normal image có thể mô hình hóa bone/anatomy
  false positive mà relation vừa chứng minh.
- Không adopt FPR như cải tiến đã chứng minh trên BTXRD. Official implementation
  có validation-mIoU model selection/threshold loop nên phần đó bị cấm chuyển
  giao. Hypothesis có điều kiện được ghi ở
  `RICH_GALLERY_FALSE_POSITIVE_RECTIFICATION_B3_DESIGN.md`, canonical-LF SHA-256
  `5f65f481cfb97d6dfaf96c1f3565dc0a9652308b07fbbe2fe39181b13aa2e0e3`.
  B3 chỉ được xem xét sau terminal audited B2 nếu BAS thực sự complementary
  nhưng negative mass còn gắn với anatomy activation: giữ exact B2 checkpoint,
  dùng train-only positive/absent-tumor regional prototypes và FPR closer-
  positive pixel retain rule, rồi so rectified-BAS với raw B2 trong matched
  pair. Không relation/consensus/count/source/size router hay sweep.
- B3 hiện chỉ là successor hypothesis: không claim/protocol/code/fit/prediction/
  GT/consumer/test/compute. Nếu B2 không cho bằng chứng semantic hữu ích hoặc B3
  sau này fail, family này phải dừng thay vì rescue post-hoc. Đây giữ đúng quy
  tắc chỉ học/adopt hiệu năng khi có terminal audited BTXRD result tốt hơn.

### Đồng bộ collaborator `013244a` và tránh duplicate BAS resolution (2026-08-01)

- Pre-push fetch phát hiện nhánh cộng tác tăng từ `8b1a38d...` lên
  `013244aded5aa1b154d2f433de9b4bc7e005d1e7`. Delta gồm mandatory
  failure-analysis contract và một BAS-B1 static design/runner/evaluator 224px.
  Không có claim `ĐANG LÀM`, real fit, frozen prediction, validation metric hay
  terminal improvement mới; vì vậy không adopt code/224px như bằng chứng hiệu
  năng và không chạy competing BAS experiment.
- Hypothesis BAS-B1 trùng family với central B2 đã chuẩn bị trước ở commit
  `d053bcf`; khác biệt chính là 224px so với fixed 448px. Chạy cả hai sẽ thành
  resolution sweep chưa có kết quả tốt hơn làm căn cứ. Coordination decision:
  chỉ central B2 448px được giữ làm planned experiment vì prior BTXRD evidence
  đã chỉ ra 448 cải thiện small-lesion representation và classifier448 có
  small-source oracle cao nhất. Collaborator 224px design chỉ là static evidence
  và không được launch song song; phải fetch/read lại log trước claim.
- Phần được kế thừa là **quy trình phân tích failure**, không phải performance
  technique: metric proxy không được che endpoint Dice, và mọi rejected arm phải
  giữ per-image precision/recall/extent, gallery/eligible/selected-source oracle,
  truncation/wrong-source/within-source regret, oracle-rank depth,
  top-1/3/5/10/20/50 restricted oracle, recoverable misses, rank correlation,
  source/hit transition và signed Dice mass trước GPU successor tiếp theo.
- B2 được hiệu chính trước claim/protocol/real data: classification và
  complementarity vẫn là GT-blind diagnostics nhưng không còn chặn việc
  materialize pair. Nếu transport/runtime/cohort/finiteness pass, 742 prediction
  file luôn freeze trước GT để actual Dice/failure dossier có thể được audit;
  diagnostics yếu vẫn khóa consumer/adoption. Evaluator và independent
  post-GT auditor nay tự tái lập complete rich-gallery candidate quality và toàn
  bộ decomposition trên, trong khi trusted `0.28872949` vẫn chỉ là integrity
  anchor. Đây là protocol/readiness correction, chưa có scientific run/result.
- Regret audit được siết thêm trước claim: ba thành phần truncation/wrong-source/
  within-source phải không âm và cộng đúng bằng full-gallery oracle trừ selected
  Dice cho từng image/arm; cả evaluator lẫn auditor độc lập đều fail-closed.
  Evaluation summary nay bind cả candidate manifest lẫn pseudo-manifest vật lý.
  Test evaluator tương ứng SHA-256
  `bccdaf6f6b6610c9aa7e386a91206f70ab2ebb3209b319cc275874c458a8f56c`.
- Một lần chạy Ruff trên **toàn bộ** legacy `project/` và `tests/` trả `1,038`
  lỗi lint đã tồn tại ngoài scope B2, chủ yếu `E402` do pattern chèn `sys.path`
  của repository. Không dùng auto-fix và không sửa code người dùng/legacy trong
  active experiment. Ruff giới hạn đúng 8 source/test B2 pass; đây là technical
  lint-debt observation, không phải scientific error/result.

### B2 GT-blind collaborator transport packager (2026-08-01)

- Sau khi static B2 được push ở commit `adf24fd78a1e24821b676ca674df8a2de8cb29b7`,
  đã kiểm tra filesystem ignored, Git tree cộng tác và một bounded Kaggle dataset
  search `btxrd rich gallery`; không nơi nào có physical rich-gallery Stage-A
  transport, và search trả `No datasets found`. Private kernel `wanwin` vẫn
  không đọc được bằng credential `itsthang333`, nên B2 **chưa đăng ký claim và
  chưa launch Kaggle**; không tự tái tạo G0/G1/G2 để lách provenance boundary.
- Packager tĩnh `project/package_rich_gallery_stage_a_transport.py` canonical-LF
  SHA-256 `2e6bbf3b41a7da7f5dd77782e98a09b42f8442ccc29e6a5c44e3b21a326741d6`
  tạo allow-list transport gồm exact Stage-A freeze/selection/371 score payload,
  candidate manifest/summary/371 physical gallery payload, rồi sinh inventory
  bytes+SHA và transport freeze. Nó kiểm tra exact source/protocol/split/G1/
  candidate hashes, cohort `371/184/187`, score-set và mọi NPZ key; path/key có
  Stage-B/evaluation/per-image/GT/polygon/test hoặc object array đều bị reject.
  Nó không mở radiograph, segmentation GT, consumer hay test.
- Ba test mới canonical-LF SHA-256
  `4acb5ecf8aed8ce8050020dd4cf378ce86861f7f0456ba425b7cb62b8d9cf61b`
  kiểm tra path escape, forbidden token, NPZ object/GT key và deterministic
  inventory. Lần chạy đầu có `2` lỗi assertion tĩnh: tokenizer chưa giữ compound
  stem `ground_truth`, và expected byte count của literal `{}\n` ghi nhầm `3`
  thay vì `4`. Sửa tokenizer giữ cả compound stem và hiệu chính assertion; sau
  đó focused packager+transport audit pass `7/7`, scoped Ruff/py_compile/
  `git diff --check` pass, full repository pass `551/551` trong 20.46 giây.
- Yêu cầu handoff cụ thể nằm tại
  `artifacts/research_handoffs/rich_gallery_b2_gt_blind_transport_request_20260801.json`,
  SHA-256 `852cd562fd1dac7d890dd7923788593fb994f94b1ebc1fdfef89b362d41adf87`.
  Cộng tác viên chỉ cần chạy packager trên output/input vật lý của họ rồi upload
  directory sinh ra dưới dạng byte-preserving artifact mà `itsthang333` đọc
  được. Các giá trị source commit/candidate hashes phải lấy exact từ freeze,
  không đoán. Đây chỉ là technical handoff, chưa có experiment/prediction/metric.

### B2 external transport blocker audit (2026-08-01)

- Sau khi packager/handoff commit `71696e7e664abb9c700db6fe744fbffc6a91bb81`
  đã push central, một fetch mới xác nhận central/HEAD cùng ở commit này và nhánh
  cộng tác vẫn ở `013244aded5aa1b154d2f433de9b4bc7e005d1e7`; không có transport
  commit, claim hay terminal result mới.
- Một bounded public Kaggle dataset query duy nhất
  `kaggle datasets list --user wanwin --search btxrd --sort-by updated` chỉ trả
  `wanwin/data-btxrd` cập nhật `2026-07-18`, size `843,837,713` bytes. Không có
  Stage-A/rich-gallery transport công khai; không download/mở dataset ảnh gốc.
  Không lặp lại private-kernel output request đã biết trả HTTP 403 và không tạo
  poll/monitor/job.
- B2 execution hiện bị chặn bởi đúng một external boundary: cộng tác viên phải
  package và cấp quyền đọc byte-preserving artifact theo handoff SHA
  `852cd562...f87`. Mọi static source/auditor/evaluator/packager đã hoàn tất;
  chạy tiếp không có artifact chỉ có thể bằng cách regenerate G0/G1/G2 hoặc bỏ
  provenance audit, đều bị cấm theo yêu cầu người dùng. Vì vậy chưa đăng ký
  `ĐANG LÀM`, chưa launch Kaggle, chưa prediction/validation GT/consumer/test.

### Đồng bộ `d4e0a5f` và B4 class-contrast BAS static readiness (2026-08-01)

- Đã fetch central `793b8797185ac99ca7d7dac49d32c8e6db6e09b7` và nhánh cộng
  tác mới `d4e0a5fd13e132e0bc8861437bacfebb4115b309`, đọc toàn bộ delta log,
  design BAS-B2 và dossier identifiability mới. Theo chỉ đạo người dùng, kết
  quả/pipeline cộng tác được **tin cậy trực tiếp**, không audit lại và không
  list/download/truy cập output hay Kaggle của cộng tác viên. Không có terminal
  BAS metric mới trong delta; kết quả tốt được kế thừa vẫn là rich-gallery G1 +
  upstream equal percentile-rank Dice overall/small/medium/large
  `0.28872949/0.15772330/0.43522933/0.38687353`.
- Chẩn đoán mới được adopt như evidence, không như selector promotable: trong
  eligible-selector regret `0.23917254`, score-dominance identifiability chiếm
  `0.13733820` (`57.42%`); `154/184` eligible oracle vắng khỏi two-score Pareto
  frontier và `104/184` mất ít nhất `0.05` Dice do một mask sai không kém hơn ở
  cả G1 lẫn upstream. Dense shared-alpha hậu nghiệm tốt nhất chỉ
  `0.29147261` (`+0.00274313`) và làm large giảm, nên weight/monotone/nonlinear
  two-score tuning bị loại. Trong severe dominance, `30/31` extreme over-extent
  là small còn `25/26` under-extent là medium/large; area expansion, source
  forcing và một global extent prior đều bị loại.
- Collaborator BAS-B2 hiện là static readiness/memory preflight, chưa có claim
  `ĐANG LÀM`, real-data fit, prediction hay terminal Dice. Để không chạy nguyên
  implementation đó, successor trung tâm B4 giữ kiến trúc hai-score đã chứng
  minh nhưng thêm observable khác: class-contrast map cố định
  `tumor/(tumor+normal)` từ hai BAS localization map, rồi harmonic coverage/
  purity và equal three-way rank với Geometry-v3 + upstream. ReCAM hỗ trợ
  cross-class competition để giảm pixel ambiguity:
  https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Class_Re-Activation_Maps_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2022_paper.html.
  FPR chứng minh activation của class vắng mặt chứa co-occurring-background cue
  và có thể rectification mà không thêm supervision:
  https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html.
  B4 chỉ dùng parameter-free class competition; full prototype FPR vẫn defer.
- B4 execution dùng exclusively accepted same-gallery cache/baseline và raw
  BTXRD input do `itsthang333` kiểm soát; không dùng rich-gallery physical
  output/checkpoint/prediction của cộng tác viên. Hai arm cố định là
  `geometry_v3_plus_upstream_equal_rank` và
  `geometry_v3_plus_upstream_plus_class_contrast_bas`; 448 px/100 epoch/batch
  32/T4x2, không weight/threshold/source/subgroup/resolution sweep. Proxy chỉ là
  diagnostics; sau technical/cohort/finiteness pass vẫn phải freeze cả hai arm
  trước GT để đo actual Dice.
- Static source tách hẳn reusable core mới, giữ nguyên mọi byte/hashes của B1
  lịch sử. Design/core runner/core auditor/B4 runner/B4 auditor/test SHA-256 lần
  lượt là `895a76cc...bcff5 / a9b69da1...5b2dc /
  82112c77...b83a4d / b916ec79...eb998 / dcd570dc...529962 /
  61999e4d...95a98`. Independent no-GT auditor không import producer hay
  segmentation dataset, tái lập class-contrast identity, ranks, score payload,
  choices, maps và pair freeze. Focused suite pass `17/17`; full repository lần
  đầu bằng Python 3.9 pass `544` và có đúng `12` known `zip(strict=True)` lỗi
  compatibility, sau documented diagnostic shim pass `556/556` trong 19.47 s.
  `py_compile` và `git diff --check` pass.
- Static-readiness artifact
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_static_readiness.json`
  SHA-256 `b93474fee39c2a49c57fcea9d7de12ca18c92667b3eca4b22c73804bb41f31c6`.
  Đây chưa phải claim: chưa mở real data/cache, chưa train/prediction/GT/
  consumer/test và chưa launch Kaggle. Trước claim phải freeze protocol/source,
  wrapper/binding/evaluator/decision, fetch lại hai branch, kiểm tra collision,
  rồi đăng ký/push `ĐANG LÀM` trung tâm.

### B4 class-contrast BAS protocol freeze (2026-08-01)

- Scientific source đã được đóng băng ở commit
  `d8f7a650822b52a6b4f0c1a0253683f51ce8e2c2`; protocol
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_v1.json`
  SHA-256 `814f5ca32c080dea4d39d64fed716c5563de524743016055b10217c85d1ffdf1`,
  trạng thái `FROZEN_PRECLAIM_NO_BINDING_NO_LAUNCH`. Kiểm tra độc lập tại
  thời điểm freeze xác nhận toàn bộ 25 source/test SHA-256 trong protocol khớp,
  collaborator output access là `false`, GT-blind diagnostics không được chặn
  việc materialize/freeze pair, và không tồn tại weight/threshold alternative.
- Input/provenance cố định gồm split CRLF SHA-256 `85511ee1...3c8c`, selector-cache
  freeze `2f6290cd...e4f2c`, square-corrected baseline checkpoint
  `58b82642...e1069`, candidate zip `426fbe9c...2687a`, ImageNet ResNet50
  `11ad3fa6...9ca` và comparator bootstrap 10,000 lần seed `20261201`.
  Hai arm, công thức class contrast, 448 px/100 epoch/batch 32 và T4x2 giữ nguyên;
  actual frozen-pair Dice là endpoint chính.
- Lần test lại trước freeze bằng Python mặc định và bundled runtime dừng ở
  collection vì môi trường đó không có `numpy`/`pytest`; đây là local dependency
  boundary trước data, không phải scientific run. Conda Python 3.9 có dependency
  chạy `25` test pass và đúng `1` known compatibility failure tại
  `zip(..., strict=True)` (Python 3.10+ API); `py_compile` và `git diff --check`
  vẫn pass. Full suite với documented compatibility shim đã pass `556/556` ở
  static-readiness commit nên không sửa scientific source vì lỗi runtime cục bộ.
- Vẫn chưa có claim/launch/data/prediction/GT/consumer/test. Bước kế tiếp là tạo
  wrapper fail-closed, execution binding và post-freeze evaluation readiness;
  chỉ sau khi các audit tĩnh pass và fetch/collision check mới đăng ký
  `EXP-20260801-codex-b4-same-gallery-bas-semantic-v1` là `ĐANG LÀM` rồi launch.

### B4 prelaunch validation-GT boundary correction (2026-08-01)

- Static wrapper review phát hiện shared `BTXRDClassificationDataset` gọi loader
  integrity chung; loader này hash file `Annotations/*.json` khi manifest có
  `annotation_sha256`. Mask không đi vào loss, nhưng đọc/hash `184` validation
  annotation trước prediction freeze vẫn vi phạm protocol. Phát hiện xảy ra
  trước wrapper/claim/data/Kaggle, nên không có leakage khoa học thực tế.
- Không sửa loader lịch sử và không tái sử dụng implementation BAS của cộng tác
  viên. B4 nay có adapter riêng `project/datasets/btxrd_image_label_only.py`:
  chỉ đọc exact split manifest, nhãn ảnh train/val và radiograph đã xác minh
  SHA-256; không resolve/open/hash annotation và từ chối split `test`. B4 runner
  bind adapter này trước khi gọi reusable training core; architecture, loss,
  gallery, hai arm và mọi hyperparameter khoa học không đổi.
- Hai regression test mới chứng minh adapter đọc được radiograph/nhãn ảnh khi
  thư mục `Annotations` không tồn tại và fail-closed với split `test`. Focused
  safety+B4 suite pass `16/16`; `py_compile` và `git diff --check` pass. Đây là
  implementation/safety correction tĩnh, chưa mở real data/cache/image/GT,
  chưa train/prediction/consumer/test và không heavy compute local.
- Protocol B4 v1 SHA `814f5ca3...df1` được giữ nguyên như bằng chứng preclaim
  nhưng sẽ bị supersede trước launch vì source commit của nó chưa có adapter
  an toàn. Phải commit/push source correction, phát hành protocol v2 đóng băng
  lại exact source hashes, rồi mới tiếp tục wrapper/binding/claim.

### B4 image-label-only protocol v2 freeze (2026-08-02)

- Safety-corrected scientific source đã được commit/push ở
  `69b9af26c3de12ac10550b9262b2ff8f5e4424e8`. Protocol kế nhiệm
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_v2.json`
  SHA-256 `958c414863c64f5160b4882feda44451add008fd000ca1120164925a0b2d298d`,
  trạng thái `FROZEN_PRECLAIM_NO_BINDING_NO_LAUNCH`, supersede v1 chỉ vì
  validation-GT input boundary. Scientific variable, exact two-arm pair,
  training recipe, input artifacts, endpoint và decision gate không đổi.
- Independent source-closure check từ exact Git commit xác nhận đủ `27/27`
  canonical-LF source/test SHA-256. V2 bind rõ adapter chỉ cho train/val,
  binary `tumor` image label, `annotation_paths_resolved=false`,
  `annotation_bytes_opened_or_hashed=false`, và `test_images_read=0`.
- Đây vẫn là protocol freeze trước claim: không real data/cache/image/GT,
  fit/prediction/evaluation/consumer/test/Kaggle. Tiếp tục chuẩn bị wrapper,
  binder, output auditor và post-freeze evaluator; claim chỉ được đăng ký/push
  sau khi toàn bộ static readiness và collision check pass.

### B4 fail-closed Kaggle wrapper/binder readiness (2026-08-02)

- Wrapper template `project/kaggle_wrappers/run_same_gallery_class_contrast_bas_b4_v1.py`
  canonical-LF SHA-256 `0a0293500de2128ab80c64b220f37b1d1eb8e675f7675b14dbcec94d37ea9eb6`
  và one-shot binder `project/bind_same_gallery_class_contrast_bas_b4_wrapper.py`
  SHA-256 `eaf8982644265f70e6b7a666017c3828b211b66015a88c4a17bf1d2736432bbc`.
  Binder chỉ thay đúng kernel version, ready flag và execution checkout; inverse
  reconstruction phải trả exact template, đồng thời re-hash protocol, auditor
  và toàn bộ source inventory từ checkout.
- Wrapper fail trước scientific input nếu không đúng source ancestor/protocol,
  exact T4x2 real-convolution guard, frozen split, cache, baseline transport,
  validation-candidate archive và ImageNet checkpoint. Dataset adapter chỉ đọc
  train/val radiograph + binary image label; không có annotation/test/evaluator
  path. Producer luôn freeze hai arm sau technical pass, independent no-GT
  auditor chạy sau pair freeze, wrapper sau cùng xác minh vật lý `742` maps,
  `742` score payloads và `371` activation payloads.
- Test wrapper SHA-256 `500e675c848e8702da0c309f8387e834e1f84f4115ddaf2110513f535869306b`;
  focused wrapper+safety+B4 suite pass `19/19`, `py_compile` và
  `git diff --check` pass. Static-readiness artifact
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_wrapper_readiness.json`
  SHA-256 `5fa0ab51aa0a2250a42f2475a0a206ae8d6534c1aa27320a1eedb17ebefca8a8`;
  đây vẫn chưa phải claim/binding/launch và chưa có scientific input.

### B4 comparator-seed protocol/readiness correction (2026-08-02)

- Static post-freeze review phát hiện protocol v2 ghi bootstrap seed `20261201`
  nhưng generic comparator frozen ở source commit chỉ chấp nhận `20261101`.
  Nếu không sửa, prediction có thể hợp lệ nhưng evaluation sẽ fail tại argument
  gate. Không có data/prediction/GT nào đã mở; đây là preclaim execution error.
- Protocol v3
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_v3.json`
  SHA-256 `a4fc4f26e184150e90b4c5da83bbf0808ff51c465f5af285109329d10178a6dc`
  supersede v2 và chỉ đổi seed thành `20261101`; model, BAS recipe, gallery,
  two-arm scores, endpoint và gate không đổi. Exact Git source closure vẫn
  `27/27`, collaborator output access vẫn `false`.
- Wrapper/binder đã rebind tĩnh sang protocol v3: template SHA-256
  `8c6c13ddc052784dacef52b136d53c1414d44d09c16ab191012eae97b5c56740`,
  binder SHA-256 `726e6452d3b7219d1852a8752f98bcc3c5f09526db94eb5a77619c4d597967a2`.
  Focused suite trên Python 3.9 pass `27`, có đúng `1` known
  `zip(strict=True)` compatibility failure; documented diagnostic shim pass
  `28/28`. Kaggle Python 3.10+ phải pass native. Readiness-v2 artifact được ghi
  ở `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_wrapper_readiness_v2.json`,
  SHA-256 `7750c31816a32b633e8e8cadb0bf2c382ad8f36bb7b14c24c9f0dbfc85847c45`.
- Chưa claim/bind/launch. Sau commit/push này, binder phải chạy trên exact clean
  checkout, sau đó fetch hai branch/collision check và đăng ký `ĐANG LÀM` trước
  khi `kaggle kernels push`.

### B4 kernel-v1 one-shot binding (2026-08-02)

- Lần gọi binder đầu tiên truyền nhầm full checkout SHA suy đoán
  `a1c3fef2499fd5173d952ab8eb53d7e44372cf92` thay vì exact HEAD. Binder dừng
  tại `git show` protocol trước khi tạo bound wrapper/binding, trước mọi input,
  data hay Kaggle; không có scientific result. Exact target được đọc trực tiếp
  rồi binder chạy lại trên clean checkout
  `a1c3fefeeca74d961d6a585409c553686a15610c`.
- One-shot binding pass: protocol/source/auditor closure `27/27`, source commit
  `69b9af2...` là ancestor, đúng `3` replacement và inverse reconstruction exact.
  Launch binding
  `artifacts/research_protocols/same_gallery_class_contrast_bas_b4_kernel_v1_launch_binding.json`
  SHA-256 `8be3a815bf44ebc43d431b7f9db308c3545b2b7baf38bf7cf6262b988a6f264a`;
  bound wrapper canonical-LF SHA-256
  `53359efd5eddc8ebb173531e202d763608c22c7d892e5d1d7dae8546b75f764d`.
- Kaggle metadata SHA-256
  `d156f47084b987946f10a45b1330718e741dae17d981ec9eed54a55f21cc5890`
  bind raw dataset + accepted baseline transport dataset + accepted selector
  cache kernel, private internet-enabled exact T4. Bound `py_compile`, metadata/
  binding JSON parse và independent inverse check đều pass.
- Vẫn chưa claim/launch/data/prediction/GT/consumer/test. Binding phải được
  commit/push, sau đó fetch central + collaborator, đọc collision claims và chỉ
  đăng ký `ĐANG LÀM` nếu scope vẫn unique.

### Đồng bộ collaborator `f30e37b` và supersede B4 trước claim/launch (2026-08-02)

- Đã fetch lại `origin/research-wsss-improvement` và
  `origin/codex/research-sync-20260731`; central/HEAD cùng ở
  `2deb5df1b58faaad0502105ceb10017cbdacc10c`, collaborator mới nhất ở
  `f30e37b063682c4e79c50b58c53dd3fbadd64478`. Đã đọc toàn bộ log trung tâm,
  toàn bộ delta collaborator và hai dossier mới. Theo chỉ đạo người dùng,
  terminal result của collaborator được **tin cậy trực tiếp**; central không
  list/download/audit lại output và không truy cập Kaggle/kernel của họ.
  Exact Git-blob SHA-256 của collaborator log, post-BAS dossier và Softplus
  design lần lượt là `901e553b3e79a27eda16d1d35a3b26425b6a4ee125b032ff4d9c8cb76012d7d4`,
  `85dbe34f366a12924d1457b6a941b121b2c4f0024639ce4d918c3d680fec08ba`
  và `9fbf7fa48dc3aa2b4e8f4fb48b3bdff7e54ad9c65c5a8fd896336961858b8107`.
- Terminal `EXP-20260801-codex-rich-gallery-bas-b2-v1` giữ control G1/upstream
  Dice/IoU `0.28872949/0.21683918`, nhưng equal three-way BAS chỉ đạt
  `0.18110635/0.12789378`, subgroup Dice
  `0.02745544/0.30171128/0.50108581`; paired overall CI95 của delta là
  `[-0.140313,-0.057568]`, BAS-only `0.04877652`. Đây không phải cải thiện.
  Exact root cause là terminal ReLU head chết: CE `0.693359375`, accuracy
  `1493/2981=0.50083864`, validation AUROC `0.5`, nondegenerate-map fraction
  `0`, range chỉ `9.70e-7`. BAS score vì thế gần như area rank
  (mean/median within-image Spearman `0.999902/0.999922`) và chọn max-area
  percentile ở mọi tumor; positive Dice mass `+5.8176` bị negative mass
  `-25.6202` áp đảo. Eligible oracle vẫn `0.52790203`, truncation regret chỉ
  `0.00039631`; thiếu observable tumor-specific candidate identity vẫn là
  bottleneck.
- Central B4 class-contrast BAS dùng cùng terminal-ReLU mechanism, nên mọi
  protocol/binding/readiness B4 được giữ làm bằng chứng nhưng trạng thái khoa
  học là **SUPERSEDED BEFORE CLAIM/LAUNCH**. Không đăng ký
  `EXP-20260801-codex-b4-same-gallery-bas-semantic-v1`, không push Kaggle,
  không mở real input/prediction/GT/consumer/test. Class contrast sau một head
  đã chết không thể phục hồi thông tin không tồn tại; launch B4 sẽ lặp đúng
  error mechanism vừa được chứng minh.
- Collaborator Softplus B2.1 ở `f30e37b` mới là bounded static mechanics probe,
  chưa có terminal Dice hay claim trong log. Central không chạy cạnh tranh và
  không adopt Softplus như performance improvement. Chỉ khi collaborator sau
  này có terminal audited result tốt hơn control thì kỹ thuật đó mới đủ điều
  kiện kế thừa vì hiệu năng.
- Successor phải thay **representation**, không đổi tên một residual cũ: R1
  normal prototypes, R2 local affinity, R3/R4 relation, S1 family balance, S4
  clustering, T1 count control, G2 negative-only/temperature, consensus,
  top-10 relation, classifier deletion/insertion, direct dense RAD-DINO và BAS
  dead-head đều đã có negative evidence. Hướng ưu tiên được mở lại từ transfer
  audit trước đây là frozen SKELEX musculoskeletal foundation descriptor trên
  exact immutable gallery. SKELEX được self-supervised pretrain trên khoảng
  1.2 triệu musculoskeletal radiograph và paper báo cáo bone-tumor transfer;
  đây là semantic source mới, không phải một objective/aggregation sweep.
  Primary sources: https://www.nature.com/articles/s41746-026-02826-9,
  https://arxiv.org/abs/2602.03076 và exact public model revision
  https://huggingface.co/skhoha/SKELEX/tree/368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e.
  Public checkpoint `model.safetensors` có 1,318,230,232 byte và LFS SHA-256
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`;
  config/preprocessor xác nhận ViT-MAE Large, 24 layer/1024 dim, patch 16,
  224 px và ImageNet normalization. License CC-BY-NC-ND-4.0 được giữ rõ;
  không redistribute/modify checkpoint trong repository hoặc output.
- Do người dùng cấm truy cập collaborator output, central không thể score trực
  tiếp physical rich-gallery union `0.28872949`. S5 vì thế là một
  **transferable same-gallery representation ablation** trên gallery/hash mà
  central sở hữu; nếu SKELEX tạo candidate evidence dương, exact scorer mới mới
  đủ căn cứ để chuyển sang rich gallery qua một GT-blind score interface sau
  này. Nó không regenerate/copy G0/G1/G2 và không được quảng bá là trực tiếp
  vượt rich-gallery baseline trước khi có actual frozen-pair Dice.
- Static S5 implementation now follows the exact Hugging Face Transformers
  `v4.50.2` ViT-MAE encoder contract inspected at
  https://github.com/huggingface/transformers/blob/v4.50.2/src/transformers/models/vit_mae/modeling_vit_mae.py:
  `mask_ratio=0` keeps all 196 patches, deterministic increasing `noise`
  preserves patch order, and the encoder still prepends one CLS token which
  must be removed before the 14x14 reshape. Exact SKELEX revision files fetched
  read-only have `config.json` SHA-256
  `b48411f4313c2ee6357586b57d185befac8c7c77cc475bc2188ec4487b1bc6f7`
  and `preprocessor_config.json` SHA-256
  `a250969c94afba52d785a0e08dd36e13aeda97c4dd2b7fd0d24b457288536cea`.
  `SKELEX_MASK_BAG_SELECTOR_S5_DESIGN.md`, the SKELEX descriptor/pooling
  primitives, one-shot runner, independent GT-blind auditor and synthetic
  tests were prepared without opening real scientific input. `py_compile` and
  `git diff --check` pass; local PyTorch tests are unavailable because the host
  environment has no PyTorch, so the fail-closed T4x2 wrapper must run them
  before any real input. S5 remains **UNCLAIMED / NOT LAUNCHED** at this point.

### Đồng bộ collaborator `d4bb653` và đăng ký S5 (2026-08-02)

- Đã fetch `origin/codex/research-sync-20260731` mới nhất tại
  `d4bb653e449adc6b9d78a6aaddbcb3615dd458b0` và đọc toàn bộ delta log từ
  `f30e37b`. Theo chỉ đạo người dùng, chỉ tin terminal statements trong log;
  không list/download/audit output và không truy cập Kaggle của collaborator.
  `EXP-20260802-codex-rich-gallery-bas-b21-softplus-probe-v1` sửa được dead
  classifier ở cấp ảnh (`train accuracy=0.839651`, validation AUROC
  `0.743199`) nhưng không sửa localization: foreground CE vẫn đúng
  `log(2)=0.693147182`, map max trung bình `2.237e-7`, median effective support
  `0.001338`, top-1%-mass `0.931514` và 184/184 argmax ở viền 10%. Vì vậy
  Softplus/BAS B2.1 không được kế thừa như cải thiện hiệu năng.
- Post-freeze oracle-feature-gap diagnostic cho rich gallery chỉ ra oracle bị
  xếp thấp hơn selected trong `84.2%/87.0%/70.7%` ảnh theo
  G1/upstream/SAM-score; median oracle-minus-selected ranks đều âm. Extent
  oracle-minus-selected đổi dấu theo subgroup: small `-0.001299`, medium
  `+0.001538`, large `+0.060391`; source mismatch cũng tăng theo scale.
  Insight chuyển giao là phải thêm **candidate-conditioned positive semantic
  evidence**, không sweep tiếp G1/upstream/SAM/area/source hoặc dùng một global
  extent prior. Điều này củng cố S5: SKELEX thay semantic representation nhưng
  giữ exact candidate/extent và phép rank control.
- Collaborator B2.2 foreground-control mới chỉ là static mechanics design,
  chưa có claim/Dice terminal. S5 không chạy lại BAS và không cạnh tranh B2.2;
  collision check trên cả hai log không thấy claim SKELEX descriptor đang làm.

### EXP-20260802-codex-rich-gallery-bas-b22-final-failure-v1

- **Nguồn đồng bộ:** tin cậy terminal statement mới trong
  `origin/codex/research-sync-20260731@added53690ed94c69d6c39cd619c0aced506bb85`
  theo chỉ đạo người dùng; không truy cập, download hoặc audit output/Kaggle của
  collaborator. Mục này supersede nhận định trước đó rằng B2.2 mới chỉ là static
  design.
- **Mechanics:** B2.2 repaired the B2.1 empty-map optimum, reaching validation
  image AUROC `0.735294` and foreground CE `0.463459`. It nevertheless failed
  the frozen mechanics contract because full-image CE worsened to `0.985558`.
  The generated tumor maps are broad anatomy maps: median activation mean
  `0.496241`, effective support `0.576708`, and 35.28% of cells exceed 0.90.
- **Exact formula:** for fixed class map `C`, B2.2 has
  `dL/dM_i=(1.2-1.5*C_i/mean(C))/N`; its box optimum activates every cell with
  above-average class evidence ratio `>0.8`. The written foreground reference
  `0.5` is an additive constant with no gradient effect. The tumor channel also
  receives no dense-negative supervision on train-normal images.
- **Area-proxy proof:** within each tumor gallery, BAS score versus candidate
  area has mean/median Spearman `0.933174/0.950055`; 81.52% of images exceed
  `0.90`. The mechanism learned common anatomy/extent rather than candidate
  tumor identity.
- **Frozen actual endpoint:** the primary G1+upstream+B2.2 fusion reaches Dice
  `0.19172607`, delta `-0.09700342` with paired CI95
  `[-0.128452,-0.047141]`. Subgroup deltas are
  `-0.129408/-0.117096/+0.152591`: B2.2 helps large lesions by expansion but
  catastrophically magnifies small/medium over-extent. It recovers three old
  misses and creates 27 new misses.
- **No metadata rescue:** a GT-only per-image oracle switch would reach
  `0.322338`, proving limited complementarity. A deterministic group-separated
  ridge using every current label-safe area/border/classifier/activation/source
  observable reaches only `0.287786`, below the immutable `0.288729` baseline.
  The B2.2 benefit cannot be safely routed with existing metadata.
- **Bottleneck update:** gallery truncation remains negligible (`0.000396`).
  B2.2 raises selector regret from `0.239569` to `0.336572`, worsening both
  within-source and cross-source terms. The missing observable is
  candidate-conditioned tumor identity plus signed extent: tumor evidence
  inside the mask, normal/tumor contrast at its boundary, and direct tumor-
  channel negatives on normal candidates.
- **Decision:** retire all BAS variants and sweeps. Preserve G1+upstream fixed
  fusion at Dice `0.2887294867`. The collaborator successor is a bounded
  candidate inside-versus-local-ring residual with train-normal candidate
  negatives, tumor-bag MIL, no global/coordinate bypass, and zero initialization
  on the immutable baseline. S5 không copy/rerun successor đó: S5 giữ gallery
  Geometry-v3 riêng, frozen SKELEX descriptors đã predeclare gồm inside,
  dilated-local-context và signed inside-minus-context, rồi chỉ sửa hai identity
  guard số học trước một correction rerun. Không đổi science sau khi thấy metric.
  Test remains locked.

#### `EXP-20260802-codex-s5-skelex-selector-v1`

- **Owner/time/status:** Codex central workstream; đăng ký
  `2026-08-02T00:44:00+07:00`; **ĐANG LÀM**. Registration commit là commit
  chứa mục này và sẽ được bổ sung exact hash trước launch.
- **Hypothesis/mục tiêu:** trên exact immutable Geometry-v3 gallery mà central
  sở hữu, frozen SKELEX ViT-MAE musculoskeletal proposal descriptor cung cấp
  candidate-conditioned tumor semantics bổ sung cho Geometry-v3/upstream,
  giảm selector regret và tạo corrected Dice cao hơn identity control. Đây là
  transferable same-gallery representation ablation; không tuyên bố score
  trực tiếp physical rich-gallery output `0.2887294867` vì output đó bị khóa
  theo chỉ đạo người dùng.
- **Khác biệt duy nhất:** giữ exact proposal masks/indices, accepted four-column
  metadata, selector recipe 16 epoch và equal percentile-rank aggregation; chỉ
  thay representation branch mới bằng frozen SKELEX layers `[8,16,24]`,
  projection `1024->128` seed 42. Không sweep layer/weight/source/resolution,
  không regenerate gallery, không global extent correction. Exact 128-square
  geometry bridge bảo toàn fractional support trước 14x14 pooling; zero support
  fail-closed, không được drop candidate.
- **Kế thừa:** trusted collaborator controls/diagnostics
  `EXP-20260801-codex-rich-gallery-bas-b2-v1`,
  `EXP-20260802-codex-rich-gallery-oracle-feature-gap-v1`,
  `EXP-20260802-codex-rich-gallery-bas-b21-softplus-probe-v1`; central accepted
  Geometry-v3 selector/cache và same-gallery rank control. Negative evidence
  R1/R2/R3/R4/T1/S1/S3/S4/G2/B2/B2.1 được dùng để loại descriptor/objective
  trùng, không quảng bá như cải thiện.
- **Frozen source/protocol:** scientific source commit
  `61927cc84ef2340768ea37f9686bf8036c81db30`; protocol
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_v1.json` SHA-256
  `036e9d1d52a4ba1ee8e2a51cd19ca4fef597c6c7ad0256e7c729c7888ea24280`.
  SKELEX exact revision `368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`,
  weight SHA-256 `81cd6e9c...d69ef8`, license CC-BY-NC-ND-4.0; checkpoint
  không được redistribute.
- **Input/provenance:** split `85511ee1...3c8c`; train/val `2981/371`;
  train gallery dataset `itsthang333/btxrd-mask-bag-geometry-v3-train-gallery-v1@1`
  manifests `ad3b52d6...58d1`/`5aec58ce...c21`; validation transport
  `itsthang333/btxrd-mask-bag-selector-baseline-v1@1` zip
  `426fbe9c...2687a`, manifests `3e9396f5...3090`/`286d1fce...6320e`;
  selector-cache freeze `2f6290cd...e4f2c`; accepted baseline checkpoint/freeze
  `58b82642...e1069`/`ec346276...e3ec3`. Raw images only từ
  `itsthang333/btxrd-raw`; training target duy nhất là image-level tumor.
- **Compute/run:** đúng một Kaggle kernel version mới
  `itsthang333/btxrd-skelex-mask-bag-selector-s5-v1`, accelerator T4x2. Wrapper
  phải chạy static/synthetic tests trước real input, verify exact public model,
  split/cache/gallery/baseline hashes và descriptor operational gate trước khi
  train selector.
- **Frozen outputs/gate:** hai arms
  `geometry_v3_plus_upstream_equal_rank` và
  `geometry_v3_plus_upstream_plus_skelex_equal_rank`; physical pair freeze +
  independent GT-blind reproduction of descriptor logits/ranks/maps trước GT.
  Sau đó evaluator/comparator cố định (`10,000` bootstrap, seed `20261101`,
  baseline per-image SHA `a26143d0...605f`) mới đọc validation GT. Promote chỉ
  khi overall corrected Dice tăng nghiêm ngặt và không regression subgroup bắt
  buộc; âm thì đóng terminal, không rescue sweep.
- **Safety at registration:** chưa mở real image/candidate, chưa prediction,
  chưa GT, chưa train consumer, chưa test. Collaborator output/Kaggle không
  được truy cập. BTXRD test khóa; heavy compute chỉ T4x2.
- Exact registration commit: `151e8e643fa15fb3c22cda21f155339dff85b192`.
- Fail-closed Kaggle wrapper đã được chuẩn bị ở execution checkout
  `8abb4943a618effa50f065c54f06cf43ab79b910`, sau đó bind version 1 với
  canonical wrapper SHA-256
  `519732f8895981979b71a66d8c3aeeab23ffce00fda023584522b0a0b72e1607`.
  Prelaunch audit
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_kernel_v1_wrapper_audit.json`
  xác nhận test tĩnh/synthetic chạy trước khi resolve BTXRD input, exact public
  SKELEX hashes, T4x2 guard, candidate/cache/baseline provenance, descriptor
  gate trước selector training, pair freeze và independent no-GT audit. Tại
  thời điểm ghi mục này: chưa push Kaggle, chưa real input/prediction/GT/test.
- Sau final fetch/collision check, version 1 đã được push thành công lên
  `itsthang333/btxrd-skelex-mask-bag-selector-s5-v1` lúc
  `2026-08-02T00:50+07:00`. Exact package wrapper SHA-256
  `519732f8895981979b71a66d8c3aeeab23ffce00fda023584522b0a0b72e1607`;
  `kernel-metadata.json` SHA-256
  `33d4a8cb2fba94c1ad2a1742004d26390faae4de989b5fca4adace2a2bd5f107`;
  branch execution binding commit
  `2819b173d0436ff918a30f1dacb29252f1af580a`. Kaggle CLI xác nhận đúng
  `Kernel version 1 successfully pushed`. Không status poll/monitor được tạo ở
  nhịp launch. Experiment vẫn **ĐANG LÀM**; chưa có terminal result,
  prediction-freeze/audit result hay validation GT/Dice. BTXRD test và mọi
  consumer kế nhiệm tiếp tục khóa.

### S5 kernel version 1 — LỖI môi trường trước encoder/training (2026-08-02)

- `EXP-20260802-codex-s5-skelex-selector-v1`, kernel private
  `itsthang333/btxrd-skelex-mask-bag-selector-s5-v1` version 1, đã terminal
  **LỖI** sau `328.4 s` trên đúng `GPU T4 x2`; Kaggle báo `Output 0 B`.
  Direct Logs payload có `7,520` byte/`32` dòng, SHA-256
  `fa3fed30b31c304f86fe3ab442aa8e07b80382cf1bceb658a1b7b5953e5239b0`.
  Do credential Kaggle CLI cục bộ không còn hợp lệ, log của chính kernel S5
  được đọc read-only qua phiên Kaggle đã đăng nhập trong in-app browser; không
  list/download/audit bất kỳ output/kernel nào của collaborator.
- Boundary chính xác: checkout `8abb4943a618effa50f065c54f06cf43ab79b910`,
  ancestry scientific source `61927cc84ef2340768ea37f9686bf8036c81db30`,
  T4x2, public SKELEX snapshot/hash, `py_compile`, `24 passed in 9.43s`, split,
  train/val candidate manifests, accepted baseline, selector-cache freeze và
  cache records đều qua guard. Runner sau đó dừng tại dòng 472, trước
  `ViTMAEForPreTraining.from_pretrained`, với
  `RuntimeError: S5 transformers version mismatch`.
- Root cause là omission đóng gói implementation-only: protocol/source pin
  `transformers==4.50.2`, nhưng wrapper S5 v1 không cài pin này và phụ thuộc
  mutable Kaggle base image. Guard đã fail-closed đúng; log v1 không in ambient
  version nên audit không suy đoán giá trị thực tế. Đây là tái diễn failure mode
  đã được ghi ở mục `2026-07-26 - RAD-DINO dense-MIL probe v1 environment
  error`, vì vậy là regression kỹ thuật có thể tránh, không phải bằng chứng âm
  cho SKELEX hay selector hypothesis.
- Không radiograph nào được mở, không encoder inference/descriptor/training,
  không validation prediction/pair freeze, không validation GT, không consumer
  và không BTXRD test; version 1 không tạo scientific result/Dice. Raw local copy
  được giữ ignored tại
  `tmp/kaggle/skelex_s5_v1_terminal_error_20260802/kernel.log`; copy này thêm một
  terminal LF nên có SHA-256
  `2e52730f45ca041069198fc354062ece8a02583dde3f0dd11fd38a3cee1cbdbf`
  và `7,521` byte. Error audit tracked tại
  `artifacts/kaggle/skelex_mask_bag_selector_s5_v1/kernel_version1_error_audit.json`,
  SHA-256
  `329ad59b715677f7b4aa11c9644234fc5cf7f65dd2da140c119b058c2c629683`.
- Correction được phép chỉ là cài exact `transformers==4.50.2` với no-cache
  trước tests/execution, đặt `TOKENIZERS_PARALLELISM=false`, và giữ nguyên guard.
  Scientific source commit, protocol SHA, SKELEX weights, split/gallery/cache,
  recipe, arms, freeze/audit/evaluator đều bất biến. Theo rule, chưa sửa hoặc
  rerun trước khi bản ghi `LỖI` này được commit và push lên branch điều phối.

### S5 version 2 — correction đã freeze, chờ launch (2026-08-02)

- Sau khi bản ghi lỗi v1 được push ở commit
  `97144cfa8f4b0124c4ac4abeb52951a2a87fdcfe`, correction implementation-only đã
  được chuẩn bị và push code canonical ở commits
  `52d0b8630142a12c2bddd715353045024fc5e557` và
  `ad441e1de1cd4df7fddebb2ff4a9dda03f3e9998`. Trạng thái
  `EXP-20260802-codex-s5-skelex-selector-v1` trở lại **ĐANG LÀM** cho đúng một
  corrected rerun version 2; đây không phải experiment/hypothesis mới.
- Wrapper v2 cài exact `transformers==4.50.2` bằng
  `--disable-pip-version-check --no-input --no-cache-dir`, kiểm tra lại exact
  imported version trước khi tiếp tục và đặt `TOKENIZERS_PARALLELISM=false`.
  T4x2 guard vẫn chạy trước cài đặt; static/synthetic tests nay chạy trước lần
  tải public SKELEX checkpoint `1,318,230,232` byte để các lỗi môi trường/code
  tương lai dừng sớm và không lãng phí network/compute.
- Canonical wrapper được giữ fail-closed/unbound trong Git; packaged Kaggle v2
  chỉ khác ba binding constants, checkout exact
  `ad441e1de1cd4df7fddebb2ff4a9dda03f3e9998`. `py_compile` pass và wrapper tests
  `2 passed in 0.03s`. Bound wrapper SHA-256
  `dbd8cd8cfb613a19ca1d789900f3a1d9f33c1caa8a6b6212e3dee37b3cdbeb04`;
  metadata giữ nguyên SHA-256
  `33d4a8cb2fba94c1ad2a1742004d26390faae4de989b5fca4adace2a2bd5f107`.
  Prelaunch audit
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_kernel_v2_wrapper_audit.json`
  SHA-256
  `259673147bff6f22d2523d06a89c05ba3d6ebb965373d4514f35f8b1b99f5c0b`.
- Scientific source commit `61927cc...`, protocol SHA `036e9d1d...`, public
  SKELEX revision/weights, split, candidate gallery, accepted cache/baseline,
  selector recipe, two arms, prediction-freeze, independent GT-blind audit,
  evaluator/comparator và decision gate đều không đổi. Chưa launch v2, chưa mở
  real input/prediction/GT/consumer/test ở bước correction này; phải push mục
  **ĐANG LÀM** này lên branch điều phối trước Kaggle launch.
- Sau khi correction claim/audit đã được push tại commit
  `e73d2cdb3c3f41c5cdc2747111e13180f94ac9bc`, Kaggle version 2 được launch lúc
  `2026-08-02T01:26+07:00` bằng Save & Run All trên chính private kernel do CLI
  credential cục bộ không còn hợp lệ. Trước Save, editor đã được kiểm tra trực
  quan có đúng `KERNEL_VERSION = 2`, `LAUNCH_BINDING_READY = True`, checkout
  `ad441e1de1cd4df7fddebb2ff4a9dda03f3e9998` và
  `EXPECTED_TRANSFORMERS_VERSION = "4.50.2"`. Kaggle xác nhận
  `Version #2 with GPU T4 x2` và `Running: just now`.
- Đây là một launch duy nhất của corrected wrapper SHA
  `dbd8cd8cfb613a19ca1d789900f3a1d9f33c1caa8a6b6212e3dee37b3cdbeb04`;
  không tạo competing job, không tạo monitor và không status-poll thêm ở nhịp
  launch. Experiment vẫn **ĐANG LÀM**; chưa có output/prediction freeze/audit,
  chưa đọc validation GT, chưa train consumer và BTXRD test tiếp tục khóa.

### S5 v2 — phát hiện static auditor precision defect sau launch (2026-08-02)

- Một bounded status check duy nhất lúc khoảng `01:33+07:00` thấy version 2 vẫn
  `RUNNING` ở `404.1 s`, đúng `GPU T4 x2`, output lúc đó `0 B`. Log xác nhận
  correction môi trường đã hoạt động: ambient `transformers 5.0.0` được thay
  bằng exact `4.50.2`, explicit imported-version assertion pass, sau đó toàn bộ
  suite mở rộng `26 passed in 11.93s`. Không status-poll hay monitor tiếp theo.
- Static pre-mortem sau launch phát hiện một defect hậu kỳ độc lập với khoa học.
  Generator tạo base/upstream/SKELEX percentile ranks bằng `torch.float32`, nên
  correlation lưu trong `gt_blind_diagnostics.json` dùng hai vector float32.
  Independent auditor dùng `_rank` trả `numpy.float64`, chỉ cast control về
  float32 nhưng để SKELEX rank float64, rồi yêu cầu reproduced mean correlation
  khớp ở tolerance phi thực tế `1e-12`.
- Synthetic no-input/no-GT reproduction seed `20261101` trên candidate counts
  `2..81` cho `7,289/7,922` finite single bags vượt `1e-12`, max delta
  `3.2557e-8`. Quan trọng hơn, `100/100` synthetic cohorts 371 ảnh đều vượt
  guard; absolute mean delta min/median/max là
  `9.9179e-12 / 2.8593e-10 / 1.1858e-9`. Vì auditor chạy sau descriptor cache,
  16-epoch selector, prediction pair freeze và physical reproduction, v2 gần
  như chắc chắn sẽ lãng phí toàn bộ compute rồi fail ở bước cuối dù prediction
  bytes đúng.
- Evidence tracked tại
  `artifacts/research_protocols/skelex_s5_v2_postlaunch_rank_precision_audit.json`,
  SHA-256
  `68e8e8d1c972a2415681d79af8d7abf955eaedbe563796054dbfff6ae44c5954`.
  Quyết định fail-early là cancel đúng running version 2 sau khi mục này được
  push, không launch competing job. Correction được phép chỉ ở auditor: cast
  tất cả rank tái tạo về `numpy.float32` trước aggregation/correlation và thêm
  regression test; generator, descriptor, selector, prediction, scientific
  protocol, freeze/GT boundary đều bất biến. Tại thời điểm ghi mục này v2 chưa
  bị cancel và chưa có terminal scientific result; validation GT/consumer/test
  vẫn khóa.

### S5 kernel version 2 — LỖI fractional-mass guard trước training (2026-08-02)

- Trong lúc static-defect record được push, version 2 tự terminal **LỖI** sau
  `413.2 s`; vì vậy không có thao tác cancel thực tế. Kaggle báo đúng `GPU T4
  x2`, container SHA-256 `37c64f7d...d461`, `Output 0 B`. Direct Logs payload
  có `12,198` byte/`78` dòng, SHA-256
  `1c7a302b93d819e82f479c73382a3e53d33f00cb181b30162b67c26ed829a444`.
- Environment correction v2 được xác nhận hoàn toàn: ambient transformers
  `5.0.0` được thay bằng `4.50.2`, exact import assertion pass, `py_compile`
  pass và `26 passed in 11.93s`. Checkout/ancestry/T4x2/public SKELEX
  download+hash/split/input/cache/gallery guards và model load đều qua. Failure
  xảy ra trong train `build_skelex_descriptor_cache`, sau khi đã mở/infer một
  phần train radiographs nhưng trước khi hoàn tất train cache, trước descriptor
  operational gate, validation descriptor cache, selector training, scoring,
  prediction freeze hay independent audit:
  `RuntimeError: S5 original/flip support mass differs` tại runner dòng 259.
- Root cause là regression numerical guard: S5 đòi absolute agreement
  `atol=1e-6` sau float32 `128->14` area reductions. Horizontal flip đổi thứ tự
  reduction nên support toán học đối xứng có thể lệch vài float32 ULP. S5 đã
  không kế thừa correction-v9 đã được terminal-prove ở commit
  `24f374fcce9100b0a56d2a85c562eb10d1fdf058`: trên `2,981` ảnh/`174,669`
  candidates, max absolute delta `0.0001220703125` nhưng chỉ bằng `0.75` của
  per-candidate four-ULP tolerance. Log v2 không in delta thực tế nên không suy
  đoán con số cho S5.
- Không có scientific result/Dice: chưa prediction/pair freeze, chưa validation
  GT, chưa consumer và chưa BTXRD test. Error audit tracked tại
  `artifacts/kaggle/skelex_mask_bag_selector_s5_v1/kernel_version2_error_audit.json`,
  SHA-256
  `bd525398ca033f014f55e3bd13d5a083cf1421842ed77c50be11d824de7e82b3`.
- Trước successor, phải sửa cả hai implementation-only blockers đã biết: runner
  dùng exact four-ULP mass-symmetry contract, giữ validity/positive-support và
  ghi max delta/tolerance; independent auditor cast mọi reproduced rank về
  float32 trước aggregation/correlation. Không thay descriptor values, selector
  recipe, gallery, arms, prediction hay protocol mechanism. Theo rule, chưa sửa
  hoặc rerun trước khi mục `LỖI` v2 này được commit/push trung tâm.

### S5 version 3 — numeric identity correction đã chuẩn bị, chưa launch (2026-08-02)

- Sau khi failure v2 được audit và push ở commit
  `367f6fcf82d5d1e248d6c7f8dc065a4d904fd4ff`, hai correction implementation-only
  được đóng băng ở source commit
  `664578758225501dc163a6fc35d9ecdb9a1947d7`. Runner thay duy nhất equality
  guard bằng per-candidate maximum của four float32 ULP và four-epsilon floor đã
  có terminal evidence ở `24f374fc...`; validity, positive support, descriptor
  bytes, selector recipe và decision gate không đổi. Auditor thay duy nhất
  reproduced percentile ranks sang `numpy.float32`, đúng arithmetic của
  generator; prediction bytes không đổi.
- Numeric correction addendum
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_v1_numeric_correction_addendum.json`
  có SHA-256
  `ded254883a13da9ec0b961970ebacbd2b61badd04c644b7b9c64747a6abd2f72`.
  Nó giữ scientific source `61927cc...` và protocol `036e9d1d...`, đồng thời
  đóng băng corrected model/runner/auditor/test SHA-256 lần lượt là
  `c0119775...9b3` / `b23b61db...0bc` / `dbf84451...049` /
  `385a4c17...779`.
- Fail-closed canonical wrapper nay xác minh exact addendum, correction ancestry
  và đúng bốn source overrides trước tests/input; wrapper/test canonical SHA-256
  là `949e3a37aff7d56dd510c6dd3027c0e10705cd9b1d33bdaf1c19646d3d54f923` /
  `fb6d122838940dc62588eab7d0710f7185477339b204faa2a66d8484bcd2d490`.
  Local `py_compile`, `git diff --check` và wrapper tests `2 passed in 0.06s`.
  Host thiếu PyTorch/NumPy nên exact four-ULP và rank32 regression tests phải
  pass trên T4x2 wrapper trước public-model download và trước mọi real input.
- Đồng bộ log collaborator `added536...` xác nhận B2.2 terminal Dice
  `0.19172607`, area-score Spearman mean/median `0.933174/0.950055` và selector
  regret tăng `0.239569 -> 0.336572`; chỉ tin log, không truy cập output/Kaggle
  collaborator. Bằng chứng này củng cố candidate-conditioned signed
  inside-vs-local-context của S5, nhưng không làm thay đổi arm đã predeclare và
  không được dùng để rescue/sweep sau metric.
- Trạng thái `EXP-20260802-codex-s5-skelex-selector-v1` trở lại **ĐANG LÀM** cho
  đúng một version-3 correction rerun sau khi mục này/code được push central và
  prelaunch binding pass. Hiện chưa launch v3, chưa mở real input sau lỗi v2,
  chưa prediction/validation GT/consumer/test; collaborator output không được
  truy cập.
- Correction/code/log sync đã được push central ở commit
  `5954be469b82a545ac797e831be0bf3563338fbc`. Exact packaged version-3 wrapper
  chỉ thay ba launch-binding constants (`version=3`, ready true, checkout commit
  trên) so với canonical unbound wrapper; SHA-256 là
  `51b09f66edd34a23d4d17401db473b65759da4e5ad26a7ef84dce97250906c7e`.
  Metadata không đổi, SHA-256
  `33d4a8cb2fba94c1ad2a1742004d26390faae4de989b5fca4adace2a2bd5f107`.
- Prelaunch audit
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_kernel_v3_wrapper_audit.json`
  có SHA-256
  `f9f4a69cc4501dad749527a7efe0fb74bc5dd9a00e5106c9bc19ec612c719676`.
  Nó xác nhận exact addendum/4 source overrides, scientific và correction
  ancestry, canonical-vs-packaged identity ngoài binding, T4x2 fail-closed,
  numeric tests trước public-model download/real input, descriptor gate trước
  selector training, physical pair freeze và independent GT-blind audit trước
  evaluator. Version 3 vẫn chưa được push Kaggle tại mốc này; mọi khóa giữ nguyên.
- Sau khi claim/prelaunch audit đã được push central tại commit
  `852f66cb0ca7b7701a8cb4ca83d3266d05a32fe2`, exact bound wrapper version 3
  được đưa vào editor của chính private kernel và kiểm tra trực quan các binding
  `KERNEL_VERSION=3`, `LAUNCH_BINDING_READY=True`, checkout
  `5954be469b82a545ac797e831be0bf3563338fbc`, correction source/addendum và
  corrected auditor hash. Kaggle Save & Run All lúc
  `2026-08-02T01:51:43+07:00` xác nhận `Version #3 with GPU T4 x2`, trạng thái
  launch ban đầu `Queued`.
- Đây là launch duy nhất của wrapper SHA `51b09f66...06c7e`; không tạo competing
  job/monitor và không status-poll sau launch trong nhịp này. Experiment vẫn
  **ĐANG LÀM**; chưa có terminal output, prediction freeze/audit hay Dice.
  Validation GT, consumer và BTXRD test tiếp tục khóa; collaborator output không
  được truy cập.

### Đồng bộ collaborator `f5c0ff6` — duplicate audit cho inside/ring (2026-08-02)

- Đã fetch/read exact 29-line log delta mới tại
  `origin/codex/research-sync-20260731@f5c0ff6d2027e243ffca4509e2a7759cfb3b7c25`;
  chỉ đọc terminal statements/design trong Git, không truy cập output/Kaggle
  collaborator. Collaborator xác định draft inside/ring residual tái hiện gần
  như đầy đủ class-agnostic mask-bag v6: RAD-DINO inside/context/difference,
  all-normal candidate negatives, tumor-bag objective và detached positive
  winner; v6 đạt image AUROC `0.813386` nhưng Dice chỉ `0.217899`, dưới
  Geometry-v3 `0.245482`.
- Insight chuyển giao là normal negatives + local ring không đủ khi positive
  candidate vẫn là latent self-reinforcing winner. Vì vậy collaborator retire
  draft trước code/GPU và chuyển sang inference-only matched-normal candidate
  transplant/sham causal diagnostic. S5 không trùng exact rerun: scientific
  delta đã đóng băng của S5 là thay RAD-DINO bằng musculoskeletal SKELEX frozen
  representation trong cùng family, nhằm trả lời representation bottleneck.
  Tuy nhiên prior thành công của selector-MIL S5 được hạ xuống; nếu terminal
  audited result không vượt identity, retire ngay, không rescue sweep hoặc lặp
  transplant mà collaborator đang chuẩn bị.

### S5 kernel version 3 — LỖI synthetic floor/ULP contract trước real input (2026-08-02)

- Đúng một bounded status check sau launch thấy version 3 terminal **LỖI** sau
  `50.2 s` trên `GPU T4 x2`, `0` output files. Direct log được tải từ link Logs
  của chính kernel vào ignored temp
  `tmp/kaggle/skelex_s5_v3_terminal_error_20260802/kernel.log`: `9,932` byte,
  `106` dòng, SHA-256
  `aa6d3d7b93ff58ca330e76c7fef8c1e0f8e91cf4dda779a1354b951a13e4f66e`.
  Không polling lặp/monitor và không truy cập collaborator output.
- Checkout `5954be469b82a545ac797e831be0bf3563338fbc`, scientific/correction
  ancestry, exact protocol/addendum/source guards, T4x2, cài/import assertion
  `transformers==4.50.2` và `py_compile` đều pass. Wrapper dừng trong suite
  synthetic trước public SKELEX download và trước resolve bất kỳ BTXRD input:
  `1 failed, 27 passed in 9.91s`, tại
  `test_mass_symmetry_uses_the_proven_four_float32_ulp_budget`.
- Root cause là **test contract error**, không phải production tolerance. Test
  yêu cầu mọi five-ULP delta đều lớn hơn `max(4 ULP, 4 epsilon)`. Với mass
  `0.25`, five ULP chỉ `1.49011612e-7`, nhỏ hơn intentional absolute floor
  `4.76837158e-7`; implementation chấp nhận điểm floor-dominated này là đúng,
  nhưng `np.all` rejection assertion sai. Ba mass `1/17/196` còn lại đều chứng
  minh five ULP vượt tolerance như dự kiến.
- Error audit tracked tại
  `artifacts/kaggle/skelex_mask_bag_selector_s5_v1/kernel_version3_test_contract_error_audit.json`,
  SHA-256
  `1785bc25e2d4c5f539746a0673ecd496db6577ec1a2edf1d5b7d42bb1d771962`.
  Không scientific result/Dice: không checkpoint download, radiograph,
  descriptor, selector, prediction/freeze, validation GT, consumer hay test.
- Correction duy nhất được phép trước successor là test-only: tách case
  floor-dominated `0.25` khỏi ULP-dominated `1/17/196`, assert rõ cả hai nhánh;
  không đổi model/runner/auditor/tolerance/protocol. Theo coordination rule,
  chưa sửa/rerun cho tới khi record `LỖI` này được commit và push central.

### S5 version 4 — test-only correction đã freeze, chưa launch (2026-08-02)

- Error record v3 đã được commit/push central ở
  `5d1cb8e4768aaf712a1014007f0d0235c174e5d2` trước mọi correction. Exact
  test-only correction sau đó được commit/push ở
  `f9e56111ddf98b474c3ea1532c2da77b68e90232`: case mass `0.25` nay assert
  intentional absolute-floor acceptance; các mass `1/17/196` assert four ULP
  accepted, five ULP rejected. Corrected test canonical-LF SHA-256 là
  `9efb49f903ce9343bcb96cc381e5315c949dd6f2d3e6942169bdc98152e687da`.
- Test-contract addendum mới
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_v1_test_contract_correction_addendum.json`
  có SHA-256
  `591858f1e5bfaefad55b9583f3904ae6447bc98986a77d7a52471a49586b74a8`;
  nó kế thừa immutable numeric addendum `ded25488...`, error audit v3
  `1785bc25...` và chỉ override một test hash. Local NumPy `2.3.5` reproduction
  của exact deltas/tolerances pass; `py_compile`, `git diff --check`, wrapper
  tests `2 passed in 0.06s`.
- Canonical wrapper được mở rộng fail-closed để verify ancestry và exact hash
  chain protocol -> numeric addendum -> test-only addendum trước tests/input;
  model/runner/auditor hash, mass tolerance, descriptor/selector/prediction và
  scientific protocol không đổi. Canonical wrapper/wrapper-test SHA-256 là
  `f87b259c7f8491972540a55ba94290b3280a0a6c6e37d688085f024fe7ac8174` /
  `c29dab0b79649972077a95b70f159468c1084f30f27e9ca7d693e40def410a1b`.
- Trạng thái `EXP-20260802-codex-s5-skelex-selector-v1` trở lại **ĐANG LÀM** cho
  đúng một version-4 test-only correction rerun sau push central/prelaunch
  audit. Chưa launch v4, chưa mở real input, prediction/GT/consumer/test; mọi
  compute nặng và collaborator output vẫn khóa.
- Correction/addendum/wrapper/log đã được push central tại commit
  `59c3f3ce8906bf18601940114a2f1611b5ffd390`. Exact packaged version-4 wrapper
  chỉ khác canonical ở ba binding constants (`version=4`, ready true, checkout
  commit trên), SHA-256
  `f6183ad252e7be289cb6cc54108a307e353db01837f8c4f8e825c3ef36a20ffd`;
  metadata không đổi SHA-256 `33d4a8cb...f107`.
- Prelaunch audit
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_kernel_v4_wrapper_audit.json`
  có SHA-256
  `843bd3799bccf4a645cc5ccc892b4b42b4ca7195111b38f526255dc58623b84b`.
  Audit xác nhận exact protocol->numeric->test-only override chain, ancestry,
  canonical/packaged identity ngoài binding, tests trước public model/real input,
  T4x2, descriptor operational gate, physical prediction pair freeze và
  independent GT-blind reproduction. Version 4 chưa push Kaggle tại mốc này.
- Sau khi prelaunch claim/audit đã được push central tại commit
  `ca08c2cb762ccb75553e78ae3a94e14eba050980`, exact bound wrapper version 4
  được thay vào chính private kernel và kiểm tra trực quan đầy đủ version/checkout,
  three-source ancestry constants, both correction addenda và corrected test
  override. Kaggle Save & Run All lúc `2026-08-02T02:02:54+07:00` xác nhận
  `Version #4 with GPU T4 x2`, trạng thái launch ban đầu `Queued`.
- Đây là launch duy nhất của wrapper SHA `f6183ad2...0ffd`; không competing job,
  không monitor và không status-poll sau launch trong nhịp này. Experiment vẫn
  **ĐANG LÀM**; chưa có terminal output/prediction/Dice. Validation GT, consumer,
  BTXRD test và collaborator output tiếp tục khóa.

### S5 v4 — post-freeze evaluation readiness khi kernel pending (2026-08-02)

- Không thực hiện status check mới khi v4 vừa launch. Static readiness contract
  được freeze tại
  `artifacts/research_protocols/skelex_mask_bag_selector_s5_v1_postfreeze_evaluation_readiness.json`,
  SHA-256
  `e2f962b2c680dde1f387ec682410107d0b1f8044e5bd5901a01e6f99864ea254`.
- Required order cố định: later bounded terminal check -> compact output
  retrieval -> inventory/wrapper physical audit -> independent GT-blind
  descriptor/logit/rank/score/map reproduction -> freeze audit/pair hashes ->
  evaluate control và SKELEX-primary riêng -> freeze two per-image tables ->
  matched comparator không mở lại GT -> predeclared decision. Evaluator chỉ
  import segmentation dataset/mở baseline GT-derived table sau khi arm/cache/
  baseline freezes, score manifests và physical maps đã pass.
- Evaluator/comparator canonical-LF SHA-256 vẫn khớp scientific protocol:
  `ccc3a493...084` / `2b868f93...19`; cả hai `py_compile` pass. Hai arm dùng
  cùng `10,000` complete-group bootstrap draws seed `20261101`. Không prediction,
  validation GT, metric, consumer, test hay local heavy compute được mở/chạy ở
  bước readiness này.
- Đúng một bounded status check lúc `2026-08-02T02:10:56+07:00`, sau khoảng
  thời gian dài hơn failure boundary v2/v3, thấy version 4 vẫn `RUNNING` ở
  `483.3 s`, đúng `GPU T4 x2`, output tạm thời `0 B`. Không mở partial log,
  không repeat poll/monitor, không output/GT/test access; experiment tiếp tục
  **ĐANG LÀM** chờ một nhịp terminal hợp lý sau.

### Đồng bộ collaborator `70df5b4` — matched-normal Stage A, chưa có efficacy result (2026-08-02)

- Đã fetch/read exact log delta mới tại
  `origin/codex/research-sync-20260731@70df5b4d9228a14b7650e06301f67be99b5017fd`;
  chỉ đọc log Git, không truy cập output/Kaggle của collaborator. Workstream
  `EXP-20260802-codex-rich-gallery-matched-normal-transplant-stage-a-v1` đã
  triển khai inference-only matched-normal transplant/sham trên immutable
  classifier448 và rich gallery; decomposition ở `pool0`, `transition1/2/3`,
  `norm5` được dùng để phân biệt input evidence, backbone erasure, pooling
  dilution và selector/fusion error. Core/no-GT/bottleneck tests pass `15/15`,
  nhưng đây chưa phải Dice result.
- Kaggle versions 1–3 của collaborator chỉ cung cấp error-boundary kỹ thuật:
  thiếu `project/config.py`, dataset replacement thiếu locked classifier448
  checkpoint, rồi case-folding sai canonical `image_id` gây
  `KeyError: 'img002739.jpeg'` trước candidate scoring. Correction giữ nguyên
  canonical case, chỉ dùng `casefold()` cho compare/hash; regression tests
  `11/11` và real manifest traversal xác minh `1,484` matched/random pairs.
  Không có efficacy evidence để adopt/promote, và không launch pipeline trùng;
  S5 tiếp tục kết luận riêng về representation bottleneck.

### S5 kernel version 4 — HOÀN THÀNH, GT-blind audit PASS nhưng efficacy gate FAIL (2026-08-02)

- `EXP-20260802-codex-s5-skelex-selector-v1` kết thúc **HOÀN THÀNH — FAIL
  GATE**. Exact private kernel
  `itsthang333/btxrd-skelex-mask-bag-selector-s5-v1`, version `4`, terminal
  successful sau `1140.5 s` trên `GPU T4 x2`, checkout
  `59c3f3ce8906bf18601940114a2f1611b5ffd390`, bound-wrapper SHA-256
  `f6183ad252e7be289cb6cc54108a307e353db01837f8c4f8e825c3ef36a20ffd`.
  Direct log `31,294` byte có SHA-256
  `7cd563307955311e39eecff2b436df0d65c01fd0baa983f01c993f8cfc6672bf`.
- Official output inventory có `1,872` file. CLI thường tải được `77` file rồi
  gặp một `kaggleusercontent` connect timeout; atomic inventory downloader giữ
  các file hoàn chỉnh và resume `1,795` file còn lại. Local temp cuối có `1,873`
  file tính cả direct log, `241,220,704` byte và không còn `.part`. Browser
  Download trước đó không tạo file hoàn chỉnh trong bounded timeout; đây chỉ là
  transport error, không phải kernel/scientific failure.
- Trước validation GT, descriptor operational gate PASS, prediction pair đã
  freeze vật lý SHA-256
  `d30ed3da98bf60ba368f27cb4747bfc715f2c16a1bcf2e6d72602cfb00f41ef0`.
  Wrapper audit SHA-256 `e86ccafe...3358`; embedded independent audit SHA-256
  `f61ed896...d186`. Independent local re-audit xác minh `742` maps, `742`
  score files, `371` descriptor-evidence files và `371` prediction mỗi arm;
  canonical-LF output trùng exact `f61ed896...d186`. Tại boundary này
  `validation_gt_read=false`, `consumer_trained=false`, `test_evaluated=false`.
- Local evaluation có các lỗi môi trường không khoa học trước output cuối:
  default Python thiếu NumPy; bundled Python thiếu PyTorch; env có PyTorch dùng
  Python `3.9.23` nên dừng ở `zip(..., strict=True)` sau cohort computation;
  lần đầu của shim thiếu target directory trong `sys.path` nên dừng ở import.
  Corrected fail-closed strict-zip shim SHA-256
  `dcf88d82e6581e7bb3a37f05bf22bb93aba83294993b6c5e12b460909e1c396a`
  kiểm tra equal/unequal lengths và pass smoke test; frozen evaluator/comparator
  không sửa, vẫn SHA-256 `ccc3a493...084` / `2b868f93...19`.
- Sau audit pass mới mở validation GT và frozen baseline per-image, chạy hai arm
  riêng với đúng `10,000` complete-group bootstrap, seed family `20261101`.
  Control `geometry_v3_plus_upstream_equal_rank` đạt Dice
  overall/small/medium/large =
  `0.25520289 / 0.12563547 / 0.39380626 / 0.37741925`, `70` complete misses;
  per-image SHA-256 `55846ae0...001c`.
- Primary `geometry_v3_plus_upstream_plus_skelex_equal_rank` đạt
  `0.26116508 / 0.11696429 / 0.40348515 / 0.44493337`, `65` complete misses;
  per-image SHA-256 `db902670...f284`. So với accepted Geometry-v3
  `0.24548239 / 0.11708058 / 0.37713552 / 0.38941265`, delta là
  `+0.01568269 / -0.00011629 / +0.02634963 / +0.05552073`; CI95 tương ứng
  `[-0.00676722,+0.03862143] / [-0.03302726,+0.03191985] /
  [-0.00473555,+0.06076645] / [-0.02078531,+0.16558063]`.
- Frozen comparator primary-minus-control có SHA-256 `4806e937...cd67`; delta
  overall/small/medium/large =
  `+0.00596219 / -0.00867119 / +0.00967889 / +0.06751412`, với CI95
  `[-0.01476152,+0.02656770] / [-0.04094832,+0.02014036] /
  [-0.01386327,+0.03281298] / [-0.00160671,+0.17478286]`. Mọi CI đều cắt 0.
- Kết luận: SKELEX representation là tín hiệu dương cục bộ cho medium/large và
  tăng mean score-quality Spearman lên `0.437970`, nhưng equal-rank aggregation
  không giải quyết small/complete-miss bottleneck, absolute candidate-count vs
  miss association còn tăng và cả bốn operational goal đều fail. Không promote,
  không post-hoc rescue/sweep và không train consumer. Candidate oracle vẫn vượt
  toàn bộ goal, nên successor phải nhắm uncertainty/routing hoặc causal evidence
  chưa trùng, không quay lại proposal generation.
- Terminal audit tracked tại
  `artifacts/kaggle/skelex_mask_bag_selector_s5_v1/kernel_version4_terminal_evaluation_audit.json`,
  SHA-256
  `18f29270e487c82bd8313931ca91a894c1949898c4377ffb4afbcbef0c7c63ec`.
  Prediction freeze/GT boundary được giữ; collaborator output không truy cập;
  BTXRD test vẫn khóa.

### Đồng bộ collaborator `9c819ec` và phân tích successor sau S5 (2026-08-02)

- Đã fetch lại `origin/research-wsss-improvement` tại
  `1782b34f0ce2382f58faf8e5d0652d36c2c8ada7` và
  `origin/codex/research-sync-20260731` tại
  `9c819ecd0956702f3c4dea880ab3f19d1e984ff3`; worktree trước phần chuẩn bị S6
  sạch ngoài file S6 mới. Toàn bộ log trung tâm không đổi từ lần đọc đầy đủ;
  delta collaborator `70df5b4..9c819ec` đã được đọc đủ `27` dòng mới.
- Claim collaborator duy nhất liên quan vẫn là matched-normal transplant Stage-A
  version 4/Stage-B, trạng thái `ĐANG LÀM`. Delta mới chỉ mở rộng schema hậu
  freeze cho matched/random layer statistics và selector-regret decomposition;
  chưa có terminal Dice hay bằng chứng hiệu năng. Không truy cập Kaggle/output
  collaborator, không kế thừa cơ chế này như cải tiến và không mở pipeline cạnh
  tranh.
- Audit toàn log cho thấy selector cùng-gallery đã dùng nhãn ảnh nhị phân ở
  `image_bag_loss`, còn taxonomy `benign/malignant` và chín `tumor_type` trong
  split chưa được dùng để học candidate ranking. Không tìm thấy experiment
  subtype/hierarchical-label tương đương. Đây là scope khác proposal generation,
  matched-normal transplant, SKELEX descriptor và mọi residual selector đã fail.
- Bằng chứng khoa học dùng để thu hẹp hypothesis:
  Cole et al., *On Label Granularity and Object Localization*, ECCV 2022,
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7044_ECCV_2022_paper.php ;
  Wang et al., *Multiple Granularity Descriptors for Fine-Grained
  Categorization*, ICCV 2015,
  https://openaccess.thecvf.com/content_iccv_2015/html/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.html ;
  Jang và Kwon, *Are Multiple Instance Learning Algorithms Learnable for
  Instances?*, NeurIPS 2024,
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/1468ecc3d7e9dc2fbf336eed9bb292e0-Abstract-Conference.html ;
  Liu và Ji, *Weakly-Supervised Residual Evidential Learning for Multi-Instance
  Uncertainty Estimation*, ICML 2024,
  https://proceedings.mlr.press/v235/liu24ac.html ;
  Li, *A Multiclass Multiple Instance Learning Method with Exact Likelihood*,
  https://arxiv.org/abs/1811.12346 ; và Choe et al., *Evaluating Weakly
  Supervised Object Localization Methods Right*, CVPR 2020,
  https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html .
  Các nguồn chỉ tạo rationale cho label granularity, multiclass MIL,
  uncertainty shrinkage và prediction-first; chưa phải bằng chứng BTXRD.

### EXP-20260802-codex-s6-label-granularity-mil-v1

- **Owner/status:** Codex main task trên `research-wsss-improvement`;
  **HOÀN THÀNH — TERMINAL AUDITED, GATE FAIL**. Đăng ký lúc
  `2026-08-02T06:04:41Z`; registration base là
  `1782b34f0ce2382f58faf8e5d0652d36c2c8ada7`; exact registration commit sẽ
  được ghi ở note kế tiếp sau khi push. Claim phải hiện diện trên central trước
  mọi real cache load/fit/prediction/Kaggle launch.
- **Kế thừa/không trùng:** kế thừa exact candidate supply và base scorer đã freeze
  của Geometry-v3 cùng negative evidence R1/R2/S1/R3/R4/S3/S4/T1/S5; không coi
  bất kỳ cơ chế reject nào là cải tiến. Không dùng output/pipeline collaborator
  và không trùng matched-normal Stage-A/Stage-B đang chạy. Không tạo proposal,
  không thay encoder/mask/geometry/evaluator và không dùng SKELEX.
- **Hypothesis:** binary MIL làm mọi pathology cạnh tranh dưới một tumor score,
  nên candidate dương tiềm ẩn không được phân giải theo semantic subtype. Một
  residual chín subtype, chỉ dùng taxonomy cấp ảnh, có thể giảm candidate regret;
  center descriptor/residual trong từng bag chặn offset anatomy/background toàn
  ảnh, còn entropy route liên tục co về coarse score khi subtype không chắc.
- **Matched finite pair:** cả `coarse_control` và
  `hierarchical_entropy_routed` dùng cùng kiến trúc chín output, exact zero-init,
  frozen base checkpoint, seed/batch/order/16 epoch, normalized SmoothMax
  `temperature=0.2`, AdamW `lr=3e-4`, weight decay `1e-4`, flip consistency
  `0.10`, residual drift `1e-3`; không hard/soft winner và không instance target.
  Control chỉ có binary BCE nên chín column exchangeable. Primary thêm loss
  binary/benign-malignant/subtype với trọng số `1/3` mỗi phần và inverse-sqrt
  train-label class weight. Inference không nhận val subtype: subtype được dự
  đoán từ bag, residual fine được shrink bằng
  `1-H(p_subtype)/log(9)` về coarse residual.
- **Exact input/provenance:** split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  selector-cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`;
  cache manifest
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`;
  baseline checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`;
  cohort `2981 train / 371 validation`, descriptor `1156`, nine train subtype
  counts `598/211/164/74/35/41/92/237/36`. Validation polygons không phải input.
- **Compute/output/gates:** static/synthetic locally; mọi real cache fit/inference
  chỉ private Kaggle T4x2/P100. Runner phải freeze đủ hai checkpoint/history,
  `742` candidate-score payload, `742` maps, label-count/diagnostic evidence và
  pair-freeze trước GT. Independent GT-blind audit phải pass source/input/hash,
  cohort, zero-init identity, exact selection/entropy route và safety locks;
  count Spearman, image AUROC, subtype macro metrics, flip agreement, entropy và
  change fraction là diagnostic, không chọn model/hyperparameter và không chặn
  Dice nếu integrity pass. Sau đó evaluator/comparator đã freeze mới đọc val GT
  với `10,000` complete-group bootstrap.
- **Decision:** mechanism chỉ pass nếu primary tốt hơn matched control ở overall
  và small mean Dice, không giảm medium/large và không tăng complete miss. Chỉ
  operational pass đồng thời ở
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336`, overall CI95 lower `>0`,
  không subgroup regression/miss increase mới authorize consumer. Nếu không,
  terminal reject, không rescue/sweep/post-hoc fusion. BTXRD test luôn khóa.
- **Static preparation at registration:** thiết kế đầy đủ tại
  `S6_LABEL_GRANULARITY_MIL_DESIGN.md`; primitive tại
  `project/models/mask_bag_label_granularity.py`; synthetic test tại
  `tests/test_mask_bag_label_granularity.py` pass `7/7`. Đây chưa phải scientific
  result: chưa load real cache, chưa fit, chưa prediction, chưa mở validation GT,
  chưa train consumer và chưa đọc BTXRD test.

### S6 source/runner/independent-auditor static readiness (2026-08-02)

- Exact registration commit `582652e2520f752723c11bcfce4656d9947d7a71`
  đã push và nhìn thấy trên `origin/research-wsss-improvement` trước mọi real-data
  action. Claim S6 tiếp tục `ĐANG LÀM`; chưa có Kaggle binding/launch.
- Model/training closure hiện gồm
  `project/models/mask_bag_label_granularity.py` và
  `project/models/mask_bag_label_granularity_training.py`. Nó khóa valid-candidate
  centering, nine-column exact zero-init, normalized subtype/pathology pooling,
  inverse-sqrt train-label weight, entropy route và matched batch/dropout order.
  Batch toàn normal được xử lý fail-safe: chỉ phần binary có gradient, hai phần
  tumor-only bằng zero; không ép mọi batch phải có tumor và không tạo instance
  label.
- Runner `project/run_mask_bag_label_granularity_s6_pair.py` không import
  evaluator/annotation. Nó join taxonomy từ exact split, verify subtype count,
  tái tạo base logits, audit zero-init trên `2981+371` record trước optimizer,
  fit đúng hai arm 16 epoch, dùng hai T4 cho disjoint validation shards, rồi ghi
  đủ two-arm score/map/checkpoint/history/diagnostic/pair freeze. Validation
  subtype không được dùng để route; chỉ bag prediction quyết định subtype.
- Independent auditor
  `project/audit_mask_bag_label_granularity_s6_output.py` không import runner,
  training orchestration hay evaluator. Nó tự load checkpoint/cache/base, tái
  tính residual, SmoothMax, subtype posterior/entropy route, score, winner,
  bag probability và physical float16 map cho cả `742` output; tolerance CPU-vs-
  GPU candidate logit được predeclare `5e-5`, map phải exact. Chỉ integrity/safety
  fail mới chặn GT; diagnostic value không chọn arm/hyperparameter.
- Focused static/synthetic suite pass `19/19`; sau correction all-normal thêm
  runner suite pass tổng `19/19`. Lần full-regression trực tiếp trên Python
  `3.9.23` có `13 failed, 574 passed`, trong đó `12` fail là known
  `zip(..., strict=True)` Python>=3.10 incompatibility của code cũ. Chạy lại bằng
  documented fail-closed strict-zip shim còn đúng `1 failed, 586 passed`: test
  legacy B1 so raw worktree CRLF SHA của `project/datasets/btxrd.py`
  (`dcb509a3...`) với canonical Git-LF SHA cũ (`96f5abaa...`). Đây là boundary
  môi trường/serialization đã biết, không liên quan S6; không sửa protocol/code
  B1 và không dùng nó để nới gate.
- Một focused S6 test ban đầu cũng gặp `TypeError: zip() takes no keyword
  arguments` trong code mới trên Python 3.9. S6 đã thay riêng các `strict=True`
  đó bằng explicit length-equality check trước `zip`; mechanism/loss/recipe
  không đổi. Sau correction focused suite pass như trên.
- Chưa mở real selector cache/radiograph, chưa fit/prediction, chưa validation
  polygon, chưa consumer/test. Bước tiếp theo là commit/push exact scientific
  source, tạo protocol đóng băng source/hash và wrapper/binding fail-closed trước
  launch.

### S6 exact source/protocol freeze, chưa binding/launch (2026-08-02)

- **Hiệu chính append-only:** note static-readiness ngay trên ghi nhầm full
  registration hash `582652e...a71`; exact commit thực tế là
  `582652edb988af1d5da7dd6bf86cfc8ab2ec1b84`. Short hash và thứ tự push ở note
  cũ đúng; hiệu chính này không xóa dấu vết cũ.
- Exact scientific source commit
  `543ee89654a0ed00e80ded16924a760585337924` đã push central. Nó chứa toàn bộ
  model/training/runner/independent-auditor và focused tests đã mô tả; chưa chứa
  protocol/binding và chưa chạy input thật.
- Frozen protocol
  `artifacts/research_protocols/rad_dino_mask_bag_label_granularity_s6_v1.json`
  có SHA-256
  `f4e17d24dfab36f01526550c7dc306fc7549494acc4545153454c61ae926bfc3`.
  Protocol khóa exact scientific commit, canonical-LF source closure, immutable
  split/cache/base, matched pair/loss/16 epoch/seed, entropy formula, T4x2,
  `742` score + `742` map, independent tolerance `5e-5`, exact float16 maps,
  diagnostic-nonblocking contract, bootstrap seed `20261201` và mọi
  GT/consumer/test lock. Không sweep/early stop/rescue.
- Protocol closure test mới normalize CRLF về canonical LF trước SHA để tránh
  lặp lại boundary B1; focused S6 protocol/source/auditor suite pass `21/21`.
  Chưa tạo wrapper/binding/package/Kaggle version; validation polygon, consumer
  và BTXRD test vẫn khóa.

### S6 unbound wrapper/binder readiness, chưa launch (2026-08-02)

- Unbound fail-closed wrapper
  `project/kaggle_wrappers/run_mask_bag_label_granularity_s6_v1.py` có
  canonical-LF SHA-256
  `7cd9bbc0248bc6d5371fb00f04c3102cf33b5e001c36a9b20f3f551130727966`,
  giữ `KERNEL_VERSION=0`, `LAUNCH_BINDING_READY=False`, checkout `UNBOUND`.
  Nó clone exact checkout, verify ancestry/protocol/toàn source closure, dựng
  split CRLF exact từ Git LF, tìm duy nhất accepted baseline/cache theo hash,
  chạy real convolution trên cả hai T4, focused tests, runner rồi independent
  GT-blind auditor. Nó không import/chạy evaluator/comparator và không cần raw
  radiograph, train gallery hay model download mới.
- One-time binder
  `project/bind_mask_bag_label_granularity_s6_wrapper.py` canonical-LF SHA-256
  `0fd97a4ae1d17282e672f8e164c1fd1b2de472b014a43c7bb28fc15e73464733`
  chỉ thay đúng ba field version/ready/checkout, bắt buộc inverse reconstruction
  về exact template, kiểm tra protocol/source hashes tại execution checkout và
  ghi launch binding độc lập. Wrapper/binder tests có SHA-256
  `4c809e3c...9148` / `8a70bc2f...71a6`.
- Focused S6 source/protocol/wrapper/binder/auditor suite pass `25/25`; py_compile
  pass. Chưa bind/package/push Kaggle, chưa real cache/fit/prediction/GT; mọi
  safety lock giữ nguyên. Sau khi commit này visible central mới được bind đúng
  version 1 tới exact execution checkout.

### Đồng bộ collaborator `e715539` — matched-normal terminal reject (2026-08-02)

- Đã fetch/read toàn bộ delta log `9c819ec..e715539` từ
  `origin/codex/research-sync-20260731`; chỉ dùng bằng chứng trong Git theo yêu
  cầu, không truy cập Kaggle/output của collaborator. Stage-A v4 đã terminal và
  independent audit pass `371` payload, `1,855` selection row, baseline
  reproduction `371/371`, không GT/test; prediction-freeze/audit SHA-256 là
  `3e9760d3b98ac5dbe1d909db74968483eaf0fff0d2bc6c70c46668a7479ab765` /
  `67c20c4b9d9c140b1d91bb89d4d7e4f3368346fe9815a4043a412729ef555580`.
- Actual Dice baseline/transplant-only/equal/primary 3:1/random-control 3:1 lần
  lượt là `0.28872949 / 0.08812762 / 0.22623612 / 0.28106662 / 0.26317766`.
  Primary 3:1 kém baseline `-0.007763`, CI95
  `[-0.038285,+0.021947]`; small giảm `-0.062685`, CI95
  `[-0.107285,-0.025327]`, dù medium/large tăng `+0.043854/+0.101229`.
  Do overall và small gate fail, matched-normal bị retire; không sweep/rescue và
  không có kỹ thuật hiệu năng dương để S6 adopt.
- Negative insight mới thu hẹp bottleneck: proposal-supply regret chỉ
  `0.000396`, eligible selector regret `0.239173`; baseline có `49` miss và
  `104` wrong-source. Mask được chọn cho small có median area `14.603x` GT,
  trong khi large chỉ `0.382x`; transplant làm đổi `143` lựa chọn và mở rộng
  mask đổi `1.427x`. Norm5 quality correlation chỉ `0.0495`, area correlation
  `0.4294`, matched-minus-random oracle-percentile gain `0.000048`, nên cơ chế
  chủ yếu truyền mass/area chứ không truyền tumor identity ổn định.
- S6 đã predeclare/freeze trước terminal result này và không dùng transplant,
  rich gallery hay classifier448. Scope của S6 vẫn là kiểm tra taxonomy cấp ảnh
  trên immutable same-gallery cache bằng matched control; không đổi protocol
  theo kết quả hậu nghiệm. Bằng chứng collaborator chỉ được giữ để diễn giải
  signed extent/selector-regret nếu S6 terminal, không được coi là cải tiến.

### S6 kernel-v1 binding và final prelaunch PASS (2026-08-02)

- Binder đã khóa version `1` tới exact execution checkout
  `c59498658f320c9cc60cde6e5453b07c0165363f`, scientific source
  `543ee89654a0ed00e80ded16924a760585337924` và protocol SHA-256
  `f4e17d24dfab36f01526550c7dc306fc7549494acc4545153454c61ae926bfc3`.
  Launch-binding SHA-256 là
  `2571dea74f8febaf338bd942bc0a831d2ffaaa09977c62ebd31160fe443d4a38`;
  exact bound-wrapper SHA-256
  `e1c855958c638addf1f2ac1c99659779363274a9c0d7138ccd9a38ecaa885281`;
  inverse reconstruction về unbound template SHA
  `7cd9bbc0248bc6d5371fb00f04c3102cf33b5e001c36a9b20f3f551130727966`
  pass với đúng ba replacement.
- Private T4 package chỉ có đúng bound wrapper `17,903` byte và
  `kernel-metadata.json` `630` byte; metadata SHA-256
  `489127ab983d604d776dfa3e7d1634a867c13d7e31758d65effb421d790972e6`.
  Input chỉ gồm dataset
  `itsthang333/btxrd-mask-bag-selector-baseline-v1` và accepted selector-cache
  kernel `itsthang333/btxrd-rad-dino-mask-bag-selector-cache-v1`; không có
  collaborator output, raw GT hay BTXRD test.
- Final focused suite pass `25/25`, bound-wrapper `py_compile` pass và package
  không có thư mục/file thừa sau khi xóa generated `__pycache__`. Final
  prelaunch audit SHA-256
  `bf99e4c0016ba195af1c164cec05b334ed550154c6bb62de03c11733cc5ad337`
  có `status=PASS`, `authorized_launch=true`, không scientific collision với
  terminal matched-normal. Full-regression boundary cũ vẫn là Python 3.9
  `zip(strict=True)` và B1 CRLF/LF, không liên quan S6.
- Tại freeze này chưa launch Kaggle, chưa load real cache/fit/prediction, chưa mở
  validation GT, chưa train consumer, chưa đọc BTXRD test. Claim
  `EXP-20260802-codex-s6-label-granularity-mil-v1` tiếp tục `ĐANG LÀM`; chỉ được
  launch sau khi exact binding/audit/log commit này visible central.

### S6 kernel version 1 — launched, chưa status-poll (2026-08-02)

- Exact binding/final-prelaunch/log commit
  `34ee708213fe97bce315ac07ce1a75136a0988b1` đã push và được fetch thấy trên
  `origin/research-wsss-improvement`; collaborator vẫn ở `e715539`, không có
  active collision mới. Ngay trước launch, package được re-verify đúng hai file,
  không directory thừa, wrapper/metadata SHA không drift.
- `kaggle kernels push` bằng Kaggle CLI `2.2.3` đã trả
  `Kernel version 1 successfully pushed` cho private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-label-granularity-s6-v1` lúc
  `2026-08-02T06:33:33.8091737Z`. Metadata yêu cầu `NvidiaTeslaT4`; runtime
  wrapper vẫn phải fail-closed nếu không thấy đúng T4x2 và mọi frozen
  source/protocol/input gate.
- Launch audit SHA-256
  `c681549e0913bd84347d545a388b31d31a332102fc57d36851f3e29325013b6b`.
  Không status-poll sau launch và không tạo monitor. Chưa đọc partial
  log/output, chưa validation GT, chưa train consumer, chưa đọc BTXRD test.
  Experiment tiếp tục **ĐANG LÀM** chờ một bounded terminal check ở nhịp sau.

### S6 post-freeze seed correction và evaluation readiness (2026-08-02)

- Static review ngay sau launch, trước bất kỳ terminal status/output/prediction
  hay GT access nào, phát hiện cùng boundary từng gặp ở B4: protocol S6 v1 ghi
  bootstrap seed `20261201`, nhưng exact generic comparator đã freeze trong
  protocol chỉ chấp nhận `20261101` và sẽ fail tại argument gate nếu giữ seed
  cũ. Đây là execution error hậu-freeze, không phải model/scientific result.
- Correction artifact
  `rad_dino_mask_bag_label_granularity_s6_v1_postfreeze_seed_correction.json`
  SHA-256
  `a0eb5f83ace094c6fac015c0331740f614799c6c62706be6073a602578ddb7b4`
  khóa seed `20261101` cho cả hai arm evaluation và matched comparator, trước
  metric. Không đổi model, loss, arm, prediction, endpoint, gate, source
  producer hay protocol SHA được ghi trong pair freeze; không rerun kernel.
- Post-freeze readiness SHA-256
  `db1048e505407d60daeb245a3e81b8a3c009c7e51d4fd8e72448e80174ad266a`
  khóa thứ tự terminal check → compact output → wrapper audit GT-blind →
  independent local reproduction → pair/audit freeze → evaluate hai arm với
  cùng seed → matched compare không mở lại GT → S6 decision. Exact evaluator /
  comparator canonical-LF SHA vẫn `ccc3a493...084` / `2b868f93...119`;
  `py_compile` pass và focused synthetic suite pass `9/9` dưới fail-closed
  Python-3.9 strict-zip shim.
- Generic evaluator `gate_decision.json` dùng mechanism checks của các selector
  cũ nên không được thay cho S6 gate. S6 mechanism đã khóa trước metric: primary
  phải tăng mean overall và small, không giảm medium/large, không tăng complete
  miss so với matched control. Consumer còn yêu cầu đủ bốn goal, primary-vs-
  accepted-Geometry-v3 overall CI95 lower `>0`, mọi safety gate; AUROC/subtype/
  count/regret/flip/entropy/change fraction tiếp tục diagnostic nonblocking đúng
  protocol. Không rescue/sweep nếu fail.
- Không status-poll/monitor/output access ở bước này; validation GT, consumer và
  BTXRD test vẫn khóa.

### S6 kernel version 1 — LỖI tại independent map audit (2026-08-02)

- Đúng một bounded status check lúc `2026-08-02T06:40:19.4527888Z` thấy private
  kernel version `1` ở `KernelWorkerStatus.ERROR`; không status-poll tiếp và
  không tạo monitor. Direct log `13,158` byte có SHA-256
  `836477c0e7fc87402753918810fdd2f299c18e42e5aa8facd9ee58267bc0c036`.
- Producer thực tế đã pass exact source/protocol/input và `48` Kaggle tests,
  chạy từ `06:33:59` đến `06:39:15Z` trên đúng `Tesla T4 x2`, validate
  `2,981 train / 371 val`, fit đủ 16 epoch và freeze vật lý cả hai arm trước GT.
  Pair freeze SHA-256
  `203dc6435b661b7410331f3eefca20c14888bcd34890b5228b4c35a0cecd36e2`;
  control/hierarchy freeze SHA-256
  `7cff6cc8d559401feceeffa3eef8ec0cd95d013ea62f86c28cb4f0c994a08c84` /
  `fa88dd2d392de347915575c35442b1ab3375bb51a65fdbaceb963635945b143c`.
- Failure xảy ra sau producer/pair freeze nhưng trước independent audit PASS và
  trước wrapper-output audit: `ValueError: S6 coarse_control map reproduction
  mismatch: IMG001871.jpeg`. Không có efficacy/Dice result và GT gate không mở.
- Exact local reproduction khóa root cause. Saved-vs-independent candidate-logit
  delta tối đa chỉ `3.8146973e-6`, nhỏ hơn tolerance `5e-5`; winner position và
  original index đều là `27`. GPU producer probability `0.2713623642921448`
  và CPU auditor probability `0.27136225761655475` chỉ lệch `1.0668e-7`, nhưng
  nằm hai phía float16 rounding midpoint nên thành `0.271484375` và
  `0.271240234375`. Vì vậy đúng toàn bộ `61,292` foreground pixel lệch một ULP
  `0.000244140625`; stored map tái tạo bit-exact khi dùng chính frozen producer
  probability. Auditor đã tự cho phép CPU/GPU score/probability tolerance nhưng
  lại yêu cầu bit-exact map từ scalar được phép lệch — đây là implementation
  inconsistency, không phải prediction drift.
- GT-blind diagnostics được producer freeze nhưng không dùng làm efficacy gate:
  control/hierarchy AUROC `0.828412/0.821233`, changed selection fraction
  `0.439353`, subtype accuracy/macro recall `0.467391/0.254918`. Các số này
  không cho phép chọn/rescue arm và chưa nói gì về Dice.
- Lần tải full output đầu tiên được dừng sau bounded wait khi đã giữ các file
  hoàn chỉnh, không có `.part`; targeted retrieval sau đó lấy direct log và core
  artifacts. Đây chỉ là transport boundary. Error audit tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_label_granularity_s6_v1/kernel_version1_error_audit.json`,
  SHA-256
  `1dacee061698b968ae6b5ca97a3e44baba22385ab6f8c23bfdc97048bc7cd933`.
- Version 1 được giữ vĩnh viễn là **LỖI**. Claim S6 chỉ tiếp tục cho một
  implementation-only independent-auditor correction: vẫn independently kiểm
  tra probability trong tolerance, winner/mask support và exact serialization
  từ frozen producer probability; không đổi model, loss, seed, epoch, scores,
  maps, protocol hay evaluator. Chưa sửa/rerun trước commit lỗi này. Validation
  GT chưa đọc, consumer chưa train và BTXRD test chưa mở.

### Đồng bộ collaborator `dc00062` — conditional/cross-view negative evidence (2026-08-02)

- Đã đọc toàn bộ Git-log delta `e715539..dc00062`; không truy cập Kaggle/output
  collaborator. Hai analyzer terminal chỉ là retrospective feasibility trên
  prediction/table đã freeze, không tạo selector/mask và không có active GPU
  claim cạnh tranh với S6.
- G1-conditional audit trên `32,519` candidate cho thấy transition2 matched
  relative-L2 partial rank correlation chỉ `0.039274` overall và `-0.013838`
  ở small; oracle-vs-baseline pair accuracy `49.14%`. Ring-mass residual lớn hơn
  nhưng matched/random giống nhau và chỉ đạt `45.7-46.9%`, nên là proxy support/
  geometry chứ không phải tumor identity. Không adopt hay sweep frozen layers.
- Cross-view feasibility có `443` train-tumor heuristic multi-view groups, nhưng
  trên validation raw same-group support chủ yếu phản ánh anatomy: sau matched
  control, median partial correlation chỉ `0.014121` và oracle-vs-baseline pair
  accuracy `18.75%`. Không append frozen cross-view cosine. High-resolution
  cross-view training design mới chỉ là hypothesis/preparation, chưa có terminal
  efficacy và chưa được S6 kế thừa.
- Delta này không đổi error boundary hay implementation-only auditor correction
  của S6; nó chỉ củng cố rằng selector successor sau S6 không nên quay lại frozen
  layer/cosine/area proxy nếu không có representation mới được kiểm chứng.

### S6 v2 independent-auditor numeric correction source (2026-08-02)

- Sau khi version-1 error audit đã push central ở commit
  `c80215cfc8a74f1d15d750affd77b688152d6273`, chỉ independent auditor và test
  của nó được sửa; producer/model/loss/recipe/protocol/prediction/evaluator giữ
  nguyên. Auditor vẫn independently tái tính candidate logits và probability,
  giữ tolerance `5e-5`, đồng thời mới bắt buộc independently reproduced argmax
  trùng frozen winner.
- Exact physical-map check giờ tách đúng hai invariant: (1) independently
  recomputed probability phải gần frozen producer probability trong tolerance;
  (2) mask của independently reproduced winner nhân frozen producer probability
  phải serialize float16 bit-exact với stored map. Helper còn fail-closed nếu
  probability ngoài `[0,1]`/không finite hoặc selected mask không binary. Điều
  này sửa contradiction float16 midpoint nhưng không nới winner/support/map
  bytes hay safety gate.
- Canonical-LF auditor/test SHA-256 mới là
  `79db19c3cbfba6373111b8221b9d61077a98e2c52eb3696b70293604e74dcc99` /
  `3b3b2721ad484e14f2b66c451c5d6bf1e8727d4916ecf6239a7e08d25aeb5946`;
  focused S6 audit/model/training/runner suite pass `20/20`, `py_compile` và
  `git diff --check` pass. Synthetic regression chứa đúng hai probabilities
  version 1 đã gặp và chứng minh chúng nằm trong tolerance nhưng khác float16.
- Đây mới là source correction tĩnh: chưa tạo correction addendum/wrapper/bind
  version 2, chưa rerun Kaggle, chưa đọc validation GT/metric, chưa consumer/test.

### S6 v2 correction addendum và unbound wrapper readiness (2026-08-02)

- Implementation-only correction addendum
  `rad_dino_mask_bag_label_granularity_s6_v1_map_audit_numeric_correction.json`
  SHA-256
  `b0dca40bf4f8bd933a902facb7bfdf5ec393c429672b0beb0b0594f2d15dfc63`
  khóa đúng hai canonical-LF override auditor/test từ protocol-v1 hashes sang
  `79db19c3...cc99` / `3b3b2721...5946`, bind correction source commit
  `7ca2f4dec72af5f509e52786980321d255a7eb68` và giữ protocol/scientific source
  SHA/commit cũ. Không file source nào khác được phép drift.
- Unbound wrapper mới canonical-LF SHA-256
  `c04c288501b95f0408c21e9e5cb4eb6bfcb1af159b82572c3eb26d63acf17492`.
  Nó verify exact correction artifact, correction commit ancestry, hai old/new
  hash pair và mọi source hash không đổi trước khi chạy producer/auditor. Runtime
  binding sẽ ghi correction source/SHA và actual corrected source closure.
  Binder canonical-LF SHA-256
  `88f61f0135075c3fa52202315f8feb4d20e84ea8ca727b1e671e639d4a205b48`
  áp cùng fail-closed rule trên exact Git checkout; vẫn chỉ có ba launch-field
  replacement và inverse reconstruction.
- Focused wrapper/protocol/auditor/model/training/runner suite ban đầu có `1`
  static-test failure: naive `source.index()` bắt tên auditor trong correction
  manifest trước subprocess runner. Execution order không đổi; test được sửa
  sang hai occurrence cuối (`rindex`) để kiểm tra actual subprocess order. Sau
  correction suite pass `24/24`, cả ba source `py_compile` và diff check pass.
  Binder integration test phải chạy sau commit vì nó cố ý dùng `git show` exact
  checkout, không chấp nhận uncommitted worktree.
- Chưa bind/package/push version 2 và không tái sử dụng output v1 để mở GT.
  Prediction/validation GT/consumer/test tiếp tục khóa.

### S6 kernel-v2 binding và final prelaunch PASS (2026-08-02)

- Sau khi wrapper/correction commit `e10db06537eedb4314faa36f173e96d0643b54ab`
  visible central, exact-checkout binder integration cùng toàn focused suite pass
  `26/26`. Binder khóa version `2` tới commit này, scientific source/protocol
  cũ và correction source `7ca2f4d...`; binding SHA-256
  `f82e511bc6fe4eb60a34e2a4e0e65d1d3d9d4745dea4b37094108be6dd2152ef`.
- Bound wrapper SHA-256
  `d84c49f517024f2866e2ab5bf347183489fbc83a6cd554ce713a8d956b0b1127`
  inverse-reconstruct exact unbound template SHA
  `c04c288501b95f0408c21e9e5cb4eb6bfcb1af159b82572c3eb26d63acf17492`
  với đúng ba launch replacement. Actual source closure chỉ đổi hai auditor/test
  hash đúng correction addendum; mọi scientific source hash khác exact protocol.
- Fresh v2 package chỉ có bound wrapper `20,146` byte và unchanged metadata
  `630` byte SHA-256 `489127ab...72e6`, không directory thừa; bound-wrapper
  `py_compile` pass. Final prelaunch audit SHA-256
  `d25b506f269a6bdfad17d922e59ed5bd0a6128063bfb3507e4693d9a323b7861`
  có `status=PASS`, `authorized_launch=true`.
- Chưa launch version 2 và chưa mở lại output v1/validation GT. Phải commit/push
  exact binding/audit/log này, fetch lại central/collaborator và xác nhận không
  collision trước một correction rerun duy nhất. Consumer/test tiếp tục khóa.

### S6 kernel version 2 — correction rerun launched, chưa status-poll (2026-08-02)

- Exact v2 binding/prelaunch commit
  `ba9fdf1bf8bbe15a5451fe8a79d63cd05734ab14` đã visible central; collaborator
  vẫn `dc00062`, không active collision. Package re-verify đúng hai file/hash
  trước launch.
- Kaggle CLI đã trả `Kernel version 2 successfully pushed` lúc
  `2026-08-02T06:59:58.1825238Z` cho cùng private kernel. Đây là rerun duy nhất
  của implementation-only independent-auditor numeric correction; scientific
  source/model/loss/recipe/prediction protocol không đổi từ v1.
- Launch audit SHA-256
  `70f4af2a21f9f4cb109251ae5d6df317b5b5f16ca690f431c627e143a36d6b9e`.
  Không post-launch status-poll, không monitor, không output/GT access. Claim S6
  tiếp tục **ĐANG LÀM** chờ một bounded terminal check; consumer và test khóa.

### S6 v2 bounded running check (2026-08-02)

- Đúng một status check lúc `2026-08-02T07:02:31.4776114Z`, khoảng `153` giây
  sau launch, thấy version 2 vẫn `KernelWorkerStatus.RUNNING`. Không repeat poll,
  không monitor, không partial log/output access và không mở GT/test/consumer.
  Claim tiếp tục **ĐANG LÀM** chờ nhịp terminal hợp lý sau.
- Nhịp kế tiếp chỉ kiểm tra đúng một lần lúc `2026-08-02T07:04:56.2189867Z` và
  version 2 vẫn `KernelWorkerStatus.RUNNING`. Không kiểm tra lần hai, không tạo
  monitor, không đọc partial log/output và không mở validation GT, consumer hay
  BTXRD test. Claim tiếp tục **ĐANG LÀM**.
- Nhịp kế tiếp chỉ kiểm tra đúng một lần lúc `2026-08-02T07:06:14.9917372Z` và
  version 2 vẫn `KernelWorkerStatus.RUNNING`. Không kiểm tra lần hai, không tạo
  monitor, không đọc partial log/output và không mở validation GT, consumer hay
  BTXRD test. Claim tiếp tục **ĐANG LÀM**.

### S6 kernel version 2 — terminal COMPLETE, independent GT-blind audit PASS (2026-08-02)

- Theo tín hiệu terminal của người dùng, đúng một status check lúc
  `2026-08-02T07:07:54.8045824Z` xác nhận version `2` của private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-label-granularity-s6-v1` là `COMPLETE`.
  Không có repeat poll/monitor. Inventory downloader của project tải đủ
  `1,501/1,501` output file (`154,476,655` byte), không `.part`/temp; direct log
  `13,546` byte có SHA-256
  `34e3089512005e3d2abdc3d2e2f00edfa7462e6fe5e76754bb38fc17533a7873`.
- Runtime log và wrapper audit xác nhận exact checkout
  `e10db06537eedb4314faa36f173e96d0643b54ab`, scientific source
  `543ee89654a0ed00e80ded16924a760585337924`, protocol
  `f4e17d24dfab36f01526550c7dc306fc7549494acc4545153454c61ae926bfc3`,
  bound wrapper `d84c49f5...b1127`, split/cache/baseline đúng frozen contract,
  `49` tests pass, đúng `Tesla T4 x2`, `2,981/371` train/validation record và đủ
  `16` epoch. Training chỉ dùng image-level labels.
- Producer giữ nguyên exact pair freeze từ v1
  `203dc6435b661b7410331f3eefca20c14888bcd34890b5228b4c35a0cecd36e2`;
  control/hierarchy freeze là `7cff6cc8...c84` / `fa88dd2d...43c`. Wrapper audit
  SHA-256 `195ad72f...35fa3`; embedded independent audit SHA-256
  `3eea80fd...4e00` đều pass trước GT.
- Independent local rerun trên chính output vừa tải và exact accepted cache,
  baseline, split tái lập đủ `742` candidate-score payload + `742` physical map;
  map delta của cả hai arm bằng `0`, max logit delta lần lượt
  `7.6293945e-6/9.5367432e-6`. Local audit SHA-256
  `b7ad03afdec41e5bc33d118107b921283590b9e07fbdf74abd5598b3b1304ca2`
  có status `PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_PASS`.
- Diagnostics vẫn nonblocking và chưa dùng chọn model: hierarchy đổi `163/371`
  lựa chọn; absolute count/probability Spearman control/hierarchy
  `0.50718233/0.49796698`. Audit terminal tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_label_granularity_s6_v1/kernel_version2_terminal_gt_blind_audit.json`,
  SHA-256
  `e2280b1d30cab29f904456a38460e7efd6b6ad414e0286e0802ea8fdc2218176`.
  Tới boundary này `validation_gt_read=false`, subtype validation không dùng
  routing, consumer chưa train, collaborator output không truy cập và BTXRD test
  chưa đọc. Sau khi evidence/log này visible central, exact frozen evaluator mới
  được phép đánh giá hai arm theo seed correction `20261101`; claim vẫn
  **ĐANG LÀM** và chưa có Dice.

### S6 kernel version 2 — terminal efficacy result, hierarchy bị loại (2026-08-02)

- Sau khi terminal GT-blind evidence đã visible central tại commit
  `1f2d5e57ac309752f7255bbdb021ca6172c07c7b`, frozen evaluator canonical-LF
  SHA-256 `ccc3a493...b084` mới đọc validation GT cho từng arm theo exact cohort
  `371/184/187`, subgroup `94/72/18`, `10,000` complete-group bootstrap và cùng
  corrected seed `20261101`. Python-3.9 strict-zip fail-closed shim được tái dùng
  byte-identical tại SHA-256 `dcf88d82...c396a`; evaluator/comparator không sửa.
- `coarse_control` Dice overall/small/medium/large là
  `0.23281193 / 0.10795317 / 0.36617726 / 0.35139083`, `58` complete miss;
  per-image/evaluation-audit SHA-256
  `1115ad60...3031` / `ddf2e904...9ae0`. So với accepted Geometry-v3, delta
  `-0.01267045 / -0.00912741 / -0.01095825 / -0.03802182`; mọi CI95 cắt 0.
- `hierarchical_entropy_routed` Dice là
  `0.23836084 / 0.11142896 / 0.36300375 / 0.40265566`, `63` complete miss;
  per-image/evaluation-audit SHA-256
  `b81286c6...ec7f` / `d92ee15a...e19f`. So với Geometry-v3, delta
  `-0.00712155 / -0.00565162 / -0.01413177 / +0.01324302`; CI95 tương ứng
  `[-0.02153718,+0.00560105] / [-0.02426985,+0.00801769] /
  [-0.04106158,+0.00813037] / [-0.02928596,+0.05039340]`.
- Frozen matched comparator không mở lại GT; comparison/per-image SHA-256
  `d37be43f49fae78553037d9f2640755fc54d4b643e5f2aecc6cd9dcd39af55a6` /
  `8bd869ca89c1fee696a85cc9cfd92909523c563aa85413bfbb9763cbe5ab092d`.
  Hierarchy trừ control là
  `+0.00554890 / +0.00347579 / -0.00317351 / +0.05126484`; mọi matched CI95
  cắt 0. Nó recover `2` miss nhưng mất `7` overlap, nên complete misses tăng
  `58 -> 63`, tập trung ở small `35 -> 42`.
- S6-specific mechanism gate **FAIL**: overall/small/large mean có hướng tăng,
  nhưng medium giảm và miss tăng. Operational gate cũng **FAIL**: không arm nào
  đạt bất kỳ bộ bốn goal đồng thời; hierarchy vẫn kém Geometry-v3 ở overall,
  small, medium và overall CI lower không dương. Image-level subtype taxonomy có
  thay đổi ranking nhưng không cung cấp positive-instance identity đủ ổn định;
  không promote hierarchy, không rescue/sweep hậu nghiệm và không train consumer.
- Terminal evaluation audit tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_label_granularity_s6_v1/kernel_version2_terminal_evaluation_audit.json`,
  SHA-256
  `6973101bf4b6327b8e7acdcb11d2972e97efe3e4947a45e7ecad9054ba190958`.
  Prediction-freeze/GT boundary giữ đúng; comparator không reopen GT;
  `consumer_trained=false`, `test_evaluated=false`, collaborator output không
  truy cập. Experiment S6 kết thúc **HOÀN THÀNH — GATE FAIL**; goal thesis vẫn
  chưa đạt và successor phải là hypothesis mới không trùng các residual/pooling/
  hierarchy mechanism đã bị loại.

### S6 post-hoc failure analysis — bắt buộc trước successor (2026-08-02)

- Phân tích này chỉ giải thích failure từ prediction/evaluation đã freeze; không
  chọn lại model, không sửa S6 và không mở rescue. Trên `184` tumor, hierarchy
  đổi `93` lựa chọn: `34` cải thiện, `32` xấu đi, còn `27` đổi candidate nhưng
  Dice không đổi. Nó recover `2` miss nhưng mất `7` overlap; toàn bộ overlap mất
  nằm ở small, làm small misses `35 -> 42`. Large có tín hiệu hướng dương
  `+0.051265` (`8` tốt/`1` xấu), nhưng n=`18` và CI vẫn cắt 0; medium giảm.
- Cơ chế chính được định vị ở **subtype identifiability**, không phải entropy
  calibration đơn thuần. Subtype chỉ đúng `86/184=46.74%`, macro recall
  `0.25492`; subtype `5/6/9` recall bằng `0`, subtype `8` chỉ `0.1667`. Khi
  subtype đúng, mean hierarchy-minus-control là `+0.021249`; khi sai là
  `-0.008229`. Tuy vậy entropy strength và subtype confidence gần như không dự
  báo signed Dice (`Spearman 0.06946/0.07869`), nên threshold-route hậu nghiệm
  vừa không được hỗ trợ vừa bị cấm.
- Training history cho thấy multi-task trade-off rõ: epoch-16 binary loss của
  hierarchy/control là `0.10578/0.04027`, consistency `0.30188/0.05960`, drift
  `4.09231/2.51199`. Hierarchy fit pathology/subtype loss xuống
  `0.08222/0.42876` nhưng localization ranking chỉ tăng Spearman
  `0.40947 -> 0.41747`, regret `0.17626 -> 0.17071`. Auxiliary taxonomy đã học
  nhưng không đồng nhất với extent-sensitive instance quality.
- Kết luận failure: cùng candidate-bag objective đang phải vừa nhận subtype vừa
  chọn extent nên rơi vào lazy MIL; entropy không phải reliability observable.
  Không chạy lại weight/epoch/threshold/fusion/subtype router. Successor chỉ hợp
  lệ nếu thêm reliable instance-level identity từ image labels hoặc global
  context được freeze riêng, và phải khác T1 self-paced residual/S4 cluster.
- Deep-search nguồn sơ cấp và giới hạn chuyển giao:
  INS trực tiếp huấn luyện instance classifier bằng true-negative bags,
  weakly-supervised contrastive và prototype-refined pseudo labels
  (https://arxiv.org/abs/2307.02249); Pixel-to-Prototype Contrast hỗ trợ
  cross-view prototype consistency nhưng cảnh báo pseudo label yếu có thể làm
  degeneration
  (https://openaccess.thecvf.com/content/CVPR2022/html/Du_Weakly_Supervised_Semantic_Segmentation_by_Pixel-to-Prototype_Contrast_CVPR_2022_paper.html).
  DTFD-MIL pseudo-bag nhắm WSI cực nhiều instance, không sửa identity cho bag
  BTXRD nhỏ
  (https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.html).
  MHIM-MIL masking hard instance bị hoãn vì small lesion có thể chỉ có một
  proposal hữu ích
  (https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Multiple_Instance_Learning_Framework_with_Masked_Hard_Instance_Mining_for_ICCV_2023_paper.html).
- Dossier machine-readable tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_label_granularity_s6_v1/kernel_version2_posthoc_failure_analysis.json`,
  SHA-256
  `7f30100cb2bcfb35782359a76052a9085903b5c616d12bdd7932618fd329db7e`.
  Chưa đăng ký/implement/launch experiment mới; prediction S6 không đổi,
  consumer/test vẫn khóa và collaborator output không truy cập.

### Đồng bộ collaborator `139ba76` và khóa collision sau S6 (2026-08-02)

- Đã fetch `origin/research-wsss-improvement` tại
  `d49572664ed523c8e626c64f1dcf9efaf0c18d17` và đọc toàn bộ log trung tâm;
  đã fetch/read toàn bộ Git-log delta `dc00062..139ba76` của
  `origin/codex/research-sync-20260731`. Chỉ đọc bằng chứng/design trong Git,
  không truy cập Kaggle hay output collaborator.
- Collaborator mới thêm
  `EXP-20260802-codex-rich-gallery-cross-view-cowitness-pair-v1` ở trạng thái
  **PRELAUNCH**, chưa có terminal efficacy/Dice. Arm full dùng same-heuristic-
  group different-view tumor co-witness đối chiếu capacity-matched different-
  group control trên cùng `384` rows/full bag; baseline bất biến là rich-gallery
  G1+upstream equal percentile-rank Dice `0.2887294867`. Protocol SHA-256 là
  `37826b209afd8c897b07d91f20bcbf133cf06be3d175b206dad14dc19a36488f`;
  pair-manifest SHA được log collaborator rút gọn `0950ed50...`.
- Bằng chứng feasibility trước claim vẫn là âm/yếu: after matched control,
  cross-view partial correlation median chỉ `0.014121` và oracle-vs-baseline
  pair accuracy `18.75%`; raw same-group support chủ yếu phản ánh anatomy. Vì
  experiment mới chưa terminal nên central không adopt co-witness như cải thiện,
  nhưng coi PRELAUNCH là claim liên quan đang được giữ và sẽ không launch một
  cross-view/co-witness/cross-view-representation experiment cạnh tranh.
- Failure-analysis gate của S6 không đổi: hierarchy fail do subtype
  identifiability thấp, multi-task trade-off và lazy MIL; entropy/confidence
  không phải reliability observable. Successor central chỉ được mở sau failure
  analysis này và phải khác cross-view collaborator, T1 self-paced contrast,
  S4 cluster, S6 hierarchy và các frozen relation/area proxies đã bị loại.
- Theo yêu cầu người dùng, quy tắc phối hợp được bổ sung vào `AGENTS.md`: mọi
  `LỖI`/reject/fail gate phải có phân tích nguyên nhân định lượng, ghi/push log
  trước khi đăng ký hoặc launch hướng kế nhiệm. Bước đồng bộ này không phải một
  experiment: chưa claim/implement/launch, chưa mở scientific input/prediction/
  validation GT/consumer/test và không chạy compute nặng.

### Chọn successor sau failure-analysis gate S6 — direct instance self-training (2026-08-02)

- Chuỗi negative evidence loại một bản sao INS/T1 đơn giản: v6 dùng normal-bag
  negatives + detached latent winner nhưng Dice chỉ `0.21789918`; T1 dùng OOF
  top-1 self-paced contrastive residual, pass count/AUROC/view gate nhưng Dice
  `0.24282104`, thấp hơn Geometry-v3; S4 cluster và S6 subtype hierarchy cũng
  fail. Vì vậy không dùng lại hard top-1 target, prototype/winner confirmation,
  bag BCE/attention pooling hoặc entropy router dưới tên mới.
- Primary source mới được đọc đầy đủ ở cấp mechanism là Ma et al., *Rethinking
  Multiple Instance Learning: Developing an Instance-Level Classifier via
  Weakly-Supervised Self-Training*, arXiv:2408.04813,
  https://arxiv.org/html/2408.04813 . Paper chuyển MIL thành semi-supervised
  **instance classification**: mọi instance trong negative bag là true negative;
  mọi instance trong positive bag là unlabeled; pseudo label mềm được entropy-
  projected theo một global positive-mass constraint, sau đó local constraint
  bắt buộc mỗi positive bag có ít nhất một positive instance. Classifier được
  cập nhật trên toàn bộ instance, không chỉ một easy winner. Trên CAMELYON16,
  paper báo adaptive mass `0.5 -> mu`, global+local constraints và soft labels
  cần thiết; external grid `mu={0.10,0.15,0.20,0.25}` có best reported ở `0.15`.
  Đây chỉ là rationale/giá trị cố định từ nguồn ngoài, không chuyển metric.
- INS (Qu et al., arXiv:2307.02249,
  https://arxiv.org/abs/2307.02249 ) xác nhận true-negative bags và direct
  instance classifier/prototype refinement có thể giảm lazy MIL. Tuy nhiên
  IWSCL/prototype pseudo-label implementation đầy đủ không được chọn vì phần
  top-instance/self-paced contrast đã gần T1 và sẽ không tạo giá trị thông tin
  đủ mới. MIL-SSL global+local soft assignment mới là delta chưa chạy trong log.
- Successor tĩnh dự kiến là một **zero-initialized residual trên exact accepted
  Geometry-v3 candidate logits**, nhưng loss hoàn toàn ở cấp instance. Epoch-wise
  soft targets dùng toàn bộ train-tumor candidates với global weighted mass
  schedule `0.5 -> 0.15` trong `20` epoch và local one-positive-per-bag; toàn bộ
  train-normal candidates có target `0`. Trọng số equal image -> family ->
  candidate ngăn count/family shortcut. Tổng fit dự kiến `40` epoch, seed `42`,
  không early stop/sweep. Original/flip consistency và small drift anchor được
  giữ như integrity regularization, không phải target generator.
- Để cô lập selector khỏi image classification/count shortcut, inference dự
  kiến chỉ đổi argmax candidate bằng base+instance residual; exact accepted bag
  probability và physical mask gallery được giữ nguyên. Baseline identity và
  primary maps sẽ freeze thành matched pair trước GT. Đây khác T1 (hard OOF
  subset + supervised contrastive + learned bag probability), v6/S6 (bag-level
  loss) và collaborator `139ba76` (cross-view co-witness/rich gallery).
- Mục này mới là source-backed static design selection, **chưa phải claim**.
  Chưa mở real cache/data, chưa fit/prediction/GT/consumer/test. Phải implement
  và synthetic-test data-independent, audit collision lại, rồi mới đăng ký
  `ĐANG LÀM`/push trước mọi real cache load hoặc Kaggle launch. Nếu static audit
  cho thấy objective trùng hoặc constraint không xác định được fail-closed thì
  retire trước claim thay vì chi compute.

### S7 primitive/training static readiness — chưa claim, chưa real input (2026-08-02)

- Data-independent implementation đã được push tại commit
  `3e1b0dfe71ad0e8b0aba31e72b839f0be30ef5d7`. Design/model/training/two test
  file canonical-LF SHA-256 lần lượt là
  `d7dd5bf8...e1d5` / `bf45579c...f498` / `fef334bb...4ac6` /
  `4d8bd618...da77` / `b65b9f29...de67`. Primitive khóa exact zero-init
  Geometry-v3 identity, weighted Bernoulli I-projection bằng float64 bisection,
  local one-positive-per-tumor-bag, all-normal target zero, equal image/family/
  candidate mass, original/flip instance loss và accepted bag-probability
  preservation.
- Focused new suite pass `12/12`; new+S6 regression pass `24/24`; `py_compile`
  và `git diff --check` pass. Full repository dưới exact audited Python-3.9
  strict-zip shim đạt `605 passed, 1 failed`; failure duy nhất là known unrelated
  B1 CRLF/LF working-tree hash cho `project/datasets/btxrd.py` (expected Git-LF
  `96f5abaa...`, observed CRLF `dcb509a3...`) đã được log từ S6, không liên quan
  S7 và không được sửa/nới.
- Failure-analysis gate được áp dụng cho hai static test error trước khi sửa:
  (1) equal-family weights bị cast float32 trước projection guard `1e-10`, làm
  tổng lệch vài ULP; correction giữ weight audit/projection float64 và chỉ cast
  tại Torch boundary; (2) fixture gọi keyword-only `smooth_mil_pool` temperature
  theo positional; correction chỉ dùng `temperature=0.2`. Cả hai xảy ra trên
  synthetic input, không đổi objective/hyperparameter/scientific output.
- Static readiness artifact
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_static_readiness.json`
  có SHA-256
  `c6d25b16941e92ee6b99848a34eebdf2adf9ae8c8483d6e3fdbae64e1298f6b3`.
  Nó ghi rõ `STATIC_READY_NO_CLAIM_NO_REAL_INPUT`: chưa mở selector cache/radiograph,
  chưa fit/prediction/validation GT/consumer/test và chưa launch Kaggle.
- Bước tiếp theo bắt buộc là commit/push readiness này, fetch lại central và
  collaborator, đọc collision claim mới. Chỉ nếu scope còn unique mới được mở
  một claim S7 `ĐANG LÀM` đầy đủ; runner/protocol/auditor/wrapper và mọi real
  cache action đều còn bị khóa.

### Đồng bộ collaborator `29a3ed6` và collision gate trước claim S7 (2026-08-02)

- Đã fetch `origin/research-wsss-improvement` tại
  `6c568febad9298b158cb62729292c1431590ee97` và đọc Git-log delta đầy đủ
  `139ba7628df884941d768c1fb29dddaa28156648..29a3ed6d1f6fe99e1fdb6d6ae3c20c562474b7fb`
  của `origin/codex/research-sync-20260731`; không truy cập Kaggle hay output của
  collaborator. Claim cross-view co-witness vẫn **PRELAUNCH**, chưa có terminal
  efficacy/Dice. Protocol transport đã được khóa lại bằng Git-archive SHA-256
  `20e6ff644b12f0406deedfd83026111c3a3808ce7f70b90c1675c4a0b772c9eb`;
  scientific fields không đổi.
- Feasibility mới `EXP-20260802-codex-rich-gallery-scale-conditional-feasibility-v1`
  cho thấy candidate area-rank expert có dấu đúng theo GT-size: beta tốt nhất
  small/medium/large là `-1.0/0/+0.5`, subgroup Dice
  `0.181215/0.435229/0.501086`, true-group routed overall `0.311904` và
  per-image three-expert oracle `0.350692`. Đây là retrospective GT-only
  mechanism evidence, không phải deployable selector và không được adopt như
  efficacy terminal.
- Ba gate annotation-free dựa vào baseline-selected area, top-five median area
  và median best-per-source area chỉ đạt `0.278291/0.271403/0.268845`, đều thấp
  hơn immutable rich-gallery baseline `0.288729`; size accuracy chỉ
  `42.9-45.1%`, area/true-size Spearman `0.043-0.096`. Root cause được chuyển
  giao là area-only routing tự củng cố lỗi: small lesion bị over-segment đã trông
  như large trong proposal-area space. Summary/per-image/audit SHA-256 phía
  collaborator lần lượt là
  `47ec2c711a42a5ca2efe6e2d4419bdc71bd8f4287df00297a2c2d414a70c5430`,
  `2ed22cc34a64232a08380acd8136cebdae0e24fb758659c42ef0a8112ed98ac0`,
  `970fbd271a771b20c048fd6736487c44ce4c9a13e55cb9027a84ce0fcf97300c`.
- Collision decision: central không launch cross-view/co-witness, area-expert
  hoặc size-gating competitor. S7 vẫn unique: dùng same-gallery accepted cache,
  true-negative bags và global+local soft targets để học **candidate identity**
  trên toàn bộ instance; không dùng cross-view pair, không dùng area expert/gate
  và giữ nguyên exact accepted bag probability. Kết quả collaborator làm mạnh
  thêm lý do khóa count/area shortcut của S7, nhưng không thay đổi objective hay
  được quảng bá như một cải tiến đã chứng minh. Chưa claim/real-input/launch ở
  boundary đồng bộ này; consumer/test/validation GT vẫn khóa.

### EXP-20260802-codex-s7-global-local-instance-v1

- **Owner/status/time:** Codex main task trên `research-wsss-improvement`;
  **ĐANG LÀM**; đăng ký lúc `2026-08-02T14:59:36+07:00`. Registration base là
  `72977b9f18f21f2a4b2c3481940bb26deffaa380`; exact registration commit sẽ
  được khóa bằng note kế tiếp sau khi commit/push. Không được mở real cache,
  fit, prediction hay launch cho đến khi claim này hiện diện trên branch trung
  tâm.
- **Kế thừa và không trùng:** kế thừa exact accepted same-gallery Geometry-v3
  candidate supply/base logits và negative evidence của R1/R2/S1/R3/R4/S3/S4/
  T1/S5/S6; trực tiếp xử lý lazy-MIL/positive-instance-identifiability đã được
  định lượng ở failure analysis S6. Không kế thừa các cơ chế đã reject như hard
  top-1/self-paced contrast, prototype/cluster, subtype hierarchy, entropy route,
  bag BCE/attention pool hoặc area/count heuristic. Không dùng cross-view pair,
  co-witness hay size expert/gate nên không trùng claim collaborator
  `EXP-20260802-codex-rich-gallery-cross-view-cowitness-pair-v1` PRELAUNCH và
  feasibility scale-conditional tại `29a3ed6`.
- **Hypothesis và delta khoa học:** theo Ma et al. MIL-SSL
  (https://arxiv.org/html/2408.04813), chuyển selector từ bag-level MIL sang
  direct all-instance classification: mọi candidate của normal bag là true
  negative; candidate trong tumor bag nhận soft target bằng weighted Bernoulli
  I-projection thỏa global positive-mass rồi local constraint ép ít nhất một
  positive/case. Việc train mọi candidate thay vì chỉ latent winner phải giảm
  lazy-MIL và selector regret mà không cần spatial GT. INS
  (https://arxiv.org/abs/2307.02249) chỉ hỗ trợ nguyên lý true-negative bag/direct
  instance classifier; prototype/self-paced phần còn lại không được dùng vì gần
  T1 đã fail.
- **Frozen design:** scalar residual MLP `1156 -> 128 -> 1`, GELU/dropout `0.1`,
  exact zero-init trên accepted candidate logits và descriptor center trong mỗi
  valid bag. Equal image -> family -> candidate weight; normal target `0`; tumor
  soft target global+local; global mass schedule `0.50 -> 0.15` trong `20` epoch;
  tổng `40` epoch, seed `42`, AdamW `lr=3e-4`, weight decay `1e-4`, batch `16`,
  flip consistency `0.10`, residual drift `0.001`, không early stop/sweep. Inference
  chọn argmax bằng average original/flip base+residual nhưng giữ nguyên exact
  accepted Geometry-v3 bag probability, cô lập candidate ranking khỏi
  classification/count/area shortcut. Immutable baseline identity và primary
  phải thành matched physical prediction pair.
- **Exact input/provenance:** split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  selector-cache freeze
  `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`;
  cache manifest
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`;
  baseline checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`;
  baseline freeze
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`;
  evaluator-only corrected baseline per-image SHA-256
  `a26143d02bacd01ec27c9d7fbaf3e20691d9974b2ee60f27eb40a88f3403605f`;
  cohort `2,981 train / 371 validation`, descriptor `1156`. Validation polygons
  và evaluator-only table không phải training input và chỉ được mở sau pair
  freeze/audit.
- **Compute/output:** static/synthetic locally; mọi real-cache fit/inference chỉ
  private Kaggle T4x2/P100. Producer phải freeze checkpoint/history, per-epoch
  target SHA/mass/local diagnostics, complete baseline/primary candidate-score
  payload, `742` physical maps, exact pair-freeze và safety manifest. Independent
  GT-blind auditor phải tái lập source/input/split/cache/base score, zero-init
  identity, target projection/weights, full score/map equality và accepted bag-
  probability preservation trước khi evaluator được phép đọc validation GT.
- **Predeclared gates:** mechanism pass chỉ khi primary-minus-accepted baseline
  có overall và small mean Dice dương, medium/large không giảm và complete misses
  không tăng. Operational pass chỉ khi đồng thời đạt
  `0.34024039 / 0.17895493 / 0.51244178 / 0.49370336`, overall matched bootstrap
  CI95 lower `>0`, không subgroup regression/miss increase. Chỉ operational pass
  mới cho phép train consumer. Nếu fail/reject/error: freeze kết quả, hoàn tất
  quantitative failure-analysis gate và push trước mọi successor; không
  post-hoc rescue/mass/epoch/threshold/fusion/area sweep. BTXRD test luôn khóa.
- **Static readiness at registration:** design/model/training/tests đã push ở
  `3e1b0dfe71ad0e8b0aba31e72b839f0be30ef5d7`; readiness artifact SHA-256
  `c6d25b16941e92ee6b99848a34eebdf2adf9ae8c8483d6e3fdbae64e1298f6b3`
  và log commit `6c568febad9298b158cb62729292c1431590ee97` xác nhận focused `12/12`,
  S6-regression `24/24`, không real input/prediction/GT/consumer/test. Runner,
  protocol, auditor và Kaggle wrapper chưa được claim là ready ở registration.

### S7 registration visibility lock (2026-08-02)

- Exact registration commit
  `d316269b141664b9881eea601491d0fe45227d80` đã push và fetch lại byte-visible
  trên `origin/research-wsss-improvement` trước mọi real-cache action. Claim
  `EXP-20260802-codex-s7-global-local-instance-v1` hiện **ĐANG LÀM**; bước kế là
  hoàn thiện runner/protocol/independent auditor/wrapper và static preflight,
  chưa launch, chưa mở real cache/validation GT, chưa prediction, consumer/test
  vẫn khóa.

### S7 runner/target-snapshot/auditor static implementation (2026-08-02)

- Producer tĩnh mới `project/run_mask_bag_global_local_instance_s7_pair.py`
  khóa recipe `40` epoch, lưu đủ `40` physical pre-epoch target snapshot
  (current logits/soft targets/equal-family weights/offsets/IDs/labels), checkpoint,
  history, matched Geometry-v3-identity/S7 score+map pair và cùng exact accepted
  bag probability. Validation được shard thật qua hai T4; wrapper chưa viết và
  chưa launch.
- Independent GT-blind auditor mới
  `project/audit_mask_bag_global_local_instance_s7_output.py` tự cài lại
  float64 96-step Bernoulli projection, equal image/family/candidate weights,
  local argmax constraint và target digest; nó đọc physical snapshots để audit
  cả `40` target assignment, rồi tái tạo final candidate logits/winner/map từ
  exact cache + baseline + checkpoint. Auditor không import evaluator/GT và chỉ
  pass nếu hai arm có bag probability byte-value như nhau, `742` score payload +
  `742` map hiện hữu và mọi safety field còn false.
- Training helper chỉ bổ sung base-logit attachment không cần subtype và callback
  physical snapshot; objective/hyperparameter đã đăng ký không đổi. Canonical-LF
  SHA-256 của training/runner/auditor/training-test/runner-test lần lượt là
  `2e2803d2...bece` / `76d0bae2...31b8` / `b64ae0a2...a138` /
  `5f1f73a3...6799` / `55ee452d...864`.
- Static execution boundary: lệnh đầu dùng accidental system Python 3.13 và dừng
  ở pytest collection vì environment đó không có `numpy`; `py_compile` đã pass,
  không test/model/data được chạy. Đây là environment/implementation boundary,
  không phải hypothesis failure và không được dùng làm evidence S7. Sau khi xác
  minh exact established env
  `C:/Users/USER/miniconda3/envs/btxrd-pseudomask/python.exe` là Python `3.9.23`,
  NumPy `1.23.5`, Torch `2.1.2+cu118`, pytest `8.3.5`, cùng source pass
  `py_compile` và focused `17/17`. Không sửa scientific code để né dependency.
- Boundary hiện tại vẫn static only: chưa load selector cache/radiograph, chưa
  fit/prediction/validation GT, chưa consumer/test và chưa Kaggle. Bước tiếp theo
  là commit/push exact source, sau đó mới tạo immutable protocol trên source
  commit đó, test wrapper/binder fail-closed và khóa prelaunch provenance.
- Sau source commit `1896c3576127cd7a3b04fb02ee666960370e5204`, một dedicated
  auditor test bổ sung kiểm tra independent projection khớp producer byte-exact,
  safety/path traversal fail-closed và float16 map serialization; focused suite
  tăng lên `20/20` pass. Test này vẫn synthetic/data-independent và sẽ được đưa
  vào exact scientific-source commit kế tiếp trước protocol freeze.

### Contingency extent-conditional sau S7 — insight, chưa claim (2026-08-02)

- Người dùng chỉ ra đúng structural mismatch: small `<1%` chủ yếu over-segment,
  medium `1-<5%` tương đối đúng extent nhưng sai vị trí/candidate, còn large
  `>=5%` under-segment; vì vậy một phép area/extent correction toàn cục không có
  dấu hợp lý cho cả ba nhóm. Git-log collaborator tại `29a3ed6` hỗ trợ định lượng:
  true-GT-group beta small/medium/large `-1.0/0/+0.5` cho Dice
  `0.181215/0.435229/0.501086`, overall `0.311904`; per-image three-expert oracle
  `0.350692`.
- Tuy nhiên nhãn nhóm extent được suy từ segmentation GT nên tuyệt đối không được
  dùng làm train/inference router. Ba deployable-looking area gate đã chỉ đạt
  `0.278291/0.271403/0.268845`, dưới rich-gallery baseline `0.288729`; small bị
  over-segment trông như large làm area-only gate tự củng cố lỗi. Vì vậy không
  được mở hard GT-size routing hay post-hoc subgroup rules.
- Contingency hợp lệ nếu S7 terminal fail hoặc chỉ sửa identity mà còn extent
  residual là một **shared identity selector + ba signed extent experts + soft
  annotation-free gate**. Expert small co support, medium giữ extent và ưu tiên
  vị trí/semantics, large mở support; gate chỉ được dùng appearance, evidence
  density, cross-source consensus và uncertainty đã có trước GT, không dùng
  selected-mask area đơn độc. Nó phải có matched shared-selector control,
  predeclare/freeze trước GT và chứng minh gate tốt hơn control.
- Đây chưa phải successor claim: extent expert oracle hiện vẫn không giải quyết
  medium goal (`0.435229 < 0.51244178`), nên ưu tiên hiện tại vẫn là S7 direct
  instance identity. Nếu S7 fail, bắt buộc phân tích riêng learning signal,
  target projection, rank transfer và subgroup/miss pattern trước khi quyết định
  có đủ bằng chứng mở conditional mixture-of-experts hay không. Không implement/
  launch contingency, không dùng nó rescue S7 và không thay đổi protocol S7.

### S7 immutable protocol freeze (2026-08-02)

- Exact scientific source commit
  `0e524807937e6fb6effde1649993825f3923c43f` đã push trước protocol; nó chứa
  producer, independent auditor và dedicated tests nhưng chưa có real-data
  action. Immutable protocol
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1.json`
  có SHA-256
  `81fbb2f40af3a49e4653a15d298858c973e88524dea06fc42c9095cec55579a1`,
  status `FROZEN_PRELAUNCH`, registration commit `d316269...27d80` và khóa exact
  input/source/training/output/evaluation/safety contract.
- Protocol khóa `40` epoch và `40` physical target snapshots, mass `0.50 -> 0.15`
  trong `20` epoch, `96` float64 projection iterations, local constraint, equal
  image/family/candidate weight, exact accepted bag probability và matched
  Geometry-v3 identity/S7 maps. Post-freeze seed là `20261202`; không rescue,
  sweep, area/subgroup routing hay consumer trước operational pass; fail bắt
  buộc failure analysis trước successor.
- Protocol/source hash test cùng primitive/training/runner/auditor synthetic
  suite pass `21/21`; JSON parse và `git diff --check` pass. Đây vẫn là static
  readiness: chưa load real cache/radiograph, chưa fit/prediction/GT, chưa
  consumer/test và chưa launch. Protocol và test phải được commit/push trước khi
  wrapper template được tạo/bind.

### S7 unbound Kaggle wrapper static readiness (2026-08-02)

- Fail-closed wrapper template
  `project/kaggle_wrappers/run_mask_bag_global_local_instance_s7_v1.py` khóa
  kernel private
  `itsthang333/btxrd-rad-dino-mask-bag-global-local-instance-s7-v1`, exact source
  commit `0e52480...c43f`, protocol SHA-256 `81fbb2f4...79a1`, split/cache/
  baseline transport hashes và T4x2 guard. Template hiện cố ý
  `KERNEL_VERSION=0`, `LAUNCH_BINDING_READY=false`, `CHECKOUT_COMMIT=UNBOUND`,
  nên không thể launch trước one-time binding.
- Wrapper clone exact checkout, verify toàn bộ protocol source hashes, dựng split
  CRLF đúng frozen SHA, tìm duy nhất exact cache/baseline, chạy focused tests,
  producer, rồi independent GT-blind auditor. Chỉ sau auditor pass nó mới tự
  kiểm đủ `40` target snapshot, `742` score payload, `742` map và ghi
  `wrapper_output_audit.json`; không gọi evaluator/comparator hay validation GT.
- Wrapper/test canonical-LF SHA-256 là
  `ea51733fbf2a0d0a35db55678208c23fc8d360eeecdce887320a6f655c446307` /
  `a227ff01a7e3a701712e171e5bec2432f1b752493e447786da342b5e375fe89b`.
  `py_compile`, wrapper+protocol+S7 focused `23/23` và `git diff --check` pass.
  Chưa bind/launch, chưa real input/prediction/GT/consumer/test.

### S7 one-time wrapper binder static readiness (2026-08-02)

- Binder `project/bind_mask_bag_global_local_instance_s7_wrapper.py` khóa exact
  unbound template SHA-256 `ea51733f...6307`, protocol SHA-256
  `81fbb2f4...79a1` và scientific source `0e52480...c43f`; chỉ thay đúng ba token
  `KERNEL_VERSION`, `LAUNCH_BINDING_READY`, `CHECKOUT_COMMIT`, rồi inverse-
  reconstruct template byte-exact. Nó cũng kiểm template/protocol/source hashes
  trực tiếp từ execution checkout và ancestor relation trước khi ghi output.
- Binder/test canonical-LF SHA-256 là
  `1070755a1e74223d7281f0460c2505064fd52406b57ff3d340a06b434b580517` /
  `ed40157eff7d35d3d0e7c2dd3c81b8af9064539e53903a9747751feeac881932`.
  Binder-focused `5/5`, toàn S7 primitive→binder `25/25`, `py_compile` và
  `git diff --check` pass. Chưa tạo bound wrapper/metadata, chưa launch hay mở
  real input; bước kế phải commit/push binder rồi fetch/collision check trước
  final checkout binding.

### S7 final unbound prelaunch regression/collision audit (2026-08-02)

- Binder đã push tại `682a97cc872dd32d04877d976984275a389d55c2`.
  Sau đó fetch lại central cùng collaborator `29a3ed6...b7fb`; collaborator không
  tiến thêm và không có claim S7/collision. Worktree trước audit sạch; không truy
  cập output/Kaggle collaborator.
- Full repository dưới exact fail-closed Python-3.9 strict-zip shim SHA-256
  `dcf88d82...c396a` đạt `618 passed, 1 failed`. Failure duy nhất vẫn là known
  unrelated B1 raw working-tree CRLF hash của `project/datasets/btxrd.py`:
  expected Git-LF `96f5abaa...4844`, observed CRLF `dcb509a3...e3d9`. Boundary
  xảy ra trong protocol-source hash test B1, file đó không được S7 import/sửa;
  không nới test hay thay line ending và không coi là scientific evidence.
- Machine-readable prelaunch readiness tại
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_prelaunch_readiness.json`.
  Exact SHA-256 là
  `8225ac537b7e7d186a8d489264454856ddff563cd97df4c20f30bcd3ad53e7ea`.
  Nó khóa focused `25/25`, full-suite boundary, exact source/protocol/template/
  binder hashes, `40/742/742` output contract và toàn bộ safety=false. Status
  `PRELAUNCH_READY_UNBOUND`: chưa bind/metadata/launch, chưa real cache/fit/
  prediction/GT/consumer/test. Chỉ sau commit/push artifact này mới được bind
  một lần vào exact checkout kế tiếp.

### S7 kernel version 1 final binding — chưa launch (2026-08-02)

- Sau unbound readiness commit `d284caca956f77f331333d9c86d1d6217aa0d55f`
  đã push, one-time binder khóa wrapper version `1` vào chính checkout đó.
  Bound wrapper SHA-256 `2e2d62d3b21a88ef542c910e741ef2c69b259fdb4d0ada8903f3604f67bc69e5`;
  launch-binding SHA-256
  `6948ba6b258c4441a510010b8bb1fceedb89b71d1731be2bf58cf61dd812ab91`;
  inverse reconstruction khớp template và cả `19` scientific source hash khớp
  Git checkout.
- Private Kaggle metadata chỉ attach exact baseline dataset + accepted selector-
  cache kernel, `machine_shape=NvidiaTeslaT4`, internet để clone exact Git;
  metadata SHA-256
  `e3dbf564e72fb3a3dec17c897ba8dccd0ddaa018709487da11e9c1a70684dc26`.
  Bound constant import và `py_compile` pass; kernel vẫn chưa push.
- Final audit tracked tại
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_kernel_v1_final_prelaunch_audit.json`.
  Exact SHA-256 là
  `76d8d4b80eeb2b855f1d1e3c7808cbee7e9963138e5d0afd8f423c0edec69454`.
  Status `FROZEN_PRELAUNCH_READY_TO_PUSH`; nó khóa version/checkout/source/
  protocol/wrapper/binding/metadata/input/output/test/collision contract. Chưa
  load real cache, chưa fit/prediction/validation GT/consumer/test. Phải commit/
  push log + binding + audit này trước đúng một `kaggle kernels push`.

### S7 launch attempt 1 — `SaveKernel` metadata error, chưa có job (2026-08-02)

- Sau prelaunch commit `5df95ef3f140c3e0e3e879ebfa6b301acbb759a7` đã push,
  đúng một lệnh `kaggle kernels push` trả HTTP `400 Bad Request` tại API
  `KernelsApiService/SaveKernel`. Một bounded `kernels list --mine --search`
  ngay sau đó trả `Not found`; không kernel/version/job được tạo, nên chưa mount
  input/GPU, chưa clone source, chưa load cache/fit/prediction và không có kết
  quả khoa học.
- Failure analysis phân loại đây là **transport metadata failure**. Bound wrapper,
  binding và metadata đã parse/compile; input/GPU metadata giống S6 đã accepted.
  Sai khác nổi bật và testable là cả slug
  `btxrd-rad-dino-mask-bag-global-local-instance-s7-v1` lẫn title đều dài đúng
  `51` ký tự, vượt established Kaggle metadata boundary `50`. Kaggle CLI `2.2.3`
  hiện chỉ bắt minimum title tại local nên server trả generic 400; official CLI
  changelog ghi rõ kernel title/slug length validation. Nguồn:
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md và
  https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md .
- Error audit tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_savekernel_error_audit.json`.
  Exact SHA-256 là
  `7977188aa8ff973c993ee6da420c6ba57dcdb8b4a18ba38f385c593ca1765509`.
  Correction duy nhất được phép sau khi audit/log này push là rút gọn transport
  slug/title/code filename xuống `<=50`, cập nhật wrapper `KERNEL` và rebind
  provenance; scientific source/protocol/input/recipe/output/gate giữ nguyên.
  Chỉ một retry transport được phép; nếu vẫn 400 phải dừng và audit response
  boundary mới, không thử ngẫu nhiên. Claim S7 vẫn `ĐANG LÀM`; GT/consumer/test
  khóa và không polling/monitor.

### S7 SaveKernel transport correction — static, chưa retry (2026-08-02)

- Sau khi error audit `7977188a...5509` đã push tại commit `a598130...d9b1`,
  correction chỉ rút gọn Kaggle transport identity: slug từ
  `btxrd-rad-dino-mask-bag-global-local-instance-s7-v1` (`51`) thành
  `btxrd-rad-dino-mask-bag-instance-s7-v1` (`38`), title tương ứng dài `38`.
  Wrapper `KERNEL`, binder `KERNEL`/template hash và wrapper assertion được cập
  nhật; không đổi producer/model/auditor/scientific source/protocol/input/recipe/
  output/gate.
- Corrected wrapper/binder/test canonical-LF SHA-256 lần lượt là
  `5a9372064ee853d8338e7a3a0e8a21254f38dc036cca5e4f95e4d7541adf41ba` /
  `7d9b0ee07beabb681ea169e26a1552a77d515c556936e55de18ebcb9fc037359` /
  `8c53c20c8841e1be9ae0f4adc1e5a7ed208ac241aabb0a08eefc41446a58643c`.
  `py_compile`, length guard, non-checkout binder/wrapper static `3/3` và
  `git diff --check` pass.
- Correction artifact
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_savekernel_metadata_correction.json`
  có SHA-256
  `18fc583e4257a325412ac7da4ce6b23f082d54af70f1a1afc42be1858fd6faa4`,
  khóa implementation-only scope và đúng một retry. Chưa rebind/metadata/retry,
  chưa kernel/job/data/GPU/prediction/GT/consumer/test. Phải commit/push correction
  rồi mới test binder against exact corrected checkout và tạo retry binding mới;
  không ghi đè binding/error evidence attempt 1.

### S7 transport retry 1 final binding — sẵn sàng push (2026-08-02)

- Correction commit `9c1acaeabfa4754a7054f7dc690bd7b3a6497fbd` đã push và
  byte-visible trên `origin/research-wsss-improvement`. Focused S7
  primitive→binder đạt lại `25/25` trên exact corrected checkout; collaborator vẫn
  ở `29a3ed6...b7fb`, không có claim collision và không truy cập output/Kaggle của
  collaborator.
- Retry dùng thư mục ignored mới, không ghi đè attempt 1. One-time binder khóa
  kernel version `1` vào corrected checkout; bound wrapper SHA-256
  `9c8dcdcd66fe42b2b3f6eeb03f3c82072878e32d1a4a6aefb6d5b5cb6ef2bfc7`,
  launch-binding SHA-256
  `1b40697d2fbe0b6b8bdc476436069dbd5f90166add90ade71dac89c8f3286857`
  và metadata SHA-256
  `519090051f1d26e88ac9b1fade03f8bc44c6ee86473b9e1ed2950fde42c58c63`.
  Slug/title/code filename đều dùng identity ngắn mới (`38/38` cho slug/title);
  private, T4x2, exact dataset/cache inputs và scientific source/protocol giữ
  nguyên.
- Retry prelaunch audit
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_kernel_v1_retry1_final_prelaunch_audit.json`
  có SHA-256
  `68df3d0249d8fe644430394ef0ff8516de5294762d1e1dd583fd0524928ee1ea`.
  Bound `py_compile`, constant import, metadata parse/length guard và
  `git diff --check` đều pass; status `FROZEN_PRELAUNCH_READY_TO_RETRY`.
- Đây là đúng retry transport duy nhất đã cho phép sau attempt 1: chưa push retry,
  chưa kernel/job/input/GPU/cache/fit/prediction/GT/consumer/test. Phải commit/push
  binding, audit và log này trước đúng một `kaggle kernels push`. Nếu server vẫn
  trả 400 thì dừng và mở error audit mới; không thử biến thể metadata ngẫu nhiên.

### S7 transport retry 1 — Kaggle accepted (2026-08-02)

- Sau prelaunch commit `427d5518dae4c2fbea4bf0f9033dd8a1952f1ca4` đã push,
  đúng một lệnh `kaggle kernels push` tại retry directory ignored trả
  `Kernel version 1 successfully pushed` cho private kernel
  `itsthang333/btxrd-rad-dino-mask-bag-instance-s7-v1`:
  https://www.kaggle.com/code/itsthang333/btxrd-rad-dino-mask-bag-instance-s7-v1 .
  Điều này xác nhận correction length `51 -> 38` đã vượt SaveKernel transport
  boundary; chưa phải scientific result và không thay đổi hypothesis/protocol.
- Launch audit
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_launch_audit.json`
  có SHA-256
  `32ddf044675a38a8d5b189d8237fff6f9229f9416024d0ed4df4a6406cdda9dc`.
  Không status check hoặc monitor được tạo trong nhịp này; trạng thái runtime chưa
  được truy vấn. Chưa download output, chưa mở validation GT, chưa consumer/test.
  Claim S7 tiếp tục **ĐANG LÀM** chờ một bounded terminal check ở thời điểm hợp lý.

### S7 bounded runtime check 1 (2026-08-02)

- Đúng một lệnh status lúc `2026-08-02T15:38:10+07:00` trả
  `KernelWorkerStatus.RUNNING` cho version `1`. Không polling thêm, không tạo
  monitor, không download/output/GT/evaluation/consumer/test. Claim S7 tiếp tục
  **ĐANG LÀM**; bước terminal chỉ được thực hiện ở một nhịp hợp lý sau.

### S7 terminal output download attempt 1 — local environment error (2026-08-02)

- Sau đúng một terminal status check xác nhận kernel version `1` là `COMPLETE`,
  lần gọi `project/download_kaggle_output_inventory.py` đầu tiên dừng ngay tại
  import với `ModuleNotFoundError: No module named 'kaggle'` vì dùng nhầm env
  `btxrd-pseudomask`. Boundary xảy ra trước API inventory/request/download; thư
  mục ignored mới chỉ được tạo rỗng, chưa có output khoa học/GT/evaluation.
- Root cause đã xác minh: env nghiên cứu không có module `kaggle`, còn exact system
  interpreter `C:/Users/USER/miniconda3/python.exe` (Python `3.13.5`) chứa Kaggle
  client và `requests 2.32.4`, cùng interpreter với CLI đã status/push thành công.
  Correction duy nhất là chạy lại nguyên utility/kernel/destination bằng system
  interpreter; không thay đổi scientific source/protocol/output selection.

### S7 terminal download + independent pre-GT audit pass (2026-08-02)

- Một bounded status check sau thông báo của người dùng xác nhận version `1`
  `COMPLETE`. Official downloader bằng system interpreter tải full inventory vào
  temp ignored mới. Parent shell chạm timeout `124s` và trả exit `124` trong khi
  child tiếp tục atomic download; kiểm tra process/boundary cho thấy child kết
  thúc bình thường, không còn `.part`. Idempotent rerun xác nhận chính xác
  `1540/1540` official files đã hiện hữu, `downloaded=0`, log `13,861` bytes;
  không status query trong downloader.
- Output root chứa `1,540` file / `190,953,689` bytes trước local-audit file;
  deterministic tree-manifest SHA-256
  `79a0f80c1e454ed79b3f792f10541d3352889a3a289add649971e7a091759038`,
  direct kernel-log SHA-256
  `0d396b0cad4416b0ccbf3d51b065d9e6afda059147a8b1a49127d5559e2edc58`.
- Independent auditor được chạy lại cục bộ từ exact frozen split/cache/baseline,
  không tin riêng audit trong kernel và chưa mở validation GT. Auditor tái lập
  byte-exact `40/40` target snapshots (`target/weight delta=0`), `742` score
  payload, `742` physical maps (`map delta=0`), accepted bag probability và pair
  freeze SHA `a3d37e43...31a63`; `182/371` selection thay đổi. Canonical-LF SHA
  của local audit bằng đúng Kaggle independent audit:
  `554e6564f5527c1536b513cbf165e0f3af876021ac55cc902d08685c005fa4c3`.
- Terminal pre-GT audit tracked tại
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_terminal_pre_gt_audit.json`,
  SHA-256
  `71923c21b8751f42f1880a9edf99b21b6572f17092628ac9f3dbc9179003caad`,
  status `PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_PASS_LOCAL`. Training chỉ
  dùng binary image-level labels; diagnostics không dùng chọn model; đến boundary
  này validation GT chưa đọc, consumer/test chưa chạy. Sau khi commit/push audit
  này mới được phép gọi evaluator matched-pair đã predeclare.

### S7 post-freeze matched-decision source — static trước GT (2026-08-02)

- Static inspection sau pre-GT audit, vẫn trước validation GT, phát hiện
  execution mismatch: S7 protocol đã khóa `10,000` bootstrap với seed
  `20261202`, trong khi generic comparator cũ
  `project/compare_mask_bag_evaluated_arms.py` hard-code `20261101`. Chạy generic
  với seed S7 sẽ fail argument gate; đổi seed S7 sang seed cũ sẽ trái protocol.
  Đây là evaluator plumbing gap, không phải model/result failure và chưa có metric
  nào được đọc.
- Không sửa generic comparator/protocol/prediction. Source mới
  `project/decide_mask_bag_global_local_instance_s7.py` thực hiện matched
  complete-group bootstrap đúng seed `20261202`, verify exact pre-GT audit + hai
  evaluation audit/output inventories, bắt identity arm khớp accepted
  Geometry-v3 per image, rồi áp đúng S7 mechanism/operational gate. Nó fail-closed
  consumer authorization và luôn cấm post-hoc rescue/sweep; matched comparison
  chỉ đọc hai frozen per-image table nên không reopen GT.
- Canonical-LF SHA-256 của decision source/test là
  `ae8d0c0611c1cfefaf5fa74156abee8448b2c908247364a30e279a3b35f3177c` /
  `864d019d9774e2a51501634b63bb585b69c00d47ea486396c261bd9b794fdef2`.
  S7 producer→decision focused suite pass `29/29`; evaluator+generic comparator+
  S7 decision pass `13/13` dưới exact documented Python-3.9 strict-zip shim SHA
  `dcf88d82...c396a`. Lần không shim đạt `12` và chỉ fail known generic
  `zip(strict=True)` compatibility boundary; S7 decision mới tự chạy Python 3.9.
- Chưa chạy evaluator thật, chưa đọc validation GT/metric, chưa consumer/test.
  Bước kế là commit/push exact source, sau đó tạo evaluation-readiness artifact
  bind source commit + pre-GT audit + exact evaluator/decision hashes trước GT.

### S7 post-freeze evaluation readiness — frozen trước GT (2026-08-02)

- Exact evaluation/decision source commit
  `61a27241f2e186ebf91a2b27c7819f5ad650885e` đã push và khớp origin. Readiness
  artifact
  `artifacts/research_protocols/rad_dino_mask_bag_global_local_instance_s7_v1_postfreeze_evaluation_readiness.json`
  có SHA-256
  `c0d29b91aa9deb158748992c81cd48d85196f9948770ce6ee64351503770e1d8`,
  status `FROZEN_AFTER_PAIR_AUDIT_BEFORE_VALIDATION_GT`.
- Artifact bind exact pre-GT audit `71923c21...3caad`, pair freeze
  `a3d37e43...31a63`, hai arm freeze/score manifest, accepted baseline freeze +
  evaluator-only per-image, evaluator SHA `ccc3a493...b084`, S7 decision SHA
  `ae8d0c06...3177` và seed `20261202`. Thứ tự bắt buộc là identity evaluation →
  freeze hash → primary evaluation → freeze hash → matched decision không reopen
  GT; output directories phải mới và không post-hoc rescue/sweep.
- Tại readiness này validation GT vẫn chưa đọc, consumer/test khóa và
  collaborator output không truy cập. Chỉ sau khi artifact/log được commit/push
  và fetch thấy trên central mới được mở evaluator cho exact frozen pair.

### Đồng bộ collaborator `d18e811` trước S7 evaluation (2026-08-02)

- Đã fetch và đọc toàn bộ collaborator `RESEARCH_LOG.md` mới tại
  `d18e811b824cc5220321a0e87b811c54db41b864`; delta từ `29a3ed6` chỉ thêm
  terminal result cho
  `EXP-20260802-codex-rich-gallery-cross-view-cowitness-pair-v1`. Không có claim
  S7/collision và không truy cập Kaggle/output của collaborator.
- Cross-view co-witness residual **không promote**: best full x0.25 Dice
  `0.28791097`, thấp hơn rich-gallery baseline `0.28872949` và gần như bằng
  matched control `0.28791170`; full/control residual Pearson `0.998657`, mean
  absolute difference `0.003738`. Head chủ yếu học source shortcut; external
  saliency selection tăng theo multiplier làm small extent phình mạnh, trong khi
  positive-negative separation chỉ `0.041663`. Prediction-first control-contrast
  salvage cũng không promote (`0.287737/0.287508`).
- Vì result terminal là âm, S7 không adopt cross-view residual/gate. Evidence này
  củng cố contingency: nếu S7 fail, soft extent gate không được dùng cross-view
  residual ở form hiện tại và phải có representation/gating signal mới chứng minh
  vượt matched control. Nó không thay đổi frozen S7 prediction/evaluator/gate;
  validation GT cho S7 vẫn chưa mở ở boundary đồng bộ này.

### S7 identity evaluation frozen trước primary (2026-08-02)

- Sau pre-GT readiness đã visible central, exact evaluator SHA
  `ccc3a493...b084` mới mở validation GT cho `geometry_v3_identity`, exact cohort
  `371/184/187`, subgroup `94/72/18`, `10,000` complete-group bootstrap seed
  `20261202`. Identity tái lập accepted Geometry-v3 Dice
  `0.24548239 / 0.11708058 / 0.37713552 / 0.38941265` và complete misses
  `53 / 33 / 18 / 2` (overall/small/medium/large).
- Per-image/evaluation-audit SHA-256 là
  `0ff1c0c44b2ab1613daf341c02b84316b3f17485375623d4f8dbcffcfcd065da` /
  `320f4e9b6b47f94694356ff80cd535a4f9f87d6015b93afddb5f486c026caf21`.
  Identity freeze artifact
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_identity_evaluation_freeze.json`
  có SHA-256
  `2be0adec78e2a212195ba17ab0fff173aaa048a843951cc4850f56a4f0817f0a`.
- Primary evaluator chưa chạy tại freeze này; matched decision chưa chạy,
  consumer/test khóa. Phải commit/push identity evidence trước khi đánh giá
  `global_local_instance` bằng cùng evaluator/seed.

### S7 primary evaluation frozen trước matched decision (2026-08-02)

- Sau identity freeze commit visible central, cùng exact evaluator/cohort/seed
  đánh giá `global_local_instance`. Dice overall/small/medium/large là
  `0.22931369 / 0.11632462 / 0.33491876 / 0.39694740`; delta so với accepted
  identity là `-0.01616870 / -0.00075596 / -0.04221675 / +0.00753476`.
  Complete misses overall giữ `53`, nhưng small tăng `33 -> 34`, medium giảm
  `18 -> 17`; tổng cộng recover `5` miss và mất `5` overlap.
- Ranking xấu đi rõ: overall selected-to-oracle regret
  `0.16359315 -> 0.17976185`, mean score-quality Spearman
  `0.44161921 -> 0.34718502`. Per-image/evaluation-audit SHA-256 là
  `35217d6a3247b9e1543cf835535492387694e327c63e9c9a4b9bcf1f4f83f14c` /
  `9fccf9bebeecf8a08c55ee4352ad69573a25a6f9d1233fab5eba2596b1281035`.
- Primary freeze artifact
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_primary_evaluation_freeze.json`
  có SHA-256
  `bda4e22e54c2cec9ee637f2f8d467f3d11f7f9747196ce828d831226129bf450`.
  Generic evaluator báo fail nhưng S7 matched decision chưa chạy tại boundary
  này; không kết luận terminal trước khi commit/push exact primary hashes. Không
  consumer/test và không rescue/sweep.

### S7 terminal matched decision — HOÀN THÀNH, GATE FAIL (2026-08-02)

- `EXP-20260802-codex-s7-global-local-instance-v1` chuyển từ **ĐANG LÀM** sang
  **HOÀN THÀNH — GATE FAIL** sau đúng matched decision đã predeclare. Kernel
  `itsthang333/btxrd-rad-dino-mask-bag-instance-s7-v1` version `1` là
  `COMPLETE`; output tree `1,540` file có SHA-256
  `79a0f80c1e454ed79b3f792f10541d3352889a3a289add649971e7a091759038`.
  Prediction pair đã freeze vật lý trước validation GT bằng SHA-256
  `a3d37e43beb8e90d0d30fcf3a86c25bf071fcb340ec62a025ee429799f031a63`
  và independent pre-GT audit `71923c21...3caad` đã pass.
- Matched complete-group bootstrap `10,000`, seed `20261202`, cho delta
  primary-minus-identity overall/small/medium/large lần lượt
  `-0.01616870 / -0.00075596 / -0.04221675 / +0.00753476`; CI95 lần lượt
  `[-0.03641940,+0.00314316] / [-0.02784805,+0.02394772] /
  [-0.07448150,-0.00805024] / [-0.04141317,+0.05611322]`. Dice primary là
  `0.22931369 / 0.11632462 / 0.33491876 / 0.39694740`, đều chưa đạt goal; medium
  suy giảm có ý nghĩa thống kê. Complete misses là `53/34/17/2` so với identity
  `53/33/18/2`: recover `5` nhưng đồng thời mất `5` overlap, nên tổng miss không
  đổi và small xấu hơn một miss.
- Mechanism gate fail vì overall/small không tăng và medium giảm; operational
  gate fail vì không goal nào đạt, overall CI lower không dương và subgroup
  non-regression fail. Matched `paired_per_image.csv`, `paired_comparison.json`,
  `gate_decision.json`, `decision_audit.json` có SHA-256 lần lượt
  `52ab093c...e1bd / 5f16ad5e...bd9c5 / 1065e791...39836 /
  330f47fe...65523`. Terminal result artifact
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_terminal_result_audit.json`
  khóa toàn bộ provenance/result; SHA-256
  `b6bd5bb9b526dfc924a7b173ac8ac7c7417ef2447ddf5717742cf290a79ebfef`.
- Đây là **thất bại khoa học của hypothesis**, không phải implementation hay
  transport failure: source/input/pair/map/audit/evaluator đều pass và identity
  tái lập accepted Geometry-v3 byte/metric. Không consumer, không post-hoc
  mass/epoch/fusion/threshold/area rescue, không BTXRD test. Theo failure-analysis
  gate, chưa được đăng ký/implement/launch successor trước khi phân tích định
  lượng cơ chế S7 và push kết luận phần loại bỏ/kế thừa.

### S7 quantitative failure-analysis gate — hoàn tất (2026-08-02)

- Read-only analyzer
  `project/analyze_mask_bag_global_local_instance_s7_failure.py` (SHA-256
  `50e4e813c29250d69c46c448bd26abd6683a93a25ced22dfee7ad8f92cdd4bbe`)
  join exact frozen pair/evaluation/cache/history; nó không sửa prediction, không
  sweep/rescue/chọn arm và chỉ đọc validation-GT-derived table đã freeze. Dossier
  `artifacts/kaggle/rad_dino_mask_bag_global_local_instance_s7_v1/kernel_version1_retry1_failure_analysis.json`
  có SHA-256
  `24ea83383974517288e78d66b10030a2174c28b6f99b99b16aba67e6b69ef9b5`;
  bind terminal audit `b6bd5bb9...ebfef`, two score manifests, cache manifest
  `8a236bdd...cc1e`, history `ff0e0eec...99b8` và frozen per-image/paired tables.
- **Không phải lỗi implementation:** exact identity tái lập Geometry-v3, `742`
  score/map mỗi arm và pair freeze đã pass; evaluator/decision hashes và cohort
  đều khớp. Đây là hypothesis failure sau khi fit hoàn tất `40` epoch trên T4x2.
  Model đổi winner `182/371` (`49.06%`): `99/184` tumor (`53.80%`) và `83/187`
  normal (`44.39%`). Các ca bị đổi có accepted base top-margin trung bình chỉ
  `0.22288` so với `0.88633` ở ca giữ nguyên, nên head đúng là tác động vào bag
  mơ hồ, nhưng tác động không có calibration: residual advantage của winner mới
  trung bình `1.57268`, vượt base penalty `0.57749` và tạo primary new-vs-old
  margin `0.99519`.
- **Failure mechanism 1 — target tự tham chiếu, không có nhãn identity mới.** Với
  tumor bag, I-projection tạo `q_i=sigmoid(z_i+b)`, là hàm đơn điệu của chính
  current logit `z_i`; local rule lại đặt current argmax thành `1`. Vì vậy target
  bảo toàn thứ hạng đang có rồi self-reinforce winner, thay vì cho biết candidate
  nào trùng tổn thương. Mọi candidate normal có target `0` có thể cho tín hiệu
  normal-vs-tumor, nhưng không định danh vị trí bên trong positive bag. Dữ liệu
  động học phù hợp cơ chế này: `55.04%` positive train bag từng đổi local top,
  `47.58%` khác winner từ epoch 1 đến 40; sau khi mass ổn định ở epoch 21 vẫn chỉ
  `22.24%` bag từng đổi, nghĩa là phần lớn winner đã khóa vào fixed point tự sinh.
  Instance loss giảm `0.11364 -> 0.09288` nhưng validation score-quality
  Spearman giảm trung bình `-0.09443`, chứng minh tối ưu pseudo-target không đồng
  nghĩa tối ưu lesion identity.
- **Failure mechanism 2 — residual quá mạnh nhưng không mang ordering evidence.**
  Raw drift tăng `74.71x`, từ `0.15745` lên `11.76377`; dù coefficient chỉ
  `0.001`, drift term cuối đã là `0.01176` trên total loss `0.10936`. Mean absolute
  residual ở ca đổi là `2.95721` so với `2.42514` ở ca giữ. Tuy nhiên Spearman
  giữa Dice delta và mean/max residual chỉ `0.0031/-0.0279`; trong ca tumor bị
  đổi, residual advantage càng lớn còn có quan hệ âm với outcome (`rho=-0.1735`).
  Đây không phải thiếu residual capacity mà là thiếu tín hiệu xếp hạng ngoại
  sinh và thiếu restraint; tăng capacity/epoch hay giảm mass hậu nghiệm không
  giải quyết đúng nguyên nhân.
- **Failure localization theo subgroup:** `99` tumor winner đổi gồm small/medium/
  large `46/40/13`. Trong medium, `20` xấu hơn, `12` tốt hơn, `8` hòa; delta sum
  `-3.03961`, lớn hơn magnitude toàn overall `-2.97504`, nên medium chịu toàn bộ
  tổn thất ròng. Small `19/15/12` xấu/tốt/hòa, large `4/8/1`; tín hiệu large
  dương không được promote vì `n=18` và CI qua `0`. Regret tăng trung bình
  `+0.01617`, riêng ca đổi `+0.03005`. Toàn bộ `182` transition vẫn cùng source
  `layercam`, nhưng chỉ `9.09%` tumor transition ở cùng family, cho thấy head chủ
  yếu nhảy giữa prompt/family trong cùng proposal source mà không biết family
  nào spatially đúng.
- **Không thể quy lỗi chỉ cho extent:** medium selected-to-GT area median giảm
  `1.30301 -> 1.22079` nhưng Dice giảm `-0.04222`; small vẫn over-segment cực mạnh
  (`36.3459x -> 30.2509x`) trong khi mean selected area gần như không đổi; large
  median coverage còn `0.42892 -> 0.40526`. Trong changed tumor, area delta và
  Dice delta gần như độc lập (`rho=-0.0119`). Vì vậy nhận định signed extent theo
  nhóm vẫn đúng về cấu trúc, nhưng một extent transform/gate đơn độc không sửa
  được medium identity và không được dùng rescue S7.
- **Loại bỏ:** current-logit I-projection + forced-current-argmax target, exact
  mass/epoch schedule S7, và unbounded scalar residual với drift `0.001` đều bị
  retire; không mass/epoch/regularization/fusion/area sweep hậu nghiệm. Không
  quảng bá true-negative all-instance head hay large delta như cải tiến đã chứng
  minh. **Giữ lại:** accepted Geometry-v3 gallery/base, exact cache/provenance,
  pair-freeze/auditor/evaluator/decision infrastructure và failure evidence về
  ambiguous low-margin bags. True-negative evidence chỉ có thể xuất hiện trong
  successor như auxiliary có matched control, không được giả định là hữu ích.
- **Điều kiện cho successor:** phải đưa vào candidate-identity evidence độc lập
  với current selector logit (không pseudo-label chính argmax), có restraint/
  abstention để giữ accepted winner khi evidence không đủ, và tách identity khỏi
  signed extent. Nếu dùng expert theo nhóm, router phải hoàn toàn GT-free và
  chứng minh tốt hơn shared identity control; selected area đơn độc và cross-view
  co-witness form đã fail đều bị cấm. Failure-analysis gate S7 nay hoàn tất;
  consumer và BTXRD test vẫn khóa. Chỉ sau commit/push note+dossier này mới được
  nghiên cứu, đăng ký và launch successor không trùng.

### Deep-search sau S7 và lựa chọn decoder-reconstruction successor (2026-08-02)

- Sau failure gate `EXP-20260802-codex-s7-global-local-instance-v1`, đã fetch/
  đọc lại central `8c02ca0...104` và toàn bộ collaborator log tại
  `d18e811b...864`; không có claim SKELEX-reconstruction đang `ĐANG LÀM`.
  Collaborator cross-view co-witness đã terminal/retired và scale-conditional
  chỉ là feasibility; không truy cập Kaggle/output của collaborator. S8 không
  chạy lại G1/G2/BAS/cross-view/area gate của họ.
- S7 chứng minh mọi target chỉ là hàm đơn điệu của current logit không thể sinh
  candidate identity mới. Deep-search vì thế ưu tiên nguồn spatial evidence
  ngoài selector. SKELEX chính thức là ViT-MAE Large pretrain self-supervised
  trên `1,296,540` MSK radiograph và báo cáo bone-tumor AUROC `0.953`; quan trọng
  hơn, decoder tạo unsupervised anomaly map bằng `10` random masks, average
  masked-pixel squared reconstruction error và không fine-tune task. Paper còn
  mô tả khi tumor bị mask hoàn toàn, model tái tạo bone trông bình thường. Nguồn:
  https://www.nature.com/articles/s41746-026-02826-9 và
  https://arxiv.org/abs/2602.03076 . Exact public revision
  https://huggingface.co/skhoha/SKELEX/tree/368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e
  có `model.safetensors` SHA-256
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`;
  license `CC-BY-NC-ND-4.0`, không redistribute/modify weights.
- Đây không phải lặp MAE-base normality probe hay S5. Probe cũ dùng ImageNet
  `facebook/vit-mae-base`, full-image pixel map và đã fail small; S5 dùng frozen
  SKELEX **encoder descriptors** rồi learned MIL residual/equal-rank, không gọi
  decoder. S8 dùng exact domain-specific **decoder reconstruction error** và
  deterministic spatial-null abstention, không fit selector/pseudo-target. Nó
  trực tiếp giải quyết S7 root cause: signal không phụ thuộc current argmax và
  có baseline-preserving restraint. COIN (https://arxiv.org/abs/2404.12832)
  hỗ trợ nguyên lý counterfactual reconstruction cho medical WSSS, nhưng S8
  không train GAN/inpainter và chỉ chuyển nguyên lý anomaly-by-reconstruction.
- Candidate score dùng signed `inside mean error - radius-2 local-ring mean
  error`; do đó tight small candidate bị phạt nếu chứa nhiều normal anatomy,
  còn large candidate chỉ được mở rộng khi phần mở rộng vẫn có reconstruction
  anomaly. Đây là một shared scale-free rule, không GT-size router, không
  selected-area heuristic. Spatial randomization giữ nguyên error histogram,
  candidate/area/base score nhưng phá alignment, cho phép kiểm tra signal có
  thật sự nằm đúng vị trí thay vì chỉ là area/anatomy proxy như BAS.

### EXP-20260802-codex-s8-skelex-reconstruction-randomization-v1

- **Owner/status/time:** Codex central workstream; **ĐANG LÀM**; đăng ký
  `2026-08-02T16:57:59+07:00` trên base
  `8c02ca00dd6e94dbee3d9ddbac6b3dddd5704104`; registration commit
  `a7ccfdcb53f87dfdff861a46359089c75cdb8dc8`. Không real cache/image/SKELEX
  inference/prediction/Kaggle trước
  khi claim visible trên `origin/research-wsss-improvement`.
- **Kế thừa và khác biệt:** kế thừa terminal S7 audit/failure dossier
  `b6bd5bb9...ebfef / 24ea8338...ef9b5`, accepted Geometry-v3 same-gallery
  baseline, exact selector cache và SKELEX public weight proven transportable ở
  S5. Không kế thừa S7 current-argmax target, S5 learned residual/equal-rank như
  kỹ thuật tốt hơn, hay bất kỳ BAS/normal-prototype/local-affinity/graph/
  subtype/cross-view mechanism đã reject. Scientific delta duy nhất là frozen
  SKELEX decoder reconstruction evidence + spatial-null selective rerank trên
  **same immutable candidates**; không regenerate proposal.
- **Exact input/provenance:** split SHA-256
  `85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c`;
  cache freeze `2f6290cd464ac8a1d204b6196f7f7a1dbe5bbcc21b8abd56ed5a61f8b41e4f2c`,
  manifest `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`;
  baseline checkpoint
  `58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069`,
  baseline freeze
  `ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3`;
  SKELEX config/preprocessor/weight SHA-256
  `b48411f4...c6f7 / a250969c...6cea / 81cd6e9c...69ef8`. Immutable oracle
  vẫn `0.40907553/0.22274949/0.59414708/0.64182537`; evaluator-only baseline
  per-image `a26143d0...605f` không phải producer input.
- **Frozen intended algorithm trước implementation:** SKELEX 224px, public
  `mask_ratio=0.75`, normalized-pixel-loss disabled, đúng `10` deterministic
  seed-42 masks cho mỗi original và horizontal flip. Chỉ squared error trên
  masked patches được tích lũy; mỗi patch phải có positive mask count. Candidate
  và radius-2 ring được area-project chính xác lên `14x14`, nhân square-content
  occupancy; ring zero-mass fail-closed. Mỗi candidate lấy mean signed
  inside-minus-ring contrast qua `20` maps và conservative LCB
  `mean - 1.96*SE`. Tie-aware within-bag rank fusion cố định
  `0.75*Geometry-v3-rank + 0.25*reconstruction-LCB-rank`.
- **Baseline-preserving randomization gate:** combined fused winner chỉ thay
  accepted winner khi family của nó cũng là winning family ở cả original-only
  và aligned-flip-only reconstruction branches, và max fused improvement đạt
  exact permutation p-value `<=0.05` so với `255` seed-`20261203` permutations
  độc lập của error values trong các content-valid patches. Null giữ error
  histogram, candidate masks/areas/families và base rank nhưng phá spatial
  alignment; statistic lấy max trên toàn bag để hiệu chỉnh multiplicity. Nếu
  bất kỳ gate nào fail, output dùng accepted winner byte-identical. Không
  threshold/weight/mask-count/seed/extent sweep hoặc validation-derived routing.
- **Compute/output:** static/synthetic locally; mọi BTXRD image inference chỉ
  private Kaggle T4x2/P100. Hai GPU chia `186/185` validation rows và phải chạy
  real CUDA guard. Freeze trước GT: exact 20-mask bank/provenance, original/
  flip patch-error payload, every candidate contrast/LCB/null p-value/family,
  full baseline/S8 score vectors, `742` maps và pair freeze. Independent
  producer-free auditor phải tái lập SKELEX input/noise/masked-error arithmetic,
  candidate projection, 255-null selector, baseline fallback, bag probability
  và physical maps trước evaluator.
- **Gates:** mechanics require nonconstant reconstruction, exact original/flip/
  null reproducibility, no baseline change when gate fails, overall+small Dice
  strictly improve, medium/large do not regress and complete misses do not
  increase. Operational pass remains simultaneous Dice
  `0.34024039/0.17895493/0.51244178/0.49370336`, overall paired CI95 lower
  `>0`, no subgroup regression/miss increase. Chỉ operational pass cho phép
  consumer. Prediction phải freeze vật lý trước validation segmentation GT;
  training supervision (nếu có; S8 frozen model không fit) không vượt image
  labels; BTXRD test luôn khóa. Fail/error bắt buộc freeze + quantitative
  failure analysis trước successor, không post-hoc rescue.

### S8 static implementation/protocol freeze và collaborator sync (2026-08-02)

- Sau khi fetch, central ở `10cff2039aa170debc6a491950af39dee5635432` và
  collaborator tiến tới `b5c3bf2362f6ec571098a72daba1237fda2f15f6`.
  Delta collaborator mới không tạo collision: latent-burden gate với old frozen
  observables đã **COMPLETE/RETIRED** và nested gate abstain toàn bộ, partial-
  consensus point estimate `0.29343648` có CI qua `0` và làm hại small/large nên
  không được adopt như cải tiến; MNR residual mới chỉ là `DESIGN`, chưa có
  terminal audited result. Không truy cập Kaggle/output collaborator.
- Scientific implementation đã freeze tại commit
  `b4543aeb9430345c9b789384943bd218816a85dd`: dataset-agnostic decoder-error/
  LCB/spatial-null primitive, validation producer, independent GT-blind
  arithmetic/physical auditor, fail-closed Kaggle wrapper và synthetic tests.
  Canonical LF SHA-256 core/runner/auditor/test lần lượt là
  `df93b09b7c1c311c1977889705efe1709ab178cb60d95366d06405598280a7e5 /`
  `a76db937db3654ca769a419f2c1713e99ae0fffe3521665bada14e62cdee43ce /`
  `613b5b4244f001765dd705b53eb0edde755bb224780022ea2d0ba4eb9055b54d /`
  `010b47c60447bd4064ec805c56f43b0c9bb9abe44ceaa88d7f1860e640c8e3ea`.
- Exact protocol
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1.json`
  được freeze ở commit `67a8f765cf75f20fd0c9df157bbb735d249fddf1`, SHA-256
  `7f81978151600dcae6827f5060e04064fb8f22ce42ae1f10dd92a5eceda6bc07`.
  SKELEX static config thực tế xác nhận input/patch `224/16`, ImageNet mean/std,
  `mask_ratio=0.75`, `norm_pix_loss=false`, transformers `4.50.2`, đúng protocol.
  Điều chỉnh implementation-only trước launch: hai T4 dùng DataParallel để chia
  `20` masked-view forwards của mỗi ảnh qua hai device, thay cho mô tả row split
  `186/185`; algorithm, cohort, score, null, gate và output không đổi.
- Focused synthetic/static QA pass `20/20`, Ruff, `py_compile`, JSON parse và
  `git diff --check` đều pass. Prelaunch static audit
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_prelaunch_static_audit.json`
  có SHA-256
  `ae5d1771d230ab4a5319f808aa605e797adec705ad1128f64798adc0b93830ac`.
  Wrapper vẫn cố ý `KERNEL_VERSION=0`, `LAUNCH_BINDING_READY=false`, checkout
  `UNBOUND`; chưa mở real cache/BTXRD image, chưa SKELEX inference/prediction,
  chưa validation GT/consumer/test. Bước kế là commit/push log+audit, sau đó
  mới bind exact checkout/kernel version và chạy prelaunch audit lần cuối.

### S8 auditor correction, wrapper provenance và one-time launch binder (2026-08-02)

- Đã hoàn tất failure/completeness boundary trước launch. Commit
  `bc7816ff6cee5a7c5e954668d1255d1b1ad04533` sửa implementation-only phần
  independent audit: packed source masks/projection metadata, exact 255-null
  distribution/statistics và physical-map reproduction; producer-vs-independent
  null synthetic regression pass. Addendum đóng băng tại commit
  `3e1fbc93425dca43490490b89ba129bfc277d88c`, SHA-256
  `dabee40fc3b607df3f82105ab9122b2b80b37c2305d7c5ed17c1c8ae1c3dca0e`, xác nhận
  `scientific_change=false`; protocol khoa học, SKELEX config, cohort, candidate
  gallery, score/gate và safety boundary không đổi.
- Canonical source hashes sau correction: core
  `e37d83f89191c1c3a2af90be5325c7523a4a2c958adbb643ace9fa24e26cffcb`, runner
  `722c7a9692b80009ddfbfe43400b5f9d083b4ba57b084c7408db9206bfd2c268`, auditor
  `144ecf5c07eceb8b29e0a6107b49d2f31ddbbd8cea8548464f3f4d1bc58efde3`, core test
  `6d9029428ce6315c779badf506ffaa131a6f9ea4d280124741053402463302b1`.
- Trong lần kiểm tra provenance trước launch phát hiện một typo implementation-only
  trong wrapper: `CACHE_MANIFEST_SHA256` thiếu ký tự `2` so với manifest freeze
  `8a236bdd735c18c62014e206e122ba5cee21c84fd0902892dfe9a8168307cc1e`. Đã sửa,
  đồng thời wrapper chạy static test core **và chính wrapper** trước khi mở input;
  canonical wrapper LF SHA sau sửa là
  `718bc39c2aef0f6d96892238d591474acf1d1e481dab3d70ddbd4e61ede0fab8` tại commit
  `5df72bf097e9a6ea912072943141d43e7be952ef` (correction-source commit vẫn là
  `bc7816f...`, không phải thay đổi khoa học).
- One-time binder/test được thêm tại commit
  `d1a791708b53e6d541e0b25fe3a33f7659001889`, binder LF SHA
  `4014cf3dce07aa04eb4b9f9d6facd5826dd0426fad62365ee042137dfaa4378c`, test LF
  SHA `693c340ed15e4fc78868f49834c3ee8e50ac570061318bac5f9e3cd926f2a20f`.
  Binder chỉ thay đúng `KERNEL_VERSION`, `LAUNCH_BINDING_READY` và exact
  `CHECKOUT_COMMIT`, rồi kiểm tra inverse byte-identical, source/protocol/addendum
  ancestor/hash. Không sửa canonical wrapper thành bound.
- QA cục bộ sau correction/binder: focused S8 tests `10 passed`, Ruff PASS,
  `py_compile` PASS, `git diff --check` PASS. Lần thử bằng `python` mặc định
  không có numpy/ruff nên bị dependency error; không phải scientific run; QA hợp lệ
  dùng `C:\Users\USER\miniconda3\envs\btxrd-pseudomask\python.exe`.
- **Safety/launch boundary:** đến mục này chưa mở selector cache/BTXRD image,
  chưa chạy SKELEX inference, chưa tạo validation prediction, chưa đọc validation
  GT, chưa train consumer/test, không truy cập output collaborator. Claim S8 vẫn
  `ĐANG LÀM`; chỉ được bind/package/launch sau khi mục này hiện diện trên
  `origin/research-wsss-improvement` và collision check lại pass.

### S8 packaged prelaunch audit và collaborator deep-bottleneck sync (2026-08-02)

- Đã đọc đầy đủ `RESEARCH_LOG.md` trên `origin/codex/research-sync-20260731`
  tại commit `454b746b1832049e26260460f3130be4e665d2ab` (không truy cập Kaggle
  hay output của collaborator). Insight có thể chuyển giao: confirmed G1
  `0.2887294867` vẫn là comparator; gallery supply không phải ceiling
  (full/eligible oracle `0.528298/0.527902`, proposal-truncation regret
  `0.000396`), selector regret `0.239173`; extent routing deployable chưa có
  (true-group `0.311904` chỉ upper bound, nested gate tái tạo baseline), và MNR
  residual đã pause vì positive-instance ambiguity/top-17 extent-insensitivity.
  Collaborator đề xuất SMILE với local subtype-conditioned heads, dense normal
  negatives, soft foreground/background, spatial consistency và tách identity/
  normalized-extent residual cho hướng kế nhiệm nếu S8 không pass. Đây là thiết
  kế tương lai, chưa phải kỹ thuật đã chứng minh và không làm thay đổi S8 claim;
  không có collision SKELEX-reconstruction.
- Bound package kiểm tra tĩnh không mở input tại
  `project/audit_skelex_reconstruction_selector_s8_kernel_v1_prelaunch.py`
  (LF SHA `e79c8f045ce7e635cb75ee9592f9353e60f5842d6152dbac3523a068f8d4ac22`).
  Audit artifact
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_kernel_v1_prelaunch_audit.json`
  có SHA-256 `1cd3d4a5b0a6be9d0b223ab82f1a716f4d33a4c2f41c22d9229658757193047d`,
  xác nhận bound wrapper `9480330152bd7c77ffc911f657ceae1518ea972657343b00bfa7fac05fb2731f`,
  template `718bc39c2aef0f6d96892238d591474acf1d1e481dab3d70ddbd4e61ede0fab8`,
  protocol/addendum/source hashes đúng và `validation_gt_read=false`,
  `consumer_trained=false`, `test_evaluated=false`, `collaborator_output_accessed=false`.
  Đây vẫn là static/no-input audit; chưa launch Kaggle.
- Final launch package được re-bind một lần vào exact pushed checkout
  `819e6aff7031a987a14156fbad0ff6b313a5eb2e`; bound-wrapper SHA-256
  `dc0cc8280822b47eb0b023e1b0dc6a2edbb55eb9542e877265abfeb64a50f74c`, launch-
  binding SHA-256 `fb76a75b30caafe45991f61e299c38c152b42173dc7aae12d912922304d5fd02`.
  Prelaunch audit artifact được supersede bằng final SHA-256
  `76db82166cffdb83e17e393bcea72bd6805782c6e75fd9f6b73c1b4ba4361ca6`;
  prior `1cd3d4a5...3047d` chỉ audit package checkout `1a8147b...`, không phải
  package sẽ launch. Final package vẫn chỉ khác canonical wrapper đúng ba launch
  fields, private T4x2, input/GT/consumer/test safety đều giữ nguyên.

### S8 Kaggle launch version 1 (2026-08-02T18:18:52+07:00)

- Sau final fetch/collision check central `b50781dc581cdad51de36159dc8d515892b58bdd`
  và collaborator `454b746b1832049e26260460f3130be4e665d2ab`, không có claim trùng,
  đã push private kernel
  `itsthang333/btxrd-skelex-reconstruction-selector-s8-v1`, **version 1**, URL
  `https://www.kaggle.com/code/itsthang333/btxrd-skelex-reconstruction-selector-s8-v1`,
  accelerator `NvidiaTeslaT4` với wrapper hard guard đúng hai T4.
- Exact execution checkout `819e6aff7031a987a14156fbad0ff6b313a5eb2e`; uploaded wrapper SHA-256
  `dc0cc8280822b47eb0b023e1b0dc6a2edbb55eb9542e877265abfeb64a50f74c`;
  launch-binding raw/canonical-LF SHA-256
  `8c37f3b42f21b209e2a197ca5c663c66e09e3e4f6d56bad34869d2c977be7d09 /`
  `fb76a75b30caafe45991f61e299c38c152b42173dc7aae12d912922304d5fd02`;
  metadata SHA-256
  `471583126dacd4199b1921126809ccc92e8b489962c0900b7429c7d8380661e1`.
  Kaggle bỏ qua ba keyword không hợp lệ nhưng kernel push thành công; input,
  code, machine shape và scientific protocol không đổi.
- Tại launch chưa có output/prediction, chưa đọc validation GT, chưa train
  consumer, chưa mở BTXRD test và không truy cập collaborator output. Chỉ thực
  hiện một bounded status check ở nhịp hợp lý; không polling/monitor lặp.
- Bounded status check duy nhất ngay sau launch trả
  `KernelWorkerStatus.RUNNING`. Không tải output/log, không kiểm tra lần hai và
  không tạo monitor; claim S8 tiếp tục `ĐANG LÀM`, mọi GT/consumer/test lock giữ nguyên.

### S8 postfreeze evaluator/decision freeze trước validation GT (2026-08-02)

- Không status-check Kaggle trong nhịp này để tránh polling. Static terminal-
  readiness audit phát hiện evaluator generic có gate không khớp S8 và bootstrap
  seed evaluation chưa được freeze riêng. Đây là lỗ hổng evaluation-protocol,
  không phải kết quả khoa học và chưa mở bất kỳ input/GT nào.
- Đã thêm matched decision source
  `project/decide_skelex_reconstruction_selector_s8.py` và synthetic test tại
  commit `035482f663cfca8969ecc545f10712b6829d191d`. Canonical LF SHA-256 source/test
  lần lượt `538ecad2b79d3b059587db3e687bd789fc7e00a31c8e106015b8198199153d97 /`
  `4e496603ef82a3780f16a68ea616c0751eaa76087d1b9976b04048a9a2610c13`;
  `4 passed`, Ruff và `py_compile` PASS. Decision kiểm tra exact pre-GT audit,
  dynamic readiness hashes, cả hai evaluator audits/output inventories, control
  byte-for-metric reproduction của accepted Geometry-v3, rồi mới tính paired
  complete-group bootstrap và đúng mechanism/operational gates đã đăng ký.
- Evaluation addendum
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_postfreeze_evaluation_addendum.json`
  được freeze **trước GT**, SHA-256
  `d79d98564dd239d4821f4a54bf93371e9a84375ce0c2cecd09830467ad53aec9`.
  Exact evaluation bootstrap là `10,000` replicates, seed `20261204`, cùng seed
  cho control/primary/matched decision. Các prediction/score hash chỉ tồn tại khi
  producer terminal nên phải được bind vào readiness artifact sau independent
  GT-blind audit và commit/push central trước evaluator. Không cho phép post-hoc
  rescue/sweep; scientific S8 algorithm không đổi.
- Safety đến đây: `validation_gt_read=false`, `consumer_trained=false`,
  `test_evaluated=false`, không truy cập collaborator output; S8 vẫn `ĐANG LÀM`.

### S8 dynamic postfreeze-readiness freezer (2026-08-02)

- Không status-check Kaggle trong nhịp này. Đã bổ sung công cụ GT-blind
  `project/freeze_skelex_reconstruction_selector_s8_postfreeze_readiness.py`
  tại commit `3b4bf305e0472236d3ca086c112b5777d93689c4`, canonical LF SHA-256
  `5a691b18d05e7582a16f23876e4183a6f21957b122a159b8ba271e7214cbf1e6`;
  test SHA `e06cb486fb3b621892f049af410dbddf6e68c9e52b44f0deeae6ee8a1647cd23`.
- Freezer chỉ được chạy sau independent S8 output audit PASS. Nó kiểm tra hash/
  safety của protocol, evaluation addendum, pre-GT audit, pair freeze, hai arm
  freezes, prediction/score manifests, exact source/split/cache/baseline và cohort,
  rồi sinh một readiness artifact one-time với toàn bộ dynamic hashes. Nó không
  import dataset loader, không mở ảnh/mask/GT và fail nếu output đã tồn tại.
- Focused freezer+decision QA `7 passed`, Ruff/`py_compile`/`git diff --check`
  PASS; tamper score manifest và GT-safety drift đều bị reject. Scientific S8
  algorithm/evaluation gate không đổi; validation GT chỉ được mở sau readiness
  artifact được commit/push central. Consumer và test vẫn khóa.

### S8 version 1 terminal ERROR và failure-analysis gate (2026-08-02)

- Kernel `itsthang333/btxrd-skelex-reconstruction-selector-s8-v1`, version 1,
  terminal `ERROR`. Direct log inventory được tải một lần vào ignored temp
  `tmp/kaggle/skelex_s8_v1_terminal_error_20260802_1828`; log SHA-256
  `0ed4c39bcee22a559de3c642bb72a5b77ce148c50e8260bf0d89d5dd47d82a79`,
  `26,573` bytes. Compact inventory tải được `11` files; không tải toàn bộ 1,866
  output objects. Error boundary nằm **sau** 371/371 validation inference,
  tạo cả hai arm prediction manifests/score manifests, 742 physical maps,
  reconstruction evidence, pair freeze và run manifest; hai T4 đều được guard
  (`Tesla T4`, `Tesla T4`, DataParallel). Không mở validation segmentation GT,
  không consumer/test, không collaborator output.
- Cụ thể independent auditor fail ở `IMG000001.jpeg`, line 353:
  `ValueError: S8 fused score arithmetic mismatch`. Đã tải đúng evidence đầu tiên
  `reconstruction_evidence/0000_IMG000001.npz` (58,205 bytes, SHA-256
  `1beb7f8eaae0bbd42c2b8ffcc0b1073feb11baeeef107f84562cc140d4c41c49`) và định
  lượng: producer serialized `combined_lcb` và auditor NumPy reconstruction lệch
  tối đa chỉ `1.1175871e-08`, nhưng exact equality tie ranking khác: producer có
  các tie groups `4+2` ở giá trị khoảng `-0.012091338`, auditor gom `6`. Vì rank
  fusion là tie-aware, fused score lệch tối đa `0.008474588` trên 10/63 candidate
  (vượt audit tolerance `2e-5`). Đây là **implementation/auditor numerical
  reproducibility error**, không phải scientific hypothesis failure; chưa được
  dùng bất kỳ metric Dice nào.
- Các artifact evidence/arms được giữ nguyên để correction audit, nhưng version 1
  được ghi `LỖI` và không được quảng bá là kết quả. Không rescue/sweep seed,
  weight, threshold, mask count, extent hay đọc GT. Failure-analysis gate đã hoàn
  tất; hướng correction duy nhất là làm independent rank reconstruction dùng
  exact serialized float32 LCB (sai số arithmetic vẫn phải allclose độc lập),
  sau đó rerun auditor/prediction trên cùng one-shot protocol trước khi xem GT.

### S8 audit-only correction implementation/prelaunch (2026-08-02)

- Sau failure gate commit `4f9075a58f514b7ed9ec1ca300ae90481f398e4f`, auditor
  correction được commit tại `969327c4fbbd635fff2e3a00d34d533af8a3c340`:
  independent LCB arithmetic/allclose vẫn giữ nguyên, chỉ tie-aware rank dùng
  exact serialized float32 LCB. Corrected auditor/test LF SHA-256
  `c972e1460332119cefd11c1145035a497748d4797482c29c51cf62980c560232 /`
  `fc9786e9b0e8bfe43fa3bb9d8cc7d9e9933270caf2f7a6de1eac797d29dc11a6`;
  regression test pass và first-image fused score reproduction max difference
  trở thành `0.0`. Correction addendum freeze tại commit
  `16f1b61ef99e866dcfced826b4b2ffb76fb0d3b5`, SHA-256
  `94e5881f763cc2cb3bd0a3f49cb563f2449140a7c576211252a45579597fc8a2`,
  xác nhận scientific algorithm/prediction không đổi.
- Full CPU replay đi qua first fused boundary nhưng tại `IMG000160` có đúng
  `1/255` null improvement lệch `0.0056818128` do CPU-vs-T4 exact tie arithmetic;
  exceedance và p-value vẫn exact `193 / 0.7578125`. Không nới tolerance hay bỏ
  null check; correction audit phải chạy lại trên T4x2 giống producer.
- Để không lãng phí rerun 371 SKELEX inference, đã tạo audit-only private-kernel
  wrapper `itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1` tại
  commits `0326cb3f05c2d0f0ba3103ae62f2b2d63b0fbc5b` và
  `f99994ffd089686e220c362c1a4dbb3bacc59d1f`. Final canonical template LF SHA
  `7e66f0d55c5b0978ddf0bab47959da8e1db422d364e7d596413ec9eda14e1714`.
  Wrapper chỉ dùng immutable version-1 kernel output theo exact pair/run/
  diagnostics/evidence/arm-manifest hashes, hard guard T4x2 và chạy corrected
  auditor; không inference/prediction mới. One-time binder commit
  `2d7c05db94532b5a0143e1d60b096c33da87426f`; focused audit/wrapper/binder tests,
  Ruff, `py_compile`, `git diff --check` PASS.
- Chưa bind/launch audit-only kernel ở mục này; validation GT, consumer và test
  vẫn khóa. Chỉ audit PASS mới cho phép dynamic readiness freeze trước evaluator.
- Final audit-only package bind exact checkout
  `34c83e41acb9139ff87c2c75bcd674becb9d440a`, kernel version `1`; bound-wrapper,
  launch-binding raw và metadata SHA-256 lần lượt
  `8d226f9bc1a2145e1c8c5048630b08428bee2c36968ef987271a9f8f7d51059f /`
  `38b87f3cde11d44538d61653d2c4affed9d174c3bcd9646260a9b11f610db547 /`
  `b404b56dd1308eeee8c6057ed2939466c99385e4ca254224f456e666cc89056f`.
  Metadata dùng duy nhất producer-kernel source version 1, private T4; packaged
  `py_compile`, inverse binding, checkout/correction/input/machine audit PASS.

### S8 audit-only kernel version 1 transport warning (2026-08-02)

- Push private kernel
  `itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1`, version 1,
  thành công, nhưng Kaggle trả warning ngay tại push:
  `not valid kernel sources: itsthang333/btxrd-skelex-reconstruction-selector-s8-v1`.
  Nguyên nhân boundary: producer version 1 terminal `ERROR` nên Kaggle không cho
  attach output của nó như notebook/kernel source, dù 1,866 output objects vẫn
  tải được qua API. Bounded status check duy nhất hiện `RUNNING`; không poll thêm.
- Version audit-only 1 thiếu producer input và không thể sinh scientific/audit
  result hợp lệ; giữ dấu vết như transport error, không sửa/rerun khi chưa terminal.
  Correction transport được chuẩn bị bằng private dataset chứa exact immutable
  producer output đã tải (371 evidence + 742 score payloads + 742 maps + 11
  manifests/log), không GT/test và không rerun inference. Validation GT,
  consumer/test tiếp tục khóa.

### S8 audit-only version 1 terminal transport ERROR (2026-08-02)

- Đúng một bounded status check lúc `2026-08-02T18:58+07:00` xác nhận private
  kernel `itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1`, version
  `1`, terminal `ERROR`. Không polling/monitor hoặc status check thứ hai. Direct
  log được lấy bằng inventory downloader, không tải 1,531 official output files,
  vào ignored temp
  `tmp/kaggle/skelex_s8_audit_v1_terminal_error_20260802_185904`; log `3,382`
  byte, SHA-256
  `146fe7b4d57b9a10ae7d8d3b5554c64b71c99b1e767bacb124346e837c37b6c1`.
- Log xác nhận exact checkout `34c83e4...40a`, ancestor/hash checks, two-T4
  fail-closed guard, auditor `py_compile` và focused regression `1 passed` đều đi
  qua. Ở giây `25.215`, trước corrected auditor, wrapper fail-closed tại
  `find_and_verify_producer_output` với
  `RuntimeError: Expected one immutable S8 producer output, found []`. Đây là
  terminal confirmation của warning upload: Kaggle từ chối terminal-ERROR
  producer như kernel source nên `/kaggle/input` không có pair freeze cần thiết.
- Error audit tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/kernel_version1_transport_error_audit.json`.
  Version 1 là **LỖI TRANSPORT**, không phải failure khoa học, không tạo prediction
  mới, không tạo audit/evaluation result và không được dùng để kết luận S8.
  Failure-analysis gate cho boundary này hoàn tất: correction duy nhất được phép
  là private dataset transport chứa exact immutable producer bytes đã tải; không
  rerun inference, không đổi scientific protocol/predictions/selector/auditor/
  evaluation gate. Validation GT chưa đọc, consumer chưa train, BTXRD test khóa
  và collaborator output không được truy cập. Phải commit/push bằng chứng này
  trước upload dataset hoặc audit-only version 2.

### S8 frozen-output transport package v1 — static/provenance ready (2026-08-02)

- Sau khi failure-analysis gate của audit-only v1 đã được commit/push tại
  `2c8bbf2`, đã chạy static transport audit trên exact producer tree, không mở
  ảnh/mask/validation GT/test. Auditor mới
  `project/audit_skelex_reconstruction_selector_s8_transport_dataset.py` có
  canonical-LF SHA-256 `9e6c51daba8be15fedf9c0e9d4f9bfa6899e42bb67f678e73533dc0d61571739`,
  và artifact
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/producer_transport_inventory_audit.json`
  SHA-256 `945cf370b2c057f777cdd1cfd84c86ef5841f4a2fbcf7a480edaea597738780e`.
  Audit PASS: exact `1,866` producer files, `172,187,320` bytes, inventory SHA
  `6caa288321e1abfd81934222eb04b590c23aedfc761b5d7eb82458a59a79b971`, gồm
  `371` reconstruction evidence, hai arm mỗi arm `371` score payload và `371`
  prediction maps; all ten root/manifests match S8 frozen hashes; không có
  GT/test-like path; log producer giữ SHA `0ed4c39b...d82a79`.
- Dataset package ignored tại
  `tmp/kaggle/skelex_s8_v1_frozen_output_dataset`: metadata SHA
  `549c7479cefa2a1b74b1fcb6b1587146ce90163d0e88899a5ef937ca748f6144`, archive
  `s8_producer_output.zip` `20,290,662` bytes SHA
  `c516437824ff7d7e32594bfe02e3f654d98d9976d2ddb40595641bf5f8ca1737`, và
  direct producer log `26,573` bytes SHA `0ed4c39b...d82a79`. Package chỉ là
  transport private, license metadata `other`, không phải public result hay
  scientific change. Metadata conventions được đối chiếu với Kaggle CLI
  tutorial: https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md .
- Package contract/addendum được ghi tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/transport_dataset_v1_package.json`;
  dataset upload và audit-only wrapper version 2 vẫn **chưa launch**. Bước tiếp
  theo duy nhất là commit/push package audit này, upload private dataset, rồi
  bind một wrapper v2 chỉ giải nén/hash exact archive và chạy corrected
  producer-output auditor; không rerun SKELEX inference, không đổi protocol,
  selector, predictions hay evaluation gates. Validation GT, consumer, test và
  collaborator output tiếp tục khóa.

### S8 transport dataset upload attempt 1 — local CLI path ERROR (2026-08-02)

- Lệnh `kaggle datasets create -p tmp/kaggle/skelex_s8_v1_frozen_output_dataset
  -r zip` dừng ở local upload-preparation boundary với
  `[Errno 2] No such file or directory` cho temp manifest path
  `C:\\Users\\USER\\AppData\\Local\\Temp\\.kaggle/uploads\\tmp/...zip.json`.
  Client chưa hoàn tất request tạo dataset/version; không có kernel/input,
  prediction, GT, consumer hay test action. Đây là tooling/path error, không phải
  scientific result và không thay đổi archive/provenance. Không tải lại output.
- Failure boundary được ghi trước retry; correction duy nhất là gọi cùng package
  bằng absolute path, giữ nguyên metadata/archive SHA và private visibility.

### S8 private frozen-output dataset upload accepted (2026-08-02)

- Sau khi attempt-1 tooling error được commit/push tại `48ac1cf`, đúng một retry
  với absolute package path đã upload thành công hai exact files và Kaggle trả
  `Your private Dataset is being created` tại
  https://www.kaggle.com/datasets/itsthang333/btxrd-skelex-s8-v1-frozen-output .
  Archive `20,290,662` bytes SHA `c5164378...ca1737`; terminal log `26,573` bytes
  SHA `0ed4c39b...d82a79`; visibility private. Ba optional tags bị từ chối nhưng
  không đổi files, provenance hay quyền truy cập. Upload receipt tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/transport_dataset_v1_upload_receipt.json`.
- Đây chỉ là transport của pair đã freeze: không inference/prediction mới, không
  validation GT/evaluator, consumer hay BTXRD test. Chưa launch audit-only v2;
  wrapper v2 phải verify exact archive hash, giải nén an toàn, verify producer
  manifests/pair rồi mới chạy corrected auditor trên T4x2.

### S8 audit-only v2 dataset-transport implementation freeze (2026-08-02)

- Transport correction implementation đã commit/push tại
  `88f20fb756495569550a29e888adc1e9c137b964`. Canonical-LF SHA-256 của wrapper
  template/binder/transport-test/binder-test lần lượt là
  `e596c57c4e425d195ec7b732cf9c329cc8b6013f1f0006ac7de7cddcb7827d9c /`
  `1fedc1ef7d810e03a1c93ddb145b0d6c9ddc350c9c39e42f2e22239ce88e3780 /`
  `4218e7da053270f77eaa7d3da8f804c63d7d43f62b92d6575d750cb0e17a0a42 /`
  `cc7594183d7fdd8e26858ed01ec13f317e8141a0ed79d4d62a31f8d7c5c6d889`.
  Focused corrected-auditor + transport + binder suite `4 passed`; Ruff,
  `py_compile` và `git diff --check` PASS.
- Wrapper chỉ thêm exact dataset/archive locator, SHA/count/uncompressed-byte
  guard và extraction chống traversal/symlink; nếu Kaggle server đã expand thì
  vẫn bắt exact pair + all immutable manifests. Scientific protocol
  `7f819781...bc07`, serialized-LCB correction `94e5881f...fc8a2`, corrected
  auditor `c972e146...0232`, pair `b2cfd59f...fa00`, predictions/selector/gates
  không đổi và không rerun SKELEX inference. Frozen correction addendum tại
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_audit_transport_correction.json`,
  SHA-256 `ee42bbe43d4f81ffba570a8aa46454cb55acbf9bb6338ed4d746aaf38ce32d1d`.
- Chưa bind/package/launch version 2 ở bước này. Sau commit/push addendum/log,
  binder phải khóa exact checkout và kernel version 2, metadata phải chuyển
  source duy nhất sang private dataset; prelaunch audit phải fail nếu còn
  producer kernel source. Validation GT/consumer/test/collaborator output khóa.

### S8 audit-only v2 exact binding và prelaunch PASS (2026-08-02)

- Private dataset `itsthang333/btxrd-skelex-s8-v1-frozen-output` trả trạng thái
  `ready` trong một bounded dataset-status query. Binder lần đầu được gọi bằng
  một full hash gõ sai cho short commit `492e243` nên `git show` fail trước khi
  tạo wrapper/binding; temp package vẫn rỗng. Rerun với exact
  `492e2435e8e2203768b5c4cb80acd66441b7b07e` thành công. Đây là static local
  provenance error đã sửa, không input/prediction/GT hay kernel action.
- Version-2 package ignored tại
  `tmp/kaggle/skelex_reconstruction_selector_s8_audit_v2_launch_492e243` bind
  exact checkout trên, version `2`. Bound wrapper SHA
  `6c7820946159269d86afcb14f005aa4f01fab304a3b45b656733fb4f381d3273`;
  launch-binding raw/canonical-LF SHA
  `9afd529a97ccdb46023981533e1ca31a78e6ec783b7771af702a513c07f1bb26 /`
  `add919baa95c9f57d0f17ea932c2c0ccd08110f60ead8b3401f8efcf834d4842`;
  metadata SHA
  `cf293291fb6d6361ae54bf6d007e6d4b1120e6fd5be2037ccdb7af6170348909`.
  Metadata có duy nhất dataset source frozen-output, `kernel_sources=[]`, private,
  T4; invalid terminal producer kernel source đã bị loại hoàn toàn.
- Static package auditor
  `project/audit_skelex_reconstruction_selector_s8_audit_v2_prelaunch.py` có
  LF SHA `88ffc378d28dfe37570c54010fb74e661a743cc513a7b0b86639fcd14fe91a78`.
  Lần Ruff đầu báo E402 do module docstring đặt sau future import; đã sửa thứ tự
  import, sau đó Ruff/`py_compile` PASS. Prelaunch artifact
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_audit_v2_prelaunch_audit.json`
  canonical-LF SHA
  `1673a4f009fcf2d0759a543cd2302fdbda91e6cb7debd9395bdeae47bf08ab10`
  trả `S8_AUDIT_ONLY_V2_FROZEN_PRELAUNCH_PASS`: inverse template, checkout,
  protocol/auditor/transport corrections, archive/pair, dataset-only metadata,
  T4x2 declaration và mọi safety lock đều khớp.
- Chưa push Kaggle version 2 ở mục này. Chỉ sau commit/push audit+log này và
  final fetch/collision check mới upload đúng package; version 2 chỉ audit exact
  predictions cũ, không inference mới. Validation GT/consumer/test khóa.

### S8 audit-only Kaggle version 2 launch (2026-08-02)

- Final fetch xác nhận central/HEAD sạch và đồng nhất tại
  `32549c8bbac88b043f24562129f7fb40d96b28a9`; collaborator vẫn
  `454b746b1832049e26260460f3130be4e665d2ab`, không có claim cạnh tranh. Kaggle
  nhận `Kernel version 2 successfully pushed` cho private kernel
  `itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1` tại
  https://www.kaggle.com/code/itsthang333/btxrd-skelex-reconstruction-selector-s8-audit-v1 .
- Exact checkout/wrapper/metadata/dataset archive giữ lần lượt
  `492e2435...b07e / 6c782094...3273 / cf293291...8909 / c5164378...ca1737`;
  `kernel_sources=[]`, dataset source duy nhất là frozen-output private dataset,
  T4 hard guard. Launch receipt tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/kernel_version2_launch_receipt.json`.
- Không status-poll sau launch và không tạo monitor. Version 2 không sinh
  prediction/inference; corrected audit PASS mới cho phép dynamic readiness
  freeze. Validation GT, consumer và BTXRD test tiếp tục khóa.
- Bounded status check duy nhất sau launch trả `KernelWorkerStatus.RUNNING`.
  Không poll lần hai, không download partial output/log và không tạo monitor;
  claim S8 tiếp tục `ĐANG LÀM` chờ nhịp terminal sau.

### S8 audit-only version 2 terminal ERROR — null replay sai device (2026-08-02)

- Sau fetch/read/collision check, đúng một bounded status query trả version `2`
  terminal `ERROR`; không repeat poll/monitor. Inventory downloader chỉ lấy
  direct log, không tải 1,539 official output files, vào ignored temp
  `tmp/kaggle/skelex_s8_audit_v2_terminal_error_20260802_192445`. Log `5,691`
  bytes, SHA-256
  `0a1ead7ffc0016dfd558f42663cc4c4a6ba96832a3e1a7430d2f4f6aee210b5d`.
  Exact checkout, two-T4 guard, static test, frozen dataset/archive, pair và all
  manifests pass; corrected auditor chạy đến `IMG000160.jpeg` rồi fail ở 255-null
  distribution, trước readiness/GT/evaluator.
- **Root cause định lượng:** producer chuyển model logits/masks về CPU, tính
  reconstruction error, candidate grid/base logits và toàn bộ
  `select_with_spatial_null` trên CPU. Auditor `_null_improvements` lại tự chọn
  `cuda:0` khi GPU có mặt. Với permutation zero-based `11` của IMG000160, chỉ
  `1/255` entry vượt tolerance: CUDA `0.0900349617`, frozen producer và independent
  CPU replay cùng `0.0957167745`, delta `-0.0056818128`. Có `23` valid candidates;
  magnitude đúng nửa một fused rank-step
  `0.5 * 0.25/(23-1) = 0.0056818182`, xác nhận float32 reduction split/merge tie.
  Exceedance/p-value vẫn exact `193 / 0.7578125`; family gate pass nhưng switch
  `false`, accepted/selected đều index `20`. Lần diagnostic import đầu thiếu
  `project` trong `sys.path`, dừng trước array analysis; rerun đúng path cho các
  số trên. Đây là tooling diagnostic error, không ảnh hưởng output.
- Failure audit tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/kernel_version2_null_device_error_audit.json`,
  SHA-256 `782baae8c660e5ad73572d8e2fb82ed979f7a5defe753223f9747653c2f82840`.
  Version 2 là **LỖI AUDITOR DEVICE**, không phải S8 scientific failure và chưa
  có Dice. Correction duy nhất: force null replay CPU cho đúng frozen producer,
  thêm regression bảo đảm CUDA availability không đổi device, rồi audit-only
  rerun trên pair cũ. Cấm đổi null seed/count/threshold, weights, candidates,
  predictions hoặc đọc GT. Failure-analysis gate hoàn tất; validation GT,
  consumer/test và collaborator output tiếp tục khóa. Phải commit/push mục này
  trước sửa auditor.

### S8 null-device auditor correction freeze (2026-08-02)

- Sau failure gate `7b67923`, auditor correction commit
  `a64d190e5cc9724507a745b167e66458f9f4b4bf` force `_null_improvements` chạy
  CPU đúng frozen producer selector device; scientific source/selector,
  prediction pair, null recipe và gates không đổi. Corrected auditor/test
  canonical-LF SHA lần lượt
  `043d28da1d5dd206eed824191562d66731968bcc4a34e5ccd72f8ce756dd608c /`
  `7e2a0c4ba34ecc287f565125c21f56c379db7d2515b356e42654da820cdb4555`.
- Regression monkeypatch `torch.cuda.is_available=True` và fail nếu bất kỳ tensor
  nào bị chuyển sang CUDA; corrected replay vẫn trả đủ `255` finite null values.
  Focused tests `2 passed`, Ruff/`py_compile`/`git diff --check` PASS. Frozen
  correction addendum tại
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_null_device_audit_correction.json`,
  SHA-256 `be1bb0bf1c253ded4999e78fea164abbfb1c4e1ae412e94e55b8ba5fe8e03725`.
- Chưa bind/launch audit-only v3. Wrapper/binder/prelaunch phải khóa addendum và
  corrected hashes mới, còn exact dataset/archive/pair/protocol giữ nguyên.
  Validation GT, evaluator, consumer và test tiếp tục khóa.

### S8 audit-only v3 binding và prelaunch PASS (2026-08-02)

- Wrapper/binder correction commit `19e1c51` khóa corrected auditor/test
  `043d28da...608c / 7e2a0c4b...4555`, null-device addendum
  `be1bb0bf...3725`, cùng serialized-LCB và transport corrections cũ. Generic
  static package auditor được cập nhật/push tại `d47286d423e85581cf88f29742b2417072c68deb`,
  LF SHA `b8e3b5a8a600ade15fc767d283a82690e3d375dd740aafa1a3af85c13b76a939`;
  nó kiểm tra exact corrected auditor/test bytes ngoài inverse wrapper/protocol/
  dataset/archive/pair/safety contracts.
- Audit-only v3 package ignored tại
  `tmp/kaggle/skelex_reconstruction_selector_s8_audit_v3_launch_d47286d`, bind
  exact checkout trên, kernel version `3`. Bound wrapper SHA
  `4b7a2ac677380d2231fbcdd0d96bf151d55d577f5b6b2434ac5fa8718df6c02a`;
  launch-binding raw/canonical-LF SHA
  `650ba68f5a199e342b7f1c7b9249e723d0ca569db23bba96b86389ee676113a5 /`
  `577318e6fc57c9b2e2dbf3812ecc41bc8a5b1a0863030fc551beeefa356d3aaa`;
  metadata SHA `cf293291...8909` vẫn dataset-only, private T4,
  `kernel_sources=[]`.
- Focused corrected-auditor/transport/binder suite `5 passed`, Ruff và
  `git diff --check` PASS. Prelaunch artifact
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_audit_v3_prelaunch_audit.json`
  canonical-LF SHA
  `5c5213a5ab3ef14a6e36caf8c9eaac997f91098be06ef68a9310781be7d9e1ec`
  trả `S8_AUDIT_ONLY_V3_FROZEN_PRELAUNCH_PASS`.
- Chưa launch v3 ở mục này. Phải commit/push readiness, final fetch/collision
  check rồi mới upload. Không inference/prediction mới; validation GT/evaluator,
  consumer và BTXRD test khóa.

### S8 audit-only Kaggle version 3 launch (2026-08-02)

- Final fetch xác nhận central/HEAD sạch, đồng nhất tại
  `8a558135fa84853e593435ce01b276a78cb27354`; collaborator vẫn `454b746...`,
  không collision. Kaggle nhận `Kernel version 3 successfully pushed` cho
  private audit-only kernel; exact checkout `d47286d...68deb`, wrapper
  `4b7a2ac6...6c02a`, metadata `cf293291...8909`, dataset archive
  `c5164378...ca1737`. Launch receipt tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/kernel_version3_launch_receipt.json`.
- V3 chỉ force independent null replay về producer CPU và audit pair cũ; không
  inference/prediction/scientific change. Không status-poll ngay sau launch và
  không tạo monitor. Validation GT/evaluator, consumer và test khóa.
- Bounded check duy nhất ở nhịp kế trả `KernelWorkerStatus.RUNNING`. Không repeat
  poll, không partial output/log và không monitor; S8 tiếp tục `ĐANG LÀM` chờ
  terminal audit.
- Bounded continuation check kế tiếp vẫn trả `KernelWorkerStatus.RUNNING`.
  Polling dừng ngay; không partial log/output, evaluator/GT hay monitor.
- Bounded continuation check thứ ba vẫn `KernelWorkerStatus.RUNNING`; không có
  terminal evidence mới. Không repeat poll, partial access hoặc monitor.
- Bounded continuation check thứ tư vẫn `KernelWorkerStatus.RUNNING`; CPU null
  replay toàn cohort chưa terminal. Không repeat poll/partial access/monitor.

### S8 audit-only v3 COMPLETE — corrected independent GT-blind audit PASS (2026-08-02)

- Bounded terminal check duy nhất trả version `3` `COMPLETE`. Compact inventory
  downloader lấy đúng ba JSON audit + direct log vào ignored temp
  `tmp/kaggle/skelex_s8_audit_v3_complete_20260802_194008`; official inventory
  `1,546`, compact downloaded `3`, không repeat status query. Direct log `7,509`
  bytes SHA `56ab2d6387dcd3095113e0bfee588617ae4890291d6fed93c1d227e08fc6f1df`.
- Runtime xác nhận checkout `d47286d...68deb`, two T4, corrected tests `2 passed`,
  exact server-expanded private transport và audit hoàn tất ở `114.38 s`.
  `audit_launch_binding.json / corrected_audit_manifest.json /
  independent_gt_blind_output_audit.json` SHA lần lượt
  `61f0b755...33b9 / d6e654ad...bd1d / 5f351b44...2026`; manifest hash-links
  khớp bytes tải về và pair freeze `b2cfd59f...fa00` khớp producer cục bộ.
- Corrected independent audit trả
  `PREDICTION_PAIR_PHYSICALLY_VERIFIED_GT_BLIND_DIAGNOSTICS_REPRODUCED`: đủ
  `371` prediction/arm, `742` physical maps, `371` candidate projections,
  `371` nonconstant reconstruction banks và `371` spatial-null distributions;
  S8 đổi `20` predictions. Audit acceptance tracked tại
  `artifacts/kaggle/skelex_reconstruction_selector_s8_audit_v1/kernel_version3_corrected_audit_acceptance.json`,
  SHA-256 `c6797f47c55b354e1421751e2f7b7852380acd37d3f477f9b1b16e12cb9cb31a`.
- Đây là operational **audit PASS**, chưa phải efficacy/Dice result. Prediction
  không đổi; validation GT chưa đọc, consumer/test khóa. Sau commit/push terminal
  acceptance này mới được chạy one-time dynamic postfreeze readiness freezer;
  evaluator chỉ được mở sau readiness artifact tiếp tục commit/push central.

### S8 dynamic postfreeze readiness frozen trước validation GT (2026-08-02)

- Sau audit acceptance commit `2bc1e15`, one-time freezer chạy trên exact local
  producer root và compact independent audit; không import dataset loader, không
  mở image/mask/GT. Readiness artifact
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_postfreeze_readiness.json`
  có canonical-LF SHA
  `9e0294a5af2d7eb62598eb000f9ee187e2a8c46c07077a52cb432319f294247d`,
  status `FROZEN_AFTER_INDEPENDENT_GT_BLIND_AUDIT_BEFORE_VALIDATION_GT`.
- Artifact khóa protocol `7f819781...bc07`, evaluation addendum
  `d79d9856...aec9`, terminal pre-GT audit `5f351b44...2026`, pair
  `b2cfd59f...fa00`, exact score/prediction/freeze manifests cho cả hai arm,
  cohort `371/184/187`, bootstrap `10,000` seed `20261204`; safety flags đều
  false cho GT/consumer/test/collaborator output.
- Chưa chạy evaluator ở mục này. Phải commit/push readiness và fetch lại central
  byte-visible trước khi mở validation segmentation GT. Sau đó evaluator chạy
  đúng hai frozen arms, audit output inventories, rồi matched decision; không
  sweep/rescue.

### S8 pre-GT evaluator test-lock correction (2026-08-02)

- Sau khi readiness đã visible central nhưng **trước lần gọi evaluator/GT đầu
  tiên**, static call-boundary audit phát hiện evaluator đã khóa dùng canonical
  `BTXRDSegmentationDataset` với full split manifest. Loader này verify mọi hàng
  eligible trước khi lọc `split="val"`; nếu chạy sẽ đọc/hash `373` test images
  và `187` test annotations. Đây là implementation/test-lock defect, không phải
  scientific failure. Evaluator chưa chạy, validation GT chưa đọc và không byte
  test nào được mở; chỉ thống kê cột image-level trong exact split manifest.
- Correction thuần implementation: sau khi full split SHA
  `85511ee1...c8c` đã pass và `371` validation rows được chọn GT-blind, evaluator
  dựng projection CSV LF tạm thời chỉ gồm đúng các hàng validation rồi đưa
  projection đó vào segmentation loader. Vì train/test không hiện diện trong
  projection, loader chỉ verify/mở 371 validation image và 184 validation
  annotations. Projection SHA được ghi vào mỗi evaluation audit; full split vẫn
  là provenance input bất biến.
- Corrected evaluator/test canonical-LF SHA lần lượt
  `757770530baf3253c729230df7be332c9762bfe638cce2ad29d6213a6419d0ca /`
  `fda8dce9af7e22df0a2ab3cfa8e3983832e8f274aa661ba290f94e46fb63c4c4`;
  `py_compile`, focused tests `8 passed` (gồm dynamic deterministic val-only
  projection/rejection test) và `git diff --check` PASS. Frozen
  correction artifact
  `artifacts/research_protocols/skelex_reconstruction_selector_s8_v1_evaluator_test_lock_correction.json`
  canonical-LF SHA
  `30284d598cc9287ace59a6276e52a6c736ac7c37b818ff717c5f4b38a95b937f`
  khóa old/new source, error boundary, readiness/pair/protocol và cùng arm order,
  10,000 bootstrap/seed `20261204`.
- Prediction/score/freeze, scientific algorithm, threshold, arm order và gate
  không đổi; không sweep/rescue. Chỉ sau khi correction + source này commit/push
  và fetch xác nhận byte-visible central mới được đánh giá control → primary →
  matched decision. Consumer và BTXRD test tiếp tục khóa.

