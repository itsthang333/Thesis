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
- **Status:** `ĐANG LÀM`.
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

