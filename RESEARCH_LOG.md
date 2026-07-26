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

