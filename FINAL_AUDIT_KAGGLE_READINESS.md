# Final Audit and Kaggle Readiness Report — BTXRD WSSS

Ngày audit: 2026-07-20  
Pipeline canonical: `thesis_final.ipynb` trên branch `pipeline`  
Baseline audit: commit `411ecc4ee649df9e9c592f4a1e04ac8043566789`  
Dataset được kiểm tra read-only: `C:\Users\quang\OneDrive\Desktop\BTXRD`

## A. Final verdict

**NO-GO cho final Kaggle run và NO-GO cho việc công bố số liệu luận văn ở trạng thái hiện tại.**

Phần code đã được harden đáng kể và vượt qua các kiểm tra tĩnh/CPU có thể chạy tại máy audit. Tuy nhiên, không được nâng verdict lên `CONDITIONAL GO` hoặc `GO` vì chưa có:

1. smoke test trên đúng Kaggle GPU image;
2. full run classifier → LayerCAM → morphology/prompt → SAM → pseudo-mask → U-Net;
3. checkpoint classifier, SAM và U-Net khả dụng trong workspace;
4. test PyTorch/CUDA cho loss, forward/backward và resume;
5. bằng chứng split độc lập theo bệnh nhân/ca/tổn thương do bản phát hành BTXRD không có các ID này;
6. final frozen config và locked-test report có checksum.

`GO` chỉ được xem xét sau khi toàn bộ checklist mục L đạt, không có blocker mở, repository ở commit sạch, và artifact cuối được tạo từ đúng commit/split/checkpoint hash.

## B. Repository state

| Thuộc tính | Trạng thái |
|---|---|
| Workspace | `C:\Users\quang\OneDrive\Desktop\Thesis` |
| Branch | `pipeline` |
| HEAD | `411ecc4ee649df9e9c592f4a1e04ac8043566789` |
| Upstream | `origin/pipeline` |
| Working tree | **Dirty** — toàn bộ hardening của audit chưa được commit |
| Canonical notebook | `thesis_final.ipynb`, 25 code cells, outputs đã được xóa |
| Historical notebooks | Không được xem là source of truth của final run |

Các thay đổi sau baseline `411ecc...`:

- Dataset/split: `project/datasets/btxrd.py`, `project/datasets/factory.py`, `project/tools/build_btxrd_split_manifest.py`.
- Classifier: `project/train_classifier.py`, `project/evaluate_classifier.py`, `project/evaluation/classification_metrics.py`.
- Pseudo-label: `project/generate_pseudo_masks.py`, `project/pseudo/extract_prompts.py`, `project/pseudo/morphology.py`, `project/pseudo/tumor_morphology.py`, `project/pseudo/sam_refine.py`, `project/pseudo/mask_selection.py`, `project/pseudo/manifest.py`.
- Segmentation/evaluation: `project/train_segmentation.py`, `project/evaluate_unet.py`, `project/evaluate_pseudo_masks.py`, `project/evaluation/segmentation_metrics.py`.
- Deployment/readiness: `project/inference.py`, `project/config.py`, `project/requirements.txt`, `project/tools/validate_kaggle_readiness.py`, `project/tools/freeze_pipeline_config.py`.
- Orchestration/tests: `thesis_final.ipynb`, `tests/test_pipeline_audit_fixes.py`, `tests/test_torch_math_and_resume.py`.
- Audit artifact: `audit/local_readiness.json` và `outputs/dataset_audit/btxrd_group_v1/*`.

Repository phải được review, commit, push, rồi notebook phải khóa chính xác commit mới; không dùng HEAD `411ecc...` cho final run vì các fix hiện chỉ nằm trong working tree.

## C. Issue verification

| Mức | Phát hiện khi audit | Trạng thái sau sửa | Vị trí/bằng chứng |
|---|---|---|---|
| P0 | Split theo ảnh không đủ chống multi-view/duplicate leakage | Đã thay bằng manifest group-aware + exact SHA-256 guard; vẫn không chứng minh patient independence | `build_btxrd_split_manifest.py`, `btxrd.py`; test group/hash overlap |
| P0 | Split có thể bị suy diễn lại khác nhau giữa các stage | Đã bắt buộc manifest authoritative cho BTXRD và checkpoint lưu hash | dataset factory, train/eval/generator/notebook |
| P0 | `predicted` và known-label protocol có thể bị trộn/ghi đè | Đã tách output, ghi run metadata, mặc định từ chối overwrite | `generate_pseudo_masks.py`, `pseudo/manifest.py` |
| P0 | U-Net có thể train với pseudo-mask thiếu/sai split/sai nội dung | Đã bắt buộc completeness, shape, binary values và SHA-256 từng mask | `btxrd.py`, `pseudo/manifest.py` |
| P0 | Resume U-Net trước đây không đủ state/provenance và có thể tiếp tục sai config | Đã lưu/khôi phục model, best model, optimizer, scaler, RNG, epoch/global step; fail khi split/config/pseudo hash/`pos_weight` lệch | `train_segmentation.py`; integration test phụ thuộc PyTorch |
| P0 | Normal images có thể làm phồng Dice trung bình | Main result nay là tumor-only; normal empty-rate/FPR báo riêng | `evaluation/segmentation_metrics.py`, hai evaluator |
| P0 | Test có nguy cơ được chạy/tuning trước freeze | Mọi test CLI bắt buộc frozen-config schema v3 được kiểm checksum/commit/artifact hashes và classifier-budget audit trước dataset construction | `thesis_final.ipynb`, `evaluation/frozen_test_guard.py`, `freeze_pipeline_config.py` |
| P1 | CAM phẳng/NaN tạo prompt giả ở góc `(0,0)` | Fail-closed: không support, không prompt | `extract_prompts.py`, `tumor_morphology.py`; unit tests |
| P1 | Negative prompt có thể nằm trong component dương hoặc bị fallback trở lại | Đã loại và test negative point ngoài component; bỏ fallback prompt biên | `tumor_morphology.py`, `sam_refine.py` |
| P1 | Candidate dưới threshold hoặc ngoài support có thể được giữ lại ngầm | Production mặc định empty; `keep-best` chỉ debug opt-in; support clip rỗng giữ empty | `mask_selection.py`; unit tests |
| P1 | Morphology guidance threshold có thể fallback về mask không hợp lệ | Đã fail-closed | `morphology.py`; test phụ thuộc PyTorch |
| P1 | Classifier report thiếu confusion/per-class và tumor gate không khớp generator | Đã thêm 10-class confusion, macro/weighted F1, OVR AUROC/AP và gate `argmax` đúng runtime | `evaluate_classifier.py` |
| P1 | AUPRC phụ thuộc thứ tự khi có score ties | Đã dùng non-interpolated Average Precision theo ngưỡng score, tie-invariant | `classification_metrics.py`; unit test |
| P1 | Segmentation report thiếu boundary/object/uncertainty/subgroup | Đã thêm HD95, ASSD, lesion detection, group-bootstrap 95% CI và subgroup | `segmentation_metrics.py` |
| P1 | Final inference còn phụ thuộc classifier+SAM | Đã tách rõ `--mode unet` mặc định; pseudo path chỉ là diagnostic | `inference.py` |
| P1 | Runtime có thể tự tải SAM | Canonical generator và pseudo diagnostic nay bắt local checkpoint, `auto_download=False` | `generate_pseudo_masks.py`, `inference.py` |
| P2 | Không có pseudo-mask ledger/provenance theo ảnh | Đã thêm CSV manifest + summary + run metadata/checksums | `pseudo/manifest.py` |
| P2 | Dependency/SAM source biến động | Đã pin exact versions và SAM official commit `6fdee8f...`; notebook cấm runtime install/download | `requirements.txt`, notebook |
| P2 | Candidate profile cũ bị gọi là “frozen” dù số liệu trước split audit | Đã đổi semantic thành candidate; final chỉ qua checksum freeze | `config.py`, `freeze_pipeline_config.py` |
| P2 | Reproducibility metadata thiếu git/seed/config | Đã bổ sung vào classifier/U-Net checkpoints và run manifests | train/eval scripts |

Tương thích artifact cũ:

- Mọi checkpoint hoặc pseudo-mask sinh trước split manifest hash `15e675...` **không được dùng cho final run**.
- Pseudo-mask cũ không có `pseudo_mask_manifest.csv`/`pseudo_mask_summary.json` sẽ bị loader mới từ chối.
- Resume checkpoint U-Net cũ thiếu resolved-config hash, pseudo-manifest hash hoặc dùng criterion chọn best khác sẽ bị từ chối hoặc chỉ được nhận diện là legacy, không đủ điều kiện final.
- CSV cũ trộn summary rows với per-image rows không được ghép với report mới.
- Các con số `val_f1`/oracle Dice ghi trong comment profile là historical, có trước audited split và **không phải kết quả luận văn cuối**.
- Workspace hiện không có file `*.pt` hoặc `*.pth`; không có checkpoint nào để xác minh end-to-end.

## D. Dataset and split integrity

Kết quả audit trực tiếp BTXRD:

| Kiểm tra | Kết quả |
|---|---:|
| Metadata rows | 3,746 |
| Source images | 3,746 |
| Canonical annotation files | 1,872 |
| Extra annotation copies `(...1)` | 5 |
| Missing metadata/source images | 0 |
| Unreadable images | 0 |
| Invalid canonical annotations | 0 |
| Eligible images | 3,725 |
| Excluded exact-duplicate images | 21 |
| Heuristic groups | 1,470 |
| Exact duplicate groups | 21 |
| Duplicate groups có label conflict | 9 |
| Near-duplicate candidates | 19 |
| Group overlap giữa split | 0 |
| Exact image-hash overlap giữa split | 0 |

Phân vùng authoritative:

| Split | Images | Groups |
|---|---:|---:|
| Train | 2,981 | 984 |
| Validation | 371 | 242 |
| Test | 373 | 244 |

- Seed: `42`; ratio mục tiêu `0.8/0.1/0.1`.
- Split manifest SHA-256: `15e675ff11ab1525de5d49aca408cfb644aa69d3a003a7b4575b206af94a92f0`.
- Một ambiguity ở `IMG001276.jpeg` được sửa trong derived manifest thành `synovial osteochondroma` dựa trên class của annotation khớp; source dataset không bị sửa.
- Năm annotation `IMG001023(1)`, `IMG001166(1)`, `IMG001189(1)`, `IMG001473(1)`, `IMG001536(1)` được ghi nhận là bản copy ngoài canonical naming.
- 19 near-duplicate candidates chỉ là diagnostic; chưa được tự động loại vì perceptual hash không đủ chứng minh cùng bệnh nhân/ca.

Giới hạn khoa học quyết định: BTXRD release không có `patient_id`, `lesion_id`, `case_id`, `study_id` hoặc `accession_id`. `group_id` hiện được suy diễn từ image ID liên tiếp và stable metadata, bỏ qua view. Cách này giảm rò rỉ multi-view rõ ràng nhưng **không chứng minh** test độc lập theo bệnh nhân hoặc tổn thương. Luận văn phải công khai giới hạn này; nếu chủ dataset cung cấp mapping ID, phải rebuild split trước mọi final run.

## E. End-to-end flow verification

Luồng WSSS canonical đã được chuẩn hóa:

1. Image-level `tumor_type` label (10 lớp, class 0 là normal) → DenseNet121 classifier.
2. Known image-level class của train image chọn target LayerCAM. Đây vẫn là weak supervision; polygon/bbox không đi vào classifier/CAM/SAM.
3. LayerCAM từ `denseblock2/3/4` → weighted fusion `0.20/0.30/0.50` → class-vs-normal contrast.
4. CAM thresholds `85/90/95` → tối đa 3 components → box, positive points và negative points.
5. SAM ViT-B tại 512 px, prompt ensemble `box_point`, `point`, `box` → candidate masks.
6. `coverage_mass_sam` chọn top-1 trong mỗi component, threshold `0.4`, clip theo image-derived support, rồi morphology fail-closed.
7. Ghi mask + per-image manifest/checksum; U-Net train chỉ trên pseudo-mask của train split.
8. U-Net checkpoint selection dùng polygon GT của held-out validation split. Đây là model development/selection, không phải weak training label.
9. Sau khi freeze config/checkpoints, đánh giá đúng một lần trên locked test.
10. Deployment cuối: **image → U-Net → mask**. Classifier/CAM/SAM chỉ dùng để tạo weak training labels hoặc diagnostic.

Hai protocol phải giữ riêng:

- `ground_truth`: dùng **known image-level class** để tạo pseudo-mask train; đây là canonical WSSS training protocol.
- `predicted`: classifier tự chọn class; chỉ là inference-realistic diagnostic của pseudo-label generator, không phải final U-Net deployment.

Fully-supervised baseline `image → polygon GT → U-Net` dùng cùng split/image size/training budget nhưng phải báo là upper-bound reference, không phải kết quả WSSS.

Khoảng cách WSSS–fully-supervised chỉ được diễn giải là **observed gap của toàn pipeline hiện tại** (classifier, CAM, SAM, selection, pseudo-label noise và optimization), không phải ước lượng nhân quả riêng cho “weak-label cost”.

## F. Formula and metric verification

### Classifier và WSSS losses

- Softmax: `p(c|x) = exp(z_c) / Σ_j exp(z_j)` cho 10 mutually-exclusive classes.
- Class-balanced CE: `L_cls = -w_y log p(y|x)`, với `w_c = N / (C n_c)` tính chỉ từ train split.
- Tumor gate thực tế: tumor khi `argmax_c p_c != 0`; normal khi argmax là class 0. Score `1-p_0` chỉ dùng AUROC/AP hoặc alternative threshold analysis, không thay đổi default runtime rule.
- LayerCAM mỗi layer: `M_c = ReLU(Σ_k A_k ⊙ ReLU(∂z_c/∂A_k))`; min-max normalization theo sample/layer, upsample, weighted fusion `0.2/0.3/0.5`, rồi normalize lại. Công thức phù hợp ý tưởng gradient theo pixel của [LayerCAM](https://pubmed.ncbi.nlm.nih.gov/34156941/).
- PuzzleCAM hybrid: `L = L_cls(full) + L_cls(reconstructed tiles) + α(t) ||CAM_full-CAM_tiles||_1`; `α` ramp tuyến tính trong nửa đầu epoch rồi giữ cố định. CAM phẳng được normalize thành zero, không khuếch đại floating-point noise. Cấu trúc full/partial consistency phù hợp mục tiêu của [Puzzle-CAM](https://arxiv.org/abs/2101.11253).
- EMA teacher: `θ_T ← d θ_T + (1-d) θ_S`; attention loss là per-sample BCE giữa student CAM và teacher soft target, nhân `confidence²`, tổng trên sample hợp lệ rồi chia số sample hợp lệ. CAM teacher degenerate bị loại, không được học như target all-background.

`btxrd_best` là candidate pure-CE 6 epochs. `btxrd_hybrid` là candidate 25 epochs với PuzzleCAM + EMA attention. Chưa profile nào được coi là final winner sau audited split.

### Mask selection

Với candidate mask `m`:

- `density = |{i∈m: CAM_i>0.5}| / |m|`;
- `mass = Σ_{i∈m} CAM_i / max(Σ_i CAM_i, ε)`;
- `sam_rank` là rank nội bộ trong cùng component, không diễn giải là calibrated probability;
- `score = 0.60·density + 0.25·mass + 0.15·sam_rank`.

Chỉ candidate đạt `score ≥ 0.4` mới được nhận. Production không `keep-best` khi tất cả fail. Mask sau selection được intersect với support; intersection rỗng trả mask rỗng, không hồi sinh candidate ngoài support.

### U-Net loss

- Weighted BCEWithLogits dùng `pos_weight = background_pixels / foreground_pixels`, tính trên actual train masks.
- Soft Dice: `D_soft = (2Σ p_i y_i + ε)/(Σp_i + Σy_i + ε)`.
- `L_UNet = 0.5·BCE_pos_weight + 0.5·(1-D_soft)`.
- Best checkpoint dùng `val_positive_dice` làm metric chính; trong tolerance `1e-4`, `val_normal_empty_case_specificity` là tie-breaker để kiểm soát false-positive trên ảnh normal.

### Reporting metrics

- `Dice = 2TP/(2TP+FP+FN)`; `IoU = TP/(TP+FP+FN)`.
- `Precision = TP/(TP+FP)`; `Recall = TP/(TP+FN)`; pixel specificity `TN/(TN+FP)`.
- Main segmentation population là tumor images only. Normal được báo bằng empty-prediction rate và false-positive case rate.
- Cả GT và prediction rỗng: per-image Dice/IoU = 1 theo convention; nhưng normal images không được trộn vào main tumor Dice.
- Chỉ một mask rỗng: Dice/IoU = 0; HD95/ASSD = undefined và được ghi `null` ở JSON, đồng thời đếm failure.
- HD95 và ASSD đo symmetric surface distance trên grid ảnh đã resize, đơn vị **pixel ở resolution đánh giá**, không phải mm vì dataset không cung cấp pixel spacing đáng tin cậy.
- Lesion detection báo cả any-overlap diagnostic và maximum-cardinality one-to-one matching ở component IoU `0.10`, `0.25`, `0.50`; không diễn giải các chỉ số này như object-detection AP.
- 95% CI dùng nonparametric bootstrap theo toàn bộ `group_id`, không resample từng image. CI vẫn kế thừa giới hạn heuristic group.
- Subgroups: center, anatomy, view, tumor type và lesion-size buckets `<1%`, `1–5%`, `≥5%` image area.
- Classifier AP dùng non-interpolated Average Precision theo distinct score thresholds, không phụ thuộc thứ tự khi tie; AUROC dùng average ranks.

Việc dùng nhiều metric theo failure mode thay vì chỉ Dice phù hợp với khuyến nghị chọn metric theo đặc tính bài toán trong [Metrics Reloaded](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/) và hướng dẫn metric segmentation y khoa [Taha & Hanbury review/guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/).

## G. Kaggle readiness

### Những phần đã sẵn sàng ở mức code

- Notebook không clone repo, không `pip install`, không download SAM khi chạy.
- Bắt branch `pipeline`, exact commit, clean tree, split hash, fresh output directory, CUDA và tối thiểu 25 GiB free disk.
- Generator/pseudo diagnostic bắt buộc local SAM checkpoint.
- Requirements pin exact versions và pin official Segment Anything tại commit [`6fdee8f...`](https://github.com/facebookresearch/segment-anything/commit/6fdee8f2727f4506cfbbe553e23b895e27956588).
- Official SAM usage cũng yêu cầu checkpoint path khi khởi tạo predictor; notebook đã chuyển checkpoint thành immutable attached input theo [SAM README](https://github.com/facebookresearch/segment-anything/blob/main/README.md).
- Mỗi stage có output riêng, checksum/provenance và resume guard.

### Readiness probe hiện tại

- `ready=false`.
- Không có `torch`, `torchvision`, `tqdm`, `pycocotools`, OpenCV, SciPy hoặc `segment_anything` trong audit runtime.
- Installed NumPy/Pillow/Pandas không khớp reference pins.
- Không có CUDA GPU.
- Không có classifier/SAM/U-Net checkpoint.
- Working tree dirty.
- Disk trống khoảng 274 GiB: chỉ tiêu storage local đạt.

Kaggle base image là moving target: official release list hiện tiếp tục thay đổi image/package versions (ví dụ v170 công bố image digest và nhiều package diffs), nên không được suy ra compatibility chỉ vì “Kaggle có sẵn PyTorch”; phải chạy preflight trên đúng session/image thực tế. Đây là suy luận từ [official Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases).

Pipeline canonical chỉ dùng SAM v1 ViT-B; checkpoint và source/wheel phải được attach local, không tải lúc chạy.\n
### Runtime/session feasibility

Một Run All bao gồm 2,981 train pseudo-masks qua DenseNet+LayerCAM+SAM, validation/test diagnostics và hai U-Net tối đa 300 epochs. Chưa có benchmark trên Kaggle, nên không thể cam kết hoàn thành trong một session. Comment lịch sử trong notebook nói một lần chạy U-Net 75 epochs khoảng 5.4 giờ; nếu gần tuyến tính thì một U-Net 300 epochs khoảng 21.6 giờ và hai U-Net vượt 43 giờ, chưa tính pseudo generation. Đây chỉ là ước lượng lịch sử, không phải benchmark đã xác minh.

Khuyến nghị chạy theo stage, lưu Kaggle Dataset artifact giữa các session: classifier → pseudo train/val → WSSS U-Net → supervised baseline → final freeze/locked test. Mỗi resume phải qua hash guard, không copy checkpoint giữa recipe/split khác nhau.

## H. Tests executed

| Kiểm tra | Kết quả |
|---|---|
| `git diff --check` | Pass; chỉ có cảnh báo LF/CRLF |
| `python -m compileall -q project tests` | Pass |
| Compile toàn bộ code cells trong `thesis_final.ipynb` | Pass, 25 cells |
| `python -m unittest discover -s tests -v` | Pass: 25 tests, 19 pass, 6 skip do thiếu PyTorch |
| Dataset inventory/hash/split audit | Pass với các counts ở mục D |
| Local Kaggle readiness probe | Expected fail; artifact `audit/local_readiness.json` |

Sáu test bị skip vì audit runtime không có PyTorch:

1. exact BCE+Dice formula;
2. PuzzleCAM tile/merge;
3. flat PuzzleCAM normalization;
4. U-Net shape + backward;
5. morphology guidance fail-closed ở torch path;
6. uninterrupted-vs-resumed next-step integration.

Do đó chưa có bằng chứng runtime cho training U-Net/resume/GPU dù syntax và logic đã được kiểm tra.

## I. Required full-run commands

Các lệnh dưới đây chạy từ repository root. Thay placeholder bằng absolute path; mỗi output directory phải mới.

```bash
# 0. Chỉ rebuild manifest nếu source dataset/mapping ID thay đổi.
python project/tools/build_btxrd_split_manifest.py \
  --dataset-root <BTXRD_ROOT> \
  --output-dir <AUDIT_DIR> \
  --seed 42

# 1. Preflight trên exact Kaggle runtime/source snapshot.
python project/tools/validate_kaggle_readiness.py \
  --dataset-root <BTXRD_ROOT> \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --sam-checkpoint <SAM_VIT_B_CHECKPOINT> \
  --output-root <NEW_RUN_ROOT> \
  --expected-commit <AUDITED_COMMIT> \
  --report-json <NEW_RUN_ROOT>_preflight.json

# 2. Classifier candidate (không dùng test).
python project/train_classifier.py \
  --pipeline-profile btxrd_hybrid \
  --data-root <BTXRD_ROOT> \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --num-workers 1 \
  --output-dir <NEW_RUN_ROOT>/classifier_btxrd_hybrid

# 3. Classifier + tumor-gate validation report.
python project/evaluate_classifier.py \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --checkpoint <NEW_RUN_ROOT>/classifier_btxrd_hybrid/best_classifier.pt \
  --image-size 320 --gate-rule argmax \
  --output-dir <NEW_RUN_ROOT>/evaluations/classifier_val

# 4. Predicted-class diagnostic trên validation.
python project/generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --classifier-checkpoint <NEW_RUN_ROOT>/classifier_btxrd_hybrid/best_classifier.pt \
  --sam-checkpoint <SAM_VIT_B_CHECKPOINT> \
  --process-all --save-visuals-limit 0 \
  --output-dir <NEW_RUN_ROOT>/pseudo_val_predicted

python project/evaluate_pseudo_masks.py \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --pred-mask-root <NEW_RUN_ROOT>/pseudo_val_predicted/masks \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/pseudo_val_predicted.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/pseudo_val_predicted.json

# 5. Known-image-label localization diagnostic trên validation.
python project/generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --classifier-checkpoint <NEW_RUN_ROOT>/classifier_btxrd_hybrid/best_classifier.pt \
  --sam-checkpoint <SAM_VIT_B_CHECKPOINT> \
  --cam-target-class ground_truth \
  --process-all --save-visuals-limit 0 \
  --output-dir <NEW_RUN_ROOT>/pseudo_val_known_label

python project/evaluate_pseudo_masks.py \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --pred-mask-root <NEW_RUN_ROOT>/pseudo_val_known_label/masks \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/pseudo_val_known_label.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/pseudo_val_known_label.json

# 6. Canonical train pseudo-masks từ weak image-level labels.
python project/generate_pseudo_masks.py \
  --pipeline-profile btxrd_best \
  --data-root <BTXRD_ROOT> --split train \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --classifier-checkpoint <NEW_RUN_ROOT>/classifier_btxrd_hybrid/best_classifier.pt \
  --sam-checkpoint <SAM_VIT_B_CHECKPOINT> \
  --cam-target-class ground_truth \
  --process-all --save-visuals-limit 0 \
  --output-dir <NEW_RUN_ROOT>/pseudo_train_known_label

# 7. WSSS U-Net: train pseudo only; validation là held-out polygon GT.
python project/train_segmentation.py \
  --data-root <BTXRD_ROOT> \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --train-split train --val-split val --image-size 320 \
  --epochs 300 --early-stop-patience 25 \
  --train-pred-mask-root <NEW_RUN_ROOT>/pseudo_train_known_label/masks \
  --output-dir <NEW_RUN_ROOT>/unet_from_pseudo \
  --multi-gpu

python project/evaluate_unet.py \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --checkpoint <NEW_RUN_ROOT>/unet_from_pseudo/best_unet.pt \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/unet_wsss_val.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/unet_wsss_val.json

# 8. Fully-supervised upper-bound baseline, tách riêng.
python project/train_segmentation.py \
  --data-root <BTXRD_ROOT> \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --train-split train --val-split val --image-size 320 \
  --epochs 300 --early-stop-patience 25 \
  --output-dir <NEW_RUN_ROOT>/unet_supervised_oracle \
  --multi-gpu

python project/evaluate_unet.py \
  --data-root <BTXRD_ROOT> --split val \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --checkpoint <NEW_RUN_ROOT>/unet_supervised_oracle/best_unet.pt \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/unet_supervised_val.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/unet_supervised_val.json

# 9. Freeze final sau khi chọn recipe bằng validation và tree vẫn clean.
python project/tools/freeze_pipeline_config.py \
  --profile btxrd_best --status final \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --classifier-checkpoint <NEW_RUN_ROOT>/classifier_btxrd_best/best_classifier.pt \
  --classifier-budget-audit <NEW_RUN_ROOT>/classifier_btxrd_best/classifier_epoch_budget_audit.json \
  --sam-checkpoint <SAM_VIT_B_CHECKPOINT> \
  --unet-checkpoint <NEW_RUN_ROOT>/unet_from_pseudo/best_unet.pt \
  --supervised-unet-checkpoint <NEW_RUN_ROOT>/unet_supervised_oracle/best_unet.pt \
  --output <NEW_RUN_ROOT>/final_frozen_config.json

python project/tools/freeze_pipeline_config.py \
  --output <NEW_RUN_ROOT>/final_frozen_config.json --verify

# 10. Locked test: chạy đúng một lần, không retune sau khi xem kết quả.
python project/evaluate_unet.py \
  --data-root <BTXRD_ROOT> --split test \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --frozen-config <NEW_RUN_ROOT>/final_frozen_config.json \
  --checkpoint <NEW_RUN_ROOT>/unet_from_pseudo/best_unet.pt \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/unet_wsss_test.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/unet_wsss_test.json

python project/evaluate_unet.py \
  --data-root <BTXRD_ROOT> --split test \
  --split-manifest <AUDIT_DIR>/split_manifest.csv \
  --frozen-config <NEW_RUN_ROOT>/final_frozen_config.json \
  --checkpoint <NEW_RUN_ROOT>/unet_supervised_oracle/best_unet.pt \
  --image-size 320 \
  --output-csv <NEW_RUN_ROOT>/evaluations/unet_supervised_test.csv \
  --output-json <NEW_RUN_ROOT>/evaluations/unet_supervised_test.json

# 11. Final deployment smoke: U-Net only.
python project/inference.py \
  --image-path <ONE_UNSEEN_XRAY> \
  --segmentation-checkpoint <NEW_RUN_ROOT>/unet_from_pseudo/best_unet.pt \
  --output-dir <NEW_RUN_ROOT>/deployment_smoke
```

Nếu resume stage 7 hoặc 8, thêm `--resume-from <.../last_unet.pt>` vào đúng command và giữ nguyên split/config/pseudo-mask manifest. Không dùng `--overwrite-existing` trừ khi chủ động chạy lại toàn bộ cùng protocol vào một directory đã kiểm tra.

## J. Expected thesis artifacts

Final package phải có tối thiểu:

- `dataset_audit_summary.json`, `split_manifest.csv`, `image_hash_manifest.csv`, exact/near-duplicate reports.
- Clean git commit ID, branch, diff-free proof và `requirements.txt`/offline dependency bundle.
- SAM source commit/checkpoint SHA-256.
- Classifier: `best_classifier.pt`, `last_classifier.pt`, `training_log.csv`, `training_metadata.json`, CAM previews, `summary.json`, per-image predictions, confusion matrix, run manifest.
- Mỗi pseudo protocol/split: `run_metadata.json`, `masks/*.png`, `pseudo_mask_manifest.csv`, `pseudo_mask_summary.json`, optional prompt-quality/oracle diagnostics.
- WSSS U-Net và supervised baseline, tách directory: `best_unet.pt`, `last_unet.pt`, `training_log.csv`.
- Mỗi segmentation evaluation: per-image CSV, summary JSON, subgroup CSV, bootstrap JSON, pixel-confusion JSON, run-manifest JSON.
- `final_frozen_config.json` và successful verify output.
- Validation report và locked-test report tách riêng; predicted/known-label/fully-supervised labels rõ ràng.
- Qualitative panels có original, GT, prediction, overlay; chọn case theo rule định trước, không cherry-pick sau test.
- Failure audit: empty masks, false-positive normal, missed lesion, worst Dice/HD95, boundary undefined count.
- `notebook_artifacts.json` từ Run All và log stdout/stderr của từng stage.

## K. Remaining blockers

1. **Scientific P0:** không có patient/case/lesion ID; chưa chứng minh independence ở đúng đơn vị lâm sàng.
2. **Execution P0:** chưa có Kaggle smoke/full run, GPU runtime hoặc checkpoints.
3. **Verification P0:** 6 PyTorch tests chưa chạy; không có actual forward/backward/resume parity evidence.
4. **Reproducibility P0:** audit fixes chưa commit; notebook final bắt clean committed snapshot.
5. **Model-selection P1:** `btxrd_best`/`btxrd_hybrid` chỉ là candidates dựa trên historical validation; chưa rerun trên audited split.
6. **Environment P1:** exact dependency/offline SAM bundle chưa được thử trên current Kaggle GPU image.
7. **Runtime P1:** chưa benchmark; full Run All có khả năng vượt một Kaggle session.
8. **Results P1:** chưa có classifier/pseudo/U-Net validation metrics, CI, subgroup hoặc locked-test tables để kiểm tra plausibility.
9. **Annotation P1:** 9 exact-duplicate conflict groups và 19 near-duplicate candidates cần human review/ghi rõ handling; 5 duplicate annotation copies cần xác nhận provenance từ chủ dataset.
10. **Metric limitation P2:** HD95/ASSD chỉ ở resized pixels; không thể báo mm khi thiếu pixel spacing.

## L. Final checklist

### Code/data safeguards

- [x] Canonical notebook xác định rõ và code cells compile.
- [x] Dataset inventory/hash audit được tạo read-only.
- [x] Group/exact-hash overlap giữa train/val/test bằng 0.
- [x] Split manifest authoritative và được hash qua mọi stage.
- [x] Predicted vs known-label protocol tách output.
- [x] Pseudo masks có manifest, completeness, binary/shape/hash validation.
- [x] U-Net train-pseudo và supervised baseline tách riêng.
- [x] Tumor-only main metrics; normal metrics riêng; CI/subgroup artifacts riêng.
- [x] Resume/config/checkpoint provenance guards được triển khai.
- [x] Final inference mặc định U-Net-only.
- [x] Runtime SAM download bị cấm trên canonical paths.

### Bắt buộc trước `CONDITIONAL GO`

- [ ] Review và commit toàn bộ audit changes lên branch `pipeline`.
- [ ] Có mapping patient/case/lesion ID, hoặc hội đồng chấp nhận rõ heuristic-group limitation.
- [ ] Human-review duplicate conflicts/near duplicates và freeze split mới nếu cần.
- [ ] Tạo Kaggle offline dependency/SAM bundle với checksums.
- [ ] Readiness preflight pass trên exact Kaggle GPU session.
- [ ] Chạy toàn bộ tests với PyTorch/CUDA, không skip/fail.
- [ ] Chạy smoke subset qua classifier → CAM → SAM → manifest → U-Net step → inference.
- [ ] Benchmark runtime/storage và chốt staged-session plan.

### Bắt buộc trước `GO`/final thesis numbers

- [ ] Full classifier training + validation report hoàn tất.
- [ ] Full train/validation pseudo-mask generation hoàn tất, manifest `complete=true`.
- [ ] WSSS U-Net và supervised baseline train/resume hoàn tất.
- [ ] Validation metrics/CI/subgroups được review và recipe được freeze.
- [ ] `final_frozen_config.json` verify thành công trên clean commit.
- [ ] Locked test chỉ chạy sau freeze, đúng một lần.
- [ ] Test artifact completeness/provenance/checksums đạt.
- [ ] Thesis tables/figures chỉ lấy từ locked artifacts, không dùng historical comments.
- [ ] Final verdict được audit lại; chỉ khi không còn blocker mới đổi thành `GO`.

**Kết luận cuối tại thời điểm báo cáo: NO-GO.**
