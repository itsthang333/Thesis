# PHÂN TÍCH NOTEBOOK THÍ NGHIỆM WEAKLY SUPERVISED BONE SEGMENTATION

**File notebook được phân tích:** `C:\Users\USER\Downloads\thesis_experiment_debug.ipynb`
**Project đối chiếu:** `D:\thesis\project`
**Ngày phân tích:** 2026-07-02

---

## I. NHẬN XÉT TỔNG QUAN

Notebook hiện tại được trình bày khá tốt cho mục đích demo pipeline: mỗi stage có markdown giải thích, có tham số nằm ngay trước lệnh chạy, có visualization và bảng metric riêng. Về mặt khoa học, notebook đã cho thấy được đường đi của ảnh X-quang từ preprocessing, classifier DenseNet121, LayerCAM, bone morphology, component prompt, SAM, mask selection, hậu xử lý và evaluation với ground truth.

Tuy nhiên, kết quả cuối cùng chưa tốt bằng một stage trung gian. Điểm quan trọng nhất là:

- Bone morphology trước SAM đang cho kết quả rất tốt trên ảnh debug: `bone_support_simple` Dice = 0.8991, IoU = 0.8167, Precision = 0.8455, Recall = 0.9599.
- Raw pseudo mask sau SAM/fusion giảm xuống Dice = 0.8187, IoU = 0.6931, Precision = 0.6953, Recall = 0.9955.
- Final mask sau morphology tiếp tục giảm xuống Dice = 0.8033, IoU = 0.6713, Precision = 0.6713, Recall = 0.9999.

Như vậy, lỗi chính hiện tại không nằm ở việc “không thấy được xương”, mà nằm ở việc SAM và hậu xử lý làm mask nở rộng thành gần silhouette bàn tay/cổ tay, làm tăng false positive rất nhiều.

---

## II. ĐIỂM MẠNH HIỆN TẠI

### 1. Notebook có cấu trúc tốt cho báo cáo khóa luận

Notebook chia rõ từng stage:

- Thiết lập môi trường;
- Kiểm tra dataset;
- Preprocessing;
- Classifier;
- LayerCAM;
- Threshold CAM;
- Bone likelihood / support;
- Component prompt;
- SAM;
- Mask selection;
- Final evaluation.

Đây là cách trình bày phù hợp để giáo sư thấy được pipeline không phải là hộp đen. Đặc biệt, các cell markdown giải thích đúng mục đích của cell, tham số cần chỉnh, và kết quả cần quan sát.

### 2. Có đối chiếu bằng số và bằng hình

Notebook không chỉ show overlay mà còn tính Dice, IoU, precision, recall, foreground ratio, TP/FP/FN. Điều này rất quan trọng vì trong bài toán segmentation, nhìn bằng mắt có thể bị đánh lừa. Ví dụ final mask nhìn có vẻ phủ hết xương, nhưng metric cho thấy precision thấp do dư quá nhiều mô mềm.

### 3. Bone-specific morphology đang là thành phần mạnh nhất

Code trong `project/pseudo/bone_morphology.py` kết hợp:

- Enhanced intensity;
- Cortical edge response;
- CAM như semantic anchor;
- Seed/support percentile;
- Morphological reconstruction;
- Component selection theo CAM và bone likelihood.

Kết quả trên ảnh debug rất đáng khích lệ:

`seed_preview`:

- Dice = 0.8111
- Precision = 0.9094
- Recall = 0.7319

`support_preview`:

- Dice = 0.8977
- Precision = 0.8425
- Recall = 0.9606

`bone_support_simple`:

- Dice = 0.8991
- Precision = 0.8455
- Recall = 0.9599

Đây là bằng chứng mạnh rằng prior về X-quang xương đang có giá trị thực sự. Nó tốt hơn LayerCAM thuần và tốt hơn final mask.

### 4. Có debug component và prompt rõ ràng

Notebook show `component_id`, area, bbox, positive points. Đây là điểm tốt vì với SAM, prompt quality quyết định rất nhiều. Việc show component/prompt giúp phân biệt lỗi do:

- CAM sai;
- Support sai;
- Bbox quá rộng;
- Point rơi vào mô mềm;
- SAM trả mask silhouette;
- Selection score chọn nhầm.

### 5. Pipeline có khả năng mở rộng thành benchmark

Project đã có:

- `generate_pseudo_masks.py` để chạy batch;
- `evaluate_ramh1200_masks.py` để đánh giá với RAM-H1200;
- `train_segmentation.py` cho supervised U-Net baseline.

Đây là nền tảng tốt để chuyển notebook demo một ảnh thành experiment có bảng số trên nhiều ảnh.

---

## III. ĐIỂM YẾU / VẤN ĐỀ KHOA HỌC HIỆN TẠI

### 1. Classifier “hand” không phải nguồn weak supervision đủ mạnh

Trong `project/datasets/ramh1200.py`, `RAMH1200ClassificationDataset` gán target = 1 cho class “hand” cho tất cả ảnh. RAM-H1200 trong project hiện tại chỉ gồm ảnh X-quang bàn tay, nên class hand gần như là nhãn hằng số.

Hệ quả:

- Classifier có probability rất cao: logit = 4.4104, sigmoid = 0.9880;
- Nhưng do tất cả ảnh đều là hand, confidence này không chứng minh model học được localization xương;
- LayerCAM từ nhãn “hand” dễ dàng học silhouette bàn tay, vùng cổ tay, hoặc artifact nên CAM không đủ sắc để tách bone instance.

Đây là vấn đề nên trình bày trung thực trong khóa luận: “hand-level image label” quá yếu cho bài toán binary visible-bone segmentation. Nó có thể dùng làm semantic anchor, nhưng không nên được xem là tín hiệu chính.

### 2. LayerCAM hiện tại bắt đúng bàn tay nhưng chưa phân đoạn xương

Ảnh output LayerCAM cho thấy heatmap có kích hoạt ở lòng bàn tay, ngón cái, một số ngón và cả vùng dưới cổ tay/nền. Overlay LayerCAM phủ lên cả mô mềm của bàn tay, không nằm dọc theo ranh giới xương. Ngoài ra ngón út yếu hơn các vùng khác.

Kết luận: LayerCAM hợp lệ để nói “model đang nhìn vào bàn tay”, nhưng chưa đủ chính xác để nói “model đang nhìn vào xương”. Nếu dùng CAM làm map chính, mask sẽ có xu hướng phủ silhouette.

### 3. Threshold CAM percentile 90 vẫn tạo component nền/cổ tay sai

`CAM_PERCENTILE = 90.0`
Threshold = 0.6361

Component table cho thấy có component lớn ở đáy ảnh:

- Component 5 area = 6246 pixel, bbox gần đáy ảnh y = 335..383.

Trên hình overlay CAM nhị phân, vùng dưới cổ tay/nền bị giữ lại khá rõ. Đây là dấu hiệu CAM có bias theo brightness/background hoặc crop artifact, không chỉ theo bone.

### 4. SAM đang làm xấu hơn kết quả

Đây là vấn đề lớn nhất của pipeline hiện tại.

Trước SAM:

- `bone_support_simple` Dice = 0.8991, IoU = 0.8167.

Sau SAM/fusion:

- `raw_pseudo_mask` Dice = 0.8187, IoU = 0.6931.

Sau final morphology:

- `final_mask` Dice = 0.8033, IoU = 0.6713.

Từ bảng candidate SAM:

- Mask 2 có selection_score cao nhất 0.7739, Dice = 0.7996, Precision = 0.6683, Recall = 0.9952.
- Mask 1 Dice = 0.7427, Precision = 0.5907, Recall = 1.0000.
- Mask 0 Dice = 0.7366, Precision = 0.5830, Recall = 1.0000.

Ba mask tốt nhất theo score đều là mask silhouette bàn tay/cổ tay, không phải mask xương. SAM có xu hướng segment object boundary bên ngoài bàn tay, vì với ảnh X-quang, ranh giới vật thể dễ hơn ranh giới xương chồng lấp bên trong.

Kết luận: SAM ViT-B pretrained natural image không tự nhiên phù hợp với bone mask nội tại trong X-quang. Nếu prompt bằng bbox/point trong bàn tay, SAM thường trả về bàn tay/mô mềm thay vì xương.

### 5. Bbox component 0 quá rộng

Component được chọn lớn nhất:

- `component_id = 0`
- Area = 24439 pixel
- Bbox = (60, 47, 311, 383)
- Positive points = ((383, 206), (232, 183), (168, 148))

Bbox này bao phủ gần toàn bàn tay và kéo xuống đáy cổ tay. Point đầu tiên (383, 206) nằm sát đáy ảnh/cổ tay. Đây là prompt rất dễ khiến SAM lấy cả silhouette bàn tay/cổ tay. Với SAM, bbox rộng + positive point bên trong object thường khuyến khích mask là object lớn nhất trong box.

### 6. Mask selection score chưa phân biệt được xương và mô mềm

Trong `project/pseudo/mask_selection.py`, `bone_hybrid` có các thành phần:

- `bone_mean`;
- `cam_mean`;
- `support_recall`;
- `support_precision`;
- `sam_quality`;
- `large_mask_penalty`;
- `soft_tissue_penalty`.

Ý tưởng đúng, nhưng trên output hiện tại score vẫn xếp cao nhất cho mask lớn. Lý do có thể là:

- `support_recall` ưu tiên mask bao hết `bone_support`, nên mask silhouette được thưởng lớn;
- `sam_quality` của mask silhouette rất cao;
- `bone_support` đã rộng, lại được constrain bằng dilation kernel 11;
- `expected_area = support_area_ratio * 2.6 + 0.03` cho phép mask lớn hơn support quá nhiều.

Kết quả là selection_score đang tối ưu recall hơn là precision.

### 7. Hậu xử lý morphology đang làm xấu kết quả

Bảng stage sau raw pseudo:

Raw pseudo mask:

- Dice = 0.8187
- Precision = 0.6953
- Recall = 0.9955
- FP = 9592

Sau closing k = 5:

- Dice = 0.8076
- Precision = 0.6774
- Recall = 0.9999
- FP = 10470

Sau fill holes <= 500 / final:

- Dice = 0.8033
- Precision = 0.6713
- Recall = 0.9999
- FP = 10763

Closing và fill holes đang làm lấp các khoảng trong bàn tay, nối thêm mô mềm, làm tăng false positive. Với ground truth xương, các khe giữa xương và khớp là thông tin đúng, không nên lấp hết.

### 8. Evaluation mới trên một ảnh là chưa đủ để kết luận

Notebook chọn `DEBUG_IMAGE_INDEX = 0`. Ảnh này là một ca cụ thể, có thể khá “dễ” cho bone morphology vì contrast xương tốt. Cần chạy ít nhất 5–10 ảnh validation để xem lỗi có lặp lại không:

- Ảnh có contrast thấp;
- Ảnh có tay xoay;
- Ảnh có implant/cannula/artifact;
- Ảnh có xương nhỏ bị mờ;
- Ảnh có crop sát ngón/cổ tay.

Nếu chỉ báo cáo một ảnh, giáo sư có thể hỏi đây có phải cherry-pick không.

### 9. `RUN_TRAIN_CLASSIFIER = True` trong notebook debug không phù hợp

Cell classifier đặt `RUN_TRAIN_CLASSIFIER = True`, `EPOCHS_CLASSIFIER = 1`. Dù output cho thấy checkpoint đã tồn tại, về mặt báo cáo nên để False trong notebook demo để tránh thông điệp rằng classifier 1 epoch là đủ tin cậy. Nếu muốn train, nên có notebook/section riêng cho training log và validation.

---

## IV. PHÂN TÍCH THEO TỪNG GIAI ĐOẠN

### 1. Dataset và ground truth

Dataset validation có ảnh debug:

`JP_SARC_P0001_20100423_7383_R.bmp`

Kích thước gốc:

- 1300 × 1700

Resize:

- 384 × 384

GT bone pixels:

- 21986 pixel
- Foreground khoảng 14.9% ảnh.

Resize 384 là chấp nhận được cho debug, nhưng với bone segmentation có chi tiết ngón tay/khớp, nên thử 512 nếu GPU cho phép. SAM và CAM có thể chậm hơn, nhưng ranh giới xương và khe khớp giữ tốt hơn.

### 2. Preprocessing

`input_tensor`:

- Shape = (1, 3, 384, 384)
- Min = -2.1008
- Max = 2.6226
- Mean = -1.2294

Ảnh sau preprocessing nhìn không bị hỏng hình học. Tuy nhiên mean âm mạnh là bình thường khi normalize ImageNet trên ảnh X-quang nền đen, nhưng nó cũng nhắc rằng DenseNet pretrained ImageNet không tối ưu cho X-quang grayscale. Nên thử thêm CLAHE / `use_clahe` và so sánh CAM.

### 3. Classifier

Kết quả:

- Target = hand
- Logit = 4.4104
- Sigmoid/weight = 0.9880
- Qua ngưỡng 0.5 = True

Kết quả này chỉ nói classifier rất tự tin đây là ảnh hand. Vì dataset classification gán hand = 1 cho mọi ảnh, confidence không phải bằng chứng localization tốt. Trong report nên viết rõ classifier chỉ được dùng để lấy gradient/feature cho CAM, không phải kết quả classification có ý nghĩa nghiên cứu mạnh.

### 4. LayerCAM

`fused_cam`:

- Min = 0.0
- Max = 1.0
- Mean = 0.3088

Hình ảnh:

- CAM phủ vùng bàn tay, lòng bàn tay, cổ tay;
- CAM còn nóng ở đáy ảnh;
- CAM không bám sát ranh giới xương;
- Ngón út và một số vùng xương mảnh bị yếu hơn.

Kết luận: LayerCAM hiện tại là coarse localization, không nên threshold trực tiếp thành pseudo mask.

### 5. CAM threshold

`CAM_PERCENTILE = 90.0`
Threshold = 0.6361

CAM nhị phân tách ra một số vùng trên bàn tay nhưng còn vùng lớn ở đáy ảnh. Nếu dùng CAM-only prompt, sẽ rất dễ đưa SAM vào vùng sai. CAM percentile cao hơn có thể giảm nền nhưng sẽ mất recall ở ngón tay; percentile thấp hơn sẽ tràn mô mềm. Vì vậy hướng đúng là giảm vai trò CAM, tăng vai trò bone likelihood/cortical edge.

### 6. Bone likelihood / bone support

Đây là stage tốt nhất hiện tại. Bone likelihood làm nổi xương và cortical edge rất rõ. Support sau reconstruction gần với GT hơn rất nhiều so với CAM.

Bảng metric cho thấy `support_preview` và `bone_support_simple` đều vượt final mask. Điều này gợi ý:

- Có thể dùng `bone_support_simple` làm pseudo label baseline;
- SAM chỉ nên được dùng nếu nó cải thiện ranh giới mà không làm mất precision;
- Nếu SAM làm giảm Dice, nên có cơ chế fallback về bone_support.

### 7. Component và prompt

Component lớn nhất có bbox quá rộng và point sát đáy ảnh. Đây là nguyên nhân trực tiếp dẫn tới SAM lấy cả cổ tay/bàn tay.

Nên tách component lớn thành các thành phần xương/cổ tay nhỏ hơn trước khi đưa sang SAM, hoặc tạo bbox theo skeleton/connected region nhỏ hơn thay vì cả support lớn.

### 8. SAM candidates

SAM trả 12 mask ứng viên. Nhưng candidate có score cao nhất là silhouette. Candidate nhỏ ở các component 1/2/3 có precision cao cho một vùng nhỏ nhưng recall gần 0, không góp đủ được segmentation toàn bộ.

Kết luận: với prompt hiện tại, SAM không segment “visible bone” mà segment “hand object”. Đây là hạn chế bản chất khi dùng SAM pretrained natural image cho X-quang nội tạng.

### 9. Fusion và final mask

`FUSION_TOPK = 3`, `BEST_PER_COMPONENT = True`, `MASK_SCORE_THRESHOLD = 0.4`.

Do `best_per_component` chọn mask tốt nhất mỗi component, component lớn 0 sẽ đưa vào mask silhouette lớn. Fusion OR các mask làm vùng dư tăng. Final morphology lại closing/fill holes nên dư thêm.

Final error map:

- Xanh: xương được bắt gần hết;
- Vàng: dư rất nhiều quanh ngón tay, lòng bàn tay, cổ tay;
- Đỏ: gần như không còn, chỉ 2 FN.

Đây là mẫu lỗi “over-segmentation”. Ưu tiên cải thiện precision, chấp nhận recall giảm nhẹ.

---

## V. GỢI Ý CẢI THIỆN ƯU TIÊN

### 1. Thêm baseline “bone morphology only”

Vì `bone_support_simple` đang Dice 0.899, nên tạo baseline chính thức:

`Final_morph_only = bone_support_simple`, có thể thêm `remove_small_objects` nhẹ, không qua SAM.

So sánh:

- CAM-only;
- Bone morphology only;
- SAM with point;
- SAM with box_point;
- SAM + constrain support;
- Final pipeline.

Nếu SAM không vượt morphology-only trên nhiều ảnh, không nên đặt SAM là thành phần bắt buộc.

### 2. Tắt hoặc giảm hậu xử lý closing/fill holes

Thử các cấu hình:

- `CLOSING_KERNEL = 0` hoặc 1;
- `MAX_HOLE_AREA = 0` hoặc rất nhỏ;
- `OPENING_KERNEL = 0` giữ nguyên;
- `GUIDANCE_THRESHOLD` tăng lên 0.35–0.50;
- `final_mask = raw_pseudo_mask & bone_support` hoặc `raw_pseudo_mask & dilation(bone_support, kernel 3)`.

Lý do: GT xương có nhiều khe khớp và khe giữa xương; closing/fill holes làm mất cấu trúc này.

### 3. Làm ràng buộc support chặt hơn sau SAM

Trong `mask_selection.py`, `constrain_to_bone_support` hiện dùng dilation `kernel_size = 11`. Nên thử:

- `kernel_size = 3` hoặc 5;
- Hoặc không dilation;
- Hoặc `clipped = fused_mask & bone_support`;
- Hoặc `clipped = fused_mask & (bone_likelihood >= percentile 70/75)`.

Mục tiêu là SAM chỉ được refine quanh xương, không được mở rộng ra mô mềm.

### 4. Phạt mask silhouette mạnh hơn trong `bone_hybrid`

Trong `score_masks bone_hybrid`, nên tăng phạt:

- Tăng weight `support_precision`;
- Giảm weight `support_recall`;
- Giảm/bỏ `sam_quality` vì SAM score cao cho silhouette không có nghĩa đúng xương;
- Tăng `large_mask_penalty`;
- Thêm penalty theo boundary/area ratio hoặc expected foreground ratio.

Vì GT foreground của ảnh debug là 14.9%, support tốt là 16.9%, raw/final là 21–22%. Có thể đặt expected_area quanh `support_area_ratio * 1.2` thay vì `* 2.6`.

### 5. Thay prompt mode cho SAM

Thử có hệ thống:

- `SAM_PROMPT_MODE = point`
- `SAM_PROMPT_MODE = joint_points`
- `SAM_PROMPT_MODE = box`
- `SAM_PROMPT_MODE = box_point`
- `NEGATIVE_POINTS_PER_COMPONENT = 2, 4, 8`
- `SAM_SINGLE_MASK = True/False`

Dự đoán:

- `box_point` hiện tại dễ tạo silhouette vì bbox lớn;
- Point-only có thể bớt silhouette nhưng dễ thiếu vùng;
- Negative points quanh bbox có thể cắt mô mềm, nhưng nếu điểm âm sai sẽ cắt xương.

Nên thêm notebook table so sánh Dice/IoU/precision/recall cho từng cấu hình trên 5–10 ảnh.

### 6. Tách component lớn trước khi prompt SAM

Component 0 gom gần toàn bộ bàn tay/cổ tay. Nên thử:

- Skeletonize/thinning support rồi tách theo distance transform;
- Watershed trên bone_likelihood;
- Tách riêng ngón tay và cổ tay theo connected components sau erosion nhẹ;
- Tạo bbox nhỏ quanh các peak thay vì bbox toàn component;
- Giới hạn bbox height/width hoặc diện tích tối đa.

Mục tiêu là SAM nhận nhiều prompt cục bộ thay vì một bbox rộng.

### 7. Xem lại thiết kế weak supervision

Nếu chỉ có nhãn “hand” cho tất cả ảnh, bài toán weakly supervised segmentation rất khó vì không có negative class và CAM không có lý do học ranh giới xương.

Hướng tốt hơn:

- Dùng nhãn hiện diện theo từng bone category từ annotation COCO, nhưng chỉ dùng label image-level, không dùng mask, để train multi-label classifier cho Capitate, Lunate, MC1..MC5, PP/MP/DP... Sau đó CAM từng bone sẽ có tính khu trú hơn “hand”.
- Nếu muốn dùng đúng weak supervision thực tế, tạo annotation điểm/scribble ít hơn mask cho một tập nhỏ, rồi train/refine.
- Dùng self-training: morphology-only pseudo mask chất lượng cao làm label ban đầu, train U-Net, rồi refine bằng teacher-student.
- Dùng supervised U-Net làm upper bound để biết trần kỹ thuật là bao nhiêu.

### 8. Chạy batch evaluation

Cần bật `RUN_BATCH_PSEUDO` và `RUN_BATCH_EVALUATION` cho 5, 10, sau đó full validation. Báo cáo nên có:

- Mean Dice/IoU;
- Std;
- Median;
- Best/worst 5;
- Số ảnh missing/skipped;
- Histogram precision/recall;
- Overlay worst cases.

Một ảnh debug rất hữu ích để giải thích pipeline, nhưng chưa đủ cho kết luận khoa học.

---

## VII. CẬP NHẬT ĐỊNH HƯỚNG CẢI THIỆN PIPELINE HIỆN TẠI

Sau khi xác định lại mục tiêu nghiên cứu, không nên hiểu đề xuất `morphology-only baseline` là bỏ pipeline hiện tại và chỉ dùng hình thái học. Mục tiêu đúng hơn là:

> CAM đóng vai trò định vị giải phẫu thô, còn phân đoạn xương cuối cùng được tạo bằng cách kết hợp CAM với bone prior, SAM/refinement và hậu xử lý có kiểm soát.

Nói cách khác, CAM không cần tự học xương hoặc tự phân đoạn xương. CAM chỉ cần giúp xác định vùng giải phẫu liên quan, ví dụ shape bàn tay/cổ tay. Các bước sau mới chịu trách nhiệm biến vùng đó thành mask xương.

Pipeline nên được bảo vệ theo hướng:

```text
Ảnh X-quang
  -> preprocessing phù hợp X-quang
  -> classifier weak label
  -> CAM định vị shape giải phẫu
  -> bone likelihood / cortical edge / morphology
  -> SAM hoặc refinement có ràng buộc
  -> hậu xử lý nhẹ, không làm dính mô mềm
  -> pseudo bone mask
```

### 1. Vấn đề hiện tại nằm ở đâu?

Từ notebook hiện tại:

- `bone_support_simple`: Dice khoảng 0.899, IoU khoảng 0.817.
- `raw_pseudo_mask` sau SAM/fusion: Dice khoảng 0.819, IoU khoảng 0.693.
- `final_mask` sau hậu xử lý: Dice khoảng 0.803, IoU khoảng 0.671.

Điều này cho thấy pipeline không thất bại ở bước tìm tín hiệu xương ban đầu. Vấn đề chính là:

1. SAM đang làm mask phình ra silhouette bàn tay/cổ tay.
2. Mask selection đang ưu tiên mask có recall cao nhưng precision thấp.
3. Closing và fill holes sau SAM tiếp tục làm tăng false positive.
4. CAM hiện tại định vị được bàn tay ở mức thô, nhưng còn nóng ở nền/cổ tay dưới ảnh và chưa thật sạch để làm localization prior.

Vì vậy hướng cải thiện nên là tinh chỉnh từng khâu trong pipeline hiện tại, không bỏ pipeline.

### 2. Cải thiện preprocessing để CAM định vị shape tốt hơn

Preprocessing hiện tại chủ yếu là resize và normalize ImageNet. Với ảnh X-quang, bước này có thể chưa đủ để CAM ổn định. Nên thử các biến thể preprocessing và so sánh CAM sinh ra, không chỉ nhìn ảnh sau preprocess.

Các cấu hình nên thử:

```text
A. Resize + ImageNet normalize như hiện tại.
B. CLAHE trước khi đưa vào classifier.
C. Percentile contrast stretch, ví dụ clip 1-99%.
D. Gamma correction nhẹ để làm nổi cấu trúc xương/bàn tay.
E. Foreground crop hoặc pad theo vùng không nền trước khi resize.
```

Tiêu chí chọn preprocessing:

- CAM ít nóng ở nền.
- CAM không bị cháy ở mép dưới ảnh.
- CAM bao phủ shape bàn tay/cổ tay hợp lý.
- CAM không bỏ mất ngón tay nhỏ hoặc vùng cổ tay quan trọng.
- Sau khi kết hợp với bone prior, Dice/IoU tăng hoặc precision tốt hơn.

Không nên chọn preprocessing chỉ vì ảnh nhìn đẹp hơn bằng mắt. Nên chọn theo CAM và metric sau pipeline.

### 3. Chỉnh CAM như localization prior, không xem CAM là segmentation

CAM nên được dùng như bản đồ mềm để giới hạn không gian tìm kiếm. Không nên threshold CAM quá cứng rồi kỳ vọng nó là mask xương.

Các tham số nên thử:

```text
CAM_PERCENTILE = 85, 90, 92, 95
Gaussian smoothing nhẹ cho CAM
Loại component CAM chạm biên dưới nếu đó là vùng nền/cổ tay sai
Ràng buộc foreground ratio của CAM trong khoảng hợp lý, ví dụ 10-30%
```

Với prompt map, nên thử giảm vai trò CAM và tăng vai trò bone likelihood:

```text
prompt_map = 0.30 * CAM + 0.60 * bone_likelihood + 0.10 * bone_support
prompt_map = 0.50 * CAM + 0.40 * bone_likelihood + 0.10 * bone_support
prompt_map = 0.20 * CAM + 0.70 * bone_likelihood + 0.10 * bone_support
```

Vì CAM hiện tại còn phình theo silhouette, cấu hình giảm CAM và tăng bone likelihood có khả năng làm mask sạch hơn.

### 4. Cải thiện prompt đưa vào SAM

Đây là phần cần ưu tiên cao. Trong kết quả hiện tại, component lớn nhất có bbox quá rộng, bao gần toàn bộ bàn tay và kéo xuống cổ tay. Khi đưa bbox rộng + point dương vào SAM, SAM rất dễ trả về mask silhouette bàn tay thay vì xương bên trong.

Các hướng thử:

```text
SAM_PROMPT_MODE = point
SAM_PROMPT_MODE = joint_points
SAM_PROMPT_MODE = box_point
NEGATIVE_POINTS_PER_COMPONENT = 2, 4, 8
BBOX_PADDING_RATIO = 0.00, 0.01, 0.02
SAM_SINGLE_MASK = True/False
```

Ưu tiên thử trước:

```text
SAM_PROMPT_MODE = joint_points
NEGATIVE_POINTS_PER_COMPONENT = 4
BBOX_PADDING_RATIO = 0.00 hoặc 0.01
```

Các luật nên thêm cho prompt:

- Không dùng positive point nằm sát biên dưới ảnh, ví dụ row gần 383 trong ảnh 384x384, nếu điểm đó làm SAM bám vào cổ tay/nền.
- Giới hạn diện tích bbox tối đa. Nếu bbox quá lớn, chia component lớn thành nhiều component nhỏ.
- Nếu component chạm biên dưới và area quá lớn, giảm ưu tiên hoặc tách riêng vùng cổ tay.
- Với component lớn, thử erosion nhẹ hoặc watershed/distance transform để tách ngón tay và cổ tay trước khi prompt SAM.
- Thêm negative points quanh bbox để báo cho SAM biết mô mềm/nền không thuộc mask xương.

Mục tiêu của SAM không phải là tự tìm toàn bộ xương, mà chỉ refine cục bộ quanh vùng bone support. Nếu SAM làm mask phình ra silhouette, phải ràng buộc lại ngay sau SAM.

### 5. Ràng buộc mask SAM chặt hơn bằng bone support

Sau khi SAM trả mask, nên cắt mask theo bone support hoặc vùng mở rộng rất nhỏ của bone support.

Hiện tại nếu dilation quá lớn, ví dụ kernel 11, mask vẫn được phép ăn vào mô mềm. Nên thử:

```text
sam_mask_clipped = sam_mask & bone_support
sam_mask_clipped = sam_mask & dilate(bone_support, kernel=3)
sam_mask_clipped = sam_mask & dilate(bone_support, kernel=5)
```

Không nên để SAM tự do mở rộng ra toàn bàn tay. Nếu dùng SAM như refinement, nó phải bị ràng buộc bởi bone likelihood/bone support.

Nên so sánh:

```text
support_clip_kernel = 0, 3, 5, 7, 11
```

Trong đó `0` nghĩa là dùng trực tiếp `bone_support` không dilation. Với lỗi hiện tại là over-segmentation, kernel nhỏ có khả năng tăng precision.

### 6. Sửa scoring chọn mask để phạt silhouette mạnh hơn

Trong `bone_hybrid`, score hiện tại vẫn chọn mask lớn vì mask lớn có recall cao và SAM score cao. Nhưng SAM score cao không đồng nghĩa mask đúng xương; nó chỉ nói SAM tự tin với object mà nó thấy, thường là biên ngoài bàn tay.

Nên chỉnh score theo hướng:

```text
Tăng:
- support_precision
- bone_likelihood_mean
- inside_support_ratio

Giảm:
- support_recall
- sam_score

Thêm penalty:
- area_penalty mạnh hơn
- outside_support_penalty
- border_touch_penalty
- foreground_ratio_penalty
```

Một công thức định hướng:

```text
score =
  0.35 * support_precision
+ 0.30 * bone_mean
+ 0.15 * cam_mean
+ 0.10 * support_recall
+ 0.05 * sam_score
- 1.20 * large_mask_penalty
- 0.80 * outside_support_penalty
- 0.30 * border_touch_penalty
```

Đây chỉ là gợi ý để thử nghiệm, không nên xem là công thức cố định. Mục tiêu là giảm false positive, không tối ưu recall bằng mọi giá.

### 7. Làm nhẹ hậu xử lý sau SAM

Hậu xử lý hiện tại làm kết quả xấu hơn:

- Closing k=5 làm FP tăng.
- Fill holes làm FP tăng.
- Final mask có recall gần 1.0 nhưng precision thấp.

Với xương bàn tay, các khe giữa xương, khớp và khoảng trống giữa cấu trúc xương là thông tin đúng. Không nên fill/closing quá mạnh như trong organ segmentation.

Các cấu hình nên thử:

```text
CLOSING_KERNEL = 0, 1, 3
OPENING_KERNEL = 0
MAX_HOLE_AREA = 0, 50, 100
MIN_FINAL_SIZE = 40 hoặc 80
GUIDANCE_THRESHOLD = 0.35, 0.45, 0.50
```

Ưu tiên đầu tiên:

```text
CLOSING_KERNEL = 0
MAX_HOLE_AREA = 0
GUIDANCE_THRESHOLD = 0.40
```

Sau đó mới tăng nhẹ nếu mask bị đứt quá nhiều.

### 8. Làm thí nghiệm grid nhỏ thay vì chỉnh cảm tính

Không nên chỉnh tham số trên một ảnh duy nhất. Nên chọn 5-10 ảnh validation có đặc điểm khác nhau:

- ảnh contrast tốt;
- ảnh contrast thấp;
- ảnh có cổ tay dài;
- ảnh có ngón bị nghiêng;
- ảnh có artifact/marker;
- ảnh có vùng xương nhỏ khó thấy.

Chạy grid nhỏ:

```text
preprocess_mode: original / CLAHE / contrast stretch
sam_prompt_mode: point / joint_points / box_point
negative_points: 0 / 4
support_clip_kernel: 0 / 3 / 5
closing_kernel: 0 / 3
max_hole_area: 0 / 50
```

Metric cần lưu:

```text
Dice
IoU
Precision
Recall
FP
FN
Foreground ratio
Number of components
```

Vì lỗi hiện tại là over-segmentation, tiêu chí chính là:

```text
Tăng precision, giảm FP, nhưng không để recall tụt quá mạnh.
```

Mục tiêu thực tế:

```text
Precision từ khoảng 0.67 lên 0.78 hoặc cao hơn.
Recall giữ khoảng 0.90-0.97.
Dice vượt raw_pseudo_mask hiện tại và tiến gần hoặc vượt bone_support_simple.
```

### 9. Thứ tự thực hiện đề xuất

Nên làm theo thứ tự sau để dễ kiểm soát:

1. Tạo script/cell evaluation cho 5-10 ảnh validation.
2. Thêm baseline `bone prior only`, `CAM only`, `CAM + bone prior`, `CAM + bone prior + SAM`.
3. Tắt hoặc làm nhẹ hậu xử lý sau SAM.
4. Thêm clipping SAM mask bằng `bone_support` với kernel 0/3/5.
5. Chỉnh `bone_hybrid` để phạt mask lớn và outside-support.
6. Thử prompt SAM: `joint_points`, negative points, bbox nhỏ.
7. Sau khi SAM ổn hơn, quay lại thử preprocessing/CLAHE/contrast stretch để cải thiện CAM.
8. Cuối cùng mới chạy full validation hoặc nhiều ảnh hơn.

Lý do không nên bắt đầu từ preprocessing trước: hiện tại lỗi lớn nhất đo được nằm sau SAM và post-processing. Preprocessing/CAM nên cải thiện sau khi đường SAM-selection-postprocess đã không còn làm hỏng mask.

### 10. Cách diễn giải trong luận

Nên viết rõ:

> Trong pipeline đề xuất, CAM không được sử dụng như mặt nạ phân đoạn xương trực tiếp. CAM đóng vai trò là bản đồ định vị yếu cho vùng giải phẫu. Mặt nạ xương được tạo bằng cách kết hợp CAM với các đặc trưng miền ảnh X-quang như cường độ cản quang, đáp ứng biên vỏ xương và tái tạo hình thái học. SAM được sử dụng như một bước refinement có ràng buộc, không được phép mở rộng tự do ra ngoài vùng bone support.

Và khi báo cáo thí nghiệm, cần chứng minh:

```text
CAM + bone prior tốt hơn CAM-only.
CAM + bone prior tốt hơn hoặc ổn định hơn bone prior-only trên nhiều ảnh.
SAM/refinement chỉ được giữ nếu cải thiện metric hoặc cải thiện chất lượng biên mà không làm precision tụt mạnh.
```
