# Kế hoạch thực nghiệm và bằng chứng phục vụ phản biện

## 1. Nguyên tắc phát biểu khoa học

Pipeline cuối không phải là một kiến trúc được sao chép nguyên vẹn từ một bài
báo. Nó là một hệ thống được tổng hợp từ các khối đã có tiền lệ: classifier và
CAM, SAM làm proposal generator, MIL để chấm điểm một bag proposal, và rank
fusion để hợp nhất hai scorer khác thang đo. Vì vậy luận văn phải tách ba loại
khẳng định:

1. **Khối có tiền lệ khoa học:** CAM/Grad-CAM/LayerCAM, SAM, MIL, Dice/IoU,
   AUROC/AP, Brier/NLL/ECE, bootstrap ghép cặp.
2. **Cách ghép do đề tài đề xuất:** ba nguồn gallery, G1 mask-bag MIL, upstream
   score, equal percentile-rank fusion.
3. **Siêu tham số do đề tài chọn:** 320/448 px, SAM ViT-B, tối đa 243 mask,
   `0.60D+0.25M+0.15R`, `tau=0.20`, và trọng số fusion `0.5/0.5`.

Loại (1) được bảo vệ bằng trích dẫn. Loại (2) và (3) phải được bảo vệ bằng
ablation trên cùng split, cùng candidate, cùng checkpoint và endpoint Dice;
không được nói rằng paper đã chứng minh đúng các con số của đề tài.

## 2. Endpoint và quần thể

- Quần thể chính: 184 ảnh tumor thuộc validation canonical.
- Endpoint chính: macro mean binary-mask Dice, mỗi ảnh có trọng số bằng nhau.
- Endpoint bổ trợ bắt buộc: IoU, precision, recall, micro Dice/IoU, predicted/GT
  area ratio, RVD, empty-mask rate, zero-overlap rate, HD95 và ASSD theo pixel,
  lesion-level precision/recall/F1 tại IoU 0.10/0.25/0.50, normal false-positive
  case rate trên 187 ảnh normal.
- Phân nhóm khóa trước: `<1%` n=94, `1–<5%` n=72, `>=5%` n=18 theo diện tích
  polygon trên ảnh native.
- Bất định: paired nonparametric bootstrap trên `group_id`; phải ghi rõ đây là
  nhóm heuristic, không được gọi là patient-level nếu không có patient ID.
- Không chọn threshold, mask, source hay scale theo GT từng ảnh. Oracle chỉ là
  diagnostic về candidate supply, không phải kết quả mô hình.
- Test đã được mở ở giao thức cuối; các thí nghiệm G4 chỉ dùng validation và
  không được mở test lại.

## 3. Công thức metric và nguồn gốc

### 3.1 Segmentation

Với `TP`, `FP`, `FN`, `TN` tính trên binary mask:

- `Dice = 2TP / (2TP + FP + FN)`.
- `IoU = TP / (TP + FP + FN)`.
- `Precision = TP / (TP + FP)`.
- `Recall = TP / (TP + FN)`.
- `RVD = (|P|-|G|)/|G|`; đồng thời báo `|P|/|G|` để diễn giải over/under extent.
- `micro Dice` gộp TP/FP/FN trước khi tính; `macro Dice` lấy trung bình Dice từng
  ảnh. Hai metric trả lời hai câu hỏi khác nhau và đều phải ghi rõ.
- `HD95` là max của hai directed 95th-percentile surface distances. `ASSD` là
  trung bình các directed surface samples ghép lại. Chỉ báo theo pixel vì BTXRD
  không cung cấp pixel spacing đáng tin để đổi sang mm.
- Lesion metric dùng connected component 8-neighbour và maximum-cardinality
  one-to-one matching tại ngưỡng IoU đã khóa.

Cơ sở lựa chọn và các failure mode của metric dựa trên
[Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z) và tổng
quan segmentation metrics của
[Taha & Hanbury](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/). Dice cao
không đồng nghĩa boundary tốt hay detect đủ lesion nhỏ, nên không được chỉ báo
một con số Dice.

### 3.2 Classification

- AUROC dùng Mann–Whitney/rank statistic và xử lý tied scores bằng average rank.
- AP là non-interpolated step-wise average precision; toàn bộ mẫu cùng score
  được đưa vào cùng threshold để metric không phụ thuộc thứ tự dòng.
- F1, MCC và balanced accuracy báo tại threshold 0.5 đã khóa.
- Brier score `mean((p-y)^2)` và binary NLL đánh giá xác suất.
- ECE dùng 15 equal-width bins; ECE là diagnostic phụ thuộc binning, không phải
  endpoint duy nhất của calibration. Phải kèm reliability table/plot.

### 3.3 Candidate và selector

- `oracle Dice = max_k Dice(M_k,G)` chỉ đo gallery có chứa mask tốt hay không.
- `selector regret = oracle Dice - selected Dice` tách lỗi supply khỏi lỗi chọn.
- `Recall@Dice=t` là tỷ lệ ảnh có ít nhất một candidate Dice đạt `t`.
- G1 dùng normalized LogSumExp pooling, một MIL pooling trơn; attention/MIL có
  tiền lệ từ [Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html), nhưng
  `tau=0.20`, loss và feature set là lựa chọn của đề tài.
- Percentile-rank xử lý hai scorer khác đơn vị. Reciprocal Rank Fusion là một
  đối chứng đã công bố
  ([Cormack et al.](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/));
  equal percentile-rank `0.5/0.5` vẫn là lựa chọn riêng và cần E8.

## 4. Ma trận thí nghiệm trả lời tám câu hỏi phản biện

### 4.1 Dòng phát triển của baseline, không phải “từ trên trời rơi xuống”

Không có một paper chứa nguyên văn toàn bộ baseline. Dòng lập luận đúng là:

1. [CAM, Zhou et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)
   chứng minh classifier có thể sinh bản đồ định vị yếu từ nhãn ảnh.
2. [Grad-CAM, Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
   mở rộng gradient-based localization; LayerCAM, Jiang et al., TIP 2021,
   DOI `10.1109/TIP.2021.3089943`, khai thác activation ở nhiều tầng để có bản
   đồ chi tiết hơn. E2 kiểm chứng lựa chọn trên BTXRD, không chọn bằng citation.
3. [SAM, Kirillov et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)
   cung cấp promptable mask proposals và predicted-IoU score.
4. [S2C, Kweon et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html)
   là tiền lệ trực tiếp cho việc nối CAM và SAM trong WSSS. Baseline tối giản vì
   vậy là classifier -> LayerCAM -> component/point-box prompt -> SAM -> chọn
   proposal. Rich gallery/G1/fusion là phần mở rộng của đề tài, không quy cho
   S2C.

Ba nguồn gallery cũng có giả thuyết cụ thể, không phải ba module ngẫu nhiên:

- LayerCAM-320 là nguồn anchor có localization tốt nhất trong E2.
- LayerCAM-448 thay đổi sampling scale để candidate supply cho lesion nhỏ không
  bị ràng buộc hoàn toàn bởi lưới 320; nó vẫn phải qua E4 để chứng minh bổ sung.
- BiomedCLIP saliency đưa vào một representation y-sinh học khác họ classifier;
  foundation model có nguồn gốc từ
  [BiomedCLIP](https://arxiv.org/abs/2303.00915). Việc nguồn này hữu ích trên
  BTXRD chỉ được khẳng định bằng source ablation/oracle E4.

G1 xuất hiện vì SAM trả về một **bag** masks chứ không phải đáp án duy nhất.
Attention/MIL cung cấp tiền lệ học từ bag label, nhưng công thức G1 cụ thể là
thiết kế đề tài. Upstream score tồn tại như đối chứng hình học/saliency rẻ và
độc lập với G1. Fusion rank tồn tại vì raw G1 logit và upstream score khác thang
đo; E8, không phải một citation, quyết định rule nào phù hợp dữ liệu này.

| ID | Câu hỏi | Arms/đối chứng bắt buộc | Endpoint | Trạng thái và bằng chứng hiện có |
|---|---|---|---|---|
| E0 | Baseline cuối có thực sự hơn pipeline gốc? | classifier-CAM; CAM+SAM upstream; G1; G1+fusion; fully supervised | Dice/IoU và paired CI | Baseline cuối common-320 Dice 0.288729; LayerCAM cũ 0.234339; fully validation khoảng 0.49. Cần bảng cùng evaluator cuối. |
| E1 | Binary hay 10 lớp? | 3 seed binary vs 3 seed 10-class, DenseNet121/320 cùng budget; sau đó cùng downstream | classification metrics và actual mask Dice | Hoàn tất và audit pass. Mean downstream Dice binary/10-class là 0.262899/0.298954; delta +0.036055. Delta dương ở cả ba seed, nhưng paired CI chỉ loại 0 ở seed 43/44. Lợi ích tập trung ở medium/large; small gần như không đổi. |
| E2 | Vì sao LayerCAM? Vì sao prompt này? | CAM, Grad-CAM, Grad-CAM++, LayerCAM x point/box/box+point | selected Dice, CAM-only Dice, proposal oracle, Recall@Dice, regret | Hoàn tất decomposition 12 arm. Marginal selected Dice CAM/Grad-CAM/Grad-CAM++/LayerCAM = 0.143837/0.137324/0.149826/**0.193677**. LayerCAM-point tốt nhất: selected 0.205224, CAM-only 0.152326, oracle 0.339441, SAM gain +0.052898. Small/medium/large oracle = 0.168793/0.496441/0.602605; regret = 0.050259/0.222191/0.220769. Kết luận: small còn thiếu supply, medium/large chủ yếu lỗi selector. |
| E3 | Vì sao SAM ViT-B? | ViT-B/L/H giữ nguyên source maps, prompts, gallery, selector | Dice, oracle, subgroup, runtime, peak VRAM, disk | Cả ba run và audit độc lập đã xong: Dice B/L/H = 0.288729/0.291185/0.279212; oracle = 0.528298/0.546000/0.510446. L hơn B +0.002456 với CI `[-0.020021,0.024824]` nhưng tốn 1.84x thời gian. H kém B -0.009517, CI `[-0.034714,0.015280]`, và tốn 3.17x thời gian/2.02x peak VRAM. ViT-B là lựa chọn accuracy-cost hợp lý, không phải tối ưu phổ quát. |
| E4 | Vì sao ba nguồn localization? | L320, C448, External, mọi cặp, cả ba | selected Dice, oracle, regret | Hoàn tất: single-source native Dice L320 0.275183, C448 0.257660, External 0.234452; all-source 0.288224. Oracle tăng từ 0.409076/0.429787/0.387303 lên 0.527902. Đây là bằng chứng complementarity, không phải mỗi nguồn riêng đều mạnh. |
| E5 | Vì sao rich gallery và cap 243? | upstream top-1; một exact prompt; 3 SAM multimasks; pre-dedup union; post-dedup; caps 27/81/162/243 | selected Dice, oracle, Recall@Dice, cost | Cap replay một phần hoàn tất: native Dice 0.269759/0.281168/0.283835/0.288224; oracle 0.443435/0.498959/0.520045/0.527902. Exact single-prompt/dedup còn thiếu và là thí nghiệm bắt buộc. |
| E6 | Vì sao G1 và công thức/loss đó? | random, SAM-IoU, upstream, G1, fusion, oracle; sau đó feature/loss ablation | Dice, regret, source error, 3-seed stability | Hoàn tất/audit. Selector controls: native Dice 0.101890/0.098902/0.225306/0.205545/0.288224; oracle 0.527902. Full feature/loss là mean 3 seed tốt nhất nhưng chỉ đạt `0.279855 +/- 0.007371`, thấp hơn R7 0.288224. Một seed bag-negative đạt 0.290598 nhưng không được chọn hậu nghiệm. |
| E7 | Upstream score từ đâu? | U0 SAM; U1 D; U2 M; U3 D+M; U4 D+M+local-R equal; U5 0.60/0.25/0.15; U6 global-R; mỗi arm upstream-only và +G1 | actual Dice + component evidence | Hoàn tất/audit pass 16 arm. Legacy U5+R7 native Dice 0.288224; source-correct U5+R7 0.289358; source-correct U6+R7 cao nhất 0.294956 (common320 0.295568), nhưng paired delta +0.006732 có CI `[-0.010248,0.024597]`. Vì vậy global rank là tín hiệu hứa hẹn, chưa phải cải thiện được chứng minh. |
| E8 | Vì sao equal percentile rank? | upstream, G1, z-score, robust-z, min-max, RRF(k=60), percentile G1 weight .25/.50/.75 | actual Dice, paired CI | Hoàn tất: native z-score 0.286375, RRF 0.285523, equal-rank 0.288224, G1-heavy 0.283986, upstream-heavy 0.269847. Chỉ được nói equal-rank tốt nhất trong tập rule đã khai báo, không phải tối ưu phổ quát. |

## 5. Thí nghiệm bổ sung bắt buộc để bằng chứng đủ mạnh

### P0 — phải có trước khi chốt chương kết quả

1. **E1 downstream binary-vs-10-class:** thay duy nhất classifier/checkpoint và
   collapsed tumor target; tái tạo CAM, gallery và selection. Báo Dice chung,
   ba subgroup và paired CI. Đây mới là bằng chứng trả lời nhãn 2 hay 10 lớp.
2. **E2 candidate decomposition:** trên cùng 12 arm, báo CAM-only p90 Dice,
   proposal oracle, Recall@Dice và selected Dice. Nếu CAM-only thấp nhưng oracle
   cao, bottleneck là selector; nếu oracle thấp, bottleneck ở localization/prompt.
3. **E3 ViT-B/L/H:** cùng seed/protocol, bao gồm peak VRAM, wall time, artifact
   size. Nếu Dice tương đương trong CI, chọn B vì Pareto cost chứ không nói B
   chính xác hơn.
4. **E5 exact construction:** thêm exact `prompt_id` và `multimask_index`; không
   dùng prompt-mode thay thế cho exact prompt. So sánh pre/post dedup và cap.
5. **E6b G1 feature/loss:** tối thiểu geometry-only, appearance-only,
   geometry+appearance; bag loss; +negative bag; +detached positive winner.
   Best và baseline cần lặp 3 seed hoặc báo rõ đây chỉ là one-seed diagnostic.
6. **E7 source-correct:** chạy 14 arm U0–U6, upstream-only và +G1. Hệ số U5 chỉ
   được gọi là được hỗ trợ nếu thắng các đối chứng đã khóa; nếu không, dùng arm
   tốt nhất như exploratory và xác nhận lại trên seed/run độc lập.
7. **Fully-supervised comparison:** cùng canonical validation và cùng bộ metric
   segmentation/subgroup/boundary; nêu rõ đây là upper-reference, không phải
   WSSS comparator cùng supervision.

### P1 — rất nên có cho hội đồng khó tính

8. **Seed robustness:** classifier đã có 3 seed; G1 cần 3 seed cho baseline và
   arm E6b tốt nhất. Báo mean±SD và paired bootstrap theo từng seed.
9. **Qualitative protocol khóa trước:** lấy cố định ảnh theo subgroup và
   outcome: TP tốt, over-segment, under-segment, wrong-location, zero-overlap,
   normal FP. Không chọn hình sau khi xem đẹp/xấu.
10. **Complexity:** trainable parameter count, FLOPs/MACs (nếu tool đáng tin),
    wall time/ảnh, throughput, peak GPU memory và disk/cache. Tách offline gallery
    generation khỏi online G1 selection.
11. **Multiplicity:** khóa một contrast chính cho mỗi E1–E8. Các contrast phụ
    báo effect size/CI; nếu tuyên bố significance cho nhiều cặp, dùng Holm
    correction. Không săn p-value sau khi xem kết quả.

## 6. Metric không nên thêm hoặc chỉ dùng có điều kiện

- Không dùng pixel accuracy làm headline vì nền chiếm đa số.
- Không đổi HD95/ASSD sang mm khi không có spacing.
- Không dùng Normalized Surface Dice nếu chưa có tolerance lâm sàng hoặc
  inter-observer được biện minh; chọn tolerance sau khi xem GT là leakage.
- FROC chỉ thêm nếu pipeline xuất score cho từng connected component và luận văn
  đưa ra claim detection. Nếu chỉ xuất một binary mask, lesion PR/F1 đã đủ rõ.
- Pixel MCC có thể báo phụ nhưng không thay Dice/IoU và lesion metrics.
- Không dùng GT-area routing, oracle mask hoặc per-image GT threshold làm kết quả
  phương pháp; chúng chỉ được ghi là ceiling/diagnostic.

## 7. Bảng cuối cần xuất

1. Bảng end-to-end WSSS so với baseline gốc và fully supervised.
2. Bảng Dice/IoU/precision/recall/area ratio/HD95/ASSD/lesion F1/normal FP.
3. Bảng `<1%`, `1–<5%`, `>=5%` kèm `n` và CI.
4. Bảng E1 binary-vs-10-class gồm classification lẫn downstream Dice.
5. Bảng E2 4 attribution x 3 prompt, selected và oracle.
6. Bảng E3 SAM B/L/H với accuracy-cost Pareto.
7. Bảng E4 source contribution và source oracle.
8. Bảng E5 gallery stages/cap/cost.
9. Bảng E6 selector/loss/features và regret decomposition.
10. Bảng E7 upstream components/formula.
11. Bảng E8 fusion rules.
12. Reliability plot, subgroup box/violin plot, oracle-vs-selected scatter,
    qualitative failure panel và runtime/VRAM chart.

## 8. Tiêu chuẩn kết luận

- “Cải thiện” phải dựa trên actual Dice và paired effect, không dựa vào loss,
  AUROC hay oracle một mình.
- CI cắt 0: nói “xu hướng/không đủ bằng chứng”, không nói “tốt hơn”.
- Oracle tăng nhưng selected không tăng: gallery supply tốt hơn, selector chưa
  khai thác được.
- Selected tăng nhưng subgroup nhỏ giảm: không được gọi là cải thiện toàn diện;
  phải báo trade-off.
- Chỉ chọn cấu hình cuối từ các lựa chọn đã khóa trước GT của lần đánh giá đó.
