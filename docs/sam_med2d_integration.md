# SAM-Med2D trong pipeline BTXRD

## Vị trí trong pipeline

HRNet vẫn được huấn luyện bình thường và sinh hai bản đồ định vị trên lưới ảnh gốc:
`hrnet_full` và `hrnet_tile`. BiomedCLIP sinh bản đồ thứ ba. Component-tree biến ba
bản đồ này thành box, điểm dương và điểm âm. Chỉ backend sinh mask được đổi từ SAM
gốc sang SAM-Med2D ViT-B.

## Đầu vào và đầu ra

Với mỗi proposal, backend nhận:

- ảnh X-quang grayscale trên lưới gốc `(H, W)`, kiểu float trong `[0, 1]`;
- box `(x0, y0, x1, y1)`, điểm dương và điểm âm trên cùng lưới ảnh gốc;
- ROI scale `1.5` cho lượt đầu hoặc `3.0` cho lượt mở rộng.

Backend cắt ROI trước, lặp grayscale thành ba kênh RGB uint8, rồi resize giữ tỉ lệ
để cạnh dài của ROI bằng 256. Đây là kích thước chính thức của checkpoint
SAM-Med2D, không phải giảm toàn bộ ảnh X-quang xuống 256. Prompt được đổi tọa độ
theo ROI. Sau suy luận, mỗi kết quả gồm:

- mask boolean `(H, W)` đã đặt lại đúng vị trí trên lưới ảnh gốc;
- `predicted_iou`: đầu ra quality head của SAM-Med2D;
- `stability`: IoU giữa hai mask khi dịch ngưỡng logit quanh `mask_threshold` một
  khoảng `±1`;
- một mask khi `multimask=false`, hoặc ba mask khi `multimask=true`.

## Chấm điểm phù hợp với SAM-Med2D

`predicted_iou` không được xem là xác suất đã hiệu chuẩn trên BTXRD. Sau gate, hệ
thống đổi `predicted_iou` và `stability` thành percentile rank riêng trong từng
nguồn (`hrnet_full`, `hrnet_tile`, `biomedclip`) rồi lấy trung bình làm
`sam_quality`. Cách này giữ thông tin thứ tự nhưng tránh phụ thuộc vào thang điểm
của SAM cũ.

Gate stability phụ thuộc kích thước mask:

| Nhóm mask | Tỉ lệ diện tích | Stability tối thiểu |
|---|---:|---:|
| tiny | `< 0.001` | `0.00` |
| small | `< 0.01` | `0.25` |
| large | còn lại | `0.50` |

Mask vẫn phải chứa đỉnh bản đồ nguồn, phủ ít nhất 25% component nguồn, có ít nhất
4 pixel và không vượt 50% ảnh. Việc nới stability cho tiny/small giữ lại tổn
thương nhỏ có biên logit mong manh; G1 và evidence nguồn sẽ xếp hạng chúng ở các
bước sau.

Trong upstream score, `sam_quality` chỉ chiếm 10% với HRNet và 15% với
BiomedCLIP. Evidence định vị (`contrast`, `purity`, `coverage`, `peak`) giữ phần
lớn trọng số. G1 được huấn luyện lại trên gallery mới; hai feature SAM đưa vào G1
được clip về `[0, 1]`.

## Đánh giá và tinh chỉnh

Stage `sam-gallery` ghi phân phối `predicted_iou`, `stability`, số mask bị loại
theo từng lý do và tỉ lệ giữ lại theo nhóm kích thước. Stage `evaluate` dùng
annotation chỉ để audit, không đưa annotation vào sinh proposal, chọn mask hay
huấn luyện. Báo cáo gồm:

- MAE giữa `predicted_iou` và IoU thật;
- tương quan và rank-correlation của `predicted_iou`, `stability` và điểm kết hợp
  với IoU thật;
- IoU thật của mask đứng đầu theo từng thước đo so với oracle IoU;
- kết quả riêng theo nguồn và theo kích thước tổn thương.

Các số này là cơ sở để đổi ba stability floor hoặc trọng số `sam_quality` sau khi
validation chạy xong. Không nên tăng ngưỡng chỉ vì điểm SAM-Med2D nhìn thấp hơn
SAM gốc.

Nguồn chính: [SAM-Med2D paper](https://arxiv.org/abs/2308.16184),
[official implementation](https://github.com/OpenGVLab/SAM-Med2D), và định nghĩa
[SAM stability score](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py).
