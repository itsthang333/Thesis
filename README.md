# BTXRD native-resolution WSSS (RTX 5090)

Pipeline mới để định vị khối u xương nhỏ từ nhãn mức ảnh. Nhánh này không chứa pipeline
DenseNet/LayerCAM cũ. Polygon chỉ được đọc trong lệnh `evaluate`, không đi vào huấn luyện,
tạo proposal, chấm điểm hay chọn mask.

## Pipeline đã khóa

```text
native radiograph
  ├─ HRNetV2-W48 full-view (long side 1536) ── hrnet_full map
  ├─ cùng HRNet trên native tiles 512/1024 ─── hrnet_tile map
  └─ frozen BiomedCLIP full + tiles ────────── biomedclip map
                         │
              component-tree proposals
              8 full + 16 tile + 8 BiomedCLIP
                         │
       SAM ViT-B ROI pass 1: scale 1.5, single mask
       SAM pass 2: chỉ ROI uncertain/novel, scale 3, multimask
                         │
        tối đa 80 raw → gates + source/size diversity → 48
                         │
        frozen RAD-DINO descriptors (2×/4× context, 448)
                         │
            G1 mask-bag MIL MLP (chỉ phần này học)
                         │
       0.5 G1 rank + 0.5 source-correct upstream rank
             + tối đa 6 multifocal unions → native mask
```

HRNet được fine-tune bằng dense image-label MIL. SAM ViT-B, BiomedCLIP và RAD-DINO được
đóng băng. G1 được train trên `selector_holdout_fold`, tách khỏi dữ liệu đã train HRNet.

## Artifact và đánh giá từng giai đoạn

Mọi stage có thể resume theo từng ảnh và ghi vào `experiment.output_dir`:

```text
checkpoints/hrnet_best.pt, hrnet_last.pt, g1_best.pt
calibration/hrnet_normal_cdf.npz
source_maps/<image_id>.npz
raw/galleries/<image_id>.npz
galleries/<image_id>.npz
descriptors/<image_id>.npz
final_masks/<image_id>.png
evaluation/<stage>/per_image.jsonl, summary.json
```

`evaluate` đo dense-map Dice/IoU/pointing accuracy, raw/selected candidate-oracle,
complete-miss, kết quả theo kích thước tổn thương và ablation gallery K=24/36/48/72.

## Chạy cục bộ

```bash
python -m pip install -e '.[train,sam,dev]'
btxrd-wsss --config configs/pipeline.yaml show-config
btxrd-wsss --config configs/pipeline.yaml preflight
pytest
```

Manifest phải giữ split/group hiện có của nghiên cứu. Chỉ dùng `build-manifest` khi thực sự
muốn tạo lại split:

```bash
btxrd-wsss --config configs/pipeline.yaml build-manifest
```

Chạy từng stage:

```bash
btxrd-wsss --config configs/pipeline.yaml train-hrnet
btxrd-wsss --config configs/pipeline.yaml source-maps
btxrd-wsss --config configs/pipeline.yaml sam-gallery
btxrd-wsss --config configs/pipeline.yaml rad-dino
btxrd-wsss --config configs/pipeline.yaml train-g1
btxrd-wsss --config configs/pipeline.yaml select
btxrd-wsss --config configs/pipeline.yaml evaluate --splits val,test
```

## Vast.ai: 1×RTX 5090

Giới hạn mặc định: `$0.60/h`, disk 400 GB, verified host, CUDA 12.8+, direct SSH.

```bash
python -m pip install vastai
vastai set api-key YOUR_KEY
MAX_DPH=0.60 DISK_GB=400 bash scripts/vast/search_fastest.sh

export OFFER_ID=12345678
export OFFER_DPH=0.55
bash scripts/vast/create_fastest.sh
```

Sau khi SSH, clone/copy repo vào `/workspace/native-wsss`, đặt BTXRD và manifest tại các
đường dẫn trong `configs/pipeline.yaml`, rồi:

```bash
bash scripts/vast/bootstrap.sh
bash scripts/vast/install_supervisor.sh
```

Supervisor chạy pipeline nền, tự resume theo checkpoint, thử lại tối đa ba lần và
tự `stop` Vast instance khi hoàn tất hoặc lỗi liên tiếp. Theo dõi bằng
`supervisorctl status btxrd-wsss` và `/workspace/logs/managed.log`.
Trước full run, epoch 1 và 100 ảnh được benchmark riêng. Dự phóng thời gian
được ghi vào `time_budget.json` để theo dõi nhưng không chặn full run. Chỉ thêm
`--enforce` khi chủ đích muốn biến giới hạn này thành hard stop.

`bootstrap.sh` tải checkpoint cần thiết và chạy một forward thật qua HRNet, BiomedCLIP,
SAM ViT-B và RAD-DINO trước khi bắt đầu huấn luyện dài.

Sau epoch HRNet đầu tiên có thể benchmark 100 ảnh; các artifact đó được giữ lại và full run
sẽ resume thay vì tính lại:

```bash
bash scripts/vast/benchmark_100.sh
```

Chi phí compute tối đa ở `$0.60/h`: 12 giờ `$7.20`, 24 giờ `$14.40`, 30 giờ `$18.00`.
Storage và bandwidth phụ thuộc offer. Trước khi destroy instance, chạy
`scripts/vast/sync_outputs.sh`; disk của instance không còn sau khi destroy.
