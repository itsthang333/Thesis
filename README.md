# Phân đoạn ảnh xương dựa trên học giám sát yếu (Weakly-Supervised Bone Segmentation)

> **Thesis project** — FracAtlas X-ray dataset  
> Pipeline: **DenseNet121 → LayerCAM → SAM ViT-B → U-Net**

---

## Mục tiêu

Phân đoạn vùng xương (bone vs background) trong ảnh X-quang chỉ sử dụng nhãn phân loại mức ảnh (image-level anatomy labels: `hand`, `leg`, `hip`, `shoulder`) thay vì nhãn pixel.

---

## Pipeline tổng quan

```
X-ray image
  │
  ▼
DenseNet121 (multi-label anatomy classifier)
  │  sigmoid scores per class
  ▼
LayerCAM (multi-layer: denseblock2/3/4, weighted fusion)
  │  confidence-filtered fused CAM [H, W]
  ▼
Adaptive threshold → connected components → peak extraction
  │  point prompts [(row, col), ...]
  ▼
SAM ViT-B (point-prompted, multimask_output=True)
  │  candidate masks [N, H, W]
  ▼
CAM-guided mask selection + fusion (mean / sum / mean_area)
  │  fused binary mask [H, W]
  ▼
Morphological refinement (closing → opening → fill_holes → remove_small)
  │  pseudo mask [H, W]
  ▼
U-Net (BCE + Dice loss)
  │
  ▼
Final segmentation mask
```

---

## Cấu trúc thư mục

```
Thesis/
├── project/                        # Source code
│   ├── train_classifier.py         # Stage 1: train DenseNet121
│   ├── generate_pseudo_masks.py    # Stage 2: LayerCAM + SAM → pseudo masks
│   ├── train_segmentation.py       # Stage 3: train U-Net
│   ├── inference.py                # Stage 4: full pipeline inference (1 ảnh)
│   ├── visualize_pipeline.py       # Debug strip 6-panel (+ --debug flag)
│   │
│   ├── datasets/
│   │   └── fracatlas.py            # Dataset loading, CLAHE, train/val split
│   │
│   ├── models/
│   │   ├── classifier.py           # DenseNet121AnatomyClassifier
│   │   ├── layercam.py             # LayerCAM (hooks on denseblock2/3/4)
│   │   ├── unet.py                 # U-Net (base_channels=64, encoder 64→1024)
│   │   └── losses.py               # bce_dice_loss, dice_coefficient, iou_score
│   │
│   └── pseudo/
│       ├── generate_layercam.py    # generate_fused_cam (weighted CAM fusion)
│       ├── extract_prompts.py      # extract_point_prompts (CAM → SAM prompts)
│       ├── sam_refine.py           # SAMPredictor wrapper
│       ├── mask_selection.py       # score_masks, select_and_fuse_masks
│       ├── morphology.py           # morphological_refinement
│       └── visualization.py        # overlay_heatmap, save_mask, tensor_to_pil
│
├── thesis_experiment.ipynb         # Notebook chạy trên Google Colab
└── FracAtlas/                      # Dataset (không commit vào git)
    ├── dataset.csv
    └── images/
```

---

## Dataset

FracAtlas: [Kaggle link](https://www.kaggle.com/datasets/mahmudulhasantasin/fracatlas-original-dataset)

`dataset.csv` cần có cột `image_id` và các cột nhãn `hand`, `leg`, `hip`, `shoulder` (giá trị 0/1).

```
FracAtlas/
├── dataset.csv
└── images/
    ├── IMG0000001.jpg
    └── ...
```

---

## Cài đặt môi trường

```bash
pip install torch torchvision numpy pillow tqdm opencv-python kagglehub
pip install git+https://github.com/facebookresearch/segment-anything.git
```

SAM checkpoint (375 MB) — tải về thủ công hoặc để `auto_download=True`:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Chạy từng stage

### Stage 1 — Train classifier

```bash
python train_classifier.py \
  --data-root ../FracAtlas \
  --target-columns hand,leg,hip,shoulder \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir outputs/classifier
```

Checkpoint lưu tại `outputs/classifier/best_classifier.pt`.

---

### Stage 2 — Sinh pseudo masks (LayerCAM + SAM)

```bash
python generate_pseudo_masks.py \
  --data-root ../FracAtlas \
  --classifier-checkpoint outputs/classifier/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --target-columns hand,leg,hip,shoulder \
  --image-size 384 \
  --confidence-threshold 0.5 \
  --cam-percentile 85.0 \
  --max-points 5 \
  --mask-score-threshold 0.4 \
  --selection-method mean \
  --output-dir outputs/pseudo_masks
```

**Flags quan trọng:**

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--confidence-threshold` | `0.5` | Sigmoid score tối thiểu để class tham gia CAM fusion |
| `--cam-percentile` | `85.0` | Ngưỡng percentile để tách foreground từ CAM |
| `--max-points` | `5` | Số lượng prompt points tối đa gửi cho SAM |
| `--mask-score-threshold` | `0.4` | Min CAM mean score để giữ mask SAM |
| `--selection-method` | `mean` | `mean` / `sum` / `mean_area` |
| `--debug` | off | Lưu debug output: SAM masks, scores.json, CAM overlay |

Kết quả:
```
outputs/pseudo_masks/
├── masks/          # binary PNG (0=background, 1=bone)
├── overlays/       # LayerCAM overlay PNG
└── debug/<stem>/   # (nếu --debug) mask_*.png, scores.json, foreground.png, ...
```

---

### Stage 3 — Train U-Net

```bash
python train_segmentation.py \
  --data-root ../FracAtlas \
  --mask-root outputs/pseudo_masks/masks \
  --image-size 384 \
  --batch-size 4 \
  --epochs 25 \
  --output-dir outputs/segmentation
```

Checkpoint lưu tại `outputs/segmentation/best_unet.pt`.

---

### Stage 4 — Inference (1 ảnh)

```bash
python inference.py \
  --image-path ../FracAtlas/images/IMG0000019.jpg \
  --classifier-checkpoint outputs/classifier/best_classifier.pt \
  --segmentation-checkpoint outputs/segmentation/best_unet.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --image-size 384 \
  --output-dir outputs/inference
```

Outputs:
```
outputs/inference/
├── <stem>_fused_layercam.png
├── <stem>_pseudo_mask.png
├── <stem>_segmentation_mask.png
└── <stem>_final_overlay.png
```

---

### Visualize pipeline (1 ảnh, debug strip)

```bash
python visualize_pipeline.py \
  --image-path ../FracAtlas/images/IMG0000019.jpg \
  --classifier-checkpoint outputs/classifier/best_classifier.pt \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --selection-method mean \
  --debug \
  --output-path outputs/viz/IMG0000019_pipeline.png
```

Sinh 6-panel figure: **Original | LayerCAM | Foreground | Prompts | SAM masks | Pseudo Mask**

Với `--debug`, thêm vào `outputs/viz/debug/<stem>/`:
- `mask_0.png`, `overlay_mask_0.png`, ... — các mask SAM candidate
- `scores.json` — CAM score + area của từng mask
- `foreground.png` — vùng foreground từ CAM threshold
- `component_*.png` — từng connected component
- `layercam_with_points.png` — CAM overlay với prompt points

---

## Notebook (Google Colab)

`thesis_experiment.ipynb` — chạy toàn bộ pipeline trên Colab, lưu output vào Google Drive.

### Cấu trúc notebook

| Section | Nội dung |
|---------|---------|
| 1. Setup | Mount Drive, clone repo, cài dependencies, tải SAM checkpoint |
| 2. Stage 1 | Train DenseNet121 classifier |
| 3. Stage 2 | Sinh pseudo masks (LayerCAM + SAM) |
| 4. Stage 3 | Train U-Net |
| 5. Inference | Chạy pipeline 1 ảnh, hiển thị kết quả |
| 6. Debug & Experiments | E1–E5: debug SAM masks, CAM coverage, selection comparison, batch 20 ảnh |
| 7. Bottleneck Analysis | Task A/B/C trên 100 ảnh: SAM success rate, bảng GOOD/PARTIAL/BAD, winner/runner-up |
| 8. Multi-Mask Fusion | So sánh top-k union vs intersection vs OR-all |
| 9. Best Strategy Training | Regen pseudo masks + retrain U-Net với strategy tốt nhất, so sánh Dice/IoU |

### Outputs trên Drive

```
ThesisOutputs/
├── classifier/          # best_classifier.pt, training_log.csv
├── pseudo_masks/        # masks/, overlays/
├── segmentation/        # best_unet.pt, training_log.csv
├── inference/           # per-image outputs
├── debug_viz/           # pipeline strips + debug outputs (E1-E3)
├── e4_selection/        # so sánh mean/sum/mean_area (E4)
├── e5_batch20/          # grid 20 ảnh đại diện (E5)
├── bottleneck_100/      # Task A/B/C — 100 ảnh analysis
├── fusion_compare/      # so sánh fusion strategies (Bước 2)
└── pseudo_masks_best_*/  # pseudo masks với best strategy
```

---

## Chi tiết kỹ thuật

### LayerCAM

Khác Grad-CAM: không dùng GAP(gradients) × activations mà dùng **element-wise** `A × relu(G)`.

```
layer_cam = relu(A * relu(G))   # element-wise, không có GAP
```

3 layers được hook: `denseblock2` (weight 0.2), `denseblock3` (0.3), `denseblock4` (0.5).  
Chỉ các class có sigmoid score ≥ `confidence_threshold` tham gia fusion.

### SAM Prompt Strategy

Từ fused CAM:
1. Threshold theo percentile → binary foreground
2. Connected components filtering (min area = 100px)
3. Lấy peak point (argmax CAM) trong mỗi component
4. Gửi từng point riêng lẻ cho SAM với `multimask_output=True` → 3 masks/point

### Mask Selection Methods

| Method | Công thức |
|--------|-----------|
| `mean` | `mean(CAM[mask])` |
| `sum` | `sum(CAM[mask])` — ưu tiên mask lớn |
| `mean_area` | `mean(CAM[mask]) × sqrt(area)` — cân bằng quality + size |

### Fusion Topk (`fusion_topk`)

| Giá trị | Hành vi |
|---------|---------|
| `0` hoặc `1` | OR tất cả masks >= threshold (mặc định) |
| `k > 1` | Union (OR) của top-k masks |
| `k < 0` | Intersection (AND) của top-\|k\| masks |

---

## Ghi chú

- Pipeline không cần GT masks ở Stage 1–2, chỉ cần image-level labels.
- SAM checkpoint **không** được commit vào git (375 MB) — tải về riêng hoặc dùng `auto_download=True`.
- Dataset FracAtlas **không** được commit vào git — tải từ Kaggle.
- Segmentation checkpoint lưu tên `best_unet.pt` (không phải `best.pt`).
