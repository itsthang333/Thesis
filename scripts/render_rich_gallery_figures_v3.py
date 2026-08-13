from __future__ import annotations

import json
import re
from pathlib import Path

import render_rich_gallery_figures_v2 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets" / "rich_gallery_figures_v3"
SOURCE_V2 = ROOT / "report_assets" / "rich_gallery_figures_v2" / "source_images"

base.OUT = OUT
base.SRC = SOURCE_V2
base.ASSETS = OUT / "assets"
base.SVG_DIR = OUT / "svg"
base.HTML_DIR = OUT / "html"
base.RENDERED = OUT / "rendered"

B = base.Box
C = base.COLORS


def segment_hits_box(p1: tuple[float, float], p2: tuple[float, float], b: B, margin: float = 5) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    left, right = b.x - margin, b.x + b.w + margin
    top, bottom = b.y - margin, b.y + b.h + margin
    if abs(y1 - y2) < 0.1:
        lo, hi = sorted((x1, x2))
        return top < y1 < bottom and max(lo, left) < min(hi, right)
    if abs(x1 - x2) < 0.1:
        lo, hi = sorted((y1, y2))
        return left < x1 < right and max(lo, top) < min(hi, bottom)
    return False


class V3Canvas(base.Canvas):
    """Canvas with text-fit and connector clearance checks."""

    def __init__(self, width: int, height: int, name: str):
        super().__init__(width, height, name)
        self.text_fit_issues: list[str] = []
        self.route_records: list[tuple[list[tuple[float, float]], set[str]]] = []

    def module(self, node_id: str, b: B, title: str, subtitle=None, **kwargs) -> None:
        title_size = float(kwargs.get("title_size", 30))
        subtitle_size = float(kwargs.get("subtitle_size", 25))
        title_width = len(title) * title_size * 0.50
        if title_width > b.w - 28:
            self.text_fit_issues.append(f"{node_id}: title {title_width:.0f}>{b.w - 28:.0f}")
        values = [] if subtitle is None else ([subtitle] if isinstance(subtitle, str) else list(subtitle))
        for value in values:
            line_width = len(value) * subtitle_size * 0.47
            if line_width > b.w - 24:
                self.text_fit_issues.append(f"{node_id}: subtitle {line_width:.0f}>{b.w - 24:.0f}")
        super().module(node_id, b, title, subtitle, **kwargs)

    def route(self, points, *, allow=(), **kwargs) -> None:
        self.route_records.append((list(points), set(allow)))
        super().route(points, **kwargs)

    def connect_lr(self, src: str, dst: str, *, mid_x: float | None = None, dashed: bool = False) -> None:
        a, b = self.nodes[src], self.nodes[dst]
        start, end = a.right, b.left
        if abs(start[1] - end[1]) < 1:
            points = [start, end]
        else:
            mx = mid_x if mid_x is not None else (start[0] + end[0]) / 2
            points = [start, (mx, start[1]), (mx, end[1]), end]
        self.route(points, dashed=dashed, allow=(src, dst))

    def audit(self) -> dict[str, object]:
        report = super().audit()
        if self.text_fit_issues:
            raise ValueError(f"Text tràn {self.name}: " + "; ".join(self.text_fit_issues))
        arrow_issues: list[str] = []
        for route_idx, (points, allowed) in enumerate(self.route_records):
            for seg_idx in range(len(points) - 1):
                p1, p2 = points[seg_idx], points[seg_idx + 1]
                for node_id, box in self.nodes.items():
                    if node_id in allowed:
                        continue
                    if segment_hits_box(p1, p2, box):
                        arrow_issues.append(f"r{route_idx}s{seg_idx} {p1}->{p2} cắt {node_id}")
        if arrow_issues:
            raise ValueError(f"Arrow cắt box {self.name}: " + "; ".join(arrow_issues))
        report.update({"text_overflow": 0, "arrow_box_intersections": 0})
        return report


def title_and_section(c: V3Canvas, title: str, section: str | None = None, section_y: float = 90) -> None:
    c.title(title)
    if section:
        c.section(section, section_y)


def fig31(a):
    c = V3Canvas(2500, 1320, "fig_3_1_overview_offline_online_v3")
    c.title("Tổng quan Rich Gallery G1 trong hai pha offline và online")
    c.section("OFFLINE — Chuẩn bị và huấn luyện", 75)
    c.image("off_xray", a["IMG000415_xray"], B(35, 150, 160, 225))
    c.text(115, 410, "Ảnh X-quang", size=25, weight=700)
    c.chip("off_label", B(40, 445, 150, 48), "Nhãn cấp ảnh", size=21)

    c.module("d10", B(290, 125, 260, 120), "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], kind="trained", title_size=30, subtitle_size=23)
    c.chip("ck10", B(650, 160, 140, 50), "θ₁₀", kind="trained", size=24)
    c.module("d2", B(290, 305, 260, 120), "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], kind="trained", title_size=30, subtitle_size=23)
    c.chip("ck2", B(650, 340, 140, 50), "θ₂", kind="trained", size=24)
    c.connect_lr("off_xray", "d10", mid_x=240)
    c.connect_lr("off_xray", "d2", mid_x=240)
    c.connect_lr("d10", "ck10")
    c.connect_lr("d2", "ck2")
    c.route([(115, 493), (240, 493), (240, 185), (290, 185)], dashed=True, allow=("off_label", "d10"))
    c.route([(115, 493), (265, 493), (265, 365), (290, 365)], dashed=True, allow=("off_label", "d2"))

    c.module("bio", B(875, 115, 200, 86), "BiomedCLIP", "Đóng băng", kind="frozen", frozen=True, title_size=27, subtitle_size=21)
    c.module("sam", B(1115, 115, 190, 86), "SAM ViT-B", "Đóng băng", kind="frozen", frozen=True, title_size=27, subtitle_size=21)
    c.module("generate_train", B(910, 275, 310, 145), "Sinh tập ứng viên", ["định vị → prompt", "→ SAM"], kind="logic", designed=True, title_size=28, subtitle_size=23)
    c.route([(790, 185), (830, 185), (830, 330), (910, 330)], allow=("ck10", "generate_train"))
    c.route([(790, 365), (850, 365), (850, 380), (910, 380)], allow=("ck2", "generate_train"))
    c.route([(975, 201), (975, 275)], dashed=True, allow=("bio", "generate_train"))
    c.route([(1210, 201), (1210, 240), (1135, 240), (1135, 275)], dashed=True, allow=("sam", "generate_train"))
    c.route([(195, 262), (225, 262), (225, 535), (860, 535), (860, 348), (910, 348)], allow=("off_xray", "generate_train"))

    c.module("train_gallery", B(1325, 290, 245, 115), "Tập ứng viên", "huấn luyện", kind="data", title_size=28, subtitle_size=23)
    c.module("rad", B(1630, 115, 205, 86), "RAD-DINO", "Đóng băng", kind="frozen", frozen=True, title_size=28, subtitle_size=21)
    c.module("repr", B(1610, 290, 245, 115), "Biểu diễn", "ứng viên", kind="feature", title_size=29, subtitle_size=23)
    c.module("g1", B(1945, 275, 230, 145), "G1", ["MIL cấp bag", "học chấm điểm"], kind="trained", title_size=34, subtitle_size=23)
    c.chip("ckg1", B(2280, 323, 150, 50), "θ_G1", kind="trained", size=24)
    c.connect_lr("generate_train", "train_gallery")
    c.connect_lr("train_gallery", "repr")
    c.route([(1732, 201), (1732, 290)], dashed=True, allow=("rad", "repr"))
    c.connect_lr("repr", "g1")
    c.connect_lr("g1", "ckg1")
    c.route([(115, 493), (115, 565), (2060, 565), (2060, 420)], dashed=True, allow=("off_label", "g1"))

    c.section("ONLINE — Suy luận và chọn mặt nạ", 630)
    c.image("on_xray", a["IMG000415_xray"], B(35, 720, 160, 225))
    c.text(115, 980, "Ảnh X-quang", size=25, weight=700)
    c.chip("on_label", B(40, 1015, 150, 48), "Nhãn cấp ảnh", size=21)
    c.module("gate", B(285, 755, 225, 105), "Cổng nhãn", "cấp ảnh", kind="data", title_size=29, subtitle_size=23)
    c.module("normal", B(285, 1090, 225, 95), "Bình thường", "mặt nạ rỗng", kind="eval", title_size=27, subtitle_size=22)
    c.connect_lr("on_xray", "gate")
    c.route([(115, 1063), (235, 1063), (235, 830), (285, 830)], dashed=True, allow=("on_label", "gate"))
    c.route([(397, 860), (397, 1090)], allow=("gate", "normal"))

    c.module("loc", B(610, 735, 275, 145), "Ba nguồn định vị", ["LayerCAM-320/448", "BiomedCLIP saliency"], kind="logic", designed=True, title_size=28, subtitle_size=22)
    c.module("heatmaps", B(990, 750, 235, 115), "Ba heatmap", "giữ riêng từng nguồn", kind="feature", title_size=29, subtitle_size=21)
    c.module("generate", B(1330, 730, 270, 155), "Sinh ứng viên", ["phân vị → prompt", "→ SAM ViT-B"], kind="logic", designed=True, title_size=29, subtitle_size=23)
    c.module("gallery", B(1690, 750, 270, 115), "Candidate Gallery", "M_i", kind="data", title_size=27, subtitle_size=24)
    c.connect_lr("gate", "loc")
    c.connect_lr("loc", "heatmaps")
    c.connect_lr("heatmaps", "generate")
    c.connect_lr("generate", "gallery")

    c.module("g1score", B(2040, 670, 220, 105), "RAD-DINO → G1", "hạng G1 · θ_G1", kind="feature", title_size=24, subtitle_size=21)
    c.module("srcscore", B(2040, 865, 220, 95), "Điểm nguồn", "hạng nguồn", kind="feature", title_size=28, subtitle_size=22)
    c.connect_lr("gallery", "g1score", mid_x=1990)
    c.connect_lr("gallery", "srcscore", mid_x=1990)
    c.module("fusion", B(2325, 740, 155, 145), "Hợp nhất", ["thứ hạng", "0,5 / 0,5"], kind="logic", designed=True, title_size=26, subtitle_size=21)
    c.connect_lr("g1score", "fusion", mid_x=2290)
    c.connect_lr("srcscore", "fusion", mid_x=2290)
    c.module("select", B(2220, 1005, 210, 90), "Chọn ứng viên", kind="logic", designed=True, title_size=25)
    c.route([(2402, 885), (2402, 960), (2325, 960), (2325, 1005)], allow=("fusion", "select"))
    c.image("final", a["selected_mask_real"], B(2250, 1150, 150, 120), stroke=C["output_border"], fit="contain")
    c.text(2325, 1300, "Mặt nạ nhị phân cuối", size=25, weight=700, fill=C["output_border"])
    c.route([(2325, 1095), (2325, 1150)], allow=("select", "final"))
    return c.save()


def fig32(a):
    c = V3Canvas(2000, 1020, "fig_3_2_multisource_localization_v3")
    c.title("Trích xuất bằng chứng định vị từ ba nguồn bổ sung")
    c.image("input", a["IMG000415_xray"], B(35, 330, 170, 270))
    c.text(120, 635, "Ảnh X-quang", size=27, weight=700)
    columns = [
        ("b320", 275, "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], "LayerCAM", ["Dense Block 2/3/4", "trọng số 0,2 / 0,3 / 0,5"], "heatmap_layercam320", "LayerCAM-320"),
        ("b448", 855, "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], "LayerCAM", ["Dense Block 2/3/4", "trọng số 0,2 / 0,3 / 0,5"], "heatmap_layercam448", "LayerCAM-448"),
        ("bbio", 1435, "BiomedCLIP", ["Toàn ảnh + vùng cục bộ"], "Saliency", ["ảnh–văn bản"], "heatmap_biomedclip", "BiomedCLIP saliency"),
    ]
    centers = []
    for node, x, model, model_sub, op, op_sub, image_key, label in columns:
        centers.append(x + 180)
        c.module(node, B(x, 120, 360, 110), model, model_sub, kind="frozen" if node == "bbio" else "trained", frozen=node == "bbio", title_size=30, subtitle_size=24)
        c.module(node + "_op", B(x + 35, 300, 290, 120), op, op_sub, kind="logic", designed=True, title_size=30, subtitle_size=21)
        c.chip(node + "_heat", B(x + 75, 480, 210, 48), "Heatmap định vị", kind="logic", size=22)
        c.image(node + "_img", a[image_key], B(x + 40, 590, 280, 280), stroke=C["logic_border"], fit="slice")
        c.text(x + 180, 910, label, size=28, weight=700)
        c.text(x + 180, 945, "Bản đồ định vị", size=22, fill=C["muted"])
        c.route([(x + 180, 230), (x + 180, 300)], allow=(node, node + "_op"))
        c.route([(x + 180, 420), (x + 180, 480)], allow=(node + "_op", node + "_heat"))
        c.route([(x + 180, 528), (x + 180, 590)], allow=(node + "_heat", node + "_img"))
    c.route([(205, 462), (245, 462), (245, 82), (centers[-1], 82)], arrow=False, allow=("input",))
    for center, (node, *_rest) in zip(centers, columns):
        c.route([(center, 82), (center, 120)], allow=(node,))
    c.route([(275, 985), (1795, 985)], arrow=False, sw=2.6, color=C["logic_border"])
    c.text(1035, 1010, "Giữ riêng từng nguồn — không trung bình bản đồ", size=27, weight=700, fill=C["logic_border"])
    return c.save()


def fig33(a):
    c = V3Canvas(2000, 1080, "fig_3_3_candidate_gallery_generation_v3")
    c.title("Sinh tập mặt nạ ứng viên từ bằng chứng định vị")
    for idx, key in enumerate(("heatmap_layercam320", "heatmap_layercam448", "heatmap_biomedclip")):
        c.image(None, a[key], B(30, 105 + idx * 135, 155, 115), fit="slice", register=False)
    c.text(107, 535, "Ba bản đồ định vị", size=26, weight=700)
    c.text(107, 570, "L320 · L448 · Bio", size=22, fill=C["muted"])
    c.module("threshold", B(255, 225, 205, 125), "Ngưỡng phân vị", ["P85 · P90 · P95", "riêng từng nguồn"], kind="logic", designed=True, title_size=25, subtitle_size=21)
    c.module("cc", B(535, 215, 220, 145), "Thành phần liên thông", ["giữ tối đa 3 vùng"], kind="logic", designed=True, title_size=18, subtitle_size=22)
    c.module("prompt", B(835, 190, 220, 190), "Tạo prompt", ["Điểm / Hộp / Kết hợp", "điểm âm từ", "vùng lân cận"], kind="logic", designed=True, title_size=30, subtitle_size=20)
    c.image("prompt_real", a["IMG000415_layercam320_p90_real"], B(1140, 115, 310, 315), stroke=C["logic_border"], fit="slice")
    c.text(1295, 470, "Ví dụ: LayerCAM-320 tại P90", size=23, weight=700)
    c.module("sam", B(1535, 220, 195, 135), "SAM ViT-B", "multimask = 3", kind="frozen", frozen=True, title_size=28, subtitle_size=22)
    c.route([(185, 290), (255, 290)], allow=("threshold",))
    c.connect_lr("threshold", "cc")
    c.connect_lr("cc", "prompt")
    c.connect_lr("prompt", "prompt_real")
    c.connect_lr("prompt_real", "sam")
    for idx in range(6):
        x = 1800 + (idx % 2) * 82
        y = 110 + (idx // 2) * 105
        c.image(None, a[f"candidate_{idx + 1:02d}"], B(x, y, 70, 70), rx=8, register=False)
    c.text(1875, 455, "Ứng viên thật", size=24, weight=700)
    c.route([(1730, 287), (1780, 287)], allow=("sam",))

    c.module("grid", B(300, 690, 220, 100), "Quy về lưới chung", "L448: 448 → 320", kind="logic", designed=True, title_size=22, subtitle_size=21)
    c.module("merge", B(620, 690, 220, 100), "Gộp ba nguồn", kind="logic", designed=True, title_size=27)
    c.module("dedupe", B(940, 675, 250, 130), "Loại trùng", ["hoàn toàn", "theo từng điểm ảnh"], kind="logic", designed=True, title_size=28, subtitle_size=22)
    c.module("gallery", B(1290, 670, 280, 140), "Candidate Gallery (M_i)", ["K_i ứng viên duy nhất", "giữ metadata + điểm nguồn"], kind="data", title_size=21, subtitle_size=19)
    c.connect_lr("grid", "merge")
    c.connect_lr("merge", "dedupe")
    c.connect_lr("dedupe", "gallery")
    c.route([(1875, 475), (1875, 620), (410, 620), (410, 690)], allow=("grid",))
    c.module("source_score", B(1660, 665, 310, 150), "Điểm nguồn", ["D — Mật độ · M — Khối lượng", "R — Hạng SAM", "0,60D + 0,25M + 0,15R"], kind="feature", title_size=29, subtitle_size=19)
    c.route([(1875, 475), (1935, 475), (1935, 640), (1815, 640), (1815, 665)], dashed=True, allow=("source_score",))
    c.route([(1660, 740), (1605, 740), (1605, 810), (1570, 810)], dashed=True, allow=("source_score", "gallery"))
    c.text(300, 940, "3 phân vị × ≤3 vùng × 3 kiểu prompt × 3 mặt nạ SAM", size=25, weight=700, anchor="start")
    c.text(300, 985, "≤ 81 ứng viên / nguồn", size=28, weight=700, anchor="start", fill=C["logic_border"])
    c.text(1290, 985, "≤ 243 ứng viên / ảnh trước lọc", size=28, weight=700, anchor="start", fill=C["logic_border"])
    return c.save()


def fig34(a):
    c = V3Canvas(2000, 880, "fig_3_4_candidate_representation_g1_v3")
    c.title("Biểu diễn ứng viên và mạng chấm điểm G1")
    c.image("xray", a["IMG000415_xray"], B(40, 120, 160, 220))
    c.text(120, 375, "Ảnh X-quang (Iᵢ)", size=25, weight=700)
    c.image("mask", a["candidate_01"], B(55, 560, 135, 135))
    c.text(122, 735, "Ứng viên (mᵢⱼ)", size=25, weight=700)
    c.module("rad", B(285, 105, 300, 115), "RAD-DINO", ["Đặc trưng lớp 4 · 8 · 12", "Đóng băng"], kind="frozen", frozen=True, title_size=30, subtitle_size=22)
    c.module("proj", B(760, 105, 240, 105), "Chiếu cố định", "768 → 128", kind="feature", title_size=28, subtitle_size=25)
    c.connect_lr("xray", "rad", mid_x=240)
    c.connect_lr("rad", "proj")
    c.module("pool", B(400, 500, 250, 175), "Pooling theo mặt nạ", ["Trong vùng", "Ngữ cảnh", "Tương phản"], kind="logic", designed=True, title_size=23, subtitle_size=24)
    c.connect_lr("mask", "pool", mid_x=300)
    c.route([(1040, 210), (1040, 425), (525, 425), (525, 500)], allow=("proj", "pool"))
    c.module("radfeat", B(760, 490, 280, 150), "Đặc trưng RAD-DINO", ["3 tầng × 3 thống kê × 128", "1152 chiều"], kind="feature", title_size=25, subtitle_size=21)
    c.connect_lr("pool", "radfeat")
    c.module("aux", B(760, 700, 340, 135), "4 đặc trưng bổ sung", ["SAM score · log(tỷ lệ diện tích)", "khối lượng M · đáp ứng trung bình μA"], kind="feature", title_size=25, subtitle_size=18)
    c.module("concat", B(1160, 500, 245, 140), "Ghép đặc trưng", "xᵢⱼ ∈ ℝ¹¹⁵⁶", kind="feature", title_size=29, subtitle_size=25)
    c.connect_lr("radfeat", "concat")
    c.route([(1040, 767), (1100, 767), (1100, 595), (1160, 595)], allow=("aux", "concat"))
    c.module("g1", B(1510, 480, 235, 180), "G1", ["LayerNorm → 256", "→ 128 → 1", "GELU · Dropout 0,10"], kind="trained", title_size=35, subtitle_size=23)
    c.module("logit", B(1830, 520, 155, 110), "Logit", "ứng viên aᵢⱼ", kind="feature", title_size=28, subtitle_size=22)
    c.connect_lr("concat", "g1")
    c.connect_lr("g1", "logit")
    c.text(1350, 820, "Điểm nguồn sˢʳᶜ nằm ngoài vector G1", size=25, weight=700, fill=C["logic_border"])
    return c.save()


def fig35(a):
    c = V3Canvas(2000, 900, "fig_3_5_g1_mil_training_v3")
    c.title("Huấn luyện G1 từ nhãn cấp ảnh bằng multiple-instance learning")
    c.image("xray", a["IMG000415_xray"], B(40, 120, 150, 205))
    c.chip("label", B(32, 370, 170, 46), "Bag dương: y_i = 1", size=20)
    c.text(40, 485, "Không có nhãn ở mức ứng viên", size=24, weight=700, anchor="start", fill=C["logic_border"])
    for idx in range(6):
        x = 45 + (idx % 3) * 85
        y = 535 + (idx // 3) * 110
        c.image(None, a[f"candidate_{idx + 1:02d}"], B(x, y, 70, 70), rx=8, register=False)
        c.text(x + 35, y + 88, f"m{idx + 1}", size=19, weight=700)
    c.text(155, 785, "Bag ứng viên", size=25, weight=700)
    c.module("g1", B(400, 285, 240, 150), "G1", ["dùng chung", "bộ trọng số"], kind="trained", title_size=32, subtitle_size=23)
    c.module("logits", B(760, 285, 250, 150), "Logit ứng viên", "a₁ · a₂ · … · aₖ", kind="feature", title_size=28, subtitle_size=24)
    c.module("pool", B(1130, 270, 245, 180), "Smooth pooling", ["T = 0,20", "Logit cấp bag"], kind="logic", designed=True, title_size=29, subtitle_size=24)
    c.module("loss", B(1500, 285, 230, 150), "Mất mát cấp bag", "BCE", kind="trained", title_size=24, subtitle_size=27)
    c.route([(270, 650), (335, 650), (335, 360), (400, 360)], allow=("g1",))
    c.connect_lr("g1", "logits")
    c.connect_lr("logits", "pool")
    c.connect_lr("pool", "loss")
    c.route([(187, 393), (240, 393), (240, 95), (1615, 95), (1615, 285)], dashed=True, allow=("label", "loss"))
    c.module("neg", B(680, 640, 270, 115), "Ràng buộc bag âm", ["áp dụng khi y_i = 0", "L_neg"], kind="eval", dashed=True, title_size=23, subtitle_size=20)
    c.module("cons", B(1010, 640, 280, 115), "Nhất quán lật", ["logit ảnh gốc ↔ ảnh lật", "L_cons"], kind="eval", dashed=True, title_size=25, subtitle_size=19)
    c.module("objective", B(1390, 630, 500, 130), "Mục tiêu huấn luyện", "L = Lbag + 0,25Lneg + 0,10Lcons", kind="logic", designed=True, title_size=29, subtitle_size=23)
    c.route([(817, 650), (817, 560), (1510, 560), (1510, 630)], dashed=True, allow=("neg", "objective"))
    c.route([(1132, 650), (1132, 595), (1570, 595), (1570, 630)], dashed=True, allow=("cons", "objective"))
    c.route([(1615, 435), (1615, 630)], dashed=True, allow=("loss", "objective"))
    return c.save()


def fig36(a):
    c = V3Canvas(2000, 800, "fig_3_6_rank_fusion_final_selection_v3")
    c.title("Hợp nhất thứ hạng và lựa chọn mặt nạ cuối")
    c.module("label", B(40, 110, 225, 95), "Nhãn cấp ảnh", "thiết lập chính", kind="data", title_size=28, subtitle_size=22)
    c.module("normal", B(40, 260, 225, 82), "Bình thường", "Mặt nạ rỗng", kind="eval", title_size=27, subtitle_size=22)
    c.module("tumor", B(40, 410, 225, 82), "Khối u", "Xếp hạng ứng viên", kind="logic", designed=True, title_size=28, subtitle_size=21)
    c.route([(152, 205), (152, 260)], allow=("label", "normal"))
    c.route([(265, 157), (305, 157), (305, 451), (265, 451)], allow=("label", "tumor"))
    c.text(40, 555, "Khi không có nhãn cấp ảnh: dùng lớp dự đoán", size=20, anchor="start", fill=C["eval_border"])
    c.module("orig", B(370, 105, 245, 90), "Logit G1 — gốc", kind="feature", title_size=25)
    c.module("flip", B(370, 245, 245, 90), "Logit G1 — lật", kind="feature", title_size=25)
    c.module("avg", B(710, 145, 220, 110), "Lấy trung bình", "logit G1", kind="logic", designed=True, title_size=27, subtitle_size=23)
    c.module("grank", B(1020, 145, 215, 110), "Hạng phân vị", "G1", kind="logic", designed=True, title_size=28, subtitle_size=24)
    c.connect_lr("orig", "avg", mid_x=665)
    c.connect_lr("flip", "avg", mid_x=665)
    c.connect_lr("avg", "grank")
    c.module("source", B(470, 415, 235, 92), "Điểm nguồn", "sˢʳᶜ", kind="feature", title_size=27, subtitle_size=24)
    c.module("srank", B(800, 405, 220, 112), "Hạng phân vị", "nguồn", kind="logic", designed=True, title_size=28, subtitle_size=24)
    c.connect_lr("source", "srank")
    c.module("fusion", B(1320, 245, 245, 145), "Hợp nhất thứ hạng", ["0,5 r_G1 +", "0,5 r_nguồn"], kind="logic", designed=True, title_size=24, subtitle_size=21)
    c.connect_lr("grank", "fusion", mid_x=1280)
    c.connect_lr("srank", "fusion", mid_x=1280)
    c.module("choose", B(1660, 230, 245, 175), "Chọn hạng cao nhất", ["đồng hạng cố định", "hợp nhất → G1 → chỉ số"], kind="logic", designed=True, title_size=24, subtitle_size=20)
    c.connect_lr("fusion", "choose")
    c.module("gallery", B(480, 660, 240, 90), "Tập mặt nạ Mᵢ", kind="data", title_size=27)
    c.module("take", B(1080, 645, 230, 115), "Lấy mặt nạ", "tại j*", kind="output", title_size=30, subtitle_size=25)
    c.connect_lr("gallery", "take")
    c.route([(1782, 405), (1782, 575), (1195, 575), (1195, 645)], allow=("choose", "take"))
    for idx in range(3):
        x = 1425 + idx * 82
        c.image(None, a[f"top_candidate_{idx + 1:02d}"], B(x, 630, 70, 70), rx=8, register=False)
        c.text(x + 35, 720, f"#{idx + 1}", size=19, weight=700)
    c.text(1525, 755, "Top-3 ứng viên · minh họa", size=20, weight=700)
    c.route([(1310, 660), (1340, 660), (1340, 610), (1525, 610), (1525, 630)], dashed=True, arrow=False, allow=("take",))
    c.image("final", a["selected_mask_real"], B(1815, 535, 160, 205), stroke=C["output_border"], fit="contain")
    c.text(1895, 770, "Mặt nạ cuối", size=25, weight=700, fill=C["output_border"])
    c.route([(1310, 702), (1765, 702), (1765, 637), (1815, 637)], allow=("take", "final"))
    return c.save()


def fig4x(_):
    c = V3Canvas(2200, 760, "fig_4_x_prediction_lock_evaluation_v3")
    c.title("Giao thức khóa dự đoán trước khi truy cập mặt nạ chuẩn")
    stages = [
        ("input", 30, "Ảnh + nhãn", "cấp ảnh", "data", 27),
        ("pipe", 310, "Quy trình WSSS", None, "logic", 25),
        ("scores", 590, "Tập ứng viên", "+ điểm số", "feature", 25),
        ("selected", 870, "Mặt nạ", "được chọn", "output", 27),
        ("save", 1150, "Lưu kết quả", "+ SHA-256", "eval", 27),
        ("lock", 1430, "KHÓA DỰ ĐOÁN", None, "logic", 24),
    ]
    for node, x, title, subtitle, kind, ts in stages:
        c.module(node, B(x, 125, 210, 105), title, subtitle, kind=kind, designed=kind == "logic", title_size=ts, subtitle_size=24)
    for (src, *_), (dst, *__) in zip(stages, stages[1:]):
        c.connect_lr(src, dst)
    c.module("frozen_pred", B(1710, 105, 240, 115), "Dự đoán đã khóa", "bất biến", kind="output", title_size=26, subtitle_size=22)
    c.connect_lr("lock", "frozen_pred")
    c.module("gt", B(1710, 300, 285, 145), "Mặt nạ chuẩn", "chỉ truy cập sau khi khóa", kind="eval", dashed=True, title_size=29, subtitle_size=22)
    c.module("metrics", B(850, 540, 340, 125), "Đánh giá", "Dice · IoU · Precision · Recall", kind="eval", dashed=True, title_size=30, subtitle_size=21)
    c.module("post", B(1430, 540, 390, 125), "Phân tích hậu nghiệm", "Oracle Dice · Selector Regret", kind="eval", dashed=True, title_size=28, subtitle_size=22)
    c.route([(1830, 220), (1680, 220), (1680, 480), (1020, 480), (1020, 540)], allow=("frozen_pred", "metrics"))
    c.route([(1830, 220), (1660, 220), (1660, 490), (1625, 490), (1625, 540)], allow=("frozen_pred", "post"))
    c.route([(1852, 445), (1852, 510), (1100, 510), (1100, 540)], dashed=True, allow=("gt", "metrics"))
    c.route([(1852, 445), (1852, 520), (1700, 520), (1700, 540)], dashed=True, allow=("gt", "post"))
    c.text(40, 720, "Mặt nạ chuẩn không tham gia định vị, tạo prompt, sinh ứng viên hoặc lựa chọn.", size=27, weight=700, anchor="start", fill=C["eval_border"])
    return c.save()


def write_support(svg_paths, audits):
    for svg_path in svg_paths:
        head = svg_path.read_text(encoding="utf-8")[:300]
        match = re.search(r'<svg[^>]+width="(\d+)"[^>]+height="(\d+)"', head)
        width, height = (int(v) for v in match.groups())
        width_mm = 160.0
        height_mm = height / width * width_mm
        html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {width_mm:.2f}mm {height_mm:.2f}mm; margin: 0; }}
html,body {{ margin:0; width:{width_mm:.2f}mm; height:{height_mm:.2f}mm; overflow:hidden; background:white; }}
img {{ display:block; width:100%; height:100%; }}
</style></head><body><img src="{svg_path.resolve().as_uri()}"></body></html>'''
        (base.HTML_DIR / f"{svg_path.stem}.html").write_text(html, encoding="utf-8")
    (OUT / "layout_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Rich Gallery G1 — figure set v3\n\n"
        "Bố cục thưa hơn, gutter dài hơn và có kiểm tra box–box, text–box, arrow–box. "
        "Ảnh minh họa tiếp tục dùng output thật IMG000415 đã tải từ Prism.\n",
        encoding="utf-8",
    )


def main():
    assets = base.prepare_assets()
    heatmap_root = OUT / "source_images"
    assets.update({
        "heatmap_biomedclip": heatmap_root / "p21_04_X13.png",
        "heatmap_layercam448": heatmap_root / "p21_05_X14.png",
        "heatmap_layercam320": heatmap_root / "p21_06_X15.png",
    })
    results = [fig31(assets), fig32(assets), fig33(assets), fig34(assets), fig35(assets), fig36(assets), fig4x(assets)]
    paths = [r[0] for r in results]
    audits = [r[1] for r in results]
    write_support(paths, audits)
    print(json.dumps({"svg": [str(p) for p in paths], "audits": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
