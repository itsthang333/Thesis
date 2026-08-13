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

    def title(self, value: str) -> None:
        """Publication figures use the external LaTeX caption as their title."""
        return None

    def section(self, value: str, y: float) -> None:
        self.text(30, y, value, size=24, weight=700, anchor="start", fill=C["text"])
        self.connectors.append(
            f'<line x1="30" y1="{y + 24}" x2="{self.width - 30}" y2="{y + 24}" '
            'stroke="#D5D5D5" stroke-width="1.8"/>'
        )

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
    c = V3Canvas(2500, 1300, "fig_3_1_overview_offline_online_v3")
    c.rect(B(25, 25, 2450, 535), "#FBFCFD", "#C9CED3", rx=16, sw=2.0, layer="background")
    c.rect(B(25, 700, 2450, 565), "#FBFCFD", "#C9CED3", rx=16, sw=2.0, layer="background")
    c.text(55, 65, "(a) OFFLINE — Chuẩn bị và huấn luyện", size=27, weight=700, anchor="start", fill=C["text"])
    c.text(55, 740, "(b) ONLINE — Suy luận và lựa chọn", size=27, weight=700, anchor="start", fill=C["text"])

    c.image("off_xray", a["IMG000415_xray"], B(60, 130, 145, 195))
    c.text(132, 355, "Tập ảnh X-quang", size=23, weight=700)
    c.chip("off_label", B(62, 395, 140, 46), "Nhãn cấp ảnh", size=20)
    c.module("loc_train", B(310, 145, 300, 160), "Huấn luyện bộ định vị", ["DenseNet-121: 10 lớp", "DenseNet-121: nhị phân"], kind="trained", title_size=25, subtitle_size=21)
    c.module("theta_loc", B(710, 165, 210, 120), "Checkpoint", "θ₁₀ · θ₂", kind="trained", title_size=27, subtitle_size=24)
    c.connect_lr("off_xray", "loc_train", mid_x=255)
    c.route([(132, 441), (255, 441), (255, 245), (310, 245)], dashed=True, allow=("off_label", "loc_train"))
    c.connect_lr("loc_train", "theta_loc")

    c.module("gallery_train", B(1030, 130, 330, 190), "Sinh Candidate Gallery", ["định vị đa nguồn", "phân vị → prompt → SAM"], kind="logic", designed=True, title_size=27, subtitle_size=22)
    c.module("gallery", B(1470, 160, 235, 130), "Tập ứng viên", "huấn luyện", kind="data", title_size=27, subtitle_size=22)
    c.module("repr", B(1810, 145, 270, 160), "Biểu diễn ứng viên", "RAD-DINO", kind="feature", title_size=26, subtitle_size=23)
    c.module("g1", B(2180, 130, 240, 190), "Huấn luyện G1", ["MIL cấp bag", "→ θ_G1"], kind="trained", title_size=29, subtitle_size=23)
    c.connect_lr("theta_loc", "gallery_train")
    c.route([(205, 227), (255, 227), (255, 350), (980, 350), (980, 225), (1030, 225)], allow=("off_xray", "gallery_train"))
    c.connect_lr("gallery_train", "gallery")
    c.connect_lr("gallery", "repr")
    c.connect_lr("repr", "g1")
    c.route([(132, 441), (132, 500), (2300, 500), (2300, 320)], dashed=True, allow=("off_label", "g1"))
    c.module("bio", B(1040, 385, 185, 80), "BiomedCLIP", kind="frozen", frozen=True, title_size=22)
    c.module("sam", B(1260, 385, 180, 80), "SAM ViT-B", kind="frozen", frozen=True, title_size=22)
    c.module("rad", B(1815, 385, 190, 80), "RAD-DINO", kind="frozen", frozen=True, title_size=22)
    c.route([(1132, 385), (1132, 320)], dashed=True, allow=("bio", "gallery_train"))
    c.route([(1350, 385), (1350, 350), (1290, 350), (1290, 320)], dashed=True, allow=("sam", "gallery_train"))
    c.route([(1910, 385), (1910, 305)], dashed=True, allow=("rad", "repr"))

    c.module("bundle", B(1920, 585, 480, 105), "Gói triển khai Θ*", ["θ₁₀ · θ₂ · θ_G1", "BiomedCLIP · SAM · RAD-DINO"], kind="output", title_size=27, subtitle_size=20)
    c.route([(815, 285), (815, 535), (2020, 535), (2020, 585)], dashed=True, allow=("theta_loc", "bundle"))
    c.route([(2300, 320), (2300, 585)], allow=("g1", "bundle"))

    c.image("on_xray", a["IMG000415_xray"], B(60, 815, 145, 195))
    c.text(132, 1040, "Ảnh X-quang mới", size=23, weight=700)
    c.chip("on_label", B(62, 1080, 140, 46), "Nhãn cấp ảnh", size=20)
    c.module("gate", B(300, 850, 230, 120), "Cổng nhãn", "đã biết", kind="data", title_size=29, subtitle_size=22)
    c.module("normal", B(300, 1090, 230, 105), "Bình thường", "mặt nạ rỗng", kind="eval", title_size=27, subtitle_size=22)
    c.connect_lr("on_xray", "gate")
    c.route([(132, 1126), (250, 1126), (250, 930), (300, 930)], dashed=True, allow=("on_label", "gate"))
    c.route([(415, 970), (415, 1090)], allow=("gate", "normal"))

    c.rect(B(610, 790, 1390, 265), "#FFFDFC", "#B7A078", dashed=True, rx=14, sw=2.0, layer="background")
    c.text(635, 820, "RICH GALLERY G1 — SUY LUẬN KHỐI U", size=21, weight=700, anchor="start", fill=C["text"])
    c.module("loc", B(650, 865, 260, 135), "Định vị đa nguồn", "θ₁₀ · θ₂ · BiomedCLIP", kind="logic", designed=True, title_size=26, subtitle_size=20)
    c.module("generate", B(1010, 850, 280, 165), "Sinh ứng viên", ["phân vị → prompt", "→ SAM ViT-B"], kind="logic", designed=True, title_size=28, subtitle_size=22)
    c.module("online_gallery", B(1390, 870, 225, 125), "Candidate Gallery", "M_i", kind="data", title_size=22, subtitle_size=23)
    c.module("rank", B(1715, 845, 250, 175), "Chấm điểm & xếp hạng", ["RAD-DINO → G1", "hợp nhất thứ hạng"], kind="logic", designed=True, title_size=22, subtitle_size=21)
    c.connect_lr("gate", "loc")
    c.connect_lr("loc", "generate")
    c.connect_lr("generate", "online_gallery")
    c.connect_lr("online_gallery", "rank")
    c.module("select", B(2090, 870, 225, 125), "Chọn ứng viên", "hạng cao nhất", kind="logic", designed=True, title_size=25, subtitle_size=21)
    c.connect_lr("rank", "select")
    c.image("final", a["selected_mask_real"], B(2140, 1085, 150, 130), stroke=C["output_border"], fit="contain")
    c.text(2215, 1240, "Mặt nạ nhị phân cuối", size=24, weight=700, fill=C["output_border"])
    c.route([(2202, 995), (2202, 1085)], allow=("select", "final"))
    c.route([(2160, 690), (2160, 755), (1835, 755), (1835, 845)], dashed=True, allow=("bundle", "rank"))
    c.text(1970, 735, "Triển khai Θ*", size=21, weight=700, fill=C["muted"])
    return c.save()


def fig32(a):
    c = V3Canvas(2000, 950, "fig_3_2_multisource_localization_v3")
    c.title("Trích xuất bằng chứng định vị đa nguồn")
    c.image("input", a["IMG000415_xray"], B(35, 330, 170, 270))
    c.text(120, 635, "Ảnh X-quang", size=27, weight=700)
    columns = [
        ("b320", 275, "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], "LayerCAM", ["Dense Block 2/3/4", "trọng số 0,2 / 0,3 / 0,5"], "heatmap_layercam320", "LayerCAM-320"),
        ("b448", 855, "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], "LayerCAM", ["Dense Block 2/3/4", "trọng số 0,2 / 0,3 / 0,5"], "heatmap_layercam448", "LayerCAM-448"),
        ("bbio", 1435, "BiomedCLIP", ["Toàn ảnh + vùng cục bộ"], "Saliency map", None, "heatmap_biomedclip", "BiomedCLIP saliency"),
    ]
    centers = []
    for node, x, model, model_sub, op, op_sub, image_key, label in columns:
        centers.append(x + 180)
        c.module(node, B(x, 120, 360, 110), model, model_sub, kind="frozen" if node == "bbio" else "trained", frozen=node == "bbio", title_size=30, subtitle_size=24)
        c.module(node + "_op", B(x + 35, 300, 290, 120), op, op_sub, kind="logic", designed=True, title_size=30, subtitle_size=21)
        c.image(node + "_img", a[image_key], B(x + 40, 520, 280, 280), stroke=C["logic_border"], fit="slice")
        c.text(x + 180, 840, label, size=28, weight=700)
        c.route([(x + 180, 230), (x + 180, 300)], allow=(node, node + "_op"))
        c.route([(x + 180, 420), (x + 180, 520)], allow=(node + "_op", node + "_img"))
    c.route([(205, 462), (245, 462), (245, 82), (centers[-1], 82)], arrow=False, allow=("input",))
    for center, (node, *_rest) in zip(centers, columns):
        c.route([(center, 82), (center, 120)], allow=(node,))
    c.route([(275, 895), (1795, 895)], arrow=False, sw=2.2, color="#AFAFAF")
    c.text(1035, 925, "Giữ riêng từng nguồn — không trung bình bản đồ", size=27, weight=700, fill=C["text"])
    return c.save()


def fig33(a):
    c = V3Canvas(2000, 1240, "fig_3_3_candidate_gallery_generation_v3")
    c.title("Sinh tập mặt nạ ứng viên từ bằng chứng định vị")
    for idx, key in enumerate(("heatmap_layercam320", "heatmap_layercam448", "heatmap_biomedclip")):
        c.image(None, a[key], B(30, 105 + idx * 135, 155, 115), fit="slice", register=False)
    c.text(107, 535, "Ba bản đồ định vị", size=26, weight=700)
    c.text(107, 570, "L320 · L448 · Bio", size=22, fill=C["muted"])
    c.module("threshold", B(255, 225, 205, 125), "Ngưỡng phân vị", ["P85 · P90 · P95", "riêng từng nguồn"], kind="logic", designed=True, title_size=25, subtitle_size=21)
    c.module("cc", B(535, 215, 220, 145), "Thành phần liên thông", ["giữ tối đa 3 vùng"], kind="logic", designed=True, title_size=18, subtitle_size=22)
    c.module("prompt", B(835, 190, 220, 190), "Tạo prompt", ["Điểm · Hộp · Điểm + hộp", "điểm âm từ", "vành ngoài"], kind="logic", designed=True, title_size=30, subtitle_size=18)
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
    c.text(1875, 455, "Mặt nạ ứng viên", size=24, weight=700)
    c.route([(1730, 287), (1780, 287)], allow=("sam",))

    c.module("grid", B(300, 690, 220, 100), "Quy về lưới chung", "L448: 448 → 320", kind="logic", designed=True, title_size=22, subtitle_size=21)
    c.module("merge", B(620, 690, 220, 100), "Gộp ba nguồn", kind="logic", designed=True, title_size=27)
    c.module("dedupe", B(940, 675, 250, 130), "Loại trùng", ["hoàn toàn", "theo từng điểm ảnh"], kind="logic", designed=True, title_size=28, subtitle_size=22)
    c.module("source_score", B(1015, 885, 310, 150), "Điểm nguồn", ["D — Mật độ · M — Khối lượng", "R — Hạng SAM", "0,60D + 0,25M + 0,15R"], kind="feature", title_size=29, subtitle_size=19)
    c.module("gallery", B(1580, 670, 330, 150), "Candidate Gallery (M_i)", ["K_i ứng viên duy nhất", "giữ metadata và điểm nguồn"], kind="data", title_size=24, subtitle_size=20)
    c.connect_lr("grid", "merge")
    c.connect_lr("merge", "dedupe")
    c.route([(1190, 740), (1385, 740), (1385, 745), (1580, 745)], allow=("dedupe", "gallery"))
    c.route([(1875, 475), (1875, 620), (410, 620), (410, 690)], allow=("grid",))
    c.route([(1875, 475), (1960, 475), (1960, 1065), (1170, 1065), (1170, 1035)], dashed=True, allow=("source_score",))
    c.route([(1325, 960), (1480, 960), (1480, 780), (1580, 780)], dashed=True, allow=("source_score", "gallery"))
    c.text(300, 1140, "3 phân vị × ≤3 vùng × 3 kiểu prompt × 3 mặt nạ SAM", size=25, weight=700, anchor="start")
    c.text(300, 1190, "≤ 81 ứng viên / nguồn", size=28, weight=700, anchor="start", fill=C["text"])
    c.text(1290, 1190, "≤ 243 ứng viên / ảnh trước lọc", size=28, weight=700, anchor="start", fill=C["text"])
    return c.save()


def fig34(a):
    c = V3Canvas(2000, 880, "fig_3_4_candidate_representation_g1_v3")
    c.title("Biểu diễn ứng viên và mạng chấm điểm G1")
    c.image("xray", a["IMG000415_xray"], B(40, 120, 160, 220))
    c.text(120, 375, "Ảnh X-quang (Iᵢ)", size=25, weight=700)
    c.image("mask", a["candidate_01"], B(55, 560, 135, 135))
    c.text(122, 735, "Ứng viên (mᵢⱼ)", size=25, weight=700)
    c.module("rad", B(285, 105, 300, 115), "RAD-DINO", ["Đặc trưng lớp 4 · 8 · 12"], kind="frozen", frozen=True, title_size=30, subtitle_size=22)
    c.module("proj", B(760, 105, 240, 105), "Chiếu cố định", "768 → 128", kind="feature", title_size=28, subtitle_size=25)
    c.connect_lr("xray", "rad", mid_x=240)
    c.connect_lr("rad", "proj")
    c.module("pool", B(400, 500, 250, 175), "Pooling theo mặt nạ", ["Trong vùng", "Ngữ cảnh", "Tương phản"], kind="logic", designed=True, title_size=23, subtitle_size=24)
    c.connect_lr("mask", "pool", mid_x=300)
    c.route([(1040, 210), (1040, 425), (525, 425), (525, 500)], allow=("proj", "pool"))
    c.module("radfeat", B(760, 490, 280, 150), "Đặc trưng RAD-DINO", ["3 lớp × 3 thống kê × 128", "1152 chiều"], kind="feature", title_size=25, subtitle_size=21)
    c.connect_lr("pool", "radfeat")
    c.module("aux", B(740, 690, 380, 155), "4 đặc trưng bổ sung", ["SAM score · log(tỷ lệ diện tích)", "khối lượng M · đáp ứng trung bình μA"], kind="feature", title_size=25, subtitle_size=20)
    c.module("concat", B(1160, 500, 245, 140), "Ghép đặc trưng", "xᵢⱼ ∈ ℝ¹¹⁵⁶", kind="feature", title_size=29, subtitle_size=25)
    c.connect_lr("radfeat", "concat")
    c.route([(1040, 767), (1100, 767), (1100, 595), (1160, 595)], allow=("aux", "concat"))
    c.module("g1", B(1510, 480, 235, 180), "G1", ["LayerNorm → 256", "→ 128 → 1", "GELU · Dropout 0,10"], kind="trained", title_size=35, subtitle_size=23)
    c.module("logit", B(1830, 520, 155, 110), "Logit", "ứng viên aᵢⱼ", kind="feature", title_size=28, subtitle_size=22)
    c.connect_lr("concat", "g1")
    c.connect_lr("g1", "logit")
    c.text(1450, 820, "Điểm nguồn (s_src) không thuộc vector đầu vào G1.", size=23, fill=C["muted"], italic=True)
    return c.save()


def fig35(a):
    c = V3Canvas(2200, 940, "fig_3_5_g1_mil_training_v3")
    c.title("Huấn luyện G1 từ nhãn cấp ảnh bằng multiple-instance learning")
    c.image("xray", a["IMG000415_xray"], B(40, 120, 150, 205))
    c.module("label", B(30, 365, 180, 78), "Bag dương", "y_i = 1", kind="data", title_size=23, subtitle_size=20)
    c.text(40, 500, "Không có nhãn ở mức ứng viên", size=24, weight=700, anchor="start", fill=C["text"])
    for idx in range(6):
        x = 45 + (idx % 3) * 85
        y = 535 + (idx // 3) * 110
        c.image(None, a[f"candidate_{idx + 1:02d}"], B(x, y, 70, 70), rx=8, register=False)
        c.text(x + 35, y + 88, f"m{idx + 1}", size=19, weight=700)
    c.text(155, 785, "Bag ứng viên", size=25, weight=700)
    c.module("repr", B(350, 285, 260, 150), "Biểu diễn ứng viên", ["x_ij ∈ R¹¹⁵⁶", "theo Hình 3.4"], kind="feature", title_size=25, subtitle_size=22)
    c.module("g1", B(720, 285, 220, 150), "G1", ["dùng chung", "trọng số"], kind="trained", title_size=32, subtitle_size=23)
    c.module("logits", B(1050, 285, 250, 150), "Logit ứng viên", "a₁ · a₂ · … · aₖ", kind="feature", title_size=28, subtitle_size=24)
    c.module("pool", B(1410, 270, 245, 180), "Smooth pooling", ["T = 0,20", "Logit cấp bag"], kind="logic", designed=True, title_size=29, subtitle_size=24)
    c.module("loss", B(1775, 285, 230, 150), "Mất mát cấp bag", "BCE", kind="trained", title_size=24, subtitle_size=27)
    c.route([(270, 650), (310, 650), (310, 360), (350, 360)], allow=("repr",))
    c.connect_lr("repr", "g1")
    c.connect_lr("g1", "logits")
    c.connect_lr("logits", "pool")
    c.connect_lr("pool", "loss")
    c.route([(210, 404), (250, 404), (250, 95), (1890, 95), (1890, 285)], dashed=True, allow=("label", "loss"))
    c.module("neg", B(930, 655, 270, 115), "Ràng buộc bag âm", ["khi y_i = 0", "L_neg"], kind="eval", dashed=True, title_size=23, subtitle_size=20)
    c.module("cons", B(1300, 655, 280, 115), "Nhất quán lật", ["gốc ↔ lật", "L_cons"], kind="eval", dashed=True, title_size=25, subtitle_size=20)
    c.module("objective", B(1690, 635, 480, 145), "Mục tiêu huấn luyện", "L_total = L_bag + 0,25 L_neg + 0,10 L_cons", kind="logic", designed=True, title_size=29, subtitle_size=21)
    c.route([(1175, 435), (1175, 595), (1065, 595), (1065, 655)], dashed=True, allow=("logits", "neg"))
    c.route([(1230, 435), (1230, 610), (1440, 610), (1440, 655)], dashed=True, allow=("logits", "cons"))
    c.route([(1200, 712), (1260, 712), (1260, 620), (1635, 620), (1635, 680), (1690, 680)], dashed=True, allow=("neg", "objective"))
    c.route([(1580, 712), (1650, 712), (1650, 730), (1690, 730)], dashed=True, allow=("cons", "objective"))
    c.route([(1890, 435), (1890, 635)], dashed=True, allow=("loss", "objective"))
    return c.save()


def fig36(a):
    c = V3Canvas(2100, 900, "fig_3_6_rank_fusion_final_selection_v3")
    c.title("Hợp nhất thứ hạng và lựa chọn mặt nạ cuối")
    c.module("label", B(40, 110, 270, 105), "Nhãn cấp ảnh đã biết", "thiết lập chính", kind="data", title_size=23, subtitle_size=22)
    c.module("normal", B(40, 260, 225, 82), "Bình thường", "Mặt nạ rỗng", kind="eval", title_size=27, subtitle_size=22)
    c.module("tumor", B(40, 410, 225, 82), "Khối u", "Xếp hạng ứng viên", kind="logic", designed=True, title_size=28, subtitle_size=21)
    c.route([(152, 205), (152, 260)], allow=("label", "normal"))
    c.route([(310, 157), (330, 157), (330, 451), (265, 451)], allow=("label", "tumor"))
    c.route([(265, 451), (330, 451), (330, 370), (340, 370)], allow=("tumor",))
    c.text(350, 370, "XẾP HẠNG", size=21, weight=700, anchor="start", fill=C["text"])
    c.route([(445, 370), (340, 370), (340, 150), (370, 150)], allow=("orig",))
    c.route([(445, 370), (340, 370), (340, 461), (470, 461)], allow=("source",))
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
    c.module("choose", B(1660, 230, 260, 175), "Chọn ứng viên cuối", ["đồng hạng:", "hợp nhất → G1 → chỉ số"], kind="logic", designed=True, title_size=24, subtitle_size=20)
    c.connect_lr("fusion", "choose")
    c.module("gallery", B(480, 700, 240, 90), "Tập mặt nạ Mᵢ", kind="data", title_size=27)
    c.module("take", B(1030, 685, 230, 115), "Lấy mặt nạ", "tại j*", kind="output", title_size=30, subtitle_size=25)
    c.connect_lr("gallery", "take")
    c.route([(1790, 405), (1790, 600), (1145, 600), (1145, 685)], allow=("choose", "take"))
    for idx in range(3):
        x = 1390 + idx * 82
        c.image(None, a[f"top_candidate_{idx + 1:02d}"], B(x, 720, 70, 70), rx=8, register=False)
        c.text(x + 35, 810, f"#{idx + 1}", size=19, weight=700)
    c.text(1490, 845, "Top-3 ứng viên · minh họa", size=20, weight=700)
    c.route([(720, 745), (800, 745), (800, 840), (1340, 840), (1340, 755), (1390, 755)], dashed=True, arrow=False, allow=("gallery",))
    c.image("final", a["selected_mask_real"], B(1870, 650, 160, 205), stroke=C["output_border"], fit="contain")
    c.text(1950, 880, "Mặt nạ cuối", size=25, weight=700, fill=C["output_border"])
    c.route([(1260, 742), (1810, 742), (1810, 752), (1870, 752)], allow=("take", "final"))
    return c.save()


def fig4x(_):
    c = V3Canvas(2200, 760, "fig_4_x_prediction_lock_evaluation_v3")
    c.title("Giao thức khóa dự đoán trước khi truy cập mặt nạ chuẩn")
    stages = [
        ("input", 30, "Ảnh X-quang", "+ nhãn cấp ảnh", "data", 25),
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
    c.module("frozen_pred", B(1710, 105, 300, 125), "Kết quả đã khóa", "dự đoán + metadata", kind="output", title_size=27, subtitle_size=21)
    c.connect_lr("lock", "frozen_pred")
    c.module("gt", B(1715, 320, 285, 145), "Mặt nạ chuẩn", "chỉ truy cập sau khi khóa", kind="eval", dashed=True, title_size=29, subtitle_size=22)
    c.module("metrics", B(850, 540, 340, 125), "Đánh giá", "Dice · IoU · Precision · Recall", kind="eval", dashed=True, title_size=30, subtitle_size=21)
    c.module("post", B(1430, 540, 390, 125), "Phân tích hậu nghiệm", "Oracle Dice · Selector Regret", kind="eval", dashed=True, title_size=28, subtitle_size=22)
    c.route([(1860, 230), (1660, 230), (1660, 495), (1020, 495), (1020, 540)], allow=("frozen_pred", "metrics"))
    c.route([(2010, 167), (2060, 167), (2060, 505), (1625, 505), (1625, 540)], allow=("frozen_pred", "post"))
    c.route([(1857, 465), (1857, 520), (1100, 520), (1100, 540)], dashed=True, allow=("gt", "metrics"))
    c.route([(1857, 465), (1857, 530), (1700, 530), (1700, 540)], dashed=True, allow=("gt", "post"))
    c.text(40, 720, "Mặt nạ chuẩn không tham gia bất kỳ bước huấn luyện, sinh ứng viên hay lựa chọn WSSS nào.", size=27, weight=700, anchor="start", fill=C["text"])
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
