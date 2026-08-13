from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets" / "rich_gallery_figures_v2"
SRC = OUT / "source_images"
ASSETS = OUT / "assets"
SVG_DIR = OUT / "svg"
HTML_DIR = OUT / "html"
RENDERED = OUT / "rendered"

XRAY = ROOT / "BTXRD" / "BTXRD" / "images" / "IMG000415.jpeg"
DIAG = (
    ROOT
    / "artifacts"
    / "section_3_4_img000415"
    / "real_rerun"
    / "biomedclip_layercam320_single"
    / "candidate_diagnostics"
    / "IMG000415.npz"
)
MERGED = (
    ROOT
    / "artifacts"
    / "section_3_4_img000415"
    / "report_real_selected"
    / "IMG000415_merged_candidate_gallery_real.npz"
)

COLORS = {
    "data_fill": "#F1EEE8",
    "data_border": "#8A8278",
    "trained_fill": "#E5D7E6",
    "trained_border": "#71556F",
    "frozen_fill": "#E0E4CF",
    "frozen_border": "#68704E",
    "logic_fill": "#F1DFC0",
    "logic_border": "#8A6A38",
    "feature_fill": "#E9D7D2",
    "feature_border": "#8A5F56",
    "output_fill": "#D3E3DE",
    "output_border": "#52766C",
    "eval_fill": "#ECEDEF",
    "eval_border": "#7D8287",
    "text": "#252525",
    "muted": "#66625E",
    "arrow": "#454545",
    "white": "#FFFFFF",
}


def ensure_dirs() -> None:
    for path in (OUT, SRC, ASSETS, SVG_DIR, HTML_DIR, RENDERED):
        path.mkdir(parents=True, exist_ok=True)


def save_png(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", optimize=True, dpi=(450, 450))


def fitted_xray(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im, cutoff=0.4)
    im = ImageEnhance.Contrast(im).enhance(1.06)
    return ImageOps.contain(im, size, Image.Resampling.LANCZOS).convert("RGB")


def mask_image(mask: np.ndarray, size: int = 520) -> Image.Image:
    mask_im = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, "L")
    mask_im = mask_im.resize((size, size), Image.Resampling.NEAREST)
    return Image.composite(
        Image.new("RGB", (size, size), "#FAFAF7"),
        Image.new("RGB", (size, size), "#151515"),
        mask_im,
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.astype(bool), b.astype(bool)
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def diverse_indices(masks: np.ndarray, scores: np.ndarray, count: int) -> list[int]:
    chosen: list[int] = []
    for raw_idx in np.argsort(scores)[::-1]:
        idx = int(raw_idx)
        area = float(masks[idx].mean())
        if not 0.001 < area < 0.55:
            continue
        if all(iou(masks[idx], masks[j]) < 0.80 for j in chosen):
            chosen.append(idx)
        if len(chosen) == count:
            break
    return chosen


def crop_prism_p90(path: Path, prompt_fusion: bool = False) -> Image.Image:
    """Crop the real middle (P90) output panel from a Prism slide asset."""
    im = Image.open(path).convert("RGB")
    top = 146 if prompt_fusion else 130
    bottom = 720 if prompt_fusion else 698
    return im.crop((640, top, 1280, bottom))


def prepare_assets() -> dict[str, Path]:
    ensure_dirs()
    required = [
        "slide21_layercam320.png",
        "slide21_layercam448.png",
        "slide21_biomedclip.png",
        "slide21_prompt_fusion.png",
        "IMG000415_selected_output_final.png",
    ]
    missing = [name for name in required if not (SRC / name).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu ảnh Prism: {missing}")

    save_png(fitted_xray(XRAY, (900, 1200)), ASSETS / "IMG000415_xray.png")
    for name in ("layercam320", "layercam448", "biomedclip"):
        crop = crop_prism_p90(SRC / f"slide21_{name}.png")
        save_png(crop, ASSETS / f"IMG000415_{name}_p90_real.png")
    prompt = crop_prism_p90(SRC / "slide21_prompt_fusion.png", prompt_fusion=True)
    save_png(prompt, ASSETS / "IMG000415_prompt_fusion_p90_real.png")

    selected_pair = Image.open(SRC / "IMG000415_selected_output_final.png").convert("RGB")
    split = int(selected_pair.width * 0.515)
    save_png(selected_pair.crop((split, 0, selected_pair.width, selected_pair.height)), ASSETS / "IMG000415_selected_overlay_real.png")

    with np.load(MERGED, allow_pickle=True) as z:
        merged_masks = z["sam_masks"].astype(np.uint8)
        source_scores = z["source_local_scores"].astype(np.float32)
        source_ids = z["source_ids"].astype(str)
    gallery_idx = diverse_indices(merged_masks, source_scores, 8)
    for rank, idx in enumerate(gallery_idx, 1):
        save_png(mask_image(merged_masks[idx]), ASSETS / f"candidate_{rank:02d}.png")

    with np.load(DIAG, allow_pickle=True) as z:
        final_mask = z["final_mask"].astype(np.uint8)
        local_masks = z["sam_masks"].astype(np.uint8)
        selection_scores = z["selection_scores"].astype(np.float32)
    top_idx = diverse_indices(local_masks, selection_scores, 3)
    for rank, idx in enumerate(top_idx, 1):
        save_png(mask_image(local_masks[idx]), ASSETS / f"top_candidate_{rank:02d}.png")
    save_png(mask_image(final_mask, 650), ASSETS / "selected_mask_real.png")

    manifest = {
        "sample": "IMG000415",
        "prism_project": "e762d1d0-23b1-41ab-9941-7a9a8bc4d5bc",
        "real_prism_outputs": required,
        "figure_2_policy": "Uses the real P90 per-source Prism outputs; no reconstructed heatmap is shown.",
        "figure_3_policy": "Uses the real P90 all-source prompt-fusion Prism output.",
        "candidate_masks": {
            "source": str(MERGED.relative_to(ROOT)),
            "indices": gallery_idx,
        },
        "top_candidates": {"source": str(DIAG.relative_to(ROOT)), "indices": top_idx},
        "final_mask": str(DIAG.relative_to(ROOT)),
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {p.stem: p for p in ASSETS.glob("*.png")}


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.y + self.h / 2

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.y + self.h / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y


class Canvas:
    def __init__(self, width: int, height: int, name: str):
        self.width, self.height, self.name = width, height, name
        self.background: list[str] = []
        self.connectors: list[str] = []
        self.foreground: list[str] = []
        self.defs: list[str] = [
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#454545"/></marker>'
        ]
        self.nodes: dict[str, Box] = {}
        self.clip_count = 0

    def register(self, node_id: str | None, box: Box) -> None:
        if not node_id:
            return
        if node_id in self.nodes:
            raise ValueError(f"Trùng node id: {node_id}")
        self.nodes[node_id] = box

    def rect(self, b: Box, fill: str, stroke: str, *, dashed: bool = False, rx: float = 18, sw: float = 2.8, layer: str = "foreground") -> None:
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        item = f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
        getattr(self, layer).append(item)

    def text(self, x: float, y: float, value: str, *, size: float = 25, weight: int = 400, anchor: str = "middle", fill: str = COLORS["text"], italic: bool = False) -> None:
        style = ' font-style="italic"' if italic else ""
        self.foreground.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{fill}"{style}>{escape(value)}</text>'
        )

    def lines(self, x: float, y: float, values: list[str], *, size: float = 23, weight: int = 400, line_height: float | None = None, fill: str = COLORS["muted"], anchor: str = "middle") -> None:
        line_height = line_height or size * 1.25
        start = y - (len(values) - 1) * line_height / 2
        for idx, value in enumerate(values):
            self.text(x, start + idx * line_height, value, size=size, weight=weight, fill=fill, anchor=anchor)

    def module(self, node_id: str, b: Box, title: str, subtitle: str | list[str] | None = None, *, kind: str = "data", designed: bool = False, frozen: bool = False, dashed: bool = False, title_size: float = 27, subtitle_size: float = 22) -> None:
        self.register(node_id, b)
        fill, border = COLORS[f"{kind}_fill"], COLORS[f"{kind}_border"]
        self.rect(b, fill, border, dashed=dashed)
        if designed:
            self.foreground.append(f'<rect x="{b.x}" y="{b.y}" width="7" height="{b.h}" rx="3.5" fill="{border}"/>')
        if frozen:
            badge = Box(b.x + b.w - 86, b.y + 7, 76, 25)
            self.rect(badge, COLORS["white"], border, rx=9, sw=1.8)
            self.text(badge.x + badge.w / 2, badge.y + badge.h / 2, "Đóng băng", size=14, weight=700, fill=border)
        if subtitle:
            values = [subtitle] if isinstance(subtitle, str) else subtitle
            title_y = b.y + b.h * (0.53 if frozen else 0.35)
            subtitle_y = b.y + b.h * (0.79 if frozen else (0.70 if len(values) == 1 else 0.72))
            self.text(b.x + b.w / 2, title_y, title, size=title_size, weight=700)
            self.lines(b.x + b.w / 2, subtitle_y, list(values), size=subtitle_size, line_height=subtitle_size * 1.16)
        else:
            self.text(b.x + b.w / 2, b.y + b.h / 2, title, size=title_size, weight=700)

    def chip(self, node_id: str, b: Box, value: str, *, kind: str = "data", dashed: bool = False, size: float = 19) -> None:
        self.register(node_id, b)
        self.rect(b, COLORS[f"{kind}_fill"], COLORS[f"{kind}_border"], dashed=dashed, rx=13, sw=2.1)
        self.text(b.x + b.w / 2, b.y + b.h / 2, value, size=size, weight=700)

    def image(self, node_id: str | None, path: Path, b: Box, *, stroke: str = COLORS["data_border"], fit: str = "meet", rx: float = 14, register: bool = True) -> None:
        if register:
            self.register(node_id, b)
        self.clip_count += 1
        clip_id = f"clip{self.clip_count}"
        self.defs.append(f'<clipPath id="{clip_id}"><rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="{rx}"/></clipPath>')
        self.foreground.append(
            f'<image href="{data_uri(path)}" x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" preserveAspectRatio="xMidYMid {fit}" clip-path="url(#{clip_id})"/>'
        )
        self.foreground.append(f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="{rx}" fill="none" stroke="{stroke}" stroke-width="2.6"/>')

    def route(self, points: list[tuple[float, float]], *, dashed: bool = False, arrow: bool = True, sw: float = 3.2, color: str = COLORS["arrow"]) -> None:
        pts = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.connectors.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash}{marker}/>')

    def connect_lr(self, src: str, dst: str, *, mid_x: float | None = None, dashed: bool = False) -> None:
        a, b = self.nodes[src], self.nodes[dst]
        start, end = a.right, b.left
        if abs(start[1] - end[1]) < 1:
            points = [start, end]
        else:
            mx = mid_x if mid_x is not None else (start[0] + end[0]) / 2
            points = [start, (mx, start[1]), (mx, end[1]), end]
        self.route(points, dashed=dashed)

    def title(self, value: str) -> None:
        self.text(30, 32, value, size=31, weight=700, anchor="start", fill=COLORS["logic_border"])
        self.route([(30, 59), (self.width - 30, 59)], arrow=False, sw=2.0, color="#C9B99C")

    def section(self, value: str, y: float) -> None:
        self.text(30, y, value, size=24, weight=700, anchor="start", fill=COLORS["logic_border"])
        self.route([(30, y + 24), (self.width - 30, y + 24)], arrow=False, sw=1.8, color="#D4C7B0")

    def audit(self) -> dict[str, object]:
        problems: list[str] = []
        node_items = list(self.nodes.items())
        for name, b in node_items:
            if b.x < 0 or b.y < 0 or b.x + b.w > self.width or b.y + b.h > self.height:
                problems.append(f"{name}: ngoài canvas")
        for idx, (name_a, a) in enumerate(node_items):
            for name_b, b in node_items[idx + 1 :]:
                overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                overlap_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                if overlap_x > 1 and overlap_y > 1:
                    problems.append(f"{name_a} chồng {name_b}: {overlap_x:.1f}×{overlap_y:.1f}")
        if problems:
            raise ValueError(f"Lỗi bố cục {self.name}: " + "; ".join(problems))
        return {"figure": self.name, "nodes": len(self.nodes), "overlaps": 0, "out_of_bounds": 0}

    def save(self) -> tuple[Path, dict[str, object]]:
        audit = self.audit()
        body = "\n".join(self.background + self.connectors + self.foreground)
        defs = "\n".join(self.defs)
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
<defs>{defs}</defs>
<rect width="100%" height="100%" fill="#FFFFFF"/>
{body}
</svg>'''
        path = SVG_DIR / f"{self.name}.svg"
        path.write_text(svg, encoding="utf-8")
        return path, audit


def fig31(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 940, "fig_3_1_overview_offline_online_v2")
    c.title("Tổng quan quy trình Rich Gallery G1 trong hai pha offline và online")
    c.section("OFFLINE — Chuẩn bị và huấn luyện", 88)
    c.image("off_xray", a["IMG000415_xray"], Box(35, 135, 125, 170))
    c.text(97, 325, "Ảnh X-quang BTXRD", size=18, weight=700)
    c.chip("off_label", Box(42, 350, 110, 40), "Nhãn cấp ảnh", size=17)
    c.module("d10", Box(220, 130, 205, 105), "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], kind="trained", title_size=25, subtitle_size=19)
    c.chip("ck10", Box(465, 160, 145, 44), "Checkpoint θ₁₀", kind="trained", size=18)
    c.module("d2", Box(220, 270, 205, 105), "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], kind="trained", title_size=25, subtitle_size=19)
    c.chip("ck2", Box(465, 300, 145, 44), "Checkpoint θ₂", kind="trained", size=18)
    c.connect_lr("off_xray", "d10", mid_x=190)
    c.connect_lr("off_xray", "d2", mid_x=190)
    c.connect_lr("d10", "ck10")
    c.connect_lr("d2", "ck2")
    c.route([(97, 390), (97, 410), (195, 410), (195, 183), (220, 183)], dashed=True)
    c.route([(97, 390), (195, 390), (195, 323), (220, 323)], dashed=True)

    for node_id, x, title in (("bio", 690, "BiomedCLIP"), ("sam", 895, "SAM ViT-B"), ("rad", 1100, "RAD-DINO")):
        c.module(node_id, Box(x, 130, 175, 86), title, "Pretrained", kind="frozen", frozen=True, title_size=23, subtitle_size=18)
    c.module("train_gallery", Box(690, 275, 180, 88), "Tập ứng viên", "huấn luyện", kind="data", title_size=23, subtitle_size=19)
    c.module("repr", Box(920, 275, 180, 88), "Biểu diễn", "ứng viên", kind="feature", title_size=23, subtitle_size=19)
    c.module("g1", Box(1150, 265, 180, 108), "G1", ["Học chấm điểm", "bằng MIL"], kind="trained", title_size=27, subtitle_size=19)
    c.chip("ckg1", Box(1380, 297, 160, 46), "Checkpoint θG1", kind="trained", size=18)
    c.connect_lr("train_gallery", "repr")
    c.connect_lr("repr", "g1")
    c.connect_lr("g1", "ckg1")
    c.route([(1187, 216), (1187, 245), (1240, 245), (1240, 265)], dashed=True)
    c.route([(152, 370), (650, 370), (650, 319), (690, 319)], dashed=True)

    c.section("ONLINE — Suy luận và chọn mặt nạ", 458)
    c.image("on_xray", a["IMG000415_xray"], Box(35, 515, 125, 180))
    c.text(97, 716, "Ảnh X-quang", size=20, weight=700)
    c.chip("on_label", Box(42, 742, 110, 40), "Nhãn cấp ảnh", size=17)
    for node_id, y, title, kind in (
        ("l320", 510, "LayerCAM-320", "logic"),
        ("l448", 610, "LayerCAM-448", "logic"),
        ("biosal", 710, "BiomedCLIP saliency", "frozen"),
    ):
        c.module(node_id, Box(220, y, 210, 70), title, kind=kind, designed=kind == "logic", title_size=21 if title == "BiomedCLIP saliency" else 23)
        c.connect_lr("on_xray", node_id, mid_x=190)
    c.module("evidence", Box(485, 610, 190, 92), "Bằng chứng định vị", "đa nguồn", kind="logic", designed=True, title_size=22, subtitle_size=19)
    for src in ("l320", "l448", "biosal"):
        c.connect_lr(src, "evidence", mid_x=455)
    c.module("generate", Box(730, 595, 200, 120), "Sinh tập ứng viên", "threshold → prompt → SAM", kind="logic", designed=True, title_size=23, subtitle_size=18)
    c.module("gallery", Box(985, 610, 165, 92), "Tập mặt nạ", "ứng viên Mᵢ", kind="data", title_size=23, subtitle_size=19)
    c.connect_lr("evidence", "generate")
    c.connect_lr("generate", "gallery")
    c.module("g1score", Box(1195, 535, 160, 82), "RAD-DINO → G1", "Logit G1", kind="feature", title_size=20, subtitle_size=18)
    c.module("srcscore", Box(1195, 685, 160, 72), "Điểm nguồn", kind="feature", title_size=22)
    c.connect_lr("gallery", "g1score", mid_x=1170)
    c.connect_lr("gallery", "srcscore", mid_x=1170)
    c.module("fusion", Box(1395, 570, 135, 92), "Hợp nhất", ["thứ hạng", "0,5 + 0,5"], kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.module("select", Box(1395, 700, 135, 72), "Chọn ứng viên", kind="logic", designed=True, title_size=20)
    c.connect_lr("g1score", "fusion", mid_x=1375)
    c.connect_lr("srcscore", "fusion", mid_x=1375)
    c.route([(1462, 662), (1462, 700)])
    c.image("final", a["IMG000415_selected_overlay_real"], Box(1550, 575, 105, 170), stroke=COLORS["output_border"])
    c.text(1602, 772, "Mặt nạ cuối", size=21, weight=700, fill=COLORS["output_border"])
    c.connect_lr("select", "final", mid_x=1540)
    return c.save()


def fig32(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 780, "fig_3_2_multisource_localization_v2")
    c.title("Trích xuất bằng chứng định vị từ ba nguồn bổ sung")
    c.image("input", a["IMG000415_xray"], Box(35, 250, 150, 245))
    c.text(110, 525, "Ảnh X-quang", size=23, weight=700)
    branches = [
        ("b320", 255, "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], "LayerCAM", ["Dense Block 2/3/4", "0,2 · 0,3 · 0,5"], "IMG000415_layercam320_p90_real", "LayerCAM-320"),
        ("b448", 735, "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], "LayerCAM", ["Dense Block 2/3/4"], "IMG000415_layercam448_p90_real", "LayerCAM-448"),
        ("bbio", 1215, "BiomedCLIP", ["Toàn ảnh + vùng cục bộ"], "Saliency", ["ảnh–văn bản"], "IMG000415_biomedclip_p90_real", "BiomedCLIP saliency"),
    ]
    for node, x, model, model_sub, op, op_sub, image_key, label in branches:
        kind = "frozen" if node == "bbio" else "trained"
        c.module(node, Box(x, 85, 310, 92), model, model_sub, kind=kind, frozen=node == "bbio", title_size=25, subtitle_size=19)
        c.module(node + "_op", Box(x + 45, 215, 220, 88), op, op_sub, kind="logic", designed=True, title_size=24, subtitle_size=18)
        c.image(node + "_img", a[image_key], Box(x + 10, 345, 290, 245), stroke=COLORS["logic_border"], fit="slice")
        c.text(x + 155, 620, label, size=23, weight=700)
        c.text(x + 155, 650, "Output thực nghiệm · P90", size=18, fill=COLORS["muted"])
        c.route([(x + 155, 177), (x + 155, 215)])
        c.route([(x + 155, 303), (x + 155, 345)])
    c.route([(185, 372), (220, 372), (220, 68), (1370, 68)], arrow=False)
    for center_x in (410, 890, 1370):
        c.route([(center_x, 68), (center_x, 85)])
    c.route([(255, 710), (1525, 710)], arrow=False, sw=2.5, color=COLORS["logic_border"])
    c.text(890, 740, "Giữ riêng từng nguồn — không trung bình bản đồ", size=23, weight=700, fill=COLORS["logic_border"])
    return c.save()


def fig33(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 900, "fig_3_3_candidate_gallery_generation_v2")
    c.title("Sinh tập mặt nạ ứng viên từ bằng chứng định vị")
    for idx, key in enumerate(("IMG000415_layercam320_p90_real", "IMG000415_layercam448_p90_real", "IMG000415_biomedclip_p90_real")):
        c.image(None, a[key], Box(30, 95 + idx * 115, 135, 100), fit="slice", register=False)
    c.text(97, 466, "Ba bản đồ định vị", size=22, weight=700)
    c.text(97, 494, "L320 · L448 · Bio", size=18, fill=COLORS["muted"])
    c.module("threshold", Box(210, 190, 180, 110), "Ngưỡng phân vị", ["P85 · P90 · P95", "riêng từng nguồn"], kind="logic", designed=True, title_size=23, subtitle_size=18)
    c.module("cc", Box(435, 185, 190, 120), "Tách thành phần", ["liên thông", "giữ tối đa 3 vùng"], kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.module("prompt", Box(670, 165, 190, 160), "Tạo prompt", ["Điểm · Hộp", "Điểm + hộp", "điểm âm: vành ngoài"], kind="logic", designed=True, title_size=24, subtitle_size=18)
    c.image("prompt_real", a["IMG000415_prompt_fusion_p90_real"], Box(905, 105, 270, 270), stroke=COLORS["logic_border"], fit="slice")
    c.text(1040, 405, "Ví dụ prompt thật tại P90", size=21, weight=700)
    c.module("sam", Box(1220, 175, 170, 120), "SAM ViT-B", "multimask = 3", kind="frozen", frozen=True, title_size=24, subtitle_size=19)
    c.route([(165, 245), (210, 245)])
    c.connect_lr("threshold", "cc")
    c.connect_lr("cc", "prompt")
    c.connect_lr("prompt", "prompt_real")
    c.connect_lr("prompt_real", "sam")
    for idx in range(6):
        x = 1430 + (idx % 3) * 75
        y = 105 + (idx // 3) * 105
        c.image(None, a[f"candidate_{idx + 1:02d}"], Box(x, y, 66, 66), rx=8, register=False)
    c.text(1515, 340, "Mặt nạ ứng viên thật", size=21, weight=700)
    c.route([(1390, 235), (1415, 235)])

    c.module("grid", Box(245, 555, 190, 88), "Quy về lưới", "320 × 320", kind="logic", designed=True, title_size=23, subtitle_size=19)
    c.module("merge", Box(490, 555, 190, 88), "Gộp ba nguồn", kind="logic", designed=True, title_size=23)
    c.module("dedupe", Box(735, 545, 215, 108), "Loại trùng hoàn toàn", "so khớp từng điểm ảnh", kind="logic", designed=True, title_size=21, subtitle_size=18)
    c.module("gallery", Box(1030, 535, 240, 125), "Candidate Gallery Mᵢ", ["giữ metadata nguồn", "≤ 243 trước lọc"], kind="data", title_size=21, subtitle_size=19)
    c.connect_lr("grid", "merge")
    c.connect_lr("merge", "dedupe")
    c.connect_lr("dedupe", "gallery")
    c.route([(1515, 365), (1515, 500), (340, 500), (340, 555)])

    c.module("source_score", Box(1360, 515, 285, 150), "Điểm nguồn", ["D — Mật độ · M — Khối lượng", "R — Hạng SAM", "0,60D + 0,25M + 0,15R"], kind="feature", title_size=24, subtitle_size=17)
    c.route([(1515, 340), (1515, 485), (1502, 485), (1502, 515)], dashed=True)
    c.text(230, 765, "3 phân vị × ≤3 vùng × 3 kiểu prompt × 3 mặt nạ SAM", size=21, weight=700, anchor="start")
    c.text(230, 805, "≤ 81 ứng viên / nguồn", size=24, weight=700, anchor="start", fill=COLORS["logic_border"])
    c.text(1030, 805, "≤ 243 ứng viên / ảnh trước lọc", size=24, weight=700, anchor="start", fill=COLORS["logic_border"])
    return c.save()


def fig34(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 720, "fig_3_4_candidate_representation_g1_v2")
    c.title("Biểu diễn ứng viên và mạng chấm điểm G1")
    c.image("xray", a["IMG000415_xray"], Box(35, 105, 145, 200))
    c.text(107, 335, "Ảnh X-quang (Iᵢ)", size=21, weight=700)
    c.image("mask", a["candidate_01"], Box(48, 445, 120, 120))
    c.text(108, 595, "Ứng viên (mᵢⱼ)", size=21, weight=700)
    c.module("rad", Box(245, 90, 190, 92), "RAD-DINO", "Trích đặc trưng đa tầng", kind="frozen", frozen=True, title_size=25, subtitle_size=17)
    c.module("layers", Box(500, 90, 205, 92), "Đặc trưng đa tầng", "lớp 4 · 8 · 12", kind="feature", title_size=23, subtitle_size=19)
    c.module("proj", Box(770, 90, 190, 92), "Chiếu cố định", "768 → 128", kind="feature", title_size=23, subtitle_size=20)
    c.connect_lr("xray", "rad", mid_x=210)
    c.connect_lr("rad", "layers")
    c.connect_lr("layers", "proj")
    c.module("pool", Box(300, 390, 230, 150), "Pooling theo mặt nạ", ["Trong vùng", "Ngữ cảnh", "Tương phản"], kind="logic", designed=True, title_size=24, subtitle_size=20)
    c.connect_lr("mask", "pool", mid_x=230)
    c.route([(865, 182), (865, 335), (415, 335), (415, 390)])
    c.module("radfeat", Box(610, 380, 250, 130), "Đặc trưng RAD-DINO", ["3 tầng × 3 thống kê × 128", "1152 chiều"], kind="feature", title_size=23, subtitle_size=18)
    c.connect_lr("pool", "radfeat")
    c.module("aux", Box(610, 555, 250, 120), "4 đặc trưng bổ sung", ["SAM score · Diện tích log", "Khối lượng · Đáp ứng TB"], kind="feature", title_size=22, subtitle_size=18)
    c.module("concat", Box(970, 395, 215, 120), "Ghép đặc trưng", "xᵢⱼ ∈ ℝ¹¹⁵⁶", kind="feature", title_size=24, subtitle_size=21)
    c.connect_lr("radfeat", "concat")
    c.route([(860, 615), (920, 615), (920, 475), (970, 475)])
    c.module("g1", Box(1260, 380, 210, 150), "G1", ["LayerNorm → 256", "→ 128 → 1", "GELU · Dropout 0,10"], kind="trained", title_size=29, subtitle_size=18)
    c.module("logit", Box(1520, 405, 135, 100), "Logit ứng viên", "aᵢⱼ", kind="feature", title_size=20, subtitle_size=22)
    c.connect_lr("concat", "g1")
    c.connect_lr("g1", "logit")
    c.text(1130, 695, "Điểm nguồn sˢʳᶜ nằm ngoài vector G1", size=21, weight=700, fill=COLORS["logic_border"])
    return c.save()


def fig35(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 720, "fig_3_5_g1_mil_training_v2")
    c.title("Huấn luyện G1 từ nhãn cấp ảnh bằng multiple-instance learning")
    c.image("xray", a["IMG000415_xray"], Box(35, 100, 135, 185))
    c.chip("label", Box(42, 320, 120, 42), "Nhãn: Khối u", size=18)
    c.text(35, 405, "Không có nhãn ở mức ứng viên", size=21, weight=700, anchor="start", fill=COLORS["logic_border"])
    for idx in range(6):
        x = 38 + (idx % 3) * 78
        y = 450 + (idx // 3) * 98
        c.image(None, a[f"candidate_{idx + 1:02d}"], Box(x, y, 66, 66), rx=8, register=False)
        c.text(x + 33, y + 80, f"m{idx + 1}", size=17, weight=700)
    c.text(150, 665, "Bag ứng viên", size=22, weight=700)
    c.module("g1", Box(340, 240, 220, 135), "G1 dùng chung", ["m₁ … mₖ", "cùng một bộ trọng số"], kind="trained", title_size=26, subtitle_size=19)
    c.module("logits", Box(640, 240, 210, 135), "Logit từng ứng viên", ["a₁ · a₂ · … · aₖ"], kind="feature", title_size=23, subtitle_size=21)
    c.module("pool", Box(930, 230, 210, 155), "Smooth pooling", ["T = 0,20", "Logit cấp bag"], kind="logic", designed=True, title_size=25, subtitle_size=20)
    c.module("loss", Box(1220, 240, 190, 135), "Mất mát cấp bag", "BCE", kind="trained", title_size=23, subtitle_size=22)
    c.route([(240, 530), (295, 530), (295, 307), (340, 307)])
    c.connect_lr("g1", "logits")
    c.connect_lr("logits", "pool")
    c.connect_lr("pool", "loss")
    c.route([(162, 341), (190, 341), (190, 120), (1315, 120), (1315, 240)], dashed=True)
    c.module("neg", Box(555, 515, 210, 82), "Ràng buộc bag âm", "Lₙₑg", kind="eval", dashed=True, title_size=21, subtitle_size=19)
    c.module("cons", Box(810, 515, 220, 82), "Nhất quán lật ngang", "L꜀ₒₙₛ", kind="eval", dashed=True, title_size=21, subtitle_size=19)
    c.module("objective", Box(1120, 500, 420, 110), "Mục tiêu huấn luyện", "L = Lbag + 0,25Lneg + 0,10Lcons", kind="logic", designed=True, title_size=24, subtitle_size=19)
    c.route([(660, 515), (660, 445), (1280, 445), (1280, 500)], dashed=True)
    c.route([(920, 515), (920, 470), (1310, 470), (1310, 500)], dashed=True)
    c.route([(1315, 375), (1315, 500)], dashed=True)
    return c.save()


def fig36(a: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 680, "fig_3_6_rank_fusion_final_selection_v2")
    c.title("Hợp nhất thứ hạng và lựa chọn mặt nạ cuối")
    c.module("label", Box(35, 95, 210, 82), "Nhãn cấp ảnh", "thiết lập chính", kind="data", title_size=24, subtitle_size=18)
    c.module("normal", Box(35, 215, 210, 72), "Bình thường", "Mặt nạ rỗng", kind="eval", title_size=22, subtitle_size=18)
    c.module("tumor", Box(35, 335, 210, 72), "Khối u", "Xếp hạng ứng viên", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.route([(140, 177), (140, 215)])
    c.route([(245, 136), (275, 136), (275, 371), (245, 371)])
    c.text(35, 455, "Triển khai không nhãn:", size=18, weight=700, anchor="start", fill=COLORS["eval_border"])
    c.text(35, 482, "có thể thay bằng lớp dự đoán", size=18, anchor="start", fill=COLORS["eval_border"])
    c.module("orig", Box(330, 90, 210, 78), "Logit G1 — ảnh gốc", kind="feature", title_size=20)
    c.module("flip", Box(330, 205, 210, 78), "Logit G1 — ảnh lật", kind="feature", title_size=20)
    c.module("avg", Box(610, 125, 190, 95), "Lấy trung bình", "logit G1", kind="logic", designed=True, title_size=23, subtitle_size=19)
    c.module("grank", Box(865, 125, 185, 95), "Hạng phân vị", "G1", kind="logic", designed=True, title_size=23, subtitle_size=20)
    c.connect_lr("orig", "avg", mid_x=575)
    c.connect_lr("flip", "avg", mid_x=575)
    c.connect_lr("avg", "grank")
    c.module("source", Box(430, 350, 210, 82), "Điểm nguồn (sˢʳᶜ)", kind="feature", title_size=21)
    c.module("srank", Box(710, 345, 190, 92), "Hạng phân vị", "nguồn", kind="logic", designed=True, title_size=23, subtitle_size=20)
    c.connect_lr("source", "srank")
    c.module("fusion", Box(1085, 205, 210, 125), "Hợp nhất thứ hạng", "0,5 G1 + 0,5 nguồn", kind="logic", designed=True, title_size=24, subtitle_size=19)
    c.connect_lr("grank", "fusion", mid_x=1065)
    c.connect_lr("srank", "fusion", mid_x=1065)
    c.module("choose", Box(1360, 190, 200, 155), "Chọn hạng cao nhất", ["đồng hạng cố định", "1. hợp nhất · 2. G1 · 3. chỉ số"], kind="logic", designed=True, title_size=22, subtitle_size=17)
    c.connect_lr("fusion", "choose")
    c.module("gallery", Box(430, 535, 210, 78), "Tập mặt nạ (Mᵢ)", kind="data", title_size=23)
    c.module("take", Box(900, 525, 210, 92), "Lấy mặt nạ", "tại j*", kind="output", title_size=24, subtitle_size=21)
    c.connect_lr("gallery", "take")
    c.route([(1460, 345), (1460, 465), (1005, 465), (1005, 525)])
    for idx in range(3):
        x = 1160 + idx * 72
        c.image(None, a[f"top_candidate_{idx + 1:02d}"], Box(x, 500, 62, 62), rx=8, register=False)
        c.text(x + 31, 580, f"#{idx + 1}", size=17, weight=700)
    c.text(1250, 615, "Top-3 ứng viên thật", size=19, weight=700)
    c.image("final", a["IMG000415_selected_overlay_real"], Box(1515, 405, 140, 205), stroke=COLORS["output_border"])
    c.text(1585, 640, "Mặt nạ cuối", size=21, weight=700, fill=COLORS["output_border"])
    c.connect_lr("take", "final", mid_x=1485)
    return c.save()


def fig4x(_: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    c = Canvas(1680, 540, "fig_4_x_prediction_lock_evaluation_v2")
    c.title("Giao thức khóa dự đoán trước khi truy cập mặt nạ chuẩn")
    stages = [
        ("input", 30, "Ảnh + nhãn", "cấp ảnh", "data"),
        ("pipe", 250, "WSSS pipeline", None, "logic"),
        ("scores", 470, "Ứng viên + điểm", None, "feature"),
        ("selected", 690, "Mặt nạ được chọn", None, "output"),
        ("save", 910, "Lưu kết quả", "+ hash", "eval"),
        ("lock", 1130, "KHÓA DỰ ĐOÁN", None, "logic"),
    ]
    for node, x, title, subtitle, kind in stages:
        c.module(node, Box(x, 105, 180, 92), title, subtitle, kind=kind, designed=kind == "logic", title_size=20 if len(title) > 13 else 23, subtitle_size=19)
    for (src, *_), (dst, *__) in zip(stages, stages[1:]):
        c.connect_lr(src, dst)
    c.module("gt", Box(1360, 85, 280, 132), "Mặt nạ chuẩn", "Chỉ dùng sau khi khóa dự đoán", kind="eval", dashed=True, title_size=25, subtitle_size=18)
    c.connect_lr("lock", "gt", dashed=True)
    c.module("metrics", Box(760, 340, 310, 105), "Đánh giá", "Dice · IoU · Precision · Recall", kind="eval", dashed=True, title_size=25, subtitle_size=18)
    c.module("post", Box(1135, 340, 360, 105), "Phân tích hậu nghiệm", "Oracle Dice · Selector Regret", kind="eval", dashed=True, title_size=24, subtitle_size=18)
    c.route([(1500, 217), (1500, 285), (915, 285), (915, 340)], dashed=True)
    c.route([(1500, 285), (1315, 285), (1315, 340)], dashed=True)
    c.text(35, 495, "Mặt nạ chuẩn không tham gia định vị, tạo prompt, sinh ứng viên hoặc lựa chọn.", size=23, weight=700, anchor="start", fill=COLORS["eval_border"])
    return c.save()


CAPTIONS = {
    "fig_3_1_overview_offline_online_v2": "Tổng quan Rich Gallery G1 trong hai pha offline và online. Pha offline huấn luyện hai bộ phân loại DenseNet-121 và bộ chấm điểm G1 từ nhãn cấp ảnh; pha online khai thác ba nguồn bằng chứng định vị, sinh tập mặt nạ ứng viên và lựa chọn một mặt nạ cuối bằng hợp nhất thứ hạng G1–nguồn.",
    "fig_3_2_multisource_localization_v2": "Trích xuất ba nguồn bằng chứng định vị. Hai bộ phân loại DenseNet-121 tạo LayerCAM ở độ phân giải 320 và 448, trong khi BiomedCLIP cung cấp saliency từ biểu diễn ảnh–văn bản. Các output P90 thực nghiệm của IMG000415 được lấy trực tiếp từ project Prism và giữ riêng theo nguồn.",
    "fig_3_3_candidate_gallery_generation_v2": "Quy trình sinh Candidate Gallery từ ba nguồn định vị. Mỗi bản đồ được cắt tại ba mức phân vị, tách thành phần liên thông và chuyển thành prompt điểm, hộp hoặc kết hợp để SAM ViT-B sinh nhiều giả thuyết mặt nạ.",
    "fig_3_4_candidate_representation_g1_v2": "Biểu diễn và chấm điểm một ứng viên. RAD-DINO đóng băng cung cấp đặc trưng tại ba tầng; thống kê trong vùng, ngữ cảnh và tương phản được ghép với bốn đặc trưng bổ sung thành vector 1156 chiều trước khi G1 sinh logit.",
    "fig_3_5_g1_mil_training_v2": "Huấn luyện G1 bằng multiple-instance learning. Các mặt nạ của cùng một ảnh tạo thành một bag không có nhãn ở mức ứng viên; smooth pooling tổng hợp logit ứng viên thành logit cấp bag để tối ưu theo nhãn cấp ảnh.",
    "fig_3_6_rank_fusion_final_selection_v2": "Hợp nhất thứ hạng và lựa chọn mặt nạ cuối. Logit G1 từ ảnh gốc và ảnh lật được lấy trung bình, sau đó kết hợp với hạng phân vị của điểm nguồn theo trọng số bằng nhau.",
    "fig_4_x_prediction_lock_evaluation_v2": "Giao thức khóa dự đoán trước khi truy cập mặt nạ chuẩn. Oracle Dice và Selector Regret chỉ được tính trong phân tích hậu nghiệm.",
}


def write_support_files(svg_paths: list[Path], audits: list[dict[str, object]]) -> None:
    for svg_path in svg_paths:
        head = svg_path.read_text(encoding="utf-8")[:300]
        match = re.search(r'<svg[^>]+width="(\d+)"[^>]+height="(\d+)"', head)
        if not match:
            raise ValueError(svg_path)
        width, height = (int(v) for v in match.groups())
        width_mm = 160.0
        height_mm = height / width * width_mm
        html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {width_mm:.2f}mm {height_mm:.2f}mm; margin: 0; }}
html,body {{ margin:0; width:{width_mm:.2f}mm; height:{height_mm:.2f}mm; overflow:hidden; background:white; }}
img {{ display:block; width:100%; height:100%; }}
</style></head><body><img src="{svg_path.resolve().as_uri()}"></body></html>'''
        (HTML_DIR / f"{svg_path.stem}.html").write_text(html, encoding="utf-8")
    (OUT / "layout_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "captions_vi.json").write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# Rich Gallery G1 — figure set v2

Bộ hình này được dựng lại từ đầu theo grid và gutter cố định. Mũi tên được vẽ ở lớp dưới các module; `layout_audit.json` xác nhận không có hộp chồng nhau hoặc nằm ngoài canvas.

## Dữ liệu ảnh

- IMG000415 được dùng nhất quán trong cả bảy hình.
- Các output LayerCAM-320, LayerCAM-448, BiomedCLIP và prompt fusion P90 lấy trực tiếp từ project Prism.
- Không dùng heatmap tái dựng trong Hình 3.2.
- Candidate masks và final mask lấy từ diagnostics thực nghiệm local của IMG000415.

## Thư mục

- `source_images/`: nguyên bản tải từ Prism.
- `assets/`: crop và mask dùng trong figure.
- `svg/`: bản vector có raster embedded.
- `rendered/`: PNG và PDF sau khi render.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    assets = prepare_assets()
    outputs = [fig31(assets), fig32(assets), fig33(assets), fig34(assets), fig35(assets), fig36(assets), fig4x(assets)]
    svg_paths = [item[0] for item in outputs]
    audits = [item[1] for item in outputs]
    write_support_files(svg_paths, audits)
    print(json.dumps({"svg": [str(p) for p in svg_paths], "audits": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
