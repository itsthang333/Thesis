from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets" / "rich_gallery_figures"
ASSETS = OUT / "assets"
SVG_DIR = OUT / "svg"
RENDER_DIR = OUT / "rendered"
WRAPPER_DIR = ROOT / "tmp" / "rich_gallery_figure_render"

XRAY_415 = ROOT / "BTXRD" / "BTXRD" / "images" / "IMG000415.jpeg"
XRAY_1215 = ROOT / "BTXRD" / "BTXRD" / "images" / "IMG001215.jpeg"
DIAG_TWO_SOURCE = (
    ROOT
    / "artifacts"
    / "section_3_4_img000415"
    / "real_rerun"
    / "biomedclip_layercam320_single"
    / "candidate_diagnostics"
    / "IMG000415.npz"
)
DIAG_448 = (
    ROOT
    / "artifacts"
    / "section_3_4_img000415"
    / "real_rerun"
    / "layercam448"
    / "debug"
    / "IMG000415"
    / "candidate_diagnostics.npz"
)
MERGED = (
    ROOT
    / "artifacts"
    / "section_3_4_img000415"
    / "report_real_selected"
    / "IMG000415_merged_candidate_gallery_real.npz"
)
MAP_448 = ROOT / "artifacts" / "section_3_4_img000415" / "layercam448_reconstructed_map.npy"


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
    "arrow": "#454545",
    "white": "#FFFFFF",
    "muted": "#6B6864",
}


def ensure_dirs() -> None:
    for path in (OUT, ASSETS, SVG_DIR, RENDER_DIR, WRAPPER_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo = float(np.nanpercentile(a, 1))
    hi = float(np.nanpercentile(a, 99))
    if hi <= lo:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def magma_rgb(a: np.ndarray) -> np.ndarray:
    a = normalize(a)
    stops = np.array(
        [
            [0.000, 0x00, 0x00, 0x04],
            [0.180, 0x2C, 0x11, 0x67],
            [0.360, 0x72, 0x1F, 0x81],
            [0.540, 0xB7, 0x37, 0x79],
            [0.720, 0xF1, 0x60, 0x5D],
            [0.880, 0xFE, 0xB0, 0x78],
            [1.000, 0xFC, 0xFD, 0xBF],
        ],
        dtype=np.float32,
    )
    flat = a.ravel()
    out = np.zeros((flat.size, 3), dtype=np.float32)
    for idx in range(len(stops) - 1):
        l, r = stops[idx], stops[idx + 1]
        sel = (flat >= l[0]) & (flat <= r[0] if idx == len(stops) - 2 else flat < r[0])
        t = (flat[sel] - l[0]) / (r[0] - l[0])
        out[sel] = l[1:] * (1.0 - t[:, None]) + r[1:] * t[:, None]
    return np.clip(out.reshape(a.shape + (3,)), 0, 255).astype(np.uint8)


def save_png(image: Image.Image, path: Path, dpi: int = 450) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(dpi, dpi))


def square_xray(path: Path, size: int = 900) -> Image.Image:
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im, cutoff=0.5)
    im = ImageEnhance.Contrast(im).enhance(1.08)
    return ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS).convert("RGB")


def full_xray(path: Path, max_height: int = 1500) -> Image.Image:
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im, cutoff=0.5)
    im.thumbnail((max_height, max_height), Image.Resampling.LANCZOS)
    return im.convert("RGB")


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.62) -> Image.Image:
    heat = Image.fromarray(magma_rgb(heatmap), "RGB").resize(base.size, Image.Resampling.BICUBIC)
    n = normalize(heatmap)
    mask = Image.fromarray(np.uint8(np.clip((n - 0.08) / 0.92, 0, 1) * 255), "L").resize(
        base.size, Image.Resampling.BILINEAR
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, base.width // 250)))
    mask = mask.point(lambda p: int(p * alpha))
    return Image.composite(heat, base, mask)


def mask_png(mask: np.ndarray, size: int = 520) -> Image.Image:
    m = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, "L")
    m = m.resize((size, size), Image.Resampling.NEAREST)
    bg = Image.new("RGB", (size, size), "#171717")
    fg = Image.new("RGB", (size, size), "#F8F8F5")
    return Image.composite(fg, bg, m)


def source_reconstruction(masks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.maximum(weights, 0.03)
    acc = np.tensordot(weights, masks.astype(np.float32), axes=(0, 0)) / float(weights.sum())
    acc = Image.fromarray(np.uint8(normalize(acc) * 255), "L").filter(ImageFilter.GaussianBlur(8))
    return np.asarray(acc, dtype=np.float32) / 255.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def select_diverse(masks: np.ndarray, scores: np.ndarray, count: int = 8) -> list[int]:
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for idx in order:
        area = float(masks[idx].mean())
        if area < 0.001 or area > 0.50:
            continue
        if all(iou(masks[idx], masks[j]) < 0.82 for j in chosen):
            chosen.append(int(idx))
        if len(chosen) == count:
            break
    for idx in order:
        if len(chosen) == count:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))
    return chosen


def extract_assets() -> dict[str, Path]:
    ensure_dirs()
    with np.load(DIAG_TWO_SOURCE, allow_pickle=True) as z:
        two = {k: z[k] for k in z.files}
    with np.load(DIAG_448, allow_pickle=True) as z:
        d448 = {k: z[k] for k in z.files}
    with np.load(MERGED, allow_pickle=True) as z:
        merged = {k: z[k] for k in z.files}

    xray_square = square_xray(XRAY_415)
    xray_full = full_xray(XRAY_415)
    xray_1215 = full_xray(XRAY_1215)
    save_png(xray_square, ASSETS / "IMG000415_xray_square.png")
    save_png(xray_full, ASSETS / "IMG000415_xray.png")
    save_png(xray_1215, ASSETS / "IMG001215_xray.png")

    source_ids = merged["source_ids"].astype(str)
    masks = merged["sam_masks"].astype(np.uint8)
    source_scores = merged["source_local_scores"].astype(np.float32)

    lc320_sel = source_ids == "LayerCAM-320"
    bio_sel = source_ids == "BiomedCLIP"
    lc320_recon = source_reconstruction(masks[lc320_sel], source_scores[lc320_sel])
    bio_recon = source_reconstruction(masks[bio_sel], source_scores[bio_sel])
    map448 = normalize(np.load(MAP_448))

    # The stored prompt map is the closest local pre-SAM evidence raster; blend it with
    # the source-specific reconstruction so the LayerCAM-320 panel retains real prompt detail.
    prompt_map = normalize(two["prompt_map"])
    lc320_map = normalize(0.72 * prompt_map + 0.28 * lc320_recon)
    bio_map = normalize(bio_recon)

    for name, hm in (
        ("IMG000415_layercam320", lc320_map),
        ("IMG000415_layercam448", map448),
        ("IMG000415_biomedclip_saliency", bio_map),
    ):
        raw = Image.fromarray(magma_rgb(hm), "RGB").resize((900, 900), Image.Resampling.BICUBIC)
        save_png(raw, ASSETS / f"{name}_map.png")
        save_png(overlay_heatmap(xray_square, hm), ASSETS / f"{name}_overlay.png")

    p90 = prompt_map >= np.percentile(prompt_map[prompt_map > 0], 90)
    save_png(mask_png(p90, 600), ASSETS / "IMG000415_component_p90.png")

    prompt = overlay_heatmap(xray_square, prompt_map, alpha=0.42)
    draw = ImageDraw.Draw(prompt)
    sx = prompt.width / 320.0
    sy = prompt.height / 320.0
    boxes = two["boxes"]
    if len(boxes):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        bx = boxes[int(np.argsort(areas)[len(areas) // 2])]
        draw.rectangle(tuple(float(v) * (sx if i % 2 == 0 else sy) for i, v in enumerate(bx)), outline="#E7C34A", width=8)
    for x, y in two["positive_points"][:: max(1, len(two["positive_points"]) // 7)][:7]:
        cx, cy = int(x * sx), int(y * sy)
        draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill="#F7E45B", outline="#403718", width=3)
    for x, y in two["negative_points"][:: max(1, len(two["negative_points"]) // 6)][:6]:
        cx, cy = int(x * sx), int(y * sy)
        draw.line((cx - 9, cy - 9, cx + 9, cy + 9), fill="#4FC3D7", width=5)
        draw.line((cx - 9, cy + 9, cx + 9, cy - 9), fill="#4FC3D7", width=5)
    save_png(prompt, ASSETS / "IMG000415_prompt_p90.png")

    selected_indices = select_diverse(masks, source_scores, 8)
    for rank, idx in enumerate(selected_indices, 1):
        save_png(mask_png(masks[idx]), ASSETS / f"IMG000415_candidate_{rank:02d}.png")
    final_mask = two["final_mask"].astype(np.uint8)
    save_png(mask_png(final_mask, 700), ASSETS / "IMG000415_selected_mask.png")

    selected_overlay = xray_square.copy()
    alpha_mask = Image.fromarray(final_mask * 150, "L").resize(selected_overlay.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", selected_overlay.size, "#D97562")
    selected_overlay = Image.composite(red, selected_overlay, alpha_mask)
    save_png(selected_overlay, ASSETS / "IMG000415_selected_overlay.png")

    manifest = {
        "sample_used": "IMG000415",
        "reason": (
            "IMG001215 X-ray is local, but its candidate diagnostics/top-3 masks are absent locally and in the supplied Prism project. "
            "IMG000415 is therefore used consistently to avoid mixing images and masks from different cases."
        ),
        "sources": {
            "xray": str(XRAY_415.relative_to(ROOT)),
            "two_source_diagnostics": str(DIAG_TWO_SOURCE.relative_to(ROOT)),
            "layercam448_diagnostics": str(DIAG_448.relative_to(ROOT)),
            "merged_gallery": str(MERGED.relative_to(ROOT)),
            "layercam448_map": str(MAP_448.relative_to(ROOT)),
        },
        "derived_panels": {
            "LayerCAM-320": "stored prompt_map blended with source-specific candidate reconstruction",
            "LayerCAM-448": "stored reconstructed LayerCAM-448 map",
            "BiomedCLIP": "source-specific reconstruction from local BiomedCLIP-origin candidates",
        },
        "candidate_indices": selected_indices,
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {p.stem: p for p in ASSETS.glob("*.png")}


def png_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


@dataclass
class SvgCanvas:
    width: int
    height: int
    name: str

    def __post_init__(self) -> None:
        self.items: list[str] = []
        self.defs: list[str] = [
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#454545"/></marker>'
        ]
        self._clip_id = 0

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str, *, rx: float = 20, sw: float = 2.8, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def line(self, points: list[tuple[float, float]], *, dashed: bool = False, arrow: bool = True, sw: float = 3.2, color: str = COLORS["arrow"]) -> None:
        p = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.items.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash}{marker}/>' )

    def text(self, x: float, y: float, value: str, *, size: float = 24, weight: int = 400, anchor: str = "middle", fill: str = COLORS["text"], italic: bool = False, family: str = "Arial") -> None:
        style = ' font-style="italic"' if italic else ""
        self.items.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}"{style}>{escape(value)}</text>'
        )

    def multiline(self, x: float, y: float, lines: list[str], *, size: float = 24, weight: int = 400, line_height: float | None = None, fill: str = COLORS["text"], anchor: str = "middle") -> None:
        line_height = line_height or size * 1.25
        start = y - (len(lines) - 1) * line_height / 2
        for idx, line in enumerate(lines):
            self.text(x, start + idx * line_height, line, size=size, weight=weight if idx == 0 else min(weight, 500), fill=fill, anchor=anchor)

    def image(self, path: Path, x: float, y: float, w: float, h: float, *, stroke: str = COLORS["data_border"], sw: float = 2.5, rx: float = 16, fit: str = "meet") -> None:
        self._clip_id += 1
        clip = f"clip{self._clip_id}"
        self.defs.append(f'<clipPath id="{clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>')
        self.items.append(
            f'<image href="{png_data_uri(path)}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid {fit}" clip-path="url(#{clip})"/>'
        )
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')

    def module(self, x: float, y: float, w: float, h: float, title: str, subtitle: str | list[str] | None = None, *, kind: str = "data", badge: str | None = None, designed: bool = False, title_size: float = 25, subtitle_size: float = 21, dashed: bool = False) -> None:
        fill = COLORS[f"{kind}_fill"]
        border = COLORS[f"{kind}_border"]
        self.rect(x, y, w, h, fill, border, dash="10 8" if dashed else None)
        if designed:
            self.items.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{border}" stroke="none"/>')
        title_y = y + (h * (0.55 if badge else 0.43) if subtitle else h / 2)
        self.text(x + w / 2, title_y, title, size=title_size, weight=700)
        if subtitle:
            lines = [subtitle] if isinstance(subtitle, str) else subtitle
            subtitle_y = y + h * (0.80 if badge else 0.71)
            self.multiline(x + w / 2, subtitle_y, list(lines), size=subtitle_size, weight=400, line_height=subtitle_size * 1.18, fill=COLORS["muted"])
        if badge:
            bw = max(82, 14 * len(badge))
            self.rect(x + w - bw - 9, y + 8, bw, 30, COLORS["white"], border, rx=11, sw=1.6)
            self.text(x + w - bw / 2 - 9, y + 23, badge, size=16, weight=700, fill=border)

    def chip(self, x: float, y: float, w: float, text: str, *, kind: str = "data", dashed: bool = False) -> None:
        self.rect(x, y, w, 42, COLORS[f"{kind}_fill"], COLORS[f"{kind}_border"], rx=14, sw=2.0, dash="8 6" if dashed else None)
        self.text(x + w / 2, y + 21, text, size=18, weight=600)

    def save(self) -> Path:
        content = "\n".join(self.items)
        defs = "\n".join(self.defs)
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
<defs>{defs}</defs>
<rect width="100%" height="100%" fill="#FFFFFF"/>
{content}
</svg>'''
        path = SVG_DIR / f"{self.name}.svg"
        path.write_text(svg, encoding="utf-8")
        return path


def title_band(c: SvgCanvas, text: str, y: int) -> None:
    c.text(30, y, text, size=24, weight=700, anchor="start", fill=COLORS["logic_border"])
    c.line([(30, y + 25), (c.width - 30, y + 25)], arrow=False, sw=2.2, color="#C8B99E")


def figure_31(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1680, 940, "fig_3_1_overview_offline_online")
    title_band(c, "OFFLINE — Chuẩn bị và huấn luyện", 34)
    c.image(a["IMG000415_xray"], 34, 88, 145, 195)
    c.text(106, 305, "Ảnh X-quang BTXRD", size=19, weight=600)
    c.chip(52, 330, 110, "Nhãn cấp ảnh")

    c.module(220, 90, 190, 94, "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], kind="trained", title_size=23, subtitle_size=18)
    c.chip(250, 204, 130, "Checkpoint θ₁₀", kind="trained")
    c.module(450, 90, 190, 94, "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], kind="trained", title_size=23, subtitle_size=18)
    c.chip(480, 204, 130, "Checkpoint θ₂", kind="trained")
    c.line([(179, 150), (220, 150)])
    c.line([(179, 168), (205, 168), (205, 137), (450, 137)])
    c.line([(162, 351), (300, 351), (300, 184)], dashed=True, sw=2.5)
    c.line([(162, 351), (530, 351), (530, 184)], dashed=True, sw=2.5)

    for x, name in [(710, "BiomedCLIP"), (910, "SAM ViT-B"), (1110, "RAD-DINO")]:
        c.module(x, 88, 170, 90, name, "Đóng băng", kind="frozen", badge="Đóng băng", title_size=22, subtitle_size=17)

    train_x = [700, 905, 1110, 1325]
    c.module(train_x[0], 230, 170, 78, "Tập ứng viên", "huấn luyện", kind="data", title_size=22, subtitle_size=18)
    c.module(train_x[1], 230, 170, 78, "Biểu diễn", "ứng viên", kind="feature", title_size=22, subtitle_size=18)
    c.module(train_x[2], 220, 175, 98, "G1", ["Học chấm điểm", "bằng MIL"], kind="trained", title_size=25, subtitle_size=18)
    c.chip(train_x[3], 246, 160, "Checkpoint θG1", kind="trained")
    c.line([(870, 269), (905, 269)])
    c.line([(1075, 269), (1110, 269)])
    c.line([(1285, 269), (1325, 269)])
    c.line([(1195, 178), (1195, 220)], dashed=True, sw=2.4)
    c.line([(162, 351), (1195, 351), (1195, 318)], dashed=True, sw=2.4)

    title_band(c, "ONLINE — Suy luận và chọn mặt nạ", 410)
    c.image(a["IMG000415_xray"], 32, 478, 145, 205)
    c.text(104, 706, "Ảnh X-quang", size=19, weight=600)
    c.chip(50, 730, 112, "Nhãn cấp ảnh")

    for y, label in [(476, "LayerCAM-320"), (565, "LayerCAM-448"), (654, "BiomedCLIP saliency")]:
        c.module(215, y, 205, 66, label, kind="logic" if "Layer" in label else "frozen", designed="Layer" in label, title_size=18 if "BiomedCLIP" in label else 20)
        c.line([(177, 575), (196, 575), (196, y + 33), (215, y + 33)])
    c.module(474, 555, 190, 90, "Bằng chứng định vị", "đa nguồn", kind="logic", designed=True, title_size=19, subtitle_size=18)
    for y in (509, 598, 687):
        c.line([(420, y), (446, y), (446, 600), (474, 600)])
    c.module(710, 540, 205, 120, "Sinh tập ứng viên", "threshold → prompt → SAM", kind="logic", designed=True, title_size=20, subtitle_size=16)
    c.line([(664, 600), (710, 600)])
    c.module(958, 548, 155, 104, "Tập mặt nạ", "ứng viên Mᵢ", kind="data", title_size=22, subtitle_size=19)
    c.line([(915, 600), (958, 600)])

    c.module(1150, 498, 150, 74, "RAD-DINO → G1", "Logit G1", kind="trained", title_size=19, subtitle_size=17)
    c.module(1150, 638, 150, 74, "Điểm nguồn", kind="feature", title_size=21)
    c.line([(1113, 580), (1132, 580), (1132, 535), (1150, 535)])
    c.line([(1113, 620), (1132, 620), (1132, 675), (1150, 675)])
    c.module(1340, 550, 165, 102, "Hợp nhất thứ hạng", "0,5 G1 + 0,5 nguồn", kind="logic", designed=True, title_size=18, subtitle_size=16)
    c.line([(1300, 535), (1320, 535), (1320, 601), (1340, 601)])
    c.line([(1300, 675), (1320, 675), (1320, 601), (1340, 601)])
    c.module(1530, 555, 120, 90, "Chọn", "ứng viên", kind="logic", designed=True, title_size=21, subtitle_size=18)
    c.line([(1505, 601), (1530, 601)])

    c.line([(1590, 645), (1590, 735)])
    c.image(a["IMG000415_selected_mask"], 1518, 742, 145, 145, stroke=COLORS["output_border"])
    c.text(1590, 910, "Mặt nạ cuối", size=22, weight=700, fill=COLORS["output_border"])
    c.chip(32, 826, 430, "Bình thường → mặt nạ rỗng · Khối u → xếp hạng", kind="eval")
    return c.save()


def figure_32(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1600, 780, "fig_3_2_multisource_localization")
    c.image(a["IMG000415_xray"], 35, 232, 210, 290)
    c.text(140, 552, "Ảnh X-quang", size=23, weight=700)
    rows = [
        (70, "DenseNet-121", ["Phân loại 10 lớp", "320 × 320"], "LayerCAM", "0,2 / 0,3 / 0,5", a["IMG000415_layercam320_overlay"], "LayerCAM-320", "trained"),
        (300, "DenseNet-121", ["Phân loại nhị phân", "448 × 448"], "LayerCAM", "Dense Block 2/3/4", a["IMG000415_layercam448_overlay"], "LayerCAM-448", "trained"),
        (530, "BiomedCLIP", ["Đóng băng"], "Toàn ảnh + vùng cục bộ", "Saliency", a["IMG000415_biomedclip_saliency_overlay"], "BiomedCLIP saliency", "frozen"),
    ]
    for y, model, sub, op, note, img, label, kind in rows:
        c.module(330, y, 220, 96, model, sub, kind=kind, badge="Đóng băng" if model == "BiomedCLIP" else None, title_size=23, subtitle_size=18)
        c.module(620, y + 3, 215, 90, op, note, kind="logic", designed=True, title_size=18 if "Toàn ảnh" in op else 22, subtitle_size=18)
        c.image(img, 990, y - 8, 205, 125)
        c.text(1092, y + 142, label, size=21, weight=700)
        c.line([(245, 377), (290, 377), (290, y + 48), (330, y + 48)])
        c.line([(550, y + 48), (620, y + 48)])
        c.line([(835, y + 48), (990, y + 48)])
    c.module(1245, 255, 305, 170, "Ba nguồn bổ sung", ["Giữ riêng từng bản đồ", "Không trung bình bản đồ"], kind="logic", designed=True, title_size=25, subtitle_size=20)
    for y in (118, 348, 578):
        c.line([(1195, y), (1220, y), (1220, 340), (1245, 340)], arrow=False, sw=2.4)
    c.line([(1245, 340), (1245, 340)], arrow=False)
    c.text(800, 748, "Giữ riêng từng nguồn — bảo toàn bằng chứng bổ sung", size=23, weight=700, fill=COLORS["logic_border"])
    return c.save()


def figure_33(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1680, 880, "fig_3_3_candidate_gallery_generation")
    heatmaps = [
        a["IMG000415_layercam320_overlay"],
        a["IMG000415_layercam448_overlay"],
        a["IMG000415_biomedclip_saliency_overlay"],
    ]
    for idx, path in enumerate(heatmaps):
        c.image(path, 28, 72 + idx * 145, 145, 125)
    c.text(100, 535, "Ba bản đồ định vị", size=22, weight=700)
    c.text(100, 566, "L320 · L448 · Bio", size=18, fill=COLORS["muted"])

    c.module(215, 170, 180, 115, "Ngưỡng phân vị", ["P85 · P90 · P95", "tính riêng từng nguồn"], kind="logic", designed=True, title_size=20, subtitle_size=18)
    c.module(445, 168, 195, 118, "Tách thành phần", ["liên thông", "giữ tối đa 3 vùng"], kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.image(a["IMG000415_component_p90"], 474, 318, 138, 138)
    c.module(690, 138, 190, 160, "Tạo prompt", ["Điểm · Hộp", "Điểm + hộp", "điểm âm từ vành ngoài"], kind="logic", designed=True, title_size=23, subtitle_size=18)
    c.image(a["IMG000415_prompt_p90"], 925, 105, 240, 240)
    c.text(1045, 373, "Ví dụ prompt tại P90", size=20, weight=700)
    c.module(1208, 166, 165, 115, "SAM ViT-B", "multimask = 3", kind="frozen", badge="Đóng băng", title_size=23, subtitle_size=18)

    c.line([(173, 260), (215, 228)])
    c.line([(395, 228), (445, 228)])
    c.line([(640, 228), (690, 218)])
    c.line([(880, 218), (925, 218)])
    c.line([(1165, 218), (1208, 218)])

    for idx in range(6):
        x = 1408 + (idx % 3) * 82
        y = 104 + (idx // 3) * 104
        c.image(a[f"IMG000415_candidate_{idx + 1:02d}"], x, y, 72, 72, rx=10)
    c.text(1495, 335, "Mặt nạ ứng viên", size=21, weight=700)
    c.line([(1373, 218), (1400, 218)])

    c.module(1050, 495, 315, 165, "Điểm nguồn", ["D — Mật độ bằng chứng", "M — Khối lượng bằng chứng", "R — Hạng SAM"], kind="feature", title_size=23, subtitle_size=18)
    c.text(1207, 648, "0,60D + 0,25M + 0,15R", size=17, weight=700, fill=COLORS["feature_border"])
    c.line([(1495, 335), (1495, 468), (1365, 468), (1365, 578)], dashed=True, sw=2.4)

    c.module(300, 650, 210, 86, "Quy về lưới", "320 × 320", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.module(565, 650, 190, 86, "Gộp ba nguồn", kind="logic", designed=True, title_size=22)
    c.module(810, 638, 210, 110, "Loại trùng hoàn toàn", "so khớp từng điểm ảnh", kind="logic", designed=True, title_size=18, subtitle_size=17)
    c.module(1400, 638, 245, 110, "Candidate Gallery Mᵢ", "giữ metadata nguồn", kind="data", title_size=20, subtitle_size=18)
    c.line([(510, 693), (565, 693)])
    c.line([(755, 693), (810, 693)])
    c.line([(1020, 693), (1400, 693)])
    c.text(330, 812, "3 phân vị × ≤3 vùng × 3 kiểu prompt × 3 mặt nạ SAM", size=19, weight=600)
    c.text(835, 812, "≤ 81 ứng viên / nguồn", size=21, weight=700, fill=COLORS["logic_border"])
    c.text(1290, 812, "≤ 243 ứng viên / ảnh trước lọc", size=21, weight=700, fill=COLORS["logic_border"])
    return c.save()


def figure_34(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1600, 730, "fig_3_4_candidate_representation_g1")
    c.image(a["IMG000415_xray"], 30, 80, 165, 230)
    c.text(112, 338, "Ảnh X-quang (Iᵢ)", size=21, weight=700)
    c.image(a["IMG000415_candidate_01"], 52, 420, 125, 125)
    c.text(114, 575, "Ứng viên (mᵢⱼ)", size=21, weight=700)

    c.module(250, 78, 190, 95, "RAD-DINO", "Đóng băng", kind="frozen", badge="Đóng băng", title_size=25, subtitle_size=18)
    c.module(500, 78, 210, 95, "Đặc trưng đa tầng", "lớp 4 · 8 · 12", kind="feature", title_size=22, subtitle_size=18)
    c.module(770, 78, 170, 95, "Chiếu cố định", "768 → 128", kind="feature", title_size=22, subtitle_size=19)
    c.line([(195, 195), (220, 195), (220, 126), (250, 126)])
    c.line([(440, 126), (500, 126)])
    c.line([(710, 126), (770, 126)])

    c.module(300, 385, 250, 150, "Pooling theo mặt nạ", ["Trong vùng", "Ngữ cảnh", "Tương phản"], kind="logic", designed=True, title_size=23, subtitle_size=20)
    c.line([(177, 482), (300, 460)])
    c.line([(855, 173), (855, 325), (550, 325), (550, 425)])
    c.module(625, 350, 250, 135, "Đặc trưng RAD-DINO", ["3 tầng × 3 thống kê × 128", "1152 chiều"], kind="feature", title_size=23, subtitle_size=19)
    c.line([(550, 460), (625, 418)])

    c.module(625, 535, 250, 155, "4 đặc trưng bổ sung", ["SAM score · Diện tích log", "Khối lượng bằng chứng", "Đáp ứng trung bình"], kind="feature", title_size=22, subtitle_size=17)
    c.module(965, 355, 210, 130, "Ghép đặc trưng", "xᵢⱼ ∈ ℝ¹¹⁵⁶", kind="feature", title_size=23, subtitle_size=21)
    c.line([(875, 418), (965, 418)])
    c.line([(875, 612), (925, 612), (925, 455), (965, 455)])
    c.module(1245, 348, 190, 145, "G1", ["LayerNorm → 256", "→ 128 → 1", "GELU · Dropout 0,10"], kind="trained", title_size=28, subtitle_size=18)
    c.line([(1175, 420), (1245, 420)])
    c.module(1470, 375, 110, 94, "Logit", "ứng viên aᵢⱼ", kind="feature", title_size=22, subtitle_size=17)
    c.line([(1435, 420), (1470, 420)])
    c.text(1100, 660, "Điểm nguồn sˢʳᶜ nằm ngoài vector G1", size=22, weight=700, fill=COLORS["logic_border"])
    return c.save()


def figure_35(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1600, 720, "fig_3_5_g1_mil_training")
    c.image(a["IMG000415_xray"], 30, 95, 150, 205)
    c.chip(45, 326, 120, "Nhãn: Khối u")
    c.text(185, 382, "Không có nhãn ở mức ứng viên", size=22, weight=700, anchor="start", fill=COLORS["logic_border"])
    for idx in range(6):
        x = 35 + (idx % 3) * 78
        y = 420 + (idx // 3) * 98
        c.image(a[f"IMG000415_candidate_{idx + 1:02d}"], x, y, 68, 68, rx=10)
        c.text(x + 34, y + 83, f"m{idx + 1}", size=16, weight=600)
    c.text(150, 640, "Bag ứng viên", size=22, weight=700)

    c.module(350, 230, 220, 150, "G1 dùng chung", ["m₁ … mₖ", "cùng một bộ trọng số"], kind="trained", title_size=25, subtitle_size=19)
    c.line([(260, 505), (310, 505), (310, 305), (350, 305)])
    c.module(650, 230, 200, 150, "Logit từng ứng viên", ["a₁", "a₂", "…  aₖ"], kind="feature", title_size=23, subtitle_size=21)
    c.line([(570, 305), (650, 305)])
    c.module(930, 230, 210, 150, "Smooth pooling", ["T = 0,20", "Logit cấp bag"], kind="logic", designed=True, title_size=24, subtitle_size=20)
    c.line([(850, 305), (930, 305)])
    c.module(1220, 230, 190, 150, "Mất mát cấp bag", "BCE", kind="trained", title_size=23, subtitle_size=20)
    c.line([(1140, 305), (1220, 305)])
    c.line([(165, 347), (285, 347), (285, 185), (1450, 185), (1450, 210), (1315, 210), (1315, 230)], dashed=True, sw=2.4)

    c.module(680, 500, 210, 88, "Ràng buộc bag âm", "Lₙₑg", kind="eval", title_size=21, subtitle_size=19)
    c.module(950, 500, 210, 88, "Nhất quán lật ngang", "L꜀ₒₙₛ", kind="eval", title_size=21, subtitle_size=19)
    c.line([(785, 500), (785, 420), (1300, 420), (1300, 380)], dashed=True, sw=2.2)
    c.line([(1055, 500), (1055, 445), (1320, 445), (1320, 380)], dashed=True, sw=2.2)
    c.module(1200, 500, 360, 88, "Mục tiêu huấn luyện", "L = Lbag + 0,25Lneg + 0,10Lcons", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.line([(1410, 380), (1410, 500)], dashed=True, sw=2.2)
    return c.save()


def figure_36(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1680, 650, "fig_3_6_rank_fusion_final_selection")
    c.module(40, 70, 200, 80, "Nhãn cấp ảnh", "thiết lập chính", kind="data", title_size=23, subtitle_size=18)
    c.module(40, 185, 200, 75, "Bình thường", "Mặt nạ rỗng", kind="eval", title_size=22, subtitle_size=18)
    c.module(40, 300, 200, 75, "Khối u", "Xếp hạng ứng viên", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.line([(140, 150), (140, 185)])
    c.line([(140, 150), (270, 150), (270, 337), (240, 337)])
    c.text(42, 420, "Triển khai không nhãn:", size=18, weight=700, anchor="start", fill=COLORS["eval_border"])
    c.text(42, 447, "có thể thay bằng lớp dự đoán", size=17, anchor="start", fill=COLORS["eval_border"])

    c.module(330, 45, 210, 85, "Logit G1 — ảnh gốc", kind="feature", title_size=20)
    c.module(330, 155, 210, 85, "Logit G1 — ảnh lật", kind="feature", title_size=20)
    c.module(590, 88, 190, 95, "Lấy trung bình", "Hạng phân vị G1", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.line([(540, 88), (565, 88), (565, 135), (590, 135)])
    c.line([(540, 198), (565, 198), (565, 135), (590, 135)])
    c.module(380, 305, 210, 90, "Điểm nguồn (sˢʳᶜ)", kind="feature", title_size=21)
    c.module(650, 305, 185, 90, "Hạng phân vị", "nguồn", kind="logic", designed=True, title_size=22, subtitle_size=18)
    c.line([(590, 350), (650, 350)])

    c.module(900, 175, 230, 135, "Hợp nhất thứ hạng", "0,5 G1 + 0,5 nguồn", kind="logic", designed=True, title_size=24, subtitle_size=20)
    c.line([(780, 135), (860, 135), (860, 242), (900, 242)])
    c.line([(835, 350), (860, 350), (860, 242), (900, 242)])
    c.module(1190, 168, 220, 150, "Chọn hạng cao nhất", ["quy tắc đồng hạng cố định", "1. Điểm hợp nhất", "2. Logit G1 · 3. Chỉ số"], kind="logic", designed=True, title_size=20, subtitle_size=17)
    c.line([(1130, 242), (1190, 242)])

    c.module(500, 500, 200, 80, "Tập mặt nạ (Mᵢ)", kind="data", title_size=22)
    c.module(930, 495, 215, 90, "Lấy mặt nạ", "tại j*", kind="output", title_size=22, subtitle_size=20)
    c.line([(700, 540), (930, 540)])
    c.line([(1300, 318), (1300, 540), (1145, 540)])

    for idx in range(3):
        c.image(a[f"IMG000415_candidate_{idx + 1:02d}"], 1440 + idx * 72, 45, 64, 64, rx=9)
        c.text(1472 + idx * 72, 125, f"#{idx + 1}", size=17, weight=700)
    c.text(1512, 160, "Top-3 ứng viên", size=19, weight=700)
    c.line([(1410, 242), (1512, 242), (1512, 175)])
    c.image(a["IMG000415_selected_mask"], 1432, 345, 170, 170, stroke=COLORS["output_border"])
    c.text(1517, 540, "Mặt nạ cuối", size=23, weight=700, fill=COLORS["output_border"])
    c.line([(1145, 540), (1380, 540), (1380, 430), (1432, 430)])
    return c.save()


def figure_4x(a: dict[str, Path]) -> Path:
    c = SvgCanvas(1600, 520, "fig_4_x_prediction_lock_evaluation")
    labels = [
        (35, "Ảnh + nhãn", "cấp ảnh", "data"),
        (245, "WSSS pipeline", None, "logic"),
        (455, "Ứng viên + điểm", None, "feature"),
        (665, "Mặt nạ được chọn", None, "output"),
        (875, "Lưu kết quả", "+ hash", "eval"),
        (1085, "KHÓA DỰ ĐOÁN", None, "logic"),
    ]
    for x, title, sub, kind in labels:
        long_title = title in {"Ứng viên + điểm", "Mặt nạ được chọn"}
        c.module(x, 80, 175, 95, title, sub, kind=kind, designed=kind == "logic", title_size=18 if long_title else 21, subtitle_size=18)
    for idx in range(len(labels) - 1):
        c.line([(labels[idx][0] + 175, 127), (labels[idx + 1][0], 127)])

    c.module(1280, 55, 270, 145, "Mặt nạ chuẩn", "Chỉ dùng sau khi khóa dự đoán", kind="eval", title_size=24, subtitle_size=18, dashed=True)
    c.line([(1260, 127), (1280, 127)], dashed=True, sw=2.4)

    c.module(760, 315, 285, 115, "Đánh giá", "Dice · IoU · Precision · Recall", kind="eval", title_size=24, subtitle_size=18, dashed=True)
    c.module(1110, 315, 330, 115, "Phân tích hậu nghiệm", "Oracle Dice · Selector Regret", kind="eval", title_size=24, subtitle_size=18, dashed=True)
    c.line([(1415, 200), (1415, 265), (902, 265), (902, 315)], dashed=True, sw=2.4)
    c.line([(1415, 265), (1275, 265), (1275, 315)], dashed=True, sw=2.4)
    c.text(50, 468, "Mặt nạ chuẩn không tham gia định vị, tạo prompt, sinh ứng viên hoặc lựa chọn.", size=22, weight=700, anchor="start", fill=COLORS["eval_border"])
    return c.save()


def write_captions() -> None:
    captions = r"""% Captions for the Rich Gallery G1 figure set
\newcommand{\CaptionRichOverview}{Tổng quan Rich Gallery G1 trong hai pha offline và online. Pha offline huấn luyện hai bộ phân loại DenseNet-121 và bộ chấm điểm G1 từ nhãn cấp ảnh; pha online khai thác ba nguồn bằng chứng định vị, sinh tập mặt nạ ứng viên và lựa chọn một mặt nạ cuối bằng hợp nhất thứ hạng G1--nguồn.}
\newcommand{\CaptionRichLocalization}{Trích xuất ba nguồn bằng chứng định vị. Hai bộ phân loại DenseNet-121 tạo LayerCAM ở độ phân giải 320 và 448, trong khi BiomedCLIP cung cấp saliency map từ biểu diễn ảnh--văn bản. Ba bản đồ được giữ riêng để bảo toàn tính bổ sung giữa các nguồn.}
\newcommand{\CaptionRichGallery}{Quy trình sinh Candidate Gallery từ ba nguồn định vị. Mỗi bản đồ được cắt tại ba mức phân vị, tách thành phần liên thông và chuyển thành prompt điểm, hộp hoặc kết hợp để SAM ViT-B sinh nhiều giả thuyết mặt nạ. Các ứng viên được quy về cùng lưới, giữ metadata nguồn và loại các mặt nạ trùng hoàn toàn trước khi hình thành $M_i$.}
\newcommand{\CaptionRichRepresentation}{Biểu diễn và chấm điểm một ứng viên. RAD-DINO đóng băng cung cấp đặc trưng tại ba tầng; mỗi ứng viên được mô tả bởi thống kê trong vùng, ngữ cảnh lân cận và độ tương phản, sau đó ghép với bốn đặc trưng bổ sung thành vector 1156 chiều. G1 ánh xạ vector này thành một logit dùng để xếp hạng ứng viên.}
\newcommand{\CaptionRichMIL}{Huấn luyện G1 bằng multiple-instance learning. Các mặt nạ của cùng một ảnh tạo thành một bag mà không có nhãn ở mức ứng viên. G1 sinh logit cho từng ứng viên; smooth pooling tổng hợp chúng thành logit cấp bag để tối ưu theo nhãn khối-u--bình-thường ở mức ảnh cùng hai ràng buộc bổ sung.}
\newcommand{\CaptionRichFusion}{Hợp nhất thứ hạng và lựa chọn mặt nạ cuối. Logit G1 từ ảnh gốc và ảnh lật được lấy trung bình, sau đó được chuyển cùng điểm nguồn sang hạng phân vị trong từng ảnh. Hai thứ hạng được kết hợp với trọng số bằng nhau để xác định chỉ số ứng viên cuối theo quy tắc đồng hạng cố định.}
\newcommand{\CaptionRichEvaluation}{Giao thức khóa dự đoán trước khi truy cập mặt nạ chuẩn. Mặt nạ chuẩn chỉ được mở sau khi đầu ra và hash đã được cố định; các chỉ số Oracle Dice và Selector Regret chỉ phục vụ phân tích hậu nghiệm.}
"""
    (OUT / "captions_vi.tex").write_text(captions, encoding="utf-8")


def write_readme(svg_paths: list[Path]) -> None:
    lines = [
        "# Rich Gallery G1 figure set",
        "",
        "Các sơ đồ được dựng độc lập theo visual system mauve–olive–ochre–clay–teal, chữ Arial, nền trắng, không shadow/gradient.",
        "",
        "## Deliverables",
        "",
    ]
    for path in svg_paths:
        lines.append(f"- `svg/{path.name}`")
    lines += [
        "",
        "## Asset choice",
        "",
        "IMG000415 được dùng nhất quán cho các X-ray, heatmap/prompt và candidate masks. IMG001215 có ảnh X-ray local nhưng không có candidate diagnostics/top-3 masks trong local hoặc Prism đã cung cấp; không trộn dữ liệu giữa hai ca.",
        "",
        "LayerCAM-448 dùng map tái dựng local. LayerCAM-320 dùng prompt map local pha với tái dựng theo nguồn. BiomedCLIP dùng tái dựng theo các candidate có nguồn BiomedCLIP. Chi tiết nằm trong `asset_manifest.json`.",
        "",
        "`captions_vi.tex` chứa caption tiếng Việt để chèn vào LaTeX.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_print_wrappers(svg_paths: list[Path]) -> None:
    for svg_path in svg_paths:
        head = svg_path.read_text(encoding="utf-8")[:300]
        match = re.search(r'<svg[^>]+width="(\d+)"[^>]+height="(\d+)"', head)
        if not match:
            raise ValueError(f"Cannot read SVG dimensions: {svg_path}")
        width, height = (int(v) for v in match.groups())
        width_mm = 160.0
        height_mm = height / width * width_mm
        uri = svg_path.resolve().as_uri()
        html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><style>
@page {{ size: {width_mm:.1f}mm {height_mm:.1f}mm; margin: 0; }}
html, body {{ margin: 0; padding: 0; width: {width_mm:.1f}mm; height: {height_mm:.1f}mm; overflow: hidden; background: white; }}
img {{ display: block; width: {width_mm:.1f}mm; height: {height_mm:.1f}mm; }}
</style></head><body><img src=\"{uri}\"></body></html>"""
        (WRAPPER_DIR / f"{svg_path.stem}.html").write_text(html, encoding="utf-8")


def main() -> None:
    assets = extract_assets()
    svg_paths = [
        figure_31(assets),
        figure_33(assets),
        figure_34(assets),
        figure_36(assets),
        figure_32(assets),
        figure_35(assets),
        figure_4x(assets),
    ]
    write_captions()
    write_readme(svg_paths)
    write_print_wrappers(svg_paths)
    print(json.dumps({"svg": [str(p) for p in svg_paths], "assets": len(assets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
