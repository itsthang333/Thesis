from __future__ import annotations

"""Render dependency-free loss/Dice training curves from a CSV log."""

import argparse
import csv
import html
from pathlib import Path


COLORS = {
    "train_loss": "#2563eb",
    "val_loss": "#dc2626",
    "train_positive_dice": "#0f766e",
    "val_positive_dice": "#d97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Training curves")
    parser.add_argument("--selected-epoch", type=int, default=None)
    return parser.parse_args()


def points(
    rows: list[dict[str, str]],
    key: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> str:
    epochs = [float(row["epoch"]) for row in rows]
    values = [float(row[key]) for row in rows]
    x_min, x_max = min(epochs), max(epochs)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-12)
    coordinates = []
    for epoch, value in zip(epochs, values):
        x = left + (epoch - x_min) / x_span * width
        y = top + height - (value - y_min) / y_span * height
        coordinates.append(f"{x:.2f},{y:.2f}")
    return " ".join(coordinates)


def main() -> None:
    args = parse_args()
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training log: {args.input}")
    required = {"epoch", *COLORS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Training log lacks columns: {sorted(missing)}")

    width, height = 1000, 680
    left, plot_width = 90, 850
    panel_height = 215
    panels = [
        (90, "Loss", ("train_loss", "val_loss")),
        (390, "Positive-image Dice", ("train_positive_dice", "val_positive_dice")),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="500" y="38" text-anchor="middle" font-family="sans-serif" '
        f'font-size="24" font-weight="700">{html.escape(args.title)}</text>',
    ]
    epochs = [float(row["epoch"]) for row in rows]
    epoch_min, epoch_max = min(epochs), max(epochs)
    epoch_span = max(epoch_max - epoch_min, 1.0)
    for top, label, keys in panels:
        values = [float(row[key]) for row in rows for key in keys]
        y_min = min(0.0, min(values))
        y_max = max(values)
        padding = max((y_max - y_min) * 0.08, 0.02)
        y_max += padding
        svg.append(
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" '
            'fill="#f8fafc" stroke="#cbd5e1"/>'
        )
        for tick in range(6):
            ratio = tick / 5
            y = top + panel_height - ratio * panel_height
            value = y_min + ratio * (y_max - y_min)
            svg.extend([
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#e2e8f0"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.2f}</text>',
            ])
        svg.append(
            f'<text x="24" y="{top + panel_height / 2:.2f}" '
            f'transform="rotate(-90 24 {top + panel_height / 2:.2f})" '
            f'text-anchor="middle" font-family="sans-serif" font-size="15">{label}</text>'
        )
        if args.selected_epoch is not None:
            selected_x = left + (
                (args.selected_epoch - epoch_min) / epoch_span * plot_width
            )
            svg.extend([
                f'<line x1="{selected_x:.2f}" y1="{top}" x2="{selected_x:.2f}" '
                f'y2="{top + panel_height}" stroke="#7c3aed" stroke-dasharray="6 5"/>',
                f'<text x="{selected_x + 6:.2f}" y="{top + 16}" '
                'font-family="sans-serif" font-size="12" fill="#7c3aed">'
                f'selected epoch {args.selected_epoch}</text>',
            ])
        for index, key in enumerate(keys):
            svg.append(
                f'<polyline points="{points(rows, key, left=left, top=top, width=plot_width, height=panel_height, y_min=y_min, y_max=y_max)}" '
                f'fill="none" stroke="{COLORS[key]}" stroke-width="2.5"/>'
            )
            legend_x = left + 15 + index * 230
            svg.extend([
                f'<line x1="{legend_x}" y1="{top + panel_height + 28}" '
                f'x2="{legend_x + 34}" y2="{top + panel_height + 28}" '
                f'stroke="{COLORS[key]}" stroke-width="3"/>',
                f'<text x="{legend_x + 42}" y="{top + panel_height + 33}" '
                f'font-family="sans-serif" font-size="13">{html.escape(key)}</text>',
            ])
    svg.extend([
        '<text x="515" y="660" text-anchor="middle" font-family="sans-serif" '
        'font-size="15">Epoch</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
