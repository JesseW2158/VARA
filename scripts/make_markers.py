"""Print-ready ArUco markers (B3).

Writes one PNG per marker at 300 dpi into runs/markers/ (gitignored) or `--out DIR`: ids 0-7 at 6 cm
(table calibration), the object ids of config/objects.yaml at 3 cm, the obstacle ids at 4 cm and 40 (the
wristband) at 4 cm, each inside a white border at least one marker bit wide with "id N  S cm  name" printed
under it, plus A4 sheets (`a4_sheet.png`, then `a4_sheet_2.png`, ...) that tile every card with thin gray cut
lines, and `markers.json` listing what was written. The PNGs carry a 300 dpi pHYs chunk so viewers and
printers know the physical size. Print at 100 % scale (never "fit to page"), measure the printed side with a
ruler, cut along the gray lines and glue each marker flat keeping its white border.

    python scripts/make_markers.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.markers import CALIB_IDS, HAND_ID, MARKER_SIDE_M, generate, load_objects  # noqa: E402

DPI = 300
A4_CM = (21.0, 29.7)
PAGE_MARGIN_CM = 0.5
CARD_GAP_CM = 0.3
LABEL_CM = 0.8
DEFAULT_OUT = ROOT / "runs" / "markers"


def px(cm: float) -> int:
    """Pixels for a length in centimeters at DPI."""
    return int(round(cm / 2.54 * DPI))


def marker_specs(objects: Mapping[int, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """What to print: `[{"id", "side_cm", "name", "kind"}]`, calibration ids first, then objects.yaml."""
    table = objects if objects is not None else load_objects()
    specs = [
        {"id": i, "side_cm": MARKER_SIDE_M["calib"] * 100, "name": f"calibration {i}", "kind": "calib"}
        for i in CALIB_IDS
    ]
    hand_seen = False
    for marker_id, entry in sorted(table.items()):
        kind = entry["kind"]
        side_cm = MARKER_SIDE_M[kind] * 100
        specs.append({"id": int(marker_id), "side_cm": side_cm, "name": str(entry["name"]), "kind": kind})
        hand_seen = hand_seen or int(marker_id) == HAND_ID
    if not hand_seen:
        specs.append({"id": HAND_ID, "side_cm": MARKER_SIDE_M["hand"] * 100, "name": "hand", "kind": "hand"})
    return specs


def border_cm(side_cm: float) -> float:
    """White quiet zone around a marker: one marker bit (side / 6) and at least 0.5 cm."""
    return max(0.5, side_cm / 6.0)


def render_card(spec: Mapping[str, Any]) -> np.ndarray:
    """A white card: the marker at its printed size, the border, and the label under it (BGR uint8)."""
    side_cm = float(spec["side_cm"])
    side = px(side_cm)
    border = px(border_cm(side_cm))
    width = side + 2 * border
    height = width + px(LABEL_CM)
    card = np.full((height, width, 3), 255, np.uint8)
    marker = generate(int(spec["id"]), side)
    card[border : border + side, border : border + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    label = f"id {spec['id']}  {side_cm:g} cm  {spec['name']}"
    scale = 1.1
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    while text_w > width - 2 * border and scale > 0.4:
        scale -= 0.1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    origin = ((width - text_w) // 2, width + (px(LABEL_CM) + text_h) // 2)
    cv2.putText(card, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return card


def png_bytes(image: np.ndarray, dpi: int = DPI) -> bytes:
    """PNG-encode an image and insert a pHYs chunk after IHDR so the file states its print size."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = buf.tobytes()
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, IHDR length, type, 13 data bytes, crc
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise RuntimeError("unexpected PNG layout")
    ppm = int(round(dpi / 0.0254))
    body = struct.pack(">IIB", ppm, ppm, 1)
    chunk = (
        struct.pack(">I", len(body))
        + b"pHYs"
        + body
        + struct.pack(">I", zlib.crc32(b"pHYs" + body) & 0xFFFFFFFF)
    )
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def write_png(path: Path, image: np.ndarray, dpi: int = DPI) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(image, dpi))
    return path


def make_sheets(cards: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Tile cards onto A4 pages at DPI (shelf packing, tallest first), with gray cut lines."""
    page_w, page_h = px(A4_CM[0]), px(A4_CM[1])
    margin, gap = px(PAGE_MARGIN_CM), px(CARD_GAP_CM)
    usable_w, usable_h = page_w - 2 * margin, page_h - 2 * margin
    pages: list[np.ndarray] = []
    page: np.ndarray | None = None
    x = y = row_h = 0
    for card in sorted(cards, key=lambda c: (-c.shape[0], -c.shape[1])):
        h, w = card.shape[:2]
        if w > usable_w or h > usable_h:
            raise ValueError(f"a {w}x{h} px card does not fit an A4 page")
        if page is not None and x + w > usable_w:
            x, y, row_h = 0, y + row_h + gap, 0
        if page is None or y + h > usable_h:
            page = np.full((page_h, page_w, 3), 255, np.uint8)
            pages.append(page)
            x = y = row_h = 0
        top, left = margin + y, margin + x
        page[top : top + h, left : left + w] = card
        cv2.rectangle(page, (left, top), (left + w - 1, top + h - 1), (160, 160, 160), 1)
        x += w + gap
        row_h = max(row_h, h)
    return pages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write print-ready ArUco marker PNGs (B3).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default runs/markers)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"dpi": DPI, "markers": [], "sheets": []}
    cards: list[np.ndarray] = []
    for spec in marker_specs():
        card = render_card(spec)
        cards.append(card)
        file = write_png(out / f"marker_{spec['id']:02d}.png", card)
        manifest["markers"].append({**spec, "file": file.name, "border_cm": border_cm(spec["side_cm"])})
        print(f"{file.name}: id {spec['id']} {spec['side_cm']:g} cm {spec['name']}")
    for index, page in enumerate(make_sheets(cards), start=1):
        name = "a4_sheet.png" if index == 1 else f"a4_sheet_{index}.png"
        write_png(out / name, page)
        manifest["sheets"].append(name)
        print(f"{name}: A4 at {DPI} dpi")
    (out / "markers.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(cards)} markers and {len(manifest['sheets'])} sheet(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
