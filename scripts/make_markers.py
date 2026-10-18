"""Print-ready ArUco markers (B3).

Writes one PNG per marker at 300 dpi into runs/markers/ (gitignored) or `--out DIR`: ids 0-7 at 6 cm
(table calibration), the object ids of config/objects.yaml at 3 cm, the obstacle ids at 4 cm, 40 (the
wristband) at 4 cm and, for I2 (five loose markers on the table), spare object-size cards at 3 cm labelled
"spare, I2 check": ids 13-17 by default (`default_spare_ids`: an id config/objects.yaml assigns is skipped
and the next free one up to 19 takes its place, with a note), or the unconfigured ids in 10-19 that
`--spare-ids ID...` names (the flag alone prints none; another id is a usage error, exit 2). Each card has
a white border at least one marker bit wide with "id N  S cm  name" printed under it, plus A4 sheets
(`a4_sheet.png`, then `a4_sheet_2.png`, ...) that tile every card with thin gray cut lines, and
`markers.json` listing what was written (every entry carries `spare`, true for the I2 cards); card or sheet
PNGs of an earlier run in the same directory that markers.json does not list are named in a note
(`stale_files`), never deleted. The PNGs carry a 300 dpi pHYs chunk so viewers and printers know the
physical size. Print at 100 % scale (never "fit to page"), measure the printed side with a ruler, cut along
the gray lines and glue each marker flat keeping its white border; the spare cards stay loose for
`scripts/touch_markers.py --loose`. That I2 check targets only the spare cards: it leaves out every id
config/objects.yaml assigns, because those cards are glued on top of objects standing on the table and a
hover planned from the table there would drive the tip into the object (`--object-cards-flat` includes them
only while they lie flat on the table with the objects taken away).

    python scripts/make_markers.py [--out DIR] [--spare-ids ID ...]
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
# The object-size ids of vision.markers (3 cm): what scripts/touch_markers.py --loose looks for.
OBJECT_IDS = range(10, 20)
# I2's spare cards, printed by default: five, so five loose markers exist with every object card glued.
SPARE_IDS = (13, 14, 15, 16, 17)
SPARE_NAME = "spare, I2 check"


def px(cm: float) -> int:
    """Pixels for a length in centimeters at DPI."""
    return int(round(cm / 2.54 * DPI))


def spare_specs(
    spare_ids: Sequence[int], objects: Mapping[int, Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Print specs for the spare object-size cards of I2: 3 cm, named "spare, I2 check", `spare: True`.

    ValueError for an id outside 10-19 (not object-size: `touch_markers.py --loose` would never look for
    it), one that config/objects.yaml (or `objects`) already assigns to an object, or a repeated id.
    """
    table = objects if objects is not None else load_objects()
    specs: list[dict[str, Any]] = []
    for raw in spare_ids:
        marker_id = int(raw)
        if marker_id not in OBJECT_IDS:
            raise ValueError(f"spare id {marker_id} is not an object-size id (10-19)")
        if marker_id in table:
            raise ValueError(
                f"spare id {marker_id} is already {table[marker_id]['name']} in config/objects.yaml"
            )
        if any(spec["id"] == marker_id for spec in specs):
            raise ValueError(f"spare id {marker_id} is given twice")
        specs.append(
            {
                "id": marker_id,
                "side_cm": MARKER_SIDE_M["object"] * 100,
                "name": SPARE_NAME,
                "kind": "object",
                "spare": True,
            }
        )
    return specs


def default_spare_ids(objects: Mapping[int, Mapping[str, Any]] | None = None) -> list[int]:
    """The spare cards printed when `--spare-ids` is not given: SPARE_IDS (13-17), except that an id
    config/objects.yaml (or `objects`) assigns is skipped and the next free object-size id up to 19 takes
    its place, so the default run keeps printing five spare cards after A11 adds objects (fewer only when
    fewer than five of 13-19 are free)."""
    table = objects if objects is not None else load_objects()
    free = [i for i in range(SPARE_IDS[0], OBJECT_IDS.stop) if i not in table]
    return free[: len(SPARE_IDS)]


def marker_specs(
    objects: Mapping[int, Mapping[str, Any]] | None = None, spare_ids: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """What to print: `[{"id", "side_cm", "name", "kind", "spare"}]`, calibration ids first, then
    objects.yaml (and the wristband), then the spare cards of `spare_ids` (see `spare_specs`; None means
    `default_spare_ids`)."""
    table = objects if objects is not None else load_objects()
    if spare_ids is None:
        spare_ids = default_spare_ids(table)
    specs: list[dict[str, Any]] = [
        {
            "id": i,
            "side_cm": MARKER_SIDE_M["calib"] * 100,
            "name": f"calibration {i}",
            "kind": "calib",
            "spare": False,
        }
        for i in CALIB_IDS
    ]
    hand_seen = False
    for marker_id, entry in sorted(table.items()):
        kind = entry["kind"]
        side_cm = MARKER_SIDE_M[kind] * 100
        specs.append(
            {
                "id": int(marker_id),
                "side_cm": side_cm,
                "name": str(entry["name"]),
                "kind": kind,
                "spare": False,
            }
        )
        hand_seen = hand_seen or int(marker_id) == HAND_ID
    if not hand_seen:
        specs.append(
            {
                "id": HAND_ID,
                "side_cm": MARKER_SIDE_M["hand"] * 100,
                "name": "hand",
                "kind": "hand",
                "spare": False,
            }
        )
    specs.extend(spare_specs(spare_ids, table))
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


def stale_files(out: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Card and sheet PNGs in `out` that `manifest` does not list: leftovers of an earlier run with more
    cards or sheets (other spare ids, a longer object list), which would otherwise be printed as well."""
    written = {str(entry["file"]) for entry in manifest["markers"]} | {str(s) for s in manifest["sheets"]}
    found = {p.name for pattern in ("marker_*.png", "a4_sheet*.png") for p in out.glob(pattern)}
    return sorted(found - written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write print-ready ArUco marker PNGs (B3).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default runs/markers)")
    parser.add_argument(
        "--spare-ids",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help="spare object-size cards for I2's loose markers: ids 10-19 not in config/objects.yaml "
        "(default 13 14 15 16 17, an id objects.yaml assigns replaced by the next free one; the flag "
        "with no ids prints none)",
    )
    args = parser.parse_args(argv)
    table = load_objects()
    spare_ids = default_spare_ids(table) if args.spare_ids is None else args.spare_ids
    if args.spare_ids is None and tuple(spare_ids) != SPARE_IDS:
        print(
            f"note: config/objects.yaml assigns some of {SPARE_IDS[0]}-{SPARE_IDS[-1]}; the spare cards "
            f"are {' '.join(str(i) for i in spare_ids) or 'none'}"
        )
    try:
        specs = marker_specs(table, spare_ids=spare_ids)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"dpi": DPI, "markers": [], "sheets": []}
    cards: list[np.ndarray] = []
    for spec in specs:
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
    spares = sum(1 for spec in specs if spec["spare"])
    print(f"wrote {len(cards)} markers ({spares} spare) and {len(manifest['sheets'])} sheet(s) to {out}")
    if spares:
        print(
            "I2: lay the spare cards flat on the table; touch_markers.py --loose leaves out the object ids "
            "(glued on objects), so the spares are its targets"
        )
    stale = stale_files(out, manifest)
    if stale:
        print(
            f"note: {out} also holds {len(stale)} file(s) of an earlier run that markers.json does not "
            f"list; delete them before printing: {' '.join(stale)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
