"""ArUco markers: dictionary, detector, detection, the marker table and overlay drawing (B2, B3, B6).

Everything uses `DICT_4X4_50`. Ids: 0-7 are the table calibration markers (6 cm), 10-19 sit on top of
objects (3 cm), 30-34 on obstacles (4 cm) and 40 is the wristband (4 cm). `marker_table()` merges
config/objects.yaml with the calibration ids (whose tape-measured centers live in config/scene.yaml), so
every id the overhead camera may see maps to exactly one entry with a `kind`, a `name` and the printed
marker side in meters.

`detect(frame)` returns `[(id, corners)]` with `corners` a 4x2 float32 array in the ArUco order: corner 0
is the marker's top-left as printed, then clockwise as seen in the image. The scene tracker reads the
yaw from corner 0 -> corner 1, so a marker is glued with that top edge pointing along the object's length.
"""

from __future__ import annotations

import math
import pathlib
import threading
from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np
import yaml

__all__ = [
    "BORDER_BITS",
    "CALIB_IDS",
    "DICT",
    "HAND_H",
    "HAND_ID",
    "MARKER_SIDE_M",
    "detect",
    "detector",
    "detector_params",
    "dictionary",
    "draw",
    "generate",
    "load_objects",
    "marker_side",
    "marker_table",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBJECTS_PATH = ROOT / "config" / "objects.yaml"
SCENE_PATH = ROOT / "config" / "scene.yaml"

DICT = cv2.aruco.DICT_4X4_50
CALIB_IDS = tuple(range(8))
HAND_ID = 40
HAND_H = 0.03  # the wristband height (meters) the contract fixes for the hand's z
BORDER_BITS = 1
# Printed marker side (meters) per kind: B3 prints them at these sizes and the FakeCamera renders them so.
MARKER_SIDE_M: dict[str, float] = {"calib": 0.06, "object": 0.03, "obstacle": 0.04, "hand": 0.04}

_LOCK = threading.RLock()
_DICTIONARY: Any = None
_DETECTOR: Any = None


def dictionary() -> Any:
    """The shared `cv2.aruco.Dictionary` for DICT_4X4_50 (created once)."""
    global _DICTIONARY
    with _LOCK:
        if _DICTIONARY is None:
            _DICTIONARY = cv2.aruco.getPredefinedDictionary(DICT)
        return _DICTIONARY


def detector_params() -> Any:
    """`cv2.aruco.DetectorParameters` tuned for 20-50 px markers in a 640x480 frame.

    Sub-pixel corner refinement is what makes 1 cm accuracy possible from 0.65 m: one pixel is about
    1.4 mm on the table, and the yaw comes from one marker edge of 20-40 px.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.05
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 33
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02  # a 3 cm marker 0.6 m away is ~22 px wide: perimeter 88 of 640
    params.markerBorderBits = BORDER_BITS
    params.detectInvertedMarker = False
    return params


def detector(params: Any | None = None) -> Any:
    """A new `cv2.aruco.ArucoDetector` (one per tracker thread; `detect()` keeps a module-wide one)."""
    return cv2.aruco.ArucoDetector(dictionary(), params if params is not None else detector_params())


def _default_detector() -> Any:
    global _DETECTOR
    with _LOCK:
        if _DETECTOR is None:
            _DETECTOR = detector()
        return _DETECTOR


def detect(frame: np.ndarray | None, det: Any | None = None) -> list[tuple[int, np.ndarray]]:
    """Detect markers in a BGR or grayscale frame: `[(id, corners 4x2 float32)]` sorted by id.

    Duplicate ids (two prints of the same marker in view) keep the larger one. An empty list when
    nothing is found; a frame of `None` or zero size also yields an empty list.
    """
    if frame is None or frame.size == 0:
        return []
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = (det if det is not None else _default_detector()).detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []
    found: dict[int, np.ndarray] = {}
    for marker_id, quad in zip(ids.ravel().tolist(), corners, strict=True):
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        previous = found.get(int(marker_id))
        if previous is None or cv2.contourArea(pts) > cv2.contourArea(previous):
            found[int(marker_id)] = pts
    return sorted(found.items())


def generate(marker_id: int, side_px: int, border_bits: int = BORDER_BITS) -> np.ndarray:
    """The marker image (uint8, side_px x side_px, black bits 0 / white 255) from generateImageMarker."""
    if not 0 <= int(marker_id) < 50:
        raise ValueError(f"DICT_4X4_50 has ids 0-49, got {marker_id}")
    if int(side_px) < 6 * (1 + 2 * border_bits):
        raise ValueError(f"side_px {side_px} is too small for a 4x4 marker with {border_bits} border bits")
    return cv2.aruco.generateImageMarker(dictionary(), int(marker_id), int(side_px), borderBits=border_bits)


def _positive(file: pathlib.Path, name: str, field: str, value: Any, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{file}: {name} {field} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{file}: {name} {field} must be {bound}, got {value!r}")
    return number


def load_objects(path: pathlib.Path | str | None = None) -> dict[int, dict[str, Any]]:
    """config/objects.yaml as `{marker id: entry}` with the printed marker `side` added per kind.

    Objects and obstacles need `h` > 0 and `size` [length > 0, width > 0] (schemas/scene.json requires
    them, and the tracker would otherwise publish nothing); the hand's `h` defaults to HAND_H.
    """
    file = pathlib.Path(path) if path is not None else OBJECTS_PATH
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: expected a mapping of marker id -> entry")
    table: dict[int, dict[str, Any]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "name" not in entry or "kind" not in entry:
            raise ValueError(f"{file}: entry {key!r} needs name and kind")
        kind = str(entry["kind"])
        if kind not in ("object", "obstacle", "hand"):
            raise ValueError(f"{file}: entry {key!r} has unknown kind {kind!r}")
        item = dict(entry)
        item["name"] = str(entry["name"])
        item["kind"] = kind
        item["side"] = MARKER_SIDE_M[kind]
        if kind == "hand":
            item["h"] = _positive(file, item["name"], "h", entry.get("h", HAND_H), allow_zero=True)
        else:
            if "h" not in entry:
                raise ValueError(f"{file}: {item['name']} needs h (height in meters)")
            item["h"] = _positive(file, item["name"], "h", entry["h"])
            size = entry.get("size")
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise ValueError(f"{file}: {item['name']} needs size [length, width]")
            item["size"] = [_positive(file, item["name"], "size", v) for v in size]
        table[int(key)] = item
    return table


def marker_table(
    objects_path: pathlib.Path | str | None = None, scene_path: pathlib.Path | str | None = None
) -> dict[int, dict[str, Any]]:
    """Every marker id the camera may see: objects.yaml entries plus `calib_0`..`calib_7`.

    Calibration entries look like `{"name": "calib_3", "kind": "calib", "h": 0.0, "side": 0.06,
    "xy": [x, y]}` with the tape-measured center from config/scene.yaml `calib_markers`.
    """
    table = load_objects(objects_path)
    scene_file = pathlib.Path(scene_path) if scene_path is not None else SCENE_PATH
    cfg = yaml.safe_load(scene_file.read_text(encoding="utf-8")) or {}
    centers = cfg.get("calib_markers") or {}
    for marker_id in CALIB_IDS:
        if marker_id in table:
            raise ValueError(f"marker id {marker_id} is reserved for calibration but appears in objects.yaml")
        xy = centers.get(marker_id)
        entry: dict[str, Any] = {
            "name": f"calib_{marker_id}",
            "kind": "calib",
            "h": 0.0,
            "side": MARKER_SIDE_M["calib"],
        }
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            entry["xy"] = [float(xy[0]), float(xy[1])]
        table[marker_id] = entry
    return dict(sorted(table.items()))


def marker_side(marker_id: int, table: Mapping[int, Mapping[str, Any]] | None = None) -> float:
    """Printed side of a marker in meters, from `marker_table()` (or the kind defaults for unknown ids)."""
    entry = (table if table is not None else marker_table()).get(int(marker_id))
    if entry is not None:
        return float(entry["side"])
    if int(marker_id) in CALIB_IDS:
        return MARKER_SIDE_M["calib"]
    if int(marker_id) == HAND_ID:
        return MARKER_SIDE_M["hand"]
    return MARKER_SIDE_M["obstacle"] if 30 <= int(marker_id) < 40 else MARKER_SIDE_M["object"]


def draw(
    frame: np.ndarray,
    dets: Iterable[tuple[int, np.ndarray]],
    labels: Mapping[int, str] | None = None,
    *,
    color: tuple[int, int, int] = (0, 220, 0),
) -> np.ndarray:
    """Draw each detection in place: the outline, a red dot on corner 0 and the id (or `labels[id]`)."""
    for marker_id, corners in dets:
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)
        c0 = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        cv2.circle(frame, c0, 3, (0, 0, 255), -1, cv2.LINE_AA)
        text = labels.get(int(marker_id), str(marker_id)) if labels else str(marker_id)
        cx, cy = pts.mean(axis=0)
        anchor = (int(round(cx)) + 6, int(round(cy)) - 6)
        cv2.putText(frame, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
