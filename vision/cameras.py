"""Cameras (B1, B8): the real USB captures, a FakeCamera that renders config/fake_scene.json, and the camera
tools `python -m vision.cameras --check | --diagnose | --list`.

`open_camera(index)` opens a capture at 640x480, 30 fps, MJPG (two such streams fit one USB bus) with
`cv2.CAP_DSHOW` on Windows and autofocus / auto-exposure switched off where the driver allows it, because
exposure flicker moves detections. `Camera` wraps it with frame-rate and drop statistics and keeps the
driver's read-back (`read_back(cap)`: width, height, fps, decoded FOURCC) as `Camera.reported`. Indices come
from config/cameras.yaml, overridden by the environment (`OVERHEAD_CAM`, `WRIST_CAM`); on the demo laptop
index 0 is the built-in webcam, so set the variables in .env.

`FakeCamera` draws the fake scene as the synthetic camera of vision/calibrate.py would see it: the table,
the tray ring, each object's and obstacle's top face, the 8 calibration markers at their config/scene.yaml
positions on the table, each object/obstacle marker at its config/fake_scene.json pose and yaw projected
from the top of the object (height h, so the pipeline's height correction is exercised), and the wristband
marker at z = 0.03 when the hand is `seen`. Things are painted in the order the camera sees them stacked:
the table markers first, then every item from the lowest to the tallest (its top face, then its marker),
the wristband last, so a raised marker hides the calibration marker under it instead of being hidden by
it. Markers are real ArUco images from `generateImageMarker`, warped with `cv2.warpPerspective` by the
per-marker projection, with a white quiet zone one bit wide. Corner 0
(the marker's printed top-left) is placed so that the edge corner 0 -> corner 1 points along the item's
yaw, which is how B3 glues them and how vision.scene reads the yaw back. `set_scene(scene)` moves things.

`get_camera(name)` honors FAKE_CAMERA / FAKE and the env overrides.

Camera tools (B1). Each writes its evidence to a fresh timestamped path under runs/setup/ (`VARA_RUNS_DIR`
moves runs/) and never overwrites; an explicit `--out` that already exists is refused (exit 2) before any
camera opens. None of them touches the arm. With FAKE_CAMERA=1 (or FAKE=1) they run on the FakeCameras,
open no device, refuse nothing and label everything synthetic (`"fake": true`, `"mode": "synthetic"`,
`"index": null`, a `synthetic` infix in the default names such as camera-check-synthetic-<ts>.json,
SYNTHETIC on the console and on the saved JPEGs).

- `--check SECONDS [--camera overhead|wrist|both] [--criterion strict|tracking] [--out FILE]`:
  `check_feeds()` reads every requested feed in one loop for SECONDS, timing each `read()` itself, and
  writes the report (default runs/setup/camera-check-<ts>.json) with the exit code. One loop means a stall
  in one feed also delays the other feed's next read, so both show the late gap (the 2026-10-17 soak log
  had identical late counts on both feeds); `max_read_s` tells which feed's read stalled.
- `--diagnose SECONDS` (also spelled `--check SECONDS --independent`), same options: the same report from
  `check_feeds(independent=True)`: one reader thread per feed, released together by a Barrier, the way the
  orchestrator reads them (the tracker thread holds the overhead camera, the MJPEG stream the wrist), so a
  late feed is named on its own. A read still blocked JOIN_TIMEOUT_S after the duration ends the run with
  that feed named, and its capture is left to process exit instead of being released under the blocked
  reader. Default runs/setup/camera-diagnose-<ts>.json.
- `--list [--max-index N] [--out DIR]`: `list_cameras()` opens OpenCV indexes 0..N-1 briefly (CAP_DSHOW on
  Windows, the default backend elsewhere), reads one frame each and saves camera-<i>.jpg plus report.json
  (opened, width, height, fps and FOURCC read-back, which configured name points at the index) under
  runs/setup/camera-list-<ts>/, so the human can tell which index is which. Never run it while the
  orchestrator holds the cameras: DirectShow gives a device to one program at a time. A Ctrl+C still
  writes report.json for the indexes done so far (`completed: false`).

Per feed the report records frames, failed reads, late gaps (longer than LATE_FACTOR = 2.5 frame periods),
long gaps (longer than LONG_GAP_S = 0.5 s, vision.scene's seen timeout; at 30 fps every long gap is also
late), effective fps (frames per second between the first and the last good read), the longest single
`read()` (for a paced fake that includes the wait for the frame period), the longest gap between good reads,
and the driver's read-back. The tools time reads with `time.perf_counter()`: on the demo laptop's Python
3.12 `time.monotonic()` is GetTickCount64 with a 15.6 ms tick (the throwaway diagnostics' 0.375, 0.063 and
0.109 s are multiples of it), too coarse against an 83 ms late threshold.

Two criteria judge the numbers; `apply_criterion(report, name)` re-judges a saved report. `strict` (B1's
bar, the default): every feed opens, zero failed reads, zero late gaps, at least 2 frames, the full duration
completed. `tracking`: every feed opens, zero failed reads, zero long gaps, effective fps >= TRACKING_MIN_FPS
= 25 on every feed, the full duration completed: what the 10 Hz tracker and its 0.5 s seen timeout need. The
report states which criterion produced the verdict; exit 0 pass, 1 fail, 2 misuse.

Evidence so far (2026-10-17, demo laptop): before the DirectShow property-order fix both feeds read back
YUY2 instead of MJPG (runs/setup/camera-format-20261017/before.json); the strict 30-minute soak was stopped
after 193.8 s with 27 late gaps on each feed by 185 s, and in a 60 s independent-reader run camera 2
delivered 18.8 fps with 172 late gaps (max read 0.375 s) while the wrist held 30 fps with none. After the
fix (MJPG read back on both; runs/setup/camera-format-20261017/after-independent-readers.json) camera 2
delivered 29.5 fps with about 2 late gaps per minute (max gap 0.11 s) and the wrist 30 fps with 4 late
gaps (max 0.109 s), zero failed reads on either. A strict 30-minute soak may therefore still fail on a
handful of 0.1 s gaps while tracking is unaffected, which is what `--criterion tracking` measures.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import pathlib
import platform
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import yaml

from vision.calibrate import IMAGE_SIZE, camera_pose, load_scene_cfg, project
from vision.markers import BORDER_BITS, HAND_H, HAND_ID, MARKER_SIDE_M, generate, marker_table

__all__ = [
    "CONFIG_PATH",
    "CRITERIA",
    "DEFAULTS",
    "LATE_FACTOR",
    "LONG_GAP_S",
    "TRACKING_MIN_FPS",
    "Camera",
    "FakeCamera",
    "apply_criterion",
    "camera_settings",
    "check_feeds",
    "evaluate_feed",
    "fourcc_name",
    "fresh_path",
    "get_camera",
    "list_cameras",
    "load_config",
    "load_fake_scene",
    "marker_corners_world",
    "open_camera",
    "quiet_zone",
    "read_back",
    "runs_setup_dir",
    "write_report",
]

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "cameras.yaml"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

DEFAULTS: dict[str, Any] = {"index": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
# Optional per-camera image settings in config/cameras.yaml, written by `open_camera` when given:
# `auto_exposure` (true leaves the camera's auto exposure on) and these numbers in driver units.
IMAGE_SETTINGS = ("exposure", "gain", "brightness")
ENV_INDEX = {"overhead": "OVERHEAD_CAM", "wrist": "WRIST_CAM"}
LATE_FACTOR = 2.5  # a gap longer than this many frame periods counts as a dropped frame
LONG_GAP_S = 0.5  # a gap longer than this flips the tracker's `seen` (vision.scene.SEEN_TIMEOUT_S)
TRACKING_MIN_FPS = 25.0  # the tracking criterion's effective frame rate floor per feed
CRITERIA = ("strict", "tracking")
CRITERION_RULES = {
    "strict": (
        "every requested feed opens, zero failed reads, zero gaps longer than "
        f"{LATE_FACTOR} frame periods, at least 2 frames per feed, the full duration completed"
    ),
    "tracking": (
        f"every requested feed opens, zero failed reads, zero gaps longer than {LONG_GAP_S} s (the tracker's "
        f"seen timeout), effective fps >= {TRACKING_MIN_FPS:g} on every feed, the full duration completed"
    ),
}
UNREPORTED: dict[str, Any] = {"width": None, "height": None, "fps": None, "fourcc": "unavailable"}
STAMP = "%Y%m%d-%H%M%S"
LIST_READ_ATTEMPTS = 3  # --list: reads tried per opened index before giving up on a frame
DEFAULT_MAX_INDEX = 8  # --list: indexes 0..7 by default
JOIN_TIMEOUT_S = 5.0  # --diagnose: grace for a reader past the duration, and the join wait after a stop

# FakeCamera colors (BGR)
TABLE_BGR = (178, 190, 200)
TRAY_FLOOR_BGR = (150, 165, 180)
TRAY_WALL_BGR = (90, 110, 140)
OBSTACLE_TOP_BGR = (70, 70, 160)
OBSTACLE_BASE_BGR = (50, 50, 110)
HAND_BGR = (140, 180, 230)
OBJECT_BGR: dict[str, tuple[int, int, int]] = {
    "pill_bottle": (70, 110, 235),
    "juice_box": (210, 150, 60),
    "tea_tin": (80, 170, 90),
}
DEFAULT_OBJECT_BGR = (120, 120, 200)
HAND_SIZE = (0.08, 0.10)  # the hand box of sim/model/build_scene.py: along x, along y

Point3 = tuple[float, float, float]
_Fill = tuple[list[Point3], tuple[int, int, int]]  # polygon in world coordinates, BGR color
_Layer = tuple[float, list[_Fill], tuple[int, list[Point3]]]  # top height, fills, (marker id, corners)


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


def load_config(path: pathlib.Path | str | None = None) -> dict[str, dict[str, Any]]:
    """config/cameras.yaml as `{name: {index, width, height, fps, fourcc, ...}}` (defaults filled in).

    Optional per camera: `auto_exposure` (bool) and `exposure`, `gain`, `brightness` (numbers in driver
    units), passed to `open_camera`; ValueError for a wrong type."""
    file = pathlib.Path(path) if path is not None else CONFIG_PATH
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) if file.exists() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: expected a mapping of camera name -> settings")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{file}: camera {name!r} must be a mapping")
        if not isinstance(entry.get("auto_exposure", False), bool):
            raise ValueError(f"{file}: camera {name!r}: auto_exposure must be true or false")
        for key in IMAGE_SETTINGS:
            value = entry.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
                raise ValueError(f"{file}: camera {name!r}: {key} must be a number (driver units)")
        out[str(name)] = {**DEFAULTS, **entry}
    return out


def camera_settings(name: str, config: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Settings for one camera: its config entry (defaults when unknown) with the env index override."""
    cfg = config if config is not None else load_config()
    settings = dict(cfg.get(name) or DEFAULTS)
    settings.setdefault("index", DEFAULTS["index"])
    env_var = ENV_INDEX.get(name, f"{name.upper()}_CAM")
    value = os.environ.get(env_var)
    if value is not None and value.strip() != "":
        try:
            settings["index"] = int(value)
        except ValueError as exc:
            raise ValueError(f"{env_var}={value!r} is not a camera index") from exc
    settings["index"] = int(settings["index"])
    settings["name"] = name
    return settings


def open_camera(
    index: int,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fourcc: str = "MJPG",
    auto_exposure: bool = False,
    exposure: float | None = None,
    gain: float | None = None,
    brightness: float | None = None,
) -> cv2.VideoCapture:
    """Open a capture (DSHOW on Windows), request MJPG at width x height and fps, disable autofocus.

    Auto exposure is switched off by default (exposure flicker moves marker detections), which keeps
    whatever exposure the camera had; `auto_exposure=True` leaves it on instead, for a camera that only
    shows the scene to people (the wrist camera: config/cameras.yaml). `exposure` (DSHOW: log2 seconds,
    e.g. -5 for 1/32 s), `gain` and `brightness` are written as given when set (driver units; a refused
    value is logged). Raises RuntimeError when the device does not open. Settings the driver refuses are
    logged, not fatal.
    """
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(index))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"camera {index} did not open (check OVERHEAD_CAM/WRIST_CAM and the USB port)")
        # DirectShow recreates its capture graph when FPS or size changes. Set FOURCC last:
        # an earlier MJPG request can otherwise be silently replaced with the default format.
        for prop, value in (
            (cv2.CAP_PROP_FPS, int(fps)),
            (cv2.CAP_PROP_FRAME_WIDTH, int(width)),
            (cv2.CAP_PROP_FRAME_HEIGHT, int(height)),
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
        ):
            if not cap.set(prop, value):
                log.warning("camera %s: requested stream property %s=%s refused", index, prop, value)
        # Exposure flicker and focus hunting move detections; the property semantics differ by backend:
        # DSHOW rounds to a boolean (0 manual / 1 auto); V4L2 defaults to raw 1 manual / 3 auto.
        # Disabling auto exposure retains the current exposure; it does not select a suitable value, so a
        # camera that opened dark stays dark (seen on the wrist camera, 2026-10-17): see `auto_exposure`.
        if auto_exposure:
            exposure_mode = 1 if os.name == "nt" else 3
        else:
            exposure_mode = 0 if os.name == "nt" else 1
        image_settings: list[tuple[int, float]] = [
            (cv2.CAP_PROP_AUTOFOCUS, 0),
            (cv2.CAP_PROP_AUTO_EXPOSURE, exposure_mode),
        ]
        for prop, setting in (
            (cv2.CAP_PROP_EXPOSURE, exposure),
            (cv2.CAP_PROP_GAIN, gain),
            (cv2.CAP_PROP_BRIGHTNESS, brightness),
        ):
            if setting is not None:
                image_settings.append((prop, float(setting)))
        for prop, value in image_settings:
            try:
                if not cap.set(prop, value):
                    log.info("camera %s: property %s not supported", index, prop)
            except cv2.error:  # pragma: no cover - backend-specific
                log.info("camera %s: property %s raised", index, prop)
        got = read_back(cap)
        reported_fourcc = got["fourcc"]
        log.info(
            "camera %s open: %sx%s at %s fps %s requested, driver reports %s (%s)",
            index,
            width,
            height,
            fps,
            fourcc,
            {k: got[k] for k in ("width", "height", "fps")},
            reported_fourcc,
        )
        if reported_fourcc != fourcc:
            log.warning(
                "camera %s: requested %s but driver reports %s; compressed USB bandwidth is unverified",
                index,
                fourcc,
                reported_fourcc,
            )
        return cap
    except BaseException:
        cap.release()
        raise


def fourcc_name(code: float | int | None) -> str:
    """The four characters of a FOURCC property value, or "unavailable" when the backend reports none."""
    if code is None:
        return "unavailable"
    try:
        value = float(code)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(value) or not 0 < value <= 0xFFFFFFFF:
        return "unavailable"
    return "".join(chr((int(value) >> (8 * i)) & 255) for i in range(4))


def _reported_number(value: Any, *, integer: bool) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return int(round(number)) if integer else round(number, 3)


def read_back(cap: Any) -> dict[str, Any]:
    """What the driver reports for an opened capture: `{"width", "height", "fps", "fourcc"}`.

    Width and height are integers, fps a float (None where the backend reports nothing positive), fourcc
    the decoded four characters or "unavailable". A capture without `get` (a test double) reports nothing.
    """
    get = getattr(cap, "get", None)
    if not callable(get):
        return dict(UNREPORTED)
    return {
        "width": _reported_number(get(cv2.CAP_PROP_FRAME_WIDTH), integer=True),
        "height": _reported_number(get(cv2.CAP_PROP_FRAME_HEIGHT), integer=True),
        "fps": _reported_number(get(cv2.CAP_PROP_FPS), integer=False),
        "fourcc": fourcc_name(get(cv2.CAP_PROP_FOURCC)),
    }


class _Stats:
    """Frame counters shared by Camera and FakeCamera."""

    def __init__(self, target_fps: float) -> None:
        self.target_fps = float(target_fps)
        self.frames = 0
        self.failures = 0
        self.late = 0
        self._times: deque[float] = deque(maxlen=120)
        self._last: float | None = None

    def tick(self, now: float) -> None:
        if self._last is not None and now - self._last > LATE_FACTOR / max(self.target_fps, 1e-6):
            self.late += 1
        self._last = now
        self.frames += 1
        self._times.append(now)

    @property
    def fps(self) -> float:
        """Frames per second over the last (up to) 120 frames; 0.0 before the second frame."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "frames": self.frames,
            "failures": self.failures,
            "late": self.late,
            "fps": round(self.fps, 2),
        }


class Camera:
    """A USB camera: `read()` returns a BGR frame or None, `release()` closes it, `fps`/`stats()` report.

    `capture` injects an opened `cv2.VideoCapture`-like object (tests); otherwise `open_camera` runs.
    `reported` is the driver's read-back right after opening (`read_back`), `target_fps` the requested rate.
    """

    is_fake = False

    def __init__(
        self,
        index: int,
        *,
        name: str = "camera",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        capture: Any | None = None,
        auto_exposure: bool = False,
        exposure: float | None = None,
        gain: float | None = None,
        brightness: float | None = None,
    ) -> None:
        self.index = int(index)
        self.name = name
        self.width, self.height = int(width), int(height)
        self._stats = _Stats(fps)
        self._cap = (
            capture
            if capture is not None
            else open_camera(
                index,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                auto_exposure=auto_exposure,
                exposure=exposure,
                gain=gain,
                brightness=brightness,
            )
        )
        self.reported: dict[str, Any] = read_back(self._cap)

    @property
    def target_fps(self) -> float:
        """The requested frame rate (the late-gap threshold is LATE_FACTOR periods of it)."""
        return self._stats.target_fps

    def read(self) -> np.ndarray | None:
        """The next frame (BGR uint8) or None when the grab failed (counted in `failures`)."""
        if self._cap is None:
            self._stats.failures += 1
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self._stats.failures += 1
            return None
        self._stats.tick(time.monotonic())
        return frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    @property
    def fps(self) -> float:
        return self._stats.fps

    @property
    def frames(self) -> int:
        return self._stats.frames

    @property
    def failures(self) -> int:
        return self._stats.failures

    @property
    def late(self) -> int:
        return self._stats.late

    def stats(self) -> dict[str, float | int]:
        return self._stats.stats()


def load_fake_scene(path: pathlib.Path | str | None = None) -> dict[str, Any]:
    """config/fake_scene.json (or `path`)."""
    import json

    file = pathlib.Path(path) if path is not None else FAKE_SCENE_PATH
    return json.loads(file.read_text(encoding="utf-8"))


def marker_corners_world(
    x: float, y: float, z: float, yaw: float, side: float
) -> list[tuple[float, float, float]]:
    """The 4 corners of a marker lying flat at (x, y, z), in ArUco order.

    Corner 0 -> corner 1 (the printed top edge) runs along `yaw`; seen from above the corners go clockwise,
    which is the order the detector returns for an un-mirrored view.
    """
    e1 = (math.cos(yaw), math.sin(yaw))
    e2 = (-math.sin(yaw), math.cos(yaw))
    half = side / 2.0
    offsets = ((-half, half), (half, half), (half, -half), (-half, -half))
    return [(x + a * e1[0] + b * e2[0], y + a * e1[1] + b * e2[1], z) for a, b in offsets]


def _box_corners(
    x: float, y: float, z: float, yaw: float, size: Sequence[float]
) -> list[tuple[float, float, float]]:
    """Footprint corners of a box of `size` [length along yaw, width across] at height z."""
    half_l, half_w = float(size[0]) / 2.0, float(size[1]) / 2.0
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return [
        (x + a * cos_y - b * sin_y, y + a * sin_y + b * cos_y, z)
        for a, b in ((-half_l, half_w), (half_l, half_w), (half_l, -half_w), (-half_l, -half_w))
    ]


def _circle_points(
    x: float, y: float, z: float, radius: float, n: int = 24
) -> list[tuple[float, float, float]]:
    return [
        (x + radius * math.cos(a), y + radius * math.sin(a), z)
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False)
    ]


def quiet_zone(corners: Sequence[Point3]) -> list[Point3]:
    """The printed square around a marker including its one-bit white border (the marker's 6 bits plus
    BORDER_BITS on each side), which is what the FakeCamera pastes and what the detector needs clear."""
    cx = sum(p[0] for p in corners) / len(corners)
    cy = sum(p[1] for p in corners) / len(corners)
    scale = (6.0 + 2.0 * BORDER_BITS) / 6.0
    return [(cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale, p[2]) for p in corners]


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two convex image polygons share any area (touching edges do not count)."""
    area, _pts = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(area) > 1e-6


class FakeCamera:
    """Renders the fake scene as the synthetic overhead camera would see it (see the module docstring).

    `scene` defaults to config/fake_scene.json; `set_scene` replaces it (a deep copy is kept). `noise` adds
    Gaussian pixel noise per frame (std in gray levels); `pace=True` makes `read()` wait for the frame period
    like a real camera (used by `--check`, never by the tracker, which paces itself).
    """

    is_fake = True

    def __init__(
        self,
        scene: Mapping[str, Any] | None = None,
        *,
        scene_cfg: Mapping[str, Any] | None = None,
        objects: Mapping[int, Mapping[str, Any]] | None = None,
        name: str = "overhead",
        fps: float = 30.0,
        noise: float = 0.0,
        pace: bool = False,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.noise = float(noise)
        self.pace = pace
        self._rng = np.random.default_rng(seed)
        self._cfg = dict(scene_cfg) if scene_cfg is not None else load_scene_cfg()
        self._c, self._hc = camera_pose(self._cfg)
        self._objects = {
            int(k): dict(v) for k, v in (objects if objects is not None else marker_table()).items()
        }
        self._calib: dict[int, tuple[float, float]] = {
            int(k): (float(v[0]), float(v[1])) for k, v in (self._cfg.get("calib_markers") or {}).items()
        }
        self._by_name = {v["name"]: (k, v) for k, v in self._objects.items()}
        self._hand_id, self._hand_h = HAND_ID, HAND_H
        for marker_id, entry in self._objects.items():
            if entry.get("kind") == "hand":
                self._hand_id, self._hand_h = marker_id, float(entry.get("h", HAND_H))
        self._scene = copy.deepcopy(dict(scene)) if scene is not None else load_fake_scene()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._patches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._stats = _Stats(fps)
        self._period = 1.0 / float(fps) if fps > 0 else 0.0
        self._next_t = time.perf_counter()  # the pacing clock (see read())
        self._released = False
        # No driver behind a fake: the frame size and pacing rate are its own, the FOURCC is None.
        self.reported: dict[str, Any] = {
            "width": int(IMAGE_SIZE[0]),
            "height": int(IMAGE_SIZE[1]),
            "fps": float(fps),
            "fourcc": None,
        }

    @property
    def target_fps(self) -> float:
        """The pacing rate (the late-gap threshold is LATE_FACTOR periods of it)."""
        return self._stats.target_fps

    # -- scene -------------------------------------------------------------------------------------------
    def set_scene(self, scene: Mapping[str, Any]) -> None:
        """Replace the rendered scene (a deep copy); the next `read()` renders it."""
        with self._lock:
            self._scene = copy.deepcopy(dict(scene))
            self._frame = None

    def scene(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._scene)

    # -- rendering ---------------------------------------------------------------------------------------
    def _project(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        return project(points, self._c, self._hc)

    def _patch(self, marker_id: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Marker image with k px per bit and a one-bit white border, plus its 4 marker corners (float32)."""
        key = (marker_id, k)
        if key not in self._patches:
            side = 6 * k
            img = generate(marker_id, side)
            padded = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_CONSTANT, value=255)
            bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
            src = np.float32([[k, k], [k + side, k], [k + side, k + side], [k, k + side]])
            self._patches[key] = (bgr, src)
        return self._patches[key]

    def _paste_marker(
        self, frame: np.ndarray, marker_id: int, corners_world: Sequence[Sequence[float]]
    ) -> None:
        dst = self._project(corners_world).astype(np.float32)
        edges = [float(np.linalg.norm(dst[(i + 1) % 4] - dst[i])) for i in range(4)]
        k = max(2, int(math.ceil(max(edges) / 6.0)))
        patch, src = self._patch(marker_id, k)
        transform = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(
            patch,
            transform,
            (frame.shape[1], frame.shape[0]),
            dst=frame,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    def _fill(
        self, frame: np.ndarray, points_world: Sequence[Sequence[float]], color: tuple[int, int, int]
    ) -> None:
        pts = np.round(self._project(points_world)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)

    def _layers(self, scene: Mapping[str, Any]) -> tuple[list[list[Point3]], list[_Layer], _Layer | None]:
        """The scene as paint layers: obstacle footprints on the table, the seen items from the lowest to
        the tallest (top face, then marker) and the wristband layer (None when the hand is unseen; it is
        painted last: the person reaches over the table, and the fake exists to exercise the pipeline,
        not to hide the hand behind a box)."""
        footprints: list[list[Point3]] = []
        layers: list[_Layer] = []
        for item in (scene.get("obstacles") or {}).values():
            if not item.get("seen", True):
                continue
            x, y, yaw, h = float(item["x"]), float(item["y"]), float(item.get("yaw", 0.0)), float(item["h"])
            size = item.get("size") or [0.04, 0.04]
            footprints.append(_box_corners(x, y, 0.0, yaw, size))
            marker = (int(item["marker"]), marker_corners_world(x, y, h, yaw, MARKER_SIDE_M["obstacle"]))
            layers.append((h, [(_box_corners(x, y, h, yaw, size), OBSTACLE_TOP_BGR)], marker))
        for name, item in (scene.get("objects") or {}).items():
            if not item.get("seen", True):
                continue
            x, y, yaw, h = float(item["x"]), float(item["y"]), float(item.get("yaw", 0.0)), float(item["h"])
            size = item.get("size") or [0.04, 0.04]
            entry = self._by_name.get(name, (None, {}))[1]
            color = OBJECT_BGR.get(name, DEFAULT_OBJECT_BGR)
            if entry.get("shape") == "cylinder":
                top = _circle_points(x, y, h, float(size[0]) / 2.0)
            else:
                top = _box_corners(x, y, h, yaw, size)
            marker = (int(item["marker"]), marker_corners_world(x, y, h, yaw, MARKER_SIDE_M["object"]))
            layers.append((h, [(top, color)], marker))
        layers.sort(key=lambda layer: layer[0])
        hand_layer: _Layer | None = None
        hand = scene.get("hand") or {}
        if hand.get("seen"):
            hx, hy = float(hand["x"]), float(hand["y"])
            hz = float(hand.get("z", self._hand_h))
            marker = (
                int(hand.get("marker", self._hand_id)),
                marker_corners_world(hx, hy, hz, 0.0, MARKER_SIDE_M["hand"]),
            )
            hand_layer = (hz, [(_box_corners(hx, hy, hz, 0.0, HAND_SIZE), HAND_BGR)], marker)
        return footprints, layers, hand_layer

    def hidden_calibration_ids(self) -> set[int]:
        """Calibration markers that a raised top face or marker covers, even partly, in the image (quiet
        zones included), so the detector cannot read them: what the real camera misses at this pose."""
        with self._lock:
            scene = copy.deepcopy(self._scene)
        _footprints, layers, hand = self._layers(scene)
        covers: list[np.ndarray] = []
        for _z, fills, (_marker_id, corners) in [*layers, *([hand] if hand is not None else [])]:
            covers.extend(self._project(points) for points, _color in fills)
            covers.append(self._project(quiet_zone(corners)))
        hidden: set[int] = set()
        for marker_id, (x, y) in self._calib.items():
            quad = self._project(quiet_zone(marker_corners_world(x, y, 0.0, 0.0, MARKER_SIDE_M["calib"])))
            if any(_overlap(cover, quad) for cover in covers):
                hidden.add(marker_id)
        return hidden

    def render(self) -> np.ndarray:
        """Draw the current scene into a fresh 640x480 BGR frame."""
        width, height = IMAGE_SIZE
        frame = np.full((height, width, 3), TABLE_BGR, np.uint8)
        with self._lock:
            scene = copy.deepcopy(self._scene)
        if self.name != "overhead":
            cv2.putText(
                frame, f"{self.name} camera (fake)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2
            )
            return frame
        tray = (scene.get("zones") or {}).get("tray")
        if tray:
            self._fill(frame, _circle_points(tray["x"], tray["y"], 0.0, float(tray["r"]), 48), TRAY_FLOOR_BGR)
            wall = float(tray.get("wall", 0.0))
            ring = np.round(self._project(_circle_points(tray["x"], tray["y"], wall, float(tray["r"]), 48)))
            cv2.polylines(
                frame, [ring.astype(np.int32).reshape(-1, 1, 2)], True, TRAY_WALL_BGR, 3, cv2.LINE_AA
            )
            for sx, sy in tray.get("slots") or []:
                su, sv = np.round(self._project([(sx, sy, 0.0)])[0]).astype(int)
                cv2.circle(frame, (int(su), int(sv)), 2, TRAY_WALL_BGR, -1, cv2.LINE_AA)
        # Painter's order for a camera looking straight down: the table markers first, then every item
        # from the lowest to the tallest, the wristband last, so what is higher hides what lies under it,
        # as the real camera sees it (a raised object marker over a calibration marker used to vanish
        # under it). `hidden_calibration_ids()` says which table markers this covers.
        for marker_id, (x, y) in sorted(self._calib.items()):
            self._paste_marker(frame, marker_id, marker_corners_world(x, y, 0.0, 0.0, MARKER_SIDE_M["calib"]))
        footprints, layers, hand = self._layers(scene)
        for points in footprints:
            self._fill(frame, points, OBSTACLE_BASE_BGR)
        for _z, fills, (marker_id, corners) in [*layers, *([hand] if hand is not None else [])]:
            for points, color in fills:
                self._fill(frame, points, color)
            self._paste_marker(frame, marker_id, corners)
        return frame

    def read(self) -> np.ndarray | None:
        """A copy of the rendered frame (re-rendered after `set_scene`), with noise if configured."""
        if self._released:
            self._stats.failures += 1
            return None
        if self.pace and self._period > 0:
            # perf_counter, not monotonic: on Windows Python 3.12 monotonic ticks every 15.6 ms, which put
            # up to 49 ms between paced frames against the checker's 83 ms late threshold
            delay = self._next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._next_t = max(self._next_t + self._period, time.perf_counter() - self._period)
        with self._lock:
            cached = self._frame
        if cached is None:
            cached = self.render()
            with self._lock:
                self._frame = cached
        frame = cached.copy()
        if self.noise > 0:
            noise = self._rng.normal(0.0, self.noise, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        self._stats.tick(time.monotonic())
        return frame

    def release(self) -> None:
        self._released = True

    @property
    def fps(self) -> float:
        return self._stats.fps

    @property
    def frames(self) -> int:
        return self._stats.frames

    @property
    def failures(self) -> int:
        return self._stats.failures

    @property
    def late(self) -> int:
        return self._stats.late

    def stats(self) -> dict[str, float | int]:
        return self._stats.stats()


def get_camera(name: str) -> Camera | FakeCamera:
    """The named camera from config/cameras.yaml and the env; a FakeCamera under FAKE_CAMERA=1 or FAKE=1.

    The fake path never parses OVERHEAD_CAM / WRIST_CAM, so a junk value in .env cannot break FAKE mode.
    """
    if _fake("FAKE_CAMERA"):
        entry = load_config().get(name) or DEFAULTS
        return FakeCamera(name=name, fps=float(entry.get("fps", DEFAULTS["fps"])))
    settings = camera_settings(name)
    return Camera(
        settings["index"],
        name=name,
        width=settings["width"],
        height=settings["height"],
        fps=settings["fps"],
        fourcc=str(settings["fourcc"]),
        auto_exposure=bool(settings.get("auto_exposure", False)),
        exposure=settings.get("exposure"),
        gain=settings.get("gain"),
        brightness=settings.get("brightness"),
    )


# -- camera tools (B1): --check, --diagnose, --list ------------------------------------------------------


def runs_setup_dir() -> pathlib.Path:
    """runs/setup/ (under `VARA_RUNS_DIR` when set): where the camera tools keep their evidence."""
    value = os.environ.get("VARA_RUNS_DIR", "").strip()
    return (pathlib.Path(value) if value else ROOT / "runs") / "setup"


def fresh_path(directory: pathlib.Path | str, stem: str, suffix: str = "") -> pathlib.Path:
    """A path `<directory>/<stem>-<UTC stamp><suffix>` that does not exist yet (a counter is appended when
    two runs share the second), so evidence is never overwritten. Creates `directory`."""
    base = pathlib.Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime(STAMP)
    candidate = base / f"{stem}-{stamp}{suffix}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = base / f"{stem}-{stamp}-{counter}{suffix}"
    return candidate


def _explicit_path(out: pathlib.Path | str, kind: str) -> pathlib.Path:
    """An explicit --out, refused when it already exists (the capture_table.py rule); parents are created
    now so an unwritable location fails before a 30-minute soak runs."""
    path = pathlib.Path(out)
    if path.exists():
        raise FileExistsError(f"{kind} already exists: {path}; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_report(report: Mapping[str, Any], path: pathlib.Path | str) -> pathlib.Path:
    """Write `report` as indented JSON to `path` (parents created). Refuses to overwrite: the file is
    created exclusively, so a file that appears between a check and the write is never replaced."""
    file = pathlib.Path(path)
    text = json.dumps(dict(report), indent=2) + "\n"  # serialize first: a bad value leaves no stub file
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError:
        raise FileExistsError(f"report already exists: {file}") from None
    return file


def _default_stem(stem: str, fake: bool) -> str:
    """The default evidence name: `camera-check` for physical runs, `camera-check-synthetic` for fakes, so a
    synthetic report can never be mistaken for physical evidence by its file name alone."""
    return f"{stem}-synthetic" if fake else stem


def _write_jpeg(path: pathlib.Path, frame: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError(f"could not encode {path.name}")
    path.write_bytes(data.tobytes())


def _stamp_synthetic(frame: np.ndarray, text: str = "SYNTHETIC (FAKE_CAMERA) - not a device") -> np.ndarray:
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


class _FeedTiming:
    """Timing one feed as the checker sees it: measured around every `read()` by the checker itself (the
    throwaway diagnostic's method), so it works the same on a Camera, a FakeCamera or a test double."""

    def __init__(self, name: str, cam: Any, target_fps: float) -> None:
        self.name = name
        self.index = getattr(cam, "index", None)
        self.fake = bool(getattr(cam, "is_fake", False))
        self.reported: Mapping[str, Any] = (
            cam.reported if isinstance(getattr(cam, "reported", None), Mapping) else UNREPORTED
        )
        self.target_fps = float(target_fps)
        self.late_gap_s = LATE_FACTOR / self.target_fps
        self.opened = True
        self.frames = 0
        self.failures = 0
        self.late = 0
        self.long = 0
        self.max_read_s = 0.0
        self.max_gap_s = 0.0
        self.first_t: float | None = None
        self.last_t: float | None = None
        self.elapsed_s = 0.0
        self.completed = False
        self.error: str | None = None
        # True while an independent reader thread may still call cam.read(): the capture must not be
        # released under it (OpenCV captures are not thread-safe); the thread clears it as its last act.
        self.reader_alive = False
        self._lock = threading.Lock()

    @classmethod
    def unopened(
        cls, name: str, target_fps: float, error: str, index: int | None = None, *, fake: bool = False
    ) -> _FeedTiming:
        """The entry of a feed whose camera did not open (`index` is the configured one, None for fakes;
        `fake` marks a FakeCamera that failed under FAKE_CAMERA)."""
        timing = cls(name, None, target_fps)
        timing.index = index
        timing.fake = fake
        timing.opened = False
        timing.error = error
        return timing

    def record(self, before: float, after: float, ok: bool) -> None:
        """One read that started at `before` and returned at `after` (`time.perf_counter()` seconds)."""
        with self._lock:
            self.max_read_s = max(self.max_read_s, after - before)
            if not ok:
                self.failures += 1
                return
            if self.last_t is None:
                self.first_t = after
            else:
                gap = after - self.last_t
                self.max_gap_s = max(self.max_gap_s, gap)
                if gap > self.late_gap_s:
                    self.late += 1
                if gap > LONG_GAP_S:
                    self.long += 1
            self.last_t = after
            self.frames += 1

    @property
    def fps(self) -> float:
        """Effective frames per second between the first and the last good read (0.0 below 2 frames)."""
        with self._lock:
            if self.frames < 2 or self.first_t is None or self.last_t is None:
                return 0.0
            span = self.last_t - self.first_t
            return (self.frames - 1) / span if span > 0 else 0.0

    def progress(self) -> str:
        with self._lock:
            frames, failures, late, long = self.frames, self.failures, self.late, self.long
        return f"{self.name}: {frames} frames, {self.fps:.1f} fps, {failures} fail, {late} late, {long} long"

    def report(self) -> dict[str, Any]:
        fps = self.fps
        with self._lock:
            return {
                "name": self.name,
                "index": self.index,
                "fake": self.fake,
                "opened": self.opened,
                "target_fps": self.target_fps,
                "frames": self.frames,
                "failures": self.failures,
                "late": self.late,
                "long": self.long,
                "fps": round(fps, 2),
                "max_read_s": round(self.max_read_s, 4),
                "max_interframe_s": round(self.max_gap_s, 4),
                "width_reported": self.reported.get("width"),
                "height_reported": self.reported.get("height"),
                "fps_reported": self.reported.get("fps"),
                "fourcc_reported": self.reported.get("fourcc"),
                "elapsed_s": round(self.elapsed_s, 3),
                "completed": self.completed,
                "error": self.error,
            }


def evaluate_feed(feed: Mapping[str, Any], criterion: str = "strict") -> list[str]:
    """Why one feed's report entry fails `criterion` (an empty list means it passes).

    `strict`: opened, zero failed reads, zero late gaps, at least 2 frames. `tracking`: opened, zero failed
    reads, zero long gaps, effective fps >= TRACKING_MIN_FPS. A recorded error or an incomplete run fails
    both.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}")
    reasons: list[str] = []
    if not feed.get("opened", True):
        reasons.append("did not open")
    if feed.get("failures", 0):
        reasons.append(f"{feed['failures']} failed reads")
    if criterion == "strict":
        if feed.get("late", 0):
            reasons.append(f"{feed['late']} late gaps (> {LATE_FACTOR} frame periods)")
        if feed.get("frames", 0) < 2:
            reasons.append("fewer than 2 frames")
    else:
        if feed.get("long", 0):
            reasons.append(f"{feed['long']} long gaps (> {LONG_GAP_S} s, the tracker's seen timeout)")
        if float(feed.get("fps", 0.0)) < TRACKING_MIN_FPS:
            reasons.append(f"effective fps {float(feed.get('fps', 0.0)):.2f} below {TRACKING_MIN_FPS:g}")
    if feed.get("error"):
        reasons.append(f"error: {feed['error']}")
    elif not feed.get("completed", True):
        reasons.append("did not run for the full duration")
    return reasons


def apply_criterion(report: Mapping[str, Any], criterion: str) -> dict[str, Any]:
    """A copy of a check report judged under `criterion`: per-feed `reasons`/`passed`, the overall
    `passed`, `exit_code`, `criterion` and `criterion_rule`. Re-judges a saved report without a new run."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}")
    out = json.loads(json.dumps(dict(report)))
    feeds = []
    for feed in out.get("feeds", []):
        reasons = evaluate_feed(feed, criterion)
        feeds.append({**feed, "reasons": reasons, "passed": not reasons})
    out["feeds"] = feeds
    missing = [name for name in out.get("requested_feeds", []) if name not in {f["name"] for f in feeds}]
    problems = list(out.get("errors", []))
    if missing:
        problems.append(f"no entry for requested feed(s): {', '.join(missing)}")
    if not out.get("completed", False):
        problems.append("the check did not complete")
    elif float(out.get("elapsed_s", 0.0)) + 0.0005 < float(out.get("requested_s", 0.0)):
        # elapsed_s is saved rounded to the millisecond: half a millisecond is rounding, not a short run
        problems.append("the check ran shorter than requested")
    passed = not problems and bool(feeds) and all(feed["passed"] for feed in feeds)
    out.update(
        criterion=criterion,
        criterion_rule=CRITERION_RULES[criterion],
        problems=problems,
        passed=passed,
        exit_code=0 if passed else 1,
    )
    return out


def _valid_duration(seconds: Any) -> bool:
    """A finite positive number of seconds (a bool is not a duration)."""
    return (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds > 0
    )


def _configured_index(name: str, config: Mapping[str, Mapping[str, Any]]) -> int | None:
    """The OpenCV index a physical feed is configured at (None under FAKE_CAMERA or for a junk env value),
    recorded for a feed that did not open so the report says which device was missing."""
    if _fake("FAKE_CAMERA"):
        return None
    try:
        return int(camera_settings(name, config)["index"])
    except (ValueError, TypeError):
        return None


def _target_fps(name: str, cam: Any, config: Mapping[str, Mapping[str, Any]]) -> float:
    value = getattr(cam, "target_fps", None)
    try:
        fps = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        fps = float("nan")
    if not math.isfinite(fps) or fps <= 0:
        fps = float((config.get(name) or DEFAULTS).get("fps", DEFAULTS["fps"]))
    return fps if math.isfinite(fps) and fps > 0 else float(DEFAULTS["fps"])


def _read_once(cam: Any, timing: _FeedTiming) -> None:
    before = time.perf_counter()
    frame = cam.read()
    after = time.perf_counter()
    timing.record(before, after, frame is not None)


STOPPED = "stopped before the full duration"


def _read_loop(cam: Any, timing: _FeedTiming, seconds: float, stop: threading.Event) -> None:
    """One feed's reader for the independent mode: reads until `seconds` have passed on its own clock
    (`completed`) or `stop` is set first (`error`). A raising read ends this reader only."""
    start = time.perf_counter()
    try:
        while not stop.is_set():
            if time.perf_counter() - start >= seconds:
                timing.completed = True
                break
            _read_once(cam, timing)
        else:
            timing.error = timing.error or STOPPED
    except BaseException as exc:  # a dead camera must not take the other reader down
        timing.error = f"{type(exc).__name__}: {exc}"
    finally:
        timing.elapsed_s = time.perf_counter() - start


def _run_independent(
    cams: Sequence[tuple[Any, _FeedTiming]], seconds: float, *, progress: bool
) -> tuple[bool, float, bool]:
    """Read every feed in its own thread, released together by a Barrier; returns (completed, elapsed_s,
    interrupted).

    The main thread prints progress and waits at most JOIN_TIMEOUT_S past the duration: a reader still
    inside `read()` then is named in its feed's `error` and keeps `reader_alive`, so the caller leaves that
    capture to process exit instead of releasing it under the reader. Ctrl+C stops every reader and marks
    the feeds interrupted. Every path sets `stop`, aborts the barrier and joins the readers.
    """
    stop = threading.Event()
    barrier = threading.Barrier(len(cams) + 1)

    def worker(cam: Any, timing: _FeedTiming) -> None:
        try:
            try:
                barrier.wait(timeout=10.0)
            except threading.BrokenBarrierError:
                if not stop.is_set():  # a stop (Ctrl+C during the start) is recorded by the main thread
                    timing.error = timing.error or "reader threads did not start together"
                return
            _read_loop(cam, timing, seconds, stop)
        finally:
            timing.reader_alive = False

    threads = [
        threading.Thread(target=worker, args=(cam, timing), name=f"camera-{timing.name}", daemon=True)
        for cam, timing in cams
    ]
    interrupted = timed_out = start_failed = False
    t0: float | None = None
    try:
        for thread, (_cam, timing) in zip(threads, cams, strict=True):
            timing.reader_alive = True
            try:
                thread.start()
            except BaseException:
                timing.reader_alive = False
                raise
        barrier.wait(timeout=10.0)
        t0 = time.perf_counter()
        deadline = t0 + seconds + JOIN_TIMEOUT_S
        next_report = t0 + 1.0
        while any(thread.is_alive() for thread in threads):
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            for thread in threads:
                thread.join(timeout=0.25)
            now = time.perf_counter()
            if progress and now >= next_report:
                next_report += 1.0
                line = " | ".join(timing.progress() for _cam, timing in cams)
                print(f"[{now - t0:6.1f} s] {line}", flush=True)
    except threading.BrokenBarrierError:
        start_failed = True
    except KeyboardInterrupt:
        interrupted = True
        print("Camera diagnostic interrupted before completion.", file=sys.stderr)
    finally:
        stop.set()
        barrier.abort()  # wakes a reader still waiting to start
        join_until = time.perf_counter() + (0.1 if timed_out else JOIN_TIMEOUT_S)
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=max(0.0, join_until - time.perf_counter()))
    now = time.perf_counter()
    for _cam, timing in cams:
        if timing.reader_alive:  # still inside read(): name it, keep its capture away from release()
            timing.completed = False
            timing.error = (
                f"read still blocked {JOIN_TIMEOUT_S:g} s after the duration; capture left to process exit"
            )
            timing.elapsed_s = now - t0 if t0 is not None else 0.0
        elif interrupted or start_failed:
            timing.completed = False
            if timing.error in (None, STOPPED):
                timing.error = "interrupted" if interrupted else "reader threads did not start together"
    # Each reader times its own loop; the longest one is the run's duration (a reader's clock starts
    # inside its thread, so the main thread's own stopwatch could read a hair shorter than requested).
    elapsed = max(timing.elapsed_s for _cam, timing in cams)
    completed = not interrupted and all(timing.completed for _cam, timing in cams)
    return completed, elapsed, interrupted


def check_feeds(
    seconds: float,
    names: Sequence[str],
    *,
    independent: bool = False,
    criterion: str = "strict",
    progress: bool = True,
) -> dict[str, Any]:
    """Read the named feeds for `seconds` and return the report (see the module docstring for its fields).

    Sequential by default (one loop reads every feed in turn); `independent=True` runs one reader thread
    per feed released together by a Barrier. Cameras are released on every exit, including Ctrl+C and a
    raising camera, and the report then says `completed: false`; the one exception is an independent
    reader still blocked inside `read()`, whose capture is left to process exit and named in `errors`.
    `criterion` picks the verdict rule (`passed`, `exit_code`, per-feed `reasons`). Raises ValueError for a
    bad duration, no names or an unknown criterion.
    """
    if not _valid_duration(seconds):
        raise ValueError("camera check needs a finite positive duration in seconds")
    if not names:
        raise ValueError("camera check needs at least one feed")
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}")
    config = load_config()
    started_at = datetime.now(timezone.utc)
    errors: list[str] = []
    cams: list[tuple[Any, _FeedTiming]] = []
    timings: list[_FeedTiming] = []
    completed, elapsed = False, 0.0
    t0: float | None = None
    try:
        for name in names:
            target_fps = _target_fps(name, None, config)
            try:
                cam = get_camera(name)
            except Exception as exc:
                message = f"{name}: cannot open ({exc})"
                print(message, file=sys.stderr)
                errors.append(message)
                timings.append(
                    _FeedTiming.unopened(
                        name,
                        target_fps,
                        f"cannot open ({exc})",
                        index=_configured_index(name, config),
                        fake=_fake("FAKE_CAMERA"),
                    )
                )
                continue
            if isinstance(cam, FakeCamera):
                cam.pace = True
            timing = _FeedTiming(name, cam, _target_fps(name, cam, config))
            cams.append((cam, timing))
            timings.append(timing)
        if not errors:
            if independent:
                completed, elapsed, interrupted = _run_independent(cams, seconds, progress=progress)
                if interrupted:
                    errors.append("interrupted before completion")
            else:
                t0 = time.perf_counter()
                next_report = t0 + 1.0
                while time.perf_counter() - t0 < seconds:
                    for cam, timing in cams:
                        try:
                            _read_once(cam, timing)
                        except Exception as exc:  # name the feed whose read raised; the check ends
                            timing.error = f"{type(exc).__name__}: {exc}"
                            raise
                    now = time.perf_counter()
                    if progress and now >= next_report:
                        next_report += 1.0
                        line = " | ".join(timing.progress() for _cam, timing in cams)
                        print(f"[{now - t0:6.1f} s] {line}", flush=True)
                completed = True
                elapsed = time.perf_counter() - t0
                for _cam, timing in cams:
                    timing.completed, timing.elapsed_s = True, elapsed
    except KeyboardInterrupt:
        print("Camera check interrupted before completion.", file=sys.stderr)
        errors.append("interrupted before completion")
        for _cam, timing in cams:
            timing.error = timing.error or "interrupted"
    except Exception as exc:
        message = f"camera check failed ({type(exc).__name__}: {exc})"
        print(message, file=sys.stderr)
        errors.append(message)
    finally:
        if t0 is not None and not completed:
            elapsed = time.perf_counter() - t0
            for _cam, timing in cams:
                timing.elapsed_s = elapsed
        for cam, timing in cams:
            if timing.reader_alive:  # OpenCV captures are not thread-safe: never release under a reader
                message = f"{timing.name}: reader still inside read(); capture left open until process exit"
                print(message, file=sys.stderr)
                errors.append(message)
                continue
            try:
                cam.release()
            except (Exception, KeyboardInterrupt) as exc:
                message = f"{timing.name}: release failed ({exc})"
                print(message, file=sys.stderr)
                errors.append(message)
    fake = _fake("FAKE_CAMERA") or any(timing.fake for timing in timings)
    report: dict[str, Any] = {
        "tool": "python -m vision.cameras " + ("--diagnose" if independent else "--check"),
        "mode": "synthetic" if fake else "physical",
        "fake": fake,
        "independent": independent,
        "scope": (
            "independent readers, one thread per feed: a diagnostic that names the late feed"
            if independent
            else "one loop reading every feed in turn: a stall in one feed also delays the other"
        ),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "requested_feeds": list(names),
        "requested_s": float(seconds),
        "elapsed_s": round(elapsed, 3),
        "completed": completed,
        "late_factor": LATE_FACTOR,
        "long_gap_s": LONG_GAP_S,
        "tracking_min_fps": TRACKING_MIN_FPS,
        "errors": errors,
        "feeds": [timing.report() for timing in timings],
    }
    return apply_criterion(report, criterion)


def _fmt_reported(feed: Mapping[str, Any]) -> str:
    """The driver read-back of a feed entry for the console: `640x480 30.0 fps MJPG`, `-` where unknown."""
    width, height = feed.get("width_reported"), feed.get("height_reported")
    size = f"{width}x{height}" if width and height else "-"
    fps = f"{feed['fps_reported']:g} fps" if feed.get("fps_reported") is not None else "- fps"
    fourcc = feed.get("fourcc_reported") or ("synthetic" if feed.get("fake") else "-")
    return f"{size} {fps} {fourcc}"


def _print_summary(report: Mapping[str, Any]) -> None:
    if report["fake"]:
        print("SYNTHETIC: FAKE_CAMERA feeds, not a device; this is not physical evidence.")
    for feed in report["feeds"]:
        if not feed["opened"]:
            print(f"{feed['name']}: did not open ({feed['error']})")
            continue
        verdict = "pass" if feed["passed"] else "FAIL: " + "; ".join(feed["reasons"])
        print(
            f"{feed['name']}: {feed['frames']} frames in {feed['elapsed_s']:.1f} s, {feed['fps']} fps, "
            f"{feed['failures']} failed reads, {feed['late']} late frames, {feed['long']} long gaps, "
            f"max read {feed['max_read_s']:.3f} s, max gap {feed['max_interframe_s']:.3f} s, "
            f"driver read-back {_fmt_reported(feed)}; {verdict}"
        )
    for problem in report["problems"]:
        print(f"problem: {problem}")
    status = "PASS" if report["passed"] else "FAIL"
    print(f"verdict: {status} under {report['criterion']} ({report['criterion_rule']})")


def _check(
    seconds: float,
    names: Sequence[str],
    *,
    independent: bool = False,
    criterion: str = "strict",
    out: pathlib.Path | str | None = None,
) -> int:
    """The --check / --diagnose command: run `check_feeds`, print the summary, write the report with its
    exit code to `out` (default a fresh runs/setup/camera-{check,diagnose}[-synthetic]-<ts>.json). Returns
    the exit code: 0 pass, 1 fail (including a report that could not be written), 2 misuse (bad duration,
    no feed, unknown criterion, existing --out), decided before any camera opens."""
    if not _valid_duration(seconds) or not names:
        print("Camera check needs a finite positive duration and at least one feed.", file=sys.stderr)
        return 2
    if criterion not in CRITERIA:
        print(f"Unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}.", file=sys.stderr)
        return 2
    stem = _default_stem("camera-diagnose" if independent else "camera-check", _fake("FAKE_CAMERA"))
    try:
        path = (
            _explicit_path(out, "report file")
            if out is not None
            else fresh_path(runs_setup_dir(), stem, ".json")
        )
    except OSError as exc:  # FileExistsError included
        print(f"Cannot use the report path: {exc}", file=sys.stderr)
        return 2
    report = check_feeds(seconds, names, independent=independent, criterion=criterion)
    _print_summary(report)
    try:
        try:
            write_report(report, path)
        except FileExistsError:  # appeared during the run: never overwrite, never lose a finished soak
            moved = fresh_path(path.parent, path.stem, path.suffix)
            print(f"{path} appeared during the run; writing {moved} instead", file=sys.stderr)
            path = write_report(report, moved)
    except OSError as exc:
        print(f"Could not write the report to {path}: {exc}", file=sys.stderr)
        return 1
    print(f"report: {path}")
    return int(report["exit_code"])


def _configured_indexes(config: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Which index each configured name points at right now, and whether the env or the YAML says so."""
    out: dict[str, dict[str, Any]] = {}
    for name in config:
        env_var = ENV_INDEX.get(name, f"{name.upper()}_CAM")
        raw = os.environ.get(env_var)
        source = env_var if raw is not None and raw.strip() != "" else "config/cameras.yaml"
        try:
            out[name] = {"index": camera_settings(name, config)["index"], "source": source, "error": None}
        except ValueError as exc:
            out[name] = {"index": None, "source": source, "error": str(exc)}
    return out


def list_cameras(
    max_index: int = DEFAULT_MAX_INDEX, out_dir: pathlib.Path | str | None = None
) -> dict[str, Any]:
    """Open OpenCV indexes 0..max_index-1 briefly, one frame each, and save camera-<i>.jpg plus report.json
    under `out_dir` (default a fresh runs/setup/camera-list-<ts>/). Returns the report.

    CAP_DSHOW on Windows, the default backend elsewhere; every capture is released, also on an exception,
    and a capture whose read or release raises is recorded in its entry's `error` without ending the
    enumeration. Under FAKE_CAMERA=1 / FAKE=1 the configured FakeCameras are listed instead (`index:
    null`, one synthetic JPEG each, default directory camera-list-synthetic-<ts>/) and no device opens.
    A KeyboardInterrupt writes report.json for the entries done so far (`completed: false`, exit code 1)
    and is then re-raised. Raises ValueError for max_index < 1 and FileExistsError for an existing
    out_dir. Never run it while the orchestrator holds the cameras. Never touches the arm.
    """
    if not isinstance(max_index, int) or isinstance(max_index, bool) or max_index < 1:
        raise ValueError("--max-index must be a positive integer")
    fake = _fake("FAKE_CAMERA")
    directory = (
        _explicit_path(out_dir, "output directory")
        if out_dir is not None
        else fresh_path(runs_setup_dir(), _default_stem("camera-list", fake))
    )
    directory.mkdir(parents=True, exist_ok=False)
    config = load_config()
    configured = _configured_indexes(config)
    started_at = datetime.now(timezone.utc)
    entries: list[dict[str, Any]] = []
    interrupted = False
    try:
        if fake:
            for name in config or {"overhead": DEFAULTS, "wrist": DEFAULTS}:
                entries.append(_list_fake(name, directory))
        else:
            for index in range(max_index):
                entries.append(
                    _list_index(index, directory, [n for n, c in configured.items() if c["index"] == index])
                )
    except KeyboardInterrupt:
        interrupted = True
    report: dict[str, Any] = {
        "tool": "python -m vision.cameras --list",
        "mode": "synthetic" if fake else "physical",
        "fake": fake,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": not interrupted,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "backend": "fake" if fake else ("CAP_DSHOW" if os.name == "nt" else "default"),
        "max_index": None if fake else max_index,
        "configured": configured,
        "cameras": entries,
        "opened": [e["index"] if e["index"] is not None else e["name"] for e in entries if e["opened"]],
        "directory": str(directory),
    }
    report["exit_code"] = 0 if report["opened"] and not interrupted else 1
    write_report(report, directory / "report.json")
    if interrupted:
        print(f"partial camera listing: {directory / 'report.json'}", file=sys.stderr)
        raise KeyboardInterrupt
    return report


def _list_fake(name: str, directory: pathlib.Path) -> dict[str, Any]:
    """One --list entry for a configured FakeCamera: its synthetic frame, stamped SYNTHETIC, no device."""
    cam = get_camera(name)
    try:
        frame = cam.read()
        entry: dict[str, Any] = {
            "index": None,
            "name": name,
            "opened": True,
            "fake": True,
            "width": int(frame.shape[1]) if frame is not None else None,
            "height": int(frame.shape[0]) if frame is not None else None,
            "fps_reported": cam.reported["fps"],
            "fourcc_reported": cam.reported["fourcc"],
            "frame": None,
            "configured_as": [name],
            "error": None if frame is not None else "the fake returned no frame",
        }
        if frame is not None:
            file = directory / f"fake-{name}.jpg"
            _write_jpeg(file, _stamp_synthetic(frame))
            entry["frame"] = file.name
        return entry
    finally:
        cam.release()


def _list_index(index: int, directory: pathlib.Path, configured_as: list[str]) -> dict[str, Any]:
    """One --list entry: open `index` (CAP_DSHOW on Windows), read back its format, save one frame as
    camera-<index>.jpg and release it. Errors are recorded in the entry; only KeyboardInterrupt escapes
    (after the release)."""
    entry: dict[str, Any] = {
        "index": index,
        "name": None,
        "opened": False,
        "fake": False,
        "width": None,
        "height": None,
        "fps_reported": None,
        "fourcc_reported": None,
        "frame": None,
        "configured_as": configured_as,
        "error": None,
    }
    cap: Any = None
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        if cap.isOpened():
            entry["opened"] = True
            reported = read_back(cap)
            entry.update(
                width=reported["width"],
                height=reported["height"],
                fps_reported=reported["fps"],
                fourcc_reported=reported["fourcc"],
            )
            frame = None
            for _attempt in range(LIST_READ_ATTEMPTS):
                ok, got = cap.read()
                if ok and got is not None and getattr(got, "size", 0):
                    frame = got
                    break
            if frame is None:
                entry["error"] = f"opened but returned no frame in {LIST_READ_ATTEMPTS} reads"
            else:
                entry["width"], entry["height"] = int(frame.shape[1]), int(frame.shape[0])
                file = directory / f"camera-{index}.jpg"
                _write_jpeg(file, frame)
                entry["frame"] = file.name
    except Exception as exc:  # one bad index must not end the enumeration
        entry["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                entry["error"] = entry["error"] or f"release failed ({type(exc).__name__}: {exc})"
    return entry


def _print_list(report: Mapping[str, Any]) -> None:
    if report["fake"]:
        print("SYNTHETIC: FAKE_CAMERA feeds, not devices; no index was opened.")
    print(f"{'index':<9}{'opened':<8}{'size':<11}{'fps':<9}{'fourcc':<13}{'configured as':<16}file / error")
    for entry in report["cameras"]:
        label = entry["name"] if entry["index"] is None else str(entry["index"])
        size = f"{entry['width']}x{entry['height']}" if entry["width"] and entry["height"] else "-"
        fps = f"{entry['fps_reported']:g}" if entry["fps_reported"] is not None else "-"
        fourcc = entry["fourcc_reported"] if entry["fourcc_reported"] is not None else "-"
        names = ", ".join(entry["configured_as"]) or "-"
        tail = entry["frame"] or entry["error"] or "-"
        opened = "yes" if entry["opened"] else "no"
        print(f"{label:<9}{opened:<8}{size:<11}{fps:<9}{fourcc:<13}{names:<16}{tail}")
    for name, item in report["configured"].items():
        where = f"index {item['index']}" if item["index"] is not None else f"invalid ({item['error']})"
        print(f"configured {name}: {where} from {item['source']}")
    if not report["opened"]:
        print("no camera opened; is another program (the orchestrator, SenseCraft) holding them?")
    print(f"report: {report['directory']}")


def _list(max_index: int, out_dir: pathlib.Path | str | None) -> int:
    """The --list command. Returns 0 when at least one index (or fake feed) opened, 1 when none did or
    the enumeration was interrupted, 2 on misuse (bad --max-index, existing --out)."""
    if not isinstance(max_index, int) or isinstance(max_index, bool) or max_index < 1:
        print("--max-index must be a positive integer.", file=sys.stderr)
        return 2
    if out_dir is not None and pathlib.Path(out_dir).exists():
        print(f"Output directory already exists: {out_dir}; choose a new path.", file=sys.stderr)
        return 2
    try:
        report = list_cameras(max_index, out_dir)
    except KeyboardInterrupt:
        print("Camera listing interrupted; report.json lists the indexes done so far.", file=sys.stderr)
        return 1
    except (FileExistsError, ValueError) as exc:
        print(f"Camera listing refused: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Camera listing failed: {exc}", file=sys.stderr)
        return 1
    _print_list(report)
    return int(report["exit_code"])


HELP_EPILOG = """\
--check reads every feed in one loop (the strict B1 soak); --diagnose (or --check with --independent) reads
each feed in its own thread so a late feed is named on its own; --list opens each OpenCV index briefly and
saves one JPEG per opened index. Never run --list (or the others) while the orchestrator or the SenseCraft
app holds the cameras: DirectShow gives a device to one program at a time. Reports go to a fresh
timestamped path under runs/setup/ (VARA_RUNS_DIR moves it); an existing --out is refused before any camera
opens. FAKE_CAMERA=1 / FAKE=1 runs on synthetic feeds, opens no device and labels the output synthetic
(default names camera-check-synthetic-<ts>.json and so on). A flag that does not apply to the chosen mode
is misuse. Exit 0 pass, 1 fail, 2 misuse. Nothing here touches the arm.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """The camera tools CLI (see HELP_EPILOG and the module docstring). Returns the exit code: 0 pass,
    1 fail, 2 misuse (argparse errors exit 2 through SystemExit)."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m vision.cameras",
        description="Camera tools (B1): soak check, independent-reader diagnostic, index listing.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", type=float, metavar="SECONDS", help="read every feed in one loop for SECONDS and judge it"
    )
    mode.add_argument(
        "--diagnose",
        type=float,
        metavar="SECONDS",
        help="one reader thread per feed for SECONDS (max read time, max gap, late/long counts per feed)",
    )
    mode.add_argument(
        "--list", action="store_true", help="open indexes 0..N-1 briefly and save one JPEG per opened index"
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="with --check: one reader thread per feed, the same run as --diagnose",
    )
    parser.add_argument(
        "--camera",
        choices=["overhead", "wrist", "both"],
        default=None,
        help="feeds for --check/--diagnose (default both)",
    )
    parser.add_argument(
        "--criterion",
        choices=list(CRITERIA),
        default=None,
        help="verdict rule for --check/--diagnose: strict = B1's bar (zero late gaps, the default); "
        f"tracking = zero long gaps (> {LONG_GAP_S} s) and >= {TRACKING_MIN_FPS:g} fps per feed",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="report file for --check/--diagnose (default runs/setup/camera-check-<ts>.json or "
        "camera-diagnose-<ts>.json), directory for --list (default runs/setup/camera-list-<ts>/); "
        "an existing path is refused",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        metavar="N",
        help=f"--list: try indexes 0..N-1 (default {DEFAULT_MAX_INDEX})",
    )
    args = parser.parse_args(argv)
    if args.list:
        misplaced = [
            flag
            for flag, given in (
                ("--camera", args.camera is not None),
                ("--criterion", args.criterion is not None),
                ("--independent", args.independent),
            )
            if given
        ]
        if misplaced:
            parser.error(f"{', '.join(misplaced)} apply to --check/--diagnose, not --list")
    elif args.max_index is not None:
        parser.error("--max-index applies to --list only")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if args.list:
        return _list(DEFAULT_MAX_INDEX if args.max_index is None else args.max_index, args.out)
    seconds = args.diagnose if args.diagnose is not None else args.check
    if seconds is None:
        parser.print_help()
        return 2
    camera = args.camera or "both"
    names = ["overhead", "wrist"] if camera == "both" else [camera]
    return _check(
        seconds,
        names,
        independent=args.diagnose is not None or args.independent,
        criterion=args.criterion or "strict",
        out=args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
