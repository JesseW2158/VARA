"""Cameras (B1, B8): the real USB captures and a FakeCamera that renders config/fake_scene.json.

`open_camera(index)` opens a capture at 640x480, 30 fps, MJPG (two such streams fit one USB bus) with
`cv2.CAP_DSHOW` on Windows and autofocus / auto-exposure switched off where the driver allows it, because
exposure flicker moves detections. `Camera` wraps it with frame-rate and drop statistics. Indices come from
config/cameras.yaml, overridden by the environment (`OVERHEAD_CAM`, `WRIST_CAM`); on the demo laptop index 0
is the built-in webcam, so set the variables in .env.

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

`get_camera(name)` honors FAKE_CAMERA / FAKE and the env overrides; `python -m vision.cameras --check
SECONDS` reports frame rate and drops (on the fake with FAKE_CAMERA=1).
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import pathlib
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import yaml

from vision.calibrate import IMAGE_SIZE, camera_pose, load_scene_cfg, project
from vision.markers import BORDER_BITS, HAND_H, HAND_ID, MARKER_SIDE_M, generate, marker_table

__all__ = [
    "CONFIG_PATH",
    "DEFAULTS",
    "Camera",
    "FakeCamera",
    "camera_settings",
    "get_camera",
    "load_config",
    "load_fake_scene",
    "marker_corners_world",
    "open_camera",
    "quiet_zone",
]

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "cameras.yaml"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

DEFAULTS: dict[str, Any] = {"index": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
ENV_INDEX = {"overhead": "OVERHEAD_CAM", "wrist": "WRIST_CAM"}
LATE_FACTOR = 2.5  # a gap longer than this many frame periods counts as a dropped frame

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
    """config/cameras.yaml as `{name: {index, width, height, fps, fourcc}}` (defaults filled in)."""
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
) -> cv2.VideoCapture:
    """Open a capture (DSHOW on Windows), request MJPG at width x height and fps, disable auto modes.

    Raises RuntimeError when the device does not open. Settings the driver refuses are logged, not fatal.
    """
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(index))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"camera {index} did not open (check OVERHEAD_CAM/WRIST_CAM and the USB port)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    # Exposure flicker and focus hunting move detections; the property semantics differ by backend:
    # DSHOW: 0.25 manual / 0.75 auto exposure; V4L2: 1 manual / 3 auto.
    for prop, value in (
        (cv2.CAP_PROP_AUTOFOCUS, 0),
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if os.name == "nt" else 1),
    ):
        try:
            if not cap.set(prop, value):
                log.info("camera %s: property %s not supported", index, prop)
        except cv2.error:  # pragma: no cover - backend-specific
            log.info("camera %s: property %s raised", index, prop)
    got = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
    log.info("camera %s open: %sx%s at %s fps requested, driver reports %s", index, width, height, fps, got)
    return cap


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
    ) -> None:
        self.index = int(index)
        self.name = name
        self.width, self.height = int(width), int(height)
        self._stats = _Stats(fps)
        self._cap = (
            capture
            if capture is not None
            else open_camera(index, width=width, height=height, fps=fps, fourcc=fourcc)
        )

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
        self._next_t = time.monotonic()
        self._released = False

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
            delay = self._next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_t = max(self._next_t + self._period, time.monotonic() - self._period)
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
    )


def _check(seconds: float, names: Sequence[str]) -> int:
    cams: list[Camera | FakeCamera] = []
    for name in names:
        try:
            cam = get_camera(name)
        except Exception as exc:
            print(f"{name}: cannot open ({exc})", file=sys.stderr)
            continue
        if isinstance(cam, FakeCamera):
            cam.pace = True
        cams.append(cam)
    if not cams:
        return 1
    t0 = time.monotonic()
    next_report = t0 + 1.0
    try:
        while time.monotonic() - t0 < seconds:
            for cam in cams:
                cam.read()
            now = time.monotonic()
            if now >= next_report:
                next_report += 1.0
                line = " | ".join(
                    f"{cam.name}: {cam.frames} frames, {cam.fps:.1f} fps, "
                    f"{cam.failures} fail, {cam.late} late"
                    for cam in cams
                )
                print(f"[{now - t0:6.1f} s] {line}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams:
            cam.release()
    bad = 0
    for cam in cams:
        s = cam.stats()
        elapsed = time.monotonic() - t0
        print(
            f"{cam.name}: {s['frames']} frames in {elapsed:.1f} s, {s['fps']} fps, "
            f"{s['failures']} failed reads, {s['late']} late frames"
        )
        bad += int(s["failures"]) + int(s["late"])
    return 0 if bad == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m vision.cameras", description="Camera feed check (B1).")
    parser.add_argument("--check", type=float, metavar="SECONDS", help="read frames for SECONDS and report")
    parser.add_argument("--camera", choices=["overhead", "wrist", "both"], default="both")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if args.check is None:
        parser.print_help()
        return 2
    names = ["overhead", "wrist"] if args.camera == "both" else [args.camera]
    return _check(args.check, names)


if __name__ == "__main__":
    sys.exit(main())
