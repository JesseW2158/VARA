"""Scene state at 10 Hz (B6): markers in, `schemas/scene.json` dicts out.

`SceneTracker(camera, H, objects, scene_cfg)` runs a thread that, ten times a second:

1. grabs a frame and detects markers (`vision.markers.detect`, one ArucoDetector per tracker);
2. maps ids through `objects` (the `marker_table()` form of config/objects.yaml) into objects, obstacles
   and the hand;
3. maps each marker center through H to the table and undoes the parallax of its height h with
   `height_correct` (the wristband uses h = 0.03); the yaw is the direction corner 0 -> corner 1, both
   corners mapped through H (the direction is unaffected by the height scaling);
4. keeps the median of the last 5 readings per item and flips `seen` to false after 0.5 s without a
   detection, keeping the last pose (a marker that comes back after a longer gap starts a fresh median);
5. publishes `{t, objects, obstacles, hand, zones}` (zones from config/scene.yaml), validated against
   schemas/scene.json. Items never detected are left out: the sandbox would otherwise place them at the
   origin, inside the arm base. Items detected once stay, with their last pose, until the tracker stops.

`latest()` is the last published dict; `snapshot(n=10)` is the per-item median of the next n frames the
camera really delivers (seen when detected in at least half of them, otherwise the last known pose with
`seen: false`), pulled synchronously when the thread is not running, so tests and the I2 script can
drive it frame by frame. A grab that returns None (the usual face of an unplugged USB camera) or raises
(a driver error, counted in `errors`) is a miss: it does not count toward n, but the scene is rebuilt
from it, so the `seen` timeouts advance and `latest()` never keeps every item seen at a stale `t` while
the camera is dead. When more than half of the grabs in the window are misses, or a running tracker
publishes nothing for 3n/hz + 2 s (a hung device), `snapshot` raises TimeoutError, so a dead camera
never passes as a scene with every item unseen at its last pose. `stop()` keeps the thread reference
(and `running` stays True) while the thread is still blocked inside `camera.read()`. An object or
obstacle entry without a positive `h` and `size` is rejected when the tracker is built.
`annotated_jpeg()` is the last frame with detections drawn, a placeholder before the first frame.

`FakeSceneTracker` serves config/fake_scene.json with a fresh `t` (the orchestrator's fast path) and
`get_tracker()` is the process-wide singleton: the fake under FAKE_CAMERA=1 / FAKE=1, else a started
SceneTracker on `get_camera("overhead")` and config/homography.npy. `python -m vision.scene --print`
prints scene JSON lines at 10 Hz; with FAKE_CAMERA=1 it runs the REAL tracker on the FakeCamera.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
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
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from vision.calibrate import (
    IMAGE_SIZE,
    camera_pose,
    height_correct,
    load_homography,
    load_scene_cfg,
    pixel_to_table,
    synthetic_homography,
)
from vision.cameras import FakeCamera, get_camera, load_fake_scene
from vision.markers import HAND_H, HAND_ID, detect, detector, draw, marker_table

__all__ = [
    "FILTER_N",
    "RATE_HZ",
    "SEEN_TIMEOUT_S",
    "FakeSceneTracker",
    "SceneTracker",
    "circular_median",
    "fake_chain_tracker",
    "get_tracker",
    "reset_tracker",
    "validate_scene",
]

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "scene.json"

RATE_HZ = 10.0
FILTER_N = 5
SEEN_TIMEOUT_S = 0.5
ROUND = 4  # published coordinates are rounded to 0.1 mm / 0.1 mrad

_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)

Reading = tuple[float, float, float]  # x, y, yaw


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


def validate_scene(scene: Any) -> None:
    """Raise ValueError with the most specific schema error when `scene` is not a valid scene dict."""
    errors = list(_VALIDATOR.iter_errors(scene))
    if errors:
        err = best_match(errors) or errors[0]
        raise ValueError(f"scene {err.json_path}: {err.message}")


def circular_median(angles: Sequence[float]) -> float:
    """Median of angles in radians that survives the wrap at +-pi (unwrapped around the last value)."""
    if not angles:
        raise ValueError("no angles")
    ref = float(angles[-1])
    unwrapped = [ref + math.remainder(float(a) - ref, 2 * math.pi) for a in angles]
    med = float(np.median(unwrapped))
    return math.remainder(med, 2 * math.pi)


class _Track:
    __slots__ = ("last_seen", "pose", "readings")

    def __init__(self, n: int) -> None:
        self.readings: deque[Reading] = deque(maxlen=n)
        self.last_seen: float | None = None
        self.pose: Reading | None = None


def _median_pose(readings: Sequence[Reading]) -> Reading:
    xs = [r[0] for r in readings]
    ys = [r[1] for r in readings]
    return float(np.median(xs)), float(np.median(ys)), circular_median([r[2] for r in readings])


def _checked_height(marker_id: int, name: str, value: Any, hc: float, *, allow_zero: bool = False) -> float:
    try:
        h = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"marker {marker_id} ({name}): h must be a number, got {value!r}") from exc
    if not math.isfinite(h) or h < 0.0 or h >= hc or (h == 0.0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"marker {marker_id} ({name}): h must be {bound} and below Hc={hc}, got {value!r}")
    return h


def _checked_entry(marker_id: int, entry: Mapping[str, Any], hc: float) -> dict[str, Any]:
    """An object/obstacle entry with a usable name, h and size, or ValueError.

    schemas/scene.json needs h > 0 and size entries > 0: a config/objects.yaml entry without them would
    make every published frame fail validation, so the tracker would run and publish nothing.
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"marker {marker_id}: entry needs a name")
    if "h" not in entry or "size" not in entry:
        raise ValueError(f"marker {marker_id} ({name}): needs h and size [length, width] in meters")
    h = _checked_height(marker_id, name, entry["h"], hc)
    raw_size = entry["size"]
    try:
        size = [float(raw_size[0]), float(raw_size[1])]
        ok = len(raw_size) == 2 and all(math.isfinite(v) and v > 0.0 for v in size)
    except (TypeError, KeyError, IndexError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"marker {marker_id} ({name}): size must be [length > 0, width > 0], got {raw_size!r}"
        )
    return {**entry, "name": name, "h": h, "size": size}


def _miss_message(missed: int, delivered: int, count: int) -> str:
    return (
        f"camera delivered no frame for {missed} of {missed + delivered} grabs while taking a "
        f"{count}-frame snapshot (unplugged or failing camera)"
    )


class SceneTracker:
    """Marker detections -> scene dicts (see the module docstring)."""

    is_fake = False

    def __init__(
        self,
        camera: Any,
        h: np.ndarray,
        objects: Mapping[int, Mapping[str, Any]],
        scene_cfg: Mapping[str, Any],
        *,
        hz: float = RATE_HZ,
        filter_n: int = FILTER_N,
        seen_timeout: float = SEEN_TIMEOUT_S,
        det: Any | None = None,
    ) -> None:
        self.camera = camera
        self.h = np.asarray(h, dtype=np.float64)
        if self.h.shape != (3, 3) or not np.all(np.isfinite(self.h)):
            raise ValueError("H must be a finite 3x3 homography (config/homography.npy)")
        self._c, self._hc = camera_pose(scene_cfg)
        self._objects: dict[int, dict[str, Any]] = {}
        self._hand_id, self._hand_h = HAND_ID, HAND_H
        for marker_id, entry in objects.items():
            kind = entry.get("kind")
            if kind == "hand":
                self._hand_id = int(marker_id)
                self._hand_h = _checked_height(
                    self._hand_id, "hand", entry.get("h", HAND_H), self._hc, allow_zero=True
                )
            elif kind in ("object", "obstacle"):
                self._objects[int(marker_id)] = _checked_entry(int(marker_id), entry, self._hc)
        self._by_name: dict[str, tuple[int, dict[str, Any]]] = {
            entry["name"]: (marker_id, entry) for marker_id, entry in self._objects.items()
        }
        if "zones" not in scene_cfg:
            raise ValueError("scene_cfg needs zones (config/scene.yaml)")
        self._zones = copy.deepcopy(scene_cfg["zones"])
        self.hz = float(hz)
        self.filter_n = int(filter_n)
        self.seen_timeout = float(seen_timeout)
        self._det = det if det is not None else detector()
        self._tracks: dict[str, _Track] = {}
        self._hand = _Track(self.filter_n)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._collectors: list[list[dict[str, Reading] | None]] = []  # None = a failed grab
        self._last_frame: np.ndarray | None = None
        self._last_dets: list[tuple[int, np.ndarray]] = []
        self._latest: dict[str, Any] = self._build_scene(time.monotonic(), time.time())
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames = 0
        self.misses = 0
        self.errors = 0

    # -- detection -> readings -----------------------------------------------------------------------
    def _readings(self, dets: Sequence[tuple[int, np.ndarray]]) -> dict[str, Reading]:
        """Per detected item: table (x, y) height-corrected and the yaw of its top edge."""
        out: dict[str, Reading] = {}
        for marker_id, corners in dets:
            if marker_id == self._hand_id:
                name, height = "hand", self._hand_h
            else:
                entry = self._objects.get(int(marker_id))
                if entry is None:
                    continue
                name, height = entry["name"], float(entry["h"])
            if name in out:
                continue
            pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            cu, cv_ = pts.mean(axis=0)
            table = pixel_to_table(self.h, cu, cv_)
            x, y = height_correct(table, self._c, height, self._hc)
            p0 = pixel_to_table(self.h, pts[0, 0], pts[0, 1])
            p1 = pixel_to_table(self.h, pts[1, 0], pts[1, 1])
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            out[name] = (x, y, yaw)
        return out

    def _update(self, readings: Mapping[str, Reading], now: float) -> None:
        """Push readings into the per-item medians (caller holds the lock)."""
        for name, reading in readings.items():
            if name == "hand":
                track = self._hand
            else:
                track = self._tracks.get(name)
                if track is None:
                    track = self._tracks[name] = _Track(self.filter_n)
            if track.last_seen is not None and now - track.last_seen > self.seen_timeout:
                track.readings.clear()  # back after a gap: do not mix stale readings into the median
            track.readings.append(reading)
            track.last_seen = now
            track.pose = _median_pose(list(track.readings))

    def _item(self, marker_id: int, entry: Mapping[str, Any], pose: Reading, seen: bool) -> dict[str, Any]:
        return {
            "marker": int(marker_id),
            "x": round(pose[0], ROUND),
            "y": round(pose[1], ROUND),
            "yaw": round(pose[2], ROUND),
            "h": float(entry["h"]),
            "size": [float(entry["size"][0]), float(entry["size"][1])],
            "seen": bool(seen),
        }

    def _hand_dict(self, pose: Reading | None, seen: bool) -> dict[str, Any]:
        x, y = (pose[0], pose[1]) if pose is not None else (0.0, 0.0)
        return {
            "marker": self._hand_id,
            "x": round(x, ROUND),
            "y": round(y, ROUND),
            "z": self._hand_h,
            "seen": seen,
        }

    def _build_scene(self, now: float, wall: float) -> dict[str, Any]:
        """The scene dict from the current tracks (caller holds the lock)."""
        scene: dict[str, Any] = {
            "t": wall,
            "objects": {},
            "obstacles": {},
            "hand": None,
            "zones": copy.deepcopy(self._zones),
        }
        for name, (marker_id, entry) in self._by_name.items():
            track = self._tracks.get(name)
            if track is None or track.pose is None:
                continue
            seen = track.last_seen is not None and now - track.last_seen <= self.seen_timeout
            group = "objects" if entry["kind"] == "object" else "obstacles"
            scene[group][name] = self._item(marker_id, entry, track.pose, seen)
        hand_seen = self._hand.last_seen is not None and now - self._hand.last_seen <= self.seen_timeout
        scene["hand"] = self._hand_dict(self._hand.pose, hand_seen)
        return scene

    def _publish(self, scene: dict[str, Any], grab: dict[str, Reading] | None) -> None:
        """Make `scene` the latest and hand `grab` (this frame's readings, None for a failed grab) to
        every running snapshot()."""
        validate_scene(scene)
        with self._cond:
            self._latest = scene
            self._seq += 1
            for collector in self._collectors:
                collector.append(grab)
            self._cond.notify_all()

    def _record_miss(self) -> None:
        """A grab that raised (camera or pipeline): every running snapshot() counts it as a miss."""
        with self._cond:
            for collector in self._collectors:
                collector.append(None)
            self._cond.notify_all()

    def process_frame(
        self, frame: np.ndarray | None, now_mono: float | None = None, now_wall: float | None = None
    ) -> dict[str, Reading]:
        """Detect, filter and publish one frame (None = a missed grab, counted in `misses`: only the
        `seen` timeouts advance, and a running snapshot() records the miss).

        `now_mono` injects the monotonic clock (tests); returns this frame's raw readings per item.
        """
        now = time.monotonic() if now_mono is None else float(now_mono)
        wall = time.time() if now_wall is None else float(now_wall)
        grabbed = frame is not None
        dets = detect(frame, self._det) if grabbed else []
        readings = self._readings(dets)
        with self._lock:
            if grabbed:
                self._last_frame = frame
                self._last_dets = dets
                self.frames += 1
            else:
                self.misses += 1
            self._update(readings, now)
            scene = self._build_scene(now, wall)
        self._publish(scene, readings if grabbed else None)
        return readings

    def _failed(self, what: str) -> None:
        """Count a failure (called inside an `except`): the first and every 100th (10 s at 10 Hz) get the
        traceback at ERROR, the rest a DEBUG line, so a dead camera does not flood the orchestrator's log."""
        self.errors += 1
        if self.errors == 1 or self.errors % 100 == 0:
            log.exception("scene tracker: %s (%d failures so far)", what, self.errors)
        else:
            log.debug("scene tracker: %s (%d failures so far)", what, self.errors)

    def _grab(self, now_mono: float | None = None) -> tuple[bool, dict[str, Reading]]:
        """One camera grab through `process_frame`: (delivered, readings).

        A `read()` that raises (a USB driver error) is a miss like one that returns None, counted in
        `errors` and logged: the scene is rebuilt from the empty reading, so the `seen` timeouts advance
        and a running snapshot() records the miss, instead of `latest()` keeping every item seen at a
        stale `t` for as long as the camera is dead.
        """
        try:
            frame = self.camera.read()
        except Exception:
            self._failed("camera.read() failed; counted as a missed frame")
            frame = None
        return frame is not None, self.process_frame(frame, now_mono)

    def step(self, now_mono: float | None = None) -> dict[str, Reading]:
        """Grab one frame from the camera and process it; a grab that returns None or raises is a miss
        (see `_grab`). Returns this frame's raw readings per item."""
        return self._grab(now_mono)[1]

    # -- thread --------------------------------------------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / self.hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # the pipeline failed on a delivered frame (camera errors end inside step())
                self._failed("frame failed")
                self._record_miss()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()  # fell behind (slow detect): do not try to catch up

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the 10 Hz thread (a no-op while it runs). RuntimeError when an earlier stop() left the
        thread blocked inside camera.read(): release the camera and let that thread die first."""
        if self.running:
            if self._stop.is_set():
                raise RuntimeError("scene tracker: the previous thread is still blocked in camera.read()")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="scene-tracker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop the thread and wait up to `timeout` s for it. Returns False when it is still blocked
        inside camera.read() (a hung device); the reference is kept, so `running` stays True and a later
        snapshot() times out instead of grabbing frames next to the stuck thread."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            log.warning("scene tracker: thread still blocked in camera.read() after %.1f s", timeout)
            return False
        self._thread = None
        return True

    def close(self) -> None:
        """Stop the thread and release the camera."""
        self.stop()
        try:
            self.camera.release()
        except Exception:
            log.exception("scene tracker: releasing the camera failed")

    # -- readers -------------------------------------------------------------------------------------
    def latest(self) -> dict[str, Any]:
        """A copy of the last published scene (empty objects before the first frame)."""
        with self._lock:
            return copy.deepcopy(self._latest)

    def wait_for_frame(self, timeout: float | None = None) -> bool:
        """Block until the next publication; False on timeout."""
        with self._cond:
            target = self._seq + 1
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._seq < target:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                if not self.running and self._thread is None:
                    return False
                self._cond.wait(remaining if remaining is not None else 0.5)
            return True

    def snapshot(self, n: int = 10, *, timeout: float | None = None) -> dict[str, Any]:
        """Median over the next n frames the camera delivers (synchronous grabs when the thread is not
        running).

        A grab that returns None or raises is a miss and does not count toward n. Raises TimeoutError
        when more than half of the grabs in the window are misses (an unplugged camera returns None on
        every read; the result would otherwise be a stale scene with every item unseen at its last pose)
        or when the running tracker publishes nothing for `timeout` seconds (default three times the n
        frames at the nominal rate plus 2 s: a hung device). A dead camera never passes as a scene.
        """
        count = max(1, int(n))
        frames: list[dict[str, Reading]] = []
        if not self.running:
            missed = 0
            while len(frames) < count:
                delivered, readings = self._grab()
                if delivered:
                    frames.append(readings)
                else:
                    missed += 1
                    if 2 * missed > count:
                        raise TimeoutError(_miss_message(missed, len(frames), count))
        else:
            collector: list[dict[str, Reading] | None] = []
            limit = timeout if timeout is not None else 3.0 * count / self.hz + 2.0
            deadline = time.monotonic() + limit
            with self._cond:
                self._collectors.append(collector)
                try:
                    while True:
                        frames = [grab for grab in collector if grab is not None]
                        missed = len(collector) - len(frames)
                        if 2 * missed > count:
                            raise TimeoutError(_miss_message(missed, len(frames), count))
                        if len(frames) >= count:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"scene tracker published {len(frames)} of {count} frames in {limit:.1f} s"
                            )
                        self._cond.wait(remaining)
                finally:
                    self._collectors.remove(collector)
            frames = frames[:count]
        with self._lock:
            return self._median_scene(frames, time.time())

    def _median_scene(self, frames: Sequence[Mapping[str, Reading]], wall: float) -> dict[str, Any]:
        """Per-item medians over raw per-frame readings (caller holds the lock)."""
        total = len(frames)
        scene: dict[str, Any] = {
            "t": wall,
            "objects": {},
            "obstacles": {},
            "hand": None,
            "zones": copy.deepcopy(self._zones),
        }
        for name, (marker_id, entry) in self._by_name.items():
            readings = [f[name] for f in frames if name in f]
            track = self._tracks.get(name)
            if readings and 2 * len(readings) >= total:
                pose, seen = _median_pose(readings), True
            elif track is not None and track.pose is not None:
                pose, seen = track.pose, False
            else:
                continue
            group = "objects" if entry["kind"] == "object" else "obstacles"
            scene[group][name] = self._item(marker_id, entry, pose, seen)
        hand_readings = [f["hand"] for f in frames if "hand" in f]
        if hand_readings and 2 * len(hand_readings) >= total:
            scene["hand"] = self._hand_dict(_median_pose(hand_readings), True)
        else:
            scene["hand"] = self._hand_dict(self._hand.pose, False)
        validate_scene(scene)
        return scene

    def last_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def annotated_jpeg(self, quality: int = 80) -> bytes:
        """The last frame with markers, names and table coordinates drawn; a placeholder before any frame."""
        with self._lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
            dets = list(self._last_dets)
            scene = copy.deepcopy(self._latest)
        if frame is None:
            frame = _placeholder("no frame yet")
        else:
            labels: dict[int, str] = {}
            for group in ("objects", "obstacles"):
                for name, item in scene[group].items():
                    labels[int(item["marker"])] = f"{name} ({item['x']:.2f}, {item['y']:.2f})"
            hand = scene["hand"]
            if hand["seen"]:
                labels[int(hand["marker"])] = f"hand ({hand['x']:.2f}, {hand['y']:.2f})"
            for marker_id, _ in dets:
                labels.setdefault(marker_id, f"c{marker_id}" if marker_id < 8 else str(marker_id))
            draw(frame, dets, labels)
            seen = sum(1 for g in ("objects", "obstacles") for it in scene[g].values() if it["seen"])
            hand_state = "seen" if hand["seen"] else "unseen"
            header = f"t={scene['t']:.1f}  {len(dets)} markers  {seen} items seen  hand {hand_state}"
            cv2.putText(frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        return _encode_jpeg(frame, quality)


def _placeholder(text: str) -> np.ndarray:
    width, height = IMAGE_SIZE
    frame = np.full((height, width, 3), 48, np.uint8)
    cv2.putText(
        frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA
    )
    return frame


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf.tobytes())


class FakeSceneTracker:
    """Serves config/fake_scene.json (or a given scene) with a fresh `t`; `set_scene` moves things."""

    is_fake = True

    def __init__(self, scene: Mapping[str, Any] | None = None) -> None:
        self._scene = copy.deepcopy(dict(scene)) if scene is not None else load_fake_scene()
        validate_scene(self._scene)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._camera: FakeCamera | None = None

    def latest(self) -> dict[str, Any]:
        with self._lock:
            scene = copy.deepcopy(self._scene)
        scene["t"] = time.time()
        return scene

    def snapshot(self, n: int = 10, *, timeout: float | None = None) -> dict[str, Any]:
        return self.latest()

    def set_scene(self, scene: Mapping[str, Any]) -> None:
        validate_scene(scene)
        with self._lock:
            self._scene = copy.deepcopy(dict(scene))
            self._jpeg = None

    def wait_for_frame(self, timeout: float | None = None) -> bool:
        time.sleep(min(1.0 / RATE_HZ, timeout if timeout is not None else 1.0 / RATE_HZ))
        return True

    @property
    def running(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def stop(self, timeout: float = 2.0) -> bool:
        return True

    def close(self) -> None:
        return None

    def annotated_jpeg(self, quality: int = 80) -> bytes:
        """The fake scene rendered by a FakeCamera with its detections drawn (cached until set_scene)."""
        with self._lock:
            if self._jpeg is not None:
                return self._jpeg
            scene = copy.deepcopy(self._scene)
        if self._camera is None:
            self._camera = FakeCamera(scene)
        else:
            self._camera.set_scene(scene)
        frame = self._camera.read()
        if frame is None:
            frame = _placeholder("fake camera")
        else:
            labels = {
                int(it["marker"]): name for g in ("objects", "obstacles") for name, it in scene[g].items()
            }
            draw(frame, detect(frame), labels)
            cv2.putText(
                frame, "FAKE_CAMERA", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
        jpeg = _encode_jpeg(frame, quality)
        with self._lock:
            self._jpeg = jpeg
        return jpeg


_TRACKER: SceneTracker | FakeSceneTracker | None = None
_TRACKER_LOCK = threading.Lock()


def get_tracker() -> SceneTracker | FakeSceneTracker:
    """Process-wide singleton: FakeSceneTracker under FAKE_CAMERA=1 / FAKE=1, else a started SceneTracker."""
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is None:
            if _fake("FAKE_CAMERA"):
                _TRACKER = FakeSceneTracker()
            else:
                tracker = SceneTracker(
                    get_camera("overhead"), load_homography(), marker_table(), load_scene_cfg()
                )
                tracker.start()
                atexit.register(tracker.close)
                _TRACKER = tracker
        return _TRACKER


def reset_tracker() -> None:
    """Drop (and close) the singleton; tests call this around flag changes."""
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is not None:
            _TRACKER.close()
        _TRACKER = None


def fake_chain_tracker(hz: float = RATE_HZ) -> SceneTracker:
    """The real SceneTracker on a FakeCamera showing config/fake_scene.json (the whole chain, no hardware)."""
    cfg = load_scene_cfg()
    table = marker_table()
    camera = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
    try:
        h = load_homography()
    except FileNotFoundError:
        print("config/homography.npy missing; using the synthetic homography", file=sys.stderr)
        h = synthetic_homography(cfg)
    return SceneTracker(camera, h, table, cfg, hz=hz)


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m vision.scene", description="Print the scene state (B6).")
    parser.add_argument("--print", action="store_true", help="print one scene JSON line per frame at 10 Hz")
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long (default: Ctrl+C)")
    parser.add_argument("--hz", type=float, default=RATE_HZ, help="tracker rate (default 10)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if not args.print:
        parser.print_help()
        return 2
    if _fake("FAKE_CAMERA"):
        tracker = fake_chain_tracker(args.hz)
    else:
        tracker = SceneTracker(
            get_camera("overhead"), load_homography(), marker_table(), load_scene_cfg(), hz=args.hz
        )
    tracker.start()
    printed = 0
    t0 = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            if not tracker.wait_for_frame(timeout=2.0):
                print("no frame within 2 s", file=sys.stderr)
                continue
            print(json.dumps(tracker.latest()), flush=True)
            printed += 1
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
    elapsed = time.monotonic() - t0
    rate = printed / elapsed if elapsed > 0 else 0.0
    print(
        f"printed {printed} scenes in {elapsed:.1f} s ({rate:.1f} Hz), {tracker.frames} frames, "
        f"{tracker.misses} missed, {tracker.errors} errors",
        file=sys.stderr,
    )
    return 0 if printed > 0 and tracker.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
