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

Lens: `SceneTracker(..., lens=)` takes a `vision.calibrate.load_active_lens()` result (default None: raw
frames). `process_frame` undistorts every delivered frame (`vision.calibrate.undistort_frame`) before
detection, so `annotated_jpeg()` and `last_frame()` show the undistorted frame; a frame of another size
than the lens's raises ValueError (a failed frame). `get_tracker()` and the CLI's physical tracker load
config/lens.npz once, before opening the camera: absent means raw frames, corrupt raises ValueError at
startup. The FakeCamera renders an ideal pinhole, so `fake_chain_tracker` and the FakeSceneTracker never
undistort; only a test that passes `lens=` explicitly gets undistortion on a fake feed. The scene dict
(schemas/scene.json allows no extra keys) never carries the lens: `lens_sha256` (the file's sha256 or None)
and `diagnostics()` (`{type, frames, misses, errors, lens}`) do, and so do the `tracker` block of the
`--accuracy` report and the stream summary line.

`FakeSceneTracker` serves config/fake_scene.json with a fresh `t` (the orchestrator's fast path) and
`get_tracker()` is the process-wide singleton: the fake under FAKE_CAMERA=1 / FAKE=1, else a started
SceneTracker on `get_camera("overhead")` and config/homography.npy.

CLI (`python -m vision.scene`, .env loaded through python-dotenv; nothing here touches the arm):

- `--print` prints one scene JSON line per frame at 10 Hz; with FAKE_CAMERA=1 (or `--fake`) it runs the
  REAL tracker on the FakeCamera (`fake_chain_tracker`).
- `--out FILE` (alone or with `--print`) also rewrites FILE with every published scene, atomically
  (`write_scene_atomic`: the JSON goes to a temporary file in the same directory and `os.replace` swaps
  it in, so a reader sees the previous file or the complete new one, never a partial write). That is the
  `--scene-file` of `python -m arm.replay --real` and `python -m arm.real_robot --real --carry-test`,
  which re-read it every tick for the hand check (under `--real` they also treat a scene older than
  `SEEN_TIMEOUT_S` plus one frame period as unseen, but the writer does not rely on that). So it never
  leaves a stale pose marked seen: it replaces FILE with the tracker's empty scene (nothing seen yet)
  before the first frame, when the tracker publishes nothing for `STALE_FEED_S` (1 s, or 3 frame periods
  at a low `--hz`: a `camera.read()` that hangs) it rewrites the last scene with every item and the hand
  `seen: false` (`unseen_scene`) until the next frame, and when it stops (`--seconds`, Ctrl+C, an error)
  it leaves the last scene marked unseen the same way. FILE may not lie inside config/, skills/, schemas/
  or tests/, and an existing FILE must be empty or hold a scene (`scene_out_path`), so a typo never
  replaces code or evidence. A fake camera (FAKE/FAKE_CAMERA=1) is refused unless `--fake` is passed.
  Provenance: schemas/scene.json has no room for a label, so before the first scene the writer also
  writes, atomically, the sidecar FILE + ".source.json" (`source_path`, `scene_source`,
  `write_scene_source`): `{writer: "vision.scene", source: "live" | "fake", tracker, camera,
  camera_type, is_fake, lens, pid, started_at, hz}`, where `source` is "live" only when the tracker's
  `is_fake` is False and its camera's `is_fake` is False (a FakeCamera, `--fake`, or a tracker without a
  camera is "fake"; `is_fake` is True exactly when `source` is "fake"; `lens` is `tracker_diagnostics`'
  lens hash; `started_at` is ISO-8601 local time with microseconds, so every start is a new identity).
  Every start rewrites it, so a restart on the same FILE never inherits an earlier writer's label, and
  when the writer stops it removes the sidecar if it is still its own (`remove_scene_source`) before the
  final all-unseen write, so a leftover FILE has no provenance. `--real` readers
  (`arm.replay.scene_file_reader(..., require_live=True)`) refuse a FILE whose sidecar is missing,
  unreadable, not this writer's or not "live", and freeze a running session when it changes. An existing
  sidecar path must be empty or hold an earlier sidecar (`scene_out_path`).
- `--accuracy LAYOUT.json [--frames 10] [--warmup 1] [--out REPORT]` is B6's acceptance check: the
  camera runs for `--warmup` seconds (`warm_up`: a just-opened camera can deliver dark frames), then
  `snapshot(frames)` is compared with a tape-measured layout. LAYOUT is written by hand after measuring
  the table (keep it with the evidence, for example measurements/scene-layout.json), in meters and
  radians, for objects, obstacles and optionally the hand:

      {"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3},
                 "pill_bottle": {"x": 0.18, "y": -0.12, "yaw": null},
                 "cereal_box": {"x": 0.28, "y": 0.07, "yaw": 0.0},
                 "hand": {"x": 0.28, "y": -0.08, "yaw": null}}}

  (illustrative numbers, not measurements). `yaw` is the direction of the marker's top edge (corner 0 to
  corner 1) and is required for objects and obstacles: null skips the yaw check (cylinders), so a
  forgotten yaw is an error rather than a silently weaker check. The hand has no yaw (null or omitted).
  Names must be those of config/objects.yaml or "hand", each once; other top-level keys (a note, the
  date) are ignored (`load_layout`). Per item the tool reports the position error in cm, the yaw error in
  degrees (wrapped to +-180) and `seen`; the check passes only when every listed item is seen with an
  error of at most 1.5 cm and 10 degrees (`accuracy_report`). The report (`{captured_at, mode, fake,
  layout, frames, warmup_s, warmup_frames, tracker, limits, items, passed, snapshot}`) is written to a new
  timestamped file under runs/table-calibration/ (VARA_RUNS_DIR respected) unless `--out REPORT` names a
  new path outside the protected folders; a report is never overwritten. Exit 0 on pass, 1 on fail, 2 on
  bad input or a camera that delivers no snapshot. A fake camera (FAKE/FAKE_CAMERA=1) is refused unless
  `--fake` is passed; then the real tracker runs on the FakeCamera (`fake_chain_tracker`) and the report
  is labelled synthetic (`mode: synthetic`, `accuracy-synthetic-<stamp>.json`).

Misuse exits 2 through argparse: no mode, `--accuracy` with `--print`, `--seconds` or `--hz`, `--frames` or
`--warmup` without `--accuracy`, a non-positive `--seconds`/`--hz`/`--frames` or a negative `--warmup`.
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
import tempfile
import threading
import time
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from vision.calibrate import (
    IMAGE_SIZE,
    camera_pose,
    height_correct,
    lens_hash,
    load_active_lens,
    load_homography,
    load_scene_cfg,
    pixel_to_table,
    synthetic_homography,
    undistort_frame,
)
from vision.cameras import FakeCamera, get_camera, load_fake_scene
from vision.markers import HAND_H, HAND_ID, detect, detector, draw, marker_table

__all__ = [
    "ACCURACY_POSITION_CM",
    "ACCURACY_YAW_DEG",
    "FILTER_N",
    "PROTECTED_DIRS",
    "RATE_HZ",
    "SEEN_TIMEOUT_S",
    "SOURCE_SUFFIX",
    "SOURCE_WRITER",
    "FakeSceneTracker",
    "SceneTracker",
    "accuracy_report",
    "circular_median",
    "configured_names",
    "fake_chain_tracker",
    "get_tracker",
    "load_layout",
    "new_report_path",
    "protected_dir",
    "remove_scene_source",
    "reset_tracker",
    "run_accuracy_check",
    "scene_out_path",
    "scene_source",
    "source_path",
    "tracker_diagnostics",
    "unseen_scene",
    "validate_scene",
    "warm_up",
    "write_scene_atomic",
    "write_scene_source",
]

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "scene.json"

RATE_HZ = 10.0
FILTER_N = 5
SEEN_TIMEOUT_S = 0.5
ROUND = 4  # published coordinates are rounded to 0.1 mm / 0.1 mrad

ACCURACY_POSITION_CM = 1.5  # B6: published positions within 1.5 cm of the tape measurements
ACCURACY_YAW_DEG = 10.0  # B6: yaw within 10 degrees
LAYOUT_MAX_M = 2.0  # a layout coordinate beyond this was typed in centimeters, not meters
WARMUP_S = 1.0  # --accuracy: seconds the camera runs before the snapshot (dark frames after opening)
EXAMPLE_LAYOUT = "measurements/scene-layout.json"  # where the --help text suggests keeping the layout
PROTECTED_DIRS = ("config", "skills", "schemas", "tests")  # the scene tools never write output in these
REPLACE_RETRIES = 20  # Windows: os.replace fails while a reader holds the file open (microseconds)
REPLACE_PAUSE_S = 0.005
STALE_FEED_S = 1.0  # --out: no publication for this long (3 frame periods at a low --hz) = a stalled feed
SOURCE_SUFFIX = ".source.json"  # --out FILE: the provenance sidecar is FILE + this (arm.replay reads it)
SOURCE_WRITER = "vision.scene"  # the sidecar's `writer`; arm.replay refuses any other

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
        lens: Mapping[str, Any] | None = None,
    ) -> None:
        self.camera = camera
        self.h = np.asarray(h, dtype=np.float64)
        if self.h.shape != (3, 3) or not np.all(np.isfinite(self.h)):
            raise ValueError("H must be a finite 3x3 homography (config/homography.npy)")
        if lens is not None and not ("sha256" in lens and "maps" in lens and "image_size" in lens):
            raise ValueError("lens must be a vision.calibrate.load_active_lens() result (or None)")
        self._lens = lens
        self.lens_sha256: str | None = lens_hash(lens)
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
        """Detect, filter and publish one raw camera frame (None = a missed grab, counted in `misses`: only
        the `seen` timeouts advance, and a running snapshot() records the miss). With a lens the frame is
        undistorted first (ValueError for a frame of another size than the lens's).

        `now_mono` injects the monotonic clock (tests); returns this frame's raw readings per item.
        """
        now = time.monotonic() if now_mono is None else float(now_mono)
        wall = time.time() if now_wall is None else float(now_wall)
        grabbed = frame is not None
        if grabbed and self._lens is not None:
            frame = undistort_frame(frame, self._lens)
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
    def diagnostics(self) -> dict[str, Any]:
        """`{type, frames, misses, errors, lens}`: the counters plus the lens file's sha256 (None while
        frames are used raw), so evidence states whether undistortion was active."""
        return {
            "type": type(self).__name__,
            "frames": self.frames,
            "misses": self.misses,
            "errors": self.errors,
            "lens": self.lens_sha256,
        }

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
            if self._lens is not None:
                header += "  undistorted"
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
    lens_sha256: str | None = None  # no camera frames, so never undistorted

    def __init__(self, scene: Mapping[str, Any] | None = None) -> None:
        self._scene = copy.deepcopy(dict(scene)) if scene is not None else load_fake_scene()
        validate_scene(self._scene)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._camera: FakeCamera | None = None

    def diagnostics(self) -> dict[str, Any]:
        """`{type, frames, misses, errors, lens}` like SceneTracker's: no counters, no lens."""
        return {"type": type(self).__name__, "frames": None, "misses": None, "errors": None, "lens": None}

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
    """Process-wide singleton: FakeSceneTracker under FAKE_CAMERA=1 / FAKE=1, else a started SceneTracker
    undistorting through config/lens.npz when it exists (loaded once, before the camera opens; a corrupt
    lens file raises ValueError here)."""
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is None:
            if _fake("FAKE_CAMERA"):
                _TRACKER = FakeSceneTracker()
            else:
                lens = load_active_lens()
                h, table, cfg = load_homography(), marker_table(), load_scene_cfg()
                camera = get_camera("overhead")
                try:
                    tracker = SceneTracker(camera, h, table, cfg, lens=lens)
                except BaseException:
                    camera.release()
                    raise
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
    """The real SceneTracker on a FakeCamera showing config/fake_scene.json (the whole chain, no hardware).

    The FakeCamera renders through the synthetic camera model of config/scene.yaml, so its frames are mapped
    back with `synthetic_homography`, never config/homography.npy: after a physical calibration the saved
    matrix describes the real camera and would misplace every synthetic marker."""
    cfg = load_scene_cfg()
    table = marker_table()
    camera = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
    return SceneTracker(camera, synthetic_homography(cfg), table, cfg, hz=hz)


# ------------------------------------------------------------------------------------------------------------
# Scene file writer: the --scene-file of the real-arm CLIs (A9, A10)
# ------------------------------------------------------------------------------------------------------------


def _runs_dir() -> pathlib.Path:
    value = os.environ.get("VARA_RUNS_DIR", "").strip()
    return pathlib.Path(value) if value else ROOT / "runs"


def protected_dir(path: pathlib.Path | str) -> str | None:
    """The protected repo folder (`"config/"`, `"skills/"`, `"schemas/"` or `"tests/"`, see
    `PROTECTED_DIRS`) that `path` lies in, or None. The scene tools never write there: the live scene file,
    accuracy reports and verification sessions must not land on a committed config, skill, schema or
    fixture (vision.verify's `--out` uses it too)."""
    file = pathlib.Path(path).resolve()
    for name in PROTECTED_DIRS:
        if file.is_relative_to((ROOT / name).resolve()):
            return f"{name}/"
    return None


def _check_replaceable(file: pathlib.Path) -> None:
    """ValueError unless the existing `file` is empty or holds a valid scene: `--out` only ever replaces a
    scene file from an earlier run, never code, .env or evidence named by mistake."""
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{file} exists and cannot be read as a scene ({exc}); not replacing it") from exc
    if not text.strip():
        return
    try:
        validate_scene(json.loads(text))
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"{file} exists and is not a scene file ({exc}); not replacing it") from exc


def scene_out_path(path: pathlib.Path | str) -> pathlib.Path:
    """The resolved path for `--out FILE` with its parent directory created.

    ValueError when the path lies inside config/, skills/, schemas/ or tests/ (`protected_dir`: the live
    file is rewritten ten times a second and must never land on a committed config, skill, schema or
    fixture), names a directory, or names an existing file that is neither empty nor a valid scene
    (`_check_replaceable`), and likewise when its sidecar path (`source_path`) is a directory or an
    existing file that is neither empty nor an earlier sidecar (`_check_source_replaceable`).
    """
    file = pathlib.Path(path).resolve()
    folder = protected_dir(file)
    if folder is not None:
        raise ValueError(f"{file} is inside {folder}: the scene tools never write there")
    if file.is_dir():
        raise ValueError(f"{file} is a directory; --out takes a file path")
    if file.exists():
        _check_replaceable(file)
    sidecar = source_path(file)
    if sidecar.is_dir():
        raise ValueError(f"{sidecar} (the provenance sidecar of {file.name}) is a directory")
    if sidecar.exists():
        _check_source_replaceable(sidecar)
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


def source_path(scene_file: pathlib.Path | str) -> pathlib.Path:
    """The provenance sidecar of a scene file: FILE + SOURCE_SUFFIX (`scene.json.source.json`)."""
    return pathlib.Path(str(scene_file) + SOURCE_SUFFIX)


def _check_source_replaceable(sidecar: pathlib.Path) -> None:
    """ValueError unless the existing `sidecar` is empty or a JSON object whose `writer` is SOURCE_WRITER:
    `--out` only replaces an earlier run's sidecar, never a file that happens to have the name."""
    try:
        text = sidecar.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{sidecar} exists and cannot be read as a sidecar ({exc}); not replacing it"
        ) from exc
    if not text.strip():
        return
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{sidecar} exists and is not a scene sidecar ({exc}); not replacing it") from exc
    if not isinstance(data, dict) or data.get("writer") != SOURCE_WRITER:
        raise ValueError(f"{sidecar} exists and was not written by {SOURCE_WRITER}; not replacing it")


def _camera_label(camera: Any) -> int | str | None:
    """The camera's device index when it has one (the real `Camera`), else its name, else None."""
    if camera is None:
        return None
    index = getattr(camera, "index", None)
    if isinstance(index, int) and not isinstance(index, bool):
        return index
    name = getattr(camera, "name", None)
    return str(name) if name is not None else None


def scene_source(tracker: Any, *, hz: float = RATE_HZ) -> dict[str, Any]:
    """The provenance sidecar of a `--out` scene file written from `tracker` (see the module docstring):
    `{writer, source, tracker, camera, camera_type, is_fake, lens, pid, started_at, hz}`.

    `source` is "live" only when `tracker.is_fake` is False and the tracker has a camera whose `is_fake`
    is False; anything else (the FakeSceneTracker, the real SceneTracker on a FakeCamera, a stand-in
    without those attributes) is "fake", so a missing attribute never passes as live. `is_fake` is
    `source == "fake"`. `started_at` (local ISO-8601 with microseconds) and `pid` identify this start."""
    camera = getattr(tracker, "camera", None)
    live = (
        getattr(tracker, "is_fake", True) is False
        and camera is not None
        and getattr(camera, "is_fake", True) is False
    )
    return {
        "writer": SOURCE_WRITER,
        "source": "live" if live else "fake",
        "tracker": type(tracker).__name__,
        "camera": _camera_label(camera),
        "camera_type": None if camera is None else type(camera).__name__,
        "is_fake": not live,
        "lens": tracker_diagnostics(tracker).get("lens"),
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "hz": float(hz),
    }


def write_scene_source(source: Mapping[str, Any], scene_file: pathlib.Path | str) -> pathlib.Path:
    """Write the provenance sidecar `source` (a `scene_source` dict) next to `scene_file`, atomically like
    `write_scene_atomic` (a reader sees the previous sidecar or the complete new one). ValueError for a
    dict that is not this writer's or does not serialize as finite JSON."""
    if not isinstance(source, Mapping) or source.get("writer") != SOURCE_WRITER:
        raise ValueError(f"a scene sidecar needs writer {SOURCE_WRITER!r}, got {source!r:.120}")
    path = source_path(scene_file)
    _write_text_atomic(json.dumps(dict(source), allow_nan=False), path)
    return path


def remove_scene_source(scene_file: pathlib.Path | str, source: Mapping[str, Any]) -> bool:
    """Remove the sidecar of `scene_file` when it is still `source` (same `pid` and `started_at`): a writer
    that stops withdraws its own provenance, never a newer writer's. True when removed or already gone,
    False when the sidecar there is another writer's or not readable as one (left in place; readers refuse
    an unreadable one anyway). On Windows the removal is retried for up to 0.1 s while a reader has the
    file open; a lasting OSError propagates."""
    path = source_path(scene_file)
    for attempt in range(REPLACE_RETRIES):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not (
                isinstance(current, dict)
                and current.get("pid") == source.get("pid")
                and current.get("started_at") == source.get("started_at")
            ):
                return False
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(REPLACE_PAUSE_S)
        except (UnicodeDecodeError, ValueError):
            return False
    return False  # not reached: the last attempt returns or raises


def unseen_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of `scene` with every object, obstacle and the hand `seen: false` (poses kept) and a fresh
    `t`: what `--out` puts in the live file while the feed stalls and leaves there when its writer stops,
    because a reader without `require_live` (arm.replay's `scene_file_reader` under `--fake`) does not
    check `t` and would otherwise keep trusting the last poses of a dead feed (a `--real` reader also
    marks a stale scene unseen itself)."""
    out = copy.deepcopy(dict(scene))
    out["t"] = time.time()
    for group in ("objects", "obstacles"):
        for item in (out.get(group) or {}).values():
            item["seen"] = False
    hand = out.get("hand")
    if isinstance(hand, dict):
        hand["seen"] = False
    return out


def write_scene_atomic(scene: Mapping[str, Any], path: pathlib.Path | str) -> pathlib.Path:
    """Validate `scene` against schemas/scene.json and write it to `path` atomically.

    The JSON goes to a temporary file in the same directory and `os.replace` swaps it in, so a reader
    (arm.replay's `scene_file_reader`, re-reading every tick) sees either the previous file or the
    complete new one, never a partial write. On Windows the replace is retried for up to 0.1 s while a
    reader holds the file open. ValueError for an invalid scene or a non-finite number; the file is then
    left as it was. No fsync: the file is a live feed, not evidence.
    """
    validate_scene(scene)
    file = pathlib.Path(path)
    _write_text_atomic(json.dumps(scene, allow_nan=False), file)
    return file


def _write_text_atomic(text: str, file: pathlib.Path) -> None:
    """`text` to a temporary file next to `file`, then `os.replace` (retried while Windows reports the
    target open); the temporary file never survives."""
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=file.parent, prefix=f".{file.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(text)
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(temporary, file)
                break
            except PermissionError:
                if attempt == REPLACE_RETRIES - 1:
                    raise
                time.sleep(REPLACE_PAUSE_S)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------------------
# Accuracy check against tape measurements (B6)
# ------------------------------------------------------------------------------------------------------------


def configured_names() -> set[str]:
    """Every item name `--accuracy` accepts: the objects and obstacles of config/objects.yaml plus "hand"."""
    names = {
        str(entry["name"]) for entry in marker_table().values() if entry.get("kind") in ("object", "obstacle")
    }
    names.add("hand")
    return names


def _layout_number(file: pathlib.Path, name: str, key: str, value: Any, limit: float, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{file}: item {name}: {key} must be a number in {unit}, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or abs(number) > limit:
        raise ValueError(
            f"{file}: item {name}: {key}={value!r} is outside +-{limit:g} ({unit}, not cm or degrees)"
        )
    return number


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json `object_pairs_hook`: a key given twice is an error, not a silent last-one-wins."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def load_layout(
    path: pathlib.Path | str, known: Collection[str] | None = None
) -> dict[str, dict[str, float | None]]:
    """The tape-measured layout file for `--accuracy` (shape in the module docstring) as
    `{name: {"x": m, "y": m, "yaw": rad | None}}`.

    ValueError for anything else: not JSON, a key given twice, no non-empty `items`, a name not in `known`
    (when given), an item key other than x, y, yaw, a coordinate beyond +-2 m or a yaw beyond +-2 pi
    (typed in cm or degrees), an object or obstacle without `yaw` (null skips its yaw check; omitting it
    is refused so a forgotten measurement never weakens the check silently), a non-null yaw for the hand
    (it has none; null or omitted).
    """
    file = pathlib.Path(path)
    try:
        data = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: not valid JSON ({exc})") from exc
    except ValueError as exc:  # a duplicate key (_unique_keys)
        raise ValueError(f"{file}: {exc}") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("items"), Mapping):
        raise ValueError(f'{file}: expected {{"items": {{"<name>": {{"x": m, "y": m, "yaw": rad | null}}}}}}')
    items = data["items"]
    if not items:
        raise ValueError(f"{file}: items is empty")
    layout: dict[str, dict[str, float | None]] = {}
    for name, entry in items.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{file}: item names must be non-empty strings, got {name!r}")
        if known is not None and name not in known:
            raise ValueError(f"{file}: unknown item {name!r}; configured names: {', '.join(sorted(known))}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"{file}: item {name}: expected {{x, y, yaw}}, got {entry!r}")
        extra = sorted(set(entry) - {"x", "y", "yaw"})
        if extra:
            raise ValueError(f"{file}: item {name}: unknown keys {extra} (only x, y, yaw)")
        pose: dict[str, float | None] = {}
        for key in ("x", "y"):
            if key not in entry:
                raise ValueError(f"{file}: item {name}: needs {key} in meters")
            pose[key] = _layout_number(file, name, key, entry[key], LAYOUT_MAX_M, "meters")
        if "yaw" not in entry and name != "hand":
            raise ValueError(
                f"{file}: item {name}: needs yaw in radians (the marker's top edge), or null to skip the "
                "yaw check (cylinders)"
            )
        yaw = entry.get("yaw")
        if yaw is None:
            pose["yaw"] = None
        elif name == "hand":
            raise ValueError(f"{file}: item hand has no yaw; use null")
        else:
            pose["yaw"] = _layout_number(file, name, "yaw", yaw, 2 * math.pi, "radians")
        layout[name] = pose
    return layout


def _find_item(scene: Mapping[str, Any], name: str) -> tuple[str | None, Mapping[str, Any] | None]:
    if name == "hand":
        hand = scene.get("hand")
        if not isinstance(hand, Mapping):
            return "hand", None
        if not hand.get("seen") and float(hand.get("x", 0.0)) == 0.0 and float(hand.get("y", 0.0)) == 0.0:
            return "hand", None  # the tracker's default pose before the wristband was ever detected
        return "hand", hand
    for group in ("objects", "obstacles"):
        item = (scene.get(group) or {}).get(name)
        if item is not None:
            return group, item
    return None, None


def accuracy_report(
    layout: Mapping[str, Mapping[str, float | None]],
    scene: Mapping[str, Any],
    *,
    position_cm: float = ACCURACY_POSITION_CM,
    yaw_deg: float = ACCURACY_YAW_DEG,
) -> dict[str, Any]:
    """Compare a layout (`load_layout`) with a published scene: `{limits, items, passed}`.

    Per item: `group`, `measured`, `published` (None when the scene never had it), `seen`, `error_cm`,
    `yaw_error_deg` (wrapped to +-180, None when the layout's yaw is null), `ok` and `problems`. `passed`
    is true only when every item is present, seen, within `position_cm` and (when checked) `yaw_deg`.
    """
    if not (math.isfinite(position_cm) and position_cm > 0 and math.isfinite(yaw_deg) and yaw_deg > 0):
        raise ValueError("limits must be finite and > 0")
    items: dict[str, dict[str, Any]] = {}
    for name, measured in layout.items():
        group, published = _find_item(scene, name)
        problems: list[str] = []
        entry: dict[str, Any] = {
            "group": group,
            "measured": {"x": measured["x"], "y": measured["y"], "yaw": measured.get("yaw")},
            "published": None,
            "seen": False,
            "error_cm": None,
            "yaw_error_deg": None,
            "ok": False,
            "problems": problems,
        }
        if published is None:
            problems.append("not in the published scene (never detected)")
        else:
            seen = bool(published.get("seen", False))
            pub_yaw = float(published["yaw"]) if "yaw" in published else None
            entry["published"] = {"x": float(published["x"]), "y": float(published["y"]), "yaw": pub_yaw}
            entry["seen"] = seen
            error_cm = 100.0 * math.hypot(
                float(published["x"]) - float(measured["x"]), float(published["y"]) - float(measured["y"])
            )
            entry["error_cm"] = round(error_cm, 2)
            if not seen:
                problems.append("not seen in the snapshot (last known pose shown)")
            if error_cm > position_cm + 1e-9:
                problems.append(f"position off by {error_cm:.1f} cm (limit {position_cm:g} cm)")
            if measured.get("yaw") is not None:
                if pub_yaw is None:
                    problems.append("published pose has no yaw")
                else:
                    yaw_error = abs(
                        math.degrees(math.remainder(pub_yaw - float(measured["yaw"]), 2 * math.pi))
                    )
                    entry["yaw_error_deg"] = round(yaw_error, 2)
                    if yaw_error > yaw_deg + 1e-9:
                        problems.append(f"yaw off by {yaw_error:.1f} deg (limit {yaw_deg:g} deg)")
        entry["ok"] = not problems
        items[name] = entry
    return {
        "limits": {"position_cm": float(position_cm), "yaw_deg": float(yaw_deg)},
        "items": items,
        "passed": bool(items) and all(entry["ok"] for entry in items.values()),
    }


def warm_up(tracker: Any, seconds: float) -> int:
    """Let a just-opened camera settle for `seconds` before a snapshot; returns the frames it delivered
    meanwhile (DirectShow can hand out dark frames right after it builds its capture graph, and a snapshot
    of those would report items unseen).

    A running tracker is waited on (`wait_for_frame`); a stopped SceneTracker grabs synchronously (`step`,
    which blocks for a real camera's frame period and spins on an unpaced FakeCamera); a tracker with
    neither, such as FakeSceneTracker, has nothing to settle and returns 0 at once, as does a `seconds`
    that is not finite and positive. A dead camera does not raise here: the snapshot that follows raises
    TimeoutError.
    """
    if not (math.isfinite(seconds) and seconds > 0):
        return 0
    start = int(getattr(tracker, "frames", 0) or 0)
    deadline = time.monotonic() + float(seconds)
    if getattr(tracker, "running", False):
        while (remaining := deadline - time.monotonic()) > 0:
            if not tracker.wait_for_frame(timeout=remaining) and not tracker.running:
                break  # the thread ended: nothing more will arrive
    elif callable(getattr(tracker, "step", None)):
        while time.monotonic() < deadline:
            tracker.step()
    else:
        return 0
    return int(getattr(tracker, "frames", 0) or 0) - start


def tracker_diagnostics(tracker: Any) -> dict[str, Any]:
    """`{type, frames, misses, errors, lens}` of any tracker: its `diagnostics()` when it has one, else
    the same keys read with getattr (a stand-in without them reports None). `lens` is the lens file's
    sha256 when frames were undistorted, None when raw."""
    method = getattr(tracker, "diagnostics", None)
    if callable(method):
        return dict(method())
    return {
        "type": type(tracker).__name__,
        "frames": getattr(tracker, "frames", None),
        "misses": getattr(tracker, "misses", None),
        "errors": getattr(tracker, "errors", None),
        "lens": getattr(tracker, "lens_sha256", None),
    }


def run_accuracy_check(
    tracker: Any,
    layout: Mapping[str, Mapping[str, float | None]],
    *,
    frames: int = 10,
    warmup: float = 0.0,
    fake: bool = False,
    layout_path: pathlib.Path | str | None = None,
    position_cm: float = ACCURACY_POSITION_CM,
    yaw_deg: float = ACCURACY_YAW_DEG,
) -> dict[str, Any]:
    """`warm_up(tracker, warmup)`, then `tracker.snapshot(frames)` scored by `accuracy_report`, with
    provenance: `{captured_at, t, mode, fake, layout, frames, warmup_s, warmup_frames, tracker, limits,
    items, passed, snapshot}` (plus `note` when synthetic); `tracker` is `tracker_diagnostics` (its `lens`
    states whether frames were undistorted). A TimeoutError from the tracker (a dead camera) propagates."""
    count = max(1, int(frames))
    warmed = warm_up(tracker, warmup)
    snapshot = tracker.snapshot(count)
    report = accuracy_report(layout, snapshot, position_cm=position_cm, yaw_deg=yaw_deg)
    out: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "t": time.time(),
        "mode": "synthetic" if fake else "physical",
        "fake": bool(fake),
        "layout": None if layout_path is None else str(layout_path),
        "frames": count,
        "warmup_s": float(warmup) if math.isfinite(warmup) and warmup > 0 else 0.0,
        "warmup_frames": warmed,
        "tracker": tracker_diagnostics(tracker),
        **report,
        "snapshot": snapshot,
    }
    if fake:
        out["note"] = "SYNTHETIC - NOT TABLE MEASUREMENTS (FakeCamera rendering config/fake_scene.json)"
    return out


def new_report_path(directory: pathlib.Path | str, prefix: str, suffix: str = ".json") -> pathlib.Path:
    """A new path `directory/<prefix>-<UTC stamp>[-<n>]<suffix>` (parents created, never an existing file)."""
    folder = pathlib.Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = folder / f"{prefix}-{stamp}{suffix}"
    n = 1
    while candidate.exists():
        candidate = folder / f"{prefix}-{stamp}-{n}{suffix}"
        n += 1
    return candidate


def _physical_tracker(hz: float = RATE_HZ) -> SceneTracker:
    """A SceneTracker on the real overhead camera (config/cameras.yaml, config/homography.npy), not
    started, undistorting through config/lens.npz when it exists. RuntimeError when `get_camera` hands back
    a fake (nothing then runs on it); ValueError for a corrupt lens file, before the camera opens."""
    lens = load_active_lens()
    camera = get_camera("overhead")
    if getattr(camera, "is_fake", True) is not False:
        try:
            camera.release()
        except Exception:
            log.exception("releasing the fake camera failed")
        raise RuntimeError(
            f"get_camera('overhead') returned a {type(camera).__name__}, a fake: unset FAKE/FAKE_CAMERA or "
            "pass --fake for a synthetic run"
        )
    try:
        return SceneTracker(camera, load_homography(), marker_table(), load_scene_cfg(), hz=hz, lens=lens)
    except BaseException:
        camera.release()
        raise


def _fmt_pose(pose: Mapping[str, Any] | None) -> str:
    if pose is None:
        return "-"
    yaw = pose.get("yaw")
    yaw_text = "-" if yaw is None else f"{yaw:+.2f}"
    return f"({pose['x']:+.3f}, {pose['y']:+.3f}, {yaw_text})"


def _print_accuracy(report: Mapping[str, Any], path: pathlib.Path | None) -> None:
    limits = report["limits"]
    if report["fake"]:
        print("SYNTHETIC accuracy check on the FakeCamera: not table measurements")
    print(
        f"snapshot: median of {report['frames']} frames after a {report['warmup_s']:g} s warm-up "
        f"({report['warmup_frames']} frames)"
    )
    print(
        f"{'item':12s} {'measured (x, y, yaw)':26s} {'published (x, y, yaw)':26s} "
        f"{'err cm':>7s} {'yaw deg':>8s} {'seen':>4s}  result"
    )
    for name, entry in report["items"].items():
        err = "-" if entry["error_cm"] is None else f"{entry['error_cm']:.2f}"
        yaw = "-" if entry["yaw_error_deg"] is None else f"{entry['yaw_error_deg']:.1f}"
        seen = "yes" if entry["seen"] else "no"
        result = "ok" if entry["ok"] else "; ".join(entry["problems"])
        print(
            f"{name:12s} {_fmt_pose(entry['measured']):26s} {_fmt_pose(entry['published']):26s} "
            f"{err:>7s} {yaw:>8s} {seen:>4s}  {result}"
        )
    failed = [name for name, entry in report["items"].items() if not entry["ok"]]
    total = len(report["items"])
    if report["passed"]:
        print(
            f"PASS: {total} of {total} items within {limits['position_cm']:g} cm "
            f"and {limits['yaw_deg']:g} deg"
        )
    else:
        print(
            f"FAIL: {len(failed)} of {total} items outside the limits ({limits['position_cm']:g} cm, "
            f"{limits['yaw_deg']:g} deg): {', '.join(failed)}"
        )
    print(f"report: {path if path is not None else 'NOT WRITTEN (see the error above)'}")


def _accuracy_report_path(out: str | None, fake: bool) -> pathlib.Path:
    """`--out REPORT` (a new file outside the protected folders, parent created) or a new timestamped
    runs/table-calibration/accuracy[-synthetic]-<UTC stamp>.json. ValueError otherwise."""
    if out is not None:
        path = pathlib.Path(out)
        folder = protected_dir(path)
        if folder is not None:
            raise ValueError(f"{path} is inside {folder}: the scene tools never write there")
        if path.exists():
            raise ValueError(f"{path} exists: a report is never overwritten; choose a new path")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return new_report_path(_runs_dir() / "table-calibration", "accuracy-synthetic" if fake else "accuracy")


def _write_new_json(path: pathlib.Path, data: Any) -> None:
    """Write JSON to a path that must not exist yet (FileExistsError otherwise: never overwritten)."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _accuracy_main(args: argparse.Namespace, frames: int, warmup: float) -> int:
    layout_path = pathlib.Path(args.accuracy)
    if not args.fake and _fake("FAKE_CAMERA"):
        print(
            "FAKE_CAMERA=1 (or FAKE=1): the snapshot would come from config/fake_scene.json, not the table; "
            "pass --fake for a synthetic run or unset the fakes",
            file=sys.stderr,
        )
        return 2
    try:
        layout = load_layout(layout_path, known=configured_names())
    except (OSError, ValueError) as exc:
        print(f"--accuracy {layout_path}: {exc}", file=sys.stderr)
        return 2
    try:
        report_path = _accuracy_report_path(args.out, args.fake)
    except (OSError, ValueError) as exc:
        print(f"--out: {exc}", file=sys.stderr)
        return 2
    try:
        tracker = fake_chain_tracker() if args.fake else _physical_tracker()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"cannot start the scene tracker: {exc}", file=sys.stderr)
        return 2
    try:
        report = run_accuracy_check(
            tracker, layout, frames=frames, warmup=warmup, fake=args.fake, layout_path=layout_path
        )
    except (TimeoutError, ValueError) as exc:  # ValueError: a frame the lens cannot undistort (size)
        print(f"no snapshot: {exc}", file=sys.stderr)
        return 2
    finally:
        tracker.close()
    try:
        _write_new_json(report_path, report)
    except OSError as exc:  # FileExistsError included: another run took the path meanwhile
        print(f"cannot write the report {report_path}: {exc}", file=sys.stderr)
        _print_accuracy(report, None)
        return 2
    _print_accuracy(report, report_path)
    return 0 if report["passed"] else 1


def _withdraw_source(out: pathlib.Path, source: Mapping[str, Any]) -> bool:
    """`remove_scene_source` for the stream CLI: False (logged) only when the removal failed with an
    OSError; a sidecar that another writer has replaced meanwhile is left in place with a warning."""
    if not source:
        return True
    try:
        if not remove_scene_source(out, source):
            log.warning("%s is not this writer's sidecar any more; left in place", source_path(out))
    except OSError as exc:
        log.error("scene file %s: removing the provenance sidecar failed: %s", out, exc)
        return False
    return True


def _stream_main(args: argparse.Namespace) -> int:
    fake_env = _fake("FAKE_CAMERA")
    out: pathlib.Path | None = None
    if args.out is not None:
        if fake_env and not args.fake:  # refused before anything is created on disk
            print(
                "FAKE_CAMERA=1 (or FAKE=1): the scene file would come from config/fake_scene.json; "
                "pass --fake for an offline run (and never feed that file to a --real command) or unset "
                "the fakes",
                file=sys.stderr,
            )
            return 2
        try:
            out = scene_out_path(args.out)
        except (OSError, ValueError) as exc:
            print(f"--out {args.out}: {exc}", file=sys.stderr)
            return 2
    fake = args.fake or fake_env
    hz = RATE_HZ if args.hz is None else float(args.hz)
    stale_s = max(STALE_FEED_S, 3.0 / hz)  # a camera.read() that hangs publishes nothing at all
    try:
        tracker = fake_chain_tracker(hz) if fake else _physical_tracker(hz)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"cannot start the scene tracker: {exc}", file=sys.stderr)
        return 2
    source: dict[str, Any] = {}
    if out is not None:
        if fake:
            print(
                f"SYNTHETIC scene file {out}: the FakeCamera rendering config/fake_scene.json; its sidecar "
                f"{source_path(out).name} says fake, so --real commands refuse it",
                file=sys.stderr,
            )
        try:
            source = scene_source(tracker, hz=hz)
            # the provenance first: a reader never pairs this writer's scenes with an earlier writer's label
            write_scene_source(source, out)
            write_scene_atomic(tracker.latest(), out)  # nothing seen yet: a stale file goes at once
        except (OSError, ValueError) as exc:
            tracker.close()
            _withdraw_source(out, source)
            print(f"--out {out}: cannot write the scene file or its sidecar: {exc}", file=sys.stderr)
            return 2
        print(
            f"scene file {out}: provenance {source_path(out).name} ({source['source']}, {source['tracker']}, "
            f"camera {source['camera']}, pid {source['pid']}), removed when this writer stops",
            file=sys.stderr,
        )
    tracker.start()
    final_unseen = stalled = False
    printed = written = write_errors = 0
    t0 = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            if not tracker.wait_for_frame(timeout=stale_s):
                print(f"no frame within {stale_s:g} s", file=sys.stderr)
                if out is not None and not stalled:
                    try:  # the readers do not check `t`: never let them follow the poses of a stalled feed
                        write_scene_atomic(unseen_scene(tracker.latest()), out)
                        stalled = True
                    except (OSError, ValueError) as exc:
                        write_errors += 1
                        log.error("scene file %s: marking the stalled feed unseen failed: %s", out, exc)
                continue
            stalled = False
            scene = tracker.latest()
            if args.print:
                print(json.dumps(scene), flush=True)
                printed += 1
            if out is not None:
                try:
                    write_scene_atomic(scene, out)
                    written += 1
                except (OSError, ValueError) as exc:
                    write_errors += 1
                    if write_errors == 1 or write_errors % 100 == 0:
                        log.error("scene file %s: write failed (%d so far): %s", out, write_errors, exc)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        if out is not None:
            # the provenance goes first: a --real reader still running freezes, a later one refuses
            if not _withdraw_source(out, source):
                write_errors += 1
            try:  # the feed stopped: never leave the last poses marked seen for a reader that keeps going
                write_scene_atomic(unseen_scene(tracker.latest()), out)
                final_unseen = True
            except (OSError, ValueError) as exc:
                write_errors += 1
                log.error("scene file %s: the final all-unseen write failed: %s", out, exc)
    elapsed = time.monotonic() - t0
    published = max(printed, written)
    rate = published / elapsed if elapsed > 0 else 0.0
    summary = f"printed {printed} scenes"
    if out is not None:
        summary += f", wrote {written} scenes to {out} ({write_errors} write errors"
        summary += "; left with every item unseen)" if final_unseen else "; NOT marked unseen)"
    lens = tracker_diagnostics(tracker)["lens"]
    print(
        f"{summary} in {elapsed:.1f} s ({rate:.1f} Hz), {tracker.frames} frames, "
        f"{tracker.misses} missed, {tracker.errors} errors, lens {lens or 'null (raw frames)'}",
        file=sys.stderr,
    )
    return 0 if published > 0 and tracker.errors == 0 and write_errors == 0 else 1


_EPILOG = (
    "examples:\n"
    "  python -m vision.scene --print --seconds 3\n"
    "  python -m vision.scene --out runs/live/scene.json\n"
    "      terminal 1, until Ctrl+C: the --scene-file of arm.replay --real and arm.real_robot --real\n"
    "      --carry-test (left with every item unseen when it stops); its provenance sidecar\n"
    "      runs/live/scene.json.source.json says live or fake (--real refuses fake) and is removed at stop\n"
    f"  python -m vision.scene --accuracy {EXAMPLE_LAYOUT}\n"
    f"  python -m vision.scene --fake --accuracy {EXAMPLE_LAYOUT} --out runs/check/accuracy-try.json\n"
    f"layout file (for example {EXAMPLE_LAYOUT}, tape-measured, meters and radians; the numbers below\n"
    "only illustrate the shape):\n"
    '  {"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3},\n'
    '             "pill_bottle": {"x": 0.18, "y": -0.12, "yaw": null},\n'
    '             "hand": {"x": 0.28, "y": -0.08, "yaw": null}}}\n'
    "yaw is required for objects and obstacles (null = not checked, for cylinders); the hand has none.\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI (see the module docstring); returns the exit code: 0 ok/pass, 1 fail, 2 bad input."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m vision.scene",
        description="Print or write the scene state (B6), or check it against tape measurements.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--print", action="store_true", help="print one scene JSON line per frame at 10 Hz")
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="with --print or alone: rewrite FILE atomically with every published scene (never inside "
        "config/, skills/, schemas/ or tests/; an existing FILE must be empty or a scene) and write its "
        "provenance sidecar FILE.source.json (live or fake) before the first scene; with "
        "--accuracy: the report path, a new file (default runs/table-calibration/accuracy-<UTC "
        "stamp>.json)",
    )
    parser.add_argument(
        "--seconds", type=float, default=None, help="--print/--out: stop after this long (default: Ctrl+C)"
    )
    parser.add_argument("--hz", type=float, default=None, help="--print/--out: tracker rate (default 10)")
    parser.add_argument(
        "--accuracy",
        metavar="LAYOUT.json",
        help=f"B6 check: compare a snapshot with a tape-measured layout file, for example {EXAMPLE_LAYOUT} "
        "(exit 0 within 1.5 cm and 10 deg, 1 outside, 2 bad input or no snapshot)",
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="--accuracy: frames in the snapshot (default 10)"
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=None,
        metavar="S",
        help=f"--accuracy: seconds the camera runs before the snapshot (default {WARMUP_S:g})",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="explicit offline run of the real tracker on the FakeCamera (config/fake_scene.json); every "
        "output is labelled synthetic",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if args.seconds is not None and not (math.isfinite(args.seconds) and args.seconds > 0):
        parser.error("--seconds must be a finite duration above 0")
    if args.hz is not None and not (math.isfinite(args.hz) and args.hz > 0):
        parser.error("--hz must be a finite rate above 0")
    if args.accuracy is not None:
        if args.print:
            parser.error("--accuracy cannot be combined with --print")
        if args.seconds is not None or args.hz is not None:
            parser.error("--accuracy takes one snapshot: --seconds and --hz do not apply")
        frames = 10 if args.frames is None else args.frames
        if frames < 1:
            parser.error("--frames must be at least 1")
        warmup = WARMUP_S if args.warmup is None else args.warmup
        if not (math.isfinite(warmup) and warmup >= 0):
            parser.error("--warmup must be a finite number of seconds >= 0")
        return _accuracy_main(args, frames, warmup)
    if args.frames is not None or args.warmup is not None:
        parser.error("--frames and --warmup only apply to --accuracy")
    if not args.print and args.out is None:
        parser.print_help()
        return 2
    return _stream_main(args)


if __name__ == "__main__":
    sys.exit(main())
