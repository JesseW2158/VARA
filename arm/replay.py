"""Trajectory validation and replay: the only way a checked path reaches the real arm (S0, A9, A13).

`validate(traj, limits, fk, box)` returns every problem; an empty list means accepted. It checks:
- the trajectory against schemas/trajectory.json (every schema error is listed);
- every waypoint has 5 finite numbers and lies within the joint limits minus 3 degrees;
- consecutive waypoints inside one `path` segment differ by at most 2 degrees per joint (60 deg/s at
  30 Hz); jumps between segments are left to replay's bridging move;
- every waypoint's tip, from `fk(q) -> (x, y, z)`, lies inside the workspace box;
- `live` segments only name `carry_to_hand` or `home`.
A malformed `limits` or `box` entry (missing, non-finite, inverted, too narrow for the margin) is reported
as a problem rather than raised, and a waypoint value too large for a float is "not a finite number".
`validate` never mutates its input.

`replay(traj, ...)` streams a validated trajectory to the driver (docs/interfaces.md, "Trajectory"):
1. `driver.read()` first; a RuntimeError/ValueError from the driver freezes the arm and returns `error`.
2. `validate(traj, load_limits(), arm.ik.fk_tip, load_box())`; any problem returns `error` with the problems
   in `events` and nothing commanded.
3. Per segment: a `path` bridges to its first waypoint (`Session.plan_path`: the joint-space sweep at
   <= 1 degree per tick when its tip stays inside the box, else a straight tip line solved with IK every
   LINE_SAMPLE_M, both validated against the limits and the box) when the arm is more than 1 degree away,
   then streams the waypoints at 30 Hz with `driver.command(q, current_gripper)`; a `grip` close uses
   `driver.close_gripper_hold()`, which waits until the jaws stop (at least GRIP_SETTLE_S; a close still
   moving at the driver's timeout counts as not holding), and compares "holding" with `expect_held` (a
   mismatch rises RISE_M, goes home and returns `grasp_missed`); a `grip` open commands gripper 1.0 and waits
   OPEN_SETTLE_S; a `live` segment runs `arm.real_robot.carry_to_hand` or `arm.real_robot.home`, and the
   next path bridges from wherever that left the arm. A sweep is planned from the present pose clamped into
   the limits minus 3 degrees, so an arm resting against a joint stop (its reading inside the margin) still
   moves back into the band at the driver's 2 degrees per tick.
4. Every command checks the STOP flag (`driver.stopped`, set by `freeze()`; the arm is already holding),
   the hand and the box: outside `carry_to_hand`, a wristband seen within HAND_FREEZE_M of the tip freezes
   the arm (`hand_too_close`); within HAND_SLOW_M the tip moves at most TIP_SPEED_NEAR_HAND, elsewhere at
   most TIP_SPEED_MAX, by splitting a waypoint over several ticks; a tick whose tip would be more than
   BOX_GATE_M outside the workspace box is never sent (the arm freezes, `error`), whatever planned it. A
   gripper close within HAND_FREEZE_M of a seen wristband is refused and aborts like a missed grasp with
   reason `hand_too_close` (the abort's own moves still run the hand check, so with the hand still there
   the arm simply freezes).
5. Any exception from the driver, IK or the scene source during a run freezes the arm and returns `error`
   (a scene source that returns anything but a dict, or a malformed hand entry, counts: a broken tracker
   never silently switches the hand check off). A failure before anything can move (config/limits.yaml,
   config/scene.yaml or arm.ik unavailable, a validate problem, torque off) returns `error` without
   freezing, and a driver that is already stopped returns `stopped`.
Returns `{"ok": bool, "reason": None | "grasp_missed" | "stopped" | "hand_too_close" | "error",
"events": [...]}`; `events` are short strings, each also passed to `on_event` when given.

`realtime` defaults to `not driver.is_fake`; when False no sleep happens and the clock is simulated, so
tests run instantly and deterministically (inject `clock`/`sleep` to observe the timing). `Pacer` and
`Session` are shared with `arm/real_robot.py`, whose primitives use the same checks and caps.

    python -m arm.replay --traj file.json            # validate only
    python -m arm.replay --fake --traj file.json     # replay on the in-memory arm, no sleeping
    python -m arm.replay --real --traj file.json     # the real arm: confirmation, 3 s countdown, Esc freezes
    python -m arm.replay --real --traj file.json --tracker --record place_in_tray   # live hand, tallied

`--scene-file scene.json` (a JSON scene re-read every tick) gives the hand check its data; a file that
cannot be read or is not a JSON object stops the CLI before anything connects. With `--real` the file
must come from a running `python -m vision.scene --out scene.json` on the real overhead camera
(`open_scene_file` -> `scene_file_reader(..., require_live=True)`, shared with arm.real_robot and
rl.policy_runner): the provenance sidecar scene.json.source.json that the writer puts down before its
first scene must say `writer: vision.scene`, `source: live`, `is_fake: false` (`check_scene_source`), and
the file must show a scene whose `t` is fresh (SCENE_SEEN_TIMEOUT_S plus one frame period) within
TRACKER_FRAME_S; otherwise (no sidecar, unreadable, another writer, a FakeCamera or FakeSceneTracker
writer, a leftover file) the CLI prints the reason and "nothing moved" and exits 2 before any
confirmation prompt and before a Driver exists. During the run a scene older than that reads as all
unseen (`unseen_copy`, as the live tracker reports a quiet camera), and a sidecar that disappears or
changes (the writer stopped, restarted or was replaced) raises from the scene source, which freezes the
arm (`error`). `--fake` needs no sidecar and reads the file as before. `--tracker` (exclusive
with `--scene-file`) takes the hand from the live overhead SceneTracker instead (`tracker_scene`:
`vision.scene.get_tracker()`, imported lazily, its `latest` as the scene callable, closed when the run
ends); with `--real` a tracker whose `is_fake` is not False (FAKE_CAMERA=1 / FAKE=1) or whose camera is
a fake, one that cannot be built (the camera does not open, no config/homography.npy) or one whose camera
delivers no new frame within TRACKER_FRAME_S (a publication after a missed grab does not count) is
refused with exit 2 before any confirmation prompt and before a Driver exists, and with `--fake` the
FakeSceneTracker serves config/fake_scene.json. An Esc that lands while `Driver.connect()` runs (which
ends by clearing the driver's stopped flag) is caught right after it (`arm.real_robot.esc_during_connect`):
the arm is frozen again and nothing is replayed.

`--record NAME` (with `--fake` or `--real`; NAME a safe file stem, `checked_record_name`) keeps the
evidence for the 4-of-5 real replay counts of A9 and A12 in `runs/calibration/replays-NAME.jsonl`
(VARA_RUNS_DIR moves runs/; appended to, never overwritten). Before anything moves, `prepare_record`
creates the folder and tallies the lines already there (exit 2 when the file is corrupt, is not a file
or its folder cannot be created, so a real attempt is not spent on a tally that cannot be kept) and the
CLI says which attempt this is. After each replay `record_attempt` appends one JSON line with the label,
the trajectory path, sha256 and skill, mode (`fake` or `real`), ok, reason, event count, elapsed seconds
and the local time, and the CLI prints the running tally `NAME: k of n ok` computed from that file
(`tally`); a line that cannot be written after the replay is reported and exits 1. A validation
failure, a declined confirmation or an Esc before connecting is not an attempt and leaves no line.

Nothing here imports `agents/`, `rl/` or `web/`, and `vision/` only inside `tracker_scene` for
`--tracker`; the module imports without MuJoCo (arm.ik is imported when a replay starts).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from arm import driver as driver_mod

__all__ = [
    "BOX_GATE_M",
    "BRIDGE_STEP_RAD",
    "BRIDGE_TOL_RAD",
    "GRIP_SETTLE_S",
    "HAND_FREEZE_M",
    "HAND_SLOW_M",
    "JOINTS",
    "LINE_JUMP_RAD",
    "LINE_SAMPLE_M",
    "LIVE_PRIMITIVES",
    "MARGIN_RAD",
    "MAX_STEP_RAD",
    "OPEN_SETTLE_S",
    "PLAN_INSET_M",
    "REASONS",
    "RECORD_NAME_RE",
    "RISE_M",
    "SCENE_POLL_S",
    "SCENE_SEEN_TIMEOUT_S",
    "SCENE_SOURCE_SUFFIX",
    "SCENE_SOURCE_WRITER",
    "STEP_TOL",
    "TICK",
    "TIP_SPEED_MAX",
    "TIP_SPEED_NEAR_HAND",
    "TRACKER_FRAME_S",
    "Abort",
    "CloseRefused",
    "Pacer",
    "SceneFileRefused",
    "Session",
    "bridge",
    "check_scene_source",
    "checked_record_name",
    "close_tracker",
    "inside_box",
    "load_box",
    "load_limits",
    "load_objects",
    "open_scene_file",
    "prepare_record",
    "record_attempt",
    "record_path",
    "replay",
    "scene_file_reader",
    "scene_source_path",
    "shrunk_box",
    "tally",
    "tracker_scene",
    "unseen_copy",
    "validate",
    "validate_path",
]

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "trajectory.json"
LIMITS_PATH = ROOT / "config" / "limits.yaml"
SCENE_PATH = ROOT / "config" / "scene.yaml"
OBJECTS_PATH = ROOT / "config" / "objects.yaml"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICK = 1 / 30
MARGIN_RAD = math.radians(3.0)  # waypoints stay this far inside the joint limits
MAX_STEP_RAD = math.radians(2.0)  # per joint between consecutive waypoints of one path (60 deg/s at 30 Hz)
BRIDGE_STEP_RAD = math.radians(1.0)  # bridging moves run at half speed (30 deg/s at 30 Hz)
BRIDGE_TOL_RAD = math.radians(1.0)  # bridge when any joint is further than this from the first waypoint
STEP_TOL = 1e-9  # float noise allowance on every inclusive comparison
LIVE_PRIMITIVES = ("carry_to_hand", "home")
REASONS = ("grasp_missed", "stopped", "hand_too_close", "error")

# Safety layer (docs/safety.md). Distances are from the wristband marker to the tip site of arm.ik.
HAND_FREEZE_M = 0.10  # a seen wristband this close to the tip freezes the arm (outside carry_to_hand)
HAND_SLOW_M = 0.15  # within this distance the tip speed is capped at TIP_SPEED_NEAR_HAND
TIP_SPEED_MAX = 0.2  # m/s, always
TIP_SPEED_NEAR_HAND = 0.1  # m/s
GRIP_SETTLE_S = 0.4  # a grip close waits at least this long, then until the jaws stop (driver timeout)
OPEN_SETTLE_S = 0.3
RISE_M = 0.05  # the rise after a missed grasp
CEILING_CLEARANCE_M = 0.005  # a rise stops this far under the box's top
MAX_COMMANDS_PER_WAYPOINT = 1000  # a waypoint that needs more ticks than this is a bug, not a slow move
CAP_FIT_ITERS = 4  # refinements of a sub-step's joint fraction until its measured tip step fits the cap

# Planning (docs/safety.md: the tip never leaves the workspace box, never goes above 0.30 m). A joint-space
# sweep between a folded and an extended pose arcs the tip upward, so when the sweep leaves the box the
# planner moves the tip on a straight line instead, solving IK every LINE_SAMPLE_M of tip travel.
LINE_SAMPLE_M = 0.005
LINE_JUMP_RAD = math.radians(20.0)  # a bigger joint change between two line samples is a reconfiguration
PLAN_INSET_M = 0.001  # planned lines keep the tip this far inside the box (sub-steps never cross a face)
BOX_GATE_M = 0.001  # a tick whose tip is further outside the box than this is never commanded

# CLI (--tracker, --record)
TRACKER_FRAME_S = 5.0  # --real --tracker is refused when the overhead camera publishes no frame in this
# (and a --real --scene-file when its file shows no fresh scene in this)
# --real --scene-file: the provenance sidecar of `python -m vision.scene --out FILE`. These mirror
# vision.scene's SOURCE_SUFFIX, SOURCE_WRITER and SEEN_TIMEOUT_S (tests/test_scene_provenance.py checks
# that they agree); vision/ is not imported here because it needs cv2.
SCENE_SOURCE_SUFFIX = ".source.json"
SCENE_SOURCE_WRITER = "vision.scene"
SCENE_SEEN_TIMEOUT_S = 0.5  # a scene older than this plus one frame period reads as all unseen
SCENE_POLL_S = 0.05  # the --real startup re-reads a stale scene file this often while it waits
RECORD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # --record NAME: a safe file stem

_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)

Fk = Callable[[list[float]], Sequence[float]]
Limits = dict[str, Sequence[float]]
Box = dict[str, Sequence[float]]
Vec3 = tuple[float, float, float]


# ------------------------------------------------------------------------------------------------------------
# validate
# ------------------------------------------------------------------------------------------------------------


def validate(traj: Any, limits: Limits, fk: Fk, box: Box) -> list[str]:
    """Return every problem with the trajectory; an empty list means it may be replayed.

    `limits` maps each joint name to its [low, high] in radians, `fk(q)` returns the tip (x, y, z) in
    meters for 5 joint angles, and `box` is `{"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}` (the
    workspace from config/scene.yaml). The input is never modified.
    """
    problems: list[str] = []
    for error in sorted(_VALIDATOR.iter_errors(traj), key=lambda e: e.json_path):
        problems.append(_schema_problem(error))
    if not isinstance(traj, dict) or not isinstance(traj.get("segments"), list):
        return problems

    ranges = _safe_ranges(limits, problems)
    safe_box = _safe_box(box, problems)
    for index, segment in enumerate(traj["segments"]):
        if not isinstance(segment, dict):
            continue  # already a schema error
        kind = segment.get("kind")
        if kind == "path":
            _check_path(f"segment {index}", segment.get("q"), ranges, fk, safe_box, problems)
        elif kind == "live":
            primitive = segment.get("primitive")
            if primitive not in LIVE_PRIMITIVES:
                allowed = ", ".join(LIVE_PRIMITIVES)
                problems.append(
                    f"segment {index}: live primitive {primitive!r} is not allowed; only {allowed}"
                )
    return problems


def validate_path(waypoints: Sequence[Sequence[float]], limits: Limits, fk: Fk, box: Box) -> list[str]:
    """The waypoint checks of `validate` (shape, limits minus 3 degrees, 2 degree steps, tip in the box)
    for a bare list of joint vectors: bridging moves, the rise after a missed grasp, `home()` sweeps."""
    problems: list[str] = []
    ranges = _safe_ranges(limits, problems)
    safe_box = _safe_box(box, problems)
    if not isinstance(waypoints, (list, tuple)):
        problems.append(f"path: expected a list of waypoints, got {type(waypoints).__name__}")
        return problems
    _check_path("path", list(waypoints), ranges, fk, safe_box, problems)
    return problems


def _schema_problem(error: ValidationError) -> str:
    detail = best_match([error]) or error  # descends into oneOf alternatives to the most specific error
    message = detail.message
    if len(message) > 300:
        message = message[:297] + "..."
    return f"schema: {detail.json_path}: {message}"


def _short(value: Any, width: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    """True for an int or float that converts to a finite float (a 400-digit int does not)."""
    if not _is_number(value):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _finite_pair(pair: Any) -> tuple[float, float] | None:
    """`(low, high)` as finite floats, or None when `pair` is not two finite numbers."""
    try:
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            return None
        low, high = float(pair[0]), float(pair[1])
    except (TypeError, ValueError, OverflowError, IndexError, KeyError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return low, high


def _safe_ranges(limits: Limits, problems: list[str]) -> dict[str, tuple[float, float]]:
    """Joint -> (low + 3 deg, high - 3 deg); a joint with no usable range is reported and skipped."""
    ranges: dict[str, tuple[float, float]] = {}
    for joint in JOINTS:
        pair = limits.get(joint) if isinstance(limits, dict) else None
        if pair is None:
            problems.append(f"limits: no [low, high] range for joint {joint!r}")
            continue
        finite = _finite_pair(pair)
        if finite is None:
            problems.append(f"limits: {joint} range {_short(pair)} is not a pair of finite numbers")
            continue
        low, high = finite
        if low >= high:
            problems.append(f"limits: {joint} range [{low}, {high}] is inverted (low >= high)")
            continue
        if high - low <= 2 * MARGIN_RAD:
            problems.append(
                f"limits: {joint} range [{low}, {high}] leaves no room inside the 3 degree margin"
            )
            continue
        ranges[joint] = (low + MARGIN_RAD, high - MARGIN_RAD)
    return ranges


def _safe_box(box: Box, problems: list[str]) -> dict[str, tuple[float, float]]:
    """Axis -> (low, high) of the workspace box; a malformed axis is reported and skipped."""
    safe: dict[str, tuple[float, float]] = {}
    for axis in "xyz":
        pair = box.get(axis) if isinstance(box, dict) else None
        finite = _finite_pair(pair) if pair is not None else None
        if finite is None:
            problems.append(f"box: no finite [low, high] range for axis {axis!r} ({_short(pair)})")
            continue
        low, high = finite
        if low >= high:
            problems.append(f"box: axis {axis!r} range [{low}, {high}] is inverted (low >= high)")
            continue
        safe[axis] = (low, high)
    return safe


def _check_path(
    where: str,
    waypoints: Any,
    ranges: dict[str, tuple[float, float]],
    fk: Fk,
    safe_box: dict[str, tuple[float, float]],
    problems: list[str],
) -> None:
    if not isinstance(waypoints, list):
        return  # already a schema error
    previous: Sequence[float] | None = None
    for k, waypoint in enumerate(waypoints):
        here = f"{where} waypoint {k}"
        well_formed = _check_waypoint(here, waypoint, problems)
        if well_formed:
            _check_limits(here, waypoint, ranges, problems)
            _check_tip(here, waypoint, fk, safe_box, problems)
            if previous is not None:
                _check_step(where, k, previous, waypoint, problems)
        previous = waypoint if well_formed else None


def _check_waypoint(where: str, waypoint: Any, problems: list[str]) -> bool:
    if not isinstance(waypoint, (list, tuple)):
        problems.append(
            f"{where}: expected a list of {len(JOINTS)} joint angles, got {type(waypoint).__name__}"
        )
        return False
    if len(waypoint) != len(JOINTS):
        problems.append(f"{where}: expected {len(JOINTS)} joint angles, got {len(waypoint)}")
        return False
    well_formed = True
    for joint, value in zip(JOINTS, waypoint, strict=True):
        if not _is_finite_number(value):
            problems.append(f"{where}: {joint} is not a finite number ({_short(value)})")
            well_formed = False
    return well_formed


def _check_limits(
    where: str, waypoint: Sequence[float], ranges: dict[str, tuple[float, float]], problems: list[str]
) -> None:
    for joint, value in zip(JOINTS, waypoint, strict=True):
        safe = ranges.get(joint)
        if safe is None:
            continue
        low, high = safe
        if value < low - STEP_TOL or value > high + STEP_TOL:
            problems.append(
                f"{where}: {joint} = {value:.4f} rad is outside [{low:.4f}, {high:.4f}] (limits minus 3 deg)"
            )


def _check_step(
    where: str, k: int, previous: Sequence[float], waypoint: Sequence[float], problems: list[str]
) -> None:
    for joint, a, b in zip(JOINTS, previous, waypoint, strict=True):
        step = abs(float(b) - float(a))
        if step > MAX_STEP_RAD + STEP_TOL:
            problems.append(
                f"{where} waypoints {k - 1}->{k}: {joint} steps {math.degrees(step):.2f} deg, "
                f"more than {math.degrees(MAX_STEP_RAD):.0f} deg per tick"
            )


def _check_tip(
    where: str,
    waypoint: Sequence[float],
    fk: Fk,
    safe_box: dict[str, tuple[float, float]],
    problems: list[str],
) -> None:
    try:
        tip = fk([float(v) for v in waypoint])  # a copy: fk may not touch the trajectory
    except Exception as exc:  # a crashing fk means the path cannot be trusted; report, do not raise
        problems.append(f"{where}: fk failed: {exc!r}")
        return
    try:
        x, y, z = float(tip[0]), float(tip[1]), float(tip[2])
    except (TypeError, ValueError, IndexError):
        problems.append(f"{where}: fk returned {_short(tip)}, not an (x, y, z) tip")
        return
    if not all(math.isfinite(v) for v in (x, y, z)):
        problems.append(f"{where}: fk returned a non-finite tip ({x}, {y}, {z})")
        return
    for axis, value in zip("xyz", (x, y, z), strict=True):
        if axis not in safe_box:
            continue  # the bad axis was already reported
        low, high = safe_box[axis]
        if value < low - STEP_TOL or value > high + STEP_TOL:
            problems.append(
                f"{where}: tip ({x:.3f}, {y:.3f}, {z:.3f}) leaves the workspace box: "
                f"{axis} not in [{low}, {high}]"
            )


# ------------------------------------------------------------------------------------------------------------
# bridge and config loaders
# ------------------------------------------------------------------------------------------------------------


def bridge(
    q_from: Sequence[float], q_to: Sequence[float], max_step_rad: float = BRIDGE_STEP_RAD
) -> list[list[float]]:
    """Joint-space waypoints from `q_from` to `q_to`, excluding `q_from` and ending exactly at `q_to`,
    with at most `max_step_rad` per joint between consecutive waypoints (one tick each at 30 Hz).

    Always returns at least `[q_to]`, so a caller can stream the result as-is.
    """
    if len(q_from) != len(q_to):
        raise ValueError(f"joint vectors differ in length: {len(q_from)} vs {len(q_to)}")
    if not max_step_rad > 0:
        raise ValueError(f"max_step_rad must be positive, got {max_step_rad!r}")
    start = [float(v) for v in q_from]
    end = [float(v) for v in q_to]
    if not all(math.isfinite(v) for v in start + end):
        raise ValueError(f"joint vectors must be finite, got {start} -> {end}")
    deltas = [b - a for a, b in zip(start, end, strict=True)]
    largest = max((abs(d) for d in deltas), default=0.0)
    steps = max(1, math.ceil(largest / max_step_rad))
    path = [[a + d * k / steps for a, d in zip(start, deltas, strict=True)] for k in range(1, steps)]
    path.append(end)
    return path


def load_limits(path: Path | str | None = None) -> dict[str, list[float]]:
    """Joint limits in radians from config/limits.yaml: `{joint: [low, high]}` for the 5 arm joints.

    Raises FileNotFoundError until Lane A writes the file (A6 calibrates it) and ValueError when a joint
    is missing or malformed. There is no silent fallback: the real arm must never be validated against
    guessed limits.
    """
    file = Path(path) if path is not None else LIMITS_PATH
    if not file.exists():
        raise FileNotFoundError(
            f"{file} is missing: Lane A writes config/limits.yaml (task A6 calibrates it)"
        )
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: expected a mapping of joint -> [low, high]")
    limits: dict[str, list[float]] = {}
    for joint, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue  # comments, gripper settings or other non-range entries
        low, high = float(pair[0]), float(pair[1])
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            raise ValueError(f"{file}: bad range for {joint!r}: {pair!r}")
        limits[str(joint)] = [low, high]
    missing = [joint for joint in JOINTS if joint not in limits]
    if missing:
        raise ValueError(f"{file}: no [low, high] range for {', '.join(missing)}")
    return limits


def load_box(path: Path | str | None = None) -> dict[str, list[float]]:
    """The workspace box from config/scene.yaml: `{"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}`."""
    file = Path(path) if path is not None else SCENE_PATH
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    workspace = raw.get("workspace") if isinstance(raw, dict) else None
    if not isinstance(workspace, dict):
        raise ValueError(f"{file}: no 'workspace' mapping")
    box: dict[str, list[float]] = {}
    for axis in "xyz":
        pair = workspace.get(axis)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{file}: workspace.{axis} must be [low, high]")
        low, high = float(pair[0]), float(pair[1])
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            raise ValueError(f"{file}: bad workspace.{axis} range {pair!r}")
        box[axis] = [low, high]
    return box


def load_objects(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """config/objects.yaml by object name: `{name: {kind, size, h, held_min, ...}}` (markers dropped)."""
    file = Path(path) if path is not None else OBJECTS_PATH
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for entry in raw.values():
            if isinstance(entry, dict) and entry.get("name"):
                out[str(entry["name"])] = dict(entry)
    return out


def shrunk_box(box: Box, inset: float) -> dict[str, list[float]]:
    """The workspace box with every face moved `inset` meters inward (a negative inset widens it)."""
    return {axis: [float(box[axis][0]) + inset, float(box[axis][1]) - inset] for axis in "xyz"}


def inside_box(point: Sequence[float], box: Box, tol: float = 0.0) -> bool:
    """True when `point` is inside `box`, allowing `tol` meters beyond each face."""
    return all(
        float(box[axis][0]) - tol <= float(v) <= float(box[axis][1]) + tol
        for axis, v in zip("xyz", point, strict=True)
    )


# ------------------------------------------------------------------------------------------------------------
# Replay machinery shared with arm/real_robot.py
# ------------------------------------------------------------------------------------------------------------


class Abort(Exception):
    """Ends a replay or primitive with `reason` (one of REASONS). Whoever raises it has already frozen the
    arm when the reason calls for it (`stopped` means someone else froze it)."""

    def __init__(self, reason: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown abort reason {reason!r}")
        super().__init__(reason)
        self.reason = reason

    def result(self, events: list[str]) -> dict[str, Any]:
        return {"ok": False, "reason": self.reason, "events": events}


class CloseRefused(Abort):
    """A gripper close was refused because a wristband is within HAND_FREEZE_M of the tip."""

    def __init__(self) -> None:
        super().__init__("hand_too_close")


class Pacer:
    """30 Hz pacing on the wall clock or on a simulated one.

    `realtime=True` uses `time.perf_counter`/`time.sleep`; `realtime=False` keeps a simulated clock that
    every sleep advances, so a replay runs instantly and its timing stays deterministic. `clock`/`sleep`
    override either source (tests inject them to observe the timing).
    """

    def __init__(
        self,
        realtime: bool,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.realtime = bool(realtime)
        self._sim = 0.0
        self.now: Callable[[], float] = (
            clock if clock is not None else (time.perf_counter if self.realtime else self._sim_now)
        )
        self.sleep: Callable[[float], None] = (
            sleep if sleep is not None else (time.sleep if self.realtime else self._sim_sleep)
        )
        self._next = self.now()

    def _sim_now(self) -> float:
        return self._sim

    def _sim_sleep(self, seconds: float) -> None:
        self._sim += max(0.0, float(seconds))

    def tick(self) -> None:
        """Wait for the next 30 Hz slot (a slot more than one tick late resyncs instead of bursting)."""
        self._next += TICK
        delay = self._next - self.now()
        if delay > 0:
            self.sleep(delay)
        elif delay < -TICK:
            self._next = self.now()

    def wait(self, seconds: float) -> None:
        """A settle wait; the next tick starts after it."""
        if seconds > 0:
            self.sleep(seconds)
        self.resync()

    def resync(self) -> None:
        self._next = self.now()


def _ik() -> Any:
    """`arm.ik`, imported on first use so this module loads without MuJoCo."""
    from arm import ik

    return ik


def _lerp(a: Sequence[float], b: Sequence[float], frac: float) -> list[float]:
    if frac >= 1.0:
        return [float(v) for v in b]
    return [x + (y - x) * frac for x, y in zip(a, b, strict=True)]


class Session:
    """One replay or primitive run: the commanded pose and gripper, pacing, events and the safety checks.

    `driver` is a `Driver`/`FakeDriver`, `scene` a zero-argument callable returning the latest scene dict
    (None: no hand data). `hand_freeze=False` (inside `carry_to_hand`) keeps the speed cap near the hand
    but not the 10 cm freeze. `cancelled()` is the primitive cancel flag of `arm.real_robot`. Construction
    loads config/limits.yaml, config/scene.yaml and arm.ik and raises on any of them failing; nothing has
    touched the driver by then. `gate_box` is the box every tick is checked against (the workspace box;
    `arm.real_robot.home` widens it for an arm that starts outside the box).
    """

    def __init__(
        self,
        driver: Any,
        scene: Callable[[], dict | None] | None = None,
        pacer: Pacer | None = None,
        on_event: Callable[[str], None] | None = None,
        *,
        events: list[str] | None = None,
        hand_freeze: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.driver = driver
        self.scene_fn = scene
        self.pacer = pacer if pacer is not None else Pacer(not getattr(driver, "is_fake", False))
        self.on_event = on_event
        self.events: list[str] = events if events is not None else []
        self.hand_freeze = hand_freeze
        self.cancelled = cancelled
        self.ik = _ik()
        self.limits = load_limits()
        self.box = load_box()
        self.gate_box: Box = self.box
        self.q: list[float] = []
        self.g: float = 1.0
        self.tip: Vec3 = (0.0, 0.0, 0.0)

    # -- events and scene ----------------------------------------------------------------------------------
    def emit(self, text: str) -> None:
        self.events.append(text)
        log.info("%s", text)
        if self.on_event is not None:
            try:
                self.on_event(text)
            except Exception:
                log.exception("on_event failed for %r", text)

    def scene_dict(self) -> dict | None:
        """The latest scene dict, or None without a scene source. A source that returns anything but a
        dict is a broken tracker: ValueError, which ends the run with `error` (never a silent "no hand")."""
        if self.scene_fn is None:
            return None
        scene = self.scene_fn()
        if not isinstance(scene, dict):
            raise ValueError(f"the scene source returned {type(scene).__name__}, not a scene dict")
        return scene

    def hand(self) -> Vec3 | None:
        """The seen wristband's (x, y, z), or None when there is no hand data or it is not seen."""
        scene = self.scene_dict()
        if scene is None:
            return None
        hand = scene.get("hand")
        if not isinstance(hand, dict) or not hand.get("seen"):
            return None
        try:
            xyz = (float(hand["x"]), float(hand["y"]), float(hand["z"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed hand entry in the scene: {_short(hand, 80)}") from exc
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"non-finite hand position in the scene: {xyz}")
        return xyz

    # -- state -----------------------------------------------------------------------------------------------
    def start(self) -> None:
        """Read the present pose (the first thing every run does). A RuntimeError/ValueError from the driver
        freezes the arm and ends the run with `error`."""
        try:
            q, g = self.driver.read()
        except (RuntimeError, ValueError) as exc:
            self.driver.freeze()
            self.emit(f"error: driver.read failed: {exc}")
            raise Abort("error") from exc
        last = self.driver.last_command
        self.q = [float(v) for v in q]
        # keep commanding the last sent gripper value (a hold at the contact reading minus 3), not the
        # present reading, which for a held object is the contact reading itself
        self.g = float(last[1]) if last is not None else float(g)
        self.tip = self._tip_of(self.q)

    refresh = start

    def _tip_of(self, q: Sequence[float]) -> Vec3:
        x, y, z = self.ik.fk_tip(q)
        return (float(x), float(y), float(z))

    def within(self, q: Sequence[float], tol_rad: float) -> bool:
        return all(abs(a - float(b)) <= tol_rad + STEP_TOL for a, b in zip(self.q, q, strict=True))

    def band_clamp(self, q: Sequence[float]) -> list[float]:
        """`q` clamped into the joint limits minus the 3 degree margin (the band every waypoint must be in).
        A sweep starts from the clamped present pose: the driver walks the arm back into the band at its 2
        degrees per tick, and the planned waypoints all validate."""
        ranges = _safe_ranges(self.limits, [])
        out = []
        for joint, value in zip(JOINTS, q, strict=True):
            v = float(value)
            if joint in ranges:
                low, high = ranges[joint]
                v = min(max(v, low), high)
            out.append(v)
        return out

    # -- checks ----------------------------------------------------------------------------------------------
    def require_ready(self) -> None:
        """Before the first command: a driver that is already stopped ends the run with `stopped`, one with
        torque off with `error`; neither freezes, since nothing has moved."""
        if self.driver.stopped:
            self.emit("driver is stopped: resume first")
            raise Abort("stopped")
        if not self.driver.torque:
            self.emit("error: torque is off; torque_on first")
            raise Abort("error")

    def check_stopped(self) -> None:
        """STOP: the driver's stopped flag (the arm is already frozen) or the primitive cancel flag."""
        if self.driver.stopped or (self.cancelled is not None and self.cancelled()):
            if not self.driver.stopped:
                self.driver.freeze()
            self.emit("stopped")
            raise Abort("stopped")

    def check_hand(self, tip: Vec3, where: str) -> float | None:
        """Freeze when a seen wristband is within HAND_FREEZE_M of `tip` (unless `hand_freeze` is off).
        Returns the hand distance, or None without hand data."""
        hand = self.hand()
        if hand is None:
            return None
        distance = math.dist(tip, hand)
        if self.hand_freeze and distance <= HAND_FREEZE_M:
            self.driver.freeze()
            self.emit(f"hand {distance:.3f} m from the tip at {where}: frozen")
            raise Abort("hand_too_close")
        return distance

    def check_box(self, tip: Vec3, where: str) -> None:
        """The last line of the workspace rule: a tick whose tip lies more than BOX_GATE_M outside
        `gate_box` freezes the arm and ends the run with `error` (validated waypoints may sit on a face;
        the sub-steps between them deviate by micrometers, never by a millimeter)."""
        if not inside_box(tip, self.gate_box, BOX_GATE_M):
            self.driver.freeze()
            x, y, z = tip
            self.emit(f"error: tip ({x:.3f}, {y:.3f}, {z:.3f}) outside the workspace box at {where}: frozen")
            raise Abort("error")

    def tip_cap(self, tip_next: Vec3) -> float:
        """The largest tip step (m) allowed this tick: TIP_SPEED_NEAR_HAND within HAND_SLOW_M of a seen
        wristband (from the current tip or the next one), TIP_SPEED_MAX elsewhere."""
        hand = self.hand()
        if hand is not None and (
            math.dist(self.tip, hand) < HAND_SLOW_M or math.dist(tip_next, hand) < HAND_SLOW_M
        ):
            return TIP_SPEED_NEAR_HAND * TICK
        return TIP_SPEED_MAX * TICK

    # -- motion ----------------------------------------------------------------------------------------------
    def command(self, q: Sequence[float], gripper: float | None = None, *, where: str = "") -> None:
        """One tick: STOP, hand and box checks, `driver.command(q, gripper)`, then wait for the next slot."""
        self.check_stopped()
        q_list = [float(v) for v in q]
        tip = self._tip_of(q_list)
        self.check_hand(tip, where)
        self.check_box(tip, where)
        g = self.g if gripper is None else float(gripper)
        if not self.driver.command(q_list, g):
            if self.driver.stopped:
                self.emit(f"stopped at {where}")
                raise Abort("stopped")
            self.driver.freeze()
            self.emit(f"error: command refused at {where} (torque off?)")
            raise Abort("error")
        last = self.driver.last_command
        if last is not None:
            self.q, self.g = [float(v) for v in last[0]], float(last[1])
        else:
            self.q, self.g = q_list, g
        self.tip = tip if self.q == q_list else self._tip_of(self.q)
        self.pacer.tick()

    def move_to_waypoint(self, q_target: Sequence[float], *, where: str = "") -> None:
        """Command `q_target`, split over several ticks when the tip would move faster than the cap."""
        target = [float(v) for v in q_target]
        tip_target = self._tip_of(target)
        for _ in range(MAX_COMMANDS_PER_WAYPOINT):
            remaining = math.dist(self.tip, tip_target)
            cap = self.tip_cap(tip_target)
            if remaining <= cap * (1 + 1e-6):
                self.command(target, where=where)
                if self.within(target, 0.0):
                    return
                continue  # the driver clipped the step (its reference differs from ours): keep going
            frac = cap / remaining
            q_next = _lerp(self.q, target, frac)
            for _ in range(CAP_FIT_ITERS):
                # the joint-space fraction is not quite linear in the tip: fit the real step to the cap,
                # which is the slow one when this tick enters the slow zone around the hand
                tip_next = self._tip_of(q_next)
                step = math.dist(self.tip, tip_next)
                cap = self.tip_cap(tip_next)
                if step <= cap * (1 + 1e-6):
                    break
                frac *= cap / step
                q_next = _lerp(self.q, target, frac)
            self.command(q_next, where=where)
        self.driver.freeze()
        self.emit(f"error: {where} did not converge in {MAX_COMMANDS_PER_WAYPOINT} ticks")
        raise Abort("error")

    def stream(self, waypoints: Sequence[Sequence[float]], *, where: str = "path") -> None:
        """Stream validated waypoints, one tick each (more when the tip speed cap splits a step)."""
        for k, waypoint in enumerate(waypoints):
            self.move_to_waypoint(waypoint, where=f"{where} waypoint {k}")

    def line_path(
        self, q_from: Sequence[float], q_to: Sequence[float], *, max_step_rad: float | None = None
    ) -> tuple[list[list[float]] | None, str]:
        """Waypoints that move the tip on a straight line from `fk(q_from)` to `fk(q_to)`, blending the
        pitch and the wrist roll: IK (`arm.ik.solve`, pitch relaxed through its ladder where needed) every
        LINE_SAMPLE_M from the previous sample, consecutive samples bridged at <= `max_step_rad`, ending
        exactly at `q_to`. Returns (None, reason) when a sample is unreachable or two samples differ by
        more than LINE_JUMP_RAD (a joint reconfiguration whose sweep would leave the line). `max_step_rad`
        defaults to BRIDGE_STEP_RAD."""
        max_step_rad = BRIDGE_STEP_RAD if max_step_rad is None else max_step_rad
        start = [float(v) for v in q_from]
        end = [float(v) for v in q_to]
        (p0, pitch0), (p1, pitch1) = self.ik.fk(start), self.ik.fk(end)
        samples = max(1, math.ceil(math.dist(p0, p1) / LINE_SAMPLE_M))
        q_prev = start
        path: list[list[float]] = []
        for k in range(1, samples + 1):
            frac = k / samples
            point = tuple(a + (b - a) * frac for a, b in zip(p0, p1, strict=True))
            pitch = pitch0 + (pitch1 - pitch0) * frac
            q0 = list(q_prev)
            q0[4] = start[4] + (end[4] - start[4]) * frac  # solve copies the roll from q0
            q_sample, ok = self.ik.solve(point, pitch, q0=q0)
            if not ok:
                x, y, z = point
                return None, f"no IK solution on the straight line at ({x:.3f}, {y:.3f}, {z:.3f})"
            q_sample = [float(v) for v in q_sample]
            jump = max(abs(a - b) for a, b in zip(q_sample, q_prev, strict=True))
            if jump > LINE_JUMP_RAD:
                return None, f"joint reconfiguration of {math.degrees(jump):.0f} deg on the straight line"
            path.extend(bridge(q_prev, q_sample, max_step_rad))
            q_prev = q_sample
        if q_prev != end:
            path.extend(bridge(q_prev, end, max_step_rad))
        return path, f"straight line, {samples} samples"

    def plan_path(
        self,
        q_target: Sequence[float],
        *,
        start: Sequence[float] | None = None,
        max_step_rad: float | None = None,
        box: Box | None = None,
    ) -> tuple[list[list[float]] | None, list[str], str]:
        """A validated path from `start` (default: the present pose) to `q_target`, at most `max_step_rad`
        (default BRIDGE_STEP_RAD) per joint per tick: the joint-space sweep when every waypoint's tip stays
        inside `box` (default: the workspace box), else the straight tip line (validated PLAN_INSET_M inside
        the box). The start is
        clamped into the limits minus 3 degrees first (`band_clamp`). Returns (path, problems, how) with
        `how` "direct" or the line's description; `path` is None with the problems when nothing validates."""
        box = self.box if box is None else box
        max_step_rad = BRIDGE_STEP_RAD if max_step_rad is None else max_step_rad
        q_from = self.band_clamp(self.q if start is None else start)
        target = [float(v) for v in q_target]
        direct = bridge(q_from, target, max_step_rad)
        problems = validate_path(direct, self.limits, self.ik.fk_tip, box)
        if not problems:
            return direct, [], "direct"
        line, how = self.line_path(q_from, target, max_step_rad=max_step_rad)
        if line is not None:
            line_problems = validate_path(line, self.limits, self.ik.fk_tip, shrunk_box(box, PLAN_INSET_M))
            if not line_problems:
                return line, [], how
            how = line_problems[0]
        problems.append(f"straight line: {how}")
        return None, problems, ""

    def bridge_to(self, q_target: Sequence[float], *, where: str = "bridge") -> None:
        """Move to `q_target` at <= BRIDGE_STEP_RAD per tick along `plan_path` (sweep, else straight line)."""
        path, problems, how = self.plan_path(q_target)
        if path is None:
            for problem in problems[:5]:
                self.emit(f"{where} refused: {problem}")
            raise Abort("error")
        self.emit(f"{where}: {len(path)} steps" + ("" if how == "direct" else f" ({how})"))
        self.stream(path, where=where)

    def rise(self, dz: float, *, where: str = "rise") -> bool:
        """Lift the tip by `dz` (pitch kept, IK on the current tip, validated); False when skipped."""
        (x, y, z), pitch = self.ik.fk(self.q)
        z_up = min(z + dz, float(self.box["z"][1]) - CEILING_CLEARANCE_M)
        if z_up <= z + 1e-6:
            self.emit(f"{where} skipped: tip already at the ceiling")
            return False
        q_up, ok = self.ik.solve((x, y, z_up), pitch, q0=self.q)
        if not ok:
            self.emit(f"{where} skipped: no IK solution {dz * 100:.0f} cm above the tip")
            return False
        path = bridge(self.q, q_up, BRIDGE_STEP_RAD)
        problems = validate_path(path, self.limits, self.ik.fk_tip, self.box)
        if problems:
            self.emit(f"{where} skipped: {problems[0]}")
            return False
        self.emit(f"{where}: {(z_up - z) * 100:.1f} cm up")
        self.stream(path, where=where)
        return True

    # -- gripper ---------------------------------------------------------------------------------------------
    def settle(self, seconds: float) -> None:
        """Account for a wait the driver did itself (the real driver sleeps inside close_gripper_hold; the
        fake one does not, so the pacer waits instead)."""
        if getattr(self.driver, "is_fake", False):
            self.pacer.wait(seconds)
        else:
            self.pacer.resync()

    def open_gripper(self, *, where: str = "grip open") -> None:
        self.check_stopped()
        self.check_hand(self.tip, where)
        if not self.driver.command(self.q, 1.0):
            if self.driver.stopped:
                self.emit(f"stopped at {where}")
                raise Abort("stopped")
            self.driver.freeze()
            self.emit(f"error: command refused at {where} (torque off?)")
            raise Abort("error")
        self.g = 1.0
        self.pacer.wait(OPEN_SETTLE_S)
        self.check_stopped()
        self.check_hand(self.tip, where)

    def close_gripper(self, obj: str | None, expect_held: bool | None) -> tuple[float, bool]:
        """Close with the driver's hold and decide "holding": the FakeDriver reports `expect_held` unless its
        `gripper_reading_override` is set; otherwise the reading must exceed the empty-closed reading by the
        object's `held_min`. Raises CloseRefused (nothing closes) when a wristband is within HAND_FREEZE_M."""
        self.check_stopped()
        hand = self.hand()
        if hand is not None:
            distance = math.dist(self.tip, hand)
            if distance <= HAND_FREEZE_M:
                self.emit(f"grip close refused: hand {distance:.3f} m from the tip")
                raise CloseRefused()

        def per_poll() -> None:  # the per-tick checks keep running while the jaws travel
            self.check_stopped()
            self.check_hand(self.tip, "grip close")

        reading = float(self.driver.close_gripper_hold(settle=GRIP_SETTLE_S, on_poll=per_poll))
        self.settle(GRIP_SETTLE_S)
        self.check_stopped()
        last = self.driver.last_command
        self.g = float(last[1]) if last is not None else reading
        override = getattr(self.driver, "gripper_reading_override", None)
        if getattr(self.driver, "grip_status", "still") == "timeout":
            # The jaws were still moving when the driver gave up waiting: the reading is mid-travel and says
            # nothing about an object, so the grasp is unconfirmed and treated as a miss.
            self.emit("grip close: the jaws were still moving after the wait; treated as not holding")
            holding = False
        elif getattr(self.driver, "is_fake", False) and override is None:
            holding = True if expect_held is None else bool(expect_held)
        else:
            # "above the empty-closed reading by at least held_min": the threshold itself counts as holding
            holding = reading >= self.empty_closed() + self.held_threshold(obj) - STEP_TOL
        self.check_hand(self.tip, "grip close")
        return reading, holding

    def held_threshold(self, obj: str | None) -> float:
        """The object's `held_min` in our 0-1 gripper units: `arm.driver.held_threshold`, which keeps
        LeRobot's 0-100 scale inside the driver (hard rule 3). An unknown object uses the smallest
        `held_min` on file."""
        return float(driver_mod.held_threshold(obj))

    def empty_closed(self) -> float:
        """The gripper reading (0-1) of an empty close: `arm.driver.empty_closed_reading()` (0.0 until A11
        records `gripper.empty_closed_pct`)."""
        return float(driver_mod.empty_closed_reading())


# ------------------------------------------------------------------------------------------------------------
# replay
# ------------------------------------------------------------------------------------------------------------


def replay(
    traj: dict,
    *,
    driver: Any = None,
    scene: Callable[[], dict | None] | None = None,
    realtime: bool | None = None,
    on_event: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Replay a trajectory on the arm (see the module docstring for the rules).

    `driver` defaults to `arm.driver.get_driver()`; `scene` is a zero-argument callable returning the
    latest scene dict (None: no hand data, so no hand check); `realtime` defaults to `not driver.is_fake`;
    `on_event(text)` receives every event; `clock`/`sleep` override the time sources (tests).
    Returns `{"ok", "reason", "events"}` and never raises for driver, IK or scene failures.
    """
    drv = driver if driver is not None else driver_mod.get_driver()
    rt = (not getattr(drv, "is_fake", False)) if realtime is None else bool(realtime)
    events: list[str] = []
    try:
        session = Session(drv, scene, Pacer(rt, clock, sleep), on_event, events=events, hand_freeze=True)
    except Exception as exc:  # config or arm.ik unavailable: nothing has moved, nothing to freeze
        return setup_failure(events, on_event, exc)
    try:
        session.start()
        problems = validate(traj, session.limits, session.ik.fk_tip, session.box)
        if problems:
            for problem in problems:
                session.emit(f"validate: {problem}")
            raise Abort("error")
        segments = traj["segments"]
        session.emit(f"validated: {len(segments)} segments")
        session.require_ready()
        for index, segment in enumerate(segments):
            kind = segment["kind"]
            if kind == "path":
                _run_path(session, index, segment)
            elif kind == "grip":
                _run_grip(session, index, segment)
            else:
                _run_live(session, index, segment)
        session.emit("done")
        return {"ok": True, "reason": None, "events": events}
    except Abort as abort:
        return abort.result(events)
    except Exception as exc:
        drv.freeze()
        session.emit(f"error: {type(exc).__name__}: {exc}")
        log.exception("replay failed; arm frozen")
        return {"ok": False, "reason": "error", "events": events}


def setup_failure(
    events: list[str], on_event: Callable[[str], None] | None, exc: BaseException
) -> dict[str, Any]:
    """The `error` result for a failure before a Session exists (config files, arm.ik): the event is
    recorded and passed to `on_event`, nothing has moved and nothing is frozen."""
    message = f"error: {type(exc).__name__}: {exc}"
    events.append(message)
    log.error("setup failed before anything moved: %s", message)
    if on_event is not None:
        try:
            on_event(message)
        except Exception:
            log.exception("on_event failed for %r", message)
    return {"ok": False, "reason": "error", "events": events}


def _run_path(session: Session, index: int, segment: dict) -> None:
    waypoints = segment["q"]
    first = [float(v) for v in waypoints[0]]
    if not session.within(first, BRIDGE_TOL_RAD):
        session.bridge_to(first, where=f"segment {index} bridge")
    session.emit(f"segment {index}: path, {len(waypoints)} waypoints")
    session.stream(waypoints, where=f"segment {index}")


def _run_grip(session: Session, index: int, segment: dict) -> None:
    obj = segment.get("object")
    label = f"segment {index}: grip {segment['action']}" + (f" {obj}" if obj else "")
    if segment["action"] == "open":
        session.open_gripper(where=label)
        session.emit(label)
        return
    expect = segment.get("expect_held")
    try:
        reading, holding = session.close_gripper(obj, expect)
    except CloseRefused as refused:
        _abort_grasp(session, refused.reason)
    session.emit(f"{label}: reading {reading:.2f}, holding={holding}")
    if expect is not None and holding != bool(expect):
        _abort_grasp(session, "grasp_missed")


def _abort_grasp(session: Session, reason: str) -> None:
    """A missed grasp or a refused close: rise RISE_M, go home, end the replay with `reason` (or with the
    reason of whatever stopped the abort itself)."""
    from arm import real_robot  # lazy: real_robot imports this module

    session.emit(f"{reason}: rising {RISE_M * 100:.0f} cm and going home")
    session.rise(RISE_M, where="rise")
    pacer = session.pacer
    result = real_robot.home(
        driver=session.driver,
        scene=session.scene_fn,
        on_event=session.on_event,
        clock=pacer.now,
        sleep=pacer.sleep,
        realtime=pacer.realtime,
    )
    session.events.extend(result["events"])  # home() already passed each through on_event
    if not result["ok"]:
        raise Abort(result["reason"])
    raise Abort(reason)


def _run_live(session: Session, index: int, segment: dict) -> None:
    from arm import real_robot  # lazy: real_robot imports this module

    primitive = segment["primitive"]
    obj = segment.get("object")
    pacer = session.pacer
    session.emit(f"segment {index}: live {primitive}" + (f" {obj}" if obj else ""))
    common = dict(
        driver=session.driver,
        scene=session.scene_fn,
        on_event=session.on_event,
        clock=pacer.now,
        sleep=pacer.sleep,
        realtime=pacer.realtime,
    )
    if primitive == "carry_to_hand":
        result = real_robot.carry_to_hand(obj, **common)
    else:
        result = real_robot.home(**common)
    session.events.extend(result["events"])  # the primitive already passed each through on_event
    if not result["ok"]:
        raise Abort(result["reason"])
    session.refresh()  # the next path bridges from wherever the primitive left the arm


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


class SceneFileRefused(ValueError):
    """A `--scene-file` that must not feed the real arm (`scene_file_reader(..., require_live=True)`): its
    provenance sidecar is missing, unreadable, not vision.scene's or not "live", or the file is not being
    refreshed. A ValueError, so a caller that only catches OSError/ValueError still refuses."""


def scene_source_path(path: Path | str) -> Path:
    """The provenance sidecar of a scene file: FILE + ".source.json" (`vision.scene.source_path`)."""
    return Path(str(path) + SCENE_SOURCE_SUFFIX)


def _source_signature(sidecar: Path) -> tuple[int, int, int]:
    """What changes when the sidecar is rewritten (os.replace gives a new file): mtime, size and file id."""
    st = sidecar.stat()
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def check_scene_source(path: Path | str) -> dict[str, Any]:
    """The provenance sidecar of scene file `path` (`scene_source_path`) when it proves a live writer:
    a JSON object written by vision.scene (`writer`), with `source` "live", `is_fake` False, a str
    `tracker` and `started_at`, an int `pid` and a positive `hz`. Anything else raises SceneFileRefused
    naming the reason (missing, unreadable, not JSON, another writer, malformed, or "fake"). Reads only."""
    sidecar = scene_source_path(path)
    try:
        text = sidecar.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SceneFileRefused(
            f"no provenance sidecar {sidecar.name}: only a running `python -m vision.scene --out "
            f"{Path(path).name}` writes one (and removes it when it stops), so nothing proves this file "
            "is live"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SceneFileRefused(f"the provenance sidecar {sidecar} cannot be read ({exc})") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SceneFileRefused(f"the provenance sidecar {sidecar} is not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SceneFileRefused(f"the provenance sidecar {sidecar} is not a JSON object: {_short(data)}")
    if data.get("writer") != SCENE_SOURCE_WRITER:
        raise SceneFileRefused(
            f"the provenance sidecar {sidecar.name} was not written by {SCENE_SOURCE_WRITER} "
            f"(writer {_short(data.get('writer'))})"
        )
    for key, kinds in (("source", str), ("tracker", str), ("started_at", str), ("is_fake", bool)):
        if not isinstance(data.get(key), kinds):
            raise SceneFileRefused(f"the provenance sidecar {sidecar.name} has no valid {key!r}")
    pid, hz = data.get("pid"), data.get("hz")
    if not isinstance(pid, int) or isinstance(pid, bool):
        raise SceneFileRefused(f"the provenance sidecar {sidecar.name} has no valid 'pid'")
    if not (_is_finite_number(hz) and hz > 0):
        raise SceneFileRefused(f"the provenance sidecar {sidecar.name} has no valid 'hz'")
    if data["source"] != "live" or data["is_fake"] is not False:
        raise SceneFileRefused(
            f"{sidecar.name} says the scene is {data['source'][:40]} ({data['tracker'][:40]} on camera "
            f"{_short(data.get('camera'))}, is_fake {data['is_fake']}, pid {pid}): a fake scene never feeds "
            "the real arm"
        )
    return data


def _scene_age(scene: Mapping[str, Any], now: float) -> float | None:
    """Seconds since the scene's `t` (wall clock), None when `t` is not a finite number."""
    t = scene.get("t")
    return float(now - t) if _is_finite_number(t) else None


def _is_stale(age: float | None, stale_after: float) -> bool:
    return age is None or not (-stale_after <= age <= stale_after)


def unseen_copy(scene: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy of `scene` with every object, obstacle and the hand `seen: false` (poses and `t` kept):
    what the live tracker reports once nothing has been detected for its seen timeout."""
    out = copy.deepcopy(dict(scene))
    for group in ("objects", "obstacles"):
        items = out.get(group)
        for item in items.values() if isinstance(items, dict) else ():
            if isinstance(item, dict):
                item["seen"] = False
    hand = out.get("hand")
    if isinstance(hand, dict):
        hand["seen"] = False
    return out


def scene_file_reader(
    path: str | None,
    *,
    require_live: bool = False,
    timeout: float | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Callable[[], dict | None] | None:
    """A scene callable that re-reads a JSON file on every call (a tracker can keep it fresh), or None
    without a path. The file is read once here: an unreadable file or one that is not a JSON object raises
    OSError/ValueError, so a CLI refuses to start instead of running with the hand check silently off.
    Later read failures (a rewrite in progress) keep the last good scene.

    `require_live=True` is the `--real` reader (every real-arm consumer of `--scene-file` passes it), and
    refuses with SceneFileRefused before returning, so before any prompt or Driver:
    - unless `check_scene_source` accepts the sidecar FILE + ".source.json" that `python -m vision.scene
      --out FILE` writes before its first scene (missing, unreadable, another writer, or "fake");
    - unless the file shows a scene whose `t` is at most `stale_after` = SCENE_SEEN_TIMEOUT_S + 1/hz (the
      sidecar's rate) from the wall clock within `timeout` seconds (default TRACKER_FRAME_S, read at call
      time; re-read every SCENE_POLL_S): a leftover file of a killed writer is never a live feed.
    The callable it returns then, on every call, re-stats the sidecar and raises RuntimeError when it is
    gone or no longer the one accepted (the writer stopped, restarted or was replaced: a new `pid` /
    `started_at` is a new writer even on the same FILE), which freezes a running session; and returns
    `unseen_copy` of a scene older than `stale_after` (including the last good scene kept over read
    failures), the way the live tracker flips everything unseen when its camera goes quiet, so a dead
    writer never leaves the hand check trusting old poses. The accepted sidecar is the callable's `source`
    attribute. `clock` (wall clock, default time.time) and `sleep` (default time.sleep) are for tests.
    Without `require_live` (`--fake`) nothing of this applies and no sidecar is needed.
    """
    if not path:
        return None
    file = Path(path)

    def load() -> dict:
        scene = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(scene, dict):
            raise ValueError(f"{file}: expected a JSON object (schemas/scene.json), got {_short(scene)}")
        return scene

    if require_live:
        return _live_scene_reader(file, load, timeout, clock, sleep)

    latest: dict[str, Any] = {"scene": load()}

    def read() -> dict | None:
        try:
            latest["scene"] = load()
        except (OSError, ValueError):
            pass  # keep the last good scene
        return latest["scene"]

    return read


def _live_scene_reader(
    file: Path,
    load: Callable[[], dict],
    timeout: float | None,
    clock: Callable[[], float] | None,
    sleep: Callable[[float], None] | None,
) -> Callable[[], dict | None]:
    """`scene_file_reader(..., require_live=True)`: the provenance and freshness checks (see there)."""
    seconds = TRACKER_FRAME_S if timeout is None else float(timeout)
    if not (math.isfinite(seconds) and seconds > 0.0):
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
    now = clock if clock is not None else time.time
    sidecar = scene_source_path(file)
    try:
        signature = _source_signature(sidecar)  # before the read: a rewrite in between shows next tick
    except FileNotFoundError:
        signature = None  # check_scene_source names the missing sidecar
    except OSError as exc:
        raise SceneFileRefused(f"the provenance sidecar {sidecar} cannot be read ({exc})") from exc
    source = check_scene_source(file)
    stale_after = SCENE_SEEN_TIMEOUT_S + 1.0 / float(source["hz"])
    scene = load()  # an unreadable file or one that is not a JSON object raises, as without require_live
    polls = math.ceil(seconds / SCENE_POLL_S)
    while _is_stale(_scene_age(scene, now()), stale_after):
        if polls <= 0:
            age = _scene_age(scene, now())
            when = "has no numeric t" if age is None else f"is {age:.1f} s old"
            raise SceneFileRefused(
                f"{file} is not being refreshed: its newest scene {when} after {seconds:g} s (fresh means "
                f"within {stale_after:.2f} s; is `python -m vision.scene --out` running with its camera "
                "delivering?)"
            )
        polls -= 1
        (sleep if sleep is not None else time.sleep)(SCENE_POLL_S)
        try:
            scene = load()
        except (OSError, ValueError):
            pass  # a rewrite in progress: try again on the next poll
    state: dict[str, Any] = {"scene": scene, "signature": signature}

    def pinned() -> None:
        """RuntimeError unless the sidecar is still the one accepted at startup."""
        try:
            current = _source_signature(sidecar)
        except OSError as exc:
            raise RuntimeError(
                f"the provenance sidecar {sidecar.name} is gone ({type(exc).__name__}): the scene writer "
                "stopped or was replaced, so the hand data is no longer proven live"
            ) from exc
        if current == state["signature"]:
            return
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"the provenance sidecar {sidecar.name} changed and cannot be read ({exc})"
            ) from exc
        if data != source:
            got = data if isinstance(data, dict) else {}
            raise RuntimeError(
                f"the scene writer changed during the run: {sidecar.name} was pid {source['pid']} started "
                f"{source['started_at']} ({source['source']}), now pid {_short(got.get('pid'))} started "
                f"{_short(got.get('started_at'))} ({_short(got.get('source'))})"
            )
        state["signature"] = current  # rewritten with the same content: still the same writer

    pinned()  # the sidecar may have gone while the file was read or the freshness wait ran

    def read() -> dict | None:
        pinned()
        try:
            state["scene"] = load()
        except (OSError, ValueError):
            pass  # keep the last good scene; it turns unseen once it is stale
        latest = state["scene"]
        if _is_stale(_scene_age(latest, now()), stale_after):
            return unseen_copy(latest)
        return latest

    read.source = dict(source)  # type: ignore[attr-defined]
    return read


def open_scene_file(path: str | None, *, real: bool) -> tuple[Callable[[], dict | None] | None, bool]:
    """The `--scene-file` step of a CLI, run before any prompt and before a Driver exists: `(scene,
    refused)`. `real` reads through `scene_file_reader(path, require_live=True)`. A refusal or a read
    failure prints the reason and "nothing moved" to stderr and returns `(None, True)`; the caller exits 2.
    With `real` the accepted provenance is printed. No path: `(None, False)`."""
    try:
        scene = scene_file_reader(path, require_live=real)
    except SceneFileRefused as exc:
        print(f"refusing --scene-file {path} for the real arm: {exc}; nothing moved", file=sys.stderr)
        return None, True
    except (OSError, ValueError) as exc:
        print(f"cannot read --scene-file {path}: {exc}; nothing moved", file=sys.stderr)
        return None, True
    source = getattr(scene, "source", None)
    if real and isinstance(source, dict):
        print(
            f"scene file {path}: live, from {source['tracker']} on camera {source.get('camera')} "
            f"(writer pid {source['pid']}, started {source['started_at']})"
        )
    return scene, False


_scene_file_reader = scene_file_reader  # the name the first review's probes used


def tracker_scene(
    fake: bool, *, hand_seen: bool | None = None, timeout: float | None = None
) -> tuple[Any, Callable[[], dict]]:
    """The live hand source of `--tracker`: `(tracker, tracker.latest)`, the pattern of
    scripts/touch_markers.py. `vision.scene` is imported here, never at module import.

    `fake`: a `FakeSceneTracker` serving config/fake_scene.json (no camera, no process-wide singleton), its
    wristband forced to `hand_seen` when that is not None. Otherwise the process-wide
    `vision.scene.get_tracker()` (the SceneTracker on the overhead camera, or the fake under FAKE_CAMERA=1
    / FAKE=1), accepted only when it is provably real and live: a tracker whose `is_fake` is not False, or
    whose `camera` (the SceneTracker has one) is not marked `is_fake = False` (the real SceneTracker on a
    FakeCamera, `vision.scene.fake_chain_tracker`, reports `is_fake` False but its frames are synthetic),
    raises RuntimeError (a fake scene must never feed the real arm), and one whose camera delivers no new
    frame within `timeout` seconds (default TRACKER_FRAME_S, read at call time) raises TimeoutError. The
    wait is `tracker.wait_for_frame(timeout=timeout)`; because the SceneTracker also publishes after a
    missed grab (an unplugged or busy camera returns None), a tracker that counts its delivered frames
    (`frames`) must also show that count growing before the deadline. Whatever `get_tracker()` raises (a
    camera that does not open, a missing config/homography.npy) propagates. A refused tracker is closed
    before the error propagates, so a CLI refuses before any confirmation prompt with the camera released;
    an accepted one is closed by the caller (`close_tracker`) when its run ends.
    """
    if fake:
        from vision.scene import FakeSceneTracker  # lazy: vision/ is Lane B's and needs cv2

        tracker = FakeSceneTracker()
        if hand_seen is not None:
            scene = tracker.latest()
            scene["hand"]["seen"] = bool(hand_seen)
            tracker.set_scene(scene)
        return tracker, tracker.latest
    seconds = TRACKER_FRAME_S if timeout is None else float(timeout)
    if not (math.isfinite(seconds) and seconds > 0.0):
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
    from vision.scene import get_tracker  # lazy: vision/ is Lane B's and needs cv2

    tracker = get_tracker()
    try:
        if getattr(tracker, "is_fake", True) is not False:
            raise RuntimeError(
                f"vision.scene.get_tracker() is a {type(tracker).__name__}: a fake scene must never feed the "
                "real arm (unset FAKE_CAMERA/FAKE or use --fake)"
            )
        camera = getattr(tracker, "camera", None)
        if camera is not None and getattr(camera, "is_fake", True) is not False:
            raise RuntimeError(
                f"vision.scene.get_tracker() runs on a {type(camera).__name__}, not the overhead camera: a "
                "fake scene must never feed the real arm (unset FAKE_CAMERA/FAKE or use --fake)"
            )
        if not _delivers_a_frame(tracker, seconds):
            raise TimeoutError(f"the overhead camera delivered no frame within {seconds:g} s")
    except BaseException:
        close_tracker(tracker)
        raise
    return tracker, tracker.latest


def _frame_count(tracker: Any) -> int | None:
    """The tracker's count of delivered camera frames (`SceneTracker.frames`), None when it keeps none."""
    frames = getattr(tracker, "frames", None)
    return frames if isinstance(frames, int) and not isinstance(frames, bool) else None


def _delivers_a_frame(tracker: Any, timeout: float) -> bool:
    """True when the tracker publishes within `timeout` s and, if it counts delivered frames, that count
    grew meanwhile: publications rebuilt from missed grabs alone are a dead camera, not a live one."""
    before = _frame_count(tracker)
    deadline = time.monotonic() + timeout
    if not tracker.wait_for_frame(timeout=timeout):
        return False
    while before is not None and (_frame_count(tracker) or 0) <= before:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not tracker.wait_for_frame(timeout=remaining):
            return False
    return True


def close_tracker(tracker: Any) -> None:
    """`tracker.close()` without ever raising (a camera that fails to release must not mask the result)."""
    try:
        tracker.close()
    except Exception:
        log.exception("closing the scene tracker failed")


def checked_record_name(name: Any) -> str:
    """`--record NAME` as a safe file stem: letters, digits, `_` and `-`, 1 to 64 characters, starting with
    a letter or digit (never a path, a dot, a space or an empty string); ValueError otherwise."""
    if not isinstance(name, str) or not RECORD_NAME_RE.match(name):
        raise ValueError(f"--record NAME must match {RECORD_NAME_RE.pattern}, got {_short(name)}")
    return name


def record_path(name: str, runs_dir: Path | str | None = None) -> Path:
    """`<runs>/calibration/replays-<name>.jsonl`: `runs` is `runs_dir`, else VARA_RUNS_DIR when set, else
    the repo's runs/. The name is checked with `checked_record_name`."""
    base = Path(runs_dir) if runs_dir is not None else Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")
    return base / "calibration" / f"replays-{checked_record_name(name)}.jsonl"


def prepare_record(name: str, runs_dir: Path | str | None = None) -> tuple[Path, int, int]:
    """The `--record` preflight, run before anything moves: `record_path(name, runs_dir)` with its folder
    created and `(path, ok, total)` of the lines already there (`tally`; (0, 0) for a new file). Raises
    ValueError for a bad name or a corrupt file and OSError when the folder cannot be created, so a CLI
    refuses before a real attempt is spent without its evidence. Writes nothing to the file."""
    path = record_path(name, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise ValueError(f"{path} is not a file")
    ok, total = tally(path)
    return path, ok, total


def record_attempt(
    name: str, record: Mapping[str, Any], runs_dir: Path | str | None = None
) -> tuple[Path, int, int]:
    """Append one replay attempt as a JSON line to `record_path(name, runs_dir)` (parents created, the file
    never overwritten) and return `(path, ok, total)`, the running tally of that file (`tally`).

    `record` must carry a boolean `ok`; the CLI adds `t` (local time), `traj` (path), `traj_sha256`,
    `skill`, `mode` (`fake` or `real`), `reason`, `events` (count) and `elapsed_s`. The line gets
    `label: name` so a file that is copied elsewhere still says what it counted. A file that does not
    tally (`prepare_record` raises) is never appended to.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("ok"), bool):
        raise ValueError(f"a replay record needs a boolean 'ok', got {_short(record, 80)}")
    path, _, _ = prepare_record(name, runs_dir)
    line = json.dumps({"label": name, **dict(record)}, allow_nan=False)
    # a last line without its newline (a hand edit) would otherwise glue two objects into one line
    unterminated = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(("\n" if unterminated else "") + line + "\n")
    ok, total = tally(path)
    return path, ok, total


def tally(path: Path | str) -> tuple[int, int]:
    """`(ok, total)` over the JSON lines of a replays file: `total` lines (blank ones skipped), `ok` of them
    with `"ok": true`. A missing file counts (0, 0); a line that is not a JSON object with a boolean `ok`
    raises ValueError naming the line (the file is evidence, so it is never silently reinterpreted)."""
    file = Path(path)
    if not file.exists():
        return 0, 0
    ok = total = 0
    for number, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{file} line {number}: not JSON ({exc})") from exc
        if not isinstance(entry, dict) or not isinstance(entry.get("ok"), bool):
            raise ValueError(f"{file} line {number}: expected an object with a boolean 'ok'")
        total += 1
        ok += int(entry["ok"])
    return ok, total


def main(argv: Sequence[str] | None = None) -> int:
    """`--traj file.json` validates; `--fake` replays on the in-memory arm; `--real` on the real arm.
    `--scene-file` or `--tracker` give the hand check its data; `--record NAME` tallies the attempts."""
    parser = argparse.ArgumentParser(
        prog="python -m arm.replay",
        description="Validate a trajectory, or replay it (--fake in memory, --real on the arm).",
    )
    parser.add_argument("--traj", required=True, help="trajectory JSON (schemas/trajectory.json)")
    parser.add_argument("--real", action="store_true", help="replay on the REAL arm after confirmation")
    parser.add_argument("--fake", action="store_true", help="replay on the in-memory FakeDriver, no sleeping")
    hand = parser.add_mutually_exclusive_group()
    hand.add_argument(
        "--scene-file",
        help="JSON scene file re-read every tick for the hand check; with --real only the file of a running "
        "`python -m vision.scene --out FILE` on the real camera (its FILE.source.json says live)",
    )
    hand.add_argument(
        "--tracker",
        action="store_true",
        help="hand check from the live overhead SceneTracker (vision.scene.get_tracker(); refused with "
        "--real when it is a fake or shows no frame in 5 s); with --fake the FakeSceneTracker",
    )
    parser.add_argument(
        "--record",
        metavar="NAME",
        help="with --fake or --real: append the attempt to runs/calibration/replays-NAME.jsonl "
        "(VARA_RUNS_DIR moves runs/) and print the running tally 'NAME: k of n ok'",
    )
    parser.add_argument("--port", help="serial port for --real; default FOLLOWER_PORT from .env")
    parser.add_argument("--id", dest="follower_id", help="LeRobot id for --real; default from config")
    cli_args = list(sys.argv[1:] if argv is None else argv)  # the command line, for the Esc record
    args = parser.parse_args(argv)
    if args.record is not None:
        if not (args.real or args.fake):
            parser.error("--record needs --fake or --real: a validate-only run is not a replay attempt")
        try:
            checked_record_name(args.record)
        except ValueError as exc:
            parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")

    traj_path = Path(args.traj)
    try:
        traj_bytes = traj_path.read_bytes()
        traj = json.loads(traj_bytes.decode("utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read --traj {args.traj}: {exc}", file=sys.stderr)
        return 2
    try:
        from arm.ik import fk_tip  # lazy so this module imports without MuJoCo
    except ImportError as exc:
        print(f"cannot import arm.ik.fk_tip for the tip check: {exc}", file=sys.stderr)
        return 2
    try:
        limits, box = load_limits(), load_box()
    except (OSError, ValueError) as exc:
        print(f"cannot load the config: {exc}", file=sys.stderr)
        return 2
    problems = validate(traj, limits, fk_tip, box)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s); nothing moved")
        return 1
    print(f"OK: {args.traj} validated ({len(traj['segments'])} segments)")
    if not (args.real or args.fake):
        return 0
    if args.record is not None:
        try:
            record_file, ok_before, total_before = prepare_record(args.record)
        except (OSError, ValueError) as exc:
            print(f"cannot use --record {args.record}: {exc}; nothing moved", file=sys.stderr)
            return 2
        print(f"recording attempt {total_before + 1} in {record_file} ({ok_before} ok so far)")

    def record(result: dict[str, Any], mode: str, elapsed: float) -> bool:
        """`--record`: one line in runs/calibration/replays-NAME.jsonl and the running tally; False when
        the evidence could not be written (the exit code then says so even after a good replay)."""
        if args.record is None:
            return True
        entry = {
            "t": datetime.now().astimezone().isoformat(timespec="seconds"),
            "traj": str(traj_path.resolve()),
            "traj_sha256": hashlib.sha256(traj_bytes).hexdigest(),
            "skill": traj.get("skill"),
            "mode": mode,
            "ok": bool(result["ok"]),
            "reason": result["reason"],
            "events": len(result["events"]),
            "elapsed_s": round(float(elapsed), 3),
        }
        try:
            path, ok, total = record_attempt(args.record, entry)
        except (OSError, ValueError) as exc:
            print(f"cannot record the attempt: {exc}", file=sys.stderr)
            return False
        print(f"{args.record}: {ok} of {total} ok ({path})")
        return True

    fake_mode = args.fake or (_fake("FAKE_ROBOT") and not args.real)
    scene: Callable[[], dict | None] | None = None
    if not args.tracker:
        scene, refused = open_scene_file(args.scene_file, real=not fake_mode)
        if refused:
            return 2
    if not fake_mode:
        from dotenv import load_dotenv

        load_dotenv()  # before the tracker: FAKE_CAMERA from .env makes it a fake, which --real refuses
    tracker: Any = None
    if args.tracker:
        try:
            tracker, scene = tracker_scene(fake_mode)
        except Exception as exc:  # any failure to build or trust the tracker refuses before anything moves
            print(f"cannot use --tracker: {type(exc).__name__}: {exc}; nothing moved", file=sys.stderr)
            return 2
    try:
        if fake_mode:
            started = time.perf_counter()
            result = replay(traj, driver=driver_mod.FakeDriver(), scene=scene, realtime=False)
            print(json.dumps(result))
            recorded = record(result, "fake", time.perf_counter() - started)
            return 0 if result["ok"] and recorded else 1

        from arm import real_robot

        if scene is None:
            print(
                "WARNING: no --scene-file or --tracker: no hand data, the 10 cm hand check is off. "
                "Keep hands out of the box."
            )
        real_robot.clear_cancel()  # discard an earlier session's cancellation before watching this startup
        drv = driver_mod.Driver(port=args.port, follower_id=args.follower_id)
        watcher = driver_mod.EscWatcher(lambda: real_robot.freeze(drv))
        watcher.start()
        try:
            if not real_robot.confirm_arm_clear(3, watcher):
                return 1
            if real_robot.esc_during_connect(watcher, drv):
                return 1
            drv.connect()  # the only place this CLI powers the arm
            connected_at = time.monotonic()  # Esc presses before this are connect-phase (A13)
            try:
                if real_robot.esc_during_connect(watcher, drv):
                    return 1
                started = time.perf_counter()
                result = replay(traj, driver=drv, scene=scene, realtime=True)
                print(json.dumps(result))
                recorded = record(result, "real", time.perf_counter() - started)
                if drv.stopped:
                    print("Arm is FROZEN with torque on. Press Enter to release the port (torque stays on).")
                    try:
                        input()
                    except EOFError:
                        pass
                return 0 if result["ok"] and recorded else 1
            finally:
                driver_mod.report_esc_latency(
                    watcher, drv, source="arm.replay", connected_at=connected_at, extra={"argv": cli_args}
                )
                drv.disconnect()  # torque stays on; the arm keeps holding its pose
        finally:
            watcher.stop()
    finally:
        if tracker is not None:
            close_tracker(tracker)


if __name__ == "__main__":
    sys.exit(main())
