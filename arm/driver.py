"""SO-101 follower driver: the only module that imports LeRobot or touches the serial port (A4, A13).

Units at this module's boundary are ours: radians for the five joints, 0.0 (closed) to 1.0 (open) for
the gripper, seconds. LeRobot's degrees and its 0-100 gripper scale exist only inside this file, converted
through `config/joint_map.yaml` (`q_rad = radians(sign * lerobot_deg + offset_deg)`).

The 0-100 scale also appears in two config files: `held_min` in `config/objects.yaml` and the optional
`gripper.empty_closed_pct` in `config/joint_map.yaml`. The public helpers `pct_to_gripper(pct)`,
`gripper_to_pct(v)`, `held_threshold(object_name)` and `empty_closed_reading()` (module functions, and the
same four as methods on `Driver` and `FakeDriver`) turn them into our 0-1 units so that no other module
reads a percentage (hard rule 3). arm/replay.py's grip check ("holding" = the reading after
`close_gripper_hold()` is at least `empty_closed_reading() + held_threshold(obj)`) must use these helpers
rather than private copies of the arithmetic.

Safety layer implemented here (docs/safety.md, CLAUDE.md "Arm hazards"):

* `command()` clips every joint to its calibrated range minus 3 degrees and to 2 degrees of change per call
  from the last commanded position, whatever the caller asks. A NaN or infinite target cannot be clipped
  (NaN compares False against every bound) and is refused with ValueError before anything is sent.
  LeRobot's `max_relative_target=5.0` (relative to the *present* position, inside `send_action`) is only
  the second layer.
* `freeze()` writes the present position as the goal and keeps torque on (STOP). `command()` is ignored
  until `resume()`. The request is recorded before the lock is taken, so whoever holds the lock when Esc
  lands (a `connect()` in progress, the next `command()`) applies the STOP itself instead of letting the
  freeze queue behind the very motion it is meant to stop; `resume()` cannot cancel a STOP that is still
  waiting for the lock.
* `connect()` never clears `stopped`. An Esc that lands before or anywhere inside the connect window (port
  open, calibration check, goal sync, LeRobot's `configure()`) leaves the driver stopped when `connect()`
  returns, with torque on and `Goal_Position` re-synced immediately before configure re-enables torque; the
  very next `command()` returns False until `resume()`. The `--real` CLIs check `stopped` right after
  connecting and refuse to move.
* Enabling torque drives each STS3215 straight to its stored `Goal_Position`; stale goals were measured up to
  168 degrees away on our arm. Every torque enable in this module, including the one hidden inside LeRobot's
  `configure()`, is preceded by `Goal_Position = Present_Position` in raw ticks (`normalize=False`).
* After `torque_off()` every write is blocked until `torque_on()`, which syncs the goals first.
* `disconnect()` never turns torque off unless asked: LeRobot's default would drop the arm and whatever
  it holds on exit, Ctrl+C or an exception.
* A gripper closed on an object is held at the contact reading minus 3 (0-100 scale) so a stalled servo
  does not overheat (`close_gripper_hold`).
* `get_driver()` hands out an *unconnected* real Driver; `connect()` powers the arm, so callers do it
  explicitly after the human confirmed the arm is clear. Until then every `command()`/`read()` raises
  RuntimeError rather than silently doing nothing.

CLI, `python -m arm.driver`, one action per run. `--real` adds the typed confirmation, the 3 s countdown
and the Esc watcher (an Esc during `connect()` leaves the arm frozen and nothing moves); `--fake` runs the
in-memory arm; anything else refuses to touch hardware. As in arm.replay and arm.real_robot, FAKE_ROBOT
only picks the fake arm when `--real` is absent; `--fake-contact` is refused with `--real` at parse time.

* `--jog [--deg D]`: every joint +D degrees (default 10) and back at 30 Hz, one after the other (A4).
* `--jog-joint NAME [--deg D] [--hold S]`: A6's increment test on one joint (`jog_joint`): out by D (A6
  uses 20), hold S seconds (default 10), back. It prints the LeRobot degrees the servo reports, our mapped
  radians, the expected model increment and the `arm.ik.fk_tip` positions to compare with the MuJoCo
  viewer, and reads again at the end of the hold. A comparison aid: it writes no configuration and refuses
  an increment that would leave the calibrated range minus 3 degrees.
* `--measure-grip [--objects ...] [--grasps N] [--out PATH]`: A11's session (`measure_grip`): jaw opening,
  empty-closed reading, N `close_gripper_hold` grasps per object. It writes a timestamped report under
  runs/calibration/ (never overwriting) and prints the lines to paste into config/joint_map.yaml and
  config/objects.yaml; nothing in config/ is written. Esc freezes the arm, ends the session and still
  writes the partial report, without paste lines; a reading taken after the Esc is discarded.

`--jog-joint` and `--measure-grip` re-send the start pose with every command, and `Driver.clip` pulls any
joint outside the band (limits minus 3 degrees) into it, so both refuse with exit 2, before anything moves,
when a joint of the start pose lies more than BAND_TOLERANCE_DEG outside the band (`band_violations`).
Then only the named joint, or only the jaws, move.

Every Esc during a `--real` run after the confirmation is measured with `esc_latency_ms` and appended to
runs/calibration/esc-latency.jsonl by `record_esc_latency` with its `esc_phase`: "session" when it came
after `connect()` returned (A13's Esc-to-freeze number, judged against 200 ms) or "connect" when it came
while connecting (nothing moved; the freeze lands when connect returns, so it is a diagnostic kept out of
the 200 ms bar). `report_esc_latency` is that end-of-session step, shared by arm.replay, arm.real_robot,
scripts/touch_markers.py and the orchestrator's console watcher; the orchestrator also records a dashboard
STOP on the real arm. `--esc-latency-report [--file PATH]` reads the records and prints A13's verdict (exit
0 every session record within 200 ms, 1 not, 2 no usable records); it touches no hardware.
`held_min_from_readings` and `objects_fit` are A11's pure rules. Default evidence paths honour
`VARA_RUNS_DIR`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LIMITS_PATH = ROOT / "config" / "limits.yaml"
JOINT_MAP_PATH = ROOT / "config" / "joint_map.yaml"
SCENE_PATH = ROOT / "config" / "scene.yaml"
OBJECTS_PATH = ROOT / "config" / "objects.yaml"  # only for `held_min` (0-100 scale) in held_threshold()

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"
MOTORS = [*JOINTS, GRIPPER]  # LeRobot's motor names for the SO-101 follower, ids 1-6 in this order
TICK = 1 / 30
MAX_STEP_RAD = math.radians(2.0)
MARGIN_RAD = math.radians(3.0)
HOLD_BACKOFF_PCT = 3.0  # hold a grasped object at the contact reading minus this, on LeRobot's 0-100 scale
# A gripper close or open is read only once the jaws have stopped (at the object, at the closed stop, or
# open):
# LeRobot drives the gripper with position gain 16 and a 50 % torque cap, so a full close takes a noticeable
# part of a second, and a fixed short wait read (and then held) the jaws mid-travel, short of a small object.
# Stopped means the reading moved at most GRIP_STILL_PCT (0-100 scale) over GRIP_STILL_POLLS polls taken
# GRIP_POLL_S apart. A gripper still moving after GRIP_TIMEOUT_S is reported unsettled ("timeout") and held at
# its last reading minus 3, so it never keeps pushing; callers treat that grasp as unconfirmed.
GRIP_POLL_S = 0.05
GRIP_STILL_PCT = 0.5
GRIP_STILL_POLLS = 4
GRIP_TIMEOUT_S = 3.0


def _check_grip_wait(settle: float, timeout: float) -> None:
    """ValueError unless `settle` >= 0 and `timeout` > 0 are finite seconds (a gripper wait's arguments)."""
    if not (math.isfinite(settle) and settle >= 0 and math.isfinite(timeout) and timeout > 0):
        raise ValueError(f"settle must be >= 0 and timeout > 0 seconds, got {settle}, {timeout}")


DEFAULT_HELD_MIN_PCT = 5.0  # held_threshold() when config/objects.yaml names no held_min at all (0-100 scale)
STRAIN_TICKS = 15  # ~1.3 degrees; same tolerance as the verified lerobot-tools/sync_goals.py
DEFAULT_FOLLOWER_ID = "my_awesome_follower_arm"
# Passed verbatim to SOFollowerRobotConfig next to port and id. `max_relative_target` must be a float:
# LeRobot's ensure_safe_goal_position raises TypeError on an int.
ROBOT_CONFIG: dict[str, Any] = {
    "use_degrees": True,
    "max_relative_target": 5.0,
    "disable_torque_on_disconnect": False,
}

# Servo position gain (P_Coefficient) written to the five arm joints at connect. LeRobot writes 16 to every
# motor; at 16 the arm sagged about 3 cm at home and could not lift a held prop to its carry height
# (2026-10-18). The gripper keeps LeRobot's 16 so it does not squeeze harder (its torque is capped at 50 %).
# VARA_ARM_P overrides it within ARM_P_RANGE; the servos only accept it while torque is off.
ARM_P_DEFAULT = 24
ARM_P_RANGE = (8, 40)
GAIN_WRITE_RETRIES = 2  # a dropped status packet on the bus is retried this many times per gain write


def arm_p_coefficient() -> int:
    """The P gain for the five arm joints: VARA_ARM_P when set, else ARM_P_DEFAULT; ValueError outside
    ARM_P_RANGE or for a non-integer value."""
    raw = os.environ.get("VARA_ARM_P", "").strip()
    if not raw:
        return ARM_P_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"VARA_ARM_P must be an integer, got {raw!r}") from None
    lo, hi = ARM_P_RANGE
    if not lo <= value <= hi:
        raise ValueError(f"VARA_ARM_P must be between {lo} and {hi}, got {value}")
    return value


def _write_arm_gains(bus: Any, p: int) -> None:
    """P_Coefficient for the five arm joints (not the gripper). Torque must be off (configure's window)."""
    for motor in JOINTS:
        bus.write("P_Coefficient", motor, p, num_retry=GAIN_WRITE_RETRIES)


# MJCF ranges (sim/model/so101_new_calib.xml), used until A6 writes config/limits.yaml.
DEFAULT_LIMITS: dict[str, list[float]] = {
    "shoulder_pan": [-1.9199, 1.9199],
    "shoulder_lift": [-1.7453, 1.7453],
    "elbow_flex": [-1.69, 1.69],
    "wrist_flex": [-1.6581, 1.6581],
    "wrist_roll": [-2.7438, 2.8412],
}
DEFAULT_JOINT_MAP: dict[str, Any] = {
    **{j: {"sign": 1, "offset_deg": 0.0} for j in JOINTS},
    "gripper": {"closed_pct": 0.0, "open_pct": 100.0},
    "lerobot_id": DEFAULT_FOLLOWER_ID,
    "lerobot_robot_type": "so101_follower",
}

__all__ = [
    "JOINTS",
    "GRIPPER",
    "MOTORS",
    "TICK",
    "MAX_STEP_RAD",
    "MARGIN_RAD",
    "HOLD_BACKOFF_PCT",
    "DEFAULT_HELD_MIN_PCT",
    "DEFAULT_LIMITS",
    "limits",
    "joint_map",
    "pct_to_gripper",
    "gripper_to_pct",
    "held_threshold",
    "empty_closed_reading",
    "held_min_from_readings",
    "objects_fit",
    "Driver",
    "FakeDriver",
    "get_driver",
    "reset_driver",
    "inspect_hardware",
    "map_inspection_joints",
    "EscWatcher",
    "jog",
    "jog_joint",
    "measure_grip",
    "esc_latency_ms",
    "record_esc_latency",
    "main",
]


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


# ------------------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------------------

_MISSING: Any = object()  # _load_yaml_cached: the file does not exist (distinct from an empty file -> None)
_YAML_CACHE: dict[Path, tuple[bytes, Any]] = {}


def _load_yaml_cached(path: Path) -> Any:
    """Parse a YAML file, re-parsing only when its bytes change; `_MISSING` when the file does not exist.

    Keyed on the content, not the mtime, so a rewrite within the same timestamp tick is still picked up.
    Reading a few hundred bytes per 30 Hz command is cheap; parsing them every time would not be.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return _MISSING
    hit = _YAML_CACHE.get(path)
    if hit is not None and hit[0] == raw:
        return hit[1]
    data = yaml.safe_load(raw)
    _YAML_CACHE[path] = (raw, data)
    return data


def _what(data: Any) -> str:
    return "an empty file" if data is None else f"a {type(data).__name__}"


def limits() -> dict[str, list[float]]:
    """Joint limits in radians, {joint: [lo, hi]}, from config/limits.yaml; MJCF defaults until it exists.

    A file that exists but holds no mapping (empty, comments only, truncated) is an error, not a fallback:
    silently reverting to the uncalibrated MJCF ranges after A6 would widen every limit.
    """
    data = _load_yaml_cached(LIMITS_PATH)
    if data is _MISSING:
        return {j: list(v) for j, v in DEFAULT_LIMITS.items()}
    if not isinstance(data, dict):
        raise ValueError(f"{LIMITS_PATH}: expected a mapping joint -> [lo, hi] in radians, got {_what(data)}")
    out: dict[str, list[float]] = {}
    for j in JOINTS:
        rng = data.get(j)
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(f"{LIMITS_PATH}: {j} must be [lo, hi] in radians")
        lo, hi = float(rng[0]), float(rng[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"{LIMITS_PATH}: {j} needs finite lo < hi, got [{lo}, {hi}]")
        if hi - lo <= 2 * MARGIN_RAD:
            raise ValueError(
                f"{LIMITS_PATH}: {j} range [{lo}, {hi}] is not wider than twice the 3 degree margin"
            )
        out[j] = [lo, hi]
    return out


def joint_map() -> dict[str, Any]:
    """Per joint `{"sign": +-1, "offset_deg": float}`, `gripper: {closed_pct, open_pct}`, `lerobot_id`.

    `gripper` also carries `empty_closed_pct` (the reading of a close on nothing, 0-100 scale) when the file
    has one; it must lie between `closed_pct` and `open_pct`. Missing file -> the identity map.
    """
    data = _load_yaml_cached(JOINT_MAP_PATH)
    out = copy.deepcopy(DEFAULT_JOINT_MAP)
    if data is _MISSING:
        return out
    if not isinstance(data, dict):
        raise ValueError(f"{JOINT_MAP_PATH}: expected a mapping, got {_what(data)}")
    for j in JOINTS:
        entry = data.get(j) or {}
        if not isinstance(entry, dict):
            raise ValueError(f"{JOINT_MAP_PATH}: {j} must be a mapping with sign and offset_deg")
        sign = entry.get("sign", 1)
        if sign not in (1, -1):
            raise ValueError(f"{JOINT_MAP_PATH}: {j}.sign must be 1 or -1, got {sign!r}")
        offset = float(entry.get("offset_deg", 0.0))
        if not math.isfinite(offset):
            raise ValueError(f"{JOINT_MAP_PATH}: {j}.offset_deg must be finite, got {offset}")
        out[j] = {"sign": int(sign), "offset_deg": offset}
    grip = data.get("gripper") or {}
    if not isinstance(grip, dict):
        raise ValueError(f"{JOINT_MAP_PATH}: gripper must be a mapping with closed_pct and open_pct")
    closed = float(grip.get("closed_pct", 0.0))
    opened = float(grip.get("open_pct", 100.0))
    if not (math.isfinite(closed) and math.isfinite(opened)) or closed == opened:
        raise ValueError(f"{JOINT_MAP_PATH}: gripper closed_pct and open_pct must be finite and differ")
    out["gripper"] = {"closed_pct": closed, "open_pct": opened}
    if grip.get("empty_closed_pct") is not None:
        try:
            empty = float(grip["empty_closed_pct"])
        except (TypeError, ValueError):
            raise ValueError(f"{JOINT_MAP_PATH}: gripper.empty_closed_pct must be a number") from None
        if not math.isfinite(empty) or not (min(closed, opened) <= empty <= max(closed, opened)):
            raise ValueError(
                f"{JOINT_MAP_PATH}: gripper.empty_closed_pct {empty} must lie between closed_pct {closed} "
                f"and open_pct {opened}"
            )
        out["gripper"]["empty_closed_pct"] = empty
    if data.get("lerobot_id"):
        out["lerobot_id"] = str(data["lerobot_id"])
    if data.get("lerobot_robot_type"):
        out["lerobot_robot_type"] = str(data["lerobot_robot_type"])
    return out


def _home_q() -> list[float]:
    """config/scene.yaml `home_q`, the FakeDriver's starting pose; all zeros with a warning when unusable."""
    data = _load_yaml_cached(SCENE_PATH)
    raw = data.get("home_q") if isinstance(data, dict) else None
    try:
        q = [float(v) for v in raw]
    except (TypeError, ValueError):
        q = []
    if len(q) == len(JOINTS) and all(math.isfinite(v) for v in q):
        return q
    log.warning(
        "%s: home_q should be %d finite radians, got %r; the FakeDriver starts at all-zero joints "
        "(mid-range on every joint, far from the rest pose)",
        SCENE_PATH,
        len(JOINTS),
        raw,
    )
    return [0.0] * len(JOINTS)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# Degrees and the 0-100 scale: LeRobot <-> ours. Only these four functions know about them.
def _lerobot_to_rad(joint: str, deg: float, jm: dict[str, Any]) -> float:
    e = jm[joint]
    return math.radians(e["sign"] * deg + e["offset_deg"])


def _rad_to_lerobot(joint: str, q: float, jm: dict[str, Any]) -> float:
    e = jm[joint]
    return e["sign"] * (math.degrees(q) - e["offset_deg"])


def _pct_to_gripper(pct: float, jm: dict[str, Any]) -> float:
    g = jm["gripper"]
    return _clamp((pct - g["closed_pct"]) / (g["open_pct"] - g["closed_pct"]), 0.0, 1.0)


def _gripper_to_pct(v: float, jm: dict[str, Any]) -> float:
    g = jm["gripper"]
    return g["closed_pct"] + _clamp(v, 0.0, 1.0) * (g["open_pct"] - g["closed_pct"])


# Public gripper-scale helpers: the 0-100 scale of config/objects.yaml `held_min` and of the joint map's
# `empty_closed_pct` stays inside this module. arm/replay.py's grip check uses these (see the docstring).
def pct_to_gripper(pct: float) -> float:
    """A LeRobot 0-100 gripper reading in our units, 0.0 (closed) to 1.0 (open), through the joint map."""
    return _pct_to_gripper(float(pct), joint_map())


def gripper_to_pct(v: float) -> float:
    """Our 0.0 (closed) to 1.0 (open) gripper value on LeRobot's 0-100 scale, through the joint map."""
    return _gripper_to_pct(float(v), joint_map())


def _objects() -> dict[str, dict[str, Any]]:
    """config/objects.yaml keyed by object name (marker ids dropped); {} when the file does not exist.

    Like `limits()` and `joint_map()`, a file that exists but holds no mapping is an error, not a fallback.
    Entries without a `name` are skipped.
    """
    data = _load_yaml_cached(OBJECTS_PATH)
    if data is _MISSING:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{OBJECTS_PATH}: expected a mapping marker id -> object entry, got {_what(data)}")
    out: dict[str, dict[str, Any]] = {}
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("name"):
            out[str(entry["name"])] = entry
    return out


def held_threshold(object_name: str | None) -> float:
    """How far above the empty-closed reading (our 0-1 units) the gripper must stop to count as holding.

    `held_min` of the object in config/objects.yaml (0-100 scale) divided by the gripper span of the joint
    map (`|open_pct - closed_pct|`). An unknown object, `None`, or an entry without `held_min` uses the
    smallest `held_min` on file; with none on file (or no file) `DEFAULT_HELD_MIN_PCT`.
    """
    objects = _objects()
    entry = objects.get(object_name) if object_name else None
    try:
        if entry is not None and entry.get("held_min") is not None:
            pct = float(entry["held_min"])
        else:
            known = [float(e["held_min"]) for e in objects.values() if e.get("held_min") is not None]
            pct = min(known) if known else DEFAULT_HELD_MIN_PCT
    except (TypeError, ValueError):
        raise ValueError(f"{OBJECTS_PATH}: held_min must be a number on the 0-100 gripper scale") from None
    if not math.isfinite(pct) or pct < 0:
        raise ValueError(f"{OBJECTS_PATH}: held_min must be finite and >= 0, got {pct}")
    g = joint_map()["gripper"]
    return pct / abs(float(g["open_pct"]) - float(g["closed_pct"]))


def empty_closed_reading() -> float:
    """The gripper reading (0-1) of a close on nothing: `gripper.empty_closed_pct` of config/joint_map.yaml
    converted through the map, or 0.0 (= `closed_pct`) when the file does not record one."""
    jm = joint_map()
    raw = jm["gripper"].get("empty_closed_pct")
    return 0.0 if raw is None else _pct_to_gripper(float(raw), jm)


# A11's object rules (docs/tasks/A-arm.md): width across the grasp at most the jaw opening minus 2 cm, height
# 6 to 11.5 cm (raised from 11 cm for the 11.4 cm juice box of 2026-10-18, which clears the open gripper at
# mid-height in tests/test_model.py). The jaw clearance is in millimeters (what the human measures) and the
# heights in meters.
JAW_CLEARANCE_MM = 20.0
OBJECT_HEIGHT_M = (0.06, 0.115)

# How far outside the commandable band (limits minus MARGIN_RAD) a start pose may lie before the A6 jog and
# the A11 grip session refuse it. Driver.clip pulls such a joint into the band, so up to this much it moves.
BAND_TOLERANCE_DEG = 0.5


def band_violations(q: Sequence[float], *, tolerance_deg: float = BAND_TOLERANCE_DEG) -> dict[str, float]:
    """Joints of `q` outside the commandable band, each with how many degrees outside it lies.

    The band is config/limits.yaml minus the 3 degree margin, the range `Driver.clip` enforces. Clip pulls a
    joint outside it back in at up to 2 degrees per command whatever the caller asks, so a tool that re-sends
    a pose it read (the A6 single-joint jog, the A11 grip session) would also move every such joint. Those
    tools refuse to start when this is not empty. A joint at most `tolerance_deg` outside is accepted.
    """
    lim = limits()
    out: dict[str, float] = {}
    for name, value in zip(JOINTS, q, strict=True):
        lo, hi = lim[name]
        excess = math.degrees(max(lo + MARGIN_RAD - float(value), float(value) - (hi - MARGIN_RAD)))
        if excess > tolerance_deg:
            out[name] = excess
    return out


def _require_in_band(q: Sequence[float], tool: str) -> None:
    """ValueError naming every out-of-band joint of the start pose `q`, before `tool` moves anything."""
    outside = band_violations(q)
    if outside:
        joints = ", ".join(f"{name} {excess:.1f} deg" for name, excess in outside.items())
        raise ValueError(
            f"{tool}: the start pose lies outside the commandable band (limits minus 3 degrees) on {joints}; "
            "the driver would pull those joints into the band, so they would move too. Move the arm to a "
            "pose inside the band first (teleoperate it, or move it by hand with torque off and run "
            "sync_goals.py again), or finish A6's limits; nothing moved"
        )


def _runs_dir() -> Path:
    """`VARA_RUNS_DIR` when set (the orchestrator's and scripts/e2e_fake.py's convention), else runs/."""
    return Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")


def held_min_from_readings(
    closed_pct: Sequence[float], empty_closed_pct: float, *, open_pct: float = 100.0
) -> float:
    """A11's `held_min` (0-100 scale): half the gap between the smallest closed reading and the empty close.

    `closed_pct` are the settled readings of closes on the object (what `close_gripper_hold` returned,
    converted with `gripper_to_pct`) and `empty_closed_pct` the reading of a close on nothing, both on
    LeRobot's 0-100 scale. "Smallest" means nearest to the empty close along the open direction, which
    `open_pct` gives (the joint map's `open_pct`; the default 100 is the usual direction, where the
    formula is `(min(closed) - empty) / 2`). ValueError when no reading is given, a value is not finite,
    or the nearest closed reading is not on the open side of the empty close: the object was not detected
    between the jaws, so no threshold can be proposed.
    """
    readings = [float(v) for v in closed_pct]
    empty = float(empty_closed_pct)
    if not readings:
        raise ValueError("held_min needs at least one closed reading")
    if not all(math.isfinite(v) for v in readings) or not math.isfinite(empty) or not math.isfinite(open_pct):
        raise ValueError(f"readings must be finite, got closed {readings}, empty {empty}, open {open_pct}")
    toward_open = 1.0 if float(open_pct) >= empty else -1.0
    gap = min(toward_open * v for v in readings) - toward_open * empty
    if gap <= 0:
        nearest = min(readings, key=lambda v: toward_open * v)
        raise ValueError(
            f"smallest closed reading {nearest:.1f} is not on the open side of the empty close {empty:.1f}: "
            "the object was not detected between the jaws"
        )
    return gap / 2


def _fit_problems(jaw_open_mm: float, objects: dict[str, dict[str, Any]] | None) -> dict[str, list[str]]:
    """Per `kind: object` entry, the A11 size rules it breaks at this jaw opening ({} when all fit)."""
    jaw = float(jaw_open_mm)
    if not math.isfinite(jaw) or jaw <= 0:
        raise ValueError(f"the jaw opening must be a finite number of mm > 0, got {jaw_open_mm!r}")
    entries = _objects() if objects is None else objects
    lo_h, hi_h = OBJECT_HEIGHT_M
    out: dict[str, list[str]] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("kind") != "object":
            continue
        try:
            width_mm = round(float(entry["size"][1]) * 1000.0, 6)  # 0.035 m is 35 mm, not 35.000000000000004
            height = float(entry["h"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError(f"{name}: an object entry needs size [length, width] and h in meters") from None
        if not (math.isfinite(width_mm) and math.isfinite(height)):
            raise ValueError(f"{name}: size and h must be finite numbers")
        problems: list[str] = []
        if width_mm > jaw - JAW_CLEARANCE_MM:
            problems.append(
                f"{name}: width {width_mm:.0f} mm across the grasp exceeds the jaw opening {jaw:.0f} mm "
                f"minus {JAW_CLEARANCE_MM:.0f} mm"
            )
        if not (lo_h <= height <= hi_h):
            problems.append(
                f"{name}: height {height * 100:.1f} cm is outside {lo_h * 100:.0f} to {hi_h * 100:.0f} cm"
            )
        if problems:
            out[str(name)] = problems
    return out


def objects_fit(jaw_open_mm: float, objects: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Names of the `kind: object` entries that break A11's size rules at this jaw opening (the misfits).

    Lists every object whose width across the grasp (`size[1]`, meters) exceeds `jaw_open_mm` minus 20 mm
    or whose height `h` is outside 0.06 to 0.115 m; an empty list means every object fits. `objects`
    defaults to config/objects.yaml keyed by name; entries of other kinds are ignored. ValueError for a
    non-finite or non-positive jaw opening and for an object entry without numeric `size` and `h`.
    """
    return sorted(_fit_problems(jaw_open_mm, objects))


# ------------------------------------------------------------------------------------------------------------
# Drivers
# ------------------------------------------------------------------------------------------------------------


class _DriverBase:
    """Clipping, the stopped/torque gates and the gripper hold, shared by Driver and FakeDriver.

    Subclasses implement the hardware verbs. Every public method takes `lock` (an RLock) around all
    hardware access, so `freeze()` from the Esc thread serializes with a `command()` in flight.
    """

    is_fake: bool = False

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.stopped = False
        self.torque = False
        self.connected = False
        self.last_command: tuple[list[float], float] | None = None
        self.goal_syncs = 0  # how many times Goal_Position = Present_Position was written
        self.frozen_at: float | None = None  # time.monotonic() of the last freeze, for the Esc latency check
        # How the last `close_gripper_hold` / `wait_gripper_still` ended: "still" (the jaws stopped),
        # "stopped" (STOP or torque off landed first) or "timeout" (still moving after the timeout).
        self.grip_status: str | None = None
        # Set by freeze() before it takes the lock, cleared when the freeze is applied. Whoever holds the lock
        # meanwhile (connect(), command(), resume()) applies the STOP itself: see `_freeze_locked`.
        self._freeze_requested = threading.Event()

    # -- hardware verbs (subclasses) -----------------------------------------------------------------------
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self, torque_off: bool = False) -> None:
        raise NotImplementedError

    def read(self) -> tuple[list[float], float]:
        """Present joints in radians and the gripper from 0.0 (closed) to 1.0 (open)."""
        raise NotImplementedError

    def _send(self, q: list[float], gripper: float) -> None:
        raise NotImplementedError

    def _sync_goals_hw(self) -> None:
        """Write Goal_Position = Present_Position for every motor; torque must not be enabled before this."""
        raise NotImplementedError

    def _torque_on_hw(self) -> None:
        raise NotImplementedError

    def _torque_off_hw(self) -> None:
        raise NotImplementedError

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    # -- shared behavior -----------------------------------------------------------------------------------
    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError(f"{type(self).__name__} is not connected: call connect() first")

    @staticmethod
    def _check_target(q: Sequence[float], gripper: float) -> tuple[list[float], float]:
        """Shape and finiteness of a target, before anything is clipped or sent.

        NaN compares False against every bound, so `_clamp` would pass it straight through: the next call
        would measure its 2 degree step from NaN (unbounded), and LeRobot's own bounding maps a NaN gripper
        to range_min, a full-torque close. An infinite joint would ride a limit for ever. Both are caller
        bugs, refused with ValueError.
        """
        if len(q) != len(JOINTS):
            raise ValueError(f"expected {len(JOINTS)} joints, got {len(q)}")
        qf = [float(v) for v in q]
        gf = float(gripper)
        bad = [name for name, v in zip(JOINTS, qf, strict=True) if not math.isfinite(v)]
        if not math.isfinite(gf):
            bad.append(GRIPPER)
        if bad:
            raise ValueError(f"non-finite target for {bad}: q={qf}, gripper={gf}")
        return qf, gf

    def clip(
        self, q: Sequence[float], gripper: float, reference: Sequence[float] | None = None
    ) -> tuple[list[float], float]:
        """Pure clipping: limits minus MARGIN_RAD first, then at most MAX_STEP_RAD from `reference`.

        Limits go first so that a reference outside the allowed range (the arm was pushed there by hand with
        torque off) still moves back at most 2 degrees per call instead of jumping to the limit. Non-finite
        targets or references raise ValueError (see `_check_target`).
        """
        qf, gf = self._check_target(q, gripper)
        ref: list[float] | None = None
        if reference is not None:
            ref = [float(v) for v in reference]
            if len(ref) != len(JOINTS) or not all(math.isfinite(v) for v in ref):
                raise ValueError(f"reference must be {len(JOINTS)} finite radians, got {ref}")
        lim = limits()
        out: list[float] = []
        for i, name in enumerate(JOINTS):
            lo, hi = lim[name]
            v = _clamp(qf[i], lo + MARGIN_RAD, hi - MARGIN_RAD)
            if ref is not None:
                v = _clamp(v, ref[i] - MAX_STEP_RAD, ref[i] + MAX_STEP_RAD)
            out.append(v)
        return out, _clamp(gf, 0.0, 1.0)

    def command(self, q: Sequence[float], gripper: float) -> bool:
        """Clip and send one 30 Hz target; returns False (nothing sent) while stopped or torque is off.

        A freeze() still waiting for the lock is applied here first (the arm holds, `stopped` is set) and
        the command is refused. Raises RuntimeError when not connected and ValueError for a malformed or
        non-finite target, in both cases before touching the gates: a forgotten connect() or garbage input
        is never a silent no-op.
        """
        with self.lock:
            self._require_connected()
            self._check_target(q, gripper)
            if self._freeze_requested.is_set():
                self._freeze_locked()
            if self.stopped or not self.torque:
                return False
            reference = self.last_command[0] if self.last_command is not None else self.read()[0]
            qc, gc = self.clip(q, gripper, reference)
            self._send(qc, gc)
            self.last_command = (qc, gc)
            return True

    def freeze(self) -> None:
        """STOP: hold the present position with torque on and ignore commands until resume(). Never raises.

        The request is visible before the lock is taken, so a `connect()` or `command()` holding the lock
        applies it before it returns, and `resume()` cannot clear a STOP that has not landed yet. Before
        `connect()` only the flag is set; `connect()` then holds the arm and keeps the flag.
        """
        self._freeze_requested.set()
        with self.lock:
            self._freeze_locked()

    def _freeze_locked(self) -> None:
        """The freeze itself; the caller holds `lock`. Never raises."""
        self._freeze_requested.clear()
        self.stopped = True
        self.last_command = None
        if self.connected:
            try:
                # Goal := present is both the hold and the stale-goal guard; with torque off the arm
                # then gets torque back so it really holds (freeze with torque on, per docs/safety.md).
                self._sync_goals_hw()
                self.goal_syncs += 1
                if not self.torque:
                    self._torque_on_hw()
                self.torque = True
            except Exception:
                log.exception("freeze: holding write failed; stopped flag is set anyway")
        self.frozen_at = time.monotonic()
        if self.connected:
            log.warning("freeze: arm stopped and holding (t=%.3f)", self.frozen_at)
        else:
            log.warning(
                "freeze: stopped flag set while not connected; connect() will hold (t=%.3f)", self.frozen_at
            )

    def resume(self) -> None:
        """Clear the STOP. A freeze() still waiting for the lock wins: it is applied, `stopped` stays set."""
        with self.lock:
            if self._freeze_requested.is_set():
                self._freeze_locked()
                log.warning("resume: a freeze was still landing; the arm stays stopped")
                return
            self.stopped = False
            self.last_command = None  # the next command measures its 2 degree step from the present pose

    def torque_off(self) -> None:
        """Maintenance only. Blocks command() until torque_on()."""
        with self.lock:
            self._require_connected()
            self._torque_off_hw()
            self.torque = False
            self.last_command = None

    def torque_on(self) -> None:
        """Sync Goal_Position = Present_Position for every motor, then enable torque.

        Order matters: these servos drive to their stored goal the instant torque is enabled, and after a
        torque-off period the goal is wherever the arm was last commanded (measured up to 168 degrees away).
        """
        with self.lock:
            self._require_connected()
            self._sync_goals_hw()
            self.goal_syncs += 1
            self._torque_on_hw()
            self.torque = True
            self.last_command = None

    def wait_gripper_still(
        self,
        *,
        settle: float = 0.0,
        timeout: float = GRIP_TIMEOUT_S,
        on_poll: Callable[[], None] | None = None,
    ) -> float:
        """Wait at least `settle` s, then until the jaws stop moving; returns the gripper reading (0.0-1.0).

        Polls every GRIP_POLL_S and counts polls rather than clock time, so the FakeDriver (whose `_sleep`
        returns at once) runs it instantly; on the real bus each read adds its own time, so the wait can run
        somewhat past `timeout`. The jaws count as stopped when the reading moved at most GRIP_STILL_PCT
        (0-100 scale) over the last GRIP_STILL_POLLS polls. `on_poll()` runs after every read, outside the
        lock: the replay passes its STOP and wristband checks, which freeze the arm and raise, so the
        per-tick safety checks keep running while the jaws travel. `grip_status` records the end: "still",
        "stopped" (STOP or torque off landed during the wait, or `on_poll` raised; the reading is then
        wherever the freeze held the jaws) or "timeout" (still moving after `timeout` s).
        """
        _check_grip_wait(settle, timeout)
        g = joint_map()["gripper"]
        tolerance = GRIP_STILL_PCT / abs(g["open_pct"] - g["closed_pct"])
        min_polls = math.ceil(settle / GRIP_POLL_S - 1e-9)
        max_polls = max(1, min_polls, math.ceil(timeout / GRIP_POLL_S - 1e-9))
        history: list[float] = []
        for n in range(1, max_polls + 1):
            self._sleep(GRIP_POLL_S)
            reading = self.read()[1]
            if self.stopped or not self.torque:
                self.grip_status = "stopped"
                return reading
            if on_poll is not None:
                try:
                    on_poll()
                except BaseException:
                    self.grip_status = "stopped"
                    raise
            history.append(reading)
            recent = history[-(GRIP_STILL_POLLS + 1) :]
            if n >= min_polls and len(recent) > GRIP_STILL_POLLS and max(recent) - min(recent) <= tolerance:
                self.grip_status = "still"
                return reading
        self.grip_status = "timeout"
        log.warning("gripper still moving after %.1f s (last reading %.3f)", timeout, history[-1])
        return history[-1]

    def close_gripper_hold(
        self,
        target: float = 0.0,
        settle: float = 0.5,
        timeout: float = GRIP_TIMEOUT_S,
        on_poll: Callable[[], None] | None = None,
    ) -> float:
        """Close toward `target`, wait until the jaws stop (at least `settle` s; `on_poll` as in
        `wait_gripper_still`), then hold at the reading minus 3 (0-100 scale).

        Returns the gripper reading the hold was set from (0.0 closed to 1.0 open). Holding a few units past
        the contact reading keeps the servo's position error small, so it is never left pushing at its torque
        cap. `grip_status` says how the wait ended. A "timeout" close (the jaws still creeping, e.g. into a
        squeezable object) is held too, at a fresh reading minus 3, so the gripper never keeps pushing toward
        `target`; callers must still treat that grasp as unconfirmed. On "stopped" the freeze already holds.
        """
        _check_grip_wait(settle, timeout)  # before the close goes out, so bad arguments move nothing
        with self.lock:
            q = list(self.last_command[0]) if self.last_command is not None else self.read()[0]
            if not self.command(q, target):
                self.grip_status = "stopped"
                return self.read()[1]
        try:
            reading = self.wait_gripper_still(settle=settle, timeout=timeout, on_poll=on_poll)
        except BaseException:
            # Ctrl+C, a bus error or a hook error mid-close: the close is still commanded and would keep
            # pushing, so hold the jaws where they are. A STOP or a hand check already froze the arm; a second
            # freeze would re-stamp frozen_at and distort the Esc latency record.
            if not self.stopped:
                self.freeze()
            raise
        with self.lock:
            if self.grip_status not in ("still", "timeout") or self.stopped or not self.torque:
                return reading
            if self.grip_status == "timeout":
                reading = self.read()[1]  # the jaws were still creeping: hold from where they are now
            g = joint_map()["gripper"]
            # 3 on LeRobot's 0-100 scale, expressed in our 0-1 units and applied toward closed (0.0), so
            # the direction does not depend on which way closed_pct/open_pct run in config/joint_map.yaml.
            backoff = HOLD_BACKOFF_PCT / abs(g["open_pct"] - g["closed_pct"])
            self.command(q, _clamp(reading - backoff, 0.0, 1.0))
            return reading

    # -- gripper scale (the module functions, for callers that hold a driver) ------------------------------
    def pct_to_gripper(self, pct: float) -> float:
        """See the module function `pct_to_gripper`."""
        return pct_to_gripper(pct)

    def gripper_to_pct(self, v: float) -> float:
        """See the module function `gripper_to_pct`."""
        return gripper_to_pct(v)

    def held_threshold(self, object_name: str | None) -> float:
        """See the module function `held_threshold`."""
        return held_threshold(object_name)

    def empty_closed_reading(self) -> float:
        """See the module function `empty_closed_reading`."""
        return empty_closed_reading()


def _sync_goals(bus: Any, *, check_strain: bool = False) -> dict[str, int]:
    """Goal_Position := Present_Position (raw ticks) for every motor, verified by reading back.

    Uses acknowledged per-motor writes, then reads the goals back. Each motor's position is read again
    immediately before its write: a later joint may have sagged while preceding joints were being synced.
    A read-back off by more than STRAIN_TICKS means a lost write and a jump at the next torque
    enable, so it raises instead of continuing. `check_strain` (connect only) also refuses when a powered
    motor's goal is already far from its position: something else is driving or straining the arm. It must
    stay off for freeze(), where a moving arm legitimately has its goal a few degrees ahead of it.
    """
    present = {m: int(v) for m, v in bus.sync_read("Present_Position", normalize=False, num_retry=2).items()}
    missing = [m for m in MOTORS if m not in present]
    if missing:
        raise RuntimeError(f"Present_Position missing for motors {missing}")
    if check_strain:
        torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        goal = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
        for register, values in (("Torque_Enable", torque), ("Goal_Position", goal)):
            missing = [m for m in MOTORS if m not in values]
            if missing:
                raise RuntimeError(f"{register} missing for motors {missing}")
        strained = sorted(m for m in MOTORS if torque[m] and abs(int(goal[m]) - present[m]) > STRAIN_TICKS)
        if strained:
            raise RuntimeError(
                f"torque is on and Goal_Position != Present_Position for {strained}; something is driving "
                "or straining the arm. Refusing to touch it (see lerobot-tools/sync_goals.py)."
            )
    for motor in MOTORS:
        latest = bus.sync_read("Present_Position", motors=[motor], normalize=False, num_retry=2)
        if motor not in latest:
            raise RuntimeError(f"Present_Position missing for motor {motor}")
        present[motor] = int(latest[motor])
        bus.write("Goal_Position", motor, present[motor], normalize=False, num_retry=2)
    back = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
    missing = [m for m in MOTORS if m not in back]
    if missing:
        raise RuntimeError(f"Goal_Position missing after sync for motors {missing}")
    off = {m: (present[m], int(back[m])) for m in MOTORS if int(back[m]) != present[m]}
    far = {m: v for m, v in off.items() if abs(v[1] - v[0]) > STRAIN_TICKS}
    if far:
        raise RuntimeError(f"Goal_Position read-back differs from Present_Position after sync: {far}")
    if off:
        log.warning("goal sync: small read-back offsets (ticks): %s", off)
    return present


def _lerobot_robot_factory(**kwargs: Any) -> Any:
    """Build LeRobot's SOFollower (0.6.2 path first, older `so101_follower` module as a fallback)."""
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    except ImportError:  # pragma: no cover - older LeRobot releases
        from lerobot.robots import so101_follower as _old

        SOFollower, SOFollowerRobotConfig = _old.SO101Follower, _old.SO101FollowerConfig
    return SOFollower(SOFollowerRobotConfig(**kwargs))


_INSPECTION_LOCK = threading.RLock()
_CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def _valid_encoder_calibration(values: Any) -> bool:
    return (
        isinstance(values, dict)
        and set(values) == set(MOTORS)
        and all(
            isinstance(values[name], dict)
            and all(type(values[name].get(field)) is int for field in _CALIBRATION_FIELDS)
            and values[name]["id"] == index
            and values[name]["drive_mode"] in (0, 1)
            and 0 <= values[name]["range_min"] < values[name]["range_max"] <= 4095
            for index, name in enumerate(MOTORS, 1)
        )
    )


def _encoder_calibrations_match(saved: Any, actual: Any) -> bool:
    return (
        _valid_encoder_calibration(saved)
        and _valid_encoder_calibration(actual)
        and all(saved[m][field] == actual[m][field] for m in MOTORS for field in _CALIBRATION_FIELDS)
    )


def map_inspection_joints(report: dict[str, Any]) -> list[float]:
    """Map a follower inspection's encoder readings to current VARA radians, without opening hardware.

    Requires a valid, matching encoder calibration and normalized readings consistent with the raw
    snapshot. This only applies the configured joint map; it does not establish that map's accuracy.
    """
    if (
        not isinstance(report, dict)
        or report.get("role") != "follower"
        or report.get("read_only") is not True
    ):
        raise ValueError("Expected a read-only follower inspection")
    calibration = report.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("matches_hardware") is not True:
        raise ValueError("Inspection encoder calibration did not match hardware")
    if not _encoder_calibrations_match(calibration.get("saved"), calibration.get("hardware")):
        raise ValueError("Inspection encoder calibration is invalid or mismatched")
    fingerprint = calibration.get("sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("Inspection needs a valid calibration SHA-256")
    jm = joint_map()
    if report.get("lerobot_id") != jm["lerobot_id"]:
        raise ValueError("Inspection follower ID differs from config/joint_map.yaml")
    mapped = []
    motors = report.get("motors", {})
    for name in JOINTS:
        motor = motors.get(name, {}) if isinstance(motors, dict) else {}
        raw = motor.get("position_raw") if isinstance(motor, dict) else None
        value = motor.get("calibrated_position_rad") if isinstance(motor, dict) else None
        lo, hi = (calibration["saved"][name][k] for k in ("range_min", "range_max"))
        if type(raw) is not int or not lo <= raw <= hi:
            raise ValueError(f"{name}: raw position is outside its encoder calibration")
        expected = (raw - (lo + hi) / 2) * math.tau / 4095
        try:
            normalized = float(value) if type(value) in (int, float) else math.nan
        except OverflowError:
            normalized = math.nan
        if not math.isfinite(normalized) or not math.isclose(normalized, expected, abs_tol=1e-9):
            raise ValueError(f"{name}: normalized encoder reading disagrees with its raw snapshot")
        mapped.append(_lerobot_to_rad(name, math.degrees(normalized), jm))
    return mapped


def _inspection_robot_factory(*, role: str, port: str, id: str) -> Any:
    if role == "follower":
        return _lerobot_robot_factory(port=port, id=id, **ROBOT_CONFIG)
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    return SOLeader(SOLeaderTeleopConfig(port=port, id=id, use_degrees=True))


def _serial_identity(port: str) -> dict[str, Any]:
    from serial.tools import list_ports

    for entry in list_ports.comports():
        if entry.device.casefold() == port.casefold():
            return {
                "serial_number": entry.serial_number,
                "description": entry.description,
                "vid": entry.vid,
                "pid": entry.pid,
            }
    return {}


def inspect_hardware(
    *,
    port: str,
    role: str = "follower",
    robot_id: str | None = None,
    robot_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Read an SO-101 board without configuring motors, syncing goals, or changing torque.

    Unlike Driver.connect(), only the bus is opened. The reviewed LeRobot bus handshake pings
    motors and reads firmware. Closing the raw port also avoids the leader's torque-off disconnect.
    Encoder calibration is reported separately from the unmeasured VARA joint-to-model map.
    Normalized joint readings here are LeRobot-centered radians, NOT MuJoCo joint coordinates.
    """
    if role not in ("follower", "leader"):
        raise ValueError("role must be follower or leader")
    if not isinstance(port, str) or not port.strip():
        raise ValueError("port is required")
    robot_id = robot_id or (
        os.environ.get("FOLLOWER_ID") or joint_map()["lerobot_id"]
        if role == "follower"
        else os.environ.get("LEADER_ID") or "my_awesome_leader_arm"
    )
    with _INSPECTION_LOCK:
        robot = (robot_factory or _inspection_robot_factory)(role=role, port=port, id=robot_id)
        bus = robot.bus
        saved = {name: asdict(value) for name, value in robot.calibration.items()}
        calibration_path = Path(robot.calibration_fpath)
        fingerprint = (
            hashlib.sha256(calibration_path.read_bytes()).hexdigest() if calibration_path.is_file() else None
        )
        try:
            bus.connect()
            actual = {name: asdict(value) for name, value in bus.read_calibration().items()}
            matches = _encoder_calibrations_match(saved, actual)
            registers = {
                reg: bus.sync_read(reg, normalize=False, num_retry=2)
                for reg in (
                    "Present_Position",
                    "Goal_Position",
                    "Torque_Enable",
                    "Present_Voltage",
                    "Present_Temperature",
                )
            }
            for reg, values in registers.items():
                if set(values) != set(MOTORS):
                    raise RuntimeError(f"Incomplete {reg}: expected all six motors")
                if not all(type(v) is int for v in values.values()):
                    raise RuntimeError(f"Invalid raw {reg}: expected integer register values")
            if any(v not in (0, 1) for v in registers["Torque_Enable"].values()):
                raise RuntimeError("Invalid Torque_Enable: expected 0 or 1")
            for reg in ("Present_Voltage", "Present_Temperature"):
                if any(not 0 <= v <= 255 for v in registers[reg].values()):
                    raise RuntimeError(f"Invalid one-byte register {reg}")
            issues = [] if matches else ["Encoder calibration is missing, invalid or does not match hardware"]
            motors = {}
            for name in MOTORS:
                present = int(registers["Present_Position"][name])
                goal = int(registers["Goal_Position"][name])
                motors[name] = {
                    "position_raw": present,
                    "goal_raw": goal,
                    "goal_error_raw": goal - present,
                    "torque_enabled": bool(registers["Torque_Enable"][name]),
                    "voltage_v": float(registers["Present_Voltage"][name]) / 10.0,
                    "temperature_c": float(registers["Present_Temperature"][name]),
                    "encoder_in_range": 0 <= present <= 4095,
                }
                if not 0 <= present <= 4095 or not 0 <= goal <= 4095:
                    issues.append(f"{name}: position or stored goal is outside 0..4095 encoder ticks")
                if motors[name]["torque_enabled"] and abs(goal - present) > STRAIN_TICKS:
                    issues.append(
                        f"{name}: powered goal differs from position by more than {STRAIN_TICKS} ticks"
                    )
                if matches:
                    lo, hi = saved[name]["range_min"], saved[name]["range_max"]
                    motors[name]["within_calibrated_range"] = lo <= present <= hi
                    if not lo <= present <= hi:
                        issues.append(f"{name}: position is outside its saved encoder calibration range")
                    # LeRobot DEGREES is centered on the calibrated range, using 4095 ticks/turn.
                    # Homing offset is already applied by the servo. Derive from this exact raw read.
                    if name == GRIPPER:
                        value = (_clamp(present, lo, hi) - lo) / (hi - lo)
                        motors[name]["calibrated_gripper"] = 1 - value if saved[name]["drive_mode"] else value
                    else:
                        motors[name]["calibrated_position_rad"] = (present - (lo + hi) / 2) * math.tau / 4095
            volts = [m["voltage_v"] for m in motors.values()]
            if role == "leader" and any(v > 8 for v in volts):
                issues.append("Leader voltage exceeds its 8 V motor limit; the Pro kit leader requires 5 V")
            if any(v < 4 for v in volts):
                issues.append("Supply voltage is below 4 V")
            return {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "read_only": True,
                "role": role,
                "port": port,
                "usb": _serial_identity(port),
                "lerobot_id": robot_id,
                "calibration": {
                    "path": str(calibration_path),
                    "sha256": fingerprint,
                    "matches_hardware": matches,
                    "saved": saved,
                    "hardware": actual,
                },
                "model_mapping": "not_assessed: A6 requires measured joint directions, offsets and tip poses",
                "issues": issues,
                "motors": motors,
            }
        finally:
            # Covers a handshake failure after the port opens too. No robot.disconnect(): the
            # leader's implementation turns torque off, as does bus.disconnect() by default.
            bus.port_handler.closePort()


class Driver(_DriverBase):
    """The real SO-101 follower through LeRobot 0.6.2.

    `robot_factory(**config_kwargs)` returns an object with `.bus`, `.configure()`, `.get_observation()` and
    `.send_action()`; the default builds `SOFollower(SOFollowerRobotConfig(...))`. Tests inject a fake.
    """

    is_fake = False

    def __init__(
        self,
        *,
        port: str | None = None,
        follower_id: str | None = None,
        robot_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        self._port = port
        self._id = follower_id
        self._robot_factory = robot_factory or _lerobot_robot_factory
        self._robot: Any = None
        self._warned_over_open = False  # read(): the over-open gripper warning, once per Driver

    @property
    def robot(self) -> Any:
        return self._robot

    @property
    def follower_id(self) -> str:
        """The LeRobot id `connect()` uses: the constructor's, else FOLLOWER_ID, else the joint map's."""
        return self._id or os.environ.get("FOLLOWER_ID") or joint_map()["lerobot_id"]

    def connect(self) -> None:
        """Open the bus, require a matching calibration, and sync before every torque enable.

        Written from SOFollower.connect() (lerobot/robots/so_follower/so_follower.py) without calling it:
        that method answers a calibration mismatch with an interactive prompt and its configure() toggles
        torque off and on without syncing Goal_Position first.

        `stopped` is never cleared here. A freeze() that ran before connect (flag only, nothing to hold yet)
        or that arrived during it (from the Esc thread, waiting for the lock, or fired inside one of the
        steps) is applied at the end, and the driver stays stopped so the caller's next command() is refused
        until resume(). The arm must be supported: configure() temporarily disables torque. Its enable
        method is guarded so any sag during configuration becomes the new holding pose, not a move back
        to the pre-configuration pose. The guard is restored even if configuration fails.
        """
        with self.lock:
            if self.connected:
                log.warning("Driver.connect: already connected")
                return
            port = self._port or os.environ.get("FOLLOWER_PORT")
            if not port:
                raise RuntimeError(
                    "FOLLOWER_PORT is not set. Run lerobot-find-port (or lerobot-tools/arm_check.py) and put "
                    "the follower's port in .env"
                )
            follower_id = self.follower_id
            arm_p = arm_p_coefficient()  # a bad VARA_ARM_P refuses before the port opens
            robot = self._robot_factory(port=port, id=follower_id, **ROBOT_CONFIG)
            bus = robot.bus
            try:
                bus.connect()  # a failed handshake can leave the serial port open
                if not bus.is_calibrated:
                    raise RuntimeError(
                        f"calibration in the motors does not match the file for id '{follower_id}' (or the "
                        "file is missing). Refusing to connect rather than prompting. Usually another "
                        "program (SenseCraft) wrote its own calibration: don't recalibrate, write the file "
                        r"back with C:\Users\jesse\lerobot-tools\restore_calibration.py (--check, then "
                        "--write; see CLAUDE.md). Only if the file is gone: lerobot-calibrate, then redo A6."
                    )
                # Before any torque enable, including the one inside configure(): goal := present.
                # Enabling torque drives each servo to its stored Goal_Position; stale goals measured
                # up to 168 degrees away would swing the arm (CLAUDE.md, Arm hazards).
                present = _sync_goals(bus, check_strain=True)
                self.goal_syncs += 1
                enable_torque = bus.enable_torque

                gains_written = False

                def sync_and_enable(motors: Any = None, num_retry: int = 0) -> None:
                    nonlocal present, gains_written
                    # LeRobot 0.6.2 torque_disabled() calls this in its finally, including after a
                    # configuration exception. The initial sync is stale if an unpowered joint sagged.
                    if not gains_written:
                        gains_written = True
                        try:
                            # torque is still off here, after configure() wrote its P of 16 to every motor
                            _write_arm_gains(bus, arm_p)
                            log.info("Driver.connect: arm joints P_Coefficient %d (gripper unchanged)", arm_p)
                        except Exception:
                            # never leave the arm limp over a gain: keep LeRobot's and power it as usual
                            log.exception("Driver.connect: could not write the arm gains; keeping LeRobot's")
                    present = _sync_goals(bus)
                    self.goal_syncs += 1
                    enable_torque(motors, num_retry=num_retry)

                bus.enable_torque = sync_and_enable
                try:
                    robot.configure()
                except BaseException:
                    try:
                        # torque_disabled() enters its try/finally only after disable_torque()
                        # finishes. A failure partway through that call otherwise leaves limp joints.
                        sync_and_enable()
                    except Exception:
                        log.exception("Driver.connect: could not restore holding after configure failed")
                    raise
                finally:
                    bus.enable_torque = enable_torque
            except BaseException:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    log.exception("Driver.connect: closing the port after a failed connect")
                    try:
                        # A failed handshake or a disconnect decorator can reject disconnect(). The
                        # reviewed LeRobot SDK raw close does not send any motor/torque writes.
                        bus.port_handler.closePort()
                    except Exception:
                        log.exception("Driver.connect: raw serial port cleanup failed")
                raise
            self._robot = robot
            self.connected = True
            self.torque = True
            self.last_command = None
            log.info("Driver connected on %s (id %s), holding at raw %s", port, follower_id, present)
            if self.stopped or self._freeze_requested.is_set():
                log.warning("Driver.connect: a STOP landed before or during connect; the arm stays frozen")
                self._freeze_locked()

    def disconnect(self, torque_off: bool = False) -> None:
        """Close the port. Torque stays on unless `torque_off=True` (do that only at the rest pose)."""
        with self.lock:
            if not self.connected:
                return
            try:
                self._robot.bus.disconnect(disable_torque=torque_off)
            except BaseException:
                try:
                    self._robot.bus.port_handler.closePort()
                except Exception:
                    log.exception("Driver.disconnect: raw serial port cleanup failed")
                raise
            finally:
                if torque_off:
                    self.torque = False
                self.connected = False
                self.last_command = None
                self._robot = None

    def read(self) -> tuple[list[float], float]:
        with self.lock:
            self._require_connected()
            obs = self._robot.get_observation()
            jm = joint_map()
            q = [_lerobot_to_rad(j, float(obs[f"{j}.pos"]), jm) for j in JOINTS]
            pct = float(obs[f"{GRIPPER}.pos"])
            g = jm["gripper"]
            beyond = pct - g["open_pct"] if g["open_pct"] >= g["closed_pct"] else g["open_pct"] - pct
            if beyond > GRIP_STILL_PCT and self.last_command is None and not self._warned_over_open:
                # Opened past open_pct (teleop, by hand, or a map with a wider open_pct): the reading clamps
                # to 1.0, so the first command re-sends 1.0 and closes the jaws to open_pct unannounced.
                self._warned_over_open = True
                log.warning(
                    "gripper reads %.1f %%, past open_pct %.1f %%: the next command closes the jaws to it",
                    pct,
                    g["open_pct"],
                )
            return q, _pct_to_gripper(pct, jm)

    def _send(self, q: list[float], gripper: float) -> None:
        jm = joint_map()
        # The five joints go through send_action so LeRobot's max_relative_target (5 degrees from the
        # present position) stays in place as the second layer. The gripper is written directly: that
        # cap would turn a close into 20 ticks of 5 % steps and the grip check would read it half open.
        self._robot.send_action(
            {f"{j}.pos": _rad_to_lerobot(j, v, jm) for j, v in zip(JOINTS, q, strict=True)}
        )
        self._robot.bus.sync_write("Goal_Position", {GRIPPER: _gripper_to_pct(gripper, jm)})

    def _sync_goals_hw(self) -> None:
        _sync_goals(self._robot.bus)

    def _torque_on_hw(self) -> None:
        self._robot.bus.enable_torque()

    def _torque_off_hw(self) -> None:
        self._robot.bus.disable_torque()


class FakeDriver(_DriverBase):
    """In-memory arm: commands take effect instantly and read() returns the commanded joints."""

    is_fake = True

    def __init__(self, q0: list[float] | None = None, gripper0: float = 1.0) -> None:
        super().__init__()
        self._q = list(q0) if q0 is not None else _home_q()
        self._g = float(gripper0)
        self.gripper_reading_override: float | None = None  # tests fake a held object (contact reading)
        # An object between the jaws: a close stops at this reading (our 0-1 units) while an open still
        # reads what was commanded. `measure_grip` sets it from --fake-contact around each object close.
        self.contact_gripper: float | None = None
        self.commands: list[tuple[list[float], float]] = []  # everything _send received, oldest first
        self.connected = True
        self.torque = True

    def connect(self) -> None:
        """Same semantics as Driver.connect(): a freeze before or during connect persists."""
        with self.lock:
            self.connected = True
            self.torque = True
            self.last_command = None
            self.goal_syncs += 1
            if self.stopped or self._freeze_requested.is_set():
                self._freeze_locked()

    def disconnect(self, torque_off: bool = False) -> None:
        with self.lock:
            self.connected = False
            if torque_off:
                self.torque = False
            self.last_command = None

    def read(self) -> tuple[list[float], float]:
        with self.lock:
            g = self._g if self.gripper_reading_override is None else self.gripper_reading_override
            if self.contact_gripper is not None:
                g = max(float(g), float(self.contact_gripper))
            return list(self._q), float(g)

    def set_present(self, q: list[float] | None = None, gripper: float | None = None) -> None:
        """Test helper: the arm was moved by hand (torque off) or sagged; no clipping applies."""
        with self.lock:
            if q is not None:
                self._q = [float(v) for v in q]
            if gripper is not None:
                self._g = float(gripper)

    def _send(self, q: list[float], gripper: float) -> None:
        self._q = list(q)
        self._g = float(gripper)
        self.commands.append((list(q), float(gripper)))

    def _sync_goals_hw(self) -> None:
        pass  # the fake's goal always equals its present position

    def _torque_on_hw(self) -> None:
        pass

    def _torque_off_hw(self) -> None:
        pass

    def _sleep(self, seconds: float) -> None:
        pass  # fake replays never sleep


_DRIVER: Driver | FakeDriver | None = None
_DRIVER_LOCK = threading.Lock()


def get_driver() -> Driver | FakeDriver:
    """Process-wide singleton: FakeDriver under FAKE_ROBOT=1 or FAKE=1, else an unconnected Driver.

    The real Driver is not connected here: connecting syncs the goals and powers the arm, so it happens
    explicitly, after the human confirmed the arm is clear (`--real` scripts, the orchestrator at start).
    Until `connect()` every read()/command() raises RuntimeError.
    """
    global _DRIVER
    with _DRIVER_LOCK:
        if _DRIVER is None:
            _DRIVER = FakeDriver() if _fake("FAKE_ROBOT") else Driver()
        return _DRIVER


def reset_driver() -> None:
    """Drop the singleton (tests; the caller disconnects a real driver first)."""
    global _DRIVER
    with _DRIVER_LOCK:
        _DRIVER = None


# ------------------------------------------------------------------------------------------------------------
# Esc watcher and CLI
# ------------------------------------------------------------------------------------------------------------


class EscWatcher:
    """Background thread that calls `callback` when Esc is pressed in the console.

    Windows: `msvcrt.kbhit()/getwch()` (needs a visible console, not a piped or background process).
    Elsewhere: `select` on stdin, put in cbreak mode while watching. `key_source` injects keys in tests.
    """

    ESC = "\x1b"

    def __init__(
        self,
        callback: Callable[[], None] | None = None,
        *,
        poll: float = 0.02,
        key_source: Callable[[], str | None] | None = None,
    ) -> None:
        self.callback = callback
        self.pressed = False
        self.pressed_at: float | None = None
        self._poll = poll
        self._key_source = key_source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._termios_state: Any = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        if self._key_source is None and os.name != "nt":
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                if os.isatty(fd):
                    self._termios_state = (fd, termios.tcgetattr(fd))
                    tty.setcbreak(fd)
            except Exception:  # not a terminal, or no termios: keep watching with plain reads
                self._termios_state = None
        self._thread = threading.Thread(target=self._run, name="esc-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._termios_state is not None:
            try:
                import termios

                fd, state = self._termios_state
                termios.tcsetattr(fd, termios.TCSADRAIN, state)
            except Exception:
                pass
            self._termios_state = None

    def _read_key(self) -> str | None:
        if self._key_source is not None:
            return self._key_source()
        if os.name == "nt":
            import msvcrt

            return msvcrt.getwch() if msvcrt.kbhit() else None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                key = self._read_key()
            except Exception:
                log.exception("EscWatcher: reading the keyboard failed; stopping the watcher")
                return
            if key == self.ESC:
                self.pressed = True
                self.pressed_at = time.monotonic()
                log.warning("Esc pressed (t=%.3f)", self.pressed_at)
                if self.callback is not None:
                    try:
                        self.callback()
                    except Exception:
                        log.exception("EscWatcher callback failed")
            elif key is None:
                self._stop.wait(self._poll)


def jog(
    driver: Driver | FakeDriver, *, deg: float = 10.0, step_deg: float = 1.0, realtime: bool = True
) -> bool:
    """Move each joint out by `deg` and back from the present pose, at most `step_deg` per 30 Hz tick.

    `deg=0` moves nothing. Returns False when the driver stopped accepting commands (Esc froze it, or
    torque is off).
    """
    if not (step_deg > 0):
        raise ValueError(f"step_deg must be positive, got {step_deg}")
    q0, g0 = driver.read()
    n = math.ceil(abs(deg) / step_deg - 1e-9) if deg else 0
    up = [min(step_deg * k, abs(deg)) for k in range(1, n + 1)]
    ramp = [*up, *up[-2::-1], 0.0] if up else []  # out to |deg| and back to the start, one step per tick
    next_tick = time.perf_counter()
    for i, name in enumerate(JOINTS):
        print(f"jog {name}: {math.degrees(q0[i]):+.1f} deg -> {math.degrees(q0[i]) + deg:+.1f} deg and back")
        for offset in ramp:
            target = list(q0)
            target[i] = q0[i] + math.copysign(math.radians(offset), deg)
            if not driver.command(target, g0):
                print("command refused (stopped or torque off); jog ends")
                return False
            if realtime:
                next_tick += TICK
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    q1, _ = driver.read()
    print("jog done; present (deg): " + ", ".join(f"{math.degrees(v):+.1f}" for v in q1))
    return True


HOLD_SLICE_S = 0.1  # jog_joint waits out its hold in slices this long so an Esc is noticed promptly


def jog_joint(
    driver: Driver | FakeDriver,
    joint: str,
    *,
    deg: float = 10.0,
    hold_s: float = 10.0,
    step_deg: float = 1.0,
    realtime: bool = True,
    fk: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """A6's increment test on one joint: out by `deg`, hold `hold_s` seconds, back; prints what to compare.

    Only `joint` moves, `step_deg` per 30 Hz tick (at most 2 degrees, the driver's own clip); the other
    joints and the gripper are re-sent unchanged. Before the move, at the increment and after the return it
    prints the LeRobot reading in degrees as the servo reports it (the number LeRobot's own tools show, so
    degrees stay inside this module), our mapped value in radians through config/joint_map.yaml, the
    expected model increment (`radians(deg)`, 0.349 for 20 degrees) and, when `fk` (`arm.ik.fk_tip`) is
    given, the tip position of the before, expected and measured joint vectors for the MuJoCo viewer
    comparison; at the end of the hold it reads again (`lerobot_deg_settled`, `delta_settled_rad`), since a
    real servo can trail its target right after the ramp. ValueError before anything moves when the
    increment would leave the calibrated range minus the 3 degree margin, or when any joint of the start pose
    lies outside that band (`band_violations`): the driver's clip would pull it in, so it would move too.
    Returns the readings; `ok` is False when the driver stopped accepting commands (Esc froze it, or torque
    is off). A comparison aid: it writes no configuration.
    """
    if joint not in JOINTS:
        raise ValueError(f"joint must be one of {JOINTS}, got {joint!r}")
    deg = float(deg)
    if not math.isfinite(deg) or deg == 0:
        raise ValueError(f"deg must be a finite, non-zero number of degrees, got {deg}")
    if not (0 < step_deg <= math.degrees(MAX_STEP_RAD)):
        raise ValueError(f"step_deg must be in (0, {math.degrees(MAX_STEP_RAD):g}] degrees, got {step_deg}")
    if not (math.isfinite(hold_s) and hold_s >= 0):
        raise ValueError(f"hold_s must be a finite number of seconds >= 0, got {hold_s}")
    i = JOINTS.index(joint)
    jm = joint_map()
    q0, g0 = driver.read()
    _require_in_band(q0, f"jog-joint {joint}")
    delta = math.radians(deg)
    q_expected = list(q0)
    q_expected[i] = q0[i] + delta
    lo, hi = limits()[joint]
    if not (lo + MARGIN_RAD - 1e-9 <= q_expected[i] <= hi - MARGIN_RAD + 1e-9):
        raise ValueError(
            f"{joint}: {math.degrees(q0[i]):+.1f} {deg:+.1f} = {math.degrees(q_expected[i]):+.1f} deg "
            f"leaves the calibrated range minus 3 degrees [{math.degrees(lo + MARGIN_RAD):+.1f}, "
            f"{math.degrees(hi - MARGIN_RAD):+.1f}] deg; nothing moved"
        )

    def tip(q: Sequence[float]) -> tuple[float, float, float] | None:
        if fk is None:
            return None
        x, y, z = fk(list(q))
        return (float(x), float(y), float(z))

    def show(label: str, q: Sequence[float]) -> None:
        parts = [f"LeRobot {_rad_to_lerobot(joint, q[i], jm):+.2f} deg", f"mapped {q[i]:+.4f} rad"]
        t = tip(q)
        if t is not None:
            parts.append("tip (%.3f, %.3f, %.3f) m" % t)
        print(f"  {label}: " + " | ".join(parts))

    result: dict[str, Any] = {
        "joint": joint,
        "deg": deg,
        "expected_delta_rad": delta,
        "hold_s": hold_s,
        "ok": False,
        "lerobot_deg_before": _rad_to_lerobot(joint, q0[i], jm),
        "q_before": list(q0),
        "q_expected": q_expected,
        "tip_before": tip(q0),
        "tip_expected": tip(q_expected),
    }
    print(
        f"jog-joint {joint}: {deg:+.1f} deg from the present pose (expected model increment "
        f"{delta:+.3f} rad), hold {hold_s:g} s, then back. Only {joint} moves."
    )
    show("before", q0)
    show(f"expected at {deg:+.1f} deg (compare with the viewer)", q_expected)
    if fk is None:
        print("  (no fk given: tip positions not printed)")

    n = math.ceil(abs(deg) / step_deg - 1e-9)
    up = [min(step_deg * k, abs(deg)) for k in range(1, n + 1)]
    down = [*up[-2::-1], 0.0]
    next_tick = time.perf_counter()

    def run(offsets: Sequence[float]) -> bool:
        nonlocal next_tick
        for offset in offsets:
            target = list(q0)
            target[i] = q0[i] + math.copysign(math.radians(offset), deg)
            if not driver.command(target, g0):
                print("command refused (stopped or torque off); jog-joint ends")
                return False
            if realtime:
                next_tick += TICK
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        return True

    if not run(up):
        return result
    q_at, _ = driver.read()
    result.update(
        lerobot_deg_at=_rad_to_lerobot(joint, q_at[i], jm),
        q_at=list(q_at),
        delta_at_rad=q_at[i] - q0[i],
        tip_at=tip(q_at),
    )
    show(f"at increment (measured delta {q_at[i] - q0[i]:+.4f} rad)", q_at)
    if hold_s > 0:
        print(f"  holding {hold_s:g} s for the viewer comparison (Esc freezes)...", flush=True)
    for _ in range(int(round(hold_s / HOLD_SLICE_S))):
        driver._sleep(HOLD_SLICE_S)  # a no-op on the FakeDriver
        if driver.stopped:
            print("stopped during the hold; jog-joint ends")
            return result
    q_settled, _ = driver.read()
    result.update(
        lerobot_deg_settled=_rad_to_lerobot(joint, q_settled[i], jm),
        delta_settled_rad=q_settled[i] - q0[i],
    )
    show(f"settled at the end of the hold (delta {q_settled[i] - q0[i]:+.4f} rad)", q_settled)
    next_tick = time.perf_counter()
    if not run(down):
        return result
    q_after, _ = driver.read()
    result.update(
        lerobot_deg_after=_rad_to_lerobot(joint, q_after[i], jm),
        q_after=list(q_after),
        delta_after_rad=q_after[i] - q0[i],
        tip_after=tip(q_after),
        ok=True,
    )
    show(f"after return (delta from start {q_after[i] - q0[i]:+.4f} rad)", q_after)
    return result


def _ask(prompt: str) -> str:
    """`input()` that reads EOF or an unreadable stdin (redirected, captured) as a blank answer."""
    try:
        return input(prompt)
    except (EOFError, OSError):
        print()  # no answer came back: end the prompt line so a redirected-stdin transcript stays readable
        return ""


def _yes_no(answer: str) -> bool | None:
    """'y'/'yes' -> True, 'n'/'no' -> False, anything else (a blank, EOF) -> None: unanswered."""
    text = answer.strip().lower()
    if text in ("y", "yes"):
        return True
    if text in ("n", "no"):
        return False
    return None


def _parse_mm(answer: str) -> float | None:
    """A positive finite number of millimeters, or None for a blank or unusable answer."""
    try:
        value = float(answer.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def measure_grip(
    driver: Driver | FakeDriver,
    *,
    objects: Sequence[str],
    grasps: int = 5,
    ask: Callable[[str], str] | None = None,
    fake_contact_pct: float | None = None,
    lerobot_id: str | None = None,
    calibration_sha256: str | None = None,
) -> dict[str, Any]:
    """A11's gripper session on a connected driver; returns the report, partial when the driver stopped.

    The five joints are read once and that same q goes with every gripper command. Sequence: open to 1.0,
    read once the jaws stop (`wait_gripper_still`, at least 1 s), ask for the jaw opening in mm; close on
    nothing (0.0), read once they stop (at least 0.5 s), re-open at once (an empty close is never left
    stalled); then per object `grasps` times: open, wait for Enter with the object between the jaws,
    `close_gripper_hold()` (the reading once the jaws stopped, held at the contact minus 3), ask whether it
    held, re-open. A reading whose jaws were still moving at the timeout is discarded with a problem entry.
    Readings are recorded on LeRobot's 0-100 scale (`gripper_to_pct`) and each object's `held_min` comes
    from `held_min_from_readings` over its grasps not answered "not held"; the jaw opening is checked
    against every object on file with `objects_fit`. `ask(prompt) -> str` replaces `input()` (EOF reads as
    blank, so a redirected stdin never blocks). On the FakeDriver `fake_contact_pct` (0-100) stands in for
    each object's contact reading through `FakeDriver.contact_gripper`; it is refused on a real driver. When
    the driver stops accepting commands (Esc froze it, torque off) the report gets `stopped: True`, a
    problem entry saying where, and everything measured before it; a reading taken after the stop (the
    jaws frozen part-way) is discarded, never recorded as a measurement. ValueError before anything moves
    when an arm joint of the start pose lies outside the commandable band (`band_violations`), because
    every command re-sends that pose and the driver's clip would move such a joint. A name listed twice is
    measured once.
    """
    names = list(dict.fromkeys(str(n) for n in objects))
    if not names:
        raise ValueError("measure_grip needs at least one object name")
    if grasps < 1:
        raise ValueError(f"grasps must be at least 1, got {grasps}")
    if fake_contact_pct is not None and not getattr(driver, "is_fake", False):
        raise ValueError(
            "fake_contact_pct is for the FakeDriver only; the real jaws report their own contact"
        )
    prompt = ask or _ask
    jm = joint_map()
    report: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fake" if getattr(driver, "is_fake", False) else "real",
        "lerobot_id": lerobot_id or jm["lerobot_id"],
        "calibration_sha256": calibration_sha256,
        "grasps": int(grasps),
        "jaw_open_mm": None,
        "open_reading_pct": None,
        "empty_closed_pct": None,
        "objects": {},
        "problems": [],
        "stopped": False,
    }
    problems: list[str] = report["problems"]
    q, _ = driver.read()
    _require_in_band(q, "measure-grip")
    report["q"] = list(q)

    def send(gripper: float, where: str) -> bool:
        """One gripper target with the fixed q; False (and a problem entry) when the driver refused it."""
        if driver.command(q, gripper) and not driver.stopped:
            return True
        return interrupted(where)

    def interrupted(where: str) -> bool:
        """Always False (session over); records why. Called when a command was refused or Esc landed."""
        report["stopped"] = True
        why = "Esc/STOP" if driver.stopped else "torque off"
        problems.append(f"{why} {where}: session ended, the report is partial")
        return False

    def session() -> None:
        if not send(1.0, "while opening the jaws"):
            return
        opened = driver.wait_gripper_still(settle=1.0)
        if driver.stopped or not driver.torque:
            interrupted("while the jaws were opening (open reading discarded)")
            return
        if driver.grip_status == "timeout":
            problems.append(
                f"the jaws were still moving {GRIP_TIMEOUT_S:g} s after the open command; "
                "open reading discarded"
            )
        else:
            report["open_reading_pct"] = _gripper_to_pct(opened, jm)
        answer = prompt("jaw opening at gripper 1.0 in mm: ")
        if driver.stopped:
            interrupted("at the jaw opening prompt")
            return
        jaw = _parse_mm(answer)
        if jaw is None:
            problems.append(
                "jaw opening not recorded (blank or not a number): run again and answer the prompt"
            )
        else:
            report["jaw_open_mm"] = jaw
            for rule_problems in _fit_problems(jaw, None).values():
                problems.extend(rule_problems)
        if not send(0.0, "while closing on nothing"):
            return
        closed = driver.wait_gripper_still(settle=0.5)
        if driver.stopped or not driver.torque:
            interrupted("during the empty close (the jaws stopped part-way; reading discarded)")
            return
        if driver.grip_status == "timeout":
            problems.append(
                f"empty close: the jaws were still moving {GRIP_TIMEOUT_S:g} s after the close command; "
                "reading discarded, so no held_min can be computed"
            )
        else:
            report["empty_closed_pct"] = _gripper_to_pct(closed, jm)
        if not send(1.0, "while re-opening after the empty close"):
            return
        for name in names:
            entry: dict[str, Any] = {"closed_pct": [], "held": [], "held_min": None}
            report["objects"][name] = entry
            for k in range(1, grasps + 1):
                where = f"grasp {k} of {grasps} on {name}"
                if not send(1.0, f"before {where}"):
                    return
                prompt(f"place {name} between the jaws, Enter: ")
                if driver.stopped:
                    interrupted(f"at the placement prompt of {where}")
                    return
                if fake_contact_pct is not None:
                    driver.contact_gripper = _pct_to_gripper(float(fake_contact_pct), jm)
                reading = driver.close_gripper_hold()
                if driver.stopped or not driver.torque:
                    interrupted(f"during the close of {where}")
                    return
                if driver.grip_status == "timeout":
                    problems.append(
                        f"{where}: the jaws were still moving {GRIP_TIMEOUT_S:g} s after the close command; "
                        "reading discarded"
                    )
                    if not send(1.0, f"while re-opening after {where}"):
                        return
                    continue
                entry["closed_pct"].append(_gripper_to_pct(reading, jm))
                entry["held"].append(_yes_no(prompt("held? y/n: ")))
                if driver.stopped:
                    interrupted(f"at the held prompt of {where}")
                    return
                if not send(1.0, f"while re-opening after {where}"):
                    return
                if fake_contact_pct is not None:
                    driver.contact_gripper = None

    try:
        session()
    except BaseException:
        if not driver.stopped:  # a crash mid-session must not leave a close pushing
            driver.freeze()
        raise
    finally:
        if fake_contact_pct is not None:
            driver.contact_gripper = None
    for name, entry in report["objects"].items():
        answers = list(zip(entry["closed_pct"], entry["held"], strict=True))
        slipped = sum(1 for _, held in answers if held is False)
        if slipped:
            problems.append(
                f"{name}: {slipped} of {len(answers)} grasps slipped (A11 needs every grasp held); "
                "the slipped grasps are left out of held_min"
            )
        # held_min separates holding from not holding, so a grasp answered "not held" never sets it: its
        # reading is what a miss looks like. Unanswered grasps (blank or EOF) still count.
        readings = [reading for reading, held in answers if held is not False]
        if readings and report["empty_closed_pct"] is not None:
            try:
                entry["held_min"] = held_min_from_readings(
                    readings, report["empty_closed_pct"], open_pct=jm["gripper"]["open_pct"]
                )
            except ValueError as exc:
                problems.append(f"{name}: {exc}")
            else:
                # A slip that reads at or beyond the threshold would count as holding during a replay.
                empty = report["empty_closed_pct"]
                toward_open = 1.0 if jm["gripper"]["open_pct"] >= empty else -1.0
                for reading, held in answers:
                    if held is False and toward_open * (reading - empty) >= entry["held_min"] - 1e-9:
                        problems.append(
                            f"{name}: a slipped grasp read {reading:.1f}, at or beyond the threshold "
                            f"{empty + toward_open * entry['held_min']:.1f}; a replay would count such a "
                            "slip as holding"
                        )
        elif not entry["closed_pct"]:
            problems.append(f"{name}: no closed reading recorded")
        elif not readings:
            problems.append(f"{name}: no grasp held, so no held_min")
    return report


def esc_latency_ms(watcher: EscWatcher | None, driver: _DriverBase | None) -> float | None:
    """Milliseconds from the Esc press (`watcher.pressed_at`) to the freeze (`driver.frozen_at`), A13's
    number. None when either timestamp is missing (no press, or the freeze never landed) or when the
    freeze on record predates the press (an older freeze; the new one has not landed)."""
    pressed = getattr(watcher, "pressed_at", None)
    frozen = getattr(driver, "frozen_at", None)
    if pressed is None or frozen is None or frozen < pressed:
        return None
    return (float(frozen) - float(pressed)) * 1000.0


ESC_PHASES = ("session", "connect")


def esc_latency_path() -> Path:
    """The A13 record file: `<runs>/calibration/esc-latency.jsonl` (runs/, or VARA_RUNS_DIR), resolved now."""
    return _runs_dir() / "calibration" / "esc-latency.jsonl"


def esc_phase(watcher: Any, connected_at: float | None) -> str:
    """Which phase an Esc press belongs to: "connect" or "session".

    "connect" when the press came before `driver.connect()` returned (`connected_at`, a `time.monotonic()`
    stamp taken right after it returned, is None or not earlier than `watcher.pressed_at`; equal stamps
    count as connect, since Windows' monotonic clock ticks every ~15.6 ms and nothing moves before the
    callers' post-connect check): nothing was moving and the freeze could only land when connect
    finished, so the number is a diagnostic, not A13's Esc-to-freeze latency. "session" otherwise: the
    arm could move, so the number counts against the 200 ms bar.
    """
    pressed = getattr(watcher, "pressed_at", None)
    if connected_at is None or (pressed is not None and float(pressed) <= float(connected_at)):
        return "connect"
    return "session"


def record_esc_latency(
    path: str | Path | None = None,
    *,
    source: str,
    latency_ms: float | None,
    extra: dict[str, Any] | None = None,
    phase: str = "session",
) -> Path:
    """Append one JSON line `{"t", "source", "phase", "latency_ms", "extra"}` to the Esc-latency log (A13).

    `t` is the local time in ISO 8601 with its UTC offset. `path` defaults to
    `<runs>/calibration/esc-latency.jsonl` (runs/, or VARA_RUNS_DIR); its parent is created. `latency_ms`
    may be None for a press whose freeze never landed. `phase` comes from `esc_phase`: "session" records
    count against A13's 200 ms bar, "connect" records (Esc pressed before connect() returned) are kept as
    diagnostics and stay out of it. Returns the path written.
    """
    if latency_ms is not None and not math.isfinite(float(latency_ms)):
        raise ValueError(f"latency_ms must be finite or None, got {latency_ms!r}")
    if phase not in ESC_PHASES:
        raise ValueError(f"phase must be one of {ESC_PHASES}, got {phase!r}")
    target = Path(path) if path is not None else esc_latency_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "t": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "source": str(source),
        "phase": phase,
        "latency_ms": None if latency_ms is None else float(latency_ms),
        "extra": extra,
    }
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return target


ESC_BAR_MS = 200.0  # A13: Esc freezes motion within 0.2 s


def esc_latency_report(path: str | Path | None = None, *, bar_ms: float = ESC_BAR_MS) -> int:
    """`python -m arm.driver --esc-latency-report`: print the Esc records and A13's verdict; the exit code.

    Reads `path` (default `<runs>/calibration/esc-latency.jsonl`) and prints one row per record (local time,
    source, phase, latency in ms), then the count and maximum of the "session" records and whether every one
    is at most `bar_ms`. "connect" records (Esc pressed before the connect returned, nothing moving) are
    listed separately and never judged. A record without `phase` predates the field and counts as
    "session"; a session record whose freeze never landed (latency null) fails. Exit 0 when every session
    record is within the bar, 1 when any is not, 2 when the file is missing, empty or malformed or holds no
    session record. Touches no hardware.
    """
    target = Path(path) if path is not None else esc_latency_path()
    if not target.is_file():
        print(f"no Esc records: {target} does not exist", file=sys.stderr)
        return 2
    records: list[dict[str, Any]] = []
    for n, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            phase = record.get("phase", "session")
            latency = record.get("latency_ms")
            if phase not in ESC_PHASES or not isinstance(record.get("source"), str):
                raise ValueError("bad phase or source")
            if latency is not None and not (
                isinstance(latency, int | float) and not isinstance(latency, bool) and math.isfinite(latency)
            ):
                raise ValueError("latency_ms must be a finite number or null")
        except (ValueError, AttributeError) as exc:
            print(f"{target}:{n}: malformed Esc record ({exc}): {line[:120]}", file=sys.stderr)
            return 2
        records.append({**record, "phase": phase})
    if not records:
        print(f"no Esc records: {target} is empty", file=sys.stderr)
        return 2

    def row(record: dict[str, Any]) -> str:
        latency = record.get("latency_ms")
        shown = "no freeze" if latency is None else f"{latency:.0f} ms"
        return f"  {record.get('t', '?'):29s}  {record['source']:32s}  {record['phase']:7s}  {shown}"

    session = [r for r in records if r["phase"] == "session"]
    connect = [r for r in records if r["phase"] == "connect"]
    print(f"Esc records in {target}:")
    for record in session:
        print(row(record))
    if connect:
        print(f"connect-phase records (Esc while connecting, nothing moving; not judged): {len(connect)}")
        for record in connect:
            print(row(record))
    if not session:
        print("no session-phase record: run the A13 drill (Esc while the arm moves)", file=sys.stderr)
        return 2
    failed = [r for r in session if r.get("latency_ms") is None or r["latency_ms"] > bar_ms]
    worst = max((r["latency_ms"] for r in session if r.get("latency_ms") is not None), default=None)
    print(
        f"{len(session)} session record(s), max "
        + ("n/a" if worst is None else f"{worst:.0f} ms")
        + f"; A13 bar {bar_ms:.0f} ms: "
        + ("PASS" if not failed else f"FAIL ({len(failed)} over the bar or never frozen)")
    )
    return 0 if not failed else 1


def report_esc_latency(
    watcher: Any,
    driver: Any,
    *,
    source: str,
    connected_at: float | None,
    extra: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path | None:
    """The shared end-of-session A13 step of every `--real` CLI: when `watcher` fired, print the Esc line
    and append one record (`record_esc_latency`) with its `esc_phase`; None when no Esc was pressed.

    A "session" press prints `Esc -> freeze command: N ms`; a "connect" press (before `connect()` returned,
    `connected_at` None or later than the press) prints a diagnostic line instead, since nothing moved. Call
    it only once the human confirmed and the connect was attempted: an Esc during the countdown aborts before
    connecting and leaves no record. Never raises: a failed write is printed, the arm is already frozen.
    """
    if not getattr(watcher, "pressed", False):
        return None
    ms = esc_latency_ms(watcher, driver)
    phase = esc_phase(watcher, connected_at)
    if ms is not None and phase == "session":
        print(f"Esc -> freeze command: {ms:.0f} ms")
    elif ms is not None:
        print(
            f"Esc during connect -> arm held when connect returned: {ms:.0f} ms "
            "(connect phase: a diagnostic, not an A13 latency)"
        )
    try:
        written = record_esc_latency(path, source=source, latency_ms=ms, extra=extra, phase=phase)
    except (OSError, ValueError) as exc:
        print(f"Esc latency not recorded: {exc}", file=sys.stderr)
        return None
    print(f"Esc latency recorded in {written} (phase {phase})")
    return written


def _calibration_sha256(driver: Any) -> str | None:
    """SHA-256 of the connected robot's calibration file (as inspect_hardware hashes it); None for a fake."""
    fpath = getattr(getattr(driver, "robot", None), "calibration_fpath", None)
    if not fpath:
        return None
    path = Path(fpath)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _report_path(out: str | None, prefix: str) -> Path:
    """A new file for a report: `out` (refused when it exists) or `<runs>/calibration/<prefix>-<UTC>.json`.

    Parent directories are created here so the refusal happens before anything moves.
    """
    if out:
        path = Path(out)
        if path.exists():
            raise FileExistsError(f"{path} exists; evidence is never overwritten, choose a new --out")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    folder = _runs_dir() / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for n in range(1, 100):
        path = folder / (f"{prefix}-{stamp}.json" if n == 1 else f"{prefix}-{stamp}-{n}.json")
        if not path.exists():
            return path
    raise FileExistsError(f"could not find a free name for {prefix}-{stamp}*.json in {folder}")


def _write_report(report: dict[str, Any], path: Path) -> Path:
    """Write the report JSON to `path`, which must not exist yet (exclusive create)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    return path


def _print_grip_summary(report: dict[str, Any], path: Path) -> None:
    """The per-object readings, the problems and the exact lines to paste into the two config files."""
    g = joint_map()["gripper"]
    print(f"gripper report written to {path}")
    jaw = report["jaw_open_mm"]
    opened = report["open_reading_pct"]
    empty = report["empty_closed_pct"]
    print(
        "  jaw opening "
        + ("not recorded" if jaw is None else f"{jaw:g} mm")
        + ", open reading "
        + ("n/a" if opened is None else f"{opened:.1f}")
        + ", empty-closed reading "
        + ("n/a" if empty is None else f"{empty:.1f}")
        + " (0-100)"
    )
    for name, entry in report["objects"].items():
        readings = ", ".join(f"{v:.1f}" for v in entry["closed_pct"])
        held = sum(1 for h in entry["held"] if h is True)
        unanswered = sum(1 for h in entry["held"] if h is None)
        held_min = "n/a" if entry["held_min"] is None else f"{entry['held_min']:.1f}"
        print(
            f"  {name}: closed readings [{readings}], held {held} of {len(entry['held'])}"
            + (f" ({unanswered} unanswered)" if unanswered else "")
            + f", held_min {held_min}"
        )
    for problem in report["problems"]:
        print(f"  problem: {problem}")
    if report.get("stopped"):
        print(
            "Session ended early (Esc/STOP or torque off): no lines to paste. The readings above are "
            "partial; run --measure-grip again to completion."
        )
        print("Nothing under config/ was written.")
        return
    if empty is None:
        # A gripper line without empty_closed_pct would drop the one already recorded in the file.
        print("config/joint_map.yaml: keep its gripper line (the empty close was not measured this time)")
    else:
        print("Paste into config/joint_map.yaml (replaces its gripper line; the loader ignores jaw_open_mm):")
        fields = [f"closed_pct: {g['closed_pct']:g}", f"open_pct: {g['open_pct']:g}"]
        fields.append(f"empty_closed_pct: {empty:.1f}")
        if jaw is not None:
            fields.append(f"jaw_open_mm: {jaw:g}")
        print("gripper: {" + ", ".join(fields) + "}")
    print("Paste into config/objects.yaml (held_min on each object's entry):")
    for name, entry in report["objects"].items():
        if entry["held_min"] is None:
            print(f"  {name}: held_min not measured")
        else:
            print(f"  {name}: held_min: {entry['held_min']:.1f}")
    print("Nothing under config/ was written; paste by hand after checking the readings.")


def _fk_tip_or_none() -> Callable[[Sequence[float]], Sequence[float]] | None:
    """`arm.ik.fk_tip`, imported lazily (it loads MuJoCo); None with a note when it is unavailable."""
    try:
        from arm.ik import fk_tip
    except Exception as exc:  # ImportError, or a model that fails to load
        print(f"arm.ik unavailable ({exc}); tip positions are not printed", file=sys.stderr)
        return None
    return fk_tip


def _countdown(seconds: int, watcher: EscWatcher) -> bool:
    try:
        answer = input("Type yes when the arm and table are clear: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Not confirmed; nothing moved.")
        return False
    for n in range(seconds, 0, -1):
        print(f"Moving the REAL arm in {n}... keep clear (Esc aborts)", flush=True)
        for _ in range(10):
            time.sleep(0.1)
            if watcher.pressed:
                print("Aborted before connecting.")
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arm.driver",
        description=(
            "SO-101 driver checks: --jog (A4), --jog-joint (A6 increment test) and --measure-grip (A11). "
            "Nothing touches hardware without --real; --fake runs the in-memory arm. Under --real: typed "
            "confirmation, 3 s countdown, Esc freezes the arm with torque on."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="use the real arm: 3 s countdown, Esc freezes it")
    mode.add_argument("--fake", action="store_true", help="use the in-memory FakeDriver (no hardware)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--jog", action="store_true", help="move each joint +DEG and back at 30 Hz")
    action.add_argument(
        "--jog-joint",
        choices=JOINTS,
        metavar="NAME",
        help=(
            "A6: move one joint (%(choices)s) +DEG, hold --hold seconds, return; prints the LeRobot "
            "degrees, mapped radians and fk tip to compare with the MuJoCo viewer. Writes no config."
        ),
    )
    action.add_argument(
        "--measure-grip",
        action="store_true",
        help=(
            "A11: jaw opening, empty-closed reading and --grasps closes per object; writes a report under "
            "runs/calibration/ and prints the lines to paste into config/joint_map.yaml and objects.yaml"
        ),
    )
    action.add_argument(
        "--esc-latency-report",
        action="store_true",
        help=(
            "A13: print the Esc records (runs/calibration/esc-latency.jsonl, or --file) and whether every "
            "session-phase record is within 200 ms; exit 0 pass, 1 fail, 2 no usable records. No hardware"
        ),
    )
    parser.add_argument("--file", metavar="PATH", help="--esc-latency-report: the records file to read")
    parser.add_argument(
        "--deg", type=float, help="--jog/--jog-joint amplitude in degrees (default 10; A6 uses 20)"
    )
    parser.add_argument(
        "--hold", type=float, metavar="SECONDS", help="--jog-joint: hold at the increment (default 10)"
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        metavar="NAME",
        help="--measure-grip: objects to grasp (default: every kind: object in config/objects.yaml)",
    )
    parser.add_argument(
        "--grasps", type=int, metavar="N", help="--measure-grip: closes per object (default 5)"
    )
    parser.add_argument(
        "--fake-contact",
        type=float,
        metavar="PCT",
        help="--measure-grip --fake: the 0-100 reading the fake jaws stop at on an object (default 25)",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="--measure-grip: report path, must not exist (default runs/calibration/gripper-<UTC>.json)",
    )
    parser.add_argument("--port", help="serial port; default FOLLOWER_PORT from .env")
    parser.add_argument(
        "--id", dest="follower_id", help=f"LeRobot id; default FOLLOWER_ID or {DEFAULT_FOLLOWER_ID}"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if args.file is not None and not args.esc_latency_report:
        parser.error("--file applies to --esc-latency-report only")
    if args.esc_latency_report:
        if args.real or args.fake:
            parser.error("--esc-latency-report reads a file; it takes neither --real nor --fake")
        return esc_latency_report(args.file)
    if not (args.jog or args.jog_joint or args.measure_grip):
        parser.print_help()
        return 2
    if args.hold is not None and not args.jog_joint:
        parser.error("--hold applies to --jog-joint only")
    if args.deg is not None and args.measure_grip:
        parser.error("--deg applies to --jog and --jog-joint only")
    if args.deg is None:
        args.deg = 10.0
    for flag, value in (
        ("--objects", args.objects),
        ("--grasps", args.grasps),
        ("--fake-contact", args.fake_contact),
        ("--out", args.out),
    ):
        if value is not None and not args.measure_grip:
            parser.error(f"{flag} applies to --measure-grip only")
    if args.fake_contact is not None and args.real:
        parser.error("--fake-contact is for --fake only; the real jaws report their own contact reading")
    if not math.isfinite(args.deg):
        parser.error("--deg must be a finite number of degrees")
    if args.jog_joint and args.deg == 0:
        parser.error("--jog-joint needs a non-zero --deg")
    hold = 10.0 if args.hold is None else args.hold
    if not (math.isfinite(hold) and hold >= 0):
        parser.error("--hold must be a finite number of seconds >= 0")
    grasps = 5 if args.grasps is None else args.grasps
    if grasps < 1:
        parser.error("--grasps must be at least 1")
    fake_contact = 25.0 if args.fake_contact is None else args.fake_contact
    if not (0.0 <= fake_contact <= 100.0):
        parser.error("--fake-contact is on the 0-100 scale")

    names: list[str] = []
    out_path: Path | None = None
    if args.measure_grip:
        on_file = _objects()
        names = (
            list(args.objects)
            if args.objects
            else [n for n, e in on_file.items() if e.get("kind") == "object"]
        )
        if not names:
            print(
                "no objects: pass --objects or list kind: object entries in config/objects.yaml",
                file=sys.stderr,
            )
            return 2
        for name in names:
            if name not in on_file:
                print(
                    f"note: {name} is not in config/objects.yaml; measuring it anyway (add its entry before "
                    "pasting held_min)",
                    file=sys.stderr,
                )
        try:
            out_path = _report_path(args.out, "gripper")
        except FileExistsError as exc:
            print(exc, file=sys.stderr)
            return 2
    action_name = (
        "--jog" if args.jog else f"--jog-joint {args.jog_joint}" if args.jog_joint else "--measure-grip"
    )

    if args.fake or (not args.real and _fake("FAKE_ROBOT")):
        driver = FakeDriver()
        if args.jog:
            ok = jog(driver, deg=args.deg, realtime=False)
            print("fake jog", "ok" if ok else "stopped", f"({len(driver.commands)} commands)")
            return 0 if ok else 1
        if args.jog_joint:
            try:
                result = jog_joint(
                    driver, args.jog_joint, deg=args.deg, hold_s=hold, realtime=False, fk=_fk_tip_or_none()
                )
            except ValueError as exc:
                print(f"refused: {exc}", file=sys.stderr)
                return 2
            print("fake jog-joint", "ok" if result["ok"] else "stopped", f"({len(driver.commands)} commands)")
            return 0 if result["ok"] else 1
        assert out_path is not None
        try:
            report = measure_grip(driver, objects=names, grasps=grasps, fake_contact_pct=fake_contact)
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        _print_grip_summary(report, _write_report(report, out_path))
        print(
            "fake measure-grip",
            "stopped" if report["stopped"] else "ok",
            f"({len(driver.commands)} commands)",
        )
        return 1 if report["stopped"] else 0
    if not args.real:
        print("Refusing to touch the arm without --real (use --fake for the in-memory arm).", file=sys.stderr)
        return 2

    from dotenv import load_dotenv

    load_dotenv()
    driver = Driver(port=args.port, follower_id=args.follower_id)
    watcher = EscWatcher(driver.freeze)
    watcher.start()
    try:
        if not _countdown(3, watcher):
            return 1
        if watcher.pressed or driver.stopped:
            print("Stopped before connecting; nothing moved.")
            return 1
        driver.connect()
        connected_at = time.monotonic()  # Esc presses before this are connect-phase (see esc_phase)
        rc = 1
        try:
            if watcher.pressed or driver.stopped:
                if not driver.stopped:  # connect() applies a pending freeze itself; never re-stamp it
                    driver.freeze()
                print(
                    f"Esc pressed while connecting: arm frozen with torque on, nothing moved ({action_name})."
                )
            elif args.jog:
                rc = 0 if jog(driver, deg=args.deg) else 1
            elif args.jog_joint:
                try:
                    result = jog_joint(
                        driver, args.jog_joint, deg=args.deg, hold_s=hold, fk=_fk_tip_or_none()
                    )
                    rc = 0 if result["ok"] else 1
                except ValueError as exc:
                    print(f"refused: {exc}", file=sys.stderr)
                    rc = 2
            else:
                assert out_path is not None
                try:
                    report = measure_grip(
                        driver,
                        objects=names,
                        grasps=grasps,
                        lerobot_id=getattr(driver, "follower_id", None),
                        calibration_sha256=_calibration_sha256(driver),
                    )
                except ValueError as exc:
                    print(f"refused: {exc}", file=sys.stderr)
                    rc = 2
                else:
                    _print_grip_summary(report, _write_report(report, out_path))
                    rc = 1 if report["stopped"] else 0
            if watcher.pressed:
                extra: dict[str, Any] = {"mode": "real", "frozen": driver.stopped}
                if not args.measure_grip:
                    extra["deg"] = args.deg
                report_esc_latency(
                    watcher,
                    driver,
                    source=f"arm.driver {action_name}",
                    connected_at=connected_at,
                    extra=extra,
                )
            if driver.stopped:
                print("Arm is FROZEN with torque on. Press Enter to release the port (torque stays on).")
                try:
                    input()
                except EOFError:
                    pass
        finally:
            driver.disconnect()  # torque stays on; the arm keeps holding its pose
        return rc
    finally:
        watcher.stop()


if __name__ == "__main__":
    sys.exit(main())
