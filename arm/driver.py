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
  returns, with torque on and `Goal_Position` re-synced to the present position after `configure()`; the
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

`python -m arm.driver --real --jog` moves each joint +10 degrees and back at 30 Hz after a 3 s countdown,
with Esc freezing the arm in place (an Esc during `connect()` leaves it frozen and nothing moves). Without
`--real` it refuses to touch hardware (`--fake` uses the in-memory arm).
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
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
    "Driver",
    "FakeDriver",
    "get_driver",
    "reset_driver",
    "EscWatcher",
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

    def close_gripper_hold(self, target: float = 0.0, settle: float = 0.5) -> float:
        """Close toward `target`, wait `settle` s, then hold at the reading minus 3 (0-100 scale).

        Returns the gripper reading after settling (0.0 closed to 1.0 open). Holding a few units past the
        contact reading keeps the servo's position error small, so it is not stalled at full torque.
        """
        with self.lock:
            q = list(self.last_command[0]) if self.last_command is not None else self.read()[0]
            if not self.command(q, target):
                return self.read()[1]
        self._sleep(settle)
        with self.lock:
            reading = self.read()[1]
            if self.stopped or not self.torque:
                return reading
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

    Mirrors the verified lerobot-tools/sync_goals.py: acknowledged per-motor writes, then the goals are
    read back. A read-back off by more than STRAIN_TICKS means a lost write and a jump at the next torque
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
        strained = sorted(
            m
            for m in MOTORS
            if torque.get(m) and abs(int(goal.get(m, present[m])) - present[m]) > STRAIN_TICKS
        )
        if strained:
            raise RuntimeError(
                f"torque is on and Goal_Position != Present_Position for {strained}; something is driving "
                "or straining the arm. Refusing to touch it (see lerobot-tools/sync_goals.py)."
            )
    for motor in MOTORS:
        bus.write("Goal_Position", motor, present[motor], normalize=False, num_retry=2)
    back = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
    off = {m: (present[m], int(back.get(m, -1))) for m in MOTORS if int(back.get(m, -1)) != present[m]}
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

    @property
    def robot(self) -> Any:
        return self._robot

    def connect(self) -> None:
        """Open the bus, require a matching calibration, sync the goals, then configure().

        Written from SOFollower.connect() (lerobot/robots/so_follower/so_follower.py) without calling it:
        that method answers a calibration mismatch with an interactive prompt and its configure() toggles
        torque off and on without syncing Goal_Position first.

        `stopped` is never cleared here. A freeze() that ran before connect (flag only, nothing to hold yet)
        or that arrived during it (from the Esc thread, waiting for the lock, or fired inside one of the
        steps) is applied at the end: goal := present once more, after configure()'s torque toggle, and the
        driver stays stopped so the caller's next command() is refused until resume().
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
            follower_id = self._id or os.environ.get("FOLLOWER_ID") or joint_map()["lerobot_id"]
            robot = self._robot_factory(port=port, id=follower_id, **ROBOT_CONFIG)
            bus = robot.bus
            bus.connect()  # opens the port, pings the six motors, checks their firmware
            try:
                if not bus.is_calibrated:
                    raise RuntimeError(
                        f"calibration in the motors does not match the file for id '{follower_id}' (or the "
                        "file is missing). Refusing to connect rather than prompting; recalibrate with "
                        "lerobot-calibrate and redo A6."
                    )
                # Before any torque enable, including the one inside configure(): goal := present.
                # Enabling torque drives each servo to its stored Goal_Position; stale goals measured
                # up to 168 degrees away would swing the arm (CLAUDE.md, Arm hazards).
                present = _sync_goals(bus, check_strain=True)
                robot.configure()  # torque_disabled(): off, PID/mode/gripper limits, on -> holds `present`
            except BaseException:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    log.exception("Driver.connect: closing the port after a failed connect")
                raise
            self._robot = robot
            self.connected = True
            self.torque = True
            self.last_command = None
            self.goal_syncs += 1
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
            return q, _pct_to_gripper(float(obs[f"{GRIPPER}.pos"]), jm)

    def _send(self, q: list[float], gripper: float) -> None:
        jm = joint_map()
        # The five joints go through send_action so LeRobot's max_relative_target (5 degrees from the
        # present position) stays in place as the second layer. The gripper is written directly: that
        # cap would turn a close into 20 ticks of 5 % steps and the 0.4 s grip check would read it half open.
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
        description="SO-101 driver check. Nothing touches hardware without --real.",
    )
    parser.add_argument("--real", action="store_true", help="use the real arm: 3 s countdown, Esc freezes it")
    parser.add_argument("--fake", action="store_true", help="use the in-memory FakeDriver (no hardware)")
    parser.add_argument("--jog", action="store_true", help="move each joint +DEG and back at 30 Hz")
    parser.add_argument("--deg", type=float, default=10.0, help="jog amplitude in degrees (default 10)")
    parser.add_argument("--port", help="serial port; default FOLLOWER_PORT from .env")
    parser.add_argument(
        "--id", dest="follower_id", help=f"LeRobot id; default FOLLOWER_ID or {DEFAULT_FOLLOWER_ID}"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if not args.jog:
        parser.print_help()
        return 2
    if args.fake or (not args.real and _fake("FAKE_ROBOT")):
        driver = FakeDriver()
        ok = jog(driver, deg=args.deg, realtime=False)
        print("fake jog", "ok" if ok else "stopped", f"({len(driver.commands)} commands)")
        return 0 if ok else 1
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
        try:
            if watcher.pressed or driver.stopped:
                driver.freeze()
                print("Esc pressed while connecting: arm frozen with torque on, nothing moved.")
                ok = False
            else:
                ok = jog(driver, deg=args.deg)
            if watcher.pressed and watcher.pressed_at is not None and driver.frozen_at is not None:
                print(f"Esc -> freeze command: {(driver.frozen_at - watcher.pressed_at) * 1000:.0f} ms")
            if driver.stopped:
                print("Arm is FROZEN with torque on. Press Enter to release the port (torque stays on).")
                try:
                    input()
                except EOFError:
                    pass
        finally:
            driver.disconnect()  # torque stays on; the arm keeps holding its pose
        return 0 if ok else 1
    finally:
        watcher.stop()


if __name__ == "__main__":
    sys.exit(main())
