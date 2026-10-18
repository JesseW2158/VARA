"""Live primitives on the real arm (A10) and the STOP/cancel flag (A13).

These are our code, never AI-written, and the only motions besides `arm.replay.replay()` of a validated
trajectory that may reach the real arm (CLAUDE.md hard rule 2). They share `arm.replay.Session`: every
tick checks the STOP flag, the hand and the workspace box, the tip moves at most 0.2 m/s and at most
0.1 m/s within 0.15 m of a seen wristband, and every planned sweep is validated against the limits and the
box before it starts (`Session.plan_path`: the joint-space sweep when its tip stays inside, else a straight
tip line solved with IK, since a sweep between a folded and an extended pose arcs the tip over the 0.30 m
ceiling).

- `home(driver, *, max_step_rad, scene, ...)`: from the present pose to `home_q` (config/scene.yaml) at
  <= 1 degree per tick (30 deg/s): the direct sweep, else the straight line, else via a raised pose. An
  arm that starts outside the box (folded at the rest pose, or sagged after torque off) is brought back as
  long as the sweep never rises above the ceiling, never goes lower than where it started and never
  reaches further out than the box (its near edge, behind which the folded arm lives, is the only
  relaxation); a joint resting inside the 3 degree margin is walked back into the band by the driver's 2
  degrees per tick (the plan starts from the pose clamped into the band).
- `freeze(driver)`: the driver's freeze (hold in place, torque on) plus the module-level cancel flag that
  any running primitive checks every tick; a cancelled primitive returns reason `stopped`.
- `carry_to_hand(obj, *, driver, scene, on_event, timeout, ...)`: with the object held, take the grasp
  center (`arm.ik.fk_grasp`) to the target, the seen wristband + 0.05 m (the handoff zone when the hand is
  unseen). A target outside the workspace box latches a fallback to the configured handoff zone; a
  valid target near the tape is inset TARGET_INSET_M to leave room for the tip. The handoff zone must
  itself have that inset. The carry has three legs, like `arm.sim_robot.SimRobot.carry_to_hand`:
  1. Rise: straight up at the present xy to the carry height, the sim's clearance rule
     (`carry_height`): the held object's longest half-extent plus CARRY_CLEARANCE_M above the tallest
     obstacle or seen object still on the table, required to fit CEILING_MARGIN_M under the box top.
     Heights and sizes come from config/objects.yaml (`arm.replay.load_objects`), or the scene's entry
     when the file lacks the name. The rise is planned whole before it starts (`_plan_rise`: IK every
     LINE_SAMPLE_M, the pitch blended toward the steepest one that reaches the top) and validated against
     the limits, the box (which holds the tip under the 0.30 m ceiling), the vertical line
     (CARRY_LINE_TOL_M) and the
     start height. The rise is skipped only when the grasp center is already at clearance. Missing
     geometry, insufficient room below the ceiling, an invalid rise, or encoder feedback below the
     required height aborts while holding the object. No travel or tracking follows a failed rise.
  2. Travel: straight lines at the carry height toward the point above the target, re-aimed at 10 Hz as
     the hand moves, each cycle's bridge validated like the tracking's (below) and never taking the grasp
     center more than CARRY_LINE_TOL_M under the carry height, with no joint-space sweep. It ends above
     the target (ABOVE_XY_M), or where no straight step at that height validates (the reach shrinks with
     height), or after TRAVEL_MAX_S, or when the clock stops advancing; the tracking goes on from there
     (an event says so).
  3. Tracking and descent: a straight line toward the target re-aimed at 10 Hz. Each cycle solves IK
     (`arm.ik.solve_grasp`) for the point at most one cycle's tip travel ahead on that line, the pitch
     blended toward -60 degrees by at most PITCH_STEP_RAD (shallower pitches are tried where the reach
     needs them), and executes `bridge(q, solution)` at <= 2 degrees per tick, all joints interpolated
     together, only after that bridge validated against the limits and the box (inset PLAN_INSET_M) and
     was checked never to take the grasp center lower than the start or the goal by more than DIVE_TOL_M
     (`_servo_path`; the descent from the carry height goes down to the goal, which that allows). Nothing
     is ever stepped toward an unvalidated pose: a solution more than SERVO_JUMP_RAD away (what the cycle's
     ticks can execute) is only used up to LINE_JUMP_RAD and then run to its end before re-aiming; when no
     point on the line is reachable, the first ticks of the validated joint-space sweep to the goal's pose
     run, else the arm holds. Once the hand has stayed within 0.08 m of the grasp center for 1 s the
     gripper opens; after `timeout` seconds of tracking (finite; capped at TIMEOUT_MAX_S) with no steady
     hand it aims for the handoff zone instead. The fallback has FALLBACK_EXTRA_S to reach that zone.
     Release requires encoder feedback within ZONE_REACHED_M of it. An unreached fallback, MAX_TRACK_S
     of tracking, or a stalled clock aborts while holding the object, with torque on.
  Then it backs off 0.05 m up, goes home and emits `handoff_done`. With `use_policy: true` in
  config/rl.yaml the RL policy (`rl.policy_runner.run_policy`, imported lazily) does the reach after the
  rise and travel, before the IK tracking; when it cannot be imported, IK is used with a warning. The
  keyword `policy` (`{"use_policy": bool, "policy": path}`, the `policy_config()` shape, checked by
  `checked_policy`) overrides config/rl.yaml for that one call (the D7 staged bring-up); None keeps the
  config. The 10 cm hand freeze is off inside this primitive (it approaches the hand on purpose); the
  speed cap near the hand stays on in every leg. While the travel and the tracking re-aim at 10 Hz, every
  cycle with a seen wristband emits its xy and its distance from the grasp center with the leg's clock
  (`carry travel t=0.3 s: hand (x, y), d m from the grasp center`; one `hand unseen` line when it
  disappears), so the hand speed of the A10 check (5 cm/s or slower) can be read from the log.

Each primitive returns `{"ok": bool, "reason": None | "stopped" | "hand_too_close" | "error",
"events": [...]}` and never raises for driver, IK or scene failures: an error while moving freezes the arm;
a setup failure (config files, arm.ik, a bad argument or policy override, torque off) returns `error`
without freezing, and a driver that is already stopped returns `stopped`. A carry whose clearance cannot
be established freezes even before motion, preserving the grasp while the failed plan is reviewed.

    python -m arm.real_robot --fake --carry-test             # in-memory arm, fake hand, no sleeping
    python -m arm.real_robot --fake --carry-test --tracker   # the same hand through the FakeSceneTracker
    python -m arm.real_robot --real --carry-test [--scene-file scene.json | --tracker]
    python -m arm.real_robot --real --home

`--real` loads .env, asks for confirmation, counts down 3 s, watches Esc (freeze) and connects the driver
only then; an Esc that lands during `connect()` (which resets the driver's stopped flag) is caught right
after it and freezes the arm before anything runs. A `--scene-file` that cannot be read stops the CLI
before anything connects; with `--real` it goes through `arm.replay.open_scene_file` (the reader shared
with arm.replay and rl.policy_runner, `scene_file_reader(..., require_live=True)`), so a file without the
live provenance sidecar of a running `python -m vision.scene --out FILE` (FILE.source.json, `source:
live`), or one that shows no fresh scene within 5 s, exits 2 with the reason and "nothing moved" before
the confirmation prompt and before a Driver exists; during the run a stale scene reads as all unseen and
a sidecar that disappears or changes freezes the arm. `--tracker` (exclusive with `--scene-file`) takes
the hand from the live overhead SceneTracker (`arm.replay.tracker_scene`: `vision.scene.get_tracker()`,
imported lazily, its `latest` as the scene callable, closed when the run ends); with `--real` a tracker
whose `is_fake` is not False (FAKE_CAMERA=1 / FAKE=1 from .env) or whose camera is a fake, one that
cannot be built (the camera does not open) or one whose camera delivers no new frame within 5 s exits 2
before the confirmation prompt and before a Driver exists, and with `--fake` the FakeSceneTracker shows
the hand of config/fake_scene.json (seen, so the carry test has a hand to follow). Nothing here imports
`agents/` or `web/`, `vision/` only through `arm.replay.tracker_scene` for `--tracker`, and
`rl.policy_runner` lazily and only when the config or the override asks for the policy.
"""

from __future__ import annotations

import argparse
import json
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

from arm import driver as driver_mod
from arm import ik
from arm.replay import (
    BRIDGE_STEP_RAD,
    LINE_JUMP_RAD,
    LINE_SAMPLE_M,
    MAX_STEP_RAD,
    PLAN_INSET_M,
    STEP_TOL,
    Abort,
    Pacer,
    Session,
    bridge,
    close_tracker,
    inside_box,
    load_objects,
    open_scene_file,
    setup_failure,
    shrunk_box,
    tracker_scene,
    validate_path,
)

__all__ = [
    "ABOVE_XY_M",
    "BACKOFF_M",
    "CARRY_CLEARANCE_M",
    "CARRY_LINE_TOL_M",
    "CEILING_MARGIN_M",
    "DIVE_TOL_M",
    "HAND_ABOVE_M",
    "HOME_STEP_RAD",
    "HOME_TOL_RAD",
    "PITCH_STEP_RAD",
    "RELEASE_STEADY_S",
    "RELEASE_WITHIN_M",
    "SERVO_JUMP_RAD",
    "START_PITCH",
    "TARGET_INSET_M",
    "TIMEOUT_MAX_S",
    "TRAVEL_MAX_S",
    "cancelled",
    "carry_height",
    "carry_to_hand",
    "checked_policy",
    "clear_cancel",
    "confirm_arm_clear",
    "esc_during_connect",
    "freeze",
    "handoff_zone",
    "home",
    "main",
    "policy_config",
]

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "config" / "scene.yaml"
RL_PATH = ROOT / "config" / "rl.yaml"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

HOME_STEP_RAD = BRIDGE_STEP_RAD  # 1 degree per tick: 30 deg/s
HOME_TOL_RAD = math.radians(1.0)  # already home when every joint is within this
RAISE_M = 0.08  # the intermediate pose of a home() sweep lifts the tip by this (at least to RAISE_MIN_Z)
RAISE_MIN_Z = 0.20
START_BELOW_SLACK_M = 0.005  # a sweep from outside the box may not go lower than the start minus this

TRACK_HZ = 10  # IK re-solves per second while tracking the hand
TICKS_PER_CYCLE = 3  # 30 Hz ticks per IK cycle
HAND_ABOVE_M = 0.05  # the grasp center aims this far above the wristband marker
RELEASE_WITHIN_M = 0.08  # the hand must stay this close to the grasp center ...
RELEASE_STEADY_S = 1.0  # ... for this long before the gripper opens
START_PITCH = math.radians(-60.0)  # the pitch the tracking blends toward (IK relaxes it when unreachable)
PITCH_STEP_RAD = math.radians(5.0)  # the pitch changes at most this much per cycle (1.5 cm at the tip)
BACKOFF_M = 0.05  # after the release, straight up
ZONE_REACHED_M = 0.02  # the fallback release happens when the grasp center is this close to the zone
FALLBACK_EXTRA_S = 5.0  # after the timeout, at most this long to reach the handoff zone
TIMEOUT_MAX_S = 60.0  # a longer `timeout` is capped: the orchestrator's whole real run has a 90 s budget
MAX_TRACK_S = 90.0  # the tracking never runs longer than this, whatever the deadlines say
STALL_CYCLES = 100  # tracking cycles with an unchanged clock reading before the tracking gives up
# The grasp center stays this far inside the box: the tip is grasp_offset() (2.5 cm) from it plus IK's
# 5 mm tolerance, so the tip stays inside too.
TARGET_INSET_M = 0.03
SERVO_JUMP_RAD = TICKS_PER_CYCLE * MAX_STEP_RAD  # 6 deg: the joint move one cycle's ticks execute
DIVE_TOL_M = 0.01  # a cycle never takes the grasp center lower than the start or the goal by more
# The carry height, the sim's rule (arm/sim_robot.py `_carry_clearance`; tests pin the two equal).
CARRY_CLEARANCE_M = 0.02  # the held object's lowest point clears the tallest thing on the table by this
CEILING_MARGIN_M = 0.01  # the carry height stays this far under the top of the workspace box
RISE_DONE_M = 0.005  # a grasp center this close under the carry height needs no rise (the sim's 5 mm)
# IK's 5 mm position tolerance plus 1 mm: how far the rise may stray from its vertical line and how far
# the travel may sink under the carry height.
CARRY_LINE_TOL_M = 0.006
ABOVE_XY_M = 0.01  # the travel ends with the grasp center this close (in xy) to above the target
TRAVEL_MAX_S = 10.0  # the travel at the carry height never runs longer than this

_CANCEL = threading.Event()

Vec3 = tuple[float, float, float]


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


# ------------------------------------------------------------------------------------------------------------
# STOP
# ------------------------------------------------------------------------------------------------------------


def cancelled() -> bool:
    """True once `freeze()` ran; every primitive checks it each tick and ends with reason `stopped`."""
    return _CANCEL.is_set()


def clear_cancel() -> None:
    """Primitives clear the flag when they start; the driver's own `stopped` flag stays authoritative."""
    _CANCEL.clear()


def freeze(driver: Any = None) -> None:
    """STOP: cancel any running primitive and freeze the arm in place with torque on (never raises)."""
    _CANCEL.set()
    drv = driver if driver is not None else driver_mod.get_driver()
    drv.freeze()


# ------------------------------------------------------------------------------------------------------------
# config helpers
# ------------------------------------------------------------------------------------------------------------


def handoff_zone(scene: dict | None = None) -> tuple[float, float, float]:
    """The handoff zone (x, y, z): from the scene's `zones.handoff` when given, else config/scene.yaml."""
    zone = None
    if isinstance(scene, dict):
        zones = scene.get("zones")
        zone = zones.get("handoff") if isinstance(zones, dict) else None
    if not isinstance(zone, dict):
        cfg = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
        zone = cfg["zones"]["handoff"]
    xyz = (float(zone["x"]), float(zone["y"]), float(zone["z"]))
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError(f"non-finite handoff zone {xyz}")
    return xyz


def policy_config() -> dict[str, Any]:
    """config/rl.yaml: `{"use_policy": bool, "policy": path}` (use_policy False when the file is missing)."""
    try:
        cfg = yaml.safe_load(RL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        cfg = None
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "use_policy": bool(cfg.get("use_policy", False)),
        "policy": str(cfg.get("policy") or "rl/policies/handoff_best.zip"),
    }


def checked_policy(policy: Any) -> dict[str, Any]:
    """A per-call policy override in the `policy_config()` shape, returned as a fresh dict: exactly the keys
    `use_policy` (a bool) and `policy` (a non-empty path, a str or `os.PathLike`, returned as a str;
    relative paths count from the repo root like config/rl.yaml's). ValueError for anything else, so a bad
    override ends a carry before it moves."""
    if not isinstance(policy, dict):
        raise ValueError(f"policy must be a dict like policy_config(), got {type(policy).__name__}")
    expected = {"use_policy", "policy"}
    if set(policy) != expected:
        raise ValueError(
            f"policy must have exactly the keys {sorted(expected)}, got {sorted(policy, key=str)}"
        )
    if not isinstance(policy["use_policy"], bool):
        raise ValueError(f"policy['use_policy'] must be a bool, got {policy['use_policy']!r}")
    raw = policy["policy"]
    path = os.fspath(raw) if isinstance(raw, (str, os.PathLike)) else None
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"policy['policy'] must be a non-empty path, got {raw!r}")
    return {"use_policy": policy["use_policy"], "policy": path}


def _home_target() -> list[float]:
    """`home_q` from config/scene.yaml as 5 finite floats (ValueError otherwise)."""
    q = [float(v) for v in ik.home_q()]
    if not all(math.isfinite(v) for v in q):
        raise ValueError(f"{SCENE_PATH}: home_q must be finite, got {q}")
    return q


def _pacer(driver: Any, realtime: bool | None, clock: Any, sleep: Any) -> Pacer:
    rt = (not getattr(driver, "is_fake", False)) if realtime is None else bool(realtime)
    return Pacer(rt, clock, sleep)


def _checked_step(max_step_rad: Any) -> float:
    """`max_step_rad` as a float in (0, MAX_STEP_RAD]; anything else is a caller bug (ValueError)."""
    step = float(max_step_rad)
    if not (math.isfinite(step) and 0.0 < step <= MAX_STEP_RAD + STEP_TOL):
        raise ValueError(
            f"max_step_rad must be a step in (0, {math.degrees(MAX_STEP_RAD):.0f} deg] per tick, got "
            f"{max_step_rad!r}"
        )
    return step


def _checked_timeout(timeout: Any) -> float:
    """`timeout` as a finite float (NaN or infinity would never fall back to the handoff zone)."""
    seconds = float(timeout)
    if not math.isfinite(seconds):
        raise ValueError(f"timeout must be a finite number of seconds, got {timeout!r}")
    return seconds


def _failure(driver: Any, session: Session, exc: BaseException) -> dict[str, Any]:
    driver.freeze()
    session.emit(f"error: {type(exc).__name__}: {exc}")
    log.exception("primitive failed; arm frozen")
    return {"ok": False, "reason": "error", "events": session.events}


# ------------------------------------------------------------------------------------------------------------
# home
# ------------------------------------------------------------------------------------------------------------


def home(
    driver: Any = None,
    *,
    max_step_rad: float | None = None,
    scene: Callable[[], dict | None] | None = None,
    on_event: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    realtime: bool | None = None,
    hand_freeze: bool = True,
) -> dict[str, Any]:
    """Move to `home_q` at <= `max_step_rad` (default HOME_STEP_RAD, 1 degree) per joint per tick along a
    validated plan (see the module docstring).

    `scene` gives the hand check its data; `hand_freeze=False` (carry_to_hand's final leg, which starts
    right above the hand) keeps only the speed cap. Returns `{"ok", "reason", "events"}`.
    """
    drv = driver if driver is not None else driver_mod.get_driver()
    clear_cancel()
    events: list[str] = []
    try:
        step = _checked_step(HOME_STEP_RAD if max_step_rad is None else max_step_rad)
        session = Session(
            drv,
            scene,
            _pacer(drv, realtime, clock, sleep),
            on_event,
            events=events,
            hand_freeze=hand_freeze,
            cancelled=cancelled,
        )
        q_home = _home_target()
    except Exception as exc:  # a bad argument, config or arm.ik unavailable: nothing moved, nothing frozen
        return setup_failure(events, on_event, exc)
    try:
        session.start()
        session.require_ready()
        if session.within(q_home, HOME_TOL_RAD):
            session.emit("home: already there")
            return {"ok": True, "reason": None, "events": events}
        path = _plan_home(session, q_home, step)
        session.emit(f"home: {len(path)} steps")
        session.stream(path, where="home")
        session.emit("home: done")
        return {"ok": True, "reason": None, "events": events}
    except Abort as abort:
        return abort.result(events)
    except Exception as exc:
        return _failure(drv, session, exc)


def _relaxed_box(box: dict[str, Sequence[float]], start_tip: Sequence[float]) -> dict[str, list[float]]:
    """The box for a sweep that starts outside it: anything behind the near edge, never above the ceiling,
    never lower than the start, never further out or to the side than the box."""
    return {
        "x": [-1.0, float(box["x"][1])],
        "y": [float(box["y"][0]), float(box["y"][1])],
        "z": [min(float(box["z"][0]), float(start_tip[2]) - START_BELOW_SLACK_M), float(box["z"][1])],
    }


def _raised_pose(session: Session) -> list[float] | None:
    """A pose with the tip lifted above the present one (same x, y and pitch), or None when IK fails."""
    (x, y, z), pitch = ik.fk(session.q)
    z_up = min(float(session.box["z"][1]) - 0.01, max(z + RAISE_M, RAISE_MIN_Z))
    if z_up <= z:
        return None
    q_up, ok = ik.solve((x, y, z_up), pitch, q0=session.q)
    return q_up if ok else None


def _plan_home(session: Session, q_home: list[float], max_step_rad: float) -> list[list[float]]:
    """The validated path to home: `Session.plan_path` (the direct sweep, else the straight line), else the
    same via a raised pose; from outside the box the three again with the relaxed box, which then also
    becomes the box every tick is checked against. Raises Abort("error") with the problems in the events
    when nothing validates."""
    boxes: list[tuple[str, dict[str, Any]]] = [("", session.box)]
    if not inside_box(session.tip, session.box):
        boxes.append(
            (
                " (start outside the box: behind its near edge, never lower)",
                _relaxed_box(session.box, session.tip),
            )
        )
    q_mid = _raised_pose(session)
    problems: list[str] = []
    for note, box in boxes:
        path, problems, how = session.plan_path(q_home, max_step_rad=max_step_rad, box=box)
        if path is None and q_mid is not None:
            first, _, how1 = session.plan_path(q_mid, max_step_rad=max_step_rad, box=box)
            if first is not None:
                second, _, how2 = session.plan_path(q_home, start=q_mid, max_step_rad=max_step_rad, box=box)
                if second is not None:
                    path, how = first + second, f"via a raised pose ({how1}, then {how2})"
        if path is not None:
            session.gate_box = box
            if note or how != "direct":
                session.emit(f"home: sweep {how}{note}")
            return path
    for problem in problems[:5]:
        session.emit(f"home refused: {problem}")
    raise Abort("error")


# ------------------------------------------------------------------------------------------------------------
# carry_to_hand
# ------------------------------------------------------------------------------------------------------------


def carry_to_hand(
    obj: str | None,
    *,
    driver: Any = None,
    scene: Callable[[], dict | None] | None = None,
    on_event: Callable[[str], None] | None = None,
    timeout: float = 10.0,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    realtime: bool | None = None,
    policy: dict | None = None,
) -> dict[str, Any]:
    """Hand the held `obj` to the tracked hand: rise to the carry height, travel above the target, then
    track and descend (see the module docstring). Ends at home with the event `handoff_done` (also sent
    through `on_event`, so the orchestrator can count the run event).

    `policy=None` follows config/rl.yaml (`policy_config()`); a dict `{"use_policy": bool, "policy": path}`
    overrides it for this call only (`checked_policy`; a malformed one is a setup failure: `error`,
    nothing moved)."""
    drv = driver if driver is not None else driver_mod.get_driver()
    clear_cancel()
    events: list[str] = []
    try:
        seconds = _checked_timeout(timeout)
        override = None if policy is None else checked_policy(policy)
        session = Session(
            drv,
            scene,
            _pacer(drv, realtime, clock, sleep),
            on_event,
            events=events,
            hand_freeze=False,
            cancelled=cancelled,
        )
        _home_target()
    except Exception as exc:  # a bad argument, config or arm.ik unavailable: nothing moved, nothing frozen
        return setup_failure(events, on_event, exc)
    try:
        session.start()
        session.require_ready()
        session.emit(f"carry_to_hand {obj or ''}: start".replace("  ", " "))
        if seconds > TIMEOUT_MAX_S:
            session.emit(f"timeout {seconds:g} s capped at {TIMEOUT_MAX_S:g} s")
            seconds = TIMEOUT_MAX_S
        target = _Target(session)
        _carry_high(session, obj, target)
        if override is None:
            cfg = policy_config()
        else:
            cfg = override
            session.emit(f"policy override: use_policy={cfg['use_policy']}, policy={cfg['policy']}")
        if cfg["use_policy"]:
            _run_policy(session, cfg, seconds, target)
        _track_hand(session, seconds, target)
        session.open_gripper(where="release")
        session.emit("released")
        session.rise(BACKOFF_M, where="back off")
        pacer = session.pacer
        result = home(
            driver=drv,
            scene=scene,
            on_event=on_event,
            clock=pacer.now,
            sleep=pacer.sleep,
            realtime=pacer.realtime,
            hand_freeze=False,
        )
        events.extend(result["events"])  # home() already passed each through on_event
        if not result["ok"]:
            raise Abort(result["reason"])
        session.refresh()
        session.emit("handoff_done")
        return {"ok": True, "reason": None, "events": events}
    except Abort as abort:
        return abort.result(events)
    except Exception as exc:
        return _failure(drv, session, exc)


def _run_policy(session: Session, cfg: dict[str, Any], timeout: float, target: _Target) -> bool:
    """Reach the hand with the RL policy; False (IK takes over) when it is unavailable or gives up."""
    try:
        from rl.policy_runner import run_policy  # lazy: rl/ is Lane D's and absent from the sandbox bundle
    except ImportError as exc:
        log.warning("the policy was asked for but rl.policy_runner is unavailable (%s): IK", exc)
        session.emit("policy unavailable: tracking with IK")
        return False
    policy_path = ROOT / cfg["policy"]
    session.emit(f"policy: {cfg['policy']}")

    def policy_scene() -> dict:
        session.check_stopped()
        scene = session.scene_dict() or {}
        hand = session.hand()
        target.goal(session, hand, fallback=False)
        if target.rejected_hand:
            # Share the latched fallback with the policy without changing tracker data.
            return {
                **scene,
                "hand": {"seen": False},
                "zones": {**scene.get("zones", {}), "handoff": dict(zip("xyz", target.zone, strict=True))},
            }
        return scene

    reached = bool(
        run_policy(
            session.driver,
            policy_scene,
            str(policy_path),
            timeout=timeout,
            safety_scene=session.scene_fn,
            cancelled=session.cancelled,
            clock=session.pacer.now,
            sleep=session.pacer.sleep,
            on_event=session.emit,
        )
    )
    session.refresh()  # the policy moved the arm through the driver
    session.check_stopped()
    session.emit("policy reached the hand" if reached else "policy gave up: tracking with IK")
    # The policy owns another Pacer on this clock. Discard our old schedule before IK resumes;
    # otherwise its first late tick resyncs without sleeping and the next command is sent immediately.
    session.pacer.resync()
    return reached


def _clamp_point(point: Sequence[float], box: dict[str, Sequence[float]]) -> tuple[float, float, float]:
    x, y, z = (
        min(max(float(v), float(box[a][0])), float(box[a][1])) for a, v in zip("xyz", point, strict=True)
    )
    return (x, y, z)


class _Target:
    """Target shared by travel, policy and tracking. Out-of-box hands latch the handoff zone;
    valid hand targets near the boundary are inset to keep the tip inside too."""

    def __init__(self, session: Session) -> None:
        self.zone = handoff_zone(session.scene_dict())
        self.box = shrunk_box(session.box, TARGET_INSET_M)
        if not inside_box(self.zone, self.box):
            raise ValueError("handoff zone is outside the workspace box with the required target inset")
        self.rejected_hand = False
        self._noted = False

    def goal(self, session: Session, hand: Vec3 | None, *, fallback: bool) -> Vec3:
        if hand is not None and not fallback and not self.rejected_hand:
            hand_target = (hand[0], hand[1], hand[2] + HAND_ABOVE_M)
            if not inside_box(hand_target, session.box):
                self.rejected_hand = True
                session.emit("hand target outside the workspace box: falling back to the handoff zone")
        raw = (
            self.zone
            if fallback or self.rejected_hand or hand is None
            else (hand[0], hand[1], hand[2] + HAND_ABOVE_M)
        )
        goal = _clamp_point(raw, self.box)
        if goal != tuple(float(v) for v in raw) and not self._noted:
            self._noted = True
            session.emit(f"target near the workspace edge: keeping the {TARGET_INSET_M:.2f} m target inset")
        return goal


def _positive(value: Any) -> float | None:
    """`value` as a finite float > 0, else None (bools and non-numbers too)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _item_height(entry: Any) -> float | None:
    return _positive(entry.get("h")) if isinstance(entry, dict) else None


def _item_reach(entry: Any) -> float | None:
    """The longest half-extent of an object (center to the farthest point, whatever its tilt), the sim's
    rule: hypot(radius, h / 2) for a cylinder (`size[1]` is its diameter, as sim/model/build_scene.py
    builds it), the half-diagonal of the box otherwise (an entry without `shape`, like a scene entry, counts
    as a box, which is never smaller)."""
    if not isinstance(entry, dict):
        return None
    size, h = entry.get("size"), _item_height(entry)
    if not isinstance(size, (list, tuple)) or len(size) != 2 or h is None:
        return None
    length, width = _positive(size[0]), _positive(size[1])
    if length is None or width is None:
        return None
    if entry.get("shape") == "cylinder":
        return math.hypot(width / 2, h / 2)
    return math.sqrt((length / 2) ** 2 + (width / 2) ** 2 + (h / 2) ** 2)


def carry_height(
    obj: str | None,
    scene: Any,
    *,
    ceiling: float,
    objects: dict[str, dict[str, Any]] | None = None,
) -> tuple[float | None, str]:
    """The grasp-center height of the carry by the sim's rule (`arm.sim_robot.SimRobot._carry_clearance`):
    the held object's longest half-extent (`_item_reach`) plus CARRY_CLEARANCE_M above the tallest
    obstacle (seen or not: an obstacle never moves and keeps its last pose) or seen object other than
    `obj`. Refused if this would exceed `ceiling` minus CEILING_MARGIN_M.

    Heights and sizes come from config/objects.yaml (`objects`, default `arm.replay.load_objects()`) by
    name, else from the scene's own entry (which the scene contract copies from that file). Returns
    `(height, how)` with a short description of the sum, or `(None, why)` when the scene, a height or the
    held object's size is missing, or the required clearance exceeds the ceiling."""
    if not isinstance(scene, dict):
        return None, "no scene data"
    config = load_objects() if objects is None else objects
    if not obj:
        return None, "the held object is not named"
    held_scene = scene.get("objects", {}).get(obj) if isinstance(scene.get("objects"), dict) else None
    reach = _item_reach(config.get(obj))
    reach = reach if reach is not None else _item_reach(held_scene)
    if reach is None:
        return None, f"no size for the held {obj}"
    tallest, tallest_name = 0.0, "the table"
    for group in ("obstacles", "objects"):
        items = scene.get(group)
        if not isinstance(items, dict):
            return None, f"the scene has no {group} mapping"
        for name, item in items.items():
            if group == "objects" and (name == obj or not (isinstance(item, dict) and item.get("seen"))):
                continue
            h = _item_height(config.get(name))
            h = h if h is not None else _item_height(item)
            if h is None:
                return None, f"no height for {name}"
            if h > tallest:
                tallest, tallest_name = h, str(name)
    height = tallest + reach + CARRY_CLEARANCE_M
    how = f"{tallest_name} {tallest:.2f} m + {obj} {reach:.3f} m + {CARRY_CLEARANCE_M:.2f} m"
    cap = float(ceiling) - CEILING_MARGIN_M
    if height > cap:
        return None, f"required height {height:.3f} m exceeds the safe ceiling {cap:.3f} m ({how})"
    return height, how


def _hold_error(session: Session, detail: str) -> None:
    """Abort a carry without releasing its object or attempting another motion."""
    session.driver.freeze()
    session.emit(f"error: {detail}; holding the object")
    raise Abort("error")


def _carry_high(session: Session, obj: str | None, target: _Target) -> None:
    """Legs 1 and 2 of the carry (module docstring): rise at the present xy to the carry height, then
    travel at that height to above the target. A failed rise aborts holding the object; bounded travel
    can end before the target and continue with validated tracking. Raises Abort on STOP or a box gate."""
    session.check_stopped()
    height, how = carry_height(obj, session.scene_dict(), ceiling=float(session.box["z"][1]))
    if height is None:
        _hold_error(session, f"carry clearance unavailable ({how})")
    target.goal(session, session.hand(), fallback=False)
    session.emit(f"carry height {height:.2f} m ({how})")
    (x, y, z), _ = ik.fk_grasp(session.q)
    if z < height - RISE_DONE_M:
        path, why = _plan_rise(session, height)
        if path is None:
            _hold_error(session, f"carry rise to {height:.2f} m refused ({why})")
        session.emit(f"carry: rising {(height - z) * 100:.1f} cm at ({x:.2f}, {y:.2f}), {len(path)} steps")
        session.stream(path, where="carry rise")
        session.refresh()
        reached = ik.fk_grasp(session.q)[0]
        if reached[2] < height - RISE_DONE_M or math.hypot(reached[0] - x, reached[1] - y) > CARRY_LINE_TOL_M:
            _hold_error(session, f"carry rise did not reach its clearance height {height:.2f} m")
    _travel(session, height, target)


def _plan_rise(session: Session, height: float) -> tuple[list[list[float]] | None, str]:
    """The rise as one validated path, or (None, why). The grasp center goes straight up from where it is
    to `height`: IK (`arm.ik.solve_grasp`) every LINE_SAMPLE_M, the pitch blended linearly from the present
    one to the steepest pitch, from START_PITCH toward horizontal in PITCH_STEP_RAD steps, that reaches the
    top exactly (the sim's rise also asks for the steepest pitch), consecutive samples bridged at <=
    MAX_STEP_RAD per joint. Refused when a sample has no IK solution or needs a joint reconfiguration
    (LINE_JUMP_RAD), when a waypoint leaves the limits or the box inset by PLAN_INSET_M (the box's top is
    the 0.30 m ceiling), strays more than CARRY_LINE_TOL_M from the vertical line, or dips more than
    DIVE_TOL_M under the start."""
    q = [float(v) for v in session.q]
    (x0, y0, z0), pitch0 = ik.fk_grasp(q)
    top = (x0, y0, float(height))
    pitch_top: float | None = None
    for k in range(round(-START_PITCH / PITCH_STEP_RAD) + 1):  # -60, -55, ... 0 degrees
        pitch = START_PITCH + k * PITCH_STEP_RAD
        q_top, ok = ik.solve_grasp(top, pitch, q0=q)
        if ok and abs(ik.fk_grasp(q_top)[1] - pitch) <= ik.PITCH_TOL:
            pitch_top = pitch
            break
    if pitch_top is None:
        return None, f"no IK solution at ({x0:.2f}, {y0:.2f}, {height:.2f})"
    samples = max(1, math.ceil((height - z0) / LINE_SAMPLE_M))
    path: list[list[float]] = []
    q_prev = q
    for k in range(1, samples + 1):
        frac = k / samples
        point = (x0, y0, z0 + (height - z0) * frac)
        q_next, ok = ik.solve_grasp(point, pitch0 + (pitch_top - pitch0) * frac, q0=q_prev)
        if not ok:
            return None, f"no IK solution at ({x0:.2f}, {y0:.2f}, {point[2]:.2f})"
        q_next = [float(v) for v in q_next]
        jump = max(abs(a - b) for a, b in zip(q_next, q_prev, strict=True))
        if jump > LINE_JUMP_RAD:
            return None, f"joint reconfiguration of {math.degrees(jump):.0f} deg at z {point[2]:.2f}"
        path.extend(bridge(q_prev, q_next, MAX_STEP_RAD))
        q_prev = q_next
    problems = validate_path(path, session.limits, ik.fk_tip, shrunk_box(session.box, PLAN_INSET_M))
    if problems:
        return None, problems[0]
    for waypoint in path:
        (x, y, z), _ = ik.fk_grasp(waypoint)
        if math.hypot(x - x0, y - y0) > CARRY_LINE_TOL_M:
            return None, f"the grasp center would stray {math.hypot(x - x0, y - y0) * 100:.1f} cm sideways"
        if z < z0 - DIVE_TOL_M:
            return None, f"the grasp center would dip to {z:.3f} m"
    if ik.fk_grasp(path[-1])[0][2] < height - RISE_DONE_M:
        return None, "the rise ends below the required clearance height"
    return path, ""


class _HandLog:
    """The 10 Hz hand line of the travel and tracking legs: while the wristband is seen, every cycle emits
    its xy and its distance from the grasp center with the leg's clock (`<where> t=1.2 s: hand (x, y),
    d m from the grasp center`), so the hand speed of the A10 check (5 cm/s or slower) can be read from
    the log; one `hand unseen` line when it disappears, nothing while there is no hand data."""

    def __init__(self, session: Session, where: str) -> None:
        self.session = session
        self.where = where
        self.seen = False

    def __call__(self, hand: Vec3 | None, grasp: Sequence[float], t: float) -> None:
        if hand is None:
            if self.seen:
                self.session.emit(f"{self.where} t={t:.1f} s: hand unseen")
            self.seen = False
            return
        self.seen = True
        self.session.emit(
            f"{self.where} t={t:.1f} s: hand ({hand[0]:.3f}, {hand[1]:.3f}), "
            f"{math.dist(grasp, hand):.3f} m from the grasp center"
        )


def _travel(session: Session, height: float, target: _Target) -> None:
    """Leg 2: straight lines of the grasp center toward the point `height` above the target, re-aimed at
    TRACK_HZ (the target follows the hand), each cycle's path from `_servo_path` with the floor `height -
    CARRY_LINE_TOL_M` and no joint-space sweep. Ends above the target (within ABOVE_XY_M), where no straight
    step at that height validates, after TRAVEL_MAX_S, or after STALL_CYCLES cycles with the same clock
    reading; the tracking goes on from wherever it ended. Every cycle logs the seen hand (`_HandLog`)."""
    pacer = session.pacer
    t0 = pacer.now()
    last_now: float | None = None
    stalled = 0
    last_note = ""
    hand_log = _HandLog(session, "carry travel")
    while True:
        session.check_stopped()
        now = pacer.now()
        stalled = stalled + 1 if now == last_now else 0
        last_now = now
        hand = session.hand()
        goal = target.goal(session, hand, fallback=False)
        (x, y, z), _ = ik.fk_grasp(session.q)
        hand_log(hand, (x, y, z), now - t0)
        if math.hypot(goal[0] - x, goal[1] - y) <= ABOVE_XY_M:
            session.emit(f"carry: above the target at {z:.2f} m")
            return
        if now - t0 >= TRAVEL_MAX_S or stalled >= STALL_CYCLES:
            session.emit(
                f"carry: travel at {height:.2f} m cut short at ({x:.2f}, {y:.2f}): tracking from here"
            )
            return
        aim = (goal[0], goal[1], height)
        path, note = _servo_path(session, aim, START_PITCH, floor_z=height - CARRY_LINE_TOL_M, sweep=False)
        if path is None:
            session.emit(
                f"carry: no straight travel at {height:.2f} m beyond ({x:.2f}, {y:.2f}) toward "
                f"({goal[0]:.2f}, {goal[1]:.2f}): tracking from here"
            )
            return
        if note and note != last_note:
            session.emit(f"carry travel: {note}")
        last_note = note
        session.stream(path, where="carry travel")


def _pitch_candidates(pitch_now: float, pitch_goal: float) -> list[float]:
    """The pitches one cycle tries, in order: the present pitch moved toward `pitch_goal` by at most
    PITCH_STEP_RAD, the present pitch, then shallower pitches (toward horizontal, never past it) in
    PITCH_STEP_RAD steps for a reach that needs them: IK's own ladder would jump a full 15 degree rung."""
    blend = pitch_now + max(-PITCH_STEP_RAD, min(PITCH_STEP_RAD, pitch_goal - pitch_now))
    out: list[float] = []
    for pitch in (blend, pitch_now, *(min(0.0, pitch_now + k * PITCH_STEP_RAD) for k in (1, 2, 3))):
        if not any(abs(pitch - seen) < 1e-9 for seen in out):
            out.append(pitch)
    return out


def _grasp_dives(path: Sequence[Sequence[float]], z_floor: float) -> bool:
    return any(ik.fk_grasp(q)[0][2] < z_floor for q in path)


def _servo_path(
    session: Session,
    goal: Vec3,
    pitch_goal: float,
    *,
    floor_z: float | None = None,
    sweep: bool = True,
) -> tuple[list[list[float]] | None, str]:
    """The validated joint-space path one 10 Hz cycle executes, or None (holding), with a note for the
    events ("" when all is well).

    The target is the point at most one cycle's tip travel (TICKS_PER_CYCLE ticks at the speed cap) ahead
    on the straight line from the grasp center to `goal`, solved with `arm.ik.solve_grasp` for each pitch
    of `_pitch_candidates`. A solution is used when `bridge(q, solution, MAX_STEP_RAD)` validates inside
    the box inset by PLAN_INSET_M and never takes the grasp center lower than the start or the goal by more
    than DIVE_TOL_M (or lower than `floor_z` when given: the travel at the carry height); the first one
    within SERVO_JUMP_RAD (what the cycle's ticks execute) wins, else the half step is tried, else the
    nearest solution within LINE_JUMP_RAD runs to its end before the next re-aim. When no point on the line
    is reachable, the first TICKS_PER_CYCLE waypoints of the validated joint-space sweep to the goal's pose
    run (unless `sweep` is False). Every path returned here is exactly what is streamed, so the executed
    motion is always validated (never a per-joint step toward a far pose).
    """
    q = [float(v) for v in session.q]
    grasp, pitch_now = ik.fk_grasp(q)
    z_floor = min(grasp[2], goal[2]) - DIVE_TOL_M if floor_z is None else float(floor_z)
    inset = shrunk_box(session.box, PLAN_INSET_M)
    remaining = math.dist(grasp, goal)
    reach = TICKS_PER_CYCLE * session.tip_cap(session.tip)
    pitches = _pitch_candidates(pitch_now, pitch_goal)

    def checked(q_next: Sequence[float], max_jump: float) -> tuple[float, list[list[float]] | None]:
        target = [float(v) for v in q_next]
        jump = max(abs(a - b) for a, b in zip(target, q, strict=True))
        if jump > max_jump + STEP_TOL:
            return jump, None
        path = bridge(q, target, MAX_STEP_RAD)
        if validate_path(path, session.limits, ik.fk_tip, inset) or _grasp_dives(path, z_floor):
            return jump, None
        return jump, path

    steps = [min(remaining, reach)]
    if remaining > reach / 2:
        steps.append(reach / 2)
    far: tuple[float, list[list[float]]] | None = None
    for step in steps:
        frac = step / remaining if remaining > 0.0 else 1.0
        point = tuple(g + (t - g) * frac for g, t in zip(grasp, goal, strict=True))
        for pitch in pitches:
            q_next, ok = ik.solve_grasp(point, pitch, q0=q)
            if not ok:
                continue
            jump, path = checked(q_next, LINE_JUMP_RAD)
            if path is None:
                continue
            if jump <= SERVO_JUMP_RAD + STEP_TOL:
                return path, ""
            if far is None or jump < far[0]:
                far = (jump, path)
    if far is not None:
        return far[1], f"re-aiming with a {math.degrees(far[0]):.0f} deg joint move"
    if sweep:
        q_full, ok = ik.solve_grasp(goal, pitch_goal, q0=q)
        if ok:
            head = bridge(q, q_full, MAX_STEP_RAD)[:TICKS_PER_CYCLE]
            if not (validate_path(head, session.limits, ik.fk_tip, inset) or _grasp_dives(head, z_floor)):
                return head, "straight approach blocked: joint-space sweep toward the target"
    x, y, z = goal
    return None, f"no IK solution toward ({x:.2f}, {y:.2f}, {z:.2f}): holding"


def _track_hand(session: Session, timeout: float, target: _Target | None = None) -> None:
    """Leg 3: straight-line tracking at TRACK_HZ until the release condition holds (the hand steady within
    RELEASE_WITHIN_M of the grasp center for RELEASE_STEADY_S, or the handoff zone reached after the
    timeout, which counts from here). Invalid hand targets latch the zone fallback. Every release
    decision uses encoder feedback. A failed fallback, MAX_TRACK_S, or STALL_CYCLES with the same clock
    reading aborts holding the object instead of releasing at an unverified position. Every cycle logs
    the seen hand against the measured grasp center (`_HandLog`)."""
    pacer = session.pacer
    t0 = pacer.now()
    deadline = t0 + timeout
    fallback_deadline = deadline + FALLBACK_EXTRA_S
    target = _Target(session) if target is None else target
    zone = target.zone
    near_since: float | None = None
    fallback = session.scene_fn is None or target.rejected_hand
    if fallback:
        fallback_deadline = t0 + FALLBACK_EXTRA_S
        session.emit("using the handoff zone")
    last_note = ""
    last_now: float | None = None
    stalled = 0
    hand_log = _HandLog(session, "carry_to_hand")
    while True:
        session.check_stopped()
        now = pacer.now()
        stalled = stalled + 1 if now == last_now else 0
        last_now = now
        hand = session.hand()
        goal = target.goal(session, hand, fallback=fallback)
        if not fallback and target.rejected_hand:
            fallback = True
            fallback_deadline = now + FALLBACK_EXTRA_S
        if not fallback and now >= deadline:
            fallback = True
            fallback_deadline = now + FALLBACK_EXTRA_S
            session.emit(f"no steady hand within {timeout:g} s: falling back to the handoff zone")
            goal = target.zone
        session.refresh()  # release decisions use encoder feedback, never the last commanded pose
        grasp_center, _ = ik.fk_grasp(session.q)
        hand_log(hand, grasp_center, now - t0)
        if not fallback and hand is not None:
            if math.dist(grasp_center, hand) <= RELEASE_WITHIN_M:
                near_since = now if near_since is None else near_since
                if now - near_since >= RELEASE_STEADY_S:
                    session.emit(f"hand steady within {RELEASE_WITHIN_M:.2f} m for {RELEASE_STEADY_S:g} s")
                    return
            else:
                near_since = None
        else:
            near_since = None
        if fallback:
            if math.dist(grasp_center, zone) <= ZONE_REACHED_M:
                session.emit("at the handoff zone")
                return
            if now >= fallback_deadline:
                _hold_error(session, "handoff zone was not reached before the fallback deadline")
        if stalled >= STALL_CYCLES:
            _hold_error(session, f"the clock did not advance for {STALL_CYCLES} cycles")
        if now - t0 >= MAX_TRACK_S:
            _hold_error(session, f"tracking ran for {MAX_TRACK_S:g} s without a release position")
        path, note = _servo_path(session, goal, START_PITCH)
        if note and note != last_note:
            session.emit(note)
        last_note = note
        if path is None:
            pacer.tick()  # holding: the clock still advances toward the deadlines
        else:
            session.stream(path, where="carry_to_hand")


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------


def confirm_arm_clear(seconds: int = 3, watcher: Any = None) -> bool:
    """Ask the human to confirm the arm is clear, then count down; False to abort (Esc, no, EOF)."""
    print("This moves the REAL arm. Keep hands out of the taped box; a teammate holds the 12 V plug.")
    try:
        answer = input("Type yes when the arm is clear: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Not confirmed; nothing moved.")
        return False
    for n in range(seconds, 0, -1):
        print(f"Moving the REAL arm in {n}... keep clear (Esc aborts)", flush=True)
        for _ in range(10):
            time.sleep(0.1)
            if watcher is not None and watcher.pressed:
                print("Aborted before connecting.")
                return False
    return True


def _fake_scene(hand_seen: bool) -> Callable[[], dict]:
    scene = json.loads(FAKE_SCENE_PATH.read_text(encoding="utf-8"))
    scene["hand"]["seen"] = hand_seen
    return lambda: scene


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arm.real_robot",
        description="Live primitives. Nothing touches hardware without --real.",
    )
    parser.add_argument("--real", action="store_true", help="use the real arm: confirmation, countdown, Esc")
    parser.add_argument("--fake", action="store_true", help="use the in-memory FakeDriver (no hardware)")
    parser.add_argument("--carry-test", action="store_true", help="run carry_to_hand on --object")
    parser.add_argument("--home", action="store_true", help="send the arm home at 30 deg/s")
    parser.add_argument("--object", default="pill_bottle", help="object name for --carry-test")
    parser.add_argument("--timeout", type=float, default=10.0, help="carry_to_hand timeout in seconds")
    hand = parser.add_mutually_exclusive_group()
    hand.add_argument(
        "--scene-file",
        help="JSON scene file re-read every tick (hand position); with --real only the file of a running "
        "`python -m vision.scene --out FILE` on the real camera (its FILE.source.json says live)",
    )
    hand.add_argument(
        "--tracker",
        action="store_true",
        help="hand position from the live overhead SceneTracker (vision.scene.get_tracker(); refused with "
        "--real when it is a fake or shows no frame in 5 s); with --fake the FakeSceneTracker",
    )
    parser.add_argument("--port", help="serial port for --real; default FOLLOWER_PORT from .env")
    parser.add_argument("--id", dest="follower_id", help="LeRobot id for --real; default from config")
    cli_args = list(sys.argv[1:] if argv is None else argv)  # the command line, for the Esc record
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.carry_test or args.home):
        parser.print_help()
        return 2

    def run(drv: Any, scene: Any, realtime: bool) -> dict[str, Any]:
        if args.carry_test:
            return carry_to_hand(
                args.object, driver=drv, scene=scene, timeout=args.timeout, realtime=realtime
            )
        return home(driver=drv, scene=scene, realtime=realtime)

    fake_mode = args.fake or (not args.real and _fake("FAKE_ROBOT"))
    if not fake_mode and not args.real:
        print("Refusing to touch the arm without --real (use --fake for the in-memory arm).", file=sys.stderr)
        return 2
    scene: Callable[[], dict | None] | None = None
    if not args.tracker:
        # with --real only a live vision.scene writer's file (its provenance sidecar) is accepted
        scene, refused = open_scene_file(args.scene_file, real=not fake_mode)
        if refused:
            return 2
    if not fake_mode:
        from dotenv import load_dotenv

        load_dotenv()  # before the tracker: FAKE_CAMERA from .env makes it a fake, which --real refuses
    tracker: Any = None
    if args.tracker:
        try:
            # the fake shows the hand, like the fake scene below: a carry test needs one to follow
            tracker, scene = tracker_scene(fake_mode, hand_seen=True if fake_mode else None)
        except Exception as exc:  # any failure to build or trust the tracker refuses before anything moves
            print(f"cannot use --tracker: {type(exc).__name__}: {exc}; nothing moved", file=sys.stderr)
            return 2
    try:
        if fake_mode:
            drv = driver_mod.FakeDriver()
            if args.carry_test:
                drv.command(drv.read()[0], 0.2)  # pretend the object is held
            result = run(drv, scene or _fake_scene(hand_seen=True), realtime=False)
            print(json.dumps(result))
            what = "carry" if args.carry_test else "home"
            print(f"fake {what} {'ok' if result['ok'] else result['reason']}: {len(drv.commands)} commands")
            return 0 if result["ok"] else 1

        if scene is None:
            print(
                "WARNING: no --scene-file or --tracker: no hand data; carry_to_hand releases at the "
                "handoff zone."
            )
        clear_cancel()  # discard an earlier session's cancellation before watching this startup
        drv = driver_mod.Driver(port=args.port, follower_id=args.follower_id)
        watcher = driver_mod.EscWatcher(lambda: freeze(drv))
        watcher.start()
        try:
            if not confirm_arm_clear(3, watcher):
                return 1
            if esc_during_connect(watcher, drv):
                return 1
            drv.connect()  # the only place this CLI powers the arm
            connected_at = time.monotonic()  # Esc presses before this are connect-phase (A13)
            try:
                if esc_during_connect(watcher, drv):
                    return 1
                result = run(drv, scene, realtime=True)
                print(json.dumps(result))
                if drv.stopped:
                    print("Arm is FROZEN with torque on. Press Enter to release the port (torque stays on).")
                    try:
                        input()
                    except EOFError:
                        pass
                return 0 if result["ok"] else 1
            finally:
                driver_mod.report_esc_latency(
                    watcher, drv, source="arm.real_robot", connected_at=connected_at, extra={"argv": cli_args}
                )
                drv.disconnect()  # torque stays on; the arm keeps holding its pose
        finally:
            watcher.stop()
    finally:
        if tracker is not None:
            close_tracker(tracker)


def esc_during_connect(watcher: Any, driver: Any) -> bool:
    """Latch every STOP source before and after connecting; refuse motion while stopped.

    A driver that is already stopped holds its pose, so it is not frozen a second time: that would
    re-stamp `frozen_at` and the Esc record would measure to the wrong freeze."""
    if not (getattr(watcher, "pressed", False) or getattr(driver, "stopped", False) or cancelled()):
        return False
    _CANCEL.set()
    if not getattr(driver, "stopped", False):
        freeze(driver)
    if getattr(driver, "connected", False):
        print("Esc pressed while connecting: arm frozen with torque on, nothing moved.")
    else:
        print("Stopped before connecting; nothing moved.")
    return True


if __name__ == "__main__":
    sys.exit(main())
