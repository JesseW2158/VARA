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
- `carry_to_hand(obj, *, driver, scene, on_event, timeout, ...)`: with the object held, move the grasp
  center (`arm.ik.fk_grasp`) toward the seen wristband + 0.05 m (the handoff zone when the hand is unseen)
  on a straight line re-aimed at 10 Hz. Each cycle solves IK (`arm.ik.solve_grasp`) for the point at most
  one cycle's tip travel ahead on that line, the pitch blended toward -60 degrees by at most PITCH_STEP_RAD
  (shallower pitches are tried where the reach needs them), and executes `bridge(q, solution)` at <= 2
  degrees per tick, all joints interpolated together, only after that bridge validated against the limits
  and the box (inset PLAN_INSET_M) and was checked never to take the grasp center lower than the start or
  the goal by more than DIVE_TOL_M (`_servo_path`). Nothing is ever stepped toward an unvalidated pose: a
  solution more than SERVO_JUMP_RAD away (what the cycle's ticks can execute) is only used up to
  LINE_JUMP_RAD and then run to its end before re-aiming; when no point on the line is reachable, the first
  ticks of the validated joint-space sweep to the goal's pose run, else the arm holds. The target is
  clamped TARGET_INSET_M inside the workspace box, so a hand at or beyond the tape is served at the
  nearest point inside and the tip never leaves the box. Once the hand has stayed within 0.08 m of the
  grasp center for 1 s the gripper opens; after `timeout` seconds (finite; capped at TIMEOUT_MAX_S) with no
  steady hand it opens at the handoff zone instead; MAX_TRACK_S of tracking, or a clock that stops
  advancing, ends the tracking where it is. Then it backs off 0.05 m up, goes home and emits
  `handoff_done`. With `use_policy: true` in
  config/rl.yaml the RL policy (`rl.policy_runner.run_policy`, imported lazily) does the reach instead of
  IK; when it cannot be imported, IK is used with a warning. The 10 cm hand freeze is off inside this
  primitive (it approaches the hand on purpose); the speed cap near the hand stays on.

Each primitive returns `{"ok": bool, "reason": None | "stopped" | "hand_too_close" | "error",
"events": [...]}` and never raises for driver, IK or scene failures: an error while moving freezes the arm;
a failure before anything can move (config files, arm.ik, a bad argument, torque off, a plan that does not
validate) returns `error` without freezing, and a driver that is already stopped returns `stopped`.

    python -m arm.real_robot --fake --carry-test             # in-memory arm, fake hand, no sleeping
    python -m arm.real_robot --real --carry-test [--scene-file scene.json]
    python -m arm.real_robot --real --home

`--real` loads .env, asks for confirmation, counts down 3 s, watches Esc (freeze) and connects the driver
only then; an Esc that lands during `connect()` (which resets the driver's stopped flag) is caught right
after it and freezes the arm before anything runs. A `--scene-file` that cannot be read stops the CLI
before anything connects. Nothing here imports `agents/`, `vision/` or `web/`; `rl.policy_runner` is
imported lazily and only when the config asks for the policy.
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
    MAX_STEP_RAD,
    PLAN_INSET_M,
    STEP_TOL,
    Abort,
    Pacer,
    Session,
    bridge,
    inside_box,
    scene_file_reader,
    setup_failure,
    shrunk_box,
    validate_path,
)

__all__ = [
    "BACKOFF_M",
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
    "cancelled",
    "carry_to_hand",
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
) -> dict[str, Any]:
    """Hand the held `obj` to the tracked hand (see the module docstring). Ends at home with the event
    `handoff_done` (also sent through `on_event`, so the orchestrator can count the run event)."""
    drv = driver if driver is not None else driver_mod.get_driver()
    clear_cancel()
    events: list[str] = []
    try:
        seconds = _checked_timeout(timeout)
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
        cfg = policy_config()
        if cfg["use_policy"]:
            _run_policy(session, cfg, seconds)
        _track_hand(session, seconds)
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


def _run_policy(session: Session, cfg: dict[str, Any], timeout: float) -> bool:
    """Reach the hand with the RL policy; False (IK takes over) when it is unavailable or gives up."""
    try:
        from rl.policy_runner import run_policy  # lazy: rl/ is Lane D's and absent from the sandbox bundle
    except ImportError as exc:
        log.warning("config/rl.yaml asks for the policy but rl.policy_runner is unavailable (%s): IK", exc)
        session.emit("policy unavailable: tracking with IK")
        return False
    policy_path = ROOT / cfg["policy"]
    session.emit(f"policy: {cfg['policy']}")
    reached = bool(run_policy(session.driver, session.scene_fn, str(policy_path), timeout=timeout))
    session.refresh()  # the policy moved the arm through the driver
    session.check_stopped()
    session.emit("policy reached the hand" if reached else "policy gave up: tracking with IK")
    return reached


def _clamp_point(point: Sequence[float], box: dict[str, Sequence[float]]) -> tuple[float, float, float]:
    x, y, z = (
        min(max(float(v), float(box[a][0])), float(box[a][1])) for a, v in zip("xyz", point, strict=True)
    )
    return (x, y, z)


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


def _servo_path(session: Session, goal: Vec3, pitch_goal: float) -> tuple[list[list[float]] | None, str]:
    """The validated joint-space path one 10 Hz cycle executes, or None (holding), with a note for the
    events ("" when all is well).

    The target is the point at most one cycle's tip travel (TICKS_PER_CYCLE ticks at the speed cap) ahead
    on the straight line from the grasp center to `goal`, solved with `arm.ik.solve_grasp` for each pitch
    of `_pitch_candidates`. A solution is used when `bridge(q, solution, MAX_STEP_RAD)` validates inside
    the box inset by PLAN_INSET_M and never takes the grasp center lower than the start or the goal by more
    than DIVE_TOL_M; the first one within SERVO_JUMP_RAD (what the cycle's ticks execute) wins, else the
    half step is tried, else the nearest solution within LINE_JUMP_RAD runs to its end before the next
    re-aim. When no point on the line is reachable, the first TICKS_PER_CYCLE waypoints of the validated
    joint-space sweep to the goal's pose run. Every path returned here is exactly what is streamed, so the
    executed motion is always validated (never a per-joint step toward a far pose).
    """
    q = [float(v) for v in session.q]
    grasp, pitch_now = ik.fk_grasp(q)
    z_floor = min(grasp[2], goal[2]) - DIVE_TOL_M
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
    q_full, ok = ik.solve_grasp(goal, pitch_goal, q0=q)
    if ok:
        sweep = bridge(q, q_full, MAX_STEP_RAD)[:TICKS_PER_CYCLE]
        if not validate_path(sweep, session.limits, ik.fk_tip, inset) and not _grasp_dives(sweep, z_floor):
            return sweep, "straight approach blocked: joint-space sweep toward the target"
    x, y, z = goal
    return None, f"no IK solution toward ({x:.2f}, {y:.2f}, {z:.2f}): holding"


def _track_hand(session: Session, timeout: float) -> None:
    """Straight-line tracking at TRACK_HZ until the release condition holds (the hand steady within
    RELEASE_WITHIN_M of the grasp center for RELEASE_STEADY_S, or the handoff zone reached after the
    timeout). Targets are clamped TARGET_INSET_M inside the workspace box. The tracking ends where it is
    after MAX_TRACK_S whatever the deadlines, and when the clock stops advancing (STALL_CYCLES cycles with
    the same reading), since the deadlines need it."""
    pacer = session.pacer
    t0 = pacer.now()
    deadline = t0 + timeout
    fallback_deadline = deadline + FALLBACK_EXTRA_S
    zone = handoff_zone(session.scene_dict())
    target_box = shrunk_box(session.box, TARGET_INSET_M)
    near_since: float | None = None
    fallback = session.scene_fn is None
    if fallback:
        session.emit("no hand data: releasing at the handoff zone")
    noted: set[str] = set()  # one-time notes: the clamp and each new IK failure streak
    last_note = ""
    last_now: float | None = None
    stalled = 0
    while True:
        session.check_stopped()
        now = pacer.now()
        stalled = stalled + 1 if now == last_now else 0
        last_now = now
        hand = session.hand()
        if not fallback and now >= deadline:
            fallback = True
            session.emit(f"no steady hand within {timeout:g} s: falling back to the handoff zone")
        grasp_center, _ = ik.fk_grasp(session.q)
        if fallback or hand is None:
            raw = zone
        else:
            raw = (hand[0], hand[1], hand[2] + HAND_ABOVE_M)
        goal = _clamp_point(raw, target_box)
        if goal != tuple(float(v) for v in raw) and "clamp" not in noted:
            noted.add("clamp")
            session.emit(
                f"target outside the workspace box (inset {TARGET_INSET_M:.2f} m): aiming at the nearest "
                "point inside"
            )
        if not fallback and hand is not None:
            if math.dist(grasp_center, hand) <= RELEASE_WITHIN_M:
                near_since = now if near_since is None else near_since
                if now - near_since >= RELEASE_STEADY_S:
                    session.emit(f"hand steady within {RELEASE_WITHIN_M:.2f} m for {RELEASE_STEADY_S:g} s")
                    return
            else:
                near_since = None
        if fallback and (math.dist(grasp_center, zone) <= ZONE_REACHED_M or now >= fallback_deadline):
            session.emit("at the handoff zone")
            return
        if stalled >= STALL_CYCLES:
            session.emit(f"the clock did not advance for {STALL_CYCLES} cycles: releasing here")
            return
        if now - t0 >= MAX_TRACK_S:
            session.emit(f"tracking ran for {MAX_TRACK_S:g} s: releasing here")
            return
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
    parser.add_argument("--scene-file", help="JSON scene file re-read every tick (hand position)")
    parser.add_argument("--port", help="serial port for --real; default FOLLOWER_PORT from .env")
    parser.add_argument("--id", dest="follower_id", help="LeRobot id for --real; default from config")
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

    try:
        scene = scene_file_reader(args.scene_file)
    except (OSError, ValueError) as exc:
        print(f"cannot read --scene-file {args.scene_file}: {exc}; nothing moved", file=sys.stderr)
        return 2
    if args.fake or (not args.real and _fake("FAKE_ROBOT")):
        drv = driver_mod.FakeDriver()
        if args.carry_test:
            drv.command(drv.read()[0], 0.2)  # pretend the object is held
        result = run(drv, scene or _fake_scene(hand_seen=True), realtime=False)
        print(json.dumps(result))
        print(
            f"fake {'carry' if args.carry_test else 'home'} {'ok' if result['ok'] else result['reason']}: "
            f"{len(drv.commands)} commands"
        )
        return 0 if result["ok"] else 1
    if not args.real:
        print("Refusing to touch the arm without --real (use --fake for the in-memory arm).", file=sys.stderr)
        return 2

    from dotenv import load_dotenv

    load_dotenv()
    if scene is None:
        print("WARNING: no --scene-file: no hand data; carry_to_hand releases at the handoff zone.")
    watcher = driver_mod.EscWatcher()
    watcher.start()
    try:
        if not confirm_arm_clear(3, watcher):
            return 1
        clear_cancel()  # only an Esc from here on counts
        drv = driver_mod.Driver(port=args.port, follower_id=args.follower_id)
        watcher.callback = lambda: freeze(drv)
        drv.connect()  # the only place this CLI powers the arm
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
            drv.disconnect()  # torque stays on; the arm keeps holding its pose
    finally:
        watcher.stop()


def esc_during_connect(watcher: Any, driver: Any) -> bool:
    """True when Esc landed while `driver.connect()` ran: `connect()` ends by clearing the driver's
    stopped flag, so the freeze the watcher sent is redone on the now powered arm and the run is refused.
    Both `--real` CLIs call this right after `connect()`."""
    if not (getattr(watcher, "pressed", False) or cancelled()):
        return False
    freeze(driver)
    print("Esc pressed while connecting: arm frozen with torque on, nothing moved.")
    return True


if __name__ == "__main__":
    sys.exit(main())
