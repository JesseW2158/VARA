"""Run the optional hand-off policy through the live primitive safety session (D7).

Every 30 Hz decision observes measured joints, preserves the wrist and gripper, and applies at most a
2 degree joint step. replay.Session supplies STOP, workspace and hand checks; a policy step is reduced
until its tip travel fits the same 0.2 m/s (0.1 m/s near a hand) cap used by IK. There is one command per
policy tick, so slowing a step never extends the policy deadline with a multi-tick waypoint.

Success requires ten readings within 3 cm of the target. A timeout or unavailable policy returns False
for the caller's IK fallback; malformed data or a failed motion freezes with torque on. Nothing here
opens the gripper. The policy is refused if its initial pose is outside the workspace or joint band.

The optional safety_scene retains the original tracked hand when carry_to_hand substitutes its safe
fallback target. cancelled, clock and sleep let that caller share its STOP flag and pacing. A FakeDriver
uses virtual time unless clocks are supplied. policy is an injected predict() object for offline tests.
"""

from __future__ import annotations

import logging
import math
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import yaml

from arm import ik
from rl.handoff_env import (
    ACTION_SIZE,
    DT,
    MARGIN_RAD,
    MAX_STEP_RAD,
    SUCCESS_RADIUS,
    SUCCESS_STEPS,
    build_observation,
)

__all__ = [
    "CEILING_Z",
    "HAND_OFFSET_Z",
    "TARGET_INSET_M",
    "ceiling_z",
    "clear_policy_cache",
    "handoff_zone",
    "load_policy",
    "run_policy",
    "target_point",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "config" / "scene.yaml"
HAND_OFFSET_Z = 0.05  # m above the wristband
CEILING_Z = 0.30  # m, the default when config/scene.yaml has no workspace ceiling
TARGET_INSET_M = 0.03  # m, targets are clamped this far inside the workspace box (as arm/real_robot.py does)

log = logging.getLogger(__name__)
_POLICIES: dict[str, Any] = {}


def load_policy(path: str | pathlib.Path) -> Any:
    """`PPO.load(path, device="cpu")`, once per path (FileNotFoundError for a missing file)."""
    key = str(pathlib.Path(path).resolve())
    model = _POLICIES.get(key)
    if model is None:
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"no policy file at {path}")
        from stable_baselines3 import (
            PPO,
        )  # lazy: torch takes seconds to import and the IK path needs none of it

        model = PPO.load(key, device="cpu")
        _POLICIES[key] = model
    return model


def clear_policy_cache() -> None:
    """Forget every loaded policy (a retrained handoff_best.zip is picked up at the next call)."""
    _POLICIES.clear()


def _scene_config() -> dict[str, Any]:
    cfg = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {}


def handoff_zone(scene: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """The handoff zone (x, y, z): the scene's `zones.handoff` when given, else config/scene.yaml."""
    zone = None
    if isinstance(scene, dict):
        zones = scene.get("zones")
        zone = zones.get("handoff") if isinstance(zones, dict) else None
    if not isinstance(zone, dict):
        zone = _scene_config()["zones"]["handoff"]
    xyz = (float(zone["x"]), float(zone["y"]), float(zone["z"]))
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError(f"non-finite handoff zone {xyz}")
    return xyz


def ceiling_z() -> float:
    """The workspace ceiling (config/scene.yaml `workspace.z[1]`), CEILING_Z when unreadable."""
    try:
        z = float(_scene_config()["workspace"]["z"][1])
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return CEILING_Z
    return z if math.isfinite(z) else CEILING_Z


def target_point(scene: dict[str, Any] | None) -> tuple[float, float, float]:
    """HAND_OFFSET_Z above a seen hand, else the handoff zone."""
    hand = scene.get("hand") if isinstance(scene, dict) else None
    if isinstance(hand, dict) and hand.get("seen"):
        try:
            xyz = (float(hand["x"]), float(hand["y"]), float(hand["z"]) + HAND_OFFSET_Z)
        except (KeyError, TypeError, ValueError):
            xyz = None
        if xyz is not None and all(math.isfinite(v) for v in xyz):
            return xyz
    return handoff_zone(scene)


def _clamp_into(point: Any, box: dict[str, list[float]]) -> np.ndarray:
    return np.array(
        [min(max(float(v), float(box[a][0])), float(box[a][1])) for a, v in zip("xyz", point, strict=True)]
    )


class _VirtualClock:
    """Time that advances only in `sleep`: a FakeDriver run takes no wall time."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


def _timing(
    driver: Any, clock: Callable[[], float] | None, sleep: Callable[[float], None] | None
) -> tuple[Callable[[], float], Callable[[float], None]]:
    if clock is not None and sleep is not None:
        return clock, sleep
    if getattr(driver, "is_fake", False):
        virtual = _VirtualClock()
        return clock or virtual.now, sleep or virtual.sleep
    return clock or time.monotonic, sleep or time.sleep


def _emitter(on_event: Callable[[str], None] | None) -> Callable[[str], None]:
    def emit(text: str) -> None:
        log.info("%s", text)
        if on_event is not None:
            try:
                on_event(text)
            except Exception:
                log.exception("policy event observer failed")

    return emit


def run_policy(
    driver: Any,
    scene: Callable[[], dict[str, Any] | None] | None,
    policy_path: str | pathlib.Path,
    *,
    timeout: float = 8.0,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    on_event: Callable[[str], None] | None = None,
    policy: Any = None,
    safety_scene: Callable[[], dict[str, Any] | None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Track the hand with the policy through the driver; True on success, False otherwise (module doc)."""
    emit = _emitter(on_event)
    seconds = float(timeout)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"timeout must be a finite, non-negative number of seconds, got {timeout!r}")
    name = pathlib.Path(str(policy_path)).name
    try:
        model = policy if policy is not None else load_policy(policy_path)
    except Exception as exc:  # a missing or corrupt zip: the IK path takes over, nothing moved yet
        emit(f"policy unavailable: {type(exc).__name__}: {exc}")
        log.warning("policy %s unavailable: %s", policy_path, exc)
        return False
    try:
        # Lazy: arm/replay.py is not part of the independent RL training sandbox bundle.
        from arm.replay import Abort, Pacer, Session, inside_box, load_box, shrunk_box

        box = load_box(SCENE_PATH)
        target_box = shrunk_box(box, TARGET_INSET_M)
        now, pause = _timing(driver, clock, sleep)
        scene_data: dict[str, Any] = {}
        session = Session(
            driver,
            safety_scene if safety_scene is not None else lambda: scene_data,
            Pacer(not getattr(driver, "is_fake", False), now, pause),
            emit,
            hand_freeze=False,
            cancelled=cancelled,
        )
        session.box = session.gate_box = box
        lim = session.limits
        lo = np.array([lim[joint][0] for joint in ik.JOINTS]) + MARGIN_RAD
        hi = np.array([lim[joint][1] for joint in ik.JOINTS]) - MARGIN_RAD
    except Exception as exc:  # bad configuration: nothing has touched the driver
        emit(f"policy unavailable: no workspace box ({type(exc).__name__}: {exc})")
        return False
    ceiling = float(box["z"][1])
    max_steps = int(math.ceil(seconds / DT)) + 1  # stalled clocks still have a finite tick budget
    prev_action = np.zeros(ACTION_SIZE)
    streak = 0
    steps = 0
    held_warned = False
    box_warned = False
    clamp_warned = False
    start = now()
    last_sent: float | None = None
    emit(f"policy start: {name}")

    def expired() -> bool:
        return now() - start >= seconds - 1e-9 or steps >= max_steps

    try:
        if driver.stopped:
            emit("policy stopped")
            return False
        if not driver.torque:
            emit("policy unavailable: torque is off")
            return False
        session.check_stopped()
        session.start()
        _finite_reading(session.q, session.g)
        if (
            not inside_box(session.tip, box)
            or np.any(np.array(session.q) < lo)
            or np.any(np.array(session.q) > hi)
        ):
            emit("policy unavailable: initial pose is outside the workspace or joint safety band")
            return False
        while True:
            session.check_stopped()
            if expired():
                emit(f"policy timeout after {steps} steps")
                return False
            q_read, gripper_read = driver.read()
            q = _finite_reading(q_read, gripper_read)
            observed_scene = scene() if scene is not None else None
            if observed_scene is not None and not isinstance(observed_scene, dict):
                raise ValueError("the policy scene source must return a dict or None")
            scene_data = observed_scene or {}
            # Validate tracking data even if the proposed target or command would be held still.
            session.hand()
            session.check_stopped()
            raw_target = np.array(target_point(observed_scene))
            target = _clamp_into(raw_target, target_box)
            if not clamp_warned and not np.allclose(target, raw_target, rtol=0.0, atol=1e-12):
                emit("policy target clamped inside the workspace box")
                clamp_warned = True
            tip = np.array(ik.fk_tip(q))
            if not np.all(np.isfinite(tip)):
                raise ValueError("non-finite measured tip")
            if not inside_box(tip, box):
                raise ValueError("measured tip left the workspace box")
            obs = build_observation(q, prev_action, target, tip=tip)
            raw, _state = model.predict(obs, deterministic=True)
            action = np.asarray(raw, dtype=np.float64).reshape(ACTION_SIZE)
            if not np.all(np.isfinite(action)):
                raise ValueError("non-finite policy action")
            action = np.clip(action, -1.0, 1.0)
            q_cmd = np.asarray(session.q)
            q_next = q_cmd.copy()
            q_next[:4] = np.clip(q_cmd[:4] + action * MAX_STEP_RAD, lo[:4], hi[:4])
            tip_next = ik.fk_tip(q_next)
            if tip_next[2] > ceiling:
                q_next = q_cmd.copy()
                if not held_warned:
                    emit("policy held below the ceiling")
                    held_warned = True
            elif not inside_box(tip_next, box):
                q_next = q_cmd.copy()
                if not box_warned:
                    emit("policy held inside the workspace box")
                    box_warned = True
            q_next = _speed_limited_step(session, q_next)
            # A late prediction may not issue an expired command or cause catch-up bursts. Session's
            # regular pacer handles ordinary ticks; this also spaces sends after an inference stall.
            if last_sent is not None:
                delay = last_sent + DT - now()
                if delay > 1e-9:
                    pause(delay)
            session.check_stopped()
            if expired():
                emit(f"policy timeout after {steps} steps")
                return False
            last_sent = now()
            session.command(q_next, where="policy")
            prev_action = (np.asarray(session.q[:4]) - q_cmd[:4]) / MAX_STEP_RAD
            steps += 1
            dist = float(np.linalg.norm(tip - target))
            streak = streak + 1 if dist < SUCCESS_RADIUS else 0
            if streak >= SUCCESS_STEPS:
                session.check_stopped()
                emit(f"policy success after {steps} steps")
                return True
    except Abort as exc:
        emit("policy stopped" if exc.reason == "stopped" else f"policy aborted: {exc.reason}")
        return False
    except Exception as exc:
        driver.freeze()
        emit(f"policy error: {type(exc).__name__}: {exc}; frozen")
        return False


def _finite_reading(joints: Any, gripper: Any) -> np.ndarray:
    q = np.asarray(joints, dtype=np.float64).reshape(len(ik.JOINTS))
    if not np.all(np.isfinite(q)) or not math.isfinite(float(gripper)):
        raise ValueError("non-finite driver reading")
    if not 0.0 <= float(gripper) <= 1.0:
        raise ValueError("driver gripper reading is outside [0, 1]")
    return q


def _speed_limited_step(session: Any, requested: np.ndarray) -> np.ndarray:
    """Fit one policy tick to Session's Cartesian cap, recomputing FK after every reduction."""
    current = np.asarray(session.q, dtype=np.float64)
    delta = requested - current
    fraction = 1.0
    for _ in range(12):
        candidate = current + fraction * delta
        tip = session.ik.fk_tip(candidate)
        step = math.dist(session.tip, tip)
        if not math.isfinite(step):
            raise ValueError("non-finite policy target tip")
        cap = session.tip_cap(tip)
        if step <= cap * (1.0 + 1e-9):
            return candidate
        fraction *= 0.99 * cap / step
    # Holding the current command is safe and bounded even with a pathological kinematic model.
    return current
