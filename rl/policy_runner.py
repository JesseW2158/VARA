"""Run the hand-off policy on the arm through the driver (task D7): the real-arm twin of HandoffEnv.step.

`run_policy(driver, scene, policy_path, *, timeout=8.0, clock=None, sleep=None, on_event=None,
policy=None) -> bool` is what `arm.real_robot.carry_to_hand` calls when config/rl.yaml says
`use_policy: true` (`run_policy(session.driver, session.scene_fn, str(policy_path), timeout=timeout)`):

- `driver` is `arm.driver.Driver` or `FakeDriver`: every target goes through `driver.command(q, gripper)`,
  which clips to the limits minus 3 degrees and to 2 degrees per call and refuses while stopped or with
  torque off (.claude/rules/rl.md: the driver enforces the limits). `scene` is a zero-argument callable
  returning the latest scene dict (`hand` with `seen`, `x`, `y`, `z`; optional `zones.handoff`) or None.
  `policy_path` is a PPO zip from rl/train.py, loaded once per path (`PPO.load(path, device="cpu")`,
  cached in this process; `clear_policy_cache()` drops it).
- The loop runs at 30 Hz (`rl.handoff_env.DT`): read the joints, take the target 5 cm above the hand when
  it is seen, else the handoff zone (the scene's `zones.handoff` or config/scene.yaml), build the 16-float
  observation with `rl.handoff_env.build_observation(q, prev_action, target)` (the tip from `arm.ik.fk` of
  the readings, never from a simulator), `policy.predict(obs, deterministic=True)` clipped to [-1, 1],
  scaled to +-2 degrees on joints 0-3 and integrated into a command that starts from the present pose and
  stays 3 degrees inside the limits, exactly like the env's command. A step whose tip would rise above the
  workspace ceiling (config/scene.yaml `workspace.z[1]`, 0.30 m) is held instead. The gripper value never
  changes: every command repeats the last commanded gripper (or the reading at the start), so a held
  object stays held.
- Returns True once the tip has been within 3 cm of the target for 10 consecutive steps, False on timeout,
  when `driver.stopped` is set (checked every step; a refused command counts) or when the policy cannot be
  loaded (the IK path takes over). `on_event` receives short strings: `policy start: <file>`,
  `policy success after N steps`, `policy timeout after N steps`, `policy stopped`,
  `policy unavailable: ...`, `policy held below the ceiling`.
- `clock` and `sleep` default to `time.monotonic`/`time.sleep` on a real driver; on a FakeDriver a virtual
  clock advances by what the loop sleeps and nothing waits, so a FakeDriver run takes no wall time and a
  timeout of 8 s is 240 steps. `policy` lets tests pass an object with `predict(obs, deterministic=True)
  -> (action, state)` instead of a file.
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
            on_event(text)

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
    now, pause = _timing(driver, clock, sleep)
    lim = ik.limits()
    lo = np.array([lim[joint][0] for joint in ik.JOINTS]) + MARGIN_RAD
    hi = np.array([lim[joint][1] for joint in ik.JOINTS]) - MARGIN_RAD
    ceiling = ceiling_z()
    max_steps = int(math.ceil(seconds / DT)) + 1  # a tick budget: a stalled clock cannot loop forever
    prev_action = np.zeros(ACTION_SIZE)
    q_cmd: np.ndarray | None = None
    gripper_hold = 0.0
    streak = 0
    steps = 0
    held_warned = False
    start = now()
    next_tick = start
    emit(f"policy start: {name}")
    while True:
        if getattr(driver, "stopped", False):
            emit("policy stopped")
            return False
        if now() - start >= seconds - 1e-9 or steps >= max_steps:  # 1e-9: summed ticks round either way
            emit(f"policy timeout after {steps} steps")
            return False
        q_read, gripper_read = driver.read()
        q = np.asarray(q_read, dtype=np.float64).reshape(len(ik.JOINTS))
        if q_cmd is None:
            q_cmd = q.copy()  # integrate from the present pose, as the env integrates from its start pose
            last = getattr(driver, "last_command", None)
            gripper_hold = float(last[1]) if last else float(gripper_read)
        target = np.array(target_point(scene() if scene is not None else None))
        tip = np.array(ik.fk_tip(q))
        obs = build_observation(q, prev_action, target, tip=tip)
        raw, _state = model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(raw, dtype=np.float64).reshape(ACTION_SIZE), -1.0, 1.0)
        q_next = q_cmd.copy()
        q_next[:4] = np.clip(q_cmd[:4] + action * MAX_STEP_RAD, lo[:4], hi[:4])
        if ik.fk_tip(q_next)[2] > ceiling:
            q_next = q_cmd.copy()
            action = np.zeros(ACTION_SIZE)
            if not held_warned:
                emit("policy held below the ceiling")
                held_warned = True
        if not driver.command([float(v) for v in q_next], gripper_hold):
            emit("policy stopped")
            return False
        last = getattr(driver, "last_command", None)
        q_cmd = np.asarray(last[0], dtype=np.float64) if last else q_next
        prev_action = action
        steps += 1
        dist = float(np.linalg.norm(tip - target))
        streak = streak + 1 if dist < SUCCESS_RADIUS else 0
        if streak >= SUCCESS_STEPS:
            emit(f"policy success after {steps} steps")
            return True
        next_tick += DT
        delay = next_tick - now()
        if delay > 0:
            pause(delay)
