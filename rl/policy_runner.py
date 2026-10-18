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

Once the loop has taken a driver reading, the run's last event (success, timeout, stopped, aborted or
error) is preceded by one `policy stats: {json}` event (`policy_stats` parses it from run_policy's or
carry_to_hand's events): per reading, taken before that tick's command, the grasp-center (`arm.ik.fk_grasp`)
and tip distances to the target (min, final, longest run within SUCCESS_RADIUS), and the tip speed between
consecutive readings against the cap that applied (0.1 m/s with the safety scene's seen hand within 0.15 m
of either tip, else 0.2 m/s; `arm.replay`'s constants). Computing it never changes the motion.

D7 staged bring-up and the M4 recorder, `python -m rl.policy_runner` (`main`), in this order:

1. `--stage fixed --target-marker ID`: the target is 5 cm above that marker, on the top face of the scene
   object (seen) or obstacle carrying it (`marker_target`); the tracked hand is replaced by that point, so
   keep hands out of the box. The gripper may be empty (`--object` only sets the carry height).
2. `--stage slow`: 5 cm above the tracked wristband, moved slowly (`--hand-speed-max`, default 0.02 m/s).
3. `--stage normal`: the same at up to 5 cm/s (default 0.05 m/s). M4 counts 5 real trials of this stage.
   Only after M4 passes does a human set `use_policy: true` in config/rl.yaml; nothing here edits it.

    python -m rl.policy_runner --fake --stage fixed --target-marker 12          # FakeDriver, fake tracker
    python -m rl.policy_runner --real --stage fixed --target-marker 12 --scene-file runs/live/scene.json
    python -m rl.policy_runner --real --stage slow --tracker
    python -m rl.policy_runner --real --stage normal --tracker --trials 5

`run_trials` runs each trial through the live primitives only: `arm.real_robot.home` (hand-checked), then
`arm.real_robot.carry_to_hand(--object, policy={"use_policy": True, "policy": --policy})`, which rises,
travels, runs this policy, finishes with IK tracking, releases (it opens the jaws; the CLI never closes
them) and goes home. The hand-speed limit only flags a trial (`HandSpeedMeter`, over a 0.5 s window);
it never ends one. A trial record holds trial, stage, ok, reason, abort, stopped, policy_ran,
policy_result, policy_success, target, target_marker, target_item, readings, policy_s, grasp_min_m,
grasp_final_m, grasp_streak_max, within_3cm (the grasp center within 3 cm for 10 consecutive readings),
tip_min_m, tip_final_m, tip_streak_max, tip_within_3cm (the policy's own success test), tip_speed_max_mps,
tip_speed_cap_ratio_max, over_cap_intervals, over_cap, elapsed_s, hand_speed_max_mps,
hand_speed_limit_mps, hand_too_fast, hand_observations, stats and events. `m4_summary` gives k of n within
3 cm and the verdict: pass only when every trial ran, k >= ceil(0.8 n) (4 of 5) and no trial was over a
cap or unverified. The report, a new `runs/rl/m4-<UTC>.json` (VARA_RUNS_DIR honoured) or `--out FILE`, is
created exclusively, never overwritten, and rewritten after every trial. Exit 0 on pass, 1 on fail or
abort, 2 on misuse or a refusal.

`--real` follows arm.real_robot's CLI with its public helpers (`confirm_arm_clear`, `esc_during_connect`,
`freeze`): .env is loaded, then a missing scene source, a fake tracker (`arm.replay.tracker_scene`), a
`--scene-file` under FAKE/FAKE_CAMERA or config/fake_scene.json, an unloadable policy, an absent marker,
the held object as the marker or an object with no carry height is refused (exit 2) before the
confirmation prompt and before a Driver exists. Then the typed confirmation, the 3 s countdown with the
Esc watcher (freeze) and only then `connect()`; trials after the first wait for Enter. Whenever Esc fired
it prints `Esc -> freeze command: N ms` (a connect-phase press is labeled a diagnostic) and appends it with
`arm.driver.record_esc_latency(source="rl.policy_runner", phase=esc_phase(...))`. The driver disconnects
with torque on. Scene files are read only through `arm.replay.open_scene_file`, the step arm.replay's and
arm.real_robot's CLIs share: under --real a file without the live provenance sidecar that
`python -m vision.scene --out` writes, or without a fresh scene, is refused (exit 2) before any prompt.
`--fake` (or FAKE_ROBOT/FAKE without --real) uses a FakeDriver on a virtual clock, holding the object, and
the FakeSceneTracker (config/fake_scene.json, hand seen) unless --scene-file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
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
    "DEFAULT_POLICY",
    "HAND_OFFSET_Z",
    "HAND_SPEED_DEFAULTS",
    "M4_TRIALS",
    "STAGES",
    "STATS_PREFIX",
    "TARGET_INSET_M",
    "HandSpeedMeter",
    "ceiling_z",
    "clear_policy_cache",
    "handoff_zone",
    "load_policy",
    "m4_summary",
    "main",
    "marker_target",
    "policy_stats",
    "run_policy",
    "run_trials",
    "target_point",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "config" / "scene.yaml"
HAND_OFFSET_Z = 0.05  # m above the wristband
CEILING_Z = 0.30  # m, the default when config/scene.yaml has no workspace ceiling
TARGET_INSET_M = 0.03  # m, targets are clamped this far inside the workspace box (as arm/real_robot.py does)
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

STATS_PREFIX = "policy stats: "  # run_policy's statistics event (`policy_stats` parses it)
_OUTCOMES = (
    "policy success",
    "policy timeout",
    "policy stopped",
    "policy aborted",
    "policy error",
    "policy unavailable",
)
STAGES = ("fixed", "slow", "normal")  # the D7 bring-up order
HAND_SPEED_DEFAULTS = {"slow": 0.02, "normal": 0.05}  # m/s, the stage's hand-speed flag (--hand-speed-max)
DEFAULT_POLICY = "rl/policies/handoff_best.zip"
DEFAULT_OBJECT = "pill_bottle"
DEFAULT_TIMEOUT_S = 8.0
M4_TRIALS = 5  # M4: 4 of 5 real tries within 3 cm, never over the speed caps
M4_CONSECUTIVE = SUCCESS_STEPS  # M4's "within 3 cm": this many consecutive readings (run_policy's count)
FAKE_HELD_GRIPPER = 0.2  # --fake: the FakeDriver starts with its jaws on the object, as arm.real_robot's fake

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
        from arm.replay import (
            HAND_SLOW_M,
            TIP_SPEED_MAX,
            TIP_SPEED_NEAR_HAND,
            Abort,
            Pacer,
            Session,
            inside_box,
            load_box,
            shrunk_box,
        )

        caps = (float(TIP_SPEED_MAX), float(TIP_SPEED_NEAR_HAND), float(HAND_SLOW_M))
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
    trace = _Trace(start, caps)
    emit(f"policy start: {name}")

    def expired() -> bool:
        return now() - start >= seconds - 1e-9 or steps >= max_steps

    def finish(text: str) -> None:
        """The run's last event, preceded by its `policy stats:` line once a reading was taken."""
        line = trace.event()
        if line is not None:
            emit(line)
        emit(text)

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
                finish(f"policy timeout after {steps} steps")
                return False
            read_at = now()
            q_read, gripper_read = driver.read()
            q = _finite_reading(q_read, gripper_read)
            observed_scene = scene() if scene is not None else None
            if observed_scene is not None and not isinstance(observed_scene, dict):
                raise ValueError("the policy scene source must return a dict or None")
            scene_data = observed_scene or {}
            # Validate tracking data even if the proposed target or command would be held still.
            hand = session.hand()
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
            trace.observe(read_at, q, tip, target, hand)
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
                finish(f"policy timeout after {steps} steps")
                return False
            last_sent = now()
            session.command(q_next, where="policy")
            prev_action = (np.asarray(session.q[:4]) - q_cmd[:4]) / MAX_STEP_RAD
            steps += 1
            dist = float(np.linalg.norm(tip - target))
            streak = streak + 1 if dist < SUCCESS_RADIUS else 0
            if streak >= SUCCESS_STEPS:
                session.check_stopped()
                finish(f"policy success after {steps} steps")
                return True
    except Abort as exc:
        finish("policy stopped" if exc.reason == "stopped" else f"policy aborted: {exc.reason}")
        return False
    except Exception as exc:
        driver.freeze()
        finish(f"policy error: {type(exc).__name__}: {exc}; frozen")
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


# ------------------------------------------------------------------------------------------------------------
# per-run statistics: the `policy stats:` event
# ------------------------------------------------------------------------------------------------------------


class _Trace:
    """Statistics of one run_policy call from its driver readings (one per policy tick, taken before that
    tick's command): grasp-center and tip distances to the tick's target, the longest runs of consecutive
    readings within SUCCESS_RADIUS, and the tip speed between consecutive readings against the cap that
    applied (`caps` = (TIP_SPEED_MAX, TIP_SPEED_NEAR_HAND, HAND_SLOW_M): the near-hand cap when the safety
    scene's seen hand is within HAND_SLOW_M of either tip, as `arm.replay.Session.tip_cap`). It never
    raises into the control loop: a failure is logged and reported as `failure` in the stats."""

    def __init__(self, start: float, caps: tuple[float, float, float]) -> None:
        self.start = float(start)
        self.speed_max, self.speed_near, self.slow_m = caps
        self.readings = 0
        self.t_last: float | None = None
        self.tip_last: np.ndarray | None = None
        self.target: list[float] | None = None
        self.grasp_min = math.inf
        self.tip_min = math.inf
        self.grasp_final: float | None = None
        self.tip_final: float | None = None
        self.grasp_streak = self.grasp_streak_max = 0
        self.tip_streak = self.tip_streak_max = 0
        self.intervals = 0
        self.near_hand = 0
        self.over_cap = 0
        self.speed_peak = 0.0
        self.ratio_peak = 0.0
        self.failure: str | None = None

    def observe(self, t: float, q: Any, tip: Any, target: Any, hand: Any) -> None:
        """Record one reading; never raises."""
        if self.failure is not None:
            return
        try:
            self._observe(float(t), q, np.asarray(tip, dtype=np.float64).reshape(3), target, hand)
        except Exception as exc:
            log.exception("policy stats failed; the motion is unaffected")
            self.failure = f"{type(exc).__name__}: {exc}"

    def _observe(self, t: float, q: Any, tip: np.ndarray, target: Any, hand: Any) -> None:
        goal = np.asarray(target, dtype=np.float64).reshape(3)
        grasp = np.asarray(ik.fk_grasp(q)[0], dtype=np.float64)
        d_grasp = float(np.linalg.norm(grasp - goal))
        d_tip = float(np.linalg.norm(tip - goal))
        if not (math.isfinite(d_grasp) and math.isfinite(d_tip) and math.isfinite(t)):
            raise ValueError("non-finite policy statistics")
        self.readings += 1
        self.grasp_final, self.tip_final = d_grasp, d_tip
        self.grasp_min = min(self.grasp_min, d_grasp)
        self.tip_min = min(self.tip_min, d_tip)
        self.grasp_streak = self.grasp_streak + 1 if d_grasp < SUCCESS_RADIUS else 0
        self.tip_streak = self.tip_streak + 1 if d_tip < SUCCESS_RADIUS else 0
        self.grasp_streak_max = max(self.grasp_streak_max, self.grasp_streak)
        self.tip_streak_max = max(self.tip_streak_max, self.tip_streak)
        if self.tip_last is not None and self.t_last is not None and t > self.t_last:
            speed = float(np.linalg.norm(tip - self.tip_last)) / (t - self.t_last)
            near = hand is not None and (
                math.dist(self.tip_last, hand) < self.slow_m or math.dist(tip, hand) < self.slow_m
            )
            cap = self.speed_near if near else self.speed_max
            self.intervals += 1
            self.near_hand += int(near)
            self.speed_peak = max(self.speed_peak, speed)
            self.ratio_peak = max(self.ratio_peak, speed / cap)
            if speed > cap * (1.0 + 1e-9):
                self.over_cap += 1
        self.tip_last, self.t_last = tip, t
        self.target = [float(v) for v in goal]

    def summary(self) -> dict[str, Any]:
        """The statistics as a JSON-ready dict (meters, seconds, m/s); the keys are `policy_stats`'s."""

        def rounded(value: float | None) -> float | None:
            return None if value is None or not math.isfinite(value) else round(float(value), 6)

        return {
            "readings": self.readings,
            "elapsed_s": rounded(None if self.t_last is None else self.t_last - self.start),
            "target": None if self.target is None else [rounded(v) for v in self.target],
            "radius_m": SUCCESS_RADIUS,
            "grasp_min_m": rounded(self.grasp_min),
            "grasp_final_m": rounded(self.grasp_final),
            "grasp_streak_max": self.grasp_streak_max,
            "tip_min_m": rounded(self.tip_min),
            "tip_final_m": rounded(self.tip_final),
            "tip_streak_max": self.tip_streak_max,
            "caps_mps": [self.speed_max, self.speed_near],
            "intervals": self.intervals,
            "near_hand_intervals": self.near_hand,
            "tip_speed_max_mps": rounded(self.speed_peak),
            "tip_speed_cap_ratio_max": rounded(self.ratio_peak),
            "over_cap_intervals": self.over_cap,
            "failure": self.failure,
        }

    def event(self) -> str | None:
        """`policy stats: {json}` once a reading (or a failure) was recorded, else None; never raises."""
        if self.readings == 0 and self.failure is None:
            return None
        try:
            return STATS_PREFIX + json.dumps(self.summary(), sort_keys=True)
        except Exception:
            log.exception("policy stats could not be serialized")
            return None


def policy_stats(events: Iterable[str]) -> dict[str, Any] | None:
    """The last `policy stats:` event in `events` (run_policy's, also among carry_to_hand's events) as a
    dict, or None when the policy took no reading. Keys: readings, elapsed_s (policy clock from its start
    to the last reading), target, radius_m, grasp_min_m, grasp_final_m, grasp_streak_max (the longest run
    of consecutive readings with the grasp center within radius_m), tip_min_m, tip_final_m,
    tip_streak_max, caps_mps ([TIP_SPEED_MAX, TIP_SPEED_NEAR_HAND]), intervals, near_hand_intervals,
    tip_speed_max_mps, tip_speed_cap_ratio_max, over_cap_intervals and failure (None, or why the
    statistics stopped). ValueError for a stats event that is not a JSON object."""
    for text in reversed(list(events)):
        if isinstance(text, str) and text.startswith(STATS_PREFIX):
            data = json.loads(text[len(STATS_PREFIX) :])
            if not isinstance(data, dict):
                raise ValueError(f"malformed policy stats event: {text!r}")
            return data
    return None


# ------------------------------------------------------------------------------------------------------------
# D7 staged bring-up and the M4 trial recorder
# ------------------------------------------------------------------------------------------------------------


def marker_target(scene: Any, marker_id: int) -> tuple[str, tuple[float, float, float]] | None:
    """`(name, (x, y, h))` of the scene item carrying `marker_id`: a seen object, or an obstacle (obstacles
    keep their last pose). The marker lies on the item's top face, so `h` is its height. None when no such
    item is in the scene; ValueError for a malformed entry or a marker id on more than one item."""
    hits: list[tuple[str, tuple[float, float, float]]] = []
    for group in ("objects", "obstacles"):
        items = scene.get(group) if isinstance(scene, dict) else None
        if not isinstance(items, dict):
            continue
        for name, item in items.items():
            marker = item.get("marker") if isinstance(item, dict) else None
            if isinstance(marker, bool) or marker != marker_id:
                continue
            if group == "objects" and not item.get("seen"):
                continue
            try:
                xyz = (float(item["x"]), float(item["y"]), float(item["h"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"marker {marker_id}: malformed scene entry {name!r}") from exc
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError(f"marker {marker_id}: non-finite position of {name!r}: {xyz}")
            hits.append((str(name), xyz))
    if len(hits) > 1:
        raise ValueError(f"marker {marker_id} is on more than one item: {[name for name, _ in hits]}")
    return hits[0] if hits else None


class HandSpeedMeter:
    """The tracked wristband's speed over a window: each observation (the first scene read at a new clock
    reading with the hand seen) is compared with the newest earlier one at least `window_s` older, and the
    peak is kept in `peak` (m/s). A window rather than consecutive frames, so millimeter marker jitter at
    30 fps does not read as a fast hand; an unseen hand restarts the window."""

    def __init__(self, clock: Callable[[], float], window_s: float = 0.5) -> None:
        self.clock = clock
        self.window = float(window_s)
        self.peak = 0.0
        self.observations = 0
        self._obs: deque[tuple[float, tuple[float, float, float]]] = deque()

    def observe(self, scene: Any) -> None:
        """Record the hand of one scene read (a malformed hand is left to the Session's checks)."""
        hand = scene.get("hand") if isinstance(scene, dict) else None
        if not isinstance(hand, dict) or not hand.get("seen"):
            self._obs.clear()
            return
        try:
            xyz = (float(hand["x"]), float(hand["y"]), float(hand["z"]))
        except (KeyError, TypeError, ValueError):
            return
        t = float(self.clock())
        if not all(math.isfinite(v) for v in (*xyz, t)) or (self._obs and t <= self._obs[-1][0]):
            return
        self.observations += 1
        while len(self._obs) >= 2 and self._obs[1][0] <= t - self.window:
            self._obs.popleft()
        if self._obs and self._obs[0][0] <= t - self.window:
            t0, xyz0 = self._obs[0]
            self.peak = max(self.peak, math.dist(xyz, xyz0) / (t - t0))
        self._obs.append((t, xyz))


def _fixed_scene(
    source: Callable[[], dict | None], marker_id: int, point: tuple[float, float, float]
) -> Callable[[], dict | None]:
    """The source's scene with its hand replaced by the fixed marker point (seen), so carry_to_hand and the
    policy aim HAND_OFFSET_Z above the marker and the near-hand speed cap applies around it."""
    hand = {"marker": int(marker_id), "x": point[0], "y": point[1], "z": point[2], "seen": True}

    def scene() -> dict | None:
        data = source()
        return {**data, "hand": dict(hand)} if isinstance(data, dict) else data

    return scene


def _metered(source: Callable[[], dict | None], meter: HandSpeedMeter) -> Callable[[], dict | None]:
    def scene() -> dict | None:
        data = source()
        meter.observe(data)
        return data

    return scene


def _policy_outcome(events: Sequence[str]) -> str | None:
    """run_policy's terminal event among a carry's events, or None when the policy never ran."""
    for text in reversed(events):
        if isinstance(text, str) and text.startswith(_OUTCOMES):
            return text
    return None


def _abort_text(result: dict[str, Any]) -> str | None:
    if result.get("ok"):
        return None
    for text in reversed(result.get("events") or []):
        if isinstance(text, str) and text.startswith(("error", "stopped", "driver is stopped")):
            return text
    return None


def _trial_record(
    k: int,
    stage: str,
    result: dict[str, Any],
    *,
    elapsed: float,
    marker: tuple[int, str] | None,
    meter: HandSpeedMeter | None,
    hand_speed_max: float | None,
    stopped: bool,
) -> dict[str, Any]:
    events = list(result.get("events") or [])
    stats = policy_stats(events)
    outcome = _policy_outcome(events)
    measured = stats is not None and stats.get("failure") is None and int(stats.get("readings") or 0) > 0
    streak = int(stats["grasp_streak_max"]) if measured and stats else 0
    over = int(stats["over_cap_intervals"]) if measured and stats else None
    hand_peak = None if meter is None else round(meter.peak, 6)

    def stat(key: str) -> Any:
        return stats.get(key) if stats else None

    return {
        "trial": k,
        "stage": stage,
        "ok": bool(result.get("ok")),
        "reason": result.get("reason"),
        "abort": _abort_text(result),
        "stopped": stopped,
        "policy_ran": stats is not None,
        "policy_result": outcome,
        "policy_success": bool(outcome and outcome.startswith("policy success")),
        "target": stat("target"),
        "target_marker": None if marker is None else marker[0],
        "target_item": None if marker is None else marker[1],
        "readings": int(stat("readings") or 0),
        "policy_s": stat("elapsed_s"),
        "grasp_min_m": stat("grasp_min_m"),
        "grasp_final_m": stat("grasp_final_m"),
        "grasp_streak_max": streak,
        "within_3cm": measured and streak >= M4_CONSECUTIVE,
        "tip_min_m": stat("tip_min_m"),
        "tip_final_m": stat("tip_final_m"),
        "tip_streak_max": int(stat("tip_streak_max") or 0),
        "tip_within_3cm": measured and int(stat("tip_streak_max") or 0) >= M4_CONSECUTIVE,
        "tip_speed_max_mps": stat("tip_speed_max_mps") if measured else None,
        "tip_speed_cap_ratio_max": stat("tip_speed_cap_ratio_max") if measured else None,
        "over_cap_intervals": over,
        "over_cap": None if over is None else over > 0,
        "elapsed_s": round(float(elapsed), 3),
        "hand_speed_max_mps": hand_peak,
        "hand_speed_limit_mps": hand_speed_max,
        "hand_too_fast": bool(
            hand_peak is not None and hand_speed_max is not None and hand_peak > hand_speed_max
        ),
        "hand_observations": None if meter is None else meter.observations,
        "stats": stats,
        "events": events,
    }


def run_trials(
    driver: Any,
    scene: Callable[[], dict | None],
    *,
    stage: str,
    policy_path: str | pathlib.Path,
    obj: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    realtime: bool,
    trials: int = M4_TRIALS,
    timeout: float = DEFAULT_TIMEOUT_S,
    target_marker: int | None = None,
    hand_speed_max: float | None = None,
    before_trial: Callable[[int], bool] | None = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Up to `trials` policy reaches on a connected driver through the live primitives only: per trial
    `arm.real_robot.home` (hand-checked against `scene`'s real hand), then `arm.real_robot.carry_to_hand(obj,
    ..., policy={"use_policy": True, "policy": policy_path})`, both on the shared `clock`/`sleep`.

    Stage `fixed` aims HAND_OFFSET_Z above the item carrying `target_marker` (`marker_target`, looked up
    at the start of each trial; the scene's hand is replaced by that point); `slow` and `normal` follow the
    tracked hand and flag a trial whose observed hand speed (`HandSpeedMeter`) exceeded `hand_speed_max`.
    `before_trial(k)` runs first and returning False ends the run; `on_trial(record)` gets each record as
    it finishes. The run ends early after a failed home, a carry that did not end ok, a STOP or an unseen
    marker. Returns the trial records (keys in the module docstring)."""
    from arm import real_robot  # lazy: arm/ is not part of the RL training sandbox bundle

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "fixed" and target_marker is None:
        raise ValueError("stage fixed needs target_marker")
    override = {"use_policy": True, "policy": str(policy_path)}
    records: list[dict[str, Any]] = []
    for k in range(1, int(trials) + 1):
        if before_trial is not None and not before_trial(k):
            break
        marker: tuple[int, str] | None = None
        meter: HandSpeedMeter | None = None
        served: Callable[[], dict | None]
        hit = None
        missing = ""
        if stage == "fixed":
            try:
                hit = marker_target(scene(), target_marker)
            except ValueError as exc:
                missing = str(exc)
            if hit is None:
                missing = missing or f"marker {target_marker} is not on a seen object or obstacle"
        start = clock()
        if missing:
            result: dict[str, Any] = {
                "ok": False,
                "reason": "error",
                "events": [f"error: {missing}; nothing moved"],
            }
        else:
            if stage == "fixed":
                assert target_marker is not None and hit is not None
                marker = (int(target_marker), hit[0])
                served = _fixed_scene(scene, target_marker, hit[1])
            else:
                meter = HandSpeedMeter(clock)
                served = _metered(scene, meter)
            homed = real_robot.home(driver=driver, scene=scene, clock=clock, sleep=sleep, realtime=realtime)
            if homed["ok"]:
                carried = real_robot.carry_to_hand(
                    obj,
                    driver=driver,
                    scene=served,
                    timeout=timeout,
                    clock=clock,
                    sleep=sleep,
                    realtime=realtime,
                    policy=override,
                )
                result = {**carried, "events": [*homed["events"], *carried["events"]]}
            else:
                result = {**homed, "events": [*homed["events"], f"home before trial {k} failed"]}
        stopped = bool(driver.stopped) or result.get("reason") == "stopped"
        record = _trial_record(
            k,
            stage,
            result,
            elapsed=clock() - start,
            marker=marker,
            meter=meter,
            hand_speed_max=hand_speed_max,
            stopped=stopped,
        )
        records.append(record)
        if on_trial is not None:
            on_trial(record)
        if not result["ok"] or stopped:
            break
    return records


def m4_summary(records: Sequence[dict[str, Any]], requested: int) -> dict[str, Any]:
    """The M4 arithmetic over trial records: `within_3cm` = k trials whose grasp center stayed within 3 cm
    for ten consecutive readings, `required` = ceil(0.8 * requested) (4 of 5), `over_cap_trials` (a reading
    interval over its tip-speed cap), `unverified_speed_trials` (the policy ran but its statistics failed),
    and `verdict` "pass" only when every requested trial ran, k >= required and no trial was over the caps
    or unverified; else "fail"."""
    n = int(requested)
    if n < 1:
        raise ValueError(f"requested must be at least 1, got {requested!r}")
    required = math.ceil(0.8 * n - 1e-9)
    within = sum(1 for t in records if t.get("within_3cm"))
    over = sum(1 for t in records if t.get("over_cap") is True)
    unverified = sum(1 for t in records if t.get("policy_ran") and t.get("over_cap") is None)
    passed = len(records) == n and within >= required and over == 0 and unverified == 0
    return {
        "trials_requested": n,
        "trials_run": len(records),
        "within_3cm": within,
        "required": required,
        "over_cap_trials": over,
        "unverified_speed_trials": unverified,
        "tip_within_3cm": sum(1 for t in records if t.get("tip_within_3cm")),
        "hand_too_fast_trials": sum(1 for t in records if t.get("hand_too_fast")),
        "stopped": any(t.get("stopped") for t in records),
        "verdict": "pass" if passed else "fail",
    }


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------


def _env_fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl.policy_runner",
        description="D7 staged bring-up of the hand-off policy and the M4 trial recorder: every trial runs "
        "arm.real_robot.carry_to_hand with the policy switched on for that call. Nothing touches hardware "
        "without --real, nothing is gripped, and config/rl.yaml is never changed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--real", action="store_true", help="the real arm: typed confirmation, 3 s countdown, Esc freezes"
    )
    mode.add_argument("--fake", action="store_true", help="the in-memory FakeDriver on a virtual clock")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        required=True,
        help="fixed: 5 cm above --target-marker (the hand ignored); slow, normal: 5 cm above the tracked "
        "hand",
    )
    parser.add_argument(
        "--target-marker", type=int, metavar="ID", help="stage fixed: marker id of a scene object or obstacle"
    )
    parser.add_argument(
        "--hand-speed-max",
        type=float,
        metavar="M_PER_S",
        help="stages slow/normal: flag trials whose observed hand moved faster "
        f"(default {HAND_SPEED_DEFAULTS['slow']} slow, {HAND_SPEED_DEFAULTS['normal']} normal)",
    )
    parser.add_argument("--trials", type=int, default=M4_TRIALS, help=f"trials (default {M4_TRIALS})")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"carry_to_hand and policy timeout in seconds (default {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help=f"policy zip (default {DEFAULT_POLICY}; a relative path counts from here, else the repo root)",
    )
    parser.add_argument(
        "--object",
        default=DEFAULT_OBJECT,
        help=f"the object assumed held, which sets the carry height (default {DEFAULT_OBJECT}); "
        "the CLI never grips it",
    )
    hand = parser.add_mutually_exclusive_group()
    hand.add_argument(
        "--tracker",
        action="store_true",
        help="scene from the live overhead SceneTracker (refused with --real when it is a fake or shows no "
        "frame in 5 s); with --fake the FakeSceneTracker (also the default with --fake)",
    )
    hand.add_argument(
        "--scene-file", metavar="FILE", help="JSON scene file re-read every tick (arm.replay's reader)"
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="report path (default runs/rl/m4-<UTC>.json, under VARA_RUNS_DIR when set); never overwritten",
    )
    parser.add_argument("--port", help="serial port for --real; default FOLLOWER_PORT from .env")
    parser.add_argument("--id", dest="follower_id", help="LeRobot id for --real; default from config")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every carry event to stderr")
    return parser


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Argument combinations argparse cannot express; parser.error exits 2."""
    if args.stage == "fixed":
        if args.target_marker is None:
            parser.error("--stage fixed needs --target-marker ID")
        if args.hand_speed_max is not None:
            parser.error("--hand-speed-max applies to --stage slow and normal (fixed ignores the hand)")
    else:
        if args.target_marker is not None:
            parser.error("--target-marker applies to --stage fixed only")
        if args.hand_speed_max is None:
            args.hand_speed_max = HAND_SPEED_DEFAULTS[args.stage]
        elif not (math.isfinite(args.hand_speed_max) and args.hand_speed_max > 0):
            parser.error(f"--hand-speed-max must be a positive speed in m/s, got {args.hand_speed_max}")
    if args.trials < 1:
        parser.error(f"--trials must be at least 1, got {args.trials}")
    if not (math.isfinite(args.timeout) and 0 < args.timeout <= 60.0):
        parser.error(f"--timeout must be in (0, 60] seconds, got {args.timeout}")
    if not args.object.strip():
        parser.error("--object must name the held object")


def _resolve_policy(text: str) -> pathlib.Path | None:
    path = pathlib.Path(text)
    for candidate in [path] if path.is_absolute() else [pathlib.Path.cwd() / path, ROOT / path]:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _real_scene_refusal(args: argparse.Namespace) -> str | None:
    """Why --real must not run with this scene source (None when it may), checked before any prompt."""
    if not (args.tracker or args.scene_file):
        return (
            "--real needs a live scene: --tracker or --scene-file FILE (the fake scene never feeds the "
            "real arm)"
        )
    if args.scene_file:
        if _env_fake("FAKE_CAMERA"):
            return "--scene-file with --real while FAKE or FAKE_CAMERA is set: that scene may be synthetic"
        if pathlib.Path(args.scene_file).resolve() == FAKE_SCENE_PATH.resolve():
            return "--scene-file config/fake_scene.json with --real: a fake scene never feeds the real arm"
    return None


def _runs_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")


def _reserve(path: pathlib.Path | None) -> pathlib.Path:
    """Create the report file exclusively (never overwriting): `path`, else a new runs/rl/m4-<UTC>.json."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.open("x", encoding="utf-8").close()
        return path
    folder = _runs_dir() / "rl"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for n in range(1, 1000):
        candidate = folder / (f"m4-{stamp}.json" if n == 1 else f"m4-{stamp}-{n}.json")
        try:
            candidate.open("x", encoding="utf-8").close()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"no free m4-{stamp}*.json name in {folder}")


def _write(path: pathlib.Path, report: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _num(value: Any, unit: str, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f} {unit}"


def _trial_line(record: dict[str, Any], n: int) -> str:
    head = f"trial {record['trial']}/{n}: "
    if not record["policy_ran"]:
        return head + f"the policy did not run ({record['abort'] or record['reason']})"
    over = " OVER" if record["over_cap"] else ""
    ratio = record["tip_speed_cap_ratio_max"]
    parts = [
        f"within 3 cm for {M4_CONSECUTIVE} readings: {'yes' if record['within_3cm'] else 'no'}",
        f"grasp center min {_num(record['grasp_min_m'], 'm')}, final {_num(record['grasp_final_m'], 'm')}",
        f"tip final {_num(record['tip_final_m'], 'm')}",
        f"tip max {_num(record['tip_speed_max_mps'], 'm/s')} ({_num(ratio, 'of its cap', 2)}{over})",
        f"{record['elapsed_s']:.1f} s",
    ]
    if record["hand_speed_max_mps"] is not None:
        fast = " FASTER than the stage limit" if record["hand_too_fast"] else ""
        parts.append(f"hand max {_num(record['hand_speed_max_mps'], 'm/s')}{fast}")
    if not record["ok"]:
        parts.append(f"ended {record['reason']}: {record['abort']}")
    return head + "; ".join(parts)


def _esc_record(
    watcher: Any, drv: Any, connected_at: float | None, extra: dict[str, Any]
) -> dict[str, Any] | None:
    """Print and record the Esc-to-freeze latency (arm.driver's convention); None if it was not recorded."""
    from arm import driver as driver_mod

    ms = driver_mod.esc_latency_ms(watcher, drv)
    phase = driver_mod.esc_phase(watcher, connected_at)
    if ms is None:
        print("Esc -> freeze command: not measured (no freeze timestamp after the press)")
    elif phase == "session":
        print(f"Esc -> freeze command: {ms:.0f} ms")
    else:
        print(
            f"Esc -> freeze command: {ms:.0f} ms (connect phase: pressed before connect() returned, nothing "
            "moved; a diagnostic, not an A13 latency)"
        )
    try:
        written = driver_mod.record_esc_latency(
            source="rl.policy_runner",
            latency_ms=ms,
            extra={**extra, "frozen": bool(drv.stopped)},
            phase=phase,
        )
    except (OSError, ValueError) as exc:
        print(f"could not record the Esc latency: {exc}", file=sys.stderr)
        return None
    print(f"Esc latency recorded in {written} (phase {phase})")
    return {"latency_ms": ms, "phase": phase, "log": str(written)}


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m rl.policy_runner` (module docstring): 0 when the M4 verdict passes, 1 when it fails or
    the run was aborted, 2 for misuse or a refusal (nothing moved)."""
    parser = _parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from arm import driver as driver_mod  # lazy: arm/ is not part of the RL training sandbox bundle
    from arm import real_robot, replay

    fake_mode = args.fake or (not args.real and _env_fake("FAKE_ROBOT"))
    if not fake_mode and not args.real:
        print("Refusing to touch the arm without --real (use --fake for the in-memory arm).", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out) if args.out else None
    if out is not None and out.exists():
        print(f"--out {out} exists; reports are never overwritten. Nothing moved.", file=sys.stderr)
        return 2
    policy_path = _resolve_policy(args.policy)
    if policy_path is None:
        print(f"no policy file at {args.policy}; nothing moved", file=sys.stderr)
        return 2
    try:
        load_policy(policy_path)  # before anything connects: a corrupt zip is refused here
    except Exception as exc:
        print(
            f"cannot load --policy {policy_path}: {type(exc).__name__}: {exc}; nothing moved", file=sys.stderr
        )
        return 2
    if not fake_mode:
        from dotenv import load_dotenv

        load_dotenv()  # before the scene source: FAKE/FAKE_CAMERA from .env makes it a fake, refused below
        refusal = _real_scene_refusal(args)
        if refusal is not None:
            print(f"refused: {refusal}; nothing moved", file=sys.stderr)
            return 2
    tracker: Any = None
    if args.scene_file:
        # The shared step of arm.replay and arm.real_robot: under --real the file needs the live provenance
        # sidecar vision.scene --out writes and a fresh scene, or it is refused before anything connects.
        scene, refused = replay.open_scene_file(args.scene_file, real=not fake_mode)
        if refused:
            return 2
        assert scene is not None
        hand_source = f"scene-file {args.scene_file}"
    else:
        try:
            tracker, scene = replay.tracker_scene(fake_mode, hand_seen=True if fake_mode else None)
        except Exception as exc:  # a fake or silent tracker under --real, or one that cannot be built
            print(f"cannot use the tracker: {type(exc).__name__}: {exc}; nothing moved", file=sys.stderr)
            return 2
        hand_source = "fake tracker" if fake_mode else "tracker"
    try:
        problem = _scene_problem(args, scene, real_robot, replay)
        if problem is not None:
            print(f"refused: {problem}; nothing moved", file=sys.stderr)
            return 2
        report: dict[str, Any] = {
            "kind": "rl-m4-trials",
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": "fake" if fake_mode else "real",
            "stage": args.stage,
            "policy": str(policy_path),
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "object": args.object,
            "target_marker": args.target_marker,
            "hand_source": hand_source,
            "hand_speed_max_mps": args.hand_speed_max,
            "timeout_s": args.timeout,
            "radius_m": SUCCESS_RADIUS,
            "consecutive": M4_CONSECUTIVE,
            "caps_mps": [replay.TIP_SPEED_MAX, replay.TIP_SPEED_NEAR_HAND],
            "config_use_policy": real_robot.policy_config()["use_policy"],
            "counts_for_m4": not fake_mode and args.stage == "normal" and args.trials >= M4_TRIALS,
            "summary": m4_summary([], args.trials),
            "esc": None,
            "trials": [],
        }
        if fake_mode:
            clock = _VirtualClock()
            drv = driver_mod.FakeDriver(gripper0=FAKE_HELD_GRIPPER)
            path = _start_report(out, report)
            if path is None:
                return 2
            _run_and_record(args, drv, scene, report, path, clock.now, clock.sleep, False, None)
            return _finish(report, path)
        return _run_real(args, scene, report, out, driver_mod, real_robot)
    finally:
        if tracker is not None:
            replay.close_tracker(tracker)


def _scene_problem(
    args: argparse.Namespace, scene: Callable[[], Any], real_robot: Any, replay: Any
) -> str | None:
    """Pre-flight on the first scene read (before any prompt): a dict, the fixed marker on a seen item other
    than the held object, and a carry height for --object (`arm.real_robot.carry_height`)."""
    first = scene()
    if not isinstance(first, dict):
        return f"the scene source returned {type(first).__name__}, not a scene dict"
    if args.stage == "fixed":
        try:
            hit = marker_target(first, args.target_marker)
        except ValueError as exc:
            return str(exc)
        if hit is None:
            return f"marker {args.target_marker} is not on a seen object or obstacle in the scene"
        if hit[0] == args.object:
            return f"marker {args.target_marker} is on {hit[0]}, the object assumed held"
    height, how = real_robot.carry_height(args.object, first, ceiling=float(replay.load_box()["z"][1]))
    if height is None:
        return f"no carry height for --object {args.object} ({how})"
    return None


def _start_report(out: pathlib.Path | None, report: dict[str, Any]) -> pathlib.Path | None:
    try:
        path = _reserve(out)
    except FileExistsError:
        print(f"--out {out} exists; reports are never overwritten. Nothing moved.", file=sys.stderr)
        return None
    _write(path, report)
    print(f"report: {path}")
    return path


def _run_and_record(
    args: argparse.Namespace,
    drv: Any,
    scene: Callable[[], dict | None],
    report: dict[str, Any],
    path: pathlib.Path,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    realtime: bool,
    before_trial: Callable[[int], bool] | None,
) -> None:
    def on_trial(record: dict[str, Any]) -> None:
        report["trials"].append(record)
        report["summary"] = m4_summary(report["trials"], args.trials)
        _write(path, report)
        print(_trial_line(record, args.trials), flush=True)

    run_trials(
        drv,
        scene,
        stage=args.stage,
        policy_path=report["policy"],
        obj=args.object,
        trials=args.trials,
        timeout=args.timeout,
        target_marker=args.target_marker,
        hand_speed_max=args.hand_speed_max if args.stage != "fixed" else None,
        clock=clock,
        sleep=sleep,
        realtime=realtime,
        before_trial=before_trial,
        on_trial=on_trial,
    )
    report["summary"] = m4_summary(report["trials"], args.trials)
    _write(path, report)


def _finish(report: dict[str, Any], path: pathlib.Path) -> int:
    s = report["summary"]
    verdict = "PASS" if s["verdict"] == "pass" else "FAIL"
    print(
        f"M4: {s['within_3cm']} of {s['trials_requested']} within 3 cm "
        f"(grasp center, needs {s['required']}), "
        f"{s['over_cap_trials']} over the tip-speed caps, {s['trials_run']} run -> {verdict}"
    )
    print(f"(the jaw tip, the policy's own success test: {s['tip_within_3cm']} of {s['trials_requested']})")
    if report["mode"] == "fake":
        print("(fake rehearsal on the FakeDriver: it does not count for M4)")
    elif not report["counts_for_m4"]:
        print(f"(stage {report['stage']}: a bring-up stage; M4 counts 5 real trials at --stage normal)")
    if s["hand_too_fast_trials"]:
        print(f"{s['hand_too_fast_trials']} trial(s) saw the hand move faster than the stage limit")
    print(f"report: {path}")
    return 0 if s["verdict"] == "pass" else 1


def _run_real(
    args: argparse.Namespace,
    scene: Callable[[], dict | None],
    report: dict[str, Any],
    out: pathlib.Path | None,
    driver_mod: Any,
    real_robot: Any,
) -> int:
    """arm.real_robot's --real gating (confirmation, countdown, Esc watcher, the Esc-during-connect latch)
    around the trials; the driver disconnects with torque on."""
    real_robot.clear_cancel()  # discard an earlier session's cancellation before watching this startup
    drv = driver_mod.Driver(port=args.port, follower_id=args.follower_id)
    watcher = driver_mod.EscWatcher(lambda: real_robot.freeze(drv))
    watcher.start()
    connected_at: float | None = None
    esc_done = False
    extra = {"mode": "real", "stage": args.stage}
    try:
        if not real_robot.confirm_arm_clear(3, watcher):
            return 1
        if real_robot.esc_during_connect(watcher, drv):
            return 1
        drv.connect()  # the only place this CLI powers the arm
        connected_at = time.monotonic()  # Esc presses before this are connect-phase (arm.driver.esc_phase)
        try:
            if real_robot.esc_during_connect(watcher, drv):
                return 1
            path = _start_report(out, report)
            if path is None:
                return 2

            def before_trial(k: int) -> bool:
                if k > 1:
                    print(
                        f"Trial {k}/{args.trials}: the arm is home. Reset the {args.object}, the wristband "
                        "or the marker, keep clear; Enter starts, Esc aborts."
                    )
                    try:
                        input()
                    except EOFError:
                        return False
                return not (watcher.pressed or drv.stopped or real_robot.cancelled())

            _run_and_record(args, drv, scene, report, path, time.perf_counter, time.sleep, True, before_trial)
            if watcher.pressed:
                extra["trials_run"] = len(report["trials"])
                report["esc"] = _esc_record(watcher, drv, connected_at, extra)
                esc_done = True
                _write(path, report)
            rc = _finish(report, path)
            if drv.stopped:
                print("Arm is FROZEN with torque on. Press Enter to release the port (torque stays on).")
                try:
                    input()
                except EOFError:
                    pass
            return rc
        finally:
            drv.disconnect()  # torque stays on; the arm keeps holding its pose
    finally:
        if watcher.pressed and not esc_done:
            _esc_record(watcher, drv, connected_at, extra)
        watcher.stop()


if __name__ == "__main__":
    sys.exit(main())
