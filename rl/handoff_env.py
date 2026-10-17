"""HandoffEnv: the SO-101 learns to bring its gripper tip to a (possibly moving) hand (task D1).

A Gymnasium environment on sim/model/arm_only.xml: the arm alone on the floor, no objects. The target is a
point 5 cm above the hand, sampled in TARGET_BOX (our table frame, meters); in DRIFT_PROB of the episodes it
drifts at up to DRIFT_SPEED[1] m/s and bounces off the walls of the box, so the policy learns to track.

Observation (16 float32, see OBS_LAYOUT and `build_observation`):
    0-4   measured joint angles (rad): true angle + a per-episode offset (+-OFFSET_MAX) + ANGLE_NOISE per step
    5-8   previous action: the one applied in this step, zeros right after reset
    9-11  tip position (m): `arm.ik.fk` of the MEASURED angles, + POS_NOISE
    12-14 observed target (m): the true target as it was `obs_lag` seconds ago (per episode), + POS_NOISE
    15    distance (m) from the observed tip to the observed target
The tip is always forward kinematics of joint readings (`arm.ik.fk`), never MuJoCo's site position, so the
real arm (rl/policy_runner.py) builds the identical vector from the driver's readings.

Action: 4 float32 in [-1, 1] for shoulder_pan, shoulder_lift, elbow_flex and wrist_flex, scaled to
+-MAX_STEP_RAD (2 degrees) per step and integrated into a position command that stays MARGIN_RAD (3 degrees)
inside the limits of config/limits.yaml (intersected with the model's ranges by `arm.ik.limits`); wrist_roll
and the gripper hold their start values. The command reaches the position actuators after the episode's
action delay (0 to DELAY_MAX steps) and the actuators' kp is scaled by a factor in GAIN_RANGE per episode.
The command lives in the measured frame, as on the real arm where the servo tracks its own encoder: the
actuator target is command - offset, so the measured joints follow the command while the true tip sits a
little off the tip the policy computes.

One step is 1/30 s = SUBSTEPS MuJoCo steps of SIM_DT. An episode is EPISODE_STEPS steps (truncated). It
terminates with success once the TRUE tip has stayed within SUCCESS_RADIUS of the TRUE target for
SUCCESS_STEPS consecutive steps, or with `info["fell"]` as soon as the true tip drops below FALL_Z. `info`
carries `success`, `dist` (true tip to true target), `fell`, `step`, `streak`, `target`, `tip` and `params`
(this episode's randomization) every step, so every reward variant is judged on the same metric. The built-in
reward is the baseline formula -d - 0.01 |a - a_prev|^2 + 1[d < SUCCESS_RADIUS] plus FALL_PENALTY on a fall;
rl/train.py replaces the formula with a reward file's `reward(obs, action, prev_action, info)` and keeps the
fall penalty, which belongs to the environment (see FALL_PENALTY).

Test hooks: `set_target(xyz, velocity=None, observed=None)`, `set_joints(q)`, `joint_angles()`, `command()`,
`applied_command()`, `actuator_targets()`, `target()` and the `params` attribute. `render_mode="rgb_array"`
draws the side camera with the target (green) and the tip (red) through a lazily created `mujoco.Renderer`.

    python -m rl.handoff_env --check
"""

from __future__ import annotations

import argparse
import math
import pathlib
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from arm import ik

__all__ = [
    "ACTION_SIZE",
    "DT",
    "EPISODE_STEPS",
    "FALL_Z",
    "MAX_STEP_RAD",
    "OBS_LAYOUT",
    "OBS_SIZE",
    "SUCCESS_RADIUS",
    "SUCCESS_STEPS",
    "TARGET_BOX",
    "EpisodeParams",
    "HandoffEnv",
    "baseline_reward",
    "build_observation",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ik.MODEL_PATH

JOINTS = ik.JOINTS  # the 5 arm joints, in the order of every joint vector
CONTROLLED = JOINTS[:4]  # the joints the policy moves; wrist_roll and the gripper stay fixed
GRIPPER_JOINT = "gripper"

HZ = 30
DT = 1.0 / HZ
SUBSTEPS = 20
SIM_DT = DT / SUBSTEPS  # 1/600 s
EPISODE_STEPS = 150  # 5 s
MAX_STEP_RAD = math.radians(2.0)  # an action of +-1 moves the command by this much per step
MARGIN_RAD = math.radians(3.0)  # commands stay this far inside the joint limits
SUCCESS_RADIUS = 0.03  # m, true tip to true target
SUCCESS_STEPS = 10  # consecutive steps within SUCCESS_RADIUS
FALL_Z = 0.03  # m, the episode ends when the true tip drops below this
# The env's own terminal cost of a fall, added on the falling step by the built-in reward and by
# rl.train.RewardWrapper on top of whatever the reward file returns. It is about the cost of hovering far from
# the target for a whole episode under the baseline reward (0.35 m x 150 steps), so ending the episode by
# diving into the table is never a shortcut: without it a 64k-step PPO run learned to fall in every episode.
FALL_PENALTY = -50.0

TARGET_BOX = ((0.20, 0.35), (-0.15, 0.15), (0.06, 0.12))  # (lo, hi) per axis, m
DRIFT_PROB = 0.3
DRIFT_SPEED = (0.01, 0.05)  # m/s when drifting
OFFSET_MAX = math.radians(2.0)  # per-episode joint offset
ANGLE_NOISE = math.radians(0.5)  # per-step measurement noise, std
POS_NOISE = 0.001  # m, per-step noise on the tip and the observed target, std
DELAY_MAX = 2  # steps of action delay
GAIN_RANGE = (0.8, 1.2)  # kp scale of the position actuators
LAG_RANGE = (0.1, 0.5)  # s, target observation lag
START_JITTER = math.radians(5.0)  # the start pose is home_q +- this on the controlled joints
GRIPPER_CTRL = 0.0  # rad, the fixed gripper angle (nearly closed, as when holding something small)

OBS_SIZE = 16
ACTION_SIZE = 4
OBS_LAYOUT: dict[str, slice] = {
    "joints": slice(0, 5),
    "prev_action": slice(5, 9),
    "tip": slice(9, 12),
    "target": slice(12, 15),
    "dist": slice(15, 16),
}

RENDER_WIDTH, RENDER_HEIGHT = 640, 480
RENDER_CAMERA = "side"
TARGET_RGBA = (0.1, 0.9, 0.2, 0.8)
TIP_RGBA = (0.95, 0.1, 0.1, 0.9)


@dataclass
class EpisodeParams:
    """What `reset()` randomized for the running episode (also `info["params"]`)."""

    joint_offset: list[float]  # rad, measured = true + offset, 5 joints
    action_delay: int  # steps between a command and the actuators seeing it
    gain_scale: float  # kp multiplier of the 4 controlled position actuators
    obs_lag_s: float  # s, the target observation lag
    obs_lag_steps: int  # the same in steps
    drifting: bool  # whether the target moves
    target_speed: float  # m/s, 0 when not drifting

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_observation(q_meas, prev_action, target_obs, tip=None) -> np.ndarray:
    """The 16-float observation from measured joints, the previous action and the observed target.

    `tip` defaults to `arm.ik.fk_tip(q_meas)`; the real-arm loop passes nothing and gets the same layout
    the policy was trained on. Returns float32 of shape (16,).
    """
    q = np.asarray(q_meas, dtype=np.float64).reshape(5)
    a = np.asarray(prev_action, dtype=np.float64).reshape(4)
    t = np.asarray(target_obs, dtype=np.float64).reshape(3)
    p = np.asarray(ik.fk_tip(q) if tip is None else tip, dtype=np.float64).reshape(3)
    d = float(np.linalg.norm(p - t))
    return np.concatenate([q, a, p, t, [d]]).astype(np.float32)


def baseline_reward(dist: float, action, prev_action, fell: bool = False) -> float:
    """The built-in reward: -d - 0.01 |a - a_prev|^2 + 1[d < SUCCESS_RADIUS] (+ FALL_PENALTY when fell)."""
    a = np.asarray(action, dtype=np.float64).reshape(4)
    p = np.asarray(prev_action, dtype=np.float64).reshape(4)
    r = -float(dist) - 0.01 * float(np.dot(a - p, a - p)) + (1.0 if dist < SUCCESS_RADIUS else 0.0)
    return r + (FALL_PENALTY if fell else 0.0)


def _vec3(value, what: str) -> np.ndarray:
    """`value` as a fresh float64 array of 3 finite numbers, or ValueError naming `what`."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be 3 finite numbers, got {value!r}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must be 3 finite numbers, got {value!r}")
    return arr.copy()


def _named(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise RuntimeError(f"{MODEL_PATH} has no {kind.name.lower()} named {name!r}")
    return idx


class HandoffEnv(gym.Env):
    """Bring the tip to a target 5 cm above a (moving) hand; see the module docstring."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": HZ}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        model_path: str | pathlib.Path | None = None,
        start_jitter: float = START_JITTER,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'rgb_array', not {render_mode!r}")
        self.render_mode = render_mode
        self._model = mujoco.MjModel.from_xml_path(str(model_path or MODEL_PATH))
        self._model.opt.timestep = SIM_DT
        self._data = mujoco.MjData(self._model)
        m = self._model
        joint_ids = [_named(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
        gripper_id = _named(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        self._qadr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self._dofadr = np.array([m.jnt_dofadr[j] for j in joint_ids])
        self._gripper_qadr = int(m.jnt_qposadr[gripper_id])
        self._gripper_dofadr = int(m.jnt_dofadr[gripper_id])
        self._act = np.array([_named(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINTS])
        self._gripper_act = _named(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)
        self._kp0 = m.actuator_gainprm[self._act, 0].copy()
        lim = ik.limits()
        self._lo = np.array([lim[name][0] for name in JOINTS]) + MARGIN_RAD
        self._hi = np.array([lim[name][1] for name in JOINTS]) - MARGIN_RAD
        self._home = np.array(ik.home_q(), dtype=np.float64)
        self._start_jitter = float(start_jitter)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)

        self.params: EpisodeParams | None = None
        self._offset = np.zeros(5)
        self._cmd = self._home.copy()  # the commanded position, measured frame
        self._pending: deque[np.ndarray] = deque(maxlen=1)  # commands on their way to the actuators
        self._target = np.zeros(3)
        self._velocity = np.zeros(3)
        self._history: deque[np.ndarray] = deque(maxlen=1)  # true targets, oldest first
        self._prev_action = np.zeros(ACTION_SIZE)
        self._steps = 0
        self._streak = 0
        self._ready = False
        self._renderer: mujoco.Renderer | None = None

    # ----------------------------------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        offset = rng.uniform(-OFFSET_MAX, OFFSET_MAX, size=5)
        delay = int(rng.integers(0, DELAY_MAX + 1))
        gain = float(rng.uniform(GAIN_RANGE[0], GAIN_RANGE[1]))
        lag_s = float(rng.uniform(LAG_RANGE[0], LAG_RANGE[1]))
        lag_steps = max(1, int(round(lag_s * HZ)))
        drifting = bool(rng.uniform() < DRIFT_PROB)
        speed = float(rng.uniform(DRIFT_SPEED[0], DRIFT_SPEED[1])) if drifting else 0.0
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        target = np.array([rng.uniform(lo, hi) for lo, hi in TARGET_BOX])
        jitter = rng.uniform(-self._start_jitter, self._start_jitter, size=4)
        self.params = EpisodeParams(
            joint_offset=[float(v) for v in offset],
            action_delay=delay,
            gain_scale=gain,
            obs_lag_s=lag_s,
            obs_lag_steps=lag_steps,
            drifting=drifting,
            target_speed=speed,
        )

        mujoco.mj_resetData(self._model, self._data)
        self._set_gain(gain)
        self._offset = offset
        start = self._home.copy()
        start[:4] += jitter
        self._place(start, delay)
        self._target = target
        self._velocity = direction * speed
        self._history = deque([target.copy() for _ in range(lag_steps + 1)], maxlen=lag_steps + 1)
        self._prev_action = np.zeros(ACTION_SIZE)
        self._steps = 0
        self._streak = 0
        self._ready = True

        q_true = self._true_q()
        tip_true = np.array(ik.fk_tip(q_true))
        dist = float(np.linalg.norm(tip_true - self._target))
        return self._observe(q_true), self._info(dist, False, False, tip_true)

    def step(self, action):
        self._require_reset("step()")
        a = np.asarray(action, dtype=np.float64).reshape(ACTION_SIZE)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action must be 4 finite numbers, got {action!r}")
        a = np.clip(a, -1.0, 1.0)
        self._cmd[:4] = np.clip(self._cmd[:4] + a * MAX_STEP_RAD, self._lo[:4], self._hi[:4])
        self._pending.append(self._cmd.copy())
        self._apply(self._pending[0])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self._model, self._data)
        self._advance_target()
        self._steps += 1

        q_true = self._true_q()
        tip_true = np.array(ik.fk_tip(q_true))
        dist = float(np.linalg.norm(tip_true - self._target))
        self._streak = self._streak + 1 if dist < SUCCESS_RADIUS else 0
        success = self._streak >= SUCCESS_STEPS
        fell = bool(tip_true[2] < FALL_Z)
        terminated = success or fell
        truncated = (not terminated) and self._steps >= EPISODE_STEPS
        reward = baseline_reward(dist, a, self._prev_action, fell)
        self._prev_action = a
        obs = self._observe(q_true)
        info = self._info(dist, success, fell, tip_true)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self._renderer.update_scene(self._data, camera=RENDER_CAMERA)
        scene = self._renderer.scene
        self._marker(scene, self._target, 0.015, TARGET_RGBA)
        self._marker(scene, ik.fk_tip(self._true_q()), 0.008, TIP_RGBA)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -------------------------------------------------------------------------------------- test hooks

    def set_target(self, xyz, velocity=None, *, observed=None) -> None:
        """Move the TRUE target to `xyz`; the drift velocity becomes `velocity` (m/s) or zero; the success
        streak restarts. The lag history is refilled with `observed` (default `xyz`), so the policy sees
        `observed` for the next `params.obs_lag_steps` steps and the true target after that. Every vector
        must be 3 finite numbers (ValueError otherwise, and nothing changes)."""
        target = _vec3(xyz, "target")
        velocity_arr = np.zeros(3) if velocity is None else _vec3(velocity, "velocity")
        seen = target if observed is None else _vec3(observed, "observed target")
        self._target = target
        self._velocity = velocity_arr
        depth = self._history.maxlen or 1
        self._history = deque([seen.copy() for _ in range(depth)], maxlen=depth)
        self._streak = 0

    def set_joints(self, q) -> None:
        """Teleport the arm: the TRUE joints become `q` (clipped so the command stays inside the limits minus
        3 degrees), velocities zero, the command and the delay queue hold that pose."""
        arr = np.asarray(q, dtype=np.float64).reshape(5)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"q must be 5 finite joint angles, got {q!r}")
        self._place(arr, (self._pending.maxlen or 1) - 1)

    def joint_angles(self) -> list[float]:
        """The TRUE joint angles (rad); the policy only ever sees the measured ones."""
        return [float(v) for v in self._true_q()]

    def command(self) -> list[float]:
        """The latest commanded position (5 joints, measured frame)."""
        return [float(v) for v in self._cmd]

    def applied_command(self) -> list[float]:
        """The command the actuators are tracking right now (measured frame): `command()` from
        `params.action_delay` steps ago. RuntimeError before the first `reset()`."""
        self._require_reset("applied_command()")
        return [float(v) for v in self._pending[0]]

    def actuator_targets(self) -> list[float]:
        """What the 5 position actuators are really tracking, read from the simulator's `ctrl` (true frame,
        i.e. `applied_command()` minus `params.joint_offset`). RuntimeError before the first `reset()`."""
        self._require_reset("actuator_targets()")
        return [float(v) for v in self._data.ctrl[self._act]]

    def target(self) -> list[float]:
        """The TRUE target (m)."""
        return [float(v) for v in self._target]

    # ----------------------------------------------------------------------------------------- internals

    def _require_reset(self, what: str) -> None:
        if not self._ready:
            raise RuntimeError(f"call reset() before {what}")

    def _set_gain(self, scale: float) -> None:
        kp = self._kp0[:4] * scale
        self._model.actuator_gainprm[self._act[:4], 0] = kp
        self._model.actuator_biasprm[self._act[:4], 1] = -kp

    def _place(self, q_true: np.ndarray, delay: int) -> None:
        """Put the arm at `q_true` at rest with the command holding it and the delay queue full of it."""
        cmd = np.clip(q_true + self._offset, self._lo, self._hi)
        q_true = cmd - self._offset
        self._cmd = cmd
        self._pending = deque([cmd.copy() for _ in range(delay + 1)], maxlen=delay + 1)
        d = self._data
        d.qpos[self._qadr] = q_true
        d.qvel[self._dofadr] = 0.0
        d.qpos[self._gripper_qadr] = GRIPPER_CTRL
        d.qvel[self._gripper_dofadr] = 0.0
        self._apply(cmd)
        mujoco.mj_forward(self._model, d)

    def _apply(self, cmd: np.ndarray) -> None:
        self._data.ctrl[self._act] = cmd - self._offset
        self._data.ctrl[self._gripper_act] = GRIPPER_CTRL

    def _true_q(self) -> np.ndarray:
        return self._data.qpos[self._qadr].copy()

    def _advance_target(self) -> None:
        if self._velocity.any():
            p = self._target + self._velocity * DT
            for axis, (lo, hi) in enumerate(TARGET_BOX):
                if p[axis] < lo:
                    p[axis] = min(hi, 2 * lo - p[axis])
                    self._velocity[axis] = abs(self._velocity[axis])
                elif p[axis] > hi:
                    p[axis] = max(lo, 2 * hi - p[axis])
                    self._velocity[axis] = -abs(self._velocity[axis])
            self._target = p
        self._history.append(self._target.copy())

    def _observe(self, q_true: np.ndarray) -> np.ndarray:
        rng = self.np_random
        q_meas = q_true + self._offset + rng.normal(0.0, ANGLE_NOISE, size=5)
        tip = np.array(ik.fk_tip(q_meas)) + rng.normal(0.0, POS_NOISE, size=3)
        target_obs = self._history[0] + rng.normal(0.0, POS_NOISE, size=3)
        return build_observation(q_meas, self._prev_action, target_obs, tip=tip)

    def _info(self, dist: float, success: bool, fell: bool, tip_true: np.ndarray) -> dict[str, Any]:
        return {
            "success": bool(success),
            "dist": float(dist),
            "fell": bool(fell),
            "step": int(self._steps),
            "streak": int(self._streak),
            "target": [float(v) for v in self._target],
            "tip": [float(v) for v in tip_true],
            "params": self.params.as_dict() if self.params is not None else {},
        }

    @staticmethod
    def _marker(scene: mujoco.MjvScene, pos, radius: float, rgba) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray(pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def _check(steps: int, seed: int) -> None:
    env = HandoffEnv()
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    episodes = successes = falls = 0
    for _ in range(steps):
        obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            episodes += 1
            successes += int(info["success"])
            falls += int(info["fell"])
            obs, info = env.reset()
    env.close()
    print(
        f"obs shape {obs.shape} ({obs.dtype}), {steps} random steps, {episodes} episodes ended, "
        f"successes {successes}, falls {falls}, last dist {info['dist']:.3f} m"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HandoffEnv: the SO-101 reaching a moving hand")
    parser.add_argument("--check", action="store_true", help="step a random policy and print the obs shape")
    parser.add_argument("--steps", type=int, default=300, help="random steps for --check (default 300)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.check:
        _check(args.steps, args.seed)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
