"""Train PPO on HandoffEnv with a reward function read from a candidate-style source file (tasks D2, D5).

    python -m rl.train --reward rl/rewards/baseline.py --steps 500000 --out rl/runs/baseline \
        [--n-envs 8] [--seed 0] [--eval-episodes 100]

The reward SOURCE is checked with `sim.candidate_check.check_source(src, entry="reward")` here, and loaded
with `sim.restricted.load_entry` inside each environment's own process (the factory carries the path, never
the function, and nothing ever imports the reward module). `RewardWrapper` replaces the env's built-in reward
with `reward(obs, action, prev_action, info)`; the success metric stays the env's `info["success"]`: the
reward sees a copy of `info`, so nothing it writes reaches Monitor, the callbacks or progress.csv.

Stable-Baselines3 PPO, MlpPolicy, SubprocVecEnv (spawn) for more than one env, DummyVecEnv for one. Writes to
`--out`: `checkpoints/` every CHECKPOINT_EVERY steps, `progress.csv` (step, mean_reward, success_rate over the
last WINDOW episodes) at least every PROGRESS_EVERY steps and at the end, `policy.zip`, and `summary.json`
with a `--eval-episodes` (default 100) evaluation on a fresh env (success rate, mean steps to success;
`--eval-episodes 0` skips it and writes the same keys as null). CPU only. A rollout is `n_steps = min(1024,
steps // n_envs)` per env, so a short smoke run (`--steps 1024 --n-envs 1`) does exactly one update.

Failures are one `error: ...` line on stderr and exit code 2: a rejected reward file, a reward that raises or
returns something that is not a finite number (under SubprocVecEnv the worker's traceback precedes the line),
a negative seed or episode count, an unwritable output directory. This CLI reads no environment variables and
does not load `.env`: the process (and the workers it spawns, which inherit its environment) executes
candidate-style reward code, which must never see an API key.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import numbers
import pathlib
import sys
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.evaluate import empty_result, run_episodes
from rl.handoff_env import ACTION_SIZE, FALL_PENALTY, HandoffEnv
from sim.candidate_check import check_source

__all__ = ["EnvFactory", "ProgressCallback", "RewardWrapper", "copy_info", "train"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINT_EVERY = 50_000  # steps between checkpoints (summed over the envs)
PROGRESS_EVERY = 20_000  # steps between progress.csv rows, at most
WINDOW = 20  # episodes averaged in a progress row
EVAL_EPISODES = 100
PROGRESS_COLUMNS = ("step", "mean_reward", "success_rate")
MAX_ROLLOUT = 1024  # n_steps per env
MIN_ROLLOUT = 64


def copy_info(info: dict[str, Any]) -> dict[str, Any]:
    """A copy of an env `info` dict one level deeper than `dict(info)`: lists and dicts inside it (the
    `target`, `tip` and `params` of HandoffEnv) are copied too, so the reward function cannot touch what
    Monitor and the callbacks read."""
    out: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            value = {k: (list(v) if isinstance(v, list) else v) for k, v in value.items()}
        elif isinstance(value, list):
            value = list(value)
        out[key] = value
    return out


class RewardWrapper(gym.Wrapper):
    """Replace the env's reward with `reward_fn(obs, action, prev_action, info)` from a reward file.

    The function sees plain Python lists and a COPY of the info dict (it may only use `math`): the
    `obs`, `info` and metrics the trainer works with are never the objects the reward touched. A reward that
    raises, or that returns anything but a finite real number (a numeric string, None, a list, NaN), raises
    ValueError, which ends the training run with a clear message rather than a silent NaN or a forged metric.
    The env's FALL_PENALTY is added on a step with `info["fell"]` whatever the file returns: a fall ends the
    episode, and under a reward that is negative per step an unpenalized early end is a shortcut the policy
    learns within a few tens of thousands of steps.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable[..., Any]) -> None:
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_action = [0.0] * ACTION_SIZE

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = [0.0] * ACTION_SIZE
        return obs, info

    def step(self, action):
        obs, _builtin, terminated, truncated, info = self.env.step(action)
        act = [float(v) for v in np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)]
        step = info.get("step")
        try:
            value = self._reward_fn(
                [float(v) for v in obs], list(act), list(self._prev_action), copy_info(info)
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: B036 - the reward is untrusted code: every failure is its fault
            raise ValueError(f"reward() raised {type(exc).__name__}: {exc} at step {step}") from exc
        if not isinstance(value, numbers.Real):  # a numeric string, None or a list is not a reward
            raise ValueError(f"reward() returned {value!r} at step {step}, not a number")
        reward = float(value)
        if not math.isfinite(reward):
            raise ValueError(f"reward() returned {reward!r} at step {step}")
        if info.get("fell"):
            reward += FALL_PENALTY
        self._prev_action = act
        return obs, reward, terminated, truncated, info


class EnvFactory:
    """A picklable env constructor: reads, checks and loads the reward source in the process that calls it."""

    def __init__(self, reward_path: str, seed: int = 0) -> None:
        self.reward_path = str(reward_path)
        self.seed = int(seed)

    def __call__(self) -> gym.Env:
        from sim.restricted import load_entry  # the one sanctioned loader of AI-written code

        src = pathlib.Path(self.reward_path).read_text(encoding="utf-8")
        reward_fn = load_entry(src, entry="reward", filename=self.reward_path)
        env = Monitor(RewardWrapper(HandoffEnv(), reward_fn))
        env.action_space.seed(self.seed)
        return env


class ProgressCallback(BaseCallback):
    """Append `step, mean_reward, success_rate` (over the last WINDOW episodes) to a CSV every `every` steps
    and once more when training ends."""

    def __init__(self, path: pathlib.Path, every: int = PROGRESS_EVERY, window: int = WINDOW) -> None:
        super().__init__(verbose=0)
        self.path = pathlib.Path(path)
        self.every = max(1, int(every))
        self.returns: deque[float] = deque(maxlen=window)
        self.successes: deque[bool] = deque(maxlen=window)
        self.rows: list[dict[str, float]] = []
        self._next = self.every

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(PROGRESS_COLUMNS)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for info, done in zip(infos, dones, strict=True):
            if done:
                self.successes.append(bool(info.get("success", False)))
                episode = info.get("episode")
                if episode is not None:
                    self.returns.append(float(episode["r"]))
        if self.num_timesteps >= self._next:
            self.write_row()
            self._next = (self.num_timesteps // self.every + 1) * self.every
        return True

    def _on_training_end(self) -> None:
        self.write_row()

    def write_row(self) -> dict[str, float]:
        row = {
            "step": int(self.num_timesteps),
            "mean_reward": float(np.mean(self.returns)) if self.returns else math.nan,
            "success_rate": float(np.mean(self.successes)) if self.successes else math.nan,
        }
        self.rows.append(row)
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([row[c] for c in PROGRESS_COLUMNS])
        if self.verbose:
            print(
                f"step {row['step']}: mean_reward {row['mean_reward']:.3f} "
                f"success_rate {row['success_rate']:.2f}",
                file=sys.stderr,
            )
        return row


def rollout_sizes(steps: int, n_envs: int) -> tuple[int, int]:
    """(n_steps per env, batch_size): one rollout never exceeds the requested steps by much and the batch
    divides the rollout."""
    n_steps = max(MIN_ROLLOUT, min(MAX_ROLLOUT, steps // max(1, n_envs)))
    total = n_steps * n_envs
    batch = next((b for b in (256, 128, 64, 32) if total % b == 0), total)
    return n_steps, batch


def make_vec_env(reward_path: str | pathlib.Path, n_envs: int, seed: int):
    """SubprocVecEnv (spawn, Windows-safe) for n_envs > 1, DummyVecEnv for 1."""
    factories = [EnvFactory(str(reward_path), seed + i) for i in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


WORKER_ERRORS = (EOFError, BrokenPipeError, ConnectionError)  # what the parent sees of a dead worker


def _close_vec_env(vec_env: DummyVecEnv | SubprocVecEnv) -> None:
    """Close the vec env; when a worker already died (its pipe is broken), terminate the rest instead of
    waiting on them, so the original error surfaces rather than a second traceback from the cleanup."""
    try:
        vec_env.close()
    except WORKER_ERRORS:
        for process in getattr(vec_env, "processes", ()):
            if process.is_alive():
                process.terminate()
        for process in getattr(vec_env, "processes", ()):
            process.join(timeout=5.0)


def train(
    reward_path: str | pathlib.Path,
    steps: int,
    out: str | pathlib.Path,
    *,
    n_envs: int = 8,
    seed: int = 0,
    eval_episodes: int = EVAL_EPISODES,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train PPO with the reward in `reward_path`, save `out/policy.zip` and return the written summary.

    Raises ValueError for a rejected reward file or a bad argument and RuntimeError when an environment
    worker dies (a reward that raises under SubprocVecEnv; the worker's own traceback goes to stderr).
    """
    reward_path = pathlib.Path(reward_path)
    out = pathlib.Path(out)
    if steps < 1 or n_envs < 1:
        raise ValueError("steps and n_envs must be at least 1")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if eval_episodes < 0:
        raise ValueError(f"eval_episodes must be at least 0, got {eval_episodes}")
    src = reward_path.read_text(encoding="utf-8")
    problems = check_source(src, entry="reward")
    if problems:
        raise ValueError(f"reward rejected: {'; '.join(problems)}")
    out.mkdir(parents=True, exist_ok=True)
    n_steps, batch_size = rollout_sizes(steps, n_envs)
    progress = ProgressCallback(out / "progress.csv")
    progress.verbose = verbose
    checkpoints = CheckpointCallback(
        save_freq=max(1, CHECKPOINT_EVERY // n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix="ppo_handoff",
    )
    started = time.time()
    try:
        vec_env = make_vec_env(reward_path, n_envs, seed)
    except WORKER_ERRORS as exc:
        raise RuntimeError("an environment worker died while starting (its traceback is above)") from exc
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            seed=seed,
            device="cpu",
            verbose=verbose,
        )
        model.learn(total_timesteps=steps, callback=[checkpoints, progress])
        policy_path = out / "policy.zip"
        model.save(str(policy_path))
    except WORKER_ERRORS as exc:
        raise RuntimeError(
            "an environment worker died during training, usually because the reward raised "
            "(its traceback is above)"
        ) from exc
    finally:
        _close_vec_env(vec_env)
    train_seconds = time.time() - started

    summary = run_episodes(model, eval_episodes, seed=seed + 10_000) if eval_episodes > 0 else empty_result()
    summary.update(
        {
            "policy": str(policy_path),
            "reward": str(reward_path),
            "steps": int(model.num_timesteps),
            "requested_steps": int(steps),
            "n_envs": int(n_envs),
            "seed": int(seed),
            "eval_seed": int(seed + 10_000),
            "train_seconds": round(train_seconds, 1),
            "final_progress": progress.rows[-1] if progress.rows else None,
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO on HandoffEnv with a reward file")
    parser.add_argument("--reward", required=True, help="reward source file (candidate-style, entry reward)")
    parser.add_argument("--steps", type=int, required=True, help="total environment steps")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not pathlib.Path(args.reward).is_file():
        print(f"error: no such reward file: {args.reward}", file=sys.stderr)
        return 2
    try:
        summary = train(
            args.reward,
            args.steps,
            args.out,
            n_envs=args.n_envs,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            verbose=1 if args.verbose else 0,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    # No load_dotenv() here on purpose: this process runs candidate-style reward code (see the module
    # docstring) and needs no key.
    multiprocessing.freeze_support()
    raise SystemExit(main())
