"""Evaluate a saved PPO policy on HandoffEnv (tasks D2, D-GO, D5): the success rate on `info["success"]`.

    python -m rl.evaluate --policy rl/runs/baseline/policy.zip --episodes 100 [--seed 0] [--stochastic]

Prints one JSON object with `success_rate`, `mean_steps_to_success` (null when no episode succeeded),
`episodes`, `successes`, `falls` and `mean_final_dist` (the keys of RESULT_KEYS), then exits 0. Episode k
uses `reset(seed=seed + k)`, so two policies evaluated with the same seed see the same targets and
randomizations (`seed` must be non-negative, as Gymnasium requires). Runs on the CPU; rl/train.py calls
`run_episodes` for the summary it writes after training, or `empty_result()` when the evaluation is skipped.
A bad argument or an unreadable policy prints `error: ...` on stderr and exits 2.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rl.handoff_env import HandoffEnv

__all__ = ["RESULT_KEYS", "empty_result", "evaluate", "load_policy", "run_episodes"]

RESULT_KEYS = ("episodes", "successes", "success_rate", "mean_steps_to_success", "falls", "mean_final_dist")


def empty_result() -> dict[str, Any]:
    """The record of an evaluation that ran no episode: every RESULT_KEYS key present, the rates null."""
    return {
        "episodes": 0,
        "successes": 0,
        "success_rate": None,
        "mean_steps_to_success": None,
        "falls": 0,
        "mean_final_dist": None,
    }


def load_policy(path: str | pathlib.Path) -> PPO:
    """Load a policy zip saved by rl/train.py onto the CPU."""
    return PPO.load(str(path), device="cpu")


def run_episodes(
    model: PPO,
    episodes: int,
    *,
    seed: int = 0,
    deterministic: bool = True,
    env: HandoffEnv | None = None,
) -> dict[str, Any]:
    """Run `episodes` episodes of `model` on `env` (a fresh HandoffEnv by default) and score them.

    Success is the env's `info["success"]`; `mean_steps_to_success` averages the step count of the
    successful episodes (None when there is none).
    """
    if episodes < 1:
        raise ValueError(f"episodes must be at least 1, got {episodes}")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    own_env = env is None
    env = HandoffEnv() if env is None else env
    successes = 0
    falls = 0
    steps_to_success: list[int] = []
    final_dists: list[float] = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            while True:
                action, _state = model.predict(obs, deterministic=deterministic)
                obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if info["success"]:
                successes += 1
                steps_to_success.append(int(info["step"]))
            falls += int(info["fell"])
            final_dists.append(float(info["dist"]))
    finally:
        if own_env:
            env.close()
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        "falls": falls,
        "mean_final_dist": float(np.mean(final_dists)),
    }


def evaluate(
    policy_path: str | pathlib.Path, episodes: int, *, seed: int = 0, deterministic: bool = True
) -> dict[str, Any]:
    """Load `policy_path` and run `run_episodes`; the result also names the policy."""
    result = run_episodes(load_policy(policy_path), episodes, seed=seed, deterministic=deterministic)
    result["policy"] = str(policy_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a hand-off policy on HandoffEnv")
    parser.add_argument("--policy", required=True, help="policy.zip written by rl/train.py")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="episode k resets with seed + k")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    args = parser.parse_args(argv)
    if not pathlib.Path(args.policy).is_file():
        print(f"error: no such policy file: {args.policy}", file=sys.stderr)
        return 2
    try:
        result = evaluate(args.policy, args.episodes, seed=args.seed, deterministic=not args.stochastic)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
