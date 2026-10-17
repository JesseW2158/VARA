"""Simulation video of the chosen policy tracking a drifting target (task D6).

    python -m rl.render_video --policy rl/policies/handoff_best.zip --out web/static/rl_sim.mp4 \
        --seconds 10 [--fps 30] [--seed 0] [--drift 0.03]

Frames come from `HandoffEnv(render_mode="rgb_array")`, an offscreen `mujoco.Renderer` of the side camera
with the target in green and the tip in red, one environment step per frame (30 Hz, so `--fps 30` plays in
real time), written with `cv2.VideoWriter` (mp4v, 640x480) under a text overlay (episode, step, distance,
successes). Every episode's target drifts at `--drift` m/s in a random direction, bouncing off the target
box (the env's own 30 % drift probability is too low for a demo clip). Exit codes: 0 written, 2 bad
arguments or a missing policy, 3 the renderer or the video writer could not initialize (no OpenGL/GLFW,
no mp4v codec): the reason is printed and the dashboard shows its fallback text. `render_video(...)` is the
function behind the CLI and raises `RendererUnavailable` instead of exiting; the tests never need a working
renderer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from rl.handoff_env import HandoffEnv
from rl.policy_runner import load_policy

__all__ = ["RendererUnavailable", "render_video"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "rl" / "policies" / "handoff_best.zip"
DEFAULT_OUT = ROOT / "web" / "static" / "rl_sim.mp4"
DEFAULT_DRIFT = 0.03  # m/s
FONT = cv2.FONT_HERSHEY_SIMPLEX


class RendererUnavailable(RuntimeError):
    """The offscreen renderer or the video writer could not start (exit code 3 in the CLI)."""


def _drift(env: HandoffEnv, rng: np.random.Generator, speed: float) -> None:
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    env.set_target(env.target(), velocity=direction * speed)


def _annotate(frame_rgb: np.ndarray, episode: int, info: dict[str, Any], successes: int) -> np.ndarray:
    frame = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    text = (
        f"episode {episode}  step {info.get('step', 0):3d}  dist {100 * float(info.get('dist', 0.0)):4.1f} cm"
    )
    text += f"  successes {successes}"
    cv2.putText(
        frame, "SO-101 hand-off policy (simulation)", (12, 26), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(frame, text, (12, 52), FONT, 0.55, (80, 255, 120), 1, cv2.LINE_AA)
    return frame


def render_video(
    policy_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    *,
    seconds: float = 10.0,
    fps: int = 30,
    seed: int = 0,
    drift: float = DEFAULT_DRIFT,
    policy: Any = None,
) -> dict[str, Any]:
    """Write `seconds` of the policy chasing a drifting target to `out_path`; returns what was written.

    Raises ValueError for bad numbers, FileNotFoundError for a missing policy and RendererUnavailable
    when the offscreen renderer or the mp4v writer cannot start.
    """
    if not (float(seconds) > 0) or int(fps) < 1:
        raise ValueError("seconds must be positive and fps at least 1")
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = policy if policy is not None else load_policy(policy_path)
    env = HandoffEnv(render_mode="rgb_array")
    writer: Any = None
    total = max(1, int(round(float(seconds) * int(fps))))
    frames = episodes = successes = 0
    width = height = 0
    rng = np.random.default_rng(int(seed))
    try:
        obs, info = env.reset(seed=int(seed))
        _drift(env, rng, float(drift))
        try:
            frame = env.render()
        except Exception as exc:  # GLFW/EGL/OSMesa failed: no OpenGL context on this machine
            raise RendererUnavailable(
                f"the offscreen MuJoCo renderer could not initialize: {type(exc).__name__}: {exc}"
            ) from exc
        if frame is None:
            raise RendererUnavailable("HandoffEnv.render() returned no frame")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RendererUnavailable(f"cv2.VideoWriter could not open {out} with the mp4v codec")
        writer.write(_annotate(frame, episodes + 1, info, successes))
        frames = 1
        while frames < total:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            writer.write(_annotate(env.render(), episodes + 1, info, successes))
            frames += 1
            if terminated or truncated:
                episodes += 1
                successes += int(info["success"])
                obs, info = env.reset(seed=int(seed) + episodes)
                _drift(env, rng, float(drift))
    finally:
        if writer is not None:
            writer.release()
        env.close()
    return {
        "out": str(out),
        "frames": frames,
        "fps": int(fps),
        "seconds": frames / int(fps),
        "episodes": episodes,
        "successes": successes,
        "width": width,
        "height": height,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.render_video", description="Record the hand-off policy chasing a drifting target"
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="policy.zip from rl/train.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="MP4 to write (mp4v)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30, help="frames per second; 30 plays in real time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", type=float, default=DEFAULT_DRIFT, help="target speed in m/s")
    args = parser.parse_args(argv)
    if not pathlib.Path(args.policy).is_file():
        print(f"error: no such policy file: {args.policy}", file=sys.stderr)
        return 2
    try:
        result = render_video(
            args.policy, args.out, seconds=args.seconds, fps=args.fps, seed=args.seed, drift=args.drift
        )
    except RendererUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
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
