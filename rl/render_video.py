"""Simulation video of the chosen policy tracking a drifting target (task D6).

    python -m rl.render_video --policy rl/policies/handoff_best.zip --out web/static/rl_sim.mp4 \
        --seconds 10 [--fps 30] [--seed 0] [--drift 0.03]

Frames come from `HandoffEnv(render_mode="rgb_array")`, an offscreen `mujoco.Renderer` of the side camera
with the target in green and the tip in red, one environment step per frame (30 Hz, so `--fps 30` plays in
real time), encoded with PyAV (H.264/yuv420p, 640x480) under a text overlay (episode, step, distance,
successes). Every episode's target drifts at `--drift` m/s in a random direction, bouncing off the target
box (the env's own 30 % drift probability is too low for a demo clip). Exit codes: 0 written, 2 bad
arguments or a missing policy, 3 the renderer or the video writer could not initialize (no OpenGL/GLFW,
no H.264 encoder): the reason is printed and the dashboard shows its fallback text. `render_video(...)` is the
function behind the CLI and raises `RendererUnavailable` instead of exiting; the tests never need a working
renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
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


class _VideoWriter:
    """Encode browser-compatible MP4; publish only after the stream is flushed."""

    def __init__(self, path: pathlib.Path, fps: int, width: int, height: int) -> None:
        try:
            import av
        except ImportError as exc:
            raise RendererUnavailable("H.264 encoding requires av from requirements-rl.txt") from exc
        self.av = av
        self.closed = False
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "20", "preset": "fast"}
        except Exception:
            self.container.close()
            raise

    def write(self, frame: np.ndarray) -> None:
        for packet in self.stream.encode(self.av.VideoFrame.from_ndarray(frame, format="bgr24")):
            self.container.mux(packet)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


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
    when the offscreen renderer or the H.264 writer cannot start. Existing output is preserved on failure.
    """
    if not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(float(fps)) or int(fps) != fps or fps < 1:
        raise ValueError("fps must be a positive integer")
    if not math.isfinite(float(drift)) or float(drift) < 0:
        raise ValueError("drift must be finite and non-negative")
    total = max(1, int(round(float(seconds) * int(fps))))
    rng = np.random.default_rng(int(seed))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = policy if policy is not None else load_policy(policy_path)
    env = HandoffEnv(render_mode="rgb_array")
    writer: _VideoWriter | None = None
    frames = episodes = successes = 0
    width = height = 0
    staging = tempfile.TemporaryDirectory(prefix=".render-", dir=out.parent)
    pending = pathlib.Path(staging.name) / "clip.mp4"
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
        try:
            writer = _VideoWriter(pending, int(fps), width, height)
        except Exception as exc:
            raise RendererUnavailable(f"H.264 video encoder could not initialize: {exc}") from exc
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
        writer.close()
        writer = None
        pending.replace(out)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                env.close()
            finally:
                staging.cleanup()
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
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="MP4 to write (H.264/yuv420p)")
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
