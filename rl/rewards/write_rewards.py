"""Reward-writing agent (task D4): one CODEGEN call returns four reward functions for HandoffEnv.

`write_rewards(*, out_dir=ROOT / "rl" / "rewards", task_dir=None, timeout=60, task_id="rewards")` builds
one prompt (the D1 environment description with the 16-float observation layout of
`rl.handoff_env.build_observation`, the action semantics, the `info` keys, the env-level FALL_PENALTY so a
reward need not penalize falls again, the baseline reward source with its D-GO result from
rl/runs/baseline/summary.json, the four shaping ideas of docs/tasks/D-rl.md D4 verbatim, and the
check_source trap list), calls `agents.llm.chat(..., purpose="write_rewards")` once, splits the reply with
`agents.candidates.extract_code_blocks`, checks every block with `check_source(src, entry="reward")`, asks
once more for each variant that was rejected or missing (with its problems; fake key `variant_<k>`), and
writes the accepted ones to `out_dir/variant_<k>.py` (k = 1..4). Returns the written paths in variant
order: fewer than four when a variant is rejected twice.

Files under `task_dir` (default `runs/rewards/<UTC timestamp>/`): `prompt.md`, `reply.md`,
`retry_<k>.md` (the retry request and its reply), `problems.json` (the problems of every rejected
attempt) and `variants.json` (what was written). `agents.llm` appends each call to `runs/<task_id>/llm.jsonl`.

Nothing here imports, compiles or executes a variant: `check_source` is a static check and the files stay
text until `rl/train_parallel.py` trains them inside Daytona sandboxes (CLAUDE.md hard rule 1). With
FAKE_LLM=1 (or FAKE=1) the reply holds the four fixtures of tests/fixtures/llm/rewards/.

    FAKE_LLM=1 python -m rl.rewards.write_rewards [--out DIR] [--task-dir DIR] [--timeout S]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from agents.candidates import extract_code, extract_code_blocks
from agents.llm import chat, codegen_model
from rl import handoff_env as he
from sim.candidate_check import check_source

__all__ = [
    "CLOSING_LINE",
    "N_VARIANTS",
    "SHAPING_IDEAS",
    "baseline_result",
    "build_prompt",
    "environment_description",
    "retry_prompt",
    "trap_list",
    "write_rewards",
]

ROOT = pathlib.Path(__file__).resolve().parents[2]
REWARDS_DIR = ROOT / "rl" / "rewards"
BASELINE_PATH = REWARDS_DIR / "baseline.py"
BASELINE_SUMMARY = ROOT / "rl" / "runs" / "baseline" / "summary.json"
RUNS_DIR = ROOT / "runs" / "rewards"
N_VARIANTS = 4
# The four shaping ideas, verbatim from docs/tasks/D-rl.md D4; variant k follows idea k.
SHAPING_IDEAS = (
    "exponential distance",
    "progress toward the target",
    "a smoothness penalty weighted by distance",
    "a success bonus with a time penalty",
)
CLOSING_LINE = "Return four Python code blocks, one per variant, nothing else."
# Every written variant opens with three comment lines: provenance, then the file-level noqa and fmt-off
# directives, so ruff skips the generated file (lint and format) without an exclusion in pyproject.toml.
HEADER_LINES = ("# ruff: noqa", "# fmt: off")
SYSTEM_PROMPT = (
    "You design reward functions for reinforcement learning on a robot arm. Reply with Python code blocks "
    "only: no prose before, between or after them."
)
FORBIDDEN_NAMES = (
    "open, exec, eval, compile, __import__, getattr, setattr, delattr, globals, locals, vars, input, "
    "breakpoint, help, dir, type, object, super, mro"
)

log = logging.getLogger(__name__)


def environment_description() -> str:
    """The D1 environment as the model must understand it, built from rl.handoff_env's constants."""
    lay = he.OBS_LAYOUT
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = he.TARGET_BOX
    joints = ", ".join(he.JOINTS)
    controlled = ", ".join(he.CONTROLLED)
    return f"""## The environment: rl/handoff_env.py, class HandoffEnv

A Gymnasium environment on a MuJoCo model of the SO-101 arm with no objects. Units are meters, radians and
seconds. One step is 1/{he.HZ} s; an episode is {he.EPISODE_STEPS} steps ({he.EPISODE_STEPS / he.HZ:g} s).

The target is a point 5 cm above a person's hand, sampled in x [{x_lo}, {x_hi}], y [{y_lo}, {y_hi}],
z [{z_lo}, {z_hi}] (table frame: origin under the shoulder, +x out from the arm, +y to the arm's left,
+z up). In {he.DRIFT_PROB:.0%} of the episodes it drifts at up to {he.DRIFT_SPEED[1]} m/s, so the policy
has to track it: there is no fixed goal pose to memorize.

`obs` is a list of {he.OBS_SIZE} floats:
- obs[{lay["joints"].start}:{lay["joints"].stop}]: the five measured joint angles in radians ({joints}), \
with a per-episode offset of up to {math.degrees(he.OFFSET_MAX):g} degrees and \
{math.degrees(he.ANGLE_NOISE):g} degrees of noise
- obs[{lay["prev_action"].start}:{lay["prev_action"].stop}]: the previous action: the \
{he.ACTION_SIZE} floats applied in the step before this one (zeros right after a reset)
- obs[{lay["tip"].start}:{lay["tip"].stop}]: the gripper tip position (x, y, z) in meters, from forward \
kinematics of the measured joints
- obs[{lay["target"].start}:{lay["target"].stop}]: the observed target (x, y, z): the true target as it was \
{he.LAG_RANGE[0]} to {he.LAG_RANGE[1]} s ago, plus {he.POS_NOISE * 1000:g} mm of noise
- obs[{lay["dist"].start}]: the distance in meters from the observed tip to the observed target

`action` and `prev_action` are lists of {he.ACTION_SIZE} floats in [-1, 1] for {controlled}; each unit
moves that joint by {math.degrees(he.MAX_STEP_RAD):g} degrees per step. wrist_roll and the gripper never
move. `prev_action` is the action of the previous step (zeros on the first step of an episode).

`info` is a dict with these keys on every step:
- info["dist"]: float, the TRUE distance in meters from the tip to the true target: the success metric \
(obs[{lay["dist"].start}] lags behind it)
- info["success"]: bool, True on the step the episode ends with success: the tip stayed within \
{he.SUCCESS_RADIUS} m of the target for {he.SUCCESS_STEPS} consecutive steps
- info["streak"]: int, consecutive steps so far within {he.SUCCESS_RADIUS} m
- info["fell"]: bool, True when the tip dropped below z = {he.FALL_Z} m; the episode ends at once
- info["step"]: int, the step number inside the episode (1..{he.EPISODE_STEPS})
- info["target"]: [x, y, z] of the true target; info["tip"]: [x, y, z] of the true tip
- info["params"]: this episode's randomization (joint_offset, action_delay, gain_scale, obs_lag_s, drifting, \
target_speed); not useful for shaping

Every variant is judged on info["success"], whatever its reward says. The environment adds a penalty of
{he.FALL_PENALTY:g} on a step with info["fell"] to every reward function, so do not penalize falls again.
The episode is truncated after {he.EPISODE_STEPS} steps."""


def baseline_result(path: str | pathlib.Path | None = None) -> str:
    """The D-GO result of the baseline reward from rl/runs/baseline/summary.json, or "not measured yet"."""
    try:
        data = json.loads(
            pathlib.Path(BASELINE_SUMMARY if path is None else path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "not measured yet"
    if not isinstance(data, dict) or data.get("success_rate") is None:
        return "not measured yet"
    try:
        rate = float(data["success_rate"])
    except (TypeError, ValueError):
        return "not measured yet"
    steps = data.get("steps") or data.get("requested_steps")
    head = f"PPO trained for {int(steps):,} steps" if isinstance(steps, (int, float)) else "PPO"
    parts = [f"success rate {rate:.0%}"]
    episodes = data.get("episodes")
    if isinstance(episodes, (int, float)) and episodes:
        parts.append(f"over {int(episodes)} evaluation episodes")
    mean = data.get("mean_steps_to_success")
    if isinstance(mean, (int, float)):
        parts.append(f"mean {float(mean):.1f} steps to success")
    return f"{head}: " + ", ".join(parts)


def trap_list() -> str:
    """The check_source rules the model must follow (DESIGN.md 8.5), as prompt text."""
    return f"""## Rules every file must follow (a static checker rejects the file otherwise)

- The file holds only: an optional docstring, the line `import math` (that exact form and nothing else: no \
numpy, no `from typing import ...`, no other import), constants `NAME = <literal>`, and `def` functions.
- The entry point is exactly `def reward(obs, action, prev_action, info):` at the top level, with those four \
parameter names and no others. It returns a finite float.
- At most 200 lines.
- No name that starts with an underscore anywhere, not even a bare `_` loop variable, and no \
`if __name__ == "__main__":` block.
- No annotation that names a forbidden thing (`info: object`) and none of these names anywhere: \
{FORBIDDEN_NAMES}.
- No class, no `with`, no `global` or `nonlocal`, no async/await, no `yield`, no bare `except:`, no \
`BaseException`, and no `return`, `break` or `continue` inside a `finally` block.
- `obs`, `action` and `prev_action` are plain Python lists of floats and `info` is a dict: use only `math` \
and built-in functions (abs, min, max, sum, len, zip, range, float). No numpy in reward functions.
- Each variant is a complete file with its own docstring, constants and `reward` function."""


def build_prompt(baseline_source: str | None = None, result: str | None = None) -> str:
    """The user prompt: environment, baseline reward and its result, the four ideas, the rules."""
    source = BASELINE_PATH.read_text(encoding="utf-8") if baseline_source is None else baseline_source
    measured = baseline_result() if result is None else result
    ideas = "\n".join(f"{k}. {idea}" for k, idea in enumerate(SHAPING_IDEAS, 1))
    return f"""You write reward functions for a reinforcement-learning policy that brings a robot arm's
gripper tip to a point 5 cm above a person's hand, so the arm can hand an object over.

{environment_description()}

## The baseline reward and its result

```python
{source.strip()}
```

Result of the baseline: {measured}.

## What to write

Four variants of `def reward(obs, action, prev_action, info)`, each shaped by a different idea, in this order:
{ideas}

Keep every return value a finite float of moderate size (per-step magnitudes around 0.01 to 2 are fine; a
success bonus may be larger). Use info["dist"] for the distance and info["success"] for the success bonus.
A progress term needs the previous distance, which is not in `info`: estimate it from the current obs and
`prev_action`, or keep the term proportional to -info["dist"] instead.

{trap_list()}

{CLOSING_LINE}"""


def retry_prompt(k: int, problems: Sequence[str]) -> str:
    """The follow-up request for one rejected variant, with the checker's problems."""
    listed = "\n".join(f"- {problem}" for problem in problems) or "- rejected"
    return (
        f"Variant {k} ({SHAPING_IDEAS[k - 1]}) was rejected by the static checker:\n{listed}\n\n"
        f"Write variant {k} again as one complete file that follows every rule above. "
        "Return one Python code block and nothing else."
    )


def generated_header(k: int, model: str, when: str) -> str:
    """The comment lines that open a written variant: provenance, then HEADER_LINES (ruff skips the file)."""
    note = (
        f"# Generated by rl/rewards/write_rewards.py: variant {k} ({SHAPING_IDEAS[k - 1]}), model {model}, "
        f"{when}. AI-written code: never edit, lint or import it."
    )
    return "\n".join((note, *HEADER_LINES)) + "\n"


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _normalized(src: str) -> str:
    """Unix line endings and exactly one trailing newline."""
    return src.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"


def _split_reply(reply: str) -> list[str | None]:
    """The first N_VARIANTS fenced blocks holding `def reward`, padded with None."""
    blocks = [block for block in extract_code_blocks(reply) if "def reward(" in block]
    out: list[str | None] = list(blocks[:N_VARIANTS])
    out.extend([None] * (N_VARIANTS - len(out)))
    return out


def write_rewards(
    *,
    out_dir: str | pathlib.Path = REWARDS_DIR,
    task_dir: str | pathlib.Path | None = None,
    timeout: float = 60,
    task_id: str | None = "rewards",
    model: str | None = None,
) -> list[pathlib.Path]:
    """Ask the code model for four reward variants, check them, write the accepted ones (module doc)."""
    out = pathlib.Path(out_dir)
    run_dir = pathlib.Path(task_dir) if task_dir is not None else RUNS_DIR / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    model_name = model or codegen_model()
    prompt = build_prompt()
    _write(run_dir / "prompt.md", prompt)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    reply = str(
        chat(messages, model_name, timeout=timeout, purpose="write_rewards", key=None, task_id=task_id)
    )
    _write(run_dir / "reply.md", reply)
    sources = _split_reply(reply)

    problems: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    written: list[pathlib.Path] = []
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for k in range(1, N_VARIANTS + 1):
        src = sources[k - 1]
        # What is checked is exactly what gets written: the header counts toward the 200-line limit too.
        text = None if src is None else generated_header(k, model_name, stamp) + _normalized(src)
        found = (
            ["no code block for this variant in the reply"] if text is None else check_source(text, "reward")
        )
        attempts = 1
        if found:
            problems[f"variant_{k}"] = list(found)
            log.warning("reward variant %d rejected (%s); asking once more", k, "; ".join(found))
            request = retry_prompt(k, found)
            retry_messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {"role": "user", "content": request},
            ]
            retry_reply = str(
                chat(
                    retry_messages,
                    model_name,
                    timeout=timeout,
                    purpose="write_rewards",
                    key=f"variant_{k}",
                    task_id=task_id,
                )
            )
            _write(run_dir / f"retry_{k}.md", f"# Request\n\n{request}\n\n# Reply\n\n{retry_reply}\n")
            text = generated_header(k, model_name, stamp) + _normalized(extract_code(retry_reply))
            found = check_source(text, "reward")
            attempts = 2
            if found:
                problems[f"variant_{k}:retry"] = list(found)
                log.error("reward variant %d rejected twice (%s); not written", k, "; ".join(found))
        record: dict[str, Any] = {
            "variant": k,
            "hint": SHAPING_IDEAS[k - 1],
            "attempts": attempts,
            "accepted": not found,
            "path": None,
        }
        if not found and text is not None:
            path = out / f"variant_{k}.py"
            _write(path, text)
            written.append(path)
            record["path"] = str(path)
        records.append(record)
    if problems:
        _write(run_dir / "problems.json", json.dumps(problems, indent=2) + "\n")
    manifest = {"out_dir": str(out), "model": model_name, "run_dir": str(run_dir), "variants": records}
    _write(run_dir / "variants.json", json.dumps(manifest, indent=2) + "\n")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.rewards.write_rewards",
        description="Ask the code model for four hand-off reward variants and write the accepted ones",
    )
    parser.add_argument("--out", default=str(REWARDS_DIR), help="directory for variant_<k>.py")
    parser.add_argument(
        "--task-dir", default=None, help="where the prompt and reply go (default runs/rewards/<ts>)"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="budget per model call in seconds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        paths = write_rewards(out_dir=args.out, task_dir=args.task_dir, timeout=args.timeout)
    except Exception as exc:  # the model call or the file system failed: one line, no traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    if len(paths) < N_VARIANTS:
        print(f"error: only {len(paths)} of {N_VARIANTS} variants were accepted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
