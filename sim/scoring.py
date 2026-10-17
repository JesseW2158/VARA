"""Scoring of candidate trials (docs/interfaces.md, "Scoring"; brief sections 4 and 8.4).

Pure functions, standard library only, shared by `sim.run_candidate` (inside the sandbox or the clean local
subprocess) and the orchestrator (ranking):

- `perturb_scene(scene, trial_index, seed)`: trial 1 is the scene as observed; trials 2.. shift every object
  by uniform +-0.01 m in x and y and +-10 degrees of yaw and every obstacle by +-0.005 m in x and y, drawn
  from `random.Random(f"{seed}:{trial_index}")` in scene order, so every candidate faces the same scenes.
  Always a deep copy; the input is never touched.
- `trial_result(index, success, stats, error)`: one entry of `out/trials.json` (brief 8.4): `trial`,
  `success`, `collisions`, `limit_hits`, `ik_failures`, `sim_s`, `error`, `first_collision`. `stats` is
  `SimRobot.stats` or None for a trial that never produced counters (a timeout), whose counters are 0 and
  whose `sim_s` is None; a non-finite `sim_s` (unreachable from SimRobot, whose clock is ticks / 30) also
  becomes None, since NaN can neither be averaged nor written as JSON. `error` and `first_collision` are
  clipped with `clip` (a candidate can raise an exception carrying megabytes; the score line and the
  dashboard phrase must stay short).
- `fail_reason(trial)`: the dashboard phrase of a failing trial, None for a clean success. Precedence inside
  one trial: "trial 1: timeout", "trial 1: error: <msg>", "trial 3: gripper hit cereal_box" (the trial's
  `first_collision`), "trial 4: limit hit", "trial 2: spec not satisfied".
- `summarize(index, trial_results, traj_path, trials=None)`: the score line, exactly schemas/score.json.
  Pass rule: trial 1 succeeded, at least `required_successes(trials)` trials succeeded (the contract's 4 of
  5, generalized as 80 % rounded up: 1 of 1, 4 of 5), zero collisions, zero limit hits and no error over
  every trial, and every requested trial ran. `fail_reason` is the first failing trial's phrase, hard
  failures (error, timeout, collision, limit hit) before a missed spec, and None on a passing candidate
  (one tolerated spec miss on a perturbed trial is not reported); `error` the first error message;
  `nominal_success` trial 1's success; `mean_sim_s` the mean over the trials that reported a simulated
  time (None when none did); `traj` the path given or None.
- `rejected_score(index, trials, problem)`: the score of a candidate `check_source` rejected: nothing ran,
  `error` and `fail_reason` are "rejected: <first problem>" (clipped).
- `clip(text, limit=MAX_TEXT_CHARS)`: `text` within `limit` characters, ending in a note of the original
  length when it was cut; idempotent, so the child's and the parent's clipping agree.
- `rank(scores)`: passing candidates first, then successes descending, then mean_sim_s ascending (a missing
  mean sorts last), then the candidate index. Returns a new list; the input is not changed.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "MAX_TEXT_CHARS",
    "OBJECT_SHIFT_M",
    "OBJECT_YAW_DEG",
    "OBSTACLE_SHIFT_M",
    "PASS_FRACTION",
    "TIMEOUT",
    "TRIAL_KEYS",
    "clip",
    "fail_reason",
    "perturb_scene",
    "rank",
    "rejected_score",
    "required_successes",
    "summarize",
    "trial_result",
]

OBJECT_SHIFT_M = 0.01
OBJECT_YAW_DEG = 10.0
OBSTACLE_SHIFT_M = 0.005
PASS_FRACTION = 0.8  # the contract's "at least 4 of 5"
TIMEOUT = "timeout"  # the error string of a trial the parent killed on the wall clock
REJECTED = "rejected"
MAX_TEXT_CHARS = 500  # error texts and collision phrases are clipped to this (the dashboard shows ~200)

# The keys of one `out/trials.json` entry, in order (brief 8.4).
TRIAL_KEYS = (
    "trial",
    "success",
    "collisions",
    "limit_hits",
    "ik_failures",
    "sim_s",
    "error",
    "first_collision",
)


def perturb_scene(scene: Mapping[str, Any], trial_index: int, seed: int | str) -> dict:
    """The scene of trial `trial_index` (1-based): a deep copy, unchanged for trial 1, perturbed for the
    rest with `random.Random(f"{seed}:{trial_index}")`. Raises ValueError for a trial index under 1."""
    index = int(trial_index)
    if index < 1:
        raise ValueError(f"trial_index must be 1 or more, got {trial_index!r}")
    out = copy.deepcopy(dict(scene))
    if index == 1:
        return out
    rng = random.Random(f"{seed}:{index}")
    for item in (out.get("objects") or {}).values():
        item["x"] = float(item["x"]) + rng.uniform(-OBJECT_SHIFT_M, OBJECT_SHIFT_M)
        item["y"] = float(item["y"]) + rng.uniform(-OBJECT_SHIFT_M, OBJECT_SHIFT_M)
        item["yaw"] = float(item["yaw"]) + math.radians(rng.uniform(-OBJECT_YAW_DEG, OBJECT_YAW_DEG))
    for item in (out.get("obstacles") or {}).values():
        item["x"] = float(item["x"]) + rng.uniform(-OBSTACLE_SHIFT_M, OBSTACLE_SHIFT_M)
        item["y"] = float(item["y"]) + rng.uniform(-OBSTACLE_SHIFT_M, OBSTACLE_SHIFT_M)
    return out


def clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """`text` within `limit` characters: returned as is when it fits, else cut and ended with a note of
    its original length (the result is exactly `limit` long, so clipping twice changes nothing). Every
    error text that reaches a score line or `out/trials.json` goes through here."""
    text = str(text)
    if len(text) <= limit:
        return text
    note = f" ... (clipped from {len(text)} characters)"
    if len(note) >= limit:
        return text[:limit]
    return text[: limit - len(note)] + note


def trial_result(index: int, success: bool, stats: Mapping[str, Any] | None, error: str | None) -> dict:
    """One entry of `out/trials.json`. `stats` is `SimRobot.stats` (or None: counters 0, `sim_s` None).
    `error` and `first_collision` are clipped to `MAX_TEXT_CHARS`."""
    counters = stats or {}
    sim_s = None if counters.get("sim_s") is None else float(counters["sim_s"])
    if sim_s is not None and not math.isfinite(sim_s):
        sim_s = None  # NaN or inf cannot be averaged or written as JSON: the trial reported no time
    first = counters.get("first_collision")
    return {
        "trial": int(index),
        "success": bool(success),
        "collisions": int(counters.get("collisions", 0) or 0),
        "limit_hits": int(counters.get("limit_hits", 0) or 0),
        "ik_failures": int(counters.get("ik_failures", 0) or 0),
        "sim_s": sim_s,
        "error": None if error is None else clip(error),
        "first_collision": None if first is None else clip(first),
    }


def fail_reason(trial: Mapping[str, Any]) -> str | None:
    """The phrase for the dashboard card when the trial fails the pass rule, else None."""
    k = int(trial["trial"])
    error = trial.get("error")
    if error == TIMEOUT:
        return f"trial {k}: timeout"
    if error:
        return f"trial {k}: error: {error}"
    if int(trial.get("collisions") or 0) > 0:
        return f"trial {k}: {trial.get('first_collision') or 'collision'}"
    if int(trial.get("limit_hits") or 0) > 0:
        return f"trial {k}: limit hit"
    if not trial.get("success"):
        return f"trial {k}: spec not satisfied"
    return None


def required_successes(trials: int) -> int:
    """How many of `trials` must succeed: the contract's 4 of 5 as 80 % rounded up, at least 1."""
    return max(1, math.ceil(PASS_FRACTION * int(trials) - 1e-9))


def summarize(
    index: int,
    trial_results: Iterable[Mapping[str, Any]],
    traj_path: str | None,
    trials: int | None = None,
) -> dict:
    """The score line (schemas/score.json) of candidate `index` from the trials that ran.

    `trials` is the requested count (default: the number of results given); results are at most that many
    and are summarized in the order given. A result that stopped the run (a timeout) is the last one.
    """
    results = [dict(t) for t in trial_results]
    n = int(trials) if trials is not None else len(results)
    if n < 1:
        raise ValueError("a score needs at least one trial")
    if len(results) > n:
        raise ValueError(f"{len(results)} results for {n} trials")
    successes = sum(1 for t in results if t.get("success"))
    nominal = any(int(t["trial"]) == 1 and t.get("success") for t in results)
    sim_times = [
        float(t["sim_s"])
        for t in results
        if isinstance(t.get("sim_s"), (int, float)) and math.isfinite(t["sim_s"])
    ]
    mean_sim_s = round(sum(sim_times) / len(sim_times), 3) if sim_times else None
    error = next((str(t["error"]) for t in results if t.get("error")), None)
    reason = _blocking_reason(results, n, nominal, successes)
    return {
        "candidate": int(index),
        "passed": reason is None,
        "successes": int(successes),
        "trials": n,
        "nominal_success": bool(nominal),
        "collisions": sum(int(t.get("collisions") or 0) for t in results),
        "limit_hits": sum(int(t.get("limit_hits") or 0) for t in results),
        "ik_failures": sum(int(t.get("ik_failures") or 0) for t in results),
        "mean_sim_s": mean_sim_s,
        "fail_reason": reason,
        "traj": None if traj_path is None else str(traj_path),
        "error": error,
    }


def _hard_failure(trial: Mapping[str, Any]) -> bool:
    """An error (timeout included), a collision or a limit hit: never tolerated, whatever the trial."""
    return (
        bool(trial.get("error"))
        or int(trial.get("collisions") or 0) > 0
        or int(trial.get("limit_hits") or 0) > 0
    )


def _blocking_reason(results: list[dict], trials: int, nominal: bool, successes: int) -> str | None:
    """Why the candidate fails, as the first failing trial's phrase, or None when it passes.

    A hard failure anywhere comes first; then trial 1 failing its spec; then too few successes (the first
    trial that missed the spec); then a trial that never ran. A spec miss on one perturbed trial among
    enough successes is tolerated by the contract, so it never shows up on a passing candidate.
    """
    for trial in results:
        if _hard_failure(trial):
            return fail_reason(trial)
    if not nominal:
        first = next((t for t in results if int(t["trial"]) == 1), None)
        return fail_reason(first) if first is not None else "trial 1: not run"
    missed = next((fail_reason(t) for t in results if not t.get("success")), None)
    if successes < required_successes(trials) and missed is not None:
        return missed
    if len(results) < trials:
        return f"trial {len(results) + 1}: not run"
    return None


def rejected_score(index: int, trials: int, problem: str) -> dict:
    """The score of a candidate that `check_source` rejected: no trial ran. The problem text is clipped."""
    message = f"{REJECTED}: {clip(problem)}"
    return {
        "candidate": int(index),
        "passed": False,
        "successes": 0,
        "trials": max(1, int(trials)),
        "nominal_success": False,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "mean_sim_s": None,
        "fail_reason": message,
        "traj": None,
        "error": message,
    }


def rank(scores: Iterable[Mapping[str, Any]]) -> list:
    """Scores ordered best first: passing, then more successes, then less simulated time, then index."""

    def key(score: Mapping[str, Any]) -> tuple[int, int, float, int]:
        mean = score.get("mean_sim_s")
        return (
            0 if score.get("passed") else 1,
            -int(score.get("successes") or 0),
            math.inf if mean is None else float(mean),
            int(score.get("candidate") or 0),
        )

    return sorted(scores, key=key)
