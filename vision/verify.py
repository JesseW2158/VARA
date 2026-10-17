"""Real verification (B7): the camera check after a real run, with the same `sim.success.evaluate`.

`before` is `snapshot(10)` taken when planning starts (the scene the sandbox saw); `take_after` waits
1 s after the arm is home, takes `snapshot(10)` and retries for up to 3 s while an object the spec names
is unseen (a TimeoutError from the tracker, the camera delivering no frames, is retried inside the same
window and then raised: a dead camera ends the task as an error instead of verifying a stale scene).
The window is measured on an injectable clock; with an injected `sleep` and no clock it runs on a clock
that the sleeps advance, so tests neither wait nor spin.
`verify` runs `evaluate(spec, before, after, args, events)` (the implicit obstacle rule
included), lists the named objects unseen in `after` and, when `task_dir` is given, saves
`before.json`, `after.json` and `verify.json` there. A malformed spec (ValueError, KeyError or TypeError
from evaluate) is a failed verification with the message as the reason, never a traceback.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from sim.success import OBSTACLE_MAX_MOVE, evaluate, resolve_object

__all__ = ["missing_objects", "spec_objects", "take_after", "verify"]

log = logging.getLogger(__name__)

_OBJECT_PREDICATES = ("in_zone", "near", "moved", "not_moved")
RETRY_PAUSE_S = 0.2
CLOCK_EPS = 1e-9  # deadline slack for clocks that add up 0.2 s steps in floating point


def spec_objects(
    spec: Mapping[str, Any], before: Mapping[str, Any], args: Mapping[str, Any] | None
) -> list[str]:
    """Every object name the spec refers to, `$params` resolved through `args` and `"*"` through `before`.

    Order of first appearance, no duplicates. A `$param` missing from `args` contributes nothing here
    (evaluate reports it as a ValueError).
    """
    if not isinstance(spec, Mapping):
        return []  # evaluate reports the malformed spec as a ValueError
    predicate = spec["success"] if "success" in spec else spec
    names: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, Mapping) or len(node) != 1:
            return
        ((key, body),) = node.items()
        if key in ("all", "any") and isinstance(body, (list, tuple)):
            for child in body:
                walk(child)
        elif key == "not":
            walk(body)
        elif key in _OBJECT_PREDICATES and isinstance(body, Mapping) and "object" in body:
            try:
                resolved = resolve_object(body["object"], before, args)
            except ValueError:
                return
            for name in resolved:
                if name not in names:
                    names.append(name)

    walk(predicate)
    return names


def missing_objects(
    spec: Mapping[str, Any],
    after: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
    args: Mapping[str, Any] | None = None,
) -> list[str]:
    """Names the spec refers to that are absent or `seen: false` in `after`."""
    objects = after.get("objects") or {}
    names = spec_objects(spec, before if before is not None else after, args)
    return [n for n in names if n not in objects or not objects[n].get("seen", False)]


def _moved_obstacles(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    after_obstacles = after.get("obstacles") or {}
    for name, start in (before.get("obstacles") or {}).items():
        end = after_obstacles.get(name)
        if end is None:
            out.append(f"obstacle {name} is missing from the after snapshot")
            continue
        moved = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
        if moved > OBSTACLE_MAX_MOVE + 1e-9:
            out.append(f"obstacle {name} moved {moved * 100:.1f} cm (max {OBSTACLE_MAX_MOVE * 100:.1f} cm)")
    return out


def verify(
    spec: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    args: Mapping[str, Any] | None,
    events: Iterable[Any] | None,
    *,
    task_dir: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    """`{"ok": bool, "missing": [names unseen in after], "reason": str}` from the success evaluator.

    Saves before.json, after.json and verify.json under `task_dir` when given (created if needed).
    """
    event_list = list(events) if events is not None else []
    arguments = dict(args) if args is not None else {}
    missing = missing_objects(spec, after, before, arguments)
    try:
        ok = bool(evaluate(spec, before, after, arguments, event_list))
        if ok:
            reason = "spec satisfied"
        else:
            problems = _moved_obstacles(before, after)
            if missing:
                problems.append("unseen after the run: " + ", ".join(missing))
            reason = "; ".join(problems) if problems else "spec not satisfied"
    except ValueError as exc:
        ok = False
        reason = f"spec error: {exc}"
    except (KeyError, TypeError) as exc:  # a predicate body missing a field, or the wrong type
        ok = False
        reason = f"spec error: {type(exc).__name__}: {exc}"
    result = {"ok": ok, "missing": missing, "reason": reason}
    if task_dir is not None:
        folder = pathlib.Path(task_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
        (folder / "after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
        record = {**result, "spec": spec, "args": arguments, "events": event_list, "t": time.time()}
        (folder / "verify.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("verify: ok=%s missing=%s reason=%s", ok, missing, reason)
    return result


def _timers(
    sleep: Callable[[float], None] | None, clock: Callable[[], float] | None
) -> tuple[Callable[[float], None], Callable[[], float]]:
    """The (pause, now) pair `take_after` runs on: `time.sleep` and `time.monotonic` by default, the
    injected ones when given. An injected `sleep` without a `clock` gets a clock that advances only by
    the sleeps, so a no-op sleep walks through the retry window in ceil(retry / 0.2) retries instead of
    spinning until the wall clock passes it, and the window does not depend on how long each `snapshot`
    takes."""
    if sleep is None:
        return time.sleep, (clock if clock is not None else time.monotonic)
    if clock is not None:
        return sleep, clock
    elapsed = [0.0]

    def pause(seconds: float) -> None:
        sleep(seconds)
        elapsed[0] += float(seconds)

    return pause, lambda: elapsed[0]


def take_after(
    tracker: Any,
    spec: Mapping[str, Any],
    args: Mapping[str, Any] | None,
    *,
    wait: float = 1.0,
    retry: float = 3.0,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    before: Mapping[str, Any] | None = None,
    n: int = 10,
) -> dict[str, Any]:
    """The after-snapshot: wait `wait` s (the arm is home), `snapshot(n)`, then retry every 0.2 s while
    a named object is unseen, for up to `retry` s measured on `clock` (default `time.monotonic`, so a
    slow `snapshot` eats into the window) and at most ceil(retry / 0.2) retries whatever the clock says.
    A TimeoutError from `snapshot` (the camera delivers no frames) is retried the same way and re-raised
    when it persists. `before` resolves a `"*"` in the spec. `sleep` and `clock` are injectable: with a
    `sleep` and no `clock` the window runs on a clock that only the sleeps advance (`_timers`), so a test
    with a no-op sleep neither waits nor spins. A non-finite or negative `wait`/`retry` counts as 0."""
    pause, now = _timers(sleep, clock)
    if math.isfinite(wait) and wait > 0:
        pause(wait)
    window = max(0.0, float(retry)) if math.isfinite(retry) else 0.0
    retries_left = math.ceil(window / RETRY_PAUSE_S - CLOCK_EPS) if window > 0 else 0
    deadline = now() + window
    while True:
        try:
            after = tracker.snapshot(n)
        except TimeoutError as exc:
            if retries_left <= 0 or now() >= deadline - CLOCK_EPS:
                raise
            log.warning("take_after: no snapshot (%s); retrying", exc)
        else:
            if not missing_objects(spec, after, before, args):
                return after
            if retries_left <= 0 or now() >= deadline - CLOCK_EPS:
                return after
        retries_left -= 1
        pause(RETRY_PAUSE_S)
