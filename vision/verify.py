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

CLI (`python -m vision.verify`, B7's live check on the table; nothing here moves the arm, so no --real):

    python -m vision.verify --before                       # snapshot(10) now -> runs/verify-check/<UTC>/
    ... move the object by hand to the end state ...
    python -m vision.verify --after --spec skills/_starter/place_in_tray.json --args '{"object": "juice_box"}'

The first snapshot of a run comes after the camera has run for `--warmup` seconds (default 1;
`vision.scene.warm_up`, nothing to settle on a FakeSceneTracker): a just-opened camera can deliver dark
frames. `--before` saves `before.json` and `session.json` (mode, time, frames, warm-up, tracker) in a new
`runs/verify-check/<UTC stamp>/` (VARA_RUNS_DIR respected) or in `--out DIR`. `--after --spec FILE [--args
JSON] [--events JSON]` reads the `before.json` of `--out DIR` or, by default, of the newest open session
(`latest_session`: the newest before.json in a folder no `--after` wrote to), takes the after snapshot
through `take_after` (its 1 s wait and 3 s retry while a named object is unseen; `--wait`, `--retry`),
runs `verify` (the implicit obstacle rule included) and saves `after.json`, `verify.json` (the record
`verify(task_dir=...)` writes) and `result.json` (mode, times, verdict, file paths) next to the before,
which is never rewritten. `--spec` takes a task spec or a saved skill file (its `spec`; the skill's `code`
is never run), validated against schemas/task_spec.json; `--args` must give every param the spec declares.
`--before --after` together prompts for Enter between the two snapshots (Ctrl+C there keeps the before for
a later `--after --out DIR`). Nothing is overwritten or mixed: an `--after` is refused in a session that
holds any of after.json, verify.json or result.json, a `--before` in a folder that holds any session file,
and `--out` may not lie inside config/, skills/, schemas/ or tests/. A physical `--after` needs the
session.json of a physical `--before`: a before of unknown provenance is refused. Exit 0 when the spec
holds, 1 when it does not, 2 on bad input, misuse (`--spec`, `--args`, `--events`, `--wait` or `--retry`
without `--after`, a negative or non-finite duration) or a camera that delivers no frames.

session.json and result.json also carry `tracker_diagnostics`, `vision.scene.tracker_diagnostics(tracker)`
= `{type, frames, misses, errors, lens}`, where `lens` is the sha256 of the config/lens.npz the tracker
undistorts every frame with, or null for raw frames (always null for the FakeSceneTracker); the CLI prints
it as a `lens:` line. An `--after` whose tracker reports another lens than the session's before is refused
(exit 2: raw and undistorted positions never meet in one check; take a new --before). A session.json
written before this record existed has no `tracker_diagnostics` and is not checked.

A fake tracker (FAKE/FAKE_CAMERA=1, or a fake installed in-process, a SceneTracker on a FakeCamera
included) is refused unless `--fake` is passed. Then a FakeSceneTracker serves config/fake_scene.json or
`--fake-scene FILE`, and the run is labelled synthetic: the printed lines start with SYNTHETIC,
session.json and result.json carry `mode: synthetic` and a note, a default session folder is named
`<UTC stamp>-synthetic` (before.json and after.json are plain scenes: schemas/scene.json has no room for a
label), and a synthetic before never pairs with a physical after or the reverse.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import pathlib
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from sim.success import OBSTACLE_MAX_MOVE, evaluate, resolve_object

__all__ = [
    "build_tracker",
    "latest_session",
    "load_spec",
    "main",
    "missing_objects",
    "new_session_dir",
    "parse_args_json",
    "parse_events_json",
    "run_after",
    "spec_objects",
    "take_after",
    "take_before",
    "verify",
]

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_SCHEMA_PATH = ROOT / "schemas" / "task_spec.json"
_OBJECT_PREDICATES = ("in_zone", "near", "moved", "not_moved")
RETRY_PAUSE_S = 0.2
CLOCK_EPS = 1e-9  # deadline slack for clocks that add up 0.2 s steps in floating point
BEFORE_FILE = "before.json"
AFTER_FILE = "after.json"
VERIFY_FILE = "verify.json"
SESSION_FILE = "session.json"
RESULT_FILE = "result.json"
_AFTER_FILES = (AFTER_FILE, VERIFY_FILE, RESULT_FILE)  # written by --after: the session is used up
SYNTHETIC_NOTE = "SYNTHETIC - a FakeSceneTracker scene, not a check on the table"
SYNTHETIC_LABEL = "synthetic"  # the default folder of a --fake session: <UTC stamp>-synthetic[-<n>]
WARMUP_S = 1.0  # the CLI's default camera warm-up before its first snapshot
WAIT_S = 1.0  # take_after's defaults, repeated for the CLI
RETRY_S = 3.0
_SESSION_NAME = re.compile(r"(\d{8}-\d{6})(?:-synthetic)?(?:-(\d+))?")  # <UTC stamp>[-synthetic][-<n>]


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
        record = _verify_record(result, spec, arguments, event_list)
        (folder / "verify.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("verify: ok=%s missing=%s reason=%s", ok, missing, reason)
    return result


def _verify_record(
    result: Mapping[str, Any], spec: Mapping[str, Any], args: Mapping[str, Any], events: list[Any]
) -> dict[str, Any]:
    """verify.json: the verdict plus the spec, args and events it was evaluated with, and the time."""
    return {**result, "spec": spec, "args": dict(args), "events": list(events), "t": time.time()}


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


# ------------------------------------------------------------------------------------------------------------
# CLI: python -m vision.verify (B7's live check; nothing here moves the arm)
# ------------------------------------------------------------------------------------------------------------


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


def _runs_dir() -> pathlib.Path:
    value = os.environ.get("VARA_RUNS_DIR", "").strip()
    return pathlib.Path(value) if value else ROOT / "runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: pathlib.Path, data: Any) -> None:
    """Write JSON to a path that must not exist yet (evidence is never overwritten)."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


@functools.lru_cache(maxsize=1)
def _spec_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SPEC_SCHEMA_PATH.read_text(encoding="utf-8")))


def load_spec(path: pathlib.Path | str) -> dict[str, Any]:
    """A task spec from a task-spec JSON file or a saved skill file (its `spec`), validated against
    schemas/task_spec.json. ValueError for anything else."""
    file = pathlib.Path(path)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: not valid JSON ({exc})") from exc
    if isinstance(data, Mapping) and "success" not in data and isinstance(data.get("spec"), Mapping):
        data = data["spec"]  # a saved skill (schemas/skill.json) carries its task spec under `spec`
    if not isinstance(data, Mapping):
        raise ValueError(f"{file}: expected a task spec object (schemas/task_spec.json) or a skill file")
    errors = list(_spec_validator().iter_errors(data))
    if errors:
        err = best_match(errors) or errors[0]
        raise ValueError(f"{file}: spec {err.json_path}: {err.message}")
    return dict(data)


def parse_args_json(text: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """`--args` as a dict; every param the spec declares must be present. ValueError otherwise."""
    try:
        args = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--args is not valid JSON ({exc})") from exc
    if not isinstance(args, dict):
        raise ValueError('--args must be a JSON object, for example {"object": "juice_box"}')
    missing = [name for name in (spec.get("params") or {}) if name not in args]
    if missing:
        raise ValueError(f"--args is missing the spec's params: {', '.join(missing)}")
    return args


def parse_events_json(text: str) -> list[Any]:
    """`--events` as a list (the run's events for `event` predicates). ValueError otherwise."""
    try:
        events = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--events is not valid JSON ({exc})") from exc
    if not isinstance(events, list):
        raise ValueError('--events must be a JSON list, for example ["handoff_done"]')
    return events


def _load_scene_file(path: pathlib.Path | str) -> dict[str, Any]:
    from vision.scene import validate_scene

    file = pathlib.Path(path)
    try:
        scene = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: not valid JSON ({exc})") from exc
    if not isinstance(scene, dict):
        raise ValueError(f"{file}: expected a scene object (schemas/scene.json)")
    validate_scene(scene)
    return scene


def build_tracker(fake: bool, fake_scene: pathlib.Path | str | None = None) -> Any:
    """The CLI's scene source: under `fake` a fresh FakeSceneTracker serving config/fake_scene.json (or the
    scene file `fake_scene`, validated), else the process-wide `vision.scene.get_tracker()`, a started
    SceneTracker on the overhead camera (or the fake FAKE_CAMERA installed, which `main` then refuses).
    vision.scene is imported here, not at module import, so the orchestrator's import stays light."""
    from vision.scene import FakeSceneTracker, get_tracker

    if fake:
        scene = _load_scene_file(fake_scene) if fake_scene is not None else None
        return FakeSceneTracker(scene)
    if fake_scene is not None:
        raise ValueError("a scene file is only served by the fake tracker (--fake)")
    return get_tracker()


def new_session_dir(root: pathlib.Path | str | None = None, *, synthetic: bool = False) -> pathlib.Path:
    """A new, created `runs/verify-check/<UTC stamp>[-synthetic][-<n>]/` (VARA_RUNS_DIR respected) or under
    `root`, `-synthetic` for a `--fake` session; `mkdir` without exist_ok picks the name, so two runs in the
    same second never share a folder."""
    base = pathlib.Path(root) if root is not None else _runs_dir() / "verify-check"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if synthetic:
        stamp = f"{stamp}-{SYNTHETIC_LABEL}"
    n = 0
    while True:
        folder = base / (stamp if n == 0 else f"{stamp}-{n}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            n += 1


def _session_order(folder: pathlib.Path) -> tuple[int, str, int]:
    """Sort key of an open session: its before.json's mtime, then the `<stamp>[-synthetic][-<n>]`
    name with n compared as a number (so `-10` follows `-9`); other names sort by the name itself."""
    match = _SESSION_NAME.fullmatch(folder.name)
    stamp, n = (match.group(1), int(match.group(2) or 0)) if match else (folder.name, 0)
    return (folder / BEFORE_FILE).stat().st_mtime_ns, stamp, n


def latest_session(root: pathlib.Path | str | None = None) -> pathlib.Path | None:
    """The session folder under runs/verify-check/ (or `root`) with the newest before.json and none of
    after.json, verify.json or result.json yet (no `--after` ran there), or None."""
    base = pathlib.Path(root) if root is not None else _runs_dir() / "verify-check"
    if not base.is_dir():
        return None
    open_sessions = [
        p
        for p in base.iterdir()
        if p.is_dir()
        and (p / BEFORE_FILE).is_file()
        and not any((p / name).exists() for name in _AFTER_FILES)
    ]
    return max(open_sessions, key=_session_order) if open_sessions else None


def _drop_if_empty(folder: pathlib.Path) -> None:
    """Remove a session folder this run created when nothing was saved in it (a failed --before)."""
    try:
        folder.rmdir()  # only succeeds on an empty folder
    except OSError:
        pass


def _summarize(scene: Mapping[str, Any]) -> str:
    objects = scene.get("objects") or {}
    obstacles = scene.get("obstacles") or {}
    seen_objects = [name for name, item in objects.items() if item.get("seen")]
    seen_obstacles = [name for name, item in obstacles.items() if item.get("seen")]
    hand = (scene.get("hand") or {}).get("seen")
    return (
        f"{len(seen_objects)} of {len(objects)} objects seen ({', '.join(seen_objects) or 'none'}), "
        f"{len(seen_obstacles)} of {len(obstacles)} obstacles seen, hand {'seen' if hand else 'unseen'}"
    )


def _warm_up(tracker: Any, seconds: float) -> int:
    """`vision.scene.warm_up` (imported here, like the trackers, to keep this module's import light)."""
    from vision.scene import warm_up

    return warm_up(tracker, seconds)


def _diagnostics(tracker: Any) -> dict[str, Any]:
    """`vision.scene.tracker_diagnostics(tracker)` (imported here, like the trackers): `{type, frames,
    misses, errors, lens}`, `lens` the tracker's lens sha256 or None for raw frames."""
    from vision.scene import tracker_diagnostics

    return tracker_diagnostics(tracker)


def _lens_text(lens: Any) -> str:
    return "null (raw frames)" if lens is None else str(lens)


def _tracker_is_fake(tracker: Any) -> bool:
    """True unless the tracker declares `is_fake = False` and so does its camera when it has one: a
    SceneTracker (is_fake False) running on a FakeCamera is a fake too."""
    if getattr(tracker, "is_fake", True) is not False:
        return True
    camera = getattr(tracker, "camera", None)
    return camera is not None and getattr(camera, "is_fake", True) is not False


def take_before(
    tracker: Any,
    out_dir: pathlib.Path | str,
    *,
    frames: int = 10,
    fake: bool = False,
    fake_scene: pathlib.Path | str | None = None,
    warmup: float = 0.0,
) -> dict[str, Any]:
    """`warm_up(tracker, warmup)`, then `tracker.snapshot(frames)` saved as `out_dir/before.json`, with
    `out_dir/session.json` (mode, time, frames, warm-up, tracker, tracker_diagnostics with the lens).
    Returns the snapshot. FileExistsError,
    before any snapshot, when the folder already holds a session file (before.json, session.json,
    after.json, verify.json or result.json: a session is never overwritten or mixed with another); a
    TimeoutError from the tracker (a dead camera) propagates."""
    folder = pathlib.Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    before_path = folder / BEFORE_FILE
    taken = [name for name in (BEFORE_FILE, SESSION_FILE, *_AFTER_FILES) if (folder / name).exists()]
    if taken:
        raise FileExistsError(
            f"{folder} already holds {', '.join(taken)}: take the new before in a new --out folder"
        )
    count = max(1, int(frames))
    warmed = _warm_up(tracker, warmup)
    before = tracker.snapshot(count)
    session: dict[str, Any] = {
        "mode": "synthetic" if fake else "physical",
        "fake": bool(fake),
        "before_at": _utc_now(),
        "frames": count,
        "warmup_s": _seconds(warmup),
        "warmup_frames": warmed,
        "tracker": type(tracker).__name__,
        "tracker_diagnostics": _diagnostics(tracker),
        "fake_scene": None if fake_scene is None else str(fake_scene),
    }
    if fake:
        session["note"] = SYNTHETIC_NOTE
    _write_new(before_path, before)
    _write_new(folder / SESSION_FILE, session)
    return before


def _seconds(value: float) -> float:
    """A duration as recorded: the value when finite and positive, else 0 (what the waits then do)."""
    return float(value) if math.isfinite(value) and value > 0 else 0.0


def _read_session(folder: pathlib.Path, fake: bool) -> dict[str, Any]:
    """The session.json `--before` wrote, checked against this run's mode.

    ValueError when it is not a JSON object with a boolean `fake`, when its mode differs from this run's
    (a check never mixes a synthetic and a physical snapshot), or when a physical run finds none (a
    before.json of unknown provenance, copied or written by hand). A synthetic run accepts a before.json
    without one: everything it writes is labelled synthetic anyway.
    """
    path = folder / SESSION_FILE
    mode = "synthetic" if fake else "physical"
    if not path.is_file():
        if fake:
            return {}
        raise ValueError(
            f"{path} is missing: a physical check needs a before taken by --before, not a before.json of "
            "unknown provenance"
        )
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(session, dict) or not isinstance(session.get("fake"), bool):
        raise ValueError(f"{path}: not a session file written by --before")
    if session["fake"] != bool(fake):
        raise ValueError(
            f"the before snapshot is {session.get('mode', 'unknown')} and this run is {mode}: a check never "
            "mixes the two"
        )
    return session


def run_after(
    tracker: Any,
    out_dir: pathlib.Path | str,
    spec: Mapping[str, Any],
    args: Mapping[str, Any] | None,
    events: Iterable[Any] | None = None,
    *,
    frames: int = 10,
    wait: float = WAIT_S,
    retry: float = RETRY_S,
    fake: bool = False,
    warmup: float = 0.0,
) -> dict[str, Any]:
    """The after half of a session: read `out_dir/before.json` (never rewritten) and its session.json,
    `warm_up(tracker, warmup)`, `take_after` (wait, then retry while a named object is unseen), `verify`,
    and write `after.json`, `verify.json` (the record `verify(task_dir=...)` writes) and `result.json` = the
    verdict plus mode, times, spec name, args, events, warm-up, `tracker_diagnostics` (with the lens) and
    file paths; returned.

    FileNotFoundError without a before.json, FileExistsError when the session was verified already,
    ValueError when the before file is not a scene, its session.json does not fit this run
    (`_read_session`) or records another lens than the tracker's, TimeoutError when the camera delivers no
    after snapshot.
    """
    folder = pathlib.Path(out_dir)
    before_path = folder / BEFORE_FILE
    if not before_path.is_file():
        raise FileNotFoundError(f"{before_path} is missing: run --before first")
    for name in (AFTER_FILE, VERIFY_FILE, RESULT_FILE):
        if (folder / name).exists():
            raise FileExistsError(
                f"{folder / name} exists: this session was verified already; take a new --before"
            )
    before = _load_scene_file(before_path)
    session = _read_session(folder, fake)
    recorded = session.get("tracker_diagnostics")
    lens = _diagnostics(tracker).get("lens")
    if isinstance(recorded, dict) and "lens" in recorded and recorded["lens"] != lens:
        raise ValueError(
            f"the before snapshot was taken with lens {_lens_text(recorded['lens'])} and this tracker uses "
            f"lens {_lens_text(lens)}: a check never mixes raw and undistorted frames; take a new --before"
        )
    arguments = dict(args) if args is not None else {}
    event_list = list(events) if events is not None else []
    count = max(1, int(frames))
    warmed = _warm_up(tracker, warmup)
    after = take_after(tracker, spec, arguments, wait=wait, retry=retry, before=before, n=count)
    result = verify(spec, before, after, arguments, event_list)  # no task_dir: before.json stays as taken
    _write_new(folder / AFTER_FILE, after)
    _write_new(folder / VERIFY_FILE, _verify_record(result, spec, arguments, event_list))
    record: dict[str, Any] = {
        **result,
        "mode": "synthetic" if fake else "physical",
        "fake": bool(fake),
        "before_at": session.get("before_at"),
        "after_at": _utc_now(),
        "frames": count,
        "warmup_s": _seconds(warmup),
        "warmup_frames": warmed,
        "wait_s": float(wait),
        "retry_s": float(retry),
        "spec_name": spec.get("name"),
        "args": arguments,
        "events": event_list,
        "tracker": type(tracker).__name__,
        "tracker_diagnostics": _diagnostics(tracker),
        "after_summary": _summarize(after),
        "files": {
            "before": str(before_path),
            "after": str(folder / AFTER_FILE),
            "verify": str(folder / VERIFY_FILE),
        },
    }
    if fake:
        record["note"] = SYNTHETIC_NOTE
    _write_new(folder / RESULT_FILE, record)
    return record


def _pause(prompt: str) -> None:
    """Wait for Enter; EOF (no terminal) counts as Enter so scripted runs never block."""
    try:
        input(prompt)
    except EOFError:
        print()


_EPILOG = """\
examples (Windows PowerShell 5.1 drops bare double quotes from native arguments, hence the \\" in --args;
in bash write --args '{"object": "juice_box"}'):
  python -m vision.verify --before
  ... move the object by hand to the end state ...
  python -m vision.verify --after --spec skills/place_in_tray.json --args '{\\"object\\":\\"juice_box\\"}'
  python -m vision.verify --before --after --spec tests/fixtures/task_spec_clear_table.json
  python -m vision.verify --fake --fake-scene scene.json --before --out runs/verify-check/offline
"""


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI (see the module docstring); returns the exit code: 0 the spec holds, 1 it does not, 2 bad
    input, misuse or no snapshot."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m vision.verify",
        description="B7 live check: a before snapshot, the object moved by hand, an after snapshot scored "
        "with sim.success.evaluate. Never moves the arm.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--before", action="store_true", help="take the before snapshot now and save it")
    parser.add_argument("--after", action="store_true", help="take the after snapshot and score --spec")
    parser.add_argument(
        "--spec",
        metavar="FILE",
        help="--after: task spec JSON (schemas/task_spec.json) or a saved skill file (its spec)",
    )
    parser.add_argument(
        "--args",
        default=None,
        metavar="JSON",
        help='--after: spec params, e.g. {"object": "juice_box"} (default {}; every declared param needed)',
    )
    parser.add_argument(
        "--events",
        default=None,
        metavar="JSON",
        help='--after: events for `event` predicates, e.g. ["handoff_done"] (default [])',
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        help="session folder (default: a new runs/verify-check/<UTC stamp>[-synthetic]/ for --before, the "
        "newest one without an after for --after; never inside config/, skills/, schemas/ or tests/)",
    )
    parser.add_argument("--frames", type=int, default=10, help="frames per snapshot (default 10)")
    parser.add_argument(
        "--warmup",
        type=float,
        default=WARMUP_S,
        metavar="S",
        help=f"seconds the camera runs before the run's first snapshot (default {WARMUP_S:g})",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="S",
        help=f"--after: seconds before the after snapshot (default {WAIT_S:g})",
    )
    parser.add_argument(
        "--retry",
        type=float,
        default=None,
        metavar="S",
        help=f"--after: seconds to retry while a named object is unseen (default {RETRY_S:g})",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="explicit offline run on a FakeSceneTracker; files labelled synthetic",
    )
    parser.add_argument(
        "--fake-scene",
        metavar="FILE",
        help="--fake: the scene JSON the fake tracker serves (schemas/scene.json)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if not (args.before or args.after):
        parser.print_help()
        return 2
    if args.after and not args.spec:
        parser.error("--after needs --spec FILE")
    if not args.after:
        only_after = (
            ("--spec", args.spec),
            ("--args", args.args),
            ("--events", args.events),
            ("--wait", args.wait),
            ("--retry", args.retry),
        )
        given = [flag for flag, value in only_after if value is not None]
        if given:
            parser.error(f"{', '.join(given)}: only with --after")
    if args.fake_scene and not args.fake:
        parser.error("--fake-scene needs --fake")
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    wait = WAIT_S if args.wait is None else args.wait
    retry = RETRY_S if args.retry is None else args.retry
    for flag, value in (("--warmup", args.warmup), ("--wait", wait), ("--retry", retry)):
        if not (math.isfinite(value) and value >= 0):
            parser.error(f"{flag} must be a finite number of seconds >= 0")
    if not args.fake and _fake("FAKE_CAMERA"):
        print(
            "FAKE_CAMERA=1 (or FAKE=1): the snapshots would come from config/fake_scene.json, not the table; "
            "pass --fake for a synthetic run or unset the fakes",
            file=sys.stderr,
        )
        return 2
    if args.out is not None:
        from vision.scene import protected_dir

        inside = protected_dir(args.out)
        if inside is not None:
            print(f"--out {args.out} is inside {inside}: the check never writes there", file=sys.stderr)
            return 2
    spec: dict[str, Any] = {}
    arguments: dict[str, Any] = {}
    events: list[Any] = []
    if args.after:
        try:
            spec = load_spec(args.spec)
            arguments = parse_args_json("{}" if args.args is None else args.args, spec)
            events = parse_events_json("[]" if args.events is None else args.events)
        except (OSError, ValueError) as exc:
            print(f"bad input: {exc}", file=sys.stderr)
            return 2
    existed = True  # whether the session folder was there before this run (else a failed --before drops it)
    try:
        if args.out is not None:
            folder = pathlib.Path(args.out)
            existed = folder.exists()
        elif args.before:
            folder = new_session_dir(synthetic=args.fake)
            existed = False
        else:
            found = latest_session()
            if found is None:
                print(
                    f"no --before session without an after under {_runs_dir() / 'verify-check'}: "
                    "run --before first (or pass --out DIR)",
                    file=sys.stderr,
                )
                return 2
            folder = found
    except OSError as exc:
        print(f"cannot use the session folder: {exc}", file=sys.stderr)
        return 2
    try:
        tracker = build_tracker(args.fake, args.fake_scene)
    except (OSError, RuntimeError, ValueError) as exc:
        if not existed:
            _drop_if_empty(folder)
        print(f"cannot start the scene tracker: {exc}", file=sys.stderr)
        return 2
    if not args.fake and _tracker_is_fake(tracker):
        if not existed:
            _drop_if_empty(folder)
        camera = getattr(tracker, "camera", None)
        on = f" on a {type(camera).__name__}" if camera is not None else ""
        print(
            f"vision.scene.get_tracker() is a {type(tracker).__name__}{on}, a fake: refusing a physical "
            "check (unset FAKE_CAMERA/FAKE or pass --fake)",
            file=sys.stderr,
        )
        return 2
    label = "SYNTHETIC " if args.fake else ""
    lens = _diagnostics(tracker).get("lens")
    undistortion = "every frame undistorted before detection" if lens is not None else "no undistortion"
    print(f"{label}lens: {'null' if lens is None else lens} ({undistortion})")
    hint = "" if args.fake else "keep the arm at home and hands out of the camera's view; "
    try:
        if args.before:
            if hint:
                warm = f"{args.warmup:g} s warm-up, then {args.frames} frames"
                print(f"before snapshot: {hint}{warm}", flush=True)
            try:
                before = take_before(
                    tracker,
                    folder,
                    frames=args.frames,
                    fake=args.fake,
                    fake_scene=args.fake_scene,
                    warmup=args.warmup,
                )
            except TimeoutError as exc:
                if not existed:
                    _drop_if_empty(folder)
                print(f"the camera delivered no before snapshot: {exc}", file=sys.stderr)
                return 2
            except (FileExistsError, OSError, ValueError) as exc:
                if not existed:
                    _drop_if_empty(folder)
                print(f"cannot save the before snapshot: {exc}", file=sys.stderr)
                return 2
            print(f"{label}before: {_summarize(before)}; saved {folder / BEFORE_FILE}")
            if args.after:
                try:
                    _pause("Move the objects by hand to the end state, then press Enter: ")
                except KeyboardInterrupt:
                    later = f"--after --spec FILE --out {folder}"
                    print(f"\ninterrupted: the before is kept; finish later with {later}", file=sys.stderr)
                    return 2
        if not args.after:
            print(f"session: {folder} (next: --after --spec FILE [--args JSON])")
            return 0
        if hint:
            print(f"after snapshot: {hint}{wait:g} s wait, then {args.frames} frames", flush=True)
        try:
            record = run_after(
                tracker,
                folder,
                spec,
                arguments,
                events,
                frames=args.frames,
                wait=wait,
                retry=retry,
                fake=args.fake,
                warmup=0.0 if args.before else args.warmup,  # a combined run warmed up for its before
            )
        except TimeoutError as exc:
            print(f"the camera delivered no after snapshot: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
            print(f"cannot verify: {exc}", file=sys.stderr)
            return 2
        print(f"{label}after: {record['after_summary']}")
        print(f"{label}verdict: ok={record['ok']} missing={record['missing']} reason={record['reason']}")
        names = (BEFORE_FILE, AFTER_FILE, VERIFY_FILE, RESULT_FILE)
        print(f"files: {', '.join(str(folder / name) for name in names)}")
        return 0 if record["ok"] else 1
    finally:
        if args.fake:
            tracker.close()  # the physical singleton closes itself at exit


if __name__ == "__main__":
    sys.exit(main())
