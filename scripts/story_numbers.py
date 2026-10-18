"""Pull the story numbers out of runs/, skills/ and the RL provenance files (D8), read-only.

    python scripts/story_numbers.py [--runs DIR] [--skills DIR] [--root DIR] [--out PATH]
                                    [--markdown] [--strict]

D8 says to pull every number (candidates per task, pass rates, time to learn a skill, real success counts,
sandbox runs) from `runs/` instead of estimating. This script reads only files that exist, never a device,
a port, a key or the network, and attaches the source path to every number it reports:

- `runs/**/t-*/events.jsonl` (schemas/event.json, the orchestrator's per-task log) with `command.txt`,
  `router.json`, `scores.json`, `selected.json` and `repair/scores.json` beside it: per task the command,
  candidates written, candidates passing their trials, repair rounds (a second `simulating` batch), seconds
  from the first event to `executing` and to `saved`, the saved skill, the end stage and the `runner` label
  of every scored candidate (`local`, `daytona`, or `none` for a candidate that never ran).
- `runs/**/summary.json`: lists written for `agents.orchestrator --bench-demo` runs (`seconds_to_executing`,
  `passing_candidates`, `saved` per run) and the policy bring-up summaries under `runs/rl_deploy/`
  (`within_3cm`, `trials`, `over_speed_trials`, `driver`).
- `skills/*.json`, `skills/_starter/`, `skills/_backup/` and `runs/archive/*/` (schemas/skill.json): the
  `sim` and `real` counters of every saved skill.
- `rl/policies/handoff_best.json`, `rl/runs/baseline/dgo.json`, `rl/runs/parallel/result.json` and
  `web/static/rl/curves.json` (or `rl/policies/curves.json`) under `--root`: the selected policy's
  evaluation, steps and hash, the D-GO decision, the reward variants with their runner, the dashboard curves.
- `runs/**/tally.json` (plus `attempts.jsonl` beside it) written by `arm.replay --out`: replay attempts.
- `runs/setup/**/*.json` camera soak results and reader diagnostics; `runs/calibration/*.json` A6 and
  gripper reports; `runs/**/results.json` touch-marker results; `runs/hardware/*.json` inspections.

Unknown layouts are skipped with a note (`notes` in the JSON, printed last), never guessed. The run folders
record the scoring runner but not whether the arm, the cameras or the model were real, so a task's `scope`
says "not recorded" for those; docs/gates.md carries that part of the story.

Public functions: `collect(runs_dir, skills_dir, root) -> dict` (the whole report), `read_task(task_dir,
runs_dir) -> (dict, notes)`, `read_events(path) -> (events, notes)`, `task_totals(tasks) -> dict`,
`render_markdown(report) -> str`, `render_summary(report) -> str`, `write_report(report, path) -> Path` and
`main(argv) -> int`. Exit codes: 0 aggregated; 1 with `--strict` when any note was recorded; 2 on misuse
(missing runs folder, an existing `--out`). The JSON is always written: `--out` defaults to
`<runs>/story/<UTC stamp>.json`, a new file every run, and `--markdown` prints the tables instead of the
short summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = ROOT / "schemas"
STORY_DIR = "story"
RUNNERS = ("daytona", "local")  # a scored candidate with one of these labels ran in a sandbox or subprocess
RL_FILES = {
    "provenance": "rl/policies/handoff_best.json",
    "dgo": "rl/runs/baseline/dgo.json",
    "parallel": "rl/runs/parallel/result.json",
}
CURVES_FILES = ("web/static/rl/curves.json", "rl/policies/curves.json")
NOT_RECORDED = "not recorded"

_VALIDATORS: dict[str, jsonschema.Draft202012Validator] = {}


# ------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------


def _validator(name: str) -> jsonschema.Draft202012Validator:
    if name not in _VALIDATORS:
        schema = json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        _VALIDATORS[name] = jsonschema.Draft202012Validator(schema)
    return _VALIDATORS[name]


def _schema_error(name: str, data: Any) -> str | None:
    error = best_match(_validator(name).iter_errors(data))
    return None if error is None else error.message


def _posix(path: Path, *bases: Path) -> str:
    """`path` relative to the first base that contains it, in posix form; absolute when none does."""
    resolved = path.resolve()
    for base in bases:
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _group(task_dir: Path, runs_dir: Path) -> str:
    """The folder holding `task_dir`, relative to `runs_dir`; "" for a task directly under the runs folder."""
    parent = _posix(task_dir.parent, runs_dir)
    return "" if parent == "." else parent


def _in_story_dir(path: Path, runs_dir: Path) -> bool:
    try:
        return STORY_DIR in path.resolve().relative_to(runs_dir.resolve()).parts
    except ValueError:
        return False


def _try_json(path: Path, notes: list[str], label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        notes.append(f"{label}: unreadable JSON ({exc.__class__.__name__}: {exc})")
        return None


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """count, mean, median, min and max of `values` (one decimal), or None entries when there are none."""
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(float(t), UTC).isoformat(timespec="seconds")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------------------
# Tasks: runs/**/t-*/events.jsonl
# ------------------------------------------------------------------------------------------------------------


def read_events(path: Path, label: str | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """The valid events of one `events.jsonl` (schemas/event.json) in file order, plus a note per bad line."""
    label = label or path.as_posix()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{label}: unreadable ({exc})"]
    events: list[dict[str, Any]] = []
    notes: list[str] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            notes.append(f"{label}:{number}: not JSON ({exc})")
            continue
        problem = _schema_error("event", event)
        if problem is not None:
            notes.append(f"{label}:{number}: off schemas/event.json ({problem})")
            continue
        events.append(event)
    return events, notes


def find_task_dirs(runs_dir: Path) -> list[Path]:
    """Every `t-*` folder holding an `events.jsonl` under `runs_dir`, benches included, sorted by path."""
    found = []
    for path in runs_dir.rglob("events.jsonl"):
        if path.parent.name.startswith("t-") and not _in_story_dir(path, runs_dir):
            found.append(path.parent)
    return sorted(found, key=lambda p: _posix(p, runs_dir))


def _outcome(events: list[dict[str, Any]]) -> str:
    stages = [e["stage"] for e in events]
    for stage in ("saved", "queued", "not_saved", "stopped", "error"):
        if stage in stages:
            return stage
    if "done" in stages:
        done = next(e["detail"] for e in reversed(events) if e["stage"] == "done")
        return "done" if done.get("ok", True) is not False else "failed"
    return "empty" if not stages else "unfinished"


def read_task(task_dir: Path, runs_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """The story numbers of one `runs/.../t-*/` folder from its events and the JSON files beside them."""
    label = _posix(task_dir, runs_dir)
    task_id = task_dir.name
    events, notes = read_events(task_dir / "events.jsonl", f"{label}/events.jsonl")
    foreign = [e for e in events if e.get("task_id") not in (task_id, None)]
    if foreign:
        notes.append(f"{label}/events.jsonl: {len(foreign)} event(s) of another task skipped")
        events = [e for e in events if e.get("task_id") in (task_id, None)]
    sources = {"events": f"{label}/events.jsonl"}

    command = None
    if (task_dir / "command.txt").is_file():
        command = (task_dir / "command.txt").read_text(encoding="utf-8").strip()
        sources["command"] = f"{label}/command.txt"
    action = None
    if (task_dir / "router.json").is_file():
        router = _try_json(task_dir / "router.json", notes, f"{label}/router.json")
        if isinstance(router, Mapping) and isinstance(router.get("action"), str):
            action = router["action"]
            sources["router"] = f"{label}/router.json"

    stages = [str(e["stage"]) for e in events]
    t0 = float(events[0]["t"]) if events else None
    rounds: list[dict[str, Any]] = []
    written: list[int] = [0]
    current: dict[str, Any] | None = None
    for event in events:
        stage, detail = event["stage"], event["detail"]
        if stage == "writing_code":
            written[-1] += 1
        elif stage == "simulating":
            current = {"candidates": list(detail.get("candidates") or []), "trials": detail.get("trials")}
            current["scored"] = []
            rounds.append(current)
            written.append(0)
        elif stage == "candidate_scored":
            if current is None:
                notes.append(f"{label}/events.jsonl: candidate_scored before any simulating event")
                current = {"candidates": [], "trials": None, "scored": []}
                rounds.append(current)
                written.append(0)
            current["scored"].append(detail)
            problem = _schema_error(
                "score", {k: detail[k] for k in _validator("score").schema["properties"] if k in detail}
            )
            if problem is not None:
                notes.append(
                    f"{label}/events.jsonl: candidate {detail.get('candidate')} off schemas/score.json "
                    f"({problem})"
                )

    round_rows: list[dict[str, Any]] = []
    runners: Counter[str] = Counter()
    for index, entry in enumerate(rounds):
        scored = entry["scored"]
        labels = Counter(str(d.get("runner") or NOT_RECORDED) for d in scored)
        runners.update(labels)
        round_rows.append(
            {
                "round": index + 1,
                "written": written[index] if index < len(written) else 0,
                "simulated": entry["candidates"],
                "trials": entry["trials"],
                "scored": len(scored),
                "passing": sorted(
                    {int(d["candidate"]) for d in scored if d.get("passed") and "candidate" in d}
                ),
                "runners": dict(labels),
            }
        )
    final_scored = rounds[-1]["scored"] if rounds else []
    candidates = sorted(
        (
            {
                "candidate": _int_or_none(d.get("candidate")),
                "passed": bool(d.get("passed")),
                "successes": _int_or_none(d.get("successes")),
                "trials": _int_or_none(d.get("trials")),
                "runner": d.get("runner"),
                "fail_reason": d.get("fail_reason") or d.get("error"),
            }
            for d in final_scored
        ),
        key=lambda c: c["candidate"] if c["candidate"] is not None else -1,
    )

    def first_t(stage: str) -> float | None:
        return next((float(e["t"]) for e in events if e["stage"] == stage), None)

    executing_t, saved_t = first_t("executing"), first_t("saved")
    seconds_to_executing = round(executing_t - t0, 1) if executing_t is not None and t0 is not None else None
    seconds_to_saved = round(saved_t - t0, 1) if saved_t is not None and t0 is not None else None
    saved_skill = next((e["detail"].get("name") for e in events if e["stage"] == "saved"), None)
    selected = next(
        (e["detail"].get("candidate") for e in reversed(events) if e["stage"] == "selected"), None
    )

    if (task_dir / "scores.json").is_file():
        sources["scores"] = f"{label}/scores.json"
        data = _try_json(task_dir / "scores.json", notes, sources["scores"])
        if isinstance(data, list):
            in_file = sorted(
                {int(d["candidate"]) for d in data if isinstance(d, Mapping) and d.get("passed")}
            )
            first_round = round_rows[0]["passing"] if round_rows else []
            if in_file != first_round:
                notes.append(
                    f"{sources['scores']}: passing {in_file} differs from the events' round 1 {first_round}"
                )
        elif data is not None:
            notes.append(f"{sources['scores']}: expected a list of score lines")
    if (task_dir / "repair" / "scores.json").is_file():
        sources["repair_scores"] = f"{label}/repair/scores.json"
    if (task_dir / "selected.json").is_file():
        sources["selected"] = f"{label}/selected.json"

    scoring = "+".join(sorted(k for k in runners if k in RUNNERS))
    if not scoring:
        scoring = "none" if runners else "unscored"
    if action is not None:
        kind = action
    elif "planning" in stages:
        kind = "run"
    elif rounds:
        kind = "learn"
    else:
        kind = "other"
    task = {
        "task_id": task_id,
        "dir": label,
        "group": _group(task_dir, runs_dir),
        "command": command,
        "kind": kind,
        "started_at": _iso(t0),
        "stages": stages,
        "end_stage": stages[-1] if stages else None,
        "outcome": _outcome(events),
        "candidates_written": written[0] if written else 0,
        "candidates_written_total": sum(written),
        "candidates_rejected": stages.count("candidate_rejected"),
        "rounds": round_rows,
        "repair_rounds": max(len(round_rows) - 1, 0),
        "passing_candidates": round_rows[-1]["passing"] if round_rows else [],
        "candidates": candidates,
        "selected": selected,
        "saved_skill": saved_skill,
        "seconds_to_executing": seconds_to_executing,
        "seconds_to_saved": seconds_to_saved,
        "runners": dict(runners),
        "scope": {"scoring": scoring, "robot": NOT_RECORDED, "camera": NOT_RECORDED, "model": NOT_RECORDED},
        "sources": sources,
    }
    return task, notes


def task_totals(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and timing statistics over `tasks` (the dicts of `read_task`), naming the files behind them."""
    runners: Counter[str] = Counter()
    for task in tasks:
        runners.update(task["runners"])
    learned = [t for t in tasks if t["saved_skill"]]
    with_rounds = [t for t in tasks if t["rounds"]]
    return {
        "tasks": len(tasks),
        "by_kind": dict(Counter(t["kind"] for t in tasks)),
        "by_outcome": dict(Counter(t["outcome"] for t in tasks)),
        "tasks_with_candidates": len(with_rounds),
        "learned_skills": len(learned),
        "learned_skill_names": sorted({t["saved_skill"] for t in learned}),
        "candidates_written": sum(t["candidates_written"] for t in tasks),
        "candidates_written_total": sum(t["candidates_written_total"] for t in tasks),
        "candidates_passing": sum(len(t["passing_candidates"]) for t in tasks),
        "repair_rounds": sum(t["repair_rounds"] for t in tasks),
        "scored_runs_by_runner": dict(runners),
        "sandbox_or_local_runs": sum(n for k, n in runners.items() if k in RUNNERS),
        "seconds_to_executing": _stats(
            [t["seconds_to_executing"] for t in tasks if t["seconds_to_executing"] is not None]
        ),
        "time_to_learn_s": _stats(
            [t["seconds_to_saved"] for t in with_rounds if t["seconds_to_saved"] is not None]
        ),
        "derived_from": [t["sources"]["events"] for t in tasks],
    }


# ------------------------------------------------------------------------------------------------------------
# Other evidence under runs/
# ------------------------------------------------------------------------------------------------------------


def read_summaries(runs_dir: Path, notes: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bench summaries (lists with `seconds_to_executing`) and policy bring-up summaries (`within_3cm`)."""
    benches: list[dict[str, Any]] = []
    deploys: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("summary.json")):
        if _in_story_dir(path, runs_dir):
            continue
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if data is None:
            continue
        if (
            isinstance(data, list)
            and data
            and all(isinstance(r, Mapping) and "seconds_to_executing" in r for r in data)
        ):
            seconds = [
                float(r["seconds_to_executing"]) for r in data if r.get("seconds_to_executing") is not None
            ]
            benches.append(
                {
                    "file": label,
                    "runs": len(data),
                    "task_ids": [r.get("task_id") for r in data],
                    "stages": [r.get("stage") for r in data],
                    "seconds_to_executing": seconds,
                    "median_s": round(statistics.median(seconds), 1) if seconds else None,
                    "passing_candidates": [r.get("passing_candidates") for r in data],
                    "saved": sum(1 for r in data if r.get("saved")),
                }
            )
        elif isinstance(data, Mapping) and "within_3cm" in data and "trials" in data:
            deploys.append(
                {
                    "file": label,
                    "stage": data.get("stage"),
                    "trials": _int_or_none(data.get("trials")),
                    "within_3cm": _int_or_none(data.get("within_3cm")),
                    "over_speed_trials": _int_or_none(data.get("over_speed_trials")),
                    "pass": data.get("pass"),
                    "driver": data.get("driver"),
                    "hand_speed_max": data.get("hand_speed_max"),
                    "policy": data.get("policy"),
                    "policy_sha256": data.get("policy_sha256"),
                    "created": data.get("created"),
                }
            )
        else:
            notes.append(f"{label}: unknown summary layout, skipped")
    return benches, deploys


def read_replay_tallies(runs_dir: Path, notes: list[str]) -> list[dict[str, Any]]:
    """`tally.json` files of `arm.replay --out` (one row per label) with the modes of `attempts.jsonl`."""
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("tally.json")):
        if _in_story_dir(path, runs_dir):
            continue
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if data is None:
            continue
        if not (
            isinstance(data, Mapping) and all(isinstance(v, Mapping) and "runs" in v for v in data.values())
        ):
            notes.append(f"{label}: unknown tally layout, skipped")
            continue
        modes: Counter[str] = Counter()
        attempts = path.with_name("attempts.jsonl")
        if attempts.is_file():
            for line in attempts.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        modes[str(json.loads(line).get("mode", NOT_RECORDED))] += 1
                    except (ValueError, AttributeError):
                        notes.append(f"{_posix(attempts, runs_dir)}: a line is not a JSON object")
        for name, tally in data.items():
            rows.append(
                {
                    "file": label,
                    "label": name,
                    "runs": _int_or_none(tally.get("runs")),
                    "successes": _int_or_none(tally.get("successes")),
                    "four_of_five": tally.get("four_of_five"),
                    "last_t": tally.get("last_t"),
                    "modes": dict(modes),
                }
            )
    return rows


def read_skills(skills_dir: Path, runs_dir: Path, notes: list[str]) -> list[dict[str, Any]]:
    """Every valid skill file in the library, its starters and backups, and the archived learned skills."""
    places = [
        ("live", sorted(skills_dir.glob("*.json"))),
        ("starter", sorted((skills_dir / "_starter").glob("*.json"))),
        ("backup", sorted((skills_dir / "_backup").glob("*.json"))),
        ("archive", sorted((runs_dir / "archive").glob("*/*.json"))),
    ]
    rows: list[dict[str, Any]] = []
    for where, paths in places:
        for path in paths:
            label = _posix(path, skills_dir, runs_dir)
            data = _try_json(path, notes, label)
            if data is None:
                continue
            problem = _schema_error("skill", data)
            if problem is not None:
                notes.append(f"{label}: off schemas/skill.json ({problem}), skipped")
                continue
            rows.append(
                {
                    "name": data["name"],
                    "where": where,
                    "file": label,
                    "trusted": bool(data["trusted"]),
                    "sim": dict(data["sim"]),
                    "real": dict(data["real"]),
                    "created": data["created"],
                }
            )
    return rows


def read_camera(runs_dir: Path, notes: list[str]) -> list[dict[str, Any]]:
    """Camera soak results and per-feed reader diagnostics under runs/setup/ (other JSON there is ignored)."""
    rows: list[dict[str, Any]] = []
    setup = runs_dir / "setup"
    if not setup.is_dir():
        return rows
    feed_keys = (
        "name",
        "index",
        "camera_index",
        "frames",
        "failures",
        "late",
        "fps",
        "max_read_s",
        "max_interframe_s",
    )
    for path in sorted(setup.rglob("*.json")):
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if isinstance(data, Mapping) and "duration_requested_s" in data and "exit_code" in data:
            elapsed = _float_or_none(data.get("elapsed_s"))
            requested = _float_or_none(data.get("duration_requested_s"))
            completed = elapsed is not None and requested is not None and elapsed >= requested
            rows.append(
                {
                    "file": label,
                    "kind": "soak_result",
                    "exit_code": _int_or_none(data.get("exit_code")),
                    "requested_s": requested,
                    "elapsed_s": elapsed,
                    "completed": completed,
                    "passed": completed and data.get("exit_code") == 0,
                    "log": data.get("log"),
                }
            )
        elif isinstance(data, Mapping) and isinstance(data.get("feeds"), list):
            rows.append(
                {
                    "file": label,
                    "kind": "soak_result",
                    "requested_s": _float_or_none(data.get("requested_s")),
                    "elapsed_s": _float_or_none(data.get("elapsed_s")),
                    "completed": data.get("completed"),
                    "passed": data.get("passed"),
                    "fake": data.get("fake"),
                    "feeds": [
                        {k: f.get(k) for k in feed_keys if k in f}
                        for f in data["feeds"]
                        if isinstance(f, Mapping)
                    ],
                }
            )
        elif (
            isinstance(data, list)
            and data
            and all(isinstance(f, Mapping) and "frames" in f and "late" in f for f in data)
        ):
            rows.append(
                {
                    "file": label,
                    "kind": "reader_diagnostic",
                    "scope": data[0].get("scope"),
                    "feeds": [{k: f.get(k) for k in feed_keys if k in f} for f in data],
                }
            )
    return rows


def read_calibration(runs_dir: Path, notes: list[str]) -> list[dict[str, Any]]:
    """A6 checks and measurement files, gripper reports, touch-marker results, hardware inspections."""
    rows: list[dict[str, Any]] = []
    calibration = runs_dir / "calibration"
    for path in sorted(calibration.rglob("*.json")) if calibration.is_dir() else []:
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if data is None:
            continue
        if isinstance(data, Mapping) and "mean_error_m" in data and "max_error_m" in data:
            rows.append(
                {
                    "file": label,
                    "kind": "a6_check",
                    "measurement_kind": data.get("measurement_kind"),
                    "all_within_tolerance": data.get("all_within_tolerance"),
                    "tolerance_m": data.get("tolerance_m"),
                    "mean_error_m": _float_or_none(data.get("mean_error_m")),
                    "max_error_m": _float_or_none(data.get("max_error_m")),
                    "poses": len(data.get("poses") or []),
                }
            )
        elif (
            isinstance(data, Mapping)
            and "empty_closed_pct" in data
            and isinstance(data.get("objects"), Mapping)
        ):
            rows.append(
                {
                    "file": label,
                    "kind": "gripper",
                    "mode": data.get("mode"),
                    "jaw_open_mm": data.get("jaw_open_mm"),
                    "empty_closed_pct": data.get("empty_closed_pct"),
                    "held_min": {
                        name: (entry.get("held_min") if isinstance(entry, Mapping) else None)
                        for name, entry in data["objects"].items()
                    },
                }
            )
        elif isinstance(data, Mapping) and "measurement_kind" in data and isinstance(data.get("poses"), list):
            poses = [p for p in data["poses"] if isinstance(p, Mapping)]
            filled = sum(1 for p in poses if all(v is not None for v in (p.get("tip_xyz") or [None])))
            rows.append(
                {
                    "file": label,
                    "kind": "a6_measurements",
                    "measurement_kind": data.get("measurement_kind"),
                    "poses": len(poses),
                    "poses_filled": filled,
                }
            )
        else:
            notes.append(f"{label}: unknown calibration layout, skipped")
    for path in sorted(runs_dir.rglob("results.json")):
        if _in_story_dir(path, runs_dir):
            continue
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if isinstance(data, Mapping) and "worst_cm" in data and "passed" in data and "markers" in data:
            rows.append(
                {
                    "file": label,
                    "kind": "touch_markers",
                    "mode": data.get("mode"),
                    "passed": data.get("passed"),
                    "aborted": data.get("aborted"),
                    "chain": data.get("chain"),
                    "loose": data.get("loose"),
                    "limit_cm": _float_or_none(data.get("limit_cm")),
                    "worst_cm": _float_or_none(data.get("worst_cm")),
                    "markers": len(data.get("markers") or []),
                }
            )
    hardware = runs_dir / "hardware"
    for path in sorted(hardware.glob("*.json")) if hardware.is_dir() else []:
        label = _posix(path, runs_dir)
        data = _try_json(path, notes, label)
        if isinstance(data, Mapping) and isinstance(data.get("arms"), Mapping):
            arms = {}
            for role, arm in data["arms"].items():
                if not isinstance(arm, Mapping):
                    continue
                calib = arm.get("calibration") if isinstance(arm.get("calibration"), Mapping) else {}
                arms[role] = {
                    "port": arm.get("port"),
                    "lerobot_id": arm.get("lerobot_id"),
                    "calibration_matches": calib.get("matches_hardware"),
                    "calibration_sha256": calib.get("sha256"),
                }
            rows.append(
                {
                    "file": label,
                    "kind": "hardware_inspection",
                    "read_only": data.get("read_only"),
                    "arms": arms,
                }
            )
        elif data is not None:
            notes.append(f"{label}: unknown hardware report layout, skipped")
    return rows


# ------------------------------------------------------------------------------------------------------------
# RL provenance under --root
# ------------------------------------------------------------------------------------------------------------


def read_rl(root: Path, runs_dir: Path, notes: list[str]) -> dict[str, Any]:
    """Selected-policy provenance, the D-GO evaluation, the parallel variants and the dashboard curves."""
    rl: dict[str, Any] = {"provenance": None, "dgo": None, "parallel": None, "curves": None, "missing": []}
    files = {key: root / rel for key, rel in RL_FILES.items()}
    for key, path in files.items():
        if not path.is_file():
            rl["missing"].append(RL_FILES[key])
    curves_path = next((root / rel for rel in CURVES_FILES if (root / rel).is_file()), None)
    if curves_path is None:
        rl["missing"].append(CURVES_FILES[0])

    if files["provenance"].is_file():
        data = _try_json(files["provenance"], notes, RL_FILES["provenance"])
        if isinstance(data, Mapping):
            evaluation = data.get("evaluation") if isinstance(data.get("evaluation"), Mapping) else {}
            rl["provenance"] = {
                "file": RL_FILES["provenance"],
                "policy": data.get("policy"),
                "sha256": data.get("sha256"),
                "source_policy": data.get("source_policy"),
                "reward": data.get("reward"),
                "steps": _int_or_none(data.get("steps")),
                "requested_steps": _int_or_none(data.get("requested_steps")),
                "training_seed": data.get("training_seed"),
                "evaluation": {
                    k: evaluation.get(k)
                    for k in (
                        "evaluated_on",
                        "seed",
                        "episodes",
                        "successes",
                        "success_rate",
                        "falls",
                        "mean_steps_to_success",
                    )
                },
                "selection_reason": data.get("selection_reason"),
                "deployment": data.get("deployment"),
            }
        elif data is not None:
            notes.append(f"{RL_FILES['provenance']}: expected an object")
    if files["dgo"].is_file():
        data = _try_json(files["dgo"], notes, RL_FILES["dgo"])
        if isinstance(data, Mapping):
            rl["dgo"] = {
                "file": RL_FILES["dgo"],
                **{
                    k: data.get(k)
                    for k in (
                        "evaluated_on",
                        "command",
                        "policy",
                        "policy_sha256",
                        "seed",
                        "episodes",
                        "successes",
                        "success_rate",
                        "falls",
                        "threshold",
                        "decision",
                    )
                },
            }
        elif data is not None:
            notes.append(f"{RL_FILES['dgo']}: expected an object")
    if rl["provenance"] and rl["dgo"]:
        rl["hash_match"] = rl["provenance"]["sha256"] == rl["dgo"]["policy_sha256"]
    if files["parallel"].is_file():
        data = _try_json(files["parallel"], notes, RL_FILES["parallel"])
        if isinstance(data, Mapping) and isinstance(data.get("variants"), list):
            variants = []
            for v in data["variants"]:
                if not isinstance(v, Mapping):
                    continue
                summary = v.get("summary") if isinstance(v.get("summary"), Mapping) else {}
                variants.append(
                    {
                        "name": v.get("name"),
                        "hint": v.get("hint"),
                        "runner": v.get("runner"),
                        "steps": _int_or_none(summary.get("steps")),
                        "episodes": _int_or_none(summary.get("episodes")),
                        "successes": _int_or_none(summary.get("successes")),
                        "success_rate": _float_or_none(summary.get("success_rate")),
                        "chosen": bool(v.get("chosen")),
                        "error": v.get("error"),
                    }
                )
            rl["parallel"] = {
                "file": RL_FILES["parallel"],
                "chosen": next((v["name"] for v in variants if v["chosen"]), data.get("chosen")),
                "runners": dict(Counter(str(v["runner"]) for v in variants)),
                "variants": variants,
            }
        elif data is not None:
            notes.append(f"{RL_FILES['parallel']}: expected an object with a variants list")
    if curves_path is not None:
        label = _posix(curves_path, root)
        data = _try_json(curves_path, notes, label)
        if isinstance(data, Mapping) and isinstance(data.get("variants"), list):
            series = []
            for v in data["variants"]:
                if not isinstance(v, Mapping):
                    continue
                success = v.get("success") if isinstance(v.get("success"), list) else v.get("success_rate")
                success = success if isinstance(success, list) else []
                series.append(
                    {
                        "name": v.get("name"),
                        "points": len(v.get("steps") or []),
                        "last_step": (v.get("steps") or [None])[-1],
                        "final_success_rate": success[-1] if success else None,
                        "chosen": bool(v.get("chosen")),
                    }
                )
            rl["curves"] = {
                "file": label,
                "chosen": data.get("chosen"),
                "created": data.get("created"),
                "note": data.get("note"),
                "series": series,
            }
        elif data is not None:
            notes.append(f"{label}: expected an object with a variants list")
    return rl


# ------------------------------------------------------------------------------------------------------------
# The report
# ------------------------------------------------------------------------------------------------------------


def collect(runs_dir: Path, skills_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    """The whole report: tasks, totals, benches, skills, RL, replays, camera and calibration, plus notes."""
    notes: list[str] = []
    tasks = []
    for task_dir in find_task_dirs(runs_dir):
        task, task_notes = read_task(task_dir, runs_dir)
        tasks.append(task)
        notes.extend(task_notes)
    benches, deploys = read_summaries(runs_dir, notes)
    skills = read_skills(skills_dir, runs_dir, notes) if skills_dir.is_dir() else []
    if not skills_dir.is_dir():
        notes.append(f"{skills_dir.as_posix()}: skills folder missing, no skill rows")
    rl = read_rl(root, runs_dir, notes)
    rl["deploy"] = deploys
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs_dir": runs_dir.resolve().as_posix(),
        "skills_dir": skills_dir.resolve().as_posix(),
        "root": root.resolve().as_posix(),
        "tasks": tasks,
        "totals": task_totals(tasks),
        "benches": benches,
        "skills": skills,
        "rl": rl,
        "replays": read_replay_tallies(runs_dir, notes),
        "camera": read_camera(runs_dir, notes),
        "calibration": read_calibration(runs_dir, notes),
        "notes": notes,
    }


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    """Write `report` as JSON to a new file at `path` (parents created); an existing file is refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


# ------------------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:g}"
    elif isinstance(value, Mapping):
        text = ", ".join(f"{k} {_cell(v)}" for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_cell(v) for v in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _seconds(value: float | None) -> str | None:
    """A seconds cell with one decimal (9.0, not 9), so a timing column lines up; None stays empty."""
    return None if value is None else f"{value:.1f}"


def _runner_text(runners: Mapping[str, int]) -> str:
    if not runners:
        return "no candidate scored"
    return "scoring " + ", ".join(f"{k} {n}" for k, n in sorted(runners.items()))


def _feed_text(feeds: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for f in feeds:
        bits = [f"{f.get('name', '?')}: {f.get('frames', '?')} frames", f"{f.get('failures', '?')} failed"]
        bits.append(f"{f.get('late', '?')} late")
        if isinstance(f.get("fps"), (int, float)):
            bits.append(f"{f['fps']:g} fps")
        if isinstance(f.get("max_read_s"), (int, float)):
            bits.append(f"max read {f['max_read_s']:.3f} s")
        parts.append(", ".join(bits))
    return "; ".join(parts)


def measured_rows(report: Mapping[str, Any]) -> list[list[Any]]:
    """The rows of the generated 'Measured results' table: check, observed result, scope, source."""
    tasks = report["tasks"]
    totals = report["totals"]
    rows: list[list[Any]] = []
    mode_note = "robot/camera/model mode not recorded in runs/ (see docs/gates.md)"

    scored = [t for t in tasks if t["rounds"]]
    if scored:
        executed = [t for t in scored if t["seconds_to_executing"] is not None]
        seconds = [t["seconds_to_executing"] for t in executed]
        saved = [t for t in scored if t["saved_skill"]]
        groups = Counter(t["group"] for t in scored)
        text = (
            f"{len(executed)} of {len(scored)} candidate-scored tasks reached executing, "
            f"{len(saved)} saved a skill"
        )
        if seconds:
            seconds_text = ", ".join(f"{s:g}" for s in seconds)
            text += f"; seconds to executing {seconds_text} (median {statistics.median(seconds):g} s)"
        learn_s = totals["time_to_learn_s"]
        if learn_s["count"]:
            text += f"; first event to saved median {learn_s['median']:g} s"
        rows.append(
            [
                "Learn tasks",
                text,
                f"{_runner_text(totals['scored_runs_by_runner'])}; {mode_note}",
                "; ".join(f"{g + '/' if g else ''}t-*/events.jsonl ({n})" for g, n in sorted(groups.items())),
            ]
        )
        per_task = "; ".join(
            f"{t['task_id']}: {len(t['passing_candidates'])} of {t['rounds'][-1]['scored']} passed"
            + (f" ({', '.join(str(c) for c in t['passing_candidates'])})" if t["passing_candidates"] else "")
            for t in scored
        )
        rows.append(
            [
                "Candidate discrimination",
                f"{totals['candidates_passing']} of {totals['candidates_written_total']} written candidates "
                f"passed their trials; {per_task}",
                f"{_runner_text(totals['scored_runs_by_runner'])}; rejected before scoring: "
                f"{sum(t['candidates_rejected'] for t in scored)}",
                "; ".join(sorted({t["sources"].get("scores", t["sources"]["events"]) for t in scored})),
            ]
        )
        repaired = [t for t in scored if t["repair_rounds"]]
        if repaired:
            repair_runners: Counter[str] = Counter()
            for t in repaired:
                repair_runners.update(t["runners"])
            detail = "; ".join(
                f"{t['task_id']}: round 1 {len(t['rounds'][0]['passing'])} of {t['rounds'][0]['scored']}, "
                f"round 2 {len(t['rounds'][1]['passing'])} of {t['rounds'][1]['scored']}"
                + (f", saved {t['saved_skill']}" if t["saved_skill"] else f", {t['outcome']}")
                for t in repaired
            )
            rows.append(
                [
                    "Repair recovery",
                    f"{len(repaired)} task(s) ran a repair round: {detail}",
                    f"{_runner_text(repair_runners)}; {mode_note}",
                    "; ".join(t["sources"].get("repair_scores", t["sources"]["events"]) for t in repaired),
                ]
            )
    runs = [t for t in tasks if not t["rounds"] and "planning" in t["stages"]]
    if runs:
        seconds = [t["seconds_to_executing"] for t in runs if t["seconds_to_executing"] is not None]
        text = f"{len(seconds)} of {len(runs)} saved-skill runs reached executing"
        if seconds:
            seconds_text = ", ".join(f"{s:g}" for s in seconds)
            text += f"; seconds to executing {seconds_text} (median {statistics.median(seconds):g} s)"
        outcomes = Counter(t["outcome"] for t in runs)
        rows.append(
            [
                "Saved-skill runs",
                f"{text}; outcomes {_cell(dict(outcomes))}",
                mode_note,
                "; ".join(t["sources"]["events"] for t in runs),
            ]
        )
    for where in ("live", "starter", "backup", "archive"):
        skills = [s for s in report["skills"] if s["where"] == where]
        if not skills:
            continue
        text = "; ".join(
            f"{s['name']}: sim {s['sim']['successes']}/{s['sim']['trials']}, "
            f"real {s['real']['successes']} of {s['real']['runs']} runs"
            + (" (trusted)" if s["trusted"] else "")
            for s in skills
        )
        rows.append(
            [
                f"Saved skills ({where})",
                text,
                "sim counters from the scoring trials; real counters as the orchestrator bumped them, "
                "robot mode not recorded",
                "; ".join(s["file"] for s in skills),
            ]
        )
    rl = report["rl"]
    if rl.get("dgo"):
        d = rl["dgo"]
        prov = rl.get("provenance") or {}
        text = f"{d['successes']}/{d['episodes']} successes, {d['falls']} falls, seed {d['seed']}"
        if prov.get("steps") is not None:
            text += f"; {prov['steps']:,} training steps"
        if d.get("decision"):
            text += f"; decision: {d['decision']}"
        sources = [d["file"]] + ([prov["file"]] if prov else [])
        if "hash_match" in rl:
            sources.append(
                "policy hash " + ("matches" if rl["hash_match"] else "DIFFERS") + " between the two"
            )
        rows.append(
            [
                "Selected PPO policy",
                text,
                "simulation (rl.evaluate on the MuJoCo environment)",
                "; ".join(sources),
            ]
        )
    elif rl.get("provenance"):
        p = rl["provenance"]
        e = p["evaluation"]
        rows.append(
            [
                "Selected PPO policy",
                f"{e.get('successes')}/{e.get('episodes')} successes, {e.get('falls')} falls, "
                f"seed {e.get('seed')}; {p.get('steps')} training steps",
                "simulation",
                p["file"],
            ]
        )
    if rl.get("parallel"):
        p = rl["parallel"]
        variants = "; ".join(
            f"{v['name']}: {v['successes']}/{v['episodes']} at {v['steps']:,} steps"
            if v["steps"] is not None
            else f"{v['name']}: {v['error'] or 'no summary'}"
            for v in p["variants"]
        )
        rows.append(
            [
                "Reward variants trained in parallel",
                f"{len(p['variants'])} variants, chosen {p['chosen']}; {variants}",
                f"runner {_cell(p['runners'])}",
                p["file"],
            ]
        )
    if rl.get("curves"):
        c = rl["curves"]
        rows.append(
            [
                "Dashboard RL curves",
                f"{len(c['series'])} series, chosen {c['chosen']}; "
                + "; ".join(
                    f"{s['name']} {s['points']} points to step {s['last_step']}" for s in c["series"]
                ),
                c.get("note") or "simulation",
                c["file"],
            ]
        )
    for d in rl.get("deploy") or []:
        rows.append(
            [
                f"Policy bring-up ({d.get('stage')})",
                f"{d['within_3cm']} of {d['trials']} tries within 3 cm, "
                f"{d['over_speed_trials']} over the speed cap, pass {_cell(d.get('pass'))}",
                f"driver {d.get('driver')}, hand speed cap {d.get('hand_speed_max')} m/s",
                d["file"],
            ]
        )
    for r in report["replays"]:
        rows.append(
            [
                f"Replays of {r['label']}",
                f"{r['successes']} of {r['runs']} attempts ok; four of five {_cell(r['four_of_five'])}",
                f"attempt modes {_cell(r['modes']) or NOT_RECORDED}",
                r["file"],
            ]
        )
    for c in report["camera"]:
        if c["kind"] == "soak_result" and "feeds" in c:
            text = (
                f"requested {c.get('requested_s')} s, elapsed {c.get('elapsed_s')} s, "
                f"completed {_cell(c.get('completed'))}, passed {_cell(c.get('passed'))}; "
                f"{_feed_text(c['feeds'])}"
            )
            scope = "fake cameras" if c.get("fake") else "physical cameras"
        elif c["kind"] == "soak_result":
            text = (
                f"requested {c.get('requested_s')} s, elapsed {c.get('elapsed_s')} s, "
                f"exit code {c.get('exit_code')}, completed {_cell(c.get('completed'))}, "
                f"passed {_cell(c.get('passed'))}"
            )
            scope = "physical cameras (wrapper result)"
        else:
            text = _feed_text(c["feeds"])
            scope = c.get("scope") or "reader diagnostic"
        rows.append([f"Camera {c['kind'].replace('_', ' ')}", text, scope, c["file"]])
    for c in report["calibration"]:
        if c["kind"] == "a6_check":
            text = (
                f"{c['poses']} poses, mean error {_cell(c['mean_error_m'])} m, "
                f"max {_cell(c['max_error_m'])} m, all within tolerance {_cell(c['all_within_tolerance'])}"
            )
            scope = f"measurement kind {c.get('measurement_kind')}"
        elif c["kind"] == "a6_measurements":
            text = f"{c['poses_filled']} of {c['poses']} poses measured"
            scope = f"measurement kind {c.get('measurement_kind')}"
        elif c["kind"] == "gripper":
            text = (
                f"jaw open {c.get('jaw_open_mm')} mm, empty closed {c.get('empty_closed_pct')} %, "
                f"held_min {_cell(c['held_min'])}"
            )
            scope = f"mode {c.get('mode')}"
        elif c["kind"] == "touch_markers":
            text = (
                f"{c['markers']} markers, worst {_cell(c['worst_cm'])} cm against {_cell(c['limit_cm'])} cm, "
                f"passed {_cell(c['passed'])}"
            )
            scope = f"mode {c.get('mode')}" + (", aborted" if c.get("aborted") else "")
        else:
            text = "; ".join(
                f"{role}: {arm.get('port')} {arm.get('lerobot_id')}, "
                f"calibration matches {_cell(arm.get('calibration_matches'))}"
                for role, arm in c["arms"].items()
            )
            scope = "read-only inspection" if c.get("read_only") else "inspection"
        rows.append([f"Calibration {c['kind'].replace('_', ' ')}", text, scope, c["file"]])
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    """The Markdown report: the generated 'Measured results' table, then one table per source kind."""
    totals = report["totals"]
    parts = [
        f"# Story numbers, generated {report['generated_at']}",
        "",
        f"Runs folder `{report['runs_dir']}`, skills folder `{report['skills_dir']}`, "
        f"root `{report['root']}`. Every row names the file it came from; task, camera and calibration "
        "paths are relative to the runs folder, RL paths to the root. Nothing here was typed in by hand.",
        "",
        "## Measured results (generated)",
        "",
        _table(["Check", "Observed result", "Scope", "Source"], measured_rows(report)),
        "",
        "## Tasks",
        "",
        _table(
            [
                "Task",
                "Group",
                "Command",
                "Kind",
                "Outcome",
                "Written",
                "Passing",
                "Repair",
                "s to executing",
                "s to saved",
                "Runners",
                "Source",
            ],
            [
                [
                    t["task_id"],
                    t["group"],
                    t["command"],
                    t["kind"],
                    t["outcome"],
                    t["candidates_written"] or None,
                    t["passing_candidates"] if t["rounds"] else None,
                    t["repair_rounds"] if t["rounds"] else None,
                    _seconds(t["seconds_to_executing"]),
                    _seconds(t["seconds_to_saved"]),
                    t["runners"] or None,
                    t["sources"]["events"],
                ]
                for t in report["tasks"]
            ],
        ),
        "",
        "## Totals",
        "",
        _table(
            ["Metric", "Value", "Derived from"],
            [
                ["Tasks", totals["tasks"], f"{len(totals['derived_from'])} events.jsonl files"],
                ["By kind", totals["by_kind"], "router.json action, else the stages"],
                ["By outcome", totals["by_outcome"], "last stages of events.jsonl"],
                [
                    "Learned skills",
                    f"{totals['learned_skills']} ({_cell(totals['learned_skill_names'])})",
                    "saved events",
                ],
                [
                    "Candidates written (round 1 / all rounds)",
                    f"{totals['candidates_written']} / {totals['candidates_written_total']}",
                    "writing_code events",
                ],
                [
                    "Candidates passing (final rounds)",
                    totals["candidates_passing"],
                    "candidate_scored events",
                ],
                ["Repair rounds", totals["repair_rounds"], "extra simulating events"],
                [
                    "Scored runs by runner",
                    totals["scored_runs_by_runner"],
                    "runner field of candidate_scored",
                ],
                ["Seconds to executing", totals["seconds_to_executing"], "first event to executing"],
                [
                    "Time to learn (s)",
                    totals["time_to_learn_s"],
                    "first event to saved, candidate-scored tasks",
                ],
            ],
        ),
        "",
    ]
    if report["benches"]:
        parts += [
            "## Bench summaries",
            "",
            _table(
                ["File", "Runs", "Stages", "Seconds to executing", "Median", "Passing candidates", "Saved"],
                [
                    [
                        b["file"],
                        b["runs"],
                        b["stages"],
                        b["seconds_to_executing"],
                        b["median_s"],
                        b["passing_candidates"],
                        b["saved"],
                    ]
                    for b in report["benches"]
                ],
            ),
            "",
        ]
    if report["skills"]:
        parts += [
            "## Saved skills",
            "",
            _table(
                [
                    "Name",
                    "Where",
                    "Trusted",
                    "Sim successes/trials",
                    "Real successes/runs",
                    "Created",
                    "File",
                ],
                [
                    [
                        s["name"],
                        s["where"],
                        s["trusted"],
                        f"{s['sim']['successes']}/{s['sim']['trials']}",
                        f"{s['real']['successes']}/{s['real']['runs']}",
                        s["created"],
                        s["file"],
                    ]
                    for s in report["skills"]
                ],
            ),
            "",
        ]
    rl = report["rl"]
    rl_rows: list[list[Any]] = []
    if rl.get("provenance"):
        p = rl["provenance"]
        rl_rows.append(
            [
                "Selected policy",
                f"{p['policy']} from {p['source_policy']}, reward {p['reward']}, {p['steps']} steps, "
                f"seed {p['training_seed']}",
                p["file"],
            ]
        )
        rl_rows.append(["Provenance evaluation", p["evaluation"], p["file"]])
        rl_rows.append(["Deployment", p.get("deployment"), p["file"]])
    if rl.get("dgo"):
        d = rl["dgo"]
        rl_rows.append(
            [
                "D-GO evaluation",
                f"{d['successes']}/{d['episodes']}, falls {d['falls']}, seed {d['seed']}, "
                f"threshold {d.get('threshold')}: {d.get('decision')}",
                d["file"],
            ]
        )
    if "hash_match" in rl:
        rl_rows.append(
            ["Policy hash matches D-GO", rl["hash_match"], f"{RL_FILES['provenance']} vs {RL_FILES['dgo']}"]
        )
    for v in (rl.get("parallel") or {}).get("variants", []):
        rl_rows.append(
            [
                f"Variant {v['name']}",
                f"runner {v['runner']}, {v['successes']}/{v['episodes']} at {v['steps']} steps"
                + (", chosen" if v["chosen"] else ""),
                rl["parallel"]["file"],
            ]
        )
    if rl.get("curves"):
        rl_rows.append(
            [
                "Dashboard curves",
                f"chosen {rl['curves']['chosen']}, {len(rl['curves']['series'])} series, "
                f"created {rl['curves']['created']}",
                rl["curves"]["file"],
            ]
        )
    for d in rl.get("deploy") or []:
        rl_rows.append(
            [
                f"Bring-up {d.get('stage')}",
                f"{d['within_3cm']}/{d['trials']} within 3 cm, {d['over_speed_trials']} over speed, "
                f"driver {d.get('driver')}, pass {_cell(d.get('pass'))}",
                d["file"],
            ]
        )
    if rl.get("missing"):
        rl_rows.append(["Missing files", rl["missing"], "not present under the root"])
    if rl_rows:
        parts += ["## RL", "", _table(["Item", "Value", "Source"], rl_rows), ""]
    if report["replays"]:
        parts += [
            "## Replay tallies",
            "",
            _table(
                ["Label", "Runs", "Successes", "Four of five", "Modes", "File"],
                [
                    [r["label"], r["runs"], r["successes"], r["four_of_five"], r["modes"], r["file"]]
                    for r in report["replays"]
                ],
            ),
            "",
        ]
    if report["camera"]:
        parts += [
            "## Camera",
            "",
            _table(
                ["Kind", "Result", "File"],
                [
                    [
                        c["kind"],
                        {k: v for k, v in c.items() if k not in ("file", "kind", "feeds")}
                        | ({"feeds": _feed_text(c["feeds"])} if "feeds" in c else {}),
                        c["file"],
                    ]
                    for c in report["camera"]
                ],
            ),
            "",
        ]
    if report["calibration"]:
        parts += [
            "## Calibration",
            "",
            _table(
                ["Kind", "Result", "File"],
                [
                    [c["kind"], {k: v for k, v in c.items() if k not in ("file", "kind")}, c["file"]]
                    for c in report["calibration"]
                ],
            ),
            "",
        ]
    parts += ["## Notes", ""]
    parts += [f"- {note}" for note in report["notes"]] or ["- none: every file read matched a known layout"]
    return "\n".join(parts) + "\n"


def render_summary(report: Mapping[str, Any]) -> str:
    """A few plain lines: task counts, timing medians, the RL evaluation and the number of notes."""
    totals = report["totals"]
    lines = [
        f"tasks {totals['tasks']} ({_cell(totals['by_kind']) or 'none'}); "
        f"outcomes {_cell(totals['by_outcome']) or 'none'}",
        f"learned skills {totals['learned_skills']}; candidates passing {totals['candidates_passing']} of "
        f"{totals['candidates_written_total']} written; repair rounds {totals['repair_rounds']}; "
        f"scored runs by runner {_cell(totals['scored_runs_by_runner']) or 'none'}",
    ]
    s, learn = totals["seconds_to_executing"], totals["time_to_learn_s"]
    if s["count"]:
        lines.append(
            f"seconds to executing: median {s['median']:g} s, mean {s['mean']:g} s over {s['count']} tasks"
        )
    if learn["count"]:
        lines.append(
            f"time to learn: median {learn['median']:g} s, mean {learn['mean']:g} s "
            f"over {learn['count']} tasks"
        )
    lines.append(
        f"skills on file {len(report['skills'])}; bench summaries {len(report['benches'])}; "
        f"replay tallies {len(report['replays'])}"
    )
    dgo = report["rl"].get("dgo")
    if dgo:
        lines.append(
            f"RL: {dgo['successes']}/{dgo['episodes']} successes, {dgo['falls']} falls, "
            f"seed {dgo['seed']} ({dgo['file']})"
        )
    elif report["rl"].get("missing"):
        lines.append(f"RL: files missing under the root: {', '.join(report['rl']['missing'])}")
    lines.append(
        f"camera evidence {len(report['camera'])}; calibration evidence {len(report['calibration'])}; "
        f"notes {len(report['notes'])}"
    )
    lines.extend(f"note: {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------


def default_out(runs_dir: Path) -> Path:
    """`<runs>/story/<UTC stamp>.json`, suffixed `-2`, `-3`, ... when that second already has a report."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = runs_dir / STORY_DIR / f"{stamp}.json"
    n = 1
    while out.exists():
        n += 1
        out = runs_dir / STORY_DIR / f"{stamp}-{n}.json"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the flags, aggregate, write the JSON report and print the summary or the Markdown tables."""
    parser = argparse.ArgumentParser(
        prog="python scripts/story_numbers.py",
        description="Aggregate story numbers from runs/, skills/ and the RL provenance files, read-only.",
    )
    parser.add_argument(
        "--runs", type=Path, default=None, help="runs folder (default: VARA_RUNS_DIR or runs/)"
    )
    parser.add_argument(
        "--skills", type=Path, default=None, help="skills folder (default: VARA_SKILLS_DIR or skills/)"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="folder holding rl/policies, rl/runs and web/static/rl"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (default: <runs>/story/<UTC stamp>.json); never overwritten",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="print the Markdown tables instead of the short summary"
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when any file was skipped or malformed (see notes)"
    )
    args = parser.parse_args(argv)

    runs_dir = args.runs or Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")
    if not runs_dir.is_dir():
        parser.error(f"runs folder not found: {runs_dir}")
    if args.skills is not None and not args.skills.is_dir():
        parser.error(f"skills folder not found: {args.skills}")
    skills_dir = args.skills or Path(os.environ.get("VARA_SKILLS_DIR") or ROOT / "skills")
    if not args.root.is_dir():
        parser.error(f"root folder not found: {args.root}")
    out = args.out if args.out is not None else default_out(runs_dir)
    if out.exists():
        parser.error(f"--out exists, choose a new path: {out}")

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # candidate hints hold a pi; a cp1252 console must not crash the report
        reconfigure(encoding="utf-8", errors="replace")

    report = collect(runs_dir, skills_dir, args.root)
    try:
        write_report(report, out)
    except OSError as exc:
        parser.error(f"cannot write {out}: {exc}")
    print(render_markdown(report) if args.markdown else render_summary(report), end="")
    print(f"wrote {out}")
    return 1 if args.strict and report["notes"] else 0


if __name__ == "__main__":
    sys.exit(main())
