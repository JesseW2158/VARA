"""Rehearsal report (A15 steps 2 and 3): what every candidate did in each learn under a runs directory.

    python scripts/rehearsal_report.py [--runs DIR] [--task t-0001 ...] [--since WHEN] [--last N] [--out PATH]

Read-only and offline: it opens only what the orchestrator wrote under `runs/<task_id>/` (`command.txt`,
`events.jsonl`, `scores.json`, `selected.json`, `repair/scores.json`) and imports no orchestrator, model,
sandbox, arm or camera code. `--runs` (alias `--runs-dir`) defaults to `VARA_RUNS_DIR`, else the repo's
`runs/`, where `python -m agents.orchestrator --bench-demo` and the dashboard write.

Every `t-<n>` folder (in number order) gives one entry: the command, the first event's time, the end stage
(`done` with its `ok`, `queued`, `stopped`, `error`) with its reason, the seconds from the first event to
`executing` or `queued` (`arm_stage` says which; the demo budget as the event stream shows it), the saved
skill's name or the `not_saved` reasons and, for a learn (a folder with `scores.json`), one row per
candidate: number, hint, PASS or FAIL, successes/trials, collisions, runner label (`local`, a sandbox,
`none` when it never ran), whether it was selected, and `fail_reason`. A normal-mode repair round
(`repair/scores.json`, its candidates numbered from 1 again) gets rows of its own, and the selected
candidate is then one of those. Every event is checked against schemas/event.json and every score line
against schemas/score.json (the orchestrator's extra `runner`, `hint` and `trial_results` keys aside).

The cereal-box verdict of docs/tasks/A-arm.md A15 step 3 counts the first round's six candidates: 2 to 4
passing, `keep` the box where it is; 5 or 6, move it 3 cm `toward` the line between the objects and the
tray and re-tape; 0 or 1, move it 3 cm `away` and re-tape. A candidate that the simulation budget cut off
(`fail_reason` exactly `timeout`) says nothing about the box, so while such candidates could still move
the count across a band edge the verdict is `rerun`. A learn with other than six candidates (a routine)
gets `n/a`. Folders without `scores.json` (a saved skill's run, a queued command) are listed with their
stage only. The run folders do not record whether the arm was real or fake; the operator's backup note
(docs/demo.md) carries that.

`--task` keeps the named tasks, `--since` those whose first event is at or after an ISO date or datetime
(local time unless an offset is given), `--last N` the last N after the other filters. `--out` also writes
the report as JSON (`{"generated_at", "runs_dir", "filters", "tasks", "summary", "errors"}`) and never
overwrites: a path ending in `.json` is that new file (refused when it exists); any other path is a
directory, created as needed, that receives a new `rehearsal-<UTC stamp>.json`. Exit 0; 1 when a task
folder could not be read (its entry carries `error`) or the JSON could not be written; 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCORE_SCHEMA: dict[str, Any] = json.loads((ROOT / "schemas" / "score.json").read_text(encoding="utf-8"))
EVENT_SCHEMA: dict[str, Any] = json.loads((ROOT / "schemas" / "event.json").read_text(encoding="utf-8"))
SCORE_VALIDATOR = jsonschema.Draft202012Validator(SCORE_SCHEMA)
EVENT_VALIDATOR = jsonschema.Draft202012Validator(EVENT_SCHEMA)
TASK_DIR_RE = re.compile(r"^t-(\d+)$")
ARM_STAGES = ("executing", "queued")  # the first of these ends the "command to arm" budget
REASON_STAGES = ("queued", "stopped", "error")  # end events whose detail says why
EXTRA_SCORE_KEYS = ("runner", "hint", "trial_results")  # the orchestrator's additions to a score line
TIMEOUT_REASON = "timeout"  # the fail_reason of a candidate the simulation budget cut off (contract)
EXPECTED_CANDIDATES = 6
KEEP_BAND = (2, 4)  # A15 step 3: this many passing candidates of 6 means the cereal box stays
ACTIONS = ("keep", "toward", "away", "rerun", "n/a")
HINT_WIDTH = 48


# --- reading one task folder ------------------------------------------------------------------------------


def task_dirs(runs: Path) -> list[Path]:
    """The `t-<n>` folders directly under `runs`, in task-number order."""
    found: list[tuple[int, Path]] = []
    for entry in runs.iterdir():
        match = TASK_DIR_RE.match(entry.name)
        if match and entry.is_dir():
            found.append((int(match.group(1)), entry))
    return [path for _, path in sorted(found)]


def read_events(path: Path) -> list[dict[str, Any]]:
    """The events of an `events.jsonl`, each checked against schemas/event.json. A partial last line (a
    write in progress) is dropped; any other line that is not JSON or not an event raises ValueError."""
    lines = path.read_text(encoding="utf-8").split("\n")
    last = max((k for k, line in enumerate(lines) if line.strip()), default=-1)
    events: list[dict[str, Any]] = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            if k == last:
                break
            raise ValueError(f"{path.name} line {k + 1} is not JSON") from None
        error = best_match(EVENT_VALIDATOR.iter_errors(event))
        if error is not None:
            where = "/".join(str(part) for part in error.absolute_path) or "<root>"
            raise ValueError(
                f"{path.name} line {k + 1} is off schemas/event.json at {where}: {error.message}"
            )
        events.append(dict(event))
    return events


def read_scores(path: Path, label: str = "scores.json") -> list[dict[str, Any]]:
    """The score lines of a scores file (`label` names it in errors), each checked against
    schemas/score.json, in candidate order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{label} is not a list of score lines")
    scores: list[dict[str, Any]] = []
    for k, entry in enumerate(data, 1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} entry {k} is not an object")
        core = {key: value for key, value in entry.items() if key not in EXTRA_SCORE_KEYS}
        error = best_match(SCORE_VALIDATOR.iter_errors(core))
        if error is not None:
            where = "/".join(str(part) for part in error.absolute_path) or "<root>"
            raise ValueError(f"{label} entry {k} is off schemas/score.json at {where}: {error.message}")
        scores.append(dict(entry))
    return sorted(scores, key=lambda score: int(score["candidate"]))


def read_selected(folder: Path, events: Sequence[Mapping[str, Any]]) -> int | None:
    """The selected candidate from `selected.json`, else from the last `selected` event, else None."""
    path = folder / "selected.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        candidate = data.get("candidate") if isinstance(data, Mapping) else None
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ValueError(f"{path.name} has no integer candidate")
        return candidate
    chosen: int | None = None
    for event in events:
        if event.get("stage") == "selected":
            candidate = event["detail"].get("candidate")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                chosen = candidate
    return chosen


def verdict(passing: int, candidates: int, unscored: int = 0) -> dict[str, Any]:
    """A15 step 3 for `passing` of `candidates`, `unscored` of them cut off by the simulation budget:
    `keep` (2 to 4), `toward` (5 or 6), `away` (0 or 1), `rerun` while the unscored candidates could still
    move the count across a band edge, `n/a` unless there are six candidates."""
    low, high = KEEP_BAND
    summary = f"{passing} of {candidates} candidates pass"
    if unscored:
        summary += f", {unscored} cut off by the simulation budget"
    if candidates != EXPECTED_CANDIDATES:
        action = "n/a"
        text = f"no cereal-box verdict: A15's rule counts {EXPECTED_CANDIDATES} candidates"
    elif passing > high:
        action = "toward"
        text = "move the cereal box 3 cm toward the line between the objects and the tray, then re-tape"
    elif passing + unscored < low:
        action = "away"
        text = "move the cereal box 3 cm away from the line between the objects and the tray, then re-tape"
    elif passing >= low and passing + unscored <= high:
        action = "keep"
        text = "keep the cereal box where it is"
    else:
        action = "rerun"
        text = "inconclusive: a timed-out candidate says nothing about the cereal box; rerun before moving it"
    if action != "n/a":
        summary += f", target {low} to {high} of {EXPECTED_CANDIDATES}"
    return {
        "passing": passing,
        "candidates": candidates,
        "unscored": unscored,
        "action": action,
        "text": f"{text} ({summary})",
    }


def candidate_row(score: Mapping[str, Any], selected: int | None) -> dict[str, Any]:
    """One report row from a score line (plus the orchestrator's `hint` and `runner`)."""
    index = int(score["candidate"])
    reason = None if score.get("fail_reason") is None else str(score["fail_reason"])
    return {
        "candidate": index,
        "hint": str(score.get("hint") or ""),
        "passed": bool(score["passed"]),
        "successes": int(score["successes"]),
        "trials": int(score["trials"]),
        "collisions": int(score["collisions"]),
        "limit_hits": int(score["limit_hits"]),
        "fail_reason": reason,
        "unscored": not score["passed"] and reason == TIMEOUT_REASON,
        "runner": str(score.get("runner") or "unknown"),
        "selected": selected is not None and index == selected,
    }


def _reason(detail: Mapping[str, Any]) -> str | None:
    text = detail.get("reason") or detail.get("error")
    return str(text) if text else None


def load_task(folder: Path) -> dict[str, Any]:
    """The report entry of one task folder; a folder that cannot be read gets `error` instead of rows."""
    entry: dict[str, Any] = {
        "task_id": folder.name,
        "command": None,
        "started": None,
        "started_at": None,
        "stage": None,
        "ok": None,
        "reason_stage": None,
        "reason": None,
        "arm_stage": None,
        "seconds_to_arm": None,
        "learn": False,
        "selected": None,
        "selected_round": None,
        "saved": None,
        "not_saved": [],
        "candidates": [],
        "passing": [],
        "repair_candidates": [],
        "repair_passing": [],
        "verdict": None,
        "error": None,
    }
    try:
        command = folder / "command.txt"
        if command.is_file():
            entry["command"] = command.read_text(encoding="utf-8").strip()
        events_path = folder / "events.jsonl"
        events = read_events(events_path) if events_path.is_file() else []
        if events:
            first = float(events[0]["t"])
            entry["started"] = first
            entry["started_at"] = _iso(first)
            entry["stage"] = str(events[-1]["stage"])
            arm = next((e for e in events if e["stage"] in ARM_STAGES), None)
            if arm is not None:
                entry["arm_stage"] = arm["stage"]
                entry["seconds_to_arm"] = round(float(arm["t"]) - first, 1)
            for event in events:
                detail, stage = event["detail"], event["stage"]
                if stage == "saved":
                    entry["saved"] = detail.get("name")
                elif stage == "not_saved":
                    entry["not_saved"].append(_reason(detail) or "")
                elif stage in REASON_STAGES:
                    entry["reason_stage"], entry["reason"] = stage, _reason(detail)
                elif stage == "done":
                    entry["ok"] = detail.get("ok", True) is not False
                    if not entry["ok"] and entry["reason"] is None and _reason(detail):
                        entry["reason_stage"], entry["reason"] = "done", _reason(detail)
        scores_path = folder / "scores.json"
        if scores_path.is_file():
            entry["learn"] = True
            selected = read_selected(folder, events)
            repair_path = folder / "repair" / "scores.json"
            repair = read_scores(repair_path, "repair/scores.json") if repair_path.is_file() else None
            # the repair round runs only when the first round had no winner, so its pick is a repair row
            rows = [candidate_row(score, None if repair else selected) for score in read_scores(scores_path)]
            entry["candidates"] = rows
            entry["passing"] = [row["candidate"] for row in rows if row["passed"]]
            if repair is not None:
                repair_rows = [candidate_row(score, selected) for score in repair]
                entry["repair_candidates"] = repair_rows
                entry["repair_passing"] = [row["candidate"] for row in repair_rows if row["passed"]]
            entry["selected"] = selected
            if selected is not None:
                entry["selected_round"] = "repair" if repair is not None else "first"
            unscored = sum(1 for row in rows if row["unscored"])
            entry["verdict"] = verdict(len(entry["passing"]), len(rows), unscored)
    except (OSError, ValueError, TypeError) as exc:  # JSONDecodeError and UnicodeDecodeError included
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


# --- the report ---------------------------------------------------------------------------------------------


def select_tasks(
    entries: Iterable[Mapping[str, Any]],
    *,
    task_ids: Sequence[str] = (),
    since: float | None = None,
    last: int | None = None,
) -> list[dict[str, Any]]:
    """`--task`, `--since` (epoch seconds, against the first event) and `--last N`, in that order."""
    kept = [dict(entry) for entry in entries]
    if task_ids:
        wanted = set(task_ids)
        kept = [entry for entry in kept if entry["task_id"] in wanted]
    if since is not None:
        kept = [entry for entry in kept if entry["started"] is not None and entry["started"] >= since]
    if last is not None:
        kept = kept[-last:] if last > 0 else []
    return kept


def summarize(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`{"learns", "verdicts": {action: count}}` over the learns that carry a verdict (ACTIONS order)."""
    actions = [task["verdict"]["action"] for task in tasks if task["verdict"]]
    return {
        "learns": len(actions),
        "verdicts": {action: actions.count(action) for action in ACTIONS if action in actions},
    }


def build_report(
    runs: Path,
    *,
    task_ids: Sequence[str] = (),
    since: float | None = None,
    last: int | None = None,
) -> dict[str, Any]:
    """Every task folder under `runs`, loaded, filtered and numbered, with the report's metadata."""
    entries = select_tasks(
        (load_task(folder) for folder in task_dirs(runs)), task_ids=task_ids, since=since, last=last
    )
    tasks = [{"run": k, **entry} for k, entry in enumerate(entries, 1)]
    return {
        "generated_at": _iso(datetime.now().timestamp()),
        "runs_dir": str(runs.resolve()),
        "filters": {"task": list(task_ids), "since": None if since is None else _iso(since), "last": last},
        "tasks": tasks,
        "summary": summarize(tasks),
        "errors": sum(1 for task in tasks if task["error"]),
    }


def _table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The candidate rows as aligned console lines, header first; the last column is not padded."""
    cells = [("cand", "result", "ok/trials", "coll", "runner", "sel", "hint", "fail_reason")]
    for row in rows:
        cells.append(
            (
                str(row["candidate"]),
                "PASS" if row["passed"] else "FAIL",
                f"{row['successes']}/{row['trials']}",
                str(row["collisions"]),
                row["runner"],
                "yes" if row["selected"] else "",
                _shorten(row["hint"], HINT_WIDTH),
                row["fail_reason"] or "-",
            )
        )
    widths = [max(len(line[i]) for line in cells) for i in range(len(cells[0]) - 1)]
    return [
        (
            "  "
            + "  ".join(cell.ljust(width) for cell, width in zip(line[:-1], widths, strict=True))
            + "  "
            + line[-1]
        ).rstrip()
        for line in cells
    ]


def format_task(entry: Mapping[str, Any]) -> str:
    """The console block of one task: a header line, then the candidate rows and the verdict of a learn."""
    command = json.dumps(entry["command"]) if entry["command"] else "(no command.txt)"
    head = [f"{entry['task_id']}  {command}"]
    stage = entry["stage"] or "no events"
    if entry["stage"] == "done" and entry["ok"] is not None:
        stage += " ok" if entry["ok"] else " not ok"
    if entry["reason"]:  # named by its stage only when that is not the end stage itself ("queued" in "done")
        own = entry["reason_stage"] in ("done", entry["stage"])
        why = entry["reason"] if own else f"{entry['reason_stage']}: {entry['reason']}"
        stage += f" ({why})"
    head.append(stage)
    if entry["seconds_to_arm"] is not None:
        head.append(f"{entry['seconds_to_arm']:.1f} s to {entry['arm_stage']}")
    if entry["selected"] is not None:
        where = " (repair round)" if entry["selected_round"] == "repair" else ""
        head.append(f"selected {entry['selected']}{where}")
    if entry["saved"]:
        head.append(f"saved {entry['saved']}")
    head.extend(f"not saved: {reason}" for reason in entry["not_saved"])
    lines = ["  |  ".join(head)]
    if entry["error"]:
        lines.append(f"  ERROR {entry['error']}")
    elif not entry["learn"]:
        lines.append("  no scores.json: not a learn, listed with its stage only")
    else:
        lines.extend(_table(entry["candidates"]))
        result = entry["verdict"]
        summary = f"passing {result['passing']} of {result['candidates']}: {entry['passing']}"
        lines.append(f"  {summary}  ->  {result['text']}")
        if entry["repair_candidates"]:
            lines.append("  repair round (repair/scores.json):")
            lines.extend(_table(entry["repair_candidates"]))
            count = len(entry["repair_candidates"])
            lines.append(
                f"  repair passing {len(entry['repair_passing'])} of {count}: {entry['repair_passing']}"
            )
    return "\n".join(lines)


def render(report: Mapping[str, Any]) -> str:
    """The whole console report: a header, one block per task, then the verdict counts."""
    tasks = report["tasks"]
    learns = sum(1 for task in tasks if task["learn"])
    header = (
        f"runs: {report['runs_dir']}  ({len(tasks)} task folder(s): {learns} learn(s), "
        f"{len(tasks) - learns} other, {report['errors']} unreadable)"
    )
    if not tasks:
        return header + "\nno task folders (t-<n>) matched"
    blocks = [header, *(format_task(task) for task in tasks)]
    summary = report["summary"]
    if summary["learns"]:
        counts = ", ".join(f"{action} {n}" for action, n in summary["verdicts"].items())
        blocks.append(f"verdicts over {summary['learns']} learn(s): {counts}")
    return "\n\n".join(blocks)


def write_report(out: Path, report: Mapping[str, Any]) -> Path:
    """Write `report` as JSON to a new file and return its path: `out` itself when it ends in `.json`
    (refused when it exists), else `rehearsal-<UTC stamp>.json` inside the directory `out`, which is
    created as needed (a numeric suffix keeps two writes in one second apart)."""
    target = Path(out)
    if target.is_dir() or target.suffix.lower() != ".json":
        if target.exists() and not target.is_dir():
            raise FileExistsError(
                f"{target} exists and is not a directory; give a new .json path or a folder"
            )
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = target / f"rehearsal-{stamp}.json"
        n = 1
        while path.exists():
            n += 1
            path = target / f"rehearsal-{stamp}-{n}.json"
    else:
        if target.exists():
            raise FileExistsError(f"{target} exists; a report is never overwritten, choose a new path")
        target.parent.mkdir(parents=True, exist_ok=True)
        path = target
    with path.open("x", encoding="utf-8", newline="\n") as handle:  # "x": never replaces a file
        handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def parse_since(text: str) -> float:
    """`--since` as epoch seconds: an ISO date or datetime, local time when it carries no offset."""
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(
            f"--since expects an ISO date or datetime like 2026-10-18 or 2026-10-18T09:00, got {text!r}"
        ) from None
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.timestamp()


# --- helpers and CLI ----------------------------------------------------------------------------------------


def _iso(epoch: float) -> str:
    """Epoch seconds as local ISO 8601 with the UTC offset, to the second."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def _utf8_stdout() -> None:
    """The hints carry a pi; a cp1252 console must print the report, not crash on it."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI of the module docstring. Exit 0; 1 for an unreadable task folder or an unwritable `--out`;
    2 for bad arguments."""
    parser = argparse.ArgumentParser(
        prog="python scripts/rehearsal_report.py",
        description="per-candidate results and the A15 cereal-box verdict of every learn under a runs dir",
    )
    parser.add_argument(
        "--runs",
        "--runs-dir",
        dest="runs",
        default=None,
        metavar="DIR",
        help="runs directory (default: VARA_RUNS_DIR, else runs/ in the repo)",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="ID",
        help="only this task id, e.g. t-0003 (repeatable)",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="WHEN",
        help="only tasks whose first event is at or after this ISO date or datetime (local time unless an "
        "offset is given)",
    )
    parser.add_argument(
        "--last", type=int, default=None, metavar="N", help="only the last N tasks after the other filters"
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="also write the report as JSON: a new .json file, or a folder (created as needed) that "
        "receives a new rehearsal-<stamp>.json",
    )
    args = parser.parse_args(argv)
    runs = Path(args.runs) if args.runs else _env_dir("VARA_RUNS_DIR", ROOT / "runs")
    if not runs.is_dir():
        parser.error(f"runs directory not found: {runs}")
    since = None
    if args.since is not None:
        try:
            since = parse_since(args.since)
        except ValueError as exc:
            parser.error(str(exc))
    if args.last is not None and args.last < 1:
        parser.error("--last needs a positive count")
    for task_id in args.task:
        if not TASK_DIR_RE.match(task_id):
            parser.error(f"--task expects an id like t-0003, got {task_id!r}")
    _utf8_stdout()
    report = build_report(runs, task_ids=args.task, since=since, last=args.last)
    print(render(report))
    code = 1 if report["errors"] else 0
    if args.out is not None:
        try:
            path = write_report(Path(args.out), report)
        except OSError as exc:
            print(f"could not write the JSON report: {exc}", file=sys.stderr)
            return 1
        print(f"JSON report: {path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
