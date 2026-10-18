"""scripts/rehearsal_report.py on a synthetic runs tree: rows, verdicts, filters, JSON output, failures.

The tree mimics what the orchestrator writes (`command.txt`, `events.jsonl`, `scores.json`,
`selected.json`, `repair/scores.json`), each event and score line checked against schemas/ as it is built;
nothing here imports the orchestrator or touches a device.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts import rehearsal_report as rr

ROOT = Path(__file__).resolve().parents[1]
EVENT_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "event.json").read_text(encoding="utf-8"))
)
SCORE_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "score.json").read_text(encoding="utf-8"))
)
HINTS = [
    "Nearest object first, default parameters.",
    "Farthest object first.",
    "Set `transit_z` to the tallest obstacle plus 0.08 m on every pick and place.",
    "Route around obstacles: rise, move out to x = 0.32 m, then sideways.",
    "Slow down to 0.08 m/s and use pitch -π/4 for targets beyond 0.30 m from the base.",
    "Fill tray slots far side first, and retry a failed grasp once.",
]
FAIL_REASONS = {
    1: "trial 1: gripper hit cereal_box",
    2: "trial 1: held pill_bottle hit juice_box",
    3: "trial 2: spec not satisfied",
    4: "trial 1: timeout",  # the candidate's own trial ran out of time: a real failure, not the budget
    5: "trial 1: gripper hit cereal_box",
    6: "rejected: robot.run_skill is not allowed",
}
T0 = 1_792_266_000.0
TASKS = ["t-0001", "t-0002", "t-0003", "t-0004", "t-0005", "t-0006"]


def event(t: float, task_id: str, stage: str, detail: dict[str, Any]) -> dict[str, Any]:
    data = {"t": t, "task_id": task_id, "stage": stage, "detail": detail}
    EVENT_VALIDATOR.validate(data)  # the synthetic tree holds only what the orchestrator could write
    return data


def score_line(index: int, passed: bool, *, budget_timeout: bool = False) -> dict[str, Any]:
    """A score as the orchestrator writes it: schemas/score.json plus runner, hint and trial_results.
    `budget_timeout` is agents.orchestrator._timeout_score: the simulation budget cut the candidate off."""
    if budget_timeout:
        core: dict[str, Any] = {
            "candidate": index,
            "passed": False,
            "successes": 0,
            "trials": 5,
            "nominal_success": False,
            "collisions": 0,
            "limit_hits": 0,
            "ik_failures": 0,
            "mean_sim_s": None,
            "fail_reason": "timeout",
            "traj": None,
            "error": "timeout",
        }
        runner, trials = "none", []
    else:
        runner = "none" if index == 6 and not passed else "local"
        error = None
        if not passed and index == 4:
            error = "timeout"
        elif not passed and index == 6:
            error = FAIL_REASONS[6]
        core = {
            "candidate": index,
            "passed": passed,
            "successes": 5 if passed or index == 2 else 0,
            "trials": 5,
            "nominal_success": passed,
            "collisions": 0 if passed else {1: 36, 2: 15, 5: 17}.get(index, 0),
            "limit_hits": 0,
            "ik_failures": 0,
            "mean_sim_s": 10.0 + index if passed else None,
            "fail_reason": None if passed else FAIL_REASONS[index],
            "traj": f"cand_{index}/traj.json" if runner == "local" else None,
            "error": error,
        }
        trials = [passed] * 5
    SCORE_VALIDATOR.validate(core)
    return {**core, "runner": runner, "hint": HINTS[index - 1], "trial_results": trials}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2) + "\n")


def write_events(path: Path, events: list[dict[str, Any]], *, tail: str = "") -> None:
    write_text(path, "".join(json.dumps(e) + "\n" for e in events) + tail)


def scored_round(
    task_id: str, t0: float, passing: set[int], candidates: int, timeouts: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The events and score lines of one round: writing_code x N, simulating, candidate_scored x N."""
    events = [
        event(t0, task_id, "writing_code", {"candidate": i, "hint": HINTS[i - 1]})
        for i in range(1, candidates + 1)
    ]
    events.append(event(t0 + 0.1, task_id, "simulating", {"candidates": list(range(1, candidates + 1))}))
    scores = [score_line(i, i in passing, budget_timeout=i in timeouts) for i in range(1, candidates + 1)]
    events += [event(t0 + 2 + k, task_id, "candidate_scored", score) for k, score in enumerate(scores)]
    return events, scores


def make_learn(
    runs: Path,
    task_id: str,
    t0: float,
    passing: set[int],
    *,
    command: str = "clear the table",
    saved: bool = True,
    candidates: int = 6,
    timeouts: frozenset[int] = frozenset(),
    repair: set[int] | None = None,
) -> Path:
    """A learn folder: N scored candidates (the `timeouts` cut off by the budget), the lowest passing one
    selected and executed, then saved. With `repair` (the repair round's passing set; the first round
    must then have none) a second round goes to repair/scores.json and the pick comes from it."""
    folder = runs / task_id
    folder.mkdir(parents=True)
    write_text(folder / "command.txt", command + "\n")
    events = [
        event(t0, task_id, "routing", {"text": command}),
        event(t0 + 0.1, task_id, "spec_ready", {"name": "clear_table"}),
    ]
    round_events, scores = scored_round(task_id, t0 + 0.2, passing, candidates, set(timeouts))
    events += round_events
    write_json(folder / "scores.json", scores[::-1])  # any order on disk: the report sorts by candidate
    pool, t_sel = passing, t0 + 9.0
    if repair is not None:
        assert not passing, "the orchestrator repairs only when no candidate passed"
        round_events, scores = scored_round(task_id, t0 + 9.0, repair, candidates, set())
        events += round_events
        write_json(folder / "repair" / "scores.json", scores)
        pool, t_sel = repair, t0 + 19.0
    selected = min(pool) if pool else None
    if selected is None:
        events.append(event(t_sel, task_id, "queued", {"task": command, "reason": "no candidate passed"}))
        events.append(event(t_sel + 0.1, task_id, "done", {"ok": False, "queued": True}))
    else:
        detail = {"candidate": selected, "hint": HINTS[selected - 1], "scores": scores}
        write_json(folder / "selected.json", detail)
        events.append(event(t_sel, task_id, "selected", detail))
        events.append(event(t_sel + 0.1, task_id, "validating", {"candidate": selected, "segments": 5}))
        events.append(event(t_sel + 0.6, task_id, "executing", {"candidate": selected, "segments": 5}))
        events.append(event(t_sel + 1.0, task_id, "verifying", {"english": "Every object ends in the tray."}))
        if saved:
            events.append(event(t_sel + 1.2, task_id, "saved", {"name": "clear_table"}))
            events.append(event(t_sel + 1.3, task_id, "done", {"ok": True, "skill": "clear_table"}))
        else:
            events.append(event(t_sel + 1.2, task_id, "not_saved", {"reason": "spec not satisfied"}))
            events.append(event(t_sel + 1.3, task_id, "done", {"ok": False, "skill": "clear_table"}))
    write_events(folder / "events.jsonl", events)
    return folder


def make_run(
    runs: Path, task_id: str, t0: float, command: str = "put the juice box in the tray", *, ok: bool = True
) -> Path:
    """A saved skill's run: no scores.json; a failed verification ends `done` with ok false and a reason."""
    folder = runs / task_id
    folder.mkdir(parents=True)
    write_text(folder / "command.txt", command + "\n")
    done = {"ok": ok, "skill": "juice_box_to_tray", "reason": None if ok else "juice_box not in tray"}
    write_events(
        folder / "events.jsonl",
        [
            event(t0, task_id, "routing", {"text": command}),
            event(t0 + 0.5, task_id, "planning", {"skill": "juice_box_to_tray", "args": {}}),
            event(t0 + 1.0, task_id, "validating", {"segments": 4}),
            event(t0 + 1.4, task_id, "executing", {"segments": 4}),
            event(t0 + 2.0, task_id, "verifying", {"english": "The juice box ends inside the tray."}),
            event(t0 + 2.2, task_id, "done", done),
        ],
    )
    return folder


def make_queued(runs: Path, task_id: str, t0: float, command: str = "fold my laundry") -> Path:
    folder = runs / task_id
    folder.mkdir(parents=True)
    write_text(folder / "command.txt", command + "\n")
    write_events(
        folder / "events.jsonl",
        [
            event(t0, task_id, "routing", {"text": command}),
            event(t0 + 0.4, task_id, "queued", {"task": command, "reason": "needs cloth handling"}),
            event(t0 + 0.5, task_id, "done", {"ok": False, "queued": True, "reason": "needs cloth handling"}),
        ],
    )
    return folder


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Six task folders plus the folders and files a real runs/ also holds, which must be ignored."""
    runs = tmp_path / "runs"
    make_learn(runs, "t-0001", T0, {3, 4})
    make_learn(runs, "t-0002", T0 + 100, {1, 2, 3, 4, 5, 6})
    make_learn(runs, "t-0003", T0 + 200, {3}, saved=False)
    make_learn(runs, "t-0004", T0 + 300, set())
    make_run(runs, "t-0005", T0 + 400)
    make_queued(runs, "t-0006", T0 + 500)
    write_json(runs / "archive" / "20261017-170420" / "clear_table.json", {"name": "clear_table"})
    (runs / "e2e_fake" / "20261017-210000" / "skills").mkdir(parents=True)
    (runs / "demo-video").mkdir()
    write_text(runs / "t-0099", "a file, not a task folder\n")
    return runs


# --- rows, verdicts, stages -------------------------------------------------------------------------------


def test_report_rows_stages_and_verdicts(tree: Path):
    report = rr.build_report(tree)
    tasks = report["tasks"]
    assert [t["task_id"] for t in tasks] == TASKS
    assert [t["run"] for t in tasks] == [1, 2, 3, 4, 5, 6]
    assert report["errors"] == 0 and report["runs_dir"] == str(tree.resolve())
    assert report["filters"] == {"task": [], "since": None, "last": None}
    assert report["summary"] == {"learns": 4, "verdicts": {"keep": 1, "toward": 1, "away": 2}}

    keep = tasks[0]
    assert keep["learn"] is True and keep["command"] == "clear the table"
    assert keep["stage"] == "done" and keep["ok"] is True and keep["reason"] is None
    assert keep["arm_stage"] == "executing" and keep["seconds_to_arm"] == pytest.approx(9.6)
    assert keep["selected"] == 3 and keep["selected_round"] == "first"
    assert keep["saved"] == "clear_table" and keep["not_saved"] == []
    assert keep["passing"] == [3, 4] and keep["repair_candidates"] == [] and keep["repair_passing"] == []
    assert keep["verdict"] == {
        "passing": 2,
        "candidates": 6,
        "unscored": 0,
        "action": "keep",
        "text": "keep the cereal box where it is (2 of 6 candidates pass, target 2 to 4 of 6)",
    }
    assert keep["started"] == T0 and keep["started_at"].startswith("2026-10-17T")
    rows = keep["candidates"]
    assert [row["candidate"] for row in rows] == [1, 2, 3, 4, 5, 6]  # sorted although written reversed
    assert rows[0] == {
        "candidate": 1,
        "hint": HINTS[0],
        "passed": False,
        "successes": 0,
        "trials": 5,
        "collisions": 36,
        "limit_hits": 0,
        "fail_reason": "trial 1: gripper hit cereal_box",
        "unscored": False,
        "runner": "local",
        "selected": False,
    }
    assert rows[1]["successes"] == 5 and rows[1]["passed"] is False  # 5/5 but collisions: rejected
    assert rows[2]["passed"] is True and rows[2]["selected"] is True and rows[2]["fail_reason"] is None
    assert rows[3]["passed"] is True and rows[3]["selected"] is False
    assert rows[5]["runner"] == "none" and rows[5]["fail_reason"].startswith("rejected: ")
    assert not any(row["unscored"] for row in rows)  # "trial 1: timeout" is the candidate's own failure

    toward = tasks[1]
    assert toward["passing"] == [1, 2, 3, 4, 5, 6] and toward["selected"] == 1
    assert toward["verdict"]["action"] == "toward"

    away = tasks[2]
    assert away["passing"] == [3] and away["verdict"]["action"] == "away"
    assert away["saved"] is None and away["not_saved"] == ["spec not satisfied"] and away["ok"] is False
    assert away["reason"] is None  # the done event carries none

    queued_learn = tasks[3]
    assert queued_learn["learn"] is True and queued_learn["selected"] is None
    assert queued_learn["selected_round"] is None
    assert queued_learn["passing"] == [] and queued_learn["verdict"]["action"] == "away"
    assert queued_learn["stage"] == "done" and queued_learn["ok"] is False
    assert (queued_learn["reason_stage"], queued_learn["reason"]) == ("queued", "no candidate passed")
    assert queued_learn["arm_stage"] == "queued" and queued_learn["seconds_to_arm"] == pytest.approx(9.0)

    run = tasks[4]
    assert run["learn"] is False and run["candidates"] == [] and run["verdict"] is None
    assert run["arm_stage"] == "executing" and run["seconds_to_arm"] == pytest.approx(1.4)
    assert run["stage"] == "done" and run["ok"] is True and run["reason"] is None

    queued = tasks[5]
    assert queued["learn"] is False and queued["seconds_to_arm"] == pytest.approx(0.4)
    assert queued["arm_stage"] == "queued" and queued["reason"] == "needs cloth handling"
    assert queued["command"] == "fold my laundry" and queued["ok"] is False


def test_render_shows_rows_fail_reasons_and_verdicts(tree: Path):
    text = rr.render(rr.build_report(tree))
    assert text.startswith(f"runs: {tree.resolve()}  (6 task folder(s): 4 learn(s), 2 other, 0 unreadable)")
    assert (
        't-0001  "clear the table"  |  done ok  |  9.6 s to executing  |  selected 3  |  saved clear_table'
        in text
    )
    assert (
        't-0004  "clear the table"  |  done not ok (queued: no candidate passed)  |  9.0 s to queued' in text
    )
    assert (
        't-0006  "fold my laundry"  |  done not ok (queued: needs cloth handling)  |  0.4 s to queued' in text
    )
    assert "s to the arm" not in text  # a queued task never reached the arm
    assert "trial 1: gripper hit cereal_box" in text
    assert "trial 1: held pill_bottle hit juice_box" in text
    assert "rejected: robot.run_skill is not allowed" in text
    assert "  passing 2 of 6: [3, 4]  ->  keep the cereal box where it is (2 of 6 candidates pass" in text
    assert "passing 6 of 6: [1, 2, 3, 4, 5, 6]  ->  move the cereal box 3 cm toward the line" in text
    assert "passing 1 of 6: [3]  ->  move the cereal box 3 cm away from the line" in text
    assert "passing 0 of 6: []  ->  move the cereal box 3 cm away" in text
    assert "not saved: spec not satisfied" in text
    assert "no scores.json: not a learn, listed with its stage only" in text
    assert text.endswith("verdicts over 4 learn(s): keep 1, toward 1, away 2")
    block = next(part for part in text.split("\n\n") if part.startswith("t-0001"))
    lines = block.splitlines()
    assert lines[1].split()[:6] == ["cand", "result", "ok/trials", "coll", "runner", "sel"]
    assert lines[2].split()[:5] == ["1", "FAIL", "0/5", "36", "local"]
    assert lines[4].split()[:6] == ["3", "PASS", "5/5", "0", "local", "yes"]
    assert "π" in lines[6] and "..." in lines[6]  # the long hint is shortened, not dropped


@pytest.mark.parametrize(
    "passing, unscored, action",
    [
        (0, 0, "away"),
        (1, 0, "away"),
        (2, 0, "keep"),
        (3, 0, "keep"),
        (4, 0, "keep"),
        (5, 0, "toward"),
        (6, 0, "toward"),
        (0, 1, "away"),  # even if the cut-off one had passed: 1 is still under 2
        (1, 1, "rerun"),  # could have been 2
        (0, 4, "rerun"),
        (2, 2, "keep"),  # at most 4 even if both had passed
        (2, 3, "rerun"),  # could have been 5
        (4, 1, "rerun"),
        (5, 1, "toward"),  # already above the band
    ],
)
def test_verdict_band(passing: int, unscored: int, action: str):
    result = rr.verdict(passing, 6, unscored)
    assert result["action"] == action and result["passing"] == passing and result["candidates"] == 6
    assert result["unscored"] == unscored
    assert ("cut off by the simulation budget" in result["text"]) is (unscored > 0)
    assert result["text"].endswith("target 2 to 4 of 6)")
    if action == "rerun":
        assert result["text"].startswith(
            "inconclusive: a timed-out candidate says nothing about the cereal box"
        )


def test_verdict_is_not_applicable_without_six_candidates():
    for passing, candidates in ((1, 1), (0, 1), (3, 3), (7, 7)):
        result = rr.verdict(passing, candidates)
        assert result["action"] == "n/a"
        counts = f"({passing} of {candidates} candidates pass)"
        assert result["text"] == f"no cereal-box verdict: A15's rule counts 6 candidates {counts}"


def test_budget_timeouts_make_the_verdict_rerun(tmp_path: Path):
    runs = tmp_path / "runs"
    make_learn(runs, "t-0001", T0, {3}, timeouts=frozenset({1, 2, 4}))
    entry = rr.build_report(runs)["tasks"][0]
    rows = {row["candidate"]: row for row in entry["candidates"]}
    assert [n for n, row in rows.items() if row["unscored"]] == [1, 2, 4]
    assert rows[1]["runner"] == "none" and rows[1]["fail_reason"] == "timeout"
    assert entry["passing"] == [3]
    assert entry["verdict"]["action"] == "rerun" and entry["verdict"]["unscored"] == 3
    text = rr.format_task(entry)
    assert "passing 1 of 6: [3]  ->  inconclusive:" in text
    assert "(1 of 6 candidates pass, 3 cut off by the simulation budget, target 2 to 4 of 6)" in text


def test_routine_learn_has_no_box_verdict(tmp_path: Path):
    runs = tmp_path / "runs"
    make_learn(runs, "t-0001", T0, {1}, command="good morning", candidates=1)
    report = rr.build_report(runs)
    entry = report["tasks"][0]
    assert entry["passing"] == [1] and entry["selected"] == 1 and entry["verdict"]["action"] == "n/a"
    assert report["summary"] == {"learns": 1, "verdicts": {"n/a": 1}}
    assert "no cereal-box verdict" in rr.format_task(entry)


def test_repair_round_rows_and_selection(tmp_path: Path):
    runs = tmp_path / "runs"
    make_learn(runs, "t-0001", T0, set(), repair={2, 5})
    entry = rr.build_report(runs)["tasks"][0]
    assert entry["passing"] == [] and entry["repair_passing"] == [2, 5]
    assert entry["selected"] == 2 and entry["selected_round"] == "repair"
    assert not any(row["selected"] for row in entry["candidates"])  # the first round's 2 failed
    assert [row["selected"] for row in entry["repair_candidates"]] == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert entry["verdict"]["action"] == "away"  # the box verdict counts the first round's six
    assert entry["seconds_to_arm"] == pytest.approx(19.6)
    text = rr.format_task(entry)
    assert "selected 2 (repair round)  |  saved clear_table" in text
    assert "  repair round (repair/scores.json):" in text
    assert text.endswith("  repair passing 2 of 6: [2, 5]")
    repair_rows = text.split("repair round (repair/scores.json):\n")[1].splitlines()
    assert repair_rows[2].split()[:6] == ["2", "PASS", "5/5", "0", "local", "yes"]

    scores = json.loads((runs / "t-0001" / "repair" / "scores.json").read_text(encoding="utf-8"))
    del scores[1]["collisions"]
    write_json(runs / "t-0001" / "repair" / "scores.json", scores)
    error = rr.build_report(runs)["tasks"][0]["error"]
    assert error.startswith("ValueError: repair/scores.json entry 2 is off schemas/score.json")


def test_failed_run_and_stopped_task_carry_their_reason(tmp_path: Path):
    runs = tmp_path / "runs"
    make_run(runs, "t-0001", T0, ok=False)
    folder = runs / "t-0002"
    write_text(folder / "command.txt", "clear the table\n")
    write_events(
        folder / "events.jsonl",
        [
            event(T0 + 10, "t-0002", "routing", {"text": "clear the table"}),
            event(T0 + 11, "t-0002", "stopped", {"reason": "Esc on the dashboard", "stage": "simulating"}),
        ],
    )
    tasks = rr.build_report(runs)["tasks"]
    assert (tasks[0]["reason_stage"], tasks[0]["reason"]) == ("done", "juice_box not in tray")
    assert (tasks[1]["reason_stage"], tasks[1]["reason"]) == ("stopped", "Esc on the dashboard")
    assert tasks[1]["seconds_to_arm"] is None and tasks[1]["arm_stage"] is None
    assert "done not ok (juice_box not in tray)  |  1.4 s to executing" in rr.format_task(tasks[0])
    assert rr.format_task(tasks[1]).startswith(
        't-0002  "clear the table"  |  stopped (Esc on the dashboard)\n'
    )


# --- filters ----------------------------------------------------------------------------------------------


def test_filters_task_since_and_last(tree: Path):
    ids = lambda report: [t["task_id"] for t in report["tasks"]]  # noqa: E731
    assert ids(rr.build_report(tree, task_ids=["t-0002", "t-0004"])) == ["t-0002", "t-0004"]
    assert ids(rr.build_report(tree, since=T0 + 150)) == ["t-0003", "t-0004", "t-0005", "t-0006"]
    assert ids(rr.build_report(tree, since=T0 + 200)) == ["t-0003", "t-0004", "t-0005", "t-0006"]
    assert ids(rr.build_report(tree, last=2)) == ["t-0005", "t-0006"]
    assert ids(rr.build_report(tree, task_ids=["t-0001", "t-0002"], last=1)) == ["t-0002"]
    report = rr.build_report(tree, since=T0 + 150, last=1)
    assert ids(report) == ["t-0006"] and report["tasks"][0]["run"] == 1
    assert report["filters"]["last"] == 1 and report["filters"]["since"].startswith("2026-10-17T")
    assert report["summary"] == {"learns": 0, "verdicts": {}}
    assert ids(rr.build_report(tree, task_ids=["t-0042"])) == []


def test_parse_since_accepts_dates_and_offsets():
    assert rr.parse_since("2026-10-17") == datetime(2026, 10, 17).astimezone().timestamp()
    assert rr.parse_since("2026-10-17T09:30") == datetime(2026, 10, 17, 9, 30).astimezone().timestamp()
    aware = datetime(2026, 10, 17, 13, 30, tzinfo=timezone.utc)
    assert rr.parse_since(aware.isoformat()) == aware.timestamp()
    with pytest.raises(ValueError, match="--since"):
        rr.parse_since("yesterday")


# --- CLI, JSON output, failures -----------------------------------------------------------------------------


def test_cli_prints_and_writes_a_new_json_file(tree: Path, tmp_path: Path, capsys):
    out = tmp_path / "evidence" / "nested" / "rehearsal.json"
    assert rr.main(["--runs", str(tree), "--task", "t-0001", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "trial 1: gripper hit cereal_box" in captured.out
    assert captured.out.rstrip().endswith(f"JSON report: {out}")
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    data = json.loads(raw)
    assert list(data) == ["generated_at", "runs_dir", "filters", "tasks", "summary", "errors"]
    assert [t["task_id"] for t in data["tasks"]] == ["t-0001"]
    assert data["tasks"][0]["passing"] == [3, 4] and data["tasks"][0]["verdict"]["action"] == "keep"
    assert data["filters"]["task"] == ["t-0001"] and "π" in data["tasks"][0]["candidates"][4]["hint"]
    assert data["summary"] == {"learns": 1, "verdicts": {"keep": 1}} and data["errors"] == 0

    # evidence is never overwritten
    assert rr.main(["--runs", str(tree), "--out", str(out)]) == 1
    assert "exists" in capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8")) == data

    # an existing folder receives a stamped file, a new one per call
    folder = tmp_path / "reports"
    folder.mkdir()
    assert rr.main(["--runs", str(tree), "--out", str(folder)]) == 0
    assert rr.main(["--runs", str(tree), "--out", str(folder)]) == 0
    capsys.readouterr()
    names = sorted(p.name for p in folder.iterdir())
    assert len(names) == 2 and all(n.startswith("rehearsal-") and n.endswith(".json") for n in names)

    # a path that does not end in .json is a folder, created as needed
    fresh = tmp_path / "new" / "rehearsals"
    assert rr.main(["--runs", str(tree), "--out", str(fresh)]) == 0
    written = capsys.readouterr().out.rstrip().splitlines()[-1].removeprefix("JSON report: ")
    assert Path(written).parent == fresh and fresh.is_dir()
    assert [p.name for p in fresh.iterdir()] == [Path(written).name]

    # an existing file that is not a .json report is never replaced
    plain = tmp_path / "notes.txt"
    plain.write_text("keep me\n", encoding="utf-8")
    assert rr.main(["--runs", str(tree), "--out", str(plain)]) == 1
    assert "is not a directory" in capsys.readouterr().err
    assert plain.read_text(encoding="utf-8") == "keep me\n"


def test_runs_dir_alias(tree: Path, capsys):
    assert rr.main(["--runs-dir", str(tree), "--task", "t-0005"]) == 0
    out = capsys.readouterr().out
    assert 't-0005  "put the juice box in the tray"  |  done ok  |  1.4 s to executing' in out
    assert "verdicts over" not in out  # no learn among the selected tasks


def test_unreadable_task_folders_are_reported_and_fail_the_exit_code(tree: Path, capsys):
    bad_list = make_run(tree, "t-0007", T0 + 600)
    write_json(bad_list / "scores.json", {"not": "a list"})
    off_schema = make_learn(tree, "t-0008", T0 + 700, {3})
    scores = json.loads((off_schema / "scores.json").read_text(encoding="utf-8"))
    del scores[0]["trials"]
    write_json(off_schema / "scores.json", scores)
    bad_events = make_queued(tree, "t-0009", T0 + 800)
    good = json.dumps(event(1.0, "t-0009", "routing", {"text": "fold my laundry"}))
    write_text(bad_events / "events.jsonl", good + '\nnot json\n{"x": 1}\n')

    assert rr.main(["--runs", str(tree)]) == 1
    out = capsys.readouterr().out
    report = rr.build_report(tree)
    by_id = {t["task_id"]: t for t in report["tasks"]}
    assert report["errors"] == 3
    assert by_id["t-0007"]["error"] == "ValueError: scores.json is not a list of score lines"
    assert "scores.json entry 1 is off schemas/score.json" in by_id["t-0008"]["error"]  # candidate 6
    assert "trials" in by_id["t-0008"]["error"]
    assert by_id["t-0009"]["error"] == "ValueError: events.jsonl line 2 is not JSON"
    assert by_id["t-0007"]["candidates"] == [] and by_id["t-0008"]["candidates"] == []
    assert by_id["t-0001"]["passing"] == [3, 4]  # the good folders are still reported
    assert out.count("  ERROR ") == 3 and "9 task folder(s)" in out and "3 unreadable" in out


def test_events_are_checked_against_the_schema_and_a_partial_last_line_is_tolerated(tmp_path: Path):
    runs = tmp_path / "runs"
    folder = make_queued(runs, "t-0001", T0)
    events = rr.read_events(folder / "events.jsonl")
    write_events(folder / "events.jsonl", events, tail='{"t": 2.0, "stage": "do')  # a write in progress
    assert [e["stage"] for e in rr.read_events(folder / "events.jsonl")] == ["routing", "queued", "done"]
    entry = rr.load_task(folder)
    assert entry["error"] is None and entry["stage"] == "done"

    for bad, where in (
        ({"t": 1.0, "task_id": "t-0001", "detail": {}}, "<root>"),  # no stage
        ({"t": 1.0, "task_id": "t-0001", "stage": "exploded", "detail": {}}, "stage"),
        ({"t": "late", "task_id": "t-0001", "stage": "done", "detail": {}}, "t"),
        ({"t": 1.0, "task_id": "t-0001", "stage": "done", "detail": {}, "extra": 1}, "<root>"),
    ):
        write_events(folder / "events.jsonl", [*events, bad])
        error = rr.load_task(folder)["error"]
        assert error.startswith(f"ValueError: events.jsonl line 4 is off schemas/event.json at {where}: "), (
            bad
        )


def test_folder_without_events_or_command(tmp_path: Path):
    runs = tmp_path / "runs"
    (runs / "t-0003").mkdir(parents=True)
    report = rr.build_report(runs)
    entry = report["tasks"][0]
    assert entry["command"] is None and entry["stage"] is None and entry["started"] is None
    assert entry["learn"] is False and entry["error"] is None
    text = rr.render(report)
    assert "t-0003  (no command.txt)  |  no events" in text
    assert [t["task_id"] for t in rr.build_report(runs, since=T0)["tasks"]] == []  # undatable: filtered out
    assert rr.render(rr.build_report(runs, since=T0)).endswith("no task folders (t-<n>) matched")


def test_cli_misuse_exits_2_and_help_exits_0(tree: Path, tmp_path: Path, capsys):
    for argv in (
        ["--runs", str(tmp_path / "missing")],
        ["--runs", str(tree), "--last", "0"],
        ["--runs", str(tree), "--last", "-3"],
        ["--runs", str(tree), "--last", "two"],
        ["--runs", str(tree), "--since", "nonsense"],
        ["--runs", str(tree), "--task", "0001"],
        ["--runs", str(tree), "--bogus"],
        ["--runs"],
    ):
        with pytest.raises(SystemExit) as info:
            rr.main(argv)
        assert info.value.code == 2, argv
        assert "usage:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        rr.main(["--help"])
    assert info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--since" in help_text and "--runs" in help_text and "--runs-dir" in help_text


def test_default_runs_dir_comes_from_the_environment(tree: Path, monkeypatch, capsys):
    monkeypatch.setenv("VARA_RUNS_DIR", str(tree))
    assert rr.main([]) == 0
    assert "t-0001" in capsys.readouterr().out
    monkeypatch.setenv("VARA_RUNS_DIR", str(tree / "nowhere"))
    with pytest.raises(SystemExit) as info:
        rr.main([])
    assert info.value.code == 2
