"""scripts/story_numbers.py pulls the story numbers from a synthetic runs tree, offline and read-only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import story_numbers

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SCRIPT = ROOT / "scripts" / "story_numbers.py"
T0 = 1792267623.7
CANDIDATES = [1, 2, 3, 4, 5, 6]
HINTS = [
    "Nearest object first, default parameters.",
    "Farthest object first.",
    "Set `transit_z` to the tallest obstacle plus 0.08 m.",
    "Route around obstacles.",
    "Slow down to 0.08 m/s and use pitch -π/4 beyond 0.30 m.",
    "Retry a failed grasp once.",
]


def _event(t: float, task_id: str, stage: str, detail: dict) -> dict:
    return {"t": t, "task_id": task_id, "stage": stage, "detail": detail}


def _score(candidate: int, passed: bool, runner: str) -> dict:
    """A scored detail as the orchestrator records it: the score line plus runner, hint and trial_results."""
    line = json.loads((FIXTURES / "score_line.json").read_text(encoding="utf-8"))
    line.update(
        {
            "candidate": candidate,
            "passed": passed,
            "nominal_success": passed,
            "successes": 5 if passed else 0,
            "collisions": 0 if passed else 5,
            "fail_reason": None if passed else "trial 1: gripper hit cereal_box",
            "traj": f"cand_{candidate}/traj.json",
        }
    )
    return {**line, "runner": runner, "hint": HINTS[candidate - 1], "trial_results": [passed] * 5}


def _scored_event(t: float, task_id: str, score: dict) -> dict:
    event = json.loads((FIXTURES / "event_candidate_scored.json").read_text(encoding="utf-8"))
    event.update({"t": t, "task_id": task_id, "detail": score})
    return event


def _write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


def write_learn(
    runs: Path,
    task_id: str,
    *,
    passing: list[int],
    repair_passing: list[int] | None = None,
    saved: str = "juice_box_to_tray",
    command: str = "put the juice box in the tray",
    runner: str = "local",
    exec_s: float = 8.6,
    saved_s: float = 9.0,
) -> Path:
    """A learn task folder as the orchestrator leaves it: events, command, router, scores, selected."""
    task = runs / task_id
    task.mkdir(parents=True)
    (task / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(task / "router.json", {"action": "learn", "spec": {"name": saved}})
    events = [
        _event(T0, task_id, "routing", {"text": command}),
        _event(T0 + 0.1, task_id, "spec_ready", {"name": saved}),
    ]
    events += [
        _event(T0 + 0.2, task_id, "writing_code", {"candidate": i, "hint": HINTS[i - 1]}) for i in CANDIDATES
    ]
    events.append(_event(T0 + 0.5, task_id, "simulating", {"candidates": CANDIDATES, "trials": 5}))
    scores = [_score(i, i in passing, runner) for i in CANDIDATES]
    events += [_scored_event(T0 + 5 + 0.1 * i, task_id, s) for i, s in enumerate(scores)]
    _write_json(task / "scores.json", scores)
    final = passing
    t = T0 + 6.0
    if repair_passing is not None:
        events += [
            _event(t, task_id, "writing_code", {"candidate": i, "hint": HINTS[i - 1]}) for i in CANDIDATES
        ]
        events.append(_event(t + 0.5, task_id, "simulating", {"candidates": CANDIDATES, "trials": 5}))
        repair = [_score(i, i in repair_passing, runner) for i in CANDIDATES]
        events += [_scored_event(t + 5 + 0.1 * i, task_id, s) for i, s in enumerate(repair)]
        _write_json(task / "repair" / "scores.json", repair)
        final = repair_passing
        t += 6.0
    if final:
        winner = min(final)
        selected = {"candidate": winner, "hint": HINTS[winner - 1], "scores": []}
        events.append(_event(t, task_id, "selected", selected))
        _write_json(task / "selected.json", selected)
        events.append(_event(t + 0.1, task_id, "validating", {"candidate": winner, "segments": 5}))
        events.append(_event(T0 + exec_s, task_id, "executing", {"candidate": winner, "segments": 5}))
        events.append(_event(T0 + exec_s + 0.2, task_id, "verifying", {"english": "It ends in the tray."}))
        events.append(_event(T0 + saved_s, task_id, "saved", {"name": saved}))
        events.append(_event(T0 + saved_s + 0.1, task_id, "done", {"ok": True, "skill": saved}))
    else:
        reason = "no candidate passed simulation"
        events.append(_event(t, task_id, "queued", {"task": command, "reason": reason}))
        events.append(_event(t + 0.1, task_id, "done", {"ok": False, "queued": True, "reason": reason}))
    _write_jsonl(task / "events.jsonl", events)
    return task


def write_run(runs: Path, task_id: str, *, skill: str = "hand_over", exec_s: float = 1.2) -> Path:
    """A saved-skill run: routing -> planning -> validating -> executing -> verifying -> done."""
    task = runs / task_id
    task.mkdir(parents=True)
    (task / "command.txt").write_text("hand me my pills\n", encoding="utf-8")
    _write_json(task / "router.json", {"action": "run", "skill": skill, "args": {"object": "pill_bottle"}})
    events = [
        _event(T0, task_id, "routing", {"text": "hand me my pills"}),
        _event(T0 + 0.3, task_id, "planning", {"skill": skill, "args": {"object": "pill_bottle"}}),
        _event(T0 + 1.0, task_id, "validating", {"skill": skill, "segments": 3}),
        _event(T0 + exec_s, task_id, "executing", {"skill": skill, "segments": 3}),
        _event(T0 + exec_s + 2.0, task_id, "verifying", {"english": "The user takes it."}),
        _event(T0 + exec_s + 2.5, task_id, "done", {"ok": True, "skill": skill}),
    ]
    _write_jsonl(task / "events.jsonl", events)
    return task


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """runs/, skills/ and a root with every layout the script reads, plus two deliberately broken files."""
    runs, skills, root = tmp_path / "runs", tmp_path / "skills", tmp_path / "root"
    write_learn(runs, "t-0001", passing=[3, 4, 5, 6])
    write_learn(
        runs,
        "t-0002",
        passing=[],
        repair_passing=CANDIDATES,
        saved="clear_table",
        command="clear the table",
        exec_s=20.3,
        saved_s=21.0,
    )
    write_learn(runs, "t-0003", passing=[], command="clear the table")
    write_run(runs, "t-0004")
    bench = runs / "verification" / "c7-bench-20261017-194847"
    write_learn(bench / "runs", "t-0001", passing=[3, 4], saved="clear_table", command="clear the table")
    final = {
        "t": T0 + 9.1,
        "task_id": "t-0001",
        "stage": "done",
        "detail": {"ok": True, "skill": "clear_table"},
    }
    _write_json(
        bench / "summary.json",
        [
            {
                "run": 1,
                "task_id": "t-0001",
                "stage": "executing",
                "seconds_to_executing": 8.6,
                "passing_candidates": [3, 4],
                "saved": True,
                "final_event": final,
            },
            {
                "run": 2,
                "task_id": "t-0002",
                "stage": "executing",
                "seconds_to_executing": 10.2,
                "passing_candidates": [3, 4],
                "saved": True,
                "final_event": final,
            },
        ],
    )
    bad = runs / "t-0005"
    bad.mkdir()
    bad_lines = [
        json.dumps(_event(T0, "t-0005", "routing", {"text": "fold my laundry"})),
        "not json",
        json.dumps({"t": T0 + 1, "task_id": "t-0005", "stage": "bogus", "detail": {}}),
        json.dumps(_event(T0 + 2, "t-0005", "queued", {"task": "fold my laundry", "reason": "unsupported"})),
        json.dumps(_event(T0 + 2.1, "t-0005", "done", {"ok": False, "queued": True})),
    ]
    (bad / "events.jsonl").write_text("\n".join(bad_lines) + "\n", encoding="utf-8")
    _write_json(runs / "verification" / "odd" / "summary.json", {"weird": 1})

    starter = json.loads((ROOT / "skills" / "_starter" / "hand_over.json").read_text(encoding="utf-8"))
    _write_json(skills / "hand_over.json", starter)
    _write_json(skills / "_starter" / "hand_over.json", starter)
    learned = json.loads((FIXTURES / "skill_clear_table.json").read_text(encoding="utf-8"))
    learned["real"] = {"runs": 1, "successes": 1}
    _write_json(skills / "clear_table.json", learned)
    _write_json(
        runs / "archive" / "20261017-170420" / "clear_table.json",
        {**learned, "real": {"runs": 2, "successes": 2}},
    )
    _write_json(skills / "_backup" / "bad.json", {"name": "bad"})

    _write_json(
        root / "rl" / "policies" / "handoff_best.json",
        {
            "policy": "handoff_best.zip",
            "sha256": "abc123",
            "source_policy": "rl/runs/baseline/policy.zip",
            "reward": "rl/rewards/baseline.py",
            "requested_steps": 500000,
            "steps": 507904,
            "training_seed": 1,
            "selection_reason": "the baseline outperformed the variants",
            "evaluation": {
                "evaluated_on": "2026-10-17",
                "seed": 0,
                "episodes": 100,
                "successes": 97,
                "success_rate": 0.97,
                "mean_steps_to_success": 47.9,
                "falls": 0,
            },
            "deployment": "simulation",
        },
    )
    _write_json(
        root / "rl" / "runs" / "baseline" / "dgo.json",
        {
            "evaluated_on": "2026-10-17",
            "command": "rl.evaluate",
            "policy": "rl/runs/baseline/policy.zip",
            "policy_sha256": "abc123",
            "seed": 0,
            "episodes": 100,
            "successes": 97,
            "success_rate": 0.97,
            "falls": 0,
            "threshold": 0.8,
            "decision": "GO",
        },
    )
    _write_json(
        root / "rl" / "runs" / "parallel" / "result.json",
        {
            "variants": [
                {
                    "name": "variant_1",
                    "hint": "exp",
                    "runner": "local",
                    "chosen": True,
                    "error": None,
                    "summary": {"steps": 151552, "episodes": 100, "successes": 41, "success_rate": 0.41},
                },
                {
                    "name": "variant_2",
                    "hint": "progress",
                    "runner": "daytona",
                    "chosen": False,
                    "error": None,
                    "summary": {"steps": 151552, "episodes": 100, "successes": 0, "success_rate": 0.0},
                },
            ]
        },
    )
    _write_json(
        root / "web" / "static" / "rl" / "curves.json",
        {
            "variants": [
                {"name": "baseline", "steps": [20000, 40000], "success": [0.1, 0.75], "chosen": True},
                {"name": "variant_1", "steps": [20000], "success_rate": [0.3], "chosen": False},
            ],
            "chosen": "baseline",
            "created": "2026-10-17T16:42:42+00:00",
            "note": "team decision",
        },
    )
    _write_json(
        runs / "rl_deploy" / "20261017-200000" / "summary.json",
        {
            "stage": "fixed",
            "trials": 5,
            "within_3cm": 4,
            "over_speed_trials": 0,
            "pass": True,
            "policy": "rl/policies/handoff_best.zip",
            "policy_sha256": "abc123",
            "driver": "fake",
            "hand_speed_max": 0.0,
            "created": "2026-10-17T20:00:00+00:00",
        },
    )
    _write_json(
        runs / "replay" / "tally.json",
        {"place_in_tray": {"runs": 5, "successes": 5, "last_t": T0, "four_of_five": True}},
    )
    _write_jsonl(
        runs / "replay" / "attempts.jsonl",
        [{"t": T0 + k, "label": "place_in_tray", "mode": "fake", "ok": True} for k in range(5)],
    )

    _write_json(
        runs / "setup" / "camera-soak-20261017" / "result.json",
        {
            "exit_code": 0,
            "started_epoch": T0,
            "elapsed_s": 1800.2,
            "duration_requested_s": 1800,
            "log": "camera-check.log",
        },
    )
    _write_json(
        runs / "setup" / "camera-format-20261017" / "after-independent-readers.json",
        [
            {
                "name": "overhead",
                "camera_index": 2,
                "fourcc": "MJPG",
                "elapsed_s": 60.0,
                "frames": 1771,
                "failures": 0,
                "late": 2,
                "fps": 29.52,
                "max_read_s": 0.11,
                "max_interframe_s": 0.11,
                "scope": "60-second diagnostic",
            },
            {
                "name": "wrist",
                "camera_index": 1,
                "fourcc": "MJPG",
                "elapsed_s": 60.0,
                "frames": 1800,
                "failures": 0,
                "late": 4,
                "fps": 29.99,
                "max_read_s": 0.109,
                "max_interframe_s": 0.109,
                "scope": "60-second diagnostic",
            },
        ],
    )
    _write_json(
        runs / "setup" / "soak-check" / "soak.json",
        {
            "requested_s": 2,
            "elapsed_s": 2.01,
            "completed": True,
            "passed": True,
            "fake": True,
            "started_at": "2026-10-17T21:00:00+00:00",
            "feeds": [
                {
                    "name": "overhead",
                    "index": None,
                    "frames": 60,
                    "failures": 0,
                    "late": 0,
                    "fps": 30.0,
                    "max_read_s": 0.034,
                    "max_interframe_s": 0.034,
                }
            ],
        },
    )
    _write_json(runs / "setup" / "markers" / "markers.json", {"side_cm": 3.0, "ids": [0, 1]})

    _write_json(
        runs / "calibration" / "check-01.json",
        {
            "measurement_kind": "physical",
            "tolerance_m": 0.015,
            "all_within_tolerance": True,
            "mean_error_m": 0.008,
            "max_error_m": 0.012,
            "poses": [{"error_m": 0.008}] * 5,
        },
    )
    _write_json(
        runs / "calibration" / "measurements-template.json",
        {
            "measurement_kind": "physical",
            "poses": [
                {"hardware_report": f"pose-{i}.json", "tip_xyz": [None, None, None]} for i in range(1, 6)
            ],
        },
    )
    _write_json(
        runs / "calibration" / "gripper-20261017.json",
        {
            "captured_at": "2026-10-17T20:00:00+00:00",
            "mode": "fake",
            "jaw_open_mm": 55.0,
            "empty_closed_pct": 0.0,
            "objects": {"pill_bottle": {"closed_pct": [25, 25], "held": [True, True], "held_min": 12.5}},
        },
    )
    _write_json(
        runs / "touch_markers" / "results.json",
        {
            "t": T0,
            "mode": "fake",
            "chain": False,
            "loose": False,
            "limit_cm": 1.5,
            "worst_cm": 0.15,
            "passed": True,
            "aborted": False,
            "markers": [{"name": "pill_bottle"}, {"name": "juice_box"}, {"name": "tea_tin"}],
        },
    )
    _write_json(
        runs / "hardware" / "20261017-continuation.json",
        {
            "read_only": True,
            "arms": {
                "follower": {
                    "role": "follower",
                    "port": "COM8",
                    "lerobot_id": "my_awesome_follower_arm",
                    "calibration": {"sha256": "5494", "matches_hardware": True},
                },
                "leader": {
                    "role": "leader",
                    "port": "COM7",
                    "lerobot_id": "my_awesome_leader_arm",
                    "calibration": {"sha256": "aaaa", "matches_hardware": False},
                },
            },
        },
    )
    return {"runs": runs, "skills": skills, "root": root}


@pytest.fixture
def report(tree: dict[str, Path]) -> dict:
    return story_numbers.collect(tree["runs"], tree["skills"], tree["root"])


def _task(report: dict, dir_name: str) -> dict:
    return next(t for t in report["tasks"] if t["dir"] == dir_name)


def _args(tree: dict[str, Path]) -> list[str]:
    return ["--runs", str(tree["runs"]), "--skills", str(tree["skills"]), "--root", str(tree["root"])]


def test_read_events_accepts_the_contract_fixture_and_rejects_bad_lines(tmp_path: Path) -> None:
    fixture = json.loads((FIXTURES / "event_candidate_scored.json").read_text(encoding="utf-8"))
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(fixture) + '\n\n{"stage": "saved"}\n', encoding="utf-8")
    events, notes = story_numbers.read_events(path)
    assert [e["stage"] for e in events] == ["candidate_scored"]
    assert len(notes) == 1 and notes[0].startswith(f"{path.as_posix()}:3: off schemas/event.json")


def test_learn_task_numbers_come_from_the_events_with_their_paths(report: dict) -> None:
    task = _task(report, "t-0001")
    assert task["command"] == "put the juice box in the tray"
    assert (task["kind"], task["outcome"], task["end_stage"], task["group"]) == ("learn", "saved", "done", "")
    assert task["candidates_written"] == 6 and task["candidates_written_total"] == 6
    assert task["passing_candidates"] == [3, 4, 5, 6]
    assert task["repair_rounds"] == 0 and len(task["rounds"]) == 1
    assert task["rounds"][0]["runners"] == {"local": 6} and task["rounds"][0]["trials"] == 5
    assert task["seconds_to_executing"] == 8.6 and task["seconds_to_saved"] == 9.0
    assert task["selected"] == 3 and task["saved_skill"] == "juice_box_to_tray"
    assert task["runners"] == {"local": 6}
    assert task["scope"] == {
        "scoring": "local",
        "robot": "not recorded",
        "camera": "not recorded",
        "model": "not recorded",
    }
    assert task["sources"] == {
        "events": "t-0001/events.jsonl",
        "command": "t-0001/command.txt",
        "router": "t-0001/router.json",
        "scores": "t-0001/scores.json",
        "selected": "t-0001/selected.json",
    }
    assert task["started_at"] == "2026-10-17T20:07:03+00:00"
    failed = [c for c in task["candidates"] if not c["passed"]]
    assert [c["candidate"] for c in failed] == [1, 2]
    assert failed[0]["fail_reason"] == "trial 1: gripper hit cereal_box" and failed[0]["runner"] == "local"


def test_repair_round_queued_task_and_saved_skill_run(report: dict) -> None:
    repaired = _task(report, "t-0002")
    assert repaired["repair_rounds"] == 1
    assert [r["passing"] for r in repaired["rounds"]] == [[], [1, 2, 3, 4, 5, 6]]
    assert [r["written"] for r in repaired["rounds"]] == [6, 6]
    assert repaired["candidates_written_total"] == 12 and repaired["runners"] == {"local": 12}
    assert repaired["passing_candidates"] == [1, 2, 3, 4, 5, 6] and repaired["saved_skill"] == "clear_table"
    assert repaired["sources"]["repair_scores"] == "t-0002/repair/scores.json"
    assert repaired["seconds_to_executing"] == 20.3 and repaired["seconds_to_saved"] == 21.0
    queued = _task(report, "t-0003")
    assert queued["outcome"] == "queued" and queued["end_stage"] == "done"
    assert queued["seconds_to_executing"] is None and queued["seconds_to_saved"] is None
    assert queued["passing_candidates"] == [] and queued["saved_skill"] is None and queued["selected"] is None
    run = _task(report, "t-0004")
    assert run["kind"] == "run" and run["rounds"] == [] and run["outcome"] == "done"
    assert run["seconds_to_executing"] == 1.2 and run["runners"] == {}
    assert run["scope"]["scoring"] == "unscored"
    nested = _task(report, "verification/c7-bench-20261017-194847/runs/t-0001")
    assert nested["group"] == "verification/c7-bench-20261017-194847/runs"
    assert nested["passing_candidates"] == [3, 4] and nested["task_id"] == "t-0001"


def test_totals_count_every_task_and_name_the_events_files(report: dict) -> None:
    totals = report["totals"]
    assert totals["tasks"] == 6
    assert totals["by_kind"] == {"learn": 4, "run": 1, "other": 1}
    assert totals["by_outcome"] == {"saved": 3, "queued": 2, "done": 1}
    assert totals["tasks_with_candidates"] == 4
    assert totals["learned_skills"] == 3
    assert totals["learned_skill_names"] == ["clear_table", "juice_box_to_tray"]
    assert totals["candidates_written"] == 24 and totals["candidates_written_total"] == 30
    assert totals["candidates_passing"] == 12 and totals["repair_rounds"] == 1
    assert totals["scored_runs_by_runner"] == {"local": 30} and totals["sandbox_or_local_runs"] == 30
    assert totals["seconds_to_executing"] == {"count": 4, "mean": 9.7, "median": 8.6, "min": 1.2, "max": 20.3}
    assert totals["time_to_learn_s"] == {"count": 3, "mean": 13.0, "median": 9.0, "min": 9.0, "max": 21.0}
    assert sorted(totals["derived_from"]) == sorted(t["sources"]["events"] for t in report["tasks"])


def test_skills_rl_bench_replay_camera_and_calibration_rows_carry_their_files(report: dict) -> None:
    skills = {(s["where"], s["name"]): s for s in report["skills"]}
    assert skills[("live", "hand_over")]["trusted"] is True
    assert skills[("live", "hand_over")]["file"] == "hand_over.json"
    assert skills[("starter", "hand_over")]["file"] == "_starter/hand_over.json"
    assert skills[("live", "clear_table")]["real"] == {"runs": 1, "successes": 1}
    assert skills[("archive", "clear_table")]["file"] == "archive/20261017-170420/clear_table.json"
    assert skills[("archive", "clear_table")]["real"] == {"runs": 2, "successes": 2}
    assert ("backup", "bad") not in skills

    rl = report["rl"]
    assert rl["dgo"]["successes"] == 97 and rl["dgo"]["file"] == "rl/runs/baseline/dgo.json"
    assert rl["provenance"]["steps"] == 507904 and rl["provenance"]["file"] == "rl/policies/handoff_best.json"
    assert rl["provenance"]["evaluation"]["successes"] == 97
    assert rl["hash_match"] is True
    assert rl["parallel"]["chosen"] == "variant_1" and rl["parallel"]["runners"] == {"local": 1, "daytona": 1}
    assert rl["parallel"]["variants"][1]["successes"] == 0
    assert rl["curves"]["chosen"] == "baseline" and rl["curves"]["file"] == "web/static/rl/curves.json"
    assert [s["final_success_rate"] for s in rl["curves"]["series"]] == [0.75, 0.3]
    assert rl["missing"] == []
    (deploy,) = rl["deploy"]
    assert deploy["file"] == "rl_deploy/20261017-200000/summary.json"
    assert (deploy["within_3cm"], deploy["trials"], deploy["over_speed_trials"]) == (4, 5, 0)
    assert deploy["driver"] == "fake" and deploy["pass"] is True

    (bench,) = report["benches"]
    assert bench["file"] == "verification/c7-bench-20261017-194847/summary.json"
    assert bench["seconds_to_executing"] == [8.6, 10.2] and bench["median_s"] == 9.4
    assert bench["saved"] == 2 and bench["task_ids"] == ["t-0001", "t-0002"]

    assert report["replays"] == [
        {
            "file": "replay/tally.json",
            "label": "place_in_tray",
            "runs": 5,
            "successes": 5,
            "four_of_five": True,
            "last_t": T0,
            "modes": {"fake": 5},
        }
    ]

    camera = {c["file"]: c for c in report["camera"]}
    wrapper = camera["setup/camera-soak-20261017/result.json"]
    assert wrapper["kind"] == "soak_result" and wrapper["completed"] is True and wrapper["passed"] is True
    diagnostic = camera["setup/camera-format-20261017/after-independent-readers.json"]
    assert diagnostic["kind"] == "reader_diagnostic" and diagnostic["feeds"][0]["late"] == 2
    soak = camera["setup/soak-check/soak.json"]
    assert soak["fake"] is True and soak["feeds"][0]["frames"] == 60 and soak["passed"] is True
    assert "setup/markers/markers.json" not in camera

    calibration = {c["kind"]: c for c in report["calibration"]}
    assert calibration["a6_check"]["max_error_m"] == 0.012 and calibration["a6_check"]["poses"] == 5
    assert calibration["a6_check"]["file"] == "calibration/check-01.json"
    assert calibration["a6_measurements"]["poses_filled"] == 0
    assert calibration["gripper"]["held_min"] == {"pill_bottle": 12.5}
    assert calibration["touch_markers"]["worst_cm"] == 0.15 and calibration["touch_markers"]["markers"] == 3
    assert calibration["hardware_inspection"]["arms"]["leader"]["calibration_matches"] is False


def test_malformed_files_become_notes_and_strict_exits_1(
    tree: dict[str, Path], tmp_path: Path, capsys
) -> None:
    report = story_numbers.collect(tree["runs"], tree["skills"], tree["root"])
    notes = report["notes"]
    assert any(n.startswith("t-0005/events.jsonl:2: not JSON") for n in notes)
    assert any(n.startswith("t-0005/events.jsonl:3: off schemas/event.json") for n in notes)
    assert any(n.startswith("verification/odd/summary.json: unknown summary layout") for n in notes)
    assert any(n.startswith("_backup/bad.json: off schemas/skill.json") for n in notes)
    assert len(notes) == 4
    bad = _task(report, "t-0005")
    assert bad["stages"] == ["routing", "queued", "done"]
    assert bad["outcome"] == "queued" and bad["kind"] == "other" and bad["rounds"] == []

    assert story_numbers.main([*_args(tree), "--strict", "--out", str(tmp_path / "strict.json")]) == 1
    assert story_numbers.main([*_args(tree), "--out", str(tmp_path / "lenient.json")]) == 0
    out = capsys.readouterr().out
    assert "note: t-0005/events.jsonl:2: not JSON" in out
    assert json.loads((tmp_path / "strict.json").read_text(encoding="utf-8"))["notes"] == notes


def test_markdown_and_summary_name_sources_and_scopes(report: dict) -> None:
    text = story_numbers.render_markdown(report)
    assert "## Measured results (generated)" in text
    assert "| Learn tasks | 3 of 4 candidate-scored tasks reached executing, 3 saved a skill" in text
    assert "| t-*/events.jsonl (3); verification/c7-bench-20261017-194847/runs/t-*/events.jsonl (1) |" in text
    assert "robot/camera/model mode not recorded in runs/" in text
    assert (
        "| Repair recovery | 1 task(s) ran a repair round: "
        "t-0002: round 1 0 of 6, round 2 6 of 6, saved clear_table" in text
    )
    assert "| Saved-skill runs | 1 of 1 saved-skill runs reached executing; seconds to executing 1.2" in text
    assert "97/100 successes, 0 falls, seed 0; 507,904 training steps" in text
    assert "rl/runs/baseline/dgo.json; rl/policies/handoff_best.json; policy hash matches" in text
    assert (
        "| Policy bring-up (fixed) | 4 of 5 tries within 3 cm, 0 over the speed cap, pass yes | driver fake"
        in text
    )
    assert (
        "| Replays of place_in_tray | 5 of 5 attempts ok; four of five yes | attempt modes fake 5 |" in text
    )
    assert (
        "| t-0001 |  | put the juice box in the tray | learn | saved | 6 | 3, 4, 5, 6 | 0 | 8.6 | 9.0 "
        "| local 6 | t-0001/events.jsonl |" in text
    )
    assert (
        "| Calibration touch markers | 3 markers, worst 0.15 cm against 1.5 cm, passed yes | mode fake |"
        in text
    )
    assert "- t-0005/events.jsonl:2: not JSON" in text
    summary = story_numbers.render_summary(report)
    assert summary.startswith("tasks 6 (learn 4, run 1, other 1); outcomes saved 3, queued 2, done 1\n")
    assert "seconds to executing: median 8.6 s, mean 9.7 s over 4 tasks" in summary
    assert "RL: 97/100 successes, 0 falls, seed 0 (rl/runs/baseline/dgo.json)" in summary


def test_empty_runs_folder_gives_an_empty_report_and_exit_0(tmp_path: Path, capsys) -> None:
    runs, skills, root = tmp_path / "runs", tmp_path / "skills", tmp_path / "root"
    for folder in (runs, skills, root):
        folder.mkdir()
    out = tmp_path / "story" / "empty.json"
    argv = [
        "--runs",
        str(runs),
        "--skills",
        str(skills),
        "--root",
        str(root),
        "--markdown",
        "--out",
        str(out),
    ]
    assert story_numbers.main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["tasks"] == [] and report["totals"]["tasks"] == 0 and report["skills"] == []
    assert report["totals"]["seconds_to_executing"]["count"] == 0 and report["totals"]["derived_from"] == []
    assert report["benches"] == [] and report["replays"] == [] and report["camera"] == []
    assert report["rl"]["missing"] == [
        "rl/policies/handoff_best.json",
        "rl/runs/baseline/dgo.json",
        "rl/runs/parallel/result.json",
        "web/static/rl/curves.json",
    ]
    assert report["notes"] == []
    text = capsys.readouterr().out
    assert "| Check | Observed result | Scope | Source |" in text
    assert "- none: every file read matched a known layout" in text
    assert text.rstrip().splitlines()[-1] == f"wrote {out}"


def test_default_out_is_a_new_timestamped_file_under_runs_story(tree: dict[str, Path], capsys) -> None:
    assert story_numbers.main(_args(tree)) == 0
    assert story_numbers.main(_args(tree)) == 0
    written = sorted((tree["runs"] / "story").glob("*.json"))
    assert len(written) == 2 and written[0] != written[1]
    assert all(json.loads(p.read_text(encoding="utf-8"))["totals"]["tasks"] == 6 for p in written)
    lines = capsys.readouterr().out.rstrip().splitlines()
    assert lines[-1] == f"wrote {written[-1]}" or lines[-1] == f"wrote {written[0]}"


def test_misuse_exits_2_and_never_overwrites(tree: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing_runs:
        story_numbers.main(["--runs", str(tmp_path / "missing")])
    assert missing_runs.value.code == 2
    with pytest.raises(SystemExit) as missing_skills:
        story_numbers.main(["--runs", str(tree["runs"]), "--skills", str(tmp_path / "nope")])
    assert missing_skills.value.code == 2
    existing = tmp_path / "exists.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as taken:
        story_numbers.main([*_args(tree), "--out", str(existing)])
    assert taken.value.code == 2 and existing.read_text(encoding="utf-8") == "{}"


def test_help_prints_every_flag_and_exits_0() -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    for flag in ("--runs", "--skills", "--root", "--out", "--markdown", "--strict"):
        assert flag in done.stdout
