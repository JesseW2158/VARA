"""VARA_REPAIR_ROUNDS: more repair rounds when no candidate passes (default 1, the contract's budget)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import orchestrator
from agents.orchestrator import Orchestrator, Queued, _Task


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.delenv("VARA_REPAIR_ROUNDS", raising=False)
    return Orchestrator()


def _wire(orch, monkeypatch, *, passes_in_round: int | None, empty_rounds: tuple[int, ...] = ()):
    """Stand-ins for the learn flow's stages. Round 0 is the first six candidates, 1.. the repair rounds.
    `passes_in_round` is the round whose candidates pass (None: never); a round in `empty_rounds` writes
    no valid candidate (_write_code raises Queued, as on a model timeout)."""
    calls: dict[str, list] = {"write": [], "simulate": [], "saved": []}
    state = {"round": -1}

    def write(task, spec, scene, failures=None, out_name="repair"):
        state["round"] += 1
        calls["write"].append((state["round"], out_name if failures is not None else "first", failures))
        if state["round"] in empty_rounds:
            raise Queued(task.text, "no valid candidate program was written")
        return [f"cand-round-{state['round']}"]

    def simulate(task, spec, cands, out_dir):
        calls["simulate"].append(Path(out_dir).name)
        return {"round": state["round"]}

    def select(task, results):
        passed = passes_in_round is not None and results["round"] == passes_in_round
        entry = f"best-of-round-{results['round']}"
        return (entry if passed else None), [entry]

    monkeypatch.setattr(orch, "_before_snapshot", lambda task: {"objects": {}})
    monkeypatch.setattr(orch, "_write_code", write)
    monkeypatch.setattr(orch, "_simulate", simulate)
    monkeypatch.setattr(orch, "_select", select)
    monkeypatch.setattr(orch, "_failure_of", lambda entry: {"from": entry})
    monkeypatch.setattr(
        orch, "_run_and_save", lambda task, spec, winner, ranked, before: calls["saved"].append(winner)
    )
    monkeypatch.setattr(orch, "emit", lambda *a, **k: None)
    return calls


def _task(tmp_path, mode="normal"):
    d = tmp_path / "t-0001"
    d.mkdir(parents=True, exist_ok=True)
    return _Task("t-0001", "move the box", d, mode)


SPEC = {
    "name": "box_right",
    "description": "d",
    "params": {},
    "success": {"moved": {"object": "tea_tin", "min": 0.05}},
    "english": "e",
}


def test_the_default_is_the_contracts_one_repair_round(orch, monkeypatch, tmp_path):
    calls = _wire(orch, monkeypatch, passes_in_round=None)
    with pytest.raises(Queued):
        orch._flow_learn(_task(tmp_path), dict(SPEC))
    assert [c[1] for c in calls["write"]] == ["first", "repair"]
    assert calls["simulate"] == ["t-0001", "repair"]


def test_more_rounds_run_until_one_passes_each_in_its_own_folder(orch, monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_REPAIR_ROUNDS", "4")
    calls = _wire(orch, monkeypatch, passes_in_round=3)
    orch._flow_learn(_task(tmp_path), dict(SPEC))
    assert [c[1] for c in calls["write"]] == [
        "first",
        "repair",
        "repair-2",
        "repair-3",
    ]  # stopped at the pass
    assert calls["saved"] == ["best-of-round-3"]
    # each round writes against the latest round's best failures
    assert calls["write"][2][2] == [{"from": "best-of-round-1"}]


def test_all_rounds_failing_ends_queued(orch, monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_REPAIR_ROUNDS", "3")
    calls = _wire(orch, monkeypatch, passes_in_round=None)
    with pytest.raises(Queued):
        orch._flow_learn(_task(tmp_path), dict(SPEC))
    assert len(calls["write"]) == 4 and calls["saved"] == []


def test_a_round_that_writes_nothing_valid_goes_on_to_the_next(orch, monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_REPAIR_ROUNDS", "3")
    calls = _wire(orch, monkeypatch, passes_in_round=3, empty_rounds=(2,))
    orch._flow_learn(_task(tmp_path), dict(SPEC))
    assert [c[1] for c in calls["write"]] == ["first", "repair", "repair-2", "repair-3"]
    assert "repair-2" not in calls["simulate"] and calls["saved"] == ["best-of-round-3"]


def test_demo_mode_still_has_no_repair_round(orch, monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_REPAIR_ROUNDS", "5")
    calls = _wire(orch, monkeypatch, passes_in_round=None)
    with pytest.raises(Queued):
        orch._flow_learn(_task(tmp_path, mode="demo"), dict(SPEC))
    assert [c[1] for c in calls["write"]] == ["first"]


@pytest.mark.parametrize(("raw", "rounds"), [("", 1), ("0", 0), ("3", 3), ("9", 5), ("-2", 0), ("many", 1)])
def test_repair_rounds_reads_and_clamps_the_setting(monkeypatch, raw, rounds):
    monkeypatch.setenv("VARA_REPAIR_ROUNDS", raw)
    assert orchestrator.repair_rounds() == rounds
