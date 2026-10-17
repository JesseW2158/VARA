"""agents.orchestrator (S0 item 5, C7, C9, C10, C11): endpoints, flows, STOP, budgets and the event stream.

Everything runs with `FAKE=1`, a temporary skills directory (a copy of the starters plus one backup skill)
and a temporary runs directory, through FastAPI's TestClient with the lifespan. The slow stages are
replaced by small stubs wherever a test is about control flow; the learn, run, routine and queue flows run
the real fake pipeline (the local runner on the MuJoCo sim). Nothing touches xAI, Daytona, the arm or a
camera, and nothing is written into the repo's skills/ or runs/.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from agents import candidates, orchestrator, sandbox_pool
from agents.orchestrator import Orchestrator
from arm import driver as driver_mod
from arm import ik
from vision import scene as scene_mod

ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = ROOT / "skills" / "_starter"
FIXTURES = ROOT / "tests" / "fixtures"
EVENT_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "event.json").read_text(encoding="utf-8"))
)
SKILL_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "skill.json").read_text(encoding="utf-8"))
)
STARTERS = ["hand_over", "place_in_tray"]
SENTINEL_KEY = "xai-SENTINEL-never-leaks-0123456789"
SLOW_S = 150.0  # the real fake pipeline (6 candidates x 5 MuJoCo trials) takes ~20 s on a laptop
HOME = [float(v) for v in ik.home_q()]
AWAY = [0.3, -1.0, 0.3, 1.0, 0.0]  # a pose well away from home for the "sent home" checks
JUICE_SPEC = json.loads((FIXTURES / "llm" / "router" / "learn_juice_box_to_tray.json").read_text())["spec"]
LEARNED = json.loads((FIXTURES / "skill_clear_table.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------- helpers


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def renamed(name: str) -> dict[str, Any]:
    skill = json.loads(json.dumps(LEARNED))
    skill["name"] = name
    skill["spec"]["name"] = name
    return skill


def route_reply(**reply: Any):
    """A `router.route` stub returning one fixed reply."""

    def route(text: str, skills: Any, scene: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(reply)

    return route


def learn_reply() -> Any:
    return route_reply(action="learn", spec=json.loads(json.dumps(JUICE_SPEC)))


def run_reply() -> Any:
    return route_reply(action="run", skill="hand_over", args={"object": "pill_bottle"})


def canned_trajectory() -> dict[str, Any]:
    """One waypoint at home: valid, inside every limit and the box, replays in no time."""
    return {
        "version": 1,
        "skill": "hand_over",
        "args": {"object": "pill_bottle"},
        "dt": 1 / 30,
        "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        "segments": [{"kind": "path", "q": [list(HOME)]}],
    }


def plan_stub(scene_after: dict[str, Any] | None = None):
    def plan(skill: Any, args: Any, scene: Any, **_kwargs: Any) -> tuple[dict, dict, dict]:
        final = json.loads(json.dumps(scene)) if scene_after is None else scene_after
        return canned_trajectory(), final, {"candidate": 0, "passed": True}

    return plan


def accepted_candidates(task_dir: Path, n: int = 2) -> list[candidates.Candidate]:
    """Fixture candidates 3 and 4 (both pass in the sim) written under `task_dir`."""
    out = []
    for _k, index in enumerate((3, 4, 1, 2, 5, 6)[:n]):
        source = (FIXTURES / "llm" / "candidates" / "juice_box_to_tray" / f"cand_{index}.py").read_text()
        path = task_dir / f"cand_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        out.append(candidates.Candidate(index, candidates.HINTS[index - 1], source, [], 1, path))
    return out


def gate_stub(gate: threading.Event, entered: threading.Event, result: Any = None):
    """A stage stub that blocks on `gate` (and flags `entered`) before returning `result`."""

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        entered.set()
        gate.wait(30)
        return result() if callable(result) else result

    return blocked


def wait_for(orch: Orchestrator, task_id: str, stages: tuple[str, ...], timeout: float = 10.0) -> dict:
    event = orchestrator.wait_for_stage(orch, task_id, stages, timeout)
    assert event is not None, (
        f"no {stages} event for {task_id} within {timeout} s: {stages_of(orch, task_id)}"
    )
    return event


def wait_idle(orch: Orchestrator, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker = orch.worker
        if orch.state != "busy" and (worker is None or not worker.is_alive()):
            return
        time.sleep(0.02)
    raise AssertionError(f"still busy after {timeout} s (state {orch.state})")


def stages_of(orch: Orchestrator, task_id: str) -> list[str]:
    return [e["stage"] for e in orch.recent_events() if e["task_id"] == task_id]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def scan_for(secret: str, folder: Path) -> list[Path]:
    hits = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt", ".py"}:
            if secret in path.read_text(encoding="utf-8", errors="replace"):
                hits.append(path)
    return hits


# ------------------------------------------------------------------------------------------------ fixtures


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """FAKE=1, a sentinel key, tmp skills (starters + one backup) and tmp runs; fresh singletons."""
    skills = tmp_path / "skills"
    shutil.copytree(STARTER_DIR, skills / "_starter")
    write_json(skills / "_backup" / "backup_demo.json", renamed("backup_demo"))
    runs = tmp_path / "runs"
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("XAI_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("VARA_SKILLS_DIR", str(skills))
    monkeypatch.setenv("VARA_RUNS_DIR", str(runs))
    driver_mod.reset_driver()
    scene_mod.reset_tracker()
    sandbox_pool.reset_pool()
    yield {"skills": skills, "runs": runs}
    driver_mod.reset_driver()
    scene_mod.reset_tracker()
    sandbox_pool.reset_pool()


@contextlib.contextmanager
def running(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Orchestrator]]:
    """A fresh Orchestrator behind the app, served by the TestClient; no thread may outlive it."""
    baseline = threading.active_count()
    orch = Orchestrator()
    monkeypatch.setattr(orchestrator, "orch", orch)
    with TestClient(orchestrator.app) as client:
        yield client, orch
    deadline = time.monotonic() + 5.0
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = [t.name for t in threading.enumerate()]
    assert threading.active_count() <= baseline, f"threads leaked: {leaked}"


@pytest.fixture
def served(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Orchestrator]]:
    with running(monkeypatch) as pair:
        yield pair


# -------------------------------------------------------------------------------------------- pages, state


def test_pages_static_and_state_shape(served: tuple[TestClient, Orchestrator]) -> None:
    client, orch = served
    assert client.get("/").status_code == 200 and "VARA" in client.get("/").text
    assert client.get("/voice").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    state = client.get("/state").json()
    assert set(state) == {"state", "mode", "carry", "task_id", "skills", "queue", "stopped"}
    assert state["state"] == "idle" and state["mode"] == "normal" and state["carry"] in ("ik", "rl")
    assert state["task_id"] is None and state["queue"] == [] and state["stopped"] is False
    assert sorted(s["name"] for s in state["skills"]) == STARTERS
    for skill in state["skills"]:
        assert set(skill) >= {"name", "description", "trusted", "real", "sim", "created"}
        assert set(skill["real"]) == {"runs", "successes"} and set(skill["sim"]) == {"successes", "trials"}
    assert orch.skills_dir.name == "skills" and orch.runs_is_default is False


def test_second_command_mid_task_gets_409(served, monkeypatch: pytest.MonkeyPatch) -> None:
    client, orch = served
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(orchestrator.router, "route", gate_stub(gate, entered, {"action": "stop"}))
    first = client.post("/command", json={"text": "hand me my pills"})
    assert first.status_code == 200 and first.json()["task_id"] == "t-0001"
    assert entered.wait(5)
    second = client.post("/command", json={"text": "hand me my pills"})
    assert second.status_code == 409 and second.json() == {"detail": "busy"}
    assert client.get("/state").json()["state"] == "busy"
    assert client.post("/torque_off").status_code == 409
    assert client.post("/reset_demo").status_code == 409
    assert client.post("/resume").status_code == 409  # no stop was requested: the task keeps running
    gate.set()
    wait_for(orch, "t-0001", ("stopped",))
    wait_idle(orch)
    assert client.get("/state").json()["state"] == "stopped"  # the router said stop: like /stop
    assert client.post("/resume").json() == {"state": "idle"}


def test_task_ids_continue_from_the_highest_runs_dir(env, monkeypatch: pytest.MonkeyPatch) -> None:
    (env["runs"] / "t-0041").mkdir(parents=True)
    (env["runs"] / "t-0007").mkdir(parents=True)
    (env["runs"] / "archive").mkdir(parents=True)
    with running(monkeypatch) as (client, orch):
        monkeypatch.setattr(orchestrator.router, "route", route_reply(action="queue", task="x", reason="r"))
        assert client.post("/command", json={"text": "x"}).json() == {"task_id": "t-0042"}
        wait_for(orch, "t-0042", ("done",))
        wait_idle(orch)
        assert (env["runs"] / "t-0042" / "events.jsonl").is_file()


# ------------------------------------------------------------------------------------------- STOP / resume


STOP_STAGES = ("routing", "writing_code", "simulating", "validating", "executing", "verifying")


def arm_stage(stage: str, orch: Orchestrator, monkeypatch: pytest.MonkeyPatch, gate, entered) -> None:
    """Make `stage` block on `gate`; the stages before it are stubbed to be instant."""
    blocked = gate_stub(gate, entered, [])
    if stage == "routing":
        monkeypatch.setattr(
            orchestrator.router,
            "route",
            gate_stub(gate, entered, {"action": "queue", "task": "x", "reason": "r"}),
        )
        return
    if stage in ("writing_code", "simulating"):
        monkeypatch.setattr(orchestrator.router, "route", learn_reply())
        if stage == "writing_code":
            monkeypatch.setattr(orchestrator.candidates, "generate", blocked)
            return
        monkeypatch.setattr(
            orchestrator.candidates, "generate", lambda *a, task_dir, **k: accepted_candidates(Path(task_dir))
        )
        monkeypatch.setattr(orch.pool, "score", blocked)
        return
    monkeypatch.setattr(orchestrator.router, "route", run_reply())
    monkeypatch.setattr(orchestrator.planner, "plan", plan_stub())
    if stage == "validating":
        monkeypatch.setattr(orchestrator.replay, "validate", blocked)
    elif stage == "executing":
        ok = {"ok": True, "reason": None, "events": ["done"]}
        monkeypatch.setattr(orchestrator.replay, "replay", gate_stub(gate, entered, ok))
    else:
        monkeypatch.setattr(
            orchestrator.verify_mod,
            "verify",
            gate_stub(gate, entered, {"ok": True, "missing": [], "reason": "x"}),
        )


@pytest.mark.parametrize("stage", STOP_STAGES)
def test_stop_cancels_every_stage_freezing_first(served, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    client, orch = served
    gate, entered = threading.Event(), threading.Event()
    arm_stage(stage, orch, monkeypatch, gate, entered)
    freeze_calls: list[tuple[bool, str]] = []
    original_freeze = orch.driver.freeze

    def freeze() -> None:
        freeze_calls.append((orch.stop_event.is_set(), orch.state))
        original_freeze()

    monkeypatch.setattr(orch.driver, "freeze", freeze)
    task_id = client.post("/command", json={"text": "go"}).json()["task_id"]
    assert entered.wait(10), f"stage {stage} never started: {stages_of(orch, task_id)}"
    wait_for(orch, task_id, (stage,))
    started = time.monotonic()
    reply = client.post("/stop")
    assert reply.status_code == 200 and reply.json() == {"state": "stopped", "task_id": task_id}
    event = wait_for(orch, task_id, ("stopped",), timeout=1.0)
    assert time.monotonic() - started < 1.0
    assert event["detail"]["reason"]
    assert freeze_calls and freeze_calls[0] == (False, "busy"), "driver.freeze() must come before the cancel"
    assert orch.driver.stopped is True
    gate.set()  # let the blocked stage finish; the worker must have abandoned it already
    wait_idle(orch)
    assert client.get("/state").json()["state"] == "stopped"
    assert stages_of(orch, task_id)[-1] == "stopped"
    assert client.post("/command", json={"text": "again"}).status_code == 409
    assert client.post("/resume").json() == {"state": "idle"}
    assert orch.driver.stopped is False
    assert client.get("/state").json()["state"] == "idle"


def test_stop_while_idle_persists_until_resume_and_resume_homes(served) -> None:
    client, orch = served
    orch.driver.set_present(AWAY, 1.0)
    reply = client.post("/stop")
    assert reply.json() == {"state": "stopped", "task_id": None}
    assert client.get("/state").json()["state"] == "stopped"
    assert client.get("/state").json()["stopped"] is True
    assert client.post("/command", json={"text": "hand me my pills"}).status_code == 409
    assert orch.recent_events()[-1]["stage"] == "stopped"
    assert client.post("/resume").json() == {"state": "idle"}
    q, gripper = orch.driver.read()
    assert all(abs(a - b) < math.radians(0.5) for a, b in zip(q, HOME, strict=True)), q
    assert client.get("/state").json()["state"] == "idle"
    assert client.post("/command", json={"text": "fold my laundry"}).status_code == 200
    wait_idle(orch)


def test_torque_off_works_when_idle(served) -> None:
    client, orch = served
    assert client.post("/torque_off").status_code == 200
    assert orch.driver.torque is False
    orch.driver.torque_on()


def test_calibrate_under_fake_camera(served) -> None:
    client, _orch = served
    assert client.post("/calibrate").json() == {"error_cm": 0.8}


# --------------------------------------------------------------------------------------- library endpoints


def test_reset_demo_leaves_the_two_starters_and_clears_the_queue(served, env) -> None:
    client, orch = served
    task_id = client.post("/command", json={"text": "fold my laundry"}).json()["task_id"]
    wait_for(orch, task_id, ("done",))
    wait_idle(orch)
    assert client.get("/state").json()["queue"] == [
        {"task": "fold my laundry", "reason": "needs cloth handling", "task_id": task_id}
    ]
    orch.library.save(renamed("learned_demo"))
    names = sorted(s["name"] for s in client.get("/state").json()["skills"])
    assert names == sorted([*STARTERS, "learned_demo"])
    reply = client.post("/reset_demo")
    assert reply.status_code == 200 and reply.json() == {"skills": STARTERS}
    assert sorted(p.stem for p in env["skills"].glob("*.json")) == STARTERS
    assert client.get("/state").json()["queue"] == []
    archived = list((env["runs"] / "archive").rglob("learned_demo.json"))
    assert len(archived) == 1


def test_load_backup_copies_backup_skills(served, env) -> None:
    client, _orch = served
    reply = client.post("/load_backup")
    assert reply.status_code == 200 and reply.json() == {"skills": ["backup_demo"]}
    assert (env["skills"] / "backup_demo.json").is_file()
    assert sorted(s["name"] for s in client.get("/state").json()["skills"]) == ["backup_demo", *STARTERS]


# --------------------------------------------------------------------------------------------- voice token


def test_voice_token_fake(served) -> None:
    client, _orch = served
    reply = client.post("/voice/token")
    assert reply.status_code == 200
    data = reply.json()
    assert data["token"] == "fake-voice-token" and data["model"] == "grok-voice-latest"
    assert data["expires_at"] > time.time() + 200
    assert SENTINEL_KEY not in reply.text


def test_voice_token_real_path_with_stubbed_httpx(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_LLM", raising=False)
    for flag in ("FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setenv("XAI_API_KEY", SENTINEL_KEY)
    skills = tmp_path / "skills"
    shutil.copytree(STARTER_DIR, skills / "_starter")
    monkeypatch.setenv("VARA_SKILLS_DIR", str(skills))
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    driver_mod.reset_driver()
    scene_mod.reset_tracker()
    sandbox_pool.reset_pool()
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> Any:
        calls.append({"url": url, **kwargs})
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"value": "ek-short-lived-123", "expires_at": 1792209900, "session": {"id": "s"}},
        )

    monkeypatch.setattr(orchestrator.httpx, "post", post)
    try:
        with running(monkeypatch) as (client, orch):
            reply = client.post("/voice/token")
            assert reply.status_code == 200
            assert reply.json() == {
                "token": "ek-short-lived-123",
                "expires_at": 1792209900,
                "model": "grok-voice-latest",
            }
            assert SENTINEL_KEY not in reply.text
            assert calls[0]["url"] == "https://api.x.ai/v1/realtime/client_secrets"
            assert calls[0]["headers"]["Authorization"] == f"Bearer {SENTINEL_KEY}"
            assert calls[0]["json"] == {"expires_after": {"seconds": 300}}
            assert SENTINEL_KEY not in client.get("/state").text
            # without a key the endpoint answers 503 and says nothing about the key
            monkeypatch.delenv("XAI_API_KEY")
            missing = client.post("/voice/token")
            assert missing.status_code == 503 and SENTINEL_KEY not in missing.text
            assert scan_for(SENTINEL_KEY, orch.runs_dir) == [] if orch.runs_dir.exists() else True
    finally:
        driver_mod.reset_driver()
        scene_mod.reset_tracker()
        sandbox_pool.reset_pool()


def test_voice_token_without_a_key_is_503(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE")
    for flag in ("FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.delenv("XAI_API_KEY")
    with running(monkeypatch) as (client, _orch):
        reply = client.post("/voice/token")
        assert reply.status_code == 503
        assert "XAI_API_KEY" in reply.json()["detail"]


# --------------------------------------------------------------------------------------------------- flows


def test_learn_flow_saves_the_juice_box_skill(served, env) -> None:
    client, orch = served
    task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
    wait_for(orch, task_id, ("done", "error", "queued", "stopped"), timeout=SLOW_S)
    wait_idle(orch)
    stages = stages_of(orch, task_id)
    assert stages[:2] == ["routing", "spec_ready"]
    assert stages.count("writing_code") == 6 and stages.count("candidate_scored") == 6
    assert stages[-6:] == ["selected", "validating", "executing", "verifying", "saved", "done"]
    events = {e["stage"]: e for e in orch.recent_events() if e["task_id"] == task_id}
    assert events["spec_ready"]["detail"]["name"] == "juice_box_to_tray"
    scored = [
        e["detail"]
        for e in orch.recent_events()
        if e["task_id"] == task_id and e["stage"] == "candidate_scored"
    ]
    assert sorted(d["candidate"] for d in scored) == [1, 2, 3, 4, 5, 6]
    for detail in scored:
        assert set(detail) >= {
            "candidate",
            "passed",
            "successes",
            "trials",
            "fail_reason",
            "runner",
            "hint",
            "trial_results",
        }
        assert detail["runner"] == "local" and len(detail["trial_results"]) == 5
    assert sum(1 for d in scored if d["passed"]) >= 2
    assert any(d["fail_reason"] and "cereal_box" in d["fail_reason"] for d in scored)
    selected = events["selected"]["detail"]
    assert selected["candidate"] == selected["scores"][0]["candidate"] and selected["scores"][0]["passed"]
    assert selected["hint"] == candidates.HINTS[selected["candidate"] - 1]
    assert events["saved"]["detail"] == {"name": "juice_box_to_tray"}
    assert events["done"]["detail"]["ok"] is True
    skill = json.loads((env["skills"] / "juice_box_to_tray.json").read_text(encoding="utf-8"))
    SKILL_VALIDATOR.validate(skill)
    assert skill["sim"]["trials"] == 5 and skill["sim"]["successes"] >= 4
    assert skill["real"] == {"runs": 1, "successes": 1} and skill["trusted"] is False
    task_dir = env["runs"] / task_id
    for name in (
        "command.txt",
        "router.json",
        "task.json",
        "scene.json",
        "scores.json",
        "selected.json",
        "traj.json",
        "replay.json",
        "before.json",
        "after.json",
        "verify.json",
        "events.jsonl",
        "grasp_1.jpg",
    ):
        assert (task_dir / name).is_file(), name
    lines = [
        json.loads(line) for line in (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [e["stage"] for e in lines] == stages
    for line in lines:
        EVENT_VALIDATOR.validate(line)
    assert scan_for(SENTINEL_KEY, env["runs"]) == []
    assert not (ROOT / "skills" / "juice_box_to_tray.json").exists()


def test_run_flow_hand_over_increments_the_real_counts(served, env) -> None:
    client, orch = served
    task_id = client.post("/command", json={"text": "hand me my pills"}).json()["task_id"]
    wait_for(orch, task_id, ("done", "error", "queued", "stopped"), timeout=SLOW_S)
    wait_idle(orch)
    assert stages_of(orch, task_id) == ["routing", "planning", "validating", "executing", "verifying", "done"]
    done = [e for e in orch.recent_events() if e["task_id"] == task_id and e["stage"] == "done"][0]
    assert done["detail"]["ok"] is True and done["detail"]["skill"] == "hand_over"
    skill = json.loads((env["skills"] / "hand_over.json").read_text(encoding="utf-8"))
    assert skill["real"] == {"runs": 1, "successes": 1}
    assert (env["runs"] / task_id / "verify.json").is_file()
    assert json.loads((env["runs"] / task_id / "verify.json").read_text())["ok"] is True


def test_queue_flow(served) -> None:
    client, orch = served
    task_id = client.post("/command", json={"text": "fold my laundry"}).json()["task_id"]
    wait_for(orch, task_id, ("done",))
    wait_idle(orch)
    assert stages_of(orch, task_id) == ["routing", "queued", "done"]
    queued = [e for e in orch.recent_events() if e["stage"] == "queued"][0]["detail"]
    assert queued == {"task": "fold my laundry", "reason": "needs cloth handling"}
    assert client.get("/state").json()["queue"][0]["task_id"] == task_id


def test_define_routine_then_run_it(served, env) -> None:
    client, orch = served
    text = "when I say good morning, hand me my pill bottle, then my juice box"
    task_id = client.post("/command", json={"text": text}).json()["task_id"]
    wait_for(orch, task_id, ("done", "error", "queued", "stopped"), timeout=SLOW_S)
    wait_idle(orch)
    stages = stages_of(orch, task_id)
    assert stages == [
        "routing",
        "spec_ready",
        "writing_code",
        "simulating",
        "candidate_scored",
        "selected",
        "validating",
        "executing",
        "verifying",
        "saved",
        "done",
    ]
    skill = json.loads((env["skills"] / "good_morning.json").read_text(encoding="utf-8"))
    SKILL_VALIDATOR.validate(skill)
    assert skill["spec"]["description"].startswith("Routine: ")
    assert "run_skill" in skill["code"] and skill["real"] == {"runs": 1, "successes": 1}
    second = client.post("/command", json={"text": "good morning"}).json()["task_id"]
    wait_for(orch, second, ("done", "error", "queued", "stopped"), timeout=SLOW_S)
    wait_idle(orch)
    assert stages_of(orch, second) == ["routing", "planning", "validating", "executing", "verifying", "done"]
    skill = json.loads((env["skills"] / "good_morning.json").read_text(encoding="utf-8"))
    assert skill["real"] == {"runs": 2, "successes": 2}


def test_not_saved_replans_the_next_best_candidate_once(served, env, monkeypatch) -> None:
    client, orch = served
    verdicts: list[bool] = []
    real_verify = orchestrator.verify_mod.verify

    def verify(spec, before, after, args, events, *, task_dir=None):
        if not verdicts:
            verdicts.append(False)
            return {"ok": False, "missing": [], "reason": "spec not satisfied"}
        verdicts.append(True)
        return real_verify(spec, before, after, args, events, task_dir=task_dir)

    monkeypatch.setattr(orchestrator.verify_mod, "verify", verify)
    task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
    wait_for(orch, task_id, ("done", "error", "queued", "stopped"), timeout=SLOW_S)
    wait_idle(orch)
    stages = stages_of(orch, task_id)
    tail = stages[stages.index("not_saved") :]
    assert tail == ["not_saved", "planning", "validating", "executing", "verifying", "saved", "done"]
    first, second = (
        e for e in orch.recent_events() if e["task_id"] == task_id and e["stage"] in ("selected", "planning")
    )
    assert first["detail"]["candidate"] != second["detail"]["candidate"]
    assert (env["skills"] / "juice_box_to_tray.json").is_file()
    assert (env["runs"] / task_id / "replan" / "skill.json").is_file()


def test_exception_inside_a_stage_ends_error_and_homes(served, monkeypatch) -> None:
    client, orch = served
    orch.driver.set_present(AWAY, 1.0)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("router exploded")

    monkeypatch.setattr(orchestrator.router, "route", boom)
    task_id = client.post("/command", json={"text": "x"}).json()["task_id"]
    event = wait_for(orch, task_id, ("error",))
    wait_idle(orch)
    assert "router exploded" in event["detail"]["error"] and event["detail"]["stage"] == "routing"
    q, _gripper = orch.driver.read()
    assert all(abs(a - b) < math.radians(0.5) for a, b in zip(q, HOME, strict=True)), q
    assert client.get("/state").json()["state"] == "idle"


def test_validation_problems_end_error_with_the_problems(served, monkeypatch) -> None:
    client, orch = served
    monkeypatch.setattr(orchestrator.router, "route", run_reply())
    bad = canned_trajectory()
    bad["segments"][0]["q"] = [[0.0, 0.0, 0.0, 0.0, 9.0]]  # wrist_roll beyond its limit
    monkeypatch.setattr(orchestrator.planner, "plan", lambda *a, **k: (bad, {}, {}))
    task_id = client.post("/command", json={"text": "x"}).json()["task_id"]
    event = wait_for(orch, task_id, ("error",))
    wait_idle(orch)
    assert event["detail"]["stage"] == "validating" and event["detail"]["problems"]
    assert "executing" not in stages_of(orch, task_id)


def test_arm_frozen_by_the_replay_needs_resume(served, monkeypatch) -> None:
    client, orch = served
    monkeypatch.setattr(orchestrator.router, "route", run_reply())
    monkeypatch.setattr(orchestrator.planner, "plan", plan_stub())

    def frozen_replay(traj, *, driver, **_kwargs):
        driver.freeze()
        return {"ok": False, "reason": "hand_too_close", "events": ["hand 0.05 m from the tip: frozen"]}

    monkeypatch.setattr(orchestrator.replay, "replay", frozen_replay)
    task_id = client.post("/command", json={"text": "x"}).json()["task_id"]
    event = wait_for(orch, task_id, ("stopped", "error", "done"))
    wait_idle(orch)
    assert event["stage"] == "stopped" and event["detail"]["reason"] == "hand_too_close"
    assert client.get("/state").json()["state"] == "stopped"
    assert client.post("/resume").json() == {"state": "idle"}


# ------------------------------------------------------------------------------------------------- budgets


def test_demo_mode_simulation_budget_ends_queued(env, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setitem(orchestrator.BUDGETS["demo"], "simulating", 0.3)
    with running(monkeypatch) as (client, orch):
        assert orch.mode == "demo" and client.get("/state").json()["mode"] == "demo"
        monkeypatch.setattr(orchestrator.router, "route", learn_reply())
        monkeypatch.setattr(
            orchestrator.candidates,
            "generate",
            lambda *a, task_dir, **k: accepted_candidates(Path(task_dir), 3),
        )
        gate, entered = threading.Event(), threading.Event()
        monkeypatch.setattr(orch.pool, "score", gate_stub(gate, entered, []))
        task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
        assert entered.wait(10)
        event = wait_for(orch, task_id, ("queued", "error", "done"), timeout=5.0)
        gate.set()
        wait_idle(orch)
        assert event["stage"] == "queued"
        assert "did not finish within 0.3 s" in event["detail"]["reason"]
        assert stages_of(orch, task_id)[-1] == "done"
        assert client.get("/state").json()["queue"][0]["task_id"] == task_id


def test_demo_mode_75_s_cap_queues_before_the_arm_moves(env, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setitem(orchestrator.BUDGETS["demo"], "to_arm", 0.5)
    with running(monkeypatch) as (client, orch):
        monkeypatch.setattr(orchestrator.router, "route", learn_reply())

        def slow_generate(*_a: Any, task_dir: Any, **_k: Any) -> list[candidates.Candidate]:
            time.sleep(0.8)
            return accepted_candidates(Path(task_dir), 3)

        monkeypatch.setattr(orchestrator.candidates, "generate", slow_generate)
        task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
        event = wait_for(orch, task_id, ("queued", "error", "done"), timeout=5.0)
        wait_idle(orch)
        assert event["stage"] == "queued" and event["detail"]["reason"] == "over the 75 s budget"
        assert "simulating" not in stages_of(orch, task_id)


def test_demo_mode_needs_three_valid_candidates(env, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    with running(monkeypatch) as (client, orch):
        monkeypatch.setattr(orchestrator.router, "route", learn_reply())
        monkeypatch.setattr(
            orchestrator.candidates,
            "generate",
            lambda *a, task_dir, **k: accepted_candidates(Path(task_dir), 2),
        )
        task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
        event = wait_for(orch, task_id, ("queued", "error", "done"), timeout=5.0)
        wait_idle(orch)
        assert event["stage"] == "queued" and "need 3" in event["detail"]["reason"]


def failing_pool_score(calls: list[dict[str, Any]]):
    """A `pool.score` stub: every candidate fails ("trial 1: spec not satisfied") through `on_score`."""

    def score(task_dir, cands, task_path, scene_path, skills_dir, *, on_score=None, **kwargs):
        indices = [int(index) for index, _path in cands]  # the orchestrator passes (index, path) pairs
        calls.append({"task_dir": Path(task_dir), "candidates": indices, **kwargs})
        out = []
        for index in indices:
            line = {
                "candidate": index,
                "passed": False,
                "successes": 0,
                "trials": 5,
                "nominal_success": False,
                "collisions": 0,
                "limit_hits": 0,
                "ik_failures": 0,
                "mean_sim_s": 12.0,
                "fail_reason": "trial 1: spec not satisfied",
                "traj": None,
                "error": None,
            }
            scored = {
                "candidate": index,
                "runner": "local",
                "score": line,
                "trial_results": [False] * 5,
                "out_dir": str(Path(task_dir) / f"cand_{index}"),
            }
            out.append(scored)
            if on_score is not None:
                on_score(scored)
        return out

    return score


def test_repair_round_runs_once_in_normal_mode_then_queues(served, monkeypatch) -> None:
    client, orch = served
    monkeypatch.setattr(orchestrator.router, "route", learn_reply())
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(orch.pool, "score", failing_pool_score(calls))
    repairs: list[list[dict[str, Any]]] = []
    real_repair = candidates.repair

    def repair(spec, scene, skills, failures, **kwargs):
        repairs.append([dict(f) for f in failures])
        return real_repair(spec, scene, skills, failures, **kwargs)

    monkeypatch.setattr(orchestrator.candidates, "repair", repair)
    task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
    event = wait_for(orch, task_id, ("queued", "error", "saved"), timeout=60.0)
    wait_idle(orch)
    assert event["stage"] == "queued" and event["detail"]["reason"] == "no candidate passed simulation"
    stages = stages_of(orch, task_id)
    assert stages.count("simulating") == 2 and stages.count("writing_code") == 12
    assert stages.count("candidate_scored") == 12 and "selected" not in stages
    assert len(calls) == 2 and calls[1]["task_dir"].name == "repair"
    assert len(repairs) == 1 and len(repairs[0]) == 3
    assert all({"index", "hint", "code", "score", "fail_reason"} <= set(f) for f in repairs[0])
    assert all(f["fail_reason"] == "trial 1: spec not satisfied" for f in repairs[0])
    assert (orch.runs_dir / task_id / "repair" / "cand_1.py").is_file()
    assert (orch.runs_dir / task_id / "repair" / "scores.json").is_file()


def test_demo_mode_skips_the_repair_round(env, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    with running(monkeypatch) as (client, orch):
        monkeypatch.setattr(orchestrator.router, "route", learn_reply())
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(orch.pool, "score", failing_pool_score(calls))
        task_id = client.post("/command", json={"text": "put the juice box in the tray"}).json()["task_id"]
        event = wait_for(orch, task_id, ("queued", "error", "saved"), timeout=60.0)
        wait_idle(orch)
        assert event["stage"] == "queued" and event["detail"]["reason"] == "no candidate passed simulation"
        assert stages_of(orch, task_id).count("simulating") == 1 and len(calls) == 1
        assert calls[0]["trials"] == 5 and calls[0]["seed"] == 7 and calls[0]["timeout"] <= 35.0


def test_run_flow_stage_timeout_is_an_error(served, monkeypatch) -> None:
    client, orch = served
    monkeypatch.setitem(orchestrator.BUDGETS["normal"], "planning", 0.2)
    monkeypatch.setattr(orchestrator.router, "route", run_reply())
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(orchestrator.planner, "plan", gate_stub(gate, entered, plan_stub()))
    task_id = client.post("/command", json={"text": "x"}).json()["task_id"]
    event = wait_for(orch, task_id, ("error", "queued", "done"), timeout=5.0)
    gate.set()
    wait_idle(orch)
    assert event["stage"] == "error" and "planning did not finish within 0.2 s" in event["detail"]["error"]


# ------------------------------------------------------------------------------------------- events, video


def test_websocket_replays_the_backlog_then_streams(served) -> None:
    client, orch = served
    client.post("/stop")
    client.post("/resume")
    with client.websocket_connect("/events") as ws:
        first = ws.receive_json()
        assert first["stage"] == "stopped" and first["task_id"] is None
        EVENT_VALIDATOR.validate(first)
        task_id = client.post("/command", json={"text": "fold my laundry"}).json()["task_id"]
        seen = []
        while len(seen) < 3:
            event = ws.receive_json()
            if event["task_id"] == task_id:
                seen.append(event["stage"])
        assert seen == ["routing", "queued", "done"]
    wait_idle(orch)


def test_video_streams_send_two_parts_within_two_seconds(env, monkeypatch) -> None:
    orch = Orchestrator()
    monkeypatch.setattr(orchestrator, "orch", orch)
    port = free_port()
    config = uvicorn.Config(orchestrator.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15.0
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        base = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{base}/state", timeout=5).status_code == 200
        assert httpx.get(f"{base}/", timeout=5).status_code == 200
        for name in ("overhead", "wrist"):
            started = time.monotonic()
            received = b""
            with httpx.stream("GET", f"{base}/video/{name}", timeout=5) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
                for chunk in response.iter_bytes():
                    received += chunk
                    if received.count(b"--frame\r\nContent-Type: image/jpeg") >= 2:
                        break
            assert time.monotonic() - started < 2.0, name
            assert received.count(b"\xff\xd8") >= 2, name  # two JPEG starts
    finally:
        server.should_exit = True
        thread.join(15)
    assert not thread.is_alive()
