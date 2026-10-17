"""The LLM fixture candidates against the real MuJoCo runner (brief section 6).

In fake mode the six programs in tests/fixtures/llm/candidates/juice_box_to_tray/ stand in for the code
model, so the end-to-end path ("put the juice box in the tray") depends on how they score: at least three
must pass, cand_1 (hint 1, no transit height) must fail by hitting the cereal box, and cand_2 (a plausible
wrong plan: the box set down 10 cm in front of the tray) must fail the success test without touching
anything. The six run here exactly as the orchestrator's local runner starts them: `python -m
sim.run_candidate` subprocesses in the clean environment (PATH, PYTHONPATH, SYSTEMROOT, TEMP, TMP), all
six at once, 5 trials each with seed 7, in one module-scoped fixture (about 35 s of wall clock on the
demo laptop against a 60 s budget; the fixture prints the time it took). The fast tests check every
fixture under tests/fixtures/llm/ with `check_source` and keep the programs short. Nothing here touches
xAI, Daytona, the arm or a camera, and nothing is written into the repo (outputs go to pytest's tmp dir).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest
from jsonschema import Draft202012Validator

from sim.candidate_check import check_source

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLM_FIXTURES = ROOT / "tests" / "fixtures" / "llm"
JUICE_DIR = LLM_FIXTURES / "candidates" / "juice_box_to_tray"
REPAIR_DIR = LLM_FIXTURES / "repair" / "juice_box_to_tray"
ROUTER_FIXTURE = LLM_FIXTURES / "router" / "learn_juice_box_to_tray.json"
SCENE_PATH = ROOT / "config" / "fake_scene.json"
SCORE_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "score.json").read_text(encoding="utf-8"))
)

INDICES = (1, 2, 3, 4, 5, 6)
TUNED_TO_PASS = (3, 4, 6)  # hints 3, 4 and 6: transit height, route around, far slots first
TRIALS = 5
SEED = "7"
RUN_LIMIT_S = 240.0  # a safety net on communicate(); the six runs together take about 35 s
MAX_FIXTURE_LINES = 60

FIXTURE_FILES = sorted(LLM_FIXTURES.rglob("*.py"))
CANDIDATE_FILES = [p for p in FIXTURE_FILES if p.parent.name != "rewards"]
REPAIR_FILES = sorted(REPAIR_DIR.glob("cand_*.py"))


def clean_env() -> dict[str, str]:
    """The environment the orchestrator's local runner gives AI-written code (brief section 1)."""
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    return env


def entry_for(path: pathlib.Path) -> str:
    """`reward` for the reward fixtures, `run` for every candidate."""
    return "reward" if path.parent.name == "rewards" else "run"


def rel(path: pathlib.Path) -> str:
    return path.relative_to(LLM_FIXTURES).as_posix()


class Run:
    """One `python -m sim.run_candidate` subprocess scoring cand_<index>.py, and what it left behind."""

    def __init__(self, index: int, task: pathlib.Path, out: pathlib.Path) -> None:
        self.index = index
        self.out = out
        self.argv = [
            sys.executable,
            "-m",
            "sim.run_candidate",
            "--candidate",
            str(JUICE_DIR / f"cand_{index}.py"),
            "--task",
            str(task),
            "--scene",
            str(SCENE_PATH),
            "--trials",
            str(TRIALS),
            "--seed",
            SEED,
            "--out",
            str(out),
        ]
        self.proc: subprocess.Popen | None = None
        self.stdout = ""
        self.stderr = ""
        self.returncode: int | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.argv,
            cwd=ROOT,
            env=clean_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def finish(self) -> None:
        assert self.proc is not None
        self.stdout, self.stderr = self.proc.communicate(timeout=RUN_LIMIT_S)
        self.returncode = self.proc.returncode

    @property
    def score(self) -> dict:
        lines = [line for line in self.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1])

    def trials(self) -> list[dict]:
        return json.loads((self.out / "trials.json").read_text(encoding="utf-8"))

    def tail(self) -> str:
        return self.stderr[-2000:]


@pytest.fixture(scope="module")
def juice_runs(tmp_path_factory) -> dict[int, Run]:
    base = tmp_path_factory.mktemp("fixture_candidates")
    spec = json.loads(ROUTER_FIXTURE.read_text(encoding="utf-8"))["spec"]
    task = base / "task.json"
    task.write_text(json.dumps(spec), encoding="utf-8")
    runs = {i: Run(i, task, base / f"cand_{i}") for i in INDICES}
    started = time.perf_counter()
    for run in runs.values():  # all six at once, like the local runner
        run.start()
    for run in runs.values():
        run.finish()
    elapsed = time.perf_counter() - started
    print(f"six juice_box_to_tray fixtures scored in {elapsed:.1f} s of wall clock")
    return runs


# ---------------------------------------------------------------- the runner's verdicts


def test_every_run_exits_0_with_one_valid_score_line(juice_runs):
    for index, run in juice_runs.items():
        assert run.returncode == 0, (index, run.tail())
        lines = [line for line in run.stdout.splitlines() if line.strip()]
        assert len(lines) == 1, (index, run.stdout)
        score = run.score
        SCORE_VALIDATOR.validate(score)
        assert score["candidate"] == index  # --index defaults to the N of cand_N.py
        assert score["trials"] == TRIALS
        assert score["error"] is None, (index, score)
        assert [t["trial"] for t in run.trials()] == list(range(1, TRIALS + 1))


def test_at_least_three_juice_fixtures_pass(juice_runs):
    """Brief section 6: the fake end-to-end path needs at least three passing candidates."""
    passing = sorted(i for i, run in juice_runs.items() if run.score["passed"])
    assert len(passing) >= 3, {i: run.score["fail_reason"] for i, run in juice_runs.items()}


@pytest.mark.parametrize("index", TUNED_TO_PASS)
def test_tuned_fixture_passes_5_of_5_without_collisions(juice_runs, index):
    score = juice_runs[index].score
    assert score["passed"] is True, (score, juice_runs[index].tail())
    assert (score["successes"], score["nominal_success"]) == (TRIALS, True)
    assert (score["collisions"], score["limit_hits"]) == (0, 0)
    assert score["fail_reason"] is None
    assert score["traj"] and (juice_runs[index].out / "traj.json").is_file()
    assert all(t["success"] and t["first_collision"] is None for t in juice_runs[index].trials())


def test_cand_1_without_transit_hits_the_cereal_box(juice_runs):
    """Hint 1 (default parameters): the low sweep from the juice box to the tray hits the obstacle."""
    score = juice_runs[1].score
    assert score["passed"] is False
    assert score["fail_reason"] and "cereal_box" in score["fail_reason"], score
    assert score["collisions"] > 0


def test_cand_2_sets_the_box_down_beside_the_tray(juice_runs):
    """Hint 2: a plausible wrong plan that touches nothing and fails only the success test."""
    score = juice_runs[2].score
    assert score["passed"] is False
    assert score["fail_reason"] and "spec not satisfied" in score["fail_reason"], score
    assert (score["successes"], score["collisions"], score["limit_hits"]) == (0, 0, 0)
    assert all(t["error"] is None and t["first_collision"] is None for t in juice_runs[2].trials())


def test_cand_5_is_a_valid_program_that_runs_every_trial(juice_runs):
    """Hint 5 (slow, pitch -pi/4 beyond 0.30 m) may pass or fail; it must never error out."""
    score = juice_runs[5].score
    assert score["error"] is None, score
    assert all(t["error"] is None for t in juice_runs[5].trials())


# ---------------------------------------------------------------- fast static checks


def test_fixture_tree_is_complete():
    assert len(CANDIDATE_FILES) == 18, [rel(p) for p in CANDIDATE_FILES]
    assert len(REPAIR_FILES) == 6, [p.name for p in REPAIR_FILES]
    assert len(FIXTURE_FILES) - len(CANDIDATE_FILES) == 4  # the reward variants


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=[rel(p) for p in FIXTURE_FILES])
def test_every_llm_fixture_passes_check_source(path):
    assert check_source(path.read_text(encoding="utf-8"), entry=entry_for(path)) == []


@pytest.mark.parametrize("path", REPAIR_FILES, ids=[p.name for p in REPAIR_FILES])
def test_every_repair_fixture_passes_check_source(path):
    source = path.read_text(encoding="utf-8")
    assert check_source(source) == []
    assert "transit_z=" in source


@pytest.mark.parametrize("path", CANDIDATE_FILES, ids=[rel(p) for p in CANDIDATE_FILES])
def test_candidate_fixtures_stay_short(path):
    assert len(path.read_text(encoding="utf-8").splitlines()) < MAX_FIXTURE_LINES
