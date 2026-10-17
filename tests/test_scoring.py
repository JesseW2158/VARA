"""C4: `sim.run_candidate` and `sim.scoring`.

The end-to-end cases run the CLI through `subprocess` with the clean environment the orchestrator's local
runner uses (PATH, PYTHONPATH, SYSTEMROOT, TEMP, TMP) and `sys.executable`; they are started together in one
module-scoped fixture (each is one parent plus one spawned MuJoCo child, so the file stays well under two
minutes on a laptop) and then inspected by the tests below. The unit cases call the scoring functions
directly. Nothing here touches xAI, Daytona, the arm or a camera.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time

import pytest
from jsonschema import Draft202012Validator

from arm import ik, replay
from sim import scoring, success
from sim.run_candidate import candidate_index

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "scoring"
STARTER = ROOT / "skills" / "_starter"
SCENE_PATH = ROOT / "config" / "fake_scene.json"
FAKE_SCENE = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
TIMEOUT_S = 5.0  # --timeout of the infinite-loop case
REFUSED = {"unknown_object"}  # cases the runner refuses with exit 2 and no score line
RUN_LIMIT_S = 240.0  # a safety net on communicate(), far above anything a case needs
TASK_SPEC = {
    "name": "juice_box_to_tray",
    "description": "Put the juice box into the tray.",
    "params": {},
    "success": {"in_zone": {"object": "juice_box", "zone": "tray"}},
    "english": "The juice box ends inside the tray.",
}


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8")))


SCORE_VALIDATOR = _validator("score.json")
TRAJ_VALIDATOR = _validator("trajectory.json")
SCENE_VALIDATOR = _validator("scene.json")


def clean_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    return env


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid still runs (no psutil in the venv)."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Run:
    """One `python -m sim.run_candidate` invocation and everything it left behind."""

    def __init__(
        self, name: str, argv: list[str], out: pathlib.Path, scene: pathlib.Path = SCENE_PATH
    ) -> None:
        self.name = name
        self.out = out
        self.argv = [sys.executable, "-m", "sim.run_candidate", *argv]
        self.argv += ["--scene", str(scene), "--seed", "7", "--out", str(out)]
        self.proc: subprocess.Popen | None = None
        self.stdout = ""
        self.stderr = ""
        self.returncode = -1
        self.elapsed = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()
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
        self.elapsed = time.perf_counter() - self.t0

    @property
    def score(self) -> dict:
        return json.loads(self.stdout.strip().splitlines()[-1])

    def trials(self) -> list:
        return json.loads((self.out / "trials.json").read_text(encoding="utf-8"))

    def out_json(self, name: str) -> dict:
        return json.loads((self.out / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Run]:
    base = tmp_path_factory.mktemp("scoring")
    task = base / "task.json"
    task.write_text(json.dumps(TASK_SPEC), encoding="utf-8")
    cand_4 = base / "cand_4.py"  # the --index default comes from the file name
    shutil.copy(FIXTURES / "drop_beside.py", cand_4)
    banana = base / "scene_banana.json"  # a valid scene naming an object the MJCF does not have
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    scene["objects"]["banana"] = dict(scene["objects"]["tea_tin"], marker=13, x=0.10, y=0.10)
    banana.write_text(json.dumps(scene), encoding="utf-8")

    def skill(name: str, obj: str) -> Run:
        argv = [
            "--skill",
            str(STARTER / f"{name}.json"),
            "--args",
            json.dumps({"object": obj}),
            "--trials",
            "5",
        ]
        return Run(name, argv, base / name)

    def candidate(name: str, path: pathlib.Path, trials: int, *extra: str) -> Run:
        argv = ["--candidate", str(path), "--task", str(task), "--trials", str(trials), *extra]
        return Run(name, argv, base / name)

    cases = [
        skill("place_in_tray", "juice_box"),
        skill("hand_over", "juice_box"),
        candidate("drop_beside", cand_4, 2),
        candidate("no_transit", FIXTURES / "no_transit.py", 1, "--index", "2"),
        candidate("trial1_fails", FIXTURES / "trial1_fails.py", 5),
        candidate("infinite_loop", FIXTURES / "infinite_loop.py", 5, "--timeout", str(TIMEOUT_S)),
        candidate("burn_time", FIXTURES / "burn_time.py", 1),
        candidate("rejected", FIXTURES / "rejected.py", 5),
        candidate("kbd_interrupt", FIXTURES / "kbd_interrupt.py", 2),
        candidate("bad_messages", FIXTURES / "bad_messages.py", 2),
        Run(
            "unknown_object",
            ["--candidate", str(FIXTURES / "no_transit.py"), "--task", str(task), "--trials", "1"],
            base / "unknown_object",
            scene=banana,
        ),
    ]
    for run in cases:
        run.start()
    for run in cases:
        run.finish()
    return {run.name: run for run in cases}


# ---------------------------------------------------------------- end to end through the CLI


def test_every_run_prints_exactly_one_valid_score_line_and_exits_0(runs):
    for name, run in runs.items():
        if name in REFUSED:
            continue
        assert run.returncode == 0, (name, run.stderr[-2000:])
        lines = [line for line in run.stdout.splitlines() if line.strip()]
        assert len(lines) == 1, (name, run.stdout)
        score = json.loads(lines[0])
        SCORE_VALIDATOR.validate(score)
        assert run.out_json("score.json") == score, name
        assert isinstance(run.trials(), list), name
        for trial in run.trials():
            assert tuple(trial) == scoring.TRIAL_KEYS, (name, trial)


@pytest.mark.parametrize("name", ["place_in_tray", "hand_over"])
def test_starter_skill_passes_5_of_5(runs, name):
    run = runs[name]
    score = run.score
    assert score["passed"] is True, (score, run.stderr[-2000:])
    assert (score["successes"], score["trials"], score["nominal_success"]) == (5, 5, True)
    assert (score["collisions"], score["limit_hits"], score["ik_failures"]) == (0, 0, 0)
    assert score["fail_reason"] is None and score["error"] is None
    assert score["candidate"] == 0  # the --skill form has no cand_N.py to read an index from
    assert 1.0 < score["mean_sim_s"] < 60.0
    assert score["traj"] == (run.out / "traj.json").as_posix()
    assert (run.out / "traj.json").is_file() and (run.out / "final_scene.json").is_file()
    trials = run.trials()
    assert [t["trial"] for t in trials] == [1, 2, 3, 4, 5]
    assert all(t["success"] and t["error"] is None and t["first_collision"] is None for t in trials)
    assert score["mean_sim_s"] == pytest.approx(sum(t["sim_s"] for t in trials) / 5, abs=1e-3)
    starter = json.loads((STARTER / f"{name}.json").read_text(encoding="utf-8"))
    assert starter["sim"] == {"successes": score["successes"], "trials": score["trials"]}  # A12 counts


def obstacle_poses(scene: dict) -> dict[str, tuple[float, float]]:
    return {name: (item["x"], item["y"]) for name, item in scene["obstacles"].items()}


def assert_trial_1_files(run: Run, traj: dict, final: dict) -> None:
    """traj.json and final_scene.json are trial 1's (the scene as observed), never a perturbed trial's:
    obstacles are mocap bodies, so trial 1 leaves them exactly where the scene file puts them while every
    perturbed trial moves them (seed 7 shifts the cereal box by millimetres), and the recorded path ticks
    plus the gripper ramps add up to trial 1's simulated time (a live segment records no ticks)."""
    assert obstacle_poses(final) == obstacle_poses(FAKE_SCENE)
    for k in range(2, 6):
        assert obstacle_poses(scoring.perturb_scene(FAKE_SCENE, k, 7)) != obstacle_poses(FAKE_SCENE)
    if all(seg["kind"] != "live" for seg in traj["segments"]):
        ticks = sum(len(seg["q"]) for seg in traj["segments"] if seg["kind"] == "path")
        grips = [seg for seg in traj["segments"] if seg["kind"] == "grip"]
        ramps = sum(0.4 if seg["action"] == "close" else 0.3 for seg in grips)
        assert ticks * traj["dt"] + ramps == pytest.approx(run.trials()[0]["sim_s"], abs=1e-3)


def test_trial_1_outputs_validate_and_replay_accepts_the_trajectory(runs):
    run = runs["place_in_tray"]
    traj = run.out_json("traj.json")
    TRAJ_VALIDATOR.validate(traj)
    assert traj["skill"] == "place_in_tray" and traj["args"] == {"object": "juice_box"}
    assert replay.validate(traj, replay.load_limits(), ik.fk_tip, replay.load_box()) == []
    final = run.out_json("final_scene.json")
    SCENE_VALIDATOR.validate(final)
    spec = json.loads((STARTER / "place_in_tray.json").read_text(encoding="utf-8"))["spec"]
    assert success.evaluate(spec, FAKE_SCENE, final, {"object": "juice_box"}, []) is True
    assert final["t"] > FAKE_SCENE["t"]  # the sim clock advanced
    assert_trial_1_files(run, traj, final)
    run = runs["hand_over"]
    traj = run.out_json("traj.json")
    TRAJ_VALIDATOR.validate(traj)
    assert traj["segments"][-1] == {"kind": "live", "primitive": "carry_to_hand", "object": "juice_box"}
    assert replay.validate(traj, replay.load_limits(), ik.fk_tip, replay.load_box()) == []
    final = run.out_json("final_scene.json")
    SCENE_VALIDATOR.validate(final)
    assert final["objects"]["juice_box"]["seen"] is False  # the user took it
    assert_trial_1_files(run, traj, final)


def test_drop_beside_fails_the_spec_and_the_candidate_prints_only_to_stderr(runs):
    run = runs["drop_beside"]
    score = run.score
    assert score["passed"] is False
    assert score["fail_reason"] == "trial 1: spec not satisfied"
    assert score["nominal_success"] is False and score["error"] is None
    assert score["collisions"] == 0 and score["successes"] == 0 and score["trials"] == 2
    assert score["candidate"] == 4  # cand_4.py
    assert score["traj"] == (run.out / "traj.json").as_posix()
    assert "drop_beside: dropping it beside the tray" in run.stderr
    assert "dropping it beside" not in run.stdout  # the out dir is named drop_beside, so match the text
    assert len(run.stdout.strip().splitlines()) == 1


def test_candidate_without_transit_fails_naming_the_cereal_box(runs):
    run = runs["no_transit"]
    score = run.score
    assert score["passed"] is False
    assert score["fail_reason"].startswith("trial 1: ") and "cereal_box" in score["fail_reason"]
    assert score["collisions"] >= 1 and score["error"] is None
    assert score["candidate"] == 2  # --index wins over the file name
    assert run.trials()[0]["first_collision"].endswith("hit cereal_box")


def test_a_candidate_whose_trial_1_fails_never_passes(runs):
    run = runs["trial1_fails"]
    score = run.score
    assert score["passed"] is False
    assert score["nominal_success"] is False
    assert score["successes"] == 4 and score["trials"] == 5
    assert score["fail_reason"] == "trial 1: spec not satisfied"
    assert score["error"] is None and score["collisions"] == 0
    assert [t["success"] for t in run.trials()] == [False, True, True, True, True]


def test_infinite_loop_ends_as_a_timeout_within_the_limit_and_leaves_no_orphan(runs):
    run = runs["infinite_loop"]
    assert run.returncode == 0
    score = run.score
    assert score["error"] == "timeout" and score["fail_reason"] == "trial 1: timeout"
    assert score["passed"] is False and score["nominal_success"] is False
    assert (score["successes"], score["trials"]) == (0, 5)
    assert score["mean_sim_s"] is None and score["traj"] is None
    trials = run.trials()
    assert len(trials) == 1 and trials[0]["error"] == "timeout" and trials[0]["sim_s"] is None
    waited = re.search(r"timeout after ([0-9.]+) s", run.stderr)
    assert waited is not None, run.stderr
    assert TIMEOUT_S <= float(waited.group(1)) < TIMEOUT_S + 3.0
    # killed by the timeout itself, not by the post-loop or shutdown kill seconds later
    killed = re.search(r"child pid (\d+) killed \(trial 1 timeout\); exit code -?\d+; no orphan", run.stderr)
    assert killed is not None, run.stderr
    pid = int(re.search(r"child pid (\d+)", run.stderr).group(1))
    assert int(killed.group(1)) == pid
    assert not pid_alive(pid)
    assert not (run.out / "traj.json").exists()


def test_burning_simulated_time_hits_the_limit_and_ends_as_an_error(runs):
    run = runs["burn_time"]
    score = run.score
    assert score["passed"] is False
    assert score["error"].startswith("SimTimeLimit")
    assert score["fail_reason"].startswith("trial 1: error: SimTimeLimit")
    trial = run.trials()[0]
    assert trial["success"] is False and trial["sim_s"] > 60.0


def test_rejected_candidate_gets_the_rejected_score_with_exit_0(runs):
    run = runs["rejected"]
    assert run.returncode == 0
    score = run.score
    assert score["error"].startswith("rejected:") and "import" in score["error"]
    assert score["fail_reason"] == score["error"]
    assert score["passed"] is False and score["nominal_success"] is False
    assert (score["successes"], score["trials"]) == (0, 5)
    assert score["mean_sim_s"] is None and score["traj"] is None
    assert (score["collisions"], score["limit_hits"], score["ik_failures"]) == (0, 0, 0)
    assert run.trials() == []
    assert "child pid" not in run.stderr  # nothing was executed


def test_a_keyboard_interrupt_raised_by_the_candidate_is_its_trial_error(runs):
    """Review round 1: `raise KeyboardInterrupt` ended the runner with exit 1 and no score line."""
    run = runs["kbd_interrupt"]
    assert run.returncode == 0, run.stderr[-2000:]
    score = run.score
    assert score["error"] == "KeyboardInterrupt: the candidate pretends to be Ctrl+C"
    assert score["fail_reason"] == "trial 1: error: " + score["error"]
    assert score["passed"] is False and (score["successes"], score["trials"]) == (0, 2)
    assert [t["error"] for t in run.trials()] == [score["error"]] * 2  # the child went on to trial 2
    assert "runner failed" not in run.stderr


def test_unprintable_and_huge_exception_messages_become_short_trial_errors(runs):
    """Review round 1: str() of a deeply nested value raised RecursionError inside the runner (exit 1), and
    a 5 MB message produced a 10 MB score line."""
    run = runs["bad_messages"]
    assert run.returncode == 0, run.stderr[-2000:]
    assert len(run.stdout) < 2000
    score = run.score
    trials = run.trials()
    assert [t["success"] for t in trials] == [False, False]
    assert trials[0]["error"] == scoring.clip("ValueError: " + "x" * 1000000)
    assert trials[0]["error"].endswith(" ... (clipped from 1000012 characters)")
    assert len(trials[0]["error"]) == scoring.MAX_TEXT_CHARS  # clipped once, by child and parent alike
    assert trials[1]["error"] == "ValueError: <message not printable: RecursionError>"
    assert score["error"] == trials[0]["error"]
    assert score["fail_reason"] == "trial 1: error: " + trials[0]["error"]
    assert "runner failed" not in run.stderr


def test_a_scene_naming_an_object_the_model_lacks_is_a_usage_error(runs):
    """Review round 1: the simulator's ValueError surfaced as 'runner failed' (exit 1)."""
    run = runs["unknown_object"]
    assert run.returncode == 2, run.stderr[-2000:]
    assert run.stdout.strip() == ""
    assert "scene names objects the model does not have" in run.stderr and "banana" in run.stderr
    assert "runner failed" not in run.stderr
    assert not (run.out / "score.json").exists() and not (run.out / "trials.json").exists()


def test_non_utf8_candidates_and_non_finite_json_are_usage_errors(tmp_path):
    """Review round 1: a candidate file that is not UTF-8 ended in an uncaught traceback (exit 1), and NaN
    in a scene passed the schema and reached MuJoCo. Four parent-only runs, started together."""
    task = tmp_path / "task.json"
    task.write_text(json.dumps(TASK_SPEC), encoding="utf-8")
    bad = tmp_path / "cand_1.py"
    bad.write_bytes(b"import math\n\ndef run(robot, args):\n    return True  # \xff\xfe\n")
    text = SCENE_PATH.read_text(encoding="utf-8")
    assert text.count('"x": 0.24,') == 1
    nan_scene = tmp_path / "scene.json"
    nan_scene.write_text(text.replace('"x": 0.24,', '"x": NaN,'), encoding="utf-8")
    good = str(FIXTURES / "no_transit.py")
    cases = {
        "utf8": (["--candidate", str(bad), "--task", str(task)], SCENE_PATH, "is not UTF-8"),
        "nan_scene": (["--candidate", good, "--task", str(task)], nan_scene, "NaN is not valid JSON"),
        "nan_args": (
            ["--candidate", good, "--task", str(task), "--args", '{"object": NaN}'],
            SCENE_PATH,
            "--args is not JSON: NaN is not valid JSON",
        ),
        "inf_args": (
            ["--candidate", good, "--task", str(task), "--args", '{"x": 1e999}'],
            SCENE_PATH,
            "1e999 is not a finite number",
        ),
    }
    runs = {}
    for name, (argv, scene, _marker) in cases.items():
        runs[name] = Run(name, [*argv, "--trials", "1"], tmp_path / name, scene=scene)
        runs[name].start()
    for name, (_argv, _scene, marker) in cases.items():
        run = runs[name]
        run.finish()
        assert run.returncode == 2, (name, run.stderr[-1000:])
        assert run.stdout.strip() == "", name
        assert marker in run.stderr, (name, run.stderr[-1000:])
        assert "child pid" not in run.stderr, name  # refused before anything was spawned
        assert not run.out.exists(), name


# ---------------------------------------------------------------- perturb_scene


def test_perturb_scene_trial_1_is_an_untouched_copy():
    out = scoring.perturb_scene(FAKE_SCENE, 1, 7)
    assert out == FAKE_SCENE and out is not FAKE_SCENE
    out["objects"]["juice_box"]["x"] = 9.0
    out["zones"]["tray"]["slots"][0][0] = 9.0
    assert FAKE_SCENE == json.loads(SCENE_PATH.read_text(encoding="utf-8"))


def test_perturb_scene_is_deterministic_bounded_and_leaves_the_input_alone():
    pristine = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    for trial in range(2, 6):
        a = scoring.perturb_scene(FAKE_SCENE, trial, 7)
        assert a == scoring.perturb_scene(FAKE_SCENE, trial, 7)
        assert a == scoring.perturb_scene(FAKE_SCENE, trial, "7")  # the CLI passes the seed as a string
        SCENE_VALIDATOR.validate(a)
        for name, item in FAKE_SCENE["objects"].items():
            got = a["objects"][name]
            assert abs(got["x"] - item["x"]) <= 0.01 and abs(got["y"] - item["y"]) <= 0.01
            assert abs(got["yaw"] - item["yaw"]) <= math.radians(10) + 1e-12
            assert (got["x"], got["y"], got["yaw"]) != (item["x"], item["y"], item["yaw"])
            assert {k: v for k, v in got.items() if k not in ("x", "y", "yaw")} == {
                k: v for k, v in item.items() if k not in ("x", "y", "yaw")
            }
        for name, item in FAKE_SCENE["obstacles"].items():
            got = a["obstacles"][name]
            assert abs(got["x"] - item["x"]) <= 0.005 and abs(got["y"] - item["y"]) <= 0.005
            assert (got["x"], got["y"]) != (item["x"], item["y"])
            assert got["yaw"] == item["yaw"]
        assert (a["t"], a["hand"], a["zones"]) == (FAKE_SCENE["t"], FAKE_SCENE["hand"], FAKE_SCENE["zones"])
    assert scoring.perturb_scene(FAKE_SCENE, 2, 7) != scoring.perturb_scene(FAKE_SCENE, 3, 7)
    assert scoring.perturb_scene(FAKE_SCENE, 2, 7) != scoring.perturb_scene(FAKE_SCENE, 2, 8)
    assert FAKE_SCENE == pristine
    with pytest.raises(ValueError):
        scoring.perturb_scene(FAKE_SCENE, 0, 7)


def test_perturb_scene_draws_exactly_the_shared_recipe():
    """random.Random(f"{seed}:{trial}"): x, y, yaw per object in scene order, then x, y per obstacle, so
    every candidate (and tests/test_sim_robot.py) sees the same five scenes."""
    rng = random.Random("7:3")
    expected = json.loads(json.dumps(FAKE_SCENE))
    for item in expected["objects"].values():
        item["x"] += rng.uniform(-0.01, 0.01)
        item["y"] += rng.uniform(-0.01, 0.01)
        item["yaw"] += math.radians(rng.uniform(-10, 10))
    for item in expected["obstacles"].values():
        item["x"] += rng.uniform(-0.005, 0.005)
        item["y"] += rng.uniform(-0.005, 0.005)
    assert scoring.perturb_scene(FAKE_SCENE, 3, 7) == expected


# ---------------------------------------------------------------- trial_result, fail_reason, summarize


def trial(k: int, success: bool = True, **over) -> dict:
    stats = {"collisions": 0, "limit_hits": 0, "ik_failures": 0, "sim_s": 10.0 + k, "first_collision": None}
    result = scoring.trial_result(k, success, stats, None)
    result.update(over)
    return result


def test_trial_result_shape():
    stats = {
        "collisions": 2,
        "limit_hits": 1,
        "ik_failures": 3,
        "sim_s": 12.5,
        "first_collision": "gripper hit hand",
    }
    full = scoring.trial_result(3, True, stats, None)
    assert tuple(full) == scoring.TRIAL_KEYS
    assert full == {
        "trial": 3,
        "success": True,
        "collisions": 2,
        "limit_hits": 1,
        "ik_failures": 3,
        "sim_s": 12.5,
        "error": None,
        "first_collision": "gripper hit hand",
    }
    empty = scoring.trial_result(2, False, None, scoring.TIMEOUT)
    assert empty == {
        "trial": 2,
        "success": False,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "sim_s": None,
        "error": "timeout",
        "first_collision": None,
    }


def test_non_finite_sim_times_count_as_unreported():
    """Unreachable from SimRobot (its clock is ticks / 30), but a NaN would poison the mean and the JSON
    score line, so the summarizer treats it like a trial that reported no time."""
    assert scoring.trial_result(1, True, {"sim_s": float("nan")}, None)["sim_s"] is None
    assert scoring.trial_result(1, True, {"sim_s": float("inf")}, None)["sim_s"] is None
    assert scoring.trial_result(1, True, {"sim_s": 12}, None)["sim_s"] == 12.0
    odd = [trial(1), dict(trial(2), sim_s=float("nan")), dict(trial(3), sim_s=float("-inf"))]
    score = scoring.summarize(0, odd, None)
    SCORE_VALIDATOR.validate(score)
    assert score["passed"] is True and score["mean_sim_s"] == 11.0
    assert "NaN" not in json.dumps(score) and "Infinity" not in json.dumps(score)
    assert scoring.summarize(0, [dict(trial(1), sim_s=float("nan"))], None)["mean_sim_s"] is None


def test_fail_reason_phrases_and_precedence():
    assert scoring.fail_reason(trial(1)) is None
    assert scoring.fail_reason(trial(2, False)) == "trial 2: spec not satisfied"
    assert scoring.fail_reason(trial(4, True, limit_hits=1)) == "trial 4: limit hit"
    hit = trial(3, True, collisions=1, first_collision="gripper hit cereal_box")
    assert scoring.fail_reason(hit) == "trial 3: gripper hit cereal_box"
    assert scoring.fail_reason(trial(1, False, error="ZeroDivisionError: division by zero")) == (
        "trial 1: error: ZeroDivisionError: division by zero"
    )
    assert scoring.fail_reason(trial(1, False, error="timeout")) == "trial 1: timeout"
    # inside one trial: timeout, error, collision, limit hit, spec
    both = trial(
        2, False, collisions=1, first_collision="held juice_box hit tray", limit_hits=1, error="boom"
    )
    assert scoring.fail_reason(both) == "trial 2: error: boom"
    both["error"] = None
    assert scoring.fail_reason(both) == "trial 2: held juice_box hit tray"
    both["collisions"] = 0
    assert scoring.fail_reason(both) == "trial 2: limit hit"
    both["limit_hits"] = 0
    assert scoring.fail_reason(both) == "trial 2: spec not satisfied"
    assert scoring.fail_reason(trial(5, True, collisions=1)) == "trial 5: collision"  # no phrase recorded


def test_summarize_passes_a_clean_candidate_and_tolerates_one_perturbed_miss():
    good = [trial(k) for k in range(1, 6)]
    score = scoring.summarize(3, good, "out/traj.json")
    SCORE_VALIDATOR.validate(score)
    assert score == {
        "candidate": 3,
        "passed": True,
        "successes": 5,
        "trials": 5,
        "nominal_success": True,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "mean_sim_s": 13.0,
        "fail_reason": None,
        "traj": "out/traj.json",
        "error": None,
    }
    four = [trial(1), trial(2), trial(3), trial(4, False), trial(5)]
    score = scoring.summarize(1, four, "t.json")
    assert score["passed"] is True and score["successes"] == 4 and score["fail_reason"] is None
    one = scoring.summarize(0, [trial(1)], None)  # the planner's single trial
    SCORE_VALIDATOR.validate(one)
    assert one["passed"] is True and one["trials"] == 1 and one["traj"] is None and one["mean_sim_s"] == 11.0
    assert scoring.summarize(0, [trial(1, False)], None)["passed"] is False
    assert scoring.required_successes(5) == 4 and scoring.required_successes(1) == 1
    assert scoring.required_successes(2) == 2 and scoring.required_successes(10) == 8


def test_summarize_fail_rules_and_reasons():
    # trial 1 failing never passes, however the rest went
    score = scoring.summarize(2, [trial(1, False)] + [trial(k) for k in range(2, 6)], "t")
    assert score["passed"] is False and score["nominal_success"] is False and score["successes"] == 4
    assert score["fail_reason"] == "trial 1: spec not satisfied" and score["error"] is None
    # 3 of 5: the first missed trial is the reason
    score = scoring.summarize(2, [trial(1), trial(2, False), trial(3), trial(4, False), trial(5)], "t")
    assert score["passed"] is False and score["fail_reason"] == "trial 2: spec not satisfied"
    # a collision anywhere fails, with the body phrase, even with 5 successes and a tolerated miss before it
    hit = trial(3, True, collisions=2, first_collision="gripper hit cereal_box")
    score = scoring.summarize(2, [trial(1), trial(2, False), hit, trial(4), trial(5)], "t")
    assert score["passed"] is False and score["fail_reason"] == "trial 3: gripper hit cereal_box"
    assert score["collisions"] == 2 and score["successes"] == 4
    # a limit hit
    score = scoring.summarize(2, [trial(1), trial(2), trial(3), trial(4, limit_hits=1), trial(5)], "t")
    assert (
        score["passed"] is False and score["fail_reason"] == "trial 4: limit hit" and score["limit_hits"] == 1
    )
    # an error: the message is both in the reason and in `error`, ik_failures are summed, not judged
    err = trial(2, False, error="RuntimeError: reset() is not available to AI code", ik_failures=2)
    score = scoring.summarize(2, [trial(1, ik_failures=1), err, trial(3), trial(4), trial(5)], "t")
    assert score["passed"] is False
    assert score["fail_reason"] == "trial 2: error: RuntimeError: reset() is not available to AI code"
    assert score["error"] == "RuntimeError: reset() is not available to AI code" and score["ik_failures"] == 3
    # a timeout at trial 2 of 5: the later trials did not run; the mean covers the trial that did
    cut = [trial(1), scoring.trial_result(2, False, None, scoring.TIMEOUT)]
    score = scoring.summarize(5, cut, "t", trials=5)
    SCORE_VALIDATOR.validate(score)
    assert score["passed"] is False and score["trials"] == 5 and score["successes"] == 1
    assert score["fail_reason"] == "trial 2: timeout" and score["error"] == "timeout"
    assert score["mean_sim_s"] == 11.0 and score["nominal_success"] is True
    # trials that silently did not run cannot pass either
    score = scoring.summarize(5, [trial(1), trial(2)], "t", trials=5)
    assert score["passed"] is False and score["fail_reason"] == "trial 3: not run"
    assert scoring.summarize(5, [], "t", trials=5)["fail_reason"] == "trial 1: not run"
    with pytest.raises(ValueError):
        scoring.summarize(0, [], None)
    with pytest.raises(ValueError):
        scoring.summarize(0, [trial(1), trial(2)], None, trials=1)


def test_clip_bounds_error_texts_wherever_they_enter_a_score():
    """Review round 1: error and fail_reason carried a candidate's 5 MB message twice."""
    assert scoring.clip("short") == "short" and scoring.clip("") == ""
    exact = "y" * scoring.MAX_TEXT_CHARS
    assert scoring.clip(exact) == exact
    clipped = scoring.clip(exact + "z")
    assert len(clipped) == scoring.MAX_TEXT_CHARS and clipped.startswith("y" * 400)
    assert clipped.endswith(" ... (clipped from 501 characters)")
    assert scoring.clip(clipped) == clipped  # idempotent: the child clips, then the parent clips again
    assert scoring.clip("abcdef", 3) == "abc"  # a limit too small for the note: a plain cut
    huge = scoring.trial_result(1, False, {"first_collision": "z" * 10000}, "e" * 10**6)
    assert huge["error"] == scoring.clip("e" * 10**6) and len(huge["error"]) == scoring.MAX_TEXT_CHARS
    assert huge["first_collision"] == scoring.clip("z" * 10000)
    score = scoring.summarize(0, [huge], None)
    SCORE_VALIDATOR.validate(score)
    assert score["fail_reason"] == "trial 1: error: " + huge["error"]
    assert len(json.dumps(score)) < 2 * scoring.MAX_TEXT_CHARS + 400
    rejected = scoring.rejected_score(0, 5, "p" * 10**6)
    assert rejected["error"] == "rejected: " + scoring.clip("p" * 10**6) and len(rejected["error"]) < 600


def test_rejected_score():
    score = scoring.rejected_score(4, 5, "line 2: only 'import math' is allowed, not 'import os'")
    SCORE_VALIDATOR.validate(score)
    assert score["candidate"] == 4 and score["passed"] is False and score["trials"] == 5
    assert score["error"] == "rejected: line 2: only 'import math' is allowed, not 'import os'"
    assert score["fail_reason"] == score["error"]
    assert score["successes"] == 0 and score["nominal_success"] is False
    assert score["mean_sim_s"] is None and score["traj"] is None


# ---------------------------------------------------------------- rank and --index


def test_rank_orders_passing_then_successes_then_sim_time_then_index():
    def score(candidate: int, passed: bool, successes: int, mean: float | None) -> dict:
        return {"candidate": candidate, "passed": passed, "successes": successes, "mean_sim_s": mean}

    scores = [
        score(6, False, 5, 9.0),  # collided: fails even with 5 successes
        score(5, True, 4, 8.0),
        score(4, True, 5, 30.0),
        score(3, True, 5, 12.0),
        score(2, False, 2, None),
        score(1, True, 5, 12.0),
        score(0, False, 3, 20.0),
    ]
    before = [dict(s) for s in scores]
    ranked = scoring.rank(scores)
    assert [s["candidate"] for s in ranked] == [1, 3, 4, 5, 6, 0, 2]
    assert scores == before  # the input is not reordered
    assert scoring.rank([]) == []


def test_candidate_index_defaults_to_the_number_in_cand_n():
    assert candidate_index("runs/t-0042/cand_3.py") == 3
    assert candidate_index(pathlib.Path("cand_12.py")) == 12
    assert candidate_index("tests/fixtures/candidate_clear_table.py") == 0
    assert candidate_index("cand_x.py") == 0
    assert candidate_index("cand_3.py.bak") == 0
    assert candidate_index(None) == 0
