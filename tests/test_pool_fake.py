"""C5/C6: `agents.sandbox_pool` without Daytona.

`LocalRunner` is exercised on the real runner (`python -m sim.run_candidate` on the MuJoCo sim, the clean
environment); `SandboxPool` against a fake Daytona client that records every call, answers `exec` with
ExecuteResponse-like objects and serves downloads from a dict. Nothing here touches xAI, Daytona, the arm or
a camera, and nothing writes into the repo's `skills/` or `runs/`.
"""

from __future__ import annotations

import ctypes
import gzip
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from agents import sandbox_pool as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
JUICE = FIXTURES / "llm" / "candidates" / "juice_box_to_tray"
JUICE_SPEC = json.loads((FIXTURES / "llm" / "router" / "learn_juice_box_to_tray.json").read_text("utf-8"))[
    "spec"
]
STARTER = ROOT / "skills" / "_starter"
SCENE_PATH = ROOT / "config" / "fake_scene.json"
SCORE_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "score.json").read_text("utf-8")))
TRAJ_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "trajectory.json").read_text("utf-8")))
SCORED_KEYS = {"candidate", "runner", "score", "trial_results", "out_dir"}
SIX_WALL_LIMIT_S = 60.0

MINIMAL_TRAJ = {
    "version": 1,
    "skill": "fake",
    "args": {},
    "dt": 1 / 30,
    "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    "segments": [{"kind": "path", "q": [[0.0, -1.35, 0.02, 1.45, 0.0]]}],
}


# ---------------------------------------------------------------- helpers


def validate_scored(scored: dict, runner: str | None = None) -> None:
    assert set(scored) == SCORED_KEYS, scored
    assert scored["runner"] in ("local", "daytona")
    if runner is not None:
        assert scored["runner"] == runner
    SCORE_VALIDATOR.validate(scored["score"])
    assert scored["score"]["candidate"] == scored["candidate"]
    assert all(isinstance(v, bool) for v in scored["trial_results"])
    assert isinstance(scored["out_dir"], str) and scored["out_dir"]


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


def make_task(base: pathlib.Path, candidates: dict[int, pathlib.Path | str]) -> tuple[pathlib.Path, list]:
    """A task dir with task.json, scene.json and cand_<i>.py (copied from a path or written from text)."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "task.json").write_text(json.dumps(JUICE_SPEC), encoding="utf-8")
    shutil.copy(SCENE_PATH, base / "scene.json")
    pairs = []
    for index, source in candidates.items():
        target = base / f"cand_{index}.py"
        if isinstance(source, pathlib.Path):
            shutil.copy(source, target)
        else:
            target.write_text(source, encoding="utf-8", newline="\n")
        pairs.append((index, target))
    return base, pairs


class RecordingRunner(sp.LocalRunner):
    """A LocalRunner that remembers the pids it started and can replace the command it runs."""

    def __init__(self, replace: list[str] | None = None) -> None:
        self.pids: list[int] = []
        self.argvs: list[list[str]] = []
        self.replace = replace

    def spawn(self, argv: list[str], **kwargs) -> subprocess.Popen:
        self.argvs.append(list(argv))
        proc = super().spawn(self.replace if self.replace is not None else argv, **kwargs)
        self.pids.append(proc.pid)
        return proc


# ---------------------------------------------------------------- LocalRunner on the real runner


@pytest.fixture(scope="module")
def six(tmp_path_factory) -> dict:
    task_dir, pairs = make_task(
        tmp_path_factory.mktemp("six"), {i: JUICE / f"cand_{i}.py" for i in range(1, 7)}
    )
    calls: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    results = sp.LocalRunner().score(
        task_dir,
        list(reversed(pairs)),  # any input order
        task_dir / "task.json",
        task_dir / "scene.json",
        None,
        trials=5,
        seed=7,
        timeout=60,
        on_score=lambda s: calls.append((s["candidate"], time.perf_counter() - t0)),
    )
    wall = time.perf_counter() - t0
    return {"task_dir": task_dir, "results": results, "calls": calls, "wall": wall}


def test_six_fixtures_scored_in_parallel_under_the_wall_clock_limit(six):
    wall = six["wall"]
    print(f"\nsix juice_box_to_tray fixtures x 5 trials on LocalRunner: {wall:.1f} s wall clock")
    assert wall < SIX_WALL_LIMIT_S
    results = six["results"]
    assert [r["candidate"] for r in results] == [1, 2, 3, 4, 5, 6]
    for scored in results:
        validate_scored(scored, "local")
        out_dir = pathlib.Path(scored["out_dir"])
        assert out_dir == six["task_dir"] / f"cand_{scored['candidate']}"
        score = scored["score"]
        assert score["trials"] == 5
        assert not str(score["error"] or "").startswith("runner:"), score
        # the runner's own files and our captured stdout/stderr agree with the scored dict
        assert json.loads((out_dir / "score.json").read_text("utf-8")) == score
        last = (out_dir / "runner.out").read_text("utf-8").strip().splitlines()[-1]
        assert json.loads(last) == score
        assert (out_dir / "runner.log").is_file()
        trials = json.loads((out_dir / "trials.json").read_text("utf-8"))
        assert scored["trial_results"] == [t["success"] for t in trials]
        assert sum(scored["trial_results"]) == score["successes"]
        if score["traj"] is not None:
            assert pathlib.Path(score["traj"]).is_file()
        if score["passed"]:
            assert score["fail_reason"] is None and scored["trial_results"][0] is True
            TRAJ_VALIDATOR.validate(json.loads((out_dir / "traj.json").read_text("utf-8")))
            assert (out_dir / "final_scene.json").is_file()
        else:
            assert score["fail_reason"]
    passed = [r["candidate"] for r in results if r["score"]["passed"]]
    print(f"passing fixtures: {passed}")


def test_on_score_fires_once_per_candidate_as_each_finishes(six):
    calls = six["calls"]
    assert sorted(c for c, _ in calls) == [1, 2, 3, 4, 5, 6]
    assert [t for _, t in calls] == sorted(t for _, t in calls)


def test_on_score_order_is_completion_order_and_listener_errors_are_swallowed(tmp_path):
    # the rejected fixture never starts MuJoCo, so it finishes long before a real candidate
    task_dir, pairs = make_task(tmp_path, {2: JUICE / "cand_3.py", 1: FIXTURES / "scoring" / "rejected.py"})
    order: list[int] = []

    def listener(scored: dict) -> None:
        order.append(scored["candidate"])
        raise RuntimeError("the listener is broken")

    results = sp.LocalRunner().score(
        task_dir, pairs, task_dir / "task.json", task_dir / "scene.json", None, trials=1, on_score=listener
    )
    assert order == [1, 2]
    assert [r["candidate"] for r in results] == [1, 2]
    assert results[0]["score"]["error"].startswith("rejected:") and results[0]["trial_results"] == []
    assert results[1]["score"]["trials"] == 1


def test_cancel_kills_the_children_within_2_s_and_leaves_no_orphan(tmp_path):
    loop = FIXTURES / "scoring" / "infinite_loop.py"
    task_dir, pairs = make_task(tmp_path, {1: loop, 2: loop, 3: loop})
    runner = RecordingRunner()
    cancel = threading.Event()
    box: dict = {}
    thread = threading.Thread(
        target=lambda: box.setdefault(
            "results",
            runner.score(
                task_dir,
                pairs,
                task_dir / "task.json",
                task_dir / "scene.json",
                None,
                trials=5,
                cancel=cancel,
            ),
        )
    )
    thread.start()
    # wait until every runner has started its trial child (the grandchild running the candidate)
    grandchildren: dict[int, int] = {}
    deadline = time.time() + 90
    while time.time() < deadline and len(grandchildren) < 3:
        for index in (1, 2, 3):
            log = task_dir / f"cand_{index}" / "runner.log"
            if log.exists():
                match = re.search(r"child pid (\d+) started", log.read_text("utf-8", "replace"))
                if match:
                    grandchildren[index] = int(match.group(1))
        time.sleep(0.1)
    assert len(grandchildren) == 3, "the runners did not start their trial children"
    assert all(pid_alive(p) for p in runner.pids) and all(pid_alive(p) for p in grandchildren.values())
    t0 = time.perf_counter()
    cancel.set()
    thread.join(10)
    returned = time.perf_counter() - t0
    assert not thread.is_alive()
    assert returned < 2.0, f"score() returned {returned:.2f} s after cancel"
    while time.perf_counter() - t0 < 2.0 and (
        any(pid_alive(p) for p in runner.pids) or any(pid_alive(p) for p in grandchildren.values())
    ):
        time.sleep(0.05)
    gone = time.perf_counter() - t0
    assert not any(pid_alive(p) for p in runner.pids), "a runner survived the cancel"
    assert not any(pid_alive(p) for p in grandchildren.values()), "an orphan trial process survived"
    assert gone < 2.0
    print(f"\ncancel: score() returned after {returned:.2f} s, every process gone after {gone:.2f} s")
    results = box["results"]
    assert [r["candidate"] for r in results] == [1, 2, 3]
    for scored in results:
        validate_scored(scored, "local")
        assert scored["score"]["error"] == "runner: cancelled"
        assert scored["score"]["fail_reason"] == "runner: cancelled"
        assert scored["score"]["passed"] is False and scored["score"]["trials"] == 5
        assert scored["score"]["successes"] == 0 and scored["trial_results"] == []


def test_cancel_already_set_starts_nothing(tmp_path):
    task_dir, pairs = make_task(tmp_path, {1: JUICE / "cand_3.py"})
    runner = RecordingRunner()
    cancel = threading.Event()
    cancel.set()
    results = runner.score(
        task_dir, pairs, task_dir / "task.json", task_dir / "scene.json", None, cancel=cancel
    )
    assert runner.pids == []
    assert results[0]["score"]["error"] == "runner: cancelled"


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (
            "print('{\"candidate\": 1'); print('this is not json')",
            "runner: exit code 0, last line is not JSON",
        ),
        ('print(\'{"candidate": 1, "passed": true}\')', "runner: exit code 0, score line invalid at"),
        ("import sys; print('[1, 2]'); sys.exit(0)", "runner: exit code 0, last line is not a JSON object"),
        ("import sys; sys.exit(1)", "runner: exit code 1, no score line"),
    ],
)
def test_garbage_or_no_score_line_becomes_a_runner_error(tmp_path, code, fragment):
    task_dir, pairs = make_task(tmp_path, {4: JUICE / "cand_3.py"})
    runner = RecordingRunner(replace=[sys.executable, "-c", code])
    results = runner.score(task_dir, pairs, task_dir / "task.json", task_dir / "scene.json", None, trials=5)
    assert len(results) == 1
    scored = results[0]
    validate_scored(scored, "local")
    assert scored["candidate"] == 4
    score = scored["score"]
    assert score["passed"] is False and score["nominal_success"] is False
    assert score["successes"] == 0 and score["trials"] == 5 and score["traj"] is None
    assert score["error"].startswith(fragment), score["error"]
    assert score["fail_reason"] == score["error"]
    assert scored["trial_results"] == []
    # the documented command was what we asked for, even though the test replaced it
    argv = runner.argvs[0]
    assert argv[:3] == [sys.executable, "-m", "sim.run_candidate"]
    assert argv[3:5] == ["--candidate", str(task_dir / "cand_4.py")]


def test_a_runner_over_the_deadline_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "GRACE_S", 0.3)
    task_dir, pairs = make_task(tmp_path, {1: JUICE / "cand_3.py"})
    runner = RecordingRunner(replace=[sys.executable, "-c", "import time; time.sleep(30)"])
    t0 = time.perf_counter()
    results = runner.score(
        task_dir, pairs, task_dir / "task.json", task_dir / "scene.json", None, trials=1, timeout=0.2
    )
    assert time.perf_counter() - t0 < 5.0
    assert results[0]["score"]["error"].startswith("runner: timeout after")
    assert not pid_alive(runner.pids[0])


def test_a_runner_that_cannot_start_is_a_runner_error(tmp_path):
    task_dir, pairs = make_task(tmp_path, {1: JUICE / "cand_3.py"})
    runner = RecordingRunner(replace=[str(tmp_path / "no-such-python.exe"), "-c", "pass"])
    results = runner.score(task_dir, pairs, task_dir / "task.json", task_dir / "scene.json", None)
    assert results[0]["score"]["error"].startswith("runner: could not start:")


def test_clean_env_is_exactly_the_documented_keys():
    env = sp.clean_env()
    expected = {key for key in sp.CLEAN_ENV_KEYS if key in os.environ} | {"PYTHONPATH"}
    assert set(env) == expected
    assert env["PYTHONPATH"] == str(ROOT)
    assert "DAYTONA_API_KEY" not in env and "XAI_API_KEY" not in env
    child = subprocess.run(
        [sys.executable, "-c", "import os, json; print(json.dumps(sorted(os.environ)))"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert {k.upper() for k in json.loads(child.stdout)} == {k.upper() for k in expected}


def test_runner_args_follow_the_documented_order():
    flags = sp.runner_args(
        candidate="c.py",
        task="t.json",
        scene="s.json",
        out="o",
        trials=5,
        seed=7,
        timeout=60,
        index=3,
        skills_dir="sk",
    )
    assert flags == [
        "--candidate", "c.py", "--task", "t.json", "--scene", "s.json", "--trials", "5", "--seed", "7",
        "--out", "o", "--index", "3", "--skills-dir", "sk", "--timeout", "60",
    ]  # fmt: skip
    flags = sp.runner_args(
        skill="k.json", args={"object": "juice_box"}, scene="s", out="o", trials=1, seed=7, timeout=30
    )
    assert flags == [
        "--skill", "k.json", "--args", '{"object": "juice_box"}', "--scene", "s", "--trials", "1",
        "--seed", "7", "--out", "o", "--timeout", "30",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        sp.runner_args(scene="s", out="o", trials=1, seed=7, timeout=1)
    with pytest.raises(ValueError):
        sp.runner_args(candidate="c", scene="s", out="o", trials=1, seed=7, timeout=1)


def test_candidates_may_be_pairs_or_objects():
    pairs = sp._jobs([(1, "a.py"), [2, pathlib.Path("b.py")], SimpleNamespace(index=3, path="c.py")])
    assert pairs == [(1, pathlib.Path("a.py")), (2, pathlib.Path("b.py")), (3, pathlib.Path("c.py"))]
    with pytest.raises(ValueError):
        sp._jobs([SimpleNamespace(index=4, path=None)])


def test_plan_passes_on_both_starters(tmp_path):
    results: dict[str, dict] = {}

    def run(name: str) -> None:
        results[name] = sp.LocalRunner().plan(
            STARTER / f"{name}.json",
            {"object": "juice_box"},
            SCENE_PATH,
            tmp_path / name,
            STARTER,
            timeout=30,
        )

    threads = [threading.Thread(target=run, args=(name,)) for name in ("place_in_tray", "hand_over")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for name, scored in results.items():
        validate_scored(scored, "local")
        score = scored["score"]
        assert score["passed"] is True, (name, score)
        assert (score["candidate"], score["successes"], score["trials"]) == (0, 1, 1)
        assert scored["trial_results"] == [True]
        assert scored["out_dir"] == str(tmp_path / name)
        out_dir = pathlib.Path(scored["out_dir"])
        TRAJ_VALIDATOR.validate(json.loads((out_dir / "traj.json").read_text("utf-8")))
        assert (out_dir / "final_scene.json").is_file()
        assert pathlib.Path(score["traj"]) == out_dir / "traj.json"


# ---------------------------------------------------------------- the fake Daytona client


class FakeExec:
    """What `sandbox.process.exec` returns: exit_code and result (stdout)."""

    def __init__(self, exit_code: int, result: str) -> None:
        self.exit_code = exit_code
        self.result = result


def _opts(flags: list[str]) -> dict[str, str]:
    return {flags[i]: flags[i + 1] for i in range(0, len(flags) - 1, 2)}


class FakeFS:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.files: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def upload_file(self, src: bytes, dst: str, timeout: int = 1800) -> None:
        assert isinstance(src, bytes), "upload_file gets bytes"
        assert dst.startswith("/"), dst
        self.files[dst] = src
        self.uploads.append(dst)

    def download_file(self, path: str) -> bytes:
        self.downloads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class FakeProcess:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox

    def exec(
        self, command: str, cwd: str | None = None, env: dict | None = None, timeout: int | None = None
    ) -> FakeExec:
        box = self.sandbox
        box.commands.append({"command": command, "cwd": cwd, "timeout": timeout})
        if box.exec_fail:
            raise RuntimeError("exec broke")
        if "tar -xzf" in command:
            box.extracted.append(box.fs.files.get(f"{sp.REMOTE_ROOT}/bundle.tgz"))
            return FakeExec(0, "")
        if f"python -m {sp.RUNNER_MODULE}" in command:
            return self._run(command)
        if command.startswith("mkdir -p"):
            return FakeExec(0, "")
        return FakeExec(127, f"unknown command: {command}")

    def _run(self, command: str) -> FakeExec:
        box = self.sandbox
        if box.runner_garbage:
            return FakeExec(1, "Traceback: the runner broke\n")
        match = re.search(rf"python -m {re.escape(sp.RUNNER_MODULE)} (.*) 2> (\S+)$", command)
        assert match, command
        opts = _opts(shlex.split(match.group(1)))
        fs = box.fs
        out = opts["--out"]
        scene = json.loads(fs.files[opts["--scene"]])
        if "--candidate" in opts:
            source = fs.files[opts["--candidate"]].decode("utf-8")
            index = int(opts["--index"])
            json.loads(fs.files[opts["--task"]])
        else:
            skill = json.loads(fs.files[opts["--skill"]])
            source = skill["code"]
            index = 0
            json.loads(opts["--args"])
        trials = int(opts["--trials"])
        if "--skills-dir" in opts:
            assert any(p.startswith(opts["--skills-dir"] + "/") for p in fs.files), "skills were not uploaded"
        passed = "# FAIL" not in source
        mean = 50.0 if "# SLOW" in source else 10.0 + index
        score = {
            "candidate": index,
            "passed": passed,
            "successes": trials if passed else 0,
            "trials": trials,
            "nominal_success": passed,
            "collisions": 0,
            "limit_hits": 0,
            "ik_failures": 0,
            "mean_sim_s": mean,
            "fail_reason": None if passed else "trial 1: spec not satisfied",
            "traj": f"{out}/traj.json",
            "error": None,
        }
        results = [
            {
                "trial": k,
                "success": passed,
                "collisions": 0,
                "limit_hits": 0,
                "ik_failures": 0,
                "sim_s": mean,
                "error": None,
                "first_collision": None,
            }
            for k in range(1, trials + 1)
        ]
        traj = dict(MINIMAL_TRAJ, skill=f"fake-{index}")
        fs.files[f"{out}/traj.json"] = json.dumps(traj).encode("utf-8")
        fs.files[f"{out}/final_scene.json"] = json.dumps(scene).encode("utf-8")
        fs.files[f"{out}/trials.json"] = json.dumps(results).encode("utf-8")
        fs.files[f"{out}/score.json"] = json.dumps(score).encode("utf-8")
        box.runs.append(opts)
        return FakeExec(0, "some earlier stdout line\n" + json.dumps(score) + "\n")


class FakeSandbox:
    def __init__(self, client: FakeDaytona, number: int) -> None:
        self.client = client
        self.id = f"sb-{number}"
        self.fs = FakeFS(self)
        self.process = FakeProcess(self)
        self.commands: list[dict] = []
        self.runs: list[dict] = []
        self.extracted: list[bytes | None] = []
        self.deleted = 0
        self.exec_fail = False
        self.runner_garbage = False
        self.delete_fail = False

    def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        self.deleted += 1
        self.client.deleted.append(self.id)
        if self.delete_fail:
            raise RuntimeError("delete broke")

    def runner_commands(self) -> list[dict]:
        return [c for c in self.commands if f"python -m {sp.RUNNER_MODULE}" in c["command"]]

    def bundle_uploads(self) -> int:
        return self.fs.uploads.count(f"{sp.REMOTE_ROOT}/bundle.tgz")


class FakeSnapshots:
    def __init__(self) -> None:
        self.created: list = []

    def create(self, params, *, on_logs=None, timeout=0):
        self.created.append(params)
        if on_logs is not None:
            on_logs("fake snapshot log")
        return SimpleNamespace(name=params.name)


class FakeDaytona:
    def __init__(self) -> None:
        self.sandboxes: list[FakeSandbox] = []
        self.create_calls: list[tuple] = []
        self.deleted: list[str] = []
        self.fail_creates = 0
        self.snapshot = FakeSnapshots()

    def create(self, params, *, timeout: float = 60, on_snapshot_create_logs=None) -> FakeSandbox:
        self.create_calls.append((params, timeout))
        if self.fail_creates > 0:
            self.fail_creates -= 1
            raise RuntimeError("create broke")
        sandbox = FakeSandbox(self, len(self.sandboxes) + 1)
        self.sandboxes.append(sandbox)
        return sandbox


class FakeLocal(sp.LocalRunner):
    """The local fallback, canned: records the calls and answers at once."""

    def __init__(self) -> None:
        self.score_calls: list[dict] = []
        self.plan_calls: list[dict] = []

    def score(
        self,
        task_dir,
        candidates,
        task_path,
        scene_path,
        skills_dir,
        *,
        trials=5,
        seed=7,
        timeout=60,
        on_score=None,
        cancel=None,
    ):
        pairs = sp._jobs(candidates)
        self.score_calls.append(
            {
                "task_dir": pathlib.Path(task_dir),
                "candidates": pairs,
                "task_path": pathlib.Path(task_path),
                "scene_path": pathlib.Path(scene_path),
                "skills_dir": None if skills_dir is None else pathlib.Path(skills_dir),
                "trials": trials,
                "seed": seed,
                "timeout": timeout,
            }
        )
        out = []
        for index, _path in pairs:
            score = dict(
                sp.runner_error(index, trials, "unused"),
                passed=True,
                successes=trials,
                nominal_success=True,
                fail_reason=None,
                error=None,
                mean_sim_s=20.0,
            )
            scored = sp.scored_dict(
                index, "local", score, [True] * trials, str(pathlib.Path(task_dir) / f"cand_{index}")
            )
            out.append(scored)
            if on_score is not None:
                on_score(scored)
        return out

    def plan(self, skill_path, args, scene_path, out_dir, skills_dir, timeout=30, *, cancel=None):
        self.plan_calls.append(
            {
                "skill_path": pathlib.Path(skill_path),
                "args": dict(args),
                "out_dir": pathlib.Path(out_dir),
                "timeout": timeout,
            }
        )
        score = dict(
            sp.runner_error(0, 1, "unused"),
            passed=True,
            successes=1,
            nominal_success=True,
            fail_reason=None,
            error=None,
            mean_sim_s=9.0,
        )
        return sp.scored_dict(0, "local", score, [True], str(out_dir))


def make_bundle_root(root: pathlib.Path) -> pathlib.Path:
    (root / "arm").mkdir(parents=True)
    (root / "arm" / "__init__.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    (root / "arm" / "__pycache__").mkdir()
    (root / "arm" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00")
    (root / "sim" / "model" / "assets").mkdir(parents=True)
    (root / "sim" / "__init__.py").write_text("", encoding="utf-8")
    (root / "sim" / "model" / "assets" / "mesh.stl").write_bytes(b"solid\n")
    (root / "sim" / "model" / "assets" / "mesh.part").write_bytes(b"partial")
    (root / "sim" / "stale.pyc").write_bytes(b"\x00")
    (root / "schemas").mkdir()
    shutil.copy(ROOT / "schemas" / "score.json", root / "schemas" / "score.json")
    (root / "config").mkdir()
    (root / "config" / "scene.yaml").write_text("a: 1\n", encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def fake(tmp_path, monkeypatch) -> dict:
    """A SandboxPool wired to a fake Daytona client, with atexit/signal stubs and a tiny bundle root."""
    client = FakeDaytona()
    monkeypatch.setattr(sp, "_daytona_factory", lambda: client)
    registered: list = []
    signals: dict[int, object] = {}
    previous_calls: list = []
    stub_signal = SimpleNamespace(
        SIGINT=2,
        SIGTERM=15,
        SIG_DFL=0,
        getsignal=lambda signum: lambda s, f: previous_calls.append((s, f)),
        signal=lambda signum, handler: signals.__setitem__(signum, handler),
        raise_signal=lambda signum: None,
    )
    monkeypatch.setattr(sp, "atexit", SimpleNamespace(register=lambda fn: registered.append(fn)))
    monkeypatch.setattr(sp, "signal", stub_signal)
    monkeypatch.setattr(sp, "_handlers_installed", False)
    monkeypatch.setattr(sp, "_previous_handlers", {})
    monkeypatch.setattr(sp, "_pools", [])
    local = FakeLocal()
    pool = sp.SandboxPool(bundle_root=make_bundle_root(tmp_path / "bundle"), local=local)
    yield {
        "pool": pool,
        "client": client,
        "local": local,
        "registered": registered,
        "signals": signals,
        "previous": previous_calls,
        "root": tmp_path / "bundle",
    }
    pool.shutdown()


def pool_task(tmp_path: pathlib.Path, sources: dict[int, str], skills: int = 2) -> dict:
    task_dir, pairs = make_task(tmp_path / "t-0042", sources)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    for name in ("place_in_tray", "hand_over")[:skills]:
        shutil.copy(STARTER / f"{name}.json", skills_dir / f"{name}.json")
    (skills_dir / "_starter").mkdir(exist_ok=True)  # subdirectories are never uploaded
    return {"task_dir": task_dir, "pairs": pairs, "skills_dir": skills_dir}


def run_flags(sandbox: FakeSandbox, n: int = -1) -> dict[str, str]:
    command = sandbox.runner_commands()[n]["command"]
    match = re.search(rf"python -m {re.escape(sp.RUNNER_MODULE)} (.*) 2> (\S+)$", command)
    assert match, command
    return _opts(shlex.split(match.group(1)))


# ---------------------------------------------------------------- SandboxPool with the fake client


def test_start_creates_seven_and_refuses_an_eighth(fake):
    pool, client = fake["pool"], fake["client"]
    assert pool.start() == 7
    assert len(client.create_calls) == 7 and len(pool.sandboxes) == 7
    for params, timeout in client.create_calls:
        assert type(params).__name__ == "CreateSandboxFromSnapshotParams"
        assert params.snapshot == "arm-sim-v1"
        assert params.auto_stop_interval == 0
        assert params.labels == {"vara": "pool"}
        assert timeout == 120
    assert [slot.role for slot in pool._slots] == ["score"] * 6 + ["plan"]
    with pytest.raises(ValueError):
        pool.start(1, 0)
    assert len(client.create_calls) == 7
    fresh = sp.SandboxPool(bundle_root=fake["root"], local=fake["local"])
    with pytest.raises(ValueError):
        fresh.start(7, 1)
    assert len(client.create_calls) == 7
    assert pool.created == 7
    # every sandbox got the bundle once, extracted in /home/daytona/vara
    for sandbox in client.sandboxes:
        assert sandbox.bundle_uploads() == 1
        extract = [c for c in sandbox.commands if "tar -xzf" in c["command"]]
        assert len(extract) == 1 and extract[0]["cwd"] == sp.REMOTE_ROOT
        assert sandbox.commands[0]["command"] == f"mkdir -p {sp.REMOTE_ROOT}"
    data = client.sandboxes[0].fs.files[f"{sp.REMOTE_ROOT}/bundle.tgz"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == [
        "arm/__init__.py",
        "config/scene.yaml",
        "schemas/score.json",
        "sim/__init__.py",
        "sim/model/assets/mesh.stl",
    ]


def test_bundle_is_deterministic_and_excludes_caches_and_partial_assets(fake):
    root = fake["root"]
    data1, sha1 = sp.build_bundle(root)
    data2, sha2 = sp.build_bundle(root)
    assert data1 == data2 and sha1 == sha2 and len(sha1) == 64
    with gzip.GzipFile(fileobj=io.BytesIO(data1)) as gz:
        raw = gz.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert "arm/__pycache__/x.cpython-312.pyc" not in members
    assert "sim/stale.pyc" not in members and "sim/model/assets/mesh.part" not in members
    assert members["arm/__init__.py"].mtime == 0 and members["arm/__init__.py"].size == 6
    files = sp.bundle_files(ROOT)
    assert pathlib.Path("sim/run_candidate.py") in files and pathlib.Path("schemas/score.json") in files
    assert not any("__pycache__" in f.parts or f.suffix in (".pyc", ".part") for f in files)


def test_bundle_is_uploaded_once_and_again_after_a_file_changes(fake, tmp_path):
    pool, client = fake["pool"], fake["client"]
    pool.start(1, 0)
    sandbox = client.sandboxes[0]
    assert sandbox.bundle_uploads() == 1
    task = pool_task(tmp_path, {1: "import math\n\ndef run(robot, args):\n    return True\n"})
    pool.score(
        task["task_dir"],
        task["pairs"],
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
        task["skills_dir"],
    )
    assert sandbox.bundle_uploads() == 1
    (fake["root"] / "arm" / "__init__.py").write_text("x = 2\ny = 3\n", encoding="utf-8", newline="\n")
    pool.score(
        task["task_dir"],
        task["pairs"],
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
        task["skills_dir"],
    )
    assert sandbox.bundle_uploads() == 2
    assert len([c for c in sandbox.commands if "tar -xzf" in c["command"]]) == 2
    with tarfile.open(fileobj=io.BytesIO(sandbox.extracted[-1]), mode="r:gz") as tar:
        assert tar.extractfile("arm/__init__.py").read() == b"x = 2\ny = 3\n"
    pool.score(
        task["task_dir"],
        task["pairs"],
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
        task["skills_dir"],
    )
    assert sandbox.bundle_uploads() == 2


def test_score_uploads_runs_the_documented_command_parses_and_downloads_only_the_winners(fake, tmp_path):
    pool, client = fake["pool"], fake["client"]
    pool.start(6, 1)
    ok = "import math\n\ndef run(robot, args):\n    return True\n"
    sources = {1: ok, 2: ok + "# FAIL\n", 3: ok, 4: ok + "# SLOW\n"}
    task = pool_task(tmp_path, sources)
    calls: list[dict] = []
    results = pool.score(
        task["task_dir"],
        list(reversed(task["pairs"])),
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
        task["skills_dir"],
        trials=5,
        seed=7,
        timeout=60,
        on_score=calls.append,
    )
    assert [r["candidate"] for r in results] == [1, 2, 3, 4]
    assert sorted(c["candidate"] for c in calls) == [1, 2, 3, 4]
    for scored in results:
        validate_scored(scored, "daytona")
        assert scored["out_dir"] == str(task["task_dir"] / f"cand_{scored['candidate']}")
        assert scored["trial_results"] == [scored["score"]["passed"]] * 5
        assert (task["task_dir"] / f"cand_{scored['candidate']}" / "trials.json").is_file()
        assert (
            scored["score"]["traj"]
            == (task["task_dir"] / f"cand_{scored['candidate']}" / "traj.json").as_posix()
        )
    assert [r["score"]["passed"] for r in results] == [True, False, True, True]
    assert results[1]["score"]["fail_reason"] == "trial 1: spec not satisfied"
    # four sandboxes ran one candidate each: uploads and the documented command
    used = [s for s in client.sandboxes if s.runs]
    assert len(used) == 4 and all(s.id != "sb-7" for s in used)  # never the planning sandbox
    for sandbox in used:
        opts = run_flags(sandbox)
        index = int(opts["--index"])
        remote = f"{sp.REMOTE_ROOT}/tasks/t-0042-1"
        assert opts["--candidate"] == f"{remote}/cand_{index}.py"
        assert opts["--task"] == f"{remote}/task.json" and opts["--scene"] == f"{remote}/scene.json"
        assert opts["--out"] == f"{remote}/cand_{index}" and opts["--skills-dir"] == f"{remote}/skills"
        assert (opts["--trials"], opts["--seed"], opts["--timeout"]) == ("5", "7", "60")
        command = sandbox.runner_commands()[0]
        assert command["cwd"] == sp.REMOTE_ROOT and command["timeout"] == 310
        assert command["command"].startswith(
            f"mkdir -p {remote}/cand_{index} && python -m sim.run_candidate --candidate"
        )
        uploaded = set(sandbox.fs.uploads)
        assert {f"{remote}/cand_{index}.py", f"{remote}/task.json", f"{remote}/scene.json"} <= uploaded
        assert {f"{remote}/skills/place_in_tray.json", f"{remote}/skills/hand_over.json"} <= uploaded
        assert not any("_starter" in path for path in uploaded)
        assert sandbox.fs.files[f"{remote}/cand_{index}.py"] == sources[index].encode("utf-8")
        assert sandbox.fs.files[f"{remote}/task.json"] == (task["task_dir"] / "task.json").read_bytes()
    # downloads: trials.json for everyone, the full set only for the two best passing (1 and 3; 4 is slow)
    by_index = {int(run_flags(s)["--index"]): s for s in used}
    for index, sandbox in by_index.items():
        remote_out = f"{sp.REMOTE_ROOT}/tasks/t-0042-1/cand_{index}"
        downloads = set(sandbox.fs.downloads)
        assert f"{remote_out}/trials.json" in downloads
        full = {f"{remote_out}/{name}" for name in ("traj.json", "final_scene.json", "score.json")}
        if index in (1, 3):
            assert full <= downloads
            assert (task["task_dir"] / f"cand_{index}" / "traj.json").is_file()
            assert (task["task_dir"] / f"cand_{index}" / "final_scene.json").is_file()
            assert (
                json.loads((task["task_dir"] / f"cand_{index}" / "score.json").read_text("utf-8"))[
                    "candidate"
                ]
                == index
            )
        else:
            assert not (full & downloads)
            assert not (task["task_dir"] / f"cand_{index}" / "traj.json").exists()
    assert fake["local"].score_calls == []


def test_more_candidates_than_sandboxes_run_locally(fake, tmp_path):
    pool = fake["pool"]
    pool.start(2, 0)
    ok = "import math\n\ndef run(robot, args):\n    return True\n"
    task = pool_task(tmp_path, {1: ok, 2: ok, 3: ok}, skills=0)
    results = pool.score(
        task["task_dir"], task["pairs"], task["task_dir"] / "task.json", task["task_dir"] / "scene.json", None
    )
    assert [r["runner"] for r in results] == ["daytona", "daytona", "local"]
    assert fake["local"].score_calls[0]["candidates"] == [(3, task["task_dir"] / "cand_3.py")]
    assert fake["local"].score_calls[0]["skills_dir"] is None
    assert "--skills-dir" not in run_flags(fake["client"].sandboxes[0])


def test_fallback_to_local_after_two_failures_of_one_sandbox(fake, tmp_path):
    pool, client, local = fake["pool"], fake["client"], fake["local"]
    pool.start(2, 0)
    broken, healthy = client.sandboxes
    broken.exec_fail = True
    ok = "import math\n\ndef run(robot, args):\n    return True\n"
    task = pool_task(tmp_path, {1: ok, 2: ok}, skills=0)
    results = pool.score(
        task["task_dir"],
        task["pairs"],
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
        None,
        trials=5,
        seed=7,
        timeout=60,
    )
    assert [r["candidate"] for r in results] == [1, 2]
    assert sorted(r["runner"] for r in results) == ["daytona", "local"]
    fallen = next(r for r in results if r["runner"] == "local")
    validate_scored(fallen, "local")
    assert len(local.score_calls) == 1
    call = local.score_calls[0]
    assert call["candidates"] == [(fallen["candidate"], task["task_dir"] / f"cand_{fallen['candidate']}.py")]
    assert (call["task_dir"], call["task_path"], call["scene_path"]) == (
        task["task_dir"],
        task["task_dir"] / "task.json",
        task["task_dir"] / "scene.json",
    )
    assert (call["trials"], call["seed"], call["timeout"]) == (5, 7, 60)
    # two attempts on the broken sandbox, then it is dead and out of the pool
    assert (
        len([c for c in broken.commands if "sim.run_candidate" in c["command"] or "mkdir -p" in c["command"]])
        >= 2
    )
    assert pool.sandboxes == [healthy]
    assert broken.deleted == 0  # deleted at shutdown, not before
    # the next task sees one live sandbox: one candidate there, the other local
    results = pool.score(
        task["task_dir"], task["pairs"], task["task_dir"] / "task.json", task["task_dir"] / "scene.json", None
    )
    assert sorted(r["runner"] for r in results) == ["daytona", "local"]
    assert len(healthy.runs) == 2
    pool.shutdown()
    assert broken.deleted == 1 and healthy.deleted == 1


def test_a_runner_that_prints_no_score_line_counts_as_a_sandbox_failure(fake, tmp_path):
    pool, client = fake["pool"], fake["client"]
    pool.start(1, 0)
    sandbox = client.sandboxes[0]
    sandbox.runner_garbage = True
    ok = "import math\n\ndef run(robot, args):\n    return True\n"
    task = pool_task(tmp_path, {1: ok}, skills=0)
    results = pool.score(
        task["task_dir"], task["pairs"], task["task_dir"] / "task.json", task["task_dir"] / "scene.json", None
    )
    assert results[0]["runner"] == "local"
    assert len(sandbox.runner_commands()) == 2 and pool.sandboxes == []


def test_create_failures_are_retried_once_and_counted_against_the_cap(fake):
    pool, client = fake["pool"], fake["client"]
    client.fail_creates = 1
    assert pool.start(1, 0) == 1
    assert len(client.create_calls) == 2 and len(client.sandboxes) == 1
    client.fail_creates = 2
    assert pool.start(1, 0) == 1  # the second sandbox never came up
    assert len(client.create_calls) == 4 and len(pool.sandboxes) == 1


def test_shutdown_deletes_every_sandbox_even_when_one_delete_raises(fake, tmp_path):
    pool, client = fake["pool"], fake["client"]
    pool.start(3, 1)
    client.sandboxes[1].delete_fail = True
    pool.shutdown()
    assert [s.deleted for s in client.sandboxes] == [1, 1, 1, 1]
    assert sorted(client.deleted) == ["sb-1", "sb-2", "sb-3", "sb-4"]
    assert pool.sandboxes == []
    pool.shutdown()  # idempotent
    assert [s.deleted for s in client.sandboxes] == [1, 1, 1, 1]
    ok = "import math\n\ndef run(robot, args):\n    return True\n"
    task = pool_task(tmp_path, {1: ok}, skills=0)
    results = pool.score(
        task["task_dir"], task["pairs"], task["task_dir"] / "task.json", task["task_dir"] / "scene.json", None
    )
    assert results[0]["runner"] == "local"
    scored = pool.plan(
        STARTER / "place_in_tray.json", {"object": "juice_box"}, SCENE_PATH, tmp_path / "plan", None
    )
    assert scored["runner"] == "local" and fake["local"].plan_calls[0]["args"] == {"object": "juice_box"}


def test_atexit_and_signal_handlers_are_registered_and_shut_the_pool_down(fake):
    pool, client = fake["pool"], fake["client"]
    assert fake["registered"] == [] and fake["signals"] == {}
    pool.start(2, 0)
    assert fake["registered"] == [sp._shutdown_all]
    assert set(fake["signals"]) == {2, 15} and all(h is sp._handle_signal for h in fake["signals"].values())
    pool.start(1, 0)  # once per process
    assert len(fake["registered"]) == 1
    sp._handle_signal(15, None)
    assert [s.deleted for s in client.sandboxes] == [1, 1, 1]
    assert fake["previous"] == [(15, None)]  # the previous handler is chained
    sp._shutdown_all()  # nothing left, nothing raised
    assert [s.deleted for s in client.sandboxes] == [1, 1, 1]


def test_plan_uses_the_planning_sandbox_and_downloads_everything(fake, tmp_path):
    pool, client = fake["pool"], fake["client"]
    pool.start(1, 1)
    out_dir = tmp_path / "t-0007" / "plan"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    shutil.copy(STARTER / "hand_over.json", skills_dir / "hand_over.json")
    scored = pool.plan(
        STARTER / "place_in_tray.json", {"object": "juice_box"}, SCENE_PATH, out_dir, skills_dir, timeout=30
    )
    validate_scored(scored, "daytona")
    assert scored["candidate"] == 0 and scored["score"]["passed"] is True
    assert scored["trial_results"] == [True] and scored["out_dir"] == str(out_dir)
    planner_box = client.sandboxes[1]
    assert planner_box.runs and not client.sandboxes[0].runs
    opts = run_flags(planner_box)
    remote = f"{sp.REMOTE_ROOT}/tasks/t-0007-1"
    assert opts["--skill"] == f"{remote}/skill.json" and json.loads(opts["--args"]) == {"object": "juice_box"}
    assert (opts["--trials"], opts["--seed"], opts["--timeout"]) == ("1", "7", "30")
    assert opts["--out"] == f"{remote}/plan" and opts["--skills-dir"] == f"{remote}/skills"
    assert "--candidate" not in opts and "--index" not in opts
    assert planner_box.runner_commands()[0]["timeout"] == 40
    assert f"{remote}/skills/hand_over.json" in planner_box.fs.uploads
    assert planner_box.fs.files[f"{remote}/skill.json"] == (STARTER / "place_in_tray.json").read_bytes()
    for name in ("traj.json", "final_scene.json", "trials.json", "score.json"):
        assert (out_dir / name).is_file(), name
    TRAJ_VALIDATOR.validate(json.loads((out_dir / "traj.json").read_text("utf-8")))
    assert scored["score"]["traj"] == (out_dir / "traj.json").as_posix()
    assert fake["local"].plan_calls == []
    # without a planning sandbox a scoring one is used; with none, the local runner
    pool.shutdown()
    pool.start(1, 0)
    scored = pool.plan(
        STARTER / "place_in_tray.json", {"object": "juice_box"}, SCENE_PATH, tmp_path / "p2", None
    )
    assert scored["runner"] == "daytona" and client.sandboxes[2].runs
    pool.shutdown()
    scored = pool.plan(
        STARTER / "place_in_tray.json", {"object": "juice_box"}, SCENE_PATH, tmp_path / "p3", None
    )
    assert scored["runner"] == "local" and len(fake["local"].plan_calls) == 1


def test_build_snapshot_uses_the_documented_image(fake):
    client = fake["client"]
    logs: list[str] = []
    result = sp.build_snapshot("arm-sim-test", on_logs=logs.append)
    assert result.name == "arm-sim-test" and logs == ["fake snapshot log"]
    params = client.snapshot.created[0]
    assert type(params).__name__ == "CreateSnapshotParams" and params.name == "arm-sim-test"
    assert params.image is not None


def test_get_pool_is_a_local_runner_under_fake_sandbox_and_without_a_key(monkeypatch):
    monkeypatch.setattr(sp, "_daytona_factory", lambda: (_ for _ in ()).throw(AssertionError("no network")))
    monkeypatch.setattr(sp, "_pool", None)
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.setenv("FAKE_SANDBOX", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    pool = sp.get_pool()
    assert isinstance(pool, sp.LocalRunner) and sp.get_pool() is pool
    sp.reset_pool()
    monkeypatch.delenv("FAKE_SANDBOX", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert isinstance(sp.get_pool(), sp.LocalRunner)
    sp.reset_pool()
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    assert isinstance(sp.get_pool(), sp.LocalRunner)
    sp.reset_pool()
    monkeypatch.delenv("FAKE", raising=False)
    pool = sp.get_pool()
    assert isinstance(pool, sp.SandboxPool) and pool.sandboxes == []  # unstarted: no client, no network
    assert sp.get_pool() is pool
    sp.reset_pool()
    assert sp.get_pool() is not pool
    sp.reset_pool()
