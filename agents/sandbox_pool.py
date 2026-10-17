"""Sandbox pool and local runner for scoring AI-written code (tasks C5 and C6; brief 9.2).

AI-written code (candidates, saved skills, routines) runs only through `python -m sim.run_candidate`, either
inside a Daytona sandbox or in a local subprocess whose environment holds nothing but PATH, PYTHONPATH and the
few variables Windows needs to start Python (`clean_env()`). This module never imports, execs or evals that
code; it starts the runner, reads the LAST non-empty line of its stdout, validates it against
`schemas/score.json` and wraps it in a "scored" dict:

    {"candidate": int, "runner": "local" | "daytona", "score": <exactly schemas/score.json>,
     "trial_results": list[bool] (from out/trials.json, [] when missing), "out_dir": str}

`LocalRunner.score(task_dir, candidates, task_path, scene_path, skills_dir, *, trials=5, seed=7, timeout=60,
on_score=None, cancel=None)` starts every candidate at once (`task_dir/cand_<i>/` holds the runner's outputs
plus `runner.out` and `runner.log`), polls every 0.1 s, kills a process (terminate, then kill, then wait) when
it exceeds `timeout * trials + 10` s or when `cancel` (a `threading.Event`) is set, calls `on_score(scored)`
from the polling thread as each finishes (exceptions in `on_score` are logged, never propagated) and returns
the scored dicts ordered by candidate index. Anything that fails to produce a valid score line becomes a
scored dict with `passed` false and `error` "runner: <reason>". `LocalRunner.plan(skill_path, args,
scene_path, out_dir, skills_dir, timeout=30)` runs the `--skill` form with one trial.

`SandboxPool` has the same two methods plus `start(n_score=6, n_plan=1)` (hard cap `MAX_SANDBOXES` = 7,
refused before anything is created), `shutdown()` (idempotent; deletes every sandbox even when one delete
raises; also run from atexit and from SIGINT/SIGTERM handlers). The code bundle (tar.gz of `arm/`, `sim/`,
`schemas/`, `config/` without `__pycache__`, `*.pyc` and `*.part`; sha256 tracked per sandbox) is uploaded
to `/home/daytona/vara/bundle.tgz` and extracted with `tar -xzf`; it is re-uploaded when its hash changes.
Per task the candidate files, `task.json`, `scene.json` and `skills/*.json` go to
`/home/daytona/vara/tasks/<task>/`, the documented runner command runs through `sandbox.process.exec`, and
`traj.json`, `final_scene.json`, `trials.json` and `score.json` are downloaded into `task_dir/cand_<i>/`
for the two best passing candidates (`trials.json` for every candidate, it is tiny). A sandbox that fails
twice is dead and its work falls back to `LocalRunner` (`runner: "local"`), as does every candidate for
which no live sandbox is free. Daytona 0.225 names, verified by introspection: `Daytona()` (reads
DAYTONA_API_KEY, DAYTONA_API_URL, DAYTONA_TARGET), `daytona.create(CreateSandboxFromSnapshotParams(...),
timeout=)`,
`sandbox.fs.upload_file(bytes, path)`, `sandbox.fs.download_file(path) -> bytes`,
`sandbox.process.exec(cmd, cwd=, timeout=) -> ExecuteResponse(exit_code, result)`, `sandbox.delete()`,
`daytona.snapshot.create(CreateSnapshotParams(name=, image=), on_logs=)`. The client comes from the
module-level `_daytona_factory()` so tests inject a fake; nothing here touches the network in tests.

`get_pool()` is the singleton the orchestrator uses: a `LocalRunner` when `FAKE_SANDBOX` (or `FAKE`) is "1"
or `DAYTONA_API_KEY` is unset, else an unstarted `SandboxPool`. `reset_pool()` is for tests.
`build_snapshot(name="arm-sim-v1")` builds the sandbox image. CLI: `python -m agents.sandbox_pool --bench`
scores three fixture tasks (6 candidates x 5 trials) on the current pool and prints the wall clock per task;
`--build-snapshot [NAME]` needs a key.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from sim import scoring

__all__ = [
    "BUNDLE_DIRS",
    "CLEAN_ENV_KEYS",
    "MAX_SANDBOXES",
    "REMOTE_ROOT",
    "SNAPSHOT",
    "LocalRunner",
    "SandboxPool",
    "build_bundle",
    "build_snapshot",
    "bundle_files",
    "clean_env",
    "get_pool",
    "main",
    "parse_score_line",
    "read_trial_results",
    "reset_pool",
    "runner_args",
    "runner_error",
    "scored_dict",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SCORE_VALIDATOR = Draft202012Validator(json.loads((SCHEMAS / "score.json").read_text(encoding="utf-8")))

RUNNER_MODULE = "sim.run_candidate"
SNAPSHOT = "arm-sim-v1"
REMOTE_ROOT = "/home/daytona/vara"
BUNDLE_DIRS = ("arm", "sim", "schemas", "config")
BUNDLE_NAME = "bundle.tgz"
MAX_SANDBOXES = 7
DEFAULT_TRIALS = 5
DEFAULT_SEED = 7
DEFAULT_TIMEOUT = 60.0
PLAN_TIMEOUT = 30.0
GRACE_S = 10.0  # added to timeout * trials before a runner is killed
POLL_S = 0.1
KILL_WAIT_S = 1.0  # after terminate() and again after kill()
CLEAN_ENV_KEYS = ("PATH", "SYSTEMROOT", "TEMP", "TMP")  # PYTHONPATH is set to ROOT
FAILURES_BEFORE_DEAD = 2
WINNERS = 2
OUTPUT_FILES = ("traj.json", "final_scene.json", "trials.json", "score.json")
STDOUT_NAME = "runner.out"
STDERR_NAME = "runner.log"
RUNNER_LABEL_LOCAL = "local"
RUNNER_LABEL_DAYTONA = "daytona"

log = logging.getLogger(__name__)


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


# ---------------------------------------------------------------------------- shared helpers


def clean_env() -> dict[str, str]:
    """The environment of a local runner: PATH, PYTHONPATH=ROOT, and SYSTEMROOT/TEMP/TMP when present."""
    env = {key: os.environ[key] for key in CLEAN_ENV_KEYS if key in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    return env


def runner_args(
    *,
    scene: str,
    out: str,
    trials: int,
    seed: int | str,
    timeout: float,
    candidate: str | None = None,
    task: str | None = None,
    skill: str | None = None,
    args: dict | None = None,
    index: int | None = None,
    skills_dir: str | None = None,
) -> list[str]:
    """The `sim.run_candidate` flags for one run, in the documented order (paths as given)."""
    if (candidate is None) == (skill is None):
        raise ValueError("give exactly one of candidate or skill")
    out_args: list[str] = []
    if candidate is not None:
        if task is None:
            raise ValueError("a candidate needs a task spec")
        out_args += ["--candidate", candidate, "--task", task]
    else:
        out_args += ["--skill", str(skill), "--args", json.dumps(args or {})]
    out_args += ["--scene", scene, "--trials", str(int(trials)), "--seed", str(seed), "--out", out]
    if index is not None:
        out_args += ["--index", str(int(index))]
    if skills_dir is not None:
        out_args += ["--skills-dir", skills_dir]
    out_args += ["--timeout", f"{float(timeout):g}"]
    return out_args


def parse_score_line(line: str) -> dict:
    """The score dict of one stdout line, validated against schemas/score.json. Raises ValueError."""
    try:
        score = json.loads(line)
    except ValueError as exc:
        raise ValueError(f"last line is not JSON: {line[:120]!r}") from exc
    if not isinstance(score, dict):
        raise ValueError(f"last line is not a JSON object: {line[:120]!r}")
    errors = sorted(SCORE_VALIDATOR.iter_errors(score), key=lambda e: list(e.absolute_path))
    if errors:
        where = "/".join(str(p) for p in errors[0].absolute_path) or "(root)"
        raise ValueError(f"score line invalid at {where}: {errors[0].message}")
    return score


def runner_error(index: int, trials: int, reason: str) -> dict:
    """The score of a candidate whose runner produced no valid score line (nothing is known about it)."""
    message = f"runner: {reason}"
    return {
        "candidate": int(index),
        "passed": False,
        "successes": 0,
        "trials": max(1, int(trials)),
        "nominal_success": False,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "mean_sim_s": None,
        "fail_reason": message,
        "traj": None,
        "error": message,
    }


def read_trial_results(out_dir: str | os.PathLike[str]) -> list[bool]:
    """The success flags of `out_dir/trials.json`, in trial order; [] when missing or unreadable."""
    path = pathlib.Path(out_dir) / "trials.json"
    try:
        trials = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(trials, list):
        return []
    try:
        return [bool(t["success"]) for t in trials]
    except (KeyError, TypeError):
        return []


def scored_dict(index: int, runner: str, score: dict, trial_results: list[bool], out_dir: str) -> dict:
    """The common result shape (brief 9.2). `score` is kept exactly as validated."""
    return {
        "candidate": int(index),
        "runner": runner,
        "score": score,
        "trial_results": list(trial_results),
        "out_dir": str(out_dir),
    }


def _last_line(text: str) -> str | None:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else None


def _call_on_score(on_score: Callable[[dict], Any] | None, scored: dict) -> None:
    if on_score is None:
        return
    try:
        on_score(scored)
    except Exception:  # noqa: BLE001 - a listener must never break scoring
        log.exception("on_score failed for candidate %s", scored.get("candidate"))


def _jobs(candidates: Iterable[Any]) -> list[tuple[int, pathlib.Path]]:
    """`(index, path)` pairs from pairs or objects with `.index` and `.path`."""
    out: list[tuple[int, pathlib.Path]] = []
    for item in candidates:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            index, path = item
        else:
            index, path = getattr(item, "index"), getattr(item, "path")  # noqa: B009 - clear error below
        if path is None:
            raise ValueError(f"candidate {index!r} has no source path")
        out.append((int(index), pathlib.Path(path)))
    return out


# ---------------------------------------------------------------------------- local runner


class _LocalJob:
    """One runner subprocess: its argv, output files, process and outcome."""

    def __init__(self, index: int, argv: list[str], out_dir: pathlib.Path, trials: int) -> None:
        self.index = index
        self.argv = argv
        self.out_dir = out_dir
        self.trials = trials
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.reason: str | None = None  # why it was killed, or why it could not start
        self.returncode: int | None = None

    @property
    def stdout_path(self) -> pathlib.Path:
        return self.out_dir / STDOUT_NAME

    @property
    def stderr_path(self) -> pathlib.Path:
        return self.out_dir / STDERR_NAME

    def start(self, spawn: Callable[..., subprocess.Popen]) -> None:
        self.started = time.monotonic()
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.stdout_path, "wb") as out, open(self.stderr_path, "wb") as err:
                self.proc = spawn(self.argv, cwd=str(ROOT), env=clean_env(), stdout=out, stderr=err)
        except Exception as exc:  # noqa: BLE001 - a start failure is this candidate's error, not ours
            self.reason = f"could not start: {type(exc).__name__}: {exc}"
            log.error("candidate %d: %s", self.index, self.reason)
            return
        log.info("candidate %d: runner pid %s started", self.index, self.proc.pid)

    def poll(self) -> bool:
        """True when the process has ended (or never started)."""
        if self.proc is None:
            return True
        code = self.proc.poll()
        if code is None:
            return False
        self.returncode = code
        return True

    def kill(self, reason: str) -> None:
        """Terminate, then kill, then wait; the runner's own child dies with it (dead-man's switch)."""
        self.reason = reason
        proc = self.proc
        if proc is None:
            return
        for action in (proc.terminate, proc.kill):
            if proc.poll() is not None:
                break
            try:
                action()
            except OSError:
                pass
            try:
                proc.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                continue
        self.returncode = proc.poll()
        if self.returncode is None:
            log.error(
                "candidate %d: runner pid %s survived terminate and kill (%s)", self.index, proc.pid, reason
            )
        else:
            log.warning("candidate %d: runner pid %s killed (%s)", self.index, proc.pid, reason)

    def collect(self, runner: str = RUNNER_LABEL_LOCAL) -> dict:
        """The scored dict once the process is gone."""
        if self.reason is not None:
            return scored_dict(
                self.index, runner, runner_error(self.index, self.trials, self.reason), [], str(self.out_dir)
            )
        try:
            text = self.stdout_path.read_bytes().decode("utf-8", "replace")
        except OSError:
            text = ""
        line = _last_line(text)
        code = self.returncode
        if line is None:
            reason = f"exit code {code}, no score line"
            log.error("candidate %d: %s (see %s)", self.index, reason, self.stderr_path)
            return scored_dict(
                self.index, runner, runner_error(self.index, self.trials, reason), [], str(self.out_dir)
            )
        try:
            score = parse_score_line(line)
        except ValueError as exc:
            reason = f"exit code {code}, {exc}"
            log.error("candidate %d: %s", self.index, reason)
            return scored_dict(
                self.index, runner, runner_error(self.index, self.trials, reason), [], str(self.out_dir)
            )
        if code not in (0, None):
            log.warning("candidate %d: runner exit code %s but a valid score line", self.index, code)
        if score["candidate"] != self.index:
            log.warning("candidate %d: the score line says candidate %s", self.index, score["candidate"])
        return scored_dict(self.index, runner, score, read_trial_results(self.out_dir), str(self.out_dir))


class LocalRunner:
    """Runs `sim.run_candidate` as local subprocesses with the clean environment (the FAKE_SANDBOX runner
    and the live fallback). See the module docstring for `score` and `plan`."""

    name = RUNNER_LABEL_LOCAL

    def spawn(self, argv: list[str], **kwargs: Any) -> subprocess.Popen:
        """Start one runner process (tests override this to record or replace the process)."""
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, **kwargs)

    def score(
        self,
        task_dir: str | os.PathLike[str],
        candidates: Iterable[Any],
        task_path: str | os.PathLike[str],
        scene_path: str | os.PathLike[str],
        skills_dir: str | os.PathLike[str] | None,
        *,
        trials: int = DEFAULT_TRIALS,
        seed: int | str = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT,
        on_score: Callable[[dict], Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> list[dict]:
        task_dir = pathlib.Path(task_dir).resolve()
        jobs: list[_LocalJob] = []
        for index, path in _jobs(candidates):
            out_dir = task_dir / f"cand_{index}"
            argv = [sys.executable, "-m", RUNNER_MODULE] + runner_args(
                candidate=str(pathlib.Path(path).resolve()),
                task=str(pathlib.Path(task_path).resolve()),
                scene=str(pathlib.Path(scene_path).resolve()),
                out=str(out_dir),
                trials=trials,
                seed=seed,
                timeout=timeout,
                index=index,
                skills_dir=None if skills_dir is None else str(pathlib.Path(skills_dir).resolve()),
            )
            jobs.append(_LocalJob(index, argv, out_dir, int(trials)))
        results = self._run(jobs, float(timeout) * int(trials) + GRACE_S, on_score, cancel)
        return sorted(results, key=lambda r: r["candidate"])

    def plan(
        self,
        skill_path: str | os.PathLike[str],
        args: dict,
        scene_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        skills_dir: str | os.PathLike[str] | None,
        timeout: float = PLAN_TIMEOUT,
        *,
        cancel: threading.Event | None = None,
    ) -> dict:
        out = pathlib.Path(out_dir).resolve()
        argv = [sys.executable, "-m", RUNNER_MODULE] + runner_args(
            skill=str(pathlib.Path(skill_path).resolve()),
            args=dict(args or {}),
            scene=str(pathlib.Path(scene_path).resolve()),
            out=str(out),
            trials=1,
            seed=DEFAULT_SEED,
            timeout=timeout,
            skills_dir=None if skills_dir is None else str(pathlib.Path(skills_dir).resolve()),
        )
        return self._run([_LocalJob(0, argv, out, 1)], float(timeout) + GRACE_S, None, cancel)[0]

    def _run(
        self,
        jobs: list[_LocalJob],
        deadline_s: float,
        on_score: Callable[[dict], Any] | None,
        cancel: threading.Event | None,
    ) -> list[dict]:
        """Start every job at once, poll every 0.1 s, kill on the deadline or cancel, collect as each ends."""
        results: list[dict] = []
        for job in jobs:
            if cancel is not None and cancel.is_set():
                job.reason = "cancelled"
                continue
            job.start(self.spawn)
        pending = [job for job in jobs if job.proc is not None]
        for job in jobs:
            if job.proc is None:
                scored = job.collect()
                results.append(scored)
                _call_on_score(on_score, scored)
        while pending:
            now = time.monotonic()
            for job in list(pending):
                if job.poll():
                    pass
                elif cancel is not None and cancel.is_set():
                    job.kill("cancelled")
                elif now - job.started > deadline_s:
                    job.kill(f"timeout after {deadline_s:.0f} s")
                else:
                    continue
                pending.remove(job)
                scored = job.collect()
                results.append(scored)
                _call_on_score(on_score, scored)
            if pending:
                time.sleep(POLL_S)
        return results


# ---------------------------------------------------------------------------- bundle


def bundle_files(root: pathlib.Path = ROOT, dirs: Sequence[str] = BUNDLE_DIRS) -> list[pathlib.Path]:
    """The files of the code bundle, relative to `root`, sorted; no `__pycache__`, `*.pyc` or `*.part`."""
    files: list[pathlib.Path] = []
    for name in dirs:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or path.suffix in (".pyc", ".part"):
                continue
            files.append(rel)
    return sorted(files, key=lambda p: p.as_posix())


def bundle_fingerprint(root: pathlib.Path = ROOT, dirs: Sequence[str] = BUNDLE_DIRS) -> tuple:
    """A cheap stamp (path, size, mtime) of the bundle files; the tar is rebuilt only when it changes."""
    stamp = []
    for rel in bundle_files(root, dirs):
        try:
            st = (root / rel).stat()
        except OSError:
            continue
        stamp.append((rel.as_posix(), st.st_size, st.st_mtime_ns))
    return tuple(stamp)


def build_bundle(root: pathlib.Path = ROOT, dirs: Sequence[str] = BUNDLE_DIRS) -> tuple[bytes, str]:
    """The deterministic tar.gz of the bundle dirs and its sha256 (same files -> same bytes)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for rel in bundle_files(root, dirs):
            path = root / rel
            info = tar.gettarinfo(str(path), arcname=rel.as_posix())
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------- Daytona pool


def _daytona_factory() -> Any:
    """The Daytona client (reads DAYTONA_API_KEY, DAYTONA_API_URL, DAYTONA_TARGET); tests replace this."""
    from daytona import Daytona

    return Daytona()


def _sandbox_params(snapshot: str) -> Any:
    from daytona import CreateSandboxFromSnapshotParams

    return CreateSandboxFromSnapshotParams(snapshot=snapshot, auto_stop_interval=0, labels={"vara": "pool"})


def build_snapshot(name: str = SNAPSHOT, *, on_logs: Callable[[str], Any] | None = print) -> Any:
    """Build the `arm-sim-v1` image: debian slim, Python 3.12, mujoco numpy pyyaml jsonschema (key needed)."""
    from daytona import CreateSnapshotParams, Image

    image = Image.debian_slim("3.12").pip_install(["mujoco", "numpy", "pyyaml", "jsonschema"])
    daytona = _daytona_factory()
    return daytona.snapshot.create(CreateSnapshotParams(name=name, image=image), on_logs=on_logs)


class _SandboxFailure(Exception):
    """The sandbox (not the candidate) failed: transport, exec, no score line, missing download."""


@dataclass
class _Slot:
    sandbox: Any
    role: str  # "score" | "plan"
    number: int
    failures: int = 0
    bundle_sha: str | None = None
    dead: bool = False
    busy: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return str(getattr(self.sandbox, "id", f"sandbox-{self.number}"))


@dataclass
class _Job:
    """One run inside a sandbox: the local inputs, where they go remotely, and the outputs to download."""

    index: int
    task_name: str
    out_name: str  # remote and local out dir name under the task dir ("cand_3" or "plan")
    local_out: pathlib.Path
    uploads: list[tuple[pathlib.Path, str]]  # (local file, remote name under the task dir)
    flags: list[str]  # runner flags with remote paths
    trials: int
    timeout: float
    download_all: bool  # the planner needs every output; score() downloads only the winners
    local: Callable[[], dict]  # the local fallback for this job

    @property
    def remote_task(self) -> str:
        return f"{REMOTE_ROOT}/tasks/{self.task_name}"

    @property
    def remote_out(self) -> str:
        return f"{self.remote_task}/{self.out_name}"


_pools_lock = threading.Lock()
_pools: list[SandboxPool] = []
_handlers_installed = False
_previous_handlers: dict[int, Any] = {}


def _shutdown_all() -> None:
    with _pools_lock:
        pools = list(_pools)
    for pool in pools:
        try:
            pool.shutdown()
        except Exception:  # noqa: BLE001 - keep going at exit
            log.exception("pool shutdown failed")


def _handle_signal(signum: int, frame: Any) -> None:
    log.warning("signal %s: deleting the sandboxes", signum)
    _shutdown_all()
    previous = _previous_handlers.get(signum)
    if callable(previous):
        previous(signum, frame)
    elif signum == getattr(signal, "SIGINT", None):
        raise KeyboardInterrupt
    else:
        try:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        except (ValueError, OSError, AttributeError):
            sys.exit(128 + int(signum))


def _install_exit_handlers() -> None:
    """atexit plus SIGINT/SIGTERM handlers that delete every pool's sandboxes (once per process)."""
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True
    atexit.register(_shutdown_all)
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            _previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError, TypeError):  # not the main thread: atexit still covers us
            log.info("could not install a %s handler (not the main thread)", name)


class SandboxPool:
    """Six scoring sandboxes and one planning sandbox on Daytona, with local fallback (brief 9.2)."""

    name = RUNNER_LABEL_DAYTONA

    def __init__(
        self,
        *,
        snapshot: str = SNAPSHOT,
        bundle_root: pathlib.Path = ROOT,
        local: LocalRunner | None = None,
        create_timeout: float = 120.0,
    ) -> None:
        self.snapshot = snapshot
        self.bundle_root = pathlib.Path(bundle_root)
        self.local = local if local is not None else LocalRunner()
        self.create_timeout = create_timeout
        self._client: Any = None
        self._slots: list[_Slot] = []
        self._lock = threading.RLock()
        self._bundle_lock = threading.Lock()
        self._bundle_cache: tuple[tuple, bytes, str] | None = None
        self._task_counter = 0
        self._created = 0

    # ----------------------------------------------------------------- lifecycle

    @property
    def sandboxes(self) -> list[Any]:
        """The live sandboxes (for the dashboard and tests)."""
        with self._lock:
            return [slot.sandbox for slot in self._slots if not slot.dead]

    @property
    def created(self) -> int:
        """How many sandboxes this pool has created in its lifetime."""
        return self._created

    def _client_or_create(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = _daytona_factory()
            return self._client

    def start(self, n_score: int = 6, n_plan: int = 1) -> int:
        """Create the sandboxes in parallel threads and install the bundle; returns how many are live.
        Refuses (ValueError, nothing created) when the pool would exceed MAX_SANDBOXES."""
        n_score, n_plan = int(n_score), int(n_plan)
        if n_score < 0 or n_plan < 0:
            raise ValueError("sandbox counts must be 0 or more")
        with self._lock:
            existing = len(self._slots)
            if existing + n_score + n_plan > MAX_SANDBOXES:
                raise ValueError(
                    f"refusing to create {n_score + n_plan} sandboxes on top of {existing}: "
                    f"the cap is {MAX_SANDBOXES}"
                )
            roles = ["score"] * n_score + ["plan"] * n_plan
            numbers = list(range(existing + 1, existing + len(roles) + 1))
        with _pools_lock:
            if self not in _pools:
                _pools.append(self)
        _install_exit_handlers()
        if not roles:
            return len(self.sandboxes)
        client = self._client_or_create()
        threads = [
            threading.Thread(target=self._create_slot, args=(client, role, number), daemon=True)
            for role, number in zip(roles, numbers, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        live = len(self.sandboxes)
        log.info("sandbox pool: %d of %d sandboxes live", live, len(roles))
        return live

    def _create_slot(self, client: Any, role: str, number: int) -> None:
        sandbox = None
        for attempt in (1, 2):
            try:
                with self._lock:
                    if self._created >= MAX_SANDBOXES:
                        log.error("sandbox %d (%s): cap reached, not created", number, role)
                        return
                    self._created += 1
                sandbox = client.create(_sandbox_params(self.snapshot), timeout=self.create_timeout)
                break
            except Exception as exc:  # noqa: BLE001 - the SDK raises many types
                log.error("sandbox %d (%s): create attempt %d failed: %s", number, role, attempt, exc)
        if sandbox is None:
            return
        slot = _Slot(sandbox=sandbox, role=role, number=number)
        with self._lock:
            self._slots.append(slot)
            # creates finish in any order; keep sandbox 1..N order for status, logs and the dashboard
            self._slots.sort(key=lambda s: s.number)
        for attempt in (1, 2):
            try:
                self._ensure_bundle(slot)
                log.info("sandbox %d (%s) ready: %s", number, role, slot.id)
                return
            except _SandboxFailure as exc:
                self._record_failure(slot, f"bundle install attempt {attempt}: {exc}")

    def shutdown(self) -> None:
        """Delete every sandbox, even when one delete raises; safe to call more than once."""
        with self._lock:
            slots = list(self._slots)
            self._slots = []
        if not slots:
            return

        def delete(slot: _Slot) -> None:
            try:
                slot.sandbox.delete()
                log.info("sandbox %s deleted", slot.id)
            except Exception as exc:  # noqa: BLE001 - keep deleting the others
                log.error("sandbox %s: delete failed: %s", slot.id, exc)

        threads = [threading.Thread(target=delete, args=(slot,), daemon=True) for slot in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # ----------------------------------------------------------------- slots

    def _acquire(self, role: str, n: int) -> list[_Slot]:
        """Up to `n` live idle slots of `role` (busy until `_release`)."""
        taken: list[_Slot] = []
        with self._lock:
            for slot in self._slots:
                if len(taken) >= n:
                    break
                if slot.dead or slot.role != role:
                    continue
                if slot.busy.acquire(blocking=False):
                    taken.append(slot)
        return taken

    def _release(self, slots: Iterable[_Slot]) -> None:
        for slot in slots:
            try:
                slot.busy.release()
            except RuntimeError:
                pass

    def _record_failure(self, slot: _Slot, reason: str) -> None:
        with self._lock:
            slot.failures += 1
            if slot.failures >= FAILURES_BEFORE_DEAD and not slot.dead:
                slot.dead = True
                log.error(
                    "sandbox %s is dead after %d failures (%s); its work runs locally",
                    slot.id,
                    slot.failures,
                    reason,
                )
                return
        log.warning("sandbox %s: failure %d: %s", slot.id, slot.failures, reason)

    # ----------------------------------------------------------------- bundle and files

    def _bundle(self) -> tuple[bytes, str]:
        fingerprint = bundle_fingerprint(self.bundle_root)
        with self._bundle_lock:
            if self._bundle_cache is None or self._bundle_cache[0] != fingerprint:
                data, sha = build_bundle(self.bundle_root)
                self._bundle_cache = (fingerprint, data, sha)
                log.info("code bundle: %d bytes, sha256 %s", len(data), sha[:12])
            return self._bundle_cache[1], self._bundle_cache[2]

    def _exec(self, slot: _Slot, command: str, *, cwd: str | None, timeout: float) -> Any:
        try:
            return slot.sandbox.process.exec(command, cwd=cwd, timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise _SandboxFailure(f"exec failed: {type(exc).__name__}: {exc}") from exc

    def _ensure_bundle(self, slot: _Slot) -> None:
        data, sha = self._bundle()
        if slot.bundle_sha == sha:
            return
        response = self._exec(slot, f"mkdir -p {REMOTE_ROOT}", cwd=None, timeout=30)
        if response.exit_code != 0:
            raise _SandboxFailure(f"mkdir failed: {getattr(response, 'result', '')!s:.200}")
        try:
            slot.sandbox.fs.upload_file(data, f"{REMOTE_ROOT}/{BUNDLE_NAME}")
        except Exception as exc:  # noqa: BLE001
            raise _SandboxFailure(f"bundle upload failed: {type(exc).__name__}: {exc}") from exc
        command = f"rm -rf {' '.join(BUNDLE_DIRS)} && tar -xzf {BUNDLE_NAME}"
        response = self._exec(slot, command, cwd=REMOTE_ROOT, timeout=120)
        if response.exit_code != 0:
            raise _SandboxFailure(f"bundle extract failed: {getattr(response, 'result', '')!s:.200}")
        slot.bundle_sha = sha
        log.info("sandbox %s: bundle %s installed", slot.id, sha[:12])

    def _upload(self, slot: _Slot, local: pathlib.Path, remote: str) -> None:
        try:
            slot.sandbox.fs.upload_file(local.read_bytes(), remote)
        except Exception as exc:  # noqa: BLE001
            raise _SandboxFailure(f"upload of {local.name} failed: {type(exc).__name__}: {exc}") from exc

    def _download(self, slot: _Slot, remote: str, local: pathlib.Path) -> None:
        try:
            data = slot.sandbox.fs.download_file(remote)
        except Exception as exc:  # noqa: BLE001
            raise _SandboxFailure(f"download of {remote} failed: {type(exc).__name__}: {exc}") from exc
        if data is None:
            raise _SandboxFailure(f"download of {remote} returned nothing")
        if isinstance(data, str):
            data = data.encode("utf-8")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)

    def _download_outputs(self, slot: _Slot, job: _Job, names: Sequence[str] = OUTPUT_FILES) -> None:
        for name in names:
            self._download(slot, f"{job.remote_out}/{name}", job.local_out / name)

    def _task_name(self, task_dir: pathlib.Path) -> str:
        with self._lock:
            self._task_counter += 1
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_dir.name) or "task"
            return f"{stem}-{self._task_counter}"

    @staticmethod
    def _skill_uploads(skills_dir: pathlib.Path | None) -> list[tuple[pathlib.Path, str]]:
        if skills_dir is None or not skills_dir.is_dir():
            return []
        return [(path, f"skills/{path.name}") for path in sorted(skills_dir.glob("*.json"))]

    # ----------------------------------------------------------------- running jobs

    def _run_attempt(self, slot: _Slot, job: _Job) -> dict:
        """One try on one sandbox; raises _SandboxFailure when the sandbox (not the candidate) fails."""
        self._ensure_bundle(slot)
        for local, name in job.uploads:
            self._upload(slot, local, f"{job.remote_task}/{name}")
        command = (
            f"mkdir -p {job.remote_out} && python -m {RUNNER_MODULE} {shlex.join(job.flags)} "
            f"2> {job.remote_out}/{STDERR_NAME}"
        )
        deadline = job.timeout * job.trials + GRACE_S
        log.info("sandbox %s: candidate %d: %s", slot.id, job.index, command)
        response = self._exec(slot, command, cwd=REMOTE_ROOT, timeout=deadline)
        line = _last_line(str(getattr(response, "result", "") or ""))
        if line is None:
            raise _SandboxFailure(f"exit code {response.exit_code}, no score line")
        try:
            score = parse_score_line(line)
        except ValueError as exc:
            raise _SandboxFailure(f"exit code {response.exit_code}, {exc}") from exc
        job.local_out.mkdir(parents=True, exist_ok=True)
        if job.download_all:
            self._download_outputs(slot, job)
        else:
            try:
                self._download(slot, f"{job.remote_out}/trials.json", job.local_out / "trials.json")
            except _SandboxFailure as exc:
                log.warning("sandbox %s: candidate %d: %s", slot.id, job.index, exc)
        if score.get("traj") is not None:
            score["traj"] = (job.local_out / "traj.json").as_posix()
        return scored_dict(
            job.index, RUNNER_LABEL_DAYTONA, score, read_trial_results(job.local_out), str(job.local_out)
        )

    def _run_job(self, slot: _Slot, job: _Job, cancel: threading.Event | None) -> dict:
        """Up to two attempts on the sandbox (dead after its second failure), then the local fallback."""
        for _attempt in (1, 2):
            if cancel is not None and cancel.is_set():
                return scored_dict(
                    job.index,
                    RUNNER_LABEL_DAYTONA,
                    runner_error(job.index, job.trials, "cancelled"),
                    [],
                    str(job.local_out),
                )
            if slot.dead:
                break
            try:
                return self._run_attempt(slot, job)
            except _SandboxFailure as exc:
                self._record_failure(slot, f"candidate {job.index}: {exc}")
        if cancel is not None and cancel.is_set():
            return scored_dict(
                job.index,
                RUNNER_LABEL_LOCAL,
                runner_error(job.index, job.trials, "cancelled"),
                [],
                str(job.local_out),
            )
        log.warning("candidate %d runs locally after sandbox failures", job.index)
        return job.local()

    def _candidate_job(
        self,
        task_name: str,
        task_dir: pathlib.Path,
        index: int,
        path: pathlib.Path,
        task_path: pathlib.Path,
        scene_path: pathlib.Path,
        skills_dir: pathlib.Path | None,
        trials: int,
        seed: int | str,
        timeout: float,
        cancel: threading.Event | None,
    ) -> _Job:
        remote_task = f"{REMOTE_ROOT}/tasks/{task_name}"
        out_name = f"cand_{index}"
        uploads = [(path, f"cand_{index}.py"), (task_path, "task.json"), (scene_path, "scene.json")]
        uploads += self._skill_uploads(skills_dir)
        flags = runner_args(
            candidate=f"{remote_task}/cand_{index}.py",
            task=f"{remote_task}/task.json",
            scene=f"{remote_task}/scene.json",
            out=f"{remote_task}/{out_name}",
            trials=trials,
            seed=seed,
            timeout=timeout,
            index=index,
            skills_dir=f"{remote_task}/skills" if skills_dir is not None else None,
        )

        def local() -> dict:
            return self.local.score(
                task_dir,
                [(index, path)],
                task_path,
                scene_path,
                skills_dir,
                trials=trials,
                seed=seed,
                timeout=timeout,
                cancel=cancel,
            )[0]

        return _Job(
            index,
            task_name,
            out_name,
            task_dir / out_name,
            uploads,
            flags,
            int(trials),
            float(timeout),
            False,
            local,
        )

    def score(
        self,
        task_dir: str | os.PathLike[str],
        candidates: Iterable[Any],
        task_path: str | os.PathLike[str],
        scene_path: str | os.PathLike[str],
        skills_dir: str | os.PathLike[str] | None,
        *,
        trials: int = DEFAULT_TRIALS,
        seed: int | str = DEFAULT_SEED,
        timeout: float = DEFAULT_TIMEOUT,
        on_score: Callable[[dict], Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> list[dict]:
        task_dir = pathlib.Path(task_dir).resolve()
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = pathlib.Path(task_path).resolve()
        scene_path = pathlib.Path(scene_path).resolve()
        skills = None if skills_dir is None else pathlib.Path(skills_dir).resolve()
        pairs = _jobs(candidates)
        if not pairs:
            return []
        task_name = self._task_name(task_dir)
        slots = self._acquire("score", len(pairs))
        results: dict[int, dict] = {}
        results_lock = threading.Lock()
        abandoned = threading.Event()

        def emit(scored: dict) -> None:
            if abandoned.is_set():
                return
            with results_lock:
                results[scored["candidate"]] = scored
            _call_on_score(on_score, scored)

        jobs = [
            self._candidate_job(
                task_name, task_dir, index, path, task_path, scene_path, skills, trials, seed, timeout, cancel
            )
            for index, path in pairs
        ]
        used: dict[int, _Slot] = {}
        threads: list[threading.Thread] = []
        try:
            for slot, job in zip(slots, jobs, strict=False):
                used[job.index] = slot
                threads.append(
                    threading.Thread(target=self._job_thread, args=(slot, job, emit, cancel), daemon=True)
                )
            local_pairs = [(job.index, pathlib.Path(dict(pairs)[job.index])) for job in jobs[len(slots) :]]
            if local_pairs:
                log.info("%d candidate(s) run locally: no free sandbox", len(local_pairs))

                def run_local() -> None:
                    try:
                        self.local.score(
                            task_dir,
                            local_pairs,
                            task_path,
                            scene_path,
                            skills,
                            trials=trials,
                            seed=seed,
                            timeout=timeout,
                            on_score=emit,
                            cancel=cancel,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.exception("local runner failed")
                        for index, _path in local_pairs:
                            emit(
                                scored_dict(
                                    index,
                                    RUNNER_LABEL_LOCAL,
                                    runner_error(index, trials, f"local runner: {exc}"),
                                    [],
                                    str(task_dir / f"cand_{index}"),
                                )
                            )

                threads.append(threading.Thread(target=run_local, daemon=True))
            for thread in threads:
                thread.start()
            self._wait(threads, cancel)
            if cancel is not None and cancel.is_set() and any(t.is_alive() for t in threads):
                abandoned.set()
                for job in jobs:
                    with results_lock:
                        if job.index not in results:
                            results[job.index] = scored_dict(
                                job.index,
                                RUNNER_LABEL_DAYTONA,
                                runner_error(job.index, trials, "cancelled"),
                                [],
                                str(job.local_out),
                            )
                log.warning(
                    "scoring cancelled; %d sandbox run(s) still finishing in the background",
                    sum(t.is_alive() for t in threads),
                )
            else:
                self._download_winners(jobs, used, results, emit)
        finally:
            if not abandoned.is_set():
                self._release(slots)
            else:
                threading.Thread(target=self._release_when_done, args=(threads, slots), daemon=True).start()
        return [results[index] for index in sorted(results)]

    def _job_thread(
        self, slot: _Slot, job: _Job, emit: Callable[[dict], None], cancel: threading.Event | None
    ) -> None:
        try:
            scored = self._run_job(slot, job, cancel)
        except Exception as exc:  # noqa: BLE001 - never lose a candidate
            log.exception("candidate %d: pool failure", job.index)
            scored = scored_dict(
                job.index,
                RUNNER_LABEL_DAYTONA,
                runner_error(job.index, job.trials, f"pool: {type(exc).__name__}: {exc}"),
                [],
                str(job.local_out),
            )
        emit(scored)

    @staticmethod
    def _wait(threads: list[threading.Thread], cancel: threading.Event | None) -> None:
        while any(t.is_alive() for t in threads):
            if cancel is not None and cancel.is_set():
                # give the threads a moment to notice the cancel themselves
                for thread in threads:
                    thread.join(POLL_S)
                return
            for thread in threads:
                thread.join(POLL_S)

    def _release_when_done(self, threads: list[threading.Thread], slots: list[_Slot]) -> None:
        for thread in threads:
            thread.join()
        self._release(slots)

    def _download_winners(
        self,
        jobs: list[_Job],
        used: dict[int, _Slot],
        results: dict[int, dict],
        emit: Callable[[dict], None],
    ) -> None:
        """Fetch traj/final_scene/trials/score for the two best passing sandbox candidates."""
        passing = [
            r["score"]
            for r in results.values()
            if r["score"].get("passed") and r["runner"] == RUNNER_LABEL_DAYTONA
        ]
        winners = [int(s["candidate"]) for s in scoring.rank(passing)[:WINNERS]]
        by_index = {job.index: job for job in jobs}
        for index in winners:
            job, slot = by_index.get(index), used.get(index)
            if job is None or slot is None:
                continue
            try:
                self._download_outputs(slot, job)
                log.info("candidate %d: outputs downloaded to %s", index, job.local_out)
            except _SandboxFailure as exc:
                self._record_failure(slot, f"candidate {index}: {exc}")
                log.warning(
                    "candidate %d: re-scoring locally because its outputs could not be downloaded", index
                )
                emit(job.local())

    def plan(
        self,
        skill_path: str | os.PathLike[str],
        args: dict,
        scene_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        skills_dir: str | os.PathLike[str] | None,
        timeout: float = PLAN_TIMEOUT,
        *,
        cancel: threading.Event | None = None,
    ) -> dict:
        skill = pathlib.Path(skill_path).resolve()
        scene = pathlib.Path(scene_path).resolve()
        out = pathlib.Path(out_dir).resolve()
        skills = None if skills_dir is None else pathlib.Path(skills_dir).resolve()
        args = dict(args or {})
        slots = self._acquire("plan", 1) or self._acquire("score", 1)
        if not slots:
            log.info("plan runs locally: no free sandbox")
            return self.local.plan(skill, args, scene, out, skills, timeout=timeout, cancel=cancel)
        slot = slots[0]
        try:
            task_name = self._task_name(out.parent if out.parent != out else out)
            remote_task = f"{REMOTE_ROOT}/tasks/{task_name}"
            uploads = [(skill, "skill.json"), (scene, "scene.json")] + self._skill_uploads(skills)
            flags = runner_args(
                skill=f"{remote_task}/skill.json",
                args=args,
                scene=f"{remote_task}/scene.json",
                out=f"{remote_task}/plan",
                trials=1,
                seed=DEFAULT_SEED,
                timeout=timeout,
                skills_dir=f"{remote_task}/skills" if skills is not None else None,
            )

            def local() -> dict:
                return self.local.plan(skill, args, scene, out, skills, timeout=timeout, cancel=cancel)

            job = _Job(0, task_name, "plan", out, uploads, flags, 1, float(timeout), True, local)
            return self._run_job(slot, job, cancel)
        finally:
            self._release(slots)


# ---------------------------------------------------------------------------- singleton


_pool: SandboxPool | LocalRunner | None = None
_pool_lock = threading.Lock()


def get_pool() -> SandboxPool | LocalRunner:
    """The process-wide pool: a LocalRunner under FAKE_SANDBOX/FAKE or without DAYTONA_API_KEY, else an
    unstarted SandboxPool (the orchestrator calls `start()`)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            if _fake("FAKE_SANDBOX") or not os.environ.get("DAYTONA_API_KEY"):
                _pool = LocalRunner()
            else:
                _pool = SandboxPool()
        return _pool


def reset_pool() -> None:
    """Forget the singleton (shutting a SandboxPool down first); for tests."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if isinstance(pool, SandboxPool):
        pool.shutdown()


# ---------------------------------------------------------------------------- bench CLI

FIXTURES = ROOT / "tests" / "fixtures"
JUICE_SPEC = FIXTURES / "llm" / "router" / "learn_juice_box_to_tray.json"
JUICE_CANDIDATES = FIXTURES / "llm" / "candidates" / "juice_box_to_tray"
CLEAR_TABLE_CANDIDATE = FIXTURES / "candidate_clear_table.py"
CLEAR_TABLE_SPEC = FIXTURES / "task_spec_clear_table.json"
STARTER_DIR = ROOT / "skills" / "_starter"
FAKE_SCENE = ROOT / "config" / "fake_scene.json"
ROUTINE_CANDIDATE = '''"""Bench: the starter place_in_tray skill through run_skill (what a routine does)."""
import math


def run(robot, args):
    return robot.run_skill("place_in_tray", {"object": "juice_box"})
'''


def _bench_tasks(out_root: pathlib.Path) -> list[dict]:
    """Three fixture tasks: juice_box_to_tray (its six fixtures), clear_table (the contract candidate x6)
    and place_in_tray (the starter skill called from a routine candidate x6, with --skills-dir)."""
    tasks: list[dict] = []
    # 1. the six juice_box_to_tray fixtures with the router fixture's spec
    task_dir = out_root / "juice_box_to_tray"
    task_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(JUICE_SPEC.read_text(encoding="utf-8"))["spec"]
    (task_dir / "task.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copy(FAKE_SCENE, task_dir / "scene.json")
    cands = []
    for i in range(1, 7):
        shutil.copy(JUICE_CANDIDATES / f"cand_{i}.py", task_dir / f"cand_{i}.py")
        cands.append((i, task_dir / f"cand_{i}.py"))
    tasks.append({"name": "juice_box_to_tray", "dir": task_dir, "candidates": cands, "skills_dir": None})
    # 2. the contract's clear_table candidate, six copies
    task_dir = out_root / "clear_table"
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLEAR_TABLE_SPEC, task_dir / "task.json")
    shutil.copy(FAKE_SCENE, task_dir / "scene.json")
    cands = []
    for i in range(1, 7):
        shutil.copy(CLEAR_TABLE_CANDIDATE, task_dir / f"cand_{i}.py")
        cands.append((i, task_dir / f"cand_{i}.py"))
    tasks.append({"name": "clear_table", "dir": task_dir, "candidates": cands, "skills_dir": None})
    # 3. the starter place_in_tray skill, run through run_skill by a routine-style candidate
    task_dir = out_root / "place_in_tray"
    task_dir.mkdir(parents=True, exist_ok=True)
    starter = json.loads((STARTER_DIR / "place_in_tray.json").read_text(encoding="utf-8"))
    spec = json.loads(json.dumps(starter["spec"]).replace("$object", "juice_box"))
    spec["params"] = {}
    (task_dir / "task.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copy(FAKE_SCENE, task_dir / "scene.json")
    cands = []
    for i in range(1, 7):
        (task_dir / f"cand_{i}.py").write_text(ROUTINE_CANDIDATE, encoding="utf-8")
        cands.append((i, task_dir / f"cand_{i}.py"))
    tasks.append({"name": "place_in_tray", "dir": task_dir, "candidates": cands, "skills_dir": STARTER_DIR})
    return tasks


def bench(out_root: pathlib.Path, trials: int = DEFAULT_TRIALS) -> list[dict]:
    """Score the three fixture tasks on the current pool; prints and returns the wall clock per task."""
    pool = get_pool()
    kind = type(pool).__name__
    print(f"pool: {kind}", flush=True)
    rows: list[dict] = []
    try:
        if isinstance(pool, SandboxPool):
            t0 = time.perf_counter()
            live = pool.start()
            print(f"start: {live} sandboxes live in {time.perf_counter() - t0:.1f} s", flush=True)
        for task in _bench_tasks(out_root):
            t0 = time.perf_counter()
            results = pool.score(
                task["dir"],
                task["candidates"],
                task["dir"] / "task.json",
                task["dir"] / "scene.json",
                task["skills_dir"],
                trials=trials,
                seed=DEFAULT_SEED,
                timeout=DEFAULT_TIMEOUT,
                on_score=lambda s: print(
                    f"  candidate {s['candidate']} ({s['runner']}): passed={s['score']['passed']} "
                    f"successes={s['score']['successes']}/{s['score']['trials']} "
                    f"fail_reason={s['score']['fail_reason']!r}",
                    flush=True,
                ),
            )
            wall = time.perf_counter() - t0
            passed = sum(1 for r in results if r["score"]["passed"])
            runners = sorted({r["runner"] for r in results})
            print(
                f"task {task['name']}: {len(results)} candidates x {trials} trials, {passed} passed, "
                f"runner {'/'.join(runners)}, {wall:.1f} s",
                flush=True,
            )
            rows.append(
                {"task": task["name"], "candidates": len(results), "passed": passed, "wall_s": round(wall, 2)}
            )
    finally:
        if isinstance(pool, SandboxPool):
            pool.shutdown()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.sandbox_pool", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--bench", action="store_true", help="score three fixture tasks and print seconds per task"
    )
    parser.add_argument(
        "--build-snapshot",
        nargs="?",
        const=SNAPSHOT,
        metavar="NAME",
        help="build the sandbox image (needs a key)",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "runs" / "bench"), help="where the bench writes (default runs/bench)"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    if ns.build_snapshot:
        if not os.environ.get("DAYTONA_API_KEY"):
            print("DAYTONA_API_KEY is not set", file=sys.stderr)
            return 2
        build_snapshot(ns.build_snapshot)
        print(f"snapshot {ns.build_snapshot} built")
        return 0
    if ns.bench:
        rows = bench(pathlib.Path(ns.out), ns.trials)
        return 0 if rows else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    sys.exit(main())
