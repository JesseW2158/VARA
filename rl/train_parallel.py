"""Parallel reward search: four reward variants, four Daytona sandboxes, one best policy (tasks D3, D5).

`train_parallel(rewards, *, steps=600_000, out_dir=ROOT / "rl" / "runs" / "parallel", local=False,
poll_s=30, n_envs=None, eval_episodes=100, seed=0, policy_dest=BEST_POLICY, curves_path=CURVES_PATH,
hints=None, report=None, max_wait_s=3600, snapshot=SNAPSHOT_NAME) -> dict` trains one PPO policy per
reward file (at most MAX_SANDBOXES = 4 files), collects `policy.zip`, `progress.csv`, `summary.json` and
`train.log` of each run in `out_dir/<variant>/`, picks the best by evaluation success rate, then by mean
steps to success, then by order, copies its policy to `rl/policies/handoff_best.zip`, and writes
`web/static/rl/curves.json` for the dashboard: `{"variants": [{"name", "hint", "steps": [...],
"success_rate": [...], "mean_reward": [...], "chosen": bool, "final_success_rate"}], "created"}` (the lists
come from progress.csv, NaN becomes null; the `success` alias per variant and the top-level `chosen` name
are what web/static/app.js reads). A variant is named after its file (`variant_3`), its `hint` is the
shaping idea of docs/tasks/D-rl.md D4 for `variant_<k>.py` and the file name otherwise. The returned dict
holds `variants` (name, hint, reward, runner, out_dir, summary, progress, policy, success_rate,
mean_steps_to_success, final_success_rate, error, chosen), `chosen` (a name or None), `policy`, `curves`,
`runner`, `steps` and `created`; it is also written to `out_dir/result.json`.

Daytona path (`local=False`): sandboxes from the snapshot SNAPSHOT_NAME ("arm-rl-v1", built once by
`build_rl_snapshot()`: debian slim 3.12 with CPU-only torch from TORCH_INDEX_URL, mujoco, numpy, pyyaml,
gymnasium, stable-baselines3, jsonschema, Resources(cpu=4, memory=8, disk=10)), created with
`auto_stop_interval=0` because a training run outlasts the idle auto-stop, one per reward, never more than
four, and only here (CLAUDE.md hard rule 8). Each gets the code bundle (a tar.gz of `rl/*.py`,
`rl/rewards/__init__.py` and `baseline.py`, `arm/ik.py`, `arm/__init__.py`, `sim/model/arm_only.xml`,
`sim/model/so101_new_calib.xml` and its STL assets, `sim/restricted.py`, `sim/candidate_check.py`,
`sim/__init__.py`, `sim/model/__init__.py`, `config/limits.yaml`, `config/scene.yaml`) and its reward as
`reward_<k>.py` in REMOTE_DIR, starts `nohup python -m rl.train --reward reward_k.py --steps N --out out/
... > train.log 2>&1 &`, is polled every `poll_s` seconds with `tail -n 1 out/progress.csv` (and a status
probe: `out/summary.json` written means done, the pid gone without it means dead), has its results
downloaded, and is deleted in a `finally` block: on success, on any exception and on KeyboardInterrupt
(SIGINT; a SIGTERM is turned into KeyboardInterrupt while sandboxes exist). The client comes from the
module-level `_daytona_factory()` so tests inject a fake with the same surface (`create`, `fs.upload_file`,
`process.exec`, `fs.download_file`, `delete`); the verified daytona 0.225 names are listed next to it.

Local path (`local=True`; also the CLI's automatic fallback when DAYTONA_API_KEY is unset): the same
trainings as subprocesses `sys.executable -m rl.train ...` with the clean environment (PATH, PYTHONPATH,
SYSTEMROOT, TEMP, TMP and USERNAME: torch needs the user name on Windows), `n_envs` LOCAL_N_ENVS (4) unless
given. Meant for tests (tiny step counts) and the fixture/baseline rewards: Grok-written variants train only
in sandboxes (.claude/rules/rl.md).

    python -m rl.train_parallel [--rewards F ...] [--steps 600000] [--out DIR] [--local] [--n-envs N]
        [--eval-episodes 100] [--seed 0] [--poll-s 30] [--build-snapshot] [--snapshot arm-rl-v1]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Sequence
from typing import Any

__all__ = [
    "BEST_POLICY",
    "BUNDLE_FILES",
    "BUNDLE_GLOBS",
    "CURVES_PATH",
    "DEFAULT_OUT",
    "DEFAULT_REWARDS",
    "DEFAULT_STEPS",
    "LOCAL_ENV_KEYS",
    "LOCAL_N_ENVS",
    "MAX_SANDBOXES",
    "REMOTE_DIR",
    "RESULT_FILES",
    "SNAPSHOT_NAME",
    "build_rl_snapshot",
    "bundle_paths",
    "clean_env",
    "hint_for",
    "make_bundle",
    "read_progress",
    "read_summary",
    "select_best",
    "train_parallel",
    "variant_names",
    "write_curves",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT_NAME = "arm-rl-v1"
MAX_SANDBOXES = 4  # hard cap on the RL sandboxes (CLAUDE.md hard rule 8)
SANDBOX_RESOURCES = {"cpu": 4, "memory": 8, "disk": 10}  # GiB for memory and disk
PIP_PACKAGES = ("mujoco", "numpy", "pyyaml", "gymnasium", "stable-baselines3", "jsonschema")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
REMOTE_DIR = "/home/daytona/vara"
DEFAULT_STEPS = 600_000
DEFAULT_OUT = ROOT / "rl" / "runs" / "parallel"
POLICIES_DIR = ROOT / "rl" / "policies"
BEST_POLICY = POLICIES_DIR / "handoff_best.zip"
CURVES_PATH = ROOT / "web" / "static" / "rl" / "curves.json"
DEFAULT_REWARDS = tuple(ROOT / "rl" / "rewards" / f"variant_{k}.py" for k in range(1, MAX_SANDBOXES + 1))
RESULT_FILES = ("policy.zip", "progress.csv", "summary.json")
BUNDLE_FILES = (
    "arm/__init__.py",
    "arm/ik.py",
    "sim/__init__.py",
    "sim/candidate_check.py",
    "sim/restricted.py",
    "sim/model/__init__.py",
    "sim/model/arm_only.xml",
    "sim/model/so101_new_calib.xml",
    "config/limits.yaml",
    "config/scene.yaml",
    "rl/rewards/__init__.py",
    "rl/rewards/baseline.py",
)
BUNDLE_GLOBS = ("rl/*.py", "sim/model/assets/*.stl")
LOCAL_ENV_KEYS = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERNAME")
LOCAL_N_ENVS = 4
CREATE_TIMEOUT_S = 120
EXEC_TIMEOUT_S = 60
MAX_WAIT_S = 3600.0  # applies by default to both local training and Daytona polling
# Shell probes run in REMOTE_DIR: `kill -0` is a shell builtin, so no procps is needed in the slim image.
STATUS_CMD = (
    "if [ -f out/summary.json ]; then echo done; "
    "elif [ -f train.pid ] && kill -0 $(cat train.pid) 2>/dev/null; then echo running; "
    "else echo dead; fi"
)
PROGRESS_CMD = "tail -n 1 out/progress.csv 2>/dev/null || true"
LOG_TAIL_CMD = "tail -n 20 train.log 2>/dev/null || true"

log = logging.getLogger(__name__)
sleep: Callable[[float], None] = time.sleep  # a module attribute so tests replace it


# ------------------------------------------------------------------------------------------- Daytona


def _daytona_factory() -> Any:
    """The Daytona client (reads DAYTONA_API_KEY, DAYTONA_API_URL and DAYTONA_TARGET from the environment).

    Tests replace this function with one returning a fake. Verified against daytona 0.225:
    `Daytona.create(params, *, timeout) -> Sandbox`, `CreateSandboxFromSnapshotParams(snapshot,
    auto_stop_interval, labels)`, `sandbox.fs.upload_file(bytes, dst)`, `sandbox.fs.download_file(path) ->
    bytes`, `sandbox.process.exec(cmd, cwd=..., timeout=...) -> ExecuteResponse(exit_code, result)`,
    `sandbox.delete()`, `daytona.snapshot.create(CreateSnapshotParams(name, image, resources), on_logs=...)`,
    `Image.debian_slim("3.12").pip_install(*packages, index_url=...)`, `Resources(cpu, memory, disk)`.
    """
    from daytona import Daytona

    return Daytona()


def build_rl_snapshot(name: str = SNAPSHOT_NAME, *, on_logs: Callable[[str], None] | None = print) -> Any:
    """Build the RL snapshot: debian slim 3.12, CPU-only torch, then the simulation and RL packages (D3)."""
    from daytona import CreateSnapshotParams, Image, Resources

    image = (
        Image.debian_slim("3.12").pip_install("torch", index_url=TORCH_INDEX_URL).pip_install(*PIP_PACKAGES)
    )
    params = CreateSnapshotParams(name=name, image=image, resources=Resources(**SANDBOX_RESOURCES))
    return _daytona_factory().snapshot.create(params, on_logs=on_logs)


# -------------------------------------------------------------------------------------------- bundle


def bundle_paths(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every file of the code bundle a training sandbox needs (FileNotFoundError when one is missing)."""
    paths = [root / rel for rel in BUNDLE_FILES]
    for pattern in BUNDLE_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle files missing under {root}: {', '.join(missing)}")
    return paths


def make_bundle(root: pathlib.Path = ROOT) -> bytes:
    """The code bundle as tar.gz bytes, paths relative to the repo root (extracted in REMOTE_DIR)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in bundle_paths(root):
            tar.add(path, arcname=path.relative_to(root).as_posix())
    return buf.getvalue()


# ------------------------------------------------------------------------------------------- helpers


def variant_names(rewards: Sequence[pathlib.Path]) -> list[str]:
    """One name per reward file: its stem, suffixed with its position when two stems collide."""
    names: list[str] = []
    for k, path in enumerate(rewards, 1):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", pathlib.Path(path).stem) or f"variant_{k}"
        names.append(stem if stem not in names else f"{stem}_{k}")
    return names


def hint_for(path: pathlib.Path) -> str:
    """The shaping idea behind `variant_<k>.py` (docs/tasks/D-rl.md D4), else the file name."""
    stem = pathlib.Path(path).stem
    match = re.fullmatch(r"variant_(\d+)", stem)
    if match:
        try:
            from rl.rewards.write_rewards import SHAPING_IDEAS
        except ImportError:  # inside a bundle without agents/
            SHAPING_IDEAS = ()
        k = int(match.group(1))
        if 1 <= k <= len(SHAPING_IDEAS):
            return SHAPING_IDEAS[k - 1]
    return stem


def clean_env() -> dict[str, str]:
    """The environment of a local training subprocess: PATH, PYTHONPATH and what Windows needs, no keys."""
    env = {key: os.environ[key] for key in LOCAL_ENV_KEYS if key in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    return env


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def read_progress(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """The rows of a progress.csv as `{"step", "mean_reward", "success_rate"}` (NaN -> None), or []."""
    rows: list[dict[str, Any]] = []
    try:
        with pathlib.Path(path).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                step = _int_or_none(raw.get("step"))
                if step is None:
                    continue
                rows.append(
                    {
                        "step": step,
                        "mean_reward": _float_or_none(raw.get("mean_reward")),
                        "success_rate": _float_or_none(raw.get("success_rate")),
                    }
                )
    except OSError:
        return []
    return rows


def _finite(value: Any) -> Any:
    """`value` with every non-finite float replaced by None, recursively (a summary may carry NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def read_summary(path: str | pathlib.Path) -> dict[str, Any] | None:
    """A run's summary.json as a dict with NaN/inf turned into None, or None when missing or malformed.

    rl/train.py writes `final_progress.mean_reward` as NaN when no episode finished inside the run, and
    Python's json module emits it as the invalid token `NaN`; curves.json and result.json never carry it.
    """
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _finite(data) if isinstance(data, dict) else None


def _tail(path: pathlib.Path) -> str | None:
    """The last non-empty line of a text file, or None."""
    try:
        lines = [
            line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()
        ]
    except OSError:
        return None
    return lines[-1] if lines else None


def _last_error(text: str | None, default: str) -> str:
    """The last `error: ...` line of a training log, else its last line, else `default`."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("error:"):
            return line
    return lines[-1] if lines else default


def _progress_line(name: str, tail: str | None) -> str:
    if not tail or tail.lower().startswith("step"):
        return f"{name}: waiting for the first progress row"
    parts = tail.split(",")
    if len(parts) >= 3:
        return (
            f"{name}: step {parts[0].strip()} mean_reward {parts[1].strip()} success_rate {parts[2].strip()}"
        )
    return f"{name}: {tail.strip()}"


def _train_command(
    reward: str, steps: int, out: str, n_envs: int | None, eval_episodes: int, seed: int
) -> list[str]:
    cmd = ["python", "-m", "rl.train", "--reward", reward, "--steps", str(int(steps)), "--out", out]
    if n_envs is not None:
        cmd += ["--n-envs", str(int(n_envs))]
    cmd += ["--seed", str(int(seed)), "--eval-episodes", str(int(eval_episodes))]
    return cmd


# ---------------------------------------------------------------------------------------- local path


def _poll_pause(started: float, poll_s: float, max_wait_s: float, runner: str) -> None:
    remaining = max_wait_s - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError(f"{runner} training exceeded {max_wait_s:g} s")
    sleep(min(poll_s, remaining))
    if time.monotonic() - started >= max_wait_s:
        raise TimeoutError(f"{runner} training exceeded {max_wait_s:g} s")


def _kill_local(proc: subprocess.Popen) -> None:
    """Stop this training and its environment workers, including a worker stuck in native code."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def _run_local(
    runs: list[dict[str, Any]],
    steps: int,
    n_envs: int,
    eval_episodes: int,
    seed: int,
    poll_s: float,
    say: Callable[[str], None],
    max_wait_s: float,
) -> None:
    """The trainings as local subprocesses of rl.train, polled until every one has ended."""
    procs: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any]] = []
    started = time.monotonic()
    try:
        for run in runs:
            vdir = pathlib.Path(run["out_dir"])
            vdir.mkdir(parents=True, exist_ok=True)
            cmd = _train_command(run["reward"], steps, str(vdir), n_envs, eval_episodes, seed)
            cmd[0] = sys.executable
            handle = (vdir / "train.log").open("wb")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=clean_env(),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
            except BaseException:
                handle.close()
                raise
            procs.append((run, proc, handle))
            say(f"{run['name']}: training started locally (pid {proc.pid}, {n_envs} envs, {steps} steps)")
        pending = set(range(len(procs)))
        while pending:
            _poll_pause(started, poll_s, max_wait_s, "local")
            for i in sorted(pending):
                run, proc, handle = procs[i]
                code = proc.poll()
                say(_progress_line(run["name"], _tail(pathlib.Path(run["out_dir"]) / "progress.csv")))
                if code is None:
                    continue
                pending.discard(i)
                handle.close()
                if code != 0:
                    log_text = None
                    try:
                        log_text = (pathlib.Path(run["out_dir"]) / "train.log").read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        pass
                    run["error"] = _last_error(log_text, f"rl.train exited with code {code}")
                    say(f"{run['name']}: failed ({run['error']})")
                else:
                    say(f"{run['name']}: finished")
            if pending and time.monotonic() - started > max_wait_s:
                raise TimeoutError(f"local training exceeded {max_wait_s:g} s")
    finally:
        for run, proc, handle in procs:
            if proc.poll() is None:
                try:
                    _kill_local(proc)
                except (OSError, subprocess.TimeoutExpired):
                    log.error("%s: training process %d did not stop", run["name"], proc.pid)
            if not handle.closed:
                handle.close()


# -------------------------------------------------------------------------------------- sandbox path


def _result_text(response: Any) -> str:
    return str(getattr(response, "result", "") or "")


def _check(response: Any, what: str) -> Any:
    code = getattr(response, "exit_code", 0)
    if code not in (0, None):
        raise RuntimeError(f"{what} failed in the sandbox (exit {code}): {_result_text(response)[:300]}")
    return response


def _install_sigterm() -> Any:
    """Turn SIGTERM into KeyboardInterrupt so the `finally` deletes the sandboxes; None when not possible."""

    def handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        return signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError, AttributeError):  # not the main thread, or no SIGTERM on this platform
        return None


def _restore_sigterm(previous: Any) -> None:
    if previous is None:
        return
    try:
        signal.signal(signal.SIGTERM, previous)
    except (ValueError, OSError, AttributeError):
        pass


def _download(run: dict[str, Any], sandbox: Any, say: Callable[[str], None]) -> None:
    out = pathlib.Path(run["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    wanted = [(f"{REMOTE_DIR}/out/{name}", name) for name in RESULT_FILES] + [
        (f"{REMOTE_DIR}/train.log", "train.log")
    ]
    for remote, name in wanted:
        try:
            data = sandbox.fs.download_file(remote)
        except Exception as exc:  # a missing file (a run that died) is reported, not fatal
            say(f"{run['name']}: {name} not downloaded ({type(exc).__name__}: {exc})")
            continue
        if data is None:
            continue
        (out / name).write_bytes(data if isinstance(data, bytes) else str(data).encode("utf-8"))


def _run_sandboxes(
    runs: list[dict[str, Any]],
    steps: int,
    n_envs: int | None,
    eval_episodes: int,
    seed: int,
    poll_s: float,
    say: Callable[[str], None],
    max_wait_s: float,
    snapshot: str,
) -> None:
    """Create one sandbox per run, train, poll, download; delete every sandbox in `finally`."""
    from daytona import CreateSandboxFromSnapshotParams

    bundle = make_bundle()
    daytona = _daytona_factory()
    sandboxes: list[tuple[dict[str, Any], Any]] = []
    previous = _install_sigterm()
    started = time.monotonic()
    try:
        for k, run in enumerate(runs, 1):
            if len(sandboxes) >= MAX_SANDBOXES:  # cannot happen: train_parallel caps the rewards first
                raise RuntimeError(f"refusing to create more than {MAX_SANDBOXES} RL sandboxes")
            params = CreateSandboxFromSnapshotParams(
                snapshot=snapshot, auto_stop_interval=0, labels={"vara": "rl", "variant": run["name"]}
            )
            sandbox = daytona.create(params, timeout=CREATE_TIMEOUT_S)
            sandboxes.append((run, sandbox))
            run["sandbox"] = str(getattr(sandbox, "id", "?"))
            say(f"{run['name']}: sandbox {run['sandbox']} created from {snapshot}")
            _check(sandbox.process.exec(f"mkdir -p {REMOTE_DIR}", timeout=EXEC_TIMEOUT_S), "mkdir")
            sandbox.fs.upload_file(bundle, f"{REMOTE_DIR}/bundle.tgz")
            sandbox.fs.upload_file(pathlib.Path(run["reward"]).read_bytes(), f"{REMOTE_DIR}/reward_{k}.py")
            _check(sandbox.process.exec("tar -xzf bundle.tgz", cwd=REMOTE_DIR, timeout=EXEC_TIMEOUT_S), "tar")
            train = " ".join(_train_command(f"reward_{k}.py", steps, "out/", n_envs, eval_episodes, seed))
            start = f"nohup {train} > train.log 2>&1 & echo $! > train.pid"
            _check(sandbox.process.exec(start, cwd=REMOTE_DIR, timeout=EXEC_TIMEOUT_S), "training start")
            say(f"{run['name']}: training started ({train})")
        pending = set(range(len(sandboxes)))
        while pending:
            _poll_pause(started, poll_s, max_wait_s, "sandbox")
            for i in sorted(pending):
                if time.monotonic() - started >= max_wait_s:
                    raise TimeoutError(f"sandbox training exceeded {max_wait_s:g} s")
                run, sandbox = sandboxes[i]
                progress = _result_text(
                    sandbox.process.exec(PROGRESS_CMD, cwd=REMOTE_DIR, timeout=EXEC_TIMEOUT_S)
                )
                say(_progress_line(run["name"], _tail_text(progress)))
                status = _tail_text(
                    _result_text(sandbox.process.exec(STATUS_CMD, cwd=REMOTE_DIR, timeout=EXEC_TIMEOUT_S))
                )
                if status == "running":
                    continue
                pending.discard(i)
                _download(run, sandbox, say)
                if status == "done":
                    say(f"{run['name']}: finished")
                    continue
                tail = _result_text(
                    sandbox.process.exec(LOG_TAIL_CMD, cwd=REMOTE_DIR, timeout=EXEC_TIMEOUT_S)
                )
                run["error"] = _last_error(tail, f"training ended without summary.json (status {status!r})")
                say(f"{run['name']}: failed ({run['error']})")
            if pending and time.monotonic() - started > max_wait_s:
                raise TimeoutError(f"sandbox training exceeded {max_wait_s:g} s")
    finally:
        _restore_sigterm(previous)
        for run, sandbox in sandboxes:
            try:
                sandbox.delete()
                say(f"{run['name']}: sandbox {run.get('sandbox', '?')} deleted")
            except Exception as exc:  # keep deleting the others; the leak is reported, not hidden
                log.error("%s: sandbox %s delete failed: %s", run["name"], run.get("sandbox", "?"), exc)
                say(f"{run['name']}: sandbox delete FAILED ({type(exc).__name__}: {exc})")


def _tail_text(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


# ----------------------------------------------------------------------------------------- results


def _collect(run: dict[str, Any]) -> dict[str, Any]:
    out = pathlib.Path(run["out_dir"])
    summary = read_summary(out / "summary.json")
    progress = read_progress(out / "progress.csv")
    policy = out / "policy.zip"
    rate = _float_or_none(summary.get("success_rate")) if summary else None
    mean = _float_or_none(summary.get("mean_steps_to_success")) if summary else None
    last = next((row["success_rate"] for row in reversed(progress) if row["success_rate"] is not None), None)
    error = run.get("error")
    if error is None and summary is None:
        error = "no summary.json after training"
    if error is None and not policy.is_file():
        error = "no policy.zip after training"
    if error is None and rate is not None and not 0.0 <= rate <= 1.0:
        error = "invalid evaluation success_rate: expected a value from 0 to 1"
    return {
        **run,
        "summary": summary,
        "progress": progress,
        "policy": str(policy) if policy.is_file() else None,
        "success_rate": rate,
        "mean_steps_to_success": mean,
        "final_success_rate": rate if rate is not None else last,
        "error": error,
        "chosen": False,
    }


def select_best(variants: Sequence[dict[str, Any]]) -> int | None:
    """Index of the best variant: highest evaluation success rate, then fewest mean steps to success, then
    the earliest; only variants with a policy file and a measured success rate count (None when none)."""
    best: int | None = None
    best_key: tuple[float, float, int] | None = None
    for i, variant in enumerate(variants):
        rate = _float_or_none(variant.get("success_rate"))
        policy = variant.get("policy")
        if (
            variant.get("error")
            or rate is None
            or not 0 <= rate <= 1
            or not policy
            or not pathlib.Path(policy).is_file()
        ):
            continue
        mean = _float_or_none(variant.get("mean_steps_to_success"))
        if mean is not None and mean <= 0:
            continue
        key = (-float(rate), float(mean) if mean is not None else math.inf, i)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def write_curves(
    variants: Sequence[dict[str, Any]], path: str | pathlib.Path, created: str
) -> dict[str, Any]:
    """Write curves.json in the documented shape (plus the dashboard's aliases) and return it."""
    chosen_name = next((v["name"] for v in variants if v.get("chosen")), None)
    entries = []
    for variant in variants:
        rows = variant.get("progress") or []
        success = [row["success_rate"] for row in rows]
        entries.append(
            {
                "name": variant["name"],
                "hint": variant["hint"],
                "steps": [row["step"] for row in rows],
                "success_rate": success,
                "mean_reward": [row["mean_reward"] for row in rows],
                "success": success,
                "chosen": bool(variant.get("chosen")),
                "final_success_rate": variant.get("final_success_rate"),
                "mean_steps_to_success": variant.get("mean_steps_to_success"),
                "error": variant.get("error"),
            }
        )
    data = {"variants": entries, "chosen": chosen_name, "created": created}
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return data


def train_parallel(
    rewards: Sequence[str | pathlib.Path],
    *,
    steps: int = DEFAULT_STEPS,
    out_dir: str | pathlib.Path = DEFAULT_OUT,
    local: bool = False,
    poll_s: float = 30.0,
    n_envs: int | None = None,
    eval_episodes: int = 100,
    seed: int = 0,
    policy_dest: str | pathlib.Path = BEST_POLICY,
    curves_path: str | pathlib.Path = CURVES_PATH,
    hints: Sequence[str] | None = None,
    report: Callable[[str], None] | None = None,
    max_wait_s: float | None = MAX_WAIT_S,
    snapshot: str = SNAPSHOT_NAME,
) -> dict[str, Any]:
    """Train every reward (at most four), pick the best policy, write curves.json; see the module doc.

    Raises ValueError for more than MAX_SANDBOXES rewards or bad numbers (before any sandbox exists),
    FileNotFoundError for a missing reward or bundle file, RuntimeError when a sandbox step fails, and
    TimeoutError past `max_wait_s`; every sandbox created is deleted on the way out.
    """
    paths = [pathlib.Path(r) for r in rewards]
    if not paths:
        raise ValueError("train_parallel needs at least one reward file")
    if len(paths) > MAX_SANDBOXES:
        raise ValueError(f"at most {MAX_SANDBOXES} reward variants train at once, got {len(paths)}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"no such reward file: {path}")
    if int(steps) < 1 or int(eval_episodes) < 0 or int(seed) < 0:
        raise ValueError("steps must be >= 1, eval_episodes >= 0 and seed >= 0")
    if not math.isfinite(float(poll_s)) or not (float(poll_s) > 0):
        raise ValueError(f"poll_s must be finite and positive, got {poll_s!r}")
    max_wait_s = MAX_WAIT_S if max_wait_s is None else float(max_wait_s)
    if not math.isfinite(max_wait_s) or max_wait_s <= 0:
        raise ValueError("max_wait_s must be finite and positive")
    if n_envs is not None and n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if not str(out_dir).strip():
        raise ValueError("out_dir must be a non-empty output directory")
    if hints is not None and len(hints) != len(paths):
        raise ValueError("hints must have one entry per reward")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = report if report is not None else (lambda line: log.info("%s", line))
    names = variant_names(paths)
    for name in names:
        if any((out / name / artifact).exists() for artifact in RESULT_FILES):
            raise ValueError(f"existing training artifacts in {out / name}; choose a fresh output directory")
    runner = "local" if local else "daytona"
    runs: list[dict[str, Any]] = [
        {
            "name": name,
            "hint": (hints[i] if hints is not None else hint_for(path)),
            "reward": str(path),
            "runner": runner,
            "out_dir": str(out / name),
            "error": None,
        }
        for i, (name, path) in enumerate(zip(names, paths, strict=True))
    ]
    if local:
        _run_local(
            runs,
            int(steps),
            int(n_envs or LOCAL_N_ENVS),
            int(eval_episodes),
            int(seed),
            float(poll_s),
            say,
            max_wait_s,
        )
    else:
        _run_sandboxes(
            runs, int(steps), n_envs, int(eval_episodes), int(seed), float(poll_s), say, max_wait_s, snapshot
        )

    variants = [_collect(run) for run in runs]
    best = select_best(variants)
    copied: str | None = None
    if best is not None:
        variants[best]["chosen"] = True
        dest = pathlib.Path(policy_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(variants[best]["policy"], dest)
        copied = str(dest)
        say(f"chosen: {variants[best]['name']} ({variants[best]['hint']}) -> {dest}")
    else:
        say("no variant produced an evaluated policy; nothing chosen")
    created = _now_iso()
    write_curves(variants, curves_path, created)
    result = {
        "variants": variants,
        "chosen": variants[best]["name"] if best is not None else None,
        "policy": copied,
        "curves": str(curves_path),
        "runner": runner,
        "steps": int(steps),
        "created": created,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------------- CLI


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.train_parallel",
        description=(
            "Train one PPO policy per reward variant in Daytona sandboxes (or locally) and keep the best"
        ),
    )
    parser.add_argument(
        "--rewards", nargs="*", default=None, help="reward files (default rl/rewards/variant_*.py)"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--local", action="store_true", help="local subprocesses instead of Daytona sandboxes"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None, help=f"envs per training (local default {LOCAL_N_ENVS})"
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=MAX_WAIT_S,
        help="maximum wall time for training and polling (default 3600 s)",
    )
    parser.add_argument(
        "--build-snapshot", action="store_true", help="build the RL snapshot first (needs a key)"
    )
    parser.add_argument("--snapshot", default=SNAPSHOT_NAME)
    parser.add_argument("--policy-dest", default=str(BEST_POLICY))
    parser.add_argument("--curves", default=str(CURVES_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    rewards = (
        [pathlib.Path(r) for r in args.rewards]
        if args.rewards
        else [p for p in DEFAULT_REWARDS if p.is_file()]
    )
    if not rewards:
        print("error: no reward files; run `python -m rl.rewards.write_rewards` first", file=sys.stderr)
        return 2
    local = args.local
    if not local and not os.environ.get("DAYTONA_API_KEY"):
        print(
            "DAYTONA_API_KEY is not set: training locally (--local) instead of in Daytona sandboxes",
            file=sys.stderr,
        )
        local = True
    try:
        if args.build_snapshot and not local:
            build_rl_snapshot(args.snapshot)
        result = train_parallel(
            rewards,
            steps=args.steps,
            out_dir=args.out,
            local=local,
            poll_s=args.poll_s,
            n_envs=args.n_envs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            policy_dest=args.policy_dest,
            curves_path=args.curves,
            snapshot=args.snapshot,
            max_wait_s=args.max_wait_s,
        )
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # the Daytona SDK's own DaytonaError family subclasses none of the above
        if not type(exc).__module__.startswith("daytona"):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for variant in result["variants"]:
        mean = variant["mean_steps_to_success"]
        steps_text = "n/a" if mean is None else f"{mean:.1f}"
        flag = "CHOSEN" if variant["chosen"] else ""
        line = f"{variant['name']:<12} {_format_rate(variant['success_rate']):>5}  steps {steps_text:>6}  "
        line += f"{flag:<6}  {variant['hint']}"
        if variant["error"]:
            line += f"  [{variant['error'][:120]}]"
        print(line)
    print(json.dumps({"chosen": result["chosen"], "policy": result["policy"], "curves": result["curves"]}))
    return 0 if result["chosen"] else 1


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
