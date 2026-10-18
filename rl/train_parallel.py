"""Parallel reward search: four reward variants, four Daytona sandboxes, one best policy (tasks D3, D5).

The D5 chain, start to finish (every command from the repo root with the project venv):

1. `FAKE_LLM=1 python -m rl.rewards.write_rewards` (or with XAI_API_KEY) writes the four checked reward
   variants `rl/rewards/variant_1.py` to `variant_4.py` (D4).
2. `python -m rl.train_parallel --build-snapshot --verify-snapshot` builds the RL snapshot SNAPSHOT_NAME
   ("arm-rl-v1") once and checks it (D3): `check_rl_snapshot(name=SNAPSHOT_NAME, *, report=None,
   out_dir=None, build=False) -> dict` (the CLI passes `build` from `--build-snapshot`; the build runs
   first and its outcome is part of the record) creates one sandbox from the snapshot, runs
   SNAPSHOT_CHECK_CMD (`python -c "import torch, stable_baselines3, mujoco; print(<versions>)"`), writes the
   record (the build's outcome, the output, the exit code, the elapsed time and the snapshot name) to
   `runs/rl/snapshot-check-<UTC stamp>.json` (under VARA_RUNS_DIR when set; a new file per call) and
   deletes the sandbox in `finally`. The command exits 0 only when the import exited 0 and the sandbox is
   gone; 1 otherwise (a failed build or import, a Daytona error, or an account that already holds
   MAX_SANDBOXES RL sandboxes; each one is in the record); 2 without DAYTONA_API_KEY (no local fallback)
   or with `--local`, `--rewards`, `--include-run` or `--no-copy`. `--check-snapshot` is an alias. It never
   trains in the same invocation, and its one sandbox counts against MAX_SANDBOXES like a training's (see
   the Daytona path below). `--verify-snapshot` alone checks an existing snapshot.
3. `python -m rl.train_parallel --out rl/runs/<fresh dir> [--include-run rl/runs/baseline]` trains one PPO
   policy per variant in four sandboxes (D5); without DAYTONA_API_KEY, or with `--local`, the same trainings
   run as local subprocesses. Each `--include-run DIR` (repeatable) adds an already trained run
   (`policy.zip`, `summary.json` and the optional `progress.csv`, as `rl/train.py` writes them) to the
   comparison under its directory name without retraining it, so the 507,904-step baseline competes with
   the trained variants; the trained rewards stay capped at MAX_SANDBOXES. `--rewards` with no files plus
   `--include-run` compares existing runs only (no training, no subprocess, no sandbox). `--out` must be a
   fresh directory: one that already holds a `result.json` or a variant's artifacts is refused before
   anything starts, so evidence is never overwritten (the default `rl/runs/parallel` already holds the
   150,000-step run on the demo laptop). The winner's policy is copied to `rl/policies/handoff_best.zip`
   unless `--no-copy` (compare without replacing the deployed policy), and `rl/policies/handoff_best.json`
   is rewritten to describe it unless it already records the same sha256 (`write_provenance()`); the
   curves go to `web/static/rl/curves.json` (tracked in git, so a fresh clone shows the measured curves);
   and `<out>/result.json` names every candidate, the chosen one, its `source_policy` and `policy_sha256`:
   the values handoff_best.json records as the deployed policy's provenance. Pass `--policy-dest` and
   `--curves` (or `--no-copy`) on smoke runs so neither the deployed policy nor the live curves are
   overwritten.
4. `python -m rl.render_video --policy rl/policies/handoff_best.zip --out web/static/rl_sim.mp4` records
   the chosen policy for the dashboard (D6).

`train_parallel(rewards, *, steps=600_000, out_dir=ROOT / "rl" / "runs" / "parallel", local=False,
poll_s=30, n_envs=None, eval_episodes=100, seed=0, policy_dest=BEST_POLICY, curves_path=CURVES_PATH,
hints=None, report=None, max_wait_s=3600, snapshot=SNAPSHOT_NAME, include_runs=None) -> dict` trains one
PPO policy per reward file (at most MAX_SANDBOXES = 4 files; the list may be empty when `include_runs` is
given), collects `policy.zip`, `progress.csv`, `summary.json` and `train.log` of each run in
`out_dir/<variant>/`, appends the included runs (`included_runs()`: a mapping name -> directory, or
directories named after their last path component; validated before anything trains), picks the best by
evaluation success rate, then by mean steps to success, then by order (trained variants first, included
runs after them), copies its policy to `policy_dest` (None skips the copy) and keeps the provenance file
next to the copy true (`write_provenance()`), and writes `curves_path` for the dashboard: `{"variants":
[{"name", "hint", "steps": [...], "success_rate": [...], "mean_reward": [...], "success": [...], "chosen":
bool, "final_success_rate", "mean_steps_to_success", "error"}], "chosen", "created"}` (the lists come from
progress.csv, NaN becomes null; the `success` alias per variant and the top-level `chosen` name are what
web/static/app.js reads). A variant is named after its file (`variant_3`), its `hint` is the shaping idea
of docs/tasks/D-rl.md D4 for `variant_<k>.py` and the file name otherwise; an included run's hint starts
with "existing run". The returned dict holds `variants` (name, hint, reward, runner, out_dir, included,
summary, progress, policy, policy_sha256, success_rate, mean_steps_to_success, final_success_rate, error,
chosen), `candidates` (every name), `included` (the included names), `chosen` (a name or None),
`source_policy` (the chosen policy file, repo-relative when inside the repo), `policy_sha256`, `policy`
(the copy, or None), `provenance` (the provenance file next to the copy, or None), `curves`, `runner`
("daytona", "local" or "existing" when nothing trained), `steps` (the training budget, None when nothing
trained) and `created`; it is also written to `out_dir/result.json`, which must not exist yet.

`write_provenance(variant, policy_dest, *, candidates, result_path, created) -> (path, rewritten)` keeps
`provenance_path(policy_dest)` (`handoff_best.zip` -> `handoff_best.json`) true to the policy just copied
there. A file that already records the same sha256 describes this very policy and is left as it is (its
curated independent evaluation stays); any other is rewritten with the fields scripts/story_numbers.py
reads (`policy`, `sha256`, `source_policy`, `reward`, `requested_steps`, `steps`, `training_seed`,
`selection_reason`, `evaluation` from the run's summary.json, `deployment`) plus `selected_by`, `result`
(this selection's result.json), `created` and `replaces` (the previous file's sha256 and source_policy).

Daytona path (`local=False`): sandboxes from the snapshot SNAPSHOT_NAME ("arm-rl-v1", built once by
`build_rl_snapshot()`: debian slim 3.12 with CPU-only torch from TORCH_INDEX_URL, mujoco, numpy, pyyaml,
gymnasium, stable-baselines3, jsonschema, Resources(cpu=4, memory=8, disk=10)), created with
`auto_stop_interval=0` because a training run outlasts the idle auto-stop, one per reward, never more than
four, and only here (CLAUDE.md hard rule 8). The cap also holds across processes: before creating any
sandbox, a training or a snapshot check counts the account's RL sandboxes (`Daytona.list()`; the `vara`
label `rl` or `rl-check`, RL_LABELS, in any state but destroyed or destroying) and refuses with
RuntimeError, before anything is created, when its own would take the total past MAX_SANDBOXES. A failed
listing is logged and the cap then holds per process only, as in agents/sandbox_pool.py. Each gets the
code bundle (a tar.gz of `rl/*.py`,
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
`check_rl_snapshot()` uses the same factory, parameters and delete-in-`finally` pattern for its one sandbox.

Local path (`local=True`; also the CLI's automatic fallback when DAYTONA_API_KEY is unset): the same
trainings as subprocesses `sys.executable -m rl.train ...` with the clean environment (PATH, PYTHONPATH,
SYSTEMROOT, TEMP, TMP and USERNAME: torch needs the user name on Windows), `n_envs` LOCAL_N_ENVS (4) unless
given. Meant for tests (tiny step counts) and the fixture/baseline rewards: Grok-written variants train only
in sandboxes (.claude/rules/rl.md).

    python -m rl.train_parallel [--rewards F ...] [--steps 600000] [--out DIR] [--local] [--n-envs N]
        [--eval-episodes 100] [--seed 0] [--poll-s 30] [--max-wait-s 3600] [--build-snapshot]
        [--snapshot arm-rl-v1] [--include-run DIR ...] [--no-copy] [--policy-dest FILE] [--curves FILE]
    python -m rl.train_parallel --verify-snapshot [--build-snapshot] [--snapshot arm-rl-v1]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "BEST_POLICY",
    "BUNDLE_FILES",
    "BUNDLE_GLOBS",
    "CURVES_PATH",
    "DEFAULT_OUT",
    "DEFAULT_REWARDS",
    "DEFAULT_STEPS",
    "INCLUDED_FILES",
    "LOCAL_ENV_KEYS",
    "LOCAL_N_ENVS",
    "MAX_SANDBOXES",
    "REMOTE_DIR",
    "RESULT_FILES",
    "RL_LABELS",
    "SNAPSHOT_CHECK_CMD",
    "SNAPSHOT_NAME",
    "build_rl_snapshot",
    "bundle_paths",
    "check_rl_snapshot",
    "clean_env",
    "hint_for",
    "included_runs",
    "make_bundle",
    "provenance_path",
    "read_progress",
    "read_summary",
    "select_best",
    "snapshot_check_dir",
    "train_parallel",
    "variant_names",
    "write_curves",
    "write_provenance",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT_NAME = "arm-rl-v1"
MAX_SANDBOXES = 4  # hard cap on the RL sandboxes (CLAUDE.md hard rule 8)
RL_LABELS = ("rl", "rl-check")  # `vara` label values of this module's sandboxes: training, snapshot check
GONE_STATES = ("destroyed", "destroying")  # sandbox states that no longer count against MAX_SANDBOXES
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
INCLUDED_FILES = ("policy.zip", "summary.json")  # what an --include-run directory must hold
SNAPSHOT_CHECK_CMD = (
    'python -c "import torch, stable_baselines3, mujoco; '
    'print(torch.__version__, stable_baselines3.__version__, mujoco.__version__)"'
)
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


def _rl_sandboxes_on_account(client: Any) -> int | None:
    """How many RL sandboxes the whole account holds now (the `vara` label in RL_LABELS, any state but
    GONE_STATES), or None when the listing fails: the failure is logged and the cap then holds per process
    only, as in agents/sandbox_pool.py. Daytona 0.225: `Daytona.list()` yields every Sandbox, whose
    `labels` is a dict and whose `state` is a SandboxState enum."""
    try:
        count = 0
        for sandbox in client.list():
            labels = getattr(sandbox, "labels", None)
            value = labels.get("vara") if isinstance(labels, Mapping) else None
            state = getattr(sandbox, "state", None)
            state = str(getattr(state, "value", state) or "").lower()
            if value in RL_LABELS and state not in GONE_STATES:
                count += 1
        return count
    except Exception as exc:  # noqa: BLE001 - the SDK raises many types; a fake may have no list()
        log.warning(
            "could not list the account's sandboxes (%s: %s); the RL cap holds per process only",
            type(exc).__name__,
            exc,
        )
        return None


def _reserve_rl_sandboxes(client: Any, wanted: int) -> None:
    """Refuse (RuntimeError) when `wanted` more RL sandboxes would take the account past MAX_SANDBOXES."""
    on_account = _rl_sandboxes_on_account(client)
    if on_account is not None and on_account + wanted > MAX_SANDBOXES:
        raise RuntimeError(
            f"refusing to create {wanted} RL sandbox(es): the Daytona account already holds {on_account} "
            f"(vara label rl or rl-check) and the cap is {MAX_SANDBOXES}; wait for the other training or "
            "snapshot check to finish, or delete the leftovers from the Daytona dashboard"
        )


def snapshot_check_dir() -> pathlib.Path:
    """Where snapshot-check records go: `<VARA_RUNS_DIR or runs>/rl` (gitignored evidence)."""
    base = os.environ.get("VARA_RUNS_DIR")
    return (pathlib.Path(base) if base else ROOT / "runs") / "rl"


def check_rl_snapshot(
    name: str = SNAPSHOT_NAME,
    *,
    report: Callable[[str], None] | None = None,
    out_dir: str | pathlib.Path | None = None,
    build: bool = False,
) -> dict[str, Any]:
    """Check the RL snapshot (D3's Done-when): a sandbox from `name` runs SNAPSHOT_CHECK_CMD, then is deleted.

    The sandbox is created the way the trainer creates its four (`CreateSandboxFromSnapshotParams` with
    `auto_stop_interval=0`, the snapshot's own resources, label `vara: rl-check`), the import line runs with
    EXEC_TIMEOUT_S, and the sandbox is deleted in `finally`: on success, on any exception and on
    KeyboardInterrupt (SIGTERM is turned into one while it exists). `build=True` (the CLI's
    `--build-snapshot`) first builds the snapshot with `build_rl_snapshot(name)`; a failed build is the
    recorded error and no sandbox is created. The record `{"snapshot", "command", "built", "sandbox_id",
    "exit_code", "output", "error", "deleted", "ok", "elapsed_s", "created", "record"}` (`built`: None when
    no build was asked for, else whether it succeeded) is written to `out_dir/snapshot-check-<UTC
    stamp>.json` (`out_dir` defaults to `snapshot_check_dir()`; a new file every call) and returned. `ok`
    is True only when the import exited 0 and the sandbox was deleted. A failing build, client, create or
    exec is recorded in `error` (`exit_code` stays None) and a failing delete too (`deleted` stays False,
    `error` "delete failed: ..." unless an earlier failure is there), not raised, so the record always
    exists; KeyboardInterrupt propagates after the delete. The one sandbox counts against MAX_SANDBOXES:
    when the account already holds MAX_SANDBOXES RL sandboxes (a training in flight) nothing is created and
    the refusal is the recorded error. Needs DAYTONA_API_KEY.
    """
    from daytona import CreateSandboxFromSnapshotParams

    say = report if report is not None else (lambda line: log.info("%s", line))
    record: dict[str, Any] = {
        "snapshot": str(name),
        "command": SNAPSHOT_CHECK_CMD,
        "built": None,
        "sandbox_id": None,
        "exit_code": None,
        "output": "",
        "error": None,
        "deleted": False,
        "ok": False,
    }
    sandbox: Any = None
    previous = _install_sigterm()
    started = time.monotonic()
    try:
        if build:
            record["built"] = False
            say(f"snapshot check: building {name} first")
            build_rl_snapshot(name)
            record["built"] = True
        daytona = _daytona_factory()
        _reserve_rl_sandboxes(daytona, 1)
        params = CreateSandboxFromSnapshotParams(
            snapshot=name, auto_stop_interval=0, labels={"vara": "rl-check"}
        )
        sandbox = daytona.create(params, timeout=CREATE_TIMEOUT_S)
        record["sandbox_id"] = str(getattr(sandbox, "id", "?"))
        say(f"snapshot check: sandbox {record['sandbox_id']} created from {name}")
        response = sandbox.process.exec(SNAPSHOT_CHECK_CMD, timeout=EXEC_TIMEOUT_S)
        code = getattr(response, "exit_code", None)
        record["exit_code"] = int(code) if code is not None else None
        record["output"] = _result_text(response).strip()
        say(f"snapshot check: exit {record['exit_code']}: {record['output'] or '(no output)'}")
    except Exception as exc:  # recorded, not raised: the record is the point of the check
        record["error"] = f"{type(exc).__name__}: {exc}"
        say(f"snapshot check: FAILED ({record['error']})")
    finally:
        _restore_sigterm(previous)
        if sandbox is not None:
            try:
                sandbox.delete()
                record["deleted"] = True
                say(f"snapshot check: sandbox {record['sandbox_id']} deleted")
            except Exception as exc:  # the leak is reported and recorded, not hidden
                log.error("snapshot check: sandbox %s delete failed: %s", record["sandbox_id"], exc)
                say(f"snapshot check: sandbox delete FAILED ({type(exc).__name__}: {exc})")
                if record["error"] is None:
                    record["error"] = f"delete failed: {type(exc).__name__}: {exc}"
        record["elapsed_s"] = round(time.monotonic() - started, 3)
        record["created"] = _now_iso()
        record["ok"] = record["error"] is None and record["exit_code"] == 0 and record["deleted"]
        target = pathlib.Path(out_dir) if out_dir is not None else snapshot_check_dir()
        path = _fresh_path(target, "snapshot-check")
        record["record"] = str(path)
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        say(f"snapshot check: record {path}")
    return record


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


def _fresh_path(directory: pathlib.Path, stem: str, suffix: str = ".json") -> pathlib.Path:
    """`directory/<stem>-<UTC stamp>[-n]<suffix>`: a path that does not exist yet (directory created)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stem}-{stamp}{suffix}"
    n = 2
    while path.exists():
        path = directory / f"{stem}-{stamp}-{n}{suffix}"
        n += 1
    return path


def _sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: str | pathlib.Path) -> str:
    """A policy path as rl/policies/handoff_best.json records it: repo-relative POSIX when inside the repo."""
    resolved = pathlib.Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repo_path(text: Any) -> str:
    """A path a summary.json recorded (absolute, or relative to the repo root where rl.train runs; either
    separator) as `_repo_relative()` gives it, "" for none: `rl\\rewards\\baseline.py` becomes
    `rl/rewards/baseline.py`."""
    if not text:
        return ""
    path = pathlib.Path(str(text).replace("\\", "/"))
    return _repo_relative(path if path.is_absolute() else ROOT / path)


def _existing_hint(summary: Mapping[str, Any]) -> str:
    """The hint of an included run: its reward's hint (`hint_for`) and its steps when the summary has them."""
    details = []
    reward = summary.get("reward")
    if reward:
        details.append(hint_for(pathlib.Path(str(reward))))
    steps = _int_or_none(summary.get("steps"))
    if steps is not None:
        details.append(f"{steps:,} steps")
    return "existing run (not retrained)" + (f": {', '.join(details)}" if details else "")


def included_runs(
    include_runs: Mapping[str, str | pathlib.Path] | Sequence[str | pathlib.Path] | None,
    taken: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """The run dicts of already trained runs that join a comparison without being retrained.

    `include_runs` is a mapping name -> run directory, or a sequence of run directories named after their
    last path component (`rl/runs/baseline` -> "baseline"). Names are sanitized like `variant_names()` and
    must be non-empty, unique and absent from `taken` (the trained variant names): ValueError otherwise.
    Each directory needs INCLUDED_FILES (`policy.zip` and `summary.json`, FileNotFoundError when one is
    missing) and a readable summary (ValueError); `progress.csv` is optional and gives the curve. Nothing
    here touches a sandbox or a subprocess, so the validation runs before any training starts.
    """
    if not include_runs:
        return []
    if isinstance(include_runs, Mapping):
        items = [(str(key), pathlib.Path(value)) for key, value in include_runs.items()]
    else:
        items = [(pathlib.Path(entry).name, pathlib.Path(entry)) for entry in include_runs]
    names = list(taken)
    runs: list[dict[str, Any]] = []
    for raw, directory in items:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)
        if not name:
            raise ValueError(f"included run {directory} needs a name")
        if name in names:
            raise ValueError(f"included run name {name!r} collides with another variant")
        if not directory.is_dir():
            raise FileNotFoundError(f"included run directory not found: {directory}")
        for artifact in INCLUDED_FILES:
            if not (directory / artifact).is_file():
                raise FileNotFoundError(f"included run {directory} has no {artifact}")
        summary = read_summary(directory / "summary.json")
        if summary is None:
            raise ValueError(f"included run {directory} has an unreadable summary.json")
        names.append(name)
        runs.append(
            {
                "name": name,
                "hint": _existing_hint(summary),
                "reward": _repo_path(summary.get("reward")),
                "runner": "existing",
                "out_dir": str(directory),
                "included": True,
                "error": None,
            }
        )
    return runs


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
    _reserve_rl_sandboxes(daytona, len(runs))  # before the SIGTERM handler and before any create
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
        error = f"no readable summary.json in {out}"
    if error is None and not policy.is_file():
        error = f"no policy.zip in {out}"
    if error is None and rate is not None and not 0.0 <= rate <= 1.0:
        error = "invalid evaluation success_rate: expected a value from 0 to 1"
    return {
        **run,
        "summary": summary,
        "progress": progress,
        "policy": str(policy) if policy.is_file() else None,
        "policy_sha256": _sha256(policy) if policy.is_file() else None,
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


def provenance_path(policy_dest: str | pathlib.Path) -> pathlib.Path:
    """The provenance file next to a deployed policy: `handoff_best.zip` -> `handoff_best.json`."""
    return pathlib.Path(policy_dest).with_suffix(".json")


def write_provenance(
    variant: Mapping[str, Any],
    policy_dest: str | pathlib.Path,
    *,
    candidates: Sequence[str],
    result_path: str | pathlib.Path,
    created: str,
) -> tuple[pathlib.Path, bool]:
    """Keep `provenance_path(policy_dest)` true to the chosen `variant`, whose policy was just copied to
    `policy_dest`; returns the path and whether it was rewritten.

    A file that already records the same sha256 describes this very policy and is left untouched, so the
    curated independent evaluation in rl/policies/handoff_best.json survives a run that selects the same
    policy again. Any other file, or none, is (re)written with the fields scripts/story_numbers.py reads:
    `policy` (the copy's name), `sha256`, `source_policy`, `reward`, `requested_steps`, `steps`,
    `training_seed`, `selection_reason`, `evaluation` (the training run's own evaluation from its
    summary.json, with `evaluated_on` the selection date) and `deployment` (a new policy is simulation-only
    until its own M4 run), plus `selected_by`, `result` (the selection's result.json), `created` and
    `replaces` (the previous file's sha256 and source_policy, when there was one).
    """
    path = provenance_path(policy_dest)
    sha = variant.get("policy_sha256")
    if not sha or not variant.get("policy"):
        raise ValueError(f"variant {variant.get('name')!r} has no policy to describe")
    previous: Any = None
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict) and previous.get("sha256") == sha:
            return path, False
    summary = variant.get("summary") or {}
    data: dict[str, Any] = {
        "policy": pathlib.Path(policy_dest).name,
        "sha256": sha,
        "source_policy": _repo_relative(variant["policy"]),
        "reward": _repo_path(variant.get("reward") or summary.get("reward")),
        "requested_steps": _int_or_none(summary.get("requested_steps")),
        "steps": _int_or_none(summary.get("steps")),
        "training_seed": _int_or_none(summary.get("seed")),
        "selection_reason": (
            f"rl.train_parallel chose {variant.get('name')} from {', '.join(candidates)}: the highest "
            "evaluation success rate, then the fewest mean steps to success"
        ),
        "selected_by": "rl.train_parallel",
        "result": _repo_relative(result_path),
        "evaluation": {
            "evaluated_on": created[:10],
            "source": "the training run's summary.json (rl.train's evaluation)",
            "seed": _int_or_none(summary.get("eval_seed")),
            "episodes": _int_or_none(summary.get("episodes")),
            "successes": _int_or_none(summary.get("successes")),
            "success_rate": _float_or_none(variant.get("success_rate")),
            "mean_steps_to_success": _float_or_none(variant.get("mean_steps_to_success")),
            "falls": _int_or_none(summary.get("falls")),
            "mean_final_dist": _float_or_none(summary.get("mean_final_dist")),
        },
        "deployment": (
            "simulation; a newly selected policy needs its own real-arm M4 validation before "
            "config/rl.yaml use_policy is set"
        ),
        "created": created,
    }
    if isinstance(previous, dict):
        data["replaces"] = {"sha256": previous.get("sha256"), "source_policy": previous.get("source_policy")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return path, True


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
    policy_dest: str | pathlib.Path | None = BEST_POLICY,
    curves_path: str | pathlib.Path = CURVES_PATH,
    hints: Sequence[str] | None = None,
    report: Callable[[str], None] | None = None,
    max_wait_s: float | None = MAX_WAIT_S,
    snapshot: str = SNAPSHOT_NAME,
    include_runs: Mapping[str, str | pathlib.Path] | Sequence[str | pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Train every reward (at most four), add the included runs, pick the best policy, write curves.json.

    See the module doc for the artifacts. `include_runs` (see `included_runs()`) joins already trained runs
    to the comparison without retraining them; with `include_runs` the reward list may be empty, and then
    nothing trains. `policy_dest=None` leaves the chosen policy where it is (the CLI's `--no-copy`) and the
    provenance file untouched; otherwise the copy's provenance is kept true (`write_provenance()`).
    Raises ValueError for more than MAX_SANDBOXES rewards, bad numbers, a bad included run, a `.json`
    `policy_dest` or an `out_dir` that already holds a result.json or variant artifacts (all before any
    sandbox or subprocess exists), FileNotFoundError for a missing reward, bundle or included-run file,
    RuntimeError when a sandbox step fails or the account's RL sandboxes would exceed MAX_SANDBOXES, and
    TimeoutError past `max_wait_s`; every sandbox created is deleted on the way out.
    """
    paths = [pathlib.Path(r) for r in rewards]
    if not paths and not include_runs:
        raise ValueError("train_parallel needs at least one reward file or an included run")
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
    if policy_dest is not None and pathlib.Path(policy_dest).suffix.lower() == ".json":
        raise ValueError(
            f"policy_dest must be the policy file; its .json provenance goes next to it: {policy_dest}"
        )
    out = pathlib.Path(out_dir)
    say = report if report is not None else (lambda line: log.info("%s", line))
    names = variant_names(paths)
    for name in names:
        if any((out / name / artifact).exists() for artifact in RESULT_FILES):
            raise ValueError(f"existing training artifacts in {out / name}; choose a fresh output directory")
    result_path = out / "result.json"
    if result_path.exists():  # one selection per directory: an earlier result is evidence, never overwritten
        raise ValueError(f"{result_path} already exists; choose a fresh output directory")
    extra = included_runs(include_runs, taken=names)  # validated before anything trains
    out.mkdir(parents=True, exist_ok=True)  # only once every check passed: a refusal leaves no directory
    runner = ("local" if local else "daytona") if paths else "existing"
    runs: list[dict[str, Any]] = [
        {
            "name": name,
            "hint": (hints[i] if hints is not None else hint_for(path)),
            "reward": str(path),
            "runner": runner,
            "out_dir": str(out / name),
            "included": False,
            "error": None,
        }
        for i, (name, path) in enumerate(zip(names, paths, strict=True))
    ]
    if not paths:
        say(f"nothing to train: comparing {len(extra)} included run(s) only")
    elif local:
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

    variants = [_collect(run) for run in [*runs, *extra]]
    candidates = [variant["name"] for variant in variants]
    best = select_best(variants)
    created = _now_iso()
    copied: str | None = None
    provenance: str | None = None
    if best is not None:
        chosen = variants[best]
        chosen["chosen"] = True
        if policy_dest is None:
            say(f"chosen: {chosen['name']} ({chosen['hint']}); policy not copied (no_copy)")
        else:
            dest = pathlib.Path(policy_dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chosen["policy"], dest)
            copied = str(dest)
            say(f"chosen: {chosen['name']} ({chosen['hint']}) -> {dest}")
            path, rewritten = write_provenance(
                chosen, dest, candidates=candidates, result_path=result_path, created=created
            )
            provenance = str(path)
            state = "rewritten" if rewritten else "unchanged (it already records this policy)"
            say(f"provenance: {path} {state}")
    else:
        say("no variant produced an evaluated policy; nothing chosen")
    write_curves(variants, curves_path, created)
    result = {
        "variants": variants,
        "candidates": candidates,
        "included": [variant["name"] for variant in variants if variant.get("included")],
        "chosen": variants[best]["name"] if best is not None else None,
        "source_policy": _repo_relative(variants[best]["policy"]) if best is not None else None,
        "policy_sha256": variants[best]["policy_sha256"] if best is not None else None,
        "policy": copied,
        "provenance": provenance,
        "curves": str(curves_path),
        "runner": runner,
        "steps": int(steps) if paths else None,
        "created": created,
    }
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return result


# --------------------------------------------------------------------------------------------- CLI


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _is_cli_error(exc: Exception) -> bool:
    """Expected failures the CLI reports as one `error:` line and exit 2 (a bug of ours stays a traceback).

    The Daytona SDK's own DaytonaError family subclasses none of the standard ones."""
    return isinstance(exc, (ValueError, RuntimeError, TimeoutError, OSError)) or type(
        exc
    ).__module__.startswith("daytona")


def _verify_snapshot_cli(args: argparse.Namespace, has_key: bool) -> int:
    """`--verify-snapshot`: build the snapshot when asked, check it, never train (exit 0 only on `ok`)."""
    if args.local:
        print("error: --verify-snapshot checks a Daytona snapshot and has no --local mode", file=sys.stderr)
        return 2
    training_only = [
        flag
        for flag, given in (
            ("--rewards", args.rewards is not None),
            ("--include-run", bool(args.include_run)),
            ("--no-copy", args.no_copy),
        )
        if given
    ]
    if training_only:
        dropped = ", ".join(training_only)
        print(
            f"error: --verify-snapshot only checks the snapshot and never trains; drop {dropped}",
            file=sys.stderr,
        )
        return 2
    if not has_key:
        print(
            "DAYTONA_API_KEY is not set: --verify-snapshot needs a Daytona key and has no local fallback",
            file=sys.stderr,
        )
        return 2
    try:  # a failed build, create, exec or delete is in the record (exit 1); this catches the record write
        record = check_rl_snapshot(args.snapshot, build=args.build_snapshot)
    except Exception as exc:
        if not _is_cli_error(exc):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    shown = [key for key in record if key not in ("command", "created")]  # the file keeps those two
    print(json.dumps({key: record[key] for key in shown}))
    return 0 if record["ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.train_parallel",
        description=(
            "Train one PPO policy per reward variant in Daytona sandboxes (or locally), compare it with "
            "already trained runs, and keep the best; or check the RL snapshot (--verify-snapshot)"
        ),
    )
    parser.add_argument(
        "--rewards",
        nargs="*",
        default=None,
        help=(
            "reward files (default rl/rewards/variant_*.py); given with no files, nothing trains and only "
            "the --include-run runs are compared"
        ),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="a fresh output directory (default rl/runs/parallel): one holding a result.json is refused",
    )
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
        "--build-snapshot",
        action="store_true",
        help="build the RL snapshot first (needs a key); with --verify-snapshot the build is in the record",
    )
    parser.add_argument("--snapshot", default=SNAPSHOT_NAME)
    parser.add_argument(
        "--verify-snapshot",
        "--check-snapshot",
        dest="verify_snapshot",
        action="store_true",
        help=(
            "check the snapshot instead of training: one sandbox runs the torch/stable_baselines3/mujoco "
            "import, the record goes to runs/rl/snapshot-check-<stamp>.json and the sandbox is deleted "
            "(needs DAYTONA_API_KEY; exit 0 only when the import passed and the sandbox is gone)"
        ),
    )
    parser.add_argument(
        "--include-run",
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "an already trained run directory (policy.zip, summary.json, progress.csv as rl.train writes "
            "them) that joins the comparison under its directory name without retraining; repeatable, "
            "e.g. --include-run rl/runs/baseline"
        ),
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help=(
            "compare without copying the chosen policy to --policy-dest (rl/policies/handoff_best.zip) "
            "and without touching its provenance (rl/policies/handoff_best.json)"
        ),
    )
    parser.add_argument(
        "--policy-dest",
        default=str(BEST_POLICY),
        help="where the chosen policy is copied (its .json provenance is kept next to it)",
    )
    parser.add_argument(
        "--curves", default=str(CURVES_PATH), help="the dashboard's curves file (web/static/rl/curves.json)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    has_key = bool(os.environ.get("DAYTONA_API_KEY"))  # only the boolean ever leaves this line
    if args.verify_snapshot:
        return _verify_snapshot_cli(args, has_key)

    if args.rewards is None:
        rewards = [p for p in DEFAULT_REWARDS if p.is_file()]
    else:
        rewards = [pathlib.Path(r) for r in args.rewards]
    include = [pathlib.Path(d) for d in (args.include_run or [])]
    if not rewards and not include:
        print(
            "error: no reward files; run `python -m rl.rewards.write_rewards` first (or pass --include-run)",
            file=sys.stderr,
        )
        return 2
    local = args.local
    if rewards and not local and not has_key:
        print(
            "DAYTONA_API_KEY is not set: training locally (--local) instead of in Daytona sandboxes",
            file=sys.stderr,
        )
        local = True
    try:
        if args.build_snapshot and rewards and not local:
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
            policy_dest=None if args.no_copy else args.policy_dest,
            curves_path=args.curves,
            snapshot=args.snapshot,
            max_wait_s=args.max_wait_s,
            include_runs=include,
        )
    except Exception as exc:
        if not _is_cli_error(exc):
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
    print(
        json.dumps(
            {
                "chosen": result["chosen"],
                "policy": result["policy"],
                "provenance": result.get("provenance"),
                "source_policy": result.get("source_policy"),
                "policy_sha256": result.get("policy_sha256"),
                "curves": result["curves"],
            }
        )
    )
    return 0 if result["chosen"] else 1


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
