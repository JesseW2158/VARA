"""I1 /verify as a script: pytest with nothing skipped, the fake e2e run and ruff, with a record per laptop.

    .venv/Scripts/python.exe scripts/verify.py                                     # Windows, about 7 minutes
    .venv/bin/python scripts/verify.py                                             # macOS and Linux
    .venv/Scripts/python.exe scripts/verify.py --skip-e2e --pytest-args "tests/test_setup_check.py"

The steps run one after the other with the repository's own interpreter (`.venv/Scripts/python.exe` or
`.venv/bin/python` next to this repo, whatever interpreter started the script; `sys.executable` with a
printed warning when neither exists), from the repo root, each echoed while it runs:

1. `pytest`: `-m pytest -q -p no:cacheprovider -rs [PYTEST_ARGS]`. It passes only when the process exits 0
   and the final summary line counts nothing failed, errored, skipped, xfailed or xpassed (S0's "nothing
   skipped"; `check_pytest`; colour codes are ignored).
2. `e2e`: `scripts/e2e_fake.py "put the juice box in the tray"`. It passes only when the process exits 0
   and the last stage line it printed is `saved` (`check_e2e`). The script forces FAKE=1 itself and learns
   into a copy of skills/, so no model, sandbox, arm or camera is touched.
3. `ruff`: `-m ruff check .`; passes when the process exits 0.

`--skip-e2e` leaves the second step out, `--only pytest|e2e|ruff` (repeatable) keeps only the named ones
and `--pytest-args "..."` appends to the pytest command (one quoted string, forward slashes in paths).
The record `runs/verify/<hostname>-<UTC stamp>.json` (under `VARA_RUNS_DIR` when set; a new file every
run, never overwritten) holds the host identity (hostname, OS, the Python version that ran this script,
git HEAD and dirty flag, UTC time), the interpreter the steps used, the arguments and, per step, the
command, exit code, seconds, pass or fail with the reason, the last lines of output and the path of its
full log in `runs/verify/<hostname>-<UTC stamp>/<step>.log` (`log_dir`; the console's last two lines are
`verify logs: <folder>` and `verify record: <file>`). A step killed at `--step-timeout` or one that cannot
be started fails with that reason and the record is still written. `complete` is true only when every step
ran with its default command, so a fast rerun is never mistaken for an I1 pass. Exit 0 when every step that
ran passed, 1 otherwise, 2 on a bad call. Only the standard library is used, `.env` is never read and no
key is printed or recorded. `verify(runner=...)` takes a replacement step runner so tests feed canned
output instead of running anything.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STEP_NAMES: tuple[str, ...] = ("pytest", "e2e", "ruff")
E2E_TEXT = "put the juice box in the tray"
PYTEST_ARGS: tuple[str, ...] = ("-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs")
NOT_CLEAN: tuple[str, ...] = ("failed", "error", "errors", "skipped", "xfailed", "xpassed")
TAIL_LINES = 40
TAIL_CHARS = 6000
DEFAULT_STEP_TIMEOUT_S = 1800.0

# pytest's final line: "3 passed, 1 skipped in 0.72s", "no tests ran in 0.01s", or wrapped in "=" without -q;
# "(0:05:00)" follows the seconds on long runs.
_SUMMARY_RE = re.compile(
    r"^(?:=+ )?(?P<body>no tests ran|\d+ [a-z]+(?:, \d+ [a-z]+)*) in [\d.]+s(?: \([^)]*\))?(?: =+)?$"
)
_COUNT_RE = re.compile(r"(\d+) ([a-z]+)")
# pytest colours its summary line even through a pipe when PY_COLORS=1 or FORCE_COLOR is set.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# scripts/e2e_fake.py prints `f"{stage:<19} {summary}"`: a stage name then at least two spaces.
_STAGE_RE = re.compile(r"^([a-z_]+)\s{2,}")

Runner = Callable[[Sequence[str], Path, Mapping[str, str], float, Path], tuple[int | None, str]]


# --- the checks --------------------------------------------------------------------------------------


def pytest_counts(output: str) -> dict[str, int] | None:
    """The counts on pytest's final summary line, such as `{"passed": 3, "skipped": 1}`; `{}` for "no tests
    ran"; None when no summary line is found. Colour codes are ignored."""
    for line in reversed(output.splitlines()):
        match = _SUMMARY_RE.match(_ANSI_RE.sub("", line).strip())
        if match:
            return {word: int(n) for n, word in _COUNT_RE.findall(match.group("body"))}
    return None


def check_pytest(exit_code: int | None, output: str) -> tuple[bool, str]:
    """Pass only when pytest exited 0 and its summary counts nothing failed, errored, skipped, xfailed or
    xpassed; the reason names what was counted."""
    counts = pytest_counts(output)
    if counts is None:
        return False, f"no pytest summary line found (exit {exit_code})"
    bad = [f"{counts[word]} {word}" for word in NOT_CLEAN if counts.get(word)]
    if bad:
        return False, "summary counts " + ", ".join(bad)
    if not counts.get("passed"):
        return False, f"no tests passed (exit {exit_code})"
    if exit_code != 0:
        return False, f"exit {exit_code} although the summary is clean"
    return True, f"{counts['passed']} passed, nothing skipped"


def e2e_stages(output: str) -> list[str]:
    """The stage names scripts/e2e_fake.py printed (its `stage  summary` lines), in order."""
    stages = []
    for line in output.splitlines():
        match = _STAGE_RE.match(line)
        if match:
            stages.append(match.group(1))
    return stages


def check_e2e(exit_code: int | None, output: str) -> tuple[bool, str]:
    """Pass only when the fake end-to-end run exited 0 and the last stage it printed is `saved`."""
    stages = e2e_stages(output)
    if not stages:
        return False, f"no stage lines printed (exit {exit_code})"
    if stages[-1] != "saved":
        return False, f"last stage {stages[-1]!r}, expected 'saved'"
    if exit_code != 0:
        return False, f"exit {exit_code} although the last stage is saved"
    return True, "last stage saved"


def check_exit(exit_code: int | None, output: str) -> tuple[bool, str]:
    """Pass only on exit code 0."""
    return exit_code == 0, f"exit {exit_code}"


# --- the environment ---------------------------------------------------------------------------------


def find_interpreter(root: Path = ROOT) -> tuple[str, bool]:
    """`root/.venv/Scripts/python.exe` or `root/.venv/bin/python` (whichever exists) and True; else
    `sys.executable` and False."""
    for relative in ("Scripts/python.exe", "bin/python"):
        candidate = root / ".venv" / relative
        if candidate.is_file():
            return str(candidate), True
    return sys.executable, False


def interpreter_version(interpreter: str) -> str | None:
    """The `X.Y.Z` version `interpreter` reports, or None when it cannot be run."""
    try:
        done = subprocess.run(
            [interpreter, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def git_output(root: Path, *args: str) -> str | None:
    """stdout of `git <args>` run in `root`, stripped; None when git is missing, fails or times out."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def host_info(root: Path = ROOT) -> dict[str, Any]:
    """The laptop's identity for the record: hostname, OS, the running Python's version, git HEAD, whether
    the working tree is dirty (None when git cannot say) and the UTC time. Reads platform/sys and runs git
    in `root`; nothing else."""
    status = git_output(root, "status", "--porcelain")
    return {
        "hostname": platform.node() or "unknown",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "git_head": git_output(root, "rev-parse", "--short", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --- the steps ---------------------------------------------------------------------------------------


def plan_steps(
    interpreter: str,
    *,
    pytest_args: str = "",
    skip_e2e: bool = False,
    only: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """The three steps in order: `name`, `command`, `check`, and `skipped` with `why` for the ones left out.

    Raises ValueError for an unknown step name in `only` or unbalanced quotes in `pytest_args`."""
    unknown = [name for name in (only or ()) if name not in STEP_NAMES]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; choose from {list(STEP_NAMES)}")
    extra = shlex.split(pytest_args) if pytest_args else []
    steps: list[dict[str, Any]] = [
        {"name": "pytest", "command": [interpreter, *PYTEST_ARGS, *extra], "check": check_pytest},
        {"name": "e2e", "command": [interpreter, "scripts/e2e_fake.py", E2E_TEXT], "check": check_e2e},
        {"name": "ruff", "command": [interpreter, "-m", "ruff", "check", "."], "check": check_exit},
    ]
    for step in steps:
        why = None
        if only and step["name"] not in only:
            why = "not in --only"
        if skip_e2e and step["name"] == "e2e":
            why = "--skip-e2e"
        step["skipped"] = why is not None
        step["why"] = why
    return steps


def run_command(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    log_path: Path,
    *,
    echo: bool = True,
) -> tuple[int | None, str]:
    """Run `command` in `cwd`, streaming its merged stdout and stderr to `log_path` (and to stdout when
    `echo`). Returns (exit code, or None when it was killed after `timeout_s`; the whole output, LF-only)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    proc = subprocess.Popen(
        list(command), cwd=str(cwd), env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    def pump() -> None:
        assert proc.stdout is not None
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").replace("\r\n", "\n")
                chunks.append(line)
                log.write(line)
                log.flush()
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()

    reader = threading.Thread(target=pump, name="verify-output", daemon=True)
    reader.start()
    code: int | None
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code = None
    reader.join(timeout=10.0)
    return code, "".join(chunks)


def tail(output: str, lines: int = TAIL_LINES, chars: int = TAIL_CHARS) -> str:
    """The last `lines` lines of `output`, cut to its last `chars` characters."""
    return "\n".join(output.splitlines()[-lines:])[-chars:]


# --- the record --------------------------------------------------------------------------------------


def runs_dir_for(root: Path = ROOT) -> Path:
    """`$VARA_RUNS_DIR/verify` when the variable is set, else `root/runs/verify`."""
    base = os.environ.get("VARA_RUNS_DIR")
    return (Path(base) if base else root / "runs") / "verify"


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or "host"


def new_record_paths(runs_dir: Path, hostname: str, stamp: str) -> tuple[Path, Path]:
    """`<runs_dir>/<hostname>-<stamp>.json` and the log folder with the same base; a base that exists in
    either form gets `-2`, `-3`, ... so a record is never overwritten."""
    base = f"{_safe_name(hostname)}-{stamp}"
    candidate, n = base, 1
    while (runs_dir / f"{candidate}.json").exists() or (runs_dir / candidate).exists():
        n += 1
        candidate = f"{base}-{n}"
    return runs_dir / f"{candidate}.json", runs_dir / candidate


def verify(
    *,
    root: Path = ROOT,
    runs_dir: Path | None = None,
    skip_e2e: bool = False,
    pytest_args: str = "",
    only: Sequence[str] | None = None,
    step_timeout: float = DEFAULT_STEP_TIMEOUT_S,
    runner: Runner | None = None,
    echo: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Run the selected steps from `root`, write the record and return `(record, its path)`.

    `runner(command, cwd, env, timeout_s, log_path) -> (exit code or None, output)` replaces the subprocess
    runner; the default streams each step's output to its log (and to stdout when `echo`). Raises
    ValueError for bad `only`/`pytest_args` and OSError when the record cannot be written."""
    runs_dir = Path(runs_dir) if runs_dir is not None else runs_dir_for(root)
    interpreter, is_venv = find_interpreter(root)
    steps = plan_steps(interpreter, pytest_args=pytest_args, skip_e2e=skip_e2e, only=only)
    host = host_info(root)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    record_path, log_dir = new_record_paths(runs_dir, host["hostname"], stamp)
    run = runner if runner is not None else functools.partial(run_command, echo=echo)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")  # the hints hold a pi; children must not crash on it
    warning = "" if is_venv else " (WARNING: no .venv under the repo; this is not the project interpreter)"
    print(f"verify: interpreter {interpreter}{warning}", flush=True)
    tree = {None: "git state unknown", True: "dirty", False: "clean"}[host["git_dirty"]]
    print(
        f"verify: host {host['hostname']}, {host['os']}, Python {host['python']}, "
        f"git {host['git_head'] or '?'} ({tree})",
        flush=True,
    )
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for k, step in enumerate(steps, 1):
        label = f"[{k}/{len(steps)}] {step['name']}"
        result: dict[str, Any] = {
            "name": step["name"],
            "command": list(step["command"]),
            "skipped": step["skipped"],
            "why": step["why"],
            "exit_code": None,
            "seconds": 0.0,
            "passed": None,
            "reason": step["why"],
            "log": None,
            "output_tail": "",
        }
        if step["skipped"]:
            print(f"{label}: skipped ({step['why']})", flush=True)
            results.append(result)
            continue
        print(f"{label}: {shlex.join(step['command'])}", flush=True)
        log_path = log_dir / f"{step['name']}.log"
        t_step = time.monotonic()
        started_ok = True
        try:
            code, output = run(step["command"], root, env, step_timeout, log_path)
        except OSError as exc:  # the interpreter cannot be started or the log cannot be written
            started_ok, code, output = False, None, f"could not run the step: {type(exc).__name__}: {exc}\n"
            print(output, end="", flush=True)
        seconds = time.monotonic() - t_step
        if not started_ok:
            passed, reason = False, output.strip()
        elif code is None:
            passed, reason = False, f"timed out after {step_timeout:g} s and was killed"
        else:
            passed, reason = step["check"](code, output)
        print(f"{label}: {'PASS' if passed else 'FAIL'} in {seconds:.1f} s ({reason})", flush=True)
        result.update(
            exit_code=code,
            seconds=round(seconds, 2),
            passed=passed,
            reason=reason,
            log=str(log_path) if log_path.is_file() else None,
            output_tail=tail(output),
        )
        results.append(result)
    ran = [r for r in results if not r["skipped"]]
    passed_all = bool(ran) and all(r["passed"] for r in ran)
    complete = not skip_e2e and not pytest_args and not only
    record = {
        "kind": "vara.verify",
        "host": host,
        "interpreter": {"path": interpreter, "is_venv": is_venv, "python": interpreter_version(interpreter)},
        "root": str(root),
        "log_dir": str(log_dir),
        "started_at": host["utc"],
        "seconds": round(time.monotonic() - started, 2),
        "args": {
            "skip_e2e": bool(skip_e2e),
            "pytest_args": pytest_args,
            "only": list(only) if only else None,
            "step_timeout_s": float(step_timeout),
        },
        "complete": complete,
        "steps": results,
        "passed": passed_all,
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    n_passed = sum(1 for r in ran if r["passed"])
    note = "" if complete else "; not a full run"
    print(
        f"verify: {'PASS' if passed_all else 'FAIL'}: {n_passed} of {len(ran)} steps passed, "
        f"{len(results) - len(ran)} skipped{note}",
        flush=True,
    )
    return record, record_path


# --- CLI ---------------------------------------------------------------------------------------------


def _utf8_stdout() -> None:
    """Echoed output holds a pi (the hints); a cp1252 console must not crash the run."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """`scripts/verify.py [--skip-e2e] [--pytest-args "..."] [--only STEP ...]`: run the I1 checks with the
    venv interpreter and write the record; exit 0 only when every step that ran passed."""
    parser = argparse.ArgumentParser(
        prog="scripts/verify.py",
        description="Run the I1 checks (pytest with nothing skipped, the fake end-to-end run, ruff) with "
        "the project's .venv interpreter and record the result under runs/verify/. "
        "Takes about seven minutes in full.",
    )
    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="leave out the fake end-to-end step (the record is then not a full I1 pass)",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        metavar="ARGS",
        help="extra pytest arguments as one quoted string, forward slashes in paths, for example "
        '"tests/test_setup_check.py" (the record is then not a full I1 pass)',
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=STEP_NAMES,
        metavar="STEP",
        help="run only this step (pytest, e2e or ruff; repeatable)",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=DEFAULT_STEP_TIMEOUT_S,
        metavar="S",
        help="seconds before a step is killed and failed (default 1800)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not echo the steps' output (the logs still get it all)"
    )
    args = parser.parse_args(argv)
    if not (math.isfinite(args.step_timeout) and args.step_timeout > 0):
        parser.error("--step-timeout must be a positive, finite number of seconds")
    try:
        shlex.split(args.pytest_args)
    except ValueError as exc:
        parser.error(f"--pytest-args: {exc}")
    _utf8_stdout()
    try:
        record, path = verify(
            skip_e2e=args.skip_e2e,
            pytest_args=args.pytest_args,
            only=args.only,
            step_timeout=args.step_timeout,
            echo=not args.quiet,
        )
    except OSError as exc:
        print(f"verify: cannot write the record: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"verify logs: {record['log_dir']}", flush=True)
    print(f"verify record: {path}", flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
