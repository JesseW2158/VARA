"""Live gates: run the key-dependent checks (C1, C2, C3, C5, C6, C8 and the A14 voice token) in order, keep
the evidence under runs/live-gates/<UTC stamp>/ and prove that no sandbox is left behind.

    python scripts/live_gates.py [--dry-run] [--only STEP[,STEP]] [--skip STEP[,STEP]] [--skip-daytona]
                                 [--skip-xai] [--continue] [--timeout STEP=SECONDS ...] [--out DIR]

Steps, in this order. Each is a child process of the interpreter running this script (run it with
.venv/Scripts/python.exe); its stdout and stderr go to `<out>/<step>.stdout.log` and `<step>.stderr.log`, the
parsed verdict to `<out>/<step>.json`, and everything to `<out>/summary.json` (rewritten after every step):

  ping                   python -m agents.llm --ping
                         C1: both models answer, each under 15 s. 2 paid model calls.
  router                 python -m agents.router --eval
                         C2: every fixture phrase (20) routed as expected by the live model. 20 to 40 calls.
  candidates             python -m agents.candidates --eval "put the juice box in the tray"
                                 --task-dir <out>/candidates --skills none
                         C3: at least 5 of 6 candidates pass check_source on the first try. 7 to 13 calls.
                         `--skills none` keeps the phrase a learn although skills/place_in_tray.json
                         exists; it is added when `agents.candidates --help` lists the flag (else the step
                         may fail with "not a learn request").
  inventory-before       python -m agents.sandbox_pool --list --label pool --wait 60 --out <out>/sandboxes
                         0 vara=pool sandboxes before any Daytona step (stop the dashboard first). Never
                         cleans up: what is there may belong to a running dashboard.
  smoke                  python scripts/daytona_smoke.py --out <out>/daytona-smoke --max-seconds 60
                         C5: a schema-valid score line from runner "daytona", total (create + score + delete)
                         within 60 s, the sandbox deleted. 1 sandbox.
  inventory-after-smoke  as inventory-before. On failure, and only after a clean inventory-before, `--cleanup
                         --label pool` runs at once; the step still fails.
  bench                  python -m agents.sandbox_pool --bench --out <out>/bench
                         C6: a SandboxPool of 7 sandboxes (the cap); three tasks of 6 candidates x 5 trials,
                         each within 35 s and scored on the sandboxes (no local fallback). 7 sandboxes.
  inventory-after-bench  as inventory-after-smoke.
  bench-run              python -m agents.orchestrator --bench-run "hand me my pills" --runs 5 --live-router
                         C8: all 5 runs reach `executing`, median within 5 s, live xAI router; fake arm,
                         cameras and sandboxes. The child gets VARA_RUNS_DIR=<out>/bench-run, so the
                         bench's own copy of the library, its t-* task folders and its summary.json land in
                         <out>/bench-run/bench/run-<stamp>/ (the saved skills are read, never written).
                         5 paid router calls.
  voice                  python scripts/live_gates.py --voice-probe
                         A14: a child with FAKE_ROBOT, FAKE_CAMERA and FAKE_SANDBOX = 1 and the live model
                         starts the orchestrator app in-process (TestClient), POSTs /voice/token and prints
                         only {status, model, expires_at, token_length, fake_llm, fake_token, detail}:
                         200 with a non-empty token from xAI passes. 1 token request. The token's value is
                         never printed, kept or written.

Selecting `smoke` or `bench` always brings inventory-before and the inventory after it (skipping one of those
is refused), so every selection that creates sandboxes proves the account is back to 0. Sandbox counts:
smoke 1, bench 7, never together: the steps run strictly one after another and the dashboard (7 sandboxes
while DAYTONA_API_KEY is set) must be stopped first. Only agents/sandbox_pool.py and scripts/daytona_smoke.py
create sandboxes; this script only starts them as children, and it never touches the arm or a camera (no
--real anywhere; the orchestrator children force their fakes on).

It refuses (exit 2, nothing run or written) when FAKE, FAKE_LLM or FAKE_SANDBOX is "1" or when XAI_API_KEY or
DAYTONA_API_KEY is missing (the gates need both; only their presence is ever reported, as booleans), and when
an explicit --out is not a new or empty folder. Each child gets only the key it uses (the other is set to an
empty value, which its own load_dotenv keeps), and every captured line is scrubbed of both key values before
it is written. Without --continue the run stops at the first failed step, but the inventory that follows a
Daytona step which ran always runs, also after a timeout or Ctrl+C (both kill the running child, which then
cannot delete its own sandboxes: that inventory finds them and its cleanup deletes them). Smoke and bench
never start after an inventory-before that found vara=pool sandboxes, even with --continue. Exit 0 when every
selected step passed, 1 otherwise; --dry-run prints the plan, the keys' presence and the refusal a real run
would give (exit 2 then, 0 when ready) and writes nothing. Tests replace `subprocess.run`.

Public helpers (tested offline): `build_plan(out_dir)`, `select_steps(only, skip)`, `refusal(out_dir)`,
`run_step(step, out_dir)`, `run_gates(plan, out_dir)`, the `parse_<step>` verdict functions, `scrub(text)`
and `child_env(needs)`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402

PYTHON = sys.executable
SCRIPT = pathlib.Path(__file__).resolve()
SMOKE_SCRIPT = ROOT / "scripts" / "daytona_smoke.py"
ROUTER_PHRASES = ROOT / "tests" / "fixtures" / "router_phrases.json"
SCORE_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "score.json").read_text(encoding="utf-8"))
)
XAI = "XAI_API_KEY"
DAYTONA = "DAYTONA_API_KEY"
SECRET_NAMES = (XAI, DAYTONA)
FAKE_FLAGS = ("FAKE", "FAKE_LLM", "FAKE_SANDBOX")
STEP_NAMES = (
    "ping",
    "router",
    "candidates",
    "inventory-before",
    "smoke",
    "inventory-after-smoke",
    "bench",
    "inventory-after-bench",
    "bench-run",
    "voice",
)
DAYTONA_STEPS = ("inventory-before", "smoke", "inventory-after-smoke", "bench", "inventory-after-bench")
XAI_STEPS = ("ping", "router", "candidates", "bench-run", "voice")
COMPANIONS = {  # the inventories that prove a sandbox-creating step left nothing behind
    "smoke": ("inventory-before", "inventory-after-smoke"),
    "bench": ("inventory-before", "inventory-after-bench"),
}
DEFAULT_TIMEOUTS: dict[str, float] = {
    "ping": 120.0,
    "router": 900.0,
    "candidates": 900.0,
    "inventory-before": 300.0,
    "smoke": 600.0,
    "inventory-after-smoke": 300.0,
    "bench": 1800.0,
    "inventory-after-bench": 300.0,
    "bench-run": 900.0,
    "voice": 300.0,
}
PING_LIMIT_MS = 15000.0
SMOKE_LIMIT_S = 60.0
BENCH_TASK_LIMIT_S = 35.0
BENCH_TASKS = 3
BENCH_RUN_TEXT = "hand me my pills"
BENCH_RUN_RUNS = 5
BENCH_RUN_TARGET_S = 5.0
INVENTORY_WAIT_S = 60
CANDIDATES_TEXT = "put the juice box in the tray"
SANDBOX_CAPS = {
    "smoke": 1,
    "bench": 7,
    "other_steps": 0,
    "rule": "never together: the Daytona steps run one after another in this process; stop the dashboard "
    "(7 sandboxes while DAYTONA_API_KEY is set) before running them",
}
SUMMARY_NAME = "summary.json"

Parser = Callable[[int | None, str, str, pathlib.Path], dict[str, Any]]


@dataclass(frozen=True)
class Step:
    """One gate: the child's argv (after the interpreter), the key it gets, its timeout and parser."""

    name: str
    argv: tuple[str, ...]
    needs: str  # the one key this child receives; the other is blanked
    timeout_s: float
    parse: Parser
    note: str
    follows: str | None = None  # an inventory checking this step: it runs whenever that step ran
    env: Mapping[str, str] = field(default_factory=dict)  # extra variables for the child
    cleanup_argv: tuple[str, ...] | None = None  # run on failure after a clean inventory-before


# ---------------------------------------------------------------------------------------------- secrets


def scrub(text: str) -> str:
    """`text` with every key value replaced by its name in brackets, so no line or file carries one."""
    for name in SECRET_NAMES:
        value = os.environ.get(name, "").strip()
        if len(value) >= 4 and value in text:
            text = text.replace(value, f"[{name}]")
    return text


def key_presence() -> dict[str, bool]:
    """Which keys are set: booleans only, the values are never put in a record."""
    return {name: bool(os.environ.get(name, "").strip()) for name in SECRET_NAMES}


def fake_flags() -> dict[str, bool]:
    """Which of FAKE, FAKE_LLM and FAKE_SANDBOX are "1" (any of them makes a gate meaningless)."""
    return {flag: os.environ.get(flag) == "1" for flag in FAKE_FLAGS}


# ---------------------------------------------------------------------------------------------- parsers

PING_RE = re.compile(r"^(?P<label>\w+) \((?P<model>[^)]*)\): .* in (?P<ms>\d+(?:\.\d+)?) ms$", re.M)
ROUTER_RE = re.compile(r"^(?P<ok>\d+)/(?P<total>\d+) phrases routed as expected$", re.M)
CANDIDATES_RE = re.compile(
    r"^(?P<first>\d+) of (?P<n>\d+) passed check_source on the first try; (?P<final>\d+) of \d+ after "
    r"regeneration; (?P<s>\d+(?:\.\d+)?) s",
    re.M,
)
SMOKE_TOTAL_RE = re.compile(r"sandbox deleted; total (?P<s>\d+(?:\.\d+)?) s")
SMOKE_SCORED_RE = re.compile(r"scored in (?P<s>\d+(?:\.\d+)?) s by runner '(?P<runner>\w+)'")
INVENTORY_RE = re.compile(r"^remaining \((?P<scope>[^)]*)\): (?P<n>\d+); written (?P<path>.+)$", re.M)
BENCH_TASK_RE = re.compile(
    r"^task (?P<task>\S+): (?P<cands>\d+) candidates x (?P<trials>\d+) trials, (?P<passed>\d+) passed, "
    r"runner (?P<runner>\S+), (?P<s>\d+(?:\.\d+)?) s$",
    re.M,
)
BENCH_POOL_RE = re.compile(r"^pool: (?P<kind>\w+)$", re.M)
BENCH_START_RE = re.compile(r"^start: (?P<live>\d+) sandboxes live in (?P<s>\d+(?:\.\d+)?) s$", re.M)
BENCH_REMAINING_RE = re.compile(r"^remaining: (?P<n>\d+|unknown)$", re.M)
BENCH_RUN_HEAD_RE = re.compile(r"^bench-run .* x(?P<runs>\d+): router=(?P<router>\S+) ", re.M)
BENCH_RUN_RE = re.compile(  # agents.orchestrator.bench_run's verdict line
    r"^(?:median (?P<median>\d+(?:\.\d+)?) s|no median) \((?P<executed>\d+) of (?P<runs>\d+) runs reached "
    r"executing; target (?P<target>\d+(?:\.\d+)?) s\): (?P<verdict>PASS|FAIL)$",
    re.M,
)
BENCH_RUN_WROTE_RE = re.compile(r"^wrote (?P<path>.+)$", re.M)


def _last_json_line(text: str) -> Any:
    """The JSON value of the last stdout line that starts with `{` (None when absent or not JSON)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def score_problem(score: Any) -> str | None:
    """Why `score` is not a schemas/score.json line (None when it is one)."""
    if not isinstance(score, dict):
        return "no JSON score line on stdout"
    errors = sorted(SCORE_VALIDATOR.iter_errors(score), key=lambda e: list(e.absolute_path))
    if errors:
        where = "/".join(str(p) for p in errors[0].absolute_path) or "(root)"
        return f"score line invalid at {where}: {errors[0].message}"
    return None


def parse_ping(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C1: exit 0, a `<label> (<model>): ... in N ms` line for router and codegen, each under 15000 ms."""
    models = {m["label"]: {"model": m["model"], "ms": float(m["ms"])} for m in PING_RE.finditer(out)}
    slow = sorted(label for label, info in models.items() if info["ms"] > PING_LIMIT_MS)
    failed = [line.strip() for line in out.splitlines() if ": FAILED " in line]
    ok = code == 0 and {"router", "codegen"} <= set(models) and not slow and not failed
    return {"pass": ok, "models": models, "limit_ms": PING_LIMIT_MS, "over_limit": slow, "failed": failed}


def _fixture_phrases() -> int | None:
    data = _read_json(ROUTER_PHRASES)
    return len(data) if isinstance(data, list) else None


def parse_router(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C2: exit 0 and `N/N phrases routed as expected` with N the size of the fixture file (20)."""
    m = ROUTER_RE.search(out)
    routed = int(m["ok"]) if m else None
    total = int(m["total"]) if m else None
    expected = _fixture_phrases() or total
    failures = [line.strip() for line in out.splitlines() if line.startswith("FAIL")]
    ok = code == 0 and m is not None and bool(total) and routed == total == expected
    return {"pass": ok, "routed": routed, "phrases": total, "expected": expected, "failures": failures}


def parse_candidates(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C3: exit 0 and `F of N passed check_source on the first try` with F >= min(5, N), N > 0."""
    m = CANDIDATES_RE.search(out)
    summary = _read_json(out_dir / "candidates" / "summary.json")
    if m is None:
        reason = "no summary line"
        if "not a learn request" in out:
            reason = "the router did not answer learn (needs --skills none, see the docstring)"
        return {
            "pass": False,
            "first_try": None,
            "candidates": None,
            "final": None,
            "seconds": None,
            "reason": reason,
            "summary": summary,
        }
    first, n = int(m["first"]), int(m["n"])
    return {
        "pass": code == 0 and n > 0 and first >= min(5, n),
        "first_try": first,
        "candidates": n,
        "final": int(m["final"]),
        "seconds": float(m["s"]),
        "reason": None,
        "summary": summary,
    }


def parse_inventory(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """0 sandboxes remain: exit 0 and `remaining (<scope>): 0; written <path>`."""
    m = INVENTORY_RE.search(out)
    remaining = int(m["n"]) if m else None
    return {
        "pass": code == 0 and remaining == 0,
        "remaining": remaining,
        "scope": m["scope"] if m else None,
        "inventory": m["path"].strip() if m else None,
        "refused": code == 2,
    }


def parse_smoke(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C5: exit 0, runner "daytona", total within 60 s, a schema-valid score line as the last stdout line,
    and the delete took. The numbers come from `<out>/daytona-smoke/result.json`, else from stderr."""
    result = _read_json(out_dir / "daytona-smoke" / "result.json")
    result = result if isinstance(result, dict) else {}
    total = SMOKE_TOTAL_RE.search(err)
    scored = SMOKE_SCORED_RE.search(err)
    seconds = result.get("seconds_total")
    if not isinstance(seconds, (int, float)):
        seconds = float(total["s"]) if total else None
    runner = result.get("runner") or (scored["runner"] if scored else None)
    score = _last_json_line(out)
    problem = score_problem(score)
    deleted = result.get("sandbox_deleted")
    ok = (
        code == 0
        and runner == "daytona"
        and seconds is not None
        and seconds <= SMOKE_LIMIT_S
        and problem is None
        and deleted is not False
    )
    return {
        "pass": ok,
        "runner": runner,
        "seconds_total": seconds,
        "seconds_create": result.get("seconds_create"),
        "seconds_score": result.get("seconds_score"),
        "limit_s": SMOKE_LIMIT_S,
        "score": score if isinstance(score, dict) else None,
        "score_problem": problem,
        "sandbox_deleted": deleted,
        "remaining_sandboxes": result.get("remaining_sandboxes"),
        "result": str(out_dir / "daytona-smoke" / "result.json") if result else None,
    }


def parse_bench(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C6: exit 0, `pool: SandboxPool`, three `task ...` lines each within 35 s and scored by `daytona`
    only. The `remaining: N` line is recorded; the inventory step after it is the proof of 0."""
    tasks = [
        {
            "task": m["task"],
            "candidates": int(m["cands"]),
            "trials": int(m["trials"]),
            "passed": int(m["passed"]),
            "runner": m["runner"],
            "seconds": float(m["s"]),
        }
        for m in BENCH_TASK_RE.finditer(out)
    ]
    pool = BENCH_POOL_RE.search(out)
    start = BENCH_START_RE.search(out)
    rem = BENCH_REMAINING_RE.search(out)
    remaining = None if rem is None or rem["n"] == "unknown" else int(rem["n"])
    slow = [t["task"] for t in tasks if t["seconds"] > BENCH_TASK_LIMIT_S]
    fallback = [t["task"] for t in tasks if t["runner"] != "daytona"]
    ok = (
        code == 0
        and pool is not None
        and pool["kind"] == "SandboxPool"
        and len(tasks) == BENCH_TASKS
        and not slow
        and not fallback
    )
    return {
        "pass": ok,
        "pool": pool["kind"] if pool else None,
        "sandboxes_live": int(start["live"]) if start else None,
        "start_s": float(start["s"]) if start else None,
        "tasks": tasks,
        "limit_s": BENCH_TASK_LIMIT_S,
        "over_limit": slow,
        "local_fallback": fallback,
        "remaining": remaining,
    }


def parse_bench_run(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """C8: exit 0, the live xAI router (`router=xAI`), all 5 runs reached `executing` and the median is
    within 5 s (`median ... : PASS`). The per-run seconds come from the summary.json it wrote."""
    head = BENCH_RUN_HEAD_RE.search(out)
    m = BENCH_RUN_RE.search(out)
    wrote = BENCH_RUN_WROTE_RE.search(out)
    summary = _read_json(pathlib.Path(wrote["path"].strip())) if wrote else None
    summary = summary if isinstance(summary, dict) else {}
    median = float(m["median"]) if m and m["median"] else None
    runs = int(m["runs"]) if m else None
    executed = int(m["executed"]) if m else None
    router = head["router"] if head else None
    ok = (
        code == 0
        and m is not None
        and m["verdict"] == "PASS"
        and router == "xAI"
        and runs == BENCH_RUN_RUNS
        and executed == runs
        and median is not None
        and median <= BENCH_RUN_TARGET_S
    )
    return {
        "pass": ok,
        "router": router,
        "runs": runs,
        "executed": executed,
        "median_s": median,
        "target_s": BENCH_RUN_TARGET_S,
        "seconds": [r.get("seconds") for r in summary.get("runs", []) if isinstance(r, dict)],
        "summary": wrote["path"].strip() if wrote else None,
    }


def parse_voice(code: int | None, out: str, err: str, out_dir: pathlib.Path) -> dict[str, Any]:
    """A14: the probe printed status 200 with token_length > 0 from the live model (not the fake token)."""
    data = _last_json_line(out)
    data = data if isinstance(data, dict) else {}
    length = data.get("token_length")
    ok = (
        code == 0
        and data.get("status") == 200
        and isinstance(length, int)
        and length > 0
        and data.get("fake_llm") is False
        and data.get("fake_token") is False
    )
    keep = ("status", "model", "expires_at", "token_length", "fake_llm", "fake_token", "detail")
    return {"pass": ok, **{key: data.get(key) for key in keep}}


# ---------------------------------------------------------------------------------------------- the plan


def child_env(needs: str | None, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: ours plus `extra`, UTF-8 output, and every key except `needs` set to an
    empty value (the child's own load_dotenv does not override it), so each child holds only its key."""
    env = dict(os.environ)
    for name in SECRET_NAMES:
        if name != needs:
            env[name] = ""
    env.update(extra or {})
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def candidates_supports_skills() -> bool:
    """Whether `python -m agents.candidates --help` lists `--skills`; a failed probe counts as no (the
    candidates step then runs without the flag and says so in its note)."""
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "agents.candidates", "--help"],
            cwd=str(ROOT),
            env=child_env(None),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "--skills" in (proc.stdout or "")


def build_plan(
    out_dir: pathlib.Path,
    *,
    timeouts: Mapping[str, float] | None = None,
    skills_flag: bool | None = None,
) -> list[Step]:
    """All ten steps with their argv for `out_dir`, in order. `skills_flag` says whether agents.candidates
    takes `--skills` (None asks it with --help)."""
    t = dict(DEFAULT_TIMEOUTS)
    t.update(timeouts or {})
    if skills_flag is None:
        skills_flag = candidates_supports_skills()
    out_dir = pathlib.Path(out_dir)
    candidates: tuple[str, ...] = (
        "-m",
        "agents.candidates",
        "--eval",
        CANDIDATES_TEXT,
        "--task-dir",
        str(out_dir / "candidates"),
    )
    candidates_note = "C3: at least 5 of 6 pass check_source on the first try; 7 to 13 paid calls"
    if skills_flag:
        candidates += ("--skills", "none")
    else:
        candidates_note += (
            "; agents.candidates has no --skills flag here, so the live router may answer run place_in_tray "
            "(not a learn) and the step fails"
        )
    sandboxes = str(out_dir / "sandboxes")
    wait = ("--wait", str(INVENTORY_WAIT_S), "--out", sandboxes)
    inventory = ("-m", "agents.sandbox_pool", "--list", "--label", "pool", *wait)
    cleanup = ("-m", "agents.sandbox_pool", "--cleanup", "--label", "pool", *wait)
    after_note = (
        "0 vara=pool sandboxes remain (else --cleanup runs, after a clean inventory-before only, and the "
        "step fails); a listing, no sandbox"
    )
    fakes_on = {"FAKE": "0", "FAKE_LLM": "0", "FAKE_ROBOT": "1", "FAKE_CAMERA": "1", "FAKE_SANDBOX": "1"}
    # bench_run copies the library (VARA_SKILLS_DIR or skills/, read only) into <runs>/bench/run-<stamp>/
    bench_run_env = {**fakes_on, "VARA_RUNS_DIR": str(out_dir / "bench-run")}
    voice_env = {
        **fakes_on,
        "VARA_SKILLS_DIR": str(out_dir / "voice" / "skills"),
        "VARA_RUNS_DIR": str(out_dir / "voice" / "runs"),
    }
    return [
        Step(
            "ping",
            ("-m", "agents.llm", "--ping"),
            XAI,
            t["ping"],
            parse_ping,
            "C1: both models answer, each under 15 s; 2 paid calls",
        ),
        Step(
            "router",
            ("-m", "agents.router", "--eval"),
            XAI,
            t["router"],
            parse_router,
            "C2: all 20 fixture phrases routed as expected by the live model; 20 to 40 paid calls",
        ),
        Step("candidates", candidates, XAI, t["candidates"], parse_candidates, candidates_note),
        Step(
            "inventory-before",
            inventory,
            DAYTONA,
            t["inventory-before"],
            parse_inventory,
            "0 vara=pool sandboxes before any Daytona step (the dashboard is stopped); never cleans up",
        ),
        Step(
            "smoke",
            (
                str(SMOKE_SCRIPT),
                "--out",
                str(out_dir / "daytona-smoke"),
                "--max-seconds",
                f"{SMOKE_LIMIT_S:g}",
            ),
            DAYTONA,
            t["smoke"],
            parse_smoke,
            "C5: a valid score line from the sandbox, total (create + score + delete) within 60 s; 1 sandbox",
        ),
        Step(
            "inventory-after-smoke",
            inventory,
            DAYTONA,
            t["inventory-after-smoke"],
            parse_inventory,
            after_note,
            follows="smoke",
            cleanup_argv=cleanup,
        ),
        Step(
            "bench",
            ("-m", "agents.sandbox_pool", "--bench", "--out", str(out_dir / "bench")),
            DAYTONA,
            t["bench"],
            parse_bench,
            "C6: 3 tasks x 6 candidates x 5 trials, each within 35 s on the sandboxes; 7 sandboxes (the cap)",
        ),
        Step(
            "inventory-after-bench",
            inventory,
            DAYTONA,
            t["inventory-after-bench"],
            parse_inventory,
            after_note,
            follows="bench",
            cleanup_argv=cleanup,
        ),
        Step(
            "bench-run",
            (
                "-m",
                "agents.orchestrator",
                "--bench-run",
                BENCH_RUN_TEXT,
                "--runs",
                str(BENCH_RUN_RUNS),
                "--live-router",
            ),
            XAI,
            t["bench-run"],
            parse_bench_run,
            "C8: 5 of 5 runs reach executing, median within 5 s, live router; fake arm, cameras and "
            "sandboxes, the bench's own copy of the library; 5 paid router calls",
            env=bench_run_env,
        ),
        Step(
            "voice",
            (str(SCRIPT), "--voice-probe"),
            XAI,
            t["voice"],
            parse_voice,
            "A14: POST /voice/token answers 200 with a token from xAI (length and expiry recorded, never the "
            "value); fake arm, cameras and sandboxes; 1 token request",
            env=voice_env,
        ),
    ]


def _names(text: str | None) -> list[str]:
    items = [item.strip() for item in (text or "").split(",") if item.strip()]
    unknown = [item for item in items if item not in STEP_NAMES]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; the steps are {', '.join(STEP_NAMES)}")
    return items


def select_steps(
    only: str | None, skip: str | None, *, skip_daytona: bool = False, skip_xai: bool = False
) -> list[str]:
    """The selected step names in plan order, each sandbox-creating step with its inventories (COMPANIONS).
    ValueError on an unknown step, an empty selection, or a skipped companion."""
    skipped = set(_names(skip))
    wanted = set(_names(only) or STEP_NAMES) - skipped
    if skip_daytona:
        wanted -= set(DAYTONA_STEPS)
    if skip_xai:
        wanted -= set(XAI_STEPS)
    for step, companions in COMPANIONS.items():
        if step not in wanted:
            continue
        dropped = [name for name in companions if name in skipped]
        if dropped:
            raise ValueError(
                f"{step} always runs with {' and '.join(companions)} (they prove it left no sandbox "
                f"behind); do not skip {', '.join(dropped)}"
            )
        wanted |= set(companions)
    selected = [name for name in STEP_NAMES if name in wanted]
    if not selected:
        raise ValueError("no step selected")
    return selected


def parse_timeouts(items: Sequence[str]) -> dict[str, float]:
    """`STEP=SECONDS` overrides; ValueError on an unknown step or a non-positive number."""
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name.strip() not in STEP_NAMES:
            raise ValueError(
                f"--timeout takes STEP=SECONDS with STEP one of {', '.join(STEP_NAMES)}: {item!r}"
            )
        try:
            seconds = float(value)
        except ValueError:
            raise ValueError(f"--timeout {item!r}: SECONDS is not a number") from None
        if not seconds > 0:
            raise ValueError(f"--timeout {item!r}: SECONDS must be positive")
        out[name.strip()] = seconds
    return out


def refusal(out_dir: pathlib.Path, *, explicit_out: bool = False) -> str | None:
    """Why a real run cannot start (None when it can): a fake flag, a missing key, a used --out."""
    fakes = [flag for flag, on in fake_flags().items() if on]
    if fakes:
        return (
            f"{', '.join(fakes)} is 1: the live gates have no fake path; set it to 0 in the shell (.env does "
            "not override a variable the shell set) or remove it from .env; nothing was run"
        )
    missing = [name for name, present in key_presence().items() if not present]
    if missing:
        return (
            f"{' and '.join(missing)} not set (the live gates need both XAI_API_KEY and DAYTONA_API_KEY in "
            ".env); nothing was run"
        )
    if explicit_out and out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        return f"{out_dir} is not a new or empty folder; choose a new --out"
    return None


def default_out_dir() -> pathlib.Path:
    """A folder that does not exist yet: `<runs>/live-gates/<UTC stamp>[-n]/` (runs: VARA_RUNS_DIR or the
    repo's runs/)."""
    base = pathlib.Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs") / "live-gates"
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path, n = base / stamp, 2
    while path.exists():
        path, n = base / f"{stamp}-{n}", n + 1
    return path


# ---------------------------------------------------------------------------------------------- running


def _decode(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shown(argv: Sequence[str]) -> str:
    """A step's command as the human would type it (`python ...`, Windows quoting for the phrases)."""
    return subprocess.list2cmdline(["python", *argv])


def run_child(
    name: str, argv: Sequence[str], timeout_s: float, env: Mapping[str, str], out_dir: pathlib.Path
) -> dict[str, Any]:
    """Run one child to completion or `timeout_s` (then it is killed), write its scrubbed stdout/stderr to
    `<out_dir>/<name>.stdout.log`/`.stderr.log` and return the record: argv (never the environment), exit
    code (None when killed), seconds, timed_out, the log paths and the scrubbed texts."""
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(ROOT),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        code, out, err = proc.returncode, _decode(proc.stdout), _decode(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out, code = True, None
        out, err = _decode(exc.stdout), _decode(exc.stderr)
        err += f"\n[live_gates] killed after the {timeout_s:g} s timeout\n"
    except OSError as exc:
        code, out, err = None, "", f"[live_gates] could not start: {type(exc).__name__}: {exc}\n"
    seconds = round(time.monotonic() - started, 2)
    out, err = scrub(out), scrub(err)
    stdout_log = out_dir / f"{name}.stdout.log"
    stderr_log = out_dir / f"{name}.stderr.log"
    _write(stdout_log, out)
    _write(stderr_log, err)
    return {
        "step": name,
        "argv": [scrub(str(a)) for a in argv],
        "exit_code": code,
        "seconds": seconds,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout": out,
        "stderr": err,
    }


CLEANUP_REFUSED = (
    "not run: inventory-before did not find a clean account, so these sandboxes may belong to another "
    "process; check, then `python -m agents.sandbox_pool --cleanup` by hand"
)


def run_step(step: Step, out_dir: pathlib.Path, *, allow_cleanup: bool = False) -> dict[str, Any]:
    """Run one step: the child, the parser, and on failure its cleanup child when `allow_cleanup` (else the
    cleanup is recorded as not run); write `<out>/<step>.json` and return the record (without the captured
    texts, which are in the logs)."""
    record = run_child(
        step.name, [PYTHON, *step.argv], step.timeout_s, child_env(step.needs, step.env), out_dir
    )
    try:
        parsed = step.parse(record["exit_code"], record["stdout"], record["stderr"], out_dir)
    except Exception as exc:  # noqa: BLE001 - unexpected output fails the step, never the run
        parsed = {"pass": False, "error": scrub(f"parse failed: {type(exc).__name__}: {exc}")}
    record["parsed"] = parsed
    record["pass"] = bool(parsed.get("pass")) and not record["timed_out"]
    record["note"] = step.note
    record["needs"] = step.needs
    if not record["pass"] and step.cleanup_argv:
        if allow_cleanup:
            cleanup = run_child(
                f"{step.name}-cleanup",
                [PYTHON, *step.cleanup_argv],
                step.timeout_s,
                child_env(step.needs, step.env),
                out_dir,
            )
            after = parse_inventory(cleanup["exit_code"], cleanup["stdout"], cleanup["stderr"], out_dir)
            record["cleanup"] = {
                key: cleanup[key]
                for key in ("argv", "exit_code", "seconds", "timed_out", "stdout_log", "stderr_log")
            }
            record["cleanup"]["remaining"] = after["remaining"]
            record["cleanup"]["ran"] = True
        else:
            record["cleanup"] = {"ran": False, "reason": CLEANUP_REFUSED}
    public = {key: value for key, value in record.items() if key not in ("stdout", "stderr")}
    _write(out_dir / f"{step.name}.json", json.dumps(public, indent=2) + "\n")
    return public


def _one_line(record: Mapping[str, Any]) -> str:
    parsed = record.get("parsed") or {}
    name = record["step"]
    if parsed.get("error"):
        return str(parsed["error"])
    if name == "ping":
        return ", ".join(f"{k} {v['ms']:.0f} ms" for k, v in sorted(parsed.get("models", {}).items()))
    if name == "router":
        return f"{parsed.get('routed')}/{parsed.get('phrases')} phrases"
    if name == "candidates":
        text = f"{parsed.get('first_try')} of {parsed.get('candidates')} first try"
        return text + (f" ({parsed['reason']})" if parsed.get("reason") else "")
    if name == "smoke":
        text = f"runner {parsed.get('runner')}, total {parsed.get('seconds_total')} s"
        return text + (f" ({parsed['score_problem']})" if parsed.get("score_problem") else "")
    if name.startswith("inventory"):
        text = f"remaining {parsed.get('remaining')}"
        cleanup = record.get("cleanup")
        if cleanup and cleanup.get("ran"):
            text += f", after cleanup {cleanup.get('remaining')}"
        elif cleanup:
            text += ", cleanup not run (the account was not clean before this run)"
        return text
    if name == "bench":
        tasks = ", ".join(
            f"{t['task']} {t['seconds']:.1f} s ({t['runner']})" for t in parsed.get("tasks", [])
        )
        return f"pool {parsed.get('pool')}; {tasks or 'no task lines'}; remaining {parsed.get('remaining')}"
    if name == "bench-run":
        return (
            f"router {parsed.get('router')}, {parsed.get('executed')} of {parsed.get('runs')} executing, "
            f"median {parsed.get('median_s')} s"
        )
    if name == "voice":
        return f"HTTP {parsed.get('status')}, token length {parsed.get('token_length')}"
    return ""


def _summary(
    plan: Sequence[Step],
    out_dir: pathlib.Path,
    records: Sequence[Mapping[str, Any]],
    *,
    started_at: str,
    t0: float,
    clean_start: bool | None,
    state: str,
) -> dict[str, Any]:
    return {
        "state": state,
        "started_at": started_at,
        "updated_at": _utc(),
        "seconds": round(time.monotonic() - t0, 2),
        "out_dir": str(out_dir),
        "python": PYTHON,
        "plan": [step.name for step in plan],
        "keys_present": key_presence(),
        "fake_flags": fake_flags(),
        "sandbox_caps": dict(SANDBOX_CAPS),
        "clean_start": clean_start,
        "steps": [dict(r) for r in records],
        "passed": state == "finished"
        and len(records) == len(plan)
        and all(r.get("pass") is True for r in records),
        "failed_steps": [r["step"] for r in records if r.get("pass") is False],
        "skipped_steps": [r["step"] for r in records if r.get("skipped")],
    }


def run_gates(
    plan: Sequence[Step], out_dir: pathlib.Path, *, continue_on_failure: bool = False
) -> dict[str, Any]:
    """Run the plan in order and return the summary, also written to `<out>/summary.json` after every step.

    A step that checks another (`follows`) runs whenever that step ran, even after a failure, a timeout or
    Ctrl+C; any other step is skipped after a failure unless `continue_on_failure`, and after Ctrl+C. A
    sandbox-creating step (smoke, bench) is always skipped when inventory-before ran and failed, also with
    `continue_on_failure`, and an inventory's cleanup runs only when inventory-before passed in this run
    (`clean_start`)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at, t0 = _utc(), time.monotonic()
    planned = {step.name for step in plan}
    records: list[dict[str, Any]] = []
    ran: set[str] = set()
    failed = interrupted = False
    clean_start: bool | None = None
    for step in plan:
        reason = None
        if step.follows is not None and step.follows in planned:
            if step.follows not in ran:
                reason = f"{step.follows} did not run"
        elif step.name in COMPANIONS and clean_start is False:
            reason = "inventory-before found vara=pool sandboxes (another VARA process?); nothing created"
        elif interrupted:
            reason = "interrupted (Ctrl+C)"
        elif failed and not continue_on_failure:
            reason = "an earlier step failed (--continue runs everything)"
        if reason is not None:
            records.append(
                {"step": step.name, "skipped": True, "pass": None, "note": step.note, "reason": reason}
            )
            print(f"{step.name:<24} skipped: {reason}", flush=True)
        else:
            print(f"{step.name:<24} running: {shown(step.argv)}", flush=True)
            try:
                record = run_step(step, out_dir, allow_cleanup=clean_start is True)
            except KeyboardInterrupt:
                interrupted = failed = True
                record = {"step": step.name, "pass": False, "interrupted": True, "note": step.note}
                _write(out_dir / f"{step.name}.json", json.dumps(record, indent=2) + "\n")
                print(
                    f"{step.name:<24} INTERRUPTED: only the inventories of steps that ran follow", flush=True
                )
            else:
                verdict = "PASS" if record["pass"] else ("TIMEOUT" if record["timed_out"] else "FAIL")
                print(f"{step.name:<24} {verdict} in {record['seconds']} s: {_one_line(record)}", flush=True)
            records.append(record)
            ran.add(step.name)
            if step.name == "inventory-before":
                clean_start = bool(record.get("pass"))
            if not record.get("pass"):
                failed = True
        summary = _summary(
            plan, out_dir, records, started_at=started_at, t0=t0, clean_start=clean_start, state="running"
        )
        _write(out_dir / SUMMARY_NAME, json.dumps(summary, indent=2) + "\n")
    summary = _summary(
        plan,
        out_dir,
        records,
        started_at=started_at,
        t0=t0,
        clean_start=clean_start,
        state="interrupted" if interrupted else "finished",
    )
    _write(out_dir / SUMMARY_NAME, json.dumps(summary, indent=2) + "\n")
    return summary


def print_plan(plan: Sequence[Step], out_dir: pathlib.Path) -> None:
    """The --dry-run output: each step's command, key, timeout, extra environment and note."""
    keys = key_presence()
    print(f"plan: {len(plan)} step(s), evidence under {out_dir}")
    for i, step in enumerate(plan, 1):
        blanked = ", ".join(name for name in SECRET_NAMES if name != step.needs)
        print(f"  {i:>2}. {step.name:<22} key {step.needs} ({blanked} blanked), timeout {step.timeout_s:g} s")
        print(f"      {shown(step.argv)}")
        if step.env:
            print(f"      env: {' '.join(f'{k}={v}' for k, v in step.env.items())}")
        if step.follows:
            cleanup = shown(step.cleanup_argv or ())
            print(f"      runs whenever {step.follows} ran; on failure, after a clean start: {cleanup}")
        print(f"      {step.note}")
    presence = ", ".join(f"{name}={keys[name]}" for name in SECRET_NAMES)
    print(f"keys present: {presence} (values never printed)")
    print(f"fake flags: {', '.join(f'{flag}={on}' for flag, on in fake_flags().items())}")
    caps = SANDBOX_CAPS
    print(f"sandbox counts: smoke {caps['smoke']}, bench {caps['bench']}; {caps['rule']}")


# ---------------------------------------------------------------------------------------------- voice probe


def voice_probe() -> int:
    """Child mode for the voice step: with FAKE_ROBOT, FAKE_CAMERA and FAKE_SANDBOX forced to 1 and the
    Daytona key blanked (no device, no sandbox) and the model flags as the parent set them, start the
    orchestrator app in this process, POST /voice/token through TestClient and print one JSON line
    {status, model, expires_at, token_length, fake_llm, fake_token, detail}. The token's value is never
    printed, kept or written. Exit 0 iff 200 with a non-empty token from the live model."""
    for flag in ("FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX"):
        os.environ[flag] = "1"
    os.environ[DAYTONA] = ""
    for name in ("VARA_SKILLS_DIR", "VARA_RUNS_DIR"):
        value = os.environ.get(name)
        if value:
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)
    from fastapi.testclient import TestClient

    import agents.orchestrator as orchestrator

    with TestClient(orchestrator.app) as client:
        reply = client.post("/voice/token")
        fake_llm = bool(orchestrator.orch.fake_llm)
    try:
        data = reply.json()
    except ValueError:
        data = {}
    data = data if isinstance(data, dict) else {}
    token = data.get("token")
    result = {
        "status": int(reply.status_code),
        "model": data.get("model"),
        "expires_at": data.get("expires_at"),
        "token_length": len(token) if isinstance(token, str) else 0,
        "fake_llm": fake_llm,
        "fake_token": token == orchestrator.FAKE_VOICE_TOKEN,
        "detail": scrub(str(data.get("detail"))) if reply.status_code != 200 else None,
    }
    print(json.dumps(result), flush=True)
    live_token = result["token_length"] > 0 and not result["fake_llm"] and not result["fake_token"]
    return 0 if result["status"] == 200 and live_token else 1


# ---------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """The CLI (see the module docstring, which --help prints)."""
    paragraphs = (__doc__ or "").split("\n\n")
    parser = argparse.ArgumentParser(
        prog="python scripts/live_gates.py",
        description=paragraphs[0],
        epilog="\n\n".join(p for p in paragraphs[2:] if not p.startswith("Public helpers")),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the keys' presence; run nothing"
    )
    parser.add_argument("--only", metavar="STEP[,STEP]", help="run only these steps (plan order is kept)")
    parser.add_argument("--skip", metavar="STEP[,STEP]", help="leave these steps out")
    parser.add_argument(
        "--skip-daytona",
        action="store_true",
        help="leave out the inventories, smoke and bench (both keys are still required)",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="leave out ping, router, candidates, bench-run and voice (both keys are still required)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_on_failure",
        action="store_true",
        help="run every selected step even after a failure (default: stop, but still run the inventory "
        "that follows a Daytona step)",
    )
    parser.add_argument(
        "--timeout",
        action="append",
        default=[],
        metavar="STEP=SECONDS",
        help="override one step's timeout (repeatable); defaults: "
        + ", ".join(f"{name} {seconds:g}" for name, seconds in DEFAULT_TIMEOUTS.items()),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="evidence folder, new or empty (default runs/live-gates/<UTC stamp>/; under VARA_RUNS_DIR "
        "when it is set)",
    )
    parser.add_argument("--voice-probe", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse, refuse or run the plan; 0 when every selected step passed, 1 if not, 2 refused."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.voice_probe:
        return voice_probe()
    load_dotenv(ROOT / ".env")
    try:
        timeouts = parse_timeouts(args.timeout)
        names = select_steps(args.only, args.skip, skip_daytona=args.skip_daytona, skip_xai=args.skip_xai)
    except ValueError as exc:
        parser.error(str(exc))
    explicit = args.out is not None
    out_dir = pathlib.Path(args.out).resolve() if explicit else default_out_dir().resolve()
    reason = refusal(out_dir, explicit_out=explicit)
    if reason and not args.dry_run:
        print(reason, file=sys.stderr)
        return 2
    skills_flag = candidates_supports_skills() if "candidates" in names else False
    plan = [s for s in build_plan(out_dir, timeouts=timeouts, skills_flag=skills_flag) if s.name in names]
    if args.dry_run:
        print_plan(plan, out_dir)
        if reason:
            print(f"a real run would refuse (exit 2): {reason}")
            return 2
        print("ready: run without --dry-run to execute")
        return 0
    summary = run_gates(plan, out_dir, continue_on_failure=args.continue_on_failure)
    if summary["passed"]:
        verdict = "all passed"
    else:
        verdict = "FAILED: " + ", ".join(summary["failed_steps"] or summary["skipped_steps"] or ["(none)"])
    print(f"{verdict} in {summary['seconds']} s; summary {out_dir / SUMMARY_NAME}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
