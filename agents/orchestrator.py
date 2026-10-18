"""Orchestrator: one sentence in, a checked arm motion out (S0 item 5, C7, C9, C10, C11).

`app` is the FastAPI application (`uvicorn agents.orchestrator:app --port 8000` serves the dashboard) and
`orch` the one `Orchestrator` behind every endpoint. Importing this module loads `.env` and must work with
`FAKE=1` and no hardware, keys or network; nothing here imports, execs or evaluates AI-written code, which
only ever runs through `sim.run_candidate` in a sandbox or the pool's clean local subprocess.

Endpoints (docs/interfaces.md "Orchestrator API", shapes from the brief 8.6):
- `POST /command {"text"}` -> `{"task_id"}`, or 409 `{"detail": "busy"}` while a task runs or the arm is
  stopped; `POST /stop` (optional body `{"source": esc|button|voice|keyboard|api, "t_client_ms": number}`,
  read only after the freeze; nothing in or about the body can refuse a STOP) -> `driver.freeze()` first,
  then the task is cancelled, `stopped` is emitted and the state stays `stopped` until `POST /resume`,
  which calls `driver.resume()`, sends the arm home at <= 30 deg/s and returns `{"state": "idle"}`; the
  `/stop` reply and the `stopped` detail carry `source`, `latency_ms` (request entry to the freeze call,
  server side; null for a stop that did not come through `/stop`) and `press_to_freeze_ms` (the client's
  `t_client_ms` to the end of the freeze call on this laptop's clock; null without a stamp or with one
  later than the freeze, i.e. from another clock), and the log gets one line
  `stop: source=<s> request->freeze <n> ms`; on the REAL arm a thread then appends an A13 record
  (`dashboard-<source>`, latency = `press_to_freeze_ms`) to runs/calibration/esc-latency.jsonl;
  `POST /torque_off` (409 while busy); `POST /calibrate` -> `{"error_cm", "check_points", "lens"}`
  (`error_cm` the fit-marker mean, 0.8 under FAKE_CAMERA; `check_points` null, or `{"n", "mean_cm",
  "max_cm", "ok", "file"}` from the recorded independent check points of docs/table-calibration.md when the
  file named by `VARA_CHECK_POINTS`, default `measurements/check-points.json`, exists; an invalid file is a
  400; the fake never reads it; `lens` the sha256 of the config/lens.npz the checked frame was undistorted
  with, the tracker's own lens, or null for raw frames and under FAKE_CAMERA); `POST /reset_demo` and
  `POST /load_backup` -> `{"skills": [names]}`;
  `POST /voice/token` -> `{"token", "expires_at", "model"}` minted from xAI with the server's key (503
  without a key, a fake token under FAKE_LLM; the key never appears in a reply, a log or a file);
  `GET /state` -> `{"state", "mode", "carry", "task_id", "skills", "queue", "stopped"}`; `WS /events`
  replays the last 200 events and then streams; `GET /` and `GET /voice` serve the pages, `/static` the
  assets, `GET /video/overhead` and `GET /video/wrist` are `multipart/x-mixed-replace` MJPEG streams that
  keep sending frames (5 fps, the fake placeholder at 2 fps).

State: `idle | busy | stopped`; `mode` is `demo` under DEMO_MODE=1 (the judging budgets of the contract
table: routing 10 s, code 30 s with >= 3 valid candidates, simulation 35 s, no repair round, 75 s from the
command to the arm moving, else `queued`), `normal` otherwise; `carry` is `rl` when config/rl.yaml says
`use_policy`, else `ik`. One worker thread per task; every stage runs in a helper thread watched by the
worker, so `/stop` cancels any stage within a few tens of milliseconds and a stage past its budget ends the
task (`queued` while learning, `error` otherwise, the arm going home either way). `arm_lock` serializes the
real-arm section (before-snapshot, validate, replay, home, after-snapshot) with `/resume`.

Flows (every stage checks the stop flag first; events validate against schemas/event.json, go to a
200-event deque, `runs/<task_id>/events.jsonl` and the WebSocket subscribers):
- `run`: routing -> planning (before = `tracker.snapshot(10)`; `agents.planner.plan`) -> validating
  (`arm.replay.validate`) -> executing (`arm.replay.replay` on the driver; one wrist still per grip close
  in `grasp_<n>.jpg`) -> verifying (after = the plan's `final_scene` under FAKE_CAMERA, else
  `vision.verify.take_after`; `vision.verify.verify`) -> the skill's real counts -> done `{"ok", "skill"}`.
- `learn`: spec_ready -> writing_code x6 -> candidate_rejected* -> simulating -> candidate_scored x6
  (score + `runner` + `hint` + `trial_results`) -> selected (`{"candidate", "hint", "scores"}`, the ranked
  list) -> validating -> executing -> verifying -> saved `{"name"}` | not_saved `{"reason"}` (then the
  next-best passing candidate is re-planned on the current scene once and replayed) -> done. With no
  passing candidate the normal mode runs one repair round; still none -> queued -> done.
- `define_routine`: the routine spec (`agents.library.routine_spec`) and the routine candidate
  (`agents.candidates.routine_candidate`) through the learn path, saved under the routine's name.
- `queue`: queued `{"task", "reason"}` -> done. `stop`: like `/stop`.

`runs/<task_id>/` holds `command.txt`, `router.json`, `task.json`, `scene.json`, `prompts/`, `cand_<i>.py`,
`cand_<i>/` (runner outputs), `scores.json`, `selected.json`, `traj.json`, `replay.json`, `before.json`,
`after.json`, `verify.json`, `grasp_<n>.jpg`, `events.jsonl` and, in the default layout, `llm.jsonl`.
`VARA_SKILLS_DIR` and `VARA_RUNS_DIR` (read when the `Orchestrator` is built) move the library and the run
folders, so tests work in temporary directories; when the skills directory has no `_starter`/`_backup`
folder of its own, the repo's are used.

    python -m agents.orchestrator [--port 8000] [--real] [--bench-demo | --bench-run TEXT] [--runs N]
                                  [--live-router]

`--real` watches Esc, asks the human to confirm the arm is clear, counts down 3 s, then connects and serves.
Esc freezes the arm like `/stop` and, on the real arm once the connect has started, appends an A13 record
(`orchestrator-console`, with its connect/session phase; `arm.driver.report_esc_latency`). Without
`--real` and without FAKE_ROBOT/FAKE the server runs but any stage
that would move the arm ends the task with `error` `{"error": "arm not connected: start with --real"}`.
`--bench-demo` forces FAKE=1 and DEMO_MODE=1 and runs five learns of "clear the table" through FastAPI's
TestClient, printing the seconds from the command to `executing` (or `queued`) for each. `--bench-run TEXT`
benchmarks the saved-skill path (routing -> planning -> validating -> executing) for TEXT, such as "hand me
my pills": one line per run with the seconds from the command to `executing`, the median of the runs that
reached `executing` against the C8 target of 5 s, and `<runs dir>/bench/run-<UTC stamp>/summary.json`
(phrase, fakes, mode, per-run seconds and end stage, median, verdict); the bench runs on a copy of the
library in that folder, with its `t-*` task folders beside the summary, so its fake-arm runs never move
the library's `real` counts or mix with the real tasks. By default every fake is on;
`--live-router` clears FAKE and FAKE_LLM for this process (the real xAI router; refused with exit 2 when
XAI_API_KEY is absent) while FAKE_ROBOT, FAKE_CAMERA and FAKE_SANDBOX stay forced, so neither bench ever
touches the arm, a camera or Daytona. A phrase the router sends to a learn is stopped at `spec_ready` and
ends the bench (a live learn would spend six code-generation calls a run). The bench exits 0 iff every
run reached `executing` and the median is within the target.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import httpx
import jsonschema
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import candidates, planner, router, sandbox_pool
from agents import library as library_mod
from arm import driver as driver_mod
from arm import ik, real_robot, replay
from sim import scoring
from vision import calibrate, cameras, markers
from vision import scene as scene_mod
from vision import verify as verify_mod

load_dotenv()

__all__ = [
    "BENCH_TARGET_S",
    "BUDGETS",
    "CHECK_POINTS_PATH",
    "EVENT_HISTORY",
    "FAKE_CALIBRATION_CM",
    "FAKE_VOICE_TOKEN",
    "NOT_CONNECTED",
    "OVER_BUDGET",
    "STAGES",
    "STOP_SOURCES",
    "VOICE_MODEL",
    "VOICE_TOKEN_URL",
    "Busy",
    "Cancelled",
    "Orchestrator",
    "OverBudget",
    "Queued",
    "StageFailed",
    "app",
    "bench_demo",
    "bench_run",
    "check_points_path",
    "main",
    "orch",
    "wait_for_stage",
]

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
SCHEMAS_DIR = ROOT / "schemas"
EVENT_SCHEMA: dict[str, Any] = json.loads((SCHEMAS_DIR / "event.json").read_text(encoding="utf-8"))
EVENT_VALIDATOR = jsonschema.Draft202012Validator(EVENT_SCHEMA)
SCORE_VALIDATOR = jsonschema.Draft202012Validator(
    json.loads((SCHEMAS_DIR / "score.json").read_text(encoding="utf-8"))
)
STAGES: tuple[str, ...] = tuple(EVENT_SCHEMA["properties"]["stage"]["enum"])
FINAL_STAGES = ("done", "stopped", "error")

EVENT_HISTORY = 200
TASK_ID_RE = re.compile(r"^t-(\d+)$")
TRIALS = 5
SEED = 7
N_CANDIDATES = 6
DEMO_MIN_VALID = 3  # demo mode proceeds with this many candidates that passed check_source
STAGE_POLL_S = 0.02  # how often the worker looks at the stop flag and the budget while a stage runs
STAGE_GRACE_S = 2.0  # a stage with its own internal budget (the model calls) gets this much more
HELPER_JOIN_S = 5.0  # /resume and shutdown wait this long for abandoned stage threads
TRIAL_TIMEOUT_S = 60.0  # wall clock per simulated trial (sim.run_candidate --timeout)
PLAN_TIMEOUT_S = 30.0
VERIFY_BUDGET_S = 30.0
FAKE_CALIBRATION_CM = 0.8
VIDEO_FPS = 5.0
FAKE_VIDEO_FPS = 2.0
WRIST_RETRY_S = 10.0  # after a failed wrist camera open, try again this much later
VOICE_MODEL = "grok-voice-latest"
VOICE_TOKEN_URL = "https://api.x.ai/v1/realtime/client_secrets"
VOICE_TOKEN_TTL_S = 300
VOICE_TIMEOUT_S = 10.0
FAKE_VOICE_TOKEN = "fake-voice-token"
ROUTINE_HINT = "Routine: run the saved skills in order."
NOT_CONNECTED = "arm not connected: start with --real"
OVER_BUDGET = "over the 75 s budget"
NO_CANDIDATE = "no candidate passed simulation"
STOP_SOURCES = ("esc", "button", "voice", "keyboard", "api")  # the /stop body's `source`
CHECK_POINTS_PATH = ROOT / "measurements" / "check-points.json"  # docs/table-calibration.md
BENCH_TARGET_S = 5.0  # C8: a saved skill from the command to the arm moving, median of five runs
BENCH_WAIT_S = 150.0  # how long a bench waits for one run to reach its end stage
BENCH_ROUTED = ("planning", "spec_ready", "queued", *FINAL_STAGES)  # what can follow `routing` first
GRIP_CLOSE_RE = re.compile(r"^segment \d+: grip close\b")
BOUNDARY = b"frame"

# The contract's budget table (seconds; `repair` on/off; `to_arm` the command-to-arm-moving cap).
BUDGETS: dict[str, dict[str, Any]] = {
    "normal": {
        "routing": 15.0,
        "writing_code": 45.0,
        "simulating": 90.0,
        "planning": 60.0,
        "verifying": VERIFY_BUDGET_S,
        "executing": 90.0,
        "repair": True,
        "to_arm": None,
    },
    "demo": {
        "routing": 10.0,
        "writing_code": 30.0,
        "simulating": 35.0,
        "planning": 35.0,
        "verifying": VERIFY_BUDGET_S,
        "executing": 90.0,
        "repair": False,
        "to_arm": 75.0,
    },
}


CONNECT_ATTEMPTS = 3  # --real startup: a motor bus glitch during setup gets this many tries
CONNECT_RETRY_S = 2.0
REPAIR_ROUNDS_DEFAULT = 1  # the contract's budget table: one repair round in normal mode
REPAIR_ROUNDS_MAX = 5


def repair_rounds() -> int:
    """How many repair rounds a learn gets in normal mode when no candidate passes: VARA_REPAIR_ROUNDS
    (0 to REPAIR_ROUNDS_MAX), else REPAIR_ROUNDS_DEFAULT. Values beyond the range are clamped and a bad
    value is ignored, both with a warning. DEMO_MODE has no repair round at all (BUDGETS)."""
    raw = os.environ.get("VARA_REPAIR_ROUNDS", "").strip()
    if not raw:
        return REPAIR_ROUNDS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        log.warning("VARA_REPAIR_ROUNDS=%r is not a whole number; using %d", raw, REPAIR_ROUNDS_DEFAULT)
        return REPAIR_ROUNDS_DEFAULT
    clamped = min(max(value, 0), REPAIR_ROUNDS_MAX)
    if clamped != value:
        log.warning("VARA_REPAIR_ROUNDS=%d is outside 0-%d; using %d", value, REPAIR_ROUNDS_MAX, clamped)
    return clamped


def _fake(flag: str) -> bool:
    return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"


# ------------------------------------------------------------------------------------------------------------
# Task control flow
# ------------------------------------------------------------------------------------------------------------


class Busy(Exception):
    """A command arrived while a task runs or the arm is stopped (HTTP 409)."""


class Cancelled(Exception):
    """The task was cancelled: `/stop`, a `stop` request, or the arm froze itself (then `stopped` was
    already emitted and the state is `stopped`)."""


class StageFailed(Exception):
    """A stage failed: the task ends with `error` (the message, plus `extra`) and the arm goes home."""

    def __init__(self, message: str, extra: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.extra = dict(extra or {})


class OverBudget(Exception):
    """A stage ran past its budget: `queued` while learning, `error` otherwise; the arm goes home."""


class Queued(Exception):
    """The task cannot be done now: `queued {task, reason}` then `done`."""

    def __init__(self, task: str, reason: str) -> None:
        super().__init__(reason)
        self.task = task
        self.reason = reason


@dataclass
class _Task:
    id: str
    text: str
    dir: Path
    mode: str
    started: float = field(default_factory=time.monotonic)
    stage: str = "routing"
    action: str | None = None
    arm_moving: bool = False
    clamped: bool = False  # the last stage budget was cut by the 75 s cap

    @property
    def deadline(self) -> float | None:
        cap = BUDGETS[self.mode]["to_arm"]
        return None if cap is None else self.started + float(cap)


@dataclass
class _Scored:
    """One candidate after simulation: the candidate, the pool's scored dict and the event detail."""

    candidate: candidates.Candidate
    scored: dict[str, Any]
    detail: dict[str, Any]

    @property
    def index(self) -> int:
        return int(self.candidate.index)

    @property
    def passed(self) -> bool:
        return bool(self.detail.get("passed"))


@dataclass(frozen=True)
class _Frozen:
    """STOP's first half, handed to its second: when the freeze was called (`time.monotonic()`), when it
    returned (wall clock, ms) and the task it cancelled."""

    called: float
    wall_ms: float
    task_id: str | None


# ------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def _library_for(skills_dir: Path, runs_dir: Path) -> library_mod.Library:
    """The library on `skills_dir`; its starter and backup folders fall back to the repo's own."""
    starter = skills_dir / "_starter"
    backup = skills_dir / "_backup"
    return library_mod.Library(
        skills_dir,
        starter_dir=starter if starter.is_dir() else ROOT / "skills" / "_starter",
        backup_dir=backup if backup.is_dir() else ROOT / "skills" / "_backup",
        archive_root=runs_dir / "archive",
    )


def _jsonable(value: Any) -> Any:
    """A JSON-serializable copy (Paths and other objects become strings)."""
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(data), indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageFailed(f"cannot read {what} {path.name}: {exc}") from None


def _scrub(message: str, secret: str) -> str:
    return message.replace(secret, "[XAI_API_KEY]") if secret else message


def _stop_source(value: Any) -> str:
    """A `/stop` body's `source` normalized to STOP_SOURCES; anything else counts as `api` (logged)."""
    if isinstance(value, str) and value.strip().lower() in STOP_SOURCES:
        return value.strip().lower()
    if value is not None:
        log.warning("stop: unknown source %r recorded as api", value)
    return "api"


def _client_ms(value: Any) -> float | None:
    """A `/stop` body's `t_client_ms` as a finite float, or None when absent or unusable (logged)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    if value is not None:
        log.warning("stop: ignoring t_client_ms %r", value)
    return None


def _stop_body(raw: bytes) -> dict[str, Any]:
    """The `/stop` body as a dict; empty, unreadable or non-object bodies give {} (logged), never an error."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:  # RecursionError: absurdly nested JSON
        log.warning("stop: ignoring an unreadable body: %s", exc)
        return {}
    if not isinstance(data, dict):
        log.warning("stop: ignoring a %s body", type(data).__name__)
        return {}
    return data


def check_points_path() -> Path:
    """The recorded independent check points `/calibrate` reports: `VARA_CHECK_POINTS` or the default."""
    return _env_path("VARA_CHECK_POINTS", CHECK_POINTS_PATH)


def _display_path(path: Path) -> str:
    """`path` relative to the repo when it is inside it (shorter on the dashboard), else as given."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return str(path)


def _timeout_score(index: int) -> dict[str, Any]:
    """The score of a candidate the simulation budget cut off (contract: unfinished -> "timeout")."""
    return {
        "candidate": int(index),
        "passed": False,
        "successes": 0,
        "trials": TRIALS,
        "nominal_success": False,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "mean_sim_s": None,
        "fail_reason": "timeout",
        "traj": None,
        "error": "timeout",
    }


def _skill_summary(skill: Mapping[str, Any]) -> dict[str, Any]:
    spec = skill.get("spec") or {}
    return {
        "name": skill.get("name"),
        "description": spec.get("description", ""),
        "params": dict(spec.get("params") or {}),
        "trusted": bool(skill.get("trusted")),
        "real": dict(skill.get("real") or {}),
        "sim": dict(skill.get("sim") or {}),
        "created": skill.get("created"),
    }


def _placeholder_jpeg(text: str) -> bytes:
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(frame, text, (24, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes() if ok else b""


def _mjpeg_part(jpeg: bytes) -> bytes:
    head = b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
    return head + b"\r\n\r\n" + jpeg + b"\r\n"


def wait_for_stage(
    source: Orchestrator | Sequence[Mapping[str, Any]],
    task_id: str,
    stages: Sequence[str],
    timeout: float,
) -> dict[str, Any] | None:
    """Poll `source` (an Orchestrator, or any sequence of events) until an event of `task_id` with a stage
    in `stages` appears; returns it, or None after `timeout` seconds (benchmarks and tests)."""
    deadline = time.monotonic() + timeout
    while True:
        events = source.recent_events() if isinstance(source, Orchestrator) else list(source)
        for event in events:
            if event.get("task_id") == task_id and event.get("stage") in stages:
                return dict(event)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


# ------------------------------------------------------------------------------------------------------------
# The orchestrator
# ------------------------------------------------------------------------------------------------------------


class Orchestrator:
    """State, events and the task worker (module docstring). Built at import; `start()` in the lifespan."""

    def __init__(self) -> None:
        self.skills_dir = _env_path("VARA_SKILLS_DIR", ROOT / "skills")
        self.runs_dir = _env_path("VARA_RUNS_DIR", ROOT / "runs")
        self.library = _library_for(self.skills_dir, self.runs_dir)
        self.mode = "demo" if os.environ.get("DEMO_MODE") == "1" else "normal"
        self.state = "idle"
        self.task_id: str | None = None
        self.queue: list[dict[str, Any]] = []
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENT_HISTORY)
        self.stop_event = threading.Event()
        self.arm_lock = threading.Lock()
        self.lock = threading.RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.worker: threading.Thread | None = None
        self.helpers: set[threading.Thread] = set()
        self.driver: Any = None
        self.tracker: Any = None
        self.pool: Any = None
        self.started = False
        self._pool_thread: threading.Thread | None = None
        self._wrist: Any = None
        self._wrist_failed_at: float | None = None
        self._placeholders: dict[str, bytes] = {}
        self._last_id = 0

    # -- properties ------------------------------------------------------------------------------------------

    @property
    def carry(self) -> str:
        return "rl" if real_robot.policy_config()["use_policy"] else "ik"

    @property
    def fake_camera(self) -> bool:
        return _fake("FAKE_CAMERA")

    @property
    def fake_llm(self) -> bool:
        return _fake("FAKE_LLM")

    @property
    def runs_is_default(self) -> bool:
        try:
            return self.runs_dir.resolve() == (ROOT / "runs").resolve()
        except OSError:
            return False

    # -- lifecycle -------------------------------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Seed the library, take the tracker, pool and driver singletons; `loop` receives the events."""
        self.loop = loop
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        seeded = self.library.ensure_seeded()
        if seeded:
            log.info("seeded the starter skills: %s", ", ".join(seeded))
        self.tracker = scene_mod.get_tracker()
        if not getattr(self.tracker, "running", True):
            self.tracker.start()
        self.pool = sandbox_pool.get_pool()
        starter = getattr(self.pool, "start", None)
        if callable(starter):
            self._pool_thread = threading.Thread(target=self._start_pool, name="pool-start", daemon=True)
            self._pool_thread.start()
        self.driver = driver_mod.get_driver()
        if getattr(self.driver, "is_fake", False):
            self.driver.connect()
        self.started = True
        log.info(
            "orchestrator ready: mode=%s carry=%s skills=%s runs=%s",
            self.mode,
            self.carry,
            self.skills_dir,
            self.runs_dir,
        )

    def _start_pool(self) -> None:
        try:
            self.pool.start()
        except Exception:
            log.exception("sandbox pool start failed; scoring falls back to the local runner")

    def shutdown(self) -> None:
        """Cancel a running task, then release the pool, the tracker and the driver (torque stays on)."""
        worker = self.worker
        if worker is not None and worker.is_alive():
            if self.driver is not None:
                self.driver.freeze()  # a task is still running: the arm holds, as after /stop
            self.stop_event.set()
            worker.join(HELPER_JOIN_S)
        self._join_helpers(HELPER_JOIN_S)
        if self._pool_thread is not None:
            self._pool_thread.join(HELPER_JOIN_S)
        for label, action in (
            ("pool", getattr(self.pool, "shutdown", None)),
            ("tracker", getattr(self.tracker, "stop", None)),
            ("driver", getattr(self.driver, "disconnect", None)),
        ):
            if callable(action):
                try:
                    action()
                except Exception:
                    log.exception("%s shutdown failed", label)
        if self._wrist is not None:
            try:
                self._wrist.release()
            except Exception:
                log.exception("wrist camera release failed")
            self._wrist = None
        self.started = False
        self.loop = None

    def _join_helpers(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self.lock:
            threads = list(self.helpers)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                log.warning("stage thread %s is still running", thread.name)
        with self.lock:
            self.helpers = {t for t in self.helpers if t.is_alive()}

    # -- events ----------------------------------------------------------------------------------------------

    def emit(self, stage: str, detail: Mapping[str, Any], task_id: str | None = None) -> dict[str, Any]:
        """Build, validate, record and publish one event; returns it."""
        event = {"t": time.time(), "task_id": task_id, "stage": stage, "detail": _jsonable(dict(detail))}
        EVENT_VALIDATOR.validate(event)
        with self.lock:
            self.events.append(event)
            subscribers = list(self.subscribers)
            loop = self.loop
        if task_id is not None:
            self._append_event(task_id, event)
        if loop is not None:
            for queue in subscribers:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                except RuntimeError:  # the loop is closed: the subscriber is gone
                    pass
        log.info("%s %s %s", task_id or "-", stage, json.dumps(event["detail"])[:160])
        return event

    def _append_event(self, task_id: str, event: Mapping[str, Any]) -> None:
        path = self.runs_dir / task_id / "events.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
        except OSError as exc:
            log.warning("could not append to %s: %s", path, exc)

    def recent_events(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)

    def subscribe(self, queue: asyncio.Queue) -> list[dict[str, Any]]:
        """Register a WebSocket queue; returns the backlog to replay first."""
        with self.lock:
            self.subscribers.add(queue)
            return list(self.events)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self.lock:
            self.subscribers.discard(queue)

    # -- the endpoints' work ---------------------------------------------------------------------------------

    def snapshot_state(self) -> dict[str, Any]:
        with self.lock:
            state, task_id, queue = self.state, self.task_id, [dict(q) for q in self.queue]
        return {
            "state": state,
            "mode": self.mode,
            "carry": self.carry,
            "task_id": task_id,
            "skills": [_skill_summary(skill) for skill in self.library.all()],
            "queue": queue,
            "stopped": bool(getattr(self.driver, "stopped", state == "stopped")),
        }

    def submit(self, text: str) -> str:
        """Start the worker for `text`; raises Busy while a task runs or the arm is stopped."""
        with self.lock:
            if self.state != "idle" or (self.worker is not None and self.worker.is_alive()):
                raise Busy(self.state)
            task_id = self._next_task_id()
            task = _Task(task_id, str(text), self.runs_dir / task_id, self.mode)
            self.state = "busy"
            self.task_id = task_id
            self.stop_event.clear()
            self.worker = threading.Thread(
                target=self._run_task, args=(task,), name=f"task-{task_id}", daemon=True
            )
            self.worker.start()
        return task_id

    def _next_task_id(self) -> str:
        highest = self._last_id
        if self.runs_dir.is_dir():
            for entry in self.runs_dir.iterdir():
                match = TASK_ID_RE.match(entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        self._last_id = highest + 1
        return f"t-{self._last_id:04d}"

    def stop(
        self,
        reason: str = "stop requested",
        *,
        source: Any = None,
        t_client_ms: Any = None,
        t_entry: float | None = None,
    ) -> dict[str, Any]:
        """STOP: freeze the arm first, then cancel the task and enter the `stopped` state.

        `source` says who pressed it (STOP_SOURCES; `keyboard` is the console Esc watcher, anything
        unknown counts as `api`), `t_entry` is `/stop`'s `time.monotonic()` on entry and `t_client_ms` the
        client's wall clock in ms at the key press; all three are optional and only looked at after the
        freeze. The reply and the `stopped` detail carry `source`, `latency_ms` (`t_entry` to the freeze
        call; null without a stamp) and `press_to_freeze_ms` (`t_client_ms` to the end of the freeze call
        on this laptop's clock; null without a stamp, or when the stamp is later than the freeze, i.e. from
        another clock), and one log line says `stop: source=<s> request->freeze <n> ms`. `/stop` runs the
        two halves itself, `_freeze_now()` before it reads its body and `_record_stop()` after, so that
        nothing is awaited before the freeze."""
        frozen = self._freeze_now()
        return self._record_stop(frozen, reason, source=source, t_client_ms=t_client_ms, t_entry=t_entry)

    def _freeze_now(self) -> _Frozen:
        """STOP's first half, synchronous: `driver.freeze()`, then the stop flag (the worker abandons its
        stage within STAGE_POLL_S and emits no `stopped` of its own while the flag is set) and `stopped`."""
        called = time.monotonic()
        if self.driver is not None:
            self.driver.freeze()
        wall_ms = time.time() * 1000.0
        self.stop_event.set()
        with self.lock:
            self.state = "stopped"
            task_id = self.task_id
        return _Frozen(called, wall_ms, task_id)

    def _record_stop(
        self,
        frozen: _Frozen,
        reason: str = "stop requested",
        *,
        source: Any = None,
        t_client_ms: Any = None,
        t_entry: float | None = None,
    ) -> dict[str, Any]:
        """STOP's second half (see `stop`): the latencies, the log line, the `stopped` event, the reply."""
        task_id = frozen.task_id
        who = _stop_source(source)
        pressed = _client_ms(t_client_ms)
        latency_ms = None if t_entry is None else round((frozen.called - t_entry) * 1000.0, 2)
        press_ms = None if pressed is None else round(frozen.wall_ms - pressed, 1)
        if press_ms is not None and press_ms < 0.0:
            log.warning(
                "stop: t_client_ms is %.0f ms after the freeze (another device's clock?); "
                "press->freeze not measured",
                -press_ms,
            )
            press_ms = None
        if latency_ms is None:
            log.info("stop: source=%s (%s)", who, reason)
        else:
            log.info(
                "stop: source=%s request->freeze %.2f ms%s",
                who,
                latency_ms,
                "" if press_ms is None else f", press->freeze {press_ms:.0f} ms",
            )
        detail = {"reason": reason, "source": who, "latency_ms": latency_ms, "press_to_freeze_ms": press_ms}
        self.emit("stopped", detail, task_id)
        return {
            "state": "stopped",
            "task_id": task_id,
            "source": who,
            "latency_ms": latency_ms,
            "press_to_freeze_ms": press_ms,
        }

    def resume(self) -> dict[str, Any]:
        """Leave `stopped`: the driver resumes, the arm goes home at <= 30 deg/s, the state is `idle`.

        Refused (`Busy`) while a task runs and no stop was requested; after `/stop` it waits for the
        cancelled worker (and the stage it abandoned) before the arm moves."""
        with self.lock:
            if self.state == "busy" and not self.stop_event.is_set():
                raise Busy("busy")
            worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(HELPER_JOIN_S)
        self._join_helpers(HELPER_JOIN_S)
        if self.driver is not None:
            self.driver.resume()
        self._home("resume")
        self.stop_event.clear()
        with self.lock:
            self.state = "idle"
            self.task_id = None
            if worker is not None and not worker.is_alive():
                self.worker = None
        return {"state": "idle"}

    def torque_off(self) -> dict[str, Any]:
        with self.lock:
            if self.state == "busy":
                raise Busy("busy")
        try:
            self.driver.torque_off()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {"torque": False, "state": self.state}

    def calibrate(self) -> dict[str, Any]:
        """The marker consistency check: `{"error_cm"}`, the fit-marker mean over the overhead frame (the
        FAKE_CALIBRATION_CM fixture under FAKE_CAMERA); 503 when it cannot run, 400 when it fails."""
        return self._marker_check()[0]

    def calibration_report(self) -> dict[str, Any]:
        """`POST /calibrate`: `calibrate()`'s marker check plus `check_points`, the recorded independent
        measurements of docs/table-calibration.md against the same homography (`_check_points`; null
        without a file and always under FAKE_CAMERA, whose fixture never reads it), and `lens`, the sha256
        of the config/lens.npz the checked frame was undistorted with (null for raw frames and under
        FAKE_CAMERA)."""
        result, homography, lens = self._marker_check()
        check_points = None if homography is None else self._check_points(homography)
        return {**result, "check_points": check_points, "lens": lens}

    def _marker_check(self) -> tuple[dict[str, Any], np.ndarray | None, str | None]:
        """`calibrate()`'s reply, the homography it checked (None for the FAKE_CAMERA fixture) and the lens
        sha256 of the frame it checked (`_calibration_frame`; None for raw frames)."""
        if self.fake_camera:
            return {"error_cm": FAKE_CALIBRATION_CM}, None, None
        try:
            frame, lens = self._calibration_frame()
            homography = calibrate.load_homography()
            metrics = calibrate.calibration_metrics(
                homography, markers.detect(frame), calibrate.calib_marker_centers()
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"calibration check failed: {exc}") from None
        if not metrics["accepted"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"calibration marker check failed: mean {metrics['mean_cm']:.3f} cm "
                    f"(limit 1.0), max {metrics['max_cm']:.3f} cm (limit 1.5); "
                    "repeat table calibration and independent check points"
                ),
            )
        return {"error_cm": round(float(metrics["mean_cm"]), 2)}, homography, lens

    def _calibration_frame(self) -> tuple[Any, str | None]:
        """The overhead frame the marker check runs on and its lens sha256 (None: raw frames), so the check
        sees pixels undistorted like the homography's fit. The tracker's `last_frame()` is already
        undistorted by the tracker's own lens (`vision.scene.tracker_diagnostics(tracker)["lens"]`) and is
        used as it is. Without a tracker frame, one frame is read from the camera and undistorted here with
        that same lens (`_tracker_lens`), loaded before the camera opens; a tracker or stand-in that reports
        no lens (the fakes) stays raw. With no tracker at all, config/lens.npz is loaded directly.
        ValueError for a corrupt lens file, a lens file that no longer matches the tracker's or no frame."""
        tracker = self.tracker
        lens_sha = None if tracker is None else scene_mod.tracker_diagnostics(tracker).get("lens")
        last_frame = getattr(tracker, "last_frame", None)
        if callable(last_frame):
            frame = last_frame()
            if frame is not None:
                return frame, lens_sha
        lens = calibrate.load_active_lens() if tracker is None else self._tracker_lens(lens_sha)
        camera = cameras.get_camera("overhead")
        try:
            frame = camera.read()
        finally:
            camera.release()
        if frame is None:
            raise ValueError("no frame from the overhead camera")
        return calibrate.undistort_frame(frame, lens), calibrate.lens_hash(lens)

    def _tracker_lens(self, sha: str | None) -> dict[str, Any] | None:
        """The lens whose sha256 the tracker reports, as a `vision.calibrate.load_active_lens()` result
        loaded once per orchestrator (the tracker keeps its own copy private); None for a raw tracker.
        ValueError when config/lens.npz is no longer that file (restart so tracker and check agree)."""
        if sha is None:
            return None
        cached = getattr(self, "_calibration_lens", None)
        if cached is not None and calibrate.lens_hash(cached) == sha:
            return cached
        lens = calibrate.load_active_lens()
        if calibrate.lens_hash(lens) != sha:
            found = calibrate.lens_hash(lens) or "absent"
            raise ValueError(
                f"config/lens.npz is {found} but the tracker undistorts with {sha}: restart the orchestrator "
                "so the tracker and this check use the same lens"
            )
        self._calibration_lens = lens
        return lens

    @staticmethod
    def _check_points(homography: Any) -> dict[str, Any] | None:
        """`/calibrate`'s `check_points`: null without a measurements file, 400 when the file is invalid,
        else `{n, mean_cm, max_cm, ok, file}` from `vision.calibrate.check_points_metrics`."""
        path = check_points_path()
        if not path.is_file():
            return None
        try:
            metrics = calibrate.check_points_metrics(homography, calibrate.load_check_points(path))
            mean_cm, max_cm = float(metrics["mean_cm"]), float(metrics["max_cm"])
            if not (math.isfinite(mean_cm) and math.isfinite(max_cm)):  # JSON cannot carry inf or NaN
                raise ValueError("a check point maps to infinity through the saved matrix")
        except (ValueError, OSError, RecursionError) as exc:  # RecursionError: absurdly nested JSON
            raise HTTPException(
                status_code=400,
                detail=(
                    f"independent check points file {_display_path(path)} is invalid: {exc}; "
                    "fix or remove it (docs/table-calibration.md)"
                ),
            ) from None
        return {
            "n": int(metrics["count"]),
            "mean_cm": round(mean_cm, 3),
            "max_cm": round(max_cm, 3),
            "ok": bool(metrics["accepted"]),
            "file": _display_path(path),
        }

    def reset_demo(self) -> dict[str, Any]:
        with self.lock:
            if self.state == "busy":
                raise Busy("busy")
            self.queue.clear()
        return {"skills": self.library.reset_demo()}

    def load_backup(self) -> dict[str, Any]:
        return {"skills": self.library.load_backup()}

    def voice_token(self) -> dict[str, Any]:
        """A short-lived xAI realtime token for the voice page; the API key never leaves the server."""
        if self.fake_llm:
            return {
                "token": FAKE_VOICE_TOKEN,
                "expires_at": int(time.time()) + VOICE_TOKEN_TTL_S,
                "model": VOICE_MODEL,
            }
        key = os.environ.get("XAI_API_KEY", "").strip()
        if not key:
            raise HTTPException(status_code=503, detail="XAI_API_KEY is not set; put it in .env")
        try:
            response = httpx.post(
                VOICE_TOKEN_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"expires_after": {"seconds": VOICE_TOKEN_TTL_S}},
                timeout=VOICE_TIMEOUT_S,
            )
            status = int(getattr(response, "status_code", 200))
            if status >= 400:
                raise HTTPException(
                    status_code=502, detail=f"xAI answered HTTP {status} to the token request"
                )
            data = response.json()
        except HTTPException:
            raise
        except Exception as exc:  # the message may quote the request: scrub it before it leaves
            message = _scrub(f"voice token request failed: {type(exc).__name__}: {exc}", key)
            raise HTTPException(status_code=502, detail=message) from None
        secret = (
            data.get("client_secret")
            if isinstance(data, dict) and isinstance(data.get("client_secret"), dict)
            else data
        )
        token = secret.get("value") if isinstance(secret, dict) else None
        if not isinstance(token, str) or not token:
            raise HTTPException(status_code=502, detail="xAI token reply carried no value")
        expires = secret.get("expires_at")
        return {"token": token, "expires_at": expires, "model": VOICE_MODEL}

    # -- video -----------------------------------------------------------------------------------------------

    def placeholder(self, text: str) -> bytes:
        jpeg = self._placeholders.get(text)
        if jpeg is None:
            jpeg = self._placeholders[text] = _placeholder_jpeg(text)
        return jpeg

    def frame_jpeg(self, name: str) -> bytes:
        """One JPEG of the named feed; never raises (a placeholder stands in for a missing frame)."""
        try:
            if name == "overhead":
                if self.tracker is None:
                    return self.placeholder("overhead: no tracker")
                return bytes(self.tracker.annotated_jpeg())
            frame = self._wrist_frame()
            if frame is None:
                return self.placeholder("wrist: no frame")
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buffer.tobytes() if ok else self.placeholder("wrist: encode failed")
        except Exception as exc:
            log.warning("%s frame failed: %s: %s", name, type(exc).__name__, exc)
            return self.placeholder(f"{name}: no feed")

    def _wrist_camera(self) -> Any:
        with self.lock:
            if self._wrist is not None:
                return self._wrist
            if self._wrist_failed_at is not None and time.monotonic() - self._wrist_failed_at < WRIST_RETRY_S:
                return None
            try:
                self._wrist = cameras.get_camera("wrist")
            except Exception as exc:
                self._wrist_failed_at = time.monotonic()
                log.warning("wrist camera unavailable: %s: %s", type(exc).__name__, exc)
                return None
            return self._wrist

    def _wrist_frame(self) -> Any:
        camera = self._wrist_camera()
        return None if camera is None else camera.read()

    async def video_frames(self, name: str) -> AsyncIterator[bytes]:
        """An endless MJPEG part stream: 5 fps, 2 fps when the feed is a fake (re-sent, never stalled)."""
        fake = self.fake_camera
        period = 1.0 / (FAKE_VIDEO_FPS if fake else VIDEO_FPS)
        while True:
            jpeg = await asyncio.to_thread(self.frame_jpeg, name)
            yield _mjpeg_part(jpeg)
            await asyncio.sleep(period)

    # -- the worker ------------------------------------------------------------------------------------------

    def _run_task(self, task: _Task) -> None:
        task.dir.mkdir(parents=True, exist_ok=True)
        _write_text(task.dir / "command.txt", task.text + "\n")
        try:
            self._run_flow(task)
        except Cancelled:
            log.info("%s cancelled during %s", task.id, task.stage)
        except Queued as exc:
            self._queued(task, exc.task, exc.reason)
            self._home(task.id)
        except OverBudget as exc:
            if task.action in ("learn", "define_routine") and not task.arm_moving:
                self._queued(task, task.text, str(exc))
            else:
                self.emit("error", {"error": str(exc), "stage": task.stage}, task.id)
            self._home(task.id)
        except StageFailed as exc:
            self.emit("error", {"error": str(exc), "stage": task.stage, **exc.extra}, task.id)
            self._home(task.id)
        except Exception as exc:  # a bug or a broken dependency: the dashboard must still see an end
            log.exception("%s failed during %s", task.id, task.stage)
            self.emit("error", {"error": f"{type(exc).__name__}: {exc}", "stage": task.stage}, task.id)
            self._home(task.id)
        finally:
            with self.lock:
                if self.state == "busy":
                    self.state = "idle"
                self.task_id = None
                if self.worker is threading.current_thread():
                    self.worker = None

    def _queued(self, task: _Task, text: str, reason: str) -> None:
        self.emit("queued", {"task": text, "reason": reason}, task.id)
        with self.lock:
            self.queue.append({"task": text, "reason": reason, "task_id": task.id})
        self.emit("done", {"ok": False, "queued": True, "reason": reason}, task.id)

    def _home(self, label: str) -> dict[str, Any] | None:
        """Send the arm home under the arm lock; skipped while frozen or unconnected. Never raises."""
        driver = self.driver
        if driver is None or not getattr(driver, "connected", False) or driver.stopped:
            return None
        try:
            with self.arm_lock:
                return self._home_locked(label)
        except Exception:
            log.exception("home failed (%s)", label)
            return None

    def _home_locked(self, label: str) -> dict[str, Any]:
        scene = self.tracker.latest if self.tracker is not None else None
        result = real_robot.home(self.driver, scene=scene)
        if not result["ok"]:
            log.warning("home (%s) ended with %s: %s", label, result["reason"], result["events"][-1:])
        return result

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Cancelled()

    def _budget(self, task: _Task, stage: str) -> float | None:
        value = BUDGETS[task.mode].get(stage)
        return None if value is None else float(value)

    def _clamp(self, task: _Task, budget: float | None) -> float | None:
        """`budget` cut to what is left of the 75 s cap (until the arm moves)."""
        task.clamped = False
        deadline = task.deadline
        if deadline is None or task.arm_moving:
            return budget
        remaining = max(0.0, deadline - time.monotonic())
        if budget is None or remaining < budget:
            task.clamped = True
            return remaining
        return budget

    def _call(
        self,
        task: _Task,
        stage: str,
        fn: Callable[[], Any],
        *,
        budget: float | None = None,
        cancel: threading.Event | None = None,
    ) -> Any:
        """Run `fn` in a helper thread while watching the stop flag and the budget (module docstring)."""
        task.stage = stage
        self._check_stop()
        limit = self._clamp(task, budget)
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # re-raised in the worker below
                box["error"] = exc

        thread = threading.Thread(target=target, name=f"{task.id}-{stage}", daemon=True)
        with self.lock:
            self.helpers.add(thread)
        thread.start()
        deadline = None if limit is None else time.monotonic() + limit
        try:
            while True:
                thread.join(STAGE_POLL_S)
                if not thread.is_alive():
                    break
                if self.stop_event.is_set():
                    if cancel is not None:
                        cancel.set()
                    raise Cancelled()
                if deadline is not None and time.monotonic() >= deadline:
                    if cancel is not None:
                        cancel.set()
                    if task.clamped:
                        raise OverBudget(OVER_BUDGET)
                    raise OverBudget(f"{stage} did not finish within {limit:g} s")
        finally:
            with self.lock:
                if not thread.is_alive():
                    self.helpers.discard(thread)
        self._check_stop()  # a stage that finished right as /stop landed is discarded like the rest
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # -- flows -----------------------------------------------------------------------------------------------

    def _run_flow(self, task: _Task) -> None:
        self.emit("routing", {"text": task.text}, task.id)
        budget = self._budget(task, "routing")
        summaries = self.library.summaries()
        scene = self.tracker.latest()
        llm_task_id = task.id if self.runs_is_default else None
        result = self._call(
            task,
            "routing",
            lambda: router.route(task.text, summaries, scene, task_id=llm_task_id, timeout=budget or 15.0),
            budget=None if budget is None else budget + STAGE_GRACE_S,
        )
        _write_json(task.dir / "router.json", result)
        action = str(result.get("action"))
        task.action = action
        if action == "stop":
            self.stop("stop requested")
            raise Cancelled()
        if action == "queue":
            raise Queued(str(result.get("task") or task.text), str(result.get("reason") or "queued"))
        if action == "run":
            self._flow_run(task, str(result["skill"]), dict(result.get("args") or {}))
        elif action == "learn":
            self._flow_learn(task, dict(result["spec"]))
        elif action == "define_routine":
            self._flow_routine(task, str(result["name"]), list(result.get("steps") or []))
        else:
            raise StageFailed(f"unknown router action {action!r}")

    def _flow_run(self, task: _Task, skill_name: str, args: dict[str, Any]) -> None:
        skill = self.library.get(skill_name)
        if skill is None:
            raise Queued(task.text, f"I don't have a skill called {skill_name.replace('_', ' ')}.")
        self.emit("planning", {"skill": skill_name, "args": args}, task.id)
        with self.arm_lock:
            before = self._before_snapshot(task)
            plan = self._call(
                task,
                "planning",
                lambda: self._plan(skill, args, before, task.dir),
                budget=self._budget(task, "planning"),
            )
            traj, final_scene = plan[0], plan[1]
            _write_json(task.dir / "traj.json", traj)
            self._validate(task, traj, {"skill": skill_name})
            events = self._execute(task, traj, {"skill": skill_name})
            verdict = self._verify(task, skill["spec"], before, args, events, final_scene)
        ok = bool(verdict["ok"])
        self._bump_real(skill_name, ok)
        self.emit("done", {"ok": ok, "skill": skill_name, "reason": verdict.get("reason")}, task.id)

    def _flow_learn(self, task: _Task, spec: dict[str, Any]) -> None:
        _write_json(task.dir / "task.json", spec)
        self.emit("spec_ready", spec, task.id)
        with self.arm_lock:
            before = self._before_snapshot(task)
        cands = self._write_code(task, spec, before)
        results = self._simulate(task, spec, cands, task.dir)
        winner, ranked = self._select(task, results)
        if winner is None and BUDGETS[task.mode]["repair"]:
            rounds = repair_rounds()
            for round_no in range(1, rounds + 1):
                # each round writes against the best failures so far (the latest round's ranking)
                failures = [self._failure_of(entry) for entry in ranked[:3]]
                out_name = "repair" if round_no == 1 else f"repair-{round_no}"
                try:
                    cands = self._write_code(task, spec, before, failures=failures, out_name=out_name)
                except Queued:
                    if round_no == rounds:
                        raise
                    continue  # nothing valid this round (a model timeout, say): the next round tries again
                results = self._simulate(task, spec, cands, task.dir / out_name)
                winner, round_ranked = self._select(task, results)
                if round_ranked:
                    ranked = round_ranked
                if winner is not None:
                    break
        if winner is None:
            raise Queued(task.text, NO_CANDIDATE)
        self._run_and_save(task, spec, winner, ranked, before)

    def _flow_routine(self, task: _Task, name: str, steps: list[dict[str, Any]]) -> None:
        try:
            spec = self.library.routine_spec(name, steps)
        except KeyError as exc:
            raise Queued(task.text, f"I can't define that routine: {exc}") from None
        except ValueError as exc:
            raise Queued(task.text, f"I can't define that routine: {exc}") from None
        code = candidates.routine_candidate(steps)
        _write_json(task.dir / "task.json", spec)
        self.emit("spec_ready", spec, task.id)
        with self.arm_lock:
            before = self._before_snapshot(task)
        path = task.dir / "cand_1.py"
        _write_text(path, code)
        self.emit("writing_code", {"candidate": 1, "hint": ROUTINE_HINT}, task.id)
        routine = candidates.Candidate(1, ROUTINE_HINT, code, [], 1, path)
        results = self._simulate(task, spec, [routine], task.dir)
        winner, ranked = self._select(task, results)
        if winner is None:
            reason = winner_reason(ranked) or NO_CANDIDATE
            raise Queued(task.text, f"the routine did not pass simulation: {reason}")
        self._run_and_save(task, spec, winner, ranked, before)

    # -- stages ----------------------------------------------------------------------------------------------

    def _before_snapshot(self, task: _Task) -> dict[str, Any]:
        """`tracker.snapshot(10)`: the scene the planner, the sandboxes and the verification all use."""
        try:
            before = self.tracker.snapshot(10)
        except TimeoutError as exc:
            raise StageFailed(f"the overhead camera delivered no frames: {exc}") from None
        _write_json(task.dir / "scene.json", before)
        return before

    def _plan(
        self, skill: Mapping[str, Any], args: Mapping[str, Any], scene: Mapping[str, Any], task_dir: Path
    ) -> Any:
        try:
            return planner.plan(
                skill, args, scene, task_dir=task_dir, pool=self.pool, skills_dir=self.skills_dir
            )
        except Exception as exc:
            raise StageFailed(f"planning failed: {exc}") from None

    def _write_code(
        self,
        task: _Task,
        spec: dict[str, Any],
        scene: dict[str, Any],
        failures: list[dict[str, Any]] | None = None,
        out_name: str = "repair",
    ) -> list[candidates.Candidate]:
        budget = self._budget(task, "writing_code") or 45.0
        hints = candidates.HINTS
        for i in range(1, N_CANDIDATES + 1):
            self.emit("writing_code", {"candidate": i, "hint": hints[(i - 1) % len(hints)]}, task.id)

        def on_rejected(candidate: candidates.Candidate) -> None:
            detail = {
                "candidate": candidate.index,
                "problems": list(candidate.problems),
                "attempt": candidate.attempt,
            }
            self.emit("candidate_rejected", detail, task.id)

        summaries = self.library.summaries()
        demo = task.mode == "demo"
        if failures is None:

            def write() -> list[candidates.Candidate]:
                return candidates.generate(
                    spec,
                    scene,
                    summaries,
                    task_dir=task.dir,
                    n=N_CANDIDATES,
                    timeout=budget,
                    demo_mode=demo,
                    on_rejected=on_rejected,
                )

        else:
            repairs = failures

            def write() -> list[candidates.Candidate]:
                return candidates.repair(
                    spec,
                    scene,
                    summaries,
                    repairs,
                    task_dir=task.dir,
                    timeout=budget,
                    n=N_CANDIDATES,
                    on_rejected=on_rejected,
                    out_name=out_name,
                )

        cands = list(self._call(task, "writing_code", write, budget=budget + STAGE_GRACE_S))
        valid = [c for c in cands if c.accepted]
        if demo and len(valid) < DEMO_MIN_VALID:
            raise Queued(
                task.text,
                f"only {len(valid)} of {len(cands)} candidate programs were valid (need {DEMO_MIN_VALID})",
            )
        if not valid:
            raise Queued(task.text, "no valid candidate program was written")
        return cands

    def _simulate(
        self, task: _Task, spec: Mapping[str, Any], cands: list[candidates.Candidate], out_dir: Path
    ) -> dict[int, _Scored]:
        """Score the accepted candidates through the pool; every candidate ends with a `candidate_scored`."""
        valid = [c for c in cands if c.accepted and c.path is not None]
        self.emit("simulating", {"candidates": [c.index for c in valid], "trials": TRIALS}, task.id)
        task_path = task.dir / "task.json"
        scene_path = task.dir / "scene.json"
        budget = self._budget(task, "simulating") or 90.0
        cancel = threading.Event()
        by_index = {c.index: c for c in cands}
        results: dict[int, _Scored] = {}
        guard = threading.Lock()

        def record(scored: Mapping[str, Any]) -> None:
            index = int(scored["candidate"])
            candidate = by_index.get(index)
            if candidate is None:
                return
            entry = _Scored(candidate, dict(scored), self._scored_detail(scored, candidate.hint))
            with guard:
                if index in results:
                    return
                results[index] = entry
            self.emit("candidate_scored", entry.detail, task.id)

        def run() -> Any:
            return self.pool.score(
                out_dir,
                [(c.index, c.path) for c in valid],
                str(task_path),
                str(scene_path),
                str(self.skills_dir),
                trials=TRIALS,
                seed=SEED,
                timeout=min(TRIAL_TIMEOUT_S, budget),
                on_score=record,
                cancel=cancel,
            )

        try:
            returned = self._call(task, "simulating", run, budget=budget, cancel=cancel)
        except OverBudget:
            cancel.set()
            with guard:
                have_any = bool(results)
            if not have_any:
                raise
            log.warning("%s: simulation over budget; scoring what finished", task.id)
            returned = None
        except Cancelled:
            cancel.set()
            raise
        for scored in returned or []:
            record(scored)
        for candidate in cands:
            with guard:
                known = candidate.index in results
            if known:
                continue
            if candidate.problems:
                score = scoring.rejected_score(candidate.index, TRIALS, candidate.problems[0])
            else:
                score = _timeout_score(candidate.index)
            record(
                {
                    "candidate": candidate.index,
                    "runner": "none",
                    "score": score,
                    "trial_results": [],
                    "out_dir": None,
                }
            )
        _write_json(out_dir / "scores.json", [results[i].detail for i in sorted(results)])
        return results

    @staticmethod
    def _scored_detail(scored: Mapping[str, Any], hint: str) -> dict[str, Any]:
        score = dict(scored.get("score") or {})
        try:
            SCORE_VALIDATOR.validate(score)
        except jsonschema.ValidationError as exc:
            log.warning("score of candidate %s is off the schema: %s", scored.get("candidate"), exc.message)
        return {
            **score,
            "candidate": int(scored.get("candidate", score.get("candidate", 0))),
            "runner": str(scored.get("runner") or "local"),
            "hint": hint,
            "trial_results": [bool(v) for v in (scored.get("trial_results") or [])],
        }

    def _select(self, task: _Task, results: dict[int, _Scored]) -> tuple[_Scored | None, list[_Scored]]:
        """Rank (`sim.scoring.rank`); emit `selected` for the best passing candidate, if any."""
        order = {
            int(d["candidate"]): k for k, d in enumerate(scoring.rank([r.detail for r in results.values()]))
        }
        ranked = sorted(results.values(), key=lambda entry: order.get(entry.index, len(order)))
        winner = ranked[0] if ranked and ranked[0].passed else None
        if winner is not None:
            detail = {
                "candidate": winner.index,
                "hint": winner.candidate.hint,
                "scores": [r.detail for r in ranked],
            }
            self.emit("selected", detail, task.id)
            _write_json(task.dir / "selected.json", detail)
        return winner, ranked

    @staticmethod
    def _failure_of(entry: _Scored) -> dict[str, Any]:
        return {
            "index": entry.index,
            "hint": entry.candidate.hint,
            "code": entry.candidate.source,
            "score": dict(entry.scored.get("score") or {}),
            "fail_reason": entry.detail.get("fail_reason") or entry.detail.get("error") or "failed",
        }

    def _run_and_save(
        self,
        task: _Task,
        spec: dict[str, Any],
        winner: _Scored,
        ranked: list[_Scored],
        before: dict[str, Any],
    ) -> None:
        traj, final_scene = self._candidate_outputs(winner)
        _write_json(task.dir / "traj.json", traj)
        with self.arm_lock:
            self._validate(task, traj, {"candidate": winner.index})
            events = self._execute(task, traj, {"candidate": winner.index})
            verdict = self._verify(task, spec, before, {}, events, final_scene)
        ok = bool(verdict["ok"])
        if ok:
            self._save(task, spec, winner)
        else:
            self.emit("not_saved", {"reason": str(verdict.get("reason") or "spec not satisfied")}, task.id)
            runner_up = next((r for r in ranked if r.passed and r.index != winner.index), None)
            if runner_up is not None:
                ok = self._replan(task, spec, runner_up)
        self.emit("done", {"ok": ok, "skill": spec["name"]}, task.id)

    def _candidate_outputs(self, entry: _Scored) -> tuple[dict[str, Any], dict[str, Any] | None]:
        out_dir = entry.scored.get("out_dir")
        if not out_dir:
            raise StageFailed(f"candidate {entry.index} has no runner output directory")
        folder = Path(out_dir)
        traj = _read_json(folder / "traj.json", "trajectory")
        final_path = folder / "final_scene.json"
        final_scene = _read_json(final_path, "final scene") if final_path.is_file() else None
        return traj, final_scene

    def _replan(self, task: _Task, spec: dict[str, Any], entry: _Scored) -> bool:
        """The not_saved path: plan the next-best passing candidate on the current scene, replay once."""
        log.info("%s: re-planning candidate %d after not_saved", task.id, entry.index)
        folder = task.dir / "replan"
        skill = library_mod.make_skill(
            spec, entry.candidate.source, dict(entry.scored.get("score") or {}), None
        )
        skill_path = folder / "skill.json"
        _write_json(skill_path, skill)
        self.emit("planning", {"candidate": entry.index, "args": {}}, task.id)
        with self.arm_lock:
            before = self._before_snapshot(task)
            scene_path = folder / "scene.json"
            _write_json(scene_path, before)

            def plan() -> Any:
                return self.pool.plan(
                    str(skill_path),
                    {},
                    str(scene_path),
                    str(folder / "out"),
                    str(self.skills_dir),
                    timeout=PLAN_TIMEOUT_S,
                )

            scored = self._call(task, "planning", plan, budget=self._budget(task, "planning"))
            score = dict((scored or {}).get("score") or {})
            if not score.get("passed"):
                self.emit(
                    "not_saved",
                    {"reason": f"re-plan failed: {score.get('fail_reason') or 'no trajectory'}"},
                    task.id,
                )
                return False
            replanned = _Scored(
                entry.candidate, dict(scored), self._scored_detail(scored, entry.candidate.hint)
            )
            traj, final_scene = self._candidate_outputs(replanned)
            _write_json(folder / "traj.json", traj)
            self._validate(task, traj, {"candidate": entry.index})
            events = self._execute(task, traj, {"candidate": entry.index})
            verdict = self._verify(task, spec, before, {}, events, final_scene)
        if verdict["ok"]:
            self._save(task, spec, entry)
            return True
        self.emit(
            "not_saved",
            {"reason": f"second attempt: {verdict.get('reason') or 'spec not satisfied'}"},
            task.id,
        )
        return False

    def _validate(self, task: _Task, traj: Mapping[str, Any], extra: Mapping[str, Any]) -> None:
        segments = traj.get("segments") if isinstance(traj, Mapping) else None
        self.emit(
            "validating", {**extra, "segments": len(segments) if isinstance(segments, list) else 0}, task.id
        )
        if not getattr(self.driver, "connected", False):
            raise StageFailed(NOT_CONNECTED)

        def check() -> list[str]:
            return list(replay.validate(traj, replay.load_limits(), ik.fk_tip, replay.load_box()))

        problems = self._call(task, "validating", check, budget=None)
        if problems:
            _write_json(task.dir / "validate.json", problems)
            raise StageFailed(f"trajectory rejected: {problems[0]}", {"problems": problems})

    def _execute(self, task: _Task, traj: Mapping[str, Any], extra: Mapping[str, Any]) -> list[str]:
        """Replay the trajectory on the arm (the 75 s cap is checked right before), then go home."""
        deadline = task.deadline
        if deadline is not None and time.monotonic() > deadline:
            raise OverBudget(OVER_BUDGET)
        if not getattr(self.driver, "connected", False):
            raise StageFailed(NOT_CONNECTED)
        task.arm_moving = True
        segments = traj.get("segments") or []
        self.emit("executing", {**extra, "segments": len(segments)}, task.id)
        grasps = [0]

        def on_event(text: str) -> None:
            if GRIP_CLOSE_RE.match(text):
                grasps[0] += 1
                self._wrist_still(task, grasps[0])

        def run() -> dict[str, Any]:
            return replay.replay(dict(traj), driver=self.driver, scene=self.tracker.latest, on_event=on_event)

        try:
            result = self._call(task, "executing", run, budget=self._budget(task, "executing"))
        except OverBudget as exc:
            self.driver.freeze()
            self.emit("error", {"error": f"{exc}: arm frozen, press Resume", "stage": "executing"}, task.id)
            self._enter_stopped(task, str(exc))
            raise Cancelled() from None
        _write_json(task.dir / "replay.json", result)
        if self.driver.stopped:
            if self.stop_event.wait(0.5):
                raise Cancelled()  # /stop did it and records `stopped` itself
            self._enter_stopped(task, str(result.get("reason") or "stopped"))
            raise Cancelled()
        if not result.get("ok"):
            tail = [str(e) for e in (result.get("events") or [])[-3:]]
            raise StageFailed(f"replay failed: {result.get('reason')}", {"events": tail})
        homed = self._home_locked(task.id)
        if self.driver.stopped:
            if self.stop_event.wait(0.5):
                raise Cancelled()
            self._enter_stopped(task, str(homed.get("reason") or "stopped"))
            raise Cancelled()
        return [str(e) for e in result.get("events") or []]

    def _enter_stopped(self, task: _Task, reason: str) -> None:
        """The arm froze itself (hand too close, a driver error, an execution overrun): `/resume` needed."""
        self.stop_event.set()
        with self.lock:
            self.state = "stopped"
        self.emit("stopped", {"reason": reason, "stage": task.stage}, task.id)

    def _wrist_still(self, task: _Task, n: int) -> None:
        """One wrist still per grasp (B8): `grasp_<n>.jpg`; a camera without a frame is tolerated."""
        try:
            frame = self._wrist_frame()
            if frame is None:
                log.warning("%s: no wrist frame for grasp %d", task.id, n)
                return
            cv2.imwrite(str(task.dir / f"grasp_{n}.jpg"), frame)
        except Exception as exc:
            log.warning("%s: wrist still %d failed: %s: %s", task.id, n, type(exc).__name__, exc)

    def _verify(
        self,
        task: _Task,
        spec: Mapping[str, Any],
        before: Mapping[str, Any],
        args: Mapping[str, Any],
        events: Sequence[str],
        final_scene: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        self.emit("verifying", {"english": spec.get("english", "")}, task.id)

        def check() -> dict[str, Any]:
            if self.fake_camera and final_scene is not None:
                after = dict(final_scene)
            else:
                after = verify_mod.take_after(self.tracker, spec, args, before=before)
            return dict(verify_mod.verify(spec, before, after, args, list(events), task_dir=task.dir))

        try:
            return self._call(task, "verifying", check, budget=self._budget(task, "verifying"))
        except TimeoutError as exc:
            raise StageFailed(f"the overhead camera delivered no after-snapshot: {exc}") from None

    def _save(self, task: _Task, spec: dict[str, Any], entry: _Scored) -> None:
        score = dict(entry.scored.get("score") or entry.detail)
        sim = {"successes": int(score.get("successes", 0)), "trials": int(score.get("trials", TRIALS))}
        try:
            skill = library_mod.make_skill(spec, entry.candidate.source, sim, {"runs": 1, "successes": 1})
            self.library.save(skill)
        except ValueError as exc:  # LibraryError included
            raise StageFailed(f"could not save the skill: {exc}") from None
        self.emit("saved", {"name": skill["name"]}, task.id)

    def _bump_real(self, name: str, ok: bool) -> None:
        skill = self.library.get(name)
        if skill is None:
            return
        real = dict(skill.get("real") or {})
        real["runs"] = int(real.get("runs", 0)) + 1
        real["successes"] = int(real.get("successes", 0)) + (1 if ok else 0)
        skill["real"] = real
        try:
            self.library.save(skill)
        except ValueError as exc:
            log.warning("could not update the real counts of %s: %s", name, exc)


def winner_reason(ranked: Sequence[_Scored]) -> str | None:
    """The best candidate's failure phrase (for a routine that did not pass)."""
    if not ranked:
        return None
    detail = ranked[0].detail
    return detail.get("fail_reason") or detail.get("error")


# ------------------------------------------------------------------------------------------------------------
# The FastAPI app
# ------------------------------------------------------------------------------------------------------------

orch = Orchestrator()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    orch.start(asyncio.get_running_loop())
    try:
        yield
    finally:
        await asyncio.to_thread(orch.shutdown)


app = FastAPI(title="VARA orchestrator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CommandBody(BaseModel):
    text: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/voice")
async def voice_page() -> FileResponse:
    return FileResponse(WEB_DIR / "voice.html")


@app.get("/state")
async def get_state() -> dict[str, Any]:
    return await asyncio.to_thread(orch.snapshot_state)


@app.get("/runtime")
async def get_runtime() -> dict[str, str]:
    """Display execution provenance without changing the shared state contract or exposing keys."""
    driver = orch.driver
    robot = "simulation" if getattr(driver, "is_fake", _fake("FAKE_ROBOT")) else "physical"
    if robot == "physical" and not getattr(driver, "connected", False):
        robot = "physical, disconnected"
    return {
        "robot": robot,
        "camera": "simulated" if _fake("FAKE_CAMERA") else "physical",
        "model": "fixtures" if _fake("FAKE_LLM") else "xAI",
        "scoring": "local"
        if _fake("FAKE_SANDBOX") or not os.environ.get("DAYTONA_API_KEY")
        else "Daytona with local fallback",
    }


@app.post("/command")
async def post_command(body: CommandBody) -> dict[str, Any]:
    try:
        task_id = orch.submit(body.text)
    except Busy:
        raise HTTPException(status_code=409, detail="busy") from None
    return {"task_id": task_id}


@app.post("/stop")
async def post_stop(request: Request) -> dict[str, Any]:
    """STOP. The arm freezes before anything is awaited; only then is the optional body `{"source",
    "t_client_ms"}` read, leniently (module docstring), and the stop recorded with it. Nothing a client
    sends, or fails to send, can refuse or delay the freeze, and the `stopped` event is recorded whatever
    becomes of the body read."""
    t_entry = time.monotonic()
    current = orch
    frozen = current._freeze_now()
    try:
        raw = await request.body()
    except Exception as exc:  # the client went away mid-body: the arm is frozen all the same
        log.warning("stop: could not read the body (%s: %s)", type(exc).__name__, exc)
        raw = b""
    except BaseException:  # the request was cancelled (shutdown): record the stop, then let it end
        current._record_stop(frozen, t_entry=t_entry)
        raise
    body = _stop_body(raw)
    reply = current._record_stop(
        frozen, source=body.get("source"), t_client_ms=body.get("t_client_ms"), t_entry=t_entry
    )
    _record_dashboard_stop(current.driver, reply)
    return reply


def _record_dashboard_stop(driver: Any, reply: dict[str, Any]) -> None:
    """A13 evidence for a dashboard/voice STOP on the REAL arm: one `esc-latency.jsonl` record with source
    `dashboard-<source>`, written on a daemon thread after the freeze (never on the STOP path). Its
    `latency_ms` is the reply's `press_to_freeze_ms` (key press to freeze on this laptop's clock: the
    number A13's 200 ms bar is about; null without a client stamp), the server-side `latency_ms` goes in
    `extra`. Nothing is recorded for the fake arm or a disconnected driver."""
    if (
        driver is None
        or getattr(driver, "is_fake", True) is not False
        or not getattr(driver, "connected", False)
    ):
        return
    path = driver_mod.esc_latency_path()  # resolved now, with this request's VARA_RUNS_DIR, not in the thread

    def write() -> None:
        try:
            driver_mod.record_esc_latency(
                path,
                source=f"dashboard-{reply.get('source')}",
                latency_ms=reply.get("press_to_freeze_ms"),
                extra={"server_latency_ms": reply.get("latency_ms"), "task_id": reply.get("task_id")},
                phase="session",
            )
        except Exception:
            log.exception("stop: could not append the Esc latency record")

    threading.Thread(target=write, name="stop-record", daemon=True).start()


@app.post("/resume")
async def post_resume() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(orch.resume)
    except Busy:
        raise HTTPException(status_code=409, detail="busy") from None


@app.post("/torque_off")
async def post_torque_off() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(orch.torque_off)
    except Busy:
        raise HTTPException(status_code=409, detail="busy") from None


@app.post("/calibrate")
async def post_calibrate() -> dict[str, Any]:
    return await asyncio.to_thread(orch.calibration_report)


@app.post("/reset_demo")
async def post_reset_demo() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(orch.reset_demo)
    except Busy:
        raise HTTPException(status_code=409, detail="busy") from None


@app.post("/load_backup")
async def post_load_backup() -> dict[str, Any]:
    return await asyncio.to_thread(orch.load_backup)


@app.post("/voice/token")
async def post_voice_token() -> dict[str, Any]:
    return await asyncio.to_thread(orch.voice_token)


@app.get("/video/{name}")
async def video(name: str) -> StreamingResponse:
    if name not in ("overhead", "wrist"):
        raise HTTPException(status_code=404, detail="no such feed")
    media = f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"
    return StreamingResponse(orch.video_frames(name), media_type=media, headers={"Cache-Control": "no-cache"})


@app.websocket("/events")
async def events_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    backlog = orch.subscribe(queue)
    receiver = asyncio.ensure_future(websocket.receive())
    getter = asyncio.ensure_future(queue.get())
    try:
        for event in backlog:
            await websocket.send_json(event)
        while True:
            done, _pending = await asyncio.wait({receiver, getter}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                message = receiver.result()
                if message.get("type") == "websocket.disconnect":
                    break
                receiver = asyncio.ensure_future(websocket.receive())
            if getter in done:
                await websocket.send_json(getter.result())
                getter = asyncio.ensure_future(queue.get())
    except Exception as exc:  # the client went away mid-send; nothing to do but let go
        log.debug("events socket closed: %s: %s", type(exc).__name__, exc)
    finally:
        orch.unsubscribe(queue)
        for future in (receiver, getter):
            future.cancel()


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------


def _bench_settle(client: Any, task_id: str, timeout: float, *, reset: bool) -> None:
    """Wait for `task_id` to end and the worker to let go; `/resume` after a stop, `/reset_demo` if asked."""
    wait_for_stage(orch, task_id, FINAL_STAGES, timeout)
    deadline = time.monotonic() + 10.0
    while orch.state == "busy" and time.monotonic() < deadline:
        time.sleep(0.05)
    if orch.state == "stopped":
        client.post("/resume")
    if reset:
        client.post("/reset_demo")


def _bench_commands(
    client: Any, text: str, runs: int, timeout: float, *, reset: bool, saved_skill: bool = False
) -> list[dict[str, Any]]:
    """The loop both benches share: `runs` commands of `text` through `client`, one at a time, each timed
    from the POST to its first `executing`/`queued`/`error`/`stopped` event (one printed line per run),
    then waited to its end, resumed when stopped and, with `reset`, followed by `/reset_demo`. Returns one
    record per run: `{run, task_id, stage, seconds, detail}` (`stage` is `rejected` for a refused POST and
    `no end stage` after `timeout`).

    With `saved_skill` (`--bench-run`) a command the router sends to a learn or a routine instead
    (`spec_ready` before any `planning`) is stopped at once and ends the loop, recorded with the stage
    `not a saved skill`: it is not the path being measured, and a live learn would spend six
    code-generation calls per run."""
    records: list[dict[str, Any]] = []
    for k in range(1, runs + 1):
        started = time.monotonic()
        reply = client.post("/command", json={"text": text})
        if reply.status_code != 200:
            print(f"run {k}: POST /command returned {reply.status_code}: {reply.text}")
            detail = {"status": reply.status_code, "reply": reply.text[:100]}
            records.append(
                {"run": k, "task_id": None, "stage": "rejected", "seconds": None, "detail": detail}
            )
            continue
        task_id = reply.json()["task_id"]
        if saved_skill:
            routed = wait_for_stage(orch, task_id, BENCH_ROUTED, timeout)
            if routed is not None and routed["stage"] == "spec_ready":
                client.post("/stop", json={"source": "api"})
                name = routed["detail"].get("name")
                print(
                    f"run {k} ({task_id}): the router started a learn of {name!r}, not a saved skill: "
                    "stopped it and the bench (check the phrase and the library)"
                )
                detail = {"routed_to": "learn", "spec": name}
                record = {"run": k, "task_id": task_id, "stage": "not a saved skill", "seconds": None}
                records.append({**record, "detail": detail})
                _bench_settle(client, task_id, timeout, reset=reset)
                break
        left = max(0.0, timeout - (time.monotonic() - started))  # one budget per run, routing included
        event = wait_for_stage(orch, task_id, ("executing", "queued", "error", "stopped"), left)
        seconds = time.monotonic() - started
        stage = event["stage"] if event else "no end stage"
        detail = json.dumps(event["detail"])[:100] if event else ""
        print(f"run {k} ({task_id}): {stage} after {seconds:.1f} s  {detail}")
        records.append(
            {
                "run": k,
                "task_id": task_id,
                "stage": stage,
                "seconds": round(seconds, 3),
                "detail": dict(event["detail"]) if event else None,
            }
        )
        _bench_settle(client, task_id, timeout, reset=reset)
    return records


def bench_demo(text: str = "clear the table", runs: int = 5, timeout: float = BENCH_WAIT_S) -> int:
    """Five demo-mode learns through the TestClient; prints seconds from the command to `executing`."""
    global orch
    os.environ["FAKE"] = "1"
    os.environ["DEMO_MODE"] = "1"
    from fastapi.testclient import TestClient

    orch = Orchestrator()
    cap = float(BUDGETS["demo"]["to_arm"])
    with TestClient(app) as client:
        client.post("/reset_demo")
        records = _bench_commands(client, text, runs, timeout, reset=True)
    within = sum(
        1
        for record in records
        if record["stage"] in ("executing", "queued")
        and record["seconds"] is not None
        and record["seconds"] <= cap
    )
    print(f"{within} of {runs} runs reached the arm or were queued within {cap:g} s")
    return 0 if within == runs else 1


def _bench_out_dir(base: Path) -> Path:
    """A new `run-<UTC stamp>` folder under `base`; never reuses one (a suffix separates two in a second)."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    folder = base / f"run-{stamp}"
    suffix = 2
    while folder.exists():
        folder = base / f"run-{stamp}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    return folder


def _bench_library(source: Path, dest: Path) -> None:
    """Copy the skill library to `dest` for one bench: its runs use the fake arm, so the `real` counts they
    move must never reach the library the dashboard and the story numbers read. `bench_run` has already
    refused a `dest` inside `source` (the copy would recurse into itself)."""
    if not source.is_dir():
        dest.mkdir(parents=True)  # the starters are seeded from the repo's skills/_starter
        return
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))


def bench_run(
    text: str = "hand me my pills",
    runs: int = 5,
    *,
    live_router: bool = False,
    timeout: float = BENCH_WAIT_S,
) -> int:
    """C8's saved-skill latency: `runs` commands of `text` through the TestClient, each timed from the
    POST to its `executing` event, the median of the runs that reached `executing` against
    BENCH_TARGET_S, and `<runs_dir>/bench/run-<UTC stamp>/summary.json` (a new folder per bench) with the
    phrase, the fakes, the mode, every run's seconds and end stage, the median and the verdict.

    FAKE_ROBOT, FAKE_CAMERA and FAKE_SANDBOX are forced on, so the arm, the cameras and Daytona are never
    touched; FAKE_LLM is forced on too unless `live_router`, which instead clears FAKE and FAKE_LLM for this
    process (the real xAI router) and returns 2 before anything starts when XAI_API_KEY is absent (only its
    presence is ever reported). A command the router sends to a learn instead of a saved skill is stopped
    at `spec_ready` and ends the bench. The bench runs on a copy of the library (VARA_SKILLS_DIR or
    skills/) in `run-<stamp>/skills/`, with its task folders in `run-<stamp>/t-*`: fake-arm runs never
    move the library's `real` counts or mix with the real tasks. Returns 0 iff every run reached
    `executing` and the median is within the target, 1 otherwise; a blank `text`, `runs` < 1 or a runs
    folder inside the skills folder is a ValueError before anything is changed or created.

    Its last two lines, which scripts/live_gates.py parses for the C8 gate, are
    `median <s> s (<n> of <runs> runs reached executing; target 5 s): PASS|FAIL` (`no median` when no run
    reached `executing`) and `wrote <summary.json path>`."""
    global orch
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    if not text.strip():
        raise ValueError("the bench needs a phrase")
    source_skills = _env_path("VARA_SKILLS_DIR", ROOT / "skills")
    bench_dir = _env_path("VARA_RUNS_DIR", ROOT / "runs") / "bench"
    if source_skills.is_dir() and bench_dir.resolve().is_relative_to(source_skills.resolve()):
        raise ValueError(f"the bench folder {bench_dir} must not be inside the skills folder {source_skills}")
    key_present = bool(os.environ.get("XAI_API_KEY", "").strip())
    if live_router:
        if not key_present:
            print(
                "--live-router needs XAI_API_KEY in .env (present: False); nothing was started",
                file=sys.stderr,
            )
            return 2
        for name in ("FAKE", "FAKE_LLM"):
            os.environ.pop(name, None)
    else:
        os.environ["FAKE_LLM"] = "1"
    for name in ("FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX"):
        os.environ[name] = "1"
    from fastapi.testclient import TestClient

    out_dir = _bench_out_dir(bench_dir)
    _bench_library(source_skills, out_dir / "skills")
    os.environ["VARA_SKILLS_DIR"] = str(out_dir / "skills")
    os.environ["VARA_RUNS_DIR"] = str(out_dir)  # the bench's t-* folders sit beside its summary
    orch = Orchestrator()
    fakes = {name: _fake(name) for name in ("FAKE_LLM", "FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX")}
    router_label = "fixtures" if fakes["FAKE_LLM"] else "xAI"
    print(
        f"bench-run {text!r} x{runs}: router={router_label} robot=fake camera=fake sandbox=fake "
        f"mode={orch.mode} (target {BENCH_TARGET_S:g} s)"
    )
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with TestClient(app) as client:
        records = _bench_commands(client, text, runs, timeout, reset=False, saved_skill=True)
    # Only a run that reached `executing` has a command-to-arm time; the others count against the verdict.
    seconds = [record["seconds"] for record in records if record["stage"] == "executing"]
    executed = len(seconds)
    median_s = round(statistics.median(seconds), 3) if seconds else None
    all_executing = executed == runs
    ok = all_executing and median_s is not None and median_s <= BENCH_TARGET_S
    summary = {
        "phrase": text,
        "runs_requested": runs,
        "router": router_label,
        "live_router": live_router,
        "xai_key_present": key_present,
        "fakes": fakes,
        "mode": orch.mode,
        "skills_source": str(source_skills),
        "skills_dir": str(orch.skills_dir),
        "target_s": BENCH_TARGET_S,
        "runs": records,
        "executing_runs": executed,
        "median_s": median_s,
        "all_executing": all_executing,
        "ok": ok,
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    median_text = "no median" if median_s is None else f"median {median_s:.1f} s"
    print(  # the format scripts/live_gates.py parses (docstring)
        f"{median_text} ({executed} of {runs} runs reached executing; target {BENCH_TARGET_S:g} s): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(f"wrote {path}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.orchestrator", description="VARA orchestrator")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--real", action="store_true", help="connect the real arm (asks for confirmation)")
    parser.add_argument("--bench-demo", action="store_true", help="five demo-mode learns with every fake")
    parser.add_argument(
        "--bench-run",
        metavar="TEXT",
        help=(
            'benchmark a saved skill, e.g. "hand me my pills": seconds from the command to the arm '
            "moving per run, the median against the C8 target of 5 s, summary.json under runs/bench/"
        ),
    )
    parser.add_argument("--runs", type=int, default=5, help="runs for --bench-demo and --bench-run")
    parser.add_argument(
        "--live-router",
        action="store_true",
        help=(
            "--bench-run with the live xAI router (needs XAI_API_KEY); the arm, the cameras and the "
            "sandboxes stay fake"
        ),
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.bench_demo and args.bench_run is not None:
        parser.error("--bench-demo and --bench-run are exclusive")
    if args.live_router and args.bench_run is None:
        parser.error("--live-router requires --bench-run")
    if args.real and (args.bench_demo or args.bench_run is not None):
        parser.error("the benchmarks never use the real arm: drop --real")
    if args.bench_run is not None and not args.bench_run.strip():
        parser.error("--bench-run needs a phrase")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.bench_demo:
        return bench_demo(runs=args.runs)
    if args.bench_run is not None:
        return bench_run(args.bench_run.strip(), args.runs, live_router=args.live_router)
    watcher = None
    driver = None
    cli_args = list(sys.argv[1:] if argv is None else argv)
    # A13: set once the connect starts; an Esc before that (the countdown) aborts and is not recorded
    console: dict[str, Any] = {"connecting": False, "connected_at": None}

    def on_esc() -> None:
        """The console Esc: STOP first, then (real arm only) the Esc record for this press."""
        orch.stop("Esc pressed", source="keyboard")
        if console["connecting"] and getattr(driver, "is_fake", True) is False:
            driver_mod.report_esc_latency(
                watcher,
                driver,
                source="orchestrator-console",
                connected_at=console["connected_at"],
                extra={"argv": cli_args},
            )

    try:
        if args.real:
            driver = driver_mod.get_driver()
            # The lifespan has not started yet: STOP must already reach the driver during connect().
            orch.driver = driver
            watcher = driver_mod.EscWatcher(on_esc)
            watcher.start()
            if getattr(driver, "is_fake", False):
                print("FAKE_ROBOT is set: --real connects the in-memory arm only.")
            elif not real_robot.confirm_arm_clear(3, watcher):
                return 1
            if watcher.pressed or driver.stopped or real_robot.cancelled() or orch.stop_event.is_set():
                print("startup stopped; nothing connected")
                return 1
            console["connecting"] = True
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                try:
                    driver.connect()
                    break
                except ConnectionError as exc:
                    # one dropped status packet during LeRobot's motor setup aborts the whole connect (its
                    # writes do not retry); a failed connect already put the arm back to holding
                    if attempt == CONNECT_ATTEMPTS or watcher.pressed or driver.stopped:
                        raise
                    print(f"connect failed ({exc}); retrying in {CONNECT_RETRY_S:.0f} s")
                    time.sleep(CONNECT_RETRY_S)
            console["connected_at"] = time.monotonic()
            if watcher.pressed or driver.stopped or real_robot.cancelled() or orch.stop_event.is_set():
                orch.stop("stopped while connecting")
                print("startup stopped; arm frozen with torque on")
                return 1
            print("arm connected; Esc in this console or STOP on the dashboard freezes it")
        elif not _fake("FAKE_ROBOT"):
            print(
                "the arm stays disconnected (start with --real to use it); "
                "tasks that would move it end as error"
            )
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if watcher is not None:
            watcher.stop()
        # The lifespan normally disconnects. Also cover an abort or exception before it starts.
        if driver is not None and getattr(driver, "connected", False):
            driver.disconnect()  # torque stays on
    return 0


if __name__ == "__main__":
    sys.exit(main())
