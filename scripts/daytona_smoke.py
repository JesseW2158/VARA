"""Daytona smoke test (task C5): one sandbox from `arm-sim-v1`, the code bundle, one scored candidate.

    python scripts/daytona_smoke.py [--snapshot arm-sim-v1] [--build-snapshot] [--trials 5] [--timeout 60]
                                    [--max-seconds 60] [--out DIR]

Needs DAYTONA_API_KEY (from `.env`); it refuses with exit 2, before creating anything, without the key, under
FAKE=1 or FAKE_SANDBOX=1, or when `--out` already holds a `result.json`, and it never runs in tests. It
creates ONE sandbox through `agents.sandbox_pool.SandboxPool` (`start(n_score=1, n_plan=0)`), which uploads
and extracts the bundle; uploads `tests/fixtures/candidate_clear_table.py`,
`tests/fixtures/task_spec_clear_table.json` and `config/fake_scene.json`; runs `python -m sim.run_candidate
...` in the sandbox; prints the score line (schemas/score.json) as the LAST stdout line; and deletes the
sandbox in a `finally` (atexit and SIGINT/SIGTERM handlers cover a crash). Progress, the `scored in X s by
runner '...'` and `sandbox deleted; total Y s` lines and the `result:` line go to stderr.

The task dir is kept, never deleted: by default a new `runs/daytona-smoke/<UTC stamp>/` (under VARA_RUNS_DIR
when set; a `-2` suffix separates two runs in one second), or the folder `--out` names. It holds the inputs,
the sandbox's `cand_1/` outputs (traj.json, score.json, trials.json...) and `result.json`, written once:

    {"snapshot", "started_at", "task_dir", "trials", "max_seconds",
     "runner": "daytona" | "local" | null, "score": <the validated score line> | null,
     "seconds_create", "seconds_score", "seconds_total" (create + score + delete: the printed total),
     "under_60": seconds_total <= 60 (the C5 verdict), "under_max": seconds_total <= max_seconds,
     "sandbox_id": str | null, "sandbox_deleted": the delete of our sandbox took (it left the account's
     vara=pool listing, or is still `destroying` after a 30 s wait; null when no sandbox was created or the
     listing failed), "remaining_sandboxes": vara=pool sandboxes still present after the shutdown,
     destroying ones included (null when the listing failed), "error": null | text, "passed", "exit_code"}

Exit 0 only when the score came from the sandbox (`runner: "daytona"`), `seconds_total` is within
`--max-seconds` and the delete of our sandbox took (an unknowable listing does not fail it); 1 otherwise
(local fallback, no sandbox, over time, a straggler, or an exception, which is recorded in `error`; Ctrl+C is
recorded as `interrupted (Ctrl+C)` and result.json written before it propagates); 2 for a refusal or a bad
flag. `--build-snapshot` builds the image first (slow, once per account). The verdict
helpers `under_limit(seconds, limit)`, `smoke_passed(record)` and `write_result(out_dir, record)` are plain
functions, testable without a pool; `_clock` is the timer the run uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from agents import sandbox_pool as sp  # noqa: E402
from agents.sandbox_pool import SNAPSHOT, SandboxPool, build_snapshot  # noqa: E402

CANDIDATE = ROOT / "tests" / "fixtures" / "candidate_clear_table.py"
TASK_SPEC = ROOT / "tests" / "fixtures" / "task_spec_clear_table.json"
SCENE = ROOT / "config" / "fake_scene.json"
RESULT_NAME = "result.json"
C5_LIMIT_S = 60.0  # task C5: a valid score line in under 60 s
DEFAULT_MAX_SECONDS = C5_LIMIT_S
DEFAULT_TRIALS = 5
RESULT_KEYS = (
    "snapshot",
    "started_at",
    "task_dir",
    "trials",
    "max_seconds",
    "runner",
    "score",
    "seconds_create",
    "seconds_score",
    "seconds_total",
    "under_60",
    "under_max",
    "sandbox_id",
    "sandbox_deleted",
    "remaining_sandboxes",
    "error",
    "passed",
    "exit_code",
)

_clock = time.perf_counter  # the run's timer (tests replace it)
log = logging.getLogger("daytona_smoke")


def default_out_dir() -> pathlib.Path:
    """A path that does not exist yet: `<runs>/daytona-smoke/<UTC stamp>[-n]/` (runs: VARA_RUNS_DIR or the
    repo's runs/)."""
    base = sp.runs_dir() / "daytona-smoke"
    stamp = sp.utc_stamp()
    path, n = base / stamp, 2
    while path.exists():
        path, n = base / f"{stamp}-{n}", n + 1
    return path


def under_limit(seconds_total: float | None, limit: float) -> bool:
    """A measured total within `limit` seconds (an unmeasured run never passes)."""
    return seconds_total is not None and float(seconds_total) <= float(limit)


def smoke_passed(record: dict) -> bool:
    """The exit-0 rule: the sandbox scored it (a score line, runner "daytona", no error), within the record's
    `max_seconds` (default 60), and our sandbox is gone (`sandbox_deleted` True, or None when unknowable)."""
    return (
        record.get("runner") == sp.RUNNER_LABEL_DAYTONA
        and isinstance(record.get("score"), dict)
        and under_limit(record.get("seconds_total"), record.get("max_seconds", DEFAULT_MAX_SECONDS))
        and record.get("sandbox_deleted") is not False
        and not record.get("error")
    )


def write_result(out_dir: str | os.PathLike[str], record: dict) -> pathlib.Path:
    """Write `<out_dir>/result.json` as a new file (FileExistsError when one is there; parents are created)
    and return its path. `under_60`, `under_max` and `passed` are computed here from the record, so the file
    and the verdict always agree; every key of RESULT_KEYS is present (None when unknown)."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = dict(record)
    data.setdefault("max_seconds", DEFAULT_MAX_SECONDS)
    data["under_60"] = under_limit(data.get("seconds_total"), C5_LIMIT_S)
    data["under_max"] = under_limit(data.get("seconds_total"), data["max_seconds"])
    data["passed"] = smoke_passed(data)
    for key in RESULT_KEYS:
        data.setdefault(key, None)
    path = out / RESULT_NAME
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return path


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds, got {text}")
    return value


def _trials(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def _after_shutdown(pool: SandboxPool, ids: list[str], record: dict) -> None:
    """Fill `sandbox_deleted` and `remaining_sandboxes` from the account's listing (left None on failure)."""
    try:
        after = sp.inventory_after_shutdown(pool.client, ids)
    except Exception as exc:  # noqa: BLE001 - not even a client: the listing is unknowable, not a crash
        log.warning("could not list the account's sandboxes: %s", type(exc).__name__)
        after = None
    if after is not None:
        # deleted: the delete took (gone, or still `destroying` after the wait); a live one is a straggler
        record["sandbox_deleted"] = not after.get("ours_active", []) if ids else None
        record["remaining_sandboxes"] = int(after["remaining"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/daytona_smoke.py",
        description=__doc__.split("\n\n")[0],
        epilog="Evidence: result.json in the task dir (see the module docstring for its keys).",
    )
    parser.add_argument("--snapshot", default=SNAPSHOT, help="sandbox snapshot name (default arm-sim-v1)")
    parser.add_argument("--build-snapshot", action="store_true", help="build the snapshot first (slow)")
    parser.add_argument("--trials", type=_trials, default=DEFAULT_TRIALS, help="trials (default 5)")
    parser.add_argument("--timeout", type=_positive, default=60.0, help="wall-clock seconds per trial")
    parser.add_argument(
        "--max-seconds",
        type=_positive,
        default=DEFAULT_MAX_SECONDS,
        help="the limit on the total (create + score + delete) for exit 0 (default 60, the C5 criterion)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="task dir, kept (default runs/daytona-smoke/<UTC stamp>/; refused when it holds a result.json)",
    )
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    if os.environ.get("FAKE") == "1" or os.environ.get("FAKE_SANDBOX") == "1":
        print(
            "FAKE or FAKE_SANDBOX is 1: the smoke test needs the live Daytona API; nothing was created",
            file=sys.stderr,
        )
        return 2
    if not os.environ.get("DAYTONA_API_KEY", "").strip():
        print("DAYTONA_API_KEY is not set (put it in .env); nothing was created", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(ns.out).resolve() if ns.out else default_out_dir().resolve()
    if (out_dir / RESULT_NAME).exists():
        print(f"{out_dir} already holds {RESULT_NAME}; choose a new --out", file=sys.stderr)
        return 2
    if ns.build_snapshot:
        print(f"building snapshot {ns.snapshot} ...", file=sys.stderr)
        build_snapshot(ns.snapshot)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / "cand_1.py"
    shutil.copy(CANDIDATE, candidate)
    shutil.copy(TASK_SPEC, out_dir / "task.json")
    shutil.copy(SCENE, out_dir / "scene.json")
    record: dict = {
        "snapshot": ns.snapshot,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_dir": str(out_dir),
        "trials": int(ns.trials),
        "max_seconds": float(ns.max_seconds),
        "runner": None,
        "score": None,
        "seconds_create": None,
        "seconds_score": None,
        "seconds_total": None,
        "sandbox_id": None,
        "sandbox_deleted": None,
        "remaining_sandboxes": None,
        "error": None,
    }

    pool = SandboxPool(snapshot=ns.snapshot)
    t0 = _clock()
    try:
        live = pool.start(n_score=1, n_plan=0)
        record["seconds_create"] = round(_clock() - t0, 3)
        ids = pool.sandbox_ids
        record["sandbox_id"] = ids[0] if ids else None
        print(f"sandbox created and bundle installed in {record['seconds_create']:.1f} s", file=sys.stderr)
        if live != 1:
            record["error"] = "the sandbox could not be created (see the log above)"
            print(record["error"], file=sys.stderr)
        else:
            t1 = _clock()
            scored = pool.score(
                out_dir,
                [(1, candidate)],
                out_dir / "task.json",
                out_dir / "scene.json",
                None,
                trials=ns.trials,
                seed=7,
                timeout=ns.timeout,
            )[0]
            record["seconds_score"] = round(_clock() - t1, 3)
            record["runner"] = scored["runner"]
            record["score"] = scored["score"]
            print(
                f"scored in {record['seconds_score']:.1f} s by runner {scored['runner']!r}", file=sys.stderr
            )
            if scored["runner"] != sp.RUNNER_LABEL_DAYTONA:
                print("the score came from the local fallback, not from Daytona", file=sys.stderr)
    except KeyboardInterrupt:
        record["error"] = "interrupted (Ctrl+C)"  # the pool's SIGINT handler already deleted the sandbox
        raise
    except Exception as exc:  # noqa: BLE001 - recorded in result.json; the exit code says it failed
        record["error"] = sp._scrub_key(f"{type(exc).__name__}: {exc}")  # never the key, even if quoted
        log.error("smoke test failed: %s", record["error"])
    finally:
        # after a signal the pool's handler has already emptied it: still check the sandbox we recorded
        ids = pool.sandbox_ids or ([record["sandbox_id"]] if record["sandbox_id"] else [])
        record["sandbox_id"] = record["sandbox_id"] or (ids[0] if ids else None)
        pool.shutdown()
        record["seconds_total"] = round(_clock() - t0, 3)
        print(f"sandbox deleted; total {record['seconds_total']:.1f} s", file=sys.stderr)
        _after_shutdown(pool, ids, record)
        record["exit_code"] = code = 0 if smoke_passed(record) else 1
        path = write_result(out_dir, record)
        print(
            f"result: {path} (runner {record['runner']!r}, total {record['seconds_total']} s, "
            f"under {ns.max_seconds:g} s: {under_limit(record['seconds_total'], ns.max_seconds)}, "
            f"remaining sandboxes: {record['remaining_sandboxes']})",
            file=sys.stderr,
        )
    if record["score"] is not None:
        print(json.dumps(record["score"]), flush=True)  # the last stdout line, as before
    return code


if __name__ == "__main__":
    sys.exit(main())
