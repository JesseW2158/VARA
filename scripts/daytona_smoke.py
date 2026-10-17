"""Daytona smoke test (task C5): one sandbox from `arm-sim-v1`, the code bundle, one scored candidate.

    python scripts/daytona_smoke.py [--snapshot arm-sim-v1] [--build-snapshot] [--trials 5] [--timeout 60]

Needs DAYTONA_API_KEY (from `.env`); exits 2 without it and never runs in tests. It creates ONE sandbox
through `agents.sandbox_pool.SandboxPool` (`start(n_score=1, n_plan=0)`), which uploads and extracts the
bundle, uploads `tests/fixtures/candidate_clear_table.py`, `tests/fixtures/task_spec_clear_table.json` and
`config/fake_scene.json`, runs `python -m sim.run_candidate ...` in the sandbox, prints the score line
(schemas/score.json) and deletes the sandbox in a `finally` (atexit and SIGINT/SIGTERM handlers cover a
crash). Exit 0 when the line came from the sandbox (`runner: "daytona"`), 1 when the pool had to fall back
to the local runner or the sandbox could not be created (the Daytona side is then broken even though a score
line may be printed). `--build-snapshot` builds the image first (slow, once per account).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from agents.sandbox_pool import SNAPSHOT, SandboxPool, build_snapshot  # noqa: E402

CANDIDATE = ROOT / "tests" / "fixtures" / "candidate_clear_table.py"
TASK_SPEC = ROOT / "tests" / "fixtures" / "task_spec_clear_table.json"
SCENE = ROOT / "config" / "fake_scene.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default=SNAPSHOT, help="sandbox snapshot name (default arm-sim-v1)")
    parser.add_argument("--build-snapshot", action="store_true", help="build the snapshot first")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0, help="wall-clock seconds per trial")
    parser.add_argument("--out", default=None, help="task dir (default: a temporary directory)")
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    if not os.environ.get("DAYTONA_API_KEY"):
        print("DAYTONA_API_KEY is not set (put it in .env); nothing was created", file=sys.stderr)
        return 2
    if ns.build_snapshot:
        print(f"building snapshot {ns.snapshot} ...", file=sys.stderr)
        build_snapshot(ns.snapshot)

    temp = None
    if ns.out is None:
        temp = tempfile.mkdtemp(prefix="daytona-smoke-")
    task_dir = pathlib.Path(ns.out or temp).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    candidate = task_dir / "cand_1.py"
    shutil.copy(CANDIDATE, candidate)
    shutil.copy(TASK_SPEC, task_dir / "task.json")
    shutil.copy(SCENE, task_dir / "scene.json")

    pool = SandboxPool(snapshot=ns.snapshot)
    code = 1
    t0 = time.perf_counter()
    try:
        live = pool.start(n_score=1, n_plan=0)
        print(f"sandbox created and bundle installed in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
        if live != 1:
            print("the sandbox could not be created (see the log above)", file=sys.stderr)
            return 1
        t1 = time.perf_counter()
        scored = pool.score(
            task_dir,
            [(1, candidate)],
            task_dir / "task.json",
            task_dir / "scene.json",
            None,
            trials=ns.trials,
            seed=7,
            timeout=ns.timeout,
        )[0]
        print(f"scored in {time.perf_counter() - t1:.1f} s by runner {scored['runner']!r}", file=sys.stderr)
        print(json.dumps(scored["score"]), flush=True)
        if scored["runner"] == "daytona":
            code = 0
        else:
            print("the score came from the local fallback, not from Daytona", file=sys.stderr)
    finally:
        pool.shutdown()
        print(f"sandbox deleted; total {time.perf_counter() - t0:.1f} s", file=sys.stderr)
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
