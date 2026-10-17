"""Fake end-to-end run of the orchestrator (S0 item 7): one command through FastAPI's TestClient.

    python scripts/e2e_fake.py "put the juice box in the tray" [--in-place] [--reset] [--timeout 240]
    python scripts/e2e_fake.py --reset            # only restore the two starter skills in skills/

`FAKE=1` is forced before the orchestrator is imported (DEMO_MODE is left as the environment has it), so no
model, sandbox, arm or camera is touched. The script POSTs `/command`, follows `runs/<task_id>/events.jsonl`
and prints one `stage  summary` line per event until an end stage (`saved`, `queued`, `done`, `stopped`,
`error`; `not_saved` is printed and followed, since the orchestrator then re-plans once). It exits 0 only
when the last stage is `saved`, or `done` with `ok` not false. For "put the juice box in the tray" the
expected trace is routing -> spec_ready -> writing_code x6 -> simulating -> candidate_scored x6 -> selected
-> validating -> executing -> verifying -> saved, and the learned `juice_box_to_tray.json` is printed.

The library is a copy by default: a fake-learned skill in the real skills/ would make the next live
"put the juice box in the tray" replay it instead of learning on stage, and tests/test_scaffold.py and
tests/test_orchestrator.py fail while it is there. With a command, skills/ is copied to
`runs/e2e_fake/<UTC stamp>/skills` (runs/ is gitignored) and the run uses the copy. `--in-place` uses skills/
itself; `--reset` then POSTs `/reset_demo`, which archives the learned skill and restores exactly the two
starters. `--reset` without a command always works on skills/. An explicit `VARA_SKILLS_DIR` (the scaffold
test's copies) wins over both; `VARA_RUNS_DIR` moves the run folders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["FAKE"] = "1"  # before the orchestrator (and every module behind it) is imported

END_STAGES = ("saved", "queued", "done", "stopped", "error")
DEFAULT_TEXT = "put the juice box in the tray"


def summarize(stage: str, detail: Mapping[str, Any]) -> str:
    """One short line per event for the console."""
    d = detail
    if stage == "routing":
        return json.dumps(d.get("text", ""))
    if stage == "spec_ready":
        return f"{d.get('name', '?')}: {d.get('english', '')}"
    if stage in ("writing_code", "candidate_rejected"):
        text = f"candidate {d.get('candidate', '?')}"
        if d.get("hint"):
            text += f": {d['hint']}"
        if d.get("problems"):
            text += f" rejected: {d['problems'][0]}"
        return text
    if stage == "simulating":
        return f"candidates {d.get('candidates', [])}, {d.get('trials', '?')} trials each"
    if stage == "candidate_scored":
        verdict = "PASS" if d.get("passed") else "FAIL"
        text = (
            f"candidate {d.get('candidate', '?')} {verdict} {d.get('successes', '?')}/{d.get('trials', '?')}"
        )
        if d.get("fail_reason"):
            text += f": {d['fail_reason']}"
        return text + f" [{d.get('runner', '?')}]"
    if stage == "selected":
        return f"candidate {d.get('candidate', '?')} ({d.get('hint', '')})"
    if stage == "planning":
        return f"{d.get('skill', d.get('candidate', '?'))} {json.dumps(d.get('args', {}))}"
    if stage in ("validating", "executing"):
        return f"{d.get('segments', '?')} segments"
    if stage == "verifying":
        return str(d.get("english", ""))
    if stage == "saved":
        return str(d.get("name", ""))
    if stage in ("not_saved", "stopped", "error", "queued"):
        return str(d.get("reason") or d.get("error") or d.get("task") or "")
    if stage == "done":
        return "ok" if d.get("ok", True) else "failed"
    return json.dumps(d)[:120]


def read_events(path: Path, start: int) -> tuple[list[dict[str, Any]], int]:
    """Events appended to `path` since byte offset `start`, and the new offset."""
    if not path.is_file():
        return [], start
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    events: list[dict[str, Any]] = []
    consumed = 0
    for line in data.split(b"\n"):
        if not line.strip():
            consumed += len(line) + 1
            continue
        try:
            events.append(json.loads(line.decode("utf-8")))
        except ValueError:
            break  # a partial line: read it again next time
        consumed += len(line) + 1
    return events, start + min(consumed, len(data))


def follow(client: Any, runs_dir: Path, task_id: str, timeout: float) -> str | None:
    """Print the task's stages as they land; returns the last stage (None when nothing ended in time)."""
    path = runs_dir / task_id / "events.jsonl"
    offset = 0
    deadline = time.monotonic() + timeout
    last: str | None = None
    while time.monotonic() < deadline:
        events, offset = read_events(path, offset)
        for event in events:
            if event.get("task_id") != task_id:
                continue
            stage = str(event.get("stage"))
            print(f"{stage:<19} {summarize(stage, event.get('detail') or {})}", flush=True)
            last = stage
            if stage in END_STAGES:
                return stage
        time.sleep(0.1)
    state = client.get("/state").json()
    print(f"timed out after {timeout:g} s (state {state.get('state')}, last stage {last})", flush=True)
    return None


def copy_library() -> Path:
    """skills/ copied to runs/e2e_fake/<UTC stamp>/skills, so a fake learn never lands in the real library."""
    runs = Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dest = runs / "e2e_fake" / stamp / "skills"
    n = 1
    while dest.exists():  # two runs in the same second
        n += 1
        dest = runs / "e2e_fake" / f"{stamp}-{n}" / "skills"
    shutil.copytree(ROOT / "skills", dest)
    print(f"skills library: a copy in {dest} (--in-place uses skills/)", flush=True)
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fake end-to-end run through the orchestrator")
    parser.add_argument("text", nargs="?", default=None, help=f'the command (default: "{DEFAULT_TEXT}")')
    parser.add_argument("--reset", action="store_true", help="POST /reset_demo afterwards (or alone)")
    parser.add_argument(
        "--in-place", action="store_true", help="learn into skills/ itself instead of a copy under runs/"
    )
    parser.add_argument("--timeout", type=float, default=240.0, help="seconds to wait for an end stage")
    args = parser.parse_args(argv)
    if args.text is None and not args.reset:
        args.text = DEFAULT_TEXT

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # the hints hold a pi; a cp1252 console must not crash the run
        reconfigure(encoding="utf-8", errors="replace")

    if args.text is not None and not args.in_place and not os.environ.get("VARA_SKILLS_DIR"):
        os.environ["VARA_SKILLS_DIR"] = str(copy_library())  # before the orchestrator reads it

    from fastapi.testclient import TestClient

    from agents import orchestrator

    code = 0
    with TestClient(orchestrator.app) as client:
        orch = orchestrator.orch
        if args.text is not None:
            reply = client.post("/command", json={"text": args.text})
            if reply.status_code != 200:
                print(f"POST /command returned HTTP {reply.status_code}: {reply.text}")
                return 1
            task_id = reply.json()["task_id"]
            print(f"task {task_id}: {args.text!r} (mode {orch.mode}, runs in {orch.runs_dir / task_id})")
            last = follow(client, orch.runs_dir, task_id, args.timeout)
            if last == "saved":
                event = next(e for e in reversed(orch.recent_events()) if e["stage"] == "saved")
                print(f"saved skill: {orch.skills_dir / (event['detail'].get('name', '?') + '.json')}")
            if last == "done":
                event = next(e for e in reversed(orch.recent_events()) if e["stage"] == "done")
                code = 0 if event["detail"].get("ok", True) is not False else 1
            else:
                code = 0 if last == "saved" else 1
            deadline = time.monotonic() + 10.0
            while orch.state == "busy" and time.monotonic() < deadline:
                time.sleep(0.05)
        if args.reset:
            reply = client.post("/reset_demo")
            if reply.status_code != 200:
                print(f"POST /reset_demo returned HTTP {reply.status_code}: {reply.text}")
                return 1
            print("reset demo, live skills: " + ", ".join(reply.json().get("skills", [])))
    return code


if __name__ == "__main__":
    sys.exit(main())
