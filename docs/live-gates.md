# Live gates: the key-dependent checks, in order, with evidence

The fixture-backed test suite proves the software; it never talks to xAI or Daytona. The gates below
need the real services and real keys, cost money or sandbox credits, and leave evidence under
`runs/live-gates/<UTC stamp>/`. One command runs them in the right order within the sandbox caps and
checks that no sandbox is left behind:

```powershell
.venv\Scripts\python.exe scripts\live_gates.py --dry-run     # the plan, key presence as booleans, nothing runs
.venv\Scripts\python.exe scripts\live_gates.py               # all ten steps, stops at the first failure
.venv\Scripts\python.exe scripts\live_gates.py --continue    # run everything even after a failure
.venv\Scripts\python.exe scripts\live_gates.py --only ping,router          # just the model gates
.venv\Scripts\python.exe scripts\live_gates.py --only smoke                # smoke plus its two inventories
.venv\Scripts\python.exe scripts\live_gates.py --skip-daytona --timeout candidates=1200
```

The script is a runner only: every step is a child `.venv` Python process of a documented command
(`agents.llm`, `agents.router`, `agents.candidates`, `agents.sandbox_pool`, `scripts/daytona_smoke.py`,
`agents.orchestrator --bench-run`, and its own `--voice-probe` mode for the token). It never passes
`--real`, never opens a port, a camera or the arm (the orchestrator children run with `FAKE_ROBOT`,
`FAKE_CAMERA` and `FAKE_SANDBOX` set to 1), and only `agents/sandbox_pool.py` and
`scripts/daytona_smoke.py` create sandboxes.

## Prerequisites

1. `.env` holds `XAI_API_KEY` and `DAYTONA_API_KEY`. The script refuses (exit 2, nothing run or written)
   when either is missing, and it reports only `true`/`false` for their presence. Each child gets only the
   key its step uses (the other is set to an empty value, which the child's own `load_dotenv` keeps), and
   every captured line is scrubbed of both values before it is written.
2. `FAKE`, `FAKE_LLM` and `FAKE_SANDBOX` are not `1`, in the shell or in `.env`. The gates have no fake path
   by design; the script refuses with exit 2 otherwise. A variable set in the shell wins over `.env`, so
   set it to `0` there or remove it from `.env`. (`FAKE_ROBOT` and `FAKE_CAMERA` may stay on.)
3. The sandbox image exists on the Daytona account. Once per account, before the first Daytona step:
   `.venv\Scripts\python.exe -m agents.sandbox_pool --build-snapshot arm-sim-v1` (slow; it builds debian
   slim + Python 3.12 + mujoco, numpy, pyyaml, jsonschema). Our code travels as a bundle at pool start,
   so code changes never need a new image.
4. **Stop the dashboard** (`uvicorn agents.orchestrator:app`) and any other VARA process first. With a
   Daytona key the orchestrator holds 7 sandboxes, the cap, and the bench step needs all 7 for itself. The
   first Daytona step, `inventory-before`, checks the account holds no `vara=pool` sandbox; if it finds
   one, smoke and bench are skipped (also under `--continue`) and nothing is cleaned up, because those
   sandboxes may belong to a running dashboard. As a second guard `SandboxPool.start()` lists the account
   and refuses (nothing created) when the existing `vara=pool` sandboxes plus the request would exceed 7.
5. `agents.candidates` takes `--skills none` (it does now): the live router then sees an empty skill list,
   so "put the juice box in the tray" is a `learn` even though `skills/place_in_tray.json` exists. The
   script asks `agents.candidates --help` and adds the flag when it is listed; on an older checkout it runs
   without it, says so in the step's note, and the step may fail with `not a learn request`.

## The steps, their cost and what passes

| # | Step | Command (child of `.venv` Python) | Passes when | Paid calls / sandboxes |
| --- | --- | --- | --- | --- |
| 1 | `ping` | `-m agents.llm --ping` | exit 0 and both `router (...)` and `codegen (...)` lines report under 15 000 ms (C1) | 2 model calls |
| 2 | `router` | `-m agents.router --eval` | `20/20 phrases routed as expected`, 20 being the size of `tests/fixtures/router_phrases.json` (C2) | 20 to 40 model calls |
| 3 | `candidates` | `-m agents.candidates --eval "put the juice box in the tray" --task-dir <out>/candidates --skills none` | exit 0 and at least 5 of 6 pass `check_source` on the first try (C3) | 7 to 13 model calls |
| 4 | `inventory-before` | `-m agents.sandbox_pool --list --label pool --wait 60 --out <out>/sandboxes` | `remaining (pool): 0`; never cleans up | 0 (a listing) |
| 5 | `smoke` | `scripts/daytona_smoke.py --out <out>/daytona-smoke --max-seconds 60` | exit 0, runner `daytona`, a schema-valid score line, total (create + score + delete) within 60 s, our sandbox deleted (C5) | 1 sandbox |
| 6 | `inventory-after-smoke` | as step 4 | `remaining (pool): 0`; otherwise, after a clean step 4 only, `--cleanup --label pool` runs at once and the step still fails | 0 |
| 7 | `bench` | `-m agents.sandbox_pool --bench --out <out>/bench` | exit 0, `pool: SandboxPool`, three `task ...` lines each within 35 s and scored by runner `daytona` (no local fallback) (C6) | 7 sandboxes, the cap |
| 8 | `inventory-after-bench` | as step 4 | as step 6 | 0 |
| 9 | `bench-run` | `-m agents.orchestrator --bench-run "hand me my pills" --runs 5 --live-router` with fake arm, cameras and sandboxes and `VARA_RUNS_DIR=<out>/bench-run` | `router=xAI`, 5 of 5 runs reach `executing`, median within 5 s (C8) | 5 router calls |
| 10 | `voice` | `scripts/live_gates.py --voice-probe` with `FAKE_ROBOT=FAKE_CAMERA=FAKE_SANDBOX=1`, `FAKE=FAKE_LLM=0`, its own `VARA_SKILLS_DIR`/`VARA_RUNS_DIR` | the orchestrator app, started in the child with `TestClient`, answers `POST /voice/token` with 200 and a non-empty token from xAI, not the fake one (A14) | 1 token request |

The voice probe prints one JSON line, `{status, model, expires_at, token_length, fake_llm, fake_token,
detail}`: the token's length and expiry, never its value, which is not printed, kept or written.

Sandbox counts: **smoke 1, bench 7, never together.** The steps run strictly one after another inside the
script, so two Daytona steps cannot overlap; the inventories prove the account is back to 0 after each
one. Per account the hard caps are 7 for scoring and planning plus 4 for RL (`rl/train_parallel.py`).
Selecting `smoke` or `bench` always brings `inventory-before` and the inventory after it; skipping one of
those is refused (exit 2).

Without `--continue` the script stops at the first failed step, but the inventory that follows a Daytona
step which ran always runs, also after a failure, a timeout or Ctrl+C. A step that exceeds its timeout
(`--timeout STEP=SECONDS` overrides the defaults that `--help` prints) is killed and counts as failed; a
killed smoke or bench cannot delete its own sandboxes, so the inventory after it finds them and its
cleanup deletes them. Exit codes: 0 when every selected step passed, 1 otherwise, 2 for a refusal or a bad
flag.

## Where the evidence lands

```text
runs/live-gates/<UTC stamp>/       (under VARA_RUNS_DIR when it is set; --out picks a new or empty folder)
  summary.json                     state, plan, steps (pass/fail and the parsed numbers per step),
                                   passed, failed_steps, skipped_steps, keys_present (booleans),
                                   fake_flags, clean_start, sandbox_caps (smoke 1, bench 7, never together)
  <step>.json                      argv (never the environment), exit_code, seconds, timed_out, timeout_s,
                                   parsed, pass, note, and for an inventory its cleanup
  <step>.stdout.log, <step>.stderr.log   the child's output, key values replaced by [XAI_API_KEY] etc.
  <step>-cleanup.*.log             only when an inventory found stragglers after a clean start
  candidates/                      the C3 eval's prompts, replies, cand_*.py, problems and summary.json
  sandboxes/inventory-<stamp>.json one listing per inventory step (cleanup-<stamp>.json after a cleanup)
  daytona-smoke/                   the C5 task dir: cand_1.py, task.json, scene.json, the cand_1/ outputs
                                   downloaded from the sandbox, and result.json (see below)
  bench/                           the C6 bench outputs per task and candidate
  bench-run/bench/run-<stamp>/     the C8 bench's library copy, its t-* task folders and summary.json
  voice/                           the probe's own skills and runs folders (no token inside)
```

`daytona-smoke/result.json` holds `snapshot`, `started_at`, `task_dir`, `trials`, `max_seconds`, `runner`,
`score` (the validated score line), `seconds_create`, `seconds_score`, `seconds_total`, `under_60` (the C5
verdict), `under_max`, `sandbox_id`, `sandbox_deleted`, `remaining_sandboxes`, `error`, `passed` and
`exit_code`. Run on its own, `scripts/daytona_smoke.py` writes the same file under
`runs/daytona-smoke/<UTC stamp>/` and never deletes that folder.

Reading it: `summary.json` -> `passed` is the overall verdict; `steps[*].pass` and `steps[*].parsed`
hold the numbers the task Done-whens ask for (`models.router.ms`, `routed`/`phrases`, `first_try`,
`seconds_total`, `tasks[*].seconds`, `remaining`, `median_s`, `token_length`). A skipped step has
`pass: null` and a `reason`. The verification record quotes those numbers and the folder path; it never
quotes a token or a key.

## Cleanup by hand

```powershell
.venv\Scripts\python.exe -m agents.sandbox_pool --list                 # table + runs/sandboxes/inventory-<stamp>.json; exit 1 while pool sandboxes remain
.venv\Scripts\python.exe -m agents.sandbox_pool --cleanup              # delete vara=pool stragglers, then list into cleanup-<stamp>.json
.venv\Scripts\python.exe -m agents.sandbox_pool --list --label all     # verdict also over vara=rl and vara=rl-check (RL training)
.venv\Scripts\python.exe -m agents.sandbox_pool --cleanup --label rl   # only when no rl/train_parallel.py run is going
```

Both commands only ever touch sandboxes carrying our `vara` label: the server filters by label and the
code checks the label again on every result, so an unlabelled sandbox is never listed or deleted. The
pool labels its sandboxes `vara=pool`; `rl/train_parallel.py` labels its own `vara=rl` (training) and
`vara=rl-check` (its image check). `--list` always prints the counts for every value; `--label` picks the
values the verdict and `--cleanup` cover (default `pool`). Without `DAYTONA_API_KEY`, or under
`FAKE`/`FAKE_SANDBOX=1`, both refuse with exit 2 before building a client and print only whether the key
is present; a listing failure exits 1 and never claims 0.
