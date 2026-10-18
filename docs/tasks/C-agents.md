# Lane C tasks: router, candidates, Daytona pool, orchestrator

**Continuation verification:** [2026-10-17 evidence](../verification-2026-10-17.md). Checked boxes have supporting evidence; hardware and live-service acceptance remains open.

Lane C owns the learning loop and the glue: route a request and write its success test, generate six candidates, score them in parallel Daytona sandboxes, hand the winner's path to Lane A, and save what the camera confirms. Build every step against the fakes first; Daytona replaces the local runner once scoring works.

**Progress** (in time order; C sleeps 7:30 to 11:30 AM Saturday)

- [x] S0 · Scaffold with fakes (in Shared tasks) · Sat 1:30 AM
- [ ] C1 · Model client · Sat 2:00 AM
- [ ] C2 · Router and success spec · Sat 3:00 AM (M1a)
- [ ] C5 · Daytona smoke test and snapshot · Sat 3:00 AM (M1a)
- [ ] C3 · Candidate prompt and strategy hints · Sat 4:00 AM
- [x] C4 · Runner on MuJoCo · Sat 5:00 AM
- [ ] C9 · Real execution and saving · Sat 6:00 AM (M1)
- [ ] C8 · Planner for saved skills · Sat 12:30 PM
- [ ] C6 · Sandbox pool · Sat 1:30 PM
- [x] C7 · Selection, repair and budgets · Sat 2:30 PM (local scoring and fixture-model acceptance)
- [x] C11 · Orchestrator hardening · Sat 3:00 PM
- [x] C10 · Routines · Sat 5:00 PM (fake definition + replay verified; real integration remains I3)

## C1 · Model client (`agents/llm.py`)

**Needs:** S0. **By:** Sat 2:00 AM. **Files:** `agents/llm.py`, `tests/test_llm_fake.py`.

xAI's API is OpenAI-compatible, so use the `openai` package:

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
resp = client.chat.completions.create(model=os.environ["CODEGEN_MODEL"], messages=msgs, temperature=0.7)
```

- `chat(messages, model, schema=None, timeout=30)`: with `schema`, request structured JSON output if the model supports it, otherwise parse the first `{...}`; validate the result either way.
- Retry 3 times with exponential backoff on 429 and 5xx, and log tokens per task to `runs/<task_id>/llm.jsonl`.
- `FAKE_LLM=1` answers from `tests/fixtures/llm/`.
- `ROUTER_MODEL` (a fast model) routes; `CODEGEN_MODEL` (the strongest your key lists in the xAI console) writes code. Both live in `.env`, so they change without code edits.

**Done when:** `pytest -q tests/test_llm_fake.py` passes and `python -m agents.llm --ping` gets an answer from both models in under 15 s.

## C2 · Router and success spec (`agents/router.py`)

**Needs:** C1. **By:** Sat 3:00 AM. **Files:** `agents/router.py`, `agents/prompts/router.md`, `tests/fixtures/router_phrases.json`.

- One `ROUTER_MODEL` call with the request, the skill list (name, English description, params) and the scene's object, obstacle and zone names. It returns one object from Contracts, Router output, validated against `schemas/router.json`.
- For `learn`, the prompt teaches the predicate vocabulary with three worked specs (hand over, put in the tray, clear the table) and says: if success can't be written with these predicates, return `queue`.
- A spec that `evaluate` finds already true on the current scene is rejected with "That already looks done."
- The same call maps spoken names to scene names ("my pills" to `pill_bottle`).
- Offline insurance: when the API is unreachable, a keyword router maps the five demo phrases to fixed actions.

**Done when:** `python -m agents.router --eval` routes all 20 phrases in `tests/fixtures/router_phrases.json` correctly with the live model, including 5 that should be `queue`, and every `learn` spec validates.

## C3 · Candidate prompt and strategy hints (`agents/prompts/write_skill.md`)

**Needs:** C1, C2. **By:** Sat 4:00 AM. **Files:** `agents/prompts/write_skill.md`, `agents/candidates.py` (generation).

The prompt holds the Robot API and its notes from Contracts verbatim, the candidate rules, the task spec (English and JSON), the current scene JSON with obstacles and hand, the two starter skills as worked examples, and one strategy hint per candidate, so the six differ by design rather than by temperature:

1. Nearest object first, default parameters.
2. Farthest object first.
3. Set `transit_z` to the tallest obstacle plus 0.08 m on every pick and place.
4. Route around obstacles: rise, move out to x = 0.32 m, then sideways.
5. Slow down to 0.08 m/s and use pitch -π/4 for targets beyond 0.30 m from the base.
6. Fill tray slots far side first, and retry a failed grasp once.

End with "Return one Python code block and nothing else." Generate all six in parallel with `CODEGEN_MODEL`, run `check_source` on each, and regenerate a rejected one once with its reasons.

**Done when:** `python -m agents.candidates --eval "put the juice box in the tray"` shows at least 5 of 6 passing `check_source` on the first try.

**Tools added 2026-10-17:** For the live run use `.\.venv\Scripts\python.exe -m agents.candidates --eval "put the juice box in the tray" --skills none --task-dir runs/c3-<stamp>`. `--skills none` hides the saved skills from the router and the prompt, so the phrase stays a learn even though a saved skill covers it, and `--skills starter` shows the starters only. `--spec FILE` generates from a task spec JSON, checked against `schemas/task_spec.json`, without routing. Both forms write `<task_dir>/summary.json` with the model, the timings and one row per candidate (its first-try and final `check_source` results). The task dir must be new or empty (default `runs/eval-<spec name>-<UTC stamp>`), and the exit code is 0 when at least 5 of 6 pass on the first try. The keyed run is the `candidates` step of `scripts/live_gates.py`, described under C5 and in [live gates](../live-gates.md).

## C4 · Runner on MuJoCo (`sim/run_candidate.py`, `sim/scoring.py`)

**Needs:** A8. **By:** Sat 5:00 AM. **Files:** `sim/run_candidate.py`, `sim/scoring.py`, `tests/test_scoring.py`.

Switch S0's runner from the kinematic stand-in to A8's MuJoCo `SimRobot`, keeping the command line:

```bash
python -m sim.run_candidate --candidate runs/t-0042/cand_3.py --task runs/t-0042/task.json \
  --scene runs/t-0042/scene.json --trials 5 --seed 7 --out runs/t-0042/cand_3
```

It re-runs `check_source`, executes the candidate with only `math` and the robot in its namespace, applies the trials and pass rule from Contracts with a 60 s wall-clock limit per trial, writes `traj.json` and `final_scene.json`, and prints one score line.

**Done when:** `pytest -q tests/test_scoring.py` passes: both starter skills pass, the drop-beside-the-tray candidate fails with a reason, a candidate whose trial 1 fails never passes, and an infinite loop ends as an error within 60 s.

## C5 · Daytona smoke test and snapshot

**Needs:** S0. **By:** Sat 3:00 AM. **Files:** `scripts/daytona_smoke.py`, `agents/sandbox_pool.py` (snapshot builder).

```python
from daytona import Daytona, CreateSnapshotParams, Image

daytona = Daytona()                                   # reads DAYTONA_API_KEY
image = Image.debian_slim("3.12").pip_install(["mujoco", "numpy", "pyyaml", "jsonschema"])
daytona.snapshot.create(CreateSnapshotParams(name="arm-sim-v1", image=image), on_logs=print)
```

Check class and method names against the [Daytona Python SDK](https://pypi.org/project/daytona/); it changes often. Keep this image small, since Lane D builds a separate RL image with PyTorch. Our code travels as a tarball uploaded when the pool starts (C6), so code changes don't need a new snapshot. The smoke script creates one sandbox from `arm-sim-v1`, uploads the bundle, scores one candidate, prints the score line and deletes the sandbox in a `finally`.

**Done when:** `python scripts/daytona_smoke.py` prints a valid score line in under 60 s, and the Daytona dashboard shows no sandbox left running.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe scripts/daytona_smoke.py` (add `--build-snapshot` the first time) keeps its evidence in a new `runs/daytona-smoke/<UTC stamp>/` (or `--out DIR`): the candidate's outputs and `result.json` with the runner, the score line, the seconds and `under_60`, which is true when create, score and delete took 60 s or less (`--max-seconds` sets the exit-code limit). The sandbox is still deleted in a `finally`. `.\.venv\Scripts\python.exe -m agents.sandbox_pool --list` stands in for the dashboard check: it prints every `vara`-labelled sandbox and `remaining (pool): N`, writes `runs/sandboxes/inventory-<UTC stamp>.json`, and exits 0 only when none remain. `.\.venv\Scripts\python.exe scripts/live_gates.py` runs the keyed checks in order as child processes (model ping, router, candidates, this smoke test, the C6 bench, the C8 bench-run and the voice token, with a sandbox inventory before and after each Daytona step) and writes every log plus `summary.json` to `runs/live-gates/<UTC stamp>/`. It refuses (exit 2) when a fake flag is on or either key is missing, and `--dry-run` prints the plan without running anything. Order: Part 6 of the [operator runbook](../runbook.md#part-6-live-services-c1-to-c8-a14-costs-model-calls-and-sandboxes).

## C6 · Sandbox pool (`agents/sandbox_pool.py`)

**Needs:** C4, C5. **By:** Sat 1:30 PM. **Files:** `agents/sandbox_pool.py`, `tests/test_pool_fake.py`.

- At orchestrator start, create 7 sandboxes in parallel threads (6 for scoring, 1 for planning) with `daytona.create(CreateSandboxFromSnapshotParams(snapshot="arm-sim-v1", auto_stop_interval=0))`, so Daytona's idle auto-stop can't shut them down mid-demo.
- Upload the code bundle (`arm/`, `sim/`, `schemas/`, `config/`) as one tarball with `sandbox.fs.upload_file(data, path)` and extract it; re-upload when its hash changes.
- Per task: upload each candidate, `task.json`, `scene.json` and the current `skills/*.json` (routines call `run_skill`). Run `sandbox.process.exec("python -m sim.run_candidate ...", timeout=60)` on all six at once, parse the last stdout line, emit `candidate_scored` as each lands, and download only the winner's and runner-up's `traj.json` and `final_scene.json`.
- Delete every sandbox on shutdown, with atexit and SIGINT/SIGTERM handlers. Never create more than 7.
- Fallback: if a create or exec fails twice, run the same command as a local subprocess whose environment holds only `PATH` and `PYTHONPATH`, and label the card "local".
- Check the organization's vCPU quota in the Daytona dashboard before Lane D's RL run, and ask at the Daytona workshop if it's too low.

**Done when:** `python -m agents.sandbox_pool --bench` finishes 6 candidates × 5 trials in under 35 s on each of 3 tasks, `pytest -q tests/test_pool_fake.py` passes, and shutdown leaves 0 sandboxes.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m agents.sandbox_pool --bench` now ends with `remaining: N`, the account's `vara=pool` sandboxes after shutdown, and writes under `runs/bench` unless `--out DIR` says otherwise. `.\.venv\Scripts\python.exe -m agents.sandbox_pool --list --label pool --wait 30` writes `runs/sandboxes/inventory-<UTC stamp>.json` and exits 0 only when none in scope remain; `--cleanup` deletes the sandboxes carrying the label (stragglers after a crash), then lists into `cleanup-<stamp>.json`. Never clean `--label rl` while a training runs. Both refuse (exit 2) without `DAYTONA_API_KEY` or under `FAKE` or `FAKE_SANDBOX`. `SandboxPool.start()` now lists the account's pool sandboxes before creating any, so the 7-sandbox cap holds across processes whenever that listing succeeds. The keyed bench is the `bench` step of `scripts/live_gates.py`, which takes an inventory before and after it and runs `--cleanup` on a failure.

## C7 · Selection, repair and budgets

**Needs:** C6. **By:** Sat 2:30 PM (M2). **Files:** `agents/orchestrator.py`, `agents/candidates.py`.

Rank passing candidates per Contracts. In normal mode, if none pass, send the model the three best failures (code, score line, `fail_reason`) with "fix these failures", get six new candidates and score again; if none pass after that, the task becomes `queue`. With `DEMO_MODE=1`, enforce the budget table exactly: no repair round, the 75 s cap from command to arm moving, and `queued` on expiry.

**Done when:** the repair round fixes at least one test task that failed the first time, and `python -m agents.orchestrator --bench-demo` shows 5 of 5 demo-mode runs reaching the arm, or `queued`, within 75 s.

**Verified 2026-10-17:** an actual local MuJoCo first round rejected all six colliding programs;
the repair pipeline produced six passing programs and saved the learned skill. Five isolated
clear-table demo runs executed and saved within 8.6–11.8 s. See the verification record for artifacts.
These checks use fixture model replies and a fake arm; C1/C5/C6 live-service gates remain open.

## C8 · Planner for saved skills (`agents/planner.py`)

**Needs:** C4, C6. **By:** Sat 12:30 PM. **Files:** `agents/planner.py`.

`plan(skill, args, scene) -> (trajectory, final_scene)` runs the skill once (trial 1 only) through `sim.run_candidate`. Trusted starter skills run in a local subprocess; learned skills run in the planning sandbox, falling back to a local subprocess labeled "local".

**Done when:** "hand me my pills" goes from command to the arm moving in under 5 s (median of 5 runs) with the live router.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m agents.orchestrator --bench-run "hand me my pills" --runs 5 --live-router` times each run from the command to its first `executing`, prints the median against the 5 s target and writes `runs/bench/run-<UTC stamp>/summary.json`, with its own copy of the library and its `t-*` task folders beside it, so the library's real counts never change. `--live-router` uses the xAI router while the arm, the cameras and the sandboxes stay fake, and exits 2 before anything starts when `XAI_API_KEY` is empty; without it every fake is forced. The exit code is 0 only when every run reached `executing` and the median is at most 5 s. The all-fake run on 2026-10-17 reached `executing` in 5 of 5 runs with a median of 1.2 s; the live-router run is the `bench-run` step of `scripts/live_gates.py` and is still to be filed.

## C9 · Real execution and saving

**Needs:** A9, A10, B7, C4. **By:** Sat 6:00 AM (M1). **Files:** `agents/orchestrator.py`, `agents/library.py`.

Under the arm lock: B7's before-snapshot at planning time, `validate`, `replay`, home, B7's after-snapshot, then `evaluate`. On a pass, write `skills/<name>.json` with the spec, code, and sim and real stats, and emit `saved`. On a fail, emit `not_saved`, re-plan the next-best passing candidate on the current scene (one trial through `sim.run_candidate`), replay it once, then give up. Every task writes its prompts, candidates, scores, trajectories, snapshots and events to `runs/<task_id>/`, which Lane D mines for the Devpost.

**Done when:** I3 passes.

## C10 · Routines

**Needs:** C7. **By:** Sat 5:00 PM. **Files:** `agents/orchestrator.py`, `agents/library.py`.

A `define_routine` result becomes a candidate whose `run()` calls `robot.run_skill(...)` for each step. It is scored against a spec that is `all` of the steps' specs, run on the real arm, and saved under its name.

**Done when:** "When I say good morning, hand me my pill bottle, then my juice box" saves, and "Good morning" replays it.

## C11 · Orchestrator hardening (`agents/orchestrator.py`)

**Needs:** C9. **By:** Sat 3:00 PM. **Files:** `agents/orchestrator.py`, `tests/test_orchestrator.py`.

Every endpoint in Contracts, with one `asyncio.Lock` for the arm, the stop flag shared with Lane A's driver, per-stage timeouts and both budget modes. `/stop` reaches `Driver.freeze()` within 0.2 s from any stage. `/voice/token` mints a short-lived xAI token on the server, following the xAI cookbook's web voice example. Run it with `uvicorn agents.orchestrator:app --port 8000`.

**Done when:** `pytest -q tests/test_orchestrator.py` passes: a second command mid-task gets 409, `/stop` cancels every stage, `/reset_demo` leaves exactly the two starter skills, and `/voice/token` never returns the API key.
