# Shared tasks

**Continuation verification:** [2026-10-17 evidence](../verification-2026-10-17.md). Checked boxes have supporting evidence; hardware and live-service acceptance remains open.

These tasks make the whole pipeline runnable before any hardware works, so four Claude Code sessions build against something real from the first hour. S0 and I1 are tonight's critical path.

**Progress**

- [x] S0 · Scaffold with fakes · Lane C · Sat 1:30 AM
- [ ] I1 · Fake end-to-end on every laptop · all · Sat 2:00 AM
- [ ] I2 · Arm meets camera · A and B · Sat 5:30 AM
- [ ] I3 · First real learn · all · Sat 6:00 AM (M1)

## S0 · Scaffold with fakes

**Owner:** C. **Needs:** the kit committed to `main`. **By:** Sat 1:30 AM. **Files:** every module in the repo map, `scripts/e2e_fake.py`, `tests/test_scaffold.py`.

1. Create every module in the repo map with the public functions that Contracts and the task files name. Where the real thing needs hardware, network or another lane's work, put a fake behind the same signature, switched by the `FAKE_*` flags.
2. Implement for real the three pure modules that `tests/contract/` already specifies: `check_source` in `agents/candidates.py`, `evaluate` in `sim/success.py`, and `validate` in `arm/replay.py`.
3. `arm/sim_robot.py`: a kinematic `SimRobot` with the full Robot API and no MuJoCo. Objects move with the gripper, nothing collides, and it records valid trajectories (joint interpolation within 2° per tick).
4. `sim/run_candidate.py` and `sim/scoring.py` working end to end on that kinematic sim, with the outputs in Contracts, Scoring. It accepts either a candidate file plus a task spec, or a skill file plus its args (A12 uses the second form). The candidate runs in a child process with a clean environment.
5. `agents/orchestrator.py` with every endpoint and the event stream from Contracts; `FAKE=1` runs fully offline. `skills/_starter/` gets placeholder starter skills that A12 replaces.
6. `web/index.html`: a minimal page with the command box, the event list, six candidate cards and a STOP button. B9 builds the real one.
7. `scripts/e2e_fake.py "<text>"`: forces `FAKE=1`, sends the command through FastAPI's `TestClient`, prints each stage, and exits 0 only if the last stage is `saved` or `done`.

**Done when:** `pytest -q` passes with nothing skipped, including all of `tests/contract/`, and `python scripts/e2e_fake.py "put the juice box in the tray"` prints stages ending in `saved`.

**Out of scope:** MuJoCo, Daytona, LeRobot, live model calls, dashboard styling.

## I1 · Fake end-to-end on every laptop

**Owner:** all. **Needs:** S0. **By:** Sat 2:00 AM.

Each teammate pulls `main`, creates their lane branch and runs `/verify`. From here on, each lane replaces one fake at a time behind the same contract and keeps `pytest -q` green.

**Done when:** `/verify` passes on all four laptops, and with `FAKE=1` the dashboard at `http://localhost:8000` shows six cards after typing "put the juice box in the tray".

## I2 · Arm meets camera

**Owner:** A and B. **Needs:** A6, A9, B6. **By:** Sat 5:30 AM. **Files:** `scripts/touch_markers.py`.

Lay 5 loose object markers on the table. The script reads their positions from the scene state, plans a move to 5 cm above each in simulation, validates and replays it on the real arm, and prompts you to measure the gap between the tip and the marker center.

**Done when:** `python scripts/touch_markers.py --real` lands within 1.5 cm of all 5 markers. If it doesn't, fix the `config/joint_map.yaml` offsets (A6) or the homography (B5) before anything else.

## I3 · First real learn

**Owner:** all. **Needs:** A10, A12, B7, C9, I2. **By:** Sat 6:00 AM (M1).

Type "put the juice box in the tray" with the live model and `FAKE_SANDBOX=1` (local scoring). The router writes the spec, candidates pass in simulation, the real arm replays the winner, the camera verifies, and the skill is saved.

**Done when:** a new `skills/<name>.json` exists with `real.successes` equal to 1, and `runs/<task_id>/events.jsonl` ends with `saved`.
