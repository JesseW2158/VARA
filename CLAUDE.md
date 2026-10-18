# Teachable home-assistant arm: project brief

Claude Code loads this file into every session, so it holds only what Claude can't infer from the code: the goal, commands, hard rules and gotchas. Keep it under 200 lines.

## What we are building

A voice-taught SO-101 arm for people with limited mobility (HackHarvard 2026). Saved skills run at once. A new task gets a success test written first, then 6 AI-written candidate programs, each scored over 5 randomized MuJoCo trials in its own Daytona sandbox. The arm replays the winner's checked joint path, the overhead camera verifies with the same test, and the skill is saved. A separate RL track (PPO, Grok-written rewards) trains the hand-off reach. Devpost closes 8:00 AM Sunday; feature freeze is 1:00 AM Sunday.

## Read before coding

- `docs/interfaces.md`: the contracts and the source of truth, mirrored in `schemas/*.json` and `tests/contract/`.
- `docs/tasks/<lane>.md`: your task. Run one with `/task <ID>`.
- `docs/safety.md`: limits the code must enforce.

## Commands

- Setup: `python3.12 -m venv .venv`, activate it, then `pip install -r requirements.txt` (Windows demo laptop: see Demo laptop below)
- Fast checks, no hardware or network: `pytest -q`
- Fake end-to-end: `python scripts/e2e_fake.py "put the juice box in the tray"`
- Orchestrator: `uvicorn agents.orchestrator:app --port 8000` (set `FAKE=1` in `.env` to run without hardware)
- Score one candidate locally: `python -m sim.run_candidate --candidate cand.py --task task.json --scene config/fake_scene.json --trials 5 --seed 7 --out runs/check`
- Lint: `ruff check .` and `ruff format .`

## Hard rules

1. AI-written code (candidates, saved code skills, reward functions) runs only inside `sim.run_candidate` in a Daytona sandbox, or in the local fallback subprocess whose environment holds only `PATH` and `PYTHONPATH`. Never import, `exec` or `eval` it in the orchestrator process, and never give it the real arm.
2. The real arm only replays a trajectory that `arm.replay.validate()` accepted, or runs our live primitives in `arm/real_robot.py` (`carry_to_hand`, `home`, `freeze`).
3. Meters, radians and seconds everywhere. Degrees and LeRobot's 0 to 100 gripper scale exist only inside `arm/driver.py`. Our gripper value runs from 0.0 (closed) to 1.0 (open).
4. STOP freezes the arm in place with torque on and cancels the task. Torque off is a separate maintenance action; the 12 V plug is the hard kill.
5. Anything that moves the real arm needs the `--real` flag. Ask the human and wait until they confirm the arm is clear before running it.
6. Keys come from `.env` through `python-dotenv`. Never print, log, commit or send `XAI_API_KEY` or `DAYTONA_API_KEY` to the browser; the voice page gets a short-lived token from `POST /voice/token`.
7. Tests never call xAI, Daytona or the arm. Use the fakes: `FAKE_LLM`, `FAKE_SANDBOX`, `FAKE_ROBOT`, `FAKE_CAMERA`, or `FAKE=1` for all four.
8. Daytona sandboxes cost credits: at most 7 for scoring and planning plus 4 for RL. Only `agents/sandbox_pool.py`, `rl/train_parallel.py` and `scripts/daytona_smoke.py` create them, and each deletes its own on shutdown.
9. Don't change `docs/interfaces.md`, `schemas/` or `tests/contract/` unless the human says the team agreed. If a contract looks wrong, stop and say so.

## Repo map

```text
arm/      driver.py (LeRobot wrapper, limits, freeze) | ik.py | sim_robot.py (Robot API in MuJoCo) | replay.py (validate + replay) | real_robot.py (live primitives)
vision/   cameras.py | markers.py | calibrate.py | scene.py | verify.py
agents/   llm.py | prompts/ | router.py | candidates.py | planner.py | library.py | sandbox_pool.py | orchestrator.py
sim/      model/ (SO-101 MJCF + scene_ours.xml) | run_candidate.py | scoring.py | success.py
rl/       handoff_env.py | train.py | train_parallel.py | rewards/ | evaluate.py | policy_runner.py | policies/
web/      index.html (dashboard) | voice.html | static/
config/   objects.yaml | scene.yaml | joint_map.yaml | limits.yaml | cameras.yaml | homography.npy | fake_scene.json
skills/   <name>.json saved skills | _starter/ | _backup/
schemas/  JSON Schema for every contract
tests/    contract/ (shared; ask before editing) | fixtures/ | per-lane tests
scripts/  e2e_fake.py | daytona_smoke.py | make_markers.py | touch_markers.py
runs/     one folder per task (gitignored)
```

## Conventions

- Python 3.12 with type hints on public functions. JSON that crosses a process or network boundary is validated with `jsonschema` against `schemas/`.
- Each file has one owning lane (see `docs/tasks/`). Use other lanes' code only through its public functions.
- Every task writes `runs/<task_id>/`: prompts, raw model replies, candidates, scores, trajectories, snapshots and `events.jsonl`.
- One commit per task, message `<ID>: <title>`. Branches `lane-a` to `lane-d`; merge to `main` only when `pytest -q` passes.
- When compacting, keep the current task ID, the files changed and the Done-when command.

## Gotchas

- LeRobot's import path changed: `lerobot.robots.so101_follower` in older releases, `lerobot.robots.so_follower` in newer ones. The demo laptop runs 0.6.2: `lerobot.robots.so_follower` (`SOFollower`, `SOFollowerRobotConfig`) and `lerobot.teleoperators.so_leader` (`SOLeader`, `SOLeaderTeleopConfig`), so the A4 snippet's import fails there.
- On Linux, serial ports swap on replug; re-run `lerobot-find-port`. Windows ties each COM number to the board's serial, but check after any rewiring anyway.
- Stalled STS3215 servos overheat. Hold the gripper at the measured contact reading minus 3 on the 0 to 100 scale.
- The Daytona SDK renames things between versions. Check `pip show daytona` and the SDK docs before writing pool code.
- On Windows, open cameras with `cv2.VideoCapture(i, cv2.CAP_DSHOW)` and activate the venv with `.venv\Scripts\activate`.
- Two OpenCV packages side by side break `cv2`. Keep exactly one; ArUco is in the main package from 4.7.

## Arm hazards (measured on our SO-101)

These bind anyone writing `arm/driver.py`. The A4 snippet handles none of the first three.

- Enabling torque drives each servo straight to its stored `Goal_Position`, and LeRobot's `connect()` (through `configure()`) turns torque off and on without syncing it. Stale targets were measured up to 168° away. Write `Goal_Position = Present_Position` before every torque enable. `max_relative_target` doesn't help: it only limits `send_action`.
- Writing `Goal_Position` switches torque on. After `torque_off()`, block every write until an explicit re-enable that syncs the goal first.
- LeRobot's disconnect turns torque off by default (`disable_torque_on_disconnect=True`), so exit, Ctrl+C or an exception drops the arm and anything it holds. Set it to False and turn torque off only on purpose, at the rest pose. STOP never disconnects.
- LeRobot's 0° is the midpoint of each joint's calibrated range, not the homing pose. `config/joint_map.yaml` and `config/limits.yaml` hold only for the current calibration; redo A6 after any recalibration. "All joints at 0" is mid-range everywhere, a large swing from rest.

## Demo laptop (Windows 11, arms attached)

Verified 2026-10-17. Full notes: `C:\Users\jesse\lerobot-tools\HANDOFF.md`, outside the repo.

- Follower on COM8 with 12 V; leader on COM7 with 5 V, never 12 V (its motor ID 2 has an 8 V limit). Before anything connects to the follower, run `arm_check.py` (read-only), then `sync_goals.py COM8` (powers the arm and holds its pose), both from `C:\Users\jesse\lerobot-tools\` in the `lerobot` env.
- Calibration ids are `my_awesome_follower_arm` and `my_awesome_leader_arm`, not `follower` and `leader` as in A3 and A4. Read the id from config; don't recalibrate just to rename it.
- LeRobot 0.6.2 (Seeed fork) is installed editable from `C:\Users\jesse\lerobot` in the conda env `lerobot` (Python 3.12, under `C:\Users\jesse\miniforge3`, not on PATH). There is no `python3.12` on PATH. The project `.venv` is also verified: Python 3.12.15, the same editable LeRobot 0.6.2, and no pip module. Run `.venv/Scripts/python.exe`; install with `uv pip install --python .venv/Scripts/python.exe ...`. Use `requirements-rl.txt` for PPO and browser-compatible video encoding.
- The SenseCraft Robotics app also opens COM7 and COM8, and only one program can hold a port. Close it first.
- LeRobot's interactive CLIs (calibrate, record) read the physical keyboard, so run them in a visible console, never in the background.
- DSHOW camera indices shift on replug or reboot (2026-10-18: overhead 0, built-in webcam 1, wrist 2). Run `python -m vision.cameras --list`, look at the saved frames, then set `OVERHEAD_CAM` and `WRIST_CAM` in `.env`.
- No NVIDIA GPU: PPO trains on the CPU or in Daytona sandboxes.
