# VARA implementation handoff

## Active camera follow-up (2026-10-17)

The user asked to continue and update this handoff throughout the work. Starting from HEAD e460eb7;
only the user/external `.env.example` camera-index edit was pending. The previous 1,406-test full gate
remains the baseline; do not rerun it unless a new change warrants it.

Root is investigating the camera-2 timing failure with bounded, non-motion diagnostics. New worker
`/root/camera_review` is reviewing DirectShow/OpenCV configuration semantics and reader behavior
without hardware access. Prior workers exhausted their usage allowances; this new attempt has not
reported a result yet. Table preparation and API credentials remain external prerequisites; preserve
`.env` and all physical configuration measurements. No robot movement is planned without calibration.

**Confirmed physical configuration bug:** on both cameras, requesting FOURCC first returned MJPG,
but the later height setting silently switched it to YUY2; FPS left it there. Every property setter
reported success. Evidence: `runs/setup/camera-format-20261017/before.json`. OpenCV 4.13 DirectShow
source confirms later geometry/FPS changes can recreate the stream without the requested format.
The camera worker has changed the order to FPS, width, height, FOURCC and added format readback logs;
focused tests are running. Root is now comparing physical format/timing after this fix.

Worker `/root/table_capture` is adding a read-only observed-pixel capture helper and a five-spot
measurement workflow to `docs/table-calibration.md`; it opens no devices during development/tests.
Neither worker edits `.env`, saved homography, joint maps or protected contracts. New changes in this
follow-up are uncommitted until checks pass.

## Current completion checkpoint (2026-10-17, Codex)

**Read this before the historical handoffs.** The user requested continued parallel work and an
immediate durable handoff. The remaining software changes have passed the full gate and were saved
in eleven task-sized local commits; a final B1 documentation commit records the camera failure. No remote push.

**Final software gate: 1,406 tests passed, no skips, in 310.96 seconds.** Ruff check passes; format
check reports 90 files already formatted. Whitespace/LF checks pass. `docs/interfaces.md`, schemas,
contract tests, synthetic homography and selected policy ZIP were not changed. Final fake juice-box
learn reached `saved`. Full evidence: [verification record](docs/verification-2026-10-17.md).

### Physical camera stability check: failed acceptance

- The attempted 30-minute check already accumulated 27 late-frame gaps on each feed by 185 seconds
  (5,240 frames each, zero failed reads). It cannot pass B1's zero-drop criterion, so it was stopped
  early after 193.77 wall seconds. It did **not** complete the requested 1,800 seconds.
- Evidence: `runs/setup/camera-soak-20261017/camera-check.log` and `result.json` (forced-stop exit
  4294967295). Unified exec session 59704 is complete. No camera-soak child remains from that run.
- The independent-reader diagnostic completed (session 66460): in 60 seconds camera 2 delivered
  1,013 frames with 172 late gaps and maximum read time 0.375 s; wrist camera 1 delivered 1,802 frames,
  about 30 fps, zero late gaps. Neither had failed reads. Artifacts:
  `runs/setup/camera-soak-20261017/independent-readers.json` and `independent-readers.log`.
  Both cameras were released. This isolates irregular timing to camera 2 during that run; its cause
  (USB/exposure/driver/host load/ongoing setup) is not established. No process from this pass remains running.
- The earlier five-second precursor passed: 148 frames per feed, about 29 fps, zero failures/late
  frames. Camera 2 now opens, superseding its initial open failure.
- Current captures in `runs/setup/cameras/current/` show camera 2 looking across the table from a
  low position and a mostly dark/occluded wrist view. Neither detected markers. The user replied
  **"Still preparing the table"**. No physical table calibration has been performed.

### Implemented and reviewed in this pass

- Fresh per-motor goal synchronization immediately before each configuration torque enable, partial
  connection cleanup, STOP preservation; all new driver tests use mocks, not motor writes.
- Offline A6 five-pose checker with raw/normalized encoder consistency and calibration/config/model
  hashes. The template `runs/calibration/measurements-template.json` deliberately contains nulls.
- Simulator preserves unseen flags/stale poses and collision bodies; incomplete carry clearance,
  rise, travel or descent aborts before release. Starters and two-handoff routine pass 5/5 each.
- Disabled PPO runtime uses replay safety for Cartesian speed, workspace and STOP; preserves grip
  and wrist; rejects expired commands. Policy-to-IK pacing regression reproduced and fixed.
- Calibration requires >=4 fitting markers, <=1 cm mean/<=1.5 cm max error; supports >=5 independent
  measured points; rejects fake cameras in live modes; writes accepted matrices atomically.
- Camera checker rejects incomplete/interrupted/missing-feed/invalid-duration/empty captures and
  releases captures on failures. Its focused camera/vision subset passed 83 tests.
- Genuine C7 repair integration turns six colliding first-round programs into passing repairs and a
  saved skill. Five clear-table rehearsals all saved, reaching execution in 8.6-11.8 seconds.
- Current xAI defaults: grok-4.3/none for routing and grok-4.7/low for generation, configurable effort,
  visible-output caps and truncated-reply rejection. Official documentation checked; 187 focused
  adapter/router/candidate tests pass. No paid xAI or Daytona calls have been made.
- Setup diagnostic opens no devices/services and emits only key-presence booleans. `/runtime` labels
  robot/camera/model/scoring provenance. Edge accepted a full clear-table learn, six cards, live
  events, H.264 video and Esc stopping the fake arm.
- README, pitch, demo guide and Devpost draft are ready with measured local results. Final recording
  and publication remain undone. Marker PDF: `runs/setup/markers/markers-a4.pdf` (13 markers, two A4 pages).

### External prerequisites and next actions

1. B1 remains open: inspect camera-2 timing after mounting/wiring/lighting is stable, then rerun
   `.venv/Scripts/python.exe -m vision.cameras --check 1800`. Do not count the interrupted run as passed.
2. The user is preparing the table. Mount camera 2 overhead, clear the wrist view, print markers at
   actual size, and measure camera Hc/c and marker 0-7 centers in `config/scene.yaml`.
   Follow [table calibration](docs/table-calibration.md), including five independent check points.
3. Reconcile leader COM7's saved/hardware encoder calibration mismatch before teleoperation. Follower
   COM8 matched its saved encoder file, but the joint map/limits still need A6 measurements. Follow
   [joint calibration](docs/joint-calibration.md) and [hardware notes](docs/hardware-calibration.md).
   Physical gripper, STOP, replay and I2/I3 acceptance remain open. `use_policy: false` stays in place.
4. Gitignored `.env` exists with the real arm ports and blank API keys for the user to fill. Last
   boolean-only check: XAI_API_KEY and DAYTONA_API_KEY absent. Never print the file/values. Once keys
   exist, run the documented live model, Daytona smoke/pool and voice checks within sandbox caps,
   cleaning up all created cloud resources. Fixtures do not satisfy those live gates.
5. `.env` and `.env.example` now select OVERHEAD_CAM=2 through a user/external change. Preserve it.
   The example's camera-index change is intentionally left unstaged/uncommitted; our model-options
   hunk was committed separately. All our code/documentation changes are committed; only the unowned example-camera edit remains.

All three worker agents have now reached usage limits. Their earlier completed work is included.
The last requested additional mapping review did not receive a final agent report; root inspected
its implementation/tests. No active worker should be assumed to remain. Parallel tools can still
run independent checks in the main session. Previous full-suite session 2459 completed successfully;
its log is `runs/verification/final-completion/20261017-160904/pytest.log`.

Use `.venv/Scripts/python.exe`; the venv has no pip (use uv). All serial/LeRobot access stays inside
`arm/driver.py`. No physical motion, goal, torque or calibration writes occurred in this pass.
Task-observer observations live at `C:/Users/jesse/.codex/skill-observations` outside this repository.

### Local implementation commits

- `c5670c0` D8: document measured results and prepare the submission story
- `0079715` C7: verify failed-round repair and successful demo budgets
- `81b8e54` C1: select current Grok models with bounded request options
- `7cb23d3` S0: add a non-invasive local setup diagnostic
- `ecf41f5` B9: label runtime provenance and distinguish marker consistency checks
- `106e2b1` B1: reject incomplete camera checks and release failed captures
- `062e4c2` B5: validate measured calibration and preserve accepted homographies
- `150e241` D7: enforce motion safety and pacing around policy execution
- `66660cd` A8: preserve scene visibility and reject incomplete simulated carries
- `a3b1007` A6: validate encoder mapping evidence and synchronize calibrated startup

## Latest follow-up (2026-10-17, Codex)

Three parallel workstreams continued from the status below:

**Combined gate: 1,242 passed, no skips, in 142.90 seconds.** Ruff check and format check pass
(81 files), text files remain LF, and protected contracts/schemas are unchanged. This follow-up is
saved in four task-sized local commits (A13, A15, A6, S0); no remote push was performed.

- Clear-table fixture candidates 3 and 4 now avoid the blocked third tray slot by placing shortest
  props first and releasing above existing tray contents near the tray center. Both pass 5/5 trials
  at seed 7 with zero collisions, limit hits or IK failures. The full fake learn reached `saved`
  (`runs/t-0011/`; isolated library `runs/e2e_fake/20261017-192552/skills/`). Scene, obstacles and
  collision checks were unchanged. The 15 cm release involves dropping props and is not physically
  validated. Seed 17 still produces rejected tea-tin/juice-box pickup collisions; this is not a
  general guarantee of clear-table success.
- Live carry now falls back to the handoff zone for out-of-box hand targets. Missing or insufficient
  clearance and incomplete rises abort holding the object. Release requires measured arrival or
  steady hand proximity; deadline, stalled-clock and tracking-limit failures retain the grasp.
  Bounded travel can still continue into validated tracking before reaching above the target.
- Added `python -m arm.inspect_hardware` and 51 tests. All serial access remains in `arm/driver.py`.
  The command opens only the bus, reads registers/calibration, and closes the raw port without any
  motor writes, including failure paths. See [hardware calibration notes](docs/hardware-calibration.md).

**New physical finding:** follower COM8 calibration matches its saved file, but leader COM7 hardware
offsets/ranges differ from its saved `my_awesome_leader_arm.json` on all six motors. No matching alternate
file exists in the local calibration cache. Both arms answered all six motors; follower 11.9-12.0 V,
31-34 C; leader 4.9 V, 26-28 C. Both reported torque off and stored goals zero. Evidence is
`runs/hardware/20261017-continuation.json`. No torque, goals or calibration were changed. Reconcile the
leader calibration before teleoperation, and still complete measured A6 before Cartesian replay.

Additional open review findings: the disabled RL policy runner needs Cartesian tip-speed enforcement
before D7; simulator snapshots currently lose incoming `seen=False` flags, so auxiliary post-handoff
runs do not establish two-object behavior. During real `Driver.connect`, LeRobot `configure()` can
temporarily remove torque between goal sync and re-enable; supported-pose validation remains required.

## Current continuation status (2026-10-17, Codex)

**Read this section before the historical handoff below.** The original handoff predates the
`027f6d7` Partial Completion commit and later fixes that were uncommitted when this continuation began. The core software and glue modules
are implemented. This continuation preserved those changes, reviewed the previous agents' reports,
and split remaining arm, orchestration and RL work across three parallel agents.

Current evidence and outstanding acceptance work are recorded in
[`docs/verification-2026-10-17.md`](docs/verification-2026-10-17.md). Task ledgers now distinguish checked
software acceptance from open hardware/live-service work. The original four audit workflows stopped
at a session limit; they did not produce completed audit verdicts.

Verified before the new fixes: **1,141 tests passed with no skips**; the fake juice-box learn reached
`saved`; the fake router matched 20/20 phrases; all six candidate fixtures passed source checks.
Five clear-table demo runs queued within 8.9-10.6 seconds. They met the time budget but did not execute
successfully because no clear-table candidate passed simulation.

**Final combined gate: 1,172 passed, no skips, in 151.54 seconds.** Ruff check and format check pass
(79 files), no tracked text files contain CRLF, and protected contracts/schemas are unchanged.
This continuation is saved as five task-sized local commits on `master` (A13, C11, D5, D6, S0).
No remote push was performed.

New work in this continuation:

- Bind STOP handlers before confirmation/countdown and connection in the orchestrator and arm CLIs;
  check every stop source before and after connecting. The orchestrator now exposes its driver to STOP
  before lifespan startup. Driver jog requires typed confirmation, and startup cleanup preserves torque.
- Bound every RL training worker response in the parent process, including a single environment;
  terminate workers stuck in Python or native math. Reject overflowing rewards and invalid arguments.
- Bound parallel training runs, reject stale output directories and failed policies, and clean up
  local process trees. Cloud transport requests still have their existing 60-second request timeout.
- Record D-GO and policy provenance: **97/100 successes, zero falls, seed 0**. The selected policy
  already matched the 507,904-step hand-written baseline; it is not a generated-reward winner.
  `rl/policies/handoff_best.json` carries the hash and independent evaluation. The original training
  summary reports 94/100 using a different evaluation seed (10001).
- Replace OpenCV's browser-incompatible mp4v video with H.264/yuv420p via PyAV. The existing video
  failed in Edge; the regenerated 10-second, 640x480 clip loads successfully. Failed renders preserve
  the prior clip. Install `requirements-rl.txt` for the new `av` dependency (19.0.1 here).
- Add `.gitattributes` to keep text files LF on Windows, and document runnable Windows commands.

The combined startup/video regression subset passed 75 tests. Both starter skills were re-scored
with the juice box, five trials, seed 7: **5/5 each, zero collisions and zero limit hits**. The fake
`good_morning` routine was also saved and then replayed to `done ok` in an isolated library.

The latest follow-up above replaces the inherited out-of-workspace clamp and incomplete-rise fallback
with handoff-zone targeting and abort-while-holding behavior. Physical validation with a held object
is still outstanding.

Hardware was checked **read-only** using the reviewed `lerobot-tools/arm_check.py` in the conda
`lerobot` environment. All six motors responded on both boards: leader COM7 / serial 5B90103224 at
4.9 V and 27-29 C; follower COM8 / serial 5B90101188 at 11.9-12.1 V and 31-34 C. No goals or torque
settings were written. Existing LeRobot calibration is not the same as VARA's joint-to-simulation
mapping: `config/joint_map.yaml` still contains A6 placeholders. Complete measured A6 and physical
camera calibration before Cartesian hardware replay. `config/rl.yaml` remains `use_policy: false`.

The baseline, curves and generated browser video exist locally. Runs/video are gitignored; regenerate
the video with `.venv/Scripts/python.exe -m rl.render_video --policy rl/policies/handoff_best.zip --out
web/static/rl_sim.mp4 --seconds 10`. Both `config/homography.npy` and the selected policy ZIP were already
tracked when this continuation began. The existing homography is synthetic, not a real-camera calibration.

## Historical handoff (2026-10-17, 13:10 local, Claude Code coordinator)

This file is for whoever picks the work up next (a person or another coding agent). It says what exists,
how it was verified, what is still running, what is open, and what must never be done without the team.
Read `CLAUDE.md` first (goal, commands, hard rules, arm hazards); it still binds everything below.

## 1. Where things stand in one paragraph

Every software module in the CLAUDE.md repo map has been implemented against the contracts in
`docs/interfaces.md` / `schemas/` / `tests/contract/`, using fakes wherever the real arm, cameras, xAI or
Daytona would be needed. Each module was written by one agent, then attacked by an independent verifier
(re-running the acceptance commands, writing probe scripts, mutation-testing the tests, blocking sockets and
devices), then fixed and re-verified. About 33,900 lines of Python exist in 27 test files plus the packages;
the last whole-tree run reported by a verifier was 741 passed with several lanes mid-edit. **Nothing has been
committed yet**: the working tree holds all of it as untracked files on `master` (single kit commit
`514540b`). Four background workflows are still running (section 6); the final whole-tree verification,
docs ticks and task-sized commits come after they finish.

## 2. Environment (verified on this laptop)

- Python venv: `.venv` built with `uv` (Python 3.12.15) because `python3.12 -m venv` fails here. Use
  `.venv/Scripts/python.exe` for everything; there is no `pip` in it (`uv pip install --python
  .venv/Scripts/python.exe ...`). Installed: mujoco 3.16, numpy 2.2, opencv-python-headless 4.13, fastapi
  0.143, uvicorn 0.54, httpx 0.28, openai 3.33, daytona 0.225.0, jsonschema 4.26, gymnasium 1.4,
  stable-baselines3 2.9, torch 2.11 CPU, lerobot 0.6.2 editable from `C:\Users\jesse\lerobot`.
- Commands: `.venv/Scripts/python.exe -m pytest -q`, `.venv/Scripts/python.exe -m ruff check .`,
  `.venv/Scripts/python.exe -m ruff format .`. The repo is LF everywhere (`core.autocrlf=true`); check new
  files for carriage returns. Timing-sensitive tests slow down when many test runs share the CPU.
- Daytona 0.225 and xAI realtime API names were verified by introspection and docs; see
  `docs/implementation-brief.md` section 4 (agents/sandbox_pool.py, orchestrator /voice/token).
- No arm, camera, `.env` or API key exists in the coordinator session. `FAKE=1` (or the individual
  `FAKE_LLM`, `FAKE_SANDBOX`, `FAKE_ROBOT`, `FAKE_CAMERA`) runs everything offline.

## 3. The brief every agent worked from

`docs/implementation-brief.md` (copied from the coordinator's scratchpad) is the binding design: repo-wide
conventions (section 1), measured geometry (2), shared config (3), public interfaces per module (4), model
files (5), fakes and the end-to-end path (6), reporting format (7), decisions after the foundation stage (8),
the glue stage (9), verified facts from the integration stage (9.8) and the open items list (9.9). Sections 8
and 9 override 2 and 4 where they differ. The contracts always win over the brief.

## 4. What was built, by stage

### Stage 1, foundation (verified, all green)

- `sim/model/`: upstream SO-101 MJCF (`so101_new_calib.xml`) edited: base at `pos="-0.0388353 0 0"` so the
  contract frame is the MuJoCo frame; gripper collision meshes replaced by fitted boxes (`gripper_housing`,
  `finger_fixed`, `finger_moving`); sites `tip` (closed-jaw midpoint) and `grasp` (2.46 cm along the closing
  direction; the jaw is asymmetric so picks target `grasp`); `arm_only.xml`; `build_scene.py` generates
  `scene_ours.xml` from `config/*` (table, tray floor + 4 walls, free bodies per object, mocap obstacles and
  hand). `tests/test_model.py` checks structure, gravity hold, grasp clearance.
- `arm/ik.py`: damped least squares with finite-difference Jacobian, planar reach pre-check, pitch ladder
  (-75, -60, -45, -30, -15, 0 deg) because the 0.251 m arm cannot point straight down at the tray; `fk`,
  `fk_tip`, `fk_grasp`, `jaw_axes`, `solve`, `solve_exact`, `solve_grasp`, `grasp_offset()`; < 2 ms per solve on
  demo targets. `config/limits.yaml` (MJCF ranges; A6 replaces them) and `config/joint_map.yaml` (sign 1 /
  offset 0 placeholders, gripper closed_pct 0 / open_pct 100, `lerobot_id: my_awesome_follower_arm`).
- `arm/driver.py`: `Driver` (LeRobot 0.6.2 `lerobot.robots.so_follower`, goal sync before every torque
  enable, strain refusal, `disable_torque_on_disconnect=False`, clips to limits-3 deg and 2 deg per call,
  non-finite targets raise, `freeze`/`resume`/`torque_off`/`torque_on`, `close_gripper_hold` holding at the
  contact reading minus 3 on the 0-100 scale), `FakeDriver`, `get_driver()`, `EscWatcher`, `--real --jog` CLI.
  `get_driver()` returns an UNCONNECTED real driver; `--real` CLIs connect after the human confirms.
- `sim/candidate_check.py` (`check_source`, re-exported by `agents/candidates.py`), `sim/success.py`
  (`evaluate`, `already_true`, `resolve_object`; obstacles resolvable by name, `$zone` substitution),
  `arm/replay.py::validate` + `bridge`, `load_limits`, `load_box`. `tests/contract/` runs with nothing skipped.
- `sim/restricted.py`: the trimmed-builtins namespace AI code runs in (restricted `__import__` serving only
  `math`, `print` to stderr, `load_entry(src, entry)` after `check_source`). The only execution points of
  AI-written code are `sim.run_candidate`'s child process and `rl.train`'s workers.
- `web/`: dashboard (`index.html`, `static/app.js`, `static/style.css`) and voice page (`voice.html`,
  `static/voice.js`) per B9/A14, Chart.js 4.4.1 pinned with SRI, verified in headless Edge against a stub
  orchestrator; later polished (no DOMContentLoaded gate, caps, deadlines, STOP re-sends).

### Stage 2, integration (verified; last two chains finishing, see section 6)

- `arm/sim_robot.py` (A8): MuJoCo `SimRobot` implementing the Robot API verbatim; grasp recipe on the
  `grasp` site, held object re-posed after every step, contact bookkeeping with body-name phrases
  ("gripper hit cereal_box", "held juice_box hit tray"), self-contacts counted, 2 deg/tick cap, every
  recorded trajectory passes `validate` (box-leaving sweeps are re-planned as Cartesian lines),
  `SimTimeLimit(BaseException)` at 60 simulated seconds, `run_skill` through `sim.restricted`,
  `run_entry` guard against candidates calling `reset`/`set_skill`. Measured `ROLL_OFFSET = -1.529` with
  `wrist_roll = angle + shoulder_pan + ROLL_OFFSET` (the brief's minus sign was wrong). One place_in_tray is
  ~11 s simulated, 0.5 s wall.
- `skills/_starter/hand_over.json`, `place_in_tray.json` (A12): trusted, measured `sim: 5/5` through the
  runner for juice_box, pill_bottle and tea_tin.
- `sim/run_candidate.py` + `sim/scoring.py` (C4): the CLI from CLAUDE.md, one spawn child streaming a JSON
  line per trial, per-trial wall-clock timeout with kill (no orphans), stdout holds only the score line,
  `out/traj.json`, `final_scene.json`, `score.json`, `trials.json`; `perturb_scene`, `summarize`, `rank`.
  A fix pass for minor findings (KeyboardInterrupt in a candidate, huge error text, NaN scene values,
  non-UTF-8 files) is running now.
- `arm/replay.py::replay` + `arm/real_robot.py` (A9/A10/A13): 30 Hz streaming, bridging at 1 deg/tick,
  tip-speed caps 0.2 and 0.1 m/s near the hand by sub-stepping, grip check against `held_min`, 10 cm hand
  freeze, close refused near the hand, `home`, `freeze`, `carry_to_hand` (IK tracking at 10 Hz, steady-hand
  release, handoff-zone fallback, optional RL policy), `--real`/`--fake` CLIs. A second fix pass is running
  for two findings: hand tracking that could push a held object into the table for far hand positions,
  and an Esc during `Driver.connect()` being discarded by the CLI (the driver side is being fixed separately).
- `agents/llm.py`, `agents/router.py`, `agents/candidates.py`, `agents/prompts/*.md` (C1-C3, C7 repair):
  xAI client with json_schema-then-text fallback, budgeted retries, key scrubbing, llm.jsonl logging; fake
  layout `tests/fixtures/llm/` (aliases, 16 router replies, 6+6+6 candidates, 4 rewards); router with the
  predicate rules, already-done check, keyword fallback, 20 phrases in `tests/fixtures/router_phrases.json`
  (`python -m agents.router --eval --fake` = 20/20); six-hint candidate generation, repair, routine candidate.
- `vision/` (B1-B8): synthetic pinhole camera (f = 450 px; the brief's 600 px could not see its own
  calibration markers), `FakeCamera` rendering real ArUco markers with height parallax, `markers`,
  `calibrate` (homography, height correction, `config/homography.npy`), `SceneTracker` at 10 Hz with
  median-5 filter and 0.5 s seen timeout (real chain on the fake camera: < 1 mm, < 3 deg), `FakeSceneTracker`,
  `verify`/`take_after`, `scripts/make_markers.py`, `scripts/touch_markers.py` (I2, `--fake` works).
- `rl/handoff_env.py`, `rl/rewards/baseline.py`, `rl/train.py`, `rl/evaluate.py` (D1/D2): Gymnasium env
  with the 16-float observation (tip from forward kinematics of the measured joints), randomizations,
  env-level fall penalty (-50, decided; the formula alone made the policy learn to fall), SB3 PPO trainer with
  the reward loaded through `sim.restricted` inside each worker, progress.csv/checkpoints/summary.json.
  A 131k-step check reached 25/50 successes; the 500k D2 run is part of the RL glue chain.

### Stage 3, glue (running now)

Skill library (`agents/library.py`), RL glue (`rl/rewards/write_rewards.py`, `rl/train_parallel.py`,
`rl/policy_runner.py`, `rl/render_video.py`, D2 baseline run + D-GO evaluation), sandbox pool and planner
(`agents/sandbox_pool.py`, `agents/planner.py`, `scripts/daytona_smoke.py`), orchestrator and end-to-end
script (`agents/orchestrator.py`, `scripts/e2e_fake.py`, `tests/test_orchestrator.py`,
`tests/test_scaffold.py`), fixture-candidate tuning against the runner, hardening of router/web/vision/test
pinning, driver hardening, and a four-lens read-only audit. Their specs are brief sections 9.1-9.6 and 9.9.

## 5. Decisions and measured facts you must not re-derive

- Frame: origin under the pan axis; `fk(home_q)` = (0.1155, 0, 0.2745) with `home_q = [0, -1.35, 0.02,
  1.45, 0]` (the kit's placeholder put the gripper inside the tea tin).
- Reach: upper arm 0.116 + forearm 0.135 = 0.251 m; straight-down picks only near the base; the tray needs
  -45 to -30 deg; transit at z 0.28 only near pitch 0 and within ~0.38 m; the workspace box contains
  unreachable corners (`move_to` returns False there by design).
- Scene layout (config/fake_scene.json and config/scene.yaml): cereal box at (0.28, 0.07), handoff zone
  and parked hand at (0.28, -0.08), tea tin at (0.16, -0.02), tray slots 5 cm apart
  `[[0.275, 0.195], [0.325, 0.195], [0.275, 0.245], [0.325, 0.245]]`. With the kit's box at (0.24, 0.11) every
  tray slot was in the box's bearing cone and the stretched forearm clipped it on every release pose. Slot 2
  still collides (SimRobot skips it); the z 0.28 transit above slot 4 is IK-infeasible. A15 re-tapes the
  real table; after any change run `python -m sim.model.build_scene` and re-score the starters.
- Driver: `command()` returns bool (False while stopped/torque off), non-finite raises, unconnected raises;
  `close_gripper_hold`; `limits()`/`joint_map()` raise on malformed config.
- Grasp: open fully (1.0) before descending; attach test on the grasp center within 1.0 cm, 20-80 % height,
  20 deg yaw; tea tin lands on its side in the tray in sim because the contract's release height plus the
  forced tilt drops it 4 cm (real-table risk; lower tray wall or nearer tray).
- Runner: `SAFE_BUILTINS` needs the restricted `__import__` or `import math` fails at exec; per-trial
  results in `out/trials.json`; the orchestrator puts `trial_results`, `runner`, `hint` into
  `candidate_scored` details (dashboard reads them).
- Router rules: learn specs use concrete names and `params: {}`, `in_zone` only with `tray`, no obstacle
  names in predicates, unseen objects -> queue.
- Vision: items never detected are omitted from the published scene; `take_after` needs `before` to resolve
  "*"; `calibration_error` raises without markers in view (orchestrator catches it).
- RL: env-level fall penalty stays; D4 prompts must name `info["fell"]`, `info["step"]`, `info["success"]`;
  policies start from home +-5 deg (widen before D7).

## 6. Workflows running at the time of writing

| Workflow | Items | State |
| --- | --- | --- |
| vara-integration | runner fix pass; replay second fix pass | last two chains |
| vara-glue-a | library; RL glue incl. D2 500k run and D-GO number | implementing |
| vara-glue-b | pool+planner; orchestrator+e2e; fixture tuning; router/web/vision hardening; test pinning | implementing |
| vara-glue-c | driver hardening; four read-only auditors | implementing |

Agent transcripts and per-agent reports (`journal.jsonl`, one `result` line per agent) live under
`C:\Users\jesse\.claude\projects\C--Users-jesse-GitHub-VARA\8a8822bb-1a8e-4bea-8e33-2fdce982b538\subagents\workflows\<run id>\`.
Each engineer's final message lists files written, exact commands with last lines, assumptions, unfinished
work, and where the brief or contract looked wrong; each verifier's verdict lists reproduced commands and
problems with severity. Read those before changing a module.

## 7. Still to do after the workflows finish (the final hardening stage)

1. Apply the verifier findings in brief section 9.9 that touch files that were in flux: rise-first
   `carry_to_hand` in both `arm/sim_robot.py` and `arm/real_robot.py` (then re-score the starters, rule in
   `.claude/rules/arm.md`), `pick()` while holding re-reads the pose, tip-speed re-measure on straight legs,
   `place()` default release on tall objects; `arm/replay.py` adopting the driver's new gripper helpers and
   failing loud on a non-dict scene; `rl/train.py` OverflowError guard, reward time budget, `--out ""`.
2. Fold in the four audit lenses' findings (safety, secrets/isolation, discipline, docs/CLI).
3. Whole-tree gates: `pytest -q` with nothing skipped, `ruff check .`, `ruff format --check .`,
   `python scripts/e2e_fake.py "put the juice box in the tray"` ending in `saved`, every hardware-free
   Done-when in `docs/tasks/*.md`, no CRLF files, `git add config/homography.npy` and
   `rl/policies/handoff_best.zip` (binaries), `.env.example` updated with every variable the code reads.
4. Tick the progress boxes in `docs/tasks/*.md` only for tasks whose hardware-free Done-when passed; write
   the D-GO number and decision at the top of `docs/tasks/D-rl.md`.
5. Commit in task-sized commits, message `<ID>: <title>`, ending with
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never push without asking.

## 8. Needs hardware, keys or the team (not done, by design)

- Arm: A1-A3 (install, ports, calibration/teleop), A4 `--real --jog`, A6 joint map and 1.5 cm accuracy check,
  A9 real replay 4/5, A10 carry test, A11 gripper measurements and object choice, A12 real replays, A13 Esc
  latency on hardware, A15 demo choreography, I2 touch markers, I3 first real learn, M gates.
- Camera: B1 30-minute soak, B3 printing, B4 mount and `Hc`/`c`, B5 live calibration, B6 live accuracy,
  B8 one wrist still per grasp (planned in the orchestrator's replay hook).
- Keys: C1 `--ping`, C2 live `--eval`, C5/C6 Daytona snapshot and pool, D3/D5 Daytona training, A14 live
  voice session; whether xAI accepts the router's oneOf/pattern JSON schema as `json_schema` output is
  unverified (a 4xx takes the plain-text path by design).
- D6 video needs a working OpenGL context (the RL glue agent reports whether it works here); D7 needs M4.

## 9. Contract disagreements found (do not change contracts without the team)

- `tests/fixtures/trajectory_ok.json` has shoulder_lift -1.75 rad, outside `config/limits.yaml` with the
  3 deg margin; the contract test passes only with its own +-1.9 limits. Never use it as a replayable path.
- `docs/interfaces.md` scene example still shows the kit's 4 cm tray slots and the old handoff zone.
- `docs/tasks/A-arm.md` A4 snippet is stale for LeRobot 0.6.2 (import path, id, 6-name JOINTS).
- The brief's `set_roll` formula sign and the 600 px synthetic camera were wrong (fixed in code, see 5).
- `in_zone` with the `handoff` zone (no radius) raises; the router never emits it.
- `tests/test_ik.py` demo-target timing: reachable targets 20 ms, unreachable ones 60 ms (CPU-load tolerance).

## 10. Rules of engagement that were followed and must stay

- AI-written code executes only through `sim.restricted.load_entry` inside `sim.run_candidate`'s child or
  `rl.train`'s workers; never in the orchestrator process.
- The real arm moves only through `arm.replay.replay()` over a `validate()`-accepted trajectory or the
  primitives in `arm/real_robot.py`, only with `--real`, after a typed confirmation and a 3 s countdown,
  with Esc watched. STOP freezes with torque on.
- Keys come from `.env` via `load_dotenv()` in entry points only; never printed, logged, returned, or sent
  to the browser (the voice page gets a 300 s token from `POST /voice/token`).
- Tests never touch the network, serial ports or cameras; verifiers ran the suites with all three blocked.
- Sandboxes are created only by `agents/sandbox_pool.py` (cap 7), `rl/train_parallel.py` (cap 4) and
  `scripts/daytona_smoke.py`, and deleted on shutdown.
