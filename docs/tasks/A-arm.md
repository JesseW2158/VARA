# Lane A tasks: arm, simulation, replay, voice

**Continuation verification:** [2026-10-17 evidence](../verification-2026-10-17.md). Checked boxes have supporting evidence; hardware and live-service acceptance remains open.

Lane A makes simulation and the real arm agree, then makes the real arm do exactly what simulation checked and nothing else. The joint map check (A6) and arm-meets-camera (I2) are the most important hours of the night: if they fail, nothing downstream works.

**Progress** (in time order)

- [x] A1 · Install · Fri 10:00 PM
- [ ] A2 · Power and ports · Fri 10:30 PM
- [ ] A3 · Calibrate and teleoperate · Fri 11:30 PM
- [ ] A11 · Gripper measurements and object choice · Sat 12:30 AM
- [ ] A5 · Simulation model · Sat 1:00 AM (tip site by 12:45 AM for Lane D)
- [ ] A4 · Driver with freeze · Sat 1:30 AM
- [ ] A6 · Joint map and accuracy check · Sat 2:30 AM
- [x] A7 · Inverse kinematics · Sat 3:00 AM (M1a)
- [x] A8 · SimRobot in MuJoCo · Sat 4:30 AM
- [ ] A9 · Replay engine · Sat 5:00 AM
- [ ] A10 · Live primitives · Sat 5:30 AM
- [ ] A12 · Starter skills · Sat 6:00 AM (M1)
- [ ] A13 · Safety layer · Sat 1:00 PM
- [ ] A14 · Voice page · Sat 3:30 PM (M3)
- [ ] A15 · Demo scene and choreography · Sat 9:00 PM (M5)

## A1 · Install

**Needs:** nothing. **By:** Fri 10:00 PM. On the demo laptop, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt "lerobot[feetech]"
```

**Done when:** `python -c "import lerobot, mujoco; print(lerobot.__version__)"` prints a version.

## A2 · Power and ports

**Needs:** A1. **By:** Fri 10:30 PM. The follower (white) takes the 12 V supply and the leader (black) the 5 V supply; never swap them. Run `lerobot-find-port` once per arm and save the results as `FOLLOWER_PORT` and `LEADER_PORT` in `.env`. Ports can swap whenever cables are replugged, so re-run it after any replug.

**Done when:** both ports are in `.env` and `lerobot-find-port` agrees after unplugging and replugging each arm.

## A3 · Calibrate and teleoperate

**Needs:** A2. **By:** Fri 11:30 PM.

On this laptop both arms already have calibration files. Reuse `my_awesome_follower_arm` and
`my_awesome_leader_arm`; recalibration is a separate physical procedure, not an installation step.
Before teleoperation follow the read-only check, pose check and goal-sync procedure in
`C:/Users/jesse/lerobot-tools/HANDOFF.md`. The generic commands below are for an initial setup.
On PowerShell use `$env:FOLLOWER_PORT` / `$env:LEADER_PORT` and put each command on one line.

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=my_awesome_leader_arm
lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=my_awesome_leader_arm
```

Assembled kits normally ship with motor IDs set. Run `lerobot-setup-motors` only if calibration reports a missing motor; it needs each motor connected on its own, which means unplugging the daisy chain. Reuse the configured calibration IDs everywhere afterward, because calibration files are keyed by ID.

**Done when:** the follower mirrors the leader smoothly through every joint's full range. Then unplug the leader: it stays unplugged except during teleoperation, and always during demos.

## A11 · Gripper measurements and object choice

**Needs:** A3. **By:** Sat 12:30 AM. **Files:** `config/objects.yaml`.

1. Measure the jaw opening at gripper 100, in mm.
2. Record the empty-closed reading: command gripper 0 and read it after 0.5 s.
3. Choose three objects and one obstacle. Rules: width across the grasp at most the jaw opening minus 2 cm; height 6 to 11 cm; a flat top that fits a 3 cm marker; under 100 g; nothing sharp, hot or fragile. Proposed: an empty pill bottle, an empty juice box, a narrow tea tin, and a cereal box about 20 cm tall as the obstacle.
4. Teleoperate 5 grasps per object and record the closed readings. Set `held_min` to half the gap between the smallest closed reading and the empty-closed reading (0 to 100 scale).
5. Write `config/objects.yaml` in the kit's format and give Lane B the final list so markers get printed.

**Done when:** `pytest -q tests/test_config.py` passes, and teleoperation holds each object 5 of 5 times.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m arm.driver --real --measure-grip --objects pill_bottle juice_box tea_tin --grasps 5` runs steps 1, 2 and 4: it asks for the jaw opening in mm, reads the empty-closed value 0.5 s after a close on nothing, then closes on each object five times and asks whether it held. It writes `runs/calibration/gripper-<UTC>.json` (or a new `--out PATH`) and prints the `gripper:` line for `config/joint_map.yaml` and `held_min` per object for `config/objects.yaml`; paste them by hand, then run `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py`. Nothing under `config/` is written. Like the A6 jog, it refuses with exit 2, before anything moves, when a joint of the start pose lies more than 0.5 degrees outside the commandable band (`config/limits.yaml` minus 3 degrees), and names the joint; the rest pose is outside the placeholder band, so pose the arm mid-range first. Esc freezes the arm and ends the session; the partial report is still written, without paste lines. The first real session must check that the Esc watcher does not take characters typed at the prompts. Offline: `.\.venv\Scripts\python.exe -m arm.driver --fake --measure-grip --objects pill_bottle --grasps 2 --fake-contact 25 --out <new path>`. Order: Part 3 of the [operator runbook](../runbook.md#part-3-arm-calibration-a6-a11-a13).

## A4 · Driver with freeze (`arm/driver.py`)

**Needs:** A3. **By:** Sat 1:30 AM. **Files:** `arm/driver.py`, `tests/test_driver_fake.py`.

Use the implemented wrapper; direct LeRobot `connect()` can enable torque against stale goals.
The installed LeRobot 0.6.2 classes are `SOFollower` and `SOFollowerRobotConfig` from
`lerobot.robots.so_follower`, imported only by `arm/driver.py`.

```python
from arm.driver import FakeDriver

robot = FakeDriver()
q, gripper = robot.read()  # five radians, separate 0.0-1.0 gripper
robot.command(q, gripper)
robot.freeze()            # keeps torque on; command() refuses until resume()
```

For physical testing use `python -m arm.driver --real --jog`, which provides the confirmation,
countdown, goal synchronization and Esc watcher. The wrapper reads the configured calibration ID.

- `read() -> (q, gripper)`: 5 joint angles in radians and the gripper from 0.0 to 1.0, converted through `config/joint_map.yaml`.
- `command(q, gripper)`: clips each joint to its calibrated range minus 3° and to 2° per 30 Hz tick, whatever the caller asks. LeRobot's `max_relative_target` is only a second layer; confirm it clips on your installed version.
- `freeze()`: command the present position as the goal, keep torque on, and set the stopped flag. `command` is ignored until `resume()`.
- `torque_off()`: explicit maintenance only, with the arm supported at rest. The wrapper sets `disable_torque_on_disconnect=False`; disconnecting preserves torque. STOP calls `freeze()`.
- One lock around all serial access. `FAKE_ROBOT=1` swaps in an in-memory arm with the same methods.

**Done when:** `pytest -q tests/test_driver_fake.py` passes, and `python -m arm.driver --real --jog` moves each joint +10° and back at 30 Hz, while Esc pressed mid-move holds the arm where it is without sagging.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m arm.driver --fake --jog` runs the same jog on the in-memory arm as a dry run; `--real` and `--fake` are mutually exclusive, and `--port` and `--id` default to `FOLLOWER_PORT` from `.env` and `my_awesome_follower_arm`. During `.\.venv\Scripts\python.exe -m arm.driver --real --jog`, every Esc pressed after the confirmation prints `Esc -> freeze command: N ms` and appends a record with its phase to `runs/calibration/esc-latency.jsonl`, which A13 judges. The power-on and goal-sync routine before this command is Part 2 of the [operator runbook](../runbook.md#part-2-power-on-routine-every-time-before-anything-connects-to-the-follower).

## A5 · Simulation model

**Needs:** A1. **By:** Sat 1:00 AM, with the tip site pushed by 12:45 AM so Lane D can start. **Files:** `sim/model/`.

```bash
git clone https://github.com/TheRobotStudio/SO-ARM100
python -m mujoco.viewer --mjcf=SO-ARM100/Simulation/SO101/scene.xml
```

Copy `Simulation/SO101/` into `sim/model/` and write `sim/model/scene_ours.xml`, which includes `so101_new_calib.xml`. First add a `tip` site at the midpoint of the jaw tips (place it by eye in the viewer) and push that alone. Then add the table, the tray as thin box walls, one free body per object and one fixed box per obstacle sized from `config/objects.yaml`, and a mocap box for the user's hand (8 × 10 × 4 cm) parked below the table whenever the hand isn't seen.

**Done when:** the viewer shows the arm, tray, objects, obstacle and hand box, and the arm holds a pose under gravity.

## A6 · Joint map and accuracy check

**Needs:** A4, A5. **By:** Sat 2:30 AM. **Files:** `config/joint_map.yaml`, `config/limits.yaml`.

For each of the 5 arm joints, start at a measured, supported reference pose and compare a positive joint increment with the same increment in the viewer (+20 degrees is +0.349 rad). Use the capped driver jog and establish clearance before each increment. Do not command all joints to 0: LeRobot zero is each calibrated range midpoint, not the rest pose. Record each joint's `sign` (+1 or -1) and `offset_deg`, plus the gripper's 0 to 100 range against the model's jaw joint range. Write the calibrated joint ranges, in radians, to `config/limits.yaml`. Then put the real arm in 5 poses spread over the workspace, tape-measure the tip from the origin, and compare with the model's forward kinematics from the same readings.

**Done when:** all 5 poses agree within 1.5 cm. A joint off by a constant needs its `offset_deg` fixed.

The [read-only hardware report](../hardware-calibration.md) records encoder calibration and its
fingerprint without moving the arms. It supports A6 evidence collection; it does not replace the
five physical measurements. The 2026-10-17 check found the leader's saved calibration differs from
its hardware, so reconcile that separately before teleoperation.

**Tools added 2026-10-17:** The sign test is `.\.venv\Scripts\python.exe -m arm.driver --real --jog-joint NAME --deg 20 --hold 10`: only that joint moves out by 20 degrees at 1 degree per 30 Hz tick, holds 10 s and returns, printing the LeRobot degrees, the mapped radians, the expected +0.349 rad and the fk tip to compare with the viewer; it writes no config. It refuses with exit 2, before anything moves, when any joint of the start pose lies more than 0.5 degrees outside the commandable band (`config/limits.yaml` minus 3 degrees), and names the joint; the rest pose is outside the placeholder band, so pose the arm mid-range first. After a read-only `.\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/reference-pose.json` at a reference pose, `.\.venv\Scripts\python.exe -m arm.check_mapping --solve runs/calibration/solve.json` computes `offset_deg = model_deg - sign * lerobot_deg` and the calibrated `[lo, hi]` radians, prints both YAML blocks to paste, and writes `runs/calibration/solve-<UTC stamp>.json` (or a new `--out PATH`; `--calibration-file PATH` requires the current encoder cache to match the report's fingerprint). It never edits `config/`. On a rerun after recalibration, restore the placeholder joint lines in both config files first, because the sign rule assumes `sign: 1`, and re-jog after pasting to confirm each sign. Full procedure: [joint calibration](../joint-calibration.md) and Part 3 of the [operator runbook](../runbook.md#part-3-arm-calibration-a6-a11-a13).

## A7 · Inverse kinematics (`arm/ik.py`)

**Needs:** A5. **By:** Sat 3:00 AM. **Files:** `arm/ik.py`, `tests/test_ik.py`.

- Export `fk(q) -> (xyz, pitch)` and `solve(xyz, pitch, q0) -> (q, ok)`.
- Solve for shoulder_pan, shoulder_lift, elbow_flex and wrist_flex; wrist_roll is set separately (A8).
- Task: tip position (3 values) and tip pitch, `pitch = asin(a_z)`, where `a` is the unit vector pointing out of the jaws (a column of the tip site's rotation matrix).
- Jacobian by finite differences: nudge each joint by 1e-4 rad, call `mujoco.mj_kinematics`, read the tip site.
- Step: Δq = Jᵀ(JJᵀ + λ²I)⁻¹e with λ = 0.02. Clamp to joint limits after every step, start from `q0`, stop after 200 iterations.
- Success: position error under 5 mm and pitch error under 3°. If pitch -π/2 is unreachable for a far target, retry at -60°, then -45°, before returning failure.

**Done when:** `pytest -q tests/test_ik.py` passes: 200 random reachable targets solve within 5 mm and 3°, each in under 20 ms, and `fk` of every solution matches its target.

## A8 · SimRobot in MuJoCo (`arm/sim_robot.py`)

**Needs:** A5, A7, S0. **By:** Sat 4:30 AM. **Files:** `arm/sim_robot.py`, `tests/test_sim_robot.py`.

Replace S0's kinematic stand-in behind the same class. The Robot API and its notes in Contracts are the spec.

- `move_to`: IK, then a joint-space path from the current joints with a trapezoidal speed profile. Joints at most 60°/s, tip at most `speed` (capped at 0.2 m/s), waypoints at 30 Hz appended to the current `path` segment.
- `transit_z`: rise straight up to it, travel at that height, then descend.
- `set_roll(angle)`: wrist_roll = angle + shoulder_pan + `roll_offset`. Measure `roll_offset` so the jaws line up with an object at yaw 0.
- Grasping without friction physics: while an object is held, after every `mj_step` set its free joint to the tip pose plus the offset captured at grasp time and zero its velocity. Attach only within the tolerance in Contracts.
- Collisions, limit hits, IK failures and the 60 s simulated-time limit (a `BaseException` subclass), as in Contracts.
- Object, obstacle and hand poses load from the scene JSON; `final_scene.json` is written from the end state.

**Done when:** `pytest -q tests/test_sim_robot.py` passes, and with `config/fake_scene.json`: `place_in_tray` on `juice_box` passes 5 of 5 trials, a candidate that drops the juice box beside the tray fails with a reason, and a candidate without `transit_z` fails with a `fail_reason` naming `cereal_box`.

## A9 · Replay engine (`arm/replay.py`)

**Needs:** A4, A6, S0. **By:** Sat 5:00 AM. **Files:** `arm/replay.py`, `tests/test_replay_fake.py`.

Implement `replay(traj)` per Contracts, Trajectory: bridging moves, 30 Hz streaming, grip checks against `held_min`, live primitives, STOP and hand checks every tick, the abort behavior and the return value. `validate` exists from S0; call it first and never move on a non-empty result.

**Done when:** `pytest -q tests/contract/test_trajectory_validate.py tests/test_replay_fake.py` passes (on the fake arm, STOP mid-path freezes, a missed grasp aborts and goes home, and a hand within 10 cm freezes), and `python -m arm.replay --real --traj <a sim-planned place_in_tray>` succeeds 4 of 5 times.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m arm.replay --real --traj <sim-planned traj.json> --tracker --record place_in_tray` appends one line per attempt to `runs/calibration/replays-place_in_tray.jsonl` (time, trajectory path and sha256, skill, mode, ok, reason, event count, elapsed seconds) and prints `place_in_tray: k of n ok`; a failed validation, a declined confirmation or a refused hand source leaves no line. The hand check takes `--tracker` or `--scene-file FILE`, never both. With `--real`, `--tracker` is refused (exit 2, before the prompt and before a Driver exists) when the tracker or its camera is a fake, cannot be built, or delivers no new frame in 5 s. With `--real`, `--scene-file FILE` accepts only the file of a running `.\.venv\Scripts\python.exe -m vision.scene --out FILE` on the real camera: its provenance sidecar `FILE.source.json` must say `writer: vision.scene`, `source: live` and `is_fake: false`, and the file must show a fresh scene within 5 s, otherwise the CLI exits 2 with nothing moved; an accepted file prints its writer's tracker, camera and pid. During the run a scene older than about 0.6 s reads as all unseen, and a sidecar that disappears or changes (the writer stopped or restarted) freezes the arm with torque on. `--tracker` is the stricter source: a writer whose camera returns only missed grabs keeps publishing fresh all-unseen scenes, which passes the scene-file check while the arm runs without hand data. Each Esc is recorded for A13 with source `arm.replay`. Offline rehearsal: the same command with `--fake` instead of `--real`.

## A10 · Live primitives (`arm/real_robot.py`)

**Needs:** A9, B6. **By:** Sat 5:30 AM. **Files:** `arm/real_robot.py`, `config/rl.yaml`.

- `home()`: the tucked pose `home_q` in `config/scene.yaml`, chosen with Lane B so no calibration or object marker is hidden from the overhead camera.
- `freeze()`: the driver's freeze, plus cancelling any running primitive.
- `carry_to_hand(object)`: with the object held, move above the tracked hand plus 5 cm (or the `handoff` zone if the hand isn't seen), at 0.1 m/s at most within 15 cm of the hand. Wait until the hand stays within 8 cm for 1 s (after 10 s, use the `handoff` zone), open, back off 5 cm upward, go home and log `handoff_done`.
- Hand tracking uses repeated IK moves at 10 Hz, or the RL policy when `config/rl.yaml` says `use_policy: true` (D7 sets it once M4 passes).

**Done when:** `hand_over` on the pill bottle succeeds 4 of 5 times with the hand still, and `python -m arm.real_robot --real --carry-test` follows a hand moving at 5 cm/s or slower.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m arm.real_robot --real --carry-test --object pill_bottle --tracker` takes the hand from the live overhead tracker, with the same refusals as A9 (a fake tracker or camera, or no new frame in 5 s, exits 2 before the prompt). The alternative is `--scene-file runs/live/scene.json` with `.\.venv\Scripts\python.exe -m vision.scene --out runs/live/scene.json` running in a second window. With `--real` that file needs the writer's live provenance sidecar `runs/live/scene.json.source.json`, which `vision.scene --out` writes before its first scene and removes when it stops, so a fake or leftover scene file is refused with nothing moved. The printed result's events include one line per seen hand at 10 Hz (`carry_to_hand t=... s: hand (x, y), d m from the grasp center`), from which the 5 cm/s hand speed is read, and each Esc is recorded for A13 with source `arm.real_robot`. `carry_to_hand(..., policy={"use_policy": True, "policy": "<zip>"})` switches the RL policy on for one call without editing `config/rl.yaml` (D7 uses it). Offline: `.\.venv\Scripts\python.exe -m arm.real_robot --fake --carry-test --tracker` ends with `handoff_done`. Order: Part 5 of the [operator runbook](../runbook.md#part-5-arm-meets-camera-i2-a9-a10-a12-i3).

## A12 · Starter skills (`skills/_starter/`)

**Needs:** A8, A10. **By:** Sat 6:00 AM. **Files:** `skills/_starter/hand_over.json`, `skills/_starter/place_in_tray.json`.

Write both in the skill format with `trusted: true`:

- `hand_over(object)`: success `{"all": [{"moved": {"object": "$object", "min": 0.05}}, {"event": "handoff_done"}]}`; code: `pick`, then `carry_to_hand`.
- `place_in_tray(object)`: success `{"in_zone": {"object": "$object", "zone": "tray"}}`; code: `pick` and `place_in_zone("tray")`, both with `transit_z` set to the tallest obstacle plus 0.08 m.

**Done when:** both pass 5 of 5 trials with `python -m sim.run_candidate --skill skills/_starter/<name>.json --args '{"object": "juice_box"}' --scene config/fake_scene.json --trials 5 --seed 7 --out runs/check`, and each replays on the real arm 4 of 5 times.

**Tools added 2026-10-17:** Tally each skill's real replays with the A9 recorder, for example `.\.venv\Scripts\python.exe -m arm.replay --real --traj <sim-planned traj.json> --tracker --record hand_over`, which appends to `runs/calibration/replays-hand_over.jsonl` and prints `hand_over: k of n ok` (`--record place_in_tray` likewise). `hand_over` needs a hand source, `--tracker` or `--scene-file` with the live provenance sidecar described under A9; without scene data its carry ends `error: carry clearance unavailable (no scene data)`.

## A13 · Safety layer

**Needs:** A9, A10. **By:** Sat 1:00 PM. **Files:** `arm/driver.py`, `arm/replay.py`, `arm/real_robot.py`, `tests/test_safety.py`.

Implement every code rule in Safety rules: the workspace box, speed caps (0.1 m/s within 15 cm of the hand), the 0.30 m ceiling, no gripper close within 10 cm of the wristband, and freeze on STOP from Esc, the dashboard and voice.

**Done when:** `pytest -q tests/test_safety.py` passes (out-of-box targets refused, speed caps hold, a close near the hand refused), and on the real arm Esc freezes motion within 0.2 s, measured from the key press to the freeze command in the log.

**Tools added 2026-10-17:** Every `--real` entry point (`arm.driver`, `arm.replay`, `arm.real_robot`, `scripts/touch_markers.py`, `rl.policy_runner` and the orchestrator's console Esc, recorded as `orchestrator-console`) appends each Esc to `runs/calibration/esc-latency.jsonl` with its source and a phase: `session` when pressed after the driver's connect returned, `connect` when pressed while connecting, with nothing moving. `.\.venv\Scripts\python.exe -m arm.driver --esc-latency-report` (or `--file PATH`) prints the records and the verdict without touching hardware. Only session-phase records count against 200 ms; connect-phase records are listed apart and never judged. It exits 0 when every session record is within 200 ms, 1 when one is over or never froze, and 2 when the file is missing, empty, malformed or has no session record. `POST /stop` takes `{source, t_client_ms}`, freezes before reading the body, and returns `source`, `latency_ms` (server side) and `press_to_freeze_ms` (key press to freeze on the laptop clock), the same fields the `stopped` event carries in `runs/<task_id>/events.jsonl`. The dashboard and the voice page log `stop <source>: server N ms, round trip N ms, press to freeze N ms`, a spoken stop adds `talk release to response N ms`, and on the real connected arm the stop is also appended to `esc-latency.jsonl` as `dashboard-<source>`. Windows timestamps tick about every 15.6 ms, so each latency is accurate to about one tick. Drill: Part 3, step 6 of the [operator runbook](../runbook.md#part-3-arm-calibration-a6-a11-a13).

## A14 · Voice page (`web/voice.html`)

**Needs:** C11 (`/voice/token`). **By:** Sat 3:30 PM, ahead of M3 at 4:00 PM. **Files:** `web/voice.html`, `web/static/voice.js`.

Start from the xAI cookbook's Web Agent (WebSocket) example and serve it at `/voice` inside the dashboard.

- The browser fetches a short-lived token from `POST /voice/token` and opens `wss://api.x.ai/v1/realtime?model=grok-voice-latest` with it as the WebSocket subprotocol `xai-client-secret.<token>`, since browsers can't set headers. The API key never reaches the page.
- Send `session.update` with a voice (for example `eve`), the instructions and three tools from Contracts, and `turn_detection` type `null` for push-to-talk.
- While the T key or the on-screen button is held, send microphone audio as `input_audio_buffer.append` (PCM, 24 kHz by default). On release, send `input_audio_buffer.commit`, then `response.create`.
- Tool calls arrive as `response.function_call_arguments.done` with a name, a `call_id` and JSON arguments. Call our endpoint, send `conversation.item.create` with a `function_call_output` item carrying the same `call_id`, then `response.create`. For `do_task`, return "On it" at once; when `/events` reports the end stage, send the summary as a text item, then `response.create`.
- Play replies from `response.output_audio.delta` events.
- Voice "stop" is a convenience. Esc and the STOP button are the safety path.

**Done when:** M3 passes in the noisiest corner you can find.

## A15 · Demo scene and choreography

**Needs:** M2. **By:** Sat 9:00 PM (M5), with Lane C.

1. Tape layout L1 from the Demo script: object spots, tray, cereal box, hand rest and camera position.
2. Run "clear the table" 5 times with `DEMO_MODE=1` and record each candidate's result and `fail_reason` from `runs/`.
3. Tune the cereal box until 2 to 4 of 6 candidates pass. If all 6 pass, move it 3 cm toward the line between the objects and the tray; if fewer than 2 pass, move it 3 cm away. Re-tape.
4. Save the best rehearsal skills to `skills/_backup/`.
5. Write the 30-second reset checklist, drill STOP with the teammate on the power plug, and run the full script three times.

**Done when:** M5 passes.

**Tools added 2026-10-17:** For steps 2 and 3, `.\.venv\Scripts\python.exe scripts/rehearsal_report.py` reads every learn under the runs folder without changing anything and prints each candidate's result and `fail_reason` plus step 3's cereal-box verdict (keep it, move it 3 cm toward or away, or rerun); `--task ID`, `--since WHEN` and `--last N` filter, and `--out PATH` also writes a new JSON report (a `.json` file, or a folder that receives `rehearsal-<stamp>.json`). For step 4, `.\.venv\Scripts\python.exe -m agents.library --save-backup clear_table --note "learned on the real arm, layout L1" --keep-real --force` validates the skill, copies it to `skills/_backup/clear_table.json` and appends the note to `skills/_backup/provenance.jsonl`. It refuses trusted skills, starter names and an existing backup without `--force`, and `--keep-real` belongs only on a skill learned on the physical arm (otherwise the real counts are reset to 0). The checked-in `skills/_backup/clear_table.json` comes from a fake learn (real 0 of 0), so replace it after the physical rehearsal. Step 5's checklist is [Physical reset, 30 seconds](../demo.md#physical-reset-30-seconds), and the order is Part 7 of the [operator runbook](../runbook.md#part-7-rehearse-freeze-submit-judge).
