# Lane A tasks: arm, simulation, replay, voice

Lane A makes simulation and the real arm agree, then makes the real arm do exactly what simulation checked and nothing else. The joint map check (A6) and arm-meets-camera (I2) are the most important hours of the night: if they fail, nothing downstream works.

**Progress** (in time order)

- [ ] A1 · Install · Fri 10:00 PM
- [ ] A2 · Power and ports · Fri 10:30 PM
- [ ] A3 · Calibrate and teleoperate · Fri 11:30 PM
- [ ] A11 · Gripper measurements and object choice · Sat 12:30 AM
- [ ] A5 · Simulation model · Sat 1:00 AM (tip site by 12:45 AM for Lane D)
- [ ] A4 · Driver with freeze · Sat 1:30 AM
- [ ] A6 · Joint map and accuracy check · Sat 2:30 AM
- [ ] A7 · Inverse kinematics · Sat 3:00 AM (M1a)
- [ ] A8 · SimRobot in MuJoCo · Sat 4:30 AM
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

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=leader
lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=leader
```

Assembled kits normally ship with motor IDs set. Run `lerobot-setup-motors` only if calibration reports a missing motor; it needs each motor connected on its own, which means unplugging the daisy chain. Use the id `follower` everywhere afterward, because the calibration file is keyed by it.

**Done when:** the follower mirrors the leader smoothly through every joint's full range. Then unplug the leader: it stays unplugged except during teleoperation, and always during demos.

## A11 · Gripper measurements and object choice

**Needs:** A3. **By:** Sat 12:30 AM. **Files:** `config/objects.yaml`.

1. Measure the jaw opening at gripper 100, in mm.
2. Record the empty-closed reading: command gripper 0 and read it after 0.5 s.
3. Choose three objects and one obstacle. Rules: width across the grasp at most the jaw opening minus 2 cm; height 6 to 11 cm; a flat top that fits a 3 cm marker; under 100 g; nothing sharp, hot or fragile. Proposed: an empty pill bottle, an empty juice box, a narrow tea tin, and a cereal box about 20 cm tall as the obstacle.
4. Teleoperate 5 grasps per object and record the closed readings. Set `held_min` to half the gap between the smallest closed reading and the empty-closed reading (0 to 100 scale).
5. Write `config/objects.yaml` in the kit's format and give Lane B the final list so markers get printed.

**Done when:** `pytest -q tests/test_config.py` passes, and teleoperation holds each object 5 of 5 times.

## A4 · Driver with freeze (`arm/driver.py`)

**Needs:** A3. **By:** Sat 1:30 AM. **Files:** `arm/driver.py`, `tests/test_driver_fake.py`.

```python
# Newer LeRobot releases: from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", use_degrees=True, max_relative_target=5.0)
robot = SO101Follower(cfg)
robot.connect()
obs = robot.get_observation()          # {"shoulder_pan.pos": degrees, ..., "gripper.pos": 0-100}
robot.send_action({f"{j}.pos": v for j, v in zip(JOINTS, target)})
```

- `read() -> (q, gripper)`: 5 joint angles in radians and the gripper from 0.0 to 1.0, converted through `config/joint_map.yaml`.
- `command(q, gripper)`: clips each joint to its calibrated range minus 3° and to 2° per 30 Hz tick, whatever the caller asks. LeRobot's `max_relative_target` is only a second layer; confirm it clips on your installed version.
- `freeze()`: command the present position as the goal, keep torque on, and set the stopped flag. `command` is ignored until `resume()`.
- `torque_off()`: maintenance only, via `robot.bus.disable_torque()` (or `robot.disconnect()`, which also cuts torque by default).
- One lock around all serial access. `FAKE_ROBOT=1` swaps in an in-memory arm with the same methods.

**Done when:** `pytest -q tests/test_driver_fake.py` passes, and `python -m arm.driver --real --jog` moves each joint +10° and back at 30 Hz, while Esc pressed mid-move holds the arm where it is without sagging.

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

For each of the 5 arm joints, send the real arm to +20° on that joint with the others at 0, set the same joint to +0.349 rad in the viewer, and compare directions side by side. Record each joint's `sign` (+1 or -1) and `offset_deg`, plus the gripper's 0 to 100 range against the model's jaw joint range. Write the calibrated joint ranges, in radians, to `config/limits.yaml`. Then put the real arm in 5 poses spread over the workspace, tape-measure the tip from the origin, and compare with the model's forward kinematics from the same readings.

**Done when:** all 5 poses agree within 1.5 cm. A joint off by a constant needs its `offset_deg` fixed.

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
- `set_roll(angle)`: wrist_roll = angle - shoulder_pan + `roll_offset`. Measure `roll_offset` so the jaws line up with an object at yaw 0.
- Grasping without friction physics: while an object is held, after every `mj_step` set its free joint to the tip pose plus the offset captured at grasp time and zero its velocity. Attach only within the tolerance in Contracts.
- Collisions, limit hits, IK failures and the 60 s simulated-time limit (a `BaseException` subclass), as in Contracts.
- Object, obstacle and hand poses load from the scene JSON; `final_scene.json` is written from the end state.

**Done when:** `pytest -q tests/test_sim_robot.py` passes, and with `config/fake_scene.json`: `place_in_tray` on `juice_box` passes 5 of 5 trials, a candidate that drops the juice box beside the tray fails with a reason, and a candidate without `transit_z` fails with a `fail_reason` naming `cereal_box`.

## A9 · Replay engine (`arm/replay.py`)

**Needs:** A4, A6, S0. **By:** Sat 5:00 AM. **Files:** `arm/replay.py`, `tests/test_replay_fake.py`.

Implement `replay(traj)` per Contracts, Trajectory: bridging moves, 30 Hz streaming, grip checks against `held_min`, live primitives, STOP and hand checks every tick, the abort behavior and the return value. `validate` exists from S0; call it first and never move on a non-empty result.

**Done when:** `pytest -q tests/contract/test_trajectory_validate.py tests/test_replay_fake.py` passes (on the fake arm, STOP mid-path freezes, a missed grasp aborts and goes home, and a hand within 10 cm freezes), and `python -m arm.replay --real --traj <a sim-planned place_in_tray>` succeeds 4 of 5 times.

## A10 · Live primitives (`arm/real_robot.py`)

**Needs:** A9, B6. **By:** Sat 5:30 AM. **Files:** `arm/real_robot.py`, `config/rl.yaml`.

- `home()`: the tucked pose `home_q` in `config/scene.yaml`, chosen with Lane B so no calibration or object marker is hidden from the overhead camera.
- `freeze()`: the driver's freeze, plus cancelling any running primitive.
- `carry_to_hand(object)`: with the object held, move above the tracked hand plus 5 cm (or the `handoff` zone if the hand isn't seen), at 0.1 m/s at most within 15 cm of the hand. Wait until the hand stays within 8 cm for 1 s (after 10 s, use the `handoff` zone), open, back off 5 cm upward, go home and log `handoff_done`.
- Hand tracking uses repeated IK moves at 10 Hz, or the RL policy when `config/rl.yaml` says `use_policy: true` (D7 sets it once M4 passes).

**Done when:** `hand_over` on the pill bottle succeeds 4 of 5 times with the hand still, and `python -m arm.real_robot --real --carry-test` follows a hand moving at 5 cm/s or slower.

## A12 · Starter skills (`skills/_starter/`)

**Needs:** A8, A10. **By:** Sat 6:00 AM. **Files:** `skills/_starter/hand_over.json`, `skills/_starter/place_in_tray.json`.

Write both in the skill format with `trusted: true`:

- `hand_over(object)`: success `{"all": [{"moved": {"object": "$object", "min": 0.05}}, {"event": "handoff_done"}]}`; code: `pick`, then `carry_to_hand`.
- `place_in_tray(object)`: success `{"in_zone": {"object": "$object", "zone": "tray"}}`; code: `pick` and `place_in_zone("tray")`, both with `transit_z` set to the tallest obstacle plus 0.08 m.

**Done when:** both pass 5 of 5 trials with `python -m sim.run_candidate --skill skills/_starter/<name>.json --args '{"object": "juice_box"}' --scene config/fake_scene.json --trials 5 --seed 7 --out runs/check`, and each replays on the real arm 4 of 5 times.

## A13 · Safety layer

**Needs:** A9, A10. **By:** Sat 1:00 PM. **Files:** `arm/driver.py`, `arm/replay.py`, `arm/real_robot.py`, `tests/test_safety.py`.

Implement every code rule in Safety rules: the workspace box, speed caps (0.1 m/s within 15 cm of the hand), the 0.30 m ceiling, no gripper close within 10 cm of the wristband, and freeze on STOP from Esc, the dashboard and voice.

**Done when:** `pytest -q tests/test_safety.py` passes (out-of-box targets refused, speed caps hold, a close near the hand refused), and on the real arm Esc freezes motion within 0.2 s, measured from the key press to the freeze command in the log.

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
