# Calibrate and run the arm: a step-by-step guide

This guide takes you from a freshly plugged-in arm to typing commands on the dashboard and watching the
arm do them. It explains what each step is for, what to type, what you should see and what to do when a
step fails. It assumes the demo laptop (Windows 11) with this repository at `C:\Users\jesse\GitHub\VARA`.

[runbook.md](runbook.md) is the short checklist version that also covers the offline demo, the cloud
services and the submission. This guide goes deeper on the arm and the camera.

## Part A: before you start

### Words used in this guide

- **Follower:** the robot arm that does the work. USB port COM8, 12 V power supply.
- **Leader:** the second, hand-held arm used to steer the follower (teleoperation). USB port COM7, 5 V
  supply only. Keep it unplugged unless you are teleoperating.
- **Torque on / off:** with torque on, the motors hold their position and the arm is stiff. With torque
  off, the arm is limp and can be moved by hand (and falls if nobody holds it).
- **Joint:** one motor. The five arm joints are `shoulder_pan` (turns the whole arm left and right),
  `shoulder_lift` (tilts the upper arm), `elbow_flex` (bends the elbow), `wrist_flex` (tilts the gripper up
  and down) and `wrist_roll` (spins the gripper). The sixth motor opens and closes the gripper.
- **Model:** the 3D simulation of the arm in MuJoCo. The software plans every move in the model first.
- **Joint map:** the translation between the motor readings and the model's joint angles: one direction
  (sign) and one offset per joint. It lives in `config/joint_map.yaml`. Until it is measured, the model and
  the real arm disagree, so nothing that plans moves may run.
- **Limits and the safe band:** each joint's allowed range lives in `config/limits.yaml`. The software never
  commands a joint closer than 3 degrees to either end; that inner range is the safe band.
- **Markers:** printed black-and-white square codes the overhead camera recognises.
- **Homography:** the measured translation from a camera pixel to a spot on the table, in
  `config/homography.npy`. The current file is synthetic, not measured.
- **Esc / STOP:** freezes the arm where it is, with torque on. It never drops the arm.

### People

Every step that moves the arm needs two people:

- **The operator** types the commands and keeps a finger on Esc. Esc works in the window that has keyboard
  focus (the console window that started the command, or the dashboard tab).
- **The plug holder** keeps a hand on the follower's 12 V plug. Pulling it is the hard stop: the arm goes
  limp, so be ready to catch it.

### Equipment

- Tape measure and a ruler with millimeters; a small spirit level or set square.
- Masking tape and a marker pen.
- The printed markers (Part D shows how to print them).
- The objects: pill bottle, juice box, tea tin, cereal box, tray.

### Rules that always apply

1. Nobody's hands inside the taped box while the arm can move.
2. Anything that moves the arm needs `--real`. It then asks you to type `yes`, counts down 3 s, and only then
   connects. Esc during the countdown cancels with nothing moved.
3. Never recalibrate the follower with LeRobot's calibration tool. Every measurement in Part C depends on its
   current calibration file. If it ever gets recalibrated, redo all of Part C.
4. Never command all joints to 0. LeRobot's 0 is the middle of each joint's range, a big swing from rest.
5. Only one program can use the follower's port (COM8) or a camera at a time. Close the SenseCraft Robotics
   app, and stop the dashboard server before running the standalone tools below.
6. Run the arm commands in a new PowerShell window in the repository folder, so no `FAKE` setting from an
   offline demo is active:

   ```powershell
   cd C:\Users\jesse\GitHub\VARA
   ```

## Part B: power on, pose by hand, power off

### B1. Power-on check (every session)

**Why:** a motor can hold an old target from a previous session. Switching torque on would make it jump
straight there (up to 168 degrees was measured). These two scripts check the arm and make every motor's
target equal its present position first.

1. Plug in the follower's USB and 12 V. Leave the leader unplugged.
2. Open a separate PowerShell window and start the `lerobot` environment:

   ```powershell
   (& 'C:\Users\jesse\miniforge3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | ?{$_} | Invoke-Expression; conda activate lerobot
   cd C:\Users\jesse\lerobot-tools
   python arm_check.py
   ```

   You should see all six motors on COM8 at about 12 V. `arm_check.py` only reads; it moves nothing.
3. Make the arm hold its present pose:

   ```powershell
   python sync_goals.py COM8
   ```

   The arm stiffens where it is. If it prints `ABORT: torque is on and target != position`, something is
   already driving the arm: find the program holding COM8 and close it.
4. Check that `.env` in the repository has `FOLLOWER_PORT=COM8` and `FAKE=0`.

Keep this `lerobot` window open; you will use `sync_goals.py` again.

### B2. Pose the arm by hand

Several steps ask you to put the arm in a pose by hand. The motor settings are cleared when the power goes
off, so the simplest way is a power cycle:

1. Person 1 holds the arm firmly.
2. Person 2 pulls the 12 V plug. The arm goes limp in person 1's hands.
3. Person 1 moves the arm into the pose.
4. Person 2 plugs the 12 V back in. The motors come back limp, so person 1 keeps holding.
5. Run `python sync_goals.py COM8` in the `lerobot` window. The arm now holds the pose and person 1 can let
   go slowly.

### B3. Power off at the end

1. If a dashboard server is running, let the task finish and press Resume so the arm goes home, then stop the
   server with Ctrl+C. The arm keeps holding its pose with torque on.
2. Person 1 supports the arm. Person 2 pulls the 12 V plug. Lower the arm gently onto its rest.

## Part C: calibrate the arm

Do the steps in order. Steps C1 to C5 measure the joint map, C6 the gripper, C7 the emergency stop. Until C5
passes, only the commands in this part may move the arm.

### C0. Start from the middle of the joints' travel

**Why:** the jog and gripper tools re-send the starting pose with every command, and the driver pulls any
joint that is outside the safe band back into it. So both tools refuse to start when a joint is outside the
band, and tell you which joint and by how much. The arm's resting pose is outside the band on three joints
(about 7, 3 and 10 degrees).

Pose the arm by hand (B2) roughly in the middle of every joint's travel: upper arm up, forearm and gripper
pointing forward, gripper in its middle roll position. If a later command says
`the start pose lies outside the commandable band ... on elbow_flex 4.2 deg` (for example), move that joint
a little more than that toward the middle and run `sync_goals.py` again.

### C1. Which way does each joint turn? (the sign test)

**Why:** a motor can count its angle the opposite way to the model. You find out by turning each joint
+20 degrees and comparing with the model.

In the model, from the pose in C0, +20 degrees looks like this:

| Joint | +20 degrees in the model |
| --- | --- |
| `shoulder_pan` | The whole arm turns toward its own right (clockwise seen from above). |
| `shoulder_lift` | The upper arm nods forward and down. |
| `elbow_flex` | The forearm nods down. |
| `wrist_flex` | The gripper nods down. |
| `wrist_roll` | The gripper spins; check the direction in the viewer. |

To see it for yourself, open the model and drag the joint's slider in the right-hand panel:

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=sim/model/scene_ours.xml
```

Now turn the real joint by +20 degrees. It holds there for 10 s and goes back. **This moves the arm.**

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --jog-joint shoulder_pan --deg 20 --hold 10
```

Type `yes`, wait for the countdown, watch which way the joint turns. Write down `1` if it matches the model
and `-1` if it turns the other way. Repeat with `shoulder_lift`, `elbow_flex`, `wrist_flex` and
`wrist_roll`. Only the named joint moves; if anything else moves, press Esc.

### C2. Hold a known pose and read the motors (the reference pose)

**Why:** to find each joint's offset, the software needs one pose where both the motor readings and the
model's angles are known. The model's zero pose is easy to set by hand:

- the arm points straight forward (not turned left or right);
- the forearm and the gripper are level, pointing straight forward;
- the jaws open up and down (not sideways);
- the jaw tip is 35.7 cm in front of the base's turning axis and 22.8 cm above the table.

Steps:

1. Close the jaws by hand and pose the arm as above (B2). Use the level on the forearm. Adjust the upper
   arm until the closed jaw tip is 22.8 cm above the table while the forearm stays level.
2. Write down where the jaw tip really is: x forward from the point on the table under the base's turning
   axis, y to the arm's left, z up, all in meters.
3. Read the motors. This only reads; nothing moves:

   ```powershell
   .\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/reference-pose.json
   ```

   Use a new file name if that file already exists; evidence files are never overwritten.

### C3. Work out the offsets and limits (the solve)

1. Create `runs/calibration/solve.json` with your five signs from C1 (the example shows all 1; use yours):

   ```json
   {
     "hardware_report": "reference-pose.json",
     "model_q_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
     "signs": {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1},
     "note": "model zero pose, forearm levelled, tip measured at (x, y, z) by NAME"
   }
   ```

2. Run the solver. It reads files only:

   ```powershell
   .\.venv\Scripts\python.exe -m arm.check_mapping --solve runs/calibration/solve.json
   ```

3. It first prints where the model puts the tip for that pose: about (0.357, 0.000, 0.228). Compare with what
   you measured in C2. If they differ by more than about 1 cm, the pose was not the zero pose: redo C2.
4. It then prints two blocks of text. Open `config/joint_map.yaml` and replace its five joint lines with the
   first block (keep the `lerobot_id`, `lerobot_robot_type` and `gripper` lines). Open `config/limits.yaml`
   and replace its five joint lines with the second block.
5. A warning that a range reaches beyond the model's range is expected on this arm (four joints do).

### C4. Check the signs again

Repeat the C1 jog for each joint. Now every joint should turn the same way as the model. If one turns the
other way, flip its sign in `solve.json`, rerun C3 with the same report, and paste again.

### C5. Accuracy check with five poses

**Why:** this is the test that the joint map is right everywhere, not only at the reference pose. Each of
five spread-out poses must agree with the model within 1.5 cm.

1. Create the measurement sheet:

   ```powershell
   .\.venv\Scripts\python.exe -m arm.check_mapping --example --out runs/calibration/measurements.json
   ```

2. For each of five different poses (reach left, reach right, reach far, close to the base, high up):
   - Close the jaws by hand, pose the arm (B2) and let it hold.
   - Tape-measure the jaw tip: x, y and z in meters from the point on the table under the base's turning
     axis (x forward, y to the arm's left, z up).
   - Read the motors without changing the pose (`pose-2.json` to `pose-5.json` for the others):

     ```powershell
     .\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/pose-1.json
     ```

   - Type the three numbers into that pose's `tip_xyz` in `runs/calibration/measurements.json`.
3. Run the check:

   ```powershell
   .\.venv\Scripts\python.exe -m arm.check_mapping runs/calibration/measurements.json --out runs/calibration/check-01.json
   ```

**Pass:** exit 0, every pose within 0.015 m. **If one joint is off by a constant amount:** redo C2 and C3
with a more carefully squared reference pose. **If the error grows the further a pose is from the reference
pose:** a sign is wrong; redo C4.

### C6. Measure the gripper

**Why:** the software needs to know how far the jaws close on each object, to tell a held object from a
missed grab. Only the jaws move in this step; the rest of the arm holds still. **This moves the gripper.**

Pose the arm so its gripper is easy to reach (B2), then run:

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --measure-grip --objects pill_bottle juice_box tea_tin --grasps 5
```

Answer its questions:

1. `jaw opening at gripper 1.0 in mm`: measure the gap between the open jaw tips with the ruler.
2. It closes on nothing and opens again by itself.
3. For every object, five times: `place pill_bottle between the jaws, Enter`. Hold the object between the
   jaws, press Enter, and the jaws close and hold gently. `held? y/n`: tug the object lightly and answer.

At the end it prints lines to paste: the `gripper:` line into `config/joint_map.yaml`, and a `held_min:` for
each object into its entry in `config/objects.yaml`. Then check the files:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py
```

If you press Esc during the session, the arm freezes, the report is kept as partial, and no paste lines are
printed. Run it again to the end.

### C7. Emergency stop drill

**Why:** Esc must freeze a moving arm within 0.2 s. **This moves the arm.**

Start a jog and press Esc while the joint is moving:

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --jog-joint shoulder_pan --deg 20 --hold 10
```

The arm freezes and stays stiff. The command prints `Esc -> freeze command: N ms` and saves the number. Press
Enter to release the port; the arm keeps holding. Do this a few times, then read the verdict:

```powershell
.\.venv\Scripts\python.exe -m arm.driver --esc-latency-report
```

**Pass:** `PASS`, every timed press at most 200 ms. An Esc pressed while the arm is still connecting is
listed separately and does not count, because nothing was moving yet.

### C8. First real move: go home

The joint map now matches the model, so the arm can go to its home pose (folded up against the base, jaws
level). It moves at 30 degrees per second at most. **This moves the arm.**

```powershell
.\.venv\Scripts\python.exe -m arm.real_robot --real --home
```

It warns that it has no hand data; that is fine for this step. If the arm heads toward anything, press Esc.

## Part D: calibrate the camera and the table

Nothing in this part moves the arm. [table-calibration.md](table-calibration.md) has every detail.

### D1. Find the overhead camera

```powershell
.\.venv\Scripts\python.exe -m vision.cameras --list
```

It saves one picture per camera under `runs/setup/`. Index 0 is the laptop's own webcam. Put the overhead
camera's number in `.env` as `OVERHEAD_CAM`.

### D2. Print the markers

```powershell
.\.venv\Scripts\python.exe scripts/make_markers.py
```

Print the sheets at 100 % scale (never "fit to page") and check one marker's side with the ruler. Then:

- glue marker 10 on the pill bottle, 11 on the juice box, 12 on the tea tin and 30 on the tape measure (the
  obstacle, standing on its edge; mount its 4 cm marker flat on a small card). On a box, glue the marker so
  its printed label reads along the long side: the jaws close across the short side. The tape measure keeps
  the slot name `cereal_box`. Starting spots are in [demo.md](demo.md); the "tray" is a taped drop area;
- tape markers 0 to 7 flat around the edge of the workspace;
- put marker 40 on the wristband for the hand-over;
- keep the spare cards 13 to 17 loose for the arm-meets-camera test in Part E.

### D3. Mount the camera and tape the table

1. Mount the overhead camera rigidly, looking straight down at the workspace.
2. Measure the camera's lens height above the table (`Hc`) and the table point straight below the lens
   (`c`, x and y in meters, same axes as before). Use a plumb line or a level.
3. Tape-measure the center of each of markers 0 to 7.
4. Write all of this into `config/scene.yaml` (`camera:` and `calib_markers:`), and delete the word
   "placeholder" from that file's comments.
5. Tape the spots for the objects, the tray, the cereal box and the hand rest (starting layout: the table at
   the end of [demo.md](demo.md)).

### D4. Can the camera see every marker?

Arm at home, every object on its spot marker side up, markers 0 to 7 uncovered:

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-1 --expect-all --frames 10
```

**Pass:** no missing ids. Otherwise it lists the missing ones: uncover or re-light them and capture again into a
new folder.

### D5. Five check spots

**Why:** to prove the camera's table positions are right, you compare them with tape measurements at five
spots that were not used for the fit.

Lay loose marker 10 flat on the table (object removed) at a spot, tape-measure its center, and capture with
those numbers (x then y, in meters). Repeat at four more spots spread over the table, each with a new folder:

```powershell
.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/table-calibration/spot-1 --check-point 0.21 -0.08
```

It records the pixel and your measurement in `measurements/check-points.json` and prints `N of 5 recorded`.

### D6. Fit the camera-to-table translation

Preview first; it saves nothing:

```powershell
.\.venv\Scripts\python.exe -m vision.calibrate --live --no-save --check-points measurements/check-points.json
```

**Pass:** mean error at most 1.0 cm and maximum at most 1.5 cm. Then save it:

```powershell
.\.venv\Scripts\python.exe -m vision.calibrate --live --check-points measurements/check-points.json
```

If the spots near the edges stay worse, the lens bends the picture: follow the lens section of
[table-calibration.md](table-calibration.md#lens-calibration-b5-fallback).

### D7. Accuracy of the object positions

Write the tape-measured object positions into `measurements/scene-layout.json` (meters; `yaw` is the
direction of a box's marker in radians, `null` for round objects):

```json
{"items": {"pill_bottle": {"x": 0.18, "y": -0.12, "yaw": null},
           "juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3},
           "cereal_box": {"x": 0.28, "y": 0.07, "yaw": 0.0}}}
```

```powershell
.\.venv\Scripts\python.exe -m vision.scene --accuracy measurements/scene-layout.json
```

**Pass:** every item within 1.5 cm and 10 degrees.

## Part E: does the arm go where the camera says?

**Why:** this joins Part C and Part D. The arm hovers 5 cm above five markers the camera found, and you
measure how far off it is. **This moves the arm.**

1. Lay the spare cards 13 to 17 flat on the table, spread out. The objects can stay where they are: their
   markers are glued on top of them, so the tool leaves 10, 11 and 12 out on purpose (a hover planned from
   the table there would hit the object).
2. Run:

   ```powershell
   .\.venv\Scripts\python.exe scripts/touch_markers.py --real --loose
   ```

3. For each card the arm hovers above it and asks for the gap. Measure from the jaw tip straight down to the
   card's center with the ruler and type it in cm (Enter keeps the software's own estimate).

**Pass:** every gap at most 1.5 cm. **If not:** fix the joint map (Part C) or the camera fit (D6) before
going further.

## Part F: run the arm

### F1. Start the dashboard in physical mode

Close any standalone tool that holds the arm or the camera. In a fresh PowerShell window:

```powershell
$env:FAKE_LLM = "1"; $env:FAKE_SANDBOX = "1"
.\.venv\Scripts\python.exe -m agents.orchestrator --real --port 8000
```

`FAKE_LLM=1` uses stored model replies for the demo phrases and `FAKE_SANDBOX=1` scores programs on this laptop,
so no API keys are needed. To use the live model and cloud scoring instead, put `XAI_API_KEY` and
`DAYTONA_API_KEY` in `.env`, build the Daytona snapshot once ([runbook.md](runbook.md), Part 6), and leave
both settings out.

It asks `Type yes when the arm is clear`, counts down 3 s and connects. **Support the arm while it
connects:** LeRobot briefly switches torque off during the connect, so the arm can sag. When it prints
`arm connected; Esc in this console or STOP on the dashboard freezes it`, open http://localhost:8000.

### F2. Read the dashboard

- The mode banner must read `Robot: physical` and `Cameras: physical`. If it says `simulation`, a fake
  setting is still on. If it says `physical, disconnected`, the arm did not connect.
- The overhead video draws every marker it sees: 10, 11, 12 and 30 should be drawn.
- Press **Check calibration**: it should answer with an error of at most 1.0 cm.

### F3. Give it something to do

Type into the command box:

- `put the juice box in the tray`: the arm picks up the juice box and puts it in the tray.
- `hand me my pills`: wear the wristband (marker 40) and hold your hand still above the hand rest; the arm
  brings the pill bottle and opens near your hand.
- `clear the table`: a new task the first time. Press Reset demo before it, so the library holds only the
  two starter skills (a saved `clear_table` would be replayed instead of learned). The dashboard shows the
  success test, six candidate programs scored in simulation, the winner's checked path, then the real move
  and the camera's verdict, and saves the skill. The second time it runs the saved skill straight away.

Put the objects back on their spots between commands.

### F4. Stop and resume

- **Stop:** press Esc (with the dashboard tab or the server window focused), click STOP, or say "stop" on the
  voice page. The arm freezes with torque on and the banner reads `STOPPED - arm frozen with torque on`.
- **Resume:** take every hand out of the taped box first, then press Resume. The arm goes home at 30 degrees
  per second at most.
- **Torque off** is a maintenance button, not a stop: the arm goes limp. It needs two clicks and is refused
  while a task runs. Only use it while someone holds the arm.

### F5. Between demo runs

Follow the 30-second checklist in [demo.md](demo.md#physical-reset-30-seconds): arm home and stiff, objects
on their spots, markers drawn, Check calibration, Reset demo, plug holder and Esc ready.

### F6. End the session

Do B3: arm home, Ctrl+C in the server window, support the arm, pull the 12 V plug, lower it to its rest.

## Part G: when something goes wrong

| What you see | What it means | What to do |
| --- | --- | --- |
| `the start pose lies outside the commandable band ... on JOINT N deg` | A joint is too close to the end of its range. | Move that joint more than N degrees toward the middle (B2) and try again. |
| `--real with FAKE=1 set: refusing` or `Refusing to touch the arm without --real` | A fake setting is on, or `--real` is missing. | Open a new PowerShell window; check `FAKE=0` in `.env`; add `--real`. |
| An error that COM8 is busy or access is denied | Another program holds the port. | Close SenseCraft, the dashboard server or any other arm tool. |
| `ABORT: torque is on and target != position` from `sync_goals.py` | Something is driving the arm right now. | Find and close the other program, then run it again. |
| The arm sags for a moment when a command connects | LeRobot switches torque off briefly while connecting. | Expected: support the arm during the connect. |
| Esc does nothing | The window you are typing in does not have keyboard focus. | Click into the console window (or the dashboard tab) first. Pull the 12 V plug if the arm is moving. |
| `refusing --scene-file ...` | The scene file did not come from the live camera writer, or it is stale. | Use `--tracker`, or run `python -m vision.scene --out FILE` in its own window. |
| Check calibration gives an error, or above 1.0 cm | The camera or the markers moved. | Fix the cause and redo D6 (and D5 if the camera moved). |
| Touch-test gaps over 1.5 cm | The joint map or the camera fit is off. | Redo C5; if it passes, redo D5 and D6. |

## Part H: what to redo after a change

| Change | Redo |
| --- | --- |
| Follower recalibrated with LeRobot | All of Part C, then Part E. |
| Arm moved on the table, or table re-taped | D3 to D7, then Part E. |
| Camera bumped or remounted | D3 (camera numbers) to D7, then Part E. |
| A marker re-printed or moved | D2 and D4; for markers 0 to 7 also D3 and D6. |
| New objects | C6 for those objects, D7. |
