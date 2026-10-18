# Operator runbook: from tonight's state to the judging table

One ordered procedure for the demo laptop (Windows 11, follower COM8 at 12 V, leader COM7 at 5 V). It starts
from the state on 2026-10-17: arm mapping (A6), grip (A11), table and camera calibration (B3 to B6) and the
live services are not done yet, and the leader's saved calibration differs from its motors. Each part needs
the one before it. If time runs short, cut from the end; Part 1 is a complete offline demo on its own.

Run everything from the repository root in PowerShell. Steps marked **MOVES THE ARM** need two people: one
with a hand on Esc, one holding the follower's 12 V plug, and the table clear. Every such command asks you
to type that the arm is clear, counts down 3 s and freezes the arm with torque on when Esc is pressed.
[tests/test_docs_runbook.py](../tests/test_docs_runbook.py) checks that every command below still exists
with the flags shown. Gate status lives in [gates.md](gates.md). For the arm and camera steps explained in
plain language (what each step is for, what you should see, what to do when it fails), read
[arm-guide.md](arm-guide.md).

## Part 1: offline demo and backup clip

1. Check the laptop. It opens no device or network connection.

   ```powershell
   .\.venv\Scripts\python.exe scripts/check_setup.py
   ```

2. Start the dashboard on an isolated copy of the library, with every fake on:

   ```powershell
   $env:FAKE = "1"; $env:DEMO_MODE = "1"
   $demoRoot = Join-Path (Get-Location) ("runs/demo-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
   $env:VARA_RUNS_DIR = Join-Path $demoRoot "runs"; $env:VARA_SKILLS_DIR = Join-Path $demoRoot "skills"
   New-Item -ItemType Directory -Path $env:VARA_SKILLS_DIR | Out-Null
   Copy-Item -LiteralPath skills/_starter -Destination $env:VARA_SKILLS_DIR -Recurse
   .\.venv\Scripts\python.exe -m agents.orchestrator --port 8000
   ```

   Open http://localhost:8000.

3. Rehearse the three-minute script in [demo.md](demo.md#three-minute-local-script).
4. Record the backup clip (Win+Alt+R): one continuous `clear the table` learn with the mode banner in frame.
   Save it as `runs/demo-video/backup.mp4` and keep a copy on a USB stick (gate M6).

## Part 2: power-on routine, every time before anything connects to the follower

1. Close the SenseCraft Robotics app; it holds COM7 and COM8.
2. Unplug the leader unless you are teleoperating. It stays unplugged during demos.
3. In its own PowerShell window, check the arm read-only, then make it hold its present pose:

   ```powershell
   (& 'C:\Users\jesse\miniforge3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | ?{$_} | Invoke-Expression; conda activate lerobot
   cd C:\Users\jesse\lerobot-tools
   python arm_check.py
   python sync_goals.py COM8
   ```

4. `.env` holds `FOLLOWER_PORT=COM8`, `LEADER_PORT=COM7`, `OVERHEAD_CAM` (see Part 4), `WRIST_CAM=1` and
   `FAKE=0`. Start every physical session in a new PowerShell window so no `FAKE` from Part 1 is set.

Teleoperation (A3) is optional. It needs the leader's calibration reconciled first; back up
`my_awesome_leader_arm.json` and recalibrate the leader only, in a visible console. Never recalibrate the
follower: every A6 result depends on its calibration file. Without teleoperation, pose the arm by hand with
torque off, then run `sync_goals.py COM8` again.

## Part 3: arm calibration (A6, A11, A13)

**Start inside the band.** The A6 jog and the A11 grip session refuse to start, with nothing moved, when a
joint lies more than 0.5 degrees outside `config/limits.yaml` minus 3 degrees, and name the joint. The rest
pose is outside the placeholder band, so pose the arm mid-range first. See
[hardware calibration](hardware-calibration.md).

1. Sign test, **MOVES THE ARM**. Open the model, move the same joint by +0.349 rad in the viewer, then jog
   the real joint by +20 degrees and compare the direction. Repeat for `shoulder_lift`, `elbow_flex`,
   `wrist_flex` and `wrist_roll`, and write down five signs (same direction 1, opposite -1).

   ```powershell
   .\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=sim/model/scene_ours.xml
   .\.venv\Scripts\python.exe -m arm.driver --real --jog-joint shoulder_pan --deg 20 --hold 10
   ```

2. Reference pose. Hold a pose you can square up and read its five model values off the viewer, then:

   ```powershell
   .\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/reference-pose.json
   ```

3. Solve. Write `runs/calibration/solve.json` as in [joint calibration](joint-calibration.md#3-solve-the-offsets-and-limits),
   run the solver and paste its two YAML blocks into `config/joint_map.yaml` and `config/limits.yaml`.

   ```powershell
   .\.venv\Scripts\python.exe -m arm.check_mapping --solve runs/calibration/solve.json
   ```

4. Five-pose check (A6 Done-when). Tape-measure the jaw tip in five spread poses, capture `pose-1.json` to
   `pose-5.json` with `arm.inspect_hardware` as in step 2, fill the template and check. Every pose must be
   within 1.5 cm.

   ```powershell
   .\.venv\Scripts\python.exe -m arm.check_mapping --example --out runs/calibration/measurements.json
   .\.venv\Scripts\python.exe -m arm.check_mapping runs/calibration/measurements.json --out runs/calibration/check-01.json
   ```

5. Gripper (A11), **MOVES THE ARM**. Paste the printed lines into `config/joint_map.yaml` and
   `config/objects.yaml`, then run the config tests. A session ended by Esc prints no paste lines.

   ```powershell
   .\.venv\Scripts\python.exe -m arm.driver --real --measure-grip --objects pill_bottle juice_box tea_tin --grasps 5
   .\.venv\Scripts\python.exe -m pytest -q tests/test_config.py
   ```

6. Esc drill (A13), **MOVES THE ARM**. Press Esc while the arm moves, then read the verdict. Only
   session-phase records count against the 200 ms bar; an Esc pressed while the driver connects is kept as a
   diagnostic.

   ```powershell
   .\.venv\Scripts\python.exe -m arm.driver --real --jog --deg 10
   .\.venv\Scripts\python.exe -m arm.driver --esc-latency-report
   ```

## Part 4: camera and table (B1, B3 to B6); nothing here moves the arm

1. Find the camera indexes. One picture per index lands under `runs/setup/`; put the overhead index in
   `OVERHEAD_CAM`. Index 0 is the laptop webcam.

   ```powershell
   .\.venv\Scripts\python.exe -m vision.cameras --list
   ```

2. Print the markers at 100 % scale and measure their printed sides. Glue 10 on the pill bottle, 11 on the
   juice box, 12 on the tea tin and 30 on the cereal box; tape 0 to 7 around the workspace; 40 goes on the
   wristband; keep the spare cards 13 to 17 loose for I2.

   ```powershell
   .\.venv\Scripts\python.exe scripts/make_markers.py
   ```

3. Mount the overhead camera rigidly. Tape-measure its lens height `Hc`, the table point under the lens `c`
   and the centers of markers 0 to 7 into `config/scene.yaml`, and remove the word placeholder from its
   comments.
4. Tape layout L1 (object spots, tray, cereal box, hand rest), starting from the table at the end of
   [demo.md](demo.md#physical-setup-still-to-complete).
5. Visibility (B3, B4), with the arm at home and every object on its spot:

   ```powershell
   .\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-1 --expect-all --frames 10
   ```

6. Five check spots (B5). Lay loose marker 10 flat on five tape-measured spots, one capture each, with that
   spot's measured `X Y` in meters (`spot-2` to `spot-5` for the others):

   ```powershell
   .\.venv\Scripts\python.exe scripts/capture_table.py --out runs/table-calibration/spot-1 --check-point 0.21 -0.08
   ```

7. Fit the homography: preview, then save. Mean error at most 1.0 cm and maximum at most 1.5 cm. If the edges
   stay worse, use the lens fallback in [table calibration](table-calibration.md#lens-calibration-b5-fallback).

   ```powershell
   .\.venv\Scripts\python.exe -m vision.calibrate --live --no-save --check-points measurements/check-points.json
   .\.venv\Scripts\python.exe -m vision.calibrate --live --check-points measurements/check-points.json
   ```

8. Accuracy (B6). Tape-measure the objects into `measurements/scene-layout.json` (shape in
   [table calibration](table-calibration.md)); every item within 1.5 cm and 10 degrees passes.

   ```powershell
   .\.venv\Scripts\python.exe -m vision.scene --accuracy measurements/scene-layout.json
   ```

9. Camera soak (B1), optional for the demo; it holds both cameras for 30 minutes. `--criterion strict` is
   B1's bar; `tracking` passes with no failed reads, no gap over 0.5 s and at least 25 fps.

   ```powershell
   .\.venv\Scripts\python.exe -m vision.cameras --check 1800 --criterion tracking
   ```

## Part 5: arm meets camera (I2, A9, A10, A12, I3)

1. Touch test (I2), **MOVES THE ARM**. Lay spare cards 13 to 17 flat on the table. The objects may stay: the
   ids `config/objects.yaml` assigns (10, 11, 12) are left out because those cards are glued on top of objects,
   and a hover planned from the table there would drive the tip into the object. Every gap must be at most
   1.5 cm; otherwise fix A6 or the homography first.

   ```powershell
   .\.venv\Scripts\python.exe scripts/touch_markers.py --real --loose
   ```

2. Physical orchestrator without keys, **MOVES THE ARM**. Fixture model replies and local scoring, the real
   arm and cameras. Type `put the juice box in the tray` and `hand me my pills` a few times each (A9, A10,
   A12). Esc in this console or STOP on the dashboard freezes the arm.

   ```powershell
   $env:FAKE_LLM = "1"; $env:FAKE_SANDBOX = "1"
   .\.venv\Scripts\python.exe -m agents.orchestrator --real --port 8000
   ```

3. First real learn (I3, M1). The same command with `XAI_API_KEY` in `.env` and `FAKE_LLM` removed from the
   window. A new task must end in `saved`.

The standalone `--real` tools (`arm.replay`, `arm.real_robot`, `rl.policy_runner`) take the hand from
`--tracker`, which opens the overhead camera itself, or from `--scene-file`. A scene file must come from a
live writer running in its own window; the tools refuse a file without the writer's live provenance
sidecar (`scene.json.source.json`), one written from a fake camera, or one with no fresh scene, and a
session freezes if the writer stops or restarts:

```powershell
.\.venv\Scripts\python.exe -m vision.scene --out runs/live/scene.json
.\.venv\Scripts\python.exe -m arm.real_robot --real --carry-test --object pill_bottle --scene-file runs/live/scene.json
```

## Part 6: live services (C1 to C8, A14); costs model calls and sandboxes

1. Put `XAI_API_KEY` and `DAYTONA_API_KEY` in `.env`. They are never printed.
2. Build the Daytona snapshot once per account; it is slow.

   ```powershell
   .\.venv\Scripts\python.exe -m agents.sandbox_pool --build-snapshot arm-sim-v1
   ```

3. Run the live gates: model ping, router, candidates, Daytona smoke, the 7-sandbox bench, the saved-skill
   timing and the voice token. Evidence lands in `runs/live-gates/<stamp>/summary.json`; afterwards no
   sandbox may remain. See [live gates](live-gates.md).

   ```powershell
   .\.venv\Scripts\python.exe scripts/live_gates.py --dry-run
   .\.venv\Scripts\python.exe scripts/live_gates.py
   .\.venv\Scripts\python.exe -m agents.sandbox_pool --list
   ```

4. Voice (A14, M3). With the orchestrator running, open http://localhost:8000/voice, hold T to talk, and test
   a task and "stop" in the noisiest corner you can find.

RL on the real arm is the first cut in [schedule.md](schedule.md). Keep `use_policy: false` in
`config/rl.yaml` and show the simulation video and curves. The staged bring-up exists for later
(`python -m rl.policy_runner --fake --stage slow --tracker`); its offline run shows the policy steering the jaw
tip, not the grasp center the M4 verdict measures, so M4 does not pass yet.

## Part 7: rehearse, freeze, submit, judge

1. Tune the scene (A15). With the physical orchestrator and `DEMO_MODE=1`, run `clear the table` five times,
   then move the cereal box 3 cm as the verdict says until 2 to 4 of 6 candidates pass.

   ```powershell
   .\.venv\Scripts\python.exe scripts/rehearsal_report.py
   ```

2. Save the best physical run as the stage backup, with its provenance note:

   ```powershell
   .\.venv\Scripts\python.exe -m agents.library --save-backup clear_table --note "learned on the real arm, layout L1" --keep-real --force
   ```

3. M5: the full script three times in a row, with the
   [physical reset checklist](demo.md#physical-reset-30-seconds) before each.
4. M6 (1:00 AM): two clean runs, then re-record `runs/demo-video/backup.mp4` with the physical arm.
5. M7 (Devpost): gather the numbers, fill the slots at the top of [devpost-draft.md](devpost-draft.md),
   upload the video, make the repository public and select both sponsor challenges. Make It Legendary
   still needs Cursor screenshots.

   ```powershell
   .\.venv\Scripts\python.exe scripts/story_numbers.py --markdown
   ```

6. M8 (judging table): the physical reset checklist, Check calibration on the dashboard, `DEMO_MODE=1`,
   Reset demo pressed, voice tested in the room.
