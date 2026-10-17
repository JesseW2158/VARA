# Demonstration and reset

The verified presentation runs the full software workflow with local MuJoCo scoring, fixture model
replies, and a fake arm/camera. Keep that mode visible. Physical motion, live voice, and cloud calls
have their own acceptance gates; the connected kit alone does not establish them.

## Start an isolated offline presentation

Follow the Windows dashboard commands in [README](../README.md#run-without-devices-or-service-calls).
They set `FAKE=1`, `DEMO_MODE=1`, and separate skill/run directories under a timestamped `runs/demo-*`
folder. Start with `hand_over` and `place_in_tray` only. Use the typed command box: a fake voice token
cannot run a live voice conversation.

For a terminal-only rehearsal:

```powershell
.\.venv\Scripts\python.exe scripts/e2e_fake.py "clear the table"
```

That script makes its own copied library. To measure the demo budget through the dashboard backend,
keep the isolated environment paths from the README, stop the server, then run:

```powershell
.\.venv\Scripts\python.exe -m agents.orchestrator --bench-demo --runs 5
```

The benchmark resets the configured library between commands. Its measured local result was five
successful executions followed by `saved`/`done ok`: 8.6, 10.2, 8.9, 9.9, and 11.8 seconds to execution.
Candidates 3 and 4 passed all five trials each; the other four were rejected. These timings use fixture
replies and fake replay, not live services or physical motion.

The dashboard was also exercised in actual Edge: a typed clear-table command reached `saved`, all six
candidate cards appeared, the 640-pixel-wide RL video loaded, and Esc stopped the fake run. Browser
evidence is in `runs/verification/browser-final/1792266692/result.json` and `dashboard.png`. This checks
the browser flow; it does not measure physical STOP latency.

## Three-minute local script

| Time | Action | Show and explain |
| --- | --- | --- |
| 0:00-0:20 | Point to the mode banner and stationary hardware | "This is our working local software demo. The physical kit still needs measured joint and camera calibration." |
| 0:20-1:00 | Type `clear the table` | The success test appears before six programs are scored. Show a cereal-box collision, the passing candidates, validation, and the saved skill. |
| 1:00-1:25 | Type `clear the table` again | The library supplies a saved program; it is re-planned and validated for the scene. |
| 1:25-1:45 | Type `Fold my laundry` | Show the queue entry without promising execution. |
| 1:45-2:10 | Show the PPO video and provenance | "Simulation only. The selected baseline achieved 97/100 successes, zero falls, seed 0." |
| 2:10-2:40 | Show one run folder or the architecture diagram | The same success predicate scores candidates and verifies the result. Programs never grade themselves or receive the physical driver. |
| 2:40-3:00 | Close on the library and next steps | Physical mapping, overhead calibration, grip tests, STOP latency, and live services are explicit remaining work. |

For a longer presentation, define `When I say good morning, hand me my pill bottle, then my juice box`,
then type `Good morning`. This routine passed local scoring and fake replay. Describe it as such.

If a command queues, show its reason. If the software errors, keep the failed run folder and use the
recorded local clip, labelled as a prior run. Load backup is appropriate only when an actual rehearsal
backup exists; do not imply that an unverified backup establishes physical acceptance.

## Offline reset, about 30 seconds

1. Wait for the task to finish. If intentionally testing fake STOP, wait for cancellation and use Resume.
2. Press Reset demo in the isolated dashboard. Confirm exactly `hand_over` and `place_in_tray` remain.
3. Confirm the mode banner still says simulation/fixtures and the run paths point to the demo copy.
4. Check the synthetic camera view and reload the RL clip if it is paused or at its end.
5. Clear the typed command field and rehearse the opening mode statement.

Reset demo changes the skill library; it does not calibrate hardware or physically reset objects.
The fake calibration check returns a fixture result and is not a table measurement. On a real arm,
Resume can move toward home, so the offline reset steps are not a physical recovery procedure.

## Physical setup still to complete

Use [hardware calibration](hardware-calibration.md), [setup inspection](setup.md), the
[A-lane tasks](tasks/A-arm.md), and [safety rules](safety.md). Record the five measured tip poses for A6,
grip/contact readings, overhead-camera calibration, and the physical STOP drill before rehearsal.
The leader's saved calibration currently differs from its motors; reconcile it before teleoperation.
Keep the leader unplugged during an actual follower demo as required by the safety rules.

The later 2026-10-17 check opened configured camera 2 and wrist camera 1: five seconds, 148 frames
each, no failed or late frames. The earlier camera-2 open failure is historical; camera 0 was identified
as the laptop webcam. Evidence is in `runs/setup/cameras/configured-check-20261017.log` and
`runs/setup/cameras/current/`. Camera 2 currently looks across the table from a low position; the wrist
view is mostly dark/occluded. Neither capture detected markers. Mount and measure the overhead camera,
clear the wrist view, and complete the 30-minute soak before claiming B1/B4 acceptance.
The later soak attempt failed early with late frames. Independent readers then measured 172 late
frames from camera 2 over one minute and none from the wrist feed. Stabilize the physical setup and
repeat the full soak; the short initial pass does not override that failure.

The checked-in homography is synthetic. Print `runs/setup/markers/markers-a4.pdf` at actual size (100%),
then measure the marker edges with a ruler before mounting them; that local PDF contains 13 markers
on two A4 pages. The marker source can be regenerated with `python scripts/make_markers.py`.

The current **simulation** layout below replaces older coordinates in this demo guide. Treat it as a
starting arrangement to measure and test physically, not as an accepted calibration. Units are meters;
+x points away from the arm, +y to its left, and z is up from the table.

| Item | x | y | Note |
| --- | --- | --- | --- |
| Pill bottle, marker 10 | 0.18 | -0.12 | Empty prop |
| Juice box, marker 11 | 0.24 | -0.04 | Yaw 0.3 rad in the fixture |
| Tea tin, marker 12 | 0.16 | -0.02 | Current fixture location |
| Cereal box, marker 30 | 0.28 | 0.07 | Height 0.20 |
| Tray center | 0.30 | 0.22 | Radius 0.07; wall 0.035 |
| Handoff zone | 0.28 | -0.08 | z = 0.10 |

The exact slots and dimensions are in [config/fake_scene.json](../config/fake_scene.json). The passing
clear-table fixtures use a 0.15 m release into the tray; that drop has no physical acceptance yet.
Some other perturbation seeds still produce pickup collisions and are correctly rejected.

For a later physical demo, tape the validated workspace, use the measured scene rather than blindly
copying these coordinates, establish a supported startup pose, place the operator at Esc, and station
another person at the follower's 12 V plug. Follow the documented STOP drill and supervised acceptance.
Do not promise a 30-second calibration or assume that loading a homography proves current accuracy.
