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

To record each candidate's result and `fail_reason` from the run folders (A15 step 2) and get the
cereal-box verdict (step 3), run the read-only report over the same runs directory, during the physical
rehearsal as well:

```powershell
.\.venv\Scripts\python.exe scripts/rehearsal_report.py --runs $env:VARA_RUNS_DIR --out $env:VARA_RUNS_DIR
```

It prints, per task, the command, the end stage and its reason, the seconds from the first event to
`executing` (or to `queued`), the selected candidate and the saved skill, then one row per candidate:
number, PASS/FAIL, successes/trials, collisions, runner, whether it was selected, the hint and the
`fail_reason` (for example `trial 1: gripper hit cereal_box`); a normal-mode repair round gets rows of its
own. The verdict follows A15 on the six first-round candidates: 2 to 4 passing keeps the cereal box where
it is; 5 or 6 means move it 3 cm toward the line between the objects and the tray; 0 or 1 means move it
3 cm away; re-tape after a move and run the five commands again. A candidate cut off by the simulation
budget (`fail_reason` exactly `timeout`, runner `none`) says nothing about the box, so while such
candidates could still change the count the verdict is `rerun`: run the commands again rather than moving
the box. A routine (one candidate) gets `n/a`, and the last line counts the verdicts. `--task t-0003`,
`--since 2026-10-18T09:00` and `--last 5` narrow the report. `--out` writes a JSON copy and never
overwrites: a path ending in `.json` must be a new file, and any other path is a folder (created as needed)
that receives a new `rehearsal-<stamp>.json`. Without `--runs` (alias `--runs-dir`) it reads
`VARA_RUNS_DIR`, else the repo's `runs/`. Run folders do not record whether the arm was real or fake, so
the report reads the same for a fake bench and a physical rehearsal; the backup's provenance note (below)
is where that is declared.

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

If a command queues, show its reason. If the software errors, keep the failed run folder and play the
recorded backup clip, labelled as a prior run. The clip has one fixed location on the demo laptop:
`runs/demo-video/backup.mp4` (`runs/` is gitignored, so it is never committed). Record it with Xbox Game
Bar (Win+Alt+R) or OBS as one continuous isolated `FAKE=1` "clear the table" learn in the dashboard with
the mode banner in frame, and say what it is when you play it: fixture replies, local MuJoCo scoring, fake
arm. Keep a copy on a USB stick or a second laptop, and copy the same file to the Devpost submission as its
backup video (Devpost embeds a hosted video, so upload it to YouTube or Vimeo and paste that link). Load
backup presents a rehearsal backup only when `skills/_backup/provenance.jsonl` says the skill was learned
on the real arm; the checked-in backup is a fake learn (next section), so describe it as one and do not
imply that an unverified backup establishes physical acceptance.

## Backups: save a rehearsal skill, load it on stage

`skills/_backup/` holds the skills that Load backup (`POST /load_backup`) copies into the live library,
for a judge group that should see a fuller library or when live learning queues past the 75 s cap
(docs/risks.md). Nothing lands there by itself. After the best rehearsal, save the learned skill:

```powershell
.\.venv\Scripts\python.exe -m agents.library --save-backup clear_table --note "learned 2026-10-18 on the real arm, taped layout L1, sim 5/5, real 1/1" --keep-real
```

A bare name reads that live skill's file from the library (`VARA_SKILLS_DIR`, else `skills/`), so run it
before Reset demo; afterwards pass the archived file, `archive/<stamp>/clear_table.json` under the runs
directory, or any other skill file, such as an isolated library's `runs/demo-<stamp>/skills/clear_table.json`.
The command validates the file against `schemas/skill.json`, refuses `trusted: true` and the starter names
`hand_over` and `place_in_tray`, writes `skills/_backup/<name>.json` atomically (`--force` replaces one,
`--name` renames it) and appends one line to `skills/_backup/provenance.jsonl`: the time, the name, the
source path, your `--note`, the git commit (`git rev-parse HEAD`), `keep_real`, and the `sim` and `real`
counts saved. `--note` is required, so every backup says where it came from; when that line cannot be
written, the backup is undone. Without `--keep-real` the real counts are reset to 0 runs, because the
orchestrator records every replay, the fake arm's included, as a real run; pass `--keep-real` only for a
skill learned with `--real` on the physical arm. The command writes only under the library it runs on:
`skills/_backup/` by default, `<VARA_SKILLS_DIR>/_backup/` when `VARA_SKILLS_DIR` is set (so a test or
isolated run never writes into the repo), `<DIR>/_backup/` with `--root DIR`, and `--backup-dir DIR`
anywhere. To save from the README's isolated environment into the repo's backups, pass
`--backup-dir skills/_backup`. Loading still falls back: while the isolated library has no `_backup` of its
own, the dashboard's Load backup and `--load-backup` read the repo's `skills/_backup/`. A refused backup
exits 1 with the reason. Commit `skills/_backup/*.json` together with `provenance.jsonl`.

The checked-in `skills/_backup/clear_table.json` was learned on 2026-10-17 with fixture model replies,
local MuJoCo scoring and the fake arm, through `scripts/e2e_fake.py "clear the table"` (candidate 3
selected with 2 of 6 passing, sim 5/5, real 0 runs; its provenance line says so). It is a software
fallback, not a rehearsal backup: Load backup shows it with 0 real runs until a learn with the real arm
replaces it (`--save-backup ... --keep-real --force`).

Load backup copies every valid backup into the library as an untrusted skill (a backup claiming
`trusted: true` is loaded untrusted), replacing a learned skill of the same name. On the dashboard it
appears in the skill list as "clear table", tagged `newest` when it is the latest learned skill, with
`execution 0/0` and `sim 5/5` (real successes/runs, then simulated successes/trials);
`python -m agents.library --list` shows the same row as `trusted no`, `real runs/ok 0/0`,
`sim ok/trials 5/5`. Say what the label says: scored in simulation, no real runs until the arm executes
it. Never press Load backup before the script's first "clear the table": with `clear_table` loaded, that
command replays the saved skill instead of learning. Reset demo archives it again, to `archive/<stamp>/`
under the runs directory.

## Offline reset, about 30 seconds

1. Wait for the task to finish. If intentionally testing fake STOP, wait for cancellation and use Resume.
2. Press Reset demo in the isolated dashboard. Confirm exactly `hand_over` and `place_in_tray` remain.
3. Confirm the mode banner still says simulation/fixtures and the run paths point to the demo copy.
4. Check the synthetic camera view and reload the RL clip if it is paused or at its end.
5. Clear the typed command field and rehearse the opening mode statement.

Reset demo changes the skill library; it does not calibrate hardware or physically reset objects.
The fake calibration check returns a fixture result and is not a table measurement. On a real arm,
Resume can move toward home, so the offline reset steps are not a physical recovery procedure.

## Physical reset, 30 seconds

For the taped layout L1 with the orchestrator started with `--real`, between two commands or two judge
groups. It is separate from the offline reset above: step 1 can move the arm, and Torque off is never part
of a reset. The times are the rehearsed targets; a step that needs longer gets it.

1. **0 to 5 s, end stage and arm.** Wait for the task card to show `done`, `saved` or `queued`. After a
   STOP (the banner reads "STOPPED - arm frozen with torque on"), everyone takes their hands out of the
   taped box first, then press Resume: the arm goes home at 30 deg/s or less. Resume also sends an idle arm
   home, so use it the same way, hands out first, whenever the arm is not at home. The arm must then hold
   the home pose with torque on, not sag. `.\.venv\Scripts\python.exe -m arm.real_robot --real --home`
   (typed confirmation, 3 s countdown, Esc watcher) is only for when no server is running: a running
   orchestrator holds the follower's serial port.
2. **5 to 15 s, objects.** Empty the tray. Put the pill bottle, juice box and tea tin back on their taped
   spots with the marker side up and the cereal box on its tape; leave the hand rest empty, keep the
   wristband and hands off the table, and cover none of the calibration markers 0 to 7. The taped spots are
   the measured layout; the simulation table below is only the starting arrangement.
3. **15 to 22 s, markers and calibration.** On the dashboard's overhead feed, which draws every detected
   marker, confirm that markers 10, 11 and 12 (pill bottle, juice box, tea tin) and 30 (cereal box) are
   drawn and the wristband's 40 is not. Press Check calibration: it must answer with an `error_cm` (the
   calibration markers' mean error) of at most 1.0. It answers with an error instead when fewer than four
   calibration markers are visible, the mean is over 1.0 cm or the maximum over 1.5 cm
   ([table calibration](table-calibration.md)). Without a running server,
   `.\.venv\Scripts\python.exe -m vision.scene --print --seconds 2` prints the scene JSON (`"seen": true`
   for pill_bottle, juice_box, tea_tin and cereal_box, `false` for the hand) and
   `.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/<stamp>` saves an
   annotated frame as evidence; both open the overhead camera themselves.
4. **22 to 26 s, library.** Before a new judge group, press Reset demo and confirm exactly `hand_over` and
   `place_in_tray` remain. Between the first and the second "clear the table" of one script, do not: the
   second command must find the skill the first one saved. Press Load backup only as the fallback (Backups
   above), never before the first "clear the table".
5. **26 to 30 s, people and mode.** The plug holder confirms aloud that they hold the follower's 12 V plug,
   and the operator confirms a hand at Esc with the dashboard focused. Both read the mode banner:
   `Robot: physical` (not `simulation` or `physical, disconnected`) and `Cameras: physical`. Clear the
   typed command box.

Abort rule: do not start the next command if the arm is not at home holding its pose with torque on, any
of markers 10, 11, 12 and 30 is not drawn, Check calibration answers with an error or an `error_cm` above
1.0, the banner shows anything but `Robot: physical` and `Cameras: physical`, or anything (a hand, a
dropped object) is inside the taped box. Fix the cause, or play the backup clip instead.

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
| Pill bottle (`pill_bottle`), marker 10 | 0.18 | -0.12 | 1.75 in across, 3.25 in tall |
| Power bank (`juice_box`), marker 11 | 0.24 | -0.04 | Yaw 0.3 rad in the fixture; 3.5 in tall |
| Box (`tea_tin`), marker 12 | 0.15 | 0.03 | Long side along x; 4 in tall |
| Tape measure (`cereal_box`, the obstacle), marker 30 | 0.28 | 0.07 | On its edge, 2.25 in tall |
| Drop area (`tray`, taped, no physical tray) | 0.19 | 0.17 | Radius 0.09; spots (0.13, 0.15), (0.22, 0.12), (0.20, 0.22) |
| Handoff zone | 0.28 | -0.08 | z = 0.10 |

The exact slots and dimensions are in [config/fake_scene.json](../config/fake_scene.json). The passing
clear-table fixtures use a 0.15 m release into the tray; that drop has no physical acceptance yet.
Some other perturbation seeds still produce pickup collisions and are correctly rejected.

For a later physical demo, tape the validated workspace, use the measured scene rather than blindly
copying these coordinates, establish a supported startup pose, place the operator at Esc, and station
another person at the follower's 12 V plug. Follow the documented STOP drill and supervised acceptance.
Do not promise a 30-second calibration or assume that loading a homography proves current accuracy.
