# SO-ARM101 hardware evidence and remaining calibration

The assembled Pro kit uses follower COM8 (12 V, USB serial `5B90101188`) and leader COM7
(5 V, USB serial `5B90103224`). Confirm these identities after wiring changes. A matching LeRobot
encoder calibration does not establish the mapping between the physical arm and MuJoCo.

## Capture without moving either arm

Close applications that hold the serial ports, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --leader-port COM7 --out runs/hardware/check-01.json
```

Use a new output filename each time; existing evidence is never overwritten. Either port can be
omitted when only that arm is being inspected (unset its corresponding environment variable too).
The command reads motor registers and closes the ports. It does not configure motors, synchronize
goals, change torque, or invoke robot-level connect/disconnect. It needs no motion countdown.

The report contains USB identity, saved calibration SHA-256, saved and hardware calibration,
raw positions/goals, torque, voltage, and temperature. Normalized encoder positions appear only
when the two calibrations match and are valid. Those radians are centered on the LeRobot range;
they are **not** physical-to-MuJoCo joint coordinates. Missing motors, incomplete reads, calibration
mismatches and reported health issues produce a nonzero exit status. An exit of zero only means
the inspection found no listed issue; it does not authorize motion or certify geometry.

## Driver checks that move the follower

The three commands below live in `python -m arm.driver` and move the real arm, so each takes
`--real`: a typed `yes` that the arm and table are clear, a 3 s countdown, and an Esc watcher that
freezes the arm in place with torque on (an Esc during the connect leaves it frozen and nothing
moves). Run `arm_check.py` and `sync_goals.py COM8` first, as for every connect. `--fake` runs the
same flow on the in-memory arm without hardware, the typed confirmation or the countdown; the A11
session still asks its questions (jaw opening, placement, `held? y/n`), and a blank answer or EOF counts
as unanswered.
Every report goes to a new timestamped file under `runs/calibration/` (`VARA_RUNS_DIR` moves it);
nothing is ever overwritten and nothing under `config/` is written by these commands.

**Start inside the band.** `--jog-joint` and `--measure-grip` re-send the pose they read at the start
with every command, and the driver pulls any joint outside its commandable band (`config/limits.yaml`
minus 3 degrees) back into it at up to 2 degrees per command. So both refuse to start, with exit 2 and
nothing moved, when any joint lies more than 0.5 degrees outside the band, and they name each such
joint and by how much. The measured rest pose is outside the placeholder band (shoulder_lift by about
6.7, elbow_flex 2.9 and wrist_flex 9.9 degrees), and a joint resting on its mechanical stop sits about
3 degrees outside any band. Move the arm to a supported pose inside the band first (teleoperate it, or
move it by hand with torque off and run `sync_goals.py COM8` again), then start. Within the band, only
the named joint or only the jaws move.

### A6: single-joint increment jog (a comparison aid, not a calibration writer)

From the measured, supported reference pose, with the viewer open on the same pose
(`python -m mujoco.viewer --mjcf=sim/model/scene_ours.xml`):

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --jog-joint shoulder_lift --deg 20 --hold 10
```

Only the named joint (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex` or `wrist_roll`)
moves (see "Start inside the band" above): out by `--deg` (A6 uses 20; the default is 10) at 1 degree
per 30 Hz tick, a hold of `--hold` seconds (default 10) for the comparison, then back. Before the
move, at the increment, at the end of the hold (the settled reading; a servo can trail its target
right after the ramp) and after the return it prints the LeRobot reading in degrees as the servo
reports it (the number LeRobot's own tools show), our mapped joint value in radians through the
current `config/joint_map.yaml`, the expected model increment (`+0.349 rad` for 20 degrees) and the
tip position from `arm.ik.fk_tip` for the before, expected and measured joint vectors. Set the same
joint in the viewer by the same increment and compare. The jog moves the mapped value through the
current `config/joint_map.yaml`, so the comparison is relative to the sign already there: the real
joint moving the same way means that sign is right, the opposite way means flip it (with the
placeholder `sign: 1`, opposite means `sign: -1`). On an A6 rerun, restore the placeholder joint
lines in both config files first ([joint calibration](joint-calibration.md), step 1).
`python -m arm.check_mapping --solve` then derives `offset_deg` and the calibrated limits from one
reference-pose report; rerun the jog under the pasted map to confirm each sign, then do the five-pose
check. The increment is refused, before anything moves, when it would leave the calibrated range
minus 3 degrees. `--fake --jog-joint shoulder_lift --deg 20 --hold 0` is the hardware-free dry run.

### A11: gripper measurement

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --measure-grip --objects pill_bottle juice_box tea_tin --grasps 5
```

The five arm joints are read once and every gripper command re-sends that same joint vector, so,
started inside the band (see above), only the jaws move. The session: open to 1.0 and ask for the jaw
opening in mm (measure it with a ruler); close on nothing, read after 0.5 s and re-open at once (an
empty close is never left stalled); then per object, `--grasps` times (default 5; default objects:
every `kind: object` in `config/objects.yaml`; a name listed twice is measured once): open, wait for
Enter with the object between the jaws, close with the 3-unit hold back-off, record the settled
reading, ask `held? y/n`, re-open. Esc at any point freezes the arm, ends the session and still writes
the partial report, but prints no lines to paste; a reading taken after the Esc (jaws frozen
part-way) is discarded and the problem list says where the session stopped. The report
(`runs/calibration/gripper-<UTC>.json`, or a new path given with `--out`) holds the jaw opening,
the open and empty-closed readings, each object's closed readings, held verdicts and
`held_min = (smallest closed reading - empty-closed reading) / 2` on the 0 to 100 scale, the
follower id and calibration hash, and any problem: an object wider than the jaw opening minus 2 cm
or outside the 6 to 11 cm height rule, a blank jaw answer, an object the jaws did not detect.

The command then prints the exact lines to paste:

- `gripper: {closed_pct: 0, open_pct: P, empty_closed_pct: Z, jaw_open_mm: J}` replaces the
  `gripper:` line of `config/joint_map.yaml`. `closed_pct` and `open_pct` are the command endpoints
  of the current map, copied from it: this arm uses 0 and 51, so our gripper 1.0 opens the jaws about
  9.7 cm, the opening the simulation plans with (100 would open them about 15 cm). `empty_closed_pct`
  is the measured close on nothing; `jaw_open_mm` is a note the loader ignores. A gripper left open
  past `open_pct` (by teleop or an older map) reads as 1.0 and closes to `open_pct` on the next command.
- `held_min: V` goes on each object's entry in `config/objects.yaml`.

Then run `pytest -q tests/test_config.py`. `--fake --measure-grip --grasps 2 --fake-contact 25` is
the hardware-free dry run (`--fake-contact` is the 0 to 100 reading the fake jaws stop at on an
object).

### A13: Esc-to-freeze latency record

Every Esc after the typed confirmation of a `--real` run of these commands appends one JSON line
`{"t", "source", "phase", "latency_ms", "extra"}` to `runs/calibration/esc-latency.jsonl` (the key
press to the freeze, both from the same monotonic clock). `phase` is `session` when the Esc came after
the connect returned: the command prints `Esc -> freeze command: N ms`, and A13 requires at most
200 ms. It is `connect` when the Esc came while the driver was connecting: nothing moved, the freeze
lands when the connect returns, so the number (often several hundred ms) is a diagnostic and stays out
of the 200 ms bar. An Esc during the countdown aborts before connecting and leaves no record. Keep the
file as the evidence of the drill.

The same record comes from every `--real` entry point: `python -m arm.replay` (source `arm.replay`),
`python -m arm.real_robot` (`arm.real_robot`), `scripts/touch_markers.py` (`touch_markers`),
`python -m rl.policy_runner` (`rl.policy_runner`), the orchestrator's console Esc
(`orchestrator-console`), and a dashboard or voice STOP while the orchestrator drives the real arm
(`dashboard-<source>`, latency from the browser's key press to the freeze on this laptop's clock). Read
the verdict with:

```powershell
.\.venv\Scripts\python.exe -m arm.driver --esc-latency-report
```

It lists every record, keeps the connect-phase ones apart, and exits 0 only when every session record
is at most 200 ms (1 when one is over or never froze, 2 when there is no session record to judge).

## Measured on 2026-10-17

Evidence: `runs/hardware/20261017-continuation.json` (local, gitignored).

| Arm | Encoder calibration | Supply | Temperature | Torque |
| --- | --- | --- | --- | --- |
| Follower COM8 | Matches saved file | 11.9-12.0 V | 31-34 C | Off on all six motors |
| Leader COM7 | Differs from saved file on all six motors | 4.9 V | 26-28 C | Off on all six motors |

The leader's hardware offsets/ranges differ from
`my_awesome_leader_arm.json`. No other matching calibration file exists in the local LeRobot
calibration cache. Preserve this report and reconcile the leader calibration before teleoperation;
do not silently replace either the saved file or the motor values. Its current report intentionally
omits normalized positions. The follower's saved calibration hash is
`5494adfa0704a2b6b9f556077f48a01f9482f378f7c575ff6a61a8d7f1bbbe6c`.

Both boards reported stored goals of zero while torque was off. Enabling torque against those goals
can move the arms. The inspection leaves them untouched. Use the existing goal-sync procedure and
supported-pose precautions before any deliberate torque enable.

## What A6 still requires

Follow [A6](tasks/A-arm.md#a6--joint-map-and-accuracy-check): compare each physical joint direction
against the model from a measured reference pose, record the signs/offsets and physical ranges, then
measure five tip poses across the workspace. All five must agree with FK within 1.5 cm. Record the
encoder calibration hash alongside those measurements so later LeRobot recalibration invalidates
the old mapping. A11 separately measures the gripper and object contact readings.

The offsets and calibrated limits are now derived by `python -m arm.check_mapping --solve` from one
reference-pose report plus the sign-test results (see [joint calibration](joint-calibration.md)); the
five tape-measured poses remain the acceptance test and the solver never edits `config/`.

`config/joint_map.yaml`, `config/limits.yaml`, and the home/camera geometry remain placeholders.
Do not infer FK offsets from raw encoder limits or from the arms appearing to share a resting pose.
Physical camera calibration and the STOP drill remain required for Cartesian replay. The leader's
calibration mismatch is an additional prerequisite for teleoperation, separate from A6.
