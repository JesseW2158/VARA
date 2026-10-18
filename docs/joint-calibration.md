# Solve and check the A6 joint map

`python -m arm.check_mapping` has two offline modes. `--solve` turns one read-only encoder report, captured
with the arm held in a reference pose whose model joint values are known, into the `offset_deg` per joint
for `config/joint_map.yaml` and the calibrated `[lo, hi]` radians for `config/limits.yaml`. The measurements
mode compares five tape-measured tip positions with forward kinematics under the current map. Both read
saved reports and local files only: neither connects to an arm, edits a config file or enables motion.

A6 runs in this order, and again from the top after any LeRobot recalibration of the follower. LeRobot's
zero is the midpoint of each joint's calibrated range, so every sign, offset and limit depends on the
calibration file. A rerun starts from the placeholder joint lines in step 1, not from the previous solve.

1. [Sign test](#1-sign-test): with the placeholder joint lines in both config files, jog one joint at a
   time and compare its direction with the model.
2. [Reference pose](#2-reference-pose-and-its-report): hold a pose whose model joint values are known and
   capture its report.
3. [Solve](#3-solve-the-offsets-and-limits): compute the offsets and limits, read the warnings, paste both
   YAML blocks.
4. [Five-pose check](#4-five-pose-measurement-check): tape-measure five tips against FK under the pasted
   map.

Before anything powers the follower, follow `C:\Users\jesse\lerobot-tools\HANDOFF.md`: close the SenseCraft
app, run `arm_check.py` (read-only) and then `sync_goals.py COM8`, and keep the table clear. Every `--real`
command asks you to type that the arm is clear, counts down 3 s and freezes the arm with torque on when Esc
is pressed.

Units: the model and everything that commands the arm use radians. Degrees appear only in `offset_deg` in
`config/joint_map.yaml` (the driver maps `q_rad = radians(sign * lerobot_deg + offset_deg)`), inside
`arm/driver.py`, and in the solver's printed table and record, which set LeRobot's degrees beside the
model's to derive `offset_deg`. The limits the solver prints for `config/limits.yaml` are radians, and
nothing it computes reaches the arm.

## 1. Sign test

The sign test needs the placeholder joint lines in both config files. A first A6 run finds them there. On a
rerun, replace the five joint lines in each file with these before anything moves, and keep `lerobot_id`,
`lerobot_robot_type` and `gripper`:

```yaml
# config/joint_map.yaml: the placeholder joint lines
shoulder_pan: {sign: 1, offset_deg: 0.0}
shoulder_lift: {sign: 1, offset_deg: 0.0}
elbow_flex: {sign: 1, offset_deg: 0.0}
wrist_flex: {sign: 1, offset_deg: 0.0}
wrist_roll: {sign: 1, offset_deg: 0.0}

# config/limits.yaml: the placeholder joint lines, the MJCF ranges (arm.driver.DEFAULT_LIMITS)
shoulder_pan: [-1.9199, 1.9199]
shoulder_lift: [-1.7453, 1.7453]
elbow_flex: [-1.69, 1.69]
wrist_flex: [-1.6581, 1.6581]
wrist_roll: [-2.7438, 2.8412]
```

The previous solve describes the old calibration, yet the jog's range check and the driver's clamp read
`config/limits.yaml`, and the direction rule below assumes `sign: 1`. Restore both files together, because
limits solved under one map are wrong under another.

Open the model and move the joint under test by +0.349 rad (+20 degrees) in the viewer's joint panel:

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=sim/model/scene_ours.xml
```

Then, with the arm supported and clearance established, jog the same physical joint by +20 degrees from its
present position, hold it there for 10 s to compare, and let it return:

```powershell
.\.venv\Scripts\python.exe -m arm.driver --real --jog-joint shoulder_pan --deg 20 --hold 10
```

Same direction as the viewer: `sign: 1`. Opposite: `sign: -1`. The rule relies on the placeholder
`sign: 1`: `--jog-joint` adds +20 degrees to the mapped value through the current `config/joint_map.yaml`,
so under a `sign: -1` the servo turns the other way and the LeRobot reading the jog prints falls by 20
degrees instead of rising. Relative to whatever map is loaded, same direction means the joint's current
sign is right and opposite means it must flip; that is how a jog after pasting the solve confirms a sign.
Repeat for `shoulder_lift`, `elbow_flex`, `wrist_flex` and `wrist_roll`, re-establishing clearance before
each increment. Never command all joints to 0: LeRobot zero is mid-range on every joint, a large swing
from rest. Write the five signs down; the solve file needs them. The gripper's 0 to 100 range is A11's
measurement and stays out of this solve.

## 2. Reference pose and its report

Choose one pose you can set physically and check with a square or level, for example the upper arm
vertical, the forearm horizontal and the wrist straight. Find its model joint values: drag the viewer's joint
sliders until the model shows the same pose and read the five values in radians, or derive them from the
model's conventions (zero is mid-range on every joint). Support the arm in that pose and, without changing
it, capture the [read-only encoder report](hardware-calibration.md):

```powershell
.\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/reference-pose.json
```

The report must hold a follower entry whose saved encoder calibration matches the hardware and whose id
equals `lerobot_id` in `config/joint_map.yaml`. A leader-only report, a failed read or a report without
calibration is refused.

## 3. Solve the offsets and limits

Write `runs/calibration/solve.json`. `hardware_report` is relative to this file or absolute; `model_q_rad`
are the values you read for your pose, in radians; `signs` come from step 1; `note` is optional free text
describing the pose. Every value below is a placeholder (the home pose from `config/scene.yaml` and all
signs 1), not a reference pose:

```json
{
  "hardware_report": "reference-pose.json",
  "model_q_rad": [0.0, -1.35, 0.02, 1.45, 0.0],
  "signs": {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1},
  "note": "which pose, how it was squared or levelled, who measured"
}
```

```powershell
.\.venv\Scripts\python.exe -m arm.check_mapping --solve runs/calibration/solve.json
```

Per joint, `offset_deg = model_deg - sign * lerobot_deg`, where `lerobot_deg` is the report's calibrated
reading: `(raw - mid) * 360 / 4095` around the saved range midpoint, the formula LeRobot's degrees mode and
`arm.inspect_hardware` share. The saved `range_min`/`range_max` span `+-(range_max - range_min) / 2 * 360 /
4095` degrees around that midpoint; both ends go through the solved sign and offset and are ordered
`lo < hi` (a sign of -1 swaps them) to give `[lo, hi]` in radians.

The command first prints the reference pose's FK tip position: tape-measure the physical jaw-tip midpoint
against it before trusting the offsets, because a wrong `model_q_rad` solves to wrong offsets without
complaint. It then prints a per-joint table and the exact lines for both config files, for example (a
synthetic run with signs `[1, -1, 1, -1, 1]` and offsets `[5, -10, 0, 20, 0]` over the follower's current
encoder ranges):

```yaml
# config/joint_map.yaml: replace its five joint lines; keep lerobot_id, lerobot_robot_type, gripper
shoulder_pan: {sign: 1, offset_deg: 5.000}
shoulder_lift: {sign: -1, offset_deg: -10.000}
elbow_flex: {sign: 1, offset_deg: 0.000}
wrist_flex: {sign: -1, offset_deg: 20.000}
wrist_roll: {sign: 1, offset_deg: 0.000}

# config/limits.yaml: replace its five joint lines
shoulder_pan: [-1.9849, 2.1594]
shoulder_lift: [-2.0265, 1.6774]
elbow_flex: [-1.7070, 1.7070]
wrist_flex: [-1.4492, 2.1473]
wrist_roll: [-3.1416, 3.1416]
```

It writes `runs/calibration/solve-<UTC stamp>.json` (or `--out`; never overwritten; `VARA_RUNS_DIR` moves
the runs folder) with the inputs, both results, the YAML text, the FK tip, the current config values and the
hashes of the report, encoder calibration, solve file, config and model files. Nothing in `config/` is
edited: paste the blocks yourself, keep the gripper line and the ids, and keep the record next to the report.

Read the warnings before pasting. A range narrower than twice the driver's 3 degree margin plus 2 degrees
leaves almost no travel. A range reaching more than 5 degrees beyond the MJCF range (the follower's current
calibration does this on four joints, as above) describes positions the model and `arm.ik` cannot
represent: check the sign and offset, then either keep the physical range or narrow it deliberately. A range
not wider than twice the margin is refused outright because the driver could not load it.

Exit 0: solved and recorded, warnings included. Exit 2: invalid or incomplete input, a report without a
follower entry, missing or mismatched encoder calibration, a non-finite value, `model_q_rad` outside the
model's ranges (degrees by mistake), a sign other than 1 or -1, or an existing output path.
`--calibration-file` works as in the measurement mode: the encoder cache the report names (when it exists
locally) or the one given must still hash to the report's fingerprint.

## 4. Five-pose measurement check

With the solved map pasted, create an incomplete measurement template:

```powershell
.\.venv\Scripts\python.exe -m arm.check_mapping --example --out runs/calibration/measurements.json
```

The five entries contain `tip_xyz: [null, null, null]`. Replace the nulls with actual measurements
in meters; the template intentionally cannot pass. Keep `measurement_kind` as `physical` for
real measurements. Synthetic test fixtures must instead say `synthetic`; their numerical result
does not establish physical calibration.

For each of five distinct, supported poses spread across the workspace, follow the reference-pose
procedure in [A6](tasks/A-arm.md#a6--joint-map-and-accuracy-check). Record the jaw-tip midpoint in
the table frame: origin below the shoulder-pan axis, +x outward, +y left, +z up. Without changing
the pose, capture a corresponding [read-only encoder report](hardware-calibration.md):

```powershell
.\.venv\Scripts\python.exe -m arm.inspect_hardware --follower-port COM8 --out runs/calibration/pose-1.json
```

Repeat with new filenames for the other poses. This command may also inspect the configured
leader; its mismatch can produce exit 1 while the report still contains a valid follower snapshot.
The mapping checker uses only the follower entry and refuses a leader-only report. Do not fill
`tip_xyz` from predicted FK coordinates: those are the values the measurements must independently test.

Each `hardware_report` path is relative to the measurements file, or may be absolute. Run:

```powershell
.\.venv\Scripts\python.exe -m arm.check_mapping runs/calibration/measurements.json --out runs/calibration/check-01.json
```

The result reports each measured and predicted tip position, Euclidean error in meters, mean/max
error, and whether **every** error is at most 0.015 m. Five copies of one joint pose are rejected.
All reports must agree on follower identity, encoder calibration contents and fingerprint. The
driver alone converts each report's calibrated encoder radians through the current joint signs
and offsets; it also checks that those readings agree with the saved raw encoders.

The output records hashes of the measurement file, every report, encoder calibration, joint map,
limits, scene config, and the kinematic model XML files. If a reported encoder calibration file
exists locally, its current hash must still match. A missing path from another laptop is labelled
unchecked and permits historical comparison only. To require a particular current cache file:

```powershell
.\.venv\Scripts\python.exe -m arm.check_mapping runs/calibration/measurements.json --calibration-file C:\path\to\my_awesome_follower_arm.json
```

Exit 0 means the supplied measurements meet the numerical tolerance (or an example was generated).
Exit 1 means at least one pose exceeds it. Exit 2 means incomplete/invalid evidence, stale encoder
calibration, or an unwritable/existing output path. Existing files are never overwritten.

Use failed poses to review the measured signs, offsets and model geometry, then update the map
deliberately and rerun the comparison. A joint off by a constant needs its `offset_deg` fixed: repeat
steps 2 and 3 with a better-verified reference pose. A wrong sign does not show at the reference pose,
because the solve absorbs it into `offset_deg`; away from that pose the mapped joint turns the wrong way,
so its error is twice its angular distance from the reference value. To find one, rerun the step 1 jog
under the pasted map: a joint that moves opposite to the viewer has the wrong sign, so flip it in
`solve.json` and rerun step 3 with the same report. A passing comparison does not prove workspace coverage,
authenticate human measurements, establish gripper/limit calibration, or complete the physical
STOP drill. Recalibrating the encoders invalidates old mapping evidence and requires repeating A6.
