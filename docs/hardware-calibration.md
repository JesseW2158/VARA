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

`config/joint_map.yaml`, `config/limits.yaml`, and the home/camera geometry remain placeholders.
Do not infer FK offsets from raw encoder limits or from the arms appearing to share a resting pose.
Physical camera calibration and the STOP drill remain required for Cartesian replay. The leader's
calibration mismatch is an additional prerequisite for teleoperation, separate from A6.
