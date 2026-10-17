# Measure and check the A6 joint mapping

`python -m arm.check_mapping` compares five or more measured physical tip positions with forward
kinematics under the current `config/joint_map.yaml`. It reads saved hardware reports and local
files only. It never connects to either arm, changes the map, or enables motion.

Create an incomplete measurement template:

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
deliberately and rerun the comparison. A passing comparison does not prove workspace coverage,
authenticate human measurements, establish gripper/limit calibration, or complete the physical
STOP drill. Recalibrating the encoders invalidates old mapping evidence and requires repeating A6.
