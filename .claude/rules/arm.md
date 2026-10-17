---
paths:
  - "arm/**"
  - "config/joint_map.yaml"
  - "config/limits.yaml"
---
# Lane A rules
- Joint vectors are [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in radians; the gripper is a separate 0.0 to 1.0 value.
- Only arm/driver.py imports lerobot or touches the serial port, always under its lock.
- The driver clamps every command to the calibrated range minus 3 degrees and to 2 degrees per 30 Hz tick, whatever the caller asks.
- Every real-arm motion goes through replay.replay() of a validated trajectory or a primitive in real_robot.py.
- Real-arm scripts take --real, count down 3 s, and freeze on Esc.
- After changing sim_robot.py, re-score both starter skills (docs/tasks/A-arm.md, A12).
