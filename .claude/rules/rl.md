---
paths:
  - "rl/**"
---
# Lane D rules
- The observation is the 16 numbers in D1. Compute the tip from joint readings with forward kinematics, never from simulator internals.
- Actions are ±2 degrees per step at 30 Hz on 4 joints; wrist_roll and the gripper stay fixed.
- Daytona RL images use CPU-only PyTorch; don't spend GPU credits on this small MLP.
- Grok-written reward files run only inside Daytona training sandboxes.
- policy_runner.py sends actions through arm/driver.py, which enforces the limits.
