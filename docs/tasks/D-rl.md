# Lane D tasks: RL with a 7:00 AM go/no-go, then the story

**D-GO result (2026-10-17): 97/100 successes, 0 falls**, independent baseline evaluation (100 episodes, seed 0). **GO for the remaining RL track; real deployment remains disabled** until D7/M4 and the physical mapping/safety gates pass. The 507,904-step baseline is already the selected `rl/policies/handoff_best.zip`; training-summary evaluation uses a different seed and reports 94/100. Current evidence: `rl/runs/baseline/dgo.json`.

**Continuation verification:** [2026-10-17 evidence](../verification-2026-10-17.md). Checked boxes have supporting evidence; hardware and live-service acceptance remains open.

Lane D proves the slow path: Grok writes four reward functions, Daytona trains four policies at once, and the best one brings the held object to a moving hand on the real arm. A go/no-go at 7:00 AM Saturday decides whether RL reaches the real arm or is shown from simulation.

**Progress** (in time order; D sleeps 3:00 to 7:00 AM Saturday)

- [x] D1 · Environment · Sat 1:30 AM
- [x] D2 · Baseline PPO, started before sleep · Sat 3:00 AM
- [x] D-GO · Go/no-go decision · Sat 7:00 AM
- [ ] D3 · RL snapshot · Sat 8:30 AM
- [x] D4 · Reward-writing agent · Sat 9:30 AM
- [ ] D5 · Parallel training on Daytona · Sat 1:00 PM
- [x] D6 · Simulation video · Sat 2:00 PM
- [ ] D7 · Real deployment · Sat 6:00 PM (M4)
- [ ] D8 · Story work · from Sat 6:00 PM

## D1 · Environment (`rl/handoff_env.py`)

**Needs:** A5's tip site. **By:** Sat 1:30 AM. **Files:** `rl/handoff_env.py`, `tests/test_env.py`.

A Gymnasium environment on Lane A's MuJoCo model, without objects.

- Target: a point 5 cm above the hand, sampled in x 0.20 to 0.35 m, y -0.15 to 0.15 m, z 0.06 to 0.12 m. In 30% of episodes it drifts at up to 5 cm/s, so the policy learns to track.
- Observation, 16 numbers: 5 arm joint angles, the previous action (4), tip position (3), observed target position (3), and distance to the observed target (1). Compute the tip from joint readings with forward kinematics in both sim and real, never from simulator internals.
- Action: position changes for shoulder_pan, shoulder_lift, elbow_flex and wrist_flex, each in [-1, 1] and scaled to ±2° per step at 30 Hz. wrist_roll and the gripper stay fixed.
- Episode: 150 steps (5 s). Success: tip within 3 cm of the true target for 10 consecutive steps.
- Randomized every episode: joint offsets ±2°, action delay 0 to 2 steps, observation noise (1 mm and 0.5°), actuator gain ±20%, and a target-observation lag of 0.1 to 0.5 s, because real hand tracking lags.
- Joints clamp 3° inside their limits; the episode ends with a penalty if the tip drops below z = 0.03 m.
- `info["success"]` and `info["dist"]` come every step, so every reward variant is judged on the same metric.

**Done when:** `pytest -q tests/test_env.py` passes (a random policy runs 1,000 steps without errors, the observation has 16 numbers, success triggers when forced) and the viewer shows the target and tip.

## D2 · Baseline PPO on the laptop

**Needs:** D1. **By:** started before 3:00 AM Saturday, before sleep. **Files:** `rl/train.py`, `rl/evaluate.py`, `rl/rewards/baseline.py`.

Install `requirements-rl.txt`. The hand-written reward:

```latex
r = -d - 0.01 \|a - a_{prev}\|^2 + 1[d < 0.03]
```

Train PPO from Stable-Baselines3 (`MlpPolicy`, 8 parallel environments with `SubprocVecEnv`, 500,000 steps) with `python -m rl.train --reward rl/rewards/baseline.py --steps 500000 --out rl/runs/baseline`, then go to sleep. If it doesn't learn, check action scaling first, then reward scale, then episode length.

**Done when:** training is running before you sleep, writing `progress.csv` and a checkpoint every 50,000 steps.

## D-GO · Go/no-go at 7:00 AM

**Needs:** D2. **By:** Sat 7:00 AM.

Run `python -m rl.evaluate --policy rl/runs/baseline/policy.zip --episodes 100`.

- 80% or more: the full track continues, D3 to D7.
- Under 80%: D3 to D5 still run, because the parallel reward search is both the fix and the Daytona story. If no variant reaches 80% by 1:00 PM, RL is shown from simulation only, D7 is dropped, `carry_to_hand` stays on IK, and D joins Lane B on the dashboard.

Post the number and the decision at the 7:00 AM standup.

**Done when:** the decision and the success rate are written at the top of this file.

## D3 · RL snapshot

**Needs:** C5 (the snapshot pattern). **By:** Sat 8:30 AM. **Files:** `rl/train_parallel.py` (snapshot builder).

A second Daytona image with `mujoco numpy pyyaml gymnasium stable-baselines3` plus CPU-only PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`); the default Linux PyTorch wheel bundles CUDA and runs to several gigabytes. Request 4 vCPU and 8 GB per sandbox (`Resources(cpu=4, memory=8)`). GPU sandboxes don't speed up a small MLP policy, so don't spend GPU credits here.

**Done when:** a sandbox from the snapshot runs `python -c "import torch, stable_baselines3, mujoco"`, then is deleted.

## D4 · Reward-writing agent (`rl/rewards/write_rewards.py`)

**Needs:** C1. **By:** Sat 9:30 AM. **Files:** `rl/rewards/write_rewards.py`, `rl/rewards/variant_*.py`.

One `CODEGEN_MODEL` call returns four functions, each `def reward(obs, action, prev_action, info) -> float`. The prompt gives the environment description above, the baseline reward with its D-GO result, and a different shaping idea for each: exponential distance, progress toward the target, a smoothness penalty weighted by distance, and a success bonus with a time penalty. Each must pass `check_source(src, entry="reward")`. These files run only inside Daytona sandboxes.

**Done when:** four variants exist and each passes the check.

## D5 · Parallel training on Daytona (`rl/train_parallel.py`)

**Needs:** D3, D4. **By:** Sat 1:00 PM. **Files:** `rl/train_parallel.py`.

1. Create 4 sandboxes from the RL snapshot with `auto_stop_interval=0`, because a training run outlasts the idle auto-stop.
2. Upload `rl/`, the model files and one reward per sandbox.
3. Start `nohup python -m rl.train --reward reward_k.py --steps 600000 --out out/ > train.log 2>&1 &` in each, and poll `tail -n 1 out/progress.csv` every 30 s.
4. Each run saves `policy.zip`, a learning curve (success rate every 20,000 steps) and a 100-episode evaluation scored on `info["success"]`.
5. Download everything, delete the sandboxes in a `finally`, pick the best by success rate and then by mean steps to success, and copy it to `rl/policies/handoff_best.zip`.

**Done when:** four curves and the chosen policy appear on the dashboard, and the Daytona dashboard shows all 4 sandboxes deleted.

## D6 · Simulation video

**Needs:** D5. **By:** Sat 2:00 PM. Record the chosen policy following a moving target with MuJoCo's offscreen renderer, saved as MP4 for the dashboard and Devpost.

**Done when:** `web/static/rl_sim.mp4` plays on the dashboard.

## D7 · Real deployment (`rl/policy_runner.py`)

**Needs:** D5, A10, A13. **By:** Sat 6:00 PM (M4). **Files:** `rl/policy_runner.py`, `config/rl.yaml`.

A 30 Hz loop that `carry_to_hand` calls when `use_policy` is true:

1. Read joints from Lane A's driver (radians, through the joint map) and compute the tip with forward kinematics.
2. Set the target 5 cm above Lane B's hand position and build the same 16-number observation.
3. Run the policy, clip the action, and send it through the driver, which still enforces speed and workspace limits.
4. Stop on success (10 consecutive steps within 3 cm) or after 8 s, then release as `carry_to_hand` does.

Bring it up in stages: target fixed on a table marker, then a slowly moving hand, then normal speed (5 cm/s at most).

**Done when:** M4 passes; then set `use_policy: true`. If the real policy oscillates or misses, leave it false, keep IK tracking, and present RL from simulation, labeled as simulation.

## D8 · Story work

**Needs:** D7, or the D-GO fallback. **By:** from Sat 6:00 PM. Own the Devpost text, video and README per Pitch and Devpost. Pull every number from `runs/` (candidates per task, pass rates, time to learn a skill, real success counts, sandbox runs) instead of estimating.

**Done when:** M7 passes.
