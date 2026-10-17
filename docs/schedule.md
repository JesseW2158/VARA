# Gates, cuts and team

The schedule restarts from midnight: the scaffold lands at 1:30 AM and a fake end-to-end run at 2:00 AM, and every later gate is tested on the demo laptop. Two decision points are new: the RL go/no-go at 7:00 AM, and the 75-second judging cap enforced from M5 on.

Each lane's bars run from Friday 10 PM to Sunday noon. The diamonds are the gates listed below, and at least two people are awake at every hour.

**Gates.** Tick one only when its test passes on the demo laptop, and hold a two-minute standup at each.

- [ ] **S0, Sat 1:30 AM.** Scaffold on `main`: `pytest -q` and the fake end-to-end run pass.
- [ ] **I1, Sat 2:00 AM.** `/verify` passes on all four laptops.
- [ ] **M1a, Sat 3:00 AM.** Each lane's first deliverable passes (Team table below).
- [ ] **M1, Sat 6:00 AM.** I3: a typed new task is learned with local scoring, replayed on the real arm, verified by the camera and saved.
- [ ] **RL go/no-go, Sat 7:00 AM.** D-GO decided and announced.
- [ ] **M2, Sat 3:00 PM.** A new typed task runs 6 Daytona candidates × 5 trials, and the winner reaches the real arm within 75 s and is saved. Repeating the command plans from the library, with the arm moving within 5 s.
- [ ] **M3, Sat 4:00 PM.** The same flow works by voice, the typed box still works, Esc freezes the arm within 0.2 s, and saying "stop" freezes it within 2 s.
- [ ] **M4, Sat 6:00 PM.** Four reward variants trained in parallel sandboxes, with results on the dashboard. The best policy brings the object to the wristband within 3 cm in 4 of 5 real tries, never over the speed limit; otherwise `carry_to_hand` stays on IK.
- [ ] **M5, Sat 9:00 PM.** The demo script runs end to end in under 3 minutes with `DEMO_MODE=1`, three times in a row, with Reset demo before each.
- [ ] **M6, Sun 1:00 AM.** Feature freeze: two clean runs in a row, then record the backup video.
- [ ] **M7, Sun 4:30 AM.** Devpost submitted with video, public repo, README and both sponsor challenges selected (edits allowed until 7:45 AM).
- [ ] **M8, Sun 8:30 AM.** Judging table set up, calibration rechecked, voice tested in the judging room, `DEMO_MODE=1` on, Reset demo pressed.

**When a gate slips by 2 hours or more,** cut in this order and say so at the standup:

1. Routines ("good morning").
2. RL on the real arm. Keep the simulation results, curves and video; `carry_to_hand` uses IK.
3. The repair round, which is already off in judging mode.
4. Dashboard polish. Keep the success-test banner, the six cards and the STOP button.

Never cut the parallel sandboxes, the voice path, STOP or the trajectory validator. The first two are what the sponsor challenges judge; the last two are safety.

**Team.** Each person owns one lane end to end; write your name next to a letter.

| Lane | Owns | First deliverable (Sat 3:00 AM) | Saturday daytime | Event slot |
| --- | --- | --- | --- | --- |
| A: Arm | LeRobot, calibration, MuJoCo model, IK, SimRobot, replay engine, live primitives, safety, voice page | A6 passes (sim and real agree within 1.5 cm) and `tests/test_ik.py` passes | Voice page, safety layer, demo scene tuning | SpaceXAI workshop, 1:30 PM |
| B: Vision | Cameras, markers, calibration, scene state, real verification, dashboard | Scene state within 1.5 cm of tape-measured positions | Dashboard | Sponsor fair, 10:00 AM (ask how judging works) |
| C: Agents | Scaffold, router and specs, candidates, scoring runner, Daytona pool, planner, orchestrator | Router writes valid specs for the demo phrases, and one Daytona sandbox scores a candidate | Sandbox pool, budgets, hardening, routines | Daytona workshop, 2:30 PM |
| D: RL and story | RL environment, reward agent, parallel training, policy runner; Devpost, video, README | Baseline PPO training running | Reward search and training on Daytona, real deployment, then Devpost | None (protect the training window) |

**How the team works:**

- Everyone codes with Claude Code from the same `CLAUDE.md` and contracts, inside Cursor for the challenge, one task per `/task` run.
- One GitHub repo, and `main` must always pass `/verify`. One branch per lane; merge at each gate after `/verify` and, for arm code, a 5-minute smoke test on the real arm.
- Contract changes are agreed at a standup, then made by one person in prose, schema and test together and pushed to `main` at once.
- Two-minute standup at every gate. Anyone blocked for 30 minutes says so out loud.
- Progress lives in each lane file's Progress list, which `/task` ticks. Grok's planning bot is a bonus for Make It Legendary if you use it.
- Sleep in shifts: D 3:00 to 7:00 AM Saturday; B 3:30 to 7:30 AM Saturday; A and C 7:30 to 11:30 AM Saturday; A and C 2:00 to 5:30 AM Sunday; B and D 5:30 to 8:00 AM Sunday.
