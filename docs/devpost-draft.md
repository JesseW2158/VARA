# VARA - submission draft

**Status:** text and storyboard prepared for team review. This document has not been submitted or
published. The video still needs recording/export. Replace no mode labels with stronger claims unless
a newer run supplies the evidence.

**Tagline:** A robot arm that builds a personal skill library from testable requests.

## Inspiration

Assistance at home depends on a person's objects, layout, and preferences. We wanted to explore an
interface where a person describes the help they want and can see how the robot will judge success.
VARA's starting point is a small tabletop arm and a deliberately limited vocabulary of tasks that can
be checked: move an object, place it in a zone, leave an obstacle alone, or complete a handoff.

## What it does

VARA accepts a typed request or a tool call from its voice interface. A saved skill is re-planned for
the current scene. A new task gets a success specification before candidate programs are generated.
Six programs each run five randomized MuJoCo trials; collisions and motion-limit failures disqualify
programs even if they satisfy the placement goal. The selected trajectory is validated before replay,
and the same success test checks the final scene before the new skill is saved. Unsupported requests
are queued. Routines compose saved skills.

Our demonstrated local workflow uses fixture model replies, local simulation, and fake hardware and
camera verification. It completes the route-generate-score-select-replay-verify-save sequence. The
physical SO-ARM101 Pro leader/follower kit is connected and inspected, but physical task execution
still requires measured joint mapping, table-camera calibration, grip checks, and supervised motion
acceptance. The xAI and Daytona integrations are implemented; live service acceptance is tracked
separately from offline results.

## How we built it

Python and FastAPI coordinate the workflow and stream events to a browser dashboard. xAI integrations
cover routing, success specifications, program generation, reward generation, and the browser voice
session. API keys stay server-side; the voice browser receives a short-lived token.

MuJoCo provides the arm, table, props, obstacles, contact checks, and trajectory recording. Generated
programs run in Daytona sandboxes or restricted local subprocesses with a minimal environment, never
inside the orchestrator and never with the physical driver. LeRobot connects the hardware through a
single driver boundary. Our own replay code validates trajectories and retains torque on STOP.
OpenCV/ArUco supplies the scene-tracking and camera-calibration pipeline.

A second track uses Gymnasium and Stable-Baselines3 PPO for reaching a hand target, including target
motion and randomized delays/noise. We implemented parallel reward experiments, but the selected
policy is the hand-written-reward baseline. It remains a simulation artifact; physical RL deployment
is disabled.

Claude Code and Codex assisted implementation, tests, and review, with shared interfaces and lane
ownership documented in the repository. This development assistance is distinct from runtime model
calls. The [README](../README.md) includes the architecture and runnable setup.

## Challenges

A candidate can put an object in the right place while hitting the cereal box on the way. Separating
the success predicate from collision and limit checks made those false successes visible. Our first
clear-table plans reached the tray but collided on later placements; revised fixture strategies now
pass the chosen demo seed while difficult perturbations remain rejected.

Hardware brought a different kind of uncertainty. Existing encoder calibration files do not prove
that physical joint angles match MuJoCo. Read-only inspection also exposed a leader calibration
mismatch. We kept these findings separate from simulation success and built inspection/setup reports
instead of treating connected motors as motion-ready hardware.

We also found that a video readable by OpenCV was not playable in the browser. The renderer now writes
H.264/yuv420p and preserves the old clip if rendering fails. Worker deadlines prevent a bad reward
function from leaving a training job stuck indefinitely.

## Accomplishments with evidence

- Five fixture-backed clear-table learns all reached fake execution and saved the skill. Times to
  execution were 8.6, 10.2, 8.9, 9.9, and 11.8 seconds: median 9.9 seconds, all below the 75-second demo cap.
- Two of six clear-table strategies passed five trials each at seed 7 with zero collisions, limit hits,
  or IK failures. Scoring retained the obstacle and collision rules.
- A repair integration run deliberately made all six initial programs collide. The genuine repair
  pipeline produced six passing fixture-backed replacements, and the selected replacement was saved.
- Both starter skills passed five trials with the juice box. A two-object handoff routine also passed
  five trials, preserving both objects' removed/hidden state after their handoffs.
- The selected PPO baseline achieved 97 successes in 100 independently evaluated simulated episodes,
  with zero falls at seed 0. Its 507,904 training steps and policy hash are recorded in
  [the provenance file](../rl/policies/handoff_best.json).

These measurements do not establish live model latency, cloud throughput, real-arm success rates,
voice reliability, or physical STOP latency. The [verification record](verification-2026-10-17.md)
and [handoff](../HANDOFF.md) distinguish completed checks from outstanding acceptance work.

## What we learned

A useful robot demonstration needs more than a plausible final pose. It needs a testable goal,
separate motion checks, reproducible run artifacts, and an honest account of what was simulated.
Keeping the skill, trajectory, and success specification distinct made repair and saved-skill reuse
possible without handing generated code a physical robot.

## What's next

Complete physical joint/grip and overhead-camera calibration, reconcile the leader calibration,
measure the supervised STOP and replay checks, and evaluate the handoff with real objects. Validate
live xAI voice/text calls and Daytona lifecycle cleanup on the configured accounts. Broaden the tested
layouts and replace the current elevated tray drop with a physically accepted placement. Only then
consider enabling the RL handoff controller on the follower.

## Built with

Python 3.12, SO-ARM101, LeRobot, MuJoCo, FastAPI, OpenCV/ArUco, NumPy, JSON Schema, Gymnasium,
Stable-Baselines3, PyTorch, and PyAV; implemented integrations with xAI Grok and Daytona.

## Before publication

- [ ] Add the actual team/member names and repository URL; verify repository visibility.
- [ ] Record/export the [two-minute storyboard](pitch.md#two-minute-video-storyboard), then add its URL.
- [ ] Keep local fixture/simulation labels visible and confirm captions match the actual footage.
- [ ] Update live-service and hardware statements only from newly captured acceptance evidence.
- [ ] Review event eligibility, sponsor categories, and all externally hosted media permissions.
- [ ] Submit and confirm the published page from the team's account.
