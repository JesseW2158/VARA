# VARA - submission draft

**Status:** text and storyboard prepared for team review. This document has not been submitted or
published. The video still needs recording/export. Replace no mode labels with stronger claims unless
a newer run supplies the evidence.

## Submission fields

Fill these in on the Devpost form; they are the only fields the draft below does not already cover.

- **Video URL:** (pending; the storyboard is in [pitch.md](pitch.md#two-minute-video-storyboard), and the
  local backup of the recording belongs at `runs/demo-video/backup.mp4`, as [demo.md](demo.md) says)
- **Repository URL:** (pending; make the repository public before M7, then paste the `https://github.com/...`
  link here and in the README)
- **Team:** (names as they appear on Devpost, one per lane A to D)
- **Sponsor challenges:**
  - [ ] Best Use of Daytona
  - [ ] Make It Legendary

Both challenges are named in [sources.md](sources.md) and [kickoff.md](kickoff.md); the
[sponsor challenges](#sponsor-challenges) section below lists the evidence each one needs and what exists.
Every number quoted on the page comes from `python scripts/story_numbers.py --markdown`, which reads
`runs/`, `skills/` and the RL provenance files and names the file behind each figure.

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

## Sponsor challenges

Select both on the form only once the "Still missing" column is empty; the M7 row of the
[gate ledger](gates.md) records who checked it and when.

| Challenge | What the judges need to see | Where it lives today | Still missing |
| --- | --- | --- | --- |
| Best Use of Daytona | Generated programs scored in Daytona sandboxes, the RL reward variants trained in parallel sandboxes, and the account left clean afterwards | `agents/sandbox_pool.py` (cap 7, `auto_stop_interval=0`, every sandbox deleted on shutdown), `rl/train_parallel.py` (cap 4, `--build-snapshot`) and `scripts/daytona_smoke.py`; offline evidence only so far: `tests/test_pool_fake.py` and `tests/test_rl_glue.py` run against a fake Daytona | A keyed run. No `candidate_scored` event under `runs/` so far says `runner: daytona` (on 2026-10-17: 150 `local`, 19 `none` for candidates that never reached a scorer), and `rl/runs/parallel/result.json` says `runner: local` for all four variants (`python scripts/story_numbers.py --markdown` prints the runner counts). After the first keyed run, cite its `runs/<task_id>/scores.json`, the `python -m agents.sandbox_pool --bench` output and a screenshot of the Daytona dashboard showing zero sandboxes afterwards |
| Make It Legendary | The project built with Cursor (per [kickoff.md](kickoff.md): Claude Code in Cursor's terminal, Cursor's agent for some tasks, screenshots in the README) and the xAI Grok path: routing, success specs, program generation, reward generation and the browser voice session | `.cursor/rules/project.mdc`; `agents/llm.py`, `agents/router.py`, `agents/candidates.py`, `rl/rewards/write_rewards.py`, `web/voice.html` with `web/static/voice.js`; offline evidence only so far: fixture replies under `FAKE_LLM=1` | No Cursor screenshots exist in the repository (there is no `docs/img/` folder and no image is tracked). Capture them while working in Cursor, add them under `docs/img/` and link them from the README. Also missing: a keyed xAI run (`python -m agents.llm --ping`, then a learn whose `runs/<task_id>/llm.jsonl` came from the live model) and the voice test in the judging room (M3) |

## Before publication

- [ ] Fill in the [submission fields](#submission-fields): team names and the repository URL; verify
      repository visibility.
- [ ] Record/export the [two-minute storyboard](pitch.md#two-minute-video-storyboard), then add its URL to
      the submission fields.
- [ ] Keep local fixture/simulation labels visible and confirm captions match the actual footage.
- [ ] Update live-service and hardware statements only from newly captured acceptance evidence.
- [ ] Review event eligibility, confirm both sponsor challenges above against the current requirements and
      the evidence table, and check all externally hosted media permissions.
- [ ] Submit and confirm the published page from the team's account.
