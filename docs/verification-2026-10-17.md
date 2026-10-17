# Continuation verification, 2026-10-17

This records checks on the Windows demo laptop after reading the original `HANDOFF.md`, the current
working tree, and the previous agents' reports. The original handoff predates commit `027f6d7`
(`Partial Completion`) and several uncommitted fixes. Existing changes were preserved.

## What was already implemented

All modules in the repository map exist: MuJoCo model and IK; driver, simulation, replay and live
primitives; camera calibration, tracking and verification; model client, routing, candidate generation,
runner, sandbox pool, planner, skill library and orchestrator; dashboard and voice page; PPO environment,
training, evaluation, reward generation, parallel training, policy runner and video renderer.

The later router, vision and web hardening was already present. Driver gripper helpers, rise-first
carry logic, isolated fake-run skill libraries and the baseline policy artifact also existed as
uncommitted changes. The four audit workflows named in the original handoff stopped at a session limit
without final audit reports; they were not completed reviews.

## Executed checks

All Python commands use `.venv/Scripts/python.exe` unless another environment is stated.

| Check | Result |
| --- | --- |
| Initial `-m pytest -q` | 1,141 passed, no skips, 122.70 seconds |
| First continuation `-m pytest -q` | **1,172 passed, no skips, 151.54 seconds** |
| First continuation Ruff and whitespace gates | Check passed; 79 files already formatted; `git diff --check` passed |
| Repository line endings and protected files | No tracked CRLF text files; no changes to `docs/interfaces.md`, `schemas/`, or `tests/contract/` |
| `scripts/e2e_fake.py "put the juice box in the tray"` | Reached `saved`; candidate 3 selected, four candidates passed 5/5 |
| `-m agents.router --eval --fake` | 20/20 phrases matched |
| `FAKE=1 -m agents.candidates --eval "put the juice box in the tray"` | 6/6 passed source checks on the first attempt |
| `-m agents.orchestrator --bench-demo` with isolated library | 5/5 queued within 75 seconds: 8.9, 9.1, 10.0, 10.2, 10.6 seconds |
| OpenCV import / ArUco | 4.13.0, ArUco available |
| LeRobot / MuJoCo import | 0.6.2 / 3.16.0 |
| Tracked binary artifacts | `config/homography.npy` and `rl/policies/handoff_best.zip` already tracked |
| `.env.example` audit | Application environment options documented; no secret file read |
| Independent baseline evaluation | 97/100 successes, zero falls, seed 0; 47.96 mean steps to success |
| Four existing reward variants | Each passes `check_source(..., entry="reward")`; fixture generation, not a live model check |
| RL video | 300 H.264/yuv420p frames, 640x480, 10 seconds; headless Edge `loadeddata` confirmed |
| C10 routine definition and replay, fake mode | `good_morning` passed 5/5, saved, then `Good morning` planned/replayed it and ended `done ok` |
| `FAKE=1 -m agents.sandbox_pool --bench` | Six candidates x five trials per task: juice box 4.1 s (4 passed), clear table 9.1 s (0 passed), place in tray 3.1 s (6 passed); local subprocesses only |
| Starter skills, juice box, five trials, seed 7 | `hand_over` and `place_in_tray`: 5/5 each, zero collisions, zero limit hits |
| Startup and video regression subset | 75 passed after adding early-STOP and driver-only STOP cases |

The five demo benchmark commands were **clear-the-table tasks that queued because no candidate passed**;
this establishes the time-budget behavior, not five successful executions. Evidence is in
`runs/verification/20261017-183918/bench-demo.log`. The successful juice-box learn is task `runs/t-0009/`,
with its copied library at `runs/e2e_fake/20261017-183854/skills/`.
The two-command routine test used a separate library and saved its evidence under
`runs/verification/routine-20261017-185154/`.
Starter outputs are under `runs/verification/starters-20261017-185736/`.
The final post-hardening juice-box check also reached `saved`: `runs/t-0010/`, copied library
`runs/e2e_fake/20261017-190047/skills/`.

The existing RL video originally failed in Edge with `DEMUXER_ERROR_NO_SUPPORTED_STREAMS`, despite
decoding in OpenCV. The renderer now uses PyAV/libx264, flushes all frames, and replaces the dashboard
artifact only after successful encoding. PyAV is in `requirements-rl.txt` (19.0.1 installed here).
`tests/test_render_video.py` passed all six codec, failed-render preservation and argument checks.
Browser evidence is in `runs/verification/video-browser/result-h264.html`; the visible simulation
frame shows both the green target and red tip. The regenerated video is `web/static/rl_sim.mp4`.

The baseline's training summary reports 94/100 on seed 10001; independent D-GO evaluation reports
97/100 on seed 0. These are separate evaluations. The selected artifact already matched the baseline,
and `rl/policies/handoff_best.json` now records its hash and provenance. It is not a reward-search winner.

Startup hardening binds Esc callbacks before confirmation/countdown and checks watcher, cancellation and
driver stop flags before/after connecting. The orchestrator's STOP handler receives the driver before
FastAPI lifespan initialization. Driver jog now also requires typed arm-clear confirmation. All such
tests use mocked devices; they do not establish physical Esc latency.

## Hardware check

Ran the reviewed, read-only `C:/Users/jesse/lerobot-tools/arm_check.py` in the conda `lerobot`
environment. It sends pings and register reads; no motor goals or torque settings were written.

| Role | Port | Board serial | Responding IDs | Supply | Temperature |
| --- | --- | --- | --- | --- | --- |
| Leader | COM7 | 5B90103224 | 1-6 | 4.9 V | 27-29 C |
| Follower | COM8 | 5B90101188 | 1-6 | 11.9-12.1 V | 31-34 C |

All raw encoder readings were within 0-4095. LeRobot calibration files already exist under the IDs
`my_awesome_leader_arm` and `my_awesome_follower_arm`. These encoder calibrations are distinct from
VARA's A6 joint-to-simulation mapping: `config/joint_map.yaml` still explicitly contains placeholder
signs and offsets. No Cartesian motion, teleoperation, goal sync or torque change was attempted.

## Follow-up continuation

Three workstreams ran in parallel: clear-table candidate repair, live-carry fallback safety, and
hardware calibration review. Root added the read-only inspection command and integrated the changes.

| Check | Result |
| --- | --- |
| Hardware inspection regression | 51 passed; fakes expose no motor-write/configure/torque APIs |
| Final follow-up full suite | **1,242 passed, no skips, 142.90 seconds** |
| Final follow-up formatting/contracts | Ruff check passed; 81 files formatted; no CRLF text or protected-contract changes |
| Carry/replay regression | 122 passed; no release on failed clearance, arrival, timer or policy assumptions |
| Fixture candidate regression | 58 passed, including scoring all six clear-table fixtures |
| Clear-table candidates 3 and 4, seed 7 | 5/5 each; zero collisions, limit hits or IK failures; mean simulated duration 30.913/40.233 s |
| `scripts/e2e_fake.py "clear the table"` | `saved`; candidate 3 selected; 13 trajectory segments validated and replayed |
| COM8 read-only calibration comparison | All six motors matched `my_awesome_follower_arm.json` |
| COM7 read-only calibration comparison | All six motors differed from `my_awesome_leader_arm.json`; no matching alternate cache file |

The new passing fixtures use shortest-first placement and a 15 cm release near the tray center,
avoiding the third slot that brought the forearm into the cereal box. The scene, obstacle, collision
checker and success tests are unchanged. Evidence is under
`runs/verification/clear-table-20261017/cand_3/` and `cand_4/`. Broader perturbations remain difficult:
candidate 3 achieved 9/10 successes with no collisions at seed 42, while seed 17 rejected both
strategies for tea-tin/juice-box contact during pickup. The elevated drop needs physical validation.

The complete clear-table fake learn used `runs/t-0011/` and preserved the normal library by saving to
`runs/e2e_fake/20261017-192552/skills/`. This supersedes the earlier no-passing-candidate demo outcome;
it does not turn the earlier queued benchmarks into successful executions.

Live carry now latches the configured handoff zone for out-of-box targets, refuses missing or
insufficient clearance, and checks measured rise completion. Tracking opens the gripper only after
encoder-confirmed zone arrival or continuous hand proximity. Missing hand detections reset the
proximity timer; timeout or stalled-clock failure freezes while preserving the grasp. The optional
policy sees the same fallback target, and its return value does not bypass the measured release check.

`-m arm.inspect_hardware` records raw register and calibration evidence without motor writes, including
partial-connection and read failure paths. Its real run correctly exited 1 for the leader calibration
mismatch. Both arms reported torque off and stored goals zero; nothing was synchronized or powered.
Full report: `runs/hardware/20261017-continuation.json`. See
[hardware calibration notes](hardware-calibration.md) for supply, temperature and A6 prerequisites.

## Software completion and calibration preparation

The next parallel pass addressed the remaining software findings without pretending that configuration
files establish physical calibration. The user confirmed both arms and cameras are connected and that
the table still needs calibration. A gitignored `.env` was created from the example, with COM8/COM7 and
the existing follower ID; API-key fields remain for the user to fill. Key values were never logged.

| Check | Result |
| --- | --- |
| Five clear-table demo rehearsals | All executed and saved; 8.6, 10.2, 8.9, 9.9, 11.8 seconds to execution; median 9.9 seconds |
| C7 repair integration | Actual first-round collision failures in 6/6 programs; fake-model repair passed 6/6 and saved the skill |
| Updated starter/carry scoring | `hand_over`, `place_in_tray`, and pill-bottle/juice-box two-handoff routine each passed 5/5, seed 7, zero collisions, limit hits and IK failures |
| Selected PPO through runtime safety wrapper | 20/20 reaches with FakeDriver, seed 7; gripper preserved and no stops |
| Real Edge dashboard | Typed clear-table request saved; six candidate cards, event updates, H.264 video ready state 4 at width 640, and Esc changed fake-arm state to stopped |
| Physical cameras, read-only smoke | DSHOW indexes 0 and 1 delivered 640x480 frames; index 2 did not open |
| Later configured-feed check | Camera 2 now opens; cameras 2 and 1 each delivered 148 frames over five seconds, about 29 fps, zero failed or late reads |
| Marker print artifact | 13 markers, two A4 pages at 300 dpi; PDF and PNG files generated locally |

Evidence directories:

- `runs/verification/c7-bench-20261017-194847/`: five-run log, summary and per-task artifacts.
- `runs/verification/c7-repair-20261017-195100/`: failures, repair prompts, second-round scores and saved skill.
- `runs/verification/scene-visibility-20261017/carry-guards/`: three fresh five-trial scoring runs.
- `runs/verification/policy-runtime-20261017-194639.json`: actual selected ZIP through the runtime wrapper.
- `runs/verification/browser-final/1792266692/`: browser result, screenshot, server log and isolated skills.
- `runs/setup/cameras/`: captured frames and device report; `runs/setup/markers/markers-a4.pdf`: print artifact.

Camera 0 was visually identified as the laptop webcam and camera 1 as the wrist view. The later
configured-feed check supersedes the earlier index-2 failure: camera 2 now opens, matching the user's
updated environment setting. Evidence is `runs/setup/cameras/configured-check-20261017.log` and
`runs/setup/cameras/current/`. The current camera-2 image looks across the table from a low position;
the wrist image is mostly dark/occluded. No markers were detected in either capture. Mount the overhead
camera and clear the wrist view. These short checks do not establish the B1 thirty-minute soak.

The subsequent 30-minute attempt failed its zero-drop criterion early: both feeds accumulated 27
late-frame gaps by 185 seconds (5,240 frames each, zero failed reads). It was stopped after 193.77 wall
seconds, rather than represented as a completed soak. The result records the forced-stop exit code
4294967295. Evidence: `runs/setup/camera-soak-20261017/camera-check.log` and `result.json`.

A separate 60-second diagnostic used independent reader threads. Camera 2 delivered 1,013 frames,
172 late gaps and a maximum read delay of 0.375 seconds; wrist camera 1 delivered 1,802 frames with
zero late gaps, about 30 fps. Neither had failed reads. This points to irregular camera-2 timing during
this run; it does not establish whether USB bandwidth, exposure, driver behavior, host load or ongoing
physical setup caused it. Evidence: `runs/setup/camera-soak-20261017/independent-readers.json` and its
log/script. Both cameras were released. B1 remains open; repeat after the physical setup is stable.

Implementation changes:

- Driver startup re-reads each motor immediately before syncing its goal, and synchronizes again
  immediately before every torque enable inside LeRobot configuration. Failed handshakes close the
  serial port; failure cleanup preserves torque intent. Configuration still temporarily disables torque,
  so a supported pose remains necessary. All new driver validation used mocks, not physical motion.
- Simulation preserves incoming visibility and last-known poses. Unseen objects cannot be picked but
  retain collision bodies; only a handoff completed in that simulator proves removal. Failed clearance,
  rise, travel or descent aborts before release; failed backoff/home does not emit a completed handoff.
- The disabled PPO runtime now uses the replay safety session for workspace, STOP and Cartesian speed
  checks (0.2 m/s, or 0.1 m/s near a hand). It preserves wrist/gripper commands, rejects invalid readings,
  and prevents expired or catch-up commands. A pacing regression at the policy-to-IK transition was
  reproduced and fixed. Native inference that never returns still requires the independent STOP path.
- Table calibration enforces mean error <=1.0 cm and maximum <=1.5 cm, requires at least four fitting
  markers, supports five independent measured check points, rejects fake cameras in live modes, and
  atomically preserves an existing matrix on failure. The dashboard reports a marker consistency check;
  it cannot certify the independent measurements. See [table calibration](table-calibration.md).
- The A6 offline checker compares five measured tip poses with saved encoder inspection reports and
  current mapping/model hashes. It refuses mismatched calibration and incomplete measurements; it
  does not invent offsets or authorize motion. See [joint calibration](joint-calibration.md).
- `scripts/check_setup.py` reports local prerequisites and key-presence booleans without opening
  hardware or network connections. `/runtime` labels the dashboard's device/model/scoring modes.
- The camera soak CLI now requires all requested feeds and a completed finite positive duration.
  Interrupted checks, empty/one-frame captures and failed partial opens cannot report acceptance;
  captures are released on errors. The 83-test camera/vision subset passed.
- Model defaults now explicitly request `grok-4.3` with no reasoning for routing and `grok-4.7` with
  low reasoning for generation, with configurable effort, visible-output limits and truncated-reply
  rejection. All 187 focused adapter/router/candidate tests passed. Official documentation supports
  these options; account access and service latency remain untested. See [model setup](setup.md#live-model-configuration).
- README, pitch, demo instructions and [Devpost draft](devpost-draft.md) now use measured local results
  and identify pending physical/cloud acceptance. No final presentation was recorded or published.

**Final combined gate: 1,406 passed, no skips, in 310.96 seconds.** The full log is
`runs/verification/final-completion/20261017-160904/pytest.log`. Ruff check passes, format check reports
90 files already formatted, and whitespace/LF/protected-file checks pass. The final fake juice-box
learn reached `saved`; its log is `runs/verification/final-completion/20261017-160702/e2e-juice.log`,
with isolated skills under `runs/e2e_fake/20261017-200702/skills/`. No live model call or physical
motion is implied by these checks. The local prerequisite report is `runs/setup/final-check.json`.

## Remaining acceptance work

- A6: measure joint signs/offsets against the model and pass the five-pose 1.5 cm accuracy test.
- A4/A9/A10/A11/A12/A13: supervised jog, replay, carry, grip and Esc latency acceptance on the follower.
  Carry now falls back to the handoff zone for out-of-box targets and retains its grasp on incomplete
  clearance or failed arrival. Mock coverage does not establish loaded-object clearance on the actual arm.
- B1/B3-B8 and I2: external camera mount, printed markers, measured homography, live accuracy, wrist
  images and five-marker touch check. The existing homography was produced with synthetic frames.
- C1-C3/C5-C6/C8, D3-D5 and A14: live model, Daytona and voice checks where the task requires them.
  Offline fixtures do not establish API compatibility or cloud cleanup on a real account.
- I1 still needs the other laptops; I3 and the M gates need the integrated physical scene.
- D7 remains off (`use_policy: false`); simulated policy results and runtime speed enforcement do not
  establish loaded-object, grasp-pose or moving-hand performance on the real arm.
- Clear-table fixtures now pass the demo seed but still fail on some broader perturbations; the 15 cm
  release drops props into the tray. Revisit the real layout/tray height under A15 before choreography.
- Reconcile the leader's hardware/saved calibration mismatch before teleoperation. Keep the arm
  supported during real connection: configure temporarily disables torque even with a fresh goal sync.
- D8/M7: review the prepared story, record the final demonstration, and publish the approved submission.

Task checkboxes are updated only where the stated acceptance can be supported by available evidence.
Hardware, live-service and human-observation checks remain open.
