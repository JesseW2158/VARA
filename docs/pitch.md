# Pitch and recording guide

VARA's central idea is a personal skill library built from requests that can be tested. Show the
success test, a rejected candidate, and the saved skill. Explain the active mode before the demo:
currently verified footage uses local MuJoCo, fixture model replies, and fake hardware/cameras.

## A 60-second talk track for the verified local demo

"The help someone needs at home is personal. VARA explores a robot arm that can build a library of
those requests.

Here, we're showing the working software pipeline in local simulation. Model responses and devices
are fixtures for this demonstration. The request is 'clear the table.' Before any program runs, the
system states what success means: every object must end inside the tray.

Six programs try the task. Each gets five trials. The ones that hit the cereal box fail, even when
they put the objects in the right place. Only a passing trajectory reaches the validator and fake
arm. The same success test checks the final scene before the skill is saved.

Five local rehearsals all saved the skill, with a median 9.9 seconds to execution. A separate PPO
baseline reached its target in 97 of 100 simulated episodes. Physical calibration and live service
acceptance are our next integration gates."

Keep the local-mode qualification in the recording. If a later live run passes its acceptance gates,
replace this talk track using that run's actual evidence and preserve its mode labels.

## Questions to answer precisely

| Question | Answer supported today |
| --- | --- |
| Is this learning? | New skills are reusable programs selected by tests; PPO separately trains a neural reaching policy. The offline demo uses fixed model-response fixtures, so it demonstrates the learning workflow rather than fresh model generation. |
| Can a candidate grade itself? | Its return value is insufficient. Our evaluator checks the independent success specification, collisions, joint-limit hits, and IK failures. |
| Where does generated code run? | In Daytona sandboxes when configured, or restricted local child processes with a minimal environment. It never executes inside the orchestrator and never receives the real driver. |
| Does simulation make motion safe? | It is one check. Physical joint/camera calibration, validated replay, measured grip behavior, and the STOP drill are still required. Unmodeled geometry and sim-to-real error remain limits. |
| Did the robot physically clear the table? | The cited five-run result is simulation plus fake replay and verification. We have not established physical clear-table acceptance. |
| What did RL learn? | Reaching and tracking in the MuJoCo environment. The selected artifact is the hand-written-reward PPO baseline, with 97/100 successes and zero falls at evaluation seed 0. Physical RL deployment is disabled. |
| Why preserve a skill? | A saved program can be re-planned for the current scene without generating six new programs. A saved skill still needs a new plan and validation. |
| How fast is it? | Five fixture-backed clear-table commands reached fake execution in 8.6, 10.2, 8.9, 9.9, and 11.8 seconds. This does not predict live xAI or Daytona latency. |
| What remains unsupported? | Unmeasured physical setups, unverified objects, and requests whose success cannot be expressed by the supported predicates. Cloth and liquids are outside the current demo scope. |

## Two-minute video storyboard

| Time | Picture | Narration / on-screen label |
| --- | --- | --- |
| 0:00-0:15 | Presenter and stationary kit; title card | "A personal skill library for a robot arm." Do not imply that a stationary physical arm is executing the adjacent simulation. |
| 0:15-0:25 | Dashboard showing mode banner and two starter skills | "Local simulation; fixture model replies; fake arm and cameras." |
| 0:25-0:55 | One continuous capture: type `clear the table`, show success spec, six candidate cards, rejection reasons, selected result, and saved skill | Explain test-first scoring. Keep the actual waiting period uncut; retain the clock and mode banner. |
| 0:55-1:10 | Type the command again; show saved-skill planning | "The saved program is re-planned for this scene." Do not claim a measured live-start latency. |
| 1:10-1:20 | Type `Fold my laundry`; show queued result | "Requests outside the supported task vocabulary are queued." |
| 1:20-1:35 | Generated PPO simulation clip | Persistent label: "Simulation. Selected hand-written-reward PPO baseline. 97/100, seed 0; zero falls." |
| 1:35-1:50 | Architecture diagram; highlight sandbox/local subprocess and validator boundaries | State where xAI and Daytona integrate; live service acceptance is tracked separately. |
| 1:50-2:00 | Current-status card and repository link | "Local workflow verified. Physical joint/camera calibration and supervised arm acceptance pending." |

The storyboard is prepared; the final video still needs recording, editing, export, and review.
Capture clean audio separately if needed. A physical STOP clip belongs only after the real drill has
been measured; the fake STOP button can be shown with its mode label.

## Recording and submission checklist

- [ ] Record the continuous local learn using [the isolated demo setup](demo.md).
- [ ] Capture saved-skill reuse, the queued task, and the PPO simulation clip.
- [ ] Keep fixture/simulation labels visible; add captions and check their timing.
- [ ] Verify every quoted number against the [verification record](verification-2026-10-17.md) and the
      generated table from `python scripts/story_numbers.py --markdown`, which names the file behind each number.
- [ ] Export a playable two-minute video and inspect the complete export with sound.
- [ ] Review [the Devpost draft](devpost-draft.md), team names, repository visibility, and media rights.
- [ ] Add the actual repository and video links, then publish from the team's account.
- [ ] Select the two sponsor challenges, Best Use of Daytona and Make It Legendary, only after checking the
      current event requirements and the evidence table in [the Devpost draft](devpost-draft.md#sponsor-challenges).

Do not claim measured Daytona sandbox counts, cloud training, live voice success, or Cursor use without
supporting records. Claude Code and Codex implementation assistance is recorded in repository history;
that attribution is separate from runtime model calls.
