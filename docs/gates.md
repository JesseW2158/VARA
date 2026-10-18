# Gate ledger

One row per gate in [schedule.md](schedule.md), in schedule order, plus the I2 sub-row that M1 depends on.
A gate's checkbox in schedule.md is ticked only after its row here is complete. The numbers in the
Evidence column come from `python scripts/story_numbers.py --markdown`, which reads `runs/`, `skills/` and
the RL provenance files and names the file behind every figure; the observations no tool records (who ran
it, which devices were real, what the ruler said) are written here by hand.

## Ticking rule

- Tick a task box in `docs/tasks/<lane>.md` only after its Done-when command ran on the demo laptop and the
  evidence path is in this ledger or in the task's own verification row.
- Tick a gate in `docs/schedule.md` only after the row below has status `pass`, a local timestamp, a name in
  Who, and an evidence path under `runs/` (or a saved log) that a second person can open.
- Status is one of `open` (no evidence yet), `partial`, `pass`, `fail` or `cut-<n>`.
- `partial` means part of the test passed, usually the software half with fakes (FAKE=1: fixture replies,
  local scoring, FakeDriver, FakeCamera), and never ticks a gate. The run folders record the scoring runner
  (`runner: local`, `daytona`, or `none` for a candidate that never reached a scorer) but not whether the
  arm, the cameras or the model were real, so the person who ran the test writes the mode in Notes.
- `cut-<n>` records a cut in the schedule's order (1 routines, 2 RL on the real arm, 3 the repair round,
  4 dashboard polish), with the standup time as the timestamp.
- Timestamps are local demo-laptop time (`YYYY-MM-DD HH:MM`, EDT, UTC-4). Folder stamps the scripts write
  (`runs/e2e_fake/<ts>`, `runs/archive/<ts>`, `runs/story/<ts>`, the `c7-bench-<ts>` folders) are UTC;
  the hand-named `runs/verification/final-completion/<ts>` folders are local.
- Rows are edited in place and never deleted. A failed attempt stays `fail` with its evidence until a later
  attempt replaces the status, and the earlier evidence path moves to Notes.
- Who is a person's name. The rows pre-filled below cite the unsigned
  [verification record](verification-2026-10-17.md); whoever reruns the check on the demo laptop replaces
  "unsigned" with their name and the new timestamp.

## Gates

| Gate | Deadline | Test | Status | Local timestamp | Who | Evidence path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | Sat 1:30 AM | `pytest -q` passes with nothing skipped; `python scripts/e2e_fake.py "put the juice box in the tray"` ends in `saved` | pass | 2026-10-17 16:09 | verification record, unsigned | `runs/verification/final-completion/20261017-160904/pytest.log` (1,406 passed, no skips, 310.96 s); `runs/verification/final-completion/20261017-160702/e2e-juice.log`; `runs/t-0012/events.jsonl` (candidate 3 selected, four of six passing, ends `saved` then `done`); `runs/e2e_fake/20261017-200702/skills/juice_box_to_tray.json` | FAKE=1 throughout. Latest full run 2026-10-17 20:45: 2,222 passed, no skips, 373.50 s (`runs/verification/resume-20261017/pytest-full-run5.log`) |
| I1 | Sat 2:00 AM | `/verify` passes on all four laptops; the FAKE=1 dashboard shows six cards for "put the juice box in the tray" | partial | 2026-10-17 16:09 | verification record, unsigned | demo laptop only: the S0 log above; `runs/verification/browser-final/1792266692/result.json` (`cards: 6`, `saved: true`, `escape_stop: true`) and `dashboard.png` beside it | three laptops open; per-laptop table below |
| M1a | Sat 3:00 AM | each lane's first deliverable (schedule.md team table) | partial | 2026-10-17 | verification record, unsigned | A: joint map measured 2026-10-17 21:39 (all five signs match the model; offsets +7.96, -1.58, -8.04, -1.06, +3.47 deg from the zero-pose reference, `runs/calibration/solve-20261018-013942-662672.json`); five-pose check 22:20 not formally passed: mean 2.2 cm, max 3.1 cm, 1 of 5 within 1.5 cm (`runs/calibration/a6-check/check-01.json`), after correcting a 4 cm origin offset the operator confirmed (x was measured from a point 4 cm behind the turning axis). Joint-offset fits do not improve it; the rest is consistent with centimeter-rounded tape measurements. Accepted by the operator to move on; the I2 touch test is the end-to-end accuracy check. B: open, the scene check needs the mounted overhead camera (`runs/setup/cameras/current/report.json` shows no markers). C: `python -m agents.router --eval --fake` 20/20 and local scoring of all six candidates (`runs/verification/20261017-183918/bench-demo.log`); a Daytona sandbox scoring a candidate: done 2026-10-17 21:14, `runs/live-gates/20261018-011414/summary.json` (smoke: runner `daytona`, 18.05 s create + score + delete, sandbox deleted, 0 left). D: baseline PPO trained, `rl/runs/baseline/summary.json` (507,904 steps) | software halves only; no lane's physical or keyed half is done |
| I2 | before M1 | `python scripts/touch_markers.py --real --loose` lands within 1.5 cm of every marker | open | | | fake runs only: `runs/touch_markers/results.json` and `runs/touch_markers_chain/results.json` (`mode: fake`, worst 0.15 cm) | needs A6, B5 and the mounted camera first |
| M1 (I3) | Sat 6:00 AM | a typed new task is learned with local scoring, replayed on the real arm, verified by the camera and saved: `skills/<name>.json` with `real.successes` 1 and `events.jsonl` ending `saved` | partial | 2026-10-17 16:07 | verification record, unsigned | `runs/t-0012/events.jsonl` (6.0 s from the first event to `executing`); `runs/e2e_fake/20261017-200702/skills/juice_box_to_tray.json` (`real: {runs: 1, successes: 1}`, counted on the FakeDriver) | fake arm and cameras, fixture replies; the real run needs the key in `.env`, I2 and `python -m agents.orchestrator --real` |
| RL go/no-go | Sat 7:00 AM | D-GO decided and announced | partial | 2026-10-17 (time not recorded) | D lane via dgo.json, unsigned | `rl/runs/baseline/dgo.json` (97/100 successes, 0 falls, seed 0, threshold 0.8, "GO for continued RL development"); `rl/policies/handoff_best.json` (same policy hash) | decided (GO), not yet announced: announce at a standup and add the time; simulation only |
| M2 | Sat 3:00 PM | a new typed task runs 6 Daytona candidates x 5 trials, the winner reaches the real arm within 75 s and is saved; repeating the command plans from the library with the arm moving within 5 s | partial | 2026-10-17 15:49 | verification record, unsigned | `runs/verification/c7-bench-20261017-194847/summary.json` (five learns, 8.6 / 10.2 / 8.9 / 9.9 / 11.8 s to executing, all saved); a library replay reached `executing` after 1.4 s in `runs/verification/routine-20261017-185154/runs/t-0002/events.jsonl` | `runner: local` on every scored candidate, fake arm. Daytona scoring measured 2026-10-17 21:12: `runs/live-gates/20261018-011134/summary.json` (bench, 7 sandboxes, 6 candidates x 5 trials per task: juice_box_to_tray 5.6 s, clear_table 10.4 s, place_in_tray 4.9 s, all `daytona`, 0 sandboxes left). Open: the live model (xAI answered 403 "Access to the chat endpoint is denied" for both models on 2026-10-17 21:10, `runs/live-gates/20261018-010951/ping.stdout.log`; fix the key's permissions in console.x.ai) and the real arm |
| M3 | Sat 4:00 PM | the same flow works by voice, the typed box still works, Esc freezes the arm within 0.2 s, saying "stop" freezes it within 2 s | open | | | software only: `web/voice.html`, `web/static/voice.js`, `tests/test_web_pages.py`; the dashboard Esc on the fake arm in `runs/verification/browser-final/1792266692/result.json` (`escape_stop: true`) | voice token from xAI works: 2026-10-17 21:10, `runs/live-gates/20261018-010951/voice.json` (HTTP 200, token length 119, value never recorded); the chat models answered 403 at the same time, so the voice session's tool calls need the key's chat permission fixed first. Esc on the real arm measured 2026-10-17 23:08 to 23:09: three presses during `--jog-joint shoulder_pan` motion froze it in 0, 16 and 0 ms (Windows' monotonic clock ticks every ~16 ms, so: at most 16 ms), `python -m arm.driver --esc-latency-report` PASS (`runs/calibration/esc-latency.jsonl`). Still open: the room test and the voice "stop" latency |
| M4 | Sat 6:00 PM | four reward variants trained in parallel sandboxes with results on the dashboard; the best policy brings the object to the wristband within 3 cm in 4 of 5 real tries, never over the speed limit | partial | 2026-10-17 12:42 | verification record, unsigned | `web/static/rl/curves.json` (5 series, chosen baseline, created 16:42 UTC); `rl/runs/parallel/result.json` (4 variants, `runner: local`, 151,552 steps each); `runs/verification/policy-runtime-20261017-194639.json` (20/20 reaches on the FakeDriver, seed 7) | trained locally, not in sandboxes; the Daytona RL snapshot `arm-rl-v1` was built and checked 2026-10-17 21:20 (`runs/rl/snapshot-check-20261018-012047.json`: torch 2.14.1+cpu, stable-baselines3 2.9.0, mujoco 3.16.0, sandbox deleted), so `python -m rl.train_parallel` can train the variants in sandboxes; real tries open (offline the policy aims the jaw tip, not the grasp center M4 measures); `config/rl.yaml` stays `use_policy: false` |
| M5 | Sat 9:00 PM | the demo script runs end to end in under 3 minutes with `DEMO_MODE=1`, three times in a row, Reset demo before each | open | | | fake rehearsal timing only: `runs/verification/c7-bench-20261017-194847/bench.log` | needs the taped layout and the physical reset checklist |
| M6 | Sun 1:00 AM | feature freeze: two clean runs in a row, then the backup video is recorded | open | | | planned path `runs/demo-video/backup.mp4` from [demo.md](demo.md) (gitignored; copy to a USB stick or a second laptop) | nothing recorded yet |
| M7 | Sun 4:30 AM | Devpost submitted with video, public repo, README and both sponsor challenges selected | open | | | `docs/devpost-draft.md` (submission fields block and the sponsor-challenge evidence table) | Best Use of Daytona and Make It Legendary; no Cursor screenshots and no keyed run in the repository yet |
| M8 | Sun 8:30 AM | judging table set up, calibration rechecked, voice tested in the judging room, `DEMO_MODE=1` on, Reset demo pressed | open | | | the `POST /calibrate` result (`error_cm` at most 1.5) and the dashboard mode banner, noted here by hand | |

## I1 per laptop

Each teammate runs `/verify` on their own laptop and fills one row; `python scripts/check_setup.py --out
runs/setup/check-<ts>.json` is the machine-readable half of the evidence.

| Laptop / owner | Lane | `pytest -q` | `e2e_fake` last stage | `ruff check .` | Dashboard six cards | Local timestamp | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Demo laptop (Windows 11, arms attached) | A | 2,222 passed, no skips, 373.50 s (2026-10-17 20:45; earlier 1,406) | `saved` | passed | yes | 2026-10-17 16:09 | `runs/verification/final-completion/20261017-160904/pytest.log`; `runs/verification/browser-final/1792266692/`; `runs/setup/final-check.json` |
| (laptop, owner) | B | | | | | | |
| (laptop, owner) | C | | | | | | |
| (laptop, owner) | D | | | | | | |
