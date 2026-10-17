# Teachable home-assistant arm (HackHarvard 2026)

A voice-taught SO-101 arm for people with limited mobility. New skills are AI-written programs scored in parallel Daytona simulations; the arm replays only a path that passed, and the overhead camera checks the result against a success test written before any code.

- Start here: [docs/kickoff.md](docs/kickoff.md)
- Brief for Claude Code: [CLAUDE.md](CLAUDE.md)
- Contracts: [docs/interfaces.md](docs/interfaces.md)
- Tasks by lane: [docs/tasks/](docs/tasks/)

The software pipeline is implemented and runs locally with fakes. See
[HANDOFF.md](HANDOFF.md) for the current implementation status and
[verification notes](docs/verification-2026-10-17.md) for measured results and remaining hardware work.

On the configured Windows demo laptop, use the project's Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/e2e_fake.py "put the juice box in the tray"
```

The end-to-end command forces fake hardware and services, scores six candidates in local MuJoCo
subprocesses, validates and replays the winning path, verifies the result, and saves the learned skill
in a copied library under `runs/e2e_fake/`. The normal skill library is preserved.

To open the dashboard with fake devices and services:

```powershell
$env:FAKE = "1"
.\.venv\Scripts\python.exe -m agents.orchestrator --port 8000
```

Open `http://localhost:8000` for the dashboard or `/voice` for the voice interface. Fake mode supplies
fixture replies and a placeholder voice token; a live voice session requires an xAI key. Environment
options are documented in [.env.example](.env.example).

The connected SO-ARM101 Pro kit uses leader COM7 at 5 V and follower COM8 at 12 V. Both arms already
have LeRobot calibration files. VARA's joint-to-simulation offsets still require the A6 measurements,
and the overhead camera needs physical calibration before Cartesian replay. Follow
[the arm tasks](docs/tasks/A-arm.md) and [safety rules](docs/safety.md) for hardware bring-up. RL
deployment remains disabled in `config/rl.yaml` until its real-arm gates pass.

Capture both arms' current registers without motion using
`python -m arm.inspect_hardware --follower-port COM8 --leader-port COM7 --out runs/hardware/check-01.json`.
See [hardware calibration notes](docs/hardware-calibration.md) for report interpretation and the
newly detected leader calibration mismatch.
