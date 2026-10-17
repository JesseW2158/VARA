# Start here

Every laptop should be building against the same contracts by 1:30 AM. Unzip the kit into a fresh repo, have Lane C run the scaffold task, then give each lane its own branch and Claude Code session.

**Kickoff, in order:**

1. **Repo (C, 5 min).** Create a private GitHub repo, unzip `starter-kit.zip` at its root, commit to `main`, push, and add the other three.
2. **Secrets (everyone, 2 min).** Copy `.env.example` to `.env` and fill in `XAI_API_KEY`, `DAYTONA_API_KEY`, `ROUTER_MODEL`, `CODEGEN_MODEL` and the arm ports. `.env` is gitignored, and the project settings deny Claude Code reading it.
3. **Python (everyone).** Python 3.12 venv, then `pip install -r requirements.txt`. The demo laptop also installs `"lerobot[feetech]"`; Lane D also installs `requirements-rl.txt`.
4. **Claude Code (everyone).** Run `claude` in the repo root (inside Cursor's terminal if you want Cursor credit; see below). Run `/context` once to confirm `CLAUDE.md` loaded.
5. **S0 scaffold (C, by Sat 1:30 AM).** On `main`, switch to plan mode (Shift+Tab) and run `/task S0`. It builds every module as a working fake and makes `pytest -q` pass. Push.
6. **Branches (everyone, after S0).** `git switch -c lane-a` (or b, c, d). One Claude Code session per lane.
7. **I1 fake end-to-end (all, by Sat 2:00 AM).** `python scripts/e2e_fake.py "put the juice box in the tray"` prints every stage through `saved` on every laptop. From then on, each lane swaps one fake for the real thing behind the same contract.

Lanes A and B keep doing hardware (A1 to A3, A11, B1 to B4) during steps 1 to 6; none of it waits for the scaffold.

**Driving Claude Code:**

- Start each task with `/task <ID>`, for example `/task A7`. It reads the task and the contracts it names, edits only the listed files, runs the task's "Done when" command until it passes, ticks the box and commits.
- Use plan mode for tasks that touch three or more files (S0, A8, B9, C6, C9). Run `/clear` between tasks; after two failed corrections, `/clear` and re-prompt with what you learned.
- Run `/verify` before every merge to `main`, and merge at each gate.
- Contracts change only by agreement. Edits to `docs/interfaces.md`, `schemas/` and `tests/contract/` trigger an approval prompt. Agree at a standup, then one person changes all three and pushes to `main`.
- Commands that move the real arm carry `--real`. Claude Code must ask before running them, and a human answers only after checking the arm is clear.
- Tests never call xAI, Daytona or the arm; fakes stand in (Contracts, Fakes).

**Cursor and Make It Legendary.** The challenge requires the project be built with Cursor. Run Claude Code in Cursor's integrated terminal, use Cursor's agent for some tasks (the dashboard is a good fit), and keep screenshots for the README. `.cursor/rules/project.mdc` points Cursor at the same files. Ask the sponsor table what evidence counts.

**Kit contents:**

| Path | What it holds |
| --- | --- |
| `CLAUDE.md` | Project brief loaded into every session |
| `.claude/settings.json` | Permissions: deny reading `.env`; ask before contract edits, `--real` runs, the Daytona smoke test and `git push` |
| `.claude/skills/task/`, `.claude/skills/verify/` | The `/task <ID>` and `/verify` commands |
| `.claude/rules/*.md` | Lane rules that load only when Claude touches that lane's files |
| `.cursor/rules/project.mdc` | The same pointers for Cursor |
| `docs/interfaces.md`, `schemas/`, `tests/contract/`, `tests/fixtures/` | The contracts in prose, JSON Schema and tests |
| `docs/tasks/*.md` | Ordered tasks per lane, each with a Done-when command |
| `docs/schedule.md`, `demo.md`, `safety.md`, `risks.md`, `pitch.md`, `sources.md` | The event plan |
| `config/objects.yaml`, `config/scene.yaml`, `config/fake_scene.json`, `.env.example`, `requirements*.txt`, `.gitignore` | Starting configuration |
