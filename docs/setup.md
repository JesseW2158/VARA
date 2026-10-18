# Inspect the local setup

From the repository root on the Windows demo laptop:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
```

The checker discovers installed modules, parses local configuration mappings, checks artifact
presence, and reports the configured ports and camera indexes. It never opens serial ports,
cameras, or network connections. It loads `.env` through `python-dotenv` and reports API keys
only as configured/not configured booleans; it does not authenticate them.

To save a report and include an earlier hardware inspection:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py --hardware-report runs/hardware/20261017-continuation.json --out runs/setup/check-01.json
```

Choose a new output path each time. Historical hardware reports are labelled historical and
cannot establish the arms' current state. Missing or malformed historical reports appear as
diagnostic errors without opening hardware.

Exit codes:

- `0`: required offline modules, configuration mappings, and artifacts are present. This checks
  availability, not working imports, simulation outcomes, service credentials, or physical calibration.
- `1`: an offline prerequisite is missing or invalid. With `--require-live-config`, outstanding live
  configuration blockers also produce `1`.
- `2`: invalid arguments or an output path that cannot be created without overwriting a file.

Manual A6 joint mapping, A11 grip measurements, B4/B5 camera calibration, and the A13 STOP drill
always remain explicitly unverified by this command, even when `--require-live-config` exits `0`;
review their evidence separately. This command never emits a motion-readiness certificate.
Removing placeholder comments cannot establish calibration, and the presence of
`homography.npy` cannot establish that it came from the physical table.

The report starts with a `host` block (`hostname`, `os`, `python`, `git_head`, `git_dirty`, `utc`), so a
saved report says which laptop and which commit it describes.

For the motor inspection workflow and measured encoder mismatch, see
[hardware calibration](hardware-calibration.md).

## Verify a laptop (I1)

`scripts/verify.py` is the `/verify` check with the venv's interpreter named explicitly. It runs the whole
test suite (any skipped, xfailed or failed test fails it), the fake end-to-end run `put the juice box in the
tray` (its last stage must be `saved`) and `ruff check .`, then writes
`runs/verify/<hostname>-<timestamp>.json` with the host block, each step's exit code, duration and output
tail, and full logs beside it. Exit 0 only when every step passed. A full run takes about seven minutes:

```powershell
.\.venv\Scripts\python.exe scripts/verify.py
```

`--skip-e2e`, `--only pytest|e2e|ruff` and `--pytest-args "tests/test_setup_check.py"` make partial
reruns; such a record says `complete: false`. Record each laptop's result in the I1 table of
[gates.md](gates.md). The key-dependent gates (model, Daytona and the voice token) are separate: see
[live gates](live-gates.md).

## Live model configuration

Defaults were checked against xAI's official documentation on 2026-10-17:

| Pipeline call | Default model | Reasoning effort | Visible output limit |
| --- | --- | --- | --- |
| Routing and success specification | `grok-4.3` | `none` | 2,048 tokens |
| Candidate generation and repair | `grok-4.7` | `low` | 8,192 tokens per candidate |
| Four reward variants | `grok-4.7` | `low` | 16,384 tokens |

This is a latency-oriented choice for the project's 10–15 second routing and 30–45 second candidate
budgets, not a measured service latency guarantee. Grok 4.3 supports disabling reasoning; Grok 4.7
supports `low` but otherwise defaults to `high`. The latter cannot disable reasoning.
[Grok 4.3 documentation](https://docs.x.ai/developers/models/grok-4.3),
[Grok 4.7 documentation](https://docs.x.ai/developers/models/grok-4.7).

Set `ROUTER_MODEL` or `CODEGEN_MODEL` to override the model. Optional `ROUTER_REASONING_EFFORT` and
`CODEGEN_REASONING_EFFORT` override effort; choose a value the selected model supports. Unknown model
overrides receive no automatic reasoning parameter. Blank values use the choices above. Earlier fast
and Grok 4 release slugs have documented redirects, so explicit current choices make behavior easier to
review. [xAI retirement guide](https://docs.x.ai/developers/migration/may-15-retirement).

The adapter retains the OpenAI-compatible Chat Completions endpoint and its JSON-schema fallback.
`max_completion_tokens` limits visible output only; it does **not** cap hidden reasoning. Requests,
backoff and validation retries share the existing time budget, and a reply reported as truncated is
rejected. [Chat Completions reference](https://docs.x.ai/developers/rest-api-reference/inference/chat-completions).

`python -m agents.llm --ping` checks authenticated access when fake mode is disabled. It makes paid API
calls and does not establish routing accuracy or candidate pass rates; those require the existing live
evaluation commands. Fixture-backed tests make no API calls.
