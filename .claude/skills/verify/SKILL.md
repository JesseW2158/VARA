---
name: verify
description: Run the no-hardware checks (pytest, the fake end-to-end run, ruff) and report each result with its output.
disable-model-invocation: true
---
1. Run `pytest -q`.
2. Run `python scripts/e2e_fake.py "put the juice box in the tray"` and confirm the last stage printed is `saved`.
3. Run `ruff check .`.

Report each command with the last lines of its output and a pass or fail. Don't fix anything unless I ask.
