---
paths:
  - "agents/**"
  - "sim/**"
  - "scripts/daytona_smoke.py"
---
# Lane C rules
- Never import, exec or eval AI-written code in this process. Run it only through `python -m sim.run_candidate` in a sandbox, or in a local subprocess whose env holds only PATH and PYTHONPATH.
- Validate every model reply against schemas/ before using it; on failure, retry once with the validation error in the prompt.
- pytest uses FAKE_LLM and FAKE_SANDBOX. Never call xAI or Daytona from tests.
- Pool sandboxes use auto_stop_interval=0 and are deleted on shutdown (atexit plus signal handlers). Cap: 7 sandboxes.
- DEMO_MODE budgets in docs/interfaces.md are hard limits, not targets.
- Write everything a task produces to runs/<task_id>/.
