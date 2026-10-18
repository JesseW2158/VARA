"""Shared test setup: tests never depend on the operator's tuning settings in .env.

agents/orchestrator.py calls load_dotenv() at import, so whatever the operator set in .env for the live
dashboard (more repair rounds, a different arm gain) would leak into every later test. Each test starts
without them; a test that needs one sets it with monkeypatch.
"""

from __future__ import annotations

import pytest

OPERATOR_TUNING = ("VARA_REPAIR_ROUNDS", "VARA_ARM_P")


@pytest.fixture(autouse=True)
def _no_operator_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OPERATOR_TUNING:
        monkeypatch.delenv(name, raising=False)
