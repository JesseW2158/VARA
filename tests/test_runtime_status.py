"""Runtime provenance never confuses fake execution with physical or cloud evidence."""

import asyncio
from types import SimpleNamespace

import pytest

from agents import orchestrator


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    for flag in ("FAKE", "FAKE_ROBOT", "FAKE_CAMERA", "FAKE_LLM", "FAKE_SANDBOX", "DAYTONA_API_KEY"):
        monkeypatch.delenv(flag, raising=False)


def test_fake_runtime_uses_explicit_simulation_labels(monkeypatch):
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "secret-must-not-appear")
    monkeypatch.setattr(orchestrator.orch, "driver", SimpleNamespace(is_fake=True, connected=True))
    assert asyncio.run(orchestrator.get_runtime()) == {
        "robot": "simulation",
        "camera": "simulated",
        "model": "fixtures",
        "scoring": "local",
    }


@pytest.mark.parametrize("connected", [False, True])
def test_real_runtime_distinguishes_connection_and_fallback(monkeypatch, connected):
    monkeypatch.setattr(orchestrator.orch, "driver", SimpleNamespace(is_fake=False, connected=connected))
    result = asyncio.run(orchestrator.get_runtime())
    assert result["robot"] == ("physical" if connected else "physical, disconnected")
    assert result["scoring"] == "local"
    monkeypatch.setenv("DAYTONA_API_KEY", "secret-must-not-appear")
    result = asyncio.run(orchestrator.get_runtime())
    assert result["scoring"] == "Daytona with local fallback"
    assert "secret" not in str(result)
