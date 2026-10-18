"""`carry_to_hand(..., policy=...)` (the D7 staged bring-up): None follows config/rl.yaml (`policy_config`,
which stays `use_policy: false`), a dict in the `policy_config()` shape overrides it for that call only in
both directions, the override reaches `_run_policy` and `rl.policy_runner.run_policy`, and a malformed
override is a setup failure that moves nothing and freezes nothing. FakeDriver only, no torch."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from arm import real_robot, replay
from arm.driver import FakeDriver

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))


def scene_with_hand(x=0.30, y=0.02, z=0.03):
    scene = json.loads(json.dumps(FAKE_SCENE))
    scene["hand"].update({"x": x, "y": y, "z": z, "seen": True})
    return lambda: scene


def carry(**kw):
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=scene_with_hand(), realtime=False, **kw)
    return result, d


def test_config_keeps_the_policy_off_and_none_follows_it(monkeypatch):
    assert real_robot.policy_config()["use_policy"] is False  # config/rl.yaml: D7 flips it only after M4
    calls = []
    monkeypatch.setattr(real_robot, "_run_policy", lambda session, cfg, timeout, target: calls.append(cfg))
    result, d = carry()
    assert result["ok"] and "handoff_done" in result["events"] and calls == []
    result, d = carry(policy=None)
    assert result["ok"] and calls == []
    assert not any(e.startswith("policy") for e in result["events"])
    # None follows whatever the config says
    monkeypatch.setattr(real_robot, "policy_config", lambda: {"use_policy": True, "policy": "cfg.zip"})
    result, d = carry(policy=None)
    assert result["ok"], result["events"]
    assert calls == [{"use_policy": True, "policy": "cfg.zip"}]


def test_override_reaches_run_policy_for_that_call_only(monkeypatch):
    assert real_robot.policy_config()["use_policy"] is False
    calls = []

    def run_policy(driver, scene_fn, path, *, timeout, safety_scene, cancelled, clock, sleep, on_event):
        calls.append((driver, path, timeout))
        on_event("policy test reached target")
        return True

    monkeypatch.setitem(sys.modules, "rl.policy_runner", types.SimpleNamespace(run_policy=run_policy))
    override = {"use_policy": True, "policy": "rl/policies/test.zip"}
    result, d = carry(policy=override, timeout=7.0)
    assert result["ok"], result["events"]
    assert len(calls) == 1
    driver, path, timeout = calls[0]
    assert driver is d and path == str(ROOT / "rl" / "policies" / "test.zip") and timeout == 7.0
    assert "policy override: use_policy=True, policy=rl/policies/test.zip" in result["events"]
    assert "policy reached the hand" in result["events"] and "handoff_done" in result["events"]
    assert override == {"use_policy": True, "policy": "rl/policies/test.zip"}  # the caller's dict untouched
    # the next call without an override is back on the config path
    result, d = carry()
    assert result["ok"] and len(calls) == 1
    assert real_robot.policy_config()["use_policy"] is False


def test_override_is_the_cfg_that_run_policy_receives(monkeypatch):
    """With `_run_policy` itself replaced, the override dict is what it receives (no config lookup)."""
    calls = []
    monkeypatch.setattr(
        real_robot, "policy_config", lambda: pytest.fail("config/rl.yaml read despite the override")
    )
    monkeypatch.setattr(real_robot, "_run_policy", lambda session, cfg, timeout, target: calls.append(cfg))
    result, d = carry(policy={"use_policy": True, "policy": "x.zip"}, timeout=3.0)
    assert result["ok"], result["events"]
    assert calls == [{"use_policy": True, "policy": "x.zip"}]


def test_override_can_switch_the_config_policy_off(monkeypatch):
    monkeypatch.setattr(real_robot, "policy_config", lambda: {"use_policy": True, "policy": "cfg.zip"})
    calls = []
    monkeypatch.setattr(real_robot, "_run_policy", lambda session, cfg, timeout, target: calls.append(cfg))
    result, d = carry(policy={"use_policy": False, "policy": "cfg.zip"})
    assert result["ok"] and "handoff_done" in result["events"] and calls == []
    assert "policy override: use_policy=False, policy=cfg.zip" in result["events"]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"use_policy": True},
        {"policy": "x.zip"},
        {"use_policy": "yes", "policy": "x.zip"},
        {"use_policy": 1, "policy": "x.zip"},
        {"use_policy": True, "policy": ""},
        {"use_policy": True, "policy": None},
        {"use_policy": True, "policy": "x.zip", "extra": 1},
        {"use_policy": True, "policy": b"x.zip"},
        {"use_policy": True, "policy": Path("   ")},
        {"use_policy": True, 1: "x.zip"},
        "x.zip",
        7,
        [],
    ],
)
def test_bad_override_is_a_setup_failure_that_moves_nothing(monkeypatch, bad):
    monkeypatch.setattr(real_robot, "_run_policy", lambda *a: pytest.fail("policy ran on a bad override"))
    result, d = carry(policy=bad)
    assert result["reason"] == "error" and not result["ok"]
    assert d.commands == [] and not d.stopped and d.torque and d.read()[1] == 0.2
    assert any("ValueError" in e and "policy" in e for e in result["events"]), result["events"]
    with pytest.raises(ValueError, match="policy"):
        real_robot.checked_policy(bad)


def test_checked_policy_returns_a_fresh_copy():
    override = {"use_policy": True, "policy": "rl/policies/handoff_best.zip"}
    checked = real_robot.checked_policy(override)
    assert checked == override and checked is not override


def test_override_takes_a_pathlike_policy(monkeypatch):
    path = Path("rl") / "policies" / "staged.zip"
    checked = real_robot.checked_policy({"use_policy": True, "policy": path})
    assert checked == {"use_policy": True, "policy": str(path)}  # a str, the policy_config() shape
    calls = []
    monkeypatch.setattr(real_robot, "_run_policy", lambda session, cfg, timeout, target: calls.append(cfg))
    result, d = carry(policy={"use_policy": True, "policy": path})
    assert result["ok"], result["events"]
    assert calls == [{"use_policy": True, "policy": str(path)}]


def test_live_segment_keeps_the_config_path(monkeypatch):
    monkeypatch.setattr(real_robot, "policy_config", lambda: {"use_policy": True, "policy": "cfg.zip"})
    calls = []
    monkeypatch.setattr(real_robot, "_run_policy", lambda session, cfg, timeout, target: calls.append(cfg))
    d = FakeDriver(gripper0=0.2)
    t = {
        "version": 1,
        "skill": "test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [{"kind": "live", "primitive": "carry_to_hand", "object": "pill_bottle"}],
    }
    result = replay.replay(t, driver=d, scene=scene_with_hand(), realtime=False)
    assert result["ok"], result["events"]
    assert calls == [{"use_policy": True, "policy": "cfg.zip"}]
