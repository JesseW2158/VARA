"""FakeDriver and the shared safety layer in arm/driver.py (A4, A13). No hardware, no lerobot import."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from arm import driver
from arm.driver import (
    JOINTS,
    MARGIN_RAD,
    MAX_STEP_RAD,
    TICK,
    EscWatcher,
    FakeDriver,
    get_driver,
    reset_driver,
)

ROOT = Path(__file__).resolve().parents[1]
Q0 = [0.0, -1.0, 0.5, 1.0, 0.0]


@pytest.fixture(autouse=True)
def default_config(monkeypatch, tmp_path):
    """Pin the MJCF fallback limits and the identity joint map, whatever config/ holds right now."""
    monkeypatch.setattr(driver, "LIMITS_PATH", tmp_path / "no_limits.yaml")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", tmp_path / "no_joint_map.yaml")
    reset_driver()
    yield
    reset_driver()


def make() -> FakeDriver:
    return FakeDriver(q0=Q0, gripper0=1.0)


def test_constants():
    assert JOINTS == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    assert TICK == pytest.approx(1 / 30)
    assert MAX_STEP_RAD == pytest.approx(math.radians(2.0))
    assert MARGIN_RAD == pytest.approx(math.radians(3.0))


def test_limits_fallback_and_file(tmp_path, monkeypatch):
    lim = driver.limits()
    assert set(lim) == set(JOINTS)
    assert lim["wrist_roll"] == [-2.7438, 2.8412]
    lim["wrist_roll"][0] = 0.0  # callers get their own copy
    assert driver.limits()["wrist_roll"][0] == -2.7438

    path = tmp_path / "limits.yaml"
    path.write_text("\n".join(f"{j}: [-1.0, 1.0]" for j in JOINTS), encoding="utf-8")
    monkeypatch.setattr(driver, "LIMITS_PATH", path)
    assert driver.limits()["shoulder_pan"] == [-1.0, 1.0]
    path.write_text("shoulder_pan: [1.0, -1.0]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        driver.limits()


def test_joint_map_fallback_and_file(tmp_path, monkeypatch):
    jm = driver.joint_map()
    assert all(jm[j] == {"sign": 1, "offset_deg": 0.0} for j in JOINTS)
    assert jm["gripper"] == {"closed_pct": 0.0, "open_pct": 100.0}
    assert jm["lerobot_id"] == "my_awesome_follower_arm"

    path = tmp_path / "joint_map.yaml"
    path.write_text(
        "shoulder_pan: {sign: -1, offset_deg: 12.5}\ngripper: {closed_pct: 4, open_pct: 96}\n"
        "lerobot_id: other_arm\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", path)
    jm = driver.joint_map()
    assert jm["shoulder_pan"] == {"sign": -1, "offset_deg": 12.5}
    assert jm["elbow_flex"] == {"sign": 1, "offset_deg": 0.0}
    assert jm["gripper"] == {"closed_pct": 4.0, "open_pct": 96.0}
    assert jm["lerobot_id"] == "other_arm"
    path.write_text("shoulder_pan: {sign: 2}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        driver.joint_map()


def test_read_units_and_initial_state():
    d = make()
    q, g = d.read()
    assert q == Q0 and g == 1.0
    assert d.is_fake and d.connected and d.torque and not d.stopped
    assert d.last_command is None
    home = FakeDriver()
    assert len(home.read()[0]) == 5  # home_q from config/scene.yaml, or zeros


def test_command_sets_state_and_read_returns_radians_and_unit_gripper():
    d = make()
    target = [v + 0.01 for v in Q0]
    assert d.command(target, 0.4) is True
    q, g = d.read()
    assert q == pytest.approx(target)
    assert g == pytest.approx(0.4)
    assert d.last_command == (pytest.approx(target), pytest.approx(0.4))
    assert all(abs(v) < 2 * math.pi for v in q) and 0.0 <= g <= 1.0


def test_step_clip_two_degrees_per_call():
    d = make()
    big = list(Q0)
    big[0] = Q0[0] + math.radians(30)
    big[1] = Q0[1] - math.radians(30)
    d.command(big, 1.0)
    q, _ = d.read()
    assert q[0] == pytest.approx(Q0[0] + MAX_STEP_RAD)
    assert q[1] == pytest.approx(Q0[1] - MAX_STEP_RAD)
    assert q[2:] == pytest.approx(Q0[2:])
    # the next call measures from the last commanded position, not from the original pose
    d.command(big, 1.0)
    assert d.read()[0][0] == pytest.approx(Q0[0] + 2 * MAX_STEP_RAD)


def test_limit_clip_with_three_degree_margin():
    d = make()
    lim = driver.limits()
    hi = lim["shoulder_pan"][1]
    # walk toward the limit 2 degrees at a time; it must stop MARGIN_RAD short of it
    for _ in range(200):
        d.command([hi + 1.0, *Q0[1:]], 1.0)
    assert d.read()[0][0] == pytest.approx(hi - MARGIN_RAD)
    lo = lim["shoulder_lift"][0]
    for _ in range(200):
        d.command([d.read()[0][0], lo - 5.0, *Q0[2:]], 1.0)
    assert d.read()[0][1] == pytest.approx(lo + MARGIN_RAD)


def test_reference_outside_limits_moves_back_two_degrees_not_a_jump():
    d = make()
    lim = driver.limits()
    hi = lim["shoulder_pan"][1]
    d.set_present([hi + math.radians(10), *Q0[1:]])  # pushed past the limit by hand
    d.command([0.0, *Q0[1:]], 1.0)
    assert d.read()[0][0] == pytest.approx(hi + math.radians(10) - MAX_STEP_RAD)


def test_gripper_clipped_to_unit_interval_without_step_limit():
    d = make()
    d.command(Q0, 7.0)
    assert d.read()[1] == 1.0
    d.command(Q0, -3.0)
    assert d.read()[1] == 0.0  # a full close in one call: the gripper has no per-call cap
    d.command(Q0, 0.35)
    assert d.read()[1] == pytest.approx(0.35)


def test_command_rejects_wrong_joint_count():
    d = make()
    with pytest.raises(ValueError):
        d.command([0.0, 0.0], 1.0)


def test_freeze_ignores_commands_and_resume_accepts_them():
    d = make()
    moved = list(Q0)
    moved[2] += 0.01
    d.command(moved, 0.5)
    d.freeze()
    assert d.stopped and d.torque and d.frozen_at is not None
    assert d.last_command is None
    before = d.read()
    assert d.command([v + 0.01 for v in Q0], 1.0) is False
    assert d.read() == before  # held where it was, gripper included
    d.resume()
    assert not d.stopped
    assert d.command([v + 0.01 for v in Q0], 1.0) is True
    assert d.read()[0] == pytest.approx([v + 0.01 for v in Q0])


def test_torque_off_blocks_and_torque_on_resyncs_and_unblocks():
    d = make()
    d.command(Q0, 1.0)
    d.torque_off()
    assert d.torque is False and d.last_command is None
    assert d.command([v + 0.01 for v in Q0], 1.0) is False
    assert d.read()[0] == Q0
    # with torque off the arm was moved by hand
    pushed = [v + math.radians(20) for v in Q0]
    d.set_present(pushed)
    syncs = d.goal_syncs
    d.torque_on()
    assert d.torque is True and d.goal_syncs == syncs + 1 and d.last_command is None
    # after the resync the 2 degree step is measured from where the arm really is
    assert d.command(Q0, 1.0) is True
    assert d.read()[0] == pytest.approx([v - MAX_STEP_RAD for v in pushed])


def test_freeze_with_torque_off_powers_and_holds():
    d = make()
    d.torque_off()
    d.freeze()
    assert d.stopped and d.torque
    assert d.command(Q0, 1.0) is False
    d.resume()
    assert d.command(Q0, 1.0) is True


def test_disconnect_keeps_torque_unless_asked():
    d = make()
    d.disconnect()
    assert d.torque and not d.connected
    with pytest.raises(RuntimeError):
        d.command(Q0, 1.0)
    d.connect()
    d.disconnect(torque_off=True)
    assert not d.torque and not d.connected


def test_close_gripper_hold_backs_off_three_percent():
    d = make()
    d.command(Q0, 1.0)
    d.gripper_reading_override = 0.20  # a held object stops the jaws here
    reading = d.close_gripper_hold()
    assert reading == pytest.approx(0.20)
    assert d.commands[-2] == (pytest.approx(Q0), pytest.approx(0.0))  # the close
    assert d.commands[-1] == (pytest.approx(Q0), pytest.approx(0.17))  # hold at reading - 3 on 0-100
    d.gripper_reading_override = None
    d.commands.clear()
    d.command(Q0, 1.0)
    assert d.close_gripper_hold(target=0.0) == pytest.approx(0.0)  # nothing between the jaws
    assert d.commands[-1][1] == pytest.approx(0.0)


def test_close_gripper_hold_refused_while_stopped():
    d = make()
    d.freeze()
    n = len(d.commands)
    assert d.close_gripper_hold() == pytest.approx(1.0)
    assert len(d.commands) == n


def test_get_driver_singleton_honors_fake_flags(monkeypatch):
    monkeypatch.setenv("FAKE_ROBOT", "1")
    monkeypatch.delenv("FAKE", raising=False)
    reset_driver()
    d = get_driver()
    assert isinstance(d, FakeDriver) and d.is_fake
    assert get_driver() is d
    reset_driver()
    monkeypatch.delenv("FAKE_ROBOT")
    monkeypatch.setenv("FAKE", "1")
    assert isinstance(get_driver(), FakeDriver)
    reset_driver()
    monkeypatch.delenv("FAKE")
    real = get_driver()
    assert isinstance(real, driver.Driver) and not real.is_fake and not real.connected


def test_lock_is_reentrant_and_freeze_waits_for_a_command_in_flight():
    d = make()
    assert isinstance(d.lock, type(threading.RLock()))
    with d.lock, d.lock:
        pass
    entered = threading.Event()
    release = threading.Event()
    original_send = d._send

    def slow_send(q, g):
        entered.set()
        release.wait(2.0)
        original_send(q, g)

    d._send = slow_send  # type: ignore[method-assign]
    t = threading.Thread(target=d.command, args=([v + 0.01 for v in Q0], 1.0))
    t.start()
    assert entered.wait(2.0)
    f = threading.Thread(target=d.freeze)
    f.start()
    time.sleep(0.05)
    assert not d.stopped  # freeze is waiting for the lock held by the command
    release.set()
    t.join(2.0)
    f.join(2.0)
    assert d.stopped and d.read()[0] == pytest.approx([v + 0.01 for v in Q0])


def test_esc_watcher_calls_freeze():
    d = make()
    keys = iter(["x", EscWatcher.ESC])
    w = EscWatcher(d.freeze, poll=0.005, key_source=lambda: next(keys, None))
    w.start()
    deadline = time.time() + 2.0
    while not d.stopped and time.time() < deadline:
        time.sleep(0.005)
    w.stop()
    assert w.pressed and w.pressed_at is not None and d.stopped
    assert d.frozen_at is not None and d.frozen_at - w.pressed_at < 0.2


def test_jog_on_fake_moves_each_joint_and_returns():
    d = make()
    assert driver.jog(d, deg=10.0, realtime=False) is True
    assert d.read()[0] == pytest.approx(Q0)
    peaks = [max(c[0][i] for c in d.commands) for i in range(5)]
    assert peaks == pytest.approx([v + math.radians(10) for v in Q0])


def test_jog_stops_when_frozen():
    d = make()
    d.freeze()
    assert driver.jog(d, deg=10.0, realtime=False) is False
    assert d.commands == []


def test_cli_refuses_hardware_without_real_flag():
    # a clean child environment: no FAKE flags, no .env side effects (python-dotenv never overrides, and
    # the --jog run without --real must refuse before load_dotenv is even reached)
    env = {k: os.environ.get(k, "") for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")} | {
        "PYTHONPATH": str(ROOT)
    }
    r = subprocess.run(
        [sys.executable, "-m", "arm.driver", "--jog"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 2
    assert "--real" in r.stderr
    r = subprocess.run(
        [sys.executable, "-m", "arm.driver", "--jog", "--fake"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "fake jog ok" in r.stdout


def test_non_finite_targets_are_refused_before_anything_is_sent():
    """NaN compares False against every bound, so clipping cannot contain it; it must be rejected."""
    d = make()
    d.command(Q0, 1.0)
    before = (d.read(), d.last_command, len(d.commands))
    for bad_q, bad_g in (
        ([math.nan, *Q0[1:]], 1.0),
        ([math.inf, *Q0[1:]], 1.0),
        ([*Q0[:4], -math.inf], 1.0),
        (Q0, math.nan),
        (Q0, math.inf),
        (Q0, -math.inf),
    ):
        with pytest.raises(ValueError, match="non-finite"):
            d.command(bad_q, bad_g)
    assert (d.read(), d.last_command, len(d.commands)) == before
    # the next command still measures its 2 degree step from the last real position
    far = list(Q0)
    far[0] += 1.5
    assert d.command(far, 1.0) is True
    assert d.read()[0][0] == pytest.approx(Q0[0] + MAX_STEP_RAD)
    # clip() refuses a non-finite reference too (a corrupt present position)
    with pytest.raises(ValueError):
        d.clip(Q0, 1.0, [math.nan] * 5)
    # garbage is refused even while stopped: never a silent False
    d.freeze()
    with pytest.raises(ValueError):
        d.command([math.nan, *Q0[1:]], 1.0)


def test_close_gripper_hold_backs_off_toward_closed_with_inverted_map(tmp_path, monkeypatch):
    """The 3-unit backoff is applied in our 0-1 direction (toward closed), not along LeRobot's 0-100 axis."""
    path = tmp_path / "joint_map.yaml"
    path.write_text("gripper: {closed_pct: 100, open_pct: 0}\n", encoding="utf-8")  # servo runs the other way
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", path)
    d = make()
    d.command(Q0, 1.0)
    d.gripper_reading_override = 0.20
    assert d.close_gripper_hold() == pytest.approx(0.20)
    assert d.commands[-1][1] == pytest.approx(0.17)  # more closed than the contact reading
    assert driver._gripper_to_pct(0.17, driver.joint_map()) == pytest.approx(83.0)  # 80 + 3 toward closed
    # a narrower span on LeRobot's scale: still 3 units there
    path.write_text("gripper: {closed_pct: 10, open_pct: 70}\n", encoding="utf-8")
    d.commands.clear()
    d.command(Q0, 1.0)
    d.close_gripper_hold()
    assert d.commands[-1][1] == pytest.approx(0.20 - 3 / 60)


def test_empty_or_scalar_config_file_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    """A truncated limits.yaml after A6 must not silently revert to the wider MJCF ranges."""
    lim = tmp_path / "limits.yaml"
    lim.write_text("# calibrated ranges go here\n", encoding="utf-8")
    monkeypatch.setattr(driver, "LIMITS_PATH", lim)
    with pytest.raises(ValueError, match="empty"):
        driver.limits()
    lim.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        driver.limits()
    jm = tmp_path / "joint_map.yaml"
    jm.write_text("", encoding="utf-8")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", jm)
    with pytest.raises(ValueError, match="empty"):
        driver.joint_map()


def test_limits_file_rejects_non_finite_and_too_narrow_ranges(tmp_path, monkeypatch):
    lim = tmp_path / "limits.yaml"
    monkeypatch.setattr(driver, "LIMITS_PATH", lim)
    base = "\n".join(f"{j}: [-1.0, 1.0]" for j in JOINTS[1:]) + "\n"
    lim.write_text(base + "shoulder_pan: [-.inf, .inf]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shoulder_pan"):
        driver.limits()
    lim.write_text(base + "shoulder_pan: [0.0, 0.05]\n", encoding="utf-8")  # 2.9 degrees wide: no room
    with pytest.raises(ValueError, match="shoulder_pan"):
        driver.limits()
    lim.write_text(base + "shoulder_pan: [0.0, 0.2]\n", encoding="utf-8")
    assert driver.limits()["shoulder_pan"] == [0.0, 0.2]


def test_config_rewrite_is_picked_up_without_clearing_the_cache(tmp_path, monkeypatch):
    """The cache is keyed on content, so a same-size rewrite within one mtime tick is still re-read."""
    lim = tmp_path / "limits.yaml"
    monkeypatch.setattr(driver, "LIMITS_PATH", lim)
    lim.write_text("\n".join(f"{j}: [-1.0, 1.0]" for j in JOINTS), encoding="utf-8")
    assert driver.limits()["shoulder_pan"] == [-1.0, 1.0]
    lim.write_text("\n".join(f"{j}: [-1.5, 1.5]" for j in JOINTS), encoding="utf-8")
    assert driver.limits()["shoulder_pan"] == [-1.5, 1.5]
    lim.unlink()
    assert driver.limits()["shoulder_pan"] == driver.DEFAULT_LIMITS["shoulder_pan"]


def test_malformed_home_q_warns_and_falls_back_to_zeros(tmp_path, monkeypatch, caplog):
    sc = tmp_path / "scene.yaml"
    sc.write_text("home_q: [0, 1, 2, 3, 4, 5]\n", encoding="utf-8")  # six entries: not our joint vector
    monkeypatch.setattr(driver, "SCENE_PATH", sc)
    with caplog.at_level(logging.WARNING, logger="arm.driver"):
        assert FakeDriver().read()[0] == [0.0] * 5
    assert "home_q" in caplog.text
    caplog.clear()
    sc.write_text("home_q: [0.0, -1.35, 0.02, 1.45, 0.0]\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="arm.driver"):
        assert FakeDriver().read()[0] == [0.0, -1.35, 0.02, 1.45, 0.0]
    assert caplog.text == ""


def test_jog_zero_degrees_moves_nothing_and_fractional_amplitude_is_exact():
    d = make()
    assert driver.jog(d, deg=0.0, realtime=False) is True
    assert d.commands == []
    assert driver.jog(d, deg=2.5, realtime=False) is True
    peaks = [max(c[0][i] for c in d.commands) for i in range(5)]
    assert peaks == pytest.approx([v + math.radians(2.5) for v in Q0])
    assert d.read()[0] == pytest.approx(Q0)
    assert len(d.commands) == 5 * 6  # per joint: 1, 2, 2.5, 2, 1, 0 degrees
    with pytest.raises(ValueError):
        driver.jog(d, step_deg=0.0, realtime=False)
