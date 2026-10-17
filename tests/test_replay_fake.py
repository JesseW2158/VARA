"""arm.replay.replay() on the FakeDriver (A9, A13): a validate failure moves nothing, waypoints stream in
order, bridging at 1 degree per tick, STOP mid-path, a missed grasp, the hand checks, live primitives and
driver errors. No hardware, no network, no sleeping (realtime off, simulated clock)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arm import driver as driver_mod
from arm import ik, real_robot, replay
from arm.driver import FakeDriver

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
HOME = [float(v) for v in ik.home_q()]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
LIMITS = replay.load_limits()
BOX = replay.load_box()


def solve(xyz, pitch=-math.pi / 2, q0=None):
    q, ok = ik.solve(xyz, pitch, q0=q0)
    assert ok, xyz
    return q


@pytest.fixture(scope="module")
def poses():
    pick = solve((0.24, -0.04, 0.0525))
    return {
        "pick": pick,
        "up": solve((0.24, -0.04, 0.15), q0=pick),
        "left": solve((0.20, 0.10, 0.08)),
        "right": solve((0.20, -0.10, 0.08)),
    }


def sweep(q_a, q_b, step=0.5 * DEG):
    """A path segment from q_a to q_b with small joint steps (tip steps stay under the 0.2 m/s cap)."""
    return [list(q_a), *replay.bridge(q_a, q_b, step)]


def traj(*segments):
    return {
        "version": 1,
        "skill": "test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": list(segments),
    }


def path(points):
    return {"kind": "path", "q": [[float(v) for v in q] for q in points]}


class Scene:
    """A scene callable the test controls: `hand(x, y, z)` shows the wristband, `hide()` hides it."""

    def __init__(self) -> None:
        self.scene = json.loads(json.dumps(FAKE_SCENE))
        self.calls = 0

    def hand(self, x: float, y: float, z: float = 0.03) -> None:
        self.scene["hand"].update({"x": x, "y": y, "z": z, "seen": True})

    def hide(self) -> None:
        self.scene["hand"]["seen"] = False

    def __call__(self) -> dict:
        self.calls += 1
        return self.scene


def tips(driver):
    return [ik.fk_tip(q) for q, _ in driver.commands]


def run(t, driver, **kw):
    return replay.replay(t, driver=driver, realtime=False, **kw)


def assert_same_path(commands, points, atol=1e-9):
    assert len(commands) == len(points), (len(commands), len(points))
    for k, (a, b) in enumerate(zip(commands, points, strict=True)):
        assert a == pytest.approx(b, abs=atol), k


def test_validate_failure_moves_nothing():
    d = FakeDriver()
    below_table = [0.0, 0.9, -0.2, 0.0, 0.0]
    assert ik.fk_tip(below_table)[2] < BOX["z"][0]
    result = run(traj(path([below_table])), d)
    assert result == {"ok": False, "reason": "error", "events": result["events"]}
    assert any(e.startswith("validate: ") and "workspace box" in e for e in result["events"])
    assert d.commands == [] and not d.stopped
    bad = traj(path([HOME]))
    del bad["dt"]
    result = run(bad, d)
    assert result["reason"] == "error" and any("schema" in e for e in result["events"])
    assert d.commands == []


def test_clean_path_streams_every_waypoint_in_order(poses):
    points = sweep(poses["pick"], poses["up"])
    assert replay.validate(traj(path(points)), LIMITS, ik.fk_tip, BOX) == []
    steps = [math.dist(ik.fk_tip(a), ik.fk_tip(b)) for a, b in zip(points[:-1], points[1:], strict=True)]
    assert max(steps) < replay.TIP_SPEED_MAX * replay.TICK  # no sub-stepping needed for this path
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    events = []
    result = run(traj(path(points)), d, on_event=events.append)
    assert result == {"ok": True, "reason": None, "events": events}
    assert_same_path([q for q, _ in d.commands], points)
    assert all(g == 1.0 for _, g in d.commands)
    assert events[-1] == "done" and not d.stopped
    assert not any("bridge" in e for e in events)


def test_bridging_from_ten_degrees_away_at_one_degree_per_tick(poses):
    start = list(poses["pick"])
    start[0] += 10 * DEG
    points = sweep(poses["pick"], poses["up"])
    d = FakeDriver(q0=start, gripper0=1.0)
    result = run(traj(path(points)), d)
    assert result["ok"], result["events"]
    assert "segment 0 bridge: 10 steps" in result["events"]
    commands = [q for q, _ in d.commands]
    bridge_cmds, rest = commands[:10], commands[10:]
    previous = start
    for q in bridge_cmds:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 1 * DEG + 1e-9
        previous = q
    assert bridge_cmds[-1] == pytest.approx(poses["pick"])
    assert_same_path(rest, points)
    # within 1 degree: no bridge, the path streams as is
    near = list(poses["pick"])
    near[0] += 0.9 * DEG
    d = FakeDriver(q0=near, gripper0=1.0)
    result = run(traj(path(points)), d)
    assert result["ok"] and not any("bridge" in e for e in result["events"])
    assert_same_path([q for q, _ in d.commands], points)


def test_stop_mid_path_stops_streaming_and_keeps_torque(poses):
    points = sweep(poses["pick"], poses["up"])
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    original = d.command
    sent = {"n": 0}

    def command(q, g):
        ok = original(q, g)
        sent["n"] += 1
        if sent["n"] == 5:
            d.freeze()  # STOP from another thread, right after the 5th command went out
        return ok

    d.command = command  # type: ignore[method-assign]
    result = run(traj(path(points)), d)
    assert result["reason"] == "stopped" and not result["ok"]
    assert len(d.commands) == 5
    assert d.stopped and d.torque
    assert d.read()[0] == pytest.approx(points[4])  # held where it was


def test_stop_from_an_on_event_hook_before_the_first_command(poses):
    points = sweep(poses["pick"], poses["up"])
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)

    def hook(event):
        if event.startswith("segment 0: path"):
            real_robot.freeze(d)

    result = run(traj(path(points)), d, on_event=hook)
    assert result["reason"] == "stopped"
    assert d.commands == [] and d.stopped and d.torque and real_robot.cancelled()
    real_robot.clear_cancel()


def test_stopped_driver_at_start_moves_nothing(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.freeze()
    result = run(traj(path(sweep(poses["pick"], poses["up"]))), d)
    assert result["reason"] == "stopped" and d.commands == []


def test_missed_grasp_rises_goes_home_and_returns_grasp_missed(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.0  # nothing between the jaws
    t = traj(
        path([poses["pick"]]), {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    )
    result = run(t, d)
    assert result["reason"] == "grasp_missed" and not result["ok"]
    assert any(e.startswith("grasp_missed: rising") for e in result["events"])
    assert any(e.startswith("rise: ") for e in result["events"])
    assert not d.stopped
    z_pick = ik.fk_tip(poses["pick"])[2]
    after_close = tips(d)[1:]
    assert max(z for _, _, z in after_close) >= z_pick + 0.045  # rose about 5 cm
    assert max(z for _, _, z in after_close) <= BOX["z"][1] + 1e-9
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)  # ended at home
    assert all(g == 0.0 for _, g in d.commands[1:])  # the hold value kept commanding during rise and home
    assert "home: done" in result["events"]


def test_grip_check_uses_held_min_through_the_joint_map(poses):
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.30  # well above empty-closed + held_min (8 on the 0-100 scale)
    result = run(traj(path([poses["pick"]]), close), d)
    assert result["ok"], result["events"]
    assert any("holding=True" in e for e in result["events"])
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.05  # closed further than held_min allows: nothing held
    result = run(traj(path([poses["pick"]]), close), d)
    assert result["reason"] == "grasp_missed"
    # without an override the fake reports expect_held, whatever it is
    for expect in (True, False):
        d = FakeDriver(q0=poses["pick"], gripper0=1.0)
        seg = dict(close, expect_held=expect)
        result = run(traj(path([poses["pick"]]), seg), d)
        assert result["ok"], result["events"]
    # an unexpected object (expect_held False, but holding) aborts the same way
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.30
    result = run(traj(path([poses["pick"]]), dict(close, expect_held=False)), d)
    assert result["reason"] == "grasp_missed"


def test_grip_open_commands_full_open_and_keeps_the_pose(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=0.2)
    result = run(traj({"kind": "grip", "action": "open", "object": "juice_box"}), d)
    assert result["ok"]
    assert d.commands == [(pytest.approx(poses["pick"]), 1.0)]


def test_hand_within_10_cm_freezes_with_hand_too_close(poses):
    points = sweep(poses["right"], poses["left"])
    scene = Scene()
    scene.hand(0.20, 0.10)  # on the table, under the end of the path
    hand = (0.20, 0.10, 0.03)
    assert math.dist(ik.fk_tip(points[0]), hand) > 0.15
    assert math.dist(ik.fk_tip(points[-1]), hand) < 0.10
    d = FakeDriver(q0=poses["right"], gripper0=1.0)
    result = run(traj(path(points)), d, scene=scene)
    assert result["reason"] == "hand_too_close" and not result["ok"]
    assert d.stopped and d.torque
    assert 0 < len(d.commands) < len(points)
    assert all(math.dist(t, hand) > 0.10 for t in tips(d))  # never commanded within 10 cm
    assert any("frozen" in e for e in result["events"])
    assert scene.calls > 0
    # the same path with the hand unseen streams to the end
    scene.hide()
    d = FakeDriver(q0=poses["right"], gripper0=1.0)
    assert run(traj(path(points)), d, scene=scene)["ok"]
    assert not d.stopped


def test_close_within_10_cm_of_the_hand_is_refused(poses):
    scene = Scene()
    scene.hand(0.24, 0.02)  # about 6 cm from the tip at the pick pose
    assert math.dist(ik.fk_tip(poses["pick"]), (0.24, 0.02, 0.03)) < 0.10
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)

    def never(*a, **k):
        raise AssertionError("close_gripper_hold must not be called near the hand")

    d.close_gripper_hold = never  # type: ignore[method-assign]
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    result = run(traj(close), d, scene=scene)
    assert result["reason"] == "hand_too_close" and not result["ok"]
    assert any(e.startswith("grip close refused") for e in result["events"])
    assert all(g == 1.0 for _, g in d.commands)  # the gripper never closed
    assert d.stopped  # the abort's rise ran the hand check and froze (the hand is still there)


def test_live_carry_to_hand_ends_at_home_with_handoff_done(poses):
    scene = Scene()
    scene.hand(0.30, 0.02)
    d = FakeDriver(q0=poses["pick"], gripper0=0.2)
    d.command(poses["pick"], 0.2)  # holding the object
    d.commands.clear()
    events = []

    def hook(event):
        events.append(event)
        if event == "handoff_done":
            scene.hide()  # the user took the object and withdrew the hand (it sits 9 cm from the pick pose)

    t = traj({"kind": "live", "primitive": "carry_to_hand", "object": "pill_bottle"}, path([poses["pick"]]))
    result = run(t, d, scene=scene, on_event=hook)
    assert result["ok"], result["events"]
    assert "handoff_done" in result["events"] and "handoff_done" in events
    assert result["events"] == events  # every event went through the hook exactly once
    grippers = [g for _, g in d.commands]
    release = grippers.index(1.0)
    assert release > 0 and all(g == 0.2 for g in grippers[:release])  # held until the release
    hand = (0.30, 0.02, 0.03)
    tip_at_release = tips(d)[release]
    assert math.dist(tip_at_release, hand) < 0.08
    assert "home: done" in result["events"]
    home_index = max(i for i, (q, _) in enumerate(d.commands) if q == pytest.approx(HOME, abs=1e-6))
    assert any(e.startswith("segment 1 bridge") for e in result["events"])  # the next path bridges from home
    assert d.commands[-1][0] == pytest.approx(poses["pick"], abs=1e-9)
    assert home_index < len(d.commands) - 1
    assert not d.stopped


def test_live_home_segment(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    result = run(traj({"kind": "live", "primitive": "home"}), d)
    assert result["ok"] and "home: done" in result["events"]
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    previous = list(poses["pick"])
    for q, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 1 * DEG + 1e-9
        previous = q


def test_driver_exception_freezes_and_reports_error(poses):
    points = sweep(poses["pick"], poses["up"])
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    original = d.command
    calls = {"n": 0}

    def flaky(q, g):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("serial glitch")
        return original(q, g)

    d.command = flaky  # type: ignore[method-assign]
    result = run(traj(path(points)), d)
    assert result["reason"] == "error" and not result["ok"]
    assert d.stopped and d.torque
    assert any("ConnectionError" in e and "serial glitch" in e for e in result["events"])
    assert len(d.commands) == 2


def test_read_failure_before_anything_moves_is_an_error_after_freeze(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)

    def boom():
        raise RuntimeError("not connected")

    d.read = boom  # type: ignore[method-assign]
    result = run(traj(path([poses["pick"]])), d)
    assert result["reason"] == "error" and d.stopped and d.commands == []
    assert any("not connected" in e for e in result["events"])


def test_scene_failure_freezes(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)

    def broken():
        raise OSError("camera unplugged")

    result = run(traj(path(sweep(poses["pick"], poses["up"]))), d, scene=broken)
    assert result["reason"] == "error" and d.stopped
    assert any("camera unplugged" in e for e in result["events"])


def test_simulated_clock_advances_without_sleeping(poses):
    points = sweep(poses["pick"], poses["up"])
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t["now"] += s

    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    segs = (
        path(points),
        {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True},
        {"kind": "grip", "action": "open", "object": "juice_box"},
    )
    started = time.perf_counter()
    result = replay.replay(traj(*segs), driver=d, realtime=False, clock=lambda: t["now"], sleep=sleep)
    assert result["ok"], result["events"]
    assert time.perf_counter() - started < 2.0
    expected = len(points) * replay.TICK + replay.GRIP_SETTLE_S + replay.OPEN_SETTLE_S
    assert t["now"] == pytest.approx(expected, abs=1e-6)
    assert len(sleeps) == len(points) + 2
    # realtime defaults to off for the fake: a 60 waypoint path returns well under its 2 s of ticks
    long_path = [list(poses["pick"])] * 60
    started = time.perf_counter()
    assert replay.replay(traj(path(long_path)), driver=FakeDriver(q0=poses["pick"]))["ok"]
    assert time.perf_counter() - started < 1.0


def test_default_driver_is_the_fake_singleton(monkeypatch, poses):
    monkeypatch.setenv("FAKE_ROBOT", "1")
    driver_mod.reset_driver()
    try:
        d = driver_mod.get_driver()
        assert isinstance(d, FakeDriver)
        result = replay.replay(traj(path(sweep(HOME, poses["up"]))), realtime=False)
        assert result["ok"], result["events"]
        assert len(d.commands) > 0
    finally:
        driver_mod.reset_driver()


def test_cli_refuses_hardware_without_real_or_fake(tmp_path):
    env = {k: os.environ.get(k, "") for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")} | {
        "PYTHONPATH": str(ROOT)
    }
    r = subprocess.run(
        [sys.executable, "-m", "arm.real_robot", "--carry-test"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 2 and "--real" in r.stderr


def test_policy_config_falls_back_to_ik_and_uses_a_runner_when_present(monkeypatch, tmp_path):
    rl_yaml = tmp_path / "rl.yaml"
    rl_yaml.write_text("use_policy: true\npolicy: rl/policies/handoff_best.zip\n", encoding="utf-8")
    monkeypatch.setattr(real_robot, "RL_PATH", rl_yaml)
    assert real_robot.policy_config() == {"use_policy": True, "policy": "rl/policies/handoff_best.zip"}
    scene = Scene()
    scene.hand(0.30, 0.02)
    # rl.policy_runner unavailable: a warning and IK tracking
    monkeypatch.setitem(sys.modules, "rl.policy_runner", None)
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=scene, realtime=False)
    assert result["ok"], result["events"]
    assert "policy unavailable: tracking with IK" in result["events"] and "handoff_done" in result["events"]
    # a runner is called with the driver, the scene callable, the policy path and the timeout
    import types

    calls = []

    def run_policy(driver, scene_fn, policy_path, *, timeout=8.0):
        calls.append((driver, scene_fn, policy_path, timeout))
        return True

    monkeypatch.setitem(sys.modules, "rl.policy_runner", types.SimpleNamespace(run_policy=run_policy))
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=scene, timeout=7.0, realtime=False)
    assert result["ok"], result["events"]
    assert len(calls) == 1
    driver, guarded_scene, policy_path, timeout = calls[0]
    assert driver is d and guarded_scene() == scene()
    assert policy_path == str(ROOT / "rl" / "policies" / "handoff_best.zip") and timeout == 7.0
    assert "policy reached the hand" in result["events"] and "handoff_done" in result["events"]
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    rl_yaml.write_text("use_policy: false\n", encoding="utf-8")
    assert real_robot.policy_config()["use_policy"] is False
    monkeypatch.setattr(real_robot, "RL_PATH", tmp_path / "missing.yaml")
    assert real_robot.policy_config()["use_policy"] is False


def test_config_failure_is_an_error_without_a_freeze(monkeypatch, tmp_path, poses):
    # config/limits.yaml unreadable: nothing can move, so nothing is frozen and the event names the file
    monkeypatch.setattr(replay, "LIMITS_PATH", tmp_path / "missing_limits.yaml")
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    events = []
    result = run(traj(path([poses["pick"]])), d, on_event=events.append)
    assert result == {"ok": False, "reason": "error", "events": events}
    assert any("missing_limits.yaml" in e for e in events)
    assert d.commands == [] and not d.stopped
    assert real_robot.home(driver=d, realtime=False)["reason"] == "error" and not d.stopped
    assert real_robot.carry_to_hand("x", driver=d, realtime=False)["reason"] == "error" and not d.stopped
    assert d.commands == []


def test_holding_threshold_is_inclusive(poses):
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.08  # exactly empty-closed (0) + held_min (8 of 100): holding
    result = run(traj(path([poses["pick"]]), close), d)
    assert result["ok"], result["events"]
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.079
    assert run(traj(path([poses["pick"]]), close), d)["reason"] == "grasp_missed"


# -- adversarial review round 2 ------------------------------------------------------------------------------


def test_torque_off_at_entry_is_an_error_without_a_freeze(poses):
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.torque_off()
    result = run(traj(path(sweep(poses["pick"], poses["up"]))), d)
    assert result["reason"] == "error" and not result["ok"]
    assert any("torque is off" in e for e in result["events"])
    assert d.commands == [] and not d.stopped and not d.torque


def test_hold_value_not_the_reading_is_commanded_after_a_close(poses):
    # the hold is the contact reading minus 3 (0-100 scale): 0.27 for a reading of 0.30; the rise, a live
    # home and the next path must keep commanding 0.27, never the present reading
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    d = FakeDriver(q0=poses["pick"], gripper0=1.0)
    d.gripper_reading_override = 0.30
    t = traj(path([poses["pick"]]), close, {"kind": "live", "primitive": "home"}, path([poses["pick"]]))
    result = run(t, d)
    assert result["ok"], result["events"]
    grippers = [g for _, g in d.commands]
    first_close = grippers.index(0.0)
    assert grippers[first_close + 1] == pytest.approx(0.27)
    assert all(g == pytest.approx(0.27) for g in grippers[first_close + 1 :]), set(grippers)
    assert len(grippers) > first_close + 10  # the home sweep and the bridge back ran with the hold value
    # a fresh replay on a driver whose last command was the hold: the hold value again, not the reading
    d.commands.clear()
    result = run(traj(path([poses["up"]])), d)
    assert result["ok"], result["events"]
    assert all(g == pytest.approx(0.27) for _, g in d.commands)


def test_scene_source_returning_no_dict_is_an_error_not_a_silent_pass(poses):
    points = sweep(poses["right"], poses["left"])
    for broken in (lambda: [1, 2, 3], lambda: "scene", lambda: None):
        d = FakeDriver(q0=poses["right"], gripper0=1.0)
        result = run(traj(path(points)), d, scene=broken)
        assert result["reason"] == "error" and not result["ok"]
        assert d.stopped and d.commands == []
        assert any("scene source returned" in e for e in result["events"]), result["events"]


def test_bridge_tolerance_is_one_degree(poses):
    points = sweep(poses["pick"], poses["up"])
    just_over = list(poses["pick"])
    just_over[0] += 1 * DEG + 1e-6
    d = FakeDriver(q0=just_over, gripper0=1.0)
    result = run(traj(path(points)), d)
    assert result["ok"] and "segment 0 bridge: 2 steps" in result["events"]  # 2 steps of just over 0.5 deg
    just_under = list(poses["pick"])
    just_under[0] += 1 * DEG - 1e-6
    d = FakeDriver(q0=just_under, gripper0=1.0)
    result = run(traj(path(points)), d)
    assert result["ok"] and not any("bridge" in e for e in result["events"])
