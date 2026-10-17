"""A13 Done-when (docs/safety.md): out-of-box and over-ceiling targets refused, tip speed caps (0.2 m/s,
0.1 m/s within 0.15 m of a seen hand) measured from driver.commands with fk_tip, no gripper close near the
hand, Esc -> freeze within 0.2 s, freeze keeps torque on, resume clears stopped. No hardware."""

from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import pytest

from arm import ik, real_robot, replay
from arm.driver import EscWatcher, FakeDriver

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
HOME = [float(v) for v in ik.home_q()]
LIMITS = replay.load_limits()
BOX = replay.load_box()
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
FAST_CAP = replay.TIP_SPEED_MAX * replay.TICK
SLOW_CAP = replay.TIP_SPEED_NEAR_HAND * replay.TICK
TOL = 1 + 1e-3  # the linearization of a 2 degree joint step costs well under 0.1 %

BELOW_TABLE = [0.0, 0.9, -0.2, 0.0, 0.0]  # tip z about -0.03
OVER_CEILING = [0.0, -0.6, -0.5, 0.0, 0.0]  # tip z about 0.49


def traj(*segments):
    return {
        "version": 1,
        "skill": "safety",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": list(segments),
    }


def path(points):
    return {"kind": "path", "q": [[float(v) for v in q] for q in points]}


def solve(xyz, pitch=-math.pi / 2, q0=None):
    q, ok = ik.solve(xyz, pitch, q0=q0)
    assert ok, xyz
    return q


def scene_with_hand(x, y, z=0.03, seen=True):
    scene = json.loads(json.dumps(FAKE_SCENE))
    scene["hand"].update({"x": x, "y": y, "z": z, "seen": seen})
    return lambda: scene


def tip_steps(driver, start_q):
    """(tip before, tip after, step length) for every command the driver received."""
    out = []
    previous = ik.fk_tip(start_q)
    for q, _ in driver.commands:
        tip = ik.fk_tip(q)
        out.append((previous, tip, math.dist(previous, tip)))
        previous = tip
    return out


def assert_speed_caps(driver, start_q, hand=None):
    steps = tip_steps(driver, start_q)
    assert steps, "nothing was commanded"
    near = 0
    for before, after, step in steps:
        assert step <= FAST_CAP * TOL, (before, after, step)
        if hand is not None and (
            math.dist(before, hand) < replay.HAND_SLOW_M or math.dist(after, hand) < replay.HAND_SLOW_M
        ):
            assert step <= SLOW_CAP * TOL, (before, after, step, math.dist(after, hand))
            near += 1
    return near


def test_fixtures_are_outside_the_box():
    assert ik.fk_tip(BELOW_TABLE)[2] < BOX["z"][0]
    assert ik.fk_tip(OVER_CEILING)[2] > BOX["z"][1]


def test_validate_refuses_out_of_box_and_over_ceiling_targets():
    for q in (BELOW_TABLE, OVER_CEILING):
        problems = replay.validate(traj(path([q])), LIMITS, ik.fk_tip, BOX)
        assert any("leaves the workspace box" in p for p in problems), problems
    far = solve((0.40, 0.0, 0.05))  # at the far edge: inside
    assert replay.validate(traj(path([far])), LIMITS, ik.fk_tip, BOX) == []
    outside = {"x": [0.10, 0.39], "y": BOX["y"], "z": BOX["z"]}  # the same pose with a tighter box
    assert replay.validate(traj(path([far])), LIMITS, ik.fk_tip, outside)
    # and replay never moves toward them
    for q in (BELOW_TABLE, OVER_CEILING):
        d = FakeDriver()
        result = replay.replay(traj(path([q])), driver=d, realtime=False)
        assert result["reason"] == "error" and d.commands == [] and not d.stopped


def test_sim_robot_refuses_out_of_box_targets():
    sim_robot = pytest.importorskip("arm.sim_robot", reason="A8 SimRobot not landed")
    try:
        robot = sim_robot.SimRobot(json.loads(json.dumps(FAKE_SCENE)))
    except Exception as exc:  # another lane is mid-edit: not this suite's business
        pytest.skip(f"SimRobot not constructible yet: {exc!r}")
    assert robot.move_to(0.45, 0.0, 0.10) is False
    assert robot.move_to(0.20, 0.0, 0.35) is False
    assert robot.move_to(0.20, 0.30, 0.10) is False


def test_bridge_from_outside_the_box_is_refused():
    d = FakeDriver(
        q0=BELOW_TABLE
    )  # the arm was pushed below the table (torque off) and replay is asked to move
    result = replay.replay(traj(path([solve((0.24, -0.04, 0.0525))])), driver=d, realtime=False)
    assert result["reason"] == "error" and d.commands == []
    assert any("bridge refused" in e for e in result["events"])


def test_tip_speed_never_exceeds_0_2_m_s():
    start = solve((0.30, -0.15, 0.10))
    end = solve((0.30, 0.15, 0.10), q0=start)
    points = [start, *replay.bridge(start, end, 2 * DEG)]  # 2 degree steps swing the tip over 1 cm per tick
    assert replay.validate(traj(path(points)), LIMITS, ik.fk_tip, BOX) == []
    raw = [math.dist(ik.fk_tip(a), ik.fk_tip(b)) for a, b in zip(points[:-1], points[1:], strict=True)]
    assert max(raw) > FAST_CAP  # the trajectory itself would exceed the cap: replay must split it
    d = FakeDriver(q0=start)
    result = replay.replay(traj(path(points)), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert len(d.commands) > len(points)
    assert_speed_caps(d, start)
    assert d.commands[-1][0] == pytest.approx(end)


def test_tip_speed_capped_at_0_1_m_s_within_15_cm_of_a_seen_hand():
    start = solve((0.30, -0.15, 0.10))
    end = solve((0.30, 0.15, 0.10), q0=start)
    points = [start, *replay.bridge(start, end, 2 * DEG)]
    hand = (0.30, 0.27, 0.03)  # 14 cm from the end of the path: slow zone, no freeze
    assert 0.10 < math.dist(ik.fk_tip(end), hand) < 0.15
    d = FakeDriver(q0=start)
    result = replay.replay(traj(path(points)), driver=d, realtime=False, scene=scene_with_hand(*hand))
    assert result["ok"], result["events"]
    assert not d.stopped
    near = assert_speed_caps(d, start, hand)
    assert near >= 3


def test_carry_to_hand_and_home_honor_the_caps_near_the_hand():
    hand = (0.30, 0.02, 0.03)
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=scene_with_hand(*hand), realtime=False)
    assert result["ok"], result["events"]
    near = assert_speed_caps(d, HOME, hand)
    assert near > 10  # the approach, the back-off and the start of the home sweep are all near the hand
    previous = HOME
    for q, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 2 * DEG + 1e-9
        previous = q
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    assert "handoff_done" in result["events"]


def test_close_near_the_hand_is_refused():
    pick = solve((0.24, -0.04, 0.0525))
    hand = (0.24, 0.02, 0.03)
    assert math.dist(ik.fk_tip(pick), hand) < replay.HAND_FREEZE_M
    d = FakeDriver(q0=pick, gripper0=1.0)
    calls = []
    d.close_gripper_hold = lambda *a, **k: calls.append(1) or 0.0  # type: ignore[method-assign]
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    result = replay.replay(traj(close), driver=d, realtime=False, scene=scene_with_hand(*hand))
    assert result["reason"] == "hand_too_close" and calls == []
    assert all(g == 1.0 for _, g in d.commands)
    # 12 cm away the close goes ahead
    d = FakeDriver(q0=pick, gripper0=1.0)
    d.gripper_reading_override = 0.3
    result = replay.replay(traj(close), driver=d, realtime=False, scene=scene_with_hand(0.24, 0.08, 0.03))
    assert result["ok"], result["events"]
    assert d.commands[0][1] == 0.0  # the close


def test_esc_freezes_within_200_ms():
    d = FakeDriver()
    pressed: dict[str, float] = {}
    frozen: dict[str, float] = {}

    def keys():
        if "t" not in pressed:
            pressed["t"] = time.perf_counter()
            return EscWatcher.ESC
        return None

    def on_esc():
        real_robot.freeze(d)
        frozen["t"] = time.perf_counter()

    watcher = EscWatcher(on_esc, poll=0.005, key_source=keys)
    watcher.start()
    deadline = time.perf_counter() + 2.0
    while "t" not in frozen and time.perf_counter() < deadline:
        time.sleep(0.002)
    watcher.stop()
    assert watcher.pressed and "t" in frozen
    assert frozen["t"] - pressed["t"] < 0.2
    assert d.stopped and d.torque and real_robot.cancelled()
    real_robot.clear_cancel()


def test_freeze_keeps_torque_and_resume_clears_stopped():
    d = FakeDriver()
    d.command([v + 0.01 for v in HOME], 0.5)
    real_robot.freeze(d)
    assert d.stopped and d.torque and real_robot.cancelled()
    held = d.read()
    assert d.command(HOME, 1.0) is False
    assert d.read() == held  # a held object stays held: gripper and pose unchanged
    d.resume()
    assert not d.stopped and d.torque
    assert d.command(HOME, 1.0) is True
    # a primitive started after resume clears the cancel flag and runs
    result = real_robot.home(driver=d, realtime=False)
    assert result["ok"] and not real_robot.cancelled()
    assert d.read()[0] == pytest.approx(HOME, abs=1e-6)


def test_cancel_flag_stops_a_running_primitive():
    d = FakeDriver(gripper0=0.2)
    hits = []

    def hook(event):
        hits.append(event)
        if event.startswith("carry_to_hand"):
            real_robot.freeze(d)

    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=scene_with_hand(0.30, 0.02), on_event=hook, realtime=False
    )
    assert result["reason"] == "stopped" and "handoff_done" not in result["events"]
    assert d.stopped and d.torque and d.commands == []
    real_robot.clear_cancel()


def test_the_tip_never_exceeds_the_ceiling_during_a_grasp_abort():
    high = solve((0.20, 0.0, 0.29), math.radians(-15))
    d = FakeDriver(q0=high, gripper0=1.0)
    d.gripper_reading_override = 0.0
    close = {"kind": "grip", "action": "close", "object": "tea_tin", "expect_held": True}
    result = replay.replay(traj(path([high]), close), driver=d, realtime=False)
    assert result["reason"] == "grasp_missed", result["events"]
    assert all(ik.fk_tip(q)[2] <= BOX["z"][1] + 1e-9 for q, _ in d.commands)
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)


# -- workspace box and ceiling during the live primitives (adversarial review round 1) -----------------------


def assert_tips_inside(driver, box=BOX, tol=1e-9):
    """Every commanded tip inside the box (exact: the planner keeps its own lines 1 mm inside)."""
    outside = [t for t in (ik.fk_tip(q) for q, _ in driver.commands) if not replay.inside_box(t, box, tol)]
    assert not outside, outside[:3]


def carry(hand, start_q=None, **kw):
    d = FakeDriver(q0=start_q, gripper0=0.2)
    if start_q is not None:
        d.command(start_q, 0.2)
        d.commands.clear()
    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=scene_with_hand(*hand), realtime=False, **kw
    )
    return result, d


@pytest.mark.parametrize("x", [0.15, 0.25, 0.35, 0.40])
@pytest.mark.parametrize("y", [-0.20, 0.0, 0.20])
def test_carry_to_hand_keeps_the_tip_in_the_box_for_hands_across_the_box(x, y):
    # from home (the pose after a `home` live segment or /resume): the joint-space sweep toward an extended
    # pose arcs the tip over the 0.30 m ceiling, which the straight-line tracking must never do
    result, d = carry((x, y, 0.03))
    assert result["ok"], result["events"]
    assert_tips_inside(d)
    assert "handoff_done" in result["events"]
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    assert_speed_caps(d, HOME, (x, y, 0.03))


def test_carry_to_hand_from_a_raised_pick_pose_stays_under_the_ceiling():
    pick_up = solve((0.16, -0.02, 0.17))  # the tea tin just lifted by 8 cm
    for hand in ((0.40, -0.20, 0.03), (0.40, 0.0, 0.03), (0.35, 0.15, 0.03)):
        result, d = carry(hand, start_q=pick_up)
        assert result["ok"], result["events"]
        assert_tips_inside(d)
        assert "handoff_done" in result["events"]


def test_hand_outside_the_box_falls_back_to_the_zone_instead_of_the_edge():
    # Even a hand near enough for edge service must use the configured zone when it leaves the box.
    result, d = carry((0.42, 0.0, 0.03))
    assert result["ok"], result["events"]
    assert_tips_inside(d)
    assert not any(e.startswith("hand steady within") for e in result["events"])
    assert any("outside the workspace box" in e for e in result["events"])
    grippers = [g for _, g in d.commands]
    release = grippers.index(1.0)
    zone = real_robot.handoff_zone(FAKE_SCENE)
    assert math.dist(ik.fk_grasp(d.commands[release][0])[0], zone) <= real_robot.ZONE_REACHED_M
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    # far outside to the side, or raised off the table: never followed out of the box, never above 0.30 m
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    for hand in ((0.30, 0.30, 0.03), (0.30, -0.32, 0.03), (0.30, 0.0, 0.26), (0.36, 0.0, 0.20)):
        result, d = carry(hand, clock=lambda: clock["t"], sleep=sleep)
        assert result["ok"], (hand, result["events"])
        assert_tips_inside(d)
        assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
        assert "handoff_done" in result["events"]


HIGH_AND_FAR = [
    ((0.30, 0.22, 0.28), 0.0),  # SimRobot's transit height above the tray
    ((0.33, 0.20, 0.28), 0.0),
    ((0.20, -0.22, 0.28), -45.0),
    ((0.28, 0.10, 0.28), -90.0),
    ((0.36, 0.0, 0.28), -90.0),
    ((0.36, -0.22, 0.03), 0.0),
    ((0.36, 0.22, 0.10), 0.0),
    ((0.36, -0.22, 0.20), -45.0),
    ((0.12, 0.0, 0.03), -90.0),
    ((0.20, 0.0, 0.29), -15.0),
]


@pytest.mark.parametrize("xyz,pitch_deg", HIGH_AND_FAR)
def test_home_from_high_and_far_poses_keeps_the_tip_in_the_box(xyz, pitch_deg):
    # the direct sweeps from these poses arc 3-6 mm over the ceiling: home() must plan around it
    q = solve(xyz, math.radians(pitch_deg))
    d = FakeDriver(q0=q)
    result = real_robot.home(driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert_tips_inside(d)
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    previous = q
    for cmd, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, cmd, strict=True)) <= 1 * DEG + 1e-9
        previous = cmd
    assert_speed_caps(d, q)


def test_home_from_every_reachable_pose_of_a_coarse_grid():
    refused = []
    for x in (0.12, 0.24, 0.36):
        for y in (-0.22, 0.0, 0.22):
            for z in (0.03, 0.15, 0.28):
                for pitch in (-90, 0):
                    q, ok = ik.solve((x, y, z), math.radians(pitch))
                    if not ok:
                        continue
                    d = FakeDriver(q0=q)
                    result = real_robot.home(driver=d, realtime=False)
                    if not result["ok"]:
                        refused.append(((x, y, z, pitch), result["events"][-1]))
                        continue
                    assert_tips_inside(d)
                    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    assert refused == []


def band_excess(q):
    """How far (rad) the joint furthest outside the limits-minus-3-degrees band is from it; 0 inside."""
    worst = 0.0
    for j, v in zip(replay.JOINTS, q, strict=True):
        low, high = LIMITS[j][0] + replay.MARGIN_RAD, LIMITS[j][1] - replay.MARGIN_RAD
        worst = max(worst, low - v, v - high)
    return worst


def in_band(q):
    return band_excess(q) <= 1e-9


RESTING_ON_A_STOP = [
    [0.0, -1.70, 1.60, 1.20, 0.0],  # folded, shoulder_lift 2.6 deg from its stop
    [0.0, LIMITS["shoulder_lift"][0] + 2 * DEG, 1.4, 1.2, 0.0],
    [0.0, -1.3, LIMITS["elbow_flex"][1] - 1 * DEG, 1.0, 0.0],
    [LIMITS["shoulder_pan"][1], -1.35, 0.02, 1.45, 0.0],  # pan exactly at the limit
]


@pytest.mark.parametrize("q", RESTING_ON_A_STOP)
def test_home_from_a_joint_resting_inside_the_margin(q):
    # the arm was pushed against a joint stop (torque off): the plan starts from the pose clamped into the
    # band and the driver walks the joint back at 2 degrees per tick; every commanded waypoint is in band
    assert not in_band(q)
    d = FakeDriver(q0=q)
    result = real_robot.home(driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert d.commands
    # the driver clips each send to 2 degrees from the present pose, so the first sends can still sit
    # inside the margin: the excess only ever shrinks, and once in band the arm stays in band
    excess = band_excess(q)
    for cmd, _ in d.commands:
        e = band_excess(cmd)
        assert e <= excess + 1e-9 and (excess > 0 or e == 0), (cmd, e, excess)
        excess = e
    assert in_band(d.commands[-1][0])
    first = d.commands[0][0]
    assert max(abs(a - b) for a, b in zip(q, first, strict=True)) <= 2 * DEG + 1e-9
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)


def test_replay_bridges_from_a_joint_resting_inside_the_margin():
    q = RESTING_ON_A_STOP[1]  # shoulder_lift 2 degrees from its stop, tip inside the box
    assert replay.inside_box(ik.fk_tip(q), BOX)
    d = FakeDriver(q0=q, gripper0=1.0)
    pick = solve((0.24, -0.04, 0.0525))
    result = replay.replay(traj(path([pick])), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert any(e.startswith("segment 0 bridge") for e in result["events"])
    assert band_excess(d.commands[0][0]) < band_excess(q) and in_band(d.commands[2][0])
    assert d.commands[-1][0] == pytest.approx(pick, abs=1e-9)
    # from a tip outside the box the bridge is still refused: replay never plans from outside
    d = FakeDriver(q0=RESTING_ON_A_STOP[0], gripper0=1.0)
    assert not replay.inside_box(ik.fk_tip(RESTING_ON_A_STOP[0]), BOX)
    result = replay.replay(traj(path([pick])), driver=d, realtime=False)
    assert result["reason"] == "error" and d.commands == []
    assert any("bridge refused" in e for e in result["events"])


def test_replay_bridge_takes_a_straight_line_when_the_sweep_would_arc_over_the_ceiling():
    transit = solve((0.30, 0.22, 0.28), 0.0)
    sweep = replay.bridge(HOME, transit, replay.BRIDGE_STEP_RAD)
    assert any("workspace box" in p for p in replay.validate_path(sweep, LIMITS, ik.fk_tip, BOX))
    d = FakeDriver(q0=HOME, gripper0=1.0)
    result = replay.replay(traj(path([transit])), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert any(e.startswith("segment 0 bridge") and "straight line" in e for e in result["events"])
    assert_tips_inside(d)
    previous = HOME
    for cmd, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, cmd, strict=True)) <= 1 * DEG + 1e-9
        previous = cmd
    assert d.commands[-1][0] == pytest.approx(transit, abs=1e-9)


def test_box_gate_never_commands_a_tick_outside_the_box():
    # whatever planned it: a tick whose tip is more than 1 mm outside the box freezes the arm instead
    near_ceiling = solve((0.20, 0.0, 0.29), math.radians(-15))
    d = FakeDriver(q0=near_ceiling)
    session = replay.Session(d, None, replay.Pacer(False))
    session.start()
    with pytest.raises(replay.Abort) as info:
        for q in replay.bridge(near_ceiling, OVER_CEILING, replay.MAX_STEP_RAD):
            session.command(q, where="test")
    assert info.value.reason == "error" and d.stopped and d.torque
    assert d.commands, "the first ticks toward the ceiling are inside and go out"
    assert all(ik.fk_tip(q)[2] <= 0.30 + 0.001 for q, _ in d.commands)  # the gate is 1 mm, literally
    assert ik.fk_tip(d.commands[-1][0])[2] > 0.29
    d = FakeDriver(q0=near_ceiling)
    session = replay.Session(d, None, replay.Pacer(False))
    session.start()
    with pytest.raises(replay.Abort):
        session.command(OVER_CEILING, where="test")
    assert d.commands == [] and d.stopped
    # the gate's tolerance pinned: half a millimeter beyond a face passes, 1.5 mm freezes
    d = FakeDriver(q0=near_ceiling)
    session = replay.Session(d, None, replay.Pacer(False))
    session.start()
    session.check_box((0.20, 0.0, 0.3005), "test")
    session.check_box((0.0995, 0.2505, 0.0195), "test")
    assert not d.stopped
    with pytest.raises(replay.Abort) as info:
        session.check_box((0.20, 0.0, 0.3015), "test")
    assert info.value.reason == "error" and d.stopped
    d = FakeDriver(q0=near_ceiling)
    session = replay.Session(d, None, replay.Pacer(False))
    session.start()
    with pytest.raises(replay.Abort):
        session.check_box((0.20, 0.0, 0.0185), "test")
    assert d.stopped


def test_cancel_flag_alone_stops_a_primitive_through_the_session():
    # the cancel flag without the driver's freeze (the Session's own STOP check, not the driver gate)
    d = FakeDriver(gripper0=0.2)

    def hook(event):
        if event.startswith("carry_to_hand"):
            real_robot._CANCEL.set()

    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=scene_with_hand(0.30, 0.02), on_event=hook, realtime=False
    )
    assert result["reason"] == "stopped" and d.commands == []
    assert d.stopped and d.torque  # the session froze the arm itself
    real_robot.clear_cancel()


# -- the executed carry motion is validated (adversarial review round 2) -----------------------------------

PILL_PICK = solve((0.18, -0.12, 0.035))
JUICE_PICK = solve((0.24, -0.04, 0.0525))
CARRY_STARTS = {
    "pill lifted 8 cm (the demo's hand_over start)": solve((0.18, -0.12, 0.115), q0=PILL_PICK),
    "juice at the table": JUICE_PICK,
    "transit over the tray at 0.28": solve((0.30, 0.22, 0.28), 0.0),
    "home": list(HOME),
}
CARRY_HANDS = [
    (0.28, -0.08, 0.03),  # the demo's hand rest
    (0.38, 0.22, 0.03),  # the corners of the taped box
    (0.38, -0.22, 0.03),
    (0.43, 0.27, 0.03),  # past a far corner
    (0.60, 0.40, 0.03),  # way off
    (0.12, 0.0, 0.03),  # next to the base
]


def grasp_heights(driver):
    """The grasp center's z for every command up to the release (the gripper opening to 1.0)."""
    grippers = [g for _, g in driver.commands]
    release = grippers.index(1.0) if 1.0 in grippers else len(grippers)
    return [ik.fk_grasp(q)[0][2] for q, _ in driver.commands[:release]]


@pytest.mark.parametrize("hand", CARRY_HANDS)
@pytest.mark.parametrize("start", list(CARRY_STARTS), ids=lambda s: s.split(" (")[0])
def test_carry_to_hand_never_dives_and_stays_validated_off_the_rehearsed_path(start, hand):
    # round 2 found the tracking driving a held object into the table and the tip out of the box when the
    # arm started anywhere but home: every executed tick must now come from a validated bridge, so the
    # grasp center (hence the held object) never goes below the start or the goal (hand + 5 cm)
    q0 = CARRY_STARTS[start]
    result, d = carry(hand, start_q=q0)
    assert result["ok"], (start, hand, result["events"])
    assert "handoff_done" in result["events"]
    assert_tips_inside(d)
    assert d.commands[-1][0] == pytest.approx(HOME, abs=1e-6)
    previous = q0
    for q, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 2 * DEG + 1e-9
        previous = q
    assert_speed_caps(d, q0, hand)
    heights = grasp_heights(d)
    assert heights, "nothing was commanded before the release"
    floor = min(ik.fk_grasp(q0)[0][2], hand[2] + real_robot.HAND_ABOVE_M) - real_robot.DIVE_TOL_M
    assert min(heights) >= floor, (start, hand, min(heights), floor)
    assert not any(e.startswith("error") for e in result["events"])


def test_carry_to_hand_reaches_the_corner_hand_without_sweep_fallbacks():
    start = CARRY_STARTS["pill lifted 8 cm (the demo's hand_over start)"]
    result, d = carry((0.38, 0.22, 0.03), start_q=start)
    assert result["ok"], result["events"]
    assert any(e.startswith("hand steady within") for e in result["events"])
    assert not any("joint-space sweep" in e or "holding" in e for e in result["events"]), result["events"]
    grippers = [g for _, g in d.commands]
    release = grippers.index(1.0)
    grasp = ik.fk_grasp(d.commands[release][0])[0]
    assert math.dist(grasp, (0.38, 0.22, 0.03)) < real_robot.RELEASE_WITHIN_M
    assert grasp[2] == pytest.approx(0.08, abs=0.01)


def servo_session(start):
    d = FakeDriver(q0=start, gripper0=0.2)
    session = replay.Session(d, None, replay.Pacer(False), hand_freeze=False)
    session.start()
    return session


def assert_on_the_line_toward(session, path, goal):
    """The path is the validated bridge from the present pose and ends on the line toward `goal`."""
    assert path == replay.bridge(session.q, path[-1], replay.MAX_STEP_RAD)
    assert replay.validate_path(path, LIMITS, ik.fk_tip, replay.shrunk_box(BOX, replay.PLAN_INSET_M)) == []
    grasp0 = ik.fk_grasp(session.q)[0]
    grasp1 = ik.fk_grasp(path[-1])[0]
    assert math.dist(grasp0, grasp1) <= real_robot.TICKS_PER_CYCLE * FAST_CAP * TOL + 0.005
    direction = [b - a for a, b in zip(grasp0, goal, strict=True)]
    moved = [b - a for a, b in zip(grasp0, grasp1, strict=True)]
    along = sum(m * u for m, u in zip(moved, direction, strict=True)) / math.dist(grasp0, goal)
    off_line = math.sqrt(max(0.0, sum(m * m for m in moved) - along * along))
    assert along > 0 and off_line < 0.006  # within IK's 5 mm of the straight line


def test_carry_servo_path_is_a_validated_bridge_from_the_present_pose(monkeypatch):
    goal = (0.28, -0.08, 0.08)
    # well inside the workspace: one cycle's ticks, no note
    session = servo_session(CARRY_STARTS["pill lifted 8 cm (the demo's hand_over start)"])
    path, note = real_robot._servo_path(session, goal, real_robot.START_PITCH)
    assert path is not None and note == ""
    assert len(path) <= real_robot.TICKS_PER_CYCLE
    assert_on_the_line_toward(session, path, goal)
    # the nearly extended transit pose: a 2 cm tip step costs more joint motion than three ticks can
    # execute, so the (longer) validated bridge runs to its end before the next re-aim, and says so
    session = servo_session(CARRY_STARTS["transit over the tray at 0.28"])
    path, note = real_robot._servo_path(session, goal, real_robot.START_PITCH)
    assert path is not None and note.startswith("re-aiming with a")
    assert real_robot.TICKS_PER_CYCLE < len(path) <= math.ceil(replay.LINE_JUMP_RAD / replay.MAX_STEP_RAD)
    assert_on_the_line_toward(session, path, goal)
    # When IK finds nothing, the clearance rise fails before any travel or release.
    monkeypatch.setattr(ik, "solve_grasp", lambda xyz, pitch=-math.pi / 2, q0=None: (list(q0), False))
    path, note = real_robot._servo_path(session, goal, real_robot.START_PITCH)
    assert path is None and "holding" in note
    start = CARRY_STARTS["pill lifted 8 cm (the demo's hand_over start)"]
    result, d = carry((0.30, 0.02, 0.03), start_q=start, timeout=0.5)
    assert result["reason"] == "error", result["events"]
    assert any("rise" in e and "refused" in e for e in result["events"])
    assert d.commands == [] and d.stopped and d.torque
    assert "handoff_done" not in result["events"] and d.read()[1] == 0.2


# -- boundaries pinned (mutations that survived round 2) ----------------------------------------------------


def test_close_refusal_boundary_is_exactly_10_cm():
    pick = solve((0.24, -0.04, 0.0525))
    tip = ik.fk_tip(pick)
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    for dist, refused in ((0.0999, True), (0.1001, False)):
        d = FakeDriver(q0=pick, gripper0=1.0)
        d.gripper_reading_override = 0.3
        calls = []
        original = d.close_gripper_hold

        def hold(*a, _orig=original, _calls=calls, **k):
            _calls.append(1)
            return _orig(*a, **k)

        d.close_gripper_hold = hold  # type: ignore[method-assign]
        scene = scene_with_hand(tip[0] + dist, tip[1], tip[2])
        result = replay.replay(traj(close), driver=d, realtime=False, scene=scene)
        if refused:
            assert result["reason"] == "hand_too_close" and calls == [], (dist, result["events"])
            assert any(e.startswith("grip close refused") for e in result["events"])
        else:
            assert result["ok"] and calls == [1], (dist, result["events"])


def test_hand_arriving_during_the_open_settle_freezes():
    pick = solve((0.24, -0.04, 0.0525))
    tip = ik.fk_tip(pick)
    d = FakeDriver(q0=pick, gripper0=0.2)
    scene = json.loads(json.dumps(FAKE_SCENE))
    scene["hand"].update({"x": tip[0] + 0.05, "y": tip[1], "z": tip[2], "seen": False})

    def source():
        scene["hand"]["seen"] = bool(d.commands)  # the hand arrives right after the open was commanded
        return scene

    result = replay.replay(traj({"kind": "grip", "action": "open"}), driver=d, realtime=False, scene=source)
    assert result["reason"] == "hand_too_close" and not result["ok"]
    assert d.stopped and d.torque
    assert d.commands == [(pytest.approx(pick), 1.0)]  # the open went out, then the settle check froze
    assert any("grip open" in e and "frozen" in e for e in result["events"])


def test_replay_and_primitives_refuse_with_torque_off_without_freezing():
    pick = solve((0.24, -0.04, 0.0525))
    hand = scene_with_hand(0.3, 0.0)
    for run in (
        lambda d: replay.replay(traj(path([pick])), driver=d, realtime=False),
        lambda d: real_robot.home(driver=d, realtime=False),
        lambda d: real_robot.carry_to_hand("pill_bottle", driver=d, scene=hand, realtime=False),
    ):
        d = FakeDriver(q0=pick, gripper0=0.2)
        d.torque_off()
        result = run(d)
        assert result["reason"] == "error" and not result["ok"]
        assert any("torque is off" in e for e in result["events"]), result["events"]
        assert d.commands == [] and not d.stopped and not d.torque


def test_home_rejects_a_bad_step_without_freezing():
    pick = solve((0.24, -0.04, 0.0525))
    for step in (0.0, -1.0, math.nan, math.inf, 5 * DEG, "fast"):
        d = FakeDriver(q0=pick)
        result = real_robot.home(driver=d, realtime=False, max_step_rad=step)
        assert result["reason"] == "error" and d.commands == [] and not d.stopped, (step, result["events"])
        assert any("max_step_rad" in e or "float" in e for e in result["events"]), (step, result["events"])
    d = FakeDriver(q0=pick)
    assert real_robot.home(driver=d, realtime=False, max_step_rad=2 * DEG)["ok"]
    previous = pick
    for q, _ in d.commands:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 2 * DEG + 1e-9
        previous = q


def test_carry_timeout_is_checked_capped_and_survives_a_stalled_clock():
    hand = (0.30, 0.02, 0.03)
    for bad in (math.nan, math.inf, -math.inf, "soon", None):
        d = FakeDriver(gripper0=0.2)
        result = real_robot.carry_to_hand(
            "pill_bottle", driver=d, scene=scene_with_hand(*hand), timeout=bad, realtime=False
        )
        assert result["reason"] == "error" and d.commands == [] and not d.stopped, (bad, result["events"])
    result, d = carry(hand, timeout=1e9)
    assert result["ok"], result["events"]
    assert any(e.startswith("timeout 1e+09 s capped at 60 s") for e in result["events"])
    # a clock that never advances: the deadlines can never come, so the tracking ends by itself
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle",
        driver=d,
        scene=scene_with_hand(0.9, 0.9, seen=False),
        clock=lambda: 0.0,
        sleep=lambda s: None,
        realtime=False,
    )
    assert result["reason"] == "error", result["events"]
    assert any("clock did not advance" in e for e in result["events"])
    assert "handoff_done" not in result["events"] and "released" not in result["events"]
    assert d.stopped and d.torque and all(g == 0.2 for _, g in d.commands)


def test_carry_with_a_broken_scene_source_fails_loud():
    for broken in (lambda: [1, 2, 3], lambda: "scene", lambda: None):
        d = FakeDriver(gripper0=0.2)
        result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=broken, realtime=False)
        assert result["reason"] == "error" and d.stopped, result["events"]
        assert any("scene source returned" in e for e in result["events"])
        assert "handoff_done" not in result["events"]
        real_robot.clear_cancel()


def test_esc_during_connect_is_caught_after_connect():
    class Watcher:
        pressed = False

    d = FakeDriver()
    real_robot.clear_cancel()
    assert real_robot.esc_during_connect(Watcher(), d) is False and not d.stopped
    watcher = Watcher()
    watcher.pressed = True
    assert real_robot.esc_during_connect(watcher, d) is True
    assert d.stopped and d.torque and real_robot.cancelled()
    d = FakeDriver()
    real_robot.clear_cancel()
    real_robot._CANCEL.set()  # the callback ran inside connect(), which then cleared the driver's flag
    assert real_robot.esc_during_connect(Watcher(), d) is True and d.stopped
    real_robot.clear_cancel()


def test_carry_tracking_has_an_absolute_time_bound(monkeypatch):
    # whatever the deadlines say (here pushed out of reach), the tracking ends after MAX_TRACK_S
    monkeypatch.setattr(real_robot, "FALLBACK_EXTRA_S", 1e9)
    monkeypatch.setattr(real_robot, "ZONE_REACHED_M", -1.0)
    monkeypatch.setattr(real_robot, "MAX_TRACK_S", 3.0)
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle",
        driver=d,
        scene=scene_with_hand(0.9, 0.9, seen=False),
        timeout=0.0,
        clock=lambda: clock["t"],
        sleep=sleep,
        realtime=False,
    )
    assert result["reason"] == "error", result["events"]
    assert any("tracking ran for 3 s" in e for e in result["events"])
    assert "at the handoff zone" not in result["events"]
    assert "handoff_done" not in result["events"] and "released" not in result["events"]
    assert d.stopped and d.torque and all(g == 0.2 for _, g in d.commands)


def test_no_hand_releases_at_the_handoff_zone_after_the_timeout():
    clock = {"t": 0.0}
    released_at = {}

    def sleep(s):
        clock["t"] += s

    def hook(event):
        if event == "released":
            released_at["t"] = clock["t"]

    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle",
        driver=d,
        scene=scene_with_hand(0.9, 0.9, seen=False),
        on_event=hook,
        clock=lambda: clock["t"],
        sleep=sleep,
        realtime=False,
    )
    assert result["ok"], result["events"]
    assert "no steady hand within 10 s: falling back to the handoff zone" in result["events"]
    assert "at the handoff zone" in result["events"]
    assert 10.0 <= released_at["t"] <= 10.0 + real_robot.FALLBACK_EXTRA_S, released_at
    grippers = [g for _, g in d.commands]
    grasp = ik.fk_grasp(d.commands[grippers.index(1.0)][0])[0]
    zone = real_robot.handoff_zone(FAKE_SCENE)
    assert math.dist(grasp, zone) <= real_robot.ZONE_REACHED_M + 0.005, (grasp, zone)


@pytest.mark.parametrize(
    "hand",
    [
        (0.09, 0.0, 0.03),
        (0.41, 0.0, 0.03),
        (0.3, 0.26, 0.03),
        (0.3, -0.26, 0.03),
        (0.3, 0.0, -0.04),
        (0.3, 0.0, 0.26),
    ],
)
def test_rejected_hand_target_latches_the_zone(hand):
    session = servo_session(HOME)
    target = real_robot._Target(session)
    assert target.goal(session, hand, fallback=False) == target.zone
    assert target.rejected_hand
    # Tracking jitter back into the box must not resume pursuit during the fallback.
    assert target.goal(session, (0.30, 0.0, 0.03), fallback=False) == target.zone
    assert len(session.events) == 1 and "falling back" in session.events[0]


def test_invalid_handoff_zone_aborts_before_motion():
    scene = scene_with_hand(0.42, 0.0)()
    scene["zones"]["handoff"]["x"] = 0.6
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=lambda: scene, realtime=False)
    assert result["reason"] == "error" and d.commands == []
    assert d.stopped and d.torque and d.read()[1] == 0.2
    assert any("handoff zone is outside" in e for e in result["events"])


@pytest.mark.parametrize("missing", ["scene", "object", "obstacles", "ceiling"])
def test_unproven_clearance_holds_the_object_without_tracking(missing):
    scene = scene_with_hand(0.30, 0.02)()
    obj = "pill_bottle"
    if missing == "object":
        obj = "unknown_held_object"
    elif missing == "obstacles":
        del scene["obstacles"]
    elif missing == "ceiling":
        scene["obstacles"]["new_obstacle"] = {"h": 0.29, "seen": True}
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        obj, driver=d, scene=None if missing == "scene" else lambda: scene, realtime=False
    )
    assert result["reason"] == "error", result["events"]
    assert any("clearance unavailable" in e for e in result["events"])
    assert d.commands == [] and d.stopped and d.torque and d.read()[1] == 0.2


def test_rise_uses_encoder_feedback_before_travel_or_policy(monkeypatch):
    start = CARRY_STARTS["pill lifted 8 cm (the demo's hand_over start)"]

    class StalledDriver(FakeDriver):
        def read(self):
            return list(start), 0.2

    d = StalledDriver(q0=start, gripper0=0.2)
    policy_calls = []
    monkeypatch.setattr(real_robot, "policy_config", lambda: {"use_policy": True, "policy": "unused.zip"})
    monkeypatch.setattr(real_robot, "_run_policy", lambda *a: policy_calls.append(a))
    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=scene_with_hand(0.30, 0.02), realtime=False
    )
    assert result["reason"] == "error", result["events"]
    assert any("rise did not reach" in e for e in result["events"])
    assert policy_calls == [] and d.commands and all(g == 0.2 for _, g in d.commands)
    assert d.stopped and d.torque and "released" not in result["events"]


@pytest.mark.parametrize("policy_succeeded", [False, True])
def test_policy_return_cannot_release_at_an_unreached_fallback(monkeypatch, policy_succeeded):
    monkeypatch.setattr(real_robot, "policy_config", lambda: {"use_policy": True, "policy": "unused.zip"})
    scene = scene_with_hand(0.42, 0.0)()
    calls = []

    def run_policy(driver, scene_fn, path, *, timeout, safety_scene, cancelled, clock, sleep, on_event):
        guarded = scene_fn()
        assert not guarded["hand"]["seen"]
        assert guarded["zones"]["handoff"] == scene["zones"]["handoff"]
        assert safety_scene()["hand"]["seen"]
        assert cancelled is real_robot.cancelled
        calls.append(guarded)
        return policy_succeeded

    monkeypatch.setitem(sys.modules, "rl.policy_runner", types.SimpleNamespace(run_policy=run_policy))
    monkeypatch.setattr(real_robot, "_servo_path", lambda *a, **kw: (None, "blocked"))
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=lambda: scene, timeout=60, realtime=False
    )
    assert result["reason"] == "error" and len(calls) == 1, result["events"]
    assert any("fallback deadline" in e for e in result["events"])
    assert "at the handoff zone" not in result["events"] and "released" not in result["events"]
    assert d.stopped and d.torque and all(g == 0.2 for _, g in d.commands)
    assert scene["hand"]["seen"]  # the policy wrapper did not mutate the tracker


def test_losing_hand_tracking_resets_the_steady_release_timer(monkeypatch):
    q, ok = ik.solve_grasp((0.28, -0.08, 0.08), real_robot.START_PITCH)
    assert ok
    pacer = replay.Pacer(False)
    scene = scene_with_hand(0.28, -0.08)()

    def source():
        scene["hand"]["seen"] = not (0.5 <= pacer.now() < 1.5)
        return scene

    d = FakeDriver(q0=q, gripper0=0.2)
    session = replay.Session(d, source, pacer, hand_freeze=False)
    session.start()
    monkeypatch.setattr(real_robot, "_servo_path", lambda *a, **kw: (None, "holding"))
    real_robot._track_hand(session, timeout=10)
    assert 2.5 <= pacer.now() < 2.6
    assert any(e.startswith("hand steady within") for e in session.events)


def test_tracking_release_uses_measured_pose_not_commanded_pose(monkeypatch):
    # Pretend a policy commanded the zone but a stalled arm still reports home.
    d = FakeDriver(gripper0=0.2)
    session = replay.Session(d, scene_with_hand(0.42, 0.0), replay.Pacer(False), hand_freeze=False)
    session.start()
    target = real_robot._Target(session)
    session.q, ok = ik.solve_grasp(target.zone, real_robot.START_PITCH)
    assert ok
    monkeypatch.setattr(real_robot, "_servo_path", lambda *a, **kw: (None, "blocked"))
    with pytest.raises(replay.Abort, match="error"):
        real_robot._track_hand(session, timeout=0, target=target)
    assert "at the handoff zone" not in session.events
    assert d.stopped and d.torque and d.read()[1] == 0.2
