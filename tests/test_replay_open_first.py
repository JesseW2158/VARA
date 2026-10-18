"""A replay that grasps starts with the gripper open, as every simulated plan does.

2026-10-18, real arm: "Pick up the tea" planned path -> grip close -> path -> carry_to_hand with no open
step (SimRobot starts with the gripper open, so its pick never opened), the real jaws were still closed from
the previous task, came down shut and the grasp missed.
"""

from __future__ import annotations

import math

import yaml

from arm import ik, replay
from arm.driver import FakeDriver

HOME = yaml.safe_load(open("config/scene.yaml", encoding="utf-8"))["home_q"]


def _above_and_down():
    up, ok = ik.solve((0.22, 0.0, 0.15), -math.pi / 2, q0=HOME)
    assert ok
    down, ok = ik.solve((0.22, 0.0, 0.10), -math.pi / 2, q0=up)
    assert ok
    return [list(HOME), *replay.bridge(HOME, up, math.radians(2.0))], replay.bridge(
        up, down, math.radians(2.0)
    )


def _traj(*segments):
    return {
        "version": 1,
        "skill": "open_first",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": list(segments),
    }


def test_a_closed_gripper_opens_before_a_plan_that_grasps():
    approach, descend = _above_and_down()
    d = FakeDriver(q0=HOME, gripper0=0.0)
    traj = _traj(
        {"kind": "path", "q": approach + descend},
        {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": False},
    )
    result = replay.replay(traj, driver=d, realtime=False)
    assert any("grip open first" in e for e in result["events"]), result["events"]
    grippers = [g for _q, g in d.commands]
    assert grippers[0] == 1.0  # opened before the first waypoint
    first_close = grippers.index(0.0)
    assert all(g == 1.0 for g in grippers[:first_close])  # the approach ran open


def test_a_plan_without_grip_steps_leaves_the_gripper_alone():
    approach, _descend = _above_and_down()
    d = FakeDriver(q0=HOME, gripper0=0.3)
    result = replay.replay(_traj({"kind": "path", "q": approach}), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert not any("grip open first" in e for e in result["events"])
    assert all(g == 0.3 for _q, g in d.commands)


def test_an_open_gripper_is_not_opened_again():
    approach, descend = _above_and_down()
    d = FakeDriver(q0=HOME, gripper0=1.0)
    traj = _traj(
        {"kind": "path", "q": approach + descend},
        {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": False},
    )
    result = replay.replay(traj, driver=d, realtime=False)
    assert not any("grip open first" in e for e in result["events"])


def test_the_start_open_holds_the_last_command_not_a_sagged_reading():
    approach, descend = _above_and_down()
    sagged = list(HOME)
    sagged[1] -= math.radians(12)
    sagged[2] += math.radians(3)
    d = FakeDriver(q0=HOME, gripper0=0.0)
    d.command(HOME, 0.0)
    d.set_present(sagged)
    before = len(d.commands)
    traj = _traj(
        {"kind": "path", "q": approach + descend},
        {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": False},
    )
    replay.replay(traj, driver=d, realtime=False)
    q_open, g_open = d.commands[before]
    assert g_open == 1.0
    assert max(abs(a - b) for a, b in zip(q_open, HOME, strict=True)) < 1e-9
