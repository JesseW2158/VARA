"""A replay that starts with the real arm sagged just behind the workspace box's near edge.

2026-10-18, real arm: after a move that ended at home, the arm rested with its tip at (0.087, 0, 0.281),
1.3 cm behind the box's near edge (x 0.10): the soft servos let the folded arm sag back under its own weight.
Every following bridge was refused at waypoint 0, so no second command could run. `Session.bridge_to` now
bridges back into the box from up to SAG_RECOVERY_M behind the near edge, for the bridge only.
"""

from __future__ import annotations

import math

import pytest
import yaml

from arm import ik, replay
from arm.driver import FakeDriver

HOME = yaml.safe_load(open("config/scene.yaml", encoding="utf-8"))["home_q"]


def _offset(q, lift_deg=0.0, elbow_deg=0.0):
    out = list(q)
    out[1] += math.radians(lift_deg)
    out[2] += math.radians(elbow_deg)
    return out


SAGGED = _offset(HOME, lift_deg=-12, elbow_deg=3)  # tip about (0.088, 0, 0.280), as the real arm read


def _traj(*qs):
    return {
        "version": 1,
        "skill": "sag_test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [{"kind": "path", "q": [list(q) for q in qs]}],
    }


def _hover_traj():
    """Home, then 5 cm up and out along the same pan: a path wholly inside the workspace box."""
    up, ok = ik.solve((0.20, 0.0, 0.20), -math.pi / 4, q0=HOME)
    assert ok
    path = [list(HOME), *replay.bridge(HOME, up, math.radians(2.0))]
    return _traj(*path)


def test_the_sagged_pose_matches_what_the_real_arm_read():
    x, y, z = ik.fk_tip(SAGGED)
    box = replay.load_box()
    assert x < box["x"][0] and box["x"][0] - x < replay.SAG_RECOVERY_M
    assert abs(x - 0.087) < 0.005 and abs(z - 0.281) < 0.005


def test_a_replay_from_a_sagged_home_bridges_back_into_the_box():
    d = FakeDriver(q0=SAGGED, gripper0=1.0)
    result = replay.replay(_hover_traj(), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert any("behind the box's near edge" in e for e in result["events"])


def test_the_gate_box_is_the_workspace_box_again_after_the_bridge():
    d = FakeDriver(q0=SAGGED, gripper0=1.0)
    s = replay.Session(d)
    s.start()
    s.bridge_to(HOME, where="segment 0 bridge")
    assert s.gate_box is s.box
    # a tick outside the strict box (even where the bridge was allowed to run) now freezes the arm
    with pytest.raises(replay.Abort):
        s.check_box(ik.fk_tip(SAGGED), "after the bridge")
    assert d.stopped


@pytest.mark.parametrize(
    ("label", "tip", "pitch"),
    [
        ("beyond the far edge", (0.43, 0.0, 0.12), -math.pi / 4),
        ("to the side", (0.20, 0.28, 0.15), -math.pi / 2),
    ],
)
def test_other_starts_outside_the_box_are_still_refused(label, tip, pitch):
    q, ok = ik.solve(tip, pitch, q0=HOME)
    assert ok, f"no IK pose for the {label} start"
    start = ik.fk_tip(q)
    box = replay.load_box()
    assert not replay.inside_box(start, box)
    assert replay.sag_recovery_box(box, start) is None
    d = FakeDriver(q0=q, gripper0=1.0)
    result = replay.replay(_hover_traj(), driver=d, realtime=False)
    assert not result["ok"] and result["reason"] == "error"
    assert any("run home first" in e for e in result["events"]), result["events"]
    assert d.commands == []  # nothing moved


def test_recovery_box_only_moves_the_near_edge_back():
    box = replay.load_box()
    rec = replay.sag_recovery_box(box, (0.087, 0.0, 0.281))
    assert rec is not None
    assert rec["x"][0] == pytest.approx(0.087 - replay.SAG_SLACK_M)
    assert rec["x"][1] == box["x"][1] and rec["y"] == list(box["y"]) and rec["z"] == list(box["z"])
    assert replay.sag_recovery_box(box, (0.15, 0.0, 0.2)) is None  # inside: no recovery needed
    assert replay.sag_recovery_box(box, (0.087, 0.0, 0.31)) is None  # behind and above the ceiling
    far = box["x"][0] - replay.SAG_RECOVERY_M - 0.005
    assert replay.sag_recovery_box(box, (far, 0.0, 0.28)) is None  # more than the cap behind the edge


def test_a_start_behind_the_near_edge_and_above_the_ceiling_is_refused():
    q = _offset(HOME, lift_deg=-14)
    q[3] -= math.radians(20)  # wrist up: tip about (0.06, 0, 0.345), behind the near edge and over 0.30
    x, _y, z = ik.fk_tip(q)
    box = replay.load_box()
    assert x < box["x"][0] and z > box["z"][1]
    d = FakeDriver(q0=q, gripper0=1.0)
    result = replay.replay(_hover_traj(), driver=d, realtime=False)
    assert not result["ok"] and result["reason"] == "error"
    assert any("run home first" in e for e in result["events"]), result["events"]
    assert d.commands == []


def test_the_same_driver_continues_from_its_last_command_and_never_walks_the_goal_out():
    # the real trigger (review 2026-10-18): one driver across replays, the last leg ended at home, and the
    # arm reads sagged behind the near edge while its goal is still home
    d = FakeDriver(q0=HOME, gripper0=1.0)
    d.command(HOME, 1.0)
    d.set_present(SAGGED)
    sent_before = len(d.commands)
    result = replay.replay(_hover_traj(), driver=d, realtime=False)
    assert result["ok"], result["events"]
    assert not any("behind the box's near edge" in e for e in result["events"])  # no recovery needed
    box = replay.load_box()
    sent = [q for q, _g in d.commands[sent_before:]]
    assert sent
    outside = [ik.fk_tip(q) for q in sent if not replay.inside_box(ik.fk_tip(q), box, replay.BOX_GATE_M)]
    assert outside == []


@pytest.mark.parametrize(
    ("label", "tip"),
    [("low, beside the base", (0.06, 0.10, 0.07)), ("at table height", (0.061, 0.0, 0.038))],
)
def test_a_start_behind_the_edge_far_from_home_is_not_recovered(label, tip):
    box = replay.load_box()
    assert replay.sag_recovery_box(box, tip) is None, label
    assert replay.sag_recovery_box(box, ik.fk_tip(SAGGED)) is not None  # the home sag still is
