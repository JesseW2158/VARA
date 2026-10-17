"""Contract: arm.replay.validate(traj, limits, fk, box) -> list[str].

limits: joint name -> [low, high] in radians (config/limits.yaml).
fk(q): returns the tip position (x, y, z) in meters for 5 joint angles.
box: {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]} (config/scene.yaml, workspace).
An empty list means accepted. Spec: docs/interfaces.md, "Trajectory". Don't edit without team agreement.
"""

import copy
import json
import math
import pathlib

import pytest

validate = pytest.importorskip("arm.replay", reason="S0 implements validate in arm/replay.py").validate

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
LIMITS = {
    "shoulder_pan": [-1.9, 1.9],
    "shoulder_lift": [-1.9, 1.9],
    "elbow_flex": [-1.7, 1.7],
    "wrist_flex": [-1.7, 1.7],
    "wrist_roll": [-2.7, 2.7],
}
BOX = {"x": [0.10, 0.40], "y": [-0.25, 0.25], "z": [0.02, 0.30]}
DEG = math.pi / 180


def fk_inside(q):
    return (0.25, 0.0, 0.10)


def traj():
    return json.loads((FIXTURES / "trajectory_ok.json").read_text())


def with_path(points):
    t = traj()
    t["segments"] = [{"kind": "path", "q": points}]
    return t


Q0 = [0.0, -1.75, 1.6, 0.75, 0.0]


def test_contract_example_is_accepted():
    assert validate(traj(), LIMITS, fk_inside, BOX) == []


def test_waypoint_inside_the_3_degree_margin_is_rejected():
    q = list(Q0)
    q[2] = 1.7 - 2 * DEG  # inside the joint range, but within 3 degrees of the limit
    assert validate(with_path([q]), LIMITS, fk_inside, BOX)


def test_waypoint_outside_the_limits_is_rejected():
    q = list(Q0)
    q[0] = 2.5
    assert validate(with_path([q]), LIMITS, fk_inside, BOX)


def test_step_over_2_degrees_per_tick_is_rejected():
    q1 = list(Q0)
    q1[0] += 3 * DEG
    assert validate(with_path([Q0, q1]), LIMITS, fk_inside, BOX)


def test_step_under_2_degrees_per_tick_is_accepted():
    q1 = list(Q0)
    q1[0] += 1.5 * DEG
    assert validate(with_path([Q0, q1]), LIMITS, fk_inside, BOX) == []


def test_jump_between_segments_is_left_to_bridging():
    q_far = list(Q0)
    q_far[0] += 20 * DEG
    t = traj()
    t["segments"] = [
        {"kind": "path", "q": [Q0]},
        {"kind": "grip", "action": "open"},
        {"kind": "path", "q": [q_far]},
    ]
    assert validate(t, LIMITS, fk_inside, BOX) == []


def test_tip_outside_the_box_is_rejected():
    assert validate(traj(), LIMITS, lambda q: (0.50, 0.0, 0.10), BOX)
    assert validate(traj(), LIMITS, lambda q: (0.25, 0.0, 0.01), BOX)
    assert validate(traj(), LIMITS, lambda q: (0.25, 0.0, 0.35), BOX)


def test_unknown_live_primitive_is_rejected():
    t = traj()
    t["segments"].append({"kind": "live", "primitive": "dance"})
    assert validate(t, LIMITS, fk_inside, BOX)


def test_wrong_waypoint_length_is_rejected():
    assert validate(with_path([[0.0, 0.0, 0.0, 0.0]]), LIMITS, fk_inside, BOX)


def test_non_finite_numbers_are_rejected():
    q = list(Q0)
    q[1] = float("nan")
    assert validate(with_path([q]), LIMITS, fk_inside, BOX)


def test_every_problem_is_reported():
    q_bad_limit = list(Q0)
    q_bad_limit[0] = 2.5
    t = with_path([Q0, q_bad_limit])  # over the limit and a huge step
    t["segments"].append({"kind": "live", "primitive": "dance"})
    problems = validate(t, LIMITS, fk_inside, BOX)
    assert len(problems) >= 3
    assert all(isinstance(p, str) and p.strip() for p in problems)


def test_input_is_not_modified():
    t = traj()
    snapshot = copy.deepcopy(t)
    validate(t, LIMITS, fk_inside, BOX)
    assert t == snapshot
