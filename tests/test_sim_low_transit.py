"""A transit height below the home pose (a short obstacle) must not fail the first move from home.

From home the grasp sits about 0.30 m up, folded near the base, where no straight descent with the jaws
down exists below about 0.19 m. `SimRobot._rise` skips that descent instead of failing, and the travel leg
reaches the transit height on its own (2026-10-18: the tape-measure obstacle, 0.0572 m tall, gives
transit_z = 0.137 m under the starter rule "tallest + 0.08").
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from arm import ik
from arm.sim_robot import SimRobot

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))


def _short_obstacle_scene() -> dict:
    scene = copy.deepcopy(FAKE_SCENE)
    for item in scene["obstacles"].values():
        item["h"] = 0.0572
    return scene


@pytest.mark.parametrize("transit_z", [0.137, 0.16, 0.18])
def test_a_low_transit_from_home_picks_without_ik_failures(transit_z):
    robot = SimRobot(_short_obstacle_scene(), record=False)
    home_grasp_z = ik.fk_grasp(robot._q_cmd)[0][2]
    assert transit_z < home_grasp_z  # the "rise" is a descent from home
    assert robot.pick("juice_box", transit_z=transit_z) is True
    stats = robot._stats
    assert stats["ik_failures"] == 0 and stats["collisions"] == 0, stats


def test_a_rise_above_the_grasp_still_moves_straight_up():
    robot = SimRobot(_short_obstacle_scene(), record=False)
    assert robot.pick("juice_box", transit_z=0.2)
    before = ik.fk_grasp(robot._q_cmd)[0]
    assert robot._rise(0.25) is True
    after = ik.fk_grasp(robot._q_cmd)[0]
    assert after[2] == pytest.approx(0.25, abs=0.005)
    assert abs(after[0] - before[0]) < 0.005 and abs(after[1] - before[1]) < 0.005


def test_an_unreachable_rise_still_counts_an_ik_failure():
    robot = SimRobot(_short_obstacle_scene(), record=False)
    assert robot.pick("juice_box", transit_z=0.2)
    assert robot._rise(0.9) is False  # far above the reach: a real rise that has no plan
    assert robot._stats["ik_failures"] == 1
