"""A8 SimRobot on the MuJoCo scene and A12 starter skills.

Most scenes here derive from config/fake_scene.json as committed (`FAKE_SCENE`), so the suite follows
A15's re-taping of the layout: the hand, the handoff zone, the slots and the cereal box are read from the
file, never hard-coded. The A8 done-when (place_in_tray on the juice box 5 of 5 perturbed trials, a candidate
that drops the box beside the tray fails the spec, a candidate without transit_z hits the cereal box) runs on
that file.

A few tests of generic SimRobot behaviour need geometry the real props of 2026-10-18 no longer have (a 20 cm
obstacle in the path of a no-transit move, a box taller than twice the default release height, the 3.5 cm
pill bottle's attach offsets, tray walls a gripper can hit, a carry that descends onto the handoff point).
They run on the kit layout (`KIT_SCENE`, `kit_model`): the inputs committed before the props changed,
rendered by sim.model.build_scene into a temp dir, so the shipped props cannot make them vacuous.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import random
import time
from datetime import datetime

import mujoco
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from known_limits import SHORT_OBSTACLE

from arm import ik, replay, sim_robot
from arm.sim_robot import DT, ROLL_OFFSET, SimRobot, SimTimeLimit
from sim import restricted, success
from sim.candidate_check import check_source
from sim.model import build_scene

ROOT = pathlib.Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills" / "_starter"
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
SCENE_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "scene.json").read_text()))
SKILL_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "skill.json").read_text()))
LIMITS = replay.load_limits()
BOX = replay.load_box()
HOME_Q = [float(v) for v in ik.home_q()]
DEG = math.pi / 180
ROBOT_LABELS = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"}
OBJECTS = ("pill_bottle", "juice_box", "tea_tin")
# The kit's 20 cm cereal box + 0.08, the kit starter rule. On the shipped layout the starter skills now take
# the tallest object or obstacle + 0.08 (0.18 m), so 0.28 is a higher, still valid transit there.
TRANSIT_Z = 0.20 + 0.08
WALL_TIMES: dict[str, float] = {}

# The kit layout: config/objects.yaml (shapes and sizes), config/scene.yaml zones.tray and
# config/fake_scene.json as committed before the 2026-10-18 props. build_scene.render of these inputs
# reproduced that commit's scene_ours.xml byte for byte (checked 2026-10-18). The scene's `h` values matter
# too: SimRobot places obstacles and computes the carry clearance from them.
KIT_OBJECTS = {
    10: {"name": "pill_bottle", "kind": "object", "shape": "cylinder", "size": [0.035, 0.035], "h": 0.07},
    11: {"name": "juice_box", "kind": "object", "shape": "box", "size": [0.063, 0.04], "h": 0.105},
    12: {"name": "tea_tin", "kind": "object", "shape": "cylinder", "size": [0.04, 0.04], "h": 0.09},
    30: {"name": "cereal_box", "kind": "obstacle", "shape": "box", "size": [0.19, 0.06], "h": 0.2},
    40: {"name": "hand", "kind": "hand", "h": 0.03},
}
KIT_SCENE = {
    "t": 1792209600.12,
    "objects": {
        "pill_bottle": {
            "marker": 10,
            "x": 0.18,
            "y": -0.12,
            "yaw": 0.0,
            "h": 0.07,
            "size": [0.035, 0.035],
            "seen": True,
        },
        "juice_box": {
            "marker": 11,
            "x": 0.24,
            "y": -0.04,
            "yaw": 0.3,
            "h": 0.105,
            "size": [0.063, 0.04],
            "seen": True,
        },
        "tea_tin": {
            "marker": 12,
            "x": 0.16,
            "y": -0.02,
            "yaw": 0.0,
            "h": 0.09,
            "size": [0.04, 0.04],
            "seen": True,
        },
    },
    "obstacles": {
        "cereal_box": {
            "marker": 30,
            "x": 0.28,
            "y": 0.07,
            "yaw": 0.0,
            "h": 0.2,
            "size": [0.19, 0.06],
            "seen": True,
        },
    },
    "hand": {"marker": 40, "x": 0.28, "y": -0.08, "z": 0.03, "seen": False},
    "zones": {
        "tray": {
            "x": 0.3,
            "y": 0.22,
            "r": 0.07,
            "wall": 0.035,
            "slots": [[0.275, 0.195], [0.325, 0.195], [0.275, 0.245], [0.325, 0.245]],
        },
        "handoff": {"x": 0.28, "y": -0.08, "z": 0.1},
    },
}


def perturb(scene: dict, trial: int, seed: int = 7) -> dict:
    """The scoring perturbation: trial 1 unchanged, then +-0.01 m and +-10 deg per object, +-0.005 m per
    obstacle, from random.Random(f"{seed}:{trial}")."""
    out = copy.deepcopy(scene)
    if trial == 1:
        return out
    rng = random.Random(f"{seed}:{trial}")
    for item in out["objects"].values():
        item["x"] += rng.uniform(-0.01, 0.01)
        item["y"] += rng.uniform(-0.01, 0.01)
        item["yaw"] += math.radians(rng.uniform(-10, 10))
    for item in out["obstacles"].values():
        item["x"] += rng.uniform(-0.005, 0.005)
        item["y"] += rng.uniform(-0.005, 0.005)
    return out


def validate(robot: SimRobot) -> list[str]:
    return replay.validate(robot.trajectory(), LIMITS, ik.fk_tip, BOX)


def paths(robot: SimRobot) -> list[list[list[float]]]:
    return [s["q"] for s in robot.trajectory()["segments"] if s["kind"] == "path"]


def max_tip_speed(robot: SimRobot) -> float:
    """The fastest tip step between consecutive recorded waypoints, in m/s (what validate's rule of thumb
    and the real arm see)."""
    worst = 0.0
    for path in paths(robot):
        tips = [ik.fk_tip(q) for q in path]
        worst = max([worst] + [math.dist(a, b) / DT for a, b in zip(tips[:-1], tips[1:], strict=True)])
    return worst


def descend_to(robot: SimRobot, name: str, along_c: float, across: float, fraction: float) -> None:
    """Open, align the jaws with `name` and put the grasp center at `fraction` of its height, offset from
    its center line by `along_c` toward the moving jaw and `across` perpendicular to the closing direction."""
    x, y, _z, yaw = robot._object_pose(name)
    gz = fraction * 2.0 * robot._object_half_height(name)
    robot.open_gripper()
    q_above, ok = robot._goal("grasp", (x, y, gz + 0.08), -math.pi / 2, robot._q_cmd)
    assert ok
    roll = robot._roll_for(yaw, float(q_above[0]))
    assert robot._move("grasp", (x, y, gz + 0.08), -math.pi / 2, 0.2, roll=roll)
    _a, c = ik.jaw_axes(robot._q_cmd)
    norm = math.hypot(c[0], c[1])
    cx, cy = c[0] / norm, c[1] / norm
    target = (x + along_c * cx - across * cy, y + along_c * cy + across * cx, gz)
    assert robot._move("grasp", target, -math.pi / 2, 0.2, straight=True)


def grips(robot: SimRobot) -> list[dict]:
    return [s for s in robot.trajectory()["segments"] if s["kind"] == "grip"]


def self_collisions(robot: SimRobot) -> list[str]:
    out = []
    for phrase in robot._collision_log:
        a, _, b = phrase.partition(" hit ")
        if a in ROBOT_LABELS and b in ROBOT_LABELS:
            out.append(phrase)
    return out


def load_skill(name: str) -> dict:
    return json.loads((STARTER / f"{name}.json").read_text(encoding="utf-8"))


def in_tray(scene: dict, name: str) -> bool:
    item, tray = scene["objects"][name], scene["zones"]["tray"]
    return math.hypot(item["x"] - tray["x"], item["y"] - tray["y"]) <= tray["r"]


@pytest.fixture(scope="module")
def robot() -> SimRobot:
    return SimRobot(FAKE_SCENE, skills_dir=STARTER)


@pytest.fixture(scope="module")
def kit_model(tmp_path_factory) -> pathlib.Path:
    """The kit layout's scene model, rendered by build_scene into a temp dir beside a copy of the arm MJCF
    whose meshdir points at sim/model/assets (the scene's include and the meshes are relative paths), so
    nothing is written into sim/model."""
    out = tmp_path_factory.mktemp("kit_model")
    model_dir = ROOT / "sim" / "model"
    arm_xml = (model_dir / "so101_new_calib.xml").read_text(encoding="utf-8")
    assert arm_xml.count('meshdir="assets"') == 1
    assets = (model_dir / "assets").resolve().as_posix()
    (out / "so101_new_calib.xml").write_text(
        arm_xml.replace('meshdir="assets"', f'meshdir="{assets}"'), encoding="utf-8", newline="\n"
    )
    path = out / "scene_kit.xml"
    text = build_scene.render(KIT_OBJECTS, {"zones": {"tray": KIT_SCENE["zones"]["tray"]}}, KIT_SCENE)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture(scope="module")
def kit_robot(kit_model) -> SimRobot:
    return SimRobot(KIT_SCENE, skills_dir=STARTER, model_path=kit_model)


def test_kit_model_has_the_kit_geometry(kit_robot):
    """The premise of the kit-pinned tests: the 20 cm cereal box, the 10.5 cm juice box, the 3.5 cm pill
    bottle and a tray with 3.5 cm walls, and the kit scene is a valid scene."""
    SCENE_VALIDATOR.validate(KIT_SCENE)
    kit_robot.reset(KIT_SCENE)
    m = kit_robot._model

    def geom(name: str) -> int:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, name
        return gid

    assert np.allclose(m.geom_size[geom("cereal_box")], [0.095, 0.03, 0.10])
    assert kit_robot._object_half_height("juice_box") == pytest.approx(0.0525)
    assert kit_robot._object_width("pill_bottle") == pytest.approx(0.035)
    wall = geom("tray_wall_nx")
    assert m.geom_pos[wall][2] + m.geom_size[wall][2] == pytest.approx(0.035)


# ---------------------------------------------------------------- model, reset, scene


def test_reset_round_trip_matches_the_fake_scene(robot):
    robot.reset(FAKE_SCENE)
    scene = robot.scene()
    SCENE_VALIDATOR.validate(scene)
    for group in ("objects", "obstacles"):
        for name, item in FAKE_SCENE[group].items():
            got = scene[group][name]
            assert abs(got["x"] - item["x"]) < 0.001 and abs(got["y"] - item["y"]) < 0.001, (name, got)
            assert abs(got["yaw"] - item["yaw"]) < 1e-6
            assert got["seen"] is True
            assert (got["marker"], got["h"], got["size"]) == (item["marker"], item["h"], item["size"])
    assert scene["hand"] == FAKE_SCENE["hand"]
    assert scene["zones"] == FAKE_SCENE["zones"]
    assert scene["t"] == FAKE_SCENE["t"]
    assert robot.stats == {
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "sim_s": 0.0,
        "first_collision": None,
    }
    assert robot.events == []
    assert validate(robot) == []  # the empty trajectory is one waypoint at home_q
    assert robot.trajectory()["segments"] == [{"kind": "path", "q": [HOME_Q]}]
    assert robot.final_scene() == scene


def test_reset_rejects_bad_scenes(robot):
    with pytest.raises(ValueError):
        robot.reset({"objects": {}})
    bad = copy.deepcopy(FAKE_SCENE)
    bad["objects"]["banana"] = dict(bad["objects"]["tea_tin"], marker=13)
    with pytest.raises(ValueError):
        robot.reset(bad)
    robot.reset(FAKE_SCENE)


@pytest.mark.parametrize("group,name", [("objects", "pill_bottle"), ("obstacles", "cereal_box")])
def test_unseen_markers_remain_unseen_after_reset_and_motion(robot, group, name):
    scene = copy.deepcopy(FAKE_SCENE)
    scene[group][name]["seen"] = False
    robot.reset(scene)
    assert robot.scene()[group][name] == scene[group][name]
    assert robot.move_to(0.25, 0.0, 0.15)
    after = robot.final_scene()
    assert after[group][name] == scene[group][name]
    SCENE_VALIDATOR.validate(after)
    # Serialization and another simulation cannot manufacture a new camera detection either.
    robot.reset(json.loads(json.dumps(after)))
    assert robot.scene()[group][name] == scene[group][name]
    robot.reset(FAKE_SCENE)
    assert robot.scene()[group][name]["seen"] is True


def test_pick_unseen_object_refuses_before_releasing_a_held_object(robot):
    scene = copy.deepcopy(FAKE_SCENE)
    scene["objects"]["pill_bottle"]["seen"] = False
    robot.reset(scene)
    assert robot.pick("juice_box", transit_z=TRANSIT_Z)
    stats, trajectory = robot.stats, robot.trajectory()
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z) is False
    assert robot.stats == stats and robot.trajectory() == trajectory
    assert robot._held == "juice_box"
    assert robot.place_in_zone("tray", transit_z=TRANSIT_Z)
    assert in_tray(robot.final_scene(), "juice_box")


def test_unseen_object_remains_a_collision_body_but_cannot_be_grasped(robot):
    scene = copy.deepcopy(FAKE_SCENE)
    scene["objects"]["pill_bottle"]["seen"] = False
    robot.reset(scene)
    descend_to(robot, "pill_bottle", 0.0, 0.0, 0.5)
    assert robot.close_gripper() is False
    assert robot._held is None
    assert robot.stats["collisions"] >= 1
    assert "pill_bottle" in robot.stats["first_collision"]
    # Physics can push the stale collision body, but it cannot update an unseen camera pose.
    assert robot.final_scene()["objects"]["pill_bottle"] == scene["objects"]["pill_bottle"]
    assert grips(robot)[-1] == {"kind": "grip", "action": "close", "expect_held": False}


def test_unseen_obstacle_still_rejects_a_colliding_candidate(kit_robot):
    """An unseen obstacle stays a collision body. Kit layout: there the 20 cm cereal box stands in the path
    of a no-transit pick and place; the shipped 5.7 cm obstacle is cleared without a transit (that layout
    property is test_candidate_without_transit_collides_with_the_cereal_box's)."""
    scene = copy.deepcopy(KIT_SCENE)
    scene["obstacles"]["cereal_box"]["seen"] = False
    kit_robot.reset(scene)
    assert kit_robot.pick("juice_box")
    kit_robot.place_in_zone("tray")
    assert kit_robot.stats["collisions"] >= 1
    assert "cereal_box" in kit_robot.stats["first_collision"]
    assert kit_robot.final_scene()["obstacles"]["cereal_box"] == scene["obstacles"]["cereal_box"]


def test_post_handoff_snapshot_cannot_resurrect_the_taken_object(robot):
    robot.reset(FAKE_SCENE)
    assert robot.run_skill("hand_over", {"object": "pill_bottle"})
    snapshot = json.loads(json.dumps(robot.final_scene()))
    taken = snapshot["objects"]["pill_bottle"]
    assert taken["seen"] is False
    robot.reset(snapshot)
    assert robot.final_scene()["objects"]["pill_bottle"] == taken
    assert robot.run_skill("hand_over", {"object": "pill_bottle"}) is False
    assert robot.stats["sim_s"] == 0.0 and robot.events == []
    # A snapshot cannot distinguish occlusion from removal: retain the stale collision body.
    x, y, _z, yaw = robot._object_pose("pill_bottle")
    assert (x, y, yaw) == pytest.approx((taken["x"], taken["y"], taken["yaw"]))


def test_sequential_handoffs_keep_both_objects_taken(robot):
    robot.reset(FAKE_SCENE)
    for name in ("pill_bottle", "juice_box"):
        assert robot.run_skill("hand_over", {"object": name})
        assert robot.final_scene()["objects"][name]["seen"] is False
    after = robot.final_scene()
    assert not after["objects"]["pill_bottle"]["seen"]
    assert not after["objects"]["juice_box"]["seen"]
    assert after["objects"]["tea_tin"]["seen"]
    assert robot.events == ["handoff_done", "handoff_done"]
    assert robot.stats["collisions"] == 0, robot._collision_log
    assert robot.stats["limit_hits"] == 0 and robot.stats["ik_failures"] == 0
    assert validate(robot) == []


def test_public_surface_is_the_robot_api_plus_runner_helpers(robot):
    allowed = {
        "scene",
        "move_to",
        "set_roll",
        "open_gripper",
        "close_gripper",
        "home",
        "pick",
        "place",
        "place_in_zone",
        "carry_to_hand",
        "run_skill",
        "reset",
        "set_skill",
        "trajectory",
        "final_scene",
        "stats",
        "events",
        "run_entry",
    }
    public = {n for n in dir(robot) if not n.startswith("_")}
    assert public == allowed
    # stats and events are copies: a candidate cannot edit the counters it is scored on
    robot.reset(FAKE_SCENE)
    robot.stats["collisions"] = 99
    robot.events.append("handoff_done")
    assert robot.stats["collisions"] == 0 and robot.events == []
    with pytest.raises(AttributeError):
        robot.stats = {}


def test_scene_and_trajectory_are_fresh_copies(robot):
    robot.reset(FAKE_SCENE)
    scene = robot.scene()
    scene["objects"]["juice_box"]["x"] = 9.0
    scene["zones"]["tray"]["r"] = 9.0
    assert robot.scene()["objects"]["juice_box"]["x"] == FAKE_SCENE["objects"]["juice_box"]["x"]
    assert robot.scene()["zones"]["tray"]["r"] == FAKE_SCENE["zones"]["tray"]["r"]
    robot.set_skill("demo", {"object": "juice_box"})
    traj = robot.trajectory()
    traj["segments"].clear()
    assert robot.trajectory()["segments"]
    assert robot.trajectory()["skill"] == "demo" and robot.trajectory()["args"] == {"object": "juice_box"}


# ---------------------------------------------------------------- moves, speed, recording


def test_move_to_inside_and_outside_the_box(robot):
    robot.reset(FAKE_SCENE)
    assert robot.move_to(0.25, 0.0, 0.15) is True
    tip = ik.fk_tip(robot._q_cmd)
    assert math.dist(tip, (0.25, 0.0, 0.15)) < 0.005
    before = robot.trajectory()
    sim_s = robot.stats["sim_s"]
    for bad in (
        (0.5, 0.0, 0.1),
        (0.25, 0.0, 0.01),
        (0.25, 0.3, 0.1),
        (0.25, 0.0, 0.31),
        (math.nan, 0.0, 0.1),
    ):
        assert robot.move_to(*bad) is False, bad
    assert robot.move_to("x", 0.0, 0.1) is False
    assert robot.trajectory() == before and robot.stats["sim_s"] == sim_s  # refused: nothing moved
    assert robot.stats["ik_failures"] == 0
    # inside the box but out of reach: an IK failure, no motion
    assert robot.move_to(0.40, 0.25, 0.02) is False
    assert robot.stats["ik_failures"] == 1 and robot.trajectory() == before


@pytest.mark.parametrize("speed", [0.2, 0.1, 5.0])
def test_speed_caps_hold_on_the_recorded_path(robot, speed):
    robot.reset(FAKE_SCENE)
    assert robot.move_to(0.30, -0.15, 0.08, speed=speed)
    assert robot.move_to(0.20, 0.15, 0.25, speed=speed)
    cap = min(speed, 0.2)
    fastest = 0.0
    for path in paths(robot):
        tips = [ik.fk_tip(q) for q in path]
        for a, b, ta, tb in zip(path[:-1], path[1:], tips[:-1], tips[1:], strict=True):
            assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 2 * DEG + 1e-9
            step = math.dist(ta, tb) / DT
            assert step <= cap + 1e-3, step
            fastest = max(fastest, step)
    assert fastest > 0.5 * cap  # the trapezoid reaches cruising speed on a long move
    assert validate(robot) == []


def test_recording_shape_and_tick_accounting(robot):
    robot.reset(FAKE_SCENE)
    robot.set_skill("probe", {})
    assert robot.move_to(0.25, 0.0, 0.15) and robot.move_to(0.25, 0.1, 0.15)
    traj = robot.trajectory()
    assert traj["version"] == 1 and traj["skill"] == "probe" and traj["args"] == {}
    assert traj["dt"] == DT and traj["joints"] == list(replay.JOINTS)
    assert [s["kind"] for s in traj["segments"]] == ["path"]  # consecutive moves extend one segment
    n_path = len(traj["segments"][0]["q"])
    assert all(len(q) == 5 and all(round(v, 6) == v for v in q) for q in traj["segments"][0]["q"])
    assert robot.stats["sim_s"] == pytest.approx(n_path * DT, abs=1e-6)
    robot.open_gripper()
    assert robot.stats["sim_s"] == pytest.approx((n_path + 9) * DT, abs=1e-6)  # 0.3 s
    assert robot.close_gripper() is False
    assert robot.stats["sim_s"] == pytest.approx((n_path + 21) * DT, abs=1e-6)  # + 0.4 s
    assert robot.move_to(0.25, -0.1, 0.15)
    kinds = [s["kind"] for s in robot.trajectory()["segments"]]
    assert kinds == ["path", "grip", "grip", "path"]  # a grip closes the open path segment
    assert grips(robot) == [
        {"kind": "grip", "action": "open"},
        {"kind": "grip", "action": "close", "expect_held": False},
    ]
    assert validate(robot) == []


def test_record_false_keeps_a_valid_minimal_trajectory():
    quiet = SimRobot(FAKE_SCENE, record=False)
    assert quiet.move_to(0.25, 0.0, 0.15)
    quiet.open_gripper()
    assert quiet.trajectory()["segments"] == [{"kind": "path", "q": [[round(v, 6) for v in quiet._q_cmd]]}]
    assert validate(quiet) == []


def test_limit_hits_count_and_the_waypoint_is_clipped(robot):
    robot.reset(FAKE_SCENE)
    hi = LIMITS["wrist_roll"][1]
    q = np.array(HOME_Q)
    q[4] = hi - 0.5 * DEG  # within 1 degree of the limit
    robot._run_path([q])
    assert robot.stats["limit_hits"] == 1
    assert robot.trajectory()["segments"][-1]["q"][-1][4] <= hi - 3 * DEG
    assert validate(robot) == []


def test_set_roll_aligns_the_jaws_with_the_width(robot):
    robot.reset(FAKE_SCENE)
    # the documented offset: at pan 0, pointing down, roll = ROLL_OFFSET closes along world y (yaw 0)
    q, ok = ik.solve_grasp((0.18, 0.0, 0.05), -math.pi / 2, HOME_Q)
    assert ok
    q[4] = ROLL_OFFSET
    _, c = ik.jaw_axes(q)
    assert abs(math.degrees(math.atan2(c[1], c[0])) % 180 - 90) < 1.0
    for y, yaw in ((-0.15, 0.3), (0.12, -0.6), (0.0, 1.2)):
        robot.reset(FAKE_SCENE)
        assert robot.move_to(0.2, y, 0.08)
        robot.set_roll(yaw)
        _, c = ik.jaw_axes(robot._q_cmd)
        closing = math.atan2(c[1], c[0])
        width_dir = yaw + math.pi / 2
        diff = (closing - width_dir) % math.pi
        assert min(diff, math.pi - diff) < 3 * DEG, (y, yaw, math.degrees(diff))
        assert validate(robot) == []


# ---------------------------------------------------------------- trajectories always validate


@pytest.mark.parametrize("slot", FAKE_SCENE["zones"]["tray"]["slots"])
def test_moves_from_every_tray_slot_validate(robot, slot):
    robot.reset(FAKE_SCENE)
    assert robot.move_to(slot[0], slot[1], 0.10)
    assert robot.move_to(0.18, -0.12, 0.12)
    q_before = robot._q_cmd.copy()
    high = robot.move_to(slot[0], slot[1], 0.28)
    if not high:  # the brief: z 0.28 above slot 4 (0.325, 0.245) is out of reach; refused without motion
        assert slot == [0.325, 0.245] and robot.stats["ik_failures"] == 1
        assert np.array_equal(robot._q_cmd, q_before)
    robot.home()
    assert robot.stats["ik_failures"] == (0 if high else 1)
    assert np.allclose(robot._q_cmd, HOME_Q)
    assert validate(robot) == []


@pytest.mark.parametrize("name", OBJECTS)
def test_home_from_every_pick_pose_validates_and_is_collision_free(robot, name):
    robot.reset(FAKE_SCENE)
    assert robot.pick(name) is True
    robot.open_gripper()
    robot.home()
    assert np.allclose(robot._q_cmd, HOME_Q)
    assert robot.stats["ik_failures"] == 0 and robot.stats["limit_hits"] == 0
    assert robot._collision_log == []
    assert validate(robot) == []


# ---------------------------------------------------------------- grasping and placing


def test_pick_attaches_and_the_object_follows_the_gripper(robot):
    robot.reset(FAKE_SCENE)
    assert robot.pick("juice_box") is True
    close = [g for g in grips(robot) if g["action"] == "close"]
    assert close == [{"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}]
    (gx, gy, gz), _ = ik.fk_grasp(robot._q_cmd)
    x, y, z, _yaw = robot._object_pose("juice_box")
    assert math.dist((x, y, z), (gx, gy, gz)) < 0.01  # centered on the grasp center, lifted
    assert z > 0.0525 + 0.06
    assert robot.move_to(0.22, 0.0, 0.20) is True
    (gx, gy, gz), _ = ik.fk_grasp(robot._q_cmd)
    x, y, z, _yaw = robot._object_pose("juice_box")
    assert math.dist((x, y, z), (gx, gy, gz)) < 0.01
    scene = robot.scene()
    assert abs(scene["objects"]["juice_box"]["x"] - x) < 1e-6
    assert robot._collision_log == []
    assert validate(robot) == []


def test_close_gripper_far_from_objects_holds_nothing(robot):
    robot.reset(FAKE_SCENE)
    assert robot.move_to(0.25, 0.0, 0.15)
    assert robot.close_gripper() is False
    assert grips(robot)[-1] == {"kind": "grip", "action": "close", "expect_held": False}
    assert robot.place(0.25, 0.1) is False and robot.carry_to_hand() is False  # nothing held
    assert robot.events == []


def test_place_in_zone_releases_inside_the_tray(robot):
    robot.reset(FAKE_SCENE)
    assert robot.pick("juice_box", transit_z=TRANSIT_Z) is True
    assert robot.place_in_zone("tray", transit_z=TRANSIT_Z) is True
    after = robot.final_scene()
    SCENE_VALIDATOR.validate(after)
    assert in_tray(after, "juice_box")
    spec = {"in_zone": {"object": "juice_box", "zone": "tray"}}
    assert success.evaluate(spec, FAKE_SCENE, after, {}, robot.events) is True
    assert grips(robot)[-1] == {"kind": "grip", "action": "open", "object": "juice_box"}
    assert robot.stats["collisions"] == 0, robot._collision_log
    assert validate(robot) == []


def test_place_in_zone_takes_the_next_free_slot(robot):
    scene = copy.deepcopy(FAKE_SCENE)
    slots = scene["zones"]["tray"]["slots"]
    scene["objects"]["tea_tin"]["x"], scene["objects"]["tea_tin"]["y"] = slots[0]  # slot 1 taken
    robot.reset(scene)
    assert robot.pick("juice_box", transit_z=TRANSIT_Z) and robot.place_in_zone("tray", transit_z=TRANSIT_Z)
    after = robot.final_scene()
    box = after["objects"]["juice_box"]
    assert math.hypot(box["x"] - slots[0][0], box["y"] - slots[0][1]) > 0.03
    assert in_tray(after, "juice_box")
    assert in_tray(after, "tea_tin")


def test_place_in_tray_passes_5_of_5_perturbed_trials(robot):
    spec = load_skill("place_in_tray")["spec"]
    for trial in range(1, 6):
        scene = perturb(FAKE_SCENE, trial)
        robot.reset(scene)
        robot.set_skill("place_in_tray", {"object": "juice_box"})
        t0 = time.perf_counter()
        assert robot.run_skill("place_in_tray", {"object": "juice_box"}) is True, trial
        wall = time.perf_counter() - t0
        after = robot.final_scene()
        assert success.evaluate(spec, scene, after, {"object": "juice_box"}, robot.events), trial
        stats = robot.stats
        assert stats["collisions"] == 0, (trial, robot._collision_log)
        assert stats["limit_hits"] == 0 and stats["ik_failures"] == 0, (trial, stats)
        assert stats["sim_s"] < 60.0
        assert validate(robot) == [], trial
        assert np.allclose(robot._q_cmd, HOME_Q)
        if trial == 1:
            WALL_TIMES["place_in_tray"] = wall
            assert wall < 10.0, wall


def test_hand_over_pill_bottle_passes_its_spec_with_the_hand_unseen(robot):
    assert FAKE_SCENE["hand"]["seen"] is False
    skill = load_skill("hand_over")
    robot.reset(FAKE_SCENE)
    robot.set_skill("hand_over", {"object": "pill_bottle"})
    assert robot.run_skill("hand_over", {"object": "pill_bottle"}) is True
    after = robot.final_scene()
    SCENE_VALIDATOR.validate(after)
    assert robot.events == ["handoff_done"]
    assert success.evaluate(skill["spec"], FAKE_SCENE, after, {"object": "pill_bottle"}, robot.events)
    assert after["objects"]["pill_bottle"]["seen"] is False  # the user took it
    segments = robot.trajectory()["segments"]
    assert segments[-1] == {"kind": "live", "primitive": "carry_to_hand", "object": "pill_bottle"}
    assert [s["kind"] for s in segments].count("live") == 1
    assert robot.stats["collisions"] == 0 and robot.stats["ik_failures"] == 0
    assert np.allclose(robot._q_cmd, HOME_Q)  # the real primitive ends at home
    assert validate(robot) == []
    assert robot.pick("pill_bottle") is False  # gone


def test_hand_over_juice_box_also_passes(robot):
    skill = load_skill("hand_over")
    robot.reset(FAKE_SCENE)
    assert robot.run_skill("hand_over", {"object": "juice_box"}) is True
    assert success.evaluate(
        skill["spec"], FAKE_SCENE, robot.final_scene(), {"object": "juice_box"}, robot.events
    )
    assert robot.stats["collisions"] == 0


def test_candidate_that_drops_the_box_beside_the_tray_fails_the_spec(robot):
    # Beside the tray read from the file: on the +x side, 3 cm past the zone's edge plus the juice box's
    # half-diagonal, so no part of the box can count as inside (the old hard-coded (0.17, 0.22) is inside
    # the re-taped zone).
    tray, box = FAKE_SCENE["zones"]["tray"], FAKE_SCENE["objects"]["juice_box"]
    reach = math.hypot(box["size"][0] / 2, box["size"][1] / 2)
    x, y = tray["x"] + tray["r"] + reach + 0.03, tray["y"]
    assert math.hypot(x - tray["x"], y - tray["y"]) > tray["r"] + reach
    assert BOX["x"][0] <= x <= BOX["x"][1] and BOX["y"][0] <= y <= BOX["y"][1]
    src = (
        "import math\n\n"
        "def run(robot, args):\n"
        "    if not robot.pick('juice_box', transit_z=0.28):\n"
        "        return False\n"
        f"    return robot.place({x:.4f}, {y:.4f}, 0.07, transit_z=0.28)\n"
    )
    run = restricted.load_entry(src)
    robot.reset(FAKE_SCENE)
    assert run(robot, {}) is True  # the candidate thinks it succeeded
    after = robot.final_scene()
    spec = {"in_zone": {"object": "juice_box", "zone": "tray"}}
    assert success.evaluate(spec, FAKE_SCENE, after, {}, robot.events) is False
    assert not in_tray(after, "juice_box")
    assert robot.stats["collisions"] == 0  # the reason is the spec, not a crash
    reason = "spec not satisfied" if not robot.stats["first_collision"] else robot.stats["first_collision"]
    assert reason == "spec not satisfied"


@pytest.mark.xfail(
    reason=SHORT_OBSTACLE,
    strict=True,
)
def test_candidate_without_transit_collides_with_the_cereal_box(robot):
    src = (
        "import math\n\n"
        "def run(robot, args):\n"
        "    if not robot.pick('juice_box'):\n"
        "        return False\n"
        "    return robot.place_in_zone('tray')\n"
    )
    run = restricted.load_entry(src)
    robot.reset(FAKE_SCENE)
    run(robot, {})
    assert robot.stats["collisions"] >= 1
    assert "cereal_box" in robot.stats["first_collision"]
    assert robot.stats["first_collision"] in ("gripper hit cereal_box", "held juice_box hit cereal_box")
    assert validate(robot) == []  # a colliding path is still a valid trajectory; scoring rejects it


# ---------------------------------------------------------------- collisions, hand, time limit


def test_no_self_collision_at_home_pick_poses_and_during_place_in_tray(robot):
    robot.reset(FAKE_SCENE)
    assert robot.move_to(0.25, 0.0, 0.12)
    robot.home()
    assert robot._collision_log == []
    for name in OBJECTS:
        robot.reset(FAKE_SCENE)
        assert robot.pick(name) is True
        assert self_collisions(robot) == [], name
    robot.reset(FAKE_SCENE)
    assert robot.run_skill("place_in_tray", {"object": "juice_box"}) is True
    assert robot._collision_log == []


def test_self_contacts_would_be_counted(kit_robot):
    """The scan counts robot-robot pairs: fold the wrist back into the forearm and the phrase names links.
    Kit layout: on the shipped one the folding gripper sweeps into the 10 cm tea box at (0.15, 0.03) first,
    an environment contact that says nothing about self-contacts."""
    robot = kit_robot
    robot.reset(KIT_SCENE)
    q = np.array(HOME_Q)
    q[3] = LIMITS["wrist_flex"][1] - 3.5 * DEG
    q[2] = LIMITS["elbow_flex"][1] - 3.5 * DEG
    for _ in range(40):
        robot._run_path([q])
    if robot.stats["collisions"]:
        a, _, b = robot.stats["first_collision"].partition(" hit ")
        assert a in ROBOT_LABELS and b in ROBOT_LABELS, robot.stats["first_collision"]
    else:
        pytest.skip("this arm does not reach a self-contact at the folded pose")


def test_hand_box_is_hit_when_the_hand_is_seen(robot):
    """The hand's position comes from the scene file (A15 moves it): descend onto it from above. Objects
    within 9 cm of the hand point are parked elsewhere first (the committed layout's hand box overlaps the
    juice box's footprint, so a seen hand shoves it into the gripper's path; a layout defect, not
    SimRobot's)."""
    seen = copy.deepcopy(FAKE_SCENE)
    hand = seen["hand"]
    spots = iter(((0.12, -0.22), (0.12, 0.22), (0.36, -0.22)))
    for item in seen["objects"].values():
        if math.hypot(item["x"] - hand["x"], item["y"] - hand["y"]) < 0.09:
            item["x"], item["y"] = next(spots)
    unseen = copy.deepcopy(seen)
    seen["hand"]["seen"] = True
    robot.reset(seen)
    assert robot.scene()["hand"]["seen"] is True
    assert robot.move_to(hand["x"], hand["y"], 0.12) is True
    assert robot.stats["collisions"] == 0, robot._collision_log
    assert robot.move_to(hand["x"], hand["y"], hand["z"] + 0.01) is True  # fingertips into the hand box
    assert robot.stats["collisions"] >= 1
    assert robot.stats["first_collision"] == "gripper hit hand"
    robot.reset(unseen)  # unseen: the box is parked under the table
    assert robot.move_to(hand["x"], hand["y"], 0.12) and robot.move_to(hand["x"], hand["y"], hand["z"] + 0.01)
    assert robot.stats["collisions"] == 0, robot._collision_log


def test_carry_to_hand_with_the_hand_seen_goes_above_it(robot):
    seen = copy.deepcopy(FAKE_SCENE)
    seen["hand"].update({"seen": True, "x": 0.26, "y": -0.10, "z": 0.03})
    robot.reset(seen)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z) and robot.carry_to_hand()
    assert robot.events == ["handoff_done"]
    assert robot.stats["collisions"] == 0, robot._collision_log


@pytest.mark.parametrize("failure", ["no_plan", "incomplete_plan", "no_feedback"])
def test_failed_carry_rise_keeps_the_object_and_never_travels(robot, monkeypatch, failure):
    robot.reset(FAKE_SCENE)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z)
    trajectory = robot.trajectory()
    calls = []
    plan = robot._plan

    def faulty_plan(*args, **kwargs):
        calls.append((args, kwargs))
        if failure == "no_plan":
            return None
        if failure == "incomplete_plan":
            return [robot._q_cmd.copy()]
        return plan(*args, **kwargs)

    monkeypatch.setattr(robot, "_plan", faulty_plan)
    if failure == "no_feedback":
        monkeypatch.setattr(robot, "_run_path", lambda path: None)
    assert robot.carry_to_hand() is False
    assert len(calls) == 1 and calls[0][1]["straight"] is True
    assert robot._held == "pill_bottle" and robot.events == []
    assert robot.stats["ik_failures"] == 1
    assert robot.trajectory() == trajectory
    assert robot.final_scene()["objects"]["pill_bottle"]["seen"] is True


def test_failed_carry_travel_does_not_fall_back_to_a_direct_sweep(robot, monkeypatch):
    robot.reset(FAKE_SCENE)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z)
    plan = robot._plan
    calls = []

    def block_travel(*args, **kwargs):
        calls.append((args, kwargs))
        return plan(*args, **kwargs) if len(calls) == 1 else None

    monkeypatch.setattr(robot, "_plan", block_travel)
    assert robot.carry_to_hand() is False
    assert len(calls) == 2 and all(kwargs["straight"] for _args, kwargs in calls)
    assert robot._held == "pill_bottle" and robot.events == []
    assert robot.stats["ik_failures"] == 1
    assert all(segment["kind"] != "live" for segment in robot.trajectory()["segments"])


def test_carry_clearance_above_the_ceiling_is_refused_without_motion(robot):
    scene = copy.deepcopy(FAKE_SCENE)
    scene["obstacles"]["cereal_box"]["h"] = 0.35
    robot.reset(scene)
    assert robot.pick("pill_bottle")
    sim_s, trajectory = robot.stats["sim_s"], robot.trajectory()
    assert robot.carry_to_hand() is False
    assert robot.stats["sim_s"] == sim_s
    assert robot.trajectory() == trajectory
    assert robot._held == "pill_bottle" and robot.events == []
    assert robot.stats["ik_failures"] == 1


def test_carry_descent_requires_measured_arrival_before_release(kit_robot, monkeypatch):
    """Kit layout: the carry clears the 20 cm cereal box at 0.26 m and descends to 0.185 m above the
    handoff point. On the shipped layout the handoff target (0.19 m) is above the clearance (0.17 m): the
    rise goes straight to it and the 3rd path is a 2-waypoint descent of zero height the arm has already
    finished (measured 2026-10-18), so stalling it proves nothing."""
    robot = kit_robot
    robot.reset(KIT_SCENE)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z)
    run_path = robot._run_path
    calls = 0
    stalled_drop: list[float] = []

    def stall_descent(path):
        nonlocal calls
        calls += 1
        if calls != 3:  # rise and travel work; the descent's actuator does not follow its command
            run_path(path)
        else:
            stalled_drop.append(ik.fk_grasp(path[0])[0][2] - ik.fk_grasp(path[-1])[0][2])

    monkeypatch.setattr(robot, "_run_path", stall_descent)
    assert robot.carry_to_hand() is False
    assert calls == 3
    assert stalled_drop[0] > 0.03, stalled_drop  # the stalled leg really is the descent
    assert robot._held == "pill_bottle" and robot.events == []
    assert robot.stats["ik_failures"] == 1


@pytest.mark.parametrize("failure", ["backoff", "home"])
def test_carry_does_not_report_completion_when_return_motion_fails(robot, monkeypatch, failure):
    robot.reset(FAKE_SCENE)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z)
    if failure == "home":
        monkeypatch.setattr(robot, "_plan_home", lambda: None)
    else:
        plan = robot._plan
        calls = 0

        def block_backoff(*args, **kwargs):
            nonlocal calls
            calls += 1
            return None if calls == 4 else plan(*args, **kwargs)

        monkeypatch.setattr(robot, "_plan", block_backoff)
    assert robot.carry_to_hand() is False
    assert robot.events == [] and robot.stats["ik_failures"] == 1
    assert all(segment["kind"] != "live" for segment in robot.trajectory()["segments"])
    assert robot.final_scene()["objects"]["pill_bottle"]["seen"] is False


def test_seen_hand_outside_workspace_uses_the_handoff_zone(robot):
    scene = copy.deepcopy(FAKE_SCENE)
    scene["hand"].update(seen=True, x=0.65, y=-0.4)
    robot.reset(scene)
    assert robot.run_skill("hand_over", {"object": "pill_bottle"})
    taken = robot.final_scene()["objects"]["pill_bottle"]
    zone = scene["zones"]["handoff"]
    assert not taken["seen"]
    assert math.hypot(taken["x"] - zone["x"], taken["y"] - zone["y"]) < 0.01
    assert robot.stats["collisions"] == 0 and robot.stats["ik_failures"] == 0
    assert robot.events == ["handoff_done"]


def test_sim_time_limit_is_raised_after_60_s(robot):
    robot.reset(FAKE_SCENE)
    assert issubclass(SimTimeLimit, BaseException) and not issubclass(SimTimeLimit, Exception)
    src = (
        "import math\n\n"
        "def run(robot, args):\n"
        "    while True:\n"
        "        try:\n"
        "            robot.move_to(0.2, 0.1, 0.15)\n"
        "            robot.move_to(0.2, -0.1, 0.15)\n"
        "        except Exception:\n"
        "            pass\n"
    )
    run = restricted.load_entry(src)
    with pytest.raises(SimTimeLimit):
        run(robot, {})
    assert robot.stats["sim_s"] > 60.0
    assert robot.stats["sim_s"] < 60.0 + DT + 1e-6
    assert validate(robot) == []
    # the latch: a candidate that swallowed the first SimTimeLimit cannot keep the arm moving; every motion
    # method raises again before a tick, the clock stays put, the reads for the runner still work
    sim_s = robot.stats["sim_s"]
    for call in (
        lambda: robot.move_to(0.25, 0.0, 0.15),
        lambda: robot.set_roll(0.5),
        robot.open_gripper,
        robot.close_gripper,
        robot.home,
        lambda: robot.pick("pill_bottle"),
        lambda: robot.place(0.2, 0.0),
        lambda: robot.place_in_zone("tray"),
        robot.carry_to_hand,
        lambda: robot.run_skill("hand_over", {"object": "pill_bottle"}),
    ):
        with pytest.raises(SimTimeLimit):
            call()
    assert robot.stats["sim_s"] == sim_s
    assert robot.final_scene()["t"] == pytest.approx(FAKE_SCENE["t"] + sim_s)
    assert validate(robot) == []
    robot.reset(FAKE_SCENE)  # the runner's reset clears it
    assert robot.move_to(0.25, 0.0, 0.15) is True


# ---------------------------------------------------------------- run_skill and the starter files


def test_run_skill_goes_through_sim_restricted(robot, monkeypatch, tmp_path):
    calls = []
    real_load_entry = restricted.load_entry

    def spy(src, entry="run", filename="<candidate>"):
        calls.append((src, entry, filename))
        return real_load_entry(src, entry, filename)

    monkeypatch.setattr(sim_robot, "load_entry", spy)
    robot.reset(FAKE_SCENE)
    assert robot.run_skill("hand_over", {"object": "pill_bottle"}) is True
    assert calls == [(load_skill("hand_over")["code"], "run", "<skill hand_over>")]
    # missing, badly named, invalid and rejected skills return False with an event
    assert robot.run_skill("no_such_skill", {}) is False
    assert robot.run_skill("../hand_over", {}) is False
    assert robot.run_skill(3, {}) is False
    rejected = dict(
        load_skill("hand_over"), name="evil", code="import os\n\ndef run(robot, args):\n    return True\n"
    )
    (tmp_path / "evil.json").write_text(json.dumps(rejected), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "invalid.json").write_text(json.dumps({"name": "invalid"}), encoding="utf-8")
    other = SimRobot(FAKE_SCENE, skills_dir=tmp_path)
    assert other.run_skill("evil", {}) is False
    assert other.run_skill("broken", {}) is False
    assert other.run_skill("invalid", {}) is False
    assert len(other.events) == 3 and all(e.startswith("run_skill:") for e in other.events)
    assert "rejected" in other.events[0]


def test_run_skill_result_is_a_bool_and_reset_is_refused_inside_a_skill(tmp_path):
    base = load_skill("hand_over")
    cheat = dict(
        base, name="cheat", code="def run(robot, args):\n    robot.reset(robot.scene())\n    return 1\n"
    )
    truthy = dict(base, name="truthy", code="def run(robot, args):\n    return [1]\n")
    for skill in (cheat, truthy):
        (tmp_path / f"{skill['name']}.json").write_text(json.dumps(skill), encoding="utf-8")
    other = SimRobot(FAKE_SCENE, skills_dir=tmp_path)
    assert other.run_skill("truthy", {}) is True
    with pytest.raises(RuntimeError):
        other.run_skill("cheat", {})


def test_routine_style_nesting_runs_both_starter_skills(robot, tmp_path):
    routine = dict(
        load_skill("hand_over"),
        name="good_morning",
        code=(
            "def run(robot, args):\n"
            "    if not robot.run_skill('place_in_tray', {'object': 'juice_box'}):\n"
            "        return False\n"
            "    return robot.run_skill('hand_over', {'object': 'pill_bottle'})\n"
        ),
    )
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in ("hand_over", "place_in_tray"):
        (skills / f"{name}.json").write_text(json.dumps(load_skill(name)), encoding="utf-8")
    (skills / "good_morning.json").write_text(json.dumps(routine), encoding="utf-8")
    other = SimRobot(FAKE_SCENE, skills_dir=skills)
    assert other.run_skill("good_morning", {}) is True
    after = other.final_scene()
    assert in_tray(after, "juice_box") and after["objects"]["pill_bottle"]["seen"] is False
    assert other.events == ["handoff_done"]
    assert other.stats["collisions"] == 0, other._collision_log
    assert replay.validate(other.trajectory(), LIMITS, ik.fk_tip, BOX) == []


@pytest.mark.parametrize("name", ["hand_over", "place_in_tray"])
def test_starter_skill_files(name):
    skill = load_skill(name)
    SKILL_VALIDATOR.validate(skill)
    assert skill["name"] == name and skill["trusted"] is True
    assert skill["spec"]["name"] == name and skill["spec"]["params"] == {"object": "object"}
    assert check_source(skill["code"]) == []
    assert skill["code"].startswith("import math\n")
    assert "def run(robot, args):" in skill["code"]
    # C4 measured both starters at 5/5 with `python -m sim.run_candidate --skill ...` (A12 Done-when)
    assert skill["sim"] == {"successes": 5, "trials": 5} and skill["real"] == {"runs": 0, "successes": 0}
    created = datetime.fromisoformat(skill["created"])
    assert created.tzinfo is not None
    if name == "hand_over":
        assert skill["spec"]["success"] == {
            "all": [{"moved": {"object": "$object", "min": 0.05}}, {"event": "handoff_done"}]
        }
        assert "carry_to_hand()" in skill["code"] and "transit_z=tallest + 0.08" in skill["code"]
    else:
        assert skill["spec"]["success"] == {"in_zone": {"object": "$object", "zone": "tray"}}
        assert 'place_in_zone("tray", transit_z=tallest + 0.08)' in skill["code"]


def test_wall_clock_of_one_place_in_tray_trial_is_reported():
    wall = WALL_TIMES.get("place_in_tray")
    assert wall is not None, "runs after test_place_in_tray_passes_5_of_5_perturbed_trials"
    print(f"\nplace_in_tray trial 1 wall clock: {wall:.2f} s")
    assert wall < 10.0


# ---------------------------------------------------------------- round-2 regressions


def test_tip_speed_cap_holds_on_straight_legs_too(robot):
    """Pick and place descents, transit rises and the home line profile the grasp center on a straight line;
    the tip, 2.5 cm away, swings further when the pitch changes along the leg (round 1 recorded 0.20 to
    0.24 m/s there). Every recorded step must stay under the cap."""
    robot.reset(FAKE_SCENE)
    assert robot.run_skill("place_in_tray", {"object": "juice_box"}) is True
    assert robot.pick("tea_tin", approach=0.05, transit_z=0.25)  # without the retime pass: 0.216 m/s
    assert robot.place(0.20, -0.18, 0.06, approach=0.05, transit_z=0.25)
    assert robot.run_skill("hand_over", {"object": "pill_bottle"}) is True
    assert robot.move_to(0.12, 0.20, 0.25) and robot.move_to(0.36, -0.20, 0.03)
    robot.home()
    assert max_tip_speed(robot) <= 0.2 + 1e-6
    assert validate(robot) == []


def test_place_with_the_default_release_height_keeps_the_juice_box_off_the_table(kit_robot):
    """The contract's default release_z = 0.05 is below the 10.5 cm juice box's half height: the release is
    raised so the held box stops 5 mm above the table instead of being pushed into it (a collision). Kit
    layout: the shipped 8.9 cm power bank's half height is under 0.05 m, and its 4.45 cm pill bottle stands
    in the gripper's way at (0.20, -0.18)."""
    robot = kit_robot
    robot.reset(KIT_SCENE)
    half = robot._object_half_height("juice_box")
    assert half == pytest.approx(0.0525) and half > 0.05  # the premise: default release_z is too low
    assert robot.pick("juice_box", transit_z=TRANSIT_Z) is True
    assert robot.place(0.20, -0.18, transit_z=TRANSIT_Z) is True
    x, y, z, _yaw = robot._object_pose("juice_box")
    assert robot.stats["collisions"] == 0, robot._collision_log
    assert math.hypot(x - 0.20, y + 0.18) < 0.02 and abs(z - 0.0525) < 0.003  # upright where it was asked
    assert validate(robot) == []
    robot.reset(KIT_SCENE)  # a short object at a low release height is unaffected
    assert robot.pick("tea_tin", transit_z=TRANSIT_Z) and robot.place(0.20, -0.18, 0.055, transit_z=TRANSIT_Z)
    assert robot.stats["collisions"] == 0, robot._collision_log


def test_carry_to_hand_rises_before_the_carry(robot):
    """Round 1 swept the held object straight from the post-pick pose: the tea tin, held 8 cm up, swung into
    the 10.5 cm juice box on its way to the handoff zone, and with the hand seen low behind the juice box the
    pill bottle went through it. The carry now rises to a clearance height, travels, then descends."""
    skill = load_skill("hand_over")
    for trial in range(1, 6):
        scene = perturb(FAKE_SCENE, trial)
        robot.reset(scene)
        assert robot.run_skill("hand_over", {"object": "tea_tin"}) is True, trial
        after = robot.final_scene()
        assert success.evaluate(skill["spec"], scene, after, {"object": "tea_tin"}, robot.events)
        assert robot.stats["collisions"] == 0, (trial, robot._collision_log)
        assert validate(robot) == []
    seen = copy.deepcopy(FAKE_SCENE)
    seen["hand"]["seen"] = True  # at the handoff spot
    robot.reset(seen)
    assert robot.run_skill("hand_over", {"object": "pill_bottle"}) is True
    assert robot.stats["collisions"] == 0, robot._collision_log
    low = copy.deepcopy(FAKE_SCENE)  # round 1's layout: the hand seen behind the juice box
    low["obstacles"]["cereal_box"].update({"x": 0.24, "y": 0.11})
    low["objects"]["tea_tin"].update({"x": 0.15, "y": 0.03})
    low["hand"].update({"seen": True, "x": 0.30, "y": 0.02, "z": 0.03})
    robot.reset(low)
    assert robot.pick("pill_bottle", transit_z=TRANSIT_Z) and robot.carry_to_hand()
    assert robot.stats["collisions"] == 0, robot._collision_log
    assert robot.events == ["handoff_done"] and np.allclose(robot._q_cmd, HOME_Q)


def test_ai_code_cannot_reset_relabel_or_shadow_the_robot(robot):
    """The cheat vectors of round 1: reset(scene_with_the_box_in_the_tray), set_skill to relabel the
    trajectory, shadowing trajectory()/final_scene() on the instance, new attributes. Refused inside
    run_entry and run_skill, and whenever AI code is on the stack (a runner calling run() directly)."""
    robot.reset(FAKE_SCENE)
    reset = restricted.load_entry("def run(robot, args):\n    robot.reset(robot.scene())\n    return True\n")
    relabel = restricted.load_entry(
        "def run(robot, args):\n    robot.set_skill('hand_over', {})\n    return 1\n"
    )
    shadow = restricted.load_entry(
        "def run(robot, args):\n    robot.final_scene = robot.scene\n    return True\n"
    )
    attach = restricted.load_entry("def run(robot, args):\n    robot.pwned = 1\n    return True\n")
    unset = restricted.load_entry("def run(robot, args):\n    del robot.trajectory\n    return True\n")
    for fn in (reset, relabel):
        with pytest.raises(RuntimeError):
            robot.run_entry(fn, {})
        with pytest.raises(RuntimeError):
            fn(robot, {})
    for fn in (shadow, attach, unset):
        with pytest.raises(AttributeError):
            robot.run_entry(fn, {})
    assert robot._depth == 0
    assert robot.stats["sim_s"] == 0.0 and "final_scene" not in vars(robot) and "pwned" not in vars(robot)
    # run_entry itself: bool result, exceptions propagate, the runner keeps reset and set_skill
    good = restricted.load_entry("def run(robot, args):\n    return robot.pick(args['object'])\n")
    assert robot.run_entry(good, {"object": "pill_bottle"}) is True
    boom = restricted.load_entry("def run(robot, args):\n    return 1 / 0\n")
    with pytest.raises(ZeroDivisionError):
        robot.run_entry(boom, {})
    with pytest.raises(TypeError):
        robot.run_entry("run", {})
    assert robot._depth == 0
    robot.set_skill("probe", {})
    robot.reset(FAKE_SCENE)
    assert robot.stats["sim_s"] == 0.0


def test_attach_tolerance_is_one_centimeter_from_the_center_line(kit_robot):
    """close_gripper attaches at 0.8 cm from the pill bottle's center line, not at 2.5 cm (the bottle then
    sits under the moving jaw and is shoved) nor 3 cm beside the fingers (nothing is touched), nor at 85 %
    of the juice box's height. Kit layout: the offsets are sized for the kit's 3.5 cm bottle; the shipped
    4.45 cm one touches a finger 3 cm to the side."""
    robot = kit_robot
    robot.reset(KIT_SCENE)
    assert robot._object_width("pill_bottle") == pytest.approx(0.035)  # the premise
    descend_to(robot, "pill_bottle", 0.008, 0.0, 0.5)
    assert robot.close_gripper() is True
    close = {"kind": "grip", "action": "close", "object": "pill_bottle", "expect_held": True}
    assert grips(robot)[-1] == close
    assert robot.move_to(0.22, -0.10, 0.18) and robot._object_pose("pill_bottle")[2] > 0.10  # it came along
    for along_c, across in ((0.025, 0.0), (0.0, -0.03), (-0.025, 0.0)):
        robot.reset(KIT_SCENE)
        descend_to(robot, "pill_bottle", along_c, across, 0.5)
        assert robot.close_gripper() is False, (along_c, across)
        assert grips(robot)[-1] == {"kind": "grip", "action": "close", "expect_held": False}
        assert robot._held is None
        if across:  # beside the fingers: untouched, and left behind when the gripper rises
            assert robot.stats["collisions"] == 0, robot._collision_log
            assert robot.move_to(0.22, -0.10, 0.18) and robot._object_pose("pill_bottle")[2] < 0.05
        else:  # under a finger: shoved (a counted contact), never attached
            assert robot.stats["collisions"] >= 1 and "pill_bottle" in robot.stats["first_collision"]
    robot.reset(KIT_SCENE)
    descend_to(robot, "juice_box", 0.0, 0.0, 0.85)
    assert robot.close_gripper() is False
    assert grips(robot)[-1] == {"kind": "grip", "action": "close", "expect_held": False}


def test_input_edge_cases_are_refused_or_sane(robot):
    """place_in_zone with a slot that is not an index (inf, nan, negative, past the end, a list) returns False
    without motion; move_to without a usable speed runs at the contract's 0.15 m/s; a target on the face
    of the workspace box is reached (the IK solution's tip used to land 1e-6 outside and count a failure)."""
    robot.reset(FAKE_SCENE)
    assert robot.pick("juice_box", transit_z=TRANSIT_Z)
    sim_s = robot.stats["sim_s"]
    for slot in (float("inf"), float("nan"), -1, 4, [0], "last"):
        assert robot.place_in_zone("tray", slot=slot, transit_z=TRANSIT_Z) is False, slot
    assert robot.stats["sim_s"] == sim_s
    assert robot.place_in_zone("tray", slot=2, transit_z=TRANSIT_Z) is True  # slot 4 at z 0.28: out of reach
    for speed in (None, "fast", [0.2]):
        robot.reset(FAKE_SCENE)
        assert robot.move_to(0.30, -0.15, 0.08, speed=speed) is True
        assert 0.1 < max_tip_speed(robot) <= 0.15 + 1e-6, speed
    faces = ((0.25, 0.0, 0.02), (0.20, 0.0, 0.30), (0.10, 0.0, 0.25), (0.40, 0.0, 0.10), (0.25, 0.25, 0.10))
    for point in faces:
        robot.reset(FAKE_SCENE)
        assert robot.move_to(*point) is True, point
        assert math.dist(ik.fk_tip(robot._q_cmd), point) < 0.005
        assert robot.stats["ik_failures"] == 0
        assert validate(robot) == []
    robot.reset(FAKE_SCENE)
    assert robot.pick("pill_bottle", transit_z=0.30) is True  # the transit at the ceiling of the box
    assert robot.stats["ik_failures"] == 0 and validate(robot) == []


# ------------------------------------------------------------- 9.9 pins: pick while holding, yaw, depth, tray


def test_pick_while_holding_reads_the_pose_after_the_drop():
    """pick() drops what it holds first; the target's pose is read after the drop has landed (it used to be
    read before, so picking the held object again descended onto its old spot and hit it)."""
    for first, second in (("juice_box", "juice_box"), ("juice_box", "pill_bottle")):
        r = SimRobot(FAKE_SCENE)
        assert r.pick(first) is True
        assert r.pick(second) is True, (first, second, r._collision_log)
        assert r._held == second and r.stats["collisions"] == 0, r._collision_log
        assert validate(r) == []


def descend_misaligned(robot: SimRobot, name: str, yaw_offset: float) -> float:
    """Like descend_to at half height, but with the jaws turned `yaw_offset` off the object's width; returns
    the measured angle (degrees) between the closing direction and the width."""
    x, y, _z, yaw = robot._object_pose(name)
    gz = robot._object_half_height(name)
    robot.open_gripper()
    q_above, ok = robot._goal("grasp", (x, y, gz + 0.08), -math.pi / 2, robot._q_cmd)
    assert ok
    roll = robot._roll_for(yaw + yaw_offset, float(q_above[0]))
    assert robot._move("grasp", (x, y, gz + 0.08), -math.pi / 2, 0.2, roll=roll)
    assert robot._move("grasp", (x, y, gz), -math.pi / 2, 0.2, straight=True)
    _a, c = ik.jaw_axes(robot._q_cmd)
    diff = (math.atan2(c[1], c[0]) - (yaw + math.pi / 2)) % math.pi
    return math.degrees(min(diff, math.pi - diff))


def test_box_attaches_only_with_the_jaws_within_20_degrees_of_its_width(robot):
    assert sim_robot.ATTACH_YAW == pytest.approx(math.radians(20.0))
    robot.reset(FAKE_SCENE)
    off = descend_misaligned(robot, "juice_box", math.radians(10.0))
    assert off < 20.0 and robot.close_gripper() is True, off
    robot.reset(FAKE_SCENE)
    off = descend_misaligned(robot, "juice_box", math.radians(26.0))
    assert off > 20.0 and robot.close_gripper() is False, off
    assert robot._held is None


def test_a_self_recursive_skill_stops_at_the_nesting_cap(tmp_path):
    assert sim_robot.MAX_SKILL_DEPTH == 8
    loop = dict(
        load_skill("hand_over"),
        name="loop",
        code="def run(robot, args):\n    return robot.run_skill('loop', {})\n",
    )
    (tmp_path / "loop.json").write_text(json.dumps(loop), encoding="utf-8")
    other = SimRobot(FAKE_SCENE, skills_dir=tmp_path)
    assert other.run_skill("loop", {}) is False  # not a RecursionError
    assert other.events == ["run_skill: loop nested too deep"]


def test_tray_contacts_are_named_after_the_tray(kit_model):
    """Body-name labels for the tray walls: the empty gripper and a held object dragged out through the -x
    wall are counted as 'gripper hit tray' and 'held juice_box hit tray'. Kit layout: its tray has 3.5 cm
    walls; the shipped zone is taped (2 mm) and its -x side, 10 cm in, is outside the workspace box."""
    far = copy.deepcopy(KIT_SCENE)
    far["objects"]["juice_box"].update({"x": 0.12, "y": -0.23})
    far["objects"]["tea_tin"].update({"x": 0.12, "y": 0.23})
    tray = far["zones"]["tray"]
    r = SimRobot(far, model_path=kit_model)
    assert r.move_to(tray["x"], tray["y"], 0.12) and r.move_to(tray["x"], tray["y"], 0.03)
    assert r._collision_log == []
    assert r.move_to(tray["x"] - 0.10, tray["y"], 0.03)
    assert r._collision_log == ["gripper hit tray"]
    r = SimRobot(far, model_path=kit_model)
    assert r.pick("juice_box", transit_z=TRANSIT_Z)
    x, y, _z, _yaw = r._object_pose("juice_box")
    assert (
        r.move_to(x, y, 0.28)
        and r.move_to(tray["x"], tray["y"], 0.28)
        and r.move_to(tray["x"], tray["y"], 0.08)
    )
    assert r._collision_log == []
    assert r.move_to(tray["x"] - 0.10, tray["y"], 0.08)
    assert r._collision_log == ["held juice_box hit tray"]
