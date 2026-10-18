"""Checks on sim/model (A5): the two MJCF files load, the frame and tip site are right, the gripper's
collision boxes follow the mesh, the grasp center clears the fingers, gravity holds the arm at home_q, and
scene_ours.xml agrees with build_scene.py."""

from __future__ import annotations

import math
import pathlib
import uuid
import warnings

import mujoco
import numpy as np
import pytest
import yaml
from known_limits import FIXED_FINGER_CLEARANCE

from arm import ik
from sim.model import build_scene
from sim.model.fit_gripper import GRASP_FROM_FIXED_FACE, inner_face_x, mesh_vertices_in_body, tip_vertices

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "sim" / "model"
ARM_ONLY = MODEL_DIR / "arm_only.xml"
SCENE_XML = MODEL_DIR / "scene_ours.xml"
SCENE_CFG = yaml.safe_load((ROOT / "config" / "scene.yaml").read_text(encoding="utf-8"))
OBJECTS = yaml.safe_load((ROOT / "config" / "objects.yaml").read_text(encoding="utf-8"))
HOME_Q = [float(v) for v in SCENE_CFG["home_q"]]
BOX = SCENE_CFG["workspace"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
DEG = math.pi / 180
OUT = np.array([0.0, 0.0, -1.0])  # jaw direction in the gripper body frame
GRIPPER_OPEN = 1.0  # rad, our gripper value 1.0 (fully open) in the sim
PICK_XY = (0.18, -0.12)  # a spot every object can be picked at straight down
# The kit's pill bottle (config/objects.yaml before the 2026-10-18 props): a fixed control object short and
# narrow enough to clear the jaws at 0.6 rad, so the 0.6 rad contacts of the shipped props are their size.
KIT_PILL = {
    "name": "kit_pill_bottle",
    "kind": "control",
    "shape": "cylinder",
    "size": [0.035, 0.035],
    "h": 0.07,
}


def name2id(m: mujoco.MjModel, kind: str, name: str) -> int:
    obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
    i = mujoco.mj_name2id(m, obj, name)
    assert i >= 0, f"{kind} {name!r} missing"
    return i


def geom_name(m: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def set_arm(m: mujoco.MjModel, d: mujoco.MjData, q: list[float], gripper: float = 0.0) -> None:
    for name, v in zip(JOINTS, q, strict=True):
        d.qpos[m.jnt_qposadr[name2id(m, "joint", name)]] = v
    d.qpos[m.jnt_qposadr[name2id(m, "joint", "gripper")]] = gripper


def moving_jaw_in_gripper_frame(m: mujoco.MjModel, gripper_angle: float) -> np.ndarray:
    """Moving-jaw mesh vertices in the gripper body frame for a given gripper joint angle."""
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[name2id(m, "joint", "gripper")]] = gripper_angle
    mujoco.mj_kinematics(m, d)
    g, j = name2id(m, "body", "gripper"), name2id(m, "body", "moving_jaw_so101_v1")
    rg, rj = d.xmat[g].reshape(3, 3), d.xmat[j].reshape(3, 3)
    v = mesh_vertices_in_body(m, "moving_jaw_so101_v1", "moving_jaw_so101_v1")
    return (rg.T @ (d.xpos[j] - d.xpos[g])) + v @ (rg.T @ rj).T


def robot_contacts(m: mujoco.MjModel, d: mujoco.MjData, gid: int) -> list[str]:
    """Names of the robot geoms in contact with geom `gid` after the last mj_forward."""
    robot = {name2id(m, "body", n) for n in ROBOT_BODIES}
    hits = []
    for i in range(d.ncon):
        c = d.contact[i]
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            if m.geom_bodyid[other] in robot:
                hits.append(geom_name(m, other))
    return sorted(hits)


def structure(m: mujoco.MjModel) -> dict:
    """Everything arm/sim_robot.py relies on in scene_ours.xml, nothing cosmetic (poses, colors, cameras)."""

    def bname(i: int) -> str:
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    return {
        "bodies": [
            (bname(b), int(m.body_mocapid[b] >= 0), round(float(m.body_mass[b]), 6)) for b in range(m.nbody)
        ],
        "joints": [
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_type[j]), bname(m.jnt_bodyid[j]))
            for j in range(m.njnt)
        ],
        "geoms": [
            (
                geom_name(m, g),
                bname(m.geom_bodyid[g]),
                int(m.geom_type[g]),
                tuple(round(float(v), 6) for v in m.geom_size[g]),
                int(m.geom_contype[g]),
                int(m.geom_conaffinity[g]),
                # tray walls and floor keep their height and offsets; other geoms sit at their body origin
                tuple(round(float(v), 6) for v in m.geom_pos[g])
                if bname(m.geom_bodyid[g]) == "tray"
                else None,
            )
            for g in range(m.ngeom)
        ],
        "actuators": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)],
        "sites": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)],
    }


@pytest.fixture(scope="module")
def arm() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ARM_ONLY))


@pytest.fixture(scope="module")
def scene() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def test_both_models_load_with_the_named_parts(arm, scene):
    for m in (arm, scene):
        for j in JOINTS + ["gripper"]:
            name2id(m, "joint", j)
            name2id(m, "actuator", j)
        name2id(m, "site", "tip")
        name2id(m, "site", "grasp")
        for g in ("gripper_housing", "finger_fixed", "finger_moving"):
            name2id(m, "geom", g)
    name2id(arm, "geom", "floor")
    name2id(arm, "camera", "overhead")


def test_scene_xml_is_what_build_scene_generates(scene):
    """The committed scene_ours.xml must match a fresh render in everything SimRobot relies on. A
    difference only in the initial poses or the camera (fake_scene.json, scene.yaml camera) is cosmetic,
    because SimRobot.reset writes every pose: it only warns."""
    expected = build_scene.render(*build_scene.load_inputs())
    if SCENE_XML.read_text(encoding="utf-8") == expected:
        return
    tmp = MODEL_DIR / f"_scene_check_{uuid.uuid4().hex}.xml"  # beside the include and the meshes
    try:
        tmp.write_text(expected, encoding="utf-8", newline="\n")
        fresh = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    assert structure(scene) == structure(fresh), (
        "scene_ours.xml is stale in names, sizes or collision classes: run python -m sim.model.build_scene"
    )
    warnings.warn(
        "scene_ours.xml differs from a fresh render only in poses or camera (cosmetic): "
        "run python -m sim.model.build_scene to refresh the viewer layout",
        stacklevel=1,
    )


def test_pan_axis_passes_through_the_origin(arm):
    d = mujoco.MjData(arm)
    mujoco.mj_forward(arm, d)
    j = name2id(arm, "joint", "shoulder_pan")
    assert np.allclose(d.xanchor[j][:2], 0.0, atol=1e-6), d.xanchor[j]
    assert abs(abs(d.xaxis[j][2]) - 1.0) < 1e-9, d.xaxis[j]


def test_scene_bodies_geoms_and_collision_classes(scene):
    m = scene
    objects = [e["name"] for e in OBJECTS.values() if e["kind"] == "object"]
    obstacles = [e["name"] for e in OBJECTS.values() if e["kind"] == "obstacle"]
    for name in objects:
        b = name2id(m, "body", name)
        j = name2id(m, "joint", name)
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == b
        g = name2id(m, "geom", name)
        assert m.geom_bodyid[g] == b and (m.geom_contype[g], m.geom_conaffinity[g]) == (1, 3)
        assert m.body_mass[b] == pytest.approx(0.05)
    for name in obstacles + ["hand", "tray"]:
        b = name2id(m, "body", name)
        assert m.body_mocapid[b] >= 0, f"{name} must be a mocap body"
        g = name2id(m, "geom", name)
        assert m.geom_bodyid[g] == b and (m.geom_contype[g], m.geom_conaffinity[g]) == (1, 3)
    for wall in ("px", "nx", "py", "ny"):
        g = name2id(m, "geom", f"tray_wall_{wall}")
        assert m.geom_bodyid[g] == name2id(m, "body", "tray")
        top = m.geom_pos[g][2] + m.geom_size[g][2]
        assert top == pytest.approx(SCENE_CFG["zones"]["tray"]["wall"])
    hand = name2id(m, "geom", "hand")
    assert np.allclose(m.geom_size[hand], [0.04, 0.05, 0.02])
    assert m.body_pos[name2id(m, "body", "hand")][2] == pytest.approx(-1.0)  # fake scene: hand not seen
    table = name2id(m, "geom", "table")
    assert m.geom_pos[table][2] + m.geom_size[table][2] == pytest.approx(0.0)
    assert (m.geom_contype[table], m.geom_conaffinity[table]) == (1, 3)
    # every robot collision geom is contype 1, conaffinity 1
    robot_bodies = {name2id(m, "body", n) for n in ROBOT_BODIES}
    robot_collision = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and m.geom_contype[g]]
    assert robot_collision, "the robot has no collision geoms"
    assert all((m.geom_contype[g], m.geom_conaffinity[g]) == (1, 1) for g in robot_collision)
    assert not any(
        m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        for g in robot_collision
        if m.geom_bodyid[g] == name2id(m, "body", "moving_jaw_so101_v1")
    )


def test_tip_site_is_the_closed_jaw_midpoint(arm):
    """The tip is within 5 mm of the midpoint of the two jaw tips with the gripper at its closed limit."""
    m = arm
    fixed = mesh_vertices_in_body(m, "gripper", "wrist_roll_follower_so101_v1")
    fixed_tip = tip_vertices(fixed, OUT).mean(0)
    closed = m.jnt_range[name2id(m, "joint", "gripper")][0]
    moving_tip = tip_vertices(moving_jaw_in_gripper_frame(m, closed), OUT).mean(0)
    midpoint = (fixed_tip + moving_tip) / 2
    tip = m.site_pos[name2id(m, "site", "tip")]
    assert np.linalg.norm(tip - midpoint) < 0.005, (tip, midpoint)
    # the closing direction (site z axis) points from the fixed jaw tip toward the moving jaw tip
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = name2id(m, "body", "gripper")
    c_body = d.xmat[g].reshape(3, 3).T @ d.site_xmat[name2id(m, "site", "tip")].reshape(3, 3)[:, 2]
    assert np.dot(moving_tip - fixed_tip, c_body) > 0


def test_tip_x_axis_points_out_of_the_jaws(arm):
    """At any pose the tip x axis runs from the wrist_roll axis toward the jaw tips (along the roll axis)."""
    m = arm
    d = mujoco.MjData(m)
    for q in ([0.0] * 5, HOME_Q, [0.4, 0.3, -0.6, 1.2, 0.8]):
        set_arm(m, d, q)
        mujoco.mj_forward(m, d)
        tip = name2id(m, "site", "tip")
        a = d.site_xmat[tip].reshape(3, 3)[:, 0]
        roll = name2id(m, "joint", "wrist_roll")
        toward_tips = d.site_xpos[tip] - d.xanchor[roll]
        toward_tips /= np.linalg.norm(toward_tips)
        assert np.dot(a, toward_tips) > 0.99, (a, toward_tips)
        assert abs(abs(np.dot(a, d.xaxis[roll])) - 1.0) < 1e-6  # parallel to the roll axis


def test_finger_boxes_follow_the_mesh(arm):
    """Each finger box ends at the mesh tip and its inner face lies on the mesh's inner face (2.5 mm)."""
    m = arm
    cases = [
        ("finger_fixed", mesh_vertices_in_body(m, "gripper", "wrist_roll_follower_so101_v1"), OUT),
        (
            "finger_moving",
            mesh_vertices_in_body(m, "moving_jaw_so101_v1", "moving_jaw_so101_v1"),
            np.array([0.0, -1.0, 0.0]),
        ),
    ]
    for name, verts, axis in cases:
        g = name2id(m, "geom", name)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        rot = rot.reshape(3, 3)
        half = m.geom_size[g]
        local = (verts - m.geom_pos[g]) @ rot  # mesh vertices in the box frame
        finger = local[local[:, 2] > -half[2] + 0.004]  # the part of the mesh along the box's length
        # the box's far end (local +z) is the fingertip
        assert abs(finger[:, 2].max() - half[2]) < 0.0025, (name, finger[:, 2].max(), half[2])
        # the inner face (local +x) touches the mesh's innermost points, at the tip and at the root
        tip_slice = finger[finger[:, 2] > half[2] - 0.012]
        root_slice = finger[finger[:, 2] < -half[2] + 0.012]
        for part in (tip_slice, root_slice):
            assert abs(part[:, 0].max() - half[0]) < 0.0025, (name, part[:, 0].max(), half[0])
        # and nothing of the mesh sticks out of the box on the inner side
        assert finger[:, 0].max() < half[0] + 0.0025
        # sanity: the mesh really extends along `axis` in its body frame
        assert (verts @ axis).max() > 0.08


@pytest.mark.xfail(reason=FIXED_FINGER_CLEARANCE, strict=True)
def test_grasp_site_sits_a_clearance_from_the_fixed_finger(arm):
    """`grasp` is GRASP_FROM_FIXED_FACE from the fixed finger box's inner face at the tip level, along the
    closing direction, and arm.ik.grasp_offset() reports its distance from `tip`."""
    m = arm
    g = name2id(m, "geom", "finger_fixed")
    tip = m.site_pos[name2id(m, "site", "tip")]
    grasp = m.site_pos[name2id(m, "site", "grasp")]
    face_x = inner_face_x(m.geom_pos[g], m.geom_quat[g], m.geom_size[g], tip[1], tip[2])
    assert grasp[0] - face_x == pytest.approx(GRASP_FROM_FIXED_FACE, abs=0.0005)
    assert 0.0 < tip[0] - face_x < 0.002  # the closed-jaw midpoint is a hair into the gap from the fixed face
    assert grasp[1] == pytest.approx(tip[1], abs=1e-6) and grasp[2] == pytest.approx(tip[2], abs=1e-6)
    assert ik.grasp_offset() == pytest.approx(grasp[0] - tip[0], abs=1e-6)
    # the widest object in objects.yaml keeps 5 mm (the IK tolerance) from the fixed finger
    widest = max(float(e["size"][1]) for e in OBJECTS.values() if e["kind"] == "object")
    assert GRASP_FROM_FIXED_FACE - widest / 2 >= 0.005 - 1e-9, widest


@pytest.fixture(scope="module")
def probe_model():
    """arm_only.xml plus mocap probes that collide with the robot: a cylinder (radius 2 cm, 3 cm long), one
    body per object of config/objects.yaml with that object's geom and one for KIT_PILL, all parked far
    below the floor."""
    spec = mujoco.MjSpec.from_file(str(ARM_ONLY))
    body = spec.worldbody.add_body(name="probe", mocap=True, pos=[0, 0, -5])
    body.add_geom(
        name="probe", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.015, 0.0], contype=1, conaffinity=1
    )
    entries = [e for e in OBJECTS.values() if e["kind"] == "object"] + [KIT_PILL]
    for i, entry in enumerate(entries):
        name = f"obj_{entry['name']}"
        b = spec.worldbody.add_body(name=name, mocap=True, pos=[0, 0, -6 - i])
        length, width = (float(v) for v in entry["size"])
        h = float(entry["h"])
        if entry["shape"] == "cylinder":
            b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[width / 2, h / 2, 0.0])
        else:
            b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2, width / 2, h / 2])
        b.geoms[0].contype = 1
        b.geoms[0].conaffinity = 1
    return spec.compile()


def place_mocap(
    m: mujoco.MjModel, d: mujoco.MjData, body: str, center: np.ndarray, frame: np.ndarray
) -> None:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, frame.reshape(-1))
    k = m.body_mocapid[name2id(m, "body", body)]
    d.mocap_pos[k] = center
    d.mocap_quat[k] = quat


def probe_contacts(
    m: mujoco.MjModel,
    q: list[float],
    gripper: float,
    center: np.ndarray,
    axis: np.ndarray,
    closing: np.ndarray,
) -> list[str]:
    """Names of the robot geoms the cylinder probe touches, its axis along `axis`, centered at `center`."""
    d = mujoco.MjData(m)
    set_arm(m, d, q, gripper)
    frame = np.stack([closing, np.cross(axis, closing), axis], axis=1)  # cylinder z along the jaw axis
    place_mocap(m, d, "probe", center, frame)
    mujoco.mj_forward(m, d)
    return robot_contacts(m, d, name2id(m, "geom", "probe"))


def gripper_pose(arm: mujoco.MjModel, q: list[float], gripper: float) -> tuple[np.ndarray, np.ndarray, ...]:
    """World positions of `tip` and `grasp` and the unit jaw / closing directions at q."""
    d = mujoco.MjData(arm)
    set_arm(arm, d, q, gripper)
    mujoco.mj_forward(arm, d)
    tip = d.site_xpos[name2id(arm, "site", "tip")].copy()
    grasp = d.site_xpos[name2id(arm, "site", "grasp")].copy()
    a, c = (np.array(v) for v in ik.jaw_axes(q))
    return tip, grasp, a, c


def test_open_fingers_leave_the_grasp_gap_empty(arm, probe_model):
    """With the gripper at 0.6 rad a 4 cm cylinder on the grasp center line touches nothing, 3 cm sideways
    either way it hits a finger. The SO-101 jaw is asymmetric (only the moving jaw swings), so the gap is
    centered on `grasp`, not on `tip`: the same cylinder on the tip line sits inside the fixed finger."""
    m = probe_model
    q, ok = ik.solve_grasp((*PICK_XY, 0.06), -math.pi / 2)
    assert ok
    tip, grasp, a, c = gripper_pose(arm, q, 0.6)
    assert a[2] < -0.98  # pointing down
    assert np.linalg.norm(grasp - tip - ik.grasp_offset() * c) < 1e-6
    assert probe_contacts(m, q, 0.6, grasp, a, c) == []
    for side in (+1, -1):
        hits = probe_contacts(m, q, 0.6, grasp + side * 0.03 * c, a, c)
        assert hits and all(h.startswith("finger") for h in hits), (side, hits)
    assert "finger_fixed" in probe_contacts(m, q, 0.6, tip, a, c)


def test_objects_centered_on_grasp_clear_the_open_gripper(probe_model):
    """The pick recipe: grasp center on the object's center line, tip at the object's mid-height, gripper
    fully open (1.0 rad): no object of config/objects.yaml touches any robot geom. At 0.6 rad every prop of
    2026-10-18 (8.3 to 10.2 cm tall, up to 4.45 cm wide) hits the moving finger and only it, which is why
    SimRobot opens fully before descending; the kit's 3.5 x 7 cm pill bottle (KIT_PILL), the control, still
    clears at 0.6 rad, so those contacts come from the props' size, not from a probe that always touches."""
    m = probe_model

    def contacts(entry: dict, opening: float) -> list[str]:
        name = entry["name"]
        h = float(entry["h"])
        center = np.array([*PICK_XY, max(h / 2, 0.02)])
        q, ok = ik.solve_grasp(center, -math.pi / 2)
        assert ok, name
        got, pitch = ik.fk_grasp(q)
        assert math.dist(got, center) < 0.005 and abs(pitch + math.pi / 2) < 3 * DEG, name
        _, c = (np.array(v) for v in ik.jaw_axes(q))
        y_axis = np.array([c[0], c[1], 0.0])
        y_axis /= np.linalg.norm(y_axis)  # the object's width runs along the closing direction
        frame = np.stack([np.cross(y_axis, [0, 0, 1.0]), y_axis, [0, 0, 1.0]], axis=1)
        d = mujoco.MjData(m)
        set_arm(m, d, q, opening)
        place_mocap(m, d, f"obj_{name}", center, frame)
        mujoco.mj_forward(m, d)
        return robot_contacts(m, d, name2id(m, "geom", f"obj_{name}"))

    hits_at_06 = {}
    for entry in (e for e in OBJECTS.values() if e["kind"] == "object"):
        hits = contacts(entry, GRIPPER_OPEN)
        assert hits == [], (entry["name"], hits)
        hits_at_06[entry["name"]] = contacts(entry, 0.6)
    assert set(hits_at_06) == {"pill_bottle", "juice_box", "tea_tin"}
    for name, hits in hits_at_06.items():
        assert set(hits) == {"finger_moving"}, (name, hits_at_06)  # never the fixed finger
    assert contacts(KIT_PILL, GRIPPER_OPEN) == [] and contacts(KIT_PILL, 0.6) == []


def test_gravity_holds_the_arm_at_home(arm):
    m = arm
    d = mujoco.MjData(m)
    set_arm(m, d, HOME_Q)
    for name, v in zip(JOINTS, HOME_Q, strict=True):
        d.ctrl[name2id(m, "actuator", name)] = v
    d.ctrl[name2id(m, "actuator", "gripper")] = 0.0
    steps = int(round(2.0 / m.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(m, d)
    drift = [
        abs(d.qpos[m.jnt_qposadr[name2id(m, "joint", n)]] - v) for n, v in zip(JOINTS, HOME_Q, strict=True)
    ]
    assert max(drift) < 2 * DEG, [round(x / DEG, 3) for x in drift]


def test_scene_objects_rest_on_the_table(scene):
    m = scene
    d = mujoco.MjData(m)
    set_arm(m, d, HOME_Q)
    for name, v in zip(JOINTS, HOME_Q, strict=True):
        d.ctrl[name2id(m, "actuator", name)] = v
    start = {}
    for e in OBJECTS.values():
        if e["kind"] == "object":
            adr = m.jnt_qposadr[name2id(m, "joint", e["name"])]
            start[e["name"]] = (adr, d.qpos[adr : adr + 3].copy(), e["h"])
    for _ in range(int(round(1.0 / m.opt.timestep))):
        mujoco.mj_step(m, d)
    for name, (adr, p0, h) in start.items():
        p = d.qpos[adr : adr + 3]
        assert np.linalg.norm(p[:2] - p0[:2]) < 0.002, (name, p0, p)
        assert abs(p[2] - h / 2) < 0.003, (name, p[2], h / 2)


def test_fk_home_is_inside_the_workspace():
    (x, y, z), pitch = ik.fk(HOME_Q)
    assert (
        BOX["x"][0] <= x <= BOX["x"][1]
        and BOX["y"][0] <= y <= BOX["y"][1]
        and BOX["z"][0] <= z <= BOX["z"][1]
    )
    assert math.dist((x, y, z), (0.111, 0.0, 0.274)) < 0.01  # the brief's number
    assert abs(pitch - math.radians(-7)) < 2 * DEG
