"""Generate sim/model/scene_ours.xml: the SO-101 plus table, tray, objects, obstacles and the hand box.

Run `python -m sim.model.build_scene` (from the repo root) after editing config/objects.yaml,
config/scene.yaml or config/fake_scene.json, and keep the generated file in the repo. Never edit
scene_ours.xml by hand. tests/test_model.py compares the file with a fresh render: a difference in anything
SimRobot relies on (names, shapes, sizes, masses, collision classes, mocap bodies, tray walls) fails the
test; a difference only in the initial poses or the camera (A15 moving the cereal box in fake_scene.json,
B4 measuring the camera) is cosmetic, since SimRobot.reset writes every pose, and only raises a warning.

Sources: `config/objects.yaml` (shapes, `size` = [length along yaw, width across], `h`), `config/scene.yaml`
(`zones.tray.r` and `.wall`, the overhead camera pose), `config/fake_scene.json` (initial poses). World
coordinates are our table frame (origin under the shoulder_pan axis, +x out from the arm, +z up, table at
z = 0), so a scene (x, y) is a MuJoCo (x, y) with no conversion.

Bodies and geoms (every name below is fixed; arm/sim_robot.py relies on them):

- Robot: the bodies of so101_new_calib.xml (`base`, `shoulder`, `upper_arm`, `lower_arm`, `wrist`, `gripper`,
  `moving_jaw_so101_v1`), joints `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper`, position actuators with the same names, sites `tip` and `grasp` in body `gripper`. Robot
  collision geoms have contype 1, conaffinity 1; only the gripper boxes are named (`gripper_housing`,
  `finger_fixed`, `finger_moving`). Look joints up with `mujoco.mj_name2id`, never by index.
- Grasp geometry (binding for pick/place in SimRobot): the SO-101 jaw is asymmetric, only the moving jaw
  swings, so `tip` (the closed-jaw midpoint) is 0.3 mm from the fixed finger's inner face and an object
  centered on `tip` is inside that finger. The grasp center is the site `grasp`, `arm.ik.grasp_offset()`
  (2.46 cm) from `tip` along the closing direction `c`: `pick` targets it with `arm.ik.solve_grasp(center)`
  (not `solve`), `close_gripper` measures its 1.0 cm attach distance from `arm.ik.fk_grasp(q)`, and the held
  object's pose offset is captured relative to the tip pose as before. Open the gripper fully (1.0) before
  descending: at 0.6 rad the moving finger hits the upper part of the 10.5 cm juice box picked at
  mid-height; fully open, every object of config/objects.yaml centered on `grasp` with the tip at its
  mid-height touches nothing (tests/test_model.py). The real jaws then push the object 5 mm (4 cm wide) to
  7.5 mm (3.5 cm) toward the fixed finger when they close, within the 1.5 cm accuracy budget of A6.
- `table`: a box geom in the world body whose top face is z = 0 (contype 1, conaffinity 3).
- `tray`: a mocap body at (zone x, zone y, 0) holding the floor geom `tray` (3 mm thick, half-size r) and
  the walls `tray_wall_px`, `tray_wall_nx`, `tray_wall_py`, `tray_wall_ny` (5 mm thick, height `wall`,
  outer faces at +-r). Five geoms cannot share one name, so map a contact to a name through
  `model.geom_bodyid` and the body name (that works for every body here).
- One free body per object (kind `object`), body, free joint and geom all named after the object, with a
  box (`size` = [l/2, w/2, h/2]) or a cylinder (`size` = [w/2, h/2]) geom, mass 0.05, contype 1,
  conaffinity 3.
- One mocap body per obstacle (kind `obstacle`), body and box geom named after it, contype 1, conaffinity 3.
- `hand`: a mocap body with the box geom `hand` (0.08 x 0.10 x 0.04 m), contype 1, conaffinity 3, parked
  at z = -1 whenever the scene's hand is not seen.

Poses at run time (SimRobot.reset):

- Object: `adr = model.jnt_qposadr[mj_name2id(JOINT, name)]`; write `data.qpos[adr:adr+7] = [x, y, h/2,
  cos(yaw/2), 0, 0, sin(yaw/2)]` and zero `data.qvel[dofadr:dofadr+6]` (`model.jnt_dofadr`), then
  `mj_forward`.
- Obstacle, tray and hand: `k = model.body_mocapid[mj_name2id(BODY, name)]`; write `data.mocap_pos[k] =
  (x, y, h/2)` (tray: z 0; hand: the scene's hand z, or -1.0 when `seen` is false) and
  `data.mocap_quat[k] = (cos(yaw/2), 0, 0, sin(yaw/2))`.
- Held object: set its geom's `geom_contype = 2` and `geom_conaffinity = 2` while held (no robot contacts,
  still collides with table, tray, obstacles, hand and other objects), restore 1 and 3 on release.

Collision matrix: robot (1, 1) hits objects, obstacles, table, tray and hand (1, 3); a held object (2, 2)
hits everything except the robot; objects hit each other.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
OUT_PATH = HERE / "scene_ours.xml"
OBJECTS_PATH = ROOT / "config" / "objects.yaml"
SCENE_CFG_PATH = ROOT / "config" / "scene.yaml"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

OBJECT_MASS = 0.05
TRAY_FLOOR = 0.003
TRAY_WALL_THICKNESS = 0.005
HAND_HALF = (0.04, 0.05, 0.02)
HAND_PARKED_Z = -1.0
TABLE_CENTER = (0.25, 0.0)
TABLE_HALF = (0.45, 0.50, 0.02)
TIMESTEP = 0.002  # SimRobot sets 1/600 itself

COLORS = {
    "pill_bottle": "0.95 0.95 0.9 1",
    "juice_box": "0.95 0.55 0.15 1",
    "tea_tin": "0.2 0.55 0.3 1",
    "cereal_box": "0.7 0.2 0.15 1",
}
DEFAULT_OBJECT_COLOR = "0.3 0.5 0.9 1"
OBSTACLE_COLOR = "0.6 0.3 0.2 1"


def _f(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _vec(*xs: float) -> str:
    return " ".join(_f(x) for x in xs)


def yaw_quat(yaw: float) -> str:
    return _vec(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def load_inputs() -> tuple[dict, dict, dict]:
    objects = yaml.safe_load(OBJECTS_PATH.read_text(encoding="utf-8"))
    scene_cfg = yaml.safe_load(SCENE_CFG_PATH.read_text(encoding="utf-8"))
    fake_scene = json.loads(FAKE_SCENE_PATH.read_text(encoding="utf-8"))
    return objects, scene_cfg, fake_scene


def _pose(name: str, group: dict, fallback_index: int) -> tuple[float, float, float]:
    item = group.get(name)
    if item is None:  # not in the fake scene: park it off the workspace, SimRobot.reset places it anyway
        return 0.55, -0.45 + 0.1 * fallback_index, 0.0
    return float(item["x"]), float(item["y"]), float(item.get("yaw", 0.0))


def _shape_geom(name: str, entry: dict, rgba: str, mass: float | None) -> str:
    length, width = (float(v) for v in entry["size"])
    h = float(entry["h"])
    mass_attr = f' mass="{_f(mass)}"' if mass is not None else ""
    if entry["shape"] == "cylinder":
        size = _vec(width / 2, h / 2)
        return (
            f'<geom name="{name}" type="cylinder" size="{size}"{mass_attr} '
            f'contype="1" conaffinity="3" rgba="{rgba}"/>'
        )
    size = _vec(length / 2, width / 2, h / 2)
    return (
        f'<geom name="{name}" type="box" size="{size}"{mass_attr} contype="1" conaffinity="3" rgba="{rgba}"/>'
    )


def render(objects: dict, scene_cfg: dict, fake_scene: dict) -> str:
    """Return the scene_ours.xml text."""
    tray = scene_cfg["zones"]["tray"]
    r, wall = float(tray["r"]), float(tray["wall"])
    t = TRAY_WALL_THICKNESS
    cam = scene_cfg.get("camera", {})
    cam_c = cam.get("c", [0.20, -0.05])
    cam_h = float(cam.get("Hc", 0.65))
    hand = fake_scene.get("hand", {})
    hand_z = float(hand.get("z", 0.03)) if hand.get("seen") else HAND_PARKED_Z
    hand_xy = (float(hand.get("x", 0.3)), float(hand.get("y", 0.0)))

    lines = [
        '<mujoco model="so101_scene_ours">',
        "  <!-- GENERATED by `python -m sim.model.build_scene` from config/objects.yaml, config/scene.yaml",
        "       and config/fake_scene.json. Do not edit; build_scene.py lists every body and geom name. -->",
        '  <include file="so101_new_calib.xml"/>',
        "",
        f'  <option timestep="{_f(TIMESTEP)}" integrator="implicitfast"/>',
        "",
        "  <visual>",
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="160" elevation="-20" offwidth="640" offheight="480"/>',
        "  </visual>",
        "",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"'
        ' height="3072"/>',
        '    <texture type="2d" name="tabletop" builtin="checker" mark="edge" rgb1="0.82 0.82 0.78"'
        ' rgb2="0.72 0.72 0.68" markrgb="0.6 0.6 0.6" width="300" height="300"/>',
        '    <material name="tabletop" texture="tabletop" texuniform="true" texrepeat="10 10"'
        ' reflectance="0.1"/>',
        "  </asset>",
        "",
        "  <worldbody>",
        '    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>',
        '    <light pos="0.6 -0.6 1.2" dir="-0.4 0.4 -0.8" diffuse="0.4 0.4 0.4" specular="0 0 0"/>',
        f'    <camera name="overhead" pos="{_vec(float(cam_c[0]), float(cam_c[1]), cam_h)}"'
        ' xyaxes="0 -1 0 1 0 0" fovy="43.6"/>',
        '    <camera name="side" pos="0.30 -1.0 0.45" xyaxes="1 0 0 0 0.35 0.94" fovy="45"/>',
        "",
        "    <!-- Table: top face at z = 0 -->",
        f'    <geom name="table" type="box" pos="{_vec(TABLE_CENTER[0], TABLE_CENTER[1], -TABLE_HALF[2])}"'
        f' size="{_vec(*TABLE_HALF)}" material="tabletop" contype="1" conaffinity="3"/>',
        "",
        f"    <!-- Tray: zone `tray`, half-size r = {_f(r)}, wall height {_f(wall)} -->",
        f'    <body name="tray" mocap="true" pos="{_vec(float(tray["x"]), float(tray["y"]), 0.0)}">',
        f'      <geom name="tray" type="box" pos="{_vec(0, 0, TRAY_FLOOR / 2)}"'
        f' size="{_vec(r, r, TRAY_FLOOR / 2)}" contype="1" conaffinity="3" rgba="0.55 0.38 0.22 1"/>',
    ]
    walls = [
        ("tray_wall_px", (r - t / 2, 0.0), (t / 2, r)),
        ("tray_wall_nx", (-(r - t / 2), 0.0), (t / 2, r)),
        ("tray_wall_py", (0.0, r - t / 2), (r, t / 2)),
        ("tray_wall_ny", (0.0, -(r - t / 2)), (r, t / 2)),
    ]
    for name, (px, py), (sx, sy) in walls:
        lines.append(
            f'      <geom name="{name}" type="box" pos="{_vec(px, py, wall / 2)}"'
            f' size="{_vec(sx, sy, wall / 2)}" contype="1" conaffinity="3" rgba="0.55 0.38 0.22 1"/>'
        )
    lines.append("    </body>")

    lines.append("")
    lines.append("    <!-- Objects: one free body each, poses from config/fake_scene.json -->")
    objs = fake_scene.get("objects", {})
    obst = fake_scene.get("obstacles", {})
    for i, (marker, entry) in enumerate(sorted(objects.items(), key=lambda kv: int(kv[0]))):
        name = entry["name"]
        if entry["kind"] == "object":
            x, y, yaw = _pose(name, objs, i)
            rgba = COLORS.get(name, DEFAULT_OBJECT_COLOR)
            lines.append(f"    <!-- {name}: marker {marker} -->")
            pos = _vec(x, y, float(entry["h"]) / 2)
            lines.append(f'    <body name="{name}" pos="{pos}" quat="{yaw_quat(yaw)}">')
            lines.append(f'      <freejoint name="{name}"/>')
            lines.append("      " + _shape_geom(name, entry, rgba, OBJECT_MASS))
            lines.append("    </body>")
    lines.append("")
    lines.append("    <!-- Obstacles: mocap bodies, never moved by skills -->")
    for i, (marker, entry) in enumerate(sorted(objects.items(), key=lambda kv: int(kv[0]))):
        name = entry["name"]
        if entry["kind"] == "obstacle":
            x, y, yaw = _pose(name, obst, i)
            rgba = COLORS.get(name, OBSTACLE_COLOR)
            lines.append(f"    <!-- {name}: marker {marker} -->")
            pos = _vec(x, y, float(entry["h"]) / 2)
            lines.append(f'    <body name="{name}" mocap="true" pos="{pos}" quat="{yaw_quat(yaw)}">')
            lines.append("      " + _shape_geom(name, entry, rgba, None))
            lines.append("    </body>")
    lines.append("")
    lines.append("    <!-- Hand (wristband) box: parked at z = -1 while the hand is not seen -->")
    lines.append(f'    <body name="hand" mocap="true" pos="{_vec(hand_xy[0], hand_xy[1], hand_z)}">')
    lines.append(
        f'      <geom name="hand" type="box" size="{_vec(*HAND_HALF)}" contype="1" conaffinity="3"'
        ' rgba="0.9 0.72 0.6 1"/>'
    )
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


def build(out_path: pathlib.Path = OUT_PATH) -> pathlib.Path:
    """Render from the config files and write `out_path`."""
    text = render(*load_inputs())
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=pathlib.Path, default=OUT_PATH, help="output path (default: scene_ours.xml)"
    )
    parser.add_argument(
        "--check", action="store_true", help="load the result with MuJoCo and print a summary"
    )
    args = parser.parse_args()
    path = build(args.out)
    print(f"wrote {path}")
    if args.check:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(path))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        print(f"bodies: {names}")
        print(f"ngeom {m.ngeom}, nmocap {m.nmocap}, nq {m.nq}")


if __name__ == "__main__":
    main()
