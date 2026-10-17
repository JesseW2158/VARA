"""Fit the gripper's collision primitives and the tip site from the SO-101 mesh vertices.

`python -m sim.model.fit_gripper` prints the `<site>` and `<geom>` lines that `so101_new_calib.xml`
carries for the gripper (NOTES in sim/model/README.md). Run it again only if the upstream meshes change;
tests/test_model.py re-checks the numbers that matter (tip midpoint, inner jaw faces, empty gap).

Method, all in each body's own frame:
- Vertices of a mesh geom are `geom_pos + R(geom_quat) @ mesh_vert`, taken from the *visual* geoms (they
  keep the meshes after the collision meshes become boxes).
- Jaw tips: the vertices within 2 mm of the extreme along the jaw axis (gripper body -z); the moving jaw is
  rotated to the closed limit of the `gripper` joint first. `tip` = midpoint of the two tip centroids.
- Finger boxes: one box per finger, its inner face (the one facing the other jaw) on the plane through the
  tip and root inner-face extremes, its length covering the finger, its thickness and width the mean of the
  tip and root cross sections (the fingers are wedges; the mean splits the error on the outer side).
- Housing box: axis-aligned bounding box of the fixed part above the finger root.
- `grasp` site: GRASP_FROM_FIXED_FACE from the fixed finger box's inner face at the tip level, along the
  closing direction. Only the moving jaw swings, so a grasped object ends up against the fixed jaw, not
  centered on `tip`; a 4 cm object (the widest in config/objects.yaml) centered on `grasp` keeps 5 mm from
  the fixed finger, the IK position tolerance, so a descending finger never touches it.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / "so101_new_calib.xml"

FINGER_ROOT_Z = -0.040  # gripper body frame: below this z the fixed part is the finger
MOVING_ROOT_Y = -0.024  # moving jaw body frame: below this y the moving part is the finger (hub above)
TIP_SLICE = 0.002
GRASP_FROM_FIXED_FACE = 0.025  # m, grasp center to the fixed finger's inner face (box) at the tip level


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, q)
    return r.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def mesh_vertices_in_body(m: mujoco.MjModel, body: str, mesh: str) -> np.ndarray:
    """Vertices of the first geom in `body` that uses `mesh`, expressed in the body frame."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, mesh)
    for g in range(m.ngeom):
        if (
            m.geom_bodyid[g] == bid
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and m.geom_dataid[g] == mid
        ):
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[adr : adr + num]
            return m.geom_pos[g] + v @ quat_to_mat(m.geom_quat[g]).T
    raise KeyError(f"no geom with mesh {mesh} in body {body}")


def tip_vertices(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Vertices within TIP_SLICE of the extreme along `axis` (unit vector pointing out of the jaws)."""
    s = v @ axis
    return v[s > s.max() - TIP_SLICE]


def fit_finger_box(
    v: np.ndarray, tip_inner: np.ndarray, root_inner: np.ndarray, inner_normal_hint: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, quat, half_sizes) of a box whose inner face passes through the two inner points.

    The box frame is x = inner normal (pointing into the gap), y = width, z = finger axis (root to tip).
    """
    axis = tip_inner - root_inner
    axis /= np.linalg.norm(axis)
    n = inner_normal_hint - axis * (inner_normal_hint @ axis)
    n /= np.linalg.norm(n)
    w = np.cross(axis, n)
    frame = np.stack([n, w, axis], axis=1)  # columns
    local = (v - root_inner) @ frame  # coordinates in the box frame, origin at the root inner point
    length = local[:, 2]
    lo, hi = length.min(), length.max()
    span = hi - lo
    tip_slice = local[length > hi - 0.2 * span]
    root_slice = local[length < lo + 0.2 * span]
    inner = max(tip_slice[:, 0].max(), root_slice[:, 0].max())  # the inner face plane (should be ~0)
    thickness = np.mean([inner - tip_slice[:, 0].min(), inner - root_slice[:, 0].min()])
    width = np.mean([np.ptp(tip_slice[:, 1]), np.ptp(root_slice[:, 1])])
    width_center = np.mean([tip_slice[:, 1].mean(), root_slice[:, 1].mean()])
    center_local = np.array([inner - thickness / 2, width_center, (lo + hi) / 2])
    pos = root_inner + frame @ center_local
    half = np.array([thickness / 2, width / 2, span / 2])
    return pos, mat_to_quat(frame), half


def inner_face_x(pos: np.ndarray, quat: np.ndarray, half: np.ndarray, y: float, z: float) -> float:
    """x of the +x face of a fitted finger box (its inner face) at body-frame (y, z), i.e. the point where the
    plane `(p - pos) . n = half[0]` with `n` the box's local x axis crosses the line (y, z)."""
    n = quat_to_mat(quat)[:, 0]
    return float(pos[0] + (half[0] - (y - pos[1]) * n[1] - (z - pos[2]) * n[2]) / n[0])


def fmt(a: np.ndarray, nd: int = 4) -> str:
    return " ".join(f"{x:.{nd}f}" for x in a)


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    gripper = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    jg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")

    vf = mesh_vertices_in_body(m, "gripper", "wrist_roll_follower_so101_v1")
    vm = mesh_vertices_in_body(m, "moving_jaw_so101_v1", "moving_jaw_so101_v1")
    out = np.array([0.0, 0.0, -1.0])  # jaw direction in the gripper body frame

    # --- tip site: closed-jaw midpoint, in the gripper body frame
    fixed_tip = tip_vertices(vf, out).mean(0)
    d.qpos[:] = 0
    d.qpos[m.jnt_qposadr[jg]] = m.jnt_range[jg][0]
    mujoco.mj_kinematics(m, d)
    rel_pos = d.xpos[jaw] - d.xpos[gripper]
    rel = d.xmat[gripper].reshape(3, 3).T @ (d.xmat[jaw].reshape(3, 3))
    vm_in_gripper = (d.xmat[gripper].reshape(3, 3).T @ rel_pos) + vm @ rel.T
    moving_tip = tip_vertices(vm_in_gripper, out).mean(0)
    tip = (fixed_tip + moving_tip) / 2
    print(f"fixed jaw tip centroid (gripper frame): {fmt(fixed_tip)}")
    print(f"moving jaw tip centroid at closed limit: {fmt(moving_tip)}")
    print(f"closed-jaw midpoint: {fmt(tip)}")
    print(f'<site name="tip" pos="{fmt(tip)}" quat="0.707107 0 0.707107 0" size="0.003" rgba="1 0 0 1"/>')

    # --- fixed finger
    fin = vf[vf[:, 2] < FINGER_ROOT_Z]
    tip_bin = fin[fin[:, 2] < fin[:, 2].min() + 0.004]
    root_bin = fin[fin[:, 2] > FINGER_ROOT_Z - 0.006]
    tip_inner = np.array([tip_bin[:, 0].max(), 0.0, tip_bin[:, 2].mean()])
    root_inner = np.array([root_bin[:, 0].max(), 0.0, root_bin[:, 2].mean()])
    pos, quat, half = fit_finger_box(fin, tip_inner, root_inner, np.array([1.0, 0.0, 0.0]))
    print(
        f"fixed finger: inner face x {tip_inner[0]:.4f} at tip, {root_inner[0]:.4f} at root; "
        f"thickness {2 * half[0]:.4f} width {2 * half[1]:.4f} length {2 * half[2]:.4f}"
    )
    print(
        f'<geom name="finger_fixed" class="collision" type="box" pos="{fmt(pos)}" quat="{fmt(quat, 6)}" '
        f'size="{fmt(half)}"/>'
    )

    # --- grasp site: GRASP_FROM_FIXED_FACE into the gap from the fixed finger box, at the tip level
    face_x = inner_face_x(pos, quat, half, tip[1], tip[2])
    grasp = np.array([face_x + GRASP_FROM_FIXED_FACE, tip[1], tip[2]])
    print(
        f"fixed finger box inner face at the tip level: x {face_x:.4f} (tip x {tip[0]:.4f}, "
        f"{(tip[0] - face_x) * 1000:.1f} mm into the gap); grasp = face + {GRASP_FROM_FIXED_FACE} m"
    )
    print(
        f'<site name="grasp" pos="{fmt(grasp)}" quat="0.707107 0 0.707107 0" size="0.003" rgba="0 0.6 1 1"/>'
    )

    # --- housing (fixed part above the finger root)
    hous = vf[vf[:, 2] >= FINGER_ROOT_Z]
    lo, hi = hous.min(0), hous.max(0)
    print(
        f'<geom name="gripper_housing" class="collision" type="box" pos="{fmt((lo + hi) / 2)}" '
        f'size="{fmt((hi - lo) / 2)}"/>'
    )

    # --- moving finger (its own body frame: finger along -y, inner side -x)
    mf = vm[vm[:, 1] < MOVING_ROOT_Y]
    tip_bin = mf[mf[:, 1] < mf[:, 1].min() + 0.004]
    root_bin = mf[mf[:, 1] > MOVING_ROOT_Y - 0.006]
    tip_inner = np.array([tip_bin[:, 0].min(), tip_bin[:, 1].mean(), 0.0])
    root_inner = np.array([root_bin[:, 0].min(), root_bin[:, 1].mean(), 0.0])
    pos, quat, half = fit_finger_box(mf, tip_inner, root_inner, np.array([-1.0, 0.0, 0.0]))
    print(
        f"moving finger: inner face x {tip_inner[0]:.4f} at tip, {root_inner[0]:.4f} at root; "
        f"thickness {2 * half[0]:.4f} width {2 * half[1]:.4f} length {2 * half[2]:.4f}"
    )
    print(
        f'<geom name="finger_moving" class="collision" type="box" pos="{fmt(pos)}" quat="{fmt(quat, 6)}" '
        f'size="{fmt(half)}"/>'
    )


if __name__ == "__main__":
    main()
