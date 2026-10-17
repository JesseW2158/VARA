"""Forward and inverse kinematics of the SO-101 tip (task A7).

Everything is in our table frame (meters, radians): origin under the shoulder_pan axis, +x out from the arm,
+z up. Joint vectors are `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`.

Two sites of sim/model/so101_new_calib.xml matter, both in body `gripper` with the same orientation:

- `tip`: the midpoint of the two jaw tips with the gripper closed. `fk`, `fk_tip`, `solve` and `solve_exact`
  work on it; `arm.replay.validate` and the hand-distance checks use `fk_tip`.
- `grasp`: the grasp center, `grasp_offset()` (about 2.4 cm) from `tip` along the closing direction `c`.
  The SO-101 jaw is asymmetric: only the moving jaw swings, the fixed jaw's inner face is 0.4 mm from
  `tip`, so an object centered on `tip` is inside the fixed finger. `fk_grasp` and `solve_grasp` work on
  `grasp`: a pick moves the grasp center onto the object's center line (see arm/sim_robot.py).

The site x axis `a` points out of the jaws, its z axis `c` is the closing direction (toward the moving jaw);
`pitch = asin(a_z)`, 0 = jaws horizontal, -pi/2 = straight down. A pitch past vertical is clamped to
[-pi/2, pi/2] (it has the same `a_z` as its mirror image).

`solve` is damped least squares on the first four joints with a finite-difference Jacobian from
`mujoco.mj_kinematics`. Before any iteration, a planar reach check (the three pitch joints share one axis,
measured from the model at load time) discards (target, pitch) pairs the arm cannot reach even with the
tolerances, and the planar closed form provides the start postures, so a solve costs a few hundred
microseconds when the pitch is reachable and stays under 20 ms even when every ladder rung has to be
proven unreachable. When the requested pitch is unreachable `solve` relaxes the pitch toward horizontal
through `PITCH_LADDER`, nearest rung first; `solve_exact` does not. wrist_roll is not
solved for: it is copied from `q0` (clipped to the limits minus 3 degrees). The model is loaded once per
process and every call takes `_LOCK`, so the module is safe to use from several threads.

    python -m arm.ik --check
"""

from __future__ import annotations

import argparse
import math
import pathlib
import threading
import time
from functools import lru_cache

import mujoco
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "sim" / "model" / "arm_only.xml"
LIMITS_PATH = ROOT / "config" / "limits.yaml"
SCENE_PATH = ROOT / "config" / "scene.yaml"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TIP_SITE = "tip"
GRASP_SITE = "grasp"

PITCH_LADDER = (-75, -60, -45, -30, -15, 0)  # degrees, tried nearest-first when the request fails
DAMPING = 0.02  # lambda of the damped least squares step
FD_STEP = 1e-4  # rad, finite-difference step for the Jacobian
MAX_ITERS = 200
MAX_STEP = 0.5  # rad, cap on the joint step of one iteration (keeps the start-up steps sane)
STALL_ITERS = 5  # give up on a start posture after this many iterations without real progress
STALL_REL = (
    0.005  # progress means the error shrinks by at least this fraction (creeping along a limit is not)
)
MIN_STEP = 1e-7  # rad, a smaller clipped step means the iterate sits in a minimum or against the limits
POLISH_ITERS = 3  # extra iterations once inside the tolerances (Gauss-Newton gets to micrometers)
POS_TOL = 0.005  # m
PITCH_TOL = math.radians(3.0)
# The pitch row of the error and the Jacobian is scaled so one tolerance of pitch weighs as much as one
# tolerance of position; at an unreachable pose the least-squares compromise then lands near both.
PITCH_WEIGHT = POS_TOL / PITCH_TOL
# A start posture that ends outside the tolerances but within this (position + weighted pitch error, m) is
# continued toward the inside of the tolerance box: the errors are dead-zoned, i.e. only the part beyond
# this fraction of each tolerance pulls, for NEAR_MISS_ITERS iterations per fraction.
NEAR_MISS = 2 * (POS_TOL + PITCH_WEIGHT * PITCH_TOL)
NEAR_MISS_DEADZONES = (0.5, 0.9)
NEAR_MISS_ITERS = 30
MARGIN = math.radians(3.0)  # IK keeps 3 degrees inside the limits, like the driver and validate()
# Slack of the planar reach check: a target POS_TOL beyond the reach at a pitch PITCH_TOL off still counts
# as solved, which moves the wrist by up to 5 mm + 0.164 m * sin(3 deg) = 14 mm; the rest is rounding.
REACH_MARGIN = 0.02

_LOCK = threading.RLock()
_MODEL: mujoco.MjModel | None = None
_DATA: mujoco.MjData | None = None
_QADR: np.ndarray | None = None  # qpos address of each of the 5 joints
_TIP = -1
_GRASP = -1
_WRIST_FLEX = -1  # joint id, for the planar reach check
_LO = np.zeros(5)  # joint limits used for clamping (config/limits.yaml intersected with the model)
_HI = np.zeros(5)
# Planar chain measured at load (q = 0, in the x-z plane): shoulder_lift axis (r, z), upper arm and forearm
# lengths, their angles from +x, the sign of a pan step on the target's bearing and of a flex step on the
# in-plane angle. None when the model is not planar (then the solver runs without the shortcuts).
_PLANAR: dict[str, float] | None = None

Vec3 = tuple[float, float, float]


def load_model() -> mujoco.MjModel:
    """Load sim/model/arm_only.xml once per process and return it."""
    global _MODEL, _DATA, _QADR, _TIP, _GRASP, _WRIST_FLEX, _LO, _HI, _PLANAR
    with _LOCK:
        if _MODEL is None:
            model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
            data = mujoco.MjData(model)
            joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
            if min(joint_ids) < 0:
                raise RuntimeError(f"{MODEL_PATH} lacks one of the joints {JOINTS}")
            tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
            grasp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
            if tip < 0 or grasp < 0:
                raise RuntimeError(f"{MODEL_PATH} needs the sites {TIP_SITE!r} and {GRASP_SITE!r}")
            lo = np.array([model.jnt_range[j][0] for j in joint_ids])
            hi = np.array([model.jnt_range[j][1] for j in joint_ids])
            cfg = _config_limits()
            for i, name in enumerate(JOINTS):
                if name in cfg:
                    lo[i] = max(lo[i], float(cfg[name][0]))
                    hi[i] = min(hi[i], float(cfg[name][1]))
            _QADR = np.array([model.jnt_qposadr[j] for j in joint_ids])
            _TIP, _GRASP, _WRIST_FLEX = tip, grasp, joint_ids[3]
            _LO, _HI = lo, hi
            _DATA = data
            _MODEL = model
            _PLANAR = _measure_planar_chain(joint_ids)
        return _MODEL


def _measure_planar_chain(joint_ids: list[int]) -> dict[str, float] | None:
    """Link geometry at q = 0 for the reach check and the analytic start postures. Caller holds _LOCK."""
    _kinematics(np.zeros(5))
    axes = _DATA.xaxis[joint_ids]
    pan_axis, flex_axes = axes[0], axes[1:4]
    if abs(abs(pan_axis[2]) - 1.0) > 1e-6 or np.any(np.abs(np.abs(flex_axes[:, 1]) - 1.0) > 1e-6):
        return None  # not the SO-101 layout: pan about z, the three pitch joints about y
    if not (np.all(flex_axes[:, 1] > 0) or np.all(flex_axes[:, 1] < 0)):
        return None
    s, e, w = (_DATA.xanchor[joint_ids[k]] for k in (1, 2, 3))
    return {
        "s_r": float(s[0]),
        "s_z": float(s[2]),
        "l1": math.hypot(e[0] - s[0], e[2] - s[2]),
        "l2": math.hypot(w[0] - e[0], w[2] - e[2]),
        "phi1": math.atan2(e[2] - s[2], e[0] - s[0]),
        "phi2": math.atan2(w[2] - e[2], w[0] - e[0]),
        # rotating about -z turns +x toward -y: a target at bearing atan2(y, x) needs pan = -bearing
        "pan_sign": -1.0 if pan_axis[2] < 0 else 1.0,
        # a positive step about +y lowers the in-plane angle atan2(z, x)
        "flex_sign": 1.0 if flex_axes[0, 1] > 0 else -1.0,
    }


def _config_limits() -> dict:
    try:
        cfg = yaml.safe_load(LIMITS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def limits() -> dict[str, list[float]]:
    """Joint limits used by the solver: config/limits.yaml intersected with the model's ranges."""
    load_model()
    return {name: [float(_LO[i]), float(_HI[i])] for i, name in enumerate(JOINTS)}


@lru_cache(maxsize=1)
def home_q() -> tuple[float, ...]:
    """`home_q` from config/scene.yaml."""
    cfg = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    q = tuple(float(v) for v in cfg["home_q"])
    if len(q) != 5:
        raise ValueError(f"home_q must have 5 joints, got {q}")
    return q


def _as_q(q) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    if arr.shape != (5,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"a joint vector has 5 finite values {JOINTS}, got {q!r}")
    return arr


def _as_target(xyz) -> np.ndarray:
    arr = np.asarray(xyz, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"a target is 3 finite coordinates (x, y, z) in meters, got {xyz!r}")
    return arr


def _as_pitch(pitch) -> float:
    p = float(pitch)
    if not math.isfinite(p):
        raise ValueError(f"pitch must be a finite angle in radians, got {pitch!r}")
    return max(-math.pi / 2, min(math.pi / 2, p))


def _kinematics(q: np.ndarray) -> None:
    """Write the 5 joint angles and run forward kinematics (positions only). Caller holds _LOCK."""
    _DATA.qpos[_QADR] = q[:5]
    mujoco.mj_kinematics(_MODEL, _DATA)


def _site_pose(site: int) -> tuple[np.ndarray, float]:
    """Position (copy) and pitch of a gripper site from the last kinematics call."""
    a_z = float(_DATA.site_xmat[site][6])  # row 2, column 0: z component of the site x axis
    return _DATA.site_xpos[site].copy(), math.asin(max(-1.0, min(1.0, a_z)))


def _fk_site(q, site_name: str) -> tuple[Vec3, float]:
    arr = _as_q(q)
    load_model()
    with _LOCK:
        _kinematics(arr)
        p, pitch = _site_pose(_TIP if site_name == TIP_SITE else _GRASP)
    return (float(p[0]), float(p[1]), float(p[2])), pitch


def fk(q) -> tuple[Vec3, float]:
    """Tip position (x, y, z) in our frame and pitch for a 5-joint vector."""
    return _fk_site(q, TIP_SITE)


def fk_tip(q) -> Vec3:
    """Tip position only; the `fk` that `arm.replay.validate` receives."""
    return fk(q)[0]


def fk_grasp(q) -> tuple[Vec3, float]:
    """Grasp-center position (site `grasp`) and pitch: where a held object's center line sits."""
    return _fk_site(q, GRASP_SITE)


def jaw_axes(q) -> tuple[Vec3, Vec3]:
    """Unit jaw direction `a` (out of the jaws) and unit closing direction `c` in our frame."""
    arr = _as_q(q)
    load_model()
    with _LOCK:
        _kinematics(arr)
        m = _DATA.site_xmat[_TIP]
        a = (float(m[0]), float(m[3]), float(m[6]))
        c = (float(m[2]), float(m[5]), float(m[8]))
    return a, c


@lru_cache(maxsize=1)
def grasp_offset() -> float:
    """Distance from `tip` to `grasp` along the closing direction `c` (a constant of the model)."""
    load_model()
    with _LOCK:
        _kinematics(np.zeros(5))
        c = _DATA.site_xmat[_TIP].reshape(3, 3)[:, 2]
        return float((_DATA.site_xpos[_GRASP] - _DATA.site_xpos[_TIP]) @ c)


def _step(jac: np.ndarray, err: np.ndarray, damping: np.ndarray) -> np.ndarray:
    """One damped least squares step, J^T (J J^T + lambda^2 I)^-1 e, capped at MAX_STEP per joint."""
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping, err)
    biggest = float(np.max(np.abs(dq)))
    if biggest > MAX_STEP:
        dq *= MAX_STEP / biggest
    return dq


def _dls(
    target: np.ndarray, pitch: float, q_start: np.ndarray, site: int, deadzone: float, max_iters: int
) -> tuple[np.ndarray, bool, float]:
    """Damped least squares from q_start (5 joints; the roll stays fixed). With `deadzone` > 0 only the part
    of each error beyond that fraction of its tolerance pulls, so the iterate settles inside the tolerance
    box instead of at the weighted compromise nearest to its center. Returns (q, ok, measure): `measure` is
    the position error plus the pitch error scaled by PITCH_WEIGHT, in meters, of the last iterate (0 on
    success). Caller holds _LOCK."""
    lo, hi = _LO[:4] + MARGIN, _HI[:4] - MARGIN
    q = q_start.astype(float).copy()
    q[:4] = np.clip(q[:4], lo, hi)
    jac = np.zeros((4, 4))
    err = np.zeros(4)
    damping = DAMPING * DAMPING * np.eye(4)
    best = math.inf
    measure = math.inf
    stalled = 0
    good: np.ndarray | None = None  # the latest iterate inside both tolerances
    polished = 0
    for _ in range(max_iters):
        _kinematics(q)
        p0, pitch0 = _site_pose(site)
        err[:3] = target - p0
        pitch_err = pitch - pitch0
        pos_err = math.sqrt(float(np.dot(err[:3], err[:3])))
        if pos_err < POS_TOL and abs(pitch_err) < PITCH_TOL:
            good = q.copy()
            if polished >= POLISH_ITERS:
                return good, True, 0.0
            polished += 1  # inside the tolerances: a few more steps for sub-millimeter accuracy
        elif good is not None:
            return good, True, 0.0  # polishing left the tolerance box (unreachable pose): keep the good one
        measure = pos_err + PITCH_WEIGHT * abs(pitch_err)
        if measure < best * (1.0 - STALL_REL):
            best, stalled = measure, 0
        else:
            stalled += 1
            if stalled >= STALL_ITERS:  # creeping along a limit or sitting in a minimum: give up here
                break
        if deadzone > 0.0:
            err[:3] *= max(0.0, 1.0 - deadzone * POS_TOL / pos_err) if pos_err > 0.0 else 0.0
            pitch_err = math.copysign(max(0.0, abs(pitch_err) - deadzone * PITCH_TOL), pitch_err)
        err[3] = PITCH_WEIGHT * pitch_err
        for j in range(4):
            qj = q[j]
            q[j] = qj + FD_STEP
            _kinematics(q)
            pj, pitchj = _site_pose(site)
            q[j] = qj
            jac[:3, j] = (pj - p0) / FD_STEP
            jac[3, j] = PITCH_WEIGHT * (pitchj - pitch0) / FD_STEP
        dq = _step(jac, err, damping)
        q_new = np.clip(q[:4] + dq, lo, hi)
        saturated = (q[:4] + dq < lo) | (q[:4] + dq > hi)
        if saturated.any() and not saturated.all():
            # joints pushed into a limit stay there; the free joints redo the step without them
            jac_free = jac.copy()
            jac_free[:, saturated] = 0.0
            dq = _step(jac_free, err, damping)
            free = ~saturated
            q_new[free] = np.clip(q[:4][free] + dq[free], lo[free], hi[free])
        if float(np.max(np.abs(q_new - q[:4]))) < MIN_STEP:
            break  # converged to a minimum that is not the target, or pinned by the limits
        q[:4] = q_new
    if good is not None:
        return good, True, 0.0
    return q, False, measure


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _planar_seeds(target: np.ndarray, pitch: float, site: int, roll: float) -> list[np.ndarray] | None:
    """Closed-form postures of the planar chain for the site at `target` with `pitch`, or None when the arm
    cannot reach the pair even with the tolerances (so DLS need not try). Caller holds _LOCK.

    The three pitch joints share one axis, so the arm is a planar 3R chain in the vertical plane through the
    pan axis: the wrist_flex axis must sit at `site - L3 * (cos, sin)(theta + delta)` where theta is the
    in-plane angle of the jaw axis and (L3, delta) the in-plane offset of the site from the wrist_flex axis
    (depends on the roll, measured here with one kinematics call). Four chains are candidates: the arm
    turned toward the target or away from it (the tip behind the shoulder), and the jaws facing +r or -r in
    that plane; each has two elbow bends. Postures are clipped to the limits minus 3 degrees.
    """
    pc = _PLANAR
    if pc is None:
        return []  # a model with another joint layout: no reach check and no closed-form postures
    q_ref = np.zeros(5)
    q_ref[4] = roll
    _kinematics(q_ref)
    w0 = _DATA.xanchor[_WRIST_FLEX]
    p0 = _DATA.site_xpos[site]
    l3 = math.hypot(p0[0] - w0[0], p0[2] - w0[2])
    delta = math.atan2(p0[2] - w0[2], p0[0] - w0[0])
    r_t = math.hypot(target[0], target[1])
    bearing = math.atan2(target[1], target[0])
    lo, hi = _LO + MARGIN, _HI - MARGIN
    seeds: list[np.ndarray] = []
    reachable = False
    for r, heading in ((r_t, bearing), (-r_t, bearing + math.pi)):  # arm toward the target, or away
        pan = _wrap(pc["pan_sign"] * heading)
        if not lo[0] <= pan <= hi[0]:
            continue
        for theta in (pitch, math.pi - pitch):  # jaws facing +r, or -r
            w_r = r - l3 * math.cos(theta + delta)
            w_z = target[2] - l3 * math.sin(theta + delta)
            d_r, d_z = w_r - pc["s_r"], w_z - pc["s_z"]
            dist = math.hypot(d_r, d_z)
            if not abs(pc["l1"] - pc["l2"]) - REACH_MARGIN <= dist <= pc["l1"] + pc["l2"] + REACH_MARGIN:
                continue
            reachable = True
            cos_beta = (dist * dist - pc["l1"] ** 2 - pc["l2"] ** 2) / (2 * pc["l1"] * pc["l2"])
            beta0 = math.acos(max(-1.0, min(1.0, cos_beta)))
            for beta in (beta0, -beta0):  # the two elbow bends
                elbow = math.atan2(pc["l2"] * math.sin(beta), pc["l1"] + pc["l2"] * math.cos(beta))
                phi1 = math.atan2(d_z, d_r) - elbow
                q1 = pc["flex_sign"] * _wrap(pc["phi1"] - phi1)
                q2 = pc["flex_sign"] * _wrap(pc["phi2"] - pc["phi1"] - beta)
                q3 = pc["flex_sign"] * _wrap(-theta) - q1 - q2
                seed = np.clip(np.array([pan, q1, q2, _wrap(q3), roll]), lo, hi)
                if not any(np.allclose(seed, s, atol=1e-6) for s in seeds):  # beta = 0 repeats a posture
                    seeds.append(seed)
    return seeds if reachable else None


def _solve_once(target: np.ndarray, pitch: float, start: np.ndarray, site: int) -> tuple[list[float], bool]:
    """DLS for one (target, pitch) from `start` and then from the planar postures, nearest first (from home
    too when the model gives no planar postures). When every start fails but one ended close to the
    tolerance box, continue from it with dead-zoned errors: the weighted compromise can sit just outside
    the box while another point of the box is reachable."""
    with _LOCK:
        planar = _planar_seeds(target, pitch, site, float(start[4]))
        if planar is None:
            return [float(v) for v in start], False
        planar.sort(key=lambda s: float(np.linalg.norm(s[:4] - start[:4])))
        if not planar:
            home = np.array(home_q())
            home[4] = start[4]
            planar = [home]
        seeds: list[np.ndarray] = [start]
        for seed in planar:
            if not any(np.allclose(seed, s, atol=1e-6) for s in seeds):
                seeds.append(seed)
        near_miss: tuple[float, np.ndarray] | None = None
        for seed in seeds:
            q, ok, measure = _dls(target, pitch, seed, site, 0.0, MAX_ITERS)
            if ok:
                q[4] = start[4]
                return [float(v) for v in q], True
            if measure < NEAR_MISS and (near_miss is None or measure < near_miss[0]):
                near_miss = (measure, q)
        if near_miss is not None:
            q = near_miss[1]
            for deadzone in NEAR_MISS_DEADZONES:
                q, ok, _ = _dls(target, pitch, q, site, deadzone, NEAR_MISS_ITERS)
                if ok:
                    q[4] = start[4]
                    return [float(v) for v in q], True
    return [float(v) for v in start], False


def _prepare(xyz, pitch, q0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Validated target, clamped pitch, q0 as given (returned on failure) and the start posture (roll
    clipped to the limits minus 3 degrees, so no returned waypoint fails validate())."""
    load_model()
    target = _as_target(xyz)
    p = _as_pitch(pitch)
    given = _as_q(home_q() if q0 is None else q0)
    start = given.copy()
    start[4] = float(np.clip(start[4], _LO[4] + MARGIN, _HI[4] - MARGIN))
    return target, p, given, start


def _solve(xyz, pitch, q0, site_name: str, ladder: bool) -> tuple[list[float], bool]:
    target, p, given, start = _prepare(xyz, pitch, q0)
    site = _TIP if site_name == TIP_SITE else _GRASP
    pitches = [p]
    if ladder:
        pitches += [math.radians(deg) for deg in sorted(PITCH_LADDER) if math.radians(deg) > p]
    for rung in pitches:
        q, ok = _solve_once(target, rung, start, site)
        if ok:
            return q, True
    return [float(v) for v in given], False


def solve_exact(xyz, pitch: float, q0) -> tuple[list[float], bool]:
    """Solve for the tip at `xyz` with exactly `pitch` (no ladder). Returns (q, ok); q is q0 when not ok."""
    return _solve(xyz, pitch, q0, TIP_SITE, ladder=False)


def solve(xyz, pitch: float = -math.pi / 2, q0=None) -> tuple[list[float], bool]:
    """Solve for the tip at `xyz`, relaxing the pitch through PITCH_LADDER when `pitch` is unreachable.

    `q0` defaults to `home_q`; the returned wrist_roll is `q0[4]` clipped to the limits minus 3 degrees.
    Use `fk(q)[1]` to read the pitch found. Raises ValueError for a non-finite or mis-shaped input.
    """
    return _solve(xyz, pitch, q0, TIP_SITE, ladder=True)


def solve_grasp(xyz, pitch: float = -math.pi / 2, q0=None) -> tuple[list[float], bool]:
    """Like `solve`, but puts the grasp center (site `grasp`) at `xyz`: the pick and place target."""
    return _solve(xyz, pitch, q0, GRASP_SITE, ladder=True)


def _check() -> None:
    home = list(home_q())
    (x, y, z), pitch = fk(home)
    print(f"home_q {home} -> tip ({x:.4f}, {y:.4f}, {z:.4f}) pitch {math.degrees(pitch):.1f} deg")
    (gx, gy, gz), _ = fk_grasp(home)
    off = grasp_offset() * 100
    print(f"        grasp center ({gx:.4f}, {gy:.4f}, {gz:.4f}), {off:.2f} cm from the tip along c")
    targets = [
        ((0.18, -0.12, 0.035), -90.0, "pick pill_bottle"),
        ((0.24, -0.04, 0.0525), -90.0, "pick juice_box"),
        ((0.15, 0.03, 0.045), -90.0, "pick tea_tin"),
        ((0.30, 0.22, 0.0975), -90.0, "tray center"),
        ((0.32, 0.24, 0.08), -90.0, "far tray slot"),
        ((0.24, 0.11, 0.28), -90.0, "transit over cereal_box"),
        ((0.30, 0.0, 0.10), -90.0, "handoff zone"),
        ((0.20, 0.0, 0.30), -90.0, "ceiling at x 0.20"),
        ((0.40, 0.0, 0.02), -90.0, "box far edge, y 0"),
        ((0.40, 0.25, 0.02), -90.0, "box far corner"),
        ((0.9, 0.0, 0.1), -90.0, "impossible"),
    ]
    solve(targets[0][0])  # warm-up
    worst = 0.0
    for xyz, deg, label in targets:
        t0 = time.perf_counter()
        q, ok = solve(xyz, math.radians(deg))
        ms = (time.perf_counter() - t0) * 1000
        worst = max(worst, ms)
        if ok:
            p, got = fk(q)
            err = math.dist(p, xyz) * 1000
            print(
                f"{label:24s} {xyz} pitch {deg:+.0f} -> ok, pitch {math.degrees(got):+.1f} deg, "
                f"err {err:.2f} mm, {ms:.1f} ms, q={[round(v, 3) for v in q]}"
            )
        else:
            print(f"{label:24s} {xyz} pitch {deg:+.0f} -> FAILED ({ms:.1f} ms)")
    for xyz, label in (((0.18, -0.12, 0.035), "pill_bottle"), ((0.24, -0.04, 0.0525), "juice_box")):
        t0 = time.perf_counter()
        q, ok = solve_grasp(xyz)
        ms = (time.perf_counter() - t0) * 1000
        worst = max(worst, ms)
        (gx, gy, gz), got = fk_grasp(q)
        print(
            f"solve_grasp {label:12s} {xyz} -> ok={ok}, grasp ({gx:.4f}, {gy:.4f}, {gz:.4f}), "
            f"pitch {math.degrees(got):+.1f} deg, {ms:.1f} ms"
        )
    print(f"slowest solve {worst:.1f} ms")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="SO-101 tip kinematics")
    parser.add_argument("--check", action="store_true", help="print fk(home_q) and solve a few targets")
    args = parser.parse_args()
    if args.check:
        _check()
    else:
        parser.print_help()
