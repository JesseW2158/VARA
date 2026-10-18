"""SimRobot: the Robot API of docs/interfaces.md on the MuJoCo scene (task A8).

This is the only place AI-written code gets a robot. A candidate or saved skill receives a `SimRobot` and
can call every public method of the Robot API (`scene`, `move_to`, `set_roll`, `open_gripper`,
`close_gripper`, `home`, `pick`, `place`, `place_in_zone`, `carry_to_hand`, `run_skill`); the runner uses
the extra helpers `run_entry(fn, args)`, `reset(scene)`, `set_skill(name, args)`, `trajectory()`,
`final_scene()`, `stats` and `events`. Nothing public hands out MuJoCo objects, file handles or the
network: `stats` and `events` are copies, `scene()` and `trajectory()` are rebuilt on every call, and every
other helper is underscore-prefixed, which the restricted namespace (sim/candidate_check.py) cannot name.

Guards against a cheating candidate: the runner calls the candidate through `run_entry(fn, args)`, inside
which (as inside `run_skill`) `reset()` and `set_skill()` raise RuntimeError; they also raise whenever AI
code is on the call stack (a frame whose globals carry a trimmed `__builtins__` dict, which is how
`sim.restricted.namespace()` and the runner's SAFE_BUILTINS look), so a runner that calls `run(robot,
args)` directly is covered too. Setting or deleting a public attribute on the instance
(`robot.trajectory = ...`, `robot.pwned = 1`) raises AttributeError. Once the 60 s limit has passed every
motion method raises SimTimeLimit again before moving, so a candidate that swallowed the first one cannot
keep going and `sim_s` stays at the limit. The runner should still build a fresh SimRobot per trial.

Physics: `sim/model/scene_ours.xml` (robot, table, tray, one free body per object, one mocap box per
obstacle, the hand box; names per sim/model/build_scene.py), timestep 1/600 s, 20 substeps per 30 Hz tick,
position actuators driven through `ctrl`. The commanded waypoints are what gets recorded; the physical arm
tracks them within a fraction of a degree, and contacts are measured on the physical state after every
substep (a 5 mm tray wall can be crossed inside one 33 ms tick at 0.2 m/s).

Grasp recipe (brief section 8.2, binding):
- `move_to` targets the `tip` site, as the contract says. `pick`, `place`, `place_in_zone` and
  `carry_to_hand` target the grasp center (site `grasp`, `arm.ik.solve_grasp` / `arm.ik.fk_grasp`),
  2.46 cm from `tip` toward the moving jaw, because only the moving jaw swings: an object centered on `tip`
  would sit inside the fixed finger. The contract sentence "attaches the object only if the tip is within
  1.0 cm of its center line" is therefore read as the grasp center.
- `pick` opens fully (1.0), transits when asked, moves the grasp center `approach` above the object with
  the roll already aligned to the object's yaw, descends on a straight vertical line to
  `max(h / 2, 0.02)`, closes, and rises by `approach`. `close_gripper` attaches the nearest object whose
  vertical center line is within 1.0 cm (xy) of the grasp center, with the grasp height between 20 % and
  80 % of the object's height and (boxes only) the jaw closing direction within 20 degrees (mod 180) of the
  object's width. The held object is exempt from robot contacts (contype 2 / conaffinity 2), its pose is
  rewritten after every substep as the tip pose composed with the offset captured at grasp time, with zero
  velocity; `open_gripper` opens the jaws first and then releases it. A released object keeps ignoring
  the robot until it has separated from the gripper (6 cm from the grasp center, or 1 s): a tilted gripper
  lets it slide along the lower finger on the way out, which is a release, not a collision. The contacts
  an object already has when it is grasped (resting on the table) are not collisions either.
- `place(x, y, release_z)` keeps the held object off the table: when the object's lowest point at the
  release pose (its actual tilt, from a kinematics pass) would be under PLACE_FLOOR_CLEARANCE above z = 0,
  the release height is raised by the deficit (the contract's default release_z = 0.05 puts the 10.5 cm
  juice box, held at mid-height, 2.5 mm into the table). The approach height moves up with it.
- `set_roll(angle)`: `wrist_roll = angle + shoulder_pan + ROLL_OFFSET`, chosen modulo pi nearest to the
  current roll inside the limits. ROLL_OFFSET = -1.529 rad was measured with `arm.ik.jaw_axes` on
  so101_new_calib.xml: at pan 0 with the jaws straight down, that roll puts the closing direction along the
  world y axis, i.e. across the width (`size[1]`) of an object at yaw 0. The pan term is `+ shoulder_pan`
  (not `- shoulder_pan` as the brief's section 4 wrote): the roll axis points down, so a positive roll
  turns the closing direction in the same sense as a positive pan; measured residual 0.4 degrees over
  pans of +-37 degrees and yaws of -0.6..0.3 rad.
- Choices the recipe makes with a forward pass of the model (no stepping, see `_pose_contacts`): `pick`
  takes, of the two rolls aligned with the yaw (any roll for a cylinder), the nearest one whose approach
  path and descent touch nothing; `place_in_zone` without `slot` takes the first free slot whose travel,
  approach and descent with the held object touch nothing, trying the roll turned by pi as well. When
  nothing is clean the nearest roll / first free slot is used and the physics counts what happens.
- Pitch: `arm.ik` relaxes an unreachable pitch to the next rung of its 15-degree ladder; `_goal` bisects
  between the request and that rung so every pose is as steep as the arm can reach (about 4 degrees), which
  keeps the wrist high over obstacles the arm leans across and a held object as upright as possible.

Motion: every move is planned before anything moves and recorded as commanded waypoints at 30 Hz
(schemas/trajectory.json, 6 decimals), trapezoidal profile, at most 0.999 * 2 degrees per joint per tick,
tip speed at most min(speed, 0.2) m/s measured with `arm.ik.fk_tip` between consecutive waypoints (a
joint-space sweep is sampled before it is timed; a straight-line leg profiles the target site, so its
finished path is measured on the tip and any step over the cap is split into joint-space sub-steps, see
`_retime`), `move_to` at the candidate's `speed` (a missing or unusable `speed` means the contract's
0.15), the pick, place, transit and carry legs at the 0.2 m/s cap, every waypoint clipped 3 degrees
inside the joint limits and its tip checked against the workspace box. A target on the boundary of the
box is nudged BOX_INSET (1 mm) inside before IK so the solution's tip does not land a hair outside. A
joint-space sweep whose tip leaves the box is re-planned as a straight Cartesian line of the target site
with IK per sample; when that fails too the move counts one `ik_failures` and returns False without
moving. Vertical pick, place and transit legs are straight lines from the start. `home()` tries the direct
sweep, then a vertical rise to a raised intermediate pose, then a straight line to the home tip. Every
trajectory this class records passes `arm.replay.validate`. A place_in_tray costs about 11 s of simulated
time.

Collisions: a contact counts when one geom belongs to the robot (or the held object) and the other to the
table, the tray, an obstacle, the hand box, another object, or another robot body (self-contacts count
too). Distinct new body pairs are counted per tick; the first is kept as `stats["first_collision"]`, body
names only ("gripper hit cereal_box", "held juice_box hit tray", "gripper hit hand"). `limit_hits` counts
commanded waypoints (before clipping) within 1 degree of a joint limit, including a `set_roll` that the
limits cannot satisfy. Sim time advances 1/30 s per tick; `close_gripper` costs 0.4 s, `open_gripper`
0.3 s; past 60 s `SimTimeLimit` (a BaseException) is raised from inside whatever method is running.

`carry_to_hand()` requires a held object. It moves the grasp center above the hand (or the handoff zone
when the hand is unseen), with the bottom of the held object 5 cm above the hand point: first a vertical
rise to a clearance height (the held object's longest half-extent plus CARRY_CLEARANCE above the tallest
object or obstacle still on the table), then a travel at that height to above the hand point, then a
straight descent. A height above the safe ceiling, an invalid rise or an unreachable travel aborts while
holding the object and counts an `ik_failures`; there is no direct-move fallback. Then it opens, backs
off 5 cm up, goes home, appends the event `handoff_done` and records only
`{"kind": "live", "primitive": "carry_to_hand", "object": name}`: the real primitive does the carry itself
and ends at home, so the sim motion of the carry is not recorded as path segments. The sim assumes the
user takes the released object (the real primitive waits for the hand): it leaves the scene, reported at
its last pose with `seen` false, which is what `moved` treats as "the user took it".

Incoming `seen` flags remain authoritative: simulation cannot rediscover an unseen marker. An unseen
object is neither a pick target nor a grasp candidate, and snapshots retain its last observed pose.
Its physical body, and every unseen obstacle, remain at their last known locations for collision checks
and clearance planning: a lost marker can mean occlusion, not removal. Only a handoff completed within
this simulation proves that an object was taken and permits parking its body away from the workspace.
The scene contract carries no removal provenance across independent runs, so reloading a post-handoff
snapshot conservatively restores that unseen object's collision body at its last reported location.

Scene note (measured, tests/test_sim_robot.py): config/fake_scene.json now has the cereal box at
(0.28, 0.07), the tea tin at (0.16, -0.02) and the handoff zone and hand at (0.28, -0.08); on that layout
the starter skills place every object in the tray and hand it over without a collision (20 of 20 scoring
perturbations for the juice box and the pill bottle; 18 of 20 for the tea tin, which, held 8 cm up, tilts
as the pitch relaxes on the rise to z 0.28 and grazes the juice box when the perturbation brings the two
within about 6 cm; relaxing the pitch later along the rise does not help, the reach forces it). A15
re-tapes the real layout and this file follows. With the box at its earlier spot (0.24, 0.11) every tray
slot's release pose put the forearm 14 to 27 mm inside the box, because the arm is nearly fully extended
to reach the tray and its forearm passes at z = 0.19 over the box's corner; whoever moves the box again
should keep it out of bearings 34 to 44 degrees at 0.17 to 0.25 m from the pan axis and at least 4 cm from
the gripper footprint at the handoff zone. The hand box (8 x 10 cm) at the handoff zone overlaps the juice
box's footprint at (0.24, -0.04): a seen hand there shoves the box 3 cm at reset.

`run_skill(name, args)` reads `skills_dir/<name>.json` (default `ROOT/skills`), validates it against
schemas/skill.json, loads the code with `sim.restricted.load_entry` (never its own exec) and calls
`run(self, args)`; a missing, invalid or rejected skill returns False and appends an event.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import mujoco
import numpy as np
from jsonschema import Draft202012Validator

from arm import ik
from arm.replay import MARGIN_RAD, MAX_STEP_RAD, load_box, load_limits
from sim.restricted import load_entry

__all__ = [
    "DT",
    "ROLL_OFFSET",
    "SIM_TIME_LIMIT",
    "SimRobot",
    "SimTimeLimit",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "sim" / "model" / "scene_ours.xml"
SCENE_SCHEMA_PATH = ROOT / "schemas" / "scene.json"
SKILL_SCHEMA_PATH = ROOT / "schemas" / "skill.json"
SKILLS_DIR = ROOT / "skills"

DT = 1.0 / 30.0
SUBSTEPS = 20
TIMESTEP = 1.0 / 600.0
SIM_TIME_LIMIT = 60.0
MAX_TICKS = int(round(SIM_TIME_LIMIT / DT))  # 1800 ticks = 60 s; the 1801st tick raises SimTimeLimit

MAX_TIP_SPEED = 0.2  # m/s, the contract's cap on `speed`
MIN_TIP_SPEED = 0.01
MAX_JOINT_STEP = 0.999 * MAX_STEP_RAD  # rad per tick (60 deg/s), a hair under validate()'s 2 degrees
LIMIT_HIT_RAD = math.radians(1.0)
LIMIT_EPS = 1e-6  # waypoints are clipped this far inside limits minus 3 degrees before rounding
BOX_EPS = 1e-6  # the tip must be this far inside the workspace box
BOX_INSET = 0.001  # m, a target on the boundary of the box is nudged this far inside before IK
RAMP_TICKS = 4  # ticks of acceleration and deceleration in the trapezoidal profile
SPEED_SAFETY = 0.995  # tip-speed sampling is chord-based; keep a little under the cap
SWEEP_SAMPLE_RAD = math.radians(0.5)  # resolution of the tip-speed measurement along a sweep
DEFAULT_SPEED = 0.15  # move_to's default, the contract's
LEG_SPEED = MAX_TIP_SPEED  # pick, place, transit and carry legs run at the cap (sim time is budgeted)
ROUND_DECIMALS = 6
PITCH_TOL = math.radians(3.0)  # the pitch tolerance of arm.ik
PITCH_BISECTIONS = 2  # refinement steps between two ladder rungs (15 degrees -> about 4)
PATH_CHECK_STRIDE = 3  # waypoints between two pose checks when a path is screened for contacts

ROLL_OFFSET = -1.529  # rad, measured (see the module docstring)
GRIPPER_CLOSED_RAD = -0.17  # our gripper value 0.0
GRIPPER_RANGE_RAD = 1.17  # our 1.0 is -0.17 + 1.17 = 1.0 rad (9.6 cm tip-to-tip gap)
GAP_AT_ZERO = 0.018  # m, inner jaw gap at gripper angle 0 (sim/model/README.md)
GAP_PER_RAD = 0.078  # m per rad of gripper angle
CLOSE_TICKS = 12  # 0.4 s
OPEN_TICKS = 9  # 0.3 s

ATTACH_XY = 0.010  # m, grasp center to the object's vertical center line
ATTACH_FRACTION = (0.2, 0.8)  # of the object's height
ATTACH_YAW = math.radians(20.0)
SLOT_FREE_RADIUS = 0.03  # m, a slot is free when no object center is within this distance
RELEASE_CLEARANCE = 0.01  # m above the wall for the bottom of the held object in place_in_zone
HAND_CLEARANCE = 0.05  # m, bottom of the held object above the hand point in carry_to_hand
HAND_BACKOFF = 0.05  # m, rise after releasing into the hand
CARRY_CLEARANCE = 0.02  # m, between the held object's longest half-extent and the tallest thing on the table
CEILING_MARGIN = 0.01  # m, the carry travels at most this far under the top of the workspace box
CARRY_RISE_DONE = 0.005  # m, the measured grasp center may finish this far below the carry height
CARRY_LINE_TOL = 0.006  # m, permitted sideways rise deviation / travel-height deficit, as on the live arm
PLACE_FLOOR_CLEARANCE = 0.005  # m, place() keeps the held object's lowest point this far above the table
RETIME_PASSES = 3  # a split step of a curved tip path can still be a hair long: measure again
HOME_RAISE_Z = (0.20, 0.25, 0.15, 0.12)  # intermediate tip heights tried when home() leaves the box
HAND_PARKED_Z = -1.0
RELEASE_SEPARATION = 0.06  # m from the grasp center at which a released object collides with the robot again
RELEASE_GRACE_TICKS = 30  # or after this many ticks (1 s), whichever comes first
DROP_SETTLE_TICKS = 15  # 0.5 s for an object dropped by pick() while holding to land before poses are read
MAX_SKILL_DEPTH = 8
SKILL_NAME_RE = r"^[a-z][a-z0-9_]{1,40}$"

JOINTS = tuple(ik.JOINTS)
GRIPPER_JOINT = "gripper"
TIP_SITE = ik.TIP_SITE
GRASP_SITE = ik.GRASP_SITE
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
BODY_LABELS = {"moving_jaw_so101_v1": "gripper", "world": "table"}
HELD_CONTYPE = (2, 2)
FREE_CONTYPE = (1, 3)

_KIND_NONE, _KIND_ROBOT, _KIND_ENV, _KIND_OBJECT = 0, 1, 2, 3

_SCENE_VALIDATOR = Draft202012Validator(json.loads(SCENE_SCHEMA_PATH.read_text(encoding="utf-8")))
_SKILL_VALIDATOR = Draft202012Validator(json.loads(SKILL_SCHEMA_PATH.read_text(encoding="utf-8")))

Vec3 = tuple[float, float, float]


class SimTimeLimit(BaseException):
    """Raised from inside whatever Robot API method is running once 60 s of simulated time have passed.

    A BaseException so that a candidate's `except Exception` cannot swallow it.
    """


def _ai_code_on_the_stack() -> bool:
    """True when a frame on the call stack runs AI code: its globals carry a trimmed `__builtins__` dict
    (`sim.restricted.namespace()` and the runner's SAFE_BUILTINS both drop `open`)."""
    frame = sys._getframe(1)
    while frame is not None:
        builtins_ = frame.f_globals.get("__builtins__")
        if isinstance(builtins_, dict) and "open" not in builtins_:
            return True
        frame = frame.f_back
    return False


def _finite(value: Any) -> float | None:
    """`value` as a finite float, or None (AI code passes whatever it likes)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _yaw_of_quat(quat: np.ndarray) -> float:
    """Heading of a body's x axis (the length direction of a box) projected on the table."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    mat = mat.reshape(3, 3)
    x_axis = mat[:, 0]
    if math.hypot(x_axis[0], x_axis[1]) < 1e-6:  # standing on an end: use the y axis instead
        y_axis = mat[:, 1]
        return math.atan2(y_axis[1], y_axis[0]) - math.pi / 2
    return math.atan2(x_axis[1], x_axis[0])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _angle_diff_mod(a: float, b: float, period: float) -> float:
    """Smallest absolute difference between two angles modulo `period`."""
    d = (a - b) % period
    return min(d, period - d)


def _profile(distance: float, v_max: float, ramp: int = RAMP_TICKS) -> list[float]:
    """Cumulative positions per tick of a trapezoidal profile covering `distance`, at most `v_max` per
    tick, accelerating over `ramp` ticks. Ends exactly at `distance`; empty when there is nothing to do."""
    if distance <= 1e-12 or v_max <= 0.0:
        return []
    accel = v_max / ramp
    ramp_steps = [accel * (i + 1) for i in range(ramp)]
    ramp_length = sum(ramp_steps)
    if distance >= 2.0 * ramp_length:
        cruise = int(math.ceil((distance - 2.0 * ramp_length) / v_max))
        steps = ramp_steps + [v_max] * cruise + ramp_steps[::-1]
    else:
        n = min(ramp, max(1, int(math.ceil(math.sqrt(distance / accel)))))
        half = [accel * (i + 1) for i in range(n)]
        steps = half + half[::-1]
    total = sum(steps)
    scale = distance / total  # <= 1 by construction, so no step grows past v_max
    out: list[float] = []
    acc = 0.0
    for step in steps:
        acc += step * scale
        out.append(acc)
    out[-1] = distance
    return out


class SimRobot:
    """The Robot API on the MuJoCo scene. See the module docstring for the recipes and the counters."""

    def __init__(
        self,
        scene: dict,
        *,
        skills_dir: pathlib.Path | None = None,
        record: bool = True,
        model_path: pathlib.Path | str | None = None,
    ) -> None:
        self._model_path = pathlib.Path(model_path) if model_path is not None else MODEL_PATH
        self._skills_dir = pathlib.Path(skills_dir) if skills_dir is not None else SKILLS_DIR
        self._record = bool(record)
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._model.opt.timestep = TIMESTEP
        self._data = mujoco.MjData(self._model)
        self._limits = load_limits()
        self._box = load_box()
        self._lo = np.array([self._limits[j][0] for j in JOINTS])
        self._hi = np.array([self._limits[j][1] for j in JOINTS])
        self._home_q = np.array(ik.home_q(), dtype=float)
        self._index_model()
        self._skill = "unnamed"
        self._args: dict = {}
        self._depth = 0
        self.reset(scene)

    def __setattr__(self, name: str, value: Any) -> None:
        # AI code may neither shadow a public method on the instance nor add attributes to it
        if not name.startswith("_"):
            raise AttributeError(f"SimRobot.{name} cannot be set")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if not name.startswith("_"):
            raise AttributeError(f"SimRobot.{name} cannot be deleted")
        object.__delattr__(self, name)

    # ------------------------------------------------------------------ model bookkeeping

    def _name2id(self, kind: mujoco.mjtObj, name: str) -> int:
        i = mujoco.mj_name2id(self._model, kind, name)
        if i < 0:
            raise RuntimeError(f"{self._model_path} has no {kind.name.lower()[6:]} named {name!r}")
        return i

    def _index_model(self) -> None:
        m = self._model
        joint = mujoco.mjtObj.mjOBJ_JOINT
        self._arm_qadr = np.array([m.jnt_qposadr[self._name2id(joint, j)] for j in JOINTS])
        self._arm_dofadr = np.array([m.jnt_dofadr[self._name2id(joint, j)] for j in JOINTS])
        self._arm_act = np.array([self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
        grip_joint = self._name2id(joint, GRIPPER_JOINT)
        self._grip_qadr = int(m.jnt_qposadr[grip_joint])
        self._grip_dofadr = int(m.jnt_dofadr[grip_joint])
        self._grip_act = self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)
        self._tip = self._name2id(mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        self._grasp = self._name2id(mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
        body = mujoco.mjtObj.mjOBJ_BODY
        self._robot_bodies = {self._name2id(body, n) for n in ROBOT_BODIES}
        # free-joint bodies are objects; mocap bodies are the tray, the obstacles and the hand
        self._objects: dict[str, dict] = {}
        self._mocap: dict[str, int] = {}
        self._kind = np.full(m.nbody, _KIND_NONE, dtype=np.int64)
        self._label: list[str] = []
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, body, b) or f"body{b}"
            self._label.append(BODY_LABELS.get(name, name))
            if b in self._robot_bodies:
                self._kind[b] = _KIND_ROBOT
            elif m.body_mocapid[b] >= 0:
                self._mocap[name] = int(m.body_mocapid[b])
                self._kind[b] = _KIND_ENV
            elif b == 0:
                self._kind[b] = _KIND_ENV  # the world body holds the table geom
            elif m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
                j = int(m.body_jntadr[b])
                g = int(m.body_geomadr[b])
                self._objects[name] = {
                    "body": b,
                    "geom": g,
                    "qadr": int(m.jnt_qposadr[j]),
                    "dofadr": int(m.jnt_dofadr[j]),
                    "half": m.geom_size[g].copy(),
                    "cylinder": bool(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER),
                }
                self._kind[b] = _KIND_OBJECT
        self._geom_body = m.geom_bodyid.copy()
        self._tray_body = self._mocap.get("tray")
        self._hand_body = self._mocap.get("hand")

    def _object_half_height(self, name: str) -> float:
        half = self._objects[name]["half"]
        return float(half[1] if self._objects[name]["cylinder"] else half[2])

    def _object_width(self, name: str) -> float:
        half = self._objects[name]["half"]
        return 2.0 * float(half[0] if self._objects[name]["cylinder"] else half[1])

    def _hold_value(self, name: str) -> float:
        """Our gripper value at which the inner jaw gap equals the object's width (the jaws stop there)."""
        angle = (self._object_width(name) - GAP_AT_ZERO) / GAP_PER_RAD
        return min(1.0, max(0.0, (angle - GRIPPER_CLOSED_RAD) / GRIPPER_RANGE_RAD))

    # ------------------------------------------------------------------ runner helpers (public, no MuJoCo)

    def _runner_only(self, what: str) -> None:
        """`reset()` and `set_skill()` belong to the runner: refused inside `run_entry` / `run_skill` and
        whenever AI code is on the call stack, however the runner called the candidate."""
        if self._depth > 0 or _ai_code_on_the_stack():
            raise RuntimeError(f"{what} is not available to AI code")

    def _check_time(self) -> None:
        """Past the limit every motion method raises again before moving anything."""
        if self._ticks > MAX_TICKS:
            raise SimTimeLimit(f"simulated time passed {SIM_TIME_LIMIT:.0f} s")

    def run_entry(self, fn: Callable[..., Any], args: dict | None) -> bool:
        """Run a candidate's entry function `fn(robot, args)` (from `sim.restricted.load_entry`) for the
        runner: as inside `run_skill`, `reset()` and `set_skill()` raise RuntimeError while it runs; the
        result is coerced to bool; SimTimeLimit and any exception propagate."""
        if not callable(fn):
            raise TypeError("run_entry needs a callable entry function")
        self._check_time()
        self._depth += 1
        try:
            result = fn(self, copy.deepcopy(dict(args)) if args else {})
        finally:
            self._depth -= 1
        return bool(result)

    def reset(self, scene: dict) -> None:
        """Place everything as the scene says, put the arm at home_q with the gripper open and zero the
        counters, the events, the trajectory and the sim clock. Raises ValueError on an invalid scene or an
        object the model does not have, RuntimeError when called from inside `run_entry` or a skill."""
        self._runner_only("reset()")
        errors = sorted(_SCENE_VALIDATOR.iter_errors(scene), key=lambda e: e.json_path)
        if errors:
            raise ValueError(f"invalid scene: {errors[0].json_path}: {errors[0].message}")
        unknown = [n for n in scene["objects"] if n not in self._objects]
        if unknown:
            raise ValueError(f"scene names objects the model does not have: {unknown}")
        unknown = [n for n in scene["obstacles"] if n not in self._mocap]
        if unknown:
            raise ValueError(f"scene names obstacles the model does not have: {unknown}")
        self._scene_in = copy.deepcopy(scene)
        m, d = self._model, self._data
        mujoco.mj_resetData(m, d)
        self._release_held(restore_only=True)
        self._held: str | None = None
        self._held_offset: tuple[np.ndarray, np.ndarray] | None = None
        d.qpos[self._arm_qadr] = self._home_q
        d.ctrl[self._arm_act] = self._home_q
        self._grip = 1.0
        d.qpos[self._grip_qadr] = self._grip_rad(1.0)
        d.ctrl[self._grip_act] = self._grip_rad(1.0)
        parked = 0
        for name, info in self._objects.items():
            item = scene["objects"].get(name)
            hz = self._object_half_height(name)
            if item is None:  # not in this scene: park it on the far corner of the table
                x, y, yaw = 0.6, -0.4 + 0.1 * parked, 0.0
                parked += 1
            else:
                x, y, yaw = float(item["x"]), float(item["y"]), float(item["yaw"])
            adr, dof = info["qadr"], info["dofadr"]
            d.qpos[adr : adr + 3] = (x, y, hz)
            d.qpos[adr + 3 : adr + 7] = _yaw_quat(yaw)
            d.qvel[dof : dof + 6] = 0.0
        for name, item in scene["obstacles"].items():
            k = self._mocap[name]
            d.mocap_pos[k] = (float(item["x"]), float(item["y"]), float(item["h"]) / 2.0)
            d.mocap_quat[k] = _yaw_quat(float(item["yaw"]))
        if self._tray_body is not None:
            tray = scene["zones"]["tray"]
            d.mocap_pos[self._tray_body] = (float(tray["x"]), float(tray["y"]), 0.0)
            d.mocap_quat[self._tray_body] = (1.0, 0.0, 0.0, 0.0)
        if self._hand_body is not None:
            hand = scene["hand"]
            z = float(hand["z"]) if hand["seen"] else HAND_PARKED_Z
            d.mocap_pos[self._hand_body] = (float(hand["x"]), float(hand["y"]), z)
            d.mocap_quat[self._hand_body] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        self._q_cmd = self._home_q.copy()
        self._ticks = 0
        self._stats: dict[str, Any] = {
            "collisions": 0,
            "limit_hits": 0,
            "ik_failures": 0,
            "sim_s": 0.0,
            "first_collision": None,
        }
        self._events: list[str] = []
        self._segments: list[dict] = []
        self._suspended = 0
        self._contact_pairs: set[tuple[int, int]] = set()
        self._collision_log: list[str] = []
        self._released: tuple[str, int] | None = None
        self._taken: dict[str, tuple[float, float, float]] = {}

    def set_skill(self, name: str, args: dict | None) -> None:
        """Name and arguments written into the recorded trajectory (runner only, like `reset`)."""
        self._runner_only("set_skill()")
        if not isinstance(name, str) or not name:
            raise ValueError("skill name must be a non-empty string")
        self._skill = name
        self._args = copy.deepcopy(dict(args)) if args else {}

    @property
    def stats(self) -> dict:
        """Copy of the counters: collisions, limit_hits, ik_failures, sim_s, first_collision."""
        return dict(self._stats)

    @property
    def events(self) -> list[str]:
        """Copy of the event log (`handoff_done` and diagnostics)."""
        return list(self._events)

    def trajectory(self) -> dict:
        """The recorded trajectory (schemas/trajectory.json). With nothing recorded it holds one path
        segment with the current commanded pose, so it is still valid."""
        segments = copy.deepcopy(self._segments)
        if not segments:
            segments = [{"kind": "path", "q": [self._rounded(self._q_cmd)]}]
        return {
            "version": 1,
            "skill": self._skill,
            "args": copy.deepcopy(self._args),
            "dt": DT,
            "joints": list(JOINTS),
            "segments": segments,
        }

    def final_scene(self) -> dict:
        """The scene as the physics has it now (same as `scene()`), for `final_scene.json`."""
        return self.scene()

    # ------------------------------------------------------------------ Robot API

    def scene(self) -> dict:
        """Current scene state (schemas/scene.json) rebuilt from the physics: objects from their free
        joints, obstacles and the hand from the mocap bodies, zones copied from the input scene.
        Unseen markers retain the input visibility and last observed pose, never a synthetic detection.
        """
        d = self._data
        src = self._scene_in
        objects: dict[str, dict] = {}
        for name, item in src["objects"].items():
            taken = self._taken.get(name)
            if taken is not None:
                x, y, yaw = taken
            elif item["seen"]:
                x, y, _z, yaw = self._object_pose(name)
            else:
                x, y, yaw = float(item["x"]), float(item["y"]), float(item["yaw"])
            objects[name] = {
                "marker": int(item["marker"]),
                "x": x,
                "y": y,
                "yaw": yaw,
                "h": float(item["h"]),
                "size": [float(item["size"][0]), float(item["size"][1])],
                "seen": bool(item["seen"]) and taken is None,
            }
        obstacles: dict[str, dict] = {}
        for name, item in src["obstacles"].items():
            k = self._mocap[name]
            obstacles[name] = {
                "marker": int(item["marker"]),
                "x": float(d.mocap_pos[k][0]),
                "y": float(d.mocap_pos[k][1]),
                "yaw": _yaw_of_quat(d.mocap_quat[k]),
                "h": float(item["h"]),
                "size": [float(item["size"][0]), float(item["size"][1])],
                "seen": bool(item["seen"]),
            }
        hand_in = src["hand"]
        hand = {
            "marker": int(hand_in["marker"]),
            "x": float(hand_in["x"]),
            "y": float(hand_in["y"]),
            "z": float(hand_in["z"]),
            "seen": bool(hand_in["seen"]),
        }
        if self._hand_body is not None:
            hand["x"] = float(d.mocap_pos[self._hand_body][0])
            hand["y"] = float(d.mocap_pos[self._hand_body][1])
            if hand["seen"]:
                hand["z"] = float(d.mocap_pos[self._hand_body][2])
        return {
            "t": float(src["t"]) + self._stats["sim_s"],
            "objects": objects,
            "obstacles": obstacles,
            "hand": hand,
            "zones": copy.deepcopy(src["zones"]),
        }

    def move_to(
        self, x: float, y: float, z: float, pitch: float = -math.pi / 2, speed: float = DEFAULT_SPEED
    ) -> bool:
        """Tip to (x, y, z) with the jaw axis at `pitch` from horizontal (-pi/2 straight down; IK relaxes it
        toward horizontal when unreachable). False, without motion, when the target is outside the
        workspace box or unreachable."""
        self._check_time()
        fx, fy, fz, p = _finite(x), _finite(y), _finite(z), _finite(pitch)
        if fx is None or fy is None or fz is None or p is None or not self._in_box((fx, fy, fz)):
            return False
        return self._move(TIP_SITE, (fx, fy, fz), p, self._speed(speed))

    def set_roll(self, angle: float) -> None:
        """Rotate the gripper about its own axis so the jaws close across the width of an object at yaw
        `angle` (radians), using the pan of the current pose. Recorded as a joint-space move."""
        self._check_time()
        a = _finite(angle)
        if a is None:
            return
        roll = self._roll_for(a, float(self._q_cmd[0]))
        q_goal = self._q_cmd.copy()
        q_goal[4] = roll
        path = self._sweep(self._q_cmd, q_goal, LEG_SPEED)
        if path is not None:
            self._run_path(path)

    def open_gripper(self) -> None:
        """Open fully over 0.3 s; a held object is released once the jaws are open."""
        self._check_time()
        held = self._held
        self._grip_ramp(1.0, OPEN_TICKS)
        if held is not None:
            self._release_held()
        segment: dict[str, Any] = {"kind": "grip", "action": "open"}
        if held is not None:
            segment["object"] = held
        self._append_segment(segment)

    def close_gripper(self) -> bool:
        """Close over 0.4 s. Returns True when an object was attached (see the module docstring for the
        attach test); the jaws then stop at the object's width."""
        self._check_time()
        name = self._attach_candidate()
        if name is not None:
            self._attach(name)
            target = self._hold_value(name)
        else:
            target = 0.0
        self._grip_ramp(target, CLOSE_TICKS)
        segment: dict[str, Any] = {"kind": "grip", "action": "close", "expect_held": name is not None}
        if name is not None:
            segment["object"] = name
        self._append_segment(segment)
        return name is not None

    def home(self) -> None:
        """Joint-space move to home_q (config/scene.yaml), via a raised pose or a straight line when the
        direct sweep would take the tip out of the workspace box. Counts an IK failure and stays put when
        no plan keeps the tip inside."""
        self._check_time()
        plan = self._plan_home()
        if plan is None:
            self._stats["ik_failures"] += 1
            return
        for path in plan:
            self._run_path(path)

    def pick(self, name: str, approach: float = 0.08, transit_z: float | None = None) -> bool:
        """Open, (transit), move the grasp center `approach` above the object with the jaws aligned to its
        yaw, descend to max(h / 2, 0.02), close, rise by `approach`. True when the object is held."""
        self._check_time()
        if not isinstance(name, str) or name not in self._objects or name not in self._scene_in["objects"]:
            return False
        if name in self._taken or not self._scene_in["objects"][name]["seen"]:
            return False
        approach_m = _finite(approach)
        if approach_m is None:
            return False
        was_holding = self._held is not None
        if was_holding or self._grip < 1.0:
            self.open_gripper()
        if was_holding:  # the dropped object lands (it may be the target) before its pose is read
            for _ in range(DROP_SETTLE_TICKS):
                self._tick(record=False)
        x, y, _z, yaw = self._object_pose(name)
        grasp_z = max(self._object_half_height(name), 0.02)
        if transit_z is not None and not self._transit(transit_z, (x, y)):
            return False
        above = (x, y, grasp_z + approach_m)
        q_probe, ok = self._goal(GRASP_SITE, above, -math.pi / 2, self._q_cmd)
        if not ok:
            self._stats["ik_failures"] += 1
            return False
        cylinder = self._objects[name]["cylinder"]
        roll = self._clear_roll(yaw, float(q_probe[0]), above, (x, y, grasp_z), free=cylinder)
        if not self._move(GRASP_SITE, above, -math.pi / 2, LEG_SPEED, roll=roll):
            return False
        if not self._move(GRASP_SITE, (x, y, grasp_z), -math.pi / 2, LEG_SPEED, straight=True):
            return False
        held = self.close_gripper()
        self._move(GRASP_SITE, (x, y, grasp_z + approach_m), -math.pi / 2, LEG_SPEED, straight=True)
        return held

    def place(
        self,
        x: float,
        y: float,
        release_z: float = 0.05,
        approach: float = 0.08,
        transit_z: float | None = None,
    ) -> bool:
        """With a held object: (transit), grasp center `approach` above (x, y, release_z), descend straight
        to release_z (raised when the held object's lowest point would come within PLACE_FLOOR_CLEARANCE
        of the table), open, rise by `approach`. False without a held object or when a leg fails."""
        self._check_time()
        if self._held is None:
            return False
        xx, yy, rz, ap = _finite(x), _finite(y), _finite(release_z), _finite(approach)
        if xx is None or yy is None or rz is None or ap is None:
            return False
        rz = self._off_the_table(xx, yy, rz)
        return self._place(xx, yy, rz, ap, transit_z, None)

    def _off_the_table(self, x: float, y: float, rz: float) -> float:
        """`rz` raised so that the held object's lowest point at the release pose (x, y, rz), with the tilt
        the pose gives it, stays PLACE_FLOOR_CLEARANCE above z = 0."""
        for _ in range(2):  # a raised pose tilts a little differently: measure once more
            q_rel, ok = self._goal(GRASP_SITE, (x, y, rz), -math.pi / 2, self._q_cmd)
            if not ok:
                return rz
            deficit = PLACE_FLOOR_CLEARANCE - self._held_lowest(q_rel)
            if deficit <= 1e-4:
                return rz
            rz += deficit
        return rz

    def _place(
        self,
        x: float,
        y: float,
        rz: float,
        ap: float,
        transit_z: Any,
        roll: float | None,
        *,
        risen: bool = False,
    ) -> bool:
        """The place legs: (rise, travel at transit_z), above by `ap`, straight down to `rz`, open, rise.
        `roll` is carried by the first leg that moves sideways; `risen` says the rise leg was done."""
        if transit_z is not None:
            if not risen and not self._rise(transit_z):
                return False
            if not self._travel((x, y), transit_z, roll):
                return False
            roll = None
        if not self._move(GRASP_SITE, (x, y, rz + ap), -math.pi / 2, LEG_SPEED, roll=roll):
            return False
        if not self._move(GRASP_SITE, (x, y, rz), -math.pi / 2, LEG_SPEED, straight=True):
            return False
        self.open_gripper()
        self._move(GRASP_SITE, (x, y, rz + ap), -math.pi / 2, LEG_SPEED, straight=True)
        return True

    def place_in_zone(self, zone: str, slot: int | None = None, transit_z: float | None = None) -> bool:
        """`place` into the zone's next free slot (no object center within 0.03 m), or `slot` when given,
        releasing at wall + h_held / 2 + 0.01 above the zone floor. Without `slot`, free slots whose
        approach path or release pose would hit something are passed over (see `_clear_slot`). A `slot`
        that is not an index into the zone's slots (negative, too large, not a number) returns False."""
        self._check_time()
        if self._held is None:
            return False
        zones = self._scene_in["zones"]
        if not isinstance(zone, str) or zone not in zones:
            return False
        spec = zones[zone]
        slots = [(float(s[0]), float(s[1])) for s in spec.get("slots", [[spec["x"], spec["y"]]])]
        wall = float(spec.get("wall", 0.0))
        release_z = wall + self._object_half_height(self._held) + RELEASE_CLEARANCE
        if slot is not None:
            try:
                index = int(slot)
            except (TypeError, ValueError, OverflowError):
                return False
            if not 0 <= index < len(slots):
                return False
            target = slots[index]
            return self._place(target[0], target[1], release_z, 0.08, transit_z, None)
        free = [s for s in slots if self._slot_free(s)]
        if not free:
            return False
        if transit_z is not None and not self._rise(transit_z):
            return False
        target, roll = self._clear_slot(free, release_z, transit_z)
        return self._place(target[0], target[1], release_z, 0.08, transit_z, roll, risen=True)

    def carry_to_hand(self) -> bool:
        """With a held object: rise to the carry clearance height, travel above the hand (or the handoff
        zone when the hand is unseen), descend, open, back off 5 cm up, home, event `handoff_done`, one
        `live` segment recorded. A failed clearance, rise or travel returns False while still holding the
        object; failed backoff/home never logs handoff_done. False without a held object.
        """
        self._check_time()
        if self._held is None:
            return False
        name = self._held
        hand = self._scene_in["hand"]
        hand_target = (float(hand["x"]), float(hand["y"]), float(hand["z"]) + HAND_CLEARANCE)
        if hand["seen"] and self._in_box(hand_target):
            point = (float(hand["x"]), float(hand["y"]), float(hand["z"]))
        else:
            zone = self._scene_in["zones"]["handoff"]
            point = (float(zone["x"]), float(zone["y"]), float(zone["z"]))
        target = (point[0], point[1], point[2] + HAND_CLEARANCE + self._object_half_height(name))
        self._suspended += 1
        try:
            if not self._carry_high((target[0], target[1]), target[2]):
                self._stats["ik_failures"] += 1
                return False
            if not self._move(GRASP_SITE, target, -math.pi / 2, LEG_SPEED, straight=True):
                return False
            if math.dist(ik.fk_grasp(self._physical_q())[0], target) > CARRY_LINE_TOL:
                self._stats["ik_failures"] += 1
                return False
            self.open_gripper()
            self._take(name)
            if not self._move(
                GRASP_SITE,
                (target[0], target[1], target[2] + HAND_BACKOFF),
                -math.pi / 2,
                LEG_SPEED,
                straight=True,
            ):
                return False
            self.home()
            if not np.allclose(self._q_cmd, self._home_q, atol=1e-6):
                return False
        finally:
            self._suspended -= 1
        self._events.append("handoff_done")
        self._append_segment({"kind": "live", "primitive": "carry_to_hand", "object": name})
        return True

    def _carry_high(self, xy: tuple[float, float], target_z: float) -> bool:
        """The carry's rise and travel: up to the clearance height at the current xy, then across to above
        `xy`. False on unavailable clearance, a failed rise or failed travel; the caller must not descend.
        The same clearance/line tolerances as the live primitive apply to the planned and measured rise.
        """
        tz = self._carry_clearance()
        if tz is None:
            return False
        tz = max(tz, target_z)
        if not math.isfinite(tz) or tz > float(self._box["z"][1]) - CEILING_MARGIN:
            return False
        (gx, gy, gz), _ = ik.fk_grasp(self._q_cmd)
        if gz < tz - CARRY_RISE_DONE:
            rise = self._plan(GRASP_SITE, (gx, gy, tz), -math.pi / 2, LEG_SPEED, self._q_cmd, straight=True)
            if rise is None:
                return False
            for q in rise:
                point, _ = ik.fk_grasp(q)
                if math.hypot(point[0] - gx, point[1] - gy) > CARRY_LINE_TOL or point[2] < gz - 0.01:
                    return False
            self._run_path(rise)
            measured, _ = ik.fk_grasp(self._physical_q())
            if (
                measured[2] < tz - CARRY_RISE_DONE
                or math.hypot(measured[0] - gx, measured[1] - gy) > CARRY_LINE_TOL
            ):
                return False
            gz = ik.fk_grasp(self._q_cmd)[0][2]
        travel = self._plan(
            GRASP_SITE, (xy[0], xy[1], max(tz, gz)), -math.pi / 2, LEG_SPEED, self._q_cmd, straight=True
        )
        if travel is None or any(ik.fk_grasp(q)[0][2] < tz - CARRY_LINE_TOL for q in travel):
            return False
        self._run_path(travel)
        measured, _ = ik.fk_grasp(self._physical_q())
        if (
            measured[2] < tz - CARRY_LINE_TOL
            or math.hypot(measured[0] - xy[0], measured[1] - xy[1]) > CARRY_LINE_TOL
        ):
            return False
        return True

    def _carry_clearance(self) -> float | None:
        """Grasp-center height at which the held object (its longest half-extent, whatever its tilt) clears
        the tallest object or obstacle still on the table by CARRY_CLEARANCE. None when that height cannot
        fit below the safe ceiling; clamping it would hide a clearance failure from candidate scoring.
        """
        tallest = max([float(item["h"]) for item in self._scene_in["obstacles"].values()] + [0.0])
        for name in self._scene_in["objects"]:
            if name != self._held and name not in self._taken:
                tallest = max(tallest, 2.0 * self._object_half_height(name))
        info = self._objects[self._held]
        half = info["half"]
        reach = math.hypot(half[0], half[1]) if info["cylinder"] else float(np.linalg.norm(half))
        height = tallest + reach + CARRY_CLEARANCE
        return (
            height if math.isfinite(height) and height <= float(self._box["z"][1]) - CEILING_MARGIN else None
        )

    def run_skill(self, name: str, args: dict | None) -> bool:
        """Run the saved skill `skills_dir/<name>.json` through sim.restricted; False (with an event) when
        the skill is missing, invalid, rejected or nested too deep."""
        self._check_time()
        if not isinstance(name, str) or re.match(SKILL_NAME_RE, name) is None:
            self._events.append(f"run_skill: bad skill name {name!r}")
            return False
        if self._depth >= MAX_SKILL_DEPTH:
            self._events.append(f"run_skill: {name} nested too deep")
            return False
        path = self._skills_dir / f"{name}.json"
        try:
            skill = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._events.append(f"run_skill: no skill {name}")
            return False
        if any(True for _ in _SKILL_VALIDATOR.iter_errors(skill)):
            self._events.append(f"run_skill: invalid skill file {name}")
            return False
        try:
            entry = load_entry(skill["code"], "run", filename=f"<skill {name}>")
        except ValueError as exc:
            self._events.append(f"run_skill: {name} rejected: {exc}")
            return False
        self._depth += 1
        try:
            result = entry(self, copy.deepcopy(dict(args)) if args else {})
        finally:
            self._depth -= 1
        return bool(result)

    # ------------------------------------------------------------------ planning

    def _speed(self, speed: Any) -> float:
        """`move_to`'s speed: the contract's default when missing or unusable, else clipped to the cap."""
        s = _finite(speed)
        if s is None:
            return DEFAULT_SPEED
        return min(MAX_TIP_SPEED, max(MIN_TIP_SPEED, s))

    def _inset(self, xyz: Vec3) -> Vec3:
        """A target inside the workspace box nudged BOX_INSET off its faces (the IK solution for a target
        on a face lands a hair outside, which validate() rejects); a target outside is left alone."""
        if not self._in_box(xyz):
            return xyz
        box = self._box
        x, y, z = (
            min(max(float(v), float(box[a][0]) + BOX_INSET), float(box[a][1]) - BOX_INSET)
            for a, v in zip("xyz", xyz, strict=True)
        )
        return (x, y, z)

    def _in_box(self, p: Sequence[float], eps: float = 0.0) -> bool:
        box = self._box
        return all(
            float(box[axis][0]) + eps <= float(v) <= float(box[axis][1]) - eps
            for axis, v in zip("xyz", p, strict=True)
        )

    def _solver(self, site: str):
        return ik.solve if site == TIP_SITE else ik.solve_grasp

    def _goal(self, site: str, xyz: Vec3, pitch: float, q0: Sequence[float]) -> tuple[list[float], bool]:
        """IK for `site` at `xyz`, as steep as the arm can reach: `arm.ik` relaxes an unreachable pitch to
        the next rung of its 15-degree ladder, so when it did, bisect between the request and that rung
        (each probe asks for an exact pitch first) and keep the steepest success. A higher wrist clears
        obstacles the arm leans over and keeps a held object more upright."""
        solver = self._solver(site)
        q, ok = solver(xyz, pitch, q0)
        if not ok:
            return q, False
        got = self._site_fk(site, q)[1]
        if abs(got - pitch) <= PITCH_TOL:
            return q, True
        unreachable, reachable = pitch, got
        for _ in range(PITCH_BISECTIONS):
            mid = 0.5 * (unreachable + reachable)
            q_mid, ok = solver(xyz, mid, q0)
            if not ok:
                break
            got = self._site_fk(site, q_mid)[1]
            if abs(got - mid) <= PITCH_TOL:
                reachable, q = got, q_mid
            else:
                unreachable = mid
        return q, True

    def _site_fk(self, site: str, q: Sequence[float]) -> tuple[Vec3, float]:
        return ik.fk(q) if site == TIP_SITE else ik.fk_grasp(q)

    def _finish(self, q: np.ndarray) -> list[float]:
        """Clip a waypoint 3 degrees (plus epsilon) inside the limits and round it to 6 decimals."""
        clipped = np.clip(q, self._lo + MARGIN_RAD + LIMIT_EPS, self._hi - MARGIN_RAD - LIMIT_EPS)
        return self._rounded(clipped)

    @staticmethod
    def _rounded(q: Sequence[float]) -> list[float]:
        return [round(float(v), ROUND_DECIMALS) for v in q]

    def _tip_inside(self, path: list[np.ndarray]) -> bool:
        return all(self._in_box(ik.fk_tip(self._finish(q)), BOX_EPS) for q in path)

    def _sweep(self, q_from: np.ndarray, q_to: np.ndarray, speed: float) -> list[np.ndarray] | None:
        """Joint-space waypoints from q_from (excluded) to q_to with the trapezoidal profile, or None when
        the tip of some waypoint leaves the workspace box. An empty list means nothing to do."""
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        delta = q_to - q_from
        biggest = float(np.max(np.abs(delta)))
        if biggest < 1e-9:
            return []
        ds = MAX_JOINT_STEP / biggest
        n = min(400, max(8, int(math.ceil(biggest / SWEEP_SAMPLE_RAD))))
        samples = np.linspace(0.0, 1.0, n + 1)
        tips = np.array([ik.fk_tip(q_from + s * delta) for s in samples])
        rate = float(np.max(np.linalg.norm(np.diff(tips, axis=0), axis=1))) * n  # m per unit s
        if rate > 0.0:
            ds = min(ds, SPEED_SAFETY * speed * DT / rate)
        path = [q_from + s * delta for s in _profile(1.0, ds)]
        return path if self._tip_inside(path) else None

    def _line(
        self, q_from: np.ndarray, site: str, xyz: Vec3, pitch: float, speed: float
    ) -> list[np.ndarray] | None:
        """Waypoints moving `site` on a straight line from its current position to `xyz` at `speed`, IK
        per sample warm-started from the previous waypoint. The end pose asks for `pitch` (as steep as
        reachable, see `_goal`) and the samples blend the pitch linearly from the start to that end, which
        keeps the joint path smooth (a burst of pitch change costs ticks at 2 degrees per tick). None when
        a sample is unreachable or a tip leaves the box."""
        solver = self._solver(site)
        p0, pitch0 = self._site_fk(site, q_from)
        start = np.asarray(p0, dtype=float)
        goal = np.asarray(xyz, dtype=float)
        length = float(np.linalg.norm(goal - start))
        q_goal, ok = self._goal(site, xyz, pitch, q_from)
        if not ok:
            return None
        if length < 1e-6:
            return self._sweep(q_from, np.asarray(q_goal), speed)
        _, pitch_goal = self._site_fk(site, q_goal)
        path: list[np.ndarray] = []
        q_prev = np.asarray(q_from, dtype=float)
        for dist in _profile(length, SPEED_SAFETY * speed * DT):
            frac = dist / length
            point = start + frac * (goal - start)
            xyz_s = (float(point[0]), float(point[1]), float(point[2]))
            q_s, ok = solver(xyz_s, pitch0 + frac * (pitch_goal - pitch0), q_prev)
            if not ok:
                return None
            q_s = np.asarray(q_s, dtype=float)
            jump = float(np.max(np.abs(q_s - q_prev)))
            if jump > MAX_JOINT_STEP:  # a joint reconfiguration: bridge it in joint space
                steps = int(math.ceil(jump / MAX_JOINT_STEP))
                for k in range(1, steps):
                    path.append(q_prev + (q_s - q_prev) * k / steps)
            path.append(q_s)
            q_prev = q_s
        path = self._retime(np.asarray(q_from, dtype=float), path, speed)
        return path if self._tip_inside(path) else None

    def _retime(self, q_from: np.ndarray, path: list[np.ndarray], speed: float) -> list[np.ndarray]:
        """Measure the tip (`arm.ik.fk_tip`) between consecutive waypoints of `path`, starting from
        `q_from`, and split every step that moves it further than `speed` allows per tick (or a joint by
        more than the 2-degree cap) into equal joint-space sub-steps. A straight-line leg profiles the
        target site, and the tip, 2.5 cm from the grasp center, swings further when the pitch changes."""
        cap = SPEED_SAFETY * speed * DT
        for _ in range(RETIME_PASSES):
            out: list[np.ndarray] = []
            prev = q_from
            tip_prev = np.array(ik.fk_tip(prev))
            split = False
            for q in path:
                tip = np.array(ik.fk_tip(q))
                chord = float(np.linalg.norm(tip - tip_prev))
                jump = float(np.max(np.abs(q - prev)))
                n = max(1, math.ceil(chord / cap - 1e-9), math.ceil(jump / MAX_JOINT_STEP - 1e-9))
                for k in range(1, n):
                    out.append(prev + (q - prev) * (k / n))
                out.append(q)
                split = split or n > 1
                prev, tip_prev = q, tip
            path = out
            if not split:
                break
        return path

    def _plan(
        self,
        site: str,
        xyz: Vec3,
        pitch: float,
        speed: float,
        q_from: np.ndarray,
        *,
        straight: bool = False,
        roll: float | None = None,
    ) -> list[np.ndarray] | None:
        """Waypoints moving `site` from the pose `q_from` to `xyz`: the joint-space sweep, or the straight
        line when the sweep leaves the workspace box or `straight` is asked. `roll` is the wrist_roll the
        move ends with. None when the target is unreachable (also with the tip inside the box)."""
        xyz = self._inset(xyz)
        q_from = np.asarray(q_from, dtype=float)
        q0 = q_from.copy()
        if roll is not None:
            q0[4] = roll
        q_goal, ok = self._goal(site, xyz, pitch, q0)
        ok = ok and self._in_box(ik.fk_tip(self._finish(np.asarray(q_goal))), BOX_EPS)
        if not ok and site == GRASP_SITE and roll is None:
            # the grasp center is 2.46 cm from the tip: near the edge of the workspace box (or of the
            # reach) the other side of the jaws may put the tip inside, so try the roll turned by pi
            for alt in self._roll_alternatives(float(q_from[4])):
                q_alt = q0.copy()
                q_alt[4] = alt
                q_try, ok_try = self._goal(site, xyz, pitch, q_alt)
                if ok_try and self._in_box(ik.fk_tip(self._finish(np.asarray(q_try))), BOX_EPS):
                    q0, q_goal, roll, ok = q_alt, q_try, alt, True
                    break
        if not ok:
            return None
        path: list[np.ndarray] | None = None
        if not straight:
            path = self._sweep(q_from, np.asarray(q_goal, dtype=float), speed)
        if path is None:
            path = self._line(q0, site, xyz, pitch, speed)
            if path is not None and roll is not None and abs(float(q_from[4]) - roll) > 1e-9:
                # the line keeps the roll fixed: turn the wrist first, then run the line from there
                q_turned = q_from.copy()
                q_turned[4] = roll
                turn = self._sweep(q_from, q_turned, speed)
                path = None if turn is None else turn + path
        return path

    def _move(
        self,
        site: str,
        xyz: Vec3,
        pitch: float,
        speed: float,
        *,
        straight: bool = False,
        roll: float | None = None,
    ) -> bool:
        """Plan from the current pose and execute; False with one `ik_failures` when no plan exists."""
        path = self._plan(site, xyz, pitch, speed, self._q_cmd, straight=straight, roll=roll)
        if path is None:
            self._stats["ik_failures"] += 1
            return False
        self._run_path(path)
        return True

    def _rise(self, transit_z: Any) -> bool:
        """Rise vertically at the current xy to transit_z on the grasp center (straight line).

        A transit_z below the grasp makes this leg a descent. Where no straight descent exists (folded at
        home near the base, the jaws cannot point down below about 0.19 m), the leg is skipped without an
        IK failure: the travel that follows reaches transit_z on its own. A short obstacle (transit_z =
        tallest + 0.08 m, 0.137 m for the 2026-10-18 tape measure) otherwise failed every pick from home.
        """
        tz = _finite(transit_z)
        if tz is None:
            self._stats["ik_failures"] += 1
            return False
        (gx, gy, gz), _ = ik.fk_grasp(self._q_cmd)
        if tz < gz:
            path = self._plan(GRASP_SITE, (gx, gy, tz), -math.pi / 2, LEG_SPEED, self._q_cmd, straight=True)
            if path is not None:
                self._run_path(path)
            return True
        return self._move(GRASP_SITE, (gx, gy, tz), -math.pi / 2, LEG_SPEED, straight=True)

    def _travel(self, xy: tuple[float, float], transit_z: Any, roll: float | None = None) -> bool:
        """Travel at transit_z to above `xy` on the grasp center (sweep, straight line as the fallback)."""
        tz = _finite(transit_z)
        if tz is None:
            self._stats["ik_failures"] += 1
            return False
        return self._move(GRASP_SITE, (xy[0], xy[1], tz), -math.pi / 2, LEG_SPEED, roll=roll)

    def _transit(self, transit_z: Any, xy: tuple[float, float]) -> bool:
        """Rise, then travel. The descent is the caller's approach move."""
        return self._rise(transit_z) and self._travel(xy, transit_z)

    def _plan_home(self) -> list[list[np.ndarray]] | None:
        home = self._home_q
        direct = self._sweep(self._q_cmd, home, LEG_SPEED)
        if direct is not None:
            return [direct]
        tip = ik.fk_tip(self._q_cmd)
        for z_up in HOME_RAISE_Z:
            raised = (tip[0], tip[1], z_up)
            if not self._in_box(raised):
                continue
            rise = self._line(self._q_cmd, TIP_SITE, raised, -math.pi / 2, LEG_SPEED)
            if not rise:
                continue
            sweep = self._sweep(rise[-1], home, LEG_SPEED)
            if sweep is not None:
                return [rise, sweep]
        home_tip, home_pitch = ik.fk(home)
        line = self._line(self._q_cmd, TIP_SITE, home_tip, home_pitch, LEG_SPEED)
        if line:
            sweep = self._sweep(line[-1], home, LEG_SPEED)
            if sweep is not None:
                return [line, sweep]
        return None

    def _roll_candidates(self, angle: float, pan: float, *, free: bool = False) -> list[float]:
        """The wrist_roll values (modulo pi) inside the limits that close the jaws across the width of an
        object at yaw `angle` with the arm at `pan`, nearest to the current roll first. With `free`
        (a cylinder, whose yaw does not matter) every 15-degree step of the roll follows them."""
        base = angle + pan + ROLL_OFFSET
        lo, hi = self._lo[4] + MARGIN_RAD + LIMIT_EPS, self._hi[4] - MARGIN_RAD - LIMIT_EPS
        current = float(self._q_cmd[4])
        aligned = [base + k * math.pi for k in range(-3, 4) if lo <= base + k * math.pi <= hi]
        aligned.sort(key=lambda r: abs(r - current))
        if not free:
            return aligned
        step = math.pi / 12
        extra = [base + k * step for k in range(-36, 37) if k % 12 and lo <= base + k * step <= hi]
        extra.sort(key=lambda r: abs(r - current))
        return aligned + extra

    def _clear_roll(self, yaw: float, pan: float, above: Vec3, grasp_point: Vec3, *, free: bool) -> float:
        """Of the rolls aligned with `yaw` (any roll for a cylinder, aligned ones first), the nearest one
        whose approach path and descent (gripper fully open) touch nothing; the nearest aligned roll when
        every candidate would (the physics then counts it)."""
        candidates = self._roll_candidates(yaw, pan, free=free)
        if not candidates:
            return self._roll_for(yaw, pan)
        for roll in candidates:
            approach = self._plan(GRASP_SITE, above, -math.pi / 2, LEG_SPEED, self._q_cmd, roll=roll)
            if approach is None:
                continue
            q_above = approach[-1] if approach else self._q_cmd
            descent = self._plan(GRASP_SITE, grasp_point, -math.pi / 2, LEG_SPEED, q_above, straight=True)
            if descent is None:
                continue
            if self._path_clear(approach, 1.0) and self._path_clear(descent, 1.0):
                return roll
        return candidates[0]

    def _clear_slot(
        self, slots: list[tuple[float, float]], release_z: float, transit_z: Any
    ) -> tuple[tuple[float, float], float | None]:
        """The first free slot (and the roll: the current one first, then its alternatives modulo pi)
        whose legs from the current pose (travel at transit_z when given, approach, straight descent to
        the release pose) touch nothing with the held object pinned; the first slot with the current roll
        when every combination would (the physics then counts it)."""
        current = float(self._q_cmd[4])
        rolls = [current] + self._roll_alternatives(current)
        grip = self._grip
        for slot in slots:
            for roll in rolls:
                wanted = None if abs(roll - current) < 1e-9 else roll
                legs: list[list[np.ndarray]] = []
                q_start = self._q_cmd
                if transit_z is not None:
                    travel = self._plan(
                        GRASP_SITE,
                        (slot[0], slot[1], float(transit_z)),
                        -math.pi / 2,
                        LEG_SPEED,
                        q_start,
                        roll=wanted,
                    )
                    if travel is None:
                        continue
                    legs.append(travel)
                    q_start = travel[-1] if travel else q_start
                    wanted = None
                above = (slot[0], slot[1], release_z + 0.08)
                approach = self._plan(GRASP_SITE, above, -math.pi / 2, LEG_SPEED, q_start, roll=wanted)
                if approach is None:
                    continue
                legs.append(approach)
                q_start = approach[-1] if approach else q_start
                descent = self._plan(
                    GRASP_SITE,
                    (slot[0], slot[1], release_z),
                    -math.pi / 2,
                    LEG_SPEED,
                    q_start,
                    straight=True,
                )
                if descent is None:
                    continue
                legs.append(descent)
                if all(self._path_clear(leg, grip) for leg in legs):
                    return slot, (None if abs(roll - current) < 1e-9 else roll)
        return slots[0], None

    def _pose_contacts(self, q: Sequence[float], grip: float) -> set[tuple[int, int]]:
        """Contacts the robot (and the held object, pinned to the tip) would have at the commanded pose
        `q` with the gripper at `grip`, from one forward pass without stepping; the state is restored."""
        m, d = self._model, self._data
        saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
        d.qpos[self._arm_qadr] = self._finish(np.asarray(q, dtype=float))
        d.qpos[self._grip_qadr] = self._grip_rad(grip)
        mujoco.mj_kinematics(m, d)
        if self._held is not None:
            self._pin_held()
        mujoco.mj_forward(m, d)
        pairs: set[tuple[int, int]] = set()
        self._scan_contacts(pairs)
        d.qpos[:] = saved_q
        d.qvel[:] = saved_v
        mujoco.mj_forward(m, d)
        return pairs

    def _path_clear(self, path: Sequence[np.ndarray], grip: float, stride: int = PATH_CHECK_STRIDE) -> bool:
        """True when no sampled waypoint of `path` (every `stride`-th and the last) has a counted contact."""
        if not path:
            return True
        samples = list(path[::stride])
        if samples[-1] is not path[-1]:
            samples.append(path[-1])
        return all(not self._pose_contacts(q, grip) for q in samples)

    def _roll_alternatives(self, current: float) -> list[float]:
        """`current` turned by pi, where the limits allow it (the jaws are symmetric modulo pi)."""
        lo, hi = self._lo[4] + MARGIN_RAD + LIMIT_EPS, self._hi[4] - MARGIN_RAD - LIMIT_EPS
        return [r for r in (current + math.pi, current - math.pi) if lo <= r <= hi]

    def _roll_for(self, angle: float, pan: float) -> float:
        """wrist_roll that closes the jaws across the width of an object at yaw `angle` with the arm at
        `pan`: angle + pan + ROLL_OFFSET modulo pi, nearest to the current roll inside the limits. A roll
        the limits cannot satisfy counts a limit hit and is clipped."""
        base = angle + pan + ROLL_OFFSET
        lo, hi = self._lo[4] + MARGIN_RAD + LIMIT_EPS, self._hi[4] - MARGIN_RAD - LIMIT_EPS
        current = float(self._q_cmd[4])
        candidates = [base + k * math.pi for k in range(-3, 4)]
        inside = [r for r in candidates if lo <= r <= hi]
        if inside:
            return min(inside, key=lambda r: abs(r - current))
        self._stats["limit_hits"] += 1
        nearest = min(candidates, key=lambda r: min(abs(r - lo), abs(r - hi)))
        return float(np.clip(nearest, lo, hi))

    # ------------------------------------------------------------------ execution

    def _run_path(self, path: Sequence[np.ndarray]) -> None:
        for q in path:
            raw = np.asarray(q, dtype=float)
            if bool(np.any(raw < self._lo + LIMIT_HIT_RAD) or np.any(raw > self._hi - LIMIT_HIT_RAD)):
                self._stats["limit_hits"] += 1
            self._q_cmd = np.array(self._finish(raw))
            self._tick(record=True)

    def _grip_rad(self, value: float) -> float:
        return GRIPPER_CLOSED_RAD + GRIPPER_RANGE_RAD * value

    def _grip_ramp(self, target: float, ticks: int) -> None:
        start = self._grip
        for k in range(1, ticks + 1):
            self._grip = start + (target - start) * k / ticks
            self._tick(record=False)
        self._grip = target

    def _tick(self, *, record: bool) -> None:
        """Command the current waypoint and gripper value, step 20 substeps, pin the held object, scan the
        contacts, advance the clock, record the waypoint and enforce the time limit."""
        m, d = self._model, self._data
        d.ctrl[self._arm_act] = self._q_cmd
        d.ctrl[self._grip_act] = self._grip_rad(self._grip)
        pairs: set[tuple[int, int]] = set()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
            if self._held is not None:
                self._pin_held()
            self._scan_contacts(pairs)
        if self._released is not None:
            self._settle_released()
        new = sorted(pairs - self._contact_pairs)
        self._contact_pairs = pairs
        for pair in new:
            phrase = self._phrase(pair)
            self._stats["collisions"] += 1
            self._collision_log.append(phrase)
            if self._stats["first_collision"] is None:
                self._stats["first_collision"] = phrase
        self._ticks += 1
        self._stats["sim_s"] = round(self._ticks * DT, 6)
        if record and self._record and self._suspended == 0:
            self._record_waypoint()
        if self._ticks > MAX_TICKS:
            raise SimTimeLimit(f"simulated time passed {SIM_TIME_LIMIT:.0f} s")

    def _scan_contacts(self, pairs: set[tuple[int, int]]) -> None:
        d = self._data
        n = int(d.ncon)
        if n == 0:
            return
        bodies = self._geom_body[d.contact.geom[:n]]
        kinds = self._kind[bodies]
        held = self._objects[self._held]["body"] if self._held is not None else -1
        robot_side = (kinds == _KIND_ROBOT) | (bodies == held)
        counted = (kinds != _KIND_NONE) & (bodies != held)
        mask = (robot_side[:, 0] & counted[:, 1]) | (robot_side[:, 1] & counted[:, 0])
        for b1, b2 in bodies[mask]:
            pairs.add((int(b1), int(b2)) if b1 < b2 else (int(b2), int(b1)))

    def _phrase(self, pair: tuple[int, int]) -> str:
        held = self._objects[self._held]["body"] if self._held is not None else -1

        def label(b: int) -> str:
            return f"held {self._label[b]}" if b == held else self._label[b]

        a, b = pair
        a_mover = a == held or self._kind[a] == _KIND_ROBOT
        b_mover = b == held or self._kind[b] == _KIND_ROBOT
        if a_mover and b_mover:
            first, second = (b, a) if b > a else (a, b)  # the more distal link first
        elif a_mover:
            first, second = a, b
        else:
            first, second = b, a
        return f"{label(first)} hit {label(second)}"

    def _record_waypoint(self) -> None:
        q = self._rounded(self._q_cmd)
        if self._segments and self._segments[-1]["kind"] == "path":
            self._segments[-1]["q"].append(q)
        else:
            self._segments.append({"kind": "path", "q": [q]})

    def _append_segment(self, segment: dict) -> None:
        if self._record and self._suspended == 0:
            self._segments.append(segment)

    # ------------------------------------------------------------------ objects and grasping

    def _object_pose(self, name: str) -> tuple[float, float, float, float]:
        adr = self._objects[name]["qadr"]
        q = self._data.qpos[adr : adr + 7]
        return float(q[0]), float(q[1]), float(q[2]), _yaw_of_quat(q[3:7])

    def _slot_free(self, slot: tuple[float, float]) -> bool:
        for name in self._objects:
            if name == self._held:
                continue
            x, y, _z, _yaw = self._object_pose(name)
            if math.hypot(x - slot[0], y - slot[1]) < SLOT_FREE_RADIUS:
                return False
        return True

    def _physical_q(self) -> list[float]:
        return [float(v) for v in self._data.qpos[self._arm_qadr]]

    def _attach_candidate(self) -> str | None:
        q = self._physical_q()
        (gx, gy, gz), _ = ik.fk_grasp(q)
        _, c = ik.jaw_axes(q)
        closing = math.atan2(c[1], c[0])
        best: tuple[float, str] | None = None
        for name, info in self._objects.items():
            item = self._scene_in["objects"].get(name)
            if name == self._held or name in self._taken or item is None or not item["seen"]:
                continue
            x, y, z, yaw = self._object_pose(name)
            dist = math.hypot(x - gx, y - gy)
            if dist > ATTACH_XY:
                continue
            h = 2.0 * self._object_half_height(name)
            fraction = (gz - (z - h / 2.0)) / h
            if not ATTACH_FRACTION[0] <= fraction <= ATTACH_FRACTION[1]:
                continue
            if not info["cylinder"]:
                width_dir = yaw + math.pi / 2.0
                if _angle_diff_mod(closing, width_dir, math.pi) > ATTACH_YAW:
                    continue
            if best is None or dist < best[0]:
                best = (dist, name)
        return None if best is None else best[1]

    def _attach(self, name: str) -> None:
        m, d = self._model, self._data
        if self._released is not None:
            self._release_held(restore_only=True)
        info = self._objects[name]
        mujoco.mj_forward(m, d)
        r_tip = d.site_xmat[self._tip].reshape(3, 3)
        p_tip = d.site_xpos[self._tip]
        adr = info["qadr"]
        p_obj = d.qpos[adr : adr + 3].copy()
        r_obj = np.zeros(9)
        mujoco.mju_quat2Mat(r_obj, d.qpos[adr + 3 : adr + 7])
        r_obj = r_obj.reshape(3, 3)
        self._held_offset = (r_tip.T @ (p_obj - p_tip), r_tip.T @ r_obj)
        self._held = name
        m.geom_contype[info["geom"]], m.geom_conaffinity[info["geom"]] = HELD_CONTYPE
        d.qvel[info["dofadr"] : info["dofadr"] + 6] = 0.0
        # the object rests on the table (or against a neighbour) when it is grasped: those contacts are
        # not collisions, so they start as known pairs and only count if they come back after lifting off
        mujoco.mj_forward(m, d)
        existing: set[tuple[int, int]] = set()
        self._scan_contacts(existing)
        self._contact_pairs |= {pair for pair in existing if info["body"] in pair}

    def _held_lowest(self, q: Sequence[float]) -> float:
        """World z of the lowest point of the held object's geom with the arm at the commanded pose `q`
        (kinematics only, the state is restored)."""
        m, d = self._model, self._data
        info = self._objects[self._held]
        saved_q = d.qpos.copy()
        d.qpos[self._arm_qadr] = self._finish(np.asarray(q, dtype=float))
        mujoco.mj_kinematics(m, d)
        self._pin_held()
        mujoco.mj_kinematics(m, d)
        g = info["geom"]
        center = float(d.geom_xpos[g][2])
        rot = d.geom_xmat[g].reshape(3, 3)
        half = info["half"]
        if info["cylinder"]:
            axis_z = min(1.0, abs(float(rot[2, 2])))
            extent = float(half[1]) * axis_z + float(half[0]) * math.sqrt(1.0 - axis_z * axis_z)
        else:
            extent = float(np.abs(rot[2, :]) @ half)
        d.qpos[:] = saved_q
        mujoco.mj_forward(m, d)
        return center - extent

    def _pin_held(self) -> None:
        d = self._data
        info = self._objects[self._held]
        p_off, r_off = self._held_offset
        r_tip = d.site_xmat[self._tip].reshape(3, 3)
        p = d.site_xpos[self._tip] + r_tip @ p_off
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (r_tip @ r_off).reshape(-1))
        adr, dof = info["qadr"], info["dofadr"]
        d.qpos[adr : adr + 3] = p
        d.qpos[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
        d.qvel[dof : dof + 6] = 0.0

    def _release_held(self, *, restore_only: bool = False) -> None:
        """Drop the held object. It keeps ignoring the robot until it has separated from the gripper
        (RELEASE_SEPARATION from the grasp center, or RELEASE_GRACE_TICKS): a tilted gripper lets the
        object slide along its lower finger on the way out, which is a release, not a collision."""
        m = self._model
        if restore_only:
            for info in self._objects.values():
                m.geom_contype[info["geom"]], m.geom_conaffinity[info["geom"]] = FREE_CONTYPE
            self._released = None
            return
        name = self._held
        self._held = None
        self._held_offset = None
        if name is not None:
            self._released = (name, RELEASE_GRACE_TICKS)

    def _take(self, name: str) -> None:
        """The user takes the released object: it leaves the scene (`seen` false at its last pose, as the
        camera would report) and is parked on the far corner of the table where nothing reaches it."""
        self._release_held(restore_only=True)
        x, y, _z, yaw = self._object_pose(name)
        self._taken[name] = (x, y, yaw)
        info = self._objects[name]
        adr, dof = info["qadr"], info["dofadr"]
        d = self._data
        d.qpos[adr : adr + 3] = (0.62, 0.42 - 0.08 * (len(self._taken) - 1), self._object_half_height(name))
        d.qpos[adr + 3 : adr + 7] = _yaw_quat(0.0)
        d.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self._model, d)

    def _settle_released(self) -> None:
        name, ticks_left = self._released
        info = self._objects[name]
        adr = info["qadr"]
        center = self._data.qpos[adr : adr + 3]
        grasp = self._data.site_xpos[self._grasp]
        if ticks_left <= 1 or float(np.linalg.norm(center - grasp)) > RELEASE_SEPARATION:
            m = self._model
            m.geom_contype[info["geom"]], m.geom_conaffinity[info["geom"]] = FREE_CONTYPE
            self._released = None
        else:
            self._released = (name, ticks_left - 1)
