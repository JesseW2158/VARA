<!-- Copied verbatim from the coordinator's scratchpad brief (DESIGN.md) that every implementation agent
read before writing code. Sections 8 and 9 override 2 and 4 where they differ. Paths under
C:/Users/jesse/AppData/Local/Temp/claude/... are session scratch locations and may no longer exist. -->

# VARA implementation brief (shared by every implementation agent)

Read this whole file before touching code. It pins the decisions that several lanes depend on so that
modules written in parallel fit together. The contracts in `docs/interfaces.md` (mirrored in `schemas/*.json`
and `tests/contract/`) always win over this file; if you find a conflict, implement the contract and say so in
your report. Never edit `docs/interfaces.md`, `schemas/`, `tests/contract/` or `CLAUDE.md`. Never commit.

Repo: `C:\Users\jesse\GitHub\VARA` (Windows 11, Git Bash available). Python: ALWAYS use the project venv:
`C:\Users\jesse\GitHub\VARA\.venv\Scripts\python.exe` (Python 3.12.15). Run tests as
`.venv/Scripts/python.exe -m pytest -q <paths>` from the repo root, lint with
`.venv/Scripts/python.exe -m ruff check <files>` and `.venv/Scripts/python.exe -m ruff format <files>`.
Installed: mujoco 3.16, numpy 2.2, opencv-python-headless 4.13 (has cv2.aruco), fastapi 0.143, starlette 1.8,
uvicorn 0.54, httpx 0.28, openai 3.33, daytona 0.225.0, jsonschema 4.26, pyyaml, python-dotenv, pytest 9,
ruff 0.17, gymnasium 1.4, stable-baselines3 2.9, torch 2.11 (CPU), lerobot 0.6.2 (editable from
`C:\Users\jesse\lerobot`), feetech-servo-sdk 1.0.0. There is no real arm, camera, xAI key or Daytona key in
this session: everything you run uses the fakes (`FAKE=1` or the individual `FAKE_*` flags).

Conventions from CLAUDE.md that bind everyone: meters, radians, seconds; joint vectors are
`[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`; gripper 0.0 closed to 1.0 open; degrees
and LeRobot's 0-100 gripper scale exist only inside `arm/driver.py`. Python 3.12, type hints on public
functions, ruff line length 110, `select = ["E", "F", "I", "B"]`, isort known-first-party
`agents, arm, sim, vision, rl, scripts`. Add `from __future__ import annotations` where convenient.

## 1. Repo-wide conventions

- `ROOT = pathlib.Path(__file__).resolve().parents[1]` in every package module (`arm/x.py` -> repo root).
- Fake flags: each module that needs one defines a 3-line helper, no shared settings module (the sandbox code
  bundle is only `arm/`, `sim/`, `schemas/`, `config/`, so `arm/` and `sim/` must never import `agents/`,
  `vision/`, `rl/` or `web/`):
  ```python
  def _fake(flag: str) -> bool:
      return os.environ.get("FAKE") == "1" or os.environ.get(flag) == "1"
  ```
  Flags: `FAKE_ROBOT` (arm/driver.py), `FAKE_CAMERA` (vision/), `FAKE_LLM` (agents/llm.py),
  `FAKE_SANDBOX` (agents/sandbox_pool.py). `DEMO_MODE=1` switches the judging budgets.
- `.env` is loaded with `python-dotenv` (`load_dotenv()`) only in entry points: `agents/orchestrator.py` at
  import, and every `if __name__ == "__main__":` CLI. Never print, log or return `XAI_API_KEY` or
  `DAYTONA_API_KEY`. Never read `.env` yourself in this session (it does not exist; settings deny it anyway).
- Packages `arm`, `sim`, `agents`, `vision`, `rl`, `rl/rewards` already have `__init__.py`. `scripts/*.py` are
  run as files: start them with `sys.path.insert(0, str(ROOT))` where `ROOT = Path(__file__).resolve().parents[1]`.
- CLI entry points use `argparse`. Scripts that move the real arm take `--real`, print a 3 s countdown, and
  watch Esc (see `arm.driver.EscWatcher`). Without `--real` they refuse to touch hardware.
- JSON that crosses a process or network boundary is validated with `jsonschema.Draft202012Validator`
  against `schemas/*.json` (load the schema once at module import).
- Every task writes to `runs/<task_id>/` (gitignored). The sandbox runner writes to its `--out` dir.
- Tests never call xAI, Daytona or the arm. Tests set the fake flags themselves with `monkeypatch.setenv`.
- Logging: `logging.getLogger(__name__)`; debugging output to stderr, never to stdout in `sim.run_candidate`
  (its stdout carries the score line).
- Windows: `subprocess` children need `SYSTEMROOT` in their environment or Python fails to start. The
  "clean environment" for AI code is therefore `{"PATH", "PYTHONPATH", "SYSTEMROOT"}` (plus `TEMP`/`TMP` if
  present). Use `sys.executable` for the interpreter. Use `cv2.VideoCapture(i, cv2.CAP_DSHOW)` on Windows.
- Threads, not asyncio, for anything that blocks (serial, subprocesses, MuJoCo). The orchestrator wraps them
  with `asyncio.to_thread` / a worker thread.

## 2. Coordinate frame, model geometry (measured on sim/model/so101_new_calib.xml)

Our frame: origin on the table directly below the shoulder_pan axis, +x straight out from the arm at pan 0,
+y to the arm's left, +z up, table at z = 0. In the upstream MJCF the pan axis is at x = 0.0388353, so the
robot's root body must be placed at `pos="-0.0388353 0 0"` (edit the copied `so101_new_calib.xml`:
`<body name="base" pos="-0.0388353 0 0" ...>`). After that edit, MuJoCo world coordinates ARE our frame.

Measured facts (upstream frame, all joints at 0, "new calibration" = every joint's zero is mid-range):
- Joint order and ranges (rad): shoulder_pan ±1.9199, shoulder_lift ±1.7453, elbow_flex ±1.69,
  wrist_flex ±1.6581, wrist_roll [-2.7438, 2.8412], gripper [-0.1745, 1.7453]. qpos indices 0..5 in the
  arm-only model; in the full scene look joints up by name (`mujoco.mj_name2id`), never assume indices.
- At q = 0 the upper arm points straight up, the forearm and gripper point along +x; the jaw tips are at
  x ≈ 0.398 (upstream) = 0.359 from the pan axis, z ≈ 0.227. Link lengths: shoulder_lift axis at
  (x 0.030 from pan axis, z 0.1166); upper arm 0.116 m; forearm 0.135 m; wrist_flex axis to jaw tip 0.166 m
  (wrist_flex to wrist_roll 0.061 + gripper 0.105). So upper arm + forearm = 0.251 m only.
- Joint axes in world at q = 0: shoulder_pan about -z (positive pan turns the arm toward -y, i.e. to the
  arm's right; check the sign numerically, do not trust this sentence), shoulder_lift, elbow_flex and
  wrist_flex about +y (positive wrist_flex pitches the jaws down from horizontal toward -z), wrist_roll about
  the jaw axis.
- Existing site `gripperframe` in body `gripper` (pos -0.0079 -0.000218 -0.0981, quat 0.707107 0 0.707107 0):
  its x-axis already points out of the jaws, its z-axis is the jaw closing direction (the jaws open along
  world z at q = 0, i.e. in the plane of the arm). Add our site `tip` to the same body:
  `<site name="tip" pos="-0.0069 -0.0005 -0.1033" quat="0.707107 0 0.707107 0" size="0.003" rgba="1 0 0 1"/>`
  (the midpoint of the fixed-jaw tip and the moving-jaw tip with the gripper closed; verify numerically and
  adjust if the mesh vertices say otherwise). `pitch = asin(a_z)` with `a` = the tip site's x-axis in world.
  pitch 0 = jaws horizontal, -pi/2 = straight down.
- Gripper joint: -0.17 rad is fully closed (tip-to-tip gap 0.5 cm), gap ≈ 1.8 cm + 7.8 cm·g for g in
  [0, 1.0] (g = 0.3 -> 4.2 cm, 0.6 -> 6.6 cm, 1.0 -> 9.6 cm). Our gripper value v in [0, 1] maps to
  g = -0.17 + v · 1.17 (v = 1 -> 1.0 rad, 9.6 cm) in the sim. The real 0-100 scale is A11's business.
- Reach is tight. With the tip pointing straight down (pitch -90°) the wrist_flex axis must be 0.166 m above
  the tip, which the 0.251 m arm cannot do far from the base or high up. Measured feasibility (our frame):
  pick pill_bottle (0.18,-0.12,0.035): -90 ok; pick juice_box (0.24,-0.04,0.0525): needs ≥ -75°; tray center
  (0.30,0.22,0.0975): needs ≥ -45°; far tray slot (0.32,0.24,0.08): needs ≥ -30°; transit at z 0.28 over
  the 0.20 m cereal box: needs -15° or 0°; handoff zone (0.30,0,0.10): needs ≥ -60°; z = 0.30 ceiling at
  x 0.20: needs ≥ -15°. Therefore `arm.ik.solve` MUST relax the pitch toward horizontal when the requested
  pitch is unreachable (section 4), and SimRobot transit moves use whatever pitch IK finds.
- Home pose (config/scene.yaml `home_q`, already updated): `[0.0, -1.35, 0.02, 1.45, 0.0]` -> tip
  (0.111, 0, 0.274), pitch -7°, the whole gripper within x ≤ 0.123, z in [0.25, 0.33], 12° from every limit.
  The kit's placeholder put the gripper 5 cm above the table inside the tea tin; do not go back to it.

## 3. Shared config files

- `config/limits.yaml` (new, Lane A/IK agent writes it): `{shoulder_pan: [-1.9199, 1.9199], shoulder_lift:
  [-1.7453, 1.7453], elbow_flex: [-1.69, 1.69], wrist_flex: [-1.6581, 1.6581], wrist_roll: [-2.7438, 2.8412]}`
  taken from the MJCF, with a header comment that A6 replaces them with the calibrated ranges.
- `config/joint_map.yaml` (new, same agent): per joint `sign: 1` and `offset_deg: 0.0` (A6 measures), plus
  `gripper: {closed_pct: 0, open_pct: 100}`, `lerobot_id: my_awesome_follower_arm`,
  `lerobot_robot_type: so101_follower`. Formula (driver): `q_rad = radians(sign * lerobot_deg + offset_deg)`.
- `config/scene.yaml`: workspace box, zones, camera placeholders, calib markers, `home_q` (see above).
- `config/objects.yaml`, `config/fake_scene.json`, `config/rl.yaml`: as in the kit.
- `config/cameras.yaml` (Lane B writes): `{overhead: {index: 0, width: 640, height: 480, fps: 30,
  fourcc: MJPG}, wrist: {index: 1, ...}}`, indices overridable by env `OVERHEAD_CAM`, `WRIST_CAM`.
- `config/homography.npy` (Lane B writes a synthetic placeholder: a pinhole camera 0.65 m above `camera.c`
  looking straight down, 640x480, f = 600 px; `vision/calibrate.py --fake` regenerates it).

## 4. Public interfaces by module (names are binding)

### arm/ik.py (A7)
- `MODEL_PATH = ROOT / "sim" / "model" / "arm_only.xml"`; `load_model() -> mujoco.MjModel` cached.
- `fk(q) -> tuple[tuple[float, float, float], float]`: tip xyz (our frame) and pitch.
- `fk_tip(q) -> tuple[float, float, float]`: xyz only (what `arm.replay.validate` receives as `fk`).
- `jaw_axes(q) -> (a, c)`: unit jaw direction and unit closing direction in world (tip site x and z axes).
- `PITCH_LADDER = (-75, -60, -45, -30, -15, 0)` degrees. `solve(xyz, pitch=-pi/2, q0=None) -> (q, ok)`:
  damped least squares on joints 0-3 with λ = 0.02, finite-difference Jacobian (1e-4 rad, `mj_kinematics`),
  clamp to limits minus 3° after every step, 200 iterations max, success = position error < 5 mm and pitch
  error < 3°. If the requested pitch fails, retry each ladder pitch that is shallower than the request
  (greater than it), nearest first; return the first success. `q[4]` (wrist_roll) is copied from `q0`.
  `q0=None` means `config/scene.yaml` `home_q`. Also export `solve_exact(xyz, pitch, q0)` without the ladder.
- `python -m arm.ik --check` prints fk(home_q) and solves a few targets. Keep a solve under 20 ms.

### arm/driver.py (A4, A13)
- `limits() -> dict[str, list[float]]` from config/limits.yaml; `JOINTS` (5 names), `TICK = 1/30`,
  `MAX_STEP_RAD = radians(2.0)`, `MARGIN_RAD = radians(3.0)`.
- `class Driver` (real) and `class FakeDriver` share: `connect()`, `disconnect()`, `read() -> (list[float],
  float)`, `command(q, gripper) -> None` (clips to limits minus 3° and to 2° per call from the last commanded
  position, ignored while `stopped` or while torque is off), `freeze()` (hold present position, torque on,
  `stopped = True`), `resume()`, `torque_off()`, `torque_on()` (syncs Goal_Position = Present_Position first),
  `stopped: bool`, `torque: bool`, `last_command: tuple[list[float], float] | None`, `is_fake: bool`,
  `lock: threading.RLock`. FakeDriver also has `gripper_reading_override: float | None` for tests.
- `get_driver() -> Driver | FakeDriver` singleton; FakeDriver when `_fake("FAKE_ROBOT")`.
- `class EscWatcher`: background thread, `msvcrt.kbhit/getwch` on Windows, `select` on stdin otherwise,
  calls a callback on Esc; `start()`, `stop()`, `pressed: bool`.
- Real path: LeRobot 0.6.2 `from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig`
  (fallback import `lerobot.robots.so101_follower`), id from env `FOLLOWER_ID` default
  `my_awesome_follower_arm`, `use_degrees=True`, `max_relative_target=5.0`, `disable_torque_on_disconnect=False`.
  Before any torque enable (including inside LeRobot's connect/configure) write Goal_Position =
  Present_Position (raw ticks, `normalize=False`) for every motor; read `C:\Users\jesse\lerobot-tools\HANDOFF.md`
  and `C:\Users\jesse\lerobot-tools\sync_goals.py` for the verified procedure and the LeRobot source under
  `C:\Users\jesse\lerobot\src\lerobot` for exact signatures. Gripper hold: close until the reading stops
  changing, then hold at the contact reading minus 3 (0-100 scale). All serial access under `lock`.

### arm/replay.py (S0 validate, A9, A13)
- `validate(traj, limits, fk, box) -> list[str]` exactly per contract and `tests/contract/test_trajectory_validate.py`.
- `load_limits()`, `load_box()` (from config), `bridge(q_from, q_to, max_step_rad=radians(1.0)) -> list[list[float]]`.
- `replay(traj, *, driver=None, scene=None, realtime=None, on_event=None) -> dict`: `driver` defaults to
  `get_driver()`; `scene` is a callable returning the latest scene dict (for the hand check), default
  `None` = no hand data; `realtime` defaults to `not driver.is_fake`. Streams at 30 Hz, bridges at ≤ 1°/tick,
  grip close waits 0.4 s and compares holding with `expect_held` (FakeDriver: holding == expect_held unless
  `gripper_reading_override` is set), live segments call `arm.real_robot`, every tick checks `driver.stopped`
  and the hand (wristband seen within 0.10 m of `fk_tip(current q)` outside carry_to_hand -> freeze, reason
  `hand_too_close`), within 0.15 m of the hand the tip speed is capped at 0.1 m/s by sub-stepping. A gripper
  close within 0.10 m of a seen wristband is refused (abort like a missed grasp, reason `hand_too_close`).
  Returns `{"ok": bool, "reason": None | "grasp_missed" | "stopped" | "hand_too_close" | "error", "events": [...]}`.
- `python -m arm.replay --real --traj file.json`.

### arm/real_robot.py (A10)
- `home(driver=None, *, max_step_rad=radians(1.0)) -> None`, `freeze(driver=None) -> None` (also cancels a
  running primitive), `carry_to_hand(obj, *, driver=None, scene=None, on_event=None, timeout=10.0) -> dict`
  per the task text (IK tracking at 10 Hz, 0.1 m/s cap within 0.15 m, open when the hand stays within 8 cm
  for 1 s, fall back to the handoff zone after 10 s, back off 5 cm up, home, event `handoff_done`). When
  `config/rl.yaml` has `use_policy: true`, import `rl.policy_runner` lazily and use it instead of IK.
- `python -m arm.real_robot --real --carry-test`.

### arm/sim_robot.py (A8) — the only place AI-written code gets a robot
- `class SimTimeLimit(BaseException)`; `SIM_TIME_LIMIT = 60.0`; `DT = 1/30`.
- `class SimRobot(scene: dict, *, skills_dir: Path | None = None, record: bool = True, model_path=None)`
  implementing the Robot API verbatim (`scene, move_to, set_roll, open_gripper, close_gripper, home, pick,
  place, place_in_zone, carry_to_hand, run_skill`). Extra, underscore-free attributes used by the runner:
  `trajectory() -> dict` (schemas/trajectory.json, `skill`/`args` set via `set_skill(name, args)`),
  `final_scene() -> dict`, `stats -> dict` with `collisions, limit_hits, ik_failures, sim_s, first_collision`
  (a phrase like "gripper hit cereal_box"), `events -> list[str]`, `reset(scene)`.
  AI code can call any public method; nothing in the API lets it reach MuJoCo, files or the network.
- Model: `sim/model/scene_ours.xml` (robot + table + tray walls + one free body per object + one mocap box
  per obstacle + hand mocap box, sizes from `config/objects.yaml`). Poses come from the scene JSON at
  `reset()`: objects by writing their free-joint qpos (x, y, z = h/2, yaw quaternion), obstacles, tray and
  hand by writing `mocap_pos/mocap_quat` (hand parked at z = -1 when `seen` is false).
- Stepping: `m.opt.timestep = 1/600`, 20 substeps per 30 Hz tick. Each tick sets `ctrl` of the 5 arm
  position actuators to the waypoint and the gripper actuator to the mapped angle, steps, then counts
  contacts. A contact counts when one geom belongs to the robot (or the held object) and the other to the
  table, tray, an obstacle, the hand box or an object that is not the current grasp target; robot
  self-contacts are not counted. Count distinct new (geom pair) events per tick and remember the first as
  `first_collision` ("gripper hit cereal_box", "held juice_box hit tray"). Sim time advances DT per tick;
  `close_gripper` costs 0.4 s, `open_gripper` 0.3 s; past 60 s raise `SimTimeLimit`.
- `move_to(x, y, z, pitch=-pi/2, speed=0.15)`: reject (return False, no motion) when xyz is outside the
  workspace box; `arm.ik.solve` from the current q (IK failure -> `ik_failures += 1`, return False); joint-space
  trapezoidal profile from the current q to the solution with joint speed ≤ 60°/s and tip speed ≤
  min(speed, 0.2) (sample the interpolation with fk to measure tip path length), waypoints at 30 Hz appended to
  the open `path` segment of the recorded trajectory. A waypoint within 1° of a limit -> `limit_hits += 1`.
- `transit_z` helpers: rise vertically to transit_z (same x, y), move at that height, descend. The pitch at
  each leg is whatever `solve` returns (expect ~0° at z 0.28).
- `set_roll(angle)`: wrist_roll target = angle - shoulder_pan + ROLL_OFFSET, measured so that angle 0 closes
  the jaws across the width of an object at yaw 0 (width is the second `size` entry, across the length that
  runs along yaw). Document the measured ROLL_OFFSET.
- Grasp: `close_gripper()` attaches the nearest object whose center line is within 1.0 cm of the tip, with the
  tip between 20 % and 80 % of its height, and whose yaw differs from the jaw closing direction by at most
  20° (mod 180°; cylinders always pass the yaw test). While held, after every mj_step write the object pose =
  tip pose ∘ offset captured at grasp time and zero its 6 velocities. `open_gripper()` releases it.
- `pick(name, approach=0.08, transit_z=None)`: open, (transit), move above the object by `approach` with
  pitch -pi/2 (IK may relax it), set_roll to the object's yaw, descend to `max(h/2, 0.02)`, close, rise back
  by `approach`. `place(x, y, release_z=0.05, approach=0.08, transit_z=None)`: (transit), above by approach,
  descend to release_z, open, rise. `place_in_zone(zone, slot=None, transit_z=None)`: next free slot (a slot
  is free when no object center is within 0.03 m of it), release at `wall + h_held/2 + 0.01`.
  `carry_to_hand()`: above the hand (+0.05 m) or the handoff zone when unseen, open, back off 0.05 up, home,
  append event `handoff_done`, record `{"kind": "live", "primitive": "carry_to_hand", "object": name}`.
  `home()`: move to `home_q` in joint space (no IK) and record it. `run_skill(name, args)`: load
  `skills_dir/<name>.json`, `check_source`, exec in the restricted namespace (section 6) and call `run`.
- Recorded trajectory: `dt = 1/30`, `path` segments hold commanded waypoints (5 joints, floats rounded to
  6 decimals), `grip` segments `{"kind": "grip", "action": "close"|"open", "object": name, "expect_held": bool}`
  (expect_held only on close), `live` segments as above. Consecutive moves extend one `path` segment; a grip or
  live segment closes it. `sim/model/build_scene.py` (section 5) writes `scene_ours.xml`.

### sim/candidate_check.py (S0) and agents/candidates.py
- `check_source(src: str, entry: str = "run") -> list[str]` lives in `sim/candidate_check.py` (the sandbox
  bundle has no `agents/`). `agents/candidates.py` does `from sim.candidate_check import check_source` and
  re-exports it (`__all__`). Rules exactly as the contract and `tests/contract/test_candidate_check.py`.
  Entry arity: `run` takes exactly `(robot, args)`, `reward` exactly `(obs, action, prev_action, info)`.
  Allowed top level: docstring, `import math` (only that exact form), `def`, and `NAME = <literal>` where
  literal means `ast.literal_eval` succeeds (numbers incl. negative, strings, None/True/False, tuples, lists,
  dicts of literals). Every `ast.Name`, attribute name, argument name and function name starting with `_` is
  rejected. Rejected names (as Name or Attribute): open exec eval compile __import__ getattr setattr delattr
  globals locals vars input breakpoint help dir type object super. Rejected nodes: ClassDef, Global, Nonlocal,
  AsyncFunctionDef, Await, AsyncFor, AsyncWith, Yield, YieldFrom, With, bare `except:`, `except BaseException`
  (alone or in a tuple). Longer than 200 lines (`len(src.splitlines()) > 200`) is rejected. A SyntaxError is one
  problem string.

### sim/success.py (S0)
- `evaluate(spec, before, after, args, events) -> bool` exactly per contract and `tests/contract/test_success.py`.
  `spec` may be a full task spec (with `success`) or a bare predicate. Implicit rule: every obstacle in
  `before` must be `not_moved` with max 0.02 (an obstacle unseen in `after` keeps its last pose, so compare
  positions regardless of `seen`). Also export `resolve_object(ref, before, args) -> list[str]` and
  `already_true(spec, scene, args) -> bool` (= evaluate(spec, scene, scene, args, [])).

### sim/run_candidate.py and sim/scoring.py (S0, C4)
- CLI: `python -m sim.run_candidate (--candidate cand.py --task task.json | --skill skill.json --args '{...}')
  --scene scene.json --trials 5 --seed 7 --out DIR [--index N] [--skills-dir skills]`. `--index` defaults to
  the number in `cand_<N>.py`, else 0. Exit code 0 even when the candidate fails; non-zero only when the
  runner itself breaks. Prints exactly one JSON line (schemas/score.json) as the LAST line of stdout; all other
  output goes to stderr. Writes `out/traj.json`, `out/final_scene.json` (trial 1) and `out/score.json`.
- `check_source` first; a rejected candidate scores `passed: false, error: "rejected: <first problem>"`,
  `trials` = requested count, `successes 0`, `nominal_success false`, `mean_sim_s null`, `traj null`.
- Trials run in ONE child process (`multiprocessing` spawn, same interpreter) that streams a JSON line per
  trial back through a pipe; the parent kills it when a trial exceeds 60 s wall clock and records
  `error: "timeout"` for that trial, marking the candidate failed with `fail_reason "trial k: timeout"`.
  The child executes the candidate with `exec(compile(src, "<candidate>", "exec"), ns)` where
  `ns = {"__builtins__": SAFE_BUILTINS, "math": math}`; SAFE_BUILTINS is the normal builtins dict minus
  open, exec, eval, compile, __import__, getattr, setattr, delattr, globals, locals, vars, input, breakpoint,
  help, dir, type, object, super, memoryview, classmethod, staticmethod, property. Candidate `print` output is
  redirected to stderr. `SimTimeLimit` and any exception end the trial with `error` set (and success false).
- `sim/scoring.py`: `perturb_scene(scene, trial_index, seed) -> dict` (trial 1 = unchanged; trials 2..5 shift
  every object by uniform ±0.01 m in x and y and ±10° yaw, obstacles ±0.005 m, using
  `random.Random(f"{seed}:{trial_index}")` so candidates share scenes), `trial_result(...)`,
  `summarize(index, trial_results, traj_path) -> dict` (the pass rule: trial 1 success, ≥ 4/5 successes, zero
  collisions, zero limit hits, no errors; `fail_reason` = first failing trial's phrase: "trial 3: gripper hit
  cereal_box", "trial 2: spec not satisfied", "trial 1: error: ..."), `rank(scores) -> list` (passing first,
  then successes desc, then mean_sim_s asc).
- `--skill` form: loads `skills/<name>.json`, uses `skill["spec"]`, `skill["code"]`, `args` from `--args`.

### agents/llm.py (C1)
- `chat(messages, model, schema=None, timeout=30, *, purpose=None, key=None, task_id=None) -> str | dict`.
  Real: `openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")`, `chat.completions.create`,
  `temperature=0.7` for code, `0.2` for routing; with `schema`, first try `response_format={"type":
  "json_schema", "json_schema": {"name": ..., "schema": schema}}`, on any 4xx fall back to plain text and parse
  the first `{...}` block; validate with jsonschema; on validation failure retry once with the error appended.
  Retry 3× with exponential backoff (1, 2, 4 s) on 429 and 5xx. Append `{"t", "task_id", "model", "purpose",
  "prompt_tokens", "completion_tokens", "ms"}` to `runs/<task_id>/llm.jsonl` when `task_id` is given.
- `FAKE_LLM`: answers from `tests/fixtures/llm/`. Layout: `tests/fixtures/llm/aliases.json` maps normalized
  phrases to a slug (normalize = lowercase, strip everything but letters, digits and spaces, collapse spaces,
  then spaces -> `_`); `tests/fixtures/llm/router/<slug>.json` holds the router reply; `tests/fixtures/llm/
  candidates/<slug>/cand_<i>.py` (i = 1..6) hold candidate sources; `tests/fixtures/llm/rewards/variant_<k>.py`
  (k = 1..4); `tests/fixtures/llm/repair/<slug>/cand_<i>.py` optional. `purpose` is one of `router`,
  `write_skill`, `repair`, `write_rewards`; `key` is the phrase (router) or `f"{slug}:{i}"` (candidates). An
  unknown router phrase returns `{"action": "queue", "task": <text>, "reason": "no fixture for this phrase"}`;
  an unknown candidate key returns `tests/fixtures/candidate_clear_table.py`. The fake returns the candidate
  wrapped in a ```python fence so the real-path code extraction is exercised.
- `router_model()`, `codegen_model()` read `ROUTER_MODEL`/`CODEGEN_MODEL` (default `grok-4.3` and `grok-4.7`
  when unset, with a warning). Routing defaults to no reasoning; code generation uses low reasoning.
  See [live model configuration](setup.md#live-model-configuration) for effort overrides, output caps,
  and current official documentation. `python -m agents.llm --ping` makes authenticated API calls.

### agents/router.py (C2)
- `route(text, skills, scene, *, task_id=None, timeout=15) -> dict` validated against schemas/router.json.
  `skills` is a list of `{"name", "description", "params"}`. Builds the prompt from `agents/prompts/router.md`
  (predicate vocabulary, the three worked specs, the "queue when the predicates cannot express success" rule,
  name mapping "my pills" -> pill_bottle). A `learn` spec that `sim.success.already_true` finds true on
  `scene` becomes `{"action": "queue", "task": text, "reason": "That already looks done."}`. On any connection
  error (not in fake mode) fall back to `keyword_route(text, skills)`.
- `keyword_route(text, skills) -> dict | None`: the five demo phrases ("hand me my pills" -> run hand_over
  pill_bottle; "clear the table" -> learn clear_table spec from the contract; "fold my laundry" -> queue;
  "when i say good morning ..." -> define_routine good_morning (hand_over pill_bottle, hand_over juice_box);
  "good morning" -> run good_morning; "stop" -> stop) plus "put the juice box in the tray" -> run
  place_in_tray {object: juice_box} when that skill exists.
- `tests/fixtures/router_phrases.json`: 20 `{"text", "action", ...expected fields}` entries, 5 of them queue.
  `python -m agents.router --eval` (live model, not run here).

### agents/candidates.py (C3, C7)
- `HINTS`: the six strategy hints from docs/tasks/C-agents.md, verbatim. `Candidate` dataclass: `index`
  (1..6), `hint`, `source`, `problems: list[str]`, `attempt`.
- `generate(spec, scene, skills, *, task_dir, n=6, timeout=45, demo_mode=False, on_rejected=None) ->
  list[Candidate]`: six parallel `chat(..., purpose="write_skill", key=f"{slug}:{i}")` calls using
  `agents/prompts/write_skill.md` (Robot API and notes verbatim from docs/interfaces.md, the candidate rules,
  the spec English + JSON, the scene JSON, the two starter skills as worked examples, one hint, and the closing
  line "Return one Python code block and nothing else."), `extract_code`, `check_source`, one regeneration with
  the problems appended (skipped in demo mode); writes `prompts/write_skill_<i>.md`, `cand_<i>.py` and
  `cand_<i>.problems.json` to `task_dir`.
- `repair(spec, scene, skills, failures, *, task_dir, timeout) -> list[Candidate]` (C7 normal mode).
- `routine_candidate(steps) -> str`: a `run(robot, args)` that calls `robot.run_skill(step["skill"],
  step["args"])` for each step and returns False on the first False.
- `extract_code(text) -> str`. `python -m agents.candidates --eval "<text>"`.

### agents/library.py (C9, C10)
- `class Library(root=ROOT / "skills")`: `all() -> list[dict]`, `get(name) -> dict | None`, `save(skill) ->
  Path` (validates schemas/skill.json), `summaries() -> list[{"name", "description", "params"}]`,
  `reset_demo() -> list[str]` (archive learned skills to `runs/archive/<timestamp>/`, copy `_starter/*.json`
  into `skills/`), `load_backup() -> list[str]`, `make_skill(spec, code, sim, real, trusted=False) -> dict`
  (`created` ISO 8601 with timezone), `routine_spec(name, steps, library) -> dict` (`all` of the steps'
  specs with `$param` substituted by the step args; description/english joined).

### agents/sandbox_pool.py (C5, C6)
- `class LocalRunner`: `score(task_dir, candidates, task_path, scene_path, skills_dir, *, trials=5, seed=7,
  timeout=60, on_score=None) -> list[dict]` runs `sim.run_candidate` for every candidate at once as
  subprocesses with the clean env (section 1), parses the last stdout line, validates against score.json, labels
  each score with `"runner": "local"`, calls `on_score(score)` as each finishes; `plan(skill_path, args,
  scene_path, out_dir, skills_dir, timeout=30) -> dict` (one trial). The `runner` key is stripped before any
  schema validation that uses `additionalProperties: false` (keep two dicts: the validated score and a wrapper
  `{"score": ..., "runner": "local"|"daytona"}`).
- `class SandboxPool`: Daytona 0.225 API: `Daytona()` (reads DAYTONA_API_KEY/DAYTONA_API_URL/DAYTONA_TARGET
  from env), `daytona.create(CreateSandboxFromSnapshotParams(snapshot="arm-sim-v1", auto_stop_interval=0,
  labels={"vara": "pool"}), timeout=120)`, `sandbox.fs.upload_file(bytes, "/home/daytona/vara/bundle.tgz")`,
  `sandbox.process.exec(cmd, cwd="/home/daytona/vara", timeout=60) -> ExecuteResponse(exit_code, result)`,
  `sandbox.fs.download_file(path) -> bytes`, `sandbox.delete()`. `start(n_score=6, n_plan=1)` creates in
  threads, `shutdown()` deletes all (atexit + SIGINT/SIGTERM handlers), hard cap 7, fallback to LocalRunner
  after two failures per sandbox with `"runner": "local"`. Snapshot builder `build_snapshot(name="arm-sim-v1")`
  uses `Image.debian_slim("3.12").pip_install(["mujoco", "numpy", "pyyaml", "jsonschema"])` and
  `daytona.snapshot.create(CreateSnapshotParams(name=name, image=image), on_logs=print)`.
- `get_pool() -> SandboxPool | LocalRunner` singleton (LocalRunner when `_fake("FAKE_SANDBOX")` or no key).
- `python -m agents.sandbox_pool --bench`. `scripts/daytona_smoke.py` per C5 (creates one sandbox, deletes in finally).

### agents/planner.py (C8)
- `plan(skill, args, scene, *, task_dir, pool) -> tuple[dict, dict, dict]` = (trajectory, final_scene,
  score); trusted skills via `LocalRunner.plan`, others via the pool's planning sandbox with local fallback.

### agents/orchestrator.py (S0, C7, C9, C10, C11)
- FastAPI `app`; every endpoint in the contract table; `GET /` serves `web/index.html`, `GET /voice` serves
  `web/voice.html`, `/static` mounts `web/static`, `GET /video/overhead` and `/video/wrist` stream MJPEG from
  `vision.scene`/`vision.cameras` (a 1-frame placeholder JPEG in fake mode). `WS /events` replays the last 200
  events on connect, then streams. Events also go to `runs/<task_id>/events.jsonl`. Task ids `t-0001`
  upward, continuing from the highest existing `runs/t-*` directory.
- One task at a time (`409` while busy); state in `{"idle", "busy", "stopped"}`; `mode` = `"demo"` when
  DEMO_MODE=1 else `"normal"`; `carry` = `"rl"` when config/rl.yaml `use_policy` else `"ik"`.
- `/stop` calls `driver.freeze()` first (within 0.2 s of the request), then cancels the task (a
  `threading.Event` checked between stages and by the pool/replay), emits `stopped`, state `stopped`.
  `/resume` sends the arm home at ≤ 30°/s and returns to `idle`. `/torque_off` 409 while busy.
- `/voice/token`: POST `https://api.x.ai/v1/realtime/client_secrets` with `{"expires_after": {"seconds":
  300}}` via httpx, return `{"token": data["value"], "expires_at": data["expires_at"], "model":
  "grok-voice-latest"}`; 503 without `XAI_API_KEY`; in `FAKE_LLM` mode return a fake token
  `"fake-voice-token"`. Never include the API key in any response or log.
- Flow per action: `learn`: routing -> spec_ready -> writing_code -> candidate_rejected* -> simulating ->
  candidate_scored×6 -> selected -> validating -> executing -> verifying -> saved | not_saved -> done; `run`:
  routing -> planning -> validating -> executing -> verifying -> done; `queue`: routing -> queued -> done;
  `define_routine`: like learn with the routine candidate and `routine_spec`; `stop`: like /stop. A `learn`
  with no passing candidate (after the repair round in normal mode) ends `queued`. Budgets per the contract
  table (normal / demo). `not_saved` re-plans the next-best passing candidate on the current scene once.
- Before-snapshot = `vision.scene.get_tracker().snapshot(10)` at planning time; after-snapshot: FAKE_CAMERA ->
  the selected candidate's `final_scene.json`; real -> `vision.verify.take_after`.
- `python -m agents.orchestrator --bench-demo`. Keep the module importable with `FAKE=1` and no hardware.

### vision/ (B1-B8)
- `vision/cameras.py`: `open_camera(index) -> cv2.VideoCapture` (DSHOW on Windows, MJPG 640x480 30 fps),
  `class Camera` with `read() -> ndarray | None`, `release()`, `class FakeCamera` (synthetic 640x480 frame with
  the fake scene drawn), `get_camera(name) -> Camera | FakeCamera`, `python -m vision.cameras --check SECONDS`.
- `vision/markers.py`: `DICT = cv2.aruco.DICT_4X4_50`, `detector()`, `detect(frame) -> list[tuple[int,
  ndarray]]` (id, 4x2 corners), `marker_table() -> dict[int, dict]` from objects.yaml, `draw(frame, dets)`.
- `vision/calibrate.py`: `fit_homography(pixels, table_xy) -> ndarray`, `pixel_to_table(H, u, v) ->
  tuple`, `height_correct(p, c, h, Hc) -> tuple`, `calibration_error(H, dets, markers) -> float` (cm), CLI
  `--fake` writes the synthetic `config/homography.npy`, `--live` fits from a frame.
- `vision/scene.py`: `class SceneTracker(camera, H, objects, scene_cfg)` thread at 10 Hz with median-5
  filtering and the 0.5 s `seen` timeout, `latest() -> dict`, `snapshot(n=10) -> dict`, `start()/stop()`,
  `annotated_jpeg() -> bytes`; `class FakeSceneTracker` serves `config/fake_scene.json` with a fresh `t` and
  `set_scene(scene)`; `get_tracker() -> SceneTracker | FakeSceneTracker` singleton; every published dict
  validates against schemas/scene.json. `python -m vision.scene --print`.
- `vision/verify.py`: `verify(spec, before, after, args, events, *, task_dir=None) -> dict` with keys `ok`,
  `missing` (objects named by the spec unseen in `after`), saving `before.json`, `after.json`, `verify.json`
  when `task_dir` is given; `take_after(tracker, spec, args, *, wait=1.0, retry=3.0) -> dict`.
  `tests/test_verify.py` on 10 fixture pairs in `tests/fixtures/verify/` (5 pass, 5 fail).

### web/ (S0, B9, A14)
- `web/index.html` + `web/static/app.js` + `web/static/style.css`: plain HTML/JS, Chart.js pinned from a CDN,
  everything in B9 (feeds, command box, timeline from `WS /events`, success-test banner labeled "written before
  any code", six candidate cards with the winner highlighted and "local" labels, DEMO countdown 75 s, library
  with Reset demo / Load backup, RL panel reading `/static/rl/curves.json` and `/static/rl_sim.mp4`, STOP +
  Esc -> `POST /stop`, Resume, Check calibration, `/voice` in an iframe).
- `web/voice.html` + `web/static/voice.js`: token from `POST /voice/token` -> `new WebSocket(
  "wss://api.x.ai/v1/realtime?model=grok-voice-latest", ["xai-client-secret." + token])`; `session.update`
  with voice `eve`, the instructions and the three tools from the contract, `turn_detection: null`; push to
  talk with the T key or the button (PCM16 24 kHz base64 `input_audio_buffer.append`, then `commit`, then
  `response.create`); handle `response.output_audio.delta` (base64 PCM16 24 kHz playback via Web Audio),
  `response.function_call_arguments.done` (name, call_id, arguments) -> call our endpoint -> `conversation.
  item.create` with `{"type": "function_call_output", "call_id", "output"}` -> `response.create`; for do_task
  answer "On it" at once and, when `/events` shows the end stage, send a user text item with the summary and
  `response.create`. No API key in the page.

### rl/ (D1-D7)
- `rl/handoff_env.py`: `class HandoffEnv(gymnasium.Env)` per D1 on `sim/model/arm_only.xml` (tip from
  `arm.ik.fk`, never from simulator internals), `obs` 16 floats, action 4 in [-1, 1] scaled to ±2° per step
  at 30 Hz, 150 steps, success = tip within 3 cm of the true target for 10 consecutive steps, randomizations
  as listed, `info["success"]`, `info["dist"]`, `render_mode="rgb_array"` via `mujoco.Renderer`.
- `rl/rewards/baseline.py`: `reward(obs, action, prev_action, info)` = -d - 0.01·|a - a_prev|² + 1[d < 0.03].
- `rl/train.py --reward path --steps N --out dir [--n-envs 8]` (SB3 PPO MlpPolicy, SubprocVecEnv, checkpoint
  every 50k, `progress.csv` with step, mean reward, success rate), `rl/evaluate.py --policy zip --episodes N`,
  `rl/rewards/write_rewards.py` (one CODEGEN call -> 4 variants, each `check_source(src, entry="reward")`),
  `rl/train_parallel.py` (4 Daytona sandboxes, snapshot `arm-rl-v1`, CPU torch), `rl/policy_runner.py`
  (`run_policy(driver, scene, policy_path, *, timeout=8.0) -> bool` using the same 16-number observation).
  Reward files are loaded with `exec` after `check_source`, never imported.

### skills/_starter (A12)
- `hand_over.json` and `place_in_tray.json`, `trusted: true`, code and specs from docs/tasks/A-arm.md A12,
  `sim: {"successes": 5, "trials": 5}` only after they really pass 5/5 with
  `python -m sim.run_candidate --skill ... --args '{"object": "juice_box"}' --scene config/fake_scene.json
  --trials 5 --seed 7 --out runs/check`, `real: {"runs": 0, "successes": 0}`, `created` ISO 8601.

## 5. sim/model files (A5)

`sim/model/` already holds the upstream copy (`so101_new_calib.xml`, `scene.xml`, assets). The model agent:
- edits `so101_new_calib.xml`: base body `pos="-0.0388353 0 0"`, the `tip` site, and replaces the two gripper
  collision meshes (`wrist_roll_follower_so101_v1` in body `gripper`, `moving_jaw_so101_v1` in body
  `moving_jaw_so101_v1`) with primitive boxes fitted to the mesh vertices: one box for the gripper housing,
  one thin box per finger, so the space between open fingers is empty (convex-hull mesh collisions would fill
  it and push objects out of the grasp). Give every robot collision geom `contype="1" conaffinity="1"`.
- writes `arm_only.xml` (includes the robot, a floor plane at z = 0, lights; used by `arm/ik.py` and
  `rl/handoff_env.py`) and `build_scene.py` (generates `scene_ours.xml` from `config/objects.yaml`,
  `config/scene.yaml`, `config/fake_scene.json`: table box top at z = 0, tray floor + 4 walls of height
  `wall` as mocap body `tray`, free bodies named after each object with a box or cylinder geom of its `size`/`h`
  (mass 0.05, `contype="1" conaffinity="3"`), mocap box per obstacle named after it (`contype="1"
  conaffinity="3"`), mocap `hand` box 0.08 × 0.10 × 0.04 (`contype="1" conaffinity="3"`), all with a
  `name` equal to the body name so SimRobot can map contacts to names). Held objects get `contype="2"
  conaffinity="2"` at run time (so they stop colliding with the robot but keep colliding with everything else).
- deletes the files we do not use (`so101_old_calib.xml`, `so101_new_calib_camera.xml`, leader/camera STLs).
- `python -m mujoco.viewer --mjcf=sim/model/scene_ours.xml` must load (cannot be run headless here; load the
  model with `mujoco.MjModel.from_xml_path` in a test instead and check gravity holds the arm with position
  control at home_q for 2 simulated seconds within 2° drift).

## 6. Fakes and the end-to-end path

`python scripts/e2e_fake.py "put the juice box in the tray"` forces `FAKE=1`, uses FastAPI's `TestClient`,
POSTs `/command`, follows `WS /events` (or polls `GET /state` plus `runs/<id>/events.jsonl`) and prints each
stage; exit 0 only when the last stage is `saved` or `done`. In fake mode the router fixture for that phrase
returns `learn` with spec name `juice_box_to_tray` (success `{"in_zone": {"object": "juice_box", "zone":
"tray"}}`), six fixture candidates are scored by `LocalRunner` on the MuJoCo sim (at least 3 pass, at least 1
fails with a `fail_reason` naming cereal_box), the winner is validated, "replayed" on the FakeDriver without
sleeping, the after-snapshot is the winner's `final_scene.json`, `evaluate` passes, and `skills/
juice_box_to_tray.json` is saved. `POST /reset_demo` removes it again.

## 7. Reporting back

Your final message is data for the coordinator, not prose for a human. Report: files written, the exact
commands you ran with their last lines of output, every assumption you made that is not in the contracts or
this brief, anything you could not finish, and any place you believe the contract or this brief is wrong.

## 8. Decisions after the foundation stage (binding; they override sections 2 and 4 where they differ)

The foundation modules exist and are verified: `arm/ik.py`, `arm/driver.py`, `arm/replay.py` (validate, bridge,
load_limits, load_box; `replay()` is still a placeholder), `sim/candidate_check.py`, `sim/success.py`,
`sim/model/*` (`so101_new_calib.xml`, `arm_only.xml`, `scene_ours.xml`, `build_scene.py`), `config/limits.yaml`,
`config/joint_map.yaml`, `web/*`. Read their docstrings before using them; this section records what was measured
and decided while verifying them.

### 8.1 Measured facts

- `fk(home_q)` = (0.1155, 0, 0.2745), pitch -6.9 deg. `fk([0]*5)` = (0.357, 0, 0.2276). Positive pan moves the tip
  toward -y; positive wrist_flex pitches the jaws down.
- The SO-101 jaw is asymmetric: only the moving jaw swings. `tip` (closed-jaw midpoint) is 0.3 mm from the fixed
  finger's inner face, so an object centered on `tip` is inside that finger. The grasp center is the site `grasp`
  = `tip + arm.ik.grasp_offset() * c` with `grasp_offset()` = 0.0246 m and `c` the closing direction (toward the
  moving jaw, the second value of `arm.ik.jaw_axes(q)`). `arm.ik.fk_grasp(q) -> (xyz, pitch)` and
  `arm.ik.solve_grasp(xyz, pitch=-pi/2, q0=None) -> (q, ok)` work on that site; `fk`, `fk_tip`, `solve`,
  `solve_exact` keep working on `tip`.
- Inner jaw gap at the tip level: 1.6 cm at gripper angle 0, 5.9 cm at 0.6 rad, 7.8 cm at 1.0 rad (our v = 1.0),
  narrowing toward the hinge (4.0 cm five cm above the tips at 0.6 rad, 5.6 cm at 1.0 rad). The moving tip rises
  2.4 cm at 1.0 rad. Fully open (v = 1.0), every object of `config/objects.yaml` centered on `grasp` with the tip at
  the object's mid-height touches nothing; at 0.6 rad the moving finger hits the juice box and tea tin.
- `arm.ik.solve` costs < 2 ms on every demo target and < 20 ms worst case; it clamps pitch to [-pi/2, pi/2],
  raises `ValueError` on non-finite or mis-shaped input, and clips wrist_roll (copied from q0) to limits - 3 deg.
- The workspace box contains unreachable points: the far corners (0.40, +-0.25, any z) and a band near x = 0.10,
  z 0.10-0.18. There `solve` returns `(q0, False)` and `SimRobot.move_to` returns False; that is expected.
- Transit height z = 0.28 (tallest obstacle 0.20 + 0.08) is reachable only at pitch near 0 and only within about
  0.38 m of the pan axis: above tray slots 1-3 yes, above slot 4 (0.325, 0.245) no. Expect `move_to` at transit
  height to come back with pitch ~0 and plan descents accordingly (descend in two legs if a direct descent sweeps
  the held object through a tray wall).
- Tray slots in `config/scene.yaml` and `config/fake_scene.json` are now 5 cm apart:
  `[[0.275, 0.195], [0.325, 0.195], [0.275, 0.245], [0.325, 0.245]]` (the old 4 cm spacing overlapped 4 cm wide
  objects). A11/A15 will re-measure with the real tray.
- Robot self-collisions are physically simulated (robot geoms contype 1 / conaffinity 1; MuJoCo excludes only
  parent-child pairs) and none occur at home_q, q = 0, the pick poses or random poses with any gripper opening.

### 8.2 SimRobot grasp recipe (A8), binding

- `move_to(x, y, z, pitch, speed)` as called by AI code targets `tip` (the contract's wording). `pick`, `place`,
  `place_in_zone` and `carry_to_hand` target the grasp center: solve with `arm.ik.solve_grasp`, descend with
  `solve_grasp`, and in `close_gripper` measure the 1.0 cm attach distance from `arm.ik.fk_grasp(q)` to the
  object's vertical center line (xy distance), with the grasp height between 20 % and 80 % of the object's height.
  The contract sentence "attaches the object only if the tip is within 1.0 cm of its center line" is read as the
  grasp center; say so in the docstring.
- `pick` opens the gripper fully (1.0) before descending. `set_roll(yaw)` aligns the closing direction across the
  object's width (`size[1]`); measure `ROLL_OFFSET` numerically with `jaw_axes` and document it.
- The held object's pose offset is captured relative to the `tip` pose (`fk` + the site's rotation) at grasp time
  and re-applied after every `mj_step` with zero velocity, exactly as section 4 says.
- Count robot self-contacts as collisions too (phrase "gripper hit upper_arm"); add a test that `home()`, the
  pick poses and a `place_in_zone` run produce none. A self-colliding joint sweep must never reach the real arm.
- Release in `place_in_zone`: at `wall + h_held/2 + 0.01` above the tray floor, measured at the grasp center.
- `transit_z` legs: rise vertically at the current xy, travel at that height, descend; each leg is a `move_to`
  on the grasp center with the pitch IK finds. A leg whose IK fails counts one `ik_failures` and returns False
  from the calling method without moving.

### 8.3 Driver facts (A4) for replay, real_robot, orchestrator, policy_runner

- `command(q, gripper) -> bool`: True when sent, False when ignored (stopped or torque off). A non-finite value
  raises `ValueError` before anything is sent; a never-connected real driver raises `RuntimeError` from
  `command()`, `read()`, `torque_on()` and `close_gripper_hold()`. It clips to limits - 3 deg, then to 2 deg per
  call from the last commanded position (or the present position right after connect/freeze/resume/torque_on).
  `limits()` and `joint_map()` raise `ValueError` on an empty or malformed config file (missing file -> MJCF
  defaults / identity map).
- `get_driver()` returns an UNCONNECTED real `Driver` outside fake mode because connecting powers the arm (goal
  sync, then LeRobot configure). The orchestrator (in `--real` runs) and the `--real` CLIs call `driver.connect()`
  once, after the human confirms the arm is clear. `FakeDriver` starts connected at `home_q` with the gripper open
  (1.0) and has `set_present(q, gripper)`, `commands` (list of what was sent) and `gripper_reading_override`.
- `read() -> (q, gripper)`. `replay()` and the live primitives call `read()` first and turn a `RuntimeError`
  or `ValueError` from the driver into `{"ok": False, "reason": "error"}` (after `freeze()`), never a traceback.
- `close_gripper_hold(target=0.0, settle=0.5) -> float` closes, waits `settle`, reads, and holds at the reading
  minus 3 on the 0-100 scale (converted); returns the reading in our 0-1 units. Replay's grip check: close with
  `close_gripper_hold`, then "holding" means the reading is above the empty-closed reading by at least the
  object's `held_min` (0-100 scale, from `config/objects.yaml`, converted to 0-1 through the joint map). FakeDriver:
  holding == `expect_held` unless `gripper_reading_override` is set.
- `freeze()` never raises and always sets `stopped`; `resume()` clears it. `torque_off()` blocks every command
  until `torque_on()`.

### 8.4 Runner (C4) amendments

- `SAFE_BUILTINS` must contain a restricted `__import__` that returns the `math` module for `name == "math"` and
  raises `ImportError` for anything else (a candidate's top-level `import math` statement calls
  `builtins.__import__`; without it every accepted candidate dies at exec time).
- The score line on stdout is exactly `schemas/score.json` (`additionalProperties: false`, so no extra keys).
  Per-trial details go to `out/trials.json`: a list of `{"trial", "success", "collisions", "limit_hits",
  "ik_failures", "sim_s", "error", "first_collision"}`. The orchestrator reads it and puts `trial_results`
  (list of booleans) plus `runner` and `hint` into the `candidate_scored` event detail (detail is free-form).
- `fail_reason` phrases: "trial 3: gripper hit cereal_box", "trial 2: spec not satisfied", "trial 1: error: <msg>",
  "trial 1: timeout", "trial 4: limit hit". `first_collision` names bodies, never geoms ("held juice_box hit
  tray", "gripper hit cereal_box", "gripper hit hand").

### 8.5 Router and candidate prompt rules (C2, C3)

- A `learn` spec uses concrete scene names, never `$params` (the router output for `learn` carries no args);
  `params` is `{}`. `in_zone` only with zone `tray` (the handoff zone has no radius; `evaluate` raises on it).
  Predicates never name an obstacle (the implicit rule already requires every obstacle to stay put). `route()`
  catches `ValueError` from `sim.success.already_true` and turns a bad spec into `{"action": "queue", ...,
  "reason": "I could not write a success test for that."}`.
- The candidate prompt (`agents/prompts/write_skill.md`) and the reward prompt must list the check_source traps
  explicitly: no `if __name__ == "__main__":`, no name starting with `_` (not even a bare `_` loop variable), no
  `from typing import ...` or any import other than `import math`, no annotations using forbidden names (`args:
  object`), no `class`, no `with`, no `BaseException`, no numpy in reward functions, at most 200 lines. Fixture
  candidates in `tests/fixtures/llm/` must pass `check_source` (test it).
- Fixture scenes: the fake router phrase "put the juice box in the tray" returns `learn` with spec
  `juice_box_to_tray` (success `{"in_zone": {"object": "juice_box", "zone": "tray"}}`, `params: {}`).

### 8.6 Orchestrator coordination (C9-C11)

- `GET /video/overhead` and `/video/wrist` are `multipart/x-mixed-replace` streams that keep sending frames (at
  least 2 fps, the fake placeholder re-sent on a timer); a single stalled part never renders in Chromium.
- `POST /stop` leaves `state` = `"stopped"` until `POST /resume`, even when no task was running. `/resume` returns
  `{"state": "idle"}`, `/stop` returns `{"state": "stopped", "task_id": ...}`.
- `GET /state` returns `{"state", "mode", "carry", "task_id", "skills": [{"name", "description", "trusted",
  "real": {"runs", "successes"}, "sim": {...}, "created"}], "queue": [{"task", "reason", "task_id"}], "stopped":
  bool}`; `/reset_demo` and `/load_backup` return `{"skills": [names]}`; `/calibrate` returns `{"error_cm": float}`
  (fake: 0.8); `/command` returns `{"task_id": id}` or 409 `{"detail": "busy"}`; `/torque_off` 409 while busy.
- `candidate_scored` detail: the score dict plus `runner`, `hint`, `trial_results`. `selected` detail:
  `{"candidate", "hint", "scores": [...]}`. `spec_ready` detail: the spec. `routing` detail: `{"text"}`.

### 8.7 Web polish (verified minor defects to fix in web/)

Card header wraps letter by letter at 1366 px with the LOCAL badge (`white-space: nowrap` on `.status`, badge on
its own row); `trialMarks()` must use `detail.trial_results` when present and otherwise show trial 1 plus a count;
`renderSkills()` must skip non-object entries and not report a render error as "orchestrator unreachable";
cap `fail_reason`/`error` text at ~200 chars with the full text in `title`; guard `atob()` in `playDelta`;
non-JSON tool `arguments` return an error `function_call_output` instead of becoming a command; load Chart.js
with `defer` and make `loadRl()` tolerate Chart arriving late; forward T keydown/keyup from the dashboard to the
voice iframe with postMessage; clamp the countdown to [0, 75].

## 9. Glue stage (orchestrator, pool, planner, library, scripts, RL glue); binding like section 8

Everything in sections 4 and 8 exists and is verified when this stage starts: `arm/sim_robot.py` (SimRobot),
`sim/run_candidate.py` + `sim/scoring.py` (the runner CLI, `out/trials.json`), `sim/restricted.py`,
`arm/replay.py::replay`, `arm/real_robot.py`, `agents/llm.py`, `agents/router.py`, `agents/candidates.py`,
`tests/fixtures/llm/*`, `vision/*`, `rl/handoff_env.py`, `rl/train.py`, `rl/evaluate.py`, `web/*`. Read their
docstrings and the W2 reports summarized in 9.8 before writing glue.

### 9.1 agents/library.py (C9, C10)

- `class Library(root: Path = ROOT / "skills")`, with `starter_dir = root / "_starter"`, `backup_dir = root /
  "_backup"`, `archive_root = ROOT / "runs" / "archive"`. Live skills are `root/*.json` only (never the
  subdirectories). `ensure_seeded()`: when `root` has no `*.json`, copy the starters in (called at orchestrator
  start so a fresh clone works).
- `all() -> list[dict]` (validated, sorted by `created`), `get(name) -> dict | None`, `save(skill) -> Path`
  (validates against schemas/skill.json, refuses names starting with `_` or not matching the schema pattern,
  writes `root/<name>.json` atomically), `delete(name) -> bool`, `summaries() -> list[{"name", "description",
  "params"}]` (description = `spec["description"]`, params = `spec["params"]`), `starter_names() -> list[str]`.
- `reset_demo() -> list[str]`: move every non-starter live skill to `runs/archive/<UTC YYYYmmdd-HHMMSS>/`, copy
  `_starter/*.json` into `root` (overwriting), return the live names (exactly the two starters).
  `load_backup() -> list[str]`: copy `_backup/*.json` into `root`, return the names copied.
- `make_skill(spec, code, sim, real, trusted=False) -> dict` (`created` = ISO 8601 with the local timezone).
  `routine_spec(name, steps, library) -> dict`: `success = {"all": [step specs' success with "$param"
  replaced by the step's args]}`, `params = {}`, `description = "Routine: " + "; then ".join(step descriptions)`,
  `english` joined the same way; raises `KeyError` for an unknown skill. Substitution walks the predicate tree
  and replaces every string value `"$x"` by `args["x"]`.
- `python -m agents.library --list`.

### 9.2 agents/sandbox_pool.py (C5, C6)

- Common result shape, a "scored" dict: `{"candidate": int, "runner": "local" | "daytona", "score": <dict
  exactly schemas/score.json>, "trial_results": list[bool], "out_dir": str}`. `trial_results` comes from
  `out/trials.json` (empty list when missing). The validated score never carries extra keys.
- `class LocalRunner`: `score(task_dir, candidates, task_path, scene_path, skills_dir, *, trials=5, seed=7,
  timeout=60, on_score=None, cancel=None) -> list[scored]`; `candidates` is a list of `(index, path)` pairs or
  objects with `.index` and `.path`; every candidate starts at once as `subprocess.Popen([sys.executable, "-m",
  "sim.run_candidate", "--candidate", path, "--task", task_path, "--scene", scene_path, "--trials", ...,
  "--seed", ..., "--out", task_dir / f"cand_{i}", "--index", i, "--skills-dir", skills_dir, "--timeout", ...],
  cwd=ROOT, env=CLEAN_ENV)` where `CLEAN_ENV = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT),
  "SYSTEMROOT": ..., "TEMP": ..., "TMP": ...}` (only keys present); the process is polled every 0.1 s, killed
  when it exceeds `timeout * trials + 10` s or when `cancel` (a `threading.Event`) is set; the last non-empty
  stdout line is parsed and validated against schemas/score.json (a parse/validation failure becomes a scored
  dict with `passed: false`, `error: "runner: <reason>"`, trials as requested, successes 0); `on_score(scored)`
  is called from the polling thread as each finishes. `plan(skill_path, args, scene_path, out_dir, skills_dir,
  timeout=30) -> scored` runs `--skill ... --args json --trials 1`.
- `class SandboxPool` per section 4 with the Daytona 0.225 names, `start(n_score=6, n_plan=1)` in threads,
  `score(...)`/`plan(...)` with the same signatures and result shape (`runner: "daytona"`), bundle = tar.gz of
  `arm/`, `sim/`, `schemas/`, `config/` (hash-tracked, re-uploaded on change, extracted with `tar -xzf` in
  `/home/daytona/vara`), per-task upload of the candidate files, `task.json`, `scene.json` and `skills/*.json`,
  `python -m sim.run_candidate ...` through `sandbox.process.exec(cmd, cwd=..., timeout=...)`, download of
  `traj.json`, `final_scene.json`, `trials.json`, `score.json` for the two best passing candidates into
  `task_dir/cand_<i>/`, fallback to `LocalRunner` after two failures of a sandbox (`runner: "local"`),
  `shutdown()` deleting every sandbox (atexit + SIGINT/SIGTERM), hard cap 7 sandboxes in `start()`.
  The Daytona client is created through a module-level factory `_daytona_factory()` so tests inject a fake.
- `get_pool() -> SandboxPool | LocalRunner` singleton: `LocalRunner` when `_fake("FAKE_SANDBOX")` or
  `DAYTONA_API_KEY` is unset. `build_snapshot(name="arm-sim-v1")`. `python -m agents.sandbox_pool --bench`
  (three fixture tasks x 6 candidates x 5 trials on the current pool, prints wall-clock per task).
- `scripts/daytona_smoke.py` (C5): needs a key, never run in tests; creates one sandbox from `arm-sim-v1`,
  uploads the bundle, scores `tests/fixtures/candidate_clear_table.py`, prints the score line, deletes the
  sandbox in `finally`.

### 9.3 agents/planner.py (C8)

- `plan(skill, args, scene, *, task_dir, pool) -> tuple[dict, dict, dict]` = (trajectory, final_scene, score):
  writes `task_dir/skill.json` and `task_dir/scene.json`; `skill["trusted"]` -> `LocalRunner().plan(...)`,
  else `pool.plan(...)` (which falls back to local); reads `traj.json`/`final_scene.json` from the out dir;
  raises `PlanError(fail_reason)` when the trial failed (`score["passed"]` false or no trajectory).

### 9.4 agents/orchestrator.py (S0 item 5, C7, C9, C10, C11)

- Module import: `load_dotenv()`, `app = FastAPI()`, one `Orchestrator` instance `orch`. The module must import
  with `FAKE=1` and no hardware, keys or network, and `uvicorn agents.orchestrator:app --port 8000` must serve
  the dashboard. `python -m agents.orchestrator [--port 8000] [--real]`: `--real` asks the human to confirm the
  arm is clear, then `driver.connect()`, then runs uvicorn; without `--real` and without `FAKE_ROBOT`/`FAKE`,
  the server starts but any stage that would move the arm ends the task with `error` detail `{"error": "arm not
  connected: start with --real"}`.
- State: `state` in `idle | busy | stopped`; `mode` (`demo` when `DEMO_MODE=1`); `carry` (`rl` when
  config/rl.yaml `use_policy` else `ik`); `current task_id`; `queue: list[{"task", "reason", "task_id"}]`;
  events: a deque of the last 200 plus `runs/<task_id>/events.jsonl`; WebSocket subscribers get every event
  (replay the deque on connect); `stop_event: threading.Event`; one worker thread per task; `arm_lock`
  (threading.Lock) around planning-before-snapshot, validate, replay, home and the after-snapshot.
- Event emission: `emit(stage, detail, task_id)` builds `{"t": time.time(), "task_id", "stage", "detail"}`,
  validates against schemas/event.json, appends to the deque and the task's `events.jsonl`, and pushes to the
  subscribers through `loop.call_soon_threadsafe`. Detail shapes per 8.6.
- Task ids `t-0001` upward, continuing from the highest `runs/t-*`. `runs/<task_id>/` holds `command.txt`,
  `router.json`, `task.json` (the spec), `scene.json` (before snapshot), `prompts/`, `cand_<i>.py`,
  `cand_<i>/` (runner outputs), `scores.json`, `selected.json`, `traj.json`, `replay.json`, `before.json`,
  `after.json`, `verify.json`, `events.jsonl`, `llm.jsonl`.
- Flows (the worker runs stages in order; every stage checks `stop_event` first and ends the task with
  `stopped` when set; a stage exceeding its budget ends the task with `error` (or `queued` while learning),
  then the arm goes home):
  - `routing`: `agents.router.route(text, library.summaries(), tracker.latest(), task_id=...)` within the
    routing budget (thread + join timeout).
  - `run`: `planning` (before = `tracker.snapshot(10)`; `planner.plan`) -> `validating` (`arm.replay.validate`
    with `load_limits()`, `arm.ik.fk_tip`, `load_box()`; problems -> `error`) -> `executing` (`arm.replay.replay
    (traj, driver=driver, scene=tracker.latest, on_event=...)`; not ok -> `error` with the reason, or `stopped`)
    -> `verifying` (after = the plan's final_scene when `FAKE_CAMERA`, else `vision.verify.take_after(tracker,
    spec, args)`; `vision.verify.verify(spec, before, after, args, events, task_dir=...)`) -> update the skill's
    `real` counts (`runs += 1`, `successes += ok`) -> `done` detail `{"ok": bool, "skill": name}`.
  - `learn`: `spec_ready` (detail = the spec; `task.json`) -> `writing_code` (`candidates.generate` with
    `on_rejected` emitting `candidate_rejected {"candidate", "problems", "attempt"}` and `writing_code
    {"candidate", "hint"}` per candidate; demo mode: proceed with >= 3 valid, else `queued`) -> `simulating`
    (`pool.score(...)` with `on_score` emitting `candidate_scored` = score + `runner` + `hint` +
    `trial_results`; `cancel = stop_event`) -> `selected` (`sim.scoring.rank`; detail `{"candidate", "hint",
    "scores"}`; no passing candidate: normal mode runs ONE repair round (`candidates.repair` with the three best
    failures, then `simulating` again), demo mode skips it; still none -> `queued` with reason "no candidate
    passed simulation") -> `validating` -> `executing` -> `verifying` -> `saved` (`library.save(make_skill(spec,
    code, sim={"successes", "trials"} from the score, real={"runs": 1, "successes": 1}))`, detail `{"name"}`)
    or `not_saved` (detail `{"reason"}`; then re-plan the next-best passing candidate on the current scene with
    `pool.plan`, replay it once, verify; pass -> `saved`, else give up) -> `done`.
  - `define_routine`: `spec = library.routine_spec(name, steps, library)` (unknown skill -> `queued`),
    `code = candidates.routine_candidate(steps)`, then the learn path with that single candidate (scored by the
    pool with `--skills-dir skills`, validated, replayed, verified) and saved under `name`.
  - `queue`: `queued` detail `{"task", "reason"}` -> `done`. `stop`: like `/stop`.
  - Budgets (docs/interfaces.md table): routing 15/10 s, code writing 45/30 s, simulation 90/35 s, repair
    on/off, command-to-arm-moving no cap / 75 s (on expiry `queued` with reason "over the 75 s budget"), real
    execution 90 s. Each stage runs in the worker; the per-stage timeout uses the thread + `join(timeout)` or
    the pool's `cancel` event.
- Endpoints: exactly the contract table plus `GET /` (web/index.html), `GET /voice` (web/voice.html), `/static`
  (web/static), `GET /video/overhead` and `GET /video/wrist` (`multipart/x-mixed-replace; boundary=frame`
  generators sending `tracker.annotated_jpeg()` / the wrist camera frame at 5 fps, a placeholder JPEG re-sent
  at 2 fps in fake mode, never a single stalled part). `POST /command`: 409 `{"detail": "busy"}` while busy or
  stopped; otherwise `{"task_id"}` and the worker starts. `POST /stop`: `driver.freeze()` FIRST (in the
  handler, before anything else), `stop_event.set()`, state `stopped`, emit `stopped`, return `{"state":
  "stopped", "task_id"}`. `POST /resume`: `driver.resume()`, `arm.real_robot.home(driver)` (FakeDriver:
  instant), `stop_event.clear()`, state `idle`, `{"state": "idle"}`. `POST /torque_off`: 409 while busy, else
  `driver.torque_off()`. `POST /calibrate`: `{"error_cm"}` (0.8 in FAKE_CAMERA, else
  `vision.calibrate.calibration_error` on a fresh frame). `POST /reset_demo`: 409 while busy; `library.
  reset_demo()`, clears the queue; `{"skills": [...]}`. `POST /load_backup`: `{"skills": [...]}`. `POST
  /voice/token`: per section 4 (httpx POST with the Bearer key; `FAKE_LLM` -> `{"token": "fake-voice-token",
  "expires_at": now + 300, "model": "grok-voice-latest"}`; no key -> 503). `GET /state`: per 8.6. `WS /events`.
- Startup/shutdown (lifespan): `library.ensure_seeded()`; `tracker = vision.scene.get_tracker()` and
  `tracker.start()` for the real tracker; `pool = get_pool()`, `pool.start()` in a thread for a SandboxPool;
  `driver = arm.driver.get_driver()`; shutdown: `pool.shutdown()`, `tracker.stop()`, `driver.disconnect()`
  (torque stays on). `python -m agents.orchestrator --bench-demo`: with `FAKE=1` and `DEMO_MODE=1`, five learns
  of "clear the table" through `TestClient`, printing time from command to `executing` for each.

### 9.5 scripts/e2e_fake.py (S0 item 7) and tests/test_scaffold.py

- `python scripts/e2e_fake.py "<text>"` sets `FAKE=1` (and `DEMO_MODE` only if already set) BEFORE importing
  the orchestrator, uses `fastapi.testclient.TestClient(app)` with the lifespan, POSTs `/command`, follows
  `runs/<task_id>/events.jsonl` (or the WebSocket) printing `stage  detail-summary` lines until an end stage
  (`saved`, `not_saved`, `queued`, `done`, `stopped`, `error`), exits 0 only when the last stage is `saved` or
  `done`. For "put the juice box in the tray" the expected trace is routing -> spec_ready -> writing_code x6
  -> simulating -> candidate_scored x6 -> selected -> validating -> executing -> verifying -> saved -> done,
  with `skills/juice_box_to_tray.json` written (and `POST /reset_demo` removing it).
- `tests/test_scaffold.py`: with `FAKE=1`, every module in the CLAUDE.md repo map imports and exposes the
  public names of sections 4, 8 and 9; `scripts/e2e_fake.py` runs as a subprocess and exits 0 (this is the
  one slow test; keep it under 90 s by letting the runner use `--trials 5` on the MuJoCo sim).

### 9.6 rl glue (D3-D7)

- `rl/rewards/write_rewards.py`: `write_rewards(*, out_dir=ROOT / "rl" / "rewards", task_dir=None) ->
  list[Path]`: one `agents.llm.chat(..., purpose="write_rewards", key="rewards")` call with the D1 environment
  description, the baseline reward source and its D-GO result (from `rl/runs/baseline/summary.json` when it
  exists, else "not measured yet"), the four shaping ideas from docs/tasks/D-rl.md D4, the check_source trap
  list (8.5), and "Return four Python code blocks, one per variant, nothing else."; split the reply into code
  blocks, `check_source(src, entry="reward")` each, ask once more for a rejected one with its problems, write
  `variant_<k>.py` (k = 1..4), save the prompt and raw reply under `runs/rewards/<timestamp>/`. In FAKE_LLM the
  fake returns the four fixtures. CLI `python -m rl.rewards.write_rewards`.
- `rl/train_parallel.py`: `build_rl_snapshot(name="arm-rl-v1")` (debian_slim 3.12, pip `mujoco numpy pyyaml
  gymnasium stable-baselines3 jsonschema` plus CPU torch via `--index-url https://download.pytorch.org/whl/cpu`,
  `Resources(cpu=4, memory=8, disk=10)`); `train_parallel(rewards, *, steps=600_000, out_dir=ROOT / "rl" /
  "runs" / "parallel")`: 4 sandboxes with `auto_stop_interval=0`, upload `rl/`, `arm/ik.py`, `arm/__init__.py`,
  `sim/model/arm_only.xml` + its assets, `sim/restricted.py`, `sim/candidate_check.py`, `sim/__init__.py`,
  `config/limits.yaml`, `config/scene.yaml`, one reward per sandbox; start `nohup python -m rl.train --reward
  reward_k.py --steps N --out out/ > train.log 2>&1 &`; poll `tail -n 1 out/progress.csv` every 30 s; download
  `policy.zip`, `progress.csv`, `summary.json`; delete the sandboxes in `finally`; pick the best by success rate
  then mean steps to success; copy it to `rl/policies/handoff_best.zip`; write `web/static/rl/curves.json` as
  `{"variants": [{"name", "hint", "steps": [...], "success_rate": [...], "mean_reward": [...], "chosen": bool,
  "final_success_rate"}], "created"}`. `--local` runs the same four trainings as local subprocesses of
  `rl.train` (used by the tests with tiny step counts) and is also the fallback when there is no Daytona key.
  Sandboxes are created only here (cap 4) and through a `_daytona_factory()` tests can replace.
- `rl/policy_runner.py`: `run_policy(driver, scene, policy_path, *, timeout=8.0, clock=None, sleep=None,
  on_event=None) -> bool`: `PPO.load(policy_path, device="cpu")`, a 30 Hz loop building the D1 observation from
  `driver.read()` (joints), `arm.ik.fk`, the target 5 cm above the scene's hand (or the handoff zone when
  unseen), previous action; `action = policy.predict(obs, deterministic=True)`, scaled to +-2 degrees, sent with
  `driver.command(q + delta, gripper)` (the driver clips); success = within 3 cm for 10 consecutive steps;
  returns False on timeout or `driver.stopped`. `arm.real_robot.carry_to_hand` calls it when `use_policy` is
  true and then releases/backs off/homes exactly as the IK path.
- `rl/render_video.py --policy zip --out web/static/rl_sim.mp4 --seconds 10` (D6): offscreen `mujoco.Renderer`
  frames of the policy tracking a drifting target, written with `cv2.VideoWriter` (mp4v); when the renderer
  cannot initialize (no OpenGL), print the reason and exit 3; never required by tests.
- `tests/test_rl_glue.py`: write_rewards in FAKE_LLM writes four variants that pass check_source and saves
  the prompt; train_parallel `--local` with `--steps 512` and a stubbed snapshot builder finishes, writes
  `curves.json` and picks a policy; policy_runner on the FakeDriver with a 512-step policy reaches the target
  when the fake driver teleports (set_present) or stops on timeout without errors; `run_policy` stops at once
  when `driver.stopped`.

### 9.7 docs and config touched by this stage

- `config/rl.yaml` stays `use_policy: false` (D7 needs the real arm). `docs/tasks/*.md` progress boxes are
  ticked only by the coordinator after the final verification.

### 9.8 Facts from the integration stage (verified; read the module docstrings for the rest)

- Fake scene layout (binding, in config/fake_scene.json and config/scene.yaml): cereal box at (0.28, 0.07), handoff
  zone (0.28, -0.08, 0.10), hand parked at (0.28, -0.08), tea tin at (0.16, -0.02), pill bottle (0.18, -0.12),
  juice box (0.24, -0.04, yaw 0.3), tray (0.30, 0.22) with the 5 cm slots. With the box at the kit's (0.24, 0.11)
  every tray slot sat inside the box's bearing cone and the forearm clipped it at z 0.19 on every release pose.
  Now slots 1, 3 and 4 are clean, slot 2 (0.325, 0.195) is not (SimRobot skips it), and the transit leg at
  z 0.28 above slot 4 is IK-infeasible. A no-transit sweep from the juice box to the tray still hits the box.
  Expect "put the juice box in the tray" to pass with transit_z; a 3-object "clear the table" is hard in sim
  (the demo narrative tolerates failures). A15 re-tapes the real table and may change these numbers.
- SimRobot: `set_roll` formula is `wrist_roll = angle + shoulder_pan + ROLL_OFFSET` (ROLL_OFFSET -1.529, mod pi);
  `stats`/`events` are read-only copies; `reset()` raises only inside `run_skill`, so the runner builds a fresh
  SimRobot per trial and a candidate that calls `reset`/`set_skill` must count as an error (runner hardening);
  `carry_to_hand` makes the object leave the scene (seen false, last pose kept) and ends at home_q; a released
  object stays robot-exempt until 6 cm from the grasp center or 1 s; `place_in_zone(slot=None)` skips slots whose
  approach would collide; one place_in_tray is ~11 s simulated / 0.5 s wall; SimTimeLimit fires at 60 s simulated
  (~2 s wall) and must be caught as BaseException by the runner.
- Replay: `on_event` receives short strings; `arm.real_robot.home()` returns `{"ok", "reason", "events"}`;
  `hand_too_close` leaves the arm frozen (needs `/resume`), as does `stopped`; `replay()` returns `error`
  without freezing when validate/config fails or torque is off; CLIs take `--scene-file` for hand data;
  `arm.real_robot.cancelled()`/`clear_cancel()`; holding threshold = `held_min / |open_pct - closed_pct|`,
  empty-closed reading from optional `gripper.empty_closed_pct` in config/joint_map.yaml.
- Vision: synthetic camera f = 450 px, image u along world -y and v along world -x; the real `SceneTracker`
  omits items never detected; `snapshot(n)` pulls frames synchronously when the thread is not running and
  raises `TimeoutError` on a dead camera; `vision.verify.take_after(tracker, spec, args, *, wait, retry, sleep,
  before, n)` needs `before` to resolve "*"; `verify()` returns reasons "spec satisfied", "spec not satisfied",
  "obstacle X moved N cm (max 2.0 cm)", "unseen after the run: ...", "spec error: ..."; `missing` is informational
  (a successful hand-over lists the taken object); `vision.calibrate.calibration_error` raises `ValueError` when
  no calibration marker is in view, so `/calibrate` catches it and reports an error; `vision.scene.
  fake_chain_tracker()` runs the real detector on the FakeCamera. B8 (one wrist still per grasp) is not done:
  the orchestrator saves `runs/<task_id>/grasp_<n>.jpg` from `vision.cameras.get_camera("wrist").read()` when
  replay reports a grip close (FakeCamera returns a placeholder frame).
- LLM/router/candidates: `route()`, `generate()` and `repair()` raise `ValueError` on a malformed `task_id`
  (`t-0001` style ids are fine); `generate(timeout)` is a whole-call budget and returns late candidates with
  `problems == ["generation timed out after N s"]`; `Candidate` has `index, hint, source, problems, attempt,
  path` and `accepted`; `repair(spec, scene, skills, failures, *, task_dir, timeout)` takes failures as dicts
  with `code`/`source`, `score`, `fail_reason`, `index`, `hint`; `agents.llm.ModelCallFailed` / `InvalidReply`;
  `router.EVAL_SKILLS`; an unseen object or "*" as a run arg becomes `queue`; `is_stop_phrase`. The fixture
  candidates in tests/fixtures/llm/candidates/juice_box_to_tray were written before the scene fix and may need
  tuning against the runner (DESIGN section 6 expectations).
- RL: `rl.handoff_env.build_observation(q_measured, prev_action, target_xyz)` is the 16-float layout the policy
  runner must reuse; env hooks `set_target`, `set_joints`, `applied_command`, `actuator_targets`; `FALL_PENALTY`
  (-50) is applied at the env level on `info["fell"]` (decision: keep it; the D4 prompt must name
  `info["fell"]`, `info["step"]`, `info["success"]`, `info["dist"]`, `info["target"]`, `info["tip"]`);
  `rl.train.train(...)`, `EnvFactory`, `RewardWrapper`, `copy_info`; `rl.evaluate.evaluate/run_episodes/
  load_policy/RESULT_KEYS`; `rl/train.py` loads no `.env` because its workers execute reward code; AI-written
  variants train only in sandboxes (`train_parallel --local` is for tests with the fixture/baseline rewards);
  policies start from home +-5 degrees (D7 risk: widen before real deployment).
- Web: tests/test_web_pages.py and tests/web/*.mjs exist (node-based checks skip without node); STOP re-sends
  up to three times on a hung `/stop`; the pages tolerate the 8.6 shapes.
- Contract modules: `sim.success` resolves obstacle names (`not_moved cereal_box` is valid) and `$zone`;
  `check_source` also rejects `BaseException`/`mro` anywhere, return/break/continue inside `finally`, and async
  comprehensions. `tests/fixtures/trajectory_ok.json` violates config/limits.yaml with the 3 degree margin
  (shoulder_lift -1.75): never use it as a replayable path; build test trajectories with SimRobot or by hand.

### 9.9 Open items carried to the final hardening stage (from the integration-stage verifiers)

- arm/driver.py: `connect()` ends with `stopped = False`, so an Esc pressed during the connect window (port
  open, goal sync, configure) is discarded and the CLI then moves the arm. `connect()` must keep a freeze that
  happened during connect, and every `--real` CLI must check `driver.stopped` / `arm.real_robot.cancelled()`
  right after connect and abort. Also add public helpers so the 0-100 gripper scale stays inside the driver:
  `pct_to_gripper(pct)`, `gripper_to_pct(v)`, `held_threshold(object_name)`, `empty_closed_reading()`; replay
  then stops reading config/objects.yaml percentages itself.
- arm/real_robot.py `carry_to_hand`: rise to a clearance height first (as the sim's pick does), reject hands
  whose target leaves the workspace box or the 0.30 m ceiling (fall back to the handoff zone), guard
  non-finite timeouts, pre-check torque like `replay()`. arm/sim_robot.py `carry_to_hand`: same rise-first
  approach so sim and real agree; `pick()` while holding must re-read the pose after releasing; re-measure the
  tip speed on `_line` legs; `place(x, y)` with the default `release_z` on a 10.5 cm object must not press it
  into the table.
- vision: `scripts/touch_markers.py --real` refuses `FAKE_CAMERA=1`; `SceneTracker` advances the seen timeouts
  when `read()` raises; `take_after` with an injected no-op sleep must not spin.
- arm/replay.py: a scene callable returning a non-dict is an error (fail loud), `--scene-file` is read and
  validated before connecting; the "holding" comparison is inclusive ("by at least").
- agents/router.py: refuse `params` != {} in learn specs; align `not_moved <obstacle>` with
  `vision.verify.missing_objects` (either refuse it in the router or make `missing_objects` ignore obstacles);
  re-raise ValueError/TypeError from `chat` instead of falling back; apply the negation/compound-sentence
  guards to every keyword rule; "*" resolving to no object -> queue "I don't see anything on the table.".
  CLI tests must patch `load_dotenv`.
- rl/train.py: catch OverflowError from a huge reward; a per-call or per-episode reward time budget (and the
  D5 poller's own deadline); refuse `--out ""`; `--check` argument validation.
- web: voice `pendingTasks` keyed through `Object.create(null)`; T forwarding skips modifier keys and forwards
  blur; `drawChart` uses reduce instead of `Math.max.apply`; the "rejected" label only for `error` starting with
  "rejected:"; node-harness regression checks for the 8.7 items.
- tests: pin `OBSTACLE_MAX_MOVE`, the inclusive predicate boundaries and exact event matching in
  tests/test_success_extra.py; pin the D1 constants and a positive-success metrics case in tests/test_env.py;
  pin ATTACH_XY/ATTACH_YAW, MAX_SKILL_DEPTH and the tray collision phrases in tests/test_sim_robot.py; pin the
  replay safety constants (box gate 1 mm, close refusal 0.10 m, bridge tolerance 1 deg).
- repo: `git add config/homography.npy` (binary) and `rl/policies/handoff_best.zip` when it exists; D-GO number
  and decision at the top of docs/tasks/D-rl.md; progress boxes in docs/tasks/*.md ticked only for tasks whose
  hardware-free Done-when passed; the real-arm/camera/key halves are listed as not done.
- Known physical risk (not a code defect): at the tray the arm's reach forces a 30-45 degree release pitch and
  the contract's release height (bottom 1 cm above the wall top) drops cylinders about 4 cm; the sim shows the
  tea tin landing on its side (its marker then faces sideways and the camera in_zone check would fail on the
  real table). A15 should use a lower tray wall or a nearer tray.
