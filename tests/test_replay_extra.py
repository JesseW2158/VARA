"""arm.replay helpers beyond the contract: bridge, load_limits, load_box, load_objects, validate_path, the
validate hardening (huge ints, NaN or inverted limits, malformed box), Pacer and the hand parsing."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arm import replay
from arm.driver import FakeDriver

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
LIMITS = {
    "shoulder_pan": [-1.9, 1.9],
    "shoulder_lift": [-1.9, 1.9],
    "elbow_flex": [-1.7, 1.7],
    "wrist_flex": [-1.7, 1.7],
    "wrist_roll": [-2.7, 2.7],
}
BOX = {"x": [0.10, 0.40], "y": [-0.25, 0.25], "z": [0.02, 0.30]}
Q0 = [0.0, -1.0, 0.5, 1.0, 0.0]


def fk_inside(q):
    return (0.25, 0.0, 0.10)


def traj(points):
    return {
        "version": 1,
        "skill": "extra",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [{"kind": "path", "q": points}],
    }


# -- bridge --------------------------------------------------------------------------------------------------


def test_bridge_steps_are_bounded_and_end_exactly_at_the_target():
    q_to = [v + d for v, d in zip(Q0, [10 * DEG, -2.5 * DEG, 0.0, 0.3 * DEG, 0.0], strict=True)]
    path = replay.bridge(Q0, q_to)
    assert len(path) == 10
    previous = Q0
    for q in path:
        assert max(abs(a - b) for a, b in zip(previous, q, strict=True)) <= 1 * DEG + 1e-12
        previous = q
    assert path[-1] == q_to
    assert path[0] != Q0
    assert replay.bridge(Q0, Q0) == [Q0]
    assert len(replay.bridge(Q0, [Q0[0] + 2.5 * DEG, *Q0[1:]])) == 3
    assert len(replay.bridge(Q0, q_to, max_step_rad=2 * DEG)) == 5
    with pytest.raises(ValueError):
        replay.bridge(Q0, Q0[:4])
    with pytest.raises(ValueError):
        replay.bridge(Q0, q_to, max_step_rad=0.0)


# -- config loaders ------------------------------------------------------------------------------------------


def test_load_limits_reads_the_config_and_rejects_bad_files(tmp_path):
    limits = replay.load_limits()
    assert set(limits) >= set(replay.JOINTS)
    assert all(lo < hi for lo, hi in limits.values())
    with pytest.raises(FileNotFoundError):
        replay.load_limits(tmp_path / "missing.yaml")
    f = tmp_path / "limits.yaml"
    f.write_text("\n".join(f"{j}: [-1.0, 1.0]" for j in replay.JOINTS[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrist_roll"):
        replay.load_limits(f)
    f.write_text("\n".join(f"{j}: [-1.0, 1.0]" for j in replay.JOINTS) + "\nshoulder_pan: [1.0, -1.0]\n")
    with pytest.raises(ValueError, match="shoulder_pan"):
        replay.load_limits(f)
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        replay.load_limits(f)
    f.write_text("\n".join(f"{j}: [-1.0, 1.0]" for j in replay.JOINTS) + "\ngripper: {closed_pct: 0}\n")
    assert replay.load_limits(f)["shoulder_pan"] == [-1.0, 1.0]  # non-range entries are skipped


def test_load_box_reads_the_workspace_and_rejects_bad_files(tmp_path):
    box = replay.load_box()
    assert box == {"x": [0.10, 0.40], "y": [-0.25, 0.25], "z": [0.02, 0.30]}
    f = tmp_path / "scene.yaml"
    f.write_text("home_q: [0, 0, 0, 0, 0]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workspace"):
        replay.load_box(f)
    f.write_text("workspace: {x: [0.1, 0.4], y: [-0.25, 0.25], z: [0.3, 0.02]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="z"):
        replay.load_box(f)
    f.write_text("workspace: {x: [0.1, 0.4, 0.5], y: [-0.25, 0.25], z: [0.02, 0.3]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="x"):
        replay.load_box(f)
    f.write_text("workspace: {x: [0.1, 0.4], y: [-0.25, .nan], z: [0.02, 0.3]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="y"):
        replay.load_box(f)


def test_load_objects_by_name(tmp_path):
    objects = replay.load_objects()
    assert {"pill_bottle", "juice_box", "tea_tin", "cereal_box"} <= set(objects)
    assert objects["juice_box"]["held_min"] == 8
    assert replay.load_objects(tmp_path / "missing.yaml") == {}
    f = tmp_path / "objects.yaml"
    f.write_text("10: {name: thing, kind: object, held_min: 12}\n11: {kind: object}\n", encoding="utf-8")
    assert replay.load_objects(f) == {"thing": {"name": "thing", "kind": "object", "held_min": 12}}


# -- validate hardening --------------------------------------------------------------------------------------


def test_huge_int_waypoint_is_not_a_finite_number():
    huge = json.loads("9" * 400)  # json.loads of a 400 digit literal: an int float() cannot represent
    with pytest.raises(OverflowError):
        float(huge)
    q = list(Q0)
    q[2] = huge
    problems = replay.validate(traj([Q0, q]), LIMITS, fk_inside, BOX)
    assert any("elbow_flex is not a finite number" in p for p in problems), problems
    assert all(len(p) < 200 for p in problems)  # the 400 digits are not echoed
    # the same inside validate_path, and as a limit
    assert any("not a finite number" in p for p in replay.validate_path([q], LIMITS, fk_inside, BOX))
    bad_limits = dict(LIMITS, wrist_flex=[-huge, huge])
    problems = replay.validate(traj([Q0]), bad_limits, fk_inside, BOX)
    assert any(p.startswith("limits: wrist_flex") for p in problems), problems


def test_non_finite_and_inverted_limits_are_reported():
    nan_limits = dict(LIMITS, shoulder_pan=[math.nan, math.nan])
    problems = replay.validate(traj([Q0]), nan_limits, fk_inside, BOX)
    assert any("limits: shoulder_pan" in p and "finite" in p for p in problems), problems
    inf_limits = dict(LIMITS, elbow_flex=[-math.inf, math.inf])
    assert any("elbow_flex" in p for p in replay.validate(traj([Q0]), inf_limits, fk_inside, BOX))
    inverted = dict(LIMITS, shoulder_lift=[1.0, -1.0])
    problems = replay.validate(traj([Q0]), inverted, fk_inside, BOX)
    assert any("shoulder_lift" in p and "inverted" in p for p in problems), problems
    empty = dict(LIMITS, wrist_roll=[0.5, 0.5])
    assert any("wrist_roll" in p for p in replay.validate(traj([Q0]), empty, fk_inside, BOX))
    narrow = dict(LIMITS, wrist_roll=[0.0, 0.05])  # 2.9 degrees: nothing fits inside the margin
    problems = replay.validate(traj([Q0]), narrow, fk_inside, BOX)
    assert any("wrist_roll" in p and "margin" in p for p in problems), problems
    missing = {k: v for k, v in LIMITS.items() if k != "wrist_flex"}
    assert any(
        "no [low, high] range for joint 'wrist_flex'" in p
        for p in replay.validate(traj([Q0]), missing, fk_inside, BOX)
    )
    malformed = dict(LIMITS, shoulder_pan=[1, 2, 3])
    assert any("shoulder_pan" in p for p in replay.validate(traj([Q0]), malformed, fk_inside, BOX))
    # a bad limit never hides a bad waypoint on another joint
    q = list(Q0)
    q[0] = 2.5
    problems = replay.validate(traj([q]), inverted, fk_inside, BOX)
    assert any("shoulder_pan = 2.5000" in p for p in problems)
    # the contract's own limits and example still pass
    assert replay.validate(traj([Q0]), LIMITS, fk_inside, BOX) == []


def test_malformed_box_is_reported_not_raised():
    for box, axis in (
        ({"x": [0.1], "y": BOX["y"], "z": BOX["z"]}, "x"),
        ({"x": BOX["x"], "y": [math.nan, 0.25], "z": BOX["z"]}, "y"),
        ({"x": BOX["x"], "y": BOX["y"], "z": [0.30, 0.02]}, "z"),
        ({"x": BOX["x"], "y": BOX["y"]}, "z"),
        ({"x": BOX["x"], "y": BOX["y"], "z": "tall"}, "z"),
    ):
        problems = replay.validate(traj([Q0]), LIMITS, fk_inside, box)
        assert any(p.startswith("box:") and f"'{axis}'" in p for p in problems), (box, problems)
    # the other axes are still checked
    box = {"x": BOX["x"], "y": BOX["y"], "z": [0.30, 0.02]}
    problems = replay.validate(traj([Q0]), LIMITS, lambda q: (0.50, 0.0, 0.10), box)
    assert any("x not in" in p for p in problems)
    assert any("box:" in p for p in problems)


def test_validate_path_applies_the_waypoint_rules():
    assert replay.validate_path([Q0], LIMITS, fk_inside, BOX) == []
    far = list(Q0)
    far[0] += 3 * DEG
    assert any("steps 3.00 deg" in p for p in replay.validate_path([Q0, far], LIMITS, fk_inside, BOX))
    outside = list(Q0)
    outside[0] = 2.5
    assert any("limits minus 3 deg" in p for p in replay.validate_path([outside], LIMITS, fk_inside, BOX))
    assert any("workspace box" in p for p in replay.validate_path([Q0], LIMITS, lambda q: (0.5, 0, 0.1), BOX))
    assert replay.validate_path("nope", LIMITS, fk_inside, BOX)
    assert any("4" in p for p in replay.validate_path([Q0[:4]], LIMITS, fk_inside, BOX))


# -- Pacer and Session helpers -------------------------------------------------------------------------------


def test_pacer_simulated_clock_and_injection():
    p = replay.Pacer(False)
    t0 = p.now()
    p.tick()
    assert p.now() - t0 == pytest.approx(replay.TICK)
    p.wait(0.4)
    assert p.now() - t0 == pytest.approx(replay.TICK + 0.4)
    started = time.perf_counter()
    for _ in range(300):
        p.tick()
    assert time.perf_counter() - started < 0.5  # 10 s of simulated ticks, no real sleeping
    assert p.now() - t0 == pytest.approx(replay.TICK * 301 + 0.4)
    clock = {"t": 100.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    p = replay.Pacer(False, clock=lambda: clock["t"], sleep=sleep)
    p.tick()
    p.tick()
    assert slept == pytest.approx([replay.TICK, replay.TICK]) and clock["t"] == pytest.approx(
        100 + 2 * replay.TICK
    )
    # a slot more than a tick late resyncs instead of bursting
    clock["t"] += 1.0
    p.tick()
    assert len(slept) == 2
    p.tick()
    assert len(slept) == 3


def test_session_hand_parsing():
    d = FakeDriver()
    scenes = {"scene": None}
    s = replay.Session(d, lambda: scenes["scene"], replay.Pacer(False))
    for broken in (None, [1, 2, 3], "scene", 7):  # a source that returns no dict is a broken tracker
        scenes["scene"] = broken
        with pytest.raises(ValueError, match="scene source returned"):
            s.scene_dict()
        with pytest.raises(ValueError):
            s.hand()
    scenes["scene"] = {"hand": {"marker": 40, "x": 0.3, "y": 0.0, "z": 0.03, "seen": False}}
    assert s.hand() is None
    scenes["scene"]["hand"]["seen"] = True
    assert s.hand() == (0.3, 0.0, 0.03)
    del scenes["scene"]["hand"]["z"]
    with pytest.raises(ValueError):
        s.hand()
    scenes["scene"]["hand"]["z"] = math.nan
    with pytest.raises(ValueError):
        s.hand()
    scenes["scene"] = {"objects": {}}  # no hand entry at all: no hand data
    assert s.hand() is None
    assert replay.Session(d, None, replay.Pacer(False)).hand() is None


def test_held_threshold_and_empty_closed_use_the_joint_map(monkeypatch, tmp_path):
    from arm import driver as driver_mod

    d = FakeDriver()
    s = replay.Session(d, None, replay.Pacer(False))
    assert s.held_threshold("juice_box") == pytest.approx(0.08)  # 8 of 100
    assert s.held_threshold("unknown") == pytest.approx(0.08)  # the smallest held_min on file
    assert s.empty_closed() == 0.0
    jm = tmp_path / "joint_map.yaml"
    jm.write_text("gripper: {closed_pct: 10, open_pct: 90, empty_closed_pct: 14}\n", encoding="utf-8")
    monkeypatch.setattr(driver_mod, "JOINT_MAP_PATH", jm)
    assert s.held_threshold("juice_box") == pytest.approx(8 / 80)
    assert s.empty_closed() == pytest.approx(4 / 80)


def test_abort_reasons_are_closed():
    with pytest.raises(ValueError):
        replay.Abort("tired")
    assert replay.Abort("stopped").result(["a"]) == {"ok": False, "reason": "stopped", "events": ["a"]}
    assert replay.CloseRefused().reason == "hand_too_close"


# -- CLI -----------------------------------------------------------------------------------------------------


def test_cli_validates_and_fake_replays(tmp_path):
    from arm import ik

    home = list(ik.home_q())
    up = list(home)
    up[0] += 1 * DEG
    t = traj([home, up])
    t["segments"].append({"kind": "grip", "action": "open"})
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(t), encoding="utf-8")
    env = {k: os.environ.get(k, "") for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")} | {
        "PYTHONPATH": str(ROOT)
    }
    r = subprocess.run(
        [sys.executable, "-m", "arm.replay", "--traj", str(f)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK:")
    r = subprocess.run(
        [sys.executable, "-m", "arm.replay", "--fake", "--traj", str(f)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["ok"] and result["reason"] is None and "done" in result["events"]
    bad = traj([[0.0, 0.9, -0.2, 0.0, 0.0]])
    f.write_text(json.dumps(bad), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "arm.replay", "--fake", "--traj", str(f)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 1 and "nothing moved" in r.stdout


# -- planner, scene file reader and CLI hardening (adversarial review round 1) ---------------------------


def test_bridge_rejects_non_finite_targets():
    with pytest.raises(ValueError):
        replay.bridge(Q0, [math.nan, *Q0[1:]])
    with pytest.raises(ValueError):
        replay.bridge([math.inf, *Q0[1:]], Q0)


def test_shrunk_box_and_inside_box():
    assert replay.shrunk_box(BOX, 0.01) == {"x": [0.11, 0.39], "y": [-0.24, 0.24], "z": [0.03, 0.29]}
    assert replay.inside_box((0.10, 0.0, 0.30), BOX)
    assert not replay.inside_box((0.10, 0.0, 0.3005), BOX)
    assert replay.inside_box((0.10, 0.0, 0.3005), BOX, 0.001)


def test_plan_path_sweeps_when_it_can_and_lines_otherwise():
    from arm import ik

    home = list(ik.home_q())
    s = replay.Session(FakeDriver(), None, replay.Pacer(False))
    s.start()
    near = list(home)
    near[0] += 5 * DEG
    path, problems, how = s.plan_path(near)
    assert how == "direct" and problems == [] and len(path) == 5 and path[-1] == near
    transit, ok = ik.solve((0.30, 0.22, 0.28), 0.0)
    assert ok
    path, problems, how = s.plan_path(transit)
    assert how.startswith("straight line") and problems == []
    assert replay.validate_path(path, s.limits, ik.fk_tip, s.box) == []
    assert path[-1] == [float(v) for v in transit]
    tips = [ik.fk_tip(q) for q in path]
    assert all(z <= s.box["z"][1] - replay.PLAN_INSET_M + 1e-9 for _, _, z in tips)
    below = [0.0, 0.9, -0.2, 0.0, 0.0]  # tip under the table: no plan keeps the tip inside
    path, problems, how = s.plan_path(near, start=below)
    assert path is None and any("workspace box" in p for p in problems)
    assert any(p.startswith("straight line:") for p in problems)


def test_scene_file_reader_reads_once_and_keeps_the_last_good_scene(tmp_path):
    assert replay.scene_file_reader(None) is None
    with pytest.raises(OSError):
        replay.scene_file_reader(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        replay.scene_file_reader(str(bad))
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        replay.scene_file_reader(str(bad))
    good = tmp_path / "scene.json"
    good.write_text(json.dumps({"hand": {"seen": True, "x": 0.3, "y": 0.0, "z": 0.03}}), encoding="utf-8")
    reader = replay.scene_file_reader(str(good))
    assert reader()["hand"]["x"] == 0.3
    good.write_text(json.dumps({"hand": {"seen": True, "x": 0.2, "y": 0.0, "z": 0.03}}), encoding="utf-8")
    assert reader()["hand"]["x"] == 0.2
    good.write_text("{partial", encoding="utf-8")  # a rewrite in progress keeps the last good scene
    assert reader()["hand"]["x"] == 0.2


def cli(args, cwd=ROOT):
    env = {k: os.environ.get(k, "") for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")} | {
        "PYTHONPATH": str(ROOT)
    }
    return subprocess.run(
        [sys.executable, "-m", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )


def test_cli_refuses_unreadable_inputs_with_one_line(tmp_path):
    from arm import ik

    r = cli(["arm.replay", "--traj", str(tmp_path / "missing.json")])
    assert r.returncode == 2 and "cannot read --traj" in r.stderr and "Traceback" not in r.stderr
    not_json = tmp_path / "not.json"
    not_json.write_text("{nope", encoding="utf-8")
    r = cli(["arm.replay", "--traj", str(not_json)])
    assert r.returncode == 2 and "cannot read --traj" in r.stderr and "Traceback" not in r.stderr
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(traj([list(ik.home_q())])), encoding="utf-8")
    for scene in (tmp_path / "typo.json", not_json):
        r = cli(["arm.replay", "--fake", "--traj", str(f), "--scene-file", str(scene)])
        assert r.returncode == 2 and "cannot read --scene-file" in r.stderr, r.stderr
        assert "{" not in r.stdout and "Traceback" not in r.stderr  # no replay ran
        r = cli(["arm.real_robot", "--fake", "--home", "--scene-file", str(scene)])
        assert r.returncode == 2 and "cannot read --scene-file" in r.stderr, r.stderr
    good = tmp_path / "scene.json"
    good.write_text(json.dumps({"hand": {"seen": False, "x": 0.3, "y": 0.0, "z": 0.03}}), encoding="utf-8")
    r = cli(["arm.replay", "--fake", "--traj", str(f), "--scene-file", str(good)])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["ok"]


# -- Esc during Driver.connect() (adversarial review round 2) ------------------------------------------------


class _StubWatcher:
    """EscWatcher stand-in: `callback` is called from inside the stub driver's connect()."""

    instances: list[_StubWatcher] = []

    def __init__(self, callback=None, **kwargs):
        self.callback = callback
        self.pressed = False
        _StubWatcher.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass


class _StubDriver(FakeDriver):
    """Starts unconnected like the real Driver; connect() can fire the Esc callback and then, exactly like
    Driver.connect(), clears the stopped flag."""

    is_fake = False
    esc_during_connect = False

    def __init__(self, *, port=None, follower_id=None):
        super().__init__()
        self.connected = False
        self.torque = False

    def connect(self):
        if _StubDriver.esc_during_connect:
            watcher = _StubWatcher.instances[-1]
            watcher.callback()
            watcher.pressed = True
        with self.lock:
            self.connected = True
            self.torque = True
            self.stopped = False
            self.last_command = None

    def _sleep(self, seconds):
        pass


@pytest.fixture
def real_cli(monkeypatch, tmp_path):
    from arm import driver as driver_mod
    from arm import ik, real_robot

    drivers: list[_StubDriver] = []

    def make(**kwargs):
        d = _StubDriver(**kwargs)
        drivers.append(d)
        return d

    monkeypatch.setattr(driver_mod, "Driver", make)
    monkeypatch.setattr(driver_mod, "EscWatcher", _StubWatcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: True)
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(replay.time, "sleep", lambda s: None)
    _StubWatcher.instances.clear()
    _StubDriver.esc_during_connect = False
    home = list(ik.home_q())
    near = list(home)
    near[0] += 0.5 * DEG
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(traj([home, near])), encoding="utf-8")
    yield f, drivers
    real_robot.clear_cancel()
    _StubDriver.esc_during_connect = False


def test_cli_esc_during_connect_freezes_and_moves_nothing(real_cli, capsys):
    from arm import real_robot

    f, drivers = real_cli
    _StubDriver.esc_during_connect = True
    rc = replay.main(["--real", "--traj", str(f)])
    assert rc == 1
    d = drivers[-1]
    assert d.commands == [] and d.stopped and d.torque and not d.connected  # frozen, then the port closed
    assert "Esc pressed while connecting" in capsys.readouterr().out
    real_robot.clear_cancel()
    rc = real_robot.main(["--real", "--home"])
    assert rc == 1
    d = drivers[-1]
    assert d.commands == [] and d.stopped and d.torque
    assert "Esc pressed while connecting" in capsys.readouterr().out


def test_cli_without_esc_runs_and_a_stale_cancel_flag_does_not_count(real_cli, capsys):
    from arm import real_robot

    f, drivers = real_cli
    real_robot._CANCEL.set()  # left over from an earlier freeze in this process
    rc = replay.main(["--real", "--traj", str(f)])
    assert rc == 0, capsys.readouterr()
    d = drivers[-1]
    assert d.commands and not d.stopped
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"]


@pytest.mark.parametrize("entry", ["replay", "home", "jog"])
@pytest.mark.parametrize("stop_at", ["countdown", "connect"])
def test_arm_cli_latches_stop_before_and_during_connect(real_cli, monkeypatch, entry, stop_at):
    from arm import driver as driver_mod
    from arm import real_robot

    path, drivers = real_cli
    connects = []
    original_connect = _StubDriver.connect

    def connect(driver):
        connects.append(driver)
        original_connect(driver)
        if stop_at == "connect":
            driver.freeze()  # no watcher.pressed or global cancellation flag

    def confirmed(seconds, watcher):
        assert watcher.callback is not None  # STOP is bound throughout confirmation/countdown
        if stop_at == "countdown":
            watcher.pressed = True
            watcher.callback()
        return True

    monkeypatch.setattr(_StubDriver, "connect", connect)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", confirmed)
    monkeypatch.setattr(driver_mod, "_countdown", confirmed)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    if entry == "replay":
        result = replay.main(["--real", "--traj", str(path)])
    elif entry == "home":
        result = real_robot.main(["--real", "--home"])
    else:
        result = driver_mod.main(["--real", "--jog"])
    assert result == 1
    assert bool(connects) == (stop_at == "connect")
    assert drivers[-1].stopped and not drivers[-1].connected
    assert drivers[-1].commands == []


def test_driver_countdown_requires_confirmation(monkeypatch):
    from arm import driver as driver_mod

    monkeypatch.setattr("builtins.input", lambda *args: "no")
    monkeypatch.setattr(driver_mod.time, "sleep", lambda *_: pytest.fail("countdown after refusal"))
    assert driver_mod._countdown(3, _StubWatcher()) is False
