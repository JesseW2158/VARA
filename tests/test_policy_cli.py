"""`python -m rl.policy_runner` (D7 staged bring-up, M4 trial recorder): the fake end-to-end run, the --real
refusals before any prompt or Driver, the --real gating with a stubbed driver and Esc, argument errors, the
report fields, the hand-speed flag, run_policy's `policy stats:` event and the M4 verdict arithmetic.

No hardware, camera, xAI or Daytona: the policy is a stub (`load_policy` monkeypatched), the arm a
FakeDriver, and the scene the FakeSceneTracker or a scene file in tmp_path."""

from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import threading
import time
import types

import numpy as np
import pytest

from arm import driver as driver_mod
from arm import ik, real_robot
from arm.driver import FakeDriver
from rl import policy_runner as pr
from rl.handoff_env import MAX_STEP_RAD

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
RL_YAML = ROOT / "config" / "rl.yaml"


class GraspPolicy:
    """A stand-in policy: one finite-difference Gauss-Newton step on the four policy joints per call,
    aiming the grasp center (`site="grasp"`, what M4 judges) or, like the trained policy, the jaw tip
    (`site="tip"`, run_policy's own success test) at the observed target."""

    def __init__(self, site: str = "grasp") -> None:
        self.site = site
        self.calls = 0

    def _point(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(ik.fk_grasp(q)[0] if self.site == "grasp" else ik.fk_tip(q), dtype=np.float64)

    def predict(self, obs, deterministic=True):
        self.calls += 1
        q = np.asarray(obs[:5], dtype=np.float64)
        goal = np.asarray(obs[12:15], dtype=np.float64)
        err = self._point(q) - goal
        jac = np.zeros((3, 4))
        for j in range(4):
            dq = q.copy()
            dq[j] += 1e-4
            jac[:, j] = (self._point(dq) - goal - err) / 1e-4
        step, *_ = np.linalg.lstsq(jac, -err, rcond=None)
        return np.clip(step / MAX_STEP_RAD, -1.0, 1.0), None


@pytest.fixture
def stub_policy(monkeypatch):
    policy = GraspPolicy()
    loaded: list[str] = []

    def load(path):
        loaded.append(str(path))
        return policy

    monkeypatch.setattr(pr, "load_policy", load)
    # The jaw tip, 2.5 cm ahead of the grasp center, crosses the target first on the descent; with
    # run_policy's own tip test switched off the stub reaches until --timeout, so the grasp-center streak
    # that M4 counts (M4_CONSECUTIVE, unchanged) can grow.
    monkeypatch.setattr(pr, "SUCCESS_STEPS", 10**6)
    return types.SimpleNamespace(policy=policy, loaded=loaded)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    out = tmp_path / "runs"
    monkeypatch.setenv("VARA_RUNS_DIR", str(out))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))
    return out


@pytest.fixture
def no_hardware(monkeypatch):
    """A Driver or EscWatcher construction fails the test: refusals must come before both."""

    def refuse(*args, **kwargs):
        raise AssertionError("a real Driver/EscWatcher was constructed")

    monkeypatch.setattr(driver_mod, "Driver", refuse)
    monkeypatch.setattr(driver_mod, "EscWatcher", refuse)
    monkeypatch.setattr(
        real_robot, "confirm_arm_clear", lambda *a, **k: pytest.fail("prompted before refusing")
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def real_env(monkeypatch):
    for name in ("FAKE", "FAKE_ROBOT", "FAKE_CAMERA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def scene_file(tmp_path, hand=(0.28, -0.08, 0.03), seen=True, name="scene.json") -> pathlib.Path:
    scene = json.loads(json.dumps(FAKE_SCENE))
    scene["hand"].update({"x": hand[0], "y": hand[1], "z": hand[2], "seen": seen})
    path = tmp_path / name
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


class _LiveCamera:
    is_fake = False
    index = 2


class _LiveTracker:
    """The attributes `vision.scene.scene_source` reads from the live overhead SceneTracker."""

    is_fake = False
    lens_sha256 = None

    def __init__(self) -> None:
        self.camera = _LiveCamera()


@contextlib.contextmanager
def live_scene_file(tmp_path, **kwargs):
    """`scene_file` as a live `python -m vision.scene --out` writer leaves it: the provenance sidecar first,
    then the scene, kept fresh from a thread (`--real --scene-file` refuses a file without a live sidecar
    or without a fresh scene: arm.replay.open_scene_file)."""
    from vision import scene as scene_mod

    path = scene_file(tmp_path, **kwargs)
    scene = json.loads(path.read_text(encoding="utf-8"))
    scene_mod.write_scene_source(scene_mod.scene_source(_LiveTracker()), path)
    halt = threading.Event()

    def refresh() -> None:
        while not halt.is_set():
            scene["t"] = time.time()
            with contextlib.suppress(OSError):  # Windows: a reader held the file; the next round writes
                scene_mod.write_scene_atomic(scene, path)
            halt.wait(0.05)

    refresh_thread = threading.Thread(target=refresh, daemon=True)
    scene["t"] = time.time()
    scene_mod.write_scene_atomic(scene, path)
    refresh_thread.start()
    try:
        yield path
    finally:
        halt.set()
        refresh_thread.join(timeout=5)


TRIAL_KEYS = {
    "trial",
    "stage",
    "ok",
    "reason",
    "abort",
    "stopped",
    "policy_ran",
    "policy_result",
    "policy_success",
    "target",
    "target_marker",
    "target_item",
    "readings",
    "policy_s",
    "grasp_min_m",
    "grasp_final_m",
    "grasp_streak_max",
    "within_3cm",
    "tip_min_m",
    "tip_final_m",
    "tip_streak_max",
    "tip_within_3cm",
    "tip_speed_max_mps",
    "tip_speed_cap_ratio_max",
    "over_cap_intervals",
    "over_cap",
    "elapsed_s",
    "hand_speed_max_mps",
    "hand_speed_limit_mps",
    "hand_too_fast",
    "hand_observations",
    "stats",
    "events",
}


# ------------------------------------------------------------------------------------------- fake runs


def test_fake_fixed_stage_runs_five_trials_end_to_end(tmp_path, runs, stub_policy, no_hardware, capsys):
    rl_before = RL_YAML.read_text(encoding="utf-8")
    out = tmp_path / "evidence" / "m4.json"
    rc = pr.main(["--fake", "--stage", "fixed", "--target-marker", "12", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "rl-m4-trials" and report["mode"] == "fake" and report["stage"] == "fixed"
    assert report["hand_source"] == "fake tracker" and report["counts_for_m4"] is False
    assert report["policy"] == str((ROOT / pr.DEFAULT_POLICY).resolve())
    assert len(report["policy_sha256"]) == 64 and report["config_use_policy"] is False
    assert report["caps_mps"] == [0.2, 0.1] and report["radius_m"] == 0.03 and report["consecutive"] == 10
    summary = report["summary"]
    assert summary["trials_run"] == 5 and summary["within_3cm"] == 5 and summary["verdict"] == "pass"
    assert summary["over_cap_trials"] == 0 and summary["required"] == 4
    assert [t["trial"] for t in report["trials"]] == [1, 2, 3, 4, 5]
    item = FAKE_SCENE["objects"]["tea_tin"]
    for trial in report["trials"]:
        assert set(trial) == TRIAL_KEYS
        assert trial["ok"] and trial["policy_ran"] and trial["within_3cm"] and not trial["over_cap"]
        assert trial["target_marker"] == 12 and trial["target_item"] == "tea_tin"
        assert trial["target"] == pytest.approx([item["x"], item["y"], item["h"] + 0.05])
        assert trial["grasp_min_m"] <= trial["grasp_final_m"] < 0.03 and trial["grasp_streak_max"] >= 10
        assert trial["policy_result"].startswith("policy timeout")  # run_policy's tip test is off
        assert 0 < trial["tip_speed_max_mps"] <= 0.1 + 1e-9  # the marker counts as the hand: 0.1 m/s cap
        assert trial["hand_speed_max_mps"] is None and trial["hand_too_fast"] is False
        assert trial["elapsed_s"] > trial["policy_s"] > 0
        events = trial["events"]
        assert f"policy override: use_policy=True, policy={report['policy']}" in events
        assert "handoff_done" in events and any(e.startswith(pr.STATS_PREFIX) for e in events)
    assert stub_policy.loaded and all(path == report["policy"] for path in stub_policy.loaded)
    assert "M4: 5 of 5 within 3 cm" in printed and "-> PASS" in printed and "does not count for M4" in printed
    assert RL_YAML.read_text(encoding="utf-8") == rl_before  # use_policy stays false
    assert real_robot.policy_config()["use_policy"] is False


def test_a_tip_aimed_policy_passes_its_own_test_but_not_the_grasp_center(tmp_path, runs, monkeypatch):
    """The trained policy aims the jaw tip (run_policy's success test); M4 is judged on the grasp center,
    2.5 cm behind the tip, so such a reach is recorded as tip_within_3cm but not within_3cm."""
    monkeypatch.setattr(pr, "load_policy", lambda path: GraspPolicy(site="tip"))
    out = tmp_path / "tip.json"
    rc = pr.main(["--fake", "--stage", "slow", "--trials", "1", "--out", str(out)])
    trial = json.loads(out.read_text(encoding="utf-8"))["trials"][0]
    assert rc == 1 and trial["policy_success"] and trial["tip_within_3cm"] and not trial["within_3cm"]
    assert trial["tip_final_m"] < 0.03 < trial["grasp_final_m"] + 0.01


def test_default_report_is_a_new_timestamped_file_under_vara_runs_dir(runs, stub_policy, no_hardware):
    args = ["--fake", "--stage", "slow", "--trials", "1"]
    assert pr.main(args) == 0
    assert pr.main(args) == 0
    files = sorted((runs / "rl").glob("m4-*.json"))
    assert len(files) == 2 and files[0] != files[1]  # never overwritten, even within one second
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["stage"] == "slow" and report["hand_speed_max_mps"] == 0.02
        trial = report["trials"][0]
        assert trial["hand_speed_max_mps"] == 0.0 and trial["hand_observations"] > 0  # the fake hand is still
        assert trial["target"] == pytest.approx([0.28, -0.08, 0.08])
    assert not list((runs / "rl").glob("*.tmp"))


def test_out_that_exists_is_refused_and_untouched(tmp_path, runs, stub_policy, no_hardware):
    out = tmp_path / "taken.json"
    out.write_text("keep", encoding="utf-8")
    rc = pr.main(["--fake", "--stage", "fixed", "--target-marker", "12", "--out", str(out)])
    assert rc == 2 and out.read_text(encoding="utf-8") == "keep"


def test_hand_that_jumps_is_flagged_but_the_trial_still_counts(tmp_path, runs, stub_policy, monkeypatch):
    path = scene_file(tmp_path)

    class JumpingHandDriver(FakeDriver):
        """Moves the wristband 0.10 m in one tick once the carry is under way."""

        def _send(self, q, gripper):
            super()._send(q, gripper)
            if len(self.commands) == 30:
                scene_file(tmp_path, hand=(0.18, -0.08, 0.03))

    monkeypatch.setattr(driver_mod, "FakeDriver", JumpingHandDriver)
    out = tmp_path / "slow.json"
    rc = pr.main(["--fake", "--stage", "slow", "--trials", "1", "--scene-file", str(path), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    trial = report["trials"][0]
    assert report["hand_source"] == f"scene-file {path}"
    assert trial["hand_too_fast"] is True and trial["hand_speed_max_mps"] > 0.02
    assert trial["hand_speed_limit_mps"] == 0.02 and report["summary"]["hand_too_fast_trials"] == 1
    assert trial["ok"]  # the flag never ends a trial
    assert trial["target"] == pytest.approx([0.18, -0.08, 0.08])  # the policy followed the moved hand
    assert rc == (0 if report["summary"]["verdict"] == "pass" else 1)


def test_unseen_target_marker_is_refused_before_anything_moves(runs, stub_policy, no_hardware, capsys):
    rc = pr.main(["--fake", "--stage", "fixed", "--target-marker", "40"])  # the wristband is no table item
    assert rc == 2 and "marker 40" in capsys.readouterr().err
    assert not (runs / "rl").exists()
    rc = pr.main(["--fake", "--stage", "fixed", "--target-marker", "10"])  # pill_bottle, the held object
    assert rc == 2 and "assumed held" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------- --real


@pytest.mark.parametrize(
    "case",
    ["no scene source", "fake tracker", "scene file under FAKE", "fake scene file", "absent marker"],
)
def test_real_refusals_come_before_any_prompt_or_driver(
    case, tmp_path, runs, stub_policy, no_hardware, monkeypatch
):
    real_env(monkeypatch)
    good = scene_file(tmp_path)
    argv = {
        "no scene source": ["--stage", "slow"],
        "fake tracker": ["--stage", "slow", "--tracker"],
        "scene file under FAKE": ["--stage", "slow", "--scene-file", str(good)],
        "fake scene file": ["--stage", "slow", "--scene-file", str(ROOT / "config" / "fake_scene.json")],
        "absent marker": ["--stage", "fixed", "--target-marker", "33", "--scene-file", str(good)],
    }[case]
    if case == "fake tracker":
        monkeypatch.setenv("FAKE_CAMERA", "1")
    if case == "scene file under FAKE":
        monkeypatch.setenv("FAKE", "1")
    assert pr.main(["--real", *argv]) == 2
    assert not (runs / "rl").exists()


def test_real_with_stdin_at_eof_never_connects(tmp_path, runs, stub_policy, monkeypatch, capsys):
    real_env(monkeypatch)
    made = []

    class StubDriver:
        stopped = False

        def __init__(self, port=None, follower_id=None):
            made.append((port, follower_id))

        def connect(self):
            raise AssertionError("connected without confirmation")

        def freeze(self):
            self.stopped = True

    class StubWatcher:
        pressed = False
        pressed_at = None

        def __init__(self, callback):
            self.callback = callback

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(driver_mod, "Driver", StubDriver)
    monkeypatch.setattr(driver_mod, "EscWatcher", StubWatcher)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with live_scene_file(tmp_path) as live:
        rc = pr.main(["--real", "--stage", "slow", "--scene-file", str(live), "--port", "COM99"])
    assert rc == 1 and made == [("COM99", None)]
    assert "Not confirmed; nothing moved." in capsys.readouterr().out
    assert not (runs / "rl").exists()


def test_real_esc_freezes_and_records_the_latency(tmp_path, runs, stub_policy, monkeypatch, capsys):
    """The --real path end to end with a FakeDriver standing in for the Driver and a virtual clock: Esc
    lands during trial 1, the carry stops, the latency is printed and appended with source and phase."""
    real_env(monkeypatch)
    watchers = []

    class Watcher:
        def __init__(self, callback):
            self.callback = callback
            self.pressed = False
            self.pressed_at = None
            watchers.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def press(self):
            self.pressed = True
            self.pressed_at = __import__("time").monotonic()
            self.callback()

    class ArmDriver(FakeDriver):
        def __init__(self, port=None, follower_id=None):
            super().__init__(gripper0=0.2)
            self.reads = 0
            self.connects = 0

        def connect(self):
            self.connects += 1
            super().connect()

        def read(self):
            reading = super().read()
            self.reads += 1
            if self.reads == 60:
                watchers[0].press()
            return reading

    drivers = []
    monkeypatch.setattr(driver_mod, "Driver", lambda **kw: drivers.append(ArmDriver(**kw)) or drivers[-1])
    monkeypatch.setattr(driver_mod, "EscWatcher", Watcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds, watcher: True)
    clock = pr._VirtualClock()
    real_time = __import__("time")
    monkeypatch.setattr(
        pr,
        "time",
        types.SimpleNamespace(perf_counter=clock.now, sleep=clock.sleep, monotonic=real_time.monotonic),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    out = tmp_path / "real.json"
    with live_scene_file(tmp_path) as live:
        rc = pr.main(["--real", "--stage", "slow", "--scene-file", str(live), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 1
    drv = drivers[0]
    assert drv.connects == 1 and drv.stopped and drv.torque  # frozen with torque on, never dropped
    assert "Esc -> freeze command:" in printed and "Arm is FROZEN with torque on" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "real" and report["esc"]["phase"] == "session"
    assert report["trials"][0]["stopped"] and report["summary"]["verdict"] == "fail"
    log = runs / "calibration" / "esc-latency.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["source"] == "rl.policy_runner" and lines[0]["phase"] == "session"
    assert lines[0]["latency_ms"] is not None and lines[0]["latency_ms"] >= 0
    assert lines[0]["extra"]["frozen"] is True and lines[0]["extra"]["stage"] == "slow"


# ------------------------------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    "argv",
    [
        ["--fake"],  # no --stage
        ["--fake", "--stage", "fixed"],  # no --target-marker
        ["--fake", "--stage", "slow", "--target-marker", "12"],
        ["--fake", "--stage", "fixed", "--target-marker", "12", "--hand-speed-max", "0.02"],
        ["--fake", "--stage", "normal", "--hand-speed-max", "0"],
        ["--fake", "--stage", "slow", "--trials", "0"],
        ["--fake", "--stage", "slow", "--timeout", "0"],
        ["--fake", "--stage", "slow", "--timeout", "nan"],
        ["--fake", "--real", "--stage", "slow"],
        ["--fake", "--stage", "slow", "--tracker", "--scene-file", "x.json"],
        ["--fake", "--stage", "sideways"],
    ],
)
def test_argument_errors_exit_2(argv, runs, no_hardware):
    with pytest.raises(SystemExit) as exc:
        pr.main(argv)
    assert exc.value.code == 2


def test_neither_fake_nor_real_is_refused(runs, no_hardware, monkeypatch, capsys):
    for name in ("FAKE", "FAKE_ROBOT"):
        monkeypatch.delenv(name, raising=False)
    assert pr.main(["--stage", "slow"]) == 2
    assert "without --real" in capsys.readouterr().err


def test_missing_policy_and_unreadable_scene_file_are_refused(tmp_path, runs, stub_policy, no_hardware):
    assert pr.main(["--fake", "--stage", "slow", "--policy", str(tmp_path / "none.zip")]) == 2
    assert pr.main(["--fake", "--stage", "slow", "--scene-file", str(tmp_path / "none.json")]) == 2
    assert not (runs / "rl").exists()


def test_help_prints_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        pr.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "python -m rl.policy_runner" in text and "--target-marker" in text and "--hand-speed-max" in text


# ------------------------------------------------------------------------------------------- pieces


def test_run_policy_emits_its_stats_before_the_last_event():
    driver = FakeDriver()
    tip = ik.fk_tip(driver.read()[0])
    scene = {"hand": {"marker": 40, "x": tip[0], "y": tip[1] - 0.05, "z": tip[2] - 0.1, "seen": True}}
    events: list[str] = []
    pr.run_policy(
        driver, lambda: scene, "unused.zip", policy=GraspPolicy(), timeout=4.0, on_event=events.append
    )
    assert events[-2].startswith(pr.STATS_PREFIX) and events[-1].startswith("policy ")
    stats = pr.policy_stats(events)
    assert stats["readings"] >= len(driver.commands) > 0 and stats["failure"] is None
    assert stats["grasp_final_m"] is not None and stats["tip_final_m"] is not None
    assert stats["tip_speed_max_mps"] <= 0.2 + 1e-9 and stats["over_cap_intervals"] == 0
    assert stats["caps_mps"] == [0.2, 0.1] and stats["intervals"] == stats["readings"] - 1
    assert pr.policy_stats(["policy start: x.zip", "policy stopped"]) is None


def test_trace_flags_an_interval_over_its_cap():
    trace = pr._Trace(0.0, (0.2, 0.1, 0.15))
    q = list(ik.home_q())
    tip = np.array(ik.fk_tip(q))
    target = tip + [0.0, 0.0, -0.05]
    trace.observe(0.0, q, tip, target, None)
    trace.observe(1 / 30, q, tip + [0.005, 0.0, 0.0], target, None)  # 0.15 m/s, under 0.2
    trace.observe(2 / 30, q, tip + [0.005, 0.0, 0.0], target, tuple(tip))  # still, hand near
    trace.observe(3 / 30, q, tip + [0.010, 0.0, 0.0], target, tuple(tip))  # 0.15 m/s near the hand: over 0.1
    stats = json.loads(trace.event()[len(pr.STATS_PREFIX) :])
    assert stats["intervals"] == 3 and stats["near_hand_intervals"] == 2 and stats["over_cap_intervals"] == 1
    assert stats["tip_speed_max_mps"] == pytest.approx(0.15) and stats[
        "tip_speed_cap_ratio_max"
    ] == pytest.approx(1.5)


def test_marker_target_rules():
    scene = json.loads(json.dumps(FAKE_SCENE))
    tea = scene["objects"]["tea_tin"]  # the marker lies on the top face: the target is (x, y, h)
    assert pr.marker_target(scene, 12) == ("tea_tin", (tea["x"], tea["y"], tea["h"]))
    assert pr.marker_target(scene, 30)[0] == "cereal_box"
    scene["objects"]["tea_tin"]["seen"] = False
    scene["obstacles"]["cereal_box"]["seen"] = False
    assert pr.marker_target(scene, 12) is None  # an unseen object is not a target
    assert pr.marker_target(scene, 30)[0] == "cereal_box"  # an obstacle keeps its last pose
    scene["objects"]["juice_box"]["marker"] = 30
    with pytest.raises(ValueError, match="more than one"):
        pr.marker_target(scene, 30)


def test_hand_speed_meter_ignores_jitter_and_measures_motion():
    t = [0.0]
    meter = pr.HandSpeedMeter(lambda: t[0])
    rng = np.random.default_rng(0)
    for k in range(90):  # 3 s at 30 fps, 1 mm marker jitter
        t[0] = k / 30
        x, y = 0.28 + rng.uniform(-0.001, 0.001), -0.08 + rng.uniform(-0.001, 0.001)
        meter.observe({"hand": {"x": x, "y": y, "z": 0.03, "seen": True}})
    assert meter.peak < 0.01 and meter.observations == 90
    moving = pr.HandSpeedMeter(lambda: t[0])
    for k in range(90):
        t[0] = k / 30
        moving.observe({"hand": {"x": 0.20 + 0.05 * t[0], "y": 0.0, "z": 0.03, "seen": True}})
    assert moving.peak == pytest.approx(0.05, rel=0.02)


@pytest.mark.parametrize(
    ("within", "over", "ran", "requested", "verdict"),
    [
        (4, 0, 5, 5, "pass"),
        (5, 0, 5, 5, "pass"),
        (3, 0, 5, 5, "fail"),
        (5, 1, 5, 5, "fail"),  # one trial over a cap
        (4, 0, 4, 5, "fail"),  # a trial missing (STOP)
        (1, 0, 1, 1, "pass"),
        (8, 0, 10, 10, "pass"),
        (7, 0, 10, 10, "fail"),
    ],
)
def test_m4_verdict_arithmetic(within, over, ran, requested, verdict):
    records = [
        {"within_3cm": k < within, "over_cap": k < over, "policy_ran": True, "tip_within_3cm": True}
        for k in range(ran)
    ]
    summary = pr.m4_summary(records, requested)
    assert summary["verdict"] == verdict
    assert summary["within_3cm"] == within and summary["required"] == math.ceil(0.8 * requested)
    assert summary["trials_run"] == ran and summary["over_cap_trials"] == over


def test_m4_unverified_speed_fails_and_bad_counts_raise():
    records = [{"within_3cm": True, "over_cap": False, "policy_ran": True} for _ in range(5)]
    records[2] = {"within_3cm": True, "over_cap": None, "policy_ran": True}
    summary = pr.m4_summary(records, 5)
    assert summary["unverified_speed_trials"] == 1 and summary["verdict"] == "fail"
    with pytest.raises(ValueError):
        pr.m4_summary([], 0)


def test_real_scene_file_without_a_live_sidecar_is_refused_before_anything(
    tmp_path, runs, stub_policy, monkeypatch, capsys
):
    """HANDOFF safety lead 4 (W2d) for this CLI too: a scene file with no provenance sidecar (a fake writer,
    a hand-made file) is refused under --real before any prompt and before a Driver exists."""
    real_env(monkeypatch)
    monkeypatch.setattr(driver_mod, "Driver", lambda **kw: pytest.fail("a Driver was constructed"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted although refusing"))
    rc = pr.main(["--real", "--stage", "slow", "--scene-file", str(scene_file(tmp_path))])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing --scene-file" in err and "nothing moved" in err
