"""`--tracker` on `python -m arm.replay` and `python -m arm.real_robot` (A10 live hand source): the
FakeSceneTracker under --fake, the --real refusals before any confirmation prompt or Driver (a fake
tracker, a real SceneTracker on the FakeCamera, one that cannot be built, no frame within 5 s, and a dead
camera whose tracker keeps publishing missed grabs without delivering a frame), the --scene-file
exclusion, the tracker closed after a run and after a refusal, and the 10 Hz hand lines of carry_to_hand
(hand xy and distance to the grasp center). No hardware, no camera; the only waits are the stubs' 10 ms
publications under sub-second timeouts (the FakeSceneTracker's wait_for_frame sleeps, but the fake path
never calls it)."""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from pathlib import Path

import pytest

from arm import driver as driver_mod
from arm import ik, real_robot, replay
from arm.driver import FakeDriver
from vision import scene as scene_mod

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
HOME = [float(v) for v in ik.home_q()]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
HAND_LINE = re.compile(
    r"^(carry travel|carry_to_hand) t=(\d+\.\d) s: hand \((-?\d+\.\d{3}), (-?\d+\.\d{3})\), "
    r"(\d+\.\d{3}) m from the grasp center$"
)


def traj_file(tmp_path: Path) -> Path:
    near = list(HOME)
    near[0] += 0.5 * DEG
    t = {
        "version": 1,
        "skill": "tracker_test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [{"kind": "path", "q": [HOME, near]}],
    }
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(t), encoding="utf-8")
    return f


def json_line(out: str) -> dict:
    return json.loads(next(line for line in out.splitlines() if line.startswith("{")))


class StubTracker:
    """A SceneTracker stand-in: `is_fake` False unless removed, frames on request, `latest` counted."""

    is_fake = False

    def __init__(self, *, frames: bool = True, hand_seen: bool = False) -> None:
        self.frames = frames
        self.waits: list[float | None] = []
        self.closed = 0
        self.latest_calls = 0
        self.scene = copy.deepcopy(FAKE_SCENE)
        self.scene["hand"]["seen"] = hand_seen

    def wait_for_frame(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self.frames

    def latest(self) -> dict:
        self.latest_calls += 1
        return copy.deepcopy(self.scene)

    def close(self) -> None:
        self.closed += 1


class StubWatcher:
    def __init__(self, callback=None, **kwargs) -> None:
        self.callback = callback
        self.pressed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class StubDriver(FakeDriver):
    """Starts unconnected like the real Driver; connect() powers it like Driver.connect()."""

    is_fake = False

    def __init__(self, *, port=None, follower_id=None) -> None:
        super().__init__()
        self.connected = False
        self.torque = False

    def connect(self) -> None:
        with self.lock:
            self.connected = True
            self.torque = True
            self.stopped = False
            self.last_command = None

    def _sleep(self, seconds: float) -> None:
        pass


@pytest.fixture
def no_dotenv(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    scene_mod.reset_tracker()
    yield
    scene_mod.reset_tracker()
    real_robot.clear_cancel()


@pytest.fixture
def no_hardware(no_dotenv, monkeypatch):
    """--real must never get past the tracker check: a Driver or a confirmation prompt fails the test."""

    def no_driver(**kwargs):
        raise AssertionError("Driver constructed although the tracker was refused")

    def no_prompt(*a, **k):
        raise AssertionError("confirmation prompt reached although the tracker was refused")

    monkeypatch.setattr(driver_mod, "Driver", no_driver)
    monkeypatch.setattr(driver_mod, "EscWatcher", no_driver)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", no_prompt)
    monkeypatch.setattr("builtins.input", no_prompt)


@pytest.fixture
def real_stubs(no_dotenv, monkeypatch):
    """A --real run on stubs: the Driver, the Esc watcher and the confirmation are stand-ins."""
    drivers: list[StubDriver] = []

    def make(**kwargs):
        d = StubDriver(**kwargs)
        drivers.append(d)
        return d

    monkeypatch.setattr(driver_mod, "Driver", make)
    monkeypatch.setattr(driver_mod, "EscWatcher", StubWatcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: True)
    monkeypatch.setattr("builtins.input", lambda *args: "")  # the "Press Enter to release the port" prompt
    monkeypatch.setattr(replay.time, "sleep", lambda s: None)
    return drivers


# -- --fake --tracker ----------------------------------------------------------------------------------------


def test_fake_tracker_feeds_the_carry_test_without_a_scene_file(no_dotenv, capsys):
    rc = real_robot.main(["--fake", "--carry-test", "--tracker"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json_line(out)
    assert result["ok"] and "handoff_done" in result["events"]
    assert any(e.startswith("hand steady within") for e in result["events"])  # the fake hand was seen
    hand_lines = [e for e in result["events"] if HAND_LINE.match(e)]
    assert hand_lines and all("(0.280, -0.080)" in e for e in hand_lines)  # the fake scene's wristband
    assert scene_mod._TRACKER is None  # the fake path never touches the process-wide singleton


def test_fake_tracker_feeds_a_replay(no_dotenv, tmp_path, capsys):
    f = traj_file(tmp_path)
    rc = replay.main(["--fake", "--traj", str(f), "--tracker"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json_line(out)
    assert result["ok"] and "done" in result["events"]
    assert scene_mod._TRACKER is None


def test_tracker_scene_helper_fake_and_refusals(no_dotenv, monkeypatch):
    tracker, scene = replay.tracker_scene(True, hand_seen=True)
    assert isinstance(tracker, scene_mod.FakeSceneTracker) and tracker.is_fake
    assert scene()["hand"]["seen"] is True and scene.__self__ is tracker  # the tracker's own latest
    assert replay.tracker_scene(True, hand_seen=False)[1]()["hand"]["seen"] is False
    assert replay.tracker_scene(True)[1]()["hand"]["seen"] is FAKE_SCENE["hand"]["seen"]
    monkeypatch.setenv("FAKE_CAMERA", "1")
    scene_mod.reset_tracker()
    with pytest.raises(RuntimeError, match="a fake scene must never feed the real arm"):
        replay.tracker_scene(False)
    stub = StubTracker(frames=False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stub)
    with pytest.raises(TimeoutError, match="no frame within 5 s"):
        replay.tracker_scene(False)
    assert stub.waits == [5.0] and stub.closed == 1  # a camera that shows nothing is released again
    stub = StubTracker(frames=False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stub)
    with pytest.raises(TimeoutError, match="no frame within 0.2 s"):
        replay.tracker_scene(False, timeout=0.2)
    assert stub.waits == [0.2]
    good = StubTracker()
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: good)
    tracker, scene = replay.tracker_scene(False)
    assert tracker is good and scene.__self__ is good and good.closed == 0


# -- --real --tracker refusals before any prompt or Driver -----------------------------------------------


@pytest.mark.parametrize("flag", ["FAKE_CAMERA", "FAKE"])
def test_real_tracker_is_refused_when_the_tracker_is_a_fake(no_hardware, monkeypatch, tmp_path, capsys, flag):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    monkeypatch.setenv(flag, "1")
    scene_mod.reset_tracker()
    f = traj_file(tmp_path)
    rc = replay.main(["--real", "--traj", str(f), "--tracker"])
    captured = capsys.readouterr()
    assert rc == 2, captured
    assert "a fake scene must never feed the real arm" in captured.err and "nothing moved" in captured.err
    assert "{" not in captured.out  # no replay ran
    rc = real_robot.main(["--real", "--carry-test", "--tracker"])
    captured = capsys.readouterr()
    assert rc == 2, captured
    assert "a fake scene must never feed the real arm" in captured.err
    assert "This moves the REAL arm" not in captured.out


def test_real_tracker_is_refused_without_a_frame(no_hardware, monkeypatch, tmp_path, capsys):
    stub = StubTracker(frames=False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stub)
    f = traj_file(tmp_path)
    assert replay.main(["--real", "--traj", str(f), "--tracker"]) == 2
    assert "no frame within 5 s" in capsys.readouterr().err
    assert stub.waits == [5.0] and stub.closed == 1
    assert real_robot.main(["--real", "--home", "--tracker"]) == 2
    assert "no frame within 5 s" in capsys.readouterr().err
    assert stub.closed == 2


def test_real_tracker_without_is_fake_is_refused(no_hardware, monkeypatch, tmp_path, capsys):
    class Unknown:  # no is_fake attribute at all: not provably real, so refused
        def wait_for_frame(self, timeout=None):
            raise AssertionError("an unknown tracker must be refused before any frame is awaited")

        def latest(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(scene_mod, "get_tracker", lambda: Unknown())
    assert replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"]) == 2
    assert "a fake scene must never feed the real arm" in capsys.readouterr().err


@pytest.mark.parametrize("cli", ["helper", "replay", "real_robot"])
def test_real_tracker_on_a_fake_camera_is_refused(no_hardware, monkeypatch, tmp_path, capsys, cli):
    """The real SceneTracker on the FakeCamera (`fake_chain_tracker`, never started here) reports
    `is_fake` False, but its frames are synthetic: refused before any frame is awaited, and released."""
    chain = scene_mod.fake_chain_tracker()
    assert chain.is_fake is False and chain.camera.is_fake is True
    waits: list[float | None] = []
    closed: list[int] = []
    release = chain.close
    monkeypatch.setattr(chain, "wait_for_frame", lambda timeout=None: waits.append(timeout) or True)
    monkeypatch.setattr(chain, "close", lambda: (closed.append(1), release()))
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: chain)
    if cli == "helper":
        with pytest.raises(
            RuntimeError, match="runs on a FakeCamera.*a fake scene must never feed the real arm"
        ):
            replay.tracker_scene(False)
    else:
        if cli == "replay":
            rc = replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"])
        else:
            rc = real_robot.main(["--real", "--carry-test", "--tracker"])
        captured = capsys.readouterr()
        assert rc == 2, captured
        assert (
            "runs on a FakeCamera" in captured.err
            and "a fake scene must never feed the real arm" in captured.err
        )
        assert "nothing moved" in captured.err and "{" not in captured.out
    assert waits == [] and closed == [1]


class CountingTracker(StubTracker):
    """Counts delivered frames like `SceneTracker.frames`. With `deliver` False every publication is a
    missed grab (an unplugged or busy camera returns None and the SceneTracker still publishes), so
    `wait_for_frame` keeps returning True while `frames` never grows."""

    def __init__(self, *, deliver: bool, start: int = 0) -> None:
        super().__init__()
        self.deliver = deliver
        self.frames = start  # an int count here, unlike StubTracker's flag

    def wait_for_frame(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        threading.Event().wait(0.01)  # one publication every 10 ms, whatever a test did to time.sleep
        if self.deliver:
            self.frames += 1
        return True


def test_publications_without_a_delivered_frame_are_refused(no_dotenv, monkeypatch):
    dead = CountingTracker(deliver=False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: dead)
    with pytest.raises(TimeoutError, match="no frame within 0.2 s"):
        replay.tracker_scene(False, timeout=0.2)
    assert dead.waits[0] == 0.2 and len(dead.waits) > 1 and all(0 < w < 0.2 for w in dead.waits[1:])
    assert dead.closed == 1  # the dead camera is released again
    # frames delivered earlier in the process do not count: the count must grow during this wait
    stale = CountingTracker(deliver=False, start=7)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stale)
    with pytest.raises(TimeoutError):
        replay.tracker_scene(False, timeout=0.1)
    assert stale.closed == 1
    live = CountingTracker(deliver=True, start=7)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: live)
    tracker, scene = replay.tracker_scene(False)
    assert tracker is live and scene.__self__ is live and live.waits == [5.0] and live.closed == 0


@pytest.mark.parametrize("cli", ["replay", "real_robot"])
def test_real_tracker_on_a_dead_camera_is_refused(no_hardware, monkeypatch, tmp_path, capsys, cli):
    dead = CountingTracker(deliver=False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: dead)
    monkeypatch.setattr(replay, "TRACKER_FRAME_S", 0.2)  # read at call time: the CLIs pass no timeout
    if cli == "replay":
        rc = replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"])
    else:
        rc = real_robot.main(["--real", "--carry-test", "--tracker"])
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "cannot use --tracker: TimeoutError: the overhead camera delivered no frame within 0.2 s" in err
    assert "nothing moved" in err and dead.closed == 1 and len(dead.waits) > 1


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("camera 1 did not open (check OVERHEAD_CAM/WRIST_CAM and the USB port)"),
        FileNotFoundError("config/homography.npy"),
        KeyError("zones"),
    ],
)
def test_real_tracker_that_cannot_be_built_is_refused(no_hardware, monkeypatch, tmp_path, capsys, error):
    def broken():
        raise error

    monkeypatch.setattr(scene_mod, "get_tracker", broken)
    assert replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"]) == 2
    err = capsys.readouterr().err
    assert f"cannot use --tracker: {type(error).__name__}" in err and "nothing moved" in err
    assert real_robot.main(["--real", "--home", "--tracker"]) == 2
    assert f"cannot use --tracker: {type(error).__name__}" in capsys.readouterr().err


def test_tracker_timeout_must_be_positive(no_dotenv, monkeypatch):
    def no_tracker():
        raise AssertionError("get_tracker() called with a bad timeout")

    monkeypatch.setattr(scene_mod, "get_tracker", no_tracker)
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive number of seconds"):
            replay.tracker_scene(False, timeout=bad)


def test_help_lists_the_new_flags(capsys):
    for main, flags in (
        (replay.main, ("--tracker", "--scene-file", "--record")),
        (real_robot.main, ("--tracker", "--scene-file")),
    ):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        out = capsys.readouterr().out
        assert exc.value.code == 0 and all(flag in out for flag in flags), out


# -- --real --tracker on stubs: the tracker's latest is the hand source and it is closed afterwards ---------


def test_real_tracker_feeds_the_replay_and_is_closed(real_stubs, monkeypatch, tmp_path, capsys):
    stub = StubTracker()
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stub)
    rc = replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert json_line(out)["ok"] and real_stubs[-1].commands
    assert stub.waits == [5.0] and stub.latest_calls > 0 and stub.closed == 1
    assert "WARNING: no --scene-file" not in out
    rc = real_robot.main(["--real", "--home", "--tracker"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert json_line(out)["ok"] and stub.closed == 2


def test_real_tracker_hand_too_close_freezes_through_the_tracker(real_stubs, monkeypatch, tmp_path, capsys):
    stub = StubTracker(hand_seen=True)
    stub.scene["hand"].update({"x": 0.3, "y": -0.08})
    tip = ik.fk_tip(HOME)
    stub.scene["hand"].update({"x": tip[0], "y": tip[1], "z": max(0.0, tip[2] - 0.05)})  # 5 cm under the tip
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: stub)
    rc = replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--tracker"])
    out = capsys.readouterr().out
    assert rc == 1
    assert json_line(out)["reason"] == "hand_too_close"
    assert real_stubs[-1].stopped and real_stubs[-1].torque and stub.closed == 1


# -- exclusivity -------------------------------------------------------------------------------------------


def test_tracker_and_scene_file_are_mutually_exclusive(tmp_path, capsys):
    f = traj_file(tmp_path)
    with pytest.raises(SystemExit) as exc:
        replay.main(["--fake", "--traj", str(f), "--tracker", "--scene-file", str(f)])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        real_robot.main(["--fake", "--home", "--tracker", "--scene-file", str(f)])
    assert exc.value.code == 2


# -- the 10 Hz hand lines ----------------------------------------------------------------------------------


def test_carry_logs_the_moving_hand_at_10_hz():
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    def scene():
        s = copy.deepcopy(FAKE_SCENE)
        s["hand"].update({"x": 0.30, "y": -0.12 + 0.05 * clock["t"], "z": 0.03, "seen": True})  # 5 cm/s +y
        return s

    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=scene, clock=lambda: clock["t"], sleep=sleep, realtime=False
    )
    assert result["ok"], result["events"]
    lines = [HAND_LINE.match(e) for e in result["events"]]
    travel = [m for m in lines if m and m.group(1) == "carry travel"]
    track = [m for m in lines if m and m.group(1) == "carry_to_hand"]
    assert travel and track
    for leg in (travel, track):
        ts = [float(m.group(2)) for m in leg]
        ys = [float(m.group(4)) for m in leg]
        assert ts == sorted(ts) and ys == sorted(ys)  # the leg's clock and the hand's y never go backwards
        assert all(abs(float(m.group(3)) - 0.30) < 1e-9 for m in leg)
        # one line per re-aim cycle: 0.1 s nominally, longer only when the tip speed cap splits a
        # cycle's waypoints over extra ticks, shorter when a holding cycle ticks once
        gaps = [b - a for a, b in zip(ts[:-1], ts[1:], strict=True)]
        assert max(gaps) <= 0.3 + 1e-9 and sum(g <= 0.1 + 1e-9 for g in gaps) >= len(gaps) // 2
    assert track[-1].group(2) and float(track[-1].group(2)) - float(track[0].group(2)) >= 1.0
    speed = (float(track[-1].group(4)) - float(track[0].group(4))) / (
        float(track[-1].group(2)) - float(track[0].group(2))
    )
    assert speed == pytest.approx(0.05, abs=0.01)  # the 5 cm/s hand can be read back from the lines
    distances = [float(m.group(5)) for m in track]
    assert distances[0] > distances[-1] and distances[-1] <= real_robot.RELEASE_WITHIN_M
    assert not any("hand unseen" in e for e in result["events"])


def test_hand_unseen_is_logged_once_and_nothing_without_hand_data(monkeypatch):
    pacer = replay.Pacer(False)
    scene = copy.deepcopy(FAKE_SCENE)
    scene["hand"].update({"x": 0.28, "y": -0.08, "z": 0.03, "seen": True})

    def source():
        scene["hand"]["seen"] = pacer.now() < 0.5
        return scene

    d = FakeDriver(gripper0=0.2)
    session = replay.Session(d, source, pacer, hand_freeze=False)
    session.start()
    monkeypatch.setattr(real_robot, "_servo_path", lambda *a, **kw: (None, "holding"))
    with pytest.raises(replay.Abort, match="error"):  # the held arm never reaches the zone: it aborts
        real_robot._track_hand(session, timeout=1.0)
    unseen = [e for e in session.events if e.endswith("hand unseen")]
    assert len(unseen) == 1 and unseen[0].startswith("carry_to_hand t=0.5 s")
    seen = [e for e in session.events if HAND_LINE.match(e)]
    assert len(seen) >= 5 and all(session.events.index(e) < session.events.index(unseen[0]) for e in seen)
    # no hand data at all: no hand lines
    d = FakeDriver(gripper0=0.2)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=None, realtime=False)
    assert result["reason"] == "error"  # no scene: no clearance; nothing about a hand either way
    assert not any(HAND_LINE.match(e) or e.endswith("hand unseen") for e in result["events"])
