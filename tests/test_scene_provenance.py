"""Scene-file provenance (HANDOFF safety lead 4): `python -m vision.scene --out FILE` writes the sidecar
FILE.source.json before its first scene and removes it when it stops; every `--real` consumer of
`--scene-file` reads through `arm.replay.scene_file_reader(..., require_live=True)`, which refuses a file
without a live vision.scene sidecar or without a fresh scene (exit 2 before any prompt or Driver), reads a
stale scene as all unseen, and freezes a running session when the sidecar goes or changes. `--fake`
keeps the old reader. No hardware, no camera device, no .env: the "live" writer is the real SceneTracker
on a FakeCamera stand-in marked `is_fake = False`."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pytest

from arm import driver as driver_mod
from arm import ik, real_robot, replay
from arm.driver import FakeDriver
from vision import scene as scene_mod
from vision.calibrate import load_scene_cfg, synthetic_homography
from vision.cameras import FakeCamera, load_fake_scene
from vision.markers import marker_table

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
HOME = [float(v) for v in ik.home_q()]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))


# -- fixtures and helpers ------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No camera device, no .env, FAKE flags cleared, runs/ and skills/ under tmp_path."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))
    scene_mod.reset_tracker()
    yield
    scene_mod.reset_tracker()
    real_robot.clear_cancel()


@pytest.fixture
def no_hardware(monkeypatch):
    """--real must never get past the scene-file check: a Driver, an Esc watcher or a prompt fails."""

    def no_driver(**kwargs):
        raise AssertionError("Driver constructed although the scene file was refused")

    def no_prompt(*a, **k):
        raise AssertionError("confirmation prompt reached although the scene file was refused")

    monkeypatch.setattr(driver_mod, "Driver", no_driver)
    monkeypatch.setattr(driver_mod, "EscWatcher", no_driver)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", no_prompt)
    monkeypatch.setattr("builtins.input", no_prompt)
    monkeypatch.setattr(replay, "TRACKER_FRAME_S", 0.1)  # a stale file is refused after 0.1 s
    monkeypatch.setattr(replay, "SCENE_POLL_S", 0.02)


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
def real_stubs(monkeypatch):
    """A --real run on stubs: the Driver, the Esc watcher and the confirmation are stand-ins."""
    drivers: list[StubDriver] = []

    def make(**kwargs):
        d = StubDriver(**kwargs)
        drivers.append(d)
        return d

    monkeypatch.setattr(driver_mod, "Driver", make)
    monkeypatch.setattr(driver_mod, "EscWatcher", StubWatcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: True)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    monkeypatch.setattr(replay.time, "sleep", lambda s: None)
    return drivers


class LiveCamera(FakeCamera):
    """FakeCamera frames from a camera that claims to be the real one (`is_fake = False`), with an index."""

    is_fake = False
    index = 1


class LiveStandIn:
    """The attributes `scene_source` reads from a live SceneTracker, nothing else."""

    is_fake = False
    lens_sha256 = "ab" * 32

    def __init__(self) -> None:
        self.camera = LiveCamera()


def live_tracker(hz: float = scene_mod.RATE_HZ) -> scene_mod.SceneTracker:
    """The real SceneTracker on the LiveCamera: what `_physical_tracker` returns on the demo laptop."""
    cfg = load_scene_cfg()
    table = marker_table()
    camera = LiveCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
    return scene_mod.SceneTracker(camera, synthetic_homography(cfg), table, cfg, hz=hz)


def live_source(**overrides) -> dict:
    source = scene_mod.scene_source(LiveStandIn())
    source.update(overrides)
    return source


def write_live(path: Path, scene: dict | None = None, *, t: float | None = None, source: dict | None = None):
    """A scene file plus its sidecar as a live writer leaves them (the sidecar first)."""
    scene = copy.deepcopy(FAKE_SCENE if scene is None else scene)
    scene["t"] = time.time() if t is None else t
    src = live_source() if source is None else source
    scene_mod.write_scene_source(src, path)
    scene_mod.write_scene_atomic(scene, path)
    return src


def traj_file(tmp_path: Path) -> Path:
    near = list(HOME)
    near[0] += 0.5 * DEG
    t = {
        "version": 1,
        "skill": "provenance_test",
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


def hand_at_tip(scene: dict) -> dict:
    out = copy.deepcopy(scene)
    tip = ik.fk_tip(HOME)
    out["hand"].update({"x": tip[0], "y": tip[1], "z": max(0.0, tip[2] - 0.05), "seen": True})
    return out


class Refresher:
    """Keeps a live scene file's `t` fresh from a thread (never time.sleep: real_stubs patches it)."""

    def __init__(self, path: Path, scene: dict) -> None:
        self.path, self.scene = path, copy.deepcopy(scene)
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.halt.is_set():
            self.scene["t"] = time.time()
            try:
                scene_mod.write_scene_atomic(self.scene, self.path)
            except OSError:  # Windows: the reader had the file open past the retries; next round
                pass
            self.halt.wait(0.05)

    def __enter__(self) -> Refresher:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.halt.set()
        self.thread.join(timeout=5)


def _seen_flags(scene: dict) -> set:
    flags = {item["seen"] for group in ("objects", "obstacles") for item in scene[group].values()}
    flags.add(scene["hand"]["seen"])
    return flags


# -- the shared constants ------------------------------------------------------------------------------------


def test_reader_constants_mirror_the_writer():
    assert replay.SCENE_SOURCE_SUFFIX == scene_mod.SOURCE_SUFFIX == ".source.json"
    assert replay.SCENE_SOURCE_WRITER == scene_mod.SOURCE_WRITER == "vision.scene"
    assert replay.SCENE_SEEN_TIMEOUT_S == scene_mod.SEEN_TIMEOUT_S
    path = Path("runs") / "live" / "scene.json"
    assert replay.scene_source_path(path) == scene_mod.source_path(path) == Path(str(path) + ".source.json")


# -- the sidecar ---------------------------------------------------------------------------------------------


def test_scene_source_labels_fake_and_live_trackers():
    fake = scene_mod.scene_source(scene_mod.FakeSceneTracker())
    assert fake["source"] == "fake" and fake["is_fake"] is True
    assert fake["tracker"] == "FakeSceneTracker" and fake["camera"] is None and fake["lens"] is None
    chain = scene_mod.fake_chain_tracker()
    try:
        on_fake_camera = scene_mod.scene_source(chain, hz=5)
    finally:
        chain.close()
    assert on_fake_camera["source"] == "fake" and on_fake_camera["is_fake"] is True  # the tracker is real
    assert on_fake_camera["tracker"] == "SceneTracker" and on_fake_camera["camera_type"] == "FakeCamera"
    assert on_fake_camera["camera"] == "overhead" and on_fake_camera["hz"] == 5.0
    live = scene_mod.scene_source(LiveStandIn())
    assert live["writer"] == "vision.scene" and live["source"] == "live" and live["is_fake"] is False
    assert live["camera"] == 1 and live["camera_type"] == "LiveCamera" and live["lens"] == "ab" * 32
    assert live["pid"] == os.getpid() and datetime.fromisoformat(live["started_at"]).tzinfo is not None
    assert scene_mod.scene_source(object())["source"] == "fake"  # no attributes: never live
    no_camera = type("NoCamera", (), {"is_fake": False})()
    assert scene_mod.scene_source(no_camera)["source"] == "fake"  # a tracker without a camera: never live


def test_write_scene_source_is_atomic_and_refuses_other_writers(tmp_path, monkeypatch):
    replaced = []
    original = os.replace

    def spy(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        original(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    out = tmp_path / "scene.json"
    source = live_source()
    sidecar = scene_mod.write_scene_source(source, out)
    assert sidecar == tmp_path / "scene.json.source.json"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == source
    assert replaced == [(replaced[0][0], "scene.json.source.json")]
    assert replaced[0][0].startswith(".scene.json.source.json.") and replaced[0][0].endswith(".tmp")
    assert [p.name for p in tmp_path.iterdir()] == ["scene.json.source.json"]  # no temp file left
    with pytest.raises(ValueError, match="writer"):
        scene_mod.write_scene_source({"writer": "someone"}, out)
    with pytest.raises(ValueError):
        scene_mod.write_scene_source(dict(source, hz=math.nan), out)
    assert json.loads(sidecar.read_text(encoding="utf-8")) == source  # unchanged after the refusals
    assert [p.name for p in tmp_path.iterdir()] == ["scene.json.source.json"]


def test_remove_scene_source_only_removes_its_own(tmp_path):
    out = tmp_path / "scene.json"
    mine = live_source()
    scene_mod.write_scene_source(mine, out)
    newer = live_source(started_at="2026-10-17T03:00:00.000001-04:00")
    scene_mod.write_scene_source(newer, out)
    assert scene_mod.remove_scene_source(out, mine) is False  # a newer writer's: left in place
    assert scene_mod.source_path(out).exists()
    assert scene_mod.remove_scene_source(out, newer) is True and not scene_mod.source_path(out).exists()
    assert scene_mod.remove_scene_source(out, newer) is True  # already gone


def test_out_path_never_replaces_a_foreign_sidecar(tmp_path):
    out = tmp_path / "scene.json"
    sidecar = scene_mod.source_path(out)
    sidecar.write_text('{"notes": "mine"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not written by vision.scene"):
        scene_mod.scene_out_path(out)
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.2"]) == 2
    assert sidecar.read_text(encoding="utf-8") == '{"notes": "mine"}' and not out.exists()
    sidecar.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a scene sidecar"):
        scene_mod.scene_out_path(out)
    sidecar.unlink()
    sidecar.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        scene_mod.scene_out_path(out)
    sidecar.rmdir()
    scene_mod.write_scene_source(live_source(), out)  # a killed writer's sidecar is replaced
    assert scene_mod.scene_out_path(out) == out.resolve()
    sidecar.write_text("", encoding="utf-8")
    assert scene_mod.scene_out_path(out) == out.resolve()


# -- the writer CLI ------------------------------------------------------------------------------------------


def _record_writes(monkeypatch) -> list[tuple[str, dict]]:
    """Every sidecar and scene write of the stream CLI, in order."""
    writes: list[tuple[str, dict]] = []
    real_source, real_scene = scene_mod.write_scene_source, scene_mod.write_scene_atomic

    def source(src, path):
        writes.append(("source", dict(src)))
        return real_source(src, path)

    def scene(sc, path):
        writes.append(("scene", copy.deepcopy(dict(sc))))
        return real_scene(sc, path)

    monkeypatch.setattr(scene_mod, "write_scene_source", source)
    monkeypatch.setattr(scene_mod, "write_scene_atomic", scene)
    return writes


def _capture_at_start(monkeypatch, name: str, factory, out: Path, seen: list) -> None:
    """Replace `scene_mod.<name>` so that the tracker's start() records the sidecar and what a --real
    reader says about the file at that moment (after the sidecar and the first scene, before any frame)."""

    def make(hz=scene_mod.RATE_HZ):
        tracker = factory(hz)
        start = tracker.start

        def recording_start():
            sidecar = json.loads(scene_mod.source_path(out).read_text(encoding="utf-8"))
            try:
                replay.check_scene_source(out)
                verdict = "accepted"
            except replay.SceneFileRefused as exc:
                verdict = str(exc)
            seen.append((sidecar, verdict))
            start()

        tracker.start = recording_start
        return tracker

    monkeypatch.setattr(scene_mod, name, make)


def test_fake_writer_labels_its_file_fake_first_and_withdraws_it(tmp_path, monkeypatch, capsys):
    out = tmp_path / "live" / "scene.json"
    writes = _record_writes(monkeypatch)
    at_start: list = []
    _capture_at_start(monkeypatch, "fake_chain_tracker", scene_mod.fake_chain_tracker, out, at_start)
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.3"]) == 0
    assert writes[0][0] == "source" and writes[1][0] == "scene"  # the label before the first scene
    assert [kind for kind, _ in writes].count("source") == 1
    ((sidecar, verdict),) = at_start
    assert sidecar == writes[0][1]
    assert sidecar["source"] == "fake" and sidecar["is_fake"] is True
    assert sidecar["camera_type"] == "FakeCamera"
    assert sidecar["tracker"] == "SceneTracker" and sidecar["pid"] == os.getpid() and sidecar["hz"] == 10.0
    assert "says the scene is fake" in verdict and "never feeds the real arm" in verdict
    assert not scene_mod.source_path(out).exists()  # withdrawn when the writer stopped
    assert [p.name for p in out.parent.iterdir()] == ["scene.json"]
    err = capsys.readouterr().err
    assert "SYNTHETIC scene file" in err and "says fake" in err


def test_live_writer_labels_its_file_live_and_a_restart_is_a_new_identity(tmp_path, monkeypatch):
    out = tmp_path / "scene.json"
    at_start: list = []
    _capture_at_start(monkeypatch, "_physical_tracker", live_tracker, out, at_start)
    assert scene_mod.main(["--out", str(out), "--seconds", "0.3"]) == 0
    assert scene_mod.main(["--out", str(out), "--seconds", "0.3", "--hz", "5"]) == 0
    (first, verdict1), (second, verdict2) = at_start
    assert verdict1 == verdict2 == "accepted"
    for sidecar in (first, second):
        assert sidecar["writer"] == "vision.scene" and sidecar["source"] == "live"
        assert sidecar["is_fake"] is False and sidecar["tracker"] == "SceneTracker"
        assert sidecar["camera"] == 1 and sidecar["camera_type"] == "LiveCamera" and sidecar["lens"] is None
        assert sidecar["pid"] == os.getpid()
    assert first["started_at"] != second["started_at"] and second["hz"] == 5.0  # same FILE, same pid
    assert not scene_mod.source_path(out).exists()


def test_a_real_reader_follows_a_live_writer_and_freezes_when_it_stops(tmp_path, monkeypatch):
    out = tmp_path / "scene.json"
    monkeypatch.setattr(scene_mod, "_physical_tracker", live_tracker)
    codes: list[int] = []
    argv = ["--out", str(out), "--seconds", "2"]
    writer = threading.Thread(target=lambda: codes.append(scene_mod.main(argv)))
    writer.start()
    try:
        deadline = time.monotonic() + 20
        reader = None
        while reader is None and writer.is_alive() and time.monotonic() < deadline:
            try:
                reader = replay.scene_file_reader(str(out), require_live=True, timeout=1.0)
            except (OSError, ValueError):  # not created yet, or no frame published yet
                time.sleep(0.02)
        assert reader is not None, "the live file was never accepted"
        assert reader.source["source"] == "live" and reader.source["pid"] == os.getpid()
        seen = None
        while seen is None and writer.is_alive() and time.monotonic() < deadline:
            scene = reader()
            if scene["objects"].get("juice_box", {}).get("seen"):
                seen = scene
            time.sleep(0.02)
        assert seen is not None, "the --real reader never saw the live juice box"
    finally:
        writer.join(timeout=30)
    assert codes == [0]
    with pytest.raises(RuntimeError, match="is gone"):
        reader()  # the writer withdrew its sidecar: a running session freezes
    with pytest.raises(replay.SceneFileRefused, match="no provenance sidecar"):
        replay.scene_file_reader(str(out), require_live=True, timeout=0.1)  # and a new one refuses


# -- the --real reader: refusals -----------------------------------------------------------------------------


def _refusal_case(case: str, out: Path) -> str:
    """Leave `out` (a fresh scene file) with the sidecar of `case`; return the expected refusal text."""
    write_live(out)
    sidecar = scene_mod.source_path(out)
    if case == "missing":
        sidecar.unlink()
        return "no provenance sidecar"
    if case == "unreadable":
        sidecar.write_bytes(b"\xff\xfe\x00 not text")
        return "cannot be read"
    if case == "not json":
        sidecar.write_text("{", encoding="utf-8")
        return "is not JSON"
    if case == "not an object":
        sidecar.write_text("[1, 2]", encoding="utf-8")
        return "not a JSON object"
    if case == "other writer":
        sidecar.write_text(json.dumps(dict(live_source(), writer="my_script")), encoding="utf-8")
        return "was not written by vision.scene"
    if case == "malformed":
        sidecar.write_text(json.dumps(dict(live_source(), pid=True)), encoding="utf-8")
        return "no valid 'pid'"
    if case == "fake":
        scene_mod.write_scene_source(scene_mod.scene_source(scene_mod.FakeSceneTracker()), out)
        return "says the scene is fake"
    if case == "live but is_fake":
        sidecar.write_text(json.dumps(dict(live_source(), is_fake=True)), encoding="utf-8")
        return "never feeds the real arm"
    if case == "stale":
        write_live(out, t=time.time() - 60.0)  # a killed writer's leftover: a live label, an old scene
        return "is not being refreshed"
    raise AssertionError(case)


REFUSALS = [
    "missing",
    "unreadable",
    "not json",
    "not an object",
    "other writer",
    "malformed",
    "fake",
    "live but is_fake",
    "stale",
]


@pytest.mark.parametrize("case", REFUSALS)
def test_require_live_refuses(case, tmp_path, monkeypatch):
    out = tmp_path / "scene.json"
    expected = _refusal_case(case, out)
    with pytest.raises(replay.SceneFileRefused, match=expected):
        replay.scene_file_reader(str(out), require_live=True, timeout=0.1, sleep=lambda s: None)
    assert replay.scene_file_reader(str(out))() is not None  # --fake: no sidecar needed, read as before


@pytest.mark.parametrize("case", REFUSALS)
def test_real_clis_refuse_before_any_prompt_or_driver(case, no_hardware, tmp_path, capsys):
    out = tmp_path / "scene.json"
    expected = _refusal_case(case, out)
    f = traj_file(tmp_path)
    assert replay.main(["--real", "--traj", str(f), "--scene-file", str(out)]) == 2
    captured = capsys.readouterr()
    assert expected in captured.err and "nothing moved" in captured.err, captured.err
    assert "refusing --scene-file" in captured.err and "{" not in captured.out
    for argv in (["--real", "--home"], ["--real", "--carry-test"]):
        assert real_robot.main([*argv, "--scene-file", str(out)]) == 2
        captured = capsys.readouterr()
        assert expected in captured.err and "nothing moved" in captured.err, captured.err
        assert "This moves the REAL arm" not in captured.out


def test_real_clis_refuse_a_file_left_by_a_stopped_writer(no_hardware, tmp_path, capsys):
    out = tmp_path / "scene.json"
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.2"]) == 0
    assert replay.main(["--real", "--traj", str(traj_file(tmp_path)), "--scene-file", str(out)]) == 2
    assert "no provenance sidecar" in capsys.readouterr().err
    assert real_robot.main(["--real", "--home", "--scene-file", str(out)]) == 2
    assert "no provenance sidecar" in capsys.readouterr().err


def test_fresh_scene_is_waited_for_then_accepted(tmp_path):
    out = tmp_path / "scene.json"
    scene = copy.deepcopy(FAKE_SCENE)
    write_live(out, scene, t=100.0)
    now = [200.0]
    sleeps: list[float] = []

    def sleep(seconds):  # the writer publishes during the second poll
        sleeps.append(seconds)
        if len(sleeps) == 2:
            same = json.loads(scene_mod.source_path(out).read_text(encoding="utf-8"))
            write_live(out, scene, t=now[0] - 0.05, source=same)

    reader = replay.scene_file_reader(
        str(out), require_live=True, timeout=1.0, clock=lambda: now[0], sleep=sleep
    )
    assert sleeps == [replay.SCENE_POLL_S] * 2
    assert reader()["t"] == pytest.approx(199.95)
    with pytest.raises(ValueError, match="positive"):
        replay.scene_file_reader(str(out), require_live=True, timeout=0)


# -- the --real reader during a run --------------------------------------------------------------------------


def test_a_stale_scene_reads_as_all_unseen(tmp_path):
    out = tmp_path / "scene.json"
    scene = copy.deepcopy(FAKE_SCENE)
    scene["hand"]["seen"] = True
    write_live(out, scene, t=1000.0)
    now = [1000.1]
    reader = replay.scene_file_reader(str(out), require_live=True, clock=lambda: now[0])
    stale_after = replay.SCENE_SEEN_TIMEOUT_S + 1 / 10.0
    assert True in _seen_flags(reader())  # fresh: as written
    now[0] = 1000.0 + stale_after + 0.01
    stale = reader()
    assert _seen_flags(stale) == {False} and stale["t"] == 1000.0  # unseen, poses and t kept
    assert stale["hand"]["x"] == scene["hand"]["x"]
    assert json.loads(out.read_text(encoding="utf-8"))["hand"]["seen"] is True  # the file is not touched
    write_live(out, scene, t=now[0], source=reader.source)  # the writer publishes again
    assert reader()["hand"]["seen"] is True
    out.write_text("{", encoding="utf-8")  # a broken rewrite: the last good scene is kept ...
    assert reader()["hand"]["seen"] is True
    now[0] += stale_after + 0.01  # ... until it is stale too
    assert _seen_flags(reader()) == {False}
    now[0] = 1000.0 - 5.0  # a scene from the future (a clock jump) is not trusted either
    write_live(out, scene, t=1000.0, source=reader.source)
    assert reader()["hand"]["seen"] is False


def test_a_stale_scene_no_longer_drives_the_hand_check(tmp_path):
    """The hand sits at the tip: a fresh scene freezes the replay (hand_too_close); the same scene once it
    is older than the seen timeout reads unseen, exactly as the live tracker would report it."""
    out = tmp_path / "scene.json"
    traj = json.loads(traj_file(tmp_path).read_text(encoding="utf-8"))
    write_live(out, hand_at_tip(FAKE_SCENE), t=500.0)
    now = [500.05]
    fresh = replay.scene_file_reader(str(out), require_live=True, clock=lambda: now[0])
    result = replay.replay(traj, driver=FakeDriver(HOME), scene=fresh, realtime=False)
    assert result["reason"] == "hand_too_close"
    stale_reader = replay.scene_file_reader(str(out), require_live=True, clock=lambda: now[0])
    now[0] = 510.0
    result = replay.replay(traj, driver=FakeDriver(HOME), scene=stale_reader, realtime=False)
    assert result["ok"], result
    legacy = replay.scene_file_reader(str(out))  # --fake keeps the old behaviour: t is not looked at
    result = replay.replay(traj, driver=FakeDriver(HOME), scene=legacy, realtime=False)
    assert result["reason"] == "hand_too_close"


@pytest.mark.parametrize("change", ["removed", "restarted", "restarted same pid", "fake writer", "garbled"])
def test_a_sidecar_change_freezes_the_session(change, tmp_path):
    out = tmp_path / "scene.json"
    source = write_live(out)
    reader = replay.scene_file_reader(str(out), require_live=True)
    assert reader()["t"] > 0
    sidecar = scene_mod.source_path(out)
    if change == "removed":
        sidecar.unlink()
    elif change == "restarted":
        scene_mod.write_scene_source(live_source(pid=source["pid"] + 1), out)
    elif change == "restarted same pid":
        scene_mod.write_scene_source(live_source(started_at="2026-10-17T05:00:00.000001-04:00"), out)
    elif change == "fake writer":
        scene_mod.write_scene_source(scene_mod.scene_source(scene_mod.FakeSceneTracker()), out)
    else:
        sidecar.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        reader()
    traj = json.loads(traj_file(tmp_path).read_text(encoding="utf-8"))
    drv = FakeDriver(HOME)
    result = replay.replay(traj, driver=drv, scene=reader, realtime=False)
    assert result["reason"] == "error" and drv.stopped  # frozen with torque on, nothing more commanded
    assert any("RuntimeError" in e for e in result["events"])


def test_an_identical_rewrite_of_the_sidecar_is_not_a_new_writer(tmp_path):
    out = tmp_path / "scene.json"
    source = write_live(out)
    reader = replay.scene_file_reader(str(out), require_live=True)
    time.sleep(0.02)
    scene_mod.write_scene_source(source, out)  # same content, new file
    assert reader()["t"] > 0 and reader()["t"] > 0


# -- the CLIs: accepted under --real, unchanged under --fake -------------------------------------------------


def test_real_clis_accept_a_live_file_and_feed_its_hand(real_stubs, tmp_path, capsys):
    out = tmp_path / "scene.json"
    write_live(out)
    f = traj_file(tmp_path)
    with Refresher(out, FAKE_SCENE):
        rc = replay.main(["--real", "--traj", str(f), "--scene-file", str(out)])
        captured = capsys.readouterr()
        assert rc == 0, captured
        assert json_line(captured.out)["ok"] and real_stubs[-1].commands
        assert f"scene file {out}: live, from LiveStandIn on camera 1" in captured.out
        assert "WARNING: no --scene-file" not in captured.out
        rc = real_robot.main(["--real", "--home", "--scene-file", str(out)])
        captured = capsys.readouterr()
        assert rc == 0, captured
        assert "live, from LiveStandIn" in captured.out
    with Refresher(out, hand_at_tip(FAKE_SCENE)):
        rc = replay.main(["--real", "--traj", str(f), "--scene-file", str(out)])
        captured = capsys.readouterr()
    assert rc == 1 and json_line(captured.out)["reason"] == "hand_too_close"
    assert real_stubs[-1].stopped and real_stubs[-1].torque


def test_fake_clis_read_a_plain_scene_file_as_before(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver_mod, "Driver", lambda **k: pytest.fail("no Driver under --fake"))
    plain = tmp_path / "scene.json"
    scene = copy.deepcopy(FAKE_SCENE)
    scene["t"] = 0.0  # ancient, no sidecar
    plain.write_text(json.dumps(scene), encoding="utf-8")
    assert replay.scene_file_reader(str(plain))() == scene  # nothing marked unseen, t not looked at
    assert replay.main(["--fake", "--traj", str(traj_file(tmp_path)), "--scene-file", str(plain)]) == 0
    assert real_robot.main(["--fake", "--home", "--scene-file", str(plain)]) == 0
    out = capsys.readouterr().out
    assert "live, from" not in out
