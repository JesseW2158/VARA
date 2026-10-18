"""Scene tooling (lane S scene-file writer, B6 accuracy check): `write_scene_atomic`, `scene_out_path`,
`unseen_scene`, `warm_up`, `python -m vision.scene --out/--print/--accuracy`, `load_layout`,
`accuracy_report` and the fake chain's homography, on the FakeCamera chain and FakeSceneTracker only. No
device, no arm, no .env."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from arm.replay import scene_file_reader
from vision import scene as scene_mod
from vision.calibrate import load_scene_cfg, synthetic_homography
from vision.cameras import FakeCamera, load_fake_scene
from vision.scene import (
    ACCURACY_POSITION_CM,
    ACCURACY_YAW_DEG,
    EXAMPLE_LAYOUT,
    FakeSceneTracker,
    accuracy_report,
    configured_names,
    load_layout,
    new_report_path,
    protected_dir,
    run_accuracy_check,
    scene_out_path,
    unseen_scene,
    warm_up,
    write_scene_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_SCHEMA = Draft202012Validator(json.loads((ROOT / "schemas" / "scene.json").read_text(encoding="utf-8")))


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No device, no .env, FAKE flags cleared, runs/ under tmp_path."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))


@pytest.fixture(scope="module")
def fake_scene():
    return load_fake_scene()


def _layout_from(scene, **overrides):
    items = {}
    for group in ("objects", "obstacles"):
        for name, item in scene[group].items():
            items[name] = {"x": item["x"], "y": item["y"], "yaw": item["yaw"]}
    for name, pose in overrides.items():
        if pose is None:
            items.pop(name, None)
        else:
            items[name] = pose
    return {"items": items}


def _write_layout(path, layout):
    path.write_text(json.dumps(layout), encoding="utf-8")
    return str(path)


def _never(*args, **kwargs):
    raise AssertionError("no tracker may be built here")


def _no_t(scene):
    return {k: v for k, v in scene.items() if k != "t"}


def _seen_flags(scene):
    flags = {name: item["seen"] for group in ("objects", "obstacles") for name, item in scene[group].items()}
    flags["hand"] = scene["hand"]["seen"]
    return flags


# -- the atomic writer ------------------------------------------------------------------------------------


def test_write_scene_atomic_round_trip_uses_os_replace_and_leaves_no_temp(tmp_path, fake_scene, monkeypatch):
    replaced = []
    original = os.replace

    def spy(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        original(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    out = tmp_path / "live" / "scene.json"
    out.parent.mkdir()
    assert write_scene_atomic(fake_scene, out) == out
    assert json.loads(out.read_text(encoding="utf-8")) == fake_scene
    assert replaced and replaced[-1][1] == "scene.json" and replaced[-1][0].startswith(".scene.json.")
    assert [p.name for p in out.parent.iterdir()] == ["scene.json"]  # the temp file is gone
    assert scene_file_reader(str(out))() == fake_scene  # what arm.replay --scene-file reads


def test_write_scene_atomic_rejects_an_invalid_scene_and_keeps_the_file(tmp_path, fake_scene):
    out = tmp_path / "scene.json"
    write_scene_atomic(fake_scene, out)
    with pytest.raises(ValueError, match="scene"):
        write_scene_atomic({"t": 1.0}, out)
    broken = copy.deepcopy(fake_scene)
    broken["objects"]["juice_box"]["x"] = math.nan
    with pytest.raises(ValueError):
        write_scene_atomic(broken, out)
    assert json.loads(out.read_text(encoding="utf-8")) == fake_scene
    assert [p.name for p in tmp_path.iterdir()] == ["scene.json"]


def test_a_reader_never_sees_a_partial_scene_file(tmp_path, fake_scene):
    """Two scenes of very different sizes rewritten as fast as possible while a reader re-reads the file:
    every successful read parses as one of them, complete."""
    small = copy.deepcopy(fake_scene)
    small["t"] = 1.0
    big = copy.deepcopy(fake_scene)
    big["t"] = 2.0
    for i in range(60):
        big["objects"][f"filler_{i}"] = dict(fake_scene["objects"]["juice_box"], marker=11)
    out = tmp_path / "scene.json"
    write_scene_atomic(small, out)
    failures: list[BaseException] = []
    done = threading.Event()

    def writer():
        try:
            for i in range(200):
                write_scene_atomic(big if i % 2 else small, out)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion below
            failures.append(exc)
        finally:
            done.set()

    reads = decode_errors = os_errors = 0
    seen_t = set()
    thread = threading.Thread(target=writer)
    thread.start()
    while not done.is_set():
        try:
            text = out.read_text(encoding="utf-8")
        except OSError:
            os_errors += 1
            continue
        try:
            scene = json.loads(text)
        except json.JSONDecodeError:
            decode_errors += 1
            continue
        reads += 1
        SCENE_SCHEMA.validate(scene)
        seen_t.add(scene["t"])
        assert len(scene["objects"]) in (3, 63), "a complete scene has exactly 3 or 63 objects"
    thread.join(timeout=30)
    assert not failures, failures
    assert decode_errors == 0 and reads > 0
    assert seen_t <= {1.0, 2.0}
    assert [p.name for p in tmp_path.iterdir()] == ["scene.json"]


@pytest.mark.parametrize("folder", ["config", "skills", "schemas", "tests"])
def test_scene_out_path_refuses_protected_dirs(folder):
    with pytest.raises(ValueError, match=f"inside {folder}/"):
        scene_out_path(ROOT / folder / "live-scene.json")
    with pytest.raises(ValueError, match=f"inside {folder}/"):
        scene_out_path(ROOT / folder / "deeper" / "scene.json")


def test_scene_out_path_creates_parents_and_refuses_directories(tmp_path):
    out = scene_out_path(tmp_path / "a" / "b" / "scene.json")
    assert out.parent.is_dir() and out.name == "scene.json" and not out.exists()
    with pytest.raises(ValueError, match="directory"):
        scene_out_path(tmp_path / "a")


def test_scene_out_path_replaces_only_an_empty_file_or_a_scene(tmp_path, fake_scene):
    code = tmp_path / "replay.py"
    code.write_text("print('hello')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a scene file"):
        scene_out_path(code)
    other_json = tmp_path / "notes.json"
    other_json.write_text('{"items": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a scene file"):
        scene_out_path(other_json)
    binary = tmp_path / "image.json"
    binary.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ValueError, match="cannot be read as a scene"):
        scene_out_path(binary)
    assert code.read_text(encoding="utf-8") == "print('hello')\n"  # untouched
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert scene_out_path(empty) == empty.resolve()
    earlier = tmp_path / "scene.json"
    write_scene_atomic(fake_scene, earlier)
    assert scene_out_path(earlier) == earlier.resolve()


def test_protected_dir_names_the_folder_or_none(tmp_path):
    assert protected_dir(ROOT / "config" / "objects.yaml") == "config/"
    assert protected_dir(ROOT / "tests" / "fixtures" / "verify" / "x.json") == "tests/"
    assert protected_dir(ROOT / "runs" / "live" / "scene.json") is None
    assert protected_dir(tmp_path / "skills" / "scene.json") is None  # only the repo's folders


def test_unseen_scene_keeps_poses_and_clears_every_seen_flag(fake_scene):
    scene = copy.deepcopy(fake_scene)
    scene["hand"]["seen"] = True
    scene["t"] = 1.0
    out = unseen_scene(scene)
    SCENE_SCHEMA.validate(out)
    assert set(_seen_flags(out).values()) == {False} and out["t"] > 1.0
    assert out["objects"]["juice_box"]["x"] == scene["objects"]["juice_box"]["x"]
    assert scene["hand"]["seen"] is True  # the input is not modified


# -- python -m vision.scene --out (stream mode) ------------------------------------------------------------


def test_out_cli_writes_a_live_scene_file_the_arm_clis_can_read(tmp_path):
    out = tmp_path / "live" / "scene.json"
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.5"]) == 0
    first = out.stat().st_mtime_ns
    scene = json.loads(out.read_text(encoding="utf-8"))
    SCENE_SCHEMA.validate(scene)
    assert (
        set(scene["objects"]) == {"pill_bottle", "juice_box", "tea_tin"}
        and "cereal_box" in scene["obstacles"]
    )
    assert scene_file_reader(str(out))()["zones"]["tray"]["r"] == pytest.approx(0.07)
    assert set(_seen_flags(scene).values()) == {False}  # the writer stopped: nothing is left marked seen
    time.sleep(0.02)
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.4"]) == 0
    assert out.stat().st_mtime_ns > first  # the file keeps being rewritten in place
    assert [p.name for p in out.parent.iterdir()] == ["scene.json"]


def test_out_file_is_live_while_the_writer_runs(tmp_path):
    """Polled the way arm.replay's scene_file_reader reads it: items seen and the file's mtime advancing
    while the writer runs, every item unseen once it stops."""
    out = tmp_path / "live" / "scene.json"
    codes: list[int] = []
    writer = threading.Thread(
        target=lambda: codes.append(scene_mod.main(["--fake", "--out", str(out), "--seconds", "1.5"]))
    )
    writer.start()
    live = None
    live_mtimes: set[int] = set()
    deadline = time.monotonic() + 20
    while (live is None or len(live_mtimes) < 2) and writer.is_alive() and time.monotonic() < deadline:
        try:
            mtime = out.stat().st_mtime_ns
            scene = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # not created yet, or replaced right now (Windows)
            time.sleep(0.01)
            continue
        SCENE_SCHEMA.validate(scene)
        if scene["objects"].get("juice_box", {}).get("seen"):
            live = live or scene
            live_mtimes.add(mtime)
        time.sleep(0.01)
    writer.join(timeout=30)
    assert codes == [0]
    assert live is not None, "the live file never showed juice_box seen"
    assert len(live_mtimes) >= 2, "the live file's mtime did not advance while the writer ran"
    assert all(live["objects"][name]["seen"] for name in ("pill_bottle", "juice_box", "tea_tin"))
    final = scene_file_reader(str(out))()
    assert set(_seen_flags(final).values()) == {False} and final["t"] >= live["t"]


def test_out_replaces_a_stale_file_before_the_first_frame(tmp_path, fake_scene, monkeypatch):
    """A file left by a writer that was killed (hand seen) is replaced by the tracker's empty scene before
    the tracker starts, so a reader never acts on the old poses."""
    out = tmp_path / "scene.json"
    stale = copy.deepcopy(fake_scene)
    stale["hand"]["seen"] = True
    write_scene_atomic(stale, out)
    at_start: list[dict] = []
    real_chain = scene_mod.fake_chain_tracker

    def chain(hz=scene_mod.RATE_HZ):
        tracker = real_chain(hz)
        start = tracker.start

        def recording_start():
            at_start.append(json.loads(out.read_text(encoding="utf-8")))
            start()

        tracker.start = recording_start
        return tracker

    monkeypatch.setattr(scene_mod, "fake_chain_tracker", chain)
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.3"]) == 0
    (first,) = at_start
    SCENE_SCHEMA.validate(first)
    assert first["objects"] == {} and first["obstacles"] == {} and first["hand"]["seen"] is False


class _Stalling:
    """A tracker whose camera.read() hangs after one frame: the first `wait_for_frame` reports a
    publication, every later one times out with nothing new. Records the live file as a reader sees it at
    each later wait."""

    frames = misses = errors = 0

    def __init__(self, scene, out: Path):
        self.scene = scene
        self.out = out
        self.waits = 0
        self.timeouts: list[float] = []
        self.during: list[dict] = []

    def latest(self):
        return copy.deepcopy(self.scene)

    def start(self):
        return None

    def close(self):
        return None

    def wait_for_frame(self, timeout=None):
        self.waits += 1
        self.timeouts.append(timeout)
        if self.waits == 1:
            return True
        self.during.append(json.loads(self.out.read_text(encoding="utf-8")))
        time.sleep(0.02)
        return False


def test_out_marks_a_stalled_feed_unseen_until_the_next_frame(tmp_path, fake_scene, monkeypatch, capsys):
    """A hung camera.read() publishes nothing, and the readers do not check `t`: after one stalled wait the
    live file holds the last scene with every item unseen (written once per stall, not on every wait)."""
    out = tmp_path / "scene.json"
    seen = copy.deepcopy(fake_scene)
    seen["hand"]["seen"] = True
    stub = _Stalling(seen, out)
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", lambda hz=scene_mod.RATE_HZ: stub)
    writes: list[dict] = []
    real_write = scene_mod.write_scene_atomic

    def counting(scene, path):
        writes.append(copy.deepcopy(dict(scene)))
        return real_write(scene, path)

    monkeypatch.setattr(scene_mod, "write_scene_atomic", counting)
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.3"]) == 0
    assert set(stub.timeouts) == {scene_mod.STALE_FEED_S} and len(stub.during) >= 2
    assert set(_seen_flags(stub.during[0]).values()) == {True}  # the frame before the stall
    for scene in stub.during[1:]:  # every read after the first stalled wait
        SCENE_SCHEMA.validate(scene)
        assert set(_seen_flags(scene).values()) == {False}
        assert scene["objects"]["juice_box"]["x"] == seen["objects"]["juice_box"]["x"]  # poses kept
    flags = [set(_seen_flags(w).values()) for w in writes]
    assert flags == [{True}, {True}, {False}, {False}]  # before start, the frame, the stall, the stop
    assert "no frame within 1 s" in capsys.readouterr().err
    stub.timeouts.clear()
    stub.waits = 0
    assert scene_mod.main(["--fake", "--out", str(out), "--seconds", "0.2", "--hz", "1"]) == 0
    assert set(stub.timeouts) == {3.0}  # three frame periods at a low rate


def test_print_with_out_prints_and_writes(tmp_path, capsys):
    out = tmp_path / "scene.json"
    assert scene_mod.main(["--fake", "--print", "--out", str(out), "--seconds", "0.5"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) >= 2
    for line in lines:
        SCENE_SCHEMA.validate(json.loads(line))
    final = json.loads(out.read_text(encoding="utf-8"))
    assert _no_t(final) == _no_t(unseen_scene(json.loads(lines[-1])))  # the last scene, marked unseen


def test_out_cli_refuses_a_fake_camera_without_fake_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    monkeypatch.setattr(scene_mod, "_physical_tracker", _never)
    out = tmp_path / "scene.json"
    assert scene_mod.main(["--out", str(out), "--seconds", "0.2"]) == 2
    assert not out.exists()
    monkeypatch.delenv("FAKE_CAMERA")
    monkeypatch.setenv("FAKE", "1")
    assert scene_mod.main(["--print", "--out", str(out), "--seconds", "0.2"]) == 2
    assert not out.exists()


def test_out_cli_refuses_protected_paths_before_building_a_tracker(monkeypatch):
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    monkeypatch.setattr(scene_mod, "_physical_tracker", _never)
    target = ROOT / "skills" / "live-scene-test.json"
    assert scene_mod.main(["--fake", "--out", str(target), "--seconds", "0.2"]) == 2
    assert not target.exists()
    assert scene_mod.main(["--fake", "--out", str(ROOT / "config" / "scene-live.json")]) == 2


def test_physical_path_refuses_an_in_process_fake_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(scene_mod, "get_camera", lambda name: FakeCamera())
    with pytest.raises(RuntimeError, match="fake"):
        scene_mod._physical_tracker()
    out = tmp_path / "scene.json"
    assert scene_mod.main(["--out", str(out), "--seconds", "0.2"]) == 2
    assert not out.exists()


@pytest.mark.parametrize("how", ["FAKE_CAMERA", "FAKE", "--fake"])
def test_print_alone_still_runs_the_real_tracker_on_the_fake_camera(monkeypatch, capsys, how):
    """`--print` without `--out` keeps its behaviour: under FAKE_CAMERA/FAKE (no --fake needed) or with
    --fake it prints valid scene lines from the real tracker on the FakeCamera, and writes no file."""
    monkeypatch.setattr(scene_mod, "_physical_tracker", _never)
    argv = ["--print", "--seconds", "0.5"]
    if how.startswith("--"):
        argv.insert(0, how)
    else:
        monkeypatch.setenv(how, "1")
    assert scene_mod.main(argv) == 0
    captured = capsys.readouterr()
    scenes = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert len(scenes) >= 2
    for scene in scenes:
        SCENE_SCHEMA.validate(scene)
    assert set(scenes[-1]["objects"]) == {"pill_bottle", "juice_box", "tea_tin"}
    assert scenes[-1]["objects"]["juice_box"]["seen"] is True
    assert f"printed {len(scenes)} scenes" in captured.err and "wrote" not in captured.err


def test_fake_chain_decodes_with_the_synthetic_homography(monkeypatch, fake_scene):
    """After a physical calibration config/homography.npy describes the real camera; the FakeCamera renders
    through the synthetic camera model, so the fake chain (every --fake mode) must keep the synthetic
    matrix or every marker would be misplaced."""
    physical = np.array([[0.0002, -0.0016, 0.55], [-0.0013, 0.0001, 0.40], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(scene_mod, "load_homography", lambda *args, **kwargs: physical)
    chain = scene_mod.fake_chain_tracker()
    try:
        assert np.allclose(chain.h, synthetic_homography(load_scene_cfg()))
        snap = chain.snapshot(3)
    finally:
        chain.close()
    for group in ("objects", "obstacles"):
        for name, item in fake_scene[group].items():
            got = snap[group][name]
            assert got["seen"] and math.hypot(got["x"] - item["x"], got["y"] - item["y"]) < 0.005, name


def test_cli_without_a_mode_prints_help_and_names_an_example_layout(capsys):
    assert scene_mod.main([]) == 2
    assert "--accuracy" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        scene_mod.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert EXAMPLE_LAYOUT in text and '"yaw": null' in text and "--warmup" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["--print", "--hz", "0"],
        ["--print", "--hz", "nan"],
        ["--print", "--seconds", "0"],
        ["--out", "x.json", "--seconds", "-1"],
        ["--print", "--seconds", "nan"],
        ["--print", "--frames", "5"],
        ["--out", "x.json", "--warmup", "1"],
        ["--accuracy", "x.json", "--frames", "0"],
        ["--accuracy", "x.json", "--print"],
        ["--accuracy", "x.json", "--seconds", "2"],
        ["--accuracy", "x.json", "--hz", "5"],
        ["--accuracy", "x.json", "--warmup", "-1"],
        ["--accuracy", "x.json", "--warmup", "inf"],
        ["--accuracy"],
        ["--frames", "x"],
    ],
)
def test_cli_misuse_exits_2_before_anything_runs(argv, monkeypatch):
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    monkeypatch.setattr(scene_mod, "_physical_tracker", _never)
    with pytest.raises(SystemExit) as exc:
        scene_mod.main(argv)
    assert exc.value.code == 2


# -- the layout file and the accuracy report (B6) ----------------------------------------------------------


def test_load_layout_accepts_the_documented_shape(tmp_path, fake_scene):
    layout = _layout_from(fake_scene, hand={"x": 0.28, "y": -0.08})  # the hand may omit its yaw
    layout["items"]["pill_bottle"]["yaw"] = None  # null = not checked
    layout["note"] = "other top-level keys are ignored"
    loaded = load_layout(_write_layout(tmp_path / "layout.json", layout), known=configured_names())
    assert set(loaded) == {"pill_bottle", "juice_box", "tea_tin", "cereal_box", "hand"}
    assert loaded["pill_bottle"]["yaw"] is None and loaded["hand"]["yaw"] is None
    assert loaded["juice_box"] == {"x": 0.24, "y": -0.04, "yaw": 0.3}
    assert configured_names() == {"pill_bottle", "juice_box", "tea_tin", "cereal_box", "hand"}


def test_load_layout_parses_the_docstring_example(tmp_path):
    """The example printed in the module docstring and --help is a valid layout (its numbers are only an
    illustration)."""
    doc = scene_mod.__doc__
    start = doc.index('{"items"')
    text = doc[start : doc.index("}}}", start) + 3]
    loaded = load_layout(_write_layout(tmp_path / "layout.json", json.loads(text)), known=configured_names())
    assert loaded["pill_bottle"]["yaw"] is None and loaded["cereal_box"]["yaw"] == 0.0


def test_load_layout_refuses_a_name_given_twice(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        '{"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3},'
        ' "juice_box": {"x": 0.30, "y": -0.04, "yaw": 0.3}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'juice_box'"):
        load_layout(path, known=configured_names())


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ({"objects": {}}, "items"),
        ({"items": {}}, "empty"),
        ({"items": {"juice_box": {"x": 24, "y": -4, "yaw": 0.3}}}, "meters"),
        ({"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 30}}}, "radians"),
        ({"items": {"juice_box": {"x": 0.24, "yaw": 0.3}}}, "needs y"),
        ({"items": {"juice_box": {"x": 0.24, "y": -0.04}}}, "needs yaw"),
        ({"items": {"cereal_box": {"x": 0.28, "y": 0.07}}}, "or null to skip"),
        ({"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw_deg": 17}}}, "unknown keys"),
        ({"items": {"juice_box": {"x": True, "y": -0.04}}}, "number"),
        ({"items": {"juice_box": {"x": "0.24", "y": -0.04}}}, "number"),
        ({"items": {"hand": {"x": 0.28, "y": -0.08, "yaw": 0.1}}}, "hand has no yaw"),
        ({"items": {"juice-box": {"x": 0.24, "y": -0.04}}}, "unknown item"),
        ({"items": {"juice_box": [0.24, -0.04]}}, "expected"),
    ],
)
def test_load_layout_fails_loud_on_bad_input(tmp_path, layout, message):
    path = _write_layout(tmp_path / "layout.json", layout)
    with pytest.raises(ValueError, match=message):
        load_layout(path, known=configured_names())


def test_load_layout_rejects_non_json(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_layout(path)
    with pytest.raises(OSError):
        load_layout(tmp_path / "missing.json")


def test_accuracy_report_passes_on_an_exact_match_and_wraps_yaw(fake_scene):
    layout = _layout_from(fake_scene)["items"]
    report = accuracy_report(layout, fake_scene)
    assert report["passed"] is True and report["limits"] == {"position_cm": 1.5, "yaw_deg": 10.0}
    assert all(
        e["ok"] and e["seen"] and e["error_cm"] == 0.0 and e["yaw_error_deg"] == 0.0
        for e in report["items"].values()
    )
    assert (
        report["items"]["juice_box"]["group"] == "objects"
        and report["items"]["cereal_box"]["group"] == "obstacles"
    )
    wrapped = copy.deepcopy(layout)
    wrapped["juice_box"]["yaw"] = 0.3 - 2 * math.pi  # the same direction, written one turn lower
    assert accuracy_report(wrapped, fake_scene)["items"]["juice_box"]["yaw_error_deg"] == 0.0
    scene = copy.deepcopy(fake_scene)
    scene["objects"]["juice_box"]["yaw"] = 3.1
    across = copy.deepcopy(layout)
    across["juice_box"]["yaw"] = -3.1  # 0.083 rad apart across the +-pi seam, not 355 degrees
    entry = accuracy_report(across, scene)["items"]["juice_box"]
    assert entry["yaw_error_deg"] == pytest.approx(math.degrees(2 * math.pi - 6.2), abs=0.01) and entry["ok"]
    off = copy.deepcopy(layout)
    off["juice_box"]["yaw"] = 0.3 + math.radians(ACCURACY_YAW_DEG + 2)
    entry = accuracy_report(off, fake_scene)["items"]["juice_box"]
    assert not entry["ok"] and "yaw off by 12.0 deg" in entry["problems"][0]
    off["juice_box"]["yaw"] = None  # a null yaw is not checked
    entry = accuracy_report(off, fake_scene)["items"]["juice_box"]
    assert entry["ok"] and entry["yaw_error_deg"] is None


def test_accuracy_report_flags_position_unseen_missing_and_the_hand(fake_scene):
    layout = _layout_from(fake_scene)["items"]
    shifted = copy.deepcopy(layout)
    shifted["pill_bottle"]["x"] += (ACCURACY_POSITION_CM + 0.1) / 100.0
    entry = accuracy_report(shifted, fake_scene)["items"]["pill_bottle"]
    assert not entry["ok"] and entry["error_cm"] == 1.6 and "position off by 1.6 cm" in entry["problems"][0]
    shifted["pill_bottle"]["x"] = layout["pill_bottle"]["x"] + 0.014
    assert accuracy_report(shifted, fake_scene)["items"]["pill_bottle"]["ok"]
    scene = copy.deepcopy(fake_scene)
    scene["objects"]["juice_box"]["seen"] = False
    del scene["objects"]["tea_tin"]
    report = accuracy_report(layout, scene)
    assert report["passed"] is False
    assert (
        not report["items"]["juice_box"]["ok"] and "not seen" in report["items"]["juice_box"]["problems"][0]
    )
    assert report["items"]["juice_box"]["error_cm"] == 0.0  # the last known pose is still compared
    tea = report["items"]["tea_tin"]
    assert tea["published"] is None and tea["group"] is None and "never detected" in tea["problems"][0]
    hand_layout = {"hand": {"x": 0.28, "y": -0.08, "yaw": None}}
    unseen = accuracy_report(hand_layout, fake_scene)["items"]["hand"]  # the fake scene's hand: unseen
    assert unseen["published"] is not None and unseen["error_cm"] == 0.0 and not unseen["ok"]
    assert "not seen" in unseen["problems"][0]
    fresh = copy.deepcopy(fake_scene)
    fresh["hand"].update(x=0.0, y=0.0, seen=False)  # the tracker's pose before the wristband was ever seen
    never = accuracy_report(hand_layout, fresh)["items"]["hand"]
    assert never["published"] is None and "never detected" in never["problems"][0]
    seen = copy.deepcopy(fake_scene)
    seen["hand"]["seen"] = True
    entry = accuracy_report(hand_layout, seen)["items"]["hand"]
    assert entry["ok"] and entry["error_cm"] == 0.0 and entry["yaw_error_deg"] is None
    assert accuracy_report({}, fake_scene)["passed"] is False
    with pytest.raises(ValueError):
        accuracy_report(layout, fake_scene, position_cm=0)


class _Running:
    """A started tracker stand-in: `wait_for_frame` delivers a frame every 10 ms until `stop_after` frames,
    then the thread ends (`running` False, no more frames)."""

    def __init__(self, stop_after: int | None = None):
        self.frames = 0
        self.alive = True
        self.stop_after = stop_after

    @property
    def running(self):
        return self.alive

    def wait_for_frame(self, timeout=None):
        if self.stop_after is not None and self.frames >= self.stop_after:
            self.alive = False
            return False
        time.sleep(0.01)
        self.frames += 1
        return True


class _Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"warm_up touched tracker.{name}")


def test_warm_up_runs_a_stopped_scene_tracker_for_the_given_time():
    chain = scene_mod.fake_chain_tracker()
    try:
        t0 = time.monotonic()
        delivered = warm_up(chain, 0.2)
        elapsed = time.monotonic() - t0
    finally:
        chain.close()
    assert delivered > 0 and chain.frames == delivered and 0.2 <= elapsed < 5.0


def test_warm_up_waits_on_a_running_tracker_and_stops_when_its_thread_ends():
    running = _Running()
    t0 = time.monotonic()
    delivered = warm_up(running, 0.15)
    assert delivered == running.frames and delivered > 0 and time.monotonic() - t0 >= 0.15
    dying = _Running(stop_after=3)
    t0 = time.monotonic()
    assert warm_up(dying, 30.0) == 3 and time.monotonic() - t0 < 5.0  # no frame will come: no 30 s wait
    chain = scene_mod.fake_chain_tracker()
    chain.start()
    try:
        assert warm_up(chain, 0.35) > 0  # the real 10 Hz thread
    finally:
        chain.close()


def test_warm_up_returns_at_once_when_there_is_nothing_to_settle():
    t0 = time.monotonic()
    assert warm_up(FakeSceneTracker(), 30.0) == 0  # no camera behind it
    for seconds in (0.0, -1.0, math.nan, math.inf):
        assert warm_up(_Untouchable(), seconds) == 0
    assert time.monotonic() - t0 < 5.0


def test_run_accuracy_check_on_a_fake_scene_tracker(fake_scene):
    layout = _layout_from(fake_scene)["items"]
    report = run_accuracy_check(FakeSceneTracker(), layout, frames=3, fake=True, layout_path="layout.json")
    assert report["passed"] is True and report["mode"] == "synthetic" and report["fake"] is True
    assert report["frames"] == 3 and report["layout"] == "layout.json" and "SYNTHETIC" in report["note"]
    assert report["tracker"]["type"] == "FakeSceneTracker"
    SCENE_SCHEMA.validate(report["snapshot"])
    moved = copy.deepcopy(fake_scene)
    moved["objects"]["juice_box"]["y"] += 0.03
    report = run_accuracy_check(FakeSceneTracker(moved), layout, fake=False)
    assert report["passed"] is False and report["mode"] == "physical" and "note" not in report
    assert [n for n, e in report["items"].items() if not e["ok"]] == ["juice_box"]


# -- python -m vision.scene --accuracy ----------------------------------------------------------------------


def test_accuracy_cli_passes_on_the_fake_chain_and_labels_the_report(tmp_path, fake_scene, capsys):
    layout = _write_layout(tmp_path / "layout.json", _layout_from(fake_scene))
    report_path = tmp_path / "reports" / "accuracy.json"
    assert scene_mod.main(["--fake", "--accuracy", layout, "--out", str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "PASS: 4 of 4 items" in out and "SYNTHETIC" in out and str(report_path) in out
    assert "median of 10 frames after a 1 s warm-up" in out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True and report["fake"] is True and report["mode"] == "synthetic"
    assert report["layout"] == layout and report["frames"] == 10 and "SYNTHETIC" in report["note"]
    assert report["warmup_s"] == 1.0 and report["warmup_frames"] > 0  # the default warm-up ran the camera
    assert set(report["items"]) == {"pill_bottle", "juice_box", "tea_tin", "cereal_box"}
    for entry in report["items"].values():
        # the synthetic rendering of the rotated juice_box marker (0.3 rad) reads ~2.6 deg off (pixel
        # quantization of a 3 cm marker); the B6 limit is 10 deg
        assert entry["seen"] and entry["error_cm"] < 0.5 and entry["yaw_error_deg"] < 5.0
    SCENE_SCHEMA.validate(report["snapshot"])
    assert report["tracker"]["type"] == "SceneTracker"
    assert report["tracker"]["frames"] == report["warmup_frames"] + 10


def test_accuracy_cli_fails_on_a_shifted_item_and_names_it(tmp_path, fake_scene, capsys):
    shifted = _layout_from(fake_scene)
    shifted["items"]["juice_box"]["x"] += 0.03
    layout = _write_layout(tmp_path / "layout.json", shifted)
    report_path = tmp_path / "accuracy.json"
    argv = ["--fake", "--accuracy", layout, "--warmup", "0", "--out", str(report_path)]
    assert scene_mod.main(argv) == 1
    out = capsys.readouterr().out
    assert "FAIL: 1 of 4 items" in out and out.rstrip().splitlines()[-2].endswith("juice_box")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False and not report["items"]["juice_box"]["ok"]
    assert report["items"]["juice_box"]["error_cm"] == pytest.approx(3.0, abs=0.3)


def test_accuracy_cli_fails_on_an_unseen_item(tmp_path, fake_scene):
    layout = _write_layout(
        tmp_path / "layout.json", _layout_from(fake_scene, hand={"x": 0.28, "y": -0.08, "yaw": None})
    )
    report_path = tmp_path / "accuracy.json"
    argv = ["--fake", "--accuracy", layout, "--warmup", "0", "--out", str(report_path)]
    assert scene_mod.main(argv) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["items"]["hand"]["seen"] is False and not report["items"]["hand"]["ok"]
    assert all(entry["ok"] for name, entry in report["items"].items() if name != "hand")
    assert report["warmup_s"] == 0.0 and report["warmup_frames"] == 0


def test_accuracy_cli_fails_on_an_item_missing_from_the_scene(tmp_path, fake_scene, monkeypatch, capsys):
    """A configured item that the camera never saw (its marker off the table) fails by name."""
    scene = copy.deepcopy(fake_scene)
    del scene["objects"]["tea_tin"]
    real_chain = scene_mod.fake_chain_tracker

    def chain(hz=scene_mod.RATE_HZ):
        tracker = real_chain(hz)
        tracker.camera.set_scene(scene)  # the FakeCamera no longer renders tea_tin's marker
        return tracker

    monkeypatch.setattr(scene_mod, "fake_chain_tracker", chain)
    layout = _write_layout(tmp_path / "layout.json", _layout_from(fake_scene))
    report_path = tmp_path / "accuracy.json"
    argv = ["--fake", "--accuracy", layout, "--warmup", "0", "--out", str(report_path)]
    assert scene_mod.main(argv) == 1
    assert capsys.readouterr().out.rstrip().splitlines()[-2].endswith("tea_tin")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tea = report["items"]["tea_tin"]
    assert tea["published"] is None and tea["seen"] is False and "never detected" in tea["problems"][0]
    assert [name for name, entry in report["items"].items() if not entry["ok"]] == ["tea_tin"]


def test_accuracy_cli_default_report_path_is_timestamped_under_runs(tmp_path, fake_scene):
    layout = _write_layout(tmp_path / "layout.json", _layout_from(fake_scene))
    assert scene_mod.main(["--fake", "--accuracy", layout, "--warmup", "0"]) == 0
    assert scene_mod.main(["--fake", "--accuracy", layout, "--warmup", "0"]) == 0
    folder = tmp_path / "runs" / "table-calibration"
    reports = sorted(folder.glob("accuracy-synthetic-*.json"))
    assert len(reports) == 2  # never overwritten, even within the same second
    assert json.loads(reports[0].read_text(encoding="utf-8"))["passed"] is True
    assert new_report_path(folder, "accuracy").name.startswith("accuracy-20")


def test_accuracy_cli_rejects_bad_input_before_building_a_tracker(tmp_path, fake_scene, monkeypatch):
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    monkeypatch.setattr(scene_mod, "_physical_tracker", _never)
    bad = _layout_from(fake_scene)
    bad["items"]["juice-box"] = bad["items"].pop("juice_box")
    assert scene_mod.main(["--fake", "--accuracy", _write_layout(tmp_path / "bad.json", bad)]) == 2
    assert scene_mod.main(["--fake", "--accuracy", str(tmp_path / "missing.json")]) == 2
    good = _write_layout(tmp_path / "good.json", _layout_from(fake_scene))
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    assert scene_mod.main(["--fake", "--accuracy", good, "--out", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "{}"
    monkeypatch.setenv("FAKE_CAMERA", "1")
    assert scene_mod.main(["--accuracy", good]) == 2  # a fake camera needs --fake
    assert not (tmp_path / "runs").exists()


def test_accuracy_cli_refuses_an_in_process_fake_camera_and_reports_a_dead_one(
    tmp_path, fake_scene, monkeypatch
):
    good = _write_layout(tmp_path / "good.json", _layout_from(fake_scene))
    monkeypatch.setattr(scene_mod, "get_camera", lambda name: FakeCamera())
    assert scene_mod.main(["--accuracy", good, "--out", str(tmp_path / "r1.json")]) == 2
    assert not (tmp_path / "r1.json").exists()

    class Dead:
        frames = misses = errors = 0

        def snapshot(self, n=10):
            raise TimeoutError("camera delivered no frame")

        def close(self):
            return None

    monkeypatch.setattr(scene_mod, "fake_chain_tracker", lambda *a, **k: Dead())
    assert scene_mod.main(["--fake", "--accuracy", good, "--out", str(tmp_path / "r2.json")]) == 2
    assert not (tmp_path / "r2.json").exists()


@pytest.mark.parametrize("folder", ["config", "skills", "schemas", "tests"])
def test_accuracy_cli_never_writes_a_report_into_a_protected_folder(
    tmp_path, fake_scene, monkeypatch, folder
):
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    good = _write_layout(tmp_path / "good.json", _layout_from(fake_scene))
    target = ROOT / folder / "accuracy-test-report.json"
    assert scene_mod.main(["--fake", "--accuracy", good, "--out", str(target)]) == 2
    assert not target.exists()


def test_accuracy_cli_refuses_a_directory_as_the_report(tmp_path, fake_scene, monkeypatch, capsys):
    monkeypatch.setattr(scene_mod, "fake_chain_tracker", _never)
    good = _write_layout(tmp_path / "good.json", _layout_from(fake_scene))
    assert scene_mod.main(["--fake", "--accuracy", good, "--out", str(tmp_path)]) == 2
    assert "never overwritten" in capsys.readouterr().err


def test_accuracy_cli_shows_the_verdict_but_exits_2_when_the_report_cannot_be_written(
    tmp_path, fake_scene, monkeypatch, capsys
):
    def refuse(path, data):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(scene_mod, "_write_new_json", refuse)
    good = _write_layout(tmp_path / "good.json", _layout_from(fake_scene))
    report_path = tmp_path / "accuracy.json"
    assert scene_mod.main(["--fake", "--accuracy", good, "--warmup", "0", "--out", str(report_path)]) == 2
    captured = capsys.readouterr()
    assert "cannot write the report" in captured.err and "Permission denied" in captured.err
    assert "PASS: 4 of 4 items" in captured.out and "report: NOT WRITTEN" in captured.out
    assert not report_path.exists()
