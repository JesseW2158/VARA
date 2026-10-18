"""Vision stack (B1-B8, I2): markers, calibration, FakeCamera, SceneTracker, CLIs and scripts. No hardware:
cv2.VideoCapture is stubbed to fail loudly wherever a real device could be opened."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from vision import cameras as cameras_mod
from vision import scene as scene_mod
from vision.calibrate import (
    FOCAL_PX,
    calibration_error,
    camera_pose,
    fit_homography,
    height_correct,
    load_homography,
    load_scene_cfg,
    pixel_to_table,
    project,
    synthetic_detections,
    synthetic_homography,
)
from vision.cameras import Camera, FakeCamera, camera_settings, get_camera, load_fake_scene
from vision.markers import CALIB_IDS, HAND_ID, MARKER_SIDE_M, detect, generate, marker_side, marker_table
from vision.scene import (
    FakeSceneTracker,
    SceneTracker,
    circular_median,
    get_tracker,
    reset_tracker,
    validate_scene,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_SCHEMA = Draft202012Validator(json.loads((ROOT / "schemas" / "scene.json").read_text(encoding="utf-8")))
POS_TOL = 0.010  # m
YAW_TOL = math.radians(5.0)


@pytest.fixture(scope="module")
def cfg():
    return load_scene_cfg()


@pytest.fixture(scope="module")
def table():
    return marker_table()


@pytest.fixture(scope="module")
def fake_scene():
    return load_fake_scene()


@pytest.fixture(scope="module")
def homography(cfg):
    return synthetic_homography(cfg)


@pytest.fixture
def no_devices(monkeypatch):
    """Any attempt to open a camera device fails the test."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)


@pytest.fixture
def tracker(fake_scene, cfg, table, homography):
    cam = FakeCamera(fake_scene, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    yield tr
    tr.close()


def _pose_errors(scene, reference):
    for group in ("objects", "obstacles"):
        for name, item in reference[group].items():
            got = scene[group][name]
            yield (
                name,
                math.hypot(got["x"] - item["x"], got["y"] - item["y"]),
                abs(math.remainder(got["yaw"] - item["yaw"], 2 * math.pi)),
                got["seen"],
            )


# -- markers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("marker_id", [0, 7, 10, 11, 12, 30, 40])
def test_generate_detect_round_trip(marker_id):
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[90:210, 90:210] = generate(marker_id, 120)
    dets = detect(canvas)
    assert [d[0] for d in dets] == [marker_id]
    corners = dets[0][1]
    assert corners.shape == (4, 2) and corners.dtype == np.float32
    assert np.allclose(corners[0], (90, 90), atol=1.0)  # corner 0 is the printed top-left
    assert np.allclose(corners[1], (210, 90), atol=1.0)
    # rotated by 35 degrees: still found, corner 0 follows the printed top-left
    rot = cv2.getRotationMatrix2D((150, 150), 35, 1.0)
    turned = cv2.warpAffine(canvas, rot, (300, 300), borderValue=255)
    dets = detect(turned)
    assert [d[0] for d in dets] == [marker_id]
    expected0 = rot @ np.array([90.0, 90.0, 1.0])
    assert np.allclose(dets[0][1][0], expected0, atol=1.5)


def test_mirrored_marker_is_not_detected():
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[90:210, 90:210] = generate(11, 120)
    assert detect(cv2.flip(canvas, 1)) == []


def test_marker_table_and_sides(table):
    assert set(CALIB_IDS) <= set(table)
    assert table[0]["kind"] == "calib" and table[0]["xy"] == [0.08, 0.34] and table[0]["side"] == 0.06
    assert table[10]["name"] == "pill_bottle" and table[10]["kind"] == "object"
    assert table[30]["name"] == "cereal_box" and table[30]["kind"] == "obstacle"
    assert table[HAND_ID]["kind"] == "hand" and table[HAND_ID]["h"] == 0.03
    assert marker_side(0, table) == MARKER_SIDE_M["calib"] == 0.06
    assert (
        marker_side(11, table) == 0.03 and marker_side(30, table) == 0.04 and marker_side(40, table) == 0.04
    )


def test_generate_rejects_bad_ids():
    with pytest.raises(ValueError):
        generate(50, 120)
    with pytest.raises(ValueError):
        generate(3, 10)


def test_detect_handles_empty_and_blank_frames():
    assert detect(None) == []
    assert detect(np.full((480, 640, 3), 200, np.uint8)) == []


# -- calibration ----------------------------------------------------------------------------------------


def test_synthetic_homography_round_trip(cfg, homography):
    c, hc = camera_pose(cfg)
    points = [tuple(v) for v in cfg["calib_markers"].values()] + [
        (0.1, -0.25),
        (0.4, 0.25),
        (0.25, 0.0),
        (0.33, 0.21),
    ]
    pixels = project([(x, y, 0.0) for x, y in points], c, hc)
    assert pixels.shape == (len(points), 2)
    for (x, y), (u, v) in zip(points, pixels, strict=True):
        assert 0 <= u < 640 and 0 <= v < 480, (x, y, u, v)
        bx, by = pixel_to_table(homography, u, v)
        assert math.hypot(bx - x, by - y) < 0.001
    refit = fit_homography(pixels, points)
    assert np.allclose(refit, homography, atol=1e-6)


def test_projection_is_right_handed_and_documented(cfg):
    c, hc = camera_pose(cfg)
    (u0, v0), (uy, vy), (ux, vx) = project([(0.2, 0.0, 0.0), (0.2, 0.1, 0.0), (0.3, 0.0, 0.0)], c, hc)
    assert uy < u0 and vy == pytest.approx(v0)  # +y goes left in the image
    assert vx < v0 and ux == pytest.approx(u0)  # +x goes up in the image
    assert u0 - uy == pytest.approx(FOCAL_PX * 0.1 / hc)


def test_height_correct_recovers_object_center(cfg, homography, fake_scene):
    c, hc = camera_pose(cfg)
    for group in ("objects", "obstacles"):
        for item in fake_scene[group].values():
            (u, v) = project([(item["x"], item["y"], item["h"])], c, hc)[0]
            table_pt = pixel_to_table(homography, u, v)
            raw_err = math.hypot(table_pt[0] - item["x"], table_pt[1] - item["y"])
            x, y = height_correct(table_pt, c, item["h"], hc)
            assert raw_err > 0.005  # the parallax is real for every item
            assert math.hypot(x - item["x"], y - item["y"]) < 1e-6
    with pytest.raises(ValueError):
        height_correct((0.2, 0.0), c, hc, hc)


def test_calibration_error_on_synthetic_data_is_zero(cfg, homography):
    markers = {int(k): tuple(v) for k, v in cfg["calib_markers"].items()}
    assert calibration_error(homography, synthetic_detections(cfg), markers) < 0.01
    shifted = homography.copy()
    shifted[0, 2] += 0.02  # 2 cm offset in x
    assert calibration_error(shifted, synthetic_detections(cfg), markers) == pytest.approx(2.0, abs=0.05)
    with pytest.raises(ValueError):
        calibration_error(homography, [(11, synthetic_detections(cfg)[0][1])], markers)


def test_fit_homography_needs_four_points():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1)])


def test_committed_homography_matches_synthetic(homography):
    assert np.allclose(load_homography(), homography, atol=1e-9)


# -- FakeCamera -----------------------------------------------------------------------------------------


def test_fake_camera_frame_detects_every_marker(fake_scene, cfg, table):
    cam = FakeCamera(fake_scene, scene_cfg=cfg, objects=table)
    frame = cam.read()
    assert frame is not None and frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    ids = {d[0] for d in detect(frame)}
    items = {it["marker"] for g in ("objects", "obstacles") for it in fake_scene[g].values()}
    assert items <= ids and HAND_ID not in ids  # every item marker; the hand is unseen in the fake scene
    # a raised top face covers a table marker in the image as it does for the real camera (the 20 cm
    # cereal box hides calib 6 at the scene.yaml camera pose); every uncovered table marker is read
    hidden = cam.hidden_calibration_ids()
    assert set(CALIB_IDS) - hidden <= ids <= items | set(CALIB_IDS)
    cam.set_scene({**copy.deepcopy(fake_scene), "objects": {}, "obstacles": {}})
    assert cam.hidden_calibration_ids() == set()
    assert {d[0] for d in detect(cam.read())} == set(CALIB_IDS)
    seen_hand = copy.deepcopy(fake_scene)
    seen_hand["hand"]["seen"] = True
    cam.set_scene(seen_hand)
    assert HAND_ID in {d[0] for d in detect(cam.read())}
    assert cam.frames == 3 and cam.failures == 0
    cam.release()
    assert cam.read() is None and cam.failures == 1


def test_fake_camera_wrist_is_a_placeholder():
    frame = FakeCamera(name="wrist").read()
    assert frame.shape == (480, 640, 3) and detect(frame) == []


# -- SceneTracker ---------------------------------------------------------------------------------------


def test_tracker_matches_fake_scene(tracker, fake_scene):
    assert tracker.latest()["objects"] == {}  # nothing published before the first frame
    for _ in range(6):
        tracker.step()
    scene = tracker.latest()
    validate_scene(scene)
    SCENE_SCHEMA.validate(scene)
    assert abs(scene["t"] - time.time()) < 5.0
    assert set(scene["objects"]) == set(fake_scene["objects"])
    assert set(scene["obstacles"]) == set(fake_scene["obstacles"])
    for name, pos_err, yaw_err, seen in _pose_errors(scene, fake_scene):
        assert seen, name
        assert pos_err < POS_TOL, (name, pos_err)
        assert yaw_err < YAW_TOL, (name, yaw_err)
    for group in ("objects", "obstacles"):
        for name, item in scene[group].items():
            assert item["marker"] == fake_scene[group][name]["marker"]
            assert (
                item["h"] == fake_scene[group][name]["h"] and item["size"] == fake_scene[group][name]["size"]
            )
    assert scene["hand"]["seen"] is False and scene["hand"]["z"] == 0.03
    assert scene["zones"] == fake_scene["zones"]


def test_tracker_hand_seen_when_the_wristband_shows(tracker, fake_scene):
    seen_hand = copy.deepcopy(fake_scene)
    seen_hand["hand"]["seen"] = True
    tracker.camera.set_scene(seen_hand)
    tracker.step()
    hand = tracker.latest()["hand"]
    assert hand["seen"] is True and hand["marker"] == HAND_ID
    assert math.hypot(hand["x"] - fake_scene["hand"]["x"], hand["y"] - fake_scene["hand"]["y"]) < POS_TOL


def test_seen_turns_false_after_half_a_second_and_keeps_the_pose(tracker, fake_scene):
    t0 = 1000.0
    tracker.step(now_mono=t0)
    before = tracker.latest()["objects"]["tea_tin"]
    assert before["seen"] is True
    gone = copy.deepcopy(fake_scene)
    del gone["objects"]["tea_tin"]
    tracker.camera.set_scene(gone)
    tracker.step(now_mono=t0 + 0.3)
    mid = tracker.latest()["objects"]["tea_tin"]
    assert mid["seen"] is True and (mid["x"], mid["y"]) == (before["x"], before["y"])
    tracker.step(now_mono=t0 + 0.6)
    after = tracker.latest()["objects"]["tea_tin"]
    assert after["seen"] is False and (after["x"], after["y"]) == (before["x"], before["y"])
    validate_scene(tracker.latest())
    # back in view: seen again, with a fresh median (not mixed with the stale readings)
    tracker.camera.set_scene(fake_scene)
    tracker.step(now_mono=t0 + 5.0)
    assert tracker.latest()["objects"]["tea_tin"]["seen"] is True


def test_items_never_detected_are_left_out(fake_scene, cfg, table, homography):
    partial = copy.deepcopy(fake_scene)
    del partial["objects"]["juice_box"]
    cam = FakeCamera(partial, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    tr.step()
    scene = tr.latest()
    assert "juice_box" not in scene["objects"] and {"pill_bottle", "tea_tin"} == set(scene["objects"])
    validate_scene(scene)


def test_missed_frames_only_advance_the_timeouts(tracker):
    tracker.step(now_mono=10.0)
    tracker.process_frame(None, now_mono=10.7)
    scene = tracker.latest()
    assert scene["objects"] and all(not it["seen"] for it in scene["objects"].values())


def test_tracker_thread_runs_near_10hz_and_snapshot_waits(tracker):
    tracker.start()
    assert tracker.running
    t0 = time.monotonic()
    snap = tracker.snapshot(5)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.35, elapsed
    validate_scene(snap)
    assert all(it["seen"] for it in snap["objects"].values())
    assert tracker.wait_for_frame(timeout=1.0)
    time.sleep(0.6)
    tracker.stop()
    assert not tracker.running
    assert 6 <= tracker.frames <= 20, tracker.frames
    assert tracker.errors == 0


class _SequenceCamera(FakeCamera):
    """A FakeCamera that shows the next scene of a list on every read()."""

    def __init__(self, scenes, **kwargs):
        super().__init__(scenes[0], **kwargs)
        self._scenes = list(scenes)
        self._i = 0

    def read(self):
        self.set_scene(self._scenes[self._i % len(self._scenes)])
        self._i += 1
        return super().read()


def test_snapshot_is_a_median_over_frames(fake_scene, cfg, table, homography):
    displaced = copy.deepcopy(fake_scene)
    displaced["objects"]["juice_box"]["x"] += 0.05
    sequence = [fake_scene] * 7 + [displaced] * 3
    cam = _SequenceCamera(sequence, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    snap = tr.snapshot(10)
    assert cam.frames == 10
    jb = snap["objects"]["juice_box"]
    assert jb["seen"] is True
    assert abs(jb["x"] - fake_scene["objects"]["juice_box"]["x"]) < POS_TOL  # the 3 outliers lose
    validate_scene(snap)


def test_snapshot_seen_needs_half_the_frames(fake_scene, cfg, table, homography):
    gone = copy.deepcopy(fake_scene)
    del gone["objects"]["tea_tin"]
    cam = _SequenceCamera([fake_scene] * 3 + [gone] * 7, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    snap = tr.snapshot(10)
    assert snap["objects"]["tea_tin"]["seen"] is False  # 3 of 10 frames is not enough
    assert abs(snap["objects"]["tea_tin"]["x"] - fake_scene["objects"]["tea_tin"]["x"]) < POS_TOL
    assert snap["objects"]["pill_bottle"]["seen"] is True


def test_annotated_jpeg_is_always_a_jpeg(tracker):
    placeholder = tracker.annotated_jpeg()
    assert isinstance(placeholder, bytes) and placeholder[:2] == b"\xff\xd8"
    tracker.step()
    annotated = tracker.annotated_jpeg()
    assert annotated[:2] == b"\xff\xd8" and annotated != placeholder
    image = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (480, 640, 3)


def test_circular_median_wraps():
    assert circular_median([0.1, -0.1, 0.0]) == pytest.approx(0.0)
    med = circular_median([math.pi - 0.05, -math.pi + 0.05, math.pi - 0.02])
    assert abs(abs(med) - math.pi) < 0.05
    with pytest.raises(ValueError):
        circular_median([])


def test_tracker_rejects_bad_homography(fake_scene, cfg, table):
    with pytest.raises(ValueError):
        SceneTracker(FakeCamera(fake_scene), np.eye(2), table, cfg)


# -- FakeSceneTracker, get_camera, get_tracker ------------------------------------------------------------


def test_fake_scene_tracker_serves_the_fake_scene(fake_scene):
    tr = FakeSceneTracker()
    scene = tr.latest()
    assert abs(scene["t"] - time.time()) < 5.0 and scene["t"] != fake_scene["t"]
    scene["t"] = fake_scene["t"]
    assert scene == fake_scene
    assert tr.snapshot(10)["objects"] == fake_scene["objects"]
    moved = copy.deepcopy(fake_scene)
    moved["objects"]["juice_box"]["x"] = 0.3
    tr.set_scene(moved)
    assert tr.latest()["objects"]["juice_box"]["x"] == 0.3
    jpeg = tr.annotated_jpeg()
    assert jpeg[:2] == b"\xff\xd8"
    with pytest.raises(ValueError):
        tr.set_scene({"t": 1.0})


@pytest.mark.parametrize("flag", ["FAKE_CAMERA", "FAKE"])
def test_get_camera_and_tracker_honor_fake_flags(no_devices, monkeypatch, flag):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    monkeypatch.setenv(flag, "1")
    reset_tracker()
    try:
        cam = get_camera("overhead")
        assert isinstance(cam, FakeCamera) and cam.is_fake
        assert isinstance(get_camera("wrist"), FakeCamera)
        tr = get_tracker()
        assert isinstance(tr, FakeSceneTracker) and get_tracker() is tr
        validate_scene(tr.latest())
    finally:
        reset_tracker()


def test_real_camera_path_is_wired_but_opens_nothing_in_tests(no_devices, monkeypatch):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    monkeypatch.setenv("OVERHEAD_CAM", "3")
    with pytest.raises(AssertionError, match=r"VideoCapture\(3"):
        get_camera("overhead")


def test_env_overrides_camera_index(monkeypatch):
    monkeypatch.setenv("OVERHEAD_CAM", "7")
    monkeypatch.setenv("WRIST_CAM", "")
    assert camera_settings("overhead")["index"] == 7
    assert camera_settings("wrist")["index"] == 1
    assert camera_settings("overhead")["fps"] == 30 and camera_settings("overhead")["fourcc"] == "MJPG"
    monkeypatch.setenv("OVERHEAD_CAM", "usb")
    with pytest.raises(ValueError):
        camera_settings("overhead")


class _FakeCapture:
    def __init__(self):
        self.calls = 0
        self.released = False

    def read(self):
        self.calls += 1
        if self.calls == 1:
            return True, np.zeros((480, 640, 3), np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_camera_wrapper_counts_frames_and_failures(no_devices):
    cap = _FakeCapture()
    cam = Camera(0, name="overhead", capture=cap)
    assert cam.read() is not None and cam.read() is None
    assert cam.frames == 1 and cam.failures == 1 and cam.stats()["fps"] == 0.0
    cam.release()
    assert cap.released and cam.read() is None


def test_cameras_check_cli_runs_on_the_fakes(no_devices, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    assert cameras_mod.main(["--check", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "overhead:" in out and "wrist:" in out and "failed reads" in out


# -- CLIs and scripts -------------------------------------------------------------------------------------


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT")}
    env.update(extra)
    return env


def test_scene_print_cli_prints_valid_json_at_10hz():
    proc = subprocess.run(
        [sys.executable, "-m", "vision.scene", "--print", "--seconds", "1.2"],
        cwd=str(ROOT),
        env=_env(FAKE_CAMERA="1"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert 8 <= len(lines) <= 16, len(lines)
    for line in lines:
        SCENE_SCHEMA.validate(json.loads(line))
    assert set(json.loads(lines[-1])["objects"]) == {"pill_bottle", "juice_box", "tea_tin"}


def test_fake_chain_tracker_runs_without_the_singleton(fake_scene):
    tr = scene_mod.fake_chain_tracker()
    try:
        snap = tr.snapshot(3)
    finally:
        tr.close()
    for name, pos_err, yaw_err, seen in _pose_errors(snap, fake_scene):
        assert seen and pos_err < POS_TOL and yaw_err < YAW_TOL, name


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_markers_writes_detectable_pngs(tmp_path):
    make_markers = _load_script("make_markers")
    out = tmp_path / "markers"
    assert make_markers.main(["--out", str(out)]) == 0
    manifest = json.loads((out / "markers.json").read_text(encoding="utf-8"))
    ids = {m["id"] for m in manifest["markers"]}
    expected = set(CALIB_IDS) | {10, 11, 12, 30, HAND_ID} | {13, 14, 15, 16, 17}  # 13-17: I2's spare cards
    assert ids == expected
    assert {m["id"] for m in manifest["markers"] if m["spare"]} == {13, 14, 15, 16, 17}
    for entry in manifest["markers"]:
        file = out / entry["file"]
        assert file.exists() and entry["side_cm"] == marker_side(entry["id"]) * 100
        data = file.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"pHYs" in data[:64]
        image = cv2.imread(str(file))
        assert [d[0] for d in detect(image)] == [entry["id"]]
        side_px = int(round(entry["side_cm"] / 2.54 * 300))
        corners = detect(image)[0][1]
        assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(side_px, abs=3)
    assert (out / "a4_sheet.png").exists() and manifest["sheets"][0] == "a4_sheet.png"
    sheet = cv2.imread(str(out / "a4_sheet.png"))
    assert sheet.shape == (3508, 2480, 3)
    assert len(detect(sheet)) >= 4


def test_touch_markers_fake_runs_end_to_end(tmp_path):
    out = tmp_path / "touch"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "touch_markers.py"), "--fake", "--out", str(out)],
        cwd=str(ROOT),
        env=_env(FAKE_CAMERA="1", FAKE_ROBOT="1"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["passed"] is True and results["mode"] == "fake"
    assert {m["name"] for m in results["markers"]} == {"pill_bottle", "juice_box", "tea_tin"}
    assert results["worst_cm"] < 1.5
    for marker in results["markers"]:
        assert marker["ok"] and marker["problems"] == [] and marker["gap_cm"] < 1.5
        assert marker["replay_go"]["ok"] and marker["replay_back"]["ok"]
    assert "PASS" in proc.stdout


def test_touch_markers_refuses_without_a_mode():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "touch_markers.py")],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2 and "--fake" in proc.stderr


# -- review round 2: dead cameras, paint order, config checks, stuck threads, the hand check ---------------


class _DyingCamera(FakeCamera):
    """Delivers `alive` frames, then None forever: an unplugged USB camera (cv2 read() -> (False, None))."""

    def __init__(self, scene, alive, **kwargs):
        super().__init__(scene, **kwargs)
        self.alive = alive
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads > self.alive:
            return None
        return super().read()


class _FlakyCamera(FakeCamera):
    """Drops every `every`-th read, as a real camera drops the odd frame."""

    def __init__(self, scene, every, **kwargs):
        super().__init__(scene, **kwargs)
        self.every = every
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads % self.every == 0:
            return None
        return super().read()


class _RaisingCamera(FakeCamera):
    def read(self):
        raise RuntimeError("usb glitch")


class _BlockedCamera(FakeCamera):
    """read() blocks until `gate` is set: a hung USB device."""

    def __init__(self, scene, gate, **kwargs):
        super().__init__(scene, **kwargs)
        self.gate = gate

    def read(self):
        self.gate.wait()
        return super().read()


def test_snapshot_raises_when_the_camera_stops_delivering_frames(fake_scene, cfg, table, homography):
    cam = _DyingCamera(fake_scene, 2, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    with pytest.raises(TimeoutError, match="no frame"):
        tr.snapshot(10)
    assert cam.reads == 8 and tr.frames == 2 and tr.misses == 6  # 2 frames, then 6 misses of 10 end it


def test_running_snapshot_fails_fast_when_the_camera_dies(fake_scene, cfg, table, homography):
    cam = _DyingCamera(fake_scene, 4, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    tr.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="no frame"):
            tr.snapshot(10)
        assert time.monotonic() - t0 < 3.0  # the misses end it, not the 5 s publish deadline
    finally:
        tr.close()


def test_snapshot_skips_dropped_frames_and_still_uses_n_real_ones(fake_scene, cfg, table, homography):
    cam = _FlakyCamera(fake_scene, 4, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    snap = tr.snapshot(10)
    validate_scene(snap)
    assert tr.frames == 10 and tr.misses == 3 and cam.reads == 13
    assert all(it["seen"] for it in snap["objects"].values()) and snap["obstacles"]["cereal_box"]["seen"]


def test_running_snapshot_fails_fast_when_every_grab_raises(fake_scene, cfg, table, homography):
    tr = SceneTracker(_RaisingCamera(fake_scene, scene_cfg=cfg, objects=table), homography, table, cfg)
    tr.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            tr.snapshot(4, timeout=5.0)
        assert time.monotonic() - t0 < 2.0 and tr.errors >= 3 and tr.frames == 0
    finally:
        tr.close()


def test_fake_camera_item_markers_cover_the_calibration_markers_under_them(
    fake_scene, cfg, table, homography
):
    moved = copy.deepcopy(fake_scene)
    moved["objects"]["tea_tin"].update(x=0.40, y=0.25)  # its raised marker lands on calib 4 in the image
    moved["objects"]["juice_box"].update(x=0.10, y=-0.25)  # on calib 1
    moved["objects"]["pill_bottle"].update(x=0.10, y=0.25)  # on calib 0
    moved["obstacles"]["cereal_box"].update(x=0.36, y=-0.17)  # on calib 7
    cam = FakeCamera(moved, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    for _ in range(3):
        tr.step()
    scene = tr.latest()
    assert set(scene["objects"]) == set(moved["objects"]) and set(scene["obstacles"]) == {"cereal_box"}
    for name, pos_err, yaw_err, seen in _pose_errors(scene, moved):
        assert seen and pos_err < POS_TOL and yaw_err < YAW_TOL, (name, pos_err, yaw_err)


def test_load_objects_rejects_a_missing_or_zero_height(tmp_path):
    from vision.markers import load_objects

    good = tmp_path / "ok.yaml"
    good.write_text(
        "10: {name: a, kind: object, size: [0.03, 0.03], h: 0.05}\n40: {name: hand, kind: hand}\n",
        encoding="utf-8",
    )
    loaded = load_objects(good)
    assert loaded[10]["h"] == 0.05 and loaded[40]["h"] == 0.03
    for body in (
        "10: {name: a, kind: object, size: [0.03, 0.03]}",
        "10: {name: a, kind: object, size: [0.03, 0.03], h: 0}",
        "30: {name: b, kind: obstacle, size: [0.03, 0.03], h: -0.1}",
        "10: {name: a, kind: object, size: [0.03, 0], h: 0.05}",
        "10: {name: a, kind: object, size: [0.03, 0.03], h: tall}",
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(body + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_objects(bad)


def test_tracker_rejects_an_entry_without_a_positive_height(fake_scene, cfg, table, homography):
    flat = {k: dict(v) for k, v in table.items()}
    flat[12]["h"] = 0.0
    with pytest.raises(ValueError, match="tea_tin"):
        SceneTracker(FakeCamera(fake_scene), homography, flat, cfg)
    no_size = {k: dict(v) for k, v in table.items()}
    del no_size[11]["size"]
    with pytest.raises(ValueError, match="juice_box"):
        SceneTracker(FakeCamera(fake_scene), homography, no_size, cfg)


def test_fake_camera_ignores_a_junk_camera_index(no_devices, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    monkeypatch.setenv("OVERHEAD_CAM", "abc")
    assert isinstance(get_camera("overhead"), FakeCamera)
    monkeypatch.delenv("FAKE_CAMERA")
    monkeypatch.delenv("FAKE", raising=False)
    with pytest.raises(ValueError):
        get_camera("overhead")


def test_stop_keeps_the_reference_to_a_thread_stuck_in_read(fake_scene, cfg, table, homography):
    gate = threading.Event()
    cam = _BlockedCamera(fake_scene, gate, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    tr.start()
    time.sleep(0.05)
    try:
        assert tr.stop(timeout=0.2) is False
        assert tr.running  # still inside camera.read(): not reported as stopped
        with pytest.raises(RuntimeError):
            tr.start()
        with pytest.raises(TimeoutError):
            tr.snapshot(2, timeout=0.3)  # no synchronous grabs next to the stuck thread
        assert cam.frames == 0
    finally:
        gate.set()
    assert tr.stop(timeout=2.0) is True and not tr.running
    tr.close()


def test_touch_markers_hands_the_tracker_to_replay_for_the_wristband_check(monkeypatch, tmp_path):
    import arm.driver as driver_mod
    import arm.replay as replay_mod

    monkeypatch.setenv("FAKE_ROBOT", "1")
    monkeypatch.setenv("FAKE_CAMERA", "1")
    calls = []

    def recording_replay(traj, *, driver=None, scene=None, realtime=None, on_event=None, **kwargs):
        calls.append(scene)
        return {"ok": True, "reason": None, "events": []}

    monkeypatch.setattr(replay_mod, "replay", recording_replay)
    driver_mod.reset_driver()
    reset_tracker()
    try:
        tracker = get_tracker()  # the singleton the script hands to replay
        _load_script("touch_markers").main(["--fake", "--out", str(tmp_path / "touch")])
        assert len(calls) == 6 and all(callable(scene) for scene in calls)  # go and back for 3 markers
        assert all(getattr(scene, "__self__", None) is tracker for scene in calls)  # its bound `latest`
        validate_scene(calls[0]())
    finally:
        driver_mod.reset_driver()
        reset_tracker()


# -- hardening round 3 (DESIGN 9.9): raising cameras, --real refusals, Esc inside connect, pinned checks ----


class _GlitchingCamera(FakeCamera):
    """Delivers frames until `dead` is set, then raises on every read(): a USB driver error, not a None."""

    dead = False

    def read(self):
        if self.dead:
            raise RuntimeError("usb glitch")
        return super().read()


def test_a_raising_camera_is_a_miss_that_advances_the_seen_timeouts(fake_scene, cfg, table, homography):
    cam = _GlitchingCamera(fake_scene, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg)
    tr.step(now_mono=10.0)
    before = tr.latest()
    assert all(it["seen"] for it in before["objects"].values())
    cam.dead = True
    assert tr.step(now_mono=10.7) == {}  # the grab raised: a miss, not an exception out of step()
    assert (tr.frames, tr.misses, tr.errors) == (1, 1, 1)
    scene = tr.latest()
    assert scene["t"] >= before["t"] and set(scene["objects"]) == set(before["objects"])
    assert all(not it["seen"] for it in scene["objects"].values())
    assert scene["obstacles"]["cereal_box"]["seen"] is False and scene["hand"]["seen"] is False
    for group in ("objects", "obstacles"):  # the last pose is kept, as for a grab that returns None
        for name, item in scene[group].items():
            assert (item["x"], item["y"], item["yaw"]) == tuple(
                before[group][name][key] for key in ("x", "y", "yaw")
            )
    validate_scene(scene)
    with pytest.raises(
        TimeoutError, match="no frame"
    ):  # synchronous snapshot: TimeoutError, not RuntimeError
        tr.snapshot(4)
    assert tr.errors == 4 and tr.frames == 1  # three more raising grabs end a 4-frame snapshot


def test_running_tracker_marks_everything_unseen_while_the_camera_raises(fake_scene, cfg, table, homography):
    cam = _GlitchingCamera(fake_scene, scene_cfg=cfg, objects=table)
    tr = SceneTracker(cam, homography, table, cfg, hz=20)
    tr.start()
    try:
        assert tr.wait_for_frame(2.0) and tr.wait_for_frame(2.0)
        assert all(it["seen"] for it in tr.latest()["objects"].values())
        cam.dead = True
        t_dead = time.monotonic()
        while time.monotonic() - t_dead < 2.5:
            scene = tr.latest()
            if all(not it["seen"] for it in scene["objects"].values()):
                break
            time.sleep(0.02)
        else:
            pytest.fail("seen stayed True while the camera raised")
        flipped_after = time.monotonic() - t_dead
        assert 0.3 <= flipped_after <= 2.4, flipped_after  # the 0.5 s timeout, advanced by the misses
        assert tr.errors >= 3 and tr.misses >= 3 and abs(scene["t"] - time.time()) < 1.5
    finally:
        tr.close()


def test_get_camera_in_fake_mode_never_parses_the_camera_index(no_devices, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    monkeypatch.setenv("OVERHEAD_CAM", "not-a-number")

    def boom(*args, **kwargs):
        raise AssertionError("camera_settings must not run in fake mode")

    monkeypatch.setattr(cameras_mod, "camera_settings", boom)
    cam = get_camera("overhead")
    assert isinstance(cam, FakeCamera) and cam.read() is not None
    monkeypatch.delenv("FAKE_CAMERA")
    monkeypatch.delenv("FAKE", raising=False)
    with pytest.raises(AssertionError, match="camera_settings"):  # the real path parses the index first
        get_camera("overhead")


def test_load_objects_rejects_a_missing_size_and_a_nonpositive_height(tmp_path):
    from vision.markers import load_objects

    for body in (
        "10: {name: a, kind: object, h: 0.05}",  # no size at all
        "30: {name: b, kind: obstacle, h: 0.2}",
        "10: {name: a, kind: object, size: [0.03], h: 0.05}",
        "10: {name: a, kind: object, size: 0.03, h: 0.05}",
        "30: {name: b, kind: obstacle, size: [0.19, 0.06], h: 0}",
        "30: {name: b, kind: obstacle, size: [0.19, 0.06], h: -0.2}",
        "30: {name: b, kind: obstacle, size: [0.19, 0.06], h: .nan}",
        "30: {name: b, kind: obstacle, size: [0.19, -0.06], h: 0.2}",
    ):
        bad = tmp_path / "objects.yaml"
        bad.write_text(body + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"size|h must"):
            load_objects(bad)
    names = {entry["name"] for entry in marker_table().values() if entry["kind"] != "calib"}
    assert names == {"pill_bottle", "juice_box", "tea_tin", "cereal_box", "hand"}  # the real config loads


def test_fake_camera_paints_table_markers_under_the_raised_ones(fake_scene, cfg, table):
    moved = copy.deepcopy(fake_scene)
    moved["objects"]["tea_tin"].update(x=0.40, y=0.25)
    moved["objects"]["juice_box"].update(x=0.10, y=-0.25)
    moved["objects"]["pill_bottle"].update(x=0.10, y=0.25)
    moved["obstacles"]["cereal_box"].update(x=0.36, y=-0.17)
    cam = FakeCamera(moved, scene_cfg=cfg, objects=table)
    hidden = cam.hidden_calibration_ids()
    assert hidden == {0, 1, 4, 7}  # each raised item lands on one table marker in the image
    ids = {d[0] for d in detect(cam.read())}
    items = {it["marker"] for g in ("objects", "obstacles") for it in moved[g].values()}
    assert items <= ids, items - ids  # the raised markers are painted last, so they stay intact
    assert set(CALIB_IDS) - hidden <= ids and ids <= items | set(CALIB_IDS)


@pytest.mark.parametrize("flag", ["FAKE_CAMERA", "FAKE", "FAKE_ROBOT"])
def test_touch_markers_real_refuses_every_fake_flag(flag, tmp_path):
    out = tmp_path / "touch"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "touch_markers.py"), "--real", "--out", str(out)],
        cwd=str(ROOT),
        env=_env(**{flag: "1"}),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "refusing" in proc.stderr and flag in proc.stderr
    assert not out.exists()


def test_touch_markers_real_refuses_a_fake_tracker_installed_in_process(monkeypatch, tmp_path, capsys):
    import dotenv

    import arm.driver as driver_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: FakeSceneTracker())
    monkeypatch.setattr(driver_mod, "get_driver", lambda: pytest.fail("the driver must not be touched"))
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("prompted although refusing"))
    rc = _load_script("touch_markers").main(["--real", "--out", str(tmp_path / "touch")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing" in err and "FakeSceneTracker" in err
    assert not (tmp_path / "touch").exists()


class _Watcher:
    """A stand-in for arm.driver.EscWatcher that never reads the console; `pressed` is set by the test."""

    def __init__(self, callback=None, pressed=False):
        self.callback, self.pressed = callback, pressed

    def start(self):
        return None

    def stop(self):
        return None


@pytest.mark.parametrize("esc_seen_by", ["driver", "watcher"])
def test_touch_markers_real_aborts_when_esc_lands_inside_connect(monkeypatch, tmp_path, capsys, esc_seen_by):
    """An Esc during connect() (port open, goal sync, configure) ends the run before any move, whether the
    driver kept its `stopped` flag or connect() cleared it and only the watcher remembers the key."""
    import dotenv

    import arm.driver as driver_mod
    import arm.replay as replay_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))  # the A13 Esc record lands here
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)

    class EscInsideConnect(driver_mod.FakeDriver):
        def connect(self):
            super().connect()
            if esc_seen_by == "driver":
                self.freeze()

    driver = EscInsideConnect()
    monkeypatch.setattr(driver_mod, "get_driver", lambda: driver)
    monkeypatch.setattr(
        driver_mod, "EscWatcher", lambda callback=None: _Watcher(callback, pressed=esc_seen_by == "watcher")
    )
    chain = scene_mod.fake_chain_tracker()  # a real SceneTracker (is_fake False) on the FakeCamera
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: chain)
    monkeypatch.setattr("builtins.input", lambda *args: "yes")
    monkeypatch.setattr(replay_mod, "replay", lambda *args, **kwargs: pytest.fail("replay must not run"))
    module = _load_script("touch_markers")
    monkeypatch.setattr(module, "_countdown", lambda seconds, watcher: True)
    try:
        rc = module.main(["--real", "--out", str(tmp_path / "touch")])
    finally:
        chain.close()
    assert rc == 1
    captured = capsys.readouterr()
    assert "Esc during connect" in captured.err and "FROZEN" in captured.out
    assert driver.commands == [] and driver.stopped and driver.torque
    results = json.loads((tmp_path / "touch" / "results.json").read_text(encoding="utf-8"))
    assert results["mode"] == "real" and results["aborted"] is True and results["passed"] is False
    assert results["markers"] == [] and results["worst_cm"] is None


@pytest.mark.parametrize(
    "option", [["--above", "0"], ["--above", "-0.02"], ["--above", "nan"], ["--limit", "0"]]
)
def test_touch_markers_rejects_a_nonpositive_hover_height_or_limit(option):
    with pytest.raises(SystemExit) as exc:  # an argparse error, before any flag or scene is touched
        _load_script("touch_markers").main(["--fake", *option])
    assert exc.value.code == 2
