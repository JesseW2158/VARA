"""Lens wiring (B5's fallback): SceneTracker and vision.calibrate's live, check and check-points paths
undistort through config/lens.npz when it exists. No hardware: cameras are FakeCameras or stand-ins serving
FakeCamera frames, the lens path is pointed at tmp_path through `vision.lens.LENS_PATH`, and
cv2.VideoCapture fails the test if anything tries to open a device."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

import vision.calibrate as cal
import vision.lens as lens_mod
import vision.scene as scene_mod
from vision import cameras, markers
from vision.calibrate import FOCAL_PX, IMAGE_SIZE, PRINCIPAL, load_scene_cfg, synthetic_homography
from vision.cameras import FakeCamera, load_fake_scene
from vision.markers import marker_table
from vision.scene import SceneTracker, validate_scene


@pytest.fixture(autouse=True)
def no_devices_no_env(monkeypatch, tmp_path):
    """No camera device, no .env, runs under tmp_path, and config/lens.npz replaced by a missing tmp file."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(lens_mod, "LENS_PATH", tmp_path / "config" / "lens.npz")
    scene_mod.reset_tracker()
    yield
    scene_mod.reset_tracker()


@pytest.fixture(scope="module")
def cfg():
    return load_scene_cfg()


@pytest.fixture(scope="module")
def table():
    return marker_table()


def _lens_file(path, k1: float = 0.0, size=IMAGE_SIZE):
    """A lens file of the synthetic pinhole (FOCAL_PX, PRINCIPAL) with radial k1 at `size`."""
    camera_matrix = np.array([[FOCAL_PX, 0.0, PRINCIPAL[0]], [0.0, FOCAL_PX, PRINCIPAL[1]], [0.0, 0.0, 1.0]])
    lens = {
        "image_size": list(size),
        "camera_matrix": camera_matrix,
        "dist_coeffs": np.array([k1, 0.0, 0.0, 0.0, 0.0]),
        "rms": 0.1,
        "created_at": "2026-10-17T00:00:00+00:00",
    }
    return lens_mod.save_lens(lens, path)


class _PhysicalStandIn:
    """Looks like the real overhead camera (`is_fake = False`) but serves FakeCamera frames."""

    is_fake = False

    def __init__(self, cfg, table):
        self._inner = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        return self._inner.read()

    def release(self):
        self.released = True


def _scene_body(scene):
    return {key: value for key, value in scene.items() if key != "t"}


def _snapshot(tracker, n=3):
    try:
        return tracker.snapshot(n)
    finally:
        tracker.close()


# -- vision.calibrate helpers ----------------------------------------------------------------------------


def test_absent_lens_file_means_raw_frames(tmp_path):
    assert cal.load_active_lens(tmp_path / "nope.npz") is None
    assert cal.load_active_lens() is None  # the default path, redirected to a missing tmp file
    frame = np.zeros((480, 640, 3), np.uint8)
    assert cal.undistort_frame(frame, None) is frame
    assert cal.lens_hash(None) is None


def test_active_lens_carries_the_file_hash_and_maps(tmp_path):
    path = _lens_file(tmp_path / "lens.npz", k1=-0.05)
    lens = cal.load_active_lens(path)
    assert lens["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == cal.lens_hash(lens)
    assert len(lens["maps"]) == 2
    frame = FakeCamera().read()
    expected = lens_mod.undistort(frame, lens)
    got = cal.undistort_frame(frame, lens)
    assert got.shape == frame.shape and not np.array_equal(got, frame)
    # remap tables (16-bit fixed point) and cv2.undistort agree up to interpolation rounding
    assert np.mean(np.abs(got.astype(int) - expected.astype(int))) < 1.0


def test_corrupt_lens_file_raises(tmp_path):
    bad = tmp_path / "lens.npz"
    bad.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="not a lens file"):
        cal.load_active_lens(bad)


def test_frame_of_another_size_is_refused(tmp_path):
    lens = cal.load_active_lens(_lens_file(tmp_path / "lens.npz", size=(320, 240)))
    with pytest.raises(ValueError, match="calibrated at 320x240"):
        cal.undistort_frame(np.zeros((480, 640, 3), np.uint8), lens)


# -- SceneTracker ----------------------------------------------------------------------------------------


def test_zero_distortion_lens_leaves_the_tracker_output_unchanged(tmp_path, cfg, table):
    lens = cal.load_active_lens(_lens_file(tmp_path / "lens.npz", k1=0.0))
    h = synthetic_homography(cfg)
    plain = SceneTracker(FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table), h, table, cfg)
    wired = SceneTracker(
        FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table), h, table, cfg, lens=lens
    )
    raw_scene, lens_scene = _snapshot(plain), _snapshot(wired)
    assert lens_scene["objects"] and all(item["seen"] for item in lens_scene["objects"].values())
    assert _scene_body(lens_scene) == _scene_body(raw_scene)
    assert np.array_equal(wired.last_frame(), plain.last_frame())
    validate_scene(lens_scene)
    validate_scene(wired.latest())
    assert "lens" not in lens_scene and "lens" not in wired.latest()  # the scene contract has no room for it
    assert plain.lens_sha256 is None and plain.diagnostics()["lens"] is None
    assert wired.lens_sha256 == lens["sha256"] == wired.diagnostics()["lens"]
    assert wired.diagnostics()["type"] == "SceneTracker" and wired.diagnostics()["frames"] == 3


def test_frames_are_undistorted_before_detection_and_in_the_stream(tmp_path, cfg, table, monkeypatch):
    lens = cal.load_active_lens(_lens_file(tmp_path / "lens.npz", k1=-0.08))
    raw = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table).read()
    seen_by_detect = []
    real_detect = scene_mod.detect

    def spy(frame, det=None):
        seen_by_detect.append(frame.copy())
        return real_detect(frame, det)

    monkeypatch.setattr(scene_mod, "detect", spy)
    camera = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
    tracker = SceneTracker(camera, synthetic_homography(cfg), table, cfg, lens=lens)
    tracker.step()
    expected = cal.undistort_frame(raw, lens)
    assert not np.array_equal(expected, raw)
    assert np.array_equal(seen_by_detect[0], expected)
    assert np.array_equal(tracker.last_frame(), expected)  # what annotated_jpeg draws on
    jpeg = tracker.annotated_jpeg()
    assert jpeg[:2] == b"\xff\xd8"
    validate_scene(tracker.latest())
    tracker.close()


def test_fake_feeds_are_never_undistorted_unless_asked(tmp_path, cfg, table, monkeypatch):
    _lens_file(lens_mod.LENS_PATH, k1=-0.08)  # a valid lens at the default path
    chain = scene_mod.fake_chain_tracker()
    try:
        assert chain.lens_sha256 is None and chain.diagnostics()["lens"] is None
    finally:
        chain.close()
    monkeypatch.setenv("FAKE_CAMERA", "1")
    fake = scene_mod.get_tracker()
    assert isinstance(fake, scene_mod.FakeSceneTracker)
    assert fake.lens_sha256 is None and fake.diagnostics() == {
        "type": "FakeSceneTracker",
        "frames": None,
        "misses": None,
        "errors": None,
        "lens": None,
    }


def test_a_frame_the_lens_cannot_take_is_a_failed_frame(tmp_path, cfg, table):
    lens = cal.load_active_lens(_lens_file(tmp_path / "lens.npz", size=(320, 240)))
    camera = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table)
    tracker = SceneTracker(camera, synthetic_homography(cfg), table, cfg, lens=lens)
    with pytest.raises(ValueError, match="calibrated at 320x240"):
        tracker.step()
    tracker.close()


def test_tracker_refuses_a_lens_that_was_not_loaded_for_it(tmp_path, cfg, table):
    plain = lens_mod.load_lens(_lens_file(tmp_path / "lens.npz"))  # no sha256, no maps
    with pytest.raises(ValueError, match="load_active_lens"):
        SceneTracker(
            FakeCamera(scene_cfg=cfg, objects=table), synthetic_homography(cfg), table, cfg, lens=plain
        )


# -- physical factories: absent, valid and corrupt lens files -------------------------------------------


def _physical_patches(monkeypatch, cfg, table):
    opened = []

    def get_camera(name):
        camera = _PhysicalStandIn(cfg, table)
        opened.append(camera)
        return camera

    monkeypatch.setattr(scene_mod, "get_camera", get_camera)
    monkeypatch.setattr(scene_mod, "load_homography", lambda *a, **k: synthetic_homography(cfg))
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    return opened


def test_physical_tracker_without_a_lens_file_is_unchanged(cfg, table, monkeypatch):
    _physical_patches(monkeypatch, cfg, table)
    assert not lens_mod.LENS_PATH.exists()
    tracker = scene_mod._physical_tracker()
    reference = SceneTracker(
        FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table), synthetic_homography(cfg), table, cfg
    )
    assert tracker.lens_sha256 is None
    assert _scene_body(_snapshot(tracker)) == _scene_body(_snapshot(reference))


def test_physical_tracker_and_singleton_load_the_lens_file(cfg, table, monkeypatch):
    opened = _physical_patches(monkeypatch, cfg, table)
    path = _lens_file(lens_mod.LENS_PATH, k1=0.0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tracker = scene_mod._physical_tracker()
    try:
        assert tracker.lens_sha256 == digest
        report = scene_mod.run_accuracy_check(tracker, {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3}})
        assert report["tracker"]["lens"] == digest and report["passed"]
        validate_scene(report["snapshot"])
    finally:
        tracker.close()
    singleton = scene_mod.get_tracker()
    assert isinstance(singleton, SceneTracker) and singleton.lens_sha256 == digest
    assert singleton.running and len(opened) == 2


def test_corrupt_lens_fails_every_physical_startup_before_a_camera_opens(cfg, table, monkeypatch, capsys):
    opened = _physical_patches(monkeypatch, cfg, table)
    lens_mod.LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lens_mod.LENS_PATH.write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(ValueError, match="not a lens file"):
        scene_mod._physical_tracker()
    with pytest.raises(ValueError, match="not a lens file"):
        scene_mod.get_tracker()
    layout = lens_mod.LENS_PATH.parent / "layout.json"
    layout.write_text(json.dumps({"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3}}}))
    assert scene_mod.main(["--accuracy", str(layout)]) == 2
    assert "not a lens file" in capsys.readouterr().err
    assert opened == []


def test_accuracy_and_stream_evidence_state_raw_frames_for_the_fake_chain(tmp_path, capsys):
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"items": {"juice_box": {"x": 0.24, "y": -0.04, "yaw": 0.3}}}))
    out = tmp_path / "report.json"
    assert scene_mod.main(["--accuracy", str(layout), "--fake", "--warmup", "0", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["tracker"]["lens"] is None and report["tracker"]["type"] == "SceneTracker"
    capsys.readouterr()
    assert scene_mod.main(["--print", "--fake", "--seconds", "0.3"]) == 0
    assert "lens null (raw frames)" in capsys.readouterr().err


# -- vision.calibrate CLI ---------------------------------------------------------------------------------


def _calibrate_camera(monkeypatch, cfg, table):
    opened = []

    def get_camera(name):
        camera = _PhysicalStandIn(cfg, table)
        opened.append(camera)
        return camera

    monkeypatch.setattr(cameras, "get_camera", get_camera)
    return opened


def test_live_fit_undistorts_before_detection_and_reports_the_lens(tmp_path, cfg, table, monkeypatch, capsys):
    opened = _calibrate_camera(monkeypatch, cfg, table)
    path = _lens_file(lens_mod.LENS_PATH, k1=0.0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = []
    real_detect = markers.detect

    def spy(frame, det=None):
        frames.append(frame)
        return real_detect(frame, det)

    monkeypatch.setattr(markers, "detect", spy)
    out = tmp_path / "homography.npy"
    assert cal.main(["--live", "--frames", "2", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"lens: {digest}" in text and out.exists()
    assert len(frames) == 2 and opened[0].released
    assert "fit-marker residuals" in text and "PASS" in text


def test_live_fit_without_a_lens_says_raw_frames(tmp_path, cfg, table, monkeypatch, capsys):
    _calibrate_camera(monkeypatch, cfg, table)
    out = tmp_path / "homography.npy"
    assert cal.main(["--live", "--no-save", "--frames", "1", "--out", str(out)]) == 0
    assert "lens: null" in capsys.readouterr().out and not out.exists()


def test_live_check_with_a_distorting_lens_detects_on_undistorted_frames(tmp_path, cfg, table, monkeypatch):
    opened = _calibrate_camera(monkeypatch, cfg, table)
    lens = cal.load_active_lens(_lens_file(lens_mod.LENS_PATH, k1=-0.08))
    raw = FakeCamera(load_fake_scene(), scene_cfg=cfg, objects=table).read()
    frames = []
    monkeypatch.setattr(markers, "detect", lambda frame, det=None: frames.append(frame) or [])
    out = tmp_path / "homography.npy"
    cal.save_homography(synthetic_homography(cfg), out)
    assert cal.main(["--check", "--frames", "1", "--out", str(out)]) == 1  # nothing detected: refused
    assert np.array_equal(frames[0], cal.undistort_frame(raw, lens)) and opened[0].released


def test_corrupt_lens_fails_calibration_before_the_camera_opens(tmp_path, cfg, table, monkeypatch, capsys):
    opened = _calibrate_camera(monkeypatch, cfg, table)
    lens_mod.LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lens_mod.LENS_PATH.write_bytes(b"garbage")
    out = tmp_path / "homography.npy"
    cal.save_homography(synthetic_homography(cfg), out)
    before = out.read_bytes()
    for mode in ("--live", "--check"):
        assert cal.main([mode, "--out", str(out)]) == 1
        err = capsys.readouterr().err
        assert "not a lens file" in err and "unchanged" in err
    assert opened == [] and out.read_bytes() == before


def test_offline_check_points_report_the_lens(tmp_path, cfg, monkeypatch, capsys):
    h = synthetic_homography(cfg)
    out = tmp_path / "homography.npy"
    cal.save_homography(h, out)
    spots = [(0.10, -0.20), (0.30, -0.15), (0.20, 0.05), (0.35, 0.10), (0.15, 0.15)]
    inverse = np.linalg.inv(h)
    points = []
    for x, y in spots:
        u, v, w = inverse @ np.array([x, y, 1.0])
        points.append({"pixel": [u / w, v / w], "table": [x, y]})
    evidence = tmp_path / "checks.json"
    evidence.write_text(json.dumps(points))

    def no_camera(*_):
        raise AssertionError("the offline check must not open a camera")

    monkeypatch.setattr(cameras, "get_camera", no_camera)
    assert cal.main(["--check", "--out", str(out), "--check-points", str(evidence)]) == 0
    assert "lens: null" in capsys.readouterr().out
    path = _lens_file(lens_mod.LENS_PATH)
    assert cal.main(["--check", "--out", str(out), "--check-points", str(evidence)]) == 0
    assert f"lens: {hashlib.sha256(path.read_bytes()).hexdigest()}" in capsys.readouterr().out


def test_fake_calibration_never_uses_the_lens(tmp_path, monkeypatch, capsys):
    lens_mod.LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lens_mod.LENS_PATH.write_bytes(b"garbage")  # would fail any path that loads it
    out = tmp_path / "homography.npy"
    assert cal.main(["--fake", "--out", str(out)]) == 0
    assert "lens: null (synthetic pinhole" in capsys.readouterr().out
