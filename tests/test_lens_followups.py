"""Lens follow-ups: scripts/capture_table.py, the orchestrator's POST /calibrate marker check and
vision.verify's evidence undistort with (or record) the config/lens.npz the tracker uses. No hardware:
cameras are stubs serving generated or FakeCamera frames, `vision.lens.LENS_PATH` points under tmp_path,
and `cv2.VideoCapture` or an unpatched `vision.cameras.get_camera` fails the test."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import vision.calibrate as cal
import vision.lens as lens_mod
from agents import orchestrator
from scripts import capture_table
from vision import cameras, markers
from vision import verify as verify_mod
from vision.calibrate import FOCAL_PX, IMAGE_SIZE, PRINCIPAL

ROOT = Path(__file__).resolve().parents[1]
REAL_DETECT = markers.detect
OTHER_SHA = "ab" * 32  # a lens sha256 no file here has


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No device, no .env, no FAKE flags, runs under tmp_path and config/lens.npz a missing tmp file."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    def no_camera(_name):
        pytest.fail("a camera was opened")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    monkeypatch.setattr(cameras, "get_camera", no_camera)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for flag in ("FAKE", "FAKE_CAMERA"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("VARA_CHECK_POINTS", str(tmp_path / "no-check-points.json"))
    monkeypatch.setattr(lens_mod, "LENS_PATH", tmp_path / "config" / "lens.npz")
    monkeypatch.setattr(capture_table, "DEFAULT_MANIFEST", tmp_path / "no-manifest" / "markers.json")
    monkeypatch.setattr(capture_table, "DEFAULT_CHECK_POINTS", tmp_path / "default" / "check-points.json")


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


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StubCamera:
    """Looks like the physical overhead camera (`is_fake = False`) and serves copies of one frame."""

    is_fake = False
    index = 2

    def __init__(self, frame):
        self.frame = frame
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        return self.frame.copy()

    def release(self):
        self.released = True


def spot_frame(row: int, col: int, side: int = 60):
    """A white 640x480 frame with marker 10 pasted at (top row, left column)."""
    frame = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 255, dtype=np.uint8)
    frame[row : row + side, col : col + side] = cv2.cvtColor(markers.generate(10, side), cv2.COLOR_GRAY2BGR)
    return frame


def use_camera(monkeypatch, frame) -> StubCamera:
    camera = StubCamera(frame)
    monkeypatch.setattr(cameras, "get_camera", lambda _name: camera)
    return camera


def record_detect(monkeypatch) -> list[np.ndarray]:
    """Every frame handed to `vision.markers.detect` from now on (the real detector still runs)."""
    seen: list[np.ndarray] = []

    def recording(frame, *args, **kwargs):
        seen.append(np.array(frame, copy=True))
        return REAL_DETECT(frame, *args, **kwargs)

    monkeypatch.setattr(markers, "detect", recording)
    return seen


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -- scripts/capture_table.py ----------------------------------------------------------------------------


def test_zero_distortion_lens_leaves_capture_detections_unchanged_and_records_its_hash(monkeypatch, tmp_path):
    frame = cameras.FakeCamera().read()
    use_camera(monkeypatch, frame)
    raw = capture_table.capture(tmp_path / "raw")
    path = _lens_file(lens_mod.LENS_PATH, k1=0.0)
    wired = capture_table.capture(tmp_path / "lens")
    assert raw["lens"] is None and wired["lens"] == _sha(path)
    assert raw["markers"] and [m["id"] for m in wired["markers"]] == [m["id"] for m in raw["markers"]]
    for a, b in zip(wired["markers"], raw["markers"], strict=True):
        assert a["pixel"] == pytest.approx(b["pixel"], abs=1e-6)
    assert "undistorted" in wired["center_method"] and "undistorted" not in raw["center_method"]
    assert _read_json(tmp_path / "lens" / "markers.json")["lens"] == wired["lens"]
    assert np.array_equal(cv2.imread(str(tmp_path / "lens" / "physical-original.png")), frame)


def test_a_distorting_lens_undistorts_before_detection_and_keeps_the_original_raw(monkeypatch, tmp_path):
    frame = cameras.FakeCamera().read()
    camera = use_camera(monkeypatch, frame)
    path = _lens_file(lens_mod.LENS_PATH, k1=-0.08)
    seen = record_detect(monkeypatch)
    report = capture_table.capture(tmp_path / "spot", frames=2)
    expected = cal.undistort_frame(frame, cal.load_active_lens(path))
    assert len(seen) == 2 and all(np.array_equal(f, expected) for f in seen)
    assert not np.array_equal(expected, frame)
    assert np.array_equal(cv2.imread(str(tmp_path / "spot" / "physical-original.png")), frame)
    by_id = {m["id"]: m["pixel"] for m in report["markers"]}
    raw_by_id = {int(i): np.asarray(c).reshape(4, 2).mean(axis=0).tolist() for i, c in REAL_DETECT(frame)}
    moved = [i for i in by_id if i in raw_by_id and np.hypot(*np.subtract(by_id[i], raw_by_id[i])) > 0.5]
    assert moved, "a k1 = -0.08 lens moves marker centers"
    assert report["lens"] == _sha(path) and camera.released


def test_check_point_provenance_records_the_lens_and_the_list_keeps_pixel_and_table(
    monkeypatch, tmp_path, capsys
):
    path = _lens_file(lens_mod.LENS_PATH, k1=0.0)
    use_camera(monkeypatch, spot_frame(100, 200))
    points = tmp_path / "measurements" / "check-points.json"
    argv = ["--out", str(tmp_path / "spot-1"), "--check-point", "0.21", "-0.08", "--append", str(points)]
    assert capture_table.main(argv) == 0
    out = capsys.readouterr().out
    assert f"lens: {_sha(path)} (config/lens.npz: every frame undistorted" in out
    (entry,) = _read_json(points)
    assert set(entry) == {"pixel", "table"}  # what vision.calibrate.load_check_points accepts
    assert entry["pixel"] == pytest.approx([229.5, 129.5], abs=0.1)
    (record,) = _read_json(capture_table.provenance_path(points))
    assert record["lens"] == _sha(path) and record["index"] == 0
    report = _read_json(tmp_path / "spot-1" / "markers.json")
    assert report["lens"] == _sha(path) and report["check_point"]["appended"] is True


def test_absent_lens_file_records_null_and_says_raw_frames(monkeypatch, tmp_path, capsys):
    assert not lens_mod.LENS_PATH.exists()
    use_camera(monkeypatch, spot_frame(100, 200))
    points = tmp_path / "check-points.json"
    argv = ["--out", str(tmp_path / "spot-1"), "--check-point", "0.21", "-0.08", "--append", str(points)]
    assert capture_table.main(argv) == 0
    assert "lens: null (no config/lens.npz: markers detected on raw frames)" in capsys.readouterr().out
    assert _read_json(tmp_path / "spot-1" / "markers.json")["lens"] is None
    (record,) = _read_json(capture_table.provenance_path(points))
    assert record["lens"] is None


def test_corrupt_lens_is_refused_before_any_camera_opens(tmp_path, capsys):
    lens_mod.LENS_PATH.parent.mkdir(parents=True)
    lens_mod.LENS_PATH.write_bytes(b"PK\x03\x04 truncated")
    out = tmp_path / "spot-1"
    points = tmp_path / "check-points.json"
    argv = ["--out", str(out), "--check-point", "0.21", "-0.08", "--append", str(points)]
    assert capture_table.main(argv) == 2  # the fixture fails the test if a camera opens
    err = capsys.readouterr().err
    assert err.startswith("Refused:") and "not a lens file" in err
    assert not out.exists() and not points.exists()
    with pytest.raises(ValueError, match="not a lens file"):
        capture_table.capture(out)


def test_fake_capture_never_reads_the_lens_file(tmp_path, capsys):
    lens_mod.LENS_PATH.parent.mkdir(parents=True)
    lens_mod.LENS_PATH.write_bytes(b"garbage")  # would fail any path that loads it
    assert capture_table.main(["--out", str(tmp_path / "fake"), "--fake"]) == 0
    assert "lens: null (synthetic FakeCamera" in capsys.readouterr().out
    assert _read_json(tmp_path / "fake" / "markers.json")["lens"] is None


def test_a_frame_of_another_size_than_the_lens_fails_the_capture(monkeypatch, tmp_path, capsys):
    _lens_file(lens_mod.LENS_PATH, size=(320, 240))
    camera = use_camera(monkeypatch, spot_frame(100, 200))
    out = tmp_path / "spot-1"
    assert capture_table.main(["--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("Capture failed:") and "move config/lens.npz aside" in err
    assert camera.released and not out.exists()


def test_a_list_recorded_with_another_lens_is_refused_before_the_camera_opens(monkeypatch, tmp_path, capsys):
    points = tmp_path / "check-points.json"
    camera = use_camera(monkeypatch, spot_frame(100, 200))
    first = ["--out", str(tmp_path / "spot-1"), "--check-point", "0.21", "-0.08", "--append", str(points)]
    assert capture_table.main(first) == 0  # raw frames: lens null
    path = _lens_file(lens_mod.LENS_PATH)
    camera.frame = spot_frame(200, 300)
    reads = camera.reads
    second = ["--out", str(tmp_path / "spot-2"), "--check-point", "0.30", "-0.10", "--append", str(points)]
    assert capture_table.main(second) == 2
    err = capsys.readouterr().err
    assert "raw and undistorted pixels do not mix" in err and f"uses lens {_sha(path)}" in err
    assert camera.reads == reads and not (tmp_path / "spot-2").exists()
    assert len(capture_table.load_check_point_list(points)) == 1
    with pytest.raises(ValueError, match="do not mix"):
        capture_table.append_check_point(
            points,
            [400.0, 300.0],
            [0.3, -0.1],
            spot_id=10,
            image="x.png",
            mode="physical",
            t="t",
            lens=_sha(path),
        )


def test_records_without_a_lens_key_count_as_raw(monkeypatch, tmp_path):
    points = tmp_path / "check-points.json"
    points.write_text(json.dumps([{"pixel": [100.0, 100.0], "table": [0.1, 0.1]}]), encoding="utf-8")
    legacy = {"index": 0, "pixel": [100.0, 100.0], "table": [0.1, 0.1], "spot_id": 10, "image": "a.png"}
    capture_table.provenance_path(points).write_text(json.dumps([legacy]), encoding="utf-8")
    camera = use_camera(monkeypatch, spot_frame(100, 200))
    report = capture_table.capture(
        tmp_path / "raw", check_point=capture_table.CheckPoint((0.21, -0.08), 10, points)
    )
    assert report["check_point"]["appended"] is True and report["lens"] is None
    assert [r.get("lens", "missing") for r in _read_json(capture_table.provenance_path(points))] == [
        "missing",
        None,
    ]
    _lens_file(lens_mod.LENS_PATH)
    camera.frame = spot_frame(200, 300)
    with pytest.raises(ValueError, match="do not mix"):
        capture_table.capture(
            tmp_path / "lens", check_point=capture_table.CheckPoint((0.3, -0.1), 10, points)
        )
    assert not (tmp_path / "lens").exists()


# -- agents/orchestrator.py POST /calibrate ---------------------------------------------------------------


def calibration_instance(monkeypatch, tracker):
    """An Orchestrator without __init__ (no server, no threads) checking the synthetic homography."""
    instance = object.__new__(orchestrator.Orchestrator)
    instance.tracker = tracker
    homography = cal.synthetic_homography()
    monkeypatch.setattr(orchestrator.calibrate, "load_homography", lambda path=None: homography.copy())
    return instance


def never_load(*args, **kwargs):
    raise AssertionError("the lens file must not be loaded here")


def test_calibrate_undistorts_a_camera_frame_with_the_trackers_lens(monkeypatch):
    path = _lens_file(lens_mod.LENS_PATH, k1=-0.08)
    lens = cal.load_active_lens(path)
    frame = cameras.FakeCamera().read()
    camera = use_camera(monkeypatch, frame)
    tracker = SimpleNamespace(last_frame=lambda: None, lens_sha256=lens["sha256"])
    instance = calibration_instance(monkeypatch, tracker)
    seen = record_detect(monkeypatch)
    accepted = {"accepted": True, "mean_cm": 0.123, "max_cm": 0.2}
    monkeypatch.setattr(orchestrator.calibrate, "calibration_metrics", lambda *_: dict(accepted))
    report = instance.calibration_report()
    assert report == {"error_cm": 0.12, "check_points": None, "lens": lens["sha256"]}
    assert np.array_equal(seen[0], cal.undistort_frame(frame, lens)) and not np.array_equal(seen[0], frame)
    assert camera.released
    monkeypatch.setattr(orchestrator.calibrate, "load_active_lens", never_load)  # loaded once, then cached
    assert instance.calibrate() == {"error_cm": 0.12}  # the method keeps its marker-check reply
    assert np.array_equal(seen[1], seen[0])


def test_calibrate_with_a_zero_distortion_lens_passes_on_the_synthetic_camera(monkeypatch):
    path = _lens_file(lens_mod.LENS_PATH, k1=0.0)
    use_camera(monkeypatch, cameras.FakeCamera().read())
    tracker = SimpleNamespace(last_frame=lambda: None, lens_sha256=_sha(path))
    report = calibration_instance(monkeypatch, tracker).calibration_report()
    assert (
        0.0 <= report["error_cm"] <= 0.3 and report["lens"] == _sha(path) and report["check_points"] is None
    )


def test_calibrate_uses_the_trackers_frame_as_it_is(monkeypatch):
    frame = cameras.FakeCamera().read()  # stands for a frame the tracker already undistorted
    tracker = SimpleNamespace(last_frame=lambda: frame.copy(), lens_sha256=OTHER_SHA)
    instance = calibration_instance(monkeypatch, tracker)
    monkeypatch.setattr(orchestrator.calibrate, "load_active_lens", never_load)
    seen = record_detect(monkeypatch)
    report = instance.calibration_report()  # the fixture fails the test if a camera opens
    assert report["lens"] == OTHER_SHA and np.array_equal(seen[0], frame)


def test_calibrate_refuses_a_lens_file_that_is_not_the_trackers(monkeypatch):
    tracker = SimpleNamespace(last_frame=lambda: None, lens_sha256=OTHER_SHA)
    instance = calibration_instance(monkeypatch, tracker)
    with pytest.raises(HTTPException) as absent:
        instance.calibration_report()  # no lens file at all; the camera never opens
    assert absent.value.status_code == 503 and "restart the orchestrator" in absent.value.detail
    path = _lens_file(lens_mod.LENS_PATH)
    with pytest.raises(HTTPException) as other:
        instance.calibration_report()
    assert other.value.status_code == 503 and _sha(path) in other.value.detail


def test_calibrate_keeps_a_stand_in_tracker_without_a_lens_raw(monkeypatch):
    _lens_file(lens_mod.LENS_PATH, k1=-0.08)  # present, but the tracker does not use it
    frame = cameras.FakeCamera().read()
    use_camera(monkeypatch, frame)
    instance = calibration_instance(monkeypatch, SimpleNamespace(last_frame=lambda: None))
    monkeypatch.setattr(orchestrator.calibrate, "load_active_lens", never_load)
    seen = record_detect(monkeypatch)
    report = instance.calibration_report()
    assert report["lens"] is None and np.array_equal(seen[0], frame)


def test_calibrate_without_a_tracker_refuses_a_corrupt_lens_before_the_camera_opens(monkeypatch):
    lens_mod.LENS_PATH.parent.mkdir(parents=True)
    lens_mod.LENS_PATH.write_bytes(b"garbage")
    instance = calibration_instance(monkeypatch, None)
    with pytest.raises(HTTPException) as exc:
        instance.calibration_report()
    assert exc.value.status_code == 503 and "not a lens file" in exc.value.detail


def test_calibrate_under_fake_camera_reports_no_lens(monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    instance = calibration_instance(monkeypatch, SimpleNamespace(lens_sha256=OTHER_SHA))
    assert instance.calibration_report() == {"error_cm": 0.8, "check_points": None, "lens": None}


# -- vision/verify.py evidence ----------------------------------------------------------------------------


def verify_fixture(name: str) -> dict:
    return _read_json(ROOT / "tests" / "fixtures" / "verify" / f"{name}.json")


class LensStub:
    """A tracker with a lens sha256 whose snapshots come from a list (the last one repeats)."""

    def __init__(self, scenes, lens):
        self.scenes = [copy.deepcopy(s) for s in scenes]
        self.lens_sha256 = lens
        self.frames = 0

    def snapshot(self, n=10):
        self.frames += n
        return copy.deepcopy(self.scenes[min(self.frames // n, len(self.scenes)) - 1])


def test_verify_cli_evidence_carries_the_tracker_diagnostics(tmp_path, capsys):
    fx = verify_fixture("pass_juice_box_in_tray")
    scenes = {name: tmp_path / f"{name}.json" for name in ("before", "after", "spec")}
    for name, file in scenes.items():
        file.write_text(json.dumps(fx[name]), encoding="utf-8")
    session = tmp_path / "session"
    argv = ["--fake", "--fake-scene", str(scenes["before"]), "--before", "--out", str(session)]
    assert verify_mod.main(argv) == 0
    assert "SYNTHETIC lens: null (no undistortion)" in capsys.readouterr().out
    expected = {"type": "FakeSceneTracker", "frames": None, "misses": None, "errors": None, "lens": None}
    assert _read_json(session / "session.json")["tracker_diagnostics"] == expected
    argv = ["--fake", "--fake-scene", str(scenes["after"]), "--after", "--spec", str(scenes["spec"])]
    assert verify_mod.main([*argv, "--wait", "0", "--out", str(session)]) == 0
    assert _read_json(session / "result.json")["tracker_diagnostics"] == expected


def test_verify_records_the_lens_and_refuses_an_after_with_another(tmp_path):
    fx = verify_fixture("pass_juice_box_in_tray")
    sha = "cd" * 32
    folder = tmp_path / "session"
    verify_mod.take_before(LensStub([fx["before"]], sha), folder, frames=2)
    session = _read_json(folder / "session.json")
    assert (
        session["tracker_diagnostics"]["lens"] == sha and session["tracker_diagnostics"]["type"] == "LensStub"
    )
    result = verify_mod.run_after(
        LensStub([fx["after"]], sha), folder, fx["spec"], {}, frames=2, wait=0, retry=0
    )
    assert result["ok"] is True and result["tracker_diagnostics"]["lens"] == sha
    assert _read_json(folder / "result.json")["tracker_diagnostics"]["frames"] == 2
    mixed = tmp_path / "mixed"
    verify_mod.take_before(LensStub([fx["before"]], None), mixed, frames=2)
    with pytest.raises(ValueError, match="never mixes raw and undistorted"):
        verify_mod.run_after(LensStub([fx["after"]], sha), mixed, fx["spec"], {}, frames=2, wait=0, retry=0)
    assert sorted(p.name for p in mixed.iterdir()) == ["before.json", "session.json"]
    legacy = tmp_path / "legacy"  # a session.json from before the diagnostics existed is not checked
    verify_mod.take_before(LensStub([fx["before"]], None), legacy, frames=2)
    data = _read_json(legacy / "session.json")
    del data["tracker_diagnostics"]
    (legacy / "session.json").write_text(json.dumps(data), encoding="utf-8")
    assert verify_mod.run_after(LensStub([fx["after"]], sha), legacy, fx["spec"], {}, wait=0, retry=0)["ok"]
