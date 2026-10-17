"""Calibration gates, independent measured points and failed-write preservation. No cameras or motors."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from agents import orchestrator
from vision import calibrate as cal
from vision import cameras

H = np.diag([0.001, 0.001, 1.0])
SPOTS = [(0.10, 0.10), (0.30, 0.10), (0.30, 0.30), (0.10, 0.30), (0.20, 0.20)]


def marker_data(errors_cm):
    markers = {index: SPOTS[index] for index in range(len(errors_cm))}
    dets = []
    for index, error in enumerate(errors_cm):
        center = np.asarray(markers[index]) * 1000 + [error * 10, 0]
        dets.append((index, center + np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])))
    return markers, dets


def check_points():
    return [{"pixel": [x * 1000, y * 1000], "table": [x, y]} for x, y in SPOTS]


@pytest.mark.parametrize(
    ("errors", "accepted"),
    [([0.0] * 4, True), ([1.5, 1.5, 0.5, 0.5], True), ([1.2] * 4, False), ([0, 0, 0, 1.6], False)],
)
def test_marker_acceptance_checks_mean_and_worst_point(errors, accepted):
    markers, dets = marker_data(errors)
    metrics = cal.calibration_metrics(H, dets, markers)
    assert metrics["count"] == 4
    assert metrics["mean_cm"] == pytest.approx(np.mean(errors))
    assert metrics["max_cm"] == pytest.approx(max(errors))
    assert metrics["accepted"] is accepted
    assert cal.calibration_error(H, dets, markers) == metrics["mean_cm"]


def test_mean_only_api_remains_compatible_but_acceptance_needs_four_unique_markers():
    markers, dets = marker_data([0.5] * 3)
    assert cal.calibration_error(H, dets[:1], markers) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="need 4"):
        cal.calibration_metrics(H, dets, markers)
    with pytest.raises(ValueError, match="duplicate"):
        cal.calibration_metrics(H, [*dets, dets[0]], markers)


def test_independent_checks_measure_accuracy_outside_the_fit_markers():
    points = check_points()
    assert cal.check_points_metrics(H, points)["accepted"]
    points[-1]["table"][0] += 0.02
    metrics = cal.check_points_metrics(H, points)
    assert metrics["mean_cm"] == pytest.approx(0.4)
    assert metrics["max_cm"] == pytest.approx(2.0)
    assert not metrics["accepted"]


@pytest.mark.parametrize("bad", ["too_few", "duplicate", "non_finite", "line", "bool", "extra_key"])
def test_independent_checks_reject_invalid_evidence(bad):
    points = check_points()
    if bad == "too_few":
        points.pop()
    elif bad == "duplicate":
        points[-1] = points[0]
    elif bad == "non_finite":
        points[0]["pixel"][0] = float("nan")
    elif bad == "line":
        for point in points:
            point["table"][1] = point["table"][0]
    elif bad == "bool":
        points[0]["table"][0] = True
    else:
        points[0]["error_cm"] = 0
    with pytest.raises(ValueError):
        cal.check_points_metrics(H, points)


@pytest.fixture
def live_cli(monkeypatch, tmp_path):
    class Camera:
        is_fake = False
        released = False

        def release(self):
            self.released = True

    camera = Camera()
    markers, dets = marker_data([0.0] * 4)
    cfg = {"calib_markers": markers}
    monkeypatch.setattr(cal, "load_scene_cfg", lambda: cfg)
    monkeypatch.setattr(cal, "_collect_live", lambda *_: dets)
    monkeypatch.setattr(cal, "fit_homography", lambda *_: H.copy())
    monkeypatch.setattr(cameras, "get_camera", lambda _: camera)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: False)
    path = tmp_path / "homography.npy"
    np.save(path, H)
    return path, camera, markers, dets


@pytest.mark.parametrize("mode", ["--live", "--check"])
@pytest.mark.parametrize("errors", [[1.2] * 4, [0, 0, 0, 1.6]])
def test_cli_rejects_bad_mean_or_worst_error_without_overwriting(live_cli, monkeypatch, capsys, mode, errors):
    path, camera, _, _ = live_cli
    markers, dets = marker_data(errors)
    monkeypatch.setattr(cal, "_collect_live", lambda *_: dets)
    monkeypatch.setattr(cal, "load_scene_cfg", lambda: {"calib_markers": markers})
    before = path.read_bytes()
    assert cal.main([mode, "--out", str(path)]) == 1
    assert path.read_bytes() == before and camera.released
    output = capsys.readouterr()
    assert "mean" in output.out and "max" in output.out and "FAIL" in output.out
    assert "unchanged" in output.err


def test_cli_refuses_fewer_than_four_known_markers(live_cli, monkeypatch):
    path, camera, _, dets = live_cli
    monkeypatch.setattr(cal, "_collect_live", lambda *_: dets[:3])
    before = path.read_bytes()
    assert cal.main(["--live", "--out", str(path)]) == 1
    assert path.read_bytes() == before and camera.released


def test_live_fit_without_independent_checks_does_not_claim_measured_accuracy(live_cli, capsys):
    path, camera, _, _ = live_cli
    assert cal.main(["--live", "--out", str(path)]) == 0
    assert camera.released and np.array_equal(np.load(path), H)
    assert "independent table accuracy remains unverified" in capsys.readouterr().out


def test_live_no_save_preserves_previous_calibration(live_cli):
    path, _, _, _ = live_cli
    old = H.copy()
    old[0, 2] = 0.15
    np.save(path, old)
    before = path.read_bytes()
    assert cal.main(["--live", "--no-save", "--out", str(path)]) == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("accepted", [False, True])
def test_live_check_points_gate_the_save(live_cli, tmp_path, accepted, capsys):
    path, _, _, _ = live_cli
    old = H.copy()
    old[0, 2] = 0.15
    np.save(path, old)
    before = path.read_bytes()
    points = check_points()
    if not accepted:
        points[-1]["table"][0] += 0.02
    evidence = tmp_path / "points.json"
    evidence.write_text(json.dumps(points), encoding="utf-8")
    assert cal.main(["--live", "--out", str(path), "--check-points", str(evidence)]) == (0 if accepted else 1)
    assert "independent measured check points" in capsys.readouterr().out
    if accepted:
        assert np.array_equal(np.load(path), H)
    else:
        assert path.read_bytes() == before


def test_recorded_check_points_can_check_a_saved_matrix_without_opening_a_camera(tmp_path, monkeypatch):
    path = tmp_path / "homography.npy"
    cal.save_homography(H, path)
    evidence = tmp_path / "points.json"
    evidence.write_text(json.dumps(check_points()), encoding="utf-8")

    def no_camera(*_):
        raise AssertionError("offline check must not open a camera")

    monkeypatch.setattr(cameras, "get_camera", no_camera)
    monkeypatch.setattr(cal, "load_scene_cfg", no_camera)
    before = path.read_bytes()
    assert cal.main(["--check", "--out", str(path), "--check-points", str(evidence)]) == 0
    assert path.read_bytes() == before


def test_live_mode_cannot_silently_save_a_fake_camera_calibration(live_cli):
    path, camera, _, _ = live_cli
    camera.is_fake = True
    before = path.read_bytes()
    assert cal.main(["--live", "--out", str(path)]) == 1
    assert path.read_bytes() == before and camera.released


def test_failed_camera_collection_releases_camera_and_preserves_file(live_cli, monkeypatch):
    path, camera, _, _ = live_cli

    def failed(*_):
        raise RuntimeError("camera read failed")

    monkeypatch.setattr(cal, "_collect_live", failed)
    before = path.read_bytes()
    assert cal.main(["--live", "--out", str(path)]) == 1
    assert path.read_bytes() == before and camera.released


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_atomic_save_keeps_previous_calibration_on_failure(tmp_path, monkeypatch, failure):
    path = tmp_path / "homography.npy"
    cal.save_homography(H, path)
    before = path.read_bytes()
    if failure == "write":

        def broken_save(stream, *_args, **_kwargs):
            stream.write(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(cal.np, "save", broken_save)
    else:

        def broken_replace(*_):
            raise PermissionError("destination locked")

        monkeypatch.setattr(cal.os, "replace", broken_replace)
    with pytest.raises(OSError):
        cal.save_homography(H * 2, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2)])
def test_invalid_homography_never_replaces_a_good_one(tmp_path, matrix):
    path = tmp_path / "homography.npy"
    cal.save_homography(H, path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        cal.save_homography(matrix, path)
    assert path.read_bytes() == before


def calibration_endpoint(monkeypatch, errors, *, frame=True):
    markers, dets = marker_data(errors)
    instance = object.__new__(orchestrator.Orchestrator)
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)
    instance.tracker = SimpleNamespace(last_frame=lambda: object() if frame else None)
    monkeypatch.setattr(orchestrator.calibrate, "load_homography", lambda: H)
    monkeypatch.setattr(orchestrator.calibrate, "calib_marker_centers", lambda: markers)
    monkeypatch.setattr(orchestrator.markers, "detect", lambda _: dets)
    return instance


@pytest.mark.parametrize("errors", [[1.2] * 4, [0, 0, 0, 1.6]])
def test_dashboard_calibration_rejects_bad_mean_or_worst_error(monkeypatch, errors):
    instance = calibration_endpoint(monkeypatch, errors)
    with pytest.raises(HTTPException) as exc:
        instance.calibrate()
    assert exc.value.status_code == 400
    assert "mean" in exc.value.detail and "max" in exc.value.detail


def test_dashboard_calibration_requires_four_markers(monkeypatch):
    instance = calibration_endpoint(monkeypatch, [0.0] * 3)
    with pytest.raises(HTTPException) as exc:
        instance.calibrate()
    assert exc.value.status_code == 503 and "need 4" in exc.value.detail


@pytest.mark.parametrize("frame_available", [False, True])
def test_dashboard_releases_its_temporary_camera_and_preserves_success_contract(monkeypatch, frame_available):
    instance = calibration_endpoint(monkeypatch, [0.5] * 4, frame=False)
    released = []
    camera = SimpleNamespace(
        read=lambda: object() if frame_available else None, release=lambda: released.append(True)
    )
    monkeypatch.setattr(orchestrator.cameras, "get_camera", lambda _: camera)
    if frame_available:
        assert instance.calibrate() == {"error_cm": 0.5}
    else:
        with pytest.raises(HTTPException) as exc:
            instance.calibrate()
        assert exc.value.status_code == 503
    assert released == [True]
