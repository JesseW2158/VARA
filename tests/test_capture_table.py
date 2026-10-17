"""Marker capture evidence and cleanup, using generated images and camera stubs only."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from scripts import capture_table
from vision import markers


class StubCamera:
    is_fake = False
    index = 2

    def __init__(self, frame):
        self.frame = frame
        self.released = False

    def read(self):
        return self.frame

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    def refuse(_):
        pytest.fail("unmocked physical camera access")

    monkeypatch.setattr(capture_table.cameras, "get_camera", refuse)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_: False)
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_CAMERA", raising=False)


@pytest.fixture
def camera(monkeypatch):
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[100:160, 200:260] = cv2.cvtColor(markers.generate(10, 60), cv2.COLOR_GRAY2BGR)
    camera = StubCamera(frame)
    monkeypatch.setattr(capture_table.cameras, "get_camera", lambda _: camera)
    return camera


def test_physical_capture_preserves_image_and_reports_only_observed_pixels(camera, tmp_path):
    out = tmp_path / "spot-1"
    report = capture_table.capture(out)
    assert report == json.loads((out / "markers.json").read_text(encoding="utf-8"))
    assert report["mode"] == "physical" and report["camera_index"] == 2
    assert report["markers"] == [{"id": 10, "pixel": pytest.approx([229.5, 129.5], abs=0.1)}]
    assert np.array_equal(cv2.imread(str(out / report["original_image"])), camera.frame)
    assert not np.array_equal(cv2.imread(str(out / report["annotated_image"])), camera.frame)
    assert camera.released


@pytest.mark.parametrize("flag", ["FAKE", "FAKE_CAMERA"])
def test_implicit_fake_cannot_create_physical_evidence(monkeypatch, tmp_path, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="unset FAKE"):
        capture_table.capture(tmp_path / "refused")
    assert list(tmp_path.iterdir()) == []


def test_unexpected_fake_camera_is_closed_and_refused(camera, tmp_path):
    camera.is_fake = True
    with pytest.raises(ValueError, match="synthetic or unidentified"):
        capture_table.capture(tmp_path / "refused")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_explicit_fake_capture_is_labeled_even_without_fake_environment(tmp_path):
    report = capture_table.capture(tmp_path / "offline", fake=True)
    assert report["mode"] == "synthetic"
    assert report["original_image"].startswith("synthetic-")
    assert report["annotated_image"].startswith("synthetic-")
    assert report["markers"]


def test_existing_directory_is_never_overwritten_or_camera_opened(tmp_path):
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "markers.json").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        capture_table.capture(existing)
    assert (existing / "markers.json").read_text(encoding="utf-8") == "original"


def test_failed_read_releases_camera_and_does_not_create_evidence(camera, tmp_path, capsys):
    camera.frame = None
    assert capture_table.main(["--out", str(tmp_path / "failed")]) == 2
    assert camera.released and list(tmp_path.iterdir()) == []
    assert "returned no frame" in capsys.readouterr().err


def test_detection_failure_still_releases_camera(camera, monkeypatch, tmp_path):
    def fail(_):
        raise RuntimeError("detector failed")

    monkeypatch.setattr(capture_table.markers, "detect", fail)
    with pytest.raises(RuntimeError, match="detector failed"):
        capture_table.capture(tmp_path / "failed")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_encoding_failure_does_not_create_evidence(camera, monkeypatch, tmp_path):
    monkeypatch.setattr(capture_table.cv2, "imencode", lambda *_: (False, None))
    with pytest.raises(OSError, match="encode"):
        capture_table.capture(tmp_path / "failed")
    assert camera.released and list(tmp_path.iterdir()) == []


def test_missing_markers_keeps_image_but_cli_returns_unsuccessful(camera, tmp_path, capsys):
    camera.frame[:] = 255
    out = tmp_path / "empty-view"
    assert capture_table.main(["--out", str(out)]) == 1
    assert json.loads((out / "markers.json").read_text(encoding="utf-8"))["markers"] == []
    assert camera.released
    assert "No marker centers available" in capsys.readouterr().out


def test_success_cli_prints_id_and_pixel(camera, tmp_path, capsys):
    assert capture_table.main(["--out", str(tmp_path / "spot-1")]) == 0
    output = capsys.readouterr().out
    assert "Saved physical frame" in output and "ID 10: pixel" in output
