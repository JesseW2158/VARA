"""B1 check success requires all requested feeds and an uninterrupted full duration."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vision import cameras


class CheckCamera:
    def __init__(self, name, clock):
        self.name, self.clock = name, clock
        self.frames = self.failures = self.late = 0
        self.fps = 30.0
        self.released = False
        self.read_error = None
        self.release_error = None
        self.return_frame = True
        self.step = 1 / 60

    def read(self):
        if self.read_error:
            raise self.read_error
        self.clock.now += self.step
        self.frames += 1
        return object() if self.return_frame else None

    def release(self):
        self.released = True
        if self.release_error:
            raise self.release_error

    def stats(self):
        return {name: getattr(self, name) for name in ("frames", "fps", "failures", "late")}


@pytest.fixture
def feeds(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cameras.time, "monotonic", lambda: clock.now)
    opened = {name: CheckCamera(name, clock) for name in ("overhead", "wrist")}
    monkeypatch.setattr(cameras, "get_camera", opened.__getitem__)
    return opened


def test_full_duration_with_both_feeds_is_required_for_success(feeds):
    assert cameras._check(0.1, ["overhead", "wrist"]) == 0
    assert all(cam.frames >= 3 and cam.released for cam in feeds.values())
    assert feeds["overhead"].clock.now >= 0.1


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_bad_duration_rejected_before_opening_any_feed(monkeypatch, duration):
    attempted = []
    monkeypatch.setattr(cameras, "get_camera", lambda name: attempted.append(name))
    assert cameras._check(duration, ["overhead", "wrist"]) == 2
    assert not attempted


@pytest.mark.parametrize("failure", [RuntimeError("not connected"), KeyboardInterrupt()])
def test_second_open_failure_closes_first_and_cannot_pass(feeds, monkeypatch, failure):
    def get(name):
        if name == "wrist":
            raise failure
        return feeds[name]

    monkeypatch.setattr(cameras, "get_camera", get)
    assert cameras._check(1800, ["overhead", "wrist"]) == 1
    assert feeds["overhead"].released and feeds["overhead"].frames == 0


@pytest.mark.parametrize("failure", [RuntimeError("capture failed"), KeyboardInterrupt()])
def test_interrupted_reads_fail_and_close_every_feed(feeds, failure):
    feeds["wrist"].read_error = failure
    assert cameras._check(1800, ["overhead", "wrist"]) == 1
    assert all(cam.released for cam in feeds.values())


def test_empty_frame_cannot_pass_even_if_capture_statistics_are_clean(feeds):
    feeds["wrist"].return_frame = False
    assert cameras._check(0.1, ["overhead", "wrist"]) == 1
    assert all(cam.released for cam in feeds.values())


def test_single_blocked_frame_is_insufficient_for_a_frame_rate_check(feeds):
    feeds["overhead"].step = 1800
    assert cameras._check(1800, ["overhead"]) == 1
    assert feeds["overhead"].frames == 1 and feeds["overhead"].released


@pytest.mark.parametrize("failure", [RuntimeError("release failed"), KeyboardInterrupt()])
def test_cleanup_failure_still_releases_remaining_feeds(feeds, failure):
    feeds["overhead"].release_error = failure
    assert cameras._check(0.1, ["overhead", "wrist"]) == 1
    assert all(cam.released for cam in feeds.values())


@pytest.mark.parametrize("failure", [RuntimeError("camera setup failed"), KeyboardInterrupt()])
def test_capture_is_released_if_open_configuration_fails(monkeypatch, failure):
    released = []

    def setting(*args):
        raise failure

    capture = SimpleNamespace(isOpened=lambda: True, set=setting, release=lambda: released.append(True))
    monkeypatch.setattr(cameras.cv2, "VideoCapture", lambda *args: capture)
    with pytest.raises(type(failure)):
        cameras.open_camera(2)
    assert released == [True]


class NegotiatingCapture:
    """Model a backend which drops MJPG while renegotiating its size or frame rate."""

    def __init__(self, *, refuse_format=False, format_report=None):
        self.refuse_format = refuse_format
        self.format_report = format_report
        self.settings = {cameras.cv2.CAP_PROP_FOURCC: cameras.cv2.VideoWriter_fourcc(*"YUY2")}
        self.calls = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop == cameras.cv2.CAP_PROP_FOURCC and self.refuse_format:
            return False
        self.settings[prop] = value
        if prop in (
            cameras.cv2.CAP_PROP_FRAME_WIDTH,
            cameras.cv2.CAP_PROP_FRAME_HEIGHT,
            cameras.cv2.CAP_PROP_FPS,
        ):
            self.settings[cameras.cv2.CAP_PROP_FOURCC] = cameras.cv2.VideoWriter_fourcc(*"YUY2")
        return True

    def get(self, prop):
        if prop == cameras.cv2.CAP_PROP_FOURCC and self.format_report is not None:
            return self.format_report
        return self.settings[prop]

    def release(self):
        self.released = True


def test_format_survives_backend_size_and_fps_renegotiation(monkeypatch, caplog):
    capture = NegotiatingCapture()
    monkeypatch.setattr(cameras.cv2, "VideoCapture", lambda *args: capture)
    with caplog.at_level(logging.INFO, logger="vision.cameras"):
        assert cameras.open_camera(2) is capture
    assert capture.get(cameras.cv2.CAP_PROP_FOURCC) == cameras.cv2.VideoWriter_fourcc(*"MJPG")
    assert "driver reports" in caplog.text and "(MJPG)" in caplog.text
    assert "unverified" not in caplog.text and not capture.released


def test_refused_format_is_reported_without_claiming_compressed_stream(monkeypatch, caplog):
    capture = NegotiatingCapture(refuse_format=True)
    monkeypatch.setattr(cameras.cv2, "VideoCapture", lambda *args: capture)
    assert cameras.open_camera(2) is capture
    assert "refused" in caplog.text and "driver reports YUY2" in caplog.text
    assert "compressed USB bandwidth is unverified" in caplog.text


@pytest.mark.parametrize("reported", [0, -1, float("nan")])
def test_missing_format_readback_is_reported_as_unverified(monkeypatch, caplog, reported):
    capture = NegotiatingCapture(format_report=reported)
    monkeypatch.setattr(cameras.cv2, "VideoCapture", lambda *args: capture)
    assert cameras.open_camera(2) is capture
    assert "driver reports unavailable" in caplog.text


@pytest.mark.parametrize(("platform", "manual"), [("nt", 0), ("posix", 1)])
def test_manual_exposure_uses_backend_value_without_changing_exposure(monkeypatch, platform, manual):
    capture = NegotiatingCapture()
    monkeypatch.setattr(cameras, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(cameras.cv2, "VideoCapture", lambda *args: capture)
    cameras.open_camera(2)
    assert (cameras.cv2.CAP_PROP_AUTO_EXPOSURE, manual) in capture.calls
    assert all(prop != cameras.cv2.CAP_PROP_EXPOSURE for prop, _ in capture.calls)
