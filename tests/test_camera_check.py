"""B1 check success requires all requested feeds and an uninterrupted full duration."""

from __future__ import annotations

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
