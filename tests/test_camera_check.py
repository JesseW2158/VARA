"""B1 check success requires all requested feeds and an uninterrupted full duration; the report file
records the per-feed timing and which criterion (strict or tracking) produced the verdict."""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest

from vision import cameras


class CheckCamera:
    """A feed on a mocked clock: every read advances the clock by `step`, or by `stalls[n]` on read n."""

    def __init__(self, name, clock):
        self.name, self.clock = name, clock
        self.frames = self.failures = self.late = 0
        self.fps = 30.0
        self.released = False
        self.read_error = None
        self.release_error = None
        self.return_frame = True
        self.step = 1 / 60
        self.stalls: dict[int, float] = {}

    def read(self):
        if self.read_error:
            raise self.read_error
        self.clock.now += self.stalls.get(self.frames + 1, self.step)
        self.frames += 1
        return object() if self.return_frame else None

    def release(self):
        self.released = True
        if self.release_error:
            raise self.release_error

    def stats(self):
        return {name: getattr(self, name) for name in ("frames", "fps", "failures", "late")}


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    """Default report files go under VARA_RUNS_DIR/setup, never into the repo's runs/."""
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture
def feeds(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cameras.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cameras.time, "perf_counter", lambda: clock.now)  # the checker's own stopwatch
    opened = {name: CheckCamera(name, clock) for name in ("overhead", "wrist")}
    monkeypatch.setattr(cameras, "get_camera", opened.__getitem__)
    return opened


def _reports(runs_dir, stem="camera-check"):
    """The run's report files, oldest first (a same-second run gets a `-2` suffix)."""
    return sorted(
        (runs_dir / "setup").glob(f"{stem}-*.json"), key=lambda p: (p.stat().st_mtime_ns, len(p.name))
    )


def _feed(report, name):
    return next(feed for feed in report["feeds"] if feed["name"] == name)


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


# -- the report file and the two criteria -----------------------------------------------------------------


def test_default_report_is_a_fresh_file_under_runs_setup_with_the_verdict(feeds, runs_dir, capsys):
    assert cameras._check(0.1, ["overhead", "wrist"]) == 0
    (path,) = _reports(runs_dir)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["tool"].endswith("--check") and report["independent"] is False
    assert report["criterion"] == "strict" and "2.5 frame periods" in report["criterion_rule"]
    assert report["passed"] is True and report["exit_code"] == 0 and report["completed"] is True
    assert report["requested_s"] == 0.1 and report["elapsed_s"] >= 0.1
    assert report["requested_feeds"] == ["overhead", "wrist"] and report["problems"] == []
    assert report["late_factor"] == 2.5 and report["long_gap_s"] == 0.5 and report["tracking_min_fps"] == 25
    for name in ("overhead", "wrist"):
        feed = _feed(report, name)
        assert feed["opened"] and feed["passed"] and feed["reasons"] == []
        assert feed["frames"] >= 3 and feed["failures"] == 0 and feed["late"] == 0 and feed["long"] == 0
        assert feed["fps"] > 0 and feed["max_read_s"] > 0 and feed["max_interframe_s"] > 0
        assert feed["index"] is None and feed["fourcc_reported"] == "unavailable"  # a stub has no driver
        assert feed["target_fps"] == 30.0 and feed["elapsed_s"] >= 0.1 and feed["completed"]
    out = capsys.readouterr().out
    assert "verdict: PASS under strict" in out and f"report: {path}" in out
    # a second run in the same second still gets its own file
    assert cameras._check(0.1, ["overhead", "wrist"]) == 0
    assert len(_reports(runs_dir)) == 2


def test_existing_out_path_is_refused_before_any_feed_opens(tmp_path, monkeypatch):
    out = tmp_path / "soak.json"
    out.write_text("keep me", encoding="utf-8")
    attempted = []
    monkeypatch.setattr(cameras, "get_camera", lambda name: attempted.append(name))
    assert cameras._check(1800, ["overhead", "wrist"], out=out) == 2
    assert not attempted and out.read_text(encoding="utf-8") == "keep me"


def test_explicit_out_path_gets_the_report_and_parent_directories(feeds, tmp_path):
    out = tmp_path / "evidence" / "soak.json"
    assert cameras._check(0.1, ["overhead", "wrist"], out=out) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["exit_code"] == 0 and [f["name"] for f in report["feeds"]] == ["overhead", "wrist"]


def test_one_stalled_read_is_late_under_strict_but_fine_for_tracking(feeds, runs_dir):
    feeds["overhead"].stalls = {5: 0.2}  # longer than 2.5 periods (0.083 s), shorter than 0.5 s
    assert cameras._check(3.0, ["overhead", "wrist"]) == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    overhead, wrist = _feed(report, "overhead"), _feed(report, "wrist")
    assert overhead["late"] == 1 and overhead["long"] == 0 and overhead["failures"] == 0
    assert overhead["max_read_s"] == pytest.approx(0.2, abs=1e-3)
    assert overhead["max_interframe_s"] == pytest.approx(0.2 + 1 / 60, abs=1e-3)
    # one loop reads both feeds in turn, so the wrist's next read waited behind the stall as well;
    # max_read_s is what names the feed whose read stalled
    assert wrist["late"] == 1 and wrist["max_read_s"] == pytest.approx(1 / 60, abs=1e-3)
    assert not overhead["passed"] and "1 late gaps" in overhead["reasons"][0]
    assert report["passed"] is False and report["exit_code"] == 1 and report["completed"] is True
    tracking = cameras.apply_criterion(report, "tracking")
    assert tracking["passed"] is True and tracking["exit_code"] == 0 and tracking["criterion"] == "tracking"
    assert all(feed["reasons"] == [] for feed in tracking["feeds"])
    assert _feed(tracking, "overhead")["fps"] >= 25
    # the same kind of run judged by the tracking criterion from the start
    feeds["overhead"].stalls = {feeds["overhead"].frames + 5: 0.2}
    out = runs_dir / "tracking.json"
    assert cameras._check(3.0, ["overhead", "wrist"], criterion="tracking", out=out) == 0
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["criterion"] == "tracking" and "seen timeout" in again["criterion_rule"]
    assert _feed(again, "overhead")["late"] == 1 and again["passed"] is True


def test_a_gap_longer_than_the_seen_timeout_fails_both_criteria(feeds, runs_dir):
    feeds["wrist"].stalls = {4: 0.6}
    assert cameras._check(3.0, ["overhead", "wrist"], criterion="tracking") == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    wrist = _feed(report, "wrist")
    assert wrist["long"] == 1 and wrist["late"] == 1 and wrist["max_read_s"] == pytest.approx(0.6, abs=1e-3)
    assert any("long gaps" in reason and "0.5 s" in reason for reason in wrist["reasons"])
    assert cameras.apply_criterion(report, "strict")["exit_code"] == 1


def test_a_slow_feed_fails_tracking_on_frame_rate(feeds, runs_dir):
    for cam in feeds.values():
        cam.step = 1 / 20  # two reads per loop: each feed sees one frame every 0.1 s
    assert cameras._check(2.0, ["overhead", "wrist"], criterion="tracking") == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    for feed in report["feeds"]:
        assert feed["fps"] == pytest.approx(10.0, abs=0.2) and feed["long"] == 0 and feed["late"] > 0
        assert any(reason.startswith("effective fps") and "below 25" in reason for reason in feed["reasons"])


def test_failed_open_is_recorded_per_feed_in_the_report(feeds, monkeypatch, runs_dir):
    def get(name):
        if name == "wrist":
            raise RuntimeError("not connected")
        return feeds[name]

    monkeypatch.setattr(cameras, "get_camera", get)
    assert cameras._check(1800, ["overhead", "wrist"]) == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    wrist = _feed(report, "wrist")
    assert (
        wrist["opened"] is False and "not connected" in wrist["error"] and "did not open" in wrist["reasons"]
    )
    assert _feed(report, "overhead")["frames"] == 0 and report["completed"] is False
    assert any("wrist: cannot open" in error for error in report["errors"])


def test_interrupted_check_still_writes_an_incomplete_report(feeds, runs_dir):
    feeds["wrist"].read_error = KeyboardInterrupt()
    assert cameras._check(1800, ["overhead", "wrist"]) == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    assert report["completed"] is False and report["exit_code"] == 1
    assert "interrupted before completion" in report["errors"]
    assert "the check did not complete" in report["problems"]


def test_a_read_that_raises_is_named_on_its_own_feed(feeds, runs_dir, capsys):
    feeds["wrist"].read_error = RuntimeError("capture failed")
    assert cameras._check(1800, ["overhead", "wrist"]) == 1
    report = json.loads(_reports(runs_dir)[-1].read_text(encoding="utf-8"))
    wrist, overhead = _feed(report, "wrist"), _feed(report, "overhead")
    assert wrist["error"] == "RuntimeError: capture failed" and wrist["passed"] is False
    assert "error: RuntimeError: capture failed" in wrist["reasons"]
    assert overhead["error"] is None and "did not run for the full duration" in overhead["reasons"]
    assert report["completed"] is False and any("capture failed" in error for error in report["errors"])
    assert all(cam.released for cam in feeds.values())
    wrist_line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("wrist:"))
    assert "FAIL:" in wrist_line and "error: RuntimeError: capture failed" in wrist_line


def test_unknown_criterion_is_misuse(feeds):
    assert cameras._check(0.1, ["overhead"], criterion="lenient") == 2
    with pytest.raises(ValueError):
        cameras.check_feeds(0.1, ["overhead"], criterion="lenient")
    with pytest.raises(ValueError):
        cameras.evaluate_feed({}, "lenient")


@pytest.mark.parametrize("duration", [True, False])
def test_a_bool_is_not_a_duration(monkeypatch, duration):
    attempted = []
    monkeypatch.setattr(cameras, "get_camera", lambda name: attempted.append(name))
    assert cameras._check(duration, ["overhead"]) == 2
    with pytest.raises(ValueError):
        cameras.check_feeds(duration, ["overhead"])
    assert not attempted


# -- independent readers on per-thread mocked clocks --------------------------------------------------------


class ThreadClock:
    """A mocked clock per thread: each independent reader advances only its own, so the readers stay
    deterministic; the main thread's clock never moves (its deadline is never reached)."""

    def __init__(self):
        self._now: dict[str, float] = {}

    @property
    def now(self):
        return self._now.get(threading.current_thread().name, 0.0)

    @now.setter
    def now(self, value):
        self._now[threading.current_thread().name] = value


@pytest.fixture
def thread_feeds(monkeypatch):
    clock = ThreadClock()
    monkeypatch.setattr(cameras.time, "perf_counter", lambda: clock.now)
    opened = {name: CheckCamera(name, clock) for name in ("overhead", "wrist")}
    for cam in opened.values():
        cam.step = 1 / 30
    monkeypatch.setattr(cameras, "get_camera", opened.__getitem__)
    return opened


def test_independent_readers_fail_only_the_feed_that_stalled(thread_feeds, runs_dir):
    thread_feeds["overhead"].stalls = {5: 0.2}  # longer than 2.5 periods (0.083 s), shorter than 0.5 s
    out = runs_dir / "diagnose.json"
    assert cameras._check(3.0, ["overhead", "wrist"], independent=True, out=out) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    overhead, wrist = _feed(report, "overhead"), _feed(report, "wrist")
    assert report["independent"] is True and report["completed"] is True and report["exit_code"] == 1
    assert overhead["late"] == 1 and overhead["long"] == 0 and overhead["passed"] is False
    assert overhead["max_read_s"] == pytest.approx(0.2, abs=1e-6)
    assert overhead["max_interframe_s"] == pytest.approx(0.2, abs=1e-6)
    # its own thread read the wrist: the overhead stall did not delay it
    assert wrist["late"] == 0 and wrist["passed"] is True and wrist["reasons"] == []
    assert wrist["max_interframe_s"] == pytest.approx(1 / 30, abs=1e-4)
    assert wrist["fps"] == pytest.approx(30.0)
    assert wrist["completed"] and wrist["elapsed_s"] >= 3.0 and report["elapsed_s"] >= 3.0
    assert cameras.apply_criterion(report, "tracking")["exit_code"] == 0
    assert all(cam.released for cam in thread_feeds.values())


def test_independent_long_gap_fails_tracking_for_that_feed_only(thread_feeds, runs_dir):
    thread_feeds["wrist"].stalls = {7: 0.6}  # longer than the tracker's 0.5 s seen timeout
    out = runs_dir / "tracking.json"
    assert cameras._check(3.0, ["overhead", "wrist"], independent=True, criterion="tracking", out=out) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    wrist, overhead = _feed(report, "wrist"), _feed(report, "overhead")
    assert wrist["long"] == 1 and wrist["late"] == 1 and wrist["passed"] is False
    assert any("long gaps" in reason for reason in wrist["reasons"])
    assert overhead["long"] == 0 and overhead["late"] == 0 and overhead["passed"] is True
    assert report["criterion"] == "tracking" and report["passed"] is False
