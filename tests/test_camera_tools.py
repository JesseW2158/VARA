"""The camera tools: --diagnose (independent readers), --list (index enumeration), the driver read-back,
synthetic labelling under FAKE_CAMERA, exit codes. No test opens a device: cv2.VideoCapture is either
forbidden or replaced by a stub."""

from __future__ import annotations

import json
import os
import re
import threading
import time

import dotenv
import numpy as np
import pytest

from vision import cameras, scene


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """No fake flags inherited, evidence under tmp_path, configured indexes known, and `main()` never loads
    the real .env into the test process (its values would outlive the test)."""
    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OVERHEAD_CAM", "2")
    monkeypatch.setenv("WRIST_CAM", "1")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    return tmp_path


@pytest.fixture
def no_devices(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in this test")

    monkeypatch.setattr(cameras.cv2, "VideoCapture", boom)


# -- constants and the read-back ---------------------------------------------------------------------------


def test_long_gap_is_the_tracker_seen_timeout():
    assert cameras.LONG_GAP_S == scene.SEEN_TIMEOUT_S == 0.5
    assert cameras.LATE_FACTOR == 2.5 and cameras.TRACKING_MIN_FPS == 25.0
    assert cameras.CRITERIA == ("strict", "tracking")


@pytest.mark.parametrize("code", [0, -1, float("nan"), float("inf"), None, "junk"])
def test_fourcc_name_reports_unavailable_for_a_missing_code(code):
    assert cameras.fourcc_name(code) == "unavailable"


def test_fourcc_name_decodes_the_driver_value():
    assert cameras.fourcc_name(cameras.cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert cameras.fourcc_name(float(cameras.cv2.VideoWriter_fourcc(*"YUY2"))) == "YUY2"


class _Capture:
    """A cv2.VideoCapture stand-in: opened or not, returns `frames` frames then fails."""

    def __init__(self, index, opened=True, frames=1, props=None):
        self.index = index
        self.opened = opened
        self.left = frames
        self.props = (
            props
            if props is not None
            else {
                cameras.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cameras.cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                cameras.cv2.CAP_PROP_FPS: 30.00003,
                cameras.cv2.CAP_PROP_FOURCC: float(cameras.cv2.VideoWriter_fourcc(*"MJPG")),
            }
        )
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        self.reads += 1
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.full((480, 640, 3), 90, np.uint8)

    def release(self):
        self.released = True


def test_read_back_decodes_the_driver_properties_and_tolerates_a_bare_stub():
    got = cameras.read_back(_Capture(1))
    assert got == {"width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"}
    assert cameras.read_back(object()) == cameras.UNREPORTED
    assert cameras.read_back(_Capture(1, props={}))["fourcc"] == "unavailable"
    assert cameras.read_back(_Capture(1, props={}))["width"] is None


def test_camera_keeps_the_read_back_and_its_target_rate(no_devices):
    cam = cameras.Camera(2, name="overhead", fps=30, capture=_Capture(2))
    assert cam.reported == {"width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"}
    assert cam.target_fps == 30.0
    fake = cameras.FakeCamera(name="wrist", fps=30.0)
    assert fake.reported == {"width": 640, "height": 480, "fps": 30.0, "fourcc": None}
    assert fake.target_fps == 30.0


# -- --check and --diagnose on the fakes -----------------------------------------------------------------


def test_check_on_the_fakes_is_labelled_synthetic_and_records_the_read_back(no_devices, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    report = cameras.check_feeds(0.3, ["overhead", "wrist"], progress=False)
    assert report["fake"] is True and report["mode"] == "synthetic"
    assert report["completed"] and report["elapsed_s"] >= 0.3 and report["exit_code"] in (0, 1)
    for feed in report["feeds"]:
        assert feed["fake"] is True and feed["index"] is None and feed["opened"]
        assert feed["width_reported"] == 640 and feed["height_reported"] == 480
        assert feed["fps_reported"] == 30.0 and feed["fourcc_reported"] is None
        assert feed["frames"] >= 5 and feed["failures"] == 0 and feed["target_fps"] == 30.0
        assert feed["max_read_s"] > 0 and feed["max_interframe_s"] > 0
    cameras._print_summary(report)
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "driver read-back 640x480 30 fps synthetic" in out


class _StallingFake(cameras.FakeCamera):
    """A paced fake whose read number `at` blocks for `stall` seconds, as a camera that hiccups."""

    def __init__(self, *args, at: int, stall: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.at, self.stall = at, stall

    def read(self):
        if self.frames + 1 == self.at:
            time.sleep(self.stall)
        return super().read()


def test_diagnose_names_only_the_feed_that_stalled(no_devices, monkeypatch):
    fakes = {
        "overhead": _StallingFake(name="overhead", fps=30.0, at=10, stall=0.15),
        "wrist": cameras.FakeCamera(name="wrist", fps=30.0),
    }
    monkeypatch.setattr(cameras, "get_camera", fakes.__getitem__)
    report = cameras.check_feeds(2.0, ["overhead", "wrist"], independent=True, progress=False)
    assert report["independent"] is True and report["completed"] is True
    assert report["elapsed_s"] >= 2.0 and report["tool"].endswith("--diagnose")
    overhead, wrist = report["feeds"]
    assert overhead["name"] == "overhead" and wrist["name"] == "wrist"
    assert overhead["late"] >= 1 and overhead["long"] == 0 and overhead["max_read_s"] >= 0.15
    assert overhead["passed"] is False and "late gaps" in overhead["reasons"][0]
    assert wrist["late"] == 0 and wrist["passed"] is True and wrist["max_read_s"] < 0.15
    assert wrist["frames"] >= 50 and wrist["elapsed_s"] >= 2.0 and wrist["completed"]
    assert report["passed"] is False and report["exit_code"] == 1
    tracking = cameras.apply_criterion(report, "tracking")
    assert tracking["passed"] is True and tracking["exit_code"] == 0
    assert all(cam._released for cam in fakes.values())


def test_diagnose_survives_a_reader_that_raises_and_releases_both(no_devices, monkeypatch):
    class Dead(cameras.FakeCamera):
        def read(self):
            raise RuntimeError("unplugged")

    fakes = {"overhead": cameras.FakeCamera(name="overhead", fps=30.0), "wrist": Dead(name="wrist", fps=30.0)}
    monkeypatch.setattr(cameras, "get_camera", fakes.__getitem__)
    report = cameras.check_feeds(0.3, ["overhead", "wrist"], independent=True, progress=False)
    assert report["completed"] is False and report["exit_code"] == 1
    overhead, wrist = report["feeds"]
    assert overhead["completed"] and overhead["frames"] >= 3
    assert wrist["error"] == "RuntimeError: unplugged" and not wrist["completed"]
    assert any(reason.startswith("error:") for reason in wrist["reasons"])
    assert all(cam._released for cam in fakes.values())


def test_diagnose_cli_writes_the_report_on_the_fakes(no_devices, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    out = tmp_path / "evidence" / "diagnose.json"
    assert cameras.main(["--diagnose", "0.4", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["independent"] is True and report["fake"] is True and report["exit_code"] == 0
    assert report["criterion"] == "strict" and [f["name"] for f in report["feeds"]] == ["overhead", "wrist"]
    assert "one thread per feed" in report["scope"]
    assert "verdict: PASS under strict" in capsys.readouterr().out


def test_check_cli_records_the_tracking_criterion(no_devices, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    out = tmp_path / "check.json"
    assert (
        cameras.main(["--check", "0.3", "--criterion", "tracking", "--camera", "wrist", "--out", str(out)])
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["criterion"] == "tracking" and "effective fps >= 25" in report["criterion_rule"]
    assert [f["name"] for f in report["feeds"]] == ["wrist"] and report["feeds"][0]["fps"] >= 25


def test_default_diagnose_report_lands_under_runs_setup(no_devices, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    assert cameras.main(["--diagnose", "0.3"]) == 0
    (path,) = (tmp_path / "runs" / "setup").glob("camera-diagnose-*.json")
    assert json.loads(path.read_text(encoding="utf-8"))["exit_code"] == 0


def test_apply_criterion_rejudges_a_saved_report():
    saved = {
        "requested_feeds": ["overhead", "wrist"],
        "requested_s": 1800.0,
        "elapsed_s": 1800.2,
        "completed": True,
        "errors": [],
        "feeds": [
            {
                "name": "overhead",
                "opened": True,
                "frames": 53000,
                "failures": 0,
                "late": 58,
                "long": 0,
                "fps": 29.5,
                "completed": True,
                "error": None,
            },
            {
                "name": "wrist",
                "opened": True,
                "frames": 54000,
                "failures": 0,
                "late": 0,
                "long": 0,
                "fps": 29.9,
                "completed": True,
                "error": None,
            },
        ],
    }
    strict = cameras.apply_criterion(saved, "strict")
    assert strict["passed"] is False and strict["exit_code"] == 1
    assert strict["feeds"][0]["reasons"] == ["58 late gaps (> 2.5 frame periods)"]
    tracking = cameras.apply_criterion(saved, "tracking")
    assert tracking["passed"] is True and tracking["exit_code"] == 0 and tracking["problems"] == []
    assert "feeds" in saved and "reasons" not in saved["feeds"][0]  # the input is left alone
    short = cameras.apply_criterion({**saved, "elapsed_s": 193.8}, "tracking")
    assert short["passed"] is False and "shorter than requested" in short["problems"][0]
    missing = cameras.apply_criterion({**saved, "feeds": saved["feeds"][:1]}, "tracking")
    assert missing["passed"] is False and "wrist" in missing["problems"][0]


# -- --list ------------------------------------------------------------------------------------------------


@pytest.fixture
def captures(monkeypatch):
    """cv2.VideoCapture stub: index 1 opens (one frame), the rest do not; records backend arguments."""
    made = []

    def factory(index, *backend):
        cap = _Capture(index, opened=index == 1)
        cap.backend = backend
        made.append(cap)
        return cap

    monkeypatch.setattr(cameras.cv2, "VideoCapture", factory)
    return made


def test_list_opens_each_index_briefly_and_releases_every_capture(captures, tmp_path):
    out = tmp_path / "list"
    report = cameras.list_cameras(4, out)
    assert [cap.index for cap in captures] == [0, 1, 2, 3] and all(cap.released for cap in captures)
    expected_backend = (cameras.cv2.CAP_DSHOW,) if os.name == "nt" else ()
    assert all(cap.backend == expected_backend for cap in captures)
    assert [entry["opened"] for entry in report["cameras"]] == [False, True, False, False]
    one = report["cameras"][1]
    assert one["width"] == 640 and one["height"] == 480 and one["fps_reported"] == 30.0
    assert one["fourcc_reported"] == "MJPG" and one["frame"] == "camera-1.jpg" and one["error"] is None
    assert one["configured_as"] == ["wrist"] and report["cameras"][2]["configured_as"] == ["overhead"]
    assert (out / "camera-1.jpg").read_bytes()[:2] == b"\xff\xd8" and not (out / "camera-0.jpg").exists()
    assert report["configured"]["overhead"] == {"index": 2, "source": "OVERHEAD_CAM", "error": None}
    assert report["fake"] is False and report["mode"] == "physical" and report["opened"] == [1]
    assert report["exit_code"] == 0 and report["max_index"] == 4
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert saved["cameras"] == report["cameras"] and saved["directory"] == str(out)


def test_list_reports_an_opened_index_without_frames_and_a_raising_one(monkeypatch, tmp_path):
    class Raising(_Capture):
        def isOpened(self):
            raise RuntimeError("driver error")

    made = []

    def factory(index, *backend):
        cap = Raising(index) if index == 0 else _Capture(index, frames=0)
        made.append(cap)
        return cap

    monkeypatch.setattr(cameras.cv2, "VideoCapture", factory)
    report = cameras.list_cameras(2, tmp_path / "list")
    assert all(cap.released for cap in made)
    assert (
        report["cameras"][0]["error"] == "RuntimeError: driver error" and not report["cameras"][0]["opened"]
    )
    assert report["cameras"][1]["opened"] and report["cameras"][1]["frame"] is None
    assert report["cameras"][1]["error"] == "opened but returned no frame in 3 reads" and made[1].reads == 3
    assert report["exit_code"] == 0  # index 1 did open: the human can still tell which device it is


def test_list_cli_exits_1_when_nothing_opens(monkeypatch, tmp_path, capsys):
    made = []

    def factory(index, *backend):
        cap = _Capture(index, opened=False)
        made.append(cap)
        return cap

    monkeypatch.setattr(cameras.cv2, "VideoCapture", factory)
    assert cameras._list(3, tmp_path / "list") == 1
    assert len(made) == 3 and all(cap.released for cap in made)
    out = capsys.readouterr().out
    assert "no camera opened" in out and "configured overhead: index 2 from OVERHEAD_CAM" in out
    assert json.loads((tmp_path / "list" / "report.json").read_text(encoding="utf-8"))["exit_code"] == 1


def test_list_on_the_fakes_opens_no_device_and_labels_everything_synthetic(
    no_devices, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("FAKE", "1")
    out = tmp_path / "list"
    assert cameras.main(["--list", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["fake"] is True and report["mode"] == "synthetic" and report["backend"] == "fake"
    assert [e["name"] for e in report["cameras"]] == ["overhead", "wrist"]
    for entry in report["cameras"]:
        assert entry["index"] is None and entry["fake"] is True and entry["opened"] is True
        assert entry["width"] == 640 and entry["height"] == 480 and entry["fourcc_reported"] is None
        assert (out / entry["frame"]).read_bytes()[:2] == b"\xff\xd8" and entry["frame"].startswith("fake-")
    assert report["opened"] == ["overhead", "wrist"] and report["exit_code"] == 0
    assert "SYNTHETIC" in capsys.readouterr().out


def test_list_default_directory_is_fresh_under_runs_setup(no_devices, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    assert cameras.main(["--list"]) == 0
    assert cameras.main(["--list"]) == 0
    dirs = sorted((tmp_path / "runs" / "setup").glob("camera-list-*"))
    assert len(dirs) == 2 and all((d / "report.json").is_file() for d in dirs)


def test_list_refuses_an_existing_directory_before_opening_anything(no_devices, tmp_path):
    out = tmp_path / "list"
    out.mkdir()
    assert cameras._list(8, out) == 2
    with pytest.raises(FileExistsError):
        cameras.list_cameras(8, out)
    assert not (out / "report.json").exists()


@pytest.mark.parametrize("bad", [0, -3])
def test_list_rejects_a_non_positive_max_index(no_devices, tmp_path, bad):
    assert cameras._list(bad, tmp_path / "list") == 2
    with pytest.raises(ValueError):
        cameras.list_cameras(bad, tmp_path / "list")


# -- CLI surface ---------------------------------------------------------------------------------------------


def test_help_prints_cleanly_and_warns_about_the_orchestrator(capsys):
    with pytest.raises(SystemExit) as exc:
        cameras.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--check" in out and "--diagnose" in out and "--list" in out and "--criterion" in out
    assert "orchestrator" in out and "touches the arm" in out and "--max-index" in out
    assert "--independent" in out and "synthetic" in out


def test_misuse_exits_2(no_devices, tmp_path, capsys):
    assert cameras.main([]) == 2  # no mode: prints the help
    assert "usage:" in capsys.readouterr().out
    assert cameras.main(["--check", "-1"]) == 2
    assert cameras.main(["--check", "nan"]) == 2
    assert cameras.main(["--list", "--max-index", "0"]) == 2
    for bad in (["--criterion", "lenient", "--check", "1"], ["--check", "1", "--list"]):
        with pytest.raises(SystemExit) as exc:
            cameras.main(bad)
        assert exc.value.code == 2
    assert not list((tmp_path / "runs").rglob("*")) if (tmp_path / "runs").exists() else True


def test_fresh_path_never_reuses_a_name_and_follows_vara_runs_dir(tmp_path, monkeypatch):
    assert cameras.runs_setup_dir() == tmp_path / "runs" / "setup"
    monkeypatch.setenv("VARA_RUNS_DIR", "")
    assert cameras.runs_setup_dir() == cameras.ROOT / "runs" / "setup"
    first = cameras.fresh_path(tmp_path / "x", "camera-check", ".json")
    first.write_text("{}", encoding="utf-8")
    second = cameras.fresh_path(tmp_path / "x", "camera-check", ".json")
    assert first != second and second.name.startswith("camera-check-") and second.suffix == ".json"
    with pytest.raises(FileExistsError):
        cameras.write_report({}, first)


def test_write_report_serializes_before_it_creates_the_file(tmp_path):
    path = tmp_path / "bad.json"
    with pytest.raises(TypeError):
        cameras.write_report({"value": object()}, path)
    assert not path.exists()
    written = cameras.write_report({"ok": True}, tmp_path / "deep" / "good.json")
    assert written.read_bytes() == b'{\n  "ok": true\n}\n'  # LF line ends on every platform


@pytest.mark.parametrize(
    "argv",
    [
        ["--list", "--criterion", "tracking"],
        ["--list", "--camera", "wrist"],
        ["--list", "--independent"],
        ["--check", "1", "--max-index", "3"],
        ["--diagnose", "1", "--max-index", "3"],
    ],
)
def test_a_flag_that_does_not_apply_to_the_mode_is_misuse(no_devices, tmp_path, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        cameras.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not --list" in err or "applies to --list only" in err
    assert not (tmp_path / "runs").exists()  # refused before any report path or camera


def test_check_independent_is_the_diagnose_run(no_devices, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    out = tmp_path / "soak.json"
    code = cameras.main(["--check", "0.4", "--independent", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == report["exit_code"]
    assert report["independent"] is True and report["tool"].endswith("--diagnose")
    assert report["fake"] is True and report["mode"] == "synthetic" and report["criterion"] == "strict"
    assert [feed["name"] for feed in report["feeds"]] == ["overhead", "wrist"]
    assert all(feed["index"] is None and feed["failures"] == 0 and feed["fake"] for feed in report["feeds"])
    assert report["started_at"] <= report["finished_at"]


_STAMPED = r"\d{8}-\d{6}(-\d+)?"


class _QuickFeed:
    """A physical-looking feed (no `is_fake`): one small frame per read, a few milliseconds apart."""

    def __init__(self, name):
        self.name, self.index, self.released = name, {"overhead": 2, "wrist": 1}[name], False
        self.reported = {"width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"}

    def read(self):
        time.sleep(0.002)
        return np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True


def test_physical_default_names_carry_no_synthetic_infix(monkeypatch, tmp_path, captures):
    made = []

    def get(name):
        made.append(_QuickFeed(name))
        return made[-1]

    monkeypatch.setattr(cameras, "get_camera", get)
    cameras._check(0.1, ["overhead", "wrist"])
    cameras._check(0.1, ["overhead"], independent=True)
    cameras.list_cameras(2)
    setup = tmp_path / "runs" / "setup"
    names = sorted(p.name for p in setup.iterdir())
    assert len(names) == 3, names
    assert any(re.fullmatch(rf"camera-check-{_STAMPED}\.json", n) for n in names)
    assert any(re.fullmatch(rf"camera-diagnose-{_STAMPED}\.json", n) for n in names)
    assert any(re.fullmatch(rf"camera-list-{_STAMPED}", n) for n in names)
    check = json.loads(next(setup.glob("camera-check-*.json")).read_text(encoding="utf-8"))
    assert check["fake"] is False and check["mode"] == "physical"
    assert [feed["index"] for feed in check["feeds"]] == [2, 1]
    assert check["feeds"][0]["fourcc_reported"] == "MJPG" and check["feeds"][0]["width_reported"] == 640
    assert len(made) == 3 and all(feed.released for feed in made)


def test_synthetic_default_names_carry_the_infix(no_devices, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    assert cameras.main(["--check", "0.2", "--camera", "wrist"]) in (0, 1)
    assert cameras.main(["--diagnose", "0.2", "--camera", "wrist"]) in (0, 1)
    assert cameras.main(["--list"]) == 0
    names = sorted(p.name for p in (tmp_path / "runs" / "setup").iterdir())
    assert len(names) == 3, names
    assert any(re.fullmatch(rf"camera-check-synthetic-{_STAMPED}\.json", n) for n in names)
    assert any(re.fullmatch(rf"camera-diagnose-synthetic-{_STAMPED}\.json", n) for n in names)
    assert any(re.fullmatch(rf"camera-list-synthetic-{_STAMPED}", n) for n in names)


def test_unopened_physical_feed_records_its_configured_index(monkeypatch):
    made = []

    def get(name):
        if name == "wrist":
            raise RuntimeError("camera 1 did not open")
        made.append(_QuickFeed(name))
        return made[-1]

    monkeypatch.setattr(cameras, "get_camera", get)
    report = cameras.check_feeds(0.1, ["overhead", "wrist"], progress=False)
    wrist = report["feeds"][1]
    assert wrist["opened"] is False and wrist["index"] == 1 and "did not open" in wrist["reasons"]
    assert wrist["fake"] is False and report["fake"] is False
    assert report["exit_code"] == 1 and all(feed.released for feed in made)
    monkeypatch.setenv("FAKE_CAMERA", "1")
    again = cameras.check_feeds(0.1, ["overhead", "wrist"], progress=False)
    assert again["feeds"][1]["opened"] is False and again["feeds"][1]["index"] is None  # fakes have none
    assert again["feeds"][1]["fake"] is True and again["fake"] is True  # a failed fake is still synthetic


class _BlockingFake(cameras.FakeCamera):
    """A paced fake whose read number `at` blocks until `gate` is set: a capture hung in the driver."""

    def __init__(self, *args, at: int, gate: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.at, self.gate = at, gate

    def read(self):
        if self.frames + 1 == self.at:
            self.gate.wait(timeout=30)
        return super().read()


def test_diagnose_names_a_blocked_read_and_never_releases_under_it(no_devices, monkeypatch):
    monkeypatch.setattr(cameras, "JOIN_TIMEOUT_S", 0.3)
    gate = threading.Event()
    fakes = {
        "overhead": cameras.FakeCamera(name="overhead", fps=30.0),
        "wrist": _BlockingFake(name="wrist", fps=30.0, at=3, gate=gate),
    }
    monkeypatch.setattr(cameras, "get_camera", fakes.__getitem__)
    started = time.perf_counter()
    try:
        report = cameras.check_feeds(0.5, ["overhead", "wrist"], independent=True, progress=False)
        took = time.perf_counter() - started
        released_while_blocked = fakes["wrist"]._released
    finally:
        gate.set()
        for thread in threading.enumerate():
            if thread.name == "camera-wrist":
                thread.join(timeout=5)
    overhead, wrist = report["feeds"]
    assert took < 0.5 + 0.3 + 2.0  # the deadline ended the run; it did not wait for the hung read
    assert "read still blocked" in wrist["error"] and wrist["completed"] is False and not wrist["passed"]
    assert overhead["completed"] and overhead["error"] is None
    assert report["completed"] is False and report["exit_code"] == 1
    assert any("wrist: reader still inside read()" in error for error in report["errors"])
    assert fakes["overhead"]._released and not released_while_blocked


def test_diagnose_ctrl_c_stops_every_reader_and_releases_both(no_devices, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt  # the first progress line, about 1 s in, stands in for Ctrl+C

    monkeypatch.setattr(cameras._FeedTiming, "progress", interrupt)
    fakes = {name: cameras.FakeCamera(name=name, fps=30.0) for name in ("overhead", "wrist")}
    monkeypatch.setattr(cameras, "get_camera", fakes.__getitem__)
    started = time.perf_counter()
    report = cameras.check_feeds(30.0, ["overhead", "wrist"], independent=True, progress=True)
    assert time.perf_counter() - started < 10.0  # stopped near 1 s, not after the requested 30 s
    assert report["completed"] is False and report["exit_code"] == 1
    assert "interrupted before completion" in report["errors"]
    assert "the check did not complete" in report["problems"]
    assert [feed["error"] for feed in report["feeds"]] == ["interrupted", "interrupted"]
    assert all(cam._released for cam in fakes.values())


def test_diagnose_readers_that_cannot_start_together_fail_and_release(no_devices, monkeypatch):
    original = threading.Barrier

    class NeverFull(original):
        """One party more than the readers and the main thread: the start can only time out."""

        def __init__(self, parties):
            super().__init__(parties + 1)

        def wait(self, timeout=None):
            return super().wait(timeout=0.2)

    monkeypatch.setattr(cameras.threading, "Barrier", NeverFull)
    fakes = {name: cameras.FakeCamera(name=name, fps=30.0) for name in ("overhead", "wrist")}
    monkeypatch.setattr(cameras, "get_camera", fakes.__getitem__)
    report = cameras.check_feeds(0.3, ["overhead", "wrist"], independent=True, progress=False)
    assert report["completed"] is False and report["exit_code"] == 1 and report["elapsed_s"] == 0.0
    assert [feed["error"] for feed in report["feeds"]] == ["reader threads did not start together"] * 2
    assert all(feed["frames"] == 0 for feed in report["feeds"])
    assert all(cam._released for cam in fakes.values())


def test_list_ctrl_c_writes_the_indexes_done_so_far(monkeypatch, tmp_path, capsys):
    class Interrupting(_Capture):
        def read(self):
            raise KeyboardInterrupt

    made = []

    def factory(index, *backend):
        cap = Interrupting(index) if index == 1 else _Capture(index, opened=index == 0)
        made.append(cap)
        return cap

    monkeypatch.setattr(cameras.cv2, "VideoCapture", factory)
    out = tmp_path / "list"
    assert cameras._list(4, out) == 1
    assert [cap.index for cap in made] == [0, 1] and all(cap.released for cap in made)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] is False and report["exit_code"] == 1
    assert [entry["index"] for entry in report["cameras"]] == [0]
    assert report["cameras"][0]["frame"] == "camera-0.jpg" and (out / "camera-0.jpg").is_file()
    assert "interrupted" in capsys.readouterr().err


def test_list_records_a_failed_release_and_keeps_going(monkeypatch, tmp_path):
    class BadRelease(_Capture):
        def release(self):
            super().release()
            raise RuntimeError("driver refused")

    made = []

    def factory(index, *backend):
        cap = BadRelease(index, opened=False) if index == 0 else _Capture(index)
        made.append(cap)
        return cap

    monkeypatch.setattr(cameras.cv2, "VideoCapture", factory)
    report = cameras.list_cameras(2, tmp_path / "list")
    assert [entry["index"] for entry in report["cameras"]] == [0, 1] and all(cap.released for cap in made)
    assert report["cameras"][0]["error"] == "release failed (RuntimeError: driver refused)"
    assert report["cameras"][1]["opened"] and report["cameras"][1]["error"] is None
    assert report["completed"] is True and report["exit_code"] == 0


def test_a_report_that_appears_during_the_run_is_never_overwritten(no_devices, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FAKE_CAMERA", "1")
    out = tmp_path / "soak.json"
    real = cameras.check_feeds

    def racing(*args, **kwargs):
        report = real(*args, **kwargs)
        out.write_text("someone else's soak", encoding="utf-8")
        return report

    monkeypatch.setattr(cameras, "check_feeds", racing)
    code = cameras._check(0.2, ["wrist"], out=out)
    assert out.read_text(encoding="utf-8") == "someone else's soak"
    (moved,) = tmp_path.glob("soak-*.json")
    assert json.loads(moved.read_text(encoding="utf-8"))["exit_code"] == code
    assert "appeared during the run" in capsys.readouterr().err


def test_elapsed_rounded_to_the_millisecond_is_not_a_short_run():
    feed = {
        "name": "wrist",
        "opened": True,
        "frames": 54000,
        "failures": 0,
        "late": 0,
        "long": 0,
        "fps": 30.0,
        "completed": True,
        "error": None,
    }
    saved = {
        "requested_feeds": ["wrist"],
        "requested_s": 1800.0004,
        "elapsed_s": 1800.0,
        "completed": True,
        "errors": [],
        "feeds": [feed],
    }
    assert cameras.apply_criterion(saved, "strict")["passed"] is True
    short = cameras.apply_criterion({**saved, "elapsed_s": 1799.99}, "strict")
    assert short["passed"] is False and "shorter than requested" in short["problems"][0]
