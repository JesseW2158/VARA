"""I2 with five loose markers: `scripts/touch_markers.py --loose` (detections -> targets, the --count and
--objects rules, the fake dry runs, the --real gating) and the spare cards of `scripts/make_markers.py`.
No hardware: the cameras are FakeCameras, the arm is the FakeDriver, and cv2.VideoCapture is never called."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.calibrate import load_scene_cfg, synthetic_homography
from vision.cameras import FakeCamera, load_fake_scene
from vision.markers import CALIB_IDS, HAND_ID, detect, marker_table
from vision.scene import FakeSceneTracker, SceneTracker, reset_tracker

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "touch_markers.py"
POS_TOL = 0.005  # m: the synthetic homography is exact, sub-pixel corners add about 1 mm
ABOVE_HOVER = 0.05  # m: the I2 hover height above the table
# --object-cards-flat: the three object cards lie flat where the objects stood, then spares 13 and 14
LOOSE_NAMES = [f"marker_{i}" for i in (10, 11, 12, 13, 14)]
# the default I2 table: the objects stand with 10-12 glued on top (left out), spares 13-17 lie flat
SPARE_NAMES = [f"marker_{i}" for i in (13, 14, 15, 16, 17)]
LEFT_OUT = [
    {"marker": 10, "name": "pill_bottle"},
    {"marker": 11, "name": "juice_box"},
    {"marker": 12, "name": "tea_tin"},
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT")}
    env.update(extra)
    return env


def _run(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        cwd=str(ROOT),
        env=env if env is not None else _env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )


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
def touch():
    return _load_script("touch_markers")


@pytest.fixture(autouse=True)
def no_devices(monkeypatch):
    """Any attempt to open a camera device in this process fails the test."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)


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


class _OverheadStandIn(FakeCamera):
    """Plays the physical overhead camera in the --real flow (`is_fake = False`, as `vision.cameras.Camera`
    has); its frames are still the FakeCamera's synthetic ones."""

    is_fake = False


class _Watcher:
    """A stand-in for arm.driver.EscWatcher that never reads the console."""

    def __init__(self, callback=None):
        self.callback, self.pressed = callback, False

    def start(self):
        return None

    def stop(self):
        return None


# -- the fake table and the detection -> targets mapping --------------------------------------------------


def test_loose_fake_scene_keeps_the_objects_standing_and_lays_five_spares_flat(touch, fake_scene):
    scene = touch.loose_fake_scene(fake_scene)
    cards = scene["objects"]
    assert list(cards) == [*fake_scene["objects"], *SPARE_NAMES]
    for name, item in fake_scene["objects"].items():
        assert cards[name] == item and cards[name]["h"] > 0.0  # glued on top, as the scene reports it
    for name, (x, y) in zip(SPARE_NAMES, touch.LOOSE_SPOTS, strict=True):
        card = cards[name]
        assert (card["x"], card["y"], card["h"]) == (x, y, 0.0)
        assert card["size"] == [0.03, 0.03] and card["seen"] is True
    assert scene["obstacles"] == {} and scene["hand"]["seen"] is False
    assert scene["zones"] == fake_scene["zones"]
    assert list(touch.loose_fake_scene(fake_scene, count=2)["objects"])[3:] == SPARE_NAMES[:2]
    bare = {**copy.deepcopy(fake_scene), "objects": {}}
    assert list(touch.loose_fake_scene(bare)["objects"]) == SPARE_NAMES


def test_loose_fake_scene_with_object_cards_flat_takes_the_objects_away(touch, fake_scene):
    scene = touch.loose_fake_scene(fake_scene, object_cards_flat=True)
    cards = scene["objects"]
    assert list(cards) == LOOSE_NAMES  # the three object cards where their objects stood, then spares 13, 14
    for item in fake_scene["objects"].values():
        card = cards[f"marker_{item['marker']}"]
        assert (card["x"], card["y"], card["yaw"]) == (item["x"], item["y"], item["yaw"])
    for card in cards.values():
        assert card["h"] == 0.0 and card["size"] == [0.03, 0.03] and card["seen"] is True
    assert len(touch.loose_fake_scene(fake_scene, count=7, object_cards_flat=True)["objects"]) == 7
    two = touch.loose_fake_scene(fake_scene, count=2, object_cards_flat=True)
    assert list(two["objects"]) == LOOSE_NAMES[:3]


def test_object_card_ids_and_leave_out_follow_objects_yaml(touch):
    assigned = touch.object_card_ids()
    assert assigned == {10: "pill_bottle", 11: "juice_box", 12: "tea_tin"}
    assert touch.object_card_ids({10: {"name": "a"}, 30: {"name": "box"}, 40: {"name": "hand"}}) == {10: "a"}
    targets = [(f"marker_{i}", {"marker": i}) for i in (10, 12, 13)]
    kept, left = touch.leave_out_object_cards(targets, assigned, object_cards_flat=False)
    assert [n for n, _ in kept] == ["marker_13"] and [n for n, _ in left] == ["marker_10", "marker_12"]
    kept, left = touch.leave_out_object_cards(targets, assigned, object_cards_flat=True)
    assert [n for n, _ in kept] == ["marker_10", "marker_12", "marker_13"] and left == []


def test_a_glued_marker_maps_into_its_object_which_is_why_it_is_left_out(
    touch, fake_scene, cfg, table, homography
):
    """The review's hazard: through the homography at height 0 a marker glued on top of an object lands
    inside the object's footprint, so a hover 5 cm above the table there is inside the object."""
    from vision.calibrate import camera_pose

    scene = touch.loose_fake_scene(fake_scene)
    cam = FakeCamera(scene, scene_cfg=cfg, objects=table)
    targets, counts = touch.loose_targets(touch.collect_frames(cam.read, 10), homography)
    assert counts == dict.fromkeys(range(10, 18), 10)
    found = dict(targets)
    for name, item in fake_scene["objects"].items():
        seen = found[f"marker_{item['marker']}"]
        offset = math.hypot(seen["x"] - item["x"], seen["y"] - item["y"])
        assert offset < min(item["size"]) / 2, name  # still over the object
        assert ABOVE_HOVER < item["h"], name  # so the 5 cm hover lies below its top
    c, hc = camera_pose(cfg)
    expected = dict(touch.fake_detections(scene, c, hc))
    for name, item in targets:  # fake_detections predicts what the camera sees, glued or flat
        assert math.hypot(item["x"] - expected[name]["x"], item["y"] - expected[name]["y"]) < POS_TOL, name
    kept, left = touch.leave_out_object_cards(targets, touch.object_card_ids(), object_cards_flat=False)
    assert [n for n, _ in kept] == SPARE_NAMES and [n for n, _ in left] == LOOSE_NAMES[:3]


def test_loose_targets_map_the_cards_through_the_homography(touch, fake_scene, cfg, table, homography):
    scene = touch.loose_fake_scene(fake_scene, object_cards_flat=True)
    cam = FakeCamera(scene, scene_cfg=cfg, objects=table)
    frames = touch.collect_frames(cam.read, 10)
    assert len(frames) == 10 and cam.frames == 10
    assert set(CALIB_IDS) <= {d[0] for d in detect(frames[0])}  # the table markers are in view...
    targets, counts = touch.loose_targets(frames, homography)
    assert [name for name, _ in targets] == LOOSE_NAMES  # ...but only ids 10-19 become targets
    assert counts == dict.fromkeys((10, 11, 12, 13, 14), 10)
    for name, item in targets:
        card = scene["objects"][name]
        assert item["marker"] == card["marker"] and item["h"] == 0.0 and item["frames"] == 10
        assert math.hypot(item["x"] - card["x"], item["y"] - card["y"]) < POS_TOL, name
        assert abs(math.remainder(item["yaw"] - card["yaw"], 2 * math.pi)) < math.radians(5), name
    with pytest.raises(ValueError):
        touch.loose_targets([], homography)


def test_a_card_seen_in_fewer_than_half_the_frames_is_not_a_target(touch, fake_scene, cfg, table, homography):
    full = touch.loose_fake_scene(fake_scene, object_cards_flat=True)
    full["hand"].update(x=0.40, y=-0.20, seen=True)  # the wristband (40) in view: never a target
    gone = copy.deepcopy(full)
    del gone["objects"]["marker_14"]
    cam = _SequenceCamera([full] * 4 + [gone] * 6, scene_cfg=cfg, objects=table)
    frames = touch.collect_frames(cam.read, 10)
    assert HAND_ID in {d[0] for d in detect(frames[0])}
    targets, counts = touch.loose_targets(frames, homography)
    assert counts == {10: 10, 11: 10, 12: 10, 13: 10, 14: 4} and HAND_ID not in counts
    assert [name for name, _ in targets] == LOOSE_NAMES[:4]  # 4 of 10 frames is not enough
    cam = _SequenceCamera([full] * 5 + [gone] * 5, scene_cfg=cfg, objects=table)
    targets, counts = touch.loose_targets(touch.collect_frames(cam.read, 10), homography)
    assert counts[14] == 5 and dict(targets)["marker_14"]["frames"] == 5  # half of them is


def test_collect_frames_treats_a_dead_camera_as_an_error(touch):
    frame = np.zeros((480, 640, 3), np.uint8)
    flaky = iter([None, frame, None, frame, frame, None] + [frame] * 7)
    got = touch.collect_frames(lambda: next(flaky), 10)
    assert len(got) == 10 and all(f is frame for f in got)  # 3 misses of 13 grabs: skipped, not counted
    dead = iter([None] * 10)
    with pytest.raises(TimeoutError, match="no frame"):
        touch.collect_frames(lambda: next(dead), 4)  # the third miss (2 * 3 > 4) ends it


def test_tracker_frames_use_the_tracker_without_a_second_camera(touch, fake_scene, cfg, table, homography):
    cam = _FlakyCamera(touch.loose_fake_scene(fake_scene), 3, scene_cfg=cfg, objects=table)
    tracker = SceneTracker(cam, homography, table, cfg, hz=30)
    next_frame = touch.tracker_frames(tracker)
    first, second = next_frame(), next_frame()  # not running: one synchronous step per frame
    assert first is not None and first.shape == (480, 640, 3) and second is not None
    assert next_frame() is None  # a missed grab, not a stale frame
    assert (tracker.frames, tracker.misses) == (2, 1)
    frames = touch.collect_frames(next_frame, 4)
    assert len(frames) == 4 and tracker.frames == 6 and cam.reads == 8
    cam.every = 10**6
    tracker.start()
    try:
        running = touch.collect_frames(touch.tracker_frames(tracker, timeout=2.0), 5)
    finally:
        tracker.close()
    assert len(running) == 5 and all(f.shape == (480, 640, 3) for f in running)


# -- the dry runs ------------------------------------------------------------------------------------------


def test_loose_fake_dry_run_touches_five_cards(tmp_path):
    out = tmp_path / "loose"
    proc = _run("--fake", "--loose", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["mode"] == "fake" and results["loose"] is True and results["count"] == 5
    assert results["loose_detections"] is None  # synthesized cards: nothing was detected
    assert results["passed"] is True and results["worst_cm"] < 1.5
    assert results["object_cards_flat"] is False and results["left_out"] == LEFT_OUT
    assert [m["name"] for m in results["markers"]] == SPARE_NAMES
    for marker in results["markers"]:
        assert marker["ok"] and marker["problems"] == [] and marker["gap_cm"] < 1.5
        assert marker["marker_center"][2] == 0.0 and marker["target"][2] == pytest.approx(0.05)
        assert marker["replay_go"]["ok"] and marker["replay_back"]["ok"]
    assert "loose markers on the table: [13, 14, 15, 16, 17]" in proc.stdout and "PASS" in proc.stdout
    assert "left out: marker 10 (pill_bottle in config/objects.yaml)" in proc.stdout


def test_loose_fake_dry_run_with_object_cards_flat_targets_them(tmp_path):
    out = tmp_path / "flat"
    proc = _run("--fake", "--loose", "--object-cards-flat", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["object_cards_flat"] is True and results["left_out"] == []
    assert [m["name"] for m in results["markers"]] == LOOSE_NAMES and results["passed"] is True
    assert "objects must be OFF the table" in proc.stdout


def test_loose_fake_chain_detects_the_cards_on_the_fake_camera(tmp_path):
    out = tmp_path / "chain"
    proc = _run("--fake", "--chain", "--loose", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["chain"] is True and results["loose"] is True and results["passed"] is True
    assert results["loose_detections"] == [{"marker": m, "frames": 10} for m in range(10, 18)]
    assert results["worst_cm"] < 0.5  # the detection error on the synthetic camera is about 1 mm
    assert [m["marker"] for m in results["markers"]] == [13, 14, 15, 16, 17]
    assert results["left_out"] == LEFT_OUT
    assert all(m["marker_center"][2] == 0.0 for m in results["markers"])


def test_loose_fake_chain_maps_through_the_synthetic_homography(monkeypatch, touch, fake_scene):
    """The FakeCamera renders through the synthetic camera model, so --fake --chain --loose maps its frames
    through `synthetic_homography`, also once B5 has saved the physical camera's config/homography.npy."""
    import vision.calibrate as calibrate_mod

    def saved(*args, **kwargs):
        raise AssertionError("FakeCamera frames must not map through the saved (physical) homography")

    monkeypatch.setattr(calibrate_mod, "load_homography", saved)
    targets, counts, hand_source = touch._loose_targets_and_hand(
        argparse.Namespace(fake=True, chain=True, real=False, object_cards_flat=True)
    )
    assert hand_source is None and counts == dict.fromkeys((10, 11, 12, 13, 14), 10)
    cards = touch.loose_fake_scene(fake_scene, object_cards_flat=True)["objects"]
    assert [name for name, _ in targets] == LOOSE_NAMES
    for name, item in targets:
        assert math.hypot(item["x"] - cards[name]["x"], item["y"] - cards[name]["y"]) < POS_TOL, name


def test_the_normal_run_is_unchanged_by_the_loose_mode(tmp_path):
    out = tmp_path / "glued"
    proc = _run("--fake", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["loose"] is False and "count" not in results and "loose_detections" not in results
    assert {m["name"] for m in results["markers"]} == {"pill_bottle", "juice_box", "tea_tin"}
    assert results["markers"][0]["marker_center"][2] > 0.0  # glued on top of the object


# -- the --count rule -------------------------------------------------------------------------------------


def _in_process(monkeypatch, touch, argv):
    """Run main() on the fakes in this process (the singletons are reset around it)."""
    import arm.driver as driver_mod

    monkeypatch.setenv("FAKE_ROBOT", "1")
    monkeypatch.setenv("FAKE_CAMERA", "1")
    driver_mod.reset_driver()
    reset_tracker()
    try:
        return touch.main(argv)
    finally:
        driver_mod.reset_driver()
        reset_tracker()


def test_fewer_cards_than_count_moves_nothing(monkeypatch, tmp_path, capsys, touch):
    import arm.driver as driver_mod

    monkeypatch.setattr(touch, "LOOSE_SPOTS", touch.LOOSE_SPOTS[:4])  # four spare cards on the table
    monkeypatch.setattr(driver_mod, "get_driver", lambda: pytest.fail("the driver must not be touched"))
    rc = _in_process(monkeypatch, touch, ["--fake", "--loose", "--out", str(tmp_path / "four")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "loose markers on the table: [13, 14, 15, 16] (4 of ids 10-19 seen, 5 needed)" in captured.out
    assert "only 4 loose markers seen; need 5 (--count); nothing moved" in captured.err
    assert not (tmp_path / "four").exists()


def test_count_lowers_the_bar_and_objects_still_filters(monkeypatch, tmp_path, touch):
    monkeypatch.setattr(touch, "LOOSE_SPOTS", touch.LOOSE_SPOTS[:4])
    rc = _in_process(
        monkeypatch, touch, ["--fake", "--loose", "--count", "4", "--out", str(tmp_path / "four")]
    )
    assert rc == 0
    results = json.loads((tmp_path / "four" / "results.json").read_text(encoding="utf-8"))
    assert results["count"] == 4 and results["passed"] is True
    assert [m["name"] for m in results["markers"]] == SPARE_NAMES[:4]
    rc = _in_process(
        monkeypatch,
        touch,
        ["--fake", "--loose", "--count", "4", "--objects", "marker_13", "--out", str(tmp_path / "one")],
    )
    assert rc == 0
    results = json.loads((tmp_path / "one" / "results.json").read_text(encoding="utf-8"))
    assert [m["name"] for m in results["markers"]] == ["marker_13"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--fake", "--count", "5"],  # --count needs --loose
        ["--fake", "--loose", "--count", "0"],
        ["--fake", "--loose", "--count", "-1"],
        ["--fake", "--loose", "--count", "five"],
        ["--fake", "--loose", "--count", "11"],  # only ten object-size ids exist
        ["--fake", "--loose", "--count", "8"],  # ...and objects.yaml assigns three of them, left out
        ["--fake", "--object-cards-flat"],  # --object-cards-flat needs --loose
    ],
)
def test_count_is_a_positive_integer_that_needs_loose(touch, argv):
    with pytest.raises(SystemExit) as exc:  # an argparse error, before any flag or scene is touched
        touch.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "names",
    [["juice_box"], ["marker_9"], ["marker_20"], ["marker_13", "marker_x"], ["marker_013"]],
)
def test_loose_objects_must_be_marker_names_of_object_size_ids(monkeypatch, capsys, touch, names):
    monkeypatch.delenv("FAKE_ROBOT", raising=False)
    with pytest.raises(SystemExit) as exc:  # a usage error, before any flag or scene is touched
        touch.main(["--fake", "--loose", "--objects", *names])
    assert exc.value.code == 2
    assert "marker_<id> for ids 10-19" in capsys.readouterr().err
    assert "FAKE_ROBOT" not in os.environ  # main() stopped before it forced the fakes


@pytest.mark.parametrize("name", ["marker_10", "marker_12"])
def test_loose_objects_naming_a_glued_object_card_is_a_usage_error(monkeypatch, capsys, touch, name):
    monkeypatch.delenv("FAKE_ROBOT", raising=False)
    with pytest.raises(SystemExit) as exc:
        touch.main(["--fake", "--loose", "--objects", name])
    assert exc.value.code == 2
    assert "glued on objects and left out" in capsys.readouterr().err
    assert "FAKE_ROBOT" not in os.environ


def test_a_named_marker_that_is_not_seen_moves_nothing(monkeypatch, tmp_path, capsys, touch):
    import arm.driver as driver_mod

    monkeypatch.setattr(driver_mod, "get_driver", lambda: pytest.fail("the driver must not be touched"))
    argv = ["--fake", "--loose", "--objects", "marker_13", "marker_18", "--out", str(tmp_path / "named")]
    assert _in_process(monkeypatch, touch, argv) == 1  # the table holds spare cards 13-17: 18 is not there
    err = capsys.readouterr().err
    assert "marker_18 named with --objects but not among the loose markers seen; nothing moved" in err
    assert not (tmp_path / "named").exists()


# -- the --real gating ------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["FAKE_CAMERA", "FAKE", "FAKE_ROBOT"])
def test_loose_real_refuses_every_fake_flag_before_any_prompt(flag, tmp_path):
    out = tmp_path / "real"
    proc = _run("--real", "--loose", "--out", str(out), env=_env(**{flag: "1"}))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "refusing" in proc.stderr and flag in proc.stderr and not out.exists()


def test_loose_real_refuses_a_fake_tracker_installed_in_process(monkeypatch, tmp_path, capsys, touch):
    import dotenv

    import arm.driver as driver_mod
    import vision.calibrate as calibrate_mod
    import vision.scene as scene_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: FakeSceneTracker())
    monkeypatch.setattr(driver_mod, "get_driver", lambda: pytest.fail("the driver must not be touched"))
    monkeypatch.setattr(
        calibrate_mod, "load_homography", lambda *args, **kwargs: pytest.fail("no detection before refusing")
    )
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("prompted although refusing"))
    rc = touch.main(["--real", "--loose", "--out", str(tmp_path / "real")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing" in err and "FakeSceneTracker" in err and not (tmp_path / "real").exists()


def test_loose_real_refuses_a_tracker_on_a_fake_camera(monkeypatch, tmp_path, capsys, touch, cfg, table):
    """A real SceneTracker (is_fake False) whose camera is a FakeCamera would feed synthetic frames to the
    real arm: refused before the homography, the prompt or the driver, exactly like a fake tracker."""
    import dotenv

    import arm.driver as driver_mod
    import vision.calibrate as calibrate_mod
    import vision.scene as scene_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    tracker = SceneTracker(FakeCamera(scene_cfg=cfg, objects=table), synthetic_homography(cfg), table, cfg)
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: tracker)
    monkeypatch.setattr(driver_mod, "get_driver", lambda: pytest.fail("the driver must not be touched"))
    monkeypatch.setattr(
        calibrate_mod, "load_homography", lambda *args, **kwargs: pytest.fail("no detection before refusing")
    )
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("prompted although refusing"))
    rc = touch.main(["--real", "--loose", "--out", str(tmp_path / "real")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing" in err and "FakeCamera" in err and "nothing moved" in err
    assert tracker.frames == 0 and not (tmp_path / "real").exists()


def test_loose_real_detects_through_the_running_tracker(
    monkeypatch, tmp_path, capsys, touch, fake_scene, cfg, table, homography
):
    """--real --loose with a real SceneTracker on a stand-in for the overhead camera (is_fake False, showing
    the five cards) and the FakeDriver standing in for the arm: the frames come from the running tracker,
    the targets from the saved homography (pinned to the synthetic one, which the stand-in's frames match,
    whatever B5 has saved), and the run goes through the confirmation, the countdown and the five gap
    prompts."""
    import dotenv

    import arm.driver as driver_mod
    import vision.calibrate as calibrate_mod
    import vision.scene as scene_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(calibrate_mod, "load_homography", lambda *args, **kwargs: homography)
    driver = driver_mod.FakeDriver()
    monkeypatch.setattr(driver_mod, "get_driver", lambda: driver)
    monkeypatch.setattr(driver_mod, "EscWatcher", _Watcher)
    camera = _OverheadStandIn(touch.loose_fake_scene(fake_scene), scene_cfg=cfg, objects=table)
    tracker = SceneTracker(camera, homography, table, cfg, hz=30)
    tracker.start()
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: tracker)
    prompts: list[str] = []

    def answer(prompt: str = "") -> str:
        prompts.append(prompt)
        return "yes" if "Type yes" in prompt else ""  # confirm once; blank keeps the fk estimate of each gap

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(touch, "_countdown", lambda seconds, watcher: True)
    try:
        rc = touch.main(["--real", "--loose", "--out", str(tmp_path / "real")])
    finally:
        tracker.close()
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    results = json.loads((tmp_path / "real" / "results.json").read_text(encoding="utf-8"))
    assert results["mode"] == "real" and results["loose"] is True and results["passed"] is True
    assert [m["marker"] for m in results["markers"]] == [13, 14, 15, 16, 17]
    assert results["loose_detections"] == [{"marker": m, "frames": 10} for m in range(10, 18)]
    assert results["left_out"] == LEFT_OUT  # the objects stand there: their glued cards are no targets
    assert sum("Type yes" in p for p in prompts) == 1 and sum("measured gap" in p for p in prompts) == 5
    assert driver.commands and driver.torque and not driver.stopped


def test_loose_real_esc_during_connect_moves_nothing_and_is_recorded(
    monkeypatch, tmp_path, capsys, touch, fake_scene, cfg, table, homography
):
    """A13 (W2b): an Esc that lands while the driver connects freezes the arm, nothing moves, and
    touch_markers appends a connect-phase record (a diagnostic, outside the 200 ms bar)."""
    import time

    import dotenv

    import arm.driver as driver_mod
    import vision.calibrate as calibrate_mod
    import vision.scene as scene_mod

    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(calibrate_mod, "load_homography", lambda *args, **kwargs: homography)
    watchers: list[_Watcher] = []

    def make_watcher(callback=None):
        watchers.append(_Watcher(callback))
        return watchers[-1]

    driver = driver_mod.FakeDriver()
    driver.disconnect()
    original_connect = driver.connect

    def connect():
        watchers[-1].pressed, watchers[-1].pressed_at = True, time.monotonic()
        watchers[-1].callback()  # the watcher's freeze lands while connecting
        original_connect()

    driver.connect = connect
    monkeypatch.setattr(driver_mod, "get_driver", lambda: driver)
    monkeypatch.setattr(driver_mod, "EscWatcher", make_watcher)
    camera = _OverheadStandIn(touch.loose_fake_scene(fake_scene), scene_cfg=cfg, objects=table)
    tracker = SceneTracker(camera, homography, table, cfg, hz=30)
    tracker.start()
    monkeypatch.setattr(scene_mod, "get_tracker", lambda: tracker)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    monkeypatch.setattr(touch, "_countdown", lambda seconds, watcher: True)
    try:
        rc = touch.main(["--real", "--loose", "--out", str(tmp_path / "real")])
    finally:
        tracker.close()
    captured = capsys.readouterr()
    assert rc == 1 and driver.commands == [] and driver.stopped and driver.torque
    assert "Esc during connect" in captured.err and "connect phase" in captured.out
    log = tmp_path / "runs" / "calibration" / "esc-latency.jsonl"
    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["source"] == "touch_markers" and record["phase"] == "connect"
    assert record["extra"] == {"argv": ["--real", "--loose", "--out", str(tmp_path / "real")]}


# -- make_markers: the spare cards -------------------------------------------------------------------------


def test_make_markers_prints_spare_cards_for_i2(tmp_path):
    make_markers = _load_script("make_markers")
    out = tmp_path / "markers"
    assert make_markers.main(["--out", str(out)]) == 0
    manifest = json.loads((out / "markers.json").read_text(encoding="utf-8"))
    spares = [m for m in manifest["markers"] if m["spare"]]
    assert [m["id"] for m in spares] == [13, 14, 15, 16, 17]
    for entry in spares:
        assert entry["name"] == "spare, I2 check" and entry["kind"] == "object" and entry["side_cm"] == 3
        image = cv2.imread(str(out / entry["file"]))
        assert [d[0] for d in detect(image)] == [entry["id"]]
    configured = set(CALIB_IDS) | {10, 11, 12, 30, HAND_ID}
    assert {m["id"] for m in manifest["markers"] if not m["spare"]} == configured
    two = tmp_path / "two"
    assert make_markers.main(["--out", str(two), "--spare-ids", "18", "19"]) == 0
    ids = {m["id"] for m in json.loads((two / "markers.json").read_text(encoding="utf-8"))["markers"]}
    assert {18, 19} <= ids and not ({13, 14, 15, 16, 17} & ids) and (two / "marker_18.png").exists()
    none = tmp_path / "none"
    assert make_markers.main(["--out", str(none), "--spare-ids"]) == 0
    manifest = json.loads((none / "markers.json").read_text(encoding="utf-8"))
    assert not any(m["spare"] for m in manifest["markers"]) and len(manifest["markers"]) == 13


@pytest.mark.parametrize("ids", [["10"], ["9"], ["20"], ["13", "13"], ["x"]])
def test_make_markers_rejects_bad_spare_ids(tmp_path, ids):
    make_markers = _load_script("make_markers")
    with pytest.raises(SystemExit) as exc:
        make_markers.main(["--out", str(tmp_path / "bad"), "--spare-ids", *ids])
    assert exc.value.code == 2
    assert not (tmp_path / "bad").exists()


def test_default_spare_ids_skip_the_ids_objects_yaml_assigns(monkeypatch, tmp_path, capsys):
    """A11 may give a new object id 13: the default run then still prints five spare cards (14, 16-19
    here, with a note) instead of failing, and naming an assigned id explicitly stays a usage error."""
    make_markers = _load_script("make_markers")
    table = make_markers.load_objects()
    grown = {**table, 13: {**table[10], "name": "biscuit_tin"}, 15: {**table[10], "name": "mug"}}
    crowded = {**table, **{i: {**table[10], "name": f"thing_{i}"} for i in range(13, 17)}}
    assert make_markers.default_spare_ids(table) == [13, 14, 15, 16, 17]
    assert make_markers.default_spare_ids(grown) == [14, 16, 17, 18, 19]
    assert make_markers.default_spare_ids(crowded) == [17, 18, 19]  # fewer than five free: all of them
    monkeypatch.setattr(make_markers, "load_objects", lambda *args, **kwargs: grown)
    out = tmp_path / "grown"
    assert make_markers.main(["--out", str(out)]) == 0
    assert "the spare cards are 14 16 17 18 19" in capsys.readouterr().out
    manifest = json.loads((out / "markers.json").read_text(encoding="utf-8"))
    assert [m["id"] for m in manifest["markers"] if m["spare"]] == [14, 16, 17, 18, 19]
    assert {m["id"]: m["name"] for m in manifest["markers"] if m["id"] in (13, 15)} == {
        13: "biscuit_tin",
        15: "mug",
    }
    with pytest.raises(SystemExit) as exc:
        make_markers.main(["--out", str(tmp_path / "bad"), "--spare-ids", "13"])
    assert exc.value.code == 2 and not (tmp_path / "bad").exists()


def test_make_markers_names_the_leftovers_of_an_earlier_run(tmp_path, capsys):
    make_markers = _load_script("make_markers")
    out = tmp_path / "markers"
    assert make_markers.main(["--out", str(out)]) == 0
    assert "earlier run" not in capsys.readouterr().out
    assert make_markers.main(["--out", str(out), "--spare-ids"]) == 0  # no spare cards this time
    printed = capsys.readouterr().out
    manifest = json.loads((out / "markers.json").read_text(encoding="utf-8"))
    stale = make_markers.stale_files(out, manifest)
    assert {f"marker_{i}.png" for i in range(13, 18)} <= set(stale)
    assert not ({m["file"] for m in manifest["markers"]} | set(manifest["sheets"])) & set(stale)
    assert all((out / name).exists() for name in stale)  # named in the note, never deleted
    assert f"also holds {len(stale)} file(s) of an earlier run" in printed and "marker_13.png" in printed


@pytest.mark.parametrize(
    "script, flags",
    [("touch_markers.py", ("--loose", "--count")), ("make_markers.py", ("--spare-ids",))],
)
def test_help_prints_the_new_flags(script, flags):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert all(flag in proc.stdout for flag in flags)
