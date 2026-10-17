"""Real verification (B7): vision.verify on the 10 recorded before/after pairs in tests/fixtures/verify/."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vision.scene import FakeSceneTracker
from vision.verify import missing_objects, spec_objects, take_after, verify

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "verify"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))
SCENE_SCHEMA = Draft202012Validator(json.loads((ROOT / "schemas" / "scene.json").read_text(encoding="utf-8")))
SPEC_SCHEMA = Draft202012Validator(
    json.loads((ROOT / "schemas" / "task_spec.json").read_text(encoding="utf-8"))
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(name: str) -> dict:
    return load(FIXTURE_DIR / f"{name}.json")


def test_ten_fixtures_five_pass_five_fail():
    assert len(FIXTURES) == 10
    expected = [load(p)["expected"] for p in FIXTURES]
    assert expected.count(True) == 5 and expected.count(False) == 5


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_recorded_pair(path):
    fx = load(path)
    assert set(fx) == {"name", "spec", "args", "events", "before", "after", "expected"}
    SCENE_SCHEMA.validate(fx["before"])
    SCENE_SCHEMA.validate(fx["after"])
    SPEC_SCHEMA.validate(fx["spec"])
    result = verify(fx["spec"], fx["before"], fx["after"], fx["args"], fx["events"])
    assert set(result) == {"ok", "missing", "reason"}
    assert result["ok"] is fx["expected"], result
    assert isinstance(result["missing"], list) and isinstance(result["reason"], str) and result["reason"]
    if fx["expected"]:
        assert result["reason"] == "spec satisfied"


def test_missing_lists_the_named_object_unseen_after():
    fx = by_name("fail_juice_box_unseen_after")
    result = verify(fx["spec"], fx["before"], fx["after"], fx["args"], fx["events"])
    assert result["ok"] is False and result["missing"] == ["juice_box"]
    assert "unseen after the run: juice_box" in result["reason"]


def test_hand_over_passes_although_the_object_is_gone():
    fx = by_name("pass_hand_over_with_event")
    result = verify(fx["spec"], fx["before"], fx["after"], fx["args"], fx["events"])
    assert result["ok"] is True and result["missing"] == ["pill_bottle"]
    without_event = verify(fx["spec"], fx["before"], fx["after"], fx["args"], [])
    assert without_event["ok"] is False


def test_obstacle_nudge_is_explained():
    fx = by_name("fail_obstacle_nudged_3cm")
    result = verify(fx["spec"], fx["before"], fx["after"], fx["args"], fx["events"])
    assert result["ok"] is False and result["missing"] == []
    assert "cereal_box" in result["reason"] and "3.0 cm" in result["reason"]


def test_wildcard_and_params_resolve_through_before_and_args():
    fx = by_name("pass_clear_table_wildcard")
    assert spec_objects(fx["spec"], fx["before"], {}) == ["pill_bottle", "juice_box", "tea_tin"]
    hand = by_name("pass_hand_over_with_event")
    assert spec_objects(hand["spec"], hand["before"], {"object": "tea_tin"}) == ["tea_tin"]
    assert spec_objects(hand["spec"], hand["before"], {}) == []  # a missing $param is evaluate's error
    after = copy.deepcopy(fx["after"])
    after["objects"]["tea_tin"]["seen"] = False
    assert missing_objects(fx["spec"], after, fx["before"], {}) == ["tea_tin"]


def test_task_dir_saving(tmp_path):
    fx = by_name("pass_juice_box_in_tray")
    task_dir = tmp_path / "runs" / "t-0042"
    result = verify(fx["spec"], fx["before"], fx["after"], fx["args"], fx["events"], task_dir=task_dir)
    assert result["ok"] is True
    assert json.loads((task_dir / "before.json").read_text()) == fx["before"]
    assert json.loads((task_dir / "after.json").read_text()) == fx["after"]
    record = json.loads((task_dir / "verify.json").read_text())
    assert record["ok"] is True and record["missing"] == [] and record["reason"] == "spec satisfied"
    assert record["spec"] == fx["spec"] and record["args"] == fx["args"] and record["events"] == fx["events"]
    assert abs(record["t"] - time.time()) < 60


def test_malformed_spec_is_a_failed_verification_not_a_traceback():
    fx = by_name("pass_juice_box_in_tray")
    bad = {"in_zone": {"object": "juice_box", "zone": "handoff"}}  # the handoff zone has no radius
    result = verify(bad, fx["before"], fx["after"], {}, [])
    assert (
        result["ok"] is False and result["reason"].startswith("spec error:") and "radius" in result["reason"]
    )
    unknown = {"looks_tidy": {"object": "*"}}
    assert verify(unknown, fx["before"], fx["after"], {}, [])["ok"] is False
    for not_a_spec in (None, [1, 2], "in_zone", 3, {"near": {"object": "tea_tin"}}):
        result = verify(not_a_spec, fx["before"], fx["after"], {}, [])
        assert result["ok"] is False and result["reason"].startswith("spec error:"), result


class _Tracker:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = 0

    def snapshot(self, n=10):
        self.calls += 1
        index = min(self.calls, len(self.snapshots)) - 1
        return copy.deepcopy(self.snapshots[index])


def test_take_after_waits_then_retries_until_the_object_is_seen():
    fx = by_name("pass_juice_box_in_tray")
    unseen = copy.deepcopy(fx["after"])
    unseen["objects"]["juice_box"]["seen"] = False
    tracker = _Tracker([unseen, unseen, fx["after"]])
    sleeps: list[float] = []
    after = take_after(tracker, fx["spec"], fx["args"], wait=1.0, retry=3.0, sleep=sleeps.append)
    assert after == fx["after"]
    assert tracker.calls == 3
    assert sleeps[0] == 1.0 and len(sleeps) == 3


def test_take_after_gives_up_after_the_retry_window():
    fx = by_name("fail_juice_box_unseen_after")
    tracker = _Tracker([fx["after"]])
    sleeps: list[float] = []
    t0 = time.monotonic()
    after = take_after(tracker, fx["spec"], fx["args"], wait=0.0, retry=0.3, sleep=sleeps.append)
    assert time.monotonic() - t0 < 0.3  # an injected sleep must not make the loop spin through the window
    assert sleeps == [0.2, 0.2] and tracker.calls == 3  # ceil(0.3 / 0.2) retries, then give up
    assert after["objects"]["juice_box"]["seen"] is False
    assert verify(fx["spec"], fx["before"], after, fx["args"], fx["events"])["missing"] == ["juice_box"]


def test_take_after_on_the_fake_scene_tracker():
    fx = by_name("pass_clear_table_wildcard")
    tracker = FakeSceneTracker(fx["after"])
    after = take_after(
        tracker, fx["spec"], {}, wait=0.0, retry=0.0, sleep=lambda s: None, before=fx["before"]
    )
    assert verify(fx["spec"], fx["before"], after, {}, [])["ok"] is True


# -- review round 2: a camera that stops delivering frames ------------------------------------------------


class _FlakyTracker:
    """snapshot() raises TimeoutError `failures` times, then returns `scene`."""

    def __init__(self, scene, failures):
        self.scene, self.failures, self.calls = scene, failures, 0

    def snapshot(self, n=10):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("camera delivered no frame")
        return copy.deepcopy(self.scene)


def test_take_after_retries_a_snapshot_timeout_inside_the_window():
    fx = by_name("pass_juice_box_in_tray")
    tracker = _FlakyTracker(fx["after"], 2)
    sleeps: list[float] = []
    after = take_after(tracker, fx["spec"], fx["args"], wait=0.0, retry=1.0, sleep=sleeps.append)
    assert after == fx["after"] and tracker.calls == 3 and sleeps == [0.2, 0.2]


def test_take_after_raises_when_the_camera_stays_dead():
    fx = by_name("pass_juice_box_in_tray")
    tracker = _FlakyTracker(fx["after"], 100)
    with pytest.raises(TimeoutError):
        take_after(tracker, fx["spec"], fx["args"], wait=0.0, retry=0.5, sleep=lambda s: None)
    assert tracker.calls == 4  # the first snapshot plus ceil(0.5 / 0.2) retries, no spinning


def test_a_camera_that_dies_after_the_before_snapshot_cannot_verify_a_hand_over():
    """A USB glitch during the run: the stale after-scene (everything unseen at its last pose) would
    satisfy `moved` + the event, so take_after must raise instead of returning it."""
    from vision.calibrate import load_scene_cfg, synthetic_homography
    from vision.cameras import FakeCamera, load_fake_scene
    from vision.markers import marker_table
    from vision.scene import SceneTracker

    class Dying(FakeCamera):
        dead = False

        def read(self):
            return None if self.dead else super().read()

    cfg, table = load_scene_cfg(), marker_table()
    cam = Dying(load_fake_scene(), scene_cfg=cfg, objects=table)
    tracker = SceneTracker(cam, synthetic_homography(cfg), table, cfg)
    before = tracker.snapshot(10)
    assert before["objects"]["pill_bottle"]["seen"]
    cam.dead = True
    fx = by_name("pass_hand_over_with_event")
    with pytest.raises(TimeoutError):
        take_after(tracker, fx["spec"], fx["args"], wait=0.0, retry=0.3, sleep=lambda s: None, before=before)


# -- hardening round 3 (DESIGN 9.9): the retry window runs on an injected clock, never on a spinning loop --


def test_take_after_runs_its_window_on_the_injected_clock():
    fx = by_name("fail_juice_box_unseen_after")  # juice_box never comes back into view
    clock = [100.0]
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    tracker = _Tracker([fx["after"]])
    t0 = time.monotonic()
    after = take_after(
        tracker, fx["spec"], fx["args"], wait=1.0, retry=3.0, sleep=sleep, clock=lambda: clock[0]
    )
    assert time.monotonic() - t0 < 1.0  # 4 s of clock time, no wall-clock waiting
    assert after["objects"]["juice_box"]["seen"] is False
    assert sleeps[0] == 1.0 and sleeps[1:] == [0.2] * 15 and tracker.calls == 16  # ceil(3.0 / 0.2) retries
    assert clock[0] == pytest.approx(104.0)


def test_take_after_with_only_a_sleep_injected_advances_its_own_clock():
    fx = by_name("fail_juice_box_unseen_after")
    tracker = _Tracker([fx["after"]])
    t0 = time.monotonic()
    take_after(tracker, fx["spec"], fx["args"], wait=1.0, retry=3.0, sleep=lambda s: None)
    assert tracker.calls == 16 and time.monotonic() - t0 < 1.0  # the no-op sleeps walk the window
    frozen = _Tracker([fx["after"]])  # a clock that never moves: the retry cap still ends the loop
    take_after(frozen, fx["spec"], fx["args"], wait=0.0, retry=3.0, sleep=lambda s: None, clock=lambda: 0.0)
    assert frozen.calls == 16


def test_take_after_counts_slow_snapshots_against_the_window():
    fx = by_name("fail_juice_box_unseen_after")
    clock = [0.0]

    class SlowTracker(_Tracker):
        def snapshot(self, n=10):
            clock[0] += 2.0  # a 10-frame snapshot on a starved camera
            return super().snapshot(n)

    tracker = SlowTracker([fx["after"]])
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    take_after(tracker, fx["spec"], fx["args"], wait=0.0, retry=3.0, sleep=sleep, clock=lambda: clock[0])
    assert tracker.calls == 2 and sleeps == [0.2]  # 2.0 s < 3.0: one retry; 4.2 s: give up
