"""A6 single-joint jog, A11 gripper measurement and A13 Esc-latency helpers of arm/driver.py.

FakeDriver only: no hardware, no serial port, no lerobot import. The CLI is exercised in-process through
`arm.driver.main` with the prompts scripted through `arm.driver._ask`.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import pytest

from arm import driver
from arm.driver import JOINTS, MARGIN_RAD, EscWatcher, FakeDriver, reset_driver

Q0 = [0.0, -1.0, 0.5, 1.0, 0.0]
OBJECTS = (
    "10: {name: pill_bottle, kind: object, shape: cylinder, size: [0.035, 0.035], h: 0.07, held_min: 8}\n"
    "11: {name: juice_box, kind: object, shape: box, size: [0.063, 0.04], h: 0.105, held_min: 8}\n"
    "12: {name: tea_tin, kind: object, shape: cylinder, size: [0.04, 0.04], h: 0.09, held_min: 8}\n"
    "30: {name: cereal_box, kind: obstacle, shape: box, size: [0.19, 0.06], h: 0.2}\n"
    "40: {name: hand, kind: hand, h: 0.03}\n"
)


@pytest.fixture(autouse=True)
def pinned_config(monkeypatch, tmp_path):
    """Identity joint map, MJCF limits, the kit's objects, evidence under tmp_path; no process singleton."""
    monkeypatch.setattr(driver, "LIMITS_PATH", tmp_path / "no_limits.yaml")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", tmp_path / "no_joint_map.yaml")
    objects = tmp_path / "objects.yaml"
    objects.write_text(OBJECTS, encoding="utf-8")
    monkeypatch.setattr(driver, "OBJECTS_PATH", objects)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    reset_driver()
    yield
    reset_driver()


def scripted(answers: list[str]):
    """An `ask` that hands out the answers in order, then blanks (EOF)."""
    queue = iter(answers)
    return lambda prompt: next(queue, "")


# -- A11 pure rules ------------------------------------------------------------------------------------------


def test_held_min_from_readings_is_half_the_gap_to_the_empty_close():
    assert driver.held_min_from_readings([25, 30], 0) == pytest.approx(12.5)
    assert driver.held_min_from_readings([30, 25], 5) == pytest.approx(10.0)
    assert driver.held_min_from_readings([25.0], 0.0) == pytest.approx(12.5)
    # an inverted map (closed_pct 100, open_pct 0): the open side runs downward
    assert driver.held_min_from_readings([75, 70], 100, open_pct=0) == pytest.approx(12.5)


@pytest.mark.parametrize(
    "closed, empty, kwargs",
    [
        ([5, 30], 5, {}),  # smallest reading equals the empty close: not detected
        ([2, 30], 5, {}),  # below it
        ([], 0, {}),
        ([math.nan, 30], 0, {}),
        ([25, 30], math.inf, {}),
        ([100, 105], 100, {"open_pct": 0}),  # inverted map: readings on the closed side
    ],
)
def test_held_min_from_readings_refuses_undetected_or_bad_readings(closed, empty, kwargs):
    with pytest.raises(ValueError):
        driver.held_min_from_readings(closed, empty, **kwargs)


def test_objects_fit_lists_the_misfits_of_the_jaw_rule():
    # juice_box and tea_tin are 40 mm across the grasp: a 55 mm jaw leaves 35 mm, a 70 mm jaw 50 mm
    assert driver.objects_fit(55) == ["juice_box", "tea_tin"]
    assert "juice_box" in driver.objects_fit(55)
    assert driver.objects_fit(70) == []
    assert driver.objects_fit(55.0 + 1e-9) == ["juice_box", "tea_tin"]  # pill_bottle's 35 mm sits on the line


def test_objects_fit_height_rule_and_explicit_objects():
    objects = {
        "tall": {"kind": "object", "size": [0.03, 0.03], "h": 0.15},
        "short": {"kind": "object", "size": [0.03, 0.03], "h": 0.05},
        "fine": {"kind": "object", "size": [0.03, 0.03], "h": 0.08},
        "cereal_box": {"kind": "obstacle", "size": [0.19, 0.06], "h": 0.2},  # obstacles are ignored
        "hand": {"kind": "hand", "h": 0.03},
    }
    assert driver.objects_fit(70, objects) == ["short", "tall"]
    for bad in (0, -5, math.nan, math.inf):
        with pytest.raises(ValueError):
            driver.objects_fit(bad, objects)
    with pytest.raises(ValueError, match="needs size"):
        driver.objects_fit(70, {"odd": {"kind": "object", "h": 0.08}})


# -- A11 measurement session on the fake ---------------------------------------------------------------------


def test_fake_driver_contact_stops_a_close_but_not_an_open():
    d = FakeDriver(q0=Q0, gripper0=1.0)
    d.contact_gripper = 0.25
    d.command(Q0, 0.0)
    assert d.read()[1] == pytest.approx(0.25)
    assert d.commands[-1][1] == 0.0  # what was commanded is still recorded
    d.command(Q0, 1.0)
    assert d.read()[1] == 1.0
    d.contact_gripper = None
    d.command(Q0, 0.0)
    assert d.read()[1] == 0.0


def test_measure_grip_fake_session_readings_and_report():
    d = FakeDriver(q0=Q0, gripper0=1.0)
    ask = scripted(["62", "", "y", "", "n"])  # jaw opening, then (place, held?) twice
    report = driver.measure_grip(d, objects=["pill_bottle"], grasps=2, ask=ask, fake_contact_pct=25)
    # every gripper command re-sends the joints read at the start
    assert d.commands and all(q == pytest.approx(Q0) for q, _ in d.commands)
    grippers = [g for _, g in d.commands]
    # open, empty close, re-open, then per grasp: open, close, hold at contact minus 3, re-open
    assert grippers == pytest.approx([1.0, 0.0, 1.0, 1.0, 0.0, 0.22, 1.0, 1.0, 0.0, 0.22, 1.0])
    first_zero = grippers.index(0.0)
    assert grippers[first_zero + 1] == 1.0  # the empty close is re-opened at once
    assert grippers[-1] == 1.0 and d.read()[1] == 1.0
    assert d.contact_gripper is None
    assert report["mode"] == "fake" and report["stopped"] is False and report["problems"] == []
    assert report["jaw_open_mm"] == 62.0
    assert report["open_reading_pct"] == pytest.approx(100.0)
    assert report["empty_closed_pct"] == pytest.approx(0.0)
    assert report["q"] == pytest.approx(Q0) and report["grasps"] == 2
    entry = report["objects"]["pill_bottle"]
    assert entry["closed_pct"] == pytest.approx([25.0, 25.0])
    assert entry["held"] == [True, False]
    assert entry["held_min"] == pytest.approx(12.5)
    assert report["lerobot_id"] == "my_awesome_follower_arm" and report["calibration_sha256"] is None
    datetime.fromisoformat(report["captured_at"])


def test_measure_grip_blank_jaw_answer_and_misfits_become_problems():
    d = FakeDriver(q0=Q0)
    report = driver.measure_grip(d, objects=["juice_box"], grasps=1, ask=scripted([""]), fake_contact_pct=25)
    assert report["jaw_open_mm"] is None
    assert any("jaw opening not recorded" in p for p in report["problems"])
    assert report["objects"]["juice_box"]["held"] == [None]  # unanswered, not a verdict
    d = FakeDriver(q0=Q0)
    report = driver.measure_grip(
        d, objects=["juice_box"], grasps=1, ask=scripted(["55"]), fake_contact_pct=25
    )
    assert report["jaw_open_mm"] == 55.0
    assert [p for p in report["problems"] if p.startswith("juice_box: width 40 mm")]
    assert [p for p in report["problems"] if p.startswith("tea_tin: width 40 mm")]
    # the object was not detected (the fake jaws close fully): a problem entry, no held_min
    d = FakeDriver(q0=Q0)
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=1, ask=scripted(["62"]), fake_contact_pct=0
    )
    assert report["objects"]["pill_bottle"]["held_min"] is None
    assert any(p.startswith("pill_bottle: smallest closed reading 0.0") for p in report["problems"])


def test_measure_grip_esc_mid_session_keeps_the_partial_report():
    d = FakeDriver(q0=Q0)
    seen: list[str] = []

    def ask(prompt: str) -> str:
        seen.append(prompt)
        if prompt.startswith("place pill_bottle") and sum(p.startswith("place") for p in seen) == 2:
            d.freeze()  # Esc lands while the human is placing the object for grasp 2
            return ""
        return "62" if prompt.startswith("jaw") else "y"

    report = driver.measure_grip(
        d, objects=["pill_bottle", "tea_tin"], grasps=3, ask=ask, fake_contact_pct=25
    )
    assert report["stopped"] is True and d.stopped
    assert report["jaw_open_mm"] == 62.0 and report["empty_closed_pct"] == pytest.approx(0.0)
    entry = report["objects"]["pill_bottle"]
    assert entry["closed_pct"] == pytest.approx([25.0]) and entry["held"] == [True]
    assert entry["held_min"] == pytest.approx(12.5)  # from what was measured before the stop
    assert "tea_tin" not in report["objects"]
    assert any(
        "Esc/STOP at the placement prompt of grasp 2 of 3 on pill_bottle" in p for p in report["problems"]
    )
    assert d.contact_gripper is None
    assert all(q == pytest.approx(Q0) for q, _ in d.commands)


def test_measure_grip_esc_during_the_close_is_reported():
    d = FakeDriver(q0=Q0)
    original = d.close_gripper_hold

    def close_then_esc(*args, **kwargs):
        reading = original(*args, **kwargs)
        d.freeze()
        return reading

    d.close_gripper_hold = close_then_esc  # type: ignore[method-assign]
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=2, ask=scripted(["62"]), fake_contact_pct=25
    )
    assert report["stopped"] and report["objects"]["pill_bottle"]["closed_pct"] == []
    assert any("during the close of grasp 1 of 2 on pill_bottle" in p for p in report["problems"])


def test_measure_grip_refuses_bad_input():
    d = FakeDriver(q0=Q0)
    with pytest.raises(ValueError):
        driver.measure_grip(d, objects=[], grasps=1, ask=scripted([]))
    with pytest.raises(ValueError):
        driver.measure_grip(d, objects=["pill_bottle"], grasps=0, ask=scripted([]))
    d.is_fake = False  # a real driver never gets a faked contact reading
    with pytest.raises(ValueError, match="FakeDriver only"):
        driver.measure_grip(d, objects=["pill_bottle"], grasps=1, ask=scripted([]), fake_contact_pct=25)
    assert d.commands == []


def test_measure_grip_cli_fake_writes_the_report_and_the_yaml_lines(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(driver, "_ask", scripted(["62", "", "y", "", "y"]))
    out = tmp_path / "evidence" / "grip.json"
    rc = driver.main(
        ["--fake", "--measure-grip", "--objects", "pill_bottle", "--grasps", "2", "--out", str(out)]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["objects"]["pill_bottle"]["closed_pct"] == [25.0, 25.0]
    assert report["objects"]["pill_bottle"]["held_min"] == 12.5 and report["empty_closed_pct"] == 0.0
    text = capsys.readouterr().out
    assert "gripper: {closed_pct: 0, open_pct: 100, empty_closed_pct: 0.0, jaw_open_mm: 62}" in text
    assert "pill_bottle: held_min: 12.5" in text
    assert "fake measure-grip ok (11 commands)" in text
    # an existing --out is refused before anything runs
    before = out.read_bytes()
    monkeypatch.setattr(driver, "_ask", lambda prompt: pytest.fail("prompted although --out exists"))
    assert driver.main(["--fake", "--measure-grip", "--objects", "pill_bottle", "--out", str(out)]) == 2
    assert out.read_bytes() == before
    assert "never overwritten" in capsys.readouterr().err


def test_measure_grip_cli_default_out_and_objects(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(driver, "_ask", lambda prompt: "")
    assert driver.main(["--fake", "--measure-grip", "--grasps", "1", "--fake-contact", "30"]) == 0
    files = sorted((tmp_path / "runs" / "calibration").glob("gripper-*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding="utf-8"))
    assert sorted(report["objects"]) == ["juice_box", "pill_bottle", "tea_tin"]  # every kind: object on file
    assert report["objects"]["tea_tin"]["closed_pct"] == [30.0]
    assert report["objects"]["tea_tin"]["held_min"] == 15.0
    assert "held_min: 15.0" in capsys.readouterr().out
    # a second run never reuses the file
    assert driver.main(["--fake", "--measure-grip", "--grasps", "1"]) == 0
    assert len(list((tmp_path / "runs" / "calibration").glob("gripper-*.json"))) == 2


def test_measure_grip_cli_esc_writes_the_partial_report_and_exits_1(monkeypatch, tmp_path, capsys):
    made: dict[str, FakeDriver] = {}

    class Recording(FakeDriver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made["driver"] = self

    placements = 0

    def ask(prompt: str) -> str:
        nonlocal placements
        if prompt.startswith("place"):
            placements += 1
            if placements == 2:
                made["driver"].freeze()
        return "62" if prompt.startswith("jaw") else "y"

    monkeypatch.setattr(driver, "FakeDriver", Recording)
    monkeypatch.setattr(driver, "_ask", ask)
    out = tmp_path / "partial.json"
    rc = driver.main(
        ["--fake", "--measure-grip", "--objects", "pill_bottle", "--grasps", "3", "--out", str(out)]
    )
    assert rc == 1 and made["driver"].stopped
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["stopped"] is True
    assert report["objects"]["pill_bottle"]["closed_pct"] == [25.0]
    text = capsys.readouterr().out
    assert "fake measure-grip stopped" in text
    # a partial session prints no paste lines: its readings are not a measurement
    assert "no lines to paste" in text and "Paste into" not in text and "gripper: {" not in text


def _esc_on_sleep(d: FakeDriver, seconds_wanted: float, jaws_at: float):
    """A `_sleep` stand-in that, on the wait of `seconds_wanted`, leaves the jaws at `jaws_at` and freezes."""

    def sleep(seconds: float) -> None:
        if seconds == seconds_wanted and not d.stopped:
            d.gripper_reading_override = jaws_at  # the jaws are part-way when Esc lands
            d.freeze()

    return sleep


def test_measure_grip_esc_during_the_empty_close_discards_the_reading(capsys):
    """Review lead (c): a mid-travel reading after Esc is never stored or printed as empty_closed_pct."""
    d = FakeDriver(q0=Q0, gripper0=1.0)
    d._sleep = _esc_on_sleep(d, 0.5, jaws_at=0.4)  # type: ignore[method-assign]
    report = driver.measure_grip(d, objects=["pill_bottle"], grasps=1, ask=scripted(["62"]))
    assert report["stopped"] is True and d.stopped
    assert report["empty_closed_pct"] is None
    assert report["open_reading_pct"] == pytest.approx(100.0) and report["jaw_open_mm"] == 62.0
    assert any("during the empty close" in p and "reading discarded" in p for p in report["problems"])
    assert report["objects"] == {}
    driver._print_grip_summary(report, Path("grip.json"))
    text = capsys.readouterr().out
    assert "empty-closed reading n/a" in text and "40.0" not in text and "gripper: {" not in text


def test_measure_grip_esc_during_the_open_wait_discards_the_reading_and_skips_the_prompt():
    d = FakeDriver(q0=Q0, gripper0=0.0)
    d._sleep = _esc_on_sleep(d, 1.0, jaws_at=0.5)  # type: ignore[method-assign]
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=1, ask=lambda p: pytest.fail(f"prompted a frozen arm: {p}")
    )
    assert report["stopped"] is True and report["open_reading_pct"] is None
    assert any("while the jaws were opening" in p for p in report["problems"])


def test_measure_grip_measures_a_name_listed_twice_once():
    d = FakeDriver(q0=Q0)
    report = driver.measure_grip(
        d,
        objects=["pill_bottle", "pill_bottle"],
        grasps=2,
        ask=scripted(["62", "", "y", "", "y"]),
        fake_contact_pct=25,
    )
    assert list(report["objects"]) == ["pill_bottle"]
    assert report["objects"]["pill_bottle"]["closed_pct"] == pytest.approx([25.0, 25.0])
    assert len(d.commands) == 11  # open, empty close, re-open, then 4 per grasp


# -- the commandable band: the A6 jog and the A11 session refuse a start pose outside it ---------------------


def _outside_band(joint: str, deg_outside: float) -> list[float]:
    """Q0 with `joint` `deg_outside` degrees below the low edge of the band (limits minus 3 degrees)."""
    lo, _ = driver.limits()[joint]
    q = list(Q0)
    q[JOINTS.index(joint)] = lo + MARGIN_RAD - math.radians(deg_outside)
    return q


def test_band_violations_names_each_joint_and_its_excess():
    assert driver.band_violations(Q0) == {}
    q = _outside_band("shoulder_lift", 6.65)
    lo, _ = driver.limits()["wrist_flex"]
    q[3] = lo + MARGIN_RAD - math.radians(9.89)
    out = driver.band_violations(q)
    assert list(out) == ["shoulder_lift", "wrist_flex"]
    assert out["shoulder_lift"] == pytest.approx(6.65) and out["wrist_flex"] == pytest.approx(9.89)
    # within the tolerance the pose is accepted (clip moves it at most that much)
    assert driver.band_violations(_outside_band("elbow_flex", driver.BAND_TOLERANCE_DEG - 0.1)) == {}
    assert "elbow_flex" in driver.band_violations(
        _outside_band("elbow_flex", driver.BAND_TOLERANCE_DEG + 0.1)
    )


def test_jog_joint_refuses_a_start_pose_outside_the_band_before_moving():
    """Review lead (a): jogging shoulder_pan from such a pose would also move shoulder_lift."""
    d = FakeDriver(q0=_outside_band("shoulder_lift", 6.65))
    with pytest.raises(
        ValueError, match=r"outside the commandable band .* shoulder_lift 6\.7 deg.*nothing moved"
    ):
        driver.jog_joint(d, "shoulder_pan", deg=20, hold_s=0, realtime=False)
    assert d.commands == []
    # the jogged joint itself outside the band is refused the same way, even when the increment leads inward
    d = FakeDriver(q0=_outside_band("wrist_flex", 9.89))
    with pytest.raises(ValueError, match="wrist_flex 9.9 deg"):
        driver.jog_joint(d, "wrist_flex", deg=20, hold_s=0, realtime=False)
    assert d.commands == []


def test_measure_grip_refuses_a_start_pose_outside_the_band_before_moving():
    d = FakeDriver(q0=_outside_band("elbow_flex", 2.92))
    with pytest.raises(ValueError, match="measure-grip: .*elbow_flex 2.9 deg"):
        driver.measure_grip(
            d, objects=["pill_bottle"], grasps=1, ask=lambda p: pytest.fail("prompted"), fake_contact_pct=25
        )
    assert d.commands == []


def test_cli_band_refusals_exit_2_and_write_nothing(monkeypatch, tmp_path, capsys):
    class OutOfBand(FakeDriver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, q0=_outside_band("shoulder_lift", 6.65), **kwargs)

    monkeypatch.setattr(driver, "FakeDriver", OutOfBand)
    monkeypatch.setattr(driver, "_ask", lambda prompt: pytest.fail("prompted"))
    assert driver.main(["--fake", "--jog-joint", "shoulder_pan", "--deg", "20", "--hold", "0"]) == 2
    assert "refused: jog-joint shoulder_pan" in capsys.readouterr().err
    out = tmp_path / "grip.json"
    assert driver.main(["--fake", "--measure-grip", "--objects", "pill_bottle", "--out", str(out)]) == 2
    assert "refused: measure-grip" in capsys.readouterr().err
    assert not out.exists()


def test_jog_joint_reads_again_at_the_end_of_the_hold():
    d = FakeDriver(q0=Q0)
    result = driver.jog_joint(d, "elbow_flex", deg=5, hold_s=0.5, realtime=False)
    assert result["ok"] is True
    assert result["delta_settled_rad"] == pytest.approx(math.radians(5))
    assert result["lerobot_deg_settled"] == pytest.approx(result["lerobot_deg_at"])


# -- A6 single-joint increment jog -------------------------------------------------------------------------


def test_jog_joint_moves_only_that_joint_and_returns():
    d = FakeDriver(q0=Q0, gripper0=0.7)
    t0 = time.perf_counter()
    result = driver.jog_joint(d, "shoulder_lift", deg=20, hold_s=10, realtime=False, fk=None)
    assert time.perf_counter() - t0 < 2.0  # the fake never sleeps through the hold
    assert result["ok"] is True
    assert d.commands
    for q, g in d.commands:
        assert q[0] == Q0[0] and q[2:] == Q0[2:] and g == 0.7
    peak = max(q[1] for q, _ in d.commands)
    assert peak == pytest.approx(Q0[1] + math.radians(20), abs=1e-6)
    assert d.read()[0] == pytest.approx(Q0)
    steps = [abs(b[0][1] - a[0][1]) for a, b in zip(d.commands, d.commands[1:], strict=False)]
    assert max(steps) <= math.radians(1.0) + 1e-9  # one degree per tick, under the driver's 2 degree clip
    assert len(d.commands) == 40  # 20 up, 19 down, back to the start
    assert result["expected_delta_rad"] == pytest.approx(math.radians(20))
    assert result["delta_at_rad"] == pytest.approx(0.349066, abs=1e-6)
    assert result["lerobot_deg_at"] - result["lerobot_deg_before"] == pytest.approx(20.0)
    assert result["delta_after_rad"] == pytest.approx(0.0, abs=1e-9)
    assert result["q_expected"][1] == pytest.approx(Q0[1] + math.radians(20))
    assert result["tip_before"] is None and result["tip_at"] is None


def test_jog_joint_reports_lerobot_degrees_with_a_sign_minus_one_map(tmp_path, monkeypatch):
    path = tmp_path / "joint_map.yaml"
    path.write_text("shoulder_lift: {sign: -1, offset_deg: 0.0}\n", encoding="utf-8")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", path)
    d = FakeDriver(q0=Q0)
    result = driver.jog_joint(d, "shoulder_lift", deg=20, hold_s=0, realtime=False)
    assert result["ok"]
    assert result["delta_at_rad"] == pytest.approx(math.radians(20), abs=1e-6)  # our units: +20 degrees
    assert result["lerobot_deg_at"] - result["lerobot_deg_before"] == pytest.approx(-20.0)  # the servo's
    assert result["lerobot_deg_before"] == pytest.approx(-math.degrees(Q0[1]))


def test_jog_joint_negative_increment_and_fractional_degrees():
    d = FakeDriver(q0=Q0)
    result = driver.jog_joint(d, "wrist_roll", deg=-2.5, hold_s=0, realtime=False)
    assert result["ok"]
    assert min(q[4] for q, _ in d.commands) == pytest.approx(Q0[4] - math.radians(2.5))
    assert len(d.commands) == 6  # 1, 2, 2.5, 2, 1, 0 degrees
    assert d.read()[0] == pytest.approx(Q0)


def test_jog_joint_refuses_an_increment_that_leaves_the_range():
    hi = driver.limits()["shoulder_pan"][1]
    start = list(Q0)
    start[0] = hi - MARGIN_RAD - math.radians(10)  # 10 degrees of room inside the margin
    d = FakeDriver(q0=start)
    with pytest.raises(ValueError, match="leaves the calibrated range minus 3 degrees"):
        driver.jog_joint(d, "shoulder_pan", deg=20, hold_s=0, realtime=False)
    assert d.commands == []
    assert driver.jog_joint(d, "shoulder_pan", deg=10, hold_s=0, realtime=False)[
        "ok"
    ]  # exactly to the margin
    assert max(q[0] for q, _ in d.commands) == pytest.approx(hi - MARGIN_RAD)


def test_jog_joint_refuses_bad_arguments():
    d = FakeDriver(q0=Q0)
    for kwargs in (
        {"deg": 0},
        {"deg": math.nan},
        {"step_deg": 0},
        {"step_deg": 3.0},  # above the driver's 2 degree clip
        {"hold_s": -1},
    ):
        with pytest.raises(ValueError):
            driver.jog_joint(d, "elbow_flex", realtime=False, **kwargs)
    with pytest.raises(ValueError, match="one of"):
        driver.jog_joint(d, "gripper", realtime=False)
    assert d.commands == []


def test_jog_joint_stops_when_frozen_before_or_during_the_hold():
    d = FakeDriver(q0=Q0)
    d.freeze()
    result = driver.jog_joint(d, "elbow_flex", deg=5, hold_s=0, realtime=False)
    assert result["ok"] is False and d.commands == [] and "q_at" not in result
    d = FakeDriver(q0=Q0)
    d._sleep = lambda seconds: d.freeze()  # Esc lands during the first hold slice
    result = driver.jog_joint(d, "elbow_flex", deg=5, hold_s=1, realtime=False)
    assert result["ok"] is False and d.stopped
    assert result["delta_at_rad"] == pytest.approx(math.radians(5))
    assert d.commands[-1][0][2] == pytest.approx(Q0[2] + math.radians(5))  # held at the increment, no return
    assert "q_after" not in result


def test_jog_joint_prints_lerobot_degrees_mapped_radians_and_tip(capsys):
    d = FakeDriver(q0=Q0)
    result = driver.jog_joint(
        d, "shoulder_lift", deg=20, hold_s=0, realtime=False, fk=lambda q: (0.1 + q[1], 0.0, 0.2)
    )
    assert result["tip_before"] == pytest.approx((0.1 + Q0[1], 0.0, 0.2))
    assert result["tip_expected"] == pytest.approx((0.1 + Q0[1] + math.radians(20), 0.0, 0.2))
    assert result["tip_at"] == pytest.approx(result["tip_expected"])
    text = capsys.readouterr().out
    assert "expected model increment +0.349 rad" in text
    assert "before: LeRobot -57.30 deg | mapped -1.0000 rad | tip (-0.900, 0.000, 0.200) m" in text
    assert "at increment (measured delta +0.3491 rad): LeRobot -37.30 deg | mapped -0.6509 rad" in text
    assert "after return (delta from start +0.0000 rad)" in text
    assert "Only shoulder_lift moves." in text


def test_jog_joint_cli_fake(capsys):
    rc = driver.main(["--fake", "--jog-joint", "shoulder_lift", "--deg", "20", "--hold", "0"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "LeRobot" in text and "mapped" in text and "rad" in text
    assert "tip (" in text  # arm.ik.fk_tip through the lazy import
    assert "fake jog-joint ok (40 commands)" in text


def test_jog_joint_cli_limit_refusal_exits_2(monkeypatch, capsys):
    hi = driver.limits()["shoulder_pan"][1]
    near = list(Q0)
    near[0] = hi - MARGIN_RAD - math.radians(5)
    monkeypatch.setattr(driver, "FakeDriver", lambda *a, **k: FakeDriver(q0=near))
    assert driver.main(["--fake", "--jog-joint", "shoulder_pan", "--deg", "20", "--hold", "0"]) == 2
    assert "refused" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--fake", "--jog", "--jog-joint", "shoulder_lift"],
        ["--fake", "--jog-joint", "shoulder_lift", "--measure-grip"],
        ["--fake", "--jog-joint", "gripper"],
        ["--fake", "--jog", "--hold", "3"],
        ["--fake", "--jog", "--objects", "pill_bottle"],
        ["--fake", "--jog", "--grasps", "2"],
        ["--fake", "--jog", "--out", "x.json"],
        ["--real", "--measure-grip", "--fake-contact", "30"],
        ["--real", "--fake", "--jog"],
        ["--fake", "--jog-joint", "shoulder_lift", "--deg", "0"],
        ["--fake", "--jog-joint", "shoulder_lift", "--hold", "-1"],
        ["--fake", "--measure-grip", "--grasps", "0"],
        ["--fake", "--measure-grip", "--fake-contact", "120"],
    ],
)
def test_cli_misuse_exits_2(argv, monkeypatch):
    monkeypatch.setattr(driver, "_ask", lambda prompt: pytest.fail("a misuse must not reach a prompt"))
    with pytest.raises(SystemExit) as exc:
        driver.main(argv)
    assert exc.value.code == 2


def test_cli_without_an_action_prints_help(capsys):
    assert driver.main(["--fake"]) == 2
    assert "--measure-grip" in capsys.readouterr().out


class _StubWatcher:
    """EscWatcher stand-in: no keyboard; a test marks the press itself and calls the callback."""

    def __init__(self, callback=None, **kwargs):
        self.callback = callback
        self.pressed = False
        self.pressed_at: float | None = None
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_real_flow_runs_each_action_after_the_confirmation_and_records_esc(monkeypatch, tmp_path, capsys):
    """--real with a stubbed Driver and watcher: STOP bound before the prompt, connect after it, the action,
    the Esc latency print and its JSON line, torque kept on disconnect."""
    made: list[FakeDriver] = []
    watchers: list[_StubWatcher] = []
    order: list[str] = []

    def make_driver(**kwargs):
        d = FakeDriver(q0=Q0)
        d.disconnect()  # a real Driver starts unconnected; connect() powers the arm
        original_connect = d.connect

        def connect():
            order.append("connect")
            original_connect()

        d.connect = connect  # type: ignore[method-assign]
        made.append(d)
        return d

    def make_watcher(callback=None, **kwargs):
        w = _StubWatcher(callback)
        watchers.append(w)
        return w

    def countdown(seconds, watcher):
        assert (
            seconds == 3 and watcher.callback is not None and watcher.started
        )  # STOP bound before the prompt
        order.append("confirm")
        return True

    monkeypatch.setattr(driver, "Driver", make_driver)
    monkeypatch.setattr(driver, "EscWatcher", make_watcher)
    monkeypatch.setattr(driver, "_countdown", countdown)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr(driver.time, "sleep", lambda s: None)  # the 30 Hz ramps do not wait in the test

    assert driver.main(["--real", "--jog", "--deg", "3"]) == 0
    assert order == ["confirm", "connect"]
    d = made[-1]
    assert d.commands and not d.stopped and not d.connected and d.torque  # the port closed, torque kept
    assert watchers[-1].started and watchers[-1].stopped
    assert not (tmp_path / "runs" / "calibration" / "esc-latency.jsonl").exists()  # no Esc, no record

    def esc_during_hold(seconds):
        w = watchers[-1]
        if not w.pressed:
            tick = time.monotonic()
            while time.monotonic() <= tick:  # the next clock tick (~15.6 ms on Windows): after connect
                pass
            w.pressed, w.pressed_at = True, time.monotonic()
            w.callback()

    monkeypatch.setattr(FakeDriver, "_sleep", lambda self, seconds: esc_during_hold(seconds))
    assert driver.main(["--real", "--jog-joint", "elbow_flex", "--deg", "5", "--hold", "1"]) == 1
    d = made[-1]
    assert d.stopped and d.torque and not d.connected
    assert d.commands[-1][0][2] == pytest.approx(Q0[2] + math.radians(5))  # frozen at the increment
    out = capsys.readouterr().out
    assert "Esc -> freeze command:" in out and "Arm is FROZEN with torque on" in out
    log = tmp_path / "runs" / "calibration" / "esc-latency.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["source"] == "arm.driver --jog-joint elbow_flex"
    assert lines[0]["latency_ms"] is not None and 0.0 <= lines[0]["latency_ms"] < 200.0
    assert lines[0]["extra"] == {"mode": "real", "deg": 5.0, "frozen": True}

    monkeypatch.setattr(FakeDriver, "_sleep", lambda self, seconds: None)
    monkeypatch.setattr(driver, "_ask", scripted(["62", "", "y"]))
    report_path = tmp_path / "real-grip.json"
    rc = driver.main(
        ["--real", "--measure-grip", "--objects", "pill_bottle", "--grasps", "1", "--out", str(report_path)]
    )
    assert rc == 0 and order[-2:] == ["confirm", "connect"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["lerobot_id"] == "my_awesome_follower_arm" and report["calibration_sha256"] is None
    assert report["jaw_open_mm"] == 62.0 and report["objects"]["pill_bottle"]["held"] == [True]
    assert all(q == pytest.approx(Q0) for q, _ in made[-1].commands)


# -- A13 Esc latency helpers -------------------------------------------------------------------------------


def test_esc_latency_ms_none_before_a_press_and_small_after():
    d = FakeDriver(q0=Q0)
    keys = iter([EscWatcher.ESC])
    w = EscWatcher(d.freeze, poll=0.005, key_source=lambda: next(keys, None))
    assert driver.esc_latency_ms(w, d) is None
    assert driver.esc_latency_ms(None, d) is None
    w.start()
    deadline = time.time() + 2.0
    while not d.stopped and time.time() < deadline:
        time.sleep(0.005)
    w.stop()
    ms = driver.esc_latency_ms(w, d)
    assert ms is not None and 0.0 <= ms < 200.0
    # a freeze that predates the press is not this press's latency
    w.pressed_at = d.frozen_at + 1.0
    assert driver.esc_latency_ms(w, d) is None
    d.frozen_at = w.pressed_at + 0.5
    assert driver.esc_latency_ms(w, d) == pytest.approx(500.0)


def test_record_esc_latency_appends_json_lines(tmp_path):
    path = tmp_path / "deep" / "er" / "esc.jsonl"
    assert (
        driver.record_esc_latency(path, source="test", latency_ms=12.5, extra={"joint": "elbow_flex"}) == path
    )
    driver.record_esc_latency(path, source="test again", latency_ms=None)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0] == {
        "t": lines[0]["t"],
        "source": "test",
        "phase": "session",
        "latency_ms": 12.5,
        "extra": {"joint": "elbow_flex"},
    }
    assert lines[1]["source"] == "test again" and lines[1]["latency_ms"] is None and lines[1]["extra"] is None
    stamp = datetime.fromisoformat(lines[0]["t"])
    assert stamp.tzinfo is not None  # local time with its UTC offset
    with pytest.raises(ValueError):
        driver.record_esc_latency(path, source="bad", latency_ms=math.nan)
    with pytest.raises(ValueError, match="phase"):
        driver.record_esc_latency(path, source="bad", latency_ms=1.0, phase="motion")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    driver.record_esc_latency(path, source="while connecting", latency_ms=600.0, phase="connect")
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["phase"] == "connect"


def test_esc_phase_splits_presses_at_the_end_of_connect():
    class W:
        pressed_at: float | None = 10.0

    assert driver.esc_phase(W(), connected_at=None) == "connect"  # connect never returned
    assert driver.esc_phase(W(), connected_at=10.5) == "connect"  # pressed while connecting
    assert driver.esc_phase(W(), connected_at=9.5) == "session"  # pressed after connect returned
    assert driver.esc_phase(W(), connected_at=10.0) == "connect"  # the same clock tick: still connecting
    W.pressed_at = None
    assert driver.esc_phase(W(), connected_at=9.5) == "session"


def test_real_flow_marks_an_esc_during_connect_as_connect_phase(monkeypatch, tmp_path, capsys):
    """Review lead (b): a press while connect() holds the lock is recorded with phase "connect", measured to
    the freeze connect applied (main never re-freezes and re-stamps it), and nothing moves."""
    made: list[FakeDriver] = []
    watchers: list[_StubWatcher] = []

    def make_driver(**kwargs):
        d = FakeDriver(q0=Q0)
        d.disconnect()
        original_connect = d.connect

        def connect():
            w = watchers[-1]
            w.pressed, w.pressed_at = True, time.monotonic()
            d._freeze_requested.set()  # the watcher's freeze() is waiting for the lock connect holds
            time.sleep(0.05)
            original_connect()  # applies the pending freeze and stamps frozen_at

        d.connect = connect  # type: ignore[method-assign]
        made.append(d)
        return d

    def make_watcher(callback=None, **kwargs):
        w = _StubWatcher(callback)
        watchers.append(w)
        return w

    monkeypatch.setattr(driver, "Driver", make_driver)
    monkeypatch.setattr(driver, "EscWatcher", make_watcher)
    monkeypatch.setattr(driver, "_countdown", lambda seconds, watcher: True)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert driver.main(["--real", "--jog-joint", "elbow_flex", "--deg", "5", "--hold", "0"]) == 1
    d = made[-1]
    assert d.stopped and d.commands == [] and d.torque
    frozen_once = d.frozen_at
    out = capsys.readouterr().out
    assert "Esc pressed while connecting" in out and "connect phase" in out
    assert "Esc -> freeze command" not in out
    log = tmp_path / "runs" / "calibration" / "esc-latency.jsonl"
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert record["phase"] == "connect" and record["latency_ms"] >= 40.0
    assert record["latency_ms"] == pytest.approx((frozen_once - watchers[-1].pressed_at) * 1000.0)


def test_cli_deg_is_refused_with_measure_grip(capsys):
    with pytest.raises(SystemExit) as exc:
        driver.main(["--fake", "--measure-grip", "--deg", "5"])
    assert exc.value.code == 2
    assert "--deg applies to --jog and --jog-joint only" in capsys.readouterr().err


def test_record_esc_latency_default_path_honours_vara_runs_dir(tmp_path):
    written = driver.record_esc_latency(source="arm.driver --jog", latency_ms=40.0)
    assert written == tmp_path / "runs" / "calibration" / "esc-latency.jsonl"
    assert json.loads(written.read_text(encoding="utf-8"))["latency_ms"] == 40.0


def test_report_path_is_new_per_call_and_refuses_an_existing_out(tmp_path):
    first = driver._report_path(None, "gripper")
    assert first.parent == tmp_path / "runs" / "calibration" and first.name.startswith("gripper-")
    first.write_text("{}", encoding="utf-8")
    second = driver._report_path(None, "gripper")
    assert second != first and not second.exists()
    chosen = tmp_path / "new" / "out.json"
    assert driver._report_path(str(chosen), "gripper") == chosen and chosen.parent.is_dir()
    with pytest.raises(FileExistsError):
        driver._report_path(str(first), "gripper")
    with pytest.raises(FileExistsError):
        driver._write_report({}, first)


def test_joint_names_are_the_five_arm_joints():
    assert JOINTS == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    assert isinstance(Path(driver.ROOT), Path)
