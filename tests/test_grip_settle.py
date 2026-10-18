"""A gripper close is read, and held, only once the jaws stop (arm/driver.py `wait_gripper_still`,
`close_gripper_hold`; the measurement session and the replay grip check use them).

Seen on the real follower on 2026-10-17: LeRobot drives the gripper with position gain 16 and a 50 % torque
cap, so the jaws take a noticeable part of a second to close. With a fixed 0.5 s wait the driver read them
mid-travel and then held them there, short of a small object. `SlowJaws` travels like the real gripper: a
fixed distance per poll toward the commanded target, stopping at an object's contact reading.
"""

from __future__ import annotations

import pytest

from arm import driver, ik, replay
from arm.driver import GRIP_POLL_S, GRIP_TIMEOUT_S, FakeDriver, reset_driver

HOME = list(ik.home_q())


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    reset_driver()
    yield
    reset_driver()


class SlowJaws(FakeDriver):
    """A FakeDriver whose jaws move `step` (0-1 units) toward the last commanded gripper per poll, stop at
    `contact_gripper` when an object is between them, and stay put once frozen."""

    def __init__(self, *args, step: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = step
        self.jaws = float(self._g)
        self.polls = 0

    def _sleep(self, seconds: float) -> None:
        self.polls += 1
        if self.stopped or not self.commands:
            return
        target = self.commands[-1][1]
        if self.contact_gripper is not None and target < self.contact_gripper:
            target = self.contact_gripper  # the object stops the jaws
        self.jaws += max(-self.step, min(self.step, target - self.jaws))

    def read(self) -> tuple[list[float], float]:
        q, _ = super().read()
        return q, self.jaws


def test_a_slow_close_on_nothing_is_read_at_the_closed_stop_not_mid_travel():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.02)  # a full close takes 50 polls, 2.5 s
    reading = d.close_gripper_hold()  # the old single 0.5 s wait read the jaws mid-travel and held them there
    assert d.grip_status == "still"
    assert reading == pytest.approx(0.0, abs=1e-9)
    assert d.polls * GRIP_POLL_S >= 2.5  # it waited for the jaws, not the clock
    assert d.commands[-1][1] == pytest.approx(0.0)  # the hold (reading minus 3 %) clamps at closed


def test_a_small_object_is_reached_and_held_just_past_its_contact_reading():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.02)
    d.contact_gripper = 0.15  # a small object: a long way to travel before contact
    reading = d.close_gripper_hold()
    assert d.grip_status == "still" and reading == pytest.approx(0.15, abs=1e-9)
    assert d.commands[-1][1] == pytest.approx(0.12)  # held 3 units (0-100 scale) past contact
    assert all(q == pytest.approx(HOME) for q, _ in d.commands)  # only the jaws moved


def test_jaws_still_moving_at_the_timeout_are_reported_and_held_so_they_never_keep_pushing():
    """Review 2026-10-17: a close left commanded at 0.0 after a timeout pushed at the torque cap with no time
    limit (through the rise and home of a missed grasp). The timeout now holds at a fresh reading minus 3,
    and `grip_status` still says "timeout" so every caller treats the grasp as unconfirmed."""
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.004)  # 0.4 % per poll: too slow to arrive in 3 s
    reading = d.close_gripper_hold()
    assert d.grip_status == "timeout"
    assert reading == pytest.approx(1.0 - 0.004 * round(GRIP_TIMEOUT_S / GRIP_POLL_S), abs=1e-6)
    assert d.commands[-1][1] == pytest.approx(reading - 0.03)  # bounded hold, not the 0.0 close


def test_an_esc_during_the_wait_ends_it_with_stopped_and_no_hold():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.02)
    original = d._sleep

    def esc_on_the_fifth_poll(seconds: float) -> None:
        original(seconds)
        if d.polls == 5:
            d.freeze()

    d._sleep = esc_on_the_fifth_poll  # type: ignore[method-assign]
    reading = d.close_gripper_hold()
    assert d.grip_status == "stopped" and d.stopped
    assert reading == pytest.approx(0.9)  # where the freeze caught the jaws
    assert d.commands[-1][1] == 0.0  # no hold command after the stop


def test_wait_gripper_still_refuses_bad_waits():
    d = FakeDriver(q0=HOME)
    for settle, timeout in ((-1.0, 3.0), (0.5, 0.0), (float("nan"), 3.0), (0.5, float("inf"))):
        with pytest.raises(ValueError):
            d.wait_gripper_still(settle=settle, timeout=timeout)


def test_an_instant_fake_still_waits_at_least_settle_and_returns_still():
    d = FakeDriver(q0=HOME, gripper0=1.0)
    d.command(HOME, 0.0)
    assert d.wait_gripper_still(settle=0.5) == 0.0 and d.grip_status == "still"


def test_measure_grip_with_slow_jaws_records_the_true_empty_close_and_contact():
    d = SlowJaws(q0=HOME, gripper0=0.0, step=0.02)
    answers = iter(["62", "", "y"])
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=1, ask=lambda prompt: next(answers, ""), fake_contact_pct=25
    )
    assert report["problems"] == [] and report["stopped"] is False
    assert report["open_reading_pct"] == pytest.approx(100.0)  # fully open before the jaw question
    assert report["empty_closed_pct"] == pytest.approx(0.0, abs=1e-6)  # closed, not a mid-travel reading
    assert report["objects"]["pill_bottle"]["closed_pct"] == pytest.approx([25.0])


def test_measure_grip_discards_a_close_that_never_stops():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.004)
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=1, ask=lambda prompt: "62" if "jaw" in prompt else "y"
    )
    assert report["empty_closed_pct"] is None and report["open_reading_pct"] is not None
    assert any("empty close: the jaws were still moving" in p for p in report["problems"])
    assert any("grasp 1 of 1 on pill_bottle: the jaws were still moving" in p for p in report["problems"])
    assert report["objects"]["pill_bottle"]["closed_pct"] == []


def test_replay_treats_a_close_that_never_stops_as_a_missed_grasp():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.004)
    traj = {
        "version": 1,
        "skill": "grip_settle_test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [
            {"kind": "path", "q": [HOME]},
            {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True},
        ],
    }
    result = replay.replay(traj, driver=d, realtime=False)
    assert result["reason"] == "grasp_missed" and not result["ok"]
    assert any("still moving after the wait; treated as not holding" in e for e in result["events"])


def test_a_timed_out_close_never_leaves_the_gripper_pushing_through_the_missed_grasp_recovery():
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.004)
    result = replay.replay(_close_traj(), driver=d, realtime=False)
    assert result["reason"] == "grasp_missed"
    after_close = [
        g for _, g in d.commands[d.commands.index(next(c for c in d.commands if c[1] < 1.0)) + 1 :]
    ]
    assert after_close and all(g > 0.5 for g in after_close)  # the bounded hold all through rise and home
    assert d.last_command[1] > 0.5


def test_a_wristband_arriving_while_the_jaws_close_freezes_them_within_a_poll():
    """Review 2026-10-17: the close's wait is variable (up to about 3 s), so the replay's per-tick hand check
    now runs on every poll of the wait instead of only before and after it."""
    import copy
    import json

    from arm import ik as ik_mod

    scene_far = json.loads(open(driver.ROOT / "config" / "fake_scene.json", encoding="utf-8").read())
    scene_far["hand"].update({"x": 0.40, "y": -0.20, "z": 0.03, "seen": True})
    tip = ik_mod.fk_tip(HOME)
    scene_near = copy.deepcopy(scene_far)
    scene_near["hand"].update({"x": tip[0], "y": tip[1], "z": tip[2] - 0.05})
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.02)
    d.contact_gripper = 0.15

    def scene():
        return scene_near if d.polls >= 12 else scene_far

    result = replay.replay(_close_traj(), driver=d, realtime=False, scene=scene)
    assert result["reason"] == "hand_too_close" and d.stopped
    assert d.polls <= 14  # frozen at the first poll that saw the hand, not when the jaws settled
    assert d.jaws > 0.15 + 0.02  # stopped short of the object
    assert any("grip close: frozen" in e for e in result["events"])


def test_a_timed_out_empty_close_prints_no_gripper_paste_line(capsys):
    d = SlowJaws(q0=HOME, gripper0=1.0, step=0.004)
    report = driver.measure_grip(
        d, objects=["pill_bottle"], grasps=1, ask=lambda p: "62" if "jaw" in p else "y"
    )
    driver._print_grip_summary(report, driver.ROOT / "grip.json")
    text = capsys.readouterr().out
    assert "gripper: {" not in text and "keep its gripper line" in text


def _close_traj() -> dict:
    return {
        "version": 1,
        "skill": "grip_settle_test",
        "args": {},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [
            {"kind": "path", "q": [HOME]},
            {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True},
        ],
    }
