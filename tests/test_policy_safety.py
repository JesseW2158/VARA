"""The optional real policy obeys the live primitive's gates, using in-memory hardware only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm import ik, real_robot, replay
from arm.driver import FakeDriver
from rl import policy_runner as pr
from rl.handoff_env import DT, MAX_STEP_RAD


class Policy:
    def __init__(self, action=(-1.0, -1.0, -1.0, -1.0)):
        self.action = action
        self.observations = []

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.observations.append(np.asarray(obs).copy())
        return self.action, None


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def near_hand():
    x, y, z = ik.fk_tip(ik.home_q())
    return {"hand": {"seen": True, "x": x, "y": y, "z": z - 0.05}}


@pytest.mark.parametrize("near", [False, True])
def test_policy_obeys_cartesian_speed_limit_without_extending_tick_budget(near):
    driver = FakeDriver(gripper0=0.23)
    policy = Policy()
    scene = near_hand() if near else {}
    assert not pr.run_policy(driver, lambda: scene, "unused.zip", policy=policy, timeout=3 * DT)
    assert len(driver.commands) == 3
    previous = np.asarray(ik.home_q())
    cap = (replay.TIP_SPEED_NEAR_HAND if near else replay.TIP_SPEED_MAX) * DT
    requested = previous.copy()
    requested[:4] -= MAX_STEP_RAD
    assert math.dist(ik.fk_tip(previous), ik.fk_tip(requested)) > cap
    for q, gripper in driver.commands:
        assert math.dist(ik.fk_tip(previous), ik.fk_tip(q)) <= cap + 1e-9
        assert np.max(np.abs(np.asarray(q) - previous)) <= MAX_STEP_RAD + 1e-9
        assert q[4] == previous[4] and gripper == 0.23
        previous = np.asarray(q)
    assert np.linalg.norm(previous - ik.home_q()) > 0.0
    # The next observation describes the action actually sent after safety reductions.
    actual_action = (np.asarray(driver.commands[0][0][:4]) - ik.home_q()[:4]) / MAX_STEP_RAD
    assert policy.observations[1][5:9] == pytest.approx(actual_action, abs=1e-6)


def test_fallback_target_still_respects_the_actual_tracked_hand():
    driver = FakeDriver(gripper0=0.23)
    pr.run_policy(
        driver,
        lambda: {"hand": {"seen": False}},
        "unused.zip",
        safety_scene=near_hand,
        policy=Policy(),
        timeout=DT,
    )
    assert len(driver.commands) == 1
    assert math.dist(ik.fk_tip(ik.home_q()), ik.fk_tip(driver.commands[0][0])) <= (
        replay.TIP_SPEED_NEAR_HAND * DT + 1e-9
    )


@pytest.mark.parametrize("action", [[math.nan] * 4, [math.inf] * 4, [0.0] * 3])
def test_invalid_policy_output_freezes_without_releasing(action):
    driver = FakeDriver(gripper0=0.23)
    assert not pr.run_policy(driver, None, "unused.zip", policy=Policy(action))
    assert driver.stopped and driver.torque and driver.read()[1] == 0.23
    assert not driver.commands


@pytest.mark.parametrize(
    "scene",
    [[], {"hand": {"seen": True, "x": "bad", "y": 0.0, "z": 0.03}}],
)
def test_broken_tracking_freezes_without_releasing(scene):
    driver = FakeDriver(gripper0=0.23)
    assert not pr.run_policy(driver, lambda: scene, "unused.zip", policy=Policy())
    assert driver.stopped and driver.torque and driver.read()[1] == 0.23
    assert not driver.commands


def test_driver_failure_during_a_run_preserves_grasp_and_stops():
    class BrokenDriver(FakeDriver):
        def read(self):
            if self.commands:
                raise RuntimeError("encoder unavailable")
            return super().read()

    driver = BrokenDriver(gripper0=0.23)
    assert not pr.run_policy(driver, None, "unused.zip", policy=Policy())
    assert driver.stopped and driver.torque
    assert len(driver.commands) == 1 and driver.commands[0][1] == 0.23


def test_torque_off_remains_off_without_a_goal_write():
    driver = FakeDriver(gripper0=0.23)
    driver.torque_off()
    syncs = driver.goal_syncs
    assert not pr.run_policy(driver, None, "unused.zip", policy=Policy())
    assert not driver.torque and not driver.commands and driver.goal_syncs == syncs


@pytest.mark.parametrize("stop_kind", ["driver", "cancel"])
def test_stop_during_inference_prevents_the_pending_command(stop_kind):
    driver = FakeDriver(gripper0=0.23)
    is_cancelled = False

    class StoppingPolicy(Policy):
        def predict(self, obs, deterministic=True):
            nonlocal is_cancelled
            if stop_kind == "driver":
                driver.freeze()
            else:
                is_cancelled = True
            return super().predict(obs, deterministic)

    assert not pr.run_policy(
        driver, None, "unused.zip", policy=StoppingPolicy(), cancelled=lambda: is_cancelled
    )
    assert not driver.commands and driver.stopped and driver.torque and driver.read()[1] == 0.23


def test_inference_that_finishes_after_the_deadline_cannot_issue_a_late_command():
    clock = Clock()
    driver = FakeDriver(gripper0=0.23)

    class LatePolicy(Policy):
        def predict(self, obs, deterministic=True):
            clock.sleep(0.2)
            return super().predict(obs, deterministic)

    assert not pr.run_policy(
        driver,
        None,
        "unused.zip",
        policy=LatePolicy(),
        timeout=0.1,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert not driver.commands and not driver.stopped and driver.torque


def test_delayed_inference_does_not_produce_catch_up_command_bursts():
    clock = Clock()
    sent = []

    class TimedDriver(FakeDriver):
        def _send(self, q, gripper):
            sent.append(clock.now())
            super()._send(q, gripper)

    class DelayedPolicy(Policy):
        def predict(self, obs, deterministic=True):
            if len(self.observations) == 1:
                clock.sleep(5 * DT)
            return super().predict(obs, deterministic)

    driver = TimedDriver(gripper0=0.23)
    pr.run_policy(
        driver,
        None,
        "unused.zip",
        policy=DelayedPolicy(),
        timeout=0.4,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert len(sent) > 3
    assert all(b - a >= DT - 1e-9 for a, b in zip(sent, sent[1:], strict=False))
    assert max(sent) < 0.4


def test_stalled_clock_has_a_finite_command_budget():
    driver = FakeDriver(gripper0=0.23)
    assert not pr.run_policy(
        driver,
        None,
        "unused.zip",
        policy=Policy(),
        timeout=3 * DT,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )
    assert len(driver.commands) <= 4
    assert not driver.stopped and driver.torque


def test_stop_on_last_success_tick_is_not_reported_as_success():
    driver = FakeDriver(gripper0=0.23)
    clock = Clock()

    def stop_on_tenth_tick(seconds):
        clock.sleep(seconds)
        if len(driver.commands) == 10:
            driver.freeze()

    # The target is the starting tip and zero actions keep every reading inside the success radius.
    events = []
    assert not pr.run_policy(
        driver,
        near_hand,
        "unused.zip",
        policy=Policy([0.0] * 4),
        on_event=events.append,
        clock=clock.now,
        sleep=stop_on_tenth_tick,
    )
    assert driver.stopped and driver.torque and not any("policy success" in event for event in events)


@pytest.mark.parametrize("reaches", [False, True])
def test_policy_to_ik_handoff_preserves_thirty_hz_command_spacing(monkeypatch, reaches):
    clock = Clock()
    sent = []

    class TimedDriver(FakeDriver):
        def _send(self, q, gripper):
            sent.append(clock.now())
            super()._send(q, gripper)

    driver = TimedDriver(gripper0=0.23)
    pacer = replay.Pacer(True, clock.now, clock.sleep)
    session = replay.Session(driver, near_hand if reaches else lambda: {}, pacer, hand_freeze=False)
    session.start()
    target = real_robot._Target(session)
    monkeypatch.setattr(pr, "load_policy", lambda _: Policy([0.0] * 4))
    assert real_robot._run_policy(session, {"policy": "unused.zip"}, 0.5, target) is reaches
    policy_commands = len(sent)
    assert policy_commands >= 10
    # Live carry always resumes IK tracking after the policy, even if the policy reports success.
    session.command(session.q, where="first IK tracking tick")
    session.command(session.q, where="second IK tracking tick")
    assert len(sent) == policy_commands + 2
    assert all(b - a >= DT - 1e-9 for a, b in zip(sent, sent[1:], strict=False)), sent
    assert all(gripper == 0.23 for _q, gripper in driver.commands)
