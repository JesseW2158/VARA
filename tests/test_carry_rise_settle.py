"""The carry rise waits for a slow real arm and tops up a loaded arm's sag, and still stops a stuck one.

2026-10-18, real arm: a hand-over held the box, rose 7.9 cm, read its joints once right after the last rise
command and stopped with "carry rise did not reach its clearance height" while the soft servos were still
catching up under the load.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from arm import ik, real_robot, replay
from arm.driver import FakeDriver

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
PILL_PICK, _ = ik.solve((0.18, -0.12, 0.035), -math.pi / 2)
START, _ = ik.solve((0.18, -0.12, 0.115), -math.pi / 2, q0=PILL_PICK)  # the demo's hand_over start


def _scene(hand=(0.28, -0.08, 0.03)):
    scene = json.loads(json.dumps(FAKE_SCENE))
    scene["hand"].update({"x": hand[0], "y": hand[1], "z": hand[2], "seen": True})
    return lambda: scene


class LaggingDriver(FakeDriver):
    """Reads its previous pose for a number of reads after each command, as a slow servo would."""

    def __init__(self, *args, lag_reads: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lag_reads = lag_reads
        self._stale: tuple[list[float], float] | None = None
        self._left = 0

    def command(self, q, gripper):
        before = super().read()
        ok = super().command(q, gripper)
        if self._left == 0:
            self._stale = (list(before[0]), before[1])
        self._left = self.lag_reads
        return ok

    def read(self):
        if self._left > 0 and self._stale is not None:
            self._left -= 1
            return list(self._stale[0]), self._stale[1]
        return super().read()


class SaggingDriver(FakeDriver):
    """Reads its grasp center a fixed distance below the commanded one, as a loaded soft arm sags."""

    def __init__(self, *args, sag_m: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sag_m = sag_m

    def read(self):
        q, g = super().read()
        (x, y, z), pitch = ik.fk_grasp(q)
        q_low, ok = ik.solve_grasp((x, y, z - self.sag_m), pitch, q0=q)
        return (list(q_low) if ok else list(q)), g


def test_a_slow_arm_is_waited_for_instead_of_stopping_the_carry():
    d = LaggingDriver(q0=START, gripper0=0.2, lag_reads=5)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=_scene(), realtime=False)
    assert not any("did not reach" in e for e in result["events"]), result["events"]
    assert not any("top-up" in e or "short of" in e for e in result["events"])


def test_a_sagging_loaded_arm_is_topped_up_to_the_carry_height():
    d = SaggingDriver(q0=START, gripper0=0.2, sag_m=0.015)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=_scene(), realtime=False)
    assert not any("did not reach" in e for e in result["events"]), result["events"]
    assert any("short of" in e and "commanding" in e for e in result["events"]), result["events"]
    cap = float(replay.load_box()["z"][1]) - real_robot.CEILING_MARGIN_M
    topup = next(e for e in result["events"] if "commanding" in e)
    height = float(topup.split(" short of ")[1].split(" m")[0])
    more_cm = float(topup.split("commanding ")[1].split(" cm")[0])
    assert abs(more_cm - 1.5) < 0.2  # the measured shortfall, commanded once more
    assert height + more_cm / 100 <= cap + 1e-6  # the top-up never aims above the ceiling cap


class BlockedDriver(FakeDriver):
    """Reads at most `z_block` for its grasp center: something stops the arm rising, whatever is commanded."""

    def __init__(self, *args, z_block: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.z_block = z_block

    def read(self):
        q, g = super().read()
        (x, y, z), pitch = ik.fk_grasp(q)
        if z <= self.z_block:
            return list(q), g
        q_low, ok = ik.solve_grasp((x, y, self.z_block), pitch, q0=q)
        return (list(q_low) if ok else list(q)), g


def test_a_blocked_arm_still_stops_holding_the_object_after_the_top_ups():
    d = BlockedDriver(q0=START, gripper0=0.2, z_block=0.12)
    result = real_robot.carry_to_hand("pill_bottle", driver=d, scene=_scene(), realtime=False)
    assert result["reason"] == "error"
    assert any("did not reach its clearance height" in e for e in result["events"]), result["events"]
    assert d.stopped and "released" not in result["events"]


def _occluding_scene(driver, hand=(0.26, 0.134, 0.03), hidden_within=0.15):
    """The wristband, hidden from the overhead camera while the grasp center is within `hidden_within`
    of it (the arm covers the marker), seen otherwise; the tracker keeps its last pose when unseen."""
    base = json.loads(json.dumps(FAKE_SCENE))
    base["hand"].update({"x": hand[0], "y": hand[1], "z": hand[2], "seen": True})

    def scene():
        q, _g = FakeDriver.read(driver)
        (gx, gy, gz), _ = ik.fk_grasp(q)
        s = json.loads(json.dumps(base))
        s["hand"]["seen"] = math.dist((gx, gy, gz), hand) > hidden_within
        return s

    return scene


def test_a_hand_the_arm_covers_keeps_its_last_position_and_the_object_is_released_there():
    # 2026-10-18, real arm: the wristband vanished under the arm 15 cm out, the carry turned to the handoff
    # zone on the other side, the wristband reappeared, and the arm went back and forth
    hand = (0.26, 0.134, 0.03)
    d = FakeDriver(q0=START, gripper0=0.2)
    result = real_robot.carry_to_hand(
        "pill_bottle", driver=d, scene=_occluding_scene(d, hand), realtime=False
    )
    events = result["events"]
    assert result["ok"], events
    assert any("hand unseen; using its last position (0.260, 0.134)" in e for e in events), events
    assert "released" in events and not any("falling back to the handoff zone" in e for e in events)
    # never turned away: once within 15 cm, the grasp center never got more than 3 cm further from the hand
    dists = [math.dist(ik.fk_grasp(q)[0], hand) for q, _g in d.commands]
    first_close = next(i for i, x in enumerate(dists) if x <= 0.15)
    release_at = next(i for i, (_q, g) in enumerate(d.commands) if i > first_close and g >= 0.99)
    assert max(dists[first_close:release_at]) <= 0.18, max(dists[first_close:release_at])


def test_the_speed_cap_near_the_hand_holds_while_it_is_covered():
    hand = (0.26, 0.134, 0.03)
    d = FakeDriver(q0=START, gripper0=0.2)
    real_robot.carry_to_hand("pill_bottle", driver=d, scene=_occluding_scene(d, hand), realtime=False)
    slow = replay.TIP_SPEED_NEAR_HAND * replay.TICK
    tips = [ik.fk_tip(q) for q, _g in d.commands]
    near = [
        math.dist(a, b)
        for a, b in zip(tips, tips[1:], strict=False)
        if min(math.dist(a, hand), math.dist(b, hand)) < replay.HAND_SLOW_M
    ]
    assert near and max(near) <= slow * (1 + 1e-6), max(near)
