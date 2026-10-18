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
