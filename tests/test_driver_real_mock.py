"""arm.driver.Driver against an injected fake SOFollower: connect order, goal sync, torque, units.

The fake mirrors the LeRobot 0.6.2 surface the driver uses (bus.connect/disconnect/is_calibrated/
sync_read/sync_write/write/enable_torque/disable_torque, robot.configure/get_observation/send_action) with
LeRobot's normalization formulas and the STS3215 behavior that matters: enabling torque drives a motor to
its stored Goal_Position. lerobot itself is never imported here.
"""

from __future__ import annotations

import math

import pytest

from arm import driver
from arm.driver import GRIPPER, JOINTS, MAX_STEP_RAD, MOTORS, Driver

# Raw calibration ranges measured on our follower (lerobot-tools/HANDOFF.md)
RANGES = {
    "shoulder_pan": (724, 3425),
    "shoulder_lift": (836, 3250),
    "elbow_flex": (843, 3068),
    "wrist_flex": (863, 3207),
    "wrist_roll": (0, 4095),
    "gripper": (2012, 3577),
}
MAX_RES = 4095


def mid(motor: str) -> float:
    lo, hi = RANGES[motor]
    return (lo + hi) / 2


class FakeBus:
    """Six STS3215 motors, raw ticks inside, LeRobot's DEGREES / RANGE_0_100 normalization outside."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.is_connected = False
        self.calibrated = True
        self.present = {m: int(mid(m)) for m in MOTORS}
        self.goal = dict(self.present)
        self.torque = dict.fromkeys(MOTORS, 0)
        self.lose_writes: set[str] = set()  # motors whose Goal_Position writes silently vanish
        self.gripper_block_raw: int | None = None  # an object stops the jaws at this raw position
        self.max_jump = 0  # largest present-position change caused by a goal write or torque enable

    # LeRobot formulas (motors_bus.py _normalize/_unnormalize)
    def _norm(self, motor: str, raw: int) -> float:
        lo, hi = RANGES[motor]
        if motor == GRIPPER:
            return (min(hi, max(lo, raw)) - lo) / (hi - lo) * 100
        return (raw - mid(motor)) * 360 / MAX_RES

    def _unnorm(self, motor: str, val: float) -> int:
        lo, hi = RANGES[motor]
        if motor == GRIPPER:
            return int(min(100.0, max(0.0, val)) / 100 * (hi - lo) + lo)
        return int(val * MAX_RES / 360 + mid(motor))

    def _settle(self, motor: str) -> None:
        """A powered motor goes to its goal (blocked jaws stop at the object)."""
        if not self.torque[motor]:
            return
        target = self.goal[motor]
        if motor == GRIPPER and self.gripper_block_raw is not None:
            target = max(target, self.gripper_block_raw)
        self.max_jump = max(self.max_jump, abs(target - self.present[motor]))
        self.present[motor] = target

    def connect(self, handshake: bool = True) -> None:
        self.log.append(("bus.connect",))
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.log.append(("bus.disconnect", disable_torque))
        if disable_torque:
            self.torque = dict.fromkeys(MOTORS, 0)
        self.is_connected = False

    @property
    def is_calibrated(self) -> bool:
        self.log.append(("is_calibrated",))
        return self.calibrated

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        self.log.append(("sync_read", data_name, normalize))
        names = MOTORS if motors is None else list(motors)
        table = {"Present_Position": self.present, "Goal_Position": self.goal, "Torque_Enable": self.torque}[
            data_name
        ]
        if normalize and data_name in ("Present_Position", "Goal_Position"):
            return {m: self._norm(m, table[m]) for m in names}
        return {m: table[m] for m in names}

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        self.log.append(("write", data_name, motor, value, normalize))
        assert data_name == "Goal_Position", data_name
        if motor in self.lose_writes:
            return
        self.goal[motor] = self._unnorm(motor, value) if normalize else int(value)
        self.torque[motor] = 1  # this firmware switches torque on when a goal is written (HANDOFF.md)
        self._settle(motor)

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0):
        self.log.append(("sync_write", data_name, dict(values), normalize))
        assert data_name == "Goal_Position", data_name
        for motor, value in values.items():
            self.goal[motor] = self._unnorm(motor, value) if normalize else int(value)
            self.torque[motor] = 1
            self._settle(motor)

    def enable_torque(self, motors=None, num_retry=0):
        self.log.append(("enable_torque",))
        for m in MOTORS:
            self.torque[m] = 1
            self._settle(m)

    def disable_torque(self, motors=None, num_retry=0):
        self.log.append(("disable_torque",))
        self.torque = dict.fromkeys(MOTORS, 0)


class FakeRobot:
    def __init__(self, log: list, **config) -> None:
        self.log = log
        self.config = config
        self.bus = FakeBus(log)

    def configure(self) -> None:
        # SOFollower.configure(): `with self.bus.torque_disabled(): ...` -> off, EEPROM writes, on
        self.log.append(("configure",))
        self.bus.disable_torque()
        self.bus.enable_torque()

    def get_observation(self) -> dict[str, float]:
        self.log.append(("get_observation",))
        return {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.log.append(("send_action", dict(action)))
        goal = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        cap = self.config["max_relative_target"]
        assert isinstance(cap, float)  # ensure_safe_goal_position raises TypeError on an int
        present = self.bus.sync_read("Present_Position")
        safe = {}
        for m, g in goal.items():
            # ensure_safe_goal_position's own order: min first, then max. A NaN diff survives both
            # (every comparison is False) and reaches _unnormalize unchanged, exactly like LeRobot.
            diff = g - present[m]
            safe_diff = min(diff, cap)
            safe_diff = max(safe_diff, -cap)
            safe[m] = present[m] + safe_diff
        self.bus.sync_write("Goal_Position", safe)
        return {f"{m}.pos": v for m, v in safe.items()}

    # LeRobot's own connect(), for the negative control: configure without a goal sync
    def lerobot_connect(self) -> None:
        self.bus.connect()
        assert self.bus.is_calibrated
        self.configure()


@pytest.fixture(autouse=True)
def default_config(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "LIMITS_PATH", tmp_path / "no_limits.yaml")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", tmp_path / "no_joint_map.yaml")
    monkeypatch.setattr(Driver, "_sleep", lambda self, s: None)
    monkeypatch.setenv("FOLLOWER_PORT", "COM8")
    monkeypatch.delenv("FOLLOWER_ID", raising=False)


def build(**kw):
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        return robot

    d = Driver(robot_factory=factory, **kw)
    return d, log, robots


def stale(bus: FakeBus) -> None:
    """Stale goals left by another program, 168 degrees away on one joint (the measured worst case)."""
    bus.goal["shoulder_lift"] = bus.present["shoulder_lift"] + int(168 * MAX_RES / 360)
    bus.goal["wrist_flex"] = bus.present["wrist_flex"] - 900
    bus.goal["gripper"] = RANGES["gripper"][0]


def goal_writes(log):
    return [e for e in log if e[0] == "write" and e[1] == "Goal_Position"]


def test_config_passed_to_lerobot():
    d, log, robots = build()
    d.connect()
    assert robots[0].config == {
        "port": "COM8",
        "id": "my_awesome_follower_arm",
        "use_degrees": True,
        "max_relative_target": 5.0,
        "disable_torque_on_disconnect": False,
    }


def test_follower_id_precedence(monkeypatch, tmp_path):
    path = tmp_path / "joint_map.yaml"
    path.write_text("lerobot_id: from_file\n", encoding="utf-8")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", path)
    d, _, robots = build()
    d.connect()
    assert robots[0].config["id"] == "from_file"
    monkeypatch.setenv("FOLLOWER_ID", "from_env")
    d, _, robots = build()
    d.connect()
    assert robots[0].config["id"] == "from_env"
    d, _, robots = build(follower_id="explicit", port="COM9")
    d.connect()
    assert robots[0].config["id"] == "explicit" and robots[0].config["port"] == "COM9"


def test_missing_port_refuses(monkeypatch):
    monkeypatch.delenv("FOLLOWER_PORT")
    d, log, _ = build()
    with pytest.raises(RuntimeError, match="FOLLOWER_PORT"):
        d.connect()
    assert log == [] and not d.connected


def test_connect_call_order_and_no_jump():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus

    significant = [
        e[:1] if e[0] != "write" else ("write", e[1], e[2], e[4]) for e in log if e[0] in SIGNIFICANT
    ]
    assert significant == [
        ("bus.connect",),
        ("is_calibrated",),
        *[("write", "Goal_Position", m, False) for m in MOTORS],
        ("configure",),
        ("disable_torque",),
        ("enable_torque",),
    ]
    # the goals were read raw and written raw from the present position, before any torque enable
    first_read = log.index(("sync_read", "Present_Position", False))
    writes = goal_writes(log)
    assert {e[2]: e[3] for e in writes} == bus.present
    assert all(e[4] is False for e in writes)
    first_enable = log.index(("enable_torque",))
    assert first_read < log.index(writes[0]) < log.index(writes[-1]) < first_enable
    assert d.connected and d.torque and not d.stopped and d.last_command is None


def test_connect_with_stale_goals_does_not_move_the_arm():
    log: list = []
    seen = {}

    def factory(**config):
        # the factory runs inside connect(): plant the stale goals before the driver sees the bus
        robot = FakeRobot(log, **config)
        stale(robot.bus)
        seen["robot"] = robot
        return robot

    d = Driver(robot_factory=factory)
    d.connect()
    bus = seen["robot"].bus
    assert bus.present == {m: int(mid(m)) for m in MOTORS}
    assert bus.max_jump == 0
    assert bus.goal == bus.present


def test_negative_control_lerobot_connect_order_would_jump():
    """The fake detects the hazard the driver guards against: configure() alone swings the arm."""
    log: list = []
    robot = FakeRobot(log, port="COM8", id="x", use_degrees=True, max_relative_target=5.0)
    stale(robot.bus)
    robot.lerobot_connect()
    assert robot.bus.max_jump >= int(168 * MAX_RES / 360)


def test_uncalibrated_refuses_without_prompt_or_torque():
    d, log, robots = build()

    def factory(**config):
        robot = FakeRobot(log, **config)
        robot.bus.calibrated = False
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match="calibration"):
        d.connect()
    kinds = [e[0] for e in log]
    assert kinds[:2] == ["bus.connect", "is_calibrated"]
    assert ("bus.disconnect", False) in log
    assert "write" not in kinds and "enable_torque" not in kinds and "configure" not in kinds
    assert not d.connected


def test_powered_and_strained_motor_refuses():
    log: list = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robot.bus.torque["elbow_flex"] = 1
        robot.bus.goal["elbow_flex"] = robot.bus.present["elbow_flex"] + 500
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match="elbow_flex"):
        d.connect()
    kinds = [e[0] for e in log]
    assert "write" not in kinds and "enable_torque" not in kinds
    assert ("bus.disconnect", False) in log


def test_lost_goal_write_is_caught_before_torque_enable():
    log: list = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        stale(robot.bus)
        robot.bus.lose_writes.add("wrist_flex")
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match="wrist_flex"):
        d.connect()
    assert "enable_torque" not in [e[0] for e in log]
    assert not d.connected


def test_torque_off_blocks_and_torque_on_resyncs_before_enable():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    d.torque_off()
    assert ("disable_torque",) == log[-1]
    assert d.torque is False
    del log[:]
    assert d.command([0.0] * 5, 1.0) is False
    assert log == []  # nothing reached the bus
    # the arm was moved by hand while limp; its goal is stale now
    bus.present["shoulder_lift"] += 300
    d.torque_on()
    kinds = [e[0] for e in log]
    writes = goal_writes(log)
    assert len(writes) == len(MOTORS) and all(e[4] is False for e in writes)
    assert {e[2]: e[3] for e in writes} == bus.present
    assert log.index(writes[-1]) < kinds.index("enable_torque")
    assert bus.max_jump == 0 and d.torque and d.last_command is None
    # the next command measures its 2 degree step from the present position
    q_now, _ = d.read()
    assert d.command([0.0] * 5, 1.0) is True
    sent = [e for e in log if e[0] == "send_action"][-1][1]
    assert sent["shoulder_lift.pos"] == pytest.approx(math.degrees(q_now[1] - MAX_STEP_RAD), abs=0.05)


def test_freeze_holds_present_with_torque_on_then_resume():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    target = [math.radians(1.5), 0.0, 0.0, 0.0, 0.0]
    assert d.command(target, 1.0) is True
    del log[:]
    d.freeze()
    kinds = [e[0] for e in log]
    assert "disable_torque" not in kinds and "send_action" not in kinds
    assert log[0] == ("sync_read", "Present_Position", False)
    writes = goal_writes(log)
    assert {e[2]: e[3] for e in writes} == bus.present and all(e[4] is False for e in writes)
    assert d.stopped and d.torque and all(bus.torque.values())
    del log[:]
    assert d.command([0.0] * 5, 1.0) is False
    assert log == []
    d.resume()
    assert d.command([0.0] * 5, 1.0) is True
    assert any(e[0] == "send_action" for e in log)


def test_freeze_after_torque_off_powers_the_arm_in_place():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    d.torque_off()
    bus.present["elbow_flex"] -= 400  # sagged while limp
    del log[:]
    d.freeze()
    kinds = [e[0] for e in log]
    assert log.index(goal_writes(log)[-1]) < kinds.index("enable_torque")
    assert bus.max_jump == 0 and d.torque and d.stopped


def test_freeze_never_raises_and_still_stops():
    d, log, robots = build()
    d.connect()

    def boom(*a, **k):
        raise ConnectionError("serial glitch")

    robots[0].bus.sync_read = boom  # type: ignore[method-assign]
    d.freeze()
    assert d.stopped
    d2 = Driver(robot_factory=lambda **c: FakeRobot([], **c))
    d2.freeze()  # not connected: just sets the flag
    assert d2.stopped and not d2.connected


def test_disconnect_passes_disable_torque_false_unless_asked():
    d, log, robots = build()
    d.connect()
    d.disconnect()
    assert log[-1] == ("bus.disconnect", False)
    assert d.torque and not d.connected and all(robots[0].bus.torque.values())
    with pytest.raises(RuntimeError):
        d.read()
    d, log, robots = build()
    d.connect()
    d.disconnect(torque_off=True)
    assert log[-1] == ("bus.disconnect", True)
    assert not d.torque and not d.connected


def test_units_read_degrees_to_radians_and_gripper_pct_to_unit():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    bus.present["shoulder_pan"] = int(mid("shoulder_pan") + 90 * MAX_RES / 360)
    bus.present["wrist_roll"] = int(mid("wrist_roll") - 45 * MAX_RES / 360)
    lo, hi = RANGES["gripper"]
    bus.present["gripper"] = int(lo + 0.5 * (hi - lo))
    q, g = d.read()
    assert q[0] == pytest.approx(math.pi / 2, abs=0.01)
    assert q[4] == pytest.approx(-math.pi / 4, abs=0.01)
    assert q[1:4] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
    assert g == pytest.approx(0.5, abs=0.001)


def test_units_command_radians_to_degrees_and_gripper_unit_to_pct():
    d, log, robots = build()
    d.connect()
    del log[:]
    target = [math.radians(1.5), math.radians(-1.0), 0.0, 0.0, 0.0]
    assert d.command(target, 0.25) is True
    sent = [e for e in log if e[0] == "send_action"][0][1]
    assert set(sent) == {f"{j}.pos": None for j in JOINTS}.keys()  # the gripper is not in send_action
    assert sent["shoulder_pan.pos"] == pytest.approx(1.5, abs=0.05)
    assert sent["shoulder_lift.pos"] == pytest.approx(-1.0, abs=0.05)
    grip = [e for e in log if e[0] == "sync_write" and GRIPPER in e[2]]
    assert grip[-1][2] == {GRIPPER: pytest.approx(25.0)} and grip[-1][3] is True
    # LeRobot's layer (5 degrees from present) stays underneath ours; a full close is one write
    assert d.command(target, 0.0) is True
    grip = [e for e in log if e[0] == "sync_write" and GRIPPER in e[2]]
    assert grip[-1][2] == {GRIPPER: pytest.approx(0.0)}
    assert robots[0].bus.present[GRIPPER] == RANGES["gripper"][0]


def test_joint_map_sign_and_offset_applied(monkeypatch, tmp_path):
    path = tmp_path / "joint_map.yaml"
    path.write_text("shoulder_pan: {sign: -1, offset_deg: 10.0}\ngripper: {closed_pct: 10, open_pct: 90}\n")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", path)
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    lo, hi = RANGES["gripper"]
    bus.present["gripper"] = int(lo + 0.5 * (hi - lo))  # 50 % on LeRobot's scale
    q, g = d.read()
    assert q[0] == pytest.approx(math.radians(10.0), abs=0.01)  # radians(sign * 0 + offset)
    assert g == pytest.approx(0.5, abs=0.001)  # (50 - 10) / (90 - 10)
    del log[:]
    d.command([math.radians(11.0), 0.0, 0.0, 0.0, 0.0], 1.0)
    sent = [e for e in log if e[0] == "send_action"][0][1]
    assert sent["shoulder_pan.pos"] == pytest.approx(-1.0, abs=0.05)  # sign * (11 - 10)
    grip = [e for e in log if e[0] == "sync_write" and GRIPPER in e[2]][-1][2]
    assert grip[GRIPPER] == pytest.approx(90.0)


def test_close_gripper_hold_backs_off_three_on_the_0_100_scale():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    lo, hi = RANGES["gripper"]
    bus.gripper_block_raw = int(lo + 0.20 * (hi - lo))  # the object stops the jaws at 20 %
    d.command([0.0] * 5, 1.0)
    del log[:]
    reading = d.close_gripper_hold(target=0.0, settle=0.5)
    assert reading == pytest.approx(0.20, abs=0.002)
    grips = [e[2][GRIPPER] for e in log if e[0] == "sync_write" and GRIPPER in e[2]]
    assert grips[0] == pytest.approx(0.0)
    assert grips[-1] == pytest.approx(reading * 100 - 3.0, abs=0.2)
    assert d.last_command is not None and d.last_command[1] == pytest.approx(grips[-1] / 100, abs=0.002)


def test_read_and_command_require_connection():
    """A forgotten connect() (get_driver() hands out an unconnected Driver) is loud, never a silent False."""
    d, log, _ = build()
    assert not d.connected and not d.torque
    with pytest.raises(RuntimeError, match="not connected"):
        d.read()
    with pytest.raises(RuntimeError, match="not connected"):
        d.command([0.0] * 5, 1.0)  # checked before the torque gate, which would otherwise return False
    with pytest.raises(RuntimeError, match="not connected"):
        d.torque_on()
    with pytest.raises(RuntimeError, match="not connected"):
        d.close_gripper_hold()
    d.torque = True  # even then, nothing goes to a bus that is not open
    with pytest.raises(RuntimeError, match="not connected"):
        d.command([0.0] * 5, 1.0)
    assert log == []


def test_non_finite_targets_never_reach_the_bus():
    """LeRobot would map a NaN gripper to range_min (a full-torque close) and raise on a NaN joint inside
    _unnormalize; the driver refuses both before anything is written."""
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    assert d.command([0.0] * 5, 1.0) is True
    present, goal, last = dict(bus.present), dict(bus.goal), d.last_command
    del log[:]
    for q, g in (([math.nan, 0.0, 0.0, 0.0, 0.0], 1.0), ([0.0] * 5, math.nan), ([0.0] * 5, math.inf)):
        with pytest.raises(ValueError, match="non-finite"):
            d.command(q, g)
    assert log == []
    assert bus.present == present and bus.goal == goal and d.last_command == last
    assert bus.present[GRIPPER] == RANGES["gripper"][1]  # still open, not slammed to range_min


def test_negative_control_fake_bus_treats_nan_like_lerobot():
    """The mock is no kinder than LeRobot 0.6.2: a NaN joint raises in _unnormalize, a NaN gripper closes."""
    robot = FakeRobot([], port="COM8", id="x", use_degrees=True, max_relative_target=5.0)
    robot.bus.connect()
    robot.bus.enable_torque()
    with pytest.raises(ValueError):
        robot.send_action({"shoulder_pan.pos": math.nan})
    robot.bus.sync_write("Goal_Position", {GRIPPER: math.nan})
    assert robot.bus.present[GRIPPER] == RANGES["gripper"][0]  # bounded(NaN) = 0 -> range_min, fully closed


SIGNIFICANT = {"bus.connect", "is_calibrated", "write", "enable_torque", "disable_torque", "configure"}
