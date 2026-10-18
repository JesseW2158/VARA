"""arm.driver.Driver against an injected fake SOFollower: connect order, goal sync, torque, units.

The fake mirrors the LeRobot 0.6.2 surface the driver uses (bus.connect/disconnect/is_calibrated/
sync_read/sync_write/write/enable_torque/disable_torque, robot.configure/get_observation/send_action) with
LeRobot's normalization formulas and the STS3215 behavior that matters: enabling torque drives a motor to
its stored Goal_Position. lerobot itself is never imported here.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

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
        self.gains: dict[str, tuple[int, int]] = {}  # motor -> (P_Coefficient written, torque at the time)
        self.fail_gain_writes = False
        self.max_jump = 0  # largest present-position change caused by a goal write or torque enable
        self.port_handler = SimpleNamespace(closePort=self.close_port)

    def close_port(self):
        self.log.append(("port.close",))
        self.is_connected = False

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
        if data_name == "P_Coefficient":  # a settings write: the servos accept it only with torque off
            if self.fail_gain_writes:
                raise ConnectionError("no status packet")
            self.gains[motor] = (int(value), self.torque[motor])
            return
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
        # the arm joints' gain, in configure's torque-off window (the gripper keeps LeRobot's)
        *[("write", "P_Coefficient", m, True) for m in driver.JOINTS],
        *[("write", "Goal_Position", m, False) for m in MOTORS],
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


@pytest.mark.parametrize("stop", [False, True])
def test_sag_during_configuration_is_held_in_place_before_enable(stop):
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus
        stale(bus)

        def configure():
            bus.disable_torque()
            bus.present["shoulder_lift"] -= 250
            bus.present["elbow_flex"] += 400
            if stop:
                d.freeze()
            bus.enable_torque()

        robot.configure = configure
        return robot

    d = Driver(robot_factory=factory)
    d.connect()
    bus = robots[0].bus
    assert bus.max_jump == 0
    assert bus.goal == bus.present
    assert bus.present["shoulder_lift"] == int(mid("shoulder_lift")) - 250
    assert bus.present["elbow_flex"] == int(mid("elbow_flex")) + 400
    assert bus.enable_torque == FakeBus.enable_torque.__get__(bus)
    assert d.stopped == stop
    if stop:
        assert not d.command(d.read()[0], 1.0)


def test_each_configuration_torque_enable_resyncs_after_a_new_sag():
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus

        def configure():
            for _ in range(2):
                bus.disable_torque()
                bus.present["shoulder_lift"] -= 100
                bus.enable_torque()

        robot.configure = configure
        return robot

    d = Driver(robot_factory=factory)
    d.connect()
    bus = robots[0].bus
    assert bus.max_jump == 0
    assert bus.goal == bus.present
    assert bus.present["shoulder_lift"] == int(mid("shoulder_lift")) - 200


def test_each_goal_write_uses_a_fresh_position_when_later_joints_sag():
    d, _, robots = build()
    d.connect()
    bus = robots[0].bus
    d.torque_off()
    original_write = bus.write

    def sag_after_write(*args, **kwargs):
        original_write(*args, **kwargs)
        for motor in MOTORS:
            if not bus.torque[motor]:
                bus.present[motor] -= 20

    bus.write = sag_after_write
    d.torque_on()
    assert bus.max_jump == 0
    assert bus.goal == bus.present
    assert bus.present["gripper"] == int(mid("gripper")) - 100


@pytest.mark.parametrize("failure", [RuntimeError("bad firmware"), KeyboardInterrupt()])
@pytest.mark.parametrize("disconnect_fails", [False, True])
def test_handshake_failure_closes_port_and_preserves_existing_torque(failure, disconnect_fails):
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus
        bus.torque = dict.fromkeys(MOTORS, 1)

        def connect(*args, **kwargs):
            bus.is_connected = True
            raise failure

        def broken_disconnect(**kwargs):
            assert kwargs == {"disable_torque": False}
            raise RuntimeError("disconnect unavailable after incomplete handshake")

        bus.connect = connect
        if disconnect_fails:
            bus.disconnect = broken_disconnect
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(type(failure)):
        d.connect()
    bus = robots[0].bus
    assert not bus.is_connected and not d.connected
    assert all(bus.torque.values())
    assert not goal_writes(log)
    assert not any(e[0] in {"configure", "disable_torque", "enable_torque"} for e in log)


@pytest.mark.parametrize("restores_on_exit", [False, True])
@pytest.mark.parametrize("previous_cycle", [False, True])
def test_configuration_failure_restores_holding_and_cleans_up(restores_on_exit, previous_cycle):
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus

        def configure():
            if previous_cycle:
                bus.disable_torque()
                bus.enable_torque()
            bus.disable_torque()
            bus.present["elbow_flex"] -= 350
            try:
                raise RuntimeError("configuration failed")
            finally:
                if restores_on_exit:
                    bus.enable_torque()

        robot.configure = configure
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match="configuration failed"):
        d.connect()
    bus = robots[0].bus
    assert not bus.is_connected and not d.connected
    assert all(bus.torque.values()) and bus.goal == bus.present and bus.max_jump == 0
    assert bus.enable_torque == FakeBus.enable_torque.__get__(bus)
    assert ("bus.disconnect", False) in log


def test_failed_goal_sync_during_configure_never_enables_stale_goals():
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus

        def configure():
            bus.disable_torque()
            bus.present["elbow_flex"] -= 350
            bus.lose_writes.add("elbow_flex")
            bus.enable_torque()

        robot.configure = configure
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match="elbow_flex"):
        d.connect()
    bus = robots[0].bus
    assert not bus.is_connected and not d.connected
    assert bus.max_jump == 0 and bus.torque["elbow_flex"] == 0
    assert not any(e[0] == "enable_torque" for e in log)
    assert bus.enable_torque == FakeBus.enable_torque.__get__(bus)


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


@pytest.mark.parametrize("register", ["Torque_Enable", "Goal_Position"])
def test_missing_strain_check_register_refuses_before_any_motor_write(register):
    log: list = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        original_read = robot.bus.sync_read

        def read(data_name, *args, **kwargs):
            values = original_read(data_name, *args, **kwargs)
            if data_name == register:
                del values["elbow_flex"]
            return values

        robot.bus.sync_read = read
        return robot

    d = Driver(robot_factory=factory)
    with pytest.raises(RuntimeError, match=f"{register} missing"):
        d.connect()
    assert not goal_writes(log)


def test_missing_goal_readback_is_not_treated_as_a_small_offset():
    bus = FakeBus([])
    bus.present["wrist_roll"] = 0
    original_read = bus.sync_read

    def read(data_name, *args, **kwargs):
        values = original_read(data_name, *args, **kwargs)
        if data_name == "Goal_Position":
            del values["wrist_roll"]
        return values

    bus.sync_read = read
    with pytest.raises(RuntimeError, match="Goal_Position missing after sync"):
        driver._sync_goals(bus)


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


def test_disconnect_failure_still_releases_the_port_without_dropping_torque():
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus

    def broken_disconnect(**kwargs):
        assert kwargs == {"disable_torque": False}
        raise RuntimeError("SDK disconnect failed")

    bus.disconnect = broken_disconnect
    with pytest.raises(RuntimeError, match="SDK disconnect failed"):
        d.disconnect()
    assert not bus.is_connected and not d.connected
    assert all(bus.torque.values())
    assert log[-1] == ("port.close",)


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


# -- A13: a STOP that lands before or during connect() survives it -----------------------------------------


def build_freezing_in(step: str):
    """A Driver whose fake robot calls freeze() inside one connect step, with stale goals planted first.

    The call runs on the connecting thread (the RLock lets it in), so it lands while `connected` is still
    False: exactly the window in which the old connect() erased it by ending with `stopped = False`.
    """
    log: list = []
    robots: list[FakeRobot] = []
    holder: dict[str, Driver] = {}

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        bus = robot.bus
        stale(bus)
        if step == "bus.connect":
            original_connect = bus.connect

            def connect(handshake: bool = True) -> None:
                original_connect(handshake)
                holder["d"].freeze()

            bus.connect = connect  # type: ignore[method-assign]
        elif step == "goal_sync":
            original_write = bus.write
            fired: list[bool] = []

            def write(data_name, motor, value, *, normalize=True, num_retry=0):
                if not fired:
                    fired.append(True)
                    holder["d"].freeze()
                original_write(data_name, motor, value, normalize=normalize, num_retry=num_retry)

            bus.write = write  # type: ignore[method-assign]
        elif step == "configure":

            def configure() -> None:
                log.append(("configure",))
                bus.disable_torque()
                holder["d"].freeze()  # Esc while configure() has torque off
                bus.enable_torque()

            robot.configure = configure  # type: ignore[method-assign]
        else:
            raise AssertionError(step)
        return robot

    d = Driver(robot_factory=factory)
    holder["d"] = d
    return d, log, robots


def assert_frozen_after_connect(d: Driver, log: list, bus: FakeBus) -> None:
    """Stopped and holding with torque on, re-synced after configure(), nothing moves until resume()."""
    assert d.connected and d.stopped and d.torque and all(bus.torque.values())
    assert bus.goal == bus.present and bus.max_jump == 0
    configure_at = log.index(("configure",))
    goal_at = [i for i, e in enumerate(log) if e[0] == "write" and e[1] == "Goal_Position"]
    assert max(goal_at) > configure_at  # goal := present once more after configure()'s torque toggle
    assert not any(e[0] == "send_action" for e in log)
    del log[:]
    assert d.command([math.radians(1.0), 0.0, 0.0, 0.0, 0.0], 1.0) is False
    assert log == []  # the very next command reaches nothing
    d.resume()
    assert d.command([math.radians(1.0), 0.0, 0.0, 0.0, 0.0], 1.0) is True
    assert any(e[0] == "send_action" for e in log)


@pytest.mark.parametrize("step", ["bus.connect", "goal_sync", "configure"])
def test_freeze_during_connect_survives_it(step):
    d, log, robots = build_freezing_in(step)
    d.connect()
    assert_frozen_after_connect(d, log, robots[0].bus)


def test_freeze_from_the_esc_thread_while_connect_holds_the_lock():
    """The real Esc path: freeze() runs on the watcher thread and waits for the lock connect() holds."""
    log: list = []
    robots: list[FakeRobot] = []
    esc: list[threading.Thread] = []
    holder: dict[str, Driver] = {}

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        stale(robot.bus)
        original_configure = robot.configure

        def configure() -> None:
            t = threading.Thread(target=holder["d"].freeze)
            t.start()
            esc.append(t)
            assert holder["d"]._freeze_requested.wait(2.0)  # Esc landed; freeze() now waits for the lock
            original_configure()

        robot.configure = configure  # type: ignore[method-assign]
        return robot

    d = Driver(robot_factory=factory)
    holder["d"] = d
    d.connect()
    assert d.stopped  # connect() applied the waiting STOP itself before releasing the lock
    esc[0].join(2.0)
    assert not esc[0].is_alive()
    assert_frozen_after_connect(d, log, robots[0].bus)


def test_freeze_before_connect_is_not_cleared_by_connect():
    log: list = []
    robots: list[FakeRobot] = []

    def factory(**config):
        robot = FakeRobot(log, **config)
        robots.append(robot)
        stale(robot.bus)
        return robot

    d = Driver(robot_factory=factory)
    d.freeze()  # Esc before connect(): nothing to hold yet, only the flag
    assert d.stopped and not d.connected and log == []
    d.connect()
    assert_frozen_after_connect(d, log, robots[0].bus)


def test_driver_gripper_scale_methods_match_the_module():
    d, _, _ = build()  # no connection needed: pure config arithmetic
    assert d.pct_to_gripper(40.0) == driver.pct_to_gripper(40.0) == pytest.approx(0.4)
    assert d.gripper_to_pct(0.4) == driver.gripper_to_pct(0.4) == pytest.approx(40.0)
    assert d.held_threshold("juice_box") == driver.held_threshold("juice_box")
    assert d.empty_closed_reading() == driver.empty_closed_reading() == 0.0


def test_a_gripper_open_past_open_pct_is_warned_once_before_the_first_command(monkeypatch, tmp_path, caplog):
    # open_pct 51 (2026-10-18): jaws left at ~98 % by an older map read as 1.0, so the first command
    # closes them to 51 % with no gripper move asked for; the driver says so once.
    jm = tmp_path / "joint_map.yaml"
    jm.write_text("gripper: {closed_pct: 0, open_pct: 51}\n", encoding="utf-8")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", jm)
    d, _log, robots = build()
    d.connect()
    lo, hi = RANGES[GRIPPER]
    robots[0].bus.present[GRIPPER] = int(lo + 0.98 * (hi - lo))
    with caplog.at_level("WARNING", logger=driver.log.name):
        q, g = d.read()
        d.read()
    assert g == 1.0
    warnings = [r for r in caplog.records if "past open_pct" in r.getMessage()]
    assert len(warnings) == 1 and "98.0 %" in warnings[0].getMessage()


def test_a_gripper_inside_the_map_is_not_warned(monkeypatch, tmp_path, caplog):
    jm = tmp_path / "joint_map.yaml"
    jm.write_text("gripper: {closed_pct: 0, open_pct: 51}\n", encoding="utf-8")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", jm)
    d, _log, robots = build()
    d.connect()
    lo, hi = RANGES[GRIPPER]
    robots[0].bus.present[GRIPPER] = int(lo + 0.40 * (hi - lo))
    with caplog.at_level("WARNING", logger=driver.log.name):
        d.read()
    assert not [r for r in caplog.records if "past open_pct" in r.getMessage()]


def test_connect_raises_the_arm_joints_gain_with_torque_off_and_leaves_the_gripper(monkeypatch):
    monkeypatch.delenv("VARA_ARM_P", raising=False)
    d, log, robots = build()
    d.connect()
    bus = robots[0].bus
    assert set(bus.gains) == set(driver.JOINTS) and driver.GRIPPER not in bus.gains
    assert all(p == driver.ARM_P_DEFAULT and torque == 0 for p, torque in bus.gains.values())
    gain_at = max(i for i, e in enumerate(log) if e[:2] == ("write", "P_Coefficient"))
    enabled_at = max(i for i, e in enumerate(log) if e == ("enable_torque",))
    assert gain_at < enabled_at  # written before the arm was powered again
    assert all(bus.torque[m] == 1 for m in driver.MOTORS)


def test_the_arm_gain_follows_vara_arm_p_and_refuses_bad_values_before_the_port_opens(monkeypatch):
    monkeypatch.setenv("VARA_ARM_P", "32")
    d, _log, robots = build()
    d.connect()
    assert {p for p, _t in robots[0].bus.gains.values()} == {32}
    for bad in ("7", "41", "stiff"):
        monkeypatch.setenv("VARA_ARM_P", bad)
        d, log, robots = build()
        with pytest.raises(ValueError):
            d.connect()
        assert robots == [] and log == []  # nothing built, nothing opened


def test_a_failed_gain_write_still_powers_the_arm(monkeypatch):
    monkeypatch.delenv("VARA_ARM_P", raising=False)
    built: list = []

    def factory(**config):
        robot = FakeRobot(built, **config)
        robot.bus.fail_gain_writes = True
        return robot

    d = Driver(robot_factory=factory)
    d.connect()
    assert d.connected and d.torque
    bus = d.robot.bus
    assert bus.gains == {} and all(bus.torque[m] == 1 for m in driver.MOTORS)
