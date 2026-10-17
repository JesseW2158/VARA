"""Read-only hardware evidence: no motor-write API is present on these fakes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from arm import driver
from arm import inspect_hardware as inspection_cli


@dataclass
class Calibration:
    id: int
    drive_mode: int = 0
    homing_offset: int = 123
    range_min: int = 1000
    range_max: int = 3000


class ReadOnlyBus:
    """Opening, reads and closing only; any attempted configuration/write fails."""

    def __init__(self, calibration: dict[str, Calibration]) -> None:
        self.calibration = dict(calibration)
        self.calls: list[tuple] = []
        self.is_open = False
        self.fail_on: str | None = None
        self.port_handler = SimpleNamespace(closePort=self.close)
        self.registers = {
            "Present_Position": dict.fromkeys(driver.MOTORS, 2100),
            "Goal_Position": dict.fromkeys(driver.MOTORS, 2100),
            "Torque_Enable": dict.fromkeys(driver.MOTORS, 0),
            "Present_Voltage": dict.fromkeys(driver.MOTORS, 120),
            "Present_Temperature": dict.fromkeys(driver.MOTORS, 31),
        }
        self.registers["Present_Position"][driver.GRIPPER] = 2750
        self.registers["Goal_Position"][driver.GRIPPER] = 2750

    def connect(self) -> None:
        self.calls.append(("connect",))
        self.is_open = True
        if self.fail_on == "connect":
            raise RuntimeError("handshake failed after opening")

    def close(self) -> None:
        self.calls.append(("close",))
        self.is_open = False

    def read_calibration(self) -> dict[str, Calibration]:
        self.calls.append(("calibration",))
        if self.fail_on == "calibration":
            raise RuntimeError("calibration read failed")
        return self.calibration

    def sync_read(self, register: str, *, normalize: bool = True, num_retry: int = 0) -> dict:
        self.calls.append(("read", register, normalize, num_retry))
        if self.fail_on == register:
            raise RuntimeError(f"{register} read failed")
        assert normalize is False, "derive normalization from the same raw snapshot"
        return dict(self.registers[register])


@pytest.fixture
def inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FOLLOWER_ID", raising=False)
    monkeypatch.delenv("LEADER_ID", raising=False)
    monkeypatch.setattr(driver, "_serial_identity", lambda port: {"serial_number": "board-123"})
    calibration = {name: Calibration(i) for i, name in enumerate(driver.MOTORS, 1)}
    calibration[driver.GRIPPER] = Calibration(6, range_min=2000, range_max=3500)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({name: asdict(value) for name, value in calibration.items()}))
    bus = ReadOnlyBus(calibration)
    robot = SimpleNamespace(calibration=calibration, calibration_fpath=path, bus=bus)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return robot

    return SimpleNamespace(bus=bus, robot=robot, factory=factory, calls=factory_calls, path=path)


@pytest.mark.parametrize("role", ["follower", "leader"])
def test_inspection_reads_without_motion_and_closes_port(inspection, role: str) -> None:
    report = driver.inspect_hardware(
        port="COM8", role=role, robot_id="measured-arm", robot_factory=inspection.factory
    )

    assert inspection.calls == [{"role": role, "port": "COM8", "id": "measured-arm"}]
    assert not inspection.bus.is_open
    assert inspection.bus.calls[0] == ("connect",)
    assert inspection.bus.calls[-1] == ("close",)
    assert {call[0] for call in inspection.bus.calls} == {"connect", "calibration", "read", "close"}
    assert report["read_only"] is True
    assert report["role"] == role
    assert report["usb"]["serial_number"] == "board-123"
    assert report["calibration"]["matches_hardware"] is True
    assert report["calibration"]["sha256"] == hashlib.sha256(inspection.path.read_bytes()).hexdigest()
    assert "not_assessed" in report["model_mapping"]
    pan = report["motors"]["shoulder_pan"]
    assert pan["calibrated_position_rad"] == pytest.approx(100 * 2 * math.pi / 4095)
    assert pan["position_raw"] == pan["goal_raw"] == 2100
    assert pan["voltage_v"] == 12.0
    assert pan["temperature_c"] == 31.0
    assert pan["torque_enabled"] is False
    assert report["motors"][driver.GRIPPER]["calibrated_gripper"] == 0.5
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("role,variable", [("follower", "FOLLOWER_ID"), ("leader", "LEADER_ID")])
def test_role_id_uses_environment(
    inspection, monkeypatch: pytest.MonkeyPatch, role: str, variable: str
) -> None:
    monkeypatch.setenv(variable, "existing-calibration")
    driver.inspect_hardware(port="COM7", role=role, robot_factory=inspection.factory)
    assert inspection.calls[0]["id"] == "existing-calibration"


@pytest.mark.parametrize("change", ["offset", "range", "missing"])
def test_mismatch_preserves_raw_evidence_without_normalized_positions(inspection, change: str) -> None:
    if change == "missing":
        del inspection.bus.calibration["shoulder_pan"]
    else:
        field = "homing_offset" if change == "offset" else "range_min"
        value = inspection.bus.calibration["shoulder_pan"]
        inspection.bus.calibration["shoulder_pan"] = replace(value, **{field: getattr(value, field) + 1})

    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)

    assert report["calibration"]["matches_hardware"] is False
    assert report["motors"]["shoulder_pan"]["position_raw"] == 2100
    assert all(
        "calibrated_position_rad" not in motor and "calibrated_gripper" not in motor
        for motor in report["motors"].values()
    )
    assert all(call[2] is False for call in inspection.bus.calls if call[0] == "read")
    assert not inspection.bus.is_open


@pytest.mark.parametrize("failure", ["connect", "calibration", "Present_Position", "Present_Temperature"])
def test_failure_after_opening_always_closes_raw_port(inspection, failure: str) -> None:
    inspection.bus.fail_on = failure
    with pytest.raises(RuntimeError, match="failed"):
        driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert inspection.bus.calls[-1] == ("close",)
    assert not inspection.bus.is_open


def test_missing_motor_register_is_an_error_and_closes_port(inspection) -> None:
    del inspection.bus.registers["Present_Voltage"]["wrist_roll"]
    with pytest.raises(RuntimeError, match="Incomplete Present_Voltage"):
        driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert not inspection.bus.is_open


@pytest.mark.parametrize(
    "register,value",
    [
        ("Present_Position", 2100.5),
        ("Present_Position", True),
        ("Goal_Position", "2100"),
        ("Goal_Position", float("nan")),
        ("Torque_Enable", 2),
        ("Torque_Enable", -1),
        ("Present_Voltage", -1),
        ("Present_Temperature", 256),
    ],
)
def test_invalid_register_values_cannot_be_reported_as_healthy(inspection, register: str, value) -> None:
    inspection.bus.registers[register]["shoulder_pan"] = value
    with pytest.raises(RuntimeError, match="Invalid"):
        driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert not inspection.bus.is_open


@pytest.mark.parametrize(
    "field,value", [("id", 99), ("range_min", 3000), ("range_max", 4096), ("drive_mode", True)]
)
def test_identical_but_invalid_calibration_never_enables_normalization(inspection, field: str, value) -> None:
    invalid = replace(inspection.robot.calibration["shoulder_pan"], **{field: value})
    inspection.robot.calibration["shoulder_pan"] = invalid
    inspection.bus.calibration["shoulder_pan"] = invalid
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert report["calibration"]["matches_hardware"] is False
    assert report["issues"]
    assert "calibrated_position_rad" not in report["motors"]["shoulder_pan"]


def test_gripper_drive_mode_mismatch_suppresses_normalization(inspection) -> None:
    inspection.robot.calibration[driver.GRIPPER] = replace(
        inspection.robot.calibration[driver.GRIPPER], drive_mode=1
    )
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert report["calibration"]["matches_hardware"] is False
    assert "calibrated_gripper" not in report["motors"][driver.GRIPPER]


def test_confirmed_gripper_drive_mode_reverses_gripper_only(inspection) -> None:
    for name in ("shoulder_pan", driver.GRIPPER):
        updated = replace(inspection.robot.calibration[name], drive_mode=1)
        inspection.robot.calibration[name] = inspection.bus.calibration[name] = updated
    inspection.bus.registers["Present_Position"][driver.GRIPPER] = 2300
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert report["calibration"]["matches_hardware"] is True
    assert report["motors"][driver.GRIPPER]["calibrated_gripper"] == pytest.approx(0.8)
    assert report["motors"]["shoulder_pan"]["calibrated_position_rad"] > 0


@pytest.mark.parametrize(
    "torque,error,flagged", [(0, 100, False), (1, 15, False), (1, 16, True), (1, -16, True)]
)
def test_strain_requires_power_and_more_than_fifteen_ticks(
    inspection, torque: int, error: int, flagged: bool
) -> None:
    inspection.bus.registers["Torque_Enable"]["shoulder_pan"] = torque
    inspection.bus.registers["Goal_Position"]["shoulder_pan"] += error
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    assert any("powered goal" in issue for issue in report["issues"]) is flagged
    assert report["motors"]["shoulder_pan"]["goal_error_raw"] == error


@pytest.mark.parametrize("register,value", [("Present_Position", -1), ("Goal_Position", 4096)])
def test_out_of_encoder_range_is_preserved_and_flagged(inspection, register: str, value: int) -> None:
    inspection.bus.registers[register]["shoulder_pan"] = value
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    field = "position_raw" if register == "Present_Position" else "goal_raw"
    assert report["motors"]["shoulder_pan"][field] == value
    assert any("0..4095" in issue for issue in report["issues"])


def test_position_outside_saved_range_has_separate_warning(inspection) -> None:
    inspection.bus.registers["Present_Position"]["shoulder_pan"] = 999
    report = driver.inspect_hardware(port="COM8", robot_factory=inspection.factory)
    pan = report["motors"]["shoulder_pan"]
    assert pan["encoder_in_range"] is True
    assert pan["within_calibrated_range"] is False
    assert any("saved encoder calibration range" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "role,voltage,flagged",
    [
        ("follower", 120, False),
        ("leader", 50, False),
        ("leader", 80, False),
        ("leader", 81, True),
        ("leader", 120, True),
        ("follower", 39, True),
    ],
)
def test_supply_issue_uses_role_and_live_voltage(inspection, role: str, voltage: int, flagged: bool) -> None:
    inspection.bus.registers["Present_Voltage"] = dict.fromkeys(driver.MOTORS, voltage)
    report = driver.inspect_hardware(port="COM8", role=role, robot_factory=inspection.factory)
    assert bool(report["issues"]) is flagged


@pytest.mark.parametrize(
    "port,role", [("", "follower"), ("  ", "follower"), (None, "follower"), ("COM8", "unknown")]
)
def test_invalid_request_never_constructs_hardware(inspection, port, role: str) -> None:
    with pytest.raises(ValueError):
        driver.inspect_hardware(port=port, role=role, robot_factory=inspection.factory)
    assert not inspection.calls


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.delenv("FOLLOWER_PORT", raising=False)
    monkeypatch.delenv("LEADER_PORT", raising=False)
    calls = []

    def inspect(**kwargs):
        calls.append(kwargs)
        return {"read_only": True, "calibration": {"matches_hardware": True}, "issues": []}

    monkeypatch.setattr(inspection_cli, "inspect_hardware", inspect)
    return calls


def test_cli_captures_both_arms_and_creates_report_directory(cli, tmp_path: Path) -> None:
    out = tmp_path / "evidence" / "arms.json"
    assert inspection_cli.main(["--follower-port", "COM8", "--leader-port", "COM7", "--out", str(out)]) == 0
    assert cli == [{"port": "COM8", "role": "follower"}, {"port": "COM7", "role": "leader"}]
    saved = json.loads(out.read_text())
    assert saved["read_only"] is True
    assert set(saved["arms"]) == {"follower", "leader"}


def test_cli_accepts_environment_ports(cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LEADER_PORT", "COM7")
    assert inspection_cli.main(["--out", str(tmp_path / "report.json")]) == 0
    assert cli == [{"port": "COM7", "role": "leader"}]


@pytest.mark.parametrize("ports", [[], ["--follower-port", "COM8", "--leader-port", "com8"]])
def test_cli_rejects_missing_or_duplicate_ports_before_capture(cli, tmp_path: Path, ports: list[str]) -> None:
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        inspection_cli.main([*ports, "--out", str(out)])
    assert exc.value.code == 2
    assert not cli
    assert not out.exists()


def test_cli_refuses_to_overwrite_evidence_before_opening_hardware(cli, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        inspection_cli.main(["--follower-port", "COM8", "--out", str(out)])
    assert not cli
    assert out.read_text() == "previous evidence"


def test_cli_records_one_arm_error_and_still_inspects_the_other(
    cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def inspect(**kwargs):
        cli.append(kwargs)
        if kwargs["role"] == "follower":
            raise RuntimeError("port unavailable")
        return {"calibration": {"matches_hardware": True}, "issues": []}

    monkeypatch.setattr(inspection_cli, "inspect_hardware", inspect)
    out = tmp_path / "report.json"
    assert inspection_cli.main(["--follower-port", "COM8", "--leader-port", "COM7", "--out", str(out)]) == 1
    saved = json.loads(out.read_text())["arms"]
    assert saved["follower"]["error"] == "RuntimeError: port unavailable"
    assert saved["leader"]["calibration"]["matches_hardware"] is True
    assert len(cli) == 2


def test_cli_mismatched_calibration_is_nonzero(cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        inspection_cli,
        "inspect_hardware",
        lambda **kwargs: {"calibration": {"matches_hardware": False}, "issues": []},
    )
    assert inspection_cli.main(["--follower-port", "COM8", "--out", str(tmp_path / "report.json")]) == 1


def test_cli_health_issue_is_nonzero_and_preserved(
    cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        inspection_cli,
        "inspect_hardware",
        lambda **kwargs: {"calibration": {"matches_hardware": True}, "issues": ["powered goal discrepancy"]},
    )
    out = tmp_path / "report.json"
    assert inspection_cli.main(["--follower-port", "COM8", "--out", str(out)]) == 1
    assert json.loads(out.read_text())["arms"]["follower"]["issues"] == ["powered goal discrepancy"]
