"""The setup diagnostic inspects local metadata without contacting any device or service."""

from __future__ import annotations

import builtins
import json
import re
import subprocess
import sys

import pytest

from scripts import check_setup


@pytest.fixture
def setup_root(tmp_path, monkeypatch):
    for name in check_setup.CONFIGS:
        path = tmp_path / "config" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('{"example": true}' if name.endswith(".json") else "example: true\n")
    (tmp_path / "config/cameras.yaml").write_text("overhead: {index: 0}\nwrist: {index: 1}\n")
    for name in check_setup.ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setattr(check_setup, "ROOT", tmp_path)
    monkeypatch.setattr(check_setup, "_available", lambda module: True)
    for name in (
        "XAI_API_KEY",
        "DAYTONA_API_KEY",
        "FOLLOWER_PORT",
        "LEADER_PORT",
        "OVERHEAD_CAM",
        "WRIST_CAM",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_offline_check_does_not_require_keys_or_certify_physical_calibration(setup_root):
    report = check_setup.build_report(setup_root, {})
    assert report["offline_prerequisites_ok"]
    assert not any(report["api_keys_configured_not_authenticated"].values())
    assert report["camera_indexes_not_opened"] == {"overhead": 0, "wrist": 1}
    assert report["manual_acceptance_required"]
    assert "ready_for_motion" not in report
    assert "calibration" in report["acceptance_note"]


@pytest.mark.parametrize("problem", ["dependency", "artifact", "empty_config", "malformed_config"])
def test_missing_or_malformed_offline_prerequisites_fail(setup_root, monkeypatch, problem):
    if problem == "dependency":
        monkeypatch.setattr(check_setup, "_available", lambda module: module != "mujoco")
    elif problem == "artifact":
        (setup_root / check_setup.ARTIFACTS[0]).unlink()
    else:
        (setup_root / "config/limits.yaml").write_text("" if problem == "empty_config" else "invalid: [")
    assert not check_setup.build_report(setup_root, {})["offline_prerequisites_ok"]


def test_optional_rl_dependencies_and_video_do_not_block_offline(setup_root, monkeypatch):
    monkeypatch.setattr(check_setup, "_available", lambda module: module not in check_setup.OPTIONAL.values())
    report = check_setup.build_report(setup_root, {})
    assert report["offline_prerequisites_ok"]
    assert not report["optional_dependencies"]["av"]
    assert any("lerobot" in blocker for blocker in report["live_configuration_blockers"])


def test_placeholder_comments_remain_explicit_but_removing_them_is_not_acceptance(setup_root):
    path = setup_root / "config/joint_map.yaml"
    path.write_text("# PLACEHOLDER until A6\nshoulder_pan: {sign: 1, offset_deg: 0}\n")
    report = check_setup.build_report(setup_root, {})
    assert report["configs"]["joint_map.yaml"]["placeholder_notice"]
    assert any("joint_map.yaml" in item for item in report["live_configuration_blockers"])
    path.write_text("shoulder_pan: {sign: 1, offset_deg: 0}\n")
    report = check_setup.build_report(setup_root, {})
    assert not report["configs"]["joint_map.yaml"]["placeholder_notice"]
    assert any("A6" in item for item in report["manual_acceptance_required"])


def test_secrets_are_never_echoed_even_if_mistakenly_used_as_ports(setup_root):
    env = {
        "XAI_API_KEY": "COM84421",
        "DAYTONA_API_KEY": "not-a-real-secret",
        "FOLLOWER_PORT": "COM84421",
        "LEADER_PORT": "not-a-real-secret",
        "OVERHEAD_CAM": "not-a-real-secret",
    }
    report = check_setup.build_report(setup_root, env)
    encoded = json.dumps(report)
    assert env["XAI_API_KEY"] not in encoded and env["DAYTONA_API_KEY"] not in encoded
    assert all(report["api_keys_configured_not_authenticated"].values())
    assert report["ports_not_probed"] == {"follower": None, "leader": None}


def test_ports_cameras_and_fake_flags_are_reported_without_probing(setup_root):
    report = check_setup.build_report(
        setup_root,
        {"FOLLOWER_PORT": "COM8", "LEADER_PORT": "com8", "OVERHEAD_CAM": "2", "WRIST_CAM": "2", "FAKE": "1"},
    )
    assert report["ports_not_probed"] == {"follower": "COM8", "leader": "com8"}
    assert report["camera_indexes_not_opened"] == {"overhead": 2, "wrist": 2}
    blockers = " ".join(report["live_configuration_blockers"])
    assert "ports must differ" in blockers and "distinct physical indexes" in blockers
    assert "Fake mode" in blockers


def test_historical_report_exposes_mismatch_without_claiming_current_state(setup_root):
    path = setup_root / "hardware.json"
    path.write_text(
        json.dumps(
            {
                "arms": {
                    "follower": {"port": "COM8", "calibration": {"matches_hardware": True}},
                    "leader": {"port": "COM7", "calibration": {"matches_hardware": False}},
                }
            }
        )
    )
    report = check_setup.build_report(setup_root, {}, path)
    historical = report["historical_hardware"]
    assert historical["historical_only"] and not historical["current_hardware_checked"]
    assert historical["arms"]["follower"]["calibration_matched"]
    assert not historical["arms"]["leader"]["calibration_matched"]
    assert any("Historical leader" in item for item in report["live_configuration_blockers"])


def test_single_arm_report_still_surfaces_its_calibration_issue(setup_root):
    path = setup_root / "hardware.json"
    path.write_text('{"arms": {"leader": {"calibration": {"matches_hardware": false}}}}')
    report = check_setup.build_report(setup_root, {}, path)
    assert not report["historical_hardware"]["arms"]["follower"]["report_present"]
    assert any("Historical leader" in item for item in report["live_configuration_blockers"])


@pytest.mark.parametrize("content", ["{}", "not json", '{"arms": {"follower": null}}'])
def test_bad_historical_report_is_reported_without_traceback(setup_root, content):
    path = setup_root / "hardware.json"
    path.write_text(content)
    assert "error" in check_setup.build_report(setup_root, {}, path)["historical_hardware"]


def test_cli_loads_dotenv_as_booleans_saves_new_file_and_preserves_existing(setup_root, capsys):
    secret = "test-secret-must-never-appear"
    (setup_root / ".env").write_text(
        f"XAI_API_KEY={secret}\nDAYTONA_API_KEY=\nFOLLOWER_PORT=COM8\nLEADER_PORT=COM7\n"
    )
    out = setup_root / "reports/setup.json"
    assert check_setup.main(["--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert secret not in text and secret not in out.read_text()
    assert json.loads(text)["api_keys_configured_not_authenticated"] == {
        "XAI_API_KEY": True,
        "DAYTONA_API_KEY": False,
    }
    before = out.read_bytes()
    with pytest.raises(SystemExit) as error:
        check_setup.main(["--out", str(out)])
    assert error.value.code == 2 and out.read_bytes() == before
    assert check_setup.main(["--require-live-config"]) == 1


def test_live_configuration_gate_can_pass_without_certifying_manual_checks(setup_root, monkeypatch, capsys):
    for name, value in {
        "XAI_API_KEY": "test-xai",
        "DAYTONA_API_KEY": "test-daytona",
        "FOLLOWER_PORT": "COM8",
        "LEADER_PORT": "COM7",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("FAKE", "FAKE_LLM", "FAKE_SANDBOX", "FAKE_ROBOT", "FAKE_CAMERA"):
        monkeypatch.delenv(name, raising=False)
    assert check_setup.main(["--require-live-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["live_configuration_blockers"]
    assert report["manual_acceptance_required"]


def test_report_carries_the_host_identity(setup_root, monkeypatch):
    answers = {("rev-parse", "--short", "HEAD"): "abc1234", ("status", "--porcelain"): " M arm/driver.py\n"}
    monkeypatch.setattr(check_setup, "_git", lambda root, *args: answers[args])
    report = check_setup.build_report(setup_root, {})
    assert list(report)[0] == "host"
    host = report["host"]
    assert set(host) == {"hostname", "os", "python", "git_head", "git_dirty", "utc"}
    assert host["hostname"] and isinstance(host["hostname"], str)
    assert host["os"] and isinstance(host["os"], str)
    assert host["python"] == sys.version.split()[0]
    assert host["git_head"] == "abc1234" and host["git_dirty"] is True
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", host["utc"])
    assert report["offline_prerequisites_ok"]  # the block changes no verdict


def test_host_block_reports_a_clean_tree_and_a_missing_git(setup_root, monkeypatch):
    monkeypatch.setattr(check_setup, "_git", lambda root, *args: "" if args[0] == "status" else "abc1234")
    assert check_setup.host_block(setup_root)["git_dirty"] is False
    monkeypatch.setattr(check_setup, "_git", lambda root, *args: None)
    host = check_setup.host_block(setup_root)
    assert host["git_head"] is None and host["git_dirty"] is None


def test_git_helper_runs_in_the_root_and_never_raises(tmp_path, monkeypatch):
    seen = []

    def outside_a_repository(args, **kwargs):
        seen.append((args, kwargs.get("cwd")))
        return subprocess.CompletedProcess(args, 128, "", "fatal: not a git repository")

    monkeypatch.setattr(check_setup.subprocess, "run", outside_a_repository)
    assert check_setup._git(tmp_path, "rev-parse", "--short", "HEAD") is None
    assert seen == [(["git", "rev-parse", "--short", "HEAD"], str(tmp_path))]

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(check_setup.subprocess, "run", missing)
    assert check_setup._git(tmp_path, "status", "--porcelain") is None

    def clean(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "dfc47e7\n", "")

    monkeypatch.setattr(check_setup.subprocess, "run", clean)
    assert check_setup._git(tmp_path, "rev-parse", "--short", "HEAD") == "dfc47e7"


def test_cli_report_and_saved_file_include_the_host_block(setup_root, capsys):
    out = setup_root / "reports" / "host.json"
    assert check_setup.main(["--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert printed["host"]["hostname"] == saved["host"]["hostname"]
    assert saved["host"]["python"] == sys.version.split()[0]
    assert "os" in saved["host"] and "utc" in saved["host"]


def test_no_hardware_camera_or_service_module_is_imported(setup_root, monkeypatch):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"lerobot", "serial", "cv2", "openai", "daytona", "httpx", "mujoco"}:
            attempted.append(name)
            raise AssertionError("Hardware/service import attempted")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert check_setup.build_report(setup_root, {})["non_invasive"]
    assert not attempted
