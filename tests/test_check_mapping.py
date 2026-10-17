"""Synthetic A6 fixtures test the checker; these are not physical calibration evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from arm import check_mapping, driver, ik


@pytest.fixture
def measurements(tmp_path, monkeypatch):
    # Every fixture explicitly carries measurement_kind=synthetic; FK-generated values cannot certify A6.
    mapping = tmp_path / "joint_map.yaml"
    mapping.write_text("lerobot_id: synthetic_follower\n")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", mapping)
    calibration = {
        name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(driver.MOTORS, 1)
    }
    cache = tmp_path / "encoder.json"
    cache.write_text(json.dumps(calibration))
    fingerprint = hashlib.sha256(cache.read_bytes()).hexdigest()
    poses, reports = [], []
    for i in range(5):
        q = [-0.4 + i * 0.2, -1.35 + i * 0.05, 0.02 + i * 0.03, 1.45 - i * 0.1, 0.0]
        motors = {}
        for name, value in zip(driver.JOINTS, q, strict=True):
            raw = round(value * 4095 / math.tau + 2047.5)
            motors[name] = {"position_raw": raw, "calibrated_position_rad": (raw - 2047.5) * math.tau / 4095}
        report = {
            "role": "follower",
            "read_only": True,
            "lerobot_id": "synthetic_follower",
            "usb": {"serial_number": "synthetic-board"},
            "motors": motors,
            "calibration": {
                "path": str(cache),
                "sha256": fingerprint,
                "matches_hardware": True,
                "saved": copy.deepcopy(calibration),
                "hardware": copy.deepcopy(calibration),
            },
        }
        report_path = tmp_path / f"pose-{i + 1}.json"
        report_path.write_text(json.dumps({"arms": {"follower": report}}))
        actual_q = driver.map_inspection_joints(report)
        measured, _ = ik.fk(actual_q)
        poses.append({"hardware_report": report_path.name, "tip_xyz": list(measured)})
        reports.append(report_path)
    source = {"measurement_kind": "synthetic", "poses": poses}
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(source))
    return SimpleNamespace(path=path, source=source, reports=reports, cache=cache, mapping=mapping)


def mutate_report(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload["arms"]["follower"])
    path.write_text(json.dumps(payload))


def test_five_accurate_synthetic_poses_record_errors_and_provenance(measurements):
    report = check_mapping.check_measurements(measurements.path)
    assert report["measurement_kind"] == "synthetic"
    assert report["all_within_tolerance"] and report["max_error_m"] == 0
    assert report["mean_error_m"] == 0 and len(report["poses"]) == 5
    assert all(row["local_calibration_file_checked"] for row in report["poses"])
    assert (
        report["config_and_model_sha256"][str(measurements.mapping)]
        == hashlib.sha256(measurements.mapping.read_bytes()).hexdigest()
    )
    assert report["measurements_sha256"] == hashlib.sha256(measurements.path.read_bytes()).hexdigest()
    assert "ready_for_motion" not in report


def test_one_inaccurate_pose_fails_even_with_good_mean(measurements, capsys):
    measurements.source["poses"][2]["tip_xyz"][0] += 0.02
    measurements.path.write_text(json.dumps(measurements.source))
    assert check_mapping.main([str(measurements.path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mean_error_m"] < 0.015 < report["max_error_m"]
    assert [row["within_tolerance"] for row in report["poses"]] == [True, True, False, True, True]


def test_boundary_fifteen_millimeters_passes(measurements):
    measurements.source["poses"][0]["tip_xyz"][0] += 0.015
    measurements.path.write_text(json.dumps(measurements.source))
    assert check_mapping.check_measurements(measurements.path)["all_within_tolerance"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, 10**400])
def test_incomplete_or_nonfinite_measurements_refused(measurements, bad):
    measurements.source["poses"][0]["tip_xyz"][1] = bad
    measurements.path.write_text(json.dumps(measurements.source))
    with pytest.raises(ValueError, match="finite measured meters"):
        check_mapping.check_measurements(measurements.path)


@pytest.mark.parametrize("duplicate", [False, True])
def test_fewer_than_five_distinct_poses_refused(measurements, duplicate):
    if duplicate:
        measurements.source["poses"] = [measurements.source["poses"][0]] * 5
    else:
        measurements.source["poses"].pop()
    measurements.path.write_text(json.dumps(measurements.source))
    with pytest.raises(ValueError, match="five distinct"):
        check_mapping.check_measurements(measurements.path)


@pytest.mark.parametrize(
    "problem", ["mismatch_flag", "bad_ranges", "different_hardware", "normalized", "leader"]
)
def test_invalid_inspection_evidence_refused(measurements, problem):
    def mutate(report):
        if problem == "mismatch_flag":
            report["calibration"]["matches_hardware"] = False
        elif problem == "bad_ranges":
            for side in ("saved", "hardware"):
                report["calibration"][side]["elbow_flex"]["range_min"] = 4095
        elif problem == "different_hardware":
            report["calibration"]["hardware"]["elbow_flex"]["homing_offset"] = 5
        elif problem == "normalized":
            report["motors"]["elbow_flex"]["calibrated_position_rad"] += 0.2
        else:
            report["role"] = "leader"

    mutate_report(measurements.reports[0], mutate)
    with pytest.raises(ValueError):
        check_mapping.check_measurements(measurements.path)


def test_leader_only_report_refused(measurements):
    measurements.reports[0].write_text('{"arms": {"leader": {"role": "leader"}}}')
    with pytest.raises(ValueError, match="follower inspection"):
        check_mapping.check_measurements(measurements.path)


def test_changed_encoder_cache_rejects_stale_evidence(measurements):
    measurements.cache.write_text(measurements.cache.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since"):
        check_mapping.check_measurements(measurements.path)


def test_missing_foreign_cache_is_explicit_and_does_not_prevent_historical_comparison(measurements):
    for path in measurements.reports:
        mutate_report(
            path, lambda report: report["calibration"].update(path="/foreign-machine/cache/encoder.json")
        )
    report = check_mapping.check_measurements(measurements.path)
    assert report["all_within_tolerance"]
    assert not any(row["local_calibration_file_checked"] for row in report["poses"])
    assert "Missing foreign cache paths" in report["scope"]
    # A supplied current cache is mandatory and allows portable evidence to be checked locally.
    report = check_mapping.check_measurements(measurements.path, calibration_file=measurements.cache)
    assert all(row["local_calibration_file_checked"] for row in report["poses"])
    with pytest.raises(ValueError, match="Explicit calibration file is missing"):
        check_mapping.check_measurements(
            measurements.path, calibration_file=measurements.path.parent / "absent.json"
        )


def test_inconsistent_calibration_fingerprints_between_poses_refused(measurements):
    measurements.cache.unlink()  # compare archived records without an available local encoder file
    mutate_report(measurements.reports[0], lambda report: report["calibration"].update(sha256="a" * 64))
    with pytest.raises(ValueError, match="same follower and encoder calibration fingerprint"):
        check_mapping.check_measurements(measurements.path)


def test_driver_applies_current_joint_sign_and_offset(measurements):
    payload = json.loads(measurements.reports[0].read_text())["arms"]["follower"]
    original = driver.map_inspection_joints(payload)
    measurements.mapping.write_text(
        "lerobot_id: synthetic_follower\nshoulder_pan: {sign: -1, offset_deg: 20}\n"
    )
    mapped = driver.map_inspection_joints(payload)
    assert mapped[0] == pytest.approx(-original[0] + math.radians(20))
    assert mapped[1:] == original[1:]
    assert not check_mapping.check_measurements(measurements.path)["all_within_tolerance"]


@pytest.mark.parametrize("bad", [float("nan"), 10**400, True])
def test_invalid_normalized_encoder_numbers_cannot_be_mapped(measurements, bad):
    payload = json.loads(measurements.reports[0].read_text())["arms"]["follower"]
    payload["motors"]["elbow_flex"]["calibrated_position_rad"] = bad
    with pytest.raises(ValueError, match="normalized encoder"):
        driver.map_inspection_joints(payload)


def test_calibration_content_must_agree_even_when_claimed_hashes_agree(measurements):
    measurements.cache.unlink()

    def alter_saved_values(report):
        for side in ("saved", "hardware"):
            report["calibration"][side]["elbow_flex"]["homing_offset"] = 42

    mutate_report(measurements.reports[0], alter_saved_values)
    with pytest.raises(ValueError, match="same follower and encoder calibration fingerprint"):
        check_mapping.check_measurements(measurements.path)


def test_different_usb_board_cannot_supply_another_pose(measurements):
    mutate_report(measurements.reports[0], lambda report: report["usb"].update(serial_number="other-board"))
    with pytest.raises(ValueError, match="same follower"):
        check_mapping.check_measurements(measurements.path)


def test_example_contains_nulls_and_never_overwrites_existing_output(tmp_path, capsys):
    path = tmp_path / "measurements.json"
    assert check_mapping.main(["--example", "--out", str(path)]) == 0
    template = json.loads(capsys.readouterr().out)
    assert all(pose["tip_xyz"] == [None, None, None] for pose in template["poses"])
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--example", "--out", str(path)])
    assert error.value.code == 2 and path.read_bytes() == before
