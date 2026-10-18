"""Synthetic reference-pose reports exercise the A6 solver; they are not physical calibration evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from types import SimpleNamespace

import pytest
import yaml

from arm import check_mapping, driver, ik

SIGNS = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": 1, "wrist_flex": -1, "wrist_roll": 1}
OFFSETS = {
    "shoulder_pan": 5.0,
    "shoulder_lift": -10.0,
    "elbow_flex": 0.0,
    "wrist_flex": 20.0,
    "wrist_roll": 0.0,
}
# Saved encoder ranges in ticks. Through these signs and offsets they stay inside the MJCF ranges, so the
# default fixture solves without review warnings.
RANGES = {
    "shoulder_pan": (900, 3200),
    "shoulder_lift": (1100, 3000),
    "elbow_flex": (1000, 3100),
    "wrist_flex": (1200, 2900),
    "wrist_roll": (500, 3600),
}
RAWS = {
    "shoulder_pan": 2100,
    "shoulder_lift": 1800,
    "elbow_flex": 2400,
    "wrist_flex": 2300,
    "wrist_roll": 2050,
}
# The demo follower's saved ranges (runs/hardware/20261017-continuation.json) reach past the MJCF ones; the
# solved example in docs/joint-calibration.md uses them with SIGNS and OFFSETS.
FOLLOWER_RANGES = {
    "shoulder_pan": (724, 3425),
    "shoulder_lift": (836, 3250),
    "elbow_flex": (843, 3068),
    "wrist_flex": (863, 3207),
    "wrist_roll": (0, 4095),
}
FOLLOWER_RAWS = {
    "shoulder_pan": 2074,
    "shoulder_lift": 2043,
    "elbow_flex": 1955,
    "wrist_flex": 2035,
    "wrist_roll": 2047,
}
DOC = check_mapping.ROOT / "docs" / "joint-calibration.md"


def build_report(path, cache, *, raws=RAWS, ranges=RANGES, signs=SIGNS, offsets=OFFSETS):
    """Write an `arm.inspect_hardware`-shaped document to `path` and its encoder cache to `cache`.

    Returns (follower report, model_q). The model pose is derived from the raw encoders through the injected
    signs and offsets, so a correct solve recovers exactly those offsets.
    """
    calibration = {}
    for index, name in enumerate(driver.MOTORS, 1):
        lo, hi = ranges.get(name, (2000, 3600))
        calibration[name] = {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": lo,
            "range_max": hi,
        }
    cache.write_text(json.dumps(calibration))
    fingerprint = hashlib.sha256(cache.read_bytes()).hexdigest()
    health = {"torque_enabled": False, "voltage_v": 12.0, "temperature_c": 31.0, "encoder_in_range": True}
    motors, model_q = {}, []
    for name in driver.JOINTS:
        lo, hi = ranges[name]
        raw = raws[name]
        lerobot_rad = (raw - (lo + hi) / 2) * math.tau / 4095
        motors[name] = {
            "position_raw": raw,
            "goal_raw": raw,
            "goal_error_raw": 0,
            **health,
            "within_calibrated_range": True,
            "calibrated_position_rad": lerobot_rad,
        }
        model_q.append(math.radians(signs[name] * math.degrees(lerobot_rad) + offsets[name]))
    motors["gripper"] = {
        "position_raw": 2300,
        "goal_raw": 2300,
        "goal_error_raw": 0,
        **health,
        "within_calibrated_range": True,
        "calibrated_gripper": 0.1875,
    }
    report = {
        "captured_at": "2026-10-17T20:00:00+00:00",
        "read_only": True,
        "role": "follower",
        "port": "COM8",
        "usb": {"serial_number": "synthetic-board", "description": "USB Serial Device", "vid": 1, "pid": 2},
        "lerobot_id": "synthetic_follower",
        "calibration": {
            "path": str(cache),
            "sha256": fingerprint,
            "matches_hardware": True,
            "saved": copy.deepcopy(calibration),
            "hardware": copy.deepcopy(calibration),
        },
        "model_mapping": "not_assessed: A6 requires measured joint directions, offsets and tip poses",
        "issues": [],
        "motors": motors,
    }
    path.write_text(json.dumps({"read_only": True, "arms": {"follower": report}}, indent=2))
    return report, model_q


@pytest.fixture
def solver(tmp_path, monkeypatch):
    mapping = tmp_path / "joint_map.yaml"
    mapping.write_text("lerobot_id: synthetic_follower\n")
    monkeypatch.setattr(driver, "JOINT_MAP_PATH", mapping)
    runs = tmp_path / "runs"
    monkeypatch.setenv("VARA_RUNS_DIR", str(runs))
    report_path = tmp_path / "reference-pose.json"
    cache = tmp_path / "synthetic_follower.json"
    report, model_q = build_report(report_path, cache)
    path = tmp_path / "solve.json"
    source = {
        "hardware_report": report_path.name,
        "model_q_rad": model_q,
        "signs": dict(SIGNS),
        "note": "synthetic",
    }
    path.write_text(json.dumps(source))
    return SimpleNamespace(
        path=path,
        source=source,
        report=report,
        report_path=report_path,
        cache=cache,
        model_q=model_q,
        mapping=mapping,
        runs=runs,
    )


def rewrite(env, mutate_source=None, mutate_report=None):
    if mutate_source is not None:
        mutate_source(env.source)
        env.path.write_text(json.dumps(env.source))
    if mutate_report is not None:
        payload = json.loads(env.report_path.read_text())
        mutate_report(payload["arms"]["follower"])
        env.report_path.write_text(json.dumps(payload))


def test_solve_offsets_recovers_injected_offsets_for_both_signs(solver):
    offsets = check_mapping.solve_offsets(solver.report, solver.model_q, SIGNS)
    assert list(offsets) == driver.JOINTS
    for name in driver.JOINTS:
        assert offsets[name] == pytest.approx(OFFSETS[name], abs=1e-6)
    # Flipping one sign moves that joint's offset by twice its reading and nothing else.
    other = check_mapping.solve_offsets(solver.report, solver.model_q, dict(SIGNS, elbow_flex=-1))
    reading = math.degrees(solver.report["motors"]["elbow_flex"]["calibrated_position_rad"])
    assert other["elbow_flex"] == pytest.approx(OFFSETS["elbow_flex"] + 2 * reading, abs=1e-6)
    for name in driver.JOINTS:
        if name != "elbow_flex":
            assert other[name] == pytest.approx(offsets[name], abs=1e-9)


def test_limits_from_calibration_orders_flipped_ranges_and_contains_the_reading(solver):
    joint_map = {name: {"sign": SIGNS[name], "offset_deg": OFFSETS[name]} for name in driver.JOINTS}
    limits = check_mapping.limits_from_calibration(solver.report, joint_map)
    assert list(limits) == driver.JOINTS
    for i, name in enumerate(driver.JOINTS):
        lo_raw, hi_raw = RANGES[name]
        half = (hi_raw - lo_raw) / 2 * 360 / 4095
        lo, hi = limits[name]
        assert lo < hi
        # A sign of -1 swaps the two ends; ordered, both signs give the offset plus or minus the half range.
        expected = [math.radians(OFFSETS[name] - half), math.radians(OFFSETS[name] + half)]
        assert [lo, hi] == pytest.approx(expected)
        assert lo <= solver.model_q[i] <= hi
    # Unordered, a flipped joint's ends come out backwards.
    half = (RANGES["wrist_flex"][1] - RANGES["wrist_flex"][0]) / 2 * 360 / 4095
    ends = [math.radians(SIGNS["wrist_flex"] * end + OFFSETS["wrist_flex"]) for end in (-half, half)]
    assert ends[0] > ends[1]
    # Given the solved map, the driver reads the reference report back as the model pose, inside the limits.
    solver.mapping.write_text("lerobot_id: synthetic_follower\n" + check_mapping.joint_map_yaml(joint_map))
    q = driver.map_inspection_joints(solver.report)
    assert q == pytest.approx(solver.model_q, abs=2e-5)
    assert all(limits[name][0] <= q[i] <= limits[name][1] for i, name in enumerate(driver.JOINTS))


def test_pure_functions_refuse_bad_joint_maps_and_signs(solver):
    with pytest.raises(ValueError, match="sign and offset_deg"):
        check_mapping.limits_from_calibration(solver.report, {"shoulder_pan": {"sign": 1}})
    with pytest.raises(ValueError, match="1 or -1"):
        check_mapping.solve_offsets(solver.report, solver.model_q, dict(SIGNS, wrist_roll=0))
    with pytest.raises(ValueError, match="exactly the joints"):
        check_mapping.solve_offsets(solver.report, solver.model_q, [1, -1, 1, -1, 1])
    with pytest.raises(ValueError, match="finite"):
        check_mapping.solve_offsets(solver.report, solver.model_q[:4], SIGNS)
    with pytest.raises(ValueError, match="follower inspection"):
        check_mapping.solve_offsets({"role": "leader"}, solver.model_q, SIGNS)


def test_solve_record_carries_provenance_hashes_and_exact_yaml(solver):
    record = check_mapping.solve(solver.path)
    assert record["kind"] == "a6_joint_map_solve" and record["note"] == "synthetic"
    assert record["lerobot_id"] == "synthetic_follower" and record["usb_serial_number"] == "synthetic-board"
    assert record["calibration_sha256"] == solver.report["calibration"]["sha256"]
    assert record["local_calibration_file_checked"] is True
    assert record["report_sha256"] == hashlib.sha256(solver.report_path.read_bytes()).hexdigest()
    assert record["solve_sha256"] == hashlib.sha256(solver.path.read_bytes()).hexdigest()
    hashes = record["config_and_model_sha256"]
    assert set(hashes) == {  # the same files the measurement check hashes, with the temp joint map
        str(solver.mapping),
        str(driver.LIMITS_PATH),
        str(driver.SCENE_PATH),
        str(ik.MODEL_PATH),
        str(ik.MODEL_PATH.parent / "so101_new_calib.xml"),
    }
    assert hashes[str(solver.mapping)] == hashlib.sha256(solver.mapping.read_bytes()).hexdigest()
    assert record["warnings"] == []
    assert record["reference"]["model_q_rad"] == solver.model_q
    assert record["reference"]["tip_xyz"] == pytest.approx(list(ik.fk_tip(solver.model_q)))
    for name in driver.JOINTS:
        assert record["joint_map"][name]["sign"] == SIGNS[name]
        assert record["joint_map"][name]["offset_deg"] == pytest.approx(OFFSETS[name], abs=1e-6)
        assert record["joints"][name]["limits_rad"] == record["limits"][name]
        assert record["joints"][name]["range_raw"] == list(RANGES[name])
    parsed = yaml.safe_load(record["joint_map_yaml"])
    assert set(parsed) == set(driver.JOINTS)
    for name in driver.JOINTS:
        assert parsed[name]["sign"] == SIGNS[name]
        assert parsed[name]["offset_deg"] == pytest.approx(OFFSETS[name], abs=5e-4)
    parsed_limits = yaml.safe_load(record["limits_yaml"])
    assert set(parsed_limits) == set(driver.JOINTS)
    for name in driver.JOINTS:
        assert parsed_limits[name] == pytest.approx(record["limits"][name], abs=5e-5)
    assert "No motion is authorized" in record["scope"]


def test_printed_yaml_loads_through_the_driver(solver, monkeypatch):
    record = check_mapping.solve(solver.path)
    solver.mapping.write_text("lerobot_id: synthetic_follower\n" + record["joint_map_yaml"])
    limits_path = solver.mapping.parent / "limits.yaml"
    limits_path.write_text(record["limits_yaml"])
    monkeypatch.setattr(driver, "LIMITS_PATH", limits_path)
    joint_map = driver.joint_map()
    loaded = driver.limits()
    for name in driver.JOINTS:
        assert joint_map[name]["sign"] == SIGNS[name]
        assert joint_map[name]["offset_deg"] == pytest.approx(OFFSETS[name], abs=5e-4)
        assert loaded[name] == pytest.approx(record["limits"][name], abs=5e-5)
    assert driver.map_inspection_joints(solver.report) == pytest.approx(solver.model_q, abs=2e-5)


def test_solve_never_edits_config_files(solver):
    before = (solver.mapping.read_bytes(), driver.LIMITS_PATH.read_bytes())
    check_mapping.solve(solver.path)
    assert (solver.mapping.read_bytes(), driver.LIMITS_PATH.read_bytes()) == before


def test_cli_solve_prints_offsets_tip_and_yaml_and_writes_timestamped_record(solver, capsys):
    assert check_mapping.main(["--solve", str(solver.path)]) == 0
    out = capsys.readouterr().out
    offset_lines = [
        line for line in out.splitlines() if re.match(r"^\w+: \{sign: -?1, offset_deg: -?\d+\.\d{3}\}$", line)
    ]
    limit_lines = [
        line for line in out.splitlines() if re.match(r"^\w+: \[-?\d+\.\d{4}, -?\d+\.\d{4}\]$", line)
    ]
    assert len(offset_lines) == 5 and len(limit_lines) == 5
    assert "FK tip (x, y, z)" in out and "WARNING" not in out
    records = sorted((solver.runs / "calibration").glob("solve-*.json"))
    assert len(records) == 1 and re.fullmatch(r"solve-\d{8}-\d{6}-\d{6}\.json", records[0].name)
    assert f"Record: {records[0]}" in out
    record = json.loads(records[0].read_text())
    assert record["joint_map_yaml"].splitlines()[1:] == offset_lines
    assert record["limits_yaml"].splitlines()[1:] == limit_lines
    assert check_mapping.main(["--solve", str(solver.path)]) == 0  # a second run gets its own file
    assert len(list((solver.runs / "calibration").glob("solve-*.json"))) == 2


def test_cli_out_is_never_overwritten_and_misuse_exits_2(solver, capsys):
    out = solver.runs / "custom" / "solve-01.json"
    assert check_mapping.main(["--solve", str(solver.path), "--out", str(out)]) == 0
    before = out.read_bytes()
    assert json.loads(before)["kind"] == "a6_joint_map_solve"
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--solve", str(solver.path), "--out", str(out)])
    assert error.value.code == 2 and out.read_bytes() == before
    for argv in (
        ["--solve", str(solver.path), "--example"],
        ["--solve", str(solver.path), str(solver.path)],
        [],
        ["--example", "--calibration-file", str(solver.cache)],
    ):
        with pytest.raises(SystemExit) as error:
            check_mapping.main(argv)
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--help"])
    assert error.value.code == 0
    assert "--solve SOLVE_JSON" in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem, match",
    [
        ("leader_only", "follower"),
        ("read_failed", "read failed"),
        ("not_object", "JSON object"),
        ("no_calibration", "calibration"),
        ("mismatch_flag", "did not match"),
        ("normalized", "disagrees"),
        ("other_follower", "differs"),
    ],
)
def test_bad_reports_are_refused(solver, problem, match):
    if problem == "leader_only":
        solver.report_path.write_text(json.dumps({"arms": {"leader": {"role": "leader", "read_only": True}}}))
    elif problem == "read_failed":
        failed = {"port": "COM8", "error": "SerialException: could not open port"}
        solver.report_path.write_text(json.dumps({"arms": {"follower": failed}}))
    elif problem == "not_object":
        solver.report_path.write_text("[1, 2]")
    else:

        def mutate(report):
            if problem == "no_calibration":
                del report["calibration"]
            elif problem == "mismatch_flag":
                report["calibration"]["matches_hardware"] = False
            elif problem == "normalized":
                report["motors"]["elbow_flex"]["calibrated_position_rad"] += 0.2
            else:
                report["lerobot_id"] = "another_follower"

        rewrite(solver, mutate_report=mutate)
    with pytest.raises(ValueError, match=match):
        check_mapping.solve(solver.path)
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--solve", str(solver.path)])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("model_q_rad", [0.1, float("nan"), 0.0, 0.0, 0.0], "finite"),
        ("model_q_rad", [0.1, float("inf"), 0.0, 0.0, 0.0], "finite"),
        ("model_q_rad", [0.1, 10**400, 0.0, 0.0, 0.0], "finite"),
        ("model_q_rad", [0.1, True, 0.0, 0.0, 0.0], "finite"),
        ("model_q_rad", [0.1, None, 0.0, 0.0, 0.0], "finite"),
        ("model_q_rad", [0.1, 0.2, 0.3, 0.4], "finite"),
        ("model_q_rad", "0.1 0.2 0.3 0.4 0.5", "finite"),
        ("model_q_rad", [0.0, 90.0, 0.0, 0.0, 0.0], "radians, not degrees"),
        ("signs", dict(SIGNS, elbow_flex=0), "1 or -1"),
        ("signs", dict(SIGNS, elbow_flex=2), "1 or -1"),
        ("signs", dict(SIGNS, elbow_flex=True), "1 or -1"),
        ("signs", dict(SIGNS, elbow_flex="1"), "1 or -1"),
        ("signs", dict(SIGNS, elbow_flex=None), "1 or -1"),
        ("signs", {k: v for k, v in SIGNS.items() if k != "wrist_roll"}, "exactly the joints"),
        ("signs", dict(SIGNS, gripper=1), "exactly the joints"),
        ("signs", [1, -1, 1, -1, 1], "exactly the joints"),
        ("hardware_report", 7, "hardware_report"),
        ("hardware_report", "", "hardware_report"),
        ("note", 5, "note must be a string"),
    ],
)
def test_bad_solve_file_values_are_refused(solver, field, value, match):
    rewrite(solver, lambda source: source.update({field: value}))
    with pytest.raises(ValueError, match=match):
        check_mapping.solve(solver.path)
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--solve", str(solver.path)])
    assert error.value.code == 2


def test_solve_file_keys_and_report_path_are_checked(solver):
    rewrite(solver, lambda source: source.update(model_q_deg=[0, 0, 0, 0, 0]))
    with pytest.raises(ValueError, match=r"unknown \['model_q_deg'\]"):
        check_mapping.solve(solver.path)
    rewrite(solver, lambda source: (source.pop("model_q_deg"), source.pop("signs")))
    with pytest.raises(ValueError, match=r"missing \['signs'\]"):
        check_mapping.solve(solver.path)
    solver.source["signs"] = dict(SIGNS)
    rewrite(solver, lambda source: source.update(hardware_report="absent.json"))
    with pytest.raises(OSError):
        check_mapping.solve(solver.path)
    with pytest.raises(SystemExit) as error:
        check_mapping.main(["--solve", str(solver.path)])
    assert error.value.code == 2
    solver.path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        check_mapping.solve(solver.path)


def test_float_signs_and_bare_reports_are_accepted(solver):
    rewrite(solver, lambda source: source.update(signs={k: float(v) for k, v in SIGNS.items()}))
    solver.report_path.write_text(json.dumps(solver.report))  # no "arms" wrapper
    record = check_mapping.solve(solver.path)
    sign = record["joint_map"]["shoulder_lift"]["sign"]
    assert sign == -1 and type(sign) is int
    assert record["joint_map"]["wrist_flex"]["offset_deg"] == pytest.approx(20.0, abs=1e-6)


def test_ranges_beyond_the_model_are_flagged_but_still_solved(solver):
    _, model_q = build_report(solver.report_path, solver.cache, ranges=FOLLOWER_RANGES, raws=FOLLOWER_RAWS)
    rewrite(solver, lambda source: source.update(model_q_rad=model_q))
    record = check_mapping.solve(solver.path)
    flagged = {warning.split(":")[0] for warning in record["warnings"]}
    assert {"shoulder_pan", "shoulder_lift", "wrist_flex", "wrist_roll"} <= flagged
    assert "elbow_flex" not in flagged  # 97.8 degrees against the model's 96.8: inside the 5 degree slack
    assert all("beyond the MJCF range" in warning for warning in record["warnings"])
    assert record["limits"]["wrist_roll"] == pytest.approx([-math.pi, math.pi])
    assert check_mapping.main(["--solve", str(solver.path)]) == 0


def test_narrow_range_warns_and_unusable_range_is_refused(solver, capsys):
    narrow = dict(RANGES, wrist_roll=(2000, 2080))  # 80 ticks = 7.0 degrees: loadable, but barely
    _, model_q = build_report(
        solver.report_path, solver.cache, ranges=narrow, raws=dict(RAWS, wrist_roll=2040)
    )
    rewrite(solver, lambda source: source.update(model_q_rad=model_q))
    record = check_mapping.solve(solver.path)
    assert any(
        warning.startswith("wrist_roll:") and "degrees wide" in warning for warning in record["warnings"]
    )
    assert check_mapping.main(["--solve", str(solver.path)]) == 0
    assert "WARNING: wrist_roll:" in capsys.readouterr().out
    unusable = dict(
        RANGES, wrist_roll=(2000, 2060)
    )  # 60 ticks = 5.3 degrees: not wider than twice the margin
    _, model_q = build_report(
        solver.report_path, solver.cache, ranges=unusable, raws=dict(RAWS, wrist_roll=2030)
    )
    rewrite(solver, lambda source: source.update(model_q_rad=model_q))
    with pytest.raises(ValueError, match="not wider than twice"):
        check_mapping.solve(solver.path)


def test_calibration_cache_is_checked_when_present(solver):
    record = check_mapping.solve(solver.path, calibration_file=solver.cache)
    assert record["local_calibration_file_checked"] is True
    with pytest.raises(ValueError, match="Explicit calibration file is missing"):
        check_mapping.solve(solver.path, calibration_file=solver.cache.parent / "absent.json")
    rewrite(
        solver, mutate_report=lambda r: r["calibration"].update(path="/foreign-machine/cache/encoder.json")
    )
    assert check_mapping.solve(solver.path)["local_calibration_file_checked"] is False
    rewrite(solver, mutate_report=lambda r: r["calibration"].update(path=str(solver.cache)))
    solver.cache.write_text(solver.cache.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since"):
        check_mapping.solve(solver.path)


def test_yaml_helpers_refuse_bad_input_and_print_no_negative_zero():
    with pytest.raises(ValueError, match="1 or -1"):
        check_mapping.joint_map_yaml({name: {"sign": 2, "offset_deg": 0.0} for name in driver.JOINTS})
    with pytest.raises(ValueError, match="offset_deg"):
        check_mapping.joint_map_yaml(
            {name: {"sign": 1, "offset_deg": float("inf")} for name in driver.JOINTS}
        )
    with pytest.raises(ValueError, match="lo < hi"):
        check_mapping.limits_yaml({name: [1.0, -1.0] for name in driver.JOINTS})
    with pytest.raises(ValueError, match="finite"):
        check_mapping.limits_yaml({name: [1.0] for name in driver.JOINTS})
    text = check_mapping.joint_map_yaml({name: {"sign": 1, "offset_deg": -0.0004} for name in driver.JOINTS})
    assert text.count("offset_deg: 0.000}") == 5 and "-0.000" not in text


def _doc_yaml_block(marker):
    """The one fenced YAML block of docs/joint-calibration.md that contains `marker`."""
    blocks = re.findall(r"```yaml\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    matching = [block for block in blocks if marker in block]
    assert len(matching) == 1, f"expected one YAML block containing {marker!r}, found {len(matching)}"
    return matching[0]


def test_doc_placeholder_lines_are_the_driver_defaults():
    # Step 1 restores these before a rerun's sign test: its direction rule assumes sign 1, and the jog's range
    # check reads config/limits.yaml. A drifted copy would start the rerun from a wrong map or wrong limits.
    sections, current = {}, []
    for line in _doc_yaml_block("the placeholder joint lines").splitlines():
        if line.startswith("# config/"):
            current = sections.setdefault(line[2:].split(":", 1)[0], [])
        else:
            current.append(line)
    assert set(sections) == {"config/joint_map.yaml", "config/limits.yaml"}
    placeholder_map = yaml.safe_load("\n".join(sections["config/joint_map.yaml"]))
    assert placeholder_map == {name: driver.DEFAULT_JOINT_MAP[name] for name in driver.JOINTS}
    assert all(entry == {"sign": 1, "offset_deg": 0.0} for entry in placeholder_map.values())
    assert yaml.safe_load("\n".join(sections["config/limits.yaml"])) == driver.DEFAULT_LIMITS


def test_doc_solved_example_is_what_the_solver_prints(solver):
    _, model_q = build_report(solver.report_path, solver.cache, ranges=FOLLOWER_RANGES, raws=FOLLOWER_RAWS)
    rewrite(solver, lambda source: source.update(model_q_rad=model_q))
    record = check_mapping.solve(solver.path)
    example = _doc_yaml_block("replace its five joint lines")
    assert example == record["joint_map_yaml"] + "\n" + record["limits_yaml"]
