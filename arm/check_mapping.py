"""Solve and check the A6 joint map from saved inspection reports; never connects to an arm or edits config/.

Two offline modes:

* ``python -m arm.check_mapping --solve solve.json`` takes one ``arm.inspect_hardware`` follower report
  captured at a known reference pose, that pose's model joint values and the signs from the A6 sign test,
  and prints ``offset_deg`` per joint (``model_deg - sign * lerobot_deg``) with the calibrated ``[lo, hi]``
  radians for config/limits.yaml, as the exact YAML blocks to paste. A timestamped record goes under
  runs/calibration/.
* ``python -m arm.check_mapping measurements.json`` compares five or more tape-measured tip positions with
  forward kinematics under the current joint map (the A6 accuracy check).

Public functions: ``solve_offsets``, ``limits_from_calibration``, ``solve``, ``joint_map_yaml``,
``limits_yaml``, ``format_solve``, ``check_measurements`` and ``example``. ``offset_deg`` and the LeRobot
degrees it is built from are config/joint_map.yaml's units; this module only prints and records them, nothing
here reaches an arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arm import driver, ik

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_M = 0.015
ENCODER_TICKS = 4095  # LeRobot degrees = (raw - mid) * 360 / 4095: the STS3215 has 4096 positions per turn
MODEL_SLACK_DEG = 5.0  # warn when a calibrated range reaches further than this beyond the MJCF range
NARROW_SLACK_DEG = 2.0  # warn when a range is not wider than twice the driver margin plus this
SOLVE_REQUIRED = ("hardware_report", "model_q_rad", "signs")
SOLVE_OPTIONAL = ("note",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_files() -> list[Path]:
    """The configuration and model files whose hashes every record carries."""
    return [
        driver.JOINT_MAP_PATH,
        driver.LIMITS_PATH,
        driver.SCENE_PATH,
        ik.MODEL_PATH,
        ik.MODEL_PATH.parent / "so101_new_calib.xml",
    ]


def example() -> dict:
    return {
        "measurement_kind": "physical",
        "instructions": "Replace every null with a tape measurement in meters, in the table frame. "
        "Keep the arm supported and unchanged between each measurement and its read-only report. "
        "Use five distinct poses spread across the workspace; never fill measurements from FK.",
        "poses": [{"hardware_report": f"pose-{i}.json", "tip_xyz": [None, None, None]} for i in range(1, 6)],
    }


def check_measurements(path: Path, *, calibration_file: Path | None = None) -> dict:
    """Check supplied evidence, not the present hardware; a passing result never enables motion."""
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or source.get("measurement_kind") not in ("physical", "synthetic"):
        raise ValueError("measurement_kind must explicitly be physical or synthetic")
    poses = source.get("poses")
    if not isinstance(poses, list) or len(poses) < 5:
        raise ValueError("At least five distinct measured poses are required")
    files = _config_files()
    hashes = {str(p): _sha(p) for p in files}
    if calibration_file is not None and not calibration_file.is_file():
        raise ValueError("Explicit calibration file is missing")
    results, distinct, calibrations, identities, saved_values = [], set(), set(), set(), set()
    for index, pose in enumerate(poses, 1):
        if not isinstance(pose, dict) or not isinstance(pose.get("hardware_report"), str):
            raise ValueError(f"Pose {index}: hardware_report must name a saved JSON file")
        measured = pose.get("tip_xyz")
        if (
            not isinstance(measured, list)
            or len(measured) != 3
            or any(type(v) not in (int, float) for v in measured)
        ):
            raise ValueError(
                f"Pose {index}: tip_xyz needs three finite measured meters; nulls are incomplete"
            )
        try:
            measured = [float(v) for v in measured]
        except OverflowError:
            raise ValueError(f"Pose {index}: tip_xyz needs finite measured meters") from None
        if not all(math.isfinite(v) for v in measured):
            raise ValueError(f"Pose {index}: tip_xyz needs finite measured meters")
        report_path = (path.parent / pose["hardware_report"]).resolve()
        report_bytes = report_path.read_bytes()
        document = json.loads(report_bytes)
        report = document.get("arms", {}).get("follower") if "arms" in document else document
        q = driver.map_inspection_joints(report)
        calibration = report["calibration"]
        fingerprint = calibration["sha256"]
        calibrations.add(fingerprint)
        saved_values.add(json.dumps(calibration["saved"], sort_keys=True))
        identities.add((report["lerobot_id"], report.get("usb", {}).get("serial_number")))
        distinct.add(tuple(report["motors"][j]["position_raw"] for j in driver.JOINTS))
        cached = calibration_file
        if cached is None and isinstance(calibration.get("path"), str):
            cached = report_path.parent / calibration["path"]
        cache_checked = cached is not None and cached.is_file()
        if cache_checked and _sha(cached) != fingerprint:
            raise ValueError(f"Pose {index}: calibration file changed since the inspection; repeat A6")
        predicted, _ = ik.fk(q)
        error = math.dist(predicted, measured)
        if not math.isfinite(error):
            raise ValueError(f"Pose {index}: error is not finite; check measurement units")
        results.append(
            {
                "pose": index,
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "q_rad": q,
                "measured_tip_xyz": measured,
                "predicted_tip_xyz": list(predicted),
                "error_m": error,
                "within_tolerance": error <= TOLERANCE_M + 1e-12,
                "local_calibration_file_checked": cache_checked,
            }
        )
    if len(distinct) < 5:
        raise ValueError("At least five distinct joint poses are required; duplicate reports do not count")
    if len(calibrations) != 1 or len(identities) != 1 or len(saved_values) != 1:
        raise ValueError("All poses must come from the same follower and encoder calibration fingerprint")
    if hashes != {str(p): _sha(p) for p in files}:
        raise ValueError("Configuration or model changed during the check; run it again")
    return {
        "measurement_kind": source["measurement_kind"],
        "tolerance_m": TOLERANCE_M,
        "all_within_tolerance": all(row["within_tolerance"] for row in results),
        "mean_error_m": sum(row["error_m"] / len(results) for row in results),
        "max_error_m": max(row["error_m"] for row in results),
        "poses": results,
        "calibration_sha256": next(iter(calibrations)),
        "config_and_model_sha256": hashes,
        "measurements_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scope": "Saved evidence compared with current configuration. Missing foreign cache paths are "
        "not checked locally. Review measurement accuracy and workspace coverage. No motion is authorized.",
    }


# ------------------------------------------------------------------------------------------------------------
# Offset and limit solver
# ------------------------------------------------------------------------------------------------------------


def _finite_floats(values: Any, count: int, what: str) -> list[float]:
    """`count` finite numbers as floats; bools, NaN, infinities, huge ints and non-sequences are refused."""
    message = f"{what} needs {count} finite numbers"
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(message)
    try:
        items = list(values)
    except TypeError:
        raise ValueError(message) from None
    if len(items) != count or any(isinstance(v, bool) or not isinstance(v, numbers.Real) for v in items):
        raise ValueError(message)
    try:
        out = [float(v) for v in items]
    except OverflowError:
        raise ValueError(message) from None
    if not all(math.isfinite(v) for v in out):
        raise ValueError(message)
    return out


def _reference_q(values: Any) -> list[float]:
    """The reference pose as five model radians inside the MJCF ranges (degrees by mistake are refused)."""
    q = _finite_floats(values, len(driver.JOINTS), "model_q_rad")
    for name, value in zip(driver.JOINTS, q, strict=True):
        lo, hi = driver.DEFAULT_LIMITS[name]
        if not lo <= value <= hi:
            raise ValueError(
                f"model_q_rad: {name} = {value} is outside the model range [{lo}, {hi}]; "
                "the reference pose is model radians, not degrees"
            )
    return q


def _signs(values: Any) -> dict[str, int]:
    """`{joint: 1 or -1}` for exactly the five joints, as the sign test recorded them."""
    if not isinstance(values, dict) or set(values) != set(driver.JOINTS):
        raise ValueError(f"signs must map exactly the joints {driver.JOINTS} to 1 or -1")
    out = {}
    for name in driver.JOINTS:
        sign = values[name]
        if isinstance(sign, bool) or not isinstance(sign, numbers.Real) or sign not in (1, -1):
            raise ValueError(f"signs: {name} must be 1 or -1 from the sign test, got {sign!r}")
        out[name] = int(sign)
    return out


def _joint_map_entries(joint_map: Any) -> dict[str, dict[str, Any]]:
    """Per joint `{"sign": +-1, "offset_deg": float}` from a map shaped like `driver.joint_map()`."""
    if not isinstance(joint_map, dict):
        raise ValueError("joint_map must map each joint to {sign, offset_deg}")
    signs, offsets = {}, {}
    for name in driver.JOINTS:
        entry = joint_map.get(name)
        if not isinstance(entry, dict) or "sign" not in entry or "offset_deg" not in entry:
            raise ValueError(f"joint_map: {name} must be a mapping with sign and offset_deg")
        signs[name], offsets[name] = entry["sign"], entry["offset_deg"]
    checked = _signs(signs)
    out = {}
    for name in driver.JOINTS:
        offset = _finite_floats([offsets[name]], 1, f"joint_map: {name}.offset_deg")[0]
        out[name] = {"sign": checked[name], "offset_deg": offset}
    return out


def _follower_report(document: Any, label: str) -> Any:
    """The follower entry of a saved `arm.inspect_hardware` document, or the document itself when bare."""
    if not isinstance(document, dict):
        raise ValueError(f"{label}: expected a JSON object")
    if "arms" not in document:
        return document
    arms = document["arms"]
    report = arms.get("follower") if isinstance(arms, dict) else None
    if report is None:
        raise ValueError(f"{label}: no follower entry; the solver needs a read-only follower inspection")
    if isinstance(report, dict) and "error" in report and "motors" not in report:
        raise ValueError(f"{label}: the follower read failed ({report['error']}); capture a new report")
    return report


def _lerobot_degrees(report: Any) -> dict[str, float]:
    """Each arm joint's LeRobot reading in degrees, from a follower report the driver accepts.

    `driver.map_inspection_joints` does the validation: a read-only follower report, encoder calibration
    present, valid and matching the hardware, the follower id of config/joint_map.yaml, and every raw position
    inside its saved range and consistent with `calibrated_position_rad`. That field is LeRobot's degrees in
    radians, (raw - mid) * 360 / 4095 with mid the saved range midpoint; LeRobot 0.6.2 ignores drive_mode in
    degrees mode, so the formula holds for every saved calibration.
    """
    driver.map_inspection_joints(report)
    return {
        name: math.degrees(float(report["motors"][name]["calibrated_position_rad"])) for name in driver.JOINTS
    }


def solve_offsets(
    report: dict[str, Any], reference_q: Sequence[float], signs: dict[str, int]
) -> dict[str, float]:
    """`offset_deg` per joint from a follower report captured at a pose whose model joints are `reference_q`.

    With the sign from the A6 sign test, offset_deg = degrees(q_model) - sign * lerobot_deg, so that the
    driver's q_rad = radians(sign * lerobot_deg + offset_deg) reproduces the model pose at the reference.
    `reference_q` is five model radians inside the MJCF ranges; `signs` is {joint: 1 or -1}. The report is
    validated as in `driver.map_inspection_joints`, so it must come from the follower config/joint_map.yaml
    names. Nothing is written anywhere.
    """
    lerobot_deg = _lerobot_degrees(report)
    q = _reference_q(reference_q)
    checked = _signs(signs)
    return {
        name: math.degrees(q[i]) - checked[name] * lerobot_deg[name] for i, name in enumerate(driver.JOINTS)
    }


def limits_from_calibration(report: dict[str, Any], joint_map: dict[str, Any]) -> dict[str, list[float]]:
    """Calibrated joint limits `{joint: [lo, hi]}` in radians from a report's saved encoder calibration.

    LeRobot's degrees span +-(range_max - range_min) / 2 * 360 / 4095 around the saved midpoint; both ends go
    through radians(sign * deg + offset_deg) and are ordered lo < hi, so a sign of -1 swaps them. `joint_map`
    is shaped like `driver.joint_map()` ({joint: {"sign", "offset_deg"}}; other keys are ignored). The report
    is validated as in `solve_offsets`. A range the driver would refuse to load (not wider than twice its
    3 degree margin) is an error, not a proposal.
    """
    _lerobot_degrees(report)
    entries = _joint_map_entries(joint_map)
    saved = report["calibration"]["saved"]
    out = {}
    for name in driver.JOINTS:
        half_deg = (saved[name]["range_max"] - saved[name]["range_min"]) / 2 * 360 / ENCODER_TICKS
        sign, offset = entries[name]["sign"], entries[name]["offset_deg"]
        lo, hi = sorted(math.radians(sign * end + offset) for end in (-half_deg, half_deg))
        if hi - lo <= 2 * driver.MARGIN_RAD:
            raise ValueError(
                f"{name}: calibrated range [{lo:.4f}, {hi:.4f}] rad is not wider than twice the driver's "
                "3 degree margin; the encoder calibration is unusable, recalibrate"
            )
        out[name] = [lo, hi]
    return out


def _review_warnings(limits: dict[str, list[float]]) -> list[str]:
    """Ranges to look at before pasting: too narrow to move in, or reaching well beyond the MJCF range."""
    out = []
    for name in driver.JOINTS:
        lo, hi = limits[name]
        if hi - lo < 2 * driver.MARGIN_RAD + math.radians(NARROW_SLACK_DEG):
            out.append(
                f"{name}: calibrated range is only {math.degrees(hi - lo):.1f} degrees wide; the driver "
                "keeps a 3 degree margin at each end, leaving almost no travel"
            )
        model_lo, model_hi = driver.DEFAULT_LIMITS[name]
        beyond = max(math.degrees(model_lo - lo), math.degrees(hi - model_hi))
        if beyond > MODEL_SLACK_DEG:
            out.append(
                f"{name}: calibrated range [{lo:.4f}, {hi:.4f}] reaches {beyond:.1f} degrees beyond the MJCF "
                f"range [{model_lo}, {model_hi}]; the model cannot represent those positions, so check the "
                "sign and offset, or narrow the limit deliberately"
            )
    return out


def joint_map_yaml(joint_map: dict[str, Any]) -> str:
    """The five joint lines of config/joint_map.yaml for `joint_map` (3 decimals) under a paste comment."""
    entries = _joint_map_entries(joint_map)
    lines = [
        "# config/joint_map.yaml: replace its five joint lines; keep lerobot_id, lerobot_robot_type, gripper"
    ]
    for name in driver.JOINTS:
        offset = round(entries[name]["offset_deg"], 3) + 0.0  # + 0.0 turns -0.0 into 0.0
        lines.append(f"{name}: {{sign: {entries[name]['sign']}, offset_deg: {offset:.3f}}}")
    return "\n".join(lines) + "\n"


def limits_yaml(limits: dict[str, Sequence[float]]) -> str:
    """The five joint lines of config/limits.yaml for `limits` (4 decimals), headed by a paste comment."""
    if not isinstance(limits, dict):
        raise ValueError("limits must map each joint to [lo, hi] radians")
    lines = ["# config/limits.yaml: replace its five joint lines"]
    for name in driver.JOINTS:
        lo, hi = _finite_floats(limits.get(name), 2, f"limits: {name}")
        if not lo < hi:
            raise ValueError(f"limits: {name} needs lo < hi, got [{lo}, {hi}]")
        lines.append(f"{name}: [{lo:.4f}, {hi:.4f}]")
    return "\n".join(lines) + "\n"


def solve(path: Path, *, calibration_file: Path | None = None) -> dict[str, Any]:
    """Solve the joint map and limits described by a solve file; returns the record and changes nothing.

    The file is a JSON object with `hardware_report` (an `arm.inspect_hardware` document, relative to the file
    or absolute), `model_q_rad` (the reference pose's five model joints in radians) and `signs` ({joint: 1 or
    -1} from the sign test); `note` is optional free text. As in `check_measurements`, the encoder calibration
    file the report names (or `calibration_file`) must still hash to the report's fingerprint when it exists
    locally, and the record carries the hashes of the configuration and model files. The record also holds the
    reference pose's FK tip for a tape check, both YAML blocks, and review warnings for ranges that are too
    narrow or reach well beyond the MJCF ranges. Nothing is written to config/ or sent to an arm.
    """
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict):
        raise ValueError("solve file must be a JSON object with hardware_report, model_q_rad and signs")
    missing = [key for key in SOLVE_REQUIRED if key not in source]
    unknown = sorted(set(source) - set(SOLVE_REQUIRED) - set(SOLVE_OPTIONAL))
    if missing or unknown:
        raise ValueError(
            f"solve file keys are {', '.join(SOLVE_REQUIRED)} and optionally note; "
            f"missing {missing}, unknown {unknown}"
        )
    if not isinstance(source["hardware_report"], str) or not source["hardware_report"]:
        raise ValueError("hardware_report must name a saved arm.inspect_hardware JSON file")
    note = source.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError("note must be a string")
    if calibration_file is not None and not calibration_file.is_file():
        raise ValueError("Explicit calibration file is missing")
    files = _config_files()
    hashes = {str(p): _sha(p) for p in files}
    report_path = (path.parent / source["hardware_report"]).resolve()
    report_bytes = report_path.read_bytes()
    report = _follower_report(json.loads(report_bytes), source["hardware_report"])
    q = _reference_q(source["model_q_rad"])
    signs = _signs(source["signs"])
    lerobot_deg = _lerobot_degrees(report)
    offsets = solve_offsets(report, q, signs)
    joint_map = {name: {"sign": signs[name], "offset_deg": offsets[name]} for name in driver.JOINTS}
    limits = limits_from_calibration(report, joint_map)
    calibration = report["calibration"]
    cached = calibration_file
    if cached is None and isinstance(calibration.get("path"), str):
        cached = report_path.parent / calibration["path"]
    cache_checked = cached is not None and cached.is_file()
    if cache_checked and _sha(cached) != calibration["sha256"]:
        raise ValueError("calibration file changed since the inspection; capture a new reference-pose report")
    tip, pitch = ik.fk(q)
    usb = report.get("usb")
    current_map = driver.joint_map()
    joints = {}
    for i, name in enumerate(driver.JOINTS):
        lo, hi = limits[name]
        joints[name] = {
            "sign": signs[name],
            "lerobot_deg": lerobot_deg[name],
            "model_deg": math.degrees(q[i]),
            "offset_deg": offsets[name],
            "range_raw": [calibration["saved"][name]["range_min"], calibration["saved"][name]["range_max"]],
            "limits_rad": [lo, hi],
            "limits_deg": [math.degrees(lo), math.degrees(hi)],
            "model_range_rad": list(driver.DEFAULT_LIMITS[name]),
        }
    record = {
        "kind": "a6_joint_map_solve",
        "solved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solve_file": str(path.resolve()),
        "solve_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "note": note,
        "report": str(report_path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "report_captured_at": report.get("captured_at"),
        "lerobot_id": report.get("lerobot_id"),
        "usb_serial_number": usb.get("serial_number") if isinstance(usb, dict) else None,
        "calibration_sha256": calibration["sha256"],
        "calibration_path": calibration.get("path"),
        "local_calibration_file_checked": cache_checked,
        "reference": {"model_q_rad": q, "tip_xyz": list(tip), "pitch_rad": pitch},
        "joints": joints,
        "joint_map": joint_map,
        "limits": limits,
        "joint_map_yaml": joint_map_yaml(joint_map),
        "limits_yaml": limits_yaml(limits),
        "current_joint_map": {name: dict(current_map[name]) for name in driver.JOINTS},
        "current_limits": driver.limits(),
        "warnings": _review_warnings(limits),
        "config_and_model_sha256": hashes,
        "scope": "Proposed joint map and limits from one reference pose and its encoder calibration. "
        "Nothing in config/ was changed: paste the YAML deliberately, then run the five-pose measurement "
        "check. Recalibrating the encoders invalidates this record. No motion is authorized.",
    }
    if hashes != {str(p): _sha(p) for p in files}:
        raise ValueError("Configuration or model changed during the solve; run it again")
    return record


def format_solve(record: dict[str, Any], record_path: Path | None = None) -> str:
    """The console summary of a `solve` record: provenance, tip to tape-check, table and both YAML blocks."""
    ref = record["reference"]
    tip = ref["tip_xyz"]
    q_text = json.dumps([round(v, 4) for v in ref["model_q_rad"]])
    lines = [
        f"A6 solve from {record['report']}",
        f"  captured {record['report_captured_at']}, follower {record['lerobot_id']} "
        f"(USB {record['usb_serial_number']}), encoder calibration {record['calibration_sha256']}",
        f"  reference model_q_rad {q_text} -> FK tip (x, y, z) = "
        f"({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}) m, pitch {math.degrees(ref['pitch_rad']):+.1f} deg",
        "  Tape-measure the physical jaw-tip midpoint against that tip before trusting the offsets.",
        "",
        f"{'joint':<14}{'sign':>5}{'lerobot_deg':>13}{'model_deg':>11}{'offset_deg':>12}"
        f"{'lo_rad':>10}{'hi_rad':>10}{'(lo_deg, hi_deg)':>20}",
    ]
    for name, row in record["joints"].items():
        lo, hi = row["limits_rad"]
        span = f"({math.degrees(lo):.1f}, {math.degrees(hi):.1f})"
        lines.append(
            f"{name:<14}{row['sign']:>+5d}{row['lerobot_deg']:>13.3f}{row['model_deg']:>11.3f}"
            f"{row['offset_deg']:>+12.3f}{lo:>10.4f}{hi:>10.4f}{span:>20}"
        )
    lines += ["", record["joint_map_yaml"].rstrip("\n"), "", record["limits_yaml"].rstrip("\n"), ""]
    lines += [f"WARNING: {warning}" for warning in record["warnings"]]
    if record_path is not None:
        lines.append(f"Record: {record_path}")
    lines.append(
        "Nothing in config/ was changed. Paste the blocks deliberately, then run the five-pose measurement "
        "check."
    )
    return "\n".join(lines) + "\n"


def _default_record_path() -> Path:
    """runs/calibration/solve-<UTC stamp>.json under VARA_RUNS_DIR or the repo's runs/."""
    runs = Path(os.environ.get("VARA_RUNS_DIR") or ROOT / "runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return runs / "calibration" / f"solve-{stamp}.json"


def _solve_main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        record = solve(args.solve, calibration_file=args.calibration_file)
        out = args.out or _default_record_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    print(format_solve(record, out), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arm.check_mapping",
        description="Solve the A6 joint map from a reference-pose report, or check measured tip poses "
        "against forward kinematics. Saved reports only: nothing here connects to an arm or edits config/.",
        epilog='The --solve file is JSON: {"hardware_report": "reference-pose.json", "model_q_rad": [five '
        'radians], "signs": {joint: 1 or -1}, "note": optional}. See docs/joint-calibration.md.',
    )
    parser.add_argument("measurements", type=Path, nargs="?", help="JSON containing measured poses")
    parser.add_argument(
        "--example", action="store_true", help="emit an incomplete template with null measurements"
    )
    parser.add_argument(
        "--solve",
        type=Path,
        metavar="SOLVE_JSON",
        help="solve offset_deg and the calibrated limits from a reference-pose inspection report; prints "
        "both YAML blocks and writes runs/calibration/solve-<stamp>.json (or --out)",
    )
    parser.add_argument(
        "--calibration-file", type=Path, help="require this current encoder calibration file to match"
    )
    parser.add_argument("--out", type=Path, help="new JSON output path; never overwritten")
    args = parser.parse_args(argv)
    modes = [args.example, args.measurements is not None, args.solve is not None]
    if sum(modes) != 1 or (args.example and args.calibration_file):
        parser.error("provide a measurements file, --solve SOLVE_JSON, or --example")
    if args.solve is not None:
        return _solve_main(parser, args)
    try:
        result = (
            example()
            if args.example
            else check_measurements(args.measurements, calibration_file=args.calibration_file)
        )
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="\n") as output:
                output.write(encoded)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    print(encoded, end="")
    return 0 if args.example or result["all_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
