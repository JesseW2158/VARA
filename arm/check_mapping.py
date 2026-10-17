"""Compare measured A6 tip poses with the current joint map, using saved reports only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from arm import driver, ik

TOLERANCE_M = 0.015


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    files = [
        driver.JOINT_MAP_PATH,
        driver.LIMITS_PATH,
        driver.SCENE_PATH,
        ik.MODEL_PATH,
        ik.MODEL_PATH.parent / "so101_new_calib.xml",
    ]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, nargs="?", help="JSON containing measured poses")
    parser.add_argument(
        "--example", action="store_true", help="emit an incomplete template with null measurements"
    )
    parser.add_argument(
        "--calibration-file", type=Path, help="require this current encoder calibration file to match"
    )
    parser.add_argument("--out", type=Path, help="new JSON output path; never overwritten")
    args = parser.parse_args(argv)
    if args.example == (args.measurements is not None) or (args.example and args.calibration_file):
        parser.error("provide a measurements file or --example")
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
