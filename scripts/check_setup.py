"""Inspect local setup without opening hardware, cameras, or network connections."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    name: name
    for name in ("fastapi", "uvicorn", "httpx", "openai", "daytona", "mujoco", "numpy", "jsonschema")
}
DEPENDENCIES.update({"python-dotenv": "dotenv", "pyyaml": "yaml", "opencv-python-headless": "cv2"})
OPTIONAL = {name: name for name in ("lerobot", "torch", "gymnasium", "av")}
OPTIONAL["stable-baselines3"] = "stable_baselines3"
CONFIGS = ("joint_map.yaml", "limits.yaml", "scene.yaml", "cameras.yaml", "objects.yaml", "fake_scene.json")
ARTIFACTS = (
    "config/homography.npy",
    "sim/model/scene_ours.xml",
    "sim/model/so101_new_calib.xml",
    "skills/_starter/hand_over.json",
    "skills/_starter/place_in_tray.json",
    "web/index.html",
    "web/voice.html",
)
MANUAL = [
    "A6: measure directions/offsets and five tip poses within 1.5 cm; encoder calibration is insufficient.",
    "A11: measure gripper contact readings and confirm each prop can be held reliably.",
    "B4/B5: identify physical cameras and measure table geometry/homography; file presence is insufficient.",
    "A13: complete the supported-pose startup and physical STOP drill before motion acceptance.",
]


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _port(value: object, secrets: tuple[str, ...] = ()) -> str | None:
    # Do not echo arbitrary environment values: a misplaced API key must never reach the report.
    if not isinstance(value, str) or any(secret in value for secret in secrets):
        return None
    return value if re.fullmatch(r"COM\d+|/dev/tty[\w./-]+", value, re.I) else None


def build_report(root: Path, env: Mapping[str, str], hardware_report: Path | None = None) -> dict:
    """Report discoverable local prerequisites; this is never a motion-readiness certificate."""
    secrets = tuple(
        env[name].strip() for name in ("XAI_API_KEY", "DAYTONA_API_KEY") if env.get(name, "").strip()
    )
    dependencies = {package: _available(module) for package, module in DEPENDENCIES.items()}
    optional = {p: _available(m) for p, m in OPTIONAL.items()}
    configs, data = {}, {}
    for name in CONFIGS:
        path = root / "config" / name
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                parsed = json.loads(text)
            else:
                import yaml

                parsed = yaml.safe_load(text)
            valid = isinstance(parsed, dict) and bool(parsed)
            data[name] = parsed if valid else {}
            configs[name] = {"valid_mapping": valid, "placeholder_notice": "placeholder" in text.lower()}
        except Exception:
            configs[name] = {"valid_mapping": False, "placeholder_notice": None}
    artifacts = {name: (root / name).is_file() for name in ARTIFACTS}
    cameras = data.get("cameras.yaml", {})
    configured_cameras = {}
    for name in ("overhead", "wrist"):
        entry = cameras.get(name, {})
        raw = env.get(f"{name.upper()}_CAM") or (entry.get("index") if isinstance(entry, dict) else None)
        valid = re.fullmatch(r"\d{1,3}", str(raw)) and not any(secret in str(raw) for secret in secrets)
        configured_cameras[name] = int(str(raw)) if valid else None
    ports = {role: _port(env.get(f"{role.upper()}_PORT"), secrets) for role in ("follower", "leader")}
    keys = {name: bool(env.get(name, "").strip()) for name in ("XAI_API_KEY", "DAYTONA_API_KEY")}
    fakes = {
        name: env.get(name) == "1"
        for name in ("FAKE", "FAKE_LLM", "FAKE_SANDBOX", "FAKE_ROBOT", "FAKE_CAMERA")
    }
    blockers = [
        f"Set {name} in .env; its value is never included here."
        for name, present in keys.items()
        if not present
    ]
    blockers += [
        f"Set a valid {role.upper()}_PORT after identifying the USB board."
        for role, port in ports.items()
        if not port
    ]
    if ports["leader"] and ports["leader"].lower() == (ports["follower"] or "").lower():
        blockers.append("Leader and follower ports must differ.")
    blockers += [
        f"Identify and set {name.upper()}_CAM." for name, index in configured_cameras.items() if index is None
    ]
    if configured_cameras["overhead"] is not None and len(set(configured_cameras.values())) == 1:
        blockers.append("Overhead and wrist cameras need distinct physical indexes.")
    if not optional["lerobot"]:
        blockers.append("Install lerobot[feetech] on the hardware laptop.")
    if any(fakes.values()):
        blockers.append("Fake mode is enabled; disable the relevant FAKE flags for live acceptance.")
    blockers += [
        f"Measure and replace placeholders in config/{name}."
        for name in CONFIGS
        if configs[name]["placeholder_notice"]
    ]
    historical = None
    if hardware_report is not None:
        historical = {"historical_only": True, "current_hardware_checked": False, "arms": {}}
        try:
            saved = json.loads(hardware_report.read_text(encoding="utf-8"))
            if not isinstance(saved.get("arms"), dict) or not saved["arms"]:
                raise ValueError("Missing arms")
            for role in ("follower", "leader"):
                if role not in saved["arms"]:
                    historical["arms"][role] = {"report_present": False}
                    continue
                arm = saved["arms"][role]
                match = arm.get("calibration", {}).get("matches_hardware")
                historical["arms"][role] = {
                    "port": _port(arm.get("port"), secrets),
                    "calibration_matched": match is True,
                }
                if match is not True:
                    blockers.append(
                        f"Historical {role} report lacks matching encoder calibration; recheck it."
                    )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            historical["error"] = "Report missing, malformed, or incomplete; no hardware was checked."
            blockers.append(historical["error"])
    return {
        "non_invasive": True,
        "dependency_check": "Module discovery only; no hardware or service imports.",
        "offline_prerequisites_ok": all(dependencies.values())
        and all(artifacts.values())
        and all(c["valid_mapping"] for c in configs.values()),
        "dependencies": dependencies,
        "optional_dependencies": optional,
        "configs": configs,
        "artifacts_present": artifacts,
        "optional_artifacts_present": {
            p: (root / p).is_file() for p in ("rl/policies/handoff_best.zip", "web/static/rl_sim.mp4")
        },
        "api_keys_configured_not_authenticated": keys,
        "ports_not_probed": ports,
        "camera_indexes_not_opened": configured_cameras,
        "fake_modes": fakes,
        "historical_hardware": historical,
        "live_configuration_blockers": blockers,
        "manual_acceptance_required": list(MANUAL),
        "acceptance_note": "Removing comments proves no calibration. Physical acceptance is manual.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="save JSON to a new path; never overwrite")
    parser.add_argument(
        "--hardware-report", type=Path, help="read a historical arm.inspect_hardware JSON report"
    )
    parser.add_argument(
        "--require-live-config",
        action="store_true",
        help="also exit 1 for live configuration blockers; does not certify physical acceptance",
    )
    args = parser.parse_args(argv)
    if _available("dotenv"):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    report = build_report(ROOT, os.environ, args.hardware_report)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="\n") as output:
                output.write(encoded)
        except OSError:
            parser.error("Cannot create --out; choose a new, writable report path.")
    print(encoded, end="")
    return int(
        not report["offline_prerequisites_ok"]
        or (args.require_live_config and bool(report["live_configuration_blockers"]))
    )


if __name__ == "__main__":
    raise SystemExit(main())
