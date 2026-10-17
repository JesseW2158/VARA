"""Capture overhead marker pixels for independent, manually measured table checks. No arm access."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import cameras, markers  # noqa: E402


def capture(out: Path, *, fake: bool = False) -> dict[str, Any]:
    """Save one frame and detected pixels in a new directory; always release the camera.

    Physical coordinates must be measured separately. FAKE / FAKE_CAMERA cannot silently produce
    physical evidence; synthetic capture requires ``fake=True`` and is labeled in every artifact.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory")
    if not fake and any(os.environ.get(flag) == "1" for flag in ("FAKE", "FAKE_CAMERA")):
        raise ValueError("unset FAKE and FAKE_CAMERA for physical capture, or use --fake for offline checks")
    camera = cameras.FakeCamera() if fake else cameras.get_camera("overhead")
    try:
        if not fake and getattr(camera, "is_fake", True) is not False:
            raise ValueError("physical capture received a synthetic or unidentified camera")
        frame = camera.read()
        if frame is None or frame.size == 0:
            raise RuntimeError("overhead camera returned no frame")
        detections = markers.detect(frame)
        observations = [
            {"id": int(marker_id), "pixel": corners.mean(axis=0).astype(float).tolist()}
            for marker_id, corners in detections
        ]
        annotated = markers.draw(frame.copy(), detections)
        for entry in observations:
            u, v = entry["pixel"]
            cv2.drawMarker(annotated, (round(u), round(v)), (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        mode = "synthetic" if fake else "physical"
        if fake:
            cv2.putText(
                annotated,
                "SYNTHETIC - NOT TABLE MEASUREMENTS",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        report = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "camera": "overhead",
            "camera_index": getattr(camera, "index", None),
            "dictionary": "DICT_4X4_50",
            "image_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "pixel_origin": "top-left; u right, v down",
            "center_method": "mean of four detected marker corners",
            "markers": observations,
            "original_image": f"{mode}-original.png",
            "annotated_image": f"{mode}-annotated.png",
            "note": "Observed pixels only. Tape-measure table coordinates separately; no calibration saved.",
        }
        # Encode before creating the directory so an image failure cannot look like completed evidence.
        images = []
        for key, image in (("original_image", frame), ("annotated_image", annotated)):
            ok, data = cv2.imencode(".png", image)
            if not ok:
                raise OSError("could not encode capture image")
            images.append((report[key], data.tobytes()))
        out.mkdir(parents=True, exist_ok=False)
        for name, data in images:
            with (out / name).open("xb") as stream:
                stream.write(data)
        with (out / "markers.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        camera.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for images and markers.json")
    parser.add_argument("--fake", action="store_true", help="explicit offline capture; not physical evidence")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        report = capture(args.out, fake=args.fake)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {report['mode']} frame to {args.out}; {len(report['markers'])} marker(s) detected.")
    for entry in report["markers"]:
        print(f"ID {entry['id']}: pixel {entry['pixel']}")
    if not report["markers"]:
        print("No marker centers available; adjust the view and capture again in a new directory.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
