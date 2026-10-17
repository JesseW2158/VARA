"""Table calibration (B5): the pixel-to-table homography, the height correction and the synthetic camera.

The overhead camera is calibrated once per table setup: the 8 calibration markers (ids 0-7, 6 cm) are
taped around the workspace, their centers tape-measured into config/scene.yaml `calib_markers`, and
`fit_homography` maps marker pixels to table meters with `cv2.findHomography(..., RANSAC)`. A table point
is `H @ [u, v, 1]` normalized (`pixel_to_table`). A marker on top of an object of height h appears farther
from the point under the lens than the object is, which `height_correct` undoes:

    p = c + (p_table - c) * (1 - h / Hc)

with `c` the table point under the lens and `Hc` the lens height (config/scene.yaml `camera`).

Synthetic camera model (config/homography.npy, vision.cameras.FakeCamera)
------------------------------------------------------------------------
A pinhole camera at C = (c.x, c.y, Hc) looking straight down, focal length FOCAL_PX = 450 px, image
640x480, principal point (320, 240), no distortion. Image axes: u runs along world -y (image right is the
arm's right) and v along world -x (image up points away from the arm base), i.e. the frame is what a
person standing behind the arm sees looking down at the table. A world point (x, y, z), z < Hc, projects to

    u = 320 - FOCAL_PX * (y - c.y) / (Hc - z)
    v = 240 - FOCAL_PX * (x - c.x) / (Hc - z)

The camera frame X = -y_w, Y = -x_w, Z = -z_w is right-handed (X x Y = Z), as a real camera is, so markers
keep their handedness and ArUco decodes them (a mirrored marker is never detected). On the table (z = 0)
the map is affine and `findHomography` recovers it from the 8 marker centers to machine precision. A
marker at height h projects where the table point `c + (p - c) * Hc / (Hc - h)` would, exactly what the
height correction undoes.

Why 450 px and not the 600 px in the brief: at f = 600 the lateral half-field at the table is only
320 * 0.65 / 600 = 0.347 m, but calibration markers 0, 2 and 4 sit 0.39 m from c.y = -0.05 (y = +0.34)
and markers 4-7 sit exactly on the top edge (x = 0.46), so the brief's camera could not see its own
calibration markers. 450 px is a 70 degree horizontal field (a Logitech C920 class webcam) and keeps every
marker at least 29 px inside the frame.

CLI: `python -m vision.calibrate --fake` writes the synthetic config/homography.npy (deterministic, so the
committed file is rewritten byte for byte); `--live` fits from the overhead camera (hardware) and
`--check` reports the error of the saved file on a live frame.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import yaml

__all__ = [
    "FOCAL_PX",
    "HOMOGRAPHY_PATH",
    "IMAGE_SIZE",
    "PRINCIPAL",
    "calib_marker_centers",
    "calibration_error",
    "camera_pose",
    "fit_homography",
    "height_correct",
    "load_homography",
    "load_scene_cfg",
    "pixel_to_table",
    "pixels_to_table",
    "project",
    "save_homography",
    "synthetic_homography",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "config" / "scene.yaml"
HOMOGRAPHY_PATH = ROOT / "config" / "homography.npy"

FOCAL_PX = 450.0
IMAGE_SIZE = (640, 480)  # width, height
PRINCIPAL = (320.0, 240.0)
RANSAC_THRESHOLD_M = 0.02  # a marker center farther than 2 cm from the fit is an outlier


def load_scene_cfg(path: pathlib.Path | str | None = None) -> dict[str, Any]:
    """config/scene.yaml as a dict (workspace, zones, camera, calib_markers, home_q)."""
    file = pathlib.Path(path) if path is not None else SCENE_PATH
    cfg = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{file}: expected a mapping")
    return cfg


def camera_pose(scene_cfg: Mapping[str, Any] | None = None) -> tuple[tuple[float, float], float]:
    """`(c, Hc)`: the table point under the lens and the lens height from config/scene.yaml `camera`."""
    cfg = scene_cfg if scene_cfg is not None else load_scene_cfg()
    cam = cfg.get("camera") or {}
    try:
        c = (float(cam["c"][0]), float(cam["c"][1]))
        hc = float(cam["Hc"])
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError("config/scene.yaml needs camera: {Hc: float, c: [x, y]}") from exc
    if not (hc > 0.0 and math.isfinite(hc) and all(math.isfinite(v) for v in c)):
        raise ValueError(f"bad camera pose c={c}, Hc={hc}")
    return c, hc


def calib_marker_centers(scene_cfg: Mapping[str, Any] | None = None) -> dict[int, tuple[float, float]]:
    """`{id: (x, y)}` tape-measured centers of the calibration markers from config/scene.yaml."""
    cfg = scene_cfg if scene_cfg is not None else load_scene_cfg()
    raw = cfg.get("calib_markers") or {}
    centers = {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    if len(centers) < 4:
        raise ValueError("config/scene.yaml calib_markers needs at least 4 markers")
    return dict(sorted(centers.items()))


def project(
    points: Iterable[Sequence[float]],
    c: Sequence[float],
    hc: float,
    *,
    f: float = FOCAL_PX,
    principal: Sequence[float] = PRINCIPAL,
) -> np.ndarray:
    """Synthetic camera: world (x, y, z) points, shape (N, 3), to pixels (u, v), shape (N, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    depth = hc - pts[:, 2]
    if np.any(depth <= 1e-6):
        raise ValueError("a point at or above the camera cannot be projected")
    u = principal[0] - f * (pts[:, 1] - c[1]) / depth
    v = principal[1] - f * (pts[:, 0] - c[0]) / depth
    return np.stack([u, v], axis=1)


def fit_homography(
    pixels: Iterable[Sequence[float]],
    table_xy: Iterable[Sequence[float]],
    *,
    ransac_threshold: float = RANSAC_THRESHOLD_M,
) -> np.ndarray:
    """Pixel -> table homography (3x3 float64, H[2, 2] = 1) from matched points, RANSAC then refined.

    Needs at least 4 pairs; the 8 calibration markers average out most lens distortion.
    """
    src = np.asarray(list(pixels), dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(list(table_xy), dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError(f"{len(src)} pixel points but {len(dst)} table points")
    if len(src) < 4:
        raise ValueError(f"a homography needs at least 4 points, got {len(src)}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("non-finite calibration points")
    h, _mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if h is None or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        raise ValueError("findHomography failed: are the points spread over the table?")
    return np.asarray(h, dtype=np.float64) / h[2, 2]


def pixel_to_table(h: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Map one pixel through H: `H @ [u, v, 1]` normalized, as (x, y) in meters."""
    x, y, w = np.asarray(h, dtype=np.float64) @ np.array([float(u), float(v), 1.0])
    if abs(w) < 1e-12:
        raise ValueError(f"pixel ({u}, {v}) maps to infinity")
    return float(x / w), float(y / w)


def pixels_to_table(h: np.ndarray, pixels: Iterable[Sequence[float]]) -> np.ndarray:
    """Vectorized `pixel_to_table`: (N, 2) pixels -> (N, 2) table points."""
    pts = np.asarray(list(pixels), dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(h, dtype=np.float64).T
    return hom[:, :2] / hom[:, 2:3]


def height_correct(p: Sequence[float], c: Sequence[float], h: float, hc: float) -> tuple[float, float]:
    """Undo the parallax of a marker at height h: `p = c + (p_table - c) * (1 - h / Hc)`."""
    if not 0.0 <= h < hc:
        raise ValueError(f"marker height {h} must be in [0, Hc={hc})")
    k = 1.0 - h / hc
    return (c[0] + (p[0] - c[0]) * k, c[1] + (p[1] - c[1]) * k)


def calibration_error(
    h: np.ndarray,
    dets: Iterable[tuple[int, np.ndarray]],
    markers: Mapping[int, Sequence[float]],
) -> float:
    """Mean distance in cm between each detected calibration marker's mapped center and its measured one.

    `dets` is what `vision.markers.detect` returns; `markers` maps marker id -> (x, y) in meters (the
    `calib_markers` of config/scene.yaml). Detections of other ids are ignored. Raises ValueError when no
    calibration marker is in view, so a blocked camera never reports 0.0 cm.
    """
    errors: list[float] = []
    for marker_id, corners in dets:
        measured = markers.get(int(marker_id))
        if measured is None:
            continue
        u, v = np.asarray(corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
        x, y = pixel_to_table(h, u, v)
        errors.append(math.hypot(x - float(measured[0]), y - float(measured[1])))
    if not errors:
        raise ValueError("no calibration marker (ids 0-7) detected")
    return float(np.mean(errors)) * 100.0


def synthetic_detections(
    scene_cfg: Mapping[str, Any] | None = None, side: float = 0.06
) -> list[tuple[int, np.ndarray]]:
    """What `detect` would return for the calibration markers seen by the synthetic camera."""
    cfg = scene_cfg if scene_cfg is not None else load_scene_cfg()
    c, hc = camera_pose(cfg)
    dets: list[tuple[int, np.ndarray]] = []
    half = side / 2.0
    for marker_id, (x, y) in calib_marker_centers(cfg).items():
        corners = [
            (x + half, y + half, 0.0),
            (x + half, y - half, 0.0),
            (x - half, y - half, 0.0),
            (x - half, y + half, 0.0),
        ]
        dets.append((marker_id, project(corners, c, hc).astype(np.float32)))
    return dets


def synthetic_homography(scene_cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    """H for the synthetic camera, fitted from the 8 calibration marker centers exactly like `--live`."""
    cfg = scene_cfg if scene_cfg is not None else load_scene_cfg()
    c, hc = camera_pose(cfg)
    centers = calib_marker_centers(cfg)
    table = [(x, y) for x, y in centers.values()]
    pixels = project([(x, y, 0.0) for x, y in table], c, hc)
    return fit_homography(pixels, table)


def save_homography(h: np.ndarray, path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Write H to config/homography.npy (or `path`) and return the path."""
    file = pathlib.Path(path) if path is not None else HOMOGRAPHY_PATH
    arr = np.asarray(h, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"a homography is a finite 3x3 matrix, got shape {arr.shape}")
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, arr)
    return file


def load_homography(path: pathlib.Path | str | None = None) -> np.ndarray:
    """Read config/homography.npy (or `path`); FileNotFoundError until B5 (or `--fake`) writes it."""
    file = pathlib.Path(path) if path is not None else HOMOGRAPHY_PATH
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing: run python -m vision.calibrate --fake (or --live)")
    arr = np.load(file)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{file} does not hold a finite 3x3 homography")
    return np.asarray(arr, dtype=np.float64)


def _collect_live(camera: Any, frames: int) -> list[tuple[int, np.ndarray]]:
    """Median corners per calibration id over `frames` frames from a camera with `read()`."""
    from vision.markers import CALIB_IDS, detect

    seen: dict[int, list[np.ndarray]] = {}
    for _ in range(max(1, frames)):
        frame = camera.read()
        if frame is None:
            continue
        for marker_id, corners in detect(frame):
            if marker_id in CALIB_IDS:
                seen.setdefault(marker_id, []).append(corners)
    return [(mid, np.median(np.stack(cs), axis=0).astype(np.float32)) for mid, cs in sorted(seen.items())]


def _report(
    h: np.ndarray, dets: list[tuple[int, np.ndarray]], markers: Mapping[int, Sequence[float]]
) -> float:
    for marker_id, corners in dets:
        measured = markers.get(marker_id)
        if measured is None:
            continue
        u, v = np.asarray(corners, dtype=np.float64).mean(axis=0)
        x, y = pixel_to_table(h, u, v)
        err = math.hypot(x - measured[0], y - measured[1]) * 100.0
        print(f"marker {marker_id}: mapped ({x:.4f}, {y:.4f}) measured {tuple(measured)} error {err:.2f} cm")
    err_cm = calibration_error(h, dets, markers)
    print(f"mean calibration error {err_cm:.3f} cm over {len(dets)} markers")
    return err_cm


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vision.calibrate", description="Fit the overhead camera's table homography."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="write the synthetic homography (no hardware)")
    mode.add_argument("--live", action="store_true", help="fit from the overhead camera and save")
    mode.add_argument("--check", action="store_true", help="report the saved homography's live error")
    parser.add_argument("--frames", type=int, default=10, help="frames to average for --live/--check")
    parser.add_argument("--out", default=str(HOMOGRAPHY_PATH), help="where to write the .npy")
    parser.add_argument("--no-save", action="store_true", help="--live: fit and report, do not write")
    args = parser.parse_args(argv)

    cfg = load_scene_cfg()
    markers = calib_marker_centers(cfg)
    if args.fake:
        h = synthetic_homography(cfg)
        err_cm = _report(h, synthetic_detections(cfg), markers)
        out = save_homography(h, args.out)
        c, hc = camera_pose(cfg)
        print(f"wrote {out}: synthetic camera at ({c[0]}, {c[1]}, {hc}) m, f={FOCAL_PX:.0f} px, {IMAGE_SIZE}")
        return 0 if err_cm < 0.1 else 1

    from dotenv import load_dotenv

    load_dotenv()
    from vision.cameras import get_camera

    camera = get_camera("overhead")
    try:
        dets = _collect_live(camera, args.frames)
    finally:
        camera.release()
    if len(dets) < 4:
        print(f"only {len(dets)} calibration markers detected; need 4 or more", file=sys.stderr)
        return 1
    if args.check:
        h = load_homography(args.out)
        err_cm = _report(h, dets, markers)
        return 0 if err_cm <= 1.5 else 1
    h = fit_homography([np.mean(c, axis=0) for _, c in dets], [markers[m] for m, _ in dets])
    err_cm = _report(h, dets, markers)
    if not args.no_save:
        print(f"wrote {save_homography(h, args.out)}")
    return 0 if err_cm <= 1.5 else 1


if __name__ == "__main__":
    sys.exit(main())
