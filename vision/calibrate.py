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

CLI: `python -m vision.calibrate --fake` writes the synthetic config/homography.npy; `--live` fits from
the overhead camera and saves only after mean error <= 1 cm and maximum error <= 1.5 cm. `--check`
checks the saved matrix without writing it. Fit-marker residuals do not establish independent accuracy:
`--check-points checks.json` supplies at least five separate measured pixel/table pairs. With --live it
also gates the save; with --check it checks those recorded pairs offline without opening a camera.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile
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
    "calibration_metrics",
    "camera_pose",
    "check_points_metrics",
    "fit_homography",
    "height_correct",
    "load_homography",
    "load_check_points",
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
MAX_MEAN_ERROR_CM = 1.0
MAX_POINT_ERROR_CM = 1.5
MIN_CHECK_POINTS = 5


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
    if not math.isfinite(ransac_threshold) or ransac_threshold <= 0:
        raise ValueError("RANSAC threshold must be finite and positive")
    h, _mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if h is None or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        raise ValueError("findHomography failed: are the points spread over the table?")
    return np.asarray(h, dtype=np.float64) / h[2, 2]


def pixel_to_table(h: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Map one pixel through H: `H @ [u, v, 1]` normalized, as (x, y) in meters."""
    x, y, w = np.asarray(h, dtype=np.float64) @ np.array([float(u), float(v), 1.0])
    if not all(math.isfinite(value) for value in (x, y, w)) or abs(w) < 1e-12:
        raise ValueError(f"pixel ({u}, {v}) maps to a non-finite point or infinity")
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
    return float(calibration_metrics(h, dets, markers, min_markers=1)["mean_cm"])


def _calibration_pairs(
    dets: Iterable[tuple[int, np.ndarray]], markers: Mapping[int, Sequence[float]]
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    ids: list[int] = []
    pixels: list[list[float]] = []
    measured: list[list[float]] = []
    for marker_id, corners in dets:
        marker_id = int(marker_id)
        target = markers.get(marker_id)
        if target is None:
            continue
        if marker_id in ids:
            raise ValueError(f"duplicate calibration marker id {marker_id}")
        points = np.asarray(corners, dtype=np.float64)
        xy = np.asarray(target, dtype=np.float64)
        if points.shape != (4, 2) or xy.shape != (2,):
            raise ValueError(f"marker {marker_id} needs four pixel corners and a measured (x, y)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(xy)):
            raise ValueError(f"non-finite coordinates for calibration marker {marker_id}")
        ids.append(marker_id)
        pixels.append(points.mean(axis=0).tolist())
        measured.append(xy.tolist())
    if not ids:
        raise ValueError("no calibration marker (ids 0-7) detected")
    return ids, pixels, measured


def _error_metrics(
    h: np.ndarray, pixels: Iterable[Sequence[float]], table: Iterable[Sequence[float]]
) -> dict[str, int | float | bool]:
    h = _checked_homography(h)
    errors = [
        math.dist(pixel_to_table(h, *pixel), measured) * 100.0
        for pixel, measured in zip(pixels, table, strict=True)
    ]
    if not errors or not all(math.isfinite(error) for error in errors):
        raise ValueError("calibration errors must be finite and non-empty")
    mean_cm, max_cm = float(np.mean(errors)), max(errors)
    return {
        "count": len(errors),
        "mean_cm": mean_cm,
        "max_cm": max_cm,
        "accepted": mean_cm <= MAX_MEAN_ERROR_CM + 1e-9 and max_cm <= MAX_POINT_ERROR_CM + 1e-9,
    }


def calibration_metrics(
    h: np.ndarray,
    dets: Iterable[tuple[int, np.ndarray]],
    markers: Mapping[int, Sequence[float]],
    *,
    min_markers: int = 4,
) -> dict[str, int | float | bool]:
    """Marker fit residuals: count, mean_cm, max_cm, accepted (mean <= 1 cm, max <= 1.5 cm).

    At least four unique known markers are required by default. These points were used to fit H, so
    passing is a consistency check; use check_points_metrics for independent measured accuracy.
    calibration_error retains its older mean-only interface, accepting one or more markers.
    """
    if not isinstance(min_markers, int) or isinstance(min_markers, bool) or min_markers < 1:
        raise ValueError("min_markers must be a positive integer")
    ids, pixels, table = _calibration_pairs(dets, markers)
    if len(ids) < min_markers:
        raise ValueError(f"only {len(ids)} calibration markers detected; need {min_markers} or more")
    return _error_metrics(h, pixels, table)


def _check_point_pairs(points: Any) -> tuple[list[list[float]], list[list[float]]]:
    if not isinstance(points, list) or len(points) < MIN_CHECK_POINTS:
        raise ValueError(f"check points must be a JSON list of at least {MIN_CHECK_POINTS} measured spots")
    pixels: list[list[float]] = []
    table: list[list[float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {"pixel", "table"}:
            raise ValueError(f"check point {index} needs exactly pixel: [u, v] and table: [x, y]")
        for name, collection in (("pixel", pixels), ("table", table)):
            values = point[name]
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
            ):
                raise ValueError(f"check point {index} {name} must have two finite numbers")
            try:
                pair = [float(v) for v in values]
            except OverflowError as exc:
                raise ValueError(f"check point {index} {name} must have two finite numbers") from exc
            if not all(math.isfinite(v) for v in pair):
                raise ValueError(f"check point {index} {name} must have two finite numbers")
            collection.append(pair)
    for name, pairs in (("pixel", pixels), ("table", table)):
        if len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError(f"check points need distinct {name} positions")
        array = np.asarray(pairs)
        if np.linalg.matrix_rank(array - array[0]) < 2:
            raise ValueError(f"check points must span the table, not lie on one {name} line")
    return pixels, table


def check_points_metrics(h: np.ndarray, points: Any) -> dict[str, int | float | bool]:
    """Accuracy on >=5 independent spots [{pixel: [u, v], table: [x, y]}, ...].

    Pixels are image coordinates and table coordinates are measured meters at z=0. Every spot must be
    distinct and the set must span two dimensions. Supply new measurements, never reused fit markers.
    The return keys and acceptance limits are the same as calibration_metrics.
    """
    return _error_metrics(h, *_check_point_pairs(points))


def load_check_points(path: pathlib.Path | str) -> list[dict[str, list[float]]]:
    """Read and validate independent point measurements; never collect or manufacture measurements."""
    points = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    _check_point_pairs(points)
    return points


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
    """Atomically save a finite invertible H. A failed write preserves the previous calibration."""
    file = pathlib.Path(path) if path is not None else HOMOGRAPHY_PATH
    arr = _checked_homography(h)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=file.parent, prefix=f".{file.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            np.save(stream, arr, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file


def _checked_homography(h: Any) -> np.ndarray:
    arr = np.asarray(h, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"a homography is a finite 3x3 matrix, got shape {arr.shape}")
    if np.linalg.matrix_rank(arr) != 3:
        raise ValueError("a homography must be invertible")
    return arr


def load_homography(path: pathlib.Path | str | None = None) -> np.ndarray:
    """Read config/homography.npy (or `path`); FileNotFoundError until B5 (or `--fake`) writes it."""
    file = pathlib.Path(path) if path is not None else HOMOGRAPHY_PATH
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing: run python -m vision.calibrate --fake (or --live)")
    return _checked_homography(np.load(file, allow_pickle=False))


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
) -> dict[str, int | float | bool]:
    for marker_id, corners in dets:
        measured = markers.get(marker_id)
        if measured is None:
            continue
        u, v = np.asarray(corners, dtype=np.float64).mean(axis=0)
        x, y = pixel_to_table(h, u, v)
        err = math.hypot(x - measured[0], y - measured[1]) * 100.0
        print(f"marker {marker_id}: mapped ({x:.4f}, {y:.4f}) measured {tuple(measured)} error {err:.2f} cm")
    metrics = calibration_metrics(h, dets, markers)
    _report_metrics("fit-marker residuals", metrics)
    return metrics


def _report_metrics(label: str, metrics: Mapping[str, int | float | bool]) -> None:
    status = "PASS" if metrics["accepted"] else "FAIL"
    print(
        f"{label}: mean {metrics['mean_cm']:.3f} cm, max {metrics['max_cm']:.3f} cm "
        f"over {metrics['count']} points: {status} (limits: mean <= {MAX_MEAN_ERROR_CM:g} cm, "
        f"max <= {MAX_POINT_ERROR_CM:g} cm)"
    )


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
    parser.add_argument(
        "--check-points",
        type=pathlib.Path,
        help="JSON of >=5 independent {pixel:[u,v],table:[x,y]} spots; --check uses it offline",
    )
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.no_save and not args.live:
        parser.error("--no-save requires --live")
    if args.check_points and args.fake:
        parser.error("--check-points requires --live or --check")
    try:
        return _run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"calibration failed: {exc}; saved homography unchanged", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    check_points = load_check_points(args.check_points) if args.check_points else None
    if args.check and check_points is not None:
        metrics = check_points_metrics(load_homography(args.out), check_points)
        _report_metrics("independent measured check points (recorded pixels; no camera opened)", metrics)
        return 0 if metrics["accepted"] else 1
    cfg = load_scene_cfg()
    markers = calib_marker_centers(cfg)
    if args.fake:
        h = synthetic_homography(cfg)
        metrics = _report(h, synthetic_detections(cfg), markers)
        if not metrics["accepted"]:
            return 1
        out = save_homography(h, args.out)
        c, hc = camera_pose(cfg)
        print(f"wrote {out}: synthetic camera at ({c[0]}, {c[1]}, {hc}) m, f={FOCAL_PX:.0f} px, {IMAGE_SIZE}")
        print("Synthetic calibration only; physical table accuracy remains unverified.")
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    from vision.cameras import get_camera

    camera = get_camera("overhead")
    try:
        if getattr(camera, "is_fake", False):
            raise ValueError("live calibration requires a real camera; use --fake for synthetic calibration")
        dets = _collect_live(camera, args.frames)
    finally:
        camera.release()
    ids, pixels, table = _calibration_pairs(dets, markers)
    if len(ids) < 4:
        raise ValueError(f"only {len(ids)} calibration markers detected; need 4 or more")
    h = load_homography(args.out) if args.check else fit_homography(pixels, table)
    metrics = _report(h, dets, markers)
    accepted = bool(metrics["accepted"])
    if check_points is not None:
        independent = check_points_metrics(h, check_points)
        _report_metrics("independent measured check points", independent)
        accepted = accepted and bool(independent["accepted"])
    else:
        print("Fit-marker residuals only; independent table accuracy remains unverified.")
    if not accepted:
        print("Calibration rejected; saved homography unchanged.", file=sys.stderr)
        return 1
    if args.live and not args.no_save:
        print(f"wrote {save_homography(h, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
