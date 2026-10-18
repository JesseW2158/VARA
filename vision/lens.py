"""Lens calibration and undistortion (B5's fallback when the table edges still read over 1.5 cm off).

The homography of vision/calibrate.py is fitted over eight spread calibration markers, which averages out
most lens distortion. When the independent check points at the table edges still fail the 1.5 cm limit,
calibrate the lens: print a checkerboard (default 9x6 inner corners, 25 mm squares), photograph it with the
overhead camera at least MIN_IMAGES times in different positions, tilts and distances so the corners cover
the whole frame (one `scripts/capture_table.py --out DIR/board-NN` per photo keeps the tracker's camera
mode), then

    python -m vision.lens --images DIR [--pattern 9x6] [--square-m 0.025] [--out config/lens.npz]

`list_images(DIR)` takes the image files directly in DIR plus each subdirectory's `physical-original.png`
(the untouched frame capture_table.py saves; its `*-annotated` copies are never used). The fit finds the
corners (`cv2.findChessboardCorners` refined with `cornerSubPix`, its window sized from the squares in
the image so it never reaches the next corner), fits `cv2.calibrateCamera` and saves the image size, the
camera matrix, the distortion coefficients, the reprojection RMS and a hash of the images to
config/lens.npz atomically. Images without the full board and byte-identical repeats are skipped (and
listed); fewer than MIN_IMAGES usable images or an RMS above MAX_RMS_PX is refused and nothing is written.
`python -m vision.lens --check IMAGE [--lens config/lens.npz]` prints the saved lens's reprojection RMS on
one more board photo (take one the fit did not use). The square size scales only the board poses: the
camera matrix and distortion do not depend on it.

In code: `load_lens()` returns None while no lens file exists (a corrupt one raises ValueError),
`undistort(frame, lens)` applies `cv2.undistort` with the optimal new camera matrix (ALPHA = 1, so no
pixel of the field of view is cropped; the calibration markers sit near the frame edges) and
`undistort_maps(lens)` builds the `cv2.remap` maps once for a capture loop.

Wiring: the physical tools load config/lens.npz once, before the camera opens, through
`vision.calibrate.load_active_lens()` (this module's `load_lens` plus the file's `sha256` and the remap
tables; None when absent, ValueError when corrupt) and pass every frame through
`vision.calibrate.undistort_frame` before marker detection: the scene tracker (`SceneTracker(lens=...)`
from `vision.scene.get_tracker()` and the `python -m vision.scene` CLI; its `last_frame()` and MJPEG stream
show the undistorted frame), `python -m vision.calibrate --live` and `--check`, `scripts/capture_table.py`
(its `physical-original.png` stays raw for this module's fits; the pixels and check points are
undistorted) and the orchestrator's `POST /calibrate` marker check (the tracker's frame, or one camera
frame undistorted with the tracker's lens). `vision.cameras.Camera` itself stays raw. The evidence value
is `vision.calibrate.lens_hash(lens)`, the file's sha256 or null for raw frames: the `lens:` line of
calibrate, capture_table and `vision.verify`, `lens` in capture_table's markers.json and check-point
provenance and in `/calibrate`'s reply, and `vision.scene.tracker_diagnostics(tracker)["lens"]`. Raw and
undistorted pixels never mix: whenever config/lens.npz is created or replaced, refit the homography
(`python -m vision.calibrate --live`) and capture the five check spots again into a new list (capture_table
refuses to append to a list recorded with another lens), then restart the orchestrator. The FakeCamera is
an ideal pinhole and never undistorted; the --fake paths never read the lens file.

`synthesize_views(dir)` (CLI `--synthesize DIR`) renders checkerboard photos of a known synthetic camera
(the FOCAL_PX pinhole of vision.calibrate plus k1 = SYNTHETIC_K1) so the fit can be exercised offline.
Those files are named `synthetic-*.png`, a fit over them carries `synthetic: true`, and the CLI refuses
to write such a fit to the default config/lens.npz.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import math
import os
import pathlib
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from vision.calibrate import FOCAL_PX, IMAGE_SIZE, PRINCIPAL

__all__ = [
    "ALPHA",
    "CAPTURE_IMAGE",
    "DEFAULT_PATTERN",
    "DEFAULT_SQUARE_M",
    "IMAGE_SUFFIXES",
    "LENS_PATH",
    "MAX_RMS_PX",
    "MIN_IMAGES",
    "SYNTHETIC_K1",
    "SYNTHETIC_PREFIX",
    "board_points",
    "calibrate_lens",
    "check_image",
    "find_corners",
    "list_images",
    "load_lens",
    "main",
    "new_camera_matrix",
    "parse_pattern",
    "render_plane",
    "save_lens",
    "synthesize_views",
    "undistort",
    "undistort_maps",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
LENS_PATH = ROOT / "config" / "lens.npz"

DEFAULT_PATTERN = (9, 6)  # inner corners per row, per column
DEFAULT_SQUARE_M = 0.025
MIN_IMAGES = 15
MAX_RMS_PX = 1.0
ALPHA = 1.0  # getOptimalNewCameraMatrix: keep every source pixel (black corners rather than a crop)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
SYNTHETIC_PREFIX = "synthetic-"
SYNTHETIC_K1 = -0.10
SYNTHETIC_VIEWS = 18

CAPTURE_IMAGE = "physical-original.png"  # the untouched frame of one scripts/capture_table.py directory

_FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
# cornerSubPix half-window: this fraction of the smallest square side in the image, clamped to the range.
# A fixed 11 px half-window (23 px wide) reaches the next corner on 15 px squares (a 25 mm square seen from
# about 0.75 m) and pulled corners up to 2.7 px off; 0.4 of the side kept every synthetic corner within
# 0.4 px of its true projection.
_SUBPIX_FRACTION = 0.4
_SUBPIX_HALF_RANGE = (3, 11)
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_DIST_LENGTHS = (4, 5, 8, 12, 14)  # the coefficient counts cv2.undistort accepts
_TEXTURE_PX_PER_SQUARE = 32
_FRAME_MARGIN_PX = 6.0


# -- board geometry -------------------------------------------------------------------------------------


def parse_pattern(text: str) -> tuple[int, int]:
    """`"9x6"` -> (9, 6): the inner corners per row and per column (each at least 3, OpenCV's minimum)."""
    parts = str(text).lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"pattern must look like 9x6 (inner corners per row x per column), got {text!r}")
    try:
        cols, rows = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"pattern must look like 9x6, got {text!r}") from exc
    return _checked_pattern((cols, rows))


def _checked_pattern(pattern: Sequence[int]) -> tuple[int, int]:
    try:
        cols, rows = int(pattern[0]), int(pattern[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"pattern must be (columns, rows) of inner corners, got {pattern!r}") from exc
    if len(pattern) != 2 or cols < 3 or rows < 3:
        raise ValueError(f"pattern needs at least 3x3 inner corners (OpenCV's minimum), got {pattern!r}")
    return cols, rows


def _checked_square(square_m: float) -> float:
    try:
        square = float(square_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"square size must be a number of meters, got {square_m!r}") from exc
    if not math.isfinite(square) or square <= 0.0:
        raise ValueError(f"square size must be positive meters, got {square_m!r}")
    return square


def board_points(pattern: Sequence[int] = DEFAULT_PATTERN, square_m: float = DEFAULT_SQUARE_M) -> np.ndarray:
    """The board's inner corners in board coordinates, (cols*rows, 3) float32 meters at z = 0.

    The order (first row left to right, then the next row) is the order `find_corners` returns.
    """
    cols, rows = _checked_pattern(pattern)
    square = _checked_square(square_m)
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def _gray(image: Any) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 2:
        return frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"an image is a 2-D gray or 3-channel BGR array, got shape {frame.shape}")


def _subpix_half_window(corners: np.ndarray, pattern: tuple[int, int]) -> int:
    """`cornerSubPix` half-window for these rough corners: _SUBPIX_FRACTION of the smallest square side."""
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    along = float(np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
    across = float(np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    low, high = _SUBPIX_HALF_RANGE
    return int(min(high, max(low, math.floor(_SUBPIX_FRACTION * min(along, across)))))


def find_corners(image: np.ndarray, pattern: Sequence[int] = DEFAULT_PATTERN) -> np.ndarray | None:
    """Sub-pixel inner corners of a full checkerboard in the image, (N, 1, 2) float32, or None.

    `cv2.findChessboardCorners` (adaptive threshold, normalized) refined with `cornerSubPix` over a window
    of 0.4 of the smallest square side seen (3-11 px half-width); a partly visible or blurred board yields
    None rather than a partial list. The order is row by row, `pattern[0]` corners per row.
    """
    cols, rows = _checked_pattern(pattern)
    gray = _gray(image)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=_FIND_FLAGS)
    if not found or corners is None or len(corners) != cols * rows:
        return None
    half = _subpix_half_window(corners, (cols, rows))
    return cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), _SUBPIX_CRITERIA)


# -- calibration ----------------------------------------------------------------------------------------


def list_images(directory: pathlib.Path | str) -> list[pathlib.Path]:
    """The board photos of `directory`, sorted: its image files (IMAGE_SUFFIXES) plus the CAPTURE_IMAGE of
    each immediate subdirectory (one `scripts/capture_table.py --out DIR/board-NN` per photo).

    Files whose stem ends in `-annotated` (capture_table.py's drawn-over copies) are never taken; finding
    no image at all is an error.
    """
    folder = pathlib.Path(directory)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory of board images")
    entries = sorted(folder.iterdir())
    files = [
        p
        for p in entries
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("-annotated")
    ]
    files.extend(sub / CAPTURE_IMAGE for sub in entries if sub.is_dir() and (sub / CAPTURE_IMAGE).is_file())
    if not files:
        raise ValueError(
            f"{folder} holds no {', '.join(IMAGE_SUFFIXES)} images and no */{CAPTURE_IMAGE} captures"
        )
    return sorted(files)


def _display_names(paths: Sequence[pathlib.Path]) -> list[str]:
    """Each path relative to the deepest folder they all share (plain file names for one folder), so the
    recorded names stay unique when every capture directory holds a `physical-original.png`."""
    resolved = [p.resolve() for p in paths]
    try:
        base = pathlib.Path(os.path.commonpath([str(p.parent) for p in resolved]))
        return [p.relative_to(base).as_posix() for p in resolved]
    except ValueError:  # paths on different drives share no folder
        return [p.as_posix() for p in resolved]


def _read_gray(path: pathlib.Path) -> tuple[np.ndarray, bytes]:
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"{path} is not an image OpenCV can decode")
    return image, data


def calibrate_lens(
    image_paths: Iterable[pathlib.Path | str],
    pattern: Sequence[int] = DEFAULT_PATTERN,
    square_m: float = DEFAULT_SQUARE_M,
) -> dict[str, Any]:
    """Fit the lens from board photos: `cv2.findChessboardCorners` + `cv2.calibrateCamera`.

    Every image must share one size. Images without the full board and byte-identical repeats (which
    would count one pose twice) are skipped and listed under `skipped` as "name (reason)"; fewer than
    MIN_IMAGES usable images or an RMS above MAX_RMS_PX raises ValueError. Names are relative to the
    folder the images share (see `list_images`). The result holds `image_size` [w, h], `camera_matrix`
    (3x3), `dist_coeffs` (k1, k2, p1, p2, k3 with k3 fixed at 0), `rms`, `per_image_rms`, `images` (the
    used names), `images_sha256` (over their names and bytes in order), `skipped`, `pattern`,
    `square_m`, `created_at` and `synthetic` (any used image's file name starts with `synthetic-`).
    """
    paths = [pathlib.Path(p) for p in image_paths]
    if not paths:
        raise ValueError("no board images given")
    cols, rows = _checked_pattern(pattern)
    square = _checked_square(square_m)
    objp = board_points((cols, rows), square)
    digest = hashlib.sha256()
    used: list[str] = []
    skipped: list[str] = []
    first_with: dict[str, str] = {}  # content hash -> the first name with those bytes
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    for path, name in zip(paths, _display_names(paths), strict=True):
        image, data = _read_gray(path)
        this_size = (int(image.shape[1]), int(image.shape[0]))
        if size is None:
            size = this_size
        elif this_size != size:
            raise ValueError(f"{name} is {this_size} but earlier images are {size}: one camera mode only")
        content = hashlib.sha256(data).hexdigest()
        if content in first_with:
            skipped.append(f"{name} (same bytes as {first_with[content]})")
            continue
        first_with[content] = name
        corners = find_corners(image, (cols, rows))
        if corners is None:
            skipped.append(f"{name} (no full {cols}x{rows} board)")
            continue
        used.append(name)
        object_points.append(objp)
        image_points.append(corners)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    if len(used) < MIN_IMAGES or size is None:
        raise ValueError(
            f"only {len(used)} usable of {len(paths)} images (the full {cols}x{rows} board, no repeats); "
            f"need {MIN_IMAGES} (skipped: {', '.join(skipped) or 'none'})"
        )
    # k3 stays 0: a webcam's distortion is k1, k2 and the tangential terms, and a free k3 trades off
    # against k1 and k2 whenever the board photos do not reach the frame corners (unstable at the edges).
    rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None, flags=cv2.CALIB_FIX_K3
    )
    dist_coeffs = np.asarray(dist, dtype=np.float64).ravel()
    per_image: list[float] = []
    for objp_i, imgp_i, rvec, tvec in zip(object_points, image_points, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(objp_i, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(projected.reshape(-1, 2) - imgp_i.reshape(-1, 2), axis=1)
        per_image.append(float(math.sqrt(float(np.mean(errors**2)))))
    rms = float(rms)
    if not math.isfinite(rms) or rms > MAX_RMS_PX:
        raise ValueError(
            f"reprojection rms {rms:.3f} px exceeds {MAX_RMS_PX:g} px: retake sharper, flatter board photos"
        )
    return {
        "image_size": [int(size[0]), int(size[1])],
        "camera_matrix": np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        "dist_coeffs": dist_coeffs,
        "rms": rms,
        "per_image_rms": per_image,
        "images": used,
        "images_sha256": digest.hexdigest(),
        "skipped": skipped,
        "pattern": [cols, rows],
        "square_m": square,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": any(pathlib.PurePosixPath(name).name.startswith(SYNTHETIC_PREFIX) for name in used),
    }


def _checked_lens(lens: Mapping[str, Any]) -> dict[str, Any]:
    """The core lens fields validated and normalized (K 3x3 float64, dist 1-D, image_size ints, rms)."""
    if not isinstance(lens, Mapping):
        raise ValueError("a lens is a mapping with image_size, camera_matrix, dist_coeffs and rms")
    missing = [key for key in ("image_size", "camera_matrix", "dist_coeffs", "rms") if key not in lens]
    if missing:
        raise ValueError(f"lens is missing {', '.join(missing)}")
    camera_matrix = np.asarray(lens["camera_matrix"], dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError(f"camera_matrix must be a finite 3x3 matrix, got shape {camera_matrix.shape}")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or not np.allclose(camera_matrix[2], [0, 0, 1]):
        raise ValueError("camera_matrix must have positive focal lengths and a [0, 0, 1] last row")
    dist = np.asarray(lens["dist_coeffs"], dtype=np.float64).ravel()
    if len(dist) not in _DIST_LENGTHS or not np.all(np.isfinite(dist)):
        raise ValueError(f"dist_coeffs must be {_DIST_LENGTHS} finite numbers, got {len(dist)}")
    try:
        width, height = (int(v) for v in np.asarray(lens["image_size"]).ravel().tolist())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image_size must be [width, height], got {lens['image_size']!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {[width, height]}")
    rms = float(np.asarray(lens["rms"], dtype=np.float64).ravel()[0])
    if not math.isfinite(rms) or rms < 0.0:
        raise ValueError(f"rms must be a finite non-negative pixel error, got {rms!r}")
    return {"image_size": [width, height], "camera_matrix": camera_matrix, "dist_coeffs": dist, "rms": rms}


def save_lens(result: Mapping[str, Any], path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Atomically save a validated lens to `path` (default config/lens.npz); a failed write leaves the
    previous file untouched. Stored keys: image_size, camera_matrix, dist_coeffs, rms, per_image_rms,
    images, images_sha256, skipped, pattern, square_m, created_at, synthetic."""
    core = _checked_lens(result)
    file = pathlib.Path(path) if path is not None else LENS_PATH
    payload: dict[str, Any] = {
        "image_size": np.asarray(core["image_size"], dtype=np.int64),
        "camera_matrix": core["camera_matrix"],
        "dist_coeffs": core["dist_coeffs"],
        "rms": np.float64(core["rms"]),
        "per_image_rms": np.asarray(list(result.get("per_image_rms") or []), dtype=np.float64),
        "images": np.asarray([str(s) for s in result.get("images") or []], dtype=str),
        "images_sha256": np.str_(str(result.get("images_sha256") or "")),
        "skipped": np.asarray([str(s) for s in result.get("skipped") or []], dtype=str),
        "pattern": np.asarray(list(result.get("pattern") or [0, 0]), dtype=np.int64),
        "square_m": np.float64(result.get("square_m") or 0.0),
        "created_at": np.str_(str(result.get("created_at") or datetime.now(timezone.utc).isoformat())),
        "synthetic": np.bool_(bool(result.get("synthetic", False))),
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=file.parent, prefix=f".{file.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file


def load_lens(path: pathlib.Path | str | None = None) -> dict[str, Any] | None:
    """The saved lens as a dict (see `save_lens`), or None when no file exists. A corrupt or incomplete
    file raises ValueError so a camera never silently runs with a half-read lens."""
    file = pathlib.Path(path) if path is not None else LENS_PATH
    if not file.exists():
        return None
    try:
        with np.load(file, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"{file} is not a lens file: {exc}") from exc
    lens = _checked_lens(raw)
    lens["per_image_rms"] = [
        float(v) for v in np.asarray(raw.get("per_image_rms", []), dtype=np.float64).ravel()
    ]
    lens["images"] = [str(v) for v in np.asarray(raw.get("images", [])).ravel().tolist()]
    lens["images_sha256"] = str(raw["images_sha256"]) if "images_sha256" in raw else ""
    lens["skipped"] = [str(v) for v in np.asarray(raw.get("skipped", [])).ravel().tolist()]
    lens["pattern"] = [int(v) for v in np.asarray(raw.get("pattern", [0, 0])).ravel().tolist()]
    lens["square_m"] = float(raw["square_m"]) if "square_m" in raw else 0.0
    lens["created_at"] = str(raw["created_at"]) if "created_at" in raw else ""
    lens["synthetic"] = bool(raw["synthetic"]) if "synthetic" in raw else False
    lens["path"] = str(file)
    return lens


# -- undistortion ---------------------------------------------------------------------------------------


def new_camera_matrix(lens: Mapping[str, Any]) -> np.ndarray:
    """The optimal new camera matrix for `undistort` (ALPHA = 1: the whole field of view is kept)."""
    core = _checked_lens(lens)
    size = (core["image_size"][0], core["image_size"][1])
    matrix, _roi = cv2.getOptimalNewCameraMatrix(
        core["camera_matrix"], core["dist_coeffs"], size, ALPHA, size
    )
    return np.asarray(matrix, dtype=np.float64)


def _frame_size_check(frame: np.ndarray, core: Mapping[str, Any]) -> None:
    if frame is None or frame.ndim not in (2, 3) or frame.size == 0:
        raise ValueError("undistort needs a 2-D gray or 3-channel BGR frame")
    width, height = core["image_size"]
    if frame.shape[1] != width or frame.shape[0] != height:
        raise ValueError(
            f"frame is {frame.shape[1]}x{frame.shape[0]} but the lens was calibrated at {width}x{height}: "
            "recalibrate the lens in the camera mode the tracker uses"
        )


def undistort(frame: np.ndarray, lens: Mapping[str, Any]) -> np.ndarray:
    """`cv2.undistort` the frame with the lens and its optimal new camera matrix (same size as the input).

    The frame must have the lens's image size. Detections on the result are pinhole pixels of
    `new_camera_matrix(lens)`, which is the camera the homography must then be fitted on.
    """
    core = _checked_lens(lens)
    _frame_size_check(frame, core)
    return cv2.undistort(frame, core["camera_matrix"], core["dist_coeffs"], None, new_camera_matrix(core))


def undistort_maps(lens: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.initUndistortRectifyMap` maps (CV_16SC2) equivalent to `undistort`, for a capture loop:
    `cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)`."""
    core = _checked_lens(lens)
    size = (core["image_size"][0], core["image_size"][1])
    map1, map2 = cv2.initUndistortRectifyMap(
        core["camera_matrix"], core["dist_coeffs"], None, new_camera_matrix(core), size, cv2.CV_16SC2
    )
    return map1, map2


def check_image(
    image: np.ndarray,
    lens: Mapping[str, Any],
    pattern: Sequence[int] = DEFAULT_PATTERN,
    square_m: float = DEFAULT_SQUARE_M,
) -> dict[str, float | int | bool]:
    """Reprojection error of the lens on one more board photo: {rms, max_px, corners, accepted}.

    The board pose comes from `cv2.solvePnP` with the lens, so the residual measures the lens model, not
    the pose. Raises ValueError when the full board is not found.
    """
    core = _checked_lens(lens)
    gray = _gray(image)
    _frame_size_check(gray, core)
    corners = find_corners(gray, pattern)
    if corners is None:
        raise ValueError("the full checkerboard was not found in the image")
    objp = board_points(pattern, square_m)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, core["camera_matrix"], core["dist_coeffs"])
    if not ok:
        raise ValueError("solvePnP failed on the board corners")
    projected, _ = cv2.projectPoints(objp, rvec, tvec, core["camera_matrix"], core["dist_coeffs"])
    errors = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    rms = float(math.sqrt(float(np.mean(errors**2))))
    return {
        "rms": rms,
        "max_px": float(errors.max()),
        "corners": int(len(errors)),
        "accepted": bool(math.isfinite(rms) and rms <= MAX_RMS_PX),
    }


# -- synthetic views ------------------------------------------------------------------------------------


def _board_texture(pattern: Sequence[int], px_per_square: int = _TEXTURE_PX_PER_SQUARE) -> np.ndarray:
    """A gray texture of the (cols+1) x (rows+1) squares plus a one-square white margin all around.

    In board coordinates the texture spans X in [-2s, (cols+1)s] and Y in [-2s, (rows+1)s] for square
    side s; the inner corners are at (i s, j s), matching `board_points`.
    """
    cols, rows = _checked_pattern(pattern)
    texture = np.full(((rows + 3) * px_per_square, (cols + 3) * px_per_square), 255, np.uint8)
    for b in range(-1, rows):
        for a in range(-1, cols):
            if (a + b) % 2 == 0:
                top, left = (b + 2) * px_per_square, (a + 2) * px_per_square
                texture[top : top + px_per_square, left : left + px_per_square] = 0
    return texture


@functools.lru_cache(maxsize=4)
def _normalized_grid(k_bytes: bytes, d_bytes: bytes, n_dist: int, width: int, height: int) -> np.ndarray:
    """Undistorted normalized homogeneous coordinates (N, 3) of every pixel, row-major (pose-independent)."""
    camera_matrix = np.frombuffer(k_bytes, dtype=np.float64).reshape(3, 3)
    dist = np.frombuffer(d_bytes, dtype=np.float64).reshape(n_dist)
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    grid = np.stack([us, vs], axis=-1).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 1e-10)
    normalized = cv2.undistortPointsIter(grid, camera_matrix, dist, None, None, criteria).reshape(-1, 2)
    return np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)


def render_plane(
    texture: np.ndarray,
    origin: Sequence[float],
    extent: Sequence[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: Sequence[float],
    tvec: Sequence[float],
    image_size: Sequence[int] = IMAGE_SIZE,
    *,
    background: int = 128,
) -> np.ndarray:
    """Photograph a flat texture through a distorting camera: the exact inverse mapping, one resampling.

    The texture covers the plane rectangle X in [origin[0], origin[0] + extent[0]], Y likewise, at Z = 0
    of a board frame posed by (rvec, tvec) as `cv2.projectPoints` understands them. Each image pixel is
    undistorted to a normalized ray, intersected with the plane and sampled from the texture
    (`cv2.remap`, bilinear); rays missing the rectangle or the plane get `background`.
    """
    k = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    d = np.ascontiguousarray(np.asarray(dist_coeffs, dtype=np.float64).ravel())
    width, height = int(image_size[0]), int(image_size[1])
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    plane = np.column_stack([rotation[:, 0], rotation[:, 1], np.asarray(tvec, dtype=np.float64).reshape(3)])
    inverse = np.linalg.inv(plane)
    board = _normalized_grid(k.tobytes(), d.tobytes(), len(d), width, height) @ inverse.T
    depth = board[:, 2]
    in_front = depth > 1e-9
    safe = np.where(in_front, depth, 1.0)
    x = np.where(in_front, board[:, 0] / safe, -1e9)
    y = np.where(in_front, board[:, 1] / safe, -1e9)
    tex_h, tex_w = texture.shape[:2]
    map_x = ((x - origin[0]) / extent[0] * tex_w - 0.5).astype(np.float32).reshape(height, width)
    map_y = ((y - origin[1]) / extent[1] * tex_h - 0.5).astype(np.float32).reshape(height, width)
    return cv2.remap(
        texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background
    )


def _synthetic_camera(k1: float) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[FOCAL_PX, 0.0, PRINCIPAL[0]], [0.0, FOCAL_PX, PRINCIPAL[1]], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return camera_matrix, np.array([float(k1), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _board_fully_visible(
    rvec: np.ndarray,
    tvec: np.ndarray,
    extent_corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist: np.ndarray,
) -> bool:
    projected, _ = cv2.projectPoints(extent_corners, rvec, tvec, camera_matrix, dist)
    p = projected.reshape(-1, 2)
    width, height = IMAGE_SIZE
    return bool(
        np.all(p[:, 0] > _FRAME_MARGIN_PX)
        and np.all(p[:, 0] < width - _FRAME_MARGIN_PX)
        and np.all(p[:, 1] > _FRAME_MARGIN_PX)
        and np.all(p[:, 1] < height - _FRAME_MARGIN_PX)
    )


def synthesize_views(
    directory: pathlib.Path | str,
    *,
    views: int = SYNTHETIC_VIEWS,
    k1: float = SYNTHETIC_K1,
    pattern: Sequence[int] = DEFAULT_PATTERN,
    square_m: float = DEFAULT_SQUARE_M,
    seed: int = 0,
) -> dict[str, Any]:
    """Write `views` checkerboard photos (`synthetic-board-NN.png`, 640x480 gray) of the synthetic camera
    (FOCAL_PX, PRINCIPAL, distortion [k1, 0, 0, 0, 0]) at random fully visible poses into `directory`.

    Returns the truth to compare a fit against: `camera_matrix`, `dist_coeffs`, `image_size`, `images`
    (the written paths) and `poses` ([rvec, tvec] per image). A folder that already holds one of those
    names is refused before anything is written (FileExistsError).
    """
    folder = pathlib.Path(directory)
    if not isinstance(views, int) or isinstance(views, bool) or views < 1:
        raise ValueError(f"views must be a positive integer, got {views!r}")
    cols, rows = _checked_pattern(pattern)
    square = _checked_square(square_m)
    if not math.isfinite(k1):
        raise ValueError("k1 must be finite")
    targets = [folder / f"{SYNTHETIC_PREFIX}board-{index:02d}.png" for index in range(1, views + 1)]
    taken = [target.name for target in targets if target.exists()]
    if taken:
        raise FileExistsError(f"{folder} already holds {', '.join(taken)}; choose a new directory")
    camera_matrix, dist = _synthetic_camera(k1)
    texture = _board_texture((cols, rows))
    origin = (-2.0 * square, -2.0 * square)
    extent = ((cols + 3) * square, (rows + 3) * square)
    extent_corners = np.array(
        [
            [origin[0], origin[1], 0.0],
            [origin[0] + extent[0], origin[1], 0.0],
            [origin[0] + extent[0], origin[1] + extent[1], 0.0],
            [origin[0], origin[1] + extent[1], 0.0],
        ],
        dtype=np.float64,
    )
    center = np.array([(cols - 1) * square / 2.0, (rows - 1) * square / 2.0, 0.0])
    width, height = IMAGE_SIZE
    rng = np.random.default_rng(seed)
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(poses) < views:
        attempts += 1
        if attempts > 200 * views:
            raise RuntimeError("could not place the board fully inside the frame; shrink the board or square")
        rvec = rng.uniform([-0.45, -0.45, -0.4], [0.45, 0.45, 0.4])
        rotation, _ = cv2.Rodrigues(rvec)
        depth = rng.uniform(0.42, 0.62)
        target_u, target_v = rng.uniform(110.0, width - 110.0), rng.uniform(95.0, height - 95.0)
        center_cam = np.array(
            [
                (target_u - PRINCIPAL[0]) / FOCAL_PX * depth,
                (target_v - PRINCIPAL[1]) / FOCAL_PX * depth,
                depth,
            ]
        )
        tvec = center_cam - rotation @ center
        if _board_fully_visible(rvec, tvec, extent_corners, camera_matrix, dist):
            poses.append((rvec, tvec))
    encoded: list[bytes] = []
    for rvec, tvec in poses:  # render and encode everything first: a failure leaves no partial set
        image = render_plane(texture, origin, extent, camera_matrix, dist, rvec, tvec, IMAGE_SIZE)
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise OSError("could not encode a synthetic board view")
        encoded.append(data.tobytes())
    folder.mkdir(parents=True, exist_ok=True)
    for target, data in zip(targets, encoded, strict=True):
        with target.open("xb") as stream:
            stream.write(data)
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist,
        "image_size": [width, height],
        "pattern": [cols, rows],
        "square_m": square,
        "images": targets,
        "poses": [(rvec.copy(), tvec.copy()) for rvec, tvec in poses],
    }


# -- CLI ------------------------------------------------------------------------------------------------


def _describe(lens: Mapping[str, Any]) -> str:
    k = np.asarray(lens["camera_matrix"], dtype=np.float64)
    d = np.asarray(lens["dist_coeffs"], dtype=np.float64).ravel()
    return (
        f"fx {k[0, 0]:.2f} fy {k[1, 1]:.2f} cx {k[0, 2]:.2f} cy {k[1, 2]:.2f} px; "
        f"dist {np.array2string(d, precision=4, separator=', ', suppress_small=True)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: `--images DIR` fits and saves, `--check IMAGE` reports, `--synthesize DIR` renders."""
    parser = argparse.ArgumentParser(
        prog="python -m vision.lens",
        description="Calibrate the overhead camera's lens from checkerboard photos (B5 fallback).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--images",
        type=pathlib.Path,
        metavar="DIR",
        help=f"board photos to fit: DIR's images plus each DIR/*/{CAPTURE_IMAGE} (capture_table.py)",
    )
    mode.add_argument("--check", type=pathlib.Path, metavar="IMAGE", help="board photo to check --lens on")
    mode.add_argument(
        "--synthesize", type=pathlib.Path, metavar="DIR", help="write synthetic board photos for offline use"
    )
    parser.add_argument("--pattern", default="9x6", help="inner corners per row x per column (default 9x6)")
    parser.add_argument(
        "--square-m",
        type=float,
        default=DEFAULT_SQUARE_M,
        metavar="M",
        help=f"printed square side in meters (default {DEFAULT_SQUARE_M:g})",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="--images: where to write the lens (default config/lens.npz)",
    )
    parser.add_argument(
        "--lens",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="--check: the lens file to check (default config/lens.npz)",
    )
    args = parser.parse_args(argv)
    try:
        pattern = parse_pattern(args.pattern)
        square = _checked_square(args.square_m)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out is not None and args.images is None:
        parser.error("--out applies only to --images")
    if args.lens is not None and args.check is None:
        parser.error("--lens applies only to --check")
    out = args.out if args.out is not None else LENS_PATH
    lens_file = args.lens if args.lens is not None else LENS_PATH
    if args.check is not None:
        failure = "lens check failed: {}"
    elif args.synthesize is not None:
        failure = "synthesis failed: {}; nothing written"
    else:
        failure = "lens calibration failed: {}; nothing written"
    try:
        if args.synthesize is not None:
            truth = synthesize_views(args.synthesize, pattern=pattern, square_m=square)
            print(
                f"wrote {len(truth['images'])} synthetic board views to {args.synthesize} "
                f"(true {_describe(truth)}); fit them with --images and an --out outside config/"
            )
            return 0
        if args.check is not None:
            lens = load_lens(lens_file)
            if lens is None:
                print(f"no lens file at {lens_file}; fit one with --images first", file=sys.stderr)
                return 1
            image, _data = _read_gray(args.check)
            metrics = check_image(image, lens, pattern, square)
            status = "PASS" if metrics["accepted"] else "FAIL"
            print(
                f"{args.check}: reprojection rms {metrics['rms']:.3f} px (max {metrics['max_px']:.3f} px) "
                f"over {metrics['corners']} corners with {lens_file}: {status} (limit {MAX_RMS_PX:g} px)"
            )
            return 0 if metrics["accepted"] else 1
        paths = list_images(args.images)
        result = calibrate_lens(paths, pattern, square)
        if result["synthetic"] and pathlib.Path(out).resolve() == LENS_PATH.resolve():
            print(
                f"{len(result['images'])} synthetic views fitted (rms {result['rms']:.3f} px) but synthetic "
                f"images cannot calibrate the real lens: pass --out somewhere other than {LENS_PATH}",
                file=sys.stderr,
            )
            return 1
        replaced = pathlib.Path(out).exists()
        file = save_lens(result, out)
        print(
            f"used {len(result['images'])} of {len(paths)} images (skipped: "
            f"{', '.join(result['skipped']) or 'none'}); rms {result['rms']:.3f} px; {_describe(result)}"
        )
        print(f"wrote {file}{' (replaced the previous lens)' if replaced else ''}")
        if result["synthetic"]:
            print("Synthetic views only; the real lens remains uncalibrated.")
        else:
            print("Refit the homography (python -m vision.calibrate --live) and re-capture the check spots.")
        return 0
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(failure.format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
